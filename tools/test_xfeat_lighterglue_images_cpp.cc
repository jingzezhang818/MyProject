#include "XFextractor.h"
#include "XFeatLighterGlue/matcher.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <torch/torch.h>

namespace
{

std::string ShapeString(const torch::Tensor& tensor)
{
    std::ostringstream oss;
    oss << "[";
    for(size_t i = 0; i < tensor.sizes().size(); ++i)
    {
        if(i != 0)
            oss << ", ";
        oss << tensor.sizes()[i];
    }
    oss << "]";
    return oss.str();
}

std::string ToLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

int CudaDeviceIndex()
{
    if(const char* env = std::getenv("XFEAT_CUDA_DEVICE"))
        return std::max(0, std::atoi(env));
    return 0;
}

torch::Device SelectDeviceAndConfigureExtractor()
{
    std::string requested = "auto";
    if(const char* env = std::getenv("XFEAT_DEVICE"))
        requested = ToLower(env);

    const int cudaIndex = CudaDeviceIndex();
    if(requested == "cpu")
    {
        setenv("XFEAT_DEVICE", "cpu", 1);
        return torch::Device(torch::kCPU);
    }

    if(requested == "cuda" || requested == "gpu")
    {
        if(!torch::cuda::is_available())
            throw std::runtime_error("XFEAT_DEVICE=cuda was requested, but torch::cuda::is_available() is false.");
        setenv("XFEAT_DEVICE", "cuda", 1);
        return torch::Device(torch::kCUDA, cudaIndex);
    }

    if(requested != "auto")
        throw std::runtime_error("XFEAT_DEVICE must be one of auto, cuda, gpu, cpu.");

    if(torch::cuda::is_available())
    {
        setenv("XFEAT_DEVICE", "cuda", 1);
        return torch::Device(torch::kCUDA, cudaIndex);
    }

    setenv("XFEAT_DEVICE", "cpu", 1);
    return torch::Device(torch::kCPU);
}

cv::Mat ReadImageForXFeat(const std::string& path)
{
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
    if(image.empty())
        throw std::runtime_error("failed to read image: " + path);

    if(image.depth() != CV_8U)
        throw std::runtime_error("only 8-bit grayscale/color images are supported: " + path);

    if(image.channels() == 4)
    {
        cv::Mat bgr;
        cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
        return bgr;
    }

    if(image.channels() != 1 && image.channels() != 3)
        throw std::runtime_error("image must be grayscale, BGR, or BGRA: " + path);

    return image;
}

torch::Tensor KeyPointsToTensor(const std::vector<cv::KeyPoint>& keypoints,
                                const torch::Device& device)
{
    if(keypoints.empty())
        return torch::empty({0, 2}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    std::vector<float> data;
    data.reserve(keypoints.size() * 2);
    for(const auto& kp : keypoints)
    {
        data.push_back(kp.pt.x);
        data.push_back(kp.pt.y);
    }

    return torch::from_blob(data.data(),
                            {static_cast<int64_t>(keypoints.size()), 2},
                            torch::TensorOptions().dtype(torch::kFloat32))
        .clone()
        .to(device);
}

torch::Tensor DescriptorsToTensor(const cv::Mat& descriptors,
                                  const torch::Device& device,
                                  const std::string& name)
{
    if(descriptors.empty())
        return torch::empty({0, 64}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    if(descriptors.type() != CV_32F)
        throw std::runtime_error(name + " descriptors must be CV_32F.");
    if(descriptors.cols != 64)
        throw std::runtime_error(name + " descriptors must have 64 columns.");

    cv::Mat continuous = descriptors.isContinuous() ? descriptors : descriptors.clone();
    auto tensor = torch::from_blob(continuous.ptr<float>(),
                                   {continuous.rows, continuous.cols},
                                   torch::TensorOptions().dtype(torch::kFloat32))
                      .clone()
                      .to(device);

    return tensor / tensor.pow(2).sum(1, true).sqrt().clamp_min(1e-12);
}

torch::Tensor ImageSizeToTensor(int width, int height, const torch::Device& device)
{
    return torch::tensor({static_cast<float>(width), static_cast<float>(height)},
                         torch::TensorOptions().dtype(torch::kFloat32).device(device));
}

struct NormStats
{
    float mean = 0.0f;
    float min = 0.0f;
    float max = 0.0f;
};

NormStats DescriptorNormStats(const torch::Tensor& descriptors)
{
    if(descriptors.numel() == 0)
        return {};

    const auto norms = descriptors.pow(2).sum(1).sqrt().to(torch::kCPU);
    return NormStats{
        norms.mean().item<float>(),
        norms.min().item<float>(),
        norms.max().item<float>()};
}

struct KeypointRange
{
    float minX = 0.0f;
    float maxX = 0.0f;
    float minY = 0.0f;
    float maxY = 0.0f;
};

struct ValidMatchRecord
{
    ORB_SLAM3::LGMatch match;
    cv::Point2f p0;
    cv::Point2f p1;
    float displacement = 0.0f;
};

struct BasicMatchStats
{
    float meanScore = 0.0f;
    float medianScore = 0.0f;
    float meanDisplacement = 0.0f;
    float medianDisplacement = 0.0f;
};

struct RansacStats
{
    bool attempted = false;
    bool success = false;
    int inliers = 0;
    float inlierRatio = 0.0f;
    float meanInlierScore = 0.0f;
    float meanOutlierScore = 0.0f;
    std::vector<uchar> mask;
};

struct EssentialOptions
{
    bool enabled = false;
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
};

KeypointRange ComputeKeypointRange(const std::vector<cv::KeyPoint>& keypoints)
{
    if(keypoints.empty())
        return {};

    KeypointRange range;
    range.minX = range.maxX = keypoints.front().pt.x;
    range.minY = range.maxY = keypoints.front().pt.y;
    for(const auto& kp : keypoints)
    {
        range.minX = std::min(range.minX, kp.pt.x);
        range.maxX = std::max(range.maxX, kp.pt.x);
        range.minY = std::min(range.minY, kp.pt.y);
        range.maxY = std::max(range.maxY, kp.pt.y);
    }
    return range;
}

void PrintImageDiagnostics(const std::string& label,
                           const cv::Mat& image,
                           const std::vector<cv::KeyPoint>& keypoints,
                           const torch::Tensor& descriptors)
{
    const auto stats = DescriptorNormStats(descriptors);
    const auto range = ComputeKeypointRange(keypoints);

    std::cout << label << " size: " << image.cols << "x" << image.rows
              << " channels=" << image.channels() << std::endl;
    std::cout << label << " N: " << keypoints.size() << std::endl;
    std::cout << label << " desc shape: " << ShapeString(descriptors) << std::endl;
    std::cout << label << " desc norm mean/min/max: "
              << stats.mean << " / " << stats.min << " / " << stats.max << std::endl;
    std::cout << label << " keypoint x range: " << range.minX << " .. " << range.maxX
              << ", y range: " << range.minY << " .. " << range.maxY << std::endl;
}

float Median(std::vector<float> values)
{
    if(values.empty())
        return 0.0f;

    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    const float hi = values[mid];
    if(values.size() % 2 != 0)
        return hi;

    std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
    return 0.5f * (values[mid - 1] + hi);
}

BasicMatchStats ComputeBasicStats(const std::vector<ValidMatchRecord>& records)
{
    if(records.empty())
        return {};

    std::vector<float> scores;
    std::vector<float> displacements;
    scores.reserve(records.size());
    displacements.reserve(records.size());

    double sumScore = 0.0;
    double sumDisplacement = 0.0;
    for(const auto& r : records)
    {
        scores.push_back(r.match.score);
        displacements.push_back(r.displacement);
        sumScore += r.match.score;
        sumDisplacement += r.displacement;
    }

    return BasicMatchStats{
        static_cast<float>(sumScore / static_cast<double>(records.size())),
        Median(scores),
        static_cast<float>(sumDisplacement / static_cast<double>(records.size())),
        Median(displacements)};
}

void ComputeMaskScoreStats(const std::vector<ValidMatchRecord>& records,
                           RansacStats& stats)
{
    if(records.empty() || stats.mask.size() != records.size())
        return;

    double inlierScore = 0.0;
    double outlierScore = 0.0;
    int inlierCount = 0;
    int outlierCount = 0;

    for(size_t i = 0; i < records.size(); ++i)
    {
        if(stats.mask[i])
        {
            inlierScore += records[i].match.score;
            ++inlierCount;
        }
        else
        {
            outlierScore += records[i].match.score;
            ++outlierCount;
        }
    }

    stats.inliers = inlierCount;
    stats.inlierRatio = static_cast<float>(inlierCount) / static_cast<float>(records.size());
    stats.meanInlierScore = inlierCount > 0 ? static_cast<float>(inlierScore / inlierCount) : 0.0f;
    stats.meanOutlierScore = outlierCount > 0 ? static_cast<float>(outlierScore / outlierCount) : 0.0f;
}

RansacStats RunFundamentalRansac(const std::vector<cv::Point2f>& pts0,
                                 const std::vector<cv::Point2f>& pts1,
                                 const std::vector<ValidMatchRecord>& records)
{
    RansacStats stats;
    if(records.size() < 8)
        return stats;

    stats.attempted = true;
    cv::Mat mask;
    cv::Mat F;
    try
    {
        F = cv::findFundamentalMat(pts0, pts1, cv::FM_RANSAC, 3.0, 0.999, mask);
    }
    catch(const cv::Exception& e)
    {
        std::cerr << "F RANSAC failed: " << e.what() << std::endl;
        return stats;
    }
    if(F.empty() || mask.empty())
        return stats;

    stats.success = true;
    stats.mask.assign(mask.begin<uchar>(), mask.end<uchar>());
    ComputeMaskScoreStats(records, stats);
    return stats;
}

RansacStats RunHomographyRansac(const std::vector<cv::Point2f>& pts0,
                                const std::vector<cv::Point2f>& pts1,
                                const std::vector<ValidMatchRecord>& records)
{
    RansacStats stats;
    if(records.size() < 4)
        return stats;

    stats.attempted = true;
    cv::Mat mask;
    cv::Mat H;
    try
    {
        H = cv::findHomography(pts0, pts1, cv::RANSAC, 3.0, mask);
    }
    catch(const cv::Exception& e)
    {
        std::cerr << "H RANSAC failed: " << e.what() << std::endl;
        return stats;
    }
    if(H.empty() || mask.empty())
        return stats;

    stats.success = true;
    stats.mask.assign(mask.begin<uchar>(), mask.end<uchar>());
    ComputeMaskScoreStats(records, stats);
    return stats;
}

RansacStats RunEssentialRansac(const std::vector<cv::Point2f>& pts0,
                               const std::vector<cv::Point2f>& pts1,
                               const std::vector<ValidMatchRecord>& records,
                               const EssentialOptions& options)
{
    RansacStats stats;
    if(!options.enabled || records.size() < 8)
        return stats;

    stats.attempted = true;
    cv::Mat K = (cv::Mat_<double>(3, 3) << options.fx, 0.0, options.cx,
                                          0.0, options.fy, options.cy,
                                          0.0, 0.0, 1.0);
    cv::Mat mask;
    cv::Mat E;
    try
    {
        E = cv::findEssentialMat(pts0, pts1, K, cv::RANSAC, 0.999, 1.0, mask);
    }
    catch(const cv::Exception& e)
    {
        std::cerr << "E RANSAC failed: " << e.what() << std::endl;
        return stats;
    }
    if(E.empty() || mask.empty())
        return stats;

    stats.success = true;
    stats.mask.assign(mask.begin<uchar>(), mask.end<uchar>());
    ComputeMaskScoreStats(records, stats);
    return stats;
}

void PrintRansacStats(const std::string& prefix,
                      const RansacStats& stats,
                      size_t validCount,
                      bool scoreStats)
{
    if(!stats.attempted)
    {
        std::cout << prefix << " skipped" << std::endl;
        return;
    }
    if(!stats.success)
    {
        std::cout << prefix << " failed" << std::endl;
        return;
    }

    std::cout << prefix << "_inliers: " << stats.inliers << std::endl;
    std::cout << prefix << "_inlier_ratio: "
              << (validCount > 0 ? stats.inlierRatio : 0.0f) << std::endl;
    if(scoreStats)
    {
        std::cout << prefix << "_mean_inlier_score: " << stats.meanInlierScore << std::endl;
        std::cout << prefix << "_mean_outlier_score: " << stats.meanOutlierScore << std::endl;
    }
}

std::vector<ValidMatchRecord> BuildValidMatchRecords(
    const std::vector<ORB_SLAM3::LGMatch>& matches,
    const std::vector<cv::KeyPoint>& keypoints0,
    const std::vector<cv::KeyPoint>& keypoints1)
{
    std::vector<ValidMatchRecord> records;
    records.reserve(matches.size());
    for(const auto& m : matches)
    {
        if(m.idx0 < 0 || m.idx0 >= static_cast<int>(keypoints0.size()) ||
           m.idx1 < 0 || m.idx1 >= static_cast<int>(keypoints1.size()))
        {
            continue;
        }

        const cv::Point2f p0 = keypoints0[m.idx0].pt;
        const cv::Point2f p1 = keypoints1[m.idx1].pt;
        const cv::Point2f d = p1 - p0;
        records.push_back(ValidMatchRecord{m, p0, p1, std::sqrt(d.x * d.x + d.y * d.y)});
    }
    return records;
}

void SaveMatchImage(const std::string& outputPath,
                    const cv::Mat& img0,
                    const cv::Mat& img1,
                    const std::vector<cv::KeyPoint>& keypoints0,
                    const std::vector<cv::KeyPoint>& keypoints1,
                    const std::vector<ValidMatchRecord>& records,
                    const RansacStats& fStats)
{
    if(outputPath.empty())
        return;

    std::vector<cv::DMatch> drawMatchesVec;
    drawMatchesVec.reserve(records.size());
    const bool useFMask = fStats.success && fStats.mask.size() == records.size();
    for(size_t i = 0; i < records.size(); ++i)
    {
        if(useFMask && !fStats.mask[i])
            continue;
        drawMatchesVec.emplace_back(records[i].match.idx0,
                                    records[i].match.idx1,
                                    1.0f - records[i].match.score);
    }

    cv::Mat out;
    cv::drawMatches(img0, keypoints0, img1, keypoints1, drawMatchesVec, out);
    if(!cv::imwrite(outputPath, out))
        throw std::runtime_error("failed to write output_png: " + outputPath);
    std::cout << "output_png: " << outputPath
              << " drawn_matches=" << drawMatchesVec.size()
              << (useFMask ? " source=F_inliers" : " source=all_valid_matches")
              << std::endl;
}

bool ParseDouble(const std::string& value, double& out)
{
    try
    {
        size_t pos = 0;
        out = std::stod(value, &pos);
        return pos == value.size();
    }
    catch(...)
    {
        return false;
    }
}

void PrintUsage()
{
    std::cerr << "Usage: ./test_xfeat_lighterglue_images_cpp "
              << "/path/to/xfeat.pt /path/to/xfeat-lighterglue.pt "
              << "/path/to/img0.png /path/to/img1.png "
              << "[top_k] [lg_threshold] [output_png] [fx fy cx cy]" << std::endl
              << "   or: ./test_xfeat_lighterglue_images_cpp "
              << "/path/to/xfeat.pt /path/to/xfeat-lighterglue.pt "
              << "/path/to/img0.png /path/to/img1.png "
              << "[top_k] [lg_threshold] [fx fy cx cy] [output_png]" << std::endl;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        torch::NoGradGuard no_grad;

        if(argc < 5 || argc > 12)
        {
            PrintUsage();
            return 1;
        }

        const std::string xfeatWeights = argv[1];
        const std::string lighterGlueWeights = argv[2];
        const std::string img0Path = argv[3];
        const std::string img1Path = argv[4];
        const int topK = argc > 5 ? std::max(1, std::stoi(argv[5])) : 2048;
        const float lgThreshold = argc > 6 ? std::stof(argv[6]) : 0.1f;
        std::string outputPng;
        EssentialOptions essentialOptions;

        if(argc == 11)
        {
            essentialOptions.enabled = ParseDouble(argv[7], essentialOptions.fx) &&
                                       ParseDouble(argv[8], essentialOptions.fy) &&
                                       ParseDouble(argv[9], essentialOptions.cx) &&
                                       ParseDouble(argv[10], essentialOptions.cy);
            if(!essentialOptions.enabled)
                throw std::runtime_error("fx fy cx cy must be numeric.");
        }
        else if(argc == 8)
        {
            outputPng = argv[7];
        }
        else if(argc == 12)
        {
            EssentialOptions tailOutputOptions;
            const bool tailOutputHasK = ParseDouble(argv[7], tailOutputOptions.fx) &&
                                        ParseDouble(argv[8], tailOutputOptions.fy) &&
                                        ParseDouble(argv[9], tailOutputOptions.cx) &&
                                        ParseDouble(argv[10], tailOutputOptions.cy);
            if(tailOutputHasK)
            {
                essentialOptions = tailOutputOptions;
                essentialOptions.enabled = true;
                outputPng = argv[11];
            }
            else
            {
                outputPng = argv[7];
                essentialOptions.enabled = ParseDouble(argv[8], essentialOptions.fx) &&
                                           ParseDouble(argv[9], essentialOptions.fy) &&
                                           ParseDouble(argv[10], essentialOptions.cx) &&
                                           ParseDouble(argv[11], essentialOptions.cy);
                if(!essentialOptions.enabled)
                    throw std::runtime_error("fx fy cx cy must be numeric.");
            }
        }
        else if(argc > 7)
        {
            PrintUsage();
            return 1;
        }

        const torch::Device device = SelectDeviceAndConfigureExtractor();

        cv::Mat img0 = ReadImageForXFeat(img0Path);
        cv::Mat img1 = ReadImageForXFeat(img1Path);

        ORB_SLAM3::XFextractor extractor(topK, 1.2f, 8, 20, 7, xfeatWeights);

        std::vector<cv::KeyPoint> keypoints0;
        std::vector<cv::KeyPoint> keypoints1;
        cv::Mat descMat0;
        cv::Mat descMat1;
        std::vector<int> noLappingArea;

        extractor(img0, cv::Mat(), keypoints0, descMat0, noLappingArea);
        extractor(img1, cv::Mat(), keypoints1, descMat1, noLappingArea);

        auto kpts0 = KeyPointsToTensor(keypoints0, device);
        auto kpts1 = KeyPointsToTensor(keypoints1, device);
        auto desc0 = DescriptorsToTensor(descMat0, device, "img0");
        auto desc1 = DescriptorsToTensor(descMat1, device, "img1");
        auto size0 = ImageSizeToTensor(img0.cols, img0.rows, device);
        auto size1 = ImageSizeToTensor(img1.cols, img1.rows, device);

        ORB_SLAM3::XFeatLightGlueConfig config;
        ORB_SLAM3::XFeatLighterGlue matcher(config);
        matcher.To(device);
        matcher.LoadWeights(lighterGlueWeights);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "device: " << device.str() << std::endl;
        std::cout << "cuda available: " << (torch::cuda::is_available() ? "true" : "false") << std::endl;
        PrintImageDiagnostics("img0", img0, keypoints0, desc0);
        PrintImageDiagnostics("img1", img1, keypoints1, desc1);
        std::cout << "LightGlue threshold: " << lgThreshold << std::endl;

        const auto matches = matcher.Match(kpts0, desc0, size0,
                                           kpts1, desc1, size1,
                                           lgThreshold);

        std::cout << "number of matches: " << matches.size() << std::endl;

        const auto validRecords = BuildValidMatchRecords(matches, keypoints0, keypoints1);
        std::vector<cv::Point2f> pts0;
        std::vector<cv::Point2f> pts1;
        pts0.reserve(validRecords.size());
        pts1.reserve(validRecords.size());
        for(const auto& r : validRecords)
        {
            pts0.push_back(r.p0);
            pts1.push_back(r.p1);
        }

        const auto basicStats = ComputeBasicStats(validRecords);
        std::cout << "raw_matches: " << matches.size() << std::endl;
        std::cout << "valid_index_matches: " << validRecords.size() << std::endl;
        std::cout << "mean_score: " << basicStats.meanScore << std::endl;
        std::cout << "median_score: " << basicStats.medianScore << std::endl;
        std::cout << "mean_displacement_px: " << basicStats.meanDisplacement << std::endl;
        std::cout << "median_displacement_px: " << basicStats.medianDisplacement << std::endl;

        const auto fStats = RunFundamentalRansac(pts0, pts1, validRecords);
        const auto hStats = RunHomographyRansac(pts0, pts1, validRecords);
        const auto eStats = RunEssentialRansac(pts0, pts1, validRecords, essentialOptions);

        PrintRansacStats("F", fStats, validRecords.size(), true);
        PrintRansacStats("H", hStats, validRecords.size(), true);
        if(essentialOptions.enabled)
            PrintRansacStats("E", eStats, validRecords.size(), false);
        else
            std::cout << "E skipped" << std::endl;

        SaveMatchImage(outputPng, img0, img1, keypoints0, keypoints1, validRecords, fStats);

        const size_t show = std::min<size_t>(20, validRecords.size());
        for(size_t i = 0; i < show; ++i)
        {
            const auto& r = validRecords[i];
            const auto& m = r.match;
            const bool fInlier = fStats.success && fStats.mask.size() == validRecords.size() && fStats.mask[i];
            const bool hInlier = hStats.success && hStats.mask.size() == validRecords.size() && hStats.mask[i];
            const bool eInlier = eStats.success && eStats.mask.size() == validRecords.size() && eStats.mask[i];
            std::cout << "match[" << i << "]: idx0=" << m.idx0
                      << " idx1=" << m.idx1
                      << " score=" << m.score
                      << " x0=" << r.p0.x
                      << " y0=" << r.p0.y
                      << " x1=" << r.p1.x
                      << " y1=" << r.p1.y
                      << " displacement=" << r.displacement
                      << " F_inlier=" << (fInlier ? 1 : 0)
                      << " H_inlier=" << (hInlier ? 1 : 0)
                      << " E_inlier=" << (eInlier ? 1 : 0)
                      << std::endl;
        }

        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "test_xfeat_lighterglue_images_cpp failed: " << e.what() << std::endl;
        return 1;
    }
}
