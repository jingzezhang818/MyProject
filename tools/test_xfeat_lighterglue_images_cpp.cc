#include "XFextractor.h"
#include "XFeatLighterGlue/matcher.hpp"

#include <algorithm>
#include <cctype>
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

void PrintUsage()
{
    std::cerr << "Usage: ./test_xfeat_lighterglue_images_cpp "
              << "/path/to/xfeat.pt /path/to/xfeat-lighterglue.pt "
              << "/path/to/img0.png /path/to/img1.png [top_k] [lg_threshold]" << std::endl;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        torch::NoGradGuard no_grad;

        if(argc < 5 || argc > 7)
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
        const size_t show = std::min<size_t>(20, matches.size());
        for(size_t i = 0; i < show; ++i)
        {
            const auto& m = matches[i];
            if(m.idx0 < 0 || m.idx0 >= static_cast<int>(keypoints0.size()) ||
               m.idx1 < 0 || m.idx1 >= static_cast<int>(keypoints1.size()))
            {
                std::cout << "match[" << i << "]: invalid index "
                          << m.idx0 << " " << m.idx1
                          << " score=" << m.score << std::endl;
                continue;
            }

            const auto& p0 = keypoints0[m.idx0].pt;
            const auto& p1 = keypoints1[m.idx1].pt;
            std::cout << "match[" << i << "]: idx0=" << m.idx0
                      << " idx1=" << m.idx1
                      << " score=" << m.score
                      << " x0=" << p0.x
                      << " y0=" << p0.y
                      << " x1=" << p1.x
                      << " y1=" << p1.y
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
