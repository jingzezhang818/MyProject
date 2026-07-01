#include "XFeatVocabulary.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace
{

struct Options
{
    std::string outputVocabulary;
    std::vector<std::string> inputs;
    int k = 10;
    int L = 5;
    int maxFrames = 0;
    int maxFeaturesPerFrame = 512;
    int maxKMeansIters = 0;
    double minChangedRatio = 0.0;
    bool vocabVerbose = true;
    bool binaryCompressed = true;
};

void PrintUsage(const char* argv0)
{
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " <output.dbow3|output.yml.gz> <descriptors.yml.gz> [more.yml.gz] [options]\n\n"
        << "Options:\n"
        << "  --k <N>                         Branching factor. Default: 10\n"
        << "  --L <N>                         Depth levels. Default: 5\n"
        << "  --max-frames <N>                0 means no limit. Default: 0\n"
        << "  --max-features-per-frame <N>    0 means all descriptors. Default: 512\n"
        << "  --max-kmeans-iters <N>          0 keeps exact convergence. Default: 0\n"
        << "  --min-changed-ratio <value>     0 keeps exact convergence. Default: 0\n"
        << "  --vocab-verbose <0|1>           Print HKMeans progress. Default: 1\n"
        << "  --binary-compressed <0|1>       Used for non-yml output. Default: 1\n";
}

int ParseInt(const std::string& value, const char* name)
{
    try
    {
        return std::stoi(value);
    }
    catch(...)
    {
        throw std::runtime_error(std::string("invalid integer for ") + name + ": " + value);
    }
}

double ParseDouble(const std::string& value, const char* name)
{
    try
    {
        return std::stod(value);
    }
    catch(...)
    {
        throw std::runtime_error(std::string("invalid double for ") + name + ": " + value);
    }
}

std::string RequireValue(int& i, int argc, char** argv, const char* option)
{
    if(i + 1 >= argc)
        throw std::runtime_error(std::string("missing value for ") + option);
    ++i;
    return argv[i];
}

Options ParseOptions(int argc, char** argv)
{
    Options options;
    std::vector<std::string> positional;

    for(int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if(arg == "--k")
            options.k = std::max(2, ParseInt(RequireValue(i, argc, argv, "--k"), "--k"));
        else if(arg == "--L")
            options.L = std::max(1, ParseInt(RequireValue(i, argc, argv, "--L"), "--L"));
        else if(arg == "--max-frames")
            options.maxFrames = std::max(0, ParseInt(RequireValue(i, argc, argv, "--max-frames"), "--max-frames"));
        else if(arg == "--max-features-per-frame")
            options.maxFeaturesPerFrame = std::max(0, ParseInt(RequireValue(i, argc, argv, "--max-features-per-frame"), "--max-features-per-frame"));
        else if(arg == "--max-kmeans-iters")
            options.maxKMeansIters = std::max(0, ParseInt(RequireValue(i, argc, argv, "--max-kmeans-iters"), "--max-kmeans-iters"));
        else if(arg == "--min-changed-ratio")
            options.minChangedRatio = std::max(0.0, ParseDouble(RequireValue(i, argc, argv, "--min-changed-ratio"), "--min-changed-ratio"));
        else if(arg == "--vocab-verbose")
            options.vocabVerbose = ParseInt(RequireValue(i, argc, argv, "--vocab-verbose"), "--vocab-verbose") != 0;
        else if(arg == "--binary-compressed")
            options.binaryCompressed = ParseInt(RequireValue(i, argc, argv, "--binary-compressed"), "--binary-compressed") != 0;
        else if(arg == "--help" || arg == "-h")
        {
            PrintUsage(argv[0]);
            std::exit(0);
        }
        else if(!arg.empty() && arg[0] == '-')
            throw std::runtime_error("unknown option: " + arg);
        else
            positional.push_back(arg);
    }

    if(positional.size() < 2)
    {
        PrintUsage(argv[0]);
        throw std::runtime_error("expected output vocabulary and at least one descriptor file");
    }

    options.outputVocabulary = positional.front();
    options.inputs.assign(positional.begin() + 1, positional.end());
    return options;
}

cv::Mat SampleRowsEvenly(const cv::Mat& descriptors, int maxRows)
{
    if(maxRows <= 0 || descriptors.rows <= maxRows)
        return descriptors.clone();

    cv::Mat sampled(maxRows, descriptors.cols, descriptors.type());
    if(maxRows == 1)
    {
        descriptors.row(0).copyTo(sampled.row(0));
        return sampled;
    }

    const double scale = static_cast<double>(descriptors.rows - 1) / static_cast<double>(maxRows - 1);
    for(int i = 0; i < maxRows; ++i)
    {
        const int row = std::min(descriptors.rows - 1, static_cast<int>(std::lround(i * scale)));
        descriptors.row(row).copyTo(sampled.row(i));
    }
    return sampled;
}

void LoadDescriptorFile(const std::string& path,
                        const Options& options,
                        std::vector<cv::Mat>& trainingFeatures,
                        int& loadedFrames,
                        long long& loadedDescriptors)
{
    const int startFrames = loadedFrames;
    const long long startDescriptors = loadedDescriptors;
    std::cout << "[XFeatVocab] loading descriptors: " << path << std::endl;

    cv::FileStorage fs(path, cv::FileStorage::READ);
    if(!fs.isOpened())
        throw std::runtime_error("failed to open descriptor file: " + path);

    const std::string format = static_cast<std::string>(fs["format"]);
    if(format != "xfeat_descriptors_v1")
        throw std::runtime_error("unexpected descriptor file format in " + path + ": " + format);

    const cv::FileNode frames = fs["frames"];
    if(frames.type() != cv::FileNode::SEQ)
        throw std::runtime_error("descriptor file has no frame sequence: " + path);

    for(cv::FileNodeIterator it = frames.begin(); it != frames.end(); ++it)
    {
        if(options.maxFrames > 0 && loadedFrames >= options.maxFrames)
            return;

        cv::Mat descriptors;
        (*it)["descriptors"] >> descriptors;
        if(descriptors.empty())
            continue;
        if(descriptors.type() != CV_32F)
            throw std::runtime_error("descriptor matrix must be CV_32F in " + path);
        if(descriptors.cols != 64)
            throw std::runtime_error("descriptor matrix must have 64 columns in " + path);

        cv::Mat sampled = SampleRowsEvenly(descriptors, options.maxFeaturesPerFrame);
        trainingFeatures.push_back(sampled);
        ++loadedFrames;
        loadedDescriptors += sampled.rows;
    }

    std::cout << "[XFeatVocab] loaded descriptors: " << path
              << " frames_added=" << (loadedFrames - startFrames)
              << " descriptors_added=" << (loadedDescriptors - startDescriptors)
              << " total_frames=" << loadedFrames
              << " total_descriptors=" << loadedDescriptors
              << std::endl;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const Options options = ParseOptions(argc, argv);

        std::vector<cv::Mat> trainingFeatures;
        int loadedFrames = 0;
        long long loadedDescriptors = 0;

        for(const std::string& input : options.inputs)
        {
            LoadDescriptorFile(input, options, trainingFeatures, loadedFrames, loadedDescriptors);
            if(options.maxFrames > 0 && loadedFrames >= options.maxFrames)
                break;
        }

        if(trainingFeatures.empty() || loadedDescriptors == 0)
            throw std::runtime_error("no descriptors were loaded for vocabulary training");

        std::cout << "[XFeatVocab] training files=" << options.inputs.size()
                  << " frames=" << loadedFrames
                  << " descriptors=" << loadedDescriptors
                  << " k=" << options.k
                  << " L=" << options.L
                  << " max_features_per_frame=" << options.maxFeaturesPerFrame
                  << " max_kmeans_iters=" << options.maxKMeansIters
                  << " min_changed_ratio=" << options.minChangedRatio
                  << std::endl;

        const auto start = std::chrono::steady_clock::now();
        ORB_SLAM3::XFeatVocabulary vocabulary(options.k, options.L, DBoW3::TF_IDF, DBoW3::L1_NORM);
        vocabulary.setKMeansStopCriteria(options.maxKMeansIters, options.minChangedRatio);
        vocabulary.setTrainingVerbose(options.vocabVerbose);
        std::cout << "[XFeatVocab] creating vocabulary tree..." << std::endl;
        vocabulary.create(trainingFeatures);
        const auto trained = std::chrono::steady_clock::now();

        std::cout << "[XFeatVocab] saving vocabulary: " << options.outputVocabulary << std::endl;
        vocabulary.save(options.outputVocabulary, options.binaryCompressed);
        const auto saved = std::chrono::steady_clock::now();

        DBoW3::BowVector testBow;
        vocabulary.transform(trainingFeatures.front(), testBow);

        const double trainSec = std::chrono::duration_cast<std::chrono::duration<double>>(trained - start).count();
        const double saveSec = std::chrono::duration_cast<std::chrono::duration<double>>(saved - trained).count();

        std::cout << "[XFeatVocab] saved=" << options.outputVocabulary
                  << " words=" << vocabulary.size()
                  << " effective_levels=" << vocabulary.getEffectiveLevels()
                  << " first_bow_words=" << testBow.size()
                  << " train_sec=" << trainSec
                  << " save_sec=" << saveSec
                  << std::endl;

        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "[XFeatVocab][ERROR] " << e.what() << std::endl;
        return 1;
    }
}
