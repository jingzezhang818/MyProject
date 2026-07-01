#include "XFextractor.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace
{

struct Options
{
    std::string mav0Dir;
    std::string timestamps;
    std::string output;
    std::string camera = "cam0";
    std::string weights = "weights/xfeat.pt";
    int step = 1;
    int maxImages = 0;
    int nFeatures = 1024;
    int nLevels = 1;
    float scaleFactor = 1.2f;
};

void PrintUsage(const char* argv0)
{
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " <euroc_mav0_dir> <timestamps.txt> <output.yml.gz> [options]\n\n"
        << "Options:\n"
        << "  --camera <cam0|cam1>           Default: cam0\n"
        << "  --weights <path>               Default: weights/xfeat.pt\n"
        << "  --step <N>                     Export every Nth image. Default: 1\n"
        << "  --max-images <N>               0 means no limit. Default: 0\n"
        << "  --nfeatures <N>                Default: 1024\n"
        << "  --nlevels <N>                  Default: 1\n"
        << "  --scale-factor <value>         Default: 1.2\n";
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

float ParseFloat(const std::string& value, const char* name)
{
    try
    {
        return std::stof(value);
    }
    catch(...)
    {
        throw std::runtime_error(std::string("invalid float for ") + name + ": " + value);
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
        if(arg == "--camera")
            options.camera = RequireValue(i, argc, argv, "--camera");
        else if(arg == "--weights")
            options.weights = RequireValue(i, argc, argv, "--weights");
        else if(arg == "--step")
            options.step = std::max(1, ParseInt(RequireValue(i, argc, argv, "--step"), "--step"));
        else if(arg == "--max-images")
            options.maxImages = std::max(0, ParseInt(RequireValue(i, argc, argv, "--max-images"), "--max-images"));
        else if(arg == "--nfeatures")
            options.nFeatures = std::max(1, ParseInt(RequireValue(i, argc, argv, "--nfeatures"), "--nfeatures"));
        else if(arg == "--nlevels")
            options.nLevels = std::max(1, ParseInt(RequireValue(i, argc, argv, "--nlevels"), "--nlevels"));
        else if(arg == "--scale-factor")
            options.scaleFactor = std::max(1.0f, ParseFloat(RequireValue(i, argc, argv, "--scale-factor"), "--scale-factor"));
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

    if(positional.size() != 3)
    {
        PrintUsage(argv[0]);
        throw std::runtime_error("expected exactly 3 positional arguments");
    }

    options.mav0Dir = positional[0];
    options.timestamps = positional[1];
    options.output = positional[2];

    if(options.camera != "cam0" && options.camera != "cam1")
        throw std::runtime_error("--camera must be cam0 or cam1");

    return options;
}

std::vector<std::string> LoadTimestampTokens(const std::string& path)
{
    std::ifstream file(path.c_str());
    if(!file.is_open())
        throw std::runtime_error("failed to open timestamp file: " + path);

    std::vector<std::string> tokens;
    std::string line;
    while(std::getline(file, line))
    {
        if(line.empty() || line[0] == '#')
            continue;

        std::stringstream ss(line);
        std::string token;
        ss >> token;
        if(!token.empty())
            tokens.push_back(token);
    }
    return tokens;
}

double TimestampToSeconds(const std::string& token)
{
    try
    {
        return std::stod(token) * 1e-9;
    }
    catch(...)
    {
        return 0.0;
    }
}

std::string JoinPath(const std::string& a, const std::string& b)
{
    if(a.empty() || a.back() == '/')
        return a + b;
    return a + "/" + b;
}

std::string ImagePath(const Options& options, const std::string& stamp)
{
    return JoinPath(JoinPath(JoinPath(options.mav0Dir, options.camera), "data"), stamp + ".png");
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const Options options = ParseOptions(argc, argv);
        const std::vector<std::string> stamps = LoadTimestampTokens(options.timestamps);
        if(stamps.empty())
            throw std::runtime_error("timestamp file contains no image names: " + options.timestamps);

        ORB_SLAM3::XFextractor extractor(options.nFeatures,
                                         options.scaleFactor,
                                         options.nLevels,
                                         20,
                                         7,
                                         options.weights);

        cv::FileStorage fs(options.output, cv::FileStorage::WRITE);
        if(!fs.isOpened())
            throw std::runtime_error("failed to open output file: " + options.output);

        fs << "format" << "xfeat_descriptors_v1";
        fs << "source" << options.mav0Dir;
        fs << "timestamps" << options.timestamps;
        fs << "camera" << options.camera;
        fs << "weights" << options.weights;
        fs << "nfeatures" << options.nFeatures;
        fs << "nlevels" << options.nLevels;
        fs << "scale_factor" << static_cast<double>(options.scaleFactor);
        fs << "step" << options.step;
        fs << "frames" << "[";

        int exportedFrames = 0;
        int skippedImages = 0;
        long long exportedDescriptors = 0;

        for(size_t i = 0; i < stamps.size(); i += static_cast<size_t>(options.step))
        {
            if(options.maxImages > 0 && exportedFrames >= options.maxImages)
                break;

            const std::string path = ImagePath(options, stamps[i]);
            cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
            if(image.empty())
            {
                ++skippedImages;
                std::cerr << "[XFeatExport] skip missing image: " << path << std::endl;
                continue;
            }
            if(image.depth() != CV_8U || (image.channels() != 1 && image.channels() != 3))
            {
                ++skippedImages;
                std::cerr << "[XFeatExport] skip unsupported image: " << path
                          << " type=" << image.type() << std::endl;
                continue;
            }

            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            std::vector<int> noStereoLappingArea;
            extractor(image, cv::Mat(), keypoints, descriptors, noStereoLappingArea);

            if(!descriptors.empty() && descriptors.type() != CV_32F)
                throw std::runtime_error("XFextractor returned non-CV_32F descriptors");
            if(!descriptors.empty() && descriptors.cols != 64)
                throw std::runtime_error("XFextractor returned descriptors with cols != 64");

            fs << "{";
            fs << "stamp" << stamps[i];
            fs << "timestamp" << TimestampToSeconds(stamps[i]);
            fs << "image" << path;
            fs << "width" << image.cols;
            fs << "height" << image.rows;
            fs << "keypoints" << static_cast<int>(keypoints.size());
            fs << "descriptors" << descriptors;
            fs << "}";

            ++exportedFrames;
            exportedDescriptors += descriptors.rows;
            if(exportedFrames % 50 == 0)
            {
                std::cout << "[XFeatExport] frames=" << exportedFrames
                          << " descriptors=" << exportedDescriptors
                          << " last=" << path << std::endl;
            }
        }

        fs << "]";
        fs << "exported_frames" << exportedFrames;
        fs << "exported_descriptors" << static_cast<double>(exportedDescriptors);
        fs << "skipped_images" << skippedImages;
        fs.release();

        std::cout << "[XFeatExport] done output=" << options.output
                  << " frames=" << exportedFrames
                  << " descriptors=" << exportedDescriptors
                  << " skipped=" << skippedImages << std::endl;
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "[XFeatExport][ERROR] " << e.what() << std::endl;
        return 1;
    }
}
