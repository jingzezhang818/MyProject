#include "XFeat.h"
#include "XFeatLighterGlue/core.hpp"
#include "XFeatLighterGlue/matcher.hpp"

#include <filesystem>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace
{

bool HasPrefix(const std::string& value, const std::string& prefix)
{
    return value.rfind(prefix, 0) == 0;
}

void StripPrefix(std::string& value, const std::string& prefix)
{
    if(HasPrefix(value, prefix))
        value.erase(0, prefix.size());
}

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

std::string MapXFeatExtractorWeightKey(const std::string& key)
{
    std::string mapped = key;
    StripPrefix(mapped, "module.");
    StripPrefix(mapped, "extractor.");
    StripPrefix(mapped, "model.");
    StripPrefix(mapped, "net.");
    StripPrefix(mapped, "extractor.model.");
    StripPrefix(mapped, "extractor.model.net.");
    return mapped;
}

bool IsPotentialXFeatExtractorWeightKey(const std::string& key)
{
    return HasPrefix(key, "skip1.") ||
           HasPrefix(key, "block1.") ||
           HasPrefix(key, "block2.") ||
           HasPrefix(key, "block3.") ||
           HasPrefix(key, "block4.") ||
           HasPrefix(key, "block5.") ||
           HasPrefix(key, "block_fusion.") ||
           HasPrefix(key, "heatmap_head.") ||
           HasPrefix(key, "keypoint_head.") ||
           HasPrefix(key, "fine_matcher.") ||
           HasPrefix(key, "norm.");
}

struct LoadReport
{
    size_t loaded = 0;
    size_t missing = 0;
    size_t unexpected = 0;
    size_t skipped = 0;
    size_t shapeMismatch = 0;
};

LoadReport LoadNamedTensorsIntoModule(
    torch::nn::Module& module,
    const std::vector<std::pair<std::string, torch::Tensor>>& weights,
    const std::function<std::string(const std::string&)>& mapKey,
    const std::function<bool(const std::string&)>& isPotentialKey,
    const std::string& logPrefix,
    bool allowMissingBuffers)
{
    torch::NoGradGuard noGrad;

    std::unordered_map<std::string, torch::Tensor> paramMap;
    std::unordered_map<std::string, torch::Tensor> bufferMap;
    std::unordered_set<std::string> seen;

    for(const auto& p : module.named_parameters(true))
        paramMap.emplace(p.key(), p.value());
    for(const auto& b : module.named_buffers(true))
        bufferMap.emplace(b.key(), b.value());

    LoadReport report;
    for(const auto& item : weights)
    {
        const std::string& loadedKey = item.first;
        const std::string mappedKey = mapKey(loadedKey);

        if(!isPotentialKey(mappedKey))
        {
            ++report.skipped;
            std::cout << logPrefix << " skipped key: " << loadedKey << std::endl;
            continue;
        }

        std::cout << logPrefix << " loaded key: " << loadedKey << std::endl;
        std::cout << logPrefix << " mapped key: " << loadedKey
                  << " -> " << mappedKey << std::endl;

        const auto& source = item.second;
        auto pIt = paramMap.find(mappedKey);
        if(pIt != paramMap.end())
        {
            if(pIt->second.sizes() != source.sizes())
            {
                ++report.shapeMismatch;
                std::cerr << logPrefix << " shape mismatch: " << loadedKey
                          << " -> " << mappedKey
                          << " expected " << ShapeString(pIt->second)
                          << " got " << ShapeString(source) << std::endl;
                continue;
            }
            pIt->second.copy_(source.to(pIt->second.scalar_type()).to(pIt->second.device()));
            seen.insert(mappedKey);
            ++report.loaded;
            continue;
        }

        auto bIt = bufferMap.find(mappedKey);
        if(bIt != bufferMap.end())
        {
            if(bIt->second.sizes() != source.sizes())
            {
                ++report.shapeMismatch;
                std::cerr << logPrefix << " shape mismatch: " << loadedKey
                          << " -> " << mappedKey
                          << " expected " << ShapeString(bIt->second)
                          << " got " << ShapeString(source) << std::endl;
                continue;
            }
            bIt->second.copy_(source.to(bIt->second.scalar_type()).to(bIt->second.device()));
            seen.insert(mappedKey);
            ++report.loaded;
            continue;
        }

        ++report.unexpected;
        std::cout << logPrefix << " unexpected key: " << loadedKey
                  << " -> " << mappedKey << std::endl;
    }

    for(const auto& p : paramMap)
    {
        if(seen.find(p.first) == seen.end())
        {
            ++report.missing;
            std::cout << logPrefix << " missing key: " << p.first << std::endl;
        }
    }
    for(const auto& b : bufferMap)
    {
        if(seen.find(b.first) == seen.end())
        {
            if(allowMissingBuffers)
                continue;
            ++report.missing;
            std::cout << logPrefix << " missing key: " << b.first << std::endl;
        }
    }

    std::cout << logPrefix << " load report: loaded=" << report.loaded
              << " missing=" << report.missing
              << " unexpected=" << report.unexpected
              << " skipped=" << report.skipped
              << " shape_mismatch=" << report.shapeMismatch
              << std::endl;

    return report;
}

void CheckStrictReport(const LoadReport& report, const std::string& name)
{
    if(report.loaded == 0)
        throw std::runtime_error(name + " did not load any tensors.");
    if(report.missing != 0 || report.unexpected != 0 || report.shapeMismatch != 0)
    {
        std::ostringstream oss;
        oss << name << " load failed: loaded=" << report.loaded
            << " missing=" << report.missing
            << " unexpected=" << report.unexpected
            << " skipped=" << report.skipped
            << " shape_mismatch=" << report.shapeMismatch;
        throw std::runtime_error(oss.str());
    }
}

void EnsureReadableFile(const std::string& path, const std::string& name)
{
    if(path.empty() || !std::filesystem::exists(path))
        throw std::runtime_error(name + " not found: " + (path.empty() ? std::string("<empty>") : path));
}

void EnsureParentDirectoryExists(const std::string& path, const std::string& name)
{
    const std::filesystem::path p(path);
    const auto parent = p.parent_path();
    if(!parent.empty() && !std::filesystem::exists(parent))
        throw std::runtime_error(name + " parent directory does not exist: " + parent.string());
}

void SaveModuleArchive(torch::nn::Module& module,
                       const std::string& outputPath,
                       const std::string& name)
{
    EnsureParentDirectoryExists(outputPath, name);
    torch::serialize::OutputArchive archive;
    module.save(archive);
    archive.save_to(outputPath);
    std::cout << "[convert] saved " << name << ": " << outputPath << std::endl;
}

void VerifyXFeatArchive(const std::string& path)
{
    ORB_SLAM3::XFeatModel model;
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    model.load(archive);
    std::cout << "[convert] verified XFeat InputArchive: " << path << std::endl;
}

void VerifyLighterGlueArchive(const std::string& path)
{
    ORB_SLAM3::XFeatLighterGlue matcher;
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    matcher.load(archive);
    std::cout << "[convert] verified XFeatLighterGlue InputArchive: " << path << std::endl;
}

void PrintUsage()
{
    std::cerr << "Usage: ./convert_xfeat_lighterglue_weights_cpp "
              << "/path/to/xfeat.pt "
              << "/path/to/xfeat-lighterglue.pt "
              << "/path/to/out_xfeat_cpp.pt "
              << "/path/to/out_lighterglue_cpp.pt"
              << std::endl;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if(argc != 5)
        {
            PrintUsage();
            return 1;
        }

        const std::string xfeatInput = argv[1];
        const std::string lighterGlueInput = argv[2];
        const std::string xfeatOutput = argv[3];
        const std::string lighterGlueOutput = argv[4];

        EnsureReadableFile(xfeatInput, "xfeat input");
        EnsureReadableFile(lighterGlueInput, "lighterglue input");

        std::cout << "[convert] loading XFeat Python state_dict: " << xfeatInput << std::endl;
        auto xfeatWeights = ORB_SLAM3::LoadPyTorchCheckpointTensors(xfeatInput);
        ORB_SLAM3::XFeatModel xfeatModel;
        auto xfeatReport = LoadNamedTensorsIntoModule(
            xfeatModel,
            xfeatWeights,
            MapXFeatExtractorWeightKey,
            IsPotentialXFeatExtractorWeightKey,
            "[convert][XFeat]",
            false);
        CheckStrictReport(xfeatReport, "XFeat");
        xfeatModel.eval();
        SaveModuleArchive(xfeatModel, xfeatOutput, "XFeat C++ archive");
        VerifyXFeatArchive(xfeatOutput);

        std::cout << "[convert] loading XFeatLighterGlue matcher weights: "
                  << lighterGlueInput << std::endl;
        ORB_SLAM3::XFeatLighterGlue matcher;
        matcher.LoadWeights(lighterGlueInput);
        matcher.eval();
        SaveModuleArchive(matcher, lighterGlueOutput, "XFeatLighterGlue matcher C++ archive");
        VerifyLighterGlueArchive(lighterGlueOutput);

        std::cout << "[convert] conversion complete." << std::endl;
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "convert_xfeat_lighterglue_weights_cpp failed: "
                  << e.what() << std::endl;
        return 1;
    }
}
