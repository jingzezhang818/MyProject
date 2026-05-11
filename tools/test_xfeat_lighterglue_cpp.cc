#include "XFeatLighterGlue/matcher.hpp"

#include <algorithm>
#include <cctype>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>

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

torch::Tensor L2NormalizeRows(const torch::Tensor& x)
{
    return x / x.pow(2).sum(1, true).sqrt().clamp_min(1e-12);
}

torch::Tensor RandomKeypoints(int64_t n, float width, float height)
{
    auto kpts = torch::rand({n, 2}, torch::TensorOptions().dtype(torch::kFloat32));
    kpts.index_put_({torch::indexing::Slice(), 0}, kpts.index({torch::indexing::Slice(), 0}) * width);
    kpts.index_put_({torch::indexing::Slice(), 1}, kpts.index({torch::indexing::Slice(), 1}) * height);
    return kpts;
}

std::string ToLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

torch::Device SelectDevice(int argc, char** argv)
{
    std::string requested = "auto";
    if(argc > 2)
        requested = ToLower(argv[2]);

    if(requested == "cpu")
        return torch::Device(torch::kCPU);

    if(requested == "cuda" || requested == "gpu")
    {
        if(!torch::cuda::is_available())
            throw std::runtime_error("CUDA was requested, but torch::cuda::is_available() is false.");
        return torch::Device(torch::kCUDA, 0);
    }

    if(requested != "auto")
        throw std::runtime_error("Usage: ./test_xfeat_lighterglue_cpp [weights.pt] [auto|cuda|cpu]");

    return torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0)
                                       : torch::Device(torch::kCPU);
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        torch::manual_seed(7);
        torch::NoGradGuard no_grad;

        const torch::Device device = SelectDevice(argc, argv);
        const float width = 640.0f;
        const float height = 480.0f;

        auto kpts0 = RandomKeypoints(512, width, height).to(device);
        auto kpts1 = RandomKeypoints(600, width, height).to(device);
        auto desc0 = L2NormalizeRows(torch::randn({512, 64}, torch::TensorOptions().dtype(torch::kFloat32))).to(device);
        auto desc1 = L2NormalizeRows(torch::randn({600, 64}, torch::TensorOptions().dtype(torch::kFloat32))).to(device);
        auto size0 = torch::tensor({width, height}, torch::TensorOptions().dtype(torch::kFloat32)).to(device);
        auto size1 = torch::tensor({width, height}, torch::TensorOptions().dtype(torch::kFloat32)).to(device);

        std::cout << "input shapes:" << std::endl;
        std::cout << "  kpts0: " << ShapeString(kpts0) << std::endl;
        std::cout << "  desc0: " << ShapeString(desc0) << std::endl;
        std::cout << "  size0: " << ShapeString(size0) << " [W, H]" << std::endl;
        std::cout << "  kpts1: " << ShapeString(kpts1) << std::endl;
        std::cout << "  desc1: " << ShapeString(desc1) << std::endl;
        std::cout << "  size1: " << ShapeString(size1) << " [W, H]" << std::endl;
        std::cout << "device: " << device.str() << std::endl;
        std::cout << "cuda available: " << (torch::cuda::is_available() ? "true" : "false") << std::endl;

        ORB_SLAM3::XFeatLightGlueConfig config;
        ORB_SLAM3::XFeatLighterGlue matcher(config);
        matcher.To(device);

        if(argc > 1)
            matcher.LoadWeights(argv[1]);
        else
            std::cerr << "[test_xfeat_lighterglue_cpp] no weight path provided; using random initialization." << std::endl;

        const auto matches = matcher.Match(kpts0, desc0, size0, kpts1, desc1, size1, config.filter_threshold);

        std::cout << "number of matches: " << matches.size() << std::endl;
        const size_t show = std::min<size_t>(10, matches.size());
        for(size_t i = 0; i < show; ++i)
        {
            std::cout << "match[" << i << "]: "
                      << matches[i].idx0 << " -> " << matches[i].idx1
                      << ", score=" << matches[i].score << std::endl;
        }

        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "test_xfeat_lighterglue_cpp failed: " << e.what() << std::endl;
        return 1;
    }
}
