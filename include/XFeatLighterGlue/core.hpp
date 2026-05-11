#ifndef XFEAT_LIGHTERGLUE_CORE_HPP
#define XFEAT_LIGHTERGLUE_CORE_HPP

#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace ORB_SLAM3
{

struct XFeatLightGlueConfig
{
    std::string name{"xfeat"};
    int input_dim{64};
    int descriptor_dim{96};
    bool add_scale_ori{false};
    int n_layers{6};
    int num_heads{1};
    bool flash{false};
    bool mp{false};
    float depth_confidence{-1.0f};
    float width_confidence{0.95f};
    float filter_threshold{0.1f};
    std::string weights;
};

struct LGMatch
{
    int idx0{-1};
    int idx1{-1};
    float score{0.0f};
};

torch::Tensor normalize_keypoints(const torch::Tensor& kpts,
                                  const torch::Tensor& image_size);

std::vector<LGMatch> filter_matches(
    const torch::Tensor& scores,
    float threshold,
    const torch::optional<torch::Tensor>& indices0 = torch::nullopt,
    const torch::optional<torch::Tensor>& indices1 = torch::nullopt);

std::vector<std::pair<std::string, torch::Tensor>> LoadPyTorchCheckpointTensors(
    const std::string& weight_path);

std::string MapXFeatLighterGlueWeightKey(const std::string& key);

} // namespace ORB_SLAM3

#endif // XFEAT_LIGHTERGLUE_CORE_HPP
