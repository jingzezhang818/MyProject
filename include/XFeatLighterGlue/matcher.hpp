#ifndef XFEAT_LIGHTERGLUE_MATCHER_HPP
#define XFEAT_LIGHTERGLUE_MATCHER_HPP

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "XFeatLighterGlue/core.hpp"

namespace ORB_SLAM3
{

class LearnableFourierPosEnc;
class MatchAssignment;
class TokenConfidence;
class TransformerLayer;

class MatchAssignment : public torch::nn::Module
{
public:
    explicit MatchAssignment(int dim);

    torch::Tensor forward(const torch::Tensor& desc0,
                          const torch::Tensor& desc1);
    torch::Tensor get_matchability(const torch::Tensor& desc);

private:
    torch::Tensor sigmoid_log_double_softmax(const torch::Tensor& sim,
                                             const torch::Tensor& z0,
                                             const torch::Tensor& z1);

    int dim_{0};
    torch::nn::Linear matchability_{nullptr};
    torch::nn::Linear final_proj_{nullptr};
};

class XFeatLighterGlue : public torch::nn::Module
{
public:
    explicit XFeatLighterGlue(const XFeatLightGlueConfig& config = XFeatLightGlueConfig());

    std::vector<LGMatch> Match(
        const torch::Tensor& kpts0,
        const torch::Tensor& desc0,
        const torch::Tensor& size0,
        const torch::Tensor& kpts1,
        const torch::Tensor& desc1,
        const torch::Tensor& size1,
        float filterThreshold = 0.1f);
    std::vector<LGMatch> MatchPrepared(
        const torch::Tensor& normalizedKpts0,
        const torch::Tensor& desc0,
        const torch::Tensor& normalizedKpts1,
        const torch::Tensor& desc1,
        float filterThreshold = 0.1f);

    bool LoadWeights(const std::string& weight_path);
    void To(const torch::Device& device);
    torch::Device Device() const { return device_; }

private:
    torch::Tensor get_pruning_mask(const torch::optional<torch::Tensor>& confidences,
                                   const torch::Tensor& scores,
                                   int layer_index) const;
    bool check_if_stop(const torch::Tensor& confidences0,
                       const torch::Tensor& confidences1,
                       int layer_index,
                       int num_points) const;

    static std::vector<char> ReadFileBytes(const std::string& filename);
    bool LoadStateDict(const c10::Dict<c10::IValue, c10::IValue>& weights);
    bool LoadNamedTensors(const std::vector<std::pair<std::string, torch::Tensor>>& weights);

    XFeatLightGlueConfig config_;
    torch::Device device_{torch::kCPU};

    torch::nn::Linear input_proj_{nullptr};
    std::shared_ptr<LearnableFourierPosEnc> posenc_;
    std::vector<std::shared_ptr<TransformerLayer>> transformers_;
    std::vector<std::shared_ptr<MatchAssignment>> log_assignment_;
    std::vector<std::shared_ptr<TokenConfidence>> token_confidence_;
    torch::Tensor confidence_thresholds_;
};

} // namespace ORB_SLAM3

#endif // XFEAT_LIGHTERGLUE_MATCHER_HPP
