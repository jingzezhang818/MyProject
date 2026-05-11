#ifndef XFEAT_LIGHTERGLUE_TRANSFORMER_HPP
#define XFEAT_LIGHTERGLUE_TRANSFORMER_HPP

#include <memory>
#include <tuple>

#include <torch/torch.h>

namespace ORB_SLAM3
{

class SelfBlock;
class CrossBlock;

class TransformerLayer : public torch::nn::Module
{
public:
    TransformerLayer(int embed_dim, int num_heads, bool flash = false, bool bias = true);

    std::tuple<torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& desc0,
        const torch::Tensor& desc1,
        const torch::Tensor& encoding0,
        const torch::Tensor& encoding1);

private:
    std::shared_ptr<SelfBlock> self_attn_;
    std::shared_ptr<CrossBlock> cross_attn_;
};

} // namespace ORB_SLAM3

#endif // XFEAT_LIGHTERGLUE_TRANSFORMER_HPP
