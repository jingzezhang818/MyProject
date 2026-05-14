#ifndef XFEAT_LIGHTERGLUE_ATTENTION_HPP
#define XFEAT_LIGHTERGLUE_ATTENTION_HPP

#include <memory>
#include <tuple>

#include <torch/torch.h>

namespace ORB_SLAM3
{

class TokenConfidence : public torch::nn::Module
{
public:
    explicit TokenConfidence(int dim);

    std::tuple<torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& desc0,
        const torch::Tensor& desc1);

private:
    torch::nn::Sequential token_{nullptr};
};

class Attention : public torch::nn::Module
{
public:
    explicit Attention(bool allow_flash = false);

    torch::Tensor forward(const torch::Tensor& q,
                          const torch::Tensor& k,
                          const torch::Tensor& v);

private:
    bool enable_flash_{false};
};

class SelfBlock : public torch::nn::Module
{
public:
    SelfBlock(int embed_dim, int num_heads, bool flash = false, bool bias = true);

    torch::Tensor forward(const torch::Tensor& x,
                          const torch::Tensor& encoding);

private:
    torch::Tensor rotate_half(const torch::Tensor& x) const;
    torch::Tensor apply_cached_rotary_emb(const torch::Tensor& freqs,
                                          const torch::Tensor& t) const;

    int embed_dim_{0};
    int num_heads_{0};
    int head_dim_{0};
    torch::nn::Linear Wqkv_{nullptr};
    std::shared_ptr<Attention> inner_attn_;
    torch::nn::Linear out_proj_{nullptr};
    torch::nn::Sequential ffn_{nullptr};
};

class CrossBlock : public torch::nn::Module
{
public:
    CrossBlock(int embed_dim, int num_heads, bool flash = false, bool bias = true);

    std::tuple<torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& x0,
        const torch::Tensor& x1,
        const torch::optional<torch::Tensor>& mask = torch::nullopt);

private:
    int embed_dim_{0};
    int heads_{0};
    float scale_{1.0f};
    bool enable_flash_{false};
    torch::nn::Linear to_qk_{nullptr};
    torch::nn::Linear to_v_{nullptr};
    torch::nn::Linear to_out_{nullptr};
    torch::nn::Sequential ffn_{nullptr};
};

} // namespace ORB_SLAM3

#endif // XFEAT_LIGHTERGLUE_ATTENTION_HPP
