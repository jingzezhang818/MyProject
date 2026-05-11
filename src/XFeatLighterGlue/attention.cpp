#include "XFeatLighterGlue/attention.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ORB_SLAM3
{

TokenConfidence::TokenConfidence(int dim)
{
    token_ = register_module(
        "token",
        torch::nn::Sequential(
            torch::nn::Linear(dim, 1),
            torch::nn::Sigmoid()));
}

std::tuple<torch::Tensor, torch::Tensor> TokenConfidence::forward(
    const torch::Tensor& desc0,
    const torch::Tensor& desc1)
{
    return std::make_tuple(
        token_->forward(desc0.detach()).squeeze(-1),
        token_->forward(desc1.detach()).squeeze(-1));
}

Attention::Attention(bool allow_flash)
{
    // First C++ version intentionally disables flash attention for correctness.
    (void)allow_flash;
    enable_flash_ = false;
}

torch::Tensor Attention::forward(const torch::Tensor& q,
                                 const torch::Tensor& k,
                                 const torch::Tensor& v)
{
    if(q.size(-2) == 0 || k.size(-2) == 0)
    {
        std::vector<int64_t> shape(q.sizes().begin(), q.sizes().end());
        shape.back() = v.size(-1);
        return torch::zeros(shape, q.options());
    }

    (void)enable_flash_;
    const double scale = 1.0 / std::sqrt(static_cast<double>(q.size(-1)));
    auto sim = torch::einsum("...id,...jd->...ij", {q, k}) * scale;
    auto attn = torch::softmax(sim, -1);
    return torch::einsum("...ij,...jd->...id", {attn, v});
}

SelfBlock::SelfBlock(int embed_dim, int num_heads, bool flash, bool bias)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      head_dim_(embed_dim / num_heads)
{
    if(embed_dim % num_heads != 0)
        throw std::invalid_argument("SelfBlock embed_dim must be divisible by num_heads.");

    Wqkv_ = register_module(
        "Wqkv",
        torch::nn::Linear(torch::nn::LinearOptions(embed_dim, 3 * embed_dim).bias(bias)));
    inner_attn_ = register_module("inner_attn", std::make_shared<Attention>(flash));
    out_proj_ = register_module(
        "out_proj",
        torch::nn::Linear(torch::nn::LinearOptions(embed_dim, embed_dim).bias(bias)));
    ffn_ = register_module(
        "ffn",
        torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(2 * embed_dim, 2 * embed_dim).bias(bias)),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({2 * embed_dim}).elementwise_affine(true)),
            torch::nn::GELU(),
            torch::nn::Linear(torch::nn::LinearOptions(2 * embed_dim, embed_dim).bias(bias))));
}

torch::Tensor SelfBlock::rotate_half(const torch::Tensor& x) const
{
    auto paired = x.reshape({x.size(0), x.size(1), x.size(2), x.size(3) / 2, 2});
    auto x1 = paired.select(-1, 0);
    auto x2 = paired.select(-1, 1);
    return torch::stack({-x2, x1}, -1).reshape_as(x);
}

torch::Tensor SelfBlock::apply_cached_rotary_emb(const torch::Tensor& freqs,
                                                 const torch::Tensor& t) const
{
    return (t * freqs.select(0, 0)) + (rotate_half(t) * freqs.select(0, 1));
}

torch::Tensor SelfBlock::forward(const torch::Tensor& x,
                                 const torch::Tensor& encoding)
{
    auto qkv = Wqkv_->forward(x);
    qkv = qkv.reshape({x.size(0), x.size(1), num_heads_, head_dim_, 3}).transpose(1, 2);

    auto q = qkv.select(-1, 0);
    auto k = qkv.select(-1, 1);
    auto v = qkv.select(-1, 2);

    q = apply_cached_rotary_emb(encoding, q);
    k = apply_cached_rotary_emb(encoding, k);

    auto context = inner_attn_->forward(q, k, v);
    auto message = out_proj_->forward(context.transpose(1, 2).reshape({x.size(0), x.size(1), embed_dim_}));
    return x + ffn_->forward(torch::cat({x, message}, -1));
}

CrossBlock::CrossBlock(int embed_dim, int num_heads, bool flash, bool bias)
    : embed_dim_(embed_dim),
      heads_(num_heads),
      scale_(1.0f / std::sqrt(static_cast<float>(embed_dim / num_heads)))
{
    (void)flash;
    if(embed_dim % num_heads != 0)
        throw std::invalid_argument("CrossBlock embed_dim must be divisible by num_heads.");

    to_qk_ = register_module(
        "to_qk",
        torch::nn::Linear(torch::nn::LinearOptions(embed_dim, embed_dim).bias(bias)));
    to_v_ = register_module(
        "to_v",
        torch::nn::Linear(torch::nn::LinearOptions(embed_dim, embed_dim).bias(bias)));
    to_out_ = register_module(
        "to_out",
        torch::nn::Linear(torch::nn::LinearOptions(embed_dim, embed_dim).bias(bias)));
    ffn_ = register_module(
        "ffn",
        torch::nn::Sequential(
            torch::nn::Linear(torch::nn::LinearOptions(2 * embed_dim, 2 * embed_dim).bias(bias)),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({2 * embed_dim}).elementwise_affine(true)),
            torch::nn::GELU(),
            torch::nn::Linear(torch::nn::LinearOptions(2 * embed_dim, embed_dim).bias(bias))));
}

std::tuple<torch::Tensor, torch::Tensor> CrossBlock::forward(
    const torch::Tensor& x0,
    const torch::Tensor& x1,
    const torch::optional<torch::Tensor>& mask)
{
    auto reshape_for_attention = [this](const torch::Tensor& t) {
        return t.reshape({t.size(0), t.size(1), heads_, embed_dim_ / heads_}).transpose(1, 2);
    };

    auto qk0 = reshape_for_attention(to_qk_->forward(x0));
    auto qk1 = reshape_for_attention(to_qk_->forward(x1));
    auto v0 = reshape_for_attention(to_v_->forward(x0));
    auto v1 = reshape_for_attention(to_v_->forward(x1));

    qk0 = qk0 * std::sqrt(scale_);
    qk1 = qk1 * std::sqrt(scale_);

    auto sim = torch::einsum("bhid,bhjd->bhij", {qk0, qk1});
    if(mask.has_value())
        sim = sim.masked_fill(~mask.value(), -std::numeric_limits<float>::infinity());

    auto attn01 = torch::softmax(sim, -1);
    auto attn10 = torch::softmax(sim.transpose(-2, -1).contiguous(), -1);

    auto m0 = torch::einsum("bhij,bhjd->bhid", {attn01, v1});
    auto m1 = torch::einsum("bhji,bhid->bhjd", {attn10, v0});

    if(mask.has_value())
    {
        m0 = m0.nan_to_num();
        m1 = m1.nan_to_num();
    }

    m0 = to_out_->forward(m0.transpose(1, 2).reshape({x0.size(0), x0.size(1), embed_dim_}));
    m1 = to_out_->forward(m1.transpose(1, 2).reshape({x1.size(0), x1.size(1), embed_dim_}));

    auto out0 = x0 + ffn_->forward(torch::cat({x0, m0}, -1));
    auto out1 = x1 + ffn_->forward(torch::cat({x1, m1}, -1));
    return std::make_tuple(out0, out1);
}

} // namespace ORB_SLAM3
