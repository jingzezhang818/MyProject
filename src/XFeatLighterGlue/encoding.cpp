#include "XFeatLighterGlue/encoding.hpp"

#include <cmath>

namespace ORB_SLAM3
{

LearnableFourierPosEnc::LearnableFourierPosEnc(
    int input_dim,
    int head_dim,
    torch::optional<int> fourier_dim,
    float gamma)
    : gamma_(gamma), head_dim_(fourier_dim.value_or(head_dim))
{
    if(head_dim_ % 2 != 0)
        throw std::invalid_argument("LearnableFourierPosEnc requires an even head dimension.");

    Wr_ = register_module(
        "Wr",
        torch::nn::Linear(torch::nn::LinearOptions(input_dim, head_dim_ / 2).bias(false)));

    const double stddev = 1.0 / (static_cast<double>(gamma_) * static_cast<double>(gamma_));
    torch::nn::init::normal_(Wr_->weight, 0.0, stddev);
}

torch::Tensor LearnableFourierPosEnc::forward(const torch::Tensor& x)
{
    auto projected = Wr_->forward(x);
    auto cosines = torch::cos(projected);
    auto sines = torch::sin(projected);

    auto emb = torch::stack({cosines, sines}, 0).unsqueeze(-3);
    return emb.repeat_interleave(2, -1);
}

} // namespace ORB_SLAM3
