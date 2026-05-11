#ifndef XFEAT_LIGHTERGLUE_ENCODING_HPP
#define XFEAT_LIGHTERGLUE_ENCODING_HPP

#include <torch/torch.h>

namespace ORB_SLAM3
{

class LearnableFourierPosEnc : public torch::nn::Module
{
public:
    LearnableFourierPosEnc(int input_dim,
                           int head_dim,
                           torch::optional<int> fourier_dim = torch::nullopt,
                           float gamma = 1.0f);

    torch::Tensor forward(const torch::Tensor& x);

private:
    float gamma_{1.0f};
    int head_dim_{0};
    torch::nn::Linear Wr_{nullptr};
};

} // namespace ORB_SLAM3

#endif // XFEAT_LIGHTERGLUE_ENCODING_HPP
