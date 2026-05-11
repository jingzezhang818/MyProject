#include "XFeatLighterGlue/core.hpp"

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <torch/torch.h>

namespace ORB_SLAM3
{
namespace
{

bool StartsWith(const std::string& value, const std::string& prefix)
{
    return value.rfind(prefix, 0) == 0;
}

bool IsEnvFlagEnabled(const char* key)
{
    const char* env = std::getenv(key);
    if(!env)
        return false;

    const std::string value(env);
    return !(value.empty() || value == "0" || value == "false" || value == "FALSE");
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

void StripPrefix(std::string& value, const std::string& prefix)
{
    if(StartsWith(value, prefix))
        value.erase(0, prefix.size());
}

bool RewriteOldAttentionKey(std::string& value,
                            const std::string& old_prefix,
                            const std::string& new_block_name)
{
    const std::string prefix = old_prefix + ".";
    if(!StartsWith(value, prefix))
        return false;

    size_t pos = prefix.size();
    size_t digit_end = pos;
    while(digit_end < value.size() && std::isdigit(static_cast<unsigned char>(value[digit_end])))
        ++digit_end;

    if(digit_end == pos || digit_end >= value.size() || value[digit_end] != '.')
        return false;

    const std::string layer_index = value.substr(pos, digit_end - pos);
    const std::string suffix = value.substr(digit_end);
    value = "transformers" + layer_index + "." + new_block_name + suffix;
    return true;
}

bool RewriteModuleListKey(std::string& value, const std::string& module_prefix)
{
    const std::string prefix = module_prefix + ".";
    if(!StartsWith(value, prefix))
        return false;

    size_t pos = prefix.size();
    size_t digit_end = pos;
    while(digit_end < value.size() && std::isdigit(static_cast<unsigned char>(value[digit_end])))
        ++digit_end;

    if(digit_end == pos || digit_end >= value.size() || value[digit_end] != '.')
        return false;

    const std::string layer_index = value.substr(pos, digit_end - pos);
    const std::string suffix = value.substr(digit_end);
    value = module_prefix + layer_index + suffix;
    return true;
}

int64_t ReadIndex(const torch::Tensor& flat_indices, int64_t local_index)
{
    if(!flat_indices.defined() || flat_indices.numel() == 0)
        return local_index;
    return flat_indices.index({local_index}).item<int64_t>();
}

} // namespace

torch::Tensor normalize_keypoints(const torch::Tensor& kpts,
                                  const torch::Tensor& image_size)
{
    if(kpts.size(-1) != 2)
        throw std::invalid_argument("normalize_keypoints expects keypoints with last dimension 2.");
    if(image_size.numel() != 2)
        throw std::invalid_argument("normalize_keypoints expects image_size [width, height].");

    // image_size order is [width, height], matching LightGlue.
    auto size = image_size.to(kpts.options()).view({2});
    auto shift = size / 2.0;
    auto scale = torch::max(size) / 2.0;
    return (kpts - shift) / scale;
}

std::vector<LGMatch> filter_matches(
    const torch::Tensor& scores,
    float threshold,
    const torch::optional<torch::Tensor>& indices0,
    const torch::optional<torch::Tensor>& indices1)
{
    if(scores.dim() != 3 || scores.size(0) != 1)
        throw std::invalid_argument("filter_matches expects assignment scores with shape [1, M+1, N+1].");

    const int64_t M = scores.size(1) - 1;
    const int64_t N = scores.size(2) - 1;
    if(M <= 0 || N <= 0)
        return {};

    const auto valid_scores = scores.slice(1, 0, M).slice(2, 0, N);
    const auto max0 = valid_scores.max(2);
    const auto max1 = valid_scores.max(1);

    auto best_score0 = std::get<0>(max0);
    auto m0 = std::get<1>(max0);
    auto m1 = std::get<1>(max1);

    auto indices0_local = torch::arange(M, m0.options()).unsqueeze(0);
    auto mutual0 = indices0_local == m1.gather(1, m0);
    auto mscores0 = torch::where(mutual0,
                                 best_score0.exp(),
                                 torch::zeros_like(best_score0));
    auto valid0 = mutual0 & (mscores0 > threshold);

    if(IsEnvFlagEnabled("XFEAT_LG_MATCH_DEBUG"))
    {
        std::cout << "[XFeatLighterGlue][debug] scores shape: " << ShapeString(scores) << std::endl;
        std::cout << "[XFeatLighterGlue][debug] scores min/max: "
                  << scores.min().item<float>() << " / "
                  << scores.max().item<float>() << std::endl;
        std::cout << "[XFeatLighterGlue][debug] mutual count: "
                  << mutual0.sum().item<int64_t>() << std::endl;
        std::cout << "[XFeatLighterGlue][debug] mscores0 max/mean: "
                  << mscores0.max().item<float>() << " / "
                  << mscores0.mean().item<float>() << std::endl;
        std::cout << "[XFeatLighterGlue][debug] valid match count: "
                  << valid0.sum().item<int64_t>() << std::endl;
    }

    auto where_valid = torch::where(valid0.squeeze(0));
    if(where_valid.empty() || where_valid[0].numel() == 0)
        return {};

    auto local0 = where_valid[0].to(torch::kCPU).contiguous();
    auto local1 = m0.squeeze(0).index_select(0, where_valid[0]).to(torch::kCPU).contiguous();
    auto match_scores = mscores0.squeeze(0).index_select(0, where_valid[0]).to(torch::kCPU).contiguous();

    torch::Tensor original0;
    torch::Tensor original1;
    if(indices0.has_value())
        original0 = indices0.value().reshape({-1}).to(torch::kCPU).contiguous();
    if(indices1.has_value())
        original1 = indices1.value().reshape({-1}).to(torch::kCPU).contiguous();

    std::vector<LGMatch> matches;
    matches.reserve(static_cast<size_t>(local0.numel()));
    for(int64_t i = 0; i < local0.numel(); ++i)
    {
        const int64_t i0 = local0.index({i}).item<int64_t>();
        const int64_t i1 = local1.index({i}).item<int64_t>();
        matches.push_back(LGMatch{
            static_cast<int>(ReadIndex(original0, i0)),
            static_cast<int>(ReadIndex(original1, i1)),
            match_scores.index({i}).item<float>()});
    }
    return matches;
}

std::string MapXFeatLighterGlueWeightKey(const std::string& key)
{
    std::string mapped = key;

    StripPrefix(mapped, "module.");
    StripPrefix(mapped, "net.");
    StripPrefix(mapped, "matcher.");

    if(RewriteOldAttentionKey(mapped, "self_attn", "self_attn"))
        return mapped;
    if(RewriteOldAttentionKey(mapped, "cross_attn", "cross_attn"))
        return mapped;

    RewriteModuleListKey(mapped, "transformers");
    RewriteModuleListKey(mapped, "log_assignment");
    RewriteModuleListKey(mapped, "token_confidence");

    return mapped;
}

} // namespace ORB_SLAM3
