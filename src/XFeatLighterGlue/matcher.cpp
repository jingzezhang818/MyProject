#include "XFeatLighterGlue/matcher.hpp"

#include "XFeatLighterGlue/attention.hpp"
#include "XFeatLighterGlue/encoding.hpp"
#include "XFeatLighterGlue/transformer.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include <caffe2/serialize/inline_container.h>
#include <torch/torch.h>

namespace ORB_SLAM3
{
namespace
{

std::string TensorSizesToString(const torch::Tensor& tensor)
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

float ConfidenceThreshold(int layer_index, int n_layers)
{
    const float progress = static_cast<float>(layer_index) / static_cast<float>(n_layers);
    const float threshold = 0.8f + 0.1f * std::exp(-4.0f * progress);
    return std::clamp(threshold, 0.0f, 1.0f);
}

void ValidateInputTensor(const torch::Tensor& tensor,
                         const std::string& name,
                         int64_t cols)
{
    if(!tensor.defined())
        throw std::invalid_argument(name + " is undefined.");
    if(tensor.dim() != 2 || tensor.size(1) != cols)
    {
        throw std::invalid_argument(name + " must have shape [N, " + std::to_string(cols) + "].");
    }
}

void ValidatePreparedTensor(const torch::Tensor& tensor,
                            const std::string& name,
                            int64_t cols)
{
    if(!tensor.defined())
        throw std::invalid_argument(name + " is undefined.");
    if(tensor.dim() != 3 || tensor.size(0) != 1 || tensor.size(2) != cols)
    {
        throw std::invalid_argument(name + " must have shape [1, N, " + std::to_string(cols) + "].");
    }
}

torch::Tensor ToFloatDeviceContiguous(const torch::Tensor& tensor,
                                      const torch::Device& device)
{
    torch::Tensor out = tensor;
    if(out.scalar_type() != torch::kFloat32)
        out = out.to(torch::kFloat32);
    if(out.device() != device)
        out = out.to(device);
    if(!out.is_contiguous())
        out = out.contiguous();
    return out;
}

int PruningThreshold(const torch::Device& device, bool flash)
{
    if(flash)
        return 1536;
    if(device.is_cuda())
        return 1024;
    return -1;
}

bool HasPrefix(const std::string& value, const std::string& prefix)
{
    return value.rfind(prefix, 0) == 0;
}

bool IsPotentialMatcherWeightKey(const std::string& key)
{
    return HasPrefix(key, "matcher.") ||
           HasPrefix(key, "input_proj.") ||
           HasPrefix(key, "posenc.") ||
           HasPrefix(key, "transformers.") ||
           HasPrefix(key, "log_assignment.") ||
           HasPrefix(key, "token_confidence.") ||
           HasPrefix(key, "self_attn.") ||
           HasPrefix(key, "cross_attn.") ||
           key == "confidence_thresholds";
}

bool IsOptionalLocalBuffer(const std::string& key)
{
    return key == "confidence_thresholds";
}

struct PickleValue
{
    enum class Kind
    {
        Other,
        Mark,
        String,
        Global,
        Int,
        Tuple,
        StorageRef,
        TensorMeta
    };

    Kind kind{Kind::Other};
    std::string text;
    std::string text2;
    int64_t int_value{0};
    std::vector<int64_t> ints;
    std::vector<PickleValue> items;

    std::string storage_key;
    std::string storage_type;
    int64_t storage_numel{0};
    int64_t storage_offset{0};
    std::vector<int64_t> size;
    std::vector<int64_t> stride;
};

uint8_t ReadU8(const std::vector<char>& data, size_t& pos)
{
    if(pos >= data.size())
        throw std::runtime_error("Unexpected end of pickle data.");
    return static_cast<uint8_t>(data[pos++]);
}

uint16_t ReadLE16(const std::vector<char>& data, size_t& pos)
{
    if(pos + 2 > data.size())
        throw std::runtime_error("Unexpected end of pickle data.");
    const auto p = reinterpret_cast<const unsigned char*>(data.data() + pos);
    pos += 2;
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

uint32_t ReadLE32(const std::vector<char>& data, size_t& pos)
{
    if(pos + 4 > data.size())
        throw std::runtime_error("Unexpected end of pickle data.");
    const auto p = reinterpret_cast<const unsigned char*>(data.data() + pos);
    pos += 4;
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

std::string ReadBytesAsString(const std::vector<char>& data, size_t& pos, size_t len)
{
    if(pos + len > data.size())
        throw std::runtime_error("Unexpected end of pickle string.");
    std::string out(data.data() + pos, data.data() + pos + len);
    pos += len;
    return out;
}

std::string ReadLine(const std::vector<char>& data, size_t& pos)
{
    const size_t begin = pos;
    while(pos < data.size() && data[pos] != '\n')
        ++pos;
    if(pos >= data.size())
        throw std::runtime_error("Unterminated pickle line.");
    std::string out(data.data() + begin, data.data() + pos);
    ++pos;
    return out;
}

PickleValue MakeOther()
{
    return PickleValue{};
}

PickleValue MakeMark()
{
    PickleValue v;
    v.kind = PickleValue::Kind::Mark;
    return v;
}

PickleValue MakeString(const std::string& s)
{
    PickleValue v;
    v.kind = PickleValue::Kind::String;
    v.text = s;
    return v;
}

PickleValue MakeGlobal(const std::string& module, const std::string& name)
{
    PickleValue v;
    v.kind = PickleValue::Kind::Global;
    v.text = module;
    v.text2 = name;
    return v;
}

PickleValue MakeInt(int64_t value)
{
    PickleValue v;
    v.kind = PickleValue::Kind::Int;
    v.int_value = value;
    return v;
}

PickleValue MakeTuple(std::vector<PickleValue> items)
{
    PickleValue v;
    v.kind = PickleValue::Kind::Tuple;
    v.items = std::move(items);
    v.ints.reserve(v.items.size());
    bool all_ints = true;
    for(const auto& item : v.items)
    {
        if(item.kind != PickleValue::Kind::Int)
        {
            all_ints = false;
            break;
        }
        v.ints.push_back(item.int_value);
    }
    if(!all_ints)
        v.ints.clear();
    return v;
}

std::vector<int64_t> TupleToInts(const PickleValue& value)
{
    if(value.kind == PickleValue::Kind::Tuple)
        return value.ints;
    return {};
}

bool IsGlobal(const PickleValue& value, const std::string& module, const std::string& name)
{
    return value.kind == PickleValue::Kind::Global &&
           value.text == module &&
           value.text2 == name;
}

PickleValue MakeStorageRef(const PickleValue& persistent_id)
{
    if(persistent_id.kind != PickleValue::Kind::Tuple ||
       persistent_id.items.size() < 5 ||
       persistent_id.items[0].kind != PickleValue::Kind::String ||
       persistent_id.items[0].text != "storage" ||
       persistent_id.items[1].kind != PickleValue::Kind::Global ||
       persistent_id.items[2].kind != PickleValue::Kind::String ||
       persistent_id.items[4].kind != PickleValue::Kind::Int)
    {
        return MakeOther();
    }

    PickleValue v;
    v.kind = PickleValue::Kind::StorageRef;
    v.storage_type = persistent_id.items[1].text2;
    v.storage_key = persistent_id.items[2].text;
    v.storage_numel = persistent_id.items[4].int_value;
    return v;
}

PickleValue MakeTensorMeta(const PickleValue& args)
{
    if(args.kind != PickleValue::Kind::Tuple || args.items.size() < 4)
        return MakeOther();
    if(args.items[0].kind != PickleValue::Kind::StorageRef ||
       args.items[1].kind != PickleValue::Kind::Int)
    {
        return MakeOther();
    }

    PickleValue v;
    v.kind = PickleValue::Kind::TensorMeta;
    v.storage_key = args.items[0].storage_key;
    v.storage_type = args.items[0].storage_type;
    v.storage_numel = args.items[0].storage_numel;
    v.storage_offset = args.items[1].int_value;
    v.size = TupleToInts(args.items[2]);
    v.stride = TupleToInts(args.items[3]);
    return v;
}

torch::ScalarType StorageTypeToScalarType(const std::string& storage_type)
{
    if(storage_type == "FloatStorage")
        return torch::kFloat32;
    if(storage_type == "DoubleStorage")
        return torch::kFloat64;
    if(storage_type == "HalfStorage")
        return torch::kFloat16;
    if(storage_type == "LongStorage")
        return torch::kInt64;
    if(storage_type == "IntStorage")
        return torch::kInt32;
    if(storage_type == "ShortStorage")
        return torch::kInt16;
    if(storage_type == "CharStorage")
        return torch::kInt8;
    if(storage_type == "ByteStorage")
        return torch::kUInt8;
    if(storage_type == "BoolStorage")
        return torch::kBool;
    throw std::runtime_error("Unsupported checkpoint storage type: " + storage_type);
}

torch::Tensor TensorFromStorage(caffe2::serialize::PyTorchStreamReader& reader,
                                const PickleValue& meta)
{
    at::DataPtr storage_ptr;
    size_t storage_bytes = 0;
    std::tie(storage_ptr, storage_bytes) = reader.getRecord("data/" + meta.storage_key);

    auto flat = torch::from_blob(storage_ptr.get(),
                                 {meta.storage_numel},
                                 torch::TensorOptions()
                                     .dtype(StorageTypeToScalarType(meta.storage_type))
                                     .device(torch::kCPU));
    return flat.as_strided(meta.size, meta.stride, meta.storage_offset).clone();
}

void PushTupleFromMark(std::vector<PickleValue>& stack)
{
    std::vector<PickleValue> items;
    while(!stack.empty() && stack.back().kind != PickleValue::Kind::Mark)
    {
        items.push_back(stack.back());
        stack.pop_back();
    }
    if(stack.empty())
        throw std::runtime_error("Malformed pickle tuple: missing MARK.");
    stack.pop_back();
    std::reverse(items.begin(), items.end());
    stack.push_back(MakeTuple(std::move(items)));
}

std::vector<std::pair<std::string, torch::Tensor>> LoadPythonCheckpointTensors(
    const std::string& weight_path)
{
    caffe2::serialize::PyTorchStreamReader reader(weight_path);

    at::DataPtr pickle_ptr;
    size_t pickle_size = 0;
    std::tie(pickle_ptr, pickle_size) = reader.getRecord("data.pkl");
    const char* pickle_data = reinterpret_cast<const char*>(pickle_ptr.get());
    std::vector<char> data(pickle_data, pickle_data + pickle_size);

    std::vector<PickleValue> stack;
    std::unordered_map<int64_t, PickleValue> memo;
    std::vector<std::pair<std::string, torch::Tensor>> tensors;

    size_t pos = 0;
    while(pos < data.size())
    {
        const uint8_t op = ReadU8(data, pos);
        switch(op)
        {
        case 0x80: // PROTO
            (void)ReadU8(data, pos);
            break;
        case 0x63: // GLOBAL
        {
            auto module = ReadLine(data, pos);
            auto name = ReadLine(data, pos);
            stack.push_back(MakeGlobal(module, name));
            break;
        }
        case 0x28: // MARK
            stack.push_back(MakeMark());
            break;
        case 0x29: // EMPTY_TUPLE
            stack.push_back(MakeTuple({}));
            break;
        case 0x58: // BINUNICODE
        {
            const auto len = ReadLE32(data, pos);
            stack.push_back(MakeString(ReadBytesAsString(data, pos, len)));
            break;
        }
        case 0x55: // SHORT_BINSTRING
        {
            const auto len = ReadU8(data, pos);
            stack.push_back(MakeString(ReadBytesAsString(data, pos, len)));
            break;
        }
        case 0x54: // BINSTRING
        {
            const auto len = ReadLE32(data, pos);
            stack.push_back(MakeString(ReadBytesAsString(data, pos, len)));
            break;
        }
        case 0x4b: // BININT1
            stack.push_back(MakeInt(ReadU8(data, pos)));
            break;
        case 0x4d: // BININT2
            stack.push_back(MakeInt(ReadLE16(data, pos)));
            break;
        case 0x4a: // BININT
            stack.push_back(MakeInt(static_cast<int32_t>(ReadLE32(data, pos))));
            break;
        case 0x88: // NEWTRUE
            stack.push_back(MakeInt(1));
            break;
        case 0x89: // NEWFALSE
            stack.push_back(MakeInt(0));
            break;
        case 0x4e: // NONE
            stack.push_back(MakeOther());
            break;
        case 0x71: { // BINPUT
            const auto idx = ReadU8(data, pos);
            if(!stack.empty())
                memo[idx] = stack.back();
            break;
        }
        case 0x72: { // LONG_BINPUT
            const auto idx = ReadLE32(data, pos);
            if(!stack.empty())
                memo[idx] = stack.back();
            break;
        }
        case 0x68: { // BINGET
            const auto idx = ReadU8(data, pos);
            auto it = memo.find(idx);
            stack.push_back(it == memo.end() ? MakeOther() : it->second);
            break;
        }
        case 0x6a: { // LONG_BINGET
            const auto idx = ReadLE32(data, pos);
            auto it = memo.find(idx);
            stack.push_back(it == memo.end() ? MakeOther() : it->second);
            break;
        }
        case 0x74: // TUPLE
            PushTupleFromMark(stack);
            break;
        case 0x85: { // TUPLE1
            auto a = stack.back();
            stack.pop_back();
            stack.push_back(MakeTuple({a}));
            break;
        }
        case 0x86: { // TUPLE2
            auto b = stack.back();
            stack.pop_back();
            auto a = stack.back();
            stack.pop_back();
            stack.push_back(MakeTuple({a, b}));
            break;
        }
        case 0x87: { // TUPLE3
            auto c = stack.back();
            stack.pop_back();
            auto b = stack.back();
            stack.pop_back();
            auto a = stack.back();
            stack.pop_back();
            stack.push_back(MakeTuple({a, b, c}));
            break;
        }
        case 0x51: { // BINPERSID
            auto persistent_id = stack.back();
            stack.pop_back();
            stack.push_back(MakeStorageRef(persistent_id));
            break;
        }
        case 0x52: { // REDUCE
            auto args = stack.back();
            stack.pop_back();
            auto callable = stack.back();
            stack.pop_back();

            PickleValue reduced = MakeOther();
            if(IsGlobal(callable, "torch._utils", "_rebuild_tensor_v2"))
                reduced = MakeTensorMeta(args);

            if(reduced.kind == PickleValue::Kind::TensorMeta &&
               !stack.empty() &&
               stack.back().kind == PickleValue::Kind::String)
            {
                tensors.emplace_back(stack.back().text, TensorFromStorage(reader, reduced));
            }
            stack.push_back(std::move(reduced));
            break;
        }
        case 0x7d: // EMPTY_DICT
        case 0x5d: // EMPTY_LIST
            stack.push_back(MakeOther());
            break;
        case 0x75: // SETITEMS
        case 0x73: // SETITEM
        case 0x62: // BUILD
        case 0x65: // APPENDS
        case 0x61: // APPEND
            break;
        case 0x30: // POP
            if(!stack.empty())
                stack.pop_back();
            break;
        case 0x31: // POP_MARK
            while(!stack.empty() && stack.back().kind != PickleValue::Kind::Mark)
                stack.pop_back();
            if(!stack.empty())
                stack.pop_back();
            break;
        case 0x2e: // STOP
            return tensors;
        default:
            throw std::runtime_error("Unsupported pickle opcode while reading checkpoint: " + std::to_string(op));
        }
    }

    return tensors;
}

} // namespace

std::vector<std::pair<std::string, torch::Tensor>> LoadPyTorchCheckpointTensors(
    const std::string& weight_path)
{
    return LoadPythonCheckpointTensors(weight_path);
}

MatchAssignment::MatchAssignment(int dim)
    : dim_(dim)
{
    matchability_ = register_module(
        "matchability",
        torch::nn::Linear(torch::nn::LinearOptions(dim_, 1).bias(true)));
    final_proj_ = register_module(
        "final_proj",
        torch::nn::Linear(torch::nn::LinearOptions(dim_, dim_).bias(true)));
}

torch::Tensor MatchAssignment::sigmoid_log_double_softmax(
    const torch::Tensor& sim,
    const torch::Tensor& z0,
    const torch::Tensor& z1)
{
    const int64_t batch_size = sim.size(0);
    const int64_t m = sim.size(1);
    const int64_t n = sim.size(2);

    auto certainties = torch::log_sigmoid(z0) + torch::log_sigmoid(z1).transpose(1, 2);
    auto scores0 = torch::log_softmax(sim, 2);
    auto scores1 = torch::log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2);

    auto scores = torch::full({batch_size, m + 1, n + 1},
                              0.0,
                              sim.options());

    using torch::indexing::Slice;
    scores.index_put_({Slice(), Slice(0, m), Slice(0, n)}, scores0 + scores1 + certainties);
    scores.index_put_({Slice(), Slice(0, m), n}, torch::log_sigmoid(-z0.squeeze(-1)));
    scores.index_put_({Slice(), m, Slice(0, n)}, torch::log_sigmoid(-z1.squeeze(-1)));
    return scores;
}

torch::Tensor MatchAssignment::forward(const torch::Tensor& desc0,
                                       const torch::Tensor& desc1)
{
    auto mdesc0 = final_proj_->forward(desc0);
    auto mdesc1 = final_proj_->forward(desc1);

    const double scale = 1.0 / std::pow(static_cast<double>(mdesc0.size(-1)), 0.25);
    mdesc0 = mdesc0 * scale;
    mdesc1 = mdesc1 * scale;

    auto sim = torch::matmul(mdesc0, mdesc1.transpose(-2, -1));
    auto z0 = matchability_->forward(desc0);
    auto z1 = matchability_->forward(desc1);
    return sigmoid_log_double_softmax(sim, z0, z1);
}

torch::Tensor MatchAssignment::get_matchability(const torch::Tensor& desc)
{
    return torch::sigmoid(matchability_->forward(desc)).squeeze(-1);
}

XFeatLighterGlue::XFeatLighterGlue(const XFeatLightGlueConfig& config)
    : config_(config), device_(torch::kCPU)
{
    if(config_.name != "xfeat")
        std::cerr << "[XFeatLighterGlue] warning: config.name is '" << config_.name
                  << "', expected 'xfeat'." << std::endl;
    if(config_.input_dim != 64)
        std::cerr << "[XFeatLighterGlue] warning: XFeat input_dim is expected to be 64." << std::endl;
    if(config_.descriptor_dim % config_.num_heads != 0)
        throw std::invalid_argument("descriptor_dim must be divisible by num_heads.");

    if(config_.input_dim != config_.descriptor_dim)
    {
        input_proj_ = register_module(
            "input_proj",
            torch::nn::Linear(torch::nn::LinearOptions(config_.input_dim, config_.descriptor_dim).bias(true)));
    }

    const int head_dim = config_.descriptor_dim / config_.num_heads;
    const int pos_dim = 2 + (config_.add_scale_ori ? 2 : 0);
    posenc_ = register_module(
        "posenc",
        std::make_shared<LearnableFourierPosEnc>(pos_dim, head_dim, head_dim));

    for(int i = 0; i < config_.n_layers; ++i)
    {
        auto layer = std::make_shared<TransformerLayer>(
            config_.descriptor_dim,
            config_.num_heads,
            config_.flash);
        transformers_.push_back(layer);
        register_module("transformers" + std::to_string(i), layer);
    }

    for(int i = 0; i < config_.n_layers; ++i)
    {
        auto assignment = std::make_shared<MatchAssignment>(config_.descriptor_dim);
        log_assignment_.push_back(assignment);
        register_module("log_assignment" + std::to_string(i), assignment);

        if(i < config_.n_layers - 1)
        {
            auto confidence = std::make_shared<TokenConfidence>(config_.descriptor_dim);
            token_confidence_.push_back(confidence);
            register_module("token_confidence" + std::to_string(i), confidence);
        }
    }

    std::vector<float> thresholds;
    thresholds.reserve(static_cast<size_t>(config_.n_layers));
    for(int i = 0; i < config_.n_layers; ++i)
        thresholds.push_back(ConfidenceThreshold(i, config_.n_layers));

    confidence_thresholds_ = register_buffer(
        "confidence_thresholds",
        torch::tensor(thresholds, torch::TensorOptions().dtype(torch::kFloat32)));

    if(!config_.weights.empty())
        LoadWeights(config_.weights);

    eval();
}

void XFeatLighterGlue::To(const torch::Device& device)
{
    device_ = device;
    torch::nn::Module::to(device);
}

torch::Tensor XFeatLighterGlue::get_pruning_mask(
    const torch::optional<torch::Tensor>& confidences,
    const torch::Tensor& scores,
    int layer_index) const
{
    auto keep = scores > (1.0f - config_.width_confidence);
    if(confidences.has_value())
    {
        auto threshold = confidence_thresholds_.index({layer_index}).to(scores.device());
        keep = keep | (confidences.value() <= threshold);
    }
    return keep;
}

bool XFeatLighterGlue::check_if_stop(const torch::Tensor& confidences0,
                                     const torch::Tensor& confidences1,
                                     int layer_index,
                                     int num_points) const
{
    auto confidences = torch::cat({confidences0, confidences1}, -1);
    auto threshold = confidence_thresholds_.index({layer_index}).to(confidences.device());
    const float num_not_confident =
        (confidences < threshold).to(torch::kFloat32).sum().item<float>();
    const float ratio_confident = 1.0f - num_not_confident / static_cast<float>(num_points);
    return ratio_confident > config_.depth_confidence;
}

std::vector<LGMatch> XFeatLighterGlue::Match(
    const torch::Tensor& kpts0,
    const torch::Tensor& desc0,
    const torch::Tensor& size0,
    const torch::Tensor& kpts1,
    const torch::Tensor& desc1,
    const torch::Tensor& size1,
    float filterThreshold)
{
    c10::InferenceMode inference_guard(true);

    ValidateInputTensor(kpts0, "kpts0", 2);
    ValidateInputTensor(kpts1, "kpts1", 2);
    ValidateInputTensor(desc0, "desc0", config_.input_dim);
    ValidateInputTensor(desc1, "desc1", config_.input_dim);
    if(kpts0.size(0) != desc0.size(0))
        throw std::invalid_argument("kpts0 and desc0 must have the same row count.");
    if(kpts1.size(0) != desc1.size(0))
        throw std::invalid_argument("kpts1 and desc1 must have the same row count.");
    if(size0.numel() != 2 || size1.numel() != 2)
        throw std::invalid_argument("size0 and size1 must have shape [2] in [width, height] order.");

    const int64_t m = kpts0.size(0);
    const int64_t n = kpts1.size(0);
    if(m == 0 || n == 0)
        return {};

    auto k0 = ToFloatDeviceContiguous(kpts0, device_).unsqueeze(0);
    auto k1 = ToFloatDeviceContiguous(kpts1, device_).unsqueeze(0);
    auto d0 = ToFloatDeviceContiguous(desc0, device_).unsqueeze(0);
    auto d1 = ToFloatDeviceContiguous(desc1, device_).unsqueeze(0);
    auto s0 = ToFloatDeviceContiguous(size0, device_).view({2});
    auto s1 = ToFloatDeviceContiguous(size1, device_).view({2});

    k0 = normalize_keypoints(k0, s0);
    k1 = normalize_keypoints(k1, s1);

    return MatchPrepared(k0, d0, k1, d1, filterThreshold);
}

std::vector<LGMatch> XFeatLighterGlue::MatchPrepared(
    const torch::Tensor& normalizedKpts0,
    const torch::Tensor& desc0,
    const torch::Tensor& normalizedKpts1,
    const torch::Tensor& desc1,
    float filterThreshold)
{
    c10::InferenceMode inference_guard(true);

    ValidatePreparedTensor(normalizedKpts0, "normalizedKpts0", 2);
    ValidatePreparedTensor(normalizedKpts1, "normalizedKpts1", 2);
    ValidatePreparedTensor(desc0, "desc0", config_.input_dim);
    ValidatePreparedTensor(desc1, "desc1", config_.input_dim);
    if(normalizedKpts0.size(1) != desc0.size(1))
        throw std::invalid_argument("normalizedKpts0 and desc0 must have the same point count.");
    if(normalizedKpts1.size(1) != desc1.size(1))
        throw std::invalid_argument("normalizedKpts1 and desc1 must have the same point count.");

    const int64_t m = normalizedKpts0.size(1);
    const int64_t n = normalizedKpts1.size(1);
    if(m == 0 || n == 0)
        return {};

    auto k0 = ToFloatDeviceContiguous(normalizedKpts0, device_);
    auto k1 = ToFloatDeviceContiguous(normalizedKpts1, device_);
    auto d0 = ToFloatDeviceContiguous(desc0, device_);
    auto d1 = ToFloatDeviceContiguous(desc1, device_);

    if(config_.add_scale_ori)
        throw std::runtime_error("XFeatLighterGlue currently supports add_scale_ori=false only.");

    if(!input_proj_.is_empty())
    {
        d0 = input_proj_->forward(d0);
        d1 = input_proj_->forward(d1);
    }

    auto encoding0 = posenc_->forward(k0);
    auto encoding1 = posenc_->forward(k1);

    auto ind0 = torch::arange(m, torch::TensorOptions().dtype(torch::kLong).device(device_)).unsqueeze(0);
    auto ind1 = torch::arange(n, torch::TensorOptions().dtype(torch::kLong).device(device_)).unsqueeze(0);

    const bool do_early_stop = config_.depth_confidence > 0.0f;
    const bool do_point_pruning = config_.width_confidence > 0.0f;
    const int pruning_threshold = PruningThreshold(device_, config_.flash);

    int last_layer = 0;
    for(int i = 0; i < config_.n_layers; ++i)
    {
        last_layer = i;
        if(d0.size(1) == 0 || d1.size(1) == 0)
            break;

        std::tie(d0, d1) = transformers_[i]->forward(d0, d1, encoding0, encoding1);
        if(i == config_.n_layers - 1)
            continue;

        torch::optional<torch::Tensor> token0 = torch::nullopt;
        torch::optional<torch::Tensor> token1 = torch::nullopt;
        if(do_early_stop)
        {
            torch::Tensor t0, t1;
            std::tie(t0, t1) = token_confidence_[i]->forward(d0, d1);
            token0 = t0;
            token1 = t1;
            if(check_if_stop(t0, t1, i, static_cast<int>(d0.size(1) + d1.size(1))))
                break;
        }

        if(do_point_pruning && d0.size(1) > pruning_threshold)
        {
            auto scores0 = log_assignment_[i]->get_matchability(d0);
            auto mask0 = get_pruning_mask(token0, scores0, i);
            auto keep0 = torch::where(mask0)[1];
            ind0 = ind0.index_select(1, keep0);
            d0 = d0.index_select(1, keep0);
            encoding0 = encoding0.index_select(-2, keep0);
        }

        if(do_point_pruning && d1.size(1) > pruning_threshold)
        {
            auto scores1 = log_assignment_[i]->get_matchability(d1);
            auto mask1 = get_pruning_mask(token1, scores1, i);
            auto keep1 = torch::where(mask1)[1];
            ind1 = ind1.index_select(1, keep1);
            d1 = d1.index_select(1, keep1);
            encoding1 = encoding1.index_select(-2, keep1);
        }
    }

    if(d0.size(1) == 0 || d1.size(1) == 0)
        return {};

    auto scores = log_assignment_[last_layer]->forward(d0, d1);
    const float threshold = filterThreshold >= 0.0f ? filterThreshold : config_.filter_threshold;
    return filter_matches(scores, threshold, ind0, ind1);
}

bool XFeatLighterGlue::LoadWeights(const std::string& weight_path)
{
    if(weight_path.empty() || !std::filesystem::exists(weight_path))
    {
        std::cerr << "[XFeatLighterGlue] warning: weight file not found: "
                  << (weight_path.empty() ? std::string("<empty>") : weight_path)
                  << ". Using random initialization." << std::endl;
        return false;
    }

    std::cout << "[XFeatLighterGlue] loading weights: " << weight_path << std::endl;

    try
    {
        torch::serialize::InputArchive archive;
        archive.load_from(weight_path);
        load(archive);
        to(device_);
        std::cout << "[XFeatLighterGlue] loaded C++ InputArchive weights: "
                  << weight_path << std::endl;
        return true;
    }
    catch(const c10::Error& e)
    {
        std::cerr << "[XFeatLighterGlue] InputArchive load failed, trying Python state_dict fallback: "
                  << e.what_without_backtrace() << std::endl;
    }

    std::vector<std::pair<std::string, torch::Tensor>> archive_tensors;
    try
    {
        archive_tensors = LoadPythonCheckpointTensors(weight_path);
    }
    catch(const std::exception& e)
    {
        std::cerr << "[XFeatLighterGlue] Python checkpoint parser failed, trying pickle_load fallback: "
                  << e.what() << std::endl;
    }

    if(!archive_tensors.empty())
        return LoadNamedTensors(archive_tensors);

    try
    {
        auto bytes = ReadFileBytes(weight_path);
        auto ivalue = torch::pickle_load(bytes);
        if(ivalue.isGenericDict())
        {
            auto dict = ivalue.toGenericDict();
            for(const auto& item : dict)
            {
                if(item.key().isString() &&
                   item.key().toStringRef() == "state_dict" &&
                   item.value().isGenericDict())
                {
                    auto nested = item.value().toGenericDict();
                    return LoadStateDict(nested);
                }
            }
            return LoadStateDict(dict);
        }
    }
    catch(const c10::Error& e)
    {
        std::cerr << "[XFeatLighterGlue] pickle_load fallback failed: "
                  << e.what_without_backtrace() << std::endl;
    }

    throw std::runtime_error(
        "XFeatLighterGlue failed to load weights as InputArchive tensors or GenericDict state_dict: " +
        weight_path);
}

std::vector<char> XFeatLighterGlue::ReadFileBytes(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if(!file)
        throw std::runtime_error("Failed to open weight file: " + filename);

    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(static_cast<size_t>(size));
    if(size > 0 && !file.read(buffer.data(), size))
        throw std::runtime_error("Failed to read weight file: " + filename);
    return buffer;
}

bool XFeatLighterGlue::LoadStateDict(const c10::Dict<c10::IValue, c10::IValue>& weights)
{
    std::vector<std::pair<std::string, torch::Tensor>> named_tensors;
    named_tensors.reserve(weights.size());

    for(const auto& w : weights)
    {
        if(w.key().isString() && w.value().isTensor())
            named_tensors.emplace_back(std::string(w.key().toStringRef()), w.value().toTensor());
        else if(w.key().isString())
        {
            std::cout << "[XFeatLighterGlue] unexpected key: "
                      << std::string(w.key().toStringRef())
                      << " (value is not a tensor)" << std::endl;
        }
        else
        {
            std::cout << "[XFeatLighterGlue] unexpected key: <non-string>" << std::endl;
        }
    }

    return LoadNamedTensors(named_tensors);
}

bool XFeatLighterGlue::LoadNamedTensors(
    const std::vector<std::pair<std::string, torch::Tensor>>& weights)
{
    std::unordered_map<std::string, torch::Tensor> param_map;
    std::unordered_map<std::string, torch::Tensor> buffer_map;
    std::unordered_set<std::string> seen;

    for(const auto& p : named_parameters())
        param_map.emplace(p.key(), p.value());
    for(const auto& b : named_buffers())
        buffer_map.emplace(b.key(), b.value());

    torch::NoGradGuard no_grad;

    size_t loaded_count = 0;
    size_t unexpected_count = 0;
    size_t mismatch_count = 0;
    size_t skipped_count = 0;

    for(const auto& w : weights)
    {
        const std::string& loaded_key = w.first;
        if(!IsPotentialMatcherWeightKey(loaded_key))
        {
            std::cout << "[XFeatLighterGlue] skipped non-matcher key: " << loaded_key << std::endl;
            ++skipped_count;
            continue;
        }

        const std::string mapped_key = MapXFeatLighterGlueWeightKey(loaded_key);
        std::cout << "[XFeatLighterGlue] loaded key: " << loaded_key << std::endl;
        std::cout << "[XFeatLighterGlue] mapped key: " << loaded_key
                  << " -> " << mapped_key << std::endl;

        const auto& source = w.second;
        auto pit = param_map.find(mapped_key);
        if(pit != param_map.end())
        {
            if(pit->second.sizes() != source.sizes())
            {
                ++mismatch_count;
                std::ostringstream oss;
                oss << "[XFeatLighterGlue] shape mismatch: " << loaded_key
                    << " -> " << mapped_key
                    << " expected " << TensorSizesToString(pit->second)
                    << " got " << TensorSizesToString(source);
                std::cerr << oss.str() << std::endl;
                throw std::runtime_error(oss.str());
            }
            pit->second.copy_(source.to(pit->second.scalar_type()).to(pit->second.device()));
            seen.insert(mapped_key);
            ++loaded_count;
            continue;
        }

        auto bit = buffer_map.find(mapped_key);
        if(bit != buffer_map.end())
        {
            if(bit->second.sizes() != source.sizes())
            {
                ++mismatch_count;
                std::ostringstream oss;
                oss << "[XFeatLighterGlue] shape mismatch: " << loaded_key
                    << " -> " << mapped_key
                    << " expected " << TensorSizesToString(bit->second)
                    << " got " << TensorSizesToString(source);
                std::cerr << oss.str() << std::endl;
                throw std::runtime_error(oss.str());
            }
            bit->second.copy_(source.to(bit->second.scalar_type()).to(bit->second.device()));
            seen.insert(mapped_key);
            ++loaded_count;
            continue;
        }

        std::cout << "[XFeatLighterGlue] unexpected key: " << loaded_key
                  << " mapped to " << mapped_key << std::endl;
        ++unexpected_count;
    }

    size_t missing_count = 0;
    for(const auto& p : param_map)
    {
        if(seen.find(p.first) == seen.end())
        {
            std::cout << "[XFeatLighterGlue] missing key: " << p.first << std::endl;
            ++missing_count;
        }
    }
    for(const auto& b : buffer_map)
    {
        if(seen.find(b.first) == seen.end() && !IsOptionalLocalBuffer(b.first))
        {
            std::cout << "[XFeatLighterGlue] missing key: " << b.first << std::endl;
            ++missing_count;
        }
    }

    std::cout << "[XFeatLighterGlue] load report: loaded=" << loaded_count
              << " missing=" << missing_count
              << " unexpected=" << unexpected_count
              << " skipped=" << skipped_count
              << " shape_mismatch=" << mismatch_count << std::endl;
    return mismatch_count == 0;
}

} // namespace ORB_SLAM3
