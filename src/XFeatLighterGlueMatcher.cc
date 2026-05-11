#include "XFeatLighterGlueMatcher.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ORB_SLAM3
{
namespace
{

std::string DefaultWeightPath()
{
    const char* env = std::getenv("XFEAT_LIGHTGLUE_WEIGHT");
    if(env && std::string(env).size() > 0)
        return std::string(env);
    return "./weights/xfeat_lighterglue_matcher_cpp.pt";
}

struct Candidate
{
    int idxCur = -1;
    MapPoint* pMP = nullptr;
    float score = 0.0f;
};

bool IsEnvFlagEnabled(const char* key)
{
    const char* env = std::getenv(key);
    if(!env)
        return false;

    const std::string v(env);
    return !(v.empty() || v == "0" || v == "false" || v == "FALSE");
}

float GetEnvFloatWithRange(const char* key, const float fallback, const float minValue, const float maxValue)
{
    const char* env = std::getenv(key);
    if(!env || env[0] == '\0')
        return fallback;

    char* end = nullptr;
    const float parsed = std::strtof(env, &end);
    if(end == env || !std::isfinite(parsed))
        return fallback;

    return std::max(minValue, std::min(maxValue, parsed));
}

bool PassMotionProjectionGate(MapPoint* pMP,
                              const Frame& CurrentFrame,
                              const int idxCur,
                              const float radius)
{
    if(!pMP || pMP->isBad() || !CurrentFrame.mpCamera)
        return false;
    if(idxCur < 0 || idxCur >= static_cast<int>(CurrentFrame.mvKeysUn.size()))
        return false;

    const Eigen::Vector3f x3Dc = CurrentFrame.GetPose() * pMP->GetWorldPos();
    if(x3Dc(2) <= 0.0f)
        return false;

    const Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dc);
    if(uv(0) < CurrentFrame.mnMinX || uv(0) > CurrentFrame.mnMaxX ||
       uv(1) < CurrentFrame.mnMinY || uv(1) > CurrentFrame.mnMaxY)
        return false;

    const cv::Point2f& kp = CurrentFrame.mvKeysUn[idxCur].pt;
    const float dx = uv(0) - kp.x;
    const float dy = uv(1) - kp.y;
    return dx * dx + dy * dy <= radius * radius;
}

int AssignBestUniqueCandidates(const std::vector<Candidate>& candidates,
                               const int nCur,
                               std::vector<MapPoint*>& vpMapPointMatches)
{
    std::vector<int> bestForCur(static_cast<size_t>(std::max(0, nCur)), -1);
    std::unordered_map<long unsigned int, int> bestForMapPoint;
    bestForMapPoint.reserve(candidates.size());

    for(int i = 0; i < static_cast<int>(candidates.size()); ++i)
    {
        const Candidate& c = candidates[i];
        if(c.idxCur < 0 || c.idxCur >= nCur || !c.pMP || c.pMP->isBad())
            continue;

        int& curBest = bestForCur[static_cast<size_t>(c.idxCur)];
        if(curBest < 0 || c.score > candidates[curBest].score)
            curBest = i;

        auto it = bestForMapPoint.find(c.pMP->mnId);
        if(it == bestForMapPoint.end() || c.score > candidates[it->second].score)
            bestForMapPoint[c.pMP->mnId] = i;
    }

    int assigned = 0;
    for(int idxCur = 0; idxCur < nCur; ++idxCur)
    {
        const int candIdx = bestForCur[static_cast<size_t>(idxCur)];
        if(candIdx < 0)
            continue;

        const Candidate& c = candidates[candIdx];
        const auto it = bestForMapPoint.find(c.pMP->mnId);
        if(it == bestForMapPoint.end() || it->second != candIdx)
            continue;

        vpMapPointMatches[idxCur] = c.pMP;
        ++assigned;
    }

    return assigned;
}

} // namespace

XFeatLighterGlueMatcher::XFeatLighterGlueMatcher(const std::string& weightPath)
    : device_(SelectDevice()), matcher_(XFeatLightGlueConfig())
{
    matcher_.To(device_);
    matcher_.LoadWeights(weightPath.empty() ? DefaultWeightPath() : weightPath);
}

torch::Device XFeatLighterGlueMatcher::SelectDevice()
{
    std::string requested = "auto";
    if(const char* env = std::getenv("XFEAT_DEVICE"))
        requested = env;
    std::transform(requested.begin(), requested.end(), requested.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    int cudaDeviceIdx = 0;
    if(const char* envDevIdx = std::getenv("XFEAT_CUDA_DEVICE"))
        cudaDeviceIdx = std::max(0, std::atoi(envDevIdx));

    if(requested == "cpu")
        return torch::Device(torch::kCPU);

    if(requested == "cuda" || requested == "gpu" || requested == "auto")
    {
        if(torch::cuda::is_available())
            return torch::Device(torch::kCUDA, cudaDeviceIdx);
        if(requested == "cuda" || requested == "gpu")
            throw std::runtime_error("XFEAT_DEVICE=cuda requested, but CUDA is not available.");
        return torch::Device(torch::kCPU);
    }

    throw std::runtime_error("XFEAT_DEVICE must be auto, cuda, gpu, or cpu.");
}

torch::Tensor XFeatLighterGlueMatcher::KeyPointsToTensor(
    const std::vector<cv::KeyPoint>& keypoints,
    int maxRows,
    const torch::Device& device)
{
    const int rows = std::max(0, std::min(maxRows, static_cast<int>(keypoints.size())));
    if(rows == 0)
        return torch::empty({0, 2}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    std::vector<float> data;
    data.reserve(static_cast<size_t>(rows) * 2);
    for(int i = 0; i < rows; ++i)
    {
        data.push_back(keypoints[i].pt.x);
        data.push_back(keypoints[i].pt.y);
    }

    return torch::from_blob(data.data(),
                            {rows, 2},
                            torch::TensorOptions().dtype(torch::kFloat32))
        .clone()
        .to(device);
}

torch::Tensor XFeatLighterGlueMatcher::DescriptorsToTensor(const cv::Mat& descriptors,
                                                           int maxRows,
                                                           const torch::Device& device)
{
    if(descriptors.empty())
        return torch::empty({0, 64}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    if(descriptors.type() != CV_32F)
        throw std::runtime_error("XFeatLighterGlueMatcher expects CV_32F descriptors.");
    if(descriptors.cols != 64)
        throw std::runtime_error("XFeatLighterGlueMatcher expects 64-D XFeat descriptors.");

    const int rows = std::max(0, std::min(maxRows, descriptors.rows));
    if(rows == 0)
        return torch::empty({0, 64}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    cv::Mat selected = descriptors.rowRange(0, rows);
    cv::Mat continuous = selected.isContinuous() ? selected : selected.clone();
    auto tensor = torch::from_blob(continuous.ptr<float>(),
                                   {continuous.rows, continuous.cols},
                                   torch::TensorOptions().dtype(torch::kFloat32))
                      .clone()
                      .to(device);
    return tensor / tensor.pow(2).sum(1, true).sqrt().clamp_min(1e-12);
}

torch::Tensor XFeatLighterGlueMatcher::ImageSizeToTensor(float width,
                                                         float height,
                                                         const torch::Device& device)
{
    width = std::max(width, 1.0f);
    height = std::max(height, 1.0f);
    return torch::tensor({width, height},
                         torch::TensorOptions().dtype(torch::kFloat32).device(device));
}

int XFeatLighterGlueMatcher::SearchByLightGlue(KeyFrame* pKF,
                                               Frame& F,
                                               std::vector<MapPoint*>& vpMapPointMatches,
                                               float minConf)
{
    lastStats_ = Stats{};
    vpMapPointMatches.assign(F.N, static_cast<MapPoint*>(nullptr));

    if(!pKF || F.N <= 0 || pKF->N <= 0 || F.mDescriptors.empty() || pKF->mDescriptors.empty())
        return 0;

    lastStats_.frame_id = F.mnId;
    lastStats_.ref_kf_id = pKF->mnId;

    const int nRef = std::min({pKF->N,
                               static_cast<int>(pKF->mvKeysUn.size()),
                               pKF->mDescriptors.rows});
    const int nCur = std::min({F.N,
                               static_cast<int>(F.mvKeysUn.size()),
                               F.mDescriptors.rows});
    lastStats_.N_ref = nRef;
    lastStats_.N_cur = nCur;

    if(nRef <= 0 || nCur <= 0)
        return 0;

    auto kptsRef = KeyPointsToTensor(pKF->mvKeysUn, nRef, device_);
    auto kptsCur = KeyPointsToTensor(F.mvKeysUn, nCur, device_);
    auto descRef = DescriptorsToTensor(pKF->mDescriptors, nRef, device_);
    auto descCur = DescriptorsToTensor(F.mDescriptors, nCur, device_);

    const float refWidth = static_cast<float>(std::max(1, pKF->mnMaxX));
    const float refHeight = static_cast<float>(std::max(1, pKF->mnMaxY));
    const float curWidth = std::max(1.0f, Frame::mnMaxX);
    const float curHeight = std::max(1.0f, Frame::mnMaxY);
    auto sizeRef = ImageSizeToTensor(refWidth, refHeight, device_);
    auto sizeCur = ImageSizeToTensor(curWidth, curHeight, device_);

    const auto matches = matcher_.Match(kptsRef, descRef, sizeRef,
                                        kptsCur, descCur, sizeCur,
                                        minConf);
    lastStats_.lg_raw_matches = static_cast<int>(matches.size());

    const std::vector<MapPoint*> refMapPoints = pKF->GetMapPointMatches();
    std::vector<Candidate> candidates;
    candidates.reserve(matches.size());
    for(const auto& m : matches)
    {
        if(m.idx0 < 0 || m.idx0 >= nRef || m.idx1 < 0 || m.idx1 >= nCur)
            continue;
        if(m.idx0 >= static_cast<int>(refMapPoints.size()))
            continue;
        MapPoint* pMP = refMapPoints[m.idx0];
        if(!pMP || pMP->isBad())
            continue;
        candidates.push_back(Candidate{m.idx1, pMP, m.score});
    }
    lastStats_.mp_valid = static_cast<int>(candidates.size());
    lastStats_.proj_gate_keep = lastStats_.mp_valid;

    const int assigned = AssignBestUniqueCandidates(candidates, F.N, vpMapPointMatches);
    lastStats_.one_to_one = assigned;
    return assigned;
}

int XFeatLighterGlueMatcher::SearchByLightGlue(const Frame& LastFrame,
                                               Frame& CurrentFrame,
                                               std::vector<MapPoint*>& vpMapPointMatches,
                                               float minConf)
{
    lastStats_ = Stats{};
    vpMapPointMatches.assign(CurrentFrame.N, static_cast<MapPoint*>(nullptr));

    if(CurrentFrame.N <= 0 || LastFrame.N <= 0 ||
       CurrentFrame.mDescriptors.empty() || LastFrame.mDescriptors.empty())
        return 0;

    lastStats_.frame_id = CurrentFrame.mnId;
    lastStats_.last_frame_id = LastFrame.mnId;

    const int nLast = std::min({LastFrame.N,
                                static_cast<int>(LastFrame.mvKeysUn.size()),
                                LastFrame.mDescriptors.rows,
                                static_cast<int>(LastFrame.mvpMapPoints.size())});
    const int nCur = std::min({CurrentFrame.N,
                               static_cast<int>(CurrentFrame.mvKeysUn.size()),
                               CurrentFrame.mDescriptors.rows});
    lastStats_.N_last = nLast;
    lastStats_.N_cur = nCur;

    if(nLast <= 0 || nCur <= 0)
        return 0;

    auto kptsLast = KeyPointsToTensor(LastFrame.mvKeysUn, nLast, device_);
    auto kptsCur = KeyPointsToTensor(CurrentFrame.mvKeysUn, nCur, device_);
    auto descLast = DescriptorsToTensor(LastFrame.mDescriptors, nLast, device_);
    auto descCur = DescriptorsToTensor(CurrentFrame.mDescriptors, nCur, device_);

    const float lastWidth = std::max(1.0f, LastFrame.mnMaxX);
    const float lastHeight = std::max(1.0f, LastFrame.mnMaxY);
    const float curWidth = std::max(1.0f, CurrentFrame.mnMaxX);
    const float curHeight = std::max(1.0f, CurrentFrame.mnMaxY);
    auto sizeLast = ImageSizeToTensor(lastWidth, lastHeight, device_);
    auto sizeCur = ImageSizeToTensor(curWidth, curHeight, device_);

    const auto matches = matcher_.Match(kptsLast, descLast, sizeLast,
                                        kptsCur, descCur, sizeCur,
                                        minConf);
    lastStats_.lg_raw_matches = static_cast<int>(matches.size());

    const bool useProjGate = IsEnvFlagEnabled("XFEAT_LG_MOTION_USE_PROJ_GATE");
    const float projGateRadius = GetEnvFloatWithRange("XFEAT_LG_MOTION_PROJ_RADIUS", 10.0f, 0.5f, 500.0f);

    std::vector<Candidate> candidates;
    candidates.reserve(matches.size());
    for(const auto& m : matches)
    {
        if(m.idx0 < 0 || m.idx0 >= nLast || m.idx1 < 0 || m.idx1 >= nCur)
            continue;

        MapPoint* pMP = LastFrame.mvpMapPoints[m.idx0];
        if(!pMP || pMP->isBad())
            continue;

        ++lastStats_.mp_valid;
        if(useProjGate && !PassMotionProjectionGate(pMP, CurrentFrame, m.idx1, projGateRadius))
            continue;

        candidates.push_back(Candidate{m.idx1, pMP, m.score});
    }
    lastStats_.proj_gate_keep = static_cast<int>(candidates.size());

    const int assigned = AssignBestUniqueCandidates(candidates, CurrentFrame.N, vpMapPointMatches);
    lastStats_.one_to_one = assigned;
    return assigned;
}

} // namespace ORB_SLAM3
