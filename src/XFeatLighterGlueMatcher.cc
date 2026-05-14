#include "XFeatLighterGlueMatcher.h"

#include "XFeatLighterGlue/core.hpp"

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

struct TriangulationCandidate
{
    int idx1 = -1;
    int idx2 = -1;
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
                               std::vector<MapPoint*>& vpMapPointMatches,
                               std::vector<float>* vMatchScores)
{
    if(vMatchScores)
        vMatchScores->assign(static_cast<size_t>(std::max(0, nCur)), -1.0f);

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
        if(vMatchScores)
            (*vMatchScores)[static_cast<size_t>(idxCur)] = c.score;
        ++assigned;
    }

    return assigned;
}

int AssignBestUniqueTriangulationPairs(const std::vector<TriangulationCandidate>& candidates,
                                       const int n1,
                                       const int n2,
                                       std::vector<std::pair<size_t, size_t>>& vMatchedPairs)
{
    std::vector<TriangulationCandidate> sorted = candidates;
    std::sort(sorted.begin(), sorted.end(), [](const TriangulationCandidate& a,
                                               const TriangulationCandidate& b) {
        return a.score > b.score;
    });

    std::vector<char> usedIdx1(static_cast<size_t>(std::max(0, n1)), 0);
    std::vector<char> usedIdx2(static_cast<size_t>(std::max(0, n2)), 0);
    vMatchedPairs.clear();
    vMatchedPairs.reserve(sorted.size());

    for(const TriangulationCandidate& c : sorted)
    {
        if(c.idx1 < 0 || c.idx1 >= n1 || c.idx2 < 0 || c.idx2 >= n2)
            continue;
        if(usedIdx1[static_cast<size_t>(c.idx1)] || usedIdx2[static_cast<size_t>(c.idx2)])
            continue;

        usedIdx1[static_cast<size_t>(c.idx1)] = 1;
        usedIdx2[static_cast<size_t>(c.idx2)] = 1;
        vMatchedPairs.emplace_back(static_cast<size_t>(c.idx1), static_cast<size_t>(c.idx2));
    }

    return static_cast<int>(vMatchedPairs.size());
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

int XFeatLighterGlueMatcher::GetKeyFrameCacheLimit()
{
    return 64;
}

bool XFeatLighterGlueMatcher::IsCacheHit(const CachedInputTensors& cached,
                                         const long unsigned int id,
                                         const int rows,
                                         const cv::Mat& descriptors,
                                         const float width,
                                         const float height)
{
    return cached.valid &&
           cached.id == id &&
           cached.rows == rows &&
           cached.descRows == descriptors.rows &&
           cached.descCols == descriptors.cols &&
           cached.descData == descriptors.data &&
           cached.width == width &&
           cached.height == height;
}

XFeatLighterGlueMatcher::CachedInputTensors XFeatLighterGlueMatcher::BuildInputTensors(
    const long unsigned int id,
    const std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& descriptors,
    const int rows,
    const float width,
    const float height,
    const torch::Device& device)
{
    CachedInputTensors input;
    input.valid = true;
    input.id = id;
    input.rows = rows;
    input.descRows = descriptors.rows;
    input.descCols = descriptors.cols;
    input.descData = descriptors.data;
    input.width = width;
    input.height = height;
    input.size = ImageSizeToTensor(width, height, device);
    input.kpts = normalize_keypoints(KeyPointsToTensor(keypoints, rows, device).unsqueeze(0),
                                     input.size);
    input.desc = DescriptorsToTensor(descriptors, rows, device).unsqueeze(0);
    return input;
}

const XFeatLighterGlueMatcher::CachedInputTensors& XFeatLighterGlueMatcher::GetKeyFrameInputTensors(
    const KeyFrame& keyFrame,
    const int rows,
    const float width,
    const float height)
{
    auto it = keyFrameInputCache_.find(keyFrame.mnId);
    if(it != keyFrameInputCache_.end() &&
       IsCacheHit(it->second, keyFrame.mnId, rows, keyFrame.mDescriptors, width, height))
    {
        return it->second;
    }

    const bool inserted = it == keyFrameInputCache_.end();
    CachedInputTensors& cached = keyFrameInputCache_[keyFrame.mnId];
    cached = BuildInputTensors(keyFrame.mnId,
                               keyFrame.mvKeysUn,
                               keyFrame.mDescriptors,
                               rows,
                               width,
                               height,
                               device_);
    if(inserted)
        keyFrameCacheOrder_.push_back(keyFrame.mnId);
    TrimKeyFrameCache();
    return keyFrameInputCache_.at(keyFrame.mnId);
}

const XFeatLighterGlueMatcher::CachedInputTensors& XFeatLighterGlueMatcher::GetFrameInputTensors(
    const Frame& frame,
    const int rows,
    const float width,
    const float height)
{
    auto it = frameInputCache_.find(frame.mnId);
    if(it != frameInputCache_.end() &&
       IsCacheHit(it->second, frame.mnId, rows, frame.mDescriptors, width, height))
    {
        return it->second;
    }

    const bool inserted = it == frameInputCache_.end();
    CachedInputTensors& cached = frameInputCache_[frame.mnId];
    cached = BuildInputTensors(frame.mnId,
                               frame.mvKeysUn,
                               frame.mDescriptors,
                               rows,
                               width,
                               height,
                               device_);
    if(inserted)
        frameCacheOrder_.push_back(frame.mnId);
    TrimFrameCache();
    return frameInputCache_.at(frame.mnId);
}

void XFeatLighterGlueMatcher::TrimKeyFrameCache()
{
    const size_t limit = static_cast<size_t>(GetKeyFrameCacheLimit());
    while(keyFrameInputCache_.size() > limit && !keyFrameCacheOrder_.empty())
    {
        const long unsigned int id = keyFrameCacheOrder_.front();
        keyFrameCacheOrder_.erase(keyFrameCacheOrder_.begin());
        keyFrameInputCache_.erase(id);
    }
}

void XFeatLighterGlueMatcher::TrimFrameCache()
{
    constexpr size_t kMaxFrameCacheEntries = 4;
    while(frameInputCache_.size() > kMaxFrameCacheEntries && !frameCacheOrder_.empty())
    {
        const long unsigned int id = frameCacheOrder_.front();
        frameCacheOrder_.erase(frameCacheOrder_.begin());
        frameInputCache_.erase(id);
    }
}

int XFeatLighterGlueMatcher::SearchByLightGlue(KeyFrame* pKF,
                                               Frame& F,
                                               std::vector<MapPoint*>& vpMapPointMatches,
                                               float minConf,
                                               std::vector<float>* vMatchScores)
{
    lastStats_ = Stats{};
    vpMapPointMatches.assign(F.N, static_cast<MapPoint*>(nullptr));
    if(vMatchScores)
        vMatchScores->assign(static_cast<size_t>(std::max(0, F.N)), -1.0f);

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

    const float refWidth = static_cast<float>(std::max(1, pKF->mnMaxX));
    const float refHeight = static_cast<float>(std::max(1, pKF->mnMaxY));
    const float curWidth = std::max(1.0f, Frame::mnMaxX);
    const float curHeight = std::max(1.0f, Frame::mnMaxY);
    const CachedInputTensors& refInput = GetKeyFrameInputTensors(*pKF, nRef, refWidth, refHeight);
    const CachedInputTensors& curInput = GetFrameInputTensors(F, nCur, curWidth, curHeight);

    const auto matches = matcher_.MatchPrepared(refInput.kpts, refInput.desc,
                                                curInput.kpts, curInput.desc,
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

    const int assigned = AssignBestUniqueCandidates(candidates, F.N, vpMapPointMatches, vMatchScores);
    lastStats_.one_to_one = assigned;
    return assigned;
}

int XFeatLighterGlueMatcher::SearchByLightGlue(const Frame& LastFrame,
                                               Frame& CurrentFrame,
                                               std::vector<MapPoint*>& vpMapPointMatches,
                                               float minConf,
                                               std::vector<float>* vMatchScores)
{
    lastStats_ = Stats{};
    vpMapPointMatches.assign(CurrentFrame.N, static_cast<MapPoint*>(nullptr));
    if(vMatchScores)
        vMatchScores->assign(static_cast<size_t>(std::max(0, CurrentFrame.N)), -1.0f);

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

    const float lastWidth = std::max(1.0f, LastFrame.mnMaxX);
    const float lastHeight = std::max(1.0f, LastFrame.mnMaxY);
    const float curWidth = std::max(1.0f, CurrentFrame.mnMaxX);
    const float curHeight = std::max(1.0f, CurrentFrame.mnMaxY);
    const CachedInputTensors& lastInput = GetFrameInputTensors(LastFrame, nLast, lastWidth, lastHeight);
    const torch::Tensor kptsLastCached = lastInput.kpts;
    const torch::Tensor descLastCached = lastInput.desc;
    const CachedInputTensors& curInput = GetFrameInputTensors(CurrentFrame, nCur, curWidth, curHeight);

    const auto matches = matcher_.MatchPrepared(kptsLastCached, descLastCached,
                                                curInput.kpts, curInput.desc,
                                                minConf);
    lastStats_.lg_raw_matches = static_cast<int>(matches.size());

    const bool useProjGate = IsEnvFlagEnabled("XFEAT_LG_MOTION_USE_PROJ_GATE");
    constexpr float kMotionProjectionGateRadius = 10.0f;

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
        if(useProjGate && !PassMotionProjectionGate(pMP, CurrentFrame, m.idx1, kMotionProjectionGateRadius))
            continue;

        candidates.push_back(Candidate{m.idx1, pMP, m.score});
    }
    lastStats_.proj_gate_keep = static_cast<int>(candidates.size());

    const int assigned = AssignBestUniqueCandidates(candidates, CurrentFrame.N, vpMapPointMatches, vMatchScores);
    lastStats_.one_to_one = assigned;
    return assigned;
}

int XFeatLighterGlueMatcher::SearchForTriangulation(KeyFrame* pKF1,
                                                    KeyFrame* pKF2,
                                                    std::vector<std::pair<size_t, size_t>>& vMatchedPairs,
                                                    bool bOnlyStereo,
                                                    bool bCoarse,
                                                    float minConf)
{
    lastStats_ = Stats{};
    vMatchedPairs.clear();
    if(!pKF1 || !pKF2 || pKF1->mDescriptors.empty() || pKF2->mDescriptors.empty())
        return 0;

    lastStats_.ref_kf_id = pKF1->mnId;
    lastStats_.last_frame_id = pKF2->mnId;

    const int n1 = std::min({pKF1->N,
                             static_cast<int>(pKF1->mvKeysUn.size()),
                             pKF1->mDescriptors.rows});
    const int n2 = std::min({pKF2->N,
                             static_cast<int>(pKF2->mvKeysUn.size()),
                             pKF2->mDescriptors.rows});
    lastStats_.N_ref = n1;
    lastStats_.N_cur = n2;
    if(n1 <= 0 || n2 <= 0)
        return 0;

    const float width1 = static_cast<float>(std::max(1, pKF1->mnMaxX));
    const float height1 = static_cast<float>(std::max(1, pKF1->mnMaxY));
    const float width2 = static_cast<float>(std::max(1, pKF2->mnMaxX));
    const float height2 = static_cast<float>(std::max(1, pKF2->mnMaxY));
    const CachedInputTensors& input1 = GetKeyFrameInputTensors(*pKF1, n1, width1, height1);
    const CachedInputTensors& input2 = GetKeyFrameInputTensors(*pKF2, n2, width2, height2);

    const auto matches = matcher_.MatchPrepared(input1.kpts, input1.desc,
                                                input2.kpts, input2.desc,
                                                minConf);
    lastStats_.lg_raw_matches = static_cast<int>(matches.size());

    Sophus::SE3f T1w = pKF1->GetPose();
    Sophus::SE3f T2w = pKF2->GetPose();
    Sophus::SE3f Tw2 = pKF2->GetPoseInverse();
    const Eigen::Vector3f Cw = pKF1->GetCameraCenter();
    const Eigen::Vector3f C2 = T2w * Cw;
    const Eigen::Vector2f ep = pKF2->mpCamera->project(C2);

    Sophus::SE3f T12;
    Sophus::SE3f Tll, Tlr, Trl, Trr;
    Eigen::Matrix3f R12 = Eigen::Matrix3f::Identity();
    Eigen::Vector3f t12 = Eigen::Vector3f::Zero();
    GeometricCamera* pCamera1 = pKF1->mpCamera;
    GeometricCamera* pCamera2 = pKF2->mpCamera;

    if(!pKF1->mpCamera2 && !pKF2->mpCamera2)
    {
        T12 = T1w * Tw2;
        R12 = T12.rotationMatrix();
        t12 = T12.translation();
    }
    else
    {
        const Sophus::SE3f Tr1w = pKF1->GetRightPose();
        const Sophus::SE3f Twr2 = pKF2->GetRightPoseInverse();
        Tll = T1w * Tw2;
        Tlr = T1w * Twr2;
        Trl = Tr1w * Tw2;
        Trr = Tr1w * Twr2;
    }

    Eigen::Matrix3f Rll = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f Rlr = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f Rrl = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f Rrr = Eigen::Matrix3f::Identity();
    Eigen::Vector3f tll = Eigen::Vector3f::Zero();
    Eigen::Vector3f tlr = Eigen::Vector3f::Zero();
    Eigen::Vector3f trl = Eigen::Vector3f::Zero();
    Eigen::Vector3f trr = Eigen::Vector3f::Zero();

    if(pKF1->mpCamera2 && pKF2->mpCamera2)
    {
        Rll = Tll.rotationMatrix();
        Rlr = Tlr.rotationMatrix();
        Rrl = Trl.rotationMatrix();
        Rrr = Trr.rotationMatrix();
        tll = Tll.translation();
        tlr = Tlr.translation();
        trl = Trl.translation();
        trr = Trr.translation();
    }

    std::vector<TriangulationCandidate> candidates;
    candidates.reserve(matches.size());
    for(const LGMatch& m : matches)
    {
        const int idx1 = m.idx0;
        const int idx2 = m.idx1;
        if(idx1 < 0 || idx1 >= n1 || idx2 < 0 || idx2 >= n2)
            continue;
        if(pKF1->GetMapPoint(idx1) || pKF2->GetMapPoint(idx2))
            continue;

        const cv::KeyPoint& kp1 = (pKF1->NLeft == -1) ? pKF1->mvKeysUn[idx1]
                                                      : (idx1 < pKF1->NLeft) ? pKF1->mvKeys[idx1]
                                                                             : pKF1->mvKeysRight[idx1 - pKF1->NLeft];
        const cv::KeyPoint& kp2 = (pKF2->NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                      : (idx2 < pKF2->NLeft) ? pKF2->mvKeys[idx2]
                                                                             : pKF2->mvKeysRight[idx2 - pKF2->NLeft];
        const bool bStereo1 = (!pKF1->mpCamera2 &&
                               idx1 < static_cast<int>(pKF1->mvuRight.size()) &&
                               pKF1->mvuRight[idx1] >= 0);
        const bool bStereo2 = (!pKF2->mpCamera2 &&
                               idx2 < static_cast<int>(pKF2->mvuRight.size()) &&
                               pKF2->mvuRight[idx2] >= 0);
        if(bOnlyStereo && (!bStereo1 || !bStereo2))
            continue;

        const bool bRight1 = (pKF1->NLeft == -1 || idx1 < pKF1->NLeft) ? false : true;
        const bool bRight2 = (pKF2->NLeft == -1 || idx2 < pKF2->NLeft) ? false : true;

        if(!bStereo1 && !bStereo2 && !pKF1->mpCamera2)
        {
            const float distex = ep(0) - kp2.pt.x;
            const float distey = ep(1) - kp2.pt.y;
            if(distex * distex + distey * distey < 100.0f * pKF2->mvScaleFactors[kp2.octave])
                continue;
        }

        if(pKF1->mpCamera2 && pKF2->mpCamera2)
        {
            if(bRight1 && bRight2)
            {
                R12 = Rrr;
                t12 = trr;
                pCamera1 = pKF1->mpCamera2;
                pCamera2 = pKF2->mpCamera2;
            }
            else if(bRight1 && !bRight2)
            {
                R12 = Rrl;
                t12 = trl;
                pCamera1 = pKF1->mpCamera2;
                pCamera2 = pKF2->mpCamera;
            }
            else if(!bRight1 && bRight2)
            {
                R12 = Rlr;
                t12 = tlr;
                pCamera1 = pKF1->mpCamera;
                pCamera2 = pKF2->mpCamera2;
            }
            else
            {
                R12 = Rll;
                t12 = tll;
                pCamera1 = pKF1->mpCamera;
                pCamera2 = pKF2->mpCamera;
            }
        }

        if(!bCoarse && !pCamera1->epipolarConstrain(pCamera2,
                                                    kp1,
                                                    kp2,
                                                    R12,
                                                    t12,
                                                    pKF1->mvLevelSigma2[kp1.octave],
                                                    pKF2->mvLevelSigma2[kp2.octave]))
        {
            continue;
        }

        candidates.push_back({idx1, idx2, m.score});
    }
    lastStats_.mp_valid = static_cast<int>(candidates.size());

    const int assigned = AssignBestUniqueTriangulationPairs(candidates, n1, n2, vMatchedPairs);
    lastStats_.one_to_one = assigned;
    return assigned;
}

} // namespace ORB_SLAM3
