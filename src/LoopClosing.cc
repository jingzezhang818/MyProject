/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#include "LoopClosing.h"

#include "Sim3Solver.h"
#include "Converter.h"
#include "Optimizer.h"
#include "ORBmatcher.h"
#include "G2oTypes.h"
#include "XFeatLighterGlueMatcher.h"
#include "CameraModels/GeometricCamera.h"

#include<algorithm>
#include<cmath>
#include<cstdlib>
#include<exception>
#include<limits>
#include<mutex>
#include<opencv2/calib3d.hpp>
#include<string>
#include<thread>
#include<unordered_map>
#include<utility>


namespace ORB_SLAM3
{
namespace
{

bool IsLoopEnvFlagEnabled(const char* key)
{
    const char* env = std::getenv(key);
    if(!env)
        return false;

    const std::string v(env);
    return !(v.empty() || v == "0" || v == "false" || v == "FALSE");
}

int GetLoopEnvIntInRange(const char* key, const int fallback, const int minValue, const int maxValue)
{
    const char* env = std::getenv(key);
    if(!env)
        return fallback;

    try
    {
        const int v = std::stoi(std::string(env));
        return std::max(minValue, std::min(maxValue, v));
    }
    catch(...)
    {
        return fallback;
    }
}

float GetLoopEnvFloatInRange(const char* key,
                             const float fallback,
                             const float minValue,
                             const float maxValue)
{
    const char* env = std::getenv(key);
    if(!env)
        return fallback;

    try
    {
        const float v = std::stof(std::string(env));
        return std::max(minValue, std::min(maxValue, v));
    }
    catch(...)
    {
        return fallback;
    }
}

bool IsXFeatLoopLightGlueEnabled()
{
    if(std::getenv("USE_ORB") != nullptr)
        return false;
    return IsLoopEnvFlagEnabled("XFEAT_USE_LIGHTGLUE_LOOP");
}

int GetOrbLoopCandidateCount()
{
    return 3;
}

bool IsXFeatLoopVocabEnabled()
{
    if(std::getenv("USE_ORB") != nullptr)
        return false;
    return IsLoopEnvFlagEnabled("XFEAT_USE_XFEAT_VOCAB");
}

bool ShouldPrintLoopVocabDebug(const unsigned long keyframeId)
{
    if(!IsLoopEnvFlagEnabled("XFEAT_DEBUG"))
        return false;

    const int interval = GetLoopEnvIntInRange("XFEAT_DIAG_INTERVAL", 30, 1, 1000000);
    return (keyframeId % static_cast<unsigned long>(interval)) == 0UL;
}

bool ShouldPrintLoopBestRejectedDebug(const unsigned long keyframeId)
{
    return ShouldPrintLoopVocabDebug(keyframeId);
}

float GetXFeatLightGlueLoopConfidence()
{
    return GetLoopEnvFloatInRange("XFEAT_LIGHTGLUE_CONF", 0.1f, 0.0f, 1.0f);
}

struct LoopLightGlueCandidateMatch
{
    int currentIdx = -1;
    int sourceIdx = -1;
    MapPoint* pMP = nullptr;
    KeyFrame* pSourceKF = nullptr;
    float score = 0.0f;
};

struct LoopLightGlueNearMiss
{
    bool valid = false;
    int candidateRank = -1;
    long long candidateKF = -1;
    long long matchedKF = -1;
    const char* failStage = "none";
    int failStageRank = -1;
    int initialMatches = -1;
    int directRaw = -1;
    int directAssigned = -1;
    int directFInliers = -1;
    int directFRejects = -1;
    int groupSize = 0;
    float groupScore = 0.0f;
    int groupPreselectKFs = 0;
    int groupProjectionKFs = 0;
    int lgCalls = 0;
    int sim3Converged = 0;
    int sim3Inliers = -1;
    int projMatches = -1;
    int optMatches = -1;
    int projOptMatches = -1;
    int covValid = -1;
    int refillCandidates = 0;
    int refillAdded = 0;
};

constexpr int kLoopLightGlueFundamentalMinMatches = 12;
constexpr double kLoopLightGlueFundamentalRansacThreshold = 20.0;
constexpr double kLoopLightGlueFundamentalConfidence = 0.99;
constexpr int kXFeatLoopTopGroupsForLightGlue = 5;
constexpr int kXFeatLoopRefillMinInitialMatches = 50;
constexpr int kXFeatLoopRefillWeakSim3Inliers = 40;
constexpr int kXFeatLoopRefillWeakProjMatches = 120;
constexpr int kXFeatLoopRefillMaxAdded = 80;
constexpr float kXFeatLoopRefillMaxDescriptorDistance = 0.60f;
constexpr float kXFeatLoopRefillMaxReprojErrorPx = 6.0f;

struct LoopGroupRefillCandidate
{
    int currentIdx = -1;
    MapPoint* pMP = nullptr;
    KeyFrame* pSourceKF = nullptr;
    float descriptorDistance = std::numeric_limits<float>::infinity();
    float reprojectionError2 = std::numeric_limits<float>::infinity();
};

struct LoopGroupRefillStats
{
    int candidateMatches = 0;
    int added = 0;
};

float LoopXFeatDescriptorDistance(const cv::Mat& a, const cv::Mat& b)
{
    if(a.empty() || b.empty() || a.type() != CV_32F || b.type() != CV_32F)
        return std::numeric_limits<float>::infinity();

    const int lenA = static_cast<int>(a.total() * a.channels());
    const int lenB = static_cast<int>(b.total() * b.channels());
    if(lenA <= 0 || lenA != lenB)
        return std::numeric_limits<float>::infinity();

    if(a.isContinuous() && b.isContinuous())
    {
        const float* pa = a.ptr<float>(0);
        const float* pb = b.ptr<float>(0);
        float dist2 = 0.0f;
        for(int i = 0; i < lenA; ++i)
        {
            const float d = pa[i] - pb[i];
            dist2 += d * d;
        }
        return std::sqrt(dist2);
    }

    return static_cast<float>(cv::norm(a, b, cv::NORM_L2));
}

bool IsBetterLoopGroupRefillCandidate(const LoopGroupRefillCandidate& candidate,
                                      const LoopGroupRefillCandidate& currentBest)
{
    if(!candidate.pMP || !candidate.pSourceKF)
        return false;
    if(!currentBest.pMP)
        return true;
    if(candidate.descriptorDistance != currentBest.descriptorDistance)
        return candidate.descriptorDistance < currentBest.descriptorDistance;
    return candidate.reprojectionError2 < currentBest.reprojectionError2;
}

bool ShouldRunLoopGroupRefill(const int initialMatches,
                              const int sim3Inliers,
                              const int projMatches)
{
    if(initialMatches < kXFeatLoopRefillMinInitialMatches)
        return false;

    return sim3Inliers < kXFeatLoopRefillWeakSim3Inliers ||
           projMatches < kXFeatLoopRefillWeakProjMatches;
}

int RefillLoopMatchesFromXFeatGroup(KeyFrame* pCurrentKF,
                                    const XFeatKeyFrameDatabase::LoopCandidateGroup& loopGroup,
                                    const Sophus::Sim3f& Scw,
                                    std::vector<MapPoint*>& vpMatchedMP,
                                    std::vector<KeyFrame*>& vpMatchedKF,
                                    std::vector<MapPoint*>& vpMapPoints,
                                    std::vector<KeyFrame*>& vpMapPointKFs,
                                    std::set<MapPoint*>& spMapPoints,
                                    LoopGroupRefillStats& stats)
{
    stats = LoopGroupRefillStats();
    if(!pCurrentKF || loopGroup.vpGroupKFs.empty() ||
       pCurrentKF->mDescriptors.empty() || pCurrentKF->mDescriptors.type() != CV_32F ||
       pCurrentKF->mXFeatWordFeatures.empty())
        return 0;

    const std::vector<MapPoint*> vpCurrentMPs = pCurrentKF->GetMapPointMatches();
    if(vpCurrentMPs.empty())
        return 0;

    Sophus::SE3f Tcw(Scw.rotationMatrix(), Scw.translation() / Scw.scale());
    std::set<MapPoint*> spAlreadyMatched(vpMatchedMP.begin(), vpMatchedMP.end());
    spAlreadyMatched.erase(static_cast<MapPoint*>(NULL));

    std::vector<LoopGroupRefillCandidate> vBestForCurrent(
        static_cast<size_t>(pCurrentKF->N));
    std::unordered_map<long unsigned int, LoopGroupRefillCandidate> mBestForMapPoint;

    const float maxReprojError2 =
        kXFeatLoopRefillMaxReprojErrorPx * kXFeatLoopRefillMaxReprojErrorPx;

    for(const auto& wordEntry : pCurrentKF->mXFeatWordFeatures)
    {
        const DBoW3::WordId wordId = wordEntry.first;
        const std::vector<unsigned int>& vCurrentIndexes = wordEntry.second;

        for(const unsigned int currentIdxU : vCurrentIndexes)
        {
            const int currentIdx = static_cast<int>(currentIdxU);
            if(currentIdx < 0 || currentIdx >= pCurrentKF->N ||
               currentIdx >= static_cast<int>(vpMatchedMP.size()) ||
               currentIdx >= static_cast<int>(vpCurrentMPs.size()))
                continue;
            if(vpMatchedMP[static_cast<size_t>(currentIdx)])
                continue;

            MapPoint* pCurrentMP = vpCurrentMPs[static_cast<size_t>(currentIdx)];
            if(!pCurrentMP || pCurrentMP->isBad())
                continue;

            const cv::Mat currentDesc = pCurrentKF->mDescriptors.row(currentIdx);
            const cv::KeyPoint& currentKp = pCurrentKF->mvKeysUn[static_cast<size_t>(currentIdx)];

            for(KeyFrame* pGroupKF : loopGroup.vpGroupKFs)
            {
                if(!pGroupKF || pGroupKF->isBad() ||
                   pGroupKF->mDescriptors.empty() || pGroupKF->mDescriptors.type() != CV_32F)
                    continue;

                const auto sourceIt = pGroupKF->mXFeatWordFeatures.find(wordId);
                if(sourceIt == pGroupKF->mXFeatWordFeatures.end())
                    continue;

                for(const unsigned int sourceIdxU : sourceIt->second)
                {
                    const int sourceIdx = static_cast<int>(sourceIdxU);
                    if(sourceIdx < 0 || sourceIdx >= pGroupKF->N ||
                       sourceIdx >= pGroupKF->mDescriptors.rows)
                        continue;

                    MapPoint* pCandidateMP = pGroupKF->GetMapPoint(static_cast<size_t>(sourceIdx));
                    if(!pCandidateMP || pCandidateMP->isBad() || spAlreadyMatched.count(pCandidateMP))
                        continue;

                    const cv::Mat candidateDesc = pGroupKF->mDescriptors.row(sourceIdx);
                    const float descriptorDistance =
                        LoopXFeatDescriptorDistance(currentDesc, candidateDesc);
                    if(descriptorDistance > kXFeatLoopRefillMaxDescriptorDistance)
                        continue;

                    const Eigen::Vector3f p3Dc = Tcw * pCandidateMP->GetWorldPos();
                    if(p3Dc(2) <= 0.0f)
                        continue;

                    const Eigen::Vector2f uv = pCurrentKF->mpCamera->project(p3Dc);
                    if(!pCurrentKF->IsInImage(uv(0), uv(1)))
                        continue;

                    const float ex = uv(0) - currentKp.pt.x;
                    const float ey = uv(1) - currentKp.pt.y;
                    float reprojectionError2 = ex * ex + ey * ey;
                    if(currentIdx < static_cast<int>(pCurrentKF->mvuRight.size()) &&
                       pCurrentKF->mvuRight[static_cast<size_t>(currentIdx)] >= 0.0f)
                    {
                        const float ur = uv(0) - pCurrentKF->mbf / p3Dc(2);
                        const float er = ur - pCurrentKF->mvuRight[static_cast<size_t>(currentIdx)];
                        reprojectionError2 += er * er;
                    }
                    if(reprojectionError2 > maxReprojError2)
                        continue;

                    LoopGroupRefillCandidate candidate;
                    candidate.currentIdx = currentIdx;
                    candidate.pMP = pCandidateMP;
                    candidate.pSourceKF = pGroupKF;
                    candidate.descriptorDistance = descriptorDistance;
                    candidate.reprojectionError2 = reprojectionError2;
                    ++stats.candidateMatches;

                    LoopGroupRefillCandidate& bestForCurrent =
                        vBestForCurrent[static_cast<size_t>(currentIdx)];
                    if(IsBetterLoopGroupRefillCandidate(candidate, bestForCurrent))
                        bestForCurrent = candidate;

                    auto mpIt = mBestForMapPoint.find(pCandidateMP->mnId);
                    if(mpIt == mBestForMapPoint.end() ||
                       IsBetterLoopGroupRefillCandidate(candidate, mpIt->second))
                    {
                        mBestForMapPoint[pCandidateMP->mnId] = candidate;
                    }
                }
            }
        }
    }

    int added = 0;
    for(const LoopGroupRefillCandidate& candidate : vBestForCurrent)
    {
        if(added >= kXFeatLoopRefillMaxAdded)
            break;
        if(!candidate.pMP || !candidate.pSourceKF)
            continue;
        const auto mpIt = mBestForMapPoint.find(candidate.pMP->mnId);
        if(mpIt == mBestForMapPoint.end() ||
           mpIt->second.currentIdx != candidate.currentIdx)
            continue;

        const size_t currentIdx = static_cast<size_t>(candidate.currentIdx);
        if(currentIdx >= vpMatchedMP.size() || vpMatchedMP[currentIdx])
            continue;

        vpMatchedMP[currentIdx] = candidate.pMP;
        if(currentIdx < vpMatchedKF.size())
            vpMatchedKF[currentIdx] = candidate.pSourceKF;
        spAlreadyMatched.insert(candidate.pMP);
        if(!spMapPoints.count(candidate.pMP))
        {
            spMapPoints.insert(candidate.pMP);
            vpMapPoints.push_back(candidate.pMP);
            vpMapPointKFs.push_back(candidate.pSourceKF);
        }
        ++added;
    }

    stats.added = added;
    return added;
}

int AssignLoopLightGlueMatches(const std::vector<LoopLightGlueCandidateMatch>& candidates,
                               const int nCurrent,
                               std::vector<MapPoint*>& vpMatchedPoints,
                               std::vector<KeyFrame*>& vpKeyFrameMatchedMP)
{
    std::vector<int> bestForCurrent(static_cast<size_t>(std::max(0, nCurrent)), -1);
    std::unordered_map<long unsigned int, int> bestForMapPoint;
    bestForMapPoint.reserve(candidates.size());

    for(int i = 0; i < static_cast<int>(candidates.size()); ++i)
    {
        const LoopLightGlueCandidateMatch& c = candidates[i];
        if(c.currentIdx < 0 || c.currentIdx >= nCurrent || !c.pMP || c.pMP->isBad() || !c.pSourceKF)
            continue;

        int& currentBest = bestForCurrent[static_cast<size_t>(c.currentIdx)];
        if(currentBest < 0 || c.score > candidates[currentBest].score)
            currentBest = i;

        auto it = bestForMapPoint.find(c.pMP->mnId);
        if(it == bestForMapPoint.end() || c.score > candidates[it->second].score)
            bestForMapPoint[c.pMP->mnId] = i;
    }

    int assigned = 0;
    for(int idx = 0; idx < nCurrent; ++idx)
    {
        const int candIdx = bestForCurrent[static_cast<size_t>(idx)];
        if(candIdx < 0)
            continue;

        const LoopLightGlueCandidateMatch& c = candidates[candIdx];
        const auto it = bestForMapPoint.find(c.pMP->mnId);
        if(it == bestForMapPoint.end() || it->second != candIdx)
            continue;

        vpMatchedPoints[static_cast<size_t>(idx)] = c.pMP;
        vpKeyFrameMatchedMP[static_cast<size_t>(idx)] = c.pSourceKF;
        ++assigned;
    }

    return assigned;
}

void FilterLoopLightGlueMatchesByFundamental(KeyFrame* pCurrentKF,
                                             KeyFrame* pSourceKF,
                                             const std::vector<LoopLightGlueCandidateMatch>& vInputMatches,
                                             std::vector<LoopLightGlueCandidateMatch>& vFilteredMatches,
                                             int& nInliers,
                                             int& nRejects)
{
    vFilteredMatches.clear();
    nInliers = 0;
    nRejects = 0;

    if(!pCurrentKF || !pSourceKF ||
       static_cast<int>(vInputMatches.size()) < kLoopLightGlueFundamentalMinMatches)
    {
        vFilteredMatches = vInputMatches;
        nInliers = static_cast<int>(vFilteredMatches.size());
        return;
    }

    std::vector<cv::Point2f> vCurrentPts;
    std::vector<cv::Point2f> vSourcePts;
    std::vector<int> vInputIndexes;
    vCurrentPts.reserve(vInputMatches.size());
    vSourcePts.reserve(vInputMatches.size());
    vInputIndexes.reserve(vInputMatches.size());

    for(int i = 0; i < static_cast<int>(vInputMatches.size()); ++i)
    {
        const LoopLightGlueCandidateMatch& match = vInputMatches[i];
        if(match.currentIdx < 0 || match.currentIdx >= static_cast<int>(pCurrentKF->mvKeysUn.size()) ||
           match.sourceIdx < 0 || match.sourceIdx >= static_cast<int>(pSourceKF->mvKeysUn.size()))
        {
            continue;
        }

        vCurrentPts.push_back(pCurrentKF->mvKeysUn[match.currentIdx].pt);
        vSourcePts.push_back(pSourceKF->mvKeysUn[match.sourceIdx].pt);
        vInputIndexes.push_back(i);
    }

    if(static_cast<int>(vInputIndexes.size()) < kLoopLightGlueFundamentalMinMatches)
    {
        vFilteredMatches = vInputMatches;
        nInliers = static_cast<int>(vFilteredMatches.size());
        return;
    }

    std::vector<uchar> vInlierMask;
    cv::findFundamentalMat(vCurrentPts,
                           vSourcePts,
                           cv::FM_RANSAC,
                           kLoopLightGlueFundamentalRansacThreshold,
                           kLoopLightGlueFundamentalConfidence,
                           vInlierMask);

    if(vInlierMask.size() != vInputIndexes.size())
    {
        vFilteredMatches = vInputMatches;
        nInliers = static_cast<int>(vFilteredMatches.size());
        return;
    }

    std::vector<unsigned char> vKeep(vInputMatches.size(), 0);
    for(int i = 0; i < static_cast<int>(vInlierMask.size()); ++i)
    {
        if(vInlierMask[i])
            vKeep[static_cast<size_t>(vInputIndexes[i])] = 1;
    }

    vFilteredMatches.reserve(vInputMatches.size());
    for(int i = 0; i < static_cast<int>(vInputMatches.size()); ++i)
    {
        if(vKeep[static_cast<size_t>(i)])
        {
            vFilteredMatches.push_back(vInputMatches[i]);
        }
        else
        {
            ++nRejects;
        }
    }
    nInliers = static_cast<int>(vFilteredMatches.size());
}

bool IsBetterLoopLightGlueNearMiss(const LoopLightGlueNearMiss& candidate,
                                   const LoopLightGlueNearMiss& currentBest)
{
    if(!candidate.valid)
        return false;
    if(!currentBest.valid)
        return true;

    if(candidate.failStageRank != currentBest.failStageRank)
        return candidate.failStageRank > currentBest.failStageRank;
    if(candidate.projOptMatches != currentBest.projOptMatches)
        return candidate.projOptMatches > currentBest.projOptMatches;
    if(candidate.optMatches != currentBest.optMatches)
        return candidate.optMatches > currentBest.optMatches;
    if(candidate.projMatches != currentBest.projMatches)
        return candidate.projMatches > currentBest.projMatches;
    if(candidate.sim3Inliers != currentBest.sim3Inliers)
        return candidate.sim3Inliers > currentBest.sim3Inliers;
    if(candidate.initialMatches != currentBest.initialMatches)
        return candidate.initialMatches > currentBest.initialMatches;
    return candidate.candidateRank < currentBest.candidateRank;
}

void PrintLoopLightGlueNearMiss(const unsigned long currentKFId,
                                const LoopLightGlueNearMiss& nearMiss,
                                const int nBoWMatches,
                                const int nBoWInliers,
                                const int nSim3Inliers,
                                const int nProjMatches,
                                const int nProjOptMatches)
{
    if(!nearMiss.valid)
        return;

    std::cout << "[LoopLG][NearMiss]"
              << " current_kf=" << currentKFId
              << " candidate_kf=" << nearMiss.candidateKF
              << " candidate_rank=" << nearMiss.candidateRank
              << " matched_kf=" << nearMiss.matchedKF
              << " fail_stage=" << nearMiss.failStage
              << " initial_matches=" << nearMiss.initialMatches
              << " direct_raw=" << nearMiss.directRaw
              << " direct_assigned=" << nearMiss.directAssigned
              << " direct_f_inliers=" << nearMiss.directFInliers
              << " direct_f_rejects=" << nearMiss.directFRejects
              << " group_size=" << nearMiss.groupSize
              << " group_score=" << nearMiss.groupScore
              << " group_preselect_kfs=" << nearMiss.groupPreselectKFs
              << " group_projection_kfs=" << nearMiss.groupProjectionKFs
              << " lg_calls=" << nearMiss.lgCalls
              << " sim3_converged=" << nearMiss.sim3Converged
              << " sim3_inliers=" << nearMiss.sim3Inliers
              << " proj_matches=" << nearMiss.projMatches
              << " opt_matches=" << nearMiss.optMatches
              << " proj_opt_matches=" << nearMiss.projOptMatches
              << " cov_valid=" << nearMiss.covValid
              << " refill_candidates=" << nearMiss.refillCandidates
              << " refill_added=" << nearMiss.refillAdded
              << " th_initial=" << nBoWMatches
              << " th_sim3_ransac=" << nBoWInliers
              << " th_opt=" << nSim3Inliers
              << " th_proj=" << nProjMatches
              << " th_proj_opt=" << nProjOptMatches
              << " th_cov=3"
              << std::endl;
}

void PrintLoopLightGlueBestRejected(const unsigned long currentKFId,
                                    const LoopLightGlueNearMiss& nearMiss,
                                    const int nBoWMatches,
                                    const int nBoWInliers,
                                    const int nSim3Inliers,
                                    const int nProjMatches,
                                    const int nProjOptMatches)
{
    if(!nearMiss.valid || nearMiss.sim3Converged == 0)
        return;

    std::cout << "[LoopLG][BestRejected]"
              << " current_kf=" << currentKFId
              << " candidate_kf=" << nearMiss.candidateKF
              << " candidate_rank=" << nearMiss.candidateRank
              << " matched_kf=" << nearMiss.matchedKF
              << " fail_stage=" << nearMiss.failStage
              << " initial_matches=" << nearMiss.initialMatches
              << " direct_raw=" << nearMiss.directRaw
              << " direct_assigned=" << nearMiss.directAssigned
              << " direct_f_inliers=" << nearMiss.directFInliers
              << " direct_f_rejects=" << nearMiss.directFRejects
              << " group_size=" << nearMiss.groupSize
              << " group_score=" << nearMiss.groupScore
              << " group_preselect_kfs=" << nearMiss.groupPreselectKFs
              << " group_projection_kfs=" << nearMiss.groupProjectionKFs
              << " lg_calls=" << nearMiss.lgCalls
              << " sim3_inliers=" << nearMiss.sim3Inliers
              << " proj_matches=" << nearMiss.projMatches
              << " opt_matches=" << nearMiss.optMatches
              << " proj_opt_matches=" << nearMiss.projOptMatches
              << " cov_valid=" << nearMiss.covValid
              << " refill_candidates=" << nearMiss.refillCandidates
              << " refill_added=" << nearMiss.refillAdded
              << " th_initial=" << nBoWMatches
              << " th_sim3_ransac=" << nBoWInliers
              << " th_opt=" << nSim3Inliers
              << " th_proj=" << nProjMatches
              << " th_proj_opt=" << nProjOptMatches
              << " th_cov=3"
              << std::endl;
}

void AppendUniqueLoopKeyFrame(std::vector<KeyFrame*>& vKFs,
                              std::set<KeyFrame*>& spAlreadyAdded,
                              KeyFrame* pKF,
                              const size_t maxSize)
{
    if(!pKF || pKF->isBad() || spAlreadyAdded.count(pKF))
        return;
    if(maxSize > 0 && vKFs.size() >= maxSize)
        return;
    vKFs.push_back(pKF);
    spAlreadyAdded.insert(pKF);
}

const char* XFeatGbaAbortReasonName(const int reason)
{
    switch(reason)
    {
    case 1: return "idx_changed";
    case 2: return "imu_state_changed";
    case 3: return "finish_requested";
    case 4: return "stop_requested";
    default: return "none";
    }
}

} // namespace

LoopClosing::LoopClosing(Atlas *pAtlas, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale, const bool bActiveLC):
    mbResetRequested(false), mbResetActiveMapRequested(false), mbFinishRequested(false), mbFinished(true), mpAtlas(pAtlas),
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpXFeatVocabulary(NULL), mpXFeatKeyFrameDB(NULL),
    mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0), mnLoopNumCoincidences(0), mnMergeNumCoincidences(0),
    mbLoopDetected(false), mbMergeDetected(false), mnLoopNumNotFound(0), mnMergeNumNotFound(0), mbActiveLC(bActiveLC)
{
    mnCovisibilityConsistencyTh = 3;
    mpLastCurrentKF = static_cast<KeyFrame*>(NULL);
    mpCurrentKF = static_cast<KeyFrame*>(NULL);
    mpLoopLastCurrentKF = static_cast<KeyFrame*>(NULL);
    mpLoopMatchedKF = static_cast<KeyFrame*>(NULL);
    mpMergeLastCurrentKF = static_cast<KeyFrame*>(NULL);
    mpMergeMatchedKF = static_cast<KeyFrame*>(NULL);

    mnXFeatLoopLgCandidatesChecked = 0;
    mnXFeatLoopLgCandidateMaxRank = -1;
    mnXFeatLoopLgSim3ConvergedCandidates = 0;
    mnXFeatLoopLgTotalCalls = 0;
    mnXFeatLoopLgAcceptCount = 0;
    mnXFeatLoopLgLastAcceptCurrentKF = -1;
    mnXFeatLoopLgLastAcceptCandidateKF = -1;
    mnXFeatLoopLgLastAcceptMatchedKF = -1;
    mnXFeatLoopLgLastAcceptRank = -1;
    mnXFeatLoopLgLastAcceptInitialMatches = -1;
    mnXFeatLoopLgLastAcceptSim3Inliers = -1;
    mnXFeatLoopLgLastAcceptProjOptMatches = -1;
    mnXFeatLoopLgLastAcceptCovValid = -1;
    mnXFeatLoopLgBestAcceptCurrentKF = -1;
    mnXFeatLoopLgBestAcceptCandidateKF = -1;
    mnXFeatLoopLgBestAcceptMatchedKF = -1;
    mnXFeatLoopLgBestAcceptRank = -1;
    mnXFeatLoopLgBestAcceptSim3Inliers = -1;
    mnXFeatLoopLgBestAcceptProjOptMatches = -1;
    mnXFeatLoopLgBestAcceptCovValid = -1;
    mnXFeatLoopCorrectionCount = 0;
    mnXFeatLoopLastCorrectionCurrentKF = -1;
    mnXFeatLoopLastCorrectionMatchedKF = -1;
    mnXFeatLoopLastCorrectionCoincidences = -1;
    mnXFeatGbaStartedCount = 0;
    mnXFeatGbaOptimizerFinishedCount = 0;
    mnXFeatGbaAppliedCount = 0;
    mnXFeatGbaAbortedCount = 0;
    mnXFeatGbaStopRequestedCount = 0;
    mnXFeatLastGbaLoopKF = -1;
    mnXFeatLastGbaAbortReason = 0;
    mbXFeatShutdownSawLoopNotFinished = false;
    mbXFeatShutdownSawGbaRunning = false;
    mbXFeatShutdownTimeout = false;

#ifdef REGISTER_TIMES

    vdDataQuery_ms.clear();
    vdEstSim3_ms.clear();
    vdPRTotal_ms.clear();

    vdMergeMaps_ms.clear();
    vdWeldingBA_ms.clear();
    vdMergeOptEss_ms.clear();
    vdMergeTotal_ms.clear();
    vnMergeKFs.clear();
    vnMergeMPs.clear();
    nMerges = 0;

    vdLoopFusion_ms.clear();
    vdLoopOptEss_ms.clear();
    vdLoopTotal_ms.clear();
    vnLoopKFs.clear();
    nLoop = 0;

    vdGBA_ms.clear();
    vdUpdateMap_ms.clear();
    vdFGBATotal_ms.clear();
    vnGBAKFs.clear();
    vnGBAMPs.clear();
    nFGBA_exec = 0;
    nFGBA_abort = 0;

#endif

    mstrFolderSubTraj = "SubTrajectories/";
    mnNumCorrection = 0;
    mnCorrectionGBA = 0;
}

void LoopClosing::SetXFeatVocabularyDatabase(XFeatVocabulary* pXFeatVoc,
                                             XFeatKeyFrameDatabase* pXFeatKFDB)
{
    mpXFeatVocabulary = pXFeatVoc;
    mpXFeatKeyFrameDB = pXFeatKFDB;
}

void LoopClosing::AddKeyFrameToDatabases(KeyFrame* pKF)
{
    if(!pKF)
        return;
    mpKeyFrameDB->add(pKF);
    if(mpXFeatKeyFrameDB)
        mpXFeatKeyFrameDB->add(pKF);
}

void LoopClosing::RecordXFeatLoopShutdownStatus(const bool bLoopCloserNotFinished,
                                                const bool bLoopCloserRunningGBA,
                                                const bool bShutdownTimeout)
{
    unique_lock<mutex> lock(mMutexXFeatLoopSummary);
    mbXFeatShutdownSawLoopNotFinished = mbXFeatShutdownSawLoopNotFinished || bLoopCloserNotFinished;
    mbXFeatShutdownSawGbaRunning = mbXFeatShutdownSawGbaRunning || bLoopCloserRunningGBA;
    mbXFeatShutdownTimeout = mbXFeatShutdownTimeout || bShutdownTimeout;
}

void LoopClosing::PrintXFeatLoopSummary()
{
    unique_lock<mutex> lock(mMutexXFeatLoopSummary);
    std::cout << "[LoopLG][Summary]"
              << " checked_candidates=" << mnXFeatLoopLgCandidatesChecked
              << " max_candidate_rank=" << mnXFeatLoopLgCandidateMaxRank
              << " sim3_converged_candidates=" << mnXFeatLoopLgSim3ConvergedCandidates
              << " total_lg_calls=" << mnXFeatLoopLgTotalCalls
              << " accept_count=" << mnXFeatLoopLgAcceptCount
              << " last_accept_current_kf=" << mnXFeatLoopLgLastAcceptCurrentKF
              << " last_accept_candidate_kf=" << mnXFeatLoopLgLastAcceptCandidateKF
              << " last_accept_matched_kf=" << mnXFeatLoopLgLastAcceptMatchedKF
              << " last_accept_rank=" << mnXFeatLoopLgLastAcceptRank
              << " last_accept_initial_matches=" << mnXFeatLoopLgLastAcceptInitialMatches
              << " last_accept_sim3_inliers=" << mnXFeatLoopLgLastAcceptSim3Inliers
              << " last_accept_proj_opt_matches=" << mnXFeatLoopLgLastAcceptProjOptMatches
              << " last_accept_cov_valid=" << mnXFeatLoopLgLastAcceptCovValid
              << " best_accept_current_kf=" << mnXFeatLoopLgBestAcceptCurrentKF
              << " best_accept_candidate_kf=" << mnXFeatLoopLgBestAcceptCandidateKF
              << " best_accept_matched_kf=" << mnXFeatLoopLgBestAcceptMatchedKF
              << " best_accept_rank=" << mnXFeatLoopLgBestAcceptRank
              << " best_accept_sim3_inliers=" << mnXFeatLoopLgBestAcceptSim3Inliers
              << " best_accept_proj_opt_matches=" << mnXFeatLoopLgBestAcceptProjOptMatches
              << " best_accept_cov_valid=" << mnXFeatLoopLgBestAcceptCovValid
              << " loop_correction_count=" << mnXFeatLoopCorrectionCount
              << " last_loop_current_kf=" << mnXFeatLoopLastCorrectionCurrentKF
              << " last_loop_matched_kf=" << mnXFeatLoopLastCorrectionMatchedKF
              << " last_loop_num_coincidences=" << mnXFeatLoopLastCorrectionCoincidences
              << " gba_started=" << mnXFeatGbaStartedCount
              << " gba_optimizer_finished=" << mnXFeatGbaOptimizerFinishedCount
              << " gba_applied=" << mnXFeatGbaAppliedCount
              << " gba_aborted=" << mnXFeatGbaAbortedCount
              << " gba_stop_requested=" << mnXFeatGbaStopRequestedCount
              << " last_gba_loop_kf=" << mnXFeatLastGbaLoopKF
              << " last_gba_abort_reason=" << XFeatGbaAbortReasonName(mnXFeatLastGbaAbortReason)
              << " shutdown_saw_loop_not_finished=" << (mbXFeatShutdownSawLoopNotFinished ? 1 : 0)
              << " shutdown_saw_gba_running=" << (mbXFeatShutdownSawGbaRunning ? 1 : 0)
              << " shutdown_timeout=" << (mbXFeatShutdownTimeout ? 1 : 0)
              << std::endl;
}

void LoopClosing::RequestShutdownGBAStop()
{
    if(!mMutexGBA.try_lock())
    {
        cout << "[LoopGBA] shutdown timeout: GBA mutex busy, continuing shutdown." << endl;
        return;
    }

    unique_lock<mutex> lock(mMutexGBA, std::adopt_lock);
    if(!mbRunningGBA)
        return;

    const bool bAlreadyStopRequested = mbStopGBA;
    mbStopGBA = true;
    {
        unique_lock<mutex> summaryLock(mMutexXFeatLoopSummary);
        if(!bAlreadyStopRequested)
            ++mnXFeatGbaStopRequestedCount;
    }
    cout << "[LoopGBA] shutdown timeout: requesting GBA stop"
         << (bAlreadyStopRequested ? " (already requested)." : ".") << endl;
}

void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}


void LoopClosing::Run()
{
    mbFinished =false;

    while(1)
    {
        if(CheckFinish()){
            break;
        }

        //NEW LOOP AND MERGE DETECTION ALGORITHM
        //----------------------------


        if(CheckNewKeyFrames())
        {
            auto cleanupCurrentPlaceRecognition = [&]() {
                if(mbLoopDetected || mnLoopNumCoincidences > 0)
                {
                    if(mpLoopLastCurrentKF)
                        mpLoopLastCurrentKF->SetErase();
                    if(mpLoopMatchedKF)
                        mpLoopMatchedKF->SetErase();
                    mnLoopNumCoincidences = 0;
                    mnLoopNumNotFound = 0;
                    mvpLoopMatchedMPs.clear();
                    mvpLoopMPs.clear();
                    mbLoopDetected = false;
                }

                if(mbMergeDetected || mnMergeNumCoincidences > 0)
                {
                    if(mpMergeLastCurrentKF)
                        mpMergeLastCurrentKF->SetErase();
                    if(mpMergeMatchedKF)
                        mpMergeMatchedKF->SetErase();
                    mnMergeNumCoincidences = 0;
                    mnMergeNumNotFound = 0;
                    mvpMergeMatchedMPs.clear();
                    mvpMergeMPs.clear();
                    mbMergeDetected = false;
                }

                if(mpCurrentKF)
                {
                    mpCurrentKF->mbCurrentPlaceRecognition = false;
                    mpCurrentKF->SetErase();
                }
            };

            if(mpLastCurrentKF)
            {
                mpLastCurrentKF->mvpLoopCandKFs.clear();
                mpLastCurrentKF->mvpMergeCandKFs.clear();
            }
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartPR = std::chrono::steady_clock::now();
#endif

            bool bFindedRegion = NewDetectCommonRegions();

            if(CheckFinish())
            {
                if(bFindedRegion)
                    cleanupCurrentPlaceRecognition();
                break;
            }

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndPR = std::chrono::steady_clock::now();

            double timePRTotal = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPR - time_StartPR).count();
            vdPRTotal_ms.push_back(timePRTotal);
#endif
            if(bFindedRegion)
            {
                if(mbMergeDetected)
                {
                    if ((mpTracker->mSensor==System::IMU_MONOCULAR || mpTracker->mSensor==System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD) &&
                        (!mpCurrentKF->GetMap()->isImuInitialized()))
                    {
                        cout << "IMU is not initilized, merge is aborted" << endl;
                    }
                    else
                    {
                        Sophus::SE3d mTmw = mpMergeMatchedKF->GetPose().cast<double>();
                        g2o::Sim3 gSmw2(mTmw.unit_quaternion(), mTmw.translation(), 1.0);
                        Sophus::SE3d mTcw = mpCurrentKF->GetPose().cast<double>();
                        g2o::Sim3 gScw1(mTcw.unit_quaternion(), mTcw.translation(), 1.0);
                        g2o::Sim3 gSw2c = mg2oMergeSlw.inverse();
                        g2o::Sim3 gSw1m = mg2oMergeSlw;

                        mSold_new = (gSw2c * gScw1);


                        if(mpCurrentKF->GetMap()->IsInertial() && mpMergeMatchedKF->GetMap()->IsInertial())
                        {
                            cout << "Merge check transformation with IMU" << endl;
                            if(mSold_new.scale()<0.90||mSold_new.scale()>1.1){
                                mpMergeLastCurrentKF->SetErase();
                                mpMergeMatchedKF->SetErase();
                                mnMergeNumCoincidences = 0;
                                mvpMergeMatchedMPs.clear();
                                mvpMergeMPs.clear();
                                mnMergeNumNotFound = 0;
                                mbMergeDetected = false;
                                Verbose::PrintMess("scale bad estimated. Abort merging", Verbose::VERBOSITY_NORMAL);
                                continue;
                            }
                            // If inertial, force only yaw
                            if ((mpTracker->mSensor==System::IMU_MONOCULAR || mpTracker->mSensor==System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD) &&
                                   mpCurrentKF->GetMap()->GetIniertialBA1())
                            {
                                Eigen::Vector3d phi = LogSO3(mSold_new.rotation().toRotationMatrix());
                                phi(0)=0;
                                phi(1)=0;
                                mSold_new = g2o::Sim3(ExpSO3(phi),mSold_new.translation(),1.0);
                            }
                        }

                        mg2oMergeSmw = gSmw2 * gSw2c * gScw1;

                        mg2oMergeScw = mg2oMergeSlw;

                        //mpTracker->SetStepByStep(true);

                        Verbose::PrintMess("*Merge detected", Verbose::VERBOSITY_QUIET);

#ifdef REGISTER_TIMES
                        std::chrono::steady_clock::time_point time_StartMerge = std::chrono::steady_clock::now();

                        nMerges += 1;
#endif
                        // TODO UNCOMMENT
                        if (mpTracker->mSensor==System::IMU_MONOCULAR ||mpTracker->mSensor==System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD)
                            MergeLocal2();
                        else
                            MergeLocal();

#ifdef REGISTER_TIMES
                        std::chrono::steady_clock::time_point time_EndMerge = std::chrono::steady_clock::now();

                        double timeMergeTotal = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndMerge - time_StartMerge).count();
                        vdMergeTotal_ms.push_back(timeMergeTotal);
#endif

                        Verbose::PrintMess("Merge finished!", Verbose::VERBOSITY_QUIET);
                    }

                    vdPR_CurrentTime.push_back(mpCurrentKF->mTimeStamp);
                    vdPR_MatchedTime.push_back(mpMergeMatchedKF->mTimeStamp);
                    vnPR_TypeRecogn.push_back(1);

                    // Reset all variables
                    mpMergeLastCurrentKF->SetErase();
                    mpMergeMatchedKF->SetErase();
                    mnMergeNumCoincidences = 0;
                    mvpMergeMatchedMPs.clear();
                    mvpMergeMPs.clear();
                    mnMergeNumNotFound = 0;
                    mbMergeDetected = false;

                    if(mbLoopDetected)
                    {
                        // Reset Loop variables
                        mpLoopLastCurrentKF->SetErase();
                        mpLoopMatchedKF->SetErase();
                        mnLoopNumCoincidences = 0;
                        mvpLoopMatchedMPs.clear();
                        mvpLoopMPs.clear();
                        mnLoopNumNotFound = 0;
                        mbLoopDetected = false;
                    }

                }

                if(mbLoopDetected)
                {
                    bool bGoodLoop = true;
                    vdPR_CurrentTime.push_back(mpCurrentKF->mTimeStamp);
                    vdPR_MatchedTime.push_back(mpLoopMatchedKF->mTimeStamp);
                    vnPR_TypeRecogn.push_back(0);

                    Verbose::PrintMess("*Loop detected", Verbose::VERBOSITY_QUIET);

                    mg2oLoopScw = mg2oLoopSlw; //*mvg2oSim3LoopTcw[nCurrentIndex];
                    if(mpCurrentKF->GetMap()->IsInertial())
                    {
                        Sophus::SE3d Twc = mpCurrentKF->GetPoseInverse().cast<double>();
                        g2o::Sim3 g2oTwc(Twc.unit_quaternion(),Twc.translation(),1.0);
                        g2o::Sim3 g2oSww_new = g2oTwc*mg2oLoopScw;

                        Eigen::Vector3d phi = LogSO3(g2oSww_new.rotation().toRotationMatrix());
                        cout << "phi = " << phi.transpose() << endl; 
                        if (fabs(phi(0))<0.008f && fabs(phi(1))<0.008f && fabs(phi(2))<0.349f)
                        {
                            if(mpCurrentKF->GetMap()->IsInertial())
                            {
                                // If inertial, force only yaw
                                if ((mpTracker->mSensor==System::IMU_MONOCULAR ||mpTracker->mSensor==System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD) &&
                                        mpCurrentKF->GetMap()->GetIniertialBA2())
                                {
                                    phi(0)=0;
                                    phi(1)=0;
                                    g2oSww_new = g2o::Sim3(ExpSO3(phi),g2oSww_new.translation(),1.0);
                                    mg2oLoopScw = g2oTwc.inverse()*g2oSww_new;
                                }
                            }

                        }
                        else
                        {
                            cout << "BAD LOOP!!!" << endl;
                            bGoodLoop = false;
                        }

                    }

                    if (bGoodLoop) {

                        mvpLoopMapPoints = mvpLoopMPs;

#ifdef REGISTER_TIMES
                        std::chrono::steady_clock::time_point time_StartLoop = std::chrono::steady_clock::now();

                        nLoop += 1;

#endif
                        CorrectLoop();
#ifdef REGISTER_TIMES
                        std::chrono::steady_clock::time_point time_EndLoop = std::chrono::steady_clock::now();

                        double timeLoopTotal = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLoop - time_StartLoop).count();
                        vdLoopTotal_ms.push_back(timeLoopTotal);
#endif

                        mnNumCorrection += 1;
                    }

                    // Reset all variables
                    mpLoopLastCurrentKF->SetErase();
                    mpLoopMatchedKF->SetErase();
                    mnLoopNumCoincidences = 0;
                    mvpLoopMatchedMPs.clear();
                    mvpLoopMPs.clear();
                    mnLoopNumNotFound = 0;
                    mbLoopDetected = false;
                }

            }
            mpLastCurrentKF = mpCurrentKF;
        }

        ResetIfRequested();

        if(CheckFinish()){
            break;
        }

        usleep(5000);
    }

    SetFinish();
}

void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> finishLock(mMutexFinish);
    if(mbFinishRequested)
        return;

    unique_lock<mutex> lock(mMutexLoopQueue);
    if(pKF->mnId!=0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}

bool LoopClosing::NewDetectCommonRegions()
{
    if(CheckFinish())
        return false;

    // To deactivate placerecognition. No loopclosing nor merging will be performed
    if(!mbActiveLC)
        return false;

    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->SetNotErase();
        mpCurrentKF->mbCurrentPlaceRecognition = true;

        mpLastMap = mpCurrentKF->GetMap();
    }

    if(CheckFinish())
    {
        mpCurrentKF->SetErase();
        mpCurrentKF->mbCurrentPlaceRecognition = false;
        return false;
    }

    if(mpLastMap->IsInertial() && !mpLastMap->GetIniertialBA2())
    {
        AddKeyFrameToDatabases(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    if(mpTracker->mSensor == System::STEREO && mpLastMap->GetAllKeyFrames().size() < 5) //12
    {
        // cout << "LoopClousure: Stereo KF inserted without check: " << mpCurrentKF->mnId << endl;
        AddKeyFrameToDatabases(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    if(mpLastMap->GetAllKeyFrames().size() < 12)
    {
        // cout << "LoopClousure: Stereo KF inserted without check, map is small: " << mpCurrentKF->mnId << endl;
        AddKeyFrameToDatabases(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    //cout << "LoopClousure: Checking KF: " << mpCurrentKF->mnId << endl;

    //Check the last candidates with geometric validation
    // Loop candidates
    bool bLoopDetectedInKF = false;
    bool bCheckSpatial = false;

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartEstSim3_1 = std::chrono::steady_clock::now();
#endif
    if(mnLoopNumCoincidences > 0)
    {
        bCheckSpatial = true;
        // Find from the last KF candidates
        Sophus::SE3d mTcl = (mpCurrentKF->GetPose() * mpLoopLastCurrentKF->GetPoseInverse()).cast<double>();
        g2o::Sim3 gScl(mTcl.unit_quaternion(),mTcl.translation(),1.0);
        g2o::Sim3 gScw = gScl * mg2oLoopSlw;
        int numProjMatches = 0;
        vector<MapPoint*> vpMatchedMPs;
        bool bCommonRegion = DetectAndReffineSim3FromLastKF(mpCurrentKF, mpLoopMatchedKF, gScw, numProjMatches, mvpLoopMPs, vpMatchedMPs);
        if(bCommonRegion)
        {

            bLoopDetectedInKF = true;

            mnLoopNumCoincidences++;
            mpLoopLastCurrentKF->SetErase();
            mpLoopLastCurrentKF = mpCurrentKF;
            mg2oLoopSlw = gScw;
            mvpLoopMatchedMPs = vpMatchedMPs;


            mbLoopDetected = mnLoopNumCoincidences >= 3;
            mnLoopNumNotFound = 0;

            if(!mbLoopDetected)
            {
                cout << "PR: Loop detected with Reffine Sim3" << endl;
            }
        }
        else
        {
            bLoopDetectedInKF = false;

            mnLoopNumNotFound++;
            if(mnLoopNumNotFound >= 2)
            {
                mpLoopLastCurrentKF->SetErase();
                mpLoopMatchedKF->SetErase();
                mnLoopNumCoincidences = 0;
                mvpLoopMatchedMPs.clear();
                mvpLoopMPs.clear();
                mnLoopNumNotFound = 0;
            }

        }
    }

    //Merge candidates
    bool bMergeDetectedInKF = false;
    if(mnMergeNumCoincidences > 0)
    {
        // Find from the last KF candidates
        Sophus::SE3d mTcl = (mpCurrentKF->GetPose() * mpMergeLastCurrentKF->GetPoseInverse()).cast<double>();

        g2o::Sim3 gScl(mTcl.unit_quaternion(), mTcl.translation(), 1.0);
        g2o::Sim3 gScw = gScl * mg2oMergeSlw;
        int numProjMatches = 0;
        vector<MapPoint*> vpMatchedMPs;
        bool bCommonRegion = DetectAndReffineSim3FromLastKF(mpCurrentKF, mpMergeMatchedKF, gScw, numProjMatches, mvpMergeMPs, vpMatchedMPs);
        if(bCommonRegion)
        {
            bMergeDetectedInKF = true;

            mnMergeNumCoincidences++;
            mpMergeLastCurrentKF->SetErase();
            mpMergeLastCurrentKF = mpCurrentKF;
            mg2oMergeSlw = gScw;
            mvpMergeMatchedMPs = vpMatchedMPs;

            mbMergeDetected = mnMergeNumCoincidences >= 3;
        }
        else
        {
            mbMergeDetected = false;
            bMergeDetectedInKF = false;

            mnMergeNumNotFound++;
            if(mnMergeNumNotFound >= 2)
            {
                mpMergeLastCurrentKF->SetErase();
                mpMergeMatchedKF->SetErase();
                mnMergeNumCoincidences = 0;
                mvpMergeMatchedMPs.clear();
                mvpMergeMPs.clear();
                mnMergeNumNotFound = 0;
            }


        }
    }  
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndEstSim3_1 = std::chrono::steady_clock::now();

        double timeEstSim3 = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndEstSim3_1 - time_StartEstSim3_1).count();
#endif

    if(mbMergeDetected || mbLoopDetected)
    {
#ifdef REGISTER_TIMES
        vdEstSim3_ms.push_back(timeEstSim3);
#endif
        AddKeyFrameToDatabases(mpCurrentKF);
        return true;
    }

    //TODO: This is only necessary if we use a minimun score for pick the best candidates
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();

    // Extract candidates from the bag of words
    vector<KeyFrame*> vpMergeBowCand, vpLoopBowCand;
    std::vector<XFeatKeyFrameDatabase::LoopCandidateGroup> vXFeatLoopGroups;
    const bool bUseLightGlueLoop = IsXFeatLoopLightGlueEnabled();
    const bool bUseXFeatVocabLoop = IsXFeatLoopVocabEnabled() &&
                                    mpXFeatVocabulary &&
                                    mpXFeatKeyFrameDB;
    const int nOrbLoopCandidates = GetOrbLoopCandidateCount();
    if(!bMergeDetectedInKF || !bLoopDetectedInKF)
    {
        // Search in BoW
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartQuery = std::chrono::steady_clock::now();
#endif
        vector<KeyFrame*> vpOrbLoopBowCand;
        if(!bMergeDetectedInKF || (!bLoopDetectedInKF && !bUseXFeatVocabLoop))
        {
            mpKeyFrameDB->DetectNBestCandidates(mpCurrentKF,
                                                vpOrbLoopBowCand,
                                                vpMergeBowCand,
                                                nOrbLoopCandidates);
        }

        if(!bLoopDetectedInKF)
        {
            if(bUseXFeatVocabLoop)
            {
                mpXFeatKeyFrameDB->DetectLoopCandidateGroups(mpCurrentKF,
                                                             vXFeatLoopGroups);
                const int nGroupsForLightGlue =
                    std::min<int>(kXFeatLoopTopGroupsForLightGlue, vXFeatLoopGroups.size());
                vpLoopBowCand.reserve(static_cast<size_t>(nGroupsForLightGlue));
                for(int i = 0; i < nGroupsForLightGlue; ++i)
                {
                    const XFeatKeyFrameDatabase::LoopCandidateGroup& group =
                        vXFeatLoopGroups[static_cast<size_t>(i)];
                    if(group.pRepresentative)
                        vpLoopBowCand.push_back(group.pRepresentative);
                }
                if(ShouldPrintLoopVocabDebug(mpCurrentKF->mnId))
                {
                    const std::vector<XFeatKeyFrameDatabase::CandidateDebugInfo> debugInfo =
                        mpXFeatKeyFrameDB->GetLastDebugInfo();
                    std::cout << "[LoopVocab]"
                              << " source=xfeat_vocab"
                              << " current_kf=" << mpCurrentKF->mnId
                              << " loop_groups=" << vXFeatLoopGroups.size()
                              << " top_groups_for_lg=" << kXFeatLoopTopGroupsForLightGlue
                              << " loop_candidates=" << vpLoopBowCand.size()
                              << " xfeat_words=" << mpCurrentKF->mXFeatBowVec.size()
                              << " sharing_kfs=" << mpXFeatKeyFrameDB->GetLastSharingWords()
                              << " max_common_words=" << mpXFeatKeyFrameDB->GetLastMaxCommonWords()
                              << " min_common_words=" << mpXFeatKeyFrameDB->GetLastMinCommonWords();
                    const int nPrint = std::min<int>(3, debugInfo.size());
                    for(int i = 0; i < nPrint; ++i)
                    {
                        std::cout << " cand" << i
                                  << "_kf=" << debugInfo[i].keyframeId
                                  << " cand" << i
                                  << "_words=" << debugInfo[i].commonWords
                                  << " cand" << i
                                  << "_score=" << debugInfo[i].score
                                  << " cand" << i
                                  << "_acc=" << debugInfo[i].accumulatedScore
                                  << " cand" << i
                                  << "_group=" << debugInfo[i].groupSize;
                    }
                    std::cout << std::endl;
                }
            }
            else
            {
                vpLoopBowCand = vpOrbLoopBowCand;
                if(IsXFeatLoopVocabEnabled() && ShouldPrintLoopVocabDebug(mpCurrentKF->mnId))
                {
                    std::cout << "[LoopVocab]"
                              << " source=orb_vocab"
                              << " current_kf=" << mpCurrentKF->mnId
                              << " reason=xfeat_vocab_unavailable"
                              << " loop_candidates=" << vpLoopBowCand.size()
                              << std::endl;
                }
            }
        }
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndQuery = std::chrono::steady_clock::now();

        double timeDataQuery = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndQuery - time_StartQuery).count();
        vdDataQuery_ms.push_back(timeDataQuery);
#endif
    }

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartEstSim3_2 = std::chrono::steady_clock::now();
#endif
    // Check the BoW candidates if the geometric candidate list is empty
    //Loop candidates
    if(!bLoopDetectedInKF && !vpLoopBowCand.empty())
    {
        mbLoopDetected = DetectCommonRegionsFromBoW(vpLoopBowCand,
                                                    mpLoopMatchedKF,
                                                    mpLoopLastCurrentKF,
                                                    mg2oLoopSlw,
                                                    mnLoopNumCoincidences,
                                                    mvpLoopMPs,
                                                    mvpLoopMatchedMPs,
                                                    bUseLightGlueLoop,
                                                    bUseXFeatVocabLoop ? &vXFeatLoopGroups : nullptr);
    }
    if(CheckFinish())
    {
        mpCurrentKF->SetErase();
        mpCurrentKF->mbCurrentPlaceRecognition = false;
        return false;
    }
    // Merge candidates
    if(!bMergeDetectedInKF && !vpMergeBowCand.empty())
    {
        mbMergeDetected = DetectCommonRegionsFromBoW(vpMergeBowCand,
                                                     mpMergeMatchedKF,
                                                     mpMergeLastCurrentKF,
                                                     mg2oMergeSlw,
                                                     mnMergeNumCoincidences,
                                                     mvpMergeMPs,
                                                     mvpMergeMatchedMPs,
                                                     false);
    }

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndEstSim3_2 = std::chrono::steady_clock::now();

        timeEstSim3 += std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndEstSim3_2 - time_StartEstSim3_2).count();
        vdEstSim3_ms.push_back(timeEstSim3);
#endif

    AddKeyFrameToDatabases(mpCurrentKF);

    if(mbMergeDetected || mbLoopDetected)
    {
        return true;
    }

    mpCurrentKF->SetErase();
    mpCurrentKF->mbCurrentPlaceRecognition = false;

    return false;
}

bool LoopClosing::DetectAndReffineSim3FromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches,
                                                 std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs)
{
    if(CheckFinish())
        return false;

    set<MapPoint*> spAlreadyMatchedMPs;
    nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);
    if(CheckFinish())
        return false;

    int nProjMatches = 30;
    int nProjOptMatches = 50;
    int nProjMatchesRep = 100;

    if(nNumProjMatches >= nProjMatches)
    {
        //Verbose::PrintMess("Sim3 reffine: There are " + to_string(nNumProjMatches) + " initial matches ", Verbose::VERBOSITY_DEBUG);
        Sophus::SE3d mTwm = pMatchedKF->GetPoseInverse().cast<double>();
        g2o::Sim3 gSwm(mTwm.unit_quaternion(),mTwm.translation(),1.0);
        g2o::Sim3 gScm = gScw * gSwm;
        Eigen::Matrix<double, 7, 7> mHessian7x7;

        bool bFixedScale = mbFixScale;       // TODO CHECK; Solo para el monocular inertial
        if(mpTracker->mSensor==System::IMU_MONOCULAR && !pCurrentKF->GetMap()->GetIniertialBA2())
            bFixedScale=false;
        if(CheckFinish())
            return false;
        int numOptMatches = Optimizer::OptimizeSim3(mpCurrentKF, pMatchedKF, vpMatchedMPs, gScm, 10, bFixedScale, mHessian7x7, true);
        if(CheckFinish())
            return false;

        //Verbose::PrintMess("Sim3 reffine: There are " + to_string(numOptMatches) + " matches after of the optimization ", Verbose::VERBOSITY_DEBUG);

        if(numOptMatches > nProjOptMatches)
        {
            g2o::Sim3 gScw_estimation(gScw.rotation(), gScw.translation(),1.0);

            vector<MapPoint*> vpMatchedMP;
            vpMatchedMP.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));

            if(CheckFinish())
                return false;
            nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw_estimation, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);
            if(CheckFinish())
                return false;
            if(nNumProjMatches >= nProjMatchesRep)
            {
                gScw = gScw_estimation;
                return true;
            }
        }
    }
    return false;
}

bool LoopClosing::DetectCommonRegionsFromBoW(std::vector<KeyFrame*> &vpBowCand, KeyFrame* &pMatchedKF2, KeyFrame* &pLastCurrentKF, g2o::Sim3 &g2oScw,
                                             int &nNumCoincidences, std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs,
                                             bool bUseLightGlueInitialMatches,
                                             const std::vector<XFeatKeyFrameDatabase::LoopCandidateGroup>* pvXFeatLoopGroups)
{
    if(CheckFinish())
        return false;

    int nBoWMatches = 20;
    int nBoWInliers = 15;
    int nSim3Inliers = 20;
    int nProjMatches = 50;
    int nProjOptMatches = 80;

    set<KeyFrame*> spConnectedKeyFrames = mpCurrentKF->GetConnectedKeyFrames();

    int nNumCovisibles = 10;

    ORBmatcher matcherBoW(0.9, true);
    ORBmatcher matcher(0.75, true);

    // Varibles to select the best numbe
    KeyFrame* pBestMatchedKF;
    KeyFrame* pBestBowCandidateKF = nullptr;
    int nBestMatchesReproj = 0;
    int nBestNumCoindicendes = 0;
    int nBestCandidateRank = -1;
    int nBestInitialMatches = 0;
    int nBestDirectRawMatches = 0;
    int nBestDirectAssigned = 0;
    int nBestLightGlueCalls = 0;
    int nBestDirectFInliers = 0;
    int nBestDirectFRejects = 0;
    int nBestGroupSize = 0;
    float fBestGroupScore = 0.0f;
    int nBestGroupPreselectKFs = 0;
    int nBestGroupProjectionKFs = 0;
    int nBestSim3Inliers = 0;
    int nBestProjMatches = -1;
    int nBestOptMatches = -1;
    int nBestProjOptMatches = -1;
    int nBestRefillCandidates = 0;
    int nBestRefillAdded = 0;
    g2o::Sim3 g2oBestScw;
    std::vector<MapPoint*> vpBestMapPoints;
    std::vector<MapPoint*> vpBestMatchedMapPoints;

    int numCandidates = vpBowCand.size();
    vector<int> vnStage(numCandidates, 0);
    vector<int> vnMatchesStage(numCandidates, 0);
    LoopLightGlueNearMiss bestNearMiss;
    LoopLightGlueNearMiss bestSim3RejectedNearMiss;
    std::unordered_map<KeyFrame*, const XFeatKeyFrameDatabase::LoopCandidateGroup*> mXFeatLoopGroupByKF;
    if(pvXFeatLoopGroups)
    {
        mXFeatLoopGroupByKF.reserve(pvXFeatLoopGroups->size());
        for(const XFeatKeyFrameDatabase::LoopCandidateGroup& group : *pvXFeatLoopGroups)
        {
            if(group.pRepresentative && !group.pRepresentative->isBad())
                mXFeatLoopGroupByKF[group.pRepresentative] = &group;
        }
    }

    //Verbose::PrintMess("BoW candidates: There are " + to_string(vpBowCand.size()) + " possible candidates ", Verbose::VERBOSITY_DEBUG);
    for(int candidateRank = 0; candidateRank < numCandidates; ++candidateRank)
    {
        KeyFrame* pKFi = vpBowCand[static_cast<size_t>(candidateRank)];
        if(CheckFinish())
            return false;

        if(!pKFi || pKFi->isBad())
            continue;

        const XFeatKeyFrameDatabase::LoopCandidateGroup* pLoopGroup = nullptr;
        const auto groupIt = mXFeatLoopGroupByKF.find(pKFi);
        if(groupIt != mXFeatLoopGroupByKF.end())
            pLoopGroup = groupIt->second;

        int loopLgCalls = 0;
        int loopLgDirectRawMatches = 0;
        int loopLgDirectAssigned = 0;
        int loopLgDirectFInliers = 0;
        int loopLgDirectFRejects = 0;
        int loopSim3Inliers = 0;
        int loopProjMatches = -1;
        int loopOptMatches = -1;
        int loopProjOptMatches = -1;
        int loopCovValid = 0;
        int loopRefillCandidates = 0;
        int loopRefillAdded = 0;
        int loopGroupPreselectKFs = 0;
        int loopGroupProjectionKFs = 0;
        bool loopSim3Converged = false;
        LoopLightGlueNearMiss loopNearMiss;
        if(bUseLightGlueInitialMatches)
        {
            loopNearMiss.valid = true;
            loopNearMiss.candidateRank = candidateRank;
            loopNearMiss.candidateKF = static_cast<long long>(pKFi->mnId);
            loopNearMiss.matchedKF = static_cast<long long>(pKFi->mnId);
            loopNearMiss.failStage = "initial_matches";
            loopNearMiss.failStageRank = 0;
            loopNearMiss.groupSize = pLoopGroup ? static_cast<int>(pLoopGroup->vpGroupKFs.size()) : 0;
            loopNearMiss.groupScore = pLoopGroup ? pLoopGroup->groupScore : 0.0f;
        }

        // std::cout << "KF candidate: " << pKFi->mnId << std::endl;
        // Current KF against KF with covisibles version
        std::vector<KeyFrame*> vpCovKFi = pKFi->GetBestCovisibilityKeyFrames(nNumCovisibles);
        if(vpCovKFi.empty())
        {
            std::cout << "Covisible list empty" << std::endl;
            vpCovKFi.push_back(pKFi);
        }
        else
        {
            vpCovKFi.push_back(vpCovKFi[0]);
            vpCovKFi[0] = pKFi;
        }


        bool bAbortByNearKF = false;
        const int nNearKFCheckCount =
            (bUseLightGlueInitialMatches && pLoopGroup) ? 1 : static_cast<int>(vpCovKFi.size());
        for(int j=0; j<nNearKFCheckCount; ++j)
        {
            if(spConnectedKeyFrames.find(vpCovKFi[j]) != spConnectedKeyFrames.end())
            {
                bAbortByNearKF = true;
                break;
            }
        }
        if(bAbortByNearKF)
        {
            //std::cout << "Check BoW aborted because is close to the matched one " << std::endl;
            continue;
        }
        //std::cout << "Check BoW continue because is far to the matched one " << std::endl;

        std::vector<std::vector<MapPoint*> > vvpMatchedMPs;
        vvpMatchedMPs.resize(vpCovKFi.size());
        std::vector<std::vector<float> > vvMatchedScores;
        vvMatchedScores.resize(vpCovKFi.size());
        std::set<MapPoint*> spMatchedMPi;
        int numBoWMatches = 0;

        KeyFrame* pMostBoWMatchesKF = pKFi;
        int nMostBoWNumMatches = 0;

        std::vector<MapPoint*> vpMatchedPoints = std::vector<MapPoint*>(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));
        std::vector<KeyFrame*> vpKeyFrameMatchedMP = std::vector<KeyFrame*>(mpCurrentKF->GetMapPointMatches().size(), static_cast<KeyFrame*>(NULL));

        if(bUseLightGlueInitialMatches)
        {
            std::vector<std::vector<LoopLightGlueCandidateMatch>> vvLoopLgCandidateMatches(vpCovKFi.size());
            int nMostBoWWindowIdx = -1;

            auto collectWindowMatches = [&](const int windowIdx,
                                             int& nFundamentalInliers,
                                             int& nFundamentalRejects) -> int {
                nFundamentalInliers = 0;
                nFundamentalRejects = 0;
                if(windowIdx < 0 || windowIdx >= static_cast<int>(vvpMatchedMPs.size()) ||
                   windowIdx >= static_cast<int>(vpCovKFi.size()) || !vpCovKFi[windowIdx])
                    return 0;

                const std::vector<MapPoint*>& vMatchedMPs = vvpMatchedMPs[windowIdx];
                const std::vector<float>& vScores = vvMatchedScores[windowIdx];
                std::vector<LoopLightGlueCandidateMatch> vWindowMatches;
                vWindowMatches.reserve(vMatchedMPs.size());
                for(int k = 0; k < static_cast<int>(vMatchedMPs.size()); ++k)
                {
                    MapPoint* pMP = vMatchedMPs[k];
                    if(!pMP || pMP->isBad())
                        continue;

                    float score = 0.0f;
                    if(k < static_cast<int>(vScores.size()) && vScores[k] >= 0.0f)
                        score = vScores[k];

                    tuple<int,int> indexes = pMP->GetIndexInKeyFrame(vpCovKFi[windowIdx]);
                    const int sourceIdx = get<0>(indexes);
                    vWindowMatches.push_back(LoopLightGlueCandidateMatch{k, sourceIdx, pMP, vpCovKFi[windowIdx], score});
                }

                std::vector<LoopLightGlueCandidateMatch> vFilteredWindowMatches;
                FilterLoopLightGlueMatchesByFundamental(mpCurrentKF,
                                                        vpCovKFi[windowIdx],
                                                        vWindowMatches,
                                                        vFilteredWindowMatches,
                                                        nFundamentalInliers,
                                                        nFundamentalRejects);
                vvLoopLgCandidateMatches[static_cast<size_t>(windowIdx)] = std::move(vFilteredWindowMatches);
                nFundamentalInliers = static_cast<int>(vvLoopLgCandidateMatches[static_cast<size_t>(windowIdx)].size());
                return nFundamentalInliers;
            };

            auto runLightGlueWindow = [&](const int windowIdx) -> int {
                if(windowIdx < 0 || windowIdx >= static_cast<int>(vpCovKFi.size()) ||
                   !vpCovKFi[windowIdx] || vpCovKFi[windowIdx]->isBad())
                    return -1;

                if(CheckFinish())
                    return -1;

                ++loopLgCalls;
                int numAccepted = 0;
                int windowRawMatches = 0;
                int windowAssignedMatches = 0;
                int windowFundamentalInliers = 0;
                int windowFundamentalRejects = 0;
                try
                {
                    static XFeatLighterGlueMatcher lightGlueMatcher;
                    lightGlueMatcher.SearchByLightGlue(mpCurrentKF,
                                                       vpCovKFi[windowIdx],
                                                       vvpMatchedMPs[windowIdx],
                                                       GetXFeatLightGlueLoopConfidence(),
                                                       &vvMatchedScores[windowIdx]);
                    const XFeatLighterGlueMatcher::Stats& lgStats = lightGlueMatcher.LastStats();
                    windowRawMatches = lgStats.lg_raw_matches;
                    windowAssignedMatches = lgStats.one_to_one;
                    numAccepted = collectWindowMatches(windowIdx,
                                                       windowFundamentalInliers,
                                                       windowFundamentalRejects);
                }
                catch(const std::exception& e)
                {
                    std::cerr << "[LoopLG][Fallback]"
                              << " current_kf=" << mpCurrentKF->mnId
                              << " candidate_kf=" << pKFi->mnId
                              << " window_kf=" << vpCovKFi[windowIdx]->mnId
                              << " error=" << e.what()
                              << std::endl;
                    const int nFallbackMatches = matcherBoW.SearchByBoW(mpCurrentKF, vpCovKFi[windowIdx], vvpMatchedMPs[windowIdx]);
                    windowRawMatches = nFallbackMatches;
                    windowAssignedMatches = nFallbackMatches;
                    vvMatchedScores[windowIdx].assign(vvpMatchedMPs[windowIdx].size(), 0.0f);
                    numAccepted = collectWindowMatches(windowIdx,
                                                       windowFundamentalInliers,
                                                       windowFundamentalRejects);
                }

                if(numAccepted > nMostBoWNumMatches)
                {
                    nMostBoWNumMatches = numAccepted;
                    pMostBoWMatchesKF = vpCovKFi[windowIdx];
                    nMostBoWWindowIdx = windowIdx;
                    loopLgDirectRawMatches = windowRawMatches;
                    loopLgDirectAssigned = windowAssignedMatches;
                    loopLgDirectFInliers = windowFundamentalInliers;
                    loopLgDirectFRejects = windowFundamentalRejects;
                }
                return numAccepted;
            };

            const int nInitialMatchWindows = std::min<int>(1, static_cast<int>(vpCovKFi.size()));
            loopGroupPreselectKFs = nInitialMatchWindows;
            for(int windowIdx = 0; windowIdx < nInitialMatchWindows; ++windowIdx)
                runLightGlueWindow(windowIdx);
            if(CheckFinish())
                return false;
            if(nMostBoWWindowIdx >= 0 &&
               nMostBoWWindowIdx < static_cast<int>(vvLoopLgCandidateMatches.size()))
            {
                numBoWMatches = AssignLoopLightGlueMatches(vvLoopLgCandidateMatches[static_cast<size_t>(nMostBoWWindowIdx)],
                                                           static_cast<int>(vpMatchedPoints.size()),
                                                           vpMatchedPoints,
                                                           vpKeyFrameMatchedMP);
            }
        }
        else
        {
            for(int j=0; j<vpCovKFi.size(); ++j)
            {
                if(!vpCovKFi[j] || vpCovKFi[j]->isBad())
                    continue;

                int num = matcherBoW.SearchByBoW(mpCurrentKF, vpCovKFi[j], vvpMatchedMPs[j]);

                if (num > nMostBoWNumMatches)
                {
                    nMostBoWNumMatches = num;
                    pMostBoWMatchesKF = vpCovKFi[j];
                }
            }

            for(int j=0; j<vpCovKFi.size(); ++j)
            {
                for(int k=0; k < vvpMatchedMPs[j].size(); ++k)
                {
                    MapPoint* pMPi_j = vvpMatchedMPs[j][k];
                    if(!pMPi_j || pMPi_j->isBad())
                        continue;

                    if(spMatchedMPi.find(pMPi_j) == spMatchedMPi.end())
                    {
                        spMatchedMPi.insert(pMPi_j);
                        numBoWMatches++;

                        vpMatchedPoints[k]= pMPi_j;
                        vpKeyFrameMatchedMP[k] = vpCovKFi[j];
                    }
                }
            }
        }

        if(bUseLightGlueInitialMatches)
        {
            loopNearMiss.initialMatches = numBoWMatches;
            loopNearMiss.directRaw = loopLgDirectRawMatches;
            loopNearMiss.directAssigned = loopLgDirectAssigned;
            loopNearMiss.directFInliers = loopLgDirectFInliers;
            loopNearMiss.directFRejects = loopLgDirectFRejects;
            loopNearMiss.groupPreselectKFs = loopGroupPreselectKFs;
            loopNearMiss.lgCalls = loopLgCalls;
            loopNearMiss.matchedKF = pMostBoWMatchesKF ? static_cast<long long>(pMostBoWMatchesKF->mnId) : -1;
        }

        //pMostBoWMatchesKF = vpCovKFi[pMostBoWMatchesKF];

        if(numBoWMatches >= nBoWMatches) // TODO pick a good threshold
        {
            if(bUseLightGlueInitialMatches)
            {
                loopNearMiss.failStage = "sim3_not_converged";
                loopNearMiss.failStageRank = 1;
            }

            // Geometric validation
            bool bFixedScale = mbFixScale;
            if(mpTracker->mSensor==System::IMU_MONOCULAR && !mpCurrentKF->GetMap()->GetIniertialBA2())
                bFixedScale=false;

            Sim3Solver solver = Sim3Solver(mpCurrentKF, pMostBoWMatchesKF, vpMatchedPoints, bFixedScale, vpKeyFrameMatchedMP);
            solver.SetRansacParameters(0.99, nBoWInliers, 300); // at least 15 inliers

            bool bNoMore = false;
            vector<bool> vbInliers;
            int nInliers = 0;
            bool bConverge = false;
            Eigen::Matrix4f mTcm;
            while(!bConverge && !bNoMore && !CheckFinish())
            {
                mTcm = solver.iterate(20,bNoMore, vbInliers, nInliers, bConverge);
                //Verbose::PrintMess("BoW guess: Solver achieve " + to_string(nInliers) + " geometrical inliers among " + to_string(nBoWInliers) + " BoW matches", Verbose::VERBOSITY_DEBUG);
            }
            if(CheckFinish())
                return false;

            loopSim3Converged = bConverge;
            loopSim3Inliers = nInliers;
            if(bUseLightGlueInitialMatches)
            {
                loopNearMiss.sim3Converged = bConverge ? 1 : 0;
                loopNearMiss.sim3Inliers = nInliers;
            }

            if(bConverge)
            {
                if(bUseLightGlueInitialMatches)
                {
                    loopNearMiss.failStage = "projection_matches";
                    loopNearMiss.failStageRank = 2;
                }

                //std::cout << "Check BoW: SolverSim3 converged" << std::endl;

                //Verbose::PrintMess("BoW guess: Convergende with " + to_string(nInliers) + " geometrical inliers among " + to_string(nBoWInliers) + " BoW matches", Verbose::VERBOSITY_DEBUG);
                // Match by reprojection
                vpCovKFi.clear();
                std::set<KeyFrame*> spProjectionKFs;
                constexpr size_t kXFeatLoopGroupProjectionMaxKFs = 15;
                if(bUseLightGlueInitialMatches && pLoopGroup)
                {
                    for(KeyFrame* pGroupKF : pLoopGroup->vpGroupKFs)
                    {
                        if(spConnectedKeyFrames.find(pGroupKF) != spConnectedKeyFrames.end())
                            continue;
                        AppendUniqueLoopKeyFrame(vpCovKFi,
                                                 spProjectionKFs,
                                                 pGroupKF,
                                                 kXFeatLoopGroupProjectionMaxKFs);
                    }
                }
                AppendUniqueLoopKeyFrame(vpCovKFi,
                                         spProjectionKFs,
                                         pMostBoWMatchesKF,
                                         kXFeatLoopGroupProjectionMaxKFs);
                const std::vector<KeyFrame*> vpBestCovisibles =
                    pMostBoWMatchesKF->GetBestCovisibilityKeyFrames(nNumCovisibles);
                for(KeyFrame* pCovKF : vpBestCovisibles)
                {
                    AppendUniqueLoopKeyFrame(vpCovKFi,
                                             spProjectionKFs,
                                             pCovKF,
                                             kXFeatLoopGroupProjectionMaxKFs);
                }
                loopGroupProjectionKFs = static_cast<int>(vpCovKFi.size());
                if(bUseLightGlueInitialMatches)
                    loopNearMiss.groupProjectionKFs = loopGroupProjectionKFs;
                set<KeyFrame*> spCheckKFs(vpCovKFi.begin(), vpCovKFi.end());

                //std::cout << "There are " << vpCovKFi.size() <<" near KFs" << std::endl;

                set<MapPoint*> spMapPoints;
                vector<MapPoint*> vpMapPoints;
                vector<KeyFrame*> vpKeyFrames;
                for(KeyFrame* pCovKFi : vpCovKFi)
                {
                    for(MapPoint* pCovMPij : pCovKFi->GetMapPointMatches())
                    {
                        if(!pCovMPij || pCovMPij->isBad())
                            continue;

                        if(spMapPoints.find(pCovMPij) == spMapPoints.end())
                        {
                            spMapPoints.insert(pCovMPij);
                            vpMapPoints.push_back(pCovMPij);
                            vpKeyFrames.push_back(pCovKFi);
                        }
                    }
                }

                //std::cout << "There are " << vpKeyFrames.size() <<" KFs which view all the mappoints" << std::endl;

                g2o::Sim3 gScm(solver.GetEstimatedRotation().cast<double>(),solver.GetEstimatedTranslation().cast<double>(), (double) solver.GetEstimatedScale());
                KeyFrame* pLoopValidationKF = (bUseLightGlueInitialMatches && pMostBoWMatchesKF) ? pMostBoWMatchesKF : pKFi;
                g2o::Sim3 gSmw(pMostBoWMatchesKF->GetRotation().cast<double>(),pMostBoWMatchesKF->GetTranslation().cast<double>(),1.0);
                g2o::Sim3 gScw = gScm*gSmw; // Similarity matrix of current from the world position
                Sophus::Sim3f mScw = Converter::toSophus(gScw);

                vector<MapPoint*> vpMatchedMP;
                vpMatchedMP.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));
                vector<KeyFrame*> vpMatchedKF;
                vpMatchedKF.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<KeyFrame*>(NULL));
                if(CheckFinish())
                    return false;
                int numProjMatches = matcher.SearchByProjection(mpCurrentKF, mScw, vpMapPoints, vpKeyFrames, vpMatchedMP, vpMatchedKF, 8, 1.5);
                if(CheckFinish())
                    return false;
                loopProjMatches = numProjMatches;
                if(bUseLightGlueInitialMatches)
                    loopNearMiss.projMatches = numProjMatches;
                //cout <<"BoW: " << numProjMatches << " matches between " << vpMapPoints.size() << " points with coarse Sim3" << endl;

                if(numProjMatches >= nProjMatches)
                {
                    if(bUseLightGlueInitialMatches && pLoopGroup &&
                       ShouldRunLoopGroupRefill(numBoWMatches, loopSim3Inliers, numProjMatches))
                    {
                        LoopGroupRefillStats refillStats;
                        const int nRefillAdded =
                            RefillLoopMatchesFromXFeatGroup(mpCurrentKF,
                                                            *pLoopGroup,
                                                            mScw,
                                                            vpMatchedMP,
                                                            vpMatchedKF,
                                                            vpMapPoints,
                                                            vpKeyFrames,
                                                            spMapPoints,
                                                            refillStats);
                        loopRefillCandidates = refillStats.candidateMatches;
                        loopRefillAdded = nRefillAdded;
                        if(nRefillAdded > 0)
                        {
                            numProjMatches += nRefillAdded;
                            loopProjMatches = numProjMatches;
                            loopNearMiss.projMatches = numProjMatches;
                        }
                        loopNearMiss.refillCandidates = loopRefillCandidates;
                        loopNearMiss.refillAdded = loopRefillAdded;
                    }

                    if(bUseLightGlueInitialMatches)
                    {
                        loopNearMiss.failStage = "optimize_sim3";
                        loopNearMiss.failStageRank = 3;
                    }

                    // Optimize Sim3 transformation with every matches
                    Eigen::Matrix<double, 7, 7> mHessian7x7;

                    bool bFixedScale = mbFixScale;
                    if(mpTracker->mSensor==System::IMU_MONOCULAR && !mpCurrentKF->GetMap()->GetIniertialBA2())
                        bFixedScale=false;

                    if(CheckFinish())
                        return false;
                    int numOptMatches = Optimizer::OptimizeSim3(mpCurrentKF, pLoopValidationKF, vpMatchedMP, gScm, 10, bFixedScale, mHessian7x7, true);
                    if(CheckFinish())
                        return false;
                    loopOptMatches = numOptMatches;
                    if(bUseLightGlueInitialMatches)
                        loopNearMiss.optMatches = numOptMatches;

                    if(numOptMatches >= nSim3Inliers)
                    {
                        if(bUseLightGlueInitialMatches)
                        {
                            loopNearMiss.failStage = "projection_opt_matches";
                            loopNearMiss.failStageRank = 4;
                        }

                        g2o::Sim3 gSmw(pMostBoWMatchesKF->GetRotation().cast<double>(),pMostBoWMatchesKF->GetTranslation().cast<double>(),1.0);
                        g2o::Sim3 gScw = gScm*gSmw; // Similarity matrix of current from the world position
                        Sophus::Sim3f mScw = Converter::toSophus(gScw);

                        vector<MapPoint*> vpMatchedMP;
                        vpMatchedMP.resize(mpCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));
                        if(CheckFinish())
                            return false;
                        int numProjOptMatches = matcher.SearchByProjection(mpCurrentKF, mScw, vpMapPoints, vpMatchedMP, 5, 1.0);
                        if(CheckFinish())
                            return false;
                        loopProjOptMatches = numProjOptMatches;
                        if(bUseLightGlueInitialMatches)
                            loopNearMiss.projOptMatches = numProjOptMatches;

                        if(numProjOptMatches >= nProjOptMatches)
                        {
                            if(bUseLightGlueInitialMatches)
                            {
                                loopNearMiss.failStage = "covisible_validation";
                                loopNearMiss.failStageRank = 5;
                            }

                            int max_x = -1, min_x = 1000000;
                            int max_y = -1, min_y = 1000000;
                            for(MapPoint* pMPi : vpMatchedMP)
                            {
                                if(!pMPi || pMPi->isBad())
                                {
                                    continue;
                                }

                                tuple<size_t,size_t> indexes = pMPi->GetIndexInKeyFrame(pLoopValidationKF);
                                int index = get<0>(indexes);
                                if(index >= 0)
                                {
                                    int coord_x = pLoopValidationKF->mvKeysUn[index].pt.x;
                                    if(coord_x < min_x)
                                    {
                                        min_x = coord_x;
                                    }
                                    if(coord_x > max_x)
                                    {
                                        max_x = coord_x;
                                    }
                                    int coord_y = pLoopValidationKF->mvKeysUn[index].pt.y;
                                    if(coord_y < min_y)
                                    {
                                        min_y = coord_y;
                                    }
                                    if(coord_y > max_y)
                                    {
                                        max_y = coord_y;
                                    }
                                }
                            }

                            int nNumKFs = 0;
                            //vpMatchedMPs = vpMatchedMP;
                            //vpMPs = vpMapPoints;
                            // Check the Sim3 transformation with the current KeyFrame covisibles
                            vector<KeyFrame*> vpCurrentCovKFs = mpCurrentKF->GetBestCovisibilityKeyFrames(nNumCovisibles);

                            int j = 0;
                            while(nNumKFs < 3 && j<vpCurrentCovKFs.size() && !CheckFinish())
                            {
                                KeyFrame* pKFj = vpCurrentCovKFs[j];
                                Sophus::SE3d mTjc = (pKFj->GetPose() * mpCurrentKF->GetPoseInverse()).cast<double>();
                                g2o::Sim3 gSjc(mTjc.unit_quaternion(),mTjc.translation(),1.0);
                                g2o::Sim3 gSjw = gSjc * gScw;
                                int numProjMatches_j = 0;
                                vector<MapPoint*> vpMatchedMPs_j;
                                bool bValid = DetectCommonRegionsFromLastKF(pKFj,pMostBoWMatchesKF, gSjw,numProjMatches_j, vpMapPoints, vpMatchedMPs_j);

                                if(bValid)
                                {
                                    Sophus::SE3f Tc_w = mpCurrentKF->GetPose();
                                    Sophus::SE3f Tw_cj = pKFj->GetPoseInverse();
                                    Sophus::SE3f Tc_cj = Tc_w * Tw_cj;
                                    Eigen::Vector3f vector_dist = Tc_cj.translation();
                                    nNumKFs++;
                                }
                                j++;
                            }
                            if(CheckFinish())
                                return false;

                            loopCovValid = nNumKFs;
                            if(bUseLightGlueInitialMatches)
                                loopNearMiss.covValid = nNumKFs;

                            if(nNumKFs < 3)
                            {
                                vnStage[candidateRank] = 8;
                                vnMatchesStage[candidateRank] = nNumKFs;
                            }
                            else if(bUseLightGlueInitialMatches)
                            {
                                loopNearMiss.failStage = "accepted_candidate";
                                loopNearMiss.failStageRank = 6;
                            }

                            if(nBestMatchesReproj < numProjOptMatches)
                            {
                                nBestMatchesReproj = numProjOptMatches;
                                nBestNumCoindicendes = nNumKFs;
                                pBestBowCandidateKF = pKFi;
                                pBestMatchedKF = pMostBoWMatchesKF;
                                nBestCandidateRank = candidateRank;
                                nBestInitialMatches = numBoWMatches;
                                nBestDirectRawMatches = loopLgDirectRawMatches;
                                nBestDirectAssigned = loopLgDirectAssigned;
                                nBestLightGlueCalls = loopLgCalls;
                                nBestDirectFInliers = loopLgDirectFInliers;
                                nBestDirectFRejects = loopLgDirectFRejects;
                                nBestGroupSize = pLoopGroup ? static_cast<int>(pLoopGroup->vpGroupKFs.size()) : 0;
                                fBestGroupScore = pLoopGroup ? pLoopGroup->groupScore : 0.0f;
                                nBestGroupPreselectKFs = loopGroupPreselectKFs;
                                nBestGroupProjectionKFs = loopGroupProjectionKFs;
                                nBestSim3Inliers = loopSim3Inliers;
                                nBestProjMatches = loopProjMatches;
                                nBestOptMatches = loopOptMatches;
                                nBestProjOptMatches = loopProjOptMatches;
                                nBestRefillCandidates = loopRefillCandidates;
                                nBestRefillAdded = loopRefillAdded;
                                g2oBestScw = gScw;
                                vpBestMapPoints = vpMapPoints;
                                vpBestMatchedMapPoints = vpMatchedMP;
                            }
                        }
                    }
                }
            }
            /*else
            {
                Verbose::PrintMess("BoW candidate: it don't match with the current one", Verbose::VERBOSITY_DEBUG);
            }*/
        }

        if(bUseLightGlueInitialMatches)
        {
            if(IsBetterLoopLightGlueNearMiss(loopNearMiss, bestNearMiss))
                bestNearMiss = loopNearMiss;
            if(loopNearMiss.sim3Converged != 0 &&
               loopNearMiss.failStageRank < 6 &&
               IsBetterLoopLightGlueNearMiss(loopNearMiss, bestSim3RejectedNearMiss))
                bestSim3RejectedNearMiss = loopNearMiss;

            unique_lock<mutex> lock(mMutexXFeatLoopSummary);
            ++mnXFeatLoopLgCandidatesChecked;
            mnXFeatLoopLgCandidateMaxRank = std::max(mnXFeatLoopLgCandidateMaxRank, candidateRank);
            if(loopSim3Converged)
                ++mnXFeatLoopLgSim3ConvergedCandidates;
            mnXFeatLoopLgTotalCalls += loopLgCalls;
        }
    }

    if(nBestMatchesReproj > 0)
    {
        pLastCurrentKF = mpCurrentKF;
        nNumCoincidences = nBestNumCoindicendes;
        pMatchedKF2 = pBestMatchedKF;
        pMatchedKF2->SetNotErase();
        g2oScw = g2oBestScw;
        vpMPs = vpBestMapPoints;
        vpMatchedMPs = vpBestMatchedMapPoints;

        const bool bAccepted = nNumCoincidences >= 3;
        if(bUseLightGlueInitialMatches && bAccepted)
        {
            std::cout << "[LoopLG][Accept]"
                      << " current_kf=" << mpCurrentKF->mnId
                      << " candidate_kf="
                      << (pBestBowCandidateKF ? static_cast<long long>(pBestBowCandidateKF->mnId) : -1)
                      << " candidate_rank=" << nBestCandidateRank
                      << " matched_kf=" << (pMatchedKF2 ? static_cast<long long>(pMatchedKF2->mnId) : -1)
                      << " initial_matches=" << nBestInitialMatches
                      << " direct_raw=" << nBestDirectRawMatches
                      << " direct_assigned=" << nBestDirectAssigned
                      << " direct_f_inliers=" << nBestDirectFInliers
                      << " direct_f_rejects=" << nBestDirectFRejects
                      << " group_size=" << nBestGroupSize
                      << " group_score=" << fBestGroupScore
                      << " group_preselect_kfs=" << nBestGroupPreselectKFs
                      << " group_projection_kfs=" << nBestGroupProjectionKFs
                      << " lg_calls=" << nBestLightGlueCalls
                      << " sim3_inliers=" << nBestSim3Inliers
                      << " proj_matches=" << nBestProjMatches
                      << " opt_matches=" << nBestOptMatches
                      << " proj_opt_matches=" << nBestProjOptMatches
                      << " refill_candidates=" << nBestRefillCandidates
                      << " refill_added=" << nBestRefillAdded
                      << " best_proj_matches=" << nBestMatchesReproj
                      << " cov_valid=" << nBestNumCoindicendes
                      << std::endl;

            unique_lock<mutex> lock(mMutexXFeatLoopSummary);
            ++mnXFeatLoopLgAcceptCount;
            mnXFeatLoopLgLastAcceptCurrentKF = static_cast<int>(mpCurrentKF->mnId);
            mnXFeatLoopLgLastAcceptCandidateKF =
                pBestBowCandidateKF ? static_cast<int>(pBestBowCandidateKF->mnId) : -1;
            mnXFeatLoopLgLastAcceptMatchedKF =
                pMatchedKF2 ? static_cast<int>(pMatchedKF2->mnId) : -1;
            mnXFeatLoopLgLastAcceptRank = nBestCandidateRank;
            mnXFeatLoopLgLastAcceptInitialMatches = nBestInitialMatches;
            mnXFeatLoopLgLastAcceptSim3Inliers = nBestSim3Inliers;
            mnXFeatLoopLgLastAcceptProjOptMatches = nBestProjOptMatches;
            mnXFeatLoopLgLastAcceptCovValid = nBestNumCoindicendes;

            const bool bBetterAcceptedLoop =
                nBestSim3Inliers > mnXFeatLoopLgBestAcceptSim3Inliers ||
                (nBestSim3Inliers == mnXFeatLoopLgBestAcceptSim3Inliers &&
                 nBestProjOptMatches > mnXFeatLoopLgBestAcceptProjOptMatches);
            if(bBetterAcceptedLoop)
            {
                mnXFeatLoopLgBestAcceptCurrentKF = static_cast<int>(mpCurrentKF->mnId);
                mnXFeatLoopLgBestAcceptCandidateKF =
                    pBestBowCandidateKF ? static_cast<int>(pBestBowCandidateKF->mnId) : -1;
                mnXFeatLoopLgBestAcceptMatchedKF =
                    pMatchedKF2 ? static_cast<int>(pMatchedKF2->mnId) : -1;
                mnXFeatLoopLgBestAcceptRank = nBestCandidateRank;
                mnXFeatLoopLgBestAcceptSim3Inliers = nBestSim3Inliers;
                mnXFeatLoopLgBestAcceptProjOptMatches = nBestProjOptMatches;
                mnXFeatLoopLgBestAcceptCovValid = nBestNumCoindicendes;
            }
        }
        else if(bUseLightGlueInitialMatches && ShouldPrintLoopVocabDebug(mpCurrentKF->mnId))
        {
            PrintLoopLightGlueNearMiss(mpCurrentKF->mnId,
                                       bestNearMiss,
                                       nBoWMatches,
                                       nBoWInliers,
                                       nSim3Inliers,
                                       nProjMatches,
                                       nProjOptMatches);
        }
        if(bUseLightGlueInitialMatches && !bAccepted && ShouldPrintLoopBestRejectedDebug(mpCurrentKF->mnId))
        {
            PrintLoopLightGlueBestRejected(mpCurrentKF->mnId,
                                           bestSim3RejectedNearMiss,
                                           nBoWMatches,
                                           nBoWInliers,
                                           nSim3Inliers,
                                           nProjMatches,
                                           nProjOptMatches);
        }
        return bAccepted;
    }
    else
    {
        int maxStage = -1;
        int maxMatched;
        for(int i=0; i<vnStage.size(); ++i)
        {
            if(vnStage[i] > maxStage)
            {
                maxStage = vnStage[i];
                maxMatched = vnMatchesStage[i];
            }
        }
        if(bUseLightGlueInitialMatches && ShouldPrintLoopVocabDebug(mpCurrentKF->mnId))
        {
            PrintLoopLightGlueNearMiss(mpCurrentKF->mnId,
                                       bestNearMiss,
                                       nBoWMatches,
                                       nBoWInliers,
                                       nSim3Inliers,
                                       nProjMatches,
                                       nProjOptMatches);
        }
        if(bUseLightGlueInitialMatches && ShouldPrintLoopBestRejectedDebug(mpCurrentKF->mnId))
        {
            PrintLoopLightGlueBestRejected(mpCurrentKF->mnId,
                                           bestSim3RejectedNearMiss,
                                           nBoWMatches,
                                           nBoWInliers,
                                           nSim3Inliers,
                                           nProjMatches,
                                           nProjOptMatches);
        }
    }
    return false;
}

bool LoopClosing::DetectCommonRegionsFromLastKF(KeyFrame* pCurrentKF, KeyFrame* pMatchedKF, g2o::Sim3 &gScw, int &nNumProjMatches,
                                                std::vector<MapPoint*> &vpMPs, std::vector<MapPoint*> &vpMatchedMPs)
{
    if(CheckFinish())
        return false;

    set<MapPoint*> spAlreadyMatchedMPs(vpMatchedMPs.begin(), vpMatchedMPs.end());
    nNumProjMatches = FindMatchesByProjection(pCurrentKF, pMatchedKF, gScw, spAlreadyMatchedMPs, vpMPs, vpMatchedMPs);
    if(CheckFinish())
        return false;

    int nProjMatches = 30;
    if(nNumProjMatches >= nProjMatches)
    {
        return true;
    }

    return false;
}

int LoopClosing::FindMatchesByProjection(KeyFrame* pCurrentKF, KeyFrame* pMatchedKFw, g2o::Sim3 &g2oScw,
                                         set<MapPoint*> &spMatchedMPinOrigin, vector<MapPoint*> &vpMapPoints,
                                         vector<MapPoint*> &vpMatchedMapPoints)
{
    if(CheckFinish())
        return 0;

    int nNumCovisibles = 10;
    vector<KeyFrame*> vpCovKFm = pMatchedKFw->GetBestCovisibilityKeyFrames(nNumCovisibles);
    int nInitialCov = vpCovKFm.size();
    vpCovKFm.push_back(pMatchedKFw);
    set<KeyFrame*> spCheckKFs(vpCovKFm.begin(), vpCovKFm.end());
    set<KeyFrame*> spCurrentCovisbles = pCurrentKF->GetConnectedKeyFrames();
    if(nInitialCov < nNumCovisibles)
    {
        for(int i=0; i<nInitialCov; ++i)
        {
            if(CheckFinish())
                return 0;

            vector<KeyFrame*> vpKFs = vpCovKFm[i]->GetBestCovisibilityKeyFrames(nNumCovisibles);
            int nInserted = 0;
            int j = 0;
            while(j < vpKFs.size() && nInserted < nNumCovisibles)
            {
                if(spCheckKFs.find(vpKFs[j]) == spCheckKFs.end() && spCurrentCovisbles.find(vpKFs[j]) == spCurrentCovisbles.end())
                {
                    spCheckKFs.insert(vpKFs[j]);
                    ++nInserted;
                }
                ++j;
            }
            vpCovKFm.insert(vpCovKFm.end(), vpKFs.begin(), vpKFs.end());
        }
    }
    set<MapPoint*> spMapPoints;
    vpMapPoints.clear();
    vpMatchedMapPoints.clear();
    for(KeyFrame* pKFi : vpCovKFm)
    {
        if(CheckFinish())
            return 0;

        for(MapPoint* pMPij : pKFi->GetMapPointMatches())
        {
            if(!pMPij || pMPij->isBad())
                continue;

            if(spMapPoints.find(pMPij) == spMapPoints.end())
            {
                spMapPoints.insert(pMPij);
                vpMapPoints.push_back(pMPij);
            }
        }
    }

    Sophus::Sim3f mScw = Converter::toSophus(g2oScw);
    ORBmatcher matcher(0.9, true);

    vpMatchedMapPoints.resize(pCurrentKF->GetMapPointMatches().size(), static_cast<MapPoint*>(NULL));
    if(CheckFinish())
        return 0;
    int num_matches = matcher.SearchByProjection(pCurrentKF, mScw, vpMapPoints, vpMatchedMapPoints, 3, 1.5);

    return num_matches;
}

void LoopClosing::CorrectLoop()
{
    //cout << "Loop detected!" << endl;
    {
        unique_lock<mutex> lock(mMutexXFeatLoopSummary);
        ++mnXFeatLoopCorrectionCount;
        mnXFeatLoopLastCorrectionCurrentKF =
            mpCurrentKF ? static_cast<int>(mpCurrentKF->mnId) : -1;
        mnXFeatLoopLastCorrectionMatchedKF =
            mpLoopMatchedKF ? static_cast<int>(mpLoopMatchedKF->mnId) : -1;
        mnXFeatLoopLastCorrectionCoincidences = mnLoopNumCoincidences;
    }

    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    mpLocalMapper->RequestStop();
    mpLocalMapper->EmptyQueue(); // Proccess keyframes in the queue

    // If a Global Bundle Adjustment is running, abort it
    if(isRunningGBA())
    {
        cout << "Stoping Global Bundle Adjustment...";
        {
            unique_lock<mutex> summaryLock(mMutexXFeatLoopSummary);
            ++mnXFeatGbaStopRequestedCount;
        }
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;

        if(mpThreadGBA)
        {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
        cout << "  Done!!" << endl;
    }

    // Wait until Local Mapping has effectively stopped
    while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
    {
        usleep(1000);
    }

    // Ensure current keyframe is updated
    //cout << "Start updating connections" << endl;
    //assert(mpCurrentKF->GetMap()->CheckEssentialGraph());
    mpCurrentKF->UpdateConnections();
    //assert(mpCurrentKF->GetMap()->CheckEssentialGraph());

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

    //std::cout << "Loop: number of connected KFs -> " + to_string(mvpCurrentConnectedKFs.size()) << std::endl;

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    CorrectedSim3[mpCurrentKF]=mg2oLoopScw;
    Sophus::SE3f Twc = mpCurrentKF->GetPoseInverse();
    Sophus::SE3f Tcw = mpCurrentKF->GetPose();
    g2o::Sim3 g2oScw(Tcw.unit_quaternion().cast<double>(),Tcw.translation().cast<double>(),1.0);
    NonCorrectedSim3[mpCurrentKF]=g2oScw;

    // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
    Sophus::SE3d correctedTcw(mg2oLoopScw.rotation(),mg2oLoopScw.translation() / mg2oLoopScw.scale());
    mpCurrentKF->SetPose(correctedTcw.cast<float>());

    Map* pLoopMap = mpCurrentKF->GetMap();

#ifdef REGISTER_TIMES
    /*KeyFrame* pKF = mpCurrentKF;
    int numKFinLoop = 0;
    while(pKF && pKF->mnId > mpLoopMatchedKF->mnId)
    {
        pKF = pKF->GetParent();
        numKFinLoop += 1;
    }
    vnLoopKFs.push_back(numKFinLoop);*/

    std::chrono::steady_clock::time_point time_StartFusion = std::chrono::steady_clock::now();
#endif

    {
        // Get Map Mutex
        unique_lock<mutex> lock(pLoopMap->mMutexMapUpdate);

        const bool bImuInit = pLoopMap->isImuInitialized();

        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKFi = *vit;

            if(pKFi!=mpCurrentKF)
            {
                Sophus::SE3f Tiw = pKFi->GetPose();
                Sophus::SE3d Tic = (Tiw * Twc).cast<double>();
                g2o::Sim3 g2oSic(Tic.unit_quaternion(),Tic.translation(),1.0);
                g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oLoopScw;
                //Pose corrected with the Sim3 of the loop closure
                CorrectedSim3[pKFi]=g2oCorrectedSiw;

                // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
                Sophus::SE3d correctedTiw(g2oCorrectedSiw.rotation(),g2oCorrectedSiw.translation() / g2oCorrectedSiw.scale());
                pKFi->SetPose(correctedTiw.cast<float>());

                //Pose without correction
                g2o::Sim3 g2oSiw(Tiw.unit_quaternion().cast<double>(),Tiw.translation().cast<double>(),1.0);
                NonCorrectedSim3[pKFi]=g2oSiw;
            }  
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;
            g2o::Sim3 g2oCorrectedSiw = mit->second;
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            /*Sophus::SE3d correctedTiw(g2oCorrectedSiw.rotation(),g2oCorrectedSiw.translation() / g2oCorrectedSiw.scale());
            pKFi->SetPose(correctedTiw.cast<float>());*/

            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
            {
                MapPoint* pMPi = vpMPsi[iMP];
                if(!pMPi)
                    continue;
                if(pMPi->isBad())
                    continue;
                if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId)
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                Eigen::Vector3d P3Dw = pMPi->GetWorldPos().cast<double>();
                Eigen::Vector3d eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(P3Dw));

                pMPi->SetWorldPos(eigCorrectedP3Dw.cast<float>());
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                pMPi->mnCorrectedReference = pKFi->mnId;
                pMPi->UpdateNormalAndDepth();
            }

            // Correct velocity according to orientation correction
            if(bImuInit)
            {
                Eigen::Quaternionf Rcor = (g2oCorrectedSiw.rotation().inverse()*g2oSiw.rotation()).cast<float>();
                pKFi->SetVelocity(Rcor*pKFi->GetVelocity());
            }

            // Make sure connections are updated
            pKFi->UpdateConnections();
        }
        // TODO Check this index increasement
        mpAtlas->GetCurrentMap()->IncreaseChangeIndex();


        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        for(size_t i=0; i<mvpLoopMatchedMPs.size(); i++)
        {
            if(mvpLoopMatchedMPs[i])
            {
                MapPoint* pLoopMP = mvpLoopMatchedMPs[i];
                MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
                if(pCurMP)
                    pCurMP->Replace(pLoopMP);
                else
                {
                    mpCurrentKF->AddMapPoint(pLoopMP,i);
                    pLoopMP->AddObservation(mpCurrentKF,i);
                    pLoopMP->ComputeDistinctiveDescriptors();
                }
            }
        }
        //cout << "LC: end replacing duplicated" << endl;
    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    SearchAndFuse(CorrectedSim3, mvpLoopMapPoints);

    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    map<KeyFrame*, set<KeyFrame*> > LoopConnections;

    for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

        // Update connections. Detect new links.
        pKFi->UpdateConnections();
        LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames();
        for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev);
        }
        for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);
        }
    }

    // Optimize graph
    bool bFixedScale = mbFixScale;
    // TODO CHECK; Solo para el monocular inertial
    if(mpTracker->mSensor==System::IMU_MONOCULAR && !mpCurrentKF->GetMap()->GetIniertialBA2())
        bFixedScale=false;

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndFusion = std::chrono::steady_clock::now();

        double timeFusion = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndFusion - time_StartFusion).count();
        vdLoopFusion_ms.push_back(timeFusion);
#endif
    //cout << "Optimize essential graph" << endl;
    if(pLoopMap->IsInertial() && pLoopMap->isImuInitialized())
    {
        Optimizer::OptimizeEssentialGraph4DoF(pLoopMap, mpLoopMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections);
    }
    else
    {
        //cout << "Loop -> Scale correction: " << mg2oLoopScw.scale() << endl;
        Optimizer::OptimizeEssentialGraph(pLoopMap, mpLoopMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, bFixedScale);
    }
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndOpt = std::chrono::steady_clock::now();

    double timeOptEss = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndOpt - time_EndFusion).count();
    vdLoopOptEss_ms.push_back(timeOptEss);
#endif

    mpAtlas->InformNewBigChange();

    // Add loop edge
    mpLoopMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpLoopMatchedKF);

    if(CheckFinish())
    {
        mpLocalMapper->Release();
        mLastLoopKFid = mpCurrentKF->mnId;
        return;
    }

    // Launch a new thread to perform Global Bundle Adjustment (Only if few keyframes, if not it would take too much time)
    if(!pLoopMap->isImuInitialized() || (pLoopMap->KeyFramesInMap()<200 && mpAtlas->CountMaps()==1))
    {
        mbRunningGBA = true;
        mbFinishedGBA = false;
        mbStopGBA = false;
        mnCorrectionGBA = mnNumCorrection;

        mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment, this, pLoopMap, mpCurrentKF->mnId);
    }

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();    

    mLastLoopKFid = mpCurrentKF->mnId; //TODO old varible, it is not use in the new algorithm
}

void LoopClosing::MergeLocal()
{
    int numTemporalKFs = 25; //Temporal KFs in the local window if the map is inertial.

    //Relationship to rebuild the essential graph, it is used two times, first in the local window and later in the rest of the map
    KeyFrame* pNewChild;
    KeyFrame* pNewParent;

    vector<KeyFrame*> vpLocalCurrentWindowKFs;
    vector<KeyFrame*> vpMergeConnectedKFs;

    // Flag that is true only when we stopped a running BA, in this case we need relaunch at the end of the merge
    bool bRelaunchBA = false;

    //Verbose::PrintMess("MERGE-VISUAL: Check Full Bundle Adjustment", Verbose::VERBOSITY_DEBUG);
    // If a Global Bundle Adjustment is running, abort it
    if(isRunningGBA())
    {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;

        if(mpThreadGBA)
        {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
        bRelaunchBA = true;
    }

    //Verbose::PrintMess("MERGE-VISUAL: Request Stop Local Mapping", Verbose::VERBOSITY_DEBUG);
    //cout << "Request Stop Local Mapping" << endl;
    mpLocalMapper->RequestStop();
    // Wait until Local Mapping has effectively stopped
    while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
    {
        usleep(1000);
    }
    //cout << "Local Map stopped" << endl;

    mpLocalMapper->EmptyQueue();

    // Merge map will become in the new active map with the local window of KFs and MPs from the current map.
    // Later, the elements of the current map will be transform to the new active map reference, in order to keep real time tracking
    Map* pCurrentMap = mpCurrentKF->GetMap();
    Map* pMergeMap = mpMergeMatchedKF->GetMap();

    //std::cout << "Merge local, Active map: " << pCurrentMap->GetId() << std::endl;
    //std::cout << "Merge local, Non-Active map: " << pMergeMap->GetId() << std::endl;

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartMerge = std::chrono::steady_clock::now();
#endif

    // Ensure current keyframe is updated
    mpCurrentKF->UpdateConnections();

    //Get the current KF and its neighbors(visual->covisibles; inertial->temporal+covisibles)
    set<KeyFrame*> spLocalWindowKFs;
    //Get MPs in the welding area from the current map
    set<MapPoint*> spLocalWindowMPs;
    if(pCurrentMap->IsInertial() && pMergeMap->IsInertial()) //TODO Check the correct initialization
    {
        KeyFrame* pKFi = mpCurrentKF;
        int nInserted = 0;
        while(pKFi && nInserted < numTemporalKFs)
        {
            spLocalWindowKFs.insert(pKFi);
            pKFi = mpCurrentKF->mPrevKF;
            nInserted++;

            set<MapPoint*> spMPi = pKFi->GetMapPoints();
            spLocalWindowMPs.insert(spMPi.begin(), spMPi.end());
        }

        pKFi = mpCurrentKF->mNextKF;
        while(pKFi)
        {
            spLocalWindowKFs.insert(pKFi);

            set<MapPoint*> spMPi = pKFi->GetMapPoints();
            spLocalWindowMPs.insert(spMPi.begin(), spMPi.end());

            pKFi = mpCurrentKF->mNextKF;
        }
    }
    else
    {
        spLocalWindowKFs.insert(mpCurrentKF);
    }

    vector<KeyFrame*> vpCovisibleKFs = mpCurrentKF->GetBestCovisibilityKeyFrames(numTemporalKFs);
    spLocalWindowKFs.insert(vpCovisibleKFs.begin(), vpCovisibleKFs.end());
    spLocalWindowKFs.insert(mpCurrentKF);
    const int nMaxTries = 5;
    int nNumTries = 0;
    while(spLocalWindowKFs.size() < numTemporalKFs && nNumTries < nMaxTries)
    {
        vector<KeyFrame*> vpNewCovKFs;
        vpNewCovKFs.empty();
        for(KeyFrame* pKFi : spLocalWindowKFs)
        {
            vector<KeyFrame*> vpKFiCov = pKFi->GetBestCovisibilityKeyFrames(numTemporalKFs/2);
            for(KeyFrame* pKFcov : vpKFiCov)
            {
                if(pKFcov && !pKFcov->isBad() && spLocalWindowKFs.find(pKFcov) == spLocalWindowKFs.end())
                {
                    vpNewCovKFs.push_back(pKFcov);
                }

            }
        }

        spLocalWindowKFs.insert(vpNewCovKFs.begin(), vpNewCovKFs.end());
        nNumTries++;
    }

    for(KeyFrame* pKFi : spLocalWindowKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        set<MapPoint*> spMPs = pKFi->GetMapPoints();
        spLocalWindowMPs.insert(spMPs.begin(), spMPs.end());
    }

    //std::cout << "[Merge]: Ma = " << to_string(pCurrentMap->GetId()) << "; #KFs = " << to_string(spLocalWindowKFs.size()) << "; #MPs = " << to_string(spLocalWindowMPs.size()) << std::endl;

    set<KeyFrame*> spMergeConnectedKFs;
    if(pCurrentMap->IsInertial() && pMergeMap->IsInertial()) //TODO Check the correct initialization
    {
        KeyFrame* pKFi = mpMergeMatchedKF;
        int nInserted = 0;
        while(pKFi && nInserted < numTemporalKFs/2)
        {
            spMergeConnectedKFs.insert(pKFi);
            pKFi = mpCurrentKF->mPrevKF;
            nInserted++;
        }

        pKFi = mpMergeMatchedKF->mNextKF;
        while(pKFi && nInserted < numTemporalKFs)
        {
            spMergeConnectedKFs.insert(pKFi);
            pKFi = mpCurrentKF->mNextKF;
        }
    }
    else
    {
        spMergeConnectedKFs.insert(mpMergeMatchedKF);
    }
    vpCovisibleKFs = mpMergeMatchedKF->GetBestCovisibilityKeyFrames(numTemporalKFs);
    spMergeConnectedKFs.insert(vpCovisibleKFs.begin(), vpCovisibleKFs.end());
    spMergeConnectedKFs.insert(mpMergeMatchedKF);
    nNumTries = 0;
    while(spMergeConnectedKFs.size() < numTemporalKFs && nNumTries < nMaxTries)
    {
        vector<KeyFrame*> vpNewCovKFs;
        for(KeyFrame* pKFi : spMergeConnectedKFs)
        {
            vector<KeyFrame*> vpKFiCov = pKFi->GetBestCovisibilityKeyFrames(numTemporalKFs/2);
            for(KeyFrame* pKFcov : vpKFiCov)
            {
                if(pKFcov && !pKFcov->isBad() && spMergeConnectedKFs.find(pKFcov) == spMergeConnectedKFs.end())
                {
                    vpNewCovKFs.push_back(pKFcov);
                }

            }
        }

        spMergeConnectedKFs.insert(vpNewCovKFs.begin(), vpNewCovKFs.end());
        nNumTries++;
    }

    set<MapPoint*> spMapPointMerge;
    for(KeyFrame* pKFi : spMergeConnectedKFs)
    {
        set<MapPoint*> vpMPs = pKFi->GetMapPoints();
        spMapPointMerge.insert(vpMPs.begin(),vpMPs.end());
    }

    vector<MapPoint*> vpCheckFuseMapPoint;
    vpCheckFuseMapPoint.reserve(spMapPointMerge.size());
    std::copy(spMapPointMerge.begin(), spMapPointMerge.end(), std::back_inserter(vpCheckFuseMapPoint));

    //std::cout << "[Merge]: Mm = " << to_string(pMergeMap->GetId()) << "; #KFs = " << to_string(spMergeConnectedKFs.size()) << "; #MPs = " << to_string(spMapPointMerge.size()) << std::endl;


    //
    Sophus::SE3d Twc = mpCurrentKF->GetPoseInverse().cast<double>();
    g2o::Sim3 g2oNonCorrectedSwc(Twc.unit_quaternion(),Twc.translation(),1.0);
    g2o::Sim3 g2oNonCorrectedScw = g2oNonCorrectedSwc.inverse();
    g2o::Sim3 g2oCorrectedScw = mg2oMergeScw; //TODO Check the transformation

    KeyFrameAndPose vCorrectedSim3, vNonCorrectedSim3;
    vCorrectedSim3[mpCurrentKF]=g2oCorrectedScw;
    vNonCorrectedSim3[mpCurrentKF]=g2oNonCorrectedScw;


#ifdef REGISTER_TIMES
    vnMergeKFs.push_back(spLocalWindowKFs.size() + spMergeConnectedKFs.size());
    vnMergeMPs.push_back(spLocalWindowMPs.size() + spMapPointMerge.size());
#endif
    for(KeyFrame* pKFi : spLocalWindowKFs)
    {
        if(!pKFi || pKFi->isBad())
        {
            Verbose::PrintMess("Bad KF in correction", Verbose::VERBOSITY_DEBUG);
            continue;
        }

        if(pKFi->GetMap() != pCurrentMap)
            Verbose::PrintMess("Other map KF, this should't happen", Verbose::VERBOSITY_DEBUG);

        g2o::Sim3 g2oCorrectedSiw;

        if(pKFi!=mpCurrentKF)
        {
            Sophus::SE3d Tiw = (pKFi->GetPose()).cast<double>();
            g2o::Sim3 g2oSiw(Tiw.unit_quaternion(),Tiw.translation(),1.0);
            //Pose without correction
            vNonCorrectedSim3[pKFi]=g2oSiw;

            Sophus::SE3d Tic = Tiw*Twc;
            g2o::Sim3 g2oSic(Tic.unit_quaternion(),Tic.translation(),1.0);
            g2oCorrectedSiw = g2oSic*mg2oMergeScw;
            vCorrectedSim3[pKFi]=g2oCorrectedSiw;
        }
        else
        {
            g2oCorrectedSiw = g2oCorrectedScw;
        }
        pKFi->mTcwMerge  = pKFi->GetPose();

        // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
        double s = g2oCorrectedSiw.scale();
        pKFi->mfScale = s;
        Sophus::SE3d correctedTiw(g2oCorrectedSiw.rotation(), g2oCorrectedSiw.translation() / s);

        pKFi->mTcwMerge = correctedTiw.cast<float>();

        if(pCurrentMap->isImuInitialized())
        {
            Eigen::Quaternionf Rcor = (g2oCorrectedSiw.rotation().inverse() * vNonCorrectedSim3[pKFi].rotation()).cast<float>();
            pKFi->mVwbMerge = Rcor * pKFi->GetVelocity();
        }

        //TODO DEBUG to know which are the KFs that had been moved to the other map
    }

    int numPointsWithCorrection = 0;

    //for(MapPoint* pMPi : spLocalWindowMPs)
    set<MapPoint*>::iterator itMP = spLocalWindowMPs.begin();
    while(itMP != spLocalWindowMPs.end())
    {
        MapPoint* pMPi = *itMP;
        if(!pMPi || pMPi->isBad())
        {
            itMP = spLocalWindowMPs.erase(itMP);
            continue;
        }

        KeyFrame* pKFref = pMPi->GetReferenceKeyFrame();
        if(vCorrectedSim3.find(pKFref) == vCorrectedSim3.end())
        {
            itMP = spLocalWindowMPs.erase(itMP);
            numPointsWithCorrection++;
            continue;
        }
        g2o::Sim3 g2oCorrectedSwi = vCorrectedSim3[pKFref].inverse();
        g2o::Sim3 g2oNonCorrectedSiw = vNonCorrectedSim3[pKFref];

        // Project with non-corrected pose and project back with corrected pose
        Eigen::Vector3d P3Dw = pMPi->GetWorldPos().cast<double>();
        Eigen::Vector3d eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oNonCorrectedSiw.map(P3Dw));
        Eigen::Quaterniond Rcor = g2oCorrectedSwi.rotation() * g2oNonCorrectedSiw.rotation();

        pMPi->mPosMerge = eigCorrectedP3Dw.cast<float>();
        pMPi->mNormalVectorMerge = Rcor.cast<float>() * pMPi->GetNormal();

        itMP++;
    }
    /*if(numPointsWithCorrection>0)
    {
        std::cout << "[Merge]: " << std::to_string(numPointsWithCorrection) << " points removed from Ma due to its reference KF is not in welding area" << std::endl;
        std::cout << "[Merge]: Ma has " << std::to_string(spLocalWindowMPs.size()) << " points" << std::endl;
    }*/

    {
        unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate); // We update the current map with the Merge information
        unique_lock<mutex> mergeLock(pMergeMap->mMutexMapUpdate); // We remove the Kfs and MPs in the merged area from the old map

        //std::cout << "Merge local window: " << spLocalWindowKFs.size() << std::endl;
        //std::cout << "[Merge]: init merging maps " << std::endl;
        for(KeyFrame* pKFi : spLocalWindowKFs)
        {
            if(!pKFi || pKFi->isBad())
            {
                //std::cout << "Bad KF in correction" << std::endl;
                continue;
            }

            //std::cout << "KF id: " << pKFi->mnId << std::endl;

            pKFi->mTcwBefMerge = pKFi->GetPose();
            pKFi->mTwcBefMerge = pKFi->GetPoseInverse();
            pKFi->SetPose(pKFi->mTcwMerge);

            // Make sure connections are updated
            pKFi->UpdateMap(pMergeMap);
            pKFi->mnMergeCorrectedForKF = mpCurrentKF->mnId;
            pMergeMap->AddKeyFrame(pKFi);
            pCurrentMap->EraseKeyFrame(pKFi);

            if(pCurrentMap->isImuInitialized())
            {
                pKFi->SetVelocity(pKFi->mVwbMerge);
            }
        }

        for(MapPoint* pMPi : spLocalWindowMPs)
        {
            if(!pMPi || pMPi->isBad())
                continue;

            pMPi->SetWorldPos(pMPi->mPosMerge);
            pMPi->SetNormalVector(pMPi->mNormalVectorMerge);
            pMPi->UpdateMap(pMergeMap);
            pMergeMap->AddMapPoint(pMPi);
            pCurrentMap->EraseMapPoint(pMPi);
        }

        mpAtlas->ChangeMap(pMergeMap);
        mpAtlas->SetMapBad(pCurrentMap);
        pMergeMap->IncreaseChangeIndex();
        //TODO for debug
        pMergeMap->ChangeId(pCurrentMap->GetId());

        //std::cout << "[Merge]: merging maps finished" << std::endl;
    }

    //Rebuild the essential graph in the local window
    pCurrentMap->GetOriginKF()->SetFirstConnection(false);
    pNewChild = mpCurrentKF->GetParent(); // Old parent, it will be the new child of this KF
    pNewParent = mpCurrentKF; // Old child, now it will be the parent of its own parent(we need eliminate this KF from children list in its old parent)
    mpCurrentKF->ChangeParent(mpMergeMatchedKF);
    while(pNewChild)
    {
        pNewChild->EraseChild(pNewParent); // We remove the relation between the old parent and the new for avoid loop
        KeyFrame * pOldParent = pNewChild->GetParent();

        pNewChild->ChangeParent(pNewParent);

        pNewParent = pNewChild;
        pNewChild = pOldParent;

    }

    //Update the connections between the local window
    mpMergeMatchedKF->UpdateConnections();

    vpMergeConnectedKFs = mpMergeMatchedKF->GetVectorCovisibleKeyFrames();
    vpMergeConnectedKFs.push_back(mpMergeMatchedKF);
    //vpCheckFuseMapPoint.reserve(spMapPointMerge.size());
    //std::copy(spMapPointMerge.begin(), spMapPointMerge.end(), std::back_inserter(vpCheckFuseMapPoint));

    // Project MapPoints observed in the neighborhood of the merge keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    //std::cout << "[Merge]: start fuse points" << std::endl;
    SearchAndFuse(vCorrectedSim3, vpCheckFuseMapPoint);
    //std::cout << "[Merge]: fuse points finished" << std::endl;

    // Update connectivity
    for(KeyFrame* pKFi : spLocalWindowKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateConnections();
    }
    for(KeyFrame* pKFi : spMergeConnectedKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateConnections();
    }

    //std::cout << "[Merge]: Start welding bundle adjustment" << std::endl;

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartWeldingBA = std::chrono::steady_clock::now();

    double timeMergeMaps = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_StartWeldingBA - time_StartMerge).count();
    vdMergeMaps_ms.push_back(timeMergeMaps);
#endif

    bool bStop = false;
    vpLocalCurrentWindowKFs.clear();
    vpMergeConnectedKFs.clear();
    std::copy(spLocalWindowKFs.begin(), spLocalWindowKFs.end(), std::back_inserter(vpLocalCurrentWindowKFs));
    std::copy(spMergeConnectedKFs.begin(), spMergeConnectedKFs.end(), std::back_inserter(vpMergeConnectedKFs));
    if (mpTracker->mSensor==System::IMU_MONOCULAR || mpTracker->mSensor==System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD)
    {
        Optimizer::MergeInertialBA(mpCurrentKF,mpMergeMatchedKF,&bStop, pCurrentMap,vCorrectedSim3);
    }
    else
    {
        Optimizer::LocalBundleAdjustment(mpCurrentKF, vpLocalCurrentWindowKFs, vpMergeConnectedKFs,&bStop);
    }

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndWeldingBA = std::chrono::steady_clock::now();

    double timeWeldingBA = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndWeldingBA - time_StartWeldingBA).count();
    vdWeldingBA_ms.push_back(timeWeldingBA);
#endif
    //std::cout << "[Merge]: Welding bundle adjustment finished" << std::endl;

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();

    //Update the non critical area from the current map to the merged map
    vector<KeyFrame*> vpCurrentMapKFs = pCurrentMap->GetAllKeyFrames();
    vector<MapPoint*> vpCurrentMapMPs = pCurrentMap->GetAllMapPoints();

    if(vpCurrentMapKFs.size() == 0){}
    else {
        if(mpTracker->mSensor == System::MONOCULAR)
        {
            unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate); // We update the current map with the Merge information

            for(KeyFrame* pKFi : vpCurrentMapKFs)
            {
                if(!pKFi || pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
                {
                    continue;
                }

                g2o::Sim3 g2oCorrectedSiw;

                Sophus::SE3d Tiw = (pKFi->GetPose()).cast<double>();
                g2o::Sim3 g2oSiw(Tiw.unit_quaternion(),Tiw.translation(),1.0);
                //Pose without correction
                vNonCorrectedSim3[pKFi]=g2oSiw;

                Sophus::SE3d Tic = Tiw*Twc;
                g2o::Sim3 g2oSim(Tic.unit_quaternion(),Tic.translation(),1.0);
                g2oCorrectedSiw = g2oSim*mg2oMergeScw;
                vCorrectedSim3[pKFi]=g2oCorrectedSiw;

                // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
                double s = g2oCorrectedSiw.scale();

                pKFi->mfScale = s;

                Sophus::SE3d correctedTiw(g2oCorrectedSiw.rotation(),g2oCorrectedSiw.translation() / s);

                pKFi->mTcwBefMerge = pKFi->GetPose();
                pKFi->mTwcBefMerge = pKFi->GetPoseInverse();

                pKFi->SetPose(correctedTiw.cast<float>());

                if(pCurrentMap->isImuInitialized())
                {
                    Eigen::Quaternionf Rcor = (g2oCorrectedSiw.rotation().inverse() * vNonCorrectedSim3[pKFi].rotation()).cast<float>();
                    pKFi->SetVelocity(Rcor * pKFi->GetVelocity()); // TODO: should add here scale s
                }

            }
            for(MapPoint* pMPi : vpCurrentMapMPs)
            {
                if(!pMPi || pMPi->isBad()|| pMPi->GetMap() != pCurrentMap)
                    continue;

                KeyFrame* pKFref = pMPi->GetReferenceKeyFrame();
                g2o::Sim3 g2oCorrectedSwi = vCorrectedSim3[pKFref].inverse();
                g2o::Sim3 g2oNonCorrectedSiw = vNonCorrectedSim3[pKFref];

                // Project with non-corrected pose and project back with corrected pose
                Eigen::Vector3d P3Dw = pMPi->GetWorldPos().cast<double>();
                Eigen::Vector3d eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oNonCorrectedSiw.map(P3Dw));
                pMPi->SetWorldPos(eigCorrectedP3Dw.cast<float>());

                pMPi->UpdateNormalAndDepth();
            }
        }

        mpLocalMapper->RequestStop();
        // Wait until Local Mapping has effectively stopped
        while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
        {
            usleep(1000);
        }

        // Optimize graph (and update the loop position for each element form the begining to the end)
        if(mpTracker->mSensor != System::MONOCULAR)
        {
            Optimizer::OptimizeEssentialGraph(mpCurrentKF, vpMergeConnectedKFs, vpLocalCurrentWindowKFs, vpCurrentMapKFs, vpCurrentMapMPs);
        }


        {
            // Get Merge Map Mutex
            unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate); // We update the current map with the Merge information
            unique_lock<mutex> mergeLock(pMergeMap->mMutexMapUpdate); // We remove the Kfs and MPs in the merged area from the old map

            //std::cout << "Merge outside KFs: " << vpCurrentMapKFs.size() << std::endl;
            for(KeyFrame* pKFi : vpCurrentMapKFs)
            {
                if(!pKFi || pKFi->isBad() || pKFi->GetMap() != pCurrentMap)
                {
                    continue;
                }
                //std::cout << "KF id: " << pKFi->mnId << std::endl;

                // Make sure connections are updated
                pKFi->UpdateMap(pMergeMap);
                pMergeMap->AddKeyFrame(pKFi);
                pCurrentMap->EraseKeyFrame(pKFi);
            }

            for(MapPoint* pMPi : vpCurrentMapMPs)
            {
                if(!pMPi || pMPi->isBad())
                    continue;

                pMPi->UpdateMap(pMergeMap);
                pMergeMap->AddMapPoint(pMPi);
                pCurrentMap->EraseMapPoint(pMPi);
            }
        }
    }

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndOptEss = std::chrono::steady_clock::now();

    double timeOptEss = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndOptEss - time_EndWeldingBA).count();
    vdMergeOptEss_ms.push_back(timeOptEss);
#endif


    mpLocalMapper->Release();

    if(bRelaunchBA && (!pCurrentMap->isImuInitialized() || (pCurrentMap->KeyFramesInMap()<200 && mpAtlas->CountMaps()==1)))
    {
        // Launch a new thread to perform Global Bundle Adjustment
        mbRunningGBA = true;
        mbFinishedGBA = false;
        mbStopGBA = false;
        mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this, pMergeMap, mpCurrentKF->mnId);
    }

    mpMergeMatchedKF->AddMergeEdge(mpCurrentKF);
    mpCurrentKF->AddMergeEdge(mpMergeMatchedKF);

    pCurrentMap->IncreaseChangeIndex();
    pMergeMap->IncreaseChangeIndex();

    mpAtlas->RemoveBadMaps();

}


void LoopClosing::MergeLocal2()
{
    //cout << "Merge detected!!!!" << endl;

    int numTemporalKFs = 11; //TODO (set by parameter): Temporal KFs in the local window if the map is inertial.

    //Relationship to rebuild the essential graph, it is used two times, first in the local window and later in the rest of the map
    KeyFrame* pNewChild;
    KeyFrame* pNewParent;

    vector<KeyFrame*> vpLocalCurrentWindowKFs;
    vector<KeyFrame*> vpMergeConnectedKFs;

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    // NonCorrectedSim3[mpCurrentKF]=mg2oLoopScw;

    // Flag that is true only when we stopped a running BA, in this case we need relaunch at the end of the merge
    bool bRelaunchBA = false;

    //cout << "Check Full Bundle Adjustment" << endl;
    // If a Global Bundle Adjustment is running, abort it
    if(isRunningGBA())
    {
        unique_lock<mutex> lock(mMutexGBA);
        mbStopGBA = true;

        mnFullBAIdx++;

        if(mpThreadGBA)
        {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
        bRelaunchBA = true;
    }


    //cout << "Request Stop Local Mapping" << endl;
    mpLocalMapper->RequestStop();
    // Wait until Local Mapping has effectively stopped
    while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
    {
        usleep(1000);
    }
    //cout << "Local Map stopped" << endl;

    Map* pCurrentMap = mpCurrentKF->GetMap();
    Map* pMergeMap = mpMergeMatchedKF->GetMap();

    {
        float s_on = mSold_new.scale();
        Sophus::SE3f T_on(mSold_new.rotation().cast<float>(), mSold_new.translation().cast<float>());

        unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);

        //cout << "KFs before empty: " << mpAtlas->GetCurrentMap()->KeyFramesInMap() << endl;
        mpLocalMapper->EmptyQueue();
        //cout << "KFs after empty: " << mpAtlas->GetCurrentMap()->KeyFramesInMap() << endl;

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        //cout << "updating active map to merge reference" << endl;
        //cout << "curr merge KF id: " << mpCurrentKF->mnId << endl;
        //cout << "curr tracking KF id: " << mpTracker->GetLastKeyFrame()->mnId << endl;
        bool bScaleVel=false;
        if(s_on!=1)
            bScaleVel=true;
        mpAtlas->GetCurrentMap()->ApplyScaledRotation(T_on,s_on,bScaleVel);
        mpTracker->UpdateFrameIMU(s_on,mpCurrentKF->GetImuBias(),mpTracker->GetLastKeyFrame());

        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    }

    const int numKFnew=pCurrentMap->KeyFramesInMap();

    if((mpTracker->mSensor==System::IMU_MONOCULAR || mpTracker->mSensor==System::IMU_STEREO || mpTracker->mSensor==System::IMU_RGBD)
       && !pCurrentMap->GetIniertialBA2()){
        // Map is not completly initialized
        Eigen::Vector3d bg, ba;
        bg << 0., 0., 0.;
        ba << 0., 0., 0.;
        Optimizer::InertialOptimization(pCurrentMap,bg,ba);
        IMU::Bias b (ba[0],ba[1],ba[2],bg[0],bg[1],bg[2]);
        unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
        mpTracker->UpdateFrameIMU(1.0f,b,mpTracker->GetLastKeyFrame());

        // Set map initialized
        pCurrentMap->SetIniertialBA2();
        pCurrentMap->SetIniertialBA1();
        pCurrentMap->SetImuInitialized();

    }


    //cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    // Load KFs and MPs from merge map
    //cout << "updating current map" << endl;
    {
        // Get Merge Map Mutex (This section stops tracking!!)
        unique_lock<mutex> currentLock(pCurrentMap->mMutexMapUpdate); // We update the current map with the Merge information
        unique_lock<mutex> mergeLock(pMergeMap->mMutexMapUpdate); // We remove the Kfs and MPs in the merged area from the old map


        vector<KeyFrame*> vpMergeMapKFs = pMergeMap->GetAllKeyFrames();
        vector<MapPoint*> vpMergeMapMPs = pMergeMap->GetAllMapPoints();


        for(KeyFrame* pKFi : vpMergeMapKFs)
        {
            if(!pKFi || pKFi->isBad() || pKFi->GetMap() != pMergeMap)
            {
                continue;
            }

            // Make sure connections are updated
            pKFi->UpdateMap(pCurrentMap);
            pCurrentMap->AddKeyFrame(pKFi);
            pMergeMap->EraseKeyFrame(pKFi);
        }

        for(MapPoint* pMPi : vpMergeMapMPs)
        {
            if(!pMPi || pMPi->isBad() || pMPi->GetMap() != pMergeMap)
                continue;

            pMPi->UpdateMap(pCurrentMap);
            pCurrentMap->AddMapPoint(pMPi);
            pMergeMap->EraseMapPoint(pMPi);
        }

        // Save non corrected poses (already merged maps)
        vector<KeyFrame*> vpKFs = pCurrentMap->GetAllKeyFrames();
        for(KeyFrame* pKFi : vpKFs)
        {
            Sophus::SE3d Tiw = (pKFi->GetPose()).cast<double>();
            g2o::Sim3 g2oSiw(Tiw.unit_quaternion(),Tiw.translation(),1.0);
            NonCorrectedSim3[pKFi]=g2oSiw;
        }
    }

    //cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    //cout << "end updating current map" << endl;

    // Critical zone
    //bool good = pCurrentMap->CheckEssentialGraph();
    /*if(!good)
        cout << "BAD ESSENTIAL GRAPH!!" << endl;*/

    //cout << "Update essential graph" << endl;
    // mpCurrentKF->UpdateConnections(); // to put at false mbFirstConnection
    pMergeMap->GetOriginKF()->SetFirstConnection(false);
    pNewChild = mpMergeMatchedKF->GetParent(); // Old parent, it will be the new child of this KF
    pNewParent = mpMergeMatchedKF; // Old child, now it will be the parent of its own parent(we need eliminate this KF from children list in its old parent)
    mpMergeMatchedKF->ChangeParent(mpCurrentKF);
    while(pNewChild)
    {
        pNewChild->EraseChild(pNewParent); // We remove the relation between the old parent and the new for avoid loop
        KeyFrame * pOldParent = pNewChild->GetParent();
        pNewChild->ChangeParent(pNewParent);
        pNewParent = pNewChild;
        pNewChild = pOldParent;

    }


    //cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    //cout << "end update essential graph" << endl;

    /*good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 1!!" << endl;*/

    //cout << "Update relationship between KFs" << endl;
    vector<MapPoint*> vpCheckFuseMapPoint; // MapPoint vector from current map to allow to fuse duplicated points with the old map (merge)
    vector<KeyFrame*> vpCurrentConnectedKFs;

    mvpMergeConnectedKFs.push_back(mpMergeMatchedKF);
    vector<KeyFrame*> aux = mpMergeMatchedKF->GetVectorCovisibleKeyFrames();
    mvpMergeConnectedKFs.insert(mvpMergeConnectedKFs.end(), aux.begin(), aux.end());
    if (mvpMergeConnectedKFs.size()>6)
        mvpMergeConnectedKFs.erase(mvpMergeConnectedKFs.begin()+6,mvpMergeConnectedKFs.end());
    /*mvpMergeConnectedKFs = mpMergeMatchedKF->GetVectorCovisibleKeyFrames();
    mvpMergeConnectedKFs.push_back(mpMergeMatchedKF);*/

    mpCurrentKF->UpdateConnections();
    vpCurrentConnectedKFs.push_back(mpCurrentKF);
    /*vpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    vpCurrentConnectedKFs.push_back(mpCurrentKF);*/
    aux = mpCurrentKF->GetVectorCovisibleKeyFrames();
    vpCurrentConnectedKFs.insert(vpCurrentConnectedKFs.end(), aux.begin(), aux.end());
    if (vpCurrentConnectedKFs.size()>6)
        vpCurrentConnectedKFs.erase(vpCurrentConnectedKFs.begin()+6,vpCurrentConnectedKFs.end());

    set<MapPoint*> spMapPointMerge;
    for(KeyFrame* pKFi : mvpMergeConnectedKFs)
    {
        set<MapPoint*> vpMPs = pKFi->GetMapPoints();
        spMapPointMerge.insert(vpMPs.begin(),vpMPs.end());
        if(spMapPointMerge.size()>1000)
            break;
    }

    /*cout << "vpCurrentConnectedKFs.size() " << vpCurrentConnectedKFs.size() << endl;
    cout << "mvpMergeConnectedKFs.size() " << mvpMergeConnectedKFs.size() << endl;
    cout << "spMapPointMerge.size() " << spMapPointMerge.size() << endl;*/


    vpCheckFuseMapPoint.reserve(spMapPointMerge.size());
    std::copy(spMapPointMerge.begin(), spMapPointMerge.end(), std::back_inserter(vpCheckFuseMapPoint));
    //cout << "Finished to update relationship between KFs" << endl;

    //cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    /*good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 2!!" << endl;*/

    //cout << "start SearchAndFuse" << endl;
    SearchAndFuse(vpCurrentConnectedKFs, vpCheckFuseMapPoint);
    //cout << "end SearchAndFuse" << endl;

    //cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    /*good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 3!!" << endl;

    cout << "Init to update connections" << endl;*/


    for(KeyFrame* pKFi : vpCurrentConnectedKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateConnections();
    }
    for(KeyFrame* pKFi : mvpMergeConnectedKFs)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateConnections();
    }
    //cout << "end update connections" << endl;

    //cout << "MergeMap init ID: " << pMergeMap->GetInitKFid() << "       CurrMap init ID: " << pCurrentMap->GetInitKFid() << endl;

    /*good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 4!!" << endl;*/

    // TODO Check: If new map is too small, we suppose that not informaiton can be propagated from new to old map
    if (numKFnew<10){
        mpLocalMapper->Release();
        return;
    }

    /*good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 5!!" << endl;*/

    // Perform BA
    bool bStopFlag=false;
    KeyFrame* pCurrKF = mpTracker->GetLastKeyFrame();
    //cout << "start MergeInertialBA" << endl;
    Optimizer::MergeInertialBA(pCurrKF, mpMergeMatchedKF, &bStopFlag, pCurrentMap,CorrectedSim3);
    //cout << "end MergeInertialBA" << endl;

    /*good = pCurrentMap->CheckEssentialGraph();
    if(!good)
        cout << "BAD ESSENTIAL GRAPH 6!!" << endl;*/

    // Release Local Mapping.
    mpLocalMapper->Release();


    return;
}

void LoopClosing::CheckObservations(set<KeyFrame*> &spKFsMap1, set<KeyFrame*> &spKFsMap2)
{
    cout << "----------------------" << endl;
    for(KeyFrame* pKFi1 : spKFsMap1)
    {
        map<KeyFrame*, int> mMatchedMP;
        set<MapPoint*> spMPs = pKFi1->GetMapPoints();

        for(MapPoint* pMPij : spMPs)
        {
            if(!pMPij || pMPij->isBad())
            {
                continue;
            }

            map<KeyFrame*, tuple<int,int>> mMPijObs = pMPij->GetObservations();
            for(KeyFrame* pKFi2 : spKFsMap2)
            {
                if(mMPijObs.find(pKFi2) != mMPijObs.end())
                {
                    if(mMatchedMP.find(pKFi2) != mMatchedMP.end())
                    {
                        mMatchedMP[pKFi2] = mMatchedMP[pKFi2] + 1;
                    }
                    else
                    {
                        mMatchedMP[pKFi2] = 1;
                    }
                }
            }

        }

        if(mMatchedMP.size() == 0)
        {
            cout << "CHECK-OBS: KF " << pKFi1->mnId << " has not any matched MP with the other map" << endl;
        }
        else
        {
            cout << "CHECK-OBS: KF " << pKFi1->mnId << " has matched MP with " << mMatchedMP.size() << " KF from the other map" << endl;
            for(pair<KeyFrame*, int> matchedKF : mMatchedMP)
            {
                cout << "   -KF: " << matchedKF.first->mnId << ", Number of matches: " << matchedKF.second << endl;
            }
        }
    }
    cout << "----------------------" << endl;
}


void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap, vector<MapPoint*> &vpMapPoints)
{
    ORBmatcher matcher(0.8);

    int total_replaces = 0;

    //cout << "[FUSE]: Initially there are " << vpMapPoints.size() << " MPs" << endl;
    //cout << "FUSE: Intially there are " << CorrectedPosesMap.size() << " KFs" << endl;
    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
        int num_replaces = 0;
        KeyFrame* pKFi = mit->first;
        Map* pMap = pKFi->GetMap();

        g2o::Sim3 g2oScw = mit->second;
        Sophus::Sim3f Scw = Converter::toSophus(g2oScw);

        vector<MapPoint*> vpReplacePoints(vpMapPoints.size(),static_cast<MapPoint*>(NULL));
        int numFused = matcher.Fuse(pKFi,Scw,vpMapPoints,4,vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        const int nLP = vpMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep)
            {


                num_replaces += 1;
                pRep->Replace(vpMapPoints[i]);

            }
        }

        total_replaces += num_replaces;
    }
    //cout << "[FUSE]: " << total_replaces << " MPs had been fused" << endl;
}


void LoopClosing::SearchAndFuse(const vector<KeyFrame*> &vConectedKFs, vector<MapPoint*> &vpMapPoints)
{
    ORBmatcher matcher(0.8);

    int total_replaces = 0;

    //cout << "FUSE-POSE: Initially there are " << vpMapPoints.size() << " MPs" << endl;
    //cout << "FUSE-POSE: Intially there are " << vConectedKFs.size() << " KFs" << endl;
    for(auto mit=vConectedKFs.begin(), mend=vConectedKFs.end(); mit!=mend;mit++)
    {
        int num_replaces = 0;
        KeyFrame* pKF = (*mit);
        Map* pMap = pKF->GetMap();
        Sophus::SE3f Tcw = pKF->GetPose();
        Sophus::Sim3f Scw(Tcw.unit_quaternion(),Tcw.translation());
        Scw.setScale(1.f);
        /*std::cout << "These should be zeros: " <<
            Scw.rotationMatrix() - Tcw.rotationMatrix() << std::endl <<
            Scw.translation() - Tcw.translation() << std::endl <<
            Scw.scale() - 1.f << std::endl;*/
        vector<MapPoint*> vpReplacePoints(vpMapPoints.size(),static_cast<MapPoint*>(NULL));
        matcher.Fuse(pKF,Scw,vpMapPoints,4,vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        const int nLP = vpMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep)
            {
                num_replaces += 1;
                pRep->Replace(vpMapPoints[i]);
            }
        }
        /*cout << "FUSE-POSE: KF " << pKF->mnId << " ->" << num_replaces << " MPs fused" << endl;
        total_replaces += num_replaces;*/
    }
    //cout << "FUSE-POSE: " << total_replaces << " MPs had been fused" << endl;
}



void LoopClosing::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
        unique_lock<mutex> lock2(mMutexReset);
        if(!mbResetRequested)
            break;
        }
        usleep(5000);
    }
}

void LoopClosing::RequestResetActiveMap(Map *pMap)
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetActiveMapRequested = true;
        mpMapToReset = pMap;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetActiveMapRequested)
                break;
        }
        usleep(3000);
    }
}

void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        cout << "Loop closer reset requested..." << endl;
        mlpLoopKeyFrameQueue.clear();
        if(mpXFeatKeyFrameDB)
            mpXFeatKeyFrameDB->clear();
        mLastLoopKFid=0;  //TODO old variable, it is not use in the new algorithm
        mbResetRequested=false;
        mbResetActiveMapRequested = false;
    }
    else if(mbResetActiveMapRequested)
    {

        for (list<KeyFrame*>::const_iterator it=mlpLoopKeyFrameQueue.begin(); it != mlpLoopKeyFrameQueue.end();)
        {
            KeyFrame* pKFi = *it;
            if(pKFi->GetMap() == mpMapToReset)
            {
                it = mlpLoopKeyFrameQueue.erase(it);
            }
            else
                ++it;
        }

        mLastLoopKFid=mpAtlas->GetLastInitKFid(); //TODO old variable, it is not use in the new algorithm
        if(mpXFeatKeyFrameDB)
            mpXFeatKeyFrameDB->clearMap(mpMapToReset);
        mbResetActiveMapRequested=false;

    }
}

void LoopClosing::RunGlobalBundleAdjustment(Map* pActiveMap, unsigned long nLoopKF)
{  
    Verbose::PrintMess("Starting Global Bundle Adjustment", Verbose::VERBOSITY_NORMAL);
    {
        unique_lock<mutex> lock(mMutexXFeatLoopSummary);
        ++mnXFeatGbaStartedCount;
        mnXFeatLastGbaLoopKF = static_cast<int>(nLoopKF);
    }

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartFGBA = std::chrono::steady_clock::now();

    nFGBA_exec += 1;

    vnGBAKFs.push_back(pActiveMap->GetAllKeyFrames().size());
    vnGBAMPs.push_back(pActiveMap->GetAllMapPoints().size());
#endif

    const bool bImuInit = pActiveMap->isImuInitialized();

    if(!bImuInit)
        Optimizer::GlobalBundleAdjustemnt(pActiveMap,10,&mbStopGBA,nLoopKF,false);
    else
        Optimizer::FullInertialBA(pActiveMap,7,false,nLoopKF,&mbStopGBA);
    {
        unique_lock<mutex> lock(mMutexXFeatLoopSummary);
        ++mnXFeatGbaOptimizerFinishedCount;
    }

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndGBA = std::chrono::steady_clock::now();

    double timeGBA = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndGBA - time_StartFGBA).count();
    vdGBA_ms.push_back(timeGBA);

    if(mbStopGBA)
    {
        nFGBA_abort += 1;
    }
#endif

    int idx =  mnFullBAIdx;
    // Optimizer::GlobalBundleAdjustemnt(mpMap,10,&mbStopGBA,nLoopKF,false);

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(idx!=mnFullBAIdx)
        {
            mbFinishedGBA = true;
            mbRunningGBA = false;
            {
                unique_lock<mutex> summaryLock(mMutexXFeatLoopSummary);
                ++mnXFeatGbaAbortedCount;
                mnXFeatLastGbaAbortReason = 1;
            }
            return;
        }

        if(!bImuInit && pActiveMap->isImuInitialized())
        {
            mbFinishedGBA = true;
            mbRunningGBA = false;
            {
                unique_lock<mutex> summaryLock(mMutexXFeatLoopSummary);
                ++mnXFeatGbaAbortedCount;
                mnXFeatLastGbaAbortReason = 2;
            }
            return;
        }

        if(!mbStopGBA)
        {
            Verbose::PrintMess("Global Bundle Adjustment finished", Verbose::VERBOSITY_NORMAL);
            Verbose::PrintMess("Updating map ...", Verbose::VERBOSITY_NORMAL);

            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped

            while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
            {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<mutex> lock(pActiveMap->mMutexMapUpdate);
            // cout << "LC: Update Map Mutex adquired" << endl;

            //pActiveMap->PrintEssentialGraph();
            // Correct keyframes starting at map first keyframe
            list<KeyFrame*> lpKFtoCheck(pActiveMap->mvpKeyFrameOrigins.begin(),pActiveMap->mvpKeyFrameOrigins.end());

            while(!lpKFtoCheck.empty())
            {
                KeyFrame* pKF = lpKFtoCheck.front();
                const set<KeyFrame*> sChilds = pKF->GetChilds();
                //cout << "---Updating KF " << pKF->mnId << " with " << sChilds.size() << " childs" << endl;
                //cout << " KF mnBAGlobalForKF: " << pKF->mnBAGlobalForKF << endl;
                Sophus::SE3f Twc = pKF->GetPoseInverse();
                //cout << "Twc: " << Twc << endl;
                //cout << "GBA: Correct KeyFrames" << endl;
                for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                {
                    KeyFrame* pChild = *sit;
                    if(!pChild || pChild->isBad())
                        continue;

                    if(pChild->mnBAGlobalForKF!=nLoopKF)
                    {
                        //cout << "++++New child with flag " << pChild->mnBAGlobalForKF << "; LoopKF: " << nLoopKF << endl;
                        //cout << " child id: " << pChild->mnId << endl;
                        Sophus::SE3f Tchildc = pChild->GetPose() * Twc;
                        //cout << "Child pose: " << Tchildc << endl;
                        //cout << "pKF->mTcwGBA: " << pKF->mTcwGBA << endl;
                        pChild->mTcwGBA = Tchildc * pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;

                        Sophus::SO3f Rcor = pChild->mTcwGBA.so3().inverse() * pChild->GetPose().so3();
                        if(pChild->isVelocitySet()){
                            pChild->mVwbGBA = Rcor * pChild->GetVelocity();
                        }
                        else
                            Verbose::PrintMess("Child velocity empty!! ", Verbose::VERBOSITY_NORMAL);


                        //cout << "Child bias: " << pChild->GetImuBias() << endl;
                        pChild->mBiasGBA = pChild->GetImuBias();


                        pChild->mnBAGlobalForKF = nLoopKF;

                    }
                    lpKFtoCheck.push_back(pChild);
                }

                //cout << "-------Update pose" << endl;
                pKF->mTcwBefGBA = pKF->GetPose();
                //cout << "pKF->mTcwBefGBA: " << pKF->mTcwBefGBA << endl;
                pKF->SetPose(pKF->mTcwGBA);
                /*cv::Mat Tco_cn = pKF->mTcwBefGBA * pKF->mTcwGBA.inv();
                cv::Vec3d trasl = Tco_cn.rowRange(0,3).col(3);
                double dist = cv::norm(trasl);
                cout << "GBA: KF " << pKF->mnId << " had been moved " << dist << " meters" << endl;
                double desvX = 0;
                double desvY = 0;
                double desvZ = 0;
                if(pKF->mbHasHessian)
                {
                    cv::Mat hessianInv = pKF->mHessianPose.inv();

                    double covX = hessianInv.at<double>(3,3);
                    desvX = std::sqrt(covX);
                    double covY = hessianInv.at<double>(4,4);
                    desvY = std::sqrt(covY);
                    double covZ = hessianInv.at<double>(5,5);
                    desvZ = std::sqrt(covZ);
                    pKF->mbHasHessian = false;
                }
                if(dist > 1)
                {
                    cout << "--To much distance correction: It has " << pKF->GetConnectedKeyFrames().size() << " connected KFs" << endl;
                    cout << "--It has " << pKF->GetCovisiblesByWeight(80).size() << " connected KF with 80 common matches or more" << endl;
                    cout << "--It has " << pKF->GetCovisiblesByWeight(50).size() << " connected KF with 50 common matches or more" << endl;
                    cout << "--It has " << pKF->GetCovisiblesByWeight(20).size() << " connected KF with 20 common matches or more" << endl;

                    cout << "--STD in meters(x, y, z): " << desvX << ", " << desvY << ", " << desvZ << endl;


                    string strNameFile = pKF->mNameFile;
                    cv::Mat imLeft = cv::imread(strNameFile, CV_LOAD_IMAGE_UNCHANGED);

                    cv::cvtColor(imLeft, imLeft, CV_GRAY2BGR);

                    vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();
                    int num_MPs = 0;
                    for(int i=0; i<vpMapPointsKF.size(); ++i)
                    {
                        if(!vpMapPointsKF[i] || vpMapPointsKF[i]->isBad())
                        {
                            continue;
                        }
                        num_MPs += 1;
                        string strNumOBs = to_string(vpMapPointsKF[i]->Observations());
                        cv::circle(imLeft, pKF->mvKeys[i].pt, 2, cv::Scalar(0, 255, 0));
                        cv::putText(imLeft, strNumOBs, pKF->mvKeys[i].pt, CV_FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0));
                    }
                    cout << "--It has " << num_MPs << " MPs matched in the map" << endl;

                    string namefile = "./test_GBA/GBA_" + to_string(nLoopKF) + "_KF" + to_string(pKF->mnId) +"_D" + to_string(dist) +".png";
                    cv::imwrite(namefile, imLeft);
                }*/


                if(pKF->bImu)
                {
                    //cout << "-------Update inertial values" << endl;
                    pKF->mVwbBefGBA = pKF->GetVelocity();
                    //if (pKF->mVwbGBA.empty())
                    //    Verbose::PrintMess("pKF->mVwbGBA is empty", Verbose::VERBOSITY_NORMAL);

                    //assert(!pKF->mVwbGBA.empty());
                    pKF->SetVelocity(pKF->mVwbGBA);
                    pKF->SetNewBias(pKF->mBiasGBA);                    
                }

                lpKFtoCheck.pop_front();
            }

            //cout << "GBA: Correct MapPoints" << endl;
            // Correct MapPoints
            const vector<MapPoint*> vpMPs = pActiveMap->GetAllMapPoints();

            for(size_t i=0; i<vpMPs.size(); i++)
            {
                MapPoint* pMP = vpMPs[i];

                if(pMP->isBad())
                    continue;

                if(pMP->mnBAGlobalForKF==nLoopKF)
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                    if(pRefKF->mnBAGlobalForKF!=nLoopKF)
                        continue;

                    /*if(pRefKF->mTcwBefGBA.empty())
                        continue;*/

                    // Map to non-corrected camera
                    // cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                    // cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                    Eigen::Vector3f Xc = pRefKF->mTcwBefGBA * pMP->GetWorldPos();

                    // Backproject using corrected camera
                    pMP->SetWorldPos(pRefKF->GetPoseInverse() * Xc);
                }
            }

            pActiveMap->InformNewBigChange();
            pActiveMap->IncreaseChangeIndex();

            // TODO Check this update
            // mpTracker->UpdateFrameIMU(1.0f, mpTracker->GetLastKeyFrame()->GetImuBias(), mpTracker->GetLastKeyFrame());

            mpLocalMapper->Release();

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndUpdateMap = std::chrono::steady_clock::now();

            double timeUpdateMap = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndUpdateMap - time_EndGBA).count();
            vdUpdateMap_ms.push_back(timeUpdateMap);

            double timeFGBA = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndUpdateMap - time_StartFGBA).count();
            vdFGBATotal_ms.push_back(timeFGBA);
#endif
            Verbose::PrintMess("Map updated!", Verbose::VERBOSITY_NORMAL);
            {
                unique_lock<mutex> summaryLock(mMutexXFeatLoopSummary);
                ++mnXFeatGbaAppliedCount;
            }
        }
        else
        {
            unique_lock<mutex> summaryLock(mMutexXFeatLoopSummary);
            ++mnXFeatGbaAbortedCount;
            mnXFeatLastGbaAbortReason = 4;
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}

void LoopClosing::RequestFinish()
{
    {
        unique_lock<mutex> lock(mMutexFinish);
        // cout << "LC: Finish requested" << endl;
        mbFinishRequested = true;
    }

    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mlpLoopKeyFrameQueue.clear();
    }
}

bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM
