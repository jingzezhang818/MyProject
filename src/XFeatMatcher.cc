#include "XFeatMatcher.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <cstdlib>
#include <string>
#include <vector>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

namespace ORB_SLAM3
{

namespace
{
//调试: 统一解析环境变量布尔开关。
bool IsEnvFlagEnabled(const char* key)
{
    const char* env = std::getenv(key);
    if(!env)
        return false;

    const std::string v(env);
    return !(v.empty() || v == "0" || v == "false" || v == "FALSE");
}

bool IsXFeatMatcherDebugEnabled()
{
    //调试: Matcher 级调试开关，设置 `XFEAT_DEBUG_MATCHER=1` 或 `XFEAT_DEBUG=1` 启用。
    static const bool enabled = IsEnvFlagEnabled("XFEAT_DEBUG_MATCHER") || IsEnvFlagEnabled("XFEAT_DEBUG");
    return enabled;
}

bool IsXFeatMatcherVerboseDebugEnabled()
{
    //调试: 更细粒度统计开关，设置 `XFEAT_DEBUG_MATCHER_VERBOSE=1` 启用。
    static const bool enabled = IsEnvFlagEnabled("XFEAT_DEBUG_MATCHER_VERBOSE");
    return enabled;
}

bool UseXFeatDescriptorBank()
{
    //调试: descriptor bank 默认关闭；设置 `XFEAT_USE_DESC_BANK=1` 手动启用。
    static const bool enabled = IsEnvFlagEnabled("XFEAT_USE_DESC_BANK");
    return enabled;
}

bool UseLegacyLevelAwareRatioGate()
{
    //调试: 兼容旧逻辑时仅在同octave下做ratio过滤；默认关闭（即所有候选都做ratio）。
    static const bool enabled = IsEnvFlagEnabled("XFEAT_LEGACY_LEVEL_RATIO");
    return enabled;
}

bool EnforceMutualForNN()
{
    //调试: NN主链默认启用mutual硬约束，避免“高召回低纯度”污染位姿优化。
    //设置 `XFEAT_ALLOW_NON_MUTUAL_NN=1` 可回退为仅记录mutual统计但不强制。
    static const bool enabled = !IsEnvFlagEnabled("XFEAT_ALLOW_NON_MUTUAL_NN");
    return enabled;
}

int GetEnvIntWithRange(const char* key, const int fallback, const int minValue, const int maxValue)
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

float GetEnvFloatWithRange(const char* key, const float fallback, const float minValue, const float maxValue)
{
    const char* env = std::getenv(key);
    if(!env)
        return fallback;

    try
    {
        const float v = std::stof(std::string(env));
        if(std::isfinite(v))
            return std::max(minValue, std::min(maxValue, v));
    }
    catch(...)
    {
    }
    return fallback;
}

bool UseXFeatMatchSpatialQuota()
{
    //调试: 匹配级空间均匀化总开关。默认开启；`XFEAT_MATCH_SPATIAL_DISABLE=1` 可关闭。
    static const bool enabled = !IsEnvFlagEnabled("XFEAT_MATCH_SPATIAL_DISABLE");
    return enabled;
}

int GetXFeatMatchSpatialGridCols()
{
    //调试: 匹配级空间均匀化网格列数。
    static const int cols = GetEnvIntWithRange("XFEAT_MATCH_SPATIAL_GRID_COLS", 8, 2, 64);
    return cols;
}

int GetXFeatMatchSpatialGridRows()
{
    //调试: 匹配级空间均匀化网格行数。
    static const int rows = GetEnvIntWithRange("XFEAT_MATCH_SPATIAL_GRID_ROWS", 6, 2, 64);
    return rows;
}

int GetXFeatMatchSpatialTrigger()
{
    //调试: 候选数低于该阈值时不做空间均匀化，避免低匹配帧被过度削减。
    static const int trigger = GetEnvIntWithRange("XFEAT_MATCH_SPATIAL_TRIGGER", 40, 1, 10000);
    return trigger;
}

float GetXFeatMatchSpatialKeepRatio()
{
    //调试: 空间均匀化后至少保留的候选比例。
    static const float ratio = GetEnvFloatWithRange("XFEAT_MATCH_SPATIAL_KEEP_RATIO", 0.90f, 0.10f, 1.00f);
    return ratio;
}

float GetXFeatMatchSpatialCapScale()
{
    //调试: 每格上限系数。值越小越强调均匀覆盖，值越大越保守。
    static const float scale = GetEnvFloatWithRange("XFEAT_MATCH_SPATIAL_CAP_SCALE", 0.85f, 0.30f, 3.00f);
    return scale;
}

bool UseXFeatProjectionLevelGate()
{
    //调试: 投影匹配的尺度层级门控总开关，默认开启；`XFEAT_PROJ_DISABLE_LEVEL_GATE=1` 可关闭。
    static const bool enabled = !IsEnvFlagEnabled("XFEAT_PROJ_DISABLE_LEVEL_GATE");
    return enabled;
}

int GetXFeatProjMapMinOffset()
{
    //调试: MapPoint投影路径的最小层级偏移（相对预测层）。
    static const int v = GetEnvIntWithRange("XFEAT_PROJ_MAP_MIN_OFFSET", -2, -8, 8);
    return v;
}

int GetXFeatProjMapMaxOffset()
{
    //调试: MapPoint投影路径的最大层级偏移（相对预测层）。
    static const int v = GetEnvIntWithRange("XFEAT_PROJ_MAP_MAX_OFFSET", 1, -8, 8);
    return v;
}

bool UseXFeatLastFrameDirectionalBias()
{
    //调试: 兼容旧逻辑时按前进/后退方向切换层级窗口；默认关闭，采用统一对称窗口。
    static const bool enabled = IsEnvFlagEnabled("XFEAT_PROJ_LAST_USE_DIR_BIAS");
    return enabled;
}

int GetXFeatProjLastNeutralMinOffset()
{
    //调试: LastFrame投影（默认统一窗口）最小层级偏移。
    static const int v = GetEnvIntWithRange("XFEAT_PROJ_LAST_MIN_OFFSET", -2, -8, 8);
    return v;
}

int GetXFeatProjLastNeutralMaxOffset()
{
    //调试: LastFrame投影（默认统一窗口）最大层级偏移。
    static const int v = GetEnvIntWithRange("XFEAT_PROJ_LAST_MAX_OFFSET", 2, -8, 8);
    return v;
}

int GetXFeatProjLastForwardMinOffset()
{
    //调试: LastFrame前进方向最小层级偏移（仅启用方向偏置时生效）。
    static const int v = GetEnvIntWithRange("XFEAT_PROJ_LAST_FWD_MIN_OFFSET", -1, -8, 8);
    return v;
}

int GetXFeatProjLastForwardMaxOffset()
{
    //调试: LastFrame前进方向最大层级偏移（仅启用方向偏置时生效）。
    static const int v = GetEnvIntWithRange("XFEAT_PROJ_LAST_FWD_MAX_OFFSET", 2, -8, 8);
    return v;
}

int GetXFeatProjLastBackwardMinOffset()
{
    //调试: LastFrame后退方向最小层级偏移（仅启用方向偏置时生效）。
    static const int v = GetEnvIntWithRange("XFEAT_PROJ_LAST_BWD_MIN_OFFSET", -2, -8, 8);
    return v;
}

int GetXFeatProjLastBackwardMaxOffset()
{
    //调试: LastFrame后退方向最大层级偏移（仅启用方向偏置时生效）。
    static const int v = GetEnvIntWithRange("XFEAT_PROJ_LAST_BWD_MAX_OFFSET", 1, -8, 8);
    return v;
}

void BuildProjectionLevelRange(const int predictedLevel,
                               const int nScaleLevels,
                               const int minOffset,
                               const int maxOffset,
                               int& minLevel,
                               int& maxLevel)
{
    if(!UseXFeatProjectionLevelGate() || nScaleLevels <= 0)
    {
        minLevel = -1;
        maxLevel = -1;
        return;
    }

    int lo = predictedLevel + minOffset;
    int hi = predictedLevel + maxOffset;
    if(lo > hi)
        std::swap(lo, hi);

    lo = std::max(0, std::min(nScaleLevels - 1, lo));
    hi = std::max(0, std::min(nScaleLevels - 1, hi));
    minLevel = lo;
    maxLevel = hi;
}

void BuildLastFrameProjectionLevelRange(const int predictedLevel,
                                        const int nScaleLevels,
                                        const bool bForward,
                                        const bool bBackward,
                                        int& minLevel,
                                        int& maxLevel)
{
    int minOffset = GetXFeatProjLastNeutralMinOffset();
    int maxOffset = GetXFeatProjLastNeutralMaxOffset();

    if(UseXFeatLastFrameDirectionalBias())
    {
        if(bForward)
        {
            minOffset = GetXFeatProjLastForwardMinOffset();
            maxOffset = GetXFeatProjLastForwardMaxOffset();
        }
        else if(bBackward)
        {
            minOffset = GetXFeatProjLastBackwardMinOffset();
            maxOffset = GetXFeatProjLastBackwardMaxOffset();
        }
    }

    BuildProjectionLevelRange(predictedLevel, nScaleLevels, minOffset, maxOffset, minLevel, maxLevel);
}

bool ShouldRejectByRatio(const float bestDist,
                        const float secondDist,
                        const float nnRatio,
                        const int bestLevel,
                        const int secondLevel)
{
    //调试: second-best不存在时，不做ratio过滤。
    if(!std::isfinite(secondDist) || secondDist <= 0.0f)
        return false;

    if(UseLegacyLevelAwareRatioGate() && bestLevel != secondLevel)
        return false;

    return bestDist >= nnRatio * secondDist;
}

int GetXFeatTriangulationKnn()
{
    //调试: 三角化阶段每个特征点保留的KNN候选个数，默认32。
    const char* env = std::getenv("XFEAT_TRIANG_KNN");
    if(env)
    {
        try
        {
            const int v = std::stoi(std::string(env));
            if(v >= 2 && v <= 128)
                return v;
        }
        catch(...)
        {
        }
    }
    return 32;
}

struct CandidateMatch
{
    int featIdx = -1;
    MapPoint* pMP = nullptr;
    float dist = std::numeric_limits<float>::infinity();
};

struct SpatialQuotaStats
{
    bool applied = false;
    int input = 0;
    int acceptedStage1 = 0;
    int acceptedFinal = 0;
    int recovered = 0;
    int dropped = 0;
    int cap = 0;
    int gridCols = 0;
    int gridRows = 0;
};

bool GetFrameKeyPointByIndex(const Frame& F, const int idx, cv::KeyPoint& kp)
{
    if(idx < 0 || idx >= F.N)
        return false;

    if(F.Nleft == -1)
    {
        if(idx < static_cast<int>(F.mvKeysUn.size()))
        {
            kp = F.mvKeysUn[idx];
            return true;
        }
        if(idx < static_cast<int>(F.mvKeys.size()))
        {
            kp = F.mvKeys[idx];
            return true;
        }
        return false;
    }

    if(idx < F.Nleft)
    {
        if(idx >= static_cast<int>(F.mvKeys.size()))
            return false;
        kp = F.mvKeys[idx];
        return true;
    }

    const int rightIdx = idx - F.Nleft;
    if(rightIdx < 0 || rightIdx >= static_cast<int>(F.mvKeysRight.size()))
        return false;

    kp = F.mvKeysRight[rightIdx];
    return true;
}

bool TryComputeFeatureCellIndex(const Frame& F,
                                const int featIdx,
                                const int gridCols,
                                const int gridRows,
                                int& cellIdx)
{
    cv::KeyPoint kp;
    if(!GetFrameKeyPointByIndex(F, featIdx, kp))
        return false;

    const float spanX = std::max(1.0f, F.mnMaxX - F.mnMinX);
    const float spanY = std::max(1.0f, F.mnMaxY - F.mnMinY);
    const float fx = (kp.pt.x - F.mnMinX) / spanX;
    const float fy = (kp.pt.y - F.mnMinY) / spanY;
    const int cellX = std::max(0, std::min(gridCols - 1, static_cast<int>(fx * static_cast<float>(gridCols))));
    const int cellY = std::max(0, std::min(gridRows - 1, static_cast<int>(fy * static_cast<float>(gridRows))));
    cellIdx = cellY * gridCols + cellX;
    return true;
}

void ApplySpatialQuotaToBestCandidates(const Frame& F,
                                       const std::vector<CandidateMatch>& candidates,
                                       const std::vector<int>& bestCandidateIdxIn,
                                       std::vector<int>& bestCandidateIdxOut,
                                       SpatialQuotaStats& stats)
{
    bestCandidateIdxOut = bestCandidateIdxIn;
    stats = SpatialQuotaStats();
    stats.gridCols = GetXFeatMatchSpatialGridCols();
    stats.gridRows = GetXFeatMatchSpatialGridRows();

    std::vector<int> orderedCandidateIdx;
    orderedCandidateIdx.reserve(bestCandidateIdxIn.size());
    for(const int candIdx : bestCandidateIdxIn)
    {
        if(candIdx >= 0 && candIdx < static_cast<int>(candidates.size()))
            orderedCandidateIdx.push_back(candIdx);
    }

    stats.input = static_cast<int>(orderedCandidateIdx.size());
    stats.acceptedStage1 = stats.input;
    stats.acceptedFinal = stats.input;
    if(stats.input <= 0)
        return;

    if(!UseXFeatMatchSpatialQuota() || stats.input < GetXFeatMatchSpatialTrigger())
        return;

    const int totalCells = std::max(1, stats.gridCols * stats.gridRows);
    const float meanPerCell = static_cast<float>(stats.input) / static_cast<float>(totalCells);
    stats.cap = std::max(1, static_cast<int>(std::ceil(meanPerCell * GetXFeatMatchSpatialCapScale())));
    if(stats.input >= totalCells)
        stats.cap = std::max(2, stats.cap);

    std::sort(orderedCandidateIdx.begin(), orderedCandidateIdx.end(),
              [&candidates](const int a, const int b) {
                  if(candidates[a].dist == candidates[b].dist)
                      return a < b;
                  return candidates[a].dist < candidates[b].dist;
              });

    std::vector<int> cellCount(totalCells, 0);
    std::vector<char> keep(candidates.size(), 0);
    std::vector<int> deferred;
    deferred.reserve(orderedCandidateIdx.size());

    //调试: 第一阶段按网格上限保留候选，优先提升匹配结果的空间覆盖度。
    for(const int candIdx : orderedCandidateIdx)
    {
        const CandidateMatch& c = candidates[candIdx];
        if(c.featIdx < 0 || c.featIdx >= F.N || !c.pMP)
            continue;

        int cellIdx = -1;
        if(!TryComputeFeatureCellIndex(F, c.featIdx, stats.gridCols, stats.gridRows, cellIdx))
        {
            keep[candIdx] = 1;
            continue;
        }

        if(cellIdx < 0 || cellIdx >= totalCells)
        {
            keep[candIdx] = 1;
            continue;
        }

        if(cellCount[cellIdx] < stats.cap)
        {
            keep[candIdx] = 1;
            ++cellCount[cellIdx];
        }
        else
        {
            deferred.push_back(candIdx);
        }
    }

    int acceptedStage1 = 0;
    for(const int candIdx : orderedCandidateIdx)
    {
        if(candIdx >= 0 && candIdx < static_cast<int>(keep.size()) && keep[candIdx])
            ++acceptedStage1;
    }

    const int minKeep = std::max(1, static_cast<int>(std::ceil(static_cast<float>(stats.input) * GetXFeatMatchSpatialKeepRatio())));
    int acceptedFinal = acceptedStage1;

    //调试: 第二阶段仅用于回填最低保留量，避免低纹理时段因为均匀化导致匹配数骤降。
    for(const int candIdx : deferred)
    {
        if(acceptedFinal >= minKeep)
            break;
        if(candIdx < 0 || candIdx >= static_cast<int>(keep.size()))
            continue;
        if(keep[candIdx])
            continue;
        keep[candIdx] = 1;
        ++acceptedFinal;
    }

    for(size_t featIdx = 0; featIdx < bestCandidateIdxOut.size(); ++featIdx)
    {
        const int candIdx = bestCandidateIdxOut[featIdx];
        if(candIdx < 0 || candIdx >= static_cast<int>(keep.size()))
            continue;
        if(!keep[candIdx])
            bestCandidateIdxOut[featIdx] = -1;
    }

    stats.applied = true;
    stats.acceptedStage1 = acceptedStage1;
    stats.acceptedFinal = acceptedFinal;
    stats.recovered = std::max(0, acceptedFinal - acceptedStage1);
    stats.dropped = std::max(0, stats.input - acceptedFinal);
}

const cv::Mat& EnsureFloatDescriptors(const cv::Mat& in, cv::Mat& buffer)
{
    if(in.type() == CV_32F)
        return in;
    in.convertTo(buffer, CV_32F);
    return buffer;
}

std::vector<cv::Mat> BuildMapPointFloatDescriptorSet(MapPoint* pMP)
{
    std::vector<cv::Mat> result;
    if(!pMP)
        return result;

    if(!UseXFeatDescriptorBank())
    {
        // XFeat稳态路径: 默认仅使用单描述子，避免多原型策略引入不稳定匹配。
        const cv::Mat fallback = pMP->GetDescriptor();
        if(!fallback.empty())
        {
            cv::Mat buffer;
            const cv::Mat& df = EnsureFloatDescriptors(fallback, buffer);
            if(!df.empty())
                result.push_back(df.clone());
        }
        return result;
    }

    // XFeat路径: 优先使用MapPoint维护的多原型描述子库。
    std::vector<cv::Mat> raw = pMP->GetXFeatDescriptorBank();
    if(raw.empty())
    {
        // 兼容旧路径: bank为空时回退到单描述子。
        const cv::Mat fallback = pMP->GetDescriptor();
        if(!fallback.empty())
            raw.push_back(fallback);
    }

    result.reserve(raw.size());
    for(const cv::Mat& d : raw)
    {
        if(d.empty())
            continue;
        cv::Mat buffer;
        const cv::Mat& df = EnsureFloatDescriptors(d, buffer);
        if(!df.empty())
            result.push_back(df.clone());
    }
    return result;
}

float DescriptorDistanceToSet(const std::vector<cv::Mat>& descriptorSet, const cv::Mat& queryDesc)
{
    float bestDist = std::numeric_limits<float>::infinity();
    for(const cv::Mat& d : descriptorSet)
    {
        const float dist = XFeatMatcher::DescriptorDistance(d, queryDesc);
        if(dist < bestDist)
            bestDist = dist;
    }
    return bestDist;
}

int SelectBestCandidatePerFeature(const std::vector<CandidateMatch>& candidates,
                                  const int nFeatures,
                                  std::vector<int>& bestCandidateIdx)
{
    bestCandidateIdx.assign(nFeatures, -1);
    std::vector<float> bestDist(nFeatures, std::numeric_limits<float>::infinity());

    for(size_t i = 0; i < candidates.size(); ++i)
    {
        const CandidateMatch& c = candidates[i];
        if(c.featIdx < 0 || c.featIdx >= nFeatures || !c.pMP)
            continue;

        if(c.dist < bestDist[c.featIdx])
        {
            bestDist[c.featIdx] = c.dist;
            bestCandidateIdx[c.featIdx] = static_cast<int>(i);
        }
    }

    int nSelected = 0;
    for(int i = 0; i < nFeatures; ++i)
    {
        if(bestCandidateIdx[i] >= 0)
            ++nSelected;
    }
    return nSelected;
}

int CountInlierMask(const cv::Mat& mask)
{
    if(mask.empty())
        return 0;

    const cv::Mat flatMask = mask.reshape(1, 1);
    const int rows = flatMask.cols;
    const uchar* ptr = flatMask.ptr<uchar>(0);
    if(!ptr)
        return 0;

    int inliers = 0;
    for(int i = 0; i < rows; ++i)
    {
        if(ptr[i] != 0)
            ++inliers;
    }
    return inliers;
}

int ApplyInitializationGeometryFilter(const Frame& F1,
                                      const Frame& F2,
                                      std::vector<int>& vnMatches12,
                                      int& geomInliersF,
                                      int& geomInliersH,
                                      char& geomModel)
{
    geomInliersF = 0;
    geomInliersH = 0;
    geomModel = 'N';

    std::vector<int> idx1List;
    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
    idx1List.reserve(vnMatches12.size());
    pts1.reserve(vnMatches12.size());
    pts2.reserve(vnMatches12.size());

    for(size_t i1 = 0; i1 < vnMatches12.size(); ++i1)
    {
        const int idx2 = vnMatches12[i1];
        if(idx2 < 0)
            continue;
        if(i1 >= F1.mvKeysUn.size())
            continue;
        if(idx2 >= static_cast<int>(F2.mvKeysUn.size()))
            continue;

        idx1List.push_back(static_cast<int>(i1));
        pts1.push_back(F1.mvKeysUn[i1].pt);
        pts2.push_back(F2.mvKeysUn[idx2].pt);
    }

    if(pts1.size() < 30)
        return 0;

    cv::Mat maskF, maskH;
    const cv::Mat F12 = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 1.5, 0.99, maskF);
    const cv::Mat H12 = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0, maskH, 2000, 0.995);

    if(!F12.empty() && maskF.rows * maskF.cols == static_cast<int>(pts1.size()))
        geomInliersF = CountInlierMask(maskF);
    if(!H12.empty() && maskH.rows * maskH.cols == static_cast<int>(pts1.size()))
        geomInliersH = CountInlierMask(maskH);

    const cv::Mat* pBestMask = nullptr;
    if(geomInliersF >= geomInliersH && geomInliersF >= 30)
    {
        pBestMask = &maskF;
        geomModel = 'F';
    }
    else if(geomInliersH >= 30)
    {
        pBestMask = &maskH;
        geomModel = 'H';
    }

    if(!pBestMask)
        return 0;

    int rejected = 0;
    for(int i = 0; i < static_cast<int>(idx1List.size()); ++i)
    {
        if(pBestMask->at<uchar>(i) == 0)
        {
            vnMatches12[idx1List[i]] = -1;
            ++rejected;
        }
    }
    return rejected;
}

} // namespace

const float XFeatMatcher::TH_LOW = 0.60f;
const float XFeatMatcher::TH_HIGH = 2.00f;
const int XFeatMatcher::HISTO_LENGTH = 30;

XFeatMatcher::XFeatMatcher(float nnratio, bool checkOri)
    : mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}

float XFeatMatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    if(a.empty() || b.empty())
        return std::numeric_limits<float>::infinity();

    if(a.type() == CV_32F && b.type() == CV_32F)
        return static_cast<float>(cv::norm(a, b, cv::NORM_L2));

    cv::Mat af, bf;
    a.convertTo(af, CV_32F);
    b.convertTo(bf, CV_32F);
    return static_cast<float>(cv::norm(af, bf, cv::NORM_L2));
}

float XFeatMatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos > 0.998f)
        return 2.5f;
    return 4.0f;
}

int XFeatMatcher::SearchByNN(KeyFrame *pKF,
                             Frame &F,
                             std::vector<MapPoint*> &vpMapPointMatches,
                             const float thHighOverride)
{
    //调试: 允许调用侧按场景覆盖绝对距离阈值，未设置时回退到全局TH_HIGH。
    const float thHigh = (std::isfinite(thHighOverride) && thHighOverride > 0.0f)
        ? std::min(2.0f, std::max(0.05f, thHighOverride))
        : TH_HIGH;

    vpMapPointMatches = std::vector<MapPoint*>(F.N, static_cast<MapPoint*>(NULL));

    int rawMatches = 0;
    int ratioMatches = 0;
    int mutualMatches = 0;
    int finalMatches = 0;
    //调试: 拒绝原因统计（仅详细日志打印）。
    int noKnnMatches = 0;
    int distRejectMatches = 0;
    int ratioRejectMatches = 0;
    int nonMutualRejectMatches = 0;
    int oneToOneConflictRejectMatches = 0;
    int queryCount = 0;

    const std::vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();
    if(pKF->mDescriptors.empty() || F.mDescriptors.empty() || vpMapPointsKF.empty())
    {
        if(IsXFeatMatcherDebugEnabled())
        {
            std::cout << "[XFeatMatcher::SearchByNN] frame=" << F.mnId
                      << " raw=" << rawMatches
                      << " ratio=" << ratioMatches
                      << " mutual=" << mutualMatches
                      << " final=" << finalMatches
                      << std::endl;
        }
        return 0;
    }

    cv::Mat descKFBuffer, descFBuffer;
    const cv::Mat& descKF = EnsureFloatDescriptors(pKF->mDescriptors, descKFBuffer);
    const cv::Mat& descF = EnsureFloatDescriptors(F.mDescriptors, descFBuffer);

    const int usableRows = std::min(static_cast<int>(vpMapPointsKF.size()), descKF.rows);
    if(usableRows <= 0 || descF.rows <= 0)
    {
        if(IsXFeatMatcherDebugEnabled())
        {
            std::cout << "[XFeatMatcher::SearchByNN] frame=" << F.mnId
                      << " raw=" << rawMatches
                      << " ratio=" << ratioMatches
                      << " mutual=" << mutualMatches
                      << " final=" << finalMatches
                      << std::endl;
        }
        return 0;
    }

    std::vector<MapPoint*> queryToMP;
    std::vector<int> queryToKFIdx;
    queryToMP.reserve(usableRows);
    queryToKFIdx.reserve(usableRows);

    for(int i = 0; i < usableRows; ++i)
    {
        MapPoint* pMP = vpMapPointsKF[i];
        if(!pMP || pMP->isBad())
            continue;

        queryToMP.push_back(pMP);
        queryToKFIdx.push_back(i);
    }

    if(queryToMP.empty())
    {
        if(IsXFeatMatcherDebugEnabled())
        {
            std::cout << "[XFeatMatcher::SearchByNN] frame=" << F.mnId
                      << " raw=" << rawMatches
                      << " ratio=" << ratioMatches
                      << " mutual=" << mutualMatches
                      << " final=" << finalMatches
                      << std::endl;
        }
        return 0;
    }

    queryCount = static_cast<int>(queryToMP.size());

    cv::Mat queryDesc(static_cast<int>(queryToMP.size()), descKF.cols, CV_32F);
    for(size_t q = 0; q < queryToKFIdx.size(); ++q)
        descKF.row(queryToKFIdx[q]).copyTo(queryDesc.row(static_cast<int>(q)));

    cv::BFMatcher matcher(cv::NORM_L2, false);

    std::vector<std::vector<cv::DMatch>> knnForward;
    matcher.knnMatch(queryDesc, descF, knnForward, 2);

    std::vector<std::vector<cv::DMatch>> knnReverse;
    matcher.knnMatch(descF, queryDesc, knnReverse, 1);

    std::vector<int> reverseBest(descF.rows, -1);
    for(int i = 0; i < static_cast<int>(knnReverse.size()); ++i)
    {
        if(!knnReverse[i].empty())
            reverseBest[i] = knnReverse[i][0].trainIdx;
    }

    std::vector<CandidateMatch> candidates;
    candidates.reserve(knnForward.size());

    for(int q = 0; q < static_cast<int>(knnForward.size()); ++q)
    {
        const std::vector<cv::DMatch>& ms = knnForward[q];
        if(ms.empty())
        {
            ++noKnnMatches;
            continue;
        }

        const cv::DMatch& best = ms[0];
        if(best.trainIdx < 0 || best.trainIdx >= descF.rows)
            continue;

        ++rawMatches;

        // XFeat descriptors are L2-normalized; valid distances are in [0, 2].
        // Keep this gate loose for relocalization robustness.
        if(best.distance > thHigh)
        {
            ++distRejectMatches;
            continue;
        }

        if(ms.size() > 1)
        {
            const cv::DMatch& second = ms[1];
            if(second.distance > 0.0f && best.distance >= mfNNratio * second.distance)
            {
                ++ratioRejectMatches;
                continue;
            }
        }

        ++ratioMatches;

        const bool bMutual = (reverseBest[best.trainIdx] == q);
        if(bMutual)
            ++mutualMatches;

        // XFeat主链: 默认开启mutual硬约束，优先保证进入PnP/位姿优化的匹配纯度。
        if(EnforceMutualForNN() && !bMutual)
        {
            ++nonMutualRejectMatches;
            continue;
        }

        candidates.push_back({best.trainIdx, queryToMP[q], best.distance});
    }

    std::vector<int> bestCandidateIdx;
    SelectBestCandidatePerFeature(candidates, F.N, bestCandidateIdx);
    std::vector<int> bestCandidateIdxSpatial;
    SpatialQuotaStats spatialStats;
    ApplySpatialQuotaToBestCandidates(F, candidates, bestCandidateIdx, bestCandidateIdxSpatial, spatialStats);

    for(int featIdx = 0; featIdx < F.N; ++featIdx)
    {
        const int candIdx = bestCandidateIdxSpatial[featIdx];
        if(candIdx < 0)
            continue;

        const CandidateMatch& c = candidates[candIdx];
        if(!c.pMP)
            continue;

        if(vpMapPointMatches[featIdx])
            continue;

        vpMapPointMatches[featIdx] = c.pMP;
        ++finalMatches;
    }

    oneToOneConflictRejectMatches = std::max(0, ratioMatches - finalMatches - spatialStats.dropped);

    if(IsXFeatMatcherDebugEnabled())
    {
        std::cout << "[XFeatMatcher::SearchByNN] frame=" << F.mnId
                  << " q=" << queryCount
                  << " raw=" << rawMatches
                  << " ratio=" << ratioMatches
                  << " mutual=" << mutualMatches
                  << " final=" << finalMatches
                  << " th_high=" << thHigh
                  << " nnratio=" << mfNNratio
                  << " enforce_mutual=" << (EnforceMutualForNN() ? 1 : 0)
                  << " spatial=" << (spatialStats.applied ? 1 : 0)
                  << " spatial_in=" << spatialStats.input
                  << " spatial_keep=" << spatialStats.acceptedFinal
                  << " spatial_drop=" << spatialStats.dropped
                  << " spatial_cap=" << spatialStats.cap
                  << " spatial_grid=" << spatialStats.gridCols << "x" << spatialStats.gridRows
                  << std::endl;

        if(IsXFeatMatcherVerboseDebugEnabled())
        {
            //调试: 详细拒绝原因，便于定位匹配数骤降发生在哪一步。
            std::cout << "[XFeatMatcher::SearchByNN][verbose] frame=" << F.mnId
                      << " no_knn=" << noKnnMatches
                      << " reject_dist=" << distRejectMatches
                      << " reject_ratio=" << ratioRejectMatches
                      << " reject_non_mutual=" << nonMutualRejectMatches
                      << " reject_one2one=" << oneToOneConflictRejectMatches
                      << " reject_spatial=" << spatialStats.dropped
                      << " desc_kf_rows=" << descKF.rows
                      << " desc_f_rows=" << descF.rows
                      << std::endl;
        }
    }

    return finalMatches;
}

int XFeatMatcher::SearchForInitialization(Frame &F1, Frame &F2,
                                          std::vector<cv::Point2f> &vbPrevMatched,
                                          std::vector<int> &vnMatches12,
                                          int windowSize)
{
    int rawMatches = 0;
    int ratioMatches = 0;
    int mutualMatches = 0;
    int geomRejectMatches = 0;
    int geomInliersF = 0;
    int geomInliersH = 0;
    char geomModel = 'N';
    //调试: 初始化阶段拒绝原因统计（仅详细日志打印）。
    int emptyWindowRejectMatches = 0;
    int distRejectMatches = 0;
    int ratioRejectMatches = 0;

    vnMatches12 = std::vector<int>(F1.mvKeysUn.size(), -1);

    if(F1.mDescriptors.empty() || F2.mDescriptors.empty())
    {
        if(IsXFeatMatcherDebugEnabled())
        {
            std::cout << "[XFeatMatcher::SearchForInitialization] frame1=" << F1.mnId
                      << " frame2=" << F2.mnId
                      << " raw=" << rawMatches
                      << " ratio=" << ratioMatches
                      << " mutual=" << mutualMatches
                      << " final=0"
                      << std::endl;
        }
        return 0;
    }

    cv::Mat desc1Buffer, desc2Buffer;
    const cv::Mat& desc1 = EnsureFloatDescriptors(F1.mDescriptors, desc1Buffer);
    const cv::Mat& desc2 = EnsureFloatDescriptors(F2.mDescriptors, desc2Buffer);

    std::vector<int> rotHist[HISTO_LENGTH];
    for(int i = 0; i < HISTO_LENGTH; ++i)
        rotHist[i].reserve(500);
    const float factor = 1.0f / HISTO_LENGTH;

    std::vector<float> vMatchedDistance(F2.mvKeysUn.size(), std::numeric_limits<float>::infinity());
    std::vector<int> vnMatches21(F2.mvKeysUn.size(), -1);

    int nmatches = 0;

    const size_t maxI1 = std::min(F1.mvKeysUn.size(), static_cast<size_t>(desc1.rows));
    for(size_t i1 = 0; i1 < maxI1; ++i1)
    {
        const cv::KeyPoint& kp1 = F1.mvKeysUn[i1];
        const int level1 = kp1.octave;
        const int minLevel2 = (F2.mnScaleLevels > 0) ? std::max(0, level1 - 1) : -1;
        const int maxLevel2 = (F2.mnScaleLevels > 0) ? std::min(F2.mnScaleLevels - 1, level1 + 1) : -1;

        const std::vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,
                                                                    vbPrevMatched[i1].y,
                                                                    windowSize,
                                                                    minLevel2,
                                                                    maxLevel2);

        if(vIndices2.empty())
        {
            ++emptyWindowRejectMatches;
            continue;
        }

        const cv::Mat d1 = desc1.row(static_cast<int>(i1));

        float bestDist = std::numeric_limits<float>::infinity();
        float bestDist2 = std::numeric_limits<float>::infinity();
        int bestIdx2 = -1;

        for(size_t idx : vIndices2)
        {
            if(idx >= static_cast<size_t>(desc2.rows) || idx >= vMatchedDistance.size())
                continue;

            const cv::Mat d2 = desc2.row(static_cast<int>(idx));
            const float dist = DescriptorDistance(d1, d2);

            if(vMatchedDistance[idx] <= dist)
                continue;

            if(dist < bestDist)
            {
                bestDist2 = bestDist;
                bestDist = dist;
                bestIdx2 = static_cast<int>(idx);
            }
            else if(dist < bestDist2)
            {
                bestDist2 = dist;
            }
        }

        if(bestIdx2 < 0)
            continue;

        ++rawMatches;

        if(bestDist > TH_LOW)
        {
            ++distRejectMatches;
            continue;
        }

        if(!(bestDist < mfNNratio * bestDist2))
        {
            ++ratioRejectMatches;
            continue;
        }

        ++ratioMatches;

        if(vnMatches21[bestIdx2] >= 0)
        {
            vnMatches12[vnMatches21[bestIdx2]] = -1;
            --nmatches;
        }

        vnMatches12[i1] = bestIdx2;
        vnMatches21[bestIdx2] = static_cast<int>(i1);
        vMatchedDistance[bestIdx2] = bestDist;
        ++nmatches;

        if(mbCheckOrientation)
        {
            float rot = F1.mvKeysUn[i1].angle - F2.mvKeysUn[bestIdx2].angle;
            if(rot < 0.0f)
                rot += 360.0f;
            int bin = round(rot * factor);
            if(bin == HISTO_LENGTH)
                bin = 0;
            assert(bin >= 0 && bin < HISTO_LENGTH);
            rotHist[bin].push_back(static_cast<int>(i1));
        }
    }

    if(mbCheckOrientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for(int i = 0; i < HISTO_LENGTH; ++i)
        {
            if(i == ind1 || i == ind2 || i == ind3)
                continue;

            for(size_t j = 0, jend = rotHist[i].size(); j < jend; ++j)
            {
                const int idx1 = rotHist[i][j];
                if(vnMatches12[idx1] >= 0)
                {
                    vnMatches12[idx1] = -1;
                    --nmatches;
                }
            }
        }
    }

    //调试: 几何一致性二次过滤（F/H RANSAC），抑制高特征数量下的重复纹理误匹配。
    geomRejectMatches = ApplyInitializationGeometryFilter(F1, F2, vnMatches12, geomInliersF, geomInliersH, geomModel);
    nmatches = std::max(0, nmatches - geomRejectMatches);

    mutualMatches = 0;
    for(size_t i1 = 0; i1 < vnMatches12.size(); ++i1)
    {
        if(vnMatches12[i1] >= 0)
            ++mutualMatches;
    }
    nmatches = mutualMatches;

    // Update prev matched points.
    for(size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; ++i1)
    {
        const int idx2 = vnMatches12[i1];
        if(idx2 >= 0 && idx2 < static_cast<int>(F2.mvKeysUn.size()))
            vbPrevMatched[i1] = F2.mvKeysUn[idx2].pt;
    }

    if(IsXFeatMatcherDebugEnabled())
    {
        std::cout << "[XFeatMatcher::SearchForInitialization] frame1=" << F1.mnId
                  << " frame2=" << F2.mnId
                  << " raw=" << rawMatches
                  << " ratio=" << ratioMatches
                  << " mutual=" << mutualMatches
                  << " final=" << nmatches
                  << " th_low=" << TH_LOW
                  << " nnratio=" << mfNNratio
                  << std::endl;

        if(IsXFeatMatcherVerboseDebugEnabled())
        {
            //调试: 初始化匹配位移统计，用于识别“高匹配但低有效基线”场景。
            std::vector<float> displacements;
            displacements.reserve(static_cast<size_t>(nmatches));
            double dispSum = 0.0;
            for(size_t i1 = 0; i1 < vnMatches12.size(); ++i1)
            {
                const int idx2 = vnMatches12[i1];
                if(idx2 < 0 || i1 >= F1.mvKeysUn.size() || idx2 >= static_cast<int>(F2.mvKeysUn.size()))
                    continue;

                const cv::Point2f d = F2.mvKeysUn[idx2].pt - F1.mvKeysUn[i1].pt;
                const float disp = std::sqrt(d.x * d.x + d.y * d.y);
                displacements.push_back(disp);
                dispSum += static_cast<double>(disp);
            }
            float meanDisp = 0.0f;
            float medianDisp = 0.0f;
            if(!displacements.empty())
            {
                meanDisp = static_cast<float>(dispSum / static_cast<double>(displacements.size()));
                const size_t mid = displacements.size() / 2;
                std::nth_element(displacements.begin(), displacements.begin() + mid, displacements.end());
                medianDisp = displacements[mid];
            }

            //调试: 详细拒绝原因，便于区分投影窗口不足与描述子过滤过严。
            std::cout << "[XFeatMatcher::SearchForInitialization][verbose] frame1=" << F1.mnId
                      << " frame2=" << F2.mnId
                      << " reject_empty_window=" << emptyWindowRejectMatches
                      << " reject_dist=" << distRejectMatches
                      << " reject_ratio=" << ratioRejectMatches
                      << " reject_geom=" << geomRejectMatches
                      << " geom_model=" << geomModel
                      << " geom_inliers_f=" << geomInliersF
                      << " geom_inliers_h=" << geomInliersH
                      << " disp_mean=" << meanDisp
                      << " disp_median=" << medianDisp
                      << " desc1_rows=" << desc1.rows
                      << " desc2_rows=" << desc2.rows
                      << std::endl;
        }
    }

    return nmatches;
}

int XFeatMatcher::SearchForTriangulation(KeyFrame *pKF1,
                                         KeyFrame *pKF2,
                                         std::vector<std::pair<size_t, size_t>> &vMatchedPairs,
                                         const bool bOnlyStereo,
                                         const bool bCoarse)
{
    int rawMatches = 0;
    int ratioMatches = 0;
    int mutualMatches = 0;
    int finalMatches = 0;
    //调试: 三角化阶段拒绝原因统计（仅详细日志打印）。
    int skipMappedKF1 = 0;
    int skipMappedKF2 = 0;
    int skipStereoKF1 = 0;
    int skipStereoKF2 = 0;
    int skipDist = 0;
    int skipRatio = 0;
    int skipEpipole = 0;
    int skipEpipolar = 0;
    int skipNoKnn = 0;

    vMatchedPairs.clear();

    if(!pKF1 || !pKF2 || pKF1->mDescriptors.empty() || pKF2->mDescriptors.empty())
    {
        if(IsXFeatMatcherDebugEnabled())
        {
            std::cout << "[XFeatMatcher::SearchForTriangulation] kf1="
                      << (pKF1 ? std::to_string(pKF1->mnId) : std::string("null"))
                      << " kf2="
                      << (pKF2 ? std::to_string(pKF2->mnId) : std::string("null"))
                      << " raw=" << rawMatches
                      << " ratio=" << ratioMatches
                      << " mutual=" << mutualMatches
                      << " final=" << finalMatches
                      << std::endl;
        }
        return 0;
    }

    cv::Mat desc1Buffer, desc2Buffer;
    const cv::Mat& desc1 = EnsureFloatDescriptors(pKF1->mDescriptors, desc1Buffer);
    const cv::Mat& desc2 = EnsureFloatDescriptors(pKF2->mDescriptors, desc2Buffer);
    if(desc1.empty() || desc2.empty())
        return 0;

    // Compute epipole in second image.
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

    const int rows1 = std::min(pKF1->N, desc1.rows);
    const int rows2 = std::min(pKF2->N, desc2.rows);
    if(rows1 <= 0 || rows2 <= 0)
        return 0;

    std::vector<int> queryToIdx1;
    queryToIdx1.reserve(rows1);
    for(int idx1 = 0; idx1 < rows1; ++idx1)
    {
        if(pKF1->GetMapPoint(idx1))
        {
            ++skipMappedKF1;
            continue;
        }

        const bool bStereo1 = (!pKF1->mpCamera2 && idx1 < static_cast<int>(pKF1->mvuRight.size()) && pKF1->mvuRight[idx1] >= 0);
        if(bOnlyStereo && !bStereo1)
        {
            ++skipStereoKF1;
            continue;
        }

        queryToIdx1.push_back(idx1);
    }

    if(queryToIdx1.empty())
    {
        if(IsXFeatMatcherDebugEnabled())
        {
            std::cout << "[XFeatMatcher::SearchForTriangulation] kf1=" << pKF1->mnId
                      << " kf2=" << pKF2->mnId
                      << " raw=" << rawMatches
                      << " ratio=" << ratioMatches
                      << " mutual=" << mutualMatches
                      << " final=" << finalMatches
                      << std::endl;
        }
        return 0;
    }

    cv::Mat queryDesc(static_cast<int>(queryToIdx1.size()), desc1.cols, CV_32F);
    for(size_t q = 0; q < queryToIdx1.size(); ++q)
        desc1.row(queryToIdx1[q]).copyTo(queryDesc.row(static_cast<int>(q)));

    cv::BFMatcher matcher(cv::NORM_L2, false);
    const int kKnn = std::max(1, std::min(GetXFeatTriangulationKnn(), rows2));

    std::vector<std::vector<cv::DMatch>> knnForward;
    matcher.knnMatch(queryDesc, desc2.rowRange(0, rows2), knnForward, kKnn);

    std::vector<std::vector<cv::DMatch>> knnReverse;
    matcher.knnMatch(desc2.rowRange(0, rows2), queryDesc, knnReverse, 1);

    std::vector<int> reverseBest(rows2, -1);
    for(int i = 0; i < static_cast<int>(knnReverse.size()); ++i)
    {
        if(!knnReverse[i].empty())
            reverseBest[i] = knnReverse[i][0].trainIdx;
    }

    struct TriangulationCandidate
    {
        int idx1 = -1;
        int idx2 = -1;
        float dist = std::numeric_limits<float>::infinity();
        bool mutual = false;
    };
    std::vector<TriangulationCandidate> candidates;
    candidates.reserve(knnForward.size());

    for(int q = 0; q < static_cast<int>(knnForward.size()); ++q)
    {
        const int idx1 = queryToIdx1[q];
        if(idx1 < 0 || idx1 >= rows1)
            continue;

        const cv::KeyPoint &kp1 = (pKF1->NLeft == -1) ? pKF1->mvKeysUn[idx1]
                                                      : (idx1 < pKF1->NLeft) ? pKF1->mvKeys[idx1]
                                                                             : pKF1->mvKeysRight[idx1 - pKF1->NLeft];
        const bool bStereo1 = (!pKF1->mpCamera2 && idx1 < static_cast<int>(pKF1->mvuRight.size()) && pKF1->mvuRight[idx1] >= 0);
        const bool bRight1 = (pKF1->NLeft == -1 || idx1 < pKF1->NLeft) ? false : true;

        float bestDist = std::numeric_limits<float>::infinity();
        float secondDist = std::numeric_limits<float>::infinity();
        int bestIdx2 = -1;

        for(const cv::DMatch& m : knnForward[q])
        {
            const int idx2 = m.trainIdx;
            if(idx2 < 0 || idx2 >= rows2)
                continue;

            if(pKF2->GetMapPoint(idx2))
            {
                ++skipMappedKF2;
                continue;
            }

            const bool bStereo2 = (!pKF2->mpCamera2 && idx2 < static_cast<int>(pKF2->mvuRight.size()) && pKF2->mvuRight[idx2] >= 0);
            if(bOnlyStereo && !bStereo2)
            {
                ++skipStereoKF2;
                continue;
            }

            if(m.distance < bestDist)
            {
                secondDist = bestDist;
                bestDist = m.distance;
                bestIdx2 = idx2;
            }
            else if(m.distance < secondDist)
            {
                secondDist = m.distance;
            }
        }

        if(bestIdx2 < 0)
        {
            ++skipNoKnn;
            continue;
        }

        ++rawMatches;

        if(bestDist > TH_LOW)
        {
            ++skipDist;
            continue;
        }

        if(std::isfinite(secondDist) && secondDist > 0.0f && bestDist >= mfNNratio * secondDist)
        {
            ++skipRatio;
            continue;
        }
        ++ratioMatches;

        const cv::KeyPoint &kp2 = (pKF2->NLeft == -1) ? pKF2->mvKeysUn[bestIdx2]
                                                      : (bestIdx2 < pKF2->NLeft) ? pKF2->mvKeys[bestIdx2]
                                                                                 : pKF2->mvKeysRight[bestIdx2 - pKF2->NLeft];
        const bool bStereo2 = (!pKF2->mpCamera2 && bestIdx2 < static_cast<int>(pKF2->mvuRight.size()) && pKF2->mvuRight[bestIdx2] >= 0);
        const bool bRight2 = (pKF2->NLeft == -1 || bestIdx2 < pKF2->NLeft) ? false : true;

        if(!bStereo1 && !bStereo2 && !pKF1->mpCamera2)
        {
            const float distex = ep(0) - kp2.pt.x;
            const float distey = ep(1) - kp2.pt.y;
            if(distex * distex + distey * distey < 100.0f * pKF2->mvScaleFactors[kp2.octave])
            {
                ++skipEpipole;
                continue;
            }
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
            ++skipEpipolar;
            continue;
        }

        const bool bMutual = (bestIdx2 >= 0 && bestIdx2 < static_cast<int>(reverseBest.size()) && reverseBest[bestIdx2] == q);
        if(bMutual)
            ++mutualMatches;

        candidates.push_back({idx1, bestIdx2, bestDist, bMutual});
    }

    // Keep one-to-one matches on KF2 side by smallest descriptor distance.
    std::sort(candidates.begin(), candidates.end(), [](const TriangulationCandidate& a, const TriangulationCandidate& b) {
        return a.dist < b.dist;
    });

    std::vector<char> usedIdx1(rows1, 0);
    std::vector<char> usedIdx2(rows2, 0);
    vMatchedPairs.reserve(candidates.size());

    for(const TriangulationCandidate& c : candidates)
    {
        if(c.idx1 < 0 || c.idx1 >= rows1 || c.idx2 < 0 || c.idx2 >= rows2)
            continue;
        if(usedIdx1[c.idx1] || usedIdx2[c.idx2])
            continue;

        usedIdx1[c.idx1] = 1;
        usedIdx2[c.idx2] = 1;
        vMatchedPairs.emplace_back(static_cast<size_t>(c.idx1), static_cast<size_t>(c.idx2));
        ++finalMatches;
    }

    if(IsXFeatMatcherDebugEnabled())
    {
        std::cout << "[XFeatMatcher::SearchForTriangulation] kf1=" << pKF1->mnId
                  << " kf2=" << pKF2->mnId
                  << " raw=" << rawMatches
                  << " ratio=" << ratioMatches
                  << " mutual=" << mutualMatches
                  << " final=" << finalMatches
                  << " th_low=" << TH_LOW
                  << " nnratio=" << mfNNratio
                  << " knn=" << kKnn
                  << std::endl;

        if(IsXFeatMatcherVerboseDebugEnabled())
        {
            //调试: 细分拒绝来源，判断是几何约束还是描述子过滤导致新增点不足。
            std::cout << "[XFeatMatcher::SearchForTriangulation][verbose] kf1=" << pKF1->mnId
                      << " kf2=" << pKF2->mnId
                      << " query_total=" << queryToIdx1.size()
                      << " skip_mapped_kf1=" << skipMappedKF1
                      << " skip_mapped_kf2=" << skipMappedKF2
                      << " skip_stereo_kf1=" << skipStereoKF1
                      << " skip_stereo_kf2=" << skipStereoKF2
                      << " skip_no_knn=" << skipNoKnn
                      << " reject_dist=" << skipDist
                      << " reject_ratio=" << skipRatio
                      << " reject_epipole=" << skipEpipole
                      << " reject_epipolar=" << skipEpipolar
                      << " candidate_pool=" << candidates.size()
                      << std::endl;
        }
    }

    return finalMatches;
}

int XFeatMatcher::SearchByProjection(Frame &F,
                                     const std::vector<MapPoint*> &vpMapPoints,
                                     const float th,
                                     const bool bFarPoints,
                                     const float thFarPoints,
                                     const float thHighOverride)
{
    //调试: 投影匹配可按调用路径覆盖绝对距离阈值，便于区分 motion/local/ref fallback 调优。
    const float thHigh = (std::isfinite(thHighOverride) && thHighOverride > 0.0f)
        ? std::min(2.0f, std::max(0.05f, thHighOverride))
        : TH_HIGH;

    int rawMatches = 0;
    int ratioMatches = 0;
    //调试: 投影匹配拒绝原因统计（仅详细日志打印）。
    int projectedMapPoints = 0;
    int skippedNotInView = 0;
    int skippedFarPoints = 0;
    int skippedBadPoints = 0;
    int skippedNoDescriptor = 0;
    int emptyAreaRejectMatches = 0;
    int distRejectMatches = 0;
    int ratioRejectMatches = 0;
    int assignRejectMatches = 0;

    if(F.mDescriptors.empty())
    {
        if(IsXFeatMatcherDebugEnabled())
        {
            std::cout << "[XFeatMatcher::SearchByProjection(MapPoints)] frame=" << F.mnId
                      << " raw=" << rawMatches
                      << " ratio=" << ratioMatches
                      << " mutual=0 final=0"
                      << std::endl;
        }
        return 0;
    }

    cv::Mat descFBuffer;
    const cv::Mat& descF = EnsureFloatDescriptors(F.mDescriptors, descFBuffer);

    std::vector<CandidateMatch> candidates;
    candidates.reserve(vpMapPoints.size() * 2);

    const bool bFactor = th != 1.0f;

    for(size_t iMP = 0; iMP < vpMapPoints.size(); ++iMP)
    {
        MapPoint* pMP = vpMapPoints[iMP];
        if(!pMP)
            continue;

        if(!pMP->mbTrackInView && !pMP->mbTrackInViewR)
        {
            ++skippedNotInView;
            continue;
        }

        if(bFarPoints && pMP->mTrackDepth > thFarPoints)
        {
            ++skippedFarPoints;
            continue;
        }

        if(pMP->isBad())
        {
            ++skippedBadPoints;
            continue;
        }

        // XFeat路径: 对每个MapPoint获取多原型描述子集合，后续匹配取最小L2距离。
        const std::vector<cv::Mat> mpDescSet = BuildMapPointFloatDescriptorSet(pMP);
        if(mpDescSet.empty())
        {
            ++skippedNoDescriptor;
            continue;
        }
        ++projectedMapPoints;

        if(pMP->mbTrackInView)
        {
            const int nPredictedLevel = pMP->mnTrackScaleLevel;
            const int nScaleLevels = static_cast<int>(F.mvScaleFactors.size());
            if(nPredictedLevel < 0 || nPredictedLevel >= nScaleLevels)
                continue;

            float r = RadiusByViewingCos(pMP->mTrackViewCos);
            if(bFactor)
                r *= th;

            int minLevel = -1;
            int maxLevel = -1;
            BuildProjectionLevelRange(nPredictedLevel,
                                      nScaleLevels,
                                      GetXFeatProjMapMinOffset(),
                                      GetXFeatProjMapMaxOffset(),
                                      minLevel,
                                      maxLevel);

            const std::vector<size_t> vIndices = F.GetFeaturesInArea(
                pMP->mTrackProjX,
                pMP->mTrackProjY,
                r * F.mvScaleFactors[nPredictedLevel],
                minLevel,
                maxLevel);
            if(vIndices.empty())
                ++emptyAreaRejectMatches;

            float bestDist = std::numeric_limits<float>::infinity();
            float bestDist2 = std::numeric_limits<float>::infinity();
            int bestLevel = -1;
            int bestLevel2 = -1;
            int bestIdx = -1;

            for(size_t idx : vIndices)
            {
                if(idx >= static_cast<size_t>(F.N) || idx >= static_cast<size_t>(descF.rows))
                    continue;

                if(F.mvpMapPoints[idx] && F.mvpMapPoints[idx]->Observations() > 0)
                    continue;

                if(F.Nleft == -1 && idx < F.mvuRight.size() && F.mvuRight[idx] > 0)
                {
                    const float er = std::fabs(pMP->mTrackProjXR - F.mvuRight[idx]);
                    if(er > r * F.mvScaleFactors[nPredictedLevel])
                        continue;
                }

                const float dist = DescriptorDistanceToSet(mpDescSet, descF.row(static_cast<int>(idx)));

                if(dist < bestDist)
                {
                    bestDist2 = bestDist;
                    bestDist = dist;
                    bestLevel2 = bestLevel;

                    if(F.Nleft == -1)
                        bestLevel = F.mvKeysUn[idx].octave;
                    else if(static_cast<int>(idx) < F.Nleft)
                        bestLevel = F.mvKeys[idx].octave;
                    else
                        bestLevel = F.mvKeysRight[idx - F.Nleft].octave;

                    bestIdx = static_cast<int>(idx);
                }
                else if(dist < bestDist2)
                {
                    if(F.Nleft == -1)
                        bestLevel2 = F.mvKeysUn[idx].octave;
                    else if(static_cast<int>(idx) < F.Nleft)
                        bestLevel2 = F.mvKeys[idx].octave;
                    else
                        bestLevel2 = F.mvKeysRight[idx - F.Nleft].octave;

                    bestDist2 = dist;
                }
            }

            if(bestIdx >= 0)
            {
                ++rawMatches;
                if(bestDist > thHigh)
                {
                    ++distRejectMatches;
                }
                else if(ShouldRejectByRatio(bestDist, bestDist2, mfNNratio, bestLevel, bestLevel2))
                {
                    // Ratio rejected.
                    ++ratioRejectMatches;
                }
                else
                {
                    ++ratioMatches;
                    candidates.push_back({bestIdx, pMP, bestDist});
                }
            }
        }

        if(F.Nleft != -1 && pMP->mbTrackInViewR)
        {
            const int nPredictedLevel = pMP->mnTrackScaleLevelR;
            const int nScaleLevels = static_cast<int>(F.mvScaleFactors.size());
            if(nPredictedLevel < 0 || nPredictedLevel >= nScaleLevels)
                continue;

            const float r = RadiusByViewingCos(pMP->mTrackViewCosR);

            int minLevel = -1;
            int maxLevel = -1;
            BuildProjectionLevelRange(nPredictedLevel,
                                      nScaleLevels,
                                      GetXFeatProjMapMinOffset(),
                                      GetXFeatProjMapMaxOffset(),
                                      minLevel,
                                      maxLevel);

            const std::vector<size_t> vIndices = F.GetFeaturesInArea(
                pMP->mTrackProjXR,
                pMP->mTrackProjYR,
                r * F.mvScaleFactors[nPredictedLevel],
                minLevel,
                maxLevel,
                true);
            if(vIndices.empty())
                ++emptyAreaRejectMatches;

            float bestDist = std::numeric_limits<float>::infinity();
            float bestDist2 = std::numeric_limits<float>::infinity();
            int bestLevel = -1;
            int bestLevel2 = -1;
            int bestIdx = -1;

            for(size_t idx : vIndices)
            {
                const int featIdx = static_cast<int>(idx) + F.Nleft;
                if(featIdx < 0 || featIdx >= F.N || featIdx >= descF.rows)
                    continue;

                if(F.mvpMapPoints[featIdx] && F.mvpMapPoints[featIdx]->Observations() > 0)
                    continue;

                const float dist = DescriptorDistanceToSet(mpDescSet, descF.row(featIdx));

                if(dist < bestDist)
                {
                    bestDist2 = bestDist;
                    bestDist = dist;
                    bestLevel2 = bestLevel;
                    bestLevel = F.mvKeysRight[idx].octave;
                    bestIdx = static_cast<int>(idx);
                }
                else if(dist < bestDist2)
                {
                    bestLevel2 = F.mvKeysRight[idx].octave;
                    bestDist2 = dist;
                }
            }

            if(bestIdx >= 0)
            {
                ++rawMatches;
                if(bestDist > thHigh)
                {
                    ++distRejectMatches;
                }
                else if(ShouldRejectByRatio(bestDist, bestDist2, mfNNratio, bestLevel, bestLevel2))
                {
                    // Ratio rejected.
                    ++ratioRejectMatches;
                }
                else
                {
                    ++ratioMatches;
                    candidates.push_back({bestIdx + F.Nleft, pMP, bestDist});
                }
            }
        }
    }

    std::vector<int> bestCandidateIdx;
    const int mutualMatches = SelectBestCandidatePerFeature(candidates, F.N, bestCandidateIdx);
    std::vector<int> bestCandidateIdxSpatial;
    SpatialQuotaStats spatialStats;
    ApplySpatialQuotaToBestCandidates(F, candidates, bestCandidateIdx, bestCandidateIdxSpatial, spatialStats);

    auto assignIfFree = [&F](const int idx, MapPoint* pMP) -> bool
    {
        if(idx < 0 || idx >= F.N || !pMP)
            return false;

        MapPoint* pCurrent = F.mvpMapPoints[idx];
        if(pCurrent == pMP)
            return false;

        if(pCurrent && pCurrent->Observations() > 0)
            return false;

        F.mvpMapPoints[idx] = pMP;
        return true;
    };

    int finalMatches = 0;

    for(int featIdx = 0; featIdx < F.N; ++featIdx)
    {
        const int candIdx = bestCandidateIdxSpatial[featIdx];
        if(candIdx < 0)
            continue;

        const CandidateMatch& c = candidates[candIdx];
        if(!c.pMP)
            continue;

        if(assignIfFree(featIdx, c.pMP))
            ++finalMatches;
        else
            ++assignRejectMatches;

        if(F.Nleft == -1)
            continue;

        if(featIdx < F.Nleft)
        {
            if(featIdx < static_cast<int>(F.mvLeftToRightMatch.size()))
            {
                const int rightIdxLocal = F.mvLeftToRightMatch[featIdx];
                if(rightIdxLocal >= 0)
                {
                    const int rightFeat = rightIdxLocal + F.Nleft;
                    if(assignIfFree(rightFeat, c.pMP))
                        ++finalMatches;
                }
            }
        }
        else
        {
            const int rightIdxLocal = featIdx - F.Nleft;
            if(rightIdxLocal >= 0 && rightIdxLocal < static_cast<int>(F.mvRightToLeftMatch.size()))
            {
                const int leftFeat = F.mvRightToLeftMatch[rightIdxLocal];
                if(leftFeat >= 0 && leftFeat < F.Nleft)
                {
                    if(assignIfFree(leftFeat, c.pMP))
                        ++finalMatches;
                }
            }
        }
    }

    if(IsXFeatMatcherDebugEnabled())
    {
        //调试: `mutual` 这里表示一对一去重后的候选数量，不是双向BF互检。
        std::cout << "[XFeatMatcher::SearchByProjection(MapPoints)] frame=" << F.mnId
                  << " raw=" << rawMatches
                  << " ratio=" << ratioMatches
                  << " mutual=" << mutualMatches
                  << " final=" << finalMatches
                  << " th_high=" << thHigh
                  << " nnratio=" << mfNNratio
                  << " level_gate=" << (UseXFeatProjectionLevelGate() ? 1 : 0)
                  << " lvl_off=[" << GetXFeatProjMapMinOffset() << "," << GetXFeatProjMapMaxOffset() << "]"
                  << " legacy_level_ratio=" << (UseLegacyLevelAwareRatioGate() ? 1 : 0)
                  << " spatial=" << (spatialStats.applied ? 1 : 0)
                  << " spatial_in=" << spatialStats.input
                  << " spatial_keep=" << spatialStats.acceptedFinal
                  << " spatial_drop=" << spatialStats.dropped
                  << " spatial_cap=" << spatialStats.cap
                  << " spatial_grid=" << spatialStats.gridCols << "x" << spatialStats.gridRows
                  << std::endl;

        if(IsXFeatMatcherVerboseDebugEnabled())
        {
            //调试: 详细拒绝原因，便于区分视锥/搜索半径问题与描述子过滤问题。
            std::cout << "[XFeatMatcher::SearchByProjection(MapPoints)][verbose] frame=" << F.mnId
                      << " input_mp=" << vpMapPoints.size()
                      << " skip_not_in_view=" << skippedNotInView
                      << " skip_far=" << skippedFarPoints
                      << " skip_bad=" << skippedBadPoints
                      << " skip_no_desc=" << skippedNoDescriptor
                      << " projected_mp=" << projectedMapPoints
                      << " reject_empty_area=" << emptyAreaRejectMatches
                      << " reject_dist=" << distRejectMatches
                      << " reject_ratio=" << ratioRejectMatches
                      << " reject_assign=" << assignRejectMatches
                      << " reject_spatial=" << spatialStats.dropped
                      << " candidate_pool=" << candidates.size()
                      << std::endl;
        }
    }

    return finalMatches;
}

int XFeatMatcher::SearchByProjection(Frame &CurrentFrame,
                                     const Frame &LastFrame,
                                     const float th,
                                     const bool bMono,
                                     const float thHighOverride)
{
    //调试: LastFrame->CurrentFrame投影路径独立阈值，便于压制运动模型阶段误匹配。
    const float thHigh = (std::isfinite(thHighOverride) && thHighOverride > 0.0f)
        ? std::min(2.0f, std::max(0.05f, thHighOverride))
        : TH_HIGH;

    int rawMatches = 0;
    int ratioMatches = 0;
    //调试: 上一帧投影匹配拒绝原因统计（仅详细日志打印）。
    int projectedMapPoints = 0;
    int skippedNoMapPoint = 0;
    int skippedLastOutlier = 0;
    int skippedNoDescriptor = 0;
    int skippedNegativeDepth = 0;
    int skippedOutsideImage = 0;
    int skippedNoScaleFactors = 0;
    int emptyAreaRejectMatches = 0;
    int distRejectMatches = 0;
    int ratioRejectMatches = 0;
    int assignRejectMatches = 0;

    if(CurrentFrame.mDescriptors.empty())
    {
        if(IsXFeatMatcherDebugEnabled())
        {
            std::cout << "[XFeatMatcher::SearchByProjection(LastFrame)] frame=" << CurrentFrame.mnId
                      << " raw=" << rawMatches
                      << " ratio=" << ratioMatches
                      << " mutual=0 final=0"
                      << std::endl;
        }
        return 0;
    }

    cv::Mat descCurrentBuffer;
    const cv::Mat& descCurrent = EnsureFloatDescriptors(CurrentFrame.mDescriptors, descCurrentBuffer);

    const Sophus::SE3f Tcw = CurrentFrame.GetPose();
    const Eigen::Vector3f twc = Tcw.inverse().translation();

    const Sophus::SE3f Tlw = LastFrame.GetPose();
    const Eigen::Vector3f tlc = Tlw * twc;

    const bool bForward = tlc(2) > CurrentFrame.mb && !bMono;
    const bool bBackward = -tlc(2) > CurrentFrame.mb && !bMono;

    std::vector<CandidateMatch> candidates;
    candidates.reserve(static_cast<size_t>(LastFrame.N) * 2);

    for(int i = 0; i < LastFrame.N; ++i)
    {
        if(i >= static_cast<int>(LastFrame.mvpMapPoints.size()))
            continue;

        MapPoint* pMP = LastFrame.mvpMapPoints[i];
        if(!pMP)
        {
            ++skippedNoMapPoint;
            continue;
        }
        if(i < static_cast<int>(LastFrame.mvbOutlier.size()) && LastFrame.mvbOutlier[i])
        {
            ++skippedLastOutlier;
            continue;
        }

        // XFeat路径: 使用MapPoint多原型描述子集合提升跨视角匹配稳定性。
        const std::vector<cv::Mat> mpDescSet = BuildMapPointFloatDescriptorSet(pMP);
        if(mpDescSet.empty())
        {
            ++skippedNoDescriptor;
            continue;
        }
        ++projectedMapPoints;

        Eigen::Vector3f x3Dw = pMP->GetWorldPos();
        Eigen::Vector3f x3Dc = Tcw * x3Dw;

        const float invzc = 1.0f / x3Dc(2);
        if(invzc < 0)
        {
            ++skippedNegativeDepth;
            continue;
        }

        Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dc);
        if(uv(0) < CurrentFrame.mnMinX || uv(0) > CurrentFrame.mnMaxX)
        {
            ++skippedOutsideImage;
            continue;
        }
        if(uv(1) < CurrentFrame.mnMinY || uv(1) > CurrentFrame.mnMaxY)
        {
            ++skippedOutsideImage;
            continue;
        }

        int nLastOctave = 0;
        if(LastFrame.Nleft == -1)
        {
            if(i >= static_cast<int>(LastFrame.mvKeysUn.size()))
                continue;
            nLastOctave = LastFrame.mvKeysUn[i].octave;
        }
        else if(i < LastFrame.Nleft)
        {
            if(i >= static_cast<int>(LastFrame.mvKeys.size()))
                continue;
            nLastOctave = LastFrame.mvKeys[i].octave;
        }
        else
        {
            const int iR = i - LastFrame.Nleft;
            if(iR < 0 || iR >= static_cast<int>(LastFrame.mvKeysRight.size()))
                continue;
            nLastOctave = LastFrame.mvKeysRight[iR].octave;
        }

        if(CurrentFrame.mvScaleFactors.empty())
        {
            ++skippedNoScaleFactors;
            continue;
        }

        nLastOctave = std::max(0, std::min(nLastOctave, static_cast<int>(CurrentFrame.mvScaleFactors.size()) - 1));

        const float radius = th * CurrentFrame.mvScaleFactors[nLastOctave];

        int minLevel = -1;
        int maxLevel = -1;
        BuildLastFrameProjectionLevelRange(nLastOctave,
                                           static_cast<int>(CurrentFrame.mvScaleFactors.size()),
                                           bForward,
                                           bBackward,
                                           minLevel,
                                           maxLevel);

        std::vector<size_t> vIndices2;
        vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0), uv(1), radius, minLevel, maxLevel);
        if(vIndices2.empty())
            ++emptyAreaRejectMatches;

        float bestDist = std::numeric_limits<float>::infinity();
        float bestDist2 = std::numeric_limits<float>::infinity();
        int bestIdx2 = -1;
        int bestLevel = -1;
        int bestLevel2 = -1;

        for(size_t i2 : vIndices2)
        {
            if(i2 >= static_cast<size_t>(CurrentFrame.N) || i2 >= static_cast<size_t>(descCurrent.rows))
                continue;

            if(CurrentFrame.mvpMapPoints[i2] && CurrentFrame.mvpMapPoints[i2]->Observations() > 0)
                continue;

            if(CurrentFrame.Nleft == -1 && i2 < CurrentFrame.mvuRight.size() && CurrentFrame.mvuRight[i2] > 0)
            {
                const float ur = uv(0) - CurrentFrame.mbf * invzc;
                const float er = std::fabs(ur - CurrentFrame.mvuRight[i2]);
                if(er > radius)
                    continue;
            }

            const float dist = DescriptorDistanceToSet(mpDescSet, descCurrent.row(static_cast<int>(i2)));
            if(dist < bestDist)
            {
                bestDist2 = bestDist;
                bestDist = dist;
                bestLevel2 = bestLevel;

                if(CurrentFrame.Nleft == -1)
                    bestLevel = CurrentFrame.mvKeysUn[i2].octave;
                else if(static_cast<int>(i2) < CurrentFrame.Nleft)
                    bestLevel = CurrentFrame.mvKeys[i2].octave;
                else
                    bestLevel = CurrentFrame.mvKeysRight[i2 - CurrentFrame.Nleft].octave;

                bestIdx2 = static_cast<int>(i2);
            }
            else if(dist < bestDist2)
            {
                if(CurrentFrame.Nleft == -1)
                    bestLevel2 = CurrentFrame.mvKeysUn[i2].octave;
                else if(static_cast<int>(i2) < CurrentFrame.Nleft)
                    bestLevel2 = CurrentFrame.mvKeys[i2].octave;
                else
                    bestLevel2 = CurrentFrame.mvKeysRight[i2 - CurrentFrame.Nleft].octave;

                bestDist2 = dist;
            }
        }

        if(bestIdx2 >= 0)
        {
            ++rawMatches;
            if(bestDist > thHigh)
            {
                ++distRejectMatches;
            }
            else if(ShouldRejectByRatio(bestDist, bestDist2, mfNNratio, bestLevel, bestLevel2))
            {
                // Ratio rejected.
                ++ratioRejectMatches;
            }
            else
            {
                ++ratioMatches;
                candidates.push_back({bestIdx2, pMP, bestDist});
            }
        }

        if(CurrentFrame.Nleft != -1)
        {
            Eigen::Vector3f x3Dr = CurrentFrame.GetRelativePoseTrl() * x3Dc;
            Eigen::Vector2f uvR = CurrentFrame.mpCamera->project(x3Dr);

            if(uvR(0) < CurrentFrame.mnMinX || uvR(0) > CurrentFrame.mnMaxX)
                continue;
            if(uvR(1) < CurrentFrame.mnMinY || uvR(1) > CurrentFrame.mnMaxY)
                continue;

            std::vector<size_t> vIndicesRight;
            vIndicesRight = CurrentFrame.GetFeaturesInArea(uvR(0), uvR(1), radius, minLevel, maxLevel, true);
            if(vIndicesRight.empty())
                ++emptyAreaRejectMatches;

            float bestDistR = std::numeric_limits<float>::infinity();
            float bestDist2R = std::numeric_limits<float>::infinity();
            int bestIdxR = -1;
            int bestLevelR = -1;
            int bestLevel2R = -1;

            for(size_t i2 : vIndicesRight)
            {
                const int featIdx = static_cast<int>(i2) + CurrentFrame.Nleft;
                if(featIdx < 0 || featIdx >= CurrentFrame.N || featIdx >= descCurrent.rows)
                    continue;

                if(CurrentFrame.mvpMapPoints[featIdx] && CurrentFrame.mvpMapPoints[featIdx]->Observations() > 0)
                    continue;

                const float dist = DescriptorDistanceToSet(mpDescSet, descCurrent.row(featIdx));
                if(dist < bestDistR)
                {
                    bestDist2R = bestDistR;
                    bestDistR = dist;
                    bestLevel2R = bestLevelR;
                    bestLevelR = CurrentFrame.mvKeysRight[i2].octave;
                    bestIdxR = static_cast<int>(i2);
                }
                else if(dist < bestDist2R)
                {
                    bestLevel2R = CurrentFrame.mvKeysRight[i2].octave;
                    bestDist2R = dist;
                }
            }

            if(bestIdxR >= 0)
            {
                ++rawMatches;
                if(bestDistR > thHigh)
                {
                    ++distRejectMatches;
                }
                else if(ShouldRejectByRatio(bestDistR, bestDist2R, mfNNratio, bestLevelR, bestLevel2R))
                {
                    // Ratio rejected.
                    ++ratioRejectMatches;
                }
                else
                {
                    ++ratioMatches;
                    candidates.push_back({bestIdxR + CurrentFrame.Nleft, pMP, bestDistR});
                }
            }
        }
    }

    std::vector<int> bestCandidateIdx;
    const int mutualMatches = SelectBestCandidatePerFeature(candidates, CurrentFrame.N, bestCandidateIdx);
    std::vector<int> bestCandidateIdxSpatial;
    SpatialQuotaStats spatialStats;
    ApplySpatialQuotaToBestCandidates(CurrentFrame, candidates, bestCandidateIdx, bestCandidateIdxSpatial, spatialStats);

    auto assignIfFree = [&CurrentFrame](const int idx, MapPoint* pMP) -> bool
    {
        if(idx < 0 || idx >= CurrentFrame.N || !pMP)
            return false;

        MapPoint* pCurrent = CurrentFrame.mvpMapPoints[idx];
        if(pCurrent == pMP)
            return false;

        if(pCurrent && pCurrent->Observations() > 0)
            return false;

        CurrentFrame.mvpMapPoints[idx] = pMP;
        return true;
    };

    int finalMatches = 0;

    for(int featIdx = 0; featIdx < CurrentFrame.N; ++featIdx)
    {
        const int candIdx = bestCandidateIdxSpatial[featIdx];
        if(candIdx < 0)
            continue;

        const CandidateMatch& c = candidates[candIdx];
        if(!c.pMP)
            continue;

        if(assignIfFree(featIdx, c.pMP))
            ++finalMatches;
        else
            ++assignRejectMatches;

        if(CurrentFrame.Nleft == -1)
            continue;

        if(featIdx < CurrentFrame.Nleft)
        {
            if(featIdx < static_cast<int>(CurrentFrame.mvLeftToRightMatch.size()))
            {
                const int rightIdxLocal = CurrentFrame.mvLeftToRightMatch[featIdx];
                if(rightIdxLocal >= 0)
                {
                    const int rightFeat = rightIdxLocal + CurrentFrame.Nleft;
                    if(assignIfFree(rightFeat, c.pMP))
                        ++finalMatches;
                }
            }
        }
        else
        {
            const int rightIdxLocal = featIdx - CurrentFrame.Nleft;
            if(rightIdxLocal >= 0 && rightIdxLocal < static_cast<int>(CurrentFrame.mvRightToLeftMatch.size()))
            {
                const int leftFeat = CurrentFrame.mvRightToLeftMatch[rightIdxLocal];
                if(leftFeat >= 0 && leftFeat < CurrentFrame.Nleft)
                {
                    if(assignIfFree(leftFeat, c.pMP))
                        ++finalMatches;
                }
            }
        }
    }

    if(IsXFeatMatcherDebugEnabled())
    {
        //调试: `mutual` 这里表示一对一去重后的候选数量，不是双向BF互检。
        std::cout << "[XFeatMatcher::SearchByProjection(LastFrame)] frame=" << CurrentFrame.mnId
                  << " raw=" << rawMatches
                  << " ratio=" << ratioMatches
                  << " mutual=" << mutualMatches
                  << " final=" << finalMatches
                  << " th_high=" << thHigh
                  << " nnratio=" << mfNNratio
                  << " level_gate=" << (UseXFeatProjectionLevelGate() ? 1 : 0)
                  << " dir_bias=" << (UseXFeatLastFrameDirectionalBias() ? 1 : 0)
                  << " lvl_off_neutral=[" << GetXFeatProjLastNeutralMinOffset() << "," << GetXFeatProjLastNeutralMaxOffset() << "]"
                  << " lvl_off_fwd=[" << GetXFeatProjLastForwardMinOffset() << "," << GetXFeatProjLastForwardMaxOffset() << "]"
                  << " lvl_off_bwd=[" << GetXFeatProjLastBackwardMinOffset() << "," << GetXFeatProjLastBackwardMaxOffset() << "]"
                  << " legacy_level_ratio=" << (UseLegacyLevelAwareRatioGate() ? 1 : 0)
                  << " spatial=" << (spatialStats.applied ? 1 : 0)
                  << " spatial_in=" << spatialStats.input
                  << " spatial_keep=" << spatialStats.acceptedFinal
                  << " spatial_drop=" << spatialStats.dropped
                  << " spatial_cap=" << spatialStats.cap
                  << " spatial_grid=" << spatialStats.gridCols << "x" << spatialStats.gridRows
                  << std::endl;

        if(IsXFeatMatcherVerboseDebugEnabled())
        {
            //调试: 详细拒绝原因，便于定位是预测投影窗口不足还是描述子过滤过严。
            std::cout << "[XFeatMatcher::SearchByProjection(LastFrame)][verbose] frame=" << CurrentFrame.mnId
                      << " input_lastN=" << LastFrame.N
                      << " skip_no_mp=" << skippedNoMapPoint
                      << " skip_last_outlier=" << skippedLastOutlier
                      << " skip_no_desc=" << skippedNoDescriptor
                      << " skip_neg_depth=" << skippedNegativeDepth
                      << " skip_outside=" << skippedOutsideImage
                      << " skip_no_scale=" << skippedNoScaleFactors
                      << " projected_mp=" << projectedMapPoints
                      << " reject_empty_area=" << emptyAreaRejectMatches
                      << " reject_dist=" << distRejectMatches
                      << " reject_ratio=" << ratioRejectMatches
                      << " reject_assign=" << assignRejectMatches
                      << " reject_spatial=" << spatialStats.dropped
                      << " candidate_pool=" << candidates.size()
                      << std::endl;
        }
    }

    return finalMatches;
}

int XFeatMatcher::SearchByProjection(Frame &CurrentFrame,
                                     KeyFrame* pKF,
                                     const std::set<MapPoint*> &sAlreadyFound,
                                     const float th,
                                     const float thHighOverride)
{
    if(!pKF)
        return 0;

    // XFeat重定位补匹配: 先按当前位姿把候选MapPoint投影到当前帧，再复用统一投影匹配主逻辑。
    const std::vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();
    if(vpMPs.empty())
        return 0;

    std::vector<MapPoint*> vpProjCandidates;
    vpProjCandidates.reserve(vpMPs.size());

    for(MapPoint* pMP : vpMPs)
    {
        if(!pMP || pMP->isBad())
            continue;
        if(sAlreadyFound.count(pMP))
            continue;

        // 这里会更新 MapPoint 的投影位置/尺度预测字段，供后续 SearchByProjection(Frame, vector<MP>) 使用。
        if(!CurrentFrame.isInFrustum(pMP, 0.5f))
            continue;

        vpProjCandidates.push_back(pMP);
    }

    if(vpProjCandidates.empty())
        return 0;

    return SearchByProjection(CurrentFrame,
                              vpProjCandidates,
                              th,
                              false,
                              50.0f,
                              thHighOverride);
}

void XFeatMatcher::ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1 = 0;
    int max2 = 0;
    int max3 = 0;

    for(int i = 0; i < L; i++)
    {
        const int s = histo[i].size();
        if(s > max1)
        {
            max3 = max2;
            max2 = max1;
            max1 = s;
            ind3 = ind2;
            ind2 = ind1;
            ind1 = i;
        }
        else if(s > max2)
        {
            max3 = max2;
            max2 = s;
            ind3 = ind2;
            ind2 = i;
        }
        else if(s > max3)
        {
            max3 = s;
            ind3 = i;
        }
    }

    if(max2 < 0.1f * static_cast<float>(max1))
    {
        ind2 = -1;
        ind3 = -1;
    }
    else if(max3 < 0.1f * static_cast<float>(max1))
    {
        ind3 = -1;
    }
}

} // namespace ORB_SLAM3
