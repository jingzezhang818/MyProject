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


#include "Tracking.h"

#include "ORBmatcher.h"
#include "XFeatMatcher.h"
#include "XFeatLighterGlueMatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "G2oTypes.h"
#include "Optimizer.h"
#include "Pinhole.h"
#include "KannalaBrandt8.h"
#include "MLPnPsolver.h"
#include "GeometricTools.h"

#include <iostream>

#include <mutex>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include <limits>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <initializer_list>
#include <map>
#include <tuple>


using namespace std;

namespace ORB_SLAM3
{

namespace
{

    bool IsEnvFlagEnabled(const char* key)
    {
        const char* env = std::getenv(key);
        if(!env)
            return false;
        const std::string v(env);
        return !(v.empty() || v == "0" || v == "false" || v == "FALSE");
    }

    //调试: 统一调试开关，设置 `XFEAT_DEBUG=1` 启用。
    bool IsXFeatDebugEnabled()
    {
        static const bool enabled = IsEnvFlagEnabled("XFEAT_DEBUG");
        return enabled;
    }

    int GetXFeatFeatureDiagInterval();
    int GetEnvIntInRange(const char* key, const int fallback, const int minValue, const int maxValue);

    bool ShouldPrintXFeatDebug(const long unsigned int frameId)
    {
        if(!IsXFeatDebugEnabled())
            return false;
        return (frameId % static_cast<long unsigned int>(GetXFeatFeatureDiagInterval())) == 0;
    }

    //诊断: 关键点空间/尺度/深度分布日志总开关（默认关闭，避免常规运行刷屏）。
    bool IsXFeatFeatureDiagEnabled()
    {
        static const bool enabled = IsEnvFlagEnabled("XFEAT_DIAG_FEATURE_DISTRIBUTION");
        return enabled;
    }

    //诊断: 深度/高成本日志开关。默认关闭，避免常规分析时日志过量。
    bool IsXFeatDeepDiagEnabled()
    {
        static const bool enabled = IsEnvFlagEnabled("XFEAT_DIAG_DEEP");
        return enabled;
    }

    bool IsXFeatMeasurementCalibDiagEnabled()
    {
        static const bool enabled = IsEnvFlagEnabled("XFEAT_DIAG_MEASUREMENT_CALIB");
        return enabled;
    }

    bool ShouldPrintXFeatMeasurementCalib(const long unsigned int frameId)
    {
        return IsXFeatMeasurementCalibDiagEnabled() &&
               (frameId % static_cast<long unsigned int>(GetXFeatFeatureDiagInterval())) == 0;
    }

    int GetXFeatFeatureDiagInterval()
    {
        //诊断: 每隔多少帧打印一次分布日志；默认每帧打印。
        static int interval = -1;
        if(interval > 0)
            return interval;

        interval = 1;
        const char* env = std::getenv("XFEAT_DIAG_INTERVAL");
        if(env)
        {
            try
            {
                interval = std::max(1, std::min(500, std::stoi(std::string(env))));
            }
            catch(...)
            {
                interval = 1;
            }
        }
        return interval;
    }

    int GetXFeatFeatureDiagGridCols()
    {
        //诊断: 空间占用统计网格列数。
        static int cols = -1;
        if(cols > 0)
            return cols;

        cols = 8;
        const char* env = std::getenv("XFEAT_DIAG_GRID_COLS");
        if(env)
        {
            try
            {
                cols = std::max(2, std::min(64, std::stoi(std::string(env))));
            }
            catch(...)
            {
                cols = 8;
            }
        }
        return cols;
    }

    int GetXFeatFeatureDiagGridRows()
    {
        //诊断: 空间占用统计网格行数。
        static int rows = -1;
        if(rows > 0)
            return rows;

        rows = 6;
        const char* env = std::getenv("XFEAT_DIAG_GRID_ROWS");
        if(env)
        {
            try
            {
                rows = std::max(2, std::min(64, std::stoi(std::string(env))));
            }
            catch(...)
            {
                rows = 6;
            }
        }
        return rows;
    }

    bool IsRuntimeFpsEnabled()
    {
        static const bool enabled = IsEnvFlagEnabled("XFEAT_FPS") || IsEnvFlagEnabled("SLAM_FPS");
        return enabled;
    }

    float GetEnvFloat(const char* key, const float defaultValue, const float minValue, const float maxValue)
    {
        const char* env = std::getenv(key);
        if(!env)
            return defaultValue;

        try
        {
            const float value = std::stof(std::string(env));
            return std::max(minValue, std::min(maxValue, value));
        }
        catch(...)
        {
            return defaultValue;
        }
    }

    std::string GetEnvLowerString(const char* key, const std::string& defaultValue)
    {
        const char* env = std::getenv(key);
        if(!env)
            return defaultValue;

        std::string value(env);
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    bool IsLocalMapFoundRatioGateEnabled()
    {
        static const bool enabled = IsEnvFlagEnabled("XFEAT_LOCALMAP_FOUND_RATIO_GATE");
        return enabled;
    }

    float GetLocalMapFoundRatioMin()
    {
        static const float minFoundRatio = GetEnvFloat("XFEAT_LOCALMAP_FOUND_RATIO_MIN", 0.25f, 0.0f, 1.0f);
        return minFoundRatio;
    }

    float GetLocalMapFoundRatioMinDepth()
    {
        static const float minDepth = GetEnvFloat("XFEAT_LOCALMAP_FOUND_RATIO_MIN_DEPTH", -1.0f, -1.0f, 10000.0f);
        return minDepth;
    }

    const std::string& GetLocalMapFoundRatioScope()
    {
        static const std::string scope = []()
        {
            const std::string envScope = GetEnvLowerString("XFEAT_LOCALMAP_FOUND_RATIO_SCOPE", "triang_only");
            if(envScope == "all")
                return std::string("all");
            return std::string("triang_only");
        }();
        return scope;
    }

    bool ShouldGateLocalMapPointByFoundRatio(const MapPoint* pMP)
    {
        if(!pMP)
            return false;

        const std::string& scope = GetLocalMapFoundRatioScope();
        if(scope == "all")
            return true;

        return pMP->GetCreationSource() == MapPoint::kCreationSourceLocalMappingTriangulation;
    }

    bool IsXFeatMotionHealthGuardEnabled()
    {
        static const bool enabled = IsEnvFlagEnabled("XFEAT_MOTION_HEALTH_GUARD");
        return enabled;
    }

    bool IsXFeatKFInsertGuardEnabled()
    {
        static const bool enabled = IsEnvFlagEnabled("XFEAT_KF_INSERT_GUARD");
        return enabled;
    }

    int GetXFeatMotionWeakMinInliers()
    {
        static const int value = GetEnvIntInRange("XFEAT_MOTION_WEAK_MIN_INLIERS", 80, 0, 10000);
        return value;
    }

    int GetXFeatMotionBadMinInliers()
    {
        static const int value = GetEnvIntInRange("XFEAT_MOTION_BAD_MIN_INLIERS", 60, 0, 10000);
        return value;
    }

    float GetXFeatMotionWeakMaxOutlier()
    {
        static const float value = GetEnvFloat("XFEAT_MOTION_WEAK_MAX_OUTLIER", 0.30f, 0.0f, 1.0f);
        return value;
    }

    float GetXFeatMotionBadMaxOutlier()
    {
        static const float value = GetEnvFloat("XFEAT_MOTION_BAD_MAX_OUTLIER", 0.40f, 0.0f, 1.0f);
        return value;
    }

    int GetXFeatMotionGuardMinObs()
    {
        static const int value = GetEnvIntInRange("XFEAT_MOTION_GUARD_MIN_OBS", 3, 0, 10000);
        return value;
    }

    float GetXFeatMotionGuardMinFoundRatio()
    {
        static const float value = GetEnvFloat("XFEAT_MOTION_GUARD_MIN_FOUND_RATIO", 0.35f, 0.0f, 1.0f);
        return value;
    }

    bool IsXFeatLocalMapCoreFirstEnabled()
    {
        static const bool enabled = IsEnvFlagEnabled("XFEAT_LOCALMAP_CORE_FIRST");
        return enabled;
    }

    int GetXFeatLocalMapCoreFirstMinPrior()
    {
        static const int value =
            GetEnvIntInRange("XFEAT_LOCALMAP_CORE_FIRST_MIN_PRIOR", 120, 0, 10000);
        return value;
    }

    int GetXFeatLocalMapCoreFirstMinCoreMatches()
    {
        static const int value =
            GetEnvIntInRange("XFEAT_LOCALMAP_CORE_FIRST_MIN_CORE_MATCHES", 40, 0, 10000);
        return value;
    }

    int GetXFeatLocalMapCoreFirstMinTotalMatches()
    {
        static const int value =
            GetEnvIntInRange("XFEAT_LOCALMAP_CORE_FIRST_MIN_TOTAL_MATCHES", 160, 0, 10000);
        return value;
    }

    int GetRuntimeFpsInterval()
    {
        static int interval = -1;
        if(interval > 0)
            return interval;

        interval = 30;
        const char* env = std::getenv("XFEAT_FPS_INTERVAL");
        if(!env)
            env = std::getenv("SLAM_FPS_INTERVAL");
        if(env)
        {
            try
            {
                interval = std::max(1, std::min(1000, std::stoi(std::string(env))));
            }
            catch(...)
            {
                interval = 30;
            }
        }
        return interval;
    }

    void LogRuntimeFps(const char* mode,
                       const std::chrono::steady_clock::time_point& start,
                       const long unsigned int frameId,
                       const double timestamp,
                       const int nFeatures,
                       const int state)
    {
        if(!IsRuntimeFpsEnabled())
            return;

        const auto end = std::chrono::steady_clock::now();
        const double totalMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
        const double procFps = totalMs > 0.0 ? 1000.0 / totalMs : 0.0;

        static bool hasLastEnd = false;
        static std::chrono::steady_clock::time_point lastEnd;
        static double sumMs = 0.0;
        static long unsigned int sampleCount = 0;

        double wallFps = 0.0;
        if(hasLastEnd)
        {
            const double wallMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - lastEnd).count();
            wallFps = wallMs > 0.0 ? 1000.0 / wallMs : 0.0;
        }

        lastEnd = end;
        hasLastEnd = true;
        sumMs += totalMs;
        ++sampleCount;

        const int interval = GetRuntimeFpsInterval();
        if(sampleCount % static_cast<long unsigned int>(interval) != 0)
            return;

        const double avgMs = sumMs / static_cast<double>(sampleCount);
        const double avgProcFps = avgMs > 0.0 ? 1000.0 / avgMs : 0.0;
        const char* frontend = std::getenv("USE_ORB") ? "ORB" : "XFeat";

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(2)
                  << "[RuntimeFPS] sample=" << sampleCount
                  << " frame=" << frameId
                  << " mode=" << mode
                  << " frontend=" << frontend
                  << " state=" << state
                  << " features=" << nFeatures
                  << " total_ms=" << totalMs
                  << " proc_fps=" << procFps
                  << " avg_proc_fps=" << avgProcFps
                  << " wall_fps=" << wallFps
                  << " ts=" << std::setprecision(6) << timestamp
                  << std::endl;
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    bool ShouldRunFeatureDiagForFrame(const unsigned long frameId)
    {
        if(!IsXFeatFeatureDiagEnabled())
            return false;
        const int interval = GetXFeatFeatureDiagInterval();
        return (interval <= 1) || ((frameId % static_cast<unsigned long>(interval)) == 0UL);
    }

    int ClampInt(const int v, const int lo, const int hi)
    {
        return std::max(lo, std::min(hi, v));
    }

    struct FeatureSpatialStats
    {
        int validPoints = 0;
        int totalCells = 0;
        int nonEmptyCells = 0;
        int maxCellCount = 0;
        float top1Share = 0.0f;
        float top4Share = 0.0f;
        float coeffVar = 0.0f;
    };

    struct MatchedDepthStats
    {
        int validDepthCount = 0;
        float p10 = 0.0f;
        float p50 = 0.0f;
        float p90 = 0.0f;
        int nearCount = 0;
        int midCount = 0;
        int farCount = 0;
    };

    struct OctaveSummaryStats
    {
        int total = 0;
        float mean = 0.0f;
        float level0Share = 0.0f;
        float level01Share = 0.0f;
        float upperHalfShare = 0.0f;
        float dominantShare = 0.0f;
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

    std::vector<int> BuildAllFeatureIndices(const Frame& F)
    {
        std::vector<int> indices;
        indices.reserve(std::max(0, F.N));
        for(int i = 0; i < F.N; ++i)
            indices.push_back(i);
        return indices;
    }

    std::vector<int> BuildMatchedInlierFeatureIndices(const Frame& F)
    {
        std::vector<int> indices;
        const int n = std::min(F.N, static_cast<int>(F.mvpMapPoints.size()));
        indices.reserve(n);
        for(int i = 0; i < n; ++i)
        {
            if(!F.mvpMapPoints[i])
                continue;
            if(i < static_cast<int>(F.mvbOutlier.size()) && F.mvbOutlier[i])
                continue;
            indices.push_back(i);
        }
        return indices;
    }

    FeatureSpatialStats ComputeFeatureSpatialStats(const Frame& F, const std::vector<int>& indices)
    {
        FeatureSpatialStats stats;
        const int gridCols = GetXFeatFeatureDiagGridCols();
        const int gridRows = GetXFeatFeatureDiagGridRows();
        stats.totalCells = std::max(1, gridCols * gridRows);
        if(indices.empty())
            return stats;

        const float minX = F.mnMinX;
        const float maxX = F.mnMaxX;
        const float minY = F.mnMinY;
        const float maxY = F.mnMaxY;
        const float spanX = std::max(1.0f, maxX - minX);
        const float spanY = std::max(1.0f, maxY - minY);

        std::vector<int> cellCount(stats.totalCells, 0);
        int validPoints = 0;

        for(const int idx : indices)
        {
            cv::KeyPoint kp;
            if(!GetFrameKeyPointByIndex(F, idx, kp))
                continue;

            const float fx = (kp.pt.x - minX) / spanX;
            const float fy = (kp.pt.y - minY) / spanY;
            const int cellX = ClampInt(static_cast<int>(fx * static_cast<float>(gridCols)), 0, gridCols - 1);
            const int cellY = ClampInt(static_cast<int>(fy * static_cast<float>(gridRows)), 0, gridRows - 1);
            const int cellIdx = cellY * gridCols + cellX;
            cellCount[cellIdx]++;
            validPoints++;
        }

        stats.validPoints = validPoints;
        if(validPoints <= 0)
            return stats;

        int sumTop4 = 0;
        std::sort(cellCount.begin(), cellCount.end(), std::greater<int>());
        for(int c : cellCount)
        {
            if(c > 0)
                stats.nonEmptyCells++;
            if(c > stats.maxCellCount)
                stats.maxCellCount = c;
        }
        for(int i = 0; i < std::min(4, static_cast<int>(cellCount.size())); ++i)
            sumTop4 += cellCount[i];

        stats.top1Share = static_cast<float>(cellCount[0]) / static_cast<float>(validPoints);
        stats.top4Share = static_cast<float>(sumTop4) / static_cast<float>(validPoints);

        const float mean = static_cast<float>(validPoints) / static_cast<float>(stats.totalCells);
        float sqSum = 0.0f;
        for(const int c : cellCount)
        {
            const float d = static_cast<float>(c) - mean;
            sqSum += d * d;
        }
        const float variance = sqSum / static_cast<float>(stats.totalCells);
        const float stdDev = std::sqrt(std::max(0.0f, variance));
        stats.coeffVar = stdDev / std::max(1e-6f, mean);
        return stats;
    }

    std::vector<int> BuildOctaveHistogram(const Frame& F, const std::vector<int>& indices)
    {
        const int levels = std::max(1, F.mnScaleLevels);
        std::vector<int> hist(static_cast<size_t>(levels), 0);
        for(const int idx : indices)
        {
            cv::KeyPoint kp;
            if(!GetFrameKeyPointByIndex(F, idx, kp))
                continue;
            const int octave = ClampInt(kp.octave, 0, levels - 1);
            hist[static_cast<size_t>(octave)]++;
        }
        return hist;
    }

    float PercentileFromSortedDepths(const std::vector<float>& sortedDepths, const float q)
    {
        if(sortedDepths.empty())
            return 0.0f;
        const float qq = std::max(0.0f, std::min(1.0f, q));
        const float idxF = qq * static_cast<float>(sortedDepths.size() - 1);
        const size_t idx = static_cast<size_t>(std::llround(idxF));
        return sortedDepths[std::min(idx, sortedDepths.size() - 1)];
    }

    MatchedDepthStats ComputeMatchedDepthStats(const Frame& F, const std::vector<int>& matchedIndices)
    {
        MatchedDepthStats stats;
        if(matchedIndices.empty() || !F.isSet())
            return stats;

        const Sophus::SE3f Tcw = F.GetPose();
        std::vector<float> depths;
        depths.reserve(matchedIndices.size());

        for(const int idx : matchedIndices)
        {
            if(idx < 0 || idx >= static_cast<int>(F.mvpMapPoints.size()))
                continue;
            MapPoint* pMP = F.mvpMapPoints[idx];
            if(!pMP || pMP->isBad())
                continue;

            const Eigen::Vector3f Pw = pMP->GetWorldPos();
            const Eigen::Vector3f Pc = Tcw * Pw;
            const float z = Pc(2);
            if(!std::isfinite(z) || z <= 0.0f)
                continue;

            depths.push_back(z);
        }

        if(depths.empty())
            return stats;

        std::sort(depths.begin(), depths.end());
        stats.validDepthCount = static_cast<int>(depths.size());
        stats.p10 = PercentileFromSortedDepths(depths, 0.10f);
        stats.p50 = PercentileFromSortedDepths(depths, 0.50f);
        stats.p90 = PercentileFromSortedDepths(depths, 0.90f);

        //诊断: 用相对中位深度做近/中/远分桶，更稳定地比较不同序列和时刻。
        const float nearTh = std::max(1e-4f, stats.p50 * 0.7f);
        const float farTh = std::max(nearTh + 1e-4f, stats.p50 * 1.5f);
        for(const float z : depths)
        {
            if(z <= nearTh)
                stats.nearCount++;
            else if(z >= farTh)
                stats.farCount++;
            else
                stats.midCount++;
        }
        return stats;
    }

    std::string FormatHistogram(const std::vector<int>& hist)
    {
        std::ostringstream oss;
        oss << "[";
        for(size_t i = 0; i < hist.size(); ++i)
        {
            if(i > 0)
                oss << ",";
            oss << "L" << i << ":" << hist[i];
        }
        oss << "]";
        return oss.str();
    }

    OctaveSummaryStats ComputeOctaveSummary(const std::vector<int>& hist)
    {
        OctaveSummaryStats stats;
        if(hist.empty())
            return stats;

        int dominant = 0;
        int level01 = 0;
        int upperHalf = 0;
        const int upperStart = std::max(1, static_cast<int>((hist.size() + 1) / 2));

        for(size_t i = 0; i < hist.size(); ++i)
        {
            const int count = hist[i];
            stats.total += count;
            stats.mean += static_cast<float>(i * static_cast<size_t>(count));
            dominant = std::max(dominant, count);
            if(i <= 1)
                level01 += count;
            if(static_cast<int>(i) >= upperStart)
                upperHalf += count;
        }

        if(stats.total <= 0)
        {
            stats.mean = 0.0f;
            return stats;
        }

        const float invTotal = 1.0f / static_cast<float>(stats.total);
        stats.mean *= invTotal;
        stats.level0Share = hist.empty() ? 0.0f : static_cast<float>(hist[0]) * invTotal;
        stats.level01Share = static_cast<float>(level01) * invTotal;
        stats.upperHalfShare = static_cast<float>(upperHalf) * invTotal;
        stats.dominantShare = static_cast<float>(dominant) * invTotal;
        return stats;
    }

    const char* TrackingStateToString(const int state)
    {
        switch(state)
        {
            case Tracking::SYSTEM_NOT_READY: return "SYSTEM_NOT_READY";
            case Tracking::NO_IMAGES_YET: return "NO_IMAGES_YET";
            case Tracking::NOT_INITIALIZED: return "NOT_INITIALIZED";
            case Tracking::OK: return "OK";
            case Tracking::RECENTLY_LOST: return "RECENTLY_LOST";
            case Tracking::LOST: return "LOST";
            default: return "UNKNOWN";
        }
    }

    void LogFeatureDistributionDiagnostics(const Frame& F,
                                           const int state,
                                           const bool bOK,
                                           const bool bUseORB,
                                           const int mnMatchesInliers)
    {
        const std::vector<int> allIdx = BuildAllFeatureIndices(F);
        const std::vector<int> matchedIdx = BuildMatchedInlierFeatureIndices(F);

        const FeatureSpatialStats spatialAll = ComputeFeatureSpatialStats(F, allIdx);
        const FeatureSpatialStats spatialMatched = ComputeFeatureSpatialStats(F, matchedIdx);

        const std::vector<int> histAll = BuildOctaveHistogram(F, allIdx);
        const std::vector<int> histMatched = BuildOctaveHistogram(F, matchedIdx);
        const OctaveSummaryStats octaveAll = ComputeOctaveSummary(histAll);
        const OctaveSummaryStats octaveMatched = ComputeOctaveSummary(histMatched);

        const MatchedDepthStats depthStats = ComputeMatchedDepthStats(F, matchedIdx);

        const int gridCols = GetXFeatFeatureDiagGridCols();
        const int gridRows = GetXFeatFeatureDiagGridRows();
        const float matchedToAllRatio = spatialAll.validPoints > 0
            ? static_cast<float>(spatialMatched.validPoints) / static_cast<float>(spatialAll.validPoints)
            : 0.0f;
        const float matchedCoverageRatio = spatialAll.nonEmptyCells > 0
            ? static_cast<float>(spatialMatched.nonEmptyCells) / static_cast<float>(spatialAll.nonEmptyCells)
            : 0.0f;

        std::cout << std::fixed << std::setprecision(3)
                  << "[XFeatDiag] frame=" << F.mnId
                  << " mode=" << (bUseORB ? "ORB" : "XFeat")
                  << " state=" << TrackingStateToString(state)
                  << " bOK=" << (bOK ? "true" : "false")
                  << " N=" << F.N
                  << " matches_inlier_frame=" << matchedIdx.size()
                  << " mnMatchesInliers=" << mnMatchesInliers
                  << " grid=" << gridCols << "x" << gridRows
                  << std::endl;

        if(IsXFeatDeepDiagEnabled())
        {
            std::cout << std::fixed << std::setprecision(3)
                      << "[XFeatDiag][spatial]"
                      << " frame_id=" << F.mnId
                      << " all{valid=" << spatialAll.validPoints
                      << ",non_empty=" << spatialAll.nonEmptyCells << "/" << spatialAll.totalCells
                      << ",top1=" << spatialAll.top1Share
                      << ",top4=" << spatialAll.top4Share
                      << ",cv=" << spatialAll.coeffVar
                      << "} matched{valid=" << spatialMatched.validPoints
                      << ",non_empty=" << spatialMatched.nonEmptyCells << "/" << spatialMatched.totalCells
                      << ",top1=" << spatialMatched.top1Share
                      << ",top4=" << spatialMatched.top4Share
                      << ",cv=" << spatialMatched.coeffVar
                      << "}"
                      << std::endl;
        }

        std::cout << std::fixed << std::setprecision(6)
                  << "[XFeatDiag][StructureSummary]"
                  << " frame_id=" << F.mnId
                  << " all_valid=" << spatialAll.validPoints
                  << " all_non_empty=" << spatialAll.nonEmptyCells
                  << " all_cells=" << spatialAll.totalCells
                  << " all_top1=" << spatialAll.top1Share
                  << " all_top4=" << spatialAll.top4Share
                  << " all_cv=" << spatialAll.coeffVar
                  << " matched_valid=" << spatialMatched.validPoints
                  << " matched_non_empty=" << spatialMatched.nonEmptyCells
                  << " matched_top1=" << spatialMatched.top1Share
                  << " matched_top4=" << spatialMatched.top4Share
                  << " matched_cv=" << spatialMatched.coeffVar
                  << " matched_to_all_ratio=" << matchedToAllRatio
                  << " matched_coverage_ratio=" << matchedCoverageRatio
                  << " matched_top4_delta=" << (spatialMatched.top4Share - spatialAll.top4Share)
                  << " matched_cv_delta=" << (spatialMatched.coeffVar - spatialAll.coeffVar)
                  << " all_octave_mean=" << octaveAll.mean
                  << " matched_octave_mean=" << octaveMatched.mean
                  << " octave_mean_delta=" << (octaveMatched.mean - octaveAll.mean)
                  << " all_octave_l0_share=" << octaveAll.level0Share
                  << " matched_octave_l0_share=" << octaveMatched.level0Share
                  << " all_octave_l01_share=" << octaveAll.level01Share
                  << " matched_octave_l01_share=" << octaveMatched.level01Share
                  << " all_octave_upper_share=" << octaveAll.upperHalfShare
                  << " matched_octave_upper_share=" << octaveMatched.upperHalfShare
                  << " matched_octave_dominant_share=" << octaveMatched.dominantShare
                  << std::endl;

        if(IsXFeatDeepDiagEnabled())
        {
            std::cout << "[XFeatDiag][octave]"
                      << " frame_id=" << F.mnId
                      << " all=" << FormatHistogram(histAll)
                      << " matched=" << FormatHistogram(histMatched)
                      << std::endl;
        }

        if(depthStats.validDepthCount > 0)
        {
            std::cout << std::fixed << std::setprecision(3)
                      << "[XFeatDiag][depth]"
                      << " frame_id=" << F.mnId
                      << " valid=" << depthStats.validDepthCount
                      << " p10=" << depthStats.p10
                      << " p50=" << depthStats.p50
                      << " p90=" << depthStats.p90
                      << " bucket(near/mid/far)="
                      << depthStats.nearCount << "/"
                      << depthStats.midCount << "/"
                      << depthStats.farCount
                      << std::endl;
        }
        else
        {
            std::cout << "[XFeatDiag][depth]"
                      << " frame_id=" << F.mnId
                      << " valid=0" << std::endl;
        }
    }

    float GetEnvFloatInRange(const char* key, const float fallback, const float minValue, const float maxValue)
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

    int GetEnvIntInRange(const char* key, const int fallback, const int minValue, const int maxValue)
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
        }
        return fallback;
    }

    struct DepthDistributionStats
    {
        int total = 0;
        int valid = 0;
        int invalid = 0;
        int gt50 = 0;
        int gt100 = 0;
        int gt1000 = 0;
        double sum = 0.0;
        std::vector<float> samples;
    };

    void AccumulateDepthDistribution(DepthDistributionStats& stats, const float depth)
    {
        stats.total++;
        if(!std::isfinite(depth) || depth <= 0.0f)
        {
            stats.invalid++;
            return;
        }

        stats.valid++;
        stats.sum += depth;
        if(depth > 50.0f)
            stats.gt50++;
        if(depth > 100.0f)
            stats.gt100++;
        if(depth > 1000.0f)
            stats.gt1000++;
        stats.samples.push_back(depth);
    }

    float DepthPercentile(std::vector<float>& sortedSamples, const float percentile)
    {
        if(sortedSamples.empty())
            return 0.0f;
        std::sort(sortedSamples.begin(), sortedSamples.end());
        const float clamped = std::max(0.0f, std::min(1.0f, percentile));
        const size_t idx = static_cast<size_t>(std::round(clamped * static_cast<float>(sortedSamples.size() - 1)));
        return sortedSamples[std::min(idx, sortedSamples.size() - 1)];
    }

    void AppendDepthDistributionStats(std::ostream& os,
                                      const char* prefix,
                                      const DepthDistributionStats& stats)
    {
        std::vector<float> sortedSamples = stats.samples;
        const float minDepth = sortedSamples.empty() ? 0.0f : DepthPercentile(sortedSamples, 0.0f);
        const float p50 = sortedSamples.empty() ? 0.0f : DepthPercentile(sortedSamples, 0.50f);
        const float p90 = sortedSamples.empty() ? 0.0f : DepthPercentile(sortedSamples, 0.90f);
        const float p99 = sortedSamples.empty() ? 0.0f : DepthPercentile(sortedSamples, 0.99f);
        const float maxDepth = sortedSamples.empty() ? 0.0f : DepthPercentile(sortedSamples, 1.0f);
        const double avg = stats.valid > 0 ? stats.sum / static_cast<double>(stats.valid) : 0.0;

        os << " " << prefix << "_total=" << stats.total
           << " " << prefix << "_valid=" << stats.valid
           << " " << prefix << "_invalid=" << stats.invalid
           << " " << prefix << "_avg=" << avg
           << " " << prefix << "_min=" << minDepth
           << " " << prefix << "_p50=" << p50
           << " " << prefix << "_p90=" << p90
           << " " << prefix << "_p99=" << p99
           << " " << prefix << "_max=" << maxDepth
           << " " << prefix << "_gt50=" << stats.gt50
           << " " << prefix << "_gt100=" << stats.gt100
           << " " << prefix << "_gt1000=" << stats.gt1000;
    }

    struct StereoDepthConstraintStats
    {
        int assoc = 0;
        int stereoObs = 0;
        int stereoDepth = 0;
        int stereoClose = 0;
        int stereoInlier = 0;
        int stereoOutlier = 0;
        int stereoCloseInlier = 0;
        DepthDistributionStats allDepth;
        DepthDistributionStats inlierDepth;
        DepthDistributionStats outlierDepth;
    };

    void AccumulateStereoDepthConstraintStats(const Frame& F,
                                              const float closeDepthThreshold,
                                              const bool useOutlierState,
                                              StereoDepthConstraintStats& stats)
    {
        const int n = std::min(F.N, static_cast<int>(F.mvpMapPoints.size()));
        for(int i = 0; i < n; ++i)
        {
            if(!F.mvpMapPoints[i])
                continue;

            stats.assoc++;
            const bool hasStereoObs = i < static_cast<int>(F.mvuRight.size()) && F.mvuRight[i] >= 0.0f;
            if(!hasStereoObs)
                continue;

            stats.stereoObs++;
            const bool hasDepth = i < static_cast<int>(F.mvDepth.size()) &&
                                  std::isfinite(F.mvDepth[i]) &&
                                  F.mvDepth[i] > 0.0f;
            if(hasDepth)
            {
                stats.stereoDepth++;
                AccumulateDepthDistribution(stats.allDepth, F.mvDepth[i]);
                if(F.mvDepth[i] < closeDepthThreshold)
                    stats.stereoClose++;
            }

            if(!useOutlierState)
                continue;

            if(F.mvbOutlier[i])
            {
                stats.stereoOutlier++;
                if(hasDepth)
                    AccumulateDepthDistribution(stats.outlierDepth, F.mvDepth[i]);
            }
            else
            {
                stats.stereoInlier++;
                if(hasDepth)
                {
                    AccumulateDepthDistribution(stats.inlierDepth, F.mvDepth[i]);
                    if(F.mvDepth[i] < closeDepthThreshold)
                        stats.stereoCloseInlier++;
                }
            }
        }
    }

    void AppendStereoDepthConstraintStats(std::ostream& os,
                                          const char* prefix,
                                          const StereoDepthConstraintStats& stats)
    {
        const float inlierRatio = stats.stereoObs > 0
            ? static_cast<float>(stats.stereoInlier) / static_cast<float>(stats.stereoObs)
            : 0.0f;
        const float outlierRatio = stats.stereoObs > 0
            ? static_cast<float>(stats.stereoOutlier) / static_cast<float>(stats.stereoObs)
            : 0.0f;

        os << " " << prefix << "_assoc=" << stats.assoc
           << " " << prefix << "_stereo_obs=" << stats.stereoObs
           << " " << prefix << "_stereo_depth=" << stats.stereoDepth
           << " " << prefix << "_stereo_close=" << stats.stereoClose
           << " " << prefix << "_stereo_inlier=" << stats.stereoInlier
           << " " << prefix << "_stereo_outlier=" << stats.stereoOutlier
           << " " << prefix << "_stereo_inlier_ratio=" << inlierRatio
           << " " << prefix << "_stereo_outlier_ratio=" << outlierRatio
           << " " << prefix << "_stereo_close_inlier=" << stats.stereoCloseInlier;
        AppendDepthDistributionStats(os, (std::string(prefix) + "_depth").c_str(), stats.allDepth);
        AppendDepthDistributionStats(os, (std::string(prefix) + "_inlier_depth").c_str(), stats.inlierDepth);
        AppendDepthDistributionStats(os, (std::string(prefix) + "_outlier_depth").c_str(), stats.outlierDepth);
    }

    struct MapPointQualityStats
    {
        int total = 0;
        int obs0 = 0;
        int obs1 = 0;
        int obs2 = 0;
        int obs3plus = 0;
        int lowObs = 0;
        int foundLt025 = 0;
        int foundLt050 = 0;
        int foundLt075 = 0;
        int depthValid = 0;
        int closeDepth = 0;
        int farDepth = 0;
        int depthGt50 = 0;
        int depthGt100 = 0;
        int depthGt1000 = 0;
        int ageValid = 0;
        int ageLe2 = 0;
        int ageLe5 = 0;
        double sumObs = 0.0;
        double sumFoundRatio = 0.0;
        double sumDepth = 0.0;
        double sumAge = 0.0;
        std::vector<float> depthSamples;
    };

    float ResolveMapPointQualityDepth(const Frame& F,
                                      MapPoint* pMP,
                                      const float depthHint)
    {
        if(pMP && F.isSet())
        {
            const Eigen::Vector3f x3Dc = F.GetPose() * pMP->GetWorldPos();
            const float z = x3Dc(2);
            if(std::isfinite(z) && z > 0.0f)
                return z;
        }

        if(std::isfinite(depthHint) && depthHint > 0.0f)
            return depthHint;

        return std::numeric_limits<float>::quiet_NaN();
    }

    void AccumulateMapPointQuality(MapPointQualityStats& stats,
                                   MapPoint* pMP,
                                   const Frame& F,
                                   const float depthHint,
                                   const float closeDepthThreshold)
    {
        if(!pMP)
            return;

        const long unsigned int frameId = F.mnId;
        stats.total++;

        const int obs = pMP->Observations();
        stats.sumObs += obs;
        if(obs <= 0)
            stats.obs0++;
        else if(obs == 1)
            stats.obs1++;
        else if(obs == 2)
            stats.obs2++;
        else
            stats.obs3plus++;
        if(obs <= 2)
            stats.lowObs++;

        const float foundRatio = pMP->GetFoundRatio();
        if(std::isfinite(foundRatio))
        {
            stats.sumFoundRatio += foundRatio;
            if(foundRatio < 0.25f)
                stats.foundLt025++;
            if(foundRatio < 0.50f)
                stats.foundLt050++;
            if(foundRatio < 0.75f)
                stats.foundLt075++;
        }

        const float depth = ResolveMapPointQualityDepth(F, pMP, depthHint);
        if(std::isfinite(depth) && depth > 0.0f)
        {
            stats.depthValid++;
            stats.sumDepth += depth;
            stats.depthSamples.push_back(depth);
            if(depth < closeDepthThreshold)
                stats.closeDepth++;
            else
                stats.farDepth++;
            if(depth > 50.0f)
                stats.depthGt50++;
            if(depth > 100.0f)
                stats.depthGt100++;
            if(depth > 1000.0f)
                stats.depthGt1000++;
        }

        if(pMP->mnFirstFrame >= 0 && frameId >= static_cast<long unsigned int>(pMP->mnFirstFrame))
        {
            const long unsigned int age = frameId - static_cast<long unsigned int>(pMP->mnFirstFrame);
            stats.ageValid++;
            stats.sumAge += static_cast<double>(age);
            if(age <= 2)
                stats.ageLe2++;
            if(age <= 5)
                stats.ageLe5++;
        }
    }

    void AppendMapPointQualityStats(std::ostream& os,
                                    const char* prefix,
                                    const MapPointQualityStats& stats)
    {
        const double avgObs = stats.total > 0 ? stats.sumObs / stats.total : 0.0;
        const double avgFound = stats.total > 0 ? stats.sumFoundRatio / stats.total : 0.0;
        const double avgDepth = stats.depthValid > 0 ? stats.sumDepth / stats.depthValid : 0.0;
        const double avgAge = stats.ageValid > 0 ? stats.sumAge / stats.ageValid : 0.0;
        std::vector<float> sortedDepths = stats.depthSamples;
        const float depthMin = sortedDepths.empty() ? 0.0f : DepthPercentile(sortedDepths, 0.0f);
        const float depthP50 = sortedDepths.empty() ? 0.0f : DepthPercentile(sortedDepths, 0.50f);
        const float depthP90 = sortedDepths.empty() ? 0.0f : DepthPercentile(sortedDepths, 0.90f);
        const float depthP99 = sortedDepths.empty() ? 0.0f : DepthPercentile(sortedDepths, 0.99f);
        const float depthMax = sortedDepths.empty() ? 0.0f : DepthPercentile(sortedDepths, 1.0f);

        os << " " << prefix << "_total=" << stats.total
           << " " << prefix << "_obs_avg=" << avgObs
           << " " << prefix << "_obs0=" << stats.obs0
           << " " << prefix << "_obs1=" << stats.obs1
           << " " << prefix << "_obs2=" << stats.obs2
           << " " << prefix << "_obs3p=" << stats.obs3plus
           << " " << prefix << "_low_obs=" << stats.lowObs
           << " " << prefix << "_found_avg=" << avgFound
           << " " << prefix << "_found_lt_025=" << stats.foundLt025
           << " " << prefix << "_found_lt_050=" << stats.foundLt050
           << " " << prefix << "_found_lt_075=" << stats.foundLt075
           << " " << prefix << "_depth_valid=" << stats.depthValid
           << " " << prefix << "_depth_avg=" << avgDepth
           << " " << prefix << "_depth_min=" << depthMin
           << " " << prefix << "_depth_p50=" << depthP50
           << " " << prefix << "_depth_p90=" << depthP90
           << " " << prefix << "_depth_p99=" << depthP99
           << " " << prefix << "_depth_max=" << depthMax
           << " " << prefix << "_depth_gt50=" << stats.depthGt50
           << " " << prefix << "_depth_gt100=" << stats.depthGt100
           << " " << prefix << "_depth_gt1000=" << stats.depthGt1000
           << " " << prefix << "_close=" << stats.closeDepth
           << " " << prefix << "_far=" << stats.farDepth
           << " " << prefix << "_age_avg=" << avgAge
           << " " << prefix << "_age_le2=" << stats.ageLe2
           << " " << prefix << "_age_le5=" << stats.ageLe5;
    }

    enum XFeatMatchSource : unsigned char
    {
        kXFeatMatchUnknown = 0,
        kXFeatMatchMotionProjection = 1,
        kXFeatMatchMotionLightGlue = 2,
        kXFeatMatchReference = 3,
        kXFeatMatchReferenceLightGlue = 4,
        kXFeatMatchLocalProjection = 5,
        kXFeatMatchLocalProjectionLightGlueVerified = 6,
        kXFeatMatchRelocalization = 7
    };

    const char* XFeatMatchSourceName(const unsigned char source)
    {
        switch(source)
        {
            case kXFeatMatchMotionProjection: return "motion_proj";
            case kXFeatMatchMotionLightGlue: return "motion_lg";
            case kXFeatMatchReference: return "ref";
            case kXFeatMatchReferenceLightGlue: return "ref_lg";
            case kXFeatMatchLocalProjection: return "local_proj";
            case kXFeatMatchLocalProjectionLightGlueVerified: return "local_proj_lg_verified";
            case kXFeatMatchRelocalization: return "relocalization";
            default: return "unknown";
        }
    }

    void EnsureXFeatMatchSourceSize(Frame& F)
    {
        if(static_cast<int>(F.mvXFeatMatchSource.size()) != F.N)
            F.mvXFeatMatchSource.assign(static_cast<size_t>(F.N), kXFeatMatchUnknown);
    }

    void ClearXFeatMatchSources(Frame& F)
    {
        F.mvXFeatMatchSource.assign(static_cast<size_t>(F.N), kXFeatMatchUnknown);
    }

    void MarkAssignedMatches(Frame& F, const unsigned char source)
    {
        EnsureXFeatMatchSourceSize(F);
        const size_t n = std::min(F.mvpMapPoints.size(), F.mvXFeatMatchSource.size());
        for(size_t i = 0; i < n; ++i)
        {
            if(F.mvpMapPoints[i])
                F.mvXFeatMatchSource[i] = source;
        }
    }

    int DepthBinIndex(const float depth)
    {
        if(!std::isfinite(depth) || depth <= 0.0f)
            return 0;
        if(depth < 5.0f)
            return 1;
        if(depth < 10.0f)
            return 2;
        if(depth < 30.0f)
            return 3;
        if(depth < 100.0f)
            return 4;
        return 5;
    }

    const char* DepthBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "0_5";
            case 2: return "5_10";
            case 3: return "10_30";
            case 4: return "30_100";
            case 5: return "100p";
            default: return "invalid";
        }
    }

    int MappingFeatureGridBin(const int gridX, const int gridY)
    {
        if(gridX < 0 || gridY < 0)
            return 0;
        if(gridX < 8 || gridX >= FRAME_GRID_COLS - 8 ||
           gridY < 6 || gridY >= FRAME_GRID_ROWS - 6)
        {
            return 1;
        }
        if(gridX < 16 || gridX >= FRAME_GRID_COLS - 16 ||
           gridY < 12 || gridY >= FRAME_GRID_ROWS - 12)
        {
            return 2;
        }
        return 3;
    }

    const char* MappingFeatureGridBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "edge";
            case 2: return "mid";
            case 3: return "center";
            default: return "missing";
        }
    }

    int MappingFeatureOctaveBin(const int octave)
    {
        if(octave < 0)
            return 0;
        if(octave == 0)
            return 1;
        if(octave == 1)
            return 2;
        return 3;
    }

    const char* MappingFeatureOctaveBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "octave0";
            case 2: return "octave1";
            case 3: return "octave2p";
            default: return "missing";
        }
    }

    int MappingFeatureResponseBin(const float response)
    {
        if(!std::isfinite(response) || response <= 0.0f)
            return 0;
        if(response <= 0.20f)
            return 1;
        if(response <= 0.50f)
            return 2;
        return 3;
    }

    const char* MappingFeatureResponseBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "low";
            case 2: return "mid";
            case 3: return "high";
            default: return "missing";
        }
    }

    int MappingFeatureStereoQualityBin(const unsigned char quality)
    {
        switch(quality)
        {
            case Frame::kStereoBirthValid:
                return 1;
            case Frame::kStereoBirthCloseHighConf:
                return 2;
            case Frame::kStereoBirthCloseLowConf:
                return 3;
            case Frame::kStereoBirthInvalid:
            default:
                return 0;
        }
    }

    const char* MappingFeatureStereoQualityBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "valid";
            case 2: return "close_high_conf";
            case 3: return "close_low_conf";
            default: return "invalid";
        }
    }

    int MappingFeatureCombinedBin(const int gridBin,
                                  const int octaveBin,
                                  const int responseBin,
                                  const int stereoQualityBin)
    {
        if(gridBin == 0 || octaveBin == 0)
            return 0;
        if(stereoQualityBin == 3)
            return 4;
        if(gridBin == 1)
            return 1;
        if(octaveBin == 3)
            return 2;
        if(responseBin == 1)
            return 3;
        return 5;
    }

    const char* MappingFeatureCombinedBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "edge_grid";
            case 2: return "coarse_scale";
            case 3: return "low_response";
            case 4: return "close_low_conf";
            case 5: return "balanced";
            default: return "missing";
        }
    }

    bool FrameFeatureGridPosition(const Frame&,
                                  const cv::KeyPoint& kp,
                                  int& gridX,
                                  int& gridY)
    {
        gridX = -1;
        gridY = -1;
        if(Frame::mnMaxX <= Frame::mnMinX || Frame::mnMaxY <= Frame::mnMinY)
            return false;
        gridX = static_cast<int>(std::round((kp.pt.x - Frame::mnMinX) * Frame::mfGridElementWidthInv));
        gridY = static_cast<int>(std::round((kp.pt.y - Frame::mnMinY) * Frame::mfGridElementHeightInv));
        if(gridX < 0 || gridX >= FRAME_GRID_COLS || gridY < 0 || gridY >= FRAME_GRID_ROWS)
        {
            gridX = -1;
            gridY = -1;
            return false;
        }
        return true;
    }

    void SetMappingFeatureQualityFromFrame(MapPoint* pMP,
                                           const Frame& F,
                                           const long unsigned int keyFrameId,
                                           const int keyPointIdx)
    {
        if(!pMP || keyPointIdx < 0)
            return;

        const cv::KeyPoint* pKeyPoint = nullptr;
        if(F.Nleft != -1 && keyPointIdx >= F.Nleft)
        {
            const int rightIdx = keyPointIdx - F.Nleft;
            if(rightIdx >= 0 && rightIdx < static_cast<int>(F.mvKeysRight.size()))
                pKeyPoint = &F.mvKeysRight[static_cast<size_t>(rightIdx)];
        }
        else if(keyPointIdx < static_cast<int>(F.mvKeysUn.size()))
        {
            pKeyPoint = &F.mvKeysUn[static_cast<size_t>(keyPointIdx)];
        }

        if(!pKeyPoint)
            return;

        int gridX = -1;
        int gridY = -1;
        FrameFeatureGridPosition(F, *pKeyPoint, gridX, gridY);
        const float depth = keyPointIdx < static_cast<int>(F.mvDepth.size())
            ? F.mvDepth[static_cast<size_t>(keyPointIdx)]
            : std::numeric_limits<float>::quiet_NaN();
        const unsigned char stereoBirthQuality =
            keyPointIdx < static_cast<int>(F.mvStereoBirthQuality.size())
                ? F.mvStereoBirthQuality[static_cast<size_t>(keyPointIdx)]
                : static_cast<unsigned char>(Frame::kStereoBirthInvalid);
        pMP->SetMappingFeatureQuality(keyFrameId,
                                      keyPointIdx,
                                      gridX,
                                      gridY,
                                      pKeyPoint->octave,
                                      pKeyPoint->response,
                                      depth,
                                      stereoBirthQuality);
    }

    constexpr size_t kMapPointCreationSourceCount = 5;

    size_t MapPointCreationSourceIndex(const MapPoint::CreationSource source)
    {
        const size_t idx = static_cast<size_t>(source);
        if(idx >= kMapPointCreationSourceCount)
            return static_cast<size_t>(MapPoint::kCreationSourceUnknown);
        return idx;
    }

    const char* MapPointCreationSourceName(const MapPoint::CreationSource source)
    {
        switch(source)
        {
            case MapPoint::kCreationSourceTrackingStereo:
                return "tracking_stereo";
            case MapPoint::kCreationSourceLocalMappingTriangulation:
                return "local_mapping_triangulation";
            case MapPoint::kCreationSourceMonoInit:
                return "mono_init";
            case MapPoint::kCreationSourceStereoInit:
                return "stereo_init";
            case MapPoint::kCreationSourceUnknown:
            default:
                return "unknown";
        }
    }

    const char* TriangulationBackendName(const MapPoint::TriangulationMatchBackend backend)
    {
        switch(backend)
        {
            case MapPoint::kTriangBackendXFeatBF:
                return "xfeat_bf";
            case MapPoint::kTriangBackendLightGlue:
                return "lightglue";
            case MapPoint::kTriangBackendORB:
                return "orb";
            case MapPoint::kTriangBackendUnknown:
            default:
                return "unknown";
        }
    }

    int TriangulationBackendBin(const MapPoint::TriangulationMatchBackend backend)
    {
        const int idx = static_cast<int>(backend);
        return (idx >= 0 && idx <= 3) ? idx : 0;
    }

    int TriangulationParallaxBin(const float cosParallax)
    {
        if(!std::isfinite(cosParallax))
            return 0;
        if(cosParallax >= 0.9992f)
            return 1;
        if(cosParallax >= 0.9985f)
            return 2;
        return 3;
    }

    const char* TriangulationParallaxBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "near_threshold";
            case 2: return "mid";
            case 3: return "strong";
            default: return "invalid";
        }
    }

    int TriangulationReprojBin(const float reprojChi2Max)
    {
        if(!std::isfinite(reprojChi2Max))
            return 0;
        if(reprojChi2Max <= 2.0f)
            return 1;
        if(reprojChi2Max <= 4.0f)
            return 2;
        return 3;
    }

    const char* TriangulationReprojBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "low";
            case 2: return "mid";
            case 3: return "high";
            default: return "invalid";
        }
    }

    int TriangulationScaleMarginBin(const float scaleLogMargin)
    {
        if(!std::isfinite(scaleLogMargin))
            return 0;
        if(scaleLogMargin <= 0.05f)
            return 1;
        if(scaleLogMargin <= 0.12f)
            return 2;
        return 3;
    }

    const char* TriangulationScaleMarginBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "tight";
            case 2: return "mid";
            case 3: return "loose";
            default: return "invalid";
        }
    }

    struct TriangulationMultiViewValidation
    {
        int extraKFs = 0;
        int validExtraKFs = 0;
        float bestChi2Ratio = std::numeric_limits<float>::quiet_NaN();
    };

    int ExtraKeyFrameCountBin(const int count)
    {
        if(count <= 0)
            return 0;
        if(count == 1)
            return 1;
        return 2;
    }

    const char* ExtraKeyFrameCountBinName(const int bin)
    {
        switch(bin)
        {
            case 0: return "none";
            case 1: return "one";
            case 2: return "two_plus";
            default: return "unknown";
        }
    }

    int BestChi2RatioBin(const float ratio)
    {
        if(!std::isfinite(ratio))
            return 0;
        if(ratio <= 0.5f)
            return 1;
        if(ratio <= 1.0f)
            return 2;
        return 3;
    }

    const char* BestChi2RatioBinName(const int bin)
    {
        switch(bin)
        {
            case 0: return "missing";
            case 1: return "low";
            case 2: return "mid";
            case 3: return "high";
            default: return "unknown";
        }
    }

    int MultiViewValidationBin(const TriangulationMultiViewValidation& validation)
    {
        if(validation.extraKFs <= 0)
            return 0;
        if(validation.validExtraKFs <= 0)
            return 1;
        if(validation.validExtraKFs == 1)
            return 2;
        return 3;
    }

    const char* MultiViewValidationBinName(const int bin)
    {
        switch(bin)
        {
            case 0: return "unconfirmed";
            case 1: return "rejected_geometry";
            case 2: return "weak_confirmed";
            case 3: return "strong_confirmed";
            default: return "unknown";
        }
    }

    bool ComputeKeyFrameObservationChi2Ratio(KeyFrame* pKF,
                                             MapPoint* pMP,
                                             const std::tuple<int,int>& observation,
                                             float& chi2Ratio)
    {
        chi2Ratio = std::numeric_limits<float>::quiet_NaN();
        if(!pKF || !pMP || pKF->isBad())
            return false;

        const int idxLeft = std::get<0>(observation);
        const int idxRight = std::get<1>(observation);
        const Eigen::Vector3f worldPos = pMP->GetWorldPos();

        if(idxLeft >= 0 && idxLeft < static_cast<int>(pKF->mvKeysUn.size()) && pKF->mpCamera)
        {
            const Eigen::Vector3f x3Dc = pKF->GetPose() * worldPos;
            const float z = x3Dc(2);
            if(!std::isfinite(z) || z <= 0.0f)
                return false;

            const Eigen::Vector2f uv = pKF->mpCamera->project(x3Dc);
            const cv::KeyPoint& kp = pKF->mvKeysUn[static_cast<size_t>(idxLeft)];
            if(kp.octave < 0 || kp.octave >= static_cast<int>(pKF->mvLevelSigma2.size()))
                return false;

            const float dx = uv(0) - kp.pt.x;
            const float dy = uv(1) - kp.pt.y;
            const float sigma2 = pKF->mvLevelSigma2[static_cast<size_t>(kp.octave)];
            if(sigma2 <= 0.0f)
                return false;

            float chi2 = (dx * dx + dy * dy) / sigma2;
            float threshold = 4.5f;
            if(idxLeft < static_cast<int>(pKF->mvuRight.size()) && pKF->mvuRight[static_cast<size_t>(idxLeft)] >= 0.0f)
            {
                const float uRight = uv(0) - pKF->mbf / z;
                const float errRight = uRight - pKF->mvuRight[static_cast<size_t>(idxLeft)];
                chi2 = (dx * dx + dy * dy + errRight * errRight) / sigma2;
                threshold = 6.0f;
            }

            chi2Ratio = chi2 / threshold;
            return std::isfinite(chi2Ratio);
        }

        if(idxRight >= 0 && pKF->NLeft != -1 && pKF->mpCamera2)
        {
            const int rightIdx = idxRight - pKF->NLeft;
            if(rightIdx < 0 || rightIdx >= static_cast<int>(pKF->mvKeysRight.size()))
                return false;

            const Eigen::Vector3f x3Dc = pKF->GetRightPose() * worldPos;
            const float z = x3Dc(2);
            if(!std::isfinite(z) || z <= 0.0f)
                return false;

            const Eigen::Vector2f uv = pKF->mpCamera2->project(x3Dc);
            const cv::KeyPoint& kp = pKF->mvKeysRight[static_cast<size_t>(rightIdx)];
            if(kp.octave < 0 || kp.octave >= static_cast<int>(pKF->mvLevelSigma2.size()))
                return false;

            const float dx = uv(0) - kp.pt.x;
            const float dy = uv(1) - kp.pt.y;
            const float sigma2 = pKF->mvLevelSigma2[static_cast<size_t>(kp.octave)];
            if(sigma2 <= 0.0f)
                return false;

            chi2Ratio = ((dx * dx + dy * dy) / sigma2) / 4.5f;
            return std::isfinite(chi2Ratio);
        }

        return false;
    }

    TriangulationMultiViewValidation EvaluateTriangulationMultiViewValidation(MapPoint* pMP)
    {
        TriangulationMultiViewValidation validation;
        if(!pMP || !pMP->HasTriangulationQuality())
            return validation;

        const std::map<KeyFrame*, std::tuple<int,int>> observations = pMP->GetObservations();
        for(const auto& kv : observations)
        {
            KeyFrame* pKF = kv.first;
            if(!pKF || pKF->isBad())
                continue;
            if(pKF->mnId == pMP->mnTriangAnchorKF1Id || pKF->mnId == pMP->mnTriangAnchorKF2Id)
                continue;

            validation.extraKFs++;
            float chi2Ratio = std::numeric_limits<float>::quiet_NaN();
            if(!ComputeKeyFrameObservationChi2Ratio(pKF, pMP, kv.second, chi2Ratio))
                continue;

            if(!std::isfinite(validation.bestChi2Ratio) || chi2Ratio < validation.bestChi2Ratio)
                validation.bestChi2Ratio = chi2Ratio;
            if(chi2Ratio <= 1.0f)
                validation.validExtraKFs++;
        }

        return validation;
    }

    bool IsLocalProjHistoryTarget(MapPoint* pMP, const float resolvedDepth)
    {
        return pMP &&
               pMP->GetCreationSource() == MapPoint::kCreationSourceLocalMappingTriangulation &&
               std::isfinite(resolvedDepth) &&
               resolvedDepth > 0.0f &&
               resolvedDepth <= 10.0f;
    }

    constexpr int kCloseReuseSourceBinCount = 2;
    constexpr int kCloseReuseDepthBinCount = 2;

    bool IsCloseReuseTarget(MapPoint* pMP, const float resolvedDepth)
    {
        if(!pMP || !std::isfinite(resolvedDepth) ||
           resolvedDepth <= 0.0f || resolvedDepth > 10.0f)
        {
            return false;
        }

        const MapPoint::CreationSource source = pMP->GetCreationSource();
        return source == MapPoint::kCreationSourceTrackingStereo ||
               source == MapPoint::kCreationSourceLocalMappingTriangulation;
    }

    int CloseReuseSourceBin(const MapPoint::CreationSource source)
    {
        if(source == MapPoint::kCreationSourceTrackingStereo)
            return 0;
        if(source == MapPoint::kCreationSourceLocalMappingTriangulation)
            return 1;
        return -1;
    }

    const char* CloseReuseSourceName(const int bin)
    {
        return bin == 0 ? "tracking_stereo" : "local_mapping_triangulation";
    }

    int CloseReuseDepthBin(const float depth)
    {
        if(!std::isfinite(depth) || depth <= 0.0f || depth > 10.0f)
            return -1;
        return depth <= 5.0f ? 0 : 1;
    }

    const char* CloseReuseDepthName(const int bin)
    {
        return bin == 0 ? "0_5" : "5_10";
    }

    constexpr int kCloseReuseFrustumStatusCount = 6;

    int CloseReuseFrustumStatusBin(const MapPoint::TrackFrustumStatus status)
    {
        const int bin = static_cast<int>(status);
        return bin >= 0 && bin < kCloseReuseFrustumStatusCount ? bin : 0;
    }

    const char* CloseReuseFrustumStatusName(const int bin)
    {
        switch(bin)
        {
            case MapPoint::kTrackFrustumAccepted: return "accepted";
            case MapPoint::kTrackFrustumNegativeDepth: return "negative_depth";
            case MapPoint::kTrackFrustumOutsideImage: return "outside_image";
            case MapPoint::kTrackFrustumScaleRange: return "scale_range";
            case MapPoint::kTrackFrustumViewAngle: return "view_angle";
            case MapPoint::kTrackFrustumUnchecked:
            default:
                return "unchecked";
        }
    }

    struct CloseReuseFunnelStats
    {
        int selected = 0;
        int alreadySeen = 0;
        int bad = 0;
        int inFrustum = 0;
        int foundGateReject = 0;
        int historyGateReject = 0;
        int admitted = 0;
        int projectionMatch = 0;
        int frustumStatus[kCloseReuseFrustumStatusCount] = {};
    };

    struct ReferenceAssociationFunnelStats
    {
        int referenceEligible = 0;
        int stableReference = 0;
        int predictedVisibleTarget = 0;
        int preAssociated = 0;
        int motionRecovered = 0;
        int selected = 0;
        int alreadySeen = 0;
        int bad = 0;
        int inFrustum = 0;
        int foundGateReject = 0;
        int historyGateReject = 0;
        int admitted = 0;
        int projectionMatch = 0;
        int frustumStatus[kCloseReuseFrustumStatusCount] = {};
    };

    constexpr int kCloseReuseContributorBinCount = 6;

    struct CloseReuseContributionStats
    {
        int selected = 0;
        int inFrustum = 0;
        int admitted = 0;
        int projectionMatch = 0;
    };

    struct MappingFeatureReuseFunnelStats
    {
        int selected = 0;
        int alreadySeen = 0;
        int bad = 0;
        int inFrustum = 0;
        int foundGateReject = 0;
        int historyGateReject = 0;
        int admitted = 0;
        int projectionMatch = 0;
    };

    float ResolveMapPointCreationDepthForCloseReuse(MapPoint* pMP)
    {
        if(!pMP)
            return std::numeric_limits<float>::quiet_NaN();
        if(pMP->GetCreationSource() == MapPoint::kCreationSourceLocalMappingTriangulation &&
           pMP->HasTriangulationQuality() && std::isfinite(pMP->mTriangDepthAtCreation))
        {
            return pMP->mTriangDepthAtCreation;
        }
        if(pMP->HasMappingFeatureQuality() && std::isfinite(pMP->mMappingFeatureDepth))
            return pMP->mMappingFeatureDepth;
        return std::numeric_limits<float>::quiet_NaN();
    }

    constexpr float kLocalMapStructureMaxDiagDepth = 1000.0f;
    constexpr int kLocalProjPriorityBinCount = 7;

    struct LocalMapVotedKFCandidate
    {
        KeyFrame* pKF = nullptr;
        int voteCount = 0;
        int visibilityScore = 0;
    };

    bool IsMapPointLightlyVisibleInFrame(const Frame& F, MapPoint* pMP)
    {
        if(!pMP || pMP->isBad() || !F.mpCamera || !F.HasPose())
            return false;

        const Eigen::Vector3f P = pMP->GetWorldPos();
        const Sophus::SE3f Tcw = F.GetPose();
        const Eigen::Vector3f Pc = Tcw.rotationMatrix() * P + Tcw.translation();
        if(Pc(2) <= 0.0f)
            return false;

        const Eigen::Vector2f uv = F.mpCamera->project(Pc);
        if(!std::isfinite(uv(0)) || !std::isfinite(uv(1)))
            return false;
        if(uv(0) < Frame::mnMinX || uv(0) > Frame::mnMaxX ||
           uv(1) < Frame::mnMinY || uv(1) > Frame::mnMaxY)
            return false;

        const Eigen::Vector3f PO = P - F.GetOw();
        const float dist = PO.norm();
        if(!std::isfinite(dist) || dist <= 0.0f)
            return false;
        if(dist < pMP->GetMinDistanceInvariance() ||
           dist > pMP->GetMaxDistanceInvariance())
            return false;

        const Eigen::Vector3f normal = pMP->GetNormal();
        const float viewCos = PO.dot(normal) / dist;
        return std::isfinite(viewCos) && viewCos >= 0.5f;
    }

    enum class LocalMapPredVisibilityStatus : unsigned char
    {
        Invalid = 0,
        Visible = 1,
        NegativeDepth = 2,
        OutsideImage = 3,
        ScaleRange = 4,
        ViewAngle = 5
    };

    LocalMapPredVisibilityStatus ClassifyMapPointLightVisibilityInFrame(
        const Frame& F,
        MapPoint* pMP)
    {
        if(!pMP || pMP->isBad() || !F.mpCamera || !F.HasPose())
            return LocalMapPredVisibilityStatus::Invalid;

        const Eigen::Vector3f P = pMP->GetWorldPos();
        const Sophus::SE3f Tcw = F.GetPose();
        const Eigen::Vector3f Pc = Tcw.rotationMatrix() * P + Tcw.translation();
        if(Pc(2) <= 0.0f)
            return LocalMapPredVisibilityStatus::NegativeDepth;

        const Eigen::Vector2f uv = F.mpCamera->project(Pc);
        if(!std::isfinite(uv(0)) || !std::isfinite(uv(1)) ||
           uv(0) < Frame::mnMinX || uv(0) > Frame::mnMaxX ||
           uv(1) < Frame::mnMinY || uv(1) > Frame::mnMaxY)
        {
            return LocalMapPredVisibilityStatus::OutsideImage;
        }

        const Eigen::Vector3f PO = P - F.GetOw();
        const float dist = PO.norm();
        if(!std::isfinite(dist) || dist <= 0.0f ||
           dist < pMP->GetMinDistanceInvariance() ||
           dist > pMP->GetMaxDistanceInvariance())
        {
            return LocalMapPredVisibilityStatus::ScaleRange;
        }

        const Eigen::Vector3f normal = pMP->GetNormal();
        const float viewCos = PO.dot(normal) / dist;
        if(!std::isfinite(viewCos) || viewCos < 0.5f)
            return LocalMapPredVisibilityStatus::ViewAngle;

        return LocalMapPredVisibilityStatus::Visible;
    }

    int EstimateLocalKeyFrameCurrentVisibility(const Frame& F, KeyFrame* pKF)
    {
        if(!pKF || pKF->isBad())
            return 0;

        const std::vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();
        if(vpMPs.empty())
            return 0;

        static constexpr int kMaxVisibilityChecks = 120;
        const int step = std::max(1, static_cast<int>(vpMPs.size()) / kMaxVisibilityChecks);
        int checked = 0;
        int visible = 0;
        for(size_t i = 0; i < vpMPs.size() && checked < kMaxVisibilityChecks; i += static_cast<size_t>(step))
        {
            ++checked;
            if(IsMapPointLightlyVisibleInFrame(F, vpMPs[i]))
                ++visible;
        }

        return visible;
    }

    int CountHistoryBits(unsigned int mask)
    {
        int count = 0;
        while(mask != 0U)
        {
            count += static_cast<int>(mask & 1U);
            mask >>= 1U;
        }
        return count;
    }

    bool IsLongLowRelevantLocalMapHistory(const MapPoint::LocalMapVisibilityHistoryStats& stats)
    {
        if(stats.evaluationCount < 5)
            return false;

        const int irrelevant = stats.negativeDepthCount + stats.outsideImageCount;
        const float irrelevantRate = static_cast<float>(irrelevant) /
                                     static_cast<float>(stats.evaluationCount);
        const int recentCount = std::min(stats.evaluationCount, 8);
        const int recentIrrelevant = CountHistoryBits(stats.recentIrrelevantMask);
        const float recentIrrelevantRate = recentCount > 0
            ? static_cast<float>(recentIrrelevant) / static_cast<float>(recentCount)
            : 0.0f;
        return irrelevantRate >= 0.70f ||
               (recentCount >= 4 && recentIrrelevantRate >= 0.75f);
    }

    bool IsCloseMidLmtAtCreation(MapPoint* pMP)
    {
        if(!pMP ||
           pMP->GetCreationSource() != MapPoint::kCreationSourceLocalMappingTriangulation)
        {
            return false;
        }
        const float depth = ResolveMapPointCreationDepthForCloseReuse(pMP);
        return std::isfinite(depth) && depth > 0.0f && depth <= 10.0f;
    }

    float LocalProjMatchRate(const MapPoint::LocalProjectionHistoryStats& stats)
    {
        return stats.attemptCount > 0
            ? static_cast<float>(stats.matchCount) / static_cast<float>(stats.attemptCount)
            : 0.0f;
    }

    float LocalProjOptOutlierRate(const MapPoint::LocalProjectionHistoryStats& stats)
    {
        const int outcomes = stats.optInlierCount + stats.optOutlierCount;
        return outcomes > 0
            ? static_cast<float>(stats.optOutlierCount) / static_cast<float>(outcomes)
            : 0.0f;
    }

    float LocalProjRecentOutlierRate(const MapPoint::LocalProjectionHistoryStats& stats)
    {
        const int valid = CountHistoryBits(stats.recentValidMask);
        if(valid <= 0)
            return 0.0f;
        const int outliers = CountHistoryBits(stats.recentValidMask & stats.recentOutlierMask);
        return static_cast<float>(outliers) / static_cast<float>(valid);
    }

    int LocalMapProvenanceBin(const Tracking::LocalMPProvenance& provenance);

    int LocalProjPriorityBin(MapPoint* pMP,
                             const Tracking::LocalMPProvenance* provenance,
                             const float resolvedDepth)
    {
        if(!pMP)
            return 0;

        if(provenance)
        {
            if(provenance->bestRole == Tracking::LocalKFProvenanceRole::Reference)
                return 1;

            const int provenanceBin = LocalMapProvenanceBin(*provenance);
            if(provenanceBin == 1)
                return 2;
            if(provenanceBin == 2)
                return 4;
            if(provenanceBin == 5)
                return 5;
        }

        if(IsLocalProjHistoryTarget(pMP, resolvedDepth))
        {
            const MapPoint::LocalProjectionHistoryStats history = pMP->GetLocalProjectionHistory();
            const int optOutcomes = history.optInlierCount + history.optOutlierCount;
            const int recentValid = CountHistoryBits(history.recentValidMask);
            const bool bStableOpt = optOutcomes >= 3 && LocalProjOptOutlierRate(history) <= 0.15f;
            const bool bStableRecent = recentValid >= 3 && LocalProjRecentOutlierRate(history) <= 0.15f;
            if(bStableOpt || bStableRecent)
                return 3;
        }

        return provenance ? 6 : 0;
    }

    const char* LocalProjPriorityBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "reference";
            case 2: return "core_confirmed";
            case 3: return "stable_history";
            case 4: return "weak_vote";
            case 5: return "single_obs";
            case 6: return "other";
            case 0:
            default:
                return "unknown";
        }
    }

    int LocalProjMatchRateBin(const MapPoint::LocalProjectionHistoryStats& stats)
    {
        if(stats.attemptCount < 3)
            return 0;
        const float rate = LocalProjMatchRate(stats);
        if(rate < 0.35f)
            return 1;
        if(rate < 0.65f)
            return 2;
        return 3;
    }

    const char* LocalProjMatchRateBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "low";
            case 2: return "mid";
            case 3: return "high";
            default: return "no_history";
        }
    }

    int LocalProjOptOutlierRateBin(const MapPoint::LocalProjectionHistoryStats& stats)
    {
        const int outcomes = stats.optInlierCount + stats.optOutlierCount;
        if(outcomes < 3)
            return 0;
        const float rate = LocalProjOptOutlierRate(stats);
        if(rate <= 0.15f)
            return 1;
        if(rate <= 0.35f)
            return 2;
        return 3;
    }

    const char* LocalProjOptOutlierRateBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "low";
            case 2: return "mid";
            case 3: return "high";
            default: return "no_history";
        }
    }

    int LocalProjRecentOutlierBin(const MapPoint::LocalProjectionHistoryStats& stats)
    {
        const int valid = CountHistoryBits(stats.recentValidMask);
        if(valid <= 0)
            return 0;
        const float rate = LocalProjRecentOutlierRate(stats);
        if(rate <= 0.15f)
            return 1;
        if(rate <= 0.40f)
            return 2;
        return 3;
    }

    const char* LocalProjRecentOutlierBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "clean";
            case 2: return "mixed";
            case 3: return "dirty";
            default: return "none";
        }
    }

    int LocalProjHistoryBin(const MapPoint::LocalProjectionHistoryStats& stats)
    {
        const int outcomes = stats.optInlierCount + stats.optOutlierCount;
        if(stats.attemptCount < 3 || outcomes < 2)
            return 0;
        if(LocalProjMatchRate(stats) < 0.50f)
            return 1;
        if(LocalProjOptOutlierRate(stats) > 0.35f)
            return 2;
        const int recentValid = CountHistoryBits(stats.recentValidMask);
        if(recentValid > 0 && LocalProjRecentOutlierRate(stats) > 0.40f)
            return 2;
        return 3;
    }

    const char* LocalProjHistoryBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "matcher_unstable";
            case 2: return "opt_unstable";
            case 3: return "stable";
            default: return "insufficient";
        }
    }

    int LocalProjFineHistoryBin(const MapPoint::LocalProjectionHistoryStats& stats)
    {
        const int outcomes = stats.optInlierCount + stats.optOutlierCount;
        const int recentValid = CountHistoryBits(stats.recentValidMask);
        if((outcomes >= 3 && LocalProjOptOutlierRate(stats) > 0.35f) ||
           (recentValid >= 3 && LocalProjRecentOutlierRate(stats) > 0.40f))
        {
            return 2;
        }
        if(stats.attemptCount >= 3 && outcomes >= 2)
            return 1;
        return 0;
    }

    const char* LocalProjFineHistoryBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "stable";
            case 2: return "opt_unstable";
            case 0:
            default:
                return "insufficient";
        }
    }

    struct ReprojectionConstraintStats
    {
        int total = 0;
        int inlier = 0;
        int outlier = 0;
        int errValid = 0;
        int inlierErrValid = 0;
        double errSum = 0.0;
        double inlierErrSum = 0.0;
        double depthSum = 0.0;
        std::vector<float> errSamples;
        std::vector<float> inlierErrSamples;
        std::vector<float> depthSamples;
    };

    void AccumulateReprojectionConstraint(ReprojectionConstraintStats& stats,
                                          const float reprojErr,
                                          const float depth,
                                          const bool isInlier)
    {
        stats.total++;
        if(isInlier)
            stats.inlier++;
        else
            stats.outlier++;

        if(std::isfinite(reprojErr))
        {
            stats.errValid++;
            stats.errSum += reprojErr;
            stats.errSamples.push_back(reprojErr);
            if(isInlier)
            {
                stats.inlierErrValid++;
                stats.inlierErrSum += reprojErr;
                stats.inlierErrSamples.push_back(reprojErr);
            }
        }

        if(std::isfinite(depth) && depth > 0.0f)
        {
            stats.depthSum += depth;
            stats.depthSamples.push_back(depth);
        }
    }

    bool ComputeFrameReprojectionError(const Frame& F,
                                       const int idx,
                                       MapPoint* pMP,
                                       float& reprojErr,
                                       float& depth)
    {
        reprojErr = std::numeric_limits<float>::quiet_NaN();
        depth = std::numeric_limits<float>::quiet_NaN();
        if(!pMP || idx < 0 || idx >= static_cast<int>(F.mvKeysUn.size()) || !F.mpCamera)
            return false;

        const Eigen::Vector3f x3Dc = F.GetPose() * pMP->GetWorldPos();
        depth = x3Dc(2);
        if(!std::isfinite(depth) || depth <= 0.0f)
            return false;

        const Eigen::Vector2f uv = F.mpCamera->project(x3Dc);
        const cv::Point2f& kp = F.mvKeysUn[static_cast<size_t>(idx)].pt;
        const float dx = uv(0) - kp.x;
        const float dy = uv(1) - kp.y;
        reprojErr = std::sqrt(dx * dx + dy * dy);
        return std::isfinite(reprojErr);
    }

    void AppendReprojectionConstraintStats(std::ostream& os,
                                           const ReprojectionConstraintStats& stats)
    {
        std::vector<float> errSamples = stats.errSamples;
        std::vector<float> inlierErrSamples = stats.inlierErrSamples;
        std::vector<float> depthSamples = stats.depthSamples;
        const double outlierRatio = stats.total > 0
            ? static_cast<double>(stats.outlier) / static_cast<double>(stats.total)
            : 0.0;
        const double errAvg = stats.errValid > 0
            ? stats.errSum / static_cast<double>(stats.errValid)
            : 0.0;
        const double inlierErrAvg = stats.inlierErrValid > 0
            ? stats.inlierErrSum / static_cast<double>(stats.inlierErrValid)
            : 0.0;
        const double depthAvg = !depthSamples.empty()
            ? stats.depthSum / static_cast<double>(depthSamples.size())
            : 0.0;

        os << " total=" << stats.total
           << " inlier=" << stats.inlier
           << " outlier=" << stats.outlier
           << " outlier_ratio=" << outlierRatio
           << " err_valid=" << stats.errValid
           << " err_avg=" << errAvg
           << " err_p50=" << (errSamples.empty() ? 0.0f : DepthPercentile(errSamples, 0.50f))
           << " err_p90=" << (errSamples.empty() ? 0.0f : DepthPercentile(errSamples, 0.90f))
           << " inlier_err_avg=" << inlierErrAvg
           << " inlier_err_p50=" << (inlierErrSamples.empty() ? 0.0f : DepthPercentile(inlierErrSamples, 0.50f))
           << " inlier_err_p90=" << (inlierErrSamples.empty() ? 0.0f : DepthPercentile(inlierErrSamples, 0.90f))
           << " depth_avg=" << depthAvg
           << " depth_p50=" << (depthSamples.empty() ? 0.0f : DepthPercentile(depthSamples, 0.50f))
           << " depth_p90=" << (depthSamples.empty() ? 0.0f : DepthPercentile(depthSamples, 0.90f));
    }

    void AppendTriangulationQualityErrorStats(std::ostream& os,
                                              const ReprojectionConstraintStats& stats)
    {
        const double outlierRatio = stats.total > 0
            ? static_cast<double>(stats.outlier) / static_cast<double>(stats.total)
            : 0.0;
        const double inlierErrAvg = stats.inlierErrValid > 0
            ? stats.inlierErrSum / static_cast<double>(stats.inlierErrValid)
            : 0.0;
        const double depthAvg = !stats.depthSamples.empty()
            ? stats.depthSum / static_cast<double>(stats.depthSamples.size())
            : 0.0;

        os << " total=" << stats.total
           << " inlier=" << stats.inlier
           << " outlier=" << stats.outlier
           << " outlier_ratio=" << outlierRatio
           << " inlier_err_avg=" << inlierErrAvg
           << " depth_avg=" << depthAvg;
    }

    void LogTrackLocalMapConstraintSourceDiagnostics(const Frame& F)
    {
        static constexpr int kNumSources = 8;
        ReprojectionConstraintStats bySource[kNumSources];

        const size_t n = std::min(F.mvpMapPoints.size(), F.mvbOutlier.size());
        for(size_t i = 0; i < n; ++i)
        {
            MapPoint* pMP = F.mvpMapPoints[i];
            if(!pMP)
                continue;

            unsigned char source = kXFeatMatchUnknown;
            if(i < F.mvXFeatMatchSource.size())
                source = F.mvXFeatMatchSource[i];
            if(source >= kNumSources)
                source = kXFeatMatchUnknown;

            float reprojErr = std::numeric_limits<float>::quiet_NaN();
            float depth = std::numeric_limits<float>::quiet_NaN();
            ComputeFrameReprojectionError(F, static_cast<int>(i), pMP, reprojErr, depth);
            const bool isInlier = !F.mvbOutlier[i];
            AccumulateReprojectionConstraint(bySource[source], reprojErr, depth, isInlier);
        }

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6);
        for(int source = 0; source < kNumSources; ++source)
        {
            if(bySource[source].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][SourceError]"
                      << " frame_id=" << F.mnId
                      << " source=" << XFeatMatchSourceName(static_cast<unsigned char>(source));
            AppendReprojectionConstraintStats(std::cout, bySource[source]);
            std::cout << std::endl;
        }

        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    void LogTrackLocalMapMapPointSourceDiagnostics(const Frame& F)
    {
        if(!IsXFeatDeepDiagEnabled())
            return;

        ReprojectionConstraintStats bySource[kMapPointCreationSourceCount];

        const size_t n = std::min(F.mvpMapPoints.size(), F.mvbOutlier.size());
        for(size_t i = 0; i < n; ++i)
        {
            MapPoint* pMP = F.mvpMapPoints[i];
            if(!pMP)
                continue;

            const MapPoint::CreationSource source = pMP->GetCreationSource();
            const size_t sourceIdx = MapPointCreationSourceIndex(source);

            float reprojErr = std::numeric_limits<float>::quiet_NaN();
            float depth = std::numeric_limits<float>::quiet_NaN();
            ComputeFrameReprojectionError(F, static_cast<int>(i), pMP, reprojErr, depth);
            const bool isInlier = !F.mvbOutlier[i];
            AccumulateReprojectionConstraint(bySource[sourceIdx], reprojErr, depth, isInlier);
        }

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6);
        for(size_t sourceIdx = 0; sourceIdx < kMapPointCreationSourceCount; ++sourceIdx)
        {
            if(bySource[sourceIdx].total <= 0)
                continue;
            const MapPoint::CreationSource source = static_cast<MapPoint::CreationSource>(sourceIdx);
            std::cout << "[TrackLocalMap][MPSourceError]"
                      << " frame_id=" << F.mnId
                      << " source=" << MapPointCreationSourceName(source);
            AppendReprojectionConstraintStats(std::cout, bySource[sourceIdx]);
            std::cout << std::endl;
        }
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    void LogTrackLocalMapJointSourceDiagnostics(const Frame& F)
    {
        static constexpr int kNumSources = 8;
        ReprojectionConstraintStats joint[kNumSources][kMapPointCreationSourceCount];

        const size_t n = std::min(F.mvpMapPoints.size(), F.mvbOutlier.size());
        for(size_t i = 0; i < n; ++i)
        {
            MapPoint* pMP = F.mvpMapPoints[i];
            if(!pMP)
                continue;

            unsigned char matchSource = kXFeatMatchUnknown;
            if(i < F.mvXFeatMatchSource.size())
                matchSource = F.mvXFeatMatchSource[i];
            if(matchSource >= kNumSources)
                matchSource = kXFeatMatchUnknown;

            const size_t mpSourceIdx = MapPointCreationSourceIndex(pMP->GetCreationSource());
            float reprojErr = std::numeric_limits<float>::quiet_NaN();
            float depth = std::numeric_limits<float>::quiet_NaN();
            ComputeFrameReprojectionError(F, static_cast<int>(i), pMP, reprojErr, depth);
            AccumulateReprojectionConstraint(joint[matchSource][mpSourceIdx],
                                             reprojErr,
                                             depth,
                                             !F.mvbOutlier[i]);
        }

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6);
        for(int matchSource = 0; matchSource < kNumSources; ++matchSource)
        {
            for(size_t mpSourceIdx = 0; mpSourceIdx < kMapPointCreationSourceCount; ++mpSourceIdx)
            {
                if(joint[matchSource][mpSourceIdx].total <= 0)
                    continue;
                std::cout << "[TrackLocalMap][SourceMPJointError]"
                          << " frame_id=" << F.mnId
                          << " match_source=" << XFeatMatchSourceName(static_cast<unsigned char>(matchSource))
                          << " mp_source=" << MapPointCreationSourceName(static_cast<MapPoint::CreationSource>(mpSourceIdx));
                AppendReprojectionConstraintStats(std::cout, joint[matchSource][mpSourceIdx]);
                std::cout << std::endl;
            }
        }
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    void LogTrackLocalMapTriangulationQualityDiagnostics(const Frame& F)
    {
        if(!IsXFeatDeepDiagEnabled())
            return;

        static constexpr int kNumDepthBins = 6;
        static constexpr int kNumQualityBins = 4;
        ReprojectionConstraintStats depthStats[kNumDepthBins];
        ReprojectionConstraintStats parallaxStats[kNumQualityBins];
        ReprojectionConstraintStats reprojStats[kNumQualityBins];
        ReprojectionConstraintStats scaleMarginStats[kNumQualityBins];
        ReprojectionConstraintStats backendStats[kNumQualityBins];

        const size_t n = std::min(F.mvpMapPoints.size(), F.mvbOutlier.size());
        for(size_t i = 0; i < n; ++i)
        {
            MapPoint* pMP = F.mvpMapPoints[i];
            if(!pMP || pMP->GetCreationSource() != MapPoint::kCreationSourceLocalMappingTriangulation)
                continue;

            unsigned char matchSource = kXFeatMatchUnknown;
            if(i < F.mvXFeatMatchSource.size())
                matchSource = F.mvXFeatMatchSource[i];
            if(matchSource != kXFeatMatchLocalProjection)
                continue;

            if(!pMP->HasTriangulationQuality())
                continue;

            float reprojErr = std::numeric_limits<float>::quiet_NaN();
            float depth = std::numeric_limits<float>::quiet_NaN();
            ComputeFrameReprojectionError(F, static_cast<int>(i), pMP, reprojErr, depth);
            const bool isInlier = !F.mvbOutlier[i];

            AccumulateReprojectionConstraint(depthStats[DepthBinIndex(pMP->mTriangDepthAtCreation)],
                                             reprojErr,
                                             depth,
                                             isInlier);
            AccumulateReprojectionConstraint(parallaxStats[TriangulationParallaxBin(pMP->mTriangCosParallax)],
                                             reprojErr,
                                             depth,
                                             isInlier);
            AccumulateReprojectionConstraint(reprojStats[TriangulationReprojBin(pMP->mTriangReprojChi2Max)],
                                             reprojErr,
                                             depth,
                                             isInlier);
            AccumulateReprojectionConstraint(scaleMarginStats[TriangulationScaleMarginBin(pMP->mTriangScaleLogMargin)],
                                             reprojErr,
                                             depth,
                                             isInlier);
            AccumulateReprojectionConstraint(backendStats[TriangulationBackendBin(pMP->mTriangMatchBackend)],
                                             reprojErr,
                                             depth,
                                             isInlier);
        }

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6);
        for(int bin = 1; bin < kNumDepthBins; ++bin)
        {
            if(depthStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][TriangQualityError]"
                      << " frame_id=" << F.mnId
                      << " dimension=depth_bin"
                      << " bin=" << DepthBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, depthStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 1; bin < kNumQualityBins; ++bin)
        {
            if(parallaxStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][TriangQualityError]"
                      << " frame_id=" << F.mnId
                      << " dimension=parallax_bin"
                      << " bin=" << TriangulationParallaxBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, parallaxStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 1; bin < kNumQualityBins; ++bin)
        {
            if(reprojStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][TriangQualityError]"
                      << " frame_id=" << F.mnId
                      << " dimension=reproj_bin"
                      << " bin=" << TriangulationReprojBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, reprojStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 1; bin < kNumQualityBins; ++bin)
        {
            if(scaleMarginStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][TriangQualityError]"
                      << " frame_id=" << F.mnId
                      << " dimension=scale_margin_bin"
                      << " bin=" << TriangulationScaleMarginBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, scaleMarginStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumQualityBins; ++bin)
        {
            if(backendStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][TriangQualityError]"
                      << " frame_id=" << F.mnId
                      << " dimension=backend_bin"
                      << " bin=" << TriangulationBackendName(static_cast<MapPoint::TriangulationMatchBackend>(bin));
            AppendTriangulationQualityErrorStats(std::cout, backendStats[bin]);
            std::cout << std::endl;
        }
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    void LogTrackLocalMapMappingFeatureOutcomeDiagnostics(const Frame& F)
    {
        if(!IsXFeatFeatureDiagEnabled())
            return;

        static constexpr int kNumGridBins = 4;
        static constexpr int kNumOctaveBins = 4;
        static constexpr int kNumResponseBins = 4;
        static constexpr int kNumDepthBins = 6;
        static constexpr int kNumStereoQualityBins = 4;
        static constexpr int kNumCombinedBins = 6;
        ReprojectionConstraintStats gridStats[kNumGridBins];
        ReprojectionConstraintStats octaveStats[kNumOctaveBins];
        ReprojectionConstraintStats responseStats[kNumResponseBins];
        ReprojectionConstraintStats depthStats[kNumDepthBins];
        ReprojectionConstraintStats stereoQualityStats[kNumStereoQualityBins];
        ReprojectionConstraintStats combinedStats[kNumCombinedBins];

        const size_t n = std::min(F.mvpMapPoints.size(), F.mvbOutlier.size());
        for(size_t i = 0; i < n; ++i)
        {
            MapPoint* pMP = F.mvpMapPoints[i];
            if(!pMP || pMP->GetCreationSource() != MapPoint::kCreationSourceLocalMappingTriangulation)
                continue;

            unsigned char matchSource = kXFeatMatchUnknown;
            if(i < F.mvXFeatMatchSource.size())
                matchSource = F.mvXFeatMatchSource[i];
            if(matchSource != kXFeatMatchLocalProjection)
                continue;

            float reprojErr = std::numeric_limits<float>::quiet_NaN();
            float depth = std::numeric_limits<float>::quiet_NaN();
            ComputeFrameReprojectionError(F, static_cast<int>(i), pMP, reprojErr, depth);
            if(!IsLocalProjHistoryTarget(pMP, depth))
                continue;

            const bool isInlier = !F.mvbOutlier[i];
            const bool hasFeatureQuality = pMP->HasMappingFeatureQuality();
            const int gridBin = hasFeatureQuality
                ? MappingFeatureGridBin(pMP->mnMappingFeatureGridX, pMP->mnMappingFeatureGridY)
                : 0;
            const int octaveBin = hasFeatureQuality
                ? MappingFeatureOctaveBin(pMP->mnMappingFeatureOctave)
                : 0;
            const int responseBin = hasFeatureQuality
                ? MappingFeatureResponseBin(pMP->mMappingFeatureResponse)
                : 0;
            const int creationDepthBin = hasFeatureQuality
                ? DepthBinIndex(pMP->mMappingFeatureDepth)
                : 0;
            const int stereoQualityBin = hasFeatureQuality
                ? MappingFeatureStereoQualityBin(pMP->mMappingFeatureStereoBirthQuality)
                : 0;
            const int combinedBin = hasFeatureQuality
                ? MappingFeatureCombinedBin(gridBin, octaveBin, responseBin, stereoQualityBin)
                : 0;

            AccumulateReprojectionConstraint(gridStats[gridBin], reprojErr, depth, isInlier);
            AccumulateReprojectionConstraint(octaveStats[octaveBin], reprojErr, depth, isInlier);
            AccumulateReprojectionConstraint(responseStats[responseBin], reprojErr, depth, isInlier);
            AccumulateReprojectionConstraint(depthStats[creationDepthBin], reprojErr, depth, isInlier);
            AccumulateReprojectionConstraint(stereoQualityStats[stereoQualityBin], reprojErr, depth, isInlier);
            AccumulateReprojectionConstraint(combinedStats[combinedBin], reprojErr, depth, isInlier);
        }

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6);
        for(int bin = 0; bin < kNumGridBins; ++bin)
        {
            if(gridStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][MappingFeatureOutcome]"
                      << " frame_id=" << F.mnId
                      << " source=local_mapping_triangulation"
                      << " dimension=grid_bin"
                      << " bin=" << MappingFeatureGridBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, gridStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumOctaveBins; ++bin)
        {
            if(octaveStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][MappingFeatureOutcome]"
                      << " frame_id=" << F.mnId
                      << " source=local_mapping_triangulation"
                      << " dimension=octave_bin"
                      << " bin=" << MappingFeatureOctaveBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, octaveStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumResponseBins; ++bin)
        {
            if(responseStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][MappingFeatureOutcome]"
                      << " frame_id=" << F.mnId
                      << " source=local_mapping_triangulation"
                      << " dimension=response_bin"
                      << " bin=" << MappingFeatureResponseBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, responseStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumDepthBins; ++bin)
        {
            if(depthStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][MappingFeatureOutcome]"
                      << " frame_id=" << F.mnId
                      << " source=local_mapping_triangulation"
                      << " dimension=creation_depth_bin"
                      << " bin=" << DepthBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, depthStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumStereoQualityBins; ++bin)
        {
            if(stereoQualityStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][MappingFeatureOutcome]"
                      << " frame_id=" << F.mnId
                      << " source=local_mapping_triangulation"
                      << " dimension=stereo_quality_bin"
                      << " bin=" << MappingFeatureStereoQualityBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, stereoQualityStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumCombinedBins; ++bin)
        {
            if(combinedStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][MappingFeatureOutcome]"
                      << " frame_id=" << F.mnId
                      << " source=local_mapping_triangulation"
                      << " dimension=mapping_feature_bin"
                      << " bin=" << MappingFeatureCombinedBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, combinedStats[bin]);
            std::cout << std::endl;
        }
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    void LogTrackLocalMapTriangulationMultiViewDiagnostics(const Frame& F)
    {
        if(!IsXFeatFeatureDiagEnabled())
            return;

        static constexpr int kNumCountBins = 3;
        static constexpr int kNumChi2Bins = 4;
        static constexpr int kNumValidationBins = 4;
        ReprojectionConstraintStats extraKfStats[kNumCountBins];
        ReprojectionConstraintStats validExtraKfStats[kNumCountBins];
        ReprojectionConstraintStats bestChi2Stats[kNumChi2Bins];
        ReprojectionConstraintStats validationStats[kNumValidationBins];

        const size_t n = std::min(F.mvpMapPoints.size(), F.mvbOutlier.size());
        for(size_t i = 0; i < n; ++i)
        {
            MapPoint* pMP = F.mvpMapPoints[i];
            if(!pMP || pMP->GetCreationSource() != MapPoint::kCreationSourceLocalMappingTriangulation)
                continue;

            unsigned char matchSource = kXFeatMatchUnknown;
            if(i < F.mvXFeatMatchSource.size())
                matchSource = F.mvXFeatMatchSource[i];
            if(matchSource != kXFeatMatchLocalProjection)
                continue;

            if(!pMP->HasTriangulationQuality() ||
               !std::isfinite(pMP->mTriangDepthAtCreation) ||
               pMP->mTriangDepthAtCreation <= 0.0f ||
               pMP->mTriangDepthAtCreation > 10.0f)
            {
                continue;
            }

            float reprojErr = std::numeric_limits<float>::quiet_NaN();
            float depth = std::numeric_limits<float>::quiet_NaN();
            ComputeFrameReprojectionError(F, static_cast<int>(i), pMP, reprojErr, depth);
            const bool isInlier = !F.mvbOutlier[i];
            const TriangulationMultiViewValidation validation =
                EvaluateTriangulationMultiViewValidation(pMP);

            AccumulateReprojectionConstraint(extraKfStats[ExtraKeyFrameCountBin(validation.extraKFs)],
                                             reprojErr,
                                             depth,
                                             isInlier);
            AccumulateReprojectionConstraint(validExtraKfStats[ExtraKeyFrameCountBin(validation.validExtraKFs)],
                                             reprojErr,
                                             depth,
                                             isInlier);
            AccumulateReprojectionConstraint(bestChi2Stats[BestChi2RatioBin(validation.bestChi2Ratio)],
                                             reprojErr,
                                             depth,
                                             isInlier);
            AccumulateReprojectionConstraint(validationStats[MultiViewValidationBin(validation)],
                                             reprojErr,
                                             depth,
                                             isInlier);
        }

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6);
        for(int bin = 0; bin < kNumCountBins; ++bin)
        {
            if(extraKfStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][TriangMultiViewError]"
                      << " frame_id=" << F.mnId
                      << " dimension=extra_kf_bin"
                      << " bin=" << ExtraKeyFrameCountBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, extraKfStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumCountBins; ++bin)
        {
            if(validExtraKfStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][TriangMultiViewError]"
                      << " frame_id=" << F.mnId
                      << " dimension=valid_extra_kf_bin"
                      << " bin=" << ExtraKeyFrameCountBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, validExtraKfStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumChi2Bins; ++bin)
        {
            if(bestChi2Stats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][TriangMultiViewError]"
                      << " frame_id=" << F.mnId
                      << " dimension=best_chi2_ratio_bin"
                      << " bin=" << BestChi2RatioBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, bestChi2Stats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumValidationBins; ++bin)
        {
            if(validationStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][TriangMultiViewError]"
                      << " frame_id=" << F.mnId
                      << " dimension=validation_bin"
                      << " bin=" << MultiViewValidationBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, validationStats[bin]);
            std::cout << std::endl;
        }
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    void LogTrackLocalMapTriangulationLocalProjHistoryDiagnostics(const Frame& F)
    {
        if(!IsXFeatFeatureDiagEnabled())
            return;

        static constexpr int kNumHistoryBins = 4;
        ReprojectionConstraintStats matchRateStats[kNumHistoryBins];
        ReprojectionConstraintStats optOutlierRateStats[kNumHistoryBins];
        ReprojectionConstraintStats recentOutlierStats[kNumHistoryBins];
        ReprojectionConstraintStats historyStats[kNumHistoryBins];

        const size_t n = std::min(F.mvpMapPoints.size(), F.mvbOutlier.size());
        for(size_t i = 0; i < n; ++i)
        {
            MapPoint* pMP = F.mvpMapPoints[i];
            if(!pMP || pMP->GetCreationSource() != MapPoint::kCreationSourceLocalMappingTriangulation)
                continue;

            unsigned char matchSource = kXFeatMatchUnknown;
            if(i < F.mvXFeatMatchSource.size())
                matchSource = F.mvXFeatMatchSource[i];
            if(matchSource != kXFeatMatchLocalProjection)
                continue;

            float reprojErr = std::numeric_limits<float>::quiet_NaN();
            float depth = std::numeric_limits<float>::quiet_NaN();
            ComputeFrameReprojectionError(F, static_cast<int>(i), pMP, reprojErr, depth);
            if(!IsLocalProjHistoryTarget(pMP, depth))
                continue;

            const bool isInlier = !F.mvbOutlier[i];
            const MapPoint::LocalProjectionHistoryStats localProjHistory =
                pMP->GetLocalProjectionHistory();

            AccumulateReprojectionConstraint(matchRateStats[LocalProjMatchRateBin(localProjHistory)],
                                             reprojErr,
                                             depth,
                                             isInlier);
            AccumulateReprojectionConstraint(optOutlierRateStats[LocalProjOptOutlierRateBin(localProjHistory)],
                                             reprojErr,
                                             depth,
                                             isInlier);
            AccumulateReprojectionConstraint(recentOutlierStats[LocalProjRecentOutlierBin(localProjHistory)],
                                             reprojErr,
                                             depth,
                                             isInlier);
            AccumulateReprojectionConstraint(historyStats[LocalProjHistoryBin(localProjHistory)],
                                             reprojErr,
                                             depth,
                                             isInlier);
        }

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6);
        for(int bin = 0; bin < kNumHistoryBins; ++bin)
        {
            if(matchRateStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][TriangLocalProjHistoryError]"
                      << " frame_id=" << F.mnId
                      << " dimension=match_rate_bin"
                      << " bin=" << LocalProjMatchRateBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, matchRateStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumHistoryBins; ++bin)
        {
            if(optOutlierRateStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][TriangLocalProjHistoryError]"
                      << " frame_id=" << F.mnId
                      << " dimension=opt_outlier_rate_bin"
                      << " bin=" << LocalProjOptOutlierRateBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, optOutlierRateStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumHistoryBins; ++bin)
        {
            if(recentOutlierStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][TriangLocalProjHistoryError]"
                      << " frame_id=" << F.mnId
                      << " dimension=recent_outlier_bin"
                      << " bin=" << LocalProjRecentOutlierBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, recentOutlierStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumHistoryBins; ++bin)
        {
            if(historyStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][TriangLocalProjHistoryError]"
                      << " frame_id=" << F.mnId
                      << " dimension=history_bin"
                      << " bin=" << LocalProjHistoryBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, historyStats[bin]);
            std::cout << std::endl;
        }
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    int LocalProjDescriptorBin(const float bestDist)
    {
        if(!std::isfinite(bestDist))
            return 0;
        if(bestDist <= 0.70f)
            return 1;
        if(bestDist <= 1.00f)
            return 2;
        if(bestDist <= 1.20f)
            return 3;
        return 4;
    }

    const char* LocalProjDescriptorBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "very_good";
            case 2: return "good";
            case 3: return "weak";
            case 4: return "near_threshold";
            case 0:
            default:
                return "missing";
        }
    }

    int LocalProjRatioBin(const float ratio, const float secondDist)
    {
        if(std::isfinite(secondDist) && secondDist > 0.0f)
        {
            if(!std::isfinite(ratio))
                return 0;
            if(ratio <= 0.70f)
                return 2;
            if(ratio <= 0.85f)
                return 3;
            return 4;
        }
        return 1;
    }

    const char* LocalProjRatioBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "no_second";
            case 2: return "strong";
            case 3: return "mid";
            case 4: return "weak";
            case 0:
            default:
                return "missing";
        }
    }

    int LocalProjCandidateCountBin(const int candidateCount)
    {
        if(candidateCount < 0)
            return 0;
        if(candidateCount <= 1)
            return 1;
        if(candidateCount <= 4)
            return 2;
        return 3;
    }

    const char* LocalProjCandidateCountBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "one";
            case 2: return "few";
            case 3: return "many";
            case 0:
            default:
                return "missing";
        }
    }

    int LocalProjNormBin(const float projErrNorm)
    {
        if(!std::isfinite(projErrNorm))
            return 0;
        if(projErrNorm <= 0.25f)
            return 1;
        if(projErrNorm <= 0.50f)
            return 2;
        if(projErrNorm <= 1.00f)
            return 3;
        return 4;
    }

    const char* LocalProjNormBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "small";
            case 2: return "mid";
            case 3: return "edge";
            case 4: return "outside";
            case 0:
            default:
                return "missing";
        }
    }

    int LocalProjNormFineBin(const float projErrNorm)
    {
        if(!std::isfinite(projErrNorm))
            return 0;
        if(projErrNorm <= 0.25f)
            return 1;
        if(projErrNorm <= 0.50f)
            return 2;
        if(projErrNorm <= 0.65f)
            return 3;
        if(projErrNorm <= 0.80f)
            return 4;
        if(projErrNorm <= 1.00f)
            return 5;
        return 6;
    }

    const char* LocalProjNormFineBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "0_25";
            case 2: return "25_50";
            case 3: return "50_65";
            case 4: return "65_80";
            case 5: return "80_100";
            case 6: return "outside";
            case 0:
            default:
                return "missing";
        }
    }

    int LocalProjLevelDiffBin(const int levelDiff)
    {
        if(levelDiff < 0)
            return 0;
        if(levelDiff == 0)
            return 1;
        if(levelDiff == 1)
            return 2;
        return 3;
    }

    const char* LocalProjLevelDiffBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "exact";
            case 2: return "off1";
            case 3: return "off2p";
            case 0:
            default:
                return "missing";
        }
    }

    int LocalProjFineDepthBin(const float depth)
    {
        if(!std::isfinite(depth) || depth <= 0.0f)
            return 0;
        if(depth < 5.0f)
            return 1;
        return 2;
    }

    const char* LocalProjFineDepthBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "0_5";
            case 2: return "5_10";
            case 0:
            default:
                return "invalid";
        }
    }

    int LocalProjMatcherQualityBin(const Frame& F, const size_t i)
    {
        if(i >= F.mvLocalProjBestDist.size())
            return 0;

        const float bestDist = F.mvLocalProjBestDist[i];
        if(!std::isfinite(bestDist))
            return 0;

        const float ratio = i < F.mvLocalProjRatio.size()
            ? F.mvLocalProjRatio[i]
            : std::numeric_limits<float>::quiet_NaN();
        const int candidateCount = i < F.mvLocalProjCandidateCount.size()
            ? F.mvLocalProjCandidateCount[i]
            : -1;
        const float projErrNorm = i < F.mvLocalProjProjErrNorm.size()
            ? F.mvLocalProjProjErrNorm[i]
            : std::numeric_limits<float>::quiet_NaN();
        const int levelDiff = i < F.mvLocalProjLevelDiff.size()
            ? F.mvLocalProjLevelDiff[i]
            : -1;

        if(bestDist > 1.20f)
            return 2;
        if((std::isfinite(ratio) && ratio > 0.85f) || candidateCount >= 5)
            return 3;
        if(std::isfinite(projErrNorm) && projErrNorm > 0.50f)
            return 4;
        if(levelDiff >= 2)
            return 5;
        return 1;
    }

    const char* LocalProjMatcherQualityBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "stable";
            case 2: return "descriptor_weak";
            case 3: return "ambiguous";
            case 4: return "projection_edge";
            case 5: return "scale_mismatch";
            case 0:
            default:
                return "missing";
        }
    }

    void LogTrackLocalMapLocalProjMatcherQualityDiagnostics(const Frame& F)
    {
        if(!IsXFeatFeatureDiagEnabled())
            return;

        static constexpr int kNumDescriptorBins = 5;
        static constexpr int kNumRatioBins = 5;
        static constexpr int kNumCandidateBins = 4;
        static constexpr int kNumProjNormBins = 5;
        static constexpr int kNumLevelDiffBins = 4;
        static constexpr int kNumMatcherQualityBins = 6;

        ReprojectionConstraintStats descriptorStats[kNumDescriptorBins];
        ReprojectionConstraintStats ratioStats[kNumRatioBins];
        ReprojectionConstraintStats candidateStats[kNumCandidateBins];
        ReprojectionConstraintStats projNormStats[kNumProjNormBins];
        ReprojectionConstraintStats levelDiffStats[kNumLevelDiffBins];
        ReprojectionConstraintStats matcherQualityStats[kNumMatcherQualityBins];

        const size_t n = std::min(F.mvpMapPoints.size(), F.mvbOutlier.size());
        for(size_t i = 0; i < n; ++i)
        {
            MapPoint* pMP = F.mvpMapPoints[i];
            if(!pMP || pMP->GetCreationSource() != MapPoint::kCreationSourceLocalMappingTriangulation)
                continue;

            unsigned char matchSource = kXFeatMatchUnknown;
            if(i < F.mvXFeatMatchSource.size())
                matchSource = F.mvXFeatMatchSource[i];
            if(matchSource != kXFeatMatchLocalProjection)
                continue;

            float reprojErr = std::numeric_limits<float>::quiet_NaN();
            float depth = std::numeric_limits<float>::quiet_NaN();
            ComputeFrameReprojectionError(F, static_cast<int>(i), pMP, reprojErr, depth);
            if(!IsLocalProjHistoryTarget(pMP, depth))
                continue;

            const bool isInlier = !F.mvbOutlier[i];
            const float bestDist = i < F.mvLocalProjBestDist.size()
                ? F.mvLocalProjBestDist[i]
                : std::numeric_limits<float>::quiet_NaN();
            const float secondDist = i < F.mvLocalProjSecondDist.size()
                ? F.mvLocalProjSecondDist[i]
                : std::numeric_limits<float>::quiet_NaN();
            const float ratio = i < F.mvLocalProjRatio.size()
                ? F.mvLocalProjRatio[i]
                : std::numeric_limits<float>::quiet_NaN();
            const int candidateCount = i < F.mvLocalProjCandidateCount.size()
                ? F.mvLocalProjCandidateCount[i]
                : -1;
            const float projErrNorm = i < F.mvLocalProjProjErrNorm.size()
                ? F.mvLocalProjProjErrNorm[i]
                : std::numeric_limits<float>::quiet_NaN();
            const int levelDiff = i < F.mvLocalProjLevelDiff.size()
                ? F.mvLocalProjLevelDiff[i]
                : -1;

            AccumulateReprojectionConstraint(descriptorStats[LocalProjDescriptorBin(bestDist)],
                                             reprojErr, depth, isInlier);
            AccumulateReprojectionConstraint(ratioStats[LocalProjRatioBin(ratio, secondDist)],
                                             reprojErr, depth, isInlier);
            AccumulateReprojectionConstraint(candidateStats[LocalProjCandidateCountBin(candidateCount)],
                                             reprojErr, depth, isInlier);
            AccumulateReprojectionConstraint(projNormStats[LocalProjNormBin(projErrNorm)],
                                             reprojErr, depth, isInlier);
            AccumulateReprojectionConstraint(levelDiffStats[LocalProjLevelDiffBin(levelDiff)],
                                             reprojErr, depth, isInlier);
            AccumulateReprojectionConstraint(matcherQualityStats[LocalProjMatcherQualityBin(F, i)],
                                             reprojErr, depth, isInlier);
        }

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6);
        for(int bin = 0; bin < kNumDescriptorBins; ++bin)
        {
            if(descriptorStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][LocalProjMatcherQualityError]"
                      << " frame_id=" << F.mnId
                      << " dimension=descriptor_bin"
                      << " bin=" << LocalProjDescriptorBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, descriptorStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumRatioBins; ++bin)
        {
            if(ratioStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][LocalProjMatcherQualityError]"
                      << " frame_id=" << F.mnId
                      << " dimension=ratio_bin"
                      << " bin=" << LocalProjRatioBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, ratioStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumCandidateBins; ++bin)
        {
            if(candidateStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][LocalProjMatcherQualityError]"
                      << " frame_id=" << F.mnId
                      << " dimension=candidate_count_bin"
                      << " bin=" << LocalProjCandidateCountBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, candidateStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumProjNormBins; ++bin)
        {
            if(projNormStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][LocalProjMatcherQualityError]"
                      << " frame_id=" << F.mnId
                      << " dimension=proj_norm_bin"
                      << " bin=" << LocalProjNormBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, projNormStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumLevelDiffBins; ++bin)
        {
            if(levelDiffStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][LocalProjMatcherQualityError]"
                      << " frame_id=" << F.mnId
                      << " dimension=level_diff_bin"
                      << " bin=" << LocalProjLevelDiffBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, levelDiffStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumMatcherQualityBins; ++bin)
        {
            if(matcherQualityStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][LocalProjMatcherQualityError]"
                      << " frame_id=" << F.mnId
                      << " dimension=matcher_quality_bin"
                      << " bin=" << LocalProjMatcherQualityBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, matcherQualityStats[bin]);
            std::cout << std::endl;
        }
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    void LogTrackLocalMapLocalProjMatcherQualityFineDiagnostics(const Frame& F)
    {
        if(!IsXFeatFeatureDiagEnabled())
            return;

        static constexpr int kNumFineBins = 7;
        static constexpr int kNumLevelDiffBins = 4;
        static constexpr int kNumDepthBins = 3;
        static constexpr int kNumHistoryBins = 3;

        ReprojectionConstraintStats fineStats[kNumFineBins];
        ReprojectionConstraintStats fineLevelStats[kNumFineBins][kNumLevelDiffBins];
        ReprojectionConstraintStats fineDepthStats[kNumFineBins][kNumDepthBins];
        ReprojectionConstraintStats fineHistoryStats[kNumFineBins][kNumHistoryBins];

        const size_t n = std::min(F.mvpMapPoints.size(), F.mvbOutlier.size());
        for(size_t i = 0; i < n; ++i)
        {
            MapPoint* pMP = F.mvpMapPoints[i];
            if(!pMP || pMP->GetCreationSource() != MapPoint::kCreationSourceLocalMappingTriangulation)
                continue;

            unsigned char matchSource = kXFeatMatchUnknown;
            if(i < F.mvXFeatMatchSource.size())
                matchSource = F.mvXFeatMatchSource[i];
            if(matchSource != kXFeatMatchLocalProjection)
                continue;

            float reprojErr = std::numeric_limits<float>::quiet_NaN();
            float depth = std::numeric_limits<float>::quiet_NaN();
            ComputeFrameReprojectionError(F, static_cast<int>(i), pMP, reprojErr, depth);
            if(!IsLocalProjHistoryTarget(pMP, depth))
                continue;

            const bool isInlier = !F.mvbOutlier[i];
            const float projErrNorm = i < F.mvLocalProjProjErrNorm.size()
                ? F.mvLocalProjProjErrNorm[i]
                : std::numeric_limits<float>::quiet_NaN();
            const int levelDiff = i < F.mvLocalProjLevelDiff.size()
                ? F.mvLocalProjLevelDiff[i]
                : -1;

            const int fineBin = LocalProjNormFineBin(projErrNorm);
            const int levelBin = LocalProjLevelDiffBin(levelDiff);
            const int depthBin = LocalProjFineDepthBin(depth);
            const int historyBin = LocalProjFineHistoryBin(pMP->GetLocalProjectionHistory());

            AccumulateReprojectionConstraint(fineStats[fineBin], reprojErr, depth, isInlier);
            AccumulateReprojectionConstraint(fineLevelStats[fineBin][levelBin], reprojErr, depth, isInlier);
            AccumulateReprojectionConstraint(fineDepthStats[fineBin][depthBin], reprojErr, depth, isInlier);
            AccumulateReprojectionConstraint(fineHistoryStats[fineBin][historyBin], reprojErr, depth, isInlier);
        }

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6);
        for(int fineBin = 0; fineBin < kNumFineBins; ++fineBin)
        {
            if(fineStats[fineBin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][LocalProjMatcherQualityFineError]"
                      << " frame_id=" << F.mnId
                      << " dimension=proj_norm_fine_bin"
                      << " bin=" << LocalProjNormFineBinName(fineBin);
            AppendTriangulationQualityErrorStats(std::cout, fineStats[fineBin]);
            std::cout << std::endl;
        }
        for(int fineBin = 0; fineBin < kNumFineBins; ++fineBin)
        {
            for(int levelBin = 0; levelBin < kNumLevelDiffBins; ++levelBin)
            {
                if(fineLevelStats[fineBin][levelBin].total <= 0)
                    continue;
                std::cout << "[TrackLocalMap][LocalProjMatcherQualityFineError]"
                          << " frame_id=" << F.mnId
                          << " dimension=proj_norm_fine_x_level_diff"
                          << " bin=" << LocalProjNormFineBinName(fineBin)
                          << "_" << LocalProjLevelDiffBinName(levelBin);
                AppendTriangulationQualityErrorStats(std::cout, fineLevelStats[fineBin][levelBin]);
                std::cout << std::endl;
            }
        }
        for(int fineBin = 0; fineBin < kNumFineBins; ++fineBin)
        {
            for(int depthBin = 0; depthBin < kNumDepthBins; ++depthBin)
            {
                if(fineDepthStats[fineBin][depthBin].total <= 0)
                    continue;
                std::cout << "[TrackLocalMap][LocalProjMatcherQualityFineError]"
                          << " frame_id=" << F.mnId
                          << " dimension=proj_norm_fine_x_depth"
                          << " bin=" << LocalProjNormFineBinName(fineBin)
                          << "_" << LocalProjFineDepthBinName(depthBin);
                AppendTriangulationQualityErrorStats(std::cout, fineDepthStats[fineBin][depthBin]);
                std::cout << std::endl;
            }
        }
        for(int fineBin = 0; fineBin < kNumFineBins; ++fineBin)
        {
            for(int historyBin = 0; historyBin < kNumHistoryBins; ++historyBin)
            {
                if(fineHistoryStats[fineBin][historyBin].total <= 0)
                    continue;
                std::cout << "[TrackLocalMap][LocalProjMatcherQualityFineError]"
                          << " frame_id=" << F.mnId
                          << " dimension=proj_norm_fine_x_history"
                          << " bin=" << LocalProjNormFineBinName(fineBin)
                          << "_" << LocalProjFineHistoryBinName(historyBin);
                AppendTriangulationQualityErrorStats(std::cout, fineHistoryStats[fineBin][historyBin]);
                std::cout << std::endl;
            }
        }
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    struct XFeatMeasurementCalibStats
    {
        int total = 0;
        int inlier = 0;
        int outlier = 0;
        double residualPxSum = 0.0;
        double chi2Sum = 0.0;
        double depthSum = 0.0;
        std::vector<float> residualPxSamples;
        std::vector<float> chi2Samples;
        std::vector<float> depthSamples;
    };

    int XFeatMeasurementDescriptorBin(const float bestDist)
    {
        if(!std::isfinite(bestDist))
            return 0;
        if(bestDist <= 0.70f)
            return 1;
        if(bestDist <= 1.00f)
            return 2;
        if(bestDist <= 1.20f)
            return 3;
        return 4;
    }

    const char* XFeatMeasurementDescriptorBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "very_good";
            case 2: return "good";
            case 3: return "weak";
            case 4: return "near_threshold";
            case 0:
            default:
                return "missing";
        }
    }

    int XFeatMeasurementRatioBin(const float bestDist, const float ratio, const float secondDist)
    {
        if(!std::isfinite(bestDist))
            return 0;
        if(!std::isfinite(ratio) || ratio < 0.0f)
        {
            if(!std::isfinite(secondDist) || secondDist <= 0.0f)
                return 1;
            return 0;
        }
        if(ratio <= 0.70f)
            return 2;
        if(ratio <= 0.85f)
            return 3;
        return 4;
    }

    const char* XFeatMeasurementRatioBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "no_second";
            case 2: return "strong";
            case 3: return "mid";
            case 4: return "weak";
            case 0:
            default:
                return "missing";
        }
    }

    int XFeatMeasurementProjNormBin(const float projNorm)
    {
        if(!std::isfinite(projNorm))
            return 0;
        if(projNorm <= 0.25f)
            return 1;
        if(projNorm <= 0.50f)
            return 2;
        if(projNorm <= 1.00f)
            return 3;
        return 4;
    }

    const char* XFeatMeasurementProjNormBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "small";
            case 2: return "mid";
            case 3: return "edge";
            case 4: return "outside";
            case 0:
            default:
                return "missing";
        }
    }

    int XFeatMeasurementLevelDiffBin(const int levelDiff)
    {
        if(levelDiff < 0)
            return 0;
        if(levelDiff == 0)
            return 1;
        if(levelDiff == 1)
            return 2;
        return 3;
    }

    const char* XFeatMeasurementLevelDiffBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "exact";
            case 2: return "off1";
            case 3: return "off2p";
            case 0:
            default:
                return "missing";
        }
    }

    int XFeatMeasurementObsTypeBin(const Frame& F, const size_t idx)
    {
        if(idx >= F.mvpMapPoints.size() || !F.mvpMapPoints[idx])
            return 0;
        if(idx < F.mvuRight.size() && std::isfinite(F.mvuRight[idx]) && F.mvuRight[idx] >= 0.0f)
            return 2;
        return 1;
    }

    const char* XFeatMeasurementObsTypeBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "mono";
            case 2: return "stereo";
            case 0:
            default:
                return "missing";
        }
    }

    float XFeatMeasurementApproxChi2(const Frame& F, const size_t idx, const float residualPx)
    {
        if(!std::isfinite(residualPx) || idx >= F.mvKeysUn.size())
            return std::numeric_limits<float>::quiet_NaN();

        const int octave = F.mvKeysUn[idx].octave;
        if(octave < 0 || octave >= static_cast<int>(F.mvLevelSigma2.size()))
            return std::numeric_limits<float>::quiet_NaN();

        const float sigma2 = F.mvLevelSigma2[static_cast<size_t>(octave)];
        if(!std::isfinite(sigma2) || sigma2 <= 1e-9f)
            return std::numeric_limits<float>::quiet_NaN();

        return residualPx * residualPx / sigma2;
    }

    void AccumulateXFeatMeasurementCalibStats(XFeatMeasurementCalibStats& stats,
                                              const bool isInlier,
                                              const float residualPx,
                                              const float chi2,
                                              const float depth)
    {
        stats.total++;
        if(isInlier)
            stats.inlier++;
        else
            stats.outlier++;

        if(std::isfinite(residualPx))
        {
            stats.residualPxSum += residualPx;
            stats.residualPxSamples.push_back(residualPx);
        }
        if(std::isfinite(chi2))
        {
            stats.chi2Sum += chi2;
            stats.chi2Samples.push_back(chi2);
        }
        if(std::isfinite(depth) && depth > 0.0f)
        {
            stats.depthSum += depth;
            stats.depthSamples.push_back(depth);
        }
    }

    void PrintXFeatMeasurementCalibStats(const Frame& F,
                                         const char* dimension,
                                         const char* bin,
                                         const XFeatMeasurementCalibStats& stats)
    {
        if(stats.total <= 0)
            return;

        std::vector<float> residualSamples = stats.residualPxSamples;
        std::vector<float> chi2Samples = stats.chi2Samples;
        const double outlierRatio = static_cast<double>(stats.outlier) /
                                    static_cast<double>(std::max(1, stats.total));
        const double residualAvg = !stats.residualPxSamples.empty()
            ? stats.residualPxSum / static_cast<double>(stats.residualPxSamples.size())
            : 0.0;
        const double chi2Avg = !stats.chi2Samples.empty()
            ? stats.chi2Sum / static_cast<double>(stats.chi2Samples.size())
            : 0.0;
        const double depthAvg = !stats.depthSamples.empty()
            ? stats.depthSum / static_cast<double>(stats.depthSamples.size())
            : 0.0;

        std::cout << "[TrackLocalMap][XFeatMeasurementCalib]"
                  << " frame_id=" << F.mnId
                  << " dimension=" << dimension
                  << " bin=" << bin
                  << " total=" << stats.total
                  << " inlier=" << stats.inlier
                  << " outlier=" << stats.outlier
                  << " outlier_ratio=" << outlierRatio
                  << " residual_px_avg=" << residualAvg
                  << " residual_px_p50=" << (residualSamples.empty() ? 0.0f : DepthPercentile(residualSamples, 0.50f))
                  << " residual_px_p90=" << (residualSamples.empty() ? 0.0f : DepthPercentile(residualSamples, 0.90f))
                  << " chi2_avg=" << chi2Avg
                  << " chi2_p90=" << (chi2Samples.empty() ? 0.0f : DepthPercentile(chi2Samples, 0.90f))
                  << " depth_avg=" << depthAvg
                  << std::endl;
    }

    void LogTrackLocalMapXFeatMeasurementCalibDiagnostics(const Frame& F)
    {
        if(!IsXFeatMeasurementCalibDiagEnabled())
            return;
        if(F.mDescriptors.empty() || F.mDescriptors.type() != CV_32F)
            return;

        static constexpr int kNumMatchSourceBins = 8;
        static constexpr int kNumMpSourceBins = static_cast<int>(kMapPointCreationSourceCount);
        static constexpr int kNumDepthBins = 6;
        static constexpr int kNumDescriptorBins = 5;
        static constexpr int kNumRatioBins = 5;
        static constexpr int kNumProjNormBins = 5;
        static constexpr int kNumLevelDiffBins = 4;
        static constexpr int kNumObsTypeBins = 3;

        XFeatMeasurementCalibStats matchSourceStats[kNumMatchSourceBins];
        XFeatMeasurementCalibStats mpSourceStats[kNumMpSourceBins];
        XFeatMeasurementCalibStats depthStats[kNumDepthBins];
        XFeatMeasurementCalibStats descriptorStats[kNumDescriptorBins];
        XFeatMeasurementCalibStats ratioStats[kNumRatioBins];
        XFeatMeasurementCalibStats projNormStats[kNumProjNormBins];
        XFeatMeasurementCalibStats levelDiffStats[kNumLevelDiffBins];
        XFeatMeasurementCalibStats obsTypeStats[kNumObsTypeBins];

        const size_t n = std::min(F.mvpMapPoints.size(), F.mvbOutlier.size());
        for(size_t i = 0; i < n; ++i)
        {
            MapPoint* pMP = F.mvpMapPoints[i];
            if(!pMP)
                continue;

            float residualPx = std::numeric_limits<float>::quiet_NaN();
            float depth = std::numeric_limits<float>::quiet_NaN();
            ComputeFrameReprojectionError(F, static_cast<int>(i), pMP, residualPx, depth);
            const float chi2 = XFeatMeasurementApproxChi2(F, i, residualPx);
            const bool isInlier = !F.mvbOutlier[i];

            const unsigned char rawMatchSource = i < F.mvXFeatMatchSource.size()
                ? F.mvXFeatMatchSource[i]
                : static_cast<unsigned char>(kXFeatMatchUnknown);
            const int matchSourceBin = rawMatchSource < kNumMatchSourceBins
                ? static_cast<int>(rawMatchSource)
                : static_cast<int>(kXFeatMatchUnknown);
            const int mpSourceBin = static_cast<int>(MapPointCreationSourceIndex(pMP->GetCreationSource()));
            const int depthBin = DepthBinIndex(depth);
            const float bestDist = i < F.mvLocalProjBestDist.size()
                ? F.mvLocalProjBestDist[i]
                : std::numeric_limits<float>::quiet_NaN();
            const float secondDist = i < F.mvLocalProjSecondDist.size()
                ? F.mvLocalProjSecondDist[i]
                : std::numeric_limits<float>::quiet_NaN();
            const float ratio = i < F.mvLocalProjRatio.size()
                ? F.mvLocalProjRatio[i]
                : std::numeric_limits<float>::quiet_NaN();
            const float projNorm = i < F.mvLocalProjProjErrNorm.size()
                ? F.mvLocalProjProjErrNorm[i]
                : std::numeric_limits<float>::quiet_NaN();
            const int levelDiff = i < F.mvLocalProjLevelDiff.size()
                ? F.mvLocalProjLevelDiff[i]
                : -1;
            const int obsTypeBin = XFeatMeasurementObsTypeBin(F, i);

            AccumulateXFeatMeasurementCalibStats(matchSourceStats[matchSourceBin], isInlier, residualPx, chi2, depth);
            AccumulateXFeatMeasurementCalibStats(mpSourceStats[mpSourceBin], isInlier, residualPx, chi2, depth);
            AccumulateXFeatMeasurementCalibStats(depthStats[depthBin], isInlier, residualPx, chi2, depth);
            AccumulateXFeatMeasurementCalibStats(descriptorStats[XFeatMeasurementDescriptorBin(bestDist)], isInlier, residualPx, chi2, depth);
            AccumulateXFeatMeasurementCalibStats(ratioStats[XFeatMeasurementRatioBin(bestDist, ratio, secondDist)], isInlier, residualPx, chi2, depth);
            AccumulateXFeatMeasurementCalibStats(projNormStats[XFeatMeasurementProjNormBin(projNorm)], isInlier, residualPx, chi2, depth);
            AccumulateXFeatMeasurementCalibStats(levelDiffStats[XFeatMeasurementLevelDiffBin(levelDiff)], isInlier, residualPx, chi2, depth);
            AccumulateXFeatMeasurementCalibStats(obsTypeStats[obsTypeBin], isInlier, residualPx, chi2, depth);
        }

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6);
        for(int bin = 0; bin < kNumMatchSourceBins; ++bin)
            PrintXFeatMeasurementCalibStats(F, "match_source", XFeatMatchSourceName(static_cast<unsigned char>(bin)), matchSourceStats[bin]);
        for(int bin = 0; bin < kNumMpSourceBins; ++bin)
            PrintXFeatMeasurementCalibStats(F, "mp_source", MapPointCreationSourceName(static_cast<MapPoint::CreationSource>(bin)), mpSourceStats[bin]);
        for(int bin = 0; bin < kNumDepthBins; ++bin)
            PrintXFeatMeasurementCalibStats(F, "depth_bin", DepthBinName(bin), depthStats[bin]);
        for(int bin = 0; bin < kNumDescriptorBins; ++bin)
            PrintXFeatMeasurementCalibStats(F, "descriptor_dist_bin", XFeatMeasurementDescriptorBinName(bin), descriptorStats[bin]);
        for(int bin = 0; bin < kNumRatioBins; ++bin)
            PrintXFeatMeasurementCalibStats(F, "ratio_bin", XFeatMeasurementRatioBinName(bin), ratioStats[bin]);
        for(int bin = 0; bin < kNumProjNormBins; ++bin)
            PrintXFeatMeasurementCalibStats(F, "proj_norm_bin", XFeatMeasurementProjNormBinName(bin), projNormStats[bin]);
        for(int bin = 0; bin < kNumLevelDiffBins; ++bin)
            PrintXFeatMeasurementCalibStats(F, "level_diff_bin", XFeatMeasurementLevelDiffBinName(bin), levelDiffStats[bin]);
        for(int bin = 0; bin < kNumObsTypeBins; ++bin)
            PrintXFeatMeasurementCalibStats(F, "obs_type", XFeatMeasurementObsTypeBinName(bin), obsTypeStats[bin]);
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    int LocalProjMapUpdateBin(const bool bMapUpdated)
    {
        return bMapUpdated ? 1 : 0;
    }

    const char* LocalProjMapUpdateBinName(const int bin)
    {
        return bin == 1 ? "map_updated" : "no_map_update";
    }

    int LocalProjPriorSupportBin(const int assocBeforeSearch)
    {
        if(assocBeforeSearch < 120)
            return 0;
        if(assocBeforeSearch < 180)
            return 1;
        return 2;
    }

    const char* LocalProjPriorSupportBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "mid";
            case 2: return "strong";
            case 0:
            default:
                return "weak";
        }
    }

    int LocalProjLocalAddBin(const int assocAddedByLocal)
    {
        if(assocAddedByLocal < 80)
            return 0;
        if(assocAddedByLocal < 130)
            return 1;
        return 2;
    }

    const char* LocalProjLocalAddBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "mid";
            case 2: return "high";
            case 0:
            default:
                return "low";
        }
    }

    float LocalProjViewCosForMatch(const Frame& F, MapPoint* pMP, const size_t idx)
    {
        if(!pMP)
            return std::numeric_limits<float>::quiet_NaN();

        unsigned char side = 0;
        if(idx < F.mvLocalProjSide.size())
            side = F.mvLocalProjSide[idx];
        if(side == 2)
            return pMP->mTrackViewCosR;
        if(side == 1)
            return pMP->mTrackViewCos;

        if(F.Nleft != -1 && static_cast<int>(idx) >= F.Nleft)
            return pMP->mTrackViewCosR;
        return pMP->mTrackViewCos;
    }

    int LocalProjViewCosBin(const float viewCos)
    {
        if(!std::isfinite(viewCos))
            return 0;
        if(viewCos < 0.70f)
            return 1;
        if(viewCos < 0.90f)
            return 2;
        return 3;
    }

    const char* LocalProjViewCosBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "oblique";
            case 2: return "mid";
            case 3: return "frontal";
            case 0:
            default:
                return "missing";
        }
    }

    int LocalProjRadiusBin(const float radiusPx)
    {
        if(!std::isfinite(radiusPx) || radiusPx <= 0.0f)
            return 0;
        if(radiusPx <= 4.0f)
            return 1;
        if(radiusPx <= 8.0f)
            return 2;
        return 3;
    }

    const char* LocalProjRadiusBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "small";
            case 2: return "mid";
            case 3: return "large";
            case 0:
            default:
                return "missing";
        }
    }

    int LocalProjPoseContextBin(const int assocBeforeSearch,
                                const bool bMapUpdated,
                                const float projErrNorm,
                                const float radiusPx)
    {
        if(assocBeforeSearch < 120)
            return 0;
        if(bMapUpdated && std::isfinite(projErrNorm) && projErrNorm > 0.50f)
            return 1;
        if(std::isfinite(radiusPx) && radiusPx > 8.0f)
            return 2;
        return 3;
    }

    const char* LocalProjPoseContextBinName(const int bin)
    {
        switch(bin)
        {
            case 0: return "weak_prior";
            case 1: return "map_updated_edge";
            case 2: return "large_radius";
            case 3:
            default:
                return "normal";
        }
    }

    int ProjectionPriorShiftBin(const float norm)
    {
        return LocalProjNormBin(norm);
    }

    const char* ProjectionPriorShiftBinName(const int bin)
    {
        return LocalProjNormBinName(bin);
    }

    int ProjectionPriorConsistencyBin(const float preNorm, const float postNorm)
    {
        if(!std::isfinite(preNorm) || !std::isfinite(postNorm))
            return 0;
        const bool priorGood = preNorm <= 0.50f;
        const bool postGood = postNorm <= 0.50f;
        if(priorGood && postGood)
            return 1;
        if(!priorGood && postGood)
            return 2;
        if(priorGood && !postGood)
            return 3;
        return 4;
    }

    int LocalMapProvenanceRoleIndex(const Tracking::LocalKFProvenanceRole role)
    {
        switch(role)
        {
            case Tracking::LocalKFProvenanceRole::Reference:
                return 1;
            case Tracking::LocalKFProvenanceRole::Voted:
                return 2;
            case Tracking::LocalKFProvenanceRole::Covis:
                return 3;
            case Tracking::LocalKFProvenanceRole::ChildParent:
                return 4;
            case Tracking::LocalKFProvenanceRole::Temporal:
                return 5;
            case Tracking::LocalKFProvenanceRole::Unknown:
            default:
                return 0;
        }
    }

    int LocalMapProvenanceRoleRank(const Tracking::LocalKFProvenanceRole role)
    {
        switch(role)
        {
            case Tracking::LocalKFProvenanceRole::Reference:
                return 5;
            case Tracking::LocalKFProvenanceRole::Voted:
                return 4;
            case Tracking::LocalKFProvenanceRole::Covis:
                return 3;
            case Tracking::LocalKFProvenanceRole::ChildParent:
                return 2;
            case Tracking::LocalKFProvenanceRole::Temporal:
                return 1;
            case Tracking::LocalKFProvenanceRole::Unknown:
            default:
                return 0;
        }
    }

    const char* LocalMapProvenanceRoleName(const int bin)
    {
        switch(bin)
        {
            case 1: return "reference";
            case 2: return "voted";
            case 3: return "covis";
            case 4: return "child_parent";
            case 5: return "temporal";
            case 0:
            default:
                return "unknown";
        }
    }

    int LocalMapVoteBin(const int voteCount)
    {
        if(voteCount <= 0)
            return 0;
        if(voteCount < 5)
            return 1;
        if(voteCount < 15)
            return 2;
        return 3;
    }

    const char* LocalMapVoteBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "low";
            case 2: return "mid";
            case 3: return "high";
            case 0:
            default:
                return "no_vote";
        }
    }

    int LocalMapObservationBin(const int obs)
    {
        if(obs <= 0)
            return 0;
        if(obs == 1)
            return 1;
        if(obs <= 3)
            return 2;
        return 3;
    }

    const char* LocalMapObservationBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "one";
            case 2: return "few";
            case 3: return "many";
            case 0:
            default:
                return "unknown";
        }
    }

    int LocalMapKFAgeBin(const long int age)
    {
        if(age < 0)
            return 0;
        if(age <= 5)
            return 1;
        if(age <= 30)
            return 2;
        return 3;
    }

    const char* LocalMapKFAgeBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "recent";
            case 2: return "mid";
            case 3: return "old";
            case 0:
            default:
                return "unknown";
        }
    }

    int LocalMapProvenanceBin(const Tracking::LocalMPProvenance& provenance)
    {
        if(provenance.frameId == 0)
            return 0;
        if(provenance.localObservations <= 1)
            return 5;
        if(provenance.bestRole == Tracking::LocalKFProvenanceRole::Reference ||
           (provenance.bestRole == Tracking::LocalKFProvenanceRole::Voted &&
            provenance.bestVoteCount >= 15 &&
            provenance.localObservations >= 4))
        {
            return 1;
        }
        if(provenance.bestRole == Tracking::LocalKFProvenanceRole::Voted &&
           provenance.bestVoteCount < 5)
        {
            return 2;
        }
        if(provenance.bestRole == Tracking::LocalKFProvenanceRole::Covis ||
           provenance.bestRole == Tracking::LocalKFProvenanceRole::ChildParent ||
           provenance.bestRole == Tracking::LocalKFProvenanceRole::Temporal)
        {
            return provenance.bestKeyFrameAge > 30 ? 4 : 3;
        }
        return 0;
    }

    const char* LocalMapProvenanceBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "core_confirmed";
            case 2: return "weak_vote";
            case 3: return "expansion";
            case 4: return "old_expansion";
            case 5: return "single_obs";
            case 0:
            default:
                return "unknown";
        }
    }

    int CloseReuseContributorBin(const Tracking::LocalMPProvenance* provenance)
    {
        if(!provenance || provenance->frameId == 0)
            return 5;
        if(provenance->bestRole == Tracking::LocalKFProvenanceRole::Reference)
            return 0;
        if(provenance->bestRole == Tracking::LocalKFProvenanceRole::Voted)
        {
            if(provenance->bestVoteCount >= 15)
                return 1;
            if(provenance->bestVoteCount >= 5)
                return 2;
            return 3;
        }
        if(provenance->bestRole == Tracking::LocalKFProvenanceRole::Covis ||
           provenance->bestRole == Tracking::LocalKFProvenanceRole::ChildParent ||
           provenance->bestRole == Tracking::LocalKFProvenanceRole::Temporal)
        {
            return 4;
        }
        return 5;
    }

    const char* CloseReuseContributorName(const int bin)
    {
        switch(bin)
        {
            case 0: return "reference";
            case 1: return "strong_vote";
            case 2: return "mid_vote";
            case 3: return "weak_vote";
            case 4: return "expansion";
            case 5:
            default:
                return "unknown";
        }
    }

    void LogTrackLocalMapKFStructureDiagnostics(
        const Frame& F,
        KeyFrame* pReferenceKF,
        KeyFrame* pLastKF,
        const std::vector<KeyFrame*>& localKeyFrames,
        const std::vector<MapPoint*>& localMapPoints,
        const std::unordered_map<KeyFrame*, Tracking::LocalKFProvenance>& localKFProvenance,
        const std::unordered_map<MapPoint*, Tracking::LocalMPProvenance>& localMPProvenance,
        const int assocBeforeSearch,
        const int assocBeforeOpt,
        const int votedKFCandidates,
        const int votedKFKept,
        const int votedKFVisibilityKept,
        const float thDepth)
    {
        if(!ShouldPrintXFeatDebug(F.mnId))
            return;

        int roleCounts[6] = {};
        int voteSum = 0;
        int voteMax = 0;
        int votedKFs = 0;
        for(KeyFrame* pKF : localKeyFrames)
        {
            if(!pKF)
                continue;
            Tracking::LocalKFProvenanceRole role = Tracking::LocalKFProvenanceRole::Unknown;
            int vote = 0;
            const auto it = localKFProvenance.find(pKF);
            if(it != localKFProvenance.end() && it->second.frameId == F.mnId)
            {
                role = it->second.role;
                vote = it->second.voteCount;
            }
            const int idx = LocalMapProvenanceRoleIndex(role);
            if(idx >= 0 && idx < 6)
                roleCounts[idx]++;
            if(vote > 0)
            {
                votedKFs++;
                voteSum += vote;
                voteMax = std::max(voteMax, vote);
            }
        }

        int mpSourceCounts[kMapPointCreationSourceCount] = {};
        int mpRoleCounts[6] = {};
        int localMPValid = 0;
        int localMPTrackInView = 0;
        int localMPLmt0To10 = 0;
        int localMPTrackingStereo0To10 = 0;
        int localMPDepth0To5 = 0;
        int localMPDepth5To10 = 0;
        int localMPDepth10To30 = 0;
        int localMPDepthInvalid = 0;
        int localMPFromSingleObs = 0;
        int localMPFromWeakVote = 0;
        int localMPFromCoreConfirmed = 0;
        double depthSum = 0.0;
        int depthValid = 0;

        for(MapPoint* pMP : localMapPoints)
        {
            if(!pMP || pMP->isBad())
                continue;
            localMPValid++;
            const MapPoint::CreationSource source = pMP->GetCreationSource();
            mpSourceCounts[MapPointCreationSourceIndex(source)]++;
            if(pMP->mbTrackInView || pMP->mbTrackInViewR)
                localMPTrackInView++;

            const float depth = ResolveMapPointQualityDepth(F, pMP, pMP->mTrackDepth);
            const bool bSaneDepth = std::isfinite(depth) &&
                                    depth > 0.0f &&
                                    depth <= kLocalMapStructureMaxDiagDepth;
            const int depthBin = bSaneDepth ? DepthBinIndex(depth) : 0;
            if(!bSaneDepth || depthBin == 0)
                localMPDepthInvalid++;
            else
            {
                depthValid++;
                depthSum += depth;
                if(depthBin == 1)
                    localMPDepth0To5++;
                else if(depthBin == 2)
                    localMPDepth5To10++;
                else if(depthBin == 3)
                    localMPDepth10To30++;
            }
            if(std::isfinite(depth) && depth > 0.0f && depth <= 10.0f)
            {
                if(source == MapPoint::kCreationSourceLocalMappingTriangulation)
                    localMPLmt0To10++;
                else if(source == MapPoint::kCreationSourceTrackingStereo)
                    localMPTrackingStereo0To10++;
            }

            const auto it = localMPProvenance.find(pMP);
            if(it != localMPProvenance.end() && it->second.frameId == F.mnId)
            {
                const int roleIdx = LocalMapProvenanceRoleIndex(it->second.bestRole);
                if(roleIdx >= 0 && roleIdx < 6)
                    mpRoleCounts[roleIdx]++;
                const int provenanceBin = LocalMapProvenanceBin(it->second);
                if(provenanceBin == 1)
                    localMPFromCoreConfirmed++;
                else if(provenanceBin == 2)
                    localMPFromWeakVote++;
                else if(provenanceBin == 5)
                    localMPFromSingleObs++;
            }
        }

        const int assocAddedByLocal = std::max(0, assocBeforeOpt - assocBeforeSearch);
        const double inFrustumRatio = localMPValid > 0
            ? static_cast<double>(localMPTrackInView) / static_cast<double>(localMPValid)
            : 0.0;
        const double voteAvg = votedKFs > 0
            ? static_cast<double>(voteSum) / static_cast<double>(votedKFs)
            : 0.0;
        const double depthAvg = depthValid > 0
            ? depthSum / static_cast<double>(depthValid)
            : 0.0;
        const long int lastKFFrameId = pLastKF ? static_cast<long int>(pLastKF->mnFrameId) : -1;
        const long int framesSinceLastKF = lastKFFrameId >= 0 &&
                                           F.mnId >= static_cast<unsigned long>(lastKFFrameId)
            ? static_cast<long int>(F.mnId - static_cast<unsigned long>(lastKFFrameId))
            : -1;

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6)
                  << "[TrackLocalMap][KFStructure]"
                  << " frame_id=" << F.mnId
                  << " ref_kf_id=" << (pReferenceKF ? static_cast<long long>(pReferenceKF->mnId) : -1)
                  << " last_kf_id=" << (pLastKF ? static_cast<long long>(pLastKF->mnId) : -1)
                  << " last_kf_frame_id=" << lastKFFrameId
                  << " frames_since_last_kf=" << framesSinceLastKF
                  << " local_kfs=" << localKeyFrames.size()
                  << " local_kfs_reference=" << roleCounts[LocalMapProvenanceRoleIndex(Tracking::LocalKFProvenanceRole::Reference)]
                  << " local_kfs_voted=" << roleCounts[LocalMapProvenanceRoleIndex(Tracking::LocalKFProvenanceRole::Voted)]
                  << " local_kfs_covis=" << roleCounts[LocalMapProvenanceRoleIndex(Tracking::LocalKFProvenanceRole::Covis)]
                  << " local_kfs_child_parent=" << roleCounts[LocalMapProvenanceRoleIndex(Tracking::LocalKFProvenanceRole::ChildParent)]
                  << " local_kfs_temporal=" << roleCounts[LocalMapProvenanceRoleIndex(Tracking::LocalKFProvenanceRole::Temporal)]
                  << " local_kfs_unknown=" << roleCounts[LocalMapProvenanceRoleIndex(Tracking::LocalKFProvenanceRole::Unknown)]
                  << " local_kfs_voted_candidates=" << votedKFCandidates
                  << " local_kfs_voted_kept=" << votedKFKept
                  << " local_kfs_voted_visibility_kept=" << votedKFVisibilityKept
                  << " local_kf_vote_avg=" << voteAvg
                  << " local_kf_vote_max=" << voteMax
                  << " local_mps=" << localMapPoints.size()
                  << " local_mps_valid=" << localMPValid
                  << " local_mps_track_in_view=" << localMPTrackInView
                  << " local_mps_in_frustum_ratio=" << inFrustumRatio
                  << " local_mps_tracking_stereo=" << mpSourceCounts[MapPointCreationSourceIndex(MapPoint::kCreationSourceTrackingStereo)]
                  << " local_mps_lmt=" << mpSourceCounts[MapPointCreationSourceIndex(MapPoint::kCreationSourceLocalMappingTriangulation)]
                  << " local_mps_mono_init=" << mpSourceCounts[MapPointCreationSourceIndex(MapPoint::kCreationSourceMonoInit)]
                  << " local_mps_stereo_init=" << mpSourceCounts[MapPointCreationSourceIndex(MapPoint::kCreationSourceStereoInit)]
                  << " local_mps_unknown_source=" << mpSourceCounts[MapPointCreationSourceIndex(MapPoint::kCreationSourceUnknown)]
                  << " local_mps_lmt_0_10=" << localMPLmt0To10
                  << " local_mps_tracking_stereo_0_10=" << localMPTrackingStereo0To10
                  << " local_mps_depth_invalid=" << localMPDepthInvalid
                  << " local_mps_depth_0_5=" << localMPDepth0To5
                  << " local_mps_depth_5_10=" << localMPDepth5To10
                  << " local_mps_depth_10_30=" << localMPDepth10To30
                  << " local_mps_depth_avg=" << depthAvg
                  << " local_mps_from_reference=" << mpRoleCounts[LocalMapProvenanceRoleIndex(Tracking::LocalKFProvenanceRole::Reference)]
                  << " local_mps_from_voted=" << mpRoleCounts[LocalMapProvenanceRoleIndex(Tracking::LocalKFProvenanceRole::Voted)]
                  << " local_mps_from_expansion="
                  << (mpRoleCounts[LocalMapProvenanceRoleIndex(Tracking::LocalKFProvenanceRole::Covis)] +
                      mpRoleCounts[LocalMapProvenanceRoleIndex(Tracking::LocalKFProvenanceRole::ChildParent)])
                  << " local_mps_from_temporal=" << mpRoleCounts[LocalMapProvenanceRoleIndex(Tracking::LocalKFProvenanceRole::Temporal)]
                  << " local_mps_from_unknown=" << mpRoleCounts[LocalMapProvenanceRoleIndex(Tracking::LocalKFProvenanceRole::Unknown)]
                  << " local_mps_core_confirmed=" << localMPFromCoreConfirmed
                  << " local_mps_weak_vote=" << localMPFromWeakVote
                  << " local_mps_single_obs=" << localMPFromSingleObs
                  << " assoc_before_search=" << assocBeforeSearch
                  << " assoc_before_opt=" << assocBeforeOpt
                  << " assoc_added_by_local=" << assocAddedByLocal
                  << " mThDepth=" << thDepth
                  << std::endl;
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    const char* ProjectionPriorConsistencyBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "prior_good_post_good";
            case 2: return "prior_bad_post_good";
            case 3: return "prior_good_post_bad";
            case 4: return "prior_bad_post_bad";
            case 0:
            default:
                return "missing";
        }
    }

    bool ProjectMapPointWithPoseLeft(const Frame& F,
                                     MapPoint* pMP,
                                     const Sophus::SE3f& Tcw,
                                     float& u,
                                     float& v,
                                     float& depth)
    {
        u = std::numeric_limits<float>::quiet_NaN();
        v = std::numeric_limits<float>::quiet_NaN();
        depth = std::numeric_limits<float>::quiet_NaN();
        if(!pMP || !F.mpCamera)
            return false;

        const Eigen::Vector3f x3Dc = Tcw * pMP->GetWorldPos();
        depth = x3Dc(2);
        if(!std::isfinite(depth) || depth <= 0.0f)
            return false;

        const Eigen::Vector2f uv = F.mpCamera->project(x3Dc);
        u = uv(0);
        v = uv(1);
        return std::isfinite(u) && std::isfinite(v);
    }

    bool IsProjectionPriorShiftTarget(const Frame& F,
                                      const size_t i,
                                      MapPoint*& pMP,
                                      float& reprojErr,
                                      float& depth)
    {
        pMP = nullptr;
        reprojErr = std::numeric_limits<float>::quiet_NaN();
        depth = std::numeric_limits<float>::quiet_NaN();
        if(i >= F.mvpMapPoints.size() || i >= F.mvbOutlier.size())
            return false;

        pMP = F.mvpMapPoints[i];
        if(!pMP || pMP->GetCreationSource() != MapPoint::kCreationSourceLocalMappingTriangulation)
            return false;

        unsigned char matchSource = kXFeatMatchUnknown;
        if(i < F.mvXFeatMatchSource.size())
            matchSource = F.mvXFeatMatchSource[i];
        if(matchSource != kXFeatMatchLocalProjection)
            return false;

        ComputeFrameReprojectionError(F, static_cast<int>(i), pMP, reprojErr, depth);
        return IsLocalProjHistoryTarget(pMP, depth);
    }

    float NormalizedPixelDistance(const float x1,
                                  const float y1,
                                  const float x2,
                                  const float y2,
                                  const float radiusPx)
    {
        if(!std::isfinite(x1) || !std::isfinite(y1) ||
           !std::isfinite(x2) || !std::isfinite(y2) ||
           !std::isfinite(radiusPx) || radiusPx <= 0.0f)
        {
            return std::numeric_limits<float>::quiet_NaN();
        }
        const float dx = x1 - x2;
        const float dy = y1 - y2;
        return std::sqrt(dx * dx + dy * dy) / radiusPx;
    }

    void LogTrackLocalMapProjectionPriorShiftDiagnostics(const Frame& F,
                                                         const Sophus::SE3f& TcwPrior,
                                                         const int assocBeforeSearch,
                                                         const int assocAddedByLocal,
                                                         const int mnMatchesInliersForDiag,
                                                         const bool bMapUpdated)
    {
        if(!IsXFeatFeatureDiagEnabled())
            return;

        static constexpr int kNumNormBins = 5;
        static constexpr int kNumConsistencyBins = 5;
        static constexpr int kNumPoseContextBins = 4;

        ReprojectionConstraintStats priorShiftStats[kNumNormBins];
        ReprojectionConstraintStats preProjStats[kNumNormBins];
        ReprojectionConstraintStats postResidualStats[kNumNormBins];
        ReprojectionConstraintStats consistencyStats[kNumConsistencyBins];
        ReprojectionConstraintStats poseContextShiftStats[kNumPoseContextBins][kNumNormBins];
        std::vector<float> priorShiftSamples;

        const Sophus::SE3f TcwOpt = F.GetPose();
        const Sophus::SE3f deltaT = TcwOpt * TcwPrior.inverse();
        static constexpr float kRadToDeg = 57.29577951308232f;
        const float rotDeltaDeg = deltaT.so3().log().norm() * kRadToDeg;
        const float transDeltaM = deltaT.translation().norm();

        const size_t n = std::min(F.mvpMapPoints.size(), F.mvbOutlier.size());
        for(size_t i = 0; i < n; ++i)
        {
            MapPoint* pMP = nullptr;
            float reprojErr = std::numeric_limits<float>::quiet_NaN();
            float depth = std::numeric_limits<float>::quiet_NaN();
            if(!IsProjectionPriorShiftTarget(F, i, pMP, reprojErr, depth))
                continue;
            const unsigned char side = i < F.mvLocalProjSide.size()
                ? F.mvLocalProjSide[i]
                : 0;
            if(side != 1)
                continue;

            const bool isInlier = !F.mvbOutlier[i];
            const float radiusPx = i < F.mvLocalProjRadiusPx.size()
                ? F.mvLocalProjRadiusPx[i]
                : std::numeric_limits<float>::quiet_NaN();
            const float preProjNorm = i < F.mvLocalProjProjErrNorm.size()
                ? F.mvLocalProjProjErrNorm[i]
                : std::numeric_limits<float>::quiet_NaN();
            const float preProjX = i < F.mvLocalProjProjX.size()
                ? F.mvLocalProjProjX[i]
                : std::numeric_limits<float>::quiet_NaN();
            const float preProjY = i < F.mvLocalProjProjY.size()
                ? F.mvLocalProjProjY[i]
                : std::numeric_limits<float>::quiet_NaN();

            cv::KeyPoint kp;
            const bool bHaveKeyPoint = GetFrameKeyPointByIndex(F, static_cast<int>(i), kp);
            float optProjX = std::numeric_limits<float>::quiet_NaN();
            float optProjY = std::numeric_limits<float>::quiet_NaN();
            float optDepth = std::numeric_limits<float>::quiet_NaN();
            const bool bProjected = ProjectMapPointWithPoseLeft(F, pMP, TcwOpt, optProjX, optProjY, optDepth);
            const float priorShiftNorm = bProjected
                ? NormalizedPixelDistance(preProjX, preProjY, optProjX, optProjY, radiusPx)
                : std::numeric_limits<float>::quiet_NaN();
            const float postResidualNorm = (bProjected && bHaveKeyPoint)
                ? NormalizedPixelDistance(optProjX, optProjY, kp.pt.x, kp.pt.y, radiusPx)
                : std::numeric_limits<float>::quiet_NaN();

            if(std::isfinite(priorShiftNorm))
                priorShiftSamples.push_back(priorShiftNorm);

            const int priorShiftBin = ProjectionPriorShiftBin(priorShiftNorm);
            const int preProjBin = ProjectionPriorShiftBin(preProjNorm);
            const int postResidualBin = ProjectionPriorShiftBin(postResidualNorm);
            const int consistencyBin = ProjectionPriorConsistencyBin(preProjNorm, postResidualNorm);
            const int poseContextBin = LocalProjPoseContextBin(assocBeforeSearch,
                                                               bMapUpdated,
                                                               preProjNorm,
                                                               radiusPx);

            AccumulateReprojectionConstraint(priorShiftStats[priorShiftBin],
                                             reprojErr, depth, isInlier);
            AccumulateReprojectionConstraint(preProjStats[preProjBin],
                                             reprojErr, depth, isInlier);
            AccumulateReprojectionConstraint(postResidualStats[postResidualBin],
                                             reprojErr, depth, isInlier);
            AccumulateReprojectionConstraint(consistencyStats[consistencyBin],
                                             reprojErr, depth, isInlier);
            AccumulateReprojectionConstraint(poseContextShiftStats[poseContextBin][priorShiftBin],
                                             reprojErr, depth, isInlier);
        }

        const double priorShiftAvg = !priorShiftSamples.empty()
            ? std::accumulate(priorShiftSamples.begin(), priorShiftSamples.end(), 0.0) /
              static_cast<double>(priorShiftSamples.size())
            : 0.0;
        const float priorShiftP90 = priorShiftSamples.empty()
            ? 0.0f
            : DepthPercentile(priorShiftSamples, 0.90f);

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6)
                  << "[TrackLocalMap][PosePriorShift]"
                  << " frame_id=" << F.mnId
                  << " mbMapUpdated=" << (bMapUpdated ? 1 : 0)
                  << " assoc_before_search=" << assocBeforeSearch
                  << " assoc_added_by_local=" << assocAddedByLocal
                  << " mnMatchesInliers=" << mnMatchesInliersForDiag
                  << " rot_delta_deg=" << rotDeltaDeg
                  << " trans_delta_m=" << transDeltaM
                  << " prior_shift_norm_avg=" << priorShiftAvg
                  << " prior_shift_norm_p90=" << priorShiftP90
                  << std::endl;

        for(int bin = 0; bin < kNumNormBins; ++bin)
        {
            if(priorShiftStats[bin].total > 0)
            {
                std::cout << "[TrackLocalMap][ProjectionPriorShiftError]"
                          << " frame_id=" << F.mnId
                          << " dimension=prior_shift_norm_bin"
                          << " bin=" << ProjectionPriorShiftBinName(bin);
                AppendTriangulationQualityErrorStats(std::cout, priorShiftStats[bin]);
                std::cout << std::endl;
            }
            if(preProjStats[bin].total > 0)
            {
                std::cout << "[TrackLocalMap][ProjectionPriorShiftError]"
                          << " frame_id=" << F.mnId
                          << " dimension=pre_proj_norm_bin"
                          << " bin=" << ProjectionPriorShiftBinName(bin);
                AppendTriangulationQualityErrorStats(std::cout, preProjStats[bin]);
                std::cout << std::endl;
            }
            if(postResidualStats[bin].total > 0)
            {
                std::cout << "[TrackLocalMap][ProjectionPriorShiftError]"
                          << " frame_id=" << F.mnId
                          << " dimension=post_residual_norm_bin"
                          << " bin=" << ProjectionPriorShiftBinName(bin);
                AppendTriangulationQualityErrorStats(std::cout, postResidualStats[bin]);
                std::cout << std::endl;
            }
        }
        for(int bin = 0; bin < kNumConsistencyBins; ++bin)
        {
            if(consistencyStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][ProjectionPriorShiftError]"
                      << " frame_id=" << F.mnId
                      << " dimension=prior_post_consistency_bin"
                      << " bin=" << ProjectionPriorConsistencyBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, consistencyStats[bin]);
            std::cout << std::endl;
        }
        for(int poseBin = 0; poseBin < kNumPoseContextBins; ++poseBin)
        {
            for(int shiftBin = 0; shiftBin < kNumNormBins; ++shiftBin)
            {
                if(poseContextShiftStats[poseBin][shiftBin].total <= 0)
                    continue;
                std::cout << "[TrackLocalMap][ProjectionPriorShiftError]"
                          << " frame_id=" << F.mnId
                          << " dimension=pose_context_x_shift_bin"
                          << " bin=" << LocalProjPoseContextBinName(poseBin)
                          << "_" << ProjectionPriorShiftBinName(shiftBin);
                AppendTriangulationQualityErrorStats(std::cout, poseContextShiftStats[poseBin][shiftBin]);
                std::cout << std::endl;
            }
        }
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    void LogTrackLocalMapLocalProjPoseContextDiagnostics(const Frame& F,
                                                        const int assocBeforeSearch,
                                                        const int assocAddedByLocal,
                                                        const bool bMapUpdated)
    {
        if(!IsXFeatFeatureDiagEnabled())
            return;

        static constexpr int kNumMapUpdateBins = 2;
        static constexpr int kNumPriorSupportBins = 3;
        static constexpr int kNumLocalAddBins = 3;
        static constexpr int kNumViewCosBins = 4;
        static constexpr int kNumRadiusBins = 4;
        static constexpr int kNumPoseContextBins = 4;

        ReprojectionConstraintStats mapUpdateStats[kNumMapUpdateBins];
        ReprojectionConstraintStats priorSupportStats[kNumPriorSupportBins];
        ReprojectionConstraintStats localAddStats[kNumLocalAddBins];
        ReprojectionConstraintStats viewCosStats[kNumViewCosBins];
        ReprojectionConstraintStats radiusStats[kNumRadiusBins];
        ReprojectionConstraintStats poseContextStats[kNumPoseContextBins];

        const int mapUpdateBin = LocalProjMapUpdateBin(bMapUpdated);
        const int priorSupportBin = LocalProjPriorSupportBin(assocBeforeSearch);
        const int localAddBin = LocalProjLocalAddBin(assocAddedByLocal);

        const size_t n = std::min(F.mvpMapPoints.size(), F.mvbOutlier.size());
        for(size_t i = 0; i < n; ++i)
        {
            MapPoint* pMP = F.mvpMapPoints[i];
            if(!pMP || pMP->GetCreationSource() != MapPoint::kCreationSourceLocalMappingTriangulation)
                continue;

            unsigned char matchSource = kXFeatMatchUnknown;
            if(i < F.mvXFeatMatchSource.size())
                matchSource = F.mvXFeatMatchSource[i];
            if(matchSource != kXFeatMatchLocalProjection)
                continue;

            float reprojErr = std::numeric_limits<float>::quiet_NaN();
            float depth = std::numeric_limits<float>::quiet_NaN();
            ComputeFrameReprojectionError(F, static_cast<int>(i), pMP, reprojErr, depth);
            if(!IsLocalProjHistoryTarget(pMP, depth))
                continue;

            const bool isInlier = !F.mvbOutlier[i];
            const float radiusPx = i < F.mvLocalProjRadiusPx.size()
                ? F.mvLocalProjRadiusPx[i]
                : std::numeric_limits<float>::quiet_NaN();
            const float projErrNorm = i < F.mvLocalProjProjErrNorm.size()
                ? F.mvLocalProjProjErrNorm[i]
                : std::numeric_limits<float>::quiet_NaN();
            const float viewCos = LocalProjViewCosForMatch(F, pMP, i);

            AccumulateReprojectionConstraint(mapUpdateStats[mapUpdateBin],
                                             reprojErr, depth, isInlier);
            AccumulateReprojectionConstraint(priorSupportStats[priorSupportBin],
                                             reprojErr, depth, isInlier);
            AccumulateReprojectionConstraint(localAddStats[localAddBin],
                                             reprojErr, depth, isInlier);
            AccumulateReprojectionConstraint(viewCosStats[LocalProjViewCosBin(viewCos)],
                                             reprojErr, depth, isInlier);
            AccumulateReprojectionConstraint(radiusStats[LocalProjRadiusBin(radiusPx)],
                                             reprojErr, depth, isInlier);
            AccumulateReprojectionConstraint(
                poseContextStats[LocalProjPoseContextBin(assocBeforeSearch,
                                                         bMapUpdated,
                                                         projErrNorm,
                                                         radiusPx)],
                reprojErr, depth, isInlier);
        }

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6);
        for(int bin = 0; bin < kNumMapUpdateBins; ++bin)
        {
            if(mapUpdateStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][LocalProjPoseContextError]"
                      << " frame_id=" << F.mnId
                      << " dimension=map_update_bin"
                      << " bin=" << LocalProjMapUpdateBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, mapUpdateStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumPriorSupportBins; ++bin)
        {
            if(priorSupportStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][LocalProjPoseContextError]"
                      << " frame_id=" << F.mnId
                      << " dimension=prior_support_bin"
                      << " bin=" << LocalProjPriorSupportBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, priorSupportStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumLocalAddBins; ++bin)
        {
            if(localAddStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][LocalProjPoseContextError]"
                      << " frame_id=" << F.mnId
                      << " dimension=local_add_bin"
                      << " bin=" << LocalProjLocalAddBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, localAddStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumViewCosBins; ++bin)
        {
            if(viewCosStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][LocalProjPoseContextError]"
                      << " frame_id=" << F.mnId
                      << " dimension=view_cos_bin"
                      << " bin=" << LocalProjViewCosBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, viewCosStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumRadiusBins; ++bin)
        {
            if(radiusStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][LocalProjPoseContextError]"
                      << " frame_id=" << F.mnId
                      << " dimension=radius_bin"
                      << " bin=" << LocalProjRadiusBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, radiusStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumPoseContextBins; ++bin)
        {
            if(poseContextStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][LocalProjPoseContextError]"
                      << " frame_id=" << F.mnId
                      << " dimension=pose_context_bin"
                      << " bin=" << LocalProjPoseContextBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, poseContextStats[bin]);
            std::cout << std::endl;
        }
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    void LogTrackLocalMapLocalMapProvenanceDiagnostics(
        const Frame& F,
        const std::unordered_map<MapPoint*, Tracking::LocalMPProvenance>& mpProvenance)
    {
        if(!IsXFeatFeatureDiagEnabled())
            return;

        static constexpr int kNumRoleBins = 6;
        static constexpr int kNumVoteBins = 4;
        static constexpr int kNumObservationBins = 4;
        static constexpr int kNumAgeBins = 4;
        static constexpr int kNumProvenanceBins = 6;

        ReprojectionConstraintStats roleStats[kNumRoleBins];
        ReprojectionConstraintStats voteStats[kNumVoteBins];
        ReprojectionConstraintStats observationStats[kNumObservationBins];
        ReprojectionConstraintStats ageStats[kNumAgeBins];
        ReprojectionConstraintStats provenanceStats[kNumProvenanceBins];

        const size_t n = std::min(F.mvpMapPoints.size(), F.mvbOutlier.size());
        for(size_t i = 0; i < n; ++i)
        {
            MapPoint* pMP = F.mvpMapPoints[i];
            if(!pMP || pMP->GetCreationSource() != MapPoint::kCreationSourceLocalMappingTriangulation)
                continue;

            unsigned char matchSource = kXFeatMatchUnknown;
            if(i < F.mvXFeatMatchSource.size())
                matchSource = F.mvXFeatMatchSource[i];
            if(matchSource != kXFeatMatchLocalProjection)
                continue;

            float reprojErr = std::numeric_limits<float>::quiet_NaN();
            float depth = std::numeric_limits<float>::quiet_NaN();
            ComputeFrameReprojectionError(F, static_cast<int>(i), pMP, reprojErr, depth);
            if(!IsLocalProjHistoryTarget(pMP, depth))
                continue;

            Tracking::LocalMPProvenance provenance;
            const auto itProv = mpProvenance.find(pMP);
            if(itProv != mpProvenance.end() && itProv->second.frameId == F.mnId)
                provenance = itProv->second;

            const bool isInlier = !F.mvbOutlier[i];
            AccumulateReprojectionConstraint(roleStats[LocalMapProvenanceRoleIndex(provenance.bestRole)],
                                             reprojErr,
                                             depth,
                                             isInlier);
            AccumulateReprojectionConstraint(voteStats[LocalMapVoteBin(provenance.bestVoteCount)],
                                             reprojErr,
                                             depth,
                                             isInlier);
            AccumulateReprojectionConstraint(observationStats[LocalMapObservationBin(provenance.localObservations)],
                                             reprojErr,
                                             depth,
                                             isInlier);
            AccumulateReprojectionConstraint(ageStats[LocalMapKFAgeBin(provenance.bestKeyFrameAge)],
                                             reprojErr,
                                             depth,
                                             isInlier);
            AccumulateReprojectionConstraint(provenanceStats[LocalMapProvenanceBin(provenance)],
                                             reprojErr,
                                             depth,
                                             isInlier);
        }

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6);
        for(int bin = 0; bin < kNumRoleBins; ++bin)
        {
            if(roleStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][LocalMapProvenanceError]"
                      << " frame_id=" << F.mnId
                      << " dimension=kf_role_bin"
                      << " bin=" << LocalMapProvenanceRoleName(bin);
            AppendTriangulationQualityErrorStats(std::cout, roleStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumVoteBins; ++bin)
        {
            if(voteStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][LocalMapProvenanceError]"
                      << " frame_id=" << F.mnId
                      << " dimension=kf_vote_bin"
                      << " bin=" << LocalMapVoteBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, voteStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumObservationBins; ++bin)
        {
            if(observationStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][LocalMapProvenanceError]"
                      << " frame_id=" << F.mnId
                      << " dimension=mp_local_obs_bin"
                      << " bin=" << LocalMapObservationBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, observationStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumAgeBins; ++bin)
        {
            if(ageStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][LocalMapProvenanceError]"
                      << " frame_id=" << F.mnId
                      << " dimension=kf_age_bin"
                      << " bin=" << LocalMapKFAgeBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, ageStats[bin]);
            std::cout << std::endl;
        }
        for(int bin = 0; bin < kNumProvenanceBins; ++bin)
        {
            if(provenanceStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][LocalMapProvenanceError]"
                      << " frame_id=" << F.mnId
                      << " dimension=provenance_bin"
                      << " bin=" << LocalMapProvenanceBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, provenanceStats[bin]);
            std::cout << std::endl;
        }
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    void LogTrackLocalMapLocalProjPriorityDiagnostics(
        const Frame& F,
        const std::unordered_map<MapPoint*, Tracking::LocalMPProvenance>& mpProvenance)
    {
        if(!IsXFeatFeatureDiagEnabled())
            return;

        ReprojectionConstraintStats priorityStats[kLocalProjPriorityBinCount];

        const size_t n = std::min(F.mvpMapPoints.size(), F.mvbOutlier.size());
        for(size_t i = 0; i < n; ++i)
        {
            MapPoint* pMP = F.mvpMapPoints[i];
            if(!pMP || pMP->GetCreationSource() != MapPoint::kCreationSourceLocalMappingTriangulation)
                continue;

            unsigned char matchSource = kXFeatMatchUnknown;
            if(i < F.mvXFeatMatchSource.size())
                matchSource = F.mvXFeatMatchSource[i];
            if(matchSource != kXFeatMatchLocalProjection)
                continue;

            float reprojErr = std::numeric_limits<float>::quiet_NaN();
            float depth = std::numeric_limits<float>::quiet_NaN();
            ComputeFrameReprojectionError(F, static_cast<int>(i), pMP, reprojErr, depth);
            if(!IsLocalProjHistoryTarget(pMP, depth))
                continue;

            const Tracking::LocalMPProvenance* pProvenance = nullptr;
            const auto itProv = mpProvenance.find(pMP);
            if(itProv != mpProvenance.end() && itProv->second.frameId == F.mnId)
                pProvenance = &itProv->second;

            const int priorityBin = LocalProjPriorityBin(pMP, pProvenance, depth);
            if(priorityBin < 0 || priorityBin >= kLocalProjPriorityBinCount)
                continue;

            AccumulateReprojectionConstraint(priorityStats[priorityBin],
                                             reprojErr,
                                             depth,
                                             !F.mvbOutlier[i]);
        }

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6);
        for(int bin = 0; bin < kLocalProjPriorityBinCount; ++bin)
        {
            if(priorityStats[bin].total <= 0)
                continue;
            std::cout << "[TrackLocalMap][LocalProjPriorityError]"
                      << " frame_id=" << F.mnId
                      << " dimension=priority_bin"
                      << " bin=" << LocalProjPriorityBinName(bin);
            AppendTriangulationQualityErrorStats(std::cout, priorityStats[bin]);
            std::cout << std::endl;
        }
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    void RecordTrackLocalMapTriangulationLocalProjHistoryOutcomes(Frame& F)
    {
        const size_t n = std::min(F.mvpMapPoints.size(), F.mvbOutlier.size());
        for(size_t i = 0; i < n; ++i)
        {
            MapPoint* pMP = F.mvpMapPoints[i];
            if(!pMP || pMP->GetCreationSource() != MapPoint::kCreationSourceLocalMappingTriangulation)
                continue;

            unsigned char matchSource = kXFeatMatchUnknown;
            if(i < F.mvXFeatMatchSource.size())
                matchSource = F.mvXFeatMatchSource[i];
            if(matchSource != kXFeatMatchLocalProjection)
                continue;

            float reprojErr = std::numeric_limits<float>::quiet_NaN();
            float depth = std::numeric_limits<float>::quiet_NaN();
            ComputeFrameReprojectionError(F, static_cast<int>(i), pMP, reprojErr, depth);
            if(!IsLocalProjHistoryTarget(pMP, depth))
                continue;

            pMP->RecordLocalProjectionOutcome(F.mnId, !F.mvbOutlier[i]);
        }
    }

    void RecordTrackLocalMapCloseReuseOutcomes(Frame& F)
    {
        if(!IsXFeatDebugEnabled() || !IsXFeatFeatureDiagEnabled())
            return;

        const size_t n = std::min(F.mvpMapPoints.size(), F.mvbOutlier.size());
        for(size_t i = 0; i < n; ++i)
        {
            MapPoint* pMP = F.mvpMapPoints[i];
            if(!pMP)
                continue;

            const unsigned char matchSource = i < F.mvXFeatMatchSource.size()
                ? F.mvXFeatMatchSource[i]
                : kXFeatMatchUnknown;
            if(matchSource != kXFeatMatchLocalProjection)
                continue;

            float reprojErr = std::numeric_limits<float>::quiet_NaN();
            float depth = std::numeric_limits<float>::quiet_NaN();
            ComputeFrameReprojectionError(F, static_cast<int>(i), pMP, reprojErr, depth);
            if(IsCloseReuseTarget(pMP, depth))
                pMP->RecordCloseReuseOutcome(F.mnId, !F.mvbOutlier[i]);
        }
    }

    void LogTrackLocalMapCloseReuseOutcomeDiagnostics(const Frame& F)
    {
        if(!IsXFeatFeatureDiagEnabled())
            return;

        ReprojectionConstraintStats stats[kCloseReuseSourceBinCount][kCloseReuseDepthBinCount];
        const size_t n = std::min(F.mvpMapPoints.size(), F.mvbOutlier.size());
        for(size_t i = 0; i < n; ++i)
        {
            MapPoint* pMP = F.mvpMapPoints[i];
            if(!pMP)
                continue;
            const unsigned char matchSource = i < F.mvXFeatMatchSource.size()
                ? F.mvXFeatMatchSource[i]
                : kXFeatMatchUnknown;
            if(matchSource != kXFeatMatchLocalProjection)
                continue;

            float reprojErr = std::numeric_limits<float>::quiet_NaN();
            float depth = std::numeric_limits<float>::quiet_NaN();
            ComputeFrameReprojectionError(F, static_cast<int>(i), pMP, reprojErr, depth);
            if(!IsCloseReuseTarget(pMP, depth))
                continue;
            const int sourceBin = CloseReuseSourceBin(pMP->GetCreationSource());
            const int depthBin = CloseReuseDepthBin(depth);
            if(sourceBin < 0 || depthBin < 0)
                continue;
            AccumulateReprojectionConstraint(stats[sourceBin][depthBin],
                                             reprojErr,
                                             depth,
                                             !F.mvbOutlier[i]);
        }

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6);
        for(int sourceBin = 0; sourceBin < kCloseReuseSourceBinCount; ++sourceBin)
        {
            for(int depthBin = 0; depthBin < kCloseReuseDepthBinCount; ++depthBin)
            {
                if(stats[sourceBin][depthBin].total <= 0)
                    continue;
                std::cout << "[TrackLocalMap][CloseReuseOutcome]"
                          << " frame_id=" << F.mnId
                          << " source=" << CloseReuseSourceName(sourceBin)
                          << " depth_bin=" << CloseReuseDepthName(depthBin);
                AppendTriangulationQualityErrorStats(std::cout, stats[sourceBin][depthBin]);
                std::cout << std::endl;
            }
        }
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    void LogTrackLocalMapCloseReuseContributionOutcomeDiagnostics(
        const Frame& F,
        const std::unordered_map<MapPoint*, Tracking::LocalMPProvenance>& mpProvenance)
    {
        if(!IsXFeatFeatureDiagEnabled())
            return;

        ReprojectionConstraintStats stats[kCloseReuseSourceBinCount][kCloseReuseContributorBinCount];
        const size_t n = std::min(F.mvpMapPoints.size(), F.mvbOutlier.size());
        for(size_t i = 0; i < n; ++i)
        {
            MapPoint* pMP = F.mvpMapPoints[i];
            if(!pMP)
                continue;
            const unsigned char matchSource = i < F.mvXFeatMatchSource.size()
                ? F.mvXFeatMatchSource[i]
                : kXFeatMatchUnknown;
            if(matchSource != kXFeatMatchLocalProjection)
                continue;

            float reprojErr = std::numeric_limits<float>::quiet_NaN();
            float depth = std::numeric_limits<float>::quiet_NaN();
            ComputeFrameReprojectionError(F, static_cast<int>(i), pMP, reprojErr, depth);
            if(!IsCloseReuseTarget(pMP, depth))
                continue;
            const int sourceBin = CloseReuseSourceBin(pMP->GetCreationSource());
            if(sourceBin < 0)
                continue;

            const Tracking::LocalMPProvenance* pProvenance = nullptr;
            const auto itProv = mpProvenance.find(pMP);
            if(itProv != mpProvenance.end() && itProv->second.frameId == F.mnId)
                pProvenance = &itProv->second;
            const int contributorBin = CloseReuseContributorBin(pProvenance);
            AccumulateReprojectionConstraint(stats[sourceBin][contributorBin],
                                             reprojErr,
                                             depth,
                                             !F.mvbOutlier[i]);
        }

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6);
        for(int sourceBin = 0; sourceBin < kCloseReuseSourceBinCount; ++sourceBin)
        {
            for(int contributorBin = 0;
                contributorBin < kCloseReuseContributorBinCount;
                ++contributorBin)
            {
                if(stats[sourceBin][contributorBin].total <= 0)
                    continue;
                std::cout << "[TrackLocalMap][CloseReuseContributionOutcome]"
                          << " frame_id=" << F.mnId
                          << " source=" << CloseReuseSourceName(sourceBin)
                          << " contributor=" << CloseReuseContributorName(contributorBin);
                AppendTriangulationQualityErrorStats(std::cout, stats[sourceBin][contributorBin]);
                std::cout << std::endl;
            }
        }
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    void LogTrackLocalMapCloseReuseCohortAliveDiagnostics(const Frame& F, Map* pMap)
    {
        if(!IsXFeatFeatureDiagEnabled() || !pMap)
            return;

        struct CohortStats
        {
            int activeEligible = 0;
            int attempted = 0;
            int matched = 0;
            int optInlier = 0;
            int optOutlier = 0;
        };
        static constexpr int kNumHorizons = 3;
        static constexpr int kHorizons[kNumHorizons] = {5, 10, 20};
        CohortStats stats[kCloseReuseSourceBinCount][kCloseReuseDepthBinCount][kNumHorizons];
        const long unsigned int invalidFrame = std::numeric_limits<long unsigned int>::max();

        const std::vector<MapPoint*> mapPoints = pMap->GetAllMapPoints();
        for(MapPoint* pMP : mapPoints)
        {
            if(!pMP || pMP->isBad() || pMP->mnFirstFrame < 0)
                continue;
            const float creationDepth = ResolveMapPointCreationDepthForCloseReuse(pMP);
            if(!IsCloseReuseTarget(pMP, creationDepth))
                continue;
            const int sourceBin = CloseReuseSourceBin(pMP->GetCreationSource());
            const int depthBin = CloseReuseDepthBin(creationDepth);
            if(sourceBin < 0 || depthBin < 0 ||
               F.mnId < static_cast<long unsigned int>(pMP->mnFirstFrame))
            {
                continue;
            }

            const long unsigned int creationFrame = static_cast<long unsigned int>(pMP->mnFirstFrame);
            const long unsigned int age = F.mnId - creationFrame;
            const MapPoint::CloseReuseCohortStats reuse = pMP->GetCloseReuseCohortStats();
            auto OccurredWithin = [&](const long unsigned int eventFrame, const int horizon)
            {
                return eventFrame != invalidFrame &&
                       eventFrame >= creationFrame &&
                       eventFrame - creationFrame <= static_cast<long unsigned int>(horizon);
            };

            for(int horizonIdx = 0; horizonIdx < kNumHorizons; ++horizonIdx)
            {
                const int horizon = kHorizons[horizonIdx];
                if(age < static_cast<long unsigned int>(horizon))
                    continue;
                CohortStats& bucket = stats[sourceBin][depthBin][horizonIdx];
                bucket.activeEligible++;
                if(OccurredWithin(reuse.firstProjectionAttemptFrame, horizon))
                    bucket.attempted++;
                if(OccurredWithin(reuse.firstProjectionMatchFrame, horizon))
                    bucket.matched++;
                if(OccurredWithin(reuse.firstOptInlierFrame, horizon))
                    bucket.optInlier++;
                if(OccurredWithin(reuse.firstOptOutlierFrame, horizon))
                    bucket.optOutlier++;
            }
        }

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6);
        for(int sourceBin = 0; sourceBin < kCloseReuseSourceBinCount; ++sourceBin)
        {
            for(int depthBin = 0; depthBin < kCloseReuseDepthBinCount; ++depthBin)
            {
                for(int horizonIdx = 0; horizonIdx < kNumHorizons; ++horizonIdx)
                {
                    const CohortStats& bucket = stats[sourceBin][depthBin][horizonIdx];
                    if(bucket.activeEligible <= 0)
                        continue;
                    std::cout << "[TrackLocalMap][CloseReuseCohortAlive]"
                              << " frame_id=" << F.mnId
                              << " source=" << CloseReuseSourceName(sourceBin)
                              << " depth_bin=" << CloseReuseDepthName(depthBin)
                              << " horizon_frames=" << kHorizons[horizonIdx]
                              << " active_eligible=" << bucket.activeEligible
                              << " attempted=" << bucket.attempted
                              << " matched=" << bucket.matched
                              << " opt_inlier=" << bucket.optInlier
                              << " opt_outlier=" << bucket.optOutlier
                              << " attempt_rate="
                              << static_cast<float>(bucket.attempted) / static_cast<float>(bucket.activeEligible)
                              << " match_rate="
                              << static_cast<float>(bucket.matched) / static_cast<float>(bucket.activeEligible)
                              << " inlier_rate="
                              << static_cast<float>(bucket.optInlier) / static_cast<float>(bucket.activeEligible)
                              << std::endl;
                }
            }
        }
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    void LogTrackLocalMapMappingRepeatabilityCohortDiagnostics(const Frame& F, Map* pMap)
    {
        if(!IsXFeatFeatureDiagEnabled() || !pMap)
            return;

        struct CohortStats
        {
            int eligible = 0;
            int attempted = 0;
            int matched = 0;
            int optInlier = 0;
            int optOutlier = 0;
            int stable = 0;
            int obsSum = 0;
            float foundRatioSum = 0.0f;
            int foundRatioValid = 0;
        };

        enum RepeatabilityDimension
        {
            kRepeatGrid = 0,
            kRepeatOctave = 1,
            kRepeatResponse = 2,
            kRepeatMappingFeature = 3,
            kRepeatDimensionCount = 4
        };

        static constexpr int kNumHorizons = 3;
        static constexpr int kHorizons[kNumHorizons] = {5, 10, 20};
        static constexpr int kMaxRepeatBins = 6;
        const long unsigned int invalidFrame = std::numeric_limits<long unsigned int>::max();

        CohortStats stats[kRepeatDimensionCount][kMaxRepeatBins][6][kNumHorizons];
        const std::vector<MapPoint*> mapPoints = pMap->GetAllMapPoints();
        for(MapPoint* pMP : mapPoints)
        {
            if(!pMP || pMP->isBad() || pMP->mnFirstFrame < 0)
                continue;
            if(pMP->GetCreationSource() != MapPoint::kCreationSourceLocalMappingTriangulation)
                continue;

            const float creationDepth = ResolveMapPointCreationDepthForCloseReuse(pMP);
            const int depthBin = DepthBinIndex(creationDepth);
            if(depthBin != 1 && depthBin != 2)
                continue;
            if(F.mnId < static_cast<long unsigned int>(pMP->mnFirstFrame))
                continue;

            const bool hasFeatureQuality = pMP->HasMappingFeatureQuality();
            const int bins[kRepeatDimensionCount] = {
                hasFeatureQuality ? MappingFeatureGridBin(pMP->mnMappingFeatureGridX, pMP->mnMappingFeatureGridY) : 0,
                hasFeatureQuality ? MappingFeatureOctaveBin(pMP->mnMappingFeatureOctave) : 0,
                hasFeatureQuality ? MappingFeatureResponseBin(pMP->mMappingFeatureResponse) : 0,
                hasFeatureQuality ? MappingFeatureCombinedBin(
                    MappingFeatureGridBin(pMP->mnMappingFeatureGridX, pMP->mnMappingFeatureGridY),
                    MappingFeatureOctaveBin(pMP->mnMappingFeatureOctave),
                    MappingFeatureResponseBin(pMP->mMappingFeatureResponse),
                    MappingFeatureStereoQualityBin(pMP->mMappingFeatureStereoBirthQuality)) : 0
            };

            const long unsigned int creationFrame = static_cast<long unsigned int>(pMP->mnFirstFrame);
            const long unsigned int age = F.mnId - creationFrame;
            const MapPoint::CloseReuseCohortStats reuse = pMP->GetCloseReuseCohortStats();
            const int observations = pMP->Observations();
            const float foundRatio = pMP->GetFoundRatio();
            const bool bStable =
                observations >= 4 && std::isfinite(foundRatio) && foundRatio >= 0.35f;

            auto OccurredWithin = [&](const long unsigned int eventFrame, const int horizon)
            {
                return eventFrame != invalidFrame &&
                       eventFrame >= creationFrame &&
                       eventFrame - creationFrame <= static_cast<long unsigned int>(horizon);
            };

            for(int dim = 0; dim < kRepeatDimensionCount; ++dim)
            {
                const int bin = bins[dim];
                if(bin < 0 || bin >= kMaxRepeatBins)
                    continue;
                for(int horizonIdx = 0; horizonIdx < kNumHorizons; ++horizonIdx)
                {
                    const int horizon = kHorizons[horizonIdx];
                    if(age < static_cast<long unsigned int>(horizon))
                        continue;
                    CohortStats& bucket = stats[dim][bin][depthBin][horizonIdx];
                    ++bucket.eligible;
                    bucket.obsSum += observations;
                    if(std::isfinite(foundRatio))
                    {
                        bucket.foundRatioSum += foundRatio;
                        ++bucket.foundRatioValid;
                    }
                    if(bStable)
                        ++bucket.stable;
                    if(OccurredWithin(reuse.firstProjectionAttemptFrame, horizon))
                        ++bucket.attempted;
                    if(OccurredWithin(reuse.firstProjectionMatchFrame, horizon))
                        ++bucket.matched;
                    if(OccurredWithin(reuse.firstOptInlierFrame, horizon))
                        ++bucket.optInlier;
                    if(OccurredWithin(reuse.firstOptOutlierFrame, horizon))
                        ++bucket.optOutlier;
                }
            }
        }

        auto DimensionName = [](const int dim)
        {
            switch(dim)
            {
                case kRepeatGrid: return "grid_bin";
                case kRepeatOctave: return "octave_bin";
                case kRepeatResponse: return "response_bin";
                case kRepeatMappingFeature: return "mapping_feature_bin";
                default: return "unknown";
            }
        };
        auto BinName = [](const int dim, const int bin)
        {
            switch(dim)
            {
                case kRepeatGrid: return MappingFeatureGridBinName(bin);
                case kRepeatOctave: return MappingFeatureOctaveBinName(bin);
                case kRepeatResponse: return MappingFeatureResponseBinName(bin);
                case kRepeatMappingFeature: return MappingFeatureCombinedBinName(bin);
                default: return "unknown";
            }
        };

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6);
        for(int dim = 0; dim < kRepeatDimensionCount; ++dim)
        {
            for(int bin = 0; bin < kMaxRepeatBins; ++bin)
            {
                for(int depthBin = 1; depthBin <= 2; ++depthBin)
                {
                    for(int horizonIdx = 0; horizonIdx < kNumHorizons; ++horizonIdx)
                    {
                        const CohortStats& bucket = stats[dim][bin][depthBin][horizonIdx];
                        if(bucket.eligible <= 0)
                            continue;
                        const float invEligible = 1.0f / static_cast<float>(bucket.eligible);
                        const int outcomes = bucket.optInlier + bucket.optOutlier;
                        std::cout << "[TrackLocalMap][MappingRepeatabilityCohort]"
                                  << " frame_id=" << F.mnId
                                  << " source=local_mapping_triangulation"
                                  << " depth_bin=" << DepthBinName(depthBin)
                                  << " dimension=" << DimensionName(dim)
                                  << " bin=" << BinName(dim, bin)
                                  << " horizon_frames=" << kHorizons[horizonIdx]
                                  << " eligible=" << bucket.eligible
                                  << " attempted=" << bucket.attempted
                                  << " matched=" << bucket.matched
                                  << " opt_inlier=" << bucket.optInlier
                                  << " opt_outlier=" << bucket.optOutlier
                                  << " stable=" << bucket.stable
                                  << " attempt_rate=" << static_cast<float>(bucket.attempted) * invEligible
                                  << " match_rate=" << static_cast<float>(bucket.matched) * invEligible
                                  << " inlier_rate=" << static_cast<float>(bucket.optInlier) * invEligible
                                  << " outlier_rate=" << static_cast<float>(bucket.optOutlier) * invEligible
                                  << " stable_rate=" << static_cast<float>(bucket.stable) * invEligible
                                  << " opt_outlier_ratio="
                                  << (outcomes > 0
                                      ? static_cast<float>(bucket.optOutlier) / static_cast<float>(outcomes)
                                      : 0.0f)
                                  << " obs_avg=" << static_cast<float>(bucket.obsSum) * invEligible
                                  << " found_ratio_avg="
                                  << (bucket.foundRatioValid > 0
                                      ? bucket.foundRatioSum / static_cast<float>(bucket.foundRatioValid)
                                      : 0.0f)
                                  << std::endl;
                    }
                }
            }
        }
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    void LogTrackLocalMapInlierMix(const Frame& F)
    {
        int totalInliers = 0;
        int motionLgInliers = 0;
        int localProjTrackingStereoInliers = 0;
        int localProjTriangulationInliers = 0;

        const size_t n = std::min(F.mvpMapPoints.size(), F.mvbOutlier.size());
        for(size_t i = 0; i < n; ++i)
        {
            MapPoint* pMP = F.mvpMapPoints[i];
            if(!pMP || F.mvbOutlier[i])
                continue;

            totalInliers++;
            unsigned char matchSource = kXFeatMatchUnknown;
            if(i < F.mvXFeatMatchSource.size())
                matchSource = F.mvXFeatMatchSource[i];

            if(matchSource == kXFeatMatchMotionLightGlue)
                motionLgInliers++;

            if(matchSource == kXFeatMatchLocalProjection)
            {
                const MapPoint::CreationSource mpSource = pMP->GetCreationSource();
                if(mpSource == MapPoint::kCreationSourceTrackingStereo)
                    localProjTrackingStereoInliers++;
                else if(mpSource == MapPoint::kCreationSourceLocalMappingTriangulation)
                    localProjTriangulationInliers++;
            }
        }

        std::cout << "[TrackLocalMap][InlierMix]"
                  << " frame_id=" << F.mnId
                  << " total_inliers=" << totalInliers
                  << " motion_lg=" << motionLgInliers
                  << " local_proj_tracking_stereo=" << localProjTrackingStereoInliers
                  << " local_proj_local_mapping_triangulation=" << localProjTriangulationInliers
                  << std::endl;
    }

    struct InlierGeometryStats
    {
        std::vector<int> indices;
        DepthDistributionStats depth;
        int depthBins[6] = {};
        int left = 0;
        int right = 0;
        int stereoObs = 0;
        int lowObs = 0;
        int ageLe5 = 0;
        int foundValid = 0;
        double obsSum = 0.0;
        double ageSum = 0.0;
        double foundSum = 0.0;
    };

    void AccumulateInlierGeometryStats(InlierGeometryStats& stats,
                                       const Frame& F,
                                       const int idx,
                                       MapPoint* pMP)
    {
        if(!pMP || idx < 0 || idx >= F.N)
            return;

        stats.indices.push_back(idx);

        if(F.Nleft >= 0 && idx >= F.Nleft)
            stats.right++;
        else
            stats.left++;

        if(idx < static_cast<int>(F.mvuRight.size()) && F.mvuRight[idx] >= 0.0f)
            stats.stereoObs++;

        const int obs = pMP->Observations();
        stats.obsSum += obs;
        if(obs <= 2)
            stats.lowObs++;

        if(pMP->mnFirstFrame >= 0 && F.mnId >= static_cast<unsigned long>(pMP->mnFirstFrame))
        {
            const double age = static_cast<double>(F.mnId - static_cast<unsigned long>(pMP->mnFirstFrame));
            stats.ageSum += age;
            if(age <= 5.0)
                stats.ageLe5++;
        }

        const float foundRatio = pMP->GetFoundRatio();
        if(std::isfinite(foundRatio))
        {
            stats.foundValid++;
            stats.foundSum += foundRatio;
        }

        const float depth = ResolveMapPointQualityDepth(F, pMP, pMP->mTrackDepth);
        AccumulateDepthDistribution(stats.depth, depth);
        const int depthBin = DepthBinIndex(depth);
        if(depthBin >= 0 && depthBin < 6)
            stats.depthBins[depthBin]++;
    }

    void PrintInlierGeometryStats(const Frame& F,
                                  const char* group,
                                  const InlierGeometryStats& stats)
    {
        const int total = static_cast<int>(stats.indices.size());
        if(total <= 0)
            return;

        FeatureSpatialStats spatial = ComputeFeatureSpatialStats(F, stats.indices);
        const std::vector<int> octaveHist = BuildOctaveHistogram(F, stats.indices);

        std::vector<float> depthSamples = stats.depth.samples;
        const float depthP50 = depthSamples.empty() ? 0.0f : DepthPercentile(depthSamples, 0.50f);
        const float depthP90 = depthSamples.empty() ? 0.0f : DepthPercentile(depthSamples, 0.90f);
        const double depthAvg = stats.depth.valid > 0
            ? stats.depth.sum / static_cast<double>(stats.depth.valid)
            : 0.0;
        const double obsAvg = total > 0 ? stats.obsSum / static_cast<double>(total) : 0.0;
        const double ageAvg = total > 0 ? stats.ageSum / static_cast<double>(total) : 0.0;
        const double foundAvg = stats.foundValid > 0
            ? stats.foundSum / static_cast<double>(stats.foundValid)
            : 0.0;

        std::cout << "[TrackLocalMap][InlierGeometry]"
                  << " frame_id=" << F.mnId
                  << " group=" << group
                  << " total=" << total
                  << " spatial_valid=" << spatial.validPoints
                  << " spatial_non_empty=" << spatial.nonEmptyCells
                  << " spatial_cells=" << spatial.totalCells
                  << " spatial_top1=" << spatial.top1Share
                  << " spatial_top4=" << spatial.top4Share
                  << " spatial_cv=" << spatial.coeffVar
                  << " octave=" << FormatHistogram(octaveHist)
                  << " left=" << stats.left
                  << " right=" << stats.right
                  << " stereo_obs=" << stats.stereoObs
                  << " depth_valid=" << stats.depth.valid
                  << " depth_invalid=" << stats.depth.invalid
                  << " depth_0_5=" << stats.depthBins[1]
                  << " depth_5_10=" << stats.depthBins[2]
                  << " depth_10_30=" << stats.depthBins[3]
                  << " depth_30_100=" << stats.depthBins[4]
                  << " depth_100p=" << stats.depthBins[5]
                  << " depth_avg=" << depthAvg
                  << " depth_p50=" << depthP50
                  << " depth_p90=" << depthP90
                  << " mp_obs_avg=" << obsAvg
                  << " mp_low_obs=" << stats.lowObs
                  << " mp_found_avg=" << foundAvg
                  << " mp_age_avg=" << ageAvg
                  << " mp_age_le5=" << stats.ageLe5
                  << std::endl;
    }

    void LogTrackLocalMapInlierGeometryDiagnostics(const Frame& F)
    {
        if(!IsXFeatFeatureDiagEnabled())
            return;

        InlierGeometryStats allInliers;
        InlierGeometryStats motionLg;
        InlierGeometryStats localProj;
        InlierGeometryStats localProjTriang;
        InlierGeometryStats localProjTrackingStereo;
        InlierGeometryStats localProjTriangEdge;

        const size_t n = std::min(F.mvpMapPoints.size(), F.mvbOutlier.size());
        for(size_t i = 0; i < n; ++i)
        {
            MapPoint* pMP = F.mvpMapPoints[i];
            if(!pMP || F.mvbOutlier[i])
                continue;

            unsigned char matchSource = kXFeatMatchUnknown;
            if(i < F.mvXFeatMatchSource.size())
                matchSource = F.mvXFeatMatchSource[i];
            const MapPoint::CreationSource mpSource = pMP->GetCreationSource();
            const int idx = static_cast<int>(i);

            AccumulateInlierGeometryStats(allInliers, F, idx, pMP);

            if(matchSource == kXFeatMatchMotionLightGlue)
                AccumulateInlierGeometryStats(motionLg, F, idx, pMP);

            if(matchSource == kXFeatMatchLocalProjection)
            {
                AccumulateInlierGeometryStats(localProj, F, idx, pMP);
                if(mpSource == MapPoint::kCreationSourceLocalMappingTriangulation)
                {
                    AccumulateInlierGeometryStats(localProjTriang, F, idx, pMP);
                    const float projErrNorm = i < F.mvLocalProjProjErrNorm.size()
                        ? F.mvLocalProjProjErrNorm[i]
                        : std::numeric_limits<float>::quiet_NaN();
                    if(std::isfinite(projErrNorm) && projErrNorm > 0.5f && projErrNorm <= 1.0f)
                        AccumulateInlierGeometryStats(localProjTriangEdge, F, idx, pMP);
                }
                else if(mpSource == MapPoint::kCreationSourceTrackingStereo)
                {
                    AccumulateInlierGeometryStats(localProjTrackingStereo, F, idx, pMP);
                }
            }
        }

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6);
        PrintInlierGeometryStats(F, "all_inliers", allInliers);
        PrintInlierGeometryStats(F, "motion_lg", motionLg);
        PrintInlierGeometryStats(F, "local_proj", localProj);
        PrintInlierGeometryStats(F, "local_proj_lmt", localProjTriang);
        PrintInlierGeometryStats(F, "local_proj_tracking_stereo", localProjTrackingStereo);
        PrintInlierGeometryStats(F, "local_proj_lmt_edge", localProjTriangEdge);
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    struct ConstraintGridStats
    {
        InlierGeometryStats inlier;
        InlierGeometryStats outlier;
    };

    void AccumulateConstraintGridStats(ConstraintGridStats& stats,
                                       const Frame& F,
                                       const int idx,
                                       MapPoint* pMP)
    {
        if(!pMP || idx < 0 || idx >= static_cast<int>(F.mvbOutlier.size()))
            return;

        if(F.mvbOutlier[static_cast<size_t>(idx)])
            AccumulateInlierGeometryStats(stats.outlier, F, idx, pMP);
        else
            AccumulateInlierGeometryStats(stats.inlier, F, idx, pMP);
    }

    void PrintConstraintGridStats(const Frame& F,
                                  const char* group,
                                  const ConstraintGridStats& stats)
    {
        const int inlier = static_cast<int>(stats.inlier.indices.size());
        const int outlier = static_cast<int>(stats.outlier.indices.size());
        const int total = inlier + outlier;
        if(total <= 0)
            return;

        const FeatureSpatialStats inlierSpatial = ComputeFeatureSpatialStats(F, stats.inlier.indices);
        const FeatureSpatialStats outlierSpatial = ComputeFeatureSpatialStats(F, stats.outlier.indices);
        const double outlierRatio = static_cast<double>(outlier) / static_cast<double>(total);
        const double inlierDepthAvg = stats.inlier.depth.valid > 0
            ? stats.inlier.depth.sum / static_cast<double>(stats.inlier.depth.valid)
            : 0.0;
        const double outlierDepthAvg = stats.outlier.depth.valid > 0
            ? stats.outlier.depth.sum / static_cast<double>(stats.outlier.depth.valid)
            : 0.0;

        std::cout << "[TrackLocalMap][ConstraintGrid]"
                  << " frame_id=" << F.mnId
                  << " group=" << group
                  << " total=" << total
                  << " inlier=" << inlier
                  << " outlier=" << outlier
                  << " outlier_ratio=" << outlierRatio
                  << " inlier_non_empty=" << inlierSpatial.nonEmptyCells
                  << " inlier_top1=" << inlierSpatial.top1Share
                  << " inlier_top4=" << inlierSpatial.top4Share
                  << " inlier_cv=" << inlierSpatial.coeffVar
                  << " outlier_non_empty=" << outlierSpatial.nonEmptyCells
                  << " outlier_top1=" << outlierSpatial.top1Share
                  << " outlier_top4=" << outlierSpatial.top4Share
                  << " outlier_cv=" << outlierSpatial.coeffVar
                  << " inlier_depth_avg=" << inlierDepthAvg
                  << " outlier_depth_avg=" << outlierDepthAvg
                  << " inlier_depth_0_5=" << stats.inlier.depthBins[1]
                  << " inlier_depth_5_10=" << stats.inlier.depthBins[2]
                  << " outlier_depth_0_5=" << stats.outlier.depthBins[1]
                  << " outlier_depth_5_10=" << stats.outlier.depthBins[2]
                  << std::endl;
    }

    void LogTrackLocalMapConstraintGridDiagnostics(const Frame& F)
    {
        if(!IsXFeatFeatureDiagEnabled())
            return;

        ConstraintGridStats allConstraints;
        ConstraintGridStats motionLg;
        ConstraintGridStats localProj;
        ConstraintGridStats localProjTriang;
        ConstraintGridStats localProjTriang0To10;
        ConstraintGridStats localProjTrackingStereo0To10;
        ConstraintGridStats localProjTriangEdge0To10;

        const size_t n = std::min(F.mvpMapPoints.size(), F.mvbOutlier.size());
        for(size_t i = 0; i < n; ++i)
        {
            MapPoint* pMP = F.mvpMapPoints[i];
            if(!pMP)
                continue;

            unsigned char matchSource = kXFeatMatchUnknown;
            if(i < F.mvXFeatMatchSource.size())
                matchSource = F.mvXFeatMatchSource[i];

            const int idx = static_cast<int>(i);
            const MapPoint::CreationSource mpSource = pMP->GetCreationSource();
            const float depth = ResolveMapPointQualityDepth(F, pMP, pMP->mTrackDepth);
            const bool depth0To10 = std::isfinite(depth) && depth > 0.0f && depth <= 10.0f;

            AccumulateConstraintGridStats(allConstraints, F, idx, pMP);

            if(matchSource == kXFeatMatchMotionLightGlue)
                AccumulateConstraintGridStats(motionLg, F, idx, pMP);

            if(matchSource != kXFeatMatchLocalProjection)
                continue;

            AccumulateConstraintGridStats(localProj, F, idx, pMP);

            if(mpSource == MapPoint::kCreationSourceLocalMappingTriangulation)
            {
                AccumulateConstraintGridStats(localProjTriang, F, idx, pMP);
                if(depth0To10)
                {
                    AccumulateConstraintGridStats(localProjTriang0To10, F, idx, pMP);
                    const float projErrNorm = i < F.mvLocalProjProjErrNorm.size()
                        ? F.mvLocalProjProjErrNorm[i]
                        : std::numeric_limits<float>::quiet_NaN();
                    if(std::isfinite(projErrNorm) && projErrNorm > 0.5f && projErrNorm <= 1.0f)
                        AccumulateConstraintGridStats(localProjTriangEdge0To10, F, idx, pMP);
                }
            }
            else if(mpSource == MapPoint::kCreationSourceTrackingStereo && depth0To10)
            {
                AccumulateConstraintGridStats(localProjTrackingStereo0To10, F, idx, pMP);
            }
        }

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6);
        PrintConstraintGridStats(F, "all_constraints", allConstraints);
        PrintConstraintGridStats(F, "motion_lg", motionLg);
        PrintConstraintGridStats(F, "local_proj", localProj);
        PrintConstraintGridStats(F, "local_proj_lmt", localProjTriang);
        PrintConstraintGridStats(F, "local_proj_lmt_0_10", localProjTriang0To10);
        PrintConstraintGridStats(F, "local_proj_tracking_stereo_0_10", localProjTrackingStereo0To10);
        PrintConstraintGridStats(F, "local_proj_lmt_edge_0_10", localProjTriangEdge0To10);
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    bool IsXFeatLightGlueLocalMapEnabled()
    {
        return IsEnvFlagEnabled("XFEAT_USE_LIGHTGLUE_LOCALMAP");
    }

    int GetXFeatLightGlueLocalMapMaxKFs()
    {
        return 3;
    }

    float GetXFeatLightGlueLocalMapProjRadius()
    {
        return 12.0f;
    }

    float GetXFeatLightGlueLocalConflictMinScore()
    {
        return 0.20f;
    }

    bool UseXFeatLightGlueMotionProjectionFirst()
    {
        const char* env = std::getenv("XFEAT_LG_MOTION_POLICY");
        if(!env || env[0] == '\0')
            return false;

        std::string policy(env);
        std::transform(policy.begin(), policy.end(), policy.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return policy == "projection_first" || policy == "proj_first";
    }

    int GetXFeatLightGlueMotionFallbackMinMatches()
    {
        return GetEnvIntInRange("XFEAT_LG_MOTION_FALLBACK_MIN_MATCHES", 20, 1, 1000);
    }

    float GetXFeatThHighRefNNStrict()
    {
        //调试: TrackReferenceKeyFrame 严格NN阈值（默认偏严格，优先精度）。
        return GetEnvFloatInRange("XFEAT_TH_HIGH_REF_NN_STRICT", 1.35f, 0.05f, 2.0f);
    }

    float GetXFeatThHighRefNNRelaxed()
    {
        //调试: TrackReferenceKeyFrame 宽松NN阈值（仅在严格匹配不足时触发）。
        return GetEnvFloatInRange("XFEAT_TH_HIGH_REF_NN_RELAXED", 1.60f, 0.05f, 2.0f);
    }

    float GetXFeatThHighRefProjectionFallback()
    {
        //调试: 参考关键帧投影兜底阈值，过大易引入低纯度补匹配。
        return GetEnvFloatInRange("XFEAT_TH_HIGH_REF_PROJ", 1.40f, 0.05f, 2.0f);
    }

    float GetXFeatThHighMotionProjection()
    {
        //调试: 运动模型投影阈值，控制 LastFrame->CurrentFrame 匹配纯度。
        return GetEnvFloatInRange("XFEAT_TH_HIGH_MOTION_PROJ", 1.45f, 0.05f, 2.0f);
    }

    float GetXFeatThHighLocalProjection()
    {
        //调试: LocalMap 投影阈值。默认保持宽松，优先保留足够的局部地图约束。
        return GetEnvFloatInRange("XFEAT_TH_HIGH_LOCAL_PROJ", 1.35f, 0.05f, 2.0f);
    }

    float GetXFeatThHighRelocNN()
    {
        //调试: 重定位初配NN阈值，默认保留更高召回以避免候选全灭。
        return GetEnvFloatInRange("XFEAT_TH_HIGH_RELOC_NN", 1.75f, 0.05f, 2.0f);
    }

    float GetXFeatThHighRelocProjCoarse()
    {
        //调试: 重定位粗窗口投影补匹配阈值（窗口大，阈值可略宽松）。
        return GetEnvFloatInRange("XFEAT_TH_HIGH_RELOC_PROJ_COARSE", 1.80f, 0.05f, 2.0f);
    }

    float GetXFeatThHighRelocProjFine()
    {
        //调试: 重定位细窗口投影补匹配阈值（窗口小，阈值应更严格）。
        return GetEnvFloatInRange("XFEAT_TH_HIGH_RELOC_PROJ_FINE", 1.60f, 0.05f, 2.0f);
    }

    bool IsXFeatLightGlueRelocEnabled()
    {
        return IsEnvFlagEnabled("XFEAT_USE_LIGHTGLUE_RELOC");
    }

    int GetXFeatLightGlueRelocFallbackMinMatches()
    {
        return GetEnvIntInRange("XFEAT_LG_RELOC_FALLBACK_MIN_MATCHES", 15, 1, 1000);
    }

    int GetXFeatRecentRelocInlierThreshold()
    {
        // XFeat 路径在重定位后前几帧通常内点偏低，默认放宽到 35。
        // 可通过 `XFEAT_RELOC_INLIER_TH` 覆盖。
        const char* env = std::getenv("XFEAT_RELOC_INLIER_TH");
        if(env)
        {
            try
            {
                const int v = std::stoi(std::string(env));
                if(v > 0)
                    return v;
            }
            catch(...)
            {
            }
        }
        return 35;
    }

    float GetXFeatInitMinBaselineDepthRatio()
    {
        // XFeat 在单目初始化中容易出现“高匹配但低平移基线”的退化解。
        // 通过 baseline/medianDepth 下限做保护，可通过环境变量覆盖。
        const char* env = std::getenv("XFEAT_INIT_MIN_BASELINE_DEPTH_RATIO");
        if(env)
        {
            try
            {
                const float v = std::stof(std::string(env));
                if(v > 0.0f)
                    return v;
            }
            catch(...)
            {
            }
        }
        return 0.01f;
    }

    bool IsXFeatMonoNearOnlyEnabled()
    {
        // Enable by setting XFEAT_MONO_NEAR_ONLY=1.
        static const bool enabled = IsEnvFlagEnabled("XFEAT_MONO_NEAR_ONLY");
        return enabled;
    }

    float GetXFeatMonoNearDepthFactor()
    {
        // If near-only mode is enabled, keep local points with depth <= factor * medianDepth.
        // Can be overridden by XFEAT_MONO_NEAR_DEPTH_FACTOR.
        const char* env = std::getenv("XFEAT_MONO_NEAR_DEPTH_FACTOR");
        if(env)
        {
            try
            {
                const float v = std::stof(std::string(env));
                if(v > 0.0f)
                    return v;
            }
            catch(...)
            {
            }
        }
        return 2.5f;
    }

    int GetXFeatInitMaxFrameSpan()
    {
        // XFeat结构化初始化策略:
        // 若同一参考帧持续过久仍未重建成功，主动重锚参考帧，避免“高匹配但旧参考退化”。
        return 35;
    }

    int GetXFeatInitReanchorMatchFloor()
    {
        // XFeat结构化初始化策略:
        // 当匹配数已经明显衰减（但仍可能 >100）且重建失败时，提前重锚参考帧。
        return 160;
    }

    int CountValidInitMatches(const std::vector<int>& matches12)
    {
        int count = 0;
        for(const int idx2 : matches12)
        {
            if(idx2 >= 0)
                ++count;
        }
        return count;
    }

    struct XFeatInitMatchCandidate
    {
        int idx1 = -1;
        int idx2 = -1;
        float disp = 0.0f;
        int cell = 0;
    };

    int PruneXFeatInitMatchesForReconstruction(const Frame& F1,
                                               const Frame& F2,
                                               std::vector<int>& matches12,
                                               float& medianDisp,
                                               float& minDispGate,
                                               int& dispFiltered,
                                               bool& dispGateRelaxed)
    {
        //调试: XFeat初始化重建前匹配裁剪，降低TwoView在超大匹配数下的退化概率。
        const int kMaxRetainedMatches = 260; // 这个地方对初始化成功率和重建质量有较大影响，过少可能导致重建失败，过多可能导致退化解。160-240是经验范围。
        const int kMinRetainedMatches = 100;  // 更稳初始化：kMax=220, kMin=100 更保守保信息：kMax=260, kMin=100
        const float kBaseMinDisp = 2.0f;
        const int kGridCols = 10;
        const int kGridRows = 8;

        medianDisp = 0.0f;
        minDispGate = 0.0f;
        dispFiltered = 0;
        dispGateRelaxed = false;

        const int totalBefore = CountValidInitMatches(matches12);
        if(totalBefore < kMinRetainedMatches)
            return 0;

        const float minX = Frame::mnMinX;
        const float maxX = Frame::mnMaxX;
        const float minY = Frame::mnMinY;
        const float maxY = Frame::mnMaxY;
        const float spanX = std::max(1.0f, maxX - minX);
        const float spanY = std::max(1.0f, maxY - minY);

        std::vector<XFeatInitMatchCandidate> candidates;
        candidates.reserve(static_cast<size_t>(totalBefore));
        std::vector<float> disps;
        disps.reserve(static_cast<size_t>(totalBefore));

        for(size_t i1 = 0; i1 < matches12.size(); ++i1)
        {
            const int idx2 = matches12[i1];
            if(idx2 < 0)
                continue;
            if(i1 >= F1.mvKeysUn.size() || idx2 >= static_cast<int>(F2.mvKeysUn.size()))
                continue;

            const cv::Point2f p1 = F1.mvKeysUn[i1].pt;
            const cv::Point2f p2 = F2.mvKeysUn[idx2].pt;
            const float dx = p2.x - p1.x;
            const float dy = p2.y - p1.y;
            const float disp = std::sqrt(dx * dx + dy * dy);

            int cellX = static_cast<int>((p1.x - minX) / spanX * static_cast<float>(kGridCols));
            int cellY = static_cast<int>((p1.y - minY) / spanY * static_cast<float>(kGridRows));
            cellX = std::max(0, std::min(kGridCols - 1, cellX));
            cellY = std::max(0, std::min(kGridRows - 1, cellY));

            candidates.push_back({static_cast<int>(i1), idx2, disp, cellY * kGridCols + cellX});
            disps.push_back(disp);
        }

        if(candidates.size() < static_cast<size_t>(kMinRetainedMatches))
            return 0;

        const size_t mid = disps.size() / 2;
        std::nth_element(disps.begin(), disps.begin() + mid, disps.end());
        medianDisp = disps[mid];
        minDispGate = std::max(kBaseMinDisp, std::min(5.0f, 0.05f * medianDisp));

        std::vector<XFeatInitMatchCandidate> pool;
        pool.reserve(candidates.size());
        for(const XFeatInitMatchCandidate& c : candidates)
        {
            if(c.disp >= minDispGate)
                pool.push_back(c);
        }
        dispFiltered = static_cast<int>(candidates.size()) - static_cast<int>(pool.size());

        if(pool.size() < static_cast<size_t>(kMinRetainedMatches))
        {
            //调试: 位移门控过严时回退，避免初始化匹配不足。
            pool = candidates;
            dispGateRelaxed = true;
        }

        std::sort(pool.begin(), pool.end(),
                  [](const XFeatInitMatchCandidate& a, const XFeatInitMatchCandidate& b)
                  {
                      return a.disp > b.disp;
                  });

        const int targetKeep = std::min(kMaxRetainedMatches, static_cast<int>(pool.size()));
        const int totalCells = kGridCols * kGridRows;
        const int perCellCap = std::max(2, (targetKeep + totalCells - 1) / totalCells);
        std::vector<int> cellUsed(totalCells, 0);

        std::vector<XFeatInitMatchCandidate> selected;
        selected.reserve(static_cast<size_t>(targetKeep));
        std::vector<char> selectedMask(pool.size(), 0);

        for(size_t i = 0; i < pool.size() && static_cast<int>(selected.size()) < targetKeep; ++i)
        {
            const XFeatInitMatchCandidate& c = pool[i];
            if(cellUsed[c.cell] >= perCellCap)
                continue;
            selected.push_back(c);
            selectedMask[i] = 1;
            ++cellUsed[c.cell];
        }

        for(size_t i = 0; i < pool.size() && static_cast<int>(selected.size()) < targetKeep; ++i)
        {
            if(selectedMask[i] != 0)
                continue;
            selected.push_back(pool[i]);
        }

        std::fill(matches12.begin(), matches12.end(), -1);
        for(const XFeatInitMatchCandidate& c : selected)
            matches12[c.idx1] = c.idx2;

        return std::max(0, totalBefore - static_cast<int>(selected.size()));
    }
}


Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Atlas *pAtlas, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, Settings* settings, const string &_nameSeq):
    mState(NO_IMAGES_YET), mSensor(sensor), mTrackedFr(0), mbStep(false),
    mbOnlyTracking(false), mbMapUpdated(false), mbVO(false), mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB),
    mbReadyToInitializate(false), mpSystem(pSys), mpViewer(NULL), bStepByStep(false),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpAtlas(pAtlas), mnLastRelocFrameId(0), time_recently_lost(5.0),
    mnInitialFrameId(0), mbCreatedMap(false), mnFirstFrameId(0), mpCamera2(nullptr), mpLastKeyFrame(static_cast<KeyFrame*>(NULL))
{
    mpORBextractorLeft = static_cast<ORBextractor*>(NULL);
    mpORBextractorRight = static_cast<ORBextractor*>(NULL);
    mpIniORBextractor = static_cast<ORBextractor*>(NULL);
    mpXFextractor = static_cast<XFextractor*>(NULL);

    // Load camera parameters from settings file
    if(settings){
        newParameterLoader(settings);
    }
    else{
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        bool b_parse_cam = ParseCamParamFile(fSettings);
        if(!b_parse_cam)
        {
            std::cout << "*Error with the camera parameters in the config file*" << std::endl;
        }

        // Load ORB parameters
        bool b_parse_orb = ParseORBParamFile(fSettings);
        if(!b_parse_orb)
        {
            std::cout << "*Error with the ORB parameters in the config file*" << std::endl;
        }

        bool b_parse_imu = true;
        if(sensor==System::IMU_MONOCULAR || sensor==System::IMU_STEREO || sensor==System::IMU_RGBD)
        {
            b_parse_imu = ParseIMUParamFile(fSettings);
            if(!b_parse_imu)
            {
                std::cout << "*Error with the IMU parameters in the config file*" << std::endl;
            }

            mnFramesToResetIMU = mMaxFrames;
        }

        if(!b_parse_cam || !b_parse_orb || !b_parse_imu)
        {
            std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
            try
            {
                throw -1;
            }
            catch(exception &e)
            {

            }
        }
    }

    initID = 0; lastID = 0;
    mbInitWith3KFs = false;
    mnNumDataset = 0;

    vector<GeometricCamera*> vpCams = mpAtlas->GetAllCameras();
    std::cout << "There are " << vpCams.size() << " cameras in the atlas" << std::endl;
    for(GeometricCamera* pCam : vpCams)
    {
        std::cout << "Camera " << pCam->GetId();
        if(pCam->GetType() == GeometricCamera::CAM_PINHOLE)
        {
            std::cout << " is pinhole" << std::endl;
        }
        else if(pCam->GetType() == GeometricCamera::CAM_FISHEYE)
        {
            std::cout << " is fisheye" << std::endl;
        }
        else
        {
            std::cout << " is unknown" << std::endl;
        }
    }

#ifdef REGISTER_TIMES
    vdRectStereo_ms.clear();
    vdResizeImage_ms.clear();
    vdORBExtract_ms.clear();
    vdStereoMatch_ms.clear();
    vdIMUInteg_ms.clear();
    vdPosePred_ms.clear();
    vdLMTrack_ms.clear();
    vdNewKF_ms.clear();
    vdTrackTotal_ms.clear();
#endif
}

#ifdef REGISTER_TIMES
double calcAverage(vector<double> v_times)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += value;
    }

    return accum / v_times.size();
}

double calcDeviation(vector<double> v_times, double average)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += pow(value - average, 2);
    }
    return sqrt(accum / v_times.size());
}

double calcAverage(vector<int> v_values)
{
    double accum = 0;
    int total = 0;
    for(double value : v_values)
    {
        if(value == 0)
            continue;
        accum += value;
        total++;
    }

    return accum / total;
}

double calcDeviation(vector<int> v_values, double average)
{
    double accum = 0;
    int total = 0;
    for(double value : v_values)
    {
        if(value == 0)
            continue;
        accum += pow(value - average, 2);
        total++;
    }
    return sqrt(accum / total);
}

void Tracking::LocalMapStats2File()
{
    ofstream f;
    f.open("LocalMapTimeStats.txt");
    f << fixed << setprecision(6);
    f << "#Stereo rect[ms], MP culling[ms], MP creation[ms], LBA[ms], KF culling[ms], Total[ms]" << endl;
    for(int i=0; i<mpLocalMapper->vdLMTotal_ms.size(); ++i)
    {
        f << mpLocalMapper->vdKFInsert_ms[i] << "," << mpLocalMapper->vdMPCulling_ms[i] << ","
          << mpLocalMapper->vdMPCreation_ms[i] << "," << mpLocalMapper->vdLBASync_ms[i] << ","
          << mpLocalMapper->vdKFCullingSync_ms[i] <<  "," << mpLocalMapper->vdLMTotal_ms[i] << endl;
    }

    f.close();

    f.open("LBA_Stats.txt");
    f << fixed << setprecision(6);
    f << "#LBA time[ms], KF opt[#], KF fixed[#], MP[#], Edges[#]" << endl;
    for(int i=0; i<mpLocalMapper->vdLBASync_ms.size(); ++i)
    {
        f << mpLocalMapper->vdLBASync_ms[i] << "," << mpLocalMapper->vnLBA_KFopt[i] << ","
          << mpLocalMapper->vnLBA_KFfixed[i] << "," << mpLocalMapper->vnLBA_MPs[i] << ","
          << mpLocalMapper->vnLBA_edges[i] << endl;
    }


    f.close();
}

void Tracking::TrackStats2File()
{
    ofstream f;
    f.open("SessionInfo.txt");
    f << fixed;
    f << "Number of KFs: " << mpAtlas->GetAllKeyFrames().size() << endl;
    f << "Number of MPs: " << mpAtlas->GetAllMapPoints().size() << endl;

    f << "OpenCV version: " << CV_VERSION << endl;

    f.close();

    f.open("TrackingTimeStats.txt");
    f << fixed << setprecision(6);

    f << "#Image Rect[ms], Image Resize[ms], ORB ext[ms], Stereo match[ms], IMU preint[ms], Pose pred[ms], LM track[ms], KF dec[ms], Total[ms]" << endl;

    for(int i=0; i<vdTrackTotal_ms.size(); ++i)
    {
        double stereo_rect = 0.0;
        if(!vdRectStereo_ms.empty())
        {
            stereo_rect = vdRectStereo_ms[i];
        }

        double resize_image = 0.0;
        if(!vdResizeImage_ms.empty())
        {
            resize_image = vdResizeImage_ms[i];
        }

        double stereo_match = 0.0;
        if(!vdStereoMatch_ms.empty())
        {
            stereo_match = vdStereoMatch_ms[i];
        }

        double imu_preint = 0.0;
        if(!vdIMUInteg_ms.empty())
        {
            imu_preint = vdIMUInteg_ms[i];
        }

        f << stereo_rect << "," << resize_image << "," << vdORBExtract_ms[i] << "," << stereo_match << "," << imu_preint << ","
          << vdPosePred_ms[i] <<  "," << vdLMTrack_ms[i] << "," << vdNewKF_ms[i] << "," << vdTrackTotal_ms[i] << endl;
    }

    f.close();
}

void Tracking::PrintTimeStats()
{
    // Save data in files
    TrackStats2File();
    LocalMapStats2File();


    ofstream f;
    f.open("ExecMean.txt");
    f << fixed;
    //Report the mean and std of each one
    std::cout << std::endl << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    f << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    cout << "OpenCV version: " << CV_VERSION << endl;
    f << "OpenCV version: " << CV_VERSION << endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    f << "---------------------------" << std::endl;
    f << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    double average, deviation;
    if(!vdRectStereo_ms.empty())
    {
        average = calcAverage(vdRectStereo_ms);
        deviation = calcDeviation(vdRectStereo_ms, average);
        std::cout << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
        f << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
    }

    if(!vdResizeImage_ms.empty())
    {
        average = calcAverage(vdResizeImage_ms);
        deviation = calcDeviation(vdResizeImage_ms, average);
        std::cout << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
        f << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdORBExtract_ms);
    deviation = calcDeviation(vdORBExtract_ms, average);
    std::cout << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;
    f << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;

    if(!vdStereoMatch_ms.empty())
    {
        average = calcAverage(vdStereoMatch_ms);
        deviation = calcDeviation(vdStereoMatch_ms, average);
        std::cout << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
        f << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
    }

    if(!vdIMUInteg_ms.empty())
    {
        average = calcAverage(vdIMUInteg_ms);
        deviation = calcDeviation(vdIMUInteg_ms, average);
        std::cout << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
        f << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdPosePred_ms);
    deviation = calcDeviation(vdPosePred_ms, average);
    std::cout << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;
    f << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdLMTrack_ms);
    deviation = calcDeviation(vdLMTrack_ms, average);
    std::cout << "LM Track: " << average << "$\\pm$" << deviation << std::endl;
    f << "LM Track: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdNewKF_ms);
    deviation = calcDeviation(vdNewKF_ms, average);
    std::cout << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;
    f << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdTrackTotal_ms);
    deviation = calcDeviation(vdTrackTotal_ms, average);
    std::cout << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping time stats
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "Local Mapping" << std::endl << std::endl;
    f << std::endl << "Local Mapping" << std::endl << std::endl;

    average = calcAverage(mpLocalMapper->vdKFInsert_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFInsert_ms, average);
    std::cout << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;
    f << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCulling_ms, average);
    std::cout << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;
    f << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCreation_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCreation_ms, average);
    std::cout << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;
    f << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLBA_ms);
    deviation = calcDeviation(mpLocalMapper->vdLBA_ms, average);
    std::cout << "LBA: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdKFCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFCulling_ms, average);
    std::cout << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;
    f << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLMTotal_ms);
    deviation = calcDeviation(mpLocalMapper->vdLMTotal_ms, average);
    std::cout << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping LBA complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_edges);
    deviation = calcDeviation(mpLocalMapper->vnLBA_edges, average);
    std::cout << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFopt);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFopt, average);
    std::cout << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFfixed);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFfixed, average);
    std::cout << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_MPs);
    deviation = calcDeviation(mpLocalMapper->vnLBA_MPs, average);
    std::cout << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    f << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    std::cout << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    std::cout << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;
    f << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    f << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;

    // Map complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "Map complexity" << std::endl;
    std::cout << "KFs in map: " << mpAtlas->GetAllKeyFrames().size() << std::endl;
    std::cout << "MPs in map: " << mpAtlas->GetAllMapPoints().size() << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "Map complexity" << std::endl;
    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    Map* pBestMap = vpMaps[0];
    for(int i=1; i<vpMaps.size(); ++i)
    {
        if(pBestMap->GetAllKeyFrames().size() < vpMaps[i]->GetAllKeyFrames().size())
        {
            pBestMap = vpMaps[i];
        }
    }

    f << "KFs in map: " << pBestMap->GetAllKeyFrames().size() << std::endl;
    f << "MPs in map: " << pBestMap->GetAllMapPoints().size() << std::endl;

    f << "---------------------------" << std::endl;
    f << std::endl << "Place Recognition (mean$\\pm$std)" << std::endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "Place Recognition (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdDataQuery_ms);
    deviation = calcDeviation(mpLoopClosing->vdDataQuery_ms, average);
    f << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdEstSim3_ms);
    deviation = calcDeviation(mpLoopClosing->vdEstSim3_ms, average);
    f << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdPRTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdPRTotal_ms, average);
    f << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << std::endl << "Loop Closing (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Loop Closing (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopFusion_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopFusion_ms, average);
    f << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopOptEss_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopOptEss_ms, average);
    f << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopTotal_ms, average);
    f << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nLoop << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nLoop << std::endl;
    average = calcAverage(mpLoopClosing->vnLoopKFs);
    deviation = calcDeviation(mpLoopClosing->vnLoopKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;

    f << std::endl << "Map Merging (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Map Merging (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeMaps_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeMaps_ms, average);
    f << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdWeldingBA_ms);
    deviation = calcDeviation(mpLoopClosing->vdWeldingBA_ms, average);
    f << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeOptEss_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeOptEss_ms, average);
    f << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeTotal_ms, average);
    f << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nMerges << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nMerges << std::endl;
    average = calcAverage(mpLoopClosing->vnMergeKFs);
    deviation = calcDeviation(mpLoopClosing->vnMergeKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vnMergeMPs);
    deviation = calcDeviation(mpLoopClosing->vnMergeMPs, average);
    f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;

    f << std::endl << "Full GBA (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Full GBA (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdGBA_ms);
    deviation = calcDeviation(mpLoopClosing->vdGBA_ms, average);
    f << "GBA: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "GBA: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdUpdateMap_ms);
    deviation = calcDeviation(mpLoopClosing->vdUpdateMap_ms, average);
    f << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdFGBATotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdFGBATotal_ms, average);
    f << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nFGBA_exec << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nFGBA_exec << std::endl;
    f << "Numb abort: " << mpLoopClosing->nFGBA_abort << std::endl;
    std::cout << "Num abort: " << mpLoopClosing->nFGBA_abort << std::endl;
    average = calcAverage(mpLoopClosing->vnGBAKFs);
    deviation = calcDeviation(mpLoopClosing->vnGBAKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vnGBAMPs);
    deviation = calcDeviation(mpLoopClosing->vnGBAMPs, average);
    f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;

    f.close();

}

#endif

Tracking::~Tracking()
{
    //f_track_stats.close();

}

void Tracking::newParameterLoader(Settings *settings) {
    mpCamera = settings->camera1();
    mpCamera = mpAtlas->AddCamera(mpCamera);

    if(settings->needToUndistort()){
        mDistCoef = settings->camera1DistortionCoef();
    }
    else{
        mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    }

    //TODO: missing image scaling and rectification
    mImageScale = 1.0f;

    mK = cv::Mat::eye(3,3,CV_32F);
    mK.at<float>(0,0) = mpCamera->getParameter(0);
    mK.at<float>(1,1) = mpCamera->getParameter(1);
    mK.at<float>(0,2) = mpCamera->getParameter(2);
    mK.at<float>(1,2) = mpCamera->getParameter(3);

    mK_.setIdentity();
    mK_(0,0) = mpCamera->getParameter(0);
    mK_(1,1) = mpCamera->getParameter(1);
    mK_(0,2) = mpCamera->getParameter(2);
    mK_(1,2) = mpCamera->getParameter(3);

    if((mSensor==System::STEREO || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD) &&
        settings->cameraType() == Settings::KannalaBrandt){
        mpCamera2 = settings->camera2();
        mpCamera2 = mpAtlas->AddCamera(mpCamera2);

        mTlr = settings->Tlr();

        mpFrameDrawer->both = true;
    }

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD ){
        mbf = settings->bf();
        mThDepth = settings->b() * settings->thDepth();
    }

    if(mSensor==System::RGBD || mSensor==System::IMU_RGBD){
        mDepthMapFactor = settings->depthMapFactor();
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

    mMinFrames = 0;
    mMaxFrames = settings->fps();
    mbRGB = settings->rgb();

    //ORB parameters
    int nFeatures = settings->nFeatures();
    int nLevels = settings->nLevels();
    int fIniThFAST = settings->initThFAST();
    int fMinThFAST = settings->minThFAST();
    float fScaleFactor = settings->scaleFactor();

    if (std::getenv("USE_ORB") == nullptr)
    {
        mpXFextractor = new XFextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    }
    else
    {
        mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

        if(mSensor==System::STEREO || mSensor==System::IMU_STEREO)
            mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

        if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR)
            mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    }

    //IMU parameters
    Sophus::SE3f Tbc = settings->Tbc();
    mInsertKFsLost = settings->insertKFsWhenLost();
    mImuFreq = settings->imuFrequency();
    mImuPer = 0.001; //1.0 / (double) mImuFreq;     //TODO: ESTO ESTA BIEN?
    float Ng = settings->noiseGyro();
    float Na = settings->noiseAcc();
    float Ngw = settings->gyroWalk();
    float Naw = settings->accWalk();

    const float sf = sqrt(mImuFreq);
    mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
}

bool Tracking::ParseCamParamFile(cv::FileStorage &fSettings)
{
    mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    cout << endl << "Camera Parameters: " << endl;
    bool b_miss_params = false;

    string sCameraName = fSettings["Camera.type"];
    if(sCameraName == "PinHole")
    {
        float fx, fy, cx, cy;
        mImageScale = 1.f;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if(!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.fy"];
        if(!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if(!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if(!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(0) = node.real();
        }
        else
        {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k2"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(1) = node.real();
        }
        else
        {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p1"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(2) = node.real();
        }
        else
        {
            std::cerr << "*Camera.p1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p2"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(3) = node.real();
        }
        else
        {
            std::cerr << "*Camera.p2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.resize(5);
            mDistCoef.at<float>(4) = node.real();
        }

        node = fSettings["Camera.imageScale"];
        if(!node.empty() && node.isReal())
        {
            mImageScale = node.real();
        }

        if(b_miss_params)
        {
            return false;
        }

        if(mImageScale != 1.f)
        {
            // K matrix parameters must be scaled.
            fx = fx * mImageScale;
            fy = fy * mImageScale;
            cx = cx * mImageScale;
            cy = cy * mImageScale;
        }

        vector<float> vCamCalib{fx,fy,cx,cy};

        mpCamera = new Pinhole(vCamCalib);

        mpCamera = mpAtlas->AddCamera(mpCamera);

        std::cout << "- Camera: Pinhole" << std::endl;
        std::cout << "- Image scale: " << mImageScale << std::endl;
        std::cout << "- fx: " << fx << std::endl;
        std::cout << "- fy: " << fy << std::endl;
        std::cout << "- cx: " << cx << std::endl;
        std::cout << "- cy: " << cy << std::endl;
        std::cout << "- k1: " << mDistCoef.at<float>(0) << std::endl;
        std::cout << "- k2: " << mDistCoef.at<float>(1) << std::endl;


        std::cout << "- p1: " << mDistCoef.at<float>(2) << std::endl;
        std::cout << "- p2: " << mDistCoef.at<float>(3) << std::endl;

        if(mDistCoef.rows==5)
            std::cout << "- k3: " << mDistCoef.at<float>(4) << std::endl;

        mK = cv::Mat::eye(3,3,CV_32F);
        mK.at<float>(0,0) = fx;
        mK.at<float>(1,1) = fy;
        mK.at<float>(0,2) = cx;
        mK.at<float>(1,2) = cy;

        mK_.setIdentity();
        mK_(0,0) = fx;
        mK_(1,1) = fy;
        mK_(0,2) = cx;
        mK_(1,2) = cy;
    }
    else if(sCameraName == "KannalaBrandt8")
    {
        float fx, fy, cx, cy;
        float k1, k2, k3, k4;
        mImageScale = 1.f;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if(!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
        node = fSettings["Camera.fy"];
        if(!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if(!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if(!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if(!node.empty() && node.isReal())
        {
            k1 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
        node = fSettings["Camera.k2"];
        if(!node.empty() && node.isReal())
        {
            k2 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if(!node.empty() && node.isReal())
        {
            k3 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k3 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k4"];
        if(!node.empty() && node.isReal())
        {
            k4 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k4 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.imageScale"];
        if(!node.empty() && node.isReal())
        {
            mImageScale = node.real();
        }

        if(!b_miss_params)
        {
            if(mImageScale != 1.f)
            {
                // K matrix parameters must be scaled.
                fx = fx * mImageScale;
                fy = fy * mImageScale;
                cx = cx * mImageScale;
                cy = cy * mImageScale;
            }

            vector<float> vCamCalib{fx,fy,cx,cy,k1,k2,k3,k4};
            mpCamera = new KannalaBrandt8(vCamCalib);
            mpCamera = mpAtlas->AddCamera(mpCamera);
            std::cout << "- Camera: Fisheye" << std::endl;
            std::cout << "- Image scale: " << mImageScale << std::endl;
            std::cout << "- fx: " << fx << std::endl;
            std::cout << "- fy: " << fy << std::endl;
            std::cout << "- cx: " << cx << std::endl;
            std::cout << "- cy: " << cy << std::endl;
            std::cout << "- k1: " << k1 << std::endl;
            std::cout << "- k2: " << k2 << std::endl;
            std::cout << "- k3: " << k3 << std::endl;
            std::cout << "- k4: " << k4 << std::endl;

            mK = cv::Mat::eye(3,3,CV_32F);
            mK.at<float>(0,0) = fx;
            mK.at<float>(1,1) = fy;
            mK.at<float>(0,2) = cx;
            mK.at<float>(1,2) = cy;

            mK_.setIdentity();
            mK_(0,0) = fx;
            mK_(1,1) = fy;
            mK_(0,2) = cx;
            mK_(1,2) = cy;
        }

        if(mSensor==System::STEREO || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD){
            // Right camera
            // Camera calibration parameters
            cv::FileNode node = fSettings["Camera2.fx"];
            if(!node.empty() && node.isReal())
            {
                fx = node.real();
            }
            else
            {
                std::cerr << "*Camera2.fx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera2.fy"];
            if(!node.empty() && node.isReal())
            {
                fy = node.real();
            }
            else
            {
                std::cerr << "*Camera2.fy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.cx"];
            if(!node.empty() && node.isReal())
            {
                cx = node.real();
            }
            else
            {
                std::cerr << "*Camera2.cx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.cy"];
            if(!node.empty() && node.isReal())
            {
                cy = node.real();
            }
            else
            {
                std::cerr << "*Camera2.cy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            // Distortion parameters
            node = fSettings["Camera2.k1"];
            if(!node.empty() && node.isReal())
            {
                k1 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k1 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera2.k2"];
            if(!node.empty() && node.isReal())
            {
                k2 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k2 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.k3"];
            if(!node.empty() && node.isReal())
            {
                k3 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k3 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.k4"];
            if(!node.empty() && node.isReal())
            {
                k4 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k4 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }


            int leftLappingBegin = -1;
            int leftLappingEnd = -1;

            int rightLappingBegin = -1;
            int rightLappingEnd = -1;

            node = fSettings["Camera.lappingBegin"];
            if(!node.empty() && node.isInt())
            {
                leftLappingBegin = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera.lappingBegin not correctly defined" << std::endl;
            }
            node = fSettings["Camera.lappingEnd"];
            if(!node.empty() && node.isInt())
            {
                leftLappingEnd = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera.lappingEnd not correctly defined" << std::endl;
            }
            node = fSettings["Camera2.lappingBegin"];
            if(!node.empty() && node.isInt())
            {
                rightLappingBegin = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera2.lappingBegin not correctly defined" << std::endl;
            }
            node = fSettings["Camera2.lappingEnd"];
            if(!node.empty() && node.isInt())
            {
                rightLappingEnd = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera2.lappingEnd not correctly defined" << std::endl;
            }

            node = fSettings["Tlr"];
            cv::Mat cvTlr;
            if(!node.empty())
            {
                cvTlr = node.mat();
                if(cvTlr.rows != 3 || cvTlr.cols != 4)
                {
                    std::cerr << "*Tlr matrix have to be a 3x4 transformation matrix*" << std::endl;
                    b_miss_params = true;
                }
            }
            else
            {
                std::cerr << "*Tlr matrix doesn't exist*" << std::endl;
                b_miss_params = true;
            }

            if(!b_miss_params)
            {
                if(mImageScale != 1.f)
                {
                    // K matrix parameters must be scaled.
                    fx = fx * mImageScale;
                    fy = fy * mImageScale;
                    cx = cx * mImageScale;
                    cy = cy * mImageScale;

                    leftLappingBegin = leftLappingBegin * mImageScale;
                    leftLappingEnd = leftLappingEnd * mImageScale;
                    rightLappingBegin = rightLappingBegin * mImageScale;
                    rightLappingEnd = rightLappingEnd * mImageScale;
                }

                static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0] = leftLappingBegin;
                static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1] = leftLappingEnd;

                mpFrameDrawer->both = true;

                vector<float> vCamCalib2{fx,fy,cx,cy,k1,k2,k3,k4};
                mpCamera2 = new KannalaBrandt8(vCamCalib2);
                mpCamera2 = mpAtlas->AddCamera(mpCamera2);

                mTlr = Converter::toSophus(cvTlr);

                static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0] = rightLappingBegin;
                static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1] = rightLappingEnd;

                std::cout << "- Camera1 Lapping: " << leftLappingBegin << ", " << leftLappingEnd << std::endl;

                std::cout << std::endl << "Camera2 Parameters:" << std::endl;
                std::cout << "- Camera: Fisheye" << std::endl;
                std::cout << "- Image scale: " << mImageScale << std::endl;
                std::cout << "- fx: " << fx << std::endl;
                std::cout << "- fy: " << fy << std::endl;
                std::cout << "- cx: " << cx << std::endl;
                std::cout << "- cy: " << cy << std::endl;
                std::cout << "- k1: " << k1 << std::endl;
                std::cout << "- k2: " << k2 << std::endl;
                std::cout << "- k3: " << k3 << std::endl;
                std::cout << "- k4: " << k4 << std::endl;

                std::cout << "- mTlr: \n" << cvTlr << std::endl;

                std::cout << "- Camera2 Lapping: " << rightLappingBegin << ", " << rightLappingEnd << std::endl;
            }
        }

        if(b_miss_params)
        {
            return false;
        }

    }
    else
    {
        std::cerr << "*Not Supported Camera Sensor*" << std::endl;
        std::cerr << "Check an example configuration file with the desired sensor" << std::endl;
    }

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD )
    {
        cv::FileNode node = fSettings["Camera.bf"];
        if(!node.empty() && node.isReal())
        {
            mbf = node.real();
            if(mImageScale != 1.f)
            {
                mbf *= mImageScale;
            }
        }
        else
        {
            std::cerr << "*Camera.bf parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

    }

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD)
    {
        float fx = mpCamera->getParameter(0);
        cv::FileNode node = fSettings["ThDepth"];
        if(!node.empty()  && node.isReal())
        {
            mThDepth = node.real();
            mThDepth = mbf*mThDepth/fx;
            cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
        }
        else
        {
            std::cerr << "*ThDepth parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }


    }

    if(mSensor==System::RGBD || mSensor==System::IMU_RGBD)
    {
        cv::FileNode node = fSettings["DepthMapFactor"];
        if(!node.empty() && node.isReal())
        {
            mDepthMapFactor = node.real();
            if(fabs(mDepthMapFactor)<1e-5)
                mDepthMapFactor=1;
            else
                mDepthMapFactor = 1.0f/mDepthMapFactor;
        }
        else
        {
            std::cerr << "*DepthMapFactor parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

    }

    if(b_miss_params)
    {
        return false;
    }

    return true;
}

bool Tracking::ParseORBParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;
    int nFeatures, nLevels, fIniThFAST, fMinThFAST;
    float fScaleFactor;

    cv::FileNode node = fSettings["ORBextractor.nFeatures"];
    if(!node.empty() && node.isInt())
    {
        nFeatures = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.nFeatures parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.scaleFactor"];
    if(!node.empty() && node.isReal())
    {
        fScaleFactor = node.real();
    }
    else
    {
        std::cerr << "*ORBextractor.scaleFactor parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.nLevels"];
    if(!node.empty() && node.isInt())
    {
        nLevels = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.nLevels parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.iniThFAST"];
    if(!node.empty() && node.isInt())
    {
        fIniThFAST = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.iniThFAST parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.minThFAST"];
    if(!node.empty() && node.isInt())
    {
        fMinThFAST = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.minThFAST parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    if(b_miss_params)
    {
        return false;
    }

    if (std::getenv("USE_ORB") == nullptr)
    {
        mpXFextractor = new XFextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    }
    else
    {
        mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

        if(mSensor==System::STEREO || mSensor==System::IMU_STEREO)
            mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

        if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR)
            mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    }

    cout << endl << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    return true;
}

bool Tracking::ParseIMUParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    cv::Mat cvTbc;
    cv::FileNode node = fSettings["Tbc"];
    if(!node.empty())
    {
        cvTbc = node.mat();
        if(cvTbc.rows != 4 || cvTbc.cols != 4)
        {
            std::cerr << "*Tbc matrix have to be a 4x4 transformation matrix*" << std::endl;
            b_miss_params = true;
        }
    }
    else
    {
        std::cerr << "*Tbc matrix doesn't exist*" << std::endl;
        b_miss_params = true;
    }
    cout << endl;
    cout << "Left camera to Imu Transform (Tbc): " << endl << cvTbc << endl;
    Eigen::Matrix<float,4,4,Eigen::RowMajor> eigTbc(cvTbc.ptr<float>(0));
    Sophus::SE3f Tbc(eigTbc);

    node = fSettings["InsertKFsWhenLost"];
    mInsertKFsLost = true;
    if(!node.empty() && node.isInt())
    {
        mInsertKFsLost = (bool) node.operator int();
    }

    if(!mInsertKFsLost)
        cout << "Do not insert keyframes when lost visual tracking " << endl;



    float Ng, Na, Ngw, Naw;

    node = fSettings["IMU.Frequency"];
    if(!node.empty() && node.isInt())
    {
        mImuFreq = node.operator int();
        mImuPer = 0.001; //1.0 / (double) mImuFreq;
    }
    else
    {
        std::cerr << "*IMU.Frequency parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseGyro"];
    if(!node.empty() && node.isReal())
    {
        Ng = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseGyro parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseAcc"];
    if(!node.empty() && node.isReal())
    {
        Na = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseAcc parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.GyroWalk"];
    if(!node.empty() && node.isReal())
    {
        Ngw = node.real();
    }
    else
    {
        std::cerr << "*IMU.GyroWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.AccWalk"];
    if(!node.empty() && node.isReal())
    {
        Naw = node.real();
    }
    else
    {
        std::cerr << "*IMU.AccWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.fastInit"];
    mFastInit = false;
    if(!node.empty())
    {
        mFastInit = static_cast<int>(fSettings["IMU.fastInit"]) != 0;
    }

    if(mFastInit)
        cout << "Fast IMU initialization. Acceleration is not checked \n";

    if(b_miss_params)
    {
        return false;
    }

    const float sf = sqrt(mImuFreq);
    cout << endl;
    cout << "IMU frequency: " << mImuFreq << " Hz" << endl;
    cout << "IMU gyro noise: " << Ng << " rad/s/sqrt(Hz)" << endl;
    cout << "IMU gyro walk: " << Ngw << " rad/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer noise: " << Na << " m/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer walk: " << Naw << " m/s^3/sqrt(Hz)" << endl;

    mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);


    return true;
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

void Tracking::SetStepByStep(bool bSet)
{
    bStepByStep = bSet;
}

bool Tracking::GetStepByStep()
{
    return bStepByStep;
}



Sophus::SE3f Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, string filename)
{
    const auto fpsStart = std::chrono::steady_clock::now();
    //cout << "GrabImageStereo" << endl;

    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;
    mImRight = imRectRight;

    if(mImGray.channels()==3)
    {
        //cout << "Image with 3 channels" << endl;
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        //cout << "Image with 4 channels" << endl;
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_BGRA2GRAY);
        }
    }

    //cout << "Incoming frame creation" << endl;

    const bool bUseORB = (std::getenv("USE_ORB") != nullptr);
    if(bUseORB)
    {
        if (mSensor == System::STEREO && !mpCamera2)
            mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera);
        else if(mSensor == System::STEREO && mpCamera2)
            mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr);
        else if(mSensor == System::IMU_STEREO && !mpCamera2)
            mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib);
        else if(mSensor == System::IMU_STEREO && mpCamera2)
            mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr,&mLastFrame,*mpImuCalib);
    }
    else
    {
        if(mpCamera2)
        {
            std::cerr << "XFeat stereo currently supports rectified pinhole stereo only. Set USE_ORB=1 for fisheye stereo." << std::endl;
            exit(-1);
        }

        if (mSensor == System::STEREO)
            mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpXFextractor,mpXFextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera);
        else if(mSensor == System::IMU_STEREO)
            mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpXFextractor,mpXFextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib);
    }

    //cout << "Incoming frame ended" << endl;

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
    vdStereoMatch_ms.push_back(mCurrentFrame.mTimeStereoMatch);
#endif

    //cout << "Tracking start" << endl;
    Track();
    LogRuntimeFps("stereo", fpsStart, mCurrentFrame.mnId, mCurrentFrame.mTimeStamp, mCurrentFrame.N, mState);
    //cout << "Tracking end" << endl;

    return mCurrentFrame.GetPose();
}


Sophus::SE3f Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, string filename)
{
    const auto fpsStart = std::chrono::steady_clock::now();
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    if (mSensor == System::RGBD)
        if (std::getenv("USE_ORB") == nullptr)
        {
            // XFeat
            mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpXFextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera);
        }
        else
        {
            // ORB
            mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera);
        }

    else if(mSensor == System::IMU_RGBD)
        mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib);






    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
#endif

    Track();
    LogRuntimeFps("rgbd", fpsStart, mCurrentFrame.mnId, mCurrentFrame.mTimeStamp, mCurrentFrame.N, mState);

    return mCurrentFrame.GetPose();
}


Sophus::SE3f Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename)
{
    const auto fpsStart = std::chrono::steady_clock::now();
    // 这里是原版代码，把输入图像统一转换成灰度图，供后续特征提取和跟踪使用。考虑到XFeat是输入的原版图像信息，rgb先做平均再输入，故做修改
    // mImGray = im;
    // if(mImGray.channels()==3)
    // {
    //     if(mbRGB)
    //         cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
    //     else
    //         cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
    // }
    // else if(mImGray.channels()==4)
    // {
    //     if(mbRGB)
    //         cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
    //     else
    //         cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
    // }

    mImGray = im;

    // ORB 路径：保持原来的灰度化逻辑
    if (std::getenv("USE_ORB") != nullptr)
    {
        if(mImGray.channels()==3)
        {
            if(mbRGB)
                cvtColor(mImGray, mImGray, cv::COLOR_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
        }
        else if(mImGray.channels()==4)
        {
            if(mbRGB)
                cvtColor(mImGray, mImGray, cv::COLOR_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, cv::COLOR_BGRA2GRAY);
        }
    }
    // XFeat 路径：保留 1 通道或 3 通道，4 通道先去 alpha
    else
    {
        if (mImGray.channels() == 4)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, cv::COLOR_RGBA2RGB);
            else
                cvtColor(mImGray, mImGray, cv::COLOR_BGRA2BGR);
        }

        // 1 通道：直接保留
        // 3 通道：直接保留
    }

    if (mSensor == System::MONOCULAR)
    {
        if (std::getenv("USE_ORB") == nullptr)
        {
            if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET ||(lastID - initID) < mMaxFrames)
                mCurrentFrame = Frame(mImGray,timestamp,mpXFextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
            else
                mCurrentFrame = Frame(mImGray,timestamp,mpXFextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
        }
        else
        {
            if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET ||(lastID - initID) < mMaxFrames)
                mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
            else
                mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
        }
    }
    else if(mSensor == System::IMU_MONOCULAR)
    {
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        {
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib);
        }
        else
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib);
    }

    if (mState==NO_IMAGES_YET)
        t0=timestamp;

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
#endif

    lastID = mCurrentFrame.mnId;
    Track();
    LogRuntimeFps("mono", fpsStart, mCurrentFrame.mnId, mCurrentFrame.mTimeStamp, mCurrentFrame.N, mState);

    return mCurrentFrame.GetPose();
}


void Tracking::GrabImuData(const IMU::Point &imuMeasurement)
{
    unique_lock<mutex> lock(mMutexImuQueue);
    mlQueueImuData.push_back(imuMeasurement);
}

void Tracking::PreintegrateIMU()
{

    if(!mCurrentFrame.mpPrevFrame)
    {
        Verbose::PrintMess("non prev frame ", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated();
        return;
    }

    mvImuFromLastFrame.clear();
    mvImuFromLastFrame.reserve(mlQueueImuData.size());
    if(mlQueueImuData.size() == 0)
    {
        Verbose::PrintMess("Not IMU data in mlQueueImuData!!", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated();
        return;
    }

    while(true)
    {
        bool bSleep = false;
        {
            unique_lock<mutex> lock(mMutexImuQueue);
            if(!mlQueueImuData.empty())
            {
                IMU::Point* m = &mlQueueImuData.front();
                cout.precision(17);
                if(m->t<mCurrentFrame.mpPrevFrame->mTimeStamp-mImuPer)
                {
                    mlQueueImuData.pop_front();
                }
                else if(m->t<mCurrentFrame.mTimeStamp-mImuPer)
                {
                    mvImuFromLastFrame.push_back(*m);
                    mlQueueImuData.pop_front();
                }
                else
                {
                    mvImuFromLastFrame.push_back(*m);
                    break;
                }
            }
            else
            {
                break;
                bSleep = true;
            }
        }
        if(bSleep)
            usleep(500);
    }

    const int n = mvImuFromLastFrame.size()-1;
    if(n==0){
        cout << "Empty IMU measurements vector!!!\n";
        return;
    }

    IMU::Preintegrated* pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias,mCurrentFrame.mImuCalib);

    for(int i=0; i<n; i++)
    {
        float tstep;
        Eigen::Vector3f acc, angVel;
        if((i==0) && (i<(n-1)))
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tini = mvImuFromLastFrame[i].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tini/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tini/tab))*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
        }
        else if(i<(n-1))
        {
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a)*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w)*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
        }
        else if((i>0) && (i==(n-1)))
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tend = mvImuFromLastFrame[i+1].t-mCurrentFrame.mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tend/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tend/tab))*0.5f;
            tstep = mCurrentFrame.mTimeStamp-mvImuFromLastFrame[i].t;
        }
        else if((i==0) && (i==(n-1)))
        {
            acc = mvImuFromLastFrame[i].a;
            angVel = mvImuFromLastFrame[i].w;
            tstep = mCurrentFrame.mTimeStamp-mCurrentFrame.mpPrevFrame->mTimeStamp;
        }

        if (!mpImuPreintegratedFromLastKF)
            cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
        mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc,angVel,tstep);
        pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc,angVel,tstep);
    }

    mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
    mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
    mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;

    mCurrentFrame.setIntegrated();

    //Verbose::PrintMess("Preintegration is finished!! ", Verbose::VERBOSITY_DEBUG);
}


bool Tracking::PredictStateIMU()
{
    if(!mCurrentFrame.mpPrevFrame)
    {
        Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
        return false;
    }

    if(mbMapUpdated && mpLastKeyFrame)
    {
        const Eigen::Vector3f twb1 = mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mpLastKeyFrame->GetVelocity();

        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float t12 = mpImuPreintegratedFromLastKF->dT;

        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
        Eigen::Vector3f twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
        Eigen::Vector3f Vwb2 = Vwb1 + t12*Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2);

        mCurrentFrame.mImuBias = mpLastKeyFrame->GetImuBias();
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    else if(!mbMapUpdated)
    {
        const Eigen::Vector3f twb1 = mLastFrame.GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.GetVelocity();
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float t12 = mCurrentFrame.mpImuPreintegratedFrame->dT;

        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaRotation(mLastFrame.mImuBias));
        Eigen::Vector3f twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaPosition(mLastFrame.mImuBias);
        Eigen::Vector3f Vwb2 = Vwb1 + t12*Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaVelocity(mLastFrame.mImuBias);

        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2);

        mCurrentFrame.mImuBias = mLastFrame.mImuBias;
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    else
        cout << "not IMU prediction!!" << endl;

    return false;
}

void Tracking::ResetFrameIMU()
{
    // TODO To implement...
}


void Tracking::Track()
{
    if (bStepByStep)
    {
        std::cout << "Tracking: Waiting to the next step" << std::endl;
        while(!mbStep && bStepByStep)
            usleep(500);
        mbStep = false;
    }

    if(mpLocalMapper->mbBadImu)
    {
        cout << "TRACK: Reset map because local mapper set the bad imu flag " << endl;
        mpSystem->ResetActiveMap();
        return;
    }

    Map* pCurrentMap = mpAtlas->GetCurrentMap();
    if(!pCurrentMap)
    {
        cout << "ERROR: There is not an active map in the atlas" << endl;
    }

    if(mState!=NO_IMAGES_YET)
    {
        if(mLastFrame.mTimeStamp>mCurrentFrame.mTimeStamp)
        {
            cerr << "ERROR: Frame with a timestamp older than previous frame detected!" << endl;
            unique_lock<mutex> lock(mMutexImuQueue);
            mlQueueImuData.clear();
            CreateMapInAtlas();
            return;
        }
        else if(mCurrentFrame.mTimeStamp>mLastFrame.mTimeStamp+1.0)
        {
            // cout << mCurrentFrame.mTimeStamp << ", " << mLastFrame.mTimeStamp << endl;
            // cout << "id last: " << mLastFrame.mnId << "    id curr: " << mCurrentFrame.mnId << endl;
            if(mpAtlas->isInertial())
            {

                if(mpAtlas->isImuInitialized())
                {
                    cout << "Timestamp jump detected. State set to LOST. Reseting IMU integration..." << endl;
                    if(!pCurrentMap->GetIniertialBA2())
                    {
                        mpSystem->ResetActiveMap();
                    }
                    else
                    {
                        CreateMapInAtlas();
                    }
                }
                else
                {
                    cout << "Timestamp jump detected, before IMU initialization. Reseting..." << endl;
                    mpSystem->ResetActiveMap();
                }
                return;
            }

        }
    }


    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpLastKeyFrame)
        mCurrentFrame.SetNewBias(mpLastKeyFrame->GetImuBias());

    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mbCreatedMap)
    {
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartPreIMU = std::chrono::steady_clock::now();
#endif
        PreintegrateIMU();
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndPreIMU = std::chrono::steady_clock::now();

        double timePreImu = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPreIMU - time_StartPreIMU).count();
        vdIMUInteg_ms.push_back(timePreImu);
#endif

    }
    mbCreatedMap = false;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);

    mbMapUpdated = false;

    int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();
    int nMapChangeIndex = pCurrentMap->GetLastMapChange();
    if(nCurMapChangeIndex>nMapChangeIndex)
    {
        pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
        mbMapUpdated = true;
    }


    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD)
        {
            StereoInitialization();
        }
        else
        {
            MonocularInitialization();
        }

        // [XFEAT_FIX_20260414] Keep viewer state in sync during initialization, otherwise it may stay at "WAITING FOR IMAGES".
        mpFrameDrawer->Update(this);

        if(mState!=OK) // If rightly initialized, mState=OK
        {
            mLastFrame = Frame(mCurrentFrame);
            return;
        }

        if(mpAtlas->GetAllMaps().size() == 1)
        {
            mnFirstFrameId = mCurrentFrame.mnId;
        }
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartPosePred = std::chrono::steady_clock::now();
#endif

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {

            // State OK
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            if(mState==OK)
            {
            //
            //     // Local Mapping might have changed some MapPoints tracked in last frame
            //     CheckReplacedInLastFrame();
            //
            //     if((!mbVelocity && !pCurrentMap->isImuInitialized()) || mCurrentFrame.mnId<mnLastRelocFrameId+2)
            //     {
            //         Verbose::PrintMess("TRACK: Track with respect to the reference KF ", Verbose::VERBOSITY_DEBUG);
            //         bOK = TrackReferenceKeyFrame();
            //     }
            //     else
            //     {
            //         Verbose::PrintMess("TRACK: Track with motion model", Verbose::VERBOSITY_DEBUG);
            //         bOK = TrackWithMotionModel();
            //         if(!bOK)
            //             bOK = TrackReferenceKeyFrame();
            //     }

            CheckReplacedInLastFrame();

            if((!mbVelocity && !pCurrentMap->isImuInitialized()) || mCurrentFrame.mnId<mnLastRelocFrameId+2)
            {
                Verbose::PrintMess("TRACK: Track with respect to the reference KF ", Verbose::VERBOSITY_DEBUG);
                bOK = TrackReferenceKeyFrame();
            }
            else
            {
                Verbose::PrintMess("TRACK: Track with motion model", Verbose::VERBOSITY_DEBUG);
                bOK = TrackWithMotionModel();

                if(!bOK)
                {
                    bOK = TrackReferenceKeyFrame();
                }
            } //end


                if (!bOK)
                {
                    const bool bUseORBTracking = (std::getenv("USE_ORB") != nullptr);
                    if ( mCurrentFrame.mnId<=(mnLastRelocFrameId+mnFramesToResetIMU) &&
                         (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD))
                    {
                        mState = LOST;
                    }
                    else if(pCurrentMap->KeyFramesInMap()>10)
                    {
                        // cout << "KF in map: " << pCurrentMap->KeyFramesInMap() << endl;
                        mState = RECENTLY_LOST;
                        mTimeStampLost = mCurrentFrame.mTimeStamp;
                    }
                    else
                    {
                        // XFeat单目路径: 小地图阶段允许短暂RECENTLY_LOST，避免单帧退化直接触发整图重置。
                        if(!bUseORBTracking && mSensor == System::MONOCULAR)
                        {
                            mState = RECENTLY_LOST;
                            mTimeStampLost = mCurrentFrame.mTimeStamp;
                        }
                        else
                        {
                            mState = LOST;
                        }
                    }
                }
            }
            else
            {

                if (mState == RECENTLY_LOST)
                {
                    Verbose::PrintMess("Lost for a short time", Verbose::VERBOSITY_NORMAL);

                    bOK = true;
                    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))
                    {
                        if(pCurrentMap->isImuInitialized())
                            PredictStateIMU();
                        else
                            bOK = false;

                        if (mCurrentFrame.mTimeStamp-mTimeStampLost>time_recently_lost)
                        {
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false;
                        }
                    }
                    else
                    {
                        bOK = Relocalization();
                        //std::cout << "mCurrentFrame.mTimeStamp:" << to_string(mCurrentFrame.mTimeStamp) << std::endl;
                        //std::cout << "mTimeStampLost:" << to_string(mTimeStampLost) << std::endl;
                        if(mCurrentFrame.mTimeStamp-mTimeStampLost>3.0f && !bOK)
                        {
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false;
                        }
                    }
                }
                else if (mState == LOST)
                {
                    Verbose::PrintMess("A new map is started...", Verbose::VERBOSITY_NORMAL);

                    if (pCurrentMap->KeyFramesInMap()<10)
                    {
                        mpSystem->ResetActiveMap();
                        Verbose::PrintMess("Reseting current map...", Verbose::VERBOSITY_NORMAL);
                    }else
                        CreateMapInAtlas();

                    if(mpLastKeyFrame)
                        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

                    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

                    return;
                }
            }

        }
        else
        {
            // Localization Mode: Local Mapping is deactivated (TODO Not available in inertial mode)
            if(mState==LOST)
            {
                if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                    Verbose::PrintMess("IMU. State LOST", Verbose::VERBOSITY_NORMAL);
                bOK = Relocalization();
            }
            else
            {
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map
                    if(mbVelocity)
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    Sophus::SE3f TcwMM;
                    if(mbVelocity)
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.GetPose();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndPosePred = std::chrono::steady_clock::now();

        double timePosePred = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPosePred - time_StartPosePred).count();
        vdPosePred_ms.push_back(timePosePred);
#endif


#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartLMTrack = std::chrono::steady_clock::now();
#endif
        // If we have an initial estimation of the camera pose and matching. Track the local map.
        // if(!mbOnlyTracking)
        // {
        //     if(bOK)
        //     {
        //         bOK = TrackLocalMap();
        //
        //     }
        //     if(!bOK)
        //         cout << "Fail to track local map!" << endl;
        // }
        //调试: TrackLocalMap() 调用和失败原因摘要。
        if(!mbOnlyTracking)
        {
            if(bOK)
            {
                bOK = TrackLocalMap();
            }

            if(!bOK)
            {
                bool in_recent_reloc_window = (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames);
                const bool bUseORBRecentReloc = (std::getenv("USE_ORB") != nullptr);
                const int recentRelocInlierTh = bUseORBRecentReloc ? 50 : GetXFeatRecentRelocInlierThreshold();

                std::cout << "[Track] Fail to track local map!"
                          << " frame=" << mCurrentFrame.mnId
                          << " mnMatchesInliers=" << mnMatchesInliers
                          << " recent_reloc_window=" << (in_recent_reloc_window ? "true" : "false")
                          << " threshold_used=";

                if(in_recent_reloc_window)
                    std::cout << recentRelocInlierTh
                              << "(recent_reloc_" << (bUseORBRecentReloc ? "ORB" : "XFeat") << ")";
                else if (mSensor == System::IMU_MONOCULAR)
                    std::cout << (mpAtlas->isImuInitialized() ? "15(imu_init)" : "50(imu_not_init)");
                else if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                    std::cout << "15(imu_stereo_or_rgbd)";
                else
                    std::cout << "30(monocular_visual)";

                std::cout << std::endl;
            }
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if(bOK)
            mState = OK;
        else if (mState == OK)
        {
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            {
                Verbose::PrintMess("Track lost for less than one second...", Verbose::VERBOSITY_NORMAL);
                if(!pCurrentMap->isImuInitialized() || !pCurrentMap->GetIniertialBA2())
                {
                    cout << "IMU is not or recently initialized. Reseting active map..." << endl;
                    mpSystem->ResetActiveMap();
                }

                mState=RECENTLY_LOST;
            }
            else
                mState=RECENTLY_LOST; // visual to lost

            /*if(mCurrentFrame.mnId>mnLastRelocFrameId+mMaxFrames)
            {*/
                mTimeStampLost = mCurrentFrame.mTimeStamp;
            //}
        }

        // Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it shluld be once mCurrFrame is completely modified)
        if((mCurrentFrame.mnId<(mnLastRelocFrameId+mnFramesToResetIMU)) && (mCurrentFrame.mnId > mnFramesToResetIMU) &&
           (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && pCurrentMap->isImuInitialized())
        {
            // TODO check this situation
            Verbose::PrintMess("Saving pointer to frame. imu needs reset...", Verbose::VERBOSITY_NORMAL);
            Frame* pF = new Frame(mCurrentFrame);
            pF->mpPrevFrame = new Frame(mLastFrame);

            // Load preintegration
            pF->mpImuPreintegratedFrame = new IMU::Preintegrated(mCurrentFrame.mpImuPreintegratedFrame);
        }

        if(pCurrentMap->isImuInitialized())
        {
            if(bOK)
            {
                if(mCurrentFrame.mnId==(mnLastRelocFrameId+mnFramesToResetIMU))
                {
                    cout << "RESETING FRAME!!!" << endl;
                    ResetFrameIMU();
                }
                else if(mCurrentFrame.mnId>(mnLastRelocFrameId+30))
                    mLastBias = mCurrentFrame.mImuBias;
            }
        }

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndLMTrack = std::chrono::steady_clock::now();

        double timeLMTrack = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLMTrack - time_StartLMTrack).count();
        vdLMTrack_ms.push_back(timeLMTrack);
#endif

        // Update drawer
        mpFrameDrawer->Update(this);
        if(mCurrentFrame.isSet())
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

        if(bOK || mState==RECENTLY_LOST)
        {
            // Update motion model
            if(mLastFrame.isSet() && mCurrentFrame.isSet())
            {
                Sophus::SE3f LastTwc = mLastFrame.GetPose().inverse();
                mVelocity = mCurrentFrame.GetPose() * LastTwc;
                mbVelocity = true;
            }
            else {
                mbVelocity = false;
            }

            if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartNewKF = std::chrono::steady_clock::now();
#endif
            bool bNeedKF = NeedNewKeyFrame();

            // Check if we need to insert a new keyframe
            // if(bNeedKF && bOK)
            if(bNeedKF && (bOK || (mInsertKFsLost && mState==RECENTLY_LOST &&
                                   (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))))
                CreateNewKeyFrame();

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndNewKF = std::chrono::steady_clock::now();

            double timeNewKF = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndNewKF - time_StartNewKF).count();
            vdNewKF_ms.push_back(timeNewKF);
#endif

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame. Only has effect if lastframe is tracked
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        //诊断: 量化“提取点/有效匹配点”的空间与尺度分布，以及匹配点深度近中远占比。
        if(ShouldRunFeatureDiagForFrame(mCurrentFrame.mnId))
        {
            const bool bUseORBDiag = (std::getenv("USE_ORB") != nullptr);
            LogFeatureDistributionDiagnostics(mCurrentFrame, mState, bOK, bUseORBDiag, mnMatchesInliers);
        }
        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(pCurrentMap->KeyFramesInMap()<=10)
            {
                mpSystem->ResetActiveMap();
                return;
            }
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                if (!pCurrentMap->isImuInitialized())
                {
                    Verbose::PrintMess("Track lost before IMU initialisation, reseting...", Verbose::VERBOSITY_QUIET);
                    mpSystem->ResetActiveMap();
                    return;
                }

            CreateMapInAtlas();

            return;
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }




    if(mState==OK || mState==RECENTLY_LOST)
    {
        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        if(mCurrentFrame.isSet())
        {
            Sophus::SE3f Tcr_ = mCurrentFrame.GetPose() * mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr_);
            mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState==LOST);
        }
        else
        {
            // This can happen if tracking is lost
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState==LOST);
        }

    }

#ifdef REGISTER_LOOP
    if (Stop()) {

        // Safe area to stop
        while(isStopped())
        {
            usleep(3000);
        }
    }
#endif
}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            if (!mCurrentFrame.mpImuPreintegrated || !mLastFrame.mpImuPreintegrated)
            {
                cout << "not IMU meas" << endl;
                return;
            }

            if (!mFastInit && (mCurrentFrame.mpImuPreintegratedFrame->avgA-mLastFrame.mpImuPreintegratedFrame->avgA).norm()<0.5)
            {
                cout << "not enough acceleration" << endl;
                return;
            }

            if(mpImuPreintegratedFromLastKF)
                delete mpImuPreintegratedFromLastKF;

            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
            mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        }

        // Set Frame pose to the origin (In case of inertial SLAM to imu)
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            Eigen::Matrix3f Rwb0 = mCurrentFrame.mImuCalib.mTcb.rotationMatrix();
            Eigen::Vector3f twb0 = mCurrentFrame.mImuCalib.mTcb.translation();
            Eigen::Vector3f Vwb0;
            Vwb0.setZero();
            mCurrentFrame.SetImuPoseVelocity(Rwb0, twb0, Vwb0);
        }
        else
            mCurrentFrame.SetPose(Sophus::SE3f());

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpAtlas->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        if(!mpCamera2){
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                float z = mCurrentFrame.mvDepth[i];
                if(z>0)
                {
                    Eigen::Vector3f x3D;
                    mCurrentFrame.UnprojectStereo(i, x3D);
                    MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());
                    pNewMP->SetCreationSource(MapPoint::kCreationSourceStereoInit);
                    pNewMP->AddObservation(pKFini,i);
                    pKFini->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                }
            }
        } else{
            for(int i = 0; i < mCurrentFrame.Nleft; i++){
                int rightIndex = mCurrentFrame.mvLeftToRightMatch[i];
                if(rightIndex != -1){
                    Eigen::Vector3f x3D = mCurrentFrame.mvStereo3Dpoints[i];

                    MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());
                    pNewMP->SetCreationSource(MapPoint::kCreationSourceStereoInit);

                    pNewMP->AddObservation(pKFini,i);
                    pNewMP->AddObservation(pKFini,rightIndex + mCurrentFrame.Nleft);

                    pKFini->AddMapPoint(pNewMP,i);
                    pKFini->AddMapPoint(pNewMP,rightIndex + mCurrentFrame.Nleft);

                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    mCurrentFrame.mvpMapPoints[rightIndex + mCurrentFrame.Nleft]=pNewMP;
                }
            }
        }

        Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);

        //cout << "Active map: " << mpAtlas->GetCurrentMap()->GetId() << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;
        //mnLastRelocFrameId = mCurrentFrame.mnId;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpAtlas->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

        mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

        mState=OK;
    }
}


void Tracking::MonocularInitialization()
{

    if(!mbReadyToInitializate)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {

            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            if (mSensor == System::IMU_MONOCULAR)
            {
                if(mpImuPreintegratedFromLastKF)
                {
                    delete mpImuPreintegratedFromLastKF;
                }
                mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
                mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;

            }

            mbReadyToInitializate = true;

            return;
        }
    }
    else
    {
        if (((int)mCurrentFrame.mvKeys.size()<=100)||((mSensor == System::IMU_MONOCULAR)&&(mLastFrame.mTimeStamp-mInitialFrame.mTimeStamp>1.0)))
        {
            mbReadyToInitializate = false;

            return;
        }

        // Find correspondences
        const bool bUseORB = (std::getenv("USE_ORB") != nullptr);
        int nmatches = 0;
        if(bUseORB)
        {
            ORBmatcher matcher(0.9, true);
            nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);
        }
        else
        {
            XFeatMatcher matcher(0.9f, false);
            nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);
        }

        // Check if there are enough correspondences
        if(nmatches<100)
        {
            mbReadyToInitializate = false;
            return;
        }

        Sophus::SE3f Tcw;
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)
        std::vector<int> vIniMatchesForReconstruct = mvIniMatches;
        int nmatchesForReconstruct = nmatches;

        if(!bUseORB)
        {
            float medianDisp = 0.0f;
            float minDispGate = 0.0f;
            int dispFiltered = 0;
            bool dispGateRelaxed = false;
            const int pruned = PruneXFeatInitMatchesForReconstruction(mInitialFrame,
                                                                      mCurrentFrame,
                                                                      vIniMatchesForReconstruct,
                                                                      medianDisp,
                                                                      minDispGate,
                                                                      dispFiltered,
                                                                      dispGateRelaxed);
            nmatchesForReconstruct = CountValidInitMatches(vIniMatchesForReconstruct);

            //调试: 输出双视图重建前裁剪统计，定位“高匹配数反而初始化失败”的门槛问题。
            if(ShouldPrintXFeatDebug(mCurrentFrame.mnId))
            {
                std::cout << "[InitMono] MATCH_PRUNE "
                          << "raw=" << nmatches
                          << " prune=" << pruned
                          << " disp_filtered=" << dispFiltered
                          << " disp_gate_relaxed=" << (dispGateRelaxed ? "true" : "false")
                          << " retained=" << nmatchesForReconstruct
                          << " disp_median=" << medianDisp
                          << " disp_gate=" << minDispGate
                          << std::endl;
            }
        }

        if(mpCamera->ReconstructWithTwoViews(mInitialFrame.mvKeysUn,mCurrentFrame.mvKeysUn,vIniMatchesForReconstruct,Tcw,mvIniP3D,vbTriangulated))
        {
            mvIniMatches.swap(vIniMatchesForReconstruct);
            nmatches = CountValidInitMatches(mvIniMatches);

            int invalidInitMatches = 0;
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                const int idx2 = mvIniMatches[i];
                if(idx2 < 0)
                    continue;

                const bool bInvalidInitIndex =
                    i >= vbTriangulated.size() ||
                    i >= mvIniP3D.size() ||
                    i >= mInitialFrame.mvKeysUn.size() ||
                    idx2 >= static_cast<int>(mCurrentFrame.mvKeysUn.size());
                if(bInvalidInitIndex || !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                    if(bInvalidInitIndex)
                        ++invalidInitMatches;
                }
            }

            if(invalidInitMatches > 0 && ShouldPrintXFeatDebug(mCurrentFrame.mnId))
            {
                std::cout << "[InitMono] INVALID_RECONSTRUCT_MATCHES "
                          << "invalid=" << invalidInitMatches
                          << " matches_size=" << mvIniMatches.size()
                          << " p3d_size=" << mvIniP3D.size()
                          << " triangulated_size=" << vbTriangulated.size()
                          << " init_keys=" << mInitialFrame.mvKeysUn.size()
                          << " current_keys=" << mCurrentFrame.mvKeysUn.size()
                          << std::endl;
            }

            if(nmatches < 50)
            {
                if(ShouldPrintXFeatDebug(mCurrentFrame.mnId))
                {
                    std::cout << "[InitMono] REANCHOR "
                              << "reason=too_few_triangulated_matches"
                              << " retained=" << nmatches
                              << std::endl;
                }
                mbReadyToInitializate = false;
                return;
            }

            // Set Frame Poses
            mInitialFrame.SetPose(Sophus::SE3f());
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
        else if(!bUseORB)
        {
            nmatches = nmatchesForReconstruct;
            const int initFrameSpan = static_cast<int>(mCurrentFrame.mnId - mInitialFrame.mnId);
            const bool bStaleReference = (initFrameSpan > GetXFeatInitMaxFrameSpan());
            const bool bDegradingReference = (initFrameSpan > 8 && nmatches < GetXFeatInitReanchorMatchFloor());
            if(bStaleReference || bDegradingReference)
            {
                //调试: 同一初始化参考帧跨度过大且重建失败，重锚到当前帧以避免持续退化。
                if(ShouldPrintXFeatDebug(mCurrentFrame.mnId))
                {
                    std::cout << "[InitMono] REANCHOR "
                              << "reason=" << (bStaleReference ? "reconstruct_failed_with_stale_reference" :
                                                 "reconstruct_failed_with_degrading_matches")
                              << " init_frame=" << mInitialFrame.mnId
                              << " current_frame=" << mCurrentFrame.mnId
                              << " span=" << initFrameSpan
                              << " nmatches=" << nmatches
                              << std::endl;
                }

                mInitialFrame = Frame(mCurrentFrame);
                mLastFrame = Frame(mCurrentFrame);
                mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
                for(size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); ++i)
                    mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;
                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
            }
        }
    }
}



void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

    if(mSensor == System::IMU_MONOCULAR)
        pKFini->mpImuPreintegrated = (IMU::Preintegrated*)(NULL);


    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpAtlas->AddKeyFrame(pKFini);
    mpAtlas->AddKeyFrame(pKFcur);

    int skippedInitMapPoints = 0;
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        const int idx2 = mvIniMatches[i];
        if(idx2<0)
            continue;

        const bool bInvalidMapPointIndex =
            i >= mvIniP3D.size() ||
            i >= static_cast<size_t>(pKFini->N) ||
            idx2 >= pKFcur->N ||
            idx2 >= static_cast<int>(mCurrentFrame.mvpMapPoints.size()) ||
            idx2 >= static_cast<int>(mCurrentFrame.mvbOutlier.size());
        if(bInvalidMapPointIndex)
        {
            ++skippedInitMapPoints;
            continue;
        }

        const cv::Point3f& p3d = mvIniP3D[i];
        if(!std::isfinite(p3d.x) || !std::isfinite(p3d.y) || !std::isfinite(p3d.z))
        {
            ++skippedInitMapPoints;
            continue;
        }

        //Create MapPoint.
        Eigen::Vector3f worldPos;
        worldPos << p3d.x, p3d.y, p3d.z;
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpAtlas->GetCurrentMap());
        pMP->SetCreationSource(MapPoint::kCreationSourceMonoInit);

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,static_cast<size_t>(idx2));

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,static_cast<size_t>(idx2));

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[static_cast<size_t>(idx2)] = pMP;
        mCurrentFrame.mvbOutlier[static_cast<size_t>(idx2)] = false;

        //Add to Map
        mpAtlas->AddMapPoint(pMP);
    }

    if(skippedInitMapPoints > 0 && ShouldPrintXFeatDebug(mCurrentFrame.mnId))
    {
        std::cout << "[InitMono] SKIP_INVALID_MAPPOINTS "
                  << "skipped=" << skippedInitMapPoints
                  << " matches_size=" << mvIniMatches.size()
                  << " p3d_size=" << mvIniP3D.size()
                  << " kf_ini_points=" << pKFini->N
                  << " kf_cur_points=" << pKFcur->N
                  << " current_frame_points=" << mCurrentFrame.mvpMapPoints.size()
                  << std::endl;
    }


    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    std::set<MapPoint*> sMPs;
    sMPs = pKFini->GetMapPoints();

    // Bundle Adjustment
    Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);
    Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(),20);

    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    const float baseline = (pKFcur->GetCameraCenter() - pKFini->GetCameraCenter()).norm();
    const float baselineDepthRatio = (medianDepth > 1e-6f) ? (baseline / medianDepth) : 0.0f;
    const bool bUseORBInit = (std::getenv("USE_ORB") != nullptr);
    const float minBaselineDepthRatio = bUseORBInit ? 0.005f : GetXFeatInitMinBaselineDepthRatio();
    const int trackedMapPoints = pKFcur->TrackedMapPoints(1);

    if(ShouldPrintXFeatDebug(mCurrentFrame.mnId))
    {
        std::cout << "[InitMono] "
                  << "medianDepth=" << medianDepth
                  << " baseline=" << baseline
                  << " baselineDepthRatio=" << baselineDepthRatio
                  << " minBaselineDepthRatio=" << minBaselineDepthRatio
                  << " trackedMapPoints=" << trackedMapPoints
                  << " mode=" << (bUseORBInit ? "ORB" : "XFeat")
                  << std::endl;
    }

    float invMedianDepth;
    if(mSensor == System::IMU_MONOCULAR)
        invMedianDepth = 4.0f/medianDepth; // 4.0f
    else
        invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || trackedMapPoints<50 || baselineDepthRatio < minBaselineDepthRatio) // TODO Check, originally 100 tracks
    {
        Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_QUIET);
        if(ShouldPrintXFeatDebug(mCurrentFrame.mnId))
        {
            std::cout << "[InitMono] REJECT "
                      << "reason="
                      << ((medianDepth < 0) ? "negative_depth" :
                          (trackedMapPoints < 50) ? "few_tracked_points" : "low_baseline_depth_ratio")
                      << std::endl;
        }
        mpSystem->ResetActiveMap();
        return;
    }

    // Scale initial baseline
    Sophus::SE3f Tc2w = pKFcur->GetPose();
    Tc2w.translation() *= invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
            pMP->UpdateNormalAndDepth();
        }
    }

    if (mSensor == System::IMU_MONOCULAR)
    {
        pKFcur->mPrevKF = pKFini;
        pKFini->mNextKF = pKFcur;
        pKFcur->mpImuPreintegrated = mpImuPreintegratedFromLastKF;

        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(),pKFcur->mImuCalib);
    }


    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);
    mpLocalMapper->mFirstTs=pKFcur->mTimeStamp;

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;
    //mnLastRelocFrameId = mInitialFrame.mnId;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpAtlas->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    // Compute here initial velocity
    vector<KeyFrame*> vKFs = mpAtlas->GetAllKeyFrames();

    Sophus::SE3f deltaT = vKFs.back()->GetPose() * vKFs.front()->GetPoseInverse();
    mbVelocity = false;
    Eigen::Vector3f phi = deltaT.so3().log();

    double aux = (mCurrentFrame.mTimeStamp-mLastFrame.mTimeStamp)/(mCurrentFrame.mTimeStamp-mInitialFrame.mTimeStamp);
    phi *= aux;

    mLastFrame = Frame(mCurrentFrame);

    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;

    initID = pKFcur->mnId;
}


void Tracking::CreateMapInAtlas()
{
    mnLastInitFrameId = mCurrentFrame.mnId;
    mpAtlas->CreateNewMap();
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
        mpAtlas->SetInertialSensor();
    mbSetInit=false;

    mnInitialFrameId = mCurrentFrame.mnId+1;
    mState = NO_IMAGES_YET;

    // Restart the variable with information about the last KF
    mbVelocity = false;
    //mnLastRelocFrameId = mnLastInitFrameId; // The last relocation KF_id is the current id, because it is the new starting point for new map
    Verbose::PrintMess("First frame id in map: " + to_string(mnLastInitFrameId+1), Verbose::VERBOSITY_NORMAL);
    mbVO = false; // Init value for know if there are enough MapPoints in the last KF
    if(mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
    {
        mbReadyToInitializate = false;
    }

    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpImuPreintegratedFromLastKF)
    {
        delete mpImuPreintegratedFromLastKF;
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
    }

    if(mpLastKeyFrame)
        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

    if(mpReferenceKF)
        mpReferenceKF = static_cast<KeyFrame*>(NULL);

    mLastFrame = Frame();
    mCurrentFrame = Frame();
    mvIniMatches.clear();

    mbCreatedMap = true;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame()
{
    mbXFeatMotionHealthValid = false;
    mbXFeatMotionHealthWeak = false;
    mbXFeatMotionHealthBad = false;

    // [XFEAT_FIX_20260414] XFeat path should prefer descriptor NN matching over ORB vocabulary BoW.
    const bool bUseORB = (std::getenv("USE_ORB") != nullptr);
    if(bUseORB)
        mCurrentFrame.ComputeBoW();

    vector<MapPoint*> vpMapPointMatches;


    int nmatches = 0;
    bool bUsedLightGlueRef = false;
    XFeatLighterGlueMatcher::Stats lgRefStats;
    int lgRefPrePoseObs = 0;
    if(bUseORB)
    {
        ORBmatcher matcher(0.7, true);
        nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);
    }
    else
    {
        if(IsEnvFlagEnabled("XFEAT_USE_LIGHTGLUE_REF"))
        {
            try
            {
                static XFeatLighterGlueMatcher lightGlueMatcher;
                const float lgConf = GetEnvFloatInRange("XFEAT_LIGHTGLUE_CONF", 0.1f, 0.0f, 1.0f);
                nmatches = lightGlueMatcher.SearchByLightGlue(mpReferenceKF,
                                                              mCurrentFrame,
                                                              vpMapPointMatches,
                                                              lgConf);
                lgRefStats = lightGlueMatcher.LastStats();
                lgRefPrePoseObs = nmatches;
                bUsedLightGlueRef = true;
            }
            catch(const std::exception& e)
            {
                std::cerr << "[TrackRefKF][LightGlue] failed, falling back to XFeatMatcher: "
                          << e.what() << std::endl;
                bUsedLightGlueRef = false;
                vpMapPointMatches.clear();
                nmatches = 0;
            }
        }

        if(!bUsedLightGlueRef)
        {
        // XFeat路径: 三路候选融合（strict主干 + relaxed/proj补空位），避免“整包替换”带来的抖动。
        auto CountAssignedMatches = [](const std::vector<MapPoint*>& vMatches) -> int
        {
            int count = 0;
            for(MapPoint* pMP : vMatches)
            {
                if(pMP)
                    ++count;
            }
            return count;
        };

        auto MergeMatchesFillEmpty = [](const std::vector<MapPoint*>& vSrc,
                                        std::vector<MapPoint*>& vDst,
                                        int& added,
                                        int& conflict) -> void
        {
            added = 0;
            conflict = 0;
            const size_t n = std::min(vSrc.size(), vDst.size());
            for(size_t i = 0; i < n; ++i)
            {
                MapPoint* pCand = vSrc[i];
                if(!pCand || pCand->isBad())
                    continue;

                if(!vDst[i])
                {
                    vDst[i] = pCand;
                    ++added;
                }
                else if(vDst[i] != pCand)
                {
                    ++conflict;
                }
            }
        };

        const int kRelaxedFuseTrigger = 28;
        const int kProjFuseTrigger = 20;

        int nmatchesStrict = 0;
        int nmatchesRelaxed = 0;
        int nmatchesProj = 0;
        int relaxedAdded = 0;
        int relaxedConflict = 0;
        int projAdded = 0;
        int projConflict = 0;
        int refUniqueMP = 0;

        // 1) strict NN主干
        std::vector<MapPoint*> vpStrictMatches;
        XFeatMatcher matcherStrict(0.7f, false);
        nmatchesStrict = matcherStrict.SearchByNN(mpReferenceKF,
                                                  mCurrentFrame,
                                                  vpStrictMatches,
                                                  GetXFeatThHighRefNNStrict());
        vpMapPointMatches = vpStrictMatches;
        nmatches = CountAssignedMatches(vpMapPointMatches);

        // 2) relaxed NN补空位（仅在strict不足时触发）
        if(nmatches < kRelaxedFuseTrigger)
        {
            std::vector<MapPoint*> vpRelaxedMatches;
            XFeatMatcher matcherRelaxed(0.85f, false);
            nmatchesRelaxed = matcherRelaxed.SearchByNN(mpReferenceKF,
                                                        mCurrentFrame,
                                                        vpRelaxedMatches,
                                                        GetXFeatThHighRefNNRelaxed());
            MergeMatchesFillEmpty(vpRelaxedMatches, vpMapPointMatches, relaxedAdded, relaxedConflict);
            nmatches = CountAssignedMatches(vpMapPointMatches);
        }

        // 3) projection补空位（仍不足时触发）
        if(nmatches < kProjFuseTrigger)
        {
            mCurrentFrame.SetPose(mLastFrame.GetPose());
            mCurrentFrame.mvpMapPoints = std::vector<MapPoint*>(mCurrentFrame.N, static_cast<MapPoint*>(NULL));
            ClearXFeatMatchSources(mCurrentFrame);

            const std::vector<MapPoint*> vpRefRaw = mpReferenceKF->GetMapPointMatches();
            std::vector<MapPoint*> vpRefUnique;
            vpRefUnique.reserve(vpRefRaw.size());
            std::unordered_set<long unsigned int> seenIds;
            seenIds.reserve(vpRefRaw.size());
            for(MapPoint* pMP : vpRefRaw)
            {
                if(!pMP || pMP->isBad())
                    continue;
                if(seenIds.insert(pMP->mnId).second)
                    vpRefUnique.push_back(pMP);
            }
            refUniqueMP = static_cast<int>(vpRefUnique.size());

            for(MapPoint* pMP : vpRefUnique)
                mCurrentFrame.isInFrustum(pMP, 0.5f);

            XFeatMatcher matcherProj(0.8f, false);
            nmatchesProj = matcherProj.SearchByProjection(mCurrentFrame,
                                                          vpRefUnique,
                                                          3.0f,
                                                          false,
                                                          50.0f,
                                                          GetXFeatThHighRefProjectionFallback());

            MergeMatchesFillEmpty(mCurrentFrame.mvpMapPoints, vpMapPointMatches, projAdded, projConflict);
            nmatches = CountAssignedMatches(vpMapPointMatches);
        }
        }
    }

    // if (std::getenv("USE_ORB") == nullptr)
    // {
    //     nmatches = matcher.SearchByNN(mpReferenceKF, mCurrentFrame, vpMapPointMatches);
    // }
    // else
    // {   
    //     nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);
    // }

    if(nmatches<15)
    {
        if(bUsedLightGlueRef && ShouldPrintXFeatDebug(mCurrentFrame.mnId))
        {
            std::cout << "[TrackRefKF][LightGlue] frame_id=" << lgRefStats.frame_id
                      << " ref_kf_id=" << lgRefStats.ref_kf_id
                      << " N_ref=" << lgRefStats.N_ref
                      << " N_cur=" << lgRefStats.N_cur
                      << " lg_raw_matches=" << lgRefStats.lg_raw_matches
                      << " mp_valid=" << lgRefStats.mp_valid
                      << " one_to_one=" << lgRefStats.one_to_one
                      << " pre_pose_obs=" << lgRefPrePoseObs
                      << " post_pose_inliers=0"
                      << " outlier_ratio=1.000000"
                      << std::endl;
        }
        cout << "TRACK_REF_KF: Less than 15 matches!!\n";
        return false;
    }

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    ClearXFeatMatchSources(mCurrentFrame);
    MarkAssignedMatches(mCurrentFrame, bUsedLightGlueRef ? kXFeatMatchReferenceLightGlue : kXFeatMatchReference);
    mCurrentFrame.SetPose(mLastFrame.GetPose());

    //mCurrentFrame.PrintPointDistribution();


    // cout << " TrackReferenceKeyFrame mLastFrame.mTcw:  " << mLastFrame.mTcw << endl;
    Optimizer::PoseOptimization(&mCurrentFrame);
    int assoc_cnt = 0, inlier_cnt = 0;
    for (int i = 0; i < mCurrentFrame.N; ++i)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            assoc_cnt++;
            if (!mCurrentFrame.mvbOutlier[i]) inlier_cnt++;
        }
    }

    if(bUsedLightGlueRef && ShouldPrintXFeatDebug(mCurrentFrame.mnId))
    {
        const float outlierRatio = assoc_cnt > 0
            ? static_cast<float>(assoc_cnt - inlier_cnt) / static_cast<float>(assoc_cnt)
            : 0.0f;
        std::cout << "[TrackRefKF][LightGlue] frame_id=" << lgRefStats.frame_id
                  << " ref_kf_id=" << lgRefStats.ref_kf_id
                  << " N_ref=" << lgRefStats.N_ref
                  << " N_cur=" << lgRefStats.N_cur
                  << " lg_raw_matches=" << lgRefStats.lg_raw_matches
                  << " mp_valid=" << lgRefStats.mp_valid
                  << " one_to_one=" << lgRefStats.one_to_one
                  << " pre_pose_obs=" << lgRefPrePoseObs
                  << " post_pose_inliers=" << inlier_cnt
                  << " outlier_ratio=" << outlierRatio
                  << std::endl;
    }

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        //if(i >= mCurrentFrame.Nleft) break;
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                EnsureXFeatMatchSourceSize(mCurrentFrame);
                mCurrentFrame.mvXFeatMatchSource[static_cast<size_t>(i)] = kXFeatMatchUnknown;
                if(i < mCurrentFrame.Nleft){
                    pMP->mbTrackInView = false;
                }
                else{
                    pMP->mbTrackInViewR = false;
                }
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        return true;
    else
        return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    Sophus::SE3f Tlr = mlRelativeFramePoses.back();
    mLastFrame.SetPose(Tlr * pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    const int Nfeat = mLastFrame.Nleft == -1? mLastFrame.N : mLastFrame.Nleft;
    vDepthIdx.reserve(Nfeat);
    for(int i=0; i<Nfeat;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
            bCreateNew = true;

        if(bCreateNew)
        {
            Eigen::Vector3f x3D;

            if(mLastFrame.Nleft == -1){
                mLastFrame.UnprojectStereo(i, x3D);
            }
            else{
                x3D = mLastFrame.UnprojectStereoFishEye(i);
            }

            MapPoint* pNewMP = new MapPoint(x3D,mpAtlas->GetCurrentMap(),&mLastFrame,i);
            pNewMP->SetCreationSource(MapPoint::kCreationSourceTrackingStereo);
            SetMappingFeatureQualityFromFrame(pNewMP, mLastFrame, 0, i);
            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;

    }
}

bool Tracking::TrackWithMotionModel()
{
    mbXFeatMotionHealthValid = false;
    mbXFeatMotionHealthWeak = false;
    mbXFeatMotionHealthBad = false;
    mnXFeatMotionHealthFrameId = mCurrentFrame.mnId;
    mnXFeatMotionHealthInliers = 0;
    mnXFeatMotionHealthAssociations = 0;
    mXFeatMotionHealthOutlierRatio = 0.0f;

    const bool bUseORB = (std::getenv("USE_ORB") != nullptr);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    if (mpAtlas->isImuInitialized() && (mCurrentFrame.mnId>mnLastRelocFrameId+mnFramesToResetIMU))
    {
        // Predict state with IMU if it is initialized and it doesnt need reset
        PredictStateIMU();
        return true;
    }
    else
    {
        mCurrentFrame.SetPose(mVelocity * mLastFrame.GetPose());
    }

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
    ClearXFeatMatchSources(mCurrentFrame);

    // Project points seen in previous frame
    int th;

    if(mSensor==System::STEREO)
        th=7;
    else
        th=15;

    int nmatches = 0;
    bool bUsedLightGlueMotion = false;
    XFeatLighterGlueMatcher::Stats lgMotionStats;
    int lgMotionPrePoseObs = 0;
    std::vector<MapPoint*> vpLightGlueMotionMatches;
    auto CountAssignedMatches = [](const std::vector<MapPoint*>& vMatches) -> int
    {
        int count = 0;
        for(MapPoint* pMP : vMatches)
        {
            if(pMP)
                ++count;
        }
        return count;
    };
    auto RestorePreservedMatches = [&](const std::vector<MapPoint*>& preservedMatches)
    {
        const size_t n = std::min(preservedMatches.size(), mCurrentFrame.mvpMapPoints.size());
        for(size_t i = 0; i < n; ++i)
        {
            if(preservedMatches[i])
                mCurrentFrame.mvpMapPoints[i] = preservedMatches[i];
        }
    };

    if(bUseORB)
    {
        ORBmatcher matcher(0.9, true);
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);
    }
    else
    {
        const bool bUseLightGlueMotion = IsEnvFlagEnabled("XFEAT_USE_LIGHTGLUE_MOTION");
        const bool bProjectionFirstLightGlueMotion = bUseLightGlueMotion && UseXFeatLightGlueMotionProjectionFirst();
        if(bProjectionFirstLightGlueMotion)
        {
            XFeatMatcher matcher(0.9f, false);
            nmatches = matcher.SearchByProjection(mCurrentFrame,
                                                  mLastFrame,
                                                  th,
                                                  mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR,
                                                  GetXFeatThHighMotionProjection());
        }

        if(bUseLightGlueMotion &&
           (!bProjectionFirstLightGlueMotion ||
            nmatches < GetXFeatLightGlueMotionFallbackMinMatches()))
        {
            try
            {
                static XFeatLighterGlueMatcher lightGlueMatcher;
                const float lgConf = GetEnvFloatInRange("XFEAT_LIGHTGLUE_CONF", 0.1f, 0.0f, 1.0f);
                nmatches = lightGlueMatcher.SearchByLightGlue(mLastFrame,
                                                              mCurrentFrame,
                                                              vpLightGlueMotionMatches,
                                                              lgConf);
                lgMotionStats = lightGlueMatcher.LastStats();
                lgMotionPrePoseObs = nmatches;
                mCurrentFrame.mvpMapPoints = vpLightGlueMotionMatches;
                bUsedLightGlueMotion = true;
            }
            catch(const std::exception& e)
            {
                std::cerr << "[TrackMotion][LightGlue] failed, falling back to XFeatMatcher: "
                          << e.what() << std::endl;
                bUsedLightGlueMotion = false;
                vpLightGlueMotionMatches.clear();
                if(!bProjectionFirstLightGlueMotion)
                {
                    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
                    ClearXFeatMatchSources(mCurrentFrame);
                    nmatches = 0;
                }
                else
                {
                    nmatches = CountAssignedMatches(mCurrentFrame.mvpMapPoints);
                }
            }
        }

        if(!bUsedLightGlueMotion && !bProjectionFirstLightGlueMotion)
        {
            XFeatMatcher matcher(0.9f, false);
            nmatches = matcher.SearchByProjection(mCurrentFrame,
                                                  mLastFrame,
                                                  th,
                                                  mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR,
                                                  GetXFeatThHighMotionProjection());
        }
        else if(bUsedLightGlueMotion && nmatches < 20)
        {
            XFeatMatcher matcher(0.9f, false);
            matcher.SearchByProjection(mCurrentFrame,
                                       mLastFrame,
                                       th,
                                       mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR,
                                       GetXFeatThHighMotionProjection());
            RestorePreservedMatches(vpLightGlueMotionMatches);
            nmatches = CountAssignedMatches(mCurrentFrame.mvpMapPoints);
        }
    }

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        Verbose::PrintMess("Not enough matches, wider window search!!", Verbose::VERBOSITY_NORMAL);
        if(!bUsedLightGlueMotion)
        {
            fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
            ClearXFeatMatchSources(mCurrentFrame);
        }

        if(bUseORB)
        {
            ORBmatcher matcher(0.9, true);
            nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);
        }
        else
        {
            const std::vector<MapPoint*> vPreservedMatches = mCurrentFrame.mvpMapPoints;
            XFeatMatcher matcher(0.9f, false);
            nmatches = matcher.SearchByProjection(mCurrentFrame,
                                                  mLastFrame,
                                                  2*th,
                                                  mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR,
                                                  GetXFeatThHighMotionProjection());
            if(bUsedLightGlueMotion)
            {
                RestorePreservedMatches(vPreservedMatches);
                nmatches = CountAssignedMatches(mCurrentFrame.mvpMapPoints);
            }
        }
        Verbose::PrintMess("Matches with wider search: " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);

    }

    if(bUsedLightGlueMotion)
        lgMotionPrePoseObs = nmatches;

    ClearXFeatMatchSources(mCurrentFrame);
    if(bUsedLightGlueMotion)
    {
        EnsureXFeatMatchSourceSize(mCurrentFrame);
        const size_t n = std::min(mCurrentFrame.mvpMapPoints.size(), mCurrentFrame.mvXFeatMatchSource.size());
        for(size_t i = 0; i < n; ++i)
        {
            if(!mCurrentFrame.mvpMapPoints[i])
                continue;
            if(i < vpLightGlueMotionMatches.size() && vpLightGlueMotionMatches[i] == mCurrentFrame.mvpMapPoints[i])
                mCurrentFrame.mvXFeatMatchSource[i] = kXFeatMatchMotionLightGlue;
            else
                mCurrentFrame.mvXFeatMatchSource[i] = kXFeatMatchMotionProjection;
        }
    }
    else
    {
        MarkAssignedMatches(mCurrentFrame, kXFeatMatchMotionProjection);
    }

    if(nmatches<20)
    {
        Verbose::PrintMess("Not enough matches!!", Verbose::VERBOSITY_NORMAL);
        if(bUsedLightGlueMotion && ShouldPrintXFeatDebug(mCurrentFrame.mnId))
        {
            std::cout << "[TrackMotion][LightGlue]"
                      << " frame_id=" << lgMotionStats.frame_id
                      << " last_frame_id=" << lgMotionStats.last_frame_id
                      << " N_last=" << lgMotionStats.N_last
                      << " N_cur=" << lgMotionStats.N_cur
                      << " lg_raw_matches=" << lgMotionStats.lg_raw_matches
                      << " mp_valid=" << lgMotionStats.mp_valid
                      << " one_to_one=" << lgMotionStats.one_to_one
                      << " proj_gate_keep=" << lgMotionStats.proj_gate_keep
                      << " pre_pose_obs=" << lgMotionPrePoseObs
                      << " post_pose_inliers=0"
                      << " outlier_ratio=1.000000"
                      << std::endl;
        }
        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            return true;
        else
            return false;
    }

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);
    //调试: PoseOptimization 后的匹配统计。
    int assoc_cnt = 0, inlier_cnt = 0;
    for (int i = 0; i < mCurrentFrame.N; ++i)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            assoc_cnt++;
            if (!mCurrentFrame.mvbOutlier[i])
                inlier_cnt++;
        }
    }
    const float outlierRatio = assoc_cnt > 0
        ? static_cast<float>(assoc_cnt - inlier_cnt) / static_cast<float>(assoc_cnt)
        : 0.0f;

    if(bUsedLightGlueMotion)
    {
        const int weakMinInliers = GetXFeatMotionWeakMinInliers();
        const int badMinInliers = GetXFeatMotionBadMinInliers();
        const float weakMaxOutlier = GetXFeatMotionWeakMaxOutlier();
        const float badMaxOutlier = GetXFeatMotionBadMaxOutlier();
        mbXFeatMotionHealthValid = true;
        mnXFeatMotionHealthFrameId = mCurrentFrame.mnId;
        mnXFeatMotionHealthInliers = inlier_cnt;
        mnXFeatMotionHealthAssociations = assoc_cnt;
        mXFeatMotionHealthOutlierRatio = outlierRatio;
        mbXFeatMotionHealthWeak = inlier_cnt < weakMinInliers || outlierRatio > weakMaxOutlier;
        mbXFeatMotionHealthBad = inlier_cnt < badMinInliers || outlierRatio > badMaxOutlier;

        if(ShouldPrintXFeatDebug(mCurrentFrame.mnId))
        {
            const std::ios::fmtflags oldFlags = std::cout.flags();
            const std::streamsize oldPrecision = std::cout.precision();
            std::cout << std::fixed << std::setprecision(6)
                      << "[TrackMotion][Health]"
                      << " frame_id=" << mCurrentFrame.mnId
                      << " guard_enabled=" << (IsXFeatMotionHealthGuardEnabled() ? 1 : 0)
                      << " assoc=" << assoc_cnt
                      << " inliers=" << inlier_cnt
                      << " outlier_ratio=" << outlierRatio
                      << " weak=" << (mbXFeatMotionHealthWeak ? 1 : 0)
                      << " bad=" << (mbXFeatMotionHealthBad ? 1 : 0)
                      << " weak_min_inliers=" << weakMinInliers
                      << " bad_min_inliers=" << badMinInliers
                      << " weak_max_outlier=" << weakMaxOutlier
                      << " bad_max_outlier=" << badMaxOutlier
                      << std::endl;
            std::cout.flags(oldFlags);
            std::cout.precision(oldPrecision);
        }
    }

    const bool bLowMotionSupportDiag =
        IsXFeatFeatureDiagEnabled() && inlier_cnt < 80;
    if(bUsedLightGlueMotion &&
       (ShouldPrintXFeatDebug(mCurrentFrame.mnId) || bLowMotionSupportDiag))
    {
        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6)
                  << "[TrackMotion][LightGlue]"
                  << " frame_id=" << lgMotionStats.frame_id
                  << " last_frame_id=" << lgMotionStats.last_frame_id
                  << " N_last=" << lgMotionStats.N_last
                  << " N_cur=" << lgMotionStats.N_cur
                  << " lg_raw_matches=" << lgMotionStats.lg_raw_matches
                  << " mp_valid=" << lgMotionStats.mp_valid
                  << " one_to_one=" << lgMotionStats.one_to_one
                  << " proj_gate_keep=" << lgMotionStats.proj_gate_keep
                  << " pre_pose_obs=" << lgMotionPrePoseObs
                  << " post_pose_inliers=" << inlier_cnt
                  << " outlier_ratio=" << outlierRatio
                  << std::endl;
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                EnsureXFeatMatchSourceSize(mCurrentFrame);
                mCurrentFrame.mvXFeatMatchSource[static_cast<size_t>(i)] = kXFeatMatchUnknown;
                if(i < mCurrentFrame.Nleft){
                    pMP->mbTrackInView = false;
                }
                else{
                    pMP->mbTrackInViewR = false;
                }
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        return true;
    else
        return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap()
{

    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    mTrackedFr++;

    UpdateLocalMap();
    const size_t nLocalKFs = mvpLocalKeyFrames.size();
    const size_t nLocalMPs = mvpLocalMapPoints.size();
    int assoc_before_search = 0;
    for (int i = 0; i < mCurrentFrame.N; ++i)
    {
        if (mCurrentFrame.mvpMapPoints[i])
            assoc_before_search++;
    }

    SearchLocalPoints();
    int assoc_before_opt = 0;
    MapPointQualityStats mpQualityBeforeOpt;
    for (int i = 0; i < mCurrentFrame.N; ++i)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            assoc_before_opt++;
            float depth = -1.0f;
            if(i >= 0 && i < static_cast<int>(mCurrentFrame.mvDepth.size()) && mCurrentFrame.mvDepth[i] > 0.0f)
                depth = mCurrentFrame.mvDepth[i];
            else
                depth = mCurrentFrame.mvpMapPoints[i]->mTrackDepth;
                AccumulateMapPointQuality(mpQualityBeforeOpt,
                                          mCurrentFrame.mvpMapPoints[i],
                                          mCurrentFrame,
                                          depth,
                                          mThDepth);
        }
    }
    LogTrackLocalMapKFStructureDiagnostics(mCurrentFrame,
                                           mpReferenceKF,
                                           mpLastKeyFrame,
                                           mvpLocalKeyFrames,
                                           mvpLocalMapPoints,
                                           mmLocalKFProvenance,
                                           mmLocalMPProvenance,
                                           assoc_before_search,
                                           assoc_before_opt,
                                           mnLocalMapVotedKFCandidates,
                                           mnLocalMapVotedKFKept,
                                           mnLocalMapVotedKFVisibilityKept,
                                           mThDepth);
    StereoDepthConstraintStats localStereoBeforeOpt;
    AccumulateStereoDepthConstraintStats(mCurrentFrame, mThDepth, false, localStereoBeforeOpt);
    const Sophus::SE3f TcwPrior = mCurrentFrame.GetPose();

    // TOO check outliers before PO
    int aux1 = 0, aux2=0;
    for(int i=0; i<mCurrentFrame.N; i++)
        if( mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if(mCurrentFrame.mvbOutlier[i])
                aux2++;
        }

    int inliers;
    if (!mpAtlas->isImuInitialized())
    {
        Optimizer::PoseOptimization(&mCurrentFrame);
    }

    else
    {
        if(mCurrentFrame.mnId<=mnLastRelocFrameId+mnFramesToResetIMU)
        {
            Verbose::PrintMess("TLM: PoseOptimization ", Verbose::VERBOSITY_DEBUG);
            Optimizer::PoseOptimization(&mCurrentFrame);
        }
        else
        {
            // if(!mbMapUpdated && mState == OK) //  && (mnMatchesInliers>30))
            if(!mbMapUpdated) //  && (mnMatchesInliers>30))
            {
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastFrame ", Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
            else
            {
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastKeyFrame ", Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
        }
    }

    aux1 = 0, aux2 = 0;
    MapPointQualityStats mpQualityInlierAfterOpt;
    MapPointQualityStats mpQualityOutlierAfterOpt;
    for(int i=0; i<mCurrentFrame.N; i++)
        if( mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            float depth = -1.0f;
            if(i >= 0 && i < static_cast<int>(mCurrentFrame.mvDepth.size()) && mCurrentFrame.mvDepth[i] > 0.0f)
                depth = mCurrentFrame.mvDepth[i];
            else
                depth = mCurrentFrame.mvpMapPoints[i]->mTrackDepth;

            if(mCurrentFrame.mvbOutlier[i])
            {
                aux2++;
                AccumulateMapPointQuality(mpQualityOutlierAfterOpt,
                                          mCurrentFrame.mvpMapPoints[i],
                                          mCurrentFrame,
                                          depth,
                                          mThDepth);
            }
            else
            {
                AccumulateMapPointQuality(mpQualityInlierAfterOpt,
                                          mCurrentFrame.mvpMapPoints[i],
                                          mCurrentFrame,
                                          depth,
                                          mThDepth);
            }
        }
    const int assoc_after_opt = aux1;
    const int outlier_after_opt = aux2;
    const int inlier_after_opt = assoc_after_opt - outlier_after_opt;
    const float outlier_ratio_after_opt = assoc_after_opt > 0
        ? static_cast<float>(outlier_after_opt) / static_cast<float>(assoc_after_opt)
        : 0.0f;
    StereoDepthConstraintStats localStereoAfterOpt;
    AccumulateStereoDepthConstraintStats(mCurrentFrame, mThDepth, true, localStereoAfterOpt);
    RecordTrackLocalMapCloseReuseOutcomes(mCurrentFrame);

    if(ShouldPrintXFeatDebug(mCurrentFrame.mnId))
    {
        LogTrackLocalMapConstraintSourceDiagnostics(mCurrentFrame);
        LogTrackLocalMapJointSourceDiagnostics(mCurrentFrame);
        LogTrackLocalMapInlierMix(mCurrentFrame);

        if(IsXFeatDeepDiagEnabled())
        {
            LogTrackLocalMapCloseReuseOutcomeDiagnostics(mCurrentFrame);
            LogTrackLocalMapCloseReuseContributionOutcomeDiagnostics(mCurrentFrame, mmLocalMPProvenance);
            LogTrackLocalMapCloseReuseCohortAliveDiagnostics(mCurrentFrame, mpAtlas->GetCurrentMap());
            LogTrackLocalMapMappingRepeatabilityCohortDiagnostics(mCurrentFrame, mpAtlas->GetCurrentMap());
            LogTrackLocalMapConstraintGridDiagnostics(mCurrentFrame);
            LogTrackLocalMapInlierGeometryDiagnostics(mCurrentFrame);
            LogTrackLocalMapMappingFeatureOutcomeDiagnostics(mCurrentFrame);
            LogTrackLocalMapLocalProjMatcherQualityFineDiagnostics(mCurrentFrame);
            LogTrackLocalMapMapPointSourceDiagnostics(mCurrentFrame);
            LogTrackLocalMapTriangulationQualityDiagnostics(mCurrentFrame);
            LogTrackLocalMapTriangulationMultiViewDiagnostics(mCurrentFrame);
            LogTrackLocalMapTriangulationLocalProjHistoryDiagnostics(mCurrentFrame);
            LogTrackLocalMapLocalProjMatcherQualityDiagnostics(mCurrentFrame);
            LogTrackLocalMapLocalProjPoseContextDiagnostics(mCurrentFrame,
                                                            assoc_before_search,
                                                            assoc_before_opt - assoc_before_search,
                                                            mbMapUpdated);
            LogTrackLocalMapProjectionPriorShiftDiagnostics(mCurrentFrame,
                                                            TcwPrior,
                                                            assoc_before_search,
                                                            assoc_before_opt - assoc_before_search,
                                                            inlier_after_opt,
                                                            mbMapUpdated);
            LogTrackLocalMapLocalMapProvenanceDiagnostics(mCurrentFrame, mmLocalMPProvenance);
            LogTrackLocalMapLocalProjPriorityDiagnostics(mCurrentFrame, mmLocalMPProvenance);
        }
    }
    if(ShouldPrintXFeatMeasurementCalib(mCurrentFrame.mnId))
    {
        LogTrackLocalMapXFeatMeasurementCalibDiagnostics(mCurrentFrame);
    }
    RecordTrackLocalMapTriangulationLocalProjHistoryOutcomes(mCurrentFrame);

    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
            {
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                EnsureXFeatMatchSourceSize(mCurrentFrame);
                mCurrentFrame.mvXFeatMatchSource[static_cast<size_t>(i)] = kXFeatMatchUnknown;
            }
        }
    }
    int assoc_after_discard = 0;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
            assoc_after_discard++;
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    mpLocalMapper->mnMatchesInliers=mnMatchesInliers;
    const bool bUseORB = (std::getenv("USE_ORB") != nullptr);
    const int recentRelocInlierTh = bUseORB ? 50 : GetXFeatRecentRelocInlierThreshold();
    const bool bRecentRelocWindow = mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames;
    auto LogTrackLocalMapSummary = [&](const bool bReturnValue, const int successThreshold, const char* decision)
    {
        const bool bFailureDiag = !bReturnValue && IsXFeatFeatureDiagEnabled();
        if(!ShouldPrintXFeatDebug(mCurrentFrame.mnId) && !bFailureDiag)
            return;

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6)
                  << "[TrackLocalMap][Diag]"
                  << " frame_id=" << mCurrentFrame.mnId
                  << " state=" << static_cast<int>(mState)
                  << " mode=" << (bUseORB ? "ORB" : "XFeat")
                  << " local_kfs=" << nLocalKFs
                  << " local_mps=" << nLocalMPs
                  << " assoc_before_search=" << assoc_before_search
                  << " assoc_before_opt=" << assoc_before_opt
                  << " assoc_added_by_local=" << (assoc_before_opt - assoc_before_search)
                  << " assoc_after_opt=" << assoc_after_opt
                  << " inlier_after_opt=" << inlier_after_opt
                  << " outlier_after_opt=" << outlier_after_opt
                  << " outlier_ratio_after_opt=" << outlier_ratio_after_opt
                  << " assoc_after_discard=" << assoc_after_discard
                  << " mnMatchesInliers=" << mnMatchesInliers
                  << " success_threshold=" << successThreshold
                  << " recent_reloc_window=" << (bRecentRelocWindow ? 1 : 0)
                  << " mbOnlyTracking=" << (mbOnlyTracking ? 1 : 0)
                  << " mbMapUpdated=" << (mbMapUpdated ? 1 : 0)
                  << " return=" << (bReturnValue ? 1 : 0)
                  << " decision=" << decision
                  << std::endl;

        if(IsXFeatFeatureDiagEnabled() && IsXFeatDeepDiagEnabled())
        {
            std::cout << std::fixed << std::setprecision(6)
                      << "[TrackLocalMap][MPQuality]"
                      << " frame_id=" << mCurrentFrame.mnId
                      << " mThDepth=" << mThDepth;
            AppendMapPointQualityStats(std::cout, "before", mpQualityBeforeOpt);
            AppendMapPointQualityStats(std::cout, "inlier", mpQualityInlierAfterOpt);
            AppendMapPointQualityStats(std::cout, "outlier", mpQualityOutlierAfterOpt);
            std::cout << std::endl;
            std::cout << std::fixed << std::setprecision(6)
                      << "[TrackLocalMap][StereoDepth]"
                      << " frame_id=" << mCurrentFrame.mnId
                      << " mThDepth=" << mThDepth;
            AppendStereoDepthConstraintStats(std::cout, "before", localStereoBeforeOpt);
            AppendStereoDepthConstraintStats(std::cout, "after", localStereoAfterOpt);
            std::cout << std::endl;
        }
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    };
    if(bRecentRelocWindow && mnMatchesInliers<recentRelocInlierTh)
    {
        //调试: 失败摘要限流。`XFEAT_DEBUG=1` 时由 XFEAT_DIAG_INTERVAL 控制频率，默认每10帧最多1条。
        static bool sTrackLocalFailLogInitialized = false;
        static long unsigned int sLastTrackLocalFailLogFrame = 0;
        const bool debugLogDue = ShouldPrintXFeatDebug(mCurrentFrame.mnId);
        const bool summaryLogDue = !IsXFeatDebugEnabled() &&
            (!sTrackLocalFailLogInitialized || mCurrentFrame.mnId > sLastTrackLocalFailLogFrame + 10);
        if(debugLogDue || summaryLogDue)
        {
            std::cout << "[TrackLocalMap] FAIL "
              << "frame=" << mCurrentFrame.mnId
              << " mnMatchesInliers=" << mnMatchesInliers
              << " threshold=" << recentRelocInlierTh
              << " mode=" << (bUseORB ? "ORB" : "XFeat")
              << std::endl;
            sLastTrackLocalFailLogFrame = mCurrentFrame.mnId;
            sTrackLocalFailLogInitialized = true;
        }
        LogTrackLocalMapSummary(false, recentRelocInlierTh, "fail_recent_reloc");
        return false;
    }


    if((mnMatchesInliers>10)&&(mState==RECENTLY_LOST))
    {
        LogTrackLocalMapSummary(true, 10, "recently_lost_recovered");
        return true;
    }


    if (mSensor == System::IMU_MONOCULAR)
    {
        const int successThreshold = mpAtlas->isImuInitialized() ? 15 : 50;
        if(mnMatchesInliers<successThreshold)
        {
            LogTrackLocalMapSummary(false, successThreshold, "fail_imu_mono");
            return false;
        }
        else
        {
            LogTrackLocalMapSummary(true, successThreshold, "ok_imu_mono");
            return true;
        }
    }
    else if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
    {
        if(mnMatchesInliers<15)
        {
            LogTrackLocalMapSummary(false, 15, "fail_imu_stereo_rgbd");
            return false;
        }
        else
        {
            LogTrackLocalMapSummary(true, 15, "ok_imu_stereo_rgbd");
            return true;
        }
    }
    else
    {
        const int successThreshold = 30;
        if(mnMatchesInliers<successThreshold)
        {
            LogTrackLocalMapSummary(false, successThreshold, "fail_visual");
            return false;
        }
        else
        {
            LogTrackLocalMapSummary(true, successThreshold, "ok_visual");
            return true;
        }
    }
}

bool Tracking::NeedNewKeyFrame()
{
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mpAtlas->GetCurrentMap()->isImuInitialized())
    {
        if (mSensor == System::IMU_MONOCULAR && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
            return true;
        else if ((mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
            return true;
        else
            return false;
    }

    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) {
        /*if(mSensor == System::MONOCULAR)
        {
            std::cout << "NeedNewKeyFrame: localmap stopped" << std::endl;
        }*/
        return false;
    }

    const int nKFs = mpAtlas->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
    {
        return false;
    }

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;

    if(mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR)
    {
        int N = (mCurrentFrame.Nleft == -1) ? mCurrentFrame.N : mCurrentFrame.Nleft;
        for(int i =0; i<N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;

            }
        }
        //Verbose::PrintMess("[NEEDNEWKF]-> closed points: " + to_string(nTrackedClose) + "; non tracked closed points: " + to_string(nNonTrackedClose), Verbose::VERBOSITY_NORMAL);// Verbose::VERBOSITY_DEBUG);
    }

    bool bNeedToInsertClose;
    bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    /*int nClosedPoints = nTrackedClose + nNonTrackedClose;
    const int thStereoClosedPoints = 15;
    if(nClosedPoints < thStereoClosedPoints && (mSensor==System::STEREO || mSensor==System::IMU_STEREO))
    {
        //Pseudo-monocular, there are not enough close points to be confident about the stereo observations.
        thRefRatio = 0.9f;
    }*/

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    if(mpCamera2) thRefRatio = 0.75f;

    if(mSensor==System::IMU_MONOCULAR)
    {
        if(mnMatchesInliers>350) // Points tracked from the local map
            thRefRatio = 0.75f;
        else
            thRefRatio = 0.90f;
    }

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = ((mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames) && bLocalMappingIdle); //mpLocalMapper->KeyframesInQueue() < 2);
    //Condition 1c: tracking is weak
    const bool c1c = mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR && mSensor!=System::IMU_STEREO && mSensor!=System::IMU_RGBD && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = (((mnMatchesInliers<nRefMatches*thRefRatio || bNeedToInsertClose)) && mnMatchesInliers>15);

    //std::cout << "NeedNewKF: c1a=" << c1a << "; c1b=" << c1b << "; c1c=" << c1c << "; c2=" << c2 << std::endl;
    // Temporal condition for Inertial cases
    bool c3 = false;
    if(mpLastKeyFrame)
    {
        if (mSensor==System::IMU_MONOCULAR)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
        else if (mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
    }

    bool c4 = false;
    if ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && (mSensor == System::IMU_MONOCULAR)) // MODIFICATION_2, originally ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && ((mSensor == System::IMU_MONOCULAR)))
        c4=true;
    else
        c4=false;

    const bool bOriginalNeedNewKF = (((c1a||c1b||c1c) && c2)||c3 ||c4);

    struct XFeatKFGuardMetrics
    {
        int inlierAssoc = 0;
        int stableInlierAssoc = 0;
        int refOverlapInlier = 0;
        int stableRefOverlapInlier = 0;
        int nonTrackedCloseHighConf = 0;
        int trackedCloseCells = 0;
        int nonTrackedCloseCells = 0;
        int novelCloseCells = 0;
        int highConfNovelCloseCells = 0;
        FeatureSpatialStats inlierSpatial;
        FeatureSpatialStats stableSpatial;
        float stableInlierShare = 0.0f;
        float refOverlapShare = 0.0f;
        float stableRefOverlapShare = 0.0f;
        float novelCloseCellShare = 0.0f;
        bool eligible = false;
        int weakInlierThreshold = 0;
        bool weakSupport = false;
        bool stableSupportOk = false;
        bool closeCoverageOk = false;
        bool novelCloseStrong = false;
        bool closeInsert = false;
    };

    auto ComputeXFeatKFGuardMetrics = [&]() -> XFeatKFGuardMetrics
    {
        XFeatKFGuardMetrics metrics;
        metrics.eligible =
            std::getenv("USE_ORB") == nullptr &&
            (mSensor == System::STEREO || mSensor == System::RGBD);
        metrics.weakInlierThreshold =
            std::max(120, static_cast<int>(0.60f * static_cast<float>(nRefMatches)));
        if(!metrics.eligible && !IsXFeatFeatureDiagEnabled())
        {
            return metrics;
        }

        std::vector<int> inlierIndices;
        std::vector<int> stableInlierIndices;
        const int nFramePoints =
            std::min(mCurrentFrame.N, static_cast<int>(mCurrentFrame.mvpMapPoints.size()));
        inlierIndices.reserve(static_cast<size_t>(std::max(0, nFramePoints)));
        stableInlierIndices.reserve(static_cast<size_t>(std::max(0, nFramePoints)));

        for(int i = 0; i < nFramePoints; ++i)
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP || pMP->isBad())
                continue;
            if(i < static_cast<int>(mCurrentFrame.mvbOutlier.size()) && mCurrentFrame.mvbOutlier[i])
                continue;

            ++metrics.inlierAssoc;
            inlierIndices.push_back(i);

            bool bObservedByReference = false;
            if(mpReferenceKF)
            {
                bObservedByReference = pMP->IsInKeyFrame(mpReferenceKF);
                if(bObservedByReference)
                    ++metrics.refOverlapInlier;
            }

            const float foundRatio = pMP->GetFoundRatio();
            const bool bStable =
                pMP->Observations() >= 3 &&
                std::isfinite(foundRatio) &&
                foundRatio >= 0.35f;
            if(bStable)
            {
                ++metrics.stableInlierAssoc;
                stableInlierIndices.push_back(i);
                if(bObservedByReference)
                    ++metrics.stableRefOverlapInlier;
            }
        }

        metrics.inlierSpatial = ComputeFeatureSpatialStats(mCurrentFrame, inlierIndices);
        metrics.stableSpatial = ComputeFeatureSpatialStats(mCurrentFrame, stableInlierIndices);

        const int gridCols = GetXFeatFeatureDiagGridCols();
        const int gridRows = GetXFeatFeatureDiagGridRows();
        const int gridCells = std::max(1, gridCols * gridRows);
        std::vector<unsigned char> trackedCloseCell(static_cast<size_t>(gridCells), 0);
        std::vector<unsigned char> nonTrackedCloseCell(static_cast<size_t>(gridCells), 0);
        std::vector<unsigned char> highConfNonTrackedCloseCell(static_cast<size_t>(gridCells), 0);
        const float spanX = std::max(1.0f, mCurrentFrame.mnMaxX - mCurrentFrame.mnMinX);
        const float spanY = std::max(1.0f, mCurrentFrame.mnMaxY - mCurrentFrame.mnMinY);
        auto ResolveCell = [&](const int idx) -> int
        {
            cv::KeyPoint kp;
            if(!GetFrameKeyPointByIndex(mCurrentFrame, idx, kp))
                return -1;
            const int cellX = ClampInt(
                static_cast<int>(((kp.pt.x - mCurrentFrame.mnMinX) / spanX) * static_cast<float>(gridCols)),
                0,
                gridCols - 1);
            const int cellY = ClampInt(
                static_cast<int>(((kp.pt.y - mCurrentFrame.mnMinY) / spanY) * static_cast<float>(gridRows)),
                0,
                gridRows - 1);
            return cellY * gridCols + cellX;
        };

        const int nLeft = (mCurrentFrame.Nleft == -1) ? mCurrentFrame.N : mCurrentFrame.Nleft;
        const int nClosePoints = std::min(nLeft, static_cast<int>(mCurrentFrame.mvDepth.size()));
        for(int i = 0; i < nClosePoints; ++i)
        {
            const float depth = mCurrentFrame.mvDepth[i];
            if(!(std::isfinite(depth) && depth > 0.0f && depth < mThDepth))
                continue;

            const int cell = ResolveCell(i);
            if(cell < 0)
                continue;

            const bool bTracked =
                i < static_cast<int>(mCurrentFrame.mvpMapPoints.size()) &&
                mCurrentFrame.mvpMapPoints[i] &&
                (i >= static_cast<int>(mCurrentFrame.mvbOutlier.size()) || !mCurrentFrame.mvbOutlier[i]);
            if(bTracked)
            {
                trackedCloseCell[static_cast<size_t>(cell)] = 1;
                continue;
            }

            nonTrackedCloseCell[static_cast<size_t>(cell)] = 1;
            const bool bHighConf =
                i < static_cast<int>(mCurrentFrame.mvStereoBirthQuality.size()) &&
                mCurrentFrame.mvStereoBirthQuality[static_cast<size_t>(i)] == Frame::kStereoBirthCloseHighConf;
            if(bHighConf)
            {
                ++metrics.nonTrackedCloseHighConf;
                highConfNonTrackedCloseCell[static_cast<size_t>(cell)] = 1;
            }
        }

        for(int cell = 0; cell < gridCells; ++cell)
        {
            const bool tracked = trackedCloseCell[static_cast<size_t>(cell)] != 0;
            const bool nonTracked = nonTrackedCloseCell[static_cast<size_t>(cell)] != 0;
            const bool highConf = highConfNonTrackedCloseCell[static_cast<size_t>(cell)] != 0;
            if(tracked)
                ++metrics.trackedCloseCells;
            if(nonTracked)
                ++metrics.nonTrackedCloseCells;
            if(nonTracked && !tracked)
                ++metrics.novelCloseCells;
            if(highConf && !tracked)
                ++metrics.highConfNovelCloseCells;
        }

        metrics.stableInlierShare = metrics.inlierAssoc > 0
            ? static_cast<float>(metrics.stableInlierAssoc) / static_cast<float>(metrics.inlierAssoc)
            : 0.0f;
        metrics.refOverlapShare = metrics.inlierAssoc > 0
            ? static_cast<float>(metrics.refOverlapInlier) / static_cast<float>(metrics.inlierAssoc)
            : 0.0f;
        metrics.stableRefOverlapShare = metrics.stableInlierAssoc > 0
            ? static_cast<float>(metrics.stableRefOverlapInlier) /
              static_cast<float>(metrics.stableInlierAssoc)
            : 0.0f;
        metrics.novelCloseCellShare = metrics.nonTrackedCloseCells > 0
            ? static_cast<float>(metrics.novelCloseCells) / static_cast<float>(metrics.nonTrackedCloseCells)
            : 0.0f;

        metrics.weakSupport =
            mnMatchesInliers < metrics.weakInlierThreshold ||
            metrics.stableInlierAssoc < 140 ||
            metrics.stableSpatial.nonEmptyCells < 18;
        metrics.stableSupportOk =
            metrics.stableInlierAssoc >= 160 &&
            metrics.stableSpatial.nonEmptyCells >= 18 &&
            metrics.stableRefOverlapShare >= 0.70f;
        metrics.closeCoverageOk =
            metrics.trackedCloseCells >= 8 || nTrackedClose >= 15;
        metrics.novelCloseStrong =
            metrics.highConfNovelCloseCells >= 8 &&
            metrics.novelCloseCellShare >= 0.50f;
        metrics.closeInsert =
            !metrics.closeCoverageOk && metrics.novelCloseStrong;

        return metrics;
    };

    const XFeatKFGuardMetrics xfeatKfGuard = ComputeXFeatKFGuardMetrics();

    const bool bXFeatKFInsertGuardEnabled = IsXFeatKFInsertGuardEnabled();
    const bool bXFeatKFInsertGuardEligible =
        bXFeatKFInsertGuardEnabled && xfeatKfGuard.eligible;
    const bool c1cNoClose =
        mSensor!=System::MONOCULAR &&
        mSensor!=System::IMU_MONOCULAR &&
        mSensor!=System::IMU_STEREO &&
        mSensor!=System::IMU_RGBD &&
        (mnMatchesInliers < nRefMatches * 0.25);
    const bool c2NoClose =
        (mnMatchesInliers < nRefMatches * thRefRatio) && mnMatchesInliers > 15;
    const bool bNeedNewKFWithoutClose =
        (((c1a || c1b || c1cNoClose) && c2NoClose) || c3 || c4);
    const bool bCloseOnlyInsertion =
        bOriginalNeedNewKF && bNeedToInsertClose && !bNeedNewKFWithoutClose;
    bool bCloseGuardSuppress = false;
    const char* closeGuardReason = "disabled";
    if(!bXFeatKFInsertGuardEnabled)
    {
        closeGuardReason = "disabled";
    }
    else if(!bXFeatKFInsertGuardEligible)
    {
        closeGuardReason = "not_eligible";
    }
    else if(!bCloseOnlyInsertion)
    {
        closeGuardReason = "not_close_only";
    }
    else if(c1a)
    {
        closeGuardReason = "max_frame";
    }
    else if(xfeatKfGuard.weakSupport)
    {
        closeGuardReason = "weak_support";
    }
    else if(!xfeatKfGuard.stableSupportOk)
    {
        closeGuardReason = "unstable_support";
    }
    else if(xfeatKfGuard.closeInsert)
    {
        closeGuardReason = "novel_close";
    }
    else
    {
        bCloseGuardSuppress = true;
        closeGuardReason = "stable_close_only";
    }

    auto LogKFDecision = [&](const bool needNewKF, const char* decision, const bool forceLog = false)
    {
        if(!needNewKF && !forceLog)
            return;
        if(!ShouldPrintXFeatDebug(mCurrentFrame.mnId))
            return;
        const long int framesSinceLastKF =
            mCurrentFrame.mnId >= mnLastKeyFrameId
                ? static_cast<long int>(mCurrentFrame.mnId - mnLastKeyFrameId)
                : -1;

        std::cout << "[Tracking][KFDecision]"
                  << " frame_id=" << mCurrentFrame.mnId
                  << " ref_kf_id=" << (mpReferenceKF ? static_cast<long long>(mpReferenceKF->mnId) : -1)
                  << " last_kf_id=" << (mpLastKeyFrame ? static_cast<long long>(mpLastKeyFrame->mnId) : -1)
                  << " frames_since_last_kf=" << framesSinceLastKF
                  << " n_kfs=" << nKFs
                  << " n_ref_matches=" << nRefMatches
                  << " mn_matches_inliers=" << mnMatchesInliers
                  << " n_tracked_close=" << nTrackedClose
                  << " n_non_tracked_close=" << nNonTrackedClose
                  << " need_insert_close=" << (bNeedToInsertClose ? 1 : 0)
                  << " local_mapping_idle=" << (bLocalMappingIdle ? 1 : 0)
                  << " local_mapping_queue=" << mpLocalMapper->KeyframesInQueue()
                  << " th_ref_ratio=" << thRefRatio
                  << " min_frames=" << mMinFrames
                  << " max_frames=" << mMaxFrames
                  << " c1a=" << (c1a ? 1 : 0)
                  << " c1b=" << (c1b ? 1 : 0)
                  << " c1c=" << (c1c ? 1 : 0)
                  << " c2=" << (c2 ? 1 : 0)
                  << " c3=" << (c3 ? 1 : 0)
                  << " c4=" << (c4 ? 1 : 0)
                  << " need_new_kf=" << (needNewKF ? 1 : 0)
                  << " decision=" << decision
                  << " kf_guard_enabled=" << (bXFeatKFInsertGuardEnabled ? 1 : 0)
                  << " kf_guard_eligible=" << (bXFeatKFInsertGuardEligible ? 1 : 0)
                  << " need_without_close=" << (bNeedNewKFWithoutClose ? 1 : 0)
                  << " close_only_insertion=" << (bCloseOnlyInsertion ? 1 : 0)
                  << " close_guard_suppressed=" << (bCloseGuardSuppress ? 1 : 0)
                  << " close_guard_reason=" << closeGuardReason
                  << " c1c_no_close=" << (c1cNoClose ? 1 : 0)
                  << " c2_no_close=" << (c2NoClose ? 1 : 0);
        std::cout << std::endl;
    };

    if(bOriginalNeedNewKF && bCloseGuardSuppress)
    {
        LogKFDecision(false, "close_guard_suppressed", true);
        return false;
    }

    if(bOriginalNeedNewKF)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle || mpLocalMapper->IsInitializing())
        {
            LogKFDecision(true, "insert_idle");
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR  && mSensor!=System::IMU_MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                {
                    LogKFDecision(true, "insert_queue_lt3");
                    return true;
                }
                else
                {
                    LogKFDecision(false, "queue_full", true);
                    return false;
                }
            }
            else
            {
                //std::cout << "NeedNewKeyFrame: localmap is busy" << std::endl;
                LogKFDecision(false, "local_mapping_busy", true);
                return false;
            }
        }
    }
    else
    {
        LogKFDecision(false, "conditions_false");
        return false;
    }
}

void Tracking::CreateNewKeyFrame()
{
    if(mpLocalMapper->IsInitializing() && !mpAtlas->isImuInitialized())
        return;

    if(!mpLocalMapper->SetNotStop(true))
        return;

    const long unsigned int previousLastKeyFrameId = mnLastKeyFrameId;
    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

    if(mpAtlas->isImuInitialized()) //  || mpLocalMapper->IsInitializing())
        pKF->bImu = true;

    pKF->SetNewBias(mCurrentFrame.mImuBias);
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mpLastKeyFrame)
    {
        pKF->mPrevKF = mpLastKeyFrame;
        mpLastKeyFrame->mNextKF = pKF;
    }
    else
        Verbose::PrintMess("No last KF in KF creation!!", Verbose::VERBOSITY_NORMAL);

    // Reset preintegration from last KF (Create new object)
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
    {
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKF->GetImuBias(),pKF->mImuCalib);
    }

    if(mSensor!=System::MONOCULAR && mSensor != System::IMU_MONOCULAR) // TODO check if incluide imu_stereo
    {
        const bool bUseStereoBirthQuality =
            (mSensor == System::STEREO || mSensor == System::IMU_STEREO) &&
            mCurrentFrame.Nleft == -1;
        mCurrentFrame.UpdatePoseMatrices();
        // cout << "create new MPs" << endl;
        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        int maxPoint = 100;
        if(mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            maxPoint = 100;

        vector<pair<float,int> > vDepthIdx;
        DepthDistributionStats createKFFrameDepthStats;
        int N = (mCurrentFrame.Nleft != -1) ? mCurrentFrame.Nleft : mCurrentFrame.N;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            AccumulateDepthDistribution(createKFFrameDepthStats, z);
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            int nCreatedPoints = 0;
            int nReusedPoints = 0;
            int nTrackingStereoCreatedTotal = 0;
            int nTrackingStereoCreatedCloseHighConf = 0;
            int nTrackingStereoSkippedCloseLowConf = 0;
            DepthDistributionStats createKFSelectedDepthStats;
            DepthDistributionStats createKFUsedDepthStats;
            DepthDistributionStats createKFCreatedDepthStats;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;
                AccumulateDepthDistribution(createKFSelectedDepthStats, vDepthIdx[j].first);

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    const bool bCloseDepth = vDepthIdx[j].first > 0.0f && vDepthIdx[j].first <= mThDepth;
                    const bool bSkipLowConfCloseStereo =
                        bUseStereoBirthQuality &&
                        bCloseDepth &&
                        i >= 0 &&
                        i < static_cast<int>(mCurrentFrame.mvStereoBirthQuality.size()) &&
                        mCurrentFrame.mvStereoBirthQuality[static_cast<size_t>(i)] == Frame::kStereoBirthCloseLowConf;
                    if(bSkipLowConfCloseStereo)
                    {
                        ++nTrackingStereoSkippedCloseLowConf;
                        continue;
                    }
                    AccumulateDepthDistribution(createKFCreatedDepthStats, vDepthIdx[j].first);
                    Eigen::Vector3f x3D;

                    if(mCurrentFrame.Nleft == -1){
                        mCurrentFrame.UnprojectStereo(i, x3D);
                    }
                    else{
                        x3D = mCurrentFrame.UnprojectStereoFishEye(i);
                    }

                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpAtlas->GetCurrentMap());
                    pNewMP->SetCreationSource(MapPoint::kCreationSourceTrackingStereo);
                    SetMappingFeatureQualityFromFrame(pNewMP, mCurrentFrame, pKF->mnId, i);
                    pNewMP->AddObservation(pKF,i);

                    //Check if it is a stereo observation in order to not
                    //duplicate mappoints
                    if(mCurrentFrame.Nleft != -1 && mCurrentFrame.mvLeftToRightMatch[i] >= 0){
                        mCurrentFrame.mvpMapPoints[mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]]=pNewMP;
                        pNewMP->AddObservation(pKF,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                        pKF->AddMapPoint(pNewMP,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                    }

                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                    nCreatedPoints++;
                    ++nTrackingStereoCreatedTotal;
                    if(bUseStereoBirthQuality &&
                       bCloseDepth &&
                       i >= 0 &&
                       i < static_cast<int>(mCurrentFrame.mvStereoBirthQuality.size()) &&
                       mCurrentFrame.mvStereoBirthQuality[static_cast<size_t>(i)] == Frame::kStereoBirthCloseHighConf)
                    {
                        ++nTrackingStereoCreatedCloseHighConf;
                    }
                }
                else
                {
                    AccumulateDepthDistribution(createKFUsedDepthStats, vDepthIdx[j].first);
                    nPoints++;
                    nReusedPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>maxPoint)
                {
                    break;
                }
            }
            if(ShouldPrintXFeatDebug(mCurrentFrame.mnId))
            {
                const std::ios::fmtflags oldFlags = std::cout.flags();
                const std::streamsize oldPrecision = std::cout.precision();
                std::cout << std::fixed << std::setprecision(6)
                          << "[DepthDiag][CreateKF]"
                          << " frame_id=" << mCurrentFrame.mnId
                          << " kf_id=" << pKF->mnId
                          << " mThDepth=" << mThDepth
                          << " maxPoint=" << maxPoint
                          << " depth_candidates=" << vDepthIdx.size()
                          << " used_or_created=" << nPoints
                          << " created=" << nCreatedPoints
                          << " reused=" << nReusedPoints
                          << " tracking_stereo_created_total=" << nTrackingStereoCreatedTotal
                          << " tracking_stereo_created_close_high_conf=" << nTrackingStereoCreatedCloseHighConf
                          << " tracking_stereo_skipped_close_low_conf=" << nTrackingStereoSkippedCloseLowConf
                          << " stereo_birth_quality_enabled=" << (bUseStereoBirthQuality ? 1 : 0);
                AppendDepthDistributionStats(std::cout, "frame", createKFFrameDepthStats);
                AppendDepthDistributionStats(std::cout, "selected_depth", createKFSelectedDepthStats);
                AppendDepthDistributionStats(std::cout, "reused_depth", createKFUsedDepthStats);
                AppendDepthDistributionStats(std::cout, "created_depth", createKFCreatedDepthStats);
                std::cout << std::endl;
                std::cout.flags(oldFlags);
                std::cout.precision(oldPrecision);
            }
            //Verbose::PrintMess("new mps for stereo KF: " + to_string(nPoints), Verbose::VERBOSITY_NORMAL);
        }
        else if(ShouldPrintXFeatDebug(mCurrentFrame.mnId))
        {
            const std::ios::fmtflags oldFlags = std::cout.flags();
            const std::streamsize oldPrecision = std::cout.precision();
            std::cout << std::fixed << std::setprecision(6)
                      << "[DepthDiag][CreateKF]"
                      << " frame_id=" << mCurrentFrame.mnId
                      << " kf_id=" << pKF->mnId
                      << " mThDepth=" << mThDepth
                      << " maxPoint=" << maxPoint
                      << " depth_candidates=0"
                      << " used_or_created=0"
                      << " created=0"
                      << " reused=0"
                      << " tracking_stereo_created_total=0"
                      << " tracking_stereo_created_close_high_conf=0"
                      << " tracking_stereo_skipped_close_low_conf=0"
                      << " stereo_birth_quality_enabled=" << (bUseStereoBirthQuality ? 1 : 0);
            AppendDepthDistributionStats(std::cout, "frame", createKFFrameDepthStats);
            std::cout << std::endl;
            std::cout.flags(oldFlags);
            std::cout.precision(oldPrecision);
        }
    }


    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;

    if(ShouldPrintXFeatDebug(mCurrentFrame.mnId))
    {
        const long int framesSincePreviousKF =
            mCurrentFrame.mnId >= previousLastKeyFrameId
                ? static_cast<long int>(mCurrentFrame.mnId - previousLastKeyFrameId)
                : -1;
        std::cout << "[Tracking][KFDecision]"
                  << " frame_id=" << mCurrentFrame.mnId
                  << " kf_id=" << pKF->mnId
                  << " created_kf=1"
                  << " frames_since_previous_kf=" << framesSincePreviousKF
                  << " local_mapping_queue_after_insert=" << mpLocalMapper->KeyframesInQueue()
                  << " mn_matches_inliers=" << mnMatchesInliers
                  << std::endl;
    }
}

void Tracking::SearchLocalPoints()
{
    EnsureXFeatMatchSourceSize(mCurrentFrame);
    mCurrentFrame.mLocalProjReferenceTargetMPIds.clear();
    int nAlreadyMatched = 0;
    int nBadAlreadyMatched = 0;
    MapPointQualityStats mpQualityAlreadyMatched;
    // Do not search map points already matched
    int alreadyIdx = 0;
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++, alreadyIdx++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
                if(alreadyIdx >= 0 && alreadyIdx < static_cast<int>(mCurrentFrame.mvXFeatMatchSource.size()))
                    mCurrentFrame.mvXFeatMatchSource[static_cast<size_t>(alreadyIdx)] = kXFeatMatchUnknown;
                nBadAlreadyMatched++;
            }
            else
            {
                nAlreadyMatched++;
                AccumulateMapPointQuality(mpQualityAlreadyMatched,
                                          pMP,
                                          mCurrentFrame,
                                          pMP->mTrackDepth,
                                          mThDepth);
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
                pMP->mbTrackInViewR = false;
            }
        }
    }

    int nToMatch=0;
    int nLocalAlreadySeen = 0;
    int nLocalBad = 0;
    int nLocalFoundRatioRejected = 0;
    int nLocalInFrustum = 0;
    int nLocalTrackInView = 0;
    MapPointQualityStats mpQualityLocalAlreadySeen;
    MapPointQualityStats mpQualityFoundRatioRejected;
    MapPointQualityStats mpQualityLocalInFrustum;
    MapPointQualityStats mpQualityLocalTrackInView;
    const bool bUseORB = (std::getenv("USE_ORB") != nullptr);
    const bool bUseLocalFoundRatioGate = IsLocalMapFoundRatioGateEnabled();
    const float localMinFoundRatio = GetLocalMapFoundRatioMin();
    const float localMinFoundRatioDepth = GetLocalMapFoundRatioMinDepth() > 0.0f
        ? GetLocalMapFoundRatioMinDepth()
        : mThDepth;
    const std::string localFoundRatioScope = GetLocalMapFoundRatioScope();
    const bool bUseLightGlueLocalMap = !bUseORB && IsXFeatLightGlueLocalMapEnabled();
    const bool bUseLocalMapCoreFirst = !bUseORB && IsXFeatLocalMapCoreFirstEnabled();
    const int coreFirstMinPrior = GetXFeatLocalMapCoreFirstMinPrior();
    const int coreFirstMinCoreMatches = GetXFeatLocalMapCoreFirstMinCoreMatches();
    const int coreFirstMinTotalMatches = GetXFeatLocalMapCoreFirstMinTotalMatches();
    int coreFirstCoreCandidates = 0;
    int coreFirstExpansionCandidates = 0;
    int coreFirstExpansionAdmitted = 0;
    int coreFirstExpansionSuppressed = 0;
    int coreFirstCoreMatches = 0;
    int coreFirstExpansionMatches = 0;
    const char* coreFirstDecision = bUseLocalMapCoreFirst ? "pending" : "disabled";
    int localFoundRatioRejectedBySource[kMapPointCreationSourceCount] = {};
    int localFoundRatioRejectedDepthQualified = 0;
    int localFoundRatioRejectedDepthQualifiedBySource[kMapPointCreationSourceCount] = {};
    int localProjCloseTrackingStereoCandidates = 0;
    int localProjCloseTrackingStereoGeomAccept = 0;
    int localProjCloseTrackingStereoRejectUr = 0;
    int localProjCloseTrackingStereoRejectDepth = 0;
    int triang0To10HistoryGateCandidate = 0;
    int triang0To10HistoryGateReject = 0;
    int triang0To10HistoryGateRejectOpt = 0;
    int triang0To10HistoryGateRejectRecent = 0;
    int triang0To10HistoryGatePass = 0;
    int triang0To10HistoryGateInsufficient = 0;
    int triang0To10ProjAttempt = 0;
    int triang0To10ProjMatch = 0;
    int triang0To10VisibilityHistoryRecorded = 0;
    int triang0To10VisibilityHistoryEvaluated = 0;
    int triang0To10VisibilityHistoryLowRelevant = 0;
    int triang0To10VisibilityHistoryLowRelevantAccepted = 0;
    int localmapLmt0To10FromReference = 0;
    int localmapLmt0To10FromVoted = 0;
    int localmapLmt0To10FromExpansion = 0;
    int localmapLmt0To10FromTemporal = 0;
    int localmapLmt0To10FromUnknown = 0;
    int triang0To10PriorityCandidate[kLocalProjPriorityBinCount] = {};
    int triang0To10PriorityMatch[kLocalProjPriorityBinCount] = {};
    int localProjTriang0To10ProjOutsideReject = 0;
    int localProjTriang0To10CompanionOutsideReject = 0;
    int refAssocMatcherInput = 0;
    int refAssocMatcherSkipNotInView = 0;
    int refAssocMatcherSkipFar = 0;
    int refAssocMatcherSkipBad = 0;
    int refAssocMatcherSkipNoDesc = 0;
    int refAssocMatcherProjected = 0;
    int refAssocMatcherSideAttempt = 0;
    int refAssocMatcherInvalidLevel = 0;
    int refAssocMatcherEmptyArea = 0;
    int refAssocMatcherAreaCandidates = 0;
    int refAssocMatcherInvalidCandidate = 0;
    int refAssocMatcherOccupiedCandidate = 0;
    int refAssocMatcherGeomUrReject = 0;
    int refAssocMatcherGeomDepthReject = 0;
    int refAssocMatcherValidCandidate = 0;
    int refAssocMatcherNoValidCandidate = 0;
    int refAssocMatcherRawCandidate = 0;
    int refAssocMatcherDistReject = 0;
    int refAssocMatcherRatioReject = 0;
    int refAssocMatcherCandidatePool = 0;
    int refAssocMatcherSpatialReject = 0;
    int refAssocMatcherAssignReject = 0;
    int refAssocMatcherOutsideReject = 0;
    int refAssocMatcherAssigned = 0;
    CloseReuseFunnelStats closeReuseFunnel[kCloseReuseSourceBinCount][kCloseReuseDepthBinCount];
    CloseReuseContributionStats closeReuseContribution[kCloseReuseSourceBinCount][kCloseReuseContributorBinCount];
    std::unordered_map<KeyFrame*, CloseReuseContributionStats> closeReuseKFContribution;
    const bool bCollectCloseReuseFunnel =
        ShouldPrintXFeatDebug(mCurrentFrame.mnId) && IsXFeatFeatureDiagEnabled();
    const bool bRecordCloseReuseCohort =
        IsXFeatDebugEnabled() && IsXFeatFeatureDiagEnabled();
    static constexpr int kMappingFeatureReuseGridBins = 4;
    static constexpr int kMappingFeatureReuseOctaveBins = 4;
    static constexpr int kMappingFeatureReuseResponseBins = 4;
    static constexpr int kMappingFeatureReuseDepthBins = 6;
    static constexpr int kMappingFeatureReuseCombinedBins = 6;
    std::vector<MapPoint*> vpLocalProjectionMapPoints;
    vpLocalProjectionMapPoints.reserve(mvpLocalMapPoints.size());
    std::vector<MapPoint*> vpCoreLocalProjectionMapPoints;
    std::vector<MapPoint*> vpExpansionLocalProjectionMapPoints;
    if(bUseLocalMapCoreFirst)
    {
        vpCoreLocalProjectionMapPoints.reserve(mvpLocalMapPoints.size());
        vpExpansionLocalProjectionMapPoints.reserve(mvpLocalMapPoints.size() / 2);
    }
    struct DeferredLocalProjectionPoint
    {
        MapPoint* pMP = nullptr;
        float depth = std::numeric_limits<float>::quiet_NaN();
    };
    std::vector<DeferredLocalProjectionPoint> vDeferredExpansionLocalPoints;
    if(bUseLocalMapCoreFirst)
    {
        vDeferredExpansionLocalPoints.reserve(mvpLocalMapPoints.size() / 2);
    }
    MappingFeatureReuseFunnelStats mappingFeatureGridFunnel[kMappingFeatureReuseGridBins];
    MappingFeatureReuseFunnelStats mappingFeatureOctaveFunnel[kMappingFeatureReuseOctaveBins];
    MappingFeatureReuseFunnelStats mappingFeatureResponseFunnel[kMappingFeatureReuseResponseBins];
    MappingFeatureReuseFunnelStats mappingFeatureDepthFunnel[kMappingFeatureReuseDepthBins];
    MappingFeatureReuseFunnelStats mappingFeatureCombinedFunnel[kMappingFeatureReuseCombinedBins];
    ReferenceAssociationFunnelStats
        referenceAssociationFunnel[kCloseReuseSourceBinCount][kCloseReuseDepthBinCount];
    struct ReferenceAssociationTargetBin
    {
        int sourceBin = -1;
        int depthBin = -1;
    };
    std::unordered_map<MapPoint*, ReferenceAssociationTargetBin> referenceAssociationTargets;
    auto referenceAssociationBucket =
        [&](MapPoint* pMP) -> ReferenceAssociationFunnelStats*
    {
        if(!bCollectCloseReuseFunnel || !pMP)
            return nullptr;
        const auto it = referenceAssociationTargets.find(pMP);
        if(it == referenceAssociationTargets.end())
            return nullptr;
        return &referenceAssociationFunnel[it->second.sourceBin][it->second.depthBin];
    };
    if(bCollectCloseReuseFunnel && mpReferenceKF)
    {
        std::unordered_set<MapPoint*> seenReferenceMapPoints;
        const std::vector<MapPoint*> referenceMapPoints = mpReferenceKF->GetMapPointMatches();
        seenReferenceMapPoints.reserve(referenceMapPoints.size());
        referenceAssociationTargets.reserve(referenceMapPoints.size());
        for(MapPoint* pMP : referenceMapPoints)
        {
            if(!pMP || pMP->isBad() || !seenReferenceMapPoints.insert(pMP).second)
                continue;
            const float depth = ResolveMapPointQualityDepth(
                mCurrentFrame, pMP, std::numeric_limits<float>::quiet_NaN());
            if(!IsCloseReuseTarget(pMP, depth))
                continue;
            const int sourceBin = CloseReuseSourceBin(pMP->GetCreationSource());
            const int depthBin = CloseReuseDepthBin(depth);
            if(sourceBin < 0 || depthBin < 0)
                continue;
            ReferenceAssociationFunnelStats& stats =
                referenceAssociationFunnel[sourceBin][depthBin];
            ++stats.referenceEligible;
            const float foundRatio = pMP->GetFoundRatio();
            const bool bStable =
                pMP->Observations() >= 3 &&
                std::isfinite(foundRatio) &&
                foundRatio >= 0.35f;
            if(!bStable)
                continue;
            ++stats.stableReference;
            if(ClassifyMapPointLightVisibilityInFrame(mCurrentFrame, pMP) !=
               LocalMapPredVisibilityStatus::Visible)
            {
                continue;
            }
            ++stats.predictedVisibleTarget;
            ReferenceAssociationTargetBin targetBin;
            targetBin.sourceBin = sourceBin;
            targetBin.depthBin = depthBin;
            referenceAssociationTargets[pMP] = targetBin;
        }

        const size_t nAssociations = std::min(mCurrentFrame.mvpMapPoints.size(),
                                              mCurrentFrame.mvXFeatMatchSource.size());
        for(size_t i = 0; i < nAssociations; ++i)
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            ReferenceAssociationFunnelStats* stats = referenceAssociationBucket(pMP);
            if(!stats)
                continue;
            ++stats->preAssociated;
            const unsigned char matchSource = mCurrentFrame.mvXFeatMatchSource[i];
            if(matchSource == kXFeatMatchMotionProjection ||
               matchSource == kXFeatMatchMotionLightGlue)
            {
                ++stats->motionRecovered;
            }
        }

        mCurrentFrame.mLocalProjReferenceTargetMPIds.reserve(referenceAssociationTargets.size());
        for(const auto& target : referenceAssociationTargets)
        {
            MapPoint* pMP = target.first;
            if(!pMP || pMP->isBad())
                continue;
            // Stable reference close/mid points predicted visible but not
            // recovered before local projection. Used only for failure diagnostics.
            if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
                continue;
            mCurrentFrame.mLocalProjReferenceTargetMPIds.insert(pMP->mnId);
        }
    }
    if(ShouldPrintXFeatDebug(mCurrentFrame.mnId) && IsXFeatFeatureDiagEnabled())
    {
        DepthDistributionStats frameDepthStats;
        for(float depth : mCurrentFrame.mvDepth)
            AccumulateDepthDistribution(frameDepthStats, depth);

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6)
                  << "[DepthDiag][Frame]"
                  << " frame_id=" << mCurrentFrame.mnId
                  << " N=" << mCurrentFrame.N
                  << " Nleft=" << mCurrentFrame.Nleft
                  << " mThDepth=" << mThDepth;
        AppendDepthDistributionStats(std::cout, "mvDepth", frameDepthStats);
        std::cout << std::endl;
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    std::unordered_set<long unsigned int> localProjectedMapPointIds;
    if(bUseLightGlueLocalMap)
        localProjectedMapPointIds.reserve(mvpLocalMapPoints.size());

    const bool bMotionGuardEnabled =
        !bUseORB && IsXFeatMotionHealthGuardEnabled();
    const bool bMotionGuardActive =
        bMotionGuardEnabled &&
        mbXFeatMotionHealthValid &&
        mnXFeatMotionHealthFrameId == mCurrentFrame.mnId &&
        (mbXFeatMotionHealthWeak || mbXFeatMotionHealthBad);
    const int motionGuardMinObs = GetXFeatMotionGuardMinObs();
    const float motionGuardMinFoundRatio = GetXFeatMotionGuardMinFoundRatio();
    int motionGuardCandidates = 0;
    int motionGuardLowRisk = 0;
    int motionGuardMidRisk = 0;
    int motionGuardHighRisk = 0;
    int motionGuardSkipped = 0;
    int motionGuardSkipLowObs = 0;
    int motionGuardSkipLowFoundRatio = 0;
    int motionGuardSkipUnknownSource = 0;
    int motionGuardSkipWeakProvenance = 0;
    int motionGuardSkipHistory = 0;
    int motionGuardSkipOther = 0;

    int lgLocalKFsUsed = 0;
    int lgLocalRawMatches = 0;
    int lgLocalMpValid = 0;
    int lgLocalOneToOne = 0;
    int lgLocalFilled = 0;
    int lgLocalOccupied = 0;
    int lgLocalDuplicateMP = 0;
    int lgLocalNotProjected = 0;
    int lgLocalProjReject = 0;
    int lgLocalVerifyCandidates = 0;
    int lgLocalVerified = 0;
    int lgLocalRejected = 0;
    int lgLocalMismatch = 0;
    int lgLocalConflictLowScore = 0;
    int lgLocalMapUpdatedEdgeCandidates = 0;
    int lgLocalMapUpdatedEdgeVerified = 0;
    int lgLocalMapUpdatedEdgeConflictUnverified = 0;
    int lgLocalMapUpdatedEdgeNoEvidence = 0;
    int lgLocalMapUpdatedEdgeNonTargetIgnored = 0;
    bool bLightGlueLocalFailed = false;
    int projectionMatches = 0;
    int projectionTh = 0;
    bool bProjectionRun = false;
    bool projectionFarFilter = false;
    float projectionFarPoints = 0.0f;
    bool projectionNearOnlyApplied = false;

    auto closeReuseBucket = [&](MapPoint* pMP, const float resolvedDepth) -> CloseReuseFunnelStats*
    {
        if(!bCollectCloseReuseFunnel || !IsCloseReuseTarget(pMP, resolvedDepth))
            return nullptr;
        const int sourceBin = CloseReuseSourceBin(pMP->GetCreationSource());
        const int depthBin = CloseReuseDepthBin(resolvedDepth);
        if(sourceBin < 0 || depthBin < 0)
            return nullptr;
        return &closeReuseFunnel[sourceBin][depthBin];
    };

    auto closeReuseContributorBucket =
        [&](MapPoint* pMP, const float resolvedDepth) -> CloseReuseContributionStats*
    {
        if(!bCollectCloseReuseFunnel || !IsCloseReuseTarget(pMP, resolvedDepth))
            return nullptr;
        const int sourceBin = CloseReuseSourceBin(pMP->GetCreationSource());
        if(sourceBin < 0)
            return nullptr;
        const Tracking::LocalMPProvenance* pProvenance = nullptr;
        const auto itProv = mmLocalMPProvenance.find(pMP);
        if(itProv != mmLocalMPProvenance.end() &&
           itProv->second.frameId == mCurrentFrame.mnId)
        {
            pProvenance = &itProv->second;
        }
        return &closeReuseContribution[sourceBin][CloseReuseContributorBin(pProvenance)];
    };

    auto closeReuseKFContributionBucket =
        [&](MapPoint* pMP, const float resolvedDepth) -> CloseReuseContributionStats*
    {
        if(!bCollectCloseReuseFunnel || !IsCloseReuseTarget(pMP, resolvedDepth))
            return nullptr;
        const auto itProv = mmLocalMPProvenance.find(pMP);
        if(itProv == mmLocalMPProvenance.end() ||
           itProv->second.frameId != mCurrentFrame.mnId ||
           !itProv->second.pBestKeyFrame)
        {
            return nullptr;
        }
        return &closeReuseKFContribution[itProv->second.pBestKeyFrame];
    };

    auto accumulateMappingFeatureReuse =
        [&](MapPoint* pMP,
            const float resolvedDepth,
            int MappingFeatureReuseFunnelStats::*field)
    {
        if(!bCollectCloseReuseFunnel || !pMP || !field)
            return;
        if(!IsLocalProjHistoryTarget(pMP, resolvedDepth))
            return;

        const bool hasFeatureQuality = pMP->HasMappingFeatureQuality();
        const int gridBin = hasFeatureQuality
            ? MappingFeatureGridBin(pMP->mnMappingFeatureGridX, pMP->mnMappingFeatureGridY)
            : 0;
        const int octaveBin = hasFeatureQuality
            ? MappingFeatureOctaveBin(pMP->mnMappingFeatureOctave)
            : 0;
        const int responseBin = hasFeatureQuality
            ? MappingFeatureResponseBin(pMP->mMappingFeatureResponse)
            : 0;
        const int creationDepthBin = hasFeatureQuality
            ? DepthBinIndex(pMP->mMappingFeatureDepth)
            : 0;
        const int stereoQualityBin = hasFeatureQuality
            ? MappingFeatureStereoQualityBin(pMP->mMappingFeatureStereoBirthQuality)
            : 0;
        const int combinedBin = hasFeatureQuality
            ? MappingFeatureCombinedBin(gridBin, octaveBin, responseBin, stereoQualityBin)
            : 0;

        if(gridBin >= 0 && gridBin < kMappingFeatureReuseGridBins)
            (mappingFeatureGridFunnel[gridBin].*field) += 1;
        if(octaveBin >= 0 && octaveBin < kMappingFeatureReuseOctaveBins)
            (mappingFeatureOctaveFunnel[octaveBin].*field) += 1;
        if(responseBin >= 0 && responseBin < kMappingFeatureReuseResponseBins)
            (mappingFeatureResponseFunnel[responseBin].*field) += 1;
        if(creationDepthBin >= 0 && creationDepthBin < kMappingFeatureReuseDepthBins)
            (mappingFeatureDepthFunnel[creationDepthBin].*field) += 1;
        if(combinedBin >= 0 && combinedBin < kMappingFeatureReuseCombinedBins)
            (mappingFeatureCombinedFunnel[combinedBin].*field) += 1;
    };

    auto shouldSkipByMotionHealthGuard =
        [&](MapPoint* pMP,
            const float resolvedDepth,
            const Tracking::LocalMPProvenance* pProvenance) -> bool
    {
        if(!bMotionGuardActive)
            return false;

        motionGuardCandidates++;

        bool bHighRisk = false;
        bool bLowObs = false;
        bool bLowFoundRatio = false;
        bool bUnknownSource = false;
        bool bWeakProvenance = false;
        bool bHistoryRisk = false;

        if(!pMP)
        {
            bHighRisk = true;
            motionGuardSkipOther++;
        }
        else
        {
            const int observations = pMP->Observations();
            bLowObs = observations < motionGuardMinObs;

            const float foundRatio = pMP->GetFoundRatio();
            bLowFoundRatio =
                !std::isfinite(foundRatio) ||
                foundRatio < motionGuardMinFoundRatio;

            bUnknownSource =
                pMP->GetCreationSource() == MapPoint::kCreationSourceUnknown;

            const Tracking::LocalKFProvenanceRole role = pProvenance
                ? pProvenance->bestRole
                : Tracking::LocalKFProvenanceRole::Unknown;
            bWeakProvenance =
                role == Tracking::LocalKFProvenanceRole::Temporal ||
                role == Tracking::LocalKFProvenanceRole::Unknown;

            if(IsLocalProjHistoryTarget(pMP, resolvedDepth))
            {
                const MapPoint::LocalProjectionHistoryStats history =
                    pMP->GetLocalProjectionHistory();
                const int optOutcomes =
                    history.optInlierCount + history.optOutlierCount;
                const int recentValid = CountHistoryBits(history.recentValidMask);
                bHistoryRisk =
                    (optOutcomes >= 3 && LocalProjOptOutlierRate(history) > 0.35f) ||
                    (recentValid >= 3 && LocalProjRecentOutlierRate(history) > 0.40f);
            }

            bHighRisk =
                bLowObs ||
                bLowFoundRatio ||
                bUnknownSource ||
                bWeakProvenance ||
                bHistoryRisk;
        }

        if(bHighRisk)
        {
            motionGuardHighRisk++;
            motionGuardSkipped++;
            if(bLowObs)
                motionGuardSkipLowObs++;
            if(bLowFoundRatio)
                motionGuardSkipLowFoundRatio++;
            if(bUnknownSource)
                motionGuardSkipUnknownSource++;
            if(bWeakProvenance)
                motionGuardSkipWeakProvenance++;
            if(bHistoryRisk)
                motionGuardSkipHistory++;
            return true;
        }

        const Tracking::LocalKFProvenanceRole role = pProvenance
            ? pProvenance->bestRole
            : Tracking::LocalKFProvenanceRole::Unknown;
        if(role == Tracking::LocalKFProvenanceRole::Reference ||
           role == Tracking::LocalKFProvenanceRole::Voted)
        {
            motionGuardLowRisk++;
        }
        else
        {
            motionGuardMidRisk++;
        }
        return false;
    };

    auto isCoreLocalProjectionPoint =
        [&](MapPoint* pMP,
            const float resolvedDepth,
            const Tracking::LocalMPProvenance* pProvenance) -> bool
    {
        if(!bUseLocalMapCoreFirst || !pMP)
            return true;

        if(!IsCloseReuseTarget(pMP, resolvedDepth))
            return true;

        const Tracking::LocalKFProvenanceRole role = pProvenance
            ? pProvenance->bestRole
            : Tracking::LocalKFProvenanceRole::Unknown;
        if(role == Tracking::LocalKFProvenanceRole::Reference)
            return true;
        if(role == Tracking::LocalKFProvenanceRole::Voted &&
           pProvenance &&
           pProvenance->bestVoteCount >= 15 &&
           pProvenance->localObservations >= 4)
        {
            return true;
        }

        const float foundRatio = pMP->GetFoundRatio();
        if(pMP->Observations() >= 4 &&
           std::isfinite(foundRatio) &&
           foundRatio >= 0.45f)
        {
            return true;
        }

        if(IsLocalProjHistoryTarget(pMP, resolvedDepth))
        {
            const int priorityBin = LocalProjPriorityBin(pMP, pProvenance, resolvedDepth);
            return priorityBin == 1 || priorityBin == 2 || priorityBin == 3;
        }

        return false;
    };

    auto admitLocalProjectionPoint = [&](MapPoint* pMP,
                                         const float resolvedDepth,
                                         std::vector<MapPoint*>& target)
    {
        if(!pMP)
            return;
        pMP->IncreaseVisible();
        target.push_back(pMP);
        nToMatch++;
        nLocalInFrustum++;
        if(CloseReuseFunnelStats* stats = closeReuseBucket(pMP, resolvedDepth))
            stats->admitted++;
        if(ReferenceAssociationFunnelStats* stats = referenceAssociationBucket(pMP))
            stats->admitted++;
        if(CloseReuseContributionStats* stats = closeReuseContributorBucket(pMP, resolvedDepth))
            stats->admitted++;
        if(CloseReuseContributionStats* stats = closeReuseKFContributionBucket(pMP, resolvedDepth))
            stats->admitted++;
        accumulateMappingFeatureReuse(pMP, resolvedDepth, &MappingFeatureReuseFunnelStats::admitted);
        if(bRecordCloseReuseCohort && IsCloseReuseTarget(pMP, resolvedDepth))
            pMP->RecordCloseReuseProjectionAttempt(mCurrentFrame.mnId);
        if(IsLocalProjHistoryTarget(pMP, resolvedDepth))
        {
            pMP->RecordLocalProjectionAttempt(mCurrentFrame.mnId);
            triang0To10ProjAttempt++;
        }
        if(pMP->GetCreationSource() == MapPoint::kCreationSourceTrackingStereo &&
           std::isfinite(resolvedDepth) && resolvedDepth <= mThDepth)
        {
            localProjCloseTrackingStereoCandidates++;
        }
        AccumulateMapPointQuality(mpQualityLocalInFrustum,
                                  pMP,
                                  mCurrentFrame,
                                  resolvedDepth,
                                  mThDepth);
        if((pMP->mbTrackInView || pMP->mbTrackInViewR) && bUseLightGlueLocalMap)
            localProjectedMapPointIds.insert(pMP->mnId);
        if(pMP->mbTrackInView)
        {
            nLocalTrackInView++;
            AccumulateMapPointQuality(mpQualityLocalTrackInView,
                                      pMP,
                                      mCurrentFrame,
                                      pMP->mTrackDepth,
                                      mThDepth);
            mCurrentFrame.mmProjectPoints[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);
        }
    };

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        const float selectedDepth = bCollectCloseReuseFunnel
            ? ResolveMapPointQualityDepth(mCurrentFrame, pMP, pMP->mTrackDepth)
            : std::numeric_limits<float>::quiet_NaN();
        CloseReuseFunnelStats* selectedStats = closeReuseBucket(pMP, selectedDepth);
        if(selectedStats)
            selectedStats->selected++;
        ReferenceAssociationFunnelStats* referenceStats = referenceAssociationBucket(pMP);
        if(referenceStats)
            referenceStats->selected++;
        if(CloseReuseContributionStats* stats = closeReuseContributorBucket(pMP, selectedDepth))
            stats->selected++;
        if(CloseReuseContributionStats* stats = closeReuseKFContributionBucket(pMP, selectedDepth))
            stats->selected++;
        accumulateMappingFeatureReuse(pMP, selectedDepth, &MappingFeatureReuseFunnelStats::selected);

        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
        {
            if(selectedStats)
                selectedStats->alreadySeen++;
            if(referenceStats)
                referenceStats->alreadySeen++;
            accumulateMappingFeatureReuse(pMP, selectedDepth, &MappingFeatureReuseFunnelStats::alreadySeen);
            nLocalAlreadySeen++;
            AccumulateMapPointQuality(mpQualityLocalAlreadySeen,
                                      pMP,
                                      mCurrentFrame,
                                      pMP->mTrackDepth,
                                      mThDepth);
            continue;
        }
        if(pMP->isBad())
        {
            if(selectedStats)
                selectedStats->bad++;
            if(referenceStats)
                referenceStats->bad++;
            accumulateMappingFeatureReuse(pMP, selectedDepth, &MappingFeatureReuseFunnelStats::bad);
            nLocalBad++;
            continue;
        }
        // Project (this fills MapPoint variables for matching). Record only
        // runtime visibility history for close/mid LMT points; it does not alter admission.
        const bool bRecordVisibilityHistory =
            bRecordCloseReuseCohort && IsCloseMidLmtAtCreation(pMP);
        const bool bInFrustum = mCurrentFrame.isInFrustum(pMP,0.5);
        if(bRecordVisibilityHistory)
        {
            pMP->RecordLocalMapVisibilityOutcome(mCurrentFrame.mnId, pMP->mTrackFrustumStatus);
            ++triang0To10VisibilityHistoryRecorded;
            const MapPoint::LocalMapVisibilityHistoryStats history =
                pMP->GetLocalMapVisibilityHistory();
            if(history.evaluationCount >= 5)
                ++triang0To10VisibilityHistoryEvaluated;
            if(IsLongLowRelevantLocalMapHistory(history))
            {
                ++triang0To10VisibilityHistoryLowRelevant;
                if(bInFrustum)
                    ++triang0To10VisibilityHistoryLowRelevantAccepted;
            }
        }
        if(bInFrustum)
        {
            const float resolvedDepth = ResolveMapPointQualityDepth(mCurrentFrame, pMP, pMP->mTrackDepth);
            CloseReuseFunnelStats* inFrustumStats = closeReuseBucket(pMP, resolvedDepth);
            if(inFrustumStats)
            {
                inFrustumStats->inFrustum++;
                inFrustumStats->frustumStatus[MapPoint::kTrackFrustumAccepted]++;
            }
            if(referenceStats)
            {
                ++referenceStats->inFrustum;
                ++referenceStats->frustumStatus[MapPoint::kTrackFrustumAccepted];
            }
            if(CloseReuseContributionStats* stats = closeReuseContributorBucket(pMP, resolvedDepth))
                stats->inFrustum++;
            if(CloseReuseContributionStats* stats = closeReuseKFContributionBucket(pMP, resolvedDepth))
                stats->inFrustum++;
            accumulateMappingFeatureReuse(pMP, resolvedDepth, &MappingFeatureReuseFunnelStats::inFrustum);

            const Tracking::LocalMPProvenance* pMotionGuardProvenance = nullptr;
            const auto itMotionGuardProv = mmLocalMPProvenance.find(pMP);
            if(itMotionGuardProv != mmLocalMPProvenance.end() &&
               itMotionGuardProv->second.frameId == mCurrentFrame.mnId)
            {
                pMotionGuardProvenance = &itMotionGuardProv->second;
            }
            if(shouldSkipByMotionHealthGuard(pMP, resolvedDepth, pMotionGuardProvenance))
            {
                pMP->mbTrackInView = false;
                pMP->mbTrackInViewR = false;
                continue;
            }

            if(bUseLocalFoundRatioGate)
            {
                const bool bGateThisMapPoint = ShouldGateLocalMapPointByFoundRatio(pMP);
                const float foundRatio = pMP->GetFoundRatio();
                const bool bDepthQualified = std::isfinite(resolvedDepth) && resolvedDepth > localMinFoundRatioDepth;
                if(bGateThisMapPoint && std::isfinite(foundRatio) && foundRatio < localMinFoundRatio)
                {
                    if(bDepthQualified)
                    {
                        if(inFrustumStats)
                            inFrustumStats->foundGateReject++;
                        if(referenceStats)
                            ++referenceStats->foundGateReject;
                        accumulateMappingFeatureReuse(pMP,
                                                      resolvedDepth,
                                                      &MappingFeatureReuseFunnelStats::foundGateReject);
                        nLocalFoundRatioRejected++;
                        localFoundRatioRejectedBySource[MapPointCreationSourceIndex(pMP->GetCreationSource())]++;
                        localFoundRatioRejectedDepthQualified++;
                        localFoundRatioRejectedDepthQualifiedBySource[MapPointCreationSourceIndex(pMP->GetCreationSource())]++;
                        AccumulateMapPointQuality(mpQualityFoundRatioRejected,
                                                  pMP,
                                                  mCurrentFrame,
                                                  resolvedDepth,
                                                  mThDepth);
                        pMP->mbTrackInView = false;
                        pMP->mbTrackInViewR = false;
                        continue;
                    }
                }
            }
            Tracking::LocalKFProvenanceRole localRole = Tracking::LocalKFProvenanceRole::Unknown;
            const Tracking::LocalMPProvenance* pLocalProvenance = nullptr;
            const auto itLocalProv = mmLocalMPProvenance.find(pMP);
            if(itLocalProv != mmLocalMPProvenance.end() &&
               itLocalProv->second.frameId == mCurrentFrame.mnId)
            {
                localRole = itLocalProv->second.bestRole;
                pLocalProvenance = &itLocalProv->second;
            }
            if(IsLocalProjHistoryTarget(pMP, resolvedDepth))
            {
                triang0To10HistoryGateCandidate++;
                const MapPoint::LocalProjectionHistoryStats localProjHistory =
                    pMP->GetLocalProjectionHistory();
                const int optOutcomes = localProjHistory.optInlierCount +
                                        localProjHistory.optOutlierCount;
                const int recentValid = CountHistoryBits(localProjHistory.recentValidMask);
                const bool bInsufficientHistory = optOutcomes < 3 && recentValid < 3;
                const bool bRejectByOptHistory =
                    optOutcomes >= 3 && LocalProjOptOutlierRate(localProjHistory) > 0.35f;
                const bool bRejectByRecentHistory =
                    recentValid >= 3 && LocalProjRecentOutlierRate(localProjHistory) > 0.40f;

                if(bInsufficientHistory)
                    triang0To10HistoryGateInsufficient++;
                if(bRejectByOptHistory || bRejectByRecentHistory)
                {
                    if(inFrustumStats)
                        inFrustumStats->historyGateReject++;
                    if(referenceStats)
                        ++referenceStats->historyGateReject;
                    accumulateMappingFeatureReuse(pMP,
                                                  resolvedDepth,
                                                  &MappingFeatureReuseFunnelStats::historyGateReject);
                    triang0To10HistoryGateReject++;
                    if(bRejectByOptHistory)
                        triang0To10HistoryGateRejectOpt++;
                    if(bRejectByRecentHistory)
                        triang0To10HistoryGateRejectRecent++;
                    pMP->mbTrackInView = false;
                    pMP->mbTrackInViewR = false;
                    continue;
                }
                triang0To10HistoryGatePass++;
            }
            if(IsLocalProjHistoryTarget(pMP, resolvedDepth))
            {
                switch(localRole)
                {
                    case Tracking::LocalKFProvenanceRole::Reference:
                        localmapLmt0To10FromReference++;
                        break;
                    case Tracking::LocalKFProvenanceRole::Voted:
                        localmapLmt0To10FromVoted++;
                        break;
                    case Tracking::LocalKFProvenanceRole::Covis:
                    case Tracking::LocalKFProvenanceRole::ChildParent:
                        localmapLmt0To10FromExpansion++;
                        break;
                    case Tracking::LocalKFProvenanceRole::Temporal:
                        localmapLmt0To10FromTemporal++;
                        break;
                    case Tracking::LocalKFProvenanceRole::Unknown:
                    default:
                        localmapLmt0To10FromUnknown++;
                        break;
                }

                const int priorityBin = LocalProjPriorityBin(pMP, pLocalProvenance, resolvedDepth);
                if(priorityBin >= 0 && priorityBin < kLocalProjPriorityBinCount)
                    triang0To10PriorityCandidate[priorityBin]++;
            }
            const bool bCoreLocalProjectionPoint =
                isCoreLocalProjectionPoint(pMP, resolvedDepth, pLocalProvenance);
            if(bUseLocalMapCoreFirst && !bCoreLocalProjectionPoint)
            {
                DeferredLocalProjectionPoint deferred;
                deferred.pMP = pMP;
                deferred.depth = resolvedDepth;
                vDeferredExpansionLocalPoints.push_back(deferred);
                ++coreFirstExpansionCandidates;
            }
            else
            {
                if(bUseLocalMapCoreFirst)
                {
                    ++coreFirstCoreCandidates;
                    admitLocalProjectionPoint(pMP,
                                              resolvedDepth,
                                              vpCoreLocalProjectionMapPoints);
                }
                else
                {
                    admitLocalProjectionPoint(pMP,
                                              resolvedDepth,
                                              vpLocalProjectionMapPoints);
                }
            }
        }
        else if(selectedStats)
        {
            selectedStats->frustumStatus[CloseReuseFrustumStatusBin(pMP->mTrackFrustumStatus)]++;
            if(referenceStats)
                referenceStats->frustumStatus[CloseReuseFrustumStatusBin(pMP->mTrackFrustumStatus)]++;
        }
        else if(referenceStats)
        {
            referenceStats->frustumStatus[CloseReuseFrustumStatusBin(pMP->mTrackFrustumStatus)]++;
        }
    }

    auto admitDeferredExpansionPoints =
        [&](const std::vector<DeferredLocalProjectionPoint>& deferredPoints)
    {
        for(const DeferredLocalProjectionPoint& deferred : deferredPoints)
        {
            if(!deferred.pMP || deferred.pMP->isBad())
                continue;
            admitLocalProjectionPoint(deferred.pMP,
                                      deferred.depth,
                                      vpExpansionLocalProjectionMapPoints);
            ++coreFirstExpansionAdmitted;
        }
    };

    auto suppressDeferredExpansionPoints =
        [&](const std::vector<DeferredLocalProjectionPoint>& deferredPoints)
    {
        for(const DeferredLocalProjectionPoint& deferred : deferredPoints)
        {
            if(!deferred.pMP)
                continue;
            deferred.pMP->mbTrackInView = false;
            deferred.pMP->mbTrackInViewR = false;
            ++coreFirstExpansionSuppressed;
        }
    };

    if(nToMatch>0 ||
       (bUseLocalMapCoreFirst && !vDeferredExpansionLocalPoints.empty()))
    {
        const bool bMonocularVisual = (mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);
        int th = 1;
        if(mSensor==System::RGBD || mSensor==System::IMU_RGBD)
            th=3;
        if(mpAtlas->isImuInitialized())
        {
            if(mpAtlas->GetCurrentMap()->GetIniertialBA2())
                th=2;
            else
                th=6;
        }
        else if(!mpAtlas->isImuInitialized() && (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD))
        {
            th=10;
        }

        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;

        if(mState==LOST || mState==RECENTLY_LOST) // Lost for less than 1 second
            th=15; // 15

        bool bFarFilter = mpLocalMapper->mbFarPoints;
        float thFarPoints = mpLocalMapper->mThFarPoints;
        bool bNearOnlyApplied = false;
        if(!bUseORB && bMonocularVisual && IsXFeatMonoNearOnlyEnabled())
        {
            float medianDepth = -1.0f;
            if(mpReferenceKF)
                medianDepth = mpReferenceKF->ComputeSceneMedianDepth(2);
            if(std::isfinite(medianDepth) && medianDepth > 0.0f)
            {
                bFarFilter = true;
                thFarPoints = medianDepth * GetXFeatMonoNearDepthFactor();
                bNearOnlyApplied = true;
            }
        }
        projectionTh = th;
        bProjectionRun = true;
        projectionFarFilter = bFarFilter;
        projectionFarPoints = thFarPoints;
        projectionNearOnlyApplied = bNearOnlyApplied;

        std::vector<unsigned char> vHadMapPointBeforeProjection(static_cast<size_t>(mCurrentFrame.N), 0);
        EnsureXFeatMatchSourceSize(mCurrentFrame);
        for(int i = 0; i < mCurrentFrame.N; ++i)
        {
            if(mCurrentFrame.mvpMapPoints[i])
                vHadMapPointBeforeProjection[static_cast<size_t>(i)] = 1;
        }

        int matches = 0;
        XFeatMatcher::ProjectionMapPointStats projectionStatsTotal;
        auto AccumulateProjectionStats = [&](const XFeatMatcher::ProjectionMapPointStats& stats)
        {
            projectionStatsTotal.close_tracking_stereo_reject_ur += stats.close_tracking_stereo_reject_ur;
            projectionStatsTotal.close_tracking_stereo_reject_depth += stats.close_tracking_stereo_reject_depth;
            projectionStatsTotal.close_tracking_stereo_accepted += stats.close_tracking_stereo_accepted;
            projectionStatsTotal.local_proj_triang_0_10_proj_outside_reject += stats.local_proj_triang_0_10_proj_outside_reject;
            projectionStatsTotal.local_proj_triang_0_10_companion_outside_reject += stats.local_proj_triang_0_10_companion_outside_reject;
            projectionStatsTotal.ref_target_input += stats.ref_target_input;
            projectionStatsTotal.ref_target_skip_not_in_view += stats.ref_target_skip_not_in_view;
            projectionStatsTotal.ref_target_skip_far += stats.ref_target_skip_far;
            projectionStatsTotal.ref_target_skip_bad += stats.ref_target_skip_bad;
            projectionStatsTotal.ref_target_skip_no_desc += stats.ref_target_skip_no_desc;
            projectionStatsTotal.ref_target_projected += stats.ref_target_projected;
            projectionStatsTotal.ref_target_side_attempt += stats.ref_target_side_attempt;
            projectionStatsTotal.ref_target_invalid_level += stats.ref_target_invalid_level;
            projectionStatsTotal.ref_target_empty_area += stats.ref_target_empty_area;
            projectionStatsTotal.ref_target_area_candidates += stats.ref_target_area_candidates;
            projectionStatsTotal.ref_target_invalid_candidate += stats.ref_target_invalid_candidate;
            projectionStatsTotal.ref_target_occupied_candidate += stats.ref_target_occupied_candidate;
            projectionStatsTotal.ref_target_geom_ur_reject += stats.ref_target_geom_ur_reject;
            projectionStatsTotal.ref_target_geom_depth_reject += stats.ref_target_geom_depth_reject;
            projectionStatsTotal.ref_target_valid_candidate += stats.ref_target_valid_candidate;
            projectionStatsTotal.ref_target_no_valid_candidate += stats.ref_target_no_valid_candidate;
            projectionStatsTotal.ref_target_raw_candidate += stats.ref_target_raw_candidate;
            projectionStatsTotal.ref_target_dist_reject += stats.ref_target_dist_reject;
            projectionStatsTotal.ref_target_ratio_reject += stats.ref_target_ratio_reject;
            projectionStatsTotal.ref_target_candidate_pool += stats.ref_target_candidate_pool;
            projectionStatsTotal.ref_target_spatial_reject += stats.ref_target_spatial_reject;
            projectionStatsTotal.ref_target_assign_reject += stats.ref_target_assign_reject;
            projectionStatsTotal.ref_target_outside_reject += stats.ref_target_outside_reject;
            projectionStatsTotal.ref_target_assigned += stats.ref_target_assigned;
        };

        if(bUseORB)
        {
            ORBmatcher matcher(0.8);
            matches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th, bFarFilter, thFarPoints);
        }
        else
        {
            if(bUseLocalMapCoreFirst)
            {
                XFeatMatcher matcher(0.8f, false);
                if(!vpCoreLocalProjectionMapPoints.empty())
                {
                    coreFirstCoreMatches = matcher.SearchByProjection(mCurrentFrame,
                                                                      vpCoreLocalProjectionMapPoints,
                                                                      th,
                                                                      bFarFilter,
                                                                      thFarPoints,
                                                                      GetXFeatThHighLocalProjection(),
                                                                      -1,
                                                                      true);
                    matches += coreFirstCoreMatches;
                    AccumulateProjectionStats(matcher.LastProjectionMapPointStats());
                }

                const bool bHasAnyExpansion = !vDeferredExpansionLocalPoints.empty();
                bool bNeedExpansion = bHasAnyExpansion;
                if(!bHasAnyExpansion)
                {
                    coreFirstDecision = "no_expansion";
                    bNeedExpansion = false;
                }
                else if(vpCoreLocalProjectionMapPoints.empty())
                {
                    coreFirstDecision = "no_core";
                }
                else if(nAlreadyMatched < coreFirstMinPrior)
                {
                    coreFirstDecision = "prior_weak";
                }
                else if(coreFirstCoreMatches < coreFirstMinCoreMatches)
                {
                    coreFirstDecision = "core_weak";
                }
                else if(nAlreadyMatched + coreFirstCoreMatches < coreFirstMinTotalMatches)
                {
                    coreFirstDecision = "total_weak";
                }
                else
                {
                    coreFirstDecision = "skip_expansion";
                    bNeedExpansion = false;
                }

                if(bNeedExpansion)
                {
                    admitDeferredExpansionPoints(vDeferredExpansionLocalPoints);
                    if(!vpExpansionLocalProjectionMapPoints.empty())
                    {
                        coreFirstExpansionMatches =
                            matcher.SearchByProjection(mCurrentFrame,
                                                       vpExpansionLocalProjectionMapPoints,
                                                       th,
                                                       bFarFilter,
                                                       thFarPoints,
                                                       GetXFeatThHighLocalProjection(),
                                                       -1,
                                                       false);
                        matches += coreFirstExpansionMatches;
                        AccumulateProjectionStats(matcher.LastProjectionMapPointStats());
                    }
                }
                else
                {
                    suppressDeferredExpansionPoints(vDeferredExpansionLocalPoints);
                }
            }
            else if(!vpLocalProjectionMapPoints.empty())
            {
                XFeatMatcher matcher(0.8f, false);
                matches = matcher.SearchByProjection(mCurrentFrame,
                                                     vpLocalProjectionMapPoints,
                                                     th,
                                                     bFarFilter,
                                                     thFarPoints,
                                                     GetXFeatThHighLocalProjection(),
                                                     -1,
                                                     true);
                AccumulateProjectionStats(matcher.LastProjectionMapPointStats());
            }

            localProjCloseTrackingStereoRejectUr = projectionStatsTotal.close_tracking_stereo_reject_ur;
            localProjCloseTrackingStereoRejectDepth = projectionStatsTotal.close_tracking_stereo_reject_depth;
            localProjTriang0To10ProjOutsideReject = projectionStatsTotal.local_proj_triang_0_10_proj_outside_reject;
            localProjTriang0To10CompanionOutsideReject = projectionStatsTotal.local_proj_triang_0_10_companion_outside_reject;
            refAssocMatcherInput = projectionStatsTotal.ref_target_input;
            refAssocMatcherSkipNotInView = projectionStatsTotal.ref_target_skip_not_in_view;
            refAssocMatcherSkipFar = projectionStatsTotal.ref_target_skip_far;
            refAssocMatcherSkipBad = projectionStatsTotal.ref_target_skip_bad;
            refAssocMatcherSkipNoDesc = projectionStatsTotal.ref_target_skip_no_desc;
            refAssocMatcherProjected = projectionStatsTotal.ref_target_projected;
            refAssocMatcherSideAttempt = projectionStatsTotal.ref_target_side_attempt;
            refAssocMatcherInvalidLevel = projectionStatsTotal.ref_target_invalid_level;
            refAssocMatcherEmptyArea = projectionStatsTotal.ref_target_empty_area;
            refAssocMatcherAreaCandidates = projectionStatsTotal.ref_target_area_candidates;
            refAssocMatcherInvalidCandidate = projectionStatsTotal.ref_target_invalid_candidate;
            refAssocMatcherOccupiedCandidate = projectionStatsTotal.ref_target_occupied_candidate;
            refAssocMatcherGeomUrReject = projectionStatsTotal.ref_target_geom_ur_reject;
            refAssocMatcherGeomDepthReject = projectionStatsTotal.ref_target_geom_depth_reject;
            refAssocMatcherValidCandidate = projectionStatsTotal.ref_target_valid_candidate;
            refAssocMatcherNoValidCandidate = projectionStatsTotal.ref_target_no_valid_candidate;
            refAssocMatcherRawCandidate = projectionStatsTotal.ref_target_raw_candidate;
            refAssocMatcherDistReject = projectionStatsTotal.ref_target_dist_reject;
            refAssocMatcherRatioReject = projectionStatsTotal.ref_target_ratio_reject;
            refAssocMatcherCandidatePool = projectionStatsTotal.ref_target_candidate_pool;
            refAssocMatcherSpatialReject = projectionStatsTotal.ref_target_spatial_reject;
            refAssocMatcherAssignReject = projectionStatsTotal.ref_target_assign_reject;
            refAssocMatcherOutsideReject = projectionStatsTotal.ref_target_outside_reject;
            refAssocMatcherAssigned = projectionStatsTotal.ref_target_assigned;
        }
        projectionMatches = matches;

        std::vector<unsigned char> vProjectionAdded(static_cast<size_t>(mCurrentFrame.N), 0);
        std::vector<unsigned char> vProjectionVerified(static_cast<size_t>(mCurrentFrame.N), 0);
        std::vector<unsigned char> vProjectionConflict(static_cast<size_t>(mCurrentFrame.N), 0);
        std::vector<unsigned char> vMapUpdatedEdgeTarget(static_cast<size_t>(mCurrentFrame.N), 0);
        auto IsMapUpdatedEdgeVerificationTarget = [&](const int idx) -> bool
        {
            if(!mbMapUpdated || idx < 0 || idx >= mCurrentFrame.N)
                return false;
            if(!vProjectionAdded[static_cast<size_t>(idx)])
                return false;
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[idx];
            if(!pMP)
                return false;
            const float resolvedDepth =
                ResolveMapPointQualityDepth(mCurrentFrame, pMP, pMP->mTrackDepth);
            if(!IsLocalProjHistoryTarget(pMP, resolvedDepth))
                return false;
            if(idx >= static_cast<int>(mCurrentFrame.mvLocalProjProjErrNorm.size()))
                return false;
            const float projErrNorm = mCurrentFrame.mvLocalProjProjErrNorm[static_cast<size_t>(idx)];
            return std::isfinite(projErrNorm) && projErrNorm > 0.50f;
        };
        for(int i = 0; i < mCurrentFrame.N; ++i)
        {
            if(!vHadMapPointBeforeProjection[static_cast<size_t>(i)] && mCurrentFrame.mvpMapPoints[i])
            {
                vProjectionAdded[static_cast<size_t>(i)] = 1;
                if(i < static_cast<int>(mCurrentFrame.mvXFeatMatchSource.size()))
                    mCurrentFrame.mvXFeatMatchSource[static_cast<size_t>(i)] = kXFeatMatchLocalProjection;
                if(bUseLightGlueLocalMap)
                {
                    ++lgLocalVerifyCandidates;
                    if(IsMapUpdatedEdgeVerificationTarget(i))
                    {
                        vMapUpdatedEdgeTarget[static_cast<size_t>(i)] = 1;
                        ++lgLocalMapUpdatedEdgeCandidates;
                    }
                }
            }
        }

        if(bUseLightGlueLocalMap && !localProjectedMapPointIds.empty())
        {
            try
            {
                static XFeatLighterGlueMatcher lightGlueMatcher;
                const float lgConf = GetEnvFloatInRange("XFEAT_LIGHTGLUE_CONF", 0.1f, 0.0f, 1.0f);
                const int maxLocalKFs = GetXFeatLightGlueLocalMapMaxKFs();
                const float projRadius = GetXFeatLightGlueLocalMapProjRadius();
                const float projRadiusSq = projRadius * projRadius;
                const float conflictMinScore = GetXFeatLightGlueLocalConflictMinScore();

                std::vector<KeyFrame*> vpLightGlueKFs;
                vpLightGlueKFs.reserve(static_cast<size_t>(maxLocalKFs));
                std::unordered_set<long unsigned int> usedKFIds;
                usedKFIds.reserve(static_cast<size_t>(maxLocalKFs) + 1);

                auto AddLightGlueKF = [&](KeyFrame* pKF)
                {
                    if(!pKF || pKF->isBad())
                        return;
                    if(static_cast<int>(vpLightGlueKFs.size()) >= maxLocalKFs)
                        return;
                    if(!usedKFIds.insert(pKF->mnId).second)
                        return;
                    vpLightGlueKFs.push_back(pKF);
                };

                AddLightGlueKF(mpReferenceKF);

                std::vector<KeyFrame*> vpSortedLocalKFs = mvpLocalKeyFrames;
                std::sort(vpSortedLocalKFs.begin(), vpSortedLocalKFs.end(),
                          [](const KeyFrame* a, const KeyFrame* b)
                          {
                              if(!a)
                                  return false;
                              if(!b)
                                  return true;
                              return a->mnId > b->mnId;
                          });
                for(KeyFrame* pKF : vpSortedLocalKFs)
                {
                    AddLightGlueKF(pKF);
                    if(static_cast<int>(vpLightGlueKFs.size()) >= maxLocalKFs)
                        break;
                }

                for(KeyFrame* pKF : vpLightGlueKFs)
                {
                    std::vector<MapPoint*> vpLightGlueMatches;
                    std::vector<float> vLightGlueScores;
                    lightGlueMatcher.SearchByLightGlue(pKF,
                                                       mCurrentFrame,
                                                       vpLightGlueMatches,
                                                       lgConf,
                                                       &vLightGlueScores);
                    const XFeatLighterGlueMatcher::Stats& lgStats = lightGlueMatcher.LastStats();
                    ++lgLocalKFsUsed;
                    lgLocalRawMatches += lgStats.lg_raw_matches;
                    lgLocalMpValid += lgStats.mp_valid;
                    lgLocalOneToOne += lgStats.one_to_one;

                    const size_t nMatches = std::min(vpLightGlueMatches.size(),
                                                     mCurrentFrame.mvpMapPoints.size());
                    for(size_t idx = 0; idx < nMatches; ++idx)
                    {
                        MapPoint* pMP = vpLightGlueMatches[idx];
                        if(!pMP || pMP->isBad())
                            continue;

                        if(localProjectedMapPointIds.find(pMP->mnId) == localProjectedMapPointIds.end())
                        {
                            ++lgLocalNotProjected;
                            continue;
                        }

                        if(idx >= mCurrentFrame.mvKeysUn.size())
                            continue;
                        const cv::Point2f& kp = mCurrentFrame.mvKeysUn[idx].pt;
                        const float dx = pMP->mTrackProjX - kp.x;
                        const float dy = pMP->mTrackProjY - kp.y;
                        if(dx * dx + dy * dy > projRadiusSq)
                        {
                            ++lgLocalProjReject;
                            continue;
                        }

                        if(mCurrentFrame.mvpMapPoints[idx])
                        {
                            const bool bMapUpdatedEdgeTarget =
                                idx < vMapUpdatedEdgeTarget.size() && vMapUpdatedEdgeTarget[idx];
                            if(vProjectionAdded[idx] && mCurrentFrame.mvpMapPoints[idx] == pMP)
                            {
                                if(bMapUpdatedEdgeTarget && !vProjectionVerified[idx])
                                {
                                    vProjectionVerified[idx] = 1;
                                    ++lgLocalVerified;
                                    ++lgLocalMapUpdatedEdgeVerified;
                                }
                            }
                            else if(vProjectionAdded[idx])
                            {
                                const float lgScore = idx < vLightGlueScores.size()
                                    ? vLightGlueScores[idx]
                                    : -1.0f;
                                if(lgScore >= conflictMinScore)
                                {
                                    if(bMapUpdatedEdgeTarget)
                                    {
                                        vProjectionConflict[idx] = 1;
                                        ++lgLocalMismatch;
                                    }
                                    else
                                    {
                                        ++lgLocalMapUpdatedEdgeNonTargetIgnored;
                                    }
                                }
                                else if(bMapUpdatedEdgeTarget)
                                {
                                    ++lgLocalConflictLowScore;
                                }
                            }
                            ++lgLocalOccupied;
                            continue;
                        }

                        // Verification mode does not add new matches; empty slots are left for the baseline path.
                    }
                }

                for(int i = 0; i < mCurrentFrame.N; ++i)
                {
                    if(!vProjectionAdded[static_cast<size_t>(i)])
                        continue;
                    if(!vMapUpdatedEdgeTarget[static_cast<size_t>(i)])
                        continue;
                    if(vProjectionVerified[static_cast<size_t>(i)])
                        continue;
                    if(!vProjectionConflict[static_cast<size_t>(i)])
                    {
                        ++lgLocalMapUpdatedEdgeNoEvidence;
                        continue;
                    }
                    ++lgLocalMapUpdatedEdgeConflictUnverified;
                }
                matches = std::max(0, matches - lgLocalRejected);
                projectionMatches = matches;
            }
            catch(const std::exception& e)
            {
                bLightGlueLocalFailed = true;
                if(ShouldPrintXFeatDebug(mCurrentFrame.mnId))
                {
                    std::cerr << "[SearchLocalPoints][LightGlue] failed, continuing with projection-only local map: "
                              << e.what() << std::endl;
                }
            }
        }

        for(int i = 0; i < mCurrentFrame.N; ++i)
        {
            if(!vProjectionAdded[static_cast<size_t>(i)] || !mCurrentFrame.mvpMapPoints[i])
                continue;
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            const float resolvedDepth = ResolveMapPointQualityDepth(mCurrentFrame, pMP, pMP->mTrackDepth);
            if(pMP->GetCreationSource() == MapPoint::kCreationSourceTrackingStereo &&
               std::isfinite(resolvedDepth) && resolvedDepth <= mThDepth)
            {
                localProjCloseTrackingStereoGeomAccept++;
            }
            if(CloseReuseFunnelStats* stats = closeReuseBucket(pMP, resolvedDepth))
                stats->projectionMatch++;
            if(ReferenceAssociationFunnelStats* stats = referenceAssociationBucket(pMP))
                stats->projectionMatch++;
            if(CloseReuseContributionStats* stats = closeReuseContributorBucket(pMP, resolvedDepth))
                stats->projectionMatch++;
            if(CloseReuseContributionStats* stats = closeReuseKFContributionBucket(pMP, resolvedDepth))
                stats->projectionMatch++;
            accumulateMappingFeatureReuse(pMP,
                                          resolvedDepth,
                                          &MappingFeatureReuseFunnelStats::projectionMatch);
            if(bRecordCloseReuseCohort && IsCloseReuseTarget(pMP, resolvedDepth))
                pMP->RecordCloseReuseProjectionMatch(mCurrentFrame.mnId);
            if(IsLocalProjHistoryTarget(pMP, resolvedDepth))
            {
                pMP->RecordLocalProjectionMatch(mCurrentFrame.mnId);
                triang0To10ProjMatch++;
                const Tracking::LocalMPProvenance* pProvenance = nullptr;
                const auto itProv = mmLocalMPProvenance.find(pMP);
                if(itProv != mmLocalMPProvenance.end() &&
                   itProv->second.frameId == mCurrentFrame.mnId)
                {
                    pProvenance = &itProv->second;
                }
                const int priorityBin = LocalProjPriorityBin(pMP, pProvenance, resolvedDepth);
                if(priorityBin >= 0 && priorityBin < kLocalProjPriorityBinCount)
                    triang0To10PriorityMatch[priorityBin]++;
            }
        }

        // if (std::getenv("USE_ORB") == nullptr)
        // {
        //     matches = matcher.SearchByNN(mCurrentFrame,mvpLocalMapPoints);
        // }
        // else
        // {
        //     matches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th, mpLocalMapper->mbFarPoints, mpLocalMapper->mThFarPoints);
        // }     
    }
    auto LogSearchLocalPointsDiagnostics = [&](const int matchesValue,
                                               const int nToMatchValue,
                                               const int thValue,
                                               const bool bProjectionRun,
                                               const bool bFarFilterValue,
                                               const float thFarPointsValue,
                                               const bool bNearOnlyAppliedValue)
    {
        const bool bLowSupportDiag = IsXFeatFeatureDiagEnabled() && matchesValue < 80;
        if(!ShouldPrintXFeatDebug(mCurrentFrame.mnId) && !bLowSupportDiag)
            return;

        const bool bDeepSearchLocalDiag = IsXFeatDeepDiagEnabled();
        const bool bHasLocalFoundRatioSourceDiag =
            bDeepSearchLocalDiag || nLocalFoundRatioRejected > 0;
        const bool bHasLocalKFVisibilityDiag = bDeepSearchLocalDiag;
        const bool bHasTriangHistoryDiag = bDeepSearchLocalDiag;
        const bool bHasTriangVisibilityHistoryDiag = bDeepSearchLocalDiag;
        const bool bHasLocalMapRoleDiag = bDeepSearchLocalDiag;
        const bool bHasPriorityDiag = bDeepSearchLocalDiag;
        const bool bHasProjectionEdgeDiag =
            bDeepSearchLocalDiag ||
            localProjTriang0To10ProjOutsideReject > 0 ||
            localProjTriang0To10CompanionOutsideReject > 0;
        const bool bHasRefAssocMatcherDiag = bDeepSearchLocalDiag;
        const bool bHasLightGlueLocalDiag =
            bDeepSearchLocalDiag ||
            bUseLightGlueLocalMap ||
            bLightGlueLocalFailed ||
            lgLocalRawMatches > 0 ||
            lgLocalFilled > 0 ||
            lgLocalVerifyCandidates > 0 ||
            lgLocalMapUpdatedEdgeCandidates > 0 ||
            lgLocalMapUpdatedEdgeConflictUnverified > 0;
        const bool bHasCoreFirstDiag =
            bDeepSearchLocalDiag ||
            bUseLocalMapCoreFirst ||
            coreFirstExpansionCandidates > 0 ||
            coreFirstExpansionSuppressed > 0;

        if(ShouldPrintXFeatDebug(mCurrentFrame.mnId) && bMotionGuardEnabled)
        {
            std::cout << "[TrackLocalMap][MotionGuard]"
                      << " frame_id=" << mCurrentFrame.mnId
                      << " enabled=1"
                      << " active=" << (bMotionGuardActive ? 1 : 0)
                      << " health_valid=" << (mbXFeatMotionHealthValid ? 1 : 0)
                      << " health_frame_id=" << mnXFeatMotionHealthFrameId
                      << " motion_assoc=" << mnXFeatMotionHealthAssociations
                      << " motion_inliers=" << mnXFeatMotionHealthInliers
                      << " motion_outlier_ratio=" << mXFeatMotionHealthOutlierRatio
                      << " motion_weak=" << (mbXFeatMotionHealthWeak ? 1 : 0)
                      << " motion_bad=" << (mbXFeatMotionHealthBad ? 1 : 0)
                      << " min_obs=" << motionGuardMinObs
                      << " min_found_ratio=" << motionGuardMinFoundRatio
                      << " candidates=" << motionGuardCandidates
                      << " low_risk=" << motionGuardLowRisk
                      << " mid_risk=" << motionGuardMidRisk
                      << " high_risk=" << motionGuardHighRisk
                      << " skipped=" << motionGuardSkipped
                      << " skip_low_obs=" << motionGuardSkipLowObs
                      << " skip_low_found_ratio=" << motionGuardSkipLowFoundRatio
                      << " skip_unknown_source=" << motionGuardSkipUnknownSource
                      << " skip_weak_provenance=" << motionGuardSkipWeakProvenance
                      << " skip_history=" << motionGuardSkipHistory
                      << " skip_other=" << motionGuardSkipOther
                      << std::endl;
        }

        std::cout << "[SearchLocalPoints] frame=" << mCurrentFrame.mnId
                  << " local_mps=" << mvpLocalMapPoints.size()
                  << " already_matched=" << nAlreadyMatched
                  << " bad_already_matched=" << nBadAlreadyMatched
                  << " local_already_seen=" << nLocalAlreadySeen
                  << " local_bad=" << nLocalBad
                  << " local_found_ratio_gate=" << (bUseLocalFoundRatioGate ? 1 : 0)
                  << " local_min_found_ratio=" << localMinFoundRatio
                  << " local_found_ratio_min_depth=" << localMinFoundRatioDepth
                  << " local_found_ratio_scope=" << localFoundRatioScope
                  << " local_found_ratio_rejected=" << nLocalFoundRatioRejected
                  << " local_kfs_voted_candidates=" << mnLocalMapVotedKFCandidates
                  << " local_kfs_voted_kept=" << mnLocalMapVotedKFKept
                  << " local_in_frustum=" << nLocalInFrustum
                  << " local_track_in_view=" << nLocalTrackInView
                  << " local_proj_close_tracking_stereo_candidates=" << localProjCloseTrackingStereoCandidates
                  << " local_proj_close_tracking_stereo_geom_accept=" << localProjCloseTrackingStereoGeomAccept
                  << " local_proj_close_tracking_stereo_reject_ur=" << localProjCloseTrackingStereoRejectUr
                  << " local_proj_close_tracking_stereo_reject_depth=" << localProjCloseTrackingStereoRejectDepth
                  << " nToMatch=" << nToMatchValue
                  << " th=" << thValue
                  << " mode=" << (bUseORB ? "ORBProj" : (bUseLightGlueLocalMap ? "XFeatProj+LightGlueLocal" : "XFeatProj"))
                  << " th_high=" << (bUseORB ? -1.0f : GetXFeatThHighLocalProjection());
        if(bHasCoreFirstDiag)
        {
            std::cout << " localmap_core_first=" << (bUseLocalMapCoreFirst ? 1 : 0)
                      << " core_first_prior=" << nAlreadyMatched
                      << " core_first_min_prior=" << coreFirstMinPrior
                      << " core_first_min_core_matches=" << coreFirstMinCoreMatches
                      << " core_first_min_total_matches=" << coreFirstMinTotalMatches
                      << " core_first_core_candidates=" << coreFirstCoreCandidates
                      << " core_first_expansion_candidates=" << coreFirstExpansionCandidates
                      << " core_first_expansion_admitted=" << coreFirstExpansionAdmitted
                      << " core_first_expansion_suppressed=" << coreFirstExpansionSuppressed
                      << " core_first_core_matches=" << coreFirstCoreMatches
                      << " core_first_expansion_matches=" << coreFirstExpansionMatches
                      << " core_first_decision=" << coreFirstDecision;
        }
        if(bHasLocalFoundRatioSourceDiag)
        {
            std::cout << " local_found_ratio_rejected_depth_qualified=" << localFoundRatioRejectedDepthQualified
                      << " local_found_ratio_rejected_depth_qualified_local_mapping_triangulation="
                      << localFoundRatioRejectedDepthQualifiedBySource[MapPointCreationSourceIndex(MapPoint::kCreationSourceLocalMappingTriangulation)]
                      << " local_found_ratio_rejected_tracking_stereo="
                      << localFoundRatioRejectedBySource[MapPointCreationSourceIndex(MapPoint::kCreationSourceTrackingStereo)]
                      << " local_found_ratio_rejected_local_mapping_triangulation="
                      << localFoundRatioRejectedBySource[MapPointCreationSourceIndex(MapPoint::kCreationSourceLocalMappingTriangulation)]
                      << " local_found_ratio_rejected_mono_init="
                      << localFoundRatioRejectedBySource[MapPointCreationSourceIndex(MapPoint::kCreationSourceMonoInit)]
                      << " local_found_ratio_rejected_stereo_init="
                      << localFoundRatioRejectedBySource[MapPointCreationSourceIndex(MapPoint::kCreationSourceStereoInit)]
                      << " local_found_ratio_rejected_unknown="
                      << localFoundRatioRejectedBySource[MapPointCreationSourceIndex(MapPoint::kCreationSourceUnknown)];
        }
        if(bHasLocalKFVisibilityDiag)
        {
            std::cout << " local_kfs_voted_visibility_kept=" << mnLocalMapVotedKFVisibilityKept;
        }
        if(bHasTriangHistoryDiag)
        {
            std::cout << " triang_0_10_history_gate_candidate=" << triang0To10HistoryGateCandidate
                      << " triang_0_10_history_gate_reject=" << triang0To10HistoryGateReject
                      << " triang_0_10_history_gate_reject_opt=" << triang0To10HistoryGateRejectOpt
                      << " triang_0_10_history_gate_reject_recent=" << triang0To10HistoryGateRejectRecent
                      << " triang_0_10_history_gate_pass=" << triang0To10HistoryGatePass
                      << " triang_0_10_history_gate_insufficient=" << triang0To10HistoryGateInsufficient
                      << " triang_0_10_proj_attempt=" << triang0To10ProjAttempt
                      << " triang_0_10_proj_match=" << triang0To10ProjMatch
                      << " triang_0_10_proj_miss=" << std::max(0, triang0To10ProjAttempt - triang0To10ProjMatch)
                      << " triang_0_10_proj_match_rate="
                      << (triang0To10ProjAttempt > 0
                          ? static_cast<float>(triang0To10ProjMatch) / static_cast<float>(triang0To10ProjAttempt)
                          : 0.0f);
        }
        if(bHasTriangVisibilityHistoryDiag)
        {
            std::cout << " triang_0_10_visibility_history_recorded=" << triang0To10VisibilityHistoryRecorded
                      << " triang_0_10_visibility_history_evaluated=" << triang0To10VisibilityHistoryEvaluated
                      << " triang_0_10_visibility_history_low_relevant=" << triang0To10VisibilityHistoryLowRelevant
                      << " triang_0_10_visibility_history_low_relevant_accepted="
                      << triang0To10VisibilityHistoryLowRelevantAccepted;
        }
        if(bHasLocalMapRoleDiag)
        {
            std::cout << " localmap_lmt_0_10_from_reference=" << localmapLmt0To10FromReference
                      << " localmap_lmt_0_10_from_voted=" << localmapLmt0To10FromVoted
                      << " localmap_lmt_0_10_from_expansion=" << localmapLmt0To10FromExpansion
                      << " localmap_lmt_0_10_from_temporal=" << localmapLmt0To10FromTemporal
                      << " localmap_lmt_0_10_from_unknown=" << localmapLmt0To10FromUnknown;
        }
        if(bHasPriorityDiag)
        {
            std::cout << " triang_0_10_priority_candidate_reference=" << triang0To10PriorityCandidate[1]
                      << " triang_0_10_priority_candidate_core_confirmed=" << triang0To10PriorityCandidate[2]
                      << " triang_0_10_priority_candidate_stable_history=" << triang0To10PriorityCandidate[3]
                      << " triang_0_10_priority_candidate_weak_vote=" << triang0To10PriorityCandidate[4]
                      << " triang_0_10_priority_candidate_single_obs=" << triang0To10PriorityCandidate[5]
                      << " triang_0_10_priority_candidate_other=" << triang0To10PriorityCandidate[6]
                      << " triang_0_10_priority_candidate_unknown=" << triang0To10PriorityCandidate[0]
                      << " triang_0_10_priority_match_reference=" << triang0To10PriorityMatch[1]
                      << " triang_0_10_priority_match_core_confirmed=" << triang0To10PriorityMatch[2]
                      << " triang_0_10_priority_match_stable_history=" << triang0To10PriorityMatch[3]
                      << " triang_0_10_priority_match_weak_vote=" << triang0To10PriorityMatch[4]
                      << " triang_0_10_priority_match_single_obs=" << triang0To10PriorityMatch[5]
                      << " triang_0_10_priority_match_other=" << triang0To10PriorityMatch[6]
                      << " triang_0_10_priority_match_unknown=" << triang0To10PriorityMatch[0];
        }
        if(bHasProjectionEdgeDiag)
        {
            std::cout << " local_proj_triang_0_10_proj_outside_reject=" << localProjTriang0To10ProjOutsideReject
                      << " local_proj_triang_0_10_companion_outside_reject=" << localProjTriang0To10CompanionOutsideReject;
        }
        if(bHasRefAssocMatcherDiag)
        {
            std::cout << " refassoc_matcher_input=" << refAssocMatcherInput
                      << " refassoc_matcher_skip_not_in_view=" << refAssocMatcherSkipNotInView
                      << " refassoc_matcher_skip_far=" << refAssocMatcherSkipFar
                      << " refassoc_matcher_skip_bad=" << refAssocMatcherSkipBad
                      << " refassoc_matcher_skip_no_desc=" << refAssocMatcherSkipNoDesc
                      << " refassoc_matcher_projected=" << refAssocMatcherProjected
                      << " refassoc_matcher_side_attempt=" << refAssocMatcherSideAttempt
                      << " refassoc_matcher_invalid_level=" << refAssocMatcherInvalidLevel
                      << " refassoc_matcher_empty_area=" << refAssocMatcherEmptyArea
                      << " refassoc_matcher_area_candidates=" << refAssocMatcherAreaCandidates
                      << " refassoc_matcher_invalid_candidate=" << refAssocMatcherInvalidCandidate
                      << " refassoc_matcher_occupied_candidate=" << refAssocMatcherOccupiedCandidate
                      << " refassoc_matcher_geom_ur_reject=" << refAssocMatcherGeomUrReject
                      << " refassoc_matcher_geom_depth_reject=" << refAssocMatcherGeomDepthReject
                      << " refassoc_matcher_valid_candidate=" << refAssocMatcherValidCandidate
                      << " refassoc_matcher_no_valid_candidate=" << refAssocMatcherNoValidCandidate
                      << " refassoc_matcher_raw_candidate=" << refAssocMatcherRawCandidate
                      << " refassoc_matcher_dist_reject=" << refAssocMatcherDistReject
                      << " refassoc_matcher_ratio_reject=" << refAssocMatcherRatioReject
                      << " refassoc_matcher_candidate_pool=" << refAssocMatcherCandidatePool
                      << " refassoc_matcher_spatial_reject=" << refAssocMatcherSpatialReject
                      << " refassoc_matcher_assign_reject=" << refAssocMatcherAssignReject
                      << " refassoc_matcher_outside_reject=" << refAssocMatcherOutsideReject
                      << " refassoc_matcher_assigned=" << refAssocMatcherAssigned;
        }
        if(bHasLightGlueLocalDiag)
        {
            std::cout << " lg_local_enabled=" << (bUseLightGlueLocalMap ? 1 : 0)
                      << " lg_local_failed=" << (bLightGlueLocalFailed ? 1 : 0)
                      << " lg_local_kfs=" << lgLocalKFsUsed
                      << " lg_local_raw_matches=" << lgLocalRawMatches
                      << " lg_local_mp_valid=" << lgLocalMpValid
                      << " lg_local_one_to_one=" << lgLocalOneToOne
                      << " lg_local_filled=" << lgLocalFilled
                      << " lg_local_occupied=" << lgLocalOccupied
                      << " lg_local_duplicate_mp=" << lgLocalDuplicateMP
                      << " lg_local_not_projected=" << lgLocalNotProjected
                      << " lg_local_proj_reject=" << lgLocalProjReject
                      << " lg_local_verify_candidates=" << lgLocalVerifyCandidates
                      << " lg_local_verified=" << lgLocalVerified
                      << " lg_local_rejected=" << lgLocalRejected
                      << " lg_local_mismatch=" << lgLocalMismatch
                      << " lg_local_conflict_low_score=" << lgLocalConflictLowScore
                      << " lg_local_conflict_min_score=" << (bUseLightGlueLocalMap ? GetXFeatLightGlueLocalConflictMinScore() : 0.0f)
                      << " lg_local_proj_radius=" << (bUseLightGlueLocalMap ? GetXFeatLightGlueLocalMapProjRadius() : 0.0f)
                      << " lg_local_map_updated_edge_candidates=" << lgLocalMapUpdatedEdgeCandidates
                      << " lg_local_map_updated_edge_verified=" << lgLocalMapUpdatedEdgeVerified
                      << " lg_local_map_updated_edge_conflict_unverified=" << lgLocalMapUpdatedEdgeConflictUnverified
                      << " lg_local_map_updated_edge_no_evidence=" << lgLocalMapUpdatedEdgeNoEvidence
                      << " lg_local_map_updated_edge_non_target_ignored=" << lgLocalMapUpdatedEdgeNonTargetIgnored;
        }
        if(bProjectionRun)
        {
            std::cout << " far_filter=" << (bFarFilterValue ? "on" : "off")
                      << " far_th=" << thFarPointsValue
                      << " near_only_applied=" << (bNearOnlyAppliedValue ? "true" : "false");
        }
        std::cout << " matches=" << matchesValue
                  << " matches_total_added=" << (matchesValue + lgLocalFilled)
                  << std::endl;

        if(!IsXFeatDeepDiagEnabled())
            return;

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6)
                  << "[SearchLocalPoints][MPQuality]"
                  << " frame=" << mCurrentFrame.mnId
                  << " mThDepth=" << mThDepth;
        AppendMapPointQualityStats(std::cout, "already", mpQualityAlreadyMatched);
        AppendMapPointQualityStats(std::cout, "seen", mpQualityLocalAlreadySeen);
        AppendMapPointQualityStats(std::cout, "rejected", mpQualityFoundRatioRejected);
        AppendMapPointQualityStats(std::cout, "frustum", mpQualityLocalInFrustum);
        AppendMapPointQualityStats(std::cout, "trackview", mpQualityLocalTrackInView);
        std::cout << std::endl;
        for(int sourceBin = 0; sourceBin < kCloseReuseSourceBinCount; ++sourceBin)
        {
            for(int depthBin = 0; depthBin < kCloseReuseDepthBinCount; ++depthBin)
            {
                const CloseReuseFunnelStats& stats = closeReuseFunnel[sourceBin][depthBin];
                if(stats.selected <= 0 && stats.admitted <= 0 && stats.projectionMatch <= 0)
                    continue;
                std::cout << "[SearchLocalPoints][CloseReuseFunnel]"
                          << " frame_id=" << mCurrentFrame.mnId
                          << " source=" << CloseReuseSourceName(sourceBin)
                          << " depth_bin=" << CloseReuseDepthName(depthBin)
                          << " selected=" << stats.selected
                          << " already_seen=" << stats.alreadySeen
                          << " bad=" << stats.bad
                          << " not_in_frustum="
                          << std::max(0, stats.selected - stats.alreadySeen - stats.bad - stats.inFrustum)
                          << " in_frustum=" << stats.inFrustum
                          << " found_gate_reject=" << stats.foundGateReject
                          << " history_gate_reject=" << stats.historyGateReject
                          << " admission_deferred="
                          << std::max(0, stats.inFrustum - stats.foundGateReject -
                                           stats.historyGateReject - stats.admitted)
                          << " admitted=" << stats.admitted
                          << " projection_match=" << stats.projectionMatch
                          << " projection_miss=" << std::max(0, stats.admitted - stats.projectionMatch)
                          << " admitted_match_rate="
                          << (stats.admitted > 0
                              ? static_cast<float>(stats.projectionMatch) /
                                static_cast<float>(stats.admitted)
                              : 0.0f)
                          << std::endl;
                for(int statusBin = 0; statusBin < kCloseReuseFrustumStatusCount; ++statusBin)
                {
                    if(stats.frustumStatus[statusBin] <= 0)
                        continue;
                    std::cout << "[SearchLocalPoints][CloseReuseFrustumReason]"
                              << " frame_id=" << mCurrentFrame.mnId
                              << " source=" << CloseReuseSourceName(sourceBin)
                              << " depth_bin=" << CloseReuseDepthName(depthBin)
                              << " reason=" << CloseReuseFrustumStatusName(statusBin)
                              << " total=" << stats.frustumStatus[statusBin]
                              << " selected=" << stats.selected
                              << std::endl;
                }
            }
        }
        for(int sourceBin = 0; sourceBin < kCloseReuseSourceBinCount; ++sourceBin)
        {
            for(int depthBin = 0; depthBin < kCloseReuseDepthBinCount; ++depthBin)
            {
                const ReferenceAssociationFunnelStats& stats =
                    referenceAssociationFunnel[sourceBin][depthBin];
                if(stats.referenceEligible <= 0 && stats.predictedVisibleTarget <= 0)
                    continue;
                std::cout << "[SearchLocalPoints][ReferenceAssociationFunnel]"
                          << " frame_id=" << mCurrentFrame.mnId
                          << " source=" << CloseReuseSourceName(sourceBin)
                          << " depth_bin=" << CloseReuseDepthName(depthBin)
                          << " reference_eligible=" << stats.referenceEligible
                          << " stable_reference=" << stats.stableReference
                          << " predicted_visible_target=" << stats.predictedVisibleTarget
                          << " pre_associated=" << stats.preAssociated
                          << " motion_recovered=" << stats.motionRecovered
                          << " selected_local=" << stats.selected
                          << " target_not_selected_local="
                          << std::max(0, stats.predictedVisibleTarget - stats.selected)
                          << " already_seen=" << stats.alreadySeen
                          << " bad=" << stats.bad
                          << " in_frustum=" << stats.inFrustum
                          << " found_gate_reject=" << stats.foundGateReject
                          << " history_gate_reject=" << stats.historyGateReject
                          << " admitted=" << stats.admitted
                          << " projection_match=" << stats.projectionMatch
                          << " projection_miss="
                          << std::max(0, stats.admitted - stats.projectionMatch)
                          << " admitted_match_rate="
                          << (stats.admitted > 0
                              ? static_cast<float>(stats.projectionMatch) /
                                static_cast<float>(stats.admitted)
                              : 0.0f)
                          << std::endl;
                for(int statusBin = 0; statusBin < kCloseReuseFrustumStatusCount; ++statusBin)
                {
                    if(stats.frustumStatus[statusBin] <= 0)
                        continue;
                    std::cout << "[SearchLocalPoints][ReferenceAssociationFrustumReason]"
                              << " frame_id=" << mCurrentFrame.mnId
                              << " source=" << CloseReuseSourceName(sourceBin)
                              << " depth_bin=" << CloseReuseDepthName(depthBin)
                              << " reason=" << CloseReuseFrustumStatusName(statusBin)
                              << " total=" << stats.frustumStatus[statusBin]
                              << " predicted_visible_target=" << stats.predictedVisibleTarget
                              << std::endl;
                }
            }
        }
        auto printMappingFeatureReuseFunnel =
            [&](const char* dimension,
                const char* binName,
                const MappingFeatureReuseFunnelStats& stats)
        {
            if(stats.selected <= 0 && stats.admitted <= 0 && stats.projectionMatch <= 0)
                return;
            const int notInFrustum =
                std::max(0, stats.selected - stats.alreadySeen - stats.bad - stats.inFrustum);
            const int projectionMiss = std::max(0, stats.admitted - stats.projectionMatch);
            std::cout << "[SearchLocalPoints][MappingFeatureReuseFunnel]"
                      << " frame_id=" << mCurrentFrame.mnId
                      << " source=local_mapping_triangulation"
                      << " dimension=" << dimension
                      << " bin=" << binName
                      << " selected=" << stats.selected
                      << " already_seen=" << stats.alreadySeen
                      << " bad=" << stats.bad
                      << " not_in_frustum=" << notInFrustum
                      << " in_frustum=" << stats.inFrustum
                      << " found_gate_reject=" << stats.foundGateReject
                      << " history_gate_reject=" << stats.historyGateReject
                      << " admitted=" << stats.admitted
                      << " projection_match=" << stats.projectionMatch
                      << " projection_miss=" << projectionMiss
                      << " frustum_rate="
                      << (stats.selected > 0
                          ? static_cast<float>(stats.inFrustum) / static_cast<float>(stats.selected)
                          : 0.0f)
                      << " admitted_match_rate="
                      << (stats.admitted > 0
                          ? static_cast<float>(stats.projectionMatch) /
                            static_cast<float>(stats.admitted)
                          : 0.0f)
                      << std::endl;
        };
        for(int bin = 0; bin < kMappingFeatureReuseGridBins; ++bin)
        {
            printMappingFeatureReuseFunnel("grid_bin",
                                           MappingFeatureGridBinName(bin),
                                           mappingFeatureGridFunnel[bin]);
        }
        for(int bin = 0; bin < kMappingFeatureReuseOctaveBins; ++bin)
        {
            printMappingFeatureReuseFunnel("octave_bin",
                                           MappingFeatureOctaveBinName(bin),
                                           mappingFeatureOctaveFunnel[bin]);
        }
        for(int bin = 0; bin < kMappingFeatureReuseResponseBins; ++bin)
        {
            printMappingFeatureReuseFunnel("response_bin",
                                           MappingFeatureResponseBinName(bin),
                                           mappingFeatureResponseFunnel[bin]);
        }
        for(int bin = 0; bin < kMappingFeatureReuseDepthBins; ++bin)
        {
            printMappingFeatureReuseFunnel("creation_depth_bin",
                                           DepthBinName(bin),
                                           mappingFeatureDepthFunnel[bin]);
        }
        for(int bin = 0; bin < kMappingFeatureReuseCombinedBins; ++bin)
        {
            printMappingFeatureReuseFunnel("mapping_feature_bin",
                                           MappingFeatureCombinedBinName(bin),
                                           mappingFeatureCombinedFunnel[bin]);
        }
        for(int sourceBin = 0; sourceBin < kCloseReuseSourceBinCount; ++sourceBin)
        {
            for(int contributorBin = 0;
                contributorBin < kCloseReuseContributorBinCount;
                ++contributorBin)
            {
                const CloseReuseContributionStats& stats =
                    closeReuseContribution[sourceBin][contributorBin];
                if(stats.selected <= 0 && stats.projectionMatch <= 0)
                    continue;
                std::cout << "[SearchLocalPoints][CloseReuseContribution]"
                          << " frame_id=" << mCurrentFrame.mnId
                          << " source=" << CloseReuseSourceName(sourceBin)
                          << " contributor=" << CloseReuseContributorName(contributorBin)
                          << " selected=" << stats.selected
                          << " in_frustum=" << stats.inFrustum
                          << " admitted=" << stats.admitted
                          << " projection_match=" << stats.projectionMatch
                          << " frustum_rate="
                          << (stats.selected > 0
                              ? static_cast<float>(stats.inFrustum) / static_cast<float>(stats.selected)
                              : 0.0f)
                          << " match_rate="
                          << (stats.admitted > 0
                              ? static_cast<float>(stats.projectionMatch) / static_cast<float>(stats.admitted)
                              : 0.0f)
                          << std::endl;
            }
        }
        int kfsWithSelected = 0;
        int kfsWithFrustum = 0;
        int kfsWithMatch = 0;
        int kfsWithNoFrustum = 0;
        int kfsWithNoMatch = 0;
        int selectedByKF = 0;
        int frustumByKF = 0;
        int matchedByKF = 0;
        int topMatches = 0;
        KeyFrame* pTopMatchKF = nullptr;
        for(const auto& entry : closeReuseKFContribution)
        {
            const CloseReuseContributionStats& stats = entry.second;
            if(stats.selected <= 0)
                continue;
            ++kfsWithSelected;
            selectedByKF += stats.selected;
            frustumByKF += stats.inFrustum;
            matchedByKF += stats.projectionMatch;
            if(stats.inFrustum > 0)
                ++kfsWithFrustum;
            else
                ++kfsWithNoFrustum;
            if(stats.projectionMatch > 0)
                ++kfsWithMatch;
            else
                ++kfsWithNoMatch;
            if(stats.projectionMatch > topMatches)
            {
                topMatches = stats.projectionMatch;
                pTopMatchKF = entry.first;
            }
        }
        if(kfsWithSelected > 0)
        {
            LocalKFProvenance topProvenance;
            const auto topIt = mmLocalKFProvenance.find(pTopMatchKF);
            if(topIt != mmLocalKFProvenance.end() &&
               topIt->second.frameId == mCurrentFrame.mnId)
            {
                topProvenance = topIt->second;
            }
            std::cout << "[SearchLocalPoints][LocalKFCloseContribution]"
                      << " frame_id=" << mCurrentFrame.mnId
                      << " selected_kfs=" << kfsWithSelected
                      << " frustum_kfs=" << kfsWithFrustum
                      << " match_kfs=" << kfsWithMatch
                      << " zero_frustum_kfs=" << kfsWithNoFrustum
                      << " no_match_kfs=" << kfsWithNoMatch
                      << " selected=" << selectedByKF
                      << " in_frustum=" << frustumByKF
                      << " projection_match=" << matchedByKF
                      << " top_match_kf_id=" << (pTopMatchKF ? static_cast<long int>(pTopMatchKF->mnId) : -1)
                      << " top_match_count=" << topMatches
                      << " top_match_role="
                      << LocalMapProvenanceRoleName(LocalMapProvenanceRoleIndex(topProvenance.role))
                      << " top_match_vote=" << topProvenance.voteCount
                      << std::endl;
        }
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    };

    LogSearchLocalPointsDiagnostics(projectionMatches,
                                    nToMatch,
                                    projectionTh,
                                    bProjectionRun,
                                    projectionFarFilter,
                                    projectionFarPoints,
                                    projectionNearOnlyApplied);
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();
    mmLocalMPProvenance.clear();

    auto updateLocalMPProvenance = [&](KeyFrame* pKF, MapPoint* pMP)
    {
        if(!pKF || !pMP || pMP->isBad())
            return;

        LocalMPProvenance& provenance = mmLocalMPProvenance[pMP];
        const bool bFirstObservation = provenance.frameId != mCurrentFrame.mnId;
        if(bFirstObservation)
        {
            provenance = LocalMPProvenance();
            provenance.frameId = mCurrentFrame.mnId;
        }
        provenance.localObservations++;

        LocalKFProvenanceRole role = LocalKFProvenanceRole::Unknown;
        int voteCount = 0;
        const auto itKFProv = mmLocalKFProvenance.find(pKF);
        if(itKFProv != mmLocalKFProvenance.end() &&
           itKFProv->second.frameId == mCurrentFrame.mnId)
        {
            role = itKFProv->second.role;
            voteCount = itKFProv->second.voteCount;
        }

        long int age = -1;
        if(mCurrentFrame.mnId >= pKF->mnFrameId)
            age = static_cast<long int>(mCurrentFrame.mnId - pKF->mnFrameId);

        const int roleRank = LocalMapProvenanceRoleRank(role);
        const int bestRoleRank = LocalMapProvenanceRoleRank(provenance.bestRole);
        const bool bBetterVote = voteCount > provenance.bestVoteCount;
        const bool bEqualVoteBetterRole =
            voteCount == provenance.bestVoteCount && roleRank > bestRoleRank;
        const bool bEqualVoteRoleNewer =
            voteCount == provenance.bestVoteCount &&
            roleRank == bestRoleRank &&
            age >= 0 &&
            (provenance.bestKeyFrameAge < 0 || age < provenance.bestKeyFrameAge);

        if(bFirstObservation || bBetterVote || bEqualVoteBetterRole || bEqualVoteRoleNewer)
        {
            provenance.bestRole = role;
            provenance.bestVoteCount = voteCount;
            provenance.bestKeyFrameAge = age;
            provenance.pBestKeyFrame = pKF;
        }
    };

    std::vector<MapPoint*> vpCandidateMapPoints;
    std::unordered_set<MapPoint*> spCandidateMapPoints;
    vpCandidateMapPoints.reserve(8192);
    spCandidateMapPoints.reserve(8192);

    for(vector<KeyFrame*>::const_reverse_iterator itKF=mvpLocalKeyFrames.rbegin(), itEndKF=mvpLocalKeyFrames.rend(); itKF!=itEndKF; ++itKF)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {

            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(!pMP->isBad())
                updateLocalMPProvenance(pKF, pMP);
            if(!pMP->isBad())
            {
                if(spCandidateMapPoints.insert(pMP).second)
                    vpCandidateMapPoints.push_back(pMP);
            }
        }
    }

    for(MapPoint* pMP : vpCandidateMapPoints)
    {
        if(!pMP || pMP->isBad())
            continue;
        if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
            continue;

        mvpLocalMapPoints.push_back(pMP);
        pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
    }

}


void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    const bool bUseORB = (std::getenv("USE_ORB") != nullptr);
    mmLocalKFProvenance.clear();
    mnLocalMapVotedKFCandidates = 0;
    mnLocalMapVotedKFKept = 0;
    mnLocalMapVotedKFVisibilityKept = 0;
    auto setLocalKFProvenance = [&](KeyFrame* pKF,
                                    const LocalKFProvenanceRole role,
                                    const int voteCount,
                                    const bool bForce)
    {
        if(!pKF)
            return;
        LocalKFProvenance& provenance = mmLocalKFProvenance[pKF];
        const bool bMissing = provenance.frameId != mCurrentFrame.mnId;
        const bool bBetterRole =
            LocalMapProvenanceRoleRank(role) > LocalMapProvenanceRoleRank(provenance.role);
        const bool bBetterVote = voteCount > provenance.voteCount;
        if(bForce || bMissing || bBetterVote || bBetterRole)
        {
            provenance.role = role;
            provenance.voteCount = voteCount;
            provenance.frameId = mCurrentFrame.mnId;
        }
    };

    auto accumulateKeyFrameVotes = [&](MapPoint* pMP)
    {
        if(!pMP || pMP->isBad())
            return;

        const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();
        for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; ++it)
        {
            keyframeCounter[it->first]++;
        }
    };

    if(!mpAtlas->isImuInitialized() || (mCurrentFrame.mnId<mnLastRelocFrameId+2))
    {
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    accumulateKeyFrameVotes(pMP);
                }
                else
                {
                    mCurrentFrame.mvpMapPoints[i]=NULL;
                }
            }
        }
    }
    else
    {
        for(int i=0; i<mLastFrame.N; i++)
        {
            // Using lastframe since current frame has not matches yet
            if(mLastFrame.mvpMapPoints[i])
            {
                MapPoint* pMP = mLastFrame.mvpMapPoints[i];
                if(!pMP)
                    continue;
                if(!pMP->isBad())
                {
                    accumulateKeyFrameVotes(pMP);
                }
                else
                {
                    // MODIFICATION
                    mLastFrame.mvpMapPoints[i]=NULL;
                }
            }
        }
    }

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    std::vector<LocalMapVotedKFCandidate> orderedVotedKFs;
    orderedVotedKFs.reserve(keyframeCounter.size());
    auto admitVotedKeyFrame = [&](const LocalMapVotedKFCandidate& candidate)
    {
        KeyFrame* pKF = candidate.pKF;
        if(!pKF || pKF->isBad())
            return;
        if(pKF->mnTrackReferenceForFrame == mCurrentFrame.mnId)
            return;

        mvpLocalKeyFrames.push_back(pKF);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        ++mnLocalMapVotedKFKept;
        mnLocalMapVotedKFVisibilityKept += candidate.visibilityScore;
        setLocalKFProvenance(pKF,
                             LocalKFProvenanceRole::Voted,
                             candidate.voteCount,
                             true);
    };

    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;
        if(!pKF || pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        LocalMapVotedKFCandidate candidate;
        candidate.pKF = pKF;
        candidate.voteCount = it->second;
        candidate.visibilityScore = bUseORB ? 0 : EstimateLocalKeyFrameCurrentVisibility(mCurrentFrame, pKF);
        orderedVotedKFs.push_back(candidate);
        admitVotedKeyFrame(candidate);
    }

    std::sort(orderedVotedKFs.begin(), orderedVotedKFs.end(),
              [](const LocalMapVotedKFCandidate& a,
                 const LocalMapVotedKFCandidate& b)
              {
                  if(a.visibilityScore != b.visibilityScore)
                      return a.visibilityScore > b.visibilityScore;
                  if(a.voteCount != b.voteCount)
                      return a.voteCount > b.voteCount;
                  if(!a.pKF)
                      return false;
                  if(!b.pKF)
                      return true;
                  return a.pKF->mnId > b.pKF->mnId;
              });

    mnLocalMapVotedKFCandidates = static_cast<int>(orderedVotedKFs.size());

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80) // 80
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);


        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    setLocalKFProvenance(pNeighKF, LocalKFProvenanceRole::Covis, 0, false);
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    setLocalKFProvenance(pChildKF, LocalKFProvenanceRole::ChildParent, 0, false);
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                setLocalKFProvenance(pParent, LocalKFProvenanceRole::ChildParent, 0, false);
                break;
            }
        }
    }

    // Add 10 last temporal KFs (mainly for IMU)
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) &&mvpLocalKeyFrames.size()<80)
    {
        KeyFrame* tempKeyFrame = mCurrentFrame.mpLastKeyFrame;

        const int Nd = 20;
        for(int i=0; i<Nd; i++){
            if (!tempKeyFrame)
                break;
            if(tempKeyFrame->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(tempKeyFrame);
                tempKeyFrame->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                setLocalKFProvenance(tempKeyFrame, LocalKFProvenanceRole::Temporal, 0, false);
                tempKeyFrame=tempKeyFrame->mPrevKF;
            }
        }
    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
        setLocalKFProvenance(pKFmax, LocalKFProvenanceRole::Reference, max, true);
    }


}

bool Tracking::Relocalization()
{
    mbXFeatMotionHealthValid = false;
    mbXFeatMotionHealthWeak = false;
    mbXFeatMotionHealthBad = false;

    Verbose::PrintMess("Starting relocalization", Verbose::VERBOSITY_NORMAL);
    // [XFEAT_RELOC_20260414] ORB keeps BoW pipeline; XFeat avoids BoW dependency in relocalization.
    const bool bUseORB = (std::getenv("USE_ORB") != nullptr);
    if(bUseORB)
        mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs;
    if(bUseORB)
    {
        vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame, mpAtlas->GetCurrentMap());
    }
    else
    {
        // [XFEAT_RELOC_20260414] XFeat path: use map keyframes as relocalization candidates.
        vpCandidateKFs = mpAtlas->GetCurrentMap()->GetAllKeyFrames();
    }

    if(vpCandidateKFs.empty()) {
        Verbose::PrintMess("There are not candidates", Verbose::VERBOSITY_NORMAL);
        return false;
    }

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcherORB(0.75, true);
    XFeatMatcher matcherXF(0.75f, false);

    vector<MLPnPsolver*> vpMLPnPsolvers;
    vpMLPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;
    //调试: 重定位阶段统计，用于快速判断是候选不足还是PnP/优化失败。
    int relocBestInliers = 0;
    int relocBestKFId = -1;
    int relocPoseSolvedCount = 0;
    const bool bUseLightGlueReloc = !bUseORB && IsXFeatLightGlueRelocEnabled();
    const int lightGlueRelocFallbackMinMatches = GetXFeatLightGlueRelocFallbackMinMatches();
    int lgRelocKFsTried = 0;
    int lgRelocKFsUsed = 0;
    int lgRelocFallbacks = 0;
    int lgRelocFailures = 0;
    int lgRelocRawMatches = 0;
    int lgRelocMpValid = 0;
    int lgRelocOneToOne = 0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches;
            // [XFEAT_RELOC_20260414] XFeat path uses descriptor NN for initial relocalization matching.
            if(bUseORB)
                nmatches = matcherORB.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            else
            {
                bool bUsedLightGlueForCandidate = false;
                if(bUseLightGlueReloc)
                {
                    ++lgRelocKFsTried;
                    try
                    {
                        static XFeatLighterGlueMatcher lightGlueMatcher;
                        const float lgConf = GetEnvFloatInRange("XFEAT_LIGHTGLUE_CONF", 0.1f, 0.0f, 1.0f);
                        nmatches = lightGlueMatcher.SearchByLightGlue(pKF,
                                                                      mCurrentFrame,
                                                                      vvpMapPointMatches[i],
                                                                      lgConf);
                        const XFeatLighterGlueMatcher::Stats& lgStats = lightGlueMatcher.LastStats();
                        lgRelocRawMatches += lgStats.lg_raw_matches;
                        lgRelocMpValid += lgStats.mp_valid;
                        lgRelocOneToOne += lgStats.one_to_one;
                        bUsedLightGlueForCandidate = nmatches >= lightGlueRelocFallbackMinMatches;
                        if(bUsedLightGlueForCandidate)
                            ++lgRelocKFsUsed;
                        else
                            ++lgRelocFallbacks;
                    }
                    catch(const std::exception& e)
                    {
                        ++lgRelocFailures;
                        bUsedLightGlueForCandidate = false;
                        vvpMapPointMatches[i].clear();
                        nmatches = 0;
                        if(ShouldPrintXFeatDebug(mCurrentFrame.mnId))
                        {
                            std::cerr << "[Reloc][LightGlue] failed for KF "
                                      << (pKF ? static_cast<long long>(pKF->mnId) : -1)
                                      << ", falling back to XFeatMatcher: "
                                      << e.what() << std::endl;
                        }
                    }
                }

                if(!bUsedLightGlueForCandidate)
                {
                    nmatches = matcherXF.SearchByNN(pKF,
                                                    mCurrentFrame,
                                                    vvpMapPointMatches[i],
                                                    GetXFeatThHighRelocNN());
                }
            }

            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                MLPnPsolver* pSolver = new MLPnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,6,0.5,5.991);  //This solver needs at least 6 points
                vpMLPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);
    XFeatMatcher matcher2XF(0.9f,false);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            MLPnPsolver* pSolver = vpMLPnPsolvers[i];
            Eigen::Matrix4f eigTcw;
            bool bTcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers, eigTcw);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(bTcw)
            {
                Sophus::SE3f Tcw(eigTcw);
                mCurrentFrame.SetPose(Tcw);
                // Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);
                ++relocPoseSolvedCount;
                if(nGood > relocBestInliers)
                {
                    relocBestInliers = nGood;
                    relocBestKFId = vpCandidateKFs[i] ? vpCandidateKFs[i]->mnId : -1;
                }

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional = 0;
                    if(bUseORB)
                    {
                        nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);
                    }
                    else
                    {
                        nadditional = matcher2XF.SearchByProjection(mCurrentFrame,
                                                                     vpCandidateKFs[i],
                                                                     sFound,
                                                                     10.0f,
                                                                     GetXFeatThHighRelocProjCoarse());
                    }

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            if(bUseORB)
                            {
                                nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);
                            }
                            else
                            {
                                nadditional = matcher2XF.SearchByProjection(mCurrentFrame,
                                                                             vpCandidateKFs[i],
                                                                             sFound,
                                                                             3.0f,
                                                                             GetXFeatThHighRelocProjFine());
                            }

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        if(ShouldPrintXFeatDebug(mCurrentFrame.mnId))
        {
            std::cout << "[Reloc] FAIL frame=" << mCurrentFrame.mnId
                      << " best_inliers=" << relocBestInliers
                      << " best_kf=" << relocBestKFId
                      << " solved_pose_cnt=" << relocPoseSolvedCount
                      << " lg_enabled=" << (bUseLightGlueReloc ? 1 : 0)
                      << " lg_kfs_tried=" << lgRelocKFsTried
                      << " lg_kfs_used=" << lgRelocKFsUsed
                      << " lg_fallbacks=" << lgRelocFallbacks
                      << " lg_failures=" << lgRelocFailures
                      << " lg_raw_matches=" << lgRelocRawMatches
                      << " lg_mp_valid=" << lgRelocMpValid
                      << " lg_one_to_one=" << lgRelocOneToOne
                      << std::endl;
        }
        return false;
    }
    else
    {
        if(ShouldPrintXFeatDebug(mCurrentFrame.mnId))
        {
            std::cout << "[Reloc] SUCCESS frame=" << mCurrentFrame.mnId
                      << " best_inliers=" << relocBestInliers
                      << " best_kf=" << relocBestKFId
                      << " solved_pose_cnt=" << relocPoseSolvedCount
                      << " lg_enabled=" << (bUseLightGlueReloc ? 1 : 0)
                      << " lg_kfs_tried=" << lgRelocKFsTried
                      << " lg_kfs_used=" << lgRelocKFsUsed
                      << " lg_fallbacks=" << lgRelocFallbacks
                      << " lg_failures=" << lgRelocFailures
                      << " lg_raw_matches=" << lgRelocRawMatches
                      << " lg_mp_valid=" << lgRelocMpValid
                      << " lg_one_to_one=" << lgRelocOneToOne
                      << std::endl;
        }
        mnLastRelocFrameId = mCurrentFrame.mnId;
        cout << "Relocalized!!" << endl;
        return true;
    }

}

void Tracking::Reset(bool bLocMap)
{
    Verbose::PrintMess("System Reseting", Verbose::VERBOSITY_NORMAL);

    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
        mpLocalMapper->RequestReset();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
    }


    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestReset();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database...", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clear();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearAtlas();
    mpAtlas->CreateNewMap();
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
        mpAtlas->SetInertialSensor();
    mnInitialFrameId = 0;

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    mbReadyToInitializate = false;
    mbSetInit=false;

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
    mCurrentFrame = Frame();
    mnLastRelocFrameId = 0;
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

void Tracking::ResetActiveMap(bool bLocMap)
{
    Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    Map* pMap = mpAtlas->GetCurrentMap();

    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_VERY_VERBOSE);
        mpLocalMapper->RequestResetActiveMap(pMap);
        Verbose::PrintMess("done", Verbose::VERBOSITY_VERY_VERBOSE);
    }

    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestResetActiveMap(pMap);
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearMap();


    //KeyFrame::nNextId = mpAtlas->GetLastInitKFid();
    //Frame::nNextId = mnLastInitFrameId;
    mnLastInitFrameId = Frame::nNextId;
    //mnLastRelocFrameId = mnLastInitFrameId;
    mState = NO_IMAGES_YET; //NOT_INITIALIZED;

    mbReadyToInitializate = false;

    list<bool> lbLost;
    // lbLost.reserve(mlbLost.size());
    unsigned int index = mnFirstFrameId;
    cout << "mnFirstFrameId = " << mnFirstFrameId << endl;
    for(Map* pMap : mpAtlas->GetAllMaps())
    {
        if(pMap->GetAllKeyFrames().size() > 0)
        {
            if(index > pMap->GetLowerKFID())
                index = pMap->GetLowerKFID();
        }
    }

    //cout << "First Frame id: " << index << endl;
    int num_lost = 0;
    cout << "mnInitialFrameId = " << mnInitialFrameId << endl;

    for(list<bool>::iterator ilbL = mlbLost.begin(); ilbL != mlbLost.end(); ilbL++)
    {
        if(index < mnInitialFrameId)
            lbLost.push_back(*ilbL);
        else
        {
            lbLost.push_back(true);
            num_lost += 1;
        }

        index++;
    }
    cout << num_lost << " Frames set to lost" << endl;

    mlbLost = lbLost;

    mnInitialFrameId = mCurrentFrame.mnId;
    mnLastRelocFrameId = mCurrentFrame.mnId;

    mCurrentFrame = Frame();
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();

    mbVelocity = false;

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

vector<MapPoint*> Tracking::GetLocalMapMPS()
{
    return mvpLocalMapPoints;
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    mK_.setIdentity();
    mK_(0,0) = fx;
    mK_(1,1) = fy;
    mK_(0,2) = cx;
    mK_(1,2) = cy;

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

void Tracking::UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame)
{
    Map * pMap = pCurrentKeyFrame->GetMap();
    unsigned int index = mnFirstFrameId;
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mlpReferences.begin();
    list<bool>::iterator lbL = mlbLost.begin();
    for(auto lit=mlRelativeFramePoses.begin(),lend=mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        while(pKF->isBad())
        {
            pKF = pKF->GetParent();
        }

        if(pKF->GetMap() == pMap)
        {
            (*lit).translation() *= s;
        }
    }

    mLastBias = b;

    mpLastKeyFrame = pCurrentKeyFrame;

    mLastFrame.SetNewBias(mLastBias);
    mCurrentFrame.SetNewBias(mLastBias);

    while(!mCurrentFrame.imuIsPreintegrated())
    {
        usleep(500);
    }


    if(mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId)
    {
        mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(),
                                      mLastFrame.mpLastKeyFrame->GetImuPosition(),
                                      mLastFrame.mpLastKeyFrame->GetVelocity());
    }
    else
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const Eigen::Vector3f twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mLastFrame.mpImuPreintegrated->dT;

        mLastFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    if (mCurrentFrame.mpImuPreintegrated)
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);

        const Eigen::Vector3f twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mCurrentFrame.mpImuPreintegrated->dT;

        mCurrentFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    mnFirstImuFrameId = mCurrentFrame.mnId;
}

void Tracking::NewDataset()
{
    mnNumDataset++;
}

int Tracking::GetNumberDataset()
{
    return mnNumDataset;
}

int Tracking::GetMatchesInliers()
{
    return mnMatchesInliers;
}

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, string strFolder)
{
    mpSystem->SaveTrajectoryEuRoC(strFolder + strNameFile_frames);
    //mpSystem->SaveKeyFrameTrajectoryEuRoC(strFolder + strNameFile_kf);
}

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, Map* pMap)
{
    mpSystem->SaveTrajectoryEuRoC(strNameFile_frames, pMap);
    if(!strNameFile_kf.empty())
        mpSystem->SaveKeyFrameTrajectoryEuRoC(strNameFile_kf, pMap);
}

float Tracking::GetImageScale()
{
    return mImageScale;
}

#ifdef REGISTER_LOOP
void Tracking::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
}

bool Tracking::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Tracking STOP" << endl;
        return true;
    }

    return false;
}

bool Tracking::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

bool Tracking::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

void Tracking::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
    mbStopRequested = false;
}
#endif

} //namespace ORB_SLAM
