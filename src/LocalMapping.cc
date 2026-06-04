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


#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "XFeatMatcher.h"
#include "XFeatLighterGlueMatcher.h"
#include "Optimizer.h"
#include "Converter.h"
#include "GeometricTools.h"

#include<mutex>
#include<chrono>
#include<cctype>
#include<cstdlib>
#include<cmath>
#include<algorithm>
#include<array>
#include<exception>
#include<iomanip>
#include<iostream>
#include<limits>
#include<map>
#include<set>
#include<string>
#include<tuple>
#include<unordered_set>
#include<vector>

namespace ORB_SLAM3
{

namespace
{
    constexpr bool kEnableMappingProbationBehavior = false;
    constexpr bool kEnableXFeatCloseMidPendingTracklet = false;
    constexpr bool kEnableMappingFeatureBudgetBehavior = false;
    constexpr bool kEnableXFeatMappingProfileBehavior = false;
    constexpr bool kEnableXFeatCloseMidPairSelectionBehavior = true;
    constexpr int kXFeatMappingProfileCloseCreatedFloor = 90;
    constexpr int kXFeatMappingProfileFallbackCreateCap = 45;
    constexpr int kPendingMappingTrackletBootstrapKFs = 5;
    constexpr int kPendingMappingTrackletMinObservations = 3;
    constexpr int kPendingMappingTrackletMinUpdatePasses = 2;
    constexpr int kPendingMappingTrackletMaxAgeKFs = 6;
    constexpr int kPendingMappingTrackletMaxConflicts = 2;

    bool IsLocalMappingEnvFlagEnabled(const char* key)
    {
        const char* env = std::getenv(key);
        if(!env)
            return false;
        const std::string v(env);
        return !(v.empty() || v == "0" || v == "false" || v == "FALSE");
    }

    int GetLocalMappingDiagInterval()
    {
        int value = 1;
        const char* env = std::getenv("XFEAT_DIAG_INTERVAL");
        if(env && env[0] != '\0')
        {
            try
            {
                value = std::max(1, std::min(1000, std::stoi(std::string(env))));
            }
            catch(...)
            {
                value = 1;
            }
        }
        return value;
    }

    bool ShouldPrintLocalMappingXFeatDebug(const long unsigned int keyframeId)
    {
        if(!IsLocalMappingEnvFlagEnabled("XFEAT_DEBUG"))
            return false;
        return (keyframeId % static_cast<long unsigned int>(GetLocalMappingDiagInterval())) == 0;
    }

    bool IsLocalMappingFeatureDiagEnabled()
    {
        static const bool enabled = IsLocalMappingEnvFlagEnabled("XFEAT_DIAG_FEATURE_DISTRIBUTION");
        return enabled;
    }

    bool IsLocalMappingDeepDiagEnabled()
    {
        static const bool enabled = IsLocalMappingEnvFlagEnabled("XFEAT_DIAG_DEEP");
        return enabled;
    }

    bool IsXFeatKeyFrameDescriptor(const KeyFrame* pKF)
    {
        return pKF && !pKF->mDescriptors.empty() && pKF->mDescriptors.type() == CV_32F;
    }

    bool IsXFeatMappingProfileEligible(const KeyFrame* pKF, const size_t idx)
    {
        if(!IsXFeatKeyFrameDescriptor(pKF))
            return true;
        if(idx >= pKF->mvXFeatMappingEligible.size())
            return false;
        return pKF->mvXFeatMappingEligible[idx] != 0;
    }

    struct ScalarDistributionStats
    {
        int total = 0;
        int valid = 0;
        int invalid = 0;
        int gt50 = 0;
        int gt100 = 0;
        double sum = 0.0;
        std::vector<float> samples;
    };

    void AccumulateScalarDistribution(ScalarDistributionStats& stats,
                                      const float value,
                                      const bool depthLike)
    {
        stats.total++;
        if(!std::isfinite(value) || (depthLike && value <= 0.0f) || (!depthLike && value < 0.0f))
        {
            stats.invalid++;
            return;
        }

        stats.valid++;
        stats.sum += value;
        stats.samples.push_back(value);
        if(depthLike)
        {
            if(value > 50.0f)
                stats.gt50++;
            if(value > 100.0f)
                stats.gt100++;
        }
    }

    float ScalarPercentile(std::vector<float>& samples, const float percentile)
    {
        if(samples.empty())
            return 0.0f;
        std::sort(samples.begin(), samples.end());
        const float clamped = std::max(0.0f, std::min(1.0f, percentile));
        const size_t idx = std::min(samples.size() - 1,
                                    static_cast<size_t>(std::round(clamped * static_cast<float>(samples.size() - 1))));
        return samples[idx];
    }

    void AppendScalarDistributionStats(std::ostream& os,
                                       const char* prefix,
                                       const ScalarDistributionStats& stats,
                                       const bool depthLike)
    {
        std::vector<float> samples = stats.samples;
        const double avg = stats.valid > 0 ? stats.sum / static_cast<double>(stats.valid) : 0.0;
        os << " " << prefix << "_total=" << stats.total
           << " " << prefix << "_valid=" << stats.valid
           << " " << prefix << "_invalid=" << stats.invalid
           << " " << prefix << "_avg=" << avg
           << " " << prefix << "_min=" << ScalarPercentile(samples, 0.0f)
           << " " << prefix << "_p50=" << ScalarPercentile(samples, 0.50f)
           << " " << prefix << "_p90=" << ScalarPercentile(samples, 0.90f)
           << " " << prefix << "_p99=" << ScalarPercentile(samples, 0.99f)
           << " " << prefix << "_max=" << ScalarPercentile(samples, 1.0f);
        if(depthLike)
        {
            os << " " << prefix << "_gt50=" << stats.gt50
               << " " << prefix << "_gt100=" << stats.gt100;
        }
    }

    float GetMapPointReferenceDepth(MapPoint* pMP)
    {
        if(!pMP)
            return -1.0f;

        KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
        if(!pRefKF)
            return -1.0f;

        const std::tuple<int,int> idxs = pMP->GetIndexInKeyFrame(pRefKF);
        const int leftIdx = std::get<0>(idxs);
        if(leftIdx >= 0 && leftIdx < static_cast<int>(pRefKF->mvDepth.size()))
            return pRefKF->mvDepth[leftIdx];
        return -1.0f;
    }

    struct RecentMapPointCullingDiag
    {
        int total = 0;
        int kept = 0;
        int already_bad = 0;
        int found_ratio_culled = 0;
        int low_obs_culled = 0;
        int aged_out_survived = 0;
        int mapping_probation_active = 0;
        int mapping_probation_kept = 0;
        int mapping_probation_promoted = 0;
        int mapping_probation_culled_no_inlier = 0;
        int mapping_probation_culled_expired = 0;
        int mapping_probation_culled_outlier = 0;
        ScalarDistributionStats allDepth;
        ScalarDistributionStats keptDepth;
        ScalarDistributionStats foundRatioCulledDepth;
        ScalarDistributionStats lowObsCulledDepth;
        ScalarDistributionStats agedOutDepth;
        ScalarDistributionStats allFoundRatio;
        ScalarDistributionStats keptFoundRatio;
        ScalarDistributionStats foundRatioCulledFoundRatio;
        ScalarDistributionStats lowObsCulledFoundRatio;
        ScalarDistributionStats agedOutFoundRatio;
    };

    enum class MappingProbationCullReason
    {
        None,
        NoInlier,
        Expired,
        OutlierHistory
    };

    constexpr size_t kMapPointCreationSourceCount = 5;

    size_t MapPointCreationSourceIndex(const MapPoint::CreationSource source)
    {
        const unsigned int idx = static_cast<unsigned int>(source);
        if(idx >= kMapPointCreationSourceCount)
            return 0;
        return static_cast<size_t>(idx);
    }

    int LocalMappingDepthBinIndex(const float depth)
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

    const char* LocalMappingDepthBinName(const int bin)
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

    constexpr int kMappingFeatureBudgetGridCols = 8;
    constexpr int kMappingFeatureBudgetGridRows = 6;
    constexpr int kMappingFeatureBudgetGridCells =
        kMappingFeatureBudgetGridCols * kMappingFeatureBudgetGridRows;

    int MappingFeatureBudgetCellIndex(const int gridX, const int gridY)
    {
        if(gridX < 0 || gridY < 0)
            return -1;
        const int cellX = std::min(kMappingFeatureBudgetGridCols - 1,
                                   std::max(0, gridX * kMappingFeatureBudgetGridCols / FRAME_GRID_COLS));
        const int cellY = std::min(kMappingFeatureBudgetGridRows - 1,
                                   std::max(0, gridY * kMappingFeatureBudgetGridRows / FRAME_GRID_ROWS));
        return cellY * kMappingFeatureBudgetGridCols + cellX;
    }

    int MappingFeatureBudgetCellCap(const int gridBin)
    {
        if(gridBin == 1)
            return 5;
        return 8;
    }

    struct MappingFeatureCreateStats
    {
        int total = 0;
        int close = 0;
        ScalarDistributionStats depth;
    };

    void AccumulateMappingFeatureCreateStats(MappingFeatureCreateStats& stats,
                                             const float depth,
                                             const bool close)
    {
        stats.total++;
        if(close)
            stats.close++;
        AccumulateScalarDistribution(stats.depth, depth, true);
    }

    enum class TriangInputTerminalEvent
    {
        StereoFallbackReject,
        ParallaxReject,
        CheiralityReject,
        ReprojectionReject,
        ScaleReject,
        Created
    };

    struct TriangInputFailureStats
    {
        int total = 0;
        int stereoFallbackReject = 0;
        int parallaxReject = 0;
        int cheiralityReject = 0;
        int reprojReject = 0;
        int reprojKf1Only = 0;
        int reprojKf2Only = 0;
        int reprojBoth = 0;
        int scaleReject = 0;
        int created = 0;
        int chi2Kf1Samples = 0;
        int chi2Kf2Samples = 0;
        int chi2MaxSamples = 0;
        int cosSamples = 0;
        int scaleResidualSamples = 0;
        double chi2Kf1Sum = 0.0;
        double chi2Kf2Sum = 0.0;
        double chi2MaxSum = 0.0;
        double cosParallaxSum = 0.0;
        double scaleResidualSum = 0.0;
    };

    struct TriangInputPairContext
    {
        int neighborRank = -1;
        int covisWeight = -1;
        float baseline = std::numeric_limits<float>::quiet_NaN();
        float baselineDepthRatio = std::numeric_limits<float>::quiet_NaN();
    };

    void AccumulateTriangInputFailureStats(TriangInputFailureStats& stats,
                                           const TriangInputTerminalEvent event,
                                           const float chi2Kf1,
                                           const float chi2Kf2,
                                           const bool reprojKf1Reject,
                                           const bool reprojKf2Reject,
                                           const float cosParallax,
                                           const float scaleResidual)
    {
        stats.total++;
        switch(event)
        {
            case TriangInputTerminalEvent::StereoFallbackReject:
                stats.stereoFallbackReject++;
                break;
            case TriangInputTerminalEvent::ParallaxReject:
                stats.parallaxReject++;
                break;
            case TriangInputTerminalEvent::CheiralityReject:
                stats.cheiralityReject++;
                break;
            case TriangInputTerminalEvent::ReprojectionReject:
                stats.reprojReject++;
                if(reprojKf1Reject && reprojKf2Reject)
                    stats.reprojBoth++;
                else if(reprojKf1Reject)
                    stats.reprojKf1Only++;
                else if(reprojKf2Reject)
                    stats.reprojKf2Only++;
                break;
            case TriangInputTerminalEvent::ScaleReject:
                stats.scaleReject++;
                break;
            case TriangInputTerminalEvent::Created:
                stats.created++;
                break;
        }

        if(std::isfinite(chi2Kf1))
        {
            stats.chi2Kf1Samples++;
            stats.chi2Kf1Sum += chi2Kf1;
        }
        if(std::isfinite(chi2Kf2))
        {
            stats.chi2Kf2Samples++;
            stats.chi2Kf2Sum += chi2Kf2;
        }
        if(std::isfinite(chi2Kf1) || std::isfinite(chi2Kf2))
        {
            float chi2Max = 0.0f;
            if(std::isfinite(chi2Kf1))
                chi2Max = std::max(chi2Max, chi2Kf1);
            if(std::isfinite(chi2Kf2))
                chi2Max = std::max(chi2Max, chi2Kf2);
            stats.chi2MaxSamples++;
            stats.chi2MaxSum += chi2Max;
        }
        if(std::isfinite(cosParallax))
        {
            stats.cosSamples++;
            stats.cosParallaxSum += cosParallax;
        }
        if(std::isfinite(scaleResidual))
        {
            stats.scaleResidualSamples++;
            stats.scaleResidualSum += scaleResidual;
        }
    }

    void PrintTriangInputFailureStats(const long unsigned int keyFrameId,
                                      const char* dimension,
                                      const char* binName,
                                      const TriangInputFailureStats& stats)
    {
        if(stats.total <= 0)
            return;

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        const auto avgOrMissing = [](const double sum, const int samples) -> double
        {
            return samples > 0 ? sum / static_cast<double>(samples) : -1.0;
        };

        std::cout << std::fixed << std::setprecision(6)
                  << "[LocalMapping][TriangInputFailure]"
                  << " kf_id=" << keyFrameId
                  << " dimension=" << dimension
                  << " bin=" << binName
                  << " total=" << stats.total
                  << " stereo_fallback_reject=" << stats.stereoFallbackReject
                  << " parallax_reject=" << stats.parallaxReject
                  << " cheirality_reject=" << stats.cheiralityReject
                  << " reproj_reject=" << stats.reprojReject
                  << " reproj_kf1_only=" << stats.reprojKf1Only
                  << " reproj_kf2_only=" << stats.reprojKf2Only
                  << " reproj_both=" << stats.reprojBoth
                  << " scale_reject=" << stats.scaleReject
                  << " created=" << stats.created
                  << " chi2_kf1_avg=" << avgOrMissing(stats.chi2Kf1Sum, stats.chi2Kf1Samples)
                  << " chi2_kf2_avg=" << avgOrMissing(stats.chi2Kf2Sum, stats.chi2Kf2Samples)
                  << " chi2_max_avg=" << avgOrMissing(stats.chi2MaxSum, stats.chi2MaxSamples)
                  << " cos_parallax_avg=" << avgOrMissing(stats.cosParallaxSum, stats.cosSamples)
                  << " scale_residual_avg=" << avgOrMissing(stats.scaleResidualSum, stats.scaleResidualSamples)
                  << std::endl;
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    int TriangInputBackendBin(const MapPoint::TriangulationMatchBackend backend)
    {
        switch(backend)
        {
            case MapPoint::kTriangBackendXFeatBF:
                return 1;
            case MapPoint::kTriangBackendLightGlue:
                return 2;
            case MapPoint::kTriangBackendORB:
            case MapPoint::kTriangBackendUnknown:
            default:
                return 0;
        }
    }

    const char* TriangInputBackendBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "xfeat_bf";
            case 2: return "lightglue";
            case 3: return "tracklet";
            default: return "unknown";
        }
    }

    int TriangInputCloseDepthBin(const float depth)
    {
        if(!std::isfinite(depth) || depth <= 0.0f || depth > 10.0f)
            return -1;
        return depth <= 5.0f ? 0 : 1;
    }

    const char* TriangInputCloseDepthBinName(const int bin)
    {
        return bin == 0 ? "0_5" : "5_10";
    }

    int TriangInputOctavePairBin(const int octave1, const int octave2)
    {
        if(octave1 < 0 || octave2 < 0)
            return 0;
        const int diff = std::abs(octave1 - octave2);
        if(diff == 0)
            return 1;
        if(diff == 1)
            return 2;
        return 3;
    }

    const char* TriangInputOctavePairBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "same";
            case 2: return "off1";
            case 3: return "off2p";
            default: return "missing";
        }
    }

    const char* TriangInputCurrentOctaveBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "0";
            case 2: return "1";
            case 3: return "2p";
            default: return "missing";
        }
    }

    int TriangInputEdgeDistanceBin(const KeyFrame* pKF, const cv::KeyPoint& kp)
    {
        if(!pKF)
            return 0;
        const float left = kp.pt.x - static_cast<float>(pKF->mnMinX);
        const float right = static_cast<float>(pKF->mnMaxX) - kp.pt.x;
        const float top = kp.pt.y - static_cast<float>(pKF->mnMinY);
        const float bottom = static_cast<float>(pKF->mnMaxY) - kp.pt.y;
        const float minDist = std::min(std::min(left, right), std::min(top, bottom));
        if(!std::isfinite(minDist))
            return 0;
        if(minDist <= 20.0f)
            return 1;
        if(minDist <= 50.0f)
            return 2;
        return 3;
    }

    const char* TriangInputEdgeDistanceBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "near_edge";
            case 2: return "mid";
            case 3: return "interior";
            default: return "missing";
        }
    }

    int TriangInputNeighborRankBin(const int rank)
    {
        if(rank < 0)
            return 0;
        if(rank == 0)
            return 1;
        if(rank < 3)
            return 2;
        if(rank < 10)
            return 3;
        return 4;
    }

    const char* TriangInputNeighborRankBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "top1";
            case 2: return "top3";
            case 3: return "top10";
            case 4: return "late";
            default: return "missing";
        }
    }

    int TriangInputKfIdDeltaBin(const KeyFrame* pKF1, const KeyFrame* pKF2)
    {
        if(!pKF1 || !pKF2)
            return 0;
        const long long id1 = static_cast<long long>(pKF1->mnId);
        const long long id2 = static_cast<long long>(pKF2->mnId);
        const long long delta = std::llabs(id1 - id2);
        if(delta <= 2)
            return 1;
        if(delta <= 5)
            return 2;
        if(delta <= 15)
            return 3;
        return 4;
    }

    const char* TriangInputKfIdDeltaBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "0_2";
            case 2: return "3_5";
            case 3: return "6_15";
            case 4: return "16p";
            default: return "missing";
        }
    }

    int TriangInputCovisWeightBin(const int weight)
    {
        if(weight < 0)
            return 0;
        if(weight < 15)
            return 1;
        if(weight < 50)
            return 2;
        return 3;
    }

    const char* TriangInputCovisWeightBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "low";
            case 2: return "mid";
            case 3: return "high";
            default: return "missing";
        }
    }

    int TriangInputBaselineBin(const float baseline)
    {
        if(!std::isfinite(baseline) || baseline < 0.0f)
            return 0;
        if(baseline < 0.05f)
            return 1;
        if(baseline < 0.15f)
            return 2;
        return 3;
    }

    const char* TriangInputBaselineBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "small";
            case 2: return "mid";
            case 3: return "large";
            default: return "missing";
        }
    }

    int TriangInputBaselineDepthRatioBin(const float ratio)
    {
        if(!std::isfinite(ratio) || ratio < 0.0f)
            return 0;
        if(ratio < 0.01f)
            return 1;
        if(ratio < 0.03f)
            return 2;
        if(ratio < 0.08f)
            return 3;
        return 4;
    }

    const char* TriangInputBaselineDepthRatioBinName(const int bin)
    {
        switch(bin)
        {
            case 1: return "weak";
            case 2: return "low";
            case 3: return "mid";
            case 4: return "strong";
            default: return "missing";
        }
    }

    bool KeyFrameFeatureGridPosition(const KeyFrame* pKF,
                                     const cv::KeyPoint& kp,
                                     int& gridX,
                                     int& gridY)
    {
        gridX = -1;
        gridY = -1;
        if(!pKF || pKF->mnMaxX <= pKF->mnMinX || pKF->mnMaxY <= pKF->mnMinY)
            return false;
        gridX = static_cast<int>(std::round((kp.pt.x - pKF->mnMinX) * pKF->mfGridElementWidthInv));
        gridY = static_cast<int>(std::round((kp.pt.y - pKF->mnMinY) * pKF->mfGridElementHeightInv));
        if(gridX < 0 || gridX >= pKF->mnGridCols || gridY < 0 || gridY >= pKF->mnGridRows)
        {
            gridX = -1;
            gridY = -1;
            return false;
        }
        return true;
    }

    void PrintMappingFeatureCreateStats(const long unsigned int keyFrameId,
                                        const long unsigned int frameId,
                                        const char* dimension,
                                        const char* binName,
                                        const MappingFeatureCreateStats& stats)
    {
        if(stats.total <= 0)
            return;
        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6)
                  << "[LocalMapping][MappingFeatureCreate]"
                  << " kf_id=" << keyFrameId
                  << " frame_id=" << frameId
                  << " source=local_mapping_triangulation"
                  << " dimension=" << dimension
                  << " bin=" << binName
                  << " created=" << stats.total
                  << " close_created=" << stats.close;
        AppendScalarDistributionStats(std::cout, "depth", stats.depth, true);
        std::cout << std::endl;
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }

    void AccumulateRecentMapPointCullingDiag(MapPoint* pMP,
                                             ScalarDistributionStats& depthStats,
                                             ScalarDistributionStats& foundRatioStats)
    {
        if(!pMP)
            return;
        AccumulateScalarDistribution(depthStats, GetMapPointReferenceDepth(pMP), true);
        AccumulateScalarDistribution(foundRatioStats, pMP->GetFoundRatio(), false);
    }

    void AccumulateRecentMapPointCullingDiagTotal(RecentMapPointCullingDiag& diag,
                                                  MapPoint* pMP)
    {
        diag.total++;
        AccumulateRecentMapPointCullingDiag(pMP, diag.allDepth, diag.allFoundRatio);
    }

    void AccumulateRecentMapPointCullingDiagKept(RecentMapPointCullingDiag& diag,
                                                 MapPoint* pMP)
    {
        diag.kept++;
        AccumulateRecentMapPointCullingDiag(pMP, diag.keptDepth, diag.keptFoundRatio);
    }

    void AccumulateRecentMapPointCullingDiagAlreadyBad(RecentMapPointCullingDiag& diag)
    {
        diag.already_bad++;
    }

    void AccumulateRecentMapPointCullingDiagFoundRatioCulled(RecentMapPointCullingDiag& diag,
                                                             MapPoint* pMP)
    {
        diag.found_ratio_culled++;
        AccumulateRecentMapPointCullingDiag(pMP,
                                            diag.foundRatioCulledDepth,
                                            diag.foundRatioCulledFoundRatio);
    }

    void AccumulateRecentMapPointCullingDiagLowObsCulled(RecentMapPointCullingDiag& diag,
                                                         MapPoint* pMP)
    {
        diag.low_obs_culled++;
        AccumulateRecentMapPointCullingDiag(pMP,
                                            diag.lowObsCulledDepth,
                                            diag.lowObsCulledFoundRatio);
    }

    void AccumulateRecentMapPointCullingDiagAgedOut(RecentMapPointCullingDiag& diag,
                                                    MapPoint* pMP)
    {
        diag.aged_out_survived++;
        AccumulateRecentMapPointCullingDiag(pMP, diag.agedOutDepth, diag.agedOutFoundRatio);
    }

    int CountMaskBitsLocalMapping(unsigned int mask)
    {
        int count = 0;
        while(mask != 0U)
        {
            count += static_cast<int>(mask & 1U);
            mask >>= 1U;
        }
        return count;
    }

    bool IsActiveCloseMidMappingProbation(MapPoint* pMP)
    {
        if(!pMP || !pMP->IsMappingProbation())
            return false;
        if(pMP->GetCreationSource() != MapPoint::kCreationSourceLocalMappingTriangulation)
            return false;
        const float depth = pMP->mMappingProbationDepthAtCreation;
        return std::isfinite(depth) && depth > 0.0f && depth <= 10.0f;
    }

    int MappingProbationAgeKf(MapPoint* pMP, const unsigned long int currentKfId)
    {
        if(!pMP)
            return 0;
        return currentKfId >= pMP->mnMappingProbationStartKF
            ? static_cast<int>(currentKfId - pMP->mnMappingProbationStartKF)
            : 0;
    }

    MappingProbationCullReason ClassifyMappingProbationCull(MapPoint* pMP,
                                                            const unsigned long int currentKfId)
    {
        if(!pMP)
            return MappingProbationCullReason::None;

        const int ageKf = MappingProbationAgeKf(pMP, currentKfId);
        const MapPoint::LocalProjectionHistoryStats history = pMP->GetLocalProjectionHistory();
        const int optOutcomes = history.optInlierCount + history.optOutlierCount;
        const float optOutlierRate = optOutcomes > 0
            ? static_cast<float>(history.optOutlierCount) / static_cast<float>(optOutcomes)
            : 0.0f;
        const int recentValid = CountMaskBitsLocalMapping(history.recentValidMask);
        const int recentOutlier = CountMaskBitsLocalMapping(history.recentOutlierMask);
        const float recentOutlierRate = recentValid > 0
            ? static_cast<float>(recentOutlier) / static_cast<float>(recentValid)
            : 0.0f;

        if((optOutcomes >= 3 && optOutlierRate > 0.35f) ||
           (recentValid >= 3 && recentOutlierRate > 0.40f))
        {
            return MappingProbationCullReason::OutlierHistory;
        }
        if(ageKf >= 4 && history.optInlierCount < 1)
            return MappingProbationCullReason::NoInlier;
        if(ageKf >= 6)
            return MappingProbationCullReason::Expired;
        return MappingProbationCullReason::None;
    }

    void AccumulateMappingProbationActive(RecentMapPointCullingDiag& diag)
    {
        diag.mapping_probation_active++;
    }

    void AccumulateMappingProbationKept(RecentMapPointCullingDiag& diag)
    {
        diag.mapping_probation_kept++;
    }

    void AccumulateMappingProbationPromoted(RecentMapPointCullingDiag& diag)
    {
        diag.mapping_probation_promoted++;
    }

    void AccumulateMappingProbationCulled(RecentMapPointCullingDiag& diag,
                                          const MappingProbationCullReason reason)
    {
        switch(reason)
        {
            case MappingProbationCullReason::NoInlier:
                diag.mapping_probation_culled_no_inlier++;
                break;
            case MappingProbationCullReason::Expired:
                diag.mapping_probation_culled_expired++;
                break;
            case MappingProbationCullReason::OutlierHistory:
                diag.mapping_probation_culled_outlier++;
                break;
            case MappingProbationCullReason::None:
            default:
                break;
        }
    }

    void AppendRecentMapPointCullingDiag(std::ostream& os,
                                         const RecentMapPointCullingDiag& diag)
    {
        os << " recent_total=" << diag.total
           << " kept=" << diag.kept
           << " already_bad=" << diag.already_bad
           << " found_ratio_culled=" << diag.found_ratio_culled
           << " low_obs_culled=" << diag.low_obs_culled
           << " aged_out_survived=" << diag.aged_out_survived
           << " mapping_probation_active=" << diag.mapping_probation_active
           << " mapping_probation_kept=" << diag.mapping_probation_kept
           << " mapping_probation_promoted=" << diag.mapping_probation_promoted
           << " mapping_probation_culled_no_inlier=" << diag.mapping_probation_culled_no_inlier
           << " mapping_probation_culled_expired=" << diag.mapping_probation_culled_expired
           << " mapping_probation_culled_outlier=" << diag.mapping_probation_culled_outlier;
        AppendScalarDistributionStats(os, "all_depth", diag.allDepth, true);
        AppendScalarDistributionStats(os, "kept_depth", diag.keptDepth, true);
        AppendScalarDistributionStats(os, "found_cull_depth", diag.foundRatioCulledDepth, true);
        AppendScalarDistributionStats(os, "lowobs_cull_depth", diag.lowObsCulledDepth, true);
        AppendScalarDistributionStats(os, "aged_depth", diag.agedOutDepth, true);
        AppendScalarDistributionStats(os, "all_found", diag.allFoundRatio, false);
        AppendScalarDistributionStats(os, "kept_found", diag.keptFoundRatio, false);
        AppendScalarDistributionStats(os, "found_cull_found", diag.foundRatioCulledFoundRatio, false);
        AppendScalarDistributionStats(os, "lowobs_cull_found", diag.lowObsCulledFoundRatio, false);
        AppendScalarDistributionStats(os, "aged_found", diag.agedOutFoundRatio, false);
    }

    bool UseXFeatLightGlueTriangulation()
    {
        return IsLocalMappingEnvFlagEnabled("XFEAT_USE_LIGHTGLUE_TRIANG");
    }

    float GetXFeatLightGlueTriangulationConfidence()
    {
        return 0.1f;
    }

    int GetXFeatLightGlueTriangulationFallbackMinMatches()
    {
        return 8;
    }

    float GetLocalMappingEnvFloat(const char* key,
                                  const float defaultValue,
                                  const float minValue,
                                  const float maxValue)
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

    int GetLocalMappingEnvInt(const char* key,
                              const int defaultValue,
                              const int minValue,
                              const int maxValue)
    {
        const char* env = std::getenv(key);
        if(!env)
            return defaultValue;

        try
        {
            const int value = std::stoi(std::string(env));
            return std::max(minValue, std::min(maxValue, value));
        }
        catch(...)
        {
            return defaultValue;
        }
    }

    std::string GetLocalMappingEnvLowerString(const char* key, const std::string& defaultValue)
    {
        const char* env = std::getenv(key);
        if(!env)
            return defaultValue;

        std::string value(env);
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    bool IsLongWeakMapPointCullingEnabled()
    {
        static const bool enabled = IsLocalMappingEnvFlagEnabled("XFEAT_LOCALMAP_LONG_WEAK_MP_CULL");
        return enabled;
    }

    const std::string& GetLongWeakMapPointCullingScope()
    {
        static const std::string scope = []()
        {
            const std::string envScope =
                GetLocalMappingEnvLowerString("XFEAT_LOCALMAP_LONG_WEAK_MP_SCOPE", "triang_only");
            if(envScope == "all")
                return std::string("all");
            return std::string("triang_only");
        }();
        return scope;
    }

    float GetLongWeakMapPointCullingMaxFoundRatio()
    {
        static const float value =
            GetLocalMappingEnvFloat("XFEAT_LOCALMAP_LONG_WEAK_MP_MAX_FOUND_RATIO", 0.25f, 0.0f, 1.0f);
        return value;
    }

    float GetLongWeakMapPointCullingMinDepth()
    {
        static const float value =
            GetLocalMappingEnvFloat("XFEAT_LOCALMAP_LONG_WEAK_MP_MIN_DEPTH", -1.0f, -1.0f, 10000.0f);
        return value;
    }

    int GetLongWeakMapPointCullingMinAgeKf()
    {
        static const int value =
            GetLocalMappingEnvInt("XFEAT_LOCALMAP_LONG_WEAK_MP_MIN_AGE_KF", 10, 1, 1000000);
        return value;
    }

    int GetLongWeakMapPointCullingInterval()
    {
        static const int value =
            GetLocalMappingEnvInt("XFEAT_LOCALMAP_LONG_WEAK_MP_INTERVAL", 5, 1, 1000000);
        return value;
    }

    bool ShouldConsiderLongWeakMapPoint(const MapPoint* pMP, const std::string& scope)
    {
        if(!pMP)
            return false;
        if(scope == "all")
            return true;
        return pMP->GetCreationSource() == MapPoint::kCreationSourceLocalMappingTriangulation;
    }

    float ResolveMapPointDepthInKeyFrame(KeyFrame* pKF, MapPoint* pMP)
    {
        if(!pKF || !pMP)
            return -1.0f;

        const Eigen::Vector3f x3Dc = pKF->GetPose() * pMP->GetWorldPos();
        const float z = x3Dc(2);
        if(std::isfinite(z) && z > 0.0f)
            return z;
        return -1.0f;
    }

    struct LongWeakMapPointQualityStats
    {
        int total = 0;
        ScalarDistributionStats obs;
        ScalarDistributionStats foundRatio;
        ScalarDistributionStats depth;
        ScalarDistributionStats age;
    };

    void AccumulateLongWeakMapPointQualityStats(LongWeakMapPointQualityStats& stats,
                                                MapPoint* pMP,
                                                const long unsigned int currentKfId,
                                                const float resolvedDepth)
    {
        if(!pMP)
            return;

        stats.total++;
        const int ageKf = std::max(0,
                                   static_cast<int>(currentKfId) -
                                   std::max(0, static_cast<int>(pMP->mnFirstKFid)));
        AccumulateScalarDistribution(stats.obs, static_cast<float>(pMP->Observations()), false);
        AccumulateScalarDistribution(stats.foundRatio, pMP->GetFoundRatio(), false);
        AccumulateScalarDistribution(stats.depth, resolvedDepth, true);
        AccumulateScalarDistribution(stats.age, static_cast<float>(ageKf), false);
    }

    void AppendLongWeakMapPointQualityStats(std::ostream& os,
                                            const char* prefix,
                                            const LongWeakMapPointQualityStats& stats)
    {
        os << " " << prefix << "_total=" << stats.total;
        AppendScalarDistributionStats(os, (std::string(prefix) + "_obs").c_str(), stats.obs, false);
        AppendScalarDistributionStats(os, (std::string(prefix) + "_found").c_str(), stats.foundRatio, false);
        AppendScalarDistributionStats(os, (std::string(prefix) + "_depth").c_str(), stats.depth, true);
        AppendScalarDistributionStats(os, (std::string(prefix) + "_age").c_str(), stats.age, false);
    }
}

LocalMapping::LocalMapping(System* pSys, Atlas *pAtlas, const float bMonocular, bool bInertial, const string &_strSeqName):
    mpSystem(pSys), mbMonocular(bMonocular), mbInertial(bInertial), mbResetRequested(false), mbResetRequestedActiveMap(false), mbFinishRequested(false), mbFinished(true), mpAtlas(pAtlas), bInitializing(false),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true),
    mIdxInit(0), mScale(1.0), mInitSect(0), mbNotBA1(true), mbNotBA2(true), mIdxIteration(0), infoInertial(Eigen::MatrixXd::Zero(9,9))
{
    mnMatchesInliers = 0;

    mbBadImu = false;

    mTinit = 0.f;

    mNumLM = 0;
    mNumKFCulling=0;

#ifdef REGISTER_TIMES
    nLBA_exec = 0;
    nLBA_abort = 0;
#endif

}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::Run()
{
    mbFinished = false;

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames() && !mbBadImu)
        {
#ifdef REGISTER_TIMES
            double timeLBA_ms = 0;
            double timeKFCulling_ms = 0;

            std::chrono::steady_clock::time_point time_StartProcessKF = std::chrono::steady_clock::now();
#endif
            // BoW conversion and insertion in Map
            ProcessNewKeyFrame();
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndProcessKF = std::chrono::steady_clock::now();

            double timeProcessKF = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndProcessKF - time_StartProcessKF).count();
            vdKFInsert_ms.push_back(timeProcessKF);
#endif

            // Check recent MapPoints
            MapPointCulling();
#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndMPCulling = std::chrono::steady_clock::now();

            double timeMPCulling = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndMPCulling - time_EndProcessKF).count();
            vdMPCulling_ms.push_back(timeMPCulling);
#endif

            // Triangulate new MapPoints
            CreateNewMapPoints();

            mbAbortBA = false;

            if(!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors();
                CullLongLivedWeakMapPoints();
            }

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndMPCreation = std::chrono::steady_clock::now();

            double timeMPCreation = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndMPCreation - time_EndMPCulling).count();
            vdMPCreation_ms.push_back(timeMPCreation);
#endif

            bool b_doneLBA = false;
            int num_FixedKF_BA = 0;
            int num_OptKF_BA = 0;
            int num_MPs_BA = 0;
            int num_edges_BA = 0;

            if(!CheckNewKeyFrames() && !stopRequested())
            {
                if(mpAtlas->KeyFramesInMap()>2)
                {

                    if(mbInertial && mpCurrentKeyFrame->GetMap()->isImuInitialized())
                    {
                        float dist = (mpCurrentKeyFrame->mPrevKF->GetCameraCenter() - mpCurrentKeyFrame->GetCameraCenter()).norm() +
                                (mpCurrentKeyFrame->mPrevKF->mPrevKF->GetCameraCenter() - mpCurrentKeyFrame->mPrevKF->GetCameraCenter()).norm();

                        if(dist>0.05)
                            mTinit += mpCurrentKeyFrame->mTimeStamp - mpCurrentKeyFrame->mPrevKF->mTimeStamp;
                        if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2())
                        {
                            if((mTinit<10.f) && (dist<0.02))
                            {
                                cout << "Not enough motion for initializing. Reseting..." << endl;
                                unique_lock<mutex> lock(mMutexReset);
                                mbResetRequestedActiveMap = true;
                                mpMapToReset = mpCurrentKeyFrame->GetMap();
                                mbBadImu = true;
                            }
                        }

                        bool bLarge = ((mpTracker->GetMatchesInliers()>75)&&mbMonocular)||((mpTracker->GetMatchesInliers()>100)&&!mbMonocular);
                        Optimizer::LocalInertialBA(mpCurrentKeyFrame, &mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA, bLarge, !mpCurrentKeyFrame->GetMap()->GetIniertialBA2());
                        b_doneLBA = true;
                    }
                    else
                    {
                        Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA,num_OptKF_BA,num_MPs_BA,num_edges_BA);
                        b_doneLBA = true;
                    }

                }
#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_EndLBA = std::chrono::steady_clock::now();

                if(b_doneLBA)
                {
                    timeLBA_ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLBA - time_EndMPCreation).count();
                    vdLBA_ms.push_back(timeLBA_ms);

                    nLBA_exec += 1;
                    if(mbAbortBA)
                    {
                        nLBA_abort += 1;
                    }
                    vnLBA_edges.push_back(num_edges_BA);
                    vnLBA_KFopt.push_back(num_OptKF_BA);
                    vnLBA_KFfixed.push_back(num_FixedKF_BA);
                    vnLBA_MPs.push_back(num_MPs_BA);
                }

#endif

                // Initialize IMU here
                if(!mpCurrentKeyFrame->GetMap()->isImuInitialized() && mbInertial)
                {
                    if (mbMonocular)
                        InitializeIMU(1e2, 1e10, true);
                    else
                        InitializeIMU(1e2, 1e5, true);
                }


                // Check redundant local Keyframes
                KeyFrameCulling();

#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_EndKFCulling = std::chrono::steady_clock::now();

                timeKFCulling_ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndKFCulling - time_EndLBA).count();
                vdKFCulling_ms.push_back(timeKFCulling_ms);
#endif

                if ((mTinit<50.0f) && mbInertial)
                {
                    if(mpCurrentKeyFrame->GetMap()->isImuInitialized() && mpTracker->mState==Tracking::OK) // Enter here everytime local-mapping is called
                    {
                        if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA1()){
                            if (mTinit>5.0f)
                            {
                                cout << "start VIBA 1" << endl;
                                mpCurrentKeyFrame->GetMap()->SetIniertialBA1();
                                if (mbMonocular)
                                    InitializeIMU(1.f, 1e5, true);
                                else
                                    InitializeIMU(1.f, 1e5, true);

                                cout << "end VIBA 1" << endl;
                            }
                        }
                        else if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2()){
                            if (mTinit>15.0f){
                                cout << "start VIBA 2" << endl;
                                mpCurrentKeyFrame->GetMap()->SetIniertialBA2();
                                if (mbMonocular)
                                    InitializeIMU(0.f, 0.f, true);
                                else
                                    InitializeIMU(0.f, 0.f, true);

                                cout << "end VIBA 2" << endl;
                            }
                        }

                        // scale refinement
                        if (((mpAtlas->KeyFramesInMap())<=200) &&
                                ((mTinit>25.0f && mTinit<25.5f)||
                                (mTinit>35.0f && mTinit<35.5f)||
                                (mTinit>45.0f && mTinit<45.5f)||
                                (mTinit>55.0f && mTinit<55.5f)||
                                (mTinit>65.0f && mTinit<65.5f)||
                                (mTinit>75.0f && mTinit<75.5f))){
                            if (mbMonocular)
                                ScaleRefinement();
                        }
                    }
                }
            }

#ifdef REGISTER_TIMES
            vdLBASync_ms.push_back(timeKFCulling_ms);
            vdKFCullingSync_ms.push_back(timeKFCulling_ms);
#endif

            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndLocalMap = std::chrono::steady_clock::now();

            double timeLocalMap = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLocalMap - time_StartProcessKF).count();
            vdLMTotal_ms.push_back(timeLocalMap);
#endif
        }
        else if(Stop() && !mbBadImu)
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())
                break;
        }

        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }

    SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}


bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    mpAtlas->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::EmptyQueue()
{
    while(CheckNewKeyFrames())
        ProcessNewKeyFrame();
}

void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;
    const bool bPrintDiag = ShouldPrintLocalMappingXFeatDebug(nCurrentKFid);
    RecentMapPointCullingDiag diag;
    std::array<RecentMapPointCullingDiag, kMapPointCreationSourceCount> sourceDiags;

    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        RecentMapPointCullingDiag* pSourceDiag = nullptr;
        if(bPrintDiag && pMP)
        {
            pSourceDiag = &sourceDiags[MapPointCreationSourceIndex(pMP->GetCreationSource())];
        }

        if(bPrintDiag && pMP)
        {
            AccumulateRecentMapPointCullingDiagTotal(diag, pMP);
            AccumulateRecentMapPointCullingDiagTotal(*pSourceDiag, pMP);
        }

        if(pMP->isBad())
        {
            if(bPrintDiag)
            {
                AccumulateRecentMapPointCullingDiagAlreadyBad(diag);
                AccumulateRecentMapPointCullingDiagAlreadyBad(*pSourceDiag);
            }
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(IsActiveCloseMidMappingProbation(pMP))
        {
            if(bPrintDiag)
            {
                AccumulateMappingProbationActive(diag);
                AccumulateMappingProbationActive(*pSourceDiag);
            }

            if(pMP->TryPromoteMappingProbation(nCurrentKFid))
            {
                if(bPrintDiag)
                {
                    AccumulateMappingProbationPromoted(diag);
                    AccumulateMappingProbationPromoted(*pSourceDiag);
                    AccumulateRecentMapPointCullingDiagAgedOut(diag, pMP);
                    AccumulateRecentMapPointCullingDiagAgedOut(*pSourceDiag, pMP);
                }
                lit = mlpRecentAddedMapPoints.erase(lit);
            }
            else
            {
                const MappingProbationCullReason cullReason =
                    ClassifyMappingProbationCull(pMP, nCurrentKFid);
                if(cullReason != MappingProbationCullReason::None)
                {
                    if(bPrintDiag)
                    {
                        AccumulateMappingProbationCulled(diag, cullReason);
                        AccumulateMappingProbationCulled(*pSourceDiag, cullReason);
                    }
                    pMP->SetBadFlag();
                    lit = mlpRecentAddedMapPoints.erase(lit);
                }
                else
                {
                    if(bPrintDiag)
                    {
                        AccumulateMappingProbationKept(diag);
                        AccumulateMappingProbationKept(*pSourceDiag);
                        AccumulateRecentMapPointCullingDiagKept(diag, pMP);
                        AccumulateRecentMapPointCullingDiagKept(*pSourceDiag, pMP);
                    }
                    lit++;
                }
            }
        }
        else if(pMP->GetFoundRatio()<0.25f)
        {
            if(bPrintDiag)
            {
                AccumulateRecentMapPointCullingDiagFoundRatioCulled(diag, pMP);
                AccumulateRecentMapPointCullingDiagFoundRatioCulled(*pSourceDiag, pMP);
            }
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            if(bPrintDiag)
            {
                AccumulateRecentMapPointCullingDiagLowObsCulled(diag, pMP);
                AccumulateRecentMapPointCullingDiagLowObsCulled(*pSourceDiag, pMP);
            }
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
        {
            if(bPrintDiag)
            {
                AccumulateRecentMapPointCullingDiagAgedOut(diag, pMP);
                AccumulateRecentMapPointCullingDiagAgedOut(*pSourceDiag, pMP);
            }
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else
        {
            if(bPrintDiag)
            {
                AccumulateRecentMapPointCullingDiagKept(diag, pMP);
                AccumulateRecentMapPointCullingDiagKept(*pSourceDiag, pMP);
            }
            lit++;
        }
    }

    if(bPrintDiag)
    {
        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6)
                  << "[LocalMapping][RecentMP]"
                  << " kf_id=" << nCurrentKFid
                  << " th_obs=" << cnThObs;
        AppendRecentMapPointCullingDiag(std::cout, diag);
        std::cout << std::endl;

        if(IsLocalMappingFeatureDiagEnabled())
        {
            for(size_t i = 0; i < sourceDiags.size(); ++i)
            {
                if(sourceDiags[i].total <= 0)
                    continue;
                const MapPoint::CreationSource source = static_cast<MapPoint::CreationSource>(i);
                std::cout << std::fixed << std::setprecision(6)
                          << "[LocalMapping][RecentMPSource]"
                          << " kf_id=" << nCurrentKFid
                          << " th_obs=" << cnThObs
                          << " source=" << MapPointCreationSourceName(source);
                AppendRecentMapPointCullingDiag(std::cout, sourceDiags[i]);
                std::cout << std::endl;
            }
        }
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);
    }
}

void LocalMapping::CullLongLivedWeakMapPoints()
{
    if(!IsLongWeakMapPointCullingEnabled() || !mpCurrentKeyFrame || !mpTracker)
        return;
    if(CheckNewKeyFrames())
        return;

    const long unsigned int currentKfId = mpCurrentKeyFrame->mnId;
    const int interval = GetLongWeakMapPointCullingInterval();
    if(interval > 1 && (currentKfId % static_cast<long unsigned int>(interval)) != 0)
        return;

    const std::string& scope = GetLongWeakMapPointCullingScope();
    const float maxFoundRatio = GetLongWeakMapPointCullingMaxFoundRatio();
    const float configuredMinDepth = GetLongWeakMapPointCullingMinDepth();
    const float minDepth = configuredMinDepth > 0.0f ? configuredMinDepth : mpTracker->GetDepthThreshold();
    const int minAgeKf = GetLongWeakMapPointCullingMinAgeKf();
    const bool bPrintDiag = ShouldPrintLocalMappingXFeatDebug(currentKfId);

    std::vector<KeyFrame*> vpCandidateKFs;
    vpCandidateKFs.reserve(mbMonocular ? 31u : 11u);
    std::unordered_set<long unsigned int> seenKfIds;
    seenKfIds.reserve(mbMonocular ? 31u : 11u);
    auto addCandidateKF = [&](KeyFrame* pKF)
    {
        if(!pKF || pKF->isBad())
            return;
        if(!seenKfIds.insert(pKF->mnId).second)
            return;
        vpCandidateKFs.push_back(pKF);
    };

    addCandidateKF(mpCurrentKeyFrame);
    const int nn = mbMonocular ? 30 : 10;
    const std::vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    for(KeyFrame* pKF : vpNeighKFs)
        addCandidateKF(pKF);

    std::vector<MapPoint*> vpCandidates;
    vpCandidates.reserve(vpCandidateKFs.size() * 200);
    std::unordered_set<long unsigned int> seenMpIds;
    seenMpIds.reserve(vpCandidateKFs.size() * 200);
    for(KeyFrame* pKF : vpCandidateKFs)
    {
        const std::vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
        for(MapPoint* pMP : vpMapPoints)
        {
            if(!pMP || pMP->isBad())
                continue;
            if(!seenMpIds.insert(pMP->mnId).second)
                continue;
            vpCandidates.push_back(pMP);
        }
    }

    LongWeakMapPointQualityStats candidateStats;
    LongWeakMapPointQualityStats culledStats;
    int candidateBySource[kMapPointCreationSourceCount] = {};
    int culledBySource[kMapPointCreationSourceCount] = {};

    for(MapPoint* pMP : vpCandidates)
    {
        const float resolvedDepth = ResolveMapPointDepthInKeyFrame(mpCurrentKeyFrame, pMP);
        const int sourceIdx = static_cast<int>(MapPointCreationSourceIndex(pMP->GetCreationSource()));

        if(ShouldConsiderLongWeakMapPoint(pMP, scope))
        {
            candidateBySource[sourceIdx]++;
            if(bPrintDiag)
                AccumulateLongWeakMapPointQualityStats(candidateStats, pMP, currentKfId, resolvedDepth);
        }

        if(!ShouldConsiderLongWeakMapPoint(pMP, scope))
            continue;

        const int ageKf = std::max(0,
                                   static_cast<int>(currentKfId) -
                                   std::max(0, static_cast<int>(pMP->mnFirstKFid)));
        const float foundRatio = pMP->GetFoundRatio();
        if(ageKf < minAgeKf)
            continue;
        if(!std::isfinite(foundRatio) || foundRatio >= maxFoundRatio)
            continue;
        if(!std::isfinite(resolvedDepth) || resolvedDepth <= minDepth)
            continue;

        culledBySource[sourceIdx]++;
        if(bPrintDiag)
            AccumulateLongWeakMapPointQualityStats(culledStats, pMP, currentKfId, resolvedDepth);
        pMP->SetBadFlag();
    }

    if(!bPrintDiag)
        return;

    const std::ios::fmtflags oldFlags = std::cout.flags();
    const std::streamsize oldPrecision = std::cout.precision();
    std::cout << std::fixed << std::setprecision(6)
              << "[LocalMapping][LongWeakMP]"
              << " kf_id=" << currentKfId
              << " interval=" << interval
              << " scope=" << scope
              << " max_found_ratio=" << maxFoundRatio
              << " min_depth=" << minDepth
              << " min_age_kf=" << minAgeKf
              << " candidate_kfs=" << vpCandidateKFs.size()
              << " unique_candidates=" << vpCandidates.size()
              << " scoped_candidates=" << candidateStats.total
              << " culled=" << culledStats.total
              << " candidate_local_mapping_triangulation="
              << candidateBySource[MapPointCreationSourceIndex(MapPoint::kCreationSourceLocalMappingTriangulation)]
              << " culled_local_mapping_triangulation="
              << culledBySource[MapPointCreationSourceIndex(MapPoint::kCreationSourceLocalMappingTriangulation)]
              << std::endl;

    if(IsLocalMappingFeatureDiagEnabled())
    {
        std::cout << std::fixed << std::setprecision(6)
                  << "[LocalMapping][LongWeakMPQuality]"
                  << " kf_id=" << currentKfId;
        AppendLongWeakMapPointQualityStats(std::cout, "candidate", candidateStats);
        AppendLongWeakMapPointQualityStats(std::cout, "culled", culledStats);
        std::cout << std::endl;

        for(size_t i = 0; i < kMapPointCreationSourceCount; ++i)
        {
            if(candidateBySource[i] <= 0 && culledBySource[i] <= 0)
                continue;
            const MapPoint::CreationSource source = static_cast<MapPoint::CreationSource>(i);
            std::cout << std::fixed << std::setprecision(6)
                      << "[LocalMapping][LongWeakMPSource]"
                      << " kf_id=" << currentKfId
                      << " source=" << MapPointCreationSourceName(source)
                      << " candidates=" << candidateBySource[i]
                      << " culled=" << culledBySource[i]
                      << std::endl;
        }
    }

    std::cout.flags(oldFlags);
    std::cout.precision(oldPrecision);
}


void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    // Keep monocular triangulation fan-out aligned with the ORB/Rover-style
    // backend contract; wider neighbor sets overproduce weak XFeat MapPoints.
    if(mbMonocular)
        nn=10;
    vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    if (mbInertial)
    {
        KeyFrame* pKF = mpCurrentKeyFrame;
        int count=0;
        while((vpNeighKFs.size()<=nn)&&(pKF->mPrevKF)&&(count++<nn))
        {
            vector<KeyFrame*>::iterator it = std::find(vpNeighKFs.begin(), vpNeighKFs.end(), pKF->mPrevKF);
            if(it==vpNeighKFs.end())
                vpNeighKFs.push_back(pKF->mPrevKF);
            pKF = pKF->mPrevKF;
        }
    }

    float th = 0.6f;
    const bool bUseORB = (std::getenv("USE_ORB") != nullptr);
    ORBmatcher matcherORB(th,false);
    XFeatMatcher matcherXFeat(0.8f,false);

    Sophus::SE3<float> sophTcw1 = mpCurrentKeyFrame->GetPose();
    Eigen::Matrix<float,3,4> eigTcw1 = sophTcw1.matrix3x4();
    Eigen::Matrix<float,3,3> Rcw1 = eigTcw1.block<3,3>(0,0);
    Eigen::Matrix<float,3,3> Rwc1 = Rcw1.transpose();
    Eigen::Vector3f tcw1 = sophTcw1.translation();
    Eigen::Vector3f Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;
    const float ratioFactorClose = 1.25f*mpCurrentKeyFrame->mfScaleFactor;
    const bool bPrintTriangDiag = ShouldPrintLocalMappingXFeatDebug(mpCurrentKeyFrame->mnId);
    int countStereo = 0;
    int countStereoGoodProj = 0;
    int countStereoAttempt = 0;
    int totalStereoPts = 0;
    int closeCandidates = 0;
    int closeRayCandidates = 0;
    int closeStereoFallbackCandidates = 0;
    int closeStereoFallbackReject = 0;
    int closeParallaxReject = 0;
    int closeReprojReject = 0;
    int closeScaleReject = 0;
    int closeCreated = 0;
    int close0To5Candidates = 0;
    int close0To5ParallaxReject = 0;
    int close0To5Created = 0;
    int close5To10Candidates = 0;
    int close5To10ParallaxReject = 0;
    int close5To10Created = 0;
    int mappingFeatureBudgetCandidate = 0;
    int mappingFeatureBudgetAccept = 0;
    int mappingFeatureBudgetReject = 0;
    int mappingFeatureBudgetRejectGrid = 0;
    int mappingFeatureBudgetRejectLowResponse = 0;
    int mappingFeatureBudgetRejectOctave2p = 0;
    int mappingFeatureBudgetRejectEdgeGrid = 0;
    int mappingFeatureBudgetRejectDuplicate = 0;
    int mappingFeatureBudgetRejectKeypointDuplicate = 0;
    int mappingFeatureBudgetUniqueCandidates = 0;
    int mappingFeatureBudgetAcceptedLowResponse = 0;
    int mappingFeatureBudgetAcceptedOctave2p = 0;
    int mappingFeatureBudgetAcceptedEdgeGrid = 0;
    int closeProfileEligibleCandidates = 0;
    int closeProfileEligibleCreated = 0;
    int closeProfileFallbackCandidates = 0;
    int closeProfileFallbackCreated = 0;
    int closePairSelectionCandidate = 0;
    int closePairSelectionUniqueCurrent = 0;
    int closePairSelectionCreated = 0;
    int closePairSelectionDuplicateReject = 0;
    int closePairSelectionBestTop1 = 0;
    int closePairSelectionBestTop3 = 0;
    int closePairSelectionBestTop10 = 0;
    int closePairSelectionBestLate = 0;
    int closePairSelection5To10Candidate = 0;
    int closePairSelection5To10Created = 0;
    int mappingFeatureRepeatabilityCandidate = 0;
    int mappingFeatureRepeatabilityRepeatableCandidate = 0;
    int mappingFeatureRepeatabilitySingleSupportCandidate = 0;
    int mappingFeatureRepeatabilityAccept = 0;
    int mappingFeatureRepeatabilityFallbackAccept = 0;
    int mappingFeatureRepeatabilityRejectSingleSupport = 0;
    int mappingFeatureRepeatabilityShadowRejectSingleSupport = 0;
    int mappingTrackletCandidate = 0;
    int mappingTrackletCreated = 0;
    int mappingTrackletUpdated = 0;
    int mappingTrackletPromoted = 0;
    int mappingTrackletExpired = 0;
    int mappingTrackletConflict = 0;
    int mappingTrackletPromotionFailed = 0;
    ScalarDistributionStats mappingTrackletPromotedDepthStats;
    int mappingFeatureBudgetCellCounts[kMappingFeatureBudgetGridCells] = {};
    int createdMpsTotal = 0;
    int createdMpsDepth0To5 = 0;
    int createdMpsDepth5To10 = 0;
    int createdMpsDepth10To30 = 0;
    int createdMpsDepth30To100 = 0;
    int createdMpsDepth100p = 0;
    int createdMpsDepthInvalid = 0;
    int mappingProbationCreated = 0;
    int mappingProbationCreated0To5 = 0;
    int mappingProbationCreated5To10 = 0;
    ScalarDistributionStats createdTriangDepthStats;
    static constexpr int kMappingFeatureGridBins = 4;
    static constexpr int kMappingFeatureOctaveBins = 4;
    static constexpr int kMappingFeatureResponseBins = 4;
    static constexpr int kMappingFeatureDepthBins = 6;
    static constexpr int kMappingFeatureCombinedBins = 6;
    MappingFeatureCreateStats mappingFeatureGridStats[kMappingFeatureGridBins];
    MappingFeatureCreateStats mappingFeatureOctaveStats[kMappingFeatureOctaveBins];
    MappingFeatureCreateStats mappingFeatureResponseStats[kMappingFeatureResponseBins];
    MappingFeatureCreateStats mappingFeatureDepthStats[kMappingFeatureDepthBins];
    MappingFeatureCreateStats mappingFeatureCombinedStats[kMappingFeatureCombinedBins];
    static constexpr int kTriangInputBackendBins = 4;
    static constexpr int kTriangInputDepthBins = 2;
    static constexpr int kTriangInputOctavePairBins = 4;
    static constexpr int kTriangInputCurrentOctaveBins = 4;
    static constexpr int kTriangInputResponseBins = 4;
    static constexpr int kTriangInputGridBins = 4;
    static constexpr int kTriangInputEdgeDistanceBins = 4;
    static constexpr int kTriangInputNeighborRankBins = 5;
    static constexpr int kTriangInputKfIdDeltaBins = 5;
    static constexpr int kTriangInputCovisWeightBins = 4;
    static constexpr int kTriangInputBaselineBins = 4;
    static constexpr int kTriangInputBaselineDepthRatioBins = 5;
    static constexpr int kTriangInputKf2OctaveBins = 4;
    static constexpr int kTriangInputKf2ResponseBins = 4;
    static constexpr int kTriangInputKf2EdgeDistanceBins = 4;
    TriangInputFailureStats triangInputBackendStats[kTriangInputBackendBins];
    TriangInputFailureStats triangInputDepthStats[kTriangInputDepthBins];
    TriangInputFailureStats triangInputOctavePairStats[kTriangInputOctavePairBins];
    TriangInputFailureStats triangInputCurrentOctaveStats[kTriangInputCurrentOctaveBins];
    TriangInputFailureStats triangInputResponseStats[kTriangInputResponseBins];
    TriangInputFailureStats triangInputGridStats[kTriangInputGridBins];
    TriangInputFailureStats triangInputEdgeDistanceStats[kTriangInputEdgeDistanceBins];
    TriangInputFailureStats triangInputNeighborRankStats[kTriangInputNeighborRankBins];
    TriangInputFailureStats triangInputKfIdDeltaStats[kTriangInputKfIdDeltaBins];
    TriangInputFailureStats triangInputCovisWeightStats[kTriangInputCovisWeightBins];
    TriangInputFailureStats triangInputBaselineStats[kTriangInputBaselineBins];
    TriangInputFailureStats triangInputBaselineDepthRatioStats[kTriangInputBaselineDepthRatioBins];
    TriangInputFailureStats triangInputKf2OctaveStats[kTriangInputKf2OctaveBins];
    TriangInputFailureStats triangInputKf2ResponseStats[kTriangInputKf2ResponseBins];
    TriangInputFailureStats triangInputKf2EdgeDistanceStats[kTriangInputKf2EdgeDistanceBins];
    const bool bCollectTriangInputFailureDiag =
        bPrintTriangDiag &&
        IsLocalMappingFeatureDiagEnabled() &&
        IsLocalMappingDeepDiagEnabled() &&
        !bUseORB &&
        IsXFeatKeyFrameDescriptor(mpCurrentKeyFrame);

    TriangInputPairContext currentTriangInputPairContext;

    auto accumulateTriangInputFailureDiagWithContext =
        [&](const TriangInputPairContext& context,
            KeyFrame* pKF2,
            const cv::KeyPoint& kp1,
            const cv::KeyPoint& kp2,
            const MapPoint::TriangulationMatchBackend backend,
            const float closeDepth,
            const TriangInputTerminalEvent event,
            const float chi2Kf1,
            const float chi2Kf2,
            const bool reprojKf1Reject,
            const bool reprojKf2Reject,
            const float cosParallax,
            const float scaleResidual)
    {
        if(!bCollectTriangInputFailureDiag || !IsXFeatKeyFrameDescriptor(pKF2))
            return;
        const int depthBin = TriangInputCloseDepthBin(closeDepth);
        if(depthBin < 0 || depthBin >= kTriangInputDepthBins)
            return;

        int gridX = -1;
        int gridY = -1;
        KeyFrameFeatureGridPosition(mpCurrentKeyFrame, kp1, gridX, gridY);

        const int backendBin = std::max(0, std::min(kTriangInputBackendBins - 1,
                                                    TriangInputBackendBin(backend)));
        const int octavePairBin = std::max(0, std::min(kTriangInputOctavePairBins - 1,
                                                       TriangInputOctavePairBin(kp1.octave, kp2.octave)));
        const int currentOctaveBin = std::max(0, std::min(kTriangInputCurrentOctaveBins - 1,
                                                          MappingFeatureOctaveBin(kp1.octave)));
        const int responseBin = std::max(0, std::min(kTriangInputResponseBins - 1,
                                                     MappingFeatureResponseBin(kp1.response)));
        const int gridBin = std::max(0, std::min(kTriangInputGridBins - 1,
                                                 MappingFeatureGridBin(gridX, gridY)));
        const int edgeDistanceBin = std::max(0, std::min(kTriangInputEdgeDistanceBins - 1,
                                                         TriangInputEdgeDistanceBin(mpCurrentKeyFrame, kp1)));
        const int neighborRankBin = std::max(0, std::min(kTriangInputNeighborRankBins - 1,
                                                         TriangInputNeighborRankBin(context.neighborRank)));
        const int kfIdDeltaBin = std::max(0, std::min(kTriangInputKfIdDeltaBins - 1,
                                                      TriangInputKfIdDeltaBin(mpCurrentKeyFrame, pKF2)));
        const int covisWeightBin = std::max(0, std::min(kTriangInputCovisWeightBins - 1,
                                                        TriangInputCovisWeightBin(context.covisWeight)));
        const int baselineBin = std::max(0, std::min(kTriangInputBaselineBins - 1,
                                                     TriangInputBaselineBin(context.baseline)));
        const int baselineDepthRatioBin = std::max(0, std::min(kTriangInputBaselineDepthRatioBins - 1,
                                                               TriangInputBaselineDepthRatioBin(context.baselineDepthRatio)));
        const int kf2OctaveBin = std::max(0, std::min(kTriangInputKf2OctaveBins - 1,
                                                      MappingFeatureOctaveBin(kp2.octave)));
        const int kf2ResponseBin = std::max(0, std::min(kTriangInputKf2ResponseBins - 1,
                                                        MappingFeatureResponseBin(kp2.response)));
        const int kf2EdgeDistanceBin = std::max(0, std::min(kTriangInputKf2EdgeDistanceBins - 1,
                                                            TriangInputEdgeDistanceBin(pKF2, kp2)));

        AccumulateTriangInputFailureStats(triangInputBackendStats[backendBin],
                                          event,
                                          chi2Kf1,
                                          chi2Kf2,
                                          reprojKf1Reject,
                                          reprojKf2Reject,
                                          cosParallax,
                                          scaleResidual);
        AccumulateTriangInputFailureStats(triangInputDepthStats[depthBin],
                                          event,
                                          chi2Kf1,
                                          chi2Kf2,
                                          reprojKf1Reject,
                                          reprojKf2Reject,
                                          cosParallax,
                                          scaleResidual);
        AccumulateTriangInputFailureStats(triangInputOctavePairStats[octavePairBin],
                                          event,
                                          chi2Kf1,
                                          chi2Kf2,
                                          reprojKf1Reject,
                                          reprojKf2Reject,
                                          cosParallax,
                                          scaleResidual);
        AccumulateTriangInputFailureStats(triangInputCurrentOctaveStats[currentOctaveBin],
                                          event,
                                          chi2Kf1,
                                          chi2Kf2,
                                          reprojKf1Reject,
                                          reprojKf2Reject,
                                          cosParallax,
                                          scaleResidual);
        AccumulateTriangInputFailureStats(triangInputResponseStats[responseBin],
                                          event,
                                          chi2Kf1,
                                          chi2Kf2,
                                          reprojKf1Reject,
                                          reprojKf2Reject,
                                          cosParallax,
                                          scaleResidual);
        AccumulateTriangInputFailureStats(triangInputGridStats[gridBin],
                                          event,
                                          chi2Kf1,
                                          chi2Kf2,
                                          reprojKf1Reject,
                                          reprojKf2Reject,
                                          cosParallax,
                                          scaleResidual);
        AccumulateTriangInputFailureStats(triangInputEdgeDistanceStats[edgeDistanceBin],
                                          event,
                                          chi2Kf1,
                                          chi2Kf2,
                                          reprojKf1Reject,
                                          reprojKf2Reject,
                                          cosParallax,
                                          scaleResidual);
        AccumulateTriangInputFailureStats(triangInputNeighborRankStats[neighborRankBin],
                                          event,
                                          chi2Kf1,
                                          chi2Kf2,
                                          reprojKf1Reject,
                                          reprojKf2Reject,
                                          cosParallax,
                                          scaleResidual);
        AccumulateTriangInputFailureStats(triangInputKfIdDeltaStats[kfIdDeltaBin],
                                          event,
                                          chi2Kf1,
                                          chi2Kf2,
                                          reprojKf1Reject,
                                          reprojKf2Reject,
                                          cosParallax,
                                          scaleResidual);
        AccumulateTriangInputFailureStats(triangInputCovisWeightStats[covisWeightBin],
                                          event,
                                          chi2Kf1,
                                          chi2Kf2,
                                          reprojKf1Reject,
                                          reprojKf2Reject,
                                          cosParallax,
                                          scaleResidual);
        AccumulateTriangInputFailureStats(triangInputBaselineStats[baselineBin],
                                          event,
                                          chi2Kf1,
                                          chi2Kf2,
                                          reprojKf1Reject,
                                          reprojKf2Reject,
                                          cosParallax,
                                          scaleResidual);
        AccumulateTriangInputFailureStats(triangInputBaselineDepthRatioStats[baselineDepthRatioBin],
                                          event,
                                          chi2Kf1,
                                          chi2Kf2,
                                          reprojKf1Reject,
                                          reprojKf2Reject,
                                          cosParallax,
                                          scaleResidual);
        AccumulateTriangInputFailureStats(triangInputKf2OctaveStats[kf2OctaveBin],
                                          event,
                                          chi2Kf1,
                                          chi2Kf2,
                                          reprojKf1Reject,
                                          reprojKf2Reject,
                                          cosParallax,
                                          scaleResidual);
        AccumulateTriangInputFailureStats(triangInputKf2ResponseStats[kf2ResponseBin],
                                          event,
                                          chi2Kf1,
                                          chi2Kf2,
                                          reprojKf1Reject,
                                          reprojKf2Reject,
                                          cosParallax,
                                          scaleResidual);
        AccumulateTriangInputFailureStats(triangInputKf2EdgeDistanceStats[kf2EdgeDistanceBin],
                                          event,
                                          chi2Kf1,
                                          chi2Kf2,
                                          reprojKf1Reject,
                                          reprojKf2Reject,
                                          cosParallax,
                                          scaleResidual);
    };

    auto accumulateTriangInputFailureDiag =
        [&](KeyFrame* pKF2,
            const cv::KeyPoint& kp1,
            const cv::KeyPoint& kp2,
            const MapPoint::TriangulationMatchBackend backend,
            const float closeDepth,
            const TriangInputTerminalEvent event,
            const float chi2Kf1,
            const float chi2Kf2,
            const bool reprojKf1Reject,
            const bool reprojKf2Reject,
            const float cosParallax,
            const float scaleResidual)
    {
        accumulateTriangInputFailureDiagWithContext(currentTriangInputPairContext,
                                                    pKF2,
                                                    kp1,
                                                    kp2,
                                                    backend,
                                                    closeDepth,
                                                    event,
                                                    chi2Kf1,
                                                    chi2Kf2,
                                                    reprojKf1Reject,
                                                    reprojKf2Reject,
                                                    cosParallax,
                                                    scaleResidual);
    };

    struct MappingTriangulationCandidate
    {
        KeyFrame* pKF1 = nullptr;
        KeyFrame* pKF2 = nullptr;
        size_t idx1 = 0;
        size_t idx2 = 0;
        cv::KeyPoint diagKp1;
        cv::KeyPoint diagKp2;
        Eigen::Vector3f x3D = Eigen::Vector3f::Zero();
        float triangDepth = -1.0f;
        float cosParallaxRays = 1.0f;
        float reprojChi2First = std::numeric_limits<float>::quiet_NaN();
        float reprojChi2Second = std::numeric_limits<float>::quiet_NaN();
        float reprojChi2Max = std::numeric_limits<float>::quiet_NaN();
        float reprojChi2Mean = std::numeric_limits<float>::quiet_NaN();
        float scaleLogResidual = std::numeric_limits<float>::quiet_NaN();
        float scaleLogMargin = std::numeric_limits<float>::quiet_NaN();
        MapPoint::TriangulationMatchBackend triangBackend = MapPoint::kTriangBackendUnknown;
        int mappingFeatureGridX = -1;
        int mappingFeatureGridY = -1;
        int mappingFeatureGridBin = 0;
        int mappingFeatureOctaveBin = 0;
        int mappingFeatureResponseBin = 0;
        int mappingFeatureStereoQualityBin = 0;
        int mappingFeatureCombinedBin = 0;
        int mappingFeatureBudgetCell = -1;
        int octave = -1;
        float response = 0.0f;
        bool bPointStereo = false;
        bool bCloseRayCandidate = false;
        bool bCloseRay0To5 = false;
        bool bCloseRay5To10 = false;
        bool bProfileEligible = true;
        float qualityScore = 0.0f;
        int repeatabilitySupport = 1;
        bool bRepeatabilitySupported = false;
        TriangInputPairContext triangInputPairContext;
    };

    std::vector<MappingTriangulationCandidate> vPendingMappingFeatureCandidates;
    std::vector<MappingTriangulationCandidate> vProfileFallbackCandidates;
    std::vector<MappingTriangulationCandidate> vCloseMidPairSelectionCandidates;

    auto computeMappingCandidateQualityScore =
        [](const MappingTriangulationCandidate& candidate) -> float
    {
        const float parallaxMargin =
            std::max(0.0f, 0.9994f - candidate.cosParallaxRays);
        const float parallaxScore = std::min(2.0f, parallaxMargin * 1000.0f);
        const float reprojPenalty =
            std::isfinite(candidate.reprojChi2Max) ? 0.45f * candidate.reprojChi2Max : 3.0f;
        const float scaleScore =
            std::isfinite(candidate.scaleLogMargin) ?
                std::max(-1.0f, std::min(1.0f, candidate.scaleLogMargin)) : -1.0f;
        const float responseScore =
            std::isfinite(candidate.response) ?
                std::max(0.0f, std::min(2.0f, candidate.response * 2.0f)) : 0.0f;
        const float octavePenalty =
            candidate.octave >= 0 ? 0.60f * static_cast<float>(candidate.octave) : 1.0f;
        const float edgePenalty = candidate.mappingFeatureGridBin == 1 ? 0.15f : 0.0f;

        return 2.0f * parallaxScore + 1.5f * scaleScore + responseScore -
               reprojPenalty - octavePenalty - edgePenalty;
    };

    auto computeCloseMidPairSelectionScore =
        [](const MappingTriangulationCandidate& candidate) -> float
    {
        const float baselineDepthRatio = candidate.triangInputPairContext.baselineDepthRatio;
        float baselineDepthScore = 0.0f;
        if(std::isfinite(baselineDepthRatio) && baselineDepthRatio > 0.0f)
        {
            baselineDepthScore = std::min(2.0f, baselineDepthRatio * 25.0f);
        }

        const float parallaxMargin =
            std::max(0.0f, 0.9994f - candidate.cosParallaxRays);
        const float parallaxScore = std::min(2.0f, parallaxMargin * 1000.0f);
        const float reprojPenalty =
            std::isfinite(candidate.reprojChi2Max) ? 0.35f * candidate.reprojChi2Max : 2.0f;
        const float scaleScore =
            std::isfinite(candidate.scaleLogMargin) ?
                std::max(-1.0f, std::min(1.0f, candidate.scaleLogMargin)) : -1.0f;

        return candidate.qualityScore +
               1.25f * baselineDepthScore +
               0.75f * parallaxScore +
               0.50f * scaleScore -
               reprojPenalty;
    };

    auto isKeyFrameSlotFree = [](KeyFrame* pKF, const size_t idx) -> bool
    {
        if(!pKF)
            return false;
        MapPoint* pExisting = pKF->GetMapPoint(idx);
        return !pExisting || pExisting->isBad();
    };

    auto createMapPointFromCandidate =
        [&](const MappingTriangulationCandidate& candidate) -> bool
    {
        KeyFrame* pKF1 = candidate.pKF1 ? candidate.pKF1 : mpCurrentKeyFrame;
        if(!pKF1 || !candidate.pKF2)
            return false;
        if(!isKeyFrameSlotFree(pKF1, candidate.idx1) ||
           !isKeyFrameSlotFree(candidate.pKF2, candidate.idx2))
        {
            return false;
        }

        MapPoint* pMP = new MapPoint(candidate.x3D,
                                     pKF1,
                                     mpAtlas->GetCurrentMap());
        pMP->SetCreationSource(MapPoint::kCreationSourceLocalMappingTriangulation);
        pMP->SetTriangulationQuality(candidate.triangDepth,
                                     candidate.cosParallaxRays,
                                     candidate.reprojChi2Max,
                                     candidate.reprojChi2Mean,
                                     candidate.scaleLogResidual,
                                     candidate.scaleLogMargin,
                                     candidate.triangBackend,
                                     pKF1->mnId,
                                     candidate.pKF2->mnId);
        pMP->SetMappingFeatureQuality(pKF1->mnId,
                                      static_cast<int>(candidate.idx1),
                                      candidate.mappingFeatureGridX,
                                      candidate.mappingFeatureGridY,
                                      candidate.octave,
                                      candidate.response,
                                      candidate.triangDepth,
                                      Frame::kStereoBirthInvalid);
        if(kEnableMappingProbationBehavior &&
           !bUseORB &&
           (candidate.bCloseRay0To5 || candidate.bCloseRay5To10))
        {
            pMP->SetMappingProbation(true,
                                     mpCurrentKeyFrame->mnId,
                                     candidate.triangDepth);
            mappingProbationCreated++;
            if(candidate.bCloseRay0To5)
                mappingProbationCreated0To5++;
            if(candidate.bCloseRay5To10)
                mappingProbationCreated5To10++;
        }
        if(pMP->HasMappingFeatureQuality())
        {
            const int depthBin = LocalMappingDepthBinIndex(pMP->mMappingFeatureDepth);
            const bool bCloseMappingFeature =
                std::isfinite(pMP->mMappingFeatureDepth) &&
                pMP->mMappingFeatureDepth > 0.0f &&
                pMP->mMappingFeatureDepth <= mpCurrentKeyFrame->mThDepth;
            AccumulateMappingFeatureCreateStats(mappingFeatureGridStats[candidate.mappingFeatureGridBin],
                                                pMP->mMappingFeatureDepth,
                                                bCloseMappingFeature);
            AccumulateMappingFeatureCreateStats(mappingFeatureOctaveStats[candidate.mappingFeatureOctaveBin],
                                                pMP->mMappingFeatureDepth,
                                                bCloseMappingFeature);
            AccumulateMappingFeatureCreateStats(mappingFeatureResponseStats[candidate.mappingFeatureResponseBin],
                                                pMP->mMappingFeatureDepth,
                                                bCloseMappingFeature);
            AccumulateMappingFeatureCreateStats(mappingFeatureDepthStats[depthBin],
                                                pMP->mMappingFeatureDepth,
                                                bCloseMappingFeature);
            AccumulateMappingFeatureCreateStats(mappingFeatureCombinedStats[candidate.mappingFeatureCombinedBin],
                                                pMP->mMappingFeatureDepth,
                                                bCloseMappingFeature);
        }

        if(candidate.bPointStereo)
            countStereo++;
        createdMpsTotal++;
        AccumulateScalarDistribution(createdTriangDepthStats, candidate.triangDepth, true);
        switch(LocalMappingDepthBinIndex(candidate.triangDepth))
        {
            case 1:
                createdMpsDepth0To5++;
                break;
            case 2:
                createdMpsDepth5To10++;
                break;
            case 3:
                createdMpsDepth10To30++;
                break;
            case 4:
                createdMpsDepth30To100++;
                break;
            case 5:
                createdMpsDepth100p++;
                break;
            case 0:
            default:
                createdMpsDepthInvalid++;
                break;
        }
        if(candidate.bCloseRayCandidate)
        {
            closeCreated++;
            if(candidate.bCloseRay0To5)
                close0To5Created++;
            if(candidate.bCloseRay5To10)
                close5To10Created++;
            accumulateTriangInputFailureDiagWithContext(candidate.triangInputPairContext,
                                                        candidate.pKF2,
                                                        candidate.diagKp1,
                                                        candidate.diagKp2,
                                                        candidate.triangBackend,
                                                        candidate.triangDepth,
                                                        TriangInputTerminalEvent::Created,
                                                        candidate.reprojChi2First,
                                                        candidate.reprojChi2Second,
                                                        false,
                                                        false,
                                                        candidate.cosParallaxRays,
                                                        candidate.scaleLogResidual);
        }

        pMP->AddObservation(pKF1, candidate.idx1);
        pMP->AddObservation(candidate.pKF2, candidate.idx2);

        pKF1->AddMapPoint(pMP, candidate.idx1);
        candidate.pKF2->AddMapPoint(pMP, candidate.idx2);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        mpAtlas->AddMapPoint(pMP);
        mlpRecentAddedMapPoints.push_back(pMP);
        return true;
    };

    auto pendingObservationKey =
        [](KeyFrame* pKF, const size_t idx) -> std::pair<long unsigned int, size_t>
    {
        return std::make_pair(pKF ? pKF->mnId : 0UL, idx);
    };

    auto erasePendingTrackletIndex = [&](PendingMappingTracklet& tracklet)
    {
        for(const PendingMappingObservation& obs : tracklet.observations)
        {
            if(obs.pKF)
                mPendingMappingObservationIndex.erase(pendingObservationKey(obs.pKF, obs.idx));
        }
    };

    auto expireOldPendingTracklets = [&]()
    {
        if(!mpCurrentKeyFrame)
            return;
        const long unsigned int currentKfId = mpCurrentKeyFrame->mnId;
        for(PendingMappingTracklet& tracklet : mvPendingMappingTracklets)
        {
            if(tracklet.bad)
                continue;
            const bool bTooOld =
                currentKfId > tracklet.firstKFid &&
                static_cast<int>(currentKfId - tracklet.firstKFid) > kPendingMappingTrackletMaxAgeKFs;
            const bool bTooManyConflicts =
                tracklet.conflictCount > kPendingMappingTrackletMaxConflicts;
            if(!bTooOld && !bTooManyConflicts)
                continue;
            erasePendingTrackletIndex(tracklet);
            tracklet.bad = true;
            mappingTrackletExpired++;
        }
    };

    auto addPendingObservation =
        [&](PendingMappingTracklet& tracklet,
            const size_t trackletIndex,
            KeyFrame* pKF,
            const size_t idx) -> bool
    {
        if(!pKF)
            return false;
        const auto key = pendingObservationKey(pKF, idx);
        const auto itExisting = mPendingMappingObservationIndex.find(key);
        if(itExisting != mPendingMappingObservationIndex.end())
        {
            if(itExisting->second == trackletIndex)
                return false;
            tracklet.conflictCount++;
            mappingTrackletConflict++;
            return false;
        }
        if(!isKeyFrameSlotFree(pKF, idx))
        {
            tracklet.conflictCount++;
            mappingTrackletConflict++;
            return false;
        }
        PendingMappingObservation obs;
        obs.pKF = pKF;
        obs.idx = idx;
        tracklet.observations.push_back(obs);
        mPendingMappingObservationIndex[key] = trackletIndex;
        return true;
    };

    auto updatePendingTrackletBest =
        [](PendingMappingTracklet& tracklet,
           const MappingTriangulationCandidate& candidate)
    {
        if(candidate.qualityScore <= tracklet.qualityScore)
            return;
        tracklet.pBestKF1 = candidate.pKF1;
        tracklet.pBestKF2 = candidate.pKF2;
        tracklet.bestIdx1 = candidate.idx1;
        tracklet.bestIdx2 = candidate.idx2;
        tracklet.x3D = candidate.x3D;
        tracklet.triangDepth = candidate.triangDepth;
        tracklet.cosParallaxRays = candidate.cosParallaxRays;
        tracklet.reprojChi2Max = candidate.reprojChi2Max;
        tracklet.reprojChi2Mean = candidate.reprojChi2Mean;
        tracklet.scaleLogResidual = candidate.scaleLogResidual;
        tracklet.scaleLogMargin = candidate.scaleLogMargin;
        tracklet.triangBackend = candidate.triangBackend;
        tracklet.mappingFeatureGridX = candidate.mappingFeatureGridX;
        tracklet.mappingFeatureGridY = candidate.mappingFeatureGridY;
        tracklet.mappingFeatureGridBin = candidate.mappingFeatureGridBin;
        tracklet.mappingFeatureOctaveBin = candidate.mappingFeatureOctaveBin;
        tracklet.mappingFeatureResponseBin = candidate.mappingFeatureResponseBin;
        tracklet.mappingFeatureStereoQualityBin = candidate.mappingFeatureStereoQualityBin;
        tracklet.mappingFeatureCombinedBin = candidate.mappingFeatureCombinedBin;
        tracklet.octave = candidate.octave;
        tracklet.response = candidate.response;
        tracklet.bCloseRay0To5 = candidate.bCloseRay0To5;
        tracklet.bCloseRay5To10 = candidate.bCloseRay5To10;
        tracklet.qualityScore = candidate.qualityScore;
    };

    auto createMapPointFromPendingTracklet =
        [&](PendingMappingTracklet& tracklet) -> bool
    {
        if(tracklet.bad ||
           tracklet.updatePasses < kPendingMappingTrackletMinUpdatePasses ||
           static_cast<int>(tracklet.observations.size()) < kPendingMappingTrackletMinObservations ||
           !tracklet.pBestKF1 ||
           !tracklet.pBestKF2)
        {
            return false;
        }

        std::vector<PendingMappingObservation> freeObservations;
        freeObservations.reserve(tracklet.observations.size());
        std::set<long unsigned int> uniqueKFs;
        std::set<long unsigned int> selectedObservationKFs;
        for(const PendingMappingObservation& obs : tracklet.observations)
        {
            if(!obs.pKF || obs.pKF->isBad())
                continue;
            if(!isKeyFrameSlotFree(obs.pKF, obs.idx))
                continue;
            if(!selectedObservationKFs.insert(obs.pKF->mnId).second)
                continue;
            freeObservations.push_back(obs);
            uniqueKFs.insert(obs.pKF->mnId);
        }

        if(static_cast<int>(uniqueKFs.size()) < kPendingMappingTrackletMinObservations ||
           freeObservations.size() < 2)
        {
            mappingTrackletPromotionFailed++;
            return false;
        }

        KeyFrame* pCreationKF = freeObservations.front().pKF;
        if(!pCreationKF)
        {
            mappingTrackletPromotionFailed++;
            return false;
        }

        MapPoint* pMP = new MapPoint(tracklet.x3D,
                                     pCreationKF,
                                     mpAtlas->GetCurrentMap());
        pMP->SetCreationSource(MapPoint::kCreationSourceLocalMappingTriangulation);
        pMP->SetTriangulationQuality(tracklet.triangDepth,
                                     tracklet.cosParallaxRays,
                                     tracklet.reprojChi2Max,
                                     tracklet.reprojChi2Mean,
                                     tracklet.scaleLogResidual,
                                     tracklet.scaleLogMargin,
                                     tracklet.triangBackend,
                                     tracklet.pBestKF1->mnId,
                                     tracklet.pBestKF2->mnId);
        pMP->SetMappingFeatureQuality(tracklet.pBestKF1->mnId,
                                      static_cast<int>(tracklet.bestIdx1),
                                      tracklet.mappingFeatureGridX,
                                      tracklet.mappingFeatureGridY,
                                      tracklet.octave,
                                      tracklet.response,
                                      tracklet.triangDepth,
                                      Frame::kStereoBirthInvalid);

        if(pMP->HasMappingFeatureQuality())
        {
            const int depthBin = LocalMappingDepthBinIndex(pMP->mMappingFeatureDepth);
            const bool bCloseMappingFeature =
                std::isfinite(pMP->mMappingFeatureDepth) &&
                pMP->mMappingFeatureDepth > 0.0f &&
                pMP->mMappingFeatureDepth <= mpCurrentKeyFrame->mThDepth;
            AccumulateMappingFeatureCreateStats(mappingFeatureGridStats[tracklet.mappingFeatureGridBin],
                                                pMP->mMappingFeatureDepth,
                                                bCloseMappingFeature);
            AccumulateMappingFeatureCreateStats(mappingFeatureOctaveStats[tracklet.mappingFeatureOctaveBin],
                                                pMP->mMappingFeatureDepth,
                                                bCloseMappingFeature);
            AccumulateMappingFeatureCreateStats(mappingFeatureResponseStats[tracklet.mappingFeatureResponseBin],
                                                pMP->mMappingFeatureDepth,
                                                bCloseMappingFeature);
            AccumulateMappingFeatureCreateStats(mappingFeatureDepthStats[depthBin],
                                                pMP->mMappingFeatureDepth,
                                                bCloseMappingFeature);
            AccumulateMappingFeatureCreateStats(mappingFeatureCombinedStats[tracklet.mappingFeatureCombinedBin],
                                                pMP->mMappingFeatureDepth,
                                                bCloseMappingFeature);
        }

        createdMpsTotal++;
        AccumulateScalarDistribution(createdTriangDepthStats, tracklet.triangDepth, true);
        AccumulateScalarDistribution(mappingTrackletPromotedDepthStats, tracklet.triangDepth, true);
        switch(LocalMappingDepthBinIndex(tracklet.triangDepth))
        {
            case 1:
                createdMpsDepth0To5++;
                break;
            case 2:
                createdMpsDepth5To10++;
                break;
            case 3:
                createdMpsDepth10To30++;
                break;
            case 4:
                createdMpsDepth30To100++;
                break;
            case 5:
                createdMpsDepth100p++;
                break;
            case 0:
            default:
                createdMpsDepthInvalid++;
                break;
        }

        closeCreated++;
        if(tracklet.bCloseRay0To5)
            close0To5Created++;
        if(tracklet.bCloseRay5To10)
            close5To10Created++;

        for(const PendingMappingObservation& obs : freeObservations)
        {
            pMP->AddObservation(obs.pKF, obs.idx);
            obs.pKF->AddMapPoint(pMP, obs.idx);
        }

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        mpAtlas->AddMapPoint(pMP);
        mlpRecentAddedMapPoints.push_back(pMP);

        erasePendingTrackletIndex(tracklet);
        tracklet.bad = true;
        mappingTrackletPromoted++;
        return true;
    };

    auto updateOrCreatePendingTracklet =
        [&](const MappingTriangulationCandidate& candidate)
    {
        if(!candidate.pKF1 || !candidate.pKF2)
            return;

        mappingTrackletCandidate++;
        const auto key1 = pendingObservationKey(candidate.pKF1, candidate.idx1);
        const auto key2 = pendingObservationKey(candidate.pKF2, candidate.idx2);
        size_t trackletIndex = std::numeric_limits<size_t>::max();
        const auto it1 = mPendingMappingObservationIndex.find(key1);
        const auto it2 = mPendingMappingObservationIndex.find(key2);
        if(it1 != mPendingMappingObservationIndex.end())
            trackletIndex = it1->second;
        if(it2 != mPendingMappingObservationIndex.end())
        {
            if(trackletIndex == std::numeric_limits<size_t>::max())
            {
                trackletIndex = it2->second;
            }
            else if(trackletIndex != it2->second &&
                    trackletIndex < mvPendingMappingTracklets.size())
            {
                mvPendingMappingTracklets[trackletIndex].conflictCount++;
                mappingTrackletConflict++;
            }
        }

        if(trackletIndex == std::numeric_limits<size_t>::max() ||
           trackletIndex >= mvPendingMappingTracklets.size() ||
           mvPendingMappingTracklets[trackletIndex].bad)
        {
            PendingMappingTracklet tracklet;
            tracklet.id = mnNextPendingMappingTrackletId++;
            tracklet.firstKFid = candidate.pKF1->mnId;
            tracklet.lastKFid = std::max(candidate.pKF1->mnId, candidate.pKF2->mnId);
            tracklet.lastUpdateKFid = mpCurrentKeyFrame ? mpCurrentKeyFrame->mnId : candidate.pKF1->mnId;
            tracklet.updatePasses = 1;
            updatePendingTrackletBest(tracklet, candidate);
            mvPendingMappingTracklets.push_back(tracklet);
            trackletIndex = mvPendingMappingTracklets.size() - 1;
            PendingMappingTracklet& storedTracklet = mvPendingMappingTracklets.back();
            addPendingObservation(storedTracklet, trackletIndex, candidate.pKF1, candidate.idx1);
            addPendingObservation(storedTracklet, trackletIndex, candidate.pKF2, candidate.idx2);
            mappingTrackletCreated++;
            return;
        }

        PendingMappingTracklet& tracklet = mvPendingMappingTracklets[trackletIndex];
        const long unsigned int currentKfId =
            mpCurrentKeyFrame ? mpCurrentKeyFrame->mnId : candidate.pKF1->mnId;
        if(tracklet.lastUpdateKFid != currentKfId)
        {
            tracklet.updatePasses++;
            tracklet.lastUpdateKFid = currentKfId;
        }
        tracklet.lastKFid =
            std::max(tracklet.lastKFid, std::max(candidate.pKF1->mnId, candidate.pKF2->mnId));
        updatePendingTrackletBest(tracklet, candidate);
        addPendingObservation(tracklet, trackletIndex, candidate.pKF1, candidate.idx1);
        addPendingObservation(tracklet, trackletIndex, candidate.pKF2, candidate.idx2);
        mappingTrackletUpdated++;
        createMapPointFromPendingTracklet(tracklet);
    };

    const bool bUsePendingMappingTracklets =
        kEnableXFeatCloseMidPendingTracklet &&
        !bUseORB &&
        mpAtlas &&
        mpAtlas->GetCurrentMap() &&
        mpAtlas->GetCurrentMap()->KeyFramesInMap() >=
            static_cast<long unsigned int>(kPendingMappingTrackletBootstrapKFs);
    expireOldPendingTracklets();

    auto processPendingMappingFeatureCandidates = [&]()
    {
        std::map<size_t, std::set<long unsigned int>> currentKeypointSupportKFs;
        for(const MappingTriangulationCandidate& candidate : vPendingMappingFeatureCandidates)
        {
            if(candidate.pKF2)
                currentKeypointSupportKFs[candidate.idx1].insert(candidate.pKF2->mnId);
        }

        for(MappingTriangulationCandidate& candidate : vPendingMappingFeatureCandidates)
        {
            const auto itSupport = currentKeypointSupportKFs.find(candidate.idx1);
            candidate.repeatabilitySupport =
                itSupport != currentKeypointSupportKFs.end()
                    ? static_cast<int>(itSupport->second.size())
                    : 1;
            candidate.bRepeatabilitySupported = candidate.repeatabilitySupport >= 2;
            if(candidate.bRepeatabilitySupported)
                candidate.qualityScore += 1.0f + 0.25f * static_cast<float>(candidate.repeatabilitySupport - 2);
        }

        std::sort(vPendingMappingFeatureCandidates.begin(),
                  vPendingMappingFeatureCandidates.end(),
                  [](const MappingTriangulationCandidate& a,
                     const MappingTriangulationCandidate& b)
                  {
                      if(a.bRepeatabilitySupported != b.bRepeatabilitySupported)
                          return a.bRepeatabilitySupported;
                      if(a.repeatabilitySupport != b.repeatabilitySupport)
                          return a.repeatabilitySupport > b.repeatabilitySupport;
                      if(a.qualityScore != b.qualityScore)
                          return a.qualityScore > b.qualityScore;
                      if(a.reprojChi2Max != b.reprojChi2Max)
                          return a.reprojChi2Max < b.reprojChi2Max;
                      if(a.cosParallaxRays != b.cosParallaxRays)
                          return a.cosParallaxRays < b.cosParallaxRays;
                      if(a.response != b.response)
                          return a.response > b.response;
                      if(a.octave != b.octave)
                          return a.octave < b.octave;
                      if(a.idx1 != b.idx1)
                          return a.idx1 < b.idx1;
                      return a.idx2 < b.idx2;
                  });

        std::unordered_set<size_t> usedCurrentKeypoints;
        usedCurrentKeypoints.reserve(vPendingMappingFeatureCandidates.size());
        std::vector<MappingTriangulationCandidate> vUniqueCandidates;
        vUniqueCandidates.reserve(vPendingMappingFeatureCandidates.size());
        for(const MappingTriangulationCandidate& candidate : vPendingMappingFeatureCandidates)
        {
            if(!usedCurrentKeypoints.insert(candidate.idx1).second)
            {
                mappingFeatureBudgetReject++;
                mappingFeatureBudgetRejectDuplicate++;
                mappingFeatureBudgetRejectKeypointDuplicate++;
                continue;
            }
            vUniqueCandidates.push_back(candidate);
        }
        mappingFeatureBudgetUniqueCandidates =
            static_cast<int>(vUniqueCandidates.size());
        for(const MappingTriangulationCandidate& candidate : vUniqueCandidates)
        {
            mappingFeatureRepeatabilityCandidate++;
            if(candidate.bRepeatabilitySupported)
                mappingFeatureRepeatabilityRepeatableCandidate++;
            else
                mappingFeatureRepeatabilitySingleSupportCandidate++;
        }

        static constexpr int kMappingFeatureBudgetDepthGroups = 3;
        static constexpr int kMappingFeatureSingleSupportGlobalCap = 45;
        static constexpr int kMappingFeatureSingleSupportGroupCap = 1;
        const int nGroups =
            kMappingFeatureBudgetGridCells * kMappingFeatureBudgetDepthGroups;
        std::vector<std::vector<size_t>> groupedCandidates(nGroups);
        for(size_t i = 0; i < vUniqueCandidates.size(); ++i)
        {
            const MappingTriangulationCandidate& candidate = vUniqueCandidates[i];
            int depthGroup = 2;
            const int depthBin = LocalMappingDepthBinIndex(candidate.triangDepth);
            if(depthBin == 1)
                depthGroup = 0;
            else if(depthBin == 2)
                depthGroup = 1;

            int cell = candidate.mappingFeatureBudgetCell;
            if(cell < 0 || cell >= kMappingFeatureBudgetGridCells)
                cell = kMappingFeatureBudgetGridCells - 1;
            groupedCandidates[cell * kMappingFeatureBudgetDepthGroups + depthGroup].push_back(i);
        }

        std::vector<size_t> groupOffsets(groupedCandidates.size(), 0);
        std::vector<int> singleSupportGroupAccepted(groupedCandidates.size(), 0);
        bool bAdvanced = true;
        while(bAdvanced)
        {
            bAdvanced = false;
            for(size_t groupIdx = 0; groupIdx < groupedCandidates.size(); ++groupIdx)
            {
                if(groupOffsets[groupIdx] >= groupedCandidates[groupIdx].size())
                    continue;
                bAdvanced = true;

                const MappingTriangulationCandidate& candidate =
                    vUniqueCandidates[groupedCandidates[groupIdx][groupOffsets[groupIdx]++]];
                bool bBudgetReject = false;
                bool bRejectGrid = false;
                bool bRejectLowResponse = false;
                bool bRejectOctave2p = false;
                bool bRejectEdgeGrid = false;

                if(!isKeyFrameSlotFree(mpCurrentKeyFrame, candidate.idx1) ||
                   !isKeyFrameSlotFree(candidate.pKF2, candidate.idx2))
                {
                    mappingFeatureBudgetReject++;
                    mappingFeatureBudgetRejectDuplicate++;
                    continue;
                }

                if(candidate.mappingFeatureBudgetCell >= 0 &&
                   candidate.mappingFeatureBudgetCell < kMappingFeatureBudgetGridCells &&
                   mappingFeatureBudgetCellCounts[candidate.mappingFeatureBudgetCell] >=
                       MappingFeatureBudgetCellCap(candidate.mappingFeatureGridBin))
                {
                    bBudgetReject = true;
                    bRejectGrid = true;
                }

                if(candidate.mappingFeatureResponseBin == 1 &&
                   mappingFeatureBudgetAcceptedLowResponse >= 80)
                {
                    bBudgetReject = true;
                    bRejectLowResponse = true;
                }

                if(candidate.mappingFeatureOctaveBin == 3 &&
                   mappingFeatureBudgetAcceptedOctave2p >= 100)
                {
                    bBudgetReject = true;
                    bRejectOctave2p = true;
                }

                if(candidate.mappingFeatureGridBin == 1 &&
                   mappingFeatureBudgetAcceptedEdgeGrid >= 70)
                {
                    bBudgetReject = true;
                    bRejectEdgeGrid = true;
                }

                if(!candidate.bRepeatabilitySupported &&
                   (mappingFeatureRepeatabilityFallbackAccept >= kMappingFeatureSingleSupportGlobalCap ||
                    singleSupportGroupAccepted[groupIdx] >= kMappingFeatureSingleSupportGroupCap))
                {
                    mappingFeatureRepeatabilityShadowRejectSingleSupport++;
                }

                if(bBudgetReject)
                {
                    mappingFeatureBudgetReject++;
                    if(bRejectGrid)
                        mappingFeatureBudgetRejectGrid++;
                    if(bRejectLowResponse)
                        mappingFeatureBudgetRejectLowResponse++;
                    if(bRejectOctave2p)
                        mappingFeatureBudgetRejectOctave2p++;
                    if(bRejectEdgeGrid)
                        mappingFeatureBudgetRejectEdgeGrid++;
                    continue;
                }

                mappingFeatureBudgetAccept++;
                mappingFeatureRepeatabilityAccept++;
                if(candidate.mappingFeatureBudgetCell >= 0 &&
                   candidate.mappingFeatureBudgetCell < kMappingFeatureBudgetGridCells)
                {
                    mappingFeatureBudgetCellCounts[candidate.mappingFeatureBudgetCell]++;
                }
                if(candidate.mappingFeatureResponseBin == 1)
                    mappingFeatureBudgetAcceptedLowResponse++;
                if(candidate.mappingFeatureOctaveBin == 3)
                    mappingFeatureBudgetAcceptedOctave2p++;
                if(candidate.mappingFeatureGridBin == 1)
                    mappingFeatureBudgetAcceptedEdgeGrid++;
                if(!candidate.bRepeatabilitySupported)
                {
                    mappingFeatureRepeatabilityFallbackAccept++;
                    singleSupportGroupAccepted[groupIdx]++;
                }

                if(bUsePendingMappingTracklets && candidate.bCloseRayCandidate)
                {
                    updateOrCreatePendingTracklet(candidate);
                    continue;
                }

                if(!createMapPointFromCandidate(candidate))
                {
                    mappingFeatureBudgetAccept--;
                    mappingFeatureRepeatabilityAccept--;
                    if(!candidate.bRepeatabilitySupported)
                        mappingFeatureRepeatabilityFallbackAccept--;
                    mappingFeatureBudgetReject++;
                    mappingFeatureBudgetRejectDuplicate++;
                }
            }
        }
        vPendingMappingFeatureCandidates.clear();
    };

    auto processProfileFallbackCandidates = [&]()
    {
        if(vProfileFallbackCandidates.empty())
            return;

        std::sort(vProfileFallbackCandidates.begin(),
                  vProfileFallbackCandidates.end(),
                  [](const MappingTriangulationCandidate& a,
                     const MappingTriangulationCandidate& b)
                  {
                      if(a.qualityScore != b.qualityScore)
                          return a.qualityScore > b.qualityScore;
                      if(a.reprojChi2Max != b.reprojChi2Max)
                          return a.reprojChi2Max < b.reprojChi2Max;
                      if(a.cosParallaxRays != b.cosParallaxRays)
                          return a.cosParallaxRays < b.cosParallaxRays;
                      if(a.response != b.response)
                          return a.response > b.response;
                      if(a.octave != b.octave)
                          return a.octave < b.octave;
                      if(a.idx1 != b.idx1)
                          return a.idx1 < b.idx1;
                      return a.idx2 < b.idx2;
                  });

        int fallbackCreatedThisPass = 0;
        for(const MappingTriangulationCandidate& candidate : vProfileFallbackCandidates)
        {
            if(closeCreated >= kXFeatMappingProfileCloseCreatedFloor ||
               fallbackCreatedThisPass >= kXFeatMappingProfileFallbackCreateCap)
            {
                break;
            }

            if(createMapPointFromCandidate(candidate))
            {
                closeProfileFallbackCreated++;
                fallbackCreatedThisPass++;
            }
        }
        vProfileFallbackCandidates.clear();
    };

    auto processCloseMidPairSelectionCandidates = [&]()
    {
        if(vCloseMidPairSelectionCandidates.empty())
            return;

        std::set<size_t> uniqueCurrentKeypoints;
        for(const MappingTriangulationCandidate& candidate : vCloseMidPairSelectionCandidates)
            uniqueCurrentKeypoints.insert(candidate.idx1);
        closePairSelectionUniqueCurrent += static_cast<int>(uniqueCurrentKeypoints.size());

        std::sort(vCloseMidPairSelectionCandidates.begin(),
                  vCloseMidPairSelectionCandidates.end(),
                  [&](const MappingTriangulationCandidate& a,
                      const MappingTriangulationCandidate& b)
                  {
                      const float scoreA = computeCloseMidPairSelectionScore(a);
                      const float scoreB = computeCloseMidPairSelectionScore(b);
                      if(scoreA != scoreB)
                          return scoreA > scoreB;
                      if(a.reprojChi2Max != b.reprojChi2Max)
                          return a.reprojChi2Max < b.reprojChi2Max;
                      if(a.cosParallaxRays != b.cosParallaxRays)
                          return a.cosParallaxRays < b.cosParallaxRays;
                      if(a.scaleLogResidual != b.scaleLogResidual)
                          return a.scaleLogResidual < b.scaleLogResidual;
                      if(a.response != b.response)
                          return a.response > b.response;
                      if(a.octave != b.octave)
                          return a.octave < b.octave;
                      if(a.idx1 != b.idx1)
                          return a.idx1 < b.idx1;
                      return a.idx2 < b.idx2;
                  });

        std::unordered_set<size_t> selectedCurrentKeypoints;
        selectedCurrentKeypoints.reserve(vCloseMidPairSelectionCandidates.size());
        for(const MappingTriangulationCandidate& candidate : vCloseMidPairSelectionCandidates)
        {
            if(selectedCurrentKeypoints.find(candidate.idx1) != selectedCurrentKeypoints.end())
            {
                closePairSelectionDuplicateReject++;
                continue;
            }

            if(!isKeyFrameSlotFree(mpCurrentKeyFrame, candidate.idx1) ||
               !isKeyFrameSlotFree(candidate.pKF2, candidate.idx2))
            {
                closePairSelectionDuplicateReject++;
                continue;
            }

            selectedCurrentKeypoints.insert(candidate.idx1);
            if(!createMapPointFromCandidate(candidate))
            {
                selectedCurrentKeypoints.erase(candidate.idx1);
                closePairSelectionDuplicateReject++;
                continue;
            }

            closePairSelectionCreated++;
            if(candidate.bCloseRay5To10)
                closePairSelection5To10Created++;

            const int rankBin = TriangInputNeighborRankBin(
                candidate.triangInputPairContext.neighborRank);
            if(rankBin == 1)
                closePairSelectionBestTop1++;
            else if(rankBin == 2)
                closePairSelectionBestTop3++;
            else if(rankBin == 3)
                closePairSelectionBestTop10++;
            else if(rankBin == 4)
                closePairSelectionBestLate++;
        }
        vCloseMidPairSelectionCandidates.clear();
    };

    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
        {
            processCloseMidPairSelectionCandidates();
            processPendingMappingFeatureCandidates();
            processProfileFallbackCandidates();
            return;
        }

        KeyFrame* pKF2 = vpNeighKFs[i];

        GeometricCamera* pCamera1 = mpCurrentKeyFrame->mpCamera, *pCamera2 = pKF2->mpCamera;

        // Check first that baseline is not too short
        Eigen::Vector3f Ow2 = pKF2->GetCameraCenter();
        Eigen::Vector3f vBaseline = Ow2-Ow1;
        const float baseline = vBaseline.norm();
        const float medianDepthKF2ForDiag = pKF2->ComputeSceneMedianDepth(2);
        const float ratioBaselineDepthForDiag =
            medianDepthKF2ForDiag > 0.0f ? baseline / medianDepthKF2ForDiag
                                         : std::numeric_limits<float>::quiet_NaN();
        currentTriangInputPairContext.neighborRank = static_cast<int>(i);
        currentTriangInputPairContext.covisWeight = mpCurrentKeyFrame->GetWeight(pKF2);
        currentTriangInputPairContext.baseline = baseline;
        currentTriangInputPairContext.baselineDepthRatio = ratioBaselineDepthForDiag;

        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)
                continue;
        }
        else
        {
            const float medianDepthKF2 = medianDepthKF2ForDiag;
            const float ratioBaselineDepth = ratioBaselineDepthForDiag;

            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        bool bCoarse = mbInertial && mpTracker->mState==Tracking::RECENTLY_LOST && mpCurrentKeyFrame->GetMap()->GetIniertialBA2();
        MapPoint::TriangulationMatchBackend triangBackend =
            bUseORB ? MapPoint::kTriangBackendORB : MapPoint::kTriangBackendXFeatBF;

        if(bUseORB)
        {
            matcherORB.SearchForTriangulation(mpCurrentKeyFrame,pKF2,vMatchedIndices,false,bCoarse);
        }
        else
        {
            bool bUsedLightGlueTriang = false;
            if(UseXFeatLightGlueTriangulation())
            {
                try
                {
                    static XFeatLighterGlueMatcher lightGlueMatcher;
                    const float lgConf = GetXFeatLightGlueTriangulationConfidence();
                    const int nLightGlueMatches = lightGlueMatcher.SearchForTriangulation(mpCurrentKeyFrame,
                                                                                          pKF2,
                                                                                          vMatchedIndices,
                                                                                          false,
                                                                                          bCoarse,
                                                                                          lgConf);
                    bUsedLightGlueTriang =
                        nLightGlueMatches >= GetXFeatLightGlueTriangulationFallbackMinMatches();
                    if(bUsedLightGlueTriang)
                        triangBackend = MapPoint::kTriangBackendLightGlue;
                    if(bPrintTriangDiag)
                    {
                        const XFeatLighterGlueMatcher::Stats& lgStats = lightGlueMatcher.LastStats();
                        std::cout << "[LocalMapping][LightGlueTriang]"
                                  << " kf1=" << mpCurrentKeyFrame->mnId
                                  << " kf2=" << pKF2->mnId
                                  << " raw_matches=" << lgStats.lg_raw_matches
                                  << " mp_valid=" << lgStats.mp_valid
                                  << " one_to_one=" << lgStats.one_to_one
                                  << " used=" << (bUsedLightGlueTriang ? 1 : 0)
                                  << " fallback_to_xfeat=" << (bUsedLightGlueTriang ? 0 : 1)
                                  << std::endl;
                    }
                }
                catch(const std::exception& e)
                {
                    vMatchedIndices.clear();
                    if(IsLocalMappingEnvFlagEnabled("XFEAT_DEBUG") ||
                       IsLocalMappingEnvFlagEnabled("XFEAT_DEBUG_MATCHER"))
                    {
                        std::cerr << "[LocalMapping][LightGlueTriang] failed, falling back to XFeatMatcher: "
                                  << e.what() << std::endl;
                    }
                }
            }

            if(!bUsedLightGlueTriang)
                matcherXFeat.SearchForTriangulation(mpCurrentKeyFrame,pKF2,vMatchedIndices,false,bCoarse);
        }

        Sophus::SE3<float> sophTcw2 = pKF2->GetPose();
        Eigen::Matrix<float,3,4> eigTcw2 = sophTcw2.matrix3x4();
        Eigen::Matrix<float,3,3> Rcw2 = eigTcw2.block<3,3>(0,0);
        Eigen::Matrix<float,3,3> Rwc2 = Rcw2.transpose();
        Eigen::Vector3f tcw2 = sophTcw2.translation();

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = (mpCurrentKeyFrame -> NLeft == -1) ? mpCurrentKeyFrame->mvKeysUn[idx1]
                                                                         : (idx1 < mpCurrentKeyFrame -> NLeft) ? mpCurrentKeyFrame -> mvKeys[idx1]
                                                                                                               : mpCurrentKeyFrame -> mvKeysRight[idx1 - mpCurrentKeyFrame -> NLeft];
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = (!mpCurrentKeyFrame->mpCamera2 && kp1_ur>=0);
            const bool bRight1 = (mpCurrentKeyFrame -> NLeft == -1 || idx1 < mpCurrentKeyFrame -> NLeft) ? false
                                                                                                         : true;

            const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                            : (idx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[idx2]
                                                                                     : pKF2 -> mvKeysRight[idx2 - pKF2 -> NLeft];

            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = (!pKF2->mpCamera2 && kp2_ur>=0);
            const bool bRight2 = (pKF2 -> NLeft == -1 || idx2 < pKF2 -> NLeft) ? false
                                                                               : true;

            if(mpCurrentKeyFrame->mpCamera2 && pKF2->mpCamera2){
                if(bRight1 && bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetRightPose();
                    Ow1 = mpCurrentKeyFrame->GetRightCameraCenter();

                    sophTcw2 = pKF2->GetRightPose();
                    Ow2 = pKF2->GetRightCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera2;
                    pCamera2 = pKF2->mpCamera2;
                }
                else if(bRight1 && !bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetRightPose();
                    Ow1 = mpCurrentKeyFrame->GetRightCameraCenter();

                    sophTcw2 = pKF2->GetPose();
                    Ow2 = pKF2->GetCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera2;
                    pCamera2 = pKF2->mpCamera;
                }
                else if(!bRight1 && bRight2){
                    sophTcw1 = mpCurrentKeyFrame->GetPose();
                    Ow1 = mpCurrentKeyFrame->GetCameraCenter();

                    sophTcw2 = pKF2->GetRightPose();
                    Ow2 = pKF2->GetRightCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera;
                    pCamera2 = pKF2->mpCamera2;
                }
                else{
                    sophTcw1 = mpCurrentKeyFrame->GetPose();
                    Ow1 = mpCurrentKeyFrame->GetCameraCenter();

                    sophTcw2 = pKF2->GetPose();
                    Ow2 = pKF2->GetCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera;
                    pCamera2 = pKF2->mpCamera;
                }
                eigTcw1 = sophTcw1.matrix3x4();
                Rcw1 = eigTcw1.block<3,3>(0,0);
                Rwc1 = Rcw1.transpose();
                tcw1 = sophTcw1.translation();

                eigTcw2 = sophTcw2.matrix3x4();
                Rcw2 = eigTcw2.block<3,3>(0,0);
                Rwc2 = Rcw2.transpose();
                tcw2 = sophTcw2.translation();
            }

            // Check parallax between rays
            Eigen::Vector3f xn1 = pCamera1->unprojectEig(kp1.pt);
            Eigen::Vector3f xn2 = pCamera2->unprojectEig(kp2.pt);

            Eigen::Vector3f ray1 = Rwc1 * xn1;
            Eigen::Vector3f ray2 = Rwc2 * xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(ray1.norm() * ray2.norm());

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            if (bStereo1 || bStereo2) totalStereoPts++;
            
            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            Eigen::Vector3f x3D;

            bool goodProj = false;
            bool bPointStereo = false;
            float stereoFallbackDepth = -1.0f;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 ||
                                                                          (cosParallaxRays<0.9996 && mbInertial) || (cosParallaxRays<0.9998 && !mbInertial)))
            {
                goodProj = GeometricTools::Triangulate(xn1, xn2, eigTcw1, eigTcw2, x3D);
                if(!goodProj)
                    continue;
            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                countStereoAttempt++;
                bPointStereo = true;
                stereoFallbackDepth = mpCurrentKeyFrame->mvDepth[idx1];
                goodProj = mpCurrentKeyFrame->UnprojectStereo(idx1, x3D);
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                countStereoAttempt++;
                bPointStereo = true;
                stereoFallbackDepth = pKF2->mvDepth[idx2];
                goodProj = pKF2->UnprojectStereo(idx2, x3D);
            }
            else
            {
                continue; //No stereo and very low parallax
            }

            if(goodProj && bPointStereo)
                countStereoGoodProj++;

            if(!goodProj)
                continue;

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3D) + tcw1(2);
            if(z1<=0)
            {
                if(bPointStereo &&
                   std::isfinite(stereoFallbackDepth) &&
                   stereoFallbackDepth > 0.0f &&
                   stereoFallbackDepth <= 10.0f)
                {
                    accumulateTriangInputFailureDiag(pKF2,
                                                     kp1,
                                                     kp2,
                                                     triangBackend,
                                                     stereoFallbackDepth,
                                                     TriangInputTerminalEvent::CheiralityReject,
                                                     std::numeric_limits<float>::quiet_NaN(),
                                                     std::numeric_limits<float>::quiet_NaN(),
                                                     false,
                                                     false,
                                                     cosParallaxRays,
                                                     std::numeric_limits<float>::quiet_NaN());
                }
                continue;
            }

            float z2 = Rcw2.row(2).dot(x3D) + tcw2(2);
            if(z2<=0)
            {
                if(bPointStereo &&
                   std::isfinite(stereoFallbackDepth) &&
                   stereoFallbackDepth > 0.0f &&
                   stereoFallbackDepth <= 10.0f)
                {
                    accumulateTriangInputFailureDiag(pKF2,
                                                     kp1,
                                                     kp2,
                                                     triangBackend,
                                                     stereoFallbackDepth,
                                                     TriangInputTerminalEvent::CheiralityReject,
                                                     std::numeric_limits<float>::quiet_NaN(),
                                                     std::numeric_limits<float>::quiet_NaN(),
                                                     false,
                                                     false,
                                                     cosParallaxRays,
                                                     std::numeric_limits<float>::quiet_NaN());
                }
                continue;
            }

            const float triangDepth = std::min(z1, z2);
            const bool bCloseRay0To5 = !bPointStereo && triangDepth <= 5.0f;
            const bool bCloseRay5To10 = !bPointStereo && triangDepth > 5.0f &&
                                        triangDepth <= std::min(10.0f, mpCurrentKeyFrame->mThDepth);
            const bool bCloseCandidate =
                bPointStereo ?
                    (std::isfinite(stereoFallbackDepth) && stereoFallbackDepth > 0.0f &&
                     stereoFallbackDepth <= mpCurrentKeyFrame->mThDepth) :
                    (triangDepth <= mpCurrentKeyFrame->mThDepth);
            const bool bCloseRayCandidate = bCloseCandidate && !bPointStereo;
            if(bCloseCandidate)
            {
                closeCandidates++;
                if(bPointStereo)
                {
                    closeStereoFallbackCandidates++;
                    closeStereoFallbackReject++;
                    accumulateTriangInputFailureDiag(pKF2,
                                                     kp1,
                                                     kp2,
                                                     triangBackend,
                                                     stereoFallbackDepth,
                                                     TriangInputTerminalEvent::StereoFallbackReject,
                                                     std::numeric_limits<float>::quiet_NaN(),
                                                     std::numeric_limits<float>::quiet_NaN(),
                                                     false,
                                                     false,
                                                     cosParallaxRays,
                                                     std::numeric_limits<float>::quiet_NaN());
                    continue;
                }

                closeRayCandidates++;
                if(bCloseRay0To5)
                    close0To5Candidates++;
                if(bCloseRay5To10)
                    close5To10Candidates++;

                const float closeParallaxThreshold = 0.9994f;
                if(cosParallaxRays >= closeParallaxThreshold)
                {
                    closeParallaxReject++;
                    if(bCloseRay0To5)
                        close0To5ParallaxReject++;
                    if(bCloseRay5To10)
                        close5To10ParallaxReject++;
                    accumulateTriangInputFailureDiag(pKF2,
                                                     kp1,
                                                     kp2,
                                                     triangBackend,
                                                     triangDepth,
                                                     TriangInputTerminalEvent::ParallaxReject,
                                                     std::numeric_limits<float>::quiet_NaN(),
                                                     std::numeric_limits<float>::quiet_NaN(),
                                                     false,
                                                     false,
                                                     cosParallaxRays,
                                                     std::numeric_limits<float>::quiet_NaN());
                    continue;
                }
            }

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3D)+tcw1(0);
            const float y1 = Rcw1.row(1).dot(x3D)+tcw1(1);
            const float invz1 = 1.0/z1;
            const float monoReprojFactor = bCloseRayCandidate ? 4.5f : 5.991f;
            const float stereoReprojFactor = bCloseRayCandidate ? 6.0f : 7.8f;
            float reprojChi2First = std::numeric_limits<float>::quiet_NaN();
            float reprojChi2Second = std::numeric_limits<float>::quiet_NaN();
            bool bReprojRejectFirst = false;
            bool bReprojRejectSecond = false;

            if(!bStereo1)
            {
                cv::Point2f uv1 = pCamera1->project(cv::Point3f(x1,y1,z1));
                float errX1 = uv1.x - kp1.pt.x;
                float errY1 = uv1.y - kp1.pt.y;
                reprojChi2First = (errX1*errX1+errY1*errY1) / sigmaSquare1;

                if(reprojChi2First > monoReprojFactor)
                    bReprojRejectFirst = true;

            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                reprojChi2First = (errX1*errX1+errY1*errY1+errX1_r*errX1_r) / sigmaSquare1;
                if(reprojChi2First > stereoReprojFactor)
                    bReprojRejectFirst = true;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3D)+tcw2(0);
            const float y2 = Rcw2.row(1).dot(x3D)+tcw2(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                cv::Point2f uv2 = pCamera2->project(cv::Point3f(x2,y2,z2));
                float errX2 = uv2.x - kp2.pt.x;
                float errY2 = uv2.y - kp2.pt.y;
                reprojChi2Second = (errX2*errX2+errY2*errY2) / sigmaSquare2;
                if(reprojChi2Second > monoReprojFactor)
                    bReprojRejectSecond = true;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                reprojChi2Second = (errX2*errX2+errY2*errY2+errX2_r*errX2_r) / sigmaSquare2;
                if(reprojChi2Second > stereoReprojFactor)
                    bReprojRejectSecond = true;
            }

            if(bReprojRejectFirst || bReprojRejectSecond)
            {
                if(bCloseRayCandidate)
                {
                    closeReprojReject++;
                    accumulateTriangInputFailureDiag(pKF2,
                                                     kp1,
                                                     kp2,
                                                     triangBackend,
                                                     triangDepth,
                                                     TriangInputTerminalEvent::ReprojectionReject,
                                                     reprojChi2First,
                                                     reprojChi2Second,
                                                     bReprojRejectFirst,
                                                     bReprojRejectSecond,
                                                     cosParallaxRays,
                                                     std::numeric_limits<float>::quiet_NaN());
                }
                continue;
            }

            //Check scale consistency
            Eigen::Vector3f normal1 = x3D - Ow1;
            float dist1 = normal1.norm();

            Eigen::Vector3f normal2 = x3D - Ow2;
            float dist2 = normal2.norm();

            if(dist1==0 || dist2==0)
                continue;

            if(mbFarPoints && (dist1>=mThFarPoints||dist2>=mThFarPoints)) // MODIFICATION
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];
            const float activeRatioFactor = bCloseRayCandidate ? ratioFactorClose : ratioFactor;
            const float ratioScale = std::max(1e-6f, ratioDist / std::max(1e-6f, ratioOctave));
            const float scaleLogResidual = std::abs(std::log(ratioScale));
            const float scaleLogMargin = std::log(std::max(1.000001f, activeRatioFactor)) - scaleLogResidual;

            if(ratioDist*activeRatioFactor<ratioOctave || ratioDist>ratioOctave*activeRatioFactor)
            {
                if(bCloseRayCandidate)
                {
                    closeScaleReject++;
                    accumulateTriangInputFailureDiag(pKF2,
                                                     kp1,
                                                     kp2,
                                                     triangBackend,
                                                     triangDepth,
                                                     TriangInputTerminalEvent::ScaleReject,
                                                     reprojChi2First,
                                                     reprojChi2Second,
                                                     false,
                                                     false,
                                                     cosParallaxRays,
                                                     scaleLogResidual);
                }
                continue;
            }

            int mappingFeatureGridX = -1;
            int mappingFeatureGridY = -1;
            KeyFrameFeatureGridPosition(mpCurrentKeyFrame, kp1, mappingFeatureGridX, mappingFeatureGridY);
            const int mappingFeatureGridBin =
                MappingFeatureGridBin(mappingFeatureGridX, mappingFeatureGridY);
            const int mappingFeatureOctaveBin = MappingFeatureOctaveBin(kp1.octave);
            const int mappingFeatureResponseBin = MappingFeatureResponseBin(kp1.response);
            const int mappingFeatureStereoQualityBin =
                MappingFeatureStereoQualityBin(Frame::kStereoBirthInvalid);
            const int mappingFeatureCombinedBin =
                MappingFeatureCombinedBin(mappingFeatureGridBin,
                                          mappingFeatureOctaveBin,
                                          mappingFeatureResponseBin,
                                          mappingFeatureStereoQualityBin);
            const int mappingFeatureBudgetCell =
                MappingFeatureBudgetCellIndex(mappingFeatureGridX, mappingFeatureGridY);
            const bool bUseMappingFeatureBudget =
                kEnableMappingFeatureBudgetBehavior && !bUseORB && bCloseRayCandidate;
            const bool bUseMappingProfile =
                kEnableXFeatMappingProfileBehavior && !bUseORB && bCloseRayCandidate;
            const bool bCurrentProfileEligible =
                !bUseMappingProfile || IsXFeatMappingProfileEligible(mpCurrentKeyFrame, static_cast<size_t>(idx1));
            const bool bNeighborProfileEligible =
                !bUseMappingProfile || IsXFeatMappingProfileEligible(pKF2, static_cast<size_t>(idx2));
            MappingTriangulationCandidate candidate;
            candidate.pKF1 = mpCurrentKeyFrame;
            candidate.pKF2 = pKF2;
            candidate.idx1 = static_cast<size_t>(idx1);
            candidate.idx2 = static_cast<size_t>(idx2);
            candidate.diagKp1 = kp1;
            candidate.diagKp2 = kp2;
            candidate.x3D = x3D;
            candidate.triangDepth = triangDepth;
            candidate.cosParallaxRays = cosParallaxRays;
            candidate.reprojChi2First = reprojChi2First;
            candidate.reprojChi2Second = reprojChi2Second;
            candidate.reprojChi2Max = std::max(reprojChi2First, reprojChi2Second);
            candidate.reprojChi2Mean = 0.5f * (reprojChi2First + reprojChi2Second);
            candidate.scaleLogResidual = scaleLogResidual;
            candidate.scaleLogMargin = scaleLogMargin;
            candidate.triangBackend = triangBackend;
            candidate.mappingFeatureGridX = mappingFeatureGridX;
            candidate.mappingFeatureGridY = mappingFeatureGridY;
            candidate.mappingFeatureGridBin = mappingFeatureGridBin;
            candidate.mappingFeatureOctaveBin = mappingFeatureOctaveBin;
            candidate.mappingFeatureResponseBin = mappingFeatureResponseBin;
            candidate.mappingFeatureStereoQualityBin = mappingFeatureStereoQualityBin;
            candidate.mappingFeatureCombinedBin = mappingFeatureCombinedBin;
            candidate.mappingFeatureBudgetCell = mappingFeatureBudgetCell;
            candidate.octave = kp1.octave;
            candidate.response = kp1.response;
            candidate.bPointStereo = bPointStereo;
            candidate.bCloseRayCandidate = bCloseRayCandidate;
            candidate.bCloseRay0To5 = bCloseRay0To5;
            candidate.bCloseRay5To10 = bCloseRay5To10;
            candidate.bProfileEligible = bCurrentProfileEligible && bNeighborProfileEligible;
            candidate.qualityScore = computeMappingCandidateQualityScore(candidate);
            candidate.triangInputPairContext = currentTriangInputPairContext;

            if(bUseMappingProfile)
            {
                if(candidate.bProfileEligible)
                {
                    closeProfileEligibleCandidates++;
                    if(createMapPointFromCandidate(candidate))
                        closeProfileEligibleCreated++;
                }
                else
                {
                    closeProfileFallbackCandidates++;
                    vProfileFallbackCandidates.push_back(candidate);
                }
                continue;
            }

            if(bUseMappingFeatureBudget)
            {
                mappingFeatureBudgetCandidate++;
                vPendingMappingFeatureCandidates.push_back(candidate);
                continue;
            }

            if(kEnableXFeatCloseMidPairSelectionBehavior && !bUseORB && bCloseRayCandidate)
            {
                closePairSelectionCandidate++;
                if(bCloseRay5To10)
                    closePairSelection5To10Candidate++;
                vCloseMidPairSelectionCandidates.push_back(candidate);
                continue;
            }

            createMapPointFromCandidate(candidate);
        }
    }

    processCloseMidPairSelectionCandidates();
    processPendingMappingFeatureCandidates();
    processProfileFallbackCandidates();

    int mappingTrackletAlive = 0;
    int mappingTrackletObsSum = 0;
    int mappingTrackletPassSum = 0;
    int mappingTrackletAgeSum = 0;
    if(mpCurrentKeyFrame)
    {
        for(const PendingMappingTracklet& tracklet : mvPendingMappingTracklets)
        {
            if(tracklet.bad)
                continue;
            mappingTrackletAlive++;
            mappingTrackletObsSum += static_cast<int>(tracklet.observations.size());
            mappingTrackletPassSum += tracklet.updatePasses;
            if(mpCurrentKeyFrame->mnId >= tracklet.firstKFid)
            {
                mappingTrackletAgeSum +=
                    static_cast<int>(mpCurrentKeyFrame->mnId - tracklet.firstKFid);
            }
        }
    }
    const double mappingTrackletObsAvg = mappingTrackletAlive > 0
        ? static_cast<double>(mappingTrackletObsSum) / mappingTrackletAlive
        : 0.0;
    const double mappingTrackletPassAvg = mappingTrackletAlive > 0
        ? static_cast<double>(mappingTrackletPassSum) / mappingTrackletAlive
        : 0.0;
    const double mappingTrackletAgeAvg = mappingTrackletAlive > 0
        ? static_cast<double>(mappingTrackletAgeSum) / mappingTrackletAlive
        : 0.0;

    if(bPrintTriangDiag)
    {
        const std::set<KeyFrame*> connectedKFs = mpCurrentKeyFrame->GetConnectedKeyFrames();
        const std::vector<KeyFrame*> covisibleKFs = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();
        const int trackedObs2 = mpCurrentKeyFrame->TrackedMapPoints(2);
        const int trackedObs3 = mpCurrentKeyFrame->TrackedMapPoints(3);
        const float medianDepth = mpCurrentKeyFrame->ComputeSceneMedianDepth(2);

        std::cout << "[LocalMapping][TriangCreate]"
                  << " kf_id=" << mpCurrentKeyFrame->mnId
                  << " close_candidates=" << closeCandidates
                  << " close_ray_candidates=" << closeRayCandidates
                  << " close_stereo_fallback_candidates=" << closeStereoFallbackCandidates
                  << " close_stereo_fallback_reject=" << closeStereoFallbackReject
                  << " close_parallax_reject=" << closeParallaxReject
                  << " close_reproj_reject=" << closeReprojReject
                  << " close_scale_reject=" << closeScaleReject
                  << " close_created=" << closeCreated
                  << " close_0_5_candidates=" << close0To5Candidates
                  << " close_0_5_parallax_reject=" << close0To5ParallaxReject
                  << " close_0_5_created=" << close0To5Created
                  << " close_5_10_candidates=" << close5To10Candidates
                  << " close_5_10_parallax_reject=" << close5To10ParallaxReject
                  << " close_5_10_created=" << close5To10Created
                  << " close_profile_eligible_candidates=" << closeProfileEligibleCandidates
                  << " close_profile_eligible_created=" << closeProfileEligibleCreated
                  << " close_profile_fallback_candidates=" << closeProfileFallbackCandidates
                  << " close_profile_fallback_created=" << closeProfileFallbackCreated
                  << " close_pair_select_active=" << (kEnableXFeatCloseMidPairSelectionBehavior && !bUseORB ? 1 : 0)
                  << " close_pair_select_candidate=" << closePairSelectionCandidate
                  << " close_pair_select_unique_current=" << closePairSelectionUniqueCurrent
                  << " close_pair_select_created=" << closePairSelectionCreated
                  << " close_pair_select_duplicate_reject=" << closePairSelectionDuplicateReject
                  << " close_pair_select_best_top1=" << closePairSelectionBestTop1
                  << " close_pair_select_best_top3=" << closePairSelectionBestTop3
                  << " close_pair_select_best_top10=" << closePairSelectionBestTop10
                  << " close_pair_select_best_late=" << closePairSelectionBestLate
                  << " close_pair_select_5_10_candidate=" << closePairSelection5To10Candidate
                  << " close_pair_select_5_10_created=" << closePairSelection5To10Created
                  << " mapping_feature_budget_candidate=" << mappingFeatureBudgetCandidate
                  << " mapping_feature_budget_unique_candidates=" << mappingFeatureBudgetUniqueCandidates
                  << " mapping_feature_budget_accept=" << mappingFeatureBudgetAccept
                  << " mapping_feature_budget_reject=" << mappingFeatureBudgetReject
                  << " mapping_feature_budget_reject_grid=" << mappingFeatureBudgetRejectGrid
                  << " mapping_feature_budget_reject_low_response=" << mappingFeatureBudgetRejectLowResponse
                  << " mapping_feature_budget_reject_octave2p=" << mappingFeatureBudgetRejectOctave2p
                  << " mapping_feature_budget_reject_edge_grid=" << mappingFeatureBudgetRejectEdgeGrid
                  << " mapping_feature_budget_reject_duplicate=" << mappingFeatureBudgetRejectDuplicate
                  << " mapping_feature_budget_reject_keypoint_duplicate=" << mappingFeatureBudgetRejectKeypointDuplicate
                  << " mapping_feature_budget_accepted_low_response=" << mappingFeatureBudgetAcceptedLowResponse
                  << " mapping_feature_budget_accepted_octave2p=" << mappingFeatureBudgetAcceptedOctave2p
                  << " mapping_feature_budget_accepted_edge_grid=" << mappingFeatureBudgetAcceptedEdgeGrid
                  << " mapping_feature_repeatability_candidate=" << mappingFeatureRepeatabilityCandidate
                  << " mapping_feature_repeatability_repeatable_candidate=" << mappingFeatureRepeatabilityRepeatableCandidate
                  << " mapping_feature_repeatability_single_support_candidate=" << mappingFeatureRepeatabilitySingleSupportCandidate
                  << " mapping_feature_repeatability_accept=" << mappingFeatureRepeatabilityAccept
                  << " mapping_feature_repeatability_fallback_accept=" << mappingFeatureRepeatabilityFallbackAccept
                  << " mapping_feature_repeatability_reject_single_support=" << mappingFeatureRepeatabilityRejectSingleSupport
                  << " mapping_feature_repeatability_shadow_reject_single_support="
                  << mappingFeatureRepeatabilityShadowRejectSingleSupport
                  << " mapping_probation_created=" << mappingProbationCreated
                  << " mapping_probation_created_0_5=" << mappingProbationCreated0To5
                  << " mapping_probation_created_5_10=" << mappingProbationCreated5To10
                  << " mapping_tracklet_enabled=" << (bUsePendingMappingTracklets ? 1 : 0)
                  << " mapping_tracklet_candidate=" << mappingTrackletCandidate
                  << " mapping_tracklet_created=" << mappingTrackletCreated
                  << " mapping_tracklet_updated=" << mappingTrackletUpdated
                  << " mapping_tracklet_promoted=" << mappingTrackletPromoted
                  << " mapping_tracklet_expired=" << mappingTrackletExpired
                  << " mapping_tracklet_conflict=" << mappingTrackletConflict
                  << " mapping_tracklet_promotion_failed=" << mappingTrackletPromotionFailed
                  << " mapping_tracklet_alive=" << mappingTrackletAlive
                  << std::endl;

        if(bCollectTriangInputFailureDiag)
        {
            for(int bin = 0; bin < kTriangInputBackendBins; ++bin)
            {
                PrintTriangInputFailureStats(mpCurrentKeyFrame->mnId,
                                             "backend_bin",
                                             TriangInputBackendBinName(bin),
                                             triangInputBackendStats[bin]);
            }
            for(int bin = 0; bin < kTriangInputDepthBins; ++bin)
            {
                PrintTriangInputFailureStats(mpCurrentKeyFrame->mnId,
                                             "depth_bin",
                                             TriangInputCloseDepthBinName(bin),
                                             triangInputDepthStats[bin]);
            }
            for(int bin = 0; bin < kTriangInputOctavePairBins; ++bin)
            {
                PrintTriangInputFailureStats(mpCurrentKeyFrame->mnId,
                                             "octave_pair_bin",
                                             TriangInputOctavePairBinName(bin),
                                             triangInputOctavePairStats[bin]);
            }
            for(int bin = 0; bin < kTriangInputCurrentOctaveBins; ++bin)
            {
                PrintTriangInputFailureStats(mpCurrentKeyFrame->mnId,
                                             "current_octave_bin",
                                             TriangInputCurrentOctaveBinName(bin),
                                             triangInputCurrentOctaveStats[bin]);
            }
            for(int bin = 0; bin < kTriangInputResponseBins; ++bin)
            {
                PrintTriangInputFailureStats(mpCurrentKeyFrame->mnId,
                                             "response_bin",
                                             MappingFeatureResponseBinName(bin),
                                             triangInputResponseStats[bin]);
            }
            for(int bin = 0; bin < kTriangInputGridBins; ++bin)
            {
                PrintTriangInputFailureStats(mpCurrentKeyFrame->mnId,
                                             "grid_bin",
                                             MappingFeatureGridBinName(bin),
                                             triangInputGridStats[bin]);
            }
            for(int bin = 0; bin < kTriangInputEdgeDistanceBins; ++bin)
            {
                PrintTriangInputFailureStats(mpCurrentKeyFrame->mnId,
                                             "edge_distance_bin",
                                             TriangInputEdgeDistanceBinName(bin),
                                             triangInputEdgeDistanceStats[bin]);
            }
            for(int bin = 0; bin < kTriangInputNeighborRankBins; ++bin)
            {
                PrintTriangInputFailureStats(mpCurrentKeyFrame->mnId,
                                             "neighbor_rank_bin",
                                             TriangInputNeighborRankBinName(bin),
                                             triangInputNeighborRankStats[bin]);
            }
            for(int bin = 0; bin < kTriangInputKfIdDeltaBins; ++bin)
            {
                PrintTriangInputFailureStats(mpCurrentKeyFrame->mnId,
                                             "kf_id_delta_bin",
                                             TriangInputKfIdDeltaBinName(bin),
                                             triangInputKfIdDeltaStats[bin]);
            }
            for(int bin = 0; bin < kTriangInputCovisWeightBins; ++bin)
            {
                PrintTriangInputFailureStats(mpCurrentKeyFrame->mnId,
                                             "covis_weight_bin",
                                             TriangInputCovisWeightBinName(bin),
                                             triangInputCovisWeightStats[bin]);
            }
            for(int bin = 0; bin < kTriangInputBaselineBins; ++bin)
            {
                PrintTriangInputFailureStats(mpCurrentKeyFrame->mnId,
                                             "baseline_bin",
                                             TriangInputBaselineBinName(bin),
                                             triangInputBaselineStats[bin]);
            }
            for(int bin = 0; bin < kTriangInputBaselineDepthRatioBins; ++bin)
            {
                PrintTriangInputFailureStats(mpCurrentKeyFrame->mnId,
                                             "baseline_depth_ratio_bin",
                                             TriangInputBaselineDepthRatioBinName(bin),
                                             triangInputBaselineDepthRatioStats[bin]);
            }
            for(int bin = 0; bin < kTriangInputKf2OctaveBins; ++bin)
            {
                PrintTriangInputFailureStats(mpCurrentKeyFrame->mnId,
                                             "kf2_octave_bin",
                                             TriangInputCurrentOctaveBinName(bin),
                                             triangInputKf2OctaveStats[bin]);
            }
            for(int bin = 0; bin < kTriangInputKf2ResponseBins; ++bin)
            {
                PrintTriangInputFailureStats(mpCurrentKeyFrame->mnId,
                                             "kf2_response_bin",
                                             MappingFeatureResponseBinName(bin),
                                             triangInputKf2ResponseStats[bin]);
            }
            for(int bin = 0; bin < kTriangInputKf2EdgeDistanceBins; ++bin)
            {
                PrintTriangInputFailureStats(mpCurrentKeyFrame->mnId,
                                             "kf2_edge_distance_bin",
                                             TriangInputEdgeDistanceBinName(bin),
                                             triangInputKf2EdgeDistanceStats[bin]);
            }
        }

        const std::ios::fmtflags oldFlags = std::cout.flags();
        const std::streamsize oldPrecision = std::cout.precision();
        std::cout << std::fixed << std::setprecision(6)
                  << "[LocalMapping][KFCreateStructure]"
                  << " kf_id=" << mpCurrentKeyFrame->mnId
                  << " neighbor_kfs=" << vpNeighKFs.size()
                  << " connected_kfs=" << connectedKFs.size()
                  << " covisible_kfs=" << covisibleKFs.size()
                  << " tracked_mps_obs2=" << trackedObs2
                  << " tracked_mps_obs3=" << trackedObs3
                  << " median_depth=" << medianDepth
                  << " created_mps_total=" << createdMpsTotal
                  << " created_mps_close=" << closeCreated
                  << " created_mps_far=" << std::max(0, createdMpsTotal - closeCreated)
                  << " created_mps_depth_invalid=" << createdMpsDepthInvalid
                  << " created_mps_depth_0_5=" << createdMpsDepth0To5
                  << " created_mps_depth_5_10=" << createdMpsDepth5To10
                  << " created_mps_depth_10_30=" << createdMpsDepth10To30
                  << " created_mps_depth_30_100=" << createdMpsDepth30To100
                  << " created_mps_depth_100p=" << createdMpsDepth100p
                  << " close_stereo_fallback_reject=" << closeStereoFallbackReject
                  << " close_parallax_reject=" << closeParallaxReject
                  << " close_reproj_reject=" << closeReprojReject
                  << " close_scale_reject=" << closeScaleReject
                  << " mapping_feature_budget_candidate=" << mappingFeatureBudgetCandidate
                  << " mapping_feature_budget_unique_candidates=" << mappingFeatureBudgetUniqueCandidates
                  << " mapping_feature_budget_reject=" << mappingFeatureBudgetReject
                  << " mapping_feature_repeatability_candidate=" << mappingFeatureRepeatabilityCandidate
                  << " mapping_feature_repeatability_repeatable_candidate=" << mappingFeatureRepeatabilityRepeatableCandidate
                  << " mapping_feature_repeatability_single_support_candidate=" << mappingFeatureRepeatabilitySingleSupportCandidate
                  << " mapping_feature_repeatability_accept=" << mappingFeatureRepeatabilityAccept
                  << " mapping_feature_repeatability_fallback_accept=" << mappingFeatureRepeatabilityFallbackAccept
                  << " mapping_feature_repeatability_reject_single_support=" << mappingFeatureRepeatabilityRejectSingleSupport
                  << " mapping_feature_repeatability_shadow_reject_single_support="
                  << mappingFeatureRepeatabilityShadowRejectSingleSupport
                  << " mapping_probation_created=" << mappingProbationCreated
                  << " mapping_probation_created_0_5=" << mappingProbationCreated0To5
                  << " mapping_probation_created_5_10=" << mappingProbationCreated5To10
                  << " mapping_tracklet_enabled=" << (bUsePendingMappingTracklets ? 1 : 0)
                  << " mapping_tracklet_candidate=" << mappingTrackletCandidate
                  << " mapping_tracklet_created=" << mappingTrackletCreated
                  << " mapping_tracklet_updated=" << mappingTrackletUpdated
                  << " mapping_tracklet_promoted=" << mappingTrackletPromoted
                  << " mapping_tracklet_expired=" << mappingTrackletExpired
                  << " mapping_tracklet_conflict=" << mappingTrackletConflict
                  << " mapping_tracklet_promotion_failed=" << mappingTrackletPromotionFailed
                  << " mapping_tracklet_alive=" << mappingTrackletAlive;
        AppendScalarDistributionStats(std::cout, "created_depth", createdTriangDepthStats, true);
        std::cout << std::endl;
        std::cout << std::fixed << std::setprecision(6)
                  << "[LocalMapping][MappingTracklet]"
                  << " kf_id=" << mpCurrentKeyFrame->mnId
                  << " frame_id=" << mpCurrentKeyFrame->mnFrameId
                  << " enabled=" << (bUsePendingMappingTracklets ? 1 : 0)
                  << " alive=" << mappingTrackletAlive
                  << " created=" << mappingTrackletCreated
                  << " updated=" << mappingTrackletUpdated
                  << " promoted=" << mappingTrackletPromoted
                  << " expired=" << mappingTrackletExpired
                  << " conflict=" << mappingTrackletConflict
                  << " promotion_failed=" << mappingTrackletPromotionFailed
                  << " obs_avg=" << mappingTrackletObsAvg
                  << " pass_avg=" << mappingTrackletPassAvg
                  << " age_avg=" << mappingTrackletAgeAvg;
        AppendScalarDistributionStats(std::cout, "promoted_depth", mappingTrackletPromotedDepthStats, true);
        std::cout << std::endl;
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrecision);

        if(IsLocalMappingFeatureDiagEnabled())
        {
            for(int bin = 0; bin < kMappingFeatureGridBins; ++bin)
            {
                PrintMappingFeatureCreateStats(mpCurrentKeyFrame->mnId,
                                               mpCurrentKeyFrame->mnFrameId,
                                               "grid_bin",
                                               MappingFeatureGridBinName(bin),
                                               mappingFeatureGridStats[bin]);
            }
            for(int bin = 0; bin < kMappingFeatureOctaveBins; ++bin)
            {
                PrintMappingFeatureCreateStats(mpCurrentKeyFrame->mnId,
                                               mpCurrentKeyFrame->mnFrameId,
                                               "octave_bin",
                                               MappingFeatureOctaveBinName(bin),
                                               mappingFeatureOctaveStats[bin]);
            }
            for(int bin = 0; bin < kMappingFeatureResponseBins; ++bin)
            {
                PrintMappingFeatureCreateStats(mpCurrentKeyFrame->mnId,
                                               mpCurrentKeyFrame->mnFrameId,
                                               "response_bin",
                                               MappingFeatureResponseBinName(bin),
                                               mappingFeatureResponseStats[bin]);
            }
            for(int bin = 0; bin < kMappingFeatureDepthBins; ++bin)
            {
                PrintMappingFeatureCreateStats(mpCurrentKeyFrame->mnId,
                                               mpCurrentKeyFrame->mnFrameId,
                                               "creation_depth_bin",
                                               LocalMappingDepthBinName(bin),
                                               mappingFeatureDepthStats[bin]);
            }
            for(int bin = 0; bin < kMappingFeatureCombinedBins; ++bin)
            {
                PrintMappingFeatureCreateStats(mpCurrentKeyFrame->mnId,
                                               mpCurrentKeyFrame->mnFrameId,
                                               "mapping_feature_bin",
                                               MappingFeatureCombinedBinName(bin),
                                               mappingFeatureCombinedStats[bin]);
            }
        }
    }
}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular)
        nn=30;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;
    }

    // Add some covisible of covisible
    // Extend to some second neighbors if abort is not requested
    for(int i=0, imax=vpTargetKFs.size(); i<imax; i++)
    {
        const vector<KeyFrame*> vpSecondNeighKFs = vpTargetKFs[i]->GetBestCovisibilityKeyFrames(20);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
            pKFi2->mnFuseTargetForKF=mpCurrentKeyFrame->mnId;
        }
        if (mbAbortBA)
            break;
    }

    // Extend to temporal neighbors
    if(mbInertial)
    {
        KeyFrame* pKFi = mpCurrentKeyFrame->mPrevKF;
        while(vpTargetKFs.size()<20 && pKFi)
        {
            if(pKFi->isBad() || pKFi->mnFuseTargetForKF==mpCurrentKeyFrame->mnId)
            {
                pKFi = pKFi->mPrevKF;
                continue;
            }
            vpTargetKFs.push_back(pKFi);
            pKFi->mnFuseTargetForKF=mpCurrentKeyFrame->mnId;
            pKFi = pKFi->mPrevKF;
        }
    }

    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;

        matcher.Fuse(pKFi,vpMapPointMatches);
        if(pKFi->NLeft != -1) matcher.Fuse(pKFi,vpMapPointMatches,true);
    }


    if (mbAbortBA)
        return;

    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);
    if(mpCurrentKeyFrame->NLeft != -1) matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates,true);


    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    const int Nd = 21;
    mpCurrentKeyFrame->UpdateBestCovisibles();
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    float redundant_th;
    if(!mbInertial)
        redundant_th = 0.9;
    else if (mbMonocular)
        redundant_th = 0.9;
    else
        redundant_th = 0.5;

    const bool bInitImu = mpAtlas->isImuInitialized();
    int count=0;

    // Compoute last KF from optimizable window:
    unsigned int last_ID;
    if (mbInertial)
    {
        int count = 0;
        KeyFrame* aux_KF = mpCurrentKeyFrame;
        while(count<Nd && aux_KF->mPrevKF)
        {
            aux_KF = aux_KF->mPrevKF;
            count++;
        }
        last_ID = aux_KF->mnId;
    }



    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        count++;
        KeyFrame* pKF = *vit;

        if((pKF->mnId==pKF->GetMap()->GetInitKFid()) || pKF->isBad())
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = (pKF -> NLeft == -1) ? pKF->mvKeysUn[i].octave
                                                                     : (i < pKF -> NLeft) ? pKF -> mvKeys[i].octave
                                                                                          : pKF -> mvKeysRight[i].octave;
                        const map<KeyFrame*, tuple<int,int>> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<KeyFrame*, tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            tuple<int,int> indexes = mit->second;
                            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
                            int scaleLeveli = -1;
                            if(pKFi -> NLeft == -1)
                                scaleLeveli = pKFi->mvKeysUn[leftIndex].octave;
                            else {
                                if (leftIndex != -1) {
                                    scaleLeveli = pKFi->mvKeys[leftIndex].octave;
                                }
                                if (rightIndex != -1) {
                                    int rightLevel = pKFi->mvKeysRight[rightIndex - pKFi->NLeft].octave;
                                    scaleLeveli = (scaleLeveli == -1 || scaleLeveli > rightLevel) ? rightLevel
                                                                                                  : scaleLeveli;
                                }
                            }

                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>thObs)
                                    break;
                            }
                        }
                        if(nObs>thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }

        if(nRedundantObservations>redundant_th*nMPs)
        {
            if (mbInertial)
            {
                if (mpAtlas->KeyFramesInMap()<=Nd)
                    continue;

                if(pKF->mnId>(mpCurrentKeyFrame->mnId-2))
                    continue;

                if(pKF->mPrevKF && pKF->mNextKF)
                {
                    const float t = pKF->mNextKF->mTimeStamp-pKF->mPrevKF->mTimeStamp;

                    if((bInitImu && (pKF->mnId<last_ID) && t<3.) || (t<0.5))
                    {
                        pKF->mNextKF->mpImuPreintegrated->MergePrevious(pKF->mpImuPreintegrated);
                        pKF->mNextKF->mPrevKF = pKF->mPrevKF;
                        pKF->mPrevKF->mNextKF = pKF->mNextKF;
                        pKF->mNextKF = NULL;
                        pKF->mPrevKF = NULL;
                        pKF->SetBadFlag();
                    }
                    else if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2() && ((pKF->GetImuPosition()-pKF->mPrevKF->GetImuPosition()).norm()<0.02) && (t<3))
                    {
                        pKF->mNextKF->mpImuPreintegrated->MergePrevious(pKF->mpImuPreintegrated);
                        pKF->mNextKF->mPrevKF = pKF->mPrevKF;
                        pKF->mPrevKF->mNextKF = pKF->mNextKF;
                        pKF->mNextKF = NULL;
                        pKF->mPrevKF = NULL;
                        pKF->SetBadFlag();
                    }
                }
            }
            else
            {
                pKF->SetBadFlag();
            }
        }
        if((count > 20 && mbAbortBA) || count>100)
        {
            break;
        }
    }
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        cout << "LM: Map reset recieved" << endl;
        mbResetRequested = true;
    }
    cout << "LM: Map reset, waiting..." << endl;

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
    cout << "LM: Map reset, Done!!!" << endl;
}

void LocalMapping::RequestResetActiveMap(Map* pMap)
{
    {
        unique_lock<mutex> lock(mMutexReset);
        cout << "LM: Active map reset recieved" << endl;
        mbResetRequestedActiveMap = true;
        mpMapToReset = pMap;
    }
    cout << "LM: Active map reset, waiting..." << endl;

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequestedActiveMap)
                break;
        }
        usleep(3000);
    }
    cout << "LM: Active map reset, Done!!!" << endl;
}

void LocalMapping::ResetIfRequested()
{
    bool executed_reset = false;
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbResetRequested)
        {
            executed_reset = true;

            cout << "LM: Reseting Atlas in Local Mapping..." << endl;
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();
            mvPendingMappingTracklets.clear();
            mPendingMappingObservationIndex.clear();
            mnNextPendingMappingTrackletId = 1;
            mbResetRequested = false;
            mbResetRequestedActiveMap = false;

            // Inertial parameters
            mTinit = 0.f;
            mbNotBA2 = true;
            mbNotBA1 = true;
            mbBadImu=false;

            mIdxInit=0;

            cout << "LM: End reseting Local Mapping..." << endl;
        }

        if(mbResetRequestedActiveMap) {
            executed_reset = true;
            cout << "LM: Reseting current map in Local Mapping..." << endl;
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();
            mvPendingMappingTracklets.clear();
            mPendingMappingObservationIndex.clear();
            mnNextPendingMappingTrackletId = 1;

            // Inertial parameters
            mTinit = 0.f;
            mbNotBA2 = true;
            mbNotBA1 = true;
            mbBadImu=false;

            mbResetRequested = false;
            mbResetRequestedActiveMap = false;
            cout << "LM: End reseting Local Mapping..." << endl;
        }
    }
    if(executed_reset)
        cout << "LM: Reset free the mutex" << endl;

}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void LocalMapping::InitializeIMU(float priorG, float priorA, bool bFIBA)
{
    if (mbResetRequested)
        return;

    float minTime;
    int nMinKF;
    if (mbMonocular)
    {
        minTime = 2.0;
        nMinKF = 10;
    }
    else
    {
        minTime = 1.0;
        nMinKF = 10;
    }


    if(mpAtlas->KeyFramesInMap()<nMinKF)
        return;

    // Retrieve all keyframe in temporal order
    list<KeyFrame*> lpKF;
    KeyFrame* pKF = mpCurrentKeyFrame;
    while(pKF->mPrevKF)
    {
        lpKF.push_front(pKF);
        pKF = pKF->mPrevKF;
    }
    lpKF.push_front(pKF);
    vector<KeyFrame*> vpKF(lpKF.begin(),lpKF.end());

    if(vpKF.size()<nMinKF)
        return;

    mFirstTs=vpKF.front()->mTimeStamp;
    if(mpCurrentKeyFrame->mTimeStamp-mFirstTs<minTime)
        return;

    bInitializing = true;

    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    const int N = vpKF.size();
    IMU::Bias b(0,0,0,0,0,0);

    // Compute and KF velocities mRwg estimation
    if (!mpCurrentKeyFrame->GetMap()->isImuInitialized())
    {
        Eigen::Matrix3f Rwg;
        Eigen::Vector3f dirG;
        dirG.setZero();
        for(vector<KeyFrame*>::iterator itKF = vpKF.begin(); itKF!=vpKF.end(); itKF++)
        {
            if (!(*itKF)->mpImuPreintegrated)
                continue;
            if (!(*itKF)->mPrevKF)
                continue;

            dirG -= (*itKF)->mPrevKF->GetImuRotation() * (*itKF)->mpImuPreintegrated->GetUpdatedDeltaVelocity();
            Eigen::Vector3f _vel = ((*itKF)->GetImuPosition() - (*itKF)->mPrevKF->GetImuPosition())/(*itKF)->mpImuPreintegrated->dT;
            (*itKF)->SetVelocity(_vel);
            (*itKF)->mPrevKF->SetVelocity(_vel);
        }

        dirG = dirG/dirG.norm();
        Eigen::Vector3f gI(0.0f, 0.0f, -1.0f);
        Eigen::Vector3f v = gI.cross(dirG);
        const float nv = v.norm();
        const float cosg = gI.dot(dirG);
        const float ang = acos(cosg);
        Eigen::Vector3f vzg = v*ang/nv;
        Rwg = Sophus::SO3f::exp(vzg).matrix();
        mRwg = Rwg.cast<double>();
        mTinit = mpCurrentKeyFrame->mTimeStamp-mFirstTs;
    }
    else
    {
        mRwg = Eigen::Matrix3d::Identity();
        mbg = mpCurrentKeyFrame->GetGyroBias().cast<double>();
        mba = mpCurrentKeyFrame->GetAccBias().cast<double>();
    }

    mScale=1.0;

    mInitTime = mpTracker->mLastFrame.mTimeStamp-vpKF.front()->mTimeStamp;

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale, mbg, mba, mbMonocular, infoInertial, false, false, priorG, priorA);

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    if (mScale<1e-1)
    {
        cout << "scale too small" << endl;
        bInitializing=false;
        return;
    }

    // Before this line we are not changing the map
    {
        unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
        if ((fabs(mScale - 1.f) > 0.00001) || !mbMonocular) {
            Sophus::SE3f Twg(mRwg.cast<float>().transpose(), Eigen::Vector3f::Zero());
            mpAtlas->GetCurrentMap()->ApplyScaledRotation(Twg, mScale, true);
            mpTracker->UpdateFrameIMU(mScale, vpKF[0]->GetImuBias(), mpCurrentKeyFrame);
        }

        // Check if initialization OK
        if (!mpAtlas->isImuInitialized())
            for (int i = 0; i < N; i++) {
                KeyFrame *pKF2 = vpKF[i];
                pKF2->bImu = true;
            }
    }

    mpTracker->UpdateFrameIMU(1.0,vpKF[0]->GetImuBias(),mpCurrentKeyFrame);
    if (!mpAtlas->isImuInitialized())
    {
        mpAtlas->SetImuInitialized();
        mpTracker->t0IMU = mpTracker->mCurrentFrame.mTimeStamp;
        mpCurrentKeyFrame->bImu = true;
    }

    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    if (bFIBA)
    {
        if (priorA!=0.f)
            Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, mpCurrentKeyFrame->mnId, NULL, true, priorG, priorA);
        else
            Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, mpCurrentKeyFrame->mnId, NULL, false);
    }

    std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();

    Verbose::PrintMess("Global Bundle Adjustment finished\nUpdating map ...", Verbose::VERBOSITY_NORMAL);

    // Get Map Mutex
    unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);

    unsigned long GBAid = mpCurrentKeyFrame->mnId;

    // Process keyframes in the queue
    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    // Correct keyframes starting at map first keyframe
    list<KeyFrame*> lpKFtoCheck(mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.begin(),mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.end());

    while(!lpKFtoCheck.empty())
    {
        KeyFrame* pKF = lpKFtoCheck.front();
        const set<KeyFrame*> sChilds = pKF->GetChilds();
        Sophus::SE3f Twc = pKF->GetPoseInverse();
        for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
        {
            KeyFrame* pChild = *sit;
            if(!pChild || pChild->isBad())
                continue;

            if(pChild->mnBAGlobalForKF!=GBAid)
            {
                Sophus::SE3f Tchildc = pChild->GetPose() * Twc;
                pChild->mTcwGBA = Tchildc * pKF->mTcwGBA;

                Sophus::SO3f Rcor = pChild->mTcwGBA.so3().inverse() * pChild->GetPose().so3();
                if(pChild->isVelocitySet()){
                    pChild->mVwbGBA = Rcor * pChild->GetVelocity();
                }
                else {
                    Verbose::PrintMess("Child velocity empty!! ", Verbose::VERBOSITY_NORMAL);
                }

                pChild->mBiasGBA = pChild->GetImuBias();
                pChild->mnBAGlobalForKF = GBAid;

            }
            lpKFtoCheck.push_back(pChild);
        }

        pKF->mTcwBefGBA = pKF->GetPose();
        pKF->SetPose(pKF->mTcwGBA);

        if(pKF->bImu)
        {
            pKF->mVwbBefGBA = pKF->GetVelocity();
            pKF->SetVelocity(pKF->mVwbGBA);
            pKF->SetNewBias(pKF->mBiasGBA);
        } else {
            cout << "KF " << pKF->mnId << " not set to inertial!! \n";
        }

        lpKFtoCheck.pop_front();
    }

    // Correct MapPoints
    const vector<MapPoint*> vpMPs = mpAtlas->GetCurrentMap()->GetAllMapPoints();

    for(size_t i=0; i<vpMPs.size(); i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        if(pMP->mnBAGlobalForKF==GBAid)
        {
            // If optimized by Global BA, just update
            pMP->SetWorldPos(pMP->mPosGBA);
        }
        else
        {
            // Update according to the correction of its reference keyframe
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

            if(pRefKF->mnBAGlobalForKF!=GBAid)
                continue;

            // Map to non-corrected camera
            Eigen::Vector3f Xc = pRefKF->mTcwBefGBA * pMP->GetWorldPos();

            // Backproject using corrected camera
            pMP->SetWorldPos(pRefKF->GetPoseInverse() * Xc);
        }
    }

    Verbose::PrintMess("Map updated!", Verbose::VERBOSITY_NORMAL);

    mnKFs=vpKF.size();
    mIdxInit++;

    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
    {
        (*lit)->SetBadFlag();
        delete *lit;
    }
    mlNewKeyFrames.clear();

    mpTracker->mState=Tracking::OK;
    bInitializing = false;

    mpCurrentKeyFrame->GetMap()->IncreaseChangeIndex();

    return;
}

void LocalMapping::ScaleRefinement()
{
    // Minimum number of keyframes to compute a solution
    // Minimum time (seconds) between first and last keyframe to compute a solution. Make the difference between monocular and stereo
    // unique_lock<mutex> lock0(mMutexImuInit);
    if (mbResetRequested)
        return;

    // Retrieve all keyframes in temporal order
    list<KeyFrame*> lpKF;
    KeyFrame* pKF = mpCurrentKeyFrame;
    while(pKF->mPrevKF)
    {
        lpKF.push_front(pKF);
        pKF = pKF->mPrevKF;
    }
    lpKF.push_front(pKF);
    vector<KeyFrame*> vpKF(lpKF.begin(),lpKF.end());

    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    const int N = vpKF.size();

    mRwg = Eigen::Matrix3d::Identity();
    mScale=1.0;

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    if (mScale<1e-1) // 1e-1
    {
        cout << "scale too small" << endl;
        bInitializing=false;
        return;
    }
    
    Sophus::SO3d so3wg(mRwg);
    // Before this line we are not changing the map
    unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    if ((fabs(mScale-1.f)>0.002)||!mbMonocular)
    {
        Sophus::SE3f Tgw(mRwg.cast<float>().transpose(),Eigen::Vector3f::Zero());
        mpAtlas->GetCurrentMap()->ApplyScaledRotation(Tgw,mScale,true);
        mpTracker->UpdateFrameIMU(mScale,mpCurrentKeyFrame->GetImuBias(),mpCurrentKeyFrame);
    }
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
    {
        (*lit)->SetBadFlag();
        delete *lit;
    }
    mlNewKeyFrames.clear();

    double t_inertial_only = std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();

    // To perform pose-inertial opt w.r.t. last keyframe
    mpCurrentKeyFrame->GetMap()->IncreaseChangeIndex();

    return;
}



bool LocalMapping::IsInitializing()
{
    return bInitializing;
}


double LocalMapping::GetCurrKFTime()
{

    if (mpCurrentKeyFrame)
    {
        return mpCurrentKeyFrame->mTimeStamp;
    }
    else
        return 0.0;
}

KeyFrame* LocalMapping::GetCurrKF()
{
    return mpCurrentKeyFrame;
}

} //namespace ORB_SLAM
