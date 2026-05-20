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
#include<string>
#include<tuple>
#include<unordered_set>
#include<vector>

namespace ORB_SLAM3
{

namespace
{
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

    constexpr size_t kMapPointCreationSourceCount = 5;

    size_t MapPointCreationSourceIndex(const MapPoint::CreationSource source)
    {
        const unsigned int idx = static_cast<unsigned int>(source);
        if(idx >= kMapPointCreationSourceCount)
            return 0;
        return static_cast<size_t>(idx);
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

    void AppendRecentMapPointCullingDiag(std::ostream& os,
                                         const RecentMapPointCullingDiag& diag)
    {
        os << " recent_total=" << diag.total
           << " kept=" << diag.kept
           << " already_bad=" << diag.already_bad
           << " found_ratio_culled=" << diag.found_ratio_culled
           << " low_obs_culled=" << diag.low_obs_culled
           << " aged_out_survived=" << diag.aged_out_survived;
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
    // For stereo inertial case
    if(mbMonocular)
        nn=30;
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
    int countStereo = 0;
    int countStereoGoodProj = 0;
    int countStereoAttempt = 0;
    int totalStereoPts = 0;

    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        GeometricCamera* pCamera1 = mpCurrentKeyFrame->mpCamera, *pCamera2 = pKF2->mpCamera;

        // Check first that baseline is not too short
        Eigen::Vector3f Ow2 = pKF2->GetCameraCenter();
        Eigen::Vector3f vBaseline = Ow2-Ow1;
        const float baseline = vBaseline.norm();

        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)
                continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2;

            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        bool bCoarse = mbInertial && mpTracker->mState==Tracking::RECENTLY_LOST && mpCurrentKeyFrame->GetMap()->GetIniertialBA2();

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
                goodProj = mpCurrentKeyFrame->UnprojectStereo(idx1, x3D);
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                countStereoAttempt++;
                bPointStereo = true;
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
                continue;

            float z2 = Rcw2.row(2).dot(x3D) + tcw2(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3D)+tcw1(0);
            const float y1 = Rcw1.row(1).dot(x3D)+tcw1(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                cv::Point2f uv1 = pCamera1->project(cv::Point3f(x1,y1,z1));
                float errX1 = uv1.x - kp1.pt.x;
                float errY1 = uv1.y - kp1.pt.y;

                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;

            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
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
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
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

            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpAtlas->GetCurrentMap());
            pMP->SetCreationSource(MapPoint::kCreationSourceLocalMappingTriangulation);
            if (bPointStereo)
                countStereo++;
            
            pMP->AddObservation(mpCurrentKeyFrame,idx1);
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpAtlas->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);
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
