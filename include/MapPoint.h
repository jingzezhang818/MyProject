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


#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"
#include "Converter.h"

#include "SerializationUtils.h"

#include <opencv2/core/core.hpp>
#include <cstddef>
#include <mutex>
#include <vector>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>

namespace ORB_SLAM3
{

class KeyFrame;
class Map;
class Frame;

class MapPoint
{

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & mnId;
        ar & mnFirstKFid;
        ar & mnFirstFrame;
        ar & nObs;
        // Variables used by the tracking
        //ar & mTrackProjX;
        //ar & mTrackProjY;
        //ar & mTrackDepth;
        //ar & mTrackDepthR;
        //ar & mTrackProjXR;
        //ar & mTrackProjYR;
        //ar & mbTrackInView;
        //ar & mbTrackInViewR;
        //ar & mnTrackScaleLevel;
        //ar & mnTrackScaleLevelR;
        //ar & mTrackViewCos;
        //ar & mTrackViewCosR;
        //ar & mnTrackReferenceForFrame;
        //ar & mnLastFrameSeen;

        // Variables used by local mapping
        //ar & mnBALocalForKF;
        //ar & mnFuseCandidateForKF;

        // Variables used by loop closing and merging
        //ar & mnLoopPointForKF;
        //ar & mnCorrectedByKF;
        //ar & mnCorrectedReference;
        //serializeMatrix(ar,mPosGBA,version);
        //ar & mnBAGlobalForKF;
        //ar & mnBALocalForMerge;
        //serializeMatrix(ar,mPosMerge,version);
        //serializeMatrix(ar,mNormalVectorMerge,version);

        // Protected variables
        ar & boost::serialization::make_array(mWorldPos.data(), mWorldPos.size());
        ar & boost::serialization::make_array(mNormalVector.data(), mNormalVector.size());
        //ar & BOOST_SERIALIZATION_NVP(mBackupObservationsId);
        //ar & mObservations;
        ar & mBackupObservationsId1;
        ar & mBackupObservationsId2;
        serializeMatrix(ar,mDescriptor,version);
        ar & mBackupRefKFId;
        //ar & mnVisible;
        //ar & mnFound;

        ar & mbBad;
        ar & mBackupReplacedId;

        ar & mfMinDistance;
        ar & mfMaxDistance;

    }


public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    enum CreationSource : unsigned char
    {
        kCreationSourceUnknown = 0,
        kCreationSourceTrackingStereo = 1,
        kCreationSourceLocalMappingTriangulation = 2,
        kCreationSourceMonoInit = 3,
        kCreationSourceStereoInit = 4
    };

    enum TriangulationMatchBackend : unsigned char
    {
        kTriangBackendUnknown = 0,
        kTriangBackendXFeatBF = 1,
        kTriangBackendLightGlue = 2,
        kTriangBackendORB = 3
    };

    enum TrackFrustumStatus : unsigned char
    {
        kTrackFrustumUnchecked = 0,
        kTrackFrustumAccepted = 1,
        kTrackFrustumNegativeDepth = 2,
        kTrackFrustumOutsideImage = 3,
        kTrackFrustumScaleRange = 4,
        kTrackFrustumViewAngle = 5
    };

    struct LocalProjectionHistoryStats
    {
        int attemptCount;
        int matchCount;
        int optInlierCount;
        int optOutlierCount;
        unsigned int recentValidMask;
        unsigned int recentOutlierMask;
    };

    struct LocalMapVisibilityHistoryStats
    {
        int evaluationCount;
        int acceptedCount;
        int negativeDepthCount;
        int outsideImageCount;
        int scaleRangeCount;
        int viewAngleCount;
        unsigned int recentAcceptedMask;
        unsigned int recentIrrelevantMask;
    };

    struct CloseReuseCohortStats
    {
        long unsigned int firstProjectionAttemptFrame;
        long unsigned int firstProjectionMatchFrame;
        long unsigned int firstOptInlierFrame;
        long unsigned int firstOptOutlierFrame;
    };

    MapPoint();

    MapPoint(const Eigen::Vector3f &Pos, KeyFrame* pRefKF, Map* pMap);
    MapPoint(const double invDepth, cv::Point2f uv_init, KeyFrame* pRefKF, KeyFrame* pHostKF, Map* pMap);
    MapPoint(const Eigen::Vector3f &Pos,  Map* pMap, Frame* pFrame, const int &idxF);

    void SetWorldPos(const Eigen::Vector3f &Pos);
    Eigen::Vector3f GetWorldPos();

    Eigen::Vector3f GetNormal();
    void SetNormalVector(const Eigen::Vector3f& normal);

    KeyFrame* GetReferenceKeyFrame();

    std::map<KeyFrame*,std::tuple<int,int>> GetObservations();
    int Observations();

    void AddObservation(KeyFrame* pKF,int idx);
    void EraseObservation(KeyFrame* pKF);

    std::tuple<int,int> GetIndexInKeyFrame(KeyFrame* pKF);
    bool IsInKeyFrame(KeyFrame* pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(MapPoint* pMP);    
    MapPoint* GetReplaced();

    void IncreaseVisible(int n=1);
    void IncreaseFound(int n=1);
    float GetFoundRatio();
    inline int GetFound(){
        return mnFound;
    }

    void ComputeDistinctiveDescriptors();

    cv::Mat GetDescriptor();
    // XFeat路径: 返回用于匹配的描述子集合（可能包含多个视角原型）。
    std::vector<cv::Mat> GetXFeatDescriptorBank();
    bool HasXFeatDescriptor(bool useDescriptorBank);
    float GetBestXFeatDescriptorDistance(const cv::Mat& query, bool useDescriptorBank);
    void GetBestXFeatDescriptorDistances(const cv::Mat& queries,
                                         const std::vector<size_t>& queryRows,
                                         std::vector<float>& distances,
                                         bool useDescriptorBank,
                                         int rowOffset = 0);
    // XFeat路径: 针对查询描述子，返回当前MapPoint最优的描述子原型。
    cv::Mat GetBestDescriptorForQuery(const cv::Mat& query);

    void UpdateNormalAndDepth();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int PredictScale(const float &currentDist, KeyFrame*pKF);
    int PredictScale(const float &currentDist, Frame* pF);

    Map* GetMap();
    void UpdateMap(Map* pMap);
    void SetCreationSource(const CreationSource source);
    CreationSource GetCreationSource() const;
    void SetTriangulationQuality(const float depthAtCreation,
                                 const float cosParallax,
                                 const float reprojChi2Max,
                                 const float reprojChi2Mean,
                                 const float scaleLogResidual,
                                 const float scaleLogMargin,
                                 const TriangulationMatchBackend backend,
                                 const long unsigned int anchorKf1Id,
                                 const long unsigned int anchorKf2Id);
    bool HasTriangulationQuality() const;
    void SetMappingFeatureQuality(const long unsigned int keyFrameId,
                                  const int keyPointIdx,
                                  const int gridX,
                                  const int gridY,
                                  const int octave,
                                  const float response,
                                  const float depth,
                                  const unsigned char stereoBirthQuality);
    bool HasMappingFeatureQuality() const;
    void SetMappingProbation(const bool active,
                             const long unsigned int startKfId,
                             const float depthAtCreation);
    bool IsMappingProbation() const;
    bool WasMappingProbation() const;
    bool IsMappingProbationPromoted() const;
    bool TryPromoteMappingProbation(const long unsigned int currentKfId);
    void RecordLocalProjectionAttempt(const long unsigned int frameId);
    void RecordLocalProjectionMatch(const long unsigned int frameId);
    void RecordLocalProjectionOutcome(const long unsigned int frameId, const bool bInlier);
    LocalProjectionHistoryStats GetLocalProjectionHistory() const;
    void RecordLocalMapVisibilityOutcome(const long unsigned int frameId,
                                         const TrackFrustumStatus status);
    LocalMapVisibilityHistoryStats GetLocalMapVisibilityHistory() const;
    void RecordCloseReuseProjectionAttempt(const long unsigned int frameId);
    void RecordCloseReuseProjectionMatch(const long unsigned int frameId);
    void RecordCloseReuseOutcome(const long unsigned int frameId, const bool bInlier);
    CloseReuseCohortStats GetCloseReuseCohortStats() const;

    void PrintObservations();

    void PreSave(set<KeyFrame*>& spKF,set<MapPoint*>& spMP);
    void PostLoad(map<long unsigned int, KeyFrame*>& mpKFid, map<long unsigned int, MapPoint*>& mpMPid);

public:
    long unsigned int mnId;
    static long unsigned int nNextId;
    long int mnFirstKFid;
    long int mnFirstFrame;
    int nObs;
    CreationSource mCreationSource;
    TrackFrustumStatus mTrackFrustumStatus;
    bool mbHasTriangulationQuality;
    float mTriangDepthAtCreation;
    float mTriangCosParallax;
    float mTriangReprojChi2Max;
    float mTriangReprojChi2Mean;
    float mTriangScaleLogResidual;
    float mTriangScaleLogMargin;
    TriangulationMatchBackend mTriangMatchBackend;
    long unsigned int mnTriangAnchorKF1Id;
    long unsigned int mnTriangAnchorKF2Id;
    bool mbHasMappingFeatureQuality;
    long unsigned int mnMappingFeatureKeyFrameId;
    int mnMappingFeatureKeyPointIdx;
    int mnMappingFeatureGridX;
    int mnMappingFeatureGridY;
    int mnMappingFeatureOctave;
    float mMappingFeatureResponse;
    float mMappingFeatureDepth;
    unsigned char mMappingFeatureStereoBirthQuality;
    bool mbMappingProbation = false;
    bool mbMappingProbationEver = false;
    bool mbMappingProbationPromoted = false;
    long unsigned int mnMappingProbationStartKF = 0;
    long unsigned int mnMappingProbationPromotedKF = 0;
    float mMappingProbationDepthAtCreation = -1.0f;
    int mLocalProjAttemptCount;
    int mLocalProjMatchCount;
    int mLocalProjOptInlierCount;
    int mLocalProjOptOutlierCount;
    unsigned int mLocalProjRecentValidMask;
    unsigned int mLocalProjRecentOutlierMask;
    long unsigned int mnLastLocalProjAttemptFrame;
    long unsigned int mnLastLocalProjMatchFrame;
    long unsigned int mnLastLocalProjOutcomeFrame;
    int mLocalMapVisibilityEvaluationCount;
    int mLocalMapVisibilityAcceptedCount;
    int mLocalMapVisibilityNegativeDepthCount;
    int mLocalMapVisibilityOutsideImageCount;
    int mLocalMapVisibilityScaleRangeCount;
    int mLocalMapVisibilityViewAngleCount;
    unsigned int mLocalMapVisibilityRecentAcceptedMask;
    unsigned int mLocalMapVisibilityRecentIrrelevantMask;
    long unsigned int mnLastLocalMapVisibilityFrame;
    long unsigned int mnFirstCloseReuseProjectionAttemptFrame;
    long unsigned int mnFirstCloseReuseProjectionMatchFrame;
    long unsigned int mnFirstCloseReuseOptInlierFrame;
    long unsigned int mnFirstCloseReuseOptOutlierFrame;

    // Variables used by the tracking
    float mTrackProjX;
    float mTrackProjY;
    float mTrackDepth;
    float mTrackDepthR;
    float mTrackProjXR;
    float mTrackProjYR;
    bool mbTrackInView, mbTrackInViewR;
    int mnTrackScaleLevel, mnTrackScaleLevelR;
    float mTrackViewCos, mTrackViewCosR;
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnLastFrameSeen;

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing
    long unsigned int mnLoopPointForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;    
    Eigen::Vector3f mPosGBA;
    long unsigned int mnBAGlobalForKF;
    long unsigned int mnBALocalForMerge;

    // Variable used by merging
    Eigen::Vector3f mPosMerge;
    Eigen::Vector3f mNormalVectorMerge;


    // Fopr inverse depth optimization
    double mInvDepth;
    double mInitU;
    double mInitV;
    KeyFrame* mpHostKF;

    static std::mutex mGlobalMutex;

    unsigned int mnOriginMapId;

protected:    

     // Position in absolute coordinates
     Eigen::Vector3f mWorldPos;

     // Keyframes observing the point and associated index in keyframe
     std::map<KeyFrame*,std::tuple<int,int> > mObservations;
     // For save relation without pointer, this is necessary for save/load function
     std::map<long unsigned int, int> mBackupObservationsId1;
     std::map<long unsigned int, int> mBackupObservationsId2;

     // Mean viewing direction
     Eigen::Vector3f mNormalVector;

     // Best descriptor to fast matching
     cv::Mat mDescriptor;
     // XFeat路径: 多原型描述子库（运行时缓存，不参与序列化）。
     std::vector<cv::Mat> mXFeatDescriptorBank;

     // Reference KeyFrame
     KeyFrame* mpRefKF;
     long unsigned int mBackupRefKFId;

     // Tracking counters
     int mnVisible;
     int mnFound;

     // Bad flag (we do not currently erase MapPoint from memory)
     bool mbBad;
     MapPoint* mpReplaced;
     // For save relation without pointer, this is necessary for save/load function
     long long int mBackupReplacedId;

     // Scale invariance distances
     float mfMinDistance;
     float mfMaxDistance;

     Map* mpMap;

     // Mutex
     std::mutex mMutexPos;
     std::mutex mMutexFeatures;
     std::mutex mMutexMap;

};

} //namespace ORB_SLAM

#endif // MAPPOINT_H
