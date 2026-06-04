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

#include "KeyFrame.h"
#include "Converter.h"
#include "ImuTypes.h"
#include <array>
#include <cmath>
#include <cstdlib>
#include<mutex>
#include <opencv2/imgproc.hpp>

namespace ORB_SLAM3
{
namespace
{
constexpr bool kEnableMappingProbationCovisVoteBehavior = false;

int GetEnvIntKeyFrame(const char* name, const int defaultValue)
{
    const char* value = std::getenv(name);
    if(!value)
        return defaultValue;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if(end == value)
        return defaultValue;
    return static_cast<int>(parsed);
}

bool IsEnvEnabledKeyFrame(const char* name)
{
    return GetEnvIntKeyFrame(name, 0) != 0;
}

bool ShouldPrintCovisVoteDiag(const long unsigned int frameId)
{
    if(!IsEnvEnabledKeyFrame("XFEAT_DEBUG") ||
       !IsEnvEnabledKeyFrame("XFEAT_DIAG_FEATURE_DISTRIBUTION"))
    {
        return false;
    }
    const int interval = std::max(1, GetEnvIntKeyFrame("XFEAT_DIAG_INTERVAL", 30));
    return frameId % static_cast<long unsigned int>(interval) == 0;
}

int MapPointCreationSourceIndexKeyFrame(const MapPoint::CreationSource source)
{
    switch(source)
    {
    case MapPoint::kCreationSourceTrackingStereo:
        return 0;
    case MapPoint::kCreationSourceLocalMappingTriangulation:
        return 1;
    case MapPoint::kCreationSourceMonoInit:
        return 2;
    case MapPoint::kCreationSourceStereoInit:
        return 3;
    case MapPoint::kCreationSourceUnknown:
    default:
        return 4;
    }
}

float ResolveMapPointDepthInKeyFrame(KeyFrame* pKF, MapPoint* pMP)
{
    if(!pKF || !pMP)
        return -1.0f;
    const Sophus::SE3f Tcw = pKF->GetPose();
    const Eigen::Vector3f Pc = Tcw * pMP->GetWorldPos();
    const float depth = Pc.z();
    if(!std::isfinite(depth) || depth <= 0.0f)
        return -1.0f;
    return depth;
}

int CountConnectedKeyFramesAboveThreshold(const std::map<KeyFrame*, int>& counter,
                                          const int threshold,
                                          int& topWeight)
{
    int connected = 0;
    topWeight = 0;
    for(const auto& item : counter)
    {
        topWeight = std::max(topWeight, item.second);
        if(item.second >= threshold)
            connected++;
    }
    if(connected == 0 && topWeight > 0)
        connected = 1;
    return connected;
}

int CountMaskBitsKeyFrame(unsigned int mask)
{
    int count = 0;
    while(mask != 0U)
    {
        count += static_cast<int>(mask & 1U);
        mask >>= 1U;
    }
    return count;
}

bool IsCloseMidLmtAtCreationKeyFrame(MapPoint* pMP)
{
    if(!pMP ||
       pMP->GetCreationSource() != MapPoint::kCreationSourceLocalMappingTriangulation ||
       !pMP->HasTriangulationQuality())
    {
        return false;
    }
    const float depth = pMP->mTriangDepthAtCreation;
    return std::isfinite(depth) && depth > 0.0f && depth <= 10.0f;
}

bool IsLongLowRelevantLmtKeyFrame(MapPoint* pMP,
                                  MapPoint::LocalMapVisibilityHistoryStats& stats)
{
    if(!IsCloseMidLmtAtCreationKeyFrame(pMP))
        return false;
    stats = pMP->GetLocalMapVisibilityHistory();
    if(stats.evaluationCount < 5)
        return false;
    const int irrelevant = stats.negativeDepthCount + stats.outsideImageCount;
    const float irrelevantRate = static_cast<float>(irrelevant) /
                                 static_cast<float>(stats.evaluationCount);
    const int recentCount = std::min(stats.evaluationCount, 8);
    const float recentIrrelevantRate = recentCount > 0
        ? static_cast<float>(CountMaskBitsKeyFrame(stats.recentIrrelevantMask)) /
          static_cast<float>(recentCount)
        : 0.0f;
    return irrelevantRate >= 0.70f ||
           (recentCount >= 4 && recentIrrelevantRate >= 0.75f);
}

struct CovisRepeatabilityVoteDecision
{
    bool target = false;
    bool weak = false;
    bool lowObs = false;
    bool lowFound = false;
    bool young = false;
    bool lowRelevance = false;
};

CovisRepeatabilityVoteDecision ClassifyCovisRepeatabilityVote(KeyFrame* pKF,
                                                              MapPoint* pMP)
{
    CovisRepeatabilityVoteDecision decision;
    if(!pKF || !IsCloseMidLmtAtCreationKeyFrame(pMP))
        return decision;

    decision.target = true;

    const int obs = pMP->Observations();
    const float foundRatio = pMP->GetFoundRatio();
    const long int firstKfId = pMP->mnFirstKFid;
    const int ageKf = firstKfId >= 0 && pKF->mnId >= static_cast<unsigned long>(firstKfId)
        ? static_cast<int>(pKF->mnId - static_cast<unsigned long>(firstKfId))
        : 0;

    MapPoint::LocalMapVisibilityHistoryStats visibilityHistory{};
    decision.lowObs = obs <= 3;
    decision.lowFound = !std::isfinite(foundRatio) || foundRatio < 0.35f;
    decision.young = ageKf <= 2;
    decision.lowRelevance = IsLongLowRelevantLmtKeyFrame(pMP, visibilityHistory);

    const bool confirmed =
        obs >= 5 &&
        std::isfinite(foundRatio) &&
        foundRatio >= 0.50f &&
        ageKf >= 3 &&
        !decision.lowRelevance;

    decision.weak = !confirmed &&
                    (decision.lowObs ||
                     decision.lowFound ||
                     decision.young ||
                     decision.lowRelevance);

    return decision;
}

}

long unsigned int KeyFrame::nNextId=0;

KeyFrame::KeyFrame():
        mnFrameId(0),  mTimeStamp(0), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
        mfGridElementWidthInv(0), mfGridElementHeightInv(0),
        mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0), mnBALocalForMerge(0),
        mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnMergeQuery(0), mnMergeWords(0), mnBAGlobalForKF(0),
        fx(0), fy(0), cx(0), cy(0), invfx(0), invfy(0), mnPlaceRecognitionQuery(0), mnPlaceRecognitionWords(0), mPlaceRecognitionScore(0),
        mbf(0), mb(0), mThDepth(0), N(0), mvKeys(static_cast<vector<cv::KeyPoint> >(NULL)), mvKeysUn(static_cast<vector<cv::KeyPoint> >(NULL)),
        mvuRight(static_cast<vector<float> >(NULL)), mvDepth(static_cast<vector<float> >(NULL)),
        mvXFeatMappingEligible(), mvXFeatMappingProfileScore(), mnScaleLevels(0), mfScaleFactor(0),
        mfLogScaleFactor(0), mvScaleFactors(0), mvLevelSigma2(0), mvInvLevelSigma2(0), mnMinX(0), mnMinY(0), mnMaxX(0),
        mnMaxY(0), mPrevKF(static_cast<KeyFrame*>(NULL)), mNextKF(static_cast<KeyFrame*>(NULL)), mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
        mbToBeErased(false), mbBad(false), mHalfBaseline(0), mbCurrentPlaceRecognition(false), mnMergeCorrectedForKF(0),
        NLeft(0),NRight(0), mnNumberOfOpt(0), mbHasVelocity(false)
{

}

KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB):
    bImu(pMap->isImuInitialized()), mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0), mnBALocalForMerge(0),
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0), mnPlaceRecognitionQuery(0), mnPlaceRecognitionWords(0), mPlaceRecognitionScore(0),
    fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
    mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
    mvuRight(F.mvuRight), mvDepth(F.mvDepth), mDescriptors(F.mDescriptors.clone()),
    mvXFeatMappingEligible(F.mvXFeatMappingEligible), mvXFeatMappingProfileScore(F.mvXFeatMappingProfileScore),
    mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
    mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
    mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
    mnMaxY(F.mnMaxY), mK_(F.mK_), mPrevKF(NULL), mNextKF(NULL), mpImuPreintegrated(F.mpImuPreintegrated),
    mImuCalib(F.mImuCalib), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
    mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), mDistCoef(F.mDistCoef), mbNotErase(false), mnDataset(F.mnDataset),
    mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb/2), mpMap(pMap), mbCurrentPlaceRecognition(false), mNameFile(F.mNameFile), mnMergeCorrectedForKF(0),
    mpCamera(F.mpCamera), mpCamera2(F.mpCamera2),
    mvLeftToRightMatch(F.mvLeftToRightMatch),mvRightToLeftMatch(F.mvRightToLeftMatch), mTlr(F.GetRelativePoseTlr()),
    mvKeysRight(F.mvKeysRight), NLeft(F.Nleft), NRight(F.Nright), mTrl(F.GetRelativePoseTrl()), mnNumberOfOpt(0), mbHasVelocity(false)
{
    mnId=nNextId++;

    mGrid.resize(mnGridCols);
    if(F.Nleft != -1)  mGridRight.resize(mnGridCols);
    for(int i=0; i<mnGridCols;i++)
    {
        mGrid[i].resize(mnGridRows);
        if(F.Nleft != -1) mGridRight[i].resize(mnGridRows);
        for(int j=0; j<mnGridRows; j++){
            mGrid[i][j] = F.mGrid[i][j];
            if(F.Nleft != -1){
                mGridRight[i][j] = F.mGridRight[i][j];
            }
        }
    }



    if(!F.HasVelocity()) {
        mVw.setZero();
        mbHasVelocity = false;
    }
    else
    {
        mVw = F.GetVelocity();
        mbHasVelocity = true;
    }

    mImuBias = F.mImuBias;
    SetPose(F.GetPose());

    mnOriginMapId = pMap->GetId();
}

void KeyFrame::ComputeBoW()
{
    if(mBowVec.empty() || mFeatVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void KeyFrame::SetPose(const Sophus::SE3f &Tcw)
{
    unique_lock<mutex> lock(mMutexPose);

    mTcw = Tcw;
    mRcw = mTcw.rotationMatrix();
    mTwc = mTcw.inverse();
    mRwc = mTwc.rotationMatrix();

    if (mImuCalib.mbIsSet) // TODO Use a flag instead of the OpenCV matrix
    {
        mOwb = mRwc * mImuCalib.mTcb.translation() + mTwc.translation();
    }
}

void KeyFrame::SetVelocity(const Eigen::Vector3f &Vw)
{
    unique_lock<mutex> lock(mMutexPose);
    mVw = Vw;
    mbHasVelocity = true;
}

Sophus::SE3f KeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTcw;
}

Sophus::SE3f KeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTwc;
}

Eigen::Vector3f KeyFrame::GetCameraCenter(){
    unique_lock<mutex> lock(mMutexPose);
    return mTwc.translation();
}

Eigen::Vector3f KeyFrame::GetImuPosition()
{
    unique_lock<mutex> lock(mMutexPose);
    return mOwb;
}

Eigen::Matrix3f KeyFrame::GetImuRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return (mTwc * mImuCalib.mTcb).rotationMatrix();
}

Sophus::SE3f KeyFrame::GetImuPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTwc * mImuCalib.mTcb;
}

Eigen::Matrix3f KeyFrame::GetRotation(){
    unique_lock<mutex> lock(mMutexPose);
    return mRcw;
}

Eigen::Vector3f KeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTcw.translation();
}

Eigen::Vector3f KeyFrame::GetVelocity()
{
    unique_lock<mutex> lock(mMutexPose);
    return mVw;
}

bool KeyFrame::isVelocitySet()
{
    unique_lock<mutex> lock(mMutexPose);
    return mbHasVelocity;
}

void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(!mConnectedKeyFrameWeights.count(pKF))
            mConnectedKeyFrameWeights[pKF]=weight;
        else if(mConnectedKeyFrameWeights[pKF]!=weight)
            mConnectedKeyFrameWeights[pKF]=weight;
        else
            return;
    }

    UpdateBestCovisibles();
}

void KeyFrame::UpdateBestCovisibles()
{
    unique_lock<mutex> lock(mMutexConnections);
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
       vPairs.push_back(make_pair(mit->second,mit->first));

    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0, iend=vPairs.size(); i<iend;i++)
    {
        if(!vPairs[i].second->isBad())
        {
            lKFs.push_front(vPairs[i].second);
            lWs.push_front(vPairs[i].first);
        }
    }

    mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());
}

set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFrame*> s;
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin();mit!=mConnectedKeyFrameWeights.end();mit++)
        s.insert(mit->first);
    return s;
}

vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections);
    if((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);

}

vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if(mvpOrderedConnectedKeyFrames.empty())
    {
        return vector<KeyFrame*>();
    }

    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,KeyFrame::weightComp);

    if(it==mvOrderedWeights.end() && mvOrderedWeights.back() < w)
    {
        return vector<KeyFrame*>();
    }
    else
    {
        int n = it-mvOrderedWeights.begin();
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
    }
}

int KeyFrame::GetWeight(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

int KeyFrame::GetNumberMPs()
{
    unique_lock<mutex> lock(mMutexFeatures);
    int numberMPs = 0;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        numberMPs++;
    }
    return numberMPs;
}

void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=pMP;
}

void KeyFrame::EraseMapPointMatch(const int &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}

void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    tuple<size_t,size_t> indexes = pMP->GetIndexInKeyFrame(this);
    size_t leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
    if(leftIndex != -1)
        mvpMapPoints[leftIndex]=static_cast<MapPoint*>(NULL);
    if(rightIndex != -1)
        mvpMapPoints[rightIndex]=static_cast<MapPoint*>(NULL);
}


void KeyFrame::ReplaceMapPointMatch(const int &idx, MapPoint* pMP)
{
    mvpMapPoints[idx]=pMP;
}

set<MapPoint*> KeyFrame::GetMapPoints()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapPoint*> s;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        MapPoint* pMP = mvpMapPoints[i];
        if(!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}

int KeyFrame::TrackedMapPoints(const int &minObs)
{
    unique_lock<mutex> lock(mMutexFeatures);

    int nPoints=0;
    const bool bCheckObs = minObs>0;
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = mvpMapPoints[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(bCheckObs)
                {
                    if(mvpMapPoints[i]->Observations()>=minObs)
                        nPoints++;
                }
                else
                    nPoints++;
            }
        }
    }

    return nPoints;
}

vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

void KeyFrame::UpdateConnections(bool upParent)
{
    map<KeyFrame*,int> KFcounter;
    map<KeyFrame*,int> shadowKFcounter;
    map<KeyFrame*,int> longRelevanceShadowKFcounter;
    map<KeyFrame*,int> covisRepeatabilityWeakKFcounter;
    map<KeyFrame*,int> mappingProbationKFcounter;
    const bool collectCovisVoteDiag = ShouldPrintCovisVoteDiag(mnFrameId);
    const bool bUseORB = std::getenv("USE_ORB") != nullptr;

    constexpr int kSourceCount = 5;
    std::array<int, kSourceCount> voteSourceCounts{};
    std::array<int, kSourceCount> mpSourceCounts{};

    int mpTotal = 0;
    int mpLmt0To10 = 0;
    int mpLmt0To10Weak = 0;
    int mpLmt0To10Confirmed = 0;
    int voteTotal = 0;
    int voteLmt0To10 = 0;
    int voteLmt0To10Weak = 0;
    int voteLmt0To10Confirmed = 0;
    int shadowSkippedVotes = 0;
    int mpLmtCreation0To10LongEvaluated = 0;
    int mpLmtCreation0To10LongLowRelevant = 0;
    int voteLmtCreation0To10LongLowRelevant = 0;
    int longRelevanceShadowSkippedVotes = 0;
    int covisRepeatabilityTargetPoints = 0;
    int covisRepeatabilityWeakPoints = 0;
    int covisRepeatabilityWeakLowObsPoints = 0;
    int covisRepeatabilityWeakLowFoundPoints = 0;
    int covisRepeatabilityWeakYoungPoints = 0;
    int covisRepeatabilityWeakLowRelevancePoints = 0;
    int covisRepeatabilityWeakVoteRaw = 0;
    int covisRepeatabilityWeakVoteEffective = 0;
    int covisRepeatabilityWeakVoteSuppressed = 0;
    int covisRepeatabilityWeakVoteKFs = 0;
    int mappingProbationTargetPoints = 0;
    int mappingProbationActivePoints = 0;
    int mappingProbationPromotedPoints = 0;
    int mappingProbationPromotedNowPoints = 0;
    int mappingProbationVoteRaw = 0;
    int mappingProbationVoteEffective = 0;
    int mappingProbationVoteSuppressed = 0;
    int mappingProbationVoteKFs = 0;

    vector<MapPoint*> vpMP;

    {
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    for(vector<MapPoint*>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        if(!pMP)
            continue;

        if(pMP->isBad())
            continue;

        const bool closeMidLmtAtCreation =
            !bUseORB && IsCloseMidLmtAtCreationKeyFrame(pMP);
        const bool wasMappingProbation =
            closeMidLmtAtCreation && pMP->WasMappingProbation();
        bool promotedMappingProbationNow = false;
        const bool activeMappingProbation =
            closeMidLmtAtCreation && pMP->IsMappingProbation();
        const bool promotedMappingProbation =
            closeMidLmtAtCreation && pMP->IsMappingProbationPromoted();

        if(wasMappingProbation)
        {
            mappingProbationTargetPoints++;
            if(activeMappingProbation)
                mappingProbationActivePoints++;
            if(promotedMappingProbation)
                mappingProbationPromotedPoints++;
            if(promotedMappingProbationNow)
                mappingProbationPromotedNowPoints++;
        }

        const CovisRepeatabilityVoteDecision repeatabilityVote =
            bUseORB ? CovisRepeatabilityVoteDecision()
                    : ClassifyCovisRepeatabilityVote(this, pMP);
        if(repeatabilityVote.target)
        {
            covisRepeatabilityTargetPoints++;
            if(repeatabilityVote.weak)
            {
                covisRepeatabilityWeakPoints++;
                if(repeatabilityVote.lowObs)
                    covisRepeatabilityWeakLowObsPoints++;
                if(repeatabilityVote.lowFound)
                    covisRepeatabilityWeakLowFoundPoints++;
                if(repeatabilityVote.young)
                    covisRepeatabilityWeakYoungPoints++;
                if(repeatabilityVote.lowRelevance)
                    covisRepeatabilityWeakLowRelevancePoints++;
            }
        }

        MapPoint::CreationSource source = MapPoint::kCreationSourceUnknown;
        int sourceIdx = MapPointCreationSourceIndexKeyFrame(source);
        bool isLmt0To10 = false;
        bool isWeakLmt0To10 = false;
        bool isConfirmedLmt0To10 = false;
        bool isLongLowRelevantLmt0To10 = false;
        if(collectCovisVoteDiag)
        {
            mpTotal++;
            source = pMP->GetCreationSource();
            sourceIdx = MapPointCreationSourceIndexKeyFrame(source);
            mpSourceCounts[sourceIdx]++;
            const int obs = pMP->Observations();
            const float foundRatio = pMP->GetFoundRatio();
            const int ageKf = mnId >= static_cast<long unsigned int>(std::max(0L, pMP->mnFirstKFid)) ?
                static_cast<int>(mnId - static_cast<long unsigned int>(std::max(0L, pMP->mnFirstKFid))) : 0;
            const float depth = ResolveMapPointDepthInKeyFrame(this, pMP);
            isLmt0To10 =
                source == MapPoint::kCreationSourceLocalMappingTriangulation &&
                depth > 0.0f && depth <= 10.0f;
            isWeakLmt0To10 =
                isLmt0To10 && (obs <= 3 || foundRatio < 0.35f || ageKf <= 2);
            isConfirmedLmt0To10 =
                isLmt0To10 && obs >= 5 && foundRatio >= 0.50f && ageKf >= 3;

            if(isLmt0To10)
            {
                mpLmt0To10++;
                if(isWeakLmt0To10)
                    mpLmt0To10Weak++;
                if(isConfirmedLmt0To10)
                    mpLmt0To10Confirmed++;
            }
            if(IsCloseMidLmtAtCreationKeyFrame(pMP))
            {
                MapPoint::LocalMapVisibilityHistoryStats visibilityHistory{};
                isLongLowRelevantLmt0To10 =
                    IsLongLowRelevantLmtKeyFrame(pMP, visibilityHistory);
                if(visibilityHistory.evaluationCount >= 5)
                    mpLmtCreation0To10LongEvaluated++;
                if(isLongLowRelevantLmt0To10)
                    mpLmtCreation0To10LongLowRelevant++;
            }
        }

        map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

        for(map<KeyFrame*,tuple<int,int>>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            if(mit->first->mnId==mnId || mit->first->isBad() || mit->first->GetMap() != mpMap)
                continue;
            if(activeMappingProbation)
            {
                mappingProbationKFcounter[mit->first]++;
                mappingProbationVoteRaw++;
            }
            else
            {
                KFcounter[mit->first]++;
            }
            if(repeatabilityVote.weak)
            {
                covisRepeatabilityWeakKFcounter[mit->first]++;
                covisRepeatabilityWeakVoteRaw++;
            }
            if(collectCovisVoteDiag)
            {
                voteTotal++;
                voteSourceCounts[sourceIdx]++;
                if(isLmt0To10)
                {
                    voteLmt0To10++;
                    if(isWeakLmt0To10)
                        voteLmt0To10Weak++;
                    if(isConfirmedLmt0To10)
                        voteLmt0To10Confirmed++;
                }
                if(isWeakLmt0To10)
                {
                    shadowSkippedVotes++;
                }
                else
                {
                    shadowKFcounter[mit->first]++;
                }
                if(isLongLowRelevantLmt0To10)
                {
                    voteLmtCreation0To10LongLowRelevant++;
                    longRelevanceShadowSkippedVotes++;
                }
                else
                {
                    longRelevanceShadowKFcounter[mit->first]++;
                }
            }

        }
    }

    covisRepeatabilityWeakVoteKFs =
        static_cast<int>(covisRepeatabilityWeakKFcounter.size());

    if(kEnableMappingProbationCovisVoteBehavior &&
       !bUseORB &&
       !mappingProbationKFcounter.empty())
    {
        static constexpr int kMappingProbationVoteDownweight = 4;
        KeyFrame* pBestProbationKF = nullptr;
        int bestProbationVotes = 0;
        for(const auto& entry : mappingProbationKFcounter)
        {
            KeyFrame* pKF = entry.first;
            if(!pKF || pKF->isBad())
                continue;
            if(entry.second > bestProbationVotes)
            {
                bestProbationVotes = entry.second;
                pBestProbationKF = pKF;
            }
            const int effectiveVotes = entry.second / kMappingProbationVoteDownweight;
            if(effectiveVotes <= 0)
                continue;
            KFcounter[pKF] += effectiveVotes;
            mappingProbationVoteEffective += effectiveVotes;
            mappingProbationVoteKFs++;
        }
        if(KFcounter.empty() && pBestProbationKF)
        {
            KFcounter[pBestProbationKF] = 1;
            mappingProbationVoteEffective = 1;
            mappingProbationVoteKFs = 1;
        }
        mappingProbationVoteSuppressed =
            std::max(0, mappingProbationVoteRaw - mappingProbationVoteEffective);
    }

    // This should not happen
    if(KFcounter.empty())
        return;

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax=0;
    KeyFrame* pKFmax=NULL;
    int th = 15;

    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(KFcounter.size());
    if(!upParent)
        cout << "UPDATE_CONN: current KF " << mnId << endl;
    for(map<KeyFrame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        if(!upParent)
            cout << "  UPDATE_CONN: KF " << mit->first->mnId << " ; num matches: " << mit->second << endl;
        if(mit->second>nmax)
        {
            nmax=mit->second;
            pKFmax=mit->first;
        }
        if(mit->second>=th)
        {
            vPairs.push_back(make_pair(mit->second,mit->first));
            (mit->first)->AddConnection(this,mit->second);
        }
    }

    if(vPairs.empty())
    {
        vPairs.push_back(make_pair(nmax,pKFmax));
        pKFmax->AddConnection(this,nmax);
    }

    if(collectCovisVoteDiag)
    {
        int actualTopWeight = 0;
        int shadowTopWeight = 0;
        int longRelevanceShadowTopWeight = 0;
        const int actualConnected = CountConnectedKeyFramesAboveThreshold(KFcounter, th, actualTopWeight);
        const int shadowConnected = CountConnectedKeyFramesAboveThreshold(shadowKFcounter, th, shadowTopWeight);
        const int longRelevanceShadowConnected =
            CountConnectedKeyFramesAboveThreshold(longRelevanceShadowKFcounter,
                                                  th,
                                                  longRelevanceShadowTopWeight);
        int shadowDroppedConnected = 0;
        int longRelevanceShadowDroppedConnected = 0;
        for(const auto& item : KFcounter)
        {
            if(item.second < th)
                continue;
            const auto itShadow = shadowKFcounter.find(item.first);
            const int shadowWeight = itShadow != shadowKFcounter.end() ? itShadow->second : 0;
            if(shadowWeight < th)
                shadowDroppedConnected++;
            const auto itLongShadow = longRelevanceShadowKFcounter.find(item.first);
            const int longShadowWeight = itLongShadow != longRelevanceShadowKFcounter.end()
                ? itLongShadow->second : 0;
            if(longShadowWeight < th)
                longRelevanceShadowDroppedConnected++;
        }

        std::cout << "[KeyFrame][CovisVoteSource]"
                  << " frame_id=" << mnFrameId
                  << " kf_id=" << mnId
                  << " mp_total=" << mpTotal
                  << " mp_tracking_stereo=" << mpSourceCounts[MapPointCreationSourceIndexKeyFrame(MapPoint::kCreationSourceTrackingStereo)]
                  << " mp_local_mapping_triangulation=" << mpSourceCounts[MapPointCreationSourceIndexKeyFrame(MapPoint::kCreationSourceLocalMappingTriangulation)]
                  << " mp_mono_init=" << mpSourceCounts[MapPointCreationSourceIndexKeyFrame(MapPoint::kCreationSourceMonoInit)]
                  << " mp_stereo_init=" << mpSourceCounts[MapPointCreationSourceIndexKeyFrame(MapPoint::kCreationSourceStereoInit)]
                  << " mp_unknown=" << mpSourceCounts[MapPointCreationSourceIndexKeyFrame(MapPoint::kCreationSourceUnknown)]
                  << " mp_lmt_0_10=" << mpLmt0To10
                  << " mp_lmt_0_10_weak=" << mpLmt0To10Weak
                  << " mp_lmt_0_10_confirmed=" << mpLmt0To10Confirmed
                  << " vote_total=" << voteTotal
                  << " vote_tracking_stereo=" << voteSourceCounts[MapPointCreationSourceIndexKeyFrame(MapPoint::kCreationSourceTrackingStereo)]
                  << " vote_local_mapping_triangulation=" << voteSourceCounts[MapPointCreationSourceIndexKeyFrame(MapPoint::kCreationSourceLocalMappingTriangulation)]
                  << " vote_mono_init=" << voteSourceCounts[MapPointCreationSourceIndexKeyFrame(MapPoint::kCreationSourceMonoInit)]
                  << " vote_stereo_init=" << voteSourceCounts[MapPointCreationSourceIndexKeyFrame(MapPoint::kCreationSourceStereoInit)]
                  << " vote_unknown=" << voteSourceCounts[MapPointCreationSourceIndexKeyFrame(MapPoint::kCreationSourceUnknown)]
                  << " vote_lmt_0_10=" << voteLmt0To10
                  << " vote_lmt_0_10_weak=" << voteLmt0To10Weak
                  << " vote_lmt_0_10_confirmed=" << voteLmt0To10Confirmed
                  << " shadow_skipped_votes=" << shadowSkippedVotes
                  << " mp_lmt_creation_0_10_long_evaluated=" << mpLmtCreation0To10LongEvaluated
                  << " mp_lmt_creation_0_10_long_low_relevant=" << mpLmtCreation0To10LongLowRelevant
                  << " vote_lmt_creation_0_10_long_low_relevant=" << voteLmtCreation0To10LongLowRelevant
                  << " long_relevance_shadow_skipped_votes=" << longRelevanceShadowSkippedVotes
                  << " repeatability_vote_target_points=" << covisRepeatabilityTargetPoints
                  << " repeatability_vote_weak_points=" << covisRepeatabilityWeakPoints
                  << " repeatability_vote_weak_lowobs_points=" << covisRepeatabilityWeakLowObsPoints
                  << " repeatability_vote_weak_lowfound_points=" << covisRepeatabilityWeakLowFoundPoints
                  << " repeatability_vote_weak_young_points=" << covisRepeatabilityWeakYoungPoints
                  << " repeatability_vote_weak_low_relevance_points=" << covisRepeatabilityWeakLowRelevancePoints
                  << " repeatability_vote_weak_raw=" << covisRepeatabilityWeakVoteRaw
                  << " repeatability_vote_weak_effective=" << covisRepeatabilityWeakVoteEffective
                  << " repeatability_vote_weak_suppressed=" << covisRepeatabilityWeakVoteSuppressed
                  << " repeatability_vote_weak_kfs=" << covisRepeatabilityWeakVoteKFs
                  << " repeatability_vote_behavior_active=0"
                  << " mapping_probation_target_points=" << mappingProbationTargetPoints
                  << " mapping_probation_active_points=" << mappingProbationActivePoints
                  << " mapping_probation_promoted_points=" << mappingProbationPromotedPoints
                  << " mapping_probation_promoted_now_points=" << mappingProbationPromotedNowPoints
                  << " mapping_probation_vote_raw=" << mappingProbationVoteRaw
                  << " mapping_probation_vote_effective=" << mappingProbationVoteEffective
                  << " mapping_probation_vote_suppressed=" << mappingProbationVoteSuppressed
                  << " mapping_probation_vote_kfs=" << mappingProbationVoteKFs
                  << " mapping_probation_behavior_active="
                  << (kEnableMappingProbationCovisVoteBehavior ? 1 : 0)
                  << " actual_kfs=" << KFcounter.size()
                  << " actual_connected=" << actualConnected
                  << " actual_top_weight=" << actualTopWeight
                  << " shadow_kfs=" << shadowKFcounter.size()
                  << " shadow_connected=" << shadowConnected
                  << " shadow_top_weight=" << shadowTopWeight
                  << " shadow_connected_delta=" << (shadowConnected - actualConnected)
                  << " shadow_top_weight_delta=" << (shadowTopWeight - actualTopWeight)
                  << " shadow_dropped_connected=" << shadowDroppedConnected
                  << " long_relevance_shadow_kfs=" << longRelevanceShadowKFcounter.size()
                  << " long_relevance_shadow_connected=" << longRelevanceShadowConnected
                  << " long_relevance_shadow_top_weight=" << longRelevanceShadowTopWeight
                  << " long_relevance_shadow_connected_delta="
                  << (longRelevanceShadowConnected - actualConnected)
                  << " long_relevance_shadow_top_weight_delta="
                  << (longRelevanceShadowTopWeight - actualTopWeight)
                  << " long_relevance_shadow_dropped_connected="
                  << longRelevanceShadowDroppedConnected
                  << std::endl;
    }

    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0; i<vPairs.size();i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);

        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());


        if(mbFirstConnection && mnId!=mpMap->GetInitKFid())
        {
            mpParent = mvpOrderedConnectedKeyFrames.front();
            mpParent->AddChild(this);
            mbFirstConnection = false;
        }

    }
}

void KeyFrame::AddChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    if(pKF == this)
    {
        cout << "ERROR: Change parent KF, the parent and child are the same KF" << endl;
        throw std::invalid_argument("The parent and child can not be the same");
    }

    mpParent = pKF;
    pKF->AddChild(this);
}

set<KeyFrame*> KeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}

KeyFrame* KeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

bool KeyFrame::hasChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

void KeyFrame::SetFirstConnection(bool bFirst)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbFirstConnection=bFirst;
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

set<KeyFrame*> KeyFrame::GetLoopEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}

void KeyFrame::AddMergeEdge(KeyFrame* pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspMergeEdges.insert(pKF);
}

set<KeyFrame*> KeyFrame::GetMergeEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspMergeEdges;
}

void KeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}

void KeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mspLoopEdges.empty())
        {
            mbNotErase = false;
        }
    }

    if(mbToBeErased)
    {
        SetBadFlag();
    }
}

void KeyFrame::SetBadFlag()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mnId==mpMap->GetInitKFid())
        {
            return;
        }
        else if(mbNotErase)
        {
            mbToBeErased = true;
            return;
        }
    }

    for(map<KeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
    {
        mit->first->EraseConnection(this);
    }

    for(size_t i=0; i<mvpMapPoints.size(); i++)
    {
        if(mvpMapPoints[i])
        {
            mvpMapPoints[i]->EraseObservation(this);
        }
    }

    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        set<KeyFrame*> sParentCandidates;
        if(mpParent)
            sParentCandidates.insert(mpParent);

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        while(!mspChildrens.empty())
        {
            bool bContinue = false;

            int max = -1;
            KeyFrame* pC;
            KeyFrame* pP;

            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(), send=mspChildrens.end(); sit!=send; sit++)
            {
                KeyFrame* pKF = *sit;
                if(pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for(size_t i=0, iend=vpConnected.size(); i<iend; i++)
                {
                    for(set<KeyFrame*>::iterator spcit=sParentCandidates.begin(), spcend=sParentCandidates.end(); spcit!=spcend; spcit++)
                    {
                        if(vpConnected[i]->mnId == (*spcit)->mnId)
                        {
                            int w = pKF->GetWeight(vpConnected[i]);
                            if(w>max)
                            {
                                pC = pKF;
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            if(bContinue)
            {
                pC->ChangeParent(pP);
                sParentCandidates.insert(pC);
                mspChildrens.erase(pC);
            }
            else
                break;
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        if(!mspChildrens.empty())
        {
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(); sit!=mspChildrens.end(); sit++)
            {
                (*sit)->ChangeParent(mpParent);
            }
        }

        if(mpParent){
            mpParent->EraseChild(this);
            mTcp = mTcw * mpParent->GetPoseInverse();
        }
        mbBad = true;
    }


    mpMap->EraseKeyFrame(this);
    mpKeyFrameDB->erase(this);
}

bool KeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}

void KeyFrame::EraseConnection(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate=true;
        }
    }

    if(bUpdate)
        UpdateBestCovisibles();
}

vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r, const bool bRight) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    float factorX = r;
    float factorY = r;

    const int nMinCellX = max(0,(int)floor((x-mnMinX-factorX)*mfGridElementWidthInv));
    if(nMinCellX>=mnGridCols)
        return vIndices;

    const int nMaxCellX = min((int)mnGridCols-1,(int)ceil((x-mnMinX+factorX)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-factorY)*mfGridElementHeightInv));
    if(nMinCellY>=mnGridRows)
        return vIndices;

    const int nMaxCellY = min((int)mnGridRows-1,(int)ceil((y-mnMinY+factorY)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = (NLeft == -1) ? mvKeysUn[vCell[j]]
                                                         : (!bRight) ? mvKeys[vCell[j]]
                                                                     : mvKeysRight[vCell[j]];
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool KeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}

bool KeyFrame::UnprojectStereo(int i, Eigen::Vector3f &x3D)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeys[i].pt.x;
        const float v = mvKeys[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        Eigen::Vector3f x3Dc(x, y, z);

        unique_lock<mutex> lock(mMutexPose);
        x3D = mRwc * x3Dc + mTwc.translation();
        return true;
    }
    else
        return false;
}

float KeyFrame::ComputeSceneMedianDepth(const int q)
{
    if(N==0)
        return -1.0;

    vector<MapPoint*> vpMapPoints;
    Eigen::Matrix3f Rcw;
    Eigen::Vector3f tcw;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        tcw = mTcw.translation();
        Rcw = mRcw;
    }

    vector<float> vDepths;
    vDepths.reserve(N);
    Eigen::Matrix<float,1,3> Rcw2 = Rcw.row(2);
    float zcw = tcw(2);
    for(int i=0; i<N; i++) {
        if(mvpMapPoints[i])
        {
            MapPoint* pMP = mvpMapPoints[i];
            Eigen::Vector3f x3Dw = pMP->GetWorldPos();
            float z = Rcw2.dot(x3Dw) + zcw;
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(),vDepths.end());

    return vDepths[(vDepths.size()-1)/q];
}

void KeyFrame::SetNewBias(const IMU::Bias &b)
{
    unique_lock<mutex> lock(mMutexPose);
    mImuBias = b;
    if(mpImuPreintegrated)
        mpImuPreintegrated->SetNewBias(b);
}

Eigen::Vector3f KeyFrame::GetGyroBias()
{
    unique_lock<mutex> lock(mMutexPose);
    return Eigen::Vector3f(mImuBias.bwx, mImuBias.bwy, mImuBias.bwz);
}

Eigen::Vector3f KeyFrame::GetAccBias()
{
    unique_lock<mutex> lock(mMutexPose);
    return Eigen::Vector3f(mImuBias.bax, mImuBias.bay, mImuBias.baz);
}

IMU::Bias KeyFrame::GetImuBias()
{
    unique_lock<mutex> lock(mMutexPose);
    return mImuBias;
}

Map* KeyFrame::GetMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mpMap;
}

void KeyFrame::UpdateMap(Map* pMap)
{
    unique_lock<mutex> lock(mMutexMap);
    mpMap = pMap;
}

void KeyFrame::PreSave(set<KeyFrame*>& spKF,set<MapPoint*>& spMP, set<GeometricCamera*>& spCam)
{
    // Save the id of each MapPoint in this KF, there can be null pointer in the vector
    mvBackupMapPointsId.clear();
    mvBackupMapPointsId.reserve(N);
    for(int i = 0; i < N; ++i)
    {

        if(mvpMapPoints[i] && spMP.find(mvpMapPoints[i]) != spMP.end()) // Checks if the element is not null
            mvBackupMapPointsId.push_back(mvpMapPoints[i]->mnId);
        else // If the element is null his value is -1 because all the id are positives
            mvBackupMapPointsId.push_back(-1);
    }
    // Save the id of each connected KF with it weight
    mBackupConnectedKeyFrameIdWeights.clear();
    for(std::map<KeyFrame*,int>::const_iterator it = mConnectedKeyFrameWeights.begin(), end = mConnectedKeyFrameWeights.end(); it != end; ++it)
    {
        if(spKF.find(it->first) != spKF.end())
            mBackupConnectedKeyFrameIdWeights[it->first->mnId] = it->second;
    }

    // Save the parent id
    mBackupParentId = -1;
    if(mpParent && spKF.find(mpParent) != spKF.end())
        mBackupParentId = mpParent->mnId;

    // Save the id of the childrens KF
    mvBackupChildrensId.clear();
    mvBackupChildrensId.reserve(mspChildrens.size());
    for(KeyFrame* pKFi : mspChildrens)
    {
        if(spKF.find(pKFi) != spKF.end())
            mvBackupChildrensId.push_back(pKFi->mnId);
    }

    // Save the id of the loop edge KF
    mvBackupLoopEdgesId.clear();
    mvBackupLoopEdgesId.reserve(mspLoopEdges.size());
    for(KeyFrame* pKFi : mspLoopEdges)
    {
        if(spKF.find(pKFi) != spKF.end())
            mvBackupLoopEdgesId.push_back(pKFi->mnId);
    }

    // Save the id of the merge edge KF
    mvBackupMergeEdgesId.clear();
    mvBackupMergeEdgesId.reserve(mspMergeEdges.size());
    for(KeyFrame* pKFi : mspMergeEdges)
    {
        if(spKF.find(pKFi) != spKF.end())
            mvBackupMergeEdgesId.push_back(pKFi->mnId);
    }

    //Camera data
    mnBackupIdCamera = -1;
    if(mpCamera && spCam.find(mpCamera) != spCam.end())
        mnBackupIdCamera = mpCamera->GetId();

    mnBackupIdCamera2 = -1;
    if(mpCamera2 && spCam.find(mpCamera2) != spCam.end())
        mnBackupIdCamera2 = mpCamera2->GetId();

    //Inertial data
    mBackupPrevKFId = -1;
    if(mPrevKF && spKF.find(mPrevKF) != spKF.end())
        mBackupPrevKFId = mPrevKF->mnId;

    mBackupNextKFId = -1;
    if(mNextKF && spKF.find(mNextKF) != spKF.end())
        mBackupNextKFId = mNextKF->mnId;

    if(mpImuPreintegrated)
        mBackupImuPreintegrated.CopyFrom(mpImuPreintegrated);
}

void KeyFrame::PostLoad(map<long unsigned int, KeyFrame*>& mpKFid, map<long unsigned int, MapPoint*>& mpMPid, map<unsigned int, GeometricCamera*>& mpCamId){
    // Rebuild the empty variables

    // Pose
    SetPose(mTcw);

    mTrl = mTlr.inverse();

    // Reference reconstruction
    // Each MapPoint sight from this KeyFrame
    mvpMapPoints.clear();
    mvpMapPoints.resize(N);
    for(int i=0; i<N; ++i)
    {
        if(mvBackupMapPointsId[i] != -1)
            mvpMapPoints[i] = mpMPid[mvBackupMapPointsId[i]];
        else
            mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
    }

    // Conected KeyFrames with him weight
    mConnectedKeyFrameWeights.clear();
    for(map<long unsigned int, int>::const_iterator it = mBackupConnectedKeyFrameIdWeights.begin(), end = mBackupConnectedKeyFrameIdWeights.end();
        it != end; ++it)
    {
        KeyFrame* pKFi = mpKFid[it->first];
        mConnectedKeyFrameWeights[pKFi] = it->second;
    }

    // Restore parent KeyFrame
    if(mBackupParentId>=0)
        mpParent = mpKFid[mBackupParentId];

    // KeyFrame childrens
    mspChildrens.clear();
    for(vector<long unsigned int>::const_iterator it = mvBackupChildrensId.begin(), end = mvBackupChildrensId.end(); it!=end; ++it)
    {
        mspChildrens.insert(mpKFid[*it]);
    }

    // Loop edge KeyFrame
    mspLoopEdges.clear();
    for(vector<long unsigned int>::const_iterator it = mvBackupLoopEdgesId.begin(), end = mvBackupLoopEdgesId.end(); it != end; ++it)
    {
        mspLoopEdges.insert(mpKFid[*it]);
    }

    // Merge edge KeyFrame
    mspMergeEdges.clear();
    for(vector<long unsigned int>::const_iterator it = mvBackupMergeEdgesId.begin(), end = mvBackupMergeEdgesId.end(); it != end; ++it)
    {
        mspMergeEdges.insert(mpKFid[*it]);
    }

    //Camera data
    if(mnBackupIdCamera >= 0)
    {
        mpCamera = mpCamId[mnBackupIdCamera];
    }
    else
    {
        cout << "ERROR: There is not a main camera in KF " << mnId << endl;
    }
    if(mnBackupIdCamera2 >= 0)
    {
        mpCamera2 = mpCamId[mnBackupIdCamera2];
    }

    //Inertial data
    if(mBackupPrevKFId != -1)
    {
        mPrevKF = mpKFid[mBackupPrevKFId];
    }
    if(mBackupNextKFId != -1)
    {
        mNextKF = mpKFid[mBackupNextKFId];
    }
    mpImuPreintegrated = &mBackupImuPreintegrated;


    // Remove all backup container
    mvBackupMapPointsId.clear();
    mBackupConnectedKeyFrameIdWeights.clear();
    mvBackupChildrensId.clear();
    mvBackupLoopEdgesId.clear();

    UpdateBestCovisibles();
}

bool KeyFrame::ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    // 3D in camera coordinates
    Eigen::Vector3f Pc = mRcw * P + mTcw.translation();
    float &PcX = Pc(0);
    float &PcY = Pc(1);
    float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        cout << "Negative depth: " << PcZ << endl;
        return false;
    }

    // Project in image and check it is not outside
    float invz = 1.0f/PcZ;
    u=fx*PcX*invz+cx;
    v=fy*PcY*invz+cy;

    // cout << "c";

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    float x = (u - cx) * invfx;
    float y = (v - cy) * invfy;
    float r2 = x * x + y * y;
    float k1 = mDistCoef.at<float>(0);
    float k2 = mDistCoef.at<float>(1);
    float p1 = mDistCoef.at<float>(2);
    float p2 = mDistCoef.at<float>(3);
    float k3 = 0;
    if(mDistCoef.total() == 5)
    {
        k3 = mDistCoef.at<float>(4);
    }

    // Radial distorsion
    float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    float u_distort = x_distort * fx + cx;
    float v_distort = y_distort * fy + cy;

    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

bool KeyFrame::ProjectPointUnDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    // 3D in camera coordinates
    Eigen::Vector3f Pc = mRcw * P + mTcw.translation();
    float &PcX = Pc(0);
    float &PcY= Pc(1);
    float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        cout << "Negative depth: " << PcZ << endl;
        return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    u = fx * PcX * invz + cx;
    v = fy * PcY * invz + cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    kp = cv::Point2f(u, v);

    return true;
}

Sophus::SE3f KeyFrame::GetRelativePoseTrl()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTrl;
}

Sophus::SE3f KeyFrame::GetRelativePoseTlr()
{
    unique_lock<mutex> lock(mMutexPose);
    return mTlr;
}

Sophus::SE3<float> KeyFrame::GetRightPose() {
    unique_lock<mutex> lock(mMutexPose);

    return mTrl * mTcw;
}

Sophus::SE3<float> KeyFrame::GetRightPoseInverse() {
    unique_lock<mutex> lock(mMutexPose);

    return mTwc * mTlr;
}

Eigen::Vector3f KeyFrame::GetRightCameraCenter() {
    unique_lock<mutex> lock(mMutexPose);

    return (mTwc * mTlr).translation();
}

Eigen::Matrix<float,3,3> KeyFrame::GetRightRotation() {
    unique_lock<mutex> lock(mMutexPose);

    return (mTrl.so3() * mTcw.so3()).matrix();
}

Eigen::Vector3f KeyFrame::GetRightTranslation() {
    unique_lock<mutex> lock(mMutexPose);
    return (mTrl * mTcw).translation();
}

void KeyFrame::SetORBVocabulary(ORBVocabulary* pORBVoc)
{
    mpORBvocabulary = pORBVoc;
}

void KeyFrame::SetKeyFrameDatabase(KeyFrameDatabase* pKFDB)
{
    mpKeyFrameDB = pKFDB;
}

} //namespace ORB_SLAM
