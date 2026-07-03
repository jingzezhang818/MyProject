#include "XFeatKeyFrameDatabase.h"

#include "KeyFrame.h"
#include "Map.h"

#include <algorithm>
#include <cstdlib>
#include <set>

namespace ORB_SLAM3
{
namespace
{

int GetEnvIntInRange(const char* name, const int fallback, const int minValue, const int maxValue)
{
    const char* env = std::getenv(name);
    if(!env || env[0] == '\0')
        return fallback;

    try
    {
        const int value = std::stoi(std::string(env));
        return std::max(minValue, std::min(maxValue, value));
    }
    catch(...)
    {
        return fallback;
    }
}

bool LoopGroupGreater(const XFeatKeyFrameDatabase::LoopCandidateGroup& a,
                      const XFeatKeyFrameDatabase::LoopCandidateGroup& b)
{
    if(a.groupScore != b.groupScore)
        return a.groupScore > b.groupScore;
    if(a.representativeScore != b.representativeScore)
        return a.representativeScore > b.representativeScore;
    if(!a.pRepresentative || !b.pRepresentative)
        return a.pRepresentative < b.pRepresentative;
    return a.pRepresentative->mnId < b.pRepresentative->mnId;
}

} // namespace

XFeatKeyFrameDatabase::XFeatKeyFrameDatabase(const XFeatVocabulary& voc):
    mpVoc(&voc),
    mnMaxFeaturesPerKeyFrame(GetEnvIntInRange("XFEAT_VOCAB_MAX_FEATURES_PER_KF", 512, 0, 4096))
{
    mvInvertedFile.resize(voc.size());
}

void XFeatKeyFrameDatabase::add(KeyFrame* pKF)
{
    if(!mpVoc || !pKF || pKF->isBad())
        return;

    pKF->ComputeXFeatBoW(mpVoc, mnMaxFeaturesPerKeyFrame);
    if(pKF->mXFeatBowVec.empty())
        return;

    std::unique_lock<std::mutex> lock(mMutex);
    for(DBoW3::BowVector::const_iterator vit = pKF->mXFeatBowVec.begin(),
        vend = pKF->mXFeatBowVec.end(); vit != vend; ++vit)
    {
        const size_t wordId = static_cast<size_t>(vit->first);
        if(wordId < mvInvertedFile.size())
            mvInvertedFile[wordId].push_back(pKF);
    }
}

void XFeatKeyFrameDatabase::erase(KeyFrame* pKF)
{
    if(!pKF)
        return;

    std::unique_lock<std::mutex> lock(mMutex);
    for(DBoW3::BowVector::const_iterator vit = pKF->mXFeatBowVec.begin(),
        vend = pKF->mXFeatBowVec.end(); vit != vend; ++vit)
    {
        const size_t wordId = static_cast<size_t>(vit->first);
        if(wordId >= mvInvertedFile.size())
            continue;

        std::list<KeyFrame*>& lKFs = mvInvertedFile[wordId];
        for(std::list<KeyFrame*>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; ++lit)
        {
            if(*lit == pKF)
            {
                lKFs.erase(lit);
                break;
            }
        }
    }
}

void XFeatKeyFrameDatabase::clear()
{
    std::unique_lock<std::mutex> lock(mMutex);
    mvInvertedFile.clear();
    if(mpVoc)
        mvInvertedFile.resize(mpVoc->size());
}

void XFeatKeyFrameDatabase::clearMap(Map* pMap)
{
    if(!pMap)
        return;

    std::unique_lock<std::mutex> lock(mMutex);
    for(std::list<KeyFrame*>& lKFs : mvInvertedFile)
    {
        for(std::list<KeyFrame*>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend;)
        {
            KeyFrame* pKFi = *lit;
            if(pKFi && pKFi->GetMap() == pMap)
                lit = lKFs.erase(lit);
            else
                ++lit;
        }
    }
}

void XFeatKeyFrameDatabase::DetectNBestLoopCandidates(KeyFrame* pKF,
                                                       std::vector<KeyFrame*>& vpLoopCand,
                                                       int nNumCandidates)
{
    vpLoopCand.clear();
    if(nNumCandidates <= 0)
        return;

    std::vector<LoopCandidateGroup> vLoopGroups;
    DetectNBestLoopCandidateGroups(pKF, vLoopGroups, nNumCandidates);
    vpLoopCand.reserve(vLoopGroups.size());
    for(const LoopCandidateGroup& group : vLoopGroups)
    {
        if(group.pRepresentative)
            vpLoopCand.push_back(group.pRepresentative);
    }
}

void XFeatKeyFrameDatabase::DetectLoopCandidateGroups(KeyFrame* pKF,
                                                      std::vector<LoopCandidateGroup>& vLoopGroups)
{
    vLoopGroups.clear();
    {
        std::unique_lock<std::mutex> debugLock(mMutexDebug);
        mvLastDebugInfo.clear();
        mnLastMaxCommonWords = 0;
        mnLastMinCommonWords = 0;
        mnLastSharingWords = 0;
    }

    if(!mpVoc || !pKF || pKF->isBad())
        return;

    pKF->ComputeXFeatBoW(mpVoc, mnMaxFeaturesPerKeyFrame);
    if(pKF->mXFeatBowVec.empty())
        return;

    std::map<KeyFrame*, int> mCommonWords;
    std::set<KeyFrame*> spConnectedKF = pKF->GetConnectedKeyFrames();
    {
        std::unique_lock<std::mutex> lock(mMutex);
        for(DBoW3::BowVector::const_iterator vit = pKF->mXFeatBowVec.begin(),
            vend = pKF->mXFeatBowVec.end(); vit != vend; ++vit)
        {
            const size_t wordId = static_cast<size_t>(vit->first);
            if(wordId >= mvInvertedFile.size())
                continue;

            const std::list<KeyFrame*>& lKFs = mvInvertedFile[wordId];
            for(KeyFrame* pKFi : lKFs)
            {
                if(!pKFi || pKFi == pKF || pKFi->isBad())
                    continue;
                if(pKFi->GetMap() != pKF->GetMap())
                    continue;
                if(spConnectedKF.count(pKFi))
                    continue;
                ++mCommonWords[pKFi];
            }
        }
    }

    if(mCommonWords.empty())
    {
        std::unique_lock<std::mutex> debugLock(mMutexDebug);
        mvLastDebugInfo.clear();
        mnLastMaxCommonWords = 0;
        mnLastMinCommonWords = 0;
        mnLastSharingWords = 0;
        return;
    }

    int maxCommonWords = 0;
    for(const auto& item : mCommonWords)
        maxCommonWords = std::max(maxCommonWords, item.second);
    const int minCommonWords = std::max(static_cast<int>(maxCommonWords * 0.5f), 8);

    std::map<KeyFrame*, float> mScore;
    for(const auto& item : mCommonWords)
    {
        KeyFrame* pKFi = item.first;
        if(!pKFi || pKFi->mXFeatBowVec.empty())
            continue;
        if(item.second < minCommonWords)
            continue;

        const float score = static_cast<float>(mpVoc->score(pKF->mXFeatBowVec, pKFi->mXFeatBowVec));
        mScore[pKFi] = score;
    }

    if(mScore.empty())
    {
        std::unique_lock<std::mutex> debugLock(mMutexDebug);
        mvLastDebugInfo.clear();
        mnLastMaxCommonWords = maxCommonWords;
        mnLastMinCommonWords = minCommonWords;
        mnLastSharingWords = static_cast<int>(mCommonWords.size());
        return;
    }

    constexpr int kGroupNeighborCount = 10;
    constexpr int kGroupNeighborMinWeight = 10;
    constexpr int kGroupScoreFrameCount = 5;
    std::map<KeyFrame*, LoopCandidateGroup> mBestGroupByRepresentative;
    for(const auto& item : mScore)
    {
        KeyFrame* pKFi = item.first;
        const float baseScore = item.second;
        std::map<KeyFrame*, LoopCandidateGroupMember> mMembers;
        mMembers[pKFi] = LoopCandidateGroupMember{pKFi, mCommonWords[pKFi], baseScore};

        const std::vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(kGroupNeighborCount);
        for(KeyFrame* pKF2 : vpNeighs)
        {
            if(!pKF2 || pKFi->GetWeight(pKF2) <= kGroupNeighborMinWeight)
                continue;
            const auto scoreIt = mScore.find(pKF2);
            if(scoreIt == mScore.end())
                continue;
            mMembers[pKF2] = LoopCandidateGroupMember{pKF2, mCommonWords[pKF2], scoreIt->second};
        }

        std::vector<LoopCandidateGroupMember> vMembers;
        vMembers.reserve(mMembers.size());
        for(const auto& member : mMembers)
            vMembers.push_back(member.second);

        std::sort(vMembers.begin(), vMembers.end(),
                  [](const LoopCandidateGroupMember& a, const LoopCandidateGroupMember& b) {
                      if(a.score != b.score)
                          return a.score > b.score;
                      if(a.commonWords != b.commonWords)
                          return a.commonWords > b.commonWords;
                      if(!a.pKF || !b.pKF)
                          return a.pKF < b.pKF;
                      return a.pKF->mnId < b.pKF->mnId;
                  });

        if(vMembers.empty() || !vMembers.front().pKF)
            continue;

        LoopCandidateGroup group;
        group.pRepresentative = vMembers.front().pKF;
        group.commonWords = vMembers.front().commonWords;
        group.representativeScore = vMembers.front().score;
        group.vMembers = vMembers;
        const int nScoreFrames = std::min<int>(kGroupScoreFrameCount, vMembers.size());
        for(int i = 0; i < nScoreFrames; ++i)
        {
            group.groupScore += vMembers[i].score;
            if(vMembers[i].pKF)
                group.vpGroupKFs.push_back(vMembers[i].pKF);
        }

        auto itGroup = mBestGroupByRepresentative.find(group.pRepresentative);
        if(itGroup == mBestGroupByRepresentative.end() ||
           LoopGroupGreater(group, itGroup->second))
            mBestGroupByRepresentative[group.pRepresentative] = group;
    }

    vLoopGroups.reserve(mBestGroupByRepresentative.size());
    for(const auto& item : mBestGroupByRepresentative)
        vLoopGroups.push_back(item.second);

    std::sort(vLoopGroups.begin(), vLoopGroups.end(), LoopGroupGreater);

    if(vLoopGroups.size() > 3 && !vLoopGroups.empty())
    {
        const float minGroupScore = vLoopGroups.front().groupScore * 0.5f;
        vLoopGroups.erase(std::remove_if(vLoopGroups.begin(),
                                         vLoopGroups.end(),
                                         [minGroupScore](const LoopCandidateGroup& group) {
                                             return group.groupScore < minGroupScore;
                                         }),
                          vLoopGroups.end());
    }

    std::set<KeyFrame*> spAlreadyAddedKF;
    std::vector<LoopCandidateGroup> vUniqueGroups;
    vUniqueGroups.reserve(vLoopGroups.size());
    for(LoopCandidateGroup& group : vLoopGroups)
    {
        KeyFrame* pKFi = group.pRepresentative;
        if(!pKFi || pKFi->isBad() || spAlreadyAddedKF.count(pKFi))
            continue;
        spAlreadyAddedKF.insert(pKFi);
        vUniqueGroups.push_back(group);
    }
    vLoopGroups.swap(vUniqueGroups);

    std::vector<CandidateDebugInfo> debugInfo;
    debugInfo.reserve(vLoopGroups.size());
    for(const LoopCandidateGroup& group : vLoopGroups)
    {
        if(!group.pRepresentative)
            continue;
        debugInfo.push_back(CandidateDebugInfo{group.pRepresentative->mnId,
                                               group.commonWords,
                                               group.representativeScore,
                                               group.groupScore,
                                               static_cast<int>(group.vpGroupKFs.size())});
    }
    std::sort(debugInfo.begin(), debugInfo.end(), [](const CandidateDebugInfo& a,
                                                     const CandidateDebugInfo& b) {
        if(a.accumulatedScore != b.accumulatedScore)
            return a.accumulatedScore > b.accumulatedScore;
        if(a.score != b.score)
            return a.score > b.score;
        return a.keyframeId < b.keyframeId;
    });
    if(debugInfo.size() > 10)
        debugInfo.resize(10);

    std::unique_lock<std::mutex> debugLock(mMutexDebug);
    mvLastDebugInfo = debugInfo;
    mnLastMaxCommonWords = maxCommonWords;
    mnLastMinCommonWords = minCommonWords;
    mnLastSharingWords = static_cast<int>(mCommonWords.size());
}

void XFeatKeyFrameDatabase::DetectNBestLoopCandidateGroups(KeyFrame* pKF,
                                                           std::vector<LoopCandidateGroup>& vLoopGroups,
                                                           int nNumCandidates)
{
    vLoopGroups.clear();
    if(nNumCandidates <= 0)
        return;

    DetectLoopCandidateGroups(pKF, vLoopGroups);
    if(static_cast<int>(vLoopGroups.size()) > nNumCandidates)
        vLoopGroups.resize(static_cast<size_t>(nNumCandidates));
}

std::vector<XFeatKeyFrameDatabase::CandidateDebugInfo> XFeatKeyFrameDatabase::GetLastDebugInfo() const
{
    std::unique_lock<std::mutex> lock(mMutexDebug);
    return mvLastDebugInfo;
}

int XFeatKeyFrameDatabase::GetLastMaxCommonWords() const
{
    std::unique_lock<std::mutex> lock(mMutexDebug);
    return mnLastMaxCommonWords;
}

int XFeatKeyFrameDatabase::GetLastMinCommonWords() const
{
    std::unique_lock<std::mutex> lock(mMutexDebug);
    return mnLastMinCommonWords;
}

int XFeatKeyFrameDatabase::GetLastSharingWords() const
{
    std::unique_lock<std::mutex> lock(mMutexDebug);
    return mnLastSharingWords;
}

} // namespace ORB_SLAM3
