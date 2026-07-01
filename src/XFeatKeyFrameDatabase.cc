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

bool AccScoreGreater(const std::pair<float, KeyFrame*>& a, const std::pair<float, KeyFrame*>& b)
{
    if(a.first != b.first)
        return a.first > b.first;
    if(!a.second || !b.second)
        return a.second < b.second;
    return a.second->mnId < b.second->mnId;
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
    if(!mpVoc || !pKF || pKF->isBad() || nNumCandidates <= 0)
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
    const int minCommonWords = static_cast<int>(maxCommonWords * 0.8f);

    std::map<KeyFrame*, float> mScore;
    std::vector<CandidateDebugInfo> debugInfo;
    debugInfo.reserve(mCommonWords.size());
    for(const auto& item : mCommonWords)
    {
        KeyFrame* pKFi = item.first;
        if(!pKFi || pKFi->mXFeatBowVec.empty())
            continue;
        if(item.second <= minCommonWords)
            continue;

        const float score = static_cast<float>(mpVoc->score(pKF->mXFeatBowVec, pKFi->mXFeatBowVec));
        mScore[pKFi] = score;
        debugInfo.push_back(CandidateDebugInfo{pKFi->mnId, item.second, score, score});
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

    std::vector<std::pair<float, KeyFrame*> > vAccScoreAndMatch;
    vAccScoreAndMatch.reserve(mScore.size());
    for(const auto& item : mScore)
    {
        KeyFrame* pKFi = item.first;
        const float baseScore = item.second;
        float accScore = baseScore;
        float bestScore = baseScore;
        KeyFrame* pBestKF = pKFi;

        const std::vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);
        for(KeyFrame* pKF2 : vpNeighs)
        {
            const auto scoreIt = mScore.find(pKF2);
            if(scoreIt == mScore.end())
                continue;
            accScore += scoreIt->second;
            if(scoreIt->second > bestScore)
            {
                bestScore = scoreIt->second;
                pBestKF = pKF2;
            }
        }
        vAccScoreAndMatch.push_back(std::make_pair(accScore, pBestKF));
        for(CandidateDebugInfo& info : debugInfo)
        {
            if(info.keyframeId == pBestKF->mnId)
            {
                info.accumulatedScore = std::max(info.accumulatedScore, accScore);
                break;
            }
        }
    }

    std::sort(vAccScoreAndMatch.begin(), vAccScoreAndMatch.end(), AccScoreGreater);

    std::set<KeyFrame*> spAlreadyAddedKF;
    vpLoopCand.reserve(static_cast<size_t>(nNumCandidates));
    for(const std::pair<float, KeyFrame*>& item : vAccScoreAndMatch)
    {
        KeyFrame* pKFi = item.second;
        if(!pKFi || pKFi->isBad() || spAlreadyAddedKF.count(pKFi))
            continue;
        vpLoopCand.push_back(pKFi);
        spAlreadyAddedKF.insert(pKFi);
        if(static_cast<int>(vpLoopCand.size()) >= nNumCandidates)
            break;
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
