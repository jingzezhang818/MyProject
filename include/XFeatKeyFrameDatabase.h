/**
* XFeat DBoW3 keyframe database used only for loop candidate retrieval.
*/

#ifndef XFEATKEYFRAMEDATABASE_H
#define XFEATKEYFRAMEDATABASE_H

#include "XFeatVocabulary.h"

#include <list>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace ORB_SLAM3
{

class KeyFrame;
class Map;

class XFeatKeyFrameDatabase
{
public:
    struct CandidateDebugInfo
    {
        long unsigned int keyframeId = 0;
        int commonWords = 0;
        float score = 0.0f;
        float accumulatedScore = 0.0f;
    };

    XFeatKeyFrameDatabase() = default;
    explicit XFeatKeyFrameDatabase(const XFeatVocabulary& voc);

    void add(KeyFrame* pKF);
    void erase(KeyFrame* pKF);
    void clear();
    void clearMap(Map* pMap);

    void DetectNBestLoopCandidates(KeyFrame* pKF,
                                   std::vector<KeyFrame*>& vpLoopCand,
                                   int nNumCandidates);

    std::vector<CandidateDebugInfo> GetLastDebugInfo() const;
    int GetLastMaxCommonWords() const;
    int GetLastMinCommonWords() const;
    int GetLastSharingWords() const;

private:
    const XFeatVocabulary* mpVoc = nullptr;
    std::vector<std::list<KeyFrame*> > mvInvertedFile;
    int mnMaxFeaturesPerKeyFrame = 512;

    mutable std::mutex mMutex;
    mutable std::mutex mMutexDebug;
    std::vector<CandidateDebugInfo> mvLastDebugInfo;
    int mnLastMaxCommonWords = 0;
    int mnLastMinCommonWords = 0;
    int mnLastSharingWords = 0;
};

} // namespace ORB_SLAM3

#endif // XFEATKEYFRAMEDATABASE_H
