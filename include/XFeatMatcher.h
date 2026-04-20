#ifndef XFEATMATCHER_H
#define XFEATMATCHER_H

#include <vector>
#include <set>
#include <utility>

#include <opencv2/core/core.hpp>

#include "MapPoint.h"
#include "KeyFrame.h"
#include "Frame.h"

namespace ORB_SLAM3
{

class XFeatMatcher
{
public:
    XFeatMatcher(float nnratio = 0.8f, bool checkOri = false);

    // Computes the L2 distance between two float descriptors.
    static float DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    // Matching for map initialization (monocular).
    int SearchForInitialization(Frame &F1, Frame &F2,
                                std::vector<cv::Point2f> &vbPrevMatched,
                                std::vector<int> &vnMatches12,
                                int windowSize = 10);

    // Matching to triangulate new MapPoints. Checks epipolar constraint.
    int SearchForTriangulation(KeyFrame *pKF1,
                               KeyFrame *pKF2,
                               std::vector<std::pair<size_t, size_t>> &vMatchedPairs,
                               const bool bOnlyStereo,
                               const bool bCoarse = false);

    // Search matches between projected MapPoints and frame keypoints.
    int SearchByProjection(Frame &F,
                           const std::vector<MapPoint*> &vpMapPoints,
                           const float th = 3.0f,
                           const bool bFarPoints = false,
                           const float thFarPoints = 50.0f,
                           const float thHighOverride = -1.0f);

    // Project points tracked in last frame to current frame and search matches.
    int SearchByProjection(Frame &CurrentFrame,
                           const Frame &LastFrame,
                           const float th,
                           const bool bMono,
                           const float thHighOverride = -1.0f);

    // Project MapPoints from a keyframe into current frame and search matches.
    // Used in relocalization refinement.
    int SearchByProjection(Frame &CurrentFrame,
                           KeyFrame* pKF,
                           const std::set<MapPoint*> &sAlreadyFound,
                           const float th,
                           const float thHighOverride = -1.0f);

    // Search matches between map points in a keyframe and frame descriptors via NN.
    int SearchByNN(KeyFrame *pKF,
                   Frame &F,
                   std::vector<MapPoint*> &vpMapPointMatches,
                   const float thHighOverride = -1.0f);

public:
    static const float TH_LOW;
    static const float TH_HIGH;
    static const int HISTO_LENGTH;

protected:
    float RadiusByViewingCos(const float &viewCos);
    void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);

    float mfNNratio;
    bool mbCheckOrientation;
};

} // namespace ORB_SLAM3

#endif // XFEATMATCHER_H
