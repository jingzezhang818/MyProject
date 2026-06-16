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
    struct ProjectionMapPointStats
    {
        int close_tracking_stereo_reject_ur = 0;
        int close_tracking_stereo_reject_depth = 0;
        int close_tracking_stereo_accepted = 0;
        int local_proj_triang_0_10_proj_outside_reject = 0;
        int local_proj_triang_0_10_companion_outside_reject = 0;
        int ref_target_input = 0;
        int ref_target_skip_not_in_view = 0;
        int ref_target_skip_far = 0;
        int ref_target_skip_bad = 0;
        int ref_target_skip_no_desc = 0;
        int ref_target_projected = 0;
        int ref_target_side_attempt = 0;
        int ref_target_invalid_level = 0;
        int ref_target_empty_area = 0;
        int ref_target_area_candidates = 0;
        int ref_target_invalid_candidate = 0;
        int ref_target_occupied_candidate = 0;
        int ref_target_geom_ur_reject = 0;
        int ref_target_geom_depth_reject = 0;
        int ref_target_valid_candidate = 0;
        int ref_target_no_valid_candidate = 0;
        int ref_target_raw_candidate = 0;
        int ref_target_dist_reject = 0;
        int ref_target_ratio_reject = 0;
        int ref_target_candidate_pool = 0;
        int ref_target_spatial_reject = 0;
        int ref_target_assign_reject = 0;
        int ref_target_outside_reject = 0;
        int ref_target_assigned = 0;
    };

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
                           const float thHighOverride = -1.0f,
                           const int maxNewMatches = -1,
                           const bool bResetDiagnostics = true);

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

    const ProjectionMapPointStats& LastProjectionMapPointStats() const { return mLastProjectionMapPointStats; }

public:
    static const float TH_LOW;
    static const float TH_HIGH;
    static const int HISTO_LENGTH;

protected:
    float RadiusByViewingCos(const float &viewCos);
    void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);

    float mfNNratio;
    bool mbCheckOrientation;
    ProjectionMapPointStats mLastProjectionMapPointStats;
};

} // namespace ORB_SLAM3

#endif // XFEATMATCHER_H
