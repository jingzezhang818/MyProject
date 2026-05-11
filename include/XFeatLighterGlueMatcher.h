#ifndef XFEAT_LIGHTERGLUE_MATCHER_SLAM_H
#define XFEAT_LIGHTERGLUE_MATCHER_SLAM_H

#include <string>
#include <vector>

#include <torch/torch.h>

#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "XFeatLighterGlue/matcher.hpp"

namespace ORB_SLAM3
{

class XFeatLighterGlueMatcher
{
public:
    struct Stats
    {
        long unsigned int frame_id = 0;
        long unsigned int ref_kf_id = 0;
        long unsigned int last_frame_id = 0;
        int N_ref = 0;
        int N_last = 0;
        int N_cur = 0;
        int lg_raw_matches = 0;
        int mp_valid = 0;
        int proj_gate_keep = 0;
        int one_to_one = 0;
    };

    explicit XFeatLighterGlueMatcher(const std::string& weightPath = std::string());

    int SearchByLightGlue(KeyFrame* pKF,
                          Frame& F,
                          std::vector<MapPoint*>& vpMapPointMatches,
                          float minConf);

    int SearchByLightGlue(const Frame& LastFrame,
                          Frame& CurrentFrame,
                          std::vector<MapPoint*>& vpMapPointMatches,
                          float minConf);

    const Stats& LastStats() const { return lastStats_; }

private:
    static torch::Device SelectDevice();
    static torch::Tensor KeyPointsToTensor(const std::vector<cv::KeyPoint>& keypoints,
                                           int maxRows,
                                           const torch::Device& device);
    static torch::Tensor DescriptorsToTensor(const cv::Mat& descriptors,
                                             int maxRows,
                                             const torch::Device& device);
    static torch::Tensor ImageSizeToTensor(float width,
                                           float height,
                                           const torch::Device& device);

    torch::Device device_{torch::kCPU};
    XFeatLighterGlue matcher_;
    Stats lastStats_;
};

} // namespace ORB_SLAM3

#endif // XFEAT_LIGHTERGLUE_MATCHER_SLAM_H
