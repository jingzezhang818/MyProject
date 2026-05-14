#ifndef XFEAT_LIGHTERGLUE_MATCHER_SLAM_H
#define XFEAT_LIGHTERGLUE_MATCHER_SLAM_H

#include <string>
#include <unordered_map>
#include <utility>
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
                          float minConf,
                          std::vector<float>* vMatchScores = nullptr);

    int SearchByLightGlue(const Frame& LastFrame,
                          Frame& CurrentFrame,
                          std::vector<MapPoint*>& vpMapPointMatches,
                          float minConf,
                          std::vector<float>* vMatchScores = nullptr);

    int SearchForTriangulation(KeyFrame* pKF1,
                               KeyFrame* pKF2,
                               std::vector<std::pair<size_t, size_t>>& vMatchedPairs,
                               bool bOnlyStereo,
                               bool bCoarse,
                               float minConf);

    const Stats& LastStats() const { return lastStats_; }

private:
    struct CachedInputTensors
    {
        bool valid = false;
        long unsigned int id = 0;
        int rows = 0;
        int descRows = 0;
        int descCols = 0;
        const unsigned char* descData = nullptr;
        float width = 0.0f;
        float height = 0.0f;
        torch::Tensor kpts;
        torch::Tensor desc;
        torch::Tensor size;
    };

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
    static int GetKeyFrameCacheLimit();
    static bool IsCacheHit(const CachedInputTensors& cached,
                           long unsigned int id,
                           int rows,
                           const cv::Mat& descriptors,
                           float width,
                           float height);
    static CachedInputTensors BuildInputTensors(long unsigned int id,
                                                const std::vector<cv::KeyPoint>& keypoints,
                                                const cv::Mat& descriptors,
                                                int rows,
                                                float width,
                                                float height,
                                                const torch::Device& device);
    const CachedInputTensors& GetKeyFrameInputTensors(const KeyFrame& keyFrame,
                                                      int rows,
                                                      float width,
                                                      float height);
    const CachedInputTensors& GetFrameInputTensors(const Frame& frame,
                                                   int rows,
                                                   float width,
                                                   float height);
    void TrimKeyFrameCache();
    void TrimFrameCache();

    torch::Device device_{torch::kCPU};
    XFeatLighterGlue matcher_;
    Stats lastStats_;
    std::unordered_map<long unsigned int, CachedInputTensors> keyFrameInputCache_;
    std::vector<long unsigned int> keyFrameCacheOrder_;
    std::unordered_map<long unsigned int, CachedInputTensors> frameInputCache_;
    std::vector<long unsigned int> frameCacheOrder_;
};

} // namespace ORB_SLAM3

#endif // XFEAT_LIGHTERGLUE_MATCHER_SLAM_H
