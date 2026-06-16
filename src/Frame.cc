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

#include "Frame.h"

#include "G2oTypes.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "ORBextractor.h"
#include "XFextractor.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include "GeometricCamera.h"

#include <thread>
#include <include/CameraModels/Pinhole.h>
#include <include/CameraModels/KannalaBrandt8.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace ORB_SLAM3
{
namespace
{
    //调试: Frame 级调试开关，跟随统一 XFeat 调试开关。
    bool IsXFeatFrameDebugEnabled()
    {
        static const bool enabled = []() {
            const char* env = std::getenv("XFEAT_DEBUG");
            if(!env)
                return false;
            const std::string v(env);
            return !(v.empty() || v == "0" || v == "false" || v == "FALSE");
        }();
        return enabled;
    }

    int GetXFeatDiagInterval()
    {
        static const int interval = []() {
            int value = 1;
            const char* env = std::getenv("XFEAT_DIAG_INTERVAL");
            if(env)
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
        }();
        return interval;
    }

    bool ShouldPrintXFeatFrameDebug(const long unsigned int frameId)
    {
        if(!IsXFeatFrameDebugEnabled())
            return false;
        return (frameId % static_cast<long unsigned int>(GetXFeatDiagInterval())) == 0;
    }

    float GetXFeatStereoDescThreshold()
    {
        static const float threshold = []() {
            float value = 1.35f;
            const char* env = std::getenv("XFEAT_STEREO_DESC_TH");
            if(env && env[0] != '\0')
            {
                char* end = nullptr;
                const float parsed = std::strtof(env, &end);
                if(end != env && std::isfinite(parsed))
                    value = parsed;
            }
            return std::max(0.05f, std::min(2.0f, value));
        }();
        return threshold;
    }

    float GetXFeatStereoMinDisparity()
    {
        //最小视差阈值(像素),过滤远距离"假立体"点.
        //视差小于该阈值时深度不确定性极高,立体观测退化为噪声.
        //对于SePT01(fx=610,基线=0.15m): 1.0px视差→深度≈91.5m, 1.5px→61.0m, 2.0px→45.8m.
        static const float minDisp = []() {
            float value = 1.0f;
            const char* env = std::getenv("XFEAT_STEREO_MIN_DISPARITY");
            if(env && env[0] != '\0')
            {
                char* end = nullptr;
                const float parsed = std::strtof(env, &end);
                if(end != env && std::isfinite(parsed))
                    value = parsed;
            }
            return std::max(0.0f, std::min(100.0f, value));
        }();
        return minDisp;
    }

    float GetXFeatStereoMaxYError()
    {
        // XFeat双目极线一致性阈值,会按keypoint octave scale放大. <=0关闭该过滤.
        static const float maxYErr = []() {
            float value = 0.0f;
            const char* env = std::getenv("XFEAT_STEREO_MAX_Y_ERR");
            if(env && env[0] != '\0')
            {
                char* end = nullptr;
                const float parsed = std::strtof(env, &end);
                if(end != env && std::isfinite(parsed))
                    value = parsed;
            }
            return std::max(0.0f, std::min(20.0f, value));
        }();
        return maxYErr;
    }

    float GetXFeatStereoMaxRelativeDepthSigma()
    {
        // sigma_z / z ~= sigma_disp / disparity. <=0关闭该过滤.
        static const float maxRelSigma = []() {
            float value = 0.0f;
            const char* env = std::getenv("XFEAT_STEREO_MAX_REL_DEPTH_SIGMA");
            if(env && env[0] != '\0')
            {
                char* end = nullptr;
                const float parsed = std::strtof(env, &end);
                if(end != env && std::isfinite(parsed))
                    value = parsed;
            }
            return std::max(-1.0f, std::min(10.0f, value));
        }();
        return maxRelSigma;
    }

    float GetXFeatStereoBirthCloseMaxYError()
    {
        const float base = GetXFeatStereoMaxYError();
        if(base > 0.0f)
            return std::max(0.1f, 0.5f * base);
        return 1.0f;
    }

    float GetXFeatStereoBirthCloseMaxRelativeDepthSigma()
    {
        const float base = GetXFeatStereoMaxRelativeDepthSigma();
        if(base > 0.0f)
            return std::max(1e-3f, 0.5f * base);
        return 0.15f;
    }

    bool IsXFeatStereoSubpixelEnabled()
    {
        static const bool enabled = []() {
            const char* env = std::getenv("XFEAT_STEREO_SUBPIXEL");
            if(!env)
                return false;
            const std::string value(env);
            return !(value.empty() || value == "0" || value == "false" || value == "FALSE");
        }();
        return enabled;
    }

    float GetXFeatStereoRatio()
    {
        //Ratio test阈值,用于过滤歧义立体匹配.
        //bestDist < secondBestDist × ratio 才接受. 设为0关闭ratio test.
        static const float ratio = []() {
            float value = 0.80f;
            const char* env = std::getenv("XFEAT_STEREO_RATIO");
            if(env && env[0] != '\0')
            {
                char* end = nullptr;
                const float parsed = std::strtof(env, &end);
                if(end != env && std::isfinite(parsed))
                    value = parsed;
            }
            return std::max(0.0f, std::min(1.0f, value));
        }();
        return ratio;
    }

    float PercentileFromSorted(const std::vector<float>& sortedValues, const float percentile)
    {
        if(sortedValues.empty())
            return -1.0f;
        const float clamped = std::max(0.0f, std::min(1.0f, percentile));
        const size_t idx = std::min(sortedValues.size() - 1,
                                    static_cast<size_t>(std::round(clamped * static_cast<float>(sortedValues.size() - 1))));
        return sortedValues[idx];
    }

    bool IsXFeatStereoMutualEnabled()
    {
        //左右互检开关,默认开启. 关闭: XFEAT_STEREO_MUTUAL=0
        static const bool enabled = []() {
            const char* env = std::getenv("XFEAT_STEREO_MUTUAL");
            if(!env || env[0] == '\0')
                return true;
            const std::string v(env);
            return !(v == "0" || v == "false" || v == "FALSE");
        }();
        return enabled;
    }

    bool IsFloatDescriptorMat(const cv::Mat& descriptors)
    {
        return !descriptors.empty() && descriptors.type() == CV_32F;
    }

    float DescriptorDistanceFloatPtr(const float* pa, const float* pb, const int length)
    {
        float dist2 = 0.0f;
        for(int i = 0; i < length; ++i)
        {
            const float d = pa[i] - pb[i];
            dist2 += d * d;
        }
        return std::sqrt(dist2);
    }

    float DescriptorDistanceFloatMat(const cv::Mat& a, const cv::Mat& b)
    {
        if(a.empty() || b.empty() || a.type() != CV_32F || b.type() != CV_32F)
            return std::numeric_limits<float>::infinity();

        const int lenA = static_cast<int>(a.total() * a.channels());
        const int lenB = static_cast<int>(b.total() * b.channels());
        if(lenA <= 0 || lenA != lenB)
            return std::numeric_limits<float>::infinity();

        if(a.isContinuous() && b.isContinuous())
            return DescriptorDistanceFloatPtr(a.ptr<float>(0), b.ptr<float>(0), lenA);

        return static_cast<float>(cv::norm(a, b, cv::NORM_L2));
    }

    float StereoDescriptorDistance(const cv::Mat& a, const cv::Mat& b, const bool bFloatDescriptors)
    {
        if(bFloatDescriptors)
            return DescriptorDistanceFloatMat(a, b);
        return static_cast<float>(ORBmatcher::DescriptorDistance(a, b));
    }

    float ZeroMeanPatchDistanceL1(const cv::Mat& patchA, const cv::Mat& patchB)
    {
        if(patchA.empty() || patchB.empty() || patchA.size() != patchB.size())
            return std::numeric_limits<float>::infinity();

        cv::Mat patchA32;
        cv::Mat patchB32;
        patchA.convertTo(patchA32, CV_32F);
        patchB.convertTo(patchB32, CV_32F);

        const cv::Scalar meanA = cv::mean(patchA32);
        const cv::Scalar meanB = cv::mean(patchB32);
        patchA32 -= meanA;
        patchB32 -= meanB;

        return static_cast<float>(cv::norm(patchA32, patchB32, cv::NORM_L1));
    }

    float ZeroMeanPatchZNCC(const cv::Mat& patchA, const cv::Mat& patchB)
    {
        if(patchA.empty() || patchB.empty() || patchA.size() != patchB.size())
            return -std::numeric_limits<float>::infinity();

        cv::Mat patchA32;
        cv::Mat patchB32;
        patchA.convertTo(patchA32, CV_32F);
        patchB.convertTo(patchB32, CV_32F);

        const cv::Scalar meanA = cv::mean(patchA32);
        const cv::Scalar meanB = cv::mean(patchB32);
        patchA32 -= meanA;
        patchB32 -= meanB;

        const double normA = cv::norm(patchA32, cv::NORM_L2);
        const double normB = cv::norm(patchB32, cv::NORM_L2);
        if(!std::isfinite(normA) || !std::isfinite(normB) || normA <= 1e-6 || normB <= 1e-6)
            return -std::numeric_limits<float>::infinity();

        const double zncc = patchA32.dot(patchB32) / (normA * normB);
        return static_cast<float>(zncc);
    }

    struct XFeatStereoSubpixelRefinement
    {
        bool success = false;
        float refinedUR = 0.0f;
        float cost1 = std::numeric_limits<float>::infinity();
        float cost2 = std::numeric_limits<float>::infinity();
        float cost3 = std::numeric_limits<float>::infinity();
        float combined1 = std::numeric_limits<float>::infinity();
        float combined2 = std::numeric_limits<float>::infinity();
        float combined3 = std::numeric_limits<float>::infinity();
        float score1 = -std::numeric_limits<float>::infinity();
        float score2 = -std::numeric_limits<float>::infinity();
        float score3 = -std::numeric_limits<float>::infinity();
        float curvatureCost = std::numeric_limits<float>::quiet_NaN();
        float marginCost = std::numeric_limits<float>::quiet_NaN();
        float curvatureCombined = std::numeric_limits<float>::quiet_NaN();
        float marginCombined = std::numeric_limits<float>::quiet_NaN();
        float peakScore = std::numeric_limits<float>::quiet_NaN();
    };

    struct XFeatStereoPatchSearchResult
    {
        bool success = false;
        float seedUR = 0.0f;
        float coarseWinnerUR = 0.0f;
        float secondBestUR = 0.0f;
        float winnerShift = 0.0f;
        float bandWidth = 0.0f;
        float bestCombined = std::numeric_limits<float>::infinity();
        float secondBestCombined = std::numeric_limits<float>::infinity();
        XFeatStereoSubpixelRefinement refinement;
    };

    float SmoothCombinedSample(const std::vector<float>& combineds, const int idx)
    {
        if(idx <= 0 || idx >= static_cast<int>(combineds.size()) - 1)
            return std::numeric_limits<float>::infinity();

        const float left = combineds[idx - 1];
        const float center = combineds[idx];
        const float right = combineds[idx + 1];
        if(!std::isfinite(left) || !std::isfinite(center) || !std::isfinite(right))
            return std::numeric_limits<float>::infinity();

        return 0.25f * left + 0.5f * center + 0.25f * right;
    }

    bool ComputeRobustFiniteRange(const std::vector<float>& values,
                                  const float /*lowerQuantile*/,
                                  const float /*upperQuantile*/,
                                  float& lower,
                                  float& upper)
    {
        lower = std::numeric_limits<float>::infinity();
        upper = -std::numeric_limits<float>::infinity();
        bool hasFinite = false;
        for(const float value : values)
        {
            if(std::isfinite(value))
            {
                lower = std::min(lower, value);
                upper = std::max(upper, value);
                hasFinite = true;
            }
        }

        if(!hasFinite)
            return false;

        return std::isfinite(lower) && std::isfinite(upper);
    }

    float NormalizeFiniteValueClamped(const float value, const float lower, const float upper)
    {
        if(!std::isfinite(value))
            return std::numeric_limits<float>::infinity();

        const float range = upper - lower;
        if(!std::isfinite(range) || range <= 1e-4f)
            return 0.5f;

        const float normalized = (value - lower) / range;
        return std::max(0.0f, std::min(1.0f, normalized));
    }

    float ComputeStereoPeakScore(const float score1, const float score2, const float score3)
    {
        if(!std::isfinite(score1) || !std::isfinite(score2) || !std::isfinite(score3))
            return -std::numeric_limits<float>::infinity();

        return score2 - std::max(score1, score3);
    }

    bool ExtractPatchAtFloatCenter(const cv::Mat& image,
                                   const cv::Point2f& center,
                                   const int halfWindow,
                                   cv::Mat& patch)
    {
        if(image.empty())
            return false;

        if(center.x - static_cast<float>(halfWindow) < 0.0f ||
           center.x + static_cast<float>(halfWindow) >= static_cast<float>(image.cols) ||
           center.y - static_cast<float>(halfWindow) < 0.0f ||
           center.y + static_cast<float>(halfWindow) >= static_cast<float>(image.rows))
        {
            return false;
        }

        cv::getRectSubPix(image, cv::Size(2 * halfWindow + 1, 2 * halfWindow + 1), center, patch);
        return !patch.empty();
    }

    bool RefineXFeatStereoSubpixel(const std::vector<cv::Mat>* pLeftPyramid,
                                   const std::vector<cv::Mat>* pRightPyramid,
                                   const std::vector<float>& scaleFactors,
                                   const std::vector<float>& invScaleFactors,
                                   const cv::KeyPoint& kpL,
                                   const cv::KeyPoint& kpR,
                                   XFeatStereoSubpixelRefinement& refinement)
    {
        if(!pLeftPyramid || !pRightPyramid)
            return false;

        const int level = kpL.octave;
        if(level < 0 ||
           level >= static_cast<int>(pLeftPyramid->size()) ||
           level >= static_cast<int>(pRightPyramid->size()) ||
           level >= static_cast<int>(scaleFactors.size()) ||
           level >= static_cast<int>(invScaleFactors.size()))
        {
            return false;
        }

        const cv::Mat& imL = (*pLeftPyramid)[level];
        const cv::Mat& imR = (*pRightPyramid)[level];
        if(imL.empty() || imR.empty())
            return false;

        const int w = 5;
        const int L = 3;
        const float invScale = invScaleFactors[level];
        const int scaleduL = cvRound(kpL.pt.x * invScale);
        const int scaledvL = cvRound(kpL.pt.y * invScale);
        const int scaleduR0 = cvRound(kpR.pt.x * invScale);
        const int scaledvR = cvRound(kpR.pt.y * invScale);

        if(scaledvL - w < 0 || scaledvL + w + 1 > imL.rows ||
           scaleduL - w < 0 || scaleduL + w + 1 > imL.cols ||
           scaledvR - w < 0 || scaledvR + w + 1 > imR.rows ||
           scaleduR0 - L - w < 0 || scaleduR0 + L + w + 1 > imR.cols)
        {
            return false;
        }

        const cv::Mat patchL = imL.rowRange(scaledvL - w, scaledvL + w + 1)
                                  .colRange(scaleduL - w, scaleduL + w + 1);

        int bestIncR = 0;
        float bestCombined = std::numeric_limits<float>::infinity();
        std::vector<float> patchDists(2 * L + 1, std::numeric_limits<float>::infinity());
        std::vector<float> patchScores(2 * L + 1, -std::numeric_limits<float>::infinity());
        for(int incR = -L; incR <= L; ++incR)
        {
            const cv::Mat patchR = imR.rowRange(scaledvR - w, scaledvR + w + 1)
                                      .colRange(scaleduR0 + incR - w, scaleduR0 + incR + w + 1);
            const float dist = ZeroMeanPatchDistanceL1(patchL, patchR);
            const float score = ZeroMeanPatchZNCC(patchL, patchR);
            patchDists[L + incR] = dist;
            patchScores[L + incR] = score;
        }

        float minCost = std::numeric_limits<float>::infinity();
        float maxCost = -std::numeric_limits<float>::infinity();
        std::vector<float> znccPenalties(2 * L + 1, std::numeric_limits<float>::infinity());
        for(const float dist : patchDists)
        {
            if(std::isfinite(dist))
            {
                minCost = std::min(minCost, dist);
                maxCost = std::max(maxCost, dist);
            }
        }
        if(!std::isfinite(minCost) || !std::isfinite(maxCost))
            return false;

        for(int incR = -L; incR <= L; ++incR)
        {
            const float score = patchScores[L + incR];
            if(!std::isfinite(score))
                continue;

            const float clampedZncc = std::max(-1.0f, std::min(1.0f, score));
            znccPenalties[L + incR] = 0.5f * (1.0f - clampedZncc);
        }

        float costLower = minCost;
        float costUpper = maxCost;
        float znccPenaltyLower = 0.0f;
        float znccPenaltyUpper = 1.0f;
        ComputeRobustFiniteRange(patchDists, 0.1f, 0.9f, costLower, costUpper);
        ComputeRobustFiniteRange(znccPenalties, 0.1f, 0.9f, znccPenaltyLower, znccPenaltyUpper);

        std::vector<float> combinedScores(2 * L + 1, std::numeric_limits<float>::infinity());
        for(int incR = -L; incR <= L; ++incR)
        {
            const float dist = patchDists[L + incR];
            const float score = patchScores[L + incR];
            if(!std::isfinite(dist) || !std::isfinite(score))
                continue;

            const float normalizedCost = NormalizeFiniteValueClamped(dist, costLower, costUpper);
            const float normalizedZnccPenalty =
                NormalizeFiniteValueClamped(znccPenalties[L + incR], znccPenaltyLower, znccPenaltyUpper);
            const float combined = 0.5f * normalizedCost + 0.5f * normalizedZnccPenalty;
            combinedScores[L + incR] = combined;
            if(combined < bestCombined)
            {
                bestCombined = combined;
                bestIncR = incR;
            }
        }

        if(bestIncR == -L || bestIncR == L)
            return false;

        const float cost1 = patchDists[L + bestIncR - 1];
        const float cost2 = patchDists[L + bestIncR];
        const float cost3 = patchDists[L + bestIncR + 1];
        const float score1 = patchScores[L + bestIncR - 1];
        const float score2 = patchScores[L + bestIncR];
        const float score3 = patchScores[L + bestIncR + 1];
        const float combined1 = combinedScores[L + bestIncR - 1];
        const float combined2 = combinedScores[L + bestIncR];
        const float combined3 = combinedScores[L + bestIncR + 1];
        const float curvatureCost = cost1 + cost3 - 2.0f * cost2;
        const float marginCost = std::min(cost1, cost3) - cost2;
        const float curvatureCombined = combined1 + combined3 - 2.0f * combined2;
        const float marginCombined = std::min(combined1, combined3) - combined2;
        const float peakScore = ComputeStereoPeakScore(score1, score2, score3);
        const float denom = 2.0f * curvatureCombined;
        if(std::fabs(denom) < 1e-6f)
            return false;

        const float deltaR = (combined1 - combined3) / denom;
        if(!std::isfinite(deltaR) || deltaR < -1.0f || deltaR > 1.0f)
            return false;

        refinement.refinedUR = scaleFactors[level] * (static_cast<float>(scaleduR0) + static_cast<float>(bestIncR) + deltaR);
        refinement.cost1 = cost1;
        refinement.cost2 = cost2;
        refinement.cost3 = cost3;
        refinement.combined1 = combined1;
        refinement.combined2 = combined2;
        refinement.combined3 = combined3;
        refinement.score1 = score1;
        refinement.score2 = score2;
        refinement.score3 = score3;
        refinement.curvatureCost = curvatureCost;
        refinement.marginCost = marginCost;
        refinement.curvatureCombined = curvatureCombined;
        refinement.marginCombined = marginCombined;
        refinement.peakScore = peakScore;
        refinement.success = std::isfinite(refinement.refinedUR) &&
                             std::isfinite(score2) &&
                             std::isfinite(curvatureCombined) &&
                             curvatureCombined > 0.0f;
        return refinement.success;
    }

    bool SearchXFeatStereoPatchLocal(const std::vector<cv::Mat>* pRefPyramid,
                                     const std::vector<cv::Mat>* pTargetPyramid,
                                     const std::vector<float>& scaleFactors,
                                     const std::vector<float>& invScaleFactors,
                                     const int level,
                                     const float uRef,
                                     const float vRef,
                                     const float uTargetSeed,
                                     XFeatStereoPatchSearchResult& result)
    {
        if(!pRefPyramid || !pTargetPyramid)
            return false;

        if(level < 0 ||
           level >= static_cast<int>(pRefPyramid->size()) ||
           level >= static_cast<int>(pTargetPyramid->size()) ||
           level >= static_cast<int>(scaleFactors.size()) ||
           level >= static_cast<int>(invScaleFactors.size()))
        {
            return false;
        }

        const cv::Mat& imRef = (*pRefPyramid)[level];
        const cv::Mat& imTarget = (*pTargetPyramid)[level];
        if(imRef.empty() || imTarget.empty())
            return false;

        const int halfWindow = 5;
        const float searchRadius = 8.0f;
        const float searchStep = 0.5f;
        const int nSamples = static_cast<int>(2.0f * searchRadius / searchStep) + 1;
        const float invScale = invScaleFactors[level];
        const float uRefScaled = uRef * invScale;
        const float vRefScaled = vRef * invScale;
        const float uSeedScaled = uTargetSeed * invScale;

        cv::Mat patchRef;
        if(!ExtractPatchAtFloatCenter(imRef, cv::Point2f(uRefScaled, vRefScaled), halfWindow, patchRef))
            return false;

        if(uSeedScaled - searchRadius - static_cast<float>(halfWindow) < 0.0f ||
           uSeedScaled + searchRadius + static_cast<float>(halfWindow) >= static_cast<float>(imTarget.cols) ||
           vRefScaled - static_cast<float>(halfWindow) < 0.0f ||
           vRefScaled + static_cast<float>(halfWindow) >= static_cast<float>(imTarget.rows))
        {
            return false;
        }

        std::vector<float> offsets;
        std::vector<float> costs(nSamples, std::numeric_limits<float>::infinity());
        std::vector<float> scores(nSamples, -std::numeric_limits<float>::infinity());
        std::vector<float> combineds(nSamples, std::numeric_limits<float>::infinity());
        offsets.reserve(nSamples);

        float minCost = std::numeric_limits<float>::infinity();
        float maxCost = -std::numeric_limits<float>::infinity();
        std::vector<float> znccPenalties(nSamples, std::numeric_limits<float>::infinity());
        for(int sampleIdx = 0; sampleIdx < nSamples; ++sampleIdx)
        {
            const float offset = -searchRadius + searchStep * static_cast<float>(sampleIdx);
            offsets.push_back(offset);

            cv::Mat patchTarget;
            if(!ExtractPatchAtFloatCenter(imTarget,
                                          cv::Point2f(uSeedScaled + offset, vRefScaled),
                                          halfWindow,
                                          patchTarget))
            {
                continue;
            }

            const float cost = ZeroMeanPatchDistanceL1(patchRef, patchTarget);
            const float score = ZeroMeanPatchZNCC(patchRef, patchTarget);
            costs[sampleIdx] = cost;
            scores[sampleIdx] = score;
            if(std::isfinite(score))
            {
                const float clampedZncc = std::max(-1.0f, std::min(1.0f, score));
                znccPenalties[sampleIdx] = 0.5f * (1.0f - clampedZncc);
            }
            if(std::isfinite(cost))
            {
                minCost = std::min(minCost, cost);
                maxCost = std::max(maxCost, cost);
            }
        }

        if(!std::isfinite(minCost) || !std::isfinite(maxCost))
            return false;

        float costLower = minCost;
        float costUpper = maxCost;
        float znccPenaltyLower = 0.0f;
        float znccPenaltyUpper = 1.0f;
        ComputeRobustFiniteRange(costs, 0.1f, 0.9f, costLower, costUpper);
        ComputeRobustFiniteRange(znccPenalties, 0.1f, 0.9f, znccPenaltyLower, znccPenaltyUpper);
        int bestIdx = -1;
        int secondBestIdx = -1;
        float bestCombined = std::numeric_limits<float>::infinity();
        float secondBestCombined = std::numeric_limits<float>::infinity();
        for(int sampleIdx = 0; sampleIdx < nSamples; ++sampleIdx)
        {
            const float cost = costs[sampleIdx];
            const float score = scores[sampleIdx];
            if(!std::isfinite(cost) || !std::isfinite(score))
                continue;

            const float normalizedCost = NormalizeFiniteValueClamped(cost, costLower, costUpper);
            const float normalizedZnccPenalty =
                NormalizeFiniteValueClamped(znccPenalties[sampleIdx], znccPenaltyLower, znccPenaltyUpper);
            const float combined = 0.5f * normalizedCost + 0.5f * normalizedZnccPenalty;
            combineds[sampleIdx] = combined;

            if(combined < bestCombined)
            {
                secondBestCombined = bestCombined;
                secondBestIdx = bestIdx;
                bestCombined = combined;
                bestIdx = sampleIdx;
            }
            else if(combined < secondBestCombined)
            {
                secondBestCombined = combined;
                secondBestIdx = sampleIdx;
            }
        }

        if(bestIdx <= 0 || bestIdx >= nSamples - 1)
            return false;

        const float combined1 = combineds[bestIdx - 1];
        const float combined2 = combineds[bestIdx];
        const float combined3 = combineds[bestIdx + 1];
        const float cost1 = costs[bestIdx - 1];
        const float cost2 = costs[bestIdx];
        const float cost3 = costs[bestIdx + 1];
        const float score1 = scores[bestIdx - 1];
        const float score2 = scores[bestIdx];
        const float score3 = scores[bestIdx + 1];
        if(!std::isfinite(combined1) || !std::isfinite(combined2) || !std::isfinite(combined3))
            return false;

        const float curvatureCost = cost1 + cost3 - 2.0f * cost2;
        const float marginCost = std::min(cost1, cost3) - cost2;
        const float curvatureCombined = combined1 + combined3 - 2.0f * combined2;
        const float marginCombined = std::min(combined1, combined3) - combined2;
        const float peakScore = ComputeStereoPeakScore(score1, score2, score3);
        const float denom = 2.0f * curvatureCombined;
        if(std::fabs(denom) < 1e-6f)
            return false;

        const float deltaSamples = (combined1 - combined3) / denom;
        if(!std::isfinite(deltaSamples) || deltaSamples < -1.0f || deltaSamples > 1.0f)
            return false;

        result.seedUR = uTargetSeed;
        result.bestCombined = bestCombined;
        result.secondBestCombined = secondBestCombined;
        result.coarseWinnerUR = scaleFactors[level] * (uSeedScaled + offsets[bestIdx]);
        result.secondBestUR = (secondBestIdx >= 0)
            ? scaleFactors[level] * (uSeedScaled + offsets[secondBestIdx])
            : std::numeric_limits<float>::quiet_NaN();
        result.winnerShift = result.coarseWinnerUR - uTargetSeed;
        result.refinement.refinedUR =
            scaleFactors[level] * (uSeedScaled + offsets[bestIdx] + deltaSamples * searchStep);
        result.refinement.cost1 = cost1;
        result.refinement.cost2 = cost2;
        result.refinement.cost3 = cost3;
        result.refinement.combined1 = combined1;
        result.refinement.combined2 = combined2;
        result.refinement.combined3 = combined3;
        result.refinement.score1 = score1;
        result.refinement.score2 = score2;
        result.refinement.score3 = score3;
        result.refinement.curvatureCost = curvatureCost;
        result.refinement.marginCost = marginCost;
        result.refinement.curvatureCombined = curvatureCombined;
        result.refinement.marginCombined = marginCombined;
        result.refinement.peakScore = peakScore;
        result.refinement.success = std::isfinite(result.refinement.refinedUR) &&
                                    std::isfinite(score2) &&
                                    std::isfinite(curvatureCombined) &&
                                    curvatureCombined > 0.0f;
        result.success = result.refinement.success;
        return result.success;
    }

    bool SearchXFeatStereoPatchBand(const std::vector<cv::Mat>* pRefPyramid,
                                    const std::vector<cv::Mat>* pTargetPyramid,
                                    const std::vector<float>& scaleFactors,
                                    const std::vector<float>& invScaleFactors,
                                    const int level,
                                    const float uRef,
                                    const float vRef,
                                    const float uTargetSeed,
                                    const std::vector<float>& uTargetSeeds,
                                    XFeatStereoPatchSearchResult& result)
    {
        if(!pRefPyramid || !pTargetPyramid || uTargetSeeds.empty())
            return false;

        if(level < 0 ||
           level >= static_cast<int>(pRefPyramid->size()) ||
           level >= static_cast<int>(pTargetPyramid->size()) ||
           level >= static_cast<int>(scaleFactors.size()) ||
           level >= static_cast<int>(invScaleFactors.size()))
        {
            return false;
        }

        const cv::Mat& imRef = (*pRefPyramid)[level];
        const cv::Mat& imTarget = (*pTargetPyramid)[level];
        if(imRef.empty() || imTarget.empty())
            return false;

        const int halfWindow = 5;
        const float minBandWidth = 8.0f;
        const float maxBandWidth = 20.0f;
        const float seedPad = 2.0f;
        const float invScale = invScaleFactors[level];
        const float uRefScaled = uRef * invScale;
        const float vRefScaled = vRef * invScale;
        const float uSeedScaled = uTargetSeed * invScale;

        cv::Mat patchRef;
        if(!ExtractPatchAtFloatCenter(imRef, cv::Point2f(uRefScaled, vRefScaled), halfWindow, patchRef))
            return false;

        float bandMin = std::numeric_limits<float>::infinity();
        float bandMax = -std::numeric_limits<float>::infinity();
        for(const float uSeed : uTargetSeeds)
        {
            const float uSeedScaledCandidate = uSeed * invScale;
            bandMin = std::min(bandMin, uSeedScaledCandidate);
            bandMax = std::max(bandMax, uSeedScaledCandidate);
        }
        if(!std::isfinite(bandMin) || !std::isfinite(bandMax))
            return false;

        bandMin -= seedPad;
        bandMax += seedPad;
        float bandWidth = bandMax - bandMin;
        if(bandWidth < minBandWidth)
        {
            bandMin = uSeedScaled - 0.5f * minBandWidth;
            bandMax = uSeedScaled + 0.5f * minBandWidth;
        }
        else if(bandWidth > maxBandWidth)
        {
            bandMin = uSeedScaled - 0.5f * maxBandWidth;
            bandMax = uSeedScaled + 0.5f * maxBandWidth;
        }
        bandWidth = bandMax - bandMin;

        if(bandMin - static_cast<float>(halfWindow) < 0.0f ||
           bandMax + static_cast<float>(halfWindow) >= static_cast<float>(imTarget.cols) ||
           vRefScaled - static_cast<float>(halfWindow) < 0.0f ||
           vRefScaled + static_cast<float>(halfWindow) >= static_cast<float>(imTarget.rows))
        {
            return false;
        }

        const int sampleMin = static_cast<int>(std::ceil(bandMin));
        const int sampleMax = static_cast<int>(std::floor(bandMax));
        const int nSamples = sampleMax - sampleMin + 1;
        if(nSamples < 3)
            return false;

        std::vector<float> costs(nSamples, std::numeric_limits<float>::infinity());
        std::vector<float> scores(nSamples, -std::numeric_limits<float>::infinity());
        std::vector<float> combineds(nSamples, std::numeric_limits<float>::infinity());
        std::vector<float> smoothedCombineds(nSamples, std::numeric_limits<float>::infinity());
        std::vector<float> znccPenalties(nSamples, std::numeric_limits<float>::infinity());

        float minCost = std::numeric_limits<float>::infinity();
        float maxCost = -std::numeric_limits<float>::infinity();
        for(int sampleIdx = 0; sampleIdx < nSamples; ++sampleIdx)
        {
            const float uTargetScaled = static_cast<float>(sampleMin + sampleIdx);

            cv::Mat patchTarget;
            if(!ExtractPatchAtFloatCenter(imTarget,
                                          cv::Point2f(uTargetScaled, vRefScaled),
                                          halfWindow,
                                          patchTarget))
            {
                continue;
            }

            const float cost = ZeroMeanPatchDistanceL1(patchRef, patchTarget);
            const float score = ZeroMeanPatchZNCC(patchRef, patchTarget);
            costs[sampleIdx] = cost;
            scores[sampleIdx] = score;
            if(std::isfinite(score))
            {
                const float clampedZncc = std::max(-1.0f, std::min(1.0f, score));
                znccPenalties[sampleIdx] = 0.5f * (1.0f - clampedZncc);
            }
            if(std::isfinite(cost))
            {
                minCost = std::min(minCost, cost);
                maxCost = std::max(maxCost, cost);
            }
        }

        if(!std::isfinite(minCost) || !std::isfinite(maxCost))
            return false;

        float costLower = minCost;
        float costUpper = maxCost;
        float znccPenaltyLower = 0.0f;
        float znccPenaltyUpper = 1.0f;
        ComputeRobustFiniteRange(costs, 0.1f, 0.9f, costLower, costUpper);
        ComputeRobustFiniteRange(znccPenalties, 0.1f, 0.9f, znccPenaltyLower, znccPenaltyUpper);
        for(int sampleIdx = 0; sampleIdx < nSamples; ++sampleIdx)
        {
            const float cost = costs[sampleIdx];
            const float score = scores[sampleIdx];
            if(!std::isfinite(cost) || !std::isfinite(score))
                continue;

            const float normalizedCost = NormalizeFiniteValueClamped(cost, costLower, costUpper);
            const float normalizedZnccPenalty =
                NormalizeFiniteValueClamped(znccPenalties[sampleIdx], znccPenaltyLower, znccPenaltyUpper);
            combineds[sampleIdx] = 0.5f * normalizedCost + 0.5f * normalizedZnccPenalty;
        }

        for(int sampleIdx = 1; sampleIdx < nSamples - 1; ++sampleIdx)
            smoothedCombineds[sampleIdx] = SmoothCombinedSample(combineds, sampleIdx);

        int bestIdx = -1;
        float bestCombined = std::numeric_limits<float>::infinity();
        for(int sampleIdx = 1; sampleIdx < nSamples - 1; ++sampleIdx)
        {
            const float combined = smoothedCombineds[sampleIdx];
            if(!std::isfinite(combined))
                continue;

            if(combined < bestCombined)
            {
                bestCombined = combined;
                bestIdx = sampleIdx;
            }
        }

        if(bestIdx <= 0 || bestIdx >= nSamples - 1)
            return false;

        int secondBestIdx = -1;
        float secondBestCombined = std::numeric_limits<float>::infinity();
        for(int sampleIdx = 1; sampleIdx < nSamples - 1; ++sampleIdx)
        {
            if(std::abs(sampleIdx - bestIdx) < 2)
                continue;

            const float combined = smoothedCombineds[sampleIdx];
            if(!std::isfinite(combined))
                continue;

            if(combined < secondBestCombined)
            {
                secondBestCombined = combined;
                secondBestIdx = sampleIdx;
            }
        }

        const float combined1 = smoothedCombineds[bestIdx - 1];
        const float combined2 = smoothedCombineds[bestIdx];
        const float combined3 = smoothedCombineds[bestIdx + 1];
        const float cost1 = costs[bestIdx - 1];
        const float cost2 = costs[bestIdx];
        const float cost3 = costs[bestIdx + 1];
        const float score1 = scores[bestIdx - 1];
        const float score2 = scores[bestIdx];
        const float score3 = scores[bestIdx + 1];
        if(!std::isfinite(combined1) || !std::isfinite(combined2) || !std::isfinite(combined3))
            return false;

        const float curvatureCost = cost1 + cost3 - 2.0f * cost2;
        const float marginCost = secondBestCombined - bestCombined;
        const float curvatureCombined = combined1 + combined3 - 2.0f * combined2;
        const float marginCombined = secondBestCombined - combined2;
        const float peakScore = ComputeStereoPeakScore(score1, score2, score3);
        const float denom = 2.0f * curvatureCombined;
        if(std::fabs(denom) < 1e-6f)
            return false;

        const float deltaSamples = (combined1 - combined3) / denom;
        if(!std::isfinite(deltaSamples) || deltaSamples < -1.0f || deltaSamples > 1.0f)
            return false;

        const float coarseWinnerScaled = static_cast<float>(sampleMin + bestIdx);
        result.seedUR = uTargetSeed;
        result.bandWidth = scaleFactors[level] * bandWidth;
        result.bestCombined = bestCombined;
        result.secondBestCombined = secondBestCombined;
        result.coarseWinnerUR = scaleFactors[level] * coarseWinnerScaled;
        result.secondBestUR = (secondBestIdx >= 0)
            ? scaleFactors[level] * static_cast<float>(sampleMin + secondBestIdx)
            : std::numeric_limits<float>::quiet_NaN();
        result.winnerShift = result.coarseWinnerUR - uTargetSeed;
        result.refinement.refinedUR = scaleFactors[level] * (coarseWinnerScaled + deltaSamples);
        result.refinement.cost1 = cost1;
        result.refinement.cost2 = cost2;
        result.refinement.cost3 = cost3;
        result.refinement.combined1 = combined1;
        result.refinement.combined2 = combined2;
        result.refinement.combined3 = combined3;
        result.refinement.score1 = score1;
        result.refinement.score2 = score2;
        result.refinement.score3 = score3;
        result.refinement.curvatureCost = curvatureCost;
        result.refinement.marginCost = marginCost;
        result.refinement.curvatureCombined = curvatureCombined;
        result.refinement.marginCombined = marginCombined;
        result.refinement.peakScore = peakScore;
        result.refinement.success = std::isfinite(result.refinement.refinedUR) &&
                                    std::isfinite(score2) &&
                                    std::isfinite(curvatureCombined) &&
                                    curvatureCombined > 0.0f;
        result.success = result.refinement.success;
        return result.success;
    }

    bool EstimateXFeatStereoSigmaDisp(const float scale,
                                      const XFeatStereoSubpixelRefinement& refinement,
                                      const float backprojErr,
                                      float& curvatureFactor,
                                      float& marginFactor,
                                      float& scoreFactor,
                                      float& lrFactor,
                                      float& sigmaDisp)
    {
        if(!refinement.success ||
           !std::isfinite(refinement.curvatureCombined) ||
           refinement.curvatureCombined <= 0.0f ||
           !std::isfinite(refinement.marginCombined) ||
           refinement.marginCombined <= 0.0f ||
           !std::isfinite(refinement.peakScore) ||
           refinement.peakScore <= 0.0f ||
           !std::isfinite(backprojErr) ||
           backprojErr < 0.0f)
            return false;

        const float baseSigma = 0.25f * std::max(scale, 1e-3f);
        curvatureFactor = std::max(0.5f, std::min(4.0f, 8.0f / refinement.curvatureCombined));
        marginFactor = std::max(0.5f, std::min(4.0f, 4.0f / std::max(refinement.marginCombined, 1e-4f)));
        scoreFactor = std::max(0.5f, std::min(4.0f, 0.15f / std::max(refinement.peakScore, 1e-4f)));
        lrFactor = std::max(0.5f, std::min(4.0f, backprojErr / std::max(0.5f * scale, 1e-4f)));
        sigmaDisp = baseSigma * std::max(std::max(curvatureFactor, marginFactor),
                                         std::max(scoreFactor, lrFactor));
        return std::isfinite(sigmaDisp) && sigmaDisp > 0.0f;
    }
}

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

//For stereo fisheye matching
cv::BFMatcher Frame::BFmatcher = cv::BFMatcher(cv::NORM_HAMMING);

Frame::Frame(): mpcpi(NULL), mpImuPreintegrated(NULL), mpPrevFrame(NULL), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mbHasPose(false), mbHasVelocity(false)
{
#ifdef REGISTER_TIMES
    mTimeStereoMatch = 0;
    mTimeORB_Ext = 0;
#endif
}


//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpcpi(frame.mpcpi),mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mpXFextractor(frame.mpXFextractor),
     mpXFextractorRight(frame.mpXFextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mK_(Converter::toMatrix3f(frame.mK)), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mvStereoBirthQuality(frame.mvStereoBirthQuality), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mvXFeatMatchSource(frame.mvXFeatMatchSource),
     mvLocalProjBestDist(frame.mvLocalProjBestDist), mvLocalProjSecondDist(frame.mvLocalProjSecondDist),
     mvLocalProjRatio(frame.mvLocalProjRatio), mvLocalProjCandidateCount(frame.mvLocalProjCandidateCount),
     mvLocalProjRadiusPx(frame.mvLocalProjRadiusPx), mvLocalProjProjX(frame.mvLocalProjProjX),
     mvLocalProjProjY(frame.mvLocalProjProjY), mvLocalProjProjErrPx(frame.mvLocalProjProjErrPx),
     mvLocalProjProjErrNorm(frame.mvLocalProjProjErrNorm), mvLocalProjPredLevel(frame.mvLocalProjPredLevel),
     mvLocalProjMatchLevel(frame.mvLocalProjMatchLevel), mvLocalProjLevelDiff(frame.mvLocalProjLevelDiff),
     mvLocalProjSide(frame.mvLocalProjSide),
     mLocalProjReferenceTargetMPIds(frame.mLocalProjReferenceTargetMPIds),
     mImuCalib(frame.mImuCalib), mnCloseMPs(frame.mnCloseMPs),
     mpImuPreintegrated(frame.mpImuPreintegrated), mpImuPreintegratedFrame(frame.mpImuPreintegratedFrame), mImuBias(frame.mImuBias),
     mnId(frame.mnId), mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors), mNameFile(frame.mNameFile), mnDataset(frame.mnDataset),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2), mpPrevFrame(frame.mpPrevFrame), mpLastKeyFrame(frame.mpLastKeyFrame),
     mbIsSet(frame.mbIsSet), mbImuPreintegrated(frame.mbImuPreintegrated), mpMutexImu(frame.mpMutexImu),
     mpCamera(frame.mpCamera), mpCamera2(frame.mpCamera2), Nleft(frame.Nleft), Nright(frame.Nright),
     monoLeft(frame.monoLeft), monoRight(frame.monoRight), mvLeftToRightMatch(frame.mvLeftToRightMatch),
     mvRightToLeftMatch(frame.mvRightToLeftMatch), mvStereo3Dpoints(frame.mvStereo3Dpoints),
     mTlr(frame.mTlr), mRlr(frame.mRlr), mtlr(frame.mtlr), mTrl(frame.mTrl),
     mvImagePyramidLeft(frame.mvImagePyramidLeft),
     mTcw(frame.mTcw), mbHasPose(false), mbHasVelocity(false)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++){
            mGrid[i][j]=frame.mGrid[i][j];
            if(frame.Nleft > 0){
                mGridRight[i][j] = frame.mGridRight[i][j];
            }
        }

    if(frame.mbHasPose)
        SetPose(frame.GetPose());

    if(frame.HasVelocity())
    {
        SetVelocity(frame.GetVelocity());
    }

    mmProjectPoints = frame.mmProjectPoints;
    mmMatchedInImage = frame.mmMatchedInImage;

#ifdef REGISTER_TIMES
    mTimeStereoMatch = frame.mTimeStereoMatch;
    mTimeORB_Ext = frame.mTimeORB_Ext;
#endif
}


Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
     mpCamera(pCamera) ,mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft,0,0);
    thread threadRight(&Frame::ExtractORB,this,1,imRight,0,0);
    threadLeft.join();
    threadRight.join();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    N = mvKeys.size();
    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
    ComputeStereoMatches();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);
    mvXFeatMatchSource = vector<unsigned char>(N, 0);
    mmProjectPoints.clear();
    mmMatchedInImage.clear();


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);



        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    if(pPrevF)
    {
        if(pPrevF->HasVelocity())
            SetVelocity(pPrevF->GetVelocity());
    }
    else
    {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();
}

Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, XFextractor* extractorLeft, XFextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL), mpORBvocabulary(voc), mpXFextractor(extractorLeft), mpXFextractorRight(extractorRight ? extractorRight : extractorLeft), mpORBextractorLeft(static_cast<ORBextractor*>(NULL)), mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
     mpCamera(pCamera), mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    mnId=nNextId++;

    XFextractor* pRightExtractor = mpXFextractorRight;

    mnScaleLevels = extractorLeft->GetLevels();
    mfScaleFactor = extractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = extractorLeft->GetScaleFactors();
    mvInvScaleFactors = extractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = extractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = extractorLeft->GetInverseScaleSigmaSquares();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    {
        vector<int> vLapping = {0,0};
        monoLeft = (*extractorLeft)(imLeft,cv::Mat(),mvKeys,mDescriptors,vLapping);
        // Cache left image pyramid before right extraction overwrites it, needed for
        // sub-pixel stereo refinement.
        mvImagePyramidLeft = extractorLeft->mvImagePyramid;
        monoRight = (*pRightExtractor)(imRight,cv::Mat(),mvKeysRight,mDescriptorsRight,vLapping);
    }
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    N = mvKeys.size();
    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
    ComputeStereoMatches();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);
    mvXFeatMatchSource = vector<unsigned char>(N, 0);
    mmProjectPoints.clear();
    mmMatchedInImage.clear();

    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    if(pPrevF)
    {
        if(pPrevF->HasVelocity())
            SetVelocity(pPrevF->GetVelocity());
    }
    else
    {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();

    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera,Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
     mpCamera(pCamera),mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    ExtractORB(0,imGray,0,0);
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif


    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvXFeatMatchSource = vector<unsigned char>(N, 0);

    mmProjectPoints.clear();
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    if(pPrevF){
        if(pPrevF->HasVelocity())
            SetVelocity(pPrevF->GetVelocity());
    }
    else{
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, XFextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera,Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpXFextractor(extractor),mpXFextractorRight(extractor),mpORBextractorLeft(static_cast<ORBextractor*>(NULL)), mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
     mpCamera(pCamera),mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpXFextractor->GetLevels();
    mfScaleFactor = mpXFextractor->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpXFextractor->GetScaleFactors();
    mvInvScaleFactors = mpXFextractor->GetInverseScaleFactors();
    mvLevelSigma2 = mpXFextractor->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpXFextractor->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    ExtractXF(0,imGray,0,0);  
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif


    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvXFeatMatchSource = vector<unsigned char>(N, 0);

    mmProjectPoints.clear();
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    if(pPrevF){
        if(pPrevF->HasVelocity())
            SetVelocity(pPrevF->GetVelocity());
    }
    else{
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();
}


Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, GeometricCamera* pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(static_cast<Pinhole*>(pCamera)->toK()), mK_(static_cast<Pinhole*>(pCamera)->toK_()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL),mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mpCamera(pCamera),
     mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    ExtractORB(0,imGray,0,1000);
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif


    N = mvKeys.size();
    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);
    mvStereoBirthQuality = vector<unsigned char>(N, kStereoBirthInvalid);
    mnCloseMPs = 0;

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvXFeatMatchSource = vector<unsigned char>(N, 0);

    mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,0);
        fy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,1);
        cx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,2);
        cy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }


    mb = mbf/fx;

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();

    if(pPrevF)
    {
        if(pPrevF->HasVelocity())
        {
            SetVelocity(pPrevF->GetVelocity());
        }
    }
    else
    {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();
}

Frame::Frame(const cv::Mat &imGray, const double &timeStamp, XFextractor* extractor,ORBVocabulary* voc, GeometricCamera* pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpXFextractor(extractor),mpXFextractorRight(extractor),mpORBextractorLeft(static_cast<ORBextractor*>(NULL)),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(static_cast<Pinhole*>(pCamera)->toK()), mK_(static_cast<Pinhole*>(pCamera)->toK_()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL),mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mpCamera(pCamera),
     mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpXFextractor->GetLevels();
    mfScaleFactor = mpXFextractor->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpXFextractor->GetScaleFactors();
    mvInvScaleFactors = mpXFextractor->GetInverseScaleFactors();
    mvLevelSigma2 = mpXFextractor->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpXFextractor->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    ExtractXF(0,imGray,0,1000);
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif
    N = mvKeys.size();  // 赋值后的特征点数量

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    //调试: XFeat提取结果摘要（在Undistort之后打印，避免 mvKeysUn 误导信息）。
    if(ShouldPrintXFeatFrameDebug(mnId))
    {
        std::cout << "[Frame::ExtractXF] frame=" << mnId
                  << " N=" << N
                  << " desc_rows=" << mDescriptors.rows
                  << " desc_cols=" << mDescriptors.cols
                  << " desc_type=" << mDescriptors.type()
                  << " mvKeysUn=" << mvKeysUn.size()
                  << std::endl;
    }

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);
    mvStereoBirthQuality = vector<unsigned char>(N, kStereoBirthInvalid);
    mnCloseMPs = 0;
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvXFeatMatchSource = vector<unsigned char>(N, 0);

    mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,0);
        fy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,1);
        cx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,2);
        cy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }


    mb = mbf/fx;

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();

    if(pPrevF)
    {
        if(pPrevF->HasVelocity())
        {
            SetVelocity(pPrevF->GetVelocity());
        }
    }
    else
    {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();
}


void Frame::AssignFeaturesToGrid()
{
    // Fill matrix with points
    const int nCells = FRAME_GRID_COLS*FRAME_GRID_ROWS;

    int nReserve = 0.5f*N/(nCells);

    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++){
            mGrid[i][j].reserve(nReserve);
            if(Nleft != -1){
                mGridRight[i][j].reserve(nReserve);
            }
        }



    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = (Nleft == -1) ? mvKeysUn[i]
                                                 : (i < Nleft) ? mvKeys[i]
                                                                 : mvKeysRight[i - Nleft];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY)){
            if(Nleft == -1 || i < Nleft)
                mGrid[nGridPosX][nGridPosY].push_back(i);
            else
                mGridRight[nGridPosX][nGridPosY].push_back(i - Nleft);
        }
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1)
{
    vector<int> vLapping = {x0,x1};
    if(flag==0)
        monoLeft = (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors,vLapping);
    else
        monoRight = (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight,vLapping);
}

void Frame::ExtractXF(int flag, const cv::Mat &im, const int x0, const int x1)
{
    vector<int> vLapping = {x0,x1};
    if(flag==0)
        monoLeft = (*mpXFextractor)(im,cv::Mat(),mvKeys,mDescriptors,vLapping);
    else
        monoRight = (*mpXFextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight,vLapping);
}

bool Frame::isSet() const {
    return mbIsSet;
}

void Frame::SetPose(const Sophus::SE3<float> &Tcw) {
    mTcw = Tcw;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
}

void Frame::SetNewBias(const IMU::Bias &b)
{
    mImuBias = b;
    if(mpImuPreintegrated)
        mpImuPreintegrated->SetNewBias(b);
}

void Frame::SetVelocity(Eigen::Vector3f Vwb)
{
    mVw = Vwb;
    mbHasVelocity = true;
}

Eigen::Vector3f Frame::GetVelocity() const
{
    return mVw;
}

void Frame::SetImuPoseVelocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::Vector3f &Vwb)
{
    mVw = Vwb;
    mbHasVelocity = true;

    Sophus::SE3f Twb(Rwb, twb);
    Sophus::SE3f Tbw = Twb.inverse();

    mTcw = mImuCalib.mTcb * Tbw;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
}

void Frame::UpdatePoseMatrices()
{
    Sophus::SE3<float> Twc = mTcw.inverse();
    mRwc = Twc.rotationMatrix();
    mOw = Twc.translation();
    mRcw = mTcw.rotationMatrix();
    mtcw = mTcw.translation();
}

Eigen::Matrix<float,3,1> Frame::GetImuPosition() const {
    return mRwc * mImuCalib.mTcb.translation() + mOw;
}

Eigen::Matrix<float,3,3> Frame::GetImuRotation() {
    return mRwc * mImuCalib.mTcb.rotationMatrix();
}

Sophus::SE3<float> Frame::GetImuPose() {
    return mTcw.inverse() * mImuCalib.mTcb;
}

Sophus::SE3f Frame::GetRelativePoseTrl()
{
    return mTrl;
}

Sophus::SE3f Frame::GetRelativePoseTlr()
{
    return mTlr;
}

Eigen::Matrix3f Frame::GetRelativePoseTlr_rotation(){
    return mTlr.rotationMatrix();
}

Eigen::Vector3f Frame::GetRelativePoseTlr_translation() {
    return mTlr.translation();
}


bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mTrackFrustumStatus = MapPoint::kTrackFrustumUnchecked;
    if(Nleft == -1){
        pMP->mbTrackInView = false;
        pMP->mTrackProjX = -1;
        pMP->mTrackProjY = -1;

        // 3D in absolute coordinates
        Eigen::Matrix<float,3,1> P = pMP->GetWorldPos();

        // 3D in camera coordinates
        const Eigen::Matrix<float,3,1> Pc = mRcw * P + mtcw;
        const float Pc_dist = Pc.norm();

        // Check positive depth
        const float &PcZ = Pc(2);
        const float invz = 1.0f/PcZ;
        if(PcZ<0.0f)
        {
            pMP->mTrackFrustumStatus = MapPoint::kTrackFrustumNegativeDepth;
            return false;
        }

        const Eigen::Vector2f uv = mpCamera->project(Pc);

        if(uv(0)<mnMinX || uv(0)>mnMaxX)
        {
            pMP->mTrackFrustumStatus = MapPoint::kTrackFrustumOutsideImage;
            return false;
        }
        if(uv(1)<mnMinY || uv(1)>mnMaxY)
        {
            pMP->mTrackFrustumStatus = MapPoint::kTrackFrustumOutsideImage;
            return false;
        }

        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjY = uv(1);

        // Check distance is in the scale invariance region of the MapPoint
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const Eigen::Vector3f PO = P - mOw;
        const float dist = PO.norm();

        if(dist<minDistance || dist>maxDistance)
        {
            pMP->mTrackFrustumStatus = MapPoint::kTrackFrustumScaleRange;
            return false;
        }

        // Check viewing angle
        Eigen::Vector3f Pn = pMP->GetNormal();

        const float viewCos = PO.dot(Pn)/dist;

        if(viewCos<viewingCosLimit)
        {
            pMP->mTrackFrustumStatus = MapPoint::kTrackFrustumViewAngle;
            return false;
        }

        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist,this);

        // Data used by the tracking
        pMP->mbTrackInView = true;
        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjXR = uv(0) - mbf*invz;

        pMP->mTrackDepth = Pc_dist;

        pMP->mTrackProjY = uv(1);
        pMP->mnTrackScaleLevel= nPredictedLevel;
        pMP->mTrackViewCos = viewCos;
        pMP->mTrackFrustumStatus = MapPoint::kTrackFrustumAccepted;

        return true;
    }
    else{
        pMP->mbTrackInView = false;
        pMP->mbTrackInViewR = false;
        pMP -> mnTrackScaleLevel = -1;
        pMP -> mnTrackScaleLevelR = -1;

        unsigned char leftStatus = static_cast<unsigned char>(MapPoint::kTrackFrustumUnchecked);
        unsigned char rightStatus = static_cast<unsigned char>(MapPoint::kTrackFrustumUnchecked);
        pMP->mbTrackInView = isInFrustumChecks(pMP,viewingCosLimit,false,&leftStatus);
        pMP->mbTrackInViewR = isInFrustumChecks(pMP,viewingCosLimit,true,&rightStatus);
        if(pMP->mbTrackInView || pMP->mbTrackInViewR)
            pMP->mTrackFrustumStatus = MapPoint::kTrackFrustumAccepted;
        else
            pMP->mTrackFrustumStatus = static_cast<MapPoint::TrackFrustumStatus>(leftStatus);

        return pMP->mbTrackInView || pMP->mbTrackInViewR;
    }
}

bool Frame::ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    // 3D in camera coordinates
    const Eigen::Vector3f Pc = mRcw * P + mtcw;
    const float &PcX = Pc(0);
    const float &PcY= Pc(1);
    const float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        cout << "Negative depth: " << PcZ << endl;
        return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    u=fx*PcX*invz+cx;
    v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    float u_distort, v_distort;

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

    u_distort = x_distort * fx + cx;
    v_distort = y_distort * fy + cy;


    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

Eigen::Vector3f Frame::inRefCoordinates(Eigen::Vector3f pCw)
{
    return mRcw * pCw + mtcw;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel, const bool bRight) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    float factorX = r;
    float factorY = r;

    const int nMinCellX = max(0,(int)floor((x-mnMinX-factorX)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
    {
        return vIndices;
    }

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+factorX)*mfGridElementWidthInv));
    if(nMaxCellX<0)
    {
        return vIndices;
    }

    const int nMinCellY = max(0,(int)floor((y-mnMinY-factorY)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
    {
        return vIndices;
    }

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+factorY)*mfGridElementHeightInv));
    if(nMaxCellY<0)
    {
        return vIndices;
    }

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = (Nleft == -1) ? mvKeysUn[vCell[j]]
                                                         : (!bRight) ? mvKeys[vCell[j]]
                                                                     : mvKeysRight[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<factorX && fabs(disty)<factorY)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);

    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat, static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);


    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }

}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        // Undistort corners
        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));
    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);
    mvStereoBirthQuality = vector<unsigned char>(N, kStereoBirthInvalid);

    const bool bFloatDescriptors = IsFloatDescriptorMat(mDescriptors) || IsFloatDescriptorMat(mDescriptorsRight);
    const float thDescDist = bFloatDescriptors ? GetXFeatStereoDescThreshold()
                                               : static_cast<float>((ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2);
    const std::vector<cv::Mat>* pLeftXFeatPyramid = nullptr;
    if(!mvImagePyramidLeft.empty() && !mvImagePyramidLeft[0].empty())
        pLeftXFeatPyramid = &mvImagePyramidLeft;
    else if(mpXFextractor && !mpXFextractor->mvImagePyramid.empty() && !mpXFextractor->mvImagePyramid[0].empty())
        pLeftXFeatPyramid = &mpXFextractor->mvImagePyramid;

    const std::vector<cv::Mat>* pRightXFeatPyramid = nullptr;
    if(mpXFextractorRight && !mpXFextractorRight->mvImagePyramid.empty() && !mpXFextractorRight->mvImagePyramid[0].empty())
        pRightXFeatPyramid = &mpXFextractorRight->mvImagePyramid;
    else if(mpXFextractor && !mpXFextractor->mvImagePyramid.empty() && !mpXFextractor->mvImagePyramid[0].empty())
        pRightXFeatPyramid = &mpXFextractor->mvImagePyramid;

    int nRows = 0;
    if(mpORBextractorLeft && !mpORBextractorLeft->mvImagePyramid.empty() && !mpORBextractorLeft->mvImagePyramid[0].empty())
        nRows = mpORBextractorLeft->mvImagePyramid[0].rows;
    else if(pRightXFeatPyramid)
        nRows = (*pRightXFeatPyramid)[0].rows;
    else
        nRows = std::max(1, static_cast<int>(std::ceil(mnMaxY - mnMinY)));

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    // XFeat stereo gating config (must be before reverse arrays)
    const float stereoRatio = GetXFeatStereoRatio();
    const bool bStereoMutual = IsXFeatStereoMutualEnabled();
    const bool bUseStereoRatio = (stereoRatio > 0.0f);
    const float stereoMaxYErr = GetXFeatStereoMaxYError();
    const float stereoMaxRelDepthSigma = GetXFeatStereoMaxRelativeDepthSigma();
    const bool bUseStereoUncertainty = (stereoMaxRelDepthSigma > 0.0f);
    const bool bUseXFeatStereoSubpixel = IsXFeatStereoSubpixelEnabled();

    // Reverse best match arrays for mutual check (built during forward search)
    std::vector<float> reverseBestDist;
    std::vector<int>   reverseBestIdx;
    if(bFloatDescriptors && bStereoMutual)
    {
        reverseBestDist.assign(Nr, std::numeric_limits<float>::infinity());
        reverseBestIdx.assign(Nr, -1);
    }
    std::vector<float> xfeatBestDist;
    std::vector<float> xfeatBestDist2;
    std::vector<int> xfeatBestIdxR;
    std::vector<std::vector<std::pair<float, int>>> xfeatTopKCandidates;
    if(bFloatDescriptors)
    {
        xfeatBestDist.assign(N, std::numeric_limits<float>::infinity());
        xfeatBestDist2.assign(N, std::numeric_limits<float>::infinity());
        xfeatBestIdxR.assign(N, -1);
        xfeatTopKCandidates.assign(N, std::vector<std::pair<float, int>>());
    }

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = std::min(nRows - 1, static_cast<int>(ceil(kpY+r)));
        const int minr = std::max(0, static_cast<int>(floor(kpY-r)));

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = bFloatDescriptors ? GetXFeatStereoMinDisparity() : 0.0f;
    const float maxD = mbf/minZ;

    //诊断: XFeat立体匹配各阶段统计
    int nDescPass = 0;
    int nRatioReject = 0;
    int nMutualReject = 0;
    int nDisparityReject = 0;
    int nEpipolarReject = 0;
    int nUncertaintyReject = 0;
    int nSubpixelRefined = 0;
    int nSubpixelFallback = 0;
    int nAccepted = 0;
    int nForwardCandidates = 0;
    int nMedianReject = 0;
    int nCloseCandidates = 0;
    int nCloseHighConf = 0;
    int nCloseLowConf = 0;
    int nRejectCloseY = 0;
    int nRejectCloseLR = 0;
    int nCloseSubpixelAttempt = 0;
    int nCloseSubpixelRefined = 0;
    int nCloseSubpixelReject = 0;
    int nCloseUncertaintyReject = 0;
    int nCloseFlatCurveReject = 0;
    int nCloseCombinedCurveReject = 0;
    int nCloseLowMarginReject = 0;
    int nCloseLRConsistencyReject = 0;
    int nCloseLowPeakReject = 0;
    int nCloseSigmaModelReject = 0;
    int nCloseQualitySigmaReject = 0;
    int nCloseScoreSigmaReject = 0;
    int nCloseCombinedSigmaReject = 0;
    int nCloseTopKCandidates = 0;
    int nCloseCandidateAmbiguousReject = 0;
    int nCloseCandidateRescoreWinnerNonBestIdx = 0;
    int nClosePatchSearchAttempt = 0;
    int nClosePatchSearchAmbiguousReject = 0;
    int nClosePatchSearchWinnerShiftGt1Px = 0;
    int nClosePatchSigmaReject = 0;
    int nCloseBandSearchAttempt = 0;
    int nCloseBandSearchAmbiguousReject = 0;
    int nCloseBandSearchWidthGt12Px = 0;
    int nCloseBandSigmaReject = 0;
    int nCloseCostVolumeAttempt = 0;
    int nCloseCostVolumeAmbiguousReject = 0;
    int nCloseCostVolumeSigmaReject = 0;

    // For each left keypoint search a match in the right image
    vector<pair<float, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const int rowL = std::max(0, std::min(nRows - 1, cvRound(vL)));
        const vector<size_t> &vCandidates = vRowIndices[rowL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        float bestDist = std::numeric_limits<float>::infinity();
        float bestDist2 = std::numeric_limits<float>::infinity();
        int bestIdxR = -1;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                if(bFloatDescriptors && stereoMaxYErr > 0.0f)
                {
                    const int maxLevel = std::max(levelL, kpR.octave);
                    const float scale = (maxLevel >= 0 && maxLevel < static_cast<int>(mvScaleFactors.size()))
                        ? mvScaleFactors[maxLevel]
                        : 1.0f;
                    if(std::fabs(vL - kpR.pt.y) > stereoMaxYErr * scale)
                    {
                        ++nEpipolarReject;
                        continue;
                    }
                }

                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const float dist = StereoDescriptorDistance(dL,dR,bFloatDescriptors);

                if(dist<bestDist)
                {
                    bestDist2 = bestDist;
                    bestDist  = dist;
                    bestIdxR  = static_cast<int>(iR);
                }
                else if(dist<bestDist2)
                {
                    bestDist2 = dist;
                }

                // XFeat mutual check is evaluated after all left/right best matches are known.
                if(bFloatDescriptors && bStereoMutual && dist < reverseBestDist[iR])
                {
                    reverseBestDist[iR] = dist;
                    reverseBestIdx[iR] = iL;
                }

                if(bFloatDescriptors)
                {
                    auto& topk = xfeatTopKCandidates[iL];
                    topk.emplace_back(dist, static_cast<int>(iR));
                    std::sort(topk.begin(), topk.end(),
                              [](const std::pair<float, int>& a, const std::pair<float, int>& b)
                              {
                                  if(a.first != b.first)
                                      return a.first < b.first;
                                  return a.second < b.second;
                              });
                    if(topk.size() > 4)
                        topk.resize(4);
                }
            }
        }

        if(bFloatDescriptors)
        {
            if(bestIdxR >= 0 && std::isfinite(bestDist))
            {
                xfeatBestDist[iL] = bestDist;
                xfeatBestDist2[iL] = bestDist2;
                xfeatBestIdxR[iL] = bestIdxR;
                ++nForwardCandidates;
            }
            continue;
        }

        // Subpixel match by correlation
        if(bestDist<thDescDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);

            int bestPatchDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestPatchDist)
                {
                    bestPatchDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                const float depth = mbf/disparity;
                mvDepth[iL]=depth;
                mvuRight[iL] = bestuR;
                mvStereoBirthQuality[iL] = (depth > 0.0f && depth <= mThDepth)
                    ? kStereoBirthCloseHighConf
                    : kStereoBirthValid;
                vDistIdx.push_back(pair<float,int>(static_cast<float>(bestPatchDist),iL));
            }
        }
    }

    if(bFloatDescriptors)
    {
        for(int iL=0; iL<N; ++iL)
        {
            const int bestIdxR = xfeatBestIdxR[iL];
            if(bestIdxR < 0)
                continue;

            const float bestDist = xfeatBestDist[iL];
            const float bestDist2 = xfeatBestDist2[iL];
            if(bestDist >= thDescDist)
                continue;

            nDescPass++;

            if(bUseStereoRatio && std::isfinite(bestDist2) && bestDist2 > 0.0f)
            {
                if(bestDist >= stereoRatio * bestDist2)
                {
                    nRatioReject++;
                    continue;
                }
            }

            if(bStereoMutual)
            {
                const int revBest = reverseBestIdx[static_cast<size_t>(bestIdxR)];
                if(revBest < 0 || revBest != iL)
                {
                    nMutualReject++;
                    continue;
                }
            }

            const cv::KeyPoint& kpL = mvKeys[iL];
            const cv::KeyPoint& kpR = mvKeysRight[static_cast<size_t>(bestIdxR)];
            const float uL = kpL.pt.x;
            const float uR0 = kpR.pt.x;
            float bestuR = uR0;
            float disparity = (uL - bestuR);
            if(disparity < minD || disparity >= maxD)
            {
                nDisparityReject++;
                continue;
            }

            const float tentativeDepth = mbf / std::max(disparity, 1e-4f);
            const bool bCloseCandidate = std::isfinite(tentativeDepth) &&
                                         tentativeDepth > 0.0f &&
                                         tentativeDepth <= mThDepth;
            const bool bRequireCloseSubpixel = bCloseCandidate;

            XFeatStereoSubpixelRefinement refinement;
            refinement.refinedUR = bestuR;
            if(bRequireCloseSubpixel)
            {
                ++nCloseSubpixelAttempt;
                ++nCloseBandSearchAttempt;
                ++nCloseCostVolumeAttempt;
                nCloseTopKCandidates += static_cast<int>(xfeatTopKCandidates[iL].size());
                const int maxLevel = std::max(kpL.octave, kpR.octave);
                const float scale = (maxLevel >= 0 && maxLevel < static_cast<int>(mvScaleFactors.size()))
                    ? mvScaleFactors[maxLevel]
                    : 1.0f;
                std::vector<float> topkSeeds;
                topkSeeds.reserve(xfeatTopKCandidates[iL].size());
                for(const auto& cand : xfeatTopKCandidates[iL])
                {
                    const int candIdxR = cand.second;
                    if(candIdxR < 0 || candIdxR >= Nr)
                        continue;
                    topkSeeds.push_back(mvKeysRight[static_cast<size_t>(candIdxR)].pt.x);
                }
                if(topkSeeds.empty())
                    topkSeeds.push_back(bestuR);
                XFeatStereoPatchSearchResult patchSearch;
                if(!SearchXFeatStereoPatchBand(pLeftXFeatPyramid,
                                              pRightXFeatPyramid,
                                              mvScaleFactors,
                                              mvInvScaleFactors,
                                              kpL.octave,
                                              kpL.pt.x,
                                              kpL.pt.y,
                                              bestuR,
                                              topkSeeds,
                                              patchSearch))
                {
                    ++nCloseLowConf;
                    ++nRejectCloseLR;
                    ++nCloseSubpixelReject;
                    mvStereoBirthQuality[iL] = kStereoBirthInvalid;
                    continue;
                }

                if(patchSearch.bandWidth > 12.0f)
                    ++nCloseBandSearchWidthGt12Px;
                if(std::isfinite(patchSearch.secondBestCombined) &&
                   std::isfinite(patchSearch.secondBestUR) &&
                   patchSearch.secondBestCombined <= 1.08f * std::max(patchSearch.bestCombined, 1e-4f) &&
                   std::fabs((uL - patchSearch.secondBestUR) - (uL - patchSearch.coarseWinnerUR)) >= 1.0f)
                {
                    ++nCloseLowConf;
                    ++nRejectCloseLR;
                    ++nCloseUncertaintyReject;
                    ++nCloseBandSearchAmbiguousReject;
                    ++nCloseCostVolumeAmbiguousReject;
                    ++nCloseCandidateAmbiguousReject;
                    ++nUncertaintyReject;
                    mvStereoBirthQuality[iL] = kStereoBirthInvalid;
                    continue;
                }

                refinement = patchSearch.refinement;
                bestuR = patchSearch.refinement.refinedUR;
                disparity = uL - bestuR;
                if(disparity < minD || disparity >= maxD)
                {
                    ++nCloseLowConf;
                    ++nRejectCloseLR;
                    ++nCloseSubpixelReject;
                    mvStereoBirthQuality[iL] = kStereoBirthInvalid;
                    continue;
                }

                ++nSubpixelRefined;
                ++nCloseSubpixelRefined;
            }
            else
            {
                const bool bSubpixelOk =
                    RefineXFeatStereoSubpixel(pLeftXFeatPyramid,
                                              pRightXFeatPyramid,
                                              mvScaleFactors,
                                              mvInvScaleFactors,
                                              kpL,
                                              kpR,
                                              refinement);
                if(bUseXFeatStereoSubpixel && bSubpixelOk)
                {
                    const float refinedDisparity = uL - refinement.refinedUR;
                    if(refinedDisparity >= minD && refinedDisparity < maxD)
                    {
                        bestuR = refinement.refinedUR;
                        disparity = refinedDisparity;
                        ++nSubpixelRefined;
                    }
                    else
                    {
                        ++nSubpixelFallback;
                    }
                }
                else if(bUseXFeatStereoSubpixel)
                {
                    ++nSubpixelFallback;
                }
            }

            if(disparity<=0) disparity=0.01f;
            const float depth = mbf/disparity;
            unsigned char birthQuality = kStereoBirthValid;
            if(depth > 0.0f && depth <= mThDepth)
            {
                ++nCloseCandidates;
                const int maxLevel = std::max(kpL.octave, kpR.octave);
                const float scale = (maxLevel >= 0 && maxLevel < static_cast<int>(mvScaleFactors.size()))
                    ? mvScaleFactors[maxLevel]
                    : 1.0f;
                const float strictMaxYErr = GetXFeatStereoBirthCloseMaxYError() * scale;
                const float verticalError = std::fabs(kpL.pt.y - kpR.pt.y);
                if(verticalError > strictMaxYErr)
                {
                    ++nCloseLowConf;
                    ++nRejectCloseY;
                    mvStereoBirthQuality[iL] = kStereoBirthInvalid;
                    continue;
                }

                XFeatStereoSubpixelRefinement backRefinement;
                XFeatStereoPatchSearchResult backPatchSearch;
                const bool bBackSubpixelOk =
                    SearchXFeatStereoPatchLocal(pRightXFeatPyramid,
                                                pLeftXFeatPyramid,
                                                mvScaleFactors,
                                                mvInvScaleFactors,
                                                kpL.octave,
                                                bestuR,
                                                kpL.pt.y,
                                                kpL.pt.x,
                                                backPatchSearch);
                if(bBackSubpixelOk)
                    backRefinement = backPatchSearch.refinement;
                const float backprojErr = bBackSubpixelOk
                    ? std::fabs(backPatchSearch.refinement.refinedUR - kpL.pt.x)
                    : std::numeric_limits<float>::infinity();
                if(!bBackSubpixelOk || !std::isfinite(backprojErr) || !std::isfinite(backRefinement.refinedUR))
                {
                    ++nCloseLowConf;
                    ++nRejectCloseLR;
                    ++nCloseUncertaintyReject;
                    ++nCloseLRConsistencyReject;
                    ++nUncertaintyReject;
                    mvStereoBirthQuality[iL] = kStereoBirthInvalid;
                    continue;
                }

                const float marginCombined = refinement.marginCombined;
                if(!std::isfinite(refinement.curvatureCost) || refinement.curvatureCost <= 0.0f)
                {
                    ++nCloseFlatCurveReject;
                }

                if(!std::isfinite(refinement.curvatureCombined) || refinement.curvatureCombined <= 0.0f)
                {
                    ++nCloseLowConf;
                    ++nRejectCloseLR;
                    ++nCloseUncertaintyReject;
                    ++nCloseCombinedCurveReject;
                    ++nUncertaintyReject;
                    mvStereoBirthQuality[iL] = kStereoBirthInvalid;
                    continue;
                }

                if(!std::isfinite(marginCombined) || marginCombined <= 0.0f)
                {
                    ++nCloseLowConf;
                    ++nRejectCloseLR;
                    ++nCloseUncertaintyReject;
                    ++nCloseLowMarginReject;
                    ++nUncertaintyReject;
                    mvStereoBirthQuality[iL] = kStereoBirthInvalid;
                    continue;
                }

                if(!std::isfinite(refinement.peakScore) || refinement.peakScore <= 0.0f)
                {
                    ++nCloseLowConf;
                    ++nRejectCloseLR;
                    ++nCloseUncertaintyReject;
                    ++nCloseLowPeakReject;
                    ++nUncertaintyReject;
                    mvStereoBirthQuality[iL] = kStereoBirthInvalid;
                    continue;
                }

                float sigmaDisp = 0.0f;
                float curvatureFactor = 0.0f;
                float marginFactor = 0.0f;
                float scoreFactor = 0.0f;
                float lrFactor = 0.0f;
                if(!EstimateXFeatStereoSigmaDisp(scale,
                                                 refinement,
                                                 backprojErr,
                                                 curvatureFactor,
                                                 marginFactor,
                                                 scoreFactor,
                                                 lrFactor,
                                                 sigmaDisp))
                {
                    ++nCloseLowConf;
                    ++nRejectCloseLR;
                    ++nCloseUncertaintyReject;
                    ++nCloseFlatCurveReject;
                    ++nUncertaintyReject;
                    mvStereoBirthQuality[iL] = kStereoBirthInvalid;
                    continue;
                }

                const float relDepthSigma = sigmaDisp / std::max(disparity, 1e-4f);
                if(!std::isfinite(relDepthSigma) ||
                   relDepthSigma > GetXFeatStereoBirthCloseMaxRelativeDepthSigma())
                {
                    ++nCloseLowConf;
                    ++nRejectCloseLR;
                    ++nCloseUncertaintyReject;
                    ++nCloseSigmaModelReject;
                    ++nCloseQualitySigmaReject;
                    ++nCloseCombinedSigmaReject;
                    ++nCloseBandSigmaReject;
                    ++nCloseCostVolumeSigmaReject;
                    if(scoreFactor >= curvatureFactor &&
                       scoreFactor >= marginFactor &&
                       scoreFactor >= lrFactor)
                    {
                        ++nCloseScoreSigmaReject;
                    }
                    ++nUncertaintyReject;
                    mvStereoBirthQuality[iL] = kStereoBirthInvalid;
                    continue;
                }

                birthQuality = kStereoBirthCloseHighConf;
                ++nCloseHighConf;
            }
            else if(bUseStereoUncertainty)
            {
                const int maxLevel = std::max(kpL.octave, kpR.octave);
                const float scale = (maxLevel >= 0 && maxLevel < static_cast<int>(mvScaleFactors.size()))
                    ? mvScaleFactors[maxLevel]
                    : 1.0f;
                const float sigmaDisp = 0.5f * scale;
                const float relDepthSigma = sigmaDisp / disparity;
                if(!std::isfinite(relDepthSigma) || relDepthSigma > stereoMaxRelDepthSigma)
                {
                    ++nUncertaintyReject;
                    continue;
                }
            }

            nAccepted++;
            mvDepth[iL]=depth;
            mvuRight[iL] = bestuR;
            mvStereoBirthQuality[iL] = birthQuality;
            vDistIdx.push_back(pair<float,int>(bestDist,iL));
        }
    }

    if(vDistIdx.empty())
    {
        if(ShouldPrintXFeatFrameDebug(mnId) && bFloatDescriptors)
        {
            std::cout << "[ComputeStereoMatches] frame=" << mnId
                      << " N=" << N
                      << " right_N=" << Nr
                      << " forward_candidates=" << nForwardCandidates
                      << " desc_pass=" << nDescPass
                      << " ratio_reject=" << nRatioReject
                      << " mutual_reject=" << nMutualReject
                      << " epipolar_reject=" << nEpipolarReject
                      << " disparity_reject=" << nDisparityReject
                      << " uncert_reject=" << nUncertaintyReject
                      << " subpix_refined=" << nSubpixelRefined
                      << " subpix_fallback=" << nSubpixelFallback
                      << " accepted_before_median=" << nAccepted
                      << " close_candidates=" << nCloseCandidates
                      << " close_high_conf=" << nCloseHighConf
                      << " close_low_conf=" << nCloseLowConf
                      << " reject_close_y=" << nRejectCloseY
                      << " reject_close_lr=" << nRejectCloseLR
                      << " close_subpix_attempt=" << nCloseSubpixelAttempt
                      << " close_subpix_refined=" << nCloseSubpixelRefined
                      << " close_subpix_reject=" << nCloseSubpixelReject
                      << " close_uncert_reject=" << nCloseUncertaintyReject
                      << " close_flat_curve_reject=" << nCloseFlatCurveReject
                      << " close_combined_curve_reject=" << nCloseCombinedCurveReject
                      << " close_low_margin_reject=" << nCloseLowMarginReject
                      << " close_lr_consistency_reject=" << nCloseLRConsistencyReject
                      << " close_low_peak_reject=" << nCloseLowPeakReject
                      << " close_sigma_model_reject=" << nCloseSigmaModelReject
                      << " close_quality_sigma_reject=" << nCloseQualitySigmaReject
                      << " close_score_sigma_reject=" << nCloseScoreSigmaReject
                      << " close_combined_sigma_reject=" << nCloseCombinedSigmaReject
                      << " close_topk_candidates=" << nCloseTopKCandidates
                      << " close_candidate_ambiguous_reject=" << nCloseCandidateAmbiguousReject
                      << " close_candidate_rescore_winner_nonbestidx=" << nCloseCandidateRescoreWinnerNonBestIdx
                      << " close_patch_search_attempt=" << nClosePatchSearchAttempt
                      << " close_patch_search_ambiguous_reject=" << nClosePatchSearchAmbiguousReject
                      << " close_patch_search_winner_shift_gt1px=" << nClosePatchSearchWinnerShiftGt1Px
                      << " close_patch_sigma_reject=" << nClosePatchSigmaReject
                      << " close_band_search_attempt=" << nCloseBandSearchAttempt
                      << " close_band_search_ambiguous_reject=" << nCloseBandSearchAmbiguousReject
                      << " close_band_search_width_gt12px=" << nCloseBandSearchWidthGt12Px
                      << " close_band_sigma_reject=" << nCloseBandSigmaReject
                      << " close_cost_volume_attempt=" << nCloseCostVolumeAttempt
                      << " close_cost_volume_ambiguous_reject=" << nCloseCostVolumeAmbiguousReject
                      << " close_cost_volume_sigma_reject=" << nCloseCostVolumeSigmaReject
                      << " median_reject=0"
                      << " accepted_after_median=0"
                      << " depth_before_median=0"
                      << " median_dist=-1"
                      << " th_dist=-1"
                      << " disp_p50=-1"
                      << " disp_p90=-1"
                      << " disp_p99=-1"
                      << " depth_p50=-1"
                      << " depth_p90=-1"
                      << " depth_p99=-1"
                      << " depth_gt50=0"
                      << " depth_gt100=0"
                      << " close_depth=0"
                      << std::endl;
        }
        return;
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=static_cast<int>(vDistIdx.size())-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
            mvStereoBirthQuality[vDistIdx[i].second]=kStereoBirthInvalid;
            if(bFloatDescriptors)
                ++nMedianReject;
        }
    }

    if(ShouldPrintXFeatFrameDebug(mnId) && bFloatDescriptors)
    {
        int nDepthBeforeMedian = static_cast<int>(vDistIdx.size());
        std::vector<float> vDepths;
        std::vector<float> vDisparities;
        vDepths.reserve(vDistIdx.size());
        vDisparities.reserve(vDistIdx.size());
        int nDepthGt50 = 0;
        int nDepthGt100 = 0;
        int nCloseDepth = 0;
        for(const auto& distIdx : vDistIdx)
        {
            const int idx = distIdx.second;
            if(idx < 0 || idx >= static_cast<int>(mvDepth.size()))
                continue;
            const float depth = mvDepth[idx];
            if(depth <= 0.0f || !std::isfinite(depth))
                continue;
            const float disparity = mbf / depth;
            vDepths.push_back(depth);
            vDisparities.push_back(disparity);
            if(depth > 50.0f)
                ++nDepthGt50;
            if(depth > 100.0f)
                ++nDepthGt100;
            if(depth < mThDepth)
                ++nCloseDepth;
        }
        std::sort(vDepths.begin(), vDepths.end());
        std::sort(vDisparities.begin(), vDisparities.end());
        std::cout << "[ComputeStereoMatches] frame=" << mnId
                  << " N=" << N
                  << " right_N=" << Nr
                  << " forward_candidates=" << nForwardCandidates
                  << " desc_pass=" << nDescPass
                  << " ratio_reject=" << nRatioReject
                  << " mutual_reject=" << nMutualReject
                  << " epipolar_reject=" << nEpipolarReject
                  << " disparity_reject=" << nDisparityReject
                  << " uncert_reject=" << nUncertaintyReject
                  << " subpix_refined=" << nSubpixelRefined
                  << " subpix_fallback=" << nSubpixelFallback
                  << " accepted_before_median=" << nAccepted
                  << " close_candidates=" << nCloseCandidates
                  << " close_high_conf=" << nCloseHighConf
                  << " close_low_conf=" << nCloseLowConf
                  << " reject_close_y=" << nRejectCloseY
                  << " reject_close_lr=" << nRejectCloseLR
                  << " close_subpix_attempt=" << nCloseSubpixelAttempt
                  << " close_subpix_refined=" << nCloseSubpixelRefined
                  << " close_subpix_reject=" << nCloseSubpixelReject
                  << " close_uncert_reject=" << nCloseUncertaintyReject
                  << " close_flat_curve_reject=" << nCloseFlatCurveReject
                  << " close_combined_curve_reject=" << nCloseCombinedCurveReject
                  << " close_low_margin_reject=" << nCloseLowMarginReject
                  << " close_lr_consistency_reject=" << nCloseLRConsistencyReject
                  << " close_low_peak_reject=" << nCloseLowPeakReject
                  << " close_sigma_model_reject=" << nCloseSigmaModelReject
                  << " close_quality_sigma_reject=" << nCloseQualitySigmaReject
                  << " close_score_sigma_reject=" << nCloseScoreSigmaReject
                  << " close_combined_sigma_reject=" << nCloseCombinedSigmaReject
                  << " close_topk_candidates=" << nCloseTopKCandidates
                  << " close_candidate_ambiguous_reject=" << nCloseCandidateAmbiguousReject
                  << " close_candidate_rescore_winner_nonbestidx=" << nCloseCandidateRescoreWinnerNonBestIdx
                  << " close_patch_search_attempt=" << nClosePatchSearchAttempt
                  << " close_patch_search_ambiguous_reject=" << nClosePatchSearchAmbiguousReject
                  << " close_patch_search_winner_shift_gt1px=" << nClosePatchSearchWinnerShiftGt1Px
                  << " close_patch_sigma_reject=" << nClosePatchSigmaReject
                  << " close_band_search_attempt=" << nCloseBandSearchAttempt
                  << " close_band_search_ambiguous_reject=" << nCloseBandSearchAmbiguousReject
                  << " close_band_search_width_gt12px=" << nCloseBandSearchWidthGt12Px
                  << " close_band_sigma_reject=" << nCloseBandSigmaReject
                  << " close_cost_volume_attempt=" << nCloseCostVolumeAttempt
                  << " close_cost_volume_ambiguous_reject=" << nCloseCostVolumeAmbiguousReject
                  << " close_cost_volume_sigma_reject=" << nCloseCostVolumeSigmaReject
                  << " median_reject=" << nMedianReject
                  << " accepted_after_median=" << (nAccepted - nMedianReject)
                  << " depth_before_median=" << nDepthBeforeMedian
                  << " median_dist=" << median
                  << " th_dist=" << thDist
                  << " disp_p50=" << PercentileFromSorted(vDisparities, 0.50f)
                  << " disp_p90=" << PercentileFromSorted(vDisparities, 0.90f)
                  << " disp_p99=" << PercentileFromSorted(vDisparities, 0.99f)
                  << " depth_p50=" << PercentileFromSorted(vDepths, 0.50f)
                  << " depth_p90=" << PercentileFromSorted(vDepths, 0.90f)
                  << " depth_p99=" << PercentileFromSorted(vDepths, 0.99f)
                  << " depth_gt50=" << nDepthGt50
                  << " depth_gt100=" << nDepthGt100
                  << " close_depth=" << nCloseDepth
                  << std::endl;
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);
    mvStereoBirthQuality = vector<unsigned char>(N, kStereoBirthInvalid);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

bool Frame::UnprojectStereo(const int &i, Eigen::Vector3f &x3D)
{
    const float z = mvDepth[i];
    if(z>0) {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        Eigen::Vector3f x3Dc(x, y, z);
        x3D = mRwc * x3Dc + mOw;
        return true;
    } else
        return false;
}

bool Frame::imuIsPreintegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    return mbImuPreintegrated;
}

void Frame::setIntegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    mbImuPreintegrated = true;
}

Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, GeometricCamera* pCamera2, Sophus::SE3f& Tlr,Frame* pPrevF, const IMU::Calib &ImuCalib)
        :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),  mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
         mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mpCamera(pCamera), mpCamera2(pCamera2),
         mbHasPose(false), mbHasVelocity(false)

{
    imgLeft = imLeft.clone();
    imgRight = imRight.clone();

    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft,static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1]);
    thread threadRight(&Frame::ExtractORB,this,1,imRight,static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1]);
    threadLeft.join();
    threadRight.join();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    Nleft = mvKeys.size();
    Nright = mvKeysRight.size();
    N = Nleft + Nright;

    if(N == 0)
        return;

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf / fx;

    // Sophus/Eigen
    mTlr = Tlr;
    mTrl = mTlr.inverse();
    mRlr = mTlr.rotationMatrix();
    mtlr = mTlr.translation();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
    ComputeStereoFishEyeMatches();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

    //Put all descriptors in the same matrix
    cv::vconcat(mDescriptors,mDescriptorsRight,mDescriptors);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(nullptr));
    mvbOutlier = vector<bool>(N,false);
    mvXFeatMatchSource = vector<unsigned char>(N, 0);

    AssignFeaturesToGrid();

    mpMutexImu = new std::mutex();

    UndistortKeyPoints();

}

void Frame::ComputeStereoFishEyeMatches() {
    //Speed it up by matching keypoints in the lapping area
    vector<cv::KeyPoint> stereoLeft(mvKeys.begin() + monoLeft, mvKeys.end());
    vector<cv::KeyPoint> stereoRight(mvKeysRight.begin() + monoRight, mvKeysRight.end());

    cv::Mat stereoDescLeft = mDescriptors.rowRange(monoLeft, mDescriptors.rows);
    cv::Mat stereoDescRight = mDescriptorsRight.rowRange(monoRight, mDescriptorsRight.rows);

    mvLeftToRightMatch = vector<int>(Nleft,-1);
    mvRightToLeftMatch = vector<int>(Nright,-1);
    mvDepth = vector<float>(Nleft,-1.0f);
    mvuRight = vector<float>(Nleft,-1);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(Nleft);
    mnCloseMPs = 0;

    //Perform a brute force between Keypoint in the left and right image
    vector<vector<cv::DMatch>> matches;

    BFmatcher.knnMatch(stereoDescLeft,stereoDescRight,matches,2);

    int nMatches = 0;
    int descMatches = 0;

    //Check matches using Lowe's ratio
    for(vector<vector<cv::DMatch>>::iterator it = matches.begin(); it != matches.end(); ++it){
        if((*it).size() >= 2 && (*it)[0].distance < (*it)[1].distance * 0.7){
            //For every good match, check parallax and reprojection error to discard spurious matches
            Eigen::Vector3f p3D;
            descMatches++;
            float sigma1 = mvLevelSigma2[mvKeys[(*it)[0].queryIdx + monoLeft].octave], sigma2 = mvLevelSigma2[mvKeysRight[(*it)[0].trainIdx + monoRight].octave];
            float depth = static_cast<KannalaBrandt8*>(mpCamera)->TriangulateMatches(mpCamera2,mvKeys[(*it)[0].queryIdx + monoLeft],mvKeysRight[(*it)[0].trainIdx + monoRight],mRlr,mtlr,sigma1,sigma2,p3D);
            if(depth > 0.0001f){
                mvLeftToRightMatch[(*it)[0].queryIdx + monoLeft] = (*it)[0].trainIdx + monoRight;
                mvRightToLeftMatch[(*it)[0].trainIdx + monoRight] = (*it)[0].queryIdx + monoLeft;
                mvStereo3Dpoints[(*it)[0].queryIdx + monoLeft] = p3D;
                mvDepth[(*it)[0].queryIdx + monoLeft] = depth;
                nMatches++;
            }
        }
    }
}

bool Frame::isInFrustumChecks(MapPoint *pMP, float viewingCosLimit, bool bRight,
                              unsigned char* pFrustumStatus) {
    if(pFrustumStatus)
        *pFrustumStatus = static_cast<unsigned char>(MapPoint::kTrackFrustumUnchecked);
    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    Eigen::Matrix3f mR;
    Eigen::Vector3f mt, twc;
    if(bRight){
        Eigen::Matrix3f Rrl = mTrl.rotationMatrix();
        Eigen::Vector3f trl = mTrl.translation();
        mR = Rrl * mRcw;
        mt = Rrl * mtcw + trl;
        twc = mRwc * mTlr.translation() + mOw;
    }
    else{
        mR = mRcw;
        mt = mtcw;
        twc = mOw;
    }

    // 3D in camera coordinates
    Eigen::Vector3f Pc = mR * P + mt;
    const float Pc_dist = Pc.norm();
    const float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        if(pFrustumStatus)
            *pFrustumStatus = static_cast<unsigned char>(MapPoint::kTrackFrustumNegativeDepth);
        return false;
    }

    // Project in image and check it is not outside
    Eigen::Vector2f uv;
    if(bRight) uv = mpCamera2->project(Pc);
    else uv = mpCamera->project(Pc);

    if(uv(0)<mnMinX || uv(0)>mnMaxX)
    {
        if(pFrustumStatus)
            *pFrustumStatus = static_cast<unsigned char>(MapPoint::kTrackFrustumOutsideImage);
        return false;
    }
    if(uv(1)<mnMinY || uv(1)>mnMaxY)
    {
        if(pFrustumStatus)
            *pFrustumStatus = static_cast<unsigned char>(MapPoint::kTrackFrustumOutsideImage);
        return false;
    }

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const Eigen::Vector3f PO = P - twc;
    const float dist = PO.norm();

    if(dist<minDistance || dist>maxDistance)
    {
        if(pFrustumStatus)
            *pFrustumStatus = static_cast<unsigned char>(MapPoint::kTrackFrustumScaleRange);
        return false;
    }

    // Check viewing angle
    Eigen::Vector3f Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn) / dist;

    if(viewCos<viewingCosLimit)
    {
        if(pFrustumStatus)
            *pFrustumStatus = static_cast<unsigned char>(MapPoint::kTrackFrustumViewAngle);
        return false;
    }

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    if(bRight){
        pMP->mTrackProjXR = uv(0);
        pMP->mTrackProjYR = uv(1);
        pMP->mnTrackScaleLevelR= nPredictedLevel;
        pMP->mTrackViewCosR = viewCos;
        pMP->mTrackDepthR = Pc_dist;
    }
    else{
        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjY = uv(1);
        pMP->mnTrackScaleLevel= nPredictedLevel;
        pMP->mTrackViewCos = viewCos;
        pMP->mTrackDepth = Pc_dist;
    }

    if(pFrustumStatus)
        *pFrustumStatus = static_cast<unsigned char>(MapPoint::kTrackFrustumAccepted);
    return true;
}

Eigen::Vector3f Frame::UnprojectStereoFishEye(const int &i){
    return mRwc * mvStereo3Dpoints[i] + mOw;
}

} //namespace ORB_SLAM
