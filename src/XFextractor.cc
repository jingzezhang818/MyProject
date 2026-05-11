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

/**
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <numeric>
#include <string>
#include <list>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <unordered_map>
#include <unordered_set>

#include "XFextractor.h"
#include "XFeatLighterGlue/core.hpp"


using namespace cv;
using namespace std;

namespace ORB_SLAM3
{

    namespace
    {
        //调试: 解析布尔环境变量，支持 0/false/FALSE 关闭。
        bool IsEnvFlagEnabled(const char* key)
        {
            const char* env = std::getenv(key);
            if(!env)
                return false;

            const std::string v(env);
            return !(v.empty() || v == "0" || v == "false" || v == "FALSE");
        }

        //调试: 解析整型环境变量，失败则回退默认值。
        int GetEnvIntWithRange(const char* key, const int fallback, const int minValue, const int maxValue)
        {
            const char* env = std::getenv(key);
            if(!env)
                return fallback;

            try
            {
                const int v = std::stoi(std::string(env));
                return std::max(minValue, std::min(maxValue, v));
            }
            catch(...)
            {
                return fallback;
            }
        }

        //调试: 解析浮点环境变量，失败则回退默认值。
        float GetEnvFloatWithRange(const char* key, const float fallback, const float minValue, const float maxValue)
        {
            const char* env = std::getenv(key);
            if(!env)
                return fallback;

            try
            {
                const float v = std::stof(std::string(env));
                if(std::isfinite(v))
                    return std::max(minValue, std::min(maxValue, v));
            }
            catch(...)
            {
            }
            return fallback;
        }

        bool IsXFeatProfileEnabled()
        {
            static const bool enabled = IsEnvFlagEnabled("XFEAT_PROFILE");
            return enabled;
        }

        int GetXFeatFixedNMSCandidateFactor()
        {
            static const int factor = GetEnvIntWithRange("XFEAT_FIXED_NMS_CANDIDATE_FACTOR", 8, 1, 16);
            return factor;
        }

        int GetXFeatFixedNMSCandidateMin()
        {
            static const int minCandidates = GetEnvIntWithRange("XFEAT_FIXED_NMS_CANDIDATE_MIN", 1024, 64, 8192);
            return minCandidates;
        }

        //调试: XFeat关键点空间均匀化开关，默认开启；设 XFEAT_UNIFORM_DISABLE=1 可关闭。
        bool UseXFeatUniformSelection()
        {
            static const bool enabled = !IsEnvFlagEnabled("XFEAT_UNIFORM_DISABLE");
            return enabled;
        }

        //调试: XFeat提点均匀化升级总开关。默认启用 OctTree+ANMS；设 XFEAT_UNIFORM_LEGACY_GRID=1 回退旧策略。
        bool UseXFeatUniformOctTreeANMS()
        {
            static const bool enabled = !IsEnvFlagEnabled("XFEAT_UNIFORM_LEGACY_GRID");
            return enabled;
        }

        int GetXFeatUniformGridCols()
        {
            //调试: 每层关键点空间均匀化网格列数。
            static const int cols = GetEnvIntWithRange("XFEAT_UNIFORM_GRID_COLS", 8, 2, 64);
            return cols;
        }

        int GetXFeatUniformGridRows()
        {
            //调试: 每层关键点空间均匀化网格行数。
            static const int rows = GetEnvIntWithRange("XFEAT_UNIFORM_GRID_ROWS", 6, 2, 64);
            return rows;
        }

        int GetXFeatUniformANMSMinRadius()
        {
            //调试: ANMS最小半径（像素，金字塔层内坐标）。
            static const int v = GetEnvIntWithRange("XFEAT_UNIFORM_ANMS_MIN_RADIUS", 2, 0, 128);
            return v;
        }

        int GetXFeatUniformANMSMaxRadius()
        {
            //调试: ANMS最大半径（像素，0表示按图像尺寸和目标点数自适应）。
            static const int v = GetEnvIntWithRange("XFEAT_UNIFORM_ANMS_MAX_RADIUS", 0, 0, 512);
            return v;
        }

        float GetXFeatUniformANMSAutoScale()
        {
            //调试: ANMS自适应半径尺度系数，越大越强调均匀覆盖。
            static const float v = GetEnvFloatWithRange("XFEAT_UNIFORM_ANMS_AUTO_SCALE", 1.5f, 0.5f, 5.0f);
            return v;
        }

        struct XFeatOctreeNode
        {
            XFeatOctreeNode()
                : bNoMore(false)
            {
            }

            void DivideNode(XFeatOctreeNode& n1,
                            XFeatOctreeNode& n2,
                            XFeatOctreeNode& n3,
                            XFeatOctreeNode& n4,
                            const float* kptPtr) const
            {
                const int halfX = std::max(1, static_cast<int>(std::ceil(static_cast<float>(UR.x - UL.x) * 0.5f)));
                const int halfY = std::max(1, static_cast<int>(std::ceil(static_cast<float>(BR.y - UL.y) * 0.5f)));

                n1.UL = UL;
                n1.UR = cv::Point2i(UL.x + halfX, UL.y);
                n1.BL = cv::Point2i(UL.x, UL.y + halfY);
                n1.BR = cv::Point2i(UL.x + halfX, UL.y + halfY);
                n1.vIdx.reserve(vIdx.size());

                n2.UL = n1.UR;
                n2.UR = UR;
                n2.BL = n1.BR;
                n2.BR = cv::Point2i(UR.x, UL.y + halfY);
                n2.vIdx.reserve(vIdx.size());

                n3.UL = n1.BL;
                n3.UR = n1.BR;
                n3.BL = BL;
                n3.BR = cv::Point2i(n1.BR.x, BL.y);
                n3.vIdx.reserve(vIdx.size());

                n4.UL = n3.UR;
                n4.UR = n2.BR;
                n4.BL = n3.BR;
                n4.BR = BR;
                n4.vIdx.reserve(vIdx.size());

                for(const int idx : vIdx)
                {
                    const float x = kptPtr[idx * 2];
                    const float y = kptPtr[idx * 2 + 1];

                    if(x < n1.UR.x)
                    {
                        if(y < n1.BR.y)
                            n1.vIdx.push_back(idx);
                        else
                            n3.vIdx.push_back(idx);
                    }
                    else
                    {
                        if(y < n1.BR.y)
                            n2.vIdx.push_back(idx);
                        else
                            n4.vIdx.push_back(idx);
                    }
                }

                if(n1.vIdx.size() <= 1)
                    n1.bNoMore = true;
                if(n2.vIdx.size() <= 1)
                    n2.bNoMore = true;
                if(n3.vIdx.size() <= 1)
                    n3.bNoMore = true;
                if(n4.vIdx.size() <= 1)
                    n4.bNoMore = true;
            }

            std::vector<int> vIdx;
            cv::Point2i UL, UR, BL, BR;
            std::list<XFeatOctreeNode>::iterator lit;
            bool bNoMore;
        };

        bool CompareNodeBySizeThenX(const std::pair<int, XFeatOctreeNode*>& a,
                                    const std::pair<int, XFeatOctreeNode*>& b)
        {
            if(a.first != b.first)
                return a.first < b.first;
            return a.second->UL.x < b.second->UL.x;
        }

        bool HasPrefix(const std::string& value, const std::string& prefix)
        {
            return value.rfind(prefix, 0) == 0;
        }

        void StripPrefix(std::string& value, const std::string& prefix)
        {
            if(HasPrefix(value, prefix))
                value.erase(0, prefix.size());
        }

        std::string MapXFeatExtractorWeightKey(const std::string& key)
        {
            std::string mapped = key;
            StripPrefix(mapped, "module.");
            StripPrefix(mapped, "extractor.");
            StripPrefix(mapped, "model.");
            StripPrefix(mapped, "net.");
            StripPrefix(mapped, "extractor.model.");
            StripPrefix(mapped, "extractor.model.net.");
            return mapped;
        }

        bool IsPotentialXFeatExtractorWeightKey(const std::string& key)
        {
            return HasPrefix(key, "skip1.") ||
                   HasPrefix(key, "block1.") ||
                   HasPrefix(key, "block2.") ||
                   HasPrefix(key, "block3.") ||
                   HasPrefix(key, "block4.") ||
                   HasPrefix(key, "block5.") ||
                   HasPrefix(key, "block_fusion.") ||
                   HasPrefix(key, "heatmap_head.") ||
                   HasPrefix(key, "keypoint_head.") ||
                   HasPrefix(key, "fine_matcher.") ||
                   HasPrefix(key, "norm.");
        }

        std::string TensorShapeString(const torch::Tensor& tensor)
        {
            std::ostringstream oss;
            oss << "[";
            for(size_t i = 0; i < tensor.sizes().size(); ++i)
            {
                if(i != 0)
                    oss << ", ";
                oss << tensor.sizes()[i];
            }
            oss << "]";
            return oss.str();
        }

        bool SameShape(const torch::Tensor& a, const torch::Tensor& b)
        {
            return a.sizes().vec() == b.sizes().vec();
        }

        bool LoadXFeatPythonStateDict(
            const std::shared_ptr<XFeatModel>& model,
            const std::vector<std::pair<std::string, torch::Tensor>>& weights,
            const torch::Device& device)
        {
            torch::NoGradGuard no_grad;

            std::unordered_map<std::string, torch::Tensor> params;
            std::unordered_map<std::string, torch::Tensor> buffers;
            std::unordered_set<std::string> loadedParams;
            std::unordered_set<std::string> loadedBuffers;

            for(const auto& p : model->named_parameters(true))
                params[p.key()] = p.value();
            for(const auto& b : model->named_buffers(true))
                buffers[b.key()] = b.value();

            int loaded = 0;
            int missing = 0;
            int unexpected = 0;
            int skipped = 0;
            int shapeMismatch = 0;

            for(const auto& item : weights)
            {
                const std::string& loadedKey = item.first;
                const std::string mappedKey = MapXFeatExtractorWeightKey(loadedKey);

                if(!IsPotentialXFeatExtractorWeightKey(mappedKey))
                {
                    ++skipped;
                    std::cout << "[XFextractor] skipped non-extractor key: " << loadedKey << std::endl;
                    continue;
                }

                std::cout << "[XFextractor] loaded key: " << loadedKey << std::endl;
                std::cout << "[XFextractor] mapped key: " << loadedKey
                          << " -> " << mappedKey << std::endl;

                auto pIt = params.find(mappedKey);
                if(pIt != params.end())
                {
                    if(!SameShape(pIt->second, item.second))
                    {
                        ++shapeMismatch;
                        std::cerr << "[XFextractor] shape mismatch: " << loadedKey
                                  << " -> " << mappedKey
                                  << " checkpoint=" << TensorShapeString(item.second)
                                  << " model=" << TensorShapeString(pIt->second)
                                  << std::endl;
                        continue;
                    }
                    pIt->second.copy_(item.second.to(device).to(pIt->second.scalar_type()));
                    loadedParams.insert(mappedKey);
                    ++loaded;
                    continue;
                }

                auto bIt = buffers.find(mappedKey);
                if(bIt != buffers.end())
                {
                    if(!SameShape(bIt->second, item.second))
                    {
                        ++shapeMismatch;
                        std::cerr << "[XFextractor] shape mismatch: " << loadedKey
                                  << " -> " << mappedKey
                                  << " checkpoint=" << TensorShapeString(item.second)
                                  << " model=" << TensorShapeString(bIt->second)
                                  << std::endl;
                        continue;
                    }
                    bIt->second.copy_(item.second.to(device).to(bIt->second.scalar_type()));
                    loadedBuffers.insert(mappedKey);
                    ++loaded;
                    continue;
                }

                ++unexpected;
                std::cout << "[XFextractor] unexpected key: " << loadedKey
                          << " -> " << mappedKey << std::endl;
            }

            for(const auto& p : params)
            {
                if(loadedParams.find(p.first) == loadedParams.end())
                {
                    ++missing;
                    std::cout << "[XFextractor] missing key: " << p.first << std::endl;
                }
            }
            for(const auto& b : buffers)
            {
                if(loadedBuffers.find(b.first) == loadedBuffers.end())
                {
                    ++missing;
                    std::cout << "[XFextractor] missing key: " << b.first << std::endl;
                }
            }

            std::cout << "[XFextractor] load report: loaded=" << loaded
                      << " missing=" << missing
                      << " unexpected=" << unexpected
                      << " skipped=" << skipped
                      << " shape_mismatch=" << shapeMismatch
                      << std::endl;

            if(shapeMismatch > 0)
                throw std::runtime_error("XFextractor checkpoint shape mismatch.");
            if(loaded == 0)
                throw std::runtime_error("XFextractor checkpoint did not contain extractor weights.");
            if(missing > 0 || unexpected > 0)
                throw std::runtime_error("XFextractor checkpoint key mismatch.");

            return true;
        }

        std::vector<int> ApplyGreedyNMSByRadius(const float* kptPtr,
                                                const std::vector<int>& sortedByScore,
                                                const int radius,
                                                const int imageWidth,
                                                const int imageHeight)
        {
            if(sortedByScore.empty())
                return {};

            if(radius <= 0)
                return sortedByScore;

            const int safeW = std::max(1, imageWidth);
            const int safeH = std::max(1, imageHeight);
            const int cellSize = std::max(1, radius);
            const int gridCols = std::max(1, (safeW + cellSize - 1) / cellSize);
            const int gridRows = std::max(1, (safeH + cellSize - 1) / cellSize);
            const int totalCells = gridCols * gridRows;

            std::vector<std::vector<int>> cellPoints(totalCells);
            std::vector<int> selected;
            selected.reserve(sortedByScore.size());

            const float radius2 = static_cast<float>(radius * radius);
            for(const int idx : sortedByScore)
            {
                const float x = kptPtr[idx * 2];
                const float y = kptPtr[idx * 2 + 1];

                int cellX = static_cast<int>(x / static_cast<float>(cellSize));
                int cellY = static_cast<int>(y / static_cast<float>(cellSize));
                cellX = std::max(0, std::min(gridCols - 1, cellX));
                cellY = std::max(0, std::min(gridRows - 1, cellY));

                bool suppressed = false;
                const int minCellX = std::max(0, cellX - 1);
                const int maxCellX = std::min(gridCols - 1, cellX + 1);
                const int minCellY = std::max(0, cellY - 1);
                const int maxCellY = std::min(gridRows - 1, cellY + 1);

                for(int cy = minCellY; cy <= maxCellY && !suppressed; ++cy)
                {
                    for(int cx = minCellX; cx <= maxCellX && !suppressed; ++cx)
                    {
                        const std::vector<int>& pts = cellPoints[cy * gridCols + cx];
                        for(const int selectedIdx : pts)
                        {
                            const float dx = x - kptPtr[selectedIdx * 2];
                            const float dy = y - kptPtr[selectedIdx * 2 + 1];
                            if(dx * dx + dy * dy < radius2)
                            {
                                suppressed = true;
                                break;
                            }
                        }
                    }
                }

                if(suppressed)
                    continue;

                selected.push_back(idx);
                cellPoints[cellY * gridCols + cellX].push_back(idx);
            }

            return selected;
        }

        std::vector<int> SelectByOctTree(const float* kptPtr,
                                         const float* scorePtr,
                                         const std::vector<int>& candidateIdx,
                                         const int keepK,
                                         const int imageWidth,
                                         const int imageHeight,
                                         const int totalPointCount)
        {
            std::vector<int> selected;
            if(candidateIdx.empty() || keepK <= 0)
                return selected;

            const int keep = std::min(keepK, static_cast<int>(candidateIdx.size()));
            if(keep <= 0)
                return selected;

            const int minX = 0;
            const int minY = 0;
            const int maxX = std::max(1, imageWidth);
            const int maxY = std::max(1, imageHeight);

            int nIni = static_cast<int>(std::round(static_cast<float>(maxX - minX) / std::max(1, maxY - minY)));
            if(nIni <= 0)
                nIni = 1;

            const float hX = static_cast<float>(maxX - minX) / nIni;

            std::list<XFeatOctreeNode> lNodes;
            std::vector<XFeatOctreeNode*> vpIniNodes(static_cast<size_t>(nIni), nullptr);

            for(int i = 0; i < nIni; ++i)
            {
                XFeatOctreeNode ni;
                ni.UL = cv::Point2i(static_cast<int>(hX * i), 0);
                ni.UR = cv::Point2i(static_cast<int>(hX * (i + 1)), 0);
                ni.BL = cv::Point2i(ni.UL.x, maxY - minY);
                ni.BR = cv::Point2i(ni.UR.x, maxY - minY);
                ni.vIdx.reserve(candidateIdx.size());

                lNodes.push_back(ni);
                vpIniNodes[i] = &lNodes.back();
            }

            for(const int idx : candidateIdx)
            {
                int bucket = static_cast<int>(kptPtr[idx * 2] / std::max(1e-3f, hX));
                bucket = std::max(0, std::min(nIni - 1, bucket));
                vpIniNodes[bucket]->vIdx.push_back(idx);
            }

            auto lit = lNodes.begin();
            while(lit != lNodes.end())
            {
                if(lit->vIdx.size() == 1)
                {
                    lit->bNoMore = true;
                    ++lit;
                }
                else if(lit->vIdx.empty())
                {
                    lit = lNodes.erase(lit);
                }
                else
                {
                    ++lit;
                }
            }

            bool bFinish = false;
            std::vector<std::pair<int, XFeatOctreeNode*>> vSizeAndPointerToNode;
            vSizeAndPointerToNode.reserve(lNodes.size() * 4);

            while(!bFinish)
            {
                const int prevSize = static_cast<int>(lNodes.size());
                int nToExpand = 0;
                vSizeAndPointerToNode.clear();

                lit = lNodes.begin();
                while(lit != lNodes.end())
                {
                    if(lit->bNoMore)
                    {
                        ++lit;
                        continue;
                    }

                    XFeatOctreeNode n1, n2, n3, n4;
                    lit->DivideNode(n1, n2, n3, n4, kptPtr);

                    if(!n1.vIdx.empty())
                    {
                        lNodes.push_front(n1);
                        if(lNodes.front().vIdx.size() > 1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.emplace_back(lNodes.front().vIdx.size(), &lNodes.front());
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(!n2.vIdx.empty())
                    {
                        lNodes.push_front(n2);
                        if(lNodes.front().vIdx.size() > 1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.emplace_back(lNodes.front().vIdx.size(), &lNodes.front());
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(!n3.vIdx.empty())
                    {
                        lNodes.push_front(n3);
                        if(lNodes.front().vIdx.size() > 1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.emplace_back(lNodes.front().vIdx.size(), &lNodes.front());
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(!n4.vIdx.empty())
                    {
                        lNodes.push_front(n4);
                        if(lNodes.front().vIdx.size() > 1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.emplace_back(lNodes.front().vIdx.size(), &lNodes.front());
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lit = lNodes.erase(lit);
                }

                if(static_cast<int>(lNodes.size()) >= keep || static_cast<int>(lNodes.size()) == prevSize)
                {
                    bFinish = true;
                }
                else if(static_cast<int>(lNodes.size()) + nToExpand * 3 > keep)
                {
                    while(!bFinish)
                    {
                        const int prevInnerSize = static_cast<int>(lNodes.size());
                        std::vector<std::pair<int, XFeatOctreeNode*>> vPrev = vSizeAndPointerToNode;
                        vSizeAndPointerToNode.clear();

                        std::sort(vPrev.begin(), vPrev.end(), CompareNodeBySizeThenX);
                        for(int j = static_cast<int>(vPrev.size()) - 1; j >= 0; --j)
                        {
                            XFeatOctreeNode n1, n2, n3, n4;
                            vPrev[j].second->DivideNode(n1, n2, n3, n4, kptPtr);

                            if(!n1.vIdx.empty())
                            {
                                lNodes.push_front(n1);
                                if(lNodes.front().vIdx.size() > 1)
                                {
                                    vSizeAndPointerToNode.emplace_back(lNodes.front().vIdx.size(), &lNodes.front());
                                    lNodes.front().lit = lNodes.begin();
                                }
                            }
                            if(!n2.vIdx.empty())
                            {
                                lNodes.push_front(n2);
                                if(lNodes.front().vIdx.size() > 1)
                                {
                                    vSizeAndPointerToNode.emplace_back(lNodes.front().vIdx.size(), &lNodes.front());
                                    lNodes.front().lit = lNodes.begin();
                                }
                            }
                            if(!n3.vIdx.empty())
                            {
                                lNodes.push_front(n3);
                                if(lNodes.front().vIdx.size() > 1)
                                {
                                    vSizeAndPointerToNode.emplace_back(lNodes.front().vIdx.size(), &lNodes.front());
                                    lNodes.front().lit = lNodes.begin();
                                }
                            }
                            if(!n4.vIdx.empty())
                            {
                                lNodes.push_front(n4);
                                if(lNodes.front().vIdx.size() > 1)
                                {
                                    vSizeAndPointerToNode.emplace_back(lNodes.front().vIdx.size(), &lNodes.front());
                                    lNodes.front().lit = lNodes.begin();
                                }
                            }

                            lNodes.erase(vPrev[j].second->lit);
                            if(static_cast<int>(lNodes.size()) >= keep)
                                break;
                        }

                        if(static_cast<int>(lNodes.size()) >= keep || static_cast<int>(lNodes.size()) == prevInnerSize)
                            bFinish = true;
                    }
                }
            }

            selected.reserve(std::min(keep, static_cast<int>(lNodes.size())));
            for(auto nodeIt = lNodes.begin(); nodeIt != lNodes.end(); ++nodeIt)
            {
                const std::vector<int>& nodeIdx = nodeIt->vIdx;
                if(nodeIdx.empty())
                    continue;

                int bestIdx = nodeIdx[0];
                float bestScore = scorePtr[bestIdx];
                for(size_t k = 1; k < nodeIdx.size(); ++k)
                {
                    const int idx = nodeIdx[k];
                    const float sc = scorePtr[idx];
                    if(sc > bestScore || (sc == bestScore && idx < bestIdx))
                    {
                        bestIdx = idx;
                        bestScore = sc;
                    }
                }
                selected.push_back(bestIdx);
            }

            if(static_cast<int>(selected.size()) > keep)
            {
                std::sort(selected.begin(), selected.end(), [&](const int a, const int b) {
                    if(scorePtr[a] == scorePtr[b])
                        return a < b;
                    return scorePtr[a] > scorePtr[b];
                });
                selected.resize(keep);
            }

            if(static_cast<int>(selected.size()) < keep)
            {
                std::vector<char> used(static_cast<size_t>(std::max(0, totalPointCount)), 0);
                for(const int idx : selected)
                {
                    if(idx >= 0 && idx < static_cast<int>(used.size()))
                        used[idx] = 1;
                }

                std::vector<int> sortedCandidate = candidateIdx;
                std::sort(sortedCandidate.begin(), sortedCandidate.end(), [&](const int a, const int b) {
                    if(scorePtr[a] == scorePtr[b])
                        return a < b;
                    return scorePtr[a] > scorePtr[b];
                });

                for(const int idx : sortedCandidate)
                {
                    if(static_cast<int>(selected.size()) >= keep)
                        break;
                    if(idx < 0 || idx >= static_cast<int>(used.size()))
                        continue;
                    if(used[idx])
                        continue;
                    selected.push_back(idx);
                    used[idx] = 1;
                }
            }

            return selected;
        }

        std::vector<int> SelectByANMSAndOctTree(const float* kptPtr,
                                                const float* scorePtr,
                                                const int nPoints,
                                                const int keepK,
                                                const int imageWidth,
                                                const int imageHeight,
                                                const std::vector<int>& sortedByScore)
        {
            std::vector<int> selected;
            if(!kptPtr || !scorePtr || nPoints <= 0 || keepK <= 0 || sortedByScore.empty())
                return selected;

            const int keep = std::min(keepK, nPoints);
            if(keep <= 0)
                return selected;

            const int safeW = std::max(1, imageWidth);
            const int safeH = std::max(1, imageHeight);
            const float area = static_cast<float>(safeW * safeH);
            const float expectedSpacing = std::sqrt(area / std::max(1, keep));
            int minRadius = GetXFeatUniformANMSMinRadius();
            int maxRadius = GetXFeatUniformANMSMaxRadius();
            if(maxRadius <= 0)
                maxRadius = std::max(minRadius, static_cast<int>(std::round(expectedSpacing * GetXFeatUniformANMSAutoScale())));
            if(maxRadius < minRadius)
                std::swap(maxRadius, minRadius);

            std::vector<int> anmsCandidates = sortedByScore;
            if(maxRadius > 0)
            {
                int lo = minRadius;
                int hi = maxRadius;
                std::vector<int> best = sortedByScore;

                while(lo <= hi)
                {
                    const int mid = lo + (hi - lo) / 2;
                    std::vector<int> cur = ApplyGreedyNMSByRadius(kptPtr, sortedByScore, mid, safeW, safeH);
                    if(static_cast<int>(cur.size()) >= keep)
                    {
                        best.swap(cur);
                        lo = mid + 1;
                    }
                    else
                    {
                        hi = mid - 1;
                    }
                }
                anmsCandidates.swap(best);
            }

            if(static_cast<int>(anmsCandidates.size()) <= keep)
                return anmsCandidates;

            selected = SelectByOctTree(kptPtr,
                                       scorePtr,
                                       anmsCandidates,
                                       keep,
                                       safeW,
                                       safeH,
                                       nPoints);
            return selected;
        }

        std::vector<int> SelectUniformTopKIndicesLegacyGrid(const float* kptPtr,
                                                            const float* scorePtr,
                                                            const int nPoints,
                                                            const int keepK,
                                                            const int imageWidth,
                                                            const int imageHeight,
                                                            const std::vector<int>& sortedByScore)
        {
            std::vector<int> selected;
            if(!kptPtr || !scorePtr || nPoints <= 0 || keepK <= 0)
                return selected;

            const int keep = std::min(keepK, nPoints);
            if(keep <= 0)
                return selected;

            const std::vector<int>& order = sortedByScore;
            const int gridCols = GetXFeatUniformGridCols();
            const int gridRows = GetXFeatUniformGridRows();
            const int totalCells = gridCols * gridRows;
            const int perCellCap = std::max(1, (keep + totalCells - 1) / totalCells);

            const float safeWidth = std::max(1.0f, static_cast<float>(imageWidth));
            const float safeHeight = std::max(1.0f, static_cast<float>(imageHeight));
            std::vector<int> cellCount(totalCells, 0);
            std::vector<char> picked(nPoints, 0);
            selected.reserve(static_cast<size_t>(keep));

            //调试: 第一阶段按网格上限选点，抑制局部纹理过密区域。
            for(const int idx : order)
            {
                if(static_cast<int>(selected.size()) >= keep)
                    break;

                const float x = kptPtr[idx * 2];
                const float y = kptPtr[idx * 2 + 1];
                int cellX = static_cast<int>((x / safeWidth) * static_cast<float>(gridCols));
                int cellY = static_cast<int>((y / safeHeight) * static_cast<float>(gridRows));
                cellX = std::max(0, std::min(gridCols - 1, cellX));
                cellY = std::max(0, std::min(gridRows - 1, cellY));
                const int cell = cellY * gridCols + cellX;
                if(cellCount[cell] >= perCellCap)
                    continue;

                selected.push_back(idx);
                picked[idx] = 1;
                ++cellCount[cell];
            }

            //调试: 第二阶段补足剩余名额，保证特征总数不因均匀化减少。
            for(const int idx : order)
            {
                if(static_cast<int>(selected.size()) >= keep)
                    break;
                if(picked[idx])
                    continue;
                selected.push_back(idx);
            }

            return selected;
        }

        std::vector<int> SelectUniformTopKIndices(const float* kptPtr,
                                                  const float* scorePtr,
                                                  const int nPoints,
                                                  const int keepK,
                                                  const int imageWidth,
                                                  const int imageHeight)
        {
            std::vector<int> selected;
            if(!kptPtr || !scorePtr || nPoints <= 0 || keepK <= 0)
                return selected;

            const int keep = std::min(keepK, nPoints);
            std::vector<int> order(nPoints);
            std::iota(order.begin(), order.end(), 0);

            //调试: 统一按分数降序排序，后续不同均匀化策略共用同一基础优先级。
            std::sort(order.begin(), order.end(), [&](const int a, const int b) {
                if(scorePtr[a] == scorePtr[b])
                    return a < b;
                return scorePtr[a] > scorePtr[b];
            });

            if(!UseXFeatUniformSelection())
            {
                selected.assign(order.begin(), order.begin() + keep);
                return selected;
            }

            if(UseXFeatUniformOctTreeANMS())
            {
                //调试: 新主路径，先ANMS抑制局部扎堆，再用OctTree做全局分配，减少“网格限额+回填”带来的反复聚集。
                selected = SelectByANMSAndOctTree(kptPtr,
                                                  scorePtr,
                                                  nPoints,
                                                  keep,
                                                  imageWidth,
                                                  imageHeight,
                                                  order);
            }
            else
            {
                //调试: 兼容旧路径，便于快速A/B验证。
                selected = SelectUniformTopKIndicesLegacyGrid(kptPtr,
                                                              scorePtr,
                                                              nPoints,
                                                              keep,
                                                              imageWidth,
                                                              imageHeight,
                                                              order);
            }

            if(static_cast<int>(selected.size()) > keep)
                selected.resize(keep);

            if(static_cast<int>(selected.size()) < keep)
            {
                std::vector<char> used(static_cast<size_t>(nPoints), 0);
                for(const int idx : selected)
                {
                    if(idx >= 0 && idx < nPoints)
                        used[idx] = 1;
                }

                for(const int idx : order)
                {
                    if(static_cast<int>(selected.size()) >= keep)
                        break;
                    if(idx < 0 || idx >= nPoints || used[idx])
                        continue;
                    selected.push_back(idx);
                    used[idx] = 1;
                }
            }

            return selected;
        }
    }

    const int PATCH_SIZE = 31;
    const int HALF_PATCH_SIZE = 15;

    XFextractor::XFextractor(int _nfeatures, float _scaleFactor, int _nlevels,
                               int _iniThFAST, int _minThFAST)
        : XFextractor(_nfeatures, _scaleFactor, _nlevels,
                      _iniThFAST, _minThFAST, "weights/xfeat.pt")
    {
    }

    XFextractor::XFextractor(int _nfeatures, float _scaleFactor, int _nlevels,
                               int _iniThFAST, int _minThFAST,
                               const std::string& modelWeightsPath):
            nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
            iniThFAST(_iniThFAST), minThFAST(_minThFAST)
    {
        mvScaleFactor.resize(nlevels);
        mvLevelSigma2.resize(nlevels);
        mvScaleFactor[0]=1.0f;
        mvLevelSigma2[0]=1.0f;
        for(int i=1; i<nlevels; i++)
        {
            mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
            mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
        }

        mvInvScaleFactor.resize(nlevels);
        mvInvLevelSigma2.resize(nlevels);
        for(int i=0; i<nlevels; i++)
        {
            mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
            mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
        }
        
        mvImagePyramid.resize(nlevels);

        mnFeaturesPerLevel.resize(nlevels);
        float factor = 1.0f / scaleFactor;
        float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

        int sumFeatures = 0;
        for( int level = 0; level < nlevels-1; level++ )
        {
            mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
            sumFeatures += mnFeaturesPerLevel[level];
            nDesiredFeaturesPerScale *= factor;
        }
        mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

        //This is for orientation
        // pre-compute the end of a row in a circular patch
        umax.resize(HALF_PATCH_SIZE + 1);

        int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
        int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
        const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
        for (v = 0; v <= vmax; ++v)
            umax[v] = cvRound(sqrt(hp2 - v * v));

        // Make sure we are symmetric
        for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
        {
            while (umax[v0] == umax[v0 + 1])
                ++v0;
            umax[v] = v0;
            ++v0;
        }
        
        // load the xfeat model
        const std::string weights = modelWeightsPath.empty() ? "weights/xfeat.pt" : modelWeightsPath;
        const std::string resolvedWeights = getModelWeightsPath(weights);
        model = std::make_shared<XFeatModel>();
        try
        {
            torch::serialize::InputArchive archive;
            archive.load_from(resolvedWeights);
            model->load(archive);
            std::cout << "[XFextractor] loaded InputArchive weights: " << resolvedWeights << std::endl;
        }
        catch(const c10::Error& e)
        {
            std::cerr << "[XFextractor] InputArchive load failed, trying Python state_dict: "
                      << e.what() << std::endl;
            auto tensors = LoadPyTorchCheckpointTensors(resolvedWeights);
            LoadXFeatPythonStateDict(model, tensors, torch::Device(torch::kCPU));
        }
        // [XFEAT_STABILITY] Force inference mode behavior for BatchNorm and
        // other training-dependent layers to avoid descriptor drift.
        model->eval();
        std::cout << "XFeat model weights loaded successfully!" << std::endl;

        // move the model to device (configurable by XFEAT_DEVICE=auto|cuda|cpu).
        // auto: try CUDA first and fallback to CPU.
        // cuda: fail-fast if CUDA cannot be used.
        // cpu : force CPU.
        std::string devicePref = "auto";
        if(const char* envDevice = std::getenv("XFEAT_DEVICE"))
            devicePref = envDevice;
        std::transform(devicePref.begin(), devicePref.end(), devicePref.begin(), ::tolower);

        int cudaDeviceIdx = 0;
        if(const char* envDevIdx = std::getenv("XFEAT_CUDA_DEVICE"))
            cudaDeviceIdx = std::max(0, std::atoi(envDevIdx));

        auto initCuda = [&]() -> bool {
            if(!torch::cuda::is_available())
            {
                std::cerr << "[XFextractor] CUDA unavailable: torch::cuda::is_available() is false." << std::endl;
                return false;
            }

            int nCudaDevices = 0;
            try
            {
                nCudaDevices = static_cast<int>(torch::cuda::device_count());
            }
            catch(const c10::Error& e)
            {
                std::cerr << "[XFextractor] CUDA device_count failed: " << e.what() << std::endl;
                return false;
            }

            if(nCudaDevices <= 0)
            {
                std::cerr << "[XFextractor] CUDA unavailable: no visible CUDA devices." << std::endl;
                return false;
            }

            if(cudaDeviceIdx >= nCudaDevices)
            {
                std::cerr << "[XFextractor] CUDA device index out of range. requested=" << cudaDeviceIdx
                          << " available=" << nCudaDevices << std::endl;
                return false;
            }

            try
            {
                torch::Device device(torch::kCUDA, cudaDeviceIdx);
                // Warm-up allocation to catch runtime/oom issues before running SLAM.
                auto warmup = torch::zeros({1}, torch::TensorOptions().device(device).dtype(torch::kFloat));
                (void)warmup;
                model->to(device);
                device_type = torch::kCUDA;
                device_ = device;
                std::cout << "Device: " << device << std::endl;
                return true;
            }
            catch(const c10::Error& e)
            {
                std::cerr << "[XFextractor] CUDA initialization failed: " << e.what() << std::endl;
                return false;
            }
        };

        bool useCuda = false;
        if(devicePref == "cpu")
        {
            useCuda = false;
        }
        else if(devicePref == "cuda")
        {
            useCuda = initCuda();
            if(!useCuda)
                throw std::runtime_error("XFEAT_DEVICE=cuda but CUDA initialization failed.");
        }
        else
        {
            // auto
            useCuda = initCuda();
        }

        if(!useCuda)
        {
            device_type = torch::kCPU;
            torch::Device device(device_type);
            device_ = device;
            std::cout << "Device: " << device << std::endl;
            model->to(device);
        }

        // load the interpolators
        bilinear = std::make_shared<InterpolateSparse2d>("bilinear");     
        nearest  = std::make_shared<InterpolateSparse2d>("nearest"); 
    }

    std::string XFextractor::getModelWeightsPath(std::string weights)
    {
        const std::filesystem::path requested(weights);
        if(requested.is_absolute())
            return static_cast<std::string>(requested);

        std::filesystem::path current_file = __FILE__;
        std::filesystem::path parent_dir = current_file.parent_path();
        std::filesystem::path full_path = parent_dir / ".." / weights;
        full_path = std::filesystem::absolute(full_path);

        return static_cast<std::string>(full_path);   
    }

    torch::Tensor XFextractor::parseInput(cv::Mat &img, const torch::Device& device)
    {   
        cv::Mat continuousImg = img;
        if(!continuousImg.isContinuous())
            continuousImg = img.clone();

        // if the image is grayscale
        if (continuousImg.channels() == 1)
        {
            torch::Tensor tensor = torch::from_blob(continuousImg.data,
                                                    {1, continuousImg.rows, continuousImg.cols, 1},
                                                    torch::kByte);
            tensor = tensor.to(torch::TensorOptions().device(device).dtype(torch::kByte), true, false);
            tensor = tensor.permute({0, 3, 1, 2}).contiguous().to(torch::kFloat).div_(255.0);
            return tensor;
        }

        // if image is in RGB format
        if (continuousImg.channels() == 3) {
            torch::Tensor tensor = torch::from_blob(continuousImg.data,
                                                    {1, continuousImg.rows, continuousImg.cols, 3},
                                                    torch::kByte);
            tensor = tensor.to(torch::TensorOptions().device(device).dtype(torch::kByte), true, false);
            tensor = tensor.permute({0, 3, 1, 2}).contiguous().to(torch::kFloat).div_(255.0);
            return tensor;
        }

        // If the image has an unsupported number of channels, throw an error
        throw std::invalid_argument("Unsupported number of channels in the input image.");  
    }

    std::tuple<torch::Tensor, double, double> XFextractor::preprocessTensor(torch::Tensor& x)
    {
        // ensure the tensor has the correct type
        x = x.to(torch::kFloat);

        // calculate new size divisible by 32
        int H = x.size(-2);
        int W = x.size(-1);
        int64_t _H = (H / 32) * 32;
        int64_t _W = (W / 32) * 32;

        // calculate resize ratios
        double rh = static_cast<double>(H) / _H;
        double rw = static_cast<double>(W) / _W;

        std::vector<int64_t> size_array = {_H, _W};
        x = torch::nn::functional::interpolate(x, torch::nn::functional::InterpolateFuncOptions().size(size_array)
                                                                                                 .mode(torch::kBilinear)
                                                                                                 .align_corners(false));
        return std::make_tuple(x, rh, rw);
    }

    torch::Tensor XFextractor::getKptsHeatmap(torch::Tensor& kpts, float softmax_temp)
    {   
        torch::Tensor scores = torch::nn::functional::softmax(kpts * softmax_temp, torch::nn::functional::SoftmaxFuncOptions(1));
        scores = scores.index({torch::indexing::Slice(), torch::indexing::Slice(0, 64), torch::indexing::Slice(), torch::indexing::Slice()});

        int B = scores.size(0);
        int H = scores.size(2);
        int W = scores.size(3);

        // reshape and permute the tensor to form heatmap
        torch::Tensor heatmap = scores.permute({0, 2, 3, 1}).reshape({B, H, W, 8, 8});
        heatmap = heatmap.permute({0, 1, 3, 2, 4}).reshape({B, 1, H*8, W*8});
        return heatmap;
    }

    torch::Tensor XFextractor::NMS(torch::Tensor& x, float threshold, int kernel_size)
    {   
        int B = x.size(0);
        int pad = kernel_size / 2;

        auto local_max = torch::nn::functional::max_pool2d(x, torch::nn::functional::MaxPool2dFuncOptions(kernel_size).stride(1)
                                                                                                                      .padding(pad));
        auto pos = (x == local_max) & (x > threshold);
        std::vector<torch::Tensor> pos_batched;
        for (int b = 0; b < pos.size(0); ++b) 
        {
            auto k = pos[b].nonzero();
            k = k.index({torch::indexing::Ellipsis, torch::indexing::Slice(1, torch::indexing::None)}).flip(-1);
            pos_batched.push_back(k);
        }

        int pad_val = 0;
        for (const auto& p : pos_batched) {
            pad_val = std::max(pad_val, static_cast<int>(p.size(0)));
        }
        
        torch::Tensor pos_tensor = torch::zeros({B, pad_val, 2}, torch::TensorOptions().dtype(torch::kLong).device(x.device()));
        for (int b = 0; b < B; ++b) {
            if (pos_batched[b].size(0) > 0) {
                pos_tensor[b].narrow(0, 0, pos_batched[b].size(0)) = pos_batched[b];
            }
        }

        return pos_tensor;
    }

    std::tuple<torch::Tensor, torch::Tensor> XFextractor::NMSFixedTopK(torch::Tensor& x, torch::Tensor& rankingScores, int topk, float threshold, int kernel_size)
    {
        const int B = x.size(0);
        const int H = x.size(2);
        const int W = x.size(3);
        const int pad = kernel_size / 2;

        if(topk <= 0 || H <= 0 || W <= 0)
        {
            torch::Tensor emptyScores = torch::empty({B, 0}, x.options());
            torch::Tensor emptyKpts = torch::empty({B, 0, 2}, torch::TensorOptions().dtype(torch::kLong).device(x.device()));
            return std::make_tuple(emptyKpts, emptyScores);
        }

        auto local_max = torch::nn::functional::max_pool2d(
            x,
            torch::nn::functional::MaxPool2dFuncOptions(kernel_size).stride(1).padding(pad));
        auto keep = (x == local_max) & (x > threshold);
        auto low = torch::full_like(x, -1.0f);
        auto nms_scores = torch::where(keep, rankingScores, low);
        auto flat = nms_scores.flatten(2).squeeze(1);

        const int candidateK = std::min(topk, static_cast<int>(flat.size(1)));
        auto topkResult = flat.topk(candidateK, -1, true, true);
        torch::Tensor scores = std::get<0>(topkResult);
        torch::Tensor indices = std::get<1>(topkResult);

        torch::Tensor xs = indices.remainder(W);
        torch::Tensor ys = torch::floor_divide(indices, W);
        torch::Tensor mkpts = torch::stack({xs, ys}, -1);
        return std::make_tuple(mkpts, scores);
    }

    // [MultiScale-XFeat] True pyramid construction: each level scales from original image.
    void XFextractor::ComputePyramid(cv::Mat image)
    {
        for (int level = 0; level < nlevels; ++level)
        {
            const float scale = mvInvScaleFactor[level];
            const Size sz(cvRound(static_cast<float>(image.cols) * scale),
                          cvRound(static_cast<float>(image.rows) * scale));

            if(level == 0)
            {
                mvImagePyramid[level] = image.clone();
            }
            else
            {
                resize(image, mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);
            }
        }
    }

    // [MultiScale-XFeat] Per-level XFeat inference and semantic keypoint/descriptor generation.
    void XFextractor::ComputeKeyPointsMultiScale(std::vector<std::vector<cv::KeyPoint>>& allKeypoints, cv::Mat& desc)
    {
        c10::InferenceMode inferenceGuard(true);

        allKeypoints.clear();
        allKeypoints.resize(nlevels);

        std::vector<cv::Mat> vDescPerLevel;
        torch::Device device(device_);
        const bool profile = IsXFeatProfileEnabled();
        double tParseH2D = 0.0;
        double tPreprocess = 0.0;
        double tForward = 0.0;
        double tNmsTopk = 0.0;
        double tSamplePackD2H = 0.0;
        double tCpuSelect = 0.0;
        int profileLevels = 0;
        int profileCandidates = 0;
        int profileValidCandidates = 0;

        auto now = []() {
            return std::chrono::steady_clock::now();
        };

        auto syncIfProfile = [&]() {
            if(profile && device_type == torch::kCUDA)
                torch::cuda::synchronize();
        };

        auto markStage = [&](std::chrono::steady_clock::time_point& stageStart, double& bucket) {
            if(!profile)
                return;
            syncIfProfile();
            const auto stageEnd = now();
            bucket += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(stageEnd - stageStart).count();
            stageStart = stageEnd;
        };

        const float maxX = static_cast<float>(mvImagePyramid[0].cols - 1);
        const float maxY = static_cast<float>(mvImagePyramid[0].rows - 1);

        for (int level = 0; level < nlevels; ++level)
        {
            if(mvImagePyramid[level].empty())
                continue;

            auto stageStart = now();
            cv::Mat levelImage = mvImagePyramid[level];
            torch::Tensor x = parseInput(levelImage, device);
            markStage(stageStart, tParseH2D);

            double rh = 1.0;
            double rw = 1.0;
            std::tie(x, rh, rw) = preprocessTensor(x);
            markStage(stageStart, tPreprocess);

            const int64_t H = x.size(2);
            const int64_t W = x.size(3);

            torch::Tensor M1, K1, H1;
            std::tie(M1, K1, H1) = model->forward(x);
            M1 = torch::nn::functional::normalize(M1, torch::nn::functional::NormalizeFuncOptions().dim(1));
            markStage(stageStart, tForward);

            torch::Tensor K1h = getKptsHeatmap(K1);
            const int levelFeatureBudget = mnFeaturesPerLevel[level];
            if(levelFeatureBudget <= 0)
                continue;

            const int nPixels = static_cast<int>(std::max<int64_t>(1, H * W));
            const int requestedCandidates = std::max(levelFeatureBudget * GetXFeatFixedNMSCandidateFactor(),
                                                     GetXFeatFixedNMSCandidateMin());
            const int fixedCandidateK = std::min(nPixels, std::max(levelFeatureBudget, requestedCandidates));

            std::vector<int64_t> fullSize = {H, W};
            torch::Tensor H1Full = torch::nn::functional::interpolate(
                H1,
                torch::nn::functional::InterpolateFuncOptions().size(fullSize)
                                                               .mode(torch::kBilinear)
                                                               .align_corners(false));
            torch::Tensor rankingScores = K1h * H1Full;

            torch::Tensor mkpts, nmsScores;
            std::tie(mkpts, nmsScores) = NMSFixedTopK(K1h, rankingScores, fixedCandidateK, 0.05f, 5);
            if(mkpts.size(1) == 0)
                continue;
            markStage(stageStart, tNmsTopk);

            torch::Tensor scores = (nearest->forward(K1h, mkpts, H, W) * bilinear->forward(H1, mkpts, H, W)).squeeze(-1);
            scores.masked_fill_(nmsScores <= 0, -1);

            const int topk = std::min(levelFeatureBudget, static_cast<int>(scores.size(1)));
            if(topk <= 0)
                continue;

            auto finalTopk = scores.topk(topk, -1, true, true);
            scores = std::get<0>(finalTopk);
            torch::Tensor idxs = std::get<1>(finalTopk);
            torch::Tensor gatherIdx = idxs.unsqueeze(-1).expand({-1, -1, 2});
            mkpts = mkpts.gather(1, gatherIdx);

            torch::Tensor feats = bilinear->forward(M1, mkpts, H, W);
            feats = torch::nn::functional::normalize(feats, torch::nn::functional::NormalizeFuncOptions().dim(-1));

            torch::Tensor packed = torch::cat({mkpts[0].to(torch::kFloat), scores[0].unsqueeze(1), feats[0]}, 1);
            // Fixed-size transfer: invalid candidates keep score <= 0 and are filtered on CPU.
            torch::Tensor packedCpu = packed.to(torch::kCPU).contiguous();
            markStage(stageStart, tSamplePackD2H);

            const int packedRows = static_cast<int>(packedCpu.size(0));
            const float* packedPtr = packedCpu.data_ptr<float>();
            std::vector<float> validKeypoints;
            std::vector<float> validScores;
            std::vector<int> validPackedRows;
            validKeypoints.reserve(static_cast<size_t>(packedRows) * 2);
            validScores.reserve(static_cast<size_t>(packedRows));
            validPackedRows.reserve(static_cast<size_t>(packedRows));

            for(int i = 0; i < packedRows; ++i)
            {
                const float* row = packedPtr + i * 67;
                const float score = row[2];
                if(score <= 0.0f)
                    continue;

                validKeypoints.push_back(row[0]);
                validKeypoints.push_back(row[1]);
                validScores.push_back(score);
                validPackedRows.push_back(i);
            }

            const int valid_cnt = static_cast<int>(validScores.size());
            if(valid_cnt == 0)
            {
                markStage(stageStart, tCpuSelect);
                continue;
            }

            const float* kptPtr = validKeypoints.data();      // [valid_cnt, 2]
            const float* scorePtr = validScores.data();       // [valid_cnt]

            //调试: 每层做空间均匀化重排，减少高纹理区域“扎堆”导致的几何退化。
            const std::vector<int> selectedIdx = SelectUniformTopKIndices(kptPtr,
                                                                           scorePtr,
                                                                           valid_cnt,
                                                                           std::min(mnFeaturesPerLevel[level], valid_cnt),
                                                                           levelImage.cols,
                                                                           levelImage.rows);
            if(selectedIdx.empty())
                continue;

            const int selected_cnt = static_cast<int>(selectedIdx.size());
            const float levelScale = mvScaleFactor[level];
            const float keypointSize = PATCH_SIZE * levelScale;

            std::vector<cv::KeyPoint>& levelKpts = allKeypoints[level];
            levelKpts.reserve(selected_cnt);

            cv::Mat levelDesc(selected_cnt, 64, CV_32F);
            for (int i = 0; i < selected_cnt; ++i)
            {
                const int srcIdx = selectedIdx[i];
                const float xScaled = kptPtr[srcIdx * 2] * static_cast<float>(rw);
                const float yScaled = kptPtr[srcIdx * 2 + 1] * static_cast<float>(rh);
                const float score = scorePtr[srcIdx];

                float xOriginal = std::max(0.0f, std::min(xScaled * levelScale, maxX));
                float yOriginal = std::max(0.0f, std::min(yScaled * levelScale, maxY));

                cv::KeyPoint kp(xOriginal, yOriginal, keypointSize, -1.0f, score, level);
                levelKpts.push_back(kp);

                const int packedRow = validPackedRows[srcIdx];
                std::memcpy(levelDesc.ptr<float>(i), packedPtr + packedRow * 67 + 3, 64 * sizeof(float));
            }

            vDescPerLevel.push_back(levelDesc);
            markStage(stageStart, tCpuSelect);
            if(profile)
            {
                ++profileLevels;
                profileCandidates += packedRows;
                profileValidCandidates += valid_cnt;
            }
        }

        if(vDescPerLevel.empty())
        {
            desc.release();
        }
        else
        {
            cv::vconcat(vDescPerLevel, desc);
        }

        if(profile)
        {
            std::cout << std::fixed << std::setprecision(3)
                      << "[XFeatProfile] levels=" << profileLevels
                      << " candidates=" << profileCandidates
                      << " valid=" << profileValidCandidates
                      << " out_desc_rows=" << desc.rows
                      << " parse_h2d_ms=" << tParseH2D
                      << " preprocess_ms=" << tPreprocess
                      << " forward_ms=" << tForward
                      << " nms_topk_ms=" << tNmsTopk
                      << " sample_pack_d2h_ms=" << tSamplePackD2H
                      << " cpu_select_ms=" << tCpuSelect
                      << std::endl;
        }
    }

    int XFextractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                                  OutputArray _descriptors, std::vector<int> &vLappingArea)
    {
        if(_image.empty())
            return -1;

        Mat image = _image.getMat();
        assert(image.type() == CV_8UC1 || image.type() == CV_8UC3);

        // [MultiScale-XFeat] Stage 1: core multi-scale extraction (without stereo overlap reordering).
        ComputePyramid(image);

        std::vector<std::vector<cv::KeyPoint>> allKeypoints;
        cv::Mat multiScaleDesc;
        ComputeKeyPointsMultiScale(allKeypoints, multiScaleDesc);

        int nkeypoints = 0;
        for (int level = 0; level < nlevels; ++level)
            nkeypoints += static_cast<int>(allKeypoints[level].size());

        if(nkeypoints == 0)
        {
            _keypoints.clear();
            _descriptors.release();
            return 0;
        }

        _keypoints = std::vector<cv::KeyPoint>(nkeypoints);
        cv::Mat desc_mat(nkeypoints, 64, CV_32F);
        CV_Assert(multiScaleDesc.rows == nkeypoints);

        // [MultiScale-XFeat] Stage 2: keep legacy stereo overlap packaging, after octave/size semantics are finalized.
        int monoIndex = 0;
        int stereoIndex = nkeypoints - 1;
        int rowOffset = 0;
        const bool hasOverlapRange = vLappingArea.size() >= 2;

        for (int level = 0; level < nlevels; ++level)
        {
            const std::vector<cv::KeyPoint>& levelKpts = allKeypoints[level];
            const int nlevel = static_cast<int>(levelKpts.size());
            for (int i = 0; i < nlevel; ++i)
            {
                const cv::KeyPoint& kp = levelKpts[i];
                const bool isStereoOverlap = hasOverlapRange && kp.pt.x >= vLappingArea[0] && kp.pt.x <= vLappingArea[1];
                const int srcRow = rowOffset + i;
                CV_Assert(srcRow < multiScaleDesc.rows);

                if(isStereoOverlap)
                {
                    _keypoints.at(stereoIndex) = kp;
                    multiScaleDesc.row(srcRow).copyTo(desc_mat.row(stereoIndex));
                    --stereoIndex;
                }
                else
                {
                    _keypoints.at(monoIndex) = kp;
                    multiScaleDesc.row(srcRow).copyTo(desc_mat.row(monoIndex));
                    ++monoIndex;
                }
            }
            rowOffset += nlevel;
        }

        desc_mat.copyTo(_descriptors);
        return monoIndex;
    }
} //namespace ORB_SLAM
