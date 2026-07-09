# XFeat SLAM 深度特征适配记录

> 状态说明（2026-07-09）：本文中的 `XFEAT_MOTION_HEALTH_GUARD` 和
> `XFEAT_KF_INSERT_GUARD` 章节保留为历史实验记录；对应环境变量和行为代码
> 已删除。`projection_first` 和 LocalMap found-ratio gate 也已从主线清理。
> 旧 `XFEAT_LOCALMAP_CORE_FIRST` 章节只作为历史实验记录；当前 LocalMap 主线及后续计划以
> `docs/airslam_xfeat_slam_adaptation_direction.md` 的 2026-07-09 记录为准。

本文档记录近期围绕 XFeat + LightGlue 前端适配 ORB-SLAM3 后端的分析结论和代码改动。目标是避免后续继续在已经验证过的方向上反复调参，也方便修改 `Tracking`、`LoopClosing`、`LocalMapping` 时先对齐当前判断。

## 当前核心判断

### LightGlue 的角色边界

- LightGlue 本身只做 2D-2D matching。
- 在当前项目中，LG 可以参与 motion tracking、reference KF tracking、relocalization、LocalMapping triangulation、loop candidate initial matching。
- LocalMap projection 本身仍然是 3D MapPoint 投影到当前帧后做描述子/窗口匹配，不应该简单替换成 LG。
- 当前实验不建议启用 `projection_first` 作为默认策略；已有 MH04 结果显示它会明显伤精度。

### 金字塔层数与 octave

- `nLevels=1` 和 `keypoint.octave=0` 是当前 XFeat 前端适配 ORB-SLAM3 后端的重要边界。
- 之前很多局部 gate、scale、sigma、mapping profile 实验收益不稳定，一个重要原因是当时仍混入了 ORB 金字塔尺度语义。
- 当前不建议重新引入 ORB-style 多层尺度语义，除非同时重做完整的尺度、sigma、projection radius 和 optimizer 权重模型。

### 关键帧频繁插入的直接原因

当前日志更支持这个判断：

```text
频繁插 KF 的直接触发主要来自 stereo/RGBD close point birth pressure，
不是 LocalMap projection 整体有效支撑崩溃。
```

ORB-SLAM3 stereo/RGBD 插帧核心条件之一是：

```cpp
bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70)
```

XFeat 特征更密，带深度的 close/mid keypoint 更多，因此 `nNonTrackedClose` 更容易变高。如果这些点没有被稳定复用成当前帧 inlier MapPoint，`nTrackedClose` 又压不住，就会频繁触发插 KF，让 `CreateNewKeyFrame()` 创建新的 close MapPoint。

这不等价于“观测点多就一定应该插 KF”。更准确的描述是：

```text
当前有很多近距离、有深度的观测点，但其中相当一部分没有成为稳定可复用的 MapPoint，
于是 ORB-SLAM3 的 close-point 插帧规则被深度特征放大。
```

### LocalMap projection 约束为什么脏

LocalMap projection 的问题不是 LG motion 匹配脏，而是局部地图候选池被深度特征放大：

```text
XFeat 点多
-> 创建更多 MapPoint
-> 形成更密的共视关系
-> UpdateLocalKeyFrames() 拉入更多 local KFs
-> UpdateLocalPoints() 收集更多 local MapPoints
-> SearchLocalPoints() 的 projection 候选池变大
-> 部分 close/mid MapPoint 虽然能通过视锥和投影窗口检查，但当前帧关联不稳定
-> local_proj 约束进入 PoseOptimization 后 outlier ratio 明显高于 motion_lg
```

日志里反复看到：

- `motion_lg` outlier ratio 很低，约 `1%` 量级。
- `local_proj` outlier ratio 明显更高，MH04 里约 `24%-26%`。
- 因此当前主要脏源是 LocalMap projection 复用路径，不是 2D-2D motion LG 本身。

这说明后续不应该简单减少所有点或禁用 close 建图，而是要区分：

- 哪些 close 点值得建图。
- 哪些 close-only KF 插入是冗余的。
- 哪些 LocalMap MapPoint 在当前帧属于高风险 projection 候选。

## 已实施代码改动

### Motion Health + Source-Aware Guard

文件：

- `src/Tracking.cc`

新增开关：

- `XFEAT_MOTION_HEALTH_GUARD=1`
- `XFEAT_MOTION_WEAK_MIN_INLIERS`，默认 `80`
- `XFEAT_MOTION_BAD_MIN_INLIERS`，默认 `60`
- `XFEAT_MOTION_WEAK_MAX_OUTLIER`，默认 `0.30`
- `XFEAT_MOTION_BAD_MAX_OUTLIER`，默认 `0.40`
- `XFEAT_MOTION_GUARD_MIN_OBS`，默认 `3`
- `XFEAT_MOTION_GUARD_MIN_FOUND_RATIO`，默认 `0.35`

核心逻辑：

- 在 `TrackWithMotionModel()` 后记录 motion pose health。
- 根据 pose optimization 后的 inlier 数和 outlier ratio 判断 weak/bad。
- 在 `SearchLocalPoints()` 的 LocalMap projection admission 前加入 source-aware guard。
- motion 正常时不改变现有行为。
- motion weak/bad 时，跳过高风险 MapPoint，但不删除 MapPoint。

高风险 MapPoint 当前主要包括：

- `Observations() < 3`
- `GetFoundRatio() < 0.35`
- provenance/source unknown
- Temporal/Unknown provenance
- local projection 历史 outlier 风险较高

新增日志：

- `[TrackMotion][Health]`
- `[TrackLocalMap][MotionGuard]`

当前结论：

- 这个 guard 是局部保护，不是主要提精度来源。
- MH04 中 motion LG 大多数时候已经很干净，所以只在 weak motion 段可能有价值。
- 不建议默认和其他新实验叠加；应作为单独 A/B 开关使用。

### Loop LG 第一阶段：候选验证前端

文件：

- `include/LoopClosing.h`
- `src/LoopClosing.cc`
- `include/XFeatLighterGlueMatcher.h`
- `src/XFeatLighterGlueMatcher.cc`

新增开关：

- `XFEAT_USE_LIGHTGLUE_LOOP=1`
- `XFEAT_LOOP_CANDIDATES`
- `XFEAT_LIGHTGLUE_LOOP_CONF`

核心逻辑：

- 保留 ORB-SLAM3 的 `ORBvoc + KeyFrameDatabase::DetectNBestCandidates()` 候选召回。
- 只替换 loop candidate 几何验证前的初始匹配。
- 原始路径使用 `ORBmatcher::SearchByBoW()`。
- XFeat + LG loop 路径使用新增 KF-KF LightGlue 接口：

```cpp
int SearchByLightGlue(KeyFrame* pKF1,
                      KeyFrame* pKF2,
                      std::vector<MapPoint*>& vpMatches12,
                      float minConf,
                      std::vector<float>* vMatchScores = nullptr);
```

匹配语义：

- `vpMatches12.size() == pKF1->N`
- LG 输出 `(idx1, idx2)` 后，取 `pKF2->GetMapPoint(idx2)` 赋给 `vpMatches12[idx1]`
- 只保留非空、非 bad MapPoint
- 做 one-to-one 去重

后续仍然使用 ORB-SLAM3 原始 loop 后端：

- `Sim3Solver`
- `SearchByProjection`
- `OptimizeSim3`
- 共视 KF 验证
- `CorrectLoop()`
- GBA

新增日志：

- `[LoopLG]`
- `[LoopLG][Accept]`

当前结论：

- direct-only KF-KF LG 匹配太薄，多数 candidate `assigned` 不足以稳定进入 Sim3。
- 少数 direct match 达标的 candidate 也可能 `sim3_converged=0`。
- 问题不是 LG 完全无效，而是 candidate 本体一个 KF 的 MapPoint 支撑不足。

### Loop LG 第二阶段：Capped Covisible Expansion

文件：

- `src/LoopClosing.cc`

新增开关：

- `XFEAT_LOOP_LG_EXPAND=1`
- `XFEAT_LOOP_LG_EXPAND_COVISIBLES`，默认 `3`
- `XFEAT_LOOP_LG_EXPAND_MIN_DIRECT`，默认 `12`
- `XFEAT_LOOP_LG_MAX_CALLS_PER_KF`，用于限制每个 current KF 的 loop LG 总调用预算

核心逻辑：

- 每个 candidate 先跑 direct current KF <-> candidate KF。
- direct assigned 达到阈值后，再最多匹配 top-N covisible KFs。
- 不回到每个 candidate 11 个窗口全部跑 LG 的高负载版本。
- 保留 ORB-SLAM3 原始几何验证门槛，不降低 `nBoWMatches / nBoWInliers / nSim3Inliers / nProjMatches / nProjOptMatches`。
- 多窗口聚合时使用 `vMatchScores` 做 score-aware one-to-one 合并，避免不同 covisible KF 的 MapPoint 无序覆盖同一个 current keypoint。

增强日志：

- `candidate_rank`
- `direct_raw`
- `direct_assigned`
- `expanded`
- `expanded_windows`
- `expansion_reason`
- `lg_calls_budget_used`
- `unique_initial`
- `sim3_converged`
- `proj_matches`
- `opt_matches`
- `proj_opt_matches`
- `cov_valid`
- `accepted_candidate`

当前结论：

- Loop LG 已经能在 MH04 中触发 loop，并且 loop 后 RMSE 能落到约 `0.15-0.19` 区间。
- loop 对 MH04 轨迹精度影响非常大。
- GPU 占用会明显升高，尤其启用 loop LG expansion 后。
- 低负载配置如 `XFEAT_LOOP_CANDIDATES=6`、`XFEAT_LOOP_LG_EXPAND_COVISIBLES=1` 是当前更值得验证的方向。
- 暂不训练 XFeat 专用 vocabulary；当前优先确认 ORBvoc 是否能稳定召回足够候选，以及 LG 几何验证能否低成本接受正确 loop。

### XFeat KF Insert Guard

文件：

- `src/Tracking.cc`

新增开关：

- `XFEAT_KF_INSERT_GUARD=1`

核心逻辑：

- 只在 XFeat stereo/RGBD 且 guard eligible 时生效。
- 只抑制 close-only 的插帧。
- 如果原始插帧条件在去掉 close pressure 后仍成立，则不抑制。
- 如果达到 max-frame、weak support、stable support 不足、novel close area 明显，则不抑制。
- 不改 `nFeatures`、`nLevels`、LocalMapping 创建逻辑，也不删除 MapPoint。

新增或扩展日志：

- `[Tracking][KFDecision]`
- `kf_guard_enabled`
- `kf_guard_eligible`
- `need_without_close`
- `close_only_insertion`
- `close_guard_suppressed`
- `close_guard_reason`
- `c1c_no_close`
- `c2_no_close`

MH04 一次 guard 运行的关键统计：

- `created_kf_lines=702`
- `final_map_kfs=612`
- 对照低负载 loop run 约 `805` created KFs、`698` final KFs
- `close_only=131`
- `suppressed=115`
- `support_only=6`
- `both=4`
- `loop_accept=2`
- 无 tracking failure

解释：

- guard 确实减少了 KF 数和 local map 规模。
- 但 local projection 没有变干净，`local_proj` outlier ratio 反而略高。
- 说明有些被抑制的 close-only KF 其实在提供有用近场锚点。
- 因此当前 guard 不应作为默认最佳配置，只能作为 A/B 入口。

后续如果继续改 guard，应更保守：

- 只抑制高支撑、短间隔、纯 close pressure 的插帧。
- 如果 `c2_no_close=1` 或 reference ratio 偏低，则允许插帧。
- 如果距离上一个 KF 已经隔了多帧，则允许插帧。
- 可以考虑只在 LocalMapping backlog/queue pressure 时抑制，而不是 mapper idle 时也抑制。

## 已验证或暂不推荐的方向

### 不建议直接减少全部特征点

原因：

- easy 场景当前精度已经很好，说明深度特征更多观测是有正向价值的。
- 问题不是点多本身，而是 close/mid 观测到稳定 MapPoint 的转化和 LocalMap projection 复用质量不稳定。

### 不建议全局减少关键帧

原因：

- 关键帧多确实会放大地图和 local map，但 close-only KF 中有一部分提供必要近场约束。
- 粗暴 guard 会减少 KF 数，却可能让 local projection 更依赖旧点，导致 outlier/误差不降反升。

### 不建议继续单点调 LocalMap projection gate

原因：

- 已有多轮 gate/cooldown/weight/history 实验显示，局部 source outlier ratio 下降不一定带来 RMSE 下降。
- LocalMap projection 脏是结构问题：local KFs、MapPoints、close birth、covis graph 和当前帧复用共同作用。

### 暂不训练 XFeat vocabulary

原因：

- 当前 loop 已经能通过 ORBvoc 召回并接受正确候选，loop 后 RMSE 明显改善。
- 目前更直接的问题是 loop 是否稳定、GPU 负载是否可控、候选验证是否低成本。
- 如果后续在 MH03/MH05/V1/V2 出现明显“有回环但 ORBvoc 召回不到候选”的证据，再规划 XFeat vocabulary / DBoW3 / FBoW。

## 后续优先修改方向

### 1. 更保守的 KF insert guard

目标不是减少所有 KF，而是减少明显冗余的 close-only KF。

建议条件：

- `close_only_insertion=1`
- `need_without_close=0`
- `c2_no_close=0`
- `mnMatchesInliers / nRefMatches >= 0.75`
- `frames_since_last_kf <= 2` 或 `<= 3`
- 当前没有 novel close area
- 可选：LocalMapping queue/backlog 有压力时才抑制

### 2. LocalMap projection 高风险 MapPoint admission

目标不是删除 MapPoint，而是当前帧弱 prior 或局部地图过大时，限制高风险点进入 projection matching。

候选信号：

- `Observations()`
- `GetFoundRatio()`
- creation source
- match source history
- local projection outlier history
- depth bin，尤其 close/mid
- current frame frustum/projection residual context

### 3. Close point birth 质量分层

目标不是减少 close 点，而是避免未验证 close/mid 观测过早进入长期地图、共视图和 pose edge。

更合理的长期方向：

- tracklet-before-MapPoint
- delayed MapPoint creation
- 先跨帧验证，再进入 covis graph
- 未验证点不参与当前帧 pose optimization

这比把 active probation MapPoint 放进 local projection pool 更合理，因为之前 active probation verification pool 已经证明会伤 RMSE。

### 4. Loop 低负载稳定性验证

优先验证：

```bash
XFEAT_USE_LIGHTGLUE_LOOP=1
XFEAT_LOOP_CANDIDATES=6
XFEAT_LOOP_LG_EXPAND=1
XFEAT_LOOP_LG_EXPAND_COVISIBLES=1
XFEAT_LOOP_LG_MAX_CALLS_PER_KF=12
```

观察：

- 是否仍有 `[LoopLG][Accept]`
- 是否出现 `*Loop detected`
- 是否触发 `mpLoopCloser is running GBA`
- RMSE median 是否稳定
- GPU 占用是否下降

## 推荐 A/B 运行边界

基线不要混太多变量：

```bash
XFEAT_DIAG_FEATURE_DISTRIBUTION=1 \
XFEAT_DIAG_INTERVAL=30 \
XFEAT_LIGHTGLUE_WEIGHT=./weights/xfeat_lighterglue_matcher_cpp.pt \
XFEAT_USE_LIGHTGLUE_MOTION=1 \
XFEAT_USE_LIGHTGLUE_TRIANG=1 \
XFEAT_USE_LIGHTGLUE_REF=1 \
XFEAT_USE_LIGHTGLUE_RELOC=1 \
XFEAT_USE_LIGHTGLUE_LOOP=1 \
XFEAT_LOOP_CANDIDATES=6 \
XFEAT_LOOP_LG_EXPAND=1 \
XFEAT_LOOP_LG_EXPAND_COVISIBLES=1 \
XFEAT_LOOP_LG_MAX_CALLS_PER_KF=12 \
./examples/Stereo/stereo_euroc \
  Vocabulary/ORBvoc.txt \
  examples/Stereo/EuRoC.yaml \
  "/mnt/d/data_set/EuRoc Wav Dataset/machine_hall/MH_04_difficult/mav0" \
  examples/Stereo/EuRoC_TimeStamps/MH04.txt \
  EuRoC_MH_04_difficult_stereo \
  2>&1 | tee experiment_logs/Stereo/EuRoC/MH_04_difficult/run_EuRoC_MH_04_difficult_stereo_loop_lg_lowload_next.log
```

测试 KF guard 时，只额外加一个变量：

```bash
XFEAT_KF_INSERT_GUARD=1
```

不要同时叠加：

- `XFEAT_MOTION_HEALTH_GUARD=1`
- 新的 projection gate
- 新的 LocalMapping birth gate
- `projection_first`

否则日志很难归因。

## 2026-06-24：LocalMap Core-First 阶段结果

### 本阶段代码改动

本阶段核心改动是 `XFEAT_LOCALMAP_CORE_FIRST=1`，只改变 `Tracking::SearchLocalPoints()` 里 LocalMap projection 候选进入匹配和 PoseOptimization 的顺序，不改变 keyframe 插入、MapPoint 创建、LoopClosing 后端或 LocalMapping 三角化逻辑。

核心逻辑：

- 将 LocalMap projection 候选分为 `core` 和 `expansion`。
- `core` 优先进入 projection matching。
- 如果已有 motion/ref prior 足够、core matches 足够、total support 足够，则跳过 `expansion`。
- 如果 prior/core/total support 不足，则允许 `expansion` 作为 fallback。
- 只是不让高风险 expansion 点参与当前帧优化，不删除 MapPoint，也不改变地图结构。

新增/使用的主要环境变量：

```bash
XFEAT_LOCALMAP_CORE_FIRST=1
XFEAT_LOCALMAP_CORE_FIRST_MIN_PRIOR=120
XFEAT_LOCALMAP_CORE_FIRST_MIN_CORE_MATCHES=40
XFEAT_LOCALMAP_CORE_FIRST_MIN_TOTAL_MATCHES=160
```

新增诊断字段在 `[SearchLocalPoints]` 中输出：

```text
localmap_core_first
core_first_prior
core_first_core_candidates
core_first_expansion_candidates
core_first_expansion_admitted
core_first_expansion_suppressed
core_first_core_matches
core_first_expansion_matches
core_first_decision
```

### MH04 三次 core-first 结果

同一配置下 MH04 跑三次，结果如下：

```text
core_first_0: RMSE 0.101133, final 741 KFs, 2 loops
core_first_1: RMSE 0.121802, final 720 KFs, 2 loops
core_first_2: RMSE 0.150705, final 737 KFs, 2 loops
ORB-SLAM3:    RMSE 0.095483, final 263 KFs
```

三次结果说明：

- `core_first` 方向有效，最好一次已经接近 ORB-SLAM3 的 MH04 结果。
- 当前配置仍不稳定，median 更接近 `0.12`，不能把 `0.10` 当作稳定水平。
- 三次都检测到 2 次 loop，因此差异不是简单的 no-loop / loop 二值问题。
- final KFs 仍在 `720-741`，明显高于 ORB-SLAM3 的 `263`，说明 keyframe/map 膨胀问题仍然存在。

### Loop 质量字段解释

`candidate_rank` 表示当前 KF 从 `KeyFrameDatabase::DetectNBestCandidates()` 取出的 loop candidate 排名。数字越小，说明词袋候选检索越靠前：

```text
candidate_rank=0  -> 当前候选是 top-1
candidate_rank=5  -> 当前候选是第 6 个候选
```

它不是几何验证结果，而是 place recognition / BoW 候选召回阶段的排序。rank 低通常更可靠，但不是绝对，因为当前仍使用 ORB vocabulary 检索 XFeat 前端生成的关键帧。

`sim3_inliers` 表示 LightGlue 初始 2D-3D 对应关系进入 `Sim3Solver` 后，被 RANSAC/Sim3 几何一致性接受的内点数量。它衡量的是 loop correction 初始 Sim3 的几何支撑强度。

需要区分：

```text
direct_raw / direct_assigned  -> LG 匹配和 MapPoint 赋值数量
sim3_inliers                  -> Sim3 几何一致的初始内点数量
proj_opt_matches              -> Sim3 投影补匹配和 OptimizeSim3 后的支持
```

`proj_opt_matches` 很高不一定代表初始 loop 很好，因为它依赖已经估计出来的 Sim3 和后续投影补匹配。若 `sim3_inliers` 低但 `proj_opt_matches` 高，需要警惕：后续 projection refinement 可能在一个较弱初值上补进大量点。

本次三次 loop 质量对比：

```text
0.101 run first loop: candidate_rank=0, sim3_inliers=82, proj_opt_matches=170
0.121 run first loop: candidate_rank=1, sim3_inliers=20, proj_opt_matches=157
0.150 run first loop: candidate_rank=5, sim3_inliers=31, proj_opt_matches=307
```

因此当前不应只看 `proj_opt_matches`，还要看 candidate rank 和 Sim3 初始内点。

### 为什么精度变好

当前较好精度主要来自两件事叠加：

1. `nLevels=1 + octave=0` 去掉了 ORB 金字塔尺度语义和 XFeat 描述子的错配。
2. `LocalMap core-first` 减少了高风险 local projection expansion 点直接进入 PoseOptimization 的机会。

三次日志中 motion LG 仍然很干净：

```text
motion_lg outlier ratio: about 0.8%-0.9%
local_proj outlier ratio: about 25%-26%
local_proj + tracking_stereo MP: about 29%-31%
local_proj + local_mapping_triangulation MP: about 23%-24%
```

所以当前提升不是因为 motion LG 忽然变好，而是因为 LocalMap projection 的脏约束被部分挡住了。

### 为什么仍然不稳定

不稳定主要来自 `core_first` fallback 路径和 loop 前局部地图状态。

三次 `core_first_decision` 分布：

```text
0.101 run: skip_expansion=258, core_weak=16
0.121 run: skip_expansion=227, core_weak=33
0.150 run: skip_expansion=173, core_weak=82
```

坏运行里更多帧进入 `core_weak` 或 `prior_weak`，当前实现会放开 expansion 来避免跟踪失败。问题是这些 expansion 点正是 LocalMap projection 中风险更高的部分。在局部地图已经膨胀时，fallback 会一次性放进几百到一千多个 expansion candidates，坏段中 local projection outlier 可超过 50%。

典型坏段特征：

```text
local_mps 到一万级
local_kfs_voted_kept 到一百级
local_mps_lmt_0_10 / tracking_stereo_0_10 明显升高
local_proj outlier ratio 大幅高于 motion_lg
```

因此当前不稳定不是点数不足，而是 fallback 时低质量局部地图点重新进入优化。

### 已验证并回退：Bounded Quality Expansion Fallback

曾实现 `XFEAT_LOCALMAP_EXPANSION_CAP=120`，目标是在 `core_first` 的 `prior_weak/core_weak` fallback 中限制 expansion candidates 数量，并优先选择质量更高的 MapPoint。该代码已回退，原因是实验结果显示它不稳定且没有带来正收益。

MH04 三次结果：

```text
cap120_0: RMSE 0.144637, 2 loops, final 734 KFs
cap120_1: RMSE 0.211911, 1 loop,  final 737 KFs
cap120_2: RMSE 0.172817, 1 loop,  final 752 KFs, LoopClosing shutdown 未完成
```

和无 cap 的 core-first 对比：

```text
no-cap core_first: RMSE 0.101 / 0.122 / 0.151
cap120:            RMSE 0.145 / 0.212 / 0.173
```

关键日志结论：

```text
cap120_0: cap_input=5388,  quality_pass=316, low_quality_admit=3900
cap120_1: cap_input=7781,  quality_pass=308, low_quality_admit=5232
cap120_2: cap_input=11423, quality_pass=163, low_quality_admit=9483
```

真正通过质量条件的 expansion 很少，但 cap 实现为了凑满 admission 数量，仍然把大量 low-quality candidates 放进 projection matching。它不是“高质量 fallback”，而是“低质量 fallback 的限量版”。

同时，cap 砍掉了不少实际能匹配上的 expansion 支撑：

```text
no-cap 0.101 run: expansion matches = 402, active avg = 14.36
cap120_0:         expansion matches = 230, active avg = 5.23

no-cap 0.121 run: expansion matches = 561, active avg = 12.20
cap120_1:         expansion matches = 294, active avg = 5.76

no-cap 0.150 run: expansion matches = 830, active avg = 8.74
cap120_2:         expansion matches = 416, active avg = 4.52
```

因此，pre-projection 数量 cap 会同时丢掉有用支撑并保留一部分脏支撑，导致局部跟踪、共视关系和 loop 时机更不稳定。

本轮经验：

- 不建议继续做 `XFEAT_LOCALMAP_EXPANSION_CAP`、`CAP=80/160` 这类 pre-projection candidate 数量调参。
- 不建议用 `Observations / found_ratio / provenance / LocalProjPriorityBin` 的静态评分直接决定 projection 前 admission。
- 质量判断应该尽量后移到 projection matching 之后，或者 PoseOptimization 之后，因为只有那时才知道这个 MapPoint 在当前帧是否真实形成了可用约束。

### 回退后的下一步方向

当时阶段曾推荐回到无 cap 的 `XFEAT_LOCALMAP_CORE_FIRST=1` 作为 baseline。按 2026-07-09 后续结果更新：旧 `CORE_FIRST`
不再作为当前主线；已经验证过的结论是 projection 前数量 cap、post-opt delayed risk guard、core-first admission-stage shadow 都不适合作为当前主线。

优先级 1：loop 质量稳定性

- cap120 结果说明 loop 对 RMSE 仍然高度敏感。
- 继续保留 LG loop，但需要把接受条件从“`proj_opt_matches` 足够”调整为同时关注：
  - `candidate_rank`
  - `sim3_inliers`
  - `proj_opt_matches`
  - 是否触发/完成 GBA
- 不建议只扩大候选数或继续增加 covisible expansion，因为 GPU 和 LoopClosing 负载已经偏高。

优先级 2：LocalMap 结构诊断

- KF 数仍稳定在 `700+`，这是深度特征带来的系统性膨胀。
- 但三次 cap 实验说明，直接减少局部候选不等于提高精度。
- 若进入 KF/map 阶段，应复用现有日志判断：
  - 哪些 KF 插入只增加了 weak local projection 负担。
  - 哪些 close/mid MapPoint 后续长期没有贡献 inlier。
  - 哪些共视 KF 只贡献 low-quality local map payload。
- 不建议先做全局 KF 抑制或全局 MapPoint culling。

优先级 3：局部约束选择

- 保留 full expansion candidates，避免提前丢掉有用支撑。
- 不再新增专用 shadow/audit 字段。
- 如果后续要做真实行为改动，应先用现有 `SourceError / SourceDepthError / SourceMPJointError / XFeatMeasurementCalib / PoseOpt ResidualCalib` 对比好坏 run，确认稳定分离后再做小范围规则。

### Shutdown 卡住问题

`core_first_exp_cap120` 的后两次运行在 shutdown 阶段出现：

```text
mpLoopCloser is not finished
mpLoopCloser is running GBA, waiting...
```

其中 `running GBA` 只出现前几轮，之后主要是 `mpLoopCloser is not finished` 持续打印。日志说明不是单纯 GBA 太慢，而是 LoopClosing 线程没有进入 `SetFinish()`。

定位到的风险点：

```text
System::Shutdown()
  -> RequestFinish(LocalMapping)
  -> RequestFinish(LoopClosing)

LoopClosing 正在 CorrectLoop / MergeLocal
  -> RequestStop(LocalMapping)
  -> while(!mpLocalMapper->isStopped()) 等待

LocalMapping 已经收到 RequestFinish，可能直接 SetFinish()
此时 isFinished()==true，但 isStopped() 未必被 LoopClosing 观察到
LoopClosing 继续等待 isStopped()
System::Shutdown 继续等待 LoopClosing isFinished()
```

已将 `LoopClosing.cc` 中等待 LocalMapping stop 的几处条件改为：

```cpp
while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
```

这个逻辑和 `RunGlobalBundleAdjustment()` 里已有等待方式一致。它不取消 GBA，也不跳过正常 loop correction；只避免 shutdown 已经要求 LocalMapping finish 时，LoopClosing 还死等 `isStopped()`。

### 已删除：Core-first post-opt shadow audit

这个 audit 曾用于回答一个问题：`XFEAT_LOCALMAP_CORE_FIRST=1` 有收益时，local projection 中到底是 core 点更稳定，还是 expansion 点更容易污染优化。

它给出的阶段性结论是：

- `expansion` 的 local_proj outlier ratio 明显更高。
- `core` 自身也不是稳定干净，仍然有较高 outlier。
- 只压 expansion 不够，直接把 post-opt high-residual 点做成 delayed risk guard 也没有收益。

后续 delayed local-proj risk guard 在 MH04 上连续变差，因此这条专用审计链路不再保留。当前代码已经删除：

- `XFEAT_LOCALMAP_CORE_AUDIT`
- `[TrackLocalMap][CoreFirstPostOptShadow]`
- `Frame::mvLocalProjAdmissionStage`
- `SearchLocalPoints()` 中的 local projection admission-stage 标记和回填

保留的内容：

- `XFEAT_LOCALMAP_CORE_FIRST=1`，这是当时仍有正向迹象的历史行为分支；当前已被 temporal expansion-last
  和 adaptive primary-first 取代。
- `[SearchLocalPoints]` 的 core / expansion candidates、matches 和 decision 统计。
- `[TrackLocalMap][SourceError]`、`[SourceDepthError]`、`[SourceMPJointError]`、`[XFeatMeasurementCalib]`、`[PoseOpt][ResidualCalib]` 等通用诊断。

后续如果继续处理 LocalMap projection，不再恢复 admission-stage shadow，而应优先使用现有 source / depth / MapPoint provenance / residual 诊断做更小范围的验证。

### 2026-06-25：core-first / loop / audit 结果收敛

当前需要把不同代码版本分开看：

```text
no-cap core-first:
  RMSE 0.101 / 0.122 / 0.150
  三次都有 2 次 loop
  final KFs 约 720-741

bounded expansion cap120:
  RMSE 0.145 / 0.212 / 0.173
  1-2 次 loop
  已回退

core-first audit low-loop:
  RMSE 0.173580
  2 次 loop
  final KFs 730
  ORB 对照 RMSE 0.087313
```

对 `0.101 / 0.122 / 0.150` 的理解：

- `0.101` 不是纯随机，它同时伴随更好的 loop 初始几何和较少 weak fallback。
- 但三次 median 更接近 `0.12-0.15`，不能把 `0.101` 当成稳定能力。
- 三次都有 loop，所以差异不是简单的 loop/no-loop，而是 loop 质量和 loop 前局部地图状态。

loop 质量字段的当前解释：

```text
candidate_rank:
  BoW 候选排序，rank 越小表示 ORBvoc 越早召回该候选。

sim3_inliers:
  LG 初始对应进入 Sim3Solver 后的几何一致内点数。

proj_opt_matches:
  Sim3 初值之后投影补匹配和 OptimizeSim3 的最终支持。
```

好结果中的第一次 loop：

```text
candidate_rank=0
sim3_inliers=82
```

最新 audit low-loop 结果：

```text
candidate_rank=2, sim3_inliers=58
candidate_rank=5, sim3_inliers=50
```

结论：

- loop 候选排序和 Sim3 初始内点会明显影响 MH04 RMSE。
- 但当前 ORBvoc 不是完全召回不到候选，暂时不把训练 XFeat vocabulary 作为第一优先级。
- 优先处理 loop 前局部地图状态，避免 `local_proj` 脏约束把 pose 和地图状态带偏。

最新 audit 的核心发现：

```text
core local_proj outlier ratio       ~= 25.7%
expansion local_proj outlier ratio  ~= 54.5%
core_tracking_stereo outlier ratio  ~= 30.0%
core_lmt outlier ratio              ~= 24.4%
```

这说明：

- `expansion` 更脏，但 `core` 也不是稳定干净。
- 继续只压 expansion 不够。
- 下一步应该处理 core 中历史上反复 high-residual / outlier 的 MapPoint。

历史 `CoreFirstPostOptShadow` audit 给出的关键证据：

```text
core shadow_rule_candidate=101, drop_inlier=0, drop_outlier=101
expansion shadow_rule_candidate=32, drop_inlier=0, drop_outlier=32
```

曾经尝试过把 post-opt 反馈转成延迟生效的风险历史：

```text
当前帧完整参与匹配和优化
-> 记录 local_proj 的 high-residual outlier
-> 后续 weak / localmap-bloated 帧降低这些 MapPoint 的优先级
```

### 2026-06-25 结论：delayed local-proj risk guard 不保留

这条路线已经实验证明不适合作为当前主线，相关行为逻辑和日志已从代码中删除。

实验结果：

```text
core-first 历史较好结果：
  0.101133 / 0.121802 / 0.150705

开启 delayed local-proj risk guard：
  run 1 RMSE = 0.245128
  run 2 RMSE = 0.241517

关闭 delayed local-proj risk guard：
  RMSE = 0.169855
```

关键日志观察：

- `risk guard` 能识别一批 high-residual outlier，但数量太少，无法改变 LocalMap projection 的整体统计。
- 开关前后 `local_proj outlier avg` 基本都在 `0.248` 附近，没有实质净化。
- `motion_lg outlier avg` 仍然很低，约 `0.010~0.012`，说明 motion LG 不是主要脏源。
- 开启 guard 后 RMSE 连续落在 `0.24` 左右；关闭后回到 `0.17`，说明该行为至少不是正向改动。
- loop 数量不能单独解释结果：开启 guard 的一次运行有 2 次 loop，但 RMSE 仍为 `0.241517`；关闭 guard 只有 1 次 loop，但 RMSE 为 `0.169855`。

代码清理记录：

- 删除 `XFEAT_LOCALMAP_RISK_GUARD` 环境变量行为。
- 删除 `[TrackLocalMap][RiskGuard]` 诊断日志。
- 删除 `MapPoint` 中 local projection high-residual risk history 字段。
- 删除 post-opt risk outcome 记录。
- 删除 SearchLocalPoints 中 high-risk MapPoint demote / risk expansion pool 逻辑。
- 删除 `XFEAT_LOCALMAP_CORE_AUDIT` 环境变量行为。
- 删除 `[TrackLocalMap][CoreFirstPostOptShadow]` 诊断日志。
- 删除 `Frame::mvLocalProjAdmissionStage` 以及 SearchLocalPoints admission-stage 记录。

后续判断：

- 该阶段曾保留 `XFEAT_LOCALMAP_CORE_FIRST=1` 作为主线；当前主线已切换到
  `XFEAT_LOCALMAP_TEMPORAL_EXPANSION_LAST=1` + adaptive `XFEAT_LOCALMAP_PRIMARY_FIRST=1`。
- 不保留 `XFEAT_LOCALMAP_CORE_AUDIT=1`，避免代码和日志继续膨胀。
- 不继续沿 delayed risk history guard 方向调阈值。
- 下一步应继续从 loop 初始质量、LocalMap 结构性筛选、共视候选池规模与来源比例入手。

## 以后改代码前的检查清单

- 如果改 KF 插入，先看 `[Tracking][KFDecision]` 中 `close_only_insertion / need_without_close / c2_no_close / close_guard_reason`。
- 如果改 LocalMap projection，先看 `motion_lg` 和 `local_proj` 的 source outlier 差异，不要把 motion 问题误判成 local projection 问题。
- 如果改 loop，先看 `[LoopLG] unique_initial / sim3_converged / proj_opt_matches / cov_valid`，不要只看 LG raw match 数。
- 如果 RMSE 改善但 loop 状态不同，要先分开 no-loop / loop / GBA 三类结果。
- 如果地图点或 KF 数下降但 RMSE 不降，不要继续沿同一 guard 参数线调。
- 如果 easy 变差，优先怀疑全局稀疏化或过强 guard 破坏了深度特征优势。

### 2026-06-25：loop 质量不稳定的第一阶段诊断改动

当前判断：

- MH04 的 RMSE 波动已经不能只用是否触发 loop 解释。
- `candidate_rank`、`sim3_inliers`、`proj_opt_matches`、GBA 是否真正完成，都会影响最终 RMSE。
- 继续调 tracking / LocalMap 前，需要先把每次运行的 loop 状态压缩成一条稳定摘要，方便把 run 按 loop 质量分组。

本阶段只做诊断，不改变行为：

- 不改 loop candidate 数量、Sim3 阈值、projection refinement 阈值。
- 不改 loop 接受规则。
- 不改 CorrectLoop / GBA 的执行策略。
- 不改 LocalMap、KF 插入、MapPoint 创建。

新增摘要日志：

```text
[LoopLG][Summary]
```

摘要记录：

- 候选验证总量：`checked_candidates / max_candidate_rank / sim3_converged_candidates`
- LG 负载与匹配量：`total_lg_calls / total_lg_raw / total_lg_assigned`
- 接受情况：`accept_count`
- 最近一次接受 loop：`last_accept_*`
- 几何最强接受 loop：`best_accept_*`
- 实际进入 CorrectLoop 次数：`loop_correction_count`
- GBA 状态：`gba_started / gba_optimizer_finished / gba_applied / gba_aborted / gba_stop_requested / last_gba_abort_reason`
- shutdown 状态：`shutdown_saw_loop_not_finished / shutdown_saw_gba_running / shutdown_timeout`

后续使用方式：

- 同一配置至少跑 3 次，先按 `[LoopLG][Summary]` 分组，再比较 RMSE。
- 如果 `accept_count` 相同但 `best_accept_sim3_inliers` 差异大，优先分析 loop 初始几何质量。
- 如果 `gba_started > gba_applied` 或 shutdown 看到 GBA/LoopClosing 未完成，RMSE 需要单独分组，不要和完整 GBA run 直接比较。
- 如果 `checked_candidates` 很多但 `sim3_converged_candidates` 很少，再考虑候选验证结构或 XFeat 专用 place recognition。

### 2026-06-25：shutdown 不再中断已触发的 GBA

触发原因：

```text
loop_summary_2:
  RMSE = 0.281441
  accept_count = 1
  gba_started = 1
  gba_optimizer_finished = 1
  gba_applied = 0
  gba_aborted = 1
  last_gba_abort_reason = finish_requested
  shutdown_saw_gba_running = 1
```

这说明该 run 已经检测到 loop，并且 GBA optimizer 已经完成，但 shutdown 的 finish 请求发生在地图更新前，导致 GBA 结果没有应用。该结果不能和 `gba_applied=1` 的 run 直接比较。

代码调整：

- `LoopClosing::RequestFinish()` 不再对正在运行的 GBA 设置 `mbStopGBA=true`，也不再递增 `mnFullBAIdx`。
- `LoopClosing::RunGlobalBundleAdjustment()` 在 optimizer 完成后，即使 `CheckFinish()==true`，也继续执行地图更新。
- `LoopClosing::RequestFinish()` 会区分当前 GBA 状态并只打印状态；普通 finish 不再设置 `mbStopGBA=true`。
- `System::Shutdown()` 保留 GBA bounded wait 兜底：GBA 持续运行超过 10 秒则请求 stop；再等 5 秒仍未结束，则继续 shutdown 并在 summary 中标记 timeout。
- `LoopClosing::RequestShutdownGBAStop()` 只作为 shutdown 超时兜底使用，不影响正常 `RequestFinish()`。
- 保留 shutdown wait 和 `[LoopLG][Summary]`，用 `gba_applied / gba_aborted / shutdown_saw_gba_running` 判断本次结果是否可比较。

预期验证：

- 如果 shutdown 期间 GBA 正在运行，日志会打印：

```text
[LoopGBA] finish requested after optimizer; waiting for GBA map update.
[LoopGBA] finish requested after optimizer; applying completed GBA before shutdown.
```

- 如果 shutdown 时 GBA optimizer 还没完成，会打印：

```text
[LoopGBA] finish requested while GBA optimizer is running; waiting for completion.
```

- 如果 GBA 真的卡住或过慢，会打印：

```text
[Shutdown] timeout waiting for GBA map update, requesting GBA stop.
[LoopGBA] shutdown timeout: requesting GBA stop.
```

- 正常结果应该满足：

```text
gba_started=1
gba_optimizer_finished=1
gba_applied=1
gba_aborted=0
```

如果 GBA 仍然没有应用，再分析是否是新 loop/merge 主动停止了 GBA，而不是 shutdown finish 请求。

### 2026-06-26：修正 RequestFinish 与 GBA stop 语义

最新一次 `loop_selection_gba_stage` 日志显示：

```text
accept_count=2
gba_started=2
gba_optimizer_finished=2
gba_applied=1
gba_aborted=1
last_gba_abort_reason=stop_requested
```

直接原因是 `RequestFinish()` 在第二次 GBA optimizer 仍运行时设置了 `mbStopGBA=true`。optimizer 随后完成，但 `RunGlobalBundleAdjustment()` 通过 `if(!mbStopGBA)` 决定是否执行 map update，因此第二次 GBA 被跳过应用。

代码调整：

- `RequestFinish()` 不再设置 `mbStopGBA=true`，也不再增加 `gba_stop_requested`。
- `RequestFinish()` 查询 GBA 状态时使用非阻塞 `try_lock`，避免 shutdown 卡在进入 bounded wait 之前。
- 普通 shutdown 只表示不再接收新 loop，并等待正在运行的 GBA 完成。
- `RequestShutdownGBAStop()` 保持为唯一的 shutdown 超时强停入口。
- `RunGlobalBundleAdjustment()` 只在 `!mbStopGBA` 时打印并执行 “applying completed GBA”；如果已经 stop，则明确打印跳过 map update。

预期：

- 正常 shutdown 期间，已触发的 GBA 应该完成 optimizer 并应用 map update。
- 只有超过 `System::Shutdown()` 的 bounded wait 后，才允许设置 `mbStopGBA=true` 放弃 GBA。

### 2026-06-26：loop 稳定性诊断收敛阶段

当前判断：

- MH04 的 RMSE 波动主要受 loop 质量和 GBA 是否完整应用影响。
- `XFEAT_LOCALMAP_RISK_GUARD`、过强的 LocalMap 风险过滤、以及继续压 projection expansion 都没有稳定收益。
- 现阶段不继续叠加 LocalMap/KF 行为开关，先把 loop 候选选择和 GBA 阶段状态看清楚。

代码调整：

- `DetectCommonRegionsFromBoW()` 曾短期增加 `[LoopLG][Selection]`，验证当前 accepted candidate 是否就是本轮几何最强候选。
- 实验显示 accepted loop 中 `chosen == best_sim3 == best_proj`，当前主要问题不是 loop candidate scoring。
- `[LoopLG][Selection]` 及对应 shadow audit 已删除，避免调试信息继续膨胀。
- `RunGlobalBundleAdjustment()` 增加少量 `[LoopGBA][Stage]`：
  - `optimizer_start / optimizer_done`
  - `wait_local_mapping / local_mapping_stopped`
  - `map_mutex_wait / map_update_start / map_update_done`
  - `aborted_*`

这次修改的目的：

- 保留 `[LoopLG][Accept]` 和 `[LoopLG][Summary]` 判断 loop 质量与 GBA 是否完整应用。
- 如果 shutdown 卡住，直接从 `[LoopGBA][Stage]` 判断卡在 optimizer、LocalMapping stop、map mutex，还是 map update。

这次明确不做：

- 不改 `XFEAT_LOOP_CANDIDATES` 默认行为。
- 不改 loop 接受阈值。
- 不改 LocalMap projection admission。
- 不训练 XFeat vocabulary。
- 不新增长期 shadow/audit 分支。

### 2026-06-26：Loop early-miss 归因诊断

背景：

- `run_EuRoC_MH_04_difficult_stereo_gba_shutdown_fix.log` 中 `RMSE=0.190314`，但两次 loop 的 GBA 都完整应用：
  - `gba_started=2`
  - `gba_optimizer_finished=2`
  - `gba_applied=2`
  - `gba_aborted=0`
- 该结果说明 RMSE 没提上去不是 GBA shutdown 造成的。
- 对比好结果可见，关键差异更像是第一次有效 loop 的时机：
  - 较好 run 曾在 `KF638/KF650` 接受第一次 loop。
  - 本次第一次接受 loop 在 `KF713`。
- 现有 `[LoopLG][Accept]` 和 `[LoopLG][Summary]` 能说明接受了什么，但不能说明 `KF600~KF720` 之间未接受的 current KF 卡在哪个阶段。

代码调整：

- 新增默认关闭诊断：
  - `XFEAT_LOOP_NEARMISS=1`
  - `XFEAT_LOOP_NEARMISS_START_KF`
  - `XFEAT_LOOP_NEARMISS_END_KF`
- 诊断同时要求 `XFEAT_DEBUG=1`。
- `DetectCommonRegionsFromBoW()` 在指定 KF 区间内，如果当前 KF 没有接受 loop，只输出一条 `[LoopLG][NearMiss]`。
- `[LoopLG][NearMiss]` 复用已有候选验证过程中的字段，不改变任何真实行为：
  - `unique_initial`
  - `direct_assigned`
  - `sim3_converged / sim3_inliers`
  - `proj_matches`
  - `opt_matches`
  - `proj_opt_matches`
  - `cov_valid`
  - `reject_reason`
- `reject_reason` 分为：
  - `initial_below`
  - `sim3_no_converge`
  - `proj_below`
  - `opt_below`
  - `proj_opt_below`
  - `cov_below`

这次明确不做：

- 不修改 loop candidate 数量。
- 不修改 LG expansion 策略。
- 不修改 Sim3 / projection / covis consistency 阈值。
- 不修改 GBA shutdown。
- 不恢复 `[LoopLG][Selection]` shadow audit。

下一步判断标准：

- 如果 `KF600~KF700` 的 NearMiss 多数是 `initial_below`，并且正确历史区域不在候选内，才进入 XFeat 专用 vocabulary / place recognition。
- 如果 NearMiss 多数已经进入 `sim3_no_converge / proj_below / cov_below`，说明 ORBvoc 已经召回候选，下一步应考虑短窗口 loop hypothesis memory，而不是先训练词袋。

## 2026-07-09：LocalMap weak fallback 收敛结论

在 XFeat 专用词袋、LightGlue loop 和 `nLevels=1` 固定后，最近三次 MH04 运行显示：

- adaptive primary-first 已经能让 normal 段更干净，但这些帧不是主要误差来源。
- `secondary_suppressed=1` 的帧 local projection 外点率下降，但数量占比小。
- weak sampled frame 的 APE 明显高于 normal frame，且典型 weak fallback 帧存在
  `primary_support≈60`、local projection 外点率超过 `40%` 的情况。
- 继续压 expansion 或恢复 MapPoint lifecycle expansion guard 意义有限，因为 expansion 最脏但占比低。
- 旧 `risk guard`、old `CORE_FIRST`、KF policy guard 都不再作为当前主线。

曾短期实现以下行为：

```text
primary support 足够:
    suppress secondary / expansion
primary support 不足或 motion weak:
    secondary 按质量排序并限量准入
    base 仍不足时才启用 expansion
```

两次运行结果显示：

```text
run_MH04_weak_secondary_admission_1.log:
  local_proj outlier = 18.6%, secondary matches = 4198, KFs = 577
run_MH04_weak_secondary_admission_2.log:
  local_proj outlier = 19.6%, secondary matches = 4309, KFs = 572
```

相比前一版 adaptive primary-first，它略微降低了 `local_proj` 外点率，但减少了 secondary
有效支撑、增加了 KeyFrame 数量，loop 几何和 RMSE 没有改善。因此该行为已回退。

当前保留的行为仍是：

```text
primary support 足够:
    suppress secondary / expansion
primary support 不足:
    admit all secondary
    base 仍不足时才启用 expansion
```

后续不再继续调 secondary 排序分数、budget 或 source/risk 阈值。如果继续处理 weak 段，应转向
二阶段 projection / pose verification 或更结构化的局部优化，而不是前置少用点。

### 2026-07-09：Two-stage LocalMap projection 实验与回退

实验曾使用默认关闭开关：

```text
XFEAT_LOCALMAP_TWO_STAGE=1
```

目标不是减少 secondary，而是在 weak/fallback 段使用 secondary 之前先细化 pose：

```text
direct + primary projection
-> direct + primary support >= 80 时，在 Frame copy 上做 PoseOptimization
-> 临时优化 inliers >= 60 时，只把 pose 写回 mCurrentFrame
-> 用新 pose 重新投影 secondary / expansion candidates
-> 最终仍走正式 TrackLocalMap PoseOptimization
```

为什么需要 Frame copy：

- `PoseOptimization()` 会改 Frame pose 和 outlier 状态。
- 第一阶段只想拿一个更准的 pose，不想污染正式 `mCurrentFrame` 的 outlier / match state。
- 因此只回写 `tmpFrame.GetPose()`。

为什么需要重新 `isInFrustum()`：

- `SearchByProjection()` 依赖 MapPoint 里由 `Frame::isInFrustum()` 写入的投影坐标。
- 如果 pose refine 后不重新投影，secondary 仍会使用旧 pose 的投影位置，two-stage 不会真正生效。

实验期间 `[TrackLocalMap][TemporalCore]` 增加了：

```text
two_stage
two_stage_min_support
two_stage_min_inliers
two_stage_support
two_stage_tried
two_stage_inliers
two_stage_accepted
```

三次 MH04 的 RMSE 为 `0.077933 / 0.073675 / 0.098668`，但进一步对比显示：

- two-stage 在采样帧中分别接受 `48 / 52 / 49` 次，行为确实生效。
- 临时优化只排除约 2% 的 direct + primary 关联。
- `primary_support=80~139` 区间的 local projection 外点率约为
  `21.8% / 21.2% / 21.6%`，与 adaptive primary-first 的 `21.1% / 21.1%`
  没有实质差异。
- 三次 1m RPE RMSE 为 `0.033559 / 0.035747 / 0.033863`，也没有优于 adaptive
  primary-first 的稳定证据。
- RMSE 差异与 loop 次数、GBA 是否完整应用以及 loop 前地图状态关系更强。
- `primary_support<80` 的最弱帧不满足二阶段优化条件，核心 weak fallback 问题没有被处理。

结论：two-stage 没有稳定、可测的精度收益。现已删除实现、环境变量和诊断字段，恢复为：

```text
primary support 足够:
    suppress secondary
primary support 不足:
    admit secondary
base support 仍不足:
    admit expansion
```
