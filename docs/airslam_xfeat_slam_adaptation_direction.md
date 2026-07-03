# AirSLAM 对 XFeat-SLAM 后续适配的启发

本文档记录对 AirSLAM 代码、当前 XFeat + LightGlue + ORB-SLAM3 改造状态、以及 `docs/` 里既有实验记录的综合分析。目标是形成后续修改方向，避免继续在已经验证收益有限的单点 gate / 阈值调参里反复。

## 背景

当前项目已经完成了这些关键改造：

- `nLevels=1 / keypoint.octave=0`，不再让 XFeat 强行继承 ORB 金字塔尺度语义。
- LightGlue 已参与 motion tracking、reference KF tracking、relocalization、LocalMapping triangulation、loop candidate validation。
- 已接入 XFeat float DBoW3 vocabulary，并通过 `XFEAT_USE_XFEAT_VOCAB=1` 替换 loop candidate retrieval。
- loop 已有 XFeat vocab + LightGlue + Sim3 + projection refinement + CorrectLoop/GBA 链路。
- 已做过多轮 KF guard、LocalMap guard、source-aware/risk-aware、core-first、tracklet、measurement weighting 等实验，其中很多方向有局部指标改善，但 RMSE 稳定性收益有限。

历史日志反复支持的核心链条是：

```text
XFeat close/mid 点多
-> close/mid MapPoint 创建多
-> 共视关系变密
-> LocalMap MapPoint 候选池变大
-> projection matching 补进更多不稳定 3D-2D 约束
-> PoseOptimization 里 local_proj outlier 高
-> medium/difficult 场景精度不稳定
```

AirSLAM 的价值在于：它同样使用 learned point feature + LightGlue/SuperGlue，但没有简单沿用 ORB-SLAM3 的大共视 LocalMap projection 机制，而是从跟踪、关键帧、地图点、局部优化和回环验证都重新设定了边界。

## AirSLAM 全流程代码结论

### 特征与匹配

AirSLAM EuRoC 配置使用 SuperPoint，`max_keypoints=400`，点匹配器默认 LightGlue：

- `/home/zjz/workspace/AirSLAM/configs/visual_odometry/vo_euroc.yaml`
- `/home/zjz/workspace/AirSLAM/src/point_matcher.cc`

`PointMatcher::MatchingPoints()` 的重要细节：

- LightGlue 输出 2D-2D matches。
- 在 `outlier_rejection=true` 时，匹配后会执行 fundamental matrix RANSAC。
- RANSAC 通过后才把 matches 交给 tracking / loop / relocalization。

这说明 AirSLAM 并不直接信任 LG raw matches。LG 是强 matcher，但几何一致性仍然单独过滤。

对当前项目的启发：

- loop LG initial matches 后应优先增加 F/RANSAC 或 E/RANSAC 预过滤。
- 这比继续调 `XFEAT_LIGHTGLUE_CONF` 更有针对性，因为当前问题经常是 raw/direct matches 足够，但 `sim3_inliers` 不稳定。
- motion/ref LG 是否也需要几何预过滤要谨慎；loop 是最适合先试的入口，因为 loop 2D-2D 错配对 Sim3 初值影响最大。

### 前端跟踪

AirSLAM 的 tracking 主要是：

```text
last keyframe/current frame LG 2D-2D
-> 继承 last keyframe 对应 MapPoint
-> PnP/pose optimization
-> 优化后只保留 inlier matches
```

相关代码：

- `/home/zjz/workspace/AirSLAM/src/map_builder.cc`

它没有在普通跟踪中把很大的 covisible LocalMap 全量投影到当前帧来补大量 3D-2D 约束。当前帧主要先依赖 reference/last-keyframe track chain，优化后再更新 track id 和 MapPoint association。

对当前项目的启发：

- 我们的 motion/ref LG 路线是合理的，日志也显示 `motion_lg` 通常很干净。
- 真正需要重构的是 `SearchLocalPoints()` 的 LocalMap projection admission，而不是继续怀疑 LG motion。
- 当前 ORB-SLAM3 风格 LocalMap projection 是 learned feature 适配中的主要边界不匹配点。

### 关键帧插入

AirSLAM 的 `AddKeyframeCheck()` 使用：

- match count；
- match ratio，相对于 reference features 和 current features；
- average parallax；
- stereo 初始化/关键帧需要足够 stereo points。

它不是主要依赖 ORB-SLAM3 stereo/RGBD 的 close-point pressure：

```cpp
bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70)
```

相关代码：

- `/home/zjz/workspace/AirSLAM/src/map_builder.cc`

对当前项目的启发：

- 我们之前的 `XFEAT_KF_INSERT_GUARD` 是在 ORB-SLAM3 close-point 插帧规则外面做抑制，属于补丁式方案。
- AirSLAM 的结构更清晰：关键帧插入应由 tracking match ratio 和 parallax 主导，而不是由“有多少 close depth feature 尚未成为 MapPoint”主导。
- 后续如果继续做关键帧策略，建议从规则定义上引入 AirSLAM 风格的 XFeat-specific KF decision，而不是继续微调 close guard。

建议目标：

```text
XFeat stereo/RGBD 模式下：
1. tracking/reference match ratio 低 -> 允许插 KF
2. 视差足够 -> 允许插 KF
3. 纯 close pressure 且 tracking support 强、视差不足、距上个 KF 很近 -> 抑制插 KF
```

这比“少插关键帧”更准确。目标不是降低 KF 数，而是降低冗余 close-only KF。

### MapPoint 创建

AirSLAM 插入关键帧时：

- 对还没有 MapPoint 的 tracked feature 创建 Mappoint。
- 有 stereo depth 时直接 back-project。
- 没有足够几何时保持 untriangulated。
- 观测数超过阈值后再 triangulate。
- 局部优化后会直接移除 outlier observation。

相关代码：

- `/home/zjz/workspace/AirSLAM/src/map.cc`

对当前项目的启发：

- “创建很多 MapPoint”本身不是唯一问题；问题是新点过早进入 covis graph / LocalMap projection / pose edges。
- 之前我们做过 pending tracklet / eligibility / reservation，结果说明“延迟创建 MapPoint”单独不够，因为它可能削弱当前 tracking support。
- 更合理的是给 MapPoint 进入不同后端环节设置生命周期：

```text
new close/mid point
-> 可用于短期 tracking / reference association
-> 暂缓扩大 covis / LocalMap projection 权重
-> 多帧稳定观测后再成为 full local-map candidate
```

这比 hard culling 或 global MapPoint reduction 更符合 AirSLAM 的思路。

### LocalMap / 局部优化

AirSLAM 的 LocalMapOptimization 默认选最近最多 5 个 frame/keyframe，然后收集这些帧观测到的 MapPoint/MapLine，再加入观察这些点的 fixed frames。

相关代码：

- `/home/zjz/workspace/AirSLAM/src/map.cc`

这和 ORB-SLAM3 的大 covisible LocalMap projection 不同。AirSLAM 的优化窗口是 temporal-core first，而不是广泛共视扩展优先。

对当前项目的启发：

- 我们的 `XFEAT_LOCALMAP_CORE_FIRST` 方向是对的，但上一版仍是在 ORB-SLAM3 大 LocalMap pool 内做排序/分流，因此效果不稳定。
- 后续应把 LocalMap 改成更明确的两层结构：

```text
core local map:
  当前 motion/ref/recent keyframes 稳定观测到的点
  reference KF / last KF / short temporal chain
  有较好 found ratio / observations / recent inlier history

expansion local map:
  covisible graph 拉进来的远关系点
  close/mid new points
  local_mapping_triangulation early-life points
```

使用策略：

- core points 正常参与 projection matching 和 PoseOptimization。
- expansion points 只在 core support 不足、或者 loop/relocalization 需要补约束时进入。
- expansion 不是删除，而是当前帧 admission 降级。

这比继续在所有 LocalMap points 上按 found ratio / source / risk 做 hard gate 更稳。

### 回环检测与验证

AirSLAM 的 map refinement loop 流程是：

```text
DBoW query
-> common words filter
-> connected frames grouping
-> group score accumulation
-> group score filter
-> top groups 做 LightGlue + RANSAC
-> 相对位姿优化
-> 在 loop group 内通过 BoW word + descriptor + reprojection 找更多 matches
-> pose graph refinement / map merge / GBA
```

相关代码：

- `/home/zjz/workspace/AirSLAM/src/map_refiner.cc`

这对我们当前 loop 不稳定非常重要。我们现在虽然已经接入 XFeat vocab，但后续验证仍偏 candidate-KF driven。日志中多次出现：

- 有多个可能 loop 区域。
- 有时早期 loop 接受，后期 700 附近 loop 不再触发。
- 有时早期没接受，后期 loop 一定接受。
- `candidate_rank`、`sim3_inliers`、`proj_opt_matches` 对最终 RMSE 影响很大。

AirSLAM 的启发：

- loop 应该从“单候选 KF 验证”逐步转为“loop region / group 验证”。
- XFeat vocab 返回的不应只是 topN KFs，而应保留 candidate group 信息。
- LightGlue 验证应在 top groups 内选择几何最强候选。
- Sim3 初值成功后，再从 group 内做 reprojection-guided 补匹配，而不是盲目扩大 LG covisible windows。

## 与当前代码的对应关系

### 已经接近 AirSLAM 的部分

这些部分方向正确，应保留：

- `TrackWithMotionModel()` 中 `XFEAT_USE_LIGHTGLUE_MOTION=1`。
- `TrackReferenceKeyFrame()` 中 `XFEAT_USE_LIGHTGLUE_REF=1`。
- `Relocalization()` 中 `XFEAT_USE_LIGHTGLUE_RELOC=1`。
- `LocalMapping::CreateNewMapPoints()` 中 `XFEAT_USE_LIGHTGLUE_TRIANG=1`。
- XFeat float vocabulary + `XFeatKeyFrameDatabase`。
- loop LG candidate validation。
- `nLevels=1 / octave=0`。

这些方向和 AirSLAM 的 learned feature 接入边界一致：LG 做 2D-2D，float descriptor 用 float vocab，不强行套 ORB binary vocabulary。

### 当前仍和 AirSLAM 差异较大的部分

#### 1. KeyFrame insertion

当前项目仍以 ORB-SLAM3 原始 close-point pressure 为核心条件之一。XFeat 点更多，导致 close-only 插帧被放大。

AirSLAM 则使用 match ratio + parallax。

后续方向：

- 新增 XFeat-specific KF decision path。
- 不再只做 `XFEAT_KF_INSERT_GUARD` 的 close-only 抑制。
- 用 match ratio、parallax、frames since last KF、tracking support 共同决策。

#### 2. LocalMap projection

当前项目的 `SearchLocalPoints()` 仍是 ORB-SLAM3 大共视 LocalMap projection 结构。已有日志显示 local_proj outlier 高于 motion_lg。

AirSLAM 则更偏 temporal-core local optimization。

后续方向：

- 明确 core local map 与 expansion local map。
- projection matching 先跑 core。
- expansion 只在 core 支撑不足时启用。
- 不再继续叠加多种 source/risk/history hard gate。

#### 3. Loop candidate validation

当前项目已接入 XFeat vocab，但 loop 验证仍是 candidate-KF driven，covisible expansion 是轻量补丁。

AirSLAM 是 group-driven。

后续方向：

- XFeatKeyFrameDatabase 输出 candidate groups，而不是只输出 top KFs。
- LoopClosing 对 top groups 做 LG + F/RANSAC + Sim3。
- Sim3 后在 group 内做 reprojection-guided MapPoint 补匹配。

#### 4. MapPoint lifecycle

当前项目中 close/mid tracking stereo 和 local_mapping_triangulation MapPoint 一旦创建，就较快参与 covis/local map。

AirSLAM 的窗口更克制，outlier observation 会在 local optimization 后移除。

后续方向：

- 不是删除新点，而是限制 early-life close/mid points 对 covis/local projection 的影响。
- 可以增加 MapPoint local-map eligibility state，但需要保持 tracking support 不被削弱。

## 优先级建议

### 第一优先级：Loop group 验证

原因：

- MH04 RMSE 波动已经明显和 loop 接受位置、candidate rank、Sim3 initial support 相关。
- XFeat vocab 已经接入，下一步不是继续训练词袋，而是让 retrieval 结果和 LG validation 形成完整契约。
- AirSLAM 直接证明 group-level loop validation 是 learned feature 系统里的合理结构。

建议修改：

1. 在 `XFeatKeyFrameDatabase` 中保留候选 group 信息：
   - deputy KF；
   - group frames；
   - group score；
   - top member scores；
   - common words。
2. `LoopClosing` 新增 group-level validation 路径：
   - 对每个 group 的 deputy/top members 做 LightGlue。
   - LG 后做 F/RANSAC 预过滤。
   - 用过滤后的 matches 进入 Sim3。
   - 选择 group 内几何最强结果，而不是只按原 candidate rank。
3. Sim3 成功后，从 group frames 内做 reprojection-guided 补匹配：
   - 不增加大量 LG 调用；
   - 使用 vocab word / descriptor / reprojection gate；
   - 目标是提升 `proj_opt_matches` 的可靠性，而不是盲目扩展窗口。

验收指标：

- `candidate_rank` 对 RMSE 的影响降低。
- `sim3_inliers` 更稳定。
- 早期 loop 与 700 附近 loop 的接受逻辑更可解释。
- 不显著增加 GPU 占用。

### 第二优先级：XFeat-specific keyframe decision

原因：

- 深度特征点多会放大 ORB-SLAM3 close-point 插帧逻辑。
- 之前 guard 只是在原规则外部抑制，能减少 KF，但不稳定改善 RMSE。
- AirSLAM 的 match ratio + parallax 是更结构化的替代规则。

建议修改：

1. 保留 ORB 模式原逻辑。
2. XFeat stereo/RGBD 模式下新增单独 decision branch。
3. 指标：
   - ref/current LG match count；
   - match ratio to reference KF；
   - match ratio to current frame；
   - average parallax；
   - frames since last KF；
   - current local map support。
4. 只把 close pressure 作为辅助信号，不作为主触发。

验收指标：

- KF 数降低或更稳定，但 `mnMatchesInliers` 不下降。
- local map size 增长放缓。
- local_proj outlier 不上升。
- easy 场景精度不被破坏。

### 第三优先级：Temporal-core LocalMap projection

原因：

- 当前 local_proj 是主要脏源。
- AirSLAM 的 temporal window 避免了大共视 LocalMap 过早污染当前帧。
- 之前 core-first 有局部有效证据，但实现仍依附于大 pool。

建议修改：

1. `UpdateLocalKeyFrames()` 或 `SearchLocalPoints()` 中为 local KFs 标注来源：
   - reference；
   - last keyframe / temporal chain；
   - strong covisible；
   - weak covisible；
   - child/parent；
   - loop-related。
2. `SearchLocalPoints()` 分两阶段：
   - stage A：core local projection；
   - stage B：expansion projection。
3. Stage B 只在以下条件启用：
   - core matches 低；
   - current tracking support 弱；
   - relocalization/recently lost；
   - loop correction 后需要补局部约束。

验收指标：

- `local_proj` outlier ratio 下降。
- `assoc_added_by_local` 不断崖。
- `mnMatchesInliers` 不低于 baseline 90%。
- local map projection candidate 数下降但 tracking 不弱。

### 第四优先级：MapPoint early-life local-map eligibility

原因：

- 单纯少建点会伤 tracking。
- 单纯让所有新点参与 LocalMap 会放大 covis 和 local_proj 污染。

建议修改：

- 新增轻量 eligibility，而不是复杂 pending tracklet：

```text
full_tracking: 可用于当前/下一帧 tracking
local_map_candidate: 可进入 SearchLocalPoints expansion
core_local_candidate: 可进入 SearchLocalPoints core
```

promotion 条件：

- observation >= 3；
- found ratio >= 0.35；
- 最近 local projection inlier；
- 被 motion/ref LG 支撑过；
- depth consistency 稳定。

注意：

- 不要默认 hard reject 当前帧 tracking association。
- 不要重新启用复杂 tracklet reservation，除非日志证明 feature repeatability 是主要瓶颈。

## 暂不建议的方向

### 不建议直接引入线特征

AirSLAM 的线特征不是孤立模块，而是和 junction、mapline、line constraint、structure graph relocalization 一整套绑定。直接迁移成本很高，且当前主要瓶颈不是缺线，而是点特征和 ORB-SLAM3 后端契约不匹配。

### 不建议继续扩大 loop LG window

历史实验已经显示扩大 candidates/windows 会提高 GPU 负载，但不能稳定解决 loop 接受质量。下一步应提升 group-level 几何验证质量，而不是增加 LG 调用次数。

### 不建议继续单点调 found ratio / descriptor / residual gate

已有日志显示：

- 某些 gate 能降低局部 outlier；
- 但可能损失 inlier 支撑；
- RMSE 不稳定；
- 多数阈值不能解释 loop 前局部地图状态和回环质量波动。

后续要做结构改动，而不是继续调单阈值。

### 不建议继续把所有问题归因于词袋训练

XFeat vocabulary 已经改善了候选语义，但 run-to-run 仍受 loop group 几何质量和 loop 前 local map 状态影响。词袋可以继续优化，但不是唯一瓶颈。

下一轮词袋优化应服务于 group-level loop：

- 输出 group candidates；
- 评估 true loop region rank；
- 看 top groups 是否稳定覆盖 590/700 等 loop regions。

## 推荐阶段计划

### 阶段 A：Loop group validation

目标：降低 loop 接受位置波动。

代码范围：

- `XFeatKeyFrameDatabase`
- `LoopClosing`
- `XFeatLighterGlueMatcher` 可选增加 F/RANSAC helper

不改：

- tracking；
- LocalMapping；
- KF insertion；
- GBA；
- vocabulary training。

实验：

- MH04 至少 5 次；
- 比较 loop accepted current_kf/candidate_kf/group_id；
- 比较 `sim3_inliers / proj_opt_matches / GBA applied / RMSE`。

### 阶段 B：XFeat-specific KF decision

目标：减少冗余 close-only KF，同时保留有用近场锚点。

代码范围：

- `Tracking::NeedNewKeyFrame()`
- 复用现有 `[Tracking][KFDecision]` 日志。

不改：

- LocalMap；
- MapPoint 创建；
- loop。

实验：

- MH04、MH03、MH05、V101/V102；
- 每个序列至少 3 次；
- 按 loop 状态分组比较 RMSE。

### 阶段 C：Temporal-core LocalMap

目标：降低 local_proj 脏约束。

代码范围：

- `Tracking::UpdateLocalKeyFrames()`
- `Tracking::UpdateLocalPoints()`
- `Tracking::SearchLocalPoints()`

不改：

- MapPoint 创建；
- loop；
- KF decision。

实验：

- 先 shadow 统计 core/expansion 的候选、匹配、inlier/outlier。
- 再启用 two-stage admission。

### 阶段 D：MapPoint early-life eligibility

目标：控制新 close/mid 点对 covis/local map 的早期影响。

代码范围：

- `MapPoint`
- `Tracking::SearchLocalPoints()`
- `LocalMapping::CreateNewMapPoints()`

前提：

- 阶段 B/C 之后仍显示 early-life close/mid points 是主要污染源。

## 最小推荐下一步

如果只选一个最值得做的改动，建议先做：

```text
Loop group validation + LG initial F/RANSAC prefilter
```

理由：

- 与 AirSLAM 最直接对应。
- 当前 RMSE 波动最明显地受 loop 接受质量影响。
- 不会干扰 tracking、keyframe、local map 的其它变量。
- 可以复用现有 XFeat vocab、LoopLG 日志和 Sim3 后端。

如果这个阶段仍然不能稳定 loop，则再回到词袋训练/召回评估；如果 loop 稳定后 RMSE 仍明显落后 ORB-SLAM3，再进入 KF decision 和 temporal-core LocalMap。

## 2026-07-01 代码收敛记录

当前 MH04 简洁配置已经能稳定跑到约 `0.11~0.13` RMSE，说明主收益来自：

- XFeat vocabulary 作为 loop retrieval；
- 每个 candidate 直接做一次 KF-KF LightGlue；
- 后续沿用 ORB-SLAM3 的 Sim3、projection refinement、CorrectLoop/GBA。

因此本次代码清理把此前实验性的 loop consistency 和 covisible expansion 路径移除，不再保留：

- `XFEAT_LOOP_CONSISTENCY*`
- `XFEAT_LOOP_LG_EXPAND*`
- `XFEAT_LIGHTGLUE_LOOP_CONF`

保留的 loop 主线是：

```text
XFeat vocabulary candidates
-> direct KF-KF LightGlue
-> AirSLAM-style fundamental RANSAC prefilter
-> Sim3Solver
-> projection refinement
-> OptimizeSim3
-> covisible validation
-> CorrectLoop/GBA
```

AirSLAM 里 `PointMatcher::MatchingPoints(..., outlier_rejection=true)` 会在 LightGlue/SuperGlue 后用 `findFundamentalMat(..., FM_RANSAC, 20, 0.99)` 过滤 2D-2D outlier。本项目借鉴这一点，只在 loop LG 初始匹配进入 Sim3 前做轻量几何过滤，不改 tracking、LocalMap、MapPoint 创建和后端优化。

后续实验重点看 `[LoopLG][Accept]`：

- `candidate_rank`
- `initial_matches`
- `direct_raw / direct_assigned`
- `direct_f_inliers / direct_f_rejects`
- `sim3_inliers`
- `proj_opt_matches`
- `cov_valid`

如果 F/RANSAC 后 `initial_matches` 明显不足或 loop 变晚，说明过滤过强，需要回退或调阈值；如果 `direct_f_rejects` 能去掉明显外点并提升 `sim3_inliers` 稳定性，则继续保留这个方向。

## 2026-07-03 代码整理记录

当前 loop 代码进一步固定为 group-driven XFeat loop validation：

```text
XFeat vocabulary group candidates
-> top groups 进入 LightGlue validation
-> direct KF-KF LightGlue + fundamental RANSAC
-> Sim3Solver
-> projection refinement
-> weak Sim3/projection 时用 group word + descriptor + reprojection 做轻量 refill
-> OptimizeSim3 / covisible validation
-> CorrectLoop/GBA
```

本次整理保留：

- `[LoopLG][Accept]`：真实接受 loop 时始终打印，用于对齐 `*Loop detected` 和 RMSE。
- `[LoopLG][NearMiss]` / `[LoopLG][BestRejected]`：只在 `XFEAT_DEBUG=1` 且满足 `XFEAT_DIAG_INTERVAL` 时打印，避免 debug 日志过密。
- `mpLoopCloser is not finished` / `mpLoopCloser is running GBA, waiting...`：保留 shutdown 等待提示，便于确认退出等待是否卡在 LoopClosing/GBA。

本次整理不再恢复：

- `XFEAT_LOOP_CONSISTENCY*`
- `XFEAT_LOOP_CANDIDATES`
- `XFEAT_LOOP_LG_EXPAND*`
- `XFEAT_LIGHTGLUE_LOOP_CONF`

后续如果 loop 仍然不稳定，优先分析 group candidate 是否覆盖真实 loop region、`direct_f_inliers` 是否足够、以及 `sim3_inliers / proj_opt_matches` 是否被 loop 前 LocalMap 状态拖低，而不是重新叠加旧的单阈值 gate。
