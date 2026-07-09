# AirSLAM 对 XFeat-SLAM 后续适配的启发

本文档记录对 AirSLAM 代码、当前 XFeat + LightGlue + ORB-SLAM3 改造状态、以及 `docs/` 里既有实验记录的综合分析。目标是形成后续修改方向，避免继续在已经验证收益有限的单点 gate / 阈值调参里反复。

## 当前主线索引

截至 2026-07-09，后续实验以本文末尾的“连续性三次实验结论”和
“primary-first 自适应准入”为当前状态。前文 lifecycle、risk guard、KF policy 等内容保留为
历史分析，不代表仍在代码中启用。

当前 LocalMap 主线只有两层行为：

```text
temporal expansion-last
+ optional primary-first adaptive admission inside base
```

其中：

- `XFEAT_LOCALMAP_TEMPORAL_EXPANSION_LAST=1`：strict/voted base 先匹配，支撑不足再补 expansion。
- `XFEAT_LOCALMAP_PRIMARY_FIRST=1`：base 内 strict continuous 先投影；motion 正常且 `direct + primary`
  达到 `XFEAT_LOCALMAP_TEMPORAL_MIN_SUPPORT` 时，跳过 remaining strict/voted secondary。
- primary-first 第一版只做排序，实验显示 secondary 仍大量进入优化，因此已升级为自适应准入。
- 高体量结构/联合来源诊断只在 `XFEAT_DIAG_DEEP=1` 时输出。

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
- two-stage admission 已完成验证并因无稳定收益回退，当前保留 adaptive primary-first。

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

## 2026-07-09：Temporal LocalMap 与 MapPoint lifecycle 阶段结论

### 本阶段做了什么

在已经固定的 XFeat vocabulary、LightGlue loop 和 `nLevels=1` 基线上，本阶段继续验证了两层 LocalMap 分流：

```text
prior associations
+ strict temporal core projection
+ voted core projection
= base support

base support 不足
-> covisible/parent-child expansion projection
```

其中：

- `strict_core` 来自最近最多 5 个时序关键帧，或被这些关键帧观测过的 MapPoint。
- `voted_core` 来自当前/上一帧 MapPoint 对观察关键帧的投票，要求 `vote >= 15` 且局部观测数至少为 4。
- `expansion` 是其余由 reference、voted、covisible、parent/child 等 LocalMap 关系带入的 MapPoint。
- `base_support = prior associations + strict_core projection matches + voted_core projection matches`。

随后试验了 MapPoint lifecycle：

```text
stable / probation / unstable / unknown
```

该状态综合 close/mid 创建来源、观测次数、found ratio 和 local projection 优化历史。在 `XFEAT_LOCALMAP_LIFECYCLE_EXPANSION_GUARD=1` 时，仅在 base support 不足、必须启用 expansion 的帧中跳过 `unstable expansion`。

### 日志证据

`run_MH04_lifecycle_expansion_guard.log` 的 67 个采样帧显示：

- base support 平均约 `197.3`，中位数 `196`。
- 55/67 帧 base support 已达到 140。
- 只有 10/67 帧实际执行 expansion fallback。
- strict core：`5541` 个优化关联，`1053` 个外点，外点率约 `19.0%`。
- voted core：`877` 个优化关联，`232` 个外点，外点率约 `26.5%`。
- expansion：`26` 个优化关联，`14` 个外点，外点率约 `53.8%`。
- expansion 只占全部优化关联约 `0.4%`。
- lifecycle guard 检查了 `1708` 个 fallback expansion 候选，跳过 `1002` 个 unstable 候选。

这证明：

1. expansion 确实比 base 更脏。
2. lifecycle guard 确实能识别并阻止大量 unstable expansion 候选。
3. 但 expansion 触发频率和最终约束占比都很低，因此该 guard 难以稳定改变整段 RMSE。
4. 当前主要约束和主要误差都来自 base，尤其是 weak 段中的 voted core 和部分 strict core。

### 经验和教训

#### 1. 指标更差不等于它主导最终误差

expansion 外点率最高，但数量只占约 `0.4%`。继续细分 stable/probation/unstable 可以改善局部统计，却很难成为主要精度来源。

以后判断一个方向是否值得转成真实行为，需要同时满足：

```text
质量差异明显
+ 覆盖帧足够多
+ 对最终优化约束贡献足够大
```

不能只根据某个桶的外点率决定。

#### 2. post-opt outlier ratio 是诊断指标，不是最终优化目标

优化已经剔除的外点不一定伤害最终位姿。真正需要关注的是：

- 优化后内点数量。
- 图像空间覆盖。
- close/mid/far 深度分布。
- 连续帧复用率。
- pose residual 分布。
- motion/ref 与 local projection 的一致性。

低外点率但内点少、分布集中，可能比高外点率但内点充足且覆盖均匀更差。

#### 3. easy 与 difficult 的差异符合当前架构问题

easy 场景中 motion LG prior 稳定、投影中心准确，密集 XFeat 提供的冗余约束是优势。medium/difficult 中快速运动、模糊和曝光变化会先削弱 pose prior；大共视 LocalMap 随后提供大量相似投影候选，密集特征会放大相关错配和地图膨胀。

因此不能用全局减点或全局严格过滤解决，否则会损失 easy 场景已有优势。行为必须对 tracking 状态和几何支撑敏感。

#### 4. 当前 strict core 不是 AirSLAM 的连续 tracking core

当前 strict core 的条件只是“来自最近时序 KF”或“被最近时序 KF 观测过”。它没有证明该 MapPoint 最近连续成为 tracking 内点。

AirSLAM 的差异在于：

- 当前帧主要从参考关键帧继承 MapPoint。
- 位姿优化后只向当前帧传播内点的 track id 和 MapPoint。
- 局部优化窗口主要由最近时序关键帧组成。
- 局部优化外点观测会从地图关系中移除。

所以 AirSLAM 的 temporal core 是由连续匹配和优化内点自然形成的；我们当前的 strict core 仍然是 ORB-SLAM3 LocalMap 拓扑上的近似。

### 本次代码收敛

删除未形成稳定精度收益的实验行为和配套诊断：

- `XFEAT_LOCALMAP_LIFECYCLE_EXPANSION_GUARD`
- MapPoint `stable/probation/unstable/unknown` lifecycle 状态
- lifecycle selected/admitted/projection/pose funnel 日志
- `XFEAT_MOTION_HEALTH_GUARD`
- motion guard 的 MapPoint admission 行为和六个专用阈值环境变量
- `XFEAT_KF_INSERT_GUARD`
- KF guard 的 close coverage/stable support 重计算和行为分支
- `XFEAT_LOCALMAP_TEMPORAL_CORE_WINDOW`，时序窗口固定为 5

保留：

- `XFEAT_LOCALMAP_TEMPORAL_EXPANSION_LAST`
- `XFEAT_LOCALMAP_TEMPORAL_MIN_SUPPORT`
- 精简的 `[TrackLocalMap][TemporalCore]` strict/voted/expansion 统计
- 精简的 `[TrackMotion][Health]`，仅用于识别 weak motion 段，不再控制 admission
- 原始 ORB-SLAM3 关键帧插入行为和核心 `[Tracking][KFDecision]` 日志

### 下一步计划：连续 tracking base

下一阶段不直接增加新的 hard gate，先验证当前 base 中“连续 tracking 内点”是否具有足够的数量和更好的几何质量。

#### 阶段 1：最小连续性统计

优先复用当前 match source、PoseOptimization outlier 和 MapPoint local projection history，只补充最少状态：

```text
last tracking-inlier frame
recent tracking-inlier mask
recent successful reuse count
```

按来源统计：

```text
motion/ref current support
recent continuous temporal support
non-continuous strict support
voted support
```

每个桶只保留：

- association / inlier / outlier。
- 图像覆盖 cell 数。
- 深度分布。

#### 阶段 2：条件化分析

把上述统计与 motion health 联合分析：

```text
normal motion
weak motion
bad motion
```

重点回答：

1. 连续 temporal 点是否在 normal 和 weak 段都比现有 strict/voted 更干净。
2. 连续 temporal 点数量和空间覆盖是否足够独立支撑位姿。
3. strict/voted 外点率升高是 MapPoint 历史质量问题，还是当前 pose prior/运动状态导致的共同退化。

#### 阶段 3：只有证据充分才改变行为

如果连续 temporal 支撑在多次 MH04 以及 MH03/MH05 中都满足“质量更高且数量/覆盖足够”，再考虑：

```text
motion/ref + continuous temporal
-> 第一次 PoseOptimization
-> 支撑不足时补 remaining strict/voted
-> expansion 仍作为最后 fallback
```

不再重复：

- expansion lifecycle 细分。
- 单阈值 MapPoint risk guard。
- 全局减少关键帧或 MapPoint。
- 只追求降低 post-opt outlier ratio。

验收必须同时看多次运行的 median RMSE、worst run、无 loop/有 loop 分组结果，以及 weak 段的内点数量和空间覆盖。

### 2026-07-09：阶段 1 最小连续性诊断已实现

本阶段只增加诊断状态，不改变 LocalMap admission、projection matching、
PoseOptimization、关键帧插入或 MapPoint 创建。

每个 MapPoint 只记录最近最多 8 次 direct tracking 的结果：

```text
last direct tracking frame
last direct inlier frame
recent direct valid mask
recent direct inlier mask
```

direct tracking 来源限定为：

```text
motion_proj
motion_lg
ref
ref_lg
```

不包括 `local_proj`、`local_proj_lg_verified` 和 relocalization。历史在
`TrackLocalMap()` 最终 PoseOptimization 完成后更新；同一 MapPoint 在同一帧只记录
一次，双目左右关联只要存在一个内点就按内点记录。

当前仅用于诊断的 continuous strict 定义为：

```text
最近 direct inlier 距当前帧不超过 2 帧
+ 最近最多 8 次 direct 尝试中至少有 2 次有效记录
+ 至少有 2 次 direct inlier
```

`SearchLocalPoints()` 在本帧 LocalMap projection 前冻结该分类，因此不会用本帧
local projection 或最终优化结果反向证明本帧候选可靠。

现有 `[TrackLocalMap][TemporalCore]` 追加以下字段：

```text
direct_{assoc,inlier,outlier,outlier_ratio,inlier_cells,close_inlier}
strict_cont_{assoc,inlier,outlier,outlier_ratio,inlier_cells,close_inlier}
strict_noncont_{assoc,inlier,outlier,outlier_ratio,inlier_cells,close_inlier}
voted_core_{inlier_cells,close_inlier}
```

其中：

- `direct` 描述当前帧最终优化后的 motion/ref 支撑。
- `strict_cont` 和 `strict_noncont` 只统计 strict core 中由 local projection
  补入的约束。
- `inlier_cells` 复用现有特征网格，用于避免只看内点数量和外点率。
- `close_inlier` 使用当前 `mThDepth`，用于观察连续支撑是否只集中在近点。

没有新增环境变量。统计仍由现有 `XFEAT_DEBUG=1` 和
`XFEAT_DIAG_INTERVAL` 控制。下一步先比较 normal/weak motion 帧中的数量、外点率
和覆盖，不把该分类直接转成 hard gate。

## 2026-07-09：连续性三次实验结论

### 运行结果

同一 MH04 配置连续运行三次：

| 日志 | RMSE | loop | KFs |
|---|---:|---:|---:|
| `run_MH04_continuity_diag.log` | 0.085130 | 3 | 569 |
| `run_MH04_continuity_diag_1.log` | 0.071928 | 2 | 518 |
| `run_MH04_continuity_diag_2.log` | 0.118919 | 1 | 520 |

三次 RMSE 差异仍主要由 loop 时机和质量决定，但连续性桶的质量排序在三次运行中稳定。
201 个诊断采样帧汇总如下：

| 来源 | assoc | outlier | outlier ratio |
|---|---:|---:|---:|
| direct motion/ref | 20391 | 206 | 1.01% |
| strict continuous | 2321 | 286 | 12.32% |
| strict non-continuous | 14439 | 2775 | 19.22% |
| voted | 2765 | 758 | 27.41% |
| expansion | 176 | 102 | 57.95% |

这说明 continuity 是有效的质量排序信号，但不能直接做 strict-only：

- strict continuous 更干净，但只占 strict local projection 支撑的一小部分。
- strict non-continuous 和 voted 虽然更脏，仍提供大量有效内点。
- expansion 最脏且贡献很小，继续保持 support 不足时才启用是合理的。

### Primary 支撑计算

把 `primary support` 定义为：

```text
current direct motion/ref inlier
+ strict continuous local projection inlier
```

201 个采样帧合计为：

- 22220 / 22712 inlier/association，外点率 2.17%。
- 平均 110.55 个 primary inlier/帧。
- 173/201 帧达到 80 个 inlier。
- 以 `inlier >= 80 && union coverage >= 18 cells` 为保守标准，168/201 帧明确足够。
- normal motion 的 152 帧中，149 帧明确足够。
- weak motion 的 49 帧中，只有 19 帧明确足够，28 帧明确不足。

结论不是“只使用 primary”，而是：

```text
大部分 normal 帧：优先让 primary 占据冲突 keypoint
weak/低支撑帧：必须保留 remaining strict/voted 补充
expansion：继续作为最后 fallback
```

### Loop 与 LocalMap 的关系

最好一次在 KF 475、504 接受 rank-0 loop；最差一次在 KF 480 已有
`sim3_inliers=35` 的候选，但 `proj_opt_matches=39` 未通过，只在 KF 501 接受 loop。
三次 LocalMap continuity 总体统计接近，说明：

- primary-first 可以处理 LocalMap 匹配冲突的平均质量问题。
- 它不能单独消除最终 RMSE 的 loop 波动。
- 验收必须同时报告 no-loop/loop、loop candidate rank、Sim3/projection 支撑，不能把单次最低 RMSE
  归因于 LocalMap 修改。

## 2026-07-09：代码收敛与 Primary-First 自适应准入

### 删除和收敛

删除两个已有实验明确不适合作为主线的行为：

- `XFEAT_LG_MOTION_POLICY=projection_first` 及其 fallback 阈值。MH04 已显示该策略明显伤精度，
  motion 路径恢复为 LightGlue 优先、异常回退 projection。
- `XFEAT_LOCALMAP_FOUND_RATIO_GATE` 及三个配套参数、拒绝统计。单一 found-ratio admission
  没有形成稳定 RMSE 收益，也与当前连续性主线重复。

默认日志收敛为：

- 保留 `[TrackMotion][Health]`、`[SearchLocalPoints]`、`[TrackLocalMap][TemporalCore]`、
  `[TrackLocalMap][SourceError]`、`[TrackLocalMap][InlierMix]`。
- `[TrackLocalMap][KFStructure]` 和 `[TrackLocalMap][SourceMPJointError]` 只在
  `XFEAT_DIAG_DEEP=1` 时输出。
- TemporalCore 默认只打印 projection/optimization 结果；selected/already-seen/frustum/admitted
  funnel 只在 deep 模式打印。

### Primary-First 第一版结果

第一版新增一个开关：

```text
XFEAT_LOCALMAP_PRIMARY_FIRST=1
```

当时只在 XFeat temporal expansion-last 路径中改变匹配顺序：

```text
已有 direct motion/ref association
-> strict continuous MapPoint projection
-> remaining strict non-continuous + voted projection
-> base support 不足时 expansion projection
-> 一次 PoseOptimization
```

`run_MH04_primary_first.log` 显示它确实生效，但 RMSE 没有明显提升。67 个采样帧聚合：

- `motion_lg`：6897 associations，6838 inliers，外点率 0.86%。
- `local_proj`：6528 associations，5234 inliers，外点率 19.82%，和之前 continuity
  三次实验的 19.6% 到 20.1% 基本一致。
- `primary_matches=870`，`secondary_matches=5618`，primary 只占新增 projection matches 的 13.4%。
- secondary 主要来自 `strict_noncont=4876` 和 `voted_core=742`，不是 expansion。
- expansion 只有 40 个 matches，说明继续细分 expansion 对当前误差主因帮助有限。

结论：第一版 primary-first 只是 greedy 占位优先级，不是约束准入控制。它能处理少量
keypoint 冲突，但 secondary 仍大量进入 PoseOptimization，因此不能显著降低 local_proj 外点率。

### 当前 Primary-First 行为

当前版本复用同一个开关，不增加新环境变量：

```text
已有 direct motion/ref association
-> strict continuous MapPoint projection
-> 如果 motion health 正常且 direct + primary >= XFEAT_LOCALMAP_TEMPORAL_MIN_SUPPORT:
       suppress remaining strict non-continuous + voted secondary
       suppress expansion
   否则:
       admit remaining strict non-continuous + voted secondary
       如果 base support 仍不足，再 admit expansion
-> 一次 PoseOptimization
```

这个改动不做：

- 不改 MapPoint creation。
- 不改关键帧插入。
- 不改 LocalMapping 三角化。
- 不做第二次 PoseOptimization。
- 不新增一组 secondary 阈值环境变量。

`[TrackLocalMap][TemporalCore]` 新增：

```text
primary_first
primary_candidates / primary_matches
primary_support
secondary_candidates / secondary_matches
secondary_suppressed / secondary_suppressed_candidates
```

### 验收顺序

1. 同一 MH04 配置做关/开自适应 primary-first A/B，各至少三次。
2. 比较 median/worst RMSE，并按 loop 次数和首次 loop 质量分组。
3. 检查 `secondary_suppressed=1` 的帧是否集中在 motion 正常、primary support 充足的 normal 段。
4. 检查 weak motion 帧是否仍能启用 secondary / expansion fallback。
5. 对比 `local_proj` 外点率是否从约 20% 下降，而不是只看单次 RMSE。

## 2026-07-09：Adaptive Primary-First 运行结论与 Weak Fallback 负结果

### 三次运行结论

在固定 XFeat vocabulary + LightGlue loop + `XFEAT_LOCALMAP_TEMPORAL_EXPANSION_LAST=1`
和 `XFEAT_LOCALMAP_PRIMARY_FIRST=1` 后，MH04 三次代表性运行显示：

- `secondary_suppressed=1` 的帧确实更干净。与旧 primary-first 同帧近似对比，suppressed 帧
  的 local projection 外点率从约 `13.8%~18.0%` 降到约 `8.0%~15.5%`。
- 但 suppressed 帧只贡献很小一部分 local projection 约束。大部分 local projection 仍来自
  weak / fallback 帧中的 secondary。
- 三次 sampled frame 聚合中，`local_proj` 外点率仍约 `20%~22%`，说明 adaptive primary-first
  主要改善 normal 段，没有解决 weak 段。
- 保存轨迹的两次中，weak sampled frame 的 APE RMSE 高于 normal frame：
  - `0.084818` run：weak `0.1058`，normal `0.0787`。
  - `0.097593` run：weak `0.1159`，normal `0.0916`。
- frame 930 是典型 weak fallback 例子：`primary_support≈58~62`，local projection 外点率约
  `42%~44%`，采样误差约 `0.21~0.23m`。

这说明当前问题不是 normal 段 primary 不够强，而是：

```text
primary 不够的 weak 段
-> 必须启用 secondary / voted / expansion
-> secondary 中混入大量历史不连续、质量一般的 MapPoint
-> local_proj 外点率上升
-> loop 前局部地图和局部轨迹状态更不稳定
```

### 代码收敛

继续保留当前主线：

```text
XFEAT_LOCALMAP_TEMPORAL_EXPANSION_LAST=1
XFEAT_LOCALMAP_PRIMARY_FIRST=1
```

不恢复：

- `XFEAT_LOCALMAP_RISK_GUARD`
- MapPoint lifecycle expansion guard
- old `XFEAT_LOCALMAP_CORE_FIRST`
- KF policy guard
- 长期 shadow/audit 日志

### 已回退：Weak Fallback Secondary Admission

曾短期试验：

```text
primary_support 不足或 motion weak
-> secondary 按质量排序
-> 按 support deficit 限量 admit secondary
-> 如果 base 仍不足，再 admit expansion
```

两次 MH04 运行显示该行为确实生效，但不适合作为主线：

| 日志 | local_proj outlier | secondary matches | KFs | loops |
|---|---:|---:|---:|---:|
| `run_MH04_weak_secondary_admission_1.log` | 18.6% | 4198 | 577 | 3 |
| `run_MH04_weak_secondary_admission_2.log` | 19.6% | 4309 | 572 | 2 |

对比前一版 adaptive primary-first：

- `local_proj` 外点率从约 `20.0%~21.8%` 略降到 `18.6%~19.6%`。
- secondary matches 从 `4667~5344` 降到 `4198~4309`。
- KeyFrame 数量从 `529~557` 增到 `572~577`。
- loop 接受没有更早、更强；最好 loop 几何也不如 `0.084818` run 中的 `proj_opt=179`。

结论：

```text
secondary 限量能降低局部 outlier 指标
但也减少了 weak 段必要内点支撑
导致关键帧更多、loop 前地图状态没有变好
因此不保留该真实行为
```

当前代码已经回到前一版 adaptive primary-first：

```text
primary support 足够:
    suppress secondary / expansion
primary support 不足:
    admit all secondary
    base 仍不足时才启用 expansion
```

后续不再继续调 secondary 排序分数、budget 或 source/risk 阈值。

### 下一步：Two-Stage LocalMap Projection

weak-secondary admission 的失败说明，问题不能通过“少用 secondary”解决。secondary 虽然更脏，
但提供了大量必要内点；减少它会降低有效支撑、增加 KeyFrame 数量。

随后验证的修改方向是：

```text
不是减少 secondary
而是在使用 secondary 之前，把当前帧 pose prior 先拉准
```

实现边界：

```text
motion/ref pose prior
-> primary projection
-> 如果 primary support 不足 140，但 direct + primary >= 80:
       copy Frame
       在临时 Frame 上做 PoseOptimization
       如果临时优化 inliers >= 60:
           只把优化后的 pose 写回 mCurrentFrame
-> 用更新后的 pose 重新投影 secondary / expansion candidates
-> 最终仍做一次正式 TrackLocalMap PoseOptimization
```

关键点：

- 不新增 MapPoint。
- 不删除或减少 secondary。
- 不复用临时 Frame 的 outlier / match state，只复用优化后的 pose。
- pose 更新后，secondary/expansion deferred candidates 必须重新 `isInFrustum()`，否则 matcher
  会继续使用旧 pose 下的投影坐标。
- 该实验通过 `XFEAT_LOCALMAP_TWO_STAGE=1` 验证，现已从代码中删除。

实验期间 `[TrackLocalMap][TemporalCore]` 曾增加：

```text
two_stage
two_stage_min_support
two_stage_min_inliers
two_stage_support
two_stage_tried
two_stage_inliers
two_stage_accepted
```

三次 MH04 结果：

| 运行 | RMSE | 1m RPE RMSE | KFs | loop / GBA |
|---|---:|---:|---:|---|
| `run_MH04_two_stage_localmap_1.log` | 0.077933 | 0.033559 | 562 | 2 / 2 applied |
| `run_MH04_two_stage_localmap_2.log` | 0.073675 | 0.035747 | 560 | 3 / 3 applied |
| `run_MH04_two_stage_localmap_3.log` | 0.098668 | 0.033863 | 539 | 3 / 2 applied |

67 个采样帧中，two-stage 分别尝试并接受 `48/48`、`52/52`、`49/49` 次，但：

- 临时优化只剔除约 2% 的 direct + primary 关联，说明输入 pose 已经较干净。
- two-stage 覆盖的 `primary_support=80~139` 区间，`local_proj` 外点率分别为
  `21.8% / 21.2% / 21.6%`。
- 前一版 adaptive primary-first 同一区间为 `21.1% / 21.1%`，没有改善。
- 1m RPE 与 adaptive primary-first 的 `0.034453 / 0.034345` 基本相同。
- 最弱的 `primary_support<80` 帧不会触发 two-stage，而这些帧仍是主要风险段。

因此低 RMSE 主要不能归因于 two-stage，而是受到 loop/GBA 和运行状态影响。该逻辑、环境变量和
诊断字段已删除，当前代码回到 adaptive primary-first。
