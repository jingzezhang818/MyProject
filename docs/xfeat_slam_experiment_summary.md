# XFeat SLAM 实验路线总结

本文档用于记录已经验证过的改动方向、主要日志结论和后续决策边界，避免在同一类 gate、budget、阈值和 shadow diagnostic 里反复兜圈子。

## 当前保留基线

当前相对安全的主线应理解为：

- `Frame.cc` stereo front-end 保留稳定版本：`single-level band + local cost volume + current-layer refinement + LR-aware sigma`。
- `CreateNewMapPoints()` 保留 close/mid 入口控质：close/mid stereo fallback reject、close ray parallax/reprojection/scale 收紧。
- `SearchLocalPoints()` 保留保守 history gate。
- `XFeatMatcher::SearchByProjection()` 保留 `local_mapping_triangulation && 0 < depth <= 10m && proj_norm > 1.0` outside gate。
- Rover-style backend contract 默认关闭：默认应看到 `rover_backend=0`、`level_gate=1`；只有显式 `XFEAT_ENABLE_ROVER_BACKEND=1` 时才跑 Rover A/B。
- `LocalMapping.cc` 中 mapping probation、pending tracklet、mapping feature budget/repeatability 等真实行为默认关闭；相关字段只作为诊断或历史记录，不应继续影响 `CreateNewMapPoints()`。
- 保留结构诊断和 shadow diagnostics，但不要默认启用已经证明不稳定的真实行为。

## 当前新增诊断：close/mid triangulation input failure

本轮目标是固定安全基线后，回答 `CreateNewMapPoints()` 中 close/mid `local_mapping_triangulation` 为什么 descriptor/tracklet 可匹配但过不了几何三角化。

代码状态：

- Rover-style broad behavior 改为显式 opt-in：`XFEAT_ENABLE_ROVER_BACKEND=1`。
- 默认恢复 ORB-like XFeat projection contract：level gate 开启，projection radius 不做 Rover clamp，Optimizer 使用 octave-aware XFeat sigma。
- 新增 deep 诊断 `[LocalMapping][TriangInputFailure]`，仅在 `XFEAT_DEBUG=1 && XFEAT_DIAG_FEATURE_DISTRIBUTION=1 && XFEAT_DIAG_DEEP=1` 时输出。
- 当前版本在原有失败原因基础上扩展 pair/KF2 上下文桶，仍然只读，不改变 `CreateNewMapPoints()` 行为。

诊断内容：

- 只统计 XFeat close/mid ray triangulation / stereo fallback candidate，不改变任何创建或拒绝行为。
- 按 `backend_bin / depth_bin / octave_pair_bin / current_octave_bin / response_bin / grid_bin / edge_distance_bin` 聚合。
- 额外按 pair/KF2 上下文聚合：`neighbor_rank_bin / kf_id_delta_bin / covis_weight_bin / baseline_bin / baseline_depth_ratio_bin / kf2_octave_bin / kf2_response_bin / kf2_edge_distance_bin`。
- 输出 terminal event：`stereo_fallback_reject / parallax_reject / cheirality_reject / reproj_reject / scale_reject / created`。
- reprojection reject 拆成 `reproj_kf1_only / reproj_kf2_only / reproj_both`，用于判断失败是否集中在某一侧 keypoint measurement。

验收边界：

- 不加 `XFEAT_DIAG_DEEP=1` 时行为应与安全基线一致。
- 加 deep 后 `[TriangInputFailure]` 的 reject/create 总量应能和 `[TriangCreate]` close 计数对账。
- 若 `parallax_reject` 集中在低 `baseline_depth_ratio`、低 `covis_weight`、靠后的 `neighbor_rank` 或较大的 `kf_id_delta`，下一轮优先考虑 `CreateNewMapPoints()` 的 neighbor KF pair selection，而不是后端 gate。
- 若 `reproj_kf2_only` 集中在 KF2 的低 response、off2p、near-edge 等桶，再考虑 KF2 侧 keypoint quality/refine shadow。

### 最新日志结论：`run_SePT01_stereo_anchor_shadow.log`

本轮 deep 诊断回答了两个问题：

1. `5_10m` close/mid LMT 的创建瓶颈仍然是 two-view 几何视差。
2. 用 stereo anchor 替代 ray triangulation 暂时没有可行覆盖面。

关键统计：

- 系统行为仍在安全基线附近：`mnMatchesInliers≈231.2`、`assoc_added_by_local≈131.0`、`local_mps≈5875.7`，无 `LOST / RECENTLY_LOST`。
- `local_proj + local_mapping_triangulation` 仍是残留脏源：`outlier_ratio≈0.210`。
- `motion_lg + local_mapping_triangulation` 很干净：`outlier_ratio≈0.0022`，说明 LMT MapPoint 并非整体不可用，问题主要出现在 current-frame local projection / close-mid 几何支撑。
- close/mid ray candidates `3910`，创建 `2664`。
- `5_10m` 创建率仍低：`created=480/1025≈0.468`，`parallax_reject=476/1025≈0.464`。
- `baseline_depth_ratio` 仍解释 parallax：`low≈0.451`、`mid≈0.341`、`strong≈0.169`。

Stereo-anchor shadow 结果：

- stereo anchor 覆盖率极低：`anchor_valid=169/3910≈0.043`。
- 可用 anchor 全部来自 `neighbor_only`，当前 KF 侧没有可用 anchor：`current_only=0`、`both=0`。
- `anchor_pass=66/3910≈0.017`，覆盖不足。
- `anchor_strong_better + anchor_better = 5/3910≈0.0013`，没有形成可用替代信号。
- `anchor_pass_ray_fail=0`，没有任何样本能证明“ray fail 但 stereo anchor 可以救回来”。
- `ray_pass_anchor_fail=36`，反而存在 ray 通过但 anchor 失败的样本。
- 在最关心的 `parallax_reject` 桶中，`anchor_valid=10/995≈0.010`、`anchor_pass_ray_fail=0`。

当前结论：

- `selective stereo-anchor creation` 不应进入真实行为；当前数据明确显示覆盖率低、质量差、没有救援样本。
- `KF2 patch/refine` 也不应作为主线：`reproj_reject` 总体约 `4%`，即使完全解决也不能解释主要创建失败。
- `CreateNewMapPoints()` 的 neighbor KF pair selection 可以作为小范围 hygiene，但不应预期它根治 RMSE，因为会在 weak parallax 与更远 KF reprojection 风险之间做 trade-off。
- 当前 close/mid LMT 问题更准确地表述为：two-view LMT 在 `5_10m` 的几何创建上先天受限，但系统又依赖这些点提供一部分 inlier 支撑；局部 gate / culling / anchor 替换都只能移动约束结构，不能稳定改变轨迹上限。

后续边界：

- 不再继续投入 `stereo-anchor replacement`。
- 不再围绕 `KF2-only reproj reject` 做 patch refine 主线。
- 不再继续 local-proj gate / cooldown / weight 线。
- 若继续以最低 RMSE 为目标，下一步应转向 segment-level pose error attribution：定位 RMSE 高的帧段中 inlier 空间/深度/source 结构、KF 插入节奏、reference reuse 与 ORB 对照的差异，而不是继续改单个 MapPoint 创建局部规则。

## 当前实验：ORB-shaped XFeat input A/B

前一版 `backend profile` 的结论：

- `mvXFeatMappingEligible` 只影响 `CreateNewMapPoints()` close/mid，且 `tracking_stereo_backend_ineligible_reject=0`，没有真正改变 ORB-SLAM 后端看到的输入形态。
- 它减少了 close/mid triangulation 创建和 local map 规模，但没有改善 `local_proj + local_mapping_triangulation` 误差或 close/mid reuse，反而削弱了 `mnMatchesInliers`。

当前代码状态：

- 默认恢复安全基线：`CreateNewMapPoints()` 不再按 `mvXFeatMappingEligible` 做真实 close/mid profile 过滤，`CreateNewKeyFrame()` 也不再因 backend profile ineligible 拒绝 close/mid `tracking_stereo` birth。
- `Frame` 仍生成 `mvXFeatMappingEligible / mvXFeatMappingProfileScore`，作为诊断和 A/B 输入。
- 新增显式 A/B 开关：`XFEAT_ORB_SHAPED_INPUT=1`。
- 开启该开关后，`Frame::BuildXFeatMappingProfile()` 会把当前帧 XFeat keypoint / descriptor / stereo depth / right coordinate / birth quality 数组 compact 到 ORB-like backend subset。
- 这样 tracking、mapping、KeyFrame、local projection 后端共同消费同一套被压缩后的 XFeat feature set，区别于之前的 mapping-only mask。

新增诊断字段：

- `[Frame][XFeatMappingProfile] compact_enabled`
- `[Frame][XFeatMappingProfile] compact_kpts`
- `[Frame][XFeatMappingProfile] compact_ratio`

验收边界：

- 不开 `XFEAT_ORB_SHAPED_INPUT` 时，行为应回到安全基线。
- 开启后，`compact_ratio` 应明显小于 1，且 `mnMatchesInliers` 不应低于安全基线 90%。
- 如果 compact 输入改善 RMSE 或结构指标，说明问题主要是 XFeat 输入形态与 ORB-SLAM 后端契约不匹配。
- 如果 compact 输入仍无收益或明显伤 tracking，说明仅靠 ORB-like feature shaping 不足，需要转向误差模型、尺度/噪声建模或更深的 learned frontend/backend 适配。

## 已移除实验：XFeat local-proj patch refinement shadow

背景结论：

- 多轮 local-proj hard gate / cooldown / soft weight / residual gate 都只能改善局部 outlier 指标，不能稳定降低 RMSE。
- `[XFeatMeasurementCalib]` 和 `[PoseOpt][ResidualCalib]` 指向一个更底层问题：一部分 XFeat `local_proj` 约束即使保留下来，像素 residual 仍偏大。
- 因此本轮不再删边或降权，而是验证“当前帧 local-proj 匹配的 2D 坐标是否能通过 patch alignment 修准”。

实验曾包含的代码变化：

- `KeyFrame` 运行时缓存 XFeat 左目 zero-mean anchor patch，不序列化。
- `MapPoint` 运行时保存一个 XFeat anchor patch，不序列化。
- `tracking_stereo` 和 `local_mapping_triangulation` 新建 MapPoint 时写入 anchor patch。
- `XFeatMatcher::SearchByProjection()` 成功 assignment 后，对 `local_proj + tracking_stereo/local_mapping_triangulation + 0_10m` 左目匹配做小窗口 patch refinement shadow。
- 本轮不改变 matcher assignment、不改变 keypoint 坐标、不改变 PoseOptimization edge measurement。

实验曾新增日志：

- `[TrackLocalMap][LocalProjPatchRefineError]`
  - `success_bin=success/failed/no_patch`
  - `shift_bin=small/mid/large`
  - `margin_bin=weak/mid/strong`
  - `source=tracking_stereo/local_mapping_triangulation`
  - `depth_bin=0_5/5_10/...`
  - 输出 `old_residual_px_avg / refined_residual_px_avg / residual_gain_avg`

验收重点：

- 如果 `success` 桶中 `residual_gain_avg > 0` 且 `refined_residual_px_avg` 明显低于 `old_residual_px_avg`，说明 local-proj 的主要问题包含 2D measurement localization，可进入下一轮真实 refined measurement。
- 如果 `success` 桶不降 residual，说明问题更可能在错误关联、MapPoint 3D 位置或 stereo/triangulation 深度，不应继续做 subpixel/patch refinement。
- 如果大量样本落在 `no_patch`，说明 anchor patch 覆盖不足，需要先分析 MapPoint 创建来源与 patch 缓存覆盖率。

结果与处理：

- `run_SePT01_localproj_patch_refine_shadow.log` 中 patch refine 覆盖正常，但整体不降残差：`success` 桶 old residual 约 `3.05px`，refined 后约 `3.28px`。
- `local_mapping_triangulation` 基本持平，`tracking_stereo` 明显变差；大 shift 样本更容易是脏匹配。
- 该方向没有证明“通过 patch alignment 修正当前 2D measurement”可以改善 local-proj 约束。
- 由于该实现还会在 KeyFrame / MapPoint 中缓存 patch，并在 projection assignment 后做局部 patch search，带来明显内存和实时性负担。
- 结论：该实验代码已移除，不应保留在安全基线中；后续继续依赖 `[XFeatMeasurementCalib]` / `[PoseOpt][ResidualCalib]` 分析 XFeat 测量模型，而不是 patch refinement。

### PoseOptimization residual calibration 诊断

背景结论：

- 现有 `[TrackLocalMap][SourceDepthError]` / `[SourceMPJointError]` 能定位脏源，但不能回答 XFeat edge 在 `PoseOptimization` 里的实际 chi2 / pixel residual 是否系统性偏大。
- `[LocalBA][SourceDepthError]` 说明部分 MapPoint 在 KeyFrame BA 中并不脏，但当前帧 pose optimization 仍可能因为 XFeat measurement noise / scale model / source mixture 不匹配而产生较大残差。

本轮修改：

- 仅在 `Optimizer::PoseOptimization()` 增加 shadow 诊断，不改变优化、权重、outlier 判定或 MapPoint 生命周期。
- 新增日志 `[PoseOpt][ResidualCalib]`，按 descriptor、match source、MapPoint source、depth bin、octave bin、obs type 聚合最终 edge residual。
- 输出 chi2 分布、像素残差分布和实际 `inv_sigma2`，用于判断 XFeat 是否需要单独 residual calibration / sigma model，而不是继续调 local-proj gate。

验收重点：

- 比较 `descriptor=XFeat` 与 ORB 或不同 `match_source/mp_source/depth_bin` 的 `chi2_p90 / residual_px_p90 / outlier_ratio`。
- 如果 XFeat close/mid 的 residual 明显高于 ORB-like expectation，但 `outlier_ratio` 不高，下一步应考虑 XFeat-specific measurement sigma / robust kernel，而不是继续筛点。
- 如果 residual 分布本身正常但 RMSE 仍差，说明问题更偏向约束空间结构或关键帧/地图组织，而不是单边残差模型。

### PoseOptimization close/mid local-proj residual_px 护栏（已回退为 shadow）

`XFEAT_PIXEL_SIGMA=1.5/2.0` A/B 的关键结果：

- 放大 sigma 会显著降低 `chi2/outlier_ratio`，但真实 `residual_px_p90` 和 `inlier_err_avg` 反而上升。
- `sigma=2.0` 让更多大像素残差 edge 变成 inlier，RMSE 偏到 `0.423857`，说明全局放松 measurement noise 不是正确方向。
- 问题不是“后端太严格”，而是部分 `local_proj` close/mid edge 真实像素残差过大，但会因为 octave/inv_sigma2 被过度容忍。

实验修改：

- 恢复默认 `XFEAT_PIXEL_SIGMA=1.0` 思路，不再全局放松权重。
- 曾在 `Optimizer::PoseOptimization()` 的 outlier 分类中加入一个窄范围 residual_px 护栏：
  - 只作用于 XFeat；
  - 只作用于 `match_source=local_proj`；
  - 只作用于 `mp_source=local_mapping_triangulation / tracking_stereo`；
  - 只作用于当前深度 `0 < depth <= 10m`；
  - 当 `sqrt(chi2 / inv_sigma2) > 4.0px` 时，即使 chi2 未超过原阈值，也作为 outlier。
- 不改 `Frame.cc` stereo、建图、local map、optimizer 权重或 MapPoint 生命周期。

实验结果：

- `residual_px_gate_candidate=2270`，`residual_px_gate_reject=191`，主要来自 `local_mapping_triangulation=167`。
- 保留下来的 `local_proj` inlier residual 变干净：
  - `local_proj + local_mapping_triangulation inlier_err_avg`: `1.933 -> 1.620`
  - `local_proj + tracking_stereo inlier_err_avg`: `1.606 -> 1.455`
- 但系统支撑变差：
  - `mnMatchesInliers`: `235.7 -> 229.3`
  - `outlier_after_opt`: `27.9 -> 36.0`
  - KF 数量: `562 -> 581`
  - `local_mps`: `5819 -> 6047`
- 该模式与前面 hard gate / cooldown / 0.8 gate 一致：局部 source 指标改善，但系统结构更紧张，不能作为最低 RMSE 的稳定方向。

当前代码状态：

- 真实 residual_px outlier 行为已回退。
- `residual_px_gate_*` 字段保留为 shadow 计数，只表示“如果启用该规则会命中多少”，不再改变 `mvbOutlier` 或 edge level。

当前诊断字段已改为 shadow 命名：

- `[PoseOptimization][Weight] residual_px_gate_shadow_threshold`
- `[PoseOptimization][Weight] residual_px_gate_shadow_candidate`
- `[PoseOptimization][Weight] residual_px_gate_shadow_high`
- `[PoseOptimization][Weight] residual_px_gate_shadow_reject`
- `[PoseOptimization][Weight] residual_px_gate_shadow_reject_lmt`
- `[PoseOptimization][Weight] residual_px_gate_shadow_reject_tracking_stereo`

如果未来重新评估，验收重点：

- `residual_px_gate_shadow_reject > 0`，但不能导致 `mnMatchesInliers` 下降超过约 10%。
- `[PoseOpt][ResidualCalib] local_proj close/mid residual_px_p90` 应下降。
- `[TrackLocalMap][SourceMPJointError] local_proj + local_mapping_triangulation/tracking_stereo` 的 `inlier_err_avg` 不应像 sigma A/B 那样升高。
- 若 RMSE 仍无收益，说明单帧 residual 护栏也不是突破口，应停止在 PoseOptimization 单边规则上继续叠加。

评估口径应统一为 SE3/evo 风格对齐。不要只看单次 RMSE；至少同时看 source metrics、segment structure、`mnMatchesInliers`、`local_mps / track_in_view`、`LOST / RECENTLY_LOST`。

## 最近结论与回退：停止 tracking/mapping tracklet 分流线

这几轮尝试过完整 tracking/mapping feature 分流的前两阶段：

- Stage 1：tracking-side XFeat close/mid tracklet shadow 诊断。
- Stage 2：confirmed tracklet -> `Frame/KeyFrame` mapping eligibility -> same-tracklet triangulation pair。
- ownership split：在 `CreateNewKeyFrame()` 中 reserve 一部分 confirmed mapping-tracklet keypoint，避免它们先被即时 `tracking_stereo` MapPoint 占用。
- tracklet-guided pair 失败归因：统计 same-tracklet pair 从加入候选到创建 MapPoint 的流失点。

关键日志结论：

- Stage 1 成立但不是解法：`confirmed_alive≈153/frame`、`confirmed_cells≈127/frame`，说明 XFeat close/mid 中确实存在跨帧 repeatable candidates。
- Stage 2 技术上打通：reservation 生效，same-tracklet pair 能被加入 `CreateNewMapPoints()` 候选。
- 但 same-tracklet pair 不能稳定通过几何建图：`run_SePT01_21.log` 中 `mapping_tracklet_pair_added=116`，只有 `geometry_pass=6`、`created=5`。
- 主要失败原因是几何重投影，而不是 budget：`mapping_tracklet_pair_reproj_reject=75`、`stereo_fallback_reject=28`、`budget_reject=0`。
- SE3/evo 口径下该轮 `RMSE≈0.421807`，没有优于安全基线，也没有改善 `local_proj + local_mapping_triangulation` 残留污染。

经验总结：

- `tracklet identity` 只能说明 descriptor/局部观测有 repeatability，不能等价于可三角化、可建图的几何约束。
- 如果 repeatable pair 大量死在 reprojection，继续做 delayed promotion / covis influence control 没有意义，因为可 promotion 的高质量 MapPoint 数量太少。
- reservation 会牺牲即时 `tracking_stereo` 支撑，但没有换来足够长期稳定 MapPoint，性价比差。
- 因此这条线不再作为主线推进：Stage 1/Stage 2 相关代码和日志已从当前代码中回退，当前只保留本文档中的历史结论。

后续边界：

- 不再以 confirmed tracklet 作为独立建图准入条件。
- 如果未来重新使用跨帧 repeatability，只能作为辅助特征，并且必须叠加明确的 KeyFrame-pair 几何预验证，例如 reprojection/parallax/scale 预检。
- 当前下一步不应进入 Stage 3 delayed promotion 或 Stage 4 covis/local-map influence control。

## 已验证方向

### 1. Stereo front-end 视差估计线

做过的主要改动：

- 从单个 `bestIdxR` 到 top-k descriptor candidates。
- patch-level local search。
- top-k 引导 band search。
- 小型一维 cost volume。
- cost normalization 与 peak-sigma 建模。
- coarse-to-fine row-band + 更强代价面。

主要结论：

- 单层 `band + local cost volume + current-layer refinement + LR-aware sigma` 是目前最稳版本之一。
- `coarse-to-fine row-band + 更强代价面` 曾导致明显轨迹退化，RMSE 到过约 `1.11`，不应继续沿该实现形态扩展。
- cost/peak/sigma 小步改动能改变 close reject 分布，但没有稳定压低轨迹 RMSE。

后续边界：

- 不再扩大搜索层级、搜索范围或重做 stereo cost volume，除非有新的跨序列证据证明 accepted close/mid disparity 本身仍是第一主因。
- 如果未来重启 stereo 线，应以离线 stereo depth 误差评估为前置条件，而不是直接看 SLAM RMSE。

### 2. `tracking_stereo` lifecycle 和 local-proj 使用线

做过的主要改动：

- found-ratio + depth gate。
- `LongWeakMP` culling。
- `CreateNewKeyFrame()` close low-conf birth filter。
- provisional / delayed promotion。
- `SearchLocalPoints()` 中 close/mid `tracking_stereo` weak admission。
- local projection history gate / cooldown。

主要结论：

- 能降低一部分 `local_proj + tracking_stereo` outlier，但通常会减少有效约束支撑。
- 完整 provisional lifecycle 降低实时性，收益不稳定。
- cooldown 可以让 source metrics 变干净，但 RMSE 仍随机波动。

后续边界：

- 不再做新的 `tracking_stereo` lifecycle/gate，除非日志证明它重新成为数量和误差双主导的 source。
- `tracking_stereo` 相关改动优先保持为保守清理，不作为主攻方向。

### 3. `CreateNewMapPoints()` close/mid triangulation 入口控质线

做过的主要改动：

- close/mid stereo fallback 直接拒绝。
- close/mid true ray triangulation 收紧 parallax、reprojection、scale consistency。
- `5_10m` depth-aware parallax 微调。
- `[LocalMapping][TriangCreate]` creation-depth 诊断。

主要结论：

- 入口控质是少数相对合理、应保留的行为改动。
- depth-aware parallax 微调在 `0.9992 / 0.9993` 等参数上没有稳定收益，属于随机波动。
- 继续调 parallax/reproj/scale 阈值收益很低。

后续边界：

- 保留统一 close/mid 入口控质。
- 不继续做 `5_10m` 或 `0_5m` 参数特化，除非新的质量桶与后续误差出现强关联。

### 4. Optimizer 权重、BA、refit 和进图后补救线

做过的主要改动：

- `PoseOptimization` close/mid `local_mapping_triangulation` 弱权重。
- `MapPointCulling()` close/mid recent confirmation。
- triangulation refit shadow diagnostic。
- Local BA source/depth diagnostic。

主要结论：

- PoseOptimization 弱权重命中很多观测，但 RMSE 曾升到约 `0.435`，说明软化影响不等于提升几何质量。
- recent confirmation 是“点进图后补救”，没有精准压低 `local_proj + local_mapping_triangulation` close/mid 误差。
- Local BA 诊断显示：close/mid `local_mapping_triangulation` 在 KeyFrame BA 中 chi2/outlier 并不明显脏；问题更多发生在当前帧 tracking/local projection 阶段。

后续边界：

- 不再通过 Optimizer 权重或 MapPointCulling 来主导解决 RMSE。
- 如果未来做 BA 后清理，必须先证明某类点在 Local BA 中也是高 chi2 / 高 outlier。

### 5. LightGlue triangulation / local verification 线

做过的主要改动：

- `XFEAT_USE_LIGHTGLUE_TRIANG=1` triangulation A/B。
- local map LightGlue verification。
- 针对 `map_updated_edge` 的 LightGlue 二次确认。

主要结论：

- LightGlue triangulation 没有稳定压低 RMSE。
- LightGlue same-MP verification 不能证明该约束对当前 pose 是干净的；`local_proj_lg_verified + local_mapping_triangulation` 反而表现很脏。
- 定向 LightGlue verification 曾使 RMSE 到约 `0.422`，不作为默认行为保留。

后续边界：

- 不再把 LightGlue local verification 转成行为规则。
- LightGlue 可以作为 motion path 或离线对照，但不是当前 local_proj 污染的直接解法。

### 6. `SearchByProjection()` matcher gate / fallback 线

做过的主要改动：

- matcher-quality shadow diagnostic。
- `proj_norm > 1.0` outside gate。
- `proj_norm > 0.8` hard gate。
- projection-edge fine shadow。
- reference stable close/mid fallback shadow：扩大 radius、放宽 level。
- 临时 `XFEAT_PROJ_DISABLE_LEVEL_GATE=1` 对照。

主要结论：

- `proj_norm > 1.0` 是安全护栏，保留。
- `0.80-1.00` outlier ratio 很高，但 hard gate 后 RMSE 两次都在约 `0.424`，说明误杀有用约束或改变支撑结构。
- 放宽 radius/level 能增加候选，但 strict pass 仍接近 0，说明“简单扩大投影窗口”不是稳定解。
- reference stable close/mid MapPoint 的复用失败已经被拆解过：很多不是阈值太紧，而是没有有效可重复候选或候选本身不能通过严格匹配。

后续边界：

- 不再扩大 `proj_norm` gate。
- 不再把 fallback shadow 转成真实匹配，除非 strict pass 明显上升且 post-opt inlier 率高。

### 7. Pose prior / projection context 线

做过的主要改动：

- `[TrackLocalMap][LocalProjPoseContextError]`。
- `[TrackLocalMap][PosePriorShift]`。
- `[TrackLocalMap][ProjectionPriorShiftError]`。

主要结论：

- `map_updated_edge` 是强污染上下文，但不是由 PoseOptimization 前后大幅位姿移动导致。
- 很多坏样本落在 prior shift small，属于 `prior_bad_post_bad`，不是“优化后能救回来”的临时边缘匹配。

后续边界：

- 不再把 pose prior shift 当作主因。
- 如果重启该方向，应关注当前帧 local map 组织和可复用 MP 形成，而不是再做 projection-edge gate。

### 8. Local map / local KF 组织线

做过的主要改动：

- voted KF budget。
- hard visibility budget。
- hybrid core KF 保留。
- core-first / expansion-fallback local projection。
- current-pose visibility score 排序。
- contribution-aware local KF selection shadow。
- weak-vote / provenance / LocalKFCloseContribution 诊断。

主要结论：

- 已经不是纯 ORB-style 共视 vote；当前代码已有 current-pose visibility score。
- 更强 contribution-aware selection 曾减少 local map，但重复运行中损伤轨迹精度，因此现在行为关闭，只保留 shadow。
- local map 大不是唯一问题；更准确是 local map 几何相关性差，close/mid 中大量 MapPoint 当前为 negative depth 或 outside image。

后续边界：

- 不再做简单“缩小 local map / hard KF budget”。
- 如果继续 local map 方向，必须是当前几何贡献和稳定支撑同时满足的策略，且必须保护 `mnMatchesInliers`。

### 9. Mapping feature selection / tracking-mapping 分流线

做过的主要改动：

- mapping feature budget。
- cell/depth 分组。
- quality-first candidate sorting。
- keypoint-unique。
- response / octave / stereo-quality / triang-quality 记录。
- `[MappingFeatureReuseFunnel]` 和 `[MappingFeatureOutcome]`。

主要结论：

- XFeat 点很多，但多出来的点没有自然变成 ORB-like 稀疏稳定地图结构。
- `MappingFeatureReuseFunnel` 显示主要损失发生在 `selected -> in_frustum`，而不是最后 matcher 失败。
- `octave / response / grid` 可以解释部分误差，但不是足够强的单变量筛选信号。

后续边界：

- 不再按单一 response/octave/grid 阈值做 mapping budget。
- 如果继续 feature selection，应做更高层的 repeatability 学习或跨帧验证，而不是单帧质量打分。

### 10. KeyFrame insertion / close reuse 线

做过的主要改动：

- close reuse funnel。
- keyframe structure / insertion diagnostics。
- close current coverage 与 KF 插入原因分析。
- 讨论过条件化关键帧插入抑制。

主要结论：

- close MapPoint 当前有效跟踪覆盖偏低，系统持续认为需要补充近点。
- 直接抑制关键帧插入会减少补点机会，风险较高。
- 高频 KF 是现象也是放大器，但不是单独靠插入抑制能解决的根因。

后续边界：

- 不建议先做关键帧插入抑制。
- 应先提升已有 close/mid MapPoint 的稳定复用或控制进入地图/共视结构的质量。

### 11. Repeatability-aware covis vote 降权线

做过的主要改动：

- 在 `Tracking::UpdateLocalKeyFrames()` 的 MapPoint -> KeyFrame vote 阶段，对 `local_mapping_triangulation && close/mid(<=10m)` 的弱复用点做降权。
- 弱点判定包括 low observation、low found-ratio、local-proj opt unstable、local-map visibility long-low-relevance。
- weak close/mid LMT vote 不直接清零，而是按 `4:1` 折算，并保留最强 weak KF 兜底，避免 local map 断开。
- 新增 `repeatability_vote_*` 诊断到 `[SearchLocalPoints]` 和 `[TrackLocalMap][StableVoteShadow]`。

主要日志结论：

- `run_SePT01_14.log` 中改动已生效，`repeatability_vote_*` 字段非零。
- 命中量太小：平均每 debug 帧约 `target_points=26.2`、`weak_points=1.77`、`weak_raw=6.85`、`weak_effective=0.23`，实际只压掉约 `6.6` 个 weak vote。
- 没有改善 local map 结构：`local_mps` 从上一版约 `4612` 增到约 `4846`，`track_in_view/local_mps` 从约 `0.112` 降到约 `0.106`。
- `local_proj + local_mapping_triangulation` 没变干净：outlier ratio 从约 `0.1896` 变到约 `0.1961`，inlier error 从约 `1.89` 变到约 `1.91`。
- `mnMatchesInliers` 小降：约 `212.3 -> 209.1`。
- KF 插入节奏基本没变：最终约 `517 KFs`，上一版约 `515 KFs`。

后续边界：

- 这条线没有解决“XFeat 额外点没有转化成稳定、可复用、当前可投影地图约束”的核心问题。
- 不建议继续加大 covis vote 降权或继续调弱点阈值。
- 如需保留，只保留诊断字段；行为改动不应作为新的强基线，除非多次运行证明 segment structure 和 RMSE 同时改善。

### 12. KeyFrame covis graph 源头 vote 分流线

当前改动：

- 将第 11 节的 `UpdateLocalKeyFrames()` repeatability vote 降权行为改回 shadow-only，避免在当前帧 local KF 选择层重复补救。
- 在 `KeyFrame::UpdateConnections()` 的共视图构建阶段，对 `local_mapping_triangulation && close/mid(creation depth <= 10m)` 的 weak/new MapPoint 降低 covis graph vote。
- weak/new 判定包括 `obs <= 3`、`found_ratio < 0.35`、`age_kf <= 2`、long-low-relevance。
- weak vote 按 `4:1` 向下折算；如果没有任何 strong connection，则保留最强 weak KF 作为兜底。
- 新增 `[KeyFrame][CovisVoteSource] repeatability_vote_*` 行为诊断；`[SearchLocalPoints] repeatability_vote_*` 和 `[StableVoteShadow] repeatability_vote_*` 明确标记为 `repeatability_vote_behavior_active=0`。

预期验证：

- `covisvote_repeatability_vote_weak_suppressed > 0`。
- `covisvote_actual_kfs / actual_connected / actual_top_weight` 不应断崖。
- 后续 `UpdateLocalKeyFrames()` 的 `local_mps` 增长、`weak_vote_share`、`local_mps_lmt_0_10` 应下降或至少不再增长。
- `track_in_view / local_mps` 应上升，`mnMatchesInliers` 不应下降超过 10%。
- 若 RMSE 或 inlier 支撑明显恶化，回退该行为，只保留 covis graph 源头诊断。

本次日志结论：`run_SePT01_covis_repeatability_vote.log`

- 改动确实生效：`[KeyFrame][CovisVoteSource] repeatability_vote_behavior_active=1`，平均每个诊断 KF 约 `target_points=91.5`、`weak_points=67.0`、`weak_raw=147.7`、`weak_effective=32.0`、`weak_suppressed=115.6`。
- `Tracking::UpdateLocalKeyFrames()` 层已正确回到 shadow-only：`[SearchLocalPoints]` 和 `[StableVoteShadow]` 中 `repeatability_vote_behavior_active=0`。
- 但结构指标没有改善，反而更宽：最终 `524 KFs`，高于前两版约 `515/517 KFs`；`local_mps` 平均约 `4975`，高于 `run13` 的约 `4612` 和 `run14` 的约 `4846`。
- 当前帧可投影比例没有改善：`track_in_view/local_mps` 约 `0.104`，低于 `run13` 的约 `0.112`，与 `run14` 的约 `0.106` 同级偏低。
- close/mid LMT 地图负担继续增加：`local_mps_lmt_0_10` 约 `2579`，高于 `run13` 的约 `2368` 和 `run14` 的约 `2496`；`single_obs` 也升到约 `602`。
- `local_proj + local_mapping_triangulation` 没有稳定变干净：outlier ratio 约 `0.201`，差于 `run14` 的约 `0.196` 和 `run13` 的约 `0.190`；inlier error 约 `1.83` 有改善，但没有带来结构收敛。
- `mnMatchesInliers` 约 `213.6`，支撑没有塌，但 KF 插入仍非常频繁，created KF per 100 frames 约 `82.5`。
- close 特征复用仍低：`0_5m stable_feature_share` 约 `0.085`，`5_10m` 约 `0.231`，没有解决“近中距离 XFeat 特征不能稳定转成可复用 MapPoint”的核心问题。

决策边界：

- `KeyFrame::UpdateConnections()` 源头 covis vote 降权不是当前结构问题的有效主线；它能压弱 vote，但没有让 local map 更小、更相关或更可投影。
- 不建议继续加大 weak vote 抑制或继续调 `obs/found/age/relevance` 阈值。
- 若后续目标是最低 RMSE，应回退该真实行为或至少不要把它作为强基线；保留 `[KeyFrame][CovisVoteSource]` 作为诊断即可。

### 13. MapPoint probation / delayed promotion 线

当前改动：

- 将第 12 节的 `UpdateLocalKeyFrames()` repeatability vote 和 `KeyFrame::UpdateConnections()` repeatability vote 都收敛为 shadow-only，不再继续调 weak vote 阈值。
- 在 `CreateNewMapPoints()` 中，对 XFeat `local_mapping_triangulation` 的 `0_5m / 5_10m` 新点标记为 mapping probation。
- probation 点不是删除，也不是禁止 tracking 使用；第一版只限制它对 `KeyFrame::UpdateConnections()` 共视图 vote 的长期结构影响。
- probation 点满足固定确认条件后 promotion：
  - `Observations() >= 5`
  - `FoundRatio >= 0.50`
  - `age_kf >= 3`
  - 不能是长期 low relevance：local map visibility 中 negative-depth / outside-image 占比过高则不晋升。
- 若当前 KeyFrame 没有任何非 probation connection，则保留最强 probation connection 的 `1` 个兜底 vote，避免共视图断开。

新增诊断：

- `[LocalMapping][TriangCreate]` / `[LocalMapping][KFCreateStructure]`
  - `mapping_probation_created`
  - `mapping_probation_created_0_5`
  - `mapping_probation_created_5_10`
- `[KeyFrame][CovisVoteSource]`
  - `mapping_probation_target_points`
  - `mapping_probation_active_points`
  - `mapping_probation_promoted_points`
  - `mapping_probation_promoted_now_points`
  - `mapping_probation_vote_raw`
  - `mapping_probation_vote_effective`
  - `mapping_probation_vote_suppressed`
  - `mapping_probation_vote_kfs`
  - `mapping_probation_behavior_active=1`
- `tools/analyze_segment_structure.py` 已聚合 probation creation、promotion、vote suppression 相关字段。

验收边界：

- `mapping_probation_created > 0`，且 `mapping_probation_vote_suppressed > 0`，证明行为真正生效。
- `mapping_probation_promoted_points` 必须随时间出现；如果 active probation 只积累不晋升，说明规则过严，会导致长期地图断供。
- 结构目标是降低 `local_mps_lmt_0_10`、`single_obs`、weak/young LMT 对 covis graph 的影响，同时不降低 `track_in_view/local_mps`。
- `mnMatchesInliers` 不应低于安全基线 90%；KF 数量不应因为 probation 导致更频繁插入。
- 如果 local map 规模下降但 `track_in_view/local_mps` 不升，或 RMSE 多次高于 `0.42`，说明这条 probation 行为应回退，只保留诊断。

本次日志结论：`run_SePT01_15.log`

- 行为确实生效：`mapping_probation_created` 平均约 `81.2/KF`，其中 `0_5m≈67.4/KF`、`5_10m≈13.8/KF`；`mapping_probation_behavior_active=1`。
- promotion 也发生了，不是完全卡死：`mapping_probation_target_points≈90.9`，`active_points≈66.2`，`promoted_points≈24.6`，`promoted_now_points≈4.4`。
- probation 对 covis vote 的抑制非常强：`mapping_probation_vote_raw≈153.7`，`effective=0`，`suppressed≈153.7`，suppressed share 基本为 `1.0`。这说明第一版真实行为是“active probation 完全不进共视 vote”，只保留无强连接时的兜底。
- local map 规模相对上一版 covis vote 实验变小：`local_mps≈4708`，低于 `covis` 的约 `4975`；`local_mps_lmt_0_10≈2412`，低于 `covis` 的约 `2579`；`single_obs≈476`，低于 `covis` 的约 `602`。
- 但结构质量没有真正改善：`track_in_view/local_mps≈0.109`，只接近 `run13` 的约 `0.112`，仍然没有显著突破；`local_track_in_view≈506` 基本没上升。
- 约束质量变差：`local_proj + local_mapping_triangulation` outlier ratio 约 `0.210`，差于 `run13≈0.190`、`run14≈0.196`、`covis≈0.201`；inlier error 约 `1.88`，回到安全基线附近，没有明显收益。
- 系统支撑也下降：`mnMatchesInliers≈207.1`，低于 `run13≈212.3`、`run14≈209.1`、`covis≈213.6`；`assoc_added_by_local≈110.2`，没有增加有效 local projection 支撑。
- KF 插入没有被抑制：最终约 `516 KFs`，created KF per 100 frames 约 `83.2`，仍然非常频繁。
- close 特征稳定复用未改善：`0_5m stable_feature_share≈0.082`，`5_10m≈0.225`，与安全基线同级，没有解决 close/mid XFeat 特征不能稳定转成可复用 MapPoint 的核心问题。

决策边界：

- 第一版 probation 能减少 local map 和 LMT close/mid 负担，但没有把减少的点转化为更高可投影比例或更干净约束。
- 当前实现不应作为强基线保留；如果继续该方向，必须改成更软的 probation，而不是 active probation 完全屏蔽 covis vote。
- 不建议继续简单加严 promotion 或继续压 vote；这会更像“砍结构”，而不是形成稳定 MapPoint。
- 若以最低 RMSE 为目标，下一步应回退或软化该真实行为，并把重点放在“probation 点如何经过多帧确认形成稳定长期点”，而不是继续 suppress covis graph。

后续代码调整：soft probation 版本

- 将 active probation 的 covis vote 从完全屏蔽改为软权重：`effective = raw / 4`，只有没有任何非 probation connection 时才使用最强 probation KF 的兜底 `1` vote。
- promotion 条件改成复用历史驱动，而不是只看 `obs/found/age`：
  - `Observations() >= 4`
  - `FoundRatio >= 0.35`
  - `local_proj_attempt >= 2`
  - `local_proj_match >= 1`
  - `local_proj_opt_inlier >= 1`
  - overall opt outlier rate 和 recent outlier rate 不能过高
  - 仍不能是长期 low relevance 点。
- 这版不新增 culling，不删除 MapPoint；目标是验证“保留短期 tracking 支撑，同时降低未验证点对长期共视结构的强影响”是否比 hard probation 更稳定。
- 下一次验收重点：
  - `mapping_probation_vote_effective > 0`，suppressed share 不应继续接近 `1.0`。
  - `mapping_probation_promoted_points` 仍应持续出现。
  - `local_mps_lmt_0_10 / single_obs` 不应回到未分流膨胀状态。
  - `track_in_view/local_mps`、`mnMatchesInliers`、`local_proj + local_mapping_triangulation` 不应差于 hard probation 和安全基线。

本次日志结论：`run_SePT01_16.log`

- soft probation 行为生效：`mapping_probation_created≈80.4/KF`，其中 `0_5m≈65.8/KF`、`5_10m≈14.5/KF`。
- promotion 仍然发生：`mapping_probation_target_points≈89.0`，`active_points≈63.5`，`promoted_points≈25.5`，`promoted_now≈5.5`。
- vote 不再完全切断：`mapping_probation_vote_raw≈145.5`，`effective≈31.8`，`suppressed≈113.7`；effective share 约 `0.219`，suppressed share 约 `0.781`。这比 hard probation 的 `effective=0 / suppressed=1.0` 更温和。
- KF 插入略有改善：最终 `509 KFs`，低于 hard probation 的 `516` 和安全基线附近的 `515`；created KF per 100 frames 约 `81.3`，与基线同级略低。
- 但 local map 又膨胀：`local_mps≈4927`，高于 hard probation 的 `≈4708` 和安全基线 `≈4612`；`local_mps_lmt_0_10≈2542`，也高于 hard probation `≈2412` 和安全基线 `≈2368`；`single_obs≈566`，仍明显高于安全基线 `≈429`。
- 当前帧可投影比例未改善：`track_in_view/local_mps≈0.105`，低于安全基线 `≈0.112`，也低于 hard probation `≈0.109`。
- 约束质量没有改善：`local_proj + local_mapping_triangulation` outlier ratio 约 `0.219`，差于 hard probation `≈0.210` 和安全基线 `≈0.190`；inlier error 约 `1.86` 虽略低，但不足以抵消 outlier 增加。
- 系统支撑未明显提升：`mnMatchesInliers≈209.0`，高于 hard probation `≈207.1`，但低于安全基线 `≈212.3`；`outlier_ratio_after_opt≈0.112`，高于安全基线 `≈0.104`。
- close 特征稳定复用仍未改善：`0_5m stable_feature_share≈0.081`，`5_10m≈0.233`，基本仍在旧浮动带。

决策边界：

- soft probation 解决了 hard probation “完全断 covis vote”的问题，但又让 local map 负担回升，且没有改善最终 tracking 约束质量。
- 当前证据不支持继续在 probation vote 权重上调参；`raw/4` 仍不能把 XFeat close/mid 点转成更稳定、可复用、当前可投影的约束。
- 如果继续 mapping feature 分流，下一步不能再只是调整 covis vote；必须引入明确的 probation expiration/culling 或更早的建图候选准入，否则 active/probation 点会继续积累并重新拉宽 local map。
- 若目标是最低 RMSE，建议把 hard/soft probation 行为都视为未验证成功，不作为默认强基线；可保留状态与诊断，用于后续更完整生命周期实验。

后续代码调整：probation lifecycle confirmation / expiration 版本

- 不继续调 `raw/4` covis vote 权重；当前 soft vote 行为暂时保留作为对照，主变量改为 recent 阶段的生命周期确认。
- `KeyFrame::UpdateConnections()` 不再调用 `TryPromoteMappingProbation()`，避免共视图更新阶段产生 promotion 副作用；promotion 统一放到 `LocalMapping::MapPointCulling()`。
- `TryPromoteMappingProbation()` 增加 `age_kf >= 3` 条件，避免刚创建后仅凭偶然一次复用过早转正。
- `MapPointCulling()` 对 active close/mid `local_mapping_triangulation` probation 点走专门规则：
  - 满足复用历史 promotion 条件则 `mapping_probation_promoted` 并离开 recent 列表。
  - 若 opt outlier history 或 recent outlier history 明显脏，则 `mapping_probation_culled_outlier`。
  - 若 `age_kf >= 4` 仍没有任何 local-proj opt inlier，则 `mapping_probation_culled_no_inlier`。
  - 若 `age_kf >= 6` 仍未转正，则 `mapping_probation_culled_expired`。
  - 未到确认/剔除条件前保留为 `mapping_probation_kept`。
- `[LocalMapping][RecentMP]` / `[LocalMapping][RecentMPSource]` 增加：
  - `mapping_probation_active`
  - `mapping_probation_kept`
  - `mapping_probation_promoted`
  - `mapping_probation_culled_no_inlier`
  - `mapping_probation_culled_expired`
  - `mapping_probation_culled_outlier`
- `tools/analyze_segment_structure.py` 已加入这些字段和 basic share 聚合。

下一次验收重点：

- `mapping_probation_promoted > 0` 且 `mapping_probation_culled_* > 0`，证明 lifecycle 真正闭环。
- `mapping_probation_active` 不应持续累积；`single_obs`、`local_mps_lmt_0_10`、`local_mps` 增长应低于 soft probation。
- `track_in_view/local_mps` 不应继续低于安全基线；`mnMatchesInliers` 不低于安全基线 90%。
- 若 probation culling 明显降低 local map 负担但 `local_proj + local_mapping_triangulation` 或 RMSE 仍不改善，说明“进图后生命周期补救”仍不足，应停止这条线，转向更早的跨帧 repeatability 建图准入。

本次日志结论：`run_SePT01_17.log`

- 统一 SE3/evo 口径：本次轨迹 `RMSE=0.413948`；同一脚本下 ORB 对照 `RMSE=0.393934`。本次回到 `0.41` 左右随机带，但仍没有缩小与 ORB 的结构性差距。
- lifecycle 行为已真正生效：`mapping_probation_created≈77.9/KF`，其中 `0_5m≈63.4/KF`、`5_10m≈14.5/KF`。
- `MapPointCulling()` 中 probation 形成闭环：平均 `mapping_probation_active≈248.8`、`kept≈190.3`、`promoted≈13.2`、`culled_no_inlier≈34.5`、`culled_expired≈10.3`、`culled_outlier≈0.6`。这说明主要剔除的是“没有后续 local-proj inlier 证据”的点，而不是已经表现为高 outlier 的点。
- 相比 soft probation `run_SePT01_16.log`，结构明显回收：
  - `local_mps≈4927 -> 4685`
  - `local_mps_lmt_0_10≈2542 -> 2367`
  - `weak_vote≈1886 -> 1661`
  - `track_in_view/local_mps≈0.103 -> 0.112`
  - `mnMatchesInliers≈209.0 -> 212.8`
- 但相比安全基线 `run_SePT01_13.log`，仍不是确定收益：
  - `local_mps≈4685` 仍略高于安全基线 `≈4612`
  - `single_obs≈504` 仍高于安全基线 `≈429`
  - `local_proj + local_mapping_triangulation` outlier ratio `≈0.205`，好于 soft probation `≈0.219`，但差于安全基线 `≈0.190`
  - `mnMatchesInliers≈212.8` 与安全基线 `≈212.3` 同级，没有突破。
- close reuse 仍未解决：`0_5m` 全部 close feature 平均 `features≈74.9`、`no_mp≈65.4`、`associated≈9.5`、`stable_feature_share≈0.081`；`5_10m` 平均 `features≈15.9`、`no_mp≈11.7`、`associated≈4.2`、`stable_feature_share≈0.210`。大量 close feature 仍没有形成可复用 MapPoint。
- segment 诊断仍指向结构问题：`700-799` 段 `local_mps≈6809`、`track_in_view_share≈0.077`、`local_proj+LMT outlier≈0.285`，说明最差段仍是 local map 很大但当前帧可投影比例很低。

决策边界：

- probation lifecycle culling 比 soft probation vote 更合理，能抑制 active/probation 点持续堆积，并把结构拉回安全基线附近。
- 但它仍属于“点进图后补救”：能降低 local map 膨胀，不能显著提高 close feature 到稳定 inlier MapPoint 的转化率。
- 不建议继续大幅调 `age_kf/no_inlier/expired` 阈值；这会重新变成 culling 参数线。
- 如果保留此版本，应把它视为结构护栏而非 RMSE 主收益来源；下一条主线仍应前移到“创建前/多帧 repeatability 证据驱动的建图准入”，否则 close feature 的 `no_mp` 和 worst segment 的低 `track_in_view/local_mps` 不会被根本解决。

后续代码调整：pre-triangulation repeatability eligibility 版本

- 不再继续调单帧 `response/octave/grid` 阈值，也不继续调 probation culling 阈值。
- 在 `CreateNewMapPoints()` 的 XFeat close/mid ray triangulation pending candidate 阶段，按当前 KF keypoint `idx1` 聚合来自多个 neighbor KF 的支持：
  - 同一 `idx1` 若被 `>=2` 个不同 neighbor KF 形成有效 triangulation candidate，视为 repeatability-supported。
  - repeatability-supported candidate 在排序中优先，并获得轻微 `qualityScore` 加成。
  - 单 neighbor 支持的 candidate 不再无差别进入建图，只作为 fallback。
- 单支持 fallback 固定限额：
  - 全局 `45/KF`
  - 每个 `grid cell x depth group` 最多 `1`
  - repeatability-supported candidate 仍走原有 cell/depth/quality budget。
- 新增/扩展 `[LocalMapping][TriangCreate]` 和 `[LocalMapping][KFCreateStructure]` 字段：
  - `mapping_feature_repeatability_candidate`
  - `mapping_feature_repeatability_repeatable_candidate`
  - `mapping_feature_repeatability_single_support_candidate`
  - `mapping_feature_repeatability_accept`
  - `mapping_feature_repeatability_fallback_accept`
  - `mapping_feature_repeatability_reject_single_support`
- `tools/analyze_segment_structure.py` 已聚合 repeatability candidate / fallback / reject share。

下一次验收重点：

- `mapping_feature_repeatability_repeatable_candidate > 0`，确认存在多邻接 KF 支持的建图候选。
- `mapping_feature_repeatability_reject_single_support > 0`，确认单支持 fallback 真的限制了弱建图输入。
- `close_created`、`mapping_probation_created` 可下降，但不能断崖；`mnMatchesInliers` 不低于安全基线 90%。
- 若 `single_obs`、`local_mps_lmt_0_10`、`weak_vote` 下降，同时 `track_in_view/local_mps` 上升，说明 pre-map repeatability eligibility 比进图后补救更接近目标。
- 若 RMSE 或 `mnMatchesInliers` 明显恶化，说明多邻接支持过严或 fallback cap 过低，应回退此行为，只保留 repeatability 诊断。

本次日志结论：`run_SePT01_18.log`

- 统一 SE3/evo 口径：本次轨迹 `RMSE=0.420374`，同一脚本下 ORB 对照 `RMSE=0.393934`。本次没有延续 `run_SePT01_17.log` 的 `0.413948`，回到偏高随机带。
- repeatability eligibility 行为已生效：
  - `mapping_feature_repeatability_candidate≈145.1/KF`
  - `repeatable_candidate≈31.8/KF`
  - `single_support_candidate≈113.2/KF`
  - `accept≈48.1/KF`
  - `fallback_accept≈18.7/KF`
  - `reject_single_support≈77.9/KF`
  说明绝大多数 close/mid 建图候选仍是单邻接 KF 支持；本轮确实把大量单支持候选挡在了进图前。
- close/mid LMT 创建量被明显压低：`mapping_probation_created≈48.1/KF`，其中 `0_5m≈37.2/KF`、`5_10m≈10.9/KF`；相比 `run_SePT01_17.log` 的 `≈77.9/KF` 明显下降。
- probation 活跃负担也下降：`mapping_probation_active≈135.2`，低于 `run_SePT01_17.log` 的 `≈248.8`；`local_mps_lmt_0_10≈1926`，低于 `run_SePT01_17.log` 的 `≈2367`。
- 但减少建图输入没有转化为更好的当前帧约束：
  - `mnMatchesInliers≈204.2`，低于 `run_SePT01_17.log` 的 `≈212.8`
  - `assoc_added_by_local≈105.3`，低于 `run_SePT01_17.log` 的 `≈112.6`
  - `track_in_view/local_mps≈0.110`，与 `run_SePT01_17.log` 的 `≈0.112` 基本同级，未上升
  - `local_proj + local_mapping_triangulation` outlier ratio `≈0.210`，略差于 `run_SePT01_17.log` 的 `≈0.205`
- close reuse 核心指标基本未变：
  - `0_5m features≈75.0`，`no_mp≈65.8`，`associated≈9.2`，`stable_feature_share≈0.081`
  - `5_10m features≈15.9`，`no_mp≈11.7`，`associated≈4.2`，`stable_feature_share≈0.213`
  这说明被保留下来的 repeatability-supported 建图输入并没有显著提升 close feature 到稳定 MapPoint 的转化率。
- KF 插入副作用明显：最终 `541 KFs`，高于 `run_SePT01_17.log` 的约 `512 KFs`；`need_insert_close=345` 次，`xfeat_kf_guard_allow_close_insert=358` 次，平均 `frames_since_previous_kf≈1.49`。系统仍然认为 close 支撑不足，并继续通过插关键帧补点。
- segment 对照仍指向原结构问题：
  - `0-99` 段 ours-vs-ORB mean ATE 差 `≈0.101`
  - `700-799` 段 `local_mps≈6291`、`track_in_view_share≈0.076`、`lmt_outlier≈0.260`
  - top-error frames 仍伴随 `local_mps` 和 `from_voted` 明显偏高。

决策边界：

- 这轮证明“多邻接 KF 支持”作为 hard eligibility 太强：它能减少 close/mid LMT 创建和 probation 活跃负担，但没有提高可复用比例，反而降低 inlier 支撑并诱发更多 close-driven KF insertion。
- 这不是有效的 RMSE 主线；当前实现不应作为默认行为继续叠加。若保留，应只保留 repeatability 诊断，或把该规则改成更软的 ranking/priority，而不是 hard reject single-support。
- 这轮进一步确认：单纯减少建图点数不能解决“XFeat 特征多但不能形成稳定、当前可投影约束”的核心问题。关键缺口仍是稳定 MapPoint 的形成质量，而不是局部地图点数量本身。
- 下一步不应继续收紧 repeatability/fallback cap；更合理的是回退该 hard eligibility 行为，保留诊断，并重新评估是否需要真正的 tracking/mapping feature 分流：先让候选进入“短期/非共视强影响”的验证池，通过多帧复用后再进入长期地图，而不是在 CreateNewMapPoints 前按单次多邻接支持直接砍掉。

后续代码调整：probation-first / soft repeatability 版本

- 回退 `run_SePT01_18.log` 中的 hard single-support eligibility 行为：
  - `mapping_feature_repeatability_reject_single_support` 保留为真实拒绝计数，但当前不再因为 single-support cap 实际拒绝。
  - 新增 `mapping_feature_repeatability_shadow_reject_single_support`，用于观察旧规则本来会拒绝多少单支持候选。
  - repeatability 仍作为排序/质量加分，避免完全丢掉诊断信号。
- active probation 点继续允许创建和短期存在，但在当前帧 local KF vote 中降权：
  - `UpdateLocalKeyFrames()` 对 `active mapping probation` 的 direct vote 使用 `raw/4`，只有没有其他 keyframe vote 时保留最强 probation KF 的 `1` vote 兜底。
  - promoted probation 点恢复正常 vote。
  - active probation 不再计入 stable vote shadow，也不计入 `NeedNewKeyFrame()` 的 stable inlier support。
- 新增/扩展诊断字段：
  - `[SearchLocalPoints] mapping_probation_vote_*`
  - `[TrackLocalMap][StableVoteShadow] mapping_probation_vote_*`
  - `[LocalMapping][TriangCreate] mapping_feature_repeatability_shadow_reject_single_support`
  - `tools/analyze_segment_structure.py` 已聚合 search/stablevote probation vote 和 repeatability shadow reject share。

下一次验收重点：

- `mapping_feature_repeatability_shadow_reject_single_support > 0` 且 `mapping_feature_repeatability_reject_single_support≈0`，证明 hard reject 已回退为 shadow。
- `mapping_probation_vote_raw > 0`、`mapping_probation_vote_effective > 0`、`mapping_probation_vote_suppressed > 0`，证明 active probation 对 local KF selection 低影响但未完全断供。
- `close_created / mapping_probation_created` 应回到 `run_17` 附近，而不是保持 `run_18` 的过低创建量。
- `local_mps_lmt_0_10` 可以略高于 `run_18`，但 `track_in_view/local_mps`、`mnMatchesInliers` 不应低于 `run_17`。
- 如果 KF 数量仍显著高于 `run_17`，或 RMSE 仍持续高于 `0.42`，说明“低影响 probation”仍不能解决稳定 MapPoint 形成，需要停止该线并转向更底层的 descriptor repeatability / temporal association 机制。

本次日志结论：`run_SePT01_19.log`

- 统一 SE3/evo 口径：本次实际轨迹来自根目录 `f_SePT01_stereo.txt`，已转换为 `CameraTrajectory_SePT01_stereo_run19.tum`；`RMSE=0.412982`。同一脚本下 ORB 对照仍为 `RMSE=0.393934`。注意 `CameraTrajectory_SePT01_stereo.tum` 是旧文件，会得到上一轮 `0.420374`，不能用于本次结论。
- soft repeatability 行为按预期生效：
  - `mapping_feature_repeatability_reject_single_support≈0`
  - `mapping_feature_repeatability_shadow_reject_single_support≈70.3/KF`
  - `mapping_feature_repeatability_candidate≈119.8/KF`
  - `repeatable_candidate≈25.6/KF`
  - `single_support_candidate≈94.2/KF`
  - `accept≈76.4/KF`
  说明 hard single-support reject 已回退为 shadow；旧 hard 规则本来仍会挡掉大量候选。
- close/mid 创建量恢复到 `run_17` 附近：`mapping_probation_created≈76.4/KF`，其中 `0_5m≈60.7/KF`、`5_10m≈15.7/KF`。这比 `run_18` 的 `≈48.1/KF` 明显恢复，避免了 hard eligibility 造成的 inlier starvation。
- active probation 对 local KF vote 的降权也生效：
  - `SearchLocalPoints / StableVoteShadow` 中 `mapping_probation_vote_raw≈59.2`、`effective≈10.8`、`suppressed≈48.5`
  - `KeyFrame][CovisVoteSource]` 中 `raw≈184.2`、`effective≈40.7`、`suppressed≈143.5`
  说明 active probation 没有完全断供，但对当前帧 local KF selection 和 covis graph 的影响被显著压低。
- 系统支撑相比 `run_18` 恢复：
  - `mnMatchesInliers≈212.6`，高于 `run_18` 的 `≈204.2`
  - `assoc_added_by_local≈112.9`，高于 `run_18` 的 `≈105.3`
  - `outlier_ratio_after_opt≈0.111`，没有异常恶化
- 但结构问题没有解决，且局部地图重新膨胀：
  - `local_mps≈4953`
  - `local_mps_lmt_0_10≈2493`
  - `weak_vote≈1851`
  - `single_obs≈580`
  - `track_in_view/local_mps≈0.107`
  这些指标比 `run_18` 更大，也差于 `run_17` 的 `track_in_view/local_mps≈0.112`。
- `local_proj + local_mapping_triangulation` 仍是主脏源之一：per-frame total 约 `88.0`，outlier ratio `≈0.224`，inlier error `≈1.89`；比 `run_17` 的 `≈0.205` 更脏。`local_proj + tracking_stereo` 也仍脏，outlier ratio `≈0.234`。
- close feature 到稳定 MapPoint 的转化率仍未改善：
  - `0_5m features≈75.0`，`no_mp≈65.3`，`associated≈9.8`，`stable_feature_share≈0.084`
  - `5_10m features≈16.0`，`no_mp≈11.7`，`associated≈4.3`，`stable_feature_share≈0.221`
  这与前几轮基本同级，说明 soft repeatability 没有解决“close feature 多但稳定可复用 MP 少”的核心问题。
- KF 插入仍偏高：最终 `539 KFs`；`need_insert_close` 总计 `334`，`xfeat_kf_guard_allow_close_insert` 总计 `338`。close 支撑不足仍持续触发插帧。
- segment 对照显示 RMSE 改善不是结构性解决：
  - `0-99` 仍是主要差距段：ours RMSE `0.387`、ORB `0.305`，diff mean `≈0.078`
  - `700-799` 虽比 `run_18` 轨迹段改善，但仍有 `local_mps≈7372`、`track_in_view_share≈0.078`、`lmt_outlier≈0.242`
  - top-error attribution 仍与大量 close reference visible、较高 `nRefMatches` 和后段 KF 累积相关。

决策边界：

- 这轮比 `run_18` 的 hard repeatability 更适合作为对照版本：它恢复了 close/mid 创建量和 inlier 支撑，避免了 hard reject 直接饿死建图输入。
- 但它没有解决结构主因：local map 继续变大，`track_in_view/local_mps` 仍低，`local_proj + local_mapping_triangulation` 仍脏，close feature 稳定复用比例没有提升。
- 不建议继续调 `mapping_probation_vote raw/4`、shadow cap 或 repeatability soft score。这些只能在“少建点”和“保留支撑”之间摆动，不能制造 ORB-like 稳定稀疏地图。
- 若保留本版本，应定位为安全性较好的结构护栏和诊断基线，而不是 RMSE 主收益来源。
- 下一步若目标是更低 RMSE，应停止在单点 eligibility / covis vote 权重上调参，转向更底层的长期稳定性机制：例如 temporal tracklet / multi-frame confirmation 后再进入长期 MapPoint 与 covis graph，或分析 XFeat keypoint 跨帧 repeatability 与 descriptor drift，确认哪些 tracking features 才适合作 mapping features。

后续代码调整：active probation verification pool 版本

- 不再继续调 single-support cap、repeatability soft score 或 covis vote downweight。
- 将 active close/mid `local_mapping_triangulation` probation 点从普通 `core / expansion` local projection 中剥离出来：
  - active probation 不再混入 core pool。
  - active probation 不再混入 expansion fallback pool。
  - active probation 单独进入受限的 verification projection pool。
- verification pool 固定限制：
  - 输入候选上限 `200/frame`
  - 新增匹配上限 `30/frame`
  - 不新增环境变量，避免继续变成参数线。
- verification pool 的目的不是补强当前 pose，而是给 active probation 点一个有限机会积累 local-proj match / opt-inlier history，用于后续 `MapPointCulling()` promotion。
- 若未进入 verification pool 或超出 cap：
  - 不 `IncreaseVisible()`
  - 不记录 local-proj attempt
  - 清理 `mbTrackInView / mbTrackInViewR`
  - 不删除 MapPoint，由 recent/probation lifecycle 后续处理。
- 新增/扩展诊断：
  - `[SearchLocalPoints] probation_pool_candidates`
  - `probation_pool_in_view`
  - `probation_pool_enabled`
  - `probation_pool_input_capped`
  - `probation_pool_suppressed`
  - `probation_pool_matches`
  - `probation_pool_match_cap_hit`
  - `[TrackLocalMap][LocalProjPoolError] match_pool=probation`
  - `tools/analyze_segment_structure.py` 和 `tools/compare_trajectory_against_orb.py` 已聚合 `search_probation_pool_*` / `pool_probation_*`。

这轮与前面 probation 的区别：

- 前面主要是“已进长期地图后的 covis/local KF vote 降权和 recent culling”。
- 这轮开始改变 active probation 在当前帧 projection 使用层的地位：未转正前不再作为普通 local-proj 候选，而是只通过受限 verification pool 参与。
- 它仍不是完整 tracklet-before-MapPoint 方案；MapPoint 已经被创建。但相比之前，它更接近“未验证点先不要成为普通当前帧约束”的 delayed mapping 行为。

下一次验收重点：

- `probation_pool_candidates > 0` 且 `probation_pool_matches > 0`，证明 active probation 仍有验证机会。
- `probation_pool_matches` 不应长期接近 cap `30`，否则 active probation 仍在过度参与 pose。
- `pool_probation_weighted_outlier_ratio` 应明显高于/低于 core 可被单独观察；若很脏但数量受控，说明分离有效。
- `mnMatchesInliers` 不应低于 run19 的 90%。
- `mapping_probation_promoted` 不能塌缩；如果 promotion 接近 0，说明 verification pool 太窄。
- `local_mps_lmt_0_10`、`weak_vote`、`single_obs` 应至少不继续比 run19 膨胀。
- 若 RMSE 仍随机但 `track_in_view/local_mps` 和 probation promotion/污染结构改善，可继续向真正 tracklet-before-MapPoint 演进；若 inlier 支撑下降或 promotion 塌缩，则回退 verification pool。

本次日志结论：`run_SePT01_20.log`

- 统一 SE3/evo 口径：本次实际轨迹来自根目录 `f_SePT01_stereo.txt`，已转换为 `CameraTrajectory_SePT01_stereo_run20.tum`；`RMSE=0.422672`，mean `0.380234`，p90 `0.623073`。同一脚本下 ORB 对照仍为 `RMSE=0.393934`。相比 `run_SePT01_19.log` 的 `RMSE=0.412982` 明显变差。
- active probation verification pool 行为已生效：
  - `probation_pool_candidates≈169.1/frame`
  - `probation_pool_in_view≈169.1/frame`
  - `probation_pool_matches≈27.6/frame`
  - `probation_pool_match_cap_hit=20/26 debug frames`
  - `probation_pool_input_capped≈3.9/frame`
  说明 active probation 点确实被从 core/expansion 分离出来，但 verification pool 几乎长期打满 `30/frame` 上限。
- 局部结构有改善，但不是 RMSE 改善：
  - `local_mps≈4748`，低于 run19 的 `≈4953`
  - `local_track_in_view≈584`
  - `track_in_view/local_mps≈0.123`，高于 run19 的 `≈0.107`
  - `mnMatchesInliers≈213.7`，与 run19 `≈212.6` 同级略高
  - `assoc_added_by_local≈116.1`，高于 run19 `≈112.9`
  这些说明 local map 可投影比例和 local projection 数量没有塌。
- 但 source 质量没有形成轨迹收益：
  - `local_proj + local_mapping_triangulation` outlier ratio `≈0.211`，略好于 run19 的 `≈0.224`
  - `local_proj + tracking_stereo` outlier ratio `≈0.269`，差于 run19 的 `≈0.234`
  - `motion_lg` 仍然很干净，`local_mapping_triangulation` outlier ratio `≈0.002`，`tracking_stereo≈0.0066`
  这再次说明：局部 source outlier 降低不等价于轨迹 RMSE 下降。
- close reuse 只有小幅变化：
  - `0_5m stable_feature_share≈0.090`，略高于 run19 的 `≈0.084`
  - `5_10m stable_feature_share≈0.239`，略高于 run19 的 `≈0.221`
  - 但 `0_5m no_mp≈65.3/frame`、`5_10m no_mp≈11.5/frame` 仍很高
  close feature 到稳定 MapPoint 的转化主问题没有被解决。
- probation lifecycle 没有因为 verification pool 明显变好：
  - `mapping_probation_active≈241.7`
  - `mapping_probation_promoted≈10.6`
  - `mapping_probation_culled_no_inlier≈34.6`
  - `mapping_probation_culled_expired≈9.1`
  run19 的 promoted 约 `13.2`，本轮反而更低；verification pool 没有提升 promotion。
- KF 插入仍没有改善：最终 `541 KFs`，`need_insert_close=326`，`xfeat_kf_guard_allow_close_insert=318`。close 支撑不足仍持续触发插帧。
- segment 对照变差明显：
  - `0-99` 段 ours RMSE `0.429`、ORB `0.305`，diff mean `≈0.115`，比 run19 的 `≈0.078` 更差。
  - `700-799` 段 ours RMSE `0.247`、ORB `0.160`，diff mean `≈0.102`，也比 run19 的 `≈0.046` 更差。
  - `700-799` 仍有 `local_mps≈7294`、`track_in_view_share≈0.077`、`lmt_outlier≈0.250`，worst segment 结构问题没有消失。
- 本次没有输出 `[TrackLocalMap][LocalProjPoolError] match_pool=probation`，因为该日志目前在 `XFEAT_DIAG_DEEP=1` 下才打印；本轮主要依据 `[SearchLocalPoints] probation_pool_*` 和 source/segment 诊断判断。

决策边界：

- 这轮证明“active probation 单独 verification pool”不是当前可保留的 RMSE 改进：它让局部结构指标看起来更好，但 RMSE 明显变差。
- 关键失败点是 verification pool 仍然贡献过多当前帧 pose 约束：`probation_pool_matches≈27.6/frame` 且多数帧打满 cap。它没有实现“只验证、不影响 pose”的目标。
- 不建议继续调 `probation match cap=30` 到另一个数值；这会重新回到 cap 参数线。若要保留该思路，应先改成真正不参与 PoseOptimization 的 validation-only tracklet，而不是通过普通 local projection 产生当前帧约束。
- 若目标是最低 RMSE，应回退本轮 active probation verification pool，回到 run19 或更早安全基线。后续若继续 delayed mapping，应前移到“非 MapPoint / 非 pose edge 的短期 tracklet 验证”，而不是把 active probation 作为受限 local-proj MapPoint 使用。

补充复盘：已有日志已经回答的信息

- `XFeatDiag][StructureSummary]` 已经说明问题不是“特征点不够”：
  - `all_valid=1000/frame`
  - `matched_valid≈258.5/frame`
  - `matched_to_all_ratio≈0.2585`
  - matched keypoints 比全体 keypoints 更集中：`all_top4≈0.396`，`matched_top4≈0.491`，`matched_cv≈1.817` 高于 `all_cv≈1.491`
  - matched keypoints 更偏低 octave：`all_octave_mean≈2.574`，`matched_octave_mean≈2.344`
  这说明当前不是简单缺特征，而是可进入匹配/地图复用的特征子集更集中、更偏尺度。
- `CloseFeatureReuseBridge` 已经直接回答 close feature 到稳定 MapPoint 的转化很弱：
  - `0_5m`：`features=47400`，`no_mp=41133`，`no_mp_share≈0.868`，`stable_feature_share≈0.090`
  - `5_10m`：`features=9949`，`no_mp=7242`，`no_mp_share≈0.728`，`stable_feature_share≈0.239`
  即大量 close/mid XFeat 观测没有变成当前帧可复用的稳定地图约束。
- `ReferenceReuseBridge` 已经说明即使是 reference 里的 predicted-visible MapPoint，当前帧重新关联也不高：
  - `0_5m`：`visible_share≈0.927`，但 `current_inlier_share≈0.364`
  - `5_10m`：`visible_share≈0.987`，但 `current_inlier_share≈0.465`
  所以瓶颈不是只有 local map 选择，也包括 reference/stable MP 的当前帧重新关联。
- `MappingFeatureReuseFunnel` 已经把 local map 低效拆出来：
  - `local_mps≈4748/frame`，`local_in_frustum≈584/frame`，`track_in_view_share≈0.123`
  - `creation_depth=0_5`：`selected=36487`，`in_frustum=4510`，`frustum_rate≈0.124`，`admitted_match_rate≈0.171`
  - `creation_depth=5_10`：`selected=12562`，`in_frustum=1842`，`frustum_rate≈0.147`，`admitted_match_rate≈0.203`
  主要流失先发生在 local map payload 到当前帧可投影，再发生在 admitted 到 projection match。
- `MappingFeatureOutcome` 显示有质量信号，但不是足够强的单变量：
  - `response=low` outlier ratio `≈0.311`，高于 `response=mid≈0.206`
  - `mapping_feature_bin=balanced` outlier ratio `≈0.239`
  - `coarse_scale≈0.236`，`edge_grid≈0.240`
  因此单靠 grid/octave/response 的硬预算很难稳定压 RMSE；这些信号更适合做长期 eligibility 的一部分，而不是单独 gate。
- `SourceMPJointError` 仍然确认污染集中在 local projection：
  - `local_proj + local_mapping_triangulation`：outlier ratio `≈0.211`，inlier err `≈1.843`
  - `local_proj + tracking_stereo`：outlier ratio `≈0.269`
  - `motion_lg + local_mapping_triangulation`：outlier ratio `≈0.002`
  - `motion_lg + tracking_stereo`：outlier ratio `≈0.007`
  motion 路径干净，local projection 复用路径是主要结构问题。
- `KFDecision` 已经说明频繁插 KF 的直接原因仍是 close coverage 不足：
  - `n_tracked_close≈11.5`
  - `n_non_tracked_close≈70.8`
  - `need_insert_close≈0.517/frame`
  - `xfeat_kf_guard_suppress≈0.125/frame`
  close 复用不足不是单靠 KF guard 可以解决；guard 只能减缓膨胀。

修正判断：

- 之前说“现有日志回答不了 close/mid MapPoint 为什么不能复用”不够准确。已有日志已经能回答到系统级链路：XFeat close features 很多，但大部分没有 MapPoint；进入 local map 的 close/mid LMT 大量不在当前视野；进入投影后 match rate 也低；最终污染主要来自 local projection，而不是 motion 或 Local BA。
- 仍缺的是更细的物理点级因果链：某个 close keypoint 为什么从建图候选到后续 frame 不能稳定重复检测/重关联。这需要跨帧 feature identity 或 tracklet-before-MapPoint，而不是继续在已有 MapPoint 上加 gate。
- 后续如果继续追求 RMSE，不应再重复单点 gate、cooldown、payload cap、KF guard 参数线。更合理的方向是回退对 RMSE 不利的 active probation 行为，并重新设计真正的短期 tracklet / delayed MapPoint 创建，使未验证的 close/mid XFeat 点不直接成为 pose edge 和 covis 结构的一部分。

代码收敛：最低 RMSE 优先的稳定基线

- 目标：先移除已经明确让 RMSE 退化的 active probation 行为，收敛到适合后续结构改造的稳定状态。
- 已禁用新建 close/mid `local_mapping_triangulation` 点的 active probation 标记：
  - `src/LocalMapping.cc` 中 `kEnableMappingProbationBehavior=false`
  - 因此 `mapping_probation_created` 预期回到 `0`
- 已禁用 active probation 对 covis graph 的真实 vote 降权：
  - `src/KeyFrame.cc` 中 `kEnableMappingProbationCovisVoteBehavior=false`
  - `[KeyFrame][CovisVoteSource] mapping_probation_behavior_active=0`
- 已禁用 active probation 的 local-proj verification pool：
  - `src/Tracking.cc` 中 `kEnableLocalProjProbationVerificationPool=false`
  - `[SearchLocalPoints] probation_pool_*` 预期为 `0`
- 保留的低风险行为/诊断：
  - `CreateNewMapPoints()` close/mid 入口控质
  - `local_proj` history gate
  - `proj_norm > 1.0` outside gate
  - KF decision 诊断和保守 KF guard
  - mapping feature / reuse / source / depth / segment 诊断
- 本次收敛不是最终结构解法，只是停止已验证无效的“进图后 probation 补救”。下一步若继续追求 RMSE，应在这个稳定状态上实现真正的 `tracklet-before-MapPoint`，即未验证 close/mid XFeat 点不创建 MapPoint、不参与 covis、不参与 pose，跨帧确认后再进入长期地图。

## 诊断分层：默认核心日志 vs deep shadow

整理原则：

- 默认日志只保留当前仍用于判断系统结构和 RMSE 方向的核心指标。
- 已经验证过但没有继续作为默认决策依据的 shadow 诊断，统一降级到 `XFEAT_DIAG_DEEP=1`。
- 诊断分层不改变 SLAM 行为，只减少默认日志噪声，避免后续在已验证无效的方向里反复兜圈。

默认核心诊断：

- `[ComputeStereoMatches]`
- `[DepthDiag][CreateKF]`
- `[LocalMapping][TriangCreate]`
- `[LocalMapping][RecentMPSource]`
- `[LocalMapping][MappingFeatureCreate]`
- `[SearchLocalPoints]`
- `[SearchLocalPoints][CloseReuseFrustumReason]`
- `[SearchLocalPoints][ReferenceAssociationFrustumReason]`
- `[SearchLocalPoints][MappingFeatureReuseFunnel]`
- `[SearchLocalPoints][LocalKFCloseContribution]`
- `[TrackLocalMap][MappingRepeatabilityCohort]`
- `[TrackLocalMap][SourceMPJointError]`
- `[TrackLocalMap][SourceDepthError]`
- `[TrackLocalMap][MappingFeatureOutcome]`
- `[TrackLocalMap][CloseReuseOutcome]`
- `[TrackLocalMap][CloseReuseContributionOutcome]`
- `[Tracking][CloseFeatureReuseBridge]`
- `[Tracking][ReferenceReuseBridge]`
- `[Tracking][KFDecision]`
- `[XFeatDiag][StructureSummary]`

Deep shadow 诊断，默认不再输出，需要额外开启 `XFEAT_DIAG_DEEP=1`：

- `[TrackLocalMap][TriangQualityError]`
- `[TrackLocalMap][TriangMultiViewError]`
- `[TrackLocalMap][TriangRefitShadowError]`
- `[TrackLocalMap][TriangLocalProjHistoryError]`
- `[TrackLocalMap][LocalProjMatcherQualityError]`
- `[TrackLocalMap][LocalProjMatcherQualityFineError]`
- `[TrackLocalMap][LocalProjPoseContextError]`
- `[TrackLocalMap][ProjectionPriorShiftError]`
- `[TrackLocalMap][LocalMapProvenanceError]`
- `[TrackLocalMap][LocalProjPriority]`
- `[TrackLocalMap][LocalProjPool]`
- `[TrackLocalMap][StableVoteShadow]`
- `[TrackLocalMap][LocalKFContributionShadow]`
- `[TrackLocalMap][LocalKFContributionShadowRetention]`
- `[LocalBA][SourceDepthError]`

近期保留字段：

- `[SearchLocalPoints] repeatability_vote_*`
- `[TrackLocalMap][StableVoteShadow] repeatability_vote_*`，仅 deep
- `[KeyFrame][CovisVoteSource] repeatability_vote_*`

本次整理后的代码状态：

- `StableVoteShadow / LocalKFContributionShadow / LocalKFContributionShadowRetention` 已降级为 `XFEAT_DIAG_DEEP=1`。
- `LocalProjMatcherQualityFineError` 已降级为 `XFEAT_DIAG_DEEP=1`。
- active mapping probation 行为仍保持关闭：`mapping_probation_created=0`、`mapping_probation_behavior_active=0`、`mapping_probation_vote_behavior_active=0`、`probation_pool_*≈0`。
- `CreateNewMapPoints()` 的 mapping feature budget/repeatability 创建行为本次没有回退；它是轨迹行为，不是日志整理。若后续要进一步收敛最低 RMSE 基线，应单独做一轮“mapping feature budget 行为开/关”A/B，不要和日志清理混在一起。

常用分析脚本：

- `tools/analyze_segment_structure.py`
- `tools/analyze_trajectory_log_attribution.py`
- `tools/compare_trajectory_against_orb.py`

## 当前较稳结论

- 当前系统不是特征点数量不足；XFeat 特征和匹配更多，但没有稳定转化成长期几何约束。
- ORB 点更少，但地图结构更稀疏、稳定、几何均衡。
- 当前 close/mid MapPoint 的主要复用损失往往发生在当前帧几何相关性阶段：大量 local map close/mid 点当前为 negative depth 或 outside image。
- SearchByProjection 内部失败链路已经拆过；简单扩大窗口、放宽 level 或加 matcher gate 都不是稳定解。
- “局部 source outlier ratio 下降”不等价于 “RMSE 下降”。多轮实验已经证明 gate/weight/cooldown 常常只改善局部统计。

## 以后避免重复的规则

- 不再只因为某个桶 outlier ratio 高就直接 hard gate；必须同时证明 `mnMatchesInliers`、空间覆盖、segment RMSE 不受损。
- 不再继续调 parallax/reproj/scale/age/obs/found-ratio 这类单阈值。
- 不再把 LightGlue verification 当作 local_proj 约束正确性的证明。
- 不再把“local map 变小”当作目标；目标应是 `track_in_view / local_mps` 上升且有效 inlier 结构稳定。
- 不再用单次 RMSE 判断方向；必须多次运行，并与 ORB segment 对照。

## 后续如果要继续推进

更值得做的方向不是重复 gate，而是系统结构级改造：

- 研究 XFeat MapPoint 的跨帧 repeatability，建立真正的多帧确认式 mapping feature selection。
- 控制新 MapPoint 对 covis graph 的早期影响，让低稳定度点不要快速拉宽 local map。
- 以 segment-level 结构指标对齐 ORB：keyframe density、stable MP ratio、track_in_view share、inlier spatial coverage。
- 如果要做行为改动，优先做小范围 shadow -> A/B -> 多次运行验证；不要一次叠加多个规则。

## 代码调整：pre-MapPoint pending tracklet 分流

目标：

- 这轮开始实现真正的 `tracking/mapping feature` 分流：close/mid XFeat `local_mapping_triangulation` 候选不再通过一次 KF-pair triangulation 后立刻成为长期 MapPoint。
- 候选先进入 `LocalMapping` 内部 pending tracklet；只有跨后续关键帧再次被 triangulation match 命中，并满足多 KF observation / 多次 update pass 条件后，才 promotion 为正式 MapPoint。
- 这和之前 active probation 的关键区别是：probation 是“已经创建 MapPoint 后再补救”，本轮是“未确认前不创建 MapPoint、不进入 Atlas、不参与 covis/local map/BA”。

实现要点：

- 只作用于 XFeat close/mid true ray triangulation；ORB、far 点、tracking stereo、初始化逻辑不改。
- 地图关键帧数 `<5` 时保持原立即创建逻辑，避免 early map 断供。
- `CreateNewMapPoints()` 现有 mapping feature budget/repeatability 排序仍作为 pending admission 选择器；通过预算的 close/mid 候选进入或更新 pending tracklet。
- promotion 固定条件：至少 `3` 个不同 KF observation、至少 `2` 次 CreateNewMapPoints pass 更新、best candidate 保留现有 close 几何质量。
- pending tracklet age 超过 `6` 个 KF 或冲突过多会过期；过期只丢弃 pending，不删除已有 MapPoint。
- 新增诊断：
  - `[LocalMapping][TriangCreate] mapping_tracklet_*`
  - `[LocalMapping][KFCreateStructure] mapping_tracklet_*`
  - `[LocalMapping][MappingTracklet]`

验收重点：

- `mapping_tracklet_created > 0`、`mapping_tracklet_updated > 0`、`mapping_tracklet_promoted > 0`。
- `close_created` 应下降，但 `mnMatchesInliers` 不应低于安全基线 90%。
- `local_mps` 增长速度、`single_obs`、weak close/mid LMT 占比应下降。
- `track_in_view / local_mps`、`CloseFeatureReuseBridge stable_feature_share`、`ReferenceReuseBridge current_inlier_share` 不能下降。
- 若两次 evo SE3 RMSE 都高于 `0.42` 且 inlier/close reuse 下降，应回退该行为。

结果与处理：

- `run_SePT01_pending_tracklet.log` 显示 pending tracklet 行为确实生效：`mapping_tracklet_created / updated / promoted` 均非零。
- 但它主要减少了 close/mid MapPoint 创建量，没有提升可复用约束结构：`mnMatchesInliers` 和 `assoc_added_by_local` 下降，KF 插入频率没有下降。
- 该实验说明“延迟创建 MapPoint”本身不足以解决 XFeat 额外特征无法稳定复用的问题。
- 代码已回到安全行为基线：`kEnableXFeatCloseMidPendingTracklet=false`。pending tracklet 相关代码和日志字段保留，便于后续若要做完整 tracklet 分流时复用。
- 同步关闭其它未稳定证明有效的真实行为护栏：KF guard 行为关闭，local-proj core/expansion 分池行为关闭；保留 shadow 诊断。

## 代码调整：tracking-side XFeat mapping tracklet shadow 诊断

目标：

- 开始完整 tracking/mapping feature 分流的第一阶段，但不改变 SLAM 行为。
- 在 `Tracking` 侧为 close/mid XFeat stereo features 建立运行时短期 tracklet，验证普通帧中是否存在足够多可跨帧重复观测的 mapping candidates。
- 这些 tracklet 不进入 Atlas、不创建 MapPoint、不参与 covis/local map/BA，也不影响 KF 插入。

实现要点：

- 仅作用于 XFeat stereo/RGBD close/mid，且要求 `StereoBirthQuality == close_high_conf`。
- 已经关联稳定 inlier MapPoint 的 feature 不进入 tracklet candidate，避免统计已解决约束。
- 使用当前优化后 pose，把旧 tracklet 的 world point 投到当前帧；再用投影距离、descriptor 距离、depth consistency 更新 tracklet。
- confirmed 条件为至少 `3` 次 observation 且 depth relative range 不超过固定阈值。
- 低频日志：
  - `[Tracking][XFeatMappingTracklet]`
  - 关键字段：`candidates / created / updated / expired / alive / confirmed_alive / confirmed_cells / match_desc_avg / match_proj_px_avg`。

验收重点：

- 如果 `confirmed_alive` 和 `confirmed_cells` 足够多，说明存在可用于后续真正 mapping eligibility 的稳定 XFeat close/mid tracklet。
- 如果 `created` 很多但 `updated / confirmed_alive` 很少，说明 XFeat close/mid feature 的跨帧 repeatability 不足，继续做 MapPoint 前分流也缺少基础。
- 本轮不看 RMSE 改善，因为行为不变；只确认 `mnMatchesInliers`、KF 数量、source metrics 与安全基线同级。

## 代码调整：confirmed tracklet -> mapping eligibility 阶段 2

目标：

- 将 tracking-side confirmed close/mid XFeat tracklet 转成 `Frame/KeyFrame` 运行时 mapping eligibility。
- `CreateNewMapPoints()` 不做 hard-only，只在 close/mid true ray triangulation 的候选排序中优先使用 eligible keypoint。
- 验证“可跨帧重复的 XFeat close/mid 观测”是否能更稳定地进入 local mapping triangulation，而不是继续让 fallback 候选主导。

实现要点：

- `Frame` 新增运行时字段：
  - `mvXFeatMappingEligible`
  - `mvXFeatMappingTrackletId`
  - `mvXFeatMappingQuality`
- `KeyFrame` 构造时从 `Frame` 拷贝上述字段；不序列化，避免改变地图文件兼容性。
- `Tracking::UpdateXFeatMappingTracklets()` 在 tracklet confirmed 后标记当前 keypoint eligibility，并写入 tracklet id / quality。
- 每个 tracklet 记录后续创建出的 KeyFrame observation；即使当时尚未 confirmed，也保存 `KeyFrame* / keypoint idx / quality`。
- tracklet 一旦 confirmed，会回填所有已记录 KeyFrame observation，补齐历史 KF 上的 `mvXFeatMappingEligible / TrackletId / Quality`。
- `CreateNewKeyFrame()` 在 KeyFrame 入队 LocalMapping 前登记当前帧 tracklet observations，避免 LocalMapping 看到的 KeyFrame 仍是纯 fallback。
- `LocalMapping::CreateNewMapPoints()` 对 close/mid mapping candidates 计算 eligibility tier：
  - same confirmed tracklet on both KFs
  - both KFs eligible
  - one KF eligible
  - fallback
- 现有 mapping feature budget/repeatability 结构保留；排序优先级改为 eligibility tier / eligibility quality 优先，然后才是 repeatability / quality score。
- fallback 不在本轮直接禁止，避免再次进入“砍点导致 RMSE 随机或退化”的 gate 路线。

新增诊断：

- `[LocalMapping][TriangCreate]`
  - `mapping_eligible_close_candidates`
  - `mapping_eligible_both_candidates`
  - `mapping_eligible_same_tracklet_candidates`
  - `mapping_eligible_one_candidates`
  - `mapping_eligible_fallback_candidates`
  - `mapping_eligible_created`
  - `mapping_eligible_both_created`
  - `mapping_eligible_same_tracklet_created`
  - `mapping_eligible_fallback_created`
  - `mapping_eligible_created_0_5`
  - `mapping_eligible_created_5_10`
- `[Tracking][XFeatMappingTracklet]`
  - `current_tracklet_tagged`
  - `current_eligible_marked`
  - `kf_observations`
  - `kf_eligible_observations`
  - `backfilled_from_confirmation`
- `[Tracking][XFeatMappingTrackletKF]`
  - `registered`
  - `marked_current_eligible`
  - `backfilled`

验收重点：

- `mapping_eligible_close_candidates > 0`，且 `mapping_eligible_created > 0`。
- `mapping_eligible_both_candidates` 和 `mapping_eligible_same_tracklet_candidates` 应从上一轮的 `0` 变为非零，否则说明跨 KeyFrame 回填仍未打通。
- `mapping_eligible_fallback_created` 允许存在，但不应长期主导 close/mid created。
- `close_created` 不能断崖；`mnMatchesInliers` 不应低于安全基线 90%。
- 后续重点看 `CloseFeatureReuseBridge stable_feature_share`、`ReferenceReuseBridge current_inlier_share`、`local_mps / track_in_view` 是否改善。

### 阶段 2 补全：confirmed tracklet 直接生成 triangulation pair

前两版阶段 2 的问题：

- 第一版只把 confirmed tracklet 写成 `Frame/KeyFrame` eligibility mask，并在 `CreateNewMapPoints()` 的候选排序中提高优先级。
- 第二版补了 KeyFrame observation 注册和历史 backfill，`kf_observations / kf_eligible_observations / backfilled` 已经有值。
- 但 `CreateNewMapPoints()` 的候选 pair 仍完全由 `SearchForTriangulation()` 的 BF/LightGlue matcher 生成；tracklet id 只作为“已有候选的排序属性”，没有参与“候选生成”。
- 因此日志里会出现 `mapping_eligible_one_candidates > 0`，但 `mapping_eligible_both_candidates=0`、`mapping_eligible_same_tracklet_candidates=0`，说明 confirmed tracklet 没有真正进入 KF-KF triangulation pair。

本轮补全内容：

- 在 `CreateNewMapPoints()` 内，对当前 KF 和每个 neighbor KF 收集 `mvXFeatMappingEligible && mvXFeatMappingTrackletId != 0` 的 left-image keypoint。
- 当前 KF 侧要求 keypoint 有 close/mid stereo depth，确保这条通路只服务 close/mid mapping split。
- 若两个 KF 上存在同一 confirmed tracklet id，直接把 `(idx_current, idx_neighbor)` 追加到 `vMatchedIndices`。
- 若该 pair 已由 BF/LightGlue 生成，则只计入 duplicate skip，不重复加入。
- tracklet-generated pair 不绕过任何后续检查：仍走 parallax、cheirality、reprojection、scale、close stereo fallback reject、mapping budget、repeatability 和 slot-free duplicate 检查。
- 新增诊断：
  - `mapping_tracklet_pair_current_eligible`
  - `mapping_tracklet_pair_neighbor_eligible`
  - `mapping_tracklet_pair_candidate`
  - `mapping_tracklet_pair_added`
  - `mapping_tracklet_pair_duplicate_skip`
  - `mapping_tracklet_pair_occupied_skip`
  - `mapping_tracklet_pair_created`

验收重点：

- `mapping_tracklet_pair_added > 0` 表示阶段 2 终于从“标注/重排”进入“候选生成”。
- `mapping_eligible_same_tracklet_candidates > 0` 表示 same-tracklet pair 通过了几何验证并进入 close/mid candidate。
- `mapping_eligible_same_tracklet_created > 0` 或 `mapping_tracklet_pair_created > 0` 表示 confirmed tracklet 实际创建了 MapPoint。
- 如果 `mapping_tracklet_pair_added > 0` 但 `same_tracklet_candidates/created` 仍为 0，则问题已经从“tracklet 没接入候选生成”转移到“confirmed tracklet pair 几何质量不足”。

### 阶段 2 补全：reservation / ownership split

`run_SePT01_mapping_tracklet_pair_stage2.log` 的关键结果：

- `mapping_tracklet_pair_current_eligible=0`
- `mapping_tracklet_pair_candidate=0`
- `mapping_tracklet_pair_added=0`
- `mapping_eligible_same_tracklet_candidates=0`
- `mapping_eligible_same_tracklet_created=0`

同时 tracking 侧仍然有大量 confirmed tracklet：

- `confirmed_alive avg≈153/frame`
- `confirmed_cells avg≈126/frame`
- `current_eligible_marked avg≈24.6/frame`
- `kf_eligible_observations avg≈515/frame`

结论：

- 阶段 2 不是缺 confirmed tracklet，而是 current KF 里的 confirmed close/mid tracklet keypoint 在进入 `LocalMapping::CreateNewMapPoints()` 前已经被 `CreateNewKeyFrame()` 创建/复用成 `tracking_stereo` MapPoint。
- `CreateNewMapPoints()` 只能使用 free keypoint slot；slot 被 `tracking_stereo` 抢占后，same-tracklet pair generator 没有 current side 输入。

本轮补充：

- 在 `Tracking::CreateNewKeyFrame()` 中新增 confirmed mapping-tracklet reservation。
- 仅作用于 XFeat stereo close/mid、`StereoBirthQuality == close_high_conf`、当前 keypoint 尚无 MapPoint、且 `mvXFeatMappingEligible && mvXFeatMappingTrackletId != 0` 的点。
- 每个 KF 最多 reserve `40` 个，每个 coarse grid cell 最多 reserve `1` 个，按 mapping quality 优先。
- 被 reserve 的 keypoint 不创建即时 `tracking_stereo` MapPoint，保留 KeyFrame slot 给 LocalMapping 的 same-tracklet triangulation pair。
- 不删除点、不改 LocalMapping 几何 gate、不影响 ORB/far/普通 tracking。

新增诊断：

- `[DepthDiag][CreateKF]`
  - `tracking_stereo_mapping_tracklet_reserve_candidate`
  - `tracking_stereo_mapping_tracklet_reserved`
  - `tracking_stereo_mapping_tracklet_reserve_cap_reject`
  - `tracking_stereo_mapping_tracklet_reserve_cell_reject`
  - `tracking_stereo_mapping_tracklet_reserve_invalid_cell_reject`

验收重点：

- `tracking_stereo_mapping_tracklet_reserved > 0`，说明 ownership split 生效。
- `mapping_tracklet_pair_current_eligible > 0`，说明 LocalMapping 终于看到了 free current-side confirmed tracklet keypoint。
- `mapping_tracklet_pair_added > 0`，说明 same-tracklet pair 进入候选生成。
- `mapping_eligible_same_tracklet_candidates/created > 0`，说明 pair 通过几何验证并创建 MapPoint。
- 如果 reserve 生效但 same-tracklet created 仍为 0，下一步才分析 confirmed tracklet pair 的 parallax/reprojection/scale 质量，而不是继续改标注/backfill。

### ORB-shaped XFeat 输入形态 A/B

背景结论：

- SuperPoint_SLAM / Rover-SLAM 更接近“深度特征先被压成 ORB-SLAM 后端能稳定消费的一套 ORB-like feature set”，而不是“tracking 全量点、mapping 局部 mask”。
- 前一版 `mvXFeatMappingEligible` 只影响 `CreateNewMapPoints()` close/mid，且 `CreateKF` 侧 `tracking_stereo_backend_ineligible_reject=0`，说明它主要是在后端局部少建点，没有真正改变 ORB-SLAM 后端看到的输入形态。
- 该版本减少了 close/mid triangulation 创建和 local map 规模，但没有改善 `local_proj + local_mapping_triangulation` 误差或 close/mid reuse，反而削弱了 `mnMatchesInliers`。

本轮修改：

- 默认恢复安全基线：
  - `CreateNewMapPoints()` 不再按 `mvXFeatMappingEligible` 做真实 close/mid profile 过滤。
  - `CreateNewKeyFrame()` 不再因 backend profile ineligible 拒绝 close/mid `tracking_stereo` birth。
  - 保留 `CreateNewMapPoints()` close/mid 入口控质、history gate、`proj_norm > 1.0` outside gate。
- 新增显式 A/B 开关：
  - `XFEAT_ORB_SHAPED_INPUT=1`
  - 开启后，`Frame::BuildXFeatMappingProfile()` 会把当前帧 XFeat keypoint / descriptor / stereo depth / right coordinate / birth quality 数组 compact 到 ORB-like backend subset。
  - 这样 tracking、mapping、KeyFrame、local projection 后端共同消费同一套被压缩后的 XFeat feature set，区别于之前的 mapping-only mask。
- 新增日志字段：
  - `[Frame][XFeatMappingProfile] compact_enabled`
  - `[Frame][XFeatMappingProfile] compact_kpts`
  - `[Frame][XFeatMappingProfile] compact_ratio`

验收重点：

- 不开 `XFEAT_ORB_SHAPED_INPUT` 时，行为应回到安全基线。
- 开启后，`compact_ratio` 应明显小于 1，且 `mnMatchesInliers` 不应低于安全基线 90%。
- 如果 compact 输入改善 RMSE 或结构指标，说明问题主要是 XFeat 输入形态与 ORB-SLAM 后端契约不匹配。
- 如果 compact 输入仍无收益或明显伤 tracking，说明仅靠 ORB-like feature shaping 不足，需要转向误差模型、尺度/噪声建模或更深的 learned frontend/backend 适配。

### XFeat 后端契约校准诊断

本轮修改目的：

- 不再新增 local-proj gate、mapping feature 分流、probation、KF 剪枝或 ORB-shaped profile 行为。
- 保持安全基线不动，只补 XFeat descriptor / ratio / projection residual / PoseOptimization chi2 的校准诊断。
- 用日志回答：XFeat 的测量质量是否和 ORB-SLAM3 后端误差模型匹配，后续是否应做 source/depth-aware sigma 或小范围 matcher 阈值校准。

代码变化：

- 新增 `[TrackLocalMap][XFeatMeasurementCalib]`，仅在 `XFEAT_DEBUG=1 && XFEAT_DIAG_FEATURE_DISTRIBUTION=1` 输出。
- 统计维度包括 `match_source / mp_source / depth_bin / descriptor_dist_bin / ratio_bin / proj_norm_bin / level_diff_bin / obs_type`。
- 每个桶输出 `total / inlier / outlier / outlier_ratio / residual_px_avg / residual_px_p50 / residual_px_p90 / chi2_avg / chi2_p90 / depth_avg`。
- 扩展 `[PoseOpt][ResidualCalib]`，增加 `descriptor_dist_avg/p90`、`ratio_avg/p90`、`proj_norm_avg/p90`。
- 将 `[PoseOptimization][Weight]` 中 residual gate 相关字段改名为 `residual_px_gate_shadow_*`，明确这些字段只是 shadow 统计，不是实际 reject。
- 新增 `tools/analyze_xfeat_measurement_calib.py`，用于聚合日志并输出 source/depth 残差概览和保守候选过滤桶。

运行与分析：

```bash
python3 tools/analyze_xfeat_measurement_calib.py \
  experiment_logs/Stereo/SePT/SePT01/run_SePT01_xfeat_measurement_calib.log
```

验收重点：

- 本轮行为应与安全基线同级，RMSE 不作为主要验收。
- `[XFeatMeasurementCalib]` 核心维度应有非零桶。
- 如果某些 descriptor / ratio / proj_norm 桶同时满足高 outlier ratio 且低 inlier 覆盖，下一轮才考虑转成真实规则。
- 如果各测量桶仍不能解释 RMSE 差距，应停止 matcher/gate 线，转向 XFeat keypoint localization uncertainty 或 stereo depth uncertainty 建模。

### XFeat `local_proj` 上下文感知测量权重

背景结论：

- `[TrackLocalMap][XFeatMeasurementCalib]` 显示 `local_proj` 明显比 `motion_lg` 脏，但 descriptor / ratio 不能稳定区分污染。
- `proj_norm=edge/outside` 和 `level_diff=off2p` 是最强信号；其中 `outside` 已作为安全 hard gate 保留，`edge` 是混合桶，不能整桶拒绝。
- 前面 `0.8 gate / cooldown / residual gate / global sigma` 都证明：硬删或全局放松会改善局部 outlier，但不稳定改善 RMSE。

本轮修改：

- 不新增 hard reject，不改 matcher、建图、stereo、local map 或关键帧策略。
- 在 `Optimizer::PoseOptimization()` 中对 XFeat `match_source=local_proj` 的高风险观测降低信息矩阵权重：
  - `0.5 < proj_norm <= 1.0` 使用 `0.70`。
  - `level_diff >= 2` 使用 `0.70`。
  - 两者同时满足使用 `0.55`。
  - 只作用于 `local_mapping_triangulation` 和 `tracking_stereo` 来源。
- 旧的 far triangulation pose weight 继续保留；若同时命中 far weight 和 context weight，取更小权重。

新增诊断：

- `[PoseOptimization][Weight]`
  - `weighted_local_proj_context`
  - `weighted_local_proj_context_edge`
  - `weighted_local_proj_context_level`
  - `weighted_local_proj_context_both`
  - `weighted_local_proj_context_lmt`
  - `weighted_local_proj_context_tracking_stereo`

验收重点：

- `weighted_local_proj_context > 0`，说明新权重实际命中。
- `mnMatchesInliers` 不应低于安全基线 90%。
- `local_proj + local_mapping_triangulation` 和 `local_proj + tracking_stereo` 的 `inlier_err_avg` 不应上升。
- 至少跑两次；若 RMSE 持续高于 `0.42` 或 inlier 支撑下降，应回退该权重行为。

结果与处理：

- 两次运行中，该权重确实命中目标边，且 source-level outlier ratio 明显下降。
- 但 `local_proj + local_mapping_triangulation` / `tracking_stereo` 的 inlier residual 没有同步下降，部分运行反而上升。
- `chi2_avg` 的下降主要来自信息矩阵变小，不代表像素测量更准。
- RMSE 仍不稳定，并出现 `0.426757` 级别退化。
- 因此该真实权重行为已回退，只保留 `[XFeatMeasurementCalib]` / `[PoseOpt][ResidualCalib]` 等校准诊断。

### Rover-style XFeat backend contract A/B

背景结论：

- SuperPoint_SLAM 走的是“把 learned feature 包装成 ORB-like pyramid/octalve/sigma”的路线。
- Rover-SLAM 更接近“learned feature 单尺度/弱尺度语义接入 ORB 后端”的路线。
- 当前 XFeat extractor 已经是真多尺度，但日志长期显示 `level_diff/off2p`、projection-edge 和 local-proj residual 是强污染信号；继续做 local-proj gate、cooldown、mapping budget 没有稳定降低 RMSE。

本轮修改目的：

- 不再改 extractor、stereo、CreateNewMapPoints、LocalMapping 或 KeyFrame 策略。
- 只验证一个更底层假设：XFeat 的 octave/scale 不应被 ORB-SLAM 后端当成 ORB octave hard contract。

代码变化：

- `XFeatMatcher::SearchByProjection()` 默认启用 Rover-style contract：
  - XFeat projection 默认关闭 octave hard level gate。
  - `predictedLevel` 仍记录到诊断与 `level_diff`，但不再限制候选层级。
  - projection radius 增加 `6px` 下限，避免 predicted scale 过小时搜索窗口过窄。
  - 可用 `XFEAT_DISABLE_ROVER_BACKEND=1` 回到旧 ORB-like XFeat 后端消费方式。
- `Optimizer` 的 XFeat visual edge sigma 改为固定像素 sigma：
  - Rover-style 默认 `XFEAT_PIXEL_SIGMA=1.5`。
  - 不再乘以 `mvScaleFactors[kp.octave]`。
  - ORB/binary descriptor 仍走原始 `mvInvLevelSigma2[kp.octave]`。
- 保留现有 `[TrackLocalMap][XFeatMeasurementCalib]`、`[PoseOpt][ResidualCalib]`、`level_diff/proj_norm` 诊断，用来判断该 backend contract 是否更匹配 XFeat。

验收重点：

- `[XFeatMatcher::SearchByProjection(MapPoints)] level_gate=0 rover_backend=1`。
- `ref_target_empty_area / ref_target_no_valid_candidate` 不应继续主导 reference stable close/mid 点复用失败。
- `level_diff=off2p` 的 outlier/residual 应下降，或至少不再明显主导污染。
- `local_proj + local_mapping_triangulation` 的 `inlier_err_avg` 应下降，而不只是 outlier ratio 下降。
- 若 RMSE 两次均高于 `0.42`，可用 `XFEAT_DISABLE_ROVER_BACKEND=1` 验证是否应回退该 backend contract。

### Close/Mid LMT Pair-Level Selection

背景结论：

- `[LocalMapping][TriangInputFailure]` 显示 close/mid `local_mapping_triangulation` 的主要创建前损失来自 parallax，不是单纯 descriptor 或 local-proj gate。
- Pair context 诊断显示：最高共视 neighbor KF 不一定提供最好的 close/mid triangulation 几何；`top1` 往往视差不足，`top3/top10` 可能有更好的 baseline/depth，但旧逻辑会按 neighbor 遍历顺序先创建 MapPoint，导致同一 current keypoint 的弱 pair 先占槽。
- 这轮不是再加 hard gate，也不是 mapping feature budget / tracklet / probation；它只改变 close/mid XFeat ray triangulation 的创建顺序：先收集通过现有几何检查的候选，再对同一 current KF keypoint 选择几何更好的 pair 创建。

代码变化：

- `LocalMapping::CreateNewMapPoints()` 新增 `kEnableXFeatCloseMidPairSelectionBehavior=true`。
- 仅作用于 `!bUseORB && bCloseRayCandidate`：
  - ORB 分支不变。
  - far triangulation 不变。
  - close/mid stereo fallback reject、parallax/reproj/scale 阈值不变。
  - `Frame.cc` stereo、`Optimizer`、`SearchLocalPoints()` 不变。
- close/mid XFeat ray candidate 通过现有几何检查后进入延迟候选池。
- 在 neighbor KF 遍历完成后，按 pair score 排序并对 current KF keypoint 去重：
  - score 偏向更强 baseline/depth、强 parallax、低 reproj chi2、scale margin、response 和低 octave。
  - 每个 current KF keypoint 最多创建一个 MapPoint。
  - 如果最高分 pair 的槽位已被占用，允许同 keypoint 的次优 pair 继续尝试。

新增诊断：

- `[LocalMapping][TriangCreate]`
  - `close_pair_select_active`
  - `close_pair_select_candidate`
  - `close_pair_select_unique_current`
  - `close_pair_select_created`
  - `close_pair_select_duplicate_reject`
  - `close_pair_select_best_top1`
  - `close_pair_select_best_top3`
  - `close_pair_select_best_top10`
  - `close_pair_select_best_late`
  - `close_pair_select_5_10_candidate`
  - `close_pair_select_5_10_created`

验收重点：

- `close_pair_select_candidate > 0` 且 `close_pair_select_created > 0`。
- `close_pair_select_created` 不应导致 `close_created` 断崖。
- `close_pair_select_best_top3/top10/late` 若明显非零，说明这轮确实绕开了“top1 covis 先占槽”的旧行为。
- `[TriangInputFailure]` 的 created 分布应更多落到较强 `baseline_depth_ratio` 桶，而不是只看 raw reject 总数。
- `local_proj + local_mapping_triangulation` 的 `inlier_err_avg / outlier_ratio` 不应恶化；`mnMatchesInliers` 不应低于安全基线 90%。
- 如果 RMSE 仍随机但 created pair 几何质量改善，保留该版本继续观察；如果 `mnMatchesInliers` 下降或 RMSE 持续高于 `0.42`，回退该行为。

运行结果：`run_SePT01_close_pair_selection.log`

生效确认：

- `close_pair_select_active=1`，本轮行为实际启用。
- 平均 `close_pair_select_candidate≈156/KF`、`close_pair_select_unique_current≈125/KF`、`close_pair_select_created≈124/KF`。
- 平均 `close_pair_select_duplicate_reject≈32/KF`，说明同一 current KF keypoint 确实存在多个 neighbor pair 候选，被择优去重。
- 最终创建来源不再主要是最高共视 `top1`：
  - `close_pair_select_best_top1=399`
  - `close_pair_select_best_top3=812`
  - `close_pair_select_best_top10=1145`
  - `close_pair_select_best_late=0`
- 这证明该修改确实绕开了“top1 covis 先占槽”的旧行为。

系统层统计：

- `mnMatchesInliers≈237.1`
- `assoc_added_by_local≈131.5`
- `local_mps≈5727`
- 全部 TrackLocalMap 诊断帧均为 `state=2 / ok_visual`，没有 `LOST / RECENTLY_LOST`。
- 与 pair-context 基线相比，支撑没有塌，inlier 数略高。

source-level 结果：

- `local_proj + local_mapping_triangulation`：
  - total `2706`
  - `outlier_ratio≈0.199`
  - `inlier_err_avg≈1.913`
- 对比：
  - `run_SePT01_triang_input_pair_context.log`: `outlier_ratio≈0.203`，`inlier_err_avg≈1.948`
  - `run_SePT01_stereo_anchor_shadow.log`: `outlier_ratio≈0.210`，`inlier_err_avg≈1.923`
- 因此本轮有轻微 source 指标改善，但量级很小，不足以单独解释稳定 RMSE 突破。

depth-level 结果：

- `local_proj 0_5m`: `outlier_ratio≈0.215`，`inlier_err_avg≈1.935`
- `local_proj 5_10m`: `outlier_ratio≈0.178`，`inlier_err_avg≈1.826`
- close/mid 的 local projection residual 仍偏高。

TriangInput 结论：

- `0_5m`: created rate `≈0.724`，parallax reject `≈0.174`
- `5_10m`: created rate `≈0.478`，parallax reject `≈0.433`
- `baseline_depth_ratio` 仍是强信号：
  - `low`: created rate `≈0.450`，parallax reject `≈0.517`
  - `mid`: created rate `≈0.552`，parallax reject `≈0.358`
  - `strong`: created rate `≈0.745`，parallax reject `≈0.145`

结论：

- 这轮修改是有效的 hygiene：它修正了 weak `top1` pair 先占槽的问题，且没有伤 tracking。
- 但它没有解决根因：`5_10m` close/mid LMT 的 two-view triangulation 仍受 parallax / baseline-depth trade-off 限制；后续 `local_proj + LMT` 仍有约 `20%` outlier 和接近 `2px` 的 inlier residual。
- 不建议继续围绕 pair score 微调。若保留该版本，应把它视为安全小修，而不是主线突破。
