# XFeat SLAM nLevels=1 后续工作流计划

> 状态说明（2026-07-09）：本文记录的是阶段性工作流。旧的
> `XFEAT_MOTION_HEALTH_GUARD`、`XFEAT_KF_INSERT_GUARD` 和 lifecycle
> expansion guard 已从代码删除；当前保留
> `XFEAT_LOCALMAP_TEMPORAL_EXPANSION_LAST`。连续 tracking base 统计已经完成，
> 当前正在验证 `XFEAT_LOCALMAP_PRIMARY_FIRST` 的自适应准入版本。最新实验数据、行为边界和验收条件见
> `docs/airslam_xfeat_slam_adaptation_direction.md`。

本文档用于指导 `nLevels=1 / octave=0` 之后的实验和代码修改。之前很多实验是在 `nLevels=8` 或仍带 ORB 金字塔尺度语义时完成的；这些结论不能全部丢弃，但需要按当前前端条件重新排序优先级。

## 总目标

当前 medium/difficult 场景的核心问题不应再理解成单一阈值问题，而是这条结构链：

```text
XFeat close/mid 点多
-> close/mid MapPoint 创建多
-> 共视关系变密
-> LocalMap MapPoint 候选池变大
-> projection matching 补进更多不稳定 3D-2D 约束
-> PoseOptimization 里 local_proj outlier 高
-> medium/difficult 场景精度不稳定
```

后续工作流的目标是：

- 保留 easy 场景中深度特征带来的观测优势。
- 在 medium/difficult 中减少不稳定 close/mid MapPoint 对 covis、LocalMap projection、PoseOptimization 的负面影响。
- 避免重新进入单点 gate、阈值反复调参、多个变量混跑导致无法归因。

## 当前固定边界

后续默认保持：

- `nLevels=1`
- `keypoint.octave=0`
- 不恢复 ORB 金字塔尺度语义。
- 不启用 `projection_first`。
- 不直接减少 `nFeatures`。
- 不全局禁用 close/mid 建图。
- 不直接训练 XFeat vocabulary，除非后续证明 ORBvoc 候选召回不足。
- 不同时打开多个新 guard 或结构改动。

当前可以作为功能基线的 LG 模块：

- `XFEAT_USE_LIGHTGLUE_MOTION=1`
- `XFEAT_USE_LIGHTGLUE_TRIANG=1`
- `XFEAT_USE_LIGHTGLUE_REF=1`
- `XFEAT_USE_LIGHTGLUE_RELOC=1`
- `XFEAT_USE_LIGHTGLUE_LOOP=1`

Loop 低负载配置建议作为默认 loop A/B 起点：

```text
XFEAT_LOOP_CANDIDATES=6
XFEAT_LOOP_LG_EXPAND=1
XFEAT_LOOP_LG_EXPAND_COVISIBLES=1
XFEAT_LOOP_LG_MAX_CALLS_PER_KF=12
```

## 旧结论如何处理

### 仍然可信

这些结论主要来自系统结构和日志归因，受 `nLevels=8` 干扰较小：

- motion LG 很干净，`local_proj` 是主要脏源。
- KF 频繁插入主要由 close point pressure 触发。
- XFeat/deep feature 会放大 MapPoint、covis graph、LocalMap payload。
- Loop 对 MH04 这类 difficult 场景精度影响很大。
- `projection_first` 在 MH04 中明显伤精度，不应作为默认策略。

### 需要在 nLevels=1 下重新验证

这些旧实验可能被金字塔尺度语义影响，不能直接作为最终否定：

- conservative KF guard
- LocalMap admission guard
- local-proj history/found-ratio/source-aware gate
- pose edge sigma / local_proj weighting
- MapPoint probation lifecycle
- pre-MapPoint tracklet / pending tracklet
- confirmed tracklet eligibility / reservation
- ORB-shaped input

### 不建议直接恢复

这些旧线成本高、变量多，不应直接整套恢复：

- 完整 pending tracklet + eligibility + reservation 线。
- active probation verification pool。
- covis vote 大幅降权。
- 多个 LocalMap pool 分层真实行为。

如果重启，必须先根据当前 `nLevels=1` 日志证明对应瓶颈仍然存在，再做局部版本。

## 实验卫生规则

每次只改一个变量：

```text
baseline
baseline + one change
```

必须分开记录：

- 是否出现 `[LoopLG][Accept]`
- 是否出现 `*Loop detected`
- 是否出现 `mpLoopCloser is running GBA`
- 是否 `mpLoopCloser is not finished`
- 是否 LOST / RECENTLY_LOST
- 最终 KF 数量
- RMSE
- GPU 负载和运行时间

不要把 no-loop run 和 loop/GBA run 直接比较 RMSE。需要按以下类别分组：

```text
no-loop
loop accepted but no GBA
loop accepted + GBA
loop closer not finished
tracking lost
```

每个候选配置至少跑 3 次，看 median，不只看单次最优。

## 阶段 0：建立 nLevels=1 结构基线

目标：

- 确认在当前 `nLevels=1 / octave=0` 下，问题链条是否仍然存在。
- 避免基于旧 `nLevels=8` 日志继续决策。

推荐命令：

```bash
XFEAT_DIAG_FEATURE_DISTRIBUTION=1 \
XFEAT_DIAG_NLEVELS1_BASELINE=1 \
XFEAT_DIAG_INTERVAL=30 \
XFEAT_DEBUG=1 \
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
  2>&1 | tee experiment_logs/Stereo/EuRoC/MH_04_difficult/run_EuRoC_MH_04_difficult_stereo_nlevels1_baseline.log
```

必看日志：

- `[Tracking][KFDecision]`
- `[Tracking][CloseFeatureReuseBridge]`
- `[Tracking][ReferenceReuseBridge]`
- `[SearchLocalPoints]`
- `[SearchLocalPoints][MappingFeatureReuseFunnel]`
- `[TrackLocalMap][SourceMPJointError]`
- `[TrackLocalMap][SourceDepthError]`
- `[LoopLG]`
- `[LoopLG][Accept]`

判断标准：

- 如果 `motion_lg` 仍明显干净，`local_proj` 仍明显更脏，则继续 LocalMap/projection 方向。
- 如果 `need_insert_close` 仍主要是 close-only，则 KF 插入问题仍来自 close pressure。
- 如果 close feature `no_mp_share` 仍很高，说明 close/mid 观测到稳定 MapPoint 的转化仍是瓶颈。
- 如果 `local_mps` 大但 `track_in_view/local_mps` 低，说明局部地图 payload 仍被放大。
- 如果 loop 状态在不同 run 中差异很大，先处理 loop 稳定性，不要急着比较前端小改动。

阶段 0 输出：

- 3 次 baseline 的 RMSE、loop 状态、KF 数、local_proj outlier、motion_lg outlier。
- 判断当前主要问题是否仍是：

```text
close/mid birth + covis/local map 膨胀 + local_proj 脏约束
```

## 阶段 1：Loop 稳定性和负载控制

目标：

- 把 loop 变量从 tracking/mapping 变量里分离。
- 降低 GPU 负载，确认低负载 loop LG 是否仍能接受正确 loop。

优先测试：

```text
XFEAT_LOOP_CANDIDATES=6
XFEAT_LOOP_LG_EXPAND_COVISIBLES=1
XFEAT_LOOP_LG_MAX_CALLS_PER_KF=12
```

对照测试：

```text
XFEAT_LOOP_CANDIDATES=10
XFEAT_LOOP_LG_EXPAND_COVISIBLES=3
XFEAT_LOOP_LG_MAX_CALLS_PER_KF=24
```

验收指标：

- `[LoopLG][Accept]` 是否稳定出现。
- `unique_initial / sim3_converged / proj_opt_matches / cov_valid` 是否足够。
- 是否出现 `mpLoopCloser is not finished`。
- GPU 占用是否可接受。
- loop/GBA 状态相同的 run 中，RMSE 是否稳定。

停止条件：

- 如果低负载配置仍能稳定 loop，则不继续扩大 candidate/window。
- 如果高负载只是增加 GPU 占用，没有提升 loop 接受率，则不继续扩。
- 如果 ORBvoc 经常没有召回正确候选，再规划 XFeat vocabulary，不提前训练。

## 阶段 2：更保守的 KF Insert Guard

目标：

- 不再全局减少 KF。
- 只抑制“高支撑、短间隔、纯 close pressure”的冗余插帧。

当前已有 `XFEAT_KF_INSERT_GUARD=1`，但上一版偏粗：

- 能减少 KF 数。
- 不能让 `local_proj` 变干净。
- 可能压掉有用近场锚点。

下一版建议条件：

```text
close_only_insertion=1
need_without_close=0
c2_no_close=0
mnMatchesInliers / nRefMatches >= 0.75
frames_since_last_kf <= 2 或 <= 3
novel close area 不明显
可选：LocalMapping queue/backlog 有压力时才抑制
```

新增或关注日志：

- `kf_guard_enabled`
- `need_without_close`
- `close_only_insertion`
- `close_guard_suppressed`
- `close_guard_reason`
- `c2_no_close`
- `frames_since_last_kf`
- `mnMatchesInliers / nRefMatches`

A/B 命令增量：

```bash
XFEAT_KF_INSERT_GUARD=1
```

验收指标：

- KF 数下降不能以 `mnMatchesInliers` 明显下降为代价。
- `local_proj` outlier ratio 不能上升。
- `track_in_view/local_mps` 不应下降。
- loop 状态必须和 baseline 分组一致再比较 RMSE。

停止条件：

- 如果 KF 数下降但 RMSE 不降、local_proj 更脏，则不继续调 KF guard 参数。
- 如果只在 no-loop/loop 状态不同的 run 中看到 RMSE 改善，不算 guard 有效。

## 阶段 3：LocalMap Projection Admission 重新设计

目标：

- 不删除 MapPoint。
- 不全局砍 local map。
- 在进入 projection matching 前，把明显高风险的 close/mid MapPoint 从当前帧 pose 约束里隔离。

不要直接恢复旧 hard gate。先做 shadow，再做轻量真实行为。

### 3.1 Shadow 版

只统计不改变行为：

- source：`tracking_stereo / local_mapping_triangulation / unknown`
- depth：`0_5 / 5_10 / 10p`
- obs：`Observations()`
- found：`GetFoundRatio()`
- local-proj history：opt outlier rate、recent outlier rate
- provenance：Reference / Voted / Covis / ChildParent / Temporal / Unknown
- current match source：motion/ref/local_proj

输出目标：

```text
如果只拒绝某一类 high-risk MP，理论上会少多少 projection candidates？
这些 high-risk MP 当前实际贡献了多少 inlier/outlier？
拒绝后可能损失多少 mnMatchesInliers？
```

判断标准：

- 如果 high-risk 桶 outlier 高，但 inlier 支撑也很高，不适合 hard gate。
- 如果 high-risk 桶 outlier 高且 inlier 贡献低，才进入轻量 gate。

### 3.2 轻量真实行为

只在以下条件之一开启：

- motion health weak/bad。
- LocalMap payload 明显过大。
- 当前帧已有足够 motion/ref 支撑。
- MapPoint 来自 close/mid 且历史明显差。

候选规则：

```text
close/mid source in {tracking_stereo, local_mapping_triangulation}
and provenance in {Temporal, Unknown, weak Covis}
and Observations < 3 or found_ratio < 0.35 or local_proj history bad
```

但必须保留 escape：

- reference/voted provenance 优先保留。
- 有 motion/ref LG 支撑过的点优先保留。
- 当前帧 inlier 支撑不足时不要继续拒绝。

验收指标：

- `local_proj` outlier ratio 下降。
- `mnMatchesInliers` 不低于 baseline 90%。
- `assoc_added_by_local` 不断崖。
- `track_in_view/local_mps` 上升或持平。
- RMSE median 改善，且 loop 状态一致。

停止条件：

- 如果只是 source outlier 下降，但 RMSE 和 inlier 支撑变差，则回退。
- 不继续调 found-ratio 单阈值，不回到旧 gate 参数线。

## 阶段 4：nLevels=1 下的 pose edge sigma / weighting 校准

目标：

- 重新校准 `nLevels=1 / octave=0` 条件下 XFeat local_proj edge 的权重。
- 避免之前 `nLevels=8` 下错误 octave/sigma 语义继续影响判断。

优先做诊断：

- `[PoseOpt][ResidualCalib]`
- `[TrackLocalMap][SourceMPJointError]`
- `[TrackLocalMap][SourceDepthError]`

关注：

- `local_proj + tracking_stereo`
- `local_proj + local_mapping_triangulation`
- `motion_lg + tracking_stereo`
- `motion_lg + local_mapping_triangulation`
- depth `0_5 / 5_10`

候选行为：

- 不先 hard reject。
- 对 high-risk local_proj close/mid edge 做小幅降权。
- 权重策略必须 source-aware，不影响 motion_lg。

验收指标：

- local_proj inlier residual 下降。
- outlier ratio 下降或持平。
- `mnMatchesInliers` 不断崖。
- RMSE median 改善。

停止条件：

- 如果只是 outlier ratio 下降但 residual/RMSE 不改善，不继续调权重。
- 如果 easy 场景变差，说明权重过强。

## 阶段 5：重新评估进图前分流

只有阶段 0 到 4 仍证明 close/mid birth 是主要瓶颈时，才重启。

不要直接恢复旧整套：

- pending tracklet
- confirmed tracklet eligibility
- same-tracklet pair
- reservation

旧结果已经说明：

```text
tracklet repeatability 存在，
但 same-tracklet pair 不一定有足够几何质量；
之前主要失败在 reprojection / stereo fallback / geometry pass，
不是没有 tracklet，也不是 budget 不够。
```

如果重启，应改变目标：

旧目标：

```text
用 tracklet 帮助创建更多稳定 MapPoint
```

新目标：

```text
用 tracklet 防止未验证 close/mid 点过早进入 covis/local map/pose
```

新的最小版本：

- 只做 tracking-side shadow tracklet。
- 不创建 MapPoint。
- 不进入 Atlas。
- 不参与 covis。
- 不参与 PoseOptimization。
- 只统计哪些 close/mid feature 有稳定跨帧 repeatability 和 depth consistency。

只有当 shadow 显示：

- confirmed tracklet 覆盖足够。
- confirmed tracklet 的几何 residual 明显好。
- 能解释当前 no_mp / local_proj outlier 问题。

才考虑真实行为。

停止条件：

- 如果 confirmed tracklet pair 仍主要 reprojection reject，不继续 tracklet 建图。
- 如果保留/分流导致 `mnMatchesInliers` 下降明显，不继续。

## 阶段 6：跨数据集验证

MH04 不能单独作为最终依据。每个候选改动至少验证：

- MH03
- MH04
- MH05
- V1_01 或 V1_02
- V2_01 或 V2_02

每个数据集记录：

- RMSE median
- loop/no-loop/GBA 状态
- final KFs
- created MapPoints
- `local_proj` outlier
- `motion_lg` outlier
- `need_insert_close`
- runtime/GPU

判断：

- 如果只提升 MH04 loop run，但伤 easy 或 V 系列，不作为默认。
- 如果 medium/difficult 提升且 easy 持平，才进入候选基线。
- 如果只降低地图规模但 RMSE 不提升，不作为主要路线。

## 推荐执行顺序

```text
0. 固定 nLevels=1 baseline，MH04 跑 3 次。
1. 分组 loop/no-loop/GBA，确认 RMSE 浮动来源。
2. 固定低负载 Loop LG，确认 GPU 和 loop 接受率。
3. 做 conservative KF guard 小改，跑 3 次。
4. 做 LocalMap admission shadow，不改行为。
5. 如果 shadow 支持，再做轻量 admission gate，跑 3 次。
6. 重新做 pose edge residual/weight 校准。
7. 只有 close/mid birth 仍是主瓶颈，才重启 tracklet-before-MapPoint，但先 shadow。
8. 最后跨 MH03/MH05/V1/V2 验证。
```

## 结果记录模板

每次实验在 log 或单独记录中保留：

```text
experiment_name:
commit/hash:
command:
dataset:
run_index:
rmse:
loop_accept:
loop_detected:
gba:
loop_closer_finished:
lost:
final_kfs:
created_mps:
local_mps_avg:
track_in_view_share:
mnMatchesInliers_avg:
assoc_added_by_local_avg:
motion_lg_outlier:
local_proj_outlier:
need_insert_close_rate:
close_only_insertion_rate:
gpu_usage:
notes:
decision: keep / retry / rollback / next_phase
```

## 当前最推荐的下一步

先不要恢复旧 tracklet 线，也不要回到全局 KF guard。当前日志已经把问题收敛到：

```text
normal 段：direct + strict continuous primary 基本够用
weak 段：primary support 不足，必须启用 secondary
已知风险：weak fallback 中 secondary/voted 的外点率偏高，但简单限量会损失必要支撑
```

因此当前主线是：

```text
阶段 0：固定 XFeat vocab + LG loop + temporal expansion-last baseline。
阶段 1：使用 adaptive primary-first 压掉 normal 段不必要 secondary。
阶段 2：保持前一版 adaptive primary-first，不继续做 secondary hard admission。
阶段 3：two-stage refined-pose projection 已验证无稳定收益并回退。
```

本阶段不再做：

- 新的 KF 插入策略。
- 全局少建点。
- expansion lifecycle 细分。
- 单阈值 risk guard。
- 大量 shadow/audit 诊断。

已验证 `weak fallback secondary admission` 能略降 local projection 外点率，但会减少有效支撑、
增加关键帧数，RMSE 没有改善，因此已回退。下一步如果继续改 LocalMap，应考虑更结构化的
二阶段 projection / pose verification，而不是继续增加 source/risk 阈值。

已回退的二阶段实验：

```text
direct + primary support >= 80
-> 临时 Frame 上 PoseOptimization
-> 临时优化 inliers >= 60 时，只回写 pose
-> secondary / expansion deferred candidates 用新 pose 重新投影
```

三次实验 RMSE 为 `0.077933 / 0.073675 / 0.098668`，但 1m RPE 和
`primary_support=80~139` 区间的 local projection 外点率均未优于 adaptive
primary-first。低 RMSE 不能从日志中归因于二阶段行为，主要仍受 loop/GBA 和运行状态影响。
实现、环境变量和专用日志字段已经删除。
