# 2026-07-09 工作日报

## 今日目标

围绕 MH04 difficult 上 XFeat + LightGlue 前端的精度波动继续做归因。重点不再继续堆单点 gate，而是验证：

- LocalMap 中哪些 MapPoint 约束真正干净、连续、值得优先使用。
- normal 段和 weak 段是否应该采用不同的 LocalMap projection 策略。
- 当前 RMSE 波动是否主要来自 LocalMap，还是仍受 loop / GBA 质量影响。
- 哪些实验逻辑应该保留，哪些应该回退，避免代码和日志继续膨胀。

## 固定基线

今天所有判断都建立在当前主线配置上：

```text
nLevels=1
keypoint.octave=0
XFeat vocabulary loop retrieval
LightGlue motion / ref / triang / loop
XFEAT_LOCALMAP_TEMPORAL_EXPANSION_LAST=1
```

其中 XFeat 词袋为：

```text
Vocabulary/xfeat/models/xfeat_float_euroc_all_k10_l5_f512_i150_r001.dbow3
```

## 主要实验结果

### 1. LocalMap 连续性诊断

MH04 连续运行三次：

| 日志 | RMSE | loop 次数 | KFs |
|---|---:|---:|---:|
| `run_MH04_continuity_diag.log` | 0.085130 | 3 | 569 |
| `run_MH04_continuity_diag_1.log` | 0.071928 | 2 | 518 |
| `run_MH04_continuity_diag_2.log` | 0.118919 | 1 | 520 |

201 个诊断采样帧的 LocalMap 质量排序：

| 来源 | assoc | outlier | outlier ratio |
|---|---:|---:|---:|
| direct motion/ref | 20391 | 206 | 1.01% |
| strict continuous | 2321 | 286 | 12.32% |
| strict non-continuous | 14439 | 2775 | 19.22% |
| voted | 2765 | 758 | 27.41% |
| expansion | 176 | 102 | 57.95% |

结论：

- direct motion/ref 最干净，说明 LightGlue motion/ref 本身不是主要脏源。
- strict continuous 明显比 strict non-continuous 和 voted 干净，是有效质量信号。
- expansion 最脏，但最终贡献很小，继续作为 fallback 合理。
- strict non-continuous / voted 虽然更脏，但仍提供大量内点，不能简单删掉。

### 2. Primary support 判断

今天把 primary support 定义为：

```text
direct motion/ref inlier + strict continuous local projection inlier
```

201 个采样帧统计：

- primary association / inlier：22712 / 22220。
- primary 外点率：2.17%。
- 平均 primary inlier：110.55 个/帧。
- 173/201 帧达到 80 个 primary inlier。
- normal motion 152 帧中，149 帧 primary 明确足够。
- weak motion 49 帧中，只有 19 帧 primary 明确足够，28 帧不足。

结论：

```text
normal 段：direct + strict continuous 基本够用。
weak 段：primary 不够，需要 secondary / voted / expansion 补支撑。
```

这解释了为什么不能直接做 strict-only 或 primary-only：normal 段会更干净，但 weak 段会失去必要约束。

### 3. Primary-first 第一版

第一版 `XFEAT_LOCALMAP_PRIMARY_FIRST=1` 只是改变匹配顺序：

```text
direct association
-> strict continuous projection
-> strict non-continuous + voted projection
-> support 不足时 expansion
-> PoseOptimization
```

`run_MH04_primary_first.log` 中 67 个采样帧显示：

- `motion_lg` 外点率：0.86%。
- `local_proj` 外点率：19.82%。
- `primary_matches=870`。
- `secondary_matches=5618`。
- primary 只占新增 projection matches 的 13.4%。
- secondary 主要来自 strict non-continuous 和 voted，不是 expansion。

结论：只改先后顺序收益有限。因为 secondary 仍大量进入 PoseOptimization，最终优化约束结构几乎没变。

### 4. Adaptive primary-first

之后把 primary-first 从“排序”升级为“准入控制”：

```text
direct association
-> strict continuous projection
-> 如果 motion 正常且 direct + primary >= XFEAT_LOCALMAP_TEMPORAL_MIN_SUPPORT:
       suppress strict non-continuous + voted secondary
       suppress expansion
   否则:
       admit secondary
       base 仍不足时 admit expansion
-> PoseOptimization
```

保留一个阈值：

```text
XFEAT_LOCALMAP_TEMPORAL_MIN_SUPPORT=140
```

运行结论：

- `secondary_suppressed=1` 的 normal 帧确实更干净。
- suppressed 帧 local projection 外点率从约 13.8%~18.0% 降到约 8.0%~15.5%。
- 但 suppressed 帧只贡献一部分约束。
- 整体 local projection 外点率仍约 20%~22%。
- weak frame 的 APE 明显高于 normal frame。
- 典型 weak frame 930：`primary_support≈58~62`，local projection 外点率约 42%~44%，采样误差约 0.21~0.23 m。

结论：

```text
adaptive primary-first 能让 normal 段更干净，
但没有解决 weak/fallback 段如何安全使用 secondary 的问题。
```

### 5. Weak secondary admission 负结果

短期验证过：

```text
primary 不足或 motion weak
-> secondary 按质量排序
-> 按 support deficit 限量 admit secondary
-> base 仍不足时 admit expansion
```

两次 MH04 结果：

| 日志 | local_proj outlier | secondary matches | KFs | loop |
|---|---:|---:|---:|---:|
| `run_MH04_weak_secondary_admission_1.log` | 18.6% | 4198 | 577 | 3 |
| `run_MH04_weak_secondary_admission_2.log` | 19.6% | 4309 | 572 | 2 |

虽然 local projection 外点率略降，但：

- secondary matches 从 4667~5344 降到 4198~4309。
- KFs 从约 529~557 增加到 572~577。
- loop 接受没有更早、更强。
- RMSE 没有稳定改善。

结论：weak 段不能简单少用 secondary。secondary 更脏，但它也提供了 weak 段必要的有效支撑。

### 6. Two-stage LocalMap projection 负结果

随后验证过 two-stage 思路：

```text
direct + primary support >= 80
-> 在临时 Frame 上做一次 PoseOptimization
-> 临时优化 inliers >= 60 时，只回写 pose
-> 用更新后的 pose 重新投影 secondary / expansion
-> 最终正式 PoseOptimization
```

三次 MH04 结果：

| 日志 | RMSE | 1m RPE RMSE | KFs | loop / GBA |
|---|---:|---:|---:|---|
| `run_MH04_two_stage_localmap_1.log` | 0.077933 | 0.033559 | 562 | 2 / 2 applied |
| `run_MH04_two_stage_localmap_2.log` | 0.073675 | 0.035747 | 560 | 3 / 3 applied |
| `run_MH04_two_stage_localmap_3.log` | 0.098668 | 0.033863 | 539 | 3 / 2 applied |

表面 RMSE 很好，但日志不支持把收益归因给 two-stage：

- two-stage 分别尝试并接受 48/48、52/52、49/49 次。
- 临时优化只剔除约 2% 的 direct + primary association，说明输入 pose 本来已经比较干净。
- `primary_support=80~139` 区间 local projection 外点率为 21.8% / 21.2% / 21.6%。
- adaptive primary-first 同一区间约 21.1% / 21.1%，没有改善。
- 1m RPE 与 adaptive primary-first 基本一致。
- 最弱的 `primary_support<80` 帧不会触发 two-stage，而这些帧才是主要风险段。

结论：

```text
two-stage 的低 RMSE 主要来自 loop/GBA 和运行状态，不是 two-stage 行为本身。
```

因此 two-stage 逻辑、环境变量和专用日志已删除。

## Loop / GBA 归因

今天的多组结果继续说明：MH04 全局 RMSE 仍然高度受 loop 接受时机、loop 几何质量和 GBA 是否完整应用影响。

典型现象：

- continuity 三次 LocalMap 统计接近，但 RMSE 从 0.071928 到 0.118919 波动。
- two-stage 三次 1m RPE 基本相近，但 APE RMSE 仍跟 loop/GBA 状态变化。
- `run_MH04_two_stage_localmap_3.log` 中有 3 次 loop correction，但 GBA `started=3 / applied=2 / aborted=1`，最终 RMSE 比另外两次差。

当前 loop 前端已不是单纯 ORBvoc：

```text
XFeat vocabulary retrieval
-> XFeat loop group
-> representative KF LightGlue
-> F-RANSAC
-> Sim3
-> group projection/refill
-> OptimizeSim3
-> CorrectLoop/GBA
```

但不稳定点仍在：

- group 在 Sim3 前参与不足，pre-Sim3 仍偏 representative-KF driven。
- `direct_f_inliers` 往往足够，但 `sim3_inliers` 仍可能偏低。
- `proj_opt_matches` 高不一定代表初始 Sim3 很强。
- loop 被早期接受后，后续潜在 loop 可能因图结构变化不再被接受。
- GBA 可能被新的 loop 或 shutdown 时机影响，导致 run-to-run APE 波动。

## 今日代码处理

最终提交：

```text
84c49af tracking: finalize adaptive primary-first local map baseline
```

提交内容：

- 保留 adaptive primary-first 作为当前 LocalMap 主线。
- 删除 `XFEAT_LOCALMAP_TWO_STAGE` 逻辑、环境变量和日志字段。
- 删除 weak-secondary admission 真实行为。
- 保留 `XFEAT_LOCALMAP_TEMPORAL_EXPANSION_LAST=1`。
- 保留 `XFEAT_LOCALMAP_PRIMARY_FIRST=1`。
- 保留 MapPoint recent direct inlier history，用于 strict continuous 判断。
- 收敛文档记录，避免后续重复恢复已验证无稳定收益的路径。

涉及文件：

- `include/MapPoint.h`
- `include/Tracking.h`
- `src/MapPoint.cc`
- `src/Tracking.cc`
- `command.md`
- `docs/airslam_xfeat_slam_adaptation_direction.md`
- `docs/xfeat_slam_deep_feature_adaptation_notes.md`
- `docs/xfeat_slam_nlevels1_workflow_plan.md`

## 今日核心结论

1. motion/ref LG 是干净的，不是当前主要误差源。

2. LocalMap 中质量排序成立：

```text
direct motion/ref
< strict continuous
< strict non-continuous
< voted
< expansion
```

3. normal 段 primary 基本够用，因此 adaptive primary-first 对 normal 段有局部正向作用。

4. weak 段 primary 不够，必须启用 secondary/voted；简单减少 secondary 会丢掉有效支撑，导致 KFs 增多、RMSE 不稳定。

5. two-stage refined pose 没有稳定贡献，因为它覆盖的是中等支撑段，而最差段 `primary_support<80` 没被解决。

6. 目前全局 APE RMSE 的主要波动仍然来自 loop/GBA，而不是局部 tracking 的短期 RPE。

7. 后续不应继续堆 LocalMap 单阈值 guard，应优先处理 loop/GBA 稳定性，或者做更结构化的后端适配。

## 下一步建议

优先级 1：稳定 loop / GBA。

- 对每次 run 继续按 loop 状态分组：accept count、candidate rank、sim3 inliers、proj opt matches、GBA applied。
- 重点看为什么 pre-Sim3 group 几何仍不稳定。
- 考虑让 group 在 Sim3 前更充分参与，但要限制 LG 调用量。
- 评估 GBA 生命周期：避免新 loop / shutdown 时机导致 GBA apply 不一致。

优先级 2：保持 LocalMap 当前主线，不继续做小 gate。

- 当前保留 adaptive primary-first。
- 不恢复 weak secondary admission。
- 不恢复 two-stage。
- 不继续调 secondary budget / source risk / lifecycle expansion guard。

优先级 3：如果 loop 稳定后仍落后 ORB-SLAM3，再考虑更深层后端适配。

- XFeat/stereo observation noise 与 optimizer 权重重新标定。
- Local BA temporal-core 化，而不是继续扩大共视 LocalMap。
- MapPoint 生命周期要和 BA / covis / projection admission 联动，不能只做当前帧局部过滤。

## 当前推荐验证配置

```bash
XFEAT_LIGHTGLUE_WEIGHT=./weights/xfeat_lighterglue_matcher_cpp.pt \
XFEAT_USE_LIGHTGLUE_MOTION=1 \
XFEAT_USE_LIGHTGLUE_TRIANG=1 \
XFEAT_USE_LIGHTGLUE_REF=1 \
XFEAT_USE_LIGHTGLUE_LOOP=1 \
XFEAT_USE_XFEAT_VOCAB=1 \
XFEAT_VOCAB_PATH=Vocabulary/xfeat/models/xfeat_float_euroc_all_k10_l5_f512_i150_r001.dbow3 \
XFEAT_VOCAB_MAX_FEATURES_PER_KF=512 \
XFEAT_LOCALMAP_TEMPORAL_EXPANSION_LAST=1 \
XFEAT_LOCALMAP_PRIMARY_FIRST=1 \
XFEAT_LOCALMAP_TEMPORAL_MIN_SUPPORT=140 \
./examples/Stereo/stereo_euroc \
  Vocabulary/ORBvoc.txt \
  examples/Stereo/EuRoC.yaml \
  "/mnt/d/data_set/EuRoc Wav Dataset/machine_hall/MH_04_difficult/mav0" \
  examples/Stereo/EuRoC_TimeStamps/MH04.txt \
  EuRoC_MH_04_difficult_stereo \
  2>&1 | tee experiment_logs/Stereo/EuRoC/MH_04_difficult/run_MH04_adaptive_primary_first_baseline.log
```
