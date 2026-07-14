# XFeat Loop 当前基线与后续方向

## 当前结论

当前 loop 主线固定为：

```text
XFeat DBoW3 candidate groups
-> 每个 group 的 representative KF
-> 单次 KF-KF LightGlue
-> Fundamental RANSAC
-> ORB-SLAM3 Sim3Solver
-> group-aware projection/refill
-> OptimizeSim3
-> covisible validation
-> CorrectLoop / GBA
```

2026-07-14 的稳定 checkpoint 以 `84c49af` 的整体行为为边界，并保留随后完成的纯诊断清理。
`examples/Stereo/EuRoC.yaml` 显式设置 `loopClosing: 1`，使用 loop-enabled 路径。
MH04 曾在这条主线附近得到 `0.071038` 和 `0.070169` 的单次 APE RMSE；由于异步 LocalMapping、
loop 接受时机和 GBA 应用仍会造成运行波动，这两个值只作为历史 best run，不作为重复性承诺。

该链路已经包含 learned-feature place recognition、2D 几何过滤、3D 几何验证、
group 补匹配和 ORB-SLAM3 全局校正。当前不再继续叠加 PnP fallback、多窗口 LG、
候选一致性开关或更宽松的 loop 接受阈值。

保留的运行开关只有：

```text
XFEAT_USE_LIGHTGLUE_LOOP=1
XFEAT_USE_XFEAT_VOCAB=1
XFEAT_VOCAB_PATH=...
XFEAT_VOCAB_MAX_FEATURES_PER_KF=512
```

`XFEAT_DEBUG=1` 仅用于临时诊断，不属于默认运行配置。

## 已确认的 Sim3 输入问题

LightGlue/Fundamental 的 2D 内点不能直接成为 Sim3 对应。每一对 Sim3 输入还要求：

```text
current feature -> 有效 current MapPoint
candidate feature -> 有效 candidate MapPoint
两侧 MapPoint 均有合法 KeyFrame observation
```

历史 Sim3 漏斗诊断得到两个结论：

1. 部分候选确实因为 current feature 没有关联 MapPoint，2D 对应无法转成 3D-3D 对应。
2. 这不是唯一原因。多数被拒候选仍有足够数量的 3D-3D 对，但这些对应无法被同一个
   Sim3 一致解释，说明 loop 两侧局部地图的几何一致性不足。

两次代表性诊断运行：

| 日志 | APE RMSE | accepted loops | GBA applied |
|---|---:|---:|---:|
| `run_MH04_loop_sim3_input_diag_1.log` | 0.070169 | 4 | 4 |
| `run_MH04_loop_sim3_input_diag_2.log` | 0.123540 | 2 | 2 |

两次 tracking/LocalMap 统计接近，最终差异主要来自 loop 接受时机、覆盖范围和完成的
全局校正次数。详细 Sim3 漏斗字段已经完成归因，不再常驻代码和日志。

代码只保留一处独立的健壮性修复：当有效 3D-3D 数量少于要求的 RANSAC 内点数时，
`SetRansacParameters()` 直接结束，避免无效的概率和对数运算。该保护不改变原有接受语义。

## PnP Loop 实验与回退

PnP 实验尝试在 Sim3 失败后使用：

```text
candidate MapPoint 3D -> current keypoint 2D
-> MLPnP RANSAC
-> KeyFrame pose-only optimization
-> PnP seed projection
-> 重新进入原 Sim3/projection/covisible 验证链
```

首版 PnP fallback 虽能求出部分 pose，但丢弃了 PnP 3D-2D 内点，后续
`proj_matches=0`。完整实验版保留了 PnP seed，三次 MH04 结果如下：

| 日志 | APE RMSE | PnP attempts | pose optimized | PnP accepted |
|---|---:|---:|---:|---:|
| `run_MH04_loop_pnp_full_1.log` | 0.100749 | 29 | 4 | 0 |
| `run_MH04_loop_pnp_full_2.log` | 0.102317 | 35 | 1 | 0 |
| `run_MH04_loop_pnp_full_3.log` | 0.078866 | 32 | 2 | 0 |

三次没有任何 loop 由 PnP 路径接受。第二、三次接受的 loop 均来自原 Sim3 路径；
第三次低 RMSE 不能归因于 PnP。PnP seed 通常只有约 20~31 个 pose inliers，后续投影
最多增长到约 27~53，无法满足现有完整地图验证要求。

因此已回退以下实验代码：

- `XFEAT_LOOP_PNP_FALLBACK`。
- KeyFrame 版 `MLPnPsolver`。
- KeyFrame pose-only `Optimizer` 接口。
- PnP seed 与 Sim3 混合验证。
- `[LoopLG][PnP]`、PnP summary 计数和常驻 Sim3 漏斗字段。

PnP 不是被证明完全不可行，而是没有证据支持把它作为 ORB-SLAM3 Sim3 链的 fallback。
如果未来重新验证，应在独立分支中实现完整的 PnP loop 前端与验收契约，而不是再次接回
现有 Sim3 的中后段。

## 保留的诊断边界

默认只保留：

```text
[LoopLG][Accept]
[LoopLG][Summary]
```

它们分别记录实际接受的候选几何和本次运行的 loop/GBA/shutdown 状态。
`[LoopLG][NearMiss]` 只在 `XFEAT_DEBUG=1` 时按现有间隔打印，用于定位某次具体实验，
不再增加新的 shadow/audit 字段。

与 NearMiss 内容重复的 `[LoopLG][BestRejected]` 及其候选缓存已删除；旧的
`vnStage/vnMatchesStage` 统计结果从未被消费，也已删除。异常情况下的
`[LoopLG][Fallback]` 继续保留，用来识别 LightGlue 调用失败后是否回退到 ORB 匹配。

比较 RMSE 时必须先按以下状态分组：

```text
no loop
loop accepted but GBA not applied
loop accepted and GBA applied
shutdown timeout / GBA aborted
```

## 下一步工作流

### 阶段 0：固定可复现实验

- 使用 `command.md` 中的稳定 loop 配置连续运行 MH04 至少 3 次。
- 记录 APE RMSE、短窗口 RPE、accepted KF pair、Sim3 inliers、projection matches、
  GBA applied 和 KeyFrame 数。
- 使用 median 和同类 loop/GBA 分组比较，不用单次 best 判断代码收益。
- 如果需要专门拆分随机采样因素，应复制一份实验 YAML 并在副本中设置 `System.RandomSeed: 0`；
  不再修改正式 EuRoC baseline 配置。

### 阶段 1：冻结 LoopClosing 行为

当前不修改 candidate 数量、F/Sim3/projection 阈值，也不增加候选 persistence、PnP fallback
或多窗口 LG。后续实验应区分 Sim3 随机采样与异步 LocalMapping 导致的 KeyFrame/MapPoint
状态变化。

### 阶段 2：处理 loop 之前的地图一致性

现有证据更支持检查以下上游问题：

- 同一真实区域在不同时段形成的 MapPoint 几何是否一致。
- loop current KF 上已有 MapPoint 的比例和空间覆盖。
- Local BA 后 observation/outlier 是否被及时清理。
- close/mid XFeat MapPoint 是否造成两段局部地图结构不一致。

目标不是减少所有地图点，而是提高进入 loop 3D 几何验证的 MapPoint 一致性。该工作应在
MapPoint 生命周期、Local BA observation 清理或局部优化窗口中完成，而不是在 LoopClosing
末端放宽规则。

### 阶段 3：仅在新证据支持时重构 group anchor

如果固定实验表明真实 loop group 已稳定召回，但 representative KF 长期不是 group 内最好的
3D 几何锚点，才考虑用 group-anchor selection 替换 representative-only。它必须是单一替换：

```text
group 内固定最多两个候选
-> LG + F + 可用 3D-3D 支撑评估
-> 选择一个 anchor
-> 只运行现有 Sim3 后半链
```

不再同时保留 representative、covisible expansion、PnP fallback 三套竞争路径。如果没有
representative 选择错误的直接日志证据，则不进入该阶段。
