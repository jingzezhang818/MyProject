# xfeatSLAM 代码修改说明

## XFeat 特征集成

### XFeatMatcher

新增独立的 `XFeatMatcher` 类，替代 ORB matching 在关键路径上的使用：

- **位置**: [include/XFeatMatcher.h](include/XFeatMatcher.h), [src/XFeatMatcher.cc](src/XFeatMatcher.cc)
- **主要方法**:
  - `SearchByNN` - 最近邻搜索，用于初始匹配
  - `SearchByProjection` - 基于投影的匹配，用于局部地图跟踪
  - `SearchForInitialization` - 双目初始化匹配

### XFeat 特征提取器

集成 XFeat 检测器与 ORB 共存：

- **位置**: [include/XFeat.h](include/XFeat.h), [include/XFextractor.h](include/XFextractor.h)
- **特性**: 类 SuperPoint 检测器，输出 float 描述子

### MapPoint 增强

扩展 MapPoint 支持多描述子 bank：

- **位置**: [include/MapPoint.h](include/MapPoint.h), [src/MapPoint.cc](src/MapPoint.cc)
- **成员**: `mDescriptorBank` 存储每个地图点的多个 float 描述子
- **Medoid 选择**: 自动选择代表描述子

## 修改的跟踪路径

| 路径 | 方法 | Matcher |
|------|------|---------|
| 单目初始化 | `SearchForInitialization` | XFeatMatcher |
| 参考关键帧跟踪 | `SearchByNN` | XFeatMatcher |
| 运动模型 | `SearchByProjection` | XFeatMatcher |
| 局部地图 | `SearchByProjection` | XFeatMatcher |
| 重定位 | `SearchByNN` | XFeatMatcher |

## TrackReferenceKeyFrame 三路融合

将 TrackReferenceKeyFrame 的 XFeat 分支从"整包替换"改为"三路融合"：

1. **strict 模式**: 严格 NN 匹配，作为主干结果
2. **relaxed 模式**: 只填充空位，不覆盖已有匹配
3. **projection fallback**: 投影匹配作为最后补位，不覆盖已有匹配

## 仍保留的 ORB 依赖

- 启动时仍加载 ORB Vocabulary
- 部分 fallback 路径仍使用 ORBmatcher
- BoW 索引仍基于 ORB

## 第一阶段：稳态基线版（运行时开关）

当前逻辑以 XFeat 作为默认前端，运行时通过少量环境变量切换 Reference/Relocalization 策略：

| 环境变量 | 作用 | 默认值 |
|---|---|---|
| `XFEAT_STAGE1_REF_RELOC_MODE` | `current`/`hybrid`，统一控制 `TrackReferenceKeyFrame` 与 `Relocalization` | `hybrid` |
| `XFEAT_STAGE1_DEBUG` | Stage1 统一调试日志开关（兼容 `XFEAT_DEBUG`） | `0` |
| `XFEAT_ENABLE_LIGHT_DUAL_DESC` | 轻量 ORB side-channel 总开关（KeyFrame ORB side desc + ORB BoW） | `1`（XFeat 前端） |
| `XFEAT_ORB_SIDE_ON_KF_ONLY` | 是否仅在关键帧常驻 ORB side 特征（`1` 时普通帧按需懒提取） | `1` |
| `XFEAT_ORB_SIDE_ON_RELOC_FRAME` | Relocalization 是否允许当前帧按需提取 ORB side | `1` |
| `XFEAT_USE_DESC_BANK` | MapPoint descriptor bank 开关 | `0` |
| `USE_ORB` | 切换为纯 ORB 前端（兼容保留） | 未设置 |

说明：
- `Initialization / MotionModel / LocalMap`：默认始终走 XFeat（除非显式 `USE_ORB=1`）。
- `XFEAT_STAGE1_REF_RELOC_MODE=current`：Reference + Relocalization 走纯当前 XFeat 逻辑。
- `XFEAT_STAGE1_REF_RELOC_MODE=hybrid`：在 XFeat 前端下，Reference/Relocalization 优先尝试 ORB side-channel（BoW 候选），再由 XFeat 做补匹配或 refine。

### 四组消融配置（把 `<run_cmd>` 替换成你当前运行命令）

1. XFeat 纯当前链路（对照）
```bash
XFEAT_STAGE1_REF_RELOC_MODE=current \
XFEAT_ENABLE_LIGHT_DUAL_DESC=1 \
XFEAT_USE_DESC_BANK=0 \
<run_cmd>
```

2. 稳态混合（Reference/Reloc ORB side-channel 主导）
```bash
XFEAT_STAGE1_REF_RELOC_MODE=hybrid \
XFEAT_ENABLE_LIGHT_DUAL_DESC=1 \
<run_cmd>
```

3. 稳态混合 + bank off（显式）
```bash
XFEAT_STAGE1_REF_RELOC_MODE=hybrid \
XFEAT_ENABLE_LIGHT_DUAL_DESC=1 \
XFEAT_USE_DESC_BANK=0 \
<run_cmd>
```

4. 稳态混合 + bank on
```bash
XFEAT_STAGE1_REF_RELOC_MODE=hybrid \
XFEAT_ENABLE_LIGHT_DUAL_DESC=1 \
XFEAT_USE_DESC_BANK=1 \
<run_cmd>
```
