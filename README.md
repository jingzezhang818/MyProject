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
