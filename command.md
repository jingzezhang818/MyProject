# XFeatSLAM 常用命令与环境变量

## 1. 命令模板（按数据集）

### 1.1 EuRoC — V1_01

#### 双目

```bash
XFEAT_USE_LIGHTGLUE_MOTION=1 \
XFEAT_LG_MOTION_USE_PROJ_GATE=0 \
XFEAT_LIGHTGLUE_WEIGHT=./weights/xfeat_lighterglue_matcher_cpp.pt \
XFEAT_DEBUG=1 \
XFEAT_DIAG_FEATURE_DISTRIBUTION=1 \
XFEAT_DIAG_INTERVAL=30 \
./examples/Stereo/stereo_euroc \
  Vocabulary/ORBvoc.txt \
  examples/Stereo/EuRoC.yaml \
  "/mnt/d/data_set/EuRoc Wav Dataset/vicon_room1/V1_01_easy/mav0" \
  examples/Stereo/V1_01_easy_TimeStamps/times.txt \
  EuRoC_V101 \
  2>&1 | tee run_EuRoC_V101.log
```

- `stereo_euroc` 最后一个参数是输出基名，程序自动保存 `f_<基名>.txt` 和 `kf_<基名>.txt`
- 代码依据：`examples/Stereo/stereo_euroc.cc:171-177`

#### 转 TUM

```bash
python convert_euroc_to_tum.py \
  --gt ground_truth_tum_V101.txt \
  --est-traj f_EuRoC_V101.txt \
  --out-est experiment_logs/Stereo/V101/CameraTrajectory_V101_tum_LG.txt \
  --timestamp-unit ns \
  --estimate-right-rot
```

脚本结束自动打印 `evo_ape` / `evo_traj` 命令（`convert_euroc_to_tum.py:548-562`）。

#### 轨迹评估

```bash
evo_ape tum ground_truth_tum_V101.txt experiment_logs/Stereo/V101/CameraTrajectory_V101_tum.txt -r trans_part -a -p
evo_traj tum experiment_logs/Stereo/V101/CameraTrajectory_V101_tum.txt --ref=ground_truth_tum_V101.txt -a -p
```

### 1.2 SePT01

#### 双目

```bash
XFEAT_DIAG_FEATURE_DISTRIBUTION=1 \
XFEAT_DIAG_INTERVAL=20 \
./examples/Stereo/stereo_euroc \
  Vocabulary/ORBvoc.txt \
  examples/Stereo/SePT01.yaml \
  /mnt/d/data_set/SePT01/mav0 \
  examples/Stereo/SePT01_TimeStamps/times.txt \
  SePT01 \
  2>&1 | tee run_SePT01.log
```

```bash
python convert_euroc_to_tum.py \
  --gt ground_truth_tum_SePT01.txt \
  --est-traj f_SePT01.txt \
  --out-est experiment_logs/Stereo/SePT01/CameraTrajectory_SePT01_tum_LG.txt \
  --timestamp-unit ns \
  --estimate-right-rot
```

#### 单目

```bash
./examples/Monocular/mono_euroc \
  Vocabulary/ORBvoc.txt \
  examples/Monocular/SePT01_cam0.yaml \
  /mnt/d/data_set/SePT01 \
  examples/Monocular/SePT01_cam0.txt \
  2>&1 | tee run_SePT01_mono.log
```

### 1.3 Moon_1

#### 单目

```bash
./examples/Monocular/mono_euroc \
  Vocabulary/ORBvoc.txt \
  examples/Monocular/Moon_1.yaml \
  /mnt/d/data_set/Moon_1 \
  examples/Monocular/Moon_1_times.txt \
  2>&1 | tee run_Moon1.log
```

## 2. 环境变量参考

> 布尔变量解析规则：变量存在且值不是 `"" / 0 / false / FALSE` 即视为开启。
> 依据：`src/Tracking.cc:56-63`、`src/XFeatMatcher.cc:19-28`、`src/XFextractor.cc:84-93`

### 2.1 通用 / 调试

| 变量 | 作用 | 默认 | 范围 | 代码依据 |
|---|---|---|---|---|
| `USE_ORB` | 切换到 ORB 前端 | 未设置 | — | `src/Tracking.cc:2328-2350` |
| `XFEAT_DEBUG` | 全局调试输出（高频） | 关 | — | `src/Tracking.cc:65-69`, `src/Frame.cc:43-53` |
| `XFEAT_DIAG_FEATURE_DISTRIBUTION` | 特征分布诊断日志 | 关 | — | `src/Tracking.cc:72-77,3322-3327` |
| `XFEAT_DIAG_INTERVAL` | 诊断日志间隔帧数 | `1` | `[1,500]` | `src/Tracking.cc:79-99` |
| `XFEAT_DIAG_GRID_COLS` | 诊断空间网格列数 | `8` | `[2,64]` | `src/Tracking.cc:102-123` |
| `XFEAT_DIAG_GRID_ROWS` | 诊断空间网格行数 | `6` | `[2,64]` | `src/Tracking.cc:125-146` |
| `XFEAT_FPS` / `SLAM_FPS` | RuntimeFPS 日志 | 关 | — | `src/Tracking.cc:148-152` |
| `XFEAT_FPS_INTERVAL` / `SLAM_FPS_INTERVAL` | RuntimeFPS 输出间隔 | `30` | `[1,1000]` | `src/Tracking.cc:154-176` |

### 2.2 跟踪 / 重定位

| 变量 | 作用 | 默认 | 范围 | 代码依据 |
|---|---|---|---|---|
| `XFEAT_MONO_NEAR_ONLY` | 单目初始化只用近点 | 关 | — | `src/Tracking.cc:675-680` |
| `XFEAT_MONO_NEAR_DEPTH_FACTOR` | near-only 深度阈值系数 | `2.5` | `>0` | `src/Tracking.cc:682-700` |
| `XFEAT_RELOC_INLIER_TH` | 重定位后内点阈值 | `35` | `>0` | `src/Tracking.cc:635-653,4475-4477` |
| `XFEAT_INIT_MIN_BASELINE_DEPTH_RATIO` | 单目初始化 baseline/medianDepth 下限 | `0.01` | `>0` | `src/Tracking.cc:655-673` |
| `XFEAT_STEREO_DESC_TH` | 立体描述子阈值 | `1.35` | `[0.05,2.0]` | `src/Frame.cc:56-71` |
| `XFEAT_STEREO_MIN_DISPARITY` | 最小视差（像素） | `1.0` | `[0,100]` | `src/Frame.cc:73-91` |
| `XFEAT_PIXEL_SIGMA` | 视觉误差像素 sigma | `1.0` | `[0.5,5.0]` | `src/Optimizer.cc:59-77` |

#### XFeat 阈值

| 变量 | 作用 | 默认 | 范围 | 代码依据 (`src/Tracking.cc`) |
|---|---|---|---|---|
| `XFEAT_TH_HIGH_REF_NN_STRICT` | RefKF 严格 NN 阈值 | `1.35` | `[0.05,2.0]` | `:587-591` |
| `XFEAT_TH_HIGH_REF_NN_RELAXED` | RefKF 宽松 NN 阈值 | `1.60` | `[0.05,2.0]` | `:593-597` |
| `XFEAT_TH_HIGH_REF_PROJ` | RefKF 投影补匹配阈值 | `1.40` | `[0.05,2.0]` | `:599-603` |
| `XFEAT_TH_HIGH_MOTION_PROJ` | 运动模型投影阈值 | `1.45` | `[0.05,2.0]` | `:605-609` |
| `XFEAT_TH_HIGH_LOCAL_PROJ` | LocalMap 投影阈值 | `1.35` | `[0.05,2.0]` | `:611-615` |
| `XFEAT_TH_HIGH_RELOC_NN` | 重定位初配 NN 阈值 | `1.75` | `[0.05,2.0]` | `:617-621` |
| `XFEAT_TH_HIGH_RELOC_PROJ_COARSE` | 重定位粗窗口投影阈值 | `1.80` | `[0.05,2.0]` | `:623-627` |
| `XFEAT_TH_HIGH_RELOC_PROJ_FINE` | 重定位细窗口投影阈值 | `1.60` | `[0.05,2.0]` | `:629-633` |

### 2.3 提取器 (XFeat)

| 变量 | 作用 | 默认 | 范围 | 代码依据 (`src/XFextractor.cc`) |
|---|---|---|---|---|
| `XFEAT_PROFILE` | 各阶段 profile 计时 | 关 | — | `:132-136,1155-1181` |
| `XFEAT_DEVICE` | 推理设备 `auto/cuda/cpu` | `auto` | — | `:891-899,953-968` |
| `XFEAT_CUDA_DEVICE` | CUDA 卡索引 | `0` | — | `:900-903` |
| `XFEAT_FIXED_NMS_CANDIDATE_FACTOR` | 固定 TopK 候选放大系数 | `8` | `[1,16]` | `:138-142,1216-1218` |
| `XFEAT_FIXED_NMS_CANDIDATE_MIN` | 固定 TopK 候选最小数 | `1024` | `[64,8192]` | `:144-148,1216-1218` |
| `XFEAT_UNIFORM_DISABLE` | 关闭提点空间均匀化 | 不设（开启） | — | `:150-155` |
| `XFEAT_UNIFORM_LEGACY_GRID` | 回退旧均匀化策略（不用 OctTree+ANMS） | 不设（用新） | — | `:157-162` |
| `XFEAT_UNIFORM_GRID_COLS` | 均匀化网格列数 | `8` | `[2,64]` | `:164-169` |
| `XFEAT_UNIFORM_GRID_ROWS` | 均匀化网格行数 | `6` | `[2,64]` | `:171-176` |
| `XFEAT_UNIFORM_ANMS_MIN_RADIUS` | ANMS 最小半径 | `2` | `[0,128]` | `:178-183` |
| `XFEAT_UNIFORM_ANMS_MAX_RADIUS` | ANMS 最大半径（`0`=自适应） | `0` | `[0,512]` | `:185-190` |
| `XFEAT_UNIFORM_ANMS_AUTO_SCALE` | ANMS 自适应尺度系数 | `1.5` | `[0.5,5.0]` | `:192-197` |

### 2.4 匹配器 (XFeatMatcher)

| 变量 | 作用 | 默认 | 范围 | 代码依据 (`src/XFeatMatcher.cc`) |
|---|---|---|---|---|
| `XFEAT_DEBUG_MATCHER` | 匹配器调试 | 关 | — | `:30-35` |
| `XFEAT_DEBUG_MATCHER_VERBOSE` | 匹配器详细统计 | 关 | — | `:37-42` |
| `XFEAT_USE_DESC_BANK` | 启用 descriptor bank | 关 | — | `:44-49` |
| `XFEAT_LEGACY_LEVEL_RATIO` | 旧版 ratio gate 逻辑 | 关 | — | `:51-56` |
| `XFEAT_ALLOW_NON_MUTUAL_NN` | 放开 NN mutual 硬约束 | 关（强制 mutual） | — | `:58-63` |
| `XFEAT_MATCH_SPATIAL_DISABLE` | 关闭匹配级空间配额 | 不设（开启） | — | `:101-106` |

#### 空间配额

| 变量 | 作用 | 默认 | 范围 | 代码依据 (`src/XFeatMatcher.cc`) |
|---|---|---|---|---|
| `XFEAT_MATCH_SPATIAL_GRID_COLS` | 空间网格列数 | `8` | `[2,64]` | `:108-113` |
| `XFEAT_MATCH_SPATIAL_GRID_ROWS` | 空间网格行数 | `6` | `[2,64]` | `:115-120` |
| `XFEAT_MATCH_SPATIAL_TRIGGER` | 低于该候选数不做空间配额 | `40` | `[1,10000]` | `:122-127` |
| `XFEAT_MATCH_SPATIAL_KEEP_RATIO` | 空间配额后最小保留比例 | `0.90` | `[0.10,1.00]` | `:129-134` |
| `XFEAT_MATCH_SPATIAL_CAP_SCALE` | 每格上限系数 | `0.85` | `[0.30,3.00]` | `:136-141` |

#### 投影匹配

| 变量 | 作用 | 默认 | 范围 | 代码依据 (`src/XFeatMatcher.cc`) |
|---|---|---|---|---|
| `XFEAT_PROJ_DISABLE_LEVEL_GATE` | 关闭投影匹配层级门控 | 不设（开启） | — | `:143-148` |
| `XFEAT_PROJ_MAP_MIN_OFFSET` | Map 投影层级最小偏移 | `-2` | `[-8,8]` | `:150-155` |
| `XFEAT_PROJ_MAP_MAX_OFFSET` | Map 投影层级最大偏移 | `1` | `[-8,8]` | `:157-162` |
| `XFEAT_PROJ_LAST_USE_DIR_BIAS` | LastFrame 投影按前后方向分窗 | 关 | — | `:164-169` |
| `XFEAT_PROJ_LAST_MIN_OFFSET` | LastFrame 统一窗口最小偏移 | `-2` | `[-8,8]` | `:171-176` |
| `XFEAT_PROJ_LAST_MAX_OFFSET` | LastFrame 统一窗口最大偏移 | `2` | `[-8,8]` | `:178-183` |
| `XFEAT_PROJ_LAST_FWD_MIN_OFFSET` | LastFrame 前进窗口最小偏移 | `-1` | `[-8,8]` | `:185-190` |
| `XFEAT_PROJ_LAST_FWD_MAX_OFFSET` | LastFrame 前进窗口最大偏移 | `2` | `[-8,8]` | `:192-197` |
| `XFEAT_PROJ_LAST_BWD_MIN_OFFSET` | LastFrame 后退窗口最小偏移 | `-2` | `[-8,8]` | `:199-204` |
| `XFEAT_PROJ_LAST_BWD_MAX_OFFSET` | LastFrame 后退窗口最大偏移 | `1` | `[-8,8]` | `:206-211` |
| `XFEAT_TRIANG_KNN` | 三角化阶段每点 KNN 候选数 | `32` | `[2,128]` | `:281-298` |

## 3. 备注

- 多数环境变量在函数内以 `static` 方式缓存，进程启动后不再动态修改。
- `XFEAT_DEBUG=1` 是高频日志开关；`XFEAT_DIAG_INTERVAL` 只控制诊断频率，不是全局限流。
