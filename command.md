# XFeatSLAM 常用命令与环境变量

## 1. 命令模板（按数据集）

### 1.1 通用约定

- 以下命令都从仓库根目录运行。
- 示例程序会按最后一个参数自动保存 `f_<trajectory_name>.txt` 和 `kf_<trajectory_name>.txt`。
- 转换后的 TUM 文件建议放到 `experiment_logs/<Monocular|Stereo>/<dataset>/<sequence>/`。
- 如果输出目录不存在，先执行对应的 `mkdir -p ...`。
- 单目轨迹评估使用 Sim3/尺度对齐：`-as`；双目轨迹评估使用 SE3 对齐：`-a`。
- 单目默认使用较保守的 XFeat 配置；需要试 LightGlue 时，把 `XFEAT_USE_LIGHTGLUE_REF=1` 或 `XFEAT_USE_LIGHTGLUE_MOTION=1` 内联加到运行命令的环境变量里。

### 1.2 EuRoC

已生成的真值文件位于 `examples/GroundTruth/EuRoC/`：

| 序列 | 数据集路径 | 单目时间戳 | 双目时间戳 | 真值 |
|---|---|---|---|---|
| `MH_01_easy` | `/mnt/d/data_set/EuRoc Wav Dataset/machine_hall/MH_01_easy` | `examples/Monocular/EuRoC_TimeStamps/MH01.txt` | `examples/Stereo/EuRoC_TimeStamps/MH01.txt` | `examples/GroundTruth/EuRoC/MH_01_easy.tum` |
| `MH_04_difficult` | `/mnt/d/data_set/EuRoc Wav Dataset/machine_hall/MH_04_difficult` | `examples/Monocular/EuRoC_TimeStamps/MH04.txt` | `examples/Stereo/EuRoC_TimeStamps/MH04.txt` | `examples/GroundTruth/EuRoC/MH_04_difficult.tum` |
| `V2_03_difficult` | `/mnt/d/data_set/EuRoc Wav Dataset/vicon_room2/V2_03_difficult` | `examples/Monocular/EuRoC_TimeStamps/V203.txt` | `examples/Stereo/EuRoC_TimeStamps/V203.txt` | `examples/GroundTruth/EuRoC/V2_03_difficult.tum` |

#### MH_01_easy 单目

运行命令：

```bash
mkdir -p experiment_logs/Monocular/EuRoC/MH_01_easy
XFEAT_DEVICE=auto \
XFEAT_CUDA_DEVICE=0 \
XFEAT_DIAG_FEATURE_DISTRIBUTION=1 \
XFEAT_DIAG_INTERVAL=30 \
XFEAT_LIGHTGLUE_WEIGHT=./weights/xfeat_lighterglue_matcher_cpp.pt \
./examples/Monocular/mono_euroc \
  Vocabulary/ORBvoc.txt \
  examples/Monocular/EuRoC.yaml \
  "/mnt/d/data_set/EuRoc Wav Dataset/machine_hall/MH_01_easy" \
  examples/Monocular/EuRoC_TimeStamps/MH01.txt \
  EuRoC_MH_01_easy_mono \
  2>&1 | tee experiment_logs/Monocular/EuRoC/MH_01_easy/run_EuRoC_MH_01_easy_mono.log
```

格式转化命令：

```bash
python convert_euroc_to_tum.py \
  --est-traj f_EuRoC_MH_01_easy_mono.txt \
  --out-est experiment_logs/Monocular/EuRoC/MH_01_easy/CameraTrajectory_EuRoC_MH_01_easy_mono.tum \
  --timestamp-unit ns
```

轨迹评估命令：

```bash
evo_ape tum examples/GroundTruth/EuRoC/MH_01_easy.tum experiment_logs/Monocular/EuRoC/MH_01_easy/CameraTrajectory_EuRoC_MH_01_easy_mono.tum -r trans_part -as -p
evo_traj tum experiment_logs/Monocular/EuRoC/MH_01_easy/CameraTrajectory_EuRoC_MH_01_easy_mono.tum --ref=examples/GroundTruth/EuRoC/MH_01_easy.tum -as -p
```

#### MH_01_easy 双目

运行命令：

```bash
mkdir -p experiment_logs/Stereo/EuRoC/MH_01_easy
XFEAT_DEVICE=auto \
XFEAT_CUDA_DEVICE=0 \
XFEAT_DIAG_FEATURE_DISTRIBUTION=1 \
XFEAT_DIAG_INTERVAL=30 \
XFEAT_LIGHTGLUE_WEIGHT=./weights/xfeat_lighterglue_matcher_cpp.pt \
XFEAT_USE_LIGHTGLUE_MOTION=1 \
XFEAT_LG_MOTION_POLICY=projection_first \
XFEAT_LG_MOTION_USE_PROJ_GATE=1 \
./examples/Stereo/stereo_euroc \
  Vocabulary/ORBvoc.txt \
  examples/Stereo/EuRoC.yaml \
  "/mnt/d/data_set/EuRoc Wav Dataset/machine_hall/MH_01_easy/mav0" \
  examples/Stereo/EuRoC_TimeStamps/MH01.txt \
  EuRoC_MH_01_easy_stereo \
  2>&1 | tee experiment_logs/Stereo/EuRoC/MH_01_easy/run_EuRoC_MH_01_easy_stereo.log
```

格式转化命令：

```bash
python convert_euroc_to_tum.py \
  --est-traj f_EuRoC_MH_01_easy_stereo.txt \
  --out-est experiment_logs/Stereo/EuRoC/MH_01_easy/CameraTrajectory_EuRoC_MH_01_easy_stereo.tum \
  --timestamp-unit ns
```

轨迹评估命令：

```bash
evo_ape tum examples/GroundTruth/EuRoC/MH_01_easy.tum experiment_logs/Stereo/EuRoC/MH_01_easy/CameraTrajectory_EuRoC_MH_01_easy_stereo.tum -r trans_part -a -p
evo_traj tum experiment_logs/Stereo/EuRoC/MH_01_easy/CameraTrajectory_EuRoC_MH_01_easy_stereo.tum --ref=examples/GroundTruth/EuRoC/MH_01_easy.tum -a -p
```

#### MH_04_difficult 单目

运行命令：

```bash
mkdir -p experiment_logs/Monocular/EuRoC/MH_04_difficult
XFEAT_DEVICE=auto \
XFEAT_CUDA_DEVICE=0 \
XFEAT_DIAG_FEATURE_DISTRIBUTION=1 \
XFEAT_DIAG_INTERVAL=30 \
XFEAT_LIGHTGLUE_WEIGHT=./weights/xfeat_lighterglue_matcher_cpp.pt \
./examples/Monocular/mono_euroc \
  Vocabulary/ORBvoc.txt \
  examples/Monocular/EuRoC.yaml \
  "/mnt/d/data_set/EuRoc Wav Dataset/machine_hall/MH_04_difficult" \
  examples/Monocular/EuRoC_TimeStamps/MH04.txt \
  EuRoC_MH_04_difficult_mono \
  2>&1 | tee experiment_logs/Monocular/EuRoC/MH_04_difficult/run_EuRoC_MH_04_difficult_mono.log
```

格式转化命令：

```bash
python convert_euroc_to_tum.py \
  --est-traj f_EuRoC_MH_04_difficult_mono.txt \
  --out-est experiment_logs/Monocular/EuRoC/MH_04_difficult/CameraTrajectory_EuRoC_MH_04_difficult_mono.tum \
  --timestamp-unit ns
```

轨迹评估命令：

```bash
evo_ape tum examples/GroundTruth/EuRoC/MH_04_difficult.tum experiment_logs/Monocular/EuRoC/MH_04_difficult/CameraTrajectory_EuRoC_MH_04_difficult_mono.tum -r trans_part -as -p
evo_traj tum experiment_logs/Monocular/EuRoC/MH_04_difficult/CameraTrajectory_EuRoC_MH_04_difficult_mono.tum --ref=examples/GroundTruth/EuRoC/MH_04_difficult.tum -as -p
```

#### MH_04_difficult 双目

运行命令：

```bash
mkdir -p experiment_logs/Stereo/EuRoC/MH_04_difficult
XFEAT_DEVICE=auto \
XFEAT_CUDA_DEVICE=0 \
XFEAT_DIAG_FEATURE_DISTRIBUTION=1 \
XFEAT_DIAG_INTERVAL=30 \
XFEAT_LIGHTGLUE_WEIGHT=./weights/xfeat_lighterglue_matcher_cpp.pt \
XFEAT_USE_LIGHTGLUE_MOTION=1 \
XFEAT_LG_MOTION_POLICY=projection_first \
XFEAT_LG_MOTION_USE_PROJ_GATE=1 \
./examples/Stereo/stereo_euroc \
  Vocabulary/ORBvoc.txt \
  examples/Stereo/EuRoC.yaml \
  "/mnt/d/data_set/EuRoc Wav Dataset/machine_hall/MH_04_difficult/mav0" \
  examples/Stereo/EuRoC_TimeStamps/MH04.txt \
  EuRoC_MH_04_difficult_stereo \
  2>&1 | tee experiment_logs/Stereo/EuRoC/MH_04_difficult/run_EuRoC_MH_04_difficult_stereo.log
```

格式转化命令：

```bash
python convert_euroc_to_tum.py \
  --est-traj f_EuRoC_MH_04_difficult_stereo.txt \
  --out-est experiment_logs/Stereo/EuRoC/MH_04_difficult/CameraTrajectory_EuRoC_MH_04_difficult_stereo.tum \
  --timestamp-unit ns
```

轨迹评估命令：

```bash
evo_ape tum examples/GroundTruth/EuRoC/MH_04_difficult.tum experiment_logs/Stereo/EuRoC/MH_04_difficult/CameraTrajectory_EuRoC_MH_04_difficult_stereo.tum -r trans_part -a -p
evo_traj tum experiment_logs/Stereo/EuRoC/MH_04_difficult/CameraTrajectory_EuRoC_MH_04_difficult_stereo.tum --ref=examples/GroundTruth/EuRoC/MH_04_difficult.tum -a -p
```

#### V2_03_difficult 单目

运行命令：

```bash
mkdir -p experiment_logs/Monocular/EuRoC/V2_03_difficult
XFEAT_DEVICE=auto \
XFEAT_CUDA_DEVICE=0 \
XFEAT_DIAG_FEATURE_DISTRIBUTION=1 \
XFEAT_DIAG_INTERVAL=30 \
XFEAT_LIGHTGLUE_WEIGHT=./weights/xfeat_lighterglue_matcher_cpp.pt \
./examples/Monocular/mono_euroc \
  Vocabulary/ORBvoc.txt \
  examples/Monocular/EuRoC.yaml \
  "/mnt/d/data_set/EuRoc Wav Dataset/vicon_room2/V2_03_difficult" \
  examples/Monocular/EuRoC_TimeStamps/V203.txt \
  EuRoC_V2_03_difficult_mono \
  2>&1 | tee experiment_logs/Monocular/EuRoC/V2_03_difficult/run_EuRoC_V2_03_difficult_mono.log
```

格式转化命令：

```bash
python convert_euroc_to_tum.py \
  --est-traj f_EuRoC_V2_03_difficult_mono.txt \
  --out-est experiment_logs/Monocular/EuRoC/V2_03_difficult/CameraTrajectory_EuRoC_V2_03_difficult_mono.tum \
  --timestamp-unit ns
```

轨迹评估命令：

```bash
evo_ape tum examples/GroundTruth/EuRoC/V2_03_difficult.tum experiment_logs/Monocular/EuRoC/V2_03_difficult/CameraTrajectory_EuRoC_V2_03_difficult_mono.tum -r trans_part -as -p
evo_traj tum experiment_logs/Monocular/EuRoC/V2_03_difficult/CameraTrajectory_EuRoC_V2_03_difficult_mono.tum --ref=examples/GroundTruth/EuRoC/V2_03_difficult.tum -as -p
```

#### V2_03_difficult 双目

运行命令：

```bash
mkdir -p experiment_logs/Stereo/EuRoC/V2_03_difficult
XFEAT_DEVICE=auto \
XFEAT_CUDA_DEVICE=0 \
XFEAT_DIAG_FEATURE_DISTRIBUTION=1 \
XFEAT_DIAG_INTERVAL=30 \
XFEAT_LIGHTGLUE_WEIGHT=./weights/xfeat_lighterglue_matcher_cpp.pt \
XFEAT_USE_LIGHTGLUE_MOTION=1 \
XFEAT_LG_MOTION_POLICY=projection_first \
XFEAT_LG_MOTION_USE_PROJ_GATE=1 \
./examples/Stereo/stereo_euroc \
  Vocabulary/ORBvoc.txt \
  examples/Stereo/EuRoC.yaml \
  "/mnt/d/data_set/EuRoc Wav Dataset/vicon_room2/V2_03_difficult/mav0" \
  examples/Stereo/EuRoC_TimeStamps/V203.txt \
  EuRoC_V2_03_difficult_stereo \
  2>&1 | tee experiment_logs/Stereo/EuRoC/V2_03_difficult/run_EuRoC_V2_03_difficult_stereo.log
```

格式转化命令：

```bash
python convert_euroc_to_tum.py \
  --est-traj f_EuRoC_V2_03_difficult_stereo.txt \
  --out-est experiment_logs/Stereo/EuRoC/V2_03_difficult/CameraTrajectory_EuRoC_V2_03_difficult_stereo.tum \
  --timestamp-unit ns
```

轨迹评估命令：

```bash
evo_ape tum examples/GroundTruth/EuRoC/V2_03_difficult.tum experiment_logs/Stereo/EuRoC/V2_03_difficult/CameraTrajectory_EuRoC_V2_03_difficult_stereo.tum -r trans_part -a -p
evo_traj tum experiment_logs/Stereo/EuRoC/V2_03_difficult/CameraTrajectory_EuRoC_V2_03_difficult_stereo.tum --ref=examples/GroundTruth/EuRoC/V2_03_difficult.tum -a -p
```

### 1.3 SePT01

已生成的真值文件位于 `examples/GroundTruth/SePT01/SePT01.tum`。

#### 单目

运行命令：

```bash
mkdir -p experiment_logs/Monocular/SePT01/SePT01
XFEAT_DEVICE=auto \
XFEAT_CUDA_DEVICE=0 \
XFEAT_DIAG_FEATURE_DISTRIBUTION=1 \
XFEAT_DIAG_INTERVAL=30 \
XFEAT_LIGHTGLUE_WEIGHT=./weights/xfeat_lighterglue_matcher_cpp.pt \
./examples/Monocular/mono_euroc \
  Vocabulary/ORBvoc.txt \
  examples/Monocular/SePT01_cam0.yaml \
  /mnt/d/data_set/SePT01 \
  examples/Monocular/SePT01_cam0.txt \
  SePT01_mono \
  2>&1 | tee experiment_logs/Monocular/SePT01/SePT01/run_SePT01_mono.log
```

格式转化命令：

```bash
python convert_euroc_to_tum.py \
  --est-traj f_SePT01_mono.txt \
  --out-est experiment_logs/Monocular/SePT01/SePT01/CameraTrajectory_SePT01_mono.tum \
  --timestamp-unit ns
```

轨迹评估命令：

```bash
evo_ape tum examples/GroundTruth/SePT01/SePT01.tum experiment_logs/Monocular/SePT01/SePT01/CameraTrajectory_SePT01_mono.tum -r trans_part -as -p
evo_traj tum experiment_logs/Monocular/SePT01/SePT01/CameraTrajectory_SePT01_mono.tum --ref=examples/GroundTruth/SePT01/SePT01.tum -as -p
```

#### 双目

运行命令：

```bash
mkdir -p experiment_logs/Stereo/SePT/SePT01
XFEAT_DEVICE=auto \
XFEAT_CUDA_DEVICE=0 \
XFEAT_DIAG_FEATURE_DISTRIBUTION=1 \
XFEAT_DIAG_INTERVAL=30 \
XFEAT_LIGHTGLUE_WEIGHT=./weights/xfeat_lighterglue_matcher_cpp.pt \
XFEAT_USE_LIGHTGLUE_MOTION=1 \
XFEAT_LG_MOTION_POLICY=projection_first \
XFEAT_LG_MOTION_USE_PROJ_GATE=1 \
./examples/Stereo/stereo_euroc \
  Vocabulary/ORBvoc.txt \
  examples/Stereo/SePT01.yaml \
  /mnt/d/data_set/SePT01/mav0 \
  examples/Stereo/SePT01_TimeStamps/times.txt \
  SePT01_stereo \
  2>&1 | tee experiment_logs/Stereo/SePT/SePT01/run_SePT01_stereo.log
```

格式转化命令：

```bash
python convert_euroc_to_tum.py \
  --est-traj f_SePT01_stereo.txt \
  --out-est experiment_logs/Stereo/SePT/SePT01/CameraTrajectory_SePT01_stereo.tum \
  --timestamp-unit ns
```

轨迹评估命令：

```bash
evo_ape tum examples/GroundTruth/SePT01/SePT01.tum experiment_logs/Stereo/SePT/SePT01/CameraTrajectory_SePT01_stereo.tum -r trans_part -a -p
evo_traj tum experiment_logs/Stereo/SePT/SePT01/CameraTrajectory_SePT01_stereo.tum --ref=examples/GroundTruth/SePT01/SePT01.tum -a -p
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

### 2.5 LightGlue 匹配

| 变量 | 作用 | 默认 | 范围 | 代码依据 |
|---|---|---|---|---|
| `XFEAT_LIGHTGLUE_WEIGHT` | LightGlue C++ 权重路径 | `./weights/xfeat_lighterglue_matcher_cpp.pt` | — | `src/XFeatLighterGlueMatcher.cc:18-25` |
| `XFEAT_LIGHTGLUE_CONF` | LightGlue 匹配最小置信度 | `0.1` | `[0,1]` | `src/Tracking.cc:4347,4736,5796` |
| `XFEAT_USE_LIGHTGLUE_REF` | TrackReferenceKeyFrame 使用 LightGlue | 关 | — | `src/Tracking.cc:4342-4353` |
| `XFEAT_USE_LIGHTGLUE_MOTION` | TrackWithMotionModel 使用 LightGlue | 关 | — | `src/Tracking.cc:4717-4745` |
| `XFEAT_LG_MOTION_POLICY` | 运动模型 LightGlue 策略；`projection_first`/`proj_first` 先跑投影匹配再按阈值回退到 LightGlue | 默认直接 LightGlue | — | `src/Tracking.cc:1092-1103,4718-4733` |
| `XFEAT_LG_MOTION_FALLBACK_MIN_MATCHES` | `projection_first` 下投影匹配少于该值时回退到 LightGlue | `20` | `[1,1000]` | `src/Tracking.cc:1105-1108,4728-4733` |
| `XFEAT_LG_MOTION_USE_PROJ_GATE` | LightGlue motion 结果再经过投影半径门控 | 关 | — | `src/XFeatLighterGlueMatcher.cc:490-491` |
| `XFEAT_USE_LIGHTGLUE_RELOC` | Relocalization 初始候选匹配使用 LightGlue，低匹配或异常回退 XFeatMatcher | 关 | — | `src/Tracking.cc` |
| `XFEAT_LG_RELOC_FALLBACK_MIN_MATCHES` | Relocalization 中 LightGlue 少于该匹配数时回退 NN | `15` | `[1,1000]` | `src/Tracking.cc` |
| `XFEAT_USE_LIGHTGLUE_TRIANG` | LocalMapping 新关键帧间三角化候选匹配使用 LightGlue，三角化几何验收不变 | 关 | — | `src/LocalMapping.cc` |
| `XFEAT_USE_LIGHTGLUE_LOCALMAP` | LocalMap 匹配启用 LightGlue | 关 | — | `src/Tracking.cc:1072-1075` |
| `XFEAT_LG_MATCH_DEBUG` | LightGlue core 匹配调试输出 | 关 | — | `src/XFeatLighterGlue/core.cpp:146` |

## 3. 备注

- 多数环境变量在函数内以 `static` 方式缓存，进程启动后不再动态修改。
- `XFEAT_DEBUG=1` 是高频日志开关；`XFEAT_DIAG_INTERVAL` 只控制诊断频率，不是全局限流。
