# XFeatSLAM

基于 ORB-SLAM3 的视觉 SLAM 系统，使用 XFeat（类 SuperPoint 轻量网络）替代 ORB 作为前端特征，支持单目/双目模式。

## 特性

- **XFeat 前端**: 轻量 CNN 提取 float 描述子，替代传统 ORB 手工特征
- **OctTree + ANMS 空间均匀化**: 特征点在图像空间均匀分布，避免局部过密
- **独立 XFeatMatcher**: 基于最近邻和投影的匹配器，覆盖初始化和跟踪全路径
- **立体匹配门控**: ratio test + 左右互检过滤歧义匹配，减少误匹配
- **运行时调参**: 全部阈值和开关通过环境变量控制，无需重编译
- **ORB 兼容**: 支持 `USE_ORB=1` 一键切回纯 ORB 前端

## 编译

依赖: OpenCV >= 4.0, Eigen3, Pangolin, Boost (serialization), CUDA (可选, 用于 XFeat 推理加速)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

如果无法加载 ORB Vocabulary，请自行从 [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) 下载 `ORBvoc.txt`。

## 运行

### 单目

```bash
./examples/Monocular/mono_euroc \
  Vocabulary/ORBvoc.txt \
  examples/Monocular/SePT01_cam0.yaml \
  /path/to/dataset \
  examples/Monocular/SePT01_cam0.txt
```

### 双目

```bash
./examples/Stereo/stereo_euroc \
  Vocabulary/ORBvoc.txt \
  examples/Stereo/EuRoC.yaml \
  /path/to/dataset/mav0 \
  examples/Stereo/V1_01_easy_TimeStamps/times.txt \
  output_basename
```

## 环境变量

完整参考见 [command.md](command.md)。常用：

| 变量 | 作用 | 默认 |
|---|---|---|
| `USE_ORB` | 切换到纯 ORB 前端 | 不设置 |
| `XFEAT_DEVICE` | 推理设备 `auto/cuda/cpu` | `auto` |
| `XFEAT_UNIFORM_DISABLE` | 关闭特征空间均匀化 | 不设置（开启） |
| `XFEAT_STEREO_RATIO` | 立体匹配 ratio test 阈值 | `0.80` |
| `XFEAT_STEREO_MUTUAL` | 左右互检开关 | 开启 |
| `XFEAT_STEREO_DESC_TH` | 立体匹配描述子阈值 | `1.35` |
| `XFEAT_STEREO_MIN_DISPARITY` | 最小视差（像素） | `1.0` |
| `XFEAT_DIAG_INTERVAL` | 诊断日志间隔帧数 | `1` |
| `XFEAT_DEBUG` | 全局调试输出 | 关 |

更多阈值和匹配器配置见 [command.md](command.md)。

## 数据集

已适配配置:
- **EuRoC** (单目/双目)
- **SePT01** (单目/双目)
- **Moon_1** (单目)
- **TUM-VI / TUM RGB-D** (单目)
- **KITTI** (单目)

## 轨迹评估

参考 [convert_euroc_to_tum.py](convert_euroc_to_tum.py) 将输出轨迹转为 TUM 格式后使用 evo 工具评估。

## License

GPLv3，同 ORB-SLAM3。
