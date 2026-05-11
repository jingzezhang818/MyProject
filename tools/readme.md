# XFeat + LightGlue 权重转换与验证

将官方 PyTorch 权重导出为 C++ 可加载格式，并验证提取器和匹配器联合流程。

## 1. 权重转换

将原始 `xfeat.pt` 和 `xfeat-lighterglue.pt` 分别转为 C++ 格式的 checkpoint：

```bash
./tools/convert_xfeat_lighterglue_weights_cpp \
  /home/zjz/workspace/accelerated_features/weights/xfeat.pt \
  /home/zjz/workspace/accelerated_features/weights/xfeat-lighterglue.pt \
  /tmp/xfeat_cpp.pt \
  /tmp/xfeat_lighterglue_matcher_cpp.pt
```

- 输出：提取器权重 `/tmp/xfeat_cpp.pt` + 匹配器权重 `/tmp/xfeat_lighterglue_matcher_cpp.pt`

## 2. 验证

### 2.1 匹配器独立加载

```bash
./tools/test_xfeat_lighterglue_cpp \
  /tmp/xfeat_lighterglue_matcher_cpp.pt \
  cuda
```

### 2.2 图片级端到端流程

```bash
./tools/test_xfeat_lighterglue_images_cpp \
  /tmp/xfeat_cpp.pt \
  /tmp/xfeat_lighterglue_matcher_cpp.pt \
  /home/zjz/workspace/accelerated_features/assets/test_pic/pic_1.png \
  /home/zjz/workspace/accelerated_features/assets/test_pic/pic_2.png \
  2048 \
  0.1 2>&1 | tee run_cpp_archive.log
```

- 参数说明：`2048` 为最大特征点数，`0.1` 为匹配阈值
