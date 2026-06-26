# XFeat 专用词袋工作计划

本文档记录进入 XFeat vocabulary 阶段后的设计边界、参考实现结论、分阶段代码计划和实验验收方法。目标是解决当前 MH04 等 medium/difficult 场景中 loop 候选召回和排序不稳定的问题，同时避免继续在 loop 几何验证、LocalMap gate、KF guard 等已经反复验证过的方向里叠加复杂逻辑。

## 背景问题

当前 XFeat + LightGlue 前端已经能在 tracking、triangulation、relocalization、loop candidate validation 中提供 2D-2D 匹配。但 loop 候选召回仍然依赖 ORB-SLAM3 原生词袋：

```text
ORBVocabulary = DBoW2::TemplatedVocabulary<FORB>
Frame/KeyFrame::ComputeBoW() -> mpORBvocabulary->transform(mDescriptors, mBowVec, mFeatVec, 4)
KeyFrameDatabase::DetectNBestCandidates() -> ORB vocabulary score / inverted file
LoopClosing -> LightGlue 验证候选
```

这意味着 XFeat 模式下存在一个结构不匹配：

```text
前端描述子: XFeat 64-D CV_32F
候选召回词袋: ORB binary vocabulary / DBoW2 FORB
几何验证: XFeat LightGlue + Sim3
```

从多次 MH04 日志看，loop 对最终 RMSE 影响很大，且不稳定性主要体现在：

- 有的 run 能较早得到 `candidate_rank=0`、`sim3_inliers` 高的 loop，RMSE 可以到 `0.10~0.13`。
- 有的 run 需要 top10/top20 才出现较好的候选，或 first loop 的 `sim3_inliers` 偏低，RMSE 回到 `0.19~0.28`。
- 增加 `XFEAT_LOOP_CANDIDATES`、LG capped expansion 可以改善几何验证覆盖，但不能从根上解决 ORBvoc 对 XFeat 场景的候选排序问题。

因此，下一阶段目标不是继续扩大 loop LG 计算量，而是让 place recognition 的描述子语义与 XFeat 前端一致。

## 参考实现结论

### SuperPoint_SLAM

SuperPoint_SLAM 的做法：

- 使用 `DBoW3::Vocabulary` 替换 ORB-SLAM2 原生 DBoW2 vocabulary。
- SuperPoint 描述子是 `CV_32F`、256 维。
- `Frame::ComputeBoW()` 和 `KeyFrame::ComputeBoW()` 直接把 float 描述子送入 DBoW3。
- `KeyFrameDatabase` 使用 DBoW3 `BowVector / FeatureVector` 做 inverted file、score 和 relocalization/loop retrieval。

可借鉴点：

- 深度特征应使用 DBoW3 或同类支持 float descriptor 的 vocabulary。
- 可以保持 ORB-SLAM 的 inverted file、common words、covisibility score accumulation 框架。
- 不需要把 float descriptor 强行伪装成 ORB binary descriptor。

不直接照搬点：

- SuperPoint 是 256-D，XFeat 是 64-D，词袋必须重新训练。
- SuperPoint_SLAM 是 ORB-SLAM2 结构，当前项目是 ORB-SLAM3，多地图、merge candidates、LoopClosing 结构不同。

### Rover-SLAM

Rover-SLAM 的做法：

- 使用 `DBoW3::Vocabulary` 作为 `SPVocabulary`。
- 词袋文件为 `voc_binary_tartan_8u_6.yml.gz`。
- SuperPoint float 描述子进词袋前做二值化：

```cpp
cv::threshold(desc.row(i), tmp, 0, 1, cv::THRESH_BINARY);
```

- 二值化后仍然是 256 维，只是每一维从 float 变成 `0/1` 的 `CV_8U`。
- 词袋结构为 `k=10, L=6`，即每个节点最多 10 个子节点，树深度 6，理论最多 `10^6` 个 visual words。

可借鉴点：

- ORB-SLAM3 结构下可以并行维护深度特征词袋字段，例如 `mBow3Vec / mFeat3Vec`。
- `DetectNBestCandidates_sp()` 的结构说明：可以把 ORB-SLAM3 的 candidate retrieval 迁到 DBoW3，同时保留 covisibility score accumulation。
- DBoW3 的 `FeatureVector` 仍可用于 BoW 分桶后的候选匹配，但我们当前 loop 已经有 LG，第一阶段无需恢复 BoW descriptor matching。

不直接照搬点：

- Rover 的词袋是 SuperPoint 256-D binary vocabulary，不能用于 XFeat 64-D float descriptor。
- Rover 二值化是工程选择，不代表 XFeat 应该默认二值化。
- Rover 代码存在新旧 `mBowVec / mBow3Vec` 混用痕迹，我们接入时应避免半替换。

## 描述子表示选择

第一阶段采用：

```text
XFeat 64-D CV_32F descriptor
-> DBoW3 float vocabulary
```

暂不采用：

```text
XFeat 64-D float descriptor
-> threshold / sign binarization
-> DBoW3 binary vocabulary
```

原因：

- 当前 XFeat matcher、projection matching、triangulation、LightGlue 输入都基于 float descriptor 语义。
- XFeat 只有 64 维，比 SuperPoint 256 维更紧凑，二值化会丢掉幅值信息，风险更高。
- DBoW3 原生支持 floating point descriptor。
- 先保留 float 能让 place recognition 与当前前端语义一致，便于归因。

二值化只作为第二阶段 ablation：

- 如果 float vocabulary 检索慢、内存大、或 recall/rank 不理想，再训练 `XFeat-binary vocabulary` 对比。
- 二值化方式不能只凭 `>0` 默认定死，应比较 sign、min-max/uint8 quantization、PCA/ITQ 等方案。

## 总体目标

核心目标：

- 提高 XFeat 模式下 loop candidate 的召回和排序稳定性。
- 降低对 `XFEAT_LOOP_CANDIDATES=10/20` 的依赖。
- 让高质量 loop 更早进入 LG + Sim3 验证。
- 保持当前 loop 后端：LightGlue candidate validation、Sim3、CorrectLoop、GBA 不做大改。

非目标：

- 不改 LocalMap projection。
- 不改 KF 插入策略。
- 不改 MapPoint 创建和 culling。
- 不继续增加 loop LG expansion 窗口。
- 不直接替换为 Rover 的 SuperPoint vocabulary。
- 不在第一阶段训练/接入 binary XFeat vocabulary。

## 阶段 0：离线数据与评估协议

目标是在改主系统前，先准备可重复评估 vocabulary 的数据和指标。

### 数据来源

训练数据建议不要只用 MH04：

```text
EuRoC: MH01, MH02, MH03, MH05, V101, V102, V201, V202
验证/重点测试: MH04
可扩展: TUM-VI / 自有序列
```

原因：

- 只用 MH04 训练会污染验证，不利于判断泛化。
- EuRoC 场景纹理、运动、光照变化有限，最好混入多个 machine hall 和 Vicon room。

### 描述子导出

新增离线导出工具，输出每帧或抽样帧的 XFeat descriptors：

```text
tools/export_xfeat_descriptors
```

输入：

- 图像目录或 EuRoC mav0 路径。
- timestamp 文件。
- 相机 yaml。
- XFeat 权重路径。
- 采样间隔，例如每 3 或 5 帧导出一次。

输出：

```text
vocabulary/xfeat/train/*.yml.gz 或 .bin
每个样本包含:
- frame_id / timestamp
- image path
- keypoints: x, y, response, octave
- descriptors: N x 64 CV_32F
```

导出约束：

- 使用当前 SLAM 中同一套 `XFextractor` 参数。
- 默认 `nLevels=1`，保持 `octave=0` 语义。
- 不做二值化。
- 保留 descriptor row 顺序，不做额外归一化，除非先确认 XFeat 输出未归一化。

### 离线评估指标

训练前后都需要能评估：

- `recall@K`: K=3/6/10/20。
- true loop candidate 的平均 rank。
- query BowVec 非空率。
- common words 分布。
- candidate score margin：top1/top2、top1/topK。
- 与当前 ORBvoc 检索结果对比。

如果可以利用轨迹或近似位姿，定义 true loop：

```text
时间间隔 > 30s
空间距离 < 1.5m 或 2.0m
视角差不过大
```

如果先不接 ground truth，则使用已发生 `[LoopLG][Accept]` 的历史日志作为 weak label，评估这些 accepted candidate 在新 vocabulary 中能排到第几。

## 阶段 1：引入 DBoW3 与 XFeatVocabulary

目标是让工程具备 XFeat float vocabulary 的加载和 transform 能力，但默认不改变 ORB 模式。

### 代码改动

引入 DBoW3：

```text
thirdparty/DBoW3
CMakeLists.txt 链接 libDBoW3
build 脚本增加 DBoW3 构建
```

新增头文件：

```cpp
// include/XFeatVocabulary.h
typedef DBoW3::Vocabulary XFeatVocabulary;
```

新增 System 成员：

```cpp
XFeatVocabulary* mpXFeatVocabulary = nullptr;
```

新增环境变量或配置：

```text
XFEAT_USE_XFEAT_VOCAB=1
XFEAT_VOCAB_PATH=Vocabulary/xfeat_float_k10_l6.yml.gz
```

加载策略：

- 默认仍加载原 `Vocabulary/ORBvoc.txt`。
- 当 `XFEAT_USE_XFEAT_VOCAB=1` 时，额外加载 XFeat vocabulary。
- 如果 XFeat vocabulary 加载失败，应明确报错并退出，不能静默回退到 ORBvoc，否则实验不可归因。

### 验收

- ORB 模式不设置 `XFEAT_USE_XFEAT_VOCAB` 时行为不变。
- XFeat 模式设置 `XFEAT_USE_XFEAT_VOCAB=1` 时能加载 DBoW3 vocabulary。
- 输出一次简洁日志：

```text
[XFeatVocab] loaded path=... size=... k=... L=... descriptor=float32x64
```

## 阶段 2：Frame/KeyFrame 增加 XFeat Bow 字段

目标是避免复用 ORB 的 `mBowVec / mFeatVec`，防止类型和语义混乱。

### 新增字段

在 `Frame` 和 `KeyFrame` 中增加：

```cpp
DBoW3::BowVector mXFeatBowVec;
DBoW3::FeatureVector mXFeatFeatVec;
```

新增函数：

```cpp
void ComputeXFeatBoW();
```

函数行为：

- 仅在 `mDescriptors.type() == CV_32F && mDescriptors.cols == 64` 时执行。
- 使用 `Converter::toDescriptorVector(mDescriptors)` 或 DBoW3 支持的 `cv::Mat` 输入。
- `levelsup` 第一版建议使用 `0` 和 `4` 做 A/B：
  - `0`: 直接按 leaf word 建 direct index，区分度高。
  - `4`: 更接近 ORB-SLAM 原始 `FeatureVector` 分桶方式，匹配候选更多但更粗。
- loop retrieval 主要使用 `BowVector` score；`FeatureVector` 第一阶段只保留以备后续 BoW-constrained matching。

### KeyFrame 序列化

第一阶段建议不把 `mXFeatBowVec/mXFeatFeatVec` 纳入 Atlas 持久化，或仅在确认 DBoW3 序列化稳定后再加。

如果 Atlas load 需要恢复：

- load 后调用 `SetXFeatVocabulary()`。
- 对已有 KeyFrame 延迟 `ComputeXFeatBoW()`。

### 验收

新增诊断：

```text
[XFeatVocab][BoW] kf=... desc_rows=... desc_cols=64 words=... feat_nodes=...
```

诊断只在：

```text
XFEAT_DEBUG=1 && frame_or_kf_id % XFEAT_DIAG_INTERVAL == 0
```

## 阶段 3：KeyFrameDatabase 增加 XFeat inverted file

目标是并行维护 XFeat 专用倒排索引，不污染 ORB inverted file。

### 新增成员

```cpp
const XFeatVocabulary* mpXFeatVoc = nullptr;
std::vector<std::list<KeyFrame*>> mvXFeatInvertedFile;
```

新增接口：

```cpp
void SetXFeatVocabulary(XFeatVocabulary* pVoc);
void addXFeat(KeyFrame* pKF);
void eraseXFeat(KeyFrame* pKF);
void clearXFeat();
void DetectNBestCandidatesXFeat(KeyFrame* pKF,
                                std::vector<KeyFrame*>& vpLoopCand,
                                std::vector<KeyFrame*>& vpMergeCand,
                                int nNumCandidates);
```

### 检索逻辑

参考 ORB-SLAM3 `DetectNBestCandidates()`，但输入换成：

```text
pKF->mXFeatBowVec
pKFi->mXFeatBowVec
mvXFeatInvertedFile
mpXFeatVoc->score(...)
```

保留原有结构：

- 排除当前 KF 的 connected KFs。
- 统计 common words。
- 用 `maxCommonWords * 0.8` 做初筛。
- 计算 vocabulary score。
- covisibility accumulation。
- 取 top N。

第一阶段不要降低几何验证门槛。

### add/erase 时机

KeyFrame 插入数据库时：

```text
ComputeBoW()          -> ORB/legacy path
ComputeXFeatBoW()    -> XFeat path, if enabled and descriptor is XFeat
KeyFrameDatabase::add()
KeyFrameDatabase::addXFeat()
```

如果 `XFEAT_USE_XFEAT_VOCAB=0`，不调用 `addXFeat()`。

### 验收

日志：

```text
[XFeatVocab][DB] add kf=... words=... inverted_size=...
[XFeatVocab][Candidates] current_kf=... sharing=... scored=... returned=... top_score=...
```

日志受 `XFEAT_DEBUG / XFEAT_DIAG_INTERVAL` 控制。

## 阶段 4：LoopClosing 使用 XFeat vocabulary 候选

目标是只替换 loop candidate retrieval，不动后端几何验证。

### 代码路径

在 `LoopClosing::NewDetectCommonRegions()` 中：

```text
if XFEAT_USE_XFEAT_VOCAB=1 && current KF has XFeat Bow:
    mpKeyFrameDB->DetectNBestCandidatesXFeat(...)
else:
    mpKeyFrameDB->DetectNBestCandidates(...)
```

后续保持当前代码：

- `XFEAT_USE_LIGHTGLUE_LOOP=1`
- `XFEAT_LOOP_CANDIDATES`
- `XFEAT_LOOP_LG_EXPAND`
- `Sim3Solver`
- `SearchByProjection`
- `OptimizeSim3`
- `CorrectLoop`
- GBA

### 与现有 loop LG 参数关系

XFeat vocabulary 接入后，建议先降低 loop LG 负载：

```text
XFEAT_LOOP_CANDIDATES=6
XFEAT_LOOP_LG_EXPAND=1
XFEAT_LOOP_LG_EXPAND_COVISIBLES=1
XFEAT_LOOP_LG_MAX_CALLS_PER_KF=12
```

如果 XFeat vocabulary 有效，应表现为：

- top6 中出现好候选的概率上升。
- `candidate_rank` 更靠前。
- `sim3_inliers` 更稳定。
- 不需要长期依赖 top20。

### 日志

保留现有简洁 loop 日志：

- `[LoopLG][Accept]`
- `[LoopLG][Summary]`
- `[LoopLG][Fallback]`

新增候选来源字段：

```text
candidate_source=orb_vocab/xfeat_vocab
```

`[LoopLG][Accept]` 不受 `XFEAT_DIAG_INTERVAL` 限制，必须保留。

## 阶段 5：训练 XFeat float vocabulary

### 训练工具

新增：

```text
tools/train_xfeat_vocabulary
```

输入：

```text
--descriptor-list vocabulary/xfeat/train/list.txt
--output Vocabulary/xfeat_float_k10_l6.yml.gz
--k 10
--L 6
--weighting TF_IDF
--scoring L1_NORM
--max-features-per-image 800
--max-images 5000
```

DBoW3 默认构造：

```cpp
DBoW3::Vocabulary voc(k, L, DBoW3::TF_IDF, DBoW3::L1_NORM);
voc.create(training_features);
voc.save(output);
```

### 参数候选

第一批建议训练：

```text
float_k10_l5
float_k10_l6
float_k8_l6
```

原因：

- `k=10, L=6` 对齐 Rover 的较大词袋，区分度强，但文件和检索成本更高。
- `k=10, L=5` 较小，适合先验证运行链路。
- `k=8, L=6` 降低分支数，可能更稳但区分度略低。

不要一次训练太多版本。第一轮最多 3 个。

### 采样策略

每帧 XFeat 点可能较多，训练不能直接全量堆入。

建议：

- 每个训练序列每 3-5 帧采样一帧。
- 每帧最多采样 800 个 descriptors。
- 采样时按 response + 空间均匀混合，不只取最高 response。
- 保证 close/mid/far 和不同图像区域都有覆盖。

### 文件命名

```text
Vocabulary/xfeat_float_euroc_k10_l6.yml.gz
Vocabulary/xfeat_float_euroc_k10_l5.yml.gz
Vocabulary/xfeat_float_euroc_k8_l6.yml.gz
```

如果未来引入二值化：

```text
Vocabulary/xfeat_binary_sign_euroc_k10_l6.yml.gz
Vocabulary/xfeat_uint8_norm_euroc_k10_l6.yml.gz
```

## 阶段 6：离线候选排序评估

在进入完整 SLAM 前，先跑 offline retrieval：

```text
tools/eval_xfeat_vocabulary_retrieval
```

对比：

```text
ORBvoc current
XFeat float k10_l5
XFeat float k10_l6
XFeat float k8_l6
```

输出：

```text
sequence
query_kf
known_good_loop_kf
orb_rank
xfeat_rank
orb_score
xfeat_score
recall@3 / recall@6 / recall@10 / recall@20
```

验收标准：

- 在 MH04 历史 good loop 对上，XFeat vocabulary 的 rank 应明显优于 ORBvoc。
- `recall@6` 应接近或超过 ORBvoc `recall@20`。
- score margin 不能过低，否则会导致 topK 排序仍随机。

如果 offline retrieval 没有明显改善，不要急着接入 SLAM 主线，应先回到训练数据、k/L、descriptor normalization 检查。

## 阶段 7：完整 SLAM A/B

### baseline 命令

保持当前低负载 loop 配置：

```bash
XFEAT_DEBUG=1 \
XFEAT_DIAG_FEATURE_DISTRIBUTION=1 \
XFEAT_DIAG_INTERVAL=30 \
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
  2>&1 | tee experiment_logs/Stereo/EuRoC/MH_04_difficult/run_MH04_orbvoc_loop_baseline.log
```

### XFeat vocabulary 命令

```bash
XFEAT_DEBUG=1 \
XFEAT_DIAG_FEATURE_DISTRIBUTION=1 \
XFEAT_DIAG_INTERVAL=30 \
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
XFEAT_USE_XFEAT_VOCAB=1 \
XFEAT_VOCAB_PATH=Vocabulary/xfeat_float_euroc_k10_l6.yml.gz \
./examples/Stereo/stereo_euroc \
  Vocabulary/ORBvoc.txt \
  examples/Stereo/EuRoC.yaml \
  "/mnt/d/data_set/EuRoc Wav Dataset/machine_hall/MH_04_difficult/mav0" \
  examples/Stereo/EuRoC_TimeStamps/MH04.txt \
  EuRoC_MH_04_difficult_stereo \
  2>&1 | tee experiment_logs/Stereo/EuRoC/MH_04_difficult/run_MH04_xfeat_vocab_k10_l6.log
```

### 必须分组统计

每个配置至少跑 3 次，记录：

- RMSE median / best / worst。
- 是否 `[LoopLG][Accept]`。
- 第一次 accept 的 `candidate_rank`。
- 第一次 accept 的 `sim3_inliers`。
- 是否 `*Loop detected`。
- 是否 `mpLoopCloser is running GBA`。
- 是否 shutdown timeout。
- KF 数量和 MapPoint 数量。
- loop 前后的轨迹误差段变化。
- GPU 占用和总运行时间。

不要把不同 loop 状态混在一起比较：

```text
no loop
loop accepted but no GBA
loop accepted + GBA finished
loop accepted + GBA shutdown timeout
tracking lost
```

## 成功标准

第一阶段成功不要求单次 RMSE 立刻达到 ORB-SLAM3 最优 `0.085`，而是要求 loop 稳定性提高：

- MH04 top6 中稳定出现高质量 candidate。
- `candidate_rank` 分布更靠前，最好常见 `0~2`。
- `sim3_inliers` 分布更高，弱 loop 减少。
- top20 不再明显优于 top6。
- RMSE median 稳定下降，而不是只出现单次 `0.10`。
- GPU 负载不显著高于当前低负载 loop 配置。

如果 XFeat vocabulary 只改善 recall 但 RMSE 仍大幅波动，下一步再看：

- loop accepted 的 Sim3 初始质量是否仍弱。
- loop 前 local map 状态是否让 correction 难以收敛。
- 是否需要 XFeat vocabulary + stricter loop accept policy。

## 风险与回退

### 风险 1：DBoW3 引入影响 ORB 模式

规避：

- XFeat vocabulary 独立开关。
- ORB 模式仍走 DBoW2/ORBvoc。
- 不替换 `mBowVec/mFeatVec` 的原语义。

### 风险 2：词袋训练过拟合 EuRoC

规避：

- 训练不包含 MH04 或至少把 MH04 留作验证。
- 加入 Vicon room 和 Machine Hall 多序列。
- 后续用 TUM-VI 或自有数据检查泛化。

### 风险 3：float vocabulary 文件大、加载慢

规避：

- 先训练 `k10_l5` 小词袋验证链路。
- 再比较 `k10_l6`。
- 如果负载不可接受，再进入 binary/uint8 ablation。

### 风险 4：候选召回变好但 false positive 增加

规避：

- 不降低 Sim3 / projection / OptimizeSim3 门槛。
- 保留 LightGlue + Sim3 几何验证。
- `[LoopLG][Accept]` 必须记录 candidate source、rank、sim3_inliers。

## 当前推荐执行顺序

1. 引入 DBoW3，但不改变默认行为。
2. 新增 XFeat descriptor export tool。
3. 训练小型 `float_k10_l5` vocabulary，跑 offline retrieval。
4. 训练 `float_k10_l6 / float_k8_l6`，比较 recall/rank。
5. 增加 `Frame/KeyFrame::ComputeXFeatBoW()` 和 XFeat inverted file。
6. LoopClosing 用 `XFEAT_USE_XFEAT_VOCAB=1` 切换候选召回。
7. MH04 固定低负载 loop 配置跑 3 次，对比 ORBvoc baseline。
8. 若有效，再扩展到 MH03/MH05/V101/V102/V201/V202。
9. 若 float 词袋无效或负载过高，再做 XFeat binary/uint8 vocabulary ablation。

## 决策边界

继续推进 XFeat vocabulary 的条件：

- offline retrieval 明显改善 accepted loop candidate rank。
- SLAM 中 `[LoopLG][Accept]` 的 rank/inliers 更稳定。
- RMSE median 优于 ORBvoc baseline，而不是只提升 best run。

暂停 vocabulary 方向的条件：

- offline recall@K 无明显改善。
- XFeat vocab 召回候选更多但几何验证通过率下降。
- GBA/shutdown 问题显著变多。
- GPU/CPU 负载超过当前 loop 配置太多，且 RMSE median 无收益。

如果暂停，应回到当前 docs 中的 segment-level attribution 工作流，而不是继续扩大 loop LG 计算量。
