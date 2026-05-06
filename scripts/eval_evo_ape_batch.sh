#!/usr/bin/env bash

# ==============================================================================
# evo_ape_batch.sh — 批量评估 SLAM 轨迹绝对位姿误差 (APE)
# ==============================================================================
#
# 功能：对多个 run 的轨迹文件，逐个调用 evo_ape 计算 APE 指标，
#       并将各 run 的 rmse/mean/median/std/min/max 汇总到一个 TSV 文件中。
#
# 使用方法：
#   1. 修改下方的【可编辑配置区】，设置真值文件、估计轨迹目录、run 编号等。
#   2. 确保已安装 evo 并在 conda 环境或系统环境中可用。
#   3. 直接运行：
#        bash scripts/eval_evo_ape_batch.sh
#   4. 运行完毕后，查看汇总结果：
#        cat experiment_logs/<序列名>/evo_ape_trans_part/summary.tsv
#
# 环境变量说明（脚本内部使用，无需手动设置）：
#   - USE_ORB: 不设置则默认使用 XFeat 提取器；设为 1 则回退到 ORB 提取器。
#   - XFEAT_DEVICE: 控制 XFeat 推理设备 (auto/cuda/cpu)，默认 auto。
#
# 依赖：
#   - evo (Python 包，用于轨迹评估)
#   - conda (可选，用于在指定环境中运行 evo)
#
# 输出：
#   - <OUT_DIR>/summary.tsv  包含各 run 的 APE 统计汇总表
#
# ==============================================================================

set -euo pipefail

# =============================
# 可编辑配置区
# =============================

# 项目根目录（脚本所在目录的上一级）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 地面真值轨迹文件（TUM 格式）
# 需要根据你的数据集修改为对应的真值文件路径
GT_FILE="$PROJECT_ROOT/ground_truth_tum_SePT01.txt"

# 估计轨迹所在的目录
# 该目录下应包含 CameraTrajectory_run<id>_tum.txt 格式的轨迹文件
EST_DIR="$PROJECT_ROOT/experiment_logs/SePT01_v01"

# 要评估的 run 编号列表，可根据需要增删
# 例如：RUN_IDS=(01 02 03 04 05 06 07 08 09 10)
RUN_IDS=(01 02 03 04 05)

# ---------- evo_ape 参数 ----------

# 位姿关系类型：trans_part 表示仅评估平移部分的误差
# 其他可选值：full（平移+旋转）、rot_part（仅旋转）、angle_deg（角度差）、angle_rad（弧度差）
POSE_RELATION="trans_part"

# 是否进行 Umeyama 对齐（1=是, 0=否）
# 对齐可以消除轨迹之间的全局刚性变换差异（旋转+平移）
ALIGN=1

# 是否进行尺度校正（1=是, 0=否）
# 单目 SLAM 的轨迹存在尺度不确定性，开启此项进行尺度对齐
CORRECT_SCALE=1

# 遇到错误时是否停止（1=停止, 0=继续评估下一个 run）
STOP_ON_ERROR=1

# ---------- 运行环境 ----------

# 是否通过 conda 环境运行 evo（1=是, 0=否，使用系统 Python）
USE_CONDA=1

# conda 环境名称（仅当 USE_CONDA=1 时生效）
CONDA_ENV_NAME="xfeat-gpu"

# ---------- 输出设置 ----------

# 输出目录：脚本只生成 summary.tsv 汇总文件，不保存图片
OUT_DIR="$EST_DIR/evo_ape_${POSE_RELATION}"

# 汇总文件完整路径
SUMMARY_FILE="$OUT_DIR/summary.tsv"

# =============================
# 说明：如何手动生成多组数据对比图
# =============================
# 以下命令供手动执行，用于生成多 run 的对比可视化图（raw/stats/histogram/box）。
#
# 步骤 1：为每个 run 单独保存评估结果（.zip 文件）
#   for i in 01 02 03 04 05; do
#     conda run -n xfeat-gpu evo_ape tum \
#       ground_truth_tum_SePT01.txt \
#       experiment_logs/SePT01/CameraTrajectory_run${i}_tum.txt \
#       -r trans_part -a -s \
#       --save_results experiment_logs/SePT01/evo_ape_trans_part/run${i}.zip \
#       -c scripts/evo_headless_config.json
#   done
#
# 步骤 2：使用 evo_res 将多个结果合并绘制对比图
#   conda run -n xfeat-gpu evo_res --use_filenames \
#     --save_plot experiment_logs/SePT01/evo_ape_trans_part/compare_runs.png \
#     experiment_logs/SePT01/evo_ape_trans_part/run01.zip \
#     experiment_logs/SePT01/evo_ape_trans_part/run02.zip \
#     experiment_logs/SePT01/evo_ape_trans_part/run03.zip \
#     experiment_logs/SePT01/evo_ape_trans_part/run04.zip \
#     experiment_logs/SePT01/evo_ape_trans_part/run05.zip \
#     -c scripts/evo_headless_config.json
#
# evo_res 会生成 4 张图：raw（原始误差曲线）、stats（统计量）、histogram（分布直方图）、box（箱线图）

# =============================
# 脚本正文：无需修改以下内容
# =============================

# 创建输出目录（如不存在自动创建）
mkdir -p "$OUT_DIR"

# 写入 TSV 表头
# 列说明：run=运行编号, exit_code=evo_ape 退出码(0正常), rmse/mean/median/std/min/max=APE 统计指标(米)
printf "run\texit_code\trmse\tmean\tmedian\tstd\tmin\tmax\n" > "$SUMMARY_FILE"

# 检查真值文件是否存在
if [ ! -f "$GT_FILE" ]; then
  echo "GT file not found: $GT_FILE"
  exit 1
fi

# -------------------------------------------------------------------
# 函数：run_evo_ape — 调用 evo_ape 对单条轨迹进行评估
# 参数：$1 = 估计轨迹文件路径
# 返回：evo_ape 的标准输出（包含 APE 统计表）
# -------------------------------------------------------------------
run_evo_ape() {
  local est_file="$1"

  # 构建 evo_ape 命令
  # evo_ape tum <真值> <估计> -r <位姿关系>
  local cmd=(evo_ape tum "$GT_FILE" "$est_file" -r "$POSE_RELATION")

  # 是否添加 -a（对齐）标志
  if [ "$ALIGN" -eq 1 ]; then
    cmd+=(-a)
  fi

  # 是否添加 -s（尺度校正）标志
  if [ "$CORRECT_SCALE" -eq 1 ]; then
    cmd+=(-s)
  fi

  # 选择运行环境：conda 或系统 Python
  # PYTHONNOUSERSITE=1 避免加载用户 site-packages 导致冲突
  if [ "$USE_CONDA" -eq 1 ]; then
    conda run -n "$CONDA_ENV_NAME" env PYTHONNOUSERSITE=1 "${cmd[@]}"
  else
    env PYTHONNOUSERSITE=1 "${cmd[@]}"
  fi
}

# -------------------------------------------------------------------
# 函数：extract_metric — 从 evo_ape 输出中提取指定指标的值
# 参数：$1 = evo_ape 的完整输出文本
#       $2 = 指标名称（rmse, mean, median, std, min, max）
# 返回：指标数值或空
# -------------------------------------------------------------------
extract_metric() {
  local text="$1"
  local key="$2"
  # evo_ape 输出中每行格式为 "<指标名>  <值>"，用 awk 匹配指标名并取第二列
  printf "%s\n" "$text" | awk -v k="$key" '$1==k{print $2; exit}'
}

# -------------------------------------------------------------------
# 主循环：遍历每个 run，执行评估并汇总
# -------------------------------------------------------------------
for run_id in "${RUN_IDS[@]}"; do
  # 拼接估计轨迹文件路径
  est_file="$EST_DIR/CameraTrajectory_run${run_id}_tum.txt"

  # 检查轨迹文件是否存在
  if [ ! -f "$est_file" ]; then
    echo "Missing EST file: $est_file"
    # 文件缺失时写入 NA 标记
    printf "%s\t%s\tNA\tNA\tNA\tNA\tNA\tNA\n" "$run_id" "missing" >> "$SUMMARY_FILE"
    if [ "$STOP_ON_ERROR" -eq 1 ]; then
      exit 1
    fi
    continue
  fi

  echo "========== Evaluating run${run_id} =========="

  # 执行 evo_ape（关闭 strict mode 以捕获非零退出码）
  set +e
  output="$(run_evo_ape "$est_file" 2>&1)"
  status=$?
  set -e

  # 打印完整输出（供终端查看）
  echo "$output"

  # 从输出中提取各项指标
  rmse="$(extract_metric "$output" "rmse")"
  mean="$(extract_metric "$output" "mean")"
  median="$(extract_metric "$output" "median")"
  std="$(extract_metric "$output" "std")"
  min_v="$(extract_metric "$output" "min")"
  max_v="$(extract_metric "$output" "max")"

  # 若某指标提取失败，则以 NA 占位
  [ -n "$rmse" ] || rmse="NA"
  [ -n "$mean" ] || mean="NA"
  [ -n "$median" ] || median="NA"
  [ -n "$std" ] || std="NA"
  [ -n "$min_v" ] || min_v="NA"
  [ -n "$max_v" ] || max_v="NA"

  # 将本次 run 的结果追加写入汇总文件（TSV 一行）
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$run_id" "$status" "$rmse" "$mean" "$median" "$std" "$min_v" "$max_v" >> "$SUMMARY_FILE"

  # 如果 evo_ape 执行失败且开启了遇错停止，则终止脚本
  if [ "$status" -ne 0 ] && [ "$STOP_ON_ERROR" -eq 1 ]; then
    echo "Stopping because STOP_ON_ERROR=1."
    exit "$status"
  fi
done

echo "Done. Summary: $SUMMARY_FILE"
