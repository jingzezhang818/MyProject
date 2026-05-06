#!/usr/bin/env bash

# ==============================================================================
# repeat_mono_euroc.sh — 单目 SLAM 重复运行脚本
# ==============================================================================
#
# 功能：对同一条序列重复运行 N 次 SLAM 程序，自动收集每次运行的：
#       - 标准输出/错误日志（.log 文件）
#       - 轨迹产物（CameraTrajectory.txt、KeyFrameTrajectory.txt 等）
#       - 运行摘要（退出码、开始/结束时间）
#
# 典型使用场景：
#   - 评估 SLAM 系统的重复性/稳定性（多次运行对比 APE 方差）
#   - 调参后批量跑实验，收集多组轨迹供后续 evo 评估
#   - 配合 eval_evo_ape_batch.sh 使用：先运行本脚本收集轨迹，再批量计算 APE
#
# 使用方法：
#   1. 修改下方的【可编辑实验配置】：
#      - RUNS：重复运行次数
#      - LOG_DIR / LOG_PREFIX：日志输出目录和文件名前缀
#      - BIN / VOCAB / SETTINGS / DATASET_DIR / TIMESTAMP_FILE：SLAM 程序参数
#      - ENV_VARS：需要注入的环境变量（XFeat 调参/调试开关）
#   2. 运行脚本：
#        bash scripts/repeat_mono_euroc.sh
#   3. 运行完毕后查看：
#      - 汇总表：cat experiment_logs/<序列名>/<prefix>_summary.tsv
#      - 单次日志：experiment_logs/<序列名>/<prefix>_<run_id>.log
#      - 轨迹产物：experiment_logs/<序列名>/CameraTrajectory_run<run_id>.txt
#
# 注意事项：
#   - 脚本在项目根目录下执行 SLAM 程序（通过子 shell cd 到 WORKDIR）
#   - 每次运行会在 WORKDIR 下生成 CameraTrajectory.txt 等文件，
#     然后自动复制到 LOG_DIR 并加上 run 编号后缀
#   - 脚本使用 set -u（未定义变量报错）但不使用 set -e（不会因命令失败自动退出），
#     而是通过检查退出码手动控制是否停止
#
# ==============================================================================

set -uo pipefail

# -----------------------------
# 可编辑实验配置
# -----------------------------

# 重复运行次数（至少 1 次）
RUNS=5

# 遇到错误时是否停止（1=任一次运行失败则终止, 0=忽略错误继续下一轮）
STOP_ON_ERROR=1

# 两次运行之间的等待秒数（0=不等待）
# 可用于让 GPU/CPU 降温，或等待文件系统写入完成
SLEEP_BETWEEN_RUNS_SEC=0

# 项目根目录（脚本所在目录的上一级）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 工作目录（SLAM 程序在此目录下执行）
# 默认与项目根目录相同；如果需要在其他路径运行，修改此项
WORKDIR="$PROJECT_ROOT"

# ---------- 日志与产物输出 ----------

# 日志输出目录
LOG_DIR="$PROJECT_ROOT/experiment_logs/SePT01_v01"

# 日志文件名前缀，最终文件名为 <prefix>_<run_id>.log
LOG_PREFIX="run_SePT01_v01"

# ---------- SLAM 程序调用参数 ----------

# 可执行文件路径
BIN="./examples/Monocular/mono_euroc"

# ORB 词袋文件（用于回环检测，XFeat 模式下仍然需要）
VOCAB="Vocabulary/ORBvoc.txt"

# YAML 配置文件（相机内参、ORB/XFeat 提取器参数等）
SETTINGS="examples/Monocular/SePT01_cam0.yaml"

# 数据集根目录
DATASET_DIR="/mnt/d/data_set/SePT01"

# 时间戳文件（记录每帧图像的时间戳，用于 TUM 格式输出）
TIMESTAMP_FILE="examples/Monocular/SePT01_cam0.txt"

# 额外命令行参数（按需添加）
# 例如：EXTRA_ARGS=("--no-viewer" "--log-level=debug")
EXTRA_ARGS=(
  # "--example-arg"
)

# ---------- 环境变量 ----------
# 注入到 SLAM 进程的环境变量，用于控制 XFeat 的行为和调试输出
#
# 常用变量说明：
#   USE_ORB=1              强制使用 ORB 提取器（默认不设置，使用 XFeat）
#   XFEAT_DEVICE=cuda      设置 XFeat 推理设备 (auto/cuda/cpu)
#   XFEAT_DEBUG=1          开启 XFeat 调试输出
#   XFEAT_DIAG_FEATURE_DISTRIBUTION=1  输出每帧特征分布诊断信息
#   XFEAT_DIAG_INTERVAL=10             每隔多少帧输出一次诊断信息
#   XFEAT_MONO_NEAR_ONLY=1             单目初始化仅使用近距离特征点
#   XFEAT_MONO_NEAR_DEPTH_FACTOR=2.5   近距离判定系数（base × factor 以内的点视为近点）
#   XFEAT_UNIFORM_DISABLE=1            禁用 OctTree/ANMS 特征均匀化
#   XFEAT_UNIFORM_GRID_COLS=8          网格均匀化列数
#   XFEAT_UNIFORM_GRID_ROWS=6          网格均匀化行数
#
ENV_VARS=(
   "XFEAT_DEBUG=1"
   "XFEAT_DIAG_FEATURE_DISTRIBUTION=1"
   "XFEAT_DIAG_INTERVAL=10"
   "XFEAT_MONO_NEAR_ONLY=1"
   "XFEAT_MONO_NEAR_DEPTH_FACTOR=2.5"
)

# ---------- 产物保存 ----------

# 是否保存每次运行的产物文件（1=保存, 0=不保存）
# 产物文件在 WORKDIR 下生成，保存到 LOG_DIR 并加 run 编号后缀
SAVE_ARTIFACTS=1

# 需要保存的产物文件列表（相对于 WORKDIR 的路径）
ARTIFACT_FILES=(
  "CameraTrajectory.txt"       # 相机轨迹（TUM 格式）
  "KeyFrameTrajectory.txt"     # 关键帧轨迹
)

# -----------------------------
# 脚本主流程（无需修改）
# -----------------------------

# 创建日志输出目录
mkdir -p "$LOG_DIR"

# 汇总文件路径（TSV 格式）
SUMMARY_FILE="$LOG_DIR/${LOG_PREFIX}_summary.tsv"

# 写入 TSV 表头
# 列说明：run=运行编号, exit_code=退出码(0正常), log_file=日志路径, started_at/ended_at=起止时间
printf "run\texit_code\tlog_file\tstarted_at\tended_at\n" > "$SUMMARY_FILE"

# 拼接完整命令行：BIN + VOCAB + SETTINGS + DATASET_DIR + TIMESTAMP_FILE + EXTRA_ARGS
CMD=("$BIN" "$VOCAB" "$SETTINGS" "$DATASET_DIR" "$TIMESTAMP_FILE" "${EXTRA_ARGS[@]}")

# 打印本次实验的配置概览
echo "Workdir: $WORKDIR"
echo "Runs: $RUNS"
echo "Log dir: $LOG_DIR"
echo "Command: ${CMD[*]}"
if [ "${#ENV_VARS[@]}" -gt 0 ]; then
  echo "Env vars: ${ENV_VARS[*]}"
else
  echo "Env vars: <none>"
fi
echo

# -------------------------------------------------------------------
# 主循环：重复运行 RUNS 次
# -------------------------------------------------------------------
for ((i=1; i<=RUNS; i++)); do
  # 格式化为两位数编号（01, 02, ...）
  run_id="$(printf "%02d" "$i")"

  # 日志文件路径
  log_file="$LOG_DIR/${LOG_PREFIX}_${run_id}.log"

  # 记录开始时间
  started_at="$(date '+%Y-%m-%d %H:%M:%S')"

  echo "========== Run ${run_id}/${RUNS} | ${started_at} =========="

  # 在子 shell 中切换到 WORKDIR 并执行 SLAM 程序
  # 使用子 shell ( ... ) 确保 cd 不影响后续循环的工作目录
  # 2>&1 将 stderr 合并到 stdout，通过 tee 同时输出到终端和日志文件
  if [ "${#ENV_VARS[@]}" -gt 0 ]; then
    (
      cd "$WORKDIR"
      env "${ENV_VARS[@]}" "${CMD[@]}" 2>&1 | tee "$log_file"
    )
  else
    (
      cd "$WORKDIR"
      "${CMD[@]}" 2>&1 | tee "$log_file"
    )
  fi

  # 获取退出码（注意：set -e 未开启，非零退出码不会终止脚本）
  run_exit=$?

  # 记录结束时间
  ended_at="$(date '+%Y-%m-%d %H:%M:%S')"

  # 将本次运行的摘要写入汇总文件
  printf "%s\t%s\t%s\t%s\t%s\n" "$run_id" "$run_exit" "$log_file" "$started_at" "$ended_at" >> "$SUMMARY_FILE"

  # 保存产物文件（如 CameraTrajectory.txt）
  if [ "$SAVE_ARTIFACTS" -eq 1 ]; then
    for f in "${ARTIFACT_FILES[@]}"; do
      src="$WORKDIR/$f"
      if [ -f "$src" ]; then
        base="$(basename "$f")"
        ext="${base##*.}"      # 提取文件扩展名
        stem="${base%.*}"       # 提取文件名（不含扩展名）
        if [ "$ext" = "$base" ]; then
          # 无扩展名的情况：保存为 <文件名>_run<id>
          cp "$src" "$LOG_DIR/${base}_run${run_id}"
        else
          # 有扩展名的情况：保存为 <文件名>_run<id>.<扩展名>
          # 例如 CameraTrajectory.txt -> CameraTrajectory_run01.txt
          cp "$src" "$LOG_DIR/${stem}_run${run_id}.${ext}"
        fi
      fi
    done
  fi

  echo "Run ${run_id} exit code: ${run_exit}"
  echo

  # 如果本次运行失败且开启了遇错停止，终止脚本
  if [ "$run_exit" -ne 0 ] && [ "$STOP_ON_ERROR" -eq 1 ]; then
    echo "Stopping because STOP_ON_ERROR=1."
    exit "$run_exit"
  fi

  # 两次运行之间等待（如配置了非零秒数）
  if [ "$i" -lt "$RUNS" ] && [ "$SLEEP_BETWEEN_RUNS_SEC" -gt 0 ]; then
    sleep "$SLEEP_BETWEEN_RUNS_SEC"
  fi
done

echo "All runs completed."
echo "Summary: $SUMMARY_FILE"
