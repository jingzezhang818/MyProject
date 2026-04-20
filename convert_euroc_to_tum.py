#!/usr/bin/env python3
"""Convert EuRoC GT CSV + estimated trajectory to evo-ready TUM files."""

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Optional, Tuple


def ns_to_s(ts_value: str) -> float:
    return float(ts_value) / 1e9


def quat_wxyz_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> list:
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n == 0.0:
        raise ValueError("Zero-norm quaternion in GT.")
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return [
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ],
        [
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ],
        [
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    ]


def rotmat_to_quat_wxyz(r: list) -> Tuple[float, float, float, float]:
    tr = r[0][0] + r[1][1] + r[2][2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2][1] - r[1][2]) / s
        qy = (r[0][2] - r[2][0]) / s
        qz = (r[1][0] - r[0][1]) / s
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
        qw = (r[2][1] - r[1][2]) / s
        qx = 0.25 * s
        qy = (r[0][1] + r[1][0]) / s
        qz = (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
        qw = (r[0][2] - r[2][0]) / s
        qx = (r[0][1] + r[1][0]) / s
        qy = 0.25 * s
        qz = (r[1][2] + r[2][1]) / s
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
        qw = (r[1][0] - r[0][1]) / s
        qx = (r[0][2] + r[2][0]) / s
        qy = (r[1][2] + r[2][1]) / s
        qz = 0.25 * s

    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n == 0.0:
        raise ValueError("Invalid rotation matrix to quaternion conversion.")
    return qw / n, qx / n, qy / n, qz / n


def matmul3(a: list, b: list) -> list:
    return [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
        ],
        [
            a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
            a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
            a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
        ],
    ]


def matvec3(a: list, v: list) -> list:
    return [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]


def load_t_bs_from_sensor_yaml(sensor_yaml: Path) -> Tuple[list, list]:
    content = sensor_yaml.read_text()
    match = re.search(r"T_BS:\s*.*?data:\s*\[(.*?)\]", content, flags=re.S)
    if not match:
        raise ValueError(f"Cannot find T_BS/data in: {sensor_yaml}")

    raw = match.group(1).replace("\n", " ")
    values = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if len(values) != 16:
        raise ValueError(f"T_BS must have 16 values, got {len(values)} in: {sensor_yaml}")

    r_bs = [
        [values[0], values[1], values[2]],
        [values[4], values[5], values[6]],
        [values[8], values[9], values[10]],
    ]
    t_bs = [values[3], values[7], values[11]]
    return r_bs, t_bs


def convert_euroc_gt_csv_to_tum(
    input_csv: Path, output_tum: Path, cam0_yaml: Optional[Path] = None
) -> Tuple[int, int]:
    r_bs, t_bs = (None, None)
    if cam0_yaml is not None:
        # EuRoC cam0 sensor.yaml has T_BS (sensor wrt body). With GT T_WB:
        # T_WC = T_WB * T_BC where T_BC == T_BS for cam0.
        r_bs, t_bs = load_t_bs_from_sensor_yaml(cam0_yaml)

    count = 0
    skipped_non_increasing = 0
    prev_ts = None
    with input_csv.open("r", newline="") as f_in, output_tum.open("w") as f_out:
        reader = csv.reader(f_in)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Empty CSV file: {input_csv}")

        for row in reader:
            if not row or len(row) < 8:
                continue

            ts = ns_to_s(row[0])
            if prev_ts is not None and ts <= prev_ts:
                skipped_non_increasing += 1
                continue
            tx, ty, tz = float(row[1]), float(row[2]), float(row[3])
            qw, qx, qy, qz = float(row[4]), float(row[5]), float(row[6]), float(row[7])

            if r_bs is not None and t_bs is not None:
                r_wb = quat_wxyz_to_rotmat(qw, qx, qy, qz)
                r_wc = matmul3(r_wb, r_bs)
                t_wb = [tx, ty, tz]
                t_wc_offset = matvec3(r_wb, t_bs)
                tx = t_wb[0] + t_wc_offset[0]
                ty = t_wb[1] + t_wc_offset[1]
                tz = t_wb[2] + t_wc_offset[2]
                qw, qx, qy, qz = rotmat_to_quat_wxyz(r_wc)

            f_out.write(f"{ts:.9f} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")
            count += 1
            prev_ts = ts
    return count, skipped_non_increasing


def convert_est_traj_ns_to_tum(input_traj: Path, output_tum: Path) -> Tuple[int, int]:
    count = 0
    skipped_non_increasing = 0
    prev_ts = None
    with input_traj.open("r") as f_in, output_tum.open("w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue

            ts = ns_to_s(parts[0])
            if prev_ts is not None and ts <= prev_ts:
                skipped_non_increasing += 1
                continue
            rest = " ".join(parts[1:8])
            f_out.write(f"{ts:.9f} {rest}\n")
            count += 1
            prev_ts = ts
    return count, skipped_non_increasing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert EuRoC GT CSV and ns-timestamp trajectory to TUM format for evo."
    )
    parser.add_argument(
        "--gt-csv",
        required=True,
        help="EuRoC ground truth CSV, e.g. mav0/state_groundtruth_estimate0/data.csv",
    )
    parser.add_argument(
        "--cam0-yaml",
        default=None,
        help="Optional EuRoC cam0/sensor.yaml. If set, GT is transformed from body/IMU to cam0.",
    )
    parser.add_argument(
        "--est-traj",
        required=True,
        help="Estimated trajectory txt with: timestamp_ns tx ty tz qx qy qz qw",
    )
    parser.add_argument(
        "--out-gt",
        default="ground_truth_tum.txt",
        help="Output GT trajectory in TUM format (seconds).",
    )
    parser.add_argument(
        "--out-est",
        default="camera_traj_tum.txt",
        help="Output estimated trajectory in TUM format (seconds).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    gt_csv = Path(args.gt_csv)
    cam0_yaml = Path(args.cam0_yaml) if args.cam0_yaml else None
    est_traj = Path(args.est_traj)
    out_gt = Path(args.out_gt)
    out_est = Path(args.out_est)

    gt_count, gt_skipped = convert_euroc_gt_csv_to_tum(gt_csv, out_gt, cam0_yaml=cam0_yaml)
    est_count, est_skipped = convert_est_traj_ns_to_tum(est_traj, out_est)

    print(f"GT converted: {gt_count} poses -> {out_gt}")
    print(f"EST converted: {est_count} poses -> {out_est}")
    if gt_skipped > 0:
        print(f"GT skipped non-increasing timestamps: {gt_skipped}")
    if est_skipped > 0:
        print(f"EST skipped non-increasing timestamps: {est_skipped}")
    if cam0_yaml:
        print(f"Applied cam0 extrinsic from: {cam0_yaml}")
    else:
        print("No cam0 extrinsic applied. GT remains in body/IMU frame.")
    print("Run evo example:")
    print(f"evo_ape tum {out_gt} {out_est} -r full -a -p")
    print("If timestamps still mismatch, tune with: --t_max_diff 0.02 or --t_offset <sec>")


if __name__ == "__main__":
    main()
