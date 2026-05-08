#!/usr/bin/env python3
# Usage:
# 1) Convert estimated trajectory to TUM:
#    python convert_euroc_to_tum.py \
#      --est-traj CameraTrajectory.txt \
#      --out-est camera_traj_tum.txt \
#      --timestamp-unit ns
#
# 2) Convert EuRoC GT CSV to TUM:
#    python convert_euroc_to_tum.py \
#      --gt /path/to/mav0/state_groundtruth_estimate0/data.csv \
#      --out-gt ground_truth_tum.txt
#
# 3) Convert EST and auto-estimate fixed right-rotation bias with GT:
#    python convert_euroc_to_tum.py \
#      --est-traj CameraTrajectory.txt \
#      --out-est camera_traj_tum_corr.txt \
#      --timestamp-unit ns \
#      --estimate-right-rot \
#      --gt Moon_1_gt_tum.txt

"""Convert trajectories to evo-ready TUM files.

Features:
1) Convert CameraTrajectory-style EST to TUM.
2) Convert EuRoC GT CSV to TUM.
3) Optionally estimate and apply a fixed right-rotation bias for EST.
"""

import argparse
import csv
import math
import re
from bisect import bisect_left
from pathlib import Path
from typing import List, Optional, Tuple


def ns_to_s(ts_value: str) -> float:
    return float(ts_value) / 1e9


def unit_to_scale(unit: str) -> float:
    return {
        "ns": 1e-9,
        "us": 1e-6,
        "ms": 1e-3,
        "s": 1.0,
    }[unit]


def quat_norm_wxyz(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    qw, qx, qy, qz = q
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n == 0.0:
        raise ValueError("Zero-norm quaternion.")
    return qw / n, qx / n, qy / n, qz / n


def quat_mul_wxyz(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_inv_wxyz(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    qw, qx, qy, qz = quat_norm_wxyz(q)
    return qw, -qx, -qy, -qz


def quat_angle_deg_wxyz(q: Tuple[float, float, float, float]) -> float:
    qw, _, _, _ = quat_norm_wxyz(q)
    w = max(-1.0, min(1.0, abs(qw)))
    return 2.0 * math.degrees(math.acos(w))


def left_quat_matrix_wxyz(q: Tuple[float, float, float, float]) -> List[List[float]]:
    qw, qx, qy, qz = q
    return [
        [qw, -qx, -qy, -qz],
        [qx, qw, -qz, qy],
        [qy, qz, qw, -qx],
        [qz, -qy, qx, qw],
    ]


def right_quat_matrix_wxyz(q: Tuple[float, float, float, float]) -> List[List[float]]:
    qw, qx, qy, qz = q
    return [
        [qw, -qx, -qy, -qz],
        [qx, qw, qz, -qy],
        [qy, -qz, qw, qx],
        [qz, qy, -qx, qw],
    ]


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


def convert_est_traj_to_tum(
    input_traj: Path, output_tum: Path, timestamp_scale: float = 1e-9
) -> Tuple[int, int]:
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

            ts = float(parts[0]) * timestamp_scale
            if prev_ts is not None and ts <= prev_ts:
                skipped_non_increasing += 1
                continue
            rest = " ".join(parts[1:8])
            f_out.write(f"{ts:.9f} {rest}\n")
            count += 1
            prev_ts = ts
    return count, skipped_non_increasing


def load_tum_file(
    tum_path: Path,
) -> List[Tuple[float, float, float, float, float, float, float, float]]:
    poses = []
    with tum_path.open("r") as f_in:
        for line in f_in:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            poses.append((ts, tx, ty, tz, qw, qx, qy, qz))
    return poses


def estimate_fixed_right_rotation_from_tum(
    gt_tum: Path, est_tum: Path, t_max_diff: float = 0.02, min_rel_angle_deg: float = 0.5
) -> Tuple[Tuple[float, float, float, float], int, int]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "NumPy is required for --estimate-right-rot. Install numpy or disable this option."
        ) from exc

    gt = load_tum_file(gt_tum)
    est = load_tum_file(est_tum)
    if not gt or not est:
        raise ValueError("Cannot estimate right rotation: empty GT or EST TUM file.")

    gt_ts = [row[0] for row in gt]

    # Associate each EST pose to nearest GT timestamp.
    matches: List[Tuple[int, int]] = []
    for i_est, (t_est, *_rest) in enumerate(est):
        j = bisect_left(gt_ts, t_est)
        cand = []
        if j < len(gt):
            cand.append(j)
        if j > 0:
            cand.append(j - 1)
        if not cand:
            continue
        j_best = min(cand, key=lambda idx: abs(gt_ts[idx] - t_est))
        if abs(gt_ts[j_best] - t_est) <= t_max_diff:
            matches.append((i_est, j_best))

    if len(matches) < 2:
        raise ValueError(
            f"Cannot estimate right rotation: insufficient matched timestamps ({len(matches)})."
        )

    blocks = []
    used_rel_pairs = 0
    for k in range(len(matches) - 1):
        i0, j0 = matches[k]
        i1, j1 = matches[k + 1]
        if i1 != i0 + 1:
            continue

        q_est_0 = quat_norm_wxyz((est[i0][4], est[i0][5], est[i0][6], est[i0][7]))
        q_est_1 = quat_norm_wxyz((est[i1][4], est[i1][5], est[i1][6], est[i1][7]))
        q_gt_0 = quat_norm_wxyz((gt[j0][4], gt[j0][5], gt[j0][6], gt[j0][7]))
        q_gt_1 = quat_norm_wxyz((gt[j1][4], gt[j1][5], gt[j1][6], gt[j1][7]))

        d_est = quat_mul_wxyz(quat_inv_wxyz(q_est_0), q_est_1)
        d_gt = quat_mul_wxyz(quat_inv_wxyz(q_gt_0), q_gt_1)

        if (
            quat_angle_deg_wxyz(d_est) < min_rel_angle_deg
            and quat_angle_deg_wxyz(d_gt) < min_rel_angle_deg
        ):
            continue

        l_mat = left_quat_matrix_wxyz(d_est)
        r_mat = right_quat_matrix_wxyz(d_gt)
        block = [[l_mat[r][c] - r_mat[r][c] for c in range(4)] for r in range(4)]
        blocks.extend(block)
        used_rel_pairs += 1

    if used_rel_pairs < 3:
        raise ValueError(
            "Cannot estimate right rotation: not enough informative relative rotation pairs "
            f"({used_rel_pairs}). Consider lowering --estimate-min-rel-angle-deg."
        )

    c_mat = np.array(blocks, dtype=float)
    _u, _s, vh = np.linalg.svd(c_mat, full_matrices=False)
    q_bias = vh[-1, :]
    q_bias = q_bias / np.linalg.norm(q_bias)
    q_bias_wxyz = (float(q_bias[0]), float(q_bias[1]), float(q_bias[2]), float(q_bias[3]))
    if q_bias_wxyz[0] < 0.0:
        q_bias_wxyz = tuple(-v for v in q_bias_wxyz)

    return quat_norm_wxyz(q_bias_wxyz), len(matches), used_rel_pairs


def apply_fixed_right_rotation_to_tum(
    input_tum: Path, output_tum: Path, q_bias_wxyz: Tuple[float, float, float, float]
) -> int:
    q_bias_wxyz = quat_norm_wxyz(q_bias_wxyz)
    count = 0
    with input_tum.open("r") as f_in, output_tum.open("w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            q_est_wxyz = quat_norm_wxyz((qw, qx, qy, qz))
            q_corr_wxyz = quat_mul_wxyz(q_est_wxyz, q_bias_wxyz)
            q_corr_wxyz = quat_norm_wxyz(q_corr_wxyz)
            qw_c, qx_c, qy_c, qz_c = q_corr_wxyz
            f_out.write(f"{ts:.9f} {tx} {ty} {tz} {qx_c} {qy_c} {qz_c} {qw_c}\n")
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert EST/GT trajectories to TUM format. "
            "--gt accepts EuRoC CSV or TUM txt."
        )
    )
    parser.add_argument(
        "--gt",
        default=None,
        help=(
            "Ground-truth input path. "
            "If suffix is .csv, treat as EuRoC GT CSV and convert to --out-gt. "
            "Otherwise treat as TUM txt."
        ),
    )
    parser.add_argument(
        "--cam0-yaml",
        default=None,
        help=(
            "Optional cam0/sensor.yaml used only when --gt is CSV. "
            "If set, GT is transformed from body/IMU to cam0."
        ),
    )
    parser.add_argument(
        "--est-traj",
        default=None,
        help="Estimated trajectory input with: timestamp tx ty tz qx qy qz qw",
    )
    parser.add_argument(
        "--out-gt",
        default="ground_truth_tum.txt",
        help="Output GT TUM path when --gt is CSV.",
    )
    parser.add_argument(
        "--out-est",
        default="camera_traj_tum.txt",
        help="Output EST TUM path when --est-traj is provided.",
    )
    parser.add_argument(
        "--timestamp-unit",
        choices=("ns", "us", "ms", "s"),
        default="ns",
        help="Unit of EST timestamps for --est-traj (default: ns).",
    )
    parser.add_argument(
        "--estimate-right-rot",
        action="store_true",
        help=(
            "Estimate fixed right-multiplication rotation bias between EST and GT, "
            "then apply it to EST quaternions. Requires --est-traj and --gt."
        ),
    )
    parser.add_argument(
        "--estimate-t-max-diff",
        type=float,
        default=0.02,
        help="Max timestamp difference (s) for GT/EST association during right-rot estimation.",
    )
    parser.add_argument(
        "--estimate-min-rel-angle-deg",
        type=float,
        default=0.5,
        help="Ignore near-static relative rotations below this angle (deg) when estimating bias.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    gt_path = Path(args.gt) if args.gt else None
    cam0_yaml = Path(args.cam0_yaml) if args.cam0_yaml else None
    est_traj = Path(args.est_traj) if args.est_traj else None
    out_gt = Path(args.out_gt)
    out_est = Path(args.out_est)

    gt_count = 0
    gt_skipped = 0
    est_count = 0
    est_skipped = 0
    gt_converted = False
    gt_input_tum = False
    est_converted = False
    gt_for_right_rot: Optional[Path] = None

    if gt_path is not None:
        if gt_path.suffix.lower() == ".csv":
            gt_count, gt_skipped = convert_euroc_gt_csv_to_tum(gt_path, out_gt, cam0_yaml=cam0_yaml)
            gt_for_right_rot = out_gt
            gt_converted = True
        else:
            gt_for_right_rot = gt_path
            gt_input_tum = True

    if est_traj is not None:
        est_count, est_skipped = convert_est_traj_to_tum(
            est_traj, out_est, timestamp_scale=unit_to_scale(args.timestamp_unit)
        )
        est_converted = True

    if args.estimate_right_rot:
        if not est_converted:
            raise ValueError("--estimate-right-rot requires --est-traj.")
        if gt_for_right_rot is None:
            raise ValueError("--estimate-right-rot requires --gt (CSV or TUM txt).")
        q_bias_wxyz, matched_count, rel_pair_count = estimate_fixed_right_rotation_from_tum(
            gt_for_right_rot,
            out_est,
            t_max_diff=args.estimate_t_max_diff,
            min_rel_angle_deg=args.estimate_min_rel_angle_deg,
        )
        tmp_out = out_est.with_suffix(out_est.suffix + ".right_rot_tmp")
        corrected_count = apply_fixed_right_rotation_to_tum(out_est, tmp_out, q_bias_wxyz)
        tmp_out.replace(out_est)
        qw_b, qx_b, qy_b, qz_b = q_bias_wxyz
        print("Estimated fixed EST right-rotation bias and applied to output EST:")
        print(f"  matched timestamps: {matched_count}")
        print(f"  informative relative pairs: {rel_pair_count}")
        print(
            "  q_bias (xyzw): "
            f"{qx_b:.9f} {qy_b:.9f} {qz_b:.9f} {qw_b:.9f} "
            f"(angle: {quat_angle_deg_wxyz(q_bias_wxyz):.6f} deg)"
        )
        print(f"  corrected EST poses written: {corrected_count}")

    if gt_converted:
        print(f"GT converted: {gt_count} poses -> {out_gt}")
    elif gt_input_tum:
        print(f"GT input detected as TUM: {gt_for_right_rot}")
    if est_converted:
        print(f"EST converted: {est_count} poses -> {out_est}")
    if gt_converted and gt_skipped > 0:
        print(f"GT skipped non-increasing timestamps: {gt_skipped}")
    if est_converted and est_skipped > 0:
        print(f"EST skipped non-increasing timestamps: {est_skipped}")
    if gt_converted:
        if cam0_yaml:
            print(f"Applied cam0 extrinsic from: {cam0_yaml}")
        else:
            print("No cam0 extrinsic applied. GT remains in body/IMU frame.")
    if not gt_converted and not gt_input_tum and not est_converted:
        print("No conversion performed. Provide --est-traj and/or --gt.")
    print("Run evo example:")
    if est_converted and gt_for_right_rot is not None:
        print(f"evo_ape tum {gt_for_right_rot} {out_est} -r trans_part -a -p")
        print(f"evo_traj tum {out_est} --ref={gt_for_right_rot} -a -p")
    elif gt_converted:
        print(f"evo_traj tum {out_gt} -a -p")
    elif gt_input_tum:
        print(f"evo_traj tum {gt_for_right_rot} -a -p")
    elif est_converted:
        print(f"evo_ape tum <gt_tum.txt> {out_est} -r trans_part -a -p")
        print(f"evo_traj tum {out_est} --ref=<gt_tum.txt> -a -p")
    else:
        print("evo_ape tum <gt_tum.txt> <est_tum.txt> -r trans_part -a -p")
        print("evo_traj tum <est_tum.txt> --ref=<gt_tum.txt> -a -p")
    print("If timestamps still mismatch, tune with: --t_max_diff 0.02 or --t_offset <sec>")


if __name__ == "__main__":
    main()
