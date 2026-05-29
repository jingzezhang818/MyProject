#!/usr/bin/env python3
"""Compare this system's trajectory against an ORB-SLAM3 trajectory.

The script aligns both trajectories to the same EuRoC ground truth, computes
per-frame ATE/RPE deltas, optionally joins this system's runtime log, and
reports which diagnostics differ on frames where this system is worse than ORB.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def load_attr_module():
    script_dir = Path(__file__).resolve().parent
    module_path = script_dir / "analyze_trajectory_log_attribution.py"
    spec = importlib.util.spec_from_file_location("trajectory_attr", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


attr = load_attr_module()
np = None


def require_numpy():
    global np
    if np is None:
        np = attr.require_numpy()
    return np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare our trajectory against ORB-SLAM3 and join our logs."
    )
    parser.add_argument("--gt", required=True, help="EuRoC ground-truth CSV path")
    parser.add_argument("--ours", required=True, help="This system's TUM trajectory")
    parser.add_argument("--orb", required=True, help="ORB-SLAM3 TUM trajectory")
    parser.add_argument("--ours-log", help="Optional runtime log from this system")
    parser.add_argument("--out", required=True, help="Output comparison CSV path")
    parser.add_argument(
        "--align",
        choices=("sim3", "se3", "none"),
        default="se3",
        help="Alignment mode for each trajectory against GT. se3 matches stereo evo --align evaluation; sim3 additionally corrects scale.",
    )
    parser.add_argument(
        "--max-time-diff",
        type=float,
        default=0.05,
        help="Maximum trajectory/GT or ours/ORB timestamp difference in seconds.",
    )
    parser.add_argument(
        "--frame-id-offset",
        type=int,
        default=0,
        help="frame_id = ours trajectory index + offset for log joining.",
    )
    parser.add_argument(
        "--segment-size",
        type=int,
        default=100,
        help="Frame count per segment in the text summary.",
    )
    parser.add_argument(
        "--top-ratio",
        type=float,
        default=0.20,
        help="Fraction of diagnostic frames with largest ours-minus-ORB ATE used for metric deltas.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of top worse rows and metric deltas to print/save.",
    )
    return parser.parse_args()


def aligned_rows(gt_path: Path, traj_path: Path, align: str, max_time_diff: float):
    np = require_numpy()
    gt = attr.read_euroc_gt(gt_path)
    est = attr.read_tum_traj(traj_path)
    matches = attr.match_by_timestamp(est, gt, max_time_diff)
    if len(matches) < 3:
        raise SystemExit(f"{traj_path}: only {len(matches)} GT matches found")

    est_pos = np.vstack([est[ei][1] for ei, _, _ in matches])
    gt_pos = np.vstack([gt[gi][1] for _, gi, _ in matches])
    aligned, scale, align_rot, align_trans = attr.align_estimate(est_pos, gt_pos, align)
    ate = np.linalg.norm(aligned - gt_pos, axis=1)

    rows = []
    prev_aligned = None
    prev_gt = None
    prev_est_rot = None
    prev_gt_rot = None
    for row_idx, (est_idx, gt_idx, time_diff) in enumerate(matches):
        est_t, est_p, est_q = est[est_idx]
        gt_t, gt_p, gt_q = gt[gt_idx]
        est_rot = align_rot @ attr.quat_wxyz_to_rot(est_q)
        gt_rot = attr.quat_wxyz_to_rot(gt_q)

        rpe_trans: Optional[float] = None
        rpe_rot_deg: Optional[float] = None
        if prev_aligned is not None and prev_gt is not None:
            rpe_trans = float(np.linalg.norm((aligned[row_idx] - prev_aligned) - (gt_p - prev_gt)))
        if prev_est_rot is not None and prev_gt_rot is not None:
            d_est = prev_est_rot.T @ est_rot
            d_gt = prev_gt_rot.T @ gt_rot
            rpe_rot_deg = attr.rot_angle_deg(d_gt.T @ d_est)

        rows.append(
            {
                "traj_index": est_idx,
                "timestamp": est_t,
                "gt_timestamp": gt_t,
                "time_diff_gt": time_diff,
                "ate": float(ate[row_idx]),
                "rpe_trans": rpe_trans,
                "rpe_rot_deg": rpe_rot_deg,
                "aligned": aligned[row_idx],
                "gt": gt_p,
                "scale": scale,
                "align_tx": align_trans[0],
                "align_ty": align_trans[1],
                "align_tz": align_trans[2],
            }
        )
        prev_aligned = aligned[row_idx]
        prev_gt = gt_p
        prev_est_rot = est_rot
        prev_gt_rot = gt_rot

    summary = {
        "est_poses": len(est),
        "matched_poses": len(matches),
        "rmse": math.sqrt(float(np.mean(ate * ate))),
        "mean": float(np.mean(ate)),
        "median": float(np.median(ate)),
        "p90": float(np.percentile(ate, 90.0)),
        "max": float(np.max(ate)),
        "scale": scale,
    }
    return rows, summary


def match_rows_by_time(ours: Sequence[Dict[str, object]],
                       orb: Sequence[Dict[str, object]],
                       max_time_diff: float) -> List[Tuple[Dict[str, object], Dict[str, object], float]]:
    orb_times = [float(row["timestamp"]) for row in orb]
    matches = []
    import bisect

    for row in ours:
        t = float(row["timestamp"])
        pos = bisect.bisect_left(orb_times, t)
        candidates = []
        if pos < len(orb_times):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        if not candidates:
            continue
        best = min(candidates, key=lambda idx: abs(orb_times[idx] - t))
        diff = abs(orb_times[best] - t)
        if diff <= max_time_diff:
            matches.append((row, orb[best], diff))
    return matches


def maybe_float(value: object) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def segment_lines(rows: Sequence[Dict[str, object]], segment_size: int) -> List[str]:
    if segment_size <= 0:
        segment_size = 100
    max_frame = max(int(row["frame_id"]) for row in rows)
    lines = ["segments frame_range n mean_diff rms_diff max_diff min_diff worse_ratio"]
    for start in range(0, max_frame + 1, segment_size):
        end = start + segment_size
        seg = [row for row in rows if start <= int(row["frame_id"]) < end]
        if not seg:
            continue
        vals = [float(row["ate_diff"]) for row in seg]
        mean = sum(vals) / len(vals)
        rms = math.sqrt(sum(v * v for v in vals) / len(vals))
        worse_ratio = sum(1 for v in vals if v > 0.0) / len(vals)
        lines.append(
            f"{start:03d}-{end - 1:03d} {len(vals)} {mean:.6f} {rms:.6f} "
            f"{max(vals):.6f} {min(vals):.6f} {worse_ratio:.3f}"
        )
    return lines


def diagnostic_metric_deltas(rows: Sequence[Dict[str, object]],
                             top_ratio: float,
                             top_k: int,
                             prefix_filter: Optional[Tuple[str, ...]] = None) -> List[Tuple[str, int, int, float, float, float, float]]:
    base_fields = {
        "frame_id",
        "timestamp",
        "orb_timestamp",
        "orb_time_diff",
        "ate_ours",
        "ate_orb",
        "ate_diff",
        "rpe_trans_ours",
        "rpe_trans_orb",
        "rpe_trans_diff",
        "rpe_rot_deg_ours",
        "rpe_rot_deg_orb",
        "rpe_rot_deg_diff",
    }
    diag_rows = [
        row for row in rows
        if any(key.startswith(("diag_", "search_", "joint_", "depth_", "geom_", "grid_", "posectx_", "priorerr_", "priorshift_", "localba_", "triangcreate_", "kfcreate_", "recentmp_", "xfeat_", "xfeatstruct_", "xfeatdepth_", "matcher_", "matcherfine_", "priority_", "provenance_", "triangquality_", "triangmv_", "trianghist_", "triangrefit_", "kfstruct_"))
               for key in row)
    ]
    if len(diag_rows) < 4:
        return []

    top_n = max(1, int(math.ceil(len(diag_rows) * max(0.0, min(1.0, top_ratio)))))
    top_ids = {
        int(row["frame_id"])
        for row in sorted(diag_rows, key=lambda row: float(row["ate_diff"]), reverse=True)[:top_n]
    }

    keys = sorted(set().union(*(row.keys() for row in diag_rows)) - base_fields)
    if prefix_filter:
        keys = [key for key in keys if key.startswith(prefix_filter)]
    deltas = []
    for key in keys:
        top_vals = [maybe_float(row.get(key)) for row in diag_rows if int(row["frame_id"]) in top_ids]
        rest_vals = [maybe_float(row.get(key)) for row in diag_rows if int(row["frame_id"]) not in top_ids]
        top_nums = [v for v in top_vals if v is not None]
        rest_nums = [v for v in rest_vals if v is not None]
        if len(top_nums) < 3 or len(rest_nums) < 3:
            continue
        top_mean = sum(top_nums) / len(top_nums)
        rest_mean = sum(rest_nums) / len(rest_nums)
        diff = top_mean - rest_mean
        denom = abs(rest_mean) if abs(rest_mean) > 1e-9 else 1.0
        deltas.append((key, len(top_nums), len(rest_nums), top_mean, rest_mean, diff, diff / denom))
    deltas.sort(key=lambda item: abs(item[5]), reverse=True)
    return deltas[:top_k]


def main() -> int:
    args = parse_args()
    np = require_numpy()
    gt_path = Path(args.gt)
    ours_rows, ours_summary = aligned_rows(gt_path, Path(args.ours), args.align, args.max_time_diff)
    orb_rows, orb_summary = aligned_rows(gt_path, Path(args.orb), args.align, args.max_time_diff)
    matches = match_rows_by_time(ours_rows, orb_rows, args.max_time_diff)
    if len(matches) < 3:
        raise SystemExit(f"only {len(matches)} ours/ORB timestamp matches found")

    log_by_frame = attr.parse_log(Path(args.ours_log)) if args.ours_log else {}
    rows: List[Dict[str, object]] = []
    for ours, orb, orb_time_diff in matches:
        frame_id = int(ours["traj_index"]) + args.frame_id_offset
        rpe_trans_ours = ours["rpe_trans"]
        rpe_trans_orb = orb["rpe_trans"]
        rpe_rot_ours = ours["rpe_rot_deg"]
        rpe_rot_orb = orb["rpe_rot_deg"]
        row: Dict[str, object] = {
            "frame_id": frame_id,
            "timestamp": f"{float(ours['timestamp']):.9f}",
            "orb_timestamp": f"{float(orb['timestamp']):.9f}",
            "orb_time_diff": f"{orb_time_diff:.9f}",
            "ate_ours": float(ours["ate"]),
            "ate_orb": float(orb["ate"]),
            "ate_diff": float(ours["ate"]) - float(orb["ate"]),
            "rpe_trans_ours": "" if rpe_trans_ours is None else rpe_trans_ours,
            "rpe_trans_orb": "" if rpe_trans_orb is None else rpe_trans_orb,
            "rpe_trans_diff": "" if rpe_trans_ours is None or rpe_trans_orb is None else rpe_trans_ours - rpe_trans_orb,
            "rpe_rot_deg_ours": "" if rpe_rot_ours is None else rpe_rot_ours,
            "rpe_rot_deg_orb": "" if rpe_rot_orb is None else rpe_rot_orb,
            "rpe_rot_deg_diff": "" if rpe_rot_ours is None or rpe_rot_orb is None else rpe_rot_ours - rpe_rot_orb,
        }
        for key, value in log_by_frame.get(frame_id, {}).items():
            if key != "frame_id":
                row[key] = value
        rows.append(row)

    out_path = Path(args.out)
    write_csv(out_path, rows)
    top_path = out_path.with_name(out_path.stem + "_top_worse.csv")
    write_csv(top_path, sorted(rows, key=lambda row: float(row["ate_diff"]), reverse=True)[: args.top_k])

    diffs = [float(row["ate_diff"]) for row in rows]
    summary_lines = [
        f"ours n={ours_summary['matched_poses']} rmse={ours_summary['rmse']:.6f} "
        f"mean={ours_summary['mean']:.6f} median={ours_summary['median']:.6f} "
        f"p90={ours_summary['p90']:.6f} max={ours_summary['max']:.6f} scale={ours_summary['scale']:.9f}",
        f"orb  n={orb_summary['matched_poses']} rmse={orb_summary['rmse']:.6f} "
        f"mean={orb_summary['mean']:.6f} median={orb_summary['median']:.6f} "
        f"p90={orb_summary['p90']:.6f} max={orb_summary['max']:.6f} scale={orb_summary['scale']:.9f}",
        f"matched={len(rows)} diff_mean={sum(diffs) / len(diffs):.6f} "
        f"diff_rms={math.sqrt(sum(v * v for v in diffs) / len(diffs)):.6f} "
        f"diff_p90={float(np.percentile(diffs, 90.0)):.6f} diff_max={max(diffs):.6f} "
        f"worse_ratio={sum(1 for v in diffs if v > 0.0) / len(diffs):.3f}",
    ]
    summary_lines.extend(segment_lines(rows, args.segment_size))
    summary_lines.append("top worse frames")
    for row in sorted(rows, key=lambda item: float(item["ate_diff"]), reverse=True)[: args.top_k]:
        summary_lines.append(
            f"frame={row['frame_id']} diff={float(row['ate_diff']):.6f} "
            f"ours={float(row['ate_ours']):.6f} orb={float(row['ate_orb']):.6f} "
            f"mnInliers={row.get('diag_mnMatchesInliers', '')} "
            f"local_add={row.get('diag_assoc_added_by_local', '')} "
            f"lmt_total={row.get('joint_local_proj_local_mapping_triangulation_total', '')} "
            f"lmt_outlier={row.get('joint_local_proj_local_mapping_triangulation_outlier_ratio', '')}"
        )

    deltas = diagnostic_metric_deltas(rows, args.top_ratio, args.top_k)
    structure_prefixes = (
        "xfeatstruct_",
        "xfeatdepth_",
        "geom_",
        "grid_",
        "kfstruct_",
        "stablevote_",
        "kfcontribshadow_",
        "kfcontribselect_",
        "kfcontribretain_",
        "closebridge_",
        "refbridge_",
        "refassoc_",
        "refassocfrustum_",
        "covisvote_",
        "kfdecision_",
        "priority_",
        "pool_",
        "provenance_",
        "search_triang_0_10_priority_",
        "search_localmap_lmt_0_10_",
        "search_core_pool_",
        "search_expansion_pool_",
        "search_probation_pool_",
        "reusefunnel_",
        "reusefrustum_",
        "reusecontrib_",
        "reusecontribout_",
        "localkfclose_",
        "reuseoutcome_",
        "reusecohort_",
    )
    structure_deltas = diagnostic_metric_deltas(rows, args.top_ratio, args.top_k, structure_prefixes)
    if deltas:
        summary_lines.extend(
            [
                "",
                "Diagnostic metric deltas on frames where ours is most worse than ORB:",
                "field,top_count,rest_count,top_mean,rest_mean,diff,relative_diff",
            ]
        )
        for key, n_top, n_rest, top_mean, rest_mean, diff, rel in deltas:
            summary_lines.append(
                f"{key},{n_top},{n_rest},{top_mean:.6f},{rest_mean:.6f},{diff:.6f},{rel:.6f}"
            )

    if structure_deltas:
        summary_lines.extend(
            [
                "",
                "Structure-focused deltas on frames where ours is most worse than ORB:",
                "field,top_count,rest_count,top_mean,rest_mean,diff,relative_diff",
            ]
        )
        for key, n_top, n_rest, top_mean, rest_mean, diff, rel in structure_deltas:
            summary_lines.append(
                f"{key},{n_top},{n_rest},{top_mean:.6f},{rest_mean:.6f},{diff:.6f},{rel:.6f}"
            )

    summary_path = out_path.with_name(out_path.stem + "_summary.txt")
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print("\n".join(summary_lines[: min(len(summary_lines), 14)]))
    if deltas:
        print("largest diagnostic deltas:")
        for key, n_top, n_rest, top_mean, rest_mean, diff, rel in deltas[:10]:
            print(
                f"  {key}: top={top_mean:.6f} rest={rest_mean:.6f} "
                f"diff={diff:.6f} rel={rel:.3f} n={n_top}/{n_rest}"
            )
    if structure_deltas:
        print("largest structure deltas:")
        for key, n_top, n_rest, top_mean, rest_mean, diff, rel in structure_deltas[:10]:
            print(
                f"  {key}: top={top_mean:.6f} rest={rest_mean:.6f} "
                f"diff={diff:.6f} rel={rel:.3f} n={n_top}/{n_rest}"
            )
    print(f"wrote {out_path}")
    print(f"wrote {top_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
