#!/usr/bin/env python3
"""Join trajectory error with SLAM debug logs.

This script is intentionally offline-only: it does not change SLAM behavior.
It aligns an estimated TUM trajectory to EuRoC ground truth, computes per-pose
ATE/RPE, joins debug log metrics by frame id, and reports which internal
metrics differ most on high-error frames.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

np = None


Pose = Tuple[float, Any, Any]


def require_numpy():
    global np
    if np is not None:
        return np
    try:
        import numpy as numpy_module
    except ImportError as exc:  # pragma: no cover - user environment dependent
        raise SystemExit(
            "numpy is required to align trajectories and compute ATE: "
            "python3 -m pip install numpy"
        ) from exc
    np = numpy_module
    return np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attribute trajectory error to per-frame SLAM diagnostics."
    )
    parser.add_argument("--gt", required=True, help="EuRoC ground-truth CSV path")
    parser.add_argument("--traj", required=True, help="Estimated TUM trajectory path")
    parser.add_argument("--log", required=True, help="SLAM runtime log path")
    parser.add_argument(
        "--out",
        required=True,
        help="Output attribution CSV path. Extra files use the same stem.",
    )
    parser.add_argument(
        "--align",
        choices=("sim3", "se3", "none"),
        default="se3",
        help="Position alignment mode. se3 is the default to match stereo evo --align evaluation; use sim3 only for scale-corrected diagnostics.",
    )
    parser.add_argument(
        "--max-time-diff",
        type=float,
        default=0.05,
        help="Maximum trajectory/GT timestamp difference in seconds.",
    )
    parser.add_argument(
        "--frame-id-offset",
        type=int,
        default=0,
        help="frame_id = trajectory_index + offset for log joining.",
    )
    parser.add_argument(
        "--top-ratio",
        type=float,
        default=0.10,
        help="Top ATE fraction used for attribution summary.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of top-error rows and metric deltas to print/save.",
    )
    return parser.parse_args()


def normalize_quat(q: np.ndarray) -> np.ndarray:
    np = require_numpy()
    norm = float(np.linalg.norm(q))
    if norm <= 0.0 or not math.isfinite(norm):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / norm


def quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    np = require_numpy()
    q = normalize_quat(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rot_angle_deg(rot: np.ndarray) -> float:
    cos_angle = (float(np.trace(rot)) - 1.0) * 0.5
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def timestamp_to_seconds(value: float) -> float:
    # EuRoC CSV timestamps are nanoseconds. TUM trajectory timestamps are seconds.
    if abs(value) > 1.0e12:
        return value * 1.0e-9
    return value


def read_euroc_gt(path: Path) -> List[Pose]:
    np = require_numpy()
    poses: List[Pose] = []
    with path.open("r", newline="") as fp:
        reader = csv.reader(fp)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 8:
                continue
            try:
                t = timestamp_to_seconds(float(row[0]))
                pos = np.array([float(row[1]), float(row[2]), float(row[3])], dtype=float)
                quat = normalize_quat(
                    np.array([float(row[4]), float(row[5]), float(row[6]), float(row[7])], dtype=float)
                )
            except ValueError:
                continue
            poses.append((t, pos, quat))
    poses.sort(key=lambda p: p[0])

    # Keep the first sample for duplicated timestamps; EuRoC files can contain duplicates.
    unique: List[Pose] = []
    seen = set()
    for pose in poses:
        key = round(pose[0], 9)
        if key in seen:
            continue
        seen.add(key)
        unique.append(pose)
    return unique


def read_tum_traj(path: Path) -> List[Pose]:
    np = require_numpy()
    poses: List[Pose] = []
    with path.open("r") as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                t = timestamp_to_seconds(float(parts[0]))
                pos = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
                # TUM stores qx qy qz qw.
                quat = normalize_quat(
                    np.array([float(parts[7]), float(parts[4]), float(parts[5]), float(parts[6])], dtype=float)
                )
            except ValueError:
                continue
            poses.append((t, pos, quat))
    poses.sort(key=lambda p: p[0])
    return poses


def match_by_timestamp(est: Sequence[Pose], gt: Sequence[Pose], max_diff: float) -> List[Tuple[int, int, float]]:
    gt_times = [p[0] for p in gt]
    matches: List[Tuple[int, int, float]] = []
    for est_idx, pose in enumerate(est):
        t = pose[0]
        pos = bisect.bisect_left(gt_times, t)
        candidates = []
        if pos < len(gt_times):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        if not candidates:
            continue
        gt_idx = min(candidates, key=lambda idx: abs(gt_times[idx] - t))
        diff = abs(gt_times[gt_idx] - t)
        if diff <= max_diff:
            matches.append((est_idx, gt_idx, diff))
    return matches


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool) -> Tuple[float, np.ndarray, np.ndarray]:
    np = require_numpy()
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src/dst must be Nx3")
    n = src.shape[0]
    if n < 3:
        raise ValueError("at least 3 matched poses are required")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / float(n)
    u, singular, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        s[2, 2] = -1.0
    rot = u @ s @ vt
    scale = 1.0
    if with_scale:
        var_src = float(np.sum(src_c * src_c) / n)
        if var_src > 0.0:
            scale = float(np.sum(singular * np.diag(s)) / var_src)
    trans = mu_dst - scale * (rot @ mu_src)
    return scale, rot, trans


def align_estimate(
    est_positions: np.ndarray,
    gt_positions: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    np = require_numpy()
    if mode == "none":
        return est_positions.copy(), 1.0, np.eye(3), np.zeros(3)
    scale, rot, trans = umeyama(est_positions, gt_positions, with_scale=(mode == "sim3"))
    aligned = (scale * (rot @ est_positions.T)).T + trans
    return aligned, scale, rot, trans


KV_RE = re.compile(r"(\w+)=([^\s]+)")


def parse_kv(line: str) -> Dict[str, str]:
    return dict(KV_RE.findall(line))


def safe_prefix(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value.strip("_")).strip("_")


def maybe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def add_prefixed(frame: Dict[str, str], prefix: str, data: Dict[str, str], skip: Iterable[str]) -> None:
    skip_set = set(skip)
    for key, value in data.items():
        if key in skip_set:
            continue
        frame[f"{prefix}_{key}"] = value


def parse_log(path: Path) -> Dict[int, Dict[str, str]]:
    frames: Dict[int, Dict[str, str]] = {}

    def frame_dict(frame_id: int) -> Dict[str, str]:
        return frames.setdefault(frame_id, {"frame_id": str(frame_id)})

    with path.open("r", errors="replace") as fp:
        for line in fp:
            if "frame_id=" not in line and "frame=" not in line and "kf_id=" not in line:
                continue
            data = parse_kv(line)
            frame_token = data.get("frame_id", data.get("frame", data.get("kf_id")))
            if frame_token is None:
                continue
            try:
                frame_id = int(float(frame_token))
            except ValueError:
                continue
            out = frame_dict(frame_id)

            if "[TrackLocalMap][Diag]" in line:
                add_prefixed(out, "diag", data, {"frame_id"})
            elif "[TrackLocalMap][InlierMix]" in line:
                add_prefixed(out, "mix", data, {"frame_id"})
            elif "[SearchLocalPoints][CloseReuseFunnel]" in line:
                source = safe_prefix(data.get("source", "unknown"))
                depth_bin = safe_prefix(data.get("depth_bin", "unknown"))
                add_prefixed(out, f"reusefunnel_{source}_{depth_bin}", data, {"frame_id", "source", "depth_bin"})
            elif "[SearchLocalPoints][CloseReuseFrustumReason]" in line:
                source = safe_prefix(data.get("source", "unknown"))
                depth_bin = safe_prefix(data.get("depth_bin", "unknown"))
                reason = safe_prefix(data.get("reason", "unknown"))
                add_prefixed(out, f"reusefrustum_{source}_{depth_bin}_{reason}", data,
                             {"frame_id", "source", "depth_bin", "reason"})
            elif "[SearchLocalPoints][ReferenceAssociationFunnel]" in line:
                source = safe_prefix(data.get("source", "unknown"))
                depth_bin = safe_prefix(data.get("depth_bin", "unknown"))
                add_prefixed(out, f"refassoc_{source}_{depth_bin}", data,
                             {"frame_id", "source", "depth_bin"})
            elif "[SearchLocalPoints][ReferenceAssociationFrustumReason]" in line:
                source = safe_prefix(data.get("source", "unknown"))
                depth_bin = safe_prefix(data.get("depth_bin", "unknown"))
                reason = safe_prefix(data.get("reason", "unknown"))
                add_prefixed(out, f"refassocfrustum_{source}_{depth_bin}_{reason}", data,
                             {"frame_id", "source", "depth_bin", "reason"})
            elif "[SearchLocalPoints][CloseReuseContribution]" in line:
                source = safe_prefix(data.get("source", "unknown"))
                contributor = safe_prefix(data.get("contributor", "unknown"))
                add_prefixed(out, f"reusecontrib_{source}_{contributor}", data,
                             {"frame_id", "source", "contributor"})
            elif "[SearchLocalPoints][LocalKFCloseContribution]" in line:
                add_prefixed(out, "localkfclose", data, {"frame_id"})
            elif "[SearchLocalPoints][MappingFeatureReuseFunnel]" in line:
                source = safe_prefix(data.get("source", "unknown"))
                dim = safe_prefix(data.get("dimension", "unknown"))
                bin_name = safe_prefix(data.get("bin", "unknown"))
                add_prefixed(out, f"mapfeatreuse_{source}_{dim}_{bin_name}", data,
                             {"frame_id", "source", "dimension", "bin"})
            elif "[UpdateLocalPoints][Payload]" in line:
                add_prefixed(out, "payload", data, {"frame"})
            elif "[SearchLocalPoints]" in line and "[MPQuality]" not in line:
                add_prefixed(out, "search", data, {"frame"})
            elif "[TrackLocalMap][SourceMPJointError]" in line:
                match_source = safe_prefix(data.get("match_source", "unknown"))
                mp_source = safe_prefix(data.get("mp_source", "unknown"))
                add_prefixed(out, f"joint_{match_source}_{mp_source}", data, {"frame_id", "match_source", "mp_source"})
            elif "[TrackLocalMap][SourceDepthError]" in line:
                source = safe_prefix(data.get("source", "unknown"))
                depth_bin = safe_prefix(data.get("depth_bin", "unknown"))
                add_prefixed(out, f"depth_{source}_{depth_bin}", data, {"frame_id", "source", "depth_bin"})
            elif "[TrackLocalMap][CloseReuseOutcome]" in line:
                source = safe_prefix(data.get("source", "unknown"))
                depth_bin = safe_prefix(data.get("depth_bin", "unknown"))
                add_prefixed(out, f"reuseoutcome_{source}_{depth_bin}", data, {"frame_id", "source", "depth_bin"})
            elif "[TrackLocalMap][CloseReuseContributionOutcome]" in line:
                source = safe_prefix(data.get("source", "unknown"))
                contributor = safe_prefix(data.get("contributor", "unknown"))
                add_prefixed(out, f"reusecontribout_{source}_{contributor}", data,
                             {"frame_id", "source", "contributor"})
            elif "[TrackLocalMap][CloseReuseCohortAlive]" in line:
                source = safe_prefix(data.get("source", "unknown"))
                depth_bin = safe_prefix(data.get("depth_bin", "unknown"))
                horizon = safe_prefix(data.get("horizon_frames", "unknown"))
                add_prefixed(out, f"reusecohort_{source}_{depth_bin}_h{horizon}", data, {"frame_id", "source", "depth_bin", "horizon_frames"})
            elif "[TrackLocalMap][MappingRepeatabilityCohort]" in line:
                source = safe_prefix(data.get("source", "unknown"))
                depth_bin = safe_prefix(data.get("depth_bin", "unknown"))
                dim = safe_prefix(data.get("dimension", "unknown"))
                bin_name = safe_prefix(data.get("bin", "unknown"))
                horizon = safe_prefix(data.get("horizon_frames", "unknown"))
                add_prefixed(out, f"maprepeat_{source}_{depth_bin}_{dim}_{bin_name}_h{horizon}", data,
                             {"frame_id", "source", "depth_bin", "dimension", "bin", "horizon_frames"})
            elif "[TrackLocalMap][InlierGeometry]" in line:
                group = safe_prefix(data.get("group", "unknown"))
                add_prefixed(out, f"geom_{group}", data, {"frame_id", "group"})
            elif "[TrackLocalMap][ConstraintGrid]" in line:
                group = safe_prefix(data.get("group", "unknown"))
                add_prefixed(out, f"grid_{group}", data, {"frame_id", "group"})
            elif "[XFeatDiag][StructureSummary]" in line:
                add_prefixed(out, "xfeatstruct", data, {"frame_id"})
            elif "[XFeatDiag][depth]" in line:
                add_prefixed(out, "xfeatdepth", data, {"frame_id"})
            elif line.startswith("[XFeatDiag] "):
                add_prefixed(out, "xfeat", data, {"frame"})
            elif "[TrackLocalMap][KFStructure]" in line:
                add_prefixed(out, "kfstruct", data, {"frame_id"})
            elif "[TrackLocalMap][StableVoteShadow]" in line:
                add_prefixed(out, "stablevote", data, {"frame_id"})
            elif "[TrackLocalMap][LocalKFContributionSelection]" in line:
                add_prefixed(out, "kfcontribselect", data, {"frame_id"})
            elif "[TrackLocalMap][LocalKFContributionShadowRetention]" in line:
                group = safe_prefix(data.get("group", "unknown"))
                add_prefixed(out, f"kfcontribretain_{group}", data, {"frame_id", "group"})
            elif "[TrackLocalMap][LocalKFContributionShadow]" in line:
                add_prefixed(out, "kfcontribshadow", data, {"frame_id"})
            elif "[Tracking][CloseFeatureReuseBridge]" in line:
                source = safe_prefix(data.get("source", "unknown"))
                depth_bin = safe_prefix(data.get("depth_bin", "unknown"))
                add_prefixed(out, f"closebridge_{source}_{depth_bin}", data,
                             {"frame_id", "source", "depth_bin"})
            elif "[Tracking][ReferenceReuseBridge]" in line:
                source = safe_prefix(data.get("source", "unknown"))
                depth_bin = safe_prefix(data.get("depth_bin", "unknown"))
                add_prefixed(out, f"refbridge_{source}_{depth_bin}", data,
                             {"frame_id", "source", "depth_bin"})
            elif "[Tracking][KFDecision]" in line:
                add_prefixed(out, "kfdecision", data, {"frame_id"})
            elif "[TrackLocalMap][LocalProjMatcherQualityError]" in line:
                dim = safe_prefix(data.get("dimension", "unknown"))
                bin_name = safe_prefix(data.get("bin", "unknown"))
                add_prefixed(out, f"matcher_{dim}_{bin_name}", data, {"frame_id", "dimension", "bin"})
            elif "[TrackLocalMap][LocalProjMatcherQualityFineError]" in line:
                dim = safe_prefix(data.get("dimension", "unknown"))
                bin_name = safe_prefix(data.get("bin", "unknown"))
                add_prefixed(out, f"matcherfine_{dim}_{bin_name}", data, {"frame_id", "dimension", "bin"})
            elif "[TrackLocalMap][LocalProjPoseContextError]" in line:
                dim = safe_prefix(data.get("dimension", "unknown"))
                bin_name = safe_prefix(data.get("bin", "unknown"))
                add_prefixed(out, f"posectx_{dim}_{bin_name}", data, {"frame_id", "dimension", "bin"})
            elif "[TrackLocalMap][ProjectionPriorShiftError]" in line:
                dim = safe_prefix(data.get("dimension", "unknown"))
                bin_name = safe_prefix(data.get("bin", "unknown"))
                add_prefixed(out, f"priorerr_{dim}_{bin_name}", data, {"frame_id", "dimension", "bin"})
            elif "[TrackLocalMap][PosePriorShift]" in line:
                add_prefixed(out, "priorshift", data, {"frame_id"})
            elif "[TrackLocalMap][TriangQualityError]" in line:
                dim = safe_prefix(data.get("dimension", "unknown"))
                bin_name = safe_prefix(data.get("bin", "unknown"))
                add_prefixed(out, f"triangquality_{dim}_{bin_name}", data, {"frame_id", "dimension", "bin"})
            elif "[TrackLocalMap][TriangMultiViewError]" in line:
                dim = safe_prefix(data.get("dimension", "unknown"))
                bin_name = safe_prefix(data.get("bin", "unknown"))
                add_prefixed(out, f"triangmv_{dim}_{bin_name}", data, {"frame_id", "dimension", "bin"})
            elif "[TrackLocalMap][TriangLocalProjHistoryError]" in line:
                dim = safe_prefix(data.get("dimension", "unknown"))
                bin_name = safe_prefix(data.get("bin", "unknown"))
                add_prefixed(out, f"trianghist_{dim}_{bin_name}", data, {"frame_id", "dimension", "bin"})
            elif "[TrackLocalMap][TriangRefitShadowError]" in line:
                dim = safe_prefix(data.get("dimension", "unknown"))
                bin_name = safe_prefix(data.get("bin", "unknown"))
                add_prefixed(out, f"triangrefit_{dim}_{bin_name}", data, {"frame_id", "dimension", "bin"})
            elif "[TrackLocalMap][LocalMapProvenanceError]" in line:
                dim = safe_prefix(data.get("dimension", "unknown"))
                bin_name = safe_prefix(data.get("bin", "unknown"))
                add_prefixed(out, f"provenance_{dim}_{bin_name}", data, {"frame_id", "dimension", "bin"})
            elif "[TrackLocalMap][LocalProjPriorityError]" in line:
                dim = safe_prefix(data.get("dimension", "unknown"))
                bin_name = safe_prefix(data.get("bin", "unknown"))
                add_prefixed(out, f"priority_{dim}_{bin_name}", data, {"frame_id", "dimension", "bin"})
            elif "[TrackLocalMap][LocalProjPoolError]" in line:
                pool = safe_prefix(data.get("match_pool", "unknown"))
                add_prefixed(out, f"pool_{pool}", data, {"frame_id", "match_pool"})
            elif "[TrackLocalMap][MappingFeatureOutcome]" in line:
                source = safe_prefix(data.get("source", "unknown"))
                dim = safe_prefix(data.get("dimension", "unknown"))
                bin_name = safe_prefix(data.get("bin", "unknown"))
                add_prefixed(out, f"mapfeatout_{source}_{dim}_{bin_name}", data, {"frame_id", "source", "dimension", "bin"})
            elif "[LocalMapping][TriangCreate]" in line:
                add_prefixed(out, "triangcreate", data, {"kf_id", "frame_id"})
            elif "[LocalMapping][MappingFeatureCreate]" in line:
                source = safe_prefix(data.get("source", "unknown"))
                dim = safe_prefix(data.get("dimension", "unknown"))
                bin_name = safe_prefix(data.get("bin", "unknown"))
                add_prefixed(out, f"mapfeatcreate_{source}_{dim}_{bin_name}", data, {"kf_id", "frame_id", "source", "dimension", "bin"})
            elif "[LocalMapping][RecentMP]" in line:
                add_prefixed(out, "recentmp", data, {"kf_id", "frame_id"})
            elif "[LocalMapping][RecentMPSource]" in line:
                source = safe_prefix(data.get("source", "unknown"))
                add_prefixed(out, f"recentmpsrc_{source}", data, {"kf_id", "frame_id", "source"})
            elif "[LocalMapping][KFCreateStructure]" in line:
                add_prefixed(out, "kfcreate", data, {"kf_id", "frame_id"})
            elif "[KeyFrame][CovisVoteSource]" in line:
                add_prefixed(out, "covisvote", data, {"kf_id", "frame_id"})
            elif "[LocalBA][SourceDepthError]" in line:
                source = safe_prefix(data.get("source", "unknown"))
                depth_bin = safe_prefix(data.get("depth_bin", "unknown"))
                add_prefixed(out, f"localba_{source}_{depth_bin}", data, {"kf_id", "source", "depth_bin"})
    return frames


def write_csv(path: Path, rows: Sequence[Dict[str, object]], preferred: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    fields = list(preferred)
    fields.extend(sorted(k for k in all_keys if k not in fields))
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_metric_deltas(rows: Sequence[Dict[str, object]], top_field: str = "top_error") -> List[Tuple[str, int, int, float, float, float, float]]:
    base_fields = {
        "traj_index",
        "frame_id",
        "timestamp",
        "gt_timestamp",
        "time_diff",
        "est_x",
        "est_y",
        "est_z",
        "gt_x",
        "gt_y",
        "gt_z",
        "aligned_x",
        "aligned_y",
        "aligned_z",
        "ate",
        "rpe_trans",
        "rpe_rot_deg",
        "top_error",
    }
    keys = sorted(set().union(*(row.keys() for row in rows)) - base_fields)
    deltas: List[Tuple[str, int, int, float, float, float, float]] = []
    for key in keys:
        top_vals = [maybe_float(row.get(key)) for row in rows if row.get(top_field) == 1]
        rest_vals = [maybe_float(row.get(key)) for row in rows if row.get(top_field) != 1]
        top_nums = [v for v in top_vals if v is not None]
        rest_nums = [v for v in rest_vals if v is not None]
        if len(top_nums) < 3 or len(rest_nums) < 3:
            continue
        top_mean = sum(top_nums) / len(top_nums)
        rest_mean = sum(rest_nums) / len(rest_nums)
        diff = top_mean - rest_mean
        denom = abs(rest_mean) if abs(rest_mean) > 1e-9 else 1.0
        rel = diff / denom
        deltas.append((key, len(top_nums), len(rest_nums), top_mean, rest_mean, diff, rel))
    deltas.sort(key=lambda item: abs(item[5]), reverse=True)
    return deltas


def build_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    np = require_numpy()
    gt = read_euroc_gt(Path(args.gt))
    est = read_tum_traj(Path(args.traj))
    matches = match_by_timestamp(est, gt, args.max_time_diff)
    if len(matches) < 3:
        raise SystemExit(
            f"Only {len(matches)} pose matches found. Increase --max-time-diff or check timestamps."
        )

    est_pos = np.vstack([est[ei][1] for ei, _, _ in matches])
    gt_pos = np.vstack([gt[gi][1] for _, gi, _ in matches])
    aligned_pos, scale, align_rot, align_trans = align_estimate(est_pos, gt_pos, args.align)
    ate = np.linalg.norm(aligned_pos - gt_pos, axis=1)

    log_by_frame = parse_log(Path(args.log))
    top_n = max(1, int(math.ceil(len(matches) * max(0.0, min(1.0, args.top_ratio)))))
    top_indices = set(np.argsort(-ate)[:top_n].tolist())

    rows: List[Dict[str, object]] = []
    prev_aligned: Optional[np.ndarray] = None
    prev_gt: Optional[np.ndarray] = None
    prev_est_rot: Optional[np.ndarray] = None
    prev_gt_rot: Optional[np.ndarray] = None

    for row_idx, (est_idx, gt_idx, time_diff) in enumerate(matches):
        est_t, est_p, est_q = est[est_idx]
        gt_t, gt_p, gt_q = gt[gt_idx]
        frame_id = est_idx + args.frame_id_offset
        aligned = aligned_pos[row_idx]

        rpe_trans = ""
        rpe_rot_deg = ""
        est_rot = align_rot @ quat_wxyz_to_rot(est_q)
        gt_rot = quat_wxyz_to_rot(gt_q)
        if prev_aligned is not None and prev_gt is not None:
            rpe_trans = float(np.linalg.norm((aligned - prev_aligned) - (gt_p - prev_gt)))
        if prev_est_rot is not None and prev_gt_rot is not None:
            d_est = prev_est_rot.T @ est_rot
            d_gt = prev_gt_rot.T @ gt_rot
            rpe_rot_deg = rot_angle_deg(d_gt.T @ d_est)

        out: Dict[str, object] = {
            "traj_index": est_idx,
            "frame_id": frame_id,
            "timestamp": f"{est_t:.9f}",
            "gt_timestamp": f"{gt_t:.9f}",
            "time_diff": f"{time_diff:.9f}",
            "est_x": est_p[0],
            "est_y": est_p[1],
            "est_z": est_p[2],
            "gt_x": gt_p[0],
            "gt_y": gt_p[1],
            "gt_z": gt_p[2],
            "aligned_x": aligned[0],
            "aligned_y": aligned[1],
            "aligned_z": aligned[2],
            "ate": float(ate[row_idx]),
            "rpe_trans": rpe_trans,
            "rpe_rot_deg": rpe_rot_deg,
            "top_error": 1 if row_idx in top_indices else 0,
        }
        log_metrics = log_by_frame.get(frame_id, {})
        out["has_log_metrics"] = 1 if log_metrics else 0
        out["diag_top_error"] = 0
        out.update(log_metrics)
        rows.append(out)

        prev_aligned = aligned
        prev_gt = gt_p
        prev_est_rot = est_rot
        prev_gt_rot = gt_rot

    diag_row_indices = [idx for idx, row in enumerate(rows) if row.get("has_log_metrics") == 1]
    diag_top_n = 0
    if diag_row_indices:
        diag_top_n = max(
            1,
            int(math.ceil(len(diag_row_indices) * max(0.0, min(1.0, args.top_ratio)))),
        )
        diag_sorted = sorted(diag_row_indices, key=lambda idx: float(rows[idx]["ate"]), reverse=True)
        for idx in diag_sorted[:diag_top_n]:
            rows[idx]["diag_top_error"] = 1

    summary = {
        "gt_poses": len(gt),
        "est_poses": len(est),
        "matched_poses": len(matches),
        "log_frames": len(log_by_frame),
        "alignment": args.align,
        "scale": scale,
        "align_tx": align_trans[0],
        "align_ty": align_trans[1],
        "align_tz": align_trans[2],
        "rmse": float(math.sqrt(float(np.mean(ate * ate)))),
        "ate_mean": float(np.mean(ate)),
        "ate_median": float(np.median(ate)),
        "ate_p90": float(np.percentile(ate, 90.0)),
        "ate_max": float(np.max(ate)),
        "top_n": top_n,
        "diag_rows": len(diag_row_indices),
        "diag_top_n": diag_top_n,
    }
    return rows, summary


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    rows, summary = build_rows(args)
    top_rows = sorted(rows, key=lambda row: float(row["ate"]), reverse=True)[: args.top_k]
    deltas = summarize_metric_deltas(rows)[: args.top_k]
    diag_deltas = summarize_metric_deltas(rows, top_field="diag_top_error")[: args.top_k]

    top_path = out_path.with_name(out_path.stem + "_top_errors.csv")
    summary_path = out_path.with_name(out_path.stem + "_summary.txt")

    preferred = [
        "traj_index",
        "frame_id",
        "timestamp",
        "gt_timestamp",
        "time_diff",
        "ate",
        "rpe_trans",
        "rpe_rot_deg",
        "top_error",
    ]
    write_csv(out_path, rows, preferred)
    write_csv(top_path, top_rows, preferred)

    lines = [
        "Trajectory attribution summary",
        f"gt_poses={summary['gt_poses']} est_poses={summary['est_poses']} matched_poses={summary['matched_poses']} log_frames={summary['log_frames']}",
        f"alignment={summary['alignment']} scale={summary['scale']:.9f} t=({summary['align_tx']:.6f},{summary['align_ty']:.6f},{summary['align_tz']:.6f})",
        f"rmse={summary['rmse']:.6f} ate_mean={summary['ate_mean']:.6f} ate_median={summary['ate_median']:.6f} ate_p90={summary['ate_p90']:.6f} ate_max={summary['ate_max']:.6f}",
        f"top_n={summary['top_n']}",
        f"diag_rows={summary['diag_rows']} diag_top_n={summary['diag_top_n']}",
        "",
        "Top metric deltas on top-error frames:",
        "field,top_count,rest_count,top_mean,rest_mean,diff,relative_diff",
    ]
    for key, n_top, n_rest, top_mean, rest_mean, diff, rel in deltas:
        lines.append(
            f"{key},{n_top},{n_rest},{top_mean:.6f},{rest_mean:.6f},{diff:.6f},{rel:.6f}"
        )
    lines.extend(
        [
            "",
            "Top metric deltas on top-error diagnostic frames:",
            "field,top_count,rest_count,top_mean,rest_mean,diff,relative_diff",
        ]
    )
    for key, n_top, n_rest, top_mean, rest_mean, diff, rel in diag_deltas:
        lines.append(
            f"{key},{n_top},{n_rest},{top_mean:.6f},{rest_mean:.6f},{diff:.6f},{rel:.6f}"
        )
    summary_path.write_text("\n".join(lines) + "\n")

    print("\n".join(lines[:6]))
    print(f"wrote {out_path}")
    print(f"wrote {top_path}")
    print(f"wrote {summary_path}")
    if diag_deltas:
        print("largest diagnostic-frame metric deltas:")
        for key, n_top, n_rest, top_mean, rest_mean, diff, rel in diag_deltas[:10]:
            print(
                f"  {key}: top={top_mean:.6f} rest={rest_mean:.6f} diff={diff:.6f} rel={rel:.3f} n={n_top}/{n_rest}"
            )
    if deltas:
        print("largest metric deltas:")
        for key, n_top, n_rest, top_mean, rest_mean, diff, rel in deltas[:10]:
            print(
                f"  {key}: top={top_mean:.6f} rest={rest_mean:.6f} diff={diff:.6f} rel={rel:.3f} n={n_top}/{n_rest}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
