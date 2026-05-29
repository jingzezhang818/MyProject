#!/usr/bin/env python3
"""Segment-level trajectory and SLAM-structure comparison against ORB-SLAM3.

This is an offline diagnostic tool. It does not change SLAM behavior.

The script aligns this system and ORB-SLAM3 to the same ground truth using the
same SE3 default used by evo stereo evaluation, joins this system's debug log by
frame id, and aggregates trajectory error plus local-map/constraint structure by
fixed frame segments.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import importlib.util
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


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
        description="Aggregate trajectory-vs-ORB error and SLAM structure by frame segment."
    )
    parser.add_argument("--gt", required=True, help="EuRoC ground-truth CSV path")
    parser.add_argument("--ours", required=True, help="This system's TUM trajectory")
    parser.add_argument("--orb", required=True, help="ORB-SLAM3 TUM trajectory")
    parser.add_argument("--ours-log", required=True, help="Runtime log from this system")
    parser.add_argument("--out", required=True, help="Output per-segment CSV path")
    parser.add_argument(
        "--align",
        choices=("sim3", "se3", "none"),
        default="se3",
        help="Alignment mode. se3 is the default to match stereo evo --align evaluation.",
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
        help="Frame count per segment.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of worst segments and deltas to print/save.",
    )
    return parser.parse_args()


def aligned_rows(gt_path: Path, traj_path: Path, align: str, max_time_diff: float) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
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

    rows: List[Dict[str, object]] = []
    prev_aligned = None
    prev_gt = None
    prev_est_rot = None
    prev_gt_rot = None
    for row_idx, (est_idx, gt_idx, time_diff) in enumerate(matches):
        est_t, _, est_q = est[est_idx]
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
            }
        )
        prev_aligned = aligned[row_idx]
        prev_gt = gt_p
        prev_est_rot = est_rot
        prev_gt_rot = gt_rot

    summary = {
        "poses": float(len(est)),
        "matched": float(len(matches)),
        "rmse": math.sqrt(float(np.mean(ate * ate))),
        "mean": float(np.mean(ate)),
        "median": float(np.median(ate)),
        "p90": float(np.percentile(ate, 90.0)),
        "max": float(np.max(ate)),
        "scale": float(scale),
        "align_tx": float(align_trans[0]),
        "align_ty": float(align_trans[1]),
        "align_tz": float(align_trans[2]),
    }
    return rows, summary


def match_rows_by_time(
    ours: Sequence[Dict[str, object]],
    orb: Sequence[Dict[str, object]],
    max_time_diff: float,
) -> List[Tuple[Dict[str, object], Dict[str, object], float]]:
    orb_times = [float(row["timestamp"]) for row in orb]
    matches: List[Tuple[Dict[str, object], Dict[str, object], float]] = []
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


def nums(rows: Sequence[Dict[str, object]], key: str) -> List[float]:
    values = [maybe_float(row.get(key)) for row in rows]
    return [v for v in values if v is not None]


def mean(values: Sequence[float]) -> str:
    if not values:
        return ""
    return f"{sum(values) / len(values):.9f}"


def total(values: Sequence[float]) -> str:
    if not values:
        return ""
    return f"{sum(values):.9f}"


def ratio(num: Sequence[float], den: Sequence[float]) -> str:
    if not num or not den:
        return ""
    den_sum = sum(den)
    if abs(den_sum) <= 1e-12:
        return ""
    return f"{sum(num) / den_sum:.9f}"


def rmse(values: Sequence[float]) -> str:
    if not values:
        return ""
    return f"{math.sqrt(sum(v * v for v in values) / len(values)):.9f}"


def percentile(values: Sequence[float], pct: float) -> str:
    if not values:
        return ""
    np = require_numpy()
    return f"{float(np.percentile(values, pct)):.9f}"


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


AVG_KEYS = [
    "diag_mnMatchesInliers",
    "diag_assoc_before_search",
    "diag_assoc_before_opt",
    "diag_assoc_added_by_local",
    "diag_outlier_ratio_after_opt",
    "diag_mbMapUpdated",
    "payload_candidates",
    "payload_added",
    "payload_suppressed",
    "payload_suppressed_lmt_0_10",
    "payload_suppressed_tracking_stereo_0_10",
    "payload_suppressed_negative_depth",
    "payload_suppressed_outside_image",
    "payload_suppressed_mid_vote",
    "payload_suppressed_weak_vote",
    "payload_suppressed_expansion",
    "payload_kept_reference",
    "payload_kept_strong_vote",
    "payload_kept_weak_visible",
    "kfstruct_local_kfs",
    "kfstruct_local_kfs_voted_candidates",
    "kfstruct_local_kfs_voted_kept",
    "kfstruct_local_kfs_voted_skipped_by_budget",
    "kfstruct_local_kfs_voted_visibility_kept",
    "kfstruct_local_kfs_voted_visibility_skipped",
    "kfstruct_local_kf_vote_avg",
    "kfstruct_local_kf_vote_max",
    "kfstruct_local_mps",
    "kfstruct_local_mps_valid",
    "kfstruct_local_mps_track_in_view",
    "kfstruct_local_mps_in_frustum_ratio",
    "kfstruct_local_mps_from_reference",
    "kfstruct_local_mps_from_voted",
    "kfstruct_local_mps_from_expansion",
    "kfstruct_local_mps_from_temporal",
    "kfstruct_local_mps_core_confirmed",
    "kfstruct_local_mps_weak_vote",
    "kfstruct_local_mps_single_obs",
    "kfstruct_local_mps_lmt",
    "kfstruct_local_mps_lmt_0_10",
    "kfstruct_local_mps_tracking_stereo",
    "kfstruct_local_mps_tracking_stereo_0_10",
    "kfstruct_local_mps_depth_0_5",
    "kfstruct_local_mps_depth_5_10",
    "kfstruct_local_mps_depth_10_30",
    "kfstruct_local_mps_depth_avg",
    "stablevote_full_vote_kfs",
    "stablevote_stable_vote_kfs",
    "stablevote_full_votes",
    "stablevote_stable_votes",
    "stablevote_lmt_votes",
    "stablevote_lmt_stable_votes",
    "stablevote_lmt_weak_votes",
    "stablevote_lmt_0_10_votes",
    "stablevote_lmt_0_10_stable_votes",
    "stablevote_lmt_0_10_weak_votes",
    "stablevote_lmt_0_10_long_relevance_evaluated_points",
    "stablevote_lmt_0_10_long_low_relevant_points",
    "stablevote_lmt_0_10_long_low_relevant_votes",
    "stablevote_repeatability_vote_target_points",
    "stablevote_repeatability_vote_weak_points",
    "stablevote_repeatability_vote_weak_lowobs_points",
    "stablevote_repeatability_vote_weak_lowfound_points",
    "stablevote_repeatability_vote_weak_opt_unstable_points",
    "stablevote_repeatability_vote_weak_low_relevance_points",
    "stablevote_repeatability_vote_weak_raw",
    "stablevote_repeatability_vote_weak_effective",
    "stablevote_repeatability_vote_weak_suppressed",
    "stablevote_repeatability_vote_weak_kfs",
    "stablevote_repeatability_vote_behavior_active",
    "stablevote_mapping_probation_vote_target_points",
    "stablevote_mapping_probation_vote_active_points",
    "stablevote_mapping_probation_vote_promoted_points",
    "stablevote_mapping_probation_vote_raw",
    "stablevote_mapping_probation_vote_effective",
    "stablevote_mapping_probation_vote_suppressed",
    "stablevote_mapping_probation_vote_kfs",
    "stablevote_mapping_probation_vote_behavior_active",
    "stablevote_lmt_10p_votes",
    "stablevote_lmt_10p_stable_votes",
    "stablevote_lmt_10p_weak_votes",
    "stablevote_actual_local_kfs",
    "stablevote_shadow_direct_kfs",
    "stablevote_shadow_local_kfs",
    "stablevote_selected_kf_overlap",
    "stablevote_selected_kf_overlap_share",
    "stablevote_actual_local_mps",
    "stablevote_shadow_local_mps",
    "stablevote_actual_lmt_0_10",
    "stablevote_shadow_lmt_0_10",
    "stablevote_actual_pred_visible",
    "stablevote_shadow_pred_visible",
    "stablevote_actual_trackable_share",
    "stablevote_shadow_trackable_share",
    "stablevote_long_relevance_shadow_vote_kfs",
    "stablevote_long_relevance_shadow_top_vote",
    "stablevote_long_relevance_shadow_direct_kfs",
    "stablevote_long_relevance_shadow_direct_overlap",
    "stablevote_long_relevance_shadow_direct_overlap_share",
    "stablevote_long_relevance_shadow_direct_mps",
    "stablevote_long_relevance_shadow_direct_lmt_0_10",
    "stablevote_long_relevance_shadow_direct_pred_visible",
    "stablevote_long_relevance_shadow_direct_trackable_share",
    "kfcontribshadow_actual_kfs",
    "kfcontribshadow_shadow_kfs",
    "kfcontribshadow_shadow_core_kfs",
    "kfcontribshadow_selected_kf_overlap_share",
    "kfcontribshadow_target_visible_close",
    "kfcontribshadow_target_met",
    "kfcontribshadow_actual_mps",
    "kfcontribshadow_shadow_mps",
    "kfcontribshadow_shadow_mps_ratio",
    "kfcontribshadow_actual_close_mps",
    "kfcontribshadow_shadow_close_mps",
    "kfcontribshadow_actual_visible_total",
    "kfcontribshadow_shadow_visible_total",
    "kfcontribshadow_actual_visible_close",
    "kfcontribshadow_shadow_visible_close",
    "kfcontribshadow_shadow_visible_close_ratio",
    "kfcontribshadow_actual_visible_lmt_close",
    "kfcontribshadow_shadow_visible_lmt_close",
    "kfcontribshadow_actual_visible_tracking_stereo_close",
    "kfcontribshadow_shadow_visible_tracking_stereo_close",
    "kfcontribshadow_actual_negative_close",
    "kfcontribshadow_shadow_negative_close",
    "kfcontribshadow_actual_outside_close",
    "kfcontribshadow_shadow_outside_close",
    "kfcontribshadow_actual_scale_close",
    "kfcontribshadow_shadow_scale_close",
    "kfcontribshadow_actual_view_angle_close",
    "kfcontribshadow_shadow_view_angle_close",
    "kfcontribselect_active",
    "kfcontribselect_baseline_direct_kfs",
    "kfcontribselect_selected_direct_kfs",
    "kfcontribselect_core_kfs",
    "kfcontribselect_fallback_kfs",
    "kfcontribselect_baseline_direct_mps",
    "kfcontribselect_selected_direct_mps",
    "kfcontribselect_target_visible_total",
    "kfcontribselect_selected_visible_total",
    "kfcontribselect_target_visible_close",
    "kfcontribselect_selected_visible_close",
    "kfcontribselect_target_met",
    "kfcontribretain_all_local_proj_actual_total",
    "kfcontribretain_all_local_proj_retained_total",
    "kfcontribretain_all_local_proj_actual_inlier",
    "kfcontribretain_all_local_proj_retained_inlier",
    "kfcontribretain_all_local_proj_inlier_retention",
    "kfcontribretain_all_local_proj_actual_inlier_cells",
    "kfcontribretain_all_local_proj_retained_inlier_cells",
    "kfcontribretain_all_local_proj_inlier_cell_retention",
    "kfcontribretain_close_0_10_actual_total",
    "kfcontribretain_close_0_10_retained_total",
    "kfcontribretain_close_0_10_actual_inlier",
    "kfcontribretain_close_0_10_retained_inlier",
    "kfcontribretain_close_0_10_inlier_retention",
    "kfcontribretain_close_0_10_actual_inlier_cells",
    "kfcontribretain_close_0_10_retained_inlier_cells",
    "kfcontribretain_close_0_10_inlier_cell_retention",
    "kfcontribretain_lmt_close_0_10_actual_total",
    "kfcontribretain_lmt_close_0_10_retained_total",
    "kfcontribretain_lmt_close_0_10_actual_inlier",
    "kfcontribretain_lmt_close_0_10_retained_inlier",
    "kfcontribretain_lmt_close_0_10_inlier_retention",
    "kfcontribretain_lmt_close_0_10_actual_inlier_cells",
    "kfcontribretain_lmt_close_0_10_retained_inlier_cells",
    "kfcontribretain_lmt_close_0_10_inlier_cell_retention",
    "kfcontribretain_tracking_stereo_close_0_10_actual_total",
    "kfcontribretain_tracking_stereo_close_0_10_retained_total",
    "kfcontribretain_tracking_stereo_close_0_10_actual_inlier",
    "kfcontribretain_tracking_stereo_close_0_10_retained_inlier",
    "kfcontribretain_tracking_stereo_close_0_10_inlier_retention",
    "kfcontribretain_tracking_stereo_close_0_10_actual_inlier_cells",
    "kfcontribretain_tracking_stereo_close_0_10_retained_inlier_cells",
    "kfcontribretain_tracking_stereo_close_0_10_inlier_cell_retention",
    "kfcontribretain_non_close_local_proj_actual_total",
    "kfcontribretain_non_close_local_proj_retained_total",
    "kfcontribretain_non_close_local_proj_actual_inlier",
    "kfcontribretain_non_close_local_proj_retained_inlier",
    "kfcontribretain_non_close_local_proj_inlier_retention",
    "kfcontribretain_non_close_local_proj_actual_inlier_cells",
    "kfcontribretain_non_close_local_proj_retained_inlier_cells",
    "kfcontribretain_non_close_local_proj_inlier_cell_retention",
    "closebridge_all_0_5_features",
    "closebridge_all_0_5_no_mp",
    "closebridge_all_0_5_associated_mp",
    "closebridge_all_0_5_inlier",
    "closebridge_all_0_5_stable_inlier",
    "closebridge_all_0_5_stable_feature_share",
    "closebridge_all_5_10_features",
    "closebridge_all_5_10_no_mp",
    "closebridge_all_5_10_associated_mp",
    "closebridge_all_5_10_inlier",
    "closebridge_all_5_10_stable_inlier",
    "closebridge_all_5_10_stable_feature_share",
    "closebridge_tracking_stereo_0_5_stable_inlier",
    "closebridge_tracking_stereo_5_10_stable_inlier",
    "closebridge_local_mapping_triangulation_0_5_stable_inlier",
    "closebridge_local_mapping_triangulation_5_10_stable_inlier",
    "refbridge_all_0_5_ref_slots",
    "refbridge_all_0_5_predicted_visible",
    "refbridge_all_0_5_visible_share",
    "refbridge_all_0_5_current_inlier",
    "refbridge_all_0_5_current_inlier_share",
    "refbridge_all_0_5_stable_current_inlier_share",
    "refbridge_all_5_10_ref_slots",
    "refbridge_all_5_10_predicted_visible",
    "refbridge_all_5_10_visible_share",
    "refbridge_all_5_10_current_inlier",
    "refbridge_all_5_10_current_inlier_share",
    "refbridge_all_5_10_stable_current_inlier_share",
    "refbridge_all_10p_ref_slots",
    "refbridge_all_10p_predicted_visible",
    "refbridge_all_10p_visible_share",
    "refbridge_all_10p_current_inlier",
    "refbridge_all_10p_current_inlier_share",
    "refbridge_all_10p_stable_current_inlier_share",
    "refassoc_tracking_stereo_0_5_predicted_visible_target",
    "refassoc_tracking_stereo_0_5_motion_recovered",
    "refassoc_tracking_stereo_0_5_selected_local",
    "refassoc_tracking_stereo_0_5_admitted",
    "refassoc_tracking_stereo_0_5_projection_match",
    "refassoc_tracking_stereo_5_10_predicted_visible_target",
    "refassoc_tracking_stereo_5_10_motion_recovered",
    "refassoc_tracking_stereo_5_10_selected_local",
    "refassoc_tracking_stereo_5_10_admitted",
    "refassoc_tracking_stereo_5_10_projection_match",
    "refassoc_local_mapping_triangulation_0_5_predicted_visible_target",
    "refassoc_local_mapping_triangulation_0_5_motion_recovered",
    "refassoc_local_mapping_triangulation_0_5_selected_local",
    "refassoc_local_mapping_triangulation_0_5_admitted",
    "refassoc_local_mapping_triangulation_0_5_projection_match",
    "refassoc_local_mapping_triangulation_5_10_predicted_visible_target",
    "refassoc_local_mapping_triangulation_5_10_motion_recovered",
    "refassoc_local_mapping_triangulation_5_10_selected_local",
    "refassoc_local_mapping_triangulation_5_10_admitted",
    "refassoc_local_mapping_triangulation_5_10_projection_match",
    "search_matches",
    "search_total_matches",
    "search_local_kfs_voted_candidates",
    "search_local_kfs_voted_kept",
    "search_local_kfs_voted_skipped_by_budget",
    "search_local_kfs_voted_visibility_kept",
    "search_local_kfs_voted_visibility_skipped",
    "search_repeatability_vote_target_points",
    "search_repeatability_vote_weak_points",
    "search_repeatability_vote_weak_lowobs_points",
    "search_repeatability_vote_weak_lowfound_points",
    "search_repeatability_vote_weak_opt_unstable_points",
    "search_repeatability_vote_weak_low_relevance_points",
    "search_repeatability_vote_weak_raw",
    "search_repeatability_vote_weak_effective",
    "search_repeatability_vote_weak_suppressed",
    "search_repeatability_vote_weak_kfs",
    "search_repeatability_vote_behavior_active",
    "search_mapping_probation_vote_target_points",
    "search_mapping_probation_vote_active_points",
    "search_mapping_probation_vote_promoted_points",
    "search_mapping_probation_vote_raw",
    "search_mapping_probation_vote_effective",
    "search_mapping_probation_vote_suppressed",
    "search_mapping_probation_vote_kfs",
    "search_mapping_probation_vote_behavior_active",
    "search_core_pool_candidates",
    "search_core_pool_in_view",
    "search_core_pool_matches",
    "search_expansion_pool_candidates",
    "search_expansion_pool_enabled",
    "search_expansion_pool_input_capped",
    "search_expansion_pool_suppressed",
    "search_expansion_pool_matches",
    "search_expansion_pool_match_cap_hit",
    "search_probation_pool_candidates",
    "search_probation_pool_in_view",
    "search_probation_pool_enabled",
    "search_probation_pool_input_capped",
    "search_probation_pool_suppressed",
    "search_probation_pool_matches",
    "search_probation_pool_match_cap_hit",
    "search_localmap_lmt_0_10_from_reference",
    "search_localmap_lmt_0_10_from_voted",
    "search_localmap_lmt_0_10_from_expansion",
    "search_localmap_lmt_0_10_from_temporal",
    "search_localmap_lmt_0_10_from_unknown",
    "search_local_proj_triang_0_10_proj_outside_reject",
    "search_refassoc_matcher_input",
    "search_refassoc_matcher_projected",
    "search_refassoc_matcher_empty_area",
    "search_refassoc_matcher_occupied_candidate",
    "search_refassoc_matcher_no_valid_candidate",
    "search_refassoc_matcher_dist_reject",
    "search_refassoc_matcher_ratio_reject",
    "search_refassoc_matcher_candidate_pool",
    "search_refassoc_matcher_spatial_reject",
    "search_refassoc_matcher_assign_reject",
    "search_refassoc_matcher_outside_reject",
    "search_refassoc_matcher_assigned",
    "search_ref_fallback_shadow_target",
    "search_ref_fallback_shadow_found_area",
    "search_ref_fallback_shadow_area_candidates",
    "search_ref_fallback_shadow_valid_candidate",
    "search_ref_fallback_shadow_raw_candidate",
    "search_ref_fallback_shadow_ratio_pass",
    "search_ref_fallback_shadow_strict_pass",
    "search_ref_fallback_shadow_best_dist_avg",
    "search_ref_fallback_shadow_proj_norm_avg",
    "search_triang_0_10_history_gate_reject",
    "search_triang_0_10_visibility_history_recorded",
    "search_triang_0_10_visibility_history_evaluated",
    "search_triang_0_10_visibility_history_low_relevant",
    "search_triang_0_10_visibility_history_low_relevant_accepted",
    "xfeatstruct_all_valid",
    "xfeatstruct_all_non_empty",
    "xfeatstruct_all_top1",
    "xfeatstruct_all_top4",
    "xfeatstruct_all_cv",
    "xfeatstruct_matched_valid",
    "xfeatstruct_matched_non_empty",
    "xfeatstruct_matched_top1",
    "xfeatstruct_matched_top4",
    "xfeatstruct_matched_cv",
    "xfeatstruct_matched_to_all_ratio",
    "xfeatstruct_matched_coverage_ratio",
    "xfeatstruct_matched_octave_mean",
    "xfeatstruct_matched_octave_l0_share",
    "xfeatstruct_matched_octave_l01_share",
    "xfeatstruct_matched_octave_upper_share",
    "joint_local_proj_local_mapping_triangulation_total",
    "joint_local_proj_local_mapping_triangulation_outlier_ratio",
    "joint_local_proj_local_mapping_triangulation_inlier_err_avg",
    "joint_local_proj_tracking_stereo_total",
    "joint_local_proj_tracking_stereo_outlier_ratio",
    "joint_local_proj_tracking_stereo_inlier_err_avg",
    "grid_local_proj_lmt_0_10_total",
    "grid_local_proj_lmt_0_10_outlier_ratio",
    "grid_local_proj_lmt_0_10_inlier_top4",
    "grid_local_proj_lmt_0_10_inlier_cv",
    "grid_local_proj_lmt_edge_0_10_total",
    "grid_local_proj_lmt_edge_0_10_outlier_ratio",
    "grid_local_proj_tracking_stereo_0_10_total",
    "grid_local_proj_tracking_stereo_0_10_outlier_ratio",
    "pool_core_total",
    "pool_core_outlier_ratio",
    "pool_core_inlier_err_avg",
    "pool_expansion_total",
    "pool_expansion_outlier_ratio",
    "pool_expansion_inlier_err_avg",
    "pool_unknown_total",
    "pool_unknown_outlier_ratio",
    "mapfeatout_local_mapping_triangulation_grid_bin_edge_total",
    "mapfeatout_local_mapping_triangulation_grid_bin_edge_outlier_ratio",
    "mapfeatout_local_mapping_triangulation_grid_bin_center_total",
    "mapfeatout_local_mapping_triangulation_grid_bin_center_outlier_ratio",
    "mapfeatout_local_mapping_triangulation_octave_bin_octave0_total",
    "mapfeatout_local_mapping_triangulation_octave_bin_octave0_outlier_ratio",
    "mapfeatout_local_mapping_triangulation_octave_bin_octave2p_total",
    "mapfeatout_local_mapping_triangulation_octave_bin_octave2p_outlier_ratio",
    "mapfeatout_local_mapping_triangulation_response_bin_low_total",
    "mapfeatout_local_mapping_triangulation_response_bin_low_outlier_ratio",
    "mapfeatout_local_mapping_triangulation_response_bin_high_total",
    "mapfeatout_local_mapping_triangulation_response_bin_high_outlier_ratio",
    "mapfeatout_local_mapping_triangulation_mapping_feature_bin_edge_grid_total",
    "mapfeatout_local_mapping_triangulation_mapping_feature_bin_edge_grid_outlier_ratio",
    "mapfeatout_local_mapping_triangulation_mapping_feature_bin_balanced_total",
    "mapfeatout_local_mapping_triangulation_mapping_feature_bin_balanced_outlier_ratio",
    "mapfeatreuse_local_mapping_triangulation_grid_bin_edge_selected",
    "mapfeatreuse_local_mapping_triangulation_grid_bin_edge_in_frustum",
    "mapfeatreuse_local_mapping_triangulation_grid_bin_edge_admitted",
    "mapfeatreuse_local_mapping_triangulation_grid_bin_edge_projection_match",
    "mapfeatreuse_local_mapping_triangulation_grid_bin_center_selected",
    "mapfeatreuse_local_mapping_triangulation_grid_bin_center_in_frustum",
    "mapfeatreuse_local_mapping_triangulation_grid_bin_center_admitted",
    "mapfeatreuse_local_mapping_triangulation_grid_bin_center_projection_match",
    "mapfeatreuse_local_mapping_triangulation_octave_bin_octave0_selected",
    "mapfeatreuse_local_mapping_triangulation_octave_bin_octave0_projection_match",
    "mapfeatreuse_local_mapping_triangulation_octave_bin_octave2p_selected",
    "mapfeatreuse_local_mapping_triangulation_octave_bin_octave2p_projection_match",
    "mapfeatreuse_local_mapping_triangulation_response_bin_low_selected",
    "mapfeatreuse_local_mapping_triangulation_response_bin_low_projection_match",
    "mapfeatreuse_local_mapping_triangulation_response_bin_high_selected",
    "mapfeatreuse_local_mapping_triangulation_response_bin_high_projection_match",
    "mapfeatreuse_local_mapping_triangulation_creation_depth_bin_0_5_selected",
    "mapfeatreuse_local_mapping_triangulation_creation_depth_bin_0_5_projection_match",
    "mapfeatreuse_local_mapping_triangulation_creation_depth_bin_5_10_selected",
    "mapfeatreuse_local_mapping_triangulation_creation_depth_bin_5_10_projection_match",
    "mapfeatreuse_local_mapping_triangulation_mapping_feature_bin_edge_grid_selected",
    "mapfeatreuse_local_mapping_triangulation_mapping_feature_bin_edge_grid_projection_match",
    "mapfeatreuse_local_mapping_triangulation_mapping_feature_bin_balanced_selected",
    "mapfeatreuse_local_mapping_triangulation_mapping_feature_bin_balanced_projection_match",
    "mapfeatcreate_local_mapping_triangulation_grid_bin_edge_created",
    "mapfeatcreate_local_mapping_triangulation_grid_bin_center_created",
    "mapfeatcreate_local_mapping_triangulation_octave_bin_octave0_created",
    "mapfeatcreate_local_mapping_triangulation_octave_bin_octave2p_created",
    "mapfeatcreate_local_mapping_triangulation_response_bin_low_created",
    "mapfeatcreate_local_mapping_triangulation_response_bin_high_created",
    "mapfeatcreate_local_mapping_triangulation_mapping_feature_bin_edge_grid_created",
    "mapfeatcreate_local_mapping_triangulation_mapping_feature_bin_balanced_created",
    "triangcreate_close_candidates",
    "triangcreate_close_created",
    "triangcreate_close_0_5_created",
    "triangcreate_close_5_10_created",
    "triangcreate_close_stereo_fallback_reject",
    "triangcreate_mapping_feature_budget_candidate",
    "triangcreate_mapping_feature_budget_unique_candidates",
    "triangcreate_mapping_feature_budget_accept",
    "triangcreate_mapping_feature_budget_reject",
    "triangcreate_mapping_feature_budget_reject_grid",
    "triangcreate_mapping_feature_budget_reject_low_response",
    "triangcreate_mapping_feature_budget_reject_octave2p",
    "triangcreate_mapping_feature_budget_reject_edge_grid",
    "triangcreate_mapping_feature_budget_reject_duplicate",
    "triangcreate_mapping_feature_budget_reject_keypoint_duplicate",
    "triangcreate_mapping_feature_repeatability_candidate",
    "triangcreate_mapping_feature_repeatability_repeatable_candidate",
    "triangcreate_mapping_feature_repeatability_single_support_candidate",
    "triangcreate_mapping_feature_repeatability_accept",
    "triangcreate_mapping_feature_repeatability_fallback_accept",
    "triangcreate_mapping_feature_repeatability_reject_single_support",
    "triangcreate_mapping_feature_repeatability_shadow_reject_single_support",
    "triangcreate_mapping_tracklet_enabled",
    "triangcreate_mapping_tracklet_candidate",
    "triangcreate_mapping_tracklet_created",
    "triangcreate_mapping_tracklet_updated",
    "triangcreate_mapping_tracklet_promoted",
    "triangcreate_mapping_tracklet_expired",
    "triangcreate_mapping_tracklet_conflict",
    "triangcreate_mapping_tracklet_promotion_failed",
    "triangcreate_mapping_tracklet_alive",
    "kfcreate_mapping_feature_repeatability_candidate",
    "kfcreate_mapping_feature_repeatability_repeatable_candidate",
    "kfcreate_mapping_feature_repeatability_single_support_candidate",
    "kfcreate_mapping_feature_repeatability_accept",
    "kfcreate_mapping_feature_repeatability_fallback_accept",
    "kfcreate_mapping_feature_repeatability_reject_single_support",
    "kfcreate_mapping_feature_repeatability_shadow_reject_single_support",
    "kfcreate_mapping_tracklet_enabled",
    "kfcreate_mapping_tracklet_candidate",
    "kfcreate_mapping_tracklet_created",
    "kfcreate_mapping_tracklet_updated",
    "kfcreate_mapping_tracklet_promoted",
    "kfcreate_mapping_tracklet_expired",
    "kfcreate_mapping_tracklet_conflict",
    "kfcreate_mapping_tracklet_promotion_failed",
    "kfcreate_mapping_tracklet_alive",
    "kfcreate_mapping_tracklet_obs_avg",
    "kfcreate_mapping_tracklet_pass_avg",
    "kfcreate_mapping_tracklet_age_avg",
    "triangcreate_mapping_probation_created",
    "triangcreate_mapping_probation_created_0_5",
    "triangcreate_mapping_probation_created_5_10",
    "kfcreate_mapping_probation_created",
    "kfcreate_mapping_probation_created_0_5",
    "kfcreate_mapping_probation_created_5_10",
    "covisvote_mp_total",
    "covisvote_mp_local_mapping_triangulation",
    "covisvote_mp_lmt_0_10",
    "covisvote_mp_lmt_0_10_weak",
    "covisvote_mp_lmt_0_10_confirmed",
    "covisvote_vote_total",
    "covisvote_vote_local_mapping_triangulation",
    "covisvote_vote_lmt_0_10",
    "covisvote_vote_lmt_0_10_weak",
    "covisvote_vote_lmt_0_10_confirmed",
    "covisvote_shadow_skipped_votes",
    "covisvote_mp_lmt_creation_0_10_long_evaluated",
    "covisvote_mp_lmt_creation_0_10_long_low_relevant",
    "covisvote_vote_lmt_creation_0_10_long_low_relevant",
    "covisvote_long_relevance_shadow_skipped_votes",
    "covisvote_repeatability_vote_target_points",
    "covisvote_repeatability_vote_weak_points",
    "covisvote_repeatability_vote_weak_lowobs_points",
    "covisvote_repeatability_vote_weak_lowfound_points",
    "covisvote_repeatability_vote_weak_young_points",
    "covisvote_repeatability_vote_weak_low_relevance_points",
    "covisvote_repeatability_vote_weak_raw",
    "covisvote_repeatability_vote_weak_effective",
    "covisvote_repeatability_vote_weak_suppressed",
    "covisvote_repeatability_vote_weak_kfs",
    "covisvote_repeatability_vote_behavior_active",
    "covisvote_mapping_probation_target_points",
    "covisvote_mapping_probation_active_points",
    "covisvote_mapping_probation_promoted_points",
    "covisvote_mapping_probation_promoted_now_points",
    "covisvote_mapping_probation_vote_raw",
    "covisvote_mapping_probation_vote_effective",
    "covisvote_mapping_probation_vote_suppressed",
    "covisvote_mapping_probation_vote_kfs",
    "covisvote_mapping_probation_behavior_active",
    "covisvote_actual_kfs",
    "covisvote_actual_connected",
    "covisvote_actual_top_weight",
    "covisvote_shadow_kfs",
    "covisvote_shadow_connected",
    "covisvote_shadow_top_weight",
    "covisvote_shadow_connected_delta",
    "covisvote_shadow_top_weight_delta",
    "covisvote_shadow_dropped_connected",
    "covisvote_long_relevance_shadow_kfs",
    "covisvote_long_relevance_shadow_connected",
    "covisvote_long_relevance_shadow_top_weight",
    "covisvote_long_relevance_shadow_connected_delta",
    "covisvote_long_relevance_shadow_top_weight_delta",
    "covisvote_long_relevance_shadow_dropped_connected",
    "recentmp_total",
    "recentmp_kept",
    "recentmp_found_ratio_culled",
    "recentmp_low_obs_culled",
    "recentmp_aged_out_survived",
    "recentmp_mapping_probation_active",
    "recentmp_mapping_probation_kept",
    "recentmp_mapping_probation_promoted",
    "recentmp_mapping_probation_culled_no_inlier",
    "recentmp_mapping_probation_culled_expired",
    "recentmp_mapping_probation_culled_outlier",
    "recentmpsrc_local_mapping_triangulation_recent_total",
    "recentmpsrc_local_mapping_triangulation_kept",
    "recentmpsrc_local_mapping_triangulation_found_ratio_culled",
    "recentmpsrc_local_mapping_triangulation_low_obs_culled",
    "recentmpsrc_local_mapping_triangulation_aged_out_survived",
    "recentmpsrc_local_mapping_triangulation_mapping_probation_active",
    "recentmpsrc_local_mapping_triangulation_mapping_probation_kept",
    "recentmpsrc_local_mapping_triangulation_mapping_probation_promoted",
    "recentmpsrc_local_mapping_triangulation_mapping_probation_culled_no_inlier",
    "recentmpsrc_local_mapping_triangulation_mapping_probation_culled_expired",
    "recentmpsrc_local_mapping_triangulation_mapping_probation_culled_outlier",
    "reusefunnel_local_mapping_triangulation_0_5_selected",
    "reusefunnel_local_mapping_triangulation_0_5_not_in_frustum",
    "reusefunnel_local_mapping_triangulation_0_5_in_frustum",
    "reusefunnel_local_mapping_triangulation_0_5_admission_deferred",
    "reusefunnel_local_mapping_triangulation_0_5_admitted",
    "reusefunnel_local_mapping_triangulation_0_5_projection_match",
    "reusefunnel_local_mapping_triangulation_5_10_selected",
    "reusefunnel_local_mapping_triangulation_5_10_not_in_frustum",
    "reusefunnel_local_mapping_triangulation_5_10_in_frustum",
    "reusefunnel_local_mapping_triangulation_5_10_admission_deferred",
    "reusefunnel_local_mapping_triangulation_5_10_admitted",
    "reusefunnel_local_mapping_triangulation_5_10_projection_match",
    "reusefunnel_tracking_stereo_0_5_selected",
    "reusefunnel_tracking_stereo_0_5_not_in_frustum",
    "reusefunnel_tracking_stereo_0_5_in_frustum",
    "reusefunnel_tracking_stereo_0_5_admission_deferred",
    "reusefunnel_tracking_stereo_0_5_admitted",
    "reusefunnel_tracking_stereo_0_5_projection_match",
    "reusefunnel_tracking_stereo_5_10_selected",
    "reusefunnel_tracking_stereo_5_10_not_in_frustum",
    "reusefunnel_tracking_stereo_5_10_in_frustum",
    "reusefunnel_tracking_stereo_5_10_admission_deferred",
    "reusefunnel_tracking_stereo_5_10_admitted",
    "reusefunnel_tracking_stereo_5_10_projection_match",
    "reusefrustum_local_mapping_triangulation_0_5_outside_image_total",
    "reusefrustum_local_mapping_triangulation_0_5_scale_range_total",
    "reusefrustum_local_mapping_triangulation_0_5_view_angle_total",
    "reusefrustum_local_mapping_triangulation_5_10_outside_image_total",
    "reusefrustum_local_mapping_triangulation_5_10_scale_range_total",
    "reusefrustum_local_mapping_triangulation_5_10_view_angle_total",
    "reusefrustum_tracking_stereo_0_5_outside_image_total",
    "reusefrustum_tracking_stereo_0_5_scale_range_total",
    "reusefrustum_tracking_stereo_0_5_view_angle_total",
    "reusefrustum_tracking_stereo_5_10_outside_image_total",
    "reusefrustum_tracking_stereo_5_10_scale_range_total",
    "reusefrustum_tracking_stereo_5_10_view_angle_total",
    "reusecontrib_local_mapping_triangulation_reference_selected",
    "reusecontrib_local_mapping_triangulation_reference_projection_match",
    "reusecontrib_local_mapping_triangulation_strong_vote_selected",
    "reusecontrib_local_mapping_triangulation_strong_vote_projection_match",
    "reusecontrib_local_mapping_triangulation_weak_vote_selected",
    "reusecontrib_local_mapping_triangulation_weak_vote_projection_match",
    "reusecontrib_local_mapping_triangulation_expansion_selected",
    "reusecontrib_local_mapping_triangulation_expansion_projection_match",
    "reusecontribout_local_mapping_triangulation_reference_outlier_ratio",
    "reusecontribout_local_mapping_triangulation_strong_vote_outlier_ratio",
    "reusecontribout_local_mapping_triangulation_weak_vote_outlier_ratio",
    "reusecontribout_local_mapping_triangulation_expansion_outlier_ratio",
    "localkfclose_selected_kfs",
    "localkfclose_frustum_kfs",
    "localkfclose_match_kfs",
    "localkfclose_zero_frustum_kfs",
    "localkfclose_no_match_kfs",
    "localkfclose_selected",
    "localkfclose_in_frustum",
    "localkfclose_projection_match",
    "reuseoutcome_local_mapping_triangulation_0_5_total",
    "reuseoutcome_local_mapping_triangulation_0_5_outlier_ratio",
    "reuseoutcome_local_mapping_triangulation_5_10_total",
    "reuseoutcome_local_mapping_triangulation_5_10_outlier_ratio",
    "reuseoutcome_tracking_stereo_0_5_total",
    "reuseoutcome_tracking_stereo_0_5_outlier_ratio",
    "reuseoutcome_tracking_stereo_5_10_total",
    "reuseoutcome_tracking_stereo_5_10_outlier_ratio",
    "reusecohort_local_mapping_triangulation_0_5_h10_active_eligible",
    "reusecohort_local_mapping_triangulation_0_5_h10_inlier_rate",
    "reusecohort_local_mapping_triangulation_5_10_h10_active_eligible",
    "reusecohort_local_mapping_triangulation_5_10_h10_inlier_rate",
    "reusecohort_tracking_stereo_0_5_h10_active_eligible",
    "reusecohort_tracking_stereo_0_5_h10_inlier_rate",
    "reusecohort_tracking_stereo_5_10_h10_active_eligible",
    "reusecohort_tracking_stereo_5_10_h10_inlier_rate",
    "maprepeat_local_mapping_triangulation_0_5_mapping_feature_bin_edge_grid_h10_eligible",
    "maprepeat_local_mapping_triangulation_0_5_mapping_feature_bin_edge_grid_h10_inlier_rate",
    "maprepeat_local_mapping_triangulation_0_5_mapping_feature_bin_balanced_h10_eligible",
    "maprepeat_local_mapping_triangulation_0_5_mapping_feature_bin_balanced_h10_inlier_rate",
    "maprepeat_local_mapping_triangulation_5_10_mapping_feature_bin_edge_grid_h10_eligible",
    "maprepeat_local_mapping_triangulation_5_10_mapping_feature_bin_edge_grid_h10_inlier_rate",
    "maprepeat_local_mapping_triangulation_5_10_mapping_feature_bin_balanced_h10_eligible",
    "maprepeat_local_mapping_triangulation_5_10_mapping_feature_bin_balanced_h10_inlier_rate",
]

KF_INSERT_NOVELTY_KEYS = [
    "frames_since_last_kf",
    "n_tracked_close",
    "n_non_tracked_close",
    "novelty_inlier_assoc",
    "novelty_stable_inlier_assoc",
    "novelty_stable_inlier_share",
    "novelty_ref_overlap_inlier",
    "novelty_ref_overlap_share",
    "novelty_stable_ref_overlap_share",
    "novelty_inlier_cells",
    "novelty_stable_inlier_cells",
    "novelty_tracked_close_cells",
    "novelty_nontracked_close_cells",
    "novelty_new_close_cells",
    "novelty_new_close_cell_share",
    "novelty_nontracked_close_highconf",
    "novelty_highconf_new_close_cells",
    "xfeat_kf_shadow_eligible",
    "xfeat_kf_shadow_would_insert",
    "xfeat_kf_shadow_would_suppress",
    "xfeat_kf_shadow_weak_support",
    "xfeat_kf_shadow_stable_support_ok",
    "xfeat_kf_shadow_close_coverage_ok",
    "xfeat_kf_shadow_novel_close_strong",
    "xfeat_kf_shadow_close_insert",
    "xfeat_kf_shadow_weak_inlier_threshold",
    "xfeat_kf_guard_active",
    "xfeat_kf_guard_suppress",
    "xfeat_kf_guard_allow_weak_support",
    "xfeat_kf_guard_allow_close_insert",
    "xfeat_kf_guard_force_insert",
]


def add_ratio_metric(out: Dict[str, object], rows: Sequence[Dict[str, object]], name: str, num_key: str, den_key: str) -> None:
    out[name] = ratio(nums(rows, num_key), nums(rows, den_key))


def add_share_metric(out: Dict[str, object], rows: Sequence[Dict[str, object]], name: str, num_key: str, den_key: str) -> None:
    ratios: List[float] = []
    for row in rows:
        num = maybe_float(row.get(num_key))
        den = maybe_float(row.get(den_key))
        if num is None or den is None or den <= 0.0:
            continue
        ratios.append(num / den)
    out[name] = mean(ratios)


def build_matched_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Dict[str, float], Dict[str, float]]:
    ours_rows, ours_summary = aligned_rows(Path(args.gt), Path(args.ours), args.align, args.max_time_diff)
    orb_rows, orb_summary = aligned_rows(Path(args.gt), Path(args.orb), args.align, args.max_time_diff)
    matches = match_rows_by_time(ours_rows, orb_rows, args.max_time_diff)
    if len(matches) < 3:
        raise SystemExit(f"only {len(matches)} ours/ORB timestamp matches found")

    log_by_frame = attr.parse_log(Path(args.ours_log))
    rows: List[Dict[str, object]] = []
    for ours, orb, orb_time_diff in matches:
        frame_id = int(ours["traj_index"]) + args.frame_id_offset
        out: Dict[str, object] = {
            "frame_id": frame_id,
            "timestamp": f"{float(ours['timestamp']):.9f}",
            "orb_timestamp": f"{float(orb['timestamp']):.9f}",
            "orb_time_diff": f"{orb_time_diff:.9f}",
            "ate_ours": float(ours["ate"]),
            "ate_orb": float(orb["ate"]),
            "ate_diff": float(ours["ate"]) - float(orb["ate"]),
            "rpe_trans_ours": "" if ours["rpe_trans"] is None else ours["rpe_trans"],
            "rpe_trans_orb": "" if orb["rpe_trans"] is None else orb["rpe_trans"],
            "rpe_trans_diff": "" if ours["rpe_trans"] is None or orb["rpe_trans"] is None else ours["rpe_trans"] - orb["rpe_trans"],
            "rpe_rot_deg_ours": "" if ours["rpe_rot_deg"] is None else ours["rpe_rot_deg"],
            "rpe_rot_deg_orb": "" if orb["rpe_rot_deg"] is None else orb["rpe_rot_deg"],
            "rpe_rot_deg_diff": "" if ours["rpe_rot_deg"] is None or orb["rpe_rot_deg"] is None else ours["rpe_rot_deg"] - orb["rpe_rot_deg"],
        }
        for key, value in log_by_frame.get(frame_id, {}).items():
            if key != "frame_id":
                out[key] = value
        out["has_log_metrics"] = 1 if frame_id in log_by_frame else 0
        rows.append(out)
    return rows, ours_summary, orb_summary


def segment_rows(rows: Sequence[Dict[str, object]], segment_size: int) -> List[Dict[str, object]]:
    if segment_size <= 0:
        segment_size = 100
    if not rows:
        return []
    max_frame = max(int(row["frame_id"]) for row in rows)
    segments: List[Dict[str, object]] = []
    for start in range(0, max_frame + 1, segment_size):
        end = start + segment_size
        seg = [row for row in rows if start <= int(row["frame_id"]) < end]
        if not seg:
            continue
        diag = [row for row in seg if int(row.get("has_log_metrics", 0)) == 1]
        out: Dict[str, object] = {
            "segment_start": start,
            "segment_end": end - 1,
            "matched_frames": len(seg),
            "diag_frames": len(diag),
            "ate_ours_rmse": rmse(nums(seg, "ate_ours")),
            "ate_orb_rmse": rmse(nums(seg, "ate_orb")),
            "ate_diff_mean": mean(nums(seg, "ate_diff")),
            "ate_diff_rmse": rmse(nums(seg, "ate_diff")),
            "ate_diff_p90": percentile(nums(seg, "ate_diff"), 90.0),
            "ate_diff_max": percentile(nums(seg, "ate_diff"), 100.0),
            "worse_ratio": f"{sum(1 for row in seg if float(row['ate_diff']) > 0.0) / len(seg):.9f}",
            "rpe_trans_ours_mean": mean(nums(seg, "rpe_trans_ours")),
            "rpe_trans_orb_mean": mean(nums(seg, "rpe_trans_orb")),
            "rpe_trans_diff_mean": mean(nums(seg, "rpe_trans_diff")),
            "rpe_rot_diff_mean": mean(nums(seg, "rpe_rot_deg_diff")),
        }

        for key in AVG_KEYS:
            out[f"{key}_avg"] = mean(nums(diag, key))

        add_ratio_metric(out, diag, "diag_outlier_after_opt_weighted_ratio", "diag_outlier_after_opt", "diag_assoc_after_opt")
        add_ratio_metric(out, diag, "joint_lmt_weighted_outlier_ratio", "joint_local_proj_local_mapping_triangulation_outlier", "joint_local_proj_local_mapping_triangulation_total")
        add_ratio_metric(out, diag, "joint_tracking_stereo_weighted_outlier_ratio", "joint_local_proj_tracking_stereo_outlier", "joint_local_proj_tracking_stereo_total")
        add_ratio_metric(out, diag, "grid_lmt_0_10_weighted_outlier_ratio", "grid_local_proj_lmt_0_10_outlier", "grid_local_proj_lmt_0_10_total")
        add_ratio_metric(out, diag, "grid_lmt_edge_0_10_weighted_outlier_ratio", "grid_local_proj_lmt_edge_0_10_outlier", "grid_local_proj_lmt_edge_0_10_total")
        add_ratio_metric(out, diag, "grid_tracking_stereo_0_10_weighted_outlier_ratio", "grid_local_proj_tracking_stereo_0_10_outlier", "grid_local_proj_tracking_stereo_0_10_total")
        add_ratio_metric(out, diag, "pool_core_weighted_outlier_ratio", "pool_core_outlier", "pool_core_total")
        add_ratio_metric(out, diag, "pool_expansion_weighted_outlier_ratio", "pool_expansion_outlier", "pool_expansion_total")

        add_share_metric(out, diag, "kfstruct_track_in_view_share", "kfstruct_local_mps_track_in_view", "kfstruct_local_mps")
        add_share_metric(out, diag, "kfstruct_voted_mp_share", "kfstruct_local_mps_from_voted", "kfstruct_local_mps")
        add_share_metric(out, diag, "kfstruct_reference_mp_share", "kfstruct_local_mps_from_reference", "kfstruct_local_mps")
        add_share_metric(out, diag, "kfstruct_core_confirmed_share", "kfstruct_local_mps_core_confirmed", "kfstruct_local_mps")
        add_share_metric(out, diag, "kfstruct_weak_vote_share", "kfstruct_local_mps_weak_vote", "kfstruct_local_mps")
        add_share_metric(out, diag, "kfstruct_single_obs_share", "kfstruct_local_mps_single_obs", "kfstruct_local_mps")
        add_share_metric(out, diag, "kfstruct_lmt_0_10_share", "kfstruct_local_mps_lmt_0_10", "kfstruct_local_mps")
        add_share_metric(out, diag, "kfstruct_tracking_stereo_0_10_share", "kfstruct_local_mps_tracking_stereo_0_10", "kfstruct_local_mps")
        add_share_metric(out, diag, "payload_suppressed_share", "payload_suppressed", "payload_candidates")
        add_share_metric(out, diag, "payload_suppressed_lmt_0_10_share", "payload_suppressed_lmt_0_10", "payload_candidates")
        add_share_metric(out, diag, "payload_suppressed_tracking_stereo_0_10_share", "payload_suppressed_tracking_stereo_0_10", "payload_candidates")
        add_share_metric(out, diag, "covisvote_lmt_0_10_weak_vote_share", "covisvote_vote_lmt_0_10_weak", "covisvote_vote_total")
        add_share_metric(out, diag, "covisvote_lmt_0_10_vote_share", "covisvote_vote_lmt_0_10", "covisvote_vote_total")
        add_share_metric(out, diag, "covisvote_shadow_skipped_vote_share", "covisvote_shadow_skipped_votes", "covisvote_vote_total")
        add_share_metric(out, diag, "covisvote_long_relevance_shadow_skipped_vote_share", "covisvote_long_relevance_shadow_skipped_votes", "covisvote_vote_total")
        add_share_metric(out, diag, "covisvote_long_low_relevant_mp_share", "covisvote_mp_lmt_creation_0_10_long_low_relevant", "covisvote_mp_lmt_creation_0_10_long_evaluated")
        add_share_metric(out, diag, "covisvote_repeatability_weak_point_share", "covisvote_repeatability_vote_weak_points", "covisvote_repeatability_vote_target_points")
        add_share_metric(out, diag, "covisvote_repeatability_weak_vote_suppressed_share", "covisvote_repeatability_vote_weak_suppressed", "covisvote_repeatability_vote_weak_raw")
        add_share_metric(out, diag, "covisvote_mapping_probation_active_point_share", "covisvote_mapping_probation_active_points", "covisvote_mapping_probation_target_points")
        add_share_metric(out, diag, "covisvote_mapping_probation_promoted_point_share", "covisvote_mapping_probation_promoted_points", "covisvote_mapping_probation_target_points")
        add_share_metric(out, diag, "covisvote_mapping_probation_vote_suppressed_share", "covisvote_mapping_probation_vote_suppressed", "covisvote_mapping_probation_vote_raw")
        add_share_metric(out, diag, "triangcreate_mapping_feature_repeatability_repeatable_share", "triangcreate_mapping_feature_repeatability_repeatable_candidate", "triangcreate_mapping_feature_repeatability_candidate")
        add_share_metric(out, diag, "triangcreate_mapping_feature_repeatability_single_support_share", "triangcreate_mapping_feature_repeatability_single_support_candidate", "triangcreate_mapping_feature_repeatability_candidate")
        add_share_metric(out, diag, "triangcreate_mapping_feature_repeatability_accept_share", "triangcreate_mapping_feature_repeatability_accept", "triangcreate_mapping_feature_repeatability_candidate")
        add_share_metric(out, diag, "triangcreate_mapping_feature_repeatability_fallback_accept_share", "triangcreate_mapping_feature_repeatability_fallback_accept", "triangcreate_mapping_feature_repeatability_single_support_candidate")
        add_share_metric(out, diag, "triangcreate_mapping_feature_repeatability_reject_single_support_share", "triangcreate_mapping_feature_repeatability_reject_single_support", "triangcreate_mapping_feature_repeatability_single_support_candidate")
        add_share_metric(out, diag, "triangcreate_mapping_feature_repeatability_shadow_reject_single_support_share", "triangcreate_mapping_feature_repeatability_shadow_reject_single_support", "triangcreate_mapping_feature_repeatability_single_support_candidate")
        add_share_metric(out, diag, "triangcreate_mapping_tracklet_created_share", "triangcreate_mapping_tracklet_created", "triangcreate_mapping_tracklet_candidate")
        add_share_metric(out, diag, "triangcreate_mapping_tracklet_updated_share", "triangcreate_mapping_tracklet_updated", "triangcreate_mapping_tracklet_candidate")
        add_share_metric(out, diag, "triangcreate_mapping_tracklet_promoted_share", "triangcreate_mapping_tracklet_promoted", "triangcreate_mapping_tracklet_candidate")
        add_share_metric(out, diag, "triangcreate_mapping_tracklet_expired_share", "triangcreate_mapping_tracklet_expired", "triangcreate_mapping_tracklet_candidate")
        add_share_metric(out, diag, "recentmp_mapping_probation_promoted_share", "recentmp_mapping_probation_promoted", "recentmp_mapping_probation_active")
        add_share_metric(out, diag, "recentmp_mapping_probation_culled_no_inlier_share", "recentmp_mapping_probation_culled_no_inlier", "recentmp_mapping_probation_active")
        add_share_metric(out, diag, "recentmp_mapping_probation_culled_expired_share", "recentmp_mapping_probation_culled_expired", "recentmp_mapping_probation_active")
        add_share_metric(out, diag, "recentmp_mapping_probation_culled_outlier_share", "recentmp_mapping_probation_culled_outlier", "recentmp_mapping_probation_active")
        add_share_metric(out, diag, "stablevote_lmt_weak_vote_share", "stablevote_lmt_weak_votes", "stablevote_lmt_votes")
        add_share_metric(out, diag, "stablevote_lmt_0_10_weak_vote_share", "stablevote_lmt_0_10_weak_votes", "stablevote_lmt_0_10_votes")
        add_share_metric(out, diag, "stablevote_lmt_10p_weak_vote_share", "stablevote_lmt_10p_weak_votes", "stablevote_lmt_10p_votes")
        add_share_metric(out, diag, "stablevote_long_low_relevant_point_share", "stablevote_lmt_0_10_long_low_relevant_points", "stablevote_lmt_0_10_long_relevance_evaluated_points")
        add_share_metric(out, diag, "stablevote_repeatability_weak_point_share", "stablevote_repeatability_vote_weak_points", "stablevote_repeatability_vote_target_points")
        add_share_metric(out, diag, "stablevote_repeatability_weak_vote_suppressed_share", "stablevote_repeatability_vote_weak_suppressed", "stablevote_repeatability_vote_weak_raw")
        add_share_metric(out, diag, "stablevote_mapping_probation_active_point_share", "stablevote_mapping_probation_vote_active_points", "stablevote_mapping_probation_vote_target_points")
        add_share_metric(out, diag, "stablevote_mapping_probation_vote_suppressed_share", "stablevote_mapping_probation_vote_suppressed", "stablevote_mapping_probation_vote_raw")
        add_share_metric(out, diag, "search_repeatability_weak_point_share", "search_repeatability_vote_weak_points", "search_repeatability_vote_target_points")
        add_share_metric(out, diag, "search_repeatability_weak_vote_suppressed_share", "search_repeatability_vote_weak_suppressed", "search_repeatability_vote_weak_raw")
        add_share_metric(out, diag, "search_mapping_probation_active_point_share", "search_mapping_probation_vote_active_points", "search_mapping_probation_vote_target_points")
        add_share_metric(out, diag, "search_mapping_probation_vote_suppressed_share", "search_mapping_probation_vote_suppressed", "search_mapping_probation_vote_raw")
        add_share_metric(out, diag, "search_long_low_relevant_share", "search_triang_0_10_visibility_history_low_relevant", "search_triang_0_10_visibility_history_evaluated")
        add_share_metric(out, diag, "search_long_low_relevant_accepted_share", "search_triang_0_10_visibility_history_low_relevant_accepted", "search_triang_0_10_visibility_history_low_relevant")
        add_share_metric(out, diag, "closebridge_0_5_no_mp_share", "closebridge_all_0_5_no_mp", "closebridge_all_0_5_features")
        add_share_metric(out, diag, "closebridge_5_10_no_mp_share", "closebridge_all_5_10_no_mp", "closebridge_all_5_10_features")
        add_share_metric(out, diag, "closebridge_0_5_stable_share", "closebridge_all_0_5_stable_inlier", "closebridge_all_0_5_features")
        add_share_metric(out, diag, "closebridge_5_10_stable_share", "closebridge_all_5_10_stable_inlier", "closebridge_all_5_10_features")
        add_share_metric(out, diag, "refbridge_0_5_current_inlier_share", "refbridge_all_0_5_current_inlier", "refbridge_all_0_5_ref_slots")
        add_share_metric(out, diag, "refbridge_5_10_current_inlier_share", "refbridge_all_5_10_current_inlier", "refbridge_all_5_10_ref_slots")
        add_share_metric(out, diag, "refbridge_10p_current_inlier_share", "refbridge_all_10p_current_inlier", "refbridge_all_10p_ref_slots")
        for source in ("tracking_stereo", "local_mapping_triangulation"):
            for depth_bin in ("0_5", "5_10"):
                prefix = f"refassoc_{source}_{depth_bin}"
                add_share_metric(out, diag, f"{prefix}_selected_share",
                                 f"{prefix}_selected_local", f"{prefix}_predicted_visible_target")
                add_share_metric(out, diag, f"{prefix}_admitted_share",
                                 f"{prefix}_admitted", f"{prefix}_predicted_visible_target")
                add_share_metric(out, diag, f"{prefix}_projection_match_share",
                                 f"{prefix}_projection_match", f"{prefix}_predicted_visible_target")
        for dim, bins in (
            ("grid_bin", ("edge", "center")),
            ("octave_bin", ("octave0", "octave2p")),
            ("response_bin", ("low", "high")),
            ("creation_depth_bin", ("0_5", "5_10")),
            ("mapping_feature_bin", ("edge_grid", "balanced")),
        ):
            for bin_name in bins:
                prefix = f"mapfeatreuse_local_mapping_triangulation_{dim}_{bin_name}"
                add_share_metric(out, diag, f"{prefix}_frustum_share",
                                 f"{prefix}_in_frustum", f"{prefix}_selected")
                add_share_metric(out, diag, f"{prefix}_projection_match_share",
                                 f"{prefix}_projection_match", f"{prefix}_selected")
                add_share_metric(out, diag, f"{prefix}_admitted_match_share",
                                 f"{prefix}_projection_match", f"{prefix}_admitted")
        for dim, bins in (
            ("mapping_feature_bin", ("edge_grid", "balanced")),
            ("octave_bin", ("octave0", "octave2p")),
            ("response_bin", ("low", "high")),
        ):
            for bin_name in bins:
                for horizon in ("5", "10", "20"):
                    for depth_bin in ("0_5", "5_10"):
                        prefix = f"maprepeat_local_mapping_triangulation_{depth_bin}_{dim}_{bin_name}_h{horizon}"
                        add_share_metric(out, diag, f"{prefix}_attempt_share",
                                         f"{prefix}_attempted", f"{prefix}_eligible")
                        add_share_metric(out, diag, f"{prefix}_match_share",
                                         f"{prefix}_matched", f"{prefix}_eligible")
                        add_share_metric(out, diag, f"{prefix}_inlier_share",
                                         f"{prefix}_opt_inlier", f"{prefix}_eligible")

        insert_rows = [
            row for row in diag
            if (
                str(row.get("kfdecision_created_kf", "")) == "1"
                if "kfdecision_created_kf" in row
                else str(row.get("kfdecision_decision", "")).startswith("insert")
            )
        ]
        decision_rows = [row for row in diag if "kfdecision_decision" in row]
        out["kfdecision_insert_count"] = len(insert_rows)
        out["kfdecision_created_kf_per_100f"] = "" if not seg else f"{100.0 * len(insert_rows) / len(seg):.9f}"
        out["kfdecision_insert_rate"] = "" if not decision_rows else f"{len(insert_rows) / len(decision_rows):.9f}"
        out["kfdecision_need_insert_close_rate"] = mean([
            1.0 if str(row.get("kfdecision_need_insert_close", "0")) in ("1", "true", "True") else 0.0
            for row in decision_rows
        ])
        out["kfdecision_xfeat_shadow_suppress_rate"] = mean([
            1.0 if str(row.get("kfdecision_xfeat_kf_shadow_would_suppress", "0")) in ("1", "true", "True") else 0.0
            for row in decision_rows
        ])
        out["kfdecision_xfeat_shadow_insert_rate"] = mean([
            1.0 if str(row.get("kfdecision_xfeat_kf_shadow_would_insert", "0")) in ("1", "true", "True") else 0.0
            for row in decision_rows
        ])
        out["kfdecision_xfeat_guard_active_rate"] = mean([
            1.0 if str(row.get("kfdecision_xfeat_kf_guard_active", "0")) in ("1", "true", "True") else 0.0
            for row in decision_rows
        ])
        out["kfdecision_xfeat_guard_suppress_rate"] = mean([
            1.0 if str(row.get("kfdecision_xfeat_kf_guard_suppress", "0")) in ("1", "true", "True") else 0.0
            for row in decision_rows
        ])
        for key in KF_INSERT_NOVELTY_KEYS:
            out[f"kfinsert_{key}_avg"] = mean(nums(insert_rows, f"kfdecision_{key}"))
        out["kfinsert_one_frame_share"] = mean([
            1.0 if maybe_float(row.get("kfdecision_frames_since_previous_kf")) == 1.0 else 0.0
            for row in insert_rows
            if maybe_float(row.get("kfdecision_frames_since_previous_kf")) is not None
        ])

        segments.append(out)
    return segments


def metric_deltas(
    segments: Sequence[Dict[str, object]],
    top_k: int,
    prefix_filter: Optional[Tuple[str, ...]] = None,
) -> List[Tuple[str, int, int, float, float, float, float]]:
    base_fields = {
        "segment_start",
        "segment_end",
        "matched_frames",
        "diag_frames",
        "ate_ours_rmse",
        "ate_orb_rmse",
        "ate_diff_mean",
        "ate_diff_rmse",
        "ate_diff_p90",
        "ate_diff_max",
        "worse_ratio",
        "rpe_trans_ours_mean",
        "rpe_trans_orb_mean",
        "rpe_trans_diff_mean",
        "rpe_rot_diff_mean",
    }
    ranked = sorted(segments, key=lambda row: maybe_float(row.get("ate_diff_mean")) or -1e9, reverse=True)
    top_n = max(1, min(top_k, max(1, len(segments) // 4)))
    top_ids = {(row["segment_start"], row["segment_end"]) for row in ranked[:top_n]}
    keys = sorted(set().union(*(row.keys() for row in segments)) - base_fields)
    if prefix_filter:
        keys = [key for key in keys if key.startswith(prefix_filter)]
    deltas: List[Tuple[str, int, int, float, float, float, float]] = []
    for key in keys:
        top_vals = [
            maybe_float(row.get(key))
            for row in segments
            if (row["segment_start"], row["segment_end"]) in top_ids
        ]
        rest_vals = [
            maybe_float(row.get(key))
            for row in segments
            if (row["segment_start"], row["segment_end"]) not in top_ids
        ]
        top_nums = [v for v in top_vals if v is not None]
        rest_nums = [v for v in rest_vals if v is not None]
        if len(top_nums) < 2 or len(rest_nums) < 2:
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
    rows, ours_summary, orb_summary = build_matched_rows(args)
    segments = segment_rows(rows, args.segment_size)
    if not segments:
        raise SystemExit("no segments produced")

    out_path = Path(args.out)
    top_path = out_path.with_name(out_path.stem + "_top_segments.csv")
    summary_path = out_path.with_name(out_path.stem + "_summary.txt")

    write_csv(out_path, segments)
    write_csv(
        top_path,
        sorted(segments, key=lambda row: maybe_float(row.get("ate_diff_mean")) or -1e9, reverse=True)[: args.top_k],
    )

    all_deltas = metric_deltas(segments, args.top_k)
    structure_prefixes = (
        "diag_",
        "kfstruct_",
        "stablevote_",
        "kfcontribshadow_",
        "kfcontribselect_",
        "kfcontribretain_",
        "closebridge_",
        "refbridge_",
        "refassoc_",
        "refassocfrustum_",
        "search_",
        "xfeatstruct_",
        "joint_",
        "grid_",
        "mapfeatout_",
        "mapfeatreuse_",
        "mapfeatcreate_",
        "triangcreate_",
        "recentmp_",
        "recentmpsrc_",
        "covisvote_",
        "kfinsert_",
        "kfdecision_",
        "reusefunnel_",
        "reusefrustum_",
        "reusecontrib_",
        "reusecontribout_",
        "localkfclose_",
        "reuseoutcome_",
        "reusecohort_",
        "maprepeat_",
    )
    structure_deltas = metric_deltas(segments, args.top_k, structure_prefixes)

    diff_vals = nums(rows, "ate_diff")
    lines = [
        "Segment structure summary",
        f"align={args.align} segment_size={args.segment_size} matched_frames={len(rows)} segments={len(segments)}",
        f"ours rmse={ours_summary['rmse']:.6f} mean={ours_summary['mean']:.6f} p90={ours_summary['p90']:.6f} scale={ours_summary['scale']:.9f}",
        f"orb  rmse={orb_summary['rmse']:.6f} mean={orb_summary['mean']:.6f} p90={orb_summary['p90']:.6f} scale={orb_summary['scale']:.9f}",
        f"diff mean={sum(diff_vals) / len(diff_vals):.6f} rms={math.sqrt(sum(v * v for v in diff_vals) / len(diff_vals)):.6f} worse_ratio={sum(1 for v in diff_vals if v > 0.0) / len(diff_vals):.3f}",
        "",
        "Worst segments by mean ours-minus-ORB ATE:",
        "segment,n,diag,ate_ours_rmse,ate_orb_rmse,ate_diff_mean,worse_ratio,mnInliers,local_mps,track_in_view_share,lmt_outlier,tracking_stereo_outlier,core_matches,expansion_matches,probation_matches,core_outlier,expansion_outlier,probation_outlier,covis_weak_vote_share,covis_shadow_connected,stable_shadow_mps,stable_shadow_trackable,stable_overlap,kf_inserts,kf_one_frame_share,kf_stable_support,kf_ref_overlap,kf_novel_close_cells,kf_highconf_novel_close_cells",
    ]
    for row in sorted(segments, key=lambda item: maybe_float(item.get("ate_diff_mean")) or -1e9, reverse=True)[: args.top_k]:
        lines.append(
            f"{row['segment_start']}-{row['segment_end']},"
            f"{row['matched_frames']},{row['diag_frames']},"
            f"{row.get('ate_ours_rmse', '')},{row.get('ate_orb_rmse', '')},"
            f"{row.get('ate_diff_mean', '')},{row.get('worse_ratio', '')},"
            f"{row.get('diag_mnMatchesInliers_avg', '')},"
            f"{row.get('kfstruct_local_mps_avg', '')},"
            f"{row.get('kfstruct_track_in_view_share', '')},"
            f"{row.get('joint_lmt_weighted_outlier_ratio', '')},"
            f"{row.get('joint_tracking_stereo_weighted_outlier_ratio', '')},"
            f"{row.get('search_core_pool_matches_avg', '')},"
            f"{row.get('search_expansion_pool_matches_avg', '')},"
            f"{row.get('search_probation_pool_matches_avg', '')},"
            f"{row.get('pool_core_weighted_outlier_ratio', '')},"
            f"{row.get('pool_expansion_weighted_outlier_ratio', '')},"
            f"{row.get('pool_probation_weighted_outlier_ratio', '')},"
            f"{row.get('covisvote_lmt_0_10_weak_vote_share', '')},"
            f"{row.get('covisvote_shadow_connected_avg', '')},"
            f"{row.get('stablevote_shadow_local_mps_avg', '')},"
            f"{row.get('stablevote_shadow_trackable_share_avg', '')},"
            f"{row.get('stablevote_selected_kf_overlap_share_avg', '')},"
            f"{row.get('kfdecision_insert_count', '')},"
            f"{row.get('kfinsert_one_frame_share', '')},"
            f"{row.get('kfinsert_novelty_stable_inlier_assoc_avg', '')},"
            f"{row.get('kfinsert_novelty_ref_overlap_share_avg', '')},"
            f"{row.get('kfinsert_novelty_new_close_cells_avg', '')},"
            f"{row.get('kfinsert_novelty_highconf_new_close_cells_avg', '')}"
        )

    if structure_deltas:
        lines.extend(
            [
                "",
                "Structure metric deltas on worst segments:",
                "field,top_count,rest_count,top_mean,rest_mean,diff,relative_diff",
            ]
        )
        for key, n_top, n_rest, top_mean, rest_mean, diff, rel in structure_deltas:
            lines.append(
                f"{key},{n_top},{n_rest},{top_mean:.6f},{rest_mean:.6f},{diff:.6f},{rel:.6f}"
            )
    if all_deltas:
        lines.extend(
            [
                "",
                "All metric deltas on worst segments:",
                "field,top_count,rest_count,top_mean,rest_mean,diff,relative_diff",
            ]
        )
        for key, n_top, n_rest, top_mean, rest_mean, diff, rel in all_deltas:
            lines.append(
                f"{key},{n_top},{n_rest},{top_mean:.6f},{rest_mean:.6f},{diff:.6f},{rel:.6f}"
            )

    summary_path.write_text("\n".join(lines) + "\n")

    print("\n".join(lines[: min(len(lines), 16)]))
    if structure_deltas:
        print("largest structure deltas:")
        for key, n_top, n_rest, top_mean, rest_mean, diff, rel in structure_deltas[:10]:
            print(
                f"  {key}: top={top_mean:.6f} rest={rest_mean:.6f} diff={diff:.6f} rel={rel:.3f} n={n_top}/{n_rest}"
            )
    print(f"wrote {out_path}")
    print(f"wrote {top_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
