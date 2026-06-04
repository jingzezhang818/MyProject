#!/usr/bin/env python3
"""Summarize XFeat measurement calibration diagnostics from a SLAM run log.

The log is intentionally aggregate-only: it contains per-frame bucket summaries,
not per-observation samples. Recommendations are therefore conservative and are
based on bucket purity/coverage rather than fitting exact thresholds.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple


KV_RE = re.compile(r"(\w+)=([^\s]+)")


def parse_kv(line: str) -> Dict[str, str]:
    return {match.group(1): match.group(2) for match in KV_RE.finditer(line)}


def as_int(fields: Dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(fields.get(key, default)))
    except (TypeError, ValueError):
        return default


def as_float(fields: Dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(fields.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass
class CalibAgg:
    total: int = 0
    inlier: int = 0
    outlier: int = 0
    residual_px_sum: float = 0.0
    chi2_sum: float = 0.0
    depth_sum: float = 0.0
    residual_px_p90_weighted: float = 0.0
    chi2_p90_weighted: float = 0.0

    def add_measurement_line(self, fields: Dict[str, str]) -> None:
        total = as_int(fields, "total")
        inlier = as_int(fields, "inlier")
        outlier = as_int(fields, "outlier")
        self.total += total
        self.inlier += inlier
        self.outlier += outlier
        self.residual_px_sum += as_float(fields, "residual_px_avg", 0.0) * total
        self.chi2_sum += as_float(fields, "chi2_avg", 0.0) * total
        self.depth_sum += as_float(fields, "depth_avg", 0.0) * total
        self.residual_px_p90_weighted += as_float(fields, "residual_px_p90", 0.0) * total
        self.chi2_p90_weighted += as_float(fields, "chi2_p90", 0.0) * total

    @property
    def outlier_ratio(self) -> float:
        return self.outlier / self.total if self.total else 0.0

    @property
    def residual_px_avg(self) -> float:
        return self.residual_px_sum / self.total if self.total else 0.0

    @property
    def residual_px_p90(self) -> float:
        return self.residual_px_p90_weighted / self.total if self.total else 0.0

    @property
    def chi2_avg(self) -> float:
        return self.chi2_sum / self.total if self.total else 0.0

    @property
    def chi2_p90(self) -> float:
        return self.chi2_p90_weighted / self.total if self.total else 0.0

    @property
    def depth_avg(self) -> float:
        return self.depth_sum / self.total if self.total else 0.0


@dataclass
class PoseOptAgg:
    edges: int = 0
    inlier_edges: int = 0
    outlier_edges: int = 0
    residual_px_sum: float = 0.0
    chi2_sum: float = 0.0
    descriptor_dist_sum: float = 0.0
    descriptor_dist_p90_weighted: float = 0.0
    ratio_sum: float = 0.0
    ratio_p90_weighted: float = 0.0
    proj_norm_sum: float = 0.0
    proj_norm_p90_weighted: float = 0.0

    def add_poseopt_line(self, fields: Dict[str, str]) -> None:
        edges = as_int(fields, "edges")
        self.edges += edges
        self.inlier_edges += as_int(fields, "inlier_edges")
        self.outlier_edges += as_int(fields, "outlier_edges")
        self.residual_px_sum += as_float(fields, "residual_px_avg", 0.0) * edges
        self.chi2_sum += as_float(fields, "chi2_avg", 0.0) * edges
        self.descriptor_dist_sum += as_float(fields, "descriptor_dist_avg", 0.0) * edges
        self.descriptor_dist_p90_weighted += as_float(fields, "descriptor_dist_p90", 0.0) * edges
        self.ratio_sum += as_float(fields, "ratio_avg", 0.0) * edges
        self.ratio_p90_weighted += as_float(fields, "ratio_p90", 0.0) * edges
        self.proj_norm_sum += as_float(fields, "proj_norm_avg", 0.0) * edges
        self.proj_norm_p90_weighted += as_float(fields, "proj_norm_p90", 0.0) * edges

    @property
    def outlier_ratio(self) -> float:
        return self.outlier_edges / self.edges if self.edges else 0.0

    def avg(self, value: float) -> float:
        return value / self.edges if self.edges else 0.0


def print_measurement_summary(measurement: Dict[Tuple[str, str], CalibAgg], min_total: int) -> None:
    print("\n[XFeatMeasurementCalib] bucket summary")
    for (dimension, bin_name), stats in sorted(
        measurement.items(), key=lambda item: (item[0][0], -item[1].total, item[0][1])
    ):
        if stats.total < min_total:
            continue
        print(
            f"dimension={dimension:24s} bin={bin_name:28s} total={stats.total:6d} "
            f"outlier_ratio={stats.outlier_ratio:6.3f} residual_px_avg={stats.residual_px_avg:6.3f} "
            f"residual_px_p90={stats.residual_px_p90:6.3f} chi2_p90={stats.chi2_p90:6.3f} "
            f"depth_avg={stats.depth_avg:7.2f}"
        )


def print_poseopt_summary(poseopt: Dict[Tuple[str, str, str], PoseOptAgg], min_total: int) -> None:
    print("\n[PoseOpt][ResidualCalib] source/depth summary")
    for (match_source, mp_source, depth_bin), stats in sorted(
        poseopt.items(), key=lambda item: (-item[1].edges, item[0])
    ):
        if stats.edges < min_total:
            continue
        print(
            f"match_source={match_source:18s} mp_source={mp_source:32s} depth_bin={depth_bin:8s} "
            f"edges={stats.edges:6d} outlier_ratio={stats.outlier_ratio:6.3f} "
            f"residual_px_avg={stats.avg(stats.residual_px_sum):6.3f} "
            f"chi2_avg={stats.avg(stats.chi2_sum):6.3f} "
            f"best_dist_avg={stats.avg(stats.descriptor_dist_sum):6.3f} "
            f"best_dist_p90={stats.avg(stats.descriptor_dist_p90_weighted):6.3f} "
            f"ratio_avg={stats.avg(stats.ratio_sum):6.3f} "
            f"ratio_p90={stats.avg(stats.ratio_p90_weighted):6.3f} "
            f"proj_norm_avg={stats.avg(stats.proj_norm_sum):6.3f} "
            f"proj_norm_p90={stats.avg(stats.proj_norm_p90_weighted):6.3f}"
        )


def print_candidate_rules(measurement: Dict[Tuple[str, str], CalibAgg], min_total: int) -> None:
    print("\nCandidate safe-filter buckets")
    found = False
    for (dimension, bin_name), stats in sorted(
        measurement.items(), key=lambda item: (-item[1].outlier_ratio, -item[1].total)
    ):
        if stats.total < min_total:
            continue
        inlier_share = stats.inlier / stats.total if stats.total else 0.0
        if stats.outlier_ratio >= 0.50 and inlier_share <= 0.10:
            found = True
            print(
                f"- {dimension}={bin_name}: total={stats.total}, "
                f"outlier_ratio={stats.outlier_ratio:.3f}, inlier_share={inlier_share:.3f}, "
                f"residual_px_p90={stats.residual_px_p90:.3f}, chi2_p90={stats.chi2_p90:.3f}"
            )
    if not found:
        print("- No bucket met the conservative rule: total>=min_total, outlier_ratio>=0.50, inlier_share<=0.10.")


def parse_log(lines: Iterable[str]) -> tuple[Dict[Tuple[str, str], CalibAgg], Dict[Tuple[str, str, str], PoseOptAgg]]:
    measurement: Dict[Tuple[str, str], CalibAgg] = defaultdict(CalibAgg)
    poseopt: Dict[Tuple[str, str, str], PoseOptAgg] = defaultdict(PoseOptAgg)

    for line in lines:
        if "[TrackLocalMap][XFeatMeasurementCalib]" in line:
            fields = parse_kv(line)
            dimension = fields.get("dimension", "unknown")
            bin_name = fields.get("bin", "unknown")
            measurement[(dimension, bin_name)].add_measurement_line(fields)
        elif "[PoseOpt][ResidualCalib]" in line and "descriptor=XFeat" in line:
            fields = parse_kv(line)
            key = (
                fields.get("match_source", "unknown"),
                fields.get("mp_source", "unknown"),
                fields.get("depth_bin", "unknown"),
            )
            poseopt[key].add_poseopt_line(fields)

    return measurement, poseopt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Run log containing XFeat calibration diagnostics")
    parser.add_argument("--min-total", type=int, default=20, help="Minimum aggregate count to print a bucket")
    args = parser.parse_args()

    if not args.log.exists():
        raise SystemExit(f"log not found: {args.log}")

    with args.log.open("r", errors="replace") as handle:
        measurement, poseopt = parse_log(handle)

    if not measurement and not poseopt:
        print("No XFeat calibration diagnostics found.")
        return 0

    print_measurement_summary(measurement, args.min_total)
    print_poseopt_summary(poseopt, args.min_total)
    print_candidate_rules(measurement, args.min_total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
