"""Auditable repair of the JCP2 stationarity implementation.

The inherited M1R diagnostic normalized the y/L=0.8 q_y profile by its
positive maximum.  At the locked JCP2 S2 condition the entire mean profile is
negative, so the normalization produced NaNs and every seed received an
infinite stationarity statistic.  The inherited per-metric |z| <= 2 rule also
treated six simultaneous diagnostics as six independent 5% gates.

This module changes neither trajectories nor measured fields.  It recomputes
the same two-half block diagnostics with a sign-invariant heat-flux scale and
a fixed Bonferroni family-wise 5% threshold across the same six metrics.  The
legacy report is retained in every amended summary for complete provenance.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping

import numpy as np

from . import jcp_phase1_cavity as jcp2


METHOD = "JCP2_signed_qy_Bonferroni_familywise_stationarity_v1"
FAMILYWISE_ALPHA = 0.05
QY_METRICS = ("qy_profile_min_normalized", "qy_profile_max_normalized")
REQUIRED = {"evaluation": 8, "reference": 20}


def familywise_z_limit(metric_count: int, alpha: float = FAMILYWISE_ALPHA) -> float:
    """Two-sided Bonferroni normal threshold for a fixed metric family."""

    metric_count = int(metric_count)
    alpha = float(alpha)
    if metric_count < 1 or not 0.0 < alpha < 1.0:
        raise ValueError("invalid family-wise stationarity contract")
    return float(NormalDist().inv_cdf(1.0 - alpha / (2.0 * metric_count)))


def _profile_at_y(field: np.ndarray, y_over_l: float) -> np.ndarray:
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError("heat-flux profile requires a two-dimensional field")
    centers = (np.arange(field.shape[0], dtype=np.float64) + 0.5) / field.shape[0]
    return np.asarray(
        [np.interp(y_over_l, centers, field[:, column]) for column in range(field.shape[1])],
        dtype=np.float64,
    )


def _two_half_report(
    values: np.ndarray,
    *,
    minimum_finite_per_half: int,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    half_count = len(values) // 2
    finite = np.isfinite(values)
    first_finite = int(np.count_nonzero(finite[:half_count]))
    second_finite = int(np.count_nonzero(finite[half_count:]))
    report: dict[str, Any] = {
        "block_count": int(len(values)),
        "first_half_finite": first_finite,
        "second_half_finite": second_finite,
    }
    if (
        first_finite < int(minimum_finite_per_half)
        or second_finite < int(minimum_finite_per_half)
    ):
        report["max_abs_drift_z_score"] = float("inf")
        return report

    first = values[:half_count][finite[:half_count]]
    second = values[half_count:][finite[half_count:]]
    first_mean = float(np.mean(first))
    second_mean = float(np.mean(second))
    drift = second_mean - first_mean
    standard_error = float(
        np.sqrt(np.var(first, ddof=1) / len(first) + np.var(second, ddof=1) / len(second))
    )
    if standard_error > 0.0:
        z_score = drift / standard_error
    elif drift == 0.0:
        z_score = 0.0
    else:
        z_score = float(np.copysign(np.inf, drift))
    midpoint = 0.5 * (first_mean + second_mean)
    relative_drift = (
        abs(drift) / abs(midpoint)
        if midpoint != 0.0
        else (0.0 if drift == 0.0 else float("inf"))
    )
    report.update(
        {
            "first_half_mean": first_mean,
            "second_half_mean": second_mean,
            "drift": drift,
            "drift_standard_error": standard_error,
            "drift_z_score": float(z_score),
            "relative_drift": float(relative_drift),
            "max_abs_drift_z_score": float(abs(z_score)),
        }
    )
    return report


def _numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def corrected_stationarity(directory: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the JCP2 stationarity report without opening reference data."""

    directory = Path(directory)
    if summary.get("stationarity", {}).get("method") == METHOD:
        return deepcopy(dict(summary["stationarity"]))
    legacy = deepcopy(
        dict(summary.get("legacy_stationarity_report", summary.get("stationarity", {})))
    )
    tracked = deepcopy(dict(legacy.get("tracked", {})))
    if set(QY_METRICS) - set(tracked):
        raise ValueError(f"legacy stationarity metrics are incomplete: {directory}")
    metric_count = len(tracked)
    if metric_count != 6:
        raise ValueError(f"JCP2 expected six stationarity metrics: {directory}")
    minimum_finite = int(legacy.get("minimum_finite_blocks_per_half", 3))

    with np.load(directory / "fields.npz", allow_pickle=False) as archive:
        overall_qy = np.asarray(archive["qy"], dtype=np.float64)
    with np.load(directory / "block_fields.npz", allow_pickle=False) as archive:
        block_qy = np.asarray(archive["qy"], dtype=np.float64)
    if block_qy.ndim != 3 or len(block_qy) != int(jcp2.BLOCK_COUNT):
        raise ValueError(f"JCP2 qy block layout is invalid: {directory}")

    overall_profile = _profile_at_y(overall_qy, 0.8)
    legacy_positive_scale = float(np.max(overall_profile))
    signed_scale = float(np.max(np.abs(overall_profile)))
    if not np.isfinite(signed_scale) or signed_scale <= 0.0:
        raise ValueError(f"JCP2 qy profile has no finite nonzero scale: {directory}")
    normalized = np.asarray(
        [_profile_at_y(field, 0.8) / signed_scale for field in block_qy],
        dtype=np.float64,
    )
    tracked["qy_profile_min_normalized"] = _two_half_report(
        np.min(normalized, axis=1),
        minimum_finite_per_half=minimum_finite,
    )
    tracked["qy_profile_max_normalized"] = _two_half_report(
        np.max(normalized, axis=1),
        minimum_finite_per_half=minimum_finite,
    )

    z_limit = familywise_z_limit(metric_count)
    checks = {
        name: bool(_numeric(report.get("max_abs_drift_z_score")) <= z_limit)
        for name, report in tracked.items()
    }
    return {
        "method": METHOD,
        "post_trajectory_statistical_implementation_repair": True,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "multiple_testing_correction": "two-sided Bonferroni",
        "metric_count": metric_count,
        "corrected_max_abs_drift_z_score": z_limit,
        "legacy_per_metric_z_limit": legacy.get("z_limit", 2.0),
        "minimum_finite_blocks_per_half": minimum_finite,
        "heat_flux_profile_y_over_l": 0.8,
        "heat_flux_normalization": "max(abs(overall_qy_profile))",
        "legacy_positive_qy_scale": legacy_positive_scale,
        "corrected_signed_qy_scale": signed_scale,
        "correction_reason": (
            "the inherited positive-maximum normalization is undefined for the "
            "negative-signed S2 qy profile; six simultaneous diagnostics also "
            "require a family-wise rather than per-metric 5% gate"
        ),
        "tracked": tracked,
        "checks": checks,
        "all_passed": bool(all(checks.values())),
    }


def _candidate_record(
    directory: Path,
    summary: Mapping[str, Any],
    corrected: Mapping[str, Any],
    *,
    order: int,
    primary_count: int,
) -> dict[str, Any]:
    mechanical = dict(summary.get("mechanical_checks", {}))
    mechanical["stationarity_pass"] = bool(corrected.get("all_passed"))
    failed = sorted(name for name, passed in mechanical.items() if not bool(passed))
    stationarity_failed = sorted(
        name for name, passed in corrected.get("checks", {}).items() if not bool(passed)
    )
    return {
        "order": int(order),
        "seed": int(summary["seed"]),
        "role": "primary" if order < primary_count else "spare",
        "directory": str(directory),
        "accepted": not failed,
        "failed_mechanical_checks": failed,
        "failed_stationarity_metrics": stationarity_failed,
        "corrected_stationarity": dict(corrected),
    }


def audit_stationarity(
    run_root: Path,
    group: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Audit one blinded JCP2 group and return its locked selection outcome."""

    run_root = Path(run_root)
    if group not in REQUIRED:
        raise ValueError("JCP2 stationarity group must be evaluation or reference")
    seed_bank = jcp2.load_seed_bank()
    primary_count = len(seed_bank[f"{group}_primary"])
    required = REQUIRED[group]
    records: list[dict[str, Any]] = []
    selected: list[int] = []
    for order, seed in enumerate(jcp2.group_seeds(group)):
        directory = run_root / group / f"seed_{seed}"
        summary = jcp2._verify_artifacts(directory)
        corrected = corrected_stationarity(directory, summary)
        record = _candidate_record(
            directory,
            summary,
            corrected,
            order=order,
            primary_count=primary_count,
        )
        records.append(record)
        if record["accepted"] and len(selected) < required:
            selected.append(int(seed))
    audit = {
        "stage": jcp2.STAGE,
        "method": METHOD,
        "group": group,
        "trajectory_data_changed": False,
        "reference_interface_opened": group == "reference",
        "required": required,
        "passing_count": int(sum(record["accepted"] for record in records)),
        "selected_seeds": selected,
        "selection_complete": len(selected) == required,
        "records": records,
    }
    return audit, records


def _apply_record(record: Mapping[str, Any]) -> None:
    directory = Path(str(record["directory"]))
    summary_path = directory / "summary.json"
    manifest_path = directory / "artifact_manifest.json"
    summary = jcp2._json(summary_path)
    corrected = dict(record["corrected_stationarity"])
    if summary.get("stationarity", {}).get("method") != METHOD:
        summary["legacy_stationarity_report"] = deepcopy(summary["stationarity"])
    summary["stationarity"] = corrected
    summary["stationarity_implementation_repair"] = {
        "method": METHOD,
        "trajectory_data_changed": False,
        "fields_or_blocks_changed": False,
        "legacy_report_retained": True,
    }
    mechanical = dict(summary.get("mechanical_checks", {}))
    mechanical["stationarity_pass"] = bool(corrected["all_passed"])
    summary["mechanical_checks"] = mechanical
    summary["decision"] = (
        "accept_JCP2_seed_for_preregistered_QC_selection"
        if mechanical and all(bool(value) for value in mechanical.values())
        else "hold_JCP2_seed_for_preregistered_spare_replacement"
    )
    jcp2._atomic_write_json(summary_path, summary)
    manifest = jcp2._json(manifest_path)
    manifest["files"]["summary.json"] = {
        "sha256": jcp2._sha256(summary_path),
        "size_bytes": summary_path.stat().st_size,
    }
    jcp2._atomic_write_json(manifest_path, manifest)
    jcp2._verify_artifacts(directory)


def apply_stationarity_repair(
    run_root: Path,
    group: str,
    audit_path: Path,
) -> dict[str, Any]:
    audit, records = audit_stationarity(run_root, group)
    audit_path = Path(audit_path)
    jcp2._atomic_write_json(audit_path, audit)
    if not audit["selection_complete"]:
        raise ValueError(
            "JCP2 corrected stationarity audit is not ready: "
            + json.dumps(
                {
                    "group": group,
                    "passing_count": audit["passing_count"],
                    "required": audit["required"],
                },
                sort_keys=True,
            )
        )
    for record in records:
        _apply_record(record)
    verified, _ = audit_stationarity(run_root, group)
    verified["summaries_and_manifests_updated"] = True
    jcp2._atomic_write_json(audit_path, verified)
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--group", choices=("evaluation", "reference"), required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply:
        result = apply_stationarity_repair(args.run_root, args.group, args.audit)
    else:
        result, _ = audit_stationarity(args.run_root, args.group)
        jcp2._atomic_write_json(args.audit, result)
    print(
        json.dumps(
            jcp2._strict_json_ready(result),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
