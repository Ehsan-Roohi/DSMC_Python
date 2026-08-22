from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = 156

EXPECTED_STAGE155_SOURCE_HEAD = "da7cf0e31248e16551c220b2f0f7dcfc535112c6"
EXPECTED_STAGE155_RUN_ID = 32530742702
EXPECTED_STAGE155_JOB_ID = 96981721936
EXPECTED_STAGE155_ARTIFACT_ID = 9471028257
EXPECTED_STAGE155_ARTIFACT_SHA256 = "7853a7835a2db4afbfd0598b7a88a36803598253dfc53e163802c94c915c93a9"
EXPECTED_STAGE155_SUMMARY_SHA256 = "87fcccabcb168cd881239839ec4a4940e948dcafbbb9b239decbd006d4c5515c"
EXPECTED_STAGE155_PAYLOAD_SHA256 = "f099657d7796b366cefa0771d0892e574d9ab9ccc4f91264e2854a029bc19ea9"
EXPECTED_STAGE155_DECISION = (
    "stage155_opposition_coefficient_imposed_"
    "stage156_support_amplitude_balance_audit"
)

PROVENANCE_MATCH_MAX = 1.0e-12
MATERIAL_ENDPOINT_DECLINE_MIN = 0.25
LOG_MIDPOINT_RESIDUAL_MAX = 0.10

NONFINITE = "stage156_nonfinite_blocker"
STAGE155_RECORD_BLOCKER = "stage156_stage155_record_blocker"
PARENT_ROUTE_BLOCKER = "stage156_parent_route_blocker"
PROVENANCE_BLOCKER = "stage156_parent_provenance_blocker"
CURVED_TREND = (
    "stage156_material_monotone_curved_"
    "stage157_fixed_ratio_curvature_audit"
)
NEAR_LOG_LINEAR = (
    "stage156_material_monotone_near_loglinear_"
    "stage157_common_ratio_decay_audit"
)
NO_MATERIAL_TREND = "stage156_no_material_monotone_ratio_trend_stop"


def validate_stage156_design(
    *,
    grid=(64, 64),
    interior_grid=(56, 56),
    kn0=10.0,
    cold_hot_ratio=0.1,
    rule=(40, 96),
    radial_scale=2.0,
    limiter="minmod",
    boundary_slope="zero",
    source_relaxation=1.0,
    correction_floor=0.05,
    witness_node=9,
    pair_sectors=(5, 6),
    support_count=3,
    material_endpoint_decline_min=MATERIAL_ENDPOINT_DECLINE_MIN,
    log_midpoint_residual_max=LOG_MIDPOINT_RESIDUAL_MAX,
    amplitude_metrics_used_for_solver=False,
    solver_rerun=False,
    solver_endpoint_advanced=False,
    cross_knudsen_extension_permitted=False,
    benchmark_or_validation_claim_permitted=False,
    physical_parameter_retuning=False,
    collision_source_retuning=False,
    floor_retuning=False,
    wall_retuning=False,
    reconstruction_retuning=False,
    transport_retuning=False,
    limiter_retuning=False,
    normalization_retuning=False,
    source_relaxation_retuning=False,
    velocity_grid_retuning=False,
):
    expected = {
        "grid": (64, 64),
        "interior_grid": (56, 56),
        "kn0": 10.0,
        "cold_hot_ratio": 0.1,
        "rule": (40, 96),
        "radial_scale": 2.0,
        "limiter": "minmod",
        "boundary_slope": "zero",
        "source_relaxation": 1.0,
        "correction_floor": 0.05,
        "witness_node": 9,
        "pair_sectors": (5, 6),
        "support_count": 3,
        "material_endpoint_decline_min": MATERIAL_ENDPOINT_DECLINE_MIN,
        "log_midpoint_residual_max": LOG_MIDPOINT_RESIDUAL_MAX,
        "amplitude_metrics_used_for_solver": False,
        "solver_rerun": False,
        "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False,
        "benchmark_or_validation_claim_permitted": False,
        "physical_parameter_retuning": False,
        "collision_source_retuning": False,
        "floor_retuning": False,
        "wall_retuning": False,
        "reconstruction_retuning": False,
        "transport_retuning": False,
        "limiter_retuning": False,
        "normalization_retuning": False,
        "source_relaxation_retuning": False,
        "velocity_grid_retuning": False,
    }
    got = locals()
    for key, value in expected.items():
        if got[key] != value:
            raise ValueError(
                f"Stage 156 frozen-design violation: {key}={got[key]!r}, expected {value!r}"
            )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_stage155_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 155
        and record.get("source_head") == EXPECTED_STAGE155_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE155_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE155_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE155_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE155_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE155_SUMMARY_SHA256
        and record.get("support_sign_geometry_sha256") == EXPECTED_STAGE155_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE155_DECISION
        and record.get("workflow_status") == "completed"
        and record.get("workflow_conclusion") == "success"
    )


def support_amplitude_balance_metrics(
    *,
    support_depth: np.ndarray,
    dominant_raw_support: np.ndarray,
    parent_raw_support: np.ndarray,
    observed_cancellation_fraction: np.ndarray,
) -> dict:
    depth = np.asarray(support_depth, dtype=float)
    dominant = np.asarray(dominant_raw_support, dtype=float)
    parent = np.asarray(parent_raw_support, dtype=float)
    observed_cancel = np.asarray(observed_cancellation_fraction, dtype=float)
    if (
        depth.shape != (3,)
        or dominant.shape != (3,)
        or parent.shape != (3,)
        or observed_cancel.shape != (3,)
    ):
        raise ValueError("Stage 156 requires the exact three-node Stage-155 support arrays")
    if not (
        np.isfinite(depth).all()
        and np.isfinite(dominant).all()
        and np.isfinite(parent).all()
        and np.isfinite(observed_cancel).all()
    ):
        raise ValueError("Stage 156 requires finite parent arrays")

    dmag = np.abs(dominant)
    pmag = np.abs(parent)
    if np.any(dmag <= 0.0) or np.any(pmag <= 0.0):
        raise ValueError("Stage 156 requires nonzero inherited raw amplitudes")

    ratio = pmag / dmag
    ratio_differences = np.diff(ratio)
    monotone_decrease = bool(np.all(ratio_differences < 0.0))
    endpoint_decline_fraction = float(1.0 - ratio[-1] / ratio[0])
    ratio_range = float(np.max(ratio) - np.min(ratio))

    log_ratio = np.log(ratio)
    midpoint_log_expected = float(0.5 * (log_ratio[0] + log_ratio[2]))
    midpoint_log_residual = float(log_ratio[1] - midpoint_log_expected)
    midpoint_log_residual_abs = abs(midpoint_log_residual)
    log_second_difference = float(log_ratio[0] - 2.0 * log_ratio[1] + log_ratio[2])

    first_drop = float(ratio[0] - ratio[1])
    second_drop = float(ratio[1] - ratio[2])
    second_to_first_drop_ratio = (
        float(second_drop / first_drop) if first_drop > 0.0 else float("inf")
    )

    predicted_cancel = 2.0 * np.minimum(dmag, pmag) / (dmag + pmag)
    cancellation_reconstruction_error = float(
        np.max(np.abs(predicted_cancel - observed_cancel))
    )

    coeff = np.polyfit(depth, log_ratio, 1)
    fitted_log = np.polyval(coeff, depth)
    residual = log_ratio - fitted_log
    ss_res = float(np.dot(residual, residual))
    centered = log_ratio - np.mean(log_ratio)
    ss_tot = float(np.dot(centered, centered))
    log_linear_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 1.0

    return {
        "support_depth": depth.tolist(),
        "dominant_raw_support": dominant.tolist(),
        "parent_raw_support": parent.tolist(),
        "amplitude_ratio_parent_to_dominant": ratio.tolist(),
        "ratio_differences": ratio_differences.tolist(),
        "monotone_decrease": monotone_decrease,
        "endpoint_decline_fraction": endpoint_decline_fraction,
        "ratio_range": ratio_range,
        "log_ratio": log_ratio.tolist(),
        "midpoint_log_expected": midpoint_log_expected,
        "midpoint_log_residual": midpoint_log_residual,
        "midpoint_log_residual_abs": midpoint_log_residual_abs,
        "log_second_difference": log_second_difference,
        "second_to_first_drop_ratio": second_to_first_drop_ratio,
        "log_linear_slope_per_cell": float(coeff[0]),
        "log_linear_r2": log_linear_r2,
        "predicted_cancellation_fraction": predicted_cancel.tolist(),
        "observed_cancellation_fraction": observed_cancel.tolist(),
        "cancellation_fraction_reconstruction_error": cancellation_reconstruction_error,
        "maximum_identity_or_provenance_error": cancellation_reconstruction_error,
    }


def classify_support_amplitude_balance(
    *,
    metrics: dict,
    stage155_record_ok=True,
    parent_route_ok=True,
    finite=True,
) -> str:
    numeric = [
        metrics.get("endpoint_decline_fraction", np.nan),
        metrics.get("midpoint_log_residual_abs", np.nan),
        metrics.get("cancellation_fraction_reconstruction_error", np.nan),
    ]
    if not finite or not np.isfinite(numeric).all():
        return NONFINITE
    if not stage155_record_ok:
        return STAGE155_RECORD_BLOCKER
    if not parent_route_ok:
        return PARENT_ROUTE_BLOCKER
    if float(metrics["maximum_identity_or_provenance_error"]) > PROVENANCE_MATCH_MAX:
        return PROVENANCE_BLOCKER

    material_monotone = bool(
        metrics["monotone_decrease"]
        and float(metrics["endpoint_decline_fraction"]) >= MATERIAL_ENDPOINT_DECLINE_MIN
    )
    if not material_monotone:
        return NO_MATERIAL_TREND
    if float(metrics["midpoint_log_residual_abs"]) <= LOG_MIDPOINT_RESIDUAL_MAX:
        return NEAR_LOG_LINEAR
    return CURVED_TREND


def run_stage156(stage155_record: Path, output_dir: Path) -> dict:
    validate_stage156_design()
    record155 = _load_json(stage155_record)
    stage155_record_ok = _check_stage155_record(record155)
    parent_route_ok = bool(
        record155.get("decision") == EXPECTED_STAGE155_DECISION
        and record155.get("aggregate", {}).get("raw_same_sign_fraction") == 1.0
        and record155.get("aggregate", {}).get("coefficient_explained_opposition_fraction") == 1.0
        and record155.get("aggregate", {}).get(
            "cancellation_fraction_reconstruction_error", 1.0
        )
        <= PROVENANCE_MATCH_MAX
    )

    km = record155.get("key_metrics", {})
    metrics = support_amplitude_balance_metrics(
        support_depth=np.asarray(km.get("support_depth"), dtype=float),
        dominant_raw_support=np.asarray(km.get("dominant_raw_support"), dtype=float),
        parent_raw_support=np.asarray(km.get("parent_raw_support"), dtype=float),
        observed_cancellation_fraction=np.asarray(
            km.get("observed_cancellation_fraction"), dtype=float
        ),
    )

    finite = bool(
        np.isfinite(
            [
                metrics["endpoint_decline_fraction"],
                metrics["midpoint_log_residual_abs"],
                metrics["log_linear_slope_per_cell"],
                metrics["log_linear_r2"],
                metrics["cancellation_fraction_reconstruction_error"],
            ]
        ).all()
    )
    decision = classify_support_amplitude_balance(
        metrics=metrics,
        stage155_record_ok=stage155_record_ok,
        parent_route_ok=parent_route_ok,
        finite=finite,
    )

    if decision == CURVED_TREND:
        conclusion = (
            "The inherited parent/dominant raw-amplitude ratio decreases materially and monotonically "
            "across the three fixed support depths, but the middle node is not close enough to the "
            "preregistered log-linear midpoint relation. The depth dependence is therefore reproducible "
            "but curved at this sampled scale; a fixed ratio-curvature audit is justified before any "
            "stronger interpretation."
        )
    elif decision == NEAR_LOG_LINEAR:
        conclusion = (
            "The inherited parent/dominant raw-amplitude ratio decreases materially and monotonically, "
            "and the fixed midpoint is consistent with the preregistered near-log-linear guard. A common "
            "ratio-decay-scale audit is justified, but no solver mechanism is established."
        )
    else:
        conclusion = (
            "The inherited three-node amplitude balance does not satisfy the preregistered material "
            "monotone-trend conditions. No narrower amplitude-scale stage is justified from this artifact."
        )

    negative_guard = (
        "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. Stage 156 is an "
        "artifact-only amplitude-balance audit; inherited amplitude ratios, monotonicity, log-midpoint "
        "residuals, and fitted descriptive slopes are diagnostics, not solver parameters. No physical, "
        "collision/source, floor, wall, reconstruction, transport, limiter, normalization, "
        "source-relaxation, or velocity-quadrature parameter is retuned; no measured amplitude trend is "
        "fed back into the solver, no solver endpoint or cross-Knudsen extension is advanced, and no "
        "benchmark or validation claim is permitted."
    )

    summary = {
        "stage": STAGE,
        "decision": decision,
        "finite": finite,
        "configuration": {
            "grid": [64, 64],
            "interior_grid": [56, 56],
            "kn0": 10.0,
            "cold_hot_ratio": 0.1,
            "rule": [40, 96],
            "radial_scale": 2.0,
            "limiter": "minmod",
            "boundary_slope": "zero",
            "source_relaxation": 1.0,
            "correction_floor": 0.05,
            "witness_node": 9,
            "pair_sectors": [5, 6],
            "support_count": 3,
            "material_endpoint_decline_min": MATERIAL_ENDPOINT_DECLINE_MIN,
            "log_midpoint_residual_max": LOG_MIDPOINT_RESIDUAL_MAX,
            "amplitude_metrics_used_for_solver": False,
            "solver_rerun": False,
            "solver_endpoint_advanced": False,
            "cross_knudsen_extension_permitted": False,
            "benchmark_or_validation_claim_permitted": False,
            "physical_parameter_retuning": False,
            "collision_source_retuning": False,
            "floor_retuning": False,
            "wall_retuning": False,
            "reconstruction_retuning": False,
            "transport_retuning": False,
            "limiter_retuning": False,
            "normalization_retuning": False,
            "source_relaxation_retuning": False,
            "velocity_grid_retuning": False,
        },
        "parents": {
            "stage155_source_head": EXPECTED_STAGE155_SOURCE_HEAD,
            "stage155_run_id": EXPECTED_STAGE155_RUN_ID,
            "stage155_job_id": EXPECTED_STAGE155_JOB_ID,
            "stage155_artifact_id": EXPECTED_STAGE155_ARTIFACT_ID,
        },
        "aggregate": {
            "stage155_record_ok": stage155_record_ok,
            "parent_route_ok": parent_route_ok,
            "monotone_decrease": metrics["monotone_decrease"],
            "endpoint_decline_fraction": metrics["endpoint_decline_fraction"],
            "midpoint_log_residual_abs": metrics["midpoint_log_residual_abs"],
            "log_linear_r2": metrics["log_linear_r2"],
            "cancellation_fraction_reconstruction_error": metrics[
                "cancellation_fraction_reconstruction_error"
            ],
            "maximum_identity_or_provenance_error": metrics[
                "maximum_identity_or_provenance_error"
            ],
        },
        "metrics": metrics,
        "scientific_conclusion": conclusion,
        "negative_result_guard": negative_guard,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    np.savez(
        output_dir / "support_amplitude_balance.npz",
        support_depth=np.asarray(metrics["support_depth"], dtype=float),
        dominant_raw_support=np.asarray(metrics["dominant_raw_support"], dtype=float),
        parent_raw_support=np.asarray(metrics["parent_raw_support"], dtype=float),
        amplitude_ratio_parent_to_dominant=np.asarray(
            metrics["amplitude_ratio_parent_to_dominant"], dtype=float
        ),
        log_ratio=np.asarray(metrics["log_ratio"], dtype=float),
        predicted_cancellation_fraction=np.asarray(
            metrics["predicted_cancellation_fraction"], dtype=float
        ),
        observed_cancellation_fraction=np.asarray(
            metrics["observed_cancellation_fraction"], dtype=float
        ),
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage155-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_stage156(args.stage155_record, args.output_dir)


if __name__ == "__main__":
    main()
