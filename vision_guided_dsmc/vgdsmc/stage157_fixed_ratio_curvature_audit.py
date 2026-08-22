from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = 157

EXPECTED_STAGE156_SOURCE_HEAD = "18ecaf6ef83163c36abceba5b7e3388234c6b8cd"
EXPECTED_STAGE156_RUN_ID = 32556022646
EXPECTED_STAGE156_JOB_ID = 96990145397
EXPECTED_STAGE156_ARTIFACT_ID = 9472845908
EXPECTED_STAGE156_ARTIFACT_SHA256 = "e868f0c54740f93b2a416a2f048e084a3b19f3bf524236767e7890e346c534ed"
EXPECTED_STAGE156_SUMMARY_SHA256 = "1bb79510d483433efb786a4733b519564c68f7418d634c83d5e66275d4dd6af2"
EXPECTED_STAGE156_PAYLOAD_SHA256 = "403021664787b14aca813b5004139b481a73d1b8de5de8c56f104a7349d2cdac"
EXPECTED_STAGE156_DECISION = (
    "stage156_material_monotone_curved_stage157_fixed_ratio_curvature_audit"
)

PROVENANCE_MATCH_MAX = 1.0e-12
DEPTH_SPACING_MATCH_MAX = 1.0e-12
RATIO_RATE_ACCELERATION_MIN = 1.5
SINGLE_CHANNEL_SHARE_MIN = 0.80

NONFINITE = "stage157_nonfinite_blocker"
STAGE156_RECORD_BLOCKER = "stage157_stage156_record_blocker"
PARENT_ROUTE_BLOCKER = "stage157_parent_route_blocker"
PROVENANCE_BLOCKER = "stage157_parent_provenance_blocker"
SPACING_BLOCKER = "stage157_support_spacing_blocker"
NO_MATERIAL_CURVATURE = "stage157_no_material_ratio_curvature_stop"
PARENT_DOMINANT = "stage157_parent_channel_curvature_stage158_parent_support_audit"
DOMINANT_DOMINANT = "stage157_dominant_channel_curvature_stage158_dominant_support_audit"
MIXED_CURVATURE = "stage157_material_mixed_channel_curvature_stop"


def validate_stage157_design(
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
    ratio_rate_acceleration_min=RATIO_RATE_ACCELERATION_MIN,
    single_channel_share_min=SINGLE_CHANNEL_SHARE_MIN,
    curvature_metrics_used_for_solver=False,
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
        "ratio_rate_acceleration_min": RATIO_RATE_ACCELERATION_MIN,
        "single_channel_share_min": SINGLE_CHANNEL_SHARE_MIN,
        "curvature_metrics_used_for_solver": False,
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
                f"Stage 157 frozen-design violation: {key}={got[key]!r}, expected {value!r}"
            )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_stage156_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 156
        and record.get("source_head") == EXPECTED_STAGE156_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE156_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE156_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE156_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE156_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE156_SUMMARY_SHA256
        and record.get("support_amplitude_balance_sha256") == EXPECTED_STAGE156_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE156_DECISION
        and record.get("workflow_status") == "completed"
        and record.get("workflow_conclusion") == "success"
    )


def fixed_ratio_curvature_metrics(
    *,
    support_depth: np.ndarray,
    dominant_raw_support: np.ndarray,
    parent_raw_support: np.ndarray,
    inherited_ratio: np.ndarray,
) -> dict:
    depth = np.asarray(support_depth, dtype=float)
    dominant = np.asarray(dominant_raw_support, dtype=float)
    parent = np.asarray(parent_raw_support, dtype=float)
    inherited = np.asarray(inherited_ratio, dtype=float)
    if depth.shape != (3,) or dominant.shape != (3,) or parent.shape != (3,) or inherited.shape != (3,):
        raise ValueError("Stage 157 requires the exact three fixed Stage-156 support nodes")
    if not np.isfinite(np.concatenate([depth, dominant, parent, inherited])).all():
        raise ValueError("Stage 157 requires finite inherited support arrays")

    dmag = np.abs(dominant)
    pmag = np.abs(parent)
    if np.any(dmag <= 0.0) or np.any(pmag <= 0.0):
        raise ValueError("Stage 157 requires nonzero inherited raw amplitudes")

    spacing = np.diff(depth)
    spacing_error = float(np.max(np.abs(spacing - spacing[0])))
    reconstructed_ratio = pmag / dmag
    provenance_error = float(np.max(np.abs(reconstructed_ratio - inherited)))

    log_dominant = np.log(dmag)
    log_parent = np.log(pmag)
    log_ratio = np.log(reconstructed_ratio)
    dominant_decay_rate = -np.diff(log_dominant) / spacing
    parent_decay_rate = -np.diff(log_parent) / spacing
    ratio_decay_rate = -np.diff(log_ratio) / spacing

    dominant_rate_change = float(dominant_decay_rate[1] - dominant_decay_rate[0])
    parent_rate_change = float(parent_decay_rate[1] - parent_decay_rate[0])
    ratio_rate_change = float(ratio_decay_rate[1] - ratio_decay_rate[0])
    rate_change_identity_error = float(
        abs(ratio_rate_change - (parent_rate_change - dominant_rate_change))
    )

    if ratio_decay_rate[0] > 0.0:
        ratio_rate_acceleration_factor = float(ratio_decay_rate[1] / ratio_decay_rate[0])
    else:
        ratio_rate_acceleration_factor = float("nan")

    parent_curvature_contribution = parent_rate_change
    dominant_curvature_contribution = -dominant_rate_change
    contribution_l1 = abs(parent_curvature_contribution) + abs(dominant_curvature_contribution)
    if contribution_l1 > 0.0:
        parent_curvature_share = float(abs(parent_curvature_contribution) / contribution_l1)
        dominant_curvature_share = float(abs(dominant_curvature_contribution) / contribution_l1)
    else:
        parent_curvature_share = 0.0
        dominant_curvature_share = 0.0

    log_second_dominant = float(log_dominant[0] - 2.0 * log_dominant[1] + log_dominant[2])
    log_second_parent = float(log_parent[0] - 2.0 * log_parent[1] + log_parent[2])
    log_second_ratio = float(log_ratio[0] - 2.0 * log_ratio[1] + log_ratio[2])
    second_difference_identity_error = float(
        abs(log_second_ratio - (log_second_parent - log_second_dominant))
    )

    max_identity_or_provenance_error = max(
        provenance_error,
        rate_change_identity_error,
        second_difference_identity_error,
    )

    return {
        "support_depth": depth.tolist(),
        "support_spacing": spacing.tolist(),
        "support_spacing_error": spacing_error,
        "dominant_raw_support": dominant.tolist(),
        "parent_raw_support": parent.tolist(),
        "amplitude_ratio_parent_to_dominant": reconstructed_ratio.tolist(),
        "ratio_provenance_error": provenance_error,
        "dominant_decay_rate_per_cell": dominant_decay_rate.tolist(),
        "parent_decay_rate_per_cell": parent_decay_rate.tolist(),
        "ratio_decay_rate_per_cell": ratio_decay_rate.tolist(),
        "dominant_rate_change_per_cell": dominant_rate_change,
        "parent_rate_change_per_cell": parent_rate_change,
        "ratio_rate_change_per_cell": ratio_rate_change,
        "ratio_rate_acceleration_factor": ratio_rate_acceleration_factor,
        "parent_curvature_contribution": parent_curvature_contribution,
        "dominant_curvature_contribution": dominant_curvature_contribution,
        "parent_curvature_share": parent_curvature_share,
        "dominant_curvature_share": dominant_curvature_share,
        "log_second_difference_dominant": log_second_dominant,
        "log_second_difference_parent": log_second_parent,
        "log_second_difference_ratio": log_second_ratio,
        "rate_change_identity_error": rate_change_identity_error,
        "second_difference_identity_error": second_difference_identity_error,
        "maximum_identity_or_provenance_error": max_identity_or_provenance_error,
    }


def classify_fixed_ratio_curvature(
    *,
    metrics: dict,
    stage156_record_ok=True,
    parent_route_ok=True,
    finite=True,
) -> str:
    numeric = [
        metrics.get("ratio_rate_acceleration_factor", np.nan),
        metrics.get("parent_curvature_share", np.nan),
        metrics.get("dominant_curvature_share", np.nan),
        metrics.get("maximum_identity_or_provenance_error", np.nan),
    ]
    if not finite or not np.isfinite(numeric).all():
        return NONFINITE
    if not stage156_record_ok:
        return STAGE156_RECORD_BLOCKER
    if not parent_route_ok:
        return PARENT_ROUTE_BLOCKER
    if float(metrics["support_spacing_error"]) > DEPTH_SPACING_MATCH_MAX:
        return SPACING_BLOCKER
    if float(metrics["maximum_identity_or_provenance_error"]) > PROVENANCE_MATCH_MAX:
        return PROVENANCE_BLOCKER

    material_curvature = bool(
        metrics["ratio_decay_rate_per_cell"][0] > 0.0
        and metrics["ratio_decay_rate_per_cell"][1] > metrics["ratio_decay_rate_per_cell"][0]
        and float(metrics["ratio_rate_acceleration_factor"]) >= RATIO_RATE_ACCELERATION_MIN
    )
    if not material_curvature:
        return NO_MATERIAL_CURVATURE
    if float(metrics["parent_curvature_share"]) >= SINGLE_CHANNEL_SHARE_MIN:
        return PARENT_DOMINANT
    if float(metrics["dominant_curvature_share"]) >= SINGLE_CHANNEL_SHARE_MIN:
        return DOMINANT_DOMINANT
    return MIXED_CURVATURE


def run_stage157(stage156_record: Path, output_dir: Path) -> dict:
    validate_stage157_design()
    record156 = _load_json(stage156_record)
    stage156_record_ok = _check_stage156_record(record156)
    parent_route_ok = bool(
        record156.get("decision") == EXPECTED_STAGE156_DECISION
        and record156.get("aggregate", {}).get("monotone_decrease") is True
        and float(record156.get("aggregate", {}).get("endpoint_decline_fraction", 0.0)) >= 0.25
        and float(record156.get("aggregate", {}).get("midpoint_log_residual_abs", 0.0)) > 0.10
        and float(record156.get("aggregate", {}).get("maximum_identity_or_provenance_error", 1.0))
        <= PROVENANCE_MATCH_MAX
    )

    km = record156.get("key_metrics", {})
    metrics = fixed_ratio_curvature_metrics(
        support_depth=np.asarray(km.get("support_depth"), dtype=float),
        dominant_raw_support=np.asarray(km.get("dominant_raw_support"), dtype=float),
        parent_raw_support=np.asarray(km.get("parent_raw_support"), dtype=float),
        inherited_ratio=np.asarray(km.get("amplitude_ratio_parent_to_dominant"), dtype=float),
    )

    finite = bool(np.isfinite([
        metrics["ratio_rate_acceleration_factor"],
        metrics["parent_curvature_share"],
        metrics["dominant_curvature_share"],
        metrics["maximum_identity_or_provenance_error"],
    ]).all())
    decision = classify_fixed_ratio_curvature(
        metrics=metrics,
        stage156_record_ok=stage156_record_ok,
        parent_route_ok=parent_route_ok,
        finite=finite,
    )

    if decision == MIXED_CURVATURE:
        conclusion = (
            "The parent/dominant amplitude-ratio attenuation rate accelerates materially between the two "
            "fixed depth intervals, but neither raw channel reaches the conservative 80% single-channel "
            "curvature-share guard. The sampled curvature is therefore a mixed-channel balance: parent "
            "decay acceleration is the larger contribution, while dominant-channel decay slowing also "
            "contributes. With only the same three fixed support nodes, no narrower single-channel stage is "
            "scientifically justified."
        )
    elif decision == PARENT_DOMINANT:
        conclusion = (
            "The fixed ratio curvature is material and at least 80% attributable to acceleration of the "
            "parent-channel log-decay rate. A parent-support audit is justified without changing the solver."
        )
    elif decision == DOMINANT_DOMINANT:
        conclusion = (
            "The fixed ratio curvature is material and at least 80% attributable to the dominant-channel "
            "log-decay-rate change. A dominant-support audit is justified without changing the solver."
        )
    else:
        conclusion = (
            "The fixed three-node ratio data do not satisfy the preregistered material-curvature route. "
            "No narrower curvature stage is justified from these artifacts."
        )

    negative_guard = (
        "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. Stage 157 is an "
        "artifact-only ratio-curvature audit; interval decay rates, discrete curvature, and channel-share "
        "attributions are diagnostics, not solver parameters. No physical, collision/source, floor, wall, "
        "reconstruction, transport, limiter, normalization, source-relaxation, or velocity-quadrature "
        "parameter is retuned; no measured rate or curvature is fed back into the solver, no solver endpoint "
        "or cross-Knudsen extension is advanced, and no benchmark or validation claim is permitted."
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
            "ratio_rate_acceleration_min": RATIO_RATE_ACCELERATION_MIN,
            "single_channel_share_min": SINGLE_CHANNEL_SHARE_MIN,
            "curvature_metrics_used_for_solver": False,
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
            "stage156_source_head": EXPECTED_STAGE156_SOURCE_HEAD,
            "stage156_run_id": EXPECTED_STAGE156_RUN_ID,
            "stage156_job_id": EXPECTED_STAGE156_JOB_ID,
            "stage156_artifact_id": EXPECTED_STAGE156_ARTIFACT_ID,
        },
        "aggregate": {
            "stage156_record_ok": stage156_record_ok,
            "parent_route_ok": parent_route_ok,
            "ratio_rate_acceleration_factor": metrics["ratio_rate_acceleration_factor"],
            "parent_curvature_share": metrics["parent_curvature_share"],
            "dominant_curvature_share": metrics["dominant_curvature_share"],
            "support_spacing_error": metrics["support_spacing_error"],
            "maximum_identity_or_provenance_error": metrics["maximum_identity_or_provenance_error"],
        },
        "metrics": metrics,
        "scientific_conclusion": conclusion,
        "negative_result_guard": negative_guard,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    np.savez(
        output_dir / "fixed_ratio_curvature.npz",
        support_depth=np.asarray(metrics["support_depth"], dtype=float),
        amplitude_ratio_parent_to_dominant=np.asarray(metrics["amplitude_ratio_parent_to_dominant"], dtype=float),
        dominant_decay_rate_per_cell=np.asarray(metrics["dominant_decay_rate_per_cell"], dtype=float),
        parent_decay_rate_per_cell=np.asarray(metrics["parent_decay_rate_per_cell"], dtype=float),
        ratio_decay_rate_per_cell=np.asarray(metrics["ratio_decay_rate_per_cell"], dtype=float),
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage156-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_stage157(args.stage156_record, args.output_dir)


if __name__ == "__main__":
    main()
