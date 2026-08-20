from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = 149
EXPECTED_STAGE148_SOURCE_HEAD = "41e67a25f3e63082d83afb0a7f8c6c5cab55ef4b"
EXPECTED_STAGE148_RUN_ID = 32383959503
EXPECTED_STAGE148_JOB_ID = 96473720313
EXPECTED_STAGE148_ARTIFACT_ID = 9419539134
EXPECTED_STAGE148_ARTIFACT_SHA256 = "642143f71355e1703ed6dda94ba873831be3bf0049f4ab7cfcf52a1a5aa3f909"
EXPECTED_STAGE148_SUMMARY_SHA256 = "8b4133b3f344dbb58cb65fb775ee751bb21a3e4b5c5028439e71a7cb1e8a433c"
EXPECTED_STAGE148_PAYLOAD_SHA256 = "da6827b2ad3aa7b1292be4ada264b3a333a822c8198cbbbe6903578a9f835da5"
EXPECTED_STAGE148_DECISION = "stage148_sample_scale_alternation_dominant_stage149_channel_scale_separation_audit"

IDENTITY_CLOSURE_MAX = 1.0e-12
SAMPLE_SCALE_RETENTION_MAX = 0.50
PERSISTENT_CHANNEL_RETENTION_MIN = 0.50

NONFINITE = "stage149_nonfinite_blocker"
STAGE148_RECORD_BLOCKER = "stage149_stage148_record_blocker"
PARENT_ROUTE_BLOCKER = "stage149_parent_route_blocker"
IDENTITY_BLOCKER = "stage149_channel_identity_closure_blocker"
COARSE_CANCELLATION = "stage149_coarse_cross_channel_cancellation_stage150_channel_sign_transition_localization_audit"
MIXED_ATTENUATION = "stage149_mixed_component_attenuation_stage150_component_scale_audit"
NO_SIGN_SEPARATION = "stage149_no_cross_channel_sign_separation_stage150_residual_shape_audit"


def validate_stage149_design(
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
    dominant_mirrored_sector=6,
    sample_scale_retention_max=SAMPLE_SCALE_RETENTION_MAX,
    persistent_channel_retention_min=PERSISTENT_CHANNEL_RETENTION_MIN,
    solver_rerun=False,
    solver_endpoint_advanced=False,
    cross_knudsen_extension_permitted=False,
    benchmark_or_validation_claim_permitted=False,
    channel_scale_metrics_used_for_solver=False,
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
        "dominant_mirrored_sector": 6,
        "sample_scale_retention_max": SAMPLE_SCALE_RETENTION_MAX,
        "persistent_channel_retention_min": PERSISTENT_CHANNEL_RETENTION_MIN,
        "solver_rerun": False,
        "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False,
        "benchmark_or_validation_claim_permitted": False,
        "channel_scale_metrics_used_for_solver": False,
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
            raise ValueError(f"Stage 149 frozen-design violation: {key}={got[key]!r}, expected {value!r}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_stage148_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 148
        and record.get("source_head") == EXPECTED_STAGE148_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE148_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE148_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE148_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE148_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE148_SUMMARY_SHA256
        and record.get("sample_scale_curvature_sha256") == EXPECTED_STAGE148_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE148_DECISION
        and record.get("workflow_status") == "completed"
        and record.get("workflow_conclusion") == "success"
    )


def _cancellation_fraction(a: float, b: float, total: float) -> float:
    denom = abs(a) + abs(b)
    if denom == 0.0:
        return 0.0
    return float(1.0 - abs(total) / denom)


def channel_scale_separation_metrics(
    fine_dominant: np.ndarray,
    fine_parent: np.ndarray,
    fine_complement: np.ndarray,
    coarse_components: np.ndarray,
    inherited_retention: np.ndarray,
) -> dict:
    fd = np.asarray(fine_dominant, dtype=float)
    fp = np.asarray(fine_parent, dtype=float)
    fc = np.asarray(fine_complement, dtype=float)
    coarse = np.asarray(coarse_components, dtype=float)
    inherited = np.asarray(inherited_retention, dtype=float)
    if not (fd.shape == fp.shape == fc.shape == (3,)):
        raise ValueError("Stage 149 requires the exact three-point Stage-148 fine curvature arrays")
    if coarse.shape != (3,) or inherited.shape != (3,):
        raise ValueError("Stage 149 requires the exact Stage-148 coarse components and retentions")
    if not all(np.isfinite(a).all() for a in (fd, fp, fc, coarse, inherited)):
        raise ValueError("Stage 149 requires finite inputs")

    fine = np.array([fd[1], fp[1], fc[1]], dtype=float)
    fine_identity = float(abs((fine[0] + fine[1]) - fine[2]))
    coarse_identity = float(abs((coarse[0] + coarse[1]) - coarse[2]))
    tiny = np.finfo(float).tiny
    recomputed_retention = np.abs(coarse) / np.maximum(np.abs(fine), tiny)
    retention_match_error = float(np.max(np.abs(recomputed_retention - inherited)))

    fine_cancel = _cancellation_fraction(fine[0], fine[1], fine[2])
    coarse_cancel = _cancellation_fraction(coarse[0], coarse[1], coarse[2])
    fine_sign_product = int(np.sign(fine[0] * fine[1]))
    coarse_sign_product = int(np.sign(coarse[0] * coarse[1]))
    min_component_retention = float(np.min(recomputed_retention[:2]))
    complement_retention = float(recomputed_retention[2])
    scale_separation_ratio = float(min_component_retention / max(complement_retention, tiny))

    return {
        "fine_center_components": fine.tolist(),
        "coarse_center_components": coarse.tolist(),
        "recomputed_retention": recomputed_retention.tolist(),
        "retention_match_error": retention_match_error,
        "fine_identity_closure": fine_identity,
        "coarse_identity_closure": coarse_identity,
        "maximum_identity_or_provenance_error": max(fine_identity, coarse_identity, retention_match_error),
        "fine_cancellation_fraction": fine_cancel,
        "coarse_cancellation_fraction": coarse_cancel,
        "cancellation_fraction_increase": float(coarse_cancel - fine_cancel),
        "fine_channel_sign_product": fine_sign_product,
        "coarse_channel_sign_product": coarse_sign_product,
        "minimum_component_retention": min_component_retention,
        "complement_retention": complement_retention,
        "minimum_component_to_complement_retention_ratio": scale_separation_ratio,
    }


def classify_channel_scale_separation(
    *, metrics: dict, stage148_record_ok: bool = True, parent_route_ok: bool = True, finite: bool = True
) -> str:
    numeric = [
        metrics.get("maximum_identity_or_provenance_error", np.nan),
        metrics.get("minimum_component_retention", np.nan),
        metrics.get("complement_retention", np.nan),
        metrics.get("coarse_cancellation_fraction", np.nan),
    ]
    if not finite or not np.isfinite(numeric).all():
        return NONFINITE
    if not stage148_record_ok:
        return STAGE148_RECORD_BLOCKER
    if not parent_route_ok:
        return PARENT_ROUTE_BLOCKER
    if float(metrics["maximum_identity_or_provenance_error"]) > IDENTITY_CLOSURE_MAX:
        return IDENTITY_BLOCKER

    complement_sample_scale = float(metrics["complement_retention"]) <= SAMPLE_SCALE_RETENTION_MAX
    both_components_persist = float(metrics["minimum_component_retention"]) >= PERSISTENT_CHANNEL_RETENTION_MIN
    fine_reinforce = int(metrics["fine_channel_sign_product"]) > 0
    coarse_cancel = int(metrics["coarse_channel_sign_product"]) < 0
    if complement_sample_scale and both_components_persist and fine_reinforce and coarse_cancel:
        return COARSE_CANCELLATION
    if complement_sample_scale and not both_components_persist:
        return MIXED_ATTENUATION
    return NO_SIGN_SEPARATION


def run_stage149(stage148_dir: Path, stage148_record: Path, output_dir: Path) -> dict:
    validate_stage149_design()
    summary148 = _load_json(stage148_dir / "summary.json")
    record148 = _load_json(stage148_record)
    stage148_record_ok = _check_stage148_record(record148)
    parent_route_ok = bool(
        summary148.get("stage") == 148
        and summary148.get("decision") == EXPECTED_STAGE148_DECISION
        and summary148.get("aggregate", {}).get("complement_alternating_energy_share", 0.0) >= 0.75
        and summary148.get("aggregate", {}).get("complement_coarse_to_fine_center_retention", 1.0) <= 0.50
    )

    with np.load(stage148_dir / "sample_scale_curvature.npz") as data:
        metrics = channel_scale_separation_metrics(
            data["fine_dominant_curvature"],
            data["fine_parent_curvature"],
            data["fine_complement_deficit"],
            data["coarse_center_components"],
            data["coarse_to_fine_retention"],
        )

    finite = bool(np.isfinite([
        metrics["maximum_identity_or_provenance_error"],
        metrics["minimum_component_retention"],
        metrics["complement_retention"],
        metrics["coarse_cancellation_fraction"],
    ]).all())
    decision = classify_channel_scale_separation(
        metrics=metrics,
        stage148_record_ok=stage148_record_ok,
        parent_route_ok=parent_route_ok,
        finite=finite,
    )

    if decision == COARSE_CANCELLATION:
        conclusion = (
            "The Stage-148 sample-scale complement attenuation is explained algebraically within the frozen decomposition by "
            "cross-channel cancellation at the two-cell scale: both component channels retain at least half of their fine-center "
            "magnitude, but they change from reinforcing at the fine center to opposing at the coarse center. This is a scale-"
            "separation identity in the retained artifact, not evidence of limiter causality, solver stability, or validation."
        )
    elif decision == MIXED_ATTENUATION:
        conclusion = (
            "The complement is sample-scale, but at least one component channel also loses more than half of its fine-center "
            "magnitude. The attenuation therefore cannot be attributed to cross-channel cancellation alone and requires a fixed "
            "component-scale audit."
        )
    else:
        conclusion = (
            "The fixed fine/coarse channel signs do not support a clean cross-channel cancellation explanation of the Stage-148 "
            "sample-scale complement attenuation. The remaining residual shape should be audited without retuning the solver."
        )

    config = {
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
        "dominant_mirrored_sector": 6,
        "sample_scale_retention_max": SAMPLE_SCALE_RETENTION_MAX,
        "persistent_channel_retention_min": PERSISTENT_CHANNEL_RETENTION_MIN,
        "channel_scale_metrics_used_for_solver": False,
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
    payload = {
        "stage": STAGE,
        "decision": decision,
        "finite": finite,
        "configuration": config,
        "parents": {
            "stage148_source_head": EXPECTED_STAGE148_SOURCE_HEAD,
            "stage148_run_id": EXPECTED_STAGE148_RUN_ID,
            "stage148_job_id": EXPECTED_STAGE148_JOB_ID,
            "stage148_artifact_id": EXPECTED_STAGE148_ARTIFACT_ID,
        },
        "aggregate": {
            "stage148_record_ok": stage148_record_ok,
            "parent_route_ok": parent_route_ok,
            "maximum_identity_or_provenance_error": metrics["maximum_identity_or_provenance_error"],
            "minimum_component_retention": metrics["minimum_component_retention"],
            "complement_retention": metrics["complement_retention"],
            "coarse_cancellation_fraction": metrics["coarse_cancellation_fraction"],
            "minimum_component_to_complement_retention_ratio": metrics["minimum_component_to_complement_retention_ratio"],
        },
        "metrics": metrics,
        "scientific_conclusion": conclusion,
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. Stage 149 is an artifact-only channel "
            "scale-separation audit; retention and cancellation fractions are diagnostics, not solver parameters. No physical, "
            "collision/source, floor, wall, reconstruction, transport, limiter, normalization, source-relaxation, or velocity-"
            "quadrature parameter is retuned; no measured scale is fed back into the solver, no solver endpoint or cross-Knudsen "
            "extension is advanced, and no benchmark or validation claim is permitted."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    np.savez(
        output_dir / "channel_scale_separation.npz",
        fine_center_components=np.asarray(metrics["fine_center_components"], dtype=float),
        coarse_center_components=np.asarray(metrics["coarse_center_components"], dtype=float),
        retention=np.asarray(metrics["recomputed_retention"], dtype=float),
        cancellation_fraction=np.asarray([
            metrics["fine_cancellation_fraction"], metrics["coarse_cancellation_fraction"]
        ], dtype=float),
        sign_product=np.asarray([
            metrics["fine_channel_sign_product"], metrics["coarse_channel_sign_product"]
        ], dtype=int),
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage148-dir", required=True, type=Path)
    parser.add_argument("--stage148-record", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    payload = run_stage149(args.stage148_dir, args.stage148_record, args.output_dir)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
