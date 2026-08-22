from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = 152
EXPECTED_STAGE151_SOURCE_HEAD = "f7143890c3750cd2e68a73e2542b43cd74052080"
EXPECTED_STAGE151_RUN_ID = 32435299998
EXPECTED_STAGE151_JOB_ID = 96635189316
EXPECTED_STAGE151_ARTIFACT_ID = 9435128987
EXPECTED_STAGE151_ARTIFACT_SHA256 = "09cc3de00db890cbe582fabb1e9543f9e1e5d7fe466fdf27dc23d50e152a23d4"
EXPECTED_STAGE151_SUMMARY_SHA256 = "f769935c13ba7c4f4faca911fb2ee50bda52e6c8e02affc699a0044a928f47d2"
EXPECTED_STAGE151_PAYLOAD_SHA256 = "ffea9bf184073318dd8c383136e8c87adc8ee6bdd69060eedc1515e7e97dcd67"
EXPECTED_STAGE151_DECISION = (
    "stage151_opposed_channels_recombine_to_single_side_dominance_"
    "stage152_left_outer_channel_balance_audit"
)

PROVENANCE_MATCH_MAX = 1.0e-12
PARENT_CHANNEL_DOMINANCE_MIN = 0.75
MATERIAL_CANCELLATION_MIN = 0.50

NONFINITE = "stage152_nonfinite_blocker"
RECORD_BLOCKER = "stage152_stage151_record_blocker"
PARENT_ROUTE_BLOCKER = "stage152_parent_route_blocker"
PROVENANCE_BLOCKER = "stage152_parent_provenance_blocker"
PARENT_DOMINANT = "stage152_left_parent_channel_dominant_stage153_left_parent_endpoint_scale_audit"
MIXED_CANCELLATION = "stage152_left_balance_mixed_cancellation_stage153_left_channel_endpoint_balance_audit"
NOT_CANCELLATION_DOMINATED = "stage152_left_balance_not_cancellation_dominated_stage153_five_point_left_shape_audit"


def validate_stage152_design(
    *, grid=(64, 64), interior_grid=(56, 56), kn0=10.0, cold_hot_ratio=0.1,
    rule=(40, 96), radial_scale=2.0, limiter="minmod", boundary_slope="zero",
    source_relaxation=1.0, correction_floor=0.05, witness_node=9,
    pair_sectors=(5, 6), dominant_mirrored_sector=6,
    parent_channel_dominance_min=PARENT_CHANNEL_DOMINANCE_MIN,
    material_cancellation_min=MATERIAL_CANCELLATION_MIN,
    balance_metrics_used_for_solver=False, solver_rerun=False,
    solver_endpoint_advanced=False, cross_knudsen_extension_permitted=False,
    benchmark_or_validation_claim_permitted=False, physical_parameter_retuning=False,
    collision_source_retuning=False, floor_retuning=False, wall_retuning=False,
    reconstruction_retuning=False, transport_retuning=False, limiter_retuning=False,
    normalization_retuning=False, source_relaxation_retuning=False,
    velocity_grid_retuning=False,
):
    expected = {
        "grid": (64, 64), "interior_grid": (56, 56), "kn0": 10.0,
        "cold_hot_ratio": 0.1, "rule": (40, 96), "radial_scale": 2.0,
        "limiter": "minmod", "boundary_slope": "zero", "source_relaxation": 1.0,
        "correction_floor": 0.05, "witness_node": 9, "pair_sectors": (5, 6),
        "dominant_mirrored_sector": 6,
        "parent_channel_dominance_min": PARENT_CHANNEL_DOMINANCE_MIN,
        "material_cancellation_min": MATERIAL_CANCELLATION_MIN,
        "balance_metrics_used_for_solver": False, "solver_rerun": False,
        "solver_endpoint_advanced": False, "cross_knudsen_extension_permitted": False,
        "benchmark_or_validation_claim_permitted": False,
        "physical_parameter_retuning": False, "collision_source_retuning": False,
        "floor_retuning": False, "wall_retuning": False,
        "reconstruction_retuning": False, "transport_retuning": False,
        "limiter_retuning": False, "normalization_retuning": False,
        "source_relaxation_retuning": False, "velocity_grid_retuning": False,
    }
    got = locals()
    for key, value in expected.items():
        if got[key] != value:
            raise ValueError(f"Stage 152 frozen-design violation: {key}={got[key]!r}, expected {value!r}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_stage151_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 151
        and record.get("source_head") == EXPECTED_STAGE151_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE151_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE151_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE151_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE151_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE151_SUMMARY_SHA256
        and record.get("outer_interval_asymmetry_sha256") == EXPECTED_STAGE151_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE151_DECISION
        and record.get("workflow_status") == "completed"
        and record.get("workflow_conclusion") == "success"
    )


def left_outer_channel_balance_metrics(
    dominant_outer: np.ndarray,
    parent_outer: np.ndarray,
    combined_outer: np.ndarray,
) -> dict:
    dominant = np.asarray(dominant_outer, dtype=float)
    parent = np.asarray(parent_outer, dtype=float)
    combined = np.asarray(combined_outer, dtype=float)
    if dominant.shape != (2,) or parent.shape != (2,) or combined.shape != (2,):
        raise ValueError("Stage 152 requires three finite two-sided outer-increment vectors")
    if not (np.isfinite(dominant).all() and np.isfinite(parent).all() and np.isfinite(combined).all()):
        raise ValueError("Stage 152 requires finite Stage-151 vectors")

    closure_vec = combined - (dominant + parent)
    closure = float(np.max(np.abs(closure_vec)))
    dleft = float(dominant[0])
    pleft = float(parent[0])
    cleft = float(combined[0])
    l1 = abs(dleft) + abs(pleft)
    if l1 <= 0.0:
        parent_share = dominant_share = cancellation = net_l1_share = 0.0
    else:
        parent_share = abs(pleft) / l1
        dominant_share = abs(dleft) / l1
        net_l1_share = abs(cleft) / l1
        cancellation = 1.0 - net_l1_share
    parent_to_dominant = float(abs(pleft) / abs(dleft)) if abs(dleft) > 0.0 else np.inf
    net_to_parent = float(abs(cleft) / abs(pleft)) if abs(pleft) > 0.0 else np.inf

    return {
        "dominant_left": dleft,
        "parent_left": pleft,
        "combined_left": cleft,
        "left_channel_sign_product": int(np.sign(dleft * pleft)),
        "left_parent_absolute_share": float(parent_share),
        "left_dominant_absolute_share": float(dominant_share),
        "left_parent_dominance_margin": float(parent_share - PARENT_CHANNEL_DOMINANCE_MIN),
        "left_cancellation_fraction": float(cancellation),
        "left_net_l1_share": float(net_l1_share),
        "left_parent_to_dominant_magnitude_ratio": parent_to_dominant,
        "left_net_to_parent_magnitude_ratio": net_to_parent,
        "right_channel_sign_product": int(np.sign(float(dominant[1] * parent[1]))),
        "recombination_closure_vector": closure_vec.tolist(),
        "maximum_identity_or_provenance_error": closure,
    }


def classify_left_outer_channel_balance(
    *, metrics: dict, stage151_record_ok=True, parent_route_ok=True, finite=True
) -> str:
    numeric = [
        metrics.get("left_parent_absolute_share", np.nan),
        metrics.get("left_cancellation_fraction", np.nan),
        metrics.get("maximum_identity_or_provenance_error", np.nan),
    ]
    if not finite or not np.isfinite(numeric).all():
        return NONFINITE
    if not stage151_record_ok:
        return RECORD_BLOCKER
    if not parent_route_ok:
        return PARENT_ROUTE_BLOCKER
    if float(metrics["maximum_identity_or_provenance_error"]) > PROVENANCE_MATCH_MAX:
        return PROVENANCE_BLOCKER
    if (
        int(metrics["left_channel_sign_product"]) < 0
        and float(metrics["left_parent_absolute_share"]) >= PARENT_CHANNEL_DOMINANCE_MIN
    ):
        return PARENT_DOMINANT
    if (
        int(metrics["left_channel_sign_product"]) < 0
        and float(metrics["left_cancellation_fraction"]) >= MATERIAL_CANCELLATION_MIN
    ):
        return MIXED_CANCELLATION
    return NOT_CANCELLATION_DOMINATED


def run_stage152(stage151_dir: Path, stage151_record: Path, output_dir: Path) -> dict:
    validate_stage152_design()
    summary151 = _load_json(stage151_dir / "summary.json")
    record151 = _load_json(stage151_record)
    stage151_record_ok = _check_stage151_record(record151)
    parent_route_ok = bool(
        summary151.get("stage") == 151
        and summary151.get("decision") == EXPECTED_STAGE151_DECISION
        and summary151.get("aggregate", {}).get("combined_stronger_side") == "left"
        and summary151.get("aggregate", {}).get("combined_stronger_absolute_share", 0.0) >= 0.75
    )

    with np.load(stage151_dir / "outer_interval_asymmetry.npz") as data:
        dominant_outer = np.asarray(data["dominant_outer_increments"], dtype=float)
        parent_outer = np.asarray(data["parent_outer_increments"], dtype=float)
        combined_outer = np.asarray(data["combined_outer_increments"], dtype=float)
    metrics = left_outer_channel_balance_metrics(dominant_outer, parent_outer, combined_outer)
    finite = bool(np.isfinite([
        metrics["left_parent_absolute_share"],
        metrics["left_cancellation_fraction"],
        metrics["maximum_identity_or_provenance_error"],
    ]).all())
    decision = classify_left_outer_channel_balance(
        metrics=metrics,
        stage151_record_ok=stage151_record_ok,
        parent_route_ok=parent_route_ok,
        finite=finite,
    )

    if decision == PARENT_DOMINANT:
        conclusion = (
            "The left outer interval is cross-channel opposed but the parent channel exceeds the fixed 75% absolute-share "
            "guard. The next artifact-only question is whether that parent-channel dominance is already present at the fine "
            "endpoint or emerges only in the coarse-scale increment."
        )
    elif decision == MIXED_CANCELLATION:
        conclusion = (
            "The left outer interval is cross-channel opposed and cancellation removes at least half of the componentwise "
            "absolute magnitude, while the parent channel remains below the fixed 75% single-channel dominance guard. "
            "Therefore the Stage-151 left localization is a mixed two-channel balance rather than a robust one-channel "
            "mechanism. The next artifact-only audit should resolve the left fine/coarse endpoint balance before any deeper "
            "mechanistic interpretation."
        )
    else:
        conclusion = (
            "The left outer interval does not meet either the fixed single-channel dominance or material-cancellation route. "
            "The remaining five-point shape must stay descriptive and artifact-only."
        )

    config = {
        "grid": [64, 64], "interior_grid": [56, 56], "kn0": 10.0, "cold_hot_ratio": 0.1,
        "rule": [40, 96], "radial_scale": 2.0, "limiter": "minmod", "boundary_slope": "zero",
        "source_relaxation": 1.0, "correction_floor": 0.05, "witness_node": 9,
        "pair_sectors": [5, 6], "dominant_mirrored_sector": 6,
        "parent_channel_dominance_min": PARENT_CHANNEL_DOMINANCE_MIN,
        "material_cancellation_min": MATERIAL_CANCELLATION_MIN,
        "balance_metrics_used_for_solver": False, "solver_rerun": False,
        "solver_endpoint_advanced": False, "cross_knudsen_extension_permitted": False,
        "benchmark_or_validation_claim_permitted": False,
        "physical_parameter_retuning": False, "collision_source_retuning": False,
        "floor_retuning": False, "wall_retuning": False, "reconstruction_retuning": False,
        "transport_retuning": False, "limiter_retuning": False, "normalization_retuning": False,
        "source_relaxation_retuning": False, "velocity_grid_retuning": False,
    }
    payload = {
        "stage": STAGE,
        "decision": decision,
        "finite": finite,
        "configuration": config,
        "parent": {
            "stage151_source_head": EXPECTED_STAGE151_SOURCE_HEAD,
            "stage151_run_id": EXPECTED_STAGE151_RUN_ID,
            "stage151_job_id": EXPECTED_STAGE151_JOB_ID,
            "stage151_artifact_id": EXPECTED_STAGE151_ARTIFACT_ID,
        },
        "aggregate": {
            "stage151_record_ok": stage151_record_ok,
            "parent_route_ok": parent_route_ok,
            "left_channel_sign_product": metrics["left_channel_sign_product"],
            "left_parent_absolute_share": metrics["left_parent_absolute_share"],
            "left_cancellation_fraction": metrics["left_cancellation_fraction"],
            "maximum_identity_or_provenance_error": metrics["maximum_identity_or_provenance_error"],
        },
        "metrics": metrics,
        "scientific_conclusion": conclusion,
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. Stage 152 is an artifact-only "
            "left outer-channel balance audit; channel shares, cancellation, and magnitude ratios are diagnostics, not solver "
            "parameters. No physical, collision/source, floor, wall, reconstruction, transport, limiter, normalization, "
            "source-relaxation, or velocity-quadrature parameter is retuned; no measured balance is fed back into the solver, "
            "no solver endpoint or cross-Knudsen extension is advanced, and no benchmark or validation claim is permitted."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        output_dir / "left_outer_channel_balance.npz",
        dominant_outer_increments=dominant_outer,
        parent_outer_increments=parent_outer,
        combined_outer_increments=combined_outer,
        left_balance_vector=np.asarray([
            metrics["dominant_left"], metrics["parent_left"], metrics["combined_left"]
        ], dtype=float),
        left_share_vector=np.asarray([
            metrics["left_dominant_absolute_share"], metrics["left_parent_absolute_share"]
        ], dtype=float),
        left_cancellation_fraction=np.asarray(metrics["left_cancellation_fraction"], dtype=float),
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 152 fixed left outer-channel balance audit")
    parser.add_argument("--stage151-dir", type=Path, required=True)
    parser.add_argument("--stage151-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage152(args.stage151_dir, args.stage151_record, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
