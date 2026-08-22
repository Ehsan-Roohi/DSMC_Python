from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = 151
EXPECTED_STAGE150_SOURCE_HEAD = "1e4fe5ff5e3542e6e0cc7c8b036c9f5ad6dda994"
EXPECTED_STAGE150_RUN_ID = 32431079197
EXPECTED_STAGE150_JOB_ID = 96622574831
EXPECTED_STAGE150_ARTIFACT_ID = 9430006423
EXPECTED_STAGE150_ARTIFACT_SHA256 = "ce7e2be6eb7e5bc93dca540430c58671642a68706012a61afea61600972b45d8"
EXPECTED_STAGE150_SUMMARY_SHA256 = "fbabcb9d83d008b0d52b96ea80c546636461c150a821a713cf773bfe31f7376a"
EXPECTED_STAGE150_PAYLOAD_SHA256 = "5e841e56d1d22d82fd73f9bfd6193857dc65b2ac42849ced065b273d264dc5c0"
EXPECTED_STAGE150_DECISION = "stage150_opposing_outer_intervals_no_single_side_dominance_stage151_outer_interval_asymmetry_audit"

PROVENANCE_MATCH_MAX = 1.0e-12
COMBINED_SIDE_DOMINANCE_MIN = 0.75

NONFINITE = "stage151_nonfinite_blocker"
RECORD_BLOCKER = "stage151_stage150_record_blocker"
PARENT_ROUTE_BLOCKER = "stage151_parent_route_blocker"
PROVENANCE_BLOCKER = "stage151_parent_provenance_blocker"
OPPOSED_RECOMBINED_DOMINANCE = "stage151_opposed_channels_recombine_to_single_side_dominance_stage152_left_outer_channel_balance_audit"
DISTRIBUTED = "stage151_distributed_outer_asymmetry_stage152_five_point_scale_balance_audit"


def validate_stage151_design(
    *, grid=(64, 64), interior_grid=(56, 56), kn0=10.0, cold_hot_ratio=0.1, rule=(40, 96),
    radial_scale=2.0, limiter="minmod", boundary_slope="zero", source_relaxation=1.0,
    correction_floor=0.05, witness_node=9, pair_sectors=(5, 6), dominant_mirrored_sector=6,
    combined_side_dominance_min=COMBINED_SIDE_DOMINANCE_MIN, asymmetry_metrics_used_for_solver=False,
    solver_rerun=False, solver_endpoint_advanced=False, cross_knudsen_extension_permitted=False,
    benchmark_or_validation_claim_permitted=False, physical_parameter_retuning=False,
    collision_source_retuning=False, floor_retuning=False, wall_retuning=False,
    reconstruction_retuning=False, transport_retuning=False, limiter_retuning=False,
    normalization_retuning=False, source_relaxation_retuning=False, velocity_grid_retuning=False,
):
    expected = {
        "grid": (64, 64), "interior_grid": (56, 56), "kn0": 10.0, "cold_hot_ratio": 0.1,
        "rule": (40, 96), "radial_scale": 2.0, "limiter": "minmod", "boundary_slope": "zero",
        "source_relaxation": 1.0, "correction_floor": 0.05, "witness_node": 9,
        "pair_sectors": (5, 6), "dominant_mirrored_sector": 6,
        "combined_side_dominance_min": COMBINED_SIDE_DOMINANCE_MIN, "asymmetry_metrics_used_for_solver": False,
        "solver_rerun": False, "solver_endpoint_advanced": False, "cross_knudsen_extension_permitted": False,
        "benchmark_or_validation_claim_permitted": False, "physical_parameter_retuning": False,
        "collision_source_retuning": False, "floor_retuning": False, "wall_retuning": False,
        "reconstruction_retuning": False, "transport_retuning": False, "limiter_retuning": False,
        "normalization_retuning": False, "source_relaxation_retuning": False, "velocity_grid_retuning": False,
    }
    got = locals()
    for key, value in expected.items():
        if got[key] != value:
            raise ValueError(f"Stage 151 frozen-design violation: {key}={got[key]!r}, expected {value!r}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_stage150_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 150
        and record.get("source_head") == EXPECTED_STAGE150_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE150_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE150_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE150_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE150_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE150_SUMMARY_SHA256
        and record.get("channel_sign_transition_localization_sha256") == EXPECTED_STAGE150_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE150_DECISION
        and record.get("workflow_status") == "completed"
        and record.get("workflow_conclusion") == "success"
    )


def _asymmetry_metrics(vec: np.ndarray) -> dict:
    q = np.asarray(vec, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("Stage 151 requires a finite two-sided outer-increment vector")
    left, right = map(float, q)
    denom = abs(left) + abs(right)
    if denom <= 0.0:
        return {
            "left": left, "right": right, "signed_sum": left + right,
            "left_absolute_share": 0.0, "right_absolute_share": 0.0,
            "stronger_side": "none", "stronger_absolute_share": 0.0,
            "absolute_asymmetry_index": 0.0, "side_sign_product": 0,
            "cancellation_fraction": 0.0,
        }
    left_share = abs(left) / denom
    right_share = abs(right) / denom
    return {
        "left": left, "right": right, "signed_sum": left + right,
        "left_absolute_share": float(left_share), "right_absolute_share": float(right_share),
        "stronger_side": "left" if left_share >= right_share else "right",
        "stronger_absolute_share": float(max(left_share, right_share)),
        "absolute_asymmetry_index": float((abs(left) - abs(right)) / denom),
        "side_sign_product": int(np.sign(left * right)),
        "cancellation_fraction": float(1.0 - abs(left + right) / denom),
    }


def outer_interval_asymmetry_metrics(
    dominant_outer: np.ndarray,
    parent_outer: np.ndarray,
    fine_components: np.ndarray,
    coarse_components: np.ndarray,
) -> dict:
    dominant_outer = np.asarray(dominant_outer, dtype=float)
    parent_outer = np.asarray(parent_outer, dtype=float)
    fine = np.asarray(fine_components, dtype=float)
    coarse = np.asarray(coarse_components, dtype=float)
    if fine.shape != (2,) or coarse.shape != (2,):
        raise ValueError("Stage 151 requires the exact Stage-150 two-component fine/coarse vectors")
    if not np.isfinite(fine).all() or not np.isfinite(coarse).all():
        raise ValueError("Stage 151 requires finite Stage-150 fine/coarse vectors")

    dominant = _asymmetry_metrics(dominant_outer)
    parent = _asymmetry_metrics(parent_outer)
    combined_outer = dominant_outer + parent_outer
    combined = _asymmetry_metrics(combined_outer)

    denom = float(np.linalg.norm(dominant_outer) * np.linalg.norm(parent_outer))
    channel_cosine = float(np.dot(dominant_outer, parent_outer) / denom) if denom > 0.0 else 0.0
    sidewise_sign_products = np.sign(dominant_outer * parent_outer).astype(int)

    fine_combined = float(np.sum(fine))
    coarse_combined = float(np.sum(coarse))
    expected_net = float(coarse_combined - fine_combined)
    recombined_net = float(np.sum(combined_outer))
    closure = float(abs(recombined_net - expected_net))

    return {
        "dominant": dominant,
        "parent": parent,
        "combined": combined,
        "combined_outer_increments": combined_outer.tolist(),
        "channel_outer_vector_cosine": channel_cosine,
        "sidewise_channel_sign_products": sidewise_sign_products.tolist(),
        "opposite_asymmetry_orientation": bool(
            dominant["absolute_asymmetry_index"] * parent["absolute_asymmetry_index"] < 0.0
        ),
        "both_sides_cross_channel_opposed": bool(np.all(sidewise_sign_products < 0)),
        "fine_combined": fine_combined,
        "coarse_combined": coarse_combined,
        "expected_combined_scale_increment": expected_net,
        "recombined_outer_scale_increment": recombined_net,
        "recombined_closure_error": closure,
        "maximum_identity_or_provenance_error": closure,
    }


def classify_outer_interval_asymmetry(
    *, metrics: dict, stage150_record_ok=True, parent_route_ok=True, finite=True
) -> str:
    numeric = [
        metrics.get("maximum_identity_or_provenance_error", np.nan),
        metrics.get("combined", {}).get("stronger_absolute_share", np.nan),
        metrics.get("channel_outer_vector_cosine", np.nan),
    ]
    if not finite or not np.isfinite(numeric).all():
        return NONFINITE
    if not stage150_record_ok:
        return RECORD_BLOCKER
    if not parent_route_ok:
        return PARENT_ROUTE_BLOCKER
    if float(metrics["maximum_identity_or_provenance_error"]) > PROVENANCE_MATCH_MAX:
        return PROVENANCE_BLOCKER
    if (
        bool(metrics["both_sides_cross_channel_opposed"])
        and bool(metrics["opposite_asymmetry_orientation"])
        and float(metrics["combined"]["stronger_absolute_share"]) >= COMBINED_SIDE_DOMINANCE_MIN
    ):
        return OPPOSED_RECOMBINED_DOMINANCE
    return DISTRIBUTED


def run_stage151(stage150_dir: Path, stage150_record: Path, output_dir: Path) -> dict:
    validate_stage151_design()
    summary150 = _load_json(stage150_dir / "summary.json")
    record150 = _load_json(stage150_record)
    stage150_record_ok = _check_stage150_record(record150)
    parent_route_ok = bool(
        summary150.get("stage") == 150
        and summary150.get("decision") == EXPECTED_STAGE150_DECISION
        and summary150.get("aggregate", {}).get("parent_channel_sign_transition") is True
        and summary150.get("aggregate", {}).get("parent_stronger_outer_absolute_share", 1.0) < 0.75
    )

    with np.load(stage150_dir / "channel_sign_transition_localization.npz") as data:
        metrics = outer_interval_asymmetry_metrics(
            data["dominant_outer_increments"],
            data["parent_outer_increments"],
            data["fine_components"],
            data["coarse_components"],
        )
        dominant_outer = np.asarray(data["dominant_outer_increments"], dtype=float)
        parent_outer = np.asarray(data["parent_outer_increments"], dtype=float)

    finite = bool(np.isfinite([
        metrics["maximum_identity_or_provenance_error"],
        metrics["combined"]["stronger_absolute_share"],
        metrics["channel_outer_vector_cosine"],
    ]).all())
    decision = classify_outer_interval_asymmetry(
        metrics=metrics,
        stage150_record_ok=stage150_record_ok,
        parent_route_ok=parent_route_ok,
        finite=finite,
    )

    if decision == OPPOSED_RECOMBINED_DOMINANCE:
        conclusion = (
            "The Stage-150 parent-channel sign transition is internally two-sided, but the two inherited component "
            "channels oppose one another on both outer intervals and have opposite left/right asymmetry orientations. "
            "After exact recombination, both outer contributions have the same sign and the left interval carries "
            "more than the fixed 75% absolute-share guard. The coarse complement attenuation is therefore left-heavy "
            "at this frozen five-point scale even though the parent channel itself is cancellation-dominated. This is "
            "an artifact-level decomposition, not evidence of limiter causality or solver stability."
        )
    else:
        conclusion = (
            "The fixed outer-interval decomposition does not support a single recombined-side localization under the "
            "preregistered guard. The remaining structure must stay distributed and artifact-only; no solver change is justified."
        )

    config = {
        "grid": [64, 64], "interior_grid": [56, 56], "kn0": 10.0, "cold_hot_ratio": 0.1,
        "rule": [40, 96], "radial_scale": 2.0, "limiter": "minmod", "boundary_slope": "zero",
        "source_relaxation": 1.0, "correction_floor": 0.05, "witness_node": 9,
        "pair_sectors": [5, 6], "dominant_mirrored_sector": 6,
        "combined_side_dominance_min": COMBINED_SIDE_DOMINANCE_MIN,
        "asymmetry_metrics_used_for_solver": False, "solver_rerun": False, "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False, "benchmark_or_validation_claim_permitted": False,
        "physical_parameter_retuning": False, "collision_source_retuning": False, "floor_retuning": False,
        "wall_retuning": False, "reconstruction_retuning": False, "transport_retuning": False,
        "limiter_retuning": False, "normalization_retuning": False, "source_relaxation_retuning": False,
        "velocity_grid_retuning": False,
    }
    payload = {
        "stage": STAGE,
        "decision": decision,
        "finite": finite,
        "configuration": config,
        "parent": {
            "stage150_source_head": EXPECTED_STAGE150_SOURCE_HEAD,
            "stage150_run_id": EXPECTED_STAGE150_RUN_ID,
            "stage150_job_id": EXPECTED_STAGE150_JOB_ID,
            "stage150_artifact_id": EXPECTED_STAGE150_ARTIFACT_ID,
        },
        "aggregate": {
            "stage150_record_ok": stage150_record_ok,
            "parent_route_ok": parent_route_ok,
            "both_sides_cross_channel_opposed": metrics["both_sides_cross_channel_opposed"],
            "opposite_asymmetry_orientation": metrics["opposite_asymmetry_orientation"],
            "channel_outer_vector_cosine": metrics["channel_outer_vector_cosine"],
            "combined_stronger_side": metrics["combined"]["stronger_side"],
            "combined_stronger_absolute_share": metrics["combined"]["stronger_absolute_share"],
            "maximum_identity_or_provenance_error": metrics["maximum_identity_or_provenance_error"],
        },
        "metrics": metrics,
        "scientific_conclusion": conclusion,
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. Stage 151 is an artifact-only "
            "outer-interval asymmetry audit; side shares, vector cosine, and recombined localization are diagnostics, "
            "not solver parameters. No physical, collision/source, floor, wall, reconstruction, transport, limiter, "
            "normalization, source-relaxation, or velocity-quadrature parameter is retuned; no measured asymmetry is fed "
            "back into the solver, no solver endpoint or cross-Knudsen extension is advanced, and no benchmark or validation "
            "claim is permitted."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        output_dir / "outer_interval_asymmetry.npz",
        dominant_outer_increments=dominant_outer,
        parent_outer_increments=parent_outer,
        combined_outer_increments=np.asarray(metrics["combined_outer_increments"], dtype=float),
        sidewise_channel_sign_products=np.asarray(metrics["sidewise_channel_sign_products"], dtype=int),
        asymmetry_indices=np.asarray([
            metrics["dominant"]["absolute_asymmetry_index"],
            metrics["parent"]["absolute_asymmetry_index"],
            metrics["combined"]["absolute_asymmetry_index"],
        ], dtype=float),
        stronger_absolute_shares=np.asarray([
            metrics["dominant"]["stronger_absolute_share"],
            metrics["parent"]["stronger_absolute_share"],
            metrics["combined"]["stronger_absolute_share"],
        ], dtype=float),
    )
    return payload


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Stage 151 fixed outer-interval asymmetry audit")
    parser.add_argument("--stage150-dir", required=True, type=Path)
    parser.add_argument("--stage150-record", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run_stage151(args.stage150_dir, args.stage150_record, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
