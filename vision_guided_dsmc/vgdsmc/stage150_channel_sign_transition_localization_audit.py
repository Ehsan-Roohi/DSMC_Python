from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = 150
EXPECTED_STAGE149_SOURCE_HEAD = "cce38dad370667d1c737d21ca09ef1b0ab80c4e8"
EXPECTED_STAGE149_RUN_ID = 32407008365
EXPECTED_STAGE149_JOB_ID = 96548337522
EXPECTED_STAGE149_ARTIFACT_ID = 9428416962
EXPECTED_STAGE149_ARTIFACT_SHA256 = "db342eacc24d33d824ebc937b2160e48abb93a798dd204498dd0659a12b75f10"
EXPECTED_STAGE149_SUMMARY_SHA256 = "b30599ac2a69a91e3d8db9d698e7b5fe8ef9199fecaad32280cdc7a8d857bd35"
EXPECTED_STAGE149_PAYLOAD_SHA256 = "0234348c649519528e03d1a883d3a16beaca9f160b9ca0d69d4f5e8846cbcd50"
EXPECTED_STAGE149_DECISION = "stage149_coarse_cross_channel_cancellation_stage150_channel_sign_transition_localization_audit"

EXPECTED_STAGE147_SOURCE_HEAD = "3dd94ff4b773ee21358a88a25e660c776988406d"
EXPECTED_STAGE147_RUN_ID = 32358081943
EXPECTED_STAGE147_JOB_ID = 96391517821
EXPECTED_STAGE147_ARTIFACT_ID = 9410888827
EXPECTED_STAGE147_ARTIFACT_SHA256 = "444949e992230e54f224aa254fcf1f4c1743b535aaacd39e38d43d5ef20709c4"
EXPECTED_STAGE147_SUMMARY_SHA256 = "bf8fa6cc06e510fa2f4fbd76726db8e03faf5237bc95cc897fbfa7d42cf0ba5b"
EXPECTED_STAGE147_PAYLOAD_SHA256 = "a77b8bc80a291fe9e440d30f7c09e768964e86b9e873a9ab9890074caf8d6c2d"
EXPECTED_STAGE147_DECISION = "stage147_material_bilateral_curvature_sign_reversal_stage148_sample_scale_curvature_alternation_audit"

PROVENANCE_MATCH_MAX = 1.0e-12
OUTER_SIDE_DOMINANCE_MIN = 0.75

NONFINITE = "stage150_nonfinite_blocker"
STAGE149_RECORD_BLOCKER = "stage150_stage149_record_blocker"
STAGE147_RECORD_BLOCKER = "stage150_stage147_record_blocker"
PARENT_ROUTE_BLOCKER = "stage150_parent_route_blocker"
PROVENANCE_BLOCKER = "stage150_parent_provenance_blocker"
OPPOSING_NO_SINGLE_SIDE = "stage150_opposing_outer_intervals_no_single_side_dominance_stage151_outer_interval_asymmetry_audit"
SINGLE_SIDE = "stage150_single_outer_interval_dominant_stage151_single_side_continuity_audit"
NO_TRANSITION = "stage150_no_channel_sign_transition_stage151_residual_scale_audit"


def validate_stage150_design(
    *, grid=(64, 64), interior_grid=(56, 56), kn0=10.0, cold_hot_ratio=0.1, rule=(40, 96),
    radial_scale=2.0, limiter="minmod", boundary_slope="zero", source_relaxation=1.0,
    correction_floor=0.05, witness_node=9, pair_sectors=(5, 6), dominant_mirrored_sector=6,
    outer_side_dominance_min=OUTER_SIDE_DOMINANCE_MIN, transition_metrics_used_for_solver=False,
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
        "outer_side_dominance_min": OUTER_SIDE_DOMINANCE_MIN, "transition_metrics_used_for_solver": False,
        "solver_rerun": False, "solver_endpoint_advanced": False, "cross_knudsen_extension_permitted": False,
        "benchmark_or_validation_claim_permitted": False, "physical_parameter_retuning": False,
        "collision_source_retuning": False, "floor_retuning": False, "wall_retuning": False,
        "reconstruction_retuning": False, "transport_retuning": False, "limiter_retuning": False,
        "normalization_retuning": False, "source_relaxation_retuning": False, "velocity_grid_retuning": False,
    }
    got = locals()
    for key, value in expected.items():
        if got[key] != value:
            raise ValueError(f"Stage 150 frozen-design violation: {key}={got[key]!r}, expected {value!r}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_stage149_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 149 and record.get("source_head") == EXPECTED_STAGE149_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE149_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE149_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE149_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE149_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE149_SUMMARY_SHA256
        and record.get("channel_scale_separation_sha256") == EXPECTED_STAGE149_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE149_DECISION
        and record.get("workflow_status") == "completed" and record.get("workflow_conclusion") == "success"
    )


def _check_stage147_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 147 and record.get("source_head") == EXPECTED_STAGE147_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE147_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE147_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE147_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE147_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE147_SUMMARY_SHA256
        and record.get("dual_channel_neighborhood_sha256") == EXPECTED_STAGE147_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE147_DECISION
        and record.get("workflow_status") == "completed" and record.get("workflow_conclusion") == "success"
    )


def _component_center_curvatures(values: np.ndarray, sign: float) -> tuple[float, float, float, float]:
    q = np.asarray(values, dtype=float)
    if q.shape != (5,) or not np.isfinite(q).all():
        raise ValueError("Stage 150 requires a finite five-point channel profile")
    fine = sign * (0.5 * (q[1] + q[3]) - q[2])
    coarse = sign * (0.5 * (q[0] + q[4]) - q[2])
    left_outer = sign * 0.5 * (q[0] - q[1])
    right_outer = sign * 0.5 * (q[4] - q[3])
    return float(fine), float(coarse), float(left_outer), float(right_outer)


def _outer_metrics(fine: float, coarse: float, left: float, right: float) -> dict:
    net = float(coarse - fine)
    closure = float(abs((left + right) - net))
    denom = abs(left) + abs(right)
    max_share = float(max(abs(left), abs(right)) / denom) if denom else 0.0
    cancellation = float(1.0 - abs(net) / denom) if denom else 0.0
    tiny = np.finfo(float).tiny
    return {
        "fine_center": fine, "coarse_center": coarse, "net_scale_increment": net,
        "left_outer_increment": left, "right_outer_increment": right, "outer_increment_closure": closure,
        "channel_sign_transition": bool(fine * coarse < 0.0), "outer_sign_product": int(np.sign(left * right)),
        "stronger_outer_side": "left" if abs(left) >= abs(right) else "right",
        "stronger_outer_absolute_share": max_share, "outer_cancellation_fraction": cancellation,
        "net_increment_to_fine_ratio": float(abs(net) / max(abs(fine), tiny)),
        "coarse_to_fine_magnitude_ratio": float(abs(coarse) / max(abs(fine), tiny)),
    }


def channel_sign_transition_metrics(dominant_signed, parent_signed, stage149_fine_components, stage149_coarse_components) -> dict:
    dominant = np.asarray(dominant_signed, dtype=float)
    parent = np.asarray(parent_signed, dtype=float)
    fine149 = np.asarray(stage149_fine_components, dtype=float)
    coarse149 = np.asarray(stage149_coarse_components, dtype=float)
    if fine149.shape != (3,) or coarse149.shape != (3,):
        raise ValueError("Stage 150 requires exact Stage-149 fine/coarse component vectors")
    if not np.isfinite(fine149).all() or not np.isfinite(coarse149).all():
        raise ValueError("Stage 150 requires finite Stage-149 components")
    d = _outer_metrics(*_component_center_curvatures(dominant, -1.0))
    p = _outer_metrics(*_component_center_curvatures(parent, +1.0))
    expected_fine = np.array([d["fine_center"], p["fine_center"], d["fine_center"] + p["fine_center"]])
    expected_coarse = np.array([d["coarse_center"], p["coarse_center"], d["coarse_center"] + p["coarse_center"]])
    provenance_error = float(max(np.max(np.abs(expected_fine - fine149)), np.max(np.abs(expected_coarse - coarse149))))
    identity_error = float(max(abs(expected_fine[2] - fine149[2]), abs(expected_coarse[2] - coarse149[2])))
    return {
        "dominant": d, "parent": p, "stage149_provenance_match_error": provenance_error,
        "channel_identity_error": identity_error,
        "maximum_identity_or_provenance_error": max(provenance_error, identity_error, d["outer_increment_closure"], p["outer_increment_closure"]),
    }


def classify_channel_sign_transition(*, metrics: dict, stage149_record_ok=True, stage147_record_ok=True, parent_route_ok=True, finite=True) -> str:
    numeric = [metrics.get("maximum_identity_or_provenance_error", np.nan), metrics.get("parent", {}).get("stronger_outer_absolute_share", np.nan), metrics.get("parent", {}).get("outer_cancellation_fraction", np.nan)]
    if not finite or not np.isfinite(numeric).all(): return NONFINITE
    if not stage149_record_ok: return STAGE149_RECORD_BLOCKER
    if not stage147_record_ok: return STAGE147_RECORD_BLOCKER
    if not parent_route_ok: return PARENT_ROUTE_BLOCKER
    if float(metrics["maximum_identity_or_provenance_error"]) > PROVENANCE_MATCH_MAX: return PROVENANCE_BLOCKER
    parent = metrics["parent"]
    if not bool(parent["channel_sign_transition"]): return NO_TRANSITION
    if int(parent["outer_sign_product"]) < 0 and float(parent["stronger_outer_absolute_share"]) < OUTER_SIDE_DOMINANCE_MIN:
        return OPPOSING_NO_SINGLE_SIDE
    return SINGLE_SIDE


def run_stage150(stage149_dir: Path, stage149_record: Path, stage147_dir: Path, stage147_record: Path, output_dir: Path) -> dict:
    validate_stage150_design()
    summary149, record149 = _load_json(stage149_dir / "summary.json"), _load_json(stage149_record)
    summary147, record147 = _load_json(stage147_dir / "summary.json"), _load_json(stage147_record)
    stage149_record_ok, stage147_record_ok = _check_stage149_record(record149), _check_stage147_record(record147)
    parent_route_ok = bool(
        summary149.get("stage") == 149 and summary149.get("decision") == EXPECTED_STAGE149_DECISION
        and summary149.get("aggregate", {}).get("complement_retention", 1.0) <= 0.50
        and summary149.get("metrics", {}).get("fine_channel_sign_product") == 1
        and summary149.get("metrics", {}).get("coarse_channel_sign_product") == -1
        and summary147.get("stage") == 147 and summary147.get("decision") == EXPECTED_STAGE147_DECISION
    )
    with np.load(stage149_dir / "channel_scale_separation.npz") as data149, np.load(stage147_dir / "dual_channel_neighborhood.npz") as data147:
        metrics = channel_sign_transition_metrics(data147["five_point_dominant_signed"], data147["five_point_parent_signed"], data149["fine_center_components"], data149["coarse_center_components"])
        depth = np.asarray(data147["five_point_depth"], dtype=float)
    finite = bool(np.isfinite([metrics["maximum_identity_or_provenance_error"], metrics["parent"]["stronger_outer_absolute_share"], metrics["parent"]["outer_cancellation_fraction"]]).all())
    decision = classify_channel_sign_transition(metrics=metrics, stage149_record_ok=stage149_record_ok, stage147_record_ok=stage147_record_ok, parent_route_ok=parent_route_ok, finite=finite)
    parent, dominant = metrics["parent"], metrics["dominant"]
    if decision == OPPOSING_NO_SINGLE_SIDE:
        conclusion = "The parent channel is the channel that changes sign between the one-cell and two-cell secant scales, but the change is not localized to a single outer interval. The left and right outer-span contributions oppose each other, and the stronger side remains below the fixed 75% absolute-share guard. The observed coarse cancellation is therefore an asymmetric two-sided scale effect in the frozen five-point profile, not a one-sided defect and not evidence of solver causality."
    elif decision == SINGLE_SIDE:
        conclusion = "The parent-channel sign transition is materially dominated by one fixed outer interval under the preregistered 75% absolute-share guard. This supports a fixed continuity audit of that side only; it does not justify any solver retuning."
    else:
        conclusion = "The exact five-point reconstruction does not preserve the Stage-149 channel sign-transition premise. Further routing must remain artifact-only until the provenance discrepancy is resolved."
    config = {
        "grid": [64, 64], "interior_grid": [56, 56], "kn0": 10.0, "cold_hot_ratio": 0.1,
        "rule": [40, 96], "radial_scale": 2.0, "limiter": "minmod", "boundary_slope": "zero",
        "source_relaxation": 1.0, "correction_floor": 0.05, "witness_node": 9, "pair_sectors": [5, 6],
        "dominant_mirrored_sector": 6, "outer_side_dominance_min": OUTER_SIDE_DOMINANCE_MIN,
        "transition_metrics_used_for_solver": False, "solver_rerun": False, "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False, "benchmark_or_validation_claim_permitted": False,
        "physical_parameter_retuning": False, "collision_source_retuning": False, "floor_retuning": False,
        "wall_retuning": False, "reconstruction_retuning": False, "transport_retuning": False,
        "limiter_retuning": False, "normalization_retuning": False, "source_relaxation_retuning": False,
        "velocity_grid_retuning": False,
    }
    payload = {
        "stage": STAGE, "decision": decision, "finite": finite, "configuration": config,
        "parents": {"stage149_source_head": EXPECTED_STAGE149_SOURCE_HEAD, "stage149_run_id": EXPECTED_STAGE149_RUN_ID, "stage149_job_id": EXPECTED_STAGE149_JOB_ID, "stage149_artifact_id": EXPECTED_STAGE149_ARTIFACT_ID, "stage147_source_head": EXPECTED_STAGE147_SOURCE_HEAD, "stage147_run_id": EXPECTED_STAGE147_RUN_ID, "stage147_job_id": EXPECTED_STAGE147_JOB_ID, "stage147_artifact_id": EXPECTED_STAGE147_ARTIFACT_ID},
        "aggregate": {"stage149_record_ok": stage149_record_ok, "stage147_record_ok": stage147_record_ok, "parent_route_ok": parent_route_ok, "maximum_identity_or_provenance_error": metrics["maximum_identity_or_provenance_error"], "parent_channel_sign_transition": parent["channel_sign_transition"], "dominant_channel_sign_transition": dominant["channel_sign_transition"], "parent_stronger_outer_absolute_share": parent["stronger_outer_absolute_share"], "parent_outer_cancellation_fraction": parent["outer_cancellation_fraction"]},
        "metrics": metrics, "five_point_depth": depth.tolist(), "scientific_conclusion": conclusion,
        "negative_result_guard": "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. Stage 150 is an artifact-only channel sign-transition localization audit; outer-interval shares and cancellation fractions are diagnostics, not solver parameters. No physical, collision/source, floor, wall, reconstruction, transport, limiter, normalization, source-relaxation, or velocity-quadrature parameter is retuned; no measured transition location or scale is fed back into the solver, no solver endpoint or cross-Knudsen extension is advanced, and no benchmark or validation claim is permitted.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    np.savez(output_dir / "channel_sign_transition_localization.npz", five_point_depth=depth, dominant_outer_increments=np.array([dominant["left_outer_increment"], dominant["right_outer_increment"]]), parent_outer_increments=np.array([parent["left_outer_increment"], parent["right_outer_increment"]]), fine_components=np.array([dominant["fine_center"], parent["fine_center"]]), coarse_components=np.array([dominant["coarse_center"], parent["coarse_center"]]), outer_cancellation_fraction=np.array([dominant["outer_cancellation_fraction"], parent["outer_cancellation_fraction"]]), stronger_outer_absolute_share=np.array([dominant["stronger_outer_absolute_share"], parent["stronger_outer_absolute_share"]]))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 150 fixed channel sign-transition localization audit")
    parser.add_argument("--stage149-dir", type=Path, required=True); parser.add_argument("--stage149-record", type=Path, required=True)
    parser.add_argument("--stage147-dir", type=Path, required=True); parser.add_argument("--stage147-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    print(json.dumps(run_stage150(args.stage149_dir, args.stage149_record, args.stage147_dir, args.stage147_record, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
