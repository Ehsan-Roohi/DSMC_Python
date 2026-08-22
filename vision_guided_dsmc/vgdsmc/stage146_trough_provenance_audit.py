from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = 146
EXPECTED_STAGE145_SOURCE_HEAD = "be333da19f0ee4c6da3845829a6ff0ed966989aa"
EXPECTED_STAGE145_RUN_ID = 32333756029
EXPECTED_STAGE145_JOB_ID = 96319199171
EXPECTED_STAGE145_ARTIFACT_ID = 9393984539
EXPECTED_STAGE145_ARTIFACT_SHA256 = "ad644a48d276817a0ba8757a745f66ac220ad19beadc331f48e9c66e8acb69fd"
EXPECTED_STAGE145_SUMMARY_SHA256 = "0654c6929d1a1c65aba84f77e46774162b77c2320f82611f203d8dca9f3f0a7a"
EXPECTED_STAGE145_PAYLOAD_SHA256 = "2a8f0b1a8dd10ebb435bc0e4b50ca9985645743d1b8d4b1191922a6022969cee"
EXPECTED_STAGE145_DECISION = "stage145_trough_isolated_to_single_sample_stage146_trough_provenance_audit"

EXPECTED_STAGE138_SOURCE_HEAD = "e513f249dd9ceb43556ab07f9d0378a791eeaa8a"
EXPECTED_STAGE138_RUN_ID = 32176056342
EXPECTED_STAGE138_JOB_ID = 95838163254
EXPECTED_STAGE138_ARTIFACT_ID = 9347568376
EXPECTED_STAGE138_ARTIFACT_SHA256 = "b8d435a64aab18c208ab3531aa52a8bc1e615d1b82faa26b0743ee3dc24785b4"
EXPECTED_STAGE138_SUMMARY_SHA256 = "61a10326c2dce8e5a47dd57333ea9acd5776f0b531dcce48204129a330860c3f"
EXPECTED_STAGE138_PAYLOAD_SHA256 = "44ddc0cc8bd3c1bc18c4cb60b953c985bdcb93ebe0c3446b5216a8cf000e03c8"
EXPECTED_STAGE138_DECISION = "stage138_depth_varying_complement_cancellation_stage139_complement_transition_geometry_audit"

PROVENANCE_MATCH_MAX = 1.0e-12
IDENTITY_CLOSURE_MAX = 1.0e-12
SINGLE_CHANNEL_DOMINANCE_MIN = 0.75

NONFINITE = "stage146_nonfinite_blocker"
STAGE145_RECORD_BLOCKER = "stage146_stage145_record_blocker"
STAGE138_RECORD_BLOCKER = "stage146_stage138_record_blocker"
PARENT_ROUTE_BLOCKER = "stage146_parent_route_blocker"
PROVENANCE_BLOCKER = "stage146_inherited_sample_provenance_blocker"
IDENTITY_BLOCKER = "stage146_channel_identity_closure_blocker"
TROUGH_BLOCKER = "stage146_parent_trough_deficit_blocker"
SINGLE_CHANNEL = "stage146_single_channel_curvature_dominance_stage147_dominant_channel_neighborhood_audit"
MIXED_REINFORCING = "stage146_mixed_reinforcing_channel_curvature_stage147_dual_channel_neighborhood_audit"
OPPOSED = "stage146_opposed_channel_curvature_cancellation_stage147_channel_cancellation_neighborhood_audit"


def validate_stage146_design(
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
    single_channel_dominance_min=SINGLE_CHANNEL_DOMINANCE_MIN,
    solver_rerun=False,
    solver_endpoint_advanced=False,
    cross_knudsen_extension_permitted=False,
    benchmark_or_validation_claim_permitted=False,
    trough_provenance_used_for_solver=False,
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
        "single_channel_dominance_min": SINGLE_CHANNEL_DOMINANCE_MIN,
        "solver_rerun": False,
        "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False,
        "benchmark_or_validation_claim_permitted": False,
        "trough_provenance_used_for_solver": False,
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
            raise ValueError(f"Stage 146 frozen-design violation: {key}={got[key]!r}, expected {value!r}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_stage145_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 145
        and record.get("source_head") == EXPECTED_STAGE145_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE145_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE145_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE145_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE145_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE145_SUMMARY_SHA256
        and record.get("trough_persistence_sha256") == EXPECTED_STAGE145_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE145_DECISION
        and record.get("workflow_status") == "completed"
        and record.get("workflow_conclusion") == "success"
    )


def _check_stage138_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 138
        and record.get("source_head") == EXPECTED_STAGE138_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE138_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE138_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE138_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE138_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE138_SUMMARY_SHA256
        and record.get("channel_rate_origin_sha256") == EXPECTED_STAGE138_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE138_DECISION
        and record.get("workflow_status") == "completed"
        and record.get("workflow_conclusion") == "success"
    )


def _local_secant(x: np.ndarray, y: np.ndarray, index: int) -> float:
    xl, xc, xr = x[index - 1], x[index], x[index + 1]
    yl, yr = y[index - 1], y[index + 1]
    return float(yl + (yr - yl) * (xc - xl) / (xr - xl))


def trough_provenance_metrics(
    depth: np.ndarray,
    dominant_signed: np.ndarray,
    parent_signed: np.ndarray,
    complement_signed: np.ndarray,
    trough_depth: float,
) -> dict:
    x = np.asarray(depth, dtype=float)
    dominant = np.asarray(dominant_signed, dtype=float)
    parent = np.asarray(parent_signed, dtype=float)
    complement = np.asarray(complement_signed, dtype=float)
    if any(a.ndim != 1 for a in (x, dominant, parent, complement)):
        raise ValueError("Stage 146 requires one-dimensional channel profiles")
    if not (x.shape == dominant.shape == parent.shape == complement.shape) or x.size < 3:
        raise ValueError("Stage 146 channel profiles must have equal length >= 3")
    if not all(np.isfinite(a).all() for a in (x, dominant, parent, complement)) or not np.isfinite(trough_depth):
        raise ValueError("Stage 146 requires finite profiles and trough depth")
    if not np.all(np.diff(x) > 0.0):
        raise ValueError("Stage 146 depth samples must be strictly increasing")

    index = int(np.argmin(np.abs(x - trough_depth)))
    depth_match_error = float(abs(x[index] - trough_depth))
    if index <= 0 or index >= x.size - 1:
        raise ValueError("Stage 146 trough depth must have immediate neighbors")

    identity_closure = float(np.max(np.abs((parent - dominant) - complement)))
    dominant_secant = _local_secant(x, dominant, index)
    parent_secant = _local_secant(x, parent, index)
    complement_secant = _local_secant(x, complement, index)

    dominant_curvature = float(dominant_secant - dominant[index])
    parent_curvature = float(parent_secant - parent[index])
    complement_deficit = float(complement_secant - complement[index])
    dominant_projected = float(-dominant_curvature)
    parent_projected = float(parent_curvature)
    decomposition_closure = float(abs((parent_projected + dominant_projected) - complement_deficit))

    abs_total = float(abs(parent_projected) + abs(dominant_projected))
    if abs_total == 0.0:
        parent_share = dominant_share = 0.0
    else:
        parent_share = float(abs(parent_projected) / abs_total)
        dominant_share = float(abs(dominant_projected) / abs_total)

    return {
        "trough_profile_index": index,
        "trough_depth": float(x[index]),
        "depth_match_error": depth_match_error,
        "dominant_value": float(dominant[index]),
        "parent_value": float(parent[index]),
        "complement_value": float(complement[index]),
        "dominant_local_secant": dominant_secant,
        "parent_local_secant": parent_secant,
        "complement_local_secant": complement_secant,
        "dominant_secant_minus_value": dominant_curvature,
        "parent_secant_minus_value": parent_curvature,
        "complement_trough_deficit": complement_deficit,
        "dominant_projected_trough_contribution": dominant_projected,
        "parent_projected_trough_contribution": parent_projected,
        "dominant_absolute_contribution_share": dominant_share,
        "parent_absolute_contribution_share": parent_share,
        "maximum_single_channel_absolute_share": max(dominant_share, parent_share),
        "projected_contributions_reinforce": bool(dominant_projected > 0.0 and parent_projected > 0.0),
        "channel_identity_closure": identity_closure,
        "trough_deficit_decomposition_closure": decomposition_closure,
    }


def classify_trough_provenance(
    *,
    metrics: dict,
    inherited_value_match_error: float = 0.0,
    inherited_secant_match_error: float = 0.0,
    stage145_record_ok: bool = True,
    stage138_record_ok: bool = True,
    parent_route_ok: bool = True,
    finite: bool = True,
) -> str:
    numeric = [
        metrics.get("channel_identity_closure", np.nan),
        metrics.get("trough_deficit_decomposition_closure", np.nan),
        metrics.get("complement_trough_deficit", np.nan),
        metrics.get("maximum_single_channel_absolute_share", np.nan),
        inherited_value_match_error,
        inherited_secant_match_error,
    ]
    if not finite or not np.isfinite(numeric).all():
        return NONFINITE
    if not stage145_record_ok:
        return STAGE145_RECORD_BLOCKER
    if not stage138_record_ok:
        return STAGE138_RECORD_BLOCKER
    if not parent_route_ok:
        return PARENT_ROUTE_BLOCKER
    if max(
        float(metrics["depth_match_error"]),
        inherited_value_match_error,
        inherited_secant_match_error,
    ) > PROVENANCE_MATCH_MAX:
        return PROVENANCE_BLOCKER
    if max(
        float(metrics["channel_identity_closure"]),
        float(metrics["trough_deficit_decomposition_closure"]),
    ) > IDENTITY_CLOSURE_MAX:
        return IDENTITY_BLOCKER
    if float(metrics["complement_trough_deficit"]) <= 0.0:
        return TROUGH_BLOCKER
    if bool(metrics["projected_contributions_reinforce"]):
        if float(metrics["maximum_single_channel_absolute_share"]) >= SINGLE_CHANNEL_DOMINANCE_MIN:
            return SINGLE_CHANNEL
        return MIXED_REINFORCING
    return OPPOSED


def run_stage146(
    stage145_dir: Path,
    stage145_record: Path,
    stage138_dir: Path,
    stage138_record: Path,
    output_dir: Path,
) -> dict:
    validate_stage146_design()
    stage145_summary = _load_json(stage145_dir / "summary.json")
    stage145_record_data = _load_json(stage145_record)
    stage138_summary = _load_json(stage138_dir / "summary.json")
    stage138_record_data = _load_json(stage138_record)

    stage145_record_ok = _check_stage145_record(stage145_record_data)
    stage138_record_ok = _check_stage138_record(stage138_record_data)
    parent_route_ok = bool(
        stage145_summary.get("stage") == 145
        and stage145_summary.get("decision") == EXPECTED_STAGE145_DECISION
        and stage138_summary.get("stage") == 138
        and stage138_summary.get("decision") == EXPECTED_STAGE138_DECISION
    )

    with np.load(stage145_dir / "trough_persistence.npz") as data:
        inherited_depth = float(np.asarray(data["trough_depth"], dtype=float)[0])
        inherited_value = float(np.asarray(data["trough_value"], dtype=float)[0])
        inherited_secant = float(np.asarray(data["local_neighbor_secant_value"], dtype=float)[0])

    with np.load(stage138_dir / "channel_rate_origin.npz") as data:
        depth = np.asarray(data["right_depth"], dtype=float)
        dominant = np.asarray(data["dominant_signed"], dtype=float)
        parent = np.asarray(data["parent_signed"], dtype=float)
        complement = np.asarray(data["complement_signed"], dtype=float)

    finite = bool(
        stage145_summary.get("finite", False)
        and stage138_summary.get("finite", True)
        and np.isfinite([inherited_depth, inherited_value, inherited_secant]).all()
    )
    metrics = trough_provenance_metrics(depth, dominant, parent, complement, inherited_depth)
    inherited_value_match_error = float(abs(metrics["complement_value"] - inherited_value))
    inherited_secant_match_error = float(abs(metrics["complement_local_secant"] - inherited_secant))

    decision = classify_trough_provenance(
        metrics=metrics,
        inherited_value_match_error=inherited_value_match_error,
        inherited_secant_match_error=inherited_secant_match_error,
        stage145_record_ok=stage145_record_ok,
        stage138_record_ok=stage138_record_ok,
        parent_route_ok=parent_route_ok,
        finite=finite,
    )

    cfg = dict(stage145_summary.get("configuration", {}))
    cfg.update({
        "single_channel_dominance_min": SINGLE_CHANNEL_DOMINANCE_MIN,
        "provenance_match_max": PROVENANCE_MATCH_MAX,
        "identity_closure_max": IDENTITY_CLOSURE_MAX,
        "trough_provenance_used_for_solver": False,
        "solver_rerun": False,
        "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False,
        "benchmark_or_validation_claim_permitted": False,
    })
    for key in list(cfg):
        if key.endswith("_retuning"):
            cfg[key] = False

    if decision == SINGLE_CHANNEL:
        conclusion = (
            "The isolated Stage-145 complement trough is reproduced exactly from the inherited Stage-138 signed channels, and at least 75% of its local secant deficit is attributable to one channel under the fixed linear decomposition. The next artifact-only step is to test whether that channel curvature persists to neighboring depths; the contribution is not a solver coefficient."
        )
    elif decision == MIXED_REINFORCING:
        conclusion = (
            "The isolated Stage-145 complement trough is reproduced exactly from the inherited Stage-138 signed channels, but neither channel reaches the fixed 75% single-channel provenance threshold. Parent and sign-reversed dominant local curvatures reinforce the trough deficit. The next artifact-only step is a dual-channel neighborhood audit to determine whether this mixed curvature is itself isolated or spatially persistent."
        )
    elif decision == OPPOSED:
        conclusion = (
            "The isolated Stage-145 complement trough is reproduced from the inherited Stage-138 channels, but the projected channel curvatures oppose one another. The next artifact-only step is a local cancellation-geometry audit; no cancellation factor is fed back into the solver."
        )
    else:
        conclusion = "Stage 146 is blocked by provenance, route, finite-data, identity-closure, or trough-deficit guards; no scientific routing is claimed."

    summary = {
        "stage": STAGE,
        "finite": bool(finite),
        "decision": decision,
        "configuration": cfg,
        "parents": {
            "stage145_run_id": EXPECTED_STAGE145_RUN_ID,
            "stage145_job_id": EXPECTED_STAGE145_JOB_ID,
            "stage145_artifact_id": EXPECTED_STAGE145_ARTIFACT_ID,
            "stage145_source_head": EXPECTED_STAGE145_SOURCE_HEAD,
            "stage138_run_id": EXPECTED_STAGE138_RUN_ID,
            "stage138_job_id": EXPECTED_STAGE138_JOB_ID,
            "stage138_artifact_id": EXPECTED_STAGE138_ARTIFACT_ID,
            "stage138_source_head": EXPECTED_STAGE138_SOURCE_HEAD,
        },
        "aggregate": {
            "stage145_record_ok": bool(stage145_record_ok),
            "stage138_record_ok": bool(stage138_record_ok),
            "parent_route_ok": bool(parent_route_ok),
            "inherited_value_match_error": inherited_value_match_error,
            "inherited_secant_match_error": inherited_secant_match_error,
            "maximum_identity_or_decomposition_closure": max(
                float(metrics["channel_identity_closure"]),
                float(metrics["trough_deficit_decomposition_closure"]),
            ),
            "single_channel_dominance_pass": bool(
                float(metrics["maximum_single_channel_absolute_share"]) >= SINGLE_CHANNEL_DOMINANCE_MIN
            ),
            "reinforcing_channel_curvature": bool(metrics["projected_contributions_reinforce"]),
        },
        "metrics": metrics,
        "scientific_conclusion": conclusion,
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. Stage 146 is an artifact-only inherited-channel trough-provenance audit; local secant curvatures and contribution shares are diagnostics, not solver parameters. No physical, collision/source, floor, wall, reconstruction, transport, limiter, normalization, source-relaxation, or velocity-quadrature parameter is retuned; no trough amplitude/location or channel-curvature contribution is fed back into the solver, no solver endpoint or cross-Knudsen extension is advanced, and no benchmark or validation claim is permitted."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    i = int(metrics["trough_profile_index"])
    sl = slice(i - 1, i + 2)
    np.savez_compressed(
        output_dir / "trough_provenance.npz",
        local_depth=depth[sl],
        local_dominant_signed=dominant[sl],
        local_parent_signed=parent[sl],
        local_complement_signed=complement[sl],
        projected_contributions=np.asarray([
            metrics["parent_projected_trough_contribution"],
            metrics["dominant_projected_trough_contribution"],
        ], dtype=float),
        absolute_contribution_shares=np.asarray([
            metrics["parent_absolute_contribution_share"],
            metrics["dominant_absolute_contribution_share"],
        ], dtype=float),
    )
    print(json.dumps({"stage": STAGE, "decision": decision, "aggregate": summary["aggregate"], "metrics": metrics}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 146 fixed inherited-channel trough-provenance audit")
    parser.add_argument("--stage145-dir", type=Path, required=True)
    parser.add_argument("--stage145-record", type=Path, required=True)
    parser.add_argument("--stage138-dir", type=Path, required=True)
    parser.add_argument("--stage138-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_stage146(args.stage145_dir, args.stage145_record, args.stage138_dir, args.stage138_record, args.output_dir)


if __name__ == "__main__":
    main()
