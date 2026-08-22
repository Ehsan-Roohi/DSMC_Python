from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = 147
EXPECTED_STAGE146_SOURCE_HEAD = "e2fb7e3aff07ebe112cce83170e03f8d87f36198"
EXPECTED_STAGE146_RUN_ID = 32334751969
EXPECTED_STAGE146_JOB_ID = 96322009451
EXPECTED_STAGE146_ARTIFACT_ID = 9401787070
EXPECTED_STAGE146_ARTIFACT_SHA256 = "b9b04c21c990ae4cb937e320255db0188f728a4bc4c72dd12565df81526f43a7"
EXPECTED_STAGE146_SUMMARY_SHA256 = "7d3046422d4b7f44089e5089f9c4416dd1b876fded67334376a27d45af20052a"
EXPECTED_STAGE146_PAYLOAD_SHA256 = "c33bdfea137093ca370e9307658b9aaf0deca33d7811a011ed4911563e4ec63d"
EXPECTED_STAGE146_DECISION = "stage146_mixed_reinforcing_channel_curvature_stage147_dual_channel_neighborhood_audit"

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
MATERIAL_NEIGHBOR_RATIO_MIN = 0.25

NONFINITE = "stage147_nonfinite_blocker"
STAGE146_RECORD_BLOCKER = "stage147_stage146_record_blocker"
STAGE138_RECORD_BLOCKER = "stage147_stage138_record_blocker"
PARENT_ROUTE_BLOCKER = "stage147_parent_route_blocker"
PROVENANCE_BLOCKER = "stage147_center_provenance_blocker"
IDENTITY_BLOCKER = "stage147_channel_identity_closure_blocker"
PERSISTENT = "stage147_dual_channel_curvature_persists_stage148_persistent_curvature_shape_audit"
BILATERAL_REVERSAL = "stage147_material_bilateral_curvature_sign_reversal_stage148_sample_scale_curvature_alternation_audit"
MIXED = "stage147_mixed_or_one_sided_curvature_stage148_extended_neighborhood_curvature_audit"


def validate_stage147_design(
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
    material_neighbor_ratio_min=MATERIAL_NEIGHBOR_RATIO_MIN,
    solver_rerun=False,
    solver_endpoint_advanced=False,
    cross_knudsen_extension_permitted=False,
    benchmark_or_validation_claim_permitted=False,
    curvature_neighborhood_used_for_solver=False,
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
        "material_neighbor_ratio_min": MATERIAL_NEIGHBOR_RATIO_MIN,
        "solver_rerun": False,
        "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False,
        "benchmark_or_validation_claim_permitted": False,
        "curvature_neighborhood_used_for_solver": False,
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
            raise ValueError(f"Stage 147 frozen-design violation: {key}={got[key]!r}, expected {value!r}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_stage146_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 146
        and record.get("source_head") == EXPECTED_STAGE146_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE146_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE146_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE146_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE146_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE146_SUMMARY_SHA256
        and record.get("trough_provenance_sha256") == EXPECTED_STAGE146_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE146_DECISION
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


def dual_channel_neighborhood_metrics(
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
        raise ValueError("Stage 147 requires one-dimensional channel profiles")
    if not (x.shape == dominant.shape == parent.shape == complement.shape) or x.size < 5:
        raise ValueError("Stage 147 channel profiles must have equal length >= 5")
    if not all(np.isfinite(a).all() for a in (x, dominant, parent, complement)) or not np.isfinite(trough_depth):
        raise ValueError("Stage 147 requires finite profiles and trough depth")
    if not np.all(np.diff(x) > 0.0):
        raise ValueError("Stage 147 depth samples must be strictly increasing")

    center = int(np.argmin(np.abs(x - trough_depth)))
    depth_match_error = float(abs(x[center] - trough_depth))
    if center < 2 or center > x.size - 3:
        raise ValueError("Stage 147 trough depth must have two samples on each side")

    full_identity_closure = float(np.max(np.abs((parent - dominant) - complement)))
    idx = np.array([center - 1, center, center + 1], dtype=int)
    dominant_projected = []
    parent_projected = []
    total_deficit = []
    closure = []
    for i in idx:
        dproj = -(_local_secant(x, dominant, int(i)) - float(dominant[i]))
        pproj = _local_secant(x, parent, int(i)) - float(parent[i])
        cdef = _local_secant(x, complement, int(i)) - float(complement[i])
        dominant_projected.append(dproj)
        parent_projected.append(pproj)
        total_deficit.append(cdef)
        closure.append(abs((dproj + pproj) - cdef))

    dominant_projected = np.asarray(dominant_projected, dtype=float)
    parent_projected = np.asarray(parent_projected, dtype=float)
    total_deficit = np.asarray(total_deficit, dtype=float)
    closure = np.asarray(closure, dtype=float)
    center_total = float(total_deficit[1])
    center_dominant = float(dominant_projected[1])
    center_parent = float(parent_projected[1])

    neighbor_total = total_deficit[[0, 2]]
    neighbor_dominant = dominant_projected[[0, 2]]
    neighbor_parent = parent_projected[[0, 2]]
    denom = max(abs(center_total), np.finfo(float).tiny)
    neighbor_ratio = np.abs(neighbor_total) / denom
    material_opposite = (
        (neighbor_total * center_total < 0.0)
        & (neighbor_dominant * center_dominant < 0.0)
        & (neighbor_parent * center_parent < 0.0)
        & (neighbor_ratio >= MATERIAL_NEIGHBOR_RATIO_MIN)
    )
    reinforcing = (dominant_projected > 0.0) & (parent_projected > 0.0) & (total_deficit > 0.0)

    shares = np.zeros((3, 2), dtype=float)
    for j in range(3):
        den = abs(dominant_projected[j]) + abs(parent_projected[j])
        if den > 0.0:
            shares[j, 0] = abs(parent_projected[j]) / den
            shares[j, 1] = abs(dominant_projected[j]) / den

    sign_seq = np.sign(total_deficit).astype(int)
    return {
        "trough_profile_index": center,
        "trough_depth": float(x[center]),
        "depth_match_error": depth_match_error,
        "neighborhood_indices": idx.tolist(),
        "neighborhood_depths": x[idx].tolist(),
        "dominant_projected_curvature": dominant_projected.tolist(),
        "parent_projected_curvature": parent_projected.tolist(),
        "complement_secant_deficit": total_deficit.tolist(),
        "curvature_sign_sequence": sign_seq.tolist(),
        "neighbor_absolute_ratio_to_center": neighbor_ratio.tolist(),
        "minimum_neighbor_absolute_ratio_to_center": float(np.min(neighbor_ratio)),
        "neighbor_reinforcing_count": int(np.count_nonzero(reinforcing[[0, 2]])),
        "neighbor_material_bilateral_reversal_count": int(np.count_nonzero(material_opposite)),
        "bilateral_material_channel_sign_reversal": bool(np.all(material_opposite)),
        "center_channels_reinforce": bool(reinforcing[1]),
        "channel_absolute_shares": shares.tolist(),
        "maximum_channel_identity_or_decomposition_closure": max(full_identity_closure, float(np.max(closure))),
    }


def classify_dual_channel_neighborhood(
    *,
    metrics: dict,
    center_metric_match_error: float = 0.0,
    local_profile_match_error: float = 0.0,
    stage146_record_ok: bool = True,
    stage138_record_ok: bool = True,
    parent_route_ok: bool = True,
    finite: bool = True,
) -> str:
    numeric = [
        metrics.get("depth_match_error", np.nan),
        metrics.get("minimum_neighbor_absolute_ratio_to_center", np.nan),
        metrics.get("maximum_channel_identity_or_decomposition_closure", np.nan),
        center_metric_match_error,
        local_profile_match_error,
    ]
    if not finite or not np.isfinite(numeric).all():
        return NONFINITE
    if not stage146_record_ok:
        return STAGE146_RECORD_BLOCKER
    if not stage138_record_ok:
        return STAGE138_RECORD_BLOCKER
    if not parent_route_ok or not bool(metrics.get("center_channels_reinforce", False)):
        return PARENT_ROUTE_BLOCKER
    if max(float(metrics["depth_match_error"]), center_metric_match_error, local_profile_match_error) > PROVENANCE_MATCH_MAX:
        return PROVENANCE_BLOCKER
    if float(metrics["maximum_channel_identity_or_decomposition_closure"]) > IDENTITY_CLOSURE_MAX:
        return IDENTITY_BLOCKER
    if int(metrics["neighbor_reinforcing_count"]) == 2:
        return PERSISTENT
    if bool(metrics["bilateral_material_channel_sign_reversal"]):
        return BILATERAL_REVERSAL
    return MIXED


def run_stage147(
    stage146_dir: Path,
    stage146_record: Path,
    stage138_dir: Path,
    stage138_record: Path,
    output_dir: Path,
) -> dict:
    validate_stage147_design()
    stage146_summary = _load_json(stage146_dir / "summary.json")
    stage146_record_data = _load_json(stage146_record)
    stage138_summary = _load_json(stage138_dir / "summary.json")
    stage138_record_data = _load_json(stage138_record)

    stage146_record_ok = _check_stage146_record(stage146_record_data)
    stage138_record_ok = _check_stage138_record(stage138_record_data)
    parent_route_ok = bool(
        stage146_summary.get("stage") == 146
        and stage146_summary.get("decision") == EXPECTED_STAGE146_DECISION
        and stage138_summary.get("stage") == 138
        and stage138_summary.get("decision") == EXPECTED_STAGE138_DECISION
    )

    with np.load(stage138_dir / "channel_rate_origin.npz") as data:
        depth = np.asarray(data["right_depth"], dtype=float)
        dominant = np.asarray(data["dominant_signed"], dtype=float)
        parent = np.asarray(data["parent_signed"], dtype=float)
        complement = np.asarray(data["complement_signed"], dtype=float)

    with np.load(stage146_dir / "trough_provenance.npz") as data:
        local_depth = np.asarray(data["local_depth"], dtype=float)
        local_dominant = np.asarray(data["local_dominant_signed"], dtype=float)
        local_parent = np.asarray(data["local_parent_signed"], dtype=float)
        local_complement = np.asarray(data["local_complement_signed"], dtype=float)
        inherited_contrib = np.asarray(data["projected_contributions"], dtype=float)

    trough_depth = float(stage146_summary["metrics"]["trough_depth"])
    metrics = dual_channel_neighborhood_metrics(depth, dominant, parent, complement, trough_depth)
    center = int(metrics["trough_profile_index"])
    sl3 = slice(center - 1, center + 2)
    local_profile_match_error = float(max(
        np.max(np.abs(depth[sl3] - local_depth)),
        np.max(np.abs(dominant[sl3] - local_dominant)),
        np.max(np.abs(parent[sl3] - local_parent)),
        np.max(np.abs(complement[sl3] - local_complement)),
    ))
    center_parent = float(metrics["parent_projected_curvature"][1])
    center_dominant = float(metrics["dominant_projected_curvature"][1])
    center_metric_match_error = float(max(
        abs(center_parent - float(stage146_summary["metrics"]["parent_projected_trough_contribution"])),
        abs(center_dominant - float(stage146_summary["metrics"]["dominant_projected_trough_contribution"])),
        np.max(np.abs(inherited_contrib - np.asarray([center_parent, center_dominant], dtype=float))),
    ))

    finite = bool(stage146_summary.get("finite", False) and stage138_summary.get("finite", True))
    decision = classify_dual_channel_neighborhood(
        metrics=metrics,
        center_metric_match_error=center_metric_match_error,
        local_profile_match_error=local_profile_match_error,
        stage146_record_ok=stage146_record_ok,
        stage138_record_ok=stage138_record_ok,
        parent_route_ok=parent_route_ok,
        finite=finite,
    )

    cfg = dict(stage146_summary.get("configuration", {}))
    cfg.update({
        "material_neighbor_ratio_min": MATERIAL_NEIGHBOR_RATIO_MIN,
        "provenance_match_max": PROVENANCE_MATCH_MAX,
        "identity_closure_max": IDENTITY_CLOSURE_MAX,
        "curvature_neighborhood_used_for_solver": False,
        "solver_rerun": False,
        "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False,
        "benchmark_or_validation_claim_permitted": False,
    })
    for key in list(cfg):
        if key.endswith("_retuning"):
            cfg[key] = False

    if decision == PERSISTENT:
        conclusion = (
            "The Stage-146 mixed reinforcing curvature persists at both immediate neighboring depths with the same signed channel cooperation. The next artifact-only step is to characterize the persistent curvature shape; no curvature scale is a solver coefficient."
        )
    elif decision == BILATERAL_REVERSAL:
        conclusion = (
            "The Stage-146 mixed reinforcing curvature is confined to the sampled trough depth. At both immediate neighboring depths, both projected channel curvatures reverse sign together, and the smaller opposite-sign neighbor remains at least the fixed 25% materiality fraction of the center deficit. The resulting [-,+,-] sampled curvature sequence is a grid-sample-scale alternation signature, not evidence of a physical sub-lobe or a tunable solver mode. The next justified artifact-only step is a sample-scale curvature-alternation audit."
        )
    elif decision == MIXED:
        conclusion = (
            "The Stage-146 mixed reinforcing curvature does not persist cleanly to both neighbors and does not satisfy the fixed bilateral material sign-reversal criterion. The next artifact-only step is an extended-neighborhood curvature audit; no solver retuning is justified."
        )
    else:
        conclusion = "Stage 147 is blocked by provenance, route, finite-data, or identity-closure guards; no scientific routing is claimed."

    summary = {
        "stage": STAGE,
        "finite": bool(finite),
        "decision": decision,
        "configuration": cfg,
        "parents": {
            "stage146_run_id": EXPECTED_STAGE146_RUN_ID,
            "stage146_job_id": EXPECTED_STAGE146_JOB_ID,
            "stage146_artifact_id": EXPECTED_STAGE146_ARTIFACT_ID,
            "stage146_source_head": EXPECTED_STAGE146_SOURCE_HEAD,
            "stage138_run_id": EXPECTED_STAGE138_RUN_ID,
            "stage138_job_id": EXPECTED_STAGE138_JOB_ID,
            "stage138_artifact_id": EXPECTED_STAGE138_ARTIFACT_ID,
            "stage138_source_head": EXPECTED_STAGE138_SOURCE_HEAD,
        },
        "aggregate": {
            "stage146_record_ok": bool(stage146_record_ok),
            "stage138_record_ok": bool(stage138_record_ok),
            "parent_route_ok": bool(parent_route_ok),
            "center_metric_match_error": center_metric_match_error,
            "local_profile_match_error": local_profile_match_error,
            "maximum_identity_or_decomposition_closure": float(metrics["maximum_channel_identity_or_decomposition_closure"]),
            "bilateral_material_channel_sign_reversal": bool(metrics["bilateral_material_channel_sign_reversal"]),
            "neighbor_reinforcing_count": int(metrics["neighbor_reinforcing_count"]),
        },
        "metrics": metrics,
        "scientific_conclusion": conclusion,
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. Stage 147 is an artifact-only dual-channel neighborhood audit; discrete secant curvatures, sign sequences, and neighbor ratios are diagnostics, not solver parameters. No physical, collision/source, floor, wall, reconstruction, transport, limiter, normalization, source-relaxation, or velocity-quadrature parameter is retuned; no curvature magnitude or alternation scale is fed back into the solver, no solver endpoint or cross-Knudsen extension is advanced, and no benchmark or validation claim is permitted."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    center = int(metrics["trough_profile_index"])
    sl5 = slice(center - 2, center + 3)
    np.savez_compressed(
        output_dir / "dual_channel_neighborhood.npz",
        five_point_depth=depth[sl5],
        five_point_dominant_signed=dominant[sl5],
        five_point_parent_signed=parent[sl5],
        five_point_complement_signed=complement[sl5],
        neighborhood_depth=np.asarray(metrics["neighborhood_depths"], dtype=float),
        dominant_projected_curvature=np.asarray(metrics["dominant_projected_curvature"], dtype=float),
        parent_projected_curvature=np.asarray(metrics["parent_projected_curvature"], dtype=float),
        complement_secant_deficit=np.asarray(metrics["complement_secant_deficit"], dtype=float),
        neighbor_absolute_ratio_to_center=np.asarray(metrics["neighbor_absolute_ratio_to_center"], dtype=float),
        channel_absolute_shares=np.asarray(metrics["channel_absolute_shares"], dtype=float),
    )
    print(json.dumps({"stage": STAGE, "decision": decision, "aggregate": summary["aggregate"], "metrics": metrics}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 147 fixed dual-channel neighborhood audit")
    parser.add_argument("--stage146-dir", type=Path, required=True)
    parser.add_argument("--stage146-record", type=Path, required=True)
    parser.add_argument("--stage138-dir", type=Path, required=True)
    parser.add_argument("--stage138-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_stage147(args.stage146_dir, args.stage146_record, args.stage138_dir, args.stage138_record, args.output_dir)


if __name__ == "__main__":
    main()
