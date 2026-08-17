from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STAGE130_RUN_ID = 31966517708
STAGE130_JOB_ID = 95212392074
STAGE130_ARTIFACT_ID = 9273190595
STAGE130_ARTIFACT_SHA256 = "a7d88febfa695b537a9622e463a9563e4ec089bfee3674a1de7b7cfbbcdba01b"
STAGE130_SUMMARY_SHA256 = "0c918ec583a84176e65a2de6db60baf3406ec9431cf6d4a6bb30a8863dcf50f3"
STAGE130_PAYLOAD_SHA256 = "d8019c6fef615a971afb49211b225b6395a1b368c8df7de7ca56bdc9cc9472f9"
STAGE130_SOURCE_HEAD = "02a5f93f78ac693d56f0c910f83a88790dfa03b1"
STAGE130_DECISION = "stage130_wall_specific_sector_carriage_stage131_cross_wall_sector_interaction_audit"

GRID = (64, 64)
INTERIOR_GRID = (56, 56)
DEPTH_COUNT = 28
WITNESS_NODE = 9
PAIR_SECTORS = (5, 6)
KNUDSEN = 10.0
COLD_HOT_RATIO = 0.1
RULE = (40, 96)
RADIAL_SCALE = 2.0
LIMITER = "minmod"
BOUNDARY_SLOPE = "zero"
SOURCE_RELAXATION = 1.0
TOLERANCE = 2.0e-5
CORRECTION_FLOOR = 0.05
DIAGNOSTIC_STEPS = 25

PARENT_CLOSURE_TOLERANCE = 1.0e-12
MIRROR_PROFILE_COSINE_MIN = 0.95
MIRROR_TRANSITION_COSINE_MIN = 0.90
MIRROR_PROFILE_RELATIVE_RESIDUAL_MAX = 0.15
MIRROR_GAIN_OVER_DIRECT_MIN = 0.15

MIRROR = "stage131_mirror_sector_exchange_collapse_stage132_crossing_phase_transition_width_audit"
DIRECT = "stage131_direct_sector_collapse_stage132_common_sector_depth_profile_audit"
COMPLEX = "stage131_no_simple_cross_wall_sector_mapping_stage132_local_angular_phase_audit"
NONFINITE = "stage131_nonfinite_cross_wall_sector_blocker_without_retuning"
CLOSURE_BLOCKER = "stage131_parent_or_sector_payload_closure_blocker_without_retuning"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_stage131_design(**overrides: object) -> None:
    frozen = {
        "stage130_run_id": STAGE130_RUN_ID,
        "stage130_job_id": STAGE130_JOB_ID,
        "stage130_artifact_id": STAGE130_ARTIFACT_ID,
        "grid": GRID,
        "interior_grid": INTERIOR_GRID,
        "depth_count": DEPTH_COUNT,
        "witness_node": WITNESS_NODE,
        "pair_sectors": PAIR_SECTORS,
        "kn0": KNUDSEN,
        "cold_hot_ratio": COLD_HOT_RATIO,
        "rule": RULE,
        "radial_scale": RADIAL_SCALE,
        "limiter": LIMITER,
        "boundary_slope": BOUNDARY_SLOPE,
        "source_relaxation": SOURCE_RELAXATION,
        "tolerance": TOLERANCE,
        "correction_floor": CORRECTION_FLOOR,
        "diagnostic_steps": DIAGNOSTIC_STEPS,
        "parent_closure_tolerance": PARENT_CLOSURE_TOLERANCE,
        "mirror_profile_cosine_min": MIRROR_PROFILE_COSINE_MIN,
        "mirror_transition_cosine_min": MIRROR_TRANSITION_COSINE_MIN,
        "mirror_profile_relative_residual_max": MIRROR_PROFILE_RELATIVE_RESIDUAL_MAX,
        "mirror_gain_over_direct_min": MIRROR_GAIN_OVER_DIRECT_MIN,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 131 is frozen to the exact completed Stage-130 node-9 sectors-5+6 artifact and its two "
            "crossing depths. It may compare direct versus mirrored sector labels only; it may not retune "
            "physics, collision/source treatment, walls, reconstruction, transport, limiter, floors, "
            "normalization, source relaxation, velocity quadrature, failed MUSCL parameters, or guards."
        )


def _verify_stage130_record(record: dict[str, object]) -> None:
    checks = (
        record.get("stage") == 130,
        record.get("source_head") == STAGE130_SOURCE_HEAD,
        record.get("workflow_status") == "completed",
        record.get("workflow_conclusion") == "success",
        record.get("workflow_run_id") == STAGE130_RUN_ID,
        record.get("workflow_job_id") == STAGE130_JOB_ID,
        record.get("artifact_id") == STAGE130_ARTIFACT_ID,
        record.get("artifact_sha256") == STAGE130_ARTIFACT_SHA256,
        record.get("summary_sha256") == STAGE130_SUMMARY_SHA256,
        record.get("fixed_sector_continuity_sha256") == STAGE130_PAYLOAD_SHA256,
        record.get("decision") == STAGE130_DECISION,
        record.get("tests", {}).get("passed") == 8,
        record.get("tests", {}).get("failed") == 0,
    )
    if not all(checks):
        raise ValueError("Committed Stage-130 provenance does not authorize Stage 131")


def _load_stage130(root: str | Path, record_path: str | Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    root = Path(root)
    expected = {
        "summary.json": STAGE130_SUMMARY_SHA256,
        "fixed_sector_continuity.npz": STAGE130_PAYLOAD_SHA256,
    }
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Stage-130 checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    _verify_stage130_record(record)
    if summary.get("stage") != 130 or summary.get("finite") is not True or summary.get("decision") != STAGE130_DECISION:
        raise ValueError("Stage-130 artifact does not authorize Stage 131")
    cfg = summary.get("configuration", {})
    required = {
        "grid": list(GRID),
        "interior_grid": list(INTERIOR_GRID),
        "kn0": KNUDSEN,
        "cold_hot_ratio": COLD_HOT_RATIO,
        "rule": list(RULE),
        "radial_scale": RADIAL_SCALE,
        "limiter": LIMITER,
        "boundary_slope": BOUNDARY_SLOPE,
        "source_relaxation": SOURCE_RELAXATION,
        "correction_floor": CORRECTION_FLOOR,
        "witness_node": WITNESS_NODE,
        "pair_sectors": list(PAIR_SECTORS),
        "solver_rerun": False,
        "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False,
    }
    if any(cfg.get(key) != value for key, value in required.items()):
        raise ValueError("Stage-130 frozen design mismatch")
    if any(value is not False for key, value in cfg.items() if key.endswith("_retuning")):
        raise ValueError("Stage-130 artifact contains forbidden retuning")
    with np.load(root / "fixed_sector_continuity.npz") as data:
        needed = {
            "depth",
            "wall0_parent_asymmetry",
            "wall1_parent_asymmetry",
            "wall0_sector_contributions",
            "wall1_sector_contributions",
            "wall0_parent_same_sign_l1",
            "wall1_parent_same_sign_l1",
            "sector_transition_deltas",
            "parent_transition_deltas",
        }
        if not needed.issubset(data.files):
            raise ValueError("Stage-130 payload is incomplete")
        arrays = {name: np.asarray(data[name], dtype=np.float64).copy() for name in needed}
    if arrays["depth"].shape != (DEPTH_COUNT,):
        raise ValueError("Stage-130 depth shape mismatch")
    for name in ("wall0_parent_asymmetry", "wall1_parent_asymmetry", "wall0_parent_same_sign_l1", "wall1_parent_same_sign_l1"):
        if arrays[name].shape != (DEPTH_COUNT,):
            raise ValueError(f"Stage-130 {name} shape mismatch")
    for name in ("wall0_sector_contributions", "wall1_sector_contributions"):
        if arrays[name].shape != (DEPTH_COUNT, 2):
            raise ValueError(f"Stage-130 {name} shape mismatch")
    if arrays["sector_transition_deltas"].shape != (2, 2) or arrays["parent_transition_deltas"].shape != (2,):
        raise ValueError("Stage-130 transition shape mismatch")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Stage-130 payload contains nonfinite values")
    return summary, arrays


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid Stage-131 cosine payload")
    return float(np.dot(x, y) / max(float(np.linalg.norm(x) * np.linalg.norm(y)), 1.0e-300))


def _positive_scale_and_residual(target: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    x = np.asarray(target, dtype=np.float64).ravel()
    y = np.asarray(reference, dtype=np.float64).ravel()
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid Stage-131 scale payload")
    denom = float(np.dot(y, y))
    alpha = max(float(np.dot(x, y)) / max(denom, 1.0e-300), 0.0)
    residual = float(np.linalg.norm(x - alpha * y) / max(float(np.linalg.norm(x)), 1.0e-300))
    return alpha, residual


def crossing_aligned_profiles(
    depth: np.ndarray,
    wall0: np.ndarray,
    wall1: np.ndarray,
    crossing0: float,
    crossing1: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.asarray(depth, dtype=np.float64)
    a = np.asarray(wall0, dtype=np.float64)
    b = np.asarray(wall1, dtype=np.float64)
    if d.shape != (DEPTH_COUNT,) or a.shape != (DEPTH_COUNT, 2) or b.shape != (DEPTH_COUNT, 2):
        raise ValueError("Invalid Stage-131 profile payload")
    z0 = d - float(crossing0)
    z1 = d - float(crossing1)
    keep = (z0 >= float(np.min(z1))) & (z0 <= float(np.max(z1)))
    if int(np.count_nonzero(keep)) < DEPTH_COUNT - 4:
        raise ValueError("Insufficient Stage-131 crossing-aligned overlap")
    z = z0[keep]
    interp = np.column_stack([np.interp(z, z1, b[:, j]) for j in range(2)])
    return z, a[keep], interp


def classify_mapping(
    *,
    direct_profile_cosine: float,
    mirrored_profile_cosine: float,
    mirrored_transition_cosine: float,
    mirrored_profile_relative_residual: float,
    finite: bool = True,
    closure: float = 0.0,
) -> str:
    values = np.asarray(
        [direct_profile_cosine, mirrored_profile_cosine, mirrored_transition_cosine, mirrored_profile_relative_residual, closure],
        dtype=np.float64,
    )
    if not finite or not np.isfinite(values).all():
        return NONFINITE
    if closure > PARENT_CLOSURE_TOLERANCE:
        return CLOSURE_BLOCKER
    mirror_gain = mirrored_profile_cosine - direct_profile_cosine
    if (
        mirrored_profile_cosine >= MIRROR_PROFILE_COSINE_MIN
        and mirrored_transition_cosine >= MIRROR_TRANSITION_COSINE_MIN
        and mirrored_profile_relative_residual <= MIRROR_PROFILE_RELATIVE_RESIDUAL_MAX
        and mirror_gain >= MIRROR_GAIN_OVER_DIRECT_MIN
    ):
        return MIRROR
    if direct_profile_cosine >= MIRROR_PROFILE_COSINE_MIN and direct_profile_cosine >= mirrored_profile_cosine:
        return DIRECT
    return COMPLEX


def run(stage130_dir: str | Path, stage130_record_path: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage131_design(**design)
    summary130, arrays = _load_stage130(stage130_dir, stage130_record_path)

    crossing0 = float(summary130["metrics"]["axis1_low"]["crossing_depth_cells"])
    crossing1 = float(summary130["metrics"]["axis1_high"]["crossing_depth_cells"])
    z, wall0, wall1 = crossing_aligned_profiles(
        arrays["depth"], arrays["wall0_sector_contributions"], arrays["wall1_sector_contributions"], crossing0, crossing1
    )
    mirrored_wall1 = wall1[:, ::-1]
    direct_profile_cosine = _cosine(wall0, wall1)
    mirrored_profile_cosine = _cosine(wall0, mirrored_wall1)
    mirrored_scale, mirrored_residual = _positive_scale_and_residual(wall0, mirrored_wall1)
    component_cosines = [_cosine(wall0[:, j], mirrored_wall1[:, j]) for j in range(2)]

    transition = arrays["sector_transition_deltas"]
    direct_transition_cosine = _cosine(transition[0], transition[1])
    mirrored_transition_cosine = _cosine(transition[0], transition[1, ::-1])
    transition_scale, transition_residual = _positive_scale_and_residual(transition[0], transition[1, ::-1])

    parent0 = np.interp(z, arrays["depth"] - crossing0, arrays["wall0_parent_asymmetry"])
    parent1 = np.interp(z, arrays["depth"] - crossing1, arrays["wall1_parent_asymmetry"])
    parent_profile_cosine = _cosine(parent0, parent1)
    parent_scale, parent_residual = _positive_scale_and_residual(parent0, parent1)

    support0 = np.interp(z, arrays["depth"] - crossing0, arrays["wall0_parent_same_sign_l1"])
    support1 = np.interp(z, arrays["depth"] - crossing1, arrays["wall1_parent_same_sign_l1"])
    support_profile_cosine = _cosine(support0, support1)

    additive0 = np.max(np.abs(np.sum(arrays["wall0_sector_contributions"], axis=1) - arrays["wall0_parent_asymmetry"]))
    additive1 = np.max(np.abs(np.sum(arrays["wall1_sector_contributions"], axis=1) - arrays["wall1_parent_asymmetry"]))
    transition_closure = np.max(np.abs(np.sum(transition, axis=1) - arrays["parent_transition_deltas"]))
    inherited_closure = float(summary130["aggregate"]["maximum_parent_closure"])
    maximum_closure = float(max(additive0, additive1, transition_closure, inherited_closure))

    finite = bool(
        np.isfinite(wall0).all()
        and np.isfinite(wall1).all()
        and np.isfinite(transition).all()
        and np.isfinite(maximum_closure)
    )
    decision = classify_mapping(
        direct_profile_cosine=direct_profile_cosine,
        mirrored_profile_cosine=mirrored_profile_cosine,
        mirrored_transition_cosine=mirrored_transition_cosine,
        mirrored_profile_relative_residual=mirrored_residual,
        finite=finite,
        closure=maximum_closure,
    )

    if decision == MIRROR:
        conclusion = (
            "After expressing wall distance relative to each Stage-129 crossing and exchanging sectors 5 and 6 on one opposite wall, the Stage-130 node-9 sector profiles collapse to a common mirrored shape. The raw wall-specific dominant-sector identity is therefore largely consistent with opposite-wall geometric sector exchange rather than two unrelated angular mechanisms. The remaining crossing-adjacent amplitude/phase mismatch is diagnostic only and motivates a fixed crossing-phase audit; it does not establish limiter causality, MUSCL stability, endpoint convergence, q_av improvement, benchmark accuracy, or validation."
        )
    elif decision == DIRECT:
        conclusion = (
            "The opposite-wall Stage-130 sector profiles are more consistent under direct sector labeling than under mirror exchange. This fixed artifact result motivates a common-sector depth-profile audit only; it does not establish limiter causality, MUSCL stability, endpoint convergence, q_av improvement, benchmark accuracy, or validation."
        )
    elif decision == COMPLEX:
        conclusion = (
            "Neither direct nor mirrored sector labeling provides a preregistered simple cross-wall collapse of the Stage-130 node-9 sector profiles. The angular support remains genuinely mixed at this diagnostic resolution and should be audited locally rather than retuned."
        )
    else:
        conclusion = (
            "Stage 131 is blocked by nonfinite data or failure to preserve the exact Stage-130 additive/transition closures. No angular interpretation or parameter change is justified."
        )

    summary = {
        "stage": 131,
        "finite": finite,
        "configuration": {
            "grid": list(GRID),
            "interior_grid": list(INTERIOR_GRID),
            "kn0": KNUDSEN,
            "cold_hot_ratio": COLD_HOT_RATIO,
            "rule": list(RULE),
            "radial_scale": RADIAL_SCALE,
            "limiter": LIMITER,
            "boundary_slope": BOUNDARY_SLOPE,
            "source_relaxation": SOURCE_RELAXATION,
            "tolerance": TOLERANCE,
            "correction_floor": CORRECTION_FLOOR,
            "diagnostic_steps": DIAGNOSTIC_STEPS,
            "witness_node": WITNESS_NODE,
            "pair_sectors": list(PAIR_SECTORS),
            "cross_wall_mapping_tested": ["direct", "sector5_sector6_exchange"],
            "depth_alignment": "depth minus each wall's fixed Stage-129 crossing depth; wall1 linearly interpolated onto wall0 relative-depth samples",
            "mirror_profile_cosine_min": MIRROR_PROFILE_COSINE_MIN,
            "mirror_transition_cosine_min": MIRROR_TRANSITION_COSINE_MIN,
            "mirror_profile_relative_residual_max": MIRROR_PROFILE_RELATIVE_RESIDUAL_MAX,
            "mirror_gain_over_direct_min": MIRROR_GAIN_OVER_DIRECT_MIN,
            "parent_closure_tolerance": PARENT_CLOSURE_TOLERANCE,
            "solver_rerun": False,
            "solver_endpoint_advanced": False,
            "physical_parameter_retuning": False,
            "wall_retuning": False,
            "collision_source_retuning": False,
            "reconstruction_retuning": False,
            "transport_retuning": False,
            "limiter_retuning": False,
            "floor_retuning": False,
            "normalization_retuning": False,
            "source_relaxation_retuning": False,
            "velocity_grid_retuning": False,
            "cross_knudsen_extension_permitted": False,
            "benchmark_or_validation_claim_permitted": False,
        },
        "parents": {
            "stage130_run_id": STAGE130_RUN_ID,
            "stage130_job_id": STAGE130_JOB_ID,
            "stage130_artifact_id": STAGE130_ARTIFACT_ID,
            "stage130_source_head": STAGE130_SOURCE_HEAD,
        },
        "metrics": {
            "overlap_sample_count": int(z.size),
            "relative_depth_min_cells": float(np.min(z)),
            "relative_depth_max_cells": float(np.max(z)),
            "direct_profile_cosine": direct_profile_cosine,
            "mirrored_profile_cosine": mirrored_profile_cosine,
            "mirror_gain_over_direct": mirrored_profile_cosine - direct_profile_cosine,
            "minimum_mirrored_component_cosine": float(min(component_cosines)),
            "mirrored_sector5_component_cosine": float(component_cosines[0]),
            "mirrored_sector6_component_cosine": float(component_cosines[1]),
            "mirrored_profile_positive_scale_wall1_to_wall0": mirrored_scale,
            "mirrored_profile_relative_l2_residual": mirrored_residual,
            "direct_transition_delta_cosine": direct_transition_cosine,
            "mirrored_transition_delta_cosine": mirrored_transition_cosine,
            "mirrored_transition_positive_scale_wall1_to_wall0": transition_scale,
            "mirrored_transition_relative_l2_residual": transition_residual,
            "parent_asymmetry_profile_cosine": parent_profile_cosine,
            "parent_asymmetry_positive_scale_wall1_to_wall0": parent_scale,
            "parent_asymmetry_relative_l2_residual": parent_residual,
            "parent_support_profile_cosine": support_profile_cosine,
        },
        "aggregate": {
            "maximum_stage130_additive_or_transition_absolute_closure": maximum_closure,
            "inherited_stage130_maximum_parent_closure": inherited_closure,
        },
        "decision": decision,
        "scientific_conclusion": conclusion,
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. No physical, collision/source, floor, wall, reconstruction, transport, limiter, normalization, source-relaxation, or velocity-quadrature parameter is retuned; no solver endpoint or cross-Knudsen extension is advanced."
        ),
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        output / "cross_wall_sector_interaction.npz",
        relative_depth=z,
        wall0_sector_profiles=wall0,
        wall1_direct_sector_profiles=wall1,
        wall1_mirrored_sector_profiles=mirrored_wall1,
        wall0_transition_delta=transition[0],
        wall1_direct_transition_delta=transition[1],
        wall1_mirrored_transition_delta=transition[1, ::-1],
        parent0_aligned=parent0,
        parent1_aligned=parent1,
        support0_aligned=support0,
        support1_aligned=support1,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 131 cross-wall sector interaction audit")
    parser.add_argument("--stage130-dir", required=True)
    parser.add_argument("--stage130-record", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run(args.stage130_dir, args.stage130_record, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
