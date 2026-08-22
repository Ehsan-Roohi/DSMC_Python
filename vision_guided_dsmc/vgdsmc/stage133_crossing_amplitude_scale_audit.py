from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STAGE132_RUN_ID = 32001930265
STAGE132_JOB_ID = 95303770402
STAGE132_ARTIFACT_ID = 9284552744
STAGE132_ARTIFACT_SHA256 = "6e150457904e188df33bd04cf73228045d5ac67500aed7be2f35b7c71acc16b1"
STAGE132_SUMMARY_SHA256 = "bbcfe2e0a60411891e81d0535d4e341346ebea3283ede3c8e7fb7f96fce70443"
STAGE132_PAYLOAD_SHA256 = "59c00c09bf0e8109946bf720f279bb151314c271f1fc67ae4b2378cb6abcc198"
STAGE132_SOURCE_HEAD = "c2c9b03b83e1c3fa546feee90ab36bb166193ecd"
STAGE132_DECISION = "stage132_common_mirrored_crossing_phase_width_stage133_crossing_amplitude_scale_audit"

GRID = (64, 64)
INTERIOR_GRID = (56, 56)
WITNESS_NODE = 9
PAIR_SECTORS = (5, 6)
DOMINANT_MIRRORED_SECTOR = 6
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
CORE_HALF_WIDTH_CELLS = 4.0
AFFINE_GAIN_MIN = 0.35
AFFINE_RESIDUAL_MAX = 0.15
SCALE_ONLY_GAIN_MIN = 0.20
OFFSET_INCREMENT_MIN = 0.20

MULTIPLICATIVE = "stage133_material_multiplicative_amplitude_scale_stage134_amplitude_origin_audit"
OFFSET_DOMINATED = "stage133_affine_offset_dominated_crossing_mismatch_stage134_additive_baseline_origin_audit"
INSUFFICIENT = "stage133_amplitude_scaling_insufficient_stage134_crossing_shape_residual_audit"
NONFINITE = "stage133_nonfinite_crossing_amplitude_blocker_without_retuning"
CLOSURE_BLOCKER = "stage133_parent_payload_closure_blocker_without_retuning"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_stage133_design(**overrides: object) -> None:
    frozen = {
        "stage132_run_id": STAGE132_RUN_ID,
        "stage132_job_id": STAGE132_JOB_ID,
        "stage132_artifact_id": STAGE132_ARTIFACT_ID,
        "grid": GRID,
        "interior_grid": INTERIOR_GRID,
        "witness_node": WITNESS_NODE,
        "pair_sectors": PAIR_SECTORS,
        "dominant_mirrored_sector": DOMINANT_MIRRORED_SECTOR,
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
        "core_half_width_cells": CORE_HALF_WIDTH_CELLS,
        "affine_gain_min": AFFINE_GAIN_MIN,
        "affine_residual_max": AFFINE_RESIDUAL_MAX,
        "scale_only_gain_min": SCALE_ONLY_GAIN_MIN,
        "offset_increment_min": OFFSET_INCREMENT_MIN,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 133 is frozen to the exact completed Stage-132 crossing profiles and preregistered amplitude-fit guards. "
            "No physical/numerical retuning, phase shifting, width refitting, failed-MUSCL retuning, solver rerun, or cross-Knudsen extension is permitted."
        )


def _verify_record(record: dict[str, object]) -> None:
    tests = record.get("tests", {})
    checks = (
        record.get("stage") == 132,
        record.get("source_head") == STAGE132_SOURCE_HEAD,
        record.get("workflow_status") == "completed",
        record.get("workflow_conclusion") == "success",
        record.get("workflow_run_id") == STAGE132_RUN_ID,
        record.get("workflow_job_id") == STAGE132_JOB_ID,
        record.get("artifact_id") == STAGE132_ARTIFACT_ID,
        record.get("artifact_sha256") == STAGE132_ARTIFACT_SHA256,
        record.get("summary_sha256") == STAGE132_SUMMARY_SHA256,
        record.get("crossing_phase_transition_width_sha256") == STAGE132_PAYLOAD_SHA256,
        record.get("decision") == STAGE132_DECISION,
        isinstance(tests, dict) and tests.get("passed") == 18,
        isinstance(tests, dict) and tests.get("failed") == 0,
    )
    if not all(checks):
        raise ValueError("Committed Stage-132 provenance does not authorize Stage 133")


def _load_parent(root: str | Path, record_path: str | Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    root = Path(root)
    for name, digest in {
        "summary.json": STAGE132_SUMMARY_SHA256,
        "crossing_phase_transition_width.npz": STAGE132_PAYLOAD_SHA256,
    }.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Stage-132 checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    _verify_record(json.loads(Path(record_path).read_text(encoding="utf-8")))
    if summary.get("stage") != 132 or summary.get("finite") is not True or summary.get("decision") != STAGE132_DECISION:
        raise ValueError("Stage-132 artifact does not authorize Stage 133")
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
        "dominant_mirrored_sector": DOMINANT_MIRRORED_SECTOR,
        "solver_rerun": False,
        "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False,
    }
    if any(cfg.get(k) != v for k, v in required.items()) or any(v is not False for k, v in cfg.items() if k.endswith("_retuning")):
        raise ValueError("Stage-132 frozen design mismatch")
    with np.load(root / "crossing_phase_transition_width.npz") as data:
        needed = (
            "relative_depth", "dominant_wall0", "dominant_wall1_mirrored", "parent_wall0", "parent_wall1"
        )
        if not set(needed).issubset(data.files):
            raise ValueError("Stage-132 payload is incomplete")
        arrays = {name: np.asarray(data[name], dtype=np.float64).copy() for name in needed}
    n = arrays["relative_depth"].size
    if n != 26 or any(arrays[name].shape != (n,) for name in needed[1:]):
        raise ValueError("Stage-132 profile shape mismatch")
    if not np.all(np.diff(arrays["relative_depth"]) > 0.0) or not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Stage-132 payload contains invalid values")
    return summary, arrays


def _relative_l2(target: np.ndarray, estimate: np.ndarray) -> float:
    denom = float(np.linalg.norm(target))
    if denom <= 1.0e-14:
        raise ValueError("Degenerate target norm")
    return float(np.linalg.norm(target - estimate) / denom)


def positive_scale_fit(target: np.ndarray, source: np.ndarray) -> dict[str, float | np.ndarray]:
    target = np.asarray(target, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    if target.shape != source.shape or target.ndim != 1 or target.size < 3 or not np.isfinite(target).all() or not np.isfinite(source).all():
        raise ValueError("Invalid amplitude-fit profiles")
    denom = float(np.dot(source, source))
    if denom <= 1.0e-14:
        raise ValueError("Degenerate source norm")
    raw = _relative_l2(target, source)
    scale = float(np.dot(source, target) / denom)
    if scale <= 0.0:
        raise ValueError("Positive scale fit became non-positive")
    fitted = scale * source
    residual = _relative_l2(target, fitted)
    gain = 0.0 if raw <= 1.0e-14 else float((raw - residual) / raw)
    return {"scale": scale, "raw_relative_l2": raw, "residual_relative_l2": residual, "gain_fraction": gain, "fitted": fitted}


def positive_affine_fit(target: np.ndarray, source: np.ndarray) -> dict[str, float | np.ndarray]:
    target = np.asarray(target, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    if target.shape != source.shape or target.ndim != 1 or target.size < 3 or not np.isfinite(target).all() or not np.isfinite(source).all():
        raise ValueError("Invalid affine amplitude-fit profiles")
    raw = _relative_l2(target, source)
    matrix = np.column_stack([source, np.ones_like(source)])
    scale, offset = np.linalg.lstsq(matrix, target, rcond=None)[0]
    scale = float(scale)
    offset = float(offset)
    if scale <= 0.0 or not np.isfinite([scale, offset]).all():
        raise ValueError("Positive affine fit became invalid")
    fitted = scale * source + offset
    residual = _relative_l2(target, fitted)
    gain = 0.0 if raw <= 1.0e-14 else float((raw - residual) / raw)
    dynamic = float(np.ptp(target))
    if dynamic <= 1.0e-14:
        raise ValueError("Degenerate target dynamic range")
    return {
        "scale": scale,
        "offset": offset,
        "offset_fraction_of_target_dynamic_range": float(abs(offset) / dynamic),
        "raw_relative_l2": raw,
        "residual_relative_l2": residual,
        "gain_fraction": gain,
        "fitted": fitted,
    }


def classify_amplitude_scale(*, dominant_scale_gain: float, parent_scale_gain: float,
                             dominant_affine_gain: float, parent_affine_gain: float,
                             dominant_affine_residual: float, parent_affine_residual: float,
                             finite: bool = True, closure: float = 0.0) -> str:
    vals = np.asarray([
        dominant_scale_gain, parent_scale_gain, dominant_affine_gain, parent_affine_gain,
        dominant_affine_residual, parent_affine_residual, closure,
    ], dtype=np.float64)
    if not finite or not np.isfinite(vals).all():
        return NONFINITE
    if closure > PARENT_CLOSURE_TOLERANCE:
        return CLOSURE_BLOCKER
    affine_good = (
        min(dominant_affine_gain, parent_affine_gain) >= AFFINE_GAIN_MIN
        and max(dominant_affine_residual, parent_affine_residual) <= AFFINE_RESIDUAL_MAX
    )
    if not affine_good:
        return INSUFFICIENT
    if min(dominant_scale_gain, parent_scale_gain) >= SCALE_ONLY_GAIN_MIN:
        return MULTIPLICATIVE
    offset_increment = min(
        dominant_affine_gain - dominant_scale_gain,
        parent_affine_gain - parent_scale_gain,
    )
    if offset_increment >= OFFSET_INCREMENT_MIN:
        return OFFSET_DOMINATED
    return INSUFFICIENT


def _public_fit(scale_fit: dict[str, float | np.ndarray], affine_fit: dict[str, float | np.ndarray]) -> dict[str, float]:
    return {
        "raw_relative_l2": float(scale_fit["raw_relative_l2"]),
        "scale_only_positive_scale_wall1_to_wall0": float(scale_fit["scale"]),
        "scale_only_relative_l2_residual": float(scale_fit["residual_relative_l2"]),
        "scale_only_gain_fraction": float(scale_fit["gain_fraction"]),
        "affine_positive_scale_wall1_to_wall0": float(affine_fit["scale"]),
        "affine_offset_wall0_units": float(affine_fit["offset"]),
        "affine_offset_fraction_of_target_dynamic_range": float(affine_fit["offset_fraction_of_target_dynamic_range"]),
        "affine_relative_l2_residual": float(affine_fit["residual_relative_l2"]),
        "affine_gain_fraction": float(affine_fit["gain_fraction"]),
        "affine_gain_increment_over_scale_only": float(affine_fit["gain_fraction"] - scale_fit["gain_fraction"]),
    }


def run(stage132_dir: str | Path, stage132_record_path: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage133_design(**design)
    parent, arrays = _load_parent(stage132_dir, stage132_record_path)
    depth = arrays["relative_depth"]
    mask = np.abs(depth) <= CORE_HALF_WIDTH_CELLS
    if int(np.count_nonzero(mask)) != 8:
        raise ValueError("Frozen Stage-133 transition-core sample count changed")
    x = depth[mask]
    d0 = arrays["dominant_wall0"][mask]
    d1 = arrays["dominant_wall1_mirrored"][mask]
    p0 = arrays["parent_wall0"][mask]
    p1 = arrays["parent_wall1"][mask]

    ds = positive_scale_fit(d0, d1)
    da = positive_affine_fit(d0, d1)
    ps = positive_scale_fit(p0, p1)
    pa = positive_affine_fit(p0, p1)
    closure = float(parent["aggregate"]["maximum_parent_closure"])
    decision = classify_amplitude_scale(
        dominant_scale_gain=float(ds["gain_fraction"]),
        parent_scale_gain=float(ps["gain_fraction"]),
        dominant_affine_gain=float(da["gain_fraction"]),
        parent_affine_gain=float(pa["gain_fraction"]),
        dominant_affine_residual=float(da["residual_relative_l2"]),
        parent_affine_residual=float(pa["residual_relative_l2"]),
        finite=True,
        closure=closure,
    )

    if decision == OFFSET_DOMINATED:
        conclusion = (
            "On the fixed +/-4-cell Stage-132 transition core, a positive multiplicative scale alone removes little of the dominant-sector mismatch, "
            "while a positive scale plus additive baseline removes a material fraction of the local mismatch in both the dominant mirrored sector and the parent profile. "
            "The residual therefore behaves more like a common additive-baseline/offset plus modest amplitude difference than a pure multiplicative amplitude, phase, or width effect. "
            "A fixed additive-baseline origin audit is the next justified diagnostic. This artifact-only result does not establish limiter causality, MUSCL stability, endpoint convergence, q_av improvement, benchmark accuracy, or validation."
        )
    elif decision == MULTIPLICATIVE:
        conclusion = (
            "On the fixed +/-4-cell Stage-132 transition core, a common positive multiplicative amplitude rescaling materially reduces the cross-wall mismatch in both retained profiles. "
            "A fixed amplitude-origin audit is the next justified diagnostic. This artifact-only result does not establish limiter causality, MUSCL stability, endpoint convergence, q_av improvement, benchmark accuracy, or validation."
        )
    elif decision == INSUFFICIENT:
        conclusion = (
            "On the fixed +/-4-cell Stage-132 transition core, preregistered amplitude rescaling is insufficient to explain the remaining cross-wall mismatch. "
            "A fixed crossing-shape residual audit is the next justified diagnostic. This artifact-only result does not establish limiter causality, MUSCL stability, endpoint convergence, q_av improvement, benchmark accuracy, or validation."
        )
    else:
        conclusion = "Stage 133 is blocked by nonfinite data or parent-closure failure without retuning."

    summary: dict[str, object] = {
        "stage": 133,
        "finite": True,
        "configuration": {
            "grid": list(GRID), "interior_grid": list(INTERIOR_GRID), "kn0": KNUDSEN,
            "cold_hot_ratio": COLD_HOT_RATIO, "rule": list(RULE), "radial_scale": RADIAL_SCALE,
            "limiter": LIMITER, "boundary_slope": BOUNDARY_SLOPE, "source_relaxation": SOURCE_RELAXATION,
            "tolerance": TOLERANCE, "correction_floor": CORRECTION_FLOOR, "diagnostic_steps": DIAGNOSTIC_STEPS,
            "witness_node": WITNESS_NODE, "pair_sectors": list(PAIR_SECTORS), "dominant_mirrored_sector": DOMINANT_MIRRORED_SECTOR,
            "core_half_width_cells": CORE_HALF_WIDTH_CELLS, "core_sample_count": int(x.size),
            "affine_gain_min": AFFINE_GAIN_MIN, "affine_residual_max": AFFINE_RESIDUAL_MAX,
            "scale_only_gain_min": SCALE_ONLY_GAIN_MIN, "offset_increment_min": OFFSET_INCREMENT_MIN,
            "phase_shift_applied": False, "width_refit_applied": False,
            "solver_rerun": False, "solver_endpoint_advanced": False, "cross_knudsen_extension_permitted": False,
            "benchmark_or_validation_claim_permitted": False,
            "physical_parameter_retuning": False, "collision_source_retuning": False, "floor_retuning": False,
            "wall_retuning": False, "reconstruction_retuning": False, "transport_retuning": False,
            "limiter_retuning": False, "normalization_retuning": False, "source_relaxation_retuning": False,
            "velocity_grid_retuning": False,
        },
        "parents": {
            "stage132_run_id": STAGE132_RUN_ID, "stage132_job_id": STAGE132_JOB_ID,
            "stage132_artifact_id": STAGE132_ARTIFACT_ID, "stage132_source_head": STAGE132_SOURCE_HEAD,
        },
        "aggregate": {
            "maximum_parent_closure": closure,
            "minimum_affine_gain_fraction": float(min(da["gain_fraction"], pa["gain_fraction"])),
            "maximum_affine_relative_l2_residual": float(max(da["residual_relative_l2"], pa["residual_relative_l2"])),
            "minimum_affine_gain_increment_over_scale_only": float(min(
                da["gain_fraction"] - ds["gain_fraction"], pa["gain_fraction"] - ps["gain_fraction"]
            )),
        },
        "metrics": {
            "core_depth_min_cells": float(x.min()), "core_depth_max_cells": float(x.max()),
            "dominant_sector": _public_fit(ds, da), "parent_profile": _public_fit(ps, pa),
            "inherited_stage132_maximum_phase_offset_cells": float(parent["aggregate"]["maximum_phase_offset_cells"]),
            "inherited_stage132_maximum_transition_width_ratio": float(parent["aggregate"]["maximum_transition_width_ratio"]),
        },
        "decision": decision,
        "scientific_conclusion": conclusion,
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. No physical, collision/source, floor, wall, reconstruction, transport, limiter, normalization, "
            "source-relaxation, or velocity-quadrature parameter is retuned; no solver endpoint or cross-Knudsen extension is advanced."
        ),
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        out / "crossing_amplitude_scale.npz",
        relative_depth_core=x,
        dominant_wall0_core=d0,
        dominant_wall1_core=d1,
        dominant_scale_only_fit=np.asarray(ds["fitted"]),
        dominant_affine_fit=np.asarray(da["fitted"]),
        dominant_affine_residual=d0 - np.asarray(da["fitted"]),
        parent_wall0_core=p0,
        parent_wall1_core=p1,
        parent_scale_only_fit=np.asarray(ps["fitted"]),
        parent_affine_fit=np.asarray(pa["fitted"]),
        parent_affine_residual=p0 - np.asarray(pa["fitted"]),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 133 crossing-amplitude scale audit")
    parser.add_argument("--stage132-dir", required=True)
    parser.add_argument("--stage132-record", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = run(args.stage132_dir, args.stage132_record, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
