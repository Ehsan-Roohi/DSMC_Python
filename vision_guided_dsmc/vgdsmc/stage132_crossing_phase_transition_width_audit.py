from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STAGE131_RUN_ID = 31984973371
STAGE131_JOB_ID = 95258262981
STAGE131_ARTIFACT_ID = 9278022287
STAGE131_ARTIFACT_SHA256 = "29122d3cdbee0e7246f274582bd04a33976a3ce83b20be456e71d7f7d07062b8"
STAGE131_SUMMARY_SHA256 = "67db62b54cc5e8de680f74d61604a2246ae5f4f6840b694644f8569a0d17b253"
STAGE131_PAYLOAD_SHA256 = "837d2373df50040208ed3b130da64caa868cfc1c614f8206310272e2a8354b85"
STAGE131_SOURCE_HEAD = "73a9cbd2605f01e1e68e6f6a8145d6a264d9f05b"
STAGE131_DECISION = "stage131_mirror_sector_exchange_collapse_stage132_crossing_phase_transition_width_audit"

GRID = (64, 64)
INTERIOR_GRID = (56, 56)
OVERLAP_COUNT = 26
WITNESS_NODE = 9
PAIR_SECTORS = (5, 6)
DOMINANT_COMPONENT = 1
DOMINANT_SECTOR = 6
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
PLATEAU_DEPTH_CELLS = 5.0
PHASE_OFFSET_MAX_CELLS = 0.5
WIDTH_RATIO_MAX = 1.15
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)

COMMON = "stage132_common_mirrored_crossing_phase_width_stage133_crossing_amplitude_scale_audit"
PHASE = "stage132_material_crossing_phase_offset_stage133_crossing_offset_origin_audit"
WIDTH = "stage132_material_transition_width_mismatch_stage133_wall_normal_phase_gradient_audit"
NONFINITE = "stage132_nonfinite_crossing_phase_width_blocker_without_retuning"
CLOSURE_BLOCKER = "stage132_parent_payload_closure_blocker_without_retuning"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_stage132_design(**overrides: object) -> None:
    frozen = {
        "stage131_run_id": STAGE131_RUN_ID,
        "stage131_job_id": STAGE131_JOB_ID,
        "stage131_artifact_id": STAGE131_ARTIFACT_ID,
        "grid": GRID,
        "interior_grid": INTERIOR_GRID,
        "overlap_count": OVERLAP_COUNT,
        "witness_node": WITNESS_NODE,
        "pair_sectors": PAIR_SECTORS,
        "dominant_component": DOMINANT_COMPONENT,
        "dominant_sector": DOMINANT_SECTOR,
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
        "plateau_depth_cells": PLATEAU_DEPTH_CELLS,
        "phase_offset_max_cells": PHASE_OFFSET_MAX_CELLS,
        "width_ratio_max": WIDTH_RATIO_MAX,
        "quantiles": QUANTILES,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 132 is frozen to the exact completed Stage-131 mirrored node-9 sectors-5+6 artifact. "
            "No physical or numerical retuning, failed-MUSCL retuning, guard relaxation, solver rerun, or cross-Knudsen extension is permitted."
        )


def _verify_record(record: dict[str, object]) -> None:
    tests = record.get("tests", {})
    checks = (
        record.get("stage") == 131,
        record.get("source_head") == STAGE131_SOURCE_HEAD,
        record.get("workflow_status") == "completed",
        record.get("workflow_conclusion") == "success",
        record.get("workflow_run_id") == STAGE131_RUN_ID,
        record.get("workflow_job_id") == STAGE131_JOB_ID,
        record.get("artifact_id") == STAGE131_ARTIFACT_ID,
        record.get("artifact_sha256") == STAGE131_ARTIFACT_SHA256,
        record.get("summary_sha256") == STAGE131_SUMMARY_SHA256,
        record.get("cross_wall_sector_interaction_sha256") == STAGE131_PAYLOAD_SHA256,
        record.get("decision") == STAGE131_DECISION,
        isinstance(tests, dict) and tests.get("passed") == 17,
        isinstance(tests, dict) and tests.get("failed") == 0,
    )
    if not all(checks):
        raise ValueError("Committed Stage-131 provenance does not authorize Stage 132")


def _load_parent(root: str | Path, record_path: str | Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    root = Path(root)
    for name, digest in {
        "summary.json": STAGE131_SUMMARY_SHA256,
        "cross_wall_sector_interaction.npz": STAGE131_PAYLOAD_SHA256,
    }.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Stage-131 checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    _verify_record(json.loads(Path(record_path).read_text(encoding="utf-8")))
    if summary.get("stage") != 131 or summary.get("finite") is not True or summary.get("decision") != STAGE131_DECISION:
        raise ValueError("Stage-131 artifact does not authorize Stage 132")
    cfg = summary.get("configuration", {})
    required = {
        "grid": list(GRID), "interior_grid": list(INTERIOR_GRID), "kn0": KNUDSEN,
        "cold_hot_ratio": COLD_HOT_RATIO, "rule": list(RULE), "radial_scale": RADIAL_SCALE,
        "limiter": LIMITER, "boundary_slope": BOUNDARY_SLOPE, "source_relaxation": SOURCE_RELAXATION,
        "correction_floor": CORRECTION_FLOOR, "witness_node": WITNESS_NODE, "pair_sectors": list(PAIR_SECTORS),
        "solver_rerun": False, "solver_endpoint_advanced": False, "cross_knudsen_extension_permitted": False,
    }
    if any(cfg.get(k) != v for k, v in required.items()) or any(v is not False for k, v in cfg.items() if k.endswith("_retuning")):
        raise ValueError("Stage-131 frozen design mismatch")
    with np.load(root / "cross_wall_sector_interaction.npz") as data:
        needed = (
            "relative_depth", "wall0_sector_profiles", "wall1_mirrored_sector_profiles",
            "wall0_transition_delta", "wall1_mirrored_transition_delta", "parent0_aligned", "parent1_aligned",
        )
        if not set(needed).issubset(data.files):
            raise ValueError("Stage-131 payload is incomplete")
        a = {name: np.asarray(data[name], dtype=np.float64).copy() for name in needed}
    if a["relative_depth"].shape != (OVERLAP_COUNT,):
        raise ValueError("Stage-131 depth shape mismatch")
    if a["wall0_sector_profiles"].shape != (OVERLAP_COUNT, 2) or a["wall1_mirrored_sector_profiles"].shape != (OVERLAP_COUNT, 2):
        raise ValueError("Stage-131 sector profile shape mismatch")
    if a["wall0_transition_delta"].shape != (2,) or a["wall1_mirrored_transition_delta"].shape != (2,):
        raise ValueError("Stage-131 transition shape mismatch")
    if a["parent0_aligned"].shape != (OVERLAP_COUNT,) or a["parent1_aligned"].shape != (OVERLAP_COUNT,):
        raise ValueError("Stage-131 parent profile shape mismatch")
    if not all(np.isfinite(x).all() for x in a.values()):
        raise ValueError("Stage-131 payload contains nonfinite values")
    return summary, a


def _crossings(x: np.ndarray, y: np.ndarray, level: float) -> list[float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64) - float(level)
    if x.ndim != 1 or y.shape != x.shape or x.size < 2 or not np.all(np.diff(x) > 0.0):
        raise ValueError("Invalid crossing grid")
    out: list[float] = []
    for i in range(x.size - 1):
        a, b = float(y[i]), float(y[i + 1])
        if a == 0.0:
            out.append(float(x[i]))
        if a * b < 0.0 or (a != 0.0 and b == 0.0):
            out.append(float(x[i] - a * (x[i + 1] - x[i]) / (b - a)))
    if y[-1] == 0.0:
        out.append(float(x[-1]))
    return out


def crossing_nearest_zero(x: np.ndarray, y: np.ndarray, level: float = 0.0) -> float:
    values = _crossings(x, y, level)
    if not values:
        raise ValueError("Profile does not cross requested level")
    return float(min(values, key=abs))


def transition_signature(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    left_mask = x <= -PLATEAU_DEPTH_CELLS
    right_mask = x >= PLATEAU_DEPTH_CELLS
    if x.shape != y.shape or x.ndim != 1 or np.count_nonzero(left_mask) < 3 or np.count_nonzero(right_mask) < 3:
        raise ValueError("Invalid transition profile")
    left = float(np.mean(y[left_mask]))
    right = float(np.mean(y[right_mask]))
    dynamic = float(abs(left - right))
    if dynamic <= 1.0e-14 or not np.isfinite([left, right, dynamic]).all():
        raise ValueError("Degenerate transition range")
    progress = (left - y) / (left - right)
    q = np.asarray([crossing_nearest_zero(x, progress, v) for v in QUANTILES], dtype=np.float64)
    if not np.all(np.diff(q) > 0.0):
        raise ValueError("Transition quantiles are not ordered")
    return {
        "left_plateau": left, "right_plateau": right, "dynamic_range": dynamic,
        "zero_crossing_cells": crossing_nearest_zero(x, y), "midpoint_depth_cells": float(q[2]),
        "width_25_75_cells": float(q[3] - q[1]), "width_10_90_cells": float(q[4] - q[0]),
        "quantile_depths": q, "progress": progress,
    }


def _ratio(a: float, b: float) -> float:
    a, b = abs(float(a)), abs(float(b))
    if min(a, b) <= 1.0e-14 or not np.isfinite([a, b]).all():
        raise ValueError("Invalid width ratio")
    return float(max(a, b) / min(a, b))


def classify_phase_width(*, dominant_zero_phase_offset_cells: float, dominant_midpoint_phase_offset_cells: float,
                         dominant_width_25_75_ratio: float, dominant_width_10_90_ratio: float,
                         parent_zero_phase_offset_cells: float, parent_midpoint_phase_offset_cells: float,
                         parent_width_25_75_ratio: float, parent_width_10_90_ratio: float,
                         finite: bool = True, closure: float = 0.0) -> str:
    vals = np.asarray([
        dominant_zero_phase_offset_cells, dominant_midpoint_phase_offset_cells,
        dominant_width_25_75_ratio, dominant_width_10_90_ratio,
        parent_zero_phase_offset_cells, parent_midpoint_phase_offset_cells,
        parent_width_25_75_ratio, parent_width_10_90_ratio, closure,
    ], dtype=np.float64)
    if not finite or not np.isfinite(vals).all():
        return NONFINITE
    if closure > PARENT_CLOSURE_TOLERANCE:
        return CLOSURE_BLOCKER
    if max(dominant_zero_phase_offset_cells, dominant_midpoint_phase_offset_cells,
           parent_zero_phase_offset_cells, parent_midpoint_phase_offset_cells) > PHASE_OFFSET_MAX_CELLS:
        return PHASE
    if max(dominant_width_25_75_ratio, dominant_width_10_90_ratio,
           parent_width_25_75_ratio, parent_width_10_90_ratio) > WIDTH_RATIO_MAX:
        return WIDTH
    return COMMON


def _pub(s: dict[str, object]) -> dict[str, float]:
    return {k: float(s[k]) for k in (
        "left_plateau", "right_plateau", "dynamic_range", "zero_crossing_cells",
        "midpoint_depth_cells", "width_25_75_cells", "width_10_90_cells",
    )}


def run(stage131_dir: str | Path, stage131_record_path: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage132_design(**design)
    parent, a = _load_parent(stage131_dir, stage131_record_path)
    weights = np.abs(a["wall0_transition_delta"]) + np.abs(a["wall1_mirrored_transition_delta"])
    if int(np.argmax(weights)) != DOMINANT_COMPONENT:
        raise ValueError("Stage-131 mirrored dominant channel changed")
    depth = a["relative_depth"]
    d0 = transition_signature(depth, a["wall0_sector_profiles"][:, DOMINANT_COMPONENT])
    d1 = transition_signature(depth, a["wall1_mirrored_sector_profiles"][:, DOMINANT_COMPONENT])
    p0 = transition_signature(depth, a["parent0_aligned"])
    p1 = transition_signature(depth, a["parent1_aligned"])

    dz = abs(float(d0["zero_crossing_cells"]) - float(d1["zero_crossing_cells"]))
    dm = abs(float(d0["midpoint_depth_cells"]) - float(d1["midpoint_depth_cells"]))
    pz = abs(float(p0["zero_crossing_cells"]) - float(p1["zero_crossing_cells"]))
    pm = abs(float(p0["midpoint_depth_cells"]) - float(p1["midpoint_depth_cells"]))
    dw25 = _ratio(float(d0["width_25_75_cells"]), float(d1["width_25_75_cells"]))
    dw10 = _ratio(float(d0["width_10_90_cells"]), float(d1["width_10_90_cells"]))
    pw25 = _ratio(float(p0["width_25_75_cells"]), float(p1["width_25_75_cells"]))
    pw10 = _ratio(float(p0["width_10_90_cells"]), float(p1["width_10_90_cells"]))
    closure = float(parent["aggregate"]["maximum_stage130_additive_or_transition_absolute_closure"])
    decision = classify_phase_width(
        dominant_zero_phase_offset_cells=dz, dominant_midpoint_phase_offset_cells=dm,
        dominant_width_25_75_ratio=dw25, dominant_width_10_90_ratio=dw10,
        parent_zero_phase_offset_cells=pz, parent_midpoint_phase_offset_cells=pm,
        parent_width_25_75_ratio=pw25, parent_width_10_90_ratio=pw10,
        finite=True, closure=closure,
    )
    if decision == COMMON:
        conclusion = (
            "After the fixed Stage-131 sector-5/6 mirror exchange and crossing alignment, both the common dominant sector-6 channel "
            "and the parent asymmetry have sub-half-cell cross-wall phase offsets and transition-width ratios within the preregistered 1.15 guard. "
            "The remaining Stage-131 crossing-adjacent mismatch is therefore not materially explained by wall-normal phase or width at this resolution; "
            "a fixed local amplitude-scale audit is the next justified diagnostic."
        )
    elif decision == PHASE:
        conclusion = "A material mirrored crossing-phase offset remains above the fixed half-cell guard; localize its origin without solver retuning."
    elif decision == WIDTH:
        conclusion = "A material mirrored transition-width mismatch remains above the fixed 1.15 guard; audit wall-normal phase-gradient structure without solver retuning."
    else:
        conclusion = "A nonfinite or parent-closure blocker prevents interpretation; no solver change is authorized."

    summary: dict[str, object] = {
        "stage": 132,
        "finite": True,
        "parents": {"stage131_run_id": STAGE131_RUN_ID, "stage131_job_id": STAGE131_JOB_ID,
                    "stage131_artifact_id": STAGE131_ARTIFACT_ID, "stage131_source_head": STAGE131_SOURCE_HEAD},
        "configuration": {
            "grid": list(GRID), "interior_grid": list(INTERIOR_GRID), "kn0": KNUDSEN,
            "cold_hot_ratio": COLD_HOT_RATIO, "rule": list(RULE), "radial_scale": RADIAL_SCALE,
            "limiter": LIMITER, "boundary_slope": BOUNDARY_SLOPE, "source_relaxation": SOURCE_RELAXATION,
            "tolerance": TOLERANCE, "correction_floor": CORRECTION_FLOOR, "diagnostic_steps": DIAGNOSTIC_STEPS,
            "witness_node": WITNESS_NODE, "pair_sectors": list(PAIR_SECTORS), "dominant_mirrored_sector": DOMINANT_SECTOR,
            "plateau_depth_cells": PLATEAU_DEPTH_CELLS, "phase_offset_max_cells": PHASE_OFFSET_MAX_CELLS,
            "width_ratio_max": WIDTH_RATIO_MAX, "quantiles": list(QUANTILES),
            "solver_rerun": False, "solver_endpoint_advanced": False, "cross_knudsen_extension_permitted": False,
            "benchmark_or_validation_claim_permitted": False, "physical_parameter_retuning": False,
            "collision_source_retuning": False, "floor_retuning": False, "wall_retuning": False,
            "reconstruction_retuning": False, "transport_retuning": False, "limiter_retuning": False,
            "normalization_retuning": False, "source_relaxation_retuning": False, "velocity_grid_retuning": False,
        },
        "metrics": {
            "dominant_sector_wall0": _pub(d0), "dominant_sector_wall1_mirrored": _pub(d1),
            "dominant_zero_phase_offset_cells": dz, "dominant_midpoint_phase_offset_cells": dm,
            "dominant_width_25_75_ratio": dw25, "dominant_width_10_90_ratio": dw10,
            "dominant_dynamic_range_ratio": _ratio(float(d0["dynamic_range"]), float(d1["dynamic_range"])),
            "parent_wall0": _pub(p0), "parent_wall1": _pub(p1),
            "parent_zero_phase_offset_cells": pz, "parent_midpoint_phase_offset_cells": pm,
            "parent_width_25_75_ratio": pw25, "parent_width_10_90_ratio": pw10,
            "parent_dynamic_range_ratio": _ratio(float(p0["dynamic_range"]), float(p1["dynamic_range"])),
            "inherited_stage131_mirrored_transition_positive_scale_wall1_to_wall0": float(parent["metrics"]["mirrored_transition_positive_scale_wall1_to_wall0"]),
            "inherited_stage131_mirrored_transition_relative_l2_residual": float(parent["metrics"]["mirrored_transition_relative_l2_residual"]),
        },
        "aggregate": {"maximum_parent_closure": closure,
                      "maximum_phase_offset_cells": max(dz, dm, pz, pm),
                      "maximum_transition_width_ratio": max(dw25, dw10, pw25, pw10)},
        "decision": decision,
        "scientific_conclusion": conclusion + " This artifact-only result does not establish limiter causality, MUSCL stability, endpoint convergence, q_av improvement, benchmark accuracy, or validation.",
        "negative_result_guard": "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. No physical, collision/source, floor, wall, reconstruction, transport, limiter, normalization, source-relaxation, or velocity-quadrature parameter is retuned; no solver endpoint or cross-Knudsen extension is advanced.",
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        out / "crossing_phase_transition_width.npz",
        relative_depth=depth,
        dominant_wall0=a["wall0_sector_profiles"][:, DOMINANT_COMPONENT],
        dominant_wall1_mirrored=a["wall1_mirrored_sector_profiles"][:, DOMINANT_COMPONENT],
        parent_wall0=a["parent0_aligned"], parent_wall1=a["parent1_aligned"],
        quantile_levels=np.asarray(QUANTILES, dtype=np.float64),
        dominant_quantile_depths=np.vstack([d0["quantile_depths"], d1["quantile_depths"]]),
        parent_quantile_depths=np.vstack([p0["quantile_depths"], p1["quantile_depths"]]),
        dominant_progress=np.vstack([d0["progress"], d1["progress"]]),
        parent_progress=np.vstack([p0["progress"], p1["progress"]]),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 132 crossing-phase transition-width audit")
    parser.add_argument("--stage131-dir", required=True)
    parser.add_argument("--stage131-record", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.stage131_dir, args.stage131_record, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
