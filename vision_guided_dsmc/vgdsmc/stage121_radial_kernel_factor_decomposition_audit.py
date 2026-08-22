from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import stage110_same_sign_slope_asymmetry_audit as s110
from . import stage119_exact_directional_moment_kernel_audit as s119
from . import stage120_kernel_residual_velocity_cell_audit as s120

STAGE120_RUN_ID = 31831746351
STAGE120_JOB_ID = 94868814023
STAGE120_ARTIFACT_ID = 9237037498
STAGE120_ARTIFACT_SHA256 = "348b4165582e325069b637d2dae74fe0e698214871ef37e9de39258093987f2d"
STAGE120_SUMMARY_SHA256 = "33feeea2dcc8660ece00bb9361d221eeb28df130a4ccec9439e8014c2dd76761"
STAGE120_CELLS_SHA256 = "a45ed560aa76d01d02d239efd3e9690d7385e538c932fbcb463e92259434aaca"
STAGE120_SOURCE_HEAD = "6f93b6b50659b7fa86630f57d0c24b3a9c6c8438"
STAGE120_DECISION = s120.RADIAL

GRID = s120.GRID
KNUDSEN = s120.KNUDSEN
COLD_HOT_RATIO = s120.COLD_HOT_RATIO
RULE = s120.RULE
RADIAL_SCALE = s120.RADIAL_SCALE
LIMITER = s120.LIMITER
BOUNDARY_SLOPE = s120.BOUNDARY_SLOPE
SOURCE_RELAXATION = s120.SOURCE_RELAXATION
TOLERANCE = s120.TOLERANCE
CORRECTION_FLOOR = s120.CORRECTION_FLOOR
DIAGNOSTIC_STEPS = s120.DIAGNOSTIC_STEPS
WALL_BAND_CELLS = s120.WALL_BAND_CELLS
DOMINANT_RADIAL_SHELL = s120.DOMINANT_RADIAL_SHELL
RADIAL_SHELL_COUNT = s120.RADIAL_SHELL_COUNT
RADIAL_NODES_PER_SHELL = s120.RADIAL_NODES_PER_SHELL
ANGULAR_SECTORS = s120.ANGULAR_SECTORS
PAIR_SECTORS = s120.PAIR_SECTORS
BANDS = s120.BANDS
PARENT_CLOSURE_TOLERANCE = 1.0e-12
KERNEL_MATERIAL_TV_REDUCTION_MIN = 0.20
KERNEL_DOMINANT_TV_REDUCTION_MIN = 0.75
KERNEL_DOMINANT_PREDICTED_OVERLAP_MIN = 0.95

DOMINANT = "stage121_radial_speed_squared_kernel_dominates_stage122_common_kernel_residual_floor_audit"
MATERIAL = "stage121_radial_speed_squared_kernel_material_but_incomplete_stage122_common_kernel_distribution_ratio_audit"
WEAK = "stage121_radial_speed_squared_kernel_weak_stage122_common_kernel_distribution_ratio_audit"
NONFINITE = "stage121_nonfinite_radial_kernel_factor_blocker_without_retuning"
CLOSURE_BLOCKER = "stage121_stage120_payload_reconstruction_blocker_without_retuning"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_stage121_design(**overrides: object) -> None:
    frozen = {
        "grid": GRID,
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
        "wall_band_cells": WALL_BAND_CELLS,
        "dominant_radial_shell": DOMINANT_RADIAL_SHELL,
        "radial_shell_count": RADIAL_SHELL_COUNT,
        "radial_nodes_per_shell": RADIAL_NODES_PER_SHELL,
        "angular_sectors": ANGULAR_SECTORS,
        "pair_sectors": PAIR_SECTORS,
        "parent_closure_tolerance": PARENT_CLOSURE_TOLERANCE,
        "kernel_material_tv_reduction_min": KERNEL_MATERIAL_TV_REDUCTION_MIN,
        "kernel_dominant_tv_reduction_min": KERNEL_DOMINANT_TV_REDUCTION_MIN,
        "kernel_dominant_predicted_overlap_min": KERNEL_DOMINANT_PREDICTED_OVERLAP_MIN,
        "stage67_run_id": s110.STAGE67_RUN_ID,
        "stage120_run_id": STAGE120_RUN_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 121 is fixed to the completed Stage-120 radial profiles and exact Stage-67 shell-1 radial nodes; "
            "it may not retune physics, wall/collision/source treatment, reconstruction, transport, floors, normalization, "
            "source relaxation, velocity quadrature, diagnostic window, decision guards, or any failed MUSCL parameter"
        )
    if RULE != (40, 96) or RADIAL_NODES_PER_SHELL != 10 or PAIR_SECTORS != (5, 6):
        raise ValueError("Stage 121 requires the exact 40x96 rule, ten shell-1 radial nodes, and sectors 5+6")


def _load_stage120(root: str | Path, record_path: str | Path):
    root = Path(root)
    expected = {
        "summary.json": STAGE120_SUMMARY_SHA256,
        "kernel_residual_velocity_cells.npz": STAGE120_CELLS_SHA256,
    }
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Stage-120 checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    if summary.get("stage") != 120 or summary.get("decision") != STAGE120_DECISION or summary.get("finite") is not True:
        raise ValueError("Stage-120 artifact does not authorize Stage 121")
    checks = (
        record.get("stage") == 120,
        record.get("decision") == STAGE120_DECISION,
        record.get("source_head") == STAGE120_SOURCE_HEAD,
        record.get("workflow_status") == "completed",
        record.get("workflow_conclusion") == "success",
        record.get("workflow_run_id") == STAGE120_RUN_ID,
        record.get("workflow_job_id") == STAGE120_JOB_ID,
        record.get("artifact_id") == STAGE120_ARTIFACT_ID,
        record.get("artifact_sha256") == STAGE120_ARTIFACT_SHA256,
        record.get("summary_sha256") == STAGE120_SUMMARY_SHA256,
        record.get("kernel_residual_velocity_cells_sha256") == STAGE120_CELLS_SHA256,
        record.get("tests", {}).get("passed") == 4,
        record.get("tests", {}).get("failed") == 0,
    )
    if not all(checks):
        raise ValueError("Committed Stage-120 provenance does not authorize Stage 121")
    with np.load(root / "kernel_residual_velocity_cells.npz") as data:
        needed = {"phi_velocity_cells", "psi_velocity_cells", "pair_sectors"}
        if not needed.issubset(data.files):
            raise ValueError("Stage-120 velocity-cell payload is incomplete")
        phi = np.asarray(data["phi_velocity_cells"], dtype=np.float64).copy()
        psi = np.asarray(data["psi_velocity_cells"], dtype=np.float64).copy()
        pair = tuple(int(v) for v in np.asarray(data["pair_sectors"]).tolist())
    if phi.shape != (3, 10, 2) or psi.shape != (3, 10, 2) or pair != PAIR_SECTORS:
        raise ValueError("Stage-120 velocity-cell payload has the wrong shape or sector pair")
    if not np.isfinite(phi).all() or not np.isfinite(psi).all() or np.any(phi < 0.0) or np.any(psi < 0.0):
        raise ValueError("Stage-120 velocity-cell payload is invalid")
    return summary, record, phi, psi


def _normalize10(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64)
    total = float(x.sum())
    if x.shape != (10,) or not np.isfinite(x).all() or np.any(x < 0.0) or total <= 0.0:
        raise ValueError("Invalid Stage-121 radial profile")
    return x / total


def _profile_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    p, q = _normalize10(a), _normalize10(b)
    return {
        "profile_cosine": float(np.dot(p, q) / max(float(np.linalg.norm(p) * np.linalg.norm(q)), 1.0e-300)),
        "overlap_coefficient": float(np.minimum(p, q).sum()),
        "total_variation_distance": float(0.5 * np.abs(p - q).sum()),
        "first_centroid_node": float(np.dot(p, np.arange(10, dtype=np.float64))),
        "second_centroid_node": float(np.dot(q, np.arange(10, dtype=np.float64))),
    }


def _radial_node_speed_squared(stage67_dir: str | Path) -> np.ndarray:
    _, distributions = s110._load_stage67(stage67_dir)
    with np.load(distributions) as data:
        vx = np.asarray(data["vx"], dtype=np.float64)
        vy = np.asarray(data["vy"], dtype=np.float64)
    shell = s110._radial_shell_indices(vx, vy) == DOMINANT_RADIAL_SHELL
    svx, svy = vx[shell], vy[shell]
    node = s119.radial_node_indices_within_shell(svx, svy)
    speed2 = svx * svx + svy * svy
    out = np.empty(10, dtype=np.float64)
    for j in range(10):
        values = speed2[node == j]
        if values.size != 96 or float(values.max() - values.min()) > 1.0e-12 * max(float(values.mean()), 1.0):
            raise ValueError("Stage-121 radial node does not have a unique speed-squared factor")
        out[j] = float(values.mean())
    if not np.all(np.diff(out) > 0.0):
        raise ValueError("Stage-121 radial speed-squared factors must be strictly increasing")
    return out


def stage121_decision(metrics: dict[str, dict[str, float]], finite: bool, parent_closure: float) -> str:
    if not finite:
        return NONFINITE
    if parent_closure > PARENT_CLOSURE_TOLERANCE:
        return CLOSURE_BLOCKER
    reductions = [float(metrics[b]["kernel_counterfactual_tv_reduction_fraction"]) for b in BANDS]
    overlaps = [float(metrics[b]["kernel_counterfactual_overlap"]) for b in BANDS]
    if all(r >= KERNEL_DOMINANT_TV_REDUCTION_MIN for r in reductions) and all(o >= KERNEL_DOMINANT_PREDICTED_OVERLAP_MIN for o in overlaps):
        return DOMINANT
    if all(r >= KERNEL_MATERIAL_TV_REDUCTION_MIN for r in reductions):
        return MATERIAL
    return WEAK


def run(stage67_dir: str | Path, stage120_dir: str | Path, stage120_record_path: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage121_design(**design)
    parent_summary, _, phi_cells, psi_cells = _load_stage120(stage120_dir, stage120_record_path)
    r2 = _radial_node_speed_squared(stage67_dir)
    phi_exact = np.asarray([_normalize10(x.sum(axis=1)) for x in phi_cells])
    psi_exact = np.asarray([_normalize10(x.sum(axis=1)) for x in psi_cells])
    phi_common = np.asarray([_normalize10(x / r2) for x in phi_exact])
    phi_from_psi_r2 = np.asarray([_normalize10(x * r2) for x in psi_exact])

    parent_closure = 0.0
    metrics: dict[str, dict[str, float]] = {}
    for i, band in enumerate(BANDS):
        original = _profile_metrics(phi_exact[i], psi_exact[i])
        common = _profile_metrics(phi_common[i], psi_exact[i])
        counterfactual = _profile_metrics(phi_exact[i], phi_from_psi_r2[i])
        parent_band = parent_summary["metrics"][band]
        parent_closure = max(
            parent_closure,
            abs(float(original["profile_cosine"]) - float(parent_band["radial_node_marginal_profile_cosine"])),
            abs(float(original["total_variation_distance"]) - float(parent_band["radial_node_marginal_total_variation"])),
        )
        original_tv = max(float(original["total_variation_distance"]), 1.0e-300)
        metrics[band] = {
            "original_exact_profile_cosine": float(original["profile_cosine"]),
            "original_exact_overlap": float(original["overlap_coefficient"]),
            "original_exact_total_variation": float(original["total_variation_distance"]),
            "common_abs_cx_profile_cosine": float(common["profile_cosine"]),
            "common_abs_cx_overlap": float(common["overlap_coefficient"]),
            "common_abs_cx_total_variation": float(common["total_variation_distance"]),
            "common_abs_cx_tv_change_fraction": float((original_tv - float(common["total_variation_distance"])) / original_tv),
            "kernel_counterfactual_profile_cosine": float(counterfactual["profile_cosine"]),
            "kernel_counterfactual_overlap": float(counterfactual["overlap_coefficient"]),
            "kernel_counterfactual_total_variation": float(counterfactual["total_variation_distance"]),
            "kernel_counterfactual_tv_reduction_fraction": float((original_tv - float(counterfactual["total_variation_distance"])) / original_tv),
            "phi_exact_centroid_node": float(original["first_centroid_node"]),
            "psi_exact_centroid_node": float(original["second_centroid_node"]),
            "phi_common_abs_cx_centroid_node": float(common["first_centroid_node"]),
            "psi_common_abs_cx_centroid_node": float(common["second_centroid_node"]),
            "kernel_counterfactual_phi_centroid_node": float(counterfactual["second_centroid_node"]),
        }

    finite = bool(
        np.isfinite(phi_exact).all()
        and np.isfinite(psi_exact).all()
        and np.isfinite(phi_common).all()
        and np.isfinite(phi_from_psi_r2).all()
        and np.isfinite(r2).all()
        and np.isfinite(parent_closure)
    )
    decision = stage121_decision(metrics, finite, parent_closure)
    aggregate = {
        "minimum_kernel_counterfactual_tv_reduction_fraction": float(min(metrics[b]["kernel_counterfactual_tv_reduction_fraction"] for b in BANDS)),
        "maximum_kernel_counterfactual_tv_reduction_fraction": float(max(metrics[b]["kernel_counterfactual_tv_reduction_fraction"] for b in BANDS)),
        "minimum_kernel_counterfactual_profile_cosine": float(min(metrics[b]["kernel_counterfactual_profile_cosine"] for b in BANDS)),
        "minimum_kernel_counterfactual_overlap": float(min(metrics[b]["kernel_counterfactual_overlap"] for b in BANDS)),
        "maximum_common_abs_cx_total_variation": float(max(metrics[b]["common_abs_cx_total_variation"] for b in BANDS)),
        "minimum_common_abs_cx_profile_cosine": float(min(metrics[b]["common_abs_cx_profile_cosine"] for b in BANDS)),
        "maximum_stage120_metric_reconstruction_absolute_error": float(parent_closure),
    }
    configuration = {
        "grid": list(GRID),
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
        "wall_band_cells": WALL_BAND_CELLS,
        "dominant_radial_shell": DOMINANT_RADIAL_SHELL,
        "radial_shell_count": RADIAL_SHELL_COUNT,
        "radial_nodes_per_shell": RADIAL_NODES_PER_SHELL,
        "angular_sectors": ANGULAR_SECTORS,
        "pair_sectors": list(PAIR_SECTORS),
        "common_kernel": "abs(c_x)",
        "phi_extra_radial_factor": "c_x^2+c_y^2=r^2",
        "parent_closure_tolerance": PARENT_CLOSURE_TOLERANCE,
        "kernel_material_tv_reduction_min": KERNEL_MATERIAL_TV_REDUCTION_MIN,
        "kernel_dominant_tv_reduction_min": KERNEL_DOMINANT_TV_REDUCTION_MIN,
        "kernel_dominant_predicted_overlap_min": KERNEL_DOMINANT_PREDICTED_OVERLAP_MIN,
        "model_retuning": False,
        "wall_retuning": False,
        "reconstruction_retuning": False,
        "source_retuning": False,
        "transport_retuning": False,
        "floor_retuning": False,
        "normalization_retuning": False,
        "velocity_grid_retuning": False,
        "failed_muscl_endpoint_rehabilitated": False,
        "stage89_one_sided_boundary_promoted": False,
        "cross_knudsen_extension_permitted": False,
        "solver_endpoint_advanced": False,
        "benchmark_or_validation_claim_permitted": False,
    }
    summary: dict[str, object] = {
        "stage": 121,
        "decision": decision,
        "finite": finite,
        "configuration": configuration,
        "parent_stage120": {"run_id": STAGE120_RUN_ID, "job_id": STAGE120_JOB_ID, "artifact_id": STAGE120_ARTIFACT_ID, "source_head": STAGE120_SOURCE_HEAD},
        "parent_reconstruction_absolute_error": float(parent_closure),
        "node_speed_squared": r2.tolist(),
        "metrics": metrics,
        "aggregate": aggregate,
        "scientific_conclusion": "Stage 121 separates the Stage-120 radial phi/psi mismatch into the fixed extra r^2 factor in the phi x-directed heat-flux kernel and the residual difference between phi and psi radial distributions under the common abs(c_x) kernel. A material counterfactual improvement means the exact moment kernel matters; incomplete agreement leaves distribution-specific radial structure that must be audited directly. This is attribution only, not a solver or benchmark improvement claim.",
        "negative_result_guard": "No model, wall, collision/source, reconstruction, transport, positivity/correction floor, normalization, source-relaxation, velocity-grid, or failed MUSCL parameter is changed. Stage 90 remains nonconverged, Stage 28 remains failed, Stage 89 remains unpromoted, and no solver endpoint or cross-Knudsen extension is attempted.",
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        out / "radial_kernel_factor_profiles.npz",
        phi_exact_radial=phi_exact,
        psi_exact_radial=psi_exact,
        phi_common_abs_cx=phi_common,
        phi_from_psi_r2=phi_from_psi_r2,
        node_speed_squared=r2,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed Stage-121 radial heat-flux kernel-factor decomposition audit")
    parser.add_argument("--stage67-dir", required=True)
    parser.add_argument("--stage120-dir", required=True)
    parser.add_argument("--stage120-record", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.stage67_dir, args.stage120_dir, args.stage120_record, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
