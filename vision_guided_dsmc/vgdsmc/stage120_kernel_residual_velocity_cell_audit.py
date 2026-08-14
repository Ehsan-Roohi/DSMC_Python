from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import stage110_same_sign_slope_asymmetry_audit as s110
from . import stage114_wall_distance_conditioned_velocity_quadrature_audit as s114
from . import stage119_exact_directional_moment_kernel_audit as s119

STAGE119_RUN_ID = 31807927456
STAGE119_JOB_ID = 94791241679
STAGE119_ARTIFACT_ID = 9230035282
STAGE119_ARTIFACT_SHA256 = "a31c7cab7ff95e0ff68d00e84927e3c16292f03331db225ae92ea2efdc49063d"
STAGE119_SUMMARY_SHA256 = "3158721e18b73a7aecf65259185352becbcf2c1d927845c9afddb8dac04a5088"
STAGE119_PROFILES_SHA256 = "49d91f353d5bfbfaa41bf17f2a745f3762a41a814278fa9bbf0980e1645ac326"
STAGE119_SOURCE_HEAD = "2ec3ba831e74e9e7b92f122aa1611cc0fa61ee43"
STAGE119_DECISION = s119.INCOMPLETE

GRID = s119.GRID
KNUDSEN = s119.KNUDSEN
COLD_HOT_RATIO = s119.COLD_HOT_RATIO
RULE = s119.RULE
RADIAL_SCALE = s119.RADIAL_SCALE
LIMITER = s119.LIMITER
BOUNDARY_SLOPE = s119.BOUNDARY_SLOPE
SOURCE_RELAXATION = s119.SOURCE_RELAXATION
TOLERANCE = s119.TOLERANCE
CORRECTION_FLOOR = s119.CORRECTION_FLOOR
DIAGNOSTIC_STEPS = s119.DIAGNOSTIC_STEPS
WALL_BAND_CELLS = s119.WALL_BAND_CELLS
DOMINANT_RADIAL_SHELL = s119.DOMINANT_RADIAL_SHELL
RADIAL_SHELL_COUNT = s119.RADIAL_SHELL_COUNT
RADIAL_NODES_PER_SHELL = s119.RADIAL_NODES_PER_SHELL
ANGULAR_SECTORS = s119.ANGULAR_SECTORS
PAIR_SECTORS = s119.PAIR_SECTORS
BANDS = s119.BANDS
PARENT_CLOSURE_TOLERANCE = 1.0e-12
RADIAL_CAPTURE_CLOSURE_TOLERANCE = 1.0e-12
VELOCITY_CELL_SHAPE = (RADIAL_NODES_PER_SHELL, len(PAIR_SECTORS))

RADIAL = "stage120_exact_kernel_residual_is_radial_within_pair_stage121_radial_kernel_factor_decomposition_audit"
MIXED = "stage120_exact_kernel_residual_has_within_pair_angular_cancellation_stage121_residual_angle_node_interaction_audit"
NONFINITE = "stage120_nonfinite_kernel_residual_velocity_cell_blocker_without_retuning"
CLOSURE_BLOCKER = "stage120_stage119_velocity_cell_marginal_closure_blocker_without_retuning"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_stage120_design(**overrides: object) -> None:
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
        "velocity_cell_shape": VELOCITY_CELL_SHAPE,
        "parent_closure_tolerance": PARENT_CLOSURE_TOLERANCE,
        "radial_capture_closure_tolerance": RADIAL_CAPTURE_CLOSURE_TOLERANCE,
        "stage67_run_id": s110.STAGE67_RUN_ID,
        "stage111_run_id": s114.STAGE111_RUN_ID,
        "stage119_run_id": STAGE119_RUN_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 120 is fixed to the exact Stage-119 x-directed moment kernel on the existing shell-1 sectors 5+6 support; "
            "it may not retune physics, wall treatment, collision/source treatment, reconstruction, floors, source relaxation, "
            "transport, normalization, velocity quadrature, diagnostic window, or any failed MUSCL parameter"
        )
    if RULE != (40, 96) or RADIAL_NODES_PER_SHELL != 10 or PAIR_SECTORS != (5, 6):
        raise ValueError("Stage 120 requires the exact 40x96 rule, ten shell-1 radial nodes, and sectors 5+6")
    if VELOCITY_CELL_SHAPE != (10, 2):
        raise ValueError("Stage 120 requires exactly twenty fixed radial-node/sector velocity cells")


def _load_stage119(parent_dir: str | Path, record_path: str | Path):
    root = Path(parent_dir)
    expected = {
        "summary.json": STAGE119_SUMMARY_SHA256,
        "exact_directional_moment_kernel_profiles.npz": STAGE119_PROFILES_SHA256,
    }
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Stage-119 checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    if summary.get("stage") != 119 or summary.get("decision") != STAGE119_DECISION or summary.get("finite") is not True:
        raise ValueError("Stage-119 artifact does not authorize Stage 120")
    checks = (
        record.get("stage") == 119,
        record.get("decision") == STAGE119_DECISION,
        record.get("source_head") == STAGE119_SOURCE_HEAD,
        record.get("workflow_status") == "completed",
        record.get("workflow_conclusion") == "success",
        record.get("workflow_run_id") == STAGE119_RUN_ID,
        record.get("workflow_job_id") == STAGE119_JOB_ID,
        record.get("artifact_id") == STAGE119_ARTIFACT_ID,
        record.get("artifact_sha256") == STAGE119_ARTIFACT_SHA256,
        record.get("summary_sha256") == STAGE119_SUMMARY_SHA256,
        record.get("exact_directional_moment_kernel_profiles_sha256") == STAGE119_PROFILES_SHA256,
        record.get("tests", {}).get("passed") == 4,
        record.get("tests", {}).get("failed") == 0,
    )
    if not all(checks):
        raise ValueError("Committed Stage-119 provenance does not authorize Stage 120")
    with np.load(root / "exact_directional_moment_kernel_profiles.npz") as data:
        needed = {"exact_phi", "exact_psi"}
        if not needed.issubset(data.files):
            raise ValueError("Stage-119 exact profile payload is incomplete")
        arrays = {name: np.asarray(data[name], dtype=np.float64).copy() for name in needed}
    for name, value in arrays.items():
        if value.shape != (3, 10) or not np.isfinite(value).all() or np.any(value < 0.0):
            raise ValueError(f"Stage-119 profile {name} is invalid")
    return summary, record, arrays


def _normalize_cells(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64)
    total = float(x.sum())
    if x.shape != VELOCITY_CELL_SHAPE or not np.isfinite(x).all() or np.any(x < 0.0) or total <= 0.0:
        raise ValueError("Invalid Stage-120 velocity-cell profile")
    return x / total


def _profile_metrics(phi: np.ndarray, psi: np.ndarray) -> dict[str, float]:
    p = np.asarray(phi, dtype=np.float64)
    q = np.asarray(psi, dtype=np.float64)
    if p.shape != q.shape or p.size == 0 or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Stage-120 profile metric inputs are invalid")
    p = p / max(float(p.sum()), 1.0e-300)
    q = q / max(float(q.sum()), 1.0e-300)
    d = p - q
    return {
        "profile_cosine": float(np.dot(p.ravel(), q.ravel()) / max(float(np.linalg.norm(p) * np.linalg.norm(q)), 1.0e-300)),
        "overlap_coefficient": float(np.minimum(p, q).sum()),
        "total_variation_distance": float(0.5 * np.abs(d).sum()),
    }


def _top_absolute_residual_share(phi: np.ndarray, psi: np.ndarray, count: int) -> float:
    p = np.asarray(phi, dtype=np.float64)
    q = np.asarray(psi, dtype=np.float64)
    p = p / max(float(p.sum()), 1.0e-300)
    q = q / max(float(q.sum()), 1.0e-300)
    residual = np.sort(np.abs(p - q).ravel())[::-1]
    total = float(residual.sum())
    return float(residual[:count].sum() / max(total, 1.0e-300))


def stage120_decision(metrics: dict[str, dict[str, float]], finite: bool, parent_closure: float) -> str:
    if not finite:
        return NONFINITE
    if parent_closure > PARENT_CLOSURE_TOLERANCE:
        return CLOSURE_BLOCKER
    maximum_radial_loss = max(abs(1.0 - float(metrics[band]["radial_tv_capture_fraction"])) for band in BANDS)
    return RADIAL if maximum_radial_loss <= RADIAL_CAPTURE_CLOSURE_TOLERANCE else MIXED


def run(
    stage67_dir: str | Path,
    stage111_dir: str | Path,
    stage119_dir: str | Path,
    stage119_record_path: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage120_design(**design)
    s67_summary, distributions = s110._load_stage67(stage67_dir)
    s111_summary, maps = s114._load_stage111(stage111_dir)
    s119_summary, s119_record, parent = _load_stage119(stage119_dir, stage119_record_path)

    velocity_cells: dict[str, list[np.ndarray]] = {"phi": [], "psi": []}
    band_masks = s114.wall_distance_band_masks()
    with np.load(distributions) as data:
        vx = np.asarray(data["vx"], dtype=np.float64)
        vy = np.asarray(data["vy"], dtype=np.float64)
        weight = np.asarray(data["weight"], dtype=np.float64)
        shell = s110._radial_shell_indices(vx, vy) == DOMINANT_RADIAL_SHELL
        svx, svy, sw = vx[shell], vy[shell], weight[shell]
        sector = s114.angular_sector_indices(svx, svy)
        node = s119.radial_node_indices_within_shell(svx, svy)
        for j in range(RADIAL_NODES_PER_SHELL):
            for k in PAIR_SECTORS:
                if np.count_nonzero((node == j) & (sector == k)) != 12:
                    raise ValueError("Each Stage-120 radial-node/sector cell must contain exactly twelve angular ordinates")
        kernels = {
            "phi": np.abs(svx) * (svx * svx + svy * svy),
            "psi": np.abs(svx),
        }
        for name in ("phi", "psi"):
            change = s119._x_same_sign_change_pointwise(np.asarray(data[name], dtype=np.float64)[..., shell])
            density = change * sw[None, None, :] * maps[f"{name}_growth_amplitude"][..., None]
            for band in BANDS:
                cells = density[band_masks[band]]
                profile = np.empty(VELOCITY_CELL_SHAPE, dtype=np.float64)
                for j in range(RADIAL_NODES_PER_SHELL):
                    for kk, sector_id in enumerate(PAIR_SECTORS):
                        mask = (node == j) & (sector == sector_id)
                        profile[j, kk] = float(np.sum(cells[:, mask] * kernels[name][mask][None, :]))
                velocity_cells[name].append(_normalize_cells(profile))

    phi_cells = np.asarray(velocity_cells["phi"], dtype=np.float64)
    psi_cells = np.asarray(velocity_cells["psi"], dtype=np.float64)
    parent_closure = {
        "phi": float(np.linalg.norm(phi_cells.sum(axis=2) - parent["exact_phi"]) / max(float(np.linalg.norm(parent["exact_phi"])), 1.0e-300)),
        "psi": float(np.linalg.norm(psi_cells.sum(axis=2) - parent["exact_psi"]) / max(float(np.linalg.norm(parent["exact_psi"])), 1.0e-300)),
    }
    maximum_parent_closure = max(parent_closure.values())

    metrics: dict[str, dict[str, float]] = {}
    for i, band in enumerate(BANDS):
        full = _profile_metrics(phi_cells[i], psi_cells[i])
        radial = _profile_metrics(phi_cells[i].sum(axis=1), psi_cells[i].sum(axis=1))
        sector = _profile_metrics(phi_cells[i].sum(axis=0), psi_cells[i].sum(axis=0))
        full_tv = max(float(full["total_variation_distance"]), 1.0e-300)
        metrics[band] = {
            "full_velocity_cell_profile_cosine": float(full["profile_cosine"]),
            "full_velocity_cell_overlap": float(full["overlap_coefficient"]),
            "full_velocity_cell_total_variation": float(full["total_variation_distance"]),
            "radial_node_marginal_profile_cosine": float(radial["profile_cosine"]),
            "radial_node_marginal_total_variation": float(radial["total_variation_distance"]),
            "pair_sector_marginal_profile_cosine": float(sector["profile_cosine"]),
            "pair_sector_marginal_total_variation": float(sector["total_variation_distance"]),
            "radial_tv_capture_fraction": float(radial["total_variation_distance"] / full_tv),
            "sector_tv_capture_fraction": float(sector["total_variation_distance"] / full_tv),
            "within_node_sign_cancellation_fraction": float(max(0.0, 1.0 - radial["total_variation_distance"] / full_tv)),
            "top_4_absolute_residual_share": _top_absolute_residual_share(phi_cells[i], psi_cells[i], 4),
            "phi_sector5_share": float(phi_cells[i, :, 0].sum()),
            "psi_sector5_share": float(psi_cells[i, :, 0].sum()),
        }

    finite = bool(
        np.isfinite(phi_cells).all()
        and np.isfinite(psi_cells).all()
        and np.isfinite(maximum_parent_closure)
        and all(np.isfinite(list(m.values())).all() for m in metrics.values())
    )
    decision = stage120_decision(metrics, finite, maximum_parent_closure)
    aggregate = {
        "minimum_full_velocity_cell_profile_cosine": float(min(metrics[b]["full_velocity_cell_profile_cosine"] for b in BANDS)),
        "maximum_full_velocity_cell_total_variation": float(max(metrics[b]["full_velocity_cell_total_variation"] for b in BANDS)),
        "minimum_radial_tv_capture_fraction": float(min(metrics[b]["radial_tv_capture_fraction"] for b in BANDS)),
        "maximum_within_node_sign_cancellation_fraction": float(max(metrics[b]["within_node_sign_cancellation_fraction"] for b in BANDS)),
        "maximum_sector_tv_capture_fraction": float(max(metrics[b]["sector_tv_capture_fraction"] for b in BANDS)),
        "maximum_top_4_absolute_residual_share": float(max(metrics[b]["top_4_absolute_residual_share"] for b in BANDS)),
        "maximum_stage119_node_marginal_reconstruction_relative_l2": float(maximum_parent_closure),
    }

    configuration = {
        "grid": list(GRID), "kn0": KNUDSEN, "cold_hot_ratio": COLD_HOT_RATIO, "rule": list(RULE),
        "radial_scale": RADIAL_SCALE, "limiter": LIMITER, "boundary_slope": BOUNDARY_SLOPE,
        "source_relaxation": SOURCE_RELAXATION, "tolerance": TOLERANCE, "correction_floor": CORRECTION_FLOOR,
        "diagnostic_steps": DIAGNOSTIC_STEPS, "wall_band_cells": WALL_BAND_CELLS,
        "dominant_radial_shell": DOMINANT_RADIAL_SHELL, "radial_shell_count": RADIAL_SHELL_COUNT,
        "radial_nodes_per_shell": RADIAL_NODES_PER_SHELL, "angular_sectors": ANGULAR_SECTORS,
        "pair_sectors": list(PAIR_SECTORS), "velocity_cell_shape": list(VELOCITY_CELL_SHAPE),
        "phi_exact_kernel": "abs(c_x)*(c_x^2+c_y^2)", "psi_exact_kernel": "abs(c_x)",
        "parent_closure_tolerance": PARENT_CLOSURE_TOLERANCE,
        "radial_capture_closure_tolerance": RADIAL_CAPTURE_CLOSURE_TOLERANCE,
        "model_retuning": False, "wall_retuning": False, "reconstruction_retuning": False,
        "source_retuning": False, "transport_retuning": False, "floor_retuning": False,
        "normalization_retuning": False, "velocity_grid_retuning": False,
        "failed_muscl_endpoint_rehabilitated": False, "stage89_one_sided_boundary_promoted": False,
        "cross_knudsen_extension_permitted": False, "solver_endpoint_advanced": False,
        "benchmark_or_validation_claim_permitted": False,
    }
    scientific_conclusion = (
        "Stage 120 resolves the remaining Stage-119 exact-kernel phi/psi discrepancy into the fixed 20 radial-node/sector cells "
        "inside shell 1 and sectors 5+6. If the radial-node marginal retains the full velocity-cell total-variation distance to "
        "closure, the discrepancy is radial-speed structured and is not produced by cancellation between sectors 5 and 6. "
        "If it does not, a node-angle interaction remains. Either outcome is diagnostic only and does not establish MUSCL-instability "
        "causality, solver convergence, q_av improvement, benchmark accuracy, or validation."
    )
    negative_result_guard = (
        "No model, wall, collision/source, reconstruction, transport, positivity/correction floor, normalization, source-relaxation, "
        "velocity-grid, or failed MUSCL parameter is changed. Stage 90 remains nonconverged, Stage 28 remains failed, Stage 89 remains "
        "unpromoted, and no solver endpoint or cross-Knudsen extension is attempted."
    )
    summary: dict[str, object] = {
        "stage": 120,
        "decision": decision,
        "finite": finite,
        "configuration": configuration,
        "parent_stage119": {"run_id": STAGE119_RUN_ID, "job_id": STAGE119_JOB_ID, "artifact_id": STAGE119_ARTIFACT_ID, "source_head": STAGE119_SOURCE_HEAD},
        "parent_closure_relative_l2": parent_closure,
        "metrics": metrics,
        "aggregate": aggregate,
        "scientific_conclusion": scientific_conclusion,
        "negative_result_guard": negative_result_guard,
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        out / "kernel_residual_velocity_cells.npz",
        phi_velocity_cells=phi_cells,
        psi_velocity_cells=psi_cells,
        pair_sectors=np.asarray(PAIR_SECTORS, dtype=np.int16),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed Stage-120 exact-kernel residual velocity-cell audit")
    parser.add_argument("--stage67-dir", required=True)
    parser.add_argument("--stage111-dir", required=True)
    parser.add_argument("--stage119-dir", required=True)
    parser.add_argument("--stage119-record", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = run(args.stage67_dir, args.stage111_dir, args.stage119_dir, args.stage119_record, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
