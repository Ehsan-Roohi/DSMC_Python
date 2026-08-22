from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from .stage34_velocity_scale_consistency import local_relaxation_time
from .stage41_projected_polar_operator_audit import (
    STAGE41_CORRECTION_FLOOR,
    mapped_polar_quadrature,
    projected_macroscopic,
)
from .stage58_conservative_solver_64x64_confirmation import build_stage58_config
from .stage79_dominant_moment_radial_angular_gradient_audit import radial_shell_indices
from .stage90_single_condition_reconstruction_solver_ab_audit import (
    COLD_HOT_RATIO,
    GRID,
    KNUDSEN,
    LIMITER,
    RADIAL_SCALE,
    RULE,
    SOURCE_RELAXATION,
    TOLERANCE,
    _validate_stage67,
    steady_muscl_iteration_step,
)
from .stage97_muscl_correction_spatial_localization_audit import WALL_BAND_CELLS, _region_masks
from .stage98_directional_operator_growth_audit import DIAGNOSTIC_STEPS, muscl_correction_components
from .stage101_interior_velocity_sector_audit import BOUNDARY_SLOPE
from .stage102_radial_speed_shell_audit import (
    MATERIAL_SHELL_GROWTH_RATIO,
    RADIAL_NODES_PER_SHELL,
    RADIAL_SHELL_COUNT,
    SHELL_PARENT_CLOSURE_TOLERANCE,
)

STAGE67_RUN_ID = 30991124477
STAGE102_RUN_ID = 31431835887
STAGE102_JOB_ID = 93596839081
STAGE102_ARTIFACT_ID = 9084348663
STAGE102_ARTIFACT_SHA256 = "1a0879d38770eb92962a4d18665e3738bb20415658b03426d308f8a36c3b690a"
STAGE102_SUMMARY_SHA256 = "8f909a0db0390a235aeb8436c53a99bbfe791739d4b991e1584015484393ca66"
STAGE102_HISTORIES_SHA256 = "caf7177a57ff72685747190d3eac45778a434c2dbeec5cfc6b8ccf0a65d26a75"
STAGE102_DECISION = "stage102_common_dominant_radial_shell_1_stage103_shell_spatial_localization_audit"
DOMINANT_RADIAL_SHELL = 1
INTERIOR_TILE_COUNT_PER_AXIS = 4
INTERIOR_TILE_COUNT = INTERIOR_TILE_COUNT_PER_AXIS**2
INTERIOR_TILE_SIZE = (GRID[0] - 2 * WALL_BAND_CELLS) // INTERIOR_TILE_COUNT_PER_AXIS
SINGLE_TILE_SHARE_GUARD = 0.25
CONTIGUOUS_2X2_SHARE_GUARD = 0.50
SPATIAL_PARENT_CLOSURE_TOLERANCE = SHELL_PARENT_CLOSURE_TOLERANCE


def validate_stage103_design(**overrides: object) -> None:
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
        "correction_floor": STAGE41_CORRECTION_FLOOR,
        "diagnostic_steps": DIAGNOSTIC_STEPS,
        "wall_band_cells": WALL_BAND_CELLS,
        "dominant_radial_shell": DOMINANT_RADIAL_SHELL,
        "interior_tile_count_per_axis": INTERIOR_TILE_COUNT_PER_AXIS,
        "interior_tile_size": INTERIOR_TILE_SIZE,
        "single_tile_share_guard": SINGLE_TILE_SHARE_GUARD,
        "contiguous_2x2_share_guard": CONTIGUOUS_2X2_SHARE_GUARD,
        "material_shell_growth_ratio": MATERIAL_SHELL_GROWTH_RATIO,
        "spatial_parent_closure_tolerance": SPATIAL_PARENT_CLOSURE_TOLERANCE,
        "stage67_run_id": STAGE67_RUN_ID,
        "stage102_run_id": STAGE102_RUN_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 103 is frozen to the exact completed Stage-67/102 lineage and the Stage-102 "
            "dominant radial shell. The 56x56 interior is partitioned a priori into a fixed 4x4 "
            "equal-area tile grid. Physics, collision/source treatment, clipping or positivity "
            "floors, source relaxation, transport parameters, wall model, limiter, quadrature, "
            "normalization, tolerance, diagnostic window, and failed solver parameters may not "
            "be retuned."
        )
    interior_extent = GRID[0] - 2 * WALL_BAND_CELLS
    if GRID[0] != GRID[1] or interior_extent != GRID[1] - 2 * WALL_BAND_CELLS:
        raise ValueError("Stage 103 requires the exact square 64x64 parent grid")
    if interior_extent % INTERIOR_TILE_COUNT_PER_AXIS != 0:
        raise ValueError("Frozen Stage-103 interior must divide exactly into equal-area tiles")


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 1.0e-300))


def _load_and_validate_stage102(root: str | Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 102 or summary.get("decision") != STAGE102_DECISION:
        raise ValueError("Stage-102 artifact does not authorize the Stage-103 shell spatial-localization audit")
    cfg = summary.get("configuration", {})
    if not isinstance(cfg, dict):
        raise ValueError("Stage-102 configuration is missing")
    frozen_checks = {
        "grid": list(GRID),
        "kn0": KNUDSEN,
        "cold_hot_ratio": COLD_HOT_RATIO,
        "rule": list(RULE),
        "radial_scale": RADIAL_SCALE,
        "limiter": LIMITER,
        "boundary_slope": BOUNDARY_SLOPE,
        "source_relaxation": SOURCE_RELAXATION,
        "tolerance": TOLERANCE,
        "correction_floor": STAGE41_CORRECTION_FLOOR,
        "diagnostic_steps": DIAGNOSTIC_STEPS,
        "wall_band_cells": WALL_BAND_CELLS,
        "radial_shell_count": RADIAL_SHELL_COUNT,
        "radial_nodes_per_shell": RADIAL_NODES_PER_SHELL,
    }
    if any(cfg.get(key) != value for key, value in frozen_checks.items()):
        raise ValueError("Stage-102 artifact does not match the frozen Stage-103 parent design")
    if summary.get("finite") is not True or summary.get("executed_steps") != DIAGNOSTIC_STEPS:
        raise ValueError("Stage-102 audit did not complete the frozen diagnostic window")
    if float(summary.get("maximum_shell_parent_closure_relative", np.inf)) > SPATIAL_PARENT_CLOSURE_TOLERANCE:
        raise ValueError("Stage-102 shell-parent closure failed")
    shell_summary = summary.get("shell_summary", {})
    for distribution in ("phi", "psi"):
        records = shell_summary.get(distribution, []) if isinstance(shell_summary, dict) else []
        if len(records) != RADIAL_SHELL_COUNT or int(records[DOMINANT_RADIAL_SHELL].get("index", -1)) != DOMINANT_RADIAL_SHELL:
            raise ValueError("Stage-102 dominant-shell record is missing")
        if float(records[DOMINANT_RADIAL_SHELL]["abs_share"]["final"]) < 0.5:
            raise ValueError("Stage-102 dominant shell is not dominant in both distributions")
        if float(records[DOMINANT_RADIAL_SHELL]["weighted_abs"]["final_to_first_ratio"]) < MATERIAL_SHELL_GROWTH_RATIO:
            raise ValueError("Stage-102 dominant shell did not grow materially in both distributions")

    with np.load(root / "interior_radial_speed_shell_histories.npz") as data:
        histories = {
            key: np.asarray(data[key], dtype=np.float64).copy()
            for key in data.files
            if key.startswith("phi_") or key.startswith("psi_")
        }
    for distribution in ("phi", "psi"):
        name = f"{distribution}_shell_weighted_abs"
        if name not in histories or histories[name].shape != (DIAGNOSTIC_STEPS, RADIAL_SHELL_COUNT):
            raise ValueError(f"Stage-102 artifact is missing the frozen {name} history")
    return summary, histories


def _interior_tile_index(shape: tuple[int, int]) -> np.ndarray:
    if tuple(shape) != tuple(GRID):
        raise ValueError(f"Stage 103 requires the exact {GRID[0]}x{GRID[1]} grid")
    tiles = np.full(shape, -1, dtype=np.int64)
    lo = WALL_BAND_CELLS
    for row in range(INTERIOR_TILE_COUNT_PER_AXIS):
        ys = slice(lo + row * INTERIOR_TILE_SIZE, lo + (row + 1) * INTERIOR_TILE_SIZE)
        for col in range(INTERIOR_TILE_COUNT_PER_AXIS):
            xs = slice(lo + col * INTERIOR_TILE_SIZE, lo + (col + 1) * INTERIOR_TILE_SIZE)
            tiles[ys, xs] = row * INTERIOR_TILE_COUNT_PER_AXIS + col
    interior = _region_masks(shape, wall_band_cells=WALL_BAND_CELLS)["interior"]
    if not np.array_equal(tiles >= 0, interior):
        raise ValueError("Stage-103 equal-area tiles do not exactly partition the frozen interior")
    return tiles


def _tile_metrics_from_shell_term(shell_term: np.ndarray, shell_weight: np.ndarray, tile_index: np.ndarray) -> dict[str, np.ndarray | float]:
    term = np.asarray(shell_term, dtype=np.float64)
    weight = np.asarray(shell_weight, dtype=np.float64)
    tiles = np.asarray(tile_index, dtype=np.int64)
    if term.ndim != 3 or term.shape[:2] != tiles.shape or term.shape[-1] != weight.size:
        raise ValueError("Stage-103 shell-term, weight, or tile shapes are inconsistent")
    if not np.isfinite(term).all() or not np.isfinite(weight).all():
        raise ValueError("Stage-103 shell term or weights are nonfinite")
    cell_abs = np.sum(np.abs(term) * weight[None, None, :], axis=-1)
    cell_signed = np.sum(term * weight[None, None, :], axis=-1)
    tile_abs = np.zeros(INTERIOR_TILE_COUNT, dtype=np.float64)
    tile_signed = np.zeros(INTERIOR_TILE_COUNT, dtype=np.float64)
    for k in range(INTERIOR_TILE_COUNT):
        mask = tiles == k
        if int(np.sum(mask)) != INTERIOR_TILE_SIZE**2:
            raise ValueError(f"Stage-103 tile {k} does not have the frozen equal area")
        tile_abs[k] = float(np.sum(cell_abs[mask]))
        tile_signed[k] = float(np.sum(cell_signed[mask]))
    total_abs = float(np.sum(tile_abs))
    return {"tile_weighted_abs": tile_abs, "tile_weighted_signed": tile_signed, "tile_abs_share": tile_abs / max(total_abs, 1.0e-300), "tile_signed_to_abs_ratio": np.abs(tile_signed) / np.maximum(tile_abs, 1.0e-300), "interior_shell_weighted_abs": total_abs, "cell_abs": cell_abs, "cell_signed": cell_signed}


def _history_summary(values: np.ndarray) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {"first": float(a[0]), "final": float(a[-1]), "minimum": float(np.min(a)), "maximum": float(np.max(a)), "final_to_first_ratio": _safe_ratio(float(a[-1]), float(a[0])), "maximum_to_first_ratio": _safe_ratio(float(np.max(a)), float(a[0]))}


def _tile_summary(histories: dict[str, np.ndarray], distribution: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for k in range(INTERIOR_TILE_COUNT):
        row, col = divmod(k, INTERIOR_TILE_COUNT_PER_AXIS)
        out.append({"index": k, "row": row, "column": col, "cell_count": INTERIOR_TILE_SIZE**2, "weighted_abs": _history_summary(histories[f"{distribution}_tile_weighted_abs"][:, k]), "abs_share": _history_summary(histories[f"{distribution}_tile_abs_share"][:, k]), "weighted_signed": _history_summary(histories[f"{distribution}_tile_weighted_signed"][:, k]), "signed_to_abs_ratio": _history_summary(histories[f"{distribution}_tile_signed_to_abs_ratio"][:, k])})
    return out


def _best_common_contiguous_2x2(histories: dict[str, np.ndarray]) -> dict[str, object]:
    best: dict[str, object] | None = None
    for row in range(INTERIOR_TILE_COUNT_PER_AXIS - 1):
        for col in range(INTERIOR_TILE_COUNT_PER_AXIS - 1):
            tiles = [row * INTERIOR_TILE_COUNT_PER_AXIS + col, row * INTERIOR_TILE_COUNT_PER_AXIS + col + 1, (row + 1) * INTERIOR_TILE_COUNT_PER_AXIS + col, (row + 1) * INTERIOR_TILE_COUNT_PER_AXIS + col + 1]
            distributions: dict[str, dict[str, float]] = {}
            for distribution in ("phi", "psi"):
                abs_hist = histories[f"{distribution}_tile_weighted_abs"][:, tiles].sum(axis=1)
                share_hist = histories[f"{distribution}_tile_abs_share"][:, tiles].sum(axis=1)
                distributions[distribution] = {"final_share": float(share_hist[-1]), "weighted_abs_growth": _safe_ratio(float(abs_hist[-1]), float(abs_hist[0]))}
            score = min(distributions[d]["final_share"] for d in ("phi", "psi"))
            candidate = {"top_left_row": row, "top_left_column": col, "tiles": tiles, "score": float(score), "distributions": distributions}
            if best is None or float(candidate["score"]) > float(best["score"]):
                best = candidate
    assert best is not None
    return best


def _effective_tile_count(final_share: np.ndarray) -> float:
    p = np.asarray(final_share, dtype=np.float64)
    return _safe_ratio(1.0, float(np.sum(p * p)))


def stage103_decision(tile_summary: dict[str, list[dict[str, object]]], histories: dict[str, np.ndarray], maximum_stage102_history_closure_relative: float, finite: bool) -> str:
    if not finite:
        return "stage103_nonfinite_replay_blocker_without_retuning"
    if maximum_stage102_history_closure_relative > SPATIAL_PARENT_CLOSURE_TOLERANCE:
        return "stage103_stage102_shell_history_closure_blocker_without_retuning"
    common_tiles: list[tuple[float, int]] = []
    for k in range(INTERIOR_TILE_COUNT):
        share = min(float(tile_summary[d][k]["abs_share"]["final"]) for d in ("phi", "psi"))
        growth = min(float(tile_summary[d][k]["weighted_abs"]["final_to_first_ratio"]) for d in ("phi", "psi"))
        if share >= SINGLE_TILE_SHARE_GUARD and growth >= MATERIAL_SHELL_GROWTH_RATIO:
            common_tiles.append((share, k))
    if common_tiles:
        _, k = max(common_tiles)
        return f"stage103_common_localized_tile_{k}_stage104_local_spatial_gradient_audit"
    block = _best_common_contiguous_2x2(histories)
    block_share = min(float(block["distributions"][d]["final_share"]) for d in ("phi", "psi"))
    block_growth = min(float(block["distributions"][d]["weighted_abs_growth"]) for d in ("phi", "psi"))
    if block_share >= CONTIGUOUS_2X2_SHARE_GUARD and block_growth >= MATERIAL_SHELL_GROWTH_RATIO:
        return "stage103_common_localized_contiguous_2x2_" f"{block['top_left_row']}_{block['top_left_column']}_stage104_local_spatial_gradient_audit"
    return "stage103_spatially_diffuse_shell1_growth_stage104_interior_gradient_scale_audit"


def run_stage103(stage67_artifact_dir: str | Path, stage102_artifact_dir: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage103_design(**design)
    _validate_stage67(stage67_artifact_dir)
    stage102_summary, stage102_histories = _load_and_validate_stage102(stage102_artifact_dir)
    cfg = build_stage58_config()
    quadrature = mapped_polar_quadrature(*RULE, radial_scale=RADIAL_SCALE)
    shells = radial_shell_indices(quadrature.vx, quadrature.vy).astype(np.int64)
    shell_mask = shells == DOMINANT_RADIAL_SHELL
    if int(np.sum(shell_mask)) != RADIAL_NODES_PER_SHELL * RULE[1]:
        raise ValueError("Stage-103 dominant radial shell does not contain the frozen velocity-point count")
    tiles = _interior_tile_index((cfg.ny, cfg.nx))
    interior = tiles >= 0
    with np.load(Path(stage67_artifact_dir) / "converged_full_distributions.npz") as saved:
        phi = np.asarray(saved["phi"], dtype=np.float64).copy()
        psi = np.asarray(saved["psi"], dtype=np.float64).copy()
        for name, actual, expected in (("vx", np.asarray(saved["vx"]), np.asarray(quadrature.vx)), ("vy", np.asarray(saved["vy"]), np.asarray(quadrature.vy)), ("weight", np.asarray(saved["weight"]), np.asarray(quadrature.weight))):
            if not np.array_equal(actual, expected):
                raise ValueError(f"Stage-67 {name} does not exactly match the frozen Stage-103 quadrature")
    history_lists: dict[str, list[np.ndarray | float]] = {}
    for distribution in ("phi", "psi"):
        for key in ("tile_weighted_abs", "tile_weighted_signed", "tile_abs_share", "tile_signed_to_abs_ratio", "interior_shell_weighted_abs", "stage102_shell_history_closure_relative"):
            history_lists[f"{distribution}_{key}"] = []
    saved_maps: dict[str, np.ndarray] = {}
    finite = True
    for step in range(DIAGNOSTIC_STEPS):
        fields = projected_macroscopic(phi, psi, quadrature)
        tau = local_relaxation_time(fields["rho"], fields["T"], cfg, mapping="paper_consistent_c0")
        nu = 1.0 / np.maximum(tau, 1.0e-14)
        dx, dy = 1.0 / cfg.nx, 1.0 / cfg.ny
        ax = np.abs(quadrature.vx) / dx
        ay = np.abs(quadrature.vy) / dy
        denominator = nu[..., None] + ax[None, None, :] + ay[None, None, :]
        for distribution, field in (("phi", phi), ("psi", psi)):
            corr_x, corr_y = muscl_correction_components(field, quadrature.vx, quadrature.vy, dx, dy)
            corr_x += corr_y
            del corr_y
            corr_x /= denominator
            shell_term = corr_x[..., shell_mask]
            metrics = _tile_metrics_from_shell_term(shell_term, quadrature.weight[shell_mask], tiles)
            del corr_x, shell_term
            parent = float(stage102_histories[f"{distribution}_shell_weighted_abs"][step, DOMINANT_RADIAL_SHELL])
            closure = _safe_ratio(abs(float(metrics["interior_shell_weighted_abs"]) - parent), parent)
            for key in ("tile_weighted_abs", "tile_weighted_signed", "tile_abs_share", "tile_signed_to_abs_ratio", "interior_shell_weighted_abs"):
                value = metrics[key]
                history_lists[f"{distribution}_{key}"].append(np.asarray(value, dtype=np.float64).copy() if isinstance(value, np.ndarray) else float(value))
            history_lists[f"{distribution}_stage102_shell_history_closure_relative"].append(closure)
            if step == 0:
                saved_maps[f"first_{distribution}_shell1_cell_abs"] = np.asarray(metrics["cell_abs"], dtype=np.float64)
                saved_maps[f"first_{distribution}_shell1_cell_signed"] = np.asarray(metrics["cell_signed"], dtype=np.float64)
            if step == DIAGNOSTIC_STEPS - 1:
                saved_maps[f"final_{distribution}_shell1_cell_abs"] = np.asarray(metrics["cell_abs"], dtype=np.float64)
                saved_maps[f"final_{distribution}_shell1_cell_signed"] = np.asarray(metrics["cell_signed"], dtype=np.float64)
        gc.collect()
        phi, psi, _ = steady_muscl_iteration_step(phi, psi, cfg, quadrature, one_sided_x_boundary=False)
        finite = bool(np.isfinite(phi).all() and np.isfinite(psi).all())
        if not finite:
            break
        gc.collect()
    histories = {key: np.asarray(values, dtype=np.float64) for key, values in history_lists.items()}
    tile_summary = {distribution: _tile_summary(histories, distribution) for distribution in ("phi", "psi")}
    max_closure = max(float(np.max(histories[f"{distribution}_stage102_shell_history_closure_relative"])) for distribution in ("phi", "psi"))
    best_block = _best_common_contiguous_2x2(histories)
    decision = stage103_decision(tile_summary, histories, max_closure, finite)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "shell1_spatial_localization_histories.npz", **histories, **saved_maps, tile_index=tiles, interior_mask=interior, radial_shell_index=shells, dominant_shell_mask=shell_mask, vx=np.asarray(quadrature.vx, dtype=np.float64), vy=np.asarray(quadrature.vy, dtype=np.float64), weight=np.asarray(quadrature.weight, dtype=np.float64))
    result: dict[str, object] = {
        "stage": 103,
        "description": "Frozen 25-step physical-space localization of the Stage-102 common dominant radial shell (shell 1). The exact 56x56 interior is partitioned a priori into sixteen equal-area 14x14 tiles; no tile boundary, solver setting, physical parameter, or failed MUSCL parameter is tuned from the data.",
        "configuration": {"grid": list(GRID), "kn0": KNUDSEN, "cold_hot_ratio": COLD_HOT_RATIO, "rule": list(RULE), "radial_scale": RADIAL_SCALE, "limiter": LIMITER, "boundary_slope": BOUNDARY_SLOPE, "source_relaxation": SOURCE_RELAXATION, "tolerance": TOLERANCE, "correction_floor": STAGE41_CORRECTION_FLOOR, "diagnostic_steps": DIAGNOSTIC_STEPS, "wall_band_cells": WALL_BAND_CELLS, "dominant_radial_shell": DOMINANT_RADIAL_SHELL, "interior_tile_count_per_axis": INTERIOR_TILE_COUNT_PER_AXIS, "interior_tile_count": INTERIOR_TILE_COUNT, "interior_tile_size": INTERIOR_TILE_SIZE, "single_tile_share_guard": SINGLE_TILE_SHARE_GUARD, "contiguous_2x2_share_guard": CONTIGUOUS_2X2_SHARE_GUARD, "material_shell_growth_ratio": MATERIAL_SHELL_GROWTH_RATIO, "spatial_parent_closure_tolerance": SPATIAL_PARENT_CLOSURE_TOLERANCE, "stage67_run_id": STAGE67_RUN_ID, "stage102_run_id": STAGE102_RUN_ID, "stage102_job_id": STAGE102_JOB_ID, "stage102_artifact_id": STAGE102_ARTIFACT_ID, "stage102_artifact_sha256": STAGE102_ARTIFACT_SHA256, "full_solver_endpoint_rerun": False, "physical_parameter_retuning": False, "collision_parameter_retuning": False, "correction_floor_retuning": False, "positivity_floor_retuning": False, "source_relaxation_retuning": False, "transport_parameter_retuning": False, "wall_model_retuning": False, "normalization_retuning": False, "limiter_retuning": False, "velocity_quadrature_retuning": False, "failed_muscl_endpoint_rehabilitated": False, "one_sided_boundary_slope_promoted": False, "cross_knudsen_extension_permitted": False, "validation_claim_permitted": False, "solver_endpoint_claim_permitted": False},
        "stage102_authorization": {"decision": stage102_summary["decision"], "maximum_shell_parent_closure_relative": stage102_summary["maximum_shell_parent_closure_relative"], "dominant_shell_final_share": {distribution: stage102_summary["shell_summary"][distribution][DOMINANT_RADIAL_SHELL]["abs_share"]["final"] for distribution in ("phi", "psi")}, "dominant_shell_growth": {distribution: stage102_summary["shell_summary"][distribution][DOMINANT_RADIAL_SHELL]["weighted_abs"]["final_to_first_ratio"] for distribution in ("phi", "psi")}},
        "executed_steps": int(histories["phi_tile_weighted_abs"].shape[0]), "finite": finite, "maximum_stage102_shell_history_closure_relative": float(max_closure), "tile_summary": tile_summary, "best_common_contiguous_2x2": best_block, "final_effective_tile_count": {distribution: _effective_tile_count(histories[f"{distribution}_tile_abs_share"][-1]) for distribution in ("phi", "psi")}, "decision": decision,
        "scientific_conclusion": "Stage 103 asks only whether the already-observed moderate-speed shell-1 correction growth is spatially localized inside the frozen cavity interior. Equal-area tiling prevents a larger region from appearing dominant merely because it contains more cells. Any localization remains a diagnostic of correction magnitude, not a causal instability mechanism, solver-accuracy result, or physical validation.",
        "negative_result_guard": "Stage 102 is a localization result, not proof of causality. Stage 101 found diffuse angular growth; Stage 100 supports same-run directional attribution but not nonlinear MUSCL stability; Stage 99 remains a negative cross-run reproducibility result; Stage 98 remains a negative cross-run replay result; Stage 90 remains nonconverged in both reconstruction arms; Stage 28 remains a failed MUSCL endpoint; and the Stage-89 one-sided boundary slope is not promoted. No failed parameter is retuned, no cross-Knudsen MUSCL extension is permitted, and no stability, accuracy, benchmark, or validation claim is authorized."
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage102-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage103(args.stage67_artifact_dir, args.stage102_artifact_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
