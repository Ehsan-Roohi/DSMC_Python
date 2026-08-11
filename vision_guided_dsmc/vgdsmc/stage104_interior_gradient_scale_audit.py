from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE103_RUN_ID = 31445957648
STAGE103_JOB_ID = 93640121182
STAGE103_ARTIFACT_ID = 9085375562
STAGE103_ARTIFACT_SHA256 = "f768105bac2a7f87c722687fd942b280701a236817c414282b1cd88129e1f6fa"
STAGE103_SUMMARY_SHA256 = "a19d325d7b72938da7673b9a66b2669623840d0e00cfbbe7696da885608eb914"
STAGE103_HISTORIES_SHA256 = "a13c1130f92613ad97b60beafb1899269409e6dae6b51991a61633f6ec6c0867"
STAGE103_DECISION = "stage103_spatially_diffuse_shell1_growth_stage104_interior_gradient_scale_audit"

GRID = (64, 64)
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
WALL_BAND_CELLS = 4
DOMINANT_RADIAL_SHELL = 1
PARENT_CLOSURE_TOLERANCE = 1.0e-12
LAGS_CELLS = (1, 2, 4, 7, 14)
GRID_SCALE_GUARD_CELLS = 2.0
MESOSCALE_UPPER_GUARD_CELLS = 7.0
INTERIOR_EXTENT = GRID[0] - 2 * WALL_BAND_CELLS


def validate_stage104_design(**overrides: object) -> None:
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
        "parent_closure_tolerance": PARENT_CLOSURE_TOLERANCE,
        "lags_cells": LAGS_CELLS,
        "grid_scale_guard_cells": GRID_SCALE_GUARD_CELLS,
        "mesoscale_upper_guard_cells": MESOSCALE_UPPER_GUARD_CELLS,
        "stage103_run_id": STAGE103_RUN_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 104 is frozen to the exact completed Stage-103 artifact and an a-priori "
            "interior gradient-scale diagnostic. Physics, collision/source treatment, clipping "
            "or positivity floors, source relaxation, transport parameters, wall model, limiter, "
            "velocity quadrature, normalization, thresholds, and the failed MUSCL endpoint may "
            "not be retuned."
        )
    if GRID[0] != GRID[1] or INTERIOR_EXTENT != 56:
        raise ValueError("Stage 104 requires the exact 64x64 parent grid and 56x56 frozen interior")
    if any(lag <= 0 or lag >= INTERIOR_EXTENT for lag in LAGS_CELLS):
        raise ValueError("Stage-104 fixed lags must lie inside the frozen interior extent")
    if not (0.0 < GRID_SCALE_GUARD_CELLS < MESOSCALE_UPPER_GUARD_CELLS):
        raise ValueError("Stage-104 scale guards must remain ordered and positive")


def _exact_interior_mask() -> np.ndarray:
    mask = np.zeros(GRID, dtype=bool)
    lo = WALL_BAND_CELLS
    mask[lo : GRID[0] - lo, lo : GRID[1] - lo] = True
    return mask


def _load_and_validate_stage103(root: str | Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 103 or summary.get("decision") != STAGE103_DECISION:
        raise ValueError("Stage-103 artifact does not authorize the Stage-104 gradient-scale audit")
    if summary.get("finite") is not True or int(summary.get("executed_steps", -1)) != DIAGNOSTIC_STEPS:
        raise ValueError("Stage-103 artifact did not complete the frozen diagnostic window")
    if float(summary.get("maximum_stage102_shell_history_closure_relative", np.inf)) > PARENT_CLOSURE_TOLERANCE:
        raise ValueError("Stage-103 parent-history closure failed")

    cfg = summary.get("configuration", {})
    if not isinstance(cfg, dict):
        raise ValueError("Stage-103 configuration is missing")
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
        "correction_floor": CORRECTION_FLOOR,
        "diagnostic_steps": DIAGNOSTIC_STEPS,
        "wall_band_cells": WALL_BAND_CELLS,
        "dominant_radial_shell": DOMINANT_RADIAL_SHELL,
    }
    if any(cfg.get(key) != value for key, value in frozen_checks.items()):
        raise ValueError("Stage-103 artifact does not match the frozen Stage-104 parent design")
    if cfg.get("failed_muscl_endpoint_rehabilitated") is not False:
        raise ValueError("Stage 104 cannot consume a rehabilitated failed MUSCL endpoint")
    if cfg.get("cross_knudsen_extension_permitted") is not False:
        raise ValueError("Stage 104 forbids a cross-Knudsen MUSCL extension")
    for key, value in cfg.items():
        if key.endswith("_retuning") and value is not False:
            raise ValueError("Stage-103 artifact reports forbidden parameter retuning")

    with np.load(root / "shell1_spatial_localization_histories.npz") as data:
        needed = {
            "interior_mask",
            "first_phi_shell1_cell_abs",
            "final_phi_shell1_cell_abs",
            "first_psi_shell1_cell_abs",
            "final_psi_shell1_cell_abs",
        }
        if not needed.issubset(data.files):
            raise ValueError("Stage-103 artifact is missing required shell-1 cell maps")
        arrays = {name: np.asarray(data[name]).copy() for name in needed}
    exact_mask = _exact_interior_mask()
    if arrays["interior_mask"].shape != GRID or not np.array_equal(arrays["interior_mask"].astype(bool), exact_mask):
        raise ValueError("Stage-103 interior mask does not match the exact frozen 56x56 interior")
    for name, array in arrays.items():
        if name == "interior_mask":
            continue
        if array.shape != GRID or not np.isfinite(array).all() or np.any(array < 0.0):
            raise ValueError(f"Stage-103 map {name} is invalid")
    return summary, arrays


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 1.0e-300))


def _lag_increment_ratio(field: np.ndarray, lag: int) -> float:
    a = np.asarray(field, dtype=np.float64)
    if a.ndim != 2 or lag <= 0 or lag >= min(a.shape):
        raise ValueError("Stage-104 lag increment requires a valid two-dimensional field and interior lag")
    dx = a[:, lag:] - a[:, :-lag]
    dy = a[lag:, :] - a[:-lag, :]
    numerator = float(np.sqrt(np.sum(dx * dx) + np.sum(dy * dy)))
    denominator = float(np.linalg.norm(a))
    return _safe_ratio(numerator, denominator)


def _growth_metrics(first_map: np.ndarray, final_map: np.ndarray) -> tuple[dict[str, object], np.ndarray]:
    first = np.asarray(first_map, dtype=np.float64)
    final = np.asarray(final_map, dtype=np.float64)
    if first.shape != GRID or final.shape != GRID or not np.isfinite(first).all() or not np.isfinite(final).all():
        raise ValueError("Stage-104 growth maps must be finite exact-grid fields")
    lo = WALL_BAND_CELLS
    first_i = first[lo : GRID[0] - lo, lo : GRID[1] - lo]
    final_i = final[lo : GRID[0] - lo, lo : GRID[1] - lo]
    growth = final_i - first_i
    if not np.isfinite(growth).all():
        raise ValueError("Stage-104 growth map is nonfinite")

    gy, gx = np.gradient(growth)
    gradient_energy = float(np.sum(gx * gx) + np.sum(gy * gy))
    growth_l2 = float(np.linalg.norm(growth))
    characteristic_length = _safe_ratio(growth_l2, np.sqrt(max(gradient_energy, 1.0e-300)))
    x_gradient_energy_share = _safe_ratio(float(np.sum(gx * gx)), gradient_energy)
    abs_growth = float(np.sum(np.abs(growth)))
    positive_growth = float(np.sum(np.clip(growth, 0.0, None)))

    metrics: dict[str, object] = {
        "relative_growth_l1": _safe_ratio(abs_growth, float(np.sum(np.abs(first_i)))),
        "relative_growth_l2": _safe_ratio(growth_l2, float(np.linalg.norm(first_i))),
        "characteristic_gradient_length_cells": characteristic_length,
        "gradient_energy": gradient_energy,
        "x_gradient_energy_share": x_gradient_energy_share,
        "positive_growth_magnitude_share": _safe_ratio(positive_growth, abs_growth),
        "minimum_growth": float(np.min(growth)),
        "maximum_growth": float(np.max(growth)),
        "lag_increment_ratio": {str(lag): _lag_increment_ratio(growth, lag) for lag in LAGS_CELLS},
    }
    return metrics, growth


def stage104_decision(metrics: dict[str, dict[str, object]], parent_closure: float, finite: bool) -> str:
    if not finite:
        return "stage104_nonfinite_gradient_metric_blocker_without_retuning"
    if parent_closure > PARENT_CLOSURE_TOLERANCE:
        return "stage104_stage103_parent_closure_blocker_without_retuning"
    scales = [float(metrics[d]["characteristic_gradient_length_cells"]) for d in ("phi", "psi")]
    minimum_scale = min(scales)
    if minimum_scale <= GRID_SCALE_GUARD_CELLS:
        return "stage104_grid_scale_shell1_gradient_stage105_limiter_activation_audit"
    if minimum_scale <= MESOSCALE_UPPER_GUARD_CELLS:
        return "stage104_mesoscale_shell1_gradient_stage105_directional_gradient_alignment_audit"
    return "stage104_broad_scale_shell1_gradient_stage105_macroscopic_gradient_coupling_audit"


def run_stage104(stage103_artifact_dir: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage104_design(**design)
    parent, arrays = _load_and_validate_stage103(stage103_artifact_dir)
    metrics: dict[str, dict[str, object]] = {}
    growth_maps: dict[str, np.ndarray] = {}
    for distribution in ("phi", "psi"):
        m, g = _growth_metrics(
            arrays[f"first_{distribution}_shell1_cell_abs"],
            arrays[f"final_{distribution}_shell1_cell_abs"],
        )
        metrics[distribution] = m
        growth_maps[distribution] = g

    finite = all(
        np.isfinite(float(metrics[d][key]))
        for d in ("phi", "psi")
        for key in (
            "relative_growth_l1",
            "relative_growth_l2",
            "characteristic_gradient_length_cells",
            "gradient_energy",
            "x_gradient_energy_share",
            "positive_growth_magnitude_share",
            "minimum_growth",
            "maximum_growth",
        )
    )
    parent_closure = float(parent["maximum_stage102_shell_history_closure_relative"])
    decision = stage104_decision(metrics, parent_closure, finite)

    summary: dict[str, object] = {
        "stage": 104,
        "description": "Frozen artifact-only spatial gradient-scale audit of the Stage-103 diffuse shell-1 correction growth inside the exact 56x56 four-cell-excluded cavity interior. Fixed lags and scale guards are preregistered and are not fitted to the observed maps.",
        "configuration": {
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
            "lags_cells": list(LAGS_CELLS),
            "grid_scale_guard_cells": GRID_SCALE_GUARD_CELLS,
            "mesoscale_upper_guard_cells": MESOSCALE_UPPER_GUARD_CELLS,
            "parent_closure_tolerance": PARENT_CLOSURE_TOLERANCE,
            "stage103_run_id": STAGE103_RUN_ID,
            "stage103_job_id": STAGE103_JOB_ID,
            "stage103_artifact_id": STAGE103_ARTIFACT_ID,
            "stage103_artifact_sha256": STAGE103_ARTIFACT_SHA256,
            "full_solver_endpoint_rerun": False,
            "failed_muscl_endpoint_rehabilitated": False,
            "one_sided_boundary_slope_promoted": False,
            "cross_knudsen_extension_permitted": False,
            "validation_claim_permitted": False,
            "solver_endpoint_claim_permitted": False,
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "correction_floor_retuning": False,
            "positivity_floor_retuning": False,
            "source_relaxation_retuning": False,
            "transport_parameter_retuning": False,
            "wall_model_retuning": False,
            "limiter_retuning": False,
            "velocity_quadrature_retuning": False,
            "normalization_retuning": False
        },
        "stage103_authorization": {
            "decision": parent["decision"],
            "maximum_stage102_shell_history_closure_relative": parent_closure,
            "final_effective_tile_count": parent["final_effective_tile_count"],
            "best_common_contiguous_2x2": parent["best_common_contiguous_2x2"]
        },
        "finite": bool(finite),
        "metrics": metrics,
        "decision": decision,
        "scientific_conclusion": "Stage 104 measures only the spatial variation scale of the already-observed diffuse shell-1 MUSCL-correction growth. A finite characteristic length or directional gradient-energy imbalance is a diagnostic of where the frozen correction field varies; it is not a causal instability mechanism, a nonlinear solver result, evidence of improved heat flux, or benchmark validation.",
        "negative_result_guard": "Stage 103 remains a diffuse localization result; Stage 102 remains a velocity-shell localization result rather than causality; Stage 101 remains angularly diffuse; Stage 100 is same-run attribution only; Stage 99 remains a negative cross-run reproducibility result; Stage 98 remains a negative cross-run replay result; Stage 90 remains nonconverged in both reconstruction arms; Stage 28 remains a failed MUSCL endpoint; and the Stage-89 one-sided boundary slope is not promoted. No failed parameter is retuned, no cross-Knudsen MUSCL extension is permitted, and no stability, accuracy, benchmark, or validation claim is authorized."
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        output_dir / "interior_gradient_scale_maps.npz",
        phi_growth_map=growth_maps["phi"],
        psi_growth_map=growth_maps["psi"],
        interior_mask=_exact_interior_mask(),
        lags_cells=np.asarray(LAGS_CELLS, dtype=np.int64)
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 104 frozen interior gradient-scale audit")
    parser.add_argument("--stage103-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = run_stage104(args.stage103_artifact_dir, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
