from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import stage110_same_sign_slope_asymmetry_audit as s110

STAGE110_RUN_ID = 31567568732
STAGE110_JOB_ID = 94022470984
STAGE110_ARTIFACT_ID = 9138171523
STAGE110_ARTIFACT_SHA256 = "1311985d50c2840ee9ce06f60eacc49d3d2d10023a22283d04325fad3459e7e4"
STAGE110_SUMMARY_SHA256 = "277d860d302e6216cbbdb5ba503acb02bfbf78a2826ba505b2eb635c07e1ef9a"
STAGE110_MAPS_SHA256 = "d63a60ff8ee51769385031066429b43efdba61e8373185e818d3813e39943265"
STAGE110_DECISION = "stage110_relative_same_sign_asymmetry_coupled_stage111_axis_conditioned_asymmetry_audit"

GRID = s110.GRID
KNUDSEN = s110.KNUDSEN
COLD_HOT_RATIO = s110.COLD_HOT_RATIO
RULE = s110.RULE
RADIAL_SCALE = s110.RADIAL_SCALE
LIMITER = s110.LIMITER
BOUNDARY_SLOPE = s110.BOUNDARY_SLOPE
SOURCE_RELAXATION = s110.SOURCE_RELAXATION
TOLERANCE = s110.TOLERANCE
CORRECTION_FLOOR = s110.CORRECTION_FLOOR
DIAGNOSTIC_STEPS = s110.DIAGNOSTIC_STEPS
WALL_BAND_CELLS = s110.WALL_BAND_CELLS
DOMINANT_RADIAL_SHELL = s110.DOMINANT_RADIAL_SHELL
RADIAL_SHELL_COUNT = s110.RADIAL_SHELL_COUNT
RADIAL_NODES_PER_SHELL = s110.RADIAL_NODES_PER_SHELL
INTERIOR_EXTENT = s110.INTERIOR_EXTENT
LOWER_QUANTILE = s110.LOWER_QUANTILE
UPPER_QUANTILE = s110.UPPER_QUANTILE
MODE_CLOSURE_TOLERANCE = 1.0e-12
AXIS_RANK_COUPLING_GUARD = 0.50
QUARTILE_AMPLITUDE_RATIO_GUARD = 1.50
AXIS_DOMINANCE_RATIO_GUARD = 1.25


def validate_stage111_design(**overrides: object) -> None:
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
        "lower_quantile": LOWER_QUANTILE,
        "upper_quantile": UPPER_QUANTILE,
        "mode_closure_tolerance": MODE_CLOSURE_TOLERANCE,
        "axis_rank_coupling_guard": AXIS_RANK_COUPLING_GUARD,
        "quartile_amplitude_ratio_guard": QUARTILE_AMPLITUDE_RATIO_GUARD,
        "axis_dominance_ratio_guard": AXIS_DOMINANCE_RATIO_GUARD,
        "stage67_run_id": s110.STAGE67_RUN_ID,
        "stage110_run_id": STAGE110_RUN_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 111 is frozen to the exact completed Stage-67 shell-1 distributions and Stage-110 "
            "same-sign slope-asymmetry artifact. It may not retune physics, collision/source treatment, "
            "floors, source relaxation, transport, wall treatment, limiter, velocity quadrature, "
            "normalization, diagnostic window, decision guards, or the failed MUSCL endpoint."
        )
    if INTERIOR_EXTENT != 56 or RADIAL_NODES_PER_SHELL != 10:
        raise ValueError("Stage 111 requires the exact 56x56 interior and four 10-node radial shells")
    if not (0.0 < AXIS_RANK_COUPLING_GUARD < 1.0):
        raise ValueError("Stage-111 rank-coupling guard must remain inside (0,1)")
    if QUARTILE_AMPLITUDE_RATIO_GUARD <= 1.0 or AXIS_DOMINANCE_RATIO_GUARD <= 1.0:
        raise ValueError("Stage-111 amplitude and axis-dominance guards must remain above unity")


def _load_stage110(root: str | Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    root = Path(root)
    expected = {
        "summary.json": STAGE110_SUMMARY_SHA256,
        "same_sign_slope_asymmetry_maps.npz": STAGE110_MAPS_SHA256,
    }
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or s110.sha256_file(path) != checksum:
            raise ValueError(f"Stage-110 checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 110 or summary.get("decision") != STAGE110_DECISION or summary.get("finite") is not True:
        raise ValueError("Stage-110 completed endpoint does not authorize Stage 111")
    cfg = summary.get("configuration", {})
    checks = {
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
    if not isinstance(cfg, dict) or any(cfg.get(k) != v for k, v in checks.items()):
        raise ValueError("Stage-110 configuration does not match the frozen Stage-111 design")
    for key, value in cfg.items():
        if key.endswith("_retuning") and value is not False:
            raise ValueError("Stage-110 parent reports forbidden retuning")
    if cfg.get("failed_muscl_endpoint_rehabilitated") is not False or cfg.get("cross_knudsen_extension_permitted") is not False:
        raise ValueError("Stage 111 cannot consume a rehabilitated or cross-Knudsen MUSCL endpoint")
    needed = {
        "phi_same_sign_change_weighted_abs",
        "phi_same_sign_centered_slope_weighted_abs",
        "phi_same_sign_relative_asymmetry",
        "phi_growth_amplitude",
        "psi_same_sign_change_weighted_abs",
        "psi_same_sign_centered_slope_weighted_abs",
        "psi_same_sign_relative_asymmetry",
        "psi_growth_amplitude",
        "common_relative_same_sign_asymmetry",
        "joint_growth_amplitude",
    }
    with np.load(root / "same_sign_slope_asymmetry_maps.npz") as data:
        if not needed.issubset(data.files):
            raise ValueError("Stage-110 map payload is missing required fields")
        arrays = {name: np.asarray(data[name], dtype=np.float64).copy() for name in needed}
    for name, value in arrays.items():
        if value.shape != (INTERIOR_EXTENT, INTERIOR_EXTENT):
            raise ValueError(f"Stage-110 map {name} has the wrong shape")
        if not np.isfinite(value).all() or np.any(value < 0.0):
            raise ValueError(f"Stage-110 map {name} is invalid")
    return summary, arrays


def _axis_same_sign_asymmetry_maps(distribution: np.ndarray, velocity_weight: np.ndarray) -> dict[str, np.ndarray]:
    f = np.asarray(distribution, dtype=np.float64)
    w = np.asarray(velocity_weight, dtype=np.float64)
    if f.ndim != 3 or f.shape[:2] != GRID or f.shape[-1] != w.size:
        raise ValueError("Stage-111 distribution/weight shapes are inconsistent")
    if not np.isfinite(f).all() or not np.isfinite(w).all() or np.any(w < 0.0):
        raise ValueError("Stage-111 inputs must be finite with nonnegative weights")
    wb = WALL_BAND_CELLS
    ys = slice(wb, f.shape[0] - wb)
    xs = slice(wb, f.shape[1] - wb)
    center = f[ys, xs]
    ww = w[None, None, :]
    total_weight = float(np.sum(w))
    out: dict[str, np.ndarray] = {}
    for axis in ("x", "y"):
        if axis == "x":
            left = center - f[ys, slice(wb - 1, f.shape[1] - wb - 1)]
            right = f[ys, slice(wb + 1, f.shape[1] - wb + 1)] - center
        else:
            left = center - f[slice(wb - 1, f.shape[0] - wb - 1), xs]
            right = f[slice(wb + 1, f.shape[0] - wb + 1), xs] - center
        same_sign = ((left > 0.0) & (right > 0.0)) | ((left < 0.0) & (right < 0.0))
        left_abs = np.abs(left)
        right_abs = np.abs(right)
        change = np.sum(np.where(same_sign, 0.5 * np.abs(left_abs - right_abs), 0.0) * ww, axis=-1)
        centered = np.sum(np.where(same_sign, 0.5 * (left_abs + right_abs), 0.0) * ww, axis=-1)
        relative = np.divide(change, centered, out=np.zeros_like(change), where=centered > 0.0)
        support = np.sum(same_sign * ww, axis=-1) / max(total_weight, 1.0e-300)
        out[f"{axis}_same_sign_change_weighted_abs"] = change
        out[f"{axis}_same_sign_centered_slope_weighted_abs"] = centered
        out[f"{axis}_same_sign_relative_asymmetry"] = relative
        out[f"{axis}_same_sign_support_weight_fraction"] = support
    return out


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 1.0e-300))


def _axis_is_coupled(metrics: dict[str, object], axis: str) -> bool:
    vals = []
    ratios = []
    for distribution in ("phi", "psi"):
        block = metrics[distribution]
        assert isinstance(block, dict)
        axis_block = block[axis]
        assert isinstance(axis_block, dict)
        vals.append(float(axis_block["coupling"]["spearman"]))
        ratios.append(float(axis_block["coupling"]["upper_to_lower_mean_amplitude_ratio"]))
    return min(vals) >= AXIS_RANK_COUPLING_GUARD and min(ratios) >= QUARTILE_AMPLITUDE_RATIO_GUARD


def stage111_decision(metrics: dict[str, object], finite: bool, max_parent_closure_relative_l2: float) -> str:
    if not finite or not np.isfinite(max_parent_closure_relative_l2):
        return "stage111_nonfinite_axis_conditioning_blocker_without_retuning"
    if max_parent_closure_relative_l2 > MODE_CLOSURE_TOLERANCE:
        return "stage111_axis_decomposition_closure_blocker_without_retuning"
    coupled = {axis: _axis_is_coupled(metrics, axis) for axis in ("x", "y")}
    if coupled["x"] and not coupled["y"]:
        return "stage111_x_axis_asymmetry_coupled_stage112_axis_specific_spatial_audit"
    if coupled["y"] and not coupled["x"]:
        return "stage111_y_axis_asymmetry_coupled_stage112_axis_specific_spatial_audit"
    if coupled["x"] and coupled["y"]:
        common = metrics["common_axis_coupling"]
        assert isinstance(common, dict)
        x = common["x"]
        y = common["y"]
        assert isinstance(x, dict) and isinstance(y, dict)
        sx = float(x["spearman"])
        sy = float(y["spearman"])
        rx = float(x["upper_to_lower_mean_amplitude_ratio"])
        ry = float(y["upper_to_lower_mean_amplitude_ratio"])
        x_dominant = sx >= AXIS_DOMINANCE_RATIO_GUARD * max(sy, 1.0e-300) and rx >= AXIS_DOMINANCE_RATIO_GUARD * max(ry, 1.0e-300)
        y_dominant = sy >= AXIS_DOMINANCE_RATIO_GUARD * max(sx, 1.0e-300) and ry >= AXIS_DOMINANCE_RATIO_GUARD * max(rx, 1.0e-300)
        if x_dominant:
            return "stage111_x_axis_dominates_stage112_axis_specific_spatial_audit"
        if y_dominant:
            return "stage111_y_axis_dominates_stage112_axis_specific_spatial_audit"
        return "stage111_both_axes_asymmetry_coupled_stage112_joint_axis_interaction_audit"
    return "stage111_axis_conditioning_not_sufficient_stage112_gradient_strength_confound_audit"


def run_stage111(stage67_artifact_dir: str | Path, stage110_artifact_dir: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage111_design(**design)
    stage67_summary, stage67_distributions = s110._load_stage67(stage67_artifact_dir)
    stage110_summary, stage110_maps = _load_stage110(stage110_artifact_dir)

    axis_maps: dict[str, dict[str, np.ndarray]] = {}
    with np.load(stage67_distributions) as saved:
        vx = np.asarray(saved["vx"], dtype=np.float64)
        vy = np.asarray(saved["vy"], dtype=np.float64)
        weight = np.asarray(saved["weight"], dtype=np.float64)
        shell_index = s110._radial_shell_indices(vx, vy)
        shell_mask = shell_index == DOMINANT_RADIAL_SHELL
        shell_point_count = int(np.count_nonzero(shell_mask))
        if shell_point_count != RADIAL_NODES_PER_SHELL * RULE[1]:
            raise ValueError("Stage-111 shell-1 support does not contain the fixed 10x96 points")
        shell_weight = weight[shell_mask]
        speed = np.hypot(vx, vy)
        shell_speed_min = float(np.min(speed[shell_mask]))
        shell_speed_max = float(np.max(speed[shell_mask]))
        shell_speed_mean = float(np.mean(speed[shell_mask]))
        for distribution in ("phi", "psi"):
            full = np.asarray(saved[distribution], dtype=np.float64)
            if full.shape != (GRID[0], GRID[1], RULE[0] * RULE[1]):
                raise ValueError(f"Stage-67 {distribution} has the wrong frozen shape")
            axis_maps[distribution] = _axis_same_sign_asymmetry_maps(full[..., shell_mask], shell_weight)

    metrics: dict[str, object] = {}
    output: dict[str, np.ndarray] = {}
    closure_values: list[float] = []
    for distribution in ("phi", "psi"):
        maps = axis_maps[distribution]
        parent_change = stage110_maps[f"{distribution}_same_sign_change_weighted_abs"]
        parent_centered = stage110_maps[f"{distribution}_same_sign_centered_slope_weighted_abs"]
        recombined_change = maps["x_same_sign_change_weighted_abs"] + maps["y_same_sign_change_weighted_abs"]
        recombined_centered = maps["x_same_sign_centered_slope_weighted_abs"] + maps["y_same_sign_centered_slope_weighted_abs"]
        recombined_relative = np.divide(recombined_change, recombined_centered, out=np.zeros_like(recombined_change), where=recombined_centered > 0.0)
        parent_relative = stage110_maps[f"{distribution}_same_sign_relative_asymmetry"]
        change_closure = _safe_ratio(float(np.linalg.norm(recombined_change - parent_change)), float(np.linalg.norm(parent_change)))
        centered_closure = _safe_ratio(float(np.linalg.norm(recombined_centered - parent_centered)), float(np.linalg.norm(parent_centered)))
        relative_closure = _safe_ratio(float(np.linalg.norm(recombined_relative - parent_relative)), float(np.linalg.norm(parent_relative)))
        closure_values.extend([change_closure, centered_closure, relative_closure])
        amplitude = stage110_maps[f"{distribution}_growth_amplitude"]
        block: dict[str, object] = {
            "parent_change_closure_relative_l2": change_closure,
            "parent_centered_slope_closure_relative_l2": centered_closure,
            "parent_relative_asymmetry_closure_relative_l2": relative_closure,
        }
        for axis in ("x", "y"):
            relative = maps[f"{axis}_same_sign_relative_asymmetry"]
            block[axis] = {
                "mean_relative_asymmetry": float(np.mean(relative)),
                "median_relative_asymmetry": float(np.median(relative)),
                "mean_same_sign_support_weight_fraction": float(np.mean(maps[f"{axis}_same_sign_support_weight_fraction"])),
                "weighted_change_share": _safe_ratio(float(np.sum(maps[f"{axis}_same_sign_change_weighted_abs"])), float(np.sum(recombined_change))),
                "coupling": s110._coupling_metrics(relative, amplitude),
            }
            for name in (
                "same_sign_change_weighted_abs",
                "same_sign_centered_slope_weighted_abs",
                "same_sign_relative_asymmetry",
                "same_sign_support_weight_fraction",
            ):
                output[f"{distribution}_{axis}_{name}"] = maps[f"{axis}_{name}"]
        output[f"{distribution}_growth_amplitude"] = amplitude
        metrics[distribution] = block

    common_axis: dict[str, object] = {}
    joint_amplitude = stage110_maps["joint_growth_amplitude"]
    for axis in ("x", "y"):
        common_relative = np.minimum(
            axis_maps["phi"][f"{axis}_same_sign_relative_asymmetry"],
            axis_maps["psi"][f"{axis}_same_sign_relative_asymmetry"],
        )
        output[f"common_{axis}_relative_same_sign_asymmetry"] = common_relative
        common_axis[axis] = s110._coupling_metrics(common_relative, joint_amplitude)
    output["joint_growth_amplitude"] = joint_amplitude
    metrics["common_axis_coupling"] = common_axis

    max_parent_closure_relative_l2 = float(max(closure_values))
    finite = all(np.isfinite(value).all() for value in output.values()) and np.isfinite(max_parent_closure_relative_l2)
    decision = stage111_decision(metrics, bool(finite), max_parent_closure_relative_l2)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "axis_conditioned_asymmetry_maps.npz", **output)
    result: dict[str, object] = {
        "stage": 111,
        "description": (
            "Frozen artifact-only decomposition of the Stage-110 relative same-sign slope-asymmetry association into x- and y-axis contributions. "
            "The exact Stage-67 shell-1 distributions are reconstructed with the unchanged four-cell-excluded interior and the axis contributions are required to close the Stage-110 parent maps before any axis interpretation is allowed."
        ),
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
            "radial_shell_count": RADIAL_SHELL_COUNT,
            "radial_nodes_per_shell": RADIAL_NODES_PER_SHELL,
            "shell_velocity_point_count": shell_point_count,
            "shell_speed_minimum": shell_speed_min,
            "shell_speed_maximum": shell_speed_max,
            "shell_speed_mean": shell_speed_mean,
            "lower_quantile": LOWER_QUANTILE,
            "upper_quantile": UPPER_QUANTILE,
            "mode_closure_tolerance": MODE_CLOSURE_TOLERANCE,
            "axis_rank_coupling_guard": AXIS_RANK_COUPLING_GUARD,
            "quartile_amplitude_ratio_guard": QUARTILE_AMPLITUDE_RATIO_GUARD,
            "axis_dominance_ratio_guard": AXIS_DOMINANCE_RATIO_GUARD,
            "stage67_run_id": s110.STAGE67_RUN_ID,
            "stage67_job_id": s110.STAGE67_JOB_ID,
            "stage67_artifact_id": s110.STAGE67_ARTIFACT_ID,
            "stage67_artifact_sha256": s110.STAGE67_ARTIFACT_SHA256,
            "stage110_run_id": STAGE110_RUN_ID,
            "stage110_job_id": STAGE110_JOB_ID,
            "stage110_artifact_id": STAGE110_ARTIFACT_ID,
            "stage110_artifact_sha256": STAGE110_ARTIFACT_SHA256,
            "full_solver_endpoint_rerun": False,
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "correction_floor_retuning": False,
            "positivity_floor_retuning": False,
            "source_relaxation_retuning": False,
            "transport_parameter_retuning": False,
            "wall_model_retuning": False,
            "normalization_retuning": False,
            "limiter_retuning": False,
            "velocity_quadrature_retuning": False,
            "failed_muscl_endpoint_rehabilitated": False,
            "one_sided_boundary_slope_promoted": False,
            "cross_knudsen_extension_permitted": False,
            "validation_claim_permitted": False,
            "solver_endpoint_claim_permitted": False,
        },
        "stage67_authorization": {"stage": stage67_summary["stage"], "decision": stage67_summary["decision"]},
        "stage110_authorization": {
            "decision": stage110_summary["decision"],
            "phi_relative_asymmetry_spearman": stage110_summary["metrics"]["phi"]["relative_asymmetry_coupling"]["spearman"],
            "psi_relative_asymmetry_spearman": stage110_summary["metrics"]["psi"]["relative_asymmetry_coupling"]["spearman"],
            "common_relative_asymmetry_spearman": stage110_summary["metrics"]["common_factor_coupling"]["relative_asymmetry"]["spearman"],
        },
        "finite": bool(finite),
        "max_parent_closure_relative_l2": max_parent_closure_relative_l2,
        "metrics": metrics,
        "decision": decision,
        "scientific_conclusion": (
            "Stage 111 asks only whether the already-observed Stage-110 relative-asymmetry association is preferentially organized by the x or y reconstruction axis. "
            "Even a strong axis association remains diagnostic: it cannot establish limiter causality, nonlinear stability, endpoint convergence, heat-flux improvement, or validation."
        ),
        "negative_result_guard": (
            "Stage 110 remains association rather than causal isolation, with same-sign gradient strength also strongly coupled. Stage 109 remains algebraic mode attribution only; Stage 108 remains association only; Stage 107 remains precursor association only; Stage 106 remains gradient-magnitude organization; Stage 105 remains directional alignment without strong single-axis dominance; Stage 104 remains mesoscopic; Stage 103 remains spatially diffuse; Stage 102 remains velocity-shell localization; Stage 101 remains angularly diffuse; Stage 100 is same-run attribution only; Stage 99 remains a negative cross-run reproducibility result; Stage 98 remains a negative cross-run replay result; Stage 90 remains nonconverged in both reconstruction arms; Stage 28 remains a failed MUSCL endpoint; and the Stage-89 one-sided boundary slope is not promoted. No failed parameter is retuned, no cross-Knudsen MUSCL extension is permitted, and no stability, accuracy, benchmark, heat-flux-improvement, or validation claim is authorized."
        ),
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Stage-111 axis-conditioned asymmetry audit")
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage110-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_stage111(args.stage67_artifact_dir, args.stage110_artifact_dir, args.output_dir)


if __name__ == "__main__":
    main()
