from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STAGE67_RUN_ID = 30991124477
STAGE67_JOB_ID = 92257254811
STAGE67_ARTIFACT_ID = 8931272132
STAGE67_ARTIFACT_SHA256 = "5b871e94bdca02bce33846657f8f4dc263d68af9e671d7eb137ca7c83da198d4"
STAGE67_SUMMARY_SHA256 = "e04043a1913b2fa9ae57fe1561aa26c70627830d648e91204093c8f1fb57b3d1"
STAGE67_DISTRIBUTIONS_SHA256 = "d4002a2765137ba517abec2d0483a3e5adcf13f415c53259018556bd14d612d1"
STAGE67_DECISION = "stage67_frozen_replay_and_residual_balance_close_stage68_independent_transport_operator_residual_audit"

STAGE109_RUN_ID = 31553717856
STAGE109_JOB_ID = 93981657168
STAGE109_ARTIFACT_ID = 9129826604
STAGE109_ARTIFACT_SHA256 = "cb00f0e5188074762a354973c4e288d42790eee5581e1038a7ebe9ce33cd62e1"
STAGE109_SUMMARY_SHA256 = "ae5942c05221b0e6bf68b22f5547a7bb762649fd30eaab7cfe381e7848267416"
STAGE109_MAPS_SHA256 = "37b57cd484a2ab0426c93599fdf0fb84b2392c93e31314738d66edd1dc7f993b"
STAGE109_DECISION = "stage109_same_sign_amplitude_mode_dominates_stage110_same_sign_slope_asymmetry_audit"

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
RADIAL_SHELL_COUNT = 4
RADIAL_NODES_PER_SHELL = RULE[0] // RADIAL_SHELL_COUNT
INTERIOR_EXTENT = GRID[0] - 2 * WALL_BAND_CELLS
LOWER_QUANTILE = 0.25
UPPER_QUANTILE = 0.75
MODE_CLOSURE_TOLERANCE = 1.0e-12
ASYMMETRY_RANK_COUPLING_GUARD = 0.50
QUARTILE_AMPLITUDE_RATIO_GUARD = 1.50


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_stage110_design(**overrides: object) -> None:
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
        "asymmetry_rank_coupling_guard": ASYMMETRY_RANK_COUPLING_GUARD,
        "quartile_amplitude_ratio_guard": QUARTILE_AMPLITUDE_RATIO_GUARD,
        "stage67_run_id": STAGE67_RUN_ID,
        "stage109_run_id": STAGE109_RUN_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 110 is frozen to the exact completed Stage-67 shell-1 distributions and Stage-109 "
            "same-sign amplitude-mode artifact. It may not retune physics, collision/source treatment, "
            "positivity or correction floors, source relaxation, transport, wall treatment, limiter, "
            "velocity quadrature, normalization, diagnostic window, decision guards, or the failed "
            "Stage-28 MUSCL endpoint."
        )
    if INTERIOR_EXTENT != 56:
        raise ValueError("Stage 110 requires the exact 56x56 four-cell-excluded interior")
    if RULE[0] % RADIAL_SHELL_COUNT != 0 or RADIAL_NODES_PER_SHELL != 10:
        raise ValueError("Stage 110 requires four fixed 10-node radial shells")
    if not (0.0 < ASYMMETRY_RANK_COUPLING_GUARD < 1.0):
        raise ValueError("Stage-110 asymmetry rank-coupling guard must remain inside (0,1)")
    if QUARTILE_AMPLITUDE_RATIO_GUARD <= 1.0:
        raise ValueError("Stage-110 quartile-amplitude guard must remain above unity")


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 1.0e-300))


def _centered_pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    aa = aa - float(np.mean(aa))
    bb = bb - float(np.mean(bb))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1.0e-300:
        return 0.0
    return float(np.dot(aa, bb) / denom)


def _average_ranks(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).ravel()
    if not np.isfinite(x).all():
        raise ValueError("Stage-110 rank input is nonfinite")
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks.reshape(np.asarray(a).shape)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _centered_pearson(_average_ranks(a), _average_ranks(b))


def _radial_shell_indices(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    speed = np.hypot(np.asarray(vx, dtype=np.float64), np.asarray(vy, dtype=np.float64))
    if speed.ndim != 1 or speed.size != RULE[0] * RULE[1]:
        raise ValueError("Stage 110 requires the exact 3840-point velocity rule")
    order = np.argsort(speed, kind="stable")
    shell_size = speed.size // RADIAL_SHELL_COUNT
    labels = np.empty(speed.size, dtype=np.int8)
    labels[order] = np.repeat(np.arange(RADIAL_SHELL_COUNT, dtype=np.int8), shell_size)
    return labels


def _load_stage67(root: str | Path) -> tuple[dict[str, object], Path]:
    root = Path(root)
    expected = {
        "summary.json": STAGE67_SUMMARY_SHA256,
        "converged_full_distributions.npz": STAGE67_DISTRIBUTIONS_SHA256,
    }
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Stage-67 checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 67 or summary.get("decision") != STAGE67_DECISION:
        raise ValueError("Stage-67 completed endpoint mismatch")
    cfg = summary.get("configuration", {})
    checks = {
        "grid": list(GRID),
        "kn0": KNUDSEN,
        "cold_hot_ratio": COLD_HOT_RATIO,
        "radial_nodes": RULE[0],
        "angular_nodes": RULE[1],
        "radial_scale": RADIAL_SCALE,
    }
    if not isinstance(cfg, dict) or any(cfg.get(k) != v for k, v in checks.items()):
        raise ValueError("Stage-67 configuration does not match the frozen Stage-110 design")
    for key, value in cfg.items():
        if key.endswith("_retuning") and value is not False:
            raise ValueError("Stage-67 parent reports forbidden retuning")
    if cfg.get("failed_muscl_endpoint_rehabilitated") is not False:
        raise ValueError("Stage 110 cannot consume a rehabilitated MUSCL endpoint")
    if cfg.get("cross_knudsen_extension_permitted") is not False:
        raise ValueError("Stage 110 forbids cross-Knudsen extension")
    return summary, root / "converged_full_distributions.npz"


def _load_stage109(root: str | Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    root = Path(root)
    expected = {
        "summary.json": STAGE109_SUMMARY_SHA256,
        "limiter_intervention_mode_decomposition_maps.npz": STAGE109_MAPS_SHA256,
    }
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Stage-109 checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 109 or summary.get("decision") != STAGE109_DECISION or summary.get("finite") is not True:
        raise ValueError("Stage-109 completed endpoint does not authorize Stage 110")
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
        raise ValueError("Stage-109 configuration does not match the frozen Stage-110 design")
    for key, value in cfg.items():
        if key.endswith("_retuning") and value is not False:
            raise ValueError("Stage-109 parent reports forbidden retuning")
    if cfg.get("failed_muscl_endpoint_rehabilitated") is not False:
        raise ValueError("Stage 110 cannot consume a rehabilitated failed MUSCL endpoint")
    if cfg.get("cross_knudsen_extension_permitted") is not False:
        raise ValueError("Stage 110 forbids cross-Knudsen extension")
    needed = {
        "phi_same_sign_amplitude_change_weighted_abs",
        "phi_same_sign_amplitude_intervention_fraction",
        "phi_growth_amplitude",
        "psi_same_sign_amplitude_change_weighted_abs",
        "psi_same_sign_amplitude_intervention_fraction",
        "psi_growth_amplitude",
        "common_same_sign_amplitude_intervention_fraction",
        "joint_growth_amplitude",
    }
    with np.load(root / "limiter_intervention_mode_decomposition_maps.npz") as data:
        if not needed.issubset(data.files):
            raise ValueError("Stage-109 map payload is missing required fields")
        arrays = {name: np.asarray(data[name], dtype=np.float64).copy() for name in needed}
    for name, value in arrays.items():
        if value.shape != (INTERIOR_EXTENT, INTERIOR_EXTENT):
            raise ValueError(f"Stage-109 map {name} has the wrong shape")
        if not np.isfinite(value).all() or np.any(value < 0.0):
            raise ValueError(f"Stage-109 map {name} is invalid")
    return summary, arrays


def _same_sign_asymmetry_maps(distribution: np.ndarray, velocity_weight: np.ndarray) -> dict[str, np.ndarray]:
    f = np.asarray(distribution, dtype=np.float64)
    w = np.asarray(velocity_weight, dtype=np.float64)
    if f.ndim != 3 or f.shape[:2] != GRID or f.shape[-1] != w.size:
        raise ValueError("Stage-110 distribution/weight shapes are inconsistent")
    if not np.isfinite(f).all() or not np.isfinite(w).all() or np.any(w < 0.0):
        raise ValueError("Stage-110 asymmetry inputs must be finite with nonnegative weights")

    wb = WALL_BAND_CELLS
    ys = slice(wb, f.shape[0] - wb)
    xs = slice(wb, f.shape[1] - wb)
    center = f[ys, xs]
    change = np.zeros(center.shape[:2], dtype=np.float64)
    centered_same_sign = np.zeros_like(change)
    support_weight = np.zeros_like(change)
    total_axis_velocity_weight = 2.0 * float(np.sum(w))
    ww = w[None, None, :]

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
        local_change = 0.5 * np.abs(left_abs - right_abs)
        local_centered = 0.5 * (left_abs + right_abs)
        change += np.sum(np.where(same_sign, local_change, 0.0) * ww, axis=-1)
        centered_same_sign += np.sum(np.where(same_sign, local_centered, 0.0) * ww, axis=-1)
        support_weight += np.sum(same_sign * ww, axis=-1)

    asymmetry_fraction = np.divide(
        change,
        centered_same_sign,
        out=np.zeros_like(change),
        where=centered_same_sign > 0.0,
    )
    return {
        "same_sign_change_weighted_abs": change,
        "same_sign_centered_slope_weighted_abs": centered_same_sign,
        "same_sign_relative_asymmetry": asymmetry_fraction,
        "same_sign_support_weight_fraction": support_weight / max(total_axis_velocity_weight, 1.0e-300),
    }


def _coupling_metrics(severity: np.ndarray, amplitude: np.ndarray) -> dict[str, float | int]:
    s = np.asarray(severity, dtype=np.float64)
    a = np.asarray(amplitude, dtype=np.float64)
    if s.shape != a.shape or s.shape != (INTERIOR_EXTENT, INTERIOR_EXTENT):
        raise ValueError("Stage-110 coupling maps must share the exact 56x56 shape")
    if not np.isfinite(s).all() or not np.isfinite(a).all() or np.any(s < 0.0) or np.any(a < 0.0):
        raise ValueError("Stage-110 coupling maps must be finite and nonnegative")
    q1 = float(np.quantile(s, LOWER_QUANTILE))
    q3 = float(np.quantile(s, UPPER_QUANTILE))
    low = s <= q1
    high = s >= q3
    mean_low = float(np.mean(a[low]))
    mean_high = float(np.mean(a[high]))
    return {
        "pearson": _centered_pearson(s, a),
        "spearman": _spearman(s, a),
        "severity_lower_quartile_threshold": q1,
        "severity_upper_quartile_threshold": q3,
        "lower_quartile_cell_count": int(np.count_nonzero(low)),
        "upper_quartile_cell_count": int(np.count_nonzero(high)),
        "mean_amplitude_lower_severity_quartile": mean_low,
        "mean_amplitude_upper_severity_quartile": mean_high,
        "upper_to_lower_mean_amplitude_ratio": _safe_ratio(mean_high, mean_low),
        "upper_severity_quartile_amplitude_share": _safe_ratio(float(np.sum(a[high])), float(np.sum(a))),
    }


def stage110_decision(metrics: dict[str, object], finite: bool, max_mode_closure_relative_l2: float) -> str:
    if not finite or not np.isfinite(max_mode_closure_relative_l2):
        return "stage110_nonfinite_same_sign_asymmetry_blocker_without_retuning"
    if max_mode_closure_relative_l2 > MODE_CLOSURE_TOLERANCE:
        return "stage110_same_sign_mode_closure_blocker_without_retuning"
    phi = metrics["phi"]
    psi = metrics["psi"]
    assert isinstance(phi, dict) and isinstance(psi, dict)
    asymmetry_coupled = min(
        float(phi["relative_asymmetry_coupling"]["spearman"]),
        float(psi["relative_asymmetry_coupling"]["spearman"]),
    ) >= ASYMMETRY_RANK_COUPLING_GUARD and min(
        float(phi["relative_asymmetry_coupling"]["upper_to_lower_mean_amplitude_ratio"]),
        float(psi["relative_asymmetry_coupling"]["upper_to_lower_mean_amplitude_ratio"]),
    ) >= QUARTILE_AMPLITUDE_RATIO_GUARD
    gradient_strength_coupled = min(
        float(phi["same_sign_gradient_strength_coupling"]["spearman"]),
        float(psi["same_sign_gradient_strength_coupling"]["spearman"]),
    ) >= ASYMMETRY_RANK_COUPLING_GUARD and min(
        float(phi["same_sign_gradient_strength_coupling"]["upper_to_lower_mean_amplitude_ratio"]),
        float(psi["same_sign_gradient_strength_coupling"]["upper_to_lower_mean_amplitude_ratio"]),
    ) >= QUARTILE_AMPLITUDE_RATIO_GUARD
    if asymmetry_coupled:
        return "stage110_relative_same_sign_asymmetry_coupled_stage111_axis_conditioned_asymmetry_audit"
    if gradient_strength_coupled:
        return "stage110_same_sign_gradient_strength_coupled_stage111_gradient_strength_conditioning_audit"
    return "stage110_same_sign_mode_not_explained_by_single_factor_stage111_joint_factor_spatial_audit"


def run_stage110(
    stage67_artifact_dir: str | Path,
    stage109_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage110_design(**design)
    stage67_summary, stage67_distributions = _load_stage67(stage67_artifact_dir)
    stage109_summary, stage109_maps = _load_stage109(stage109_artifact_dir)

    with np.load(stage67_distributions) as saved:
        vx = np.asarray(saved["vx"], dtype=np.float64)
        vy = np.asarray(saved["vy"], dtype=np.float64)
        weight = np.asarray(saved["weight"], dtype=np.float64)
        shell_index = _radial_shell_indices(vx, vy)
        shell_mask = shell_index == DOMINANT_RADIAL_SHELL
        shell_point_count = int(np.count_nonzero(shell_mask))
        if shell_point_count != RADIAL_NODES_PER_SHELL * RULE[1]:
            raise ValueError("Stage-110 shell-1 support does not contain the fixed 10x96 points")
        speed = np.hypot(vx, vy)
        shell_speed_min = float(np.min(speed[shell_mask]))
        shell_speed_max = float(np.max(speed[shell_mask]))
        shell_speed_mean = float(np.mean(speed[shell_mask]))
        shell_weight = weight[shell_mask]

        asymmetry_maps: dict[str, dict[str, np.ndarray]] = {}
        for distribution in ("phi", "psi"):
            full = np.asarray(saved[distribution], dtype=np.float64)
            if full.shape != (GRID[0], GRID[1], RULE[0] * RULE[1]):
                raise ValueError(f"Stage-67 {distribution} has the wrong frozen shape")
            selected = np.asarray(full[..., shell_mask], dtype=np.float64).copy()
            asymmetry_maps[distribution] = _same_sign_asymmetry_maps(selected, shell_weight)
            del selected, full

    metrics: dict[str, object] = {}
    output: dict[str, np.ndarray] = {}
    closure_values: list[float] = []
    for distribution in ("phi", "psi"):
        maps = asymmetry_maps[distribution]
        parent_change = stage109_maps[f"{distribution}_same_sign_amplitude_change_weighted_abs"]
        diff = maps["same_sign_change_weighted_abs"] - parent_change
        closure_l2 = _safe_ratio(float(np.linalg.norm(diff)), float(np.linalg.norm(parent_change)))
        closure_max_abs = float(np.max(np.abs(diff)))
        closure_values.append(closure_l2)
        amplitude = stage109_maps[f"{distribution}_growth_amplitude"]
        parent_fraction = stage109_maps[f"{distribution}_same_sign_amplitude_intervention_fraction"]
        relative_asymmetry = maps["same_sign_relative_asymmetry"]
        gradient_strength = maps["same_sign_centered_slope_weighted_abs"]
        metrics[distribution] = {
            "mode_closure_relative_l2": closure_l2,
            "mode_closure_max_abs": closure_max_abs,
            "mean_relative_asymmetry": float(np.mean(relative_asymmetry)),
            "median_relative_asymmetry": float(np.median(relative_asymmetry)),
            "upper_quartile_relative_asymmetry": float(np.quantile(relative_asymmetry, UPPER_QUANTILE)),
            "mean_same_sign_support_weight_fraction": float(np.mean(maps["same_sign_support_weight_fraction"])),
            "relative_asymmetry_coupling": _coupling_metrics(relative_asymmetry, amplitude),
            "same_sign_gradient_strength_coupling": _coupling_metrics(gradient_strength, amplitude),
            "relative_asymmetry_vs_stage109_same_sign_fraction_pearson": _centered_pearson(relative_asymmetry, parent_fraction),
            "relative_asymmetry_vs_stage109_same_sign_fraction_spearman": _spearman(relative_asymmetry, parent_fraction),
        }
        for name, value in maps.items():
            output[f"{distribution}_{name}"] = value
        output[f"{distribution}_growth_amplitude"] = amplitude
        output[f"{distribution}_stage109_same_sign_intervention_fraction"] = parent_fraction

    common_relative_asymmetry = np.minimum(
        asymmetry_maps["phi"]["same_sign_relative_asymmetry"],
        asymmetry_maps["psi"]["same_sign_relative_asymmetry"],
    )
    common_gradient_strength = np.minimum(
        asymmetry_maps["phi"]["same_sign_centered_slope_weighted_abs"],
        asymmetry_maps["psi"]["same_sign_centered_slope_weighted_abs"],
    )
    joint_amplitude = stage109_maps["joint_growth_amplitude"]
    metrics["common_factor_coupling"] = {
        "relative_asymmetry": _coupling_metrics(common_relative_asymmetry, joint_amplitude),
        "same_sign_gradient_strength": _coupling_metrics(common_gradient_strength, joint_amplitude),
    }
    output["common_relative_same_sign_asymmetry"] = common_relative_asymmetry
    output["common_same_sign_gradient_strength"] = common_gradient_strength
    output["joint_growth_amplitude"] = joint_amplitude
    output["stage109_common_same_sign_intervention_fraction"] = stage109_maps[
        "common_same_sign_amplitude_intervention_fraction"
    ]

    max_mode_closure_relative_l2 = float(max(closure_values))
    finite = all(np.isfinite(np.asarray(value)).all() for value in output.values())
    finite = finite and np.isfinite(max_mode_closure_relative_l2)
    decision = stage110_decision(metrics, finite, max_mode_closure_relative_l2)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "same_sign_slope_asymmetry_maps.npz", **output)
    result: dict[str, object] = {
        "stage": 110,
        "description": (
            "Frozen artifact-only audit of the Stage-109 dominant same-sign minmod amplitude mode. "
            "Within shell 1 and the fixed four-cell-excluded interior, Stage 110 reconstructs the exact "
            "same-sign limiter change as one-half of the weighted one-sided slope-magnitude imbalance, then "
            "separates relative slope asymmetry from underlying same-sign centered-gradient strength and "
            "tests their associations with the already-observed 25-step correction-growth amplitude."
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
            "asymmetry_rank_coupling_guard": ASYMMETRY_RANK_COUPLING_GUARD,
            "quartile_amplitude_ratio_guard": QUARTILE_AMPLITUDE_RATIO_GUARD,
            "stage67_run_id": STAGE67_RUN_ID,
            "stage67_job_id": STAGE67_JOB_ID,
            "stage67_artifact_id": STAGE67_ARTIFACT_ID,
            "stage67_artifact_sha256": STAGE67_ARTIFACT_SHA256,
            "stage109_run_id": STAGE109_RUN_ID,
            "stage109_job_id": STAGE109_JOB_ID,
            "stage109_artifact_id": STAGE109_ARTIFACT_ID,
            "stage109_artifact_sha256": STAGE109_ARTIFACT_SHA256,
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
        "stage67_authorization": {
            "stage": stage67_summary["stage"],
            "decision": stage67_summary["decision"],
        },
        "stage109_authorization": {
            "decision": stage109_summary["decision"],
            "phi_same_sign_weighted_change_share": stage109_summary["metrics"]["phi"]["same_sign_amplitude_weighted_change_share"],
            "psi_same_sign_weighted_change_share": stage109_summary["metrics"]["psi"]["same_sign_amplitude_weighted_change_share"],
            "phi_same_sign_spearman": stage109_summary["metrics"]["phi"]["same_sign_amplitude_coupling"]["spearman"],
            "psi_same_sign_spearman": stage109_summary["metrics"]["psi"]["same_sign_amplitude_coupling"]["spearman"],
        },
        "finite": bool(finite),
        "max_mode_closure_relative_l2": max_mode_closure_relative_l2,
        "metrics": metrics,
        "decision": decision,
        "scientific_conclusion": (
            "Stage 110 tests whether the dominant Stage-109 same-sign amplitude-limiting association is "
            "carried by relative one-sided slope-magnitude asymmetry, by underlying same-sign gradient strength, "
            "or by neither factor alone. Any association is diagnostic attribution only and does not establish "
            "limiter causality, nonlinear solver stability, endpoint convergence, Table 3/Table 6 improvement, "
            "heat-flux accuracy, or validation."
        ),
        "negative_result_guard": (
            "Stage 109 remains algebraic mode attribution only; Stage 108 remains association only; Stage 107 "
            "remains precursor association only; Stage 106 remains gradient-magnitude organization; Stage 105 "
            "remains directional alignment without strong single-axis dominance; Stage 104 remains mesoscopic; "
            "Stage 103 remains spatially diffuse; Stage 102 remains velocity-shell localization; Stage 101 remains "
            "angularly diffuse; Stage 100 is same-run attribution only; Stage 99 remains a negative cross-run "
            "reproducibility result; Stage 98 remains a negative cross-run replay result; Stage 90 remains "
            "nonconverged in both reconstruction arms; Stage 28 remains a failed MUSCL endpoint; and the Stage-89 "
            "one-sided boundary slope is not promoted. No failed parameter is retuned, no cross-Knudsen MUSCL "
            "extension is permitted, and no stability, accuracy, benchmark, heat-flux-improvement, or validation "
            "claim is authorized."
        ),
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Stage-110 same-sign slope asymmetry audit")
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage109-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_stage110(args.stage67_artifact_dir, args.stage109_artifact_dir, args.output_dir)


if __name__ == "__main__":
    main()
