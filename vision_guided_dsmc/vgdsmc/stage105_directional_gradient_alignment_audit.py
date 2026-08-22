from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE104_RUN_ID = 31448978626
STAGE104_JOB_ID = 93649171038
STAGE104_ARTIFACT_ID = 9090545022
STAGE104_ARTIFACT_SHA256 = "b90a22fd86fcbdbebfad219d5ac9f8fef7cc493386b04c57f9ce30935da13f4d"
STAGE104_SUMMARY_SHA256 = "5ff1e7f4bae6ec0cf473ae61272143c9616dc83942052cccdcd90320008cfc95"
STAGE104_MAPS_SHA256 = "88e45ddb944f498e34c305d2d2585475213937f825d9938a7c334451e11bdc64"
STAGE104_DECISION = "stage104_mesoscale_shell1_gradient_stage105_directional_gradient_alignment_audit"

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
INTERIOR_EXTENT = 56
PARENT_CLOSURE_TOLERANCE = 1.0e-12
STRONG_ALIGNMENT_COSINE = 0.80
STRONG_ALIGNMENT_WEIGHT_SHARE = 0.75
PRINCIPAL_AXIS_ANISOTROPY_GUARD = 0.25


def validate_stage105_design(**overrides: object) -> None:
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
        "strong_alignment_cosine": STRONG_ALIGNMENT_COSINE,
        "strong_alignment_weight_share": STRONG_ALIGNMENT_WEIGHT_SHARE,
        "principal_axis_anisotropy_guard": PRINCIPAL_AXIS_ANISOTROPY_GUARD,
        "stage104_run_id": STAGE104_RUN_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 105 is frozen to the exact completed Stage-104 artifact and fixed directional-alignment "
            "guards. Physics, collision/source treatment, clipping or positivity floors, source relaxation, "
            "transport parameters, wall model, limiter, velocity quadrature, normalization, thresholds, and "
            "the failed MUSCL endpoint may not be retuned."
        )
    if INTERIOR_EXTENT != GRID[0] - 2 * WALL_BAND_CELLS:
        raise ValueError("Stage 105 requires the exact 56x56 four-cell-excluded interior")
    if not (0.0 < STRONG_ALIGNMENT_COSINE < 1.0):
        raise ValueError("Stage-105 cosine guard must remain inside (0,1)")
    if not (0.0 < STRONG_ALIGNMENT_WEIGHT_SHARE < 1.0):
        raise ValueError("Stage-105 alignment-share guard must remain inside (0,1)")
    if not (0.0 < PRINCIPAL_AXIS_ANISOTROPY_GUARD < 1.0):
        raise ValueError("Stage-105 anisotropy guard must remain inside (0,1)")


def _load_and_validate_stage104(root: str | Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 104 or summary.get("decision") != STAGE104_DECISION:
        raise ValueError("Stage-104 artifact does not authorize the Stage-105 directional alignment audit")
    if summary.get("finite") is not True:
        raise ValueError("Stage-104 artifact is nonfinite")
    cfg = summary.get("configuration", {})
    if not isinstance(cfg, dict):
        raise ValueError("Stage-104 configuration is missing")
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
        raise ValueError("Stage-104 artifact does not match the frozen Stage-105 parent design")
    if cfg.get("failed_muscl_endpoint_rehabilitated") is not False:
        raise ValueError("Stage 105 cannot consume a rehabilitated failed MUSCL endpoint")
    if cfg.get("cross_knudsen_extension_permitted") is not False:
        raise ValueError("Stage 105 forbids a cross-Knudsen MUSCL extension")
    for key, value in cfg.items():
        if key.endswith("_retuning") and value is not False:
            raise ValueError("Stage-104 artifact reports forbidden parameter retuning")

    metrics = summary.get("metrics", {})
    if not isinstance(metrics, dict) or set(metrics) != {"phi", "psi"}:
        raise ValueError("Stage-104 metrics are incomplete")
    for distribution in ("phi", "psi"):
        scale = float(metrics[distribution]["characteristic_gradient_length_cells"])
        if not (2.0 < scale <= 7.0):
            raise ValueError("Stage-104 artifact is not on the preregistered mesoscale branch")

    with np.load(root / "interior_gradient_scale_maps.npz") as data:
        needed = {"phi_growth_map", "psi_growth_map", "interior_mask", "lags_cells"}
        if not needed.issubset(data.files):
            raise ValueError("Stage-104 artifact is missing required gradient-scale maps")
        arrays = {name: np.asarray(data[name]).copy() for name in needed}
    if arrays["phi_growth_map"].shape != (INTERIOR_EXTENT, INTERIOR_EXTENT):
        raise ValueError("Stage-104 phi growth map has the wrong shape")
    if arrays["psi_growth_map"].shape != (INTERIOR_EXTENT, INTERIOR_EXTENT):
        raise ValueError("Stage-104 psi growth map has the wrong shape")
    if arrays["interior_mask"].shape != GRID or int(np.sum(arrays["interior_mask"].astype(bool))) != INTERIOR_EXTENT**2:
        raise ValueError("Stage-104 interior mask is not the exact frozen support")
    for name in ("phi_growth_map", "psi_growth_map"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Stage-104 map {name} is nonfinite")
    return summary, arrays


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 1.0e-300))


def _principal_axis(gx: np.ndarray, gy: np.ndarray) -> tuple[float, float]:
    xx = float(np.sum(gx * gx))
    yy = float(np.sum(gy * gy))
    xy = float(np.sum(gx * gy))
    total = xx + yy
    angle = 0.5 * np.arctan2(2.0 * xy, xx - yy)
    anisotropy = _safe_ratio(float(np.hypot(xx - yy, 2.0 * xy)), total)
    return float(np.degrees(angle)), anisotropy


def _orientation_difference_degrees(angle_a: float, angle_b: float) -> float:
    diff = abs(float(angle_a) - float(angle_b)) % 180.0
    return float(min(diff, 180.0 - diff))


def _same_sign_product_share(a: np.ndarray, b: np.ndarray) -> float:
    product = np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64)
    magnitude = np.abs(product)
    return _safe_ratio(float(np.sum(magnitude[product >= 0.0])), float(np.sum(magnitude)))


def _alignment_metrics(phi: np.ndarray, psi: np.ndarray) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    phi_gy, phi_gx = np.gradient(np.asarray(phi, dtype=np.float64))
    psi_gy, psi_gx = np.gradient(np.asarray(psi, dtype=np.float64))
    phi_energy = phi_gx * phi_gx + phi_gy * phi_gy
    psi_energy = psi_gx * psi_gx + psi_gy * psi_gy
    pair_weight = np.sqrt(phi_energy * psi_energy)
    dot = phi_gx * psi_gx + phi_gy * psi_gy
    local_cosine = np.divide(dot, pair_weight, out=np.zeros_like(dot), where=pair_weight > 0.0)

    global_cosine = _safe_ratio(float(np.sum(dot)), float(np.sqrt(np.sum(phi_energy) * np.sum(psi_energy))))
    aligned_share = _safe_ratio(
        float(np.sum(pair_weight[local_cosine >= STRONG_ALIGNMENT_COSINE])), float(np.sum(pair_weight))
    )
    opposed_share = _safe_ratio(
        float(np.sum(pair_weight[local_cosine <= -STRONG_ALIGNMENT_COSINE])), float(np.sum(pair_weight))
    )
    positive_dot_share = _safe_ratio(float(np.sum(np.clip(dot, 0.0, None))), float(np.sum(np.abs(dot))))
    phi_angle, phi_anisotropy = _principal_axis(phi_gx, phi_gy)
    psi_angle, psi_anisotropy = _principal_axis(psi_gx, psi_gy)

    metrics: dict[str, object] = {
        "global_gradient_cosine": global_cosine,
        "strongly_aligned_magnitude_share": aligned_share,
        "strongly_opposed_magnitude_share": opposed_share,
        "positive_dot_magnitude_share": positive_dot_share,
        "principal_axis_angle_degrees": {"phi": phi_angle, "psi": psi_angle},
        "principal_axis_angle_difference_degrees": _orientation_difference_degrees(phi_angle, psi_angle),
        "principal_axis_anisotropy": {"phi": phi_anisotropy, "psi": psi_anisotropy},
        "x_gradient_same_sign_product_share": _same_sign_product_share(phi_gx, psi_gx),
        "y_gradient_same_sign_product_share": _same_sign_product_share(phi_gy, psi_gy),
        "gradient_energy": {"phi": float(np.sum(phi_energy)), "psi": float(np.sum(psi_energy))},
        "x_gradient_energy_share": {
            "phi": _safe_ratio(float(np.sum(phi_gx * phi_gx)), float(np.sum(phi_energy))),
            "psi": _safe_ratio(float(np.sum(psi_gx * psi_gx)), float(np.sum(psi_energy))),
        },
    }
    arrays = {
        "phi_gx": phi_gx,
        "phi_gy": phi_gy,
        "psi_gx": psi_gx,
        "psi_gy": psi_gy,
        "local_gradient_cosine": local_cosine,
        "pair_gradient_weight": pair_weight,
    }
    return metrics, arrays


def stage105_decision(metrics: dict[str, object], parent_closure: float, finite: bool) -> str:
    if not finite:
        return "stage105_nonfinite_alignment_metric_blocker_without_retuning"
    if parent_closure > PARENT_CLOSURE_TOLERANCE:
        return "stage105_stage104_parent_closure_blocker_without_retuning"
    cosine = float(metrics["global_gradient_cosine"])
    aligned_share = float(metrics["strongly_aligned_magnitude_share"])
    opposed_share = float(metrics["strongly_opposed_magnitude_share"])
    anisotropy = metrics["principal_axis_anisotropy"]
    minimum_anisotropy = min(float(anisotropy["phi"]), float(anisotropy["psi"]))
    if cosine >= STRONG_ALIGNMENT_COSINE and aligned_share >= STRONG_ALIGNMENT_WEIGHT_SHARE:
        if minimum_anisotropy >= PRINCIPAL_AXIS_ANISOTROPY_GUARD:
            return "stage105_common_strong_axis_alignment_stage106_directional_limiter_activation_audit"
        return "stage105_common_gradient_alignment_without_axis_dominance_stage106_gradient_magnitude_coupling_audit"
    if cosine <= -STRONG_ALIGNMENT_COSINE and opposed_share >= STRONG_ALIGNMENT_WEIGHT_SHARE:
        return "stage105_common_opposed_gradient_stage106_signed_cancellation_audit"
    return "stage105_mixed_gradient_alignment_stage106_spatial_phase_relation_audit"


def run_stage105(stage104_artifact_dir: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage105_design(**design)
    parent, parent_arrays = _load_and_validate_stage104(stage104_artifact_dir)
    metrics, arrays = _alignment_metrics(parent_arrays["phi_growth_map"], parent_arrays["psi_growth_map"])

    parent_x_share_error = max(
        abs(float(metrics["x_gradient_energy_share"][distribution]) - float(parent["metrics"][distribution]["x_gradient_energy_share"]))
        for distribution in ("phi", "psi")
    )
    parent_energy_relative_error = max(
        abs(float(metrics["gradient_energy"][distribution]) - float(parent["metrics"][distribution]["gradient_energy"]))
        / max(abs(float(parent["metrics"][distribution]["gradient_energy"])), 1.0e-300)
        for distribution in ("phi", "psi")
    )
    parent_closure = max(parent_x_share_error, parent_energy_relative_error)
    scalar_values = [
        float(metrics["global_gradient_cosine"]),
        float(metrics["strongly_aligned_magnitude_share"]),
        float(metrics["strongly_opposed_magnitude_share"]),
        float(metrics["positive_dot_magnitude_share"]),
        float(metrics["principal_axis_angle_difference_degrees"]),
        float(metrics["x_gradient_same_sign_product_share"]),
        float(metrics["y_gradient_same_sign_product_share"]),
        *[float(v) for v in metrics["principal_axis_angle_degrees"].values()],
        *[float(v) for v in metrics["principal_axis_anisotropy"].values()],
        *[float(v) for v in metrics["gradient_energy"].values()],
        *[float(v) for v in metrics["x_gradient_energy_share"].values()],
    ]
    finite = bool(np.isfinite(scalar_values).all())
    decision = stage105_decision(metrics, parent_closure, finite)

    summary: dict[str, object] = {
        "stage": 105,
        "description": "Frozen artifact-only directional alignment audit of the Stage-104 mesoscopic shell-1 growth gradients. The cosine, aligned-weight, and principal-axis-anisotropy guards are fixed diagnostic thresholds and do not alter the solver.",
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
            "strong_alignment_cosine": STRONG_ALIGNMENT_COSINE,
            "strong_alignment_weight_share": STRONG_ALIGNMENT_WEIGHT_SHARE,
            "principal_axis_anisotropy_guard": PRINCIPAL_AXIS_ANISOTROPY_GUARD,
            "parent_closure_tolerance": PARENT_CLOSURE_TOLERANCE,
            "stage104_run_id": STAGE104_RUN_ID,
            "stage104_job_id": STAGE104_JOB_ID,
            "stage104_artifact_id": STAGE104_ARTIFACT_ID,
            "stage104_artifact_sha256": STAGE104_ARTIFACT_SHA256,
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
            "normalization_retuning": False,
        },
        "stage104_authorization": {
            "decision": parent["decision"],
            "characteristic_gradient_length_cells": {
                distribution: float(parent["metrics"][distribution]["characteristic_gradient_length_cells"])
                for distribution in ("phi", "psi")
            },
            "positive_growth_magnitude_share": {
                distribution: float(parent["metrics"][distribution]["positive_growth_magnitude_share"])
                for distribution in ("phi", "psi")
            },
        },
        "finite": finite,
        "parent_closure_relative": parent_closure,
        "metrics": metrics,
        "decision": decision,
        "scientific_conclusion": "Stage 105 tests only whether the two already-observed mesoscopic shell-1 growth maps share gradient directions. Strong phi/psi gradient alignment would identify common spatial organization across reduced distributions, while weak principal-axis anisotropy would still rule out a single strongly preferred cavity axis. Neither outcome establishes a causal instability mechanism, nonlinear MUSCL stability, heat-flux improvement, or benchmark validation.",
        "negative_result_guard": "Stage 104 remains a mesoscopic scale diagnostic rather than causality; Stage 103 remains spatially diffuse; Stage 102 remains a velocity-shell localization result; Stage 101 remains angularly diffuse; Stage 100 is same-run attribution only; Stage 99 remains a negative cross-run reproducibility result; Stage 98 remains a negative cross-run replay result; Stage 90 remains nonconverged in both reconstruction arms; Stage 28 remains a failed MUSCL endpoint; and the Stage-89 one-sided boundary slope is not promoted. No failed parameter is retuned, no cross-Knudsen MUSCL extension is permitted, and no stability, accuracy, benchmark, heat-flux-improvement, or validation claim is authorized.",
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(output_dir / "directional_gradient_alignment_maps.npz", **arrays)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 105 frozen directional gradient-alignment audit")
    parser.add_argument("--stage104-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = run_stage105(args.stage104_artifact_dir, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
