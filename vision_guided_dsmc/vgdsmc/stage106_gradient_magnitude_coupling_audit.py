from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE105_RUN_ID = 31463907070
STAGE105_JOB_ID = 93692732878
STAGE105_ARTIFACT_ID = 9101336753
STAGE105_ARTIFACT_SHA256 = "f7daf8c85cc9f5d20ddd812fe71fa59a03e88e3e1501017cd41b9f89daacdd7b"
STAGE105_SUMMARY_SHA256 = "82a80cfac05cee2e3f4cd86799c5a91315ffeff7b49770b661ef9d53a08ec2cf"
STAGE105_MAPS_SHA256 = "27d5a557fc6c4442b7a0b29deb527f9540fb5f715b9e60b5ea8202df5ec63ee8"
STAGE105_DECISION = "stage105_common_gradient_alignment_without_axis_dominance_stage106_gradient_magnitude_coupling_audit"

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
MAGNITUDE_COSINE_GUARD = 0.80
MAGNITUDE_PEARSON_GUARD = 0.75
UPPER_QUANTILE = 0.75
UPPER_QUARTILE_OVERLAP_GUARD = 0.50


def validate_stage106_design(**overrides: object) -> None:
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
        "magnitude_cosine_guard": MAGNITUDE_COSINE_GUARD,
        "magnitude_pearson_guard": MAGNITUDE_PEARSON_GUARD,
        "upper_quantile": UPPER_QUANTILE,
        "upper_quartile_overlap_guard": UPPER_QUARTILE_OVERLAP_GUARD,
        "stage105_run_id": STAGE105_RUN_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 106 is frozen to the exact completed Stage-105 artifact and fixed magnitude-coupling "
            "diagnostics. Physics, collision/source treatment, clipping or positivity floors, source relaxation, "
            "transport parameters, wall model, limiter, velocity quadrature, normalization, thresholds, and "
            "the failed MUSCL endpoint may not be retuned."
        )
    if INTERIOR_EXTENT != GRID[0] - 2 * WALL_BAND_CELLS:
        raise ValueError("Stage 106 requires the exact 56x56 four-cell-excluded interior")
    if not (0.0 < MAGNITUDE_COSINE_GUARD < 1.0):
        raise ValueError("Stage-106 cosine guard must remain inside (0,1)")
    if not (0.0 < MAGNITUDE_PEARSON_GUARD < 1.0):
        raise ValueError("Stage-106 Pearson guard must remain inside (0,1)")
    if UPPER_QUANTILE != 0.75:
        raise ValueError("Stage-106 upper-quantile support is fixed to the upper quartile")
    if not (0.0 < UPPER_QUARTILE_OVERLAP_GUARD < 1.0):
        raise ValueError("Stage-106 overlap guard must remain inside (0,1)")


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 1.0e-300))


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return _safe_ratio(float(np.linalg.norm(a - b)), float(np.linalg.norm(b)))


def _centered_pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    aa = aa - float(np.mean(aa))
    bb = bb - float(np.mean(bb))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1.0e-300:
        return 0.0
    return float(np.dot(aa, bb) / denom)


def _load_and_validate_stage105(root: str | Path) -> tuple[dict[str, object], dict[str, np.ndarray], float]:
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 105 or summary.get("decision") != STAGE105_DECISION:
        raise ValueError("Stage-105 artifact does not authorize the Stage-106 gradient-magnitude coupling audit")
    if summary.get("finite") is not True:
        raise ValueError("Stage-105 artifact is nonfinite")
    if float(summary.get("parent_closure_relative", np.inf)) > PARENT_CLOSURE_TOLERANCE:
        raise ValueError("Stage-105 artifact failed its own parent-closure gate")

    cfg = summary.get("configuration", {})
    if not isinstance(cfg, dict):
        raise ValueError("Stage-105 configuration is missing")
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
        raise ValueError("Stage-105 artifact does not match the frozen Stage-106 parent design")
    if cfg.get("failed_muscl_endpoint_rehabilitated") is not False:
        raise ValueError("Stage 106 cannot consume a rehabilitated failed MUSCL endpoint")
    if cfg.get("one_sided_boundary_slope_promoted") is not False:
        raise ValueError("Stage 106 cannot consume a promoted one-sided boundary reconstruction")
    if cfg.get("cross_knudsen_extension_permitted") is not False:
        raise ValueError("Stage 106 forbids a cross-Knudsen MUSCL extension")
    for key, value in cfg.items():
        if key.endswith("_retuning") and value is not False:
            raise ValueError("Stage-105 artifact reports forbidden parameter retuning")

    metrics = summary.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("Stage-105 metrics are missing")
    if float(metrics.get("global_gradient_cosine", -np.inf)) < 0.80:
        raise ValueError("Stage-105 artifact is not on the authorized common-gradient-alignment branch")
    if float(metrics.get("strongly_aligned_magnitude_share", -np.inf)) < 0.75:
        raise ValueError("Stage-105 artifact lacks the required common-gradient-alignment support")
    anisotropy = metrics.get("principal_axis_anisotropy", {})
    if not isinstance(anisotropy, dict) or min(float(anisotropy.get("phi", np.inf)), float(anisotropy.get("psi", np.inf))) >= 0.25:
        raise ValueError("Stage-105 artifact is not on the no-strong-axis-dominance branch")

    with np.load(root / "directional_gradient_alignment_maps.npz") as data:
        needed = {
            "phi_gx", "phi_gy", "psi_gx", "psi_gy",
            "local_gradient_cosine", "pair_gradient_weight",
        }
        if set(data.files) != needed:
            raise ValueError("Stage-105 artifact has an unexpected directional-map payload")
        arrays = {name: np.asarray(data[name], dtype=np.float64).copy() for name in needed}
    for name, array in arrays.items():
        if array.shape != (INTERIOR_EXTENT, INTERIOR_EXTENT):
            raise ValueError(f"Stage-105 map {name} has the wrong shape")
        if not np.isfinite(array).all():
            raise ValueError(f"Stage-105 map {name} is nonfinite")

    phi_gx, phi_gy = arrays["phi_gx"], arrays["phi_gy"]
    psi_gx, psi_gy = arrays["psi_gx"], arrays["psi_gy"]
    phi_mag = np.hypot(phi_gx, phi_gy)
    psi_mag = np.hypot(psi_gx, psi_gy)
    pair_weight = phi_mag * psi_mag
    dot = phi_gx * psi_gx + phi_gy * psi_gy
    local_cosine = np.divide(dot, pair_weight, out=np.zeros_like(dot), where=pair_weight > 0.0)
    global_cosine = _safe_ratio(float(np.sum(dot)), float(np.linalg.norm(phi_mag) * np.linalg.norm(psi_mag)))
    closure = max(
        _relative_l2(pair_weight, arrays["pair_gradient_weight"]),
        _relative_l2(local_cosine, arrays["local_gradient_cosine"]),
        abs(global_cosine - float(metrics["global_gradient_cosine"])),
    )
    return summary, arrays, closure


def _magnitude_coupling_metrics(arrays: dict[str, np.ndarray]) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    phi_mag = np.hypot(arrays["phi_gx"], arrays["phi_gy"])
    psi_mag = np.hypot(arrays["psi_gx"], arrays["psi_gy"])

    phi_norm = phi_mag / max(float(np.sqrt(np.mean(phi_mag * phi_mag))), 1.0e-300)
    psi_norm = psi_mag / max(float(np.sqrt(np.mean(psi_mag * psi_mag))), 1.0e-300)
    magnitude_cosine = _safe_ratio(float(np.sum(phi_mag * psi_mag)), float(np.linalg.norm(phi_mag) * np.linalg.norm(psi_mag)))
    pearson = _centered_pearson(phi_mag, psi_mag)

    phi_threshold = float(np.quantile(phi_mag, UPPER_QUANTILE))
    psi_threshold = float(np.quantile(psi_mag, UPPER_QUANTILE))
    phi_high = phi_mag >= phi_threshold
    psi_high = psi_mag >= psi_threshold
    common_high = phi_high & psi_high
    union_high = phi_high | psi_high
    n_phi = int(np.count_nonzero(phi_high))
    n_psi = int(np.count_nonzero(psi_high))
    n_common = int(np.count_nonzero(common_high))
    n_union = int(np.count_nonzero(union_high))
    overlap_coefficient = _safe_ratio(float(n_common), float(min(n_phi, n_psi)))
    jaccard = _safe_ratio(float(n_common), float(n_union))

    pair_weight = phi_mag * psi_mag
    common_pair_weight_share = _safe_ratio(float(np.sum(pair_weight[common_high])), float(np.sum(pair_weight)))

    valid = pair_weight > 0.0
    log_ratio = np.zeros_like(phi_mag)
    log_ratio[valid] = np.log(np.maximum(phi_norm[valid], 1.0e-300) / np.maximum(psi_norm[valid], 1.0e-300))
    weight_sum = float(np.sum(pair_weight[valid]))
    weighted_log_ratio_mean = _safe_ratio(float(np.sum(pair_weight[valid] * log_ratio[valid])), weight_sum)
    weighted_log_ratio_std = float(
        np.sqrt(
            _safe_ratio(
                float(np.sum(pair_weight[valid] * (log_ratio[valid] - weighted_log_ratio_mean) ** 2)),
                weight_sum,
            )
        )
    )

    metrics: dict[str, object] = {
        "gradient_magnitude_cosine": magnitude_cosine,
        "gradient_magnitude_pearson": pearson,
        "upper_quartile_thresholds": {"phi": phi_threshold, "psi": psi_threshold},
        "upper_quartile_cell_counts": {"phi": n_phi, "psi": n_psi, "common": n_common, "union": n_union},
        "upper_quartile_overlap_coefficient": overlap_coefficient,
        "upper_quartile_jaccard": jaccard,
        "common_upper_quartile_pair_weight_share": common_pair_weight_share,
        "normalized_magnitude_weighted_log_ratio_mean": weighted_log_ratio_mean,
        "normalized_magnitude_weighted_log_ratio_std": weighted_log_ratio_std,
    }
    output_arrays = {
        "phi_gradient_magnitude": phi_mag,
        "psi_gradient_magnitude": psi_mag,
        "phi_normalized_gradient_magnitude": phi_norm,
        "psi_normalized_gradient_magnitude": psi_norm,
        "common_upper_quartile_mask": common_high.astype(np.uint8),
        "magnitude_product_weight": pair_weight,
    }
    return metrics, output_arrays


def stage106_decision(metrics: dict[str, object], parent_closure: float, finite: bool) -> str:
    if not finite:
        return "stage106_nonfinite_magnitude_metric_blocker_without_retuning"
    if parent_closure > PARENT_CLOSURE_TOLERANCE:
        return "stage106_stage105_parent_closure_blocker_without_retuning"
    cosine = float(metrics["gradient_magnitude_cosine"])
    pearson = float(metrics["gradient_magnitude_pearson"])
    overlap = float(metrics["upper_quartile_overlap_coefficient"])
    if cosine >= MAGNITUDE_COSINE_GUARD and pearson >= MAGNITUDE_PEARSON_GUARD:
        if overlap >= UPPER_QUARTILE_OVERLAP_GUARD:
            return "stage106_common_gradient_magnitude_coupling_stage107_frozen_limiter_activation_colocation_audit"
        return "stage106_global_magnitude_coupling_diffuse_upper_support_stage107_high_gradient_support_topology_audit"
    if overlap >= UPPER_QUARTILE_OVERLAP_GUARD:
        return "stage106_high_gradient_support_overlap_without_linear_coupling_stage107_rank_coupling_audit"
    return "stage106_gradient_amplitude_decoupling_stage107_spatial_phase_amplitude_audit"


def run_stage106(stage105_artifact_dir: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage106_design(**design)
    parent, arrays, parent_closure = _load_and_validate_stage105(stage105_artifact_dir)
    metrics, output_arrays = _magnitude_coupling_metrics(arrays)

    scalar_values = [
        float(metrics["gradient_magnitude_cosine"]),
        float(metrics["gradient_magnitude_pearson"]),
        float(metrics["upper_quartile_overlap_coefficient"]),
        float(metrics["upper_quartile_jaccard"]),
        float(metrics["common_upper_quartile_pair_weight_share"]),
        float(metrics["normalized_magnitude_weighted_log_ratio_mean"]),
        float(metrics["normalized_magnitude_weighted_log_ratio_std"]),
        float(parent_closure),
    ]
    finite = bool(np.isfinite(scalar_values).all()) and all(np.isfinite(a).all() for a in output_arrays.values())
    decision = stage106_decision(metrics, parent_closure, finite)

    summary: dict[str, object] = {
        "stage": 106,
        "description": "Frozen artifact-only gradient-magnitude coupling audit of the Stage-105 common-direction shell-1 growth gradients. It tests amplitude co-organization without rerunning or retuning the failed MUSCL endpoint.",
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
            "magnitude_cosine_guard": MAGNITUDE_COSINE_GUARD,
            "magnitude_pearson_guard": MAGNITUDE_PEARSON_GUARD,
            "upper_quantile": UPPER_QUANTILE,
            "upper_quartile_overlap_guard": UPPER_QUARTILE_OVERLAP_GUARD,
            "parent_closure_tolerance": PARENT_CLOSURE_TOLERANCE,
            "stage105_run_id": STAGE105_RUN_ID,
            "stage105_job_id": STAGE105_JOB_ID,
            "stage105_artifact_id": STAGE105_ARTIFACT_ID,
            "stage105_artifact_sha256": STAGE105_ARTIFACT_SHA256,
            "full_solver_endpoint_rerun": False,
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "source_relaxation_retuning": False,
            "positivity_floor_retuning": False,
            "correction_floor_retuning": False,
            "transport_parameter_retuning": False,
            "wall_model_retuning": False,
            "limiter_retuning": False,
            "velocity_quadrature_retuning": False,
            "normalization_retuning": False,
            "failed_muscl_endpoint_rehabilitated": False,
            "one_sided_boundary_slope_promoted": False,
            "cross_knudsen_extension_permitted": False,
            "validation_claim_permitted": False,
            "solver_endpoint_claim_permitted": False,
        },
        "stage105_authorization": {
            "decision": parent["decision"],
            "global_gradient_cosine": parent["metrics"]["global_gradient_cosine"],
            "strongly_aligned_magnitude_share": parent["metrics"]["strongly_aligned_magnitude_share"],
            "principal_axis_anisotropy": parent["metrics"]["principal_axis_anisotropy"],
        },
        "parent_closure_relative": parent_closure,
        "metrics": metrics,
        "finite": finite,
        "decision": decision,
        "scientific_conclusion": (
            "Stage 106 tests whether the already-observed phi/psi gradient-direction alignment is accompanied by "
            "co-located gradient strength after scale normalization. Strong coupling supports a shared spatial "
            "organization of the diagnostic correction maps; weak coupling would preserve the directional result "
            "while rejecting common amplitude organization. Neither outcome establishes causality, nonlinear MUSCL "
            "stability, endpoint convergence, accuracy, heat-flux improvement, or benchmark validation."
        ),
        "negative_result_guard": (
            "Stage 105 remains a common-gradient-direction diagnostic without strong single-axis dominance; Stage 104 "
            "remains a mesoscopic scale diagnostic rather than causality; Stage 103 remains spatially diffuse; Stage 102 "
            "remains a velocity-shell localization result; Stage 101 remains angularly diffuse; Stage 100 is same-run "
            "attribution only; Stage 99 remains a negative cross-run reproducibility result; Stage 98 remains a negative "
            "cross-run replay result; Stage 90 remains nonconverged in both reconstruction arms; Stage 28 remains a failed "
            "MUSCL endpoint; and the Stage-89 one-sided boundary slope is not promoted. No failed parameter is retuned, no "
            "cross-Knudsen MUSCL extension is permitted, and no stability, accuracy, benchmark, heat-flux-improvement, or "
            "validation claim is authorized."
        ),
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(out / "gradient_magnitude_coupling_maps.npz", **output_arrays)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen Stage-106 gradient-magnitude coupling audit")
    parser.add_argument("--stage105-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_stage106(args.stage105_artifact_dir, args.output_dir)


if __name__ == "__main__":
    main()
