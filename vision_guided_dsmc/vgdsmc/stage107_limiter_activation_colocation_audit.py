from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .stage41_projected_polar_operator_audit import STAGE41_CORRECTION_FLOOR
from .stage79_dominant_moment_radial_angular_gradient_audit import (
    RADIAL_NODES_PER_SHELL,
    radial_shell_indices,
)
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
)
from .stage97_muscl_correction_spatial_localization_audit import WALL_BAND_CELLS
from .stage98_directional_operator_growth_audit import DIAGNOSTIC_STEPS

STAGE67_RUN_ID = 30991124477
STAGE67_JOB_ID = 92257254811
STAGE67_ARTIFACT_ID = 8931272132
STAGE67_ARTIFACT_SHA256 = "5b871e94bdca02bce33846657f8f4dc263d68af9e671d7eb137ca7c83da198d4"
STAGE106_RUN_ID = 31493094393
STAGE106_JOB_ID = 93784067866
STAGE106_ARTIFACT_ID = 9109429586
STAGE106_ARTIFACT_SHA256 = "29603416869f6d86d8dd8de734436acf5463750601699332798f0c56b64b7112"
STAGE106_SUMMARY_SHA256 = "2eb04fe9b1e1eeff963189b9d7c6af39f0738a5c181a2e0b2847502d0878f80b"
STAGE106_MAPS_SHA256 = "84c1009f54c71454b31a074563d2339a6cf7e791e6062ef3e2a4da59ed8a4413"
STAGE106_DECISION = "stage106_common_gradient_magnitude_coupling_stage107_frozen_limiter_activation_colocation_audit"

BOUNDARY_SLOPE = "zero"
DOMINANT_RADIAL_SHELL = 1
INTERIOR_EXTENT = GRID[0] - 2 * WALL_BAND_CELLS
SUPPORT_QUANTILE = 0.75
COLOCATION_ENRICHMENT_GUARD = 1.25
COLOCATION_OVERLAP_GUARD = 0.50
PAIR_WEIGHT_SHARE_GUARD = 0.75
PARENT_CLOSURE_TOLERANCE = 1.0e-12
ACTIVATION_DEFINITION = "weighted_minmod_intervention_relative_to_centered_slope_at_stage67_pre_replay_state"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_stage107_design(**overrides: object) -> None:
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
        "support_quantile": SUPPORT_QUANTILE,
        "colocation_enrichment_guard": COLOCATION_ENRICHMENT_GUARD,
        "colocation_overlap_guard": COLOCATION_OVERLAP_GUARD,
        "pair_weight_share_guard": PAIR_WEIGHT_SHARE_GUARD,
        "parent_closure_tolerance": PARENT_CLOSURE_TOLERANCE,
        "activation_definition": ACTIVATION_DEFINITION,
        "stage67_run_id": STAGE67_RUN_ID,
        "stage106_run_id": STAGE106_RUN_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 107 is frozen to the exact Stage-67 pre-replay distribution and completed "
            "Stage-106 high-gradient support. It may not retune physics, collision/source treatment, "
            "positivity or correction floors, source relaxation, transport parameters, wall model, "
            "limiter, quadrature, normalization, tolerance, diagnostic window, or the failed MUSCL endpoint."
        )
    if INTERIOR_EXTENT != 56:
        raise ValueError("Stage 107 requires the exact 56x56 four-cell-excluded interior")
    if SUPPORT_QUANTILE != 0.75:
        raise ValueError("Stage 107 retains the Stage-106 upper-quartile support convention")
    if not (1.0 < COLOCATION_ENRICHMENT_GUARD < 2.0):
        raise ValueError("Stage-107 enrichment guard must remain a fixed diagnostic guard above unity")
    if not (0.0 < COLOCATION_OVERLAP_GUARD < 1.0):
        raise ValueError("Stage-107 overlap guard must remain inside (0,1)")
    if not (0.0 < PAIR_WEIGHT_SHARE_GUARD < 1.0):
        raise ValueError("Stage-107 pair-weight guard must remain inside (0,1)")


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


def _load_and_validate_stage106(root: str | Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    root = Path(root)
    expected = {
        "summary.json": STAGE106_SUMMARY_SHA256,
        "gradient_magnitude_coupling_maps.npz": STAGE106_MAPS_SHA256,
    }
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Stage-106 checksum mismatch: {name}")

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 106 or summary.get("decision") != STAGE106_DECISION:
        raise ValueError("Stage-106 completed endpoint does not authorize Stage 107")
    if summary.get("finite") is not True:
        raise ValueError("Stage-106 completed endpoint is nonfinite")
    if float(summary.get("parent_closure_relative", np.inf)) > PARENT_CLOSURE_TOLERANCE:
        raise ValueError("Stage-106 parent closure failed")

    cfg = summary.get("configuration", {})
    if not isinstance(cfg, dict):
        raise ValueError("Stage-106 configuration is missing")
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
        "dominant_radial_shell": DOMINANT_RADIAL_SHELL,
        "upper_quantile": SUPPORT_QUANTILE,
    }
    if any(cfg.get(key) != value for key, value in frozen_checks.items()):
        raise ValueError("Stage-106 artifact does not match the frozen Stage-107 design")
    if cfg.get("failed_muscl_endpoint_rehabilitated") is not False:
        raise ValueError("Stage 107 cannot consume a rehabilitated failed MUSCL endpoint")
    if cfg.get("one_sided_boundary_slope_promoted") is not False:
        raise ValueError("Stage 107 cannot consume a promoted one-sided boundary reconstruction")
    if cfg.get("cross_knudsen_extension_permitted") is not False:
        raise ValueError("Stage 107 forbids cross-Knudsen extension")
    if cfg.get("validation_claim_permitted") is not False:
        raise ValueError("Stage 107 cannot consume a validation-authorized parent")
    for key, value in cfg.items():
        if key.endswith("_retuning") and value is not False:
            raise ValueError("Stage-106 artifact reports forbidden retuning")

    metrics = summary.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("Stage-106 metrics are missing")
    if float(metrics.get("gradient_magnitude_cosine", -np.inf)) < 0.80:
        raise ValueError("Stage-106 magnitude-coupling authorization is absent")
    if float(metrics.get("gradient_magnitude_pearson", -np.inf)) < 0.75:
        raise ValueError("Stage-106 linear magnitude coupling is below its frozen guard")
    if float(metrics.get("upper_quartile_overlap_coefficient", -np.inf)) < 0.50:
        raise ValueError("Stage-106 high-gradient support overlap is below its frozen guard")

    with np.load(root / "gradient_magnitude_coupling_maps.npz") as data:
        expected_names = {
            "phi_gradient_magnitude",
            "psi_gradient_magnitude",
            "phi_normalized_gradient_magnitude",
            "psi_normalized_gradient_magnitude",
            "common_upper_quartile_mask",
            "magnitude_product_weight",
        }
        if set(data.files) != expected_names:
            raise ValueError("Stage-106 map payload is unexpected")
        arrays = {name: np.asarray(data[name]).copy() for name in expected_names}

    for name, value in arrays.items():
        if value.shape != (INTERIOR_EXTENT, INTERIOR_EXTENT):
            raise ValueError(f"Stage-106 map {name} has the wrong shape")
        if not np.isfinite(value).all():
            raise ValueError(f"Stage-106 map {name} is nonfinite")
    common = arrays["common_upper_quartile_mask"].astype(bool)
    if int(np.count_nonzero(common)) != int(metrics["upper_quartile_cell_counts"]["common"]):
        raise ValueError("Stage-106 common-support cell count does not close")
    return summary, arrays


def _limiter_intervention_maps(
    distribution: np.ndarray,
    velocity_weight: np.ndarray,
    wall_band_cells: int = WALL_BAND_CELLS,
) -> dict[str, np.ndarray]:
    f = np.asarray(distribution, dtype=np.float64)
    w = np.asarray(velocity_weight, dtype=np.float64)
    if f.ndim != 3 or f.shape[-1] != w.size:
        raise ValueError("Stage-107 distribution/weight shapes are inconsistent")
    if f.shape[0] <= 2 * wall_band_cells or f.shape[1] <= 2 * wall_band_cells:
        raise ValueError("Stage-107 field is too small for the frozen interior")
    if np.any(w < 0.0) or not np.isfinite(f).all() or not np.isfinite(w).all():
        raise ValueError("Stage-107 limiter inputs must be finite with nonnegative weights")

    ys = slice(wall_band_cells, f.shape[0] - wall_band_cells)
    xs = slice(wall_band_cells, f.shape[1] - wall_band_cells)
    yc = np.arange(wall_band_cells, f.shape[0] - wall_band_cells)
    xc = np.arange(wall_band_cells, f.shape[1] - wall_band_cells)

    intervention = np.zeros((yc.size, xc.size), dtype=np.float64)
    centered_total = np.zeros_like(intervention)
    zeroed_weight = np.zeros_like(intervention)
    total_weight = 2.0 * float(np.sum(w))

    center = f[ys, xs]
    for axis in ("x", "y"):
        if axis == "x":
            left = center - f[ys, slice(wall_band_cells - 1, f.shape[1] - wall_band_cells - 1)]
            right = f[ys, slice(wall_band_cells + 1, f.shape[1] - wall_band_cells + 1)] - center
        else:
            left = center - f[slice(wall_band_cells - 1, f.shape[0] - wall_band_cells - 1), xs]
            right = f[slice(wall_band_cells + 1, f.shape[0] - wall_band_cells + 1), xs] - center

        same = ((left > 0.0) & (right > 0.0)) | ((left < 0.0) & (right < 0.0))
        limited = np.where(same, np.sign(left) * np.minimum(np.abs(left), np.abs(right)), 0.0)
        centered = 0.5 * (left + right)
        ww = w[None, None, :]
        intervention += np.sum(np.abs(centered - limited) * ww, axis=-1)
        centered_total += np.sum(np.abs(centered) * ww, axis=-1)
        zeroed = (~same) & ((left != 0.0) | (right != 0.0))
        zeroed_weight += np.sum(zeroed * ww, axis=-1)

    intervention_fraction = np.divide(
        intervention,
        centered_total,
        out=np.zeros_like(intervention),
        where=centered_total > 0.0,
    )
    zeroed_weight_fraction = zeroed_weight / max(total_weight, 1.0e-300)
    return {
        "intervention_fraction": intervention_fraction,
        "zeroed_velocity_weight_fraction": zeroed_weight_fraction,
        "centered_slope_weighted_abs": centered_total,
        "limiter_change_weighted_abs": intervention,
    }


def _support_colocation_metrics(
    intervention: np.ndarray,
    common_high_gradient_mask: np.ndarray,
    pair_gradient_weight: np.ndarray,
) -> tuple[dict[str, float | int], np.ndarray]:
    a = np.asarray(intervention, dtype=np.float64)
    common = np.asarray(common_high_gradient_mask, dtype=bool)
    pair_weight = np.asarray(pair_gradient_weight, dtype=np.float64)
    if a.shape != common.shape or a.shape != pair_weight.shape:
        raise ValueError("Stage-107 colocation maps have inconsistent shapes")
    if not np.isfinite(a).all() or not np.isfinite(pair_weight).all():
        raise ValueError("Stage-107 colocation maps are nonfinite")

    outside = ~common
    mean_inside = float(np.mean(a[common])) if np.any(common) else 0.0
    mean_outside = float(np.mean(a[outside])) if np.any(outside) else 0.0
    enrichment = _safe_ratio(mean_inside, mean_outside)

    threshold = float(np.quantile(a, SUPPORT_QUANTILE))
    high = a >= threshold
    n_high = int(np.count_nonzero(high))
    n_common = int(np.count_nonzero(common))
    n_overlap = int(np.count_nonzero(high & common))
    n_union = int(np.count_nonzero(high | common))
    overlap = _safe_ratio(float(n_overlap), float(min(n_high, n_common)))
    jaccard = _safe_ratio(float(n_overlap), float(n_union))
    pair_share = _safe_ratio(float(np.sum(pair_weight[high])), float(np.sum(pair_weight)))

    metrics: dict[str, float | int] = {
        "mean_inside_stage106_common_high_gradient_support": mean_inside,
        "mean_outside_stage106_common_high_gradient_support": mean_outside,
        "inside_to_outside_enrichment": enrichment,
        "upper_quartile_threshold": threshold,
        "upper_quartile_cell_count": n_high,
        "stage106_common_high_gradient_cell_count": n_common,
        "overlap_cell_count": n_overlap,
        "upper_quartile_overlap_coefficient": overlap,
        "upper_quartile_jaccard": jaccard,
        "stage106_pair_gradient_weight_share_in_high_intervention_support": pair_share,
    }
    return metrics, high


def stage107_decision(metrics: dict[str, object], finite: bool) -> str:
    if not finite:
        return "stage107_nonfinite_limiter_colocation_blocker_without_retuning"
    common = metrics["joint_intervention_colocation"]
    assert isinstance(common, dict)
    enrichment = float(common["inside_to_outside_enrichment"])
    overlap = float(common["upper_quartile_overlap_coefficient"])
    pair_share = float(common["stage106_pair_gradient_weight_share_in_high_intervention_support"])

    passed = (
        enrichment >= COLOCATION_ENRICHMENT_GUARD,
        overlap >= COLOCATION_OVERLAP_GUARD,
        pair_share >= PAIR_WEIGHT_SHARE_GUARD,
    )
    if all(passed):
        return "stage107_limiter_intervention_colocated_stage108_limiter_severity_correction_amplitude_coupling_audit"
    if sum(passed) >= 2:
        return "stage107_partial_limiter_colocation_stage108_continuous_intervention_rank_coupling_audit"
    return "stage107_no_material_limiter_colocation_stage108_unlimited_gradient_smoothness_audit"


def run_stage107(
    stage67_artifact_dir: str | Path,
    stage106_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage107_design(**design)
    stage67_summary = _validate_stage67(stage67_artifact_dir)
    stage106_summary, stage106_maps = _load_and_validate_stage106(stage106_artifact_dir)

    stage67_path = Path(stage67_artifact_dir) / "converged_full_distributions.npz"
    distribution_maps: dict[str, dict[str, np.ndarray]] = {}
    shell_speed_min = shell_speed_max = shell_speed_mean = float("nan")
    shell_point_count = 0

    with np.load(stage67_path) as saved:
        vx = np.asarray(saved["vx"], dtype=np.float64)
        vy = np.asarray(saved["vy"], dtype=np.float64)
        weight = np.asarray(saved["weight"], dtype=np.float64)
        shell_index = radial_shell_indices(vx, vy)
        shell_mask = shell_index == DOMINANT_RADIAL_SHELL
        shell_point_count = int(np.count_nonzero(shell_mask))
        expected_points = RADIAL_NODES_PER_SHELL * RULE[1]
        if shell_point_count != expected_points:
            raise ValueError("Stage-107 shell-1 point count does not match the frozen 10x96 support")
        speed = np.hypot(vx, vy)
        shell_speed_min = float(np.min(speed[shell_mask]))
        shell_speed_max = float(np.max(speed[shell_mask]))
        shell_speed_mean = float(np.mean(speed[shell_mask]))
        shell_weight = weight[shell_mask]

        for distribution in ("phi", "psi"):
            full = np.asarray(saved[distribution], dtype=np.float64)
            if full.shape != (GRID[0], GRID[1], RULE[0] * RULE[1]):
                raise ValueError(f"Stage-67 {distribution} has the wrong frozen shape")
            selected = np.asarray(full[..., shell_mask], dtype=np.float64).copy()
            distribution_maps[distribution] = _limiter_intervention_maps(selected, shell_weight)
            del selected, full

    common_high = stage106_maps["common_upper_quartile_mask"].astype(bool)
    pair_weight = np.asarray(stage106_maps["magnitude_product_weight"], dtype=np.float64)

    metrics: dict[str, object] = {}
    high_supports: dict[str, np.ndarray] = {}
    for distribution in ("phi", "psi"):
        m, high = _support_colocation_metrics(
            distribution_maps[distribution]["intervention_fraction"],
            common_high,
            pair_weight,
        )
        metrics[f"{distribution}_intervention_colocation"] = m
        high_supports[distribution] = high

    joint_intervention = np.minimum(
        distribution_maps["phi"]["intervention_fraction"],
        distribution_maps["psi"]["intervention_fraction"],
    )
    joint_metrics, joint_high = _support_colocation_metrics(joint_intervention, common_high, pair_weight)
    metrics["joint_intervention_colocation"] = joint_metrics
    metrics["phi_psi_intervention_pearson"] = _centered_pearson(
        distribution_maps["phi"]["intervention_fraction"],
        distribution_maps["psi"]["intervention_fraction"],
    )
    metrics["mean_intervention_fraction"] = {
        distribution: float(np.mean(distribution_maps[distribution]["intervention_fraction"]))
        for distribution in ("phi", "psi")
    }
    metrics["mean_zeroed_velocity_weight_fraction"] = {
        distribution: float(np.mean(distribution_maps[distribution]["zeroed_velocity_weight_fraction"]))
        for distribution in ("phi", "psi")
    }

    output_arrays = {
        "phi_intervention_fraction": distribution_maps["phi"]["intervention_fraction"],
        "psi_intervention_fraction": distribution_maps["psi"]["intervention_fraction"],
        "joint_intervention_fraction": joint_intervention,
        "phi_zeroed_velocity_weight_fraction": distribution_maps["phi"]["zeroed_velocity_weight_fraction"],
        "psi_zeroed_velocity_weight_fraction": distribution_maps["psi"]["zeroed_velocity_weight_fraction"],
        "stage106_common_high_gradient_mask": common_high.astype(np.uint8),
        "phi_high_intervention_support": high_supports["phi"].astype(np.uint8),
        "psi_high_intervention_support": high_supports["psi"].astype(np.uint8),
        "joint_high_intervention_support": joint_high.astype(np.uint8),
        "stage106_pair_gradient_weight": pair_weight,
    }
    finite = all(np.isfinite(np.asarray(value)).all() for value in output_arrays.values())
    scalar_metrics = [
        float(metrics["phi_psi_intervention_pearson"]),
        *(float(v) for v in metrics["mean_intervention_fraction"].values()),
        *(float(v) for v in metrics["mean_zeroed_velocity_weight_fraction"].values()),
    ]
    for name in (
        "phi_intervention_colocation",
        "psi_intervention_colocation",
        "joint_intervention_colocation",
    ):
        block = metrics[name]
        assert isinstance(block, dict)
        scalar_metrics.extend(float(v) for v in block.values())
    finite = finite and bool(np.isfinite(scalar_metrics).all())
    decision = stage107_decision(metrics, finite)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "limiter_activation_colocation_maps.npz", **output_arrays)

    result: dict[str, object] = {
        "stage": 107,
        "description": (
            "Frozen pre-replay minmod-limiter intervention colocation audit. It measures where, in the "
            "Stage-67 initial distribution and Stage-102 dominant radial shell 1, minmod changes the "
            "centered x/y slope and asks whether that pre-existing intervention is enriched under the "
            "later Stage-106 common high-gradient shell-1 growth support."
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
            "correction_floor": STAGE41_CORRECTION_FLOOR,
            "diagnostic_steps": DIAGNOSTIC_STEPS,
            "wall_band_cells": WALL_BAND_CELLS,
            "dominant_radial_shell": DOMINANT_RADIAL_SHELL,
            "shell_velocity_point_count": shell_point_count,
            "shell_speed_minimum": shell_speed_min,
            "shell_speed_maximum": shell_speed_max,
            "shell_speed_mean": shell_speed_mean,
            "support_quantile": SUPPORT_QUANTILE,
            "colocation_enrichment_guard": COLOCATION_ENRICHMENT_GUARD,
            "colocation_overlap_guard": COLOCATION_OVERLAP_GUARD,
            "pair_weight_share_guard": PAIR_WEIGHT_SHARE_GUARD,
            "parent_closure_tolerance": PARENT_CLOSURE_TOLERANCE,
            "activation_definition": ACTIVATION_DEFINITION,
            "activation_state": "stage67_pre_replay_initial_state",
            "stage67_run_id": STAGE67_RUN_ID,
            "stage67_job_id": STAGE67_JOB_ID,
            "stage67_artifact_id": STAGE67_ARTIFACT_ID,
            "stage67_artifact_sha256": STAGE67_ARTIFACT_SHA256,
            "stage106_run_id": STAGE106_RUN_ID,
            "stage106_job_id": STAGE106_JOB_ID,
            "stage106_artifact_id": STAGE106_ARTIFACT_ID,
            "stage106_artifact_sha256": STAGE106_ARTIFACT_SHA256,
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
        "stage106_authorization": {
            "decision": stage106_summary["decision"],
            "gradient_magnitude_cosine": stage106_summary["metrics"]["gradient_magnitude_cosine"],
            "gradient_magnitude_pearson": stage106_summary["metrics"]["gradient_magnitude_pearson"],
            "upper_quartile_overlap_coefficient": stage106_summary["metrics"]["upper_quartile_overlap_coefficient"],
            "common_upper_quartile_pair_weight_share": stage106_summary["metrics"]["common_upper_quartile_pair_weight_share"],
        },
        "finite": finite,
        "metrics": metrics,
        "decision": decision,
        "scientific_conclusion": (
            "Stage 107 is a precursor-association diagnostic only. Colocation would show that the frozen "
            "minmod intervention pattern present before the 25-step replay is spatially associated with "
            "the later common shell-1 correction-growth gradients; lack of colocation would reject a simple "
            "limiter-activation precursor interpretation. Neither outcome establishes causality, nonlinear "
            "MUSCL stability, endpoint convergence, accuracy, heat-flux improvement, or validation."
        ),
        "negative_result_guard": (
            "Stage 106 remains a gradient-magnitude organization diagnostic; Stage 105 remains directional "
            "alignment without strong single-axis dominance; Stage 104 remains a mesoscopic scale diagnostic; "
            "Stage 103 remains spatially diffuse; Stage 102 remains a velocity-shell localization result; "
            "Stage 101 remains angularly diffuse; Stage 100 is same-run attribution only; Stage 99 remains a "
            "negative cross-run reproducibility result; Stage 98 remains a negative cross-run replay result; "
            "Stage 90 remains nonconverged in both reconstruction arms; Stage 28 remains a failed MUSCL endpoint; "
            "and the Stage-89 one-sided boundary slope is not promoted. No failed parameter is retuned, no "
            "cross-Knudsen MUSCL extension is permitted, and no stability, accuracy, benchmark, heat-flux-"
            "improvement, or validation claim is authorized."
        ),
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Stage-107 limiter-activation colocation audit")
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage106-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_stage107(args.stage67_artifact_dir, args.stage106_artifact_dir, args.output_dir)


if __name__ == "__main__":
    main()
