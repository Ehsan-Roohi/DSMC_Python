from __future__ import annotations

import argparse
import hashlib
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

STAGE107_RUN_ID = 31514716731
STAGE107_JOB_ID = 93856877768
STAGE107_ARTIFACT_ID = 9116979302
STAGE107_ARTIFACT_SHA256 = "20d6f21059699507e3aa0a66cb10a15b5e8ad8fd4faca83bed8cc4f700608b82"
STAGE107_SUMMARY_SHA256 = "7f9b03aa09048d2b6b2b0de678e83dcfa6b3dce623a04c0b9cd24730efac409d"
STAGE107_MAPS_SHA256 = "99f7e1e5b1716b2a4544a9c5828a580e01c04dea9864226ecc16daaf332964ad"
STAGE107_DECISION = "stage107_limiter_intervention_colocated_stage108_limiter_severity_correction_amplitude_coupling_audit"

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
RANK_COUPLING_GUARD = 0.40
QUARTILE_AMPLITUDE_RATIO_GUARD = 1.50
UPPER_QUANTILE = 0.75
LOWER_QUANTILE = 0.25


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_stage108_design(**overrides: object) -> None:
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
        "rank_coupling_guard": RANK_COUPLING_GUARD,
        "quartile_amplitude_ratio_guard": QUARTILE_AMPLITUDE_RATIO_GUARD,
        "upper_quantile": UPPER_QUANTILE,
        "lower_quantile": LOWER_QUANTILE,
        "stage104_run_id": STAGE104_RUN_ID,
        "stage107_run_id": STAGE107_RUN_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 108 is frozen to the exact completed Stage-104 correction-growth amplitude maps and "
            "Stage-107 limiter-severity maps. It may not retune physics, collision/source treatment, floors, "
            "source relaxation, transport, wall treatment, limiter, velocity quadrature, normalization, "
            "thresholds, diagnostic window, or the failed MUSCL endpoint."
        )
    if INTERIOR_EXTENT != GRID[0] - 2 * WALL_BAND_CELLS:
        raise ValueError("Stage 108 requires the exact 56x56 four-cell-excluded interior")
    if not (0.0 < RANK_COUPLING_GUARD < 1.0):
        raise ValueError("Stage-108 rank-coupling guard must remain inside (0,1)")
    if QUARTILE_AMPLITUDE_RATIO_GUARD <= 1.0:
        raise ValueError("Stage-108 quartile amplitude ratio guard must remain above unity")
    if (LOWER_QUANTILE, UPPER_QUANTILE) != (0.25, 0.75):
        raise ValueError("Stage-108 severity strata are fixed to lower/upper quartiles")


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
        raise ValueError("Stage-108 rank input is nonfinite")
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


def _load_stage104(root: str | Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    root = Path(root)
    expected = {
        "summary.json": STAGE104_SUMMARY_SHA256,
        "interior_gradient_scale_maps.npz": STAGE104_MAPS_SHA256,
    }
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Stage-104 checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 104 or summary.get("decision") != STAGE104_DECISION or summary.get("finite") is not True:
        raise ValueError("Stage-104 completed endpoint is not the frozen amplitude parent")
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
        raise ValueError("Stage-104 configuration does not match the frozen Stage-108 design")
    for k, v in cfg.items():
        if k.endswith("_retuning") and v is not False:
            raise ValueError("Stage-104 parent reports forbidden retuning")
    if cfg.get("failed_muscl_endpoint_rehabilitated") is not False or cfg.get("cross_knudsen_extension_permitted") is not False:
        raise ValueError("Stage 108 cannot consume a rehabilitated or cross-Knudsen MUSCL endpoint")
    with np.load(root / "interior_gradient_scale_maps.npz") as data:
        if set(data.files) != {"phi_growth_map", "psi_growth_map", "interior_mask", "lags_cells"}:
            raise ValueError("Stage-104 map payload is unexpected")
        arrays = {name: np.asarray(data[name]).copy() for name in data.files}
    for name in ("phi_growth_map", "psi_growth_map"):
        a = arrays[name]
        if a.shape != (INTERIOR_EXTENT, INTERIOR_EXTENT) or not np.isfinite(a).all() or np.any(a < 0.0):
            raise ValueError(f"Stage-104 {name} is invalid")
    return summary, arrays


def _load_stage107(root: str | Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    root = Path(root)
    expected = {
        "summary.json": STAGE107_SUMMARY_SHA256,
        "limiter_activation_colocation_maps.npz": STAGE107_MAPS_SHA256,
    }
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Stage-107 checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 107 or summary.get("decision") != STAGE107_DECISION or summary.get("finite") is not True:
        raise ValueError("Stage-107 completed endpoint does not authorize Stage 108")
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
        raise ValueError("Stage-107 configuration does not match the frozen Stage-108 design")
    for k, v in cfg.items():
        if k.endswith("_retuning") and v is not False:
            raise ValueError("Stage-107 parent reports forbidden retuning")
    if cfg.get("failed_muscl_endpoint_rehabilitated") is not False or cfg.get("cross_knudsen_extension_permitted") is not False:
        raise ValueError("Stage 108 cannot consume a rehabilitated or cross-Knudsen MUSCL endpoint")
    with np.load(root / "limiter_activation_colocation_maps.npz") as data:
        needed = {
            "phi_intervention_fraction",
            "psi_intervention_fraction",
            "joint_intervention_fraction",
            "phi_zeroed_velocity_weight_fraction",
            "psi_zeroed_velocity_weight_fraction",
        }
        if not needed.issubset(data.files):
            raise ValueError("Stage-107 map payload is missing severity fields")
        arrays = {name: np.asarray(data[name], dtype=np.float64).copy() for name in needed}
    for name, a in arrays.items():
        if a.shape != (INTERIOR_EXTENT, INTERIOR_EXTENT) or not np.isfinite(a).all() or np.any(a < 0.0):
            raise ValueError(f"Stage-107 {name} is invalid")
    return summary, arrays


def _coupling_metrics(
    severity: np.ndarray,
    amplitude: np.ndarray,
) -> tuple[dict[str, float | int], dict[str, np.ndarray]]:
    s = np.asarray(severity, dtype=np.float64)
    a = np.asarray(amplitude, dtype=np.float64)
    if s.shape != a.shape or s.shape != (INTERIOR_EXTENT, INTERIOR_EXTENT):
        raise ValueError("Stage-108 severity/amplitude maps must share the exact 56x56 shape")
    if not np.isfinite(s).all() or not np.isfinite(a).all() or np.any(s < 0.0) or np.any(a < 0.0):
        raise ValueError("Stage-108 severity/amplitude maps must be finite and nonnegative")
    q1 = float(np.quantile(s, LOWER_QUANTILE))
    q3 = float(np.quantile(s, UPPER_QUANTILE))
    low = s <= q1
    high = s >= q3
    mean_low = float(np.mean(a[low]))
    mean_high = float(np.mean(a[high]))
    median_low = float(np.median(a[low]))
    median_high = float(np.median(a[high]))
    amplitude_sum = float(np.sum(a))
    metrics: dict[str, float | int] = {
        "pearson": _centered_pearson(s, a),
        "spearman": _spearman(s, a),
        "severity_lower_quartile_threshold": q1,
        "severity_upper_quartile_threshold": q3,
        "lower_quartile_cell_count": int(np.count_nonzero(low)),
        "upper_quartile_cell_count": int(np.count_nonzero(high)),
        "mean_amplitude_lower_severity_quartile": mean_low,
        "mean_amplitude_upper_severity_quartile": mean_high,
        "upper_to_lower_mean_amplitude_ratio": _safe_ratio(mean_high, mean_low),
        "upper_to_lower_median_amplitude_ratio": _safe_ratio(median_high, median_low),
        "upper_severity_quartile_amplitude_share": _safe_ratio(float(np.sum(a[high])), amplitude_sum),
    }
    return metrics, {
        "low_support": low.astype(np.uint8),
        "high_support": high.astype(np.uint8),
    }


def stage108_decision(metrics: dict[str, object], finite: bool) -> str:
    if not finite:
        return "stage108_nonfinite_severity_amplitude_coupling_blocker_without_retuning"
    blocks = [metrics["phi"], metrics["psi"], metrics["joint"]]
    assert all(isinstance(block, dict) for block in blocks)
    rank_pass = all(float(block["spearman"]) >= RANK_COUPLING_GUARD for block in blocks)
    ratio_pass = all(
        float(block["upper_to_lower_mean_amplitude_ratio"]) >= QUARTILE_AMPLITUDE_RATIO_GUARD
        for block in blocks
    )
    if rank_pass and ratio_pass:
        return "stage108_continuous_limiter_severity_coupling_stage109_limiter_intervention_mode_decomposition_audit"
    if rank_pass or ratio_pass:
        return "stage108_partial_continuous_severity_coupling_stage109_spatial_monotonicity_audit"
    return "stage108_no_continuous_severity_coupling_stage109_unlimited_gradient_smoothness_audit"


def run_stage108(
    stage104_artifact_dir: str | Path,
    stage107_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage108_design(**design)
    stage104, a104 = _load_stage104(stage104_artifact_dir)
    stage107, a107 = _load_stage107(stage107_artifact_dir)

    metrics: dict[str, object] = {}
    output: dict[str, np.ndarray] = {}
    for distribution in ("phi", "psi"):
        severity = a107[f"{distribution}_intervention_fraction"]
        amplitude = a104[f"{distribution}_growth_map"]
        block, support = _coupling_metrics(severity, amplitude)
        metrics[distribution] = block
        output[f"{distribution}_intervention_fraction"] = severity
        output[f"{distribution}_growth_amplitude"] = amplitude
        output[f"{distribution}_low_severity_support"] = support["low_support"]
        output[f"{distribution}_high_severity_support"] = support["high_support"]

    joint_severity = a107["joint_intervention_fraction"]
    joint_amplitude = np.sqrt(a104["phi_growth_map"] * a104["psi_growth_map"])
    joint_block, joint_support = _coupling_metrics(joint_severity, joint_amplitude)
    metrics["joint"] = joint_block
    output["joint_intervention_fraction"] = joint_severity
    output["joint_growth_amplitude"] = joint_amplitude
    output["joint_low_severity_support"] = joint_support["low_support"]
    output["joint_high_severity_support"] = joint_support["high_support"]
    output["phi_zeroed_velocity_weight_fraction"] = a107["phi_zeroed_velocity_weight_fraction"]
    output["psi_zeroed_velocity_weight_fraction"] = a107["psi_zeroed_velocity_weight_fraction"]

    finite = all(np.isfinite(v).all() for v in output.values())
    for block in metrics.values():
        assert isinstance(block, dict)
        finite = finite and bool(np.isfinite([float(v) for v in block.values()]).all())
    decision = stage108_decision(metrics, finite)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "limiter_severity_correction_amplitude_coupling_maps.npz",
        **output,
    )
    result: dict[str, object] = {
        "stage": 108,
        "description": (
            "Frozen artifact-only continuous coupling audit between Stage-107 pre-replay minmod "
            "intervention severity and the Stage-104 25-step shell-1 correction-growth amplitude. "
            "It replaces threshold-only colocation with rank and quartile-stratified amplitude "
            "diagnostics without rerunning or retuning the solver."
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
            "rank_coupling_guard": RANK_COUPLING_GUARD,
            "quartile_amplitude_ratio_guard": QUARTILE_AMPLITUDE_RATIO_GUARD,
            "lower_quantile": LOWER_QUANTILE,
            "upper_quantile": UPPER_QUANTILE,
            "stage104_run_id": STAGE104_RUN_ID,
            "stage104_job_id": STAGE104_JOB_ID,
            "stage104_artifact_id": STAGE104_ARTIFACT_ID,
            "stage104_artifact_sha256": STAGE104_ARTIFACT_SHA256,
            "stage107_run_id": STAGE107_RUN_ID,
            "stage107_job_id": STAGE107_JOB_ID,
            "stage107_artifact_id": STAGE107_ARTIFACT_ID,
            "stage107_artifact_sha256": STAGE107_ARTIFACT_SHA256,
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
        "stage104_authorization": {
            "decision": stage104["decision"],
            "phi_characteristic_gradient_length_cells": stage104["metrics"]["phi"]["characteristic_gradient_length_cells"],
            "psi_characteristic_gradient_length_cells": stage104["metrics"]["psi"]["characteristic_gradient_length_cells"],
        },
        "stage107_authorization": {
            "decision": stage107["decision"],
            "joint_inside_to_outside_enrichment": stage107["metrics"]["joint_intervention_colocation"]["inside_to_outside_enrichment"],
            "joint_upper_quartile_overlap_coefficient": stage107["metrics"]["joint_intervention_colocation"]["upper_quartile_overlap_coefficient"],
            "joint_pair_gradient_weight_share": stage107["metrics"]["joint_intervention_colocation"]["stage106_pair_gradient_weight_share_in_high_intervention_support"],
        },
        "finite": finite,
        "metrics": metrics,
        "decision": decision,
        "scientific_conclusion": (
            "Stage 108 tests whether stronger frozen pre-replay limiter intervention is monotonically "
            "associated with larger later shell-1 correction-growth amplitude. Even strong rank/quartile "
            "coupling is an association diagnostic only and does not establish limiter causality, nonlinear "
            "solver stability, endpoint convergence, Table 3/Table 6 improvement, heat-flux accuracy, or validation."
        ),
        "negative_result_guard": (
            "Stage 107 is precursor association only; Stage 106 remains gradient-magnitude organization; "
            "Stage 105 remains directional alignment without strong single-axis dominance; Stage 104 remains "
            "mesoscopic; Stage 103 remains spatially diffuse; Stage 102 remains velocity-shell localization; "
            "Stage 101 remains angularly diffuse; Stage 100 is same-run attribution only; Stage 99 remains a "
            "negative cross-run reproducibility result; Stage 98 remains a negative cross-run replay result; "
            "Stage 90 remains nonconverged in both reconstruction arms; Stage 28 remains a failed MUSCL endpoint; "
            "and the Stage-89 one-sided boundary slope is not promoted. No failed parameter is retuned, no "
            "cross-Knudsen MUSCL extension is permitted, and no stability, accuracy, benchmark, heat-flux-"
            "improvement, or validation claim is authorized."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen Stage-108 limiter-severity/correction-amplitude coupling audit"
    )
    parser.add_argument("--stage104-artifact-dir", required=True)
    parser.add_argument("--stage107-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_stage108(
        args.stage104_artifact_dir,
        args.stage107_artifact_dir,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
