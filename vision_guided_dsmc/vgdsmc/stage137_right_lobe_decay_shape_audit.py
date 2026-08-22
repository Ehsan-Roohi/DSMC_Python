from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = 137
EXPECTED_STAGE136_SOURCE_HEAD = "a498faa95cecf9712b8401f389e6776f02e4aa0f"
EXPECTED_STAGE136_RUN_ID = 32140741740
EXPECTED_STAGE136_JOB_ID = 95722525504
EXPECTED_STAGE136_ARTIFACT_ID = 9325758887
EXPECTED_STAGE136_ARTIFACT_SHA256 = "176d1cb7e6b45ba7e780d1cd09254ee4c0aa27dab086eca1de92e2b891e5f0b0"
EXPECTED_STAGE136_SUMMARY_SHA256 = "eaf3706b2c7bca1678b0680dcf06daa076ccd36eeb91dccb9b8602bd8378b64c"
EXPECTED_STAGE136_PAYLOAD_SHA256 = "f8829d07450f3832dc45a878943ac69415390e89208f9bb859e838fa9c160015"
EXPECTED_STAGE136_DECISION = "stage136_distributed_common_right_lobe_support_stage137_right_lobe_decay_shape_audit"

LOG_LINEAR_R2_MIN = 0.90
RELATIVE_L2_RESIDUAL_MAX = 0.10
MONOTONE_STEP_FRACTION_MIN = 5.0 / 6.0
COMMON_RATE_RELATIVE_DIFFERENCE_MAX = 0.25
PROFILE_CLOSURE_MAX = 1.0e-12

NONFINITE = "stage137_nonfinite_blocker"
PARENT_RECORD_BLOCKER = "stage137_parent_record_blocker"
PARENT_PROFILE_BLOCKER = "stage137_parent_profile_closure_blocker"
SMOOTH_COMMON_RATE = "stage137_smooth_common_decay_rate_stage138_common_decay_scale_audit"
SMOOTH_RATE_SPLIT = "stage137_smooth_channel_rate_split_stage138_channel_rate_origin_audit"
RESOLVED_STRUCTURE = "stage137_resolved_decay_structure_stage138_curvature_localization_audit"


def validate_stage137_design(
    *,
    grid=(64, 64),
    interior_grid=(56, 56),
    kn0=10.0,
    cold_hot_ratio=0.1,
    rule=(40, 96),
    radial_scale=2.0,
    limiter="minmod",
    boundary_slope="zero",
    source_relaxation=1.0,
    correction_floor=0.05,
    witness_node=9,
    pair_sectors=(5, 6),
    dominant_mirrored_sector=6,
    log_linear_r2_min=LOG_LINEAR_R2_MIN,
    relative_l2_residual_max=RELATIVE_L2_RESIDUAL_MAX,
    monotone_step_fraction_min=MONOTONE_STEP_FRACTION_MIN,
    common_rate_relative_difference_max=COMMON_RATE_RELATIVE_DIFFERENCE_MAX,
    solver_rerun=False,
    solver_endpoint_advanced=False,
    physical_parameter_retuning=False,
):
    expected = {
        "grid": (64, 64),
        "interior_grid": (56, 56),
        "kn0": 10.0,
        "cold_hot_ratio": 0.1,
        "rule": (40, 96),
        "radial_scale": 2.0,
        "limiter": "minmod",
        "boundary_slope": "zero",
        "source_relaxation": 1.0,
        "correction_floor": 0.05,
        "witness_node": 9,
        "pair_sectors": (5, 6),
        "dominant_mirrored_sector": 6,
        "log_linear_r2_min": LOG_LINEAR_R2_MIN,
        "relative_l2_residual_max": RELATIVE_L2_RESIDUAL_MAX,
        "monotone_step_fraction_min": MONOTONE_STEP_FRACTION_MIN,
        "common_rate_relative_difference_max": COMMON_RATE_RELATIVE_DIFFERENCE_MAX,
        "solver_rerun": False,
        "solver_endpoint_advanced": False,
        "physical_parameter_retuning": False,
    }
    got = locals()
    for key, value in expected.items():
        if got[key] != value:
            raise ValueError(f"Stage 137 frozen-design violation: {key}={got[key]!r}, expected {value!r}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_stage136_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 136
        and record.get("source_head") == EXPECTED_STAGE136_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE136_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE136_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE136_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE136_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE136_SUMMARY_SHA256
        and record.get("right_lobe_depth_support_sha256") == EXPECTED_STAGE136_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE136_DECISION
    )


def profile_cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.ndim != 1 or bb.ndim != 1 or aa.shape != bb.shape:
        raise ValueError("Stage 137 profile cosine requires equal one-dimensional arrays")
    if not np.isfinite(np.concatenate([aa, bb])).all():
        raise ValueError("Stage 137 profile cosine inputs must be finite")
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    return float(np.dot(aa, bb) / (na * nb)) if na > 0.0 and nb > 0.0 else 0.0


def decay_metrics(depth: np.ndarray, magnitude: np.ndarray) -> tuple[dict[str, float | int | bool], np.ndarray]:
    x = np.asarray(depth, dtype=float)
    y = np.asarray(magnitude, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 4:
        raise ValueError("Stage 137 decay audit requires equal one-dimensional profiles with at least four samples")
    if not np.isfinite(np.concatenate([x, y])).all() or np.any(y <= 0.0):
        raise ValueError("Stage 137 decay profiles must be finite and strictly positive in magnitude")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("Stage 137 depth coordinates must be strictly increasing")

    z = x - x[0]
    logy = np.log(y)
    design = np.column_stack([np.ones_like(z), z])
    intercept, slope = np.linalg.lstsq(design, logy, rcond=None)[0]
    logfit = design @ np.array([intercept, slope])
    fit = np.exp(logfit)
    ss_res = float(np.sum((logy - logfit) ** 2))
    ss_tot = float(np.sum((logy - np.mean(logy)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    relative_l2 = float(np.linalg.norm(y - fit) / np.linalg.norm(y))
    monotone_fraction = float(np.mean(np.diff(y) <= 0.0))
    local_rates = -np.diff(logy) / np.diff(x)
    decay_rate = float(-slope)
    mean_local_rate = float(np.mean(local_rates))
    local_rate_cv = float(np.std(local_rates) / abs(mean_local_rate)) if abs(mean_local_rate) > 0.0 else float("inf")
    metrics = {
        "sample_count": int(x.size),
        "positive_decay_rate": bool(decay_rate > 0.0),
        "decay_rate_per_cell": decay_rate,
        "diagnostic_log_intercept": float(intercept),
        "log_linear_r2": float(r2),
        "relative_l2_residual": relative_l2,
        "nonincreasing_step_fraction": monotone_fraction,
        "minimum_local_decay_rate_per_cell": float(np.min(local_rates)),
        "maximum_local_decay_rate_per_cell": float(np.max(local_rates)),
        "local_decay_rate_cv": local_rate_cv,
        "endpoint_to_nearest_magnitude_ratio": float(y[-1] / y[0]),
    }
    return metrics, fit


def classify_decay_shape(
    *,
    dominant: dict,
    parent: dict,
    common_rate_relative_difference: float,
    finite: bool = True,
    parent_record_ok: bool = True,
    parent_profile_closure: float = 0.0,
) -> str:
    numeric = [float(common_rate_relative_difference), float(parent_profile_closure)]
    for block in (dominant, parent):
        numeric.extend(float(v) for v in block.values() if isinstance(v, (float, int)) and not isinstance(v, bool))
    if not finite or not np.isfinite(numeric).all():
        return NONFINITE
    if not parent_record_ok:
        return PARENT_RECORD_BLOCKER
    if parent_profile_closure > PROFILE_CLOSURE_MAX:
        return PARENT_PROFILE_BLOCKER

    def smooth(block: dict) -> bool:
        return bool(
            block["positive_decay_rate"]
            and float(block["log_linear_r2"]) >= LOG_LINEAR_R2_MIN
            and float(block["relative_l2_residual"]) <= RELATIVE_L2_RESIDUAL_MAX
            and float(block["nonincreasing_step_fraction"]) >= MONOTONE_STEP_FRACTION_MIN
        )

    if not (smooth(dominant) and smooth(parent)):
        return RESOLVED_STRUCTURE
    if float(common_rate_relative_difference) <= COMMON_RATE_RELATIVE_DIFFERENCE_MAX:
        return SMOOTH_COMMON_RATE
    return SMOOTH_RATE_SPLIT


def run_stage137(stage136_dir: Path, stage136_record: Path, output_dir: Path) -> dict:
    validate_stage137_design()
    parent_summary = _load_json(stage136_dir / "summary.json")
    parent_record = _load_json(stage136_record)
    record_ok = _check_stage136_record(parent_record)
    if parent_summary.get("stage") != 136 or parent_summary.get("decision") != EXPECTED_STAGE136_DECISION:
        raise ValueError("Stage 137 requires the completed Stage 136 distributed-common route")

    with np.load(stage136_dir / "right_lobe_depth_support.npz") as data:
        depth = np.asarray(data["right_depth"], dtype=float)
        dominant_difference = np.asarray(data["dominant_right_difference"], dtype=float)
        parent_difference = np.asarray(data["parent_right_difference"], dtype=float)
        dominant_parent_norm = np.asarray(data["dominant_normalized_magnitude"], dtype=float)
        parent_parent_norm = np.asarray(data["parent_normalized_magnitude"], dtype=float)

    dominant_magnitude = np.abs(dominant_difference)
    parent_magnitude = np.abs(parent_difference)
    dominant_norm = dominant_magnitude / np.sum(dominant_magnitude)
    parent_norm = parent_magnitude / np.sum(parent_magnitude)
    profile_array_closure = max(
        float(np.max(np.abs(dominant_norm - dominant_parent_norm))),
        float(np.max(np.abs(parent_norm - parent_parent_norm))),
    )
    cosine = profile_cosine(dominant_norm, parent_norm)
    parent_cosine = float(parent_summary["aggregate"]["normalized_profile_cosine"])
    parent_profile_closure = max(profile_array_closure, abs(cosine - parent_cosine))

    dominant_metrics, dominant_fit = decay_metrics(depth, dominant_magnitude)
    parent_metrics, parent_fit = decay_metrics(depth, parent_magnitude)
    kd = float(dominant_metrics["decay_rate_per_cell"])
    kp = float(parent_metrics["decay_rate_per_cell"])
    mean_rate = 0.5 * (abs(kd) + abs(kp))
    rate_relative_difference = abs(kd - kp) / mean_rate if mean_rate > 0.0 else float("inf")

    common_profile = np.sqrt(dominant_norm * parent_norm)
    common_profile /= np.sum(common_profile)
    common_metrics, common_fit = decay_metrics(depth, common_profile)
    finite = bool(
        parent_summary.get("finite", False)
        and np.isfinite(
            np.concatenate([
                depth,
                dominant_difference,
                parent_difference,
                dominant_fit,
                parent_fit,
                common_profile,
                common_fit,
            ])
        ).all()
    )
    decision = classify_decay_shape(
        dominant=dominant_metrics,
        parent=parent_metrics,
        common_rate_relative_difference=rate_relative_difference,
        finite=finite,
        parent_record_ok=record_ok,
        parent_profile_closure=parent_profile_closure,
    )

    cfg = dict(parent_summary["configuration"])
    cfg.update({
        "log_linear_r2_min": LOG_LINEAR_R2_MIN,
        "relative_l2_residual_max": RELATIVE_L2_RESIDUAL_MAX,
        "monotone_step_fraction_min": MONOTONE_STEP_FRACTION_MIN,
        "common_rate_relative_difference_max": COMMON_RATE_RELATIVE_DIFFERENCE_MAX,
        "diagnostic_decay_fit_used_for_solver": False,
        "amplitude_refit_applied": False,
        "phase_shift_applied": False,
        "width_refit_applied": False,
        "solver_rerun": False,
        "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False,
        "benchmark_or_validation_claim_permitted": False,
    })
    for key in list(cfg):
        if key.endswith("_retuning"):
            cfg[key] = False

    if decision == SMOOTH_RATE_SPLIT:
        conclusion = (
            "Both fixed Stage-136 right-lobe channels are individually consistent with smooth log-linear attenuation under the preregistered residual, R2, and monotonicity guards, but their diagnostic decay rates differ by more than the fixed common-rate tolerance. "
            "The next justified artifact-only diagnostic is therefore a channel-rate origin audit; no solver coefficient or physical parameter is changed. "
            "This does not establish limiter causality, MUSCL stability, endpoint convergence, q_av improvement, benchmark accuracy, or validation."
        )
    elif decision == SMOOTH_COMMON_RATE:
        conclusion = (
            "Both fixed Stage-136 right-lobe channels satisfy the preregistered smooth log-linear attenuation guards and their diagnostic decay rates agree within the fixed common-rate tolerance. "
            "The next justified artifact-only diagnostic is a common decay-scale audit; no solver coefficient or physical parameter is changed."
        )
    elif decision == RESOLVED_STRUCTURE:
        conclusion = (
            "At least one fixed Stage-136 right-lobe channel is not adequately represented by the preregistered smooth log-linear attenuation diagnostic. "
            "The next justified artifact-only diagnostic is to localize the resolved non-log-linear curvature; no solver coefficient or physical parameter is changed."
        )
    else:
        conclusion = "Stage 137 is blocked by a finite-data, parent-record, or parent-profile closure guard; no scientific routing is claimed."

    summary = {
        "stage": STAGE,
        "finite": finite,
        "configuration": cfg,
        "parents": {
            "stage136_run_id": EXPECTED_STAGE136_RUN_ID,
            "stage136_job_id": EXPECTED_STAGE136_JOB_ID,
            "stage136_artifact_id": EXPECTED_STAGE136_ARTIFACT_ID,
            "stage136_source_head": EXPECTED_STAGE136_SOURCE_HEAD,
        },
        "metrics": {
            "dominant_sector": dominant_metrics,
            "parent_profile": parent_metrics,
            "common_geometric_profile": common_metrics,
            "normalized_parent_profile_cosine": cosine,
            "channel_decay_rate_relative_difference": float(rate_relative_difference),
        },
        "aggregate": {
            "minimum_log_linear_r2": min(float(dominant_metrics["log_linear_r2"]), float(parent_metrics["log_linear_r2"])),
            "maximum_relative_l2_residual": max(float(dominant_metrics["relative_l2_residual"]), float(parent_metrics["relative_l2_residual"])),
            "minimum_nonincreasing_step_fraction": min(float(dominant_metrics["nonincreasing_step_fraction"]), float(parent_metrics["nonincreasing_step_fraction"])),
            "channel_decay_rate_relative_difference": float(rate_relative_difference),
            "common_profile_relative_l2_residual": float(common_metrics["relative_l2_residual"]),
            "maximum_parent_profile_closure": float(parent_profile_closure),
            "parent_record_ok": record_ok,
        },
        "decision": decision,
        "scientific_conclusion": conclusion,
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. The Stage-137 decay rates are diagnostic fits only and are not solver parameters. No physical, collision/source, floor, wall, reconstruction, transport, limiter, normalization, source-relaxation, or velocity-quadrature parameter is retuned; no solver endpoint or cross-Knudsen extension is advanced."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        output_dir / "right_lobe_decay_shape.npz",
        right_depth=depth,
        dominant_magnitude=dominant_magnitude,
        parent_magnitude=parent_magnitude,
        dominant_normalized_magnitude=dominant_norm,
        parent_normalized_magnitude=parent_norm,
        dominant_log_linear_fit=dominant_fit,
        parent_log_linear_fit=parent_fit,
        common_geometric_profile=common_profile,
        common_log_linear_fit=common_fit,
    )
    print(json.dumps({"stage": STAGE, "decision": decision, "aggregate": summary["aggregate"]}, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 137 fixed right-lobe decay-shape audit")
    parser.add_argument("--stage136-dir", type=Path, required=True)
    parser.add_argument("--stage136-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_stage137(args.stage136_dir, args.stage136_record, args.output_dir)


if __name__ == "__main__":
    main()
