from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = 143
EXPECTED_STAGE142_SOURCE_HEAD = "adfa66abb8f07a9226f69f1d621d1ed2ceb1a5af"
EXPECTED_STAGE142_RUN_ID = 32270422318
EXPECTED_STAGE142_JOB_ID = 96125222151
EXPECTED_STAGE142_ARTIFACT_ID = 9382343898
EXPECTED_STAGE142_ARTIFACT_SHA256 = "91e0056454a15fa46fc6734f34787682f4af8191114af13a85532ea666e90da0"
EXPECTED_STAGE142_SUMMARY_SHA256 = "29eaa1bbd8307fcbfabe50fddec0c4f983332ffd719828b1c8f0faa1a34838ea"
EXPECTED_STAGE142_PAYLOAD_SHA256 = "c6b7e92b4dae20ff324ccb66d248e4a87a939598b303c4f2b4c65b14fc7cc7ce"
EXPECTED_STAGE142_DECISION = "stage142_persistent_sampled_sign_with_weak_endpoint_margin_stage143_positive_lobe_scale_audit"

MIN_LATER_POSITIVE_SAMPLES = 4
MIN_LATER_POSITIVE_SIGN_COHERENCE = 1.0
COHERENT_CV_MAX = 0.5
COHERENT_EFFECTIVE_COUNT_MIN = 3.0
COHERENT_PEAK_TO_MEDIAN_MAX = 2.0

NONFINITE = "stage143_nonfinite_blocker"
PARENT_RECORD_BLOCKER = "stage143_parent_record_blocker"
PARENT_ROUTE_BLOCKER = "stage143_parent_route_blocker"
INSUFFICIENT_SUPPORT = "stage143_insufficient_positive_lobe_support_blocker"
BROAD_COHERENT = "stage143_broad_coherent_positive_lobe_scale_stage144_positive_lobe_shape_audit"
CONCENTRATED = "stage143_concentrated_positive_lobe_scale_stage144_positive_lobe_concentration_audit"
VARIABLE = "stage143_variable_positive_lobe_scale_stage144_positive_lobe_variability_audit"


def validate_stage143_design(
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
    min_later_positive_samples=MIN_LATER_POSITIVE_SAMPLES,
    min_later_positive_sign_coherence=MIN_LATER_POSITIVE_SIGN_COHERENCE,
    coherent_cv_max=COHERENT_CV_MAX,
    coherent_effective_count_min=COHERENT_EFFECTIVE_COUNT_MIN,
    coherent_peak_to_median_max=COHERENT_PEAK_TO_MEDIAN_MAX,
    solver_rerun=False,
    solver_endpoint_advanced=False,
    cross_knudsen_extension_permitted=False,
    benchmark_or_validation_claim_permitted=False,
    physical_parameter_retuning=False,
    collision_source_retuning=False,
    floor_retuning=False,
    wall_retuning=False,
    reconstruction_retuning=False,
    transport_retuning=False,
    limiter_retuning=False,
    normalization_retuning=False,
    source_relaxation_retuning=False,
    velocity_grid_retuning=False,
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
        "min_later_positive_samples": MIN_LATER_POSITIVE_SAMPLES,
        "min_later_positive_sign_coherence": MIN_LATER_POSITIVE_SIGN_COHERENCE,
        "coherent_cv_max": COHERENT_CV_MAX,
        "coherent_effective_count_min": COHERENT_EFFECTIVE_COUNT_MIN,
        "coherent_peak_to_median_max": COHERENT_PEAK_TO_MEDIAN_MAX,
        "solver_rerun": False,
        "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False,
        "benchmark_or_validation_claim_permitted": False,
        "physical_parameter_retuning": False,
        "collision_source_retuning": False,
        "floor_retuning": False,
        "wall_retuning": False,
        "reconstruction_retuning": False,
        "transport_retuning": False,
        "limiter_retuning": False,
        "normalization_retuning": False,
        "source_relaxation_retuning": False,
        "velocity_grid_retuning": False,
    }
    got = locals()
    for key, value in expected.items():
        if got[key] != value:
            raise ValueError(f"Stage 143 frozen-design violation: {key}={got[key]!r}, expected {value!r}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_stage142_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 142
        and record.get("source_head") == EXPECTED_STAGE142_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE142_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE142_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE142_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE142_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE142_SUMMARY_SHA256
        and record.get("sampled_sign_margin_sha256") == EXPECTED_STAGE142_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE142_DECISION
        and record.get("workflow_status") == "completed"
        and record.get("workflow_conclusion") == "success"
    )


def effective_support_count(values: np.ndarray) -> float:
    y = np.abs(np.asarray(values, dtype=float))
    if y.ndim != 1 or y.size == 0 or not np.isfinite(y).all():
        return float("nan")
    denom = float(np.dot(y, y))
    if denom == 0.0:
        return 0.0
    return float(np.sum(y) ** 2 / denom)


def classify_positive_lobe_scale(
    *,
    later_count: int,
    sign_coherence: float,
    coefficient_of_variation: float,
    effective_count: float,
    peak_to_median: float,
    parent_record_ok: bool = True,
    parent_route_ok: bool = True,
    sufficient_support: bool = True,
    finite: bool = True,
) -> str:
    numeric = [sign_coherence, coefficient_of_variation, effective_count, peak_to_median]
    if not finite or not np.isfinite(numeric).all():
        return NONFINITE
    if not parent_record_ok:
        return PARENT_RECORD_BLOCKER
    if not parent_route_ok:
        return PARENT_ROUTE_BLOCKER
    if not sufficient_support or later_count < MIN_LATER_POSITIVE_SAMPLES:
        return INSUFFICIENT_SUPPORT
    coherent = (
        sign_coherence >= MIN_LATER_POSITIVE_SIGN_COHERENCE
        and coefficient_of_variation <= COHERENT_CV_MAX
        and effective_count >= COHERENT_EFFECTIVE_COUNT_MIN
        and peak_to_median <= COHERENT_PEAK_TO_MEDIAN_MAX
    )
    if coherent:
        return BROAD_COHERENT
    if effective_count < COHERENT_EFFECTIVE_COUNT_MIN:
        return CONCENTRATED
    return VARIABLE


def run_stage143(stage142_dir: Path, stage142_record: Path, output_dir: Path) -> dict:
    validate_stage143_design()
    parent_summary = _load_json(stage142_dir / "summary.json")
    parent_record = _load_json(stage142_record)
    parent_record_ok = _check_stage142_record(parent_record)
    parent_route_ok = bool(
        parent_summary.get("stage") == 142
        and parent_summary.get("decision") == EXPECTED_STAGE142_DECISION
    )

    with np.load(stage142_dir / "sampled_sign_margin.npz") as data:
        positive_depth = np.asarray(data["positive_side_depth"], dtype=float)
        positive_values = np.asarray(data["positive_side_values"], dtype=float)
        later_values = np.asarray(data["later_positive_values"], dtype=float)

    finite = bool(
        parent_summary.get("finite", False)
        and np.isfinite(positive_depth).all()
        and np.isfinite(positive_values).all()
        and np.isfinite(later_values).all()
    )
    sufficient_support = bool(
        positive_depth.ndim == 1
        and positive_values.ndim == 1
        and later_values.ndim == 1
        and positive_depth.shape == positive_values.shape
        and positive_values.size == later_values.size + 1
        and later_values.size >= MIN_LATER_POSITIVE_SAMPLES
        and np.all(np.diff(positive_depth) > 0.0)
    )

    if sufficient_support:
        later_depth = positive_depth[1:]
        sign_coherence = float(np.mean(later_values > 0.0))
        abs_later = np.abs(later_values)
        mean_abs = float(np.mean(abs_later))
        median_abs = float(np.median(abs_later))
        std_abs = float(np.std(abs_later))
        coefficient_of_variation = float(std_abs / mean_abs) if mean_abs > 0.0 else float("nan")
        median_abs_deviation = float(np.median(np.abs(abs_later - median_abs)))
        relative_mad = float(median_abs_deviation / median_abs) if median_abs > 0.0 else float("nan")
        effective_count = effective_support_count(abs_later)
        peak_to_median = float(np.max(abs_later) / median_abs) if median_abs > 0.0 else float("nan")
        minimum_to_median = float(np.min(abs_later) / median_abs) if median_abs > 0.0 else float("nan")
        weak_endpoint_to_median = float(abs(positive_values[0]) / median_abs) if median_abs > 0.0 else float("nan")
        later_l1_fraction = float(np.sum(abs_later) / np.sum(np.abs(positive_values)))
        span_cells = float(later_depth[-1] - later_depth[0])
    else:
        later_depth = np.asarray([], dtype=float)
        sign_coherence = 0.0
        mean_abs = median_abs = std_abs = float("nan")
        coefficient_of_variation = relative_mad = float("nan")
        effective_count = peak_to_median = minimum_to_median = float("nan")
        weak_endpoint_to_median = later_l1_fraction = span_cells = float("nan")

    decision = classify_positive_lobe_scale(
        later_count=int(later_values.size),
        sign_coherence=sign_coherence,
        coefficient_of_variation=coefficient_of_variation,
        effective_count=effective_count,
        peak_to_median=peak_to_median,
        parent_record_ok=parent_record_ok,
        parent_route_ok=parent_route_ok,
        sufficient_support=sufficient_support,
        finite=finite,
    )

    cfg = dict(parent_summary.get("configuration", {}))
    cfg.update({
        "min_later_positive_samples": MIN_LATER_POSITIVE_SAMPLES,
        "min_later_positive_sign_coherence": MIN_LATER_POSITIVE_SIGN_COHERENCE,
        "coherent_cv_max": COHERENT_CV_MAX,
        "coherent_effective_count_min": COHERENT_EFFECTIVE_COUNT_MIN,
        "coherent_peak_to_median_max": COHERENT_PEAK_TO_MEDIAN_MAX,
        "positive_lobe_scale_used_for_solver": False,
        "solver_rerun": False,
        "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False,
        "benchmark_or_validation_claim_permitted": False,
    })
    for key in list(cfg):
        if key.endswith("_retuning"):
            cfg[key] = False

    if decision == BROAD_COHERENT:
        conclusion = (
            "The four later positive-side samples define a broad, coherent positive-lobe amplitude scale under the fixed Stage-143 guards: "
            "sign coherence is complete, amplitude dispersion is moderate, effective support spans at least three samples, and no single sample "
            "dominates the median scale. This strengthens the interpretation that the positive lobe is not created by the weak first positive "
            "endpoint alone. The next justified artifact-only stage is a positive-lobe shape audit; the measured scale is not fed back into the solver."
        )
    elif decision == CONCENTRATED:
        conclusion = (
            "The later positive-side support is too concentrated to define a broad characteristic lobe scale under the fixed Stage-143 guards. "
            "The next justified artifact-only stage is a positive-lobe concentration audit; no measured amplitude is used as a solver parameter."
        )
    else:
        conclusion = (
            "The later positive-side samples remain positive but do not satisfy the full fixed broad-scale coherence guards. The next justified "
            "artifact-only stage is a positive-lobe variability audit; no measured amplitude is used as a solver parameter."
        )

    summary = {
        "stage": STAGE,
        "finite": bool(finite),
        "decision": decision,
        "configuration": cfg,
        "parents": {
            "stage142_run_id": EXPECTED_STAGE142_RUN_ID,
            "stage142_job_id": EXPECTED_STAGE142_JOB_ID,
            "stage142_artifact_id": EXPECTED_STAGE142_ARTIFACT_ID,
            "stage142_source_head": EXPECTED_STAGE142_SOURCE_HEAD,
        },
        "aggregate": {
            "parent_record_ok": bool(parent_record_ok),
            "parent_route_ok": bool(parent_route_ok),
            "sufficient_support": bool(sufficient_support),
            "later_positive_sign_coherence_pass": bool(sign_coherence >= MIN_LATER_POSITIVE_SIGN_COHERENCE),
            "coefficient_of_variation_pass": bool(np.isfinite(coefficient_of_variation) and coefficient_of_variation <= COHERENT_CV_MAX),
            "effective_support_pass": bool(np.isfinite(effective_count) and effective_count >= COHERENT_EFFECTIVE_COUNT_MIN),
            "peak_to_median_pass": bool(np.isfinite(peak_to_median) and peak_to_median <= COHERENT_PEAK_TO_MEDIAN_MAX),
        },
        "metrics": {
            "later_positive_count": int(later_values.size),
            "later_positive_span_cells": span_cells,
            "later_positive_sign_coherence": sign_coherence,
            "later_positive_mean_abs": mean_abs,
            "later_positive_median_abs": median_abs,
            "later_positive_std_abs": std_abs,
            "later_positive_coefficient_of_variation": coefficient_of_variation,
            "later_positive_relative_mad": relative_mad,
            "later_positive_effective_support_count": effective_count,
            "later_positive_peak_to_median_ratio": peak_to_median,
            "later_positive_minimum_to_median_ratio": minimum_to_median,
            "weak_endpoint_to_later_median_ratio": weak_endpoint_to_median,
            "later_positive_fraction_of_positive_side_l1": later_l1_fraction,
        },
        "scientific_conclusion": conclusion,
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. Stage 143 is an artifact-only positive-lobe scale audit; "
            "amplitude scale, dispersion, and effective-support measures are diagnostics, not solver parameters. No physical, collision/source, floor, "
            "wall, reconstruction, transport, limiter, normalization, source-relaxation, or velocity-quadrature parameter is retuned; no solver "
            "endpoint or cross-Knudsen extension is advanced."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        output_dir / "positive_lobe_scale.npz",
        positive_side_depth=positive_depth,
        positive_side_values=positive_values,
        later_positive_depth=later_depth,
        later_positive_values=later_values,
    )
    print(json.dumps({"stage": STAGE, "decision": decision, "aggregate": summary["aggregate"], "metrics": summary["metrics"]}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 143 fixed positive-lobe scale audit")
    parser.add_argument("--stage142-dir", type=Path, required=True)
    parser.add_argument("--stage142-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_stage143(args.stage142_dir, args.stage142_record, args.output_dir)


if __name__ == "__main__":
    main()
