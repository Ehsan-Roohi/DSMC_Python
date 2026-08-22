from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = 144
EXPECTED_STAGE143_SOURCE_HEAD = "fb248ec860d418e7255ffe5125b5b63c79a43d96"
EXPECTED_STAGE143_RUN_ID = 32300755939
EXPECTED_STAGE143_JOB_ID = 96222497169
EXPECTED_STAGE143_ARTIFACT_ID = 9387415154
EXPECTED_STAGE143_ARTIFACT_SHA256 = "27ac802af5d18b35deb6c71c1e992a1a606572a5ab849d1dc82c759222027651"
EXPECTED_STAGE143_SUMMARY_SHA256 = "a2ad7ae79afec2135adebe4182dc9c9ec6c05b69d0951c9f34db0e6ee8f91722"
EXPECTED_STAGE143_PAYLOAD_SHA256 = "0d71eee395b623cb7f9470f06863ae71cd5c4aa64f64bde979b43b8360d7e44a"
EXPECTED_STAGE143_DECISION = "stage143_broad_coherent_positive_lobe_scale_stage144_positive_lobe_shape_audit"

DOMINANT_DEVIATION_ENERGY_SHARE_MIN = 0.75
LEAVE_ONE_OUT_CV_REDUCTION_MIN = 0.50
MATERIAL_RELATIVE_DEVIATION_MIN = 0.25
PARENT_METRIC_CLOSURE_MAX = 1.0e-12

NONFINITE = "stage144_nonfinite_blocker"
PARENT_RECORD_BLOCKER = "stage144_parent_record_blocker"
PARENT_ROUTE_BLOCKER = "stage144_parent_route_blocker"
PARENT_METRIC_BLOCKER = "stage144_parent_metric_closure_blocker"
INSUFFICIENT_SUPPORT = "stage144_insufficient_positive_lobe_support_blocker"
SINGLE_TROUGH = "stage144_single_sample_dominated_positive_lobe_trough_stage145_trough_persistence_audit"
SINGLE_PEAK = "stage144_single_sample_dominated_positive_lobe_peak_stage145_peak_persistence_audit"
DISTRIBUTED = "stage144_distributed_positive_lobe_shape_stage145_distributed_shape_audit"


def validate_stage144_design(
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
    dominant_deviation_energy_share_min=DOMINANT_DEVIATION_ENERGY_SHARE_MIN,
    leave_one_out_cv_reduction_min=LEAVE_ONE_OUT_CV_REDUCTION_MIN,
    material_relative_deviation_min=MATERIAL_RELATIVE_DEVIATION_MIN,
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
        "dominant_deviation_energy_share_min": DOMINANT_DEVIATION_ENERGY_SHARE_MIN,
        "leave_one_out_cv_reduction_min": LEAVE_ONE_OUT_CV_REDUCTION_MIN,
        "material_relative_deviation_min": MATERIAL_RELATIVE_DEVIATION_MIN,
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
            raise ValueError(f"Stage 144 frozen-design violation: {key}={got[key]!r}, expected {value!r}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_stage143_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 143
        and record.get("source_head") == EXPECTED_STAGE143_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE143_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE143_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE143_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE143_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE143_SUMMARY_SHA256
        and record.get("positive_lobe_scale_sha256") == EXPECTED_STAGE143_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE143_DECISION
        and record.get("workflow_status") == "completed"
        and record.get("workflow_conclusion") == "success"
    )


def coefficient_of_variation(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or y.size == 0 or not np.isfinite(y).all():
        return float("nan")
    mean = float(np.mean(np.abs(y)))
    return float(np.std(np.abs(y)) / mean) if mean > 0.0 else float("nan")


def positive_lobe_shape_metrics(values: np.ndarray) -> tuple[dict[str, float | int], np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or y.size < 4:
        raise ValueError("Stage 144 shape audit requires at least four one-dimensional samples")
    if not np.isfinite(y).all() or np.any(y <= 0.0):
        raise ValueError("Stage 144 shape audit requires finite strictly positive samples")

    scale = float(np.median(y))
    if scale <= 0.0:
        raise ValueError("Stage 144 median scale must be positive")
    normalized = y / scale
    deviation = normalized - 1.0
    energy = deviation * deviation
    total_energy = float(np.sum(energy))
    if total_energy > 0.0:
        fractions = energy / total_energy
        dominant_index = int(np.argmax(fractions))
        dominant_share = float(fractions[dominant_index])
    else:
        fractions = np.zeros_like(energy)
        dominant_index = 0
        dominant_share = 0.0

    full_cv = coefficient_of_variation(y)
    loo_cv = np.empty(y.size, dtype=float)
    for i in range(y.size):
        loo_cv[i] = coefficient_of_variation(np.delete(y, i))
    dominant_loo_cv = float(loo_cv[dominant_index])
    cv_reduction = float(1.0 - dominant_loo_cv / full_cv) if full_cv > 0.0 else 0.0

    metrics = {
        "sample_count": int(y.size),
        "median_scale": scale,
        "full_coefficient_of_variation": full_cv,
        "dominant_sample_index": dominant_index,
        "dominant_normalized_value": float(normalized[dominant_index]),
        "dominant_relative_deviation": float(deviation[dominant_index]),
        "dominant_absolute_relative_deviation": float(abs(deviation[dominant_index])),
        "dominant_deviation_energy_share": dominant_share,
        "leave_dominant_out_coefficient_of_variation": dominant_loo_cv,
        "leave_dominant_out_cv_reduction_fraction": cv_reduction,
        "minimum_leave_one_out_cv": float(np.min(loo_cv)),
        "maximum_leave_one_out_cv": float(np.max(loo_cv)),
    }
    return metrics, normalized, fractions, loo_cv


def classify_positive_lobe_shape(
    *,
    dominant_energy_share: float,
    leave_one_out_cv_reduction: float,
    dominant_relative_deviation: float,
    sample_count: int,
    parent_record_ok: bool = True,
    parent_route_ok: bool = True,
    parent_metric_closure: float = 0.0,
    finite: bool = True,
) -> str:
    numeric = [dominant_energy_share, leave_one_out_cv_reduction, dominant_relative_deviation, parent_metric_closure]
    if not finite or not np.isfinite(numeric).all():
        return NONFINITE
    if not parent_record_ok:
        return PARENT_RECORD_BLOCKER
    if not parent_route_ok:
        return PARENT_ROUTE_BLOCKER
    if parent_metric_closure > PARENT_METRIC_CLOSURE_MAX:
        return PARENT_METRIC_BLOCKER
    if sample_count < 4:
        return INSUFFICIENT_SUPPORT

    single_sample_material = bool(
        dominant_energy_share >= DOMINANT_DEVIATION_ENERGY_SHARE_MIN
        and leave_one_out_cv_reduction >= LEAVE_ONE_OUT_CV_REDUCTION_MIN
        and abs(dominant_relative_deviation) >= MATERIAL_RELATIVE_DEVIATION_MIN
    )
    if single_sample_material:
        return SINGLE_TROUGH if dominant_relative_deviation < 0.0 else SINGLE_PEAK
    return DISTRIBUTED


def run_stage144(stage143_dir: Path, stage143_record: Path, output_dir: Path) -> dict:
    validate_stage144_design()
    parent_summary = _load_json(stage143_dir / "summary.json")
    parent_record = _load_json(stage143_record)
    parent_record_ok = _check_stage143_record(parent_record)
    parent_route_ok = bool(
        parent_summary.get("stage") == 143
        and parent_summary.get("decision") == EXPECTED_STAGE143_DECISION
    )

    with np.load(stage143_dir / "positive_lobe_scale.npz") as data:
        depth = np.asarray(data["later_positive_depth"], dtype=float)
        values = np.asarray(data["later_positive_values"], dtype=float)

    sufficient_support = bool(
        depth.ndim == 1
        and values.ndim == 1
        and depth.shape == values.shape
        and values.size >= 4
        and np.all(np.diff(depth) > 0.0)
    )
    finite = bool(
        parent_summary.get("finite", False)
        and sufficient_support
        and np.isfinite(depth).all()
        and np.isfinite(values).all()
        and np.all(values > 0.0)
    )

    if sufficient_support and finite:
        metrics, normalized, energy_fraction, loo_cv = positive_lobe_shape_metrics(values)
        parent_metrics = parent_summary.get("metrics", {})
        parent_metric_closure = max(
            abs(float(metrics["full_coefficient_of_variation"]) - float(parent_metrics["later_positive_coefficient_of_variation"])),
            abs(float(metrics["median_scale"]) - float(parent_metrics["later_positive_median_abs"])),
        )
    else:
        metrics = {
            "sample_count": int(values.size),
            "median_scale": float("nan"),
            "full_coefficient_of_variation": float("nan"),
            "dominant_sample_index": -1,
            "dominant_normalized_value": float("nan"),
            "dominant_relative_deviation": float("nan"),
            "dominant_absolute_relative_deviation": float("nan"),
            "dominant_deviation_energy_share": float("nan"),
            "leave_dominant_out_coefficient_of_variation": float("nan"),
            "leave_dominant_out_cv_reduction_fraction": float("nan"),
            "minimum_leave_one_out_cv": float("nan"),
            "maximum_leave_one_out_cv": float("nan"),
        }
        normalized = np.full_like(values, np.nan)
        energy_fraction = np.full_like(values, np.nan)
        loo_cv = np.full_like(values, np.nan)
        parent_metric_closure = float("inf")

    decision = classify_positive_lobe_shape(
        dominant_energy_share=float(metrics["dominant_deviation_energy_share"]),
        leave_one_out_cv_reduction=float(metrics["leave_dominant_out_cv_reduction_fraction"]),
        dominant_relative_deviation=float(metrics["dominant_relative_deviation"]),
        sample_count=int(metrics["sample_count"]),
        parent_record_ok=parent_record_ok,
        parent_route_ok=parent_route_ok,
        parent_metric_closure=parent_metric_closure,
        finite=finite,
    )

    cfg = dict(parent_summary.get("configuration", {}))
    cfg.update({
        "dominant_deviation_energy_share_min": DOMINANT_DEVIATION_ENERGY_SHARE_MIN,
        "leave_one_out_cv_reduction_min": LEAVE_ONE_OUT_CV_REDUCTION_MIN,
        "material_relative_deviation_min": MATERIAL_RELATIVE_DEVIATION_MIN,
        "parent_metric_closure_max": PARENT_METRIC_CLOSURE_MAX,
        "positive_lobe_shape_used_for_solver": False,
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

    if decision == SINGLE_TROUGH:
        conclusion = (
            "The broad Stage-143 positive-lobe support is not shape-uniform: one later positive sample carries a supermajority of the median-normalized deviation energy, lies materially below the median scale, and removing it sharply reduces the remaining coefficient of variation. "
            "The next justified artifact-only diagnostic is a trough-persistence audit using fixed neighboring support; the identified trough amplitude or location is not fed back into the solver."
        )
    elif decision == SINGLE_PEAK:
        conclusion = (
            "The broad Stage-143 positive-lobe support contains one materially high sample that dominates the median-normalized shape variation and whose removal sharply reduces the remaining coefficient of variation. "
            "The next justified artifact-only diagnostic is a peak-persistence audit; the identified peak amplitude or location is not fed back into the solver."
        )
    elif decision == DISTRIBUTED:
        conclusion = (
            "No single later positive sample satisfies all fixed materiality, deviation-energy, and leave-one-out reduction guards. The remaining positive-lobe shape variation is therefore treated as distributed at this sampled resolution, and the next justified artifact-only diagnostic is a distributed-shape audit."
        )
    else:
        conclusion = "Stage 144 is blocked by a finite-data, parent-record, parent-route, support, or parent-metric closure guard; no scientific routing is claimed."

    summary = {
        "stage": STAGE,
        "finite": bool(finite),
        "decision": decision,
        "configuration": cfg,
        "parents": {
            "stage143_run_id": EXPECTED_STAGE143_RUN_ID,
            "stage143_job_id": EXPECTED_STAGE143_JOB_ID,
            "stage143_artifact_id": EXPECTED_STAGE143_ARTIFACT_ID,
            "stage143_source_head": EXPECTED_STAGE143_SOURCE_HEAD,
        },
        "aggregate": {
            "parent_record_ok": bool(parent_record_ok),
            "parent_route_ok": bool(parent_route_ok),
            "sufficient_support": bool(sufficient_support),
            "maximum_parent_metric_closure": float(parent_metric_closure),
            "dominant_deviation_energy_share_pass": bool(
                np.isfinite(float(metrics["dominant_deviation_energy_share"]))
                and float(metrics["dominant_deviation_energy_share"]) >= DOMINANT_DEVIATION_ENERGY_SHARE_MIN
            ),
            "leave_one_out_cv_reduction_pass": bool(
                np.isfinite(float(metrics["leave_dominant_out_cv_reduction_fraction"]))
                and float(metrics["leave_dominant_out_cv_reduction_fraction"]) >= LEAVE_ONE_OUT_CV_REDUCTION_MIN
            ),
            "material_relative_deviation_pass": bool(
                np.isfinite(float(metrics["dominant_relative_deviation"]))
                and abs(float(metrics["dominant_relative_deviation"])) >= MATERIAL_RELATIVE_DEVIATION_MIN
            ),
        },
        "metrics": metrics,
        "scientific_conclusion": conclusion,
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. Stage 144 is an artifact-only positive-lobe shape audit; "
            "median-normalized deviations, leave-one-out coefficients of variation, and sample influence measures are diagnostics, not solver parameters. "
            "No physical, collision/source, floor, wall, reconstruction, transport, limiter, normalization, source-relaxation, or velocity-quadrature parameter is retuned; "
            "no solver endpoint or cross-Knudsen extension is advanced, and no benchmark or validation claim is permitted."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        output_dir / "positive_lobe_shape.npz",
        later_positive_depth=depth,
        later_positive_values=values,
        median_normalized_profile=normalized,
        deviation_energy_fraction=energy_fraction,
        leave_one_out_cv=loo_cv,
    )
    print(json.dumps({"stage": STAGE, "decision": decision, "aggregate": summary["aggregate"], "metrics": summary["metrics"]}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 144 fixed positive-lobe shape audit")
    parser.add_argument("--stage143-dir", type=Path, required=True)
    parser.add_argument("--stage143-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_stage144(args.stage143_dir, args.stage143_record, args.output_dir)


if __name__ == "__main__":
    main()
