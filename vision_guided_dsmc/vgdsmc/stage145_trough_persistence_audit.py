from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = 145
EXPECTED_STAGE144_SOURCE_HEAD = "cb9d05f06893502ad1c9c91d385923a5e63252bd"
EXPECTED_STAGE144_RUN_ID = 32315737077
EXPECTED_STAGE144_JOB_ID = 96267408392
EXPECTED_STAGE144_ARTIFACT_ID = 9393643659
EXPECTED_STAGE144_ARTIFACT_SHA256 = "59e99107e4566eb995710341e07cbf2725763cbd4f6be7a2f36326bee0568970"
EXPECTED_STAGE144_SUMMARY_SHA256 = "8f63bc6fd4f72696118f634957c2d933d434b85c426a84eb754000d8ec9f30d8"
EXPECTED_STAGE144_PAYLOAD_SHA256 = "794534c55513cbab6af3864a2c652526af67d823e992065fc08b00ce7f694f9c"
EXPECTED_STAGE144_DECISION = "stage144_single_sample_dominated_positive_lobe_trough_stage145_trough_persistence_audit"

MATERIAL_DEPRESSION_MIN = 0.25
PERSISTENT_SUPPORT_MIN_SAMPLES = 2
PARENT_METRIC_CLOSURE_MAX = 1.0e-12

NONFINITE = "stage145_nonfinite_blocker"
PARENT_RECORD_BLOCKER = "stage145_parent_record_blocker"
PARENT_ROUTE_BLOCKER = "stage145_parent_route_blocker"
PARENT_METRIC_BLOCKER = "stage145_parent_metric_closure_blocker"
NEIGHBOR_SUPPORT_BLOCKER = "stage145_neighbor_support_blocker"
PARENT_TROUGH_BLOCKER = "stage145_parent_trough_materiality_blocker"
ISOLATED = "stage145_trough_isolated_to_single_sample_stage146_trough_provenance_audit"
PERSISTENT = "stage145_trough_persists_over_neighbor_support_stage146_trough_width_audit"


def validate_stage145_design(
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
    material_depression_min=MATERIAL_DEPRESSION_MIN,
    persistent_support_min_samples=PERSISTENT_SUPPORT_MIN_SAMPLES,
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
        "material_depression_min": MATERIAL_DEPRESSION_MIN,
        "persistent_support_min_samples": PERSISTENT_SUPPORT_MIN_SAMPLES,
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
            raise ValueError(f"Stage 145 frozen-design violation: {key}={got[key]!r}, expected {value!r}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_stage144_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 144
        and record.get("source_head") == EXPECTED_STAGE144_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE144_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE144_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE144_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE144_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE144_SUMMARY_SHA256
        and record.get("positive_lobe_shape_sha256") == EXPECTED_STAGE144_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE144_DECISION
        and record.get("workflow_status") == "completed"
        and record.get("workflow_conclusion") == "success"
    )


def _shape_metrics(values: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(values, dtype=float)
    scale = float(np.median(y))
    normalized = y / scale
    deviation = normalized - 1.0
    energy = deviation * deviation
    fractions = energy / float(np.sum(energy))
    dominant_index = int(np.argmax(fractions))
    full_cv = float(np.std(np.abs(y)) / np.mean(np.abs(y)))
    loo = np.delete(y, dominant_index)
    loo_cv = float(np.std(np.abs(loo)) / np.mean(np.abs(loo)))
    return {
        "dominant_sample_index": dominant_index,
        "dominant_relative_deviation": float(deviation[dominant_index]),
        "dominant_deviation_energy_share": float(fractions[dominant_index]),
        "full_coefficient_of_variation": full_cv,
        "leave_dominant_out_coefficient_of_variation": loo_cv,
    }


def trough_persistence_metrics(depth: np.ndarray, values: np.ndarray, trough_index: int) -> tuple[dict, np.ndarray]:
    x = np.asarray(depth, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 4:
        raise ValueError("Stage 145 requires at least four one-dimensional depth/value samples")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(y <= 0.0):
        raise ValueError("Stage 145 requires finite strictly positive values")
    if not np.all(np.diff(x) > 0.0):
        raise ValueError("Stage 145 depth samples must be strictly increasing")
    if trough_index <= 0 or trough_index >= y.size - 1:
        raise ValueError("Stage 145 trough must have immediate left and right neighbors")

    nontrough = np.delete(y, trough_index)
    baseline = float(np.median(nontrough))
    depression = np.maximum(0.0, 1.0 - y / baseline)
    material = depression >= MATERIAL_DEPRESSION_MIN

    lo = trough_index
    hi = trough_index
    while lo > 0 and material[lo - 1]:
        lo -= 1
    while hi + 1 < y.size and material[hi + 1]:
        hi += 1
    support_count = int(hi - lo + 1) if material[trough_index] else 0
    support_span = float(x[hi] - x[lo]) if support_count > 0 else 0.0

    xl, xc, xr = x[trough_index - 1], x[trough_index], x[trough_index + 1]
    yl, yc, yr = y[trough_index - 1], y[trough_index], y[trough_index + 1]
    secant = float(yl + (yr - yl) * (xc - xl) / (xr - xl))
    secant_ratio = float(yc / secant)
    secant_deficit = float(1.0 - secant_ratio)
    neighbor_pair_relative_difference = float(abs(yl - yr) / np.median([yl, yr]))

    metrics = {
        "sample_count": int(y.size),
        "trough_index": int(trough_index),
        "trough_depth": float(x[trough_index]),
        "trough_value": float(y[trough_index]),
        "nontrough_median": baseline,
        "trough_relative_depression_to_nontrough_median": float(depression[trough_index]),
        "left_neighbor_relative_depression": float(depression[trough_index - 1]),
        "right_neighbor_relative_depression": float(depression[trough_index + 1]),
        "neighbor_material_depression_count": int(material[trough_index - 1]) + int(material[trough_index + 1]),
        "contiguous_material_support_count": support_count,
        "contiguous_material_support_span_cells": support_span,
        "local_neighbor_secant_value": secant,
        "trough_to_local_neighbor_secant_ratio": secant_ratio,
        "trough_relative_deficit_to_local_neighbor_secant": secant_deficit,
        "neighbor_pair_relative_difference": neighbor_pair_relative_difference,
    }
    return metrics, material.astype(np.int8)


def classify_trough_persistence(
    *,
    trough_relative_depression: float,
    contiguous_support_count: int,
    parent_record_ok: bool = True,
    parent_route_ok: bool = True,
    parent_metric_closure: float = 0.0,
    has_neighbor_support: bool = True,
    finite: bool = True,
) -> str:
    numeric = [trough_relative_depression, parent_metric_closure]
    if not finite or not np.isfinite(numeric).all():
        return NONFINITE
    if not parent_record_ok:
        return PARENT_RECORD_BLOCKER
    if not parent_route_ok:
        return PARENT_ROUTE_BLOCKER
    if parent_metric_closure > PARENT_METRIC_CLOSURE_MAX:
        return PARENT_METRIC_BLOCKER
    if not has_neighbor_support:
        return NEIGHBOR_SUPPORT_BLOCKER
    if trough_relative_depression < MATERIAL_DEPRESSION_MIN:
        return PARENT_TROUGH_BLOCKER
    if contiguous_support_count >= PERSISTENT_SUPPORT_MIN_SAMPLES:
        return PERSISTENT
    return ISOLATED


def run_stage145(stage144_dir: Path, stage144_record: Path, output_dir: Path) -> dict:
    validate_stage145_design()
    parent_summary = _load_json(stage144_dir / "summary.json")
    parent_record = _load_json(stage144_record)
    parent_record_ok = _check_stage144_record(parent_record)
    parent_route_ok = bool(
        parent_summary.get("stage") == 144
        and parent_summary.get("decision") == EXPECTED_STAGE144_DECISION
    )

    with np.load(stage144_dir / "positive_lobe_shape.npz") as data:
        depth = np.asarray(data["later_positive_depth"], dtype=float)
        values = np.asarray(data["later_positive_values"], dtype=float)

    finite = bool(
        parent_summary.get("finite", False)
        and depth.ndim == 1
        and values.ndim == 1
        and depth.shape == values.shape
        and depth.size >= 4
        and np.isfinite(depth).all()
        and np.isfinite(values).all()
        and np.all(values > 0.0)
        and np.all(np.diff(depth) > 0.0)
    )
    parent_metrics = parent_summary.get("metrics", {})
    trough_index = int(parent_metrics.get("dominant_sample_index", -1))
    has_neighbor_support = bool(finite and 0 < trough_index < values.size - 1)

    if finite and has_neighbor_support:
        metrics, material_mask = trough_persistence_metrics(depth, values, trough_index)
        reproduced = _shape_metrics(values)
        parent_metric_closure = max(
            abs(float(reproduced["dominant_sample_index"]) - float(parent_metrics["dominant_sample_index"])),
            abs(float(reproduced["dominant_relative_deviation"]) - float(parent_metrics["dominant_relative_deviation"])),
            abs(float(reproduced["dominant_deviation_energy_share"]) - float(parent_metrics["dominant_deviation_energy_share"])),
            abs(float(reproduced["full_coefficient_of_variation"]) - float(parent_metrics["full_coefficient_of_variation"])),
            abs(float(reproduced["leave_dominant_out_coefficient_of_variation"]) - float(parent_metrics["leave_dominant_out_coefficient_of_variation"])),
        )
    else:
        metrics = {
            "sample_count": int(values.size),
            "trough_index": trough_index,
            "trough_depth": float("nan"),
            "trough_value": float("nan"),
            "nontrough_median": float("nan"),
            "trough_relative_depression_to_nontrough_median": float("nan"),
            "left_neighbor_relative_depression": float("nan"),
            "right_neighbor_relative_depression": float("nan"),
            "neighbor_material_depression_count": -1,
            "contiguous_material_support_count": 0,
            "contiguous_material_support_span_cells": float("nan"),
            "local_neighbor_secant_value": float("nan"),
            "trough_to_local_neighbor_secant_ratio": float("nan"),
            "trough_relative_deficit_to_local_neighbor_secant": float("nan"),
            "neighbor_pair_relative_difference": float("nan"),
        }
        material_mask = np.zeros(values.shape, dtype=np.int8)
        parent_metric_closure = float("inf")

    decision = classify_trough_persistence(
        trough_relative_depression=float(metrics["trough_relative_depression_to_nontrough_median"]),
        contiguous_support_count=int(metrics["contiguous_material_support_count"]),
        parent_record_ok=parent_record_ok,
        parent_route_ok=parent_route_ok,
        parent_metric_closure=parent_metric_closure,
        has_neighbor_support=has_neighbor_support,
        finite=finite,
    )

    cfg = dict(parent_summary.get("configuration", {}))
    cfg.update({
        "material_depression_min": MATERIAL_DEPRESSION_MIN,
        "persistent_support_min_samples": PERSISTENT_SUPPORT_MIN_SAMPLES,
        "parent_metric_closure_max": PARENT_METRIC_CLOSURE_MAX,
        "trough_persistence_used_for_solver": False,
        "trough_amplitude_refit_applied": False,
        "trough_location_shift_applied": False,
        "solver_rerun": False,
        "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False,
        "benchmark_or_validation_claim_permitted": False,
    })
    for key in list(cfg):
        if key.endswith("_retuning"):
            cfg[key] = False

    if decision == ISOLATED:
        conclusion = (
            "The Stage-144 trough is materially depressed at its sampled depth but does not persist to either immediate neighbor under the fixed 25% depression guard. "
            "Its contiguous material support is one sample, so it is treated as an isolated sampled trough rather than a resolved broad sub-lobe. The next justified artifact-only diagnostic is a fixed trough-provenance audit of that inherited sample location; no amplitude or location is fed back into the solver."
        )
    elif decision == PERSISTENT:
        conclusion = (
            "The Stage-144 trough remains materially depressed across at least two contiguous fixed samples, so the feature has resolved neighboring support at the current sampled depth resolution. "
            "The next justified artifact-only diagnostic is a fixed trough-width audit; no width, amplitude, or location is fed back into the solver."
        )
    else:
        conclusion = "Stage 145 is blocked by a finite-data, parent-record, parent-route, parent-metric, neighbor-support, or parent-trough materiality guard; no scientific routing is claimed."

    summary = {
        "stage": STAGE,
        "finite": bool(finite),
        "decision": decision,
        "configuration": cfg,
        "parents": {
            "stage144_run_id": EXPECTED_STAGE144_RUN_ID,
            "stage144_job_id": EXPECTED_STAGE144_JOB_ID,
            "stage144_artifact_id": EXPECTED_STAGE144_ARTIFACT_ID,
            "stage144_source_head": EXPECTED_STAGE144_SOURCE_HEAD,
        },
        "aggregate": {
            "parent_record_ok": bool(parent_record_ok),
            "parent_route_ok": bool(parent_route_ok),
            "has_neighbor_support": bool(has_neighbor_support),
            "maximum_parent_metric_closure": float(parent_metric_closure),
            "trough_material_depression_pass": bool(
                np.isfinite(float(metrics["trough_relative_depression_to_nontrough_median"]))
                and float(metrics["trough_relative_depression_to_nontrough_median"]) >= MATERIAL_DEPRESSION_MIN
            ),
            "persistent_neighbor_support_pass": bool(
                int(metrics["contiguous_material_support_count"]) >= PERSISTENT_SUPPORT_MIN_SAMPLES
            ),
        },
        "metrics": metrics,
        "scientific_conclusion": conclusion,
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. Stage 145 is an artifact-only trough-persistence audit; local secant deficit, depression masks, and support counts are diagnostics, not solver parameters. "
            "No physical, collision/source, floor, wall, reconstruction, transport, limiter, normalization, source-relaxation, or velocity-quadrature parameter is retuned; no trough amplitude/location is fed back into the solver, no solver endpoint or cross-Knudsen extension is advanced, and no benchmark or validation claim is permitted."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        output_dir / "trough_persistence.npz",
        later_positive_depth=depth,
        later_positive_values=values,
        material_depression_mask=material_mask,
        trough_index=np.asarray([trough_index], dtype=np.int64),
        trough_depth=np.asarray([metrics["trough_depth"]], dtype=float),
        trough_value=np.asarray([metrics["trough_value"]], dtype=float),
        local_neighbor_secant_value=np.asarray([metrics["local_neighbor_secant_value"]], dtype=float),
    )
    print(json.dumps({"stage": STAGE, "decision": decision, "aggregate": summary["aggregate"], "metrics": metrics}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 145 fixed neighboring-support trough-persistence audit")
    parser.add_argument("--stage144-dir", type=Path, required=True)
    parser.add_argument("--stage144-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_stage145(args.stage144_dir, args.stage144_record, args.output_dir)


if __name__ == "__main__":
    main()
