from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = 139
EXPECTED_STAGE138_SOURCE_HEAD = "e513f249dd9ceb43556ab07f9d0378a791eeaa8a"
EXPECTED_STAGE138_RUN_ID = 32176056342
EXPECTED_STAGE138_JOB_ID = 95838163254
EXPECTED_STAGE138_ARTIFACT_ID = 9347568376
EXPECTED_STAGE138_ARTIFACT_SHA256 = "b8d435a64aab18c208ab3531aa52a8bc1e615d1b82faa26b0743ee3dc24785b4"
EXPECTED_STAGE138_SUMMARY_SHA256 = "61a10326c2dce8e5a47dd57333ea9acd5776f0b531dcce48204129a330860c3f"
EXPECTED_STAGE138_PAYLOAD_SHA256 = "44ddc0cc8bd3c1bc18c4cb60b953c985bdcb93ebe0c3446b5216a8cf000e03c8"
EXPECTED_STAGE138_DECISION = "stage138_depth_varying_complement_cancellation_stage139_complement_transition_geometry_audit"

PARENT_IDENTITY_CLOSURE_MAX = 1.0e-12
TRANSITION_BRACKET_WIDTH_MAX_CELLS = 1.0 + 1.0e-12
SIDE_SIGN_COHERENCE_MIN = 1.0
WELL_CENTERED_EDGE_CLEARANCE_MIN_FRACTION = 0.25

NONFINITE = "stage139_nonfinite_blocker"
PARENT_RECORD_BLOCKER = "stage139_parent_record_blocker"
PARENT_IDENTITY_BLOCKER = "stage139_parent_identity_closure_blocker"
UNRESOLVED_GEOMETRY = "stage139_unresolved_transition_geometry_stage140_signed_complement_residual_audit"
UNDERRESOLVED_CROSSING = "stage139_underresolved_crossing_stage140_transition_resolution_audit"
NODE_PROXIMATE_CROSSING = "stage139_node_proximate_crossing_stage140_local_interpolation_robustness_audit"
WELL_LOCALIZED_CROSSING = "stage139_well_localized_crossing_stage140_signed_lobe_balance_audit"


def validate_stage139_design(
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
    parent_identity_closure_max=PARENT_IDENTITY_CLOSURE_MAX,
    transition_bracket_width_max_cells=TRANSITION_BRACKET_WIDTH_MAX_CELLS,
    side_sign_coherence_min=SIDE_SIGN_COHERENCE_MIN,
    well_centered_edge_clearance_min_fraction=WELL_CENTERED_EDGE_CLEARANCE_MIN_FRACTION,
    solver_rerun=False,
    solver_endpoint_advanced=False,
    cross_knudsen_extension_permitted=False,
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
        "parent_identity_closure_max": PARENT_IDENTITY_CLOSURE_MAX,
        "transition_bracket_width_max_cells": TRANSITION_BRACKET_WIDTH_MAX_CELLS,
        "side_sign_coherence_min": SIDE_SIGN_COHERENCE_MIN,
        "well_centered_edge_clearance_min_fraction": WELL_CENTERED_EDGE_CLEARANCE_MIN_FRACTION,
        "solver_rerun": False,
        "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False,
        "physical_parameter_retuning": False,
    }
    got = locals()
    for key, value in expected.items():
        if got[key] != value:
            raise ValueError(f"Stage 139 frozen-design violation: {key}={got[key]!r}, expected {value!r}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_stage138_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 138
        and record.get("source_head") == EXPECTED_STAGE138_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE138_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE138_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE138_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE138_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE138_SUMMARY_SHA256
        and record.get("channel_rate_origin_sha256") == EXPECTED_STAGE138_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE138_DECISION
    )


def sign_change_brackets(values: np.ndarray) -> list[int]:
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or y.size < 2 or not np.isfinite(y).all():
        raise ValueError("Stage 139 sign-transition input must be a finite one-dimensional array")
    brackets: list[int] = []
    for i in range(y.size - 1):
        if y[i] == 0.0 or y[i] * y[i + 1] < 0.0:
            brackets.append(i)
    if y[-1] == 0.0 and (not brackets or brackets[-1] != y.size - 2):
        brackets.append(y.size - 2)
    return brackets


def interpolated_crossing(depth: np.ndarray, values: np.ndarray, lower_index: int) -> float:
    x = np.asarray(depth, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("Stage 139 crossing arrays must have equal one-dimensional shape")
    if lower_index < 0 or lower_index >= x.size - 1:
        raise ValueError("Stage 139 crossing bracket index is out of range")
    x0, x1 = float(x[lower_index]), float(x[lower_index + 1])
    y0, y1 = float(y[lower_index]), float(y[lower_index + 1])
    if x1 <= x0 or not np.isfinite([x0, x1, y0, y1]).all():
        raise ValueError("Stage 139 crossing bracket must be finite and ordered")
    if y0 == 0.0:
        return x0
    if y1 == 0.0:
        return x1
    if y0 * y1 >= 0.0:
        raise ValueError("Stage 139 interpolation requires a signed zero-crossing bracket")
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def signed_piecewise_linear_abs_areas(depth: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    x = np.asarray(depth, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise ValueError("Stage 139 area audit requires equal one-dimensional arrays")
    if not np.isfinite(np.concatenate([x, y])).all() or np.any(np.diff(x) <= 0.0):
        raise ValueError("Stage 139 area inputs must be finite and strictly ordered")
    negative_area = 0.0
    positive_area = 0.0
    for i in range(x.size - 1):
        x0, x1 = float(x[i]), float(x[i + 1])
        y0, y1 = float(y[i]), float(y[i + 1])
        dx = x1 - x0
        if y0 == 0.0 and y1 == 0.0:
            continue
        if y0 * y1 >= 0.0:
            area = 0.5 * (abs(y0) + abs(y1)) * dx
            if y0 < 0.0 or (y0 == 0.0 and y1 < 0.0):
                negative_area += area
            else:
                positive_area += area
            continue
        xc = x0 - y0 * dx / (y1 - y0)
        left_area = 0.5 * abs(y0) * (xc - x0)
        right_area = 0.5 * abs(y1) * (x1 - xc)
        if y0 < 0.0:
            negative_area += left_area
            positive_area += right_area
        else:
            positive_area += left_area
            negative_area += right_area
    return float(negative_area), float(positive_area)


def classify_transition_geometry(
    *,
    sign_change_count: int,
    left_sign_coherence: float,
    right_sign_coherence: float,
    bracket_width_cells: float,
    edge_clearance_fraction: float,
    parent_identity_closure: float,
    finite: bool = True,
    parent_record_ok: bool = True,
) -> str:
    numeric = [left_sign_coherence, right_sign_coherence, bracket_width_cells, edge_clearance_fraction, parent_identity_closure]
    if not finite or not np.isfinite(numeric).all():
        return NONFINITE
    if not parent_record_ok:
        return PARENT_RECORD_BLOCKER
    if parent_identity_closure > PARENT_IDENTITY_CLOSURE_MAX:
        return PARENT_IDENTITY_BLOCKER
    if sign_change_count != 1 or min(left_sign_coherence, right_sign_coherence) < SIDE_SIGN_COHERENCE_MIN:
        return UNRESOLVED_GEOMETRY
    if bracket_width_cells > TRANSITION_BRACKET_WIDTH_MAX_CELLS:
        return UNDERRESOLVED_CROSSING
    if edge_clearance_fraction < WELL_CENTERED_EDGE_CLEARANCE_MIN_FRACTION:
        return NODE_PROXIMATE_CROSSING
    return WELL_LOCALIZED_CROSSING


def run_stage139(stage138_dir: Path, stage138_record: Path, output_dir: Path) -> dict:
    validate_stage139_design()
    stage138_summary = _load_json(stage138_dir / "summary.json")
    stage138_observed = _load_json(stage138_record)
    record_ok = _check_stage138_record(stage138_observed)
    if stage138_summary.get("stage") != 138 or stage138_summary.get("decision") != EXPECTED_STAGE138_DECISION:
        raise ValueError("Stage 139 requires the completed Stage 138 depth-varying cancellation route")

    with np.load(stage138_dir / "channel_rate_origin.npz") as data:
        depth = np.asarray(data["right_depth"], dtype=float)
        dominant = np.asarray(data["dominant_signed"], dtype=float)
        parent = np.asarray(data["parent_signed"], dtype=float)
        complement = np.asarray(data["complement_signed"], dtype=float)

    finite = bool(stage138_summary.get("finite", False) and np.isfinite(np.concatenate([depth, dominant, parent, complement])).all())
    parent_identity_closure = float(np.max(np.abs(parent - dominant - complement)))
    brackets = sign_change_brackets(complement)
    sign_change_count = len(brackets)

    crossing_depth = float("nan")
    lower_index = -1
    upper_index = -1
    lower_depth = float("nan")
    upper_depth = float("nan")
    bracket_width = float("nan")
    crossing_fraction_from_lower = float("nan")
    edge_clearance_cells = float("nan")
    edge_clearance_fraction = float("nan")
    left_sign_coherence = 0.0
    right_sign_coherence = 0.0
    local_secant_slope = float("nan")

    if sign_change_count == 1:
        lower_index = int(brackets[0])
        upper_index = lower_index + 1
        lower_depth = float(depth[lower_index])
        upper_depth = float(depth[upper_index])
        crossing_depth = interpolated_crossing(depth, complement, lower_index)
        bracket_width = upper_depth - lower_depth
        crossing_fraction_from_lower = (crossing_depth - lower_depth) / bracket_width
        edge_clearance_cells = min(crossing_depth - lower_depth, upper_depth - crossing_depth)
        edge_clearance_fraction = edge_clearance_cells / bracket_width
        left = complement[: lower_index + 1]
        right = complement[upper_index:]
        left_reference = np.sign(left[0])
        right_reference = np.sign(right[-1])
        left_sign_coherence = float(np.mean(np.sign(left) == left_reference))
        right_sign_coherence = float(np.mean(np.sign(right) == right_reference))
        local_secant_slope = float((complement[upper_index] - complement[lower_index]) / bracket_width)

    negative_area, positive_area = signed_piecewise_linear_abs_areas(depth, complement)
    total_abs_area = negative_area + positive_area
    positive_area_fraction = positive_area / total_abs_area if total_abs_area > 0.0 else float("nan")
    negative_area_fraction = negative_area / total_abs_area if total_abs_area > 0.0 else float("nan")

    left_support_length = crossing_depth - float(depth[0]) if np.isfinite(crossing_depth) else float("nan")
    right_support_length = float(depth[-1]) - crossing_depth if np.isfinite(crossing_depth) else float("nan")

    decision = classify_transition_geometry(
        sign_change_count=sign_change_count,
        left_sign_coherence=left_sign_coherence,
        right_sign_coherence=right_sign_coherence,
        bracket_width_cells=bracket_width,
        edge_clearance_fraction=edge_clearance_fraction,
        parent_identity_closure=parent_identity_closure,
        finite=finite,
        parent_record_ok=record_ok,
    )

    cfg = dict(stage138_summary["configuration"])
    cfg.update({
        "parent_identity_closure_max": PARENT_IDENTITY_CLOSURE_MAX,
        "transition_bracket_width_max_cells": TRANSITION_BRACKET_WIDTH_MAX_CELLS,
        "side_sign_coherence_min": SIDE_SIGN_COHERENCE_MIN,
        "well_centered_edge_clearance_min_fraction": WELL_CENTERED_EDGE_CLEARANCE_MIN_FRACTION,
        "interpolated_crossing_used_for_solver": False,
        "solver_rerun": False,
        "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False,
        "benchmark_or_validation_claim_permitted": False,
    })
    for key in list(cfg):
        if key.endswith("_retuning"):
            cfg[key] = False

    if decision == NODE_PROXIMATE_CROSSING:
        conclusion = (
            "The signed complementary channel has one coherent zero crossing inside a single one-cell depth bracket, but the linearly interpolated crossing lies within the preregistered quarter-bracket edge-clearance guard of a sampled depth. "
            "The transition is therefore localized but node-proximate at the current seven-point support. The next justified artifact-only stage is a local interpolation-robustness audit; the interpolated crossing is not a solver parameter and is not fed back into the method."
        )
    elif decision == WELL_LOCALIZED_CROSSING:
        conclusion = (
            "The signed complementary channel has one coherent, well-centered zero crossing inside a single fixed depth bracket. The next justified artifact-only stage is a signed-lobe balance audit. This remains a localization result, not a causal solver or validation result."
        )
    elif decision == UNDERRESOLVED_CROSSING:
        conclusion = (
            "The complementary-channel sign transition is coherent but spans more than the fixed one-cell localization guard. The next justified action is a transition-resolution audit using existing artifacts only; no solver parameter is changed."
        )
    else:
        conclusion = (
            "Stage 139 cannot localize a unique coherent complementary-channel transition under the fixed provenance, closure, and geometry guards. No solver or physical parameter is changed."
        )

    summary = {
        "stage": STAGE,
        "finite": finite,
        "configuration": cfg,
        "parents": {
            "stage138_run_id": EXPECTED_STAGE138_RUN_ID,
            "stage138_job_id": EXPECTED_STAGE138_JOB_ID,
            "stage138_artifact_id": EXPECTED_STAGE138_ARTIFACT_ID,
            "stage138_source_head": EXPECTED_STAGE138_SOURCE_HEAD,
        },
        "metrics": {
            "parent_identity_closure": parent_identity_closure,
            "complement_sign_change_count": sign_change_count,
            "crossing_lower_depth_cells": lower_depth,
            "crossing_upper_depth_cells": upper_depth,
            "crossing_depth_cells": crossing_depth,
            "crossing_bracket_width_cells": bracket_width,
            "crossing_fraction_from_lower": crossing_fraction_from_lower,
            "crossing_edge_clearance_cells": edge_clearance_cells,
            "crossing_edge_clearance_fraction": edge_clearance_fraction,
            "left_sign_coherence": left_sign_coherence,
            "right_sign_coherence": right_sign_coherence,
            "local_crossing_secant_slope_per_cell": local_secant_slope,
            "left_support_length_cells": left_support_length,
            "right_support_length_cells": right_support_length,
            "negative_lobe_abs_area": negative_area,
            "positive_lobe_abs_area": positive_area,
            "negative_lobe_abs_area_fraction": negative_area_fraction,
            "positive_lobe_abs_area_fraction": positive_area_fraction,
        },
        "aggregate": {
            "parent_record_ok": record_ok,
            "parent_identity_closure": parent_identity_closure,
            "complement_sign_change_count": sign_change_count,
            "minimum_side_sign_coherence": min(left_sign_coherence, right_sign_coherence),
            "crossing_bracket_width_cells": bracket_width,
            "crossing_edge_clearance_fraction": edge_clearance_fraction,
        },
        "decision": decision,
        "scientific_conclusion": conclusion,
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. Stage 139 is an artifact-only sign-transition geometry audit; the interpolated crossing is not a solver parameter. "
            "No physical, collision/source, floor, wall, reconstruction, transport, limiter, normalization, source-relaxation, or velocity-quadrature parameter is retuned; no solver endpoint or cross-Knudsen extension is advanced."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        output_dir / "complement_transition_geometry.npz",
        right_depth=depth,
        dominant_signed=dominant,
        parent_signed=parent,
        complement_signed=complement,
        crossing_bracket_indices=np.asarray([lower_index, upper_index], dtype=int),
        crossing_depth=np.asarray([crossing_depth], dtype=float),
        crossing_edge_clearance_fraction=np.asarray([edge_clearance_fraction], dtype=float),
        signed_lobe_abs_areas=np.asarray([negative_area, positive_area], dtype=float),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 139 fixed complement transition geometry audit")
    parser.add_argument("--stage138-dir", type=Path, required=True)
    parser.add_argument("--stage138-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_stage139(args.stage138_dir, args.stage138_record, args.output_dir)
    print(json.dumps({"stage": STAGE, "aggregate": summary["aggregate"], "decision": summary["decision"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
