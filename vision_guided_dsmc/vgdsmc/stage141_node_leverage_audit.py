from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = 141
EXPECTED_STAGE140_SOURCE_HEAD = "4a3e39c30c4bcbe5f01b21d85b8986e2c5de971c"
EXPECTED_STAGE140_RUN_ID = 32223772145
EXPECTED_STAGE140_JOB_ID = 95979233769
EXPECTED_STAGE140_ARTIFACT_ID = 9361267867
EXPECTED_STAGE140_ARTIFACT_SHA256 = "8d0c67f8d60618614c0eabd6b2b614b1f97abe07c2dc657a791e53531bba4205"
EXPECTED_STAGE140_SUMMARY_SHA256 = "30575bcbf62eafb395190dd59a34bd49f6c7f308d762dbf219989705681b4c0a"
EXPECTED_STAGE140_PAYLOAD_SHA256 = "fc5a34f9fd3d95930bddf05de865a419feb0f6d6b7598ededa02f67699bdf58d"
EXPECTED_STAGE140_DECISION = "stage140_interpolation_robust_but_node_proximate_stage141_node_leverage_audit"

PARENT_IDENTITY_CLOSURE_MAX = 1.0e-12
SMALL_ENDPOINT_ABS_RATIO_MAX = 0.25
MATERIAL_RAW_SENSITIVITY_RATIO_MIN = 2.0
ROBUST_DELETE_ROOT_SHIFT_MAX_CELLS = 0.25

NONFINITE = "stage141_nonfinite_blocker"
PARENT_RECORD_BLOCKER = "stage141_parent_record_blocker"
PARENT_ROUTE_BLOCKER = "stage141_parent_route_blocker"
INSUFFICIENT_SUPPORT = "stage141_insufficient_sample_support_blocker"
MATERIAL_LEVERAGE = "stage141_material_node_leverage_stage142_sampled_sign_margin_audit"
LOW_LEVERAGE = "stage141_low_node_leverage_stage142_signed_lobe_balance_audit"
MIXED_LEVERAGE = "stage141_mixed_node_leverage_stage142_sampled_transition_support_audit"


def validate_stage141_design(
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
    small_endpoint_abs_ratio_max=SMALL_ENDPOINT_ABS_RATIO_MAX,
    material_raw_sensitivity_ratio_min=MATERIAL_RAW_SENSITIVITY_RATIO_MIN,
    robust_delete_root_shift_max_cells=ROBUST_DELETE_ROOT_SHIFT_MAX_CELLS,
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
        "small_endpoint_abs_ratio_max": SMALL_ENDPOINT_ABS_RATIO_MAX,
        "material_raw_sensitivity_ratio_min": MATERIAL_RAW_SENSITIVITY_RATIO_MIN,
        "robust_delete_root_shift_max_cells": ROBUST_DELETE_ROOT_SHIFT_MAX_CELLS,
        "solver_rerun": False,
        "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False,
        "physical_parameter_retuning": False,
    }
    got = locals()
    for key, value in expected.items():
        if got[key] != value:
            raise ValueError(f"Stage 141 frozen-design violation: {key}={got[key]!r}, expected {value!r}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_stage140_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 140
        and record.get("source_head") == EXPECTED_STAGE140_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE140_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE140_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE140_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE140_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE140_SUMMARY_SHA256
        and record.get("local_interpolation_robustness_sha256") == EXPECTED_STAGE140_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE140_DECISION
    )


def lagrange3_value(x_eval: float, x_nodes: np.ndarray, y_nodes: np.ndarray) -> float:
    x = np.asarray(x_nodes, dtype=float)
    y = np.asarray(y_nodes, dtype=float)
    if x.shape != (3,) or y.shape != (3,) or not np.isfinite(np.concatenate([x, y])).all():
        raise ValueError("Stage 141 three-point interpolation requires finite length-three arrays")
    if np.unique(x).size != 3:
        raise ValueError("Stage 141 interpolation nodes must be distinct")
    value = 0.0
    for i in range(3):
        basis = 1.0
        for j in range(3):
            if i != j:
                basis *= (x_eval - x[j]) / (x[i] - x[j])
        value += y[i] * basis
    return float(value)


def quadratic_root_if_bracketed(
    depth: np.ndarray,
    values: np.ndarray,
    stencil_indices: tuple[int, int, int],
    lower_index: int,
    iterations: int = 80,
) -> float | None:
    x = np.asarray(depth, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 5:
        raise ValueError("Stage 141 root arrays require at least five equal one-dimensional samples")
    if not np.isfinite(np.concatenate([x, y])).all() or np.any(np.diff(x) <= 0.0):
        raise ValueError("Stage 141 root arrays must be finite and strictly ordered")
    if lower_index < 1 or lower_index + 3 >= x.size:
        raise ValueError("Stage 141 crossing bracket lacks fixed deletion support")
    idx = np.asarray(stencil_indices, dtype=int)
    if idx.shape != (3,) or np.any(idx < 0) or np.any(idx >= x.size) or np.unique(idx).size != 3:
        raise ValueError("Stage 141 quadratic stencil must contain three valid unique indices")
    xs = x[idx]
    ys = y[idx]
    lo = float(x[lower_index])
    hi = float(x[lower_index + 1])
    flo = lagrange3_value(lo, xs, ys)
    fhi = lagrange3_value(hi, xs, ys)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        return None
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        fmid = lagrange3_value(mid, xs, ys)
        if fmid == 0.0:
            return mid
        if flo * fmid < 0.0:
            hi = mid
        else:
            lo = mid
            flo = fmid
    return float(0.5 * (lo + hi))


def linear_root_and_sensitivities(depth: np.ndarray, values: np.ndarray, lower_index: int) -> tuple[float, float, float]:
    x = np.asarray(depth, dtype=float)
    y = np.asarray(values, dtype=float)
    x0, x1 = float(x[lower_index]), float(x[lower_index + 1])
    y0, y1 = float(y[lower_index]), float(y[lower_index + 1])
    if not (np.isfinite([x0, x1, y0, y1]).all() and x1 > x0 and y0 * y1 < 0.0):
        raise ValueError("Stage 141 linear sensitivity requires a finite signed one-cell bracket")
    dx = x1 - x0
    den = y1 - y0
    root = x0 - y0 * dx / den
    lower_abs_sensitivity = abs(-dx * y1 / (den * den))
    upper_abs_sensitivity = abs(dx * y0 / (den * den))
    return float(root), float(lower_abs_sensitivity), float(upper_abs_sensitivity)


def classify_node_leverage(
    *,
    upper_endpoint_abs_ratio_to_lower: float,
    upper_to_lower_raw_sensitivity_ratio: float,
    upper_deletion_root_retention_fraction: float,
    maximum_retained_upper_deletion_root_shift_cells: float,
    parent_record_ok: bool = True,
    parent_route_ok: bool = True,
    sufficient_support: bool = True,
    finite: bool = True,
) -> str:
    numeric = [
        upper_endpoint_abs_ratio_to_lower,
        upper_to_lower_raw_sensitivity_ratio,
        upper_deletion_root_retention_fraction,
        maximum_retained_upper_deletion_root_shift_cells,
    ]
    if not finite or not np.isfinite(numeric).all():
        return NONFINITE
    if not parent_record_ok:
        return PARENT_RECORD_BLOCKER
    if not parent_route_ok:
        return PARENT_ROUTE_BLOCKER
    if not sufficient_support:
        return INSUFFICIENT_SUPPORT
    small_endpoint = upper_endpoint_abs_ratio_to_lower <= SMALL_ENDPOINT_ABS_RATIO_MAX
    high_raw_sensitivity = upper_to_lower_raw_sensitivity_ratio >= MATERIAL_RAW_SENSITIVITY_RATIO_MIN
    deletion_fragile = (
        upper_deletion_root_retention_fraction < 1.0
        or maximum_retained_upper_deletion_root_shift_cells > ROBUST_DELETE_ROOT_SHIFT_MAX_CELLS
    )
    if small_endpoint and high_raw_sensitivity and deletion_fragile:
        return MATERIAL_LEVERAGE
    if upper_deletion_root_retention_fraction == 1.0 and maximum_retained_upper_deletion_root_shift_cells <= ROBUST_DELETE_ROOT_SHIFT_MAX_CELLS:
        return LOW_LEVERAGE
    return MIXED_LEVERAGE


def run_stage141(stage140_dir: Path, stage140_record: Path, output_dir: Path) -> dict:
    validate_stage141_design()
    parent_summary = _load_json(stage140_dir / "summary.json")
    parent_record = _load_json(stage140_record)
    parent_record_ok = _check_stage140_record(parent_record)
    parent_route_ok = bool(parent_summary.get("stage") == 140 and parent_summary.get("decision") == EXPECTED_STAGE140_DECISION)

    with np.load(stage140_dir / "local_interpolation_robustness.npz") as data:
        depth = np.asarray(data["right_depth"], dtype=float)
        complement = np.asarray(data["complement_signed"], dtype=float)
        bracket = np.asarray(data["crossing_bracket_indices"], dtype=int)
        candidate_roots = np.asarray(data["candidate_crossing_depths"], dtype=float)

    finite = bool(parent_summary.get("finite", False) and np.isfinite(np.concatenate([depth, complement, candidate_roots])).all())
    sufficient_support = bool(depth.shape == (7,) and complement.shape == (7,) and bracket.shape == (2,))
    if not sufficient_support:
        lower_index = 1
    else:
        if bracket[1] != bracket[0] + 1:
            raise ValueError("Stage 141 requires the exact one-cell Stage 140 crossing bracket")
        lower_index = int(bracket[0])
        if lower_index < 1 or lower_index + 3 >= depth.size:
            sufficient_support = False

    root, lower_sensitivity, upper_sensitivity = linear_root_and_sensitivities(depth, complement, lower_index)
    y_lower = float(complement[lower_index])
    y_upper = float(complement[lower_index + 1])
    upper_abs_ratio = float(abs(y_upper) / abs(y_lower))
    swing_fraction = float(abs(y_upper) / (abs(y_lower) + abs(y_upper)))
    sensitivity_ratio = float(upper_sensitivity / lower_sensitivity)

    # Fixed node-deletion bridges. Neither of the first two stencils uses the
    # small positive upper bracket node. They deliberately use opposite local
    # support sides to diagnose whether retaining a bracketed root depends on
    # that single sample rather than on a fitted parameter.
    upper_delete_left_stencil = (lower_index - 1, lower_index, lower_index + 2)
    upper_delete_right_stencil = (lower_index, lower_index + 2, lower_index + 3)
    lower_delete_stencil = (lower_index - 1, lower_index + 1, lower_index + 2)
    upper_left_root = quadratic_root_if_bracketed(depth, complement, upper_delete_left_stencil, lower_index)
    upper_right_root = quadratic_root_if_bracketed(depth, complement, upper_delete_right_stencil, lower_index)
    lower_deleted_root = quadratic_root_if_bracketed(depth, complement, lower_delete_stencil, lower_index)

    upper_deleted_roots = [r for r in (upper_left_root, upper_right_root) if r is not None]
    upper_retention_fraction = float(len(upper_deleted_roots) / 2.0)
    upper_shifts = [abs(r - root) for r in upper_deleted_roots]
    max_upper_shift = float(max(upper_shifts)) if upper_shifts else float(depth[lower_index + 1] - depth[lower_index])
    upper_deletion_disagreement = bool((upper_left_root is None) != (upper_right_root is None))
    lower_deleted_shift = float(abs(lower_deleted_root - root)) if lower_deleted_root is not None else float(depth[lower_index + 1] - depth[lower_index])

    decision = classify_node_leverage(
        upper_endpoint_abs_ratio_to_lower=upper_abs_ratio,
        upper_to_lower_raw_sensitivity_ratio=sensitivity_ratio,
        upper_deletion_root_retention_fraction=upper_retention_fraction,
        maximum_retained_upper_deletion_root_shift_cells=max_upper_shift,
        parent_record_ok=parent_record_ok,
        parent_route_ok=parent_route_ok,
        sufficient_support=sufficient_support,
        finite=finite,
    )

    cfg = dict(parent_summary["configuration"])
    cfg.update({
        "small_endpoint_abs_ratio_max": SMALL_ENDPOINT_ABS_RATIO_MAX,
        "material_raw_sensitivity_ratio_min": MATERIAL_RAW_SENSITIVITY_RATIO_MIN,
        "robust_delete_root_shift_max_cells": ROBUST_DELETE_ROOT_SHIFT_MAX_CELLS,
        "node_deletion_stencils": {
            "upper_left_bridge": list(upper_delete_left_stencil),
            "upper_right_bridge": list(upper_delete_right_stencil),
            "lower_bridge": list(lower_delete_stencil),
        },
        "counterfactual_root_used_for_solver": False,
        "solver_rerun": False,
        "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False,
        "benchmark_or_validation_claim_permitted": False,
    })
    for key in list(cfg):
        if key.endswith("_retuning"):
            cfg[key] = False

    if decision == MATERIAL_LEVERAGE:
        conclusion = (
            "The Stage-140 crossing is stable across fixed interpolants that retain both bracket endpoints, but the small positive "
            "upper bracket node has material raw secant leverage and the two fixed deletion bridges do not both preserve a root in "
            "the parent bracket. The precise sub-cell crossing is therefore node-leveraged rather than independently supported by "
            "the neighboring samples. This does not erase the sampled sign change itself. The next justified artifact-only stage is "
            "a sampled sign-margin audit of the positive node and its positive-side neighbors; no counterfactual root is fed back into the solver."
        )
    elif decision == LOW_LEVERAGE:
        conclusion = (
            "Both fixed deletion bridges preserve the parent-bracket crossing within the quarter-cell shift guard, so the inferred "
            "transition is not materially controlled by the small positive endpoint at this sampled-support level. The next justified "
            "artifact-only stage is a signed-lobe balance audit; no root estimate is used as a solver parameter."
        )
    else:
        conclusion = (
            "The fixed node-deletion diagnostics give mixed support for the precise sub-cell crossing. The next justified artifact-only "
            "step is a sampled transition-support audit, with no solver rerun or parameter retuning."
        )

    metrics = {
        "parent_linear_crossing_depth_cells": root,
        "upper_endpoint_abs_ratio_to_lower": upper_abs_ratio,
        "upper_endpoint_fraction_of_bracket_swing": swing_fraction,
        "lower_endpoint_abs_root_sensitivity_cells_per_value": lower_sensitivity,
        "upper_endpoint_abs_root_sensitivity_cells_per_value": upper_sensitivity,
        "upper_to_lower_raw_sensitivity_ratio": sensitivity_ratio,
        "upper_delete_left_root_in_parent_bracket": upper_left_root is not None,
        "upper_delete_right_root_in_parent_bracket": upper_right_root is not None,
        "upper_delete_left_root_depth_cells": upper_left_root,
        "upper_delete_right_root_depth_cells": upper_right_root,
        "upper_deletion_root_retention_fraction": upper_retention_fraction,
        "upper_deletion_disagreement": upper_deletion_disagreement,
        "maximum_retained_upper_deletion_root_shift_cells": max_upper_shift,
        "lower_deleted_root_in_parent_bracket": lower_deleted_root is not None,
        "lower_deleted_root_depth_cells": lower_deleted_root,
        "lower_deleted_root_shift_cells": lower_deleted_shift,
        "parent_candidate_root_span_cells": float(np.max(candidate_roots) - np.min(candidate_roots)),
    }

    summary = {
        "stage": STAGE,
        "finite": finite,
        "configuration": cfg,
        "parents": {
            "stage140_run_id": EXPECTED_STAGE140_RUN_ID,
            "stage140_job_id": EXPECTED_STAGE140_JOB_ID,
            "stage140_artifact_id": EXPECTED_STAGE140_ARTIFACT_ID,
            "stage140_source_head": EXPECTED_STAGE140_SOURCE_HEAD,
        },
        "metrics": metrics,
        "aggregate": {
            "parent_record_ok": parent_record_ok,
            "parent_route_ok": parent_route_ok,
            "sufficient_support": sufficient_support,
            "upper_endpoint_is_small": upper_abs_ratio <= SMALL_ENDPOINT_ABS_RATIO_MAX,
            "upper_endpoint_raw_sensitivity_is_material": sensitivity_ratio >= MATERIAL_RAW_SENSITIVITY_RATIO_MIN,
            "upper_deletion_root_retention_fraction": upper_retention_fraction,
            "upper_deletion_disagreement": upper_deletion_disagreement,
        },
        "decision": decision,
        "scientific_conclusion": conclusion,
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. Stage 141 is an artifact-only sampled-node "
            "leverage audit; deletion bridges and root sensitivities are diagnostics, not solver parameters. No physical, collision/source, "
            "floor, wall, reconstruction, transport, limiter, normalization, source-relaxation, or velocity-quadrature parameter is retuned; "
            "no solver endpoint or cross-Knudsen extension is advanced."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    np.savez(
        output_dir / "node_leverage.npz",
        right_depth=depth,
        complement_signed=complement,
        crossing_bracket_indices=bracket,
        parent_candidate_roots=candidate_roots,
        deletion_stencil_indices=np.asarray(
            [upper_delete_left_stencil, upper_delete_right_stencil, lower_delete_stencil], dtype=int
        ),
        deletion_root_exists=np.asarray(
            [upper_left_root is not None, upper_right_root is not None, lower_deleted_root is not None], dtype=bool
        ),
        deletion_root_depths=np.asarray(
            [
                np.nan if upper_left_root is None else upper_left_root,
                np.nan if upper_right_root is None else upper_right_root,
                np.nan if lower_deleted_root is None else lower_deleted_root,
            ],
            dtype=float,
        ),
        endpoint_abs_root_sensitivities=np.asarray([lower_sensitivity, upper_sensitivity], dtype=float),
    )
    print(json.dumps({"stage": STAGE, "decision": decision, "aggregate": summary["aggregate"], "metrics": metrics}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 141 fixed sampled-node leverage audit")
    parser.add_argument("--stage140-dir", type=Path, required=True)
    parser.add_argument("--stage140-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_stage141(args.stage140_dir, args.stage140_record, args.output_dir)


if __name__ == "__main__":
    main()
