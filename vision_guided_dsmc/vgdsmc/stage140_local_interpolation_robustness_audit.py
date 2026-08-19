from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = 140
EXPECTED_STAGE139_SOURCE_HEAD = "7878669fc252a5045e41ee756fb52dcb4017c5aa"
EXPECTED_STAGE139_RUN_ID = 32203217905
EXPECTED_STAGE139_JOB_ID = 95921160528
EXPECTED_STAGE139_ARTIFACT_ID = 9354291820
EXPECTED_STAGE139_ARTIFACT_SHA256 = "bebedba3e04bcaf8242a514a39c4feeaea8302dd248354c505059bb0fa204b05"
EXPECTED_STAGE139_SUMMARY_SHA256 = "289794bad96bdccc926758b6f1d7684d576da3b5f8a91788c25a390011f39c79"
EXPECTED_STAGE139_PAYLOAD_SHA256 = "0fb8071ba946a2cbea586f1fd2393e57ac8f4747e713bd5bdc82125d27a7703a"
EXPECTED_STAGE139_DECISION = "stage139_node_proximate_crossing_stage140_local_interpolation_robustness_audit"

PARENT_IDENTITY_CLOSURE_MAX = 1.0e-12
ROOT_SPREAD_MAX_CELLS = 0.25
NODE_PROXIMATE_EDGE_CLEARANCE_FRACTION = 0.25

NONFINITE = "stage140_nonfinite_blocker"
PARENT_RECORD_BLOCKER = "stage140_parent_record_blocker"
PARENT_ROUTE_BLOCKER = "stage140_parent_route_blocker"
INSUFFICIENT_STENCILS = "stage140_insufficient_local_stencils_blocker"
INTERPOLATION_SENSITIVE = "stage140_interpolation_sensitive_stage141_sampled_transition_support_audit"
ROBUST_NODE_PROXIMATE = "stage140_interpolation_robust_but_node_proximate_stage141_node_leverage_audit"
ROBUST_CENTERED = "stage140_interpolation_robust_and_centered_stage141_signed_lobe_balance_audit"


def validate_stage140_design(
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
    root_spread_max_cells=ROOT_SPREAD_MAX_CELLS,
    node_proximate_edge_clearance_fraction=NODE_PROXIMATE_EDGE_CLEARANCE_FRACTION,
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
        "root_spread_max_cells": ROOT_SPREAD_MAX_CELLS,
        "node_proximate_edge_clearance_fraction": NODE_PROXIMATE_EDGE_CLEARANCE_FRACTION,
        "solver_rerun": False,
        "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False,
        "physical_parameter_retuning": False,
    }
    got = locals()
    for key, value in expected.items():
        if got[key] != value:
            raise ValueError(f"Stage 140 frozen-design violation: {key}={got[key]!r}, expected {value!r}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_stage139_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 139
        and record.get("source_head") == EXPECTED_STAGE139_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE139_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE139_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE139_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE139_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE139_SUMMARY_SHA256
        and record.get("complement_transition_geometry_sha256") == EXPECTED_STAGE139_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE139_DECISION
    )


def lagrange3_value(x_eval: float, x_nodes: np.ndarray, y_nodes: np.ndarray) -> float:
    x = np.asarray(x_nodes, dtype=float)
    y = np.asarray(y_nodes, dtype=float)
    if x.shape != (3,) or y.shape != (3,) or not np.isfinite(np.concatenate([x, y])).all():
        raise ValueError("Stage 140 three-point interpolation requires finite length-three arrays")
    if np.unique(x).size != 3:
        raise ValueError("Stage 140 interpolation nodes must be distinct")
    value = 0.0
    for i in range(3):
        basis = 1.0
        for j in range(3):
            if i != j:
                basis *= (x_eval - x[j]) / (x[i] - x[j])
        value += y[i] * basis
    return float(value)


def quadratic_root_in_bracket(
    depth: np.ndarray,
    values: np.ndarray,
    stencil_indices: tuple[int, int, int],
    lower_index: int,
    iterations: int = 80,
) -> float:
    x = np.asarray(depth, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 3:
        raise ValueError("Stage 140 root arrays must be equal finite one-dimensional arrays")
    if not np.isfinite(np.concatenate([x, y])).all() or np.any(np.diff(x) <= 0.0):
        raise ValueError("Stage 140 root arrays must be finite and strictly ordered")
    if lower_index < 0 or lower_index >= x.size - 1:
        raise ValueError("Stage 140 crossing bracket is out of range")
    idx = np.asarray(stencil_indices, dtype=int)
    if idx.shape != (3,) or np.any(idx < 0) or np.any(idx >= x.size) or np.unique(idx).size != 3:
        raise ValueError("Stage 140 quadratic stencil must contain three valid unique indices")
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
    if flo * fhi >= 0.0:
        raise ValueError("Stage 140 quadratic root requires a signed parent bracket")
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        fmid = lagrange3_value(mid, xs, ys)
        if fmid == 0.0:
            return mid
        if flo * fmid < 0.0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
    return float(0.5 * (lo + hi))


def linear_root(depth: np.ndarray, values: np.ndarray, lower_index: int) -> float:
    x = np.asarray(depth, dtype=float)
    y = np.asarray(values, dtype=float)
    if lower_index < 0 or lower_index >= x.size - 1:
        raise ValueError("Stage 140 linear crossing bracket is out of range")
    x0, x1 = float(x[lower_index]), float(x[lower_index + 1])
    y0, y1 = float(y[lower_index]), float(y[lower_index + 1])
    if y0 == 0.0:
        return x0
    if y1 == 0.0:
        return x1
    if y0 * y1 >= 0.0:
        raise ValueError("Stage 140 linear interpolation requires a signed zero-crossing bracket")
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def local_root_candidates(depth: np.ndarray, values: np.ndarray, lower_index: int) -> tuple[list[str], np.ndarray]:
    x = np.asarray(depth, dtype=float)
    y = np.asarray(values, dtype=float)
    names = ["linear_secant"]
    roots = [linear_root(x, y, lower_index)]
    if lower_index - 1 >= 0:
        names.append("left_quadratic")
        roots.append(quadratic_root_in_bracket(x, y, (lower_index - 1, lower_index, lower_index + 1), lower_index))
    if lower_index + 2 < x.size:
        names.append("right_quadratic")
        roots.append(quadratic_root_in_bracket(x, y, (lower_index, lower_index + 1, lower_index + 2), lower_index))
    return names, np.asarray(roots, dtype=float)


def classify_interpolation_robustness(
    *,
    candidate_count: int,
    all_roots_in_parent_bracket: bool,
    root_span_cells: float,
    minimum_edge_clearance_fraction: float,
    parent_record_ok: bool = True,
    parent_route_ok: bool = True,
    finite: bool = True,
) -> str:
    numeric = [root_span_cells, minimum_edge_clearance_fraction]
    if not finite or not np.isfinite(numeric).all():
        return NONFINITE
    if not parent_record_ok:
        return PARENT_RECORD_BLOCKER
    if not parent_route_ok:
        return PARENT_ROUTE_BLOCKER
    if candidate_count < 3:
        return INSUFFICIENT_STENCILS
    if (not all_roots_in_parent_bracket) or root_span_cells > ROOT_SPREAD_MAX_CELLS:
        return INTERPOLATION_SENSITIVE
    if minimum_edge_clearance_fraction < NODE_PROXIMATE_EDGE_CLEARANCE_FRACTION:
        return ROBUST_NODE_PROXIMATE
    return ROBUST_CENTERED


def run_stage140(stage139_dir: Path, stage139_record: Path, output_dir: Path) -> dict:
    validate_stage140_design()
    parent_summary = _load_json(stage139_dir / "summary.json")
    parent_record = _load_json(stage139_record)
    parent_record_ok = _check_stage139_record(parent_record)
    parent_route_ok = bool(parent_summary.get("stage") == 139 and parent_summary.get("decision") == EXPECTED_STAGE139_DECISION)

    with np.load(stage139_dir / "complement_transition_geometry.npz") as data:
        depth = np.asarray(data["right_depth"], dtype=float)
        dominant = np.asarray(data["dominant_signed"], dtype=float)
        parent = np.asarray(data["parent_signed"], dtype=float)
        complement = np.asarray(data["complement_signed"], dtype=float)
        bracket = np.asarray(data["crossing_bracket_indices"], dtype=int)
        parent_crossing = float(np.asarray(data["crossing_depth"], dtype=float)[0])

    finite = bool(parent_summary.get("finite", False) and np.isfinite(np.concatenate([depth, dominant, parent, complement])).all())
    identity_closure = float(np.max(np.abs(parent - dominant - complement)))
    if bracket.shape != (2,) or bracket[1] != bracket[0] + 1:
        raise ValueError("Stage 140 requires the exact one-cell Stage 139 crossing bracket")
    lower_index = int(bracket[0])
    lower_depth = float(depth[lower_index])
    upper_depth = float(depth[lower_index + 1])
    bracket_width = upper_depth - lower_depth

    names, roots = local_root_candidates(depth, complement, lower_index)
    linear = float(roots[0])
    all_in_bracket = bool(np.all((roots >= lower_depth) & (roots <= upper_depth)))
    root_span = float(np.max(roots) - np.min(roots))
    max_shift = float(np.max(np.abs(roots - linear)))
    edge_clearance_cells = np.minimum(roots - lower_depth, upper_depth - roots)
    edge_clearance_fractions = edge_clearance_cells / bracket_width
    minimum_edge_clearance_fraction = float(np.min(edge_clearance_fractions))
    maximum_edge_clearance_fraction = float(np.max(edge_clearance_fractions))
    parent_crossing_reproduction_error = float(abs(parent_crossing - linear))

    decision = classify_interpolation_robustness(
        candidate_count=len(roots),
        all_roots_in_parent_bracket=all_in_bracket,
        root_span_cells=root_span,
        minimum_edge_clearance_fraction=minimum_edge_clearance_fraction,
        parent_record_ok=parent_record_ok,
        parent_route_ok=parent_route_ok and identity_closure <= PARENT_IDENTITY_CLOSURE_MAX,
        finite=finite,
    )

    cfg = dict(parent_summary["configuration"])
    cfg.update({
        "parent_identity_closure_max": PARENT_IDENTITY_CLOSURE_MAX,
        "root_spread_max_cells": ROOT_SPREAD_MAX_CELLS,
        "node_proximate_edge_clearance_fraction": NODE_PROXIMATE_EDGE_CLEARANCE_FRACTION,
        "local_interpolation_candidates": names,
        "interpolated_crossing_used_for_solver": False,
        "solver_rerun": False,
        "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False,
        "benchmark_or_validation_claim_permitted": False,
    })
    for key in list(cfg):
        if key.endswith("_retuning"):
            cfg[key] = False

    if decision == ROBUST_NODE_PROXIMATE:
        conclusion = (
            "The Stage-139 complement crossing remains inside the same one-cell bracket under the fixed linear, left-quadratic, and right-quadratic local interpolants, and their sub-cell roots span less than the inherited quarter-cell robustness guard. The precise crossing is therefore interpolation-robust at this local-stencil level but remains node-proximate. The next justified artifact-only stage is a node-leverage audit to quantify how strongly the small positive sampled node controls the sign-transition inference; no interpolated root is fed back into the solver."
        )
    elif decision == ROBUST_CENTERED:
        conclusion = (
            "The complementary-channel crossing is stable across the fixed local interpolants and remains well centered inside the parent bracket. The next justified artifact-only stage is a signed-lobe balance audit; no interpolated root is used as a solver parameter."
        )
    elif decision == INTERPOLATION_SENSITIVE:
        conclusion = (
            "The inferred sub-cell complement crossing is sensitive to the fixed neighboring interpolation stencils. The next justified artifact-only stage is therefore a sampled transition-support audit, without solver or physical retuning."
        )
    else:
        conclusion = (
            "Stage 140 cannot establish local interpolation robustness under the frozen provenance and support guards. No solver endpoint is advanced and no physical or numerical parameter is changed."
        )

    summary = {
        "stage": STAGE,
        "finite": finite,
        "configuration": cfg,
        "parents": {
            "stage139_run_id": EXPECTED_STAGE139_RUN_ID,
            "stage139_job_id": EXPECTED_STAGE139_JOB_ID,
            "stage139_artifact_id": EXPECTED_STAGE139_ARTIFACT_ID,
            "stage139_source_head": EXPECTED_STAGE139_SOURCE_HEAD,
        },
        "metrics": {
            "parent_identity_closure": identity_closure,
            "candidate_count": len(roots),
            "all_roots_in_parent_bracket": all_in_bracket,
            "parent_bracket_lower_depth_cells": lower_depth,
            "parent_bracket_upper_depth_cells": upper_depth,
            "parent_bracket_width_cells": bracket_width,
            "parent_linear_crossing_depth_cells": parent_crossing,
            "linear_crossing_reproduction_error_cells": parent_crossing_reproduction_error,
            "linear_crossing_depth_cells": linear,
            "left_quadratic_crossing_depth_cells": float(roots[names.index("left_quadratic")]) if "left_quadratic" in names else float("nan"),
            "right_quadratic_crossing_depth_cells": float(roots[names.index("right_quadratic")]) if "right_quadratic" in names else float("nan"),
            "root_span_cells": root_span,
            "maximum_shift_from_linear_cells": max_shift,
            "minimum_edge_clearance_fraction": minimum_edge_clearance_fraction,
            "maximum_edge_clearance_fraction": maximum_edge_clearance_fraction,
        },
        "aggregate": {
            "candidate_count": len(roots),
            "all_roots_in_parent_bracket": all_in_bracket,
            "root_span_cells": root_span,
            "maximum_shift_from_linear_cells": max_shift,
            "minimum_edge_clearance_fraction": minimum_edge_clearance_fraction,
            "parent_identity_closure": identity_closure,
            "parent_record_ok": parent_record_ok,
            "parent_route_ok": parent_route_ok,
        },
        "decision": decision,
        "scientific_conclusion": conclusion,
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. Stage 140 is an artifact-only local interpolation robustness audit; candidate sub-cell roots are not solver parameters. No physical, collision/source, floor, wall, reconstruction, transport, limiter, normalization, source-relaxation, or velocity-quadrature parameter is retuned; no solver endpoint or cross-Knudsen extension is advanced."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        output_dir / "local_interpolation_robustness.npz",
        right_depth=depth,
        complement_signed=complement,
        crossing_bracket_indices=bracket,
        candidate_names=np.asarray(names, dtype="U32"),
        candidate_crossing_depths=roots,
        candidate_edge_clearance_fractions=edge_clearance_fractions,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 140 local interpolation robustness audit")
    parser.add_argument("--stage139-dir", type=Path, required=True)
    parser.add_argument("--stage139-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_stage140(args.stage139_dir, args.stage139_record, args.output_dir)
    print(json.dumps({"stage": STAGE, "decision": summary["decision"], "aggregate": summary["aggregate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
