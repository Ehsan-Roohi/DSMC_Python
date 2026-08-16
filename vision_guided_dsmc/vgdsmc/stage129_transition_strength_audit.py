from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STAGE128_RUN_ID = 31940630751
STAGE128_JOB_ID = 95149141476
STAGE128_ARTIFACT_ID = 9264993674
STAGE128_ARTIFACT_SHA256 = "52442695ec5c677b1eeb7cc4ed3ca8fb6f1036c5d84f2c45594f344c78e50867"
STAGE128_SUMMARY_SHA256 = "343b1257865c664dd630159ba6eef2a670de22d3b7e449cede30f43d859a102c"
STAGE128_PAYLOAD_SHA256 = "c6b1c2c5c786738a85ec849777d9dd84d96751eb568cec5d38a391fd85388657"
STAGE128_SOURCE_HEAD = "b2043a318ed677ce519535dbb4e8168295227579"
STAGE128_COMPLETION_COMMIT = "8cdf7a762e6ee32e66813fb277c50b8d44b9fc36"
STAGE128_DECISION = "stage128_fixed_node_sign_continuous_transition_reproduced_stage129_transition_strength_audit"

DEPTH_COUNT = 28
WITNESS_NODE = 9
MID_DEPTH_SLICE = (5, 14)
INNER_DEPTH_SLICE = (15, 28)
ASYMMETRY_MEDIAN_MIN = 0.50
TRANSITION_SUPPORT_RATIO_MIN = 0.50
MAX_TRANSITION_WIDTH_CELLS = 6.0
MAX_CROSS_WALL_WIDTH_DIFFERENCE_CELLS = 2.0
PARENT_PROFILE_CLOSURE_TOLERANCE = 1.0e-12

MATERIAL_TRANSITION = (
    "stage129_material_sign_continuous_transition_"
    "stage130_fixed_sector_continuity_audit"
)
WEAK_TRANSITION = (
    "stage129_sign_continuous_crossing_but_weak_transition_"
    "stage130_transition_strength_blocker_audit"
)
NONFINITE = "stage129_nonfinite_transition_strength_blocker_without_retuning"
PROVENANCE_BLOCKER = "stage129_stage128_provenance_blocker_without_retuning"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_stage129_design(**overrides: object) -> None:
    frozen = {
        "depth_count": DEPTH_COUNT,
        "witness_node": WITNESS_NODE,
        "mid_depth_slice": MID_DEPTH_SLICE,
        "inner_depth_slice": INNER_DEPTH_SLICE,
        "asymmetry_median_min": ASYMMETRY_MEDIAN_MIN,
        "transition_support_ratio_min": TRANSITION_SUPPORT_RATIO_MIN,
        "max_transition_width_cells": MAX_TRANSITION_WIDTH_CELLS,
        "max_cross_wall_width_difference_cells": MAX_CROSS_WALL_WIDTH_DIFFERENCE_CELLS,
        "parent_profile_closure_tolerance": PARENT_PROFILE_CLOSURE_TOLERANCE,
        "stage128_run_id": STAGE128_RUN_ID,
        "stage128_job_id": STAGE128_JOB_ID,
        "stage128_artifact_id": STAGE128_ARTIFACT_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 129 is fixed to the exact completed Stage-128 artifact, the preregistered "
            "sign-continuous radial-node-9 witness, inherited mid/inner depth bands, and fixed "
            "dimensionless strength guards. It may not retune physics, collision/source treatment, "
            "walls, reconstruction, transport, limiter, floors, normalization, source relaxation, "
            "velocity quadrature, failed MUSCL parameters, or diagnostic thresholds."
        )


def _verify_record(record: dict[str, object]) -> None:
    checks = (
        record.get("stage") == 128,
        record.get("source_head") == STAGE128_SOURCE_HEAD,
        record.get("workflow_status") == "completed",
        record.get("workflow_conclusion") == "success",
        record.get("workflow_run_id") == STAGE128_RUN_ID,
        record.get("workflow_job_id") == STAGE128_JOB_ID,
        record.get("artifact_id") == STAGE128_ARTIFACT_ID,
        record.get("artifact_sha256") == STAGE128_ARTIFACT_SHA256,
        record.get("summary_sha256") == STAGE128_SUMMARY_SHA256,
        record.get("radial_node_continuity_sha256") == STAGE128_PAYLOAD_SHA256,
        record.get("decision") == STAGE128_DECISION,
        record.get("tests", {}).get("passed") == 17,
        record.get("tests", {}).get("failed") == 0,
        record.get("aggregate", {}).get("sign_continuous_reproducing_fixed_nodes") == [WITNESS_NODE],
    )
    if not all(checks):
        raise ValueError("Committed Stage-128 provenance does not authorize Stage 129")


def _load_inputs(stage128_dir: str | Path, stage128_record_path: str | Path):
    root = Path(stage128_dir)
    summary_path = root / "summary.json"
    payload_path = root / "radial_node_continuity.npz"
    if not summary_path.is_file() or sha256_file(summary_path) != STAGE128_SUMMARY_SHA256:
        raise ValueError("Stage-128 summary checksum mismatch")
    if not payload_path.is_file() or sha256_file(payload_path) != STAGE128_PAYLOAD_SHA256:
        raise ValueError("Stage-128 payload checksum mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    record = json.loads(Path(stage128_record_path).read_text(encoding="utf-8"))
    _verify_record(record)
    if summary.get("stage") != 128 or summary.get("decision") != STAGE128_DECISION or summary.get("finite") is not True:
        raise ValueError("Stage-128 artifact does not authorize Stage 129")
    if summary.get("aggregate", {}).get("sign_continuous_reproducing_fixed_nodes") != [WITNESS_NODE]:
        raise ValueError("Stage-128 sign-continuous witness changed")
    needed = {
        "depth",
        "parent_wall0_asymmetry",
        "parent_wall1_asymmetry",
        f"node{WITNESS_NODE}_wall0_asymmetry",
        f"node{WITNESS_NODE}_wall0_same_sign_l1",
        f"node{WITNESS_NODE}_wall0_net_sign",
        f"node{WITNESS_NODE}_wall1_asymmetry",
        f"node{WITNESS_NODE}_wall1_same_sign_l1",
        f"node{WITNESS_NODE}_wall1_net_sign",
    }
    with np.load(payload_path) as data:
        if not needed.issubset(data.files):
            raise ValueError("Stage-128 payload is incomplete")
        arrays = {name: np.asarray(data[name]).copy() for name in needed}
    for name, array in arrays.items():
        if name == "depth":
            if array.shape != (DEPTH_COUNT,):
                raise ValueError("Stage-128 depth shape mismatch")
        elif array.shape != (DEPTH_COUNT,):
            raise ValueError(f"Stage-128 profile shape mismatch: {name}")
    return summary, arrays


def crossing_depth(asymmetry: np.ndarray) -> tuple[float, int]:
    a = np.asarray(asymmetry, dtype=np.float64)
    if a.shape != (DEPTH_COUNT,) or not np.isfinite(a).all():
        raise ValueError("Invalid Stage-129 asymmetry profile")
    hits: list[tuple[float, int]] = []
    for k in range(DEPTH_COUNT - 1):
        left, right = float(a[k]), float(a[k + 1])
        if left == 0.0:
            hits.append((float(k + 1), k))
        elif left * right < 0.0:
            x = float(k + 1) + abs(left) / (abs(left) + abs(right))
            hits.append((x, k))
    if float(a[-1]) == 0.0:
        hits.append((float(DEPTH_COUNT), DEPTH_COUNT - 1))
    if len(hits) != 1:
        raise ValueError("Stage 129 requires exactly one Stage-128 crossing per wall")
    return hits[0]


def _first_descending_crossing(depth: np.ndarray, normalized: np.ndarray, level: float) -> float:
    x = np.asarray(depth, dtype=np.float64)
    y = np.asarray(normalized, dtype=np.float64)
    for k in range(y.size - 1):
        if y[k] >= level >= y[k + 1] and y[k] != y[k + 1]:
            return float(x[k] + (y[k] - level) / (y[k] - y[k + 1]) * (x[k + 1] - x[k]))
    raise ValueError(f"Stage-129 normalized profile does not cross level {level}")


def transition_metrics(depth: np.ndarray, asymmetry: np.ndarray, support: np.ndarray) -> dict[str, float | bool]:
    x = np.asarray(depth, dtype=np.float64)
    a = np.asarray(asymmetry, dtype=np.float64)
    s = np.asarray(support, dtype=np.float64)
    if x.shape != (DEPTH_COUNT,) or a.shape != (DEPTH_COUNT,) or s.shape != (DEPTH_COUNT,):
        raise ValueError("Stage-129 transition profile shape mismatch")
    if not np.isfinite(x).all() or not np.isfinite(a).all() or not np.isfinite(s).all() or np.any(s <= 0.0):
        raise ValueError("Stage-129 transition profile is nonfinite or has nonpositive support")
    crossing, bracket_index = crossing_depth(a)
    m0, m1 = MID_DEPTH_SLICE
    i0, i1 = INNER_DEPTH_SLICE
    mid_mask = (x >= m0) & (x <= m1)
    inner_mask = (x >= i0) & (x <= i1)
    mid_median = float(np.median(a[mid_mask]))
    inner_median = float(np.median(a[inner_mask]))
    contrast = mid_median - inner_median
    if contrast <= 0.0:
        raise ValueError("Stage-129 witness does not preserve the expected high-to-low transition")
    normalized = (a - inner_median) / contrast
    x75 = _first_descending_crossing(x, normalized, 0.75)
    x25 = _first_descending_crossing(x, normalized, 0.25)
    width = x25 - x75
    support_at_crossing = float(np.sqrt(s[bracket_index] * s[min(bracket_index + 1, DEPTH_COUNT - 1)]))
    reference_support = float(np.sqrt(np.median(s[mid_mask]) * np.median(s[inner_mask])))
    support_ratio = support_at_crossing / reference_support
    material = bool(
        mid_median >= ASYMMETRY_MEDIAN_MIN
        and inner_median <= -ASYMMETRY_MEDIAN_MIN
        and support_ratio >= TRANSITION_SUPPORT_RATIO_MIN
        and 0.0 < width <= MAX_TRANSITION_WIDTH_CELLS
    )
    return {
        "crossing_depth_cells": crossing,
        "mid_band_asymmetry_median": mid_median,
        "inner_band_asymmetry_median": inner_median,
        "median_side_to_side_contrast": contrast,
        "transition_x75_cells": x75,
        "transition_x25_cells": x25,
        "transition_width_25_to_75_cells": width,
        "crossing_same_sign_support": support_at_crossing,
        "reference_same_sign_support": reference_support,
        "crossing_support_ratio": support_ratio,
        "material_strength_per_wall": material,
    }


def stage129_decision(*, finite: bool, provenance_ok: bool, wall_material: list[bool], width_difference: float) -> str:
    if not provenance_ok:
        return PROVENANCE_BLOCKER
    if not finite:
        return NONFINITE
    if all(wall_material) and width_difference <= MAX_CROSS_WALL_WIDTH_DIFFERENCE_CELLS:
        return MATERIAL_TRANSITION
    return WEAK_TRANSITION


def run(stage128_dir: str | Path, stage128_record_path: str | Path,
        output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage129_design(**design)
    summary128, arrays = _load_inputs(stage128_dir, stage128_record_path)
    depth = np.asarray(arrays["depth"], dtype=np.float64)
    wall_metrics = []
    for wall in range(2):
        wall_metrics.append(transition_metrics(
            depth,
            arrays[f"node{WITNESS_NODE}_wall{wall}_asymmetry"],
            arrays[f"node{WITNESS_NODE}_wall{wall}_same_sign_l1"],
        ))
    widths = [float(m["transition_width_25_to_75_cells"]) for m in wall_metrics]
    width_difference = abs(widths[0] - widths[1])
    parent_closure = float(summary128["aggregate"]["maximum_parent_profile_closure_rel_l2"])
    finite = bool(
        np.isfinite(depth).all()
        and all(np.isfinite(float(v)) for m in wall_metrics for v in m.values() if not isinstance(v, bool))
        and np.isfinite(width_difference)
        and np.isfinite(parent_closure)
    )
    provenance_ok = bool(parent_closure <= PARENT_PROFILE_CLOSURE_TOLERANCE)
    decision = stage129_decision(
        finite=finite,
        provenance_ok=provenance_ok,
        wall_material=[bool(m["material_strength_per_wall"]) for m in wall_metrics],
        width_difference=width_difference,
    )
    result = {
        "stage": 129,
        "finite": finite,
        "configuration": {
            "artifact_only": True,
            "witness_node": WITNESS_NODE,
            "mid_depth_slice_cells": list(MID_DEPTH_SLICE),
            "inner_depth_slice_cells": list(INNER_DEPTH_SLICE),
            "asymmetry_median_min": ASYMMETRY_MEDIAN_MIN,
            "transition_support_ratio_min": TRANSITION_SUPPORT_RATIO_MIN,
            "max_transition_width_cells": MAX_TRANSITION_WIDTH_CELLS,
            "max_cross_wall_width_difference_cells": MAX_CROSS_WALL_WIDTH_DIFFERENCE_CELLS,
            "solver_rerun": False,
            "solver_endpoint_advanced": False,
            "model_retuning": False,
            "wall_retuning": False,
            "source_retuning": False,
            "reconstruction_retuning": False,
            "transport_retuning": False,
            "limiter_retuning": False,
            "floor_retuning": False,
            "normalization_retuning": False,
            "source_relaxation_retuning": False,
            "velocity_grid_retuning": False,
            "cross_knudsen_extension_permitted": False,
            "benchmark_or_validation_claim_permitted": False,
        },
        "parent_stage128": {
            "run_id": STAGE128_RUN_ID,
            "job_id": STAGE128_JOB_ID,
            "artifact_id": STAGE128_ARTIFACT_ID,
            "source_head": STAGE128_SOURCE_HEAD,
            "completion_commit": STAGE128_COMPLETION_COMMIT,
            "decision": STAGE128_DECISION,
        },
        "metrics": {
            "axis1_low": wall_metrics[0],
            "axis1_high": wall_metrics[1],
        },
        "aggregate": {
            "maximum_parent_profile_closure_rel_l2": parent_closure,
            "minimum_mid_band_asymmetry_median": float(min(m["mid_band_asymmetry_median"] for m in wall_metrics)),
            "maximum_inner_band_asymmetry_median": float(max(m["inner_band_asymmetry_median"] for m in wall_metrics)),
            "minimum_crossing_support_ratio": float(min(m["crossing_support_ratio"] for m in wall_metrics)),
            "maximum_transition_width_25_to_75_cells": float(max(widths)),
            "cross_wall_transition_width_difference_cells": float(width_difference),
            "material_strength_wall_count": int(sum(bool(m["material_strength_per_wall"]) for m in wall_metrics)),
        },
        "decision": decision,
        "scientific_conclusion": (
            "The sign-continuous radial-node-9 witness is tested for material transition strength using "
            "only inherited wall-distance bands and fixed dimensionless guards. Passing both walls means "
            "the side switch is supported by substantial same-sign residual weight and a finite-width "
            "high-to-low asymmetry transition, rather than being only a near-zero sign crossing. This "
            "still does not establish limiter causality, MUSCL stability, endpoint convergence, q_av "
            "improvement, benchmark accuracy, or validation."
            if decision == MATERIAL_TRANSITION else
            "The sign-continuous radial-node-9 crossing does not satisfy all preregistered transition-strength "
            "guards on both walls. The crossing should therefore be treated as structurally reproducible but "
            "not yet materially strong; no causal or solver claim is justified."
        ),
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. No physical, "
            "collision/source, floor, wall, reconstruction, transport, limiter, normalization, "
            "source-relaxation, or velocity-quadrature parameter is retuned; no solver endpoint or "
            "cross-Knudsen extension is advanced."
        ),
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        out / "transition_strength.npz",
        depth=depth,
        wall0_asymmetry=np.asarray(arrays[f"node{WITNESS_NODE}_wall0_asymmetry"], dtype=np.float64),
        wall1_asymmetry=np.asarray(arrays[f"node{WITNESS_NODE}_wall1_asymmetry"], dtype=np.float64),
        wall0_same_sign_l1=np.asarray(arrays[f"node{WITNESS_NODE}_wall0_same_sign_l1"], dtype=np.float64),
        wall1_same_sign_l1=np.asarray(arrays[f"node{WITNESS_NODE}_wall1_same_sign_l1"], dtype=np.float64),
        wall0_net_sign=np.asarray(arrays[f"node{WITNESS_NODE}_wall0_net_sign"], dtype=np.int8),
        wall1_net_sign=np.asarray(arrays[f"node{WITNESS_NODE}_wall1_net_sign"], dtype=np.int8),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-129 sign-continuous transition-strength audit")
    parser.add_argument("--stage128-dir", required=True)
    parser.add_argument("--stage128-record", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.stage128_dir, args.stage128_record, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
