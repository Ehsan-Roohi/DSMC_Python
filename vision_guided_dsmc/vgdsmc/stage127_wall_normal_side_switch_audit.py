from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STAGE126_RUN_ID = 31924459693
STAGE126_JOB_ID = 95109713428
STAGE126_ARTIFACT_ID = 9260078422
STAGE126_ARTIFACT_SHA256 = "5bc2060341ffdafe8c8ee185a82b8c70a6cefec0158393cebe835281d01c952c"
STAGE126_SUMMARY_SHA256 = "9230ca76842f7e470c1fd97fdff5b94a78e9f8b9c1ebeac639603a6a6d56c9fc"
STAGE126_PAYLOAD_SHA256 = "935db7028d35dd980aab7e7cb31af60d8287b7bfe7fb518646764b8ff28d6c46"
STAGE126_SOURCE_HEAD = "3d224b044fc4f6258cb70cea0193850310732992"
STAGE126_COMPLETION_COMMIT = "b692372fbb074ad51caddaf495ac824e2ecb9f1e"
STAGE126_DECISION = (
    "stage126_bilateral_wall_consistent_with_depth_side_reversal_"
    "stage127_wall_normal_side_switch_audit"
)

GRID = (56, 56)
DEPTH_COUNT = 28
BANDS = ("near_1_4", "mid_5_14", "inner_15_28")
WALLS = ("axis1_low", "axis1_high")
TANGENTIAL_SIDES = ("axis0_low", "axis0_high")

# Fixed artifact-classification guards only; these do not alter or tune the solver.
PARENT_METRIC_CLOSURE_TOLERANCE = 1.0e-12
CROSS_WALL_PROFILE_COSINE_MIN = 0.95
DEPTH_SIDE_AGREEMENT_MIN = 0.90
MAX_CROSSING_SEPARATION_CELLS = 2.0
EXPECTED_CROSSINGS_PER_WALL = 1

BILATERAL_SWITCH = (
    "stage127_bilateral_single_transition_zone_"
    "stage128_radial_node_continuity_audit"
)
WALL_SPECIFIC_SWITCH = (
    "stage127_wall_specific_or_multiple_transition_"
    "stage128_wall_specific_depth_profile_audit"
)
NO_SWITCH = (
    "stage127_band_side_reversal_not_reproduced_as_depth_switch_"
    "stage128_band_aggregation_origin_audit"
)
NONFINITE = "stage127_nonfinite_depth_profile_blocker_without_retuning"
CLOSURE_BLOCKER = "stage127_parent_metric_closure_blocker_without_retuning"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_stage127_design(**overrides: object) -> None:
    frozen = {
        "grid": GRID,
        "depth_count": DEPTH_COUNT,
        "bands": BANDS,
        "walls": WALLS,
        "tangential_sides": TANGENTIAL_SIDES,
        "parent_metric_closure_tolerance": PARENT_METRIC_CLOSURE_TOLERANCE,
        "cross_wall_profile_cosine_min": CROSS_WALL_PROFILE_COSINE_MIN,
        "depth_side_agreement_min": DEPTH_SIDE_AGREEMENT_MIN,
        "max_crossing_separation_cells": MAX_CROSSING_SEPARATION_CELLS,
        "expected_crossings_per_wall": EXPECTED_CROSSINGS_PER_WALL,
        "stage126_run_id": STAGE126_RUN_ID,
        "stage126_job_id": STAGE126_JOB_ID,
        "stage126_artifact_id": STAGE126_ARTIFACT_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 127 is fixed to the exact completed Stage-126 artifact and preregistered "
            "depth-profile guards; it may not retune physics, collision/source treatment, wall "
            "treatment, reconstruction, transport, limiter, floors, normalization, source "
            "relaxation, velocity quadrature, failed MUSCL parameters, or diagnostic thresholds"
        )


def _load_stage126(root: str | Path, record_path: str | Path):
    root = Path(root)
    expected = {
        "summary.json": STAGE126_SUMMARY_SHA256,
        "wall_side_asymmetry.npz": STAGE126_PAYLOAD_SHA256,
    }
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Stage-126 checksum mismatch: {name}")

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    if summary.get("stage") != 126 or summary.get("decision") != STAGE126_DECISION or summary.get("finite") is not True:
        raise ValueError("Stage-126 artifact does not authorize Stage 127")
    checks = (
        record.get("stage") == 126,
        record.get("decision") == STAGE126_DECISION,
        record.get("source_head") == STAGE126_SOURCE_HEAD,
        record.get("workflow_status") == "completed",
        record.get("workflow_conclusion") == "success",
        record.get("workflow_run_id") == STAGE126_RUN_ID,
        record.get("workflow_job_id") == STAGE126_JOB_ID,
        record.get("artifact_id") == STAGE126_ARTIFACT_ID,
        record.get("artifact_sha256") == STAGE126_ARTIFACT_SHA256,
        record.get("summary_sha256") == STAGE126_SUMMARY_SHA256,
        record.get("wall_side_asymmetry_sha256") == STAGE126_PAYLOAD_SHA256,
        record.get("tests", {}).get("passed") == 17,
        record.get("tests", {}).get("failed") == 0,
    )
    if not all(checks):
        raise ValueError("Committed Stage-126 provenance does not authorize Stage 127")

    with np.load(root / "wall_side_asymmetry.npz") as data:
        needed = {
            "dominant_node_residual",
            "band_index",
            "pass_mask",
            "dominant_nodes",
            "net_signs",
            "wall_same_sign_l1_fractions",
            "wall_tangential_side_fractions",
            "band_tangential_side_codes",
        }
        if not needed.issubset(data.files):
            raise ValueError("Stage-126 payload is incomplete")
        arrays = {name: np.asarray(data[name]).copy() for name in needed}

    field = np.asarray(arrays["dominant_node_residual"], dtype=np.float64)
    band_index = np.asarray(arrays["band_index"], dtype=np.int8)
    passed = np.asarray(arrays["pass_mask"], dtype=bool)
    dominant_nodes = np.asarray(arrays["dominant_nodes"], dtype=np.int16)
    net_signs = np.asarray(arrays["net_signs"], dtype=np.int8)
    parent_wall_fraction = np.asarray(arrays["wall_same_sign_l1_fractions"], dtype=np.float64)
    parent_side_fraction = np.asarray(arrays["wall_tangential_side_fractions"], dtype=np.float64)
    parent_band_side = np.asarray(arrays["band_tangential_side_codes"], dtype=np.int8)

    if field.shape != GRID or band_index.shape != GRID or passed.shape != GRID:
        raise ValueError("Stage-126 spatial-map shape mismatch")
    if dominant_nodes.shape != (3,) or net_signs.shape != (3,) or parent_band_side.shape != (3,):
        raise ValueError("Stage-126 band metadata shape mismatch")
    if parent_wall_fraction.shape != (3, 2) or parent_side_fraction.shape != (3, 2, 2):
        raise ValueError("Stage-126 wall-side fraction shape mismatch")
    if not np.isfinite(field).all() or not np.isfinite(parent_wall_fraction).all() or not np.isfinite(parent_side_fraction).all():
        raise ValueError("Stage-126 depth-profile inputs are nonfinite")
    if not np.all(np.isin(net_signs, (-1, 1))) or not np.all(np.isin(parent_band_side, (0, 1))):
        raise ValueError("Stage-126 sign/side metadata is invalid")

    return summary, record, {
        "field": field,
        "band_index": band_index,
        "passed": passed,
        "dominant_nodes": dominant_nodes,
        "net_signs": net_signs,
        "parent_wall_fraction": parent_wall_fraction,
        "parent_side_fraction": parent_side_fraction,
        "parent_band_side": parent_band_side,
    }


def _column_for_depth(wall_index: int, depth: int) -> int:
    if wall_index not in (0, 1) or not 1 <= depth <= DEPTH_COUNT:
        raise ValueError("Invalid Stage-127 wall/depth index")
    return depth - 1 if wall_index == 0 else GRID[1] - depth


def wall_depth_profile(
    field: np.ndarray,
    band_index: np.ndarray,
    net_signs: np.ndarray,
    wall_index: int,
) -> dict[str, np.ndarray]:
    x = np.asarray(field, dtype=np.float64)
    bands = np.asarray(band_index, dtype=np.int8)
    signs = np.asarray(net_signs, dtype=np.int8)
    if x.shape != GRID or bands.shape != GRID or signs.shape != (3,) or wall_index not in (0, 1):
        raise ValueError("Invalid Stage-127 wall-depth payload")

    low_fraction = np.zeros(DEPTH_COUNT, dtype=np.float64)
    high_fraction = np.zeros(DEPTH_COUNT, dtype=np.float64)
    same_l1 = np.zeros(DEPTH_COUNT, dtype=np.float64)
    band_code = np.zeros(DEPTH_COUNT, dtype=np.int8)
    signed_residual = np.zeros(DEPTH_COUNT, dtype=np.float64)
    uncancelled = np.zeros(DEPTH_COUNT, dtype=np.float64)

    for d in range(1, DEPTH_COUNT + 1):
        j = _column_for_depth(wall_index, d)
        unique_band = np.unique(bands[:, j])
        if unique_band.size != 1 or int(unique_band[0]) not in (0, 1, 2):
            raise ValueError("Stage-127 wall-distance line crosses inconsistent parent bands")
        b = int(unique_band[0])
        s = int(signs[b])
        values = x[:, j]
        same = (values * s) > 0.0
        weights = np.abs(values)
        total_same = float(np.sum(weights[same]))
        total_l1 = float(np.sum(weights))
        if total_same <= 0.0 or total_l1 <= 0.0:
            raise ValueError("Stage-127 wall-distance line has empty residual support")
        lo = float(np.sum(weights[: GRID[0] // 2][same[: GRID[0] // 2]]) / total_same)
        hi = float(np.sum(weights[GRID[0] // 2 :][same[GRID[0] // 2 :]]) / total_same)
        low_fraction[d - 1] = lo
        high_fraction[d - 1] = hi
        same_l1[d - 1] = total_same
        band_code[d - 1] = b
        signed_residual[d - 1] = float(np.sum(values))
        uncancelled[d - 1] = abs(signed_residual[d - 1]) / total_l1

    asymmetry = high_fraction - low_fraction
    dominant_side = np.where(asymmetry >= 0.0, 1, 0).astype(np.int8)
    dominance = np.maximum(low_fraction, high_fraction)
    return {
        "depth": np.arange(1, DEPTH_COUNT + 1, dtype=np.int16),
        "band_code": band_code,
        "same_sign_l1": same_l1,
        "axis0_low_fraction": low_fraction,
        "axis0_high_fraction": high_fraction,
        "tangential_asymmetry": asymmetry,
        "dominant_side_code": dominant_side,
        "dominance_fraction": dominance,
        "signed_residual": signed_residual,
        "uncancelled_fraction": uncancelled,
    }


def crossing_depths(asymmetry: np.ndarray) -> np.ndarray:
    a = np.asarray(asymmetry, dtype=np.float64)
    if a.shape != (DEPTH_COUNT,) or not np.isfinite(a).all():
        raise ValueError("Invalid Stage-127 asymmetry profile")
    crossings: list[float] = []
    for k in range(DEPTH_COUNT - 1):
        left = float(a[k])
        right = float(a[k + 1])
        if left == 0.0:
            crossings.append(float(k + 1))
        elif left * right < 0.0:
            crossings.append(float(k + 1) + abs(left) / (abs(left) + abs(right)))
    if float(a[-1]) == 0.0:
        crossings.append(float(DEPTH_COUNT))
    if not crossings:
        return np.empty(0, dtype=np.float64)
    out = np.asarray(crossings, dtype=np.float64)
    keep = np.ones(out.size, dtype=bool)
    if out.size > 1:
        keep[1:] = np.diff(out) > 1.0e-12
    return out[keep]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid Stage-127 cosine payload")
    return float(np.dot(x, y) / max(float(np.linalg.norm(x) * np.linalg.norm(y)), 1.0e-300))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Invalid Stage-127 correlation payload")
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    return _cosine(xc, yc)


def _stage126_fraction_closure(
    profiles: list[dict[str, np.ndarray]],
    parent_wall_fraction: np.ndarray,
    parent_side_fraction: np.ndarray,
) -> float:
    wall_fraction = np.zeros((3, 2), dtype=np.float64)
    side_fraction = np.zeros((3, 2, 2), dtype=np.float64)
    band_totals = np.zeros(3, dtype=np.float64)

    for w, p in enumerate(profiles):
        for b in range(3):
            mask = p["band_code"] == b
            total = float(np.sum(p["same_sign_l1"][mask]))
            if total <= 0.0:
                raise ValueError("Stage-127 parent-closure band is empty")
            side_fraction[b, w, 0] = float(
                np.sum(p["same_sign_l1"][mask] * p["axis0_low_fraction"][mask]) / total
            )
            side_fraction[b, w, 1] = float(
                np.sum(p["same_sign_l1"][mask] * p["axis0_high_fraction"][mask]) / total
            )
            wall_fraction[b, w] = total
            band_totals[b] += total

    wall_fraction = wall_fraction / band_totals[:, None]
    return float(
        max(
            np.max(np.abs(wall_fraction - np.asarray(parent_wall_fraction, dtype=np.float64))),
            np.max(np.abs(side_fraction - np.asarray(parent_side_fraction, dtype=np.float64))),
        )
    )


def stage127_decision(
    *,
    finite: bool,
    parent_metric_closure: float,
    crossing_counts: list[int],
    crossing_separation: float,
    cross_wall_profile_cosine: float,
    depth_side_agreement: float,
) -> str:
    if not finite:
        return NONFINITE
    if parent_metric_closure > PARENT_METRIC_CLOSURE_TOLERANCE:
        return CLOSURE_BLOCKER
    if all(c == EXPECTED_CROSSINGS_PER_WALL for c in crossing_counts):
        if (
            crossing_separation <= MAX_CROSSING_SEPARATION_CELLS
            and cross_wall_profile_cosine >= CROSS_WALL_PROFILE_COSINE_MIN
            and depth_side_agreement >= DEPTH_SIDE_AGREEMENT_MIN
        ):
            return BILATERAL_SWITCH
        return WALL_SPECIFIC_SWITCH
    if any(c > 0 for c in crossing_counts):
        return WALL_SPECIFIC_SWITCH
    return NO_SWITCH


def run(stage126_dir: str | Path, stage126_record_path: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage127_design(**design)
    parent_summary, _, a = _load_stage126(stage126_dir, stage126_record_path)
    profiles = [wall_depth_profile(a["field"], a["band_index"], a["net_signs"], w) for w in range(2)]
    parent_closure = _stage126_fraction_closure(profiles, a["parent_wall_fraction"], a["parent_side_fraction"])

    crossings = [crossing_depths(p["tangential_asymmetry"]) for p in profiles]
    crossing_counts = [int(c.size) for c in crossings]
    crossing_separation = (
        abs(float(crossings[0][0]) - float(crossings[1][0]))
        if crossing_counts == [1, 1]
        else float("inf")
    )
    asym0 = profiles[0]["tangential_asymmetry"]
    asym1 = profiles[1]["tangential_asymmetry"]
    profile_cosine = _cosine(asym0, asym1)
    profile_pearson = _pearson(asym0, asym1)
    side_agreement = float(np.mean(profiles[0]["dominant_side_code"] == profiles[1]["dominant_side_code"]))

    finite = bool(
        np.isfinite(a["field"]).all()
        and np.isfinite(parent_closure)
        and np.isfinite(profile_cosine)
        and np.isfinite(profile_pearson)
        and all(np.isfinite(p["tangential_asymmetry"]).all() for p in profiles)
    )
    decision = stage127_decision(
        finite=finite,
        parent_metric_closure=parent_closure,
        crossing_counts=crossing_counts,
        crossing_separation=crossing_separation,
        cross_wall_profile_cosine=profile_cosine,
        depth_side_agreement=side_agreement,
    )

    first_low = []
    for p in profiles:
        idx = np.flatnonzero(p["dominant_side_code"] == 0)
        first_low.append(int(p["depth"][idx[0]]) if idx.size else None)

    metrics: dict[str, dict[str, object]] = {}
    for w, wall in enumerate(WALLS):
        p = profiles[w]
        metrics[wall] = {
            "crossing_count": crossing_counts[w],
            "interpolated_zero_crossing_depth_cells": float(crossings[w][0]) if crossing_counts[w] == 1 else None,
            "first_axis0_low_dominant_depth_cells": first_low[w],
            "minimum_depth_dominance_fraction": float(np.min(p["dominance_fraction"])),
            "maximum_depth_dominance_fraction": float(np.max(p["dominance_fraction"])),
            "depth_14_asymmetry_axis0_high_minus_low": float(p["tangential_asymmetry"][13]),
            "depth_15_asymmetry_axis0_high_minus_low": float(p["tangential_asymmetry"][14]),
            "depth_16_asymmetry_axis0_high_minus_low": float(p["tangential_asymmetry"][15]),
            "near_side_sequence": [TANGENTIAL_SIDES[int(v)] for v in p["dominant_side_code"][:4]],
            "mid_side_sequence": [TANGENTIAL_SIDES[int(v)] for v in p["dominant_side_code"][4:14]],
            "inner_side_sequence": [TANGENTIAL_SIDES[int(v)] for v in p["dominant_side_code"][14:]],
        }

    aggregate = {
        "maximum_stage126_parent_fraction_abs_error": float(parent_closure),
        "cross_wall_asymmetry_profile_cosine": float(profile_cosine),
        "cross_wall_asymmetry_profile_pearson": float(profile_pearson),
        "cross_wall_depth_side_agreement_fraction": float(side_agreement),
        "crossing_depth_separation_cells": float(crossing_separation),
        "wall_crossing_depths_cells": [float(c[0]) if c.size == 1 else None for c in crossings],
        "first_axis0_low_dominant_depths_cells": first_low,
        "depths_with_cross_wall_side_disagreement": [
            int(i + 1)
            for i in np.flatnonzero(profiles[0]["dominant_side_code"] != profiles[1]["dominant_side_code"])
        ],
    }

    if decision == BILATERAL_SWITCH:
        scientific_conclusion = (
            "The Stage-126 band-level tangential-side reversal resolves into one wall-normal zero crossing on each opposite wall, "
            "with highly coherent 28-depth asymmetry profiles and a crossing separation within the fixed two-cell guard. The switch "
            "is therefore bilateral and localized to a narrow transition zone rather than being a one-wall or broad-band averaging "
            "artifact. Because the Stage-125/126 dominant radial node and band net sign also change at the mid-to-inner band boundary, "
            "this does not yet identify a physical depth-switch mechanism; a fixed radial-node-continuity audit is required next. "
            "No limiter causality, MUSCL stability, endpoint convergence, q_av improvement, benchmark accuracy, or validation is claimed."
        )
    elif decision == WALL_SPECIFIC_SWITCH:
        scientific_conclusion = (
            "A wall-normal side switch is present but does not satisfy the preregistered bilateral coherence guards. The Stage-126 "
            "band reversal must therefore be treated as wall-specific or multi-transition structure; a fixed wall-specific depth-profile "
            "audit is justified and no solver parameter is changed."
        )
    elif decision == NO_SWITCH:
        scientific_conclusion = (
            "The Stage-126 band-level reversal does not reproduce as a wall-normal depth crossing in the resolved profiles. It should "
            "therefore be treated as a band-aggregation effect pending one fixed aggregation-origin audit; no solver parameter is changed."
        )
    else:
        scientific_conclusion = (
            "Stage 127 is blocked by nonfinite data or failure to reproduce the checksum-verified Stage-126 wall-side fractions. No "
            "depth-switch interpretation or parameter change is justified."
        )

    summary = {
        "stage": 127,
        "parent_stage126": {
            "run_id": STAGE126_RUN_ID,
            "job_id": STAGE126_JOB_ID,
            "artifact_id": STAGE126_ARTIFACT_ID,
            "source_head": STAGE126_SOURCE_HEAD,
            "completion_commit": STAGE126_COMPLETION_COMMIT,
            "decision": parent_summary["decision"],
        },
        "configuration": {
            "grid": list(GRID),
            "depth_count_per_wall": DEPTH_COUNT,
            "bands": list(BANDS),
            "walls": list(WALLS),
            "tangential_sides": list(TANGENTIAL_SIDES),
            "asymmetry_definition": "axis0_high same-sign L1 fraction minus axis0_low same-sign L1 fraction at each wall-normal depth",
            "crossing_interpolation": "linear between adjacent integer-depth asymmetry samples of opposite sign",
            "parent_metric_closure_tolerance": PARENT_METRIC_CLOSURE_TOLERANCE,
            "cross_wall_profile_cosine_min": CROSS_WALL_PROFILE_COSINE_MIN,
            "depth_side_agreement_min": DEPTH_SIDE_AGREEMENT_MIN,
            "max_crossing_separation_cells": MAX_CROSSING_SEPARATION_CELLS,
            "expected_crossings_per_wall": EXPECTED_CROSSINGS_PER_WALL,
            "artifact_only": True,
            "solver_rerun": False,
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
            "solver_endpoint_advanced": False,
            "cross_knudsen_extension_permitted": False,
            "benchmark_or_validation_claim_permitted": False,
        },
        "metrics": metrics,
        "aggregate": aggregate,
        "finite": finite,
        "decision": decision,
        "scientific_conclusion": scientific_conclusion,
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. No physical, collision/source, floor, wall, "
            "reconstruction, transport, limiter, normalization, source-relaxation, or velocity-quadrature parameter is retuned; no "
            "solver endpoint or cross-Knudsen extension is advanced."
        ),
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        output / "wall_normal_side_switch.npz",
        depth=np.arange(1, DEPTH_COUNT + 1, dtype=np.int16),
        wall0_band_code=profiles[0]["band_code"],
        wall1_band_code=profiles[1]["band_code"],
        wall0_axis0_low_fraction=profiles[0]["axis0_low_fraction"],
        wall0_axis0_high_fraction=profiles[0]["axis0_high_fraction"],
        wall1_axis0_low_fraction=profiles[1]["axis0_low_fraction"],
        wall1_axis0_high_fraction=profiles[1]["axis0_high_fraction"],
        wall0_tangential_asymmetry=profiles[0]["tangential_asymmetry"],
        wall1_tangential_asymmetry=profiles[1]["tangential_asymmetry"],
        wall0_dominant_side_code=profiles[0]["dominant_side_code"],
        wall1_dominant_side_code=profiles[1]["dominant_side_code"],
        wall0_same_sign_l1=profiles[0]["same_sign_l1"],
        wall1_same_sign_l1=profiles[1]["same_sign_l1"],
        wall0_uncancelled_fraction=profiles[0]["uncancelled_fraction"],
        wall1_uncancelled_fraction=profiles[1]["uncancelled_fraction"],
        parent_dominant_nodes=a["dominant_nodes"],
        parent_net_signs=a["net_signs"],
        parent_band_tangential_side_codes=a["parent_band_side"],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 127 fixed wall-normal side-switch localization audit")
    parser.add_argument("--stage126-dir", required=True)
    parser.add_argument("--stage126-record", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run(args.stage126_dir, args.stage126_record, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
