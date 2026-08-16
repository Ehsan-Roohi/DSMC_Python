from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STAGE125_RUN_ID = 31911866789
STAGE125_JOB_ID = 95078076620
STAGE125_ARTIFACT_ID = 9256993363
STAGE125_ARTIFACT_SHA256 = "3528093dfe7687325f595a57b61e47e87648d389498f01a79fabcd0e18758994"
STAGE125_SUMMARY_SHA256 = "f6ab22f038cdd9204ec84b4c9b02bc414001276895908d4202ac8dda24ee7e9b"
STAGE125_PAYLOAD_SHA256 = "ac982e826aa5a9748f02911021c397ae7426e9b92732da612b1ecfffe0ec2970"
STAGE125_SOURCE_HEAD = "58f09e06a40721d4de446ac91035a78514b43e54"
STAGE125_COMPLETION_COMMIT = "350cb36f00f3c868f2c64a8e85c00a17887099fe"
STAGE125_DECISION = (
    "stage125_persistent_same_sign_halfspace_localization_"
    "stage126_wall_side_asymmetry_audit"
)

GRID = (56, 56)
BANDS = ("near_1_4", "mid_5_14", "inner_15_28")
WALL_HALVES = ("axis1_low", "axis1_high")
TANGENTIAL_SIDES = ("axis0_low", "axis0_high")

# Fixed artifact-classification guards only. They do not tune the DVM or failed MUSCL endpoint.
PARENT_METRIC_CLOSURE_TOLERANCE = 1.0e-12
WALL_BALANCE_MIN = 0.35
PER_WALL_TANGENTIAL_DOMINANCE_MIN = 0.70

BILATERAL_REVERSAL = (
    "stage126_bilateral_wall_consistent_with_depth_side_reversal_"
    "stage127_wall_normal_side_switch_audit"
)
BILATERAL_SINGLE_SIDE = (
    "stage126_bilateral_wall_consistent_single_tangential_side_"
    "stage127_tangential_boundary_origin_audit"
)
ONE_WALL = (
    "stage126_one_wall_dominates_same_sign_remainder_"
    "stage127_wall_specific_amplitude_audit"
)
INCOHERENT = (
    "stage126_cross_wall_tangential_side_incoherence_"
    "stage127_wall_specific_topology_audit"
)
NONFINITE = "stage126_nonfinite_wall_side_blocker_without_retuning"
CLOSURE_BLOCKER = "stage126_parent_metric_closure_blocker_without_retuning"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_stage126_design(**overrides: object) -> None:
    frozen = {
        "grid": GRID,
        "bands": BANDS,
        "wall_halves": WALL_HALVES,
        "tangential_sides": TANGENTIAL_SIDES,
        "parent_metric_closure_tolerance": PARENT_METRIC_CLOSURE_TOLERANCE,
        "wall_balance_min": WALL_BALANCE_MIN,
        "per_wall_tangential_dominance_min": PER_WALL_TANGENTIAL_DOMINANCE_MIN,
        "stage125_run_id": STAGE125_RUN_ID,
        "stage125_job_id": STAGE125_JOB_ID,
        "stage125_artifact_id": STAGE125_ARTIFACT_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 126 is fixed to the exact completed Stage-125 artifact and preregistered "
            "wall-side classification guards; it may not retune physics, wall/collision/source "
            "treatment, reconstruction, transport, limiter, floors, normalization, source "
            "relaxation, velocity quadrature, failed MUSCL parameters, or diagnostic thresholds"
        )


def _load_stage125(root: str | Path, record_path: str | Path):
    root = Path(root)
    expected = {
        "summary.json": STAGE125_SUMMARY_SHA256,
        "dominant_node_spatial_sign.npz": STAGE125_PAYLOAD_SHA256,
    }
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Stage-125 checksum mismatch: {name}")

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    if summary.get("stage") != 125 or summary.get("decision") != STAGE125_DECISION or summary.get("finite") is not True:
        raise ValueError("Stage-125 artifact does not authorize Stage 126")
    checks = (
        record.get("stage") == 125,
        record.get("decision") == STAGE125_DECISION,
        record.get("source_head") == STAGE125_SOURCE_HEAD,
        record.get("workflow_status") == "completed",
        record.get("workflow_conclusion") == "success",
        record.get("workflow_run_id") == STAGE125_RUN_ID,
        record.get("workflow_job_id") == STAGE125_JOB_ID,
        record.get("artifact_id") == STAGE125_ARTIFACT_ID,
        record.get("artifact_sha256") == STAGE125_ARTIFACT_SHA256,
        record.get("summary_sha256") == STAGE125_SUMMARY_SHA256,
        record.get("dominant_node_spatial_sign_sha256") == STAGE125_PAYLOAD_SHA256,
        record.get("tests", {}).get("passed") == 14,
        record.get("tests", {}).get("failed") == 0,
    )
    if not all(checks):
        raise ValueError("Committed Stage-125 provenance does not authorize Stage 126")

    with np.load(root / "dominant_node_spatial_sign.npz") as data:
        needed = {
            "dominant_node_residual",
            "band_index",
            "pass_mask",
            "dominant_nodes",
            "net_signs",
            "parent_node_net",
            "parent_node_abs",
            "parent_node_uncancelled",
        }
        if not needed.issubset(data.files):
            raise ValueError("Stage-125 payload is incomplete")
        arrays = {name: np.asarray(data[name]).copy() for name in needed}

    field = np.asarray(arrays["dominant_node_residual"], dtype=np.float64)
    band_index = np.asarray(arrays["band_index"], dtype=np.int8)
    passed = np.asarray(arrays["pass_mask"], dtype=bool)
    dominant_nodes = np.asarray(arrays["dominant_nodes"], dtype=np.int16)
    net_signs = np.asarray(arrays["net_signs"], dtype=np.int8)
    if field.shape != GRID or band_index.shape != GRID or passed.shape != GRID:
        raise ValueError("Stage-125 spatial map shape mismatch")
    if dominant_nodes.shape != (3,) or net_signs.shape != (3,):
        raise ValueError("Stage-125 band metadata shape mismatch")
    if not np.isfinite(field).all():
        raise ValueError("Stage-125 wall-side input is nonfinite")
    if not np.all(np.isin(net_signs, (-1, 1))):
        raise ValueError("Stage-125 band net signs are invalid")
    return summary, record, {
        "field": field,
        "band_index": band_index,
        "passed": passed,
        "dominant_nodes": dominant_nodes,
        "net_signs": net_signs,
    }


def recompute_halfspace_fractions(
    field: np.ndarray, band_mask: np.ndarray, net_sign: int
) -> dict[str, float]:
    x = np.asarray(field, dtype=np.float64)
    band = np.asarray(band_mask, dtype=bool)
    if x.shape != GRID or band.shape != GRID or net_sign not in (-1, 1):
        raise ValueError("Invalid Stage-126 halfspace payload")
    same = band & ((x * net_sign) > 0.0)
    weights = np.abs(x)
    total = float(np.sum(weights[same]))
    if total <= 0.0:
        raise ValueError("Stage-126 same-sign support is empty")
    ii, jj = np.indices(GRID)
    masks = {
        "axis0_low": ii < GRID[0] // 2,
        "axis0_high": ii >= GRID[0] // 2,
        "axis1_low": jj < GRID[1] // 2,
        "axis1_high": jj >= GRID[1] // 2,
    }
    return {name: float(np.sum(weights[same & mask]) / total) for name, mask in masks.items()}


def wall_side_metrics(
    field: np.ndarray, band_mask: np.ndarray, net_sign: int
) -> dict[str, object]:
    x = np.asarray(field, dtype=np.float64)
    band = np.asarray(band_mask, dtype=bool)
    if x.shape != GRID or band.shape != GRID or net_sign not in (-1, 1):
        raise ValueError("Invalid Stage-126 wall-side payload")
    same = band & ((x * net_sign) > 0.0)
    weights = np.abs(x)
    total_same = float(np.sum(weights[same]))
    if total_same <= 0.0:
        raise ValueError("Stage-126 same-sign support is empty")

    ii, jj = np.indices(GRID)
    wall_masks = (jj < GRID[1] // 2, jj >= GRID[1] // 2)
    side_masks = (ii < GRID[0] // 2, ii >= GRID[0] // 2)

    walls: dict[str, dict[str, object]] = {}
    dominant_codes: list[int] = []
    wall_shares: list[float] = []
    dominant_fractions: list[float] = []
    for wall_name, wall_mask in zip(WALL_HALVES, wall_masks):
        same_wall = same & wall_mask
        wall_same_l1 = float(np.sum(weights[same_wall]))
        wall_share = wall_same_l1 / total_same
        if wall_same_l1 <= 0.0:
            low_fraction = high_fraction = 0.0
            dominant_code = -1
            dominant_fraction = 0.0
        else:
            low_fraction = float(np.sum(weights[same_wall & side_masks[0]]) / wall_same_l1)
            high_fraction = float(np.sum(weights[same_wall & side_masks[1]]) / wall_same_l1)
            dominant_code = 0 if low_fraction >= high_fraction else 1
            dominant_fraction = max(low_fraction, high_fraction)
        all_wall = band & wall_mask
        wall_total_l1 = float(np.sum(weights[all_wall]))
        wall_signed = float(np.sum(x[all_wall]))
        walls[wall_name] = {
            "same_sign_l1_fraction_of_band": wall_share,
            "axis0_low_fraction_within_wall_same_sign": low_fraction,
            "axis0_high_fraction_within_wall_same_sign": high_fraction,
            "dominant_tangential_side": TANGENTIAL_SIDES[dominant_code] if dominant_code >= 0 else None,
            "dominant_tangential_side_fraction": dominant_fraction,
            "signed_residual": wall_signed,
            "uncancelled_fraction": abs(wall_signed) / wall_total_l1 if wall_total_l1 > 0.0 else 0.0,
        }
        dominant_codes.append(dominant_code)
        wall_shares.append(wall_share)
        dominant_fractions.append(dominant_fraction)

    combined = recompute_halfspace_fractions(x, band, net_sign)
    combined_side_code = 0 if combined["axis0_low"] >= combined["axis0_high"] else 1
    return {
        "walls": walls,
        "cross_wall_same_tangential_side": bool(dominant_codes[0] == dominant_codes[1] and dominant_codes[0] >= 0),
        "combined_dominant_tangential_side": TANGENTIAL_SIDES[combined_side_code],
        "combined_dominant_tangential_side_fraction": max(combined["axis0_low"], combined["axis0_high"]),
        "minimum_wall_same_sign_l1_fraction": min(wall_shares),
        "minimum_per_wall_tangential_dominance_fraction": min(dominant_fractions),
        "halfspace_l1_fractions": combined,
    }


def stage126_decision(
    *,
    finite: bool,
    parent_metric_closure: float,
    minimum_wall_balance: float,
    minimum_per_wall_tangential_dominance: float,
    cross_wall_same_side: list[bool],
    band_side_codes: list[int],
) -> str:
    if not finite:
        return NONFINITE
    if parent_metric_closure > PARENT_METRIC_CLOSURE_TOLERANCE:
        return CLOSURE_BLOCKER
    if minimum_wall_balance < WALL_BALANCE_MIN:
        return ONE_WALL
    coherent = all(cross_wall_same_side) and minimum_per_wall_tangential_dominance >= PER_WALL_TANGENTIAL_DOMINANCE_MIN
    if not coherent:
        return INCOHERENT
    reversals = sum(int(a != b) for a, b in zip(band_side_codes[:-1], band_side_codes[1:]))
    if reversals >= 1:
        return BILATERAL_REVERSAL
    return BILATERAL_SINGLE_SIDE


def run(stage125_dir: str | Path, stage125_record_path: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage126_design(**design)
    parent_summary, _, a = _load_stage125(stage125_dir, stage125_record_path)

    metrics: dict[str, dict[str, object]] = {}
    wall_balance: list[float] = []
    wall_dom: list[float] = []
    coherent: list[bool] = []
    band_side_codes: list[int] = []
    parent_metric_closure = 0.0
    wall_fraction_array = np.zeros((3, 2), dtype=np.float64)
    tangential_fraction_array = np.zeros((3, 2, 2), dtype=np.float64)

    for i, band_name in enumerate(BANDS):
        mask = a["band_index"] == i
        m = wall_side_metrics(a["field"], mask, int(a["net_signs"][i]))
        parent_halfspaces = parent_summary["metrics"][band_name]["halfspace_l1_fractions"]
        for name, value in m["halfspace_l1_fractions"].items():
            parent_metric_closure = max(parent_metric_closure, abs(float(value) - float(parent_halfspaces[name])))
        expected_side = parent_summary["metrics"][band_name]["dominant_halfspace"]
        if m["combined_dominant_tangential_side"] != expected_side:
            raise ValueError("Stage-125 dominant tangential side does not reproduce")
        for w, wall_name in enumerate(WALL_HALVES):
            wm = m["walls"][wall_name]
            wall_fraction_array[i, w] = float(wm["same_sign_l1_fraction_of_band"])
            tangential_fraction_array[i, w, 0] = float(wm["axis0_low_fraction_within_wall_same_sign"])
            tangential_fraction_array[i, w, 1] = float(wm["axis0_high_fraction_within_wall_same_sign"])
        metrics[band_name] = {
            "dominant_radial_node": int(a["dominant_nodes"][i]),
            "net_sign": int(a["net_signs"][i]),
            **m,
        }
        wall_balance.append(float(m["minimum_wall_same_sign_l1_fraction"]))
        wall_dom.append(float(m["minimum_per_wall_tangential_dominance_fraction"]))
        coherent.append(bool(m["cross_wall_same_tangential_side"]))
        band_side_codes.append(TANGENTIAL_SIDES.index(str(m["combined_dominant_tangential_side"])))

    finite = bool(
        np.isfinite(a["field"]).all()
        and np.isfinite(parent_metric_closure)
        and np.isfinite(wall_fraction_array).all()
        and np.isfinite(tangential_fraction_array).all()
    )
    reversal_count = sum(int(a0 != a1) for a0, a1 in zip(band_side_codes[:-1], band_side_codes[1:]))
    decision = stage126_decision(
        finite=finite,
        parent_metric_closure=parent_metric_closure,
        minimum_wall_balance=min(wall_balance),
        minimum_per_wall_tangential_dominance=min(wall_dom),
        cross_wall_same_side=coherent,
        band_side_codes=band_side_codes,
    )

    if decision == BILATERAL_REVERSAL:
        scientific_conclusion = (
            "The Stage-125 side localization is bilateral across the two opposite wall-normal halves: "
            "each half contributes materially and independently selects the same tangential side within a band. "
            "The preferred tangential side is axis0_high in the near and mid wall-distance bands but axis0_low in "
            "the inner band, so the asymmetry is not a one-wall artifact and reverses with wall-normal depth. "
            "A fixed wall-normal side-switch localization audit is justified. This remains artifact attribution "
            "only and does not establish limiter causality, MUSCL stability, endpoint convergence, q_av improvement, "
            "benchmark accuracy, or validation."
        )
    elif decision == BILATERAL_SINGLE_SIDE:
        scientific_conclusion = (
            "Both opposite wall-normal halves materially support the same tangential-side localization at all "
            "three wall-distance bands without a depth reversal. A fixed tangential-boundary-origin audit is justified; "
            "no solver parameter is changed."
        )
    elif decision == ONE_WALL:
        scientific_conclusion = (
            "At least one wall-distance band is dominated by a single wall-normal half under the fixed balance guard. "
            "A wall-specific amplitude audit is justified before any broader spatial interpretation; no solver parameter is changed."
        )
    elif decision == INCOHERENT:
        scientific_conclusion = (
            "The two opposite wall-normal halves do not consistently select the same tangential side under the fixed "
            "dominance guard. A wall-specific topology audit is justified; no solver parameter is changed."
        )
    else:
        scientific_conclusion = (
            "Stage 126 is blocked by nonfinite data or failure to reproduce the exact Stage-125 halfspace metrics. "
            "No wall-side interpretation or parameter change is justified."
        )

    aggregate = {
        "maximum_parent_halfspace_fraction_abs_error": float(parent_metric_closure),
        "minimum_wall_same_sign_l1_fraction": float(min(wall_balance)),
        "minimum_per_wall_tangential_dominance_fraction": float(min(wall_dom)),
        "cross_wall_same_side_band_count": int(sum(coherent)),
        "depth_tangential_side_reversal_count": int(reversal_count),
        "band_tangential_side_sequence": [TANGENTIAL_SIDES[c] for c in band_side_codes],
    }
    summary = {
        "stage": 126,
        "parent_stage125": {
            "run_id": STAGE125_RUN_ID,
            "job_id": STAGE125_JOB_ID,
            "artifact_id": STAGE125_ARTIFACT_ID,
            "source_head": STAGE125_SOURCE_HEAD,
            "completion_commit": STAGE125_COMPLETION_COMMIT,
            "decision": parent_summary["decision"],
        },
        "configuration": {
            "grid": list(GRID),
            "bands": list(BANDS),
            "wall_halves": list(WALL_HALVES),
            "tangential_sides": list(TANGENTIAL_SIDES),
            "parent_metric_closure_tolerance": PARENT_METRIC_CLOSURE_TOLERANCE,
            "wall_balance_min": WALL_BALANCE_MIN,
            "per_wall_tangential_dominance_min": PER_WALL_TANGENTIAL_DOMINANCE_MIN,
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
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. No physical, collision/source, "
            "floor, wall, reconstruction, transport, limiter, normalization, source-relaxation, or velocity-quadrature "
            "parameter is retuned; no solver endpoint or cross-Knudsen extension is advanced."
        ),
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        output / "wall_side_asymmetry.npz",
        dominant_node_residual=a["field"],
        band_index=a["band_index"],
        pass_mask=a["passed"],
        dominant_nodes=a["dominant_nodes"],
        net_signs=a["net_signs"],
        wall_same_sign_l1_fractions=wall_fraction_array,
        wall_tangential_side_fractions=tangential_fraction_array,
        band_tangential_side_codes=np.asarray(band_side_codes, dtype=np.int8),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 126 fixed wall-side asymmetry audit")
    parser.add_argument("--stage125-dir", required=True)
    parser.add_argument("--stage125-record", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run(args.stage125_dir, args.stage125_record, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
