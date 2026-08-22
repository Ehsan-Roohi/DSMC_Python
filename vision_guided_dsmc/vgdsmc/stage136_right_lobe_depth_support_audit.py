from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = 136
EXPECTED_STAGE135_SOURCE_HEAD = "71442743df6128d7b457b6f7522ccb9a27dce56e"
EXPECTED_STAGE135_RUN_ID = 32070772876
EXPECTED_STAGE135_JOB_ID = 95513279085
EXPECTED_STAGE135_ARTIFACT_ID = 9310452785
EXPECTED_STAGE135_ARTIFACT_SHA256 = "ef3de56bb2496fe2fc75e816d350455ee6da4986f38189c53a09d2f2b570d5c1"
EXPECTED_STAGE135_SUMMARY_SHA256 = "e959a53f3ecfee13073b32dfd530dc67d8930897a69cde8cdec33bf539197354"
EXPECTED_STAGE135_PAYLOAD_SHA256 = "3d8500ff85a4dbb5dd59d5ec5742ab49c35e2fb6b620b394d2fba46f0bf224b1"
RIGHT_SUPPORT_MIN_DEPTH = 5.0
NEAR_SAMPLE_COUNT = 3
NEAR_LOCALIZATION_SHARE_MIN = 0.75
BROAD_EFFECTIVE_COUNT_MIN = 4.5
COMMON_PROFILE_COSINE_MIN = 0.95
CLOSURE_MAX = 1.0e-12

NONFINITE = "stage136_nonfinite_blocker"
PARENT_CLOSURE_BLOCKER = "stage136_parent_closure_blocker"
PARENT_RECORD_BLOCKER = "stage136_parent_record_blocker"
NEAR_LOCALIZED = "stage136_crossing_localized_right_lobe_stage137_crossing_local_shape_audit"
DISTRIBUTED_COMMON = "stage136_distributed_common_right_lobe_support_stage137_right_lobe_decay_shape_audit"
DISTRIBUTED_MIXED = "stage136_distributed_mixed_right_lobe_support_stage137_channel_specific_depth_audit"


def validate_stage136_design(
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
    right_support_min_depth=RIGHT_SUPPORT_MIN_DEPTH,
    near_sample_count=NEAR_SAMPLE_COUNT,
    solver_rerun=False,
    solver_endpoint_advanced=False,
    physical_parameter_retuning=False,
):
    expected = {
        "grid": (64, 64), "interior_grid": (56, 56), "kn0": 10.0,
        "cold_hot_ratio": 0.1, "rule": (40, 96), "radial_scale": 2.0,
        "limiter": "minmod", "boundary_slope": "zero", "source_relaxation": 1.0,
        "correction_floor": 0.05, "witness_node": 9, "pair_sectors": (5, 6),
        "dominant_mirrored_sector": 6, "right_support_min_depth": RIGHT_SUPPORT_MIN_DEPTH,
        "near_sample_count": NEAR_SAMPLE_COUNT, "solver_rerun": False,
        "solver_endpoint_advanced": False, "physical_parameter_retuning": False,
    }
    got = locals()
    for key, value in expected.items():
        if got[key] != value:
            raise ValueError(f"Stage 136 frozen-design violation: {key}={got[key]!r}, expected {value!r}")


def _sign_coherence(values: np.ndarray) -> float:
    denom = float(np.sum(np.abs(values)))
    return abs(float(np.sum(values))) / denom if denom > 0.0 else 0.0


def depth_support_metrics(
    relative_depth: np.ndarray,
    difference: np.ndarray,
    right_mask: np.ndarray,
    *,
    near_sample_count: int = NEAR_SAMPLE_COUNT,
) -> dict[str, float | int | bool]:
    x = np.asarray(relative_depth, dtype=float)
    d = np.asarray(difference, dtype=float)
    mask = np.asarray(right_mask, dtype=bool)
    if x.ndim != 1 or d.ndim != 1 or mask.ndim != 1 or not (x.shape == d.shape == mask.shape):
        raise ValueError("Stage 136 requires equal-length one-dimensional depth, difference, and mask arrays")
    if not np.isfinite(np.concatenate([x, d])).all():
        raise ValueError("Stage 136 depth-support inputs must be finite")
    if near_sample_count <= 0:
        raise ValueError("near_sample_count must be positive")
    xr = x[mask]
    dr = d[mask]
    if xr.size < 2 * near_sample_count:
        raise ValueError("Stage 136 right-lobe support is too short for fixed near/far diagnostics")
    if np.any(xr < RIGHT_SUPPORT_MIN_DEPTH):
        raise ValueError("Stage 136 right-lobe mask includes depths below the frozen support threshold")
    order = np.argsort(xr)
    xr = xr[order]
    dr = dr[order]
    w = np.abs(dr)
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("Stage 136 right-lobe mismatch has zero L1 magnitude")
    p = w / total
    centroid = float(np.sum(xr * p))
    spread = float(np.sqrt(np.sum((xr - centroid) ** 2 * p)))
    cumulative = np.cumsum(p)
    half_idx = int(np.searchsorted(cumulative, 0.5, side="left"))
    monotone_steps = int(np.sum(np.diff(w) <= 0.0))
    return {
        "sample_count": int(xr.size),
        "l1_mismatch": total,
        "sign_coherence": _sign_coherence(dr),
        "uniform_negative_sign": bool(np.all(dr < 0.0)),
        "near_l1_share": float(np.sum(p[:near_sample_count])),
        "far_l1_share": float(np.sum(p[-near_sample_count:])),
        "max_single_sample_share": float(np.max(p)),
        "effective_sample_count": float(1.0 / np.sum(p * p)),
        "weighted_centroid_depth_cells": centroid,
        "weighted_spread_cells": spread,
        "half_l1_depth_cells": float(xr[half_idx]),
        "endpoint_to_nearest_magnitude_ratio": float(w[-1] / w[0]),
        "nonincreasing_step_fraction": float(monotone_steps / (w.size - 1)),
    }


def profile_cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.shape != bb.shape or aa.ndim != 1:
        raise ValueError("Stage 136 profile cosine requires equal one-dimensional arrays")
    if not np.isfinite(np.concatenate([aa, bb])).all():
        raise ValueError("Stage 136 profile cosine inputs must be finite")
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    return float(np.dot(aa, bb) / (na * nb)) if na > 0.0 and nb > 0.0 else 0.0


def classify_depth_support(
    *,
    dominant: dict,
    parent: dict,
    common_profile_cosine: float,
    finite: bool = True,
    parent_closure: float = 0.0,
    parent_record_ok: bool = True,
) -> str:
    numeric = [float(common_profile_cosine), float(parent_closure)]
    for block in (dominant, parent):
        numeric.extend(float(v) for v in block.values() if isinstance(v, (float, int)) and not isinstance(v, bool))
    if not finite or not np.isfinite(numeric).all():
        return NONFINITE
    if not parent_record_ok:
        return PARENT_RECORD_BLOCKER
    if parent_closure > CLOSURE_MAX:
        return PARENT_CLOSURE_BLOCKER
    if min(float(dominant["near_l1_share"]), float(parent["near_l1_share"])) >= NEAR_LOCALIZATION_SHARE_MIN:
        return NEAR_LOCALIZED
    broad = min(float(dominant["effective_sample_count"]), float(parent["effective_sample_count"])) >= BROAD_EFFECTIVE_COUNT_MIN
    common = float(common_profile_cosine) >= COMMON_PROFILE_COSINE_MIN
    if broad and common:
        return DISTRIBUTED_COMMON
    return DISTRIBUTED_MIXED


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_stage135_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 135
        and record.get("source_head") == EXPECTED_STAGE135_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE135_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE135_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE135_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE135_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE135_SUMMARY_SHA256
        and record.get("fixed_lobe_amplitude_origin_sha256") == EXPECTED_STAGE135_PAYLOAD_SHA256
        and record.get("decision") == "stage135_right_negative_lobe_dominant_stage136_right_lobe_depth_support_audit"
    )


def run_stage136(stage135_dir: Path, stage135_record: Path, output_dir: Path) -> dict:
    validate_stage136_design()
    stage135 = _load_json(stage135_dir / "summary.json")
    record135 = _load_json(stage135_record)
    record_ok = _check_stage135_record(record135)
    if stage135.get("stage") != 135 or stage135.get("decision") != "stage135_right_negative_lobe_dominant_stage136_right_lobe_depth_support_audit":
        raise ValueError("Stage 136 requires the completed Stage 135 right-lobe route")

    with np.load(stage135_dir / "fixed_lobe_amplitude_origin.npz") as data:
        x = np.asarray(data["relative_depth"], dtype=float)
        dominant_diff = np.asarray(data["dominant_cross_wall_difference"], dtype=float)
        parent_diff = np.asarray(data["parent_cross_wall_difference"], dtype=float)
        right_mask = np.asarray(data["right_lobe_mask"], dtype=bool)

    dominant = depth_support_metrics(x, dominant_diff, right_mask)
    parent = depth_support_metrics(x, parent_diff, right_mask)
    dom_abs = np.abs(dominant_diff[right_mask])
    par_abs = np.abs(parent_diff[right_mask])
    cosine = profile_cosine(dom_abs, par_abs)
    dom_norm = dom_abs / np.sum(dom_abs)
    par_norm = par_abs / np.sum(par_abs)
    tv_distance = float(0.5 * np.sum(np.abs(dom_norm - par_norm)))
    parent_closure = float(stage135["aggregate"]["maximum_parent_closure"])
    finite = bool(stage135.get("finite", False) and np.isfinite(np.concatenate([x, dominant_diff, parent_diff])).all())
    decision = classify_depth_support(
        dominant=dominant,
        parent=parent,
        common_profile_cosine=cosine,
        finite=finite,
        parent_closure=parent_closure,
        parent_record_ok=record_ok,
    )

    cfg = dict(stage135["configuration"])
    cfg.update({
        "right_support_min_depth_cells": RIGHT_SUPPORT_MIN_DEPTH,
        "near_sample_count": NEAR_SAMPLE_COUNT,
        "near_localization_share_min": NEAR_LOCALIZATION_SHARE_MIN,
        "broad_effective_count_min": BROAD_EFFECTIVE_COUNT_MIN,
        "common_profile_cosine_min": COMMON_PROFILE_COSINE_MIN,
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

    if decision == DISTRIBUTED_COMMON:
        conclusion = (
            "The Stage-135 right/negative lobe is not confined to the samples nearest the crossing. Both retained channels have broad effective depth support and their normalized right-lobe magnitude profiles are strongly aligned. "
            "The next justified artifact-only diagnostic is a fixed right-lobe decay-shape audit to determine whether the common depth dependence is smooth attenuation or contains resolved structure. "
            "This does not establish limiter causality, MUSCL stability, endpoint convergence, q_av improvement, benchmark accuracy, or validation."
        )
    elif decision == NEAR_LOCALIZED:
        conclusion = (
            "The Stage-135 right-lobe mismatch is concentrated in the fixed samples nearest the crossing in both retained channels. A crossing-local shape audit is justified next; no solver or parameter change is implied."
        )
    elif decision == DISTRIBUTED_MIXED:
        conclusion = (
            "The Stage-135 right-lobe mismatch is not jointly crossing-localized, but the two retained channels do not satisfy the preregistered common-profile criterion. A channel-specific depth audit is justified next; no solver or parameter change is implied."
        )
    else:
        conclusion = "Stage 136 is blocked by a finite-data, parent-record, or exact-parent closure guard; no scientific routing is claimed."

    summary = {
        "stage": STAGE,
        "finite": finite,
        "configuration": cfg,
        "parents": {
            "stage135_run_id": EXPECTED_STAGE135_RUN_ID,
            "stage135_job_id": EXPECTED_STAGE135_JOB_ID,
            "stage135_artifact_id": EXPECTED_STAGE135_ARTIFACT_ID,
            "stage135_source_head": EXPECTED_STAGE135_SOURCE_HEAD,
        },
        "metrics": {
            "dominant_sector": dominant,
            "parent_profile": parent,
            "normalized_right_lobe_profile_cosine": cosine,
            "normalized_right_lobe_profile_tv_distance": tv_distance,
        },
        "aggregate": {
            "maximum_near_l1_share": max(float(dominant["near_l1_share"]), float(parent["near_l1_share"])),
            "minimum_effective_sample_count": min(float(dominant["effective_sample_count"]), float(parent["effective_sample_count"])),
            "minimum_sign_coherence": min(float(dominant["sign_coherence"]), float(parent["sign_coherence"])),
            "normalized_profile_cosine": cosine,
            "normalized_profile_tv_distance": tv_distance,
            "maximum_parent_closure": parent_closure,
            "parent_record_ok": record_ok,
        },
        "decision": decision,
        "scientific_conclusion": conclusion,
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. No physical, collision/source, floor, wall, reconstruction, transport, limiter, normalization, source-relaxation, or velocity-quadrature parameter is retuned; no solver endpoint or cross-Knudsen extension is advanced."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    xr = x[right_mask]
    np.savez_compressed(
        output_dir / "right_lobe_depth_support.npz",
        right_depth=xr,
        dominant_right_difference=dominant_diff[right_mask],
        parent_right_difference=parent_diff[right_mask],
        dominant_normalized_magnitude=dom_norm,
        parent_normalized_magnitude=par_norm,
    )
    print(json.dumps({"stage": STAGE, "decision": decision, "aggregate": summary["aggregate"]}, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 136 right-lobe depth-support audit")
    parser.add_argument("--stage135-dir", type=Path, required=True)
    parser.add_argument("--stage135-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_stage136(args.stage135_dir, args.stage135_record, args.output_dir)


if __name__ == "__main__":
    main()
