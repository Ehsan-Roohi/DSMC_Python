from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = 135
EXPECTED_STAGE134_SOURCE_HEAD = "e23493c4bc3091dd7f790ac69e8a2170a385e02d"
EXPECTED_STAGE134_RUN_ID = 32059776540
EXPECTED_STAGE134_ARTIFACT_ID = 9300793468
EXPECTED_STAGE134_SUMMARY_SHA256 = "a3d02ccb3e2fa364f67749f82b54f259bf230189ee172accac8563ef9684b702"
EXPECTED_STAGE134_PAYLOAD_SHA256 = "95959e271dc6d55d7cb1a984d8cd797777fd24b2a9c2418f941254640ef2caa4"
EXPECTED_STAGE132_RUN_ID = 32001930265
EXPECTED_STAGE132_ARTIFACT_ID = 9284552744
PLATEAU_DEPTH_CELLS = 5.0
LOBE_L1_SHARE_MIN = 0.70
SIGN_COHERENCE_MIN = 0.95
CLOSURE_MAX = 1.0e-12

NONFINITE = "stage135_nonfinite_blocker"
PARENT_CLOSURE_BLOCKER = "stage135_parent_closure_blocker"
SHIFT_CLOSURE_BLOCKER = "stage135_stage134_lobe_shift_closure_blocker"
RIGHT_LOBE = "stage135_right_negative_lobe_dominant_stage136_right_lobe_depth_support_audit"
LEFT_LOBE = "stage135_left_positive_lobe_dominant_stage136_left_lobe_depth_support_audit"
MIXED_LOBES = "stage135_mixed_lobe_amplitude_origin_stage136_paired_lobe_residual_audit"


def validate_stage135_design(
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
    plateau_depth_cells=PLATEAU_DEPTH_CELLS,
    solver_rerun=False,
    solver_endpoint_advanced=False,
    physical_parameter_retuning=False,
):
    expected = {
        "grid": (64, 64), "interior_grid": (56, 56), "kn0": 10.0,
        "cold_hot_ratio": 0.1, "rule": (40, 96), "radial_scale": 2.0,
        "limiter": "minmod", "boundary_slope": "zero", "source_relaxation": 1.0,
        "correction_floor": 0.05, "witness_node": 9, "pair_sectors": (5, 6),
        "dominant_mirrored_sector": 6, "plateau_depth_cells": PLATEAU_DEPTH_CELLS,
        "solver_rerun": False, "solver_endpoint_advanced": False,
        "physical_parameter_retuning": False,
    }
    got = locals()
    for key, value in expected.items():
        if got[key] != value:
            raise ValueError(f"Stage 135 frozen-design violation: {key}={got[key]!r}, expected {value!r}")


def _sign_coherence(values: np.ndarray) -> float:
    denom = float(np.sum(np.abs(values)))
    return abs(float(np.sum(values))) / denom if denom > 0.0 else 0.0


def _constant_shift_relative_l2(values: np.ndarray) -> float:
    denom = float(np.linalg.norm(values))
    if denom == 0.0:
        return 0.0
    mean = float(np.mean(values))
    return float(np.linalg.norm(values - mean) / denom)


def lobe_mismatch_metrics(
    relative_depth: np.ndarray,
    wall0: np.ndarray,
    wall1: np.ndarray,
    *,
    plateau_depth_cells: float = PLATEAU_DEPTH_CELLS,
) -> dict[str, float | bool | int]:
    x = np.asarray(relative_depth, dtype=float)
    a0 = np.asarray(wall0, dtype=float)
    a1 = np.asarray(wall1, dtype=float)
    if x.ndim != 1 or a0.ndim != 1 or a1.ndim != 1 or not (x.shape == a0.shape == a1.shape):
        raise ValueError("Stage 135 requires equal-length one-dimensional depth/profile arrays")
    if not np.isfinite(np.concatenate([x, a0, a1])).all():
        raise ValueError("Stage 135 lobe inputs must be finite")
    left = x <= -float(plateau_depth_cells)
    right = x >= float(plateau_depth_cells)
    if not left.any() or not right.any():
        raise ValueError("Stage 135 fixed lobe supports are empty")
    diff = a0 - a1
    dl = diff[left]
    dr = diff[right]
    left_l1 = float(np.sum(np.abs(dl)))
    right_l1 = float(np.sum(np.abs(dr)))
    total = left_l1 + right_l1
    return {
        "left_sample_count": int(dl.size),
        "right_sample_count": int(dr.size),
        "left_mean_shift_wall0_minus_wall1": float(np.mean(dl)),
        "right_mean_shift_wall0_minus_wall1": float(np.mean(dr)),
        "left_l1_mismatch": left_l1,
        "right_l1_mismatch": right_l1,
        "left_l1_share": left_l1 / total if total > 0.0 else 0.0,
        "right_l1_share": right_l1 / total if total > 0.0 else 0.0,
        "left_sign_coherence": _sign_coherence(dl),
        "right_sign_coherence": _sign_coherence(dr),
        "left_constant_shift_relative_l2_residual": _constant_shift_relative_l2(dl),
        "right_constant_shift_relative_l2_residual": _constant_shift_relative_l2(dr),
        "left_uniform_sign": bool(np.all(dl >= 0.0) or np.all(dl <= 0.0)),
        "right_uniform_sign": bool(np.all(dr >= 0.0) or np.all(dr <= 0.0)),
    }


def classify_lobe_origin(
    *,
    dominant: dict,
    parent: dict,
    finite: bool = True,
    parent_closure: float = 0.0,
    shift_closure: float = 0.0,
) -> str:
    numeric = []
    for block in (dominant, parent):
        numeric.extend(float(v) for v in block.values() if isinstance(v, (float, int)) and not isinstance(v, bool))
    numeric.extend([float(parent_closure), float(shift_closure)])
    if not finite or not np.isfinite(numeric).all():
        return NONFINITE
    if parent_closure > CLOSURE_MAX:
        return PARENT_CLOSURE_BLOCKER
    if shift_closure > CLOSURE_MAX:
        return SHIFT_CLOSURE_BLOCKER
    right = (
        min(float(dominant["right_l1_share"]), float(parent["right_l1_share"])) >= LOBE_L1_SHARE_MIN
        and min(float(dominant["right_sign_coherence"]), float(parent["right_sign_coherence"])) >= SIGN_COHERENCE_MIN
        and float(dominant["right_mean_shift_wall0_minus_wall1"]) < 0.0
        and float(parent["right_mean_shift_wall0_minus_wall1"]) < 0.0
    )
    if right:
        return RIGHT_LOBE
    left = (
        min(float(dominant["left_l1_share"]), float(parent["left_l1_share"])) >= LOBE_L1_SHARE_MIN
        and min(float(dominant["left_sign_coherence"]), float(parent["left_sign_coherence"])) >= SIGN_COHERENCE_MIN
        and float(dominant["left_mean_shift_wall0_minus_wall1"]) > 0.0
        and float(parent["left_mean_shift_wall0_minus_wall1"]) > 0.0
    )
    if left:
        return LEFT_LOBE
    return MIXED_LOBES


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_parent_records(stage132: dict, stage134: dict, record134: dict) -> None:
    assert stage132["stage"] == 132
    assert stage134["stage"] == 134
    assert stage134["decision"] == "stage134_opposite_side_lobe_imbalance_stage135_fixed_lobe_amplitude_origin_audit"
    assert record134["stage"] == 134
    assert record134["source_head"] == EXPECTED_STAGE134_SOURCE_HEAD
    assert record134["workflow_run_id"] == EXPECTED_STAGE134_RUN_ID
    assert record134["artifact_id"] == EXPECTED_STAGE134_ARTIFACT_ID
    assert record134["summary_sha256"] == EXPECTED_STAGE134_SUMMARY_SHA256
    assert record134["additive_baseline_origin_sha256"] == EXPECTED_STAGE134_PAYLOAD_SHA256
    assert stage134["parents"]["stage132_run_id"] == EXPECTED_STAGE132_RUN_ID
    assert stage134["parents"]["stage132_artifact_id"] == EXPECTED_STAGE132_ARTIFACT_ID


def run_stage135(stage132_dir: Path, stage134_dir: Path, stage134_record: Path, output_dir: Path) -> dict:
    validate_stage135_design()
    stage132 = _load_json(stage132_dir / "summary.json")
    stage134 = _load_json(stage134_dir / "summary.json")
    record134 = _load_json(stage134_record)
    _check_parent_records(stage132, stage134, record134)

    with np.load(stage132_dir / "crossing_phase_transition_width.npz") as data:
        x = np.asarray(data["relative_depth"], dtype=float)
        dominant_wall0 = np.asarray(data["dominant_wall0"], dtype=float)
        dominant_wall1 = np.asarray(data["dominant_wall1_mirrored"], dtype=float)
        parent_wall0 = np.asarray(data["parent_wall0"], dtype=float)
        parent_wall1 = np.asarray(data["parent_wall1"], dtype=float)

    dominant = lobe_mismatch_metrics(x, dominant_wall0, dominant_wall1)
    parent = lobe_mismatch_metrics(x, parent_wall0, parent_wall1)

    expected_shifts = np.array([
        stage134["metrics"]["dominant_sector"]["left_shift_wall0_minus_wall1"],
        stage134["metrics"]["dominant_sector"]["right_shift_wall0_minus_wall1"],
        stage134["metrics"]["parent_profile"]["left_shift_wall0_minus_wall1"],
        stage134["metrics"]["parent_profile"]["right_shift_wall0_minus_wall1"],
    ], dtype=float)
    measured_shifts = np.array([
        dominant["left_mean_shift_wall0_minus_wall1"],
        dominant["right_mean_shift_wall0_minus_wall1"],
        parent["left_mean_shift_wall0_minus_wall1"],
        parent["right_mean_shift_wall0_minus_wall1"],
    ], dtype=float)
    shift_closure = float(np.max(np.abs(measured_shifts - expected_shifts)))
    parent_closure = max(
        float(stage132["aggregate"]["maximum_parent_closure"]),
        float(stage134["aggregate"]["maximum_parent_closure"]),
    )
    numeric = np.concatenate([x, dominant_wall0, dominant_wall1, parent_wall0, parent_wall1])
    finite = bool(stage132["finite"] and stage134["finite"] and np.isfinite(numeric).all())
    decision = classify_lobe_origin(
        dominant=dominant,
        parent=parent,
        finite=finite,
        parent_closure=parent_closure,
        shift_closure=shift_closure,
    )

    cfg = dict(stage134["configuration"])
    cfg.update({
        "plateau_depth_cells": PLATEAU_DEPTH_CELLS,
        "lobe_l1_share_min": LOBE_L1_SHARE_MIN,
        "sign_coherence_min": SIGN_COHERENCE_MIN,
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

    if decision == RIGHT_LOBE:
        conclusion = (
            "The fixed Stage-132 cross-wall mismatch is carried predominantly by the right/negative lobe in both the dominant mirrored sector and the parent profile, with coherent sign across the full fixed tail support. "
            "Because the right-lobe constant-shift residual remains nonzero, this is not evidence for one depth-independent amplitude offset; the next justified artifact-only diagnostic is a right-lobe depth-support audit. "
            "This does not establish limiter causality, MUSCL stability, endpoint convergence, q_av improvement, benchmark accuracy, or validation."
        )
    elif decision == LEFT_LOBE:
        conclusion = (
            "The fixed Stage-132 cross-wall mismatch is carried predominantly by the left/positive lobe in both retained channels. A fixed left-lobe depth-support audit is justified next; no solver or parameter change is implied."
        )
    elif decision == MIXED_LOBES:
        conclusion = (
            "Neither fixed lobe carries a preregistered dominant share of the cross-wall mismatch in both retained channels. A paired-lobe residual audit is justified next; no solver or parameter change is implied."
        )
    else:
        conclusion = "Stage 135 is blocked by a finite-data or exact-parent closure guard; no scientific routing is claimed."

    summary = {
        "stage": STAGE,
        "finite": finite,
        "configuration": cfg,
        "parents": {
            "stage132_run_id": EXPECTED_STAGE132_RUN_ID,
            "stage132_artifact_id": EXPECTED_STAGE132_ARTIFACT_ID,
            "stage134_run_id": EXPECTED_STAGE134_RUN_ID,
            "stage134_artifact_id": EXPECTED_STAGE134_ARTIFACT_ID,
            "stage134_source_head": EXPECTED_STAGE134_SOURCE_HEAD,
        },
        "metrics": {
            "dominant_sector": dominant,
            "parent_profile": parent,
            "inherited_stage134_minimum_odd_fraction": float(stage134["aggregate"]["minimum_odd_fraction"]),
            "inherited_stage134_maximum_even_fraction": float(stage134["aggregate"]["maximum_even_fraction"]),
        },
        "aggregate": {
            "minimum_right_lobe_l1_share": min(float(dominant["right_l1_share"]), float(parent["right_l1_share"])),
            "minimum_right_lobe_sign_coherence": min(float(dominant["right_sign_coherence"]), float(parent["right_sign_coherence"])),
            "maximum_right_lobe_constant_shift_relative_l2_residual": max(float(dominant["right_constant_shift_relative_l2_residual"]), float(parent["right_constant_shift_relative_l2_residual"])),
            "maximum_stage134_lobe_shift_closure": shift_closure,
            "maximum_parent_closure": parent_closure,
        },
        "decision": decision,
        "scientific_conclusion": conclusion,
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. No physical, collision/source, floor, wall, reconstruction, transport, limiter, normalization, source-relaxation, or velocity-quadrature parameter is retuned; no solver endpoint or cross-Knudsen extension is advanced."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    diff_dom = dominant_wall0 - dominant_wall1
    diff_parent = parent_wall0 - parent_wall1
    left_mask = x <= -PLATEAU_DEPTH_CELLS
    right_mask = x >= PLATEAU_DEPTH_CELLS
    np.savez_compressed(
        output_dir / "fixed_lobe_amplitude_origin.npz",
        relative_depth=x,
        dominant_cross_wall_difference=diff_dom,
        parent_cross_wall_difference=diff_parent,
        left_lobe_mask=left_mask.astype(np.uint8),
        right_lobe_mask=right_mask.astype(np.uint8),
        dominant_lobe_mean_shifts=np.array([
            dominant["left_mean_shift_wall0_minus_wall1"],
            dominant["right_mean_shift_wall0_minus_wall1"],
        ], dtype=float),
        parent_lobe_mean_shifts=np.array([
            parent["left_mean_shift_wall0_minus_wall1"],
            parent["right_mean_shift_wall0_minus_wall1"],
        ], dtype=float),
        dominant_lobe_l1=np.array([dominant["left_l1_mismatch"], dominant["right_l1_mismatch"]], dtype=float),
        parent_lobe_l1=np.array([parent["left_l1_mismatch"], parent["right_l1_mismatch"]], dtype=float),
        dominant_lobe_sign_coherence=np.array([dominant["left_sign_coherence"], dominant["right_sign_coherence"]], dtype=float),
        parent_lobe_sign_coherence=np.array([parent["left_sign_coherence"], parent["right_sign_coherence"]], dtype=float),
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage132-dir", type=Path, required=True)
    p.add_argument("--stage134-dir", type=Path, required=True)
    p.add_argument("--stage134-record", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    summary = run_stage135(args.stage132_dir, args.stage134_dir, args.stage134_record, args.output_dir)
    print(json.dumps({"stage": summary["stage"], "decision": summary["decision"], "aggregate": summary["aggregate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
