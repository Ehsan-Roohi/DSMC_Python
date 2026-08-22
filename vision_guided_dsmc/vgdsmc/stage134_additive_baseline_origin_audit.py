from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = 134
EXPECTED_STAGE133_SOURCE_HEAD = "8d428d1b09d8d9905470d3bacb99ea328d74b81a"
EXPECTED_STAGE133_RUN_ID = 32019594478
EXPECTED_STAGE133_ARTIFACT_ID = 9291710769
EXPECTED_STAGE133_SUMMARY_SHA256 = "6746241f4fc9c89de9cf833327d8d4c4f3130e085285856e7bc995e3570d2e"
EXPECTED_STAGE133_PAYLOAD_SHA256 = "f66c67f3335b6a3267ce117d198a9f13049a1d8f0fa41682ac5fa68fa41b3f62"
EXPECTED_STAGE132_RUN_ID = 32001930265
EXPECTED_STAGE132_ARTIFACT_ID = 9284552744
PLATEAU_DEPTH_CELLS = 5.0
ODD_DOMINANCE_MIN = 0.5
GLOBAL_BASELINE_FRACTION_MIN = 0.75
GLOBAL_OFFSET_RELATIVE_MISMATCH_MAX = 0.5
CLOSURE_MAX = 1.0e-12

NONFINITE = "stage134_nonfinite_blocker"
CLOSURE_BLOCKER = "stage134_parent_closure_blocker"
GLOBAL_BASELINE = "stage134_global_additive_baseline_stage135_baseline_support_audit"
LOBE_IMBALANCE = "stage134_opposite_side_lobe_imbalance_stage135_fixed_lobe_amplitude_origin_audit"
MIXED = "stage134_mixed_baseline_lobe_origin_stage135_crossing_residual_component_audit"


def validate_stage134_design(
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
            raise ValueError(f"Stage 134 frozen-design violation: {key}={got[key]!r}, expected {value!r}")


def decompose_plateau_shift(left_shift: float, right_shift: float) -> dict[str, float | bool]:
    left_shift = float(left_shift)
    right_shift = float(right_shift)
    even = 0.5 * (left_shift + right_shift)
    odd = 0.5 * (left_shift - right_shift)
    denom = abs(even) + abs(odd)
    return {
        "left_shift_wall0_minus_wall1": left_shift,
        "right_shift_wall0_minus_wall1": right_shift,
        "even_global_baseline_component": even,
        "odd_lobe_imbalance_component": odd,
        "even_fraction": abs(even) / denom if denom > 0.0 else 0.0,
        "odd_fraction": abs(odd) / denom if denom > 0.0 else 0.0,
        "opposite_side_signs": bool(left_shift * right_shift < 0.0),
    }


def classify_baseline_origin(
    *, dominant_left_shift: float, dominant_right_shift: float,
    parent_left_shift: float, parent_right_shift: float,
    dominant_even_fraction: float, parent_even_fraction: float,
    dominant_odd_fraction: float, parent_odd_fraction: float,
    dominant_offset_even_relative_mismatch: float,
    parent_offset_even_relative_mismatch: float,
    finite: bool = True, closure: float = 0.0,
) -> str:
    vals = np.asarray([
        dominant_left_shift, dominant_right_shift, parent_left_shift, parent_right_shift,
        dominant_even_fraction, parent_even_fraction, dominant_odd_fraction, parent_odd_fraction,
        dominant_offset_even_relative_mismatch, parent_offset_even_relative_mismatch, closure,
    ], dtype=float)
    if not finite or not np.isfinite(vals).all():
        return NONFINITE
    if closure > CLOSURE_MAX:
        return CLOSURE_BLOCKER
    both_opposite = (dominant_left_shift * dominant_right_shift < 0.0 and parent_left_shift * parent_right_shift < 0.0)
    if both_opposite and min(dominant_odd_fraction, parent_odd_fraction) >= ODD_DOMINANCE_MIN:
        return LOBE_IMBALANCE
    no_opposite = (dominant_left_shift * dominant_right_shift >= 0.0 and parent_left_shift * parent_right_shift >= 0.0)
    if (no_opposite
            and min(dominant_even_fraction, parent_even_fraction) >= GLOBAL_BASELINE_FRACTION_MIN
            and max(dominant_offset_even_relative_mismatch, parent_offset_even_relative_mismatch)
            <= GLOBAL_OFFSET_RELATIVE_MISMATCH_MAX):
        return GLOBAL_BASELINE
    return MIXED


def _relative_mismatch(a: float, b: float) -> float:
    scale = max(abs(float(a)), abs(float(b)), np.finfo(float).tiny)
    return abs(float(a) - float(b)) / scale


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_parent_records(stage132: dict, stage133: dict, record133: dict) -> None:
    assert stage132["stage"] == 132
    assert stage133["stage"] == 133
    assert stage133["decision"] == "stage133_affine_offset_dominated_crossing_mismatch_stage134_additive_baseline_origin_audit"
    assert record133["stage"] == 133
    assert record133["source_head"] == EXPECTED_STAGE133_SOURCE_HEAD
    assert record133["workflow_run_id"] == EXPECTED_STAGE133_RUN_ID
    assert record133["artifact_id"] == EXPECTED_STAGE133_ARTIFACT_ID
    assert record133["summary_sha256"] == EXPECTED_STAGE133_SUMMARY_SHA256
    assert record133["crossing_amplitude_scale_sha256"] == EXPECTED_STAGE133_PAYLOAD_SHA256
    assert stage133["parents"]["stage132_run_id"] == EXPECTED_STAGE132_RUN_ID
    assert stage133["parents"]["stage132_artifact_id"] == EXPECTED_STAGE132_ARTIFACT_ID


def run_stage134(stage132_dir: Path, stage133_dir: Path, stage133_record: Path, output_dir: Path) -> dict:
    validate_stage134_design()
    stage132 = _load_json(stage132_dir / "summary.json")
    stage133 = _load_json(stage133_dir / "summary.json")
    record133 = _load_json(stage133_record)
    _check_parent_records(stage132, stage133, record133)

    m132 = stage132["metrics"]
    dominant_left = m132["dominant_sector_wall0"]["left_plateau"] - m132["dominant_sector_wall1_mirrored"]["left_plateau"]
    dominant_right = m132["dominant_sector_wall0"]["right_plateau"] - m132["dominant_sector_wall1_mirrored"]["right_plateau"]
    parent_left = m132["parent_wall0"]["left_plateau"] - m132["parent_wall1"]["left_plateau"]
    parent_right = m132["parent_wall0"]["right_plateau"] - m132["parent_wall1"]["right_plateau"]
    dominant = decompose_plateau_shift(dominant_left, dominant_right)
    parent = decompose_plateau_shift(parent_left, parent_right)

    dominant_offset = stage133["metrics"]["dominant_sector"]["affine_offset_wall0_units"]
    parent_offset = stage133["metrics"]["parent_profile"]["affine_offset_wall0_units"]
    for block, offset in ((dominant, dominant_offset), (parent, parent_offset)):
        even = float(block["even_global_baseline_component"])
        block["stage133_affine_offset_wall0_units"] = float(offset)
        block["stage133_offset_to_even_ratio"] = abs(float(offset)) / max(abs(even), np.finfo(float).tiny)
        block["stage133_offset_even_relative_mismatch"] = _relative_mismatch(offset, even)

    closure = max(float(stage132["aggregate"]["maximum_parent_closure"]), float(stage133["aggregate"]["maximum_parent_closure"]))
    numeric = []
    for block in (dominant, parent):
        numeric.extend(float(v) for v in block.values() if isinstance(v, (float, int)) and not isinstance(v, bool))
    finite = bool(stage132["finite"] and stage133["finite"] and np.isfinite(numeric).all())
    decision = classify_baseline_origin(
        dominant_left_shift=dominant_left, dominant_right_shift=dominant_right,
        parent_left_shift=parent_left, parent_right_shift=parent_right,
        dominant_even_fraction=float(dominant["even_fraction"]), parent_even_fraction=float(parent["even_fraction"]),
        dominant_odd_fraction=float(dominant["odd_fraction"]), parent_odd_fraction=float(parent["odd_fraction"]),
        dominant_offset_even_relative_mismatch=float(dominant["stage133_offset_even_relative_mismatch"]),
        parent_offset_even_relative_mismatch=float(parent["stage133_offset_even_relative_mismatch"]),
        finite=finite, closure=closure,
    )

    cfg = dict(stage133["configuration"])
    cfg.update({
        "plateau_depth_cells": PLATEAU_DEPTH_CELLS,
        "odd_dominance_min": ODD_DOMINANCE_MIN,
        "global_baseline_fraction_min": GLOBAL_BASELINE_FRACTION_MIN,
        "global_offset_relative_mismatch_max": GLOBAL_OFFSET_RELATIVE_MISMATCH_MAX,
        "phase_shift_applied": False, "width_refit_applied": False, "amplitude_refit_applied": False,
        "solver_rerun": False, "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False, "benchmark_or_validation_claim_permitted": False,
    })
    for key in list(cfg):
        if key.endswith("_retuning"):
            cfg[key] = False

    if decision == LOBE_IMBALANCE:
        conclusion = ("The Stage-133 affine offset is not supported as one uniform cross-wall additive baseline. "
                      "For both the dominant mirrored sector and the parent asymmetry, the fixed Stage-132 left and right plateau shifts have opposite signs, and the odd left/right lobe-imbalance component exceeds the even global-baseline component. "
                      "The next justified fixed diagnostic is therefore a lobe-amplitude-origin audit. This artifact-only attribution does not establish limiter causality, MUSCL stability, endpoint convergence, q_av improvement, benchmark accuracy, or validation.")
    elif decision == GLOBAL_BASELINE:
        conclusion = "The fixed plateau shifts are consistent with a common additive baseline on both sides of the crossing. A fixed baseline-support audit is justified next; no solver or parameter change is implied."
    elif decision == MIXED:
        conclusion = "The fixed plateau shifts contain mixed global-baseline and lobe-imbalance structure, so neither origin alone is sufficient. A fixed residual-component audit is justified next; no solver or parameter change is implied."
    else:
        conclusion = "Stage 134 is blocked by a finite-data or parent-closure guard; no scientific routing is claimed."

    summary = {
        "stage": STAGE, "finite": finite, "configuration": cfg,
        "parents": {
            "stage132_run_id": EXPECTED_STAGE132_RUN_ID, "stage132_artifact_id": EXPECTED_STAGE132_ARTIFACT_ID,
            "stage133_run_id": EXPECTED_STAGE133_RUN_ID, "stage133_artifact_id": EXPECTED_STAGE133_ARTIFACT_ID,
            "stage133_source_head": EXPECTED_STAGE133_SOURCE_HEAD,
        },
        "metrics": {
            "dominant_sector": dominant, "parent_profile": parent,
            "inherited_stage132_maximum_phase_offset_cells": float(stage132["aggregate"]["maximum_phase_offset_cells"]),
            "inherited_stage132_maximum_transition_width_ratio": float(stage132["aggregate"]["maximum_transition_width_ratio"]),
            "inherited_stage133_minimum_affine_gain_fraction": float(stage133["aggregate"]["minimum_affine_gain_fraction"]),
        },
        "aggregate": {
            "minimum_odd_fraction": min(float(dominant["odd_fraction"]), float(parent["odd_fraction"])),
            "maximum_even_fraction": max(float(dominant["even_fraction"]), float(parent["even_fraction"])),
            "minimum_stage133_offset_to_plateau_even_ratio": min(float(dominant["stage133_offset_to_even_ratio"]), float(parent["stage133_offset_to_even_ratio"])),
            "maximum_parent_closure": closure,
        },
        "decision": decision, "scientific_conclusion": conclusion,
        "negative_result_guard": ("Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. No physical, collision/source, floor, wall, reconstruction, transport, limiter, normalization, source-relaxation, or velocity-quadrature parameter is retuned; no solver endpoint or cross-Knudsen extension is advanced."),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        output_dir / "additive_baseline_origin.npz",
        dominant_plateau_shifts=np.array([dominant_left, dominant_right], dtype=float),
        parent_plateau_shifts=np.array([parent_left, parent_right], dtype=float),
        dominant_even_odd=np.array([dominant["even_global_baseline_component"], dominant["odd_lobe_imbalance_component"]], dtype=float),
        parent_even_odd=np.array([parent["even_global_baseline_component"], parent["odd_lobe_imbalance_component"]], dtype=float),
        stage133_affine_offsets=np.array([dominant_offset, parent_offset], dtype=float),
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage132-dir", type=Path, required=True)
    p.add_argument("--stage133-dir", type=Path, required=True)
    p.add_argument("--stage133-record", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    summary = run_stage134(args.stage132_dir, args.stage133_dir, args.stage133_record, args.output_dir)
    print(json.dumps({"stage": summary["stage"], "decision": summary["decision"], "aggregate": summary["aggregate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
