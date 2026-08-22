from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = 142
EXPECTED_STAGE141_SOURCE_HEAD = "454e138842abec194cfc8b3eff0fda5390974cb8"
EXPECTED_STAGE141_RUN_ID = 32245156606
EXPECTED_STAGE141_JOB_ID = 96043973990
EXPECTED_STAGE141_ARTIFACT_ID = 9371343268
EXPECTED_STAGE141_ARTIFACT_SHA256 = "29bdee0c3c6e4e85496350e5ea683760abf83746193e4ef622ee3de241f099c2"
EXPECTED_STAGE141_SUMMARY_SHA256 = "17b00cb49902b727cc45510068f147d2864a953f0ec164cce4fd1fde686b3a45"
EXPECTED_STAGE141_PAYLOAD_SHA256 = "f9fc9181ef480e7d59017a06e08faaddd1711932b114e66822d9d7a7d4dab5e6"
EXPECTED_STAGE141_DECISION = "stage141_material_node_leverage_stage142_sampled_sign_margin_audit"

WEAK_ENDPOINT_RATIO_MAX = 0.25
MIN_POSITIVE_RUN_SAMPLES = 4
MIN_LATER_POSITIVE_SIGN_COHERENCE = 1.0
MAX_DELETION_COUNT = 3

NONFINITE = "stage142_nonfinite_blocker"
PARENT_RECORD_BLOCKER = "stage142_parent_record_blocker"
PARENT_ROUTE_BLOCKER = "stage142_parent_route_blocker"
INSUFFICIENT_SUPPORT = "stage142_insufficient_sample_support_blocker"
PERSISTENT_WEAK_MARGIN = "stage142_persistent_sampled_sign_with_weak_endpoint_margin_stage143_positive_lobe_scale_audit"
PERSISTENT_STRONG_MARGIN = "stage142_persistent_sampled_sign_with_strong_endpoint_margin_stage143_signed_lobe_balance_audit"
MIXED_SUPPORT = "stage142_mixed_sampled_sign_support_stage143_transition_support_audit"


def validate_stage142_design(
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
    weak_endpoint_ratio_max=WEAK_ENDPOINT_RATIO_MAX,
    min_positive_run_samples=MIN_POSITIVE_RUN_SAMPLES,
    min_later_positive_sign_coherence=MIN_LATER_POSITIVE_SIGN_COHERENCE,
    max_deletion_count=MAX_DELETION_COUNT,
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
        "weak_endpoint_ratio_max": WEAK_ENDPOINT_RATIO_MAX,
        "min_positive_run_samples": MIN_POSITIVE_RUN_SAMPLES,
        "min_later_positive_sign_coherence": MIN_LATER_POSITIVE_SIGN_COHERENCE,
        "max_deletion_count": MAX_DELETION_COUNT,
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
            raise ValueError(f"Stage 142 frozen-design violation: {key}={got[key]!r}, expected {value!r}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_stage141_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 141
        and record.get("source_head") == EXPECTED_STAGE141_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE141_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE141_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE141_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE141_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE141_SUMMARY_SHA256
        and record.get("node_leverage_sha256") == EXPECTED_STAGE141_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE141_DECISION
        and record.get("workflow_status") == "completed"
        and record.get("workflow_conclusion") == "success"
    )


def positive_run_length(values: np.ndarray) -> int:
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError("Stage 142 positive-run values must be a finite one-dimensional array")
    count = 0
    for value in y:
        if value > 0.0:
            count += 1
        else:
            break
    return count


def classify_sampled_sign_margin(
    *,
    positive_run_length_samples: int,
    later_positive_sign_coherence: float,
    upper_to_later_positive_median_ratio: float,
    deletion_sign_retention_fraction: float,
    parent_record_ok: bool = True,
    parent_route_ok: bool = True,
    sufficient_support: bool = True,
    finite: bool = True,
) -> str:
    numeric = [
        later_positive_sign_coherence,
        upper_to_later_positive_median_ratio,
        deletion_sign_retention_fraction,
    ]
    if not finite or not np.isfinite(numeric).all():
        return NONFINITE
    if not parent_record_ok:
        return PARENT_RECORD_BLOCKER
    if not parent_route_ok:
        return PARENT_ROUTE_BLOCKER
    if not sufficient_support:
        return INSUFFICIENT_SUPPORT
    persistent = (
        positive_run_length_samples >= MIN_POSITIVE_RUN_SAMPLES
        and later_positive_sign_coherence >= MIN_LATER_POSITIVE_SIGN_COHERENCE
        and deletion_sign_retention_fraction == 1.0
    )
    if persistent and upper_to_later_positive_median_ratio <= WEAK_ENDPOINT_RATIO_MAX:
        return PERSISTENT_WEAK_MARGIN
    if persistent:
        return PERSISTENT_STRONG_MARGIN
    return MIXED_SUPPORT


def run_stage142(stage141_dir: Path, stage141_record: Path, output_dir: Path) -> dict:
    validate_stage142_design()
    parent_summary = _load_json(stage141_dir / "summary.json")
    parent_record = _load_json(stage141_record)
    parent_record_ok = _check_stage141_record(parent_record)
    parent_route_ok = bool(parent_summary.get("stage") == 141 and parent_summary.get("decision") == EXPECTED_STAGE141_DECISION)

    with np.load(stage141_dir / "node_leverage.npz") as data:
        depth = np.asarray(data["right_depth"], dtype=float)
        complement = np.asarray(data["complement_signed"], dtype=float)
        bracket = np.asarray(data["crossing_bracket_indices"], dtype=int)

    finite = bool(parent_summary.get("finite", False) and np.isfinite(np.concatenate([depth, complement])).all())
    sufficient_support = bool(
        depth.ndim == 1
        and complement.ndim == 1
        and depth.shape == complement.shape
        and depth.size >= 7
        and bracket.shape == (2,)
        and np.all(np.diff(depth) > 0.0)
    )
    if sufficient_support:
        lower_index = int(bracket[0])
        upper_index = int(bracket[1])
        sufficient_support = bool(
            upper_index == lower_index + 1
            and lower_index >= 0
            and upper_index + MAX_DELETION_COUNT < depth.size
            and complement[lower_index] < 0.0
            and complement[upper_index] > 0.0
        )
    else:
        lower_index, upper_index = 0, 1

    if not sufficient_support:
        positive_side = np.asarray([], dtype=float)
        later_positive = np.asarray([], dtype=float)
        positive_run = 0
        later_coherence = 0.0
        upper_to_median = float("nan")
        min_later_to_upper = float("nan")
        upper_to_lower = float("nan")
        upper_l1_fraction = float("nan")
        deletion_counts = np.arange(1, MAX_DELETION_COUNT + 1, dtype=int)
        deletion_retained = np.zeros(MAX_DELETION_COUNT, dtype=bool)
        deletion_widths = np.full(MAX_DELETION_COUNT, np.nan)
    else:
        lower = float(complement[lower_index])
        upper = float(complement[upper_index])
        positive_side = np.asarray(complement[upper_index:], dtype=float)
        later_positive = np.asarray(complement[upper_index + 1 :], dtype=float)
        positive_run = positive_run_length(positive_side)
        later_coherence = float(np.mean(later_positive > 0.0))
        later_median_abs = float(np.median(np.abs(later_positive)))
        upper_to_median = float(abs(upper) / later_median_abs)
        min_later_to_upper = float(np.min(np.abs(later_positive)) / abs(upper))
        upper_to_lower = float(abs(upper) / abs(lower))
        upper_l1_fraction = float(abs(upper) / np.sum(np.abs(positive_side)))
        deletion_counts = np.arange(1, MAX_DELETION_COUNT + 1, dtype=int)
        deletion_retained = np.asarray(
            [lower * float(complement[upper_index + count]) < 0.0 for count in deletion_counts],
            dtype=bool,
        )
        deletion_widths = np.asarray(
            [float(depth[upper_index + count] - depth[lower_index]) for count in deletion_counts],
            dtype=float,
        )

    deletion_fraction = float(np.mean(deletion_retained)) if deletion_retained.size else 0.0
    positive_span = float(depth[upper_index + positive_run - 1] - depth[upper_index]) if sufficient_support and positive_run else 0.0

    decision = classify_sampled_sign_margin(
        positive_run_length_samples=positive_run,
        later_positive_sign_coherence=later_coherence,
        upper_to_later_positive_median_ratio=upper_to_median,
        deletion_sign_retention_fraction=deletion_fraction,
        parent_record_ok=parent_record_ok,
        parent_route_ok=parent_route_ok,
        sufficient_support=sufficient_support,
        finite=finite,
    )

    cfg = dict(parent_summary.get("configuration", {}))
    cfg.update({
        "weak_endpoint_ratio_max": WEAK_ENDPOINT_RATIO_MAX,
        "min_positive_run_samples": MIN_POSITIVE_RUN_SAMPLES,
        "min_later_positive_sign_coherence": MIN_LATER_POSITIVE_SIGN_COHERENCE,
        "max_deletion_count": MAX_DELETION_COUNT,
        "sampled_sign_margin_used_for_solver": False,
        "counterfactual_root_used_for_solver": False,
        "solver_rerun": False,
        "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False,
        "benchmark_or_validation_claim_permitted": False,
    })
    for key in list(cfg):
        if key.endswith("_retuning"):
            cfg[key] = False

    if decision == PERSISTENT_WEAK_MARGIN:
        conclusion = (
            "The first positive sampled node remains a weak-amplitude endpoint, but the positive sign persists across the entire retained "
            "positive-side sample run and remains present after deleting each of the first three positive samples in turn. Thus the existence "
            "of a sampled positive lobe is independently supported by neighboring samples even though the precise sub-cell crossing remains "
            "node-leveraged. The next justified artifact-only stage is a positive-lobe scale audit; no sign margin or root is fed back into the solver."
        )
    elif decision == PERSISTENT_STRONG_MARGIN:
        conclusion = (
            "The positive-side sign is persistent and the first positive endpoint is not weak relative to the later positive-side median. "
            "The next justified artifact-only stage is a signed-lobe balance audit; no sampled margin is used as a solver parameter."
        )
    else:
        conclusion = (
            "The fixed sampled-sign diagnostics do not give uniformly persistent positive-side support. The next justified artifact-only stage "
            "is a transition-support audit; no interpolation, deletion bridge, or sign margin is used as a solver parameter."
        )

    summary = {
        "stage": STAGE,
        "finite": bool(finite),
        "decision": decision,
        "configuration": cfg,
        "parents": {
            "stage141_run_id": EXPECTED_STAGE141_RUN_ID,
            "stage141_job_id": EXPECTED_STAGE141_JOB_ID,
            "stage141_artifact_id": EXPECTED_STAGE141_ARTIFACT_ID,
            "stage141_source_head": EXPECTED_STAGE141_SOURCE_HEAD,
        },
        "aggregate": {
            "parent_record_ok": bool(parent_record_ok),
            "parent_route_ok": bool(parent_route_ok),
            "sufficient_support": bool(sufficient_support),
            "positive_run_is_persistent": bool(positive_run >= MIN_POSITIVE_RUN_SAMPLES),
            "later_positive_sign_coherence_pass": bool(later_coherence >= MIN_LATER_POSITIVE_SIGN_COHERENCE),
            "deletion_sign_retention_fraction": deletion_fraction,
            "weak_endpoint_margin": bool(np.isfinite(upper_to_median) and upper_to_median <= WEAK_ENDPOINT_RATIO_MAX),
        },
        "metrics": {
            "positive_run_length_samples": int(positive_run),
            "positive_run_span_cells": positive_span,
            "later_positive_count": int(later_positive.size),
            "later_positive_sign_coherence": later_coherence,
            "upper_endpoint_abs_ratio_to_lower": upper_to_lower,
            "upper_to_later_positive_median_ratio": upper_to_median,
            "minimum_later_positive_to_upper_ratio": min_later_to_upper,
            "upper_fraction_of_positive_side_l1": upper_l1_fraction,
            "deletion_sign_retention_fraction": deletion_fraction,
            "maximum_retained_deletion_bracket_width_cells": float(np.max(deletion_widths)) if deletion_widths.size and np.isfinite(deletion_widths).any() else float("nan"),
        },
        "scientific_conclusion": conclusion,
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. Stage 142 is an artifact-only sampled sign-margin audit; "
            "sampled margins and deletion widths are diagnostics, not solver parameters. No physical, collision/source, floor, wall, reconstruction, "
            "transport, limiter, normalization, source-relaxation, or velocity-quadrature parameter is retuned; no solver endpoint or cross-Knudsen "
            "extension is advanced."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        output_dir / "sampled_sign_margin.npz",
        right_depth=depth,
        complement_signed=complement,
        positive_side_depth=depth[upper_index:] if sufficient_support else np.asarray([], dtype=float),
        positive_side_values=positive_side,
        later_positive_values=later_positive,
        deletion_counts=deletion_counts,
        deletion_sign_retained=deletion_retained,
        deletion_bracket_widths_cells=deletion_widths,
    )
    print(json.dumps({"stage": STAGE, "decision": decision, "aggregate": summary["aggregate"], "metrics": summary["metrics"]}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 142 sampled sign-margin audit")
    parser.add_argument("--stage141-dir", type=Path, required=True)
    parser.add_argument("--stage141-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_stage142(args.stage141_dir, args.stage141_record, args.output_dir)


if __name__ == "__main__":
    main()
