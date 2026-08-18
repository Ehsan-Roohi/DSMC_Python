from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = 138
EXPECTED_STAGE136_SOURCE_HEAD = "a498faa95cecf9712b8401f389e6776f02e4aa0f"
EXPECTED_STAGE136_RUN_ID = 32140741740
EXPECTED_STAGE136_JOB_ID = 95722525504
EXPECTED_STAGE136_ARTIFACT_ID = 9325758887
EXPECTED_STAGE136_ARTIFACT_SHA256 = "176d1cb7e6b45ba7e780d1cd09254ee4c0aa27dab086eca1de92e2b891e5f0b0"
EXPECTED_STAGE136_SUMMARY_SHA256 = "eaf3706b2c7bca1678b0680dcf06daa076ccd36eeb91dccb9b8602bd8378b64c"
EXPECTED_STAGE136_PAYLOAD_SHA256 = "f8829d07450f3832dc45a878943ac69415390e89208f9bb859e838fa9c160015"

EXPECTED_STAGE137_SOURCE_HEAD = "9699d9096c60019a8faaffc2e7404a6a03628b17"
EXPECTED_STAGE137_RUN_ID = 32141400728
EXPECTED_STAGE137_JOB_ID = 95724651080
EXPECTED_STAGE137_ARTIFACT_ID = 9338389400
EXPECTED_STAGE137_ARTIFACT_SHA256 = "8c3264d8395f5adad390b3dbf35cb4cf70f421ad8c38ca63c1cb1363d3b9bf7d"
EXPECTED_STAGE137_SUMMARY_SHA256 = "1a655f9761d021def68996d4848508f7b035db82b584d11d3464eaa455a5a5f5"
EXPECTED_STAGE137_PAYLOAD_SHA256 = "05a8119eaf668aee355194428840596baacc79a79eb88446351bc2143ad5d3b6"
EXPECTED_STAGE137_DECISION = "stage137_smooth_channel_rate_split_stage138_channel_rate_origin_audit"

RATE_IDENTITY_CLOSURE_MAX = 1.0e-12
PARENT_PROFILE_CLOSURE_MAX = 1.0e-12
RATE_SPLIT_EXPLAINED_FRACTION_MIN = 0.95
RATIO_SWING_MIN = 0.25
ENDPOINT_CANCELLATION_FRACTION_MIN = 0.25

NONFINITE = "stage138_nonfinite_blocker"
PARENT_RECORD_BLOCKER = "stage138_parent_record_blocker"
PARENT_PROFILE_BLOCKER = "stage138_parent_profile_closure_blocker"
DEPTH_VARYING_CANCELLATION = "stage138_depth_varying_complement_cancellation_stage139_complement_transition_geometry_audit"
SAME_SIGN_MIXTURE = "stage138_same_sign_mixture_rate_split_stage139_mixture_weight_gradient_audit"
UNRESOLVED_RATE_ORIGIN = "stage138_unresolved_rate_origin_stage139_signed_channel_residual_audit"


def validate_stage138_design(
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
    rate_identity_closure_max=RATE_IDENTITY_CLOSURE_MAX,
    rate_split_explained_fraction_min=RATE_SPLIT_EXPLAINED_FRACTION_MIN,
    ratio_swing_min=RATIO_SWING_MIN,
    endpoint_cancellation_fraction_min=ENDPOINT_CANCELLATION_FRACTION_MIN,
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
        "rate_identity_closure_max": RATE_IDENTITY_CLOSURE_MAX,
        "rate_split_explained_fraction_min": RATE_SPLIT_EXPLAINED_FRACTION_MIN,
        "ratio_swing_min": RATIO_SWING_MIN,
        "endpoint_cancellation_fraction_min": ENDPOINT_CANCELLATION_FRACTION_MIN,
        "solver_rerun": False,
        "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False,
        "physical_parameter_retuning": False,
    }
    got = locals()
    for key, value in expected.items():
        if got[key] != value:
            raise ValueError(f"Stage 138 frozen-design violation: {key}={got[key]!r}, expected {value!r}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_stage137_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 137
        and record.get("source_head") == EXPECTED_STAGE137_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE137_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE137_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE137_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE137_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE137_SUMMARY_SHA256
        and record.get("right_lobe_decay_shape_sha256") == EXPECTED_STAGE137_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE137_DECISION
    )


def fitted_decay_rate(depth: np.ndarray, magnitude: np.ndarray) -> float:
    x = np.asarray(depth, dtype=float)
    y = np.asarray(magnitude, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 3:
        raise ValueError("Stage 138 decay-rate fit requires equal one-dimensional arrays with at least three samples")
    if not np.isfinite(np.concatenate([x, y])).all() or np.any(y <= 0.0) or np.any(np.diff(x) <= 0.0):
        raise ValueError("Stage 138 decay-rate inputs must be finite, positive in magnitude, and ordered in depth")
    z = x - x[0]
    design = np.column_stack([np.ones_like(z), z])
    slope = float(np.linalg.lstsq(design, np.log(y), rcond=None)[0][1])
    return -slope


def first_sign_change_depth(depth: np.ndarray, values: np.ndarray) -> float | None:
    x = np.asarray(depth, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("Stage 138 sign-change audit requires equal one-dimensional arrays")
    for i in range(y.size - 1):
        if y[i] == 0.0:
            return float(x[i])
        if y[i] * y[i + 1] < 0.0:
            return float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    if y[-1] == 0.0:
        return float(x[-1])
    return None


def pointwise_cancellation_fraction(dominant: np.ndarray, complement: np.ndarray) -> np.ndarray:
    d = np.asarray(dominant, dtype=float)
    c = np.asarray(complement, dtype=float)
    if d.shape != c.shape:
        raise ValueError("Stage 138 cancellation inputs must have equal shape")
    denom = np.abs(d) + np.abs(c)
    parent = d + c
    out = np.zeros_like(denom)
    mask = denom > 0.0
    out[mask] = 1.0 - np.abs(parent[mask]) / denom[mask]
    return out


def classify_rate_origin(
    *,
    sign_change_count: int,
    ratio_swing: float,
    endpoint_cancellation_fraction: float,
    rate_identity_closure: float,
    rate_split_explained_fraction: float,
    finite: bool = True,
    parent_record_ok: bool = True,
    parent_profile_closure: float = 0.0,
) -> str:
    numeric = [ratio_swing, endpoint_cancellation_fraction, rate_identity_closure, rate_split_explained_fraction, parent_profile_closure]
    if not finite or not np.isfinite(numeric).all():
        return NONFINITE
    if not parent_record_ok:
        return PARENT_RECORD_BLOCKER
    if parent_profile_closure > PARENT_PROFILE_CLOSURE_MAX:
        return PARENT_PROFILE_BLOCKER
    if rate_identity_closure > RATE_IDENTITY_CLOSURE_MAX or rate_split_explained_fraction < RATE_SPLIT_EXPLAINED_FRACTION_MIN:
        return UNRESOLVED_RATE_ORIGIN
    if sign_change_count >= 1 and ratio_swing >= RATIO_SWING_MIN and endpoint_cancellation_fraction >= ENDPOINT_CANCELLATION_FRACTION_MIN:
        return DEPTH_VARYING_CANCELLATION
    return SAME_SIGN_MIXTURE


def run_stage138(stage136_dir: Path, stage137_dir: Path, stage137_record: Path, output_dir: Path) -> dict:
    validate_stage138_design()
    stage136_summary = _load_json(stage136_dir / "summary.json")
    stage137_summary = _load_json(stage137_dir / "summary.json")
    stage137_observed = _load_json(stage137_record)
    record_ok = _check_stage137_record(stage137_observed)
    if stage136_summary.get("stage") != 136:
        raise ValueError("Stage 138 requires the exact completed Stage 136 signed profiles")
    if stage137_summary.get("stage") != 137 or stage137_summary.get("decision") != EXPECTED_STAGE137_DECISION:
        raise ValueError("Stage 138 requires the completed Stage 137 smooth channel-rate-split route")

    with np.load(stage136_dir / "right_lobe_depth_support.npz") as d136:
        depth = np.asarray(d136["right_depth"], dtype=float)
        dominant = np.asarray(d136["dominant_right_difference"], dtype=float)
        parent = np.asarray(d136["parent_right_difference"], dtype=float)
    with np.load(stage137_dir / "right_lobe_decay_shape.npz") as d137:
        stage137_depth = np.asarray(d137["right_depth"], dtype=float)
        stage137_dominant_magnitude = np.asarray(d137["dominant_magnitude"], dtype=float)
        stage137_parent_magnitude = np.asarray(d137["parent_magnitude"], dtype=float)

    parent_profile_closure = max(
        float(np.max(np.abs(depth - stage137_depth))),
        float(np.max(np.abs(np.abs(dominant) - stage137_dominant_magnitude))),
        float(np.max(np.abs(np.abs(parent) - stage137_parent_magnitude))),
    )
    complement = parent - dominant
    if np.any(dominant == 0.0):
        raise ValueError("Stage 138 dominant signed profile contains a zero sample")
    ratio = complement / dominant
    mixing_factor = parent / dominant
    if np.any(mixing_factor <= 0.0):
        raise ValueError("Stage 138 requires parent and dominant channels to retain a common sign")

    kd = fitted_decay_rate(depth, np.abs(dominant))
    kp = fitted_decay_rate(depth, np.abs(parent))
    kmix = fitted_decay_rate(depth, mixing_factor)
    split = kp - kd
    rate_identity_closure = abs(kp - (kd + kmix))
    rate_split_explained_fraction = abs(kmix) / abs(split) if abs(split) > 0.0 else 1.0

    signs = np.sign(complement)
    sign_change_count = int(np.sum(signs[:-1] * signs[1:] < 0.0))
    crossing_depth = first_sign_change_depth(depth, complement)
    cancellation = pointwise_cancellation_fraction(dominant, complement)
    ratio_swing = float(np.max(ratio) - np.min(ratio))
    endpoint_cancellation = float(cancellation[-1])
    finite = bool(
        stage136_summary.get("finite", False)
        and stage137_summary.get("finite", False)
        and np.isfinite(np.concatenate([depth, dominant, parent, complement, ratio, mixing_factor, cancellation])).all()
    )

    decision = classify_rate_origin(
        sign_change_count=sign_change_count,
        ratio_swing=ratio_swing,
        endpoint_cancellation_fraction=endpoint_cancellation,
        rate_identity_closure=rate_identity_closure,
        rate_split_explained_fraction=rate_split_explained_fraction,
        finite=finite,
        parent_record_ok=record_ok,
        parent_profile_closure=parent_profile_closure,
    )

    z = depth - depth[0]
    design = np.column_stack([np.ones_like(z), z])
    coef = np.linalg.lstsq(design, np.log(mixing_factor), rcond=None)[0]
    mixing_fit = np.exp(design @ coef)

    cfg = dict(stage137_summary["configuration"])
    cfg.update({
        "rate_identity_closure_max": RATE_IDENTITY_CLOSURE_MAX,
        "rate_split_explained_fraction_min": RATE_SPLIT_EXPLAINED_FRACTION_MIN,
        "ratio_swing_min": RATIO_SWING_MIN,
        "endpoint_cancellation_fraction_min": ENDPOINT_CANCELLATION_FRACTION_MIN,
        "signed_complement_used_for_solver": False,
        "diagnostic_decay_fit_used_for_solver": False,
        "solver_rerun": False,
        "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False,
        "benchmark_or_validation_claim_permitted": False,
    })
    for key in list(cfg):
        if key.endswith("_retuning"):
            cfg[key] = False

    if decision == DEPTH_VARYING_CANCELLATION:
        conclusion = (
            "The Stage-137 decay-rate split is explained by a depth-varying signed complementary channel rather than by a second fitted solver coefficient. "
            "The complement reinforces the dominant negative channel near the crossing, changes sign, and opposes it farther into the right lobe; the fitted-rate identity k_parent = k_dominant + k_mixing closes to numerical precision. "
            "The next justified artifact-only diagnostic is to localize the complement sign transition geometrically. This is decomposition, not causality, solver validation, or evidence of Table 3/Table 6 improvement."
        )
    elif decision == SAME_SIGN_MIXTURE:
        conclusion = (
            "The Stage-137 rate split closes as a depth-varying channel-mixture effect without a material sign-changing cancellation under the fixed guards. "
            "The next justified artifact-only diagnostic is to audit the mixture-weight gradient; no solver coefficient or physical parameter is changed."
        )
    else:
        conclusion = (
            "Stage 138 cannot assign the Stage-137 rate split to the preregistered signed-mixture mechanisms without violating a provenance, closure, or finite-data guard. No solver or physical retuning is performed."
        )

    summary = {
        "stage": STAGE,
        "finite": finite,
        "configuration": cfg,
        "parents": {
            "stage136_run_id": EXPECTED_STAGE136_RUN_ID,
            "stage136_job_id": EXPECTED_STAGE136_JOB_ID,
            "stage136_artifact_id": EXPECTED_STAGE136_ARTIFACT_ID,
            "stage136_source_head": EXPECTED_STAGE136_SOURCE_HEAD,
            "stage137_run_id": EXPECTED_STAGE137_RUN_ID,
            "stage137_job_id": EXPECTED_STAGE137_JOB_ID,
            "stage137_artifact_id": EXPECTED_STAGE137_ARTIFACT_ID,
            "stage137_source_head": EXPECTED_STAGE137_SOURCE_HEAD,
        },
        "metrics": {
            "dominant_decay_rate_per_cell": kd,
            "parent_decay_rate_per_cell": kp,
            "mixing_decay_rate_per_cell": kmix,
            "rate_split_per_cell": split,
            "mixing_fraction_of_parent_decay_rate": float(kmix / kp) if kp != 0.0 else float("nan"),
            "rate_split_explained_fraction": float(rate_split_explained_fraction),
            "rate_identity_closure": float(rate_identity_closure),
            "complement_sign_change_count": sign_change_count,
            "first_complement_zero_crossing_depth_cells": crossing_depth,
            "nearest_complement_to_dominant_ratio": float(ratio[0]),
            "endpoint_complement_to_dominant_ratio": float(ratio[-1]),
            "complement_ratio_swing": ratio_swing,
            "maximum_pointwise_cancellation_fraction": float(np.max(cancellation)),
            "endpoint_cancellation_fraction": endpoint_cancellation,
            "opposite_sign_complement_l1_share": float(np.sum(np.abs(complement[signs != np.sign(dominant)])) / np.sum(np.abs(complement))),
        },
        "aggregate": {
            "maximum_parent_profile_closure": parent_profile_closure,
            "rate_identity_closure": float(rate_identity_closure),
            "rate_split_explained_fraction": float(rate_split_explained_fraction),
            "complement_sign_change_count": sign_change_count,
            "complement_ratio_swing": ratio_swing,
            "endpoint_cancellation_fraction": endpoint_cancellation,
            "parent_record_ok": record_ok,
        },
        "decision": decision,
        "scientific_conclusion": conclusion,
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. Stage 138 is a signed artifact decomposition only; its mixing rate is not a solver parameter. No physical, collision/source, floor, wall, reconstruction, transport, limiter, normalization, source-relaxation, or velocity-quadrature parameter is retuned; no solver endpoint or cross-Knudsen extension is advanced."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        output_dir / "channel_rate_origin.npz",
        right_depth=depth,
        dominant_signed=dominant,
        parent_signed=parent,
        complement_signed=complement,
        complement_to_dominant_ratio=ratio,
        mixing_factor=mixing_factor,
        pointwise_cancellation_fraction=cancellation,
        mixing_log_linear_fit=mixing_fit,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 138 fixed signed channel-rate origin audit")
    parser.add_argument("--stage136-dir", type=Path, required=True)
    parser.add_argument("--stage137-dir", type=Path, required=True)
    parser.add_argument("--stage137-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_stage138(args.stage136_dir, args.stage137_dir, args.stage137_record, args.output_dir)
    print(json.dumps({"stage": STAGE, "decision": summary["decision"], "aggregate": summary["aggregate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
