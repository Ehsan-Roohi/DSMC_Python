from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = 153

EXPECTED_STAGE152_SOURCE_HEAD = "38409ba5bcf585fd7ada33bae9662bf66f32ef53"
EXPECTED_STAGE152_RUN_ID = 32450348284
EXPECTED_STAGE152_JOB_ID = 96677573546
EXPECTED_STAGE152_ARTIFACT_ID = 9443308178
EXPECTED_STAGE152_ARTIFACT_SHA256 = "551f65a315f856c59825248531259f46a9f4f46c59e141357113ba88f85150b9"
EXPECTED_STAGE152_SUMMARY_SHA256 = "e4e4a79e37321a540855ff8a84ecef07d2f36e48a4239ff532bd927948337291"
EXPECTED_STAGE152_PAYLOAD_SHA256 = "cce055806167ad30d408d162680ddf02670823d507f1b260dc76cfe5451949d4"
EXPECTED_STAGE152_DECISION = "stage152_left_balance_mixed_cancellation_stage153_left_channel_endpoint_balance_audit"

EXPECTED_STAGE150_SOURCE_HEAD = "1e4fe5ff5e3542e6e0cc7c8b036c9f5ad6dda994"
EXPECTED_STAGE150_RUN_ID = 32431079197
EXPECTED_STAGE150_JOB_ID = 96622574831
EXPECTED_STAGE150_ARTIFACT_ID = 9430006423
EXPECTED_STAGE150_ARTIFACT_SHA256 = "ce7e2be6eb7e5bc93dca540430c58671642a68706012a61afea61600972b45d8"
EXPECTED_STAGE150_SUMMARY_SHA256 = "fbabcb9d83d008b0d52b96ea80c546636461c150a821a713cf773bfe31f7376a"
EXPECTED_STAGE150_PAYLOAD_SHA256 = "5e841e56d1d22d82fd73f9bfd6193857dc65b2ac42849ced065b273d264dc5c0"
EXPECTED_STAGE150_DECISION = "stage150_opposing_outer_intervals_no_single_side_dominance_stage151_outer_interval_asymmetry_audit"

PROVENANCE_MATCH_MAX = 1.0e-12
COARSE_ENDPOINT_CANCELLATION_MIN = 0.75
SINGLE_CHANNEL_DOMINANCE_MIN = 0.75

NONFINITE = "stage153_nonfinite_blocker"
STAGE152_RECORD_BLOCKER = "stage153_stage152_record_blocker"
STAGE150_RECORD_BLOCKER = "stage153_stage150_record_blocker"
PARENT_ROUTE_BLOCKER = "stage153_parent_route_blocker"
PROVENANCE_BLOCKER = "stage153_parent_provenance_blocker"
COARSE_CANCELLATION = (
    "stage153_coarse_endpoint_cancellation_without_single_channel_dominance_"
    "stage154_coarse_endpoint_support_audit"
)
COARSE_SINGLE_CHANNEL = (
    "stage153_coarse_endpoint_single_channel_dominance_"
    "stage154_dominant_endpoint_continuity_audit"
)
DISTRIBUTED_ENDPOINT_BALANCE = (
    "stage153_endpoint_balance_distributed_"
    "stage154_five_point_endpoint_shape_audit"
)


def validate_stage153_design(
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
    coarse_endpoint_cancellation_min=COARSE_ENDPOINT_CANCELLATION_MIN,
    single_channel_dominance_min=SINGLE_CHANNEL_DOMINANCE_MIN,
    endpoint_metrics_used_for_solver=False,
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
        "coarse_endpoint_cancellation_min": COARSE_ENDPOINT_CANCELLATION_MIN,
        "single_channel_dominance_min": SINGLE_CHANNEL_DOMINANCE_MIN,
        "endpoint_metrics_used_for_solver": False,
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
            raise ValueError(
                f"Stage 153 frozen-design violation: {key}={got[key]!r}, expected {value!r}"
            )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_stage152_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 152
        and record.get("source_head") == EXPECTED_STAGE152_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE152_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE152_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE152_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE152_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE152_SUMMARY_SHA256
        and record.get("left_outer_channel_balance_sha256") == EXPECTED_STAGE152_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE152_DECISION
        and record.get("workflow_status") == "completed"
        and record.get("workflow_conclusion") == "success"
    )


def _check_stage150_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 150
        and record.get("source_head") == EXPECTED_STAGE150_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE150_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE150_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE150_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE150_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE150_SUMMARY_SHA256
        and record.get("channel_sign_transition_localization_sha256") == EXPECTED_STAGE150_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE150_DECISION
        and record.get("workflow_status") == "completed"
        and record.get("workflow_conclusion") == "success"
    )


def _endpoint_pair_metrics(components: np.ndarray) -> dict:
    x = np.asarray(components, dtype=float)
    if x.shape != (2,) or not np.isfinite(x).all():
        raise ValueError("Stage 153 requires a finite two-channel endpoint vector")
    dominant, parent = map(float, x)
    total = dominant + parent
    l1 = abs(dominant) + abs(parent)
    if l1 <= 0.0:
        dominant_share = parent_share = cancellation = retention = 0.0
    else:
        dominant_share = abs(dominant) / l1
        parent_share = abs(parent) / l1
        retention = abs(total) / l1
        cancellation = 1.0 - retention
    return {
        "dominant": dominant,
        "parent": parent,
        "combined": total,
        "channel_sign_product": int(np.sign(dominant * parent)),
        "dominant_absolute_share": float(dominant_share),
        "parent_absolute_share": float(parent_share),
        "stronger_absolute_share": float(max(dominant_share, parent_share)),
        "stronger_channel": "dominant" if dominant_share >= parent_share else "parent",
        "cross_channel_cancellation_fraction": float(cancellation),
        "combined_l1_retention": float(retention),
    }


def left_channel_endpoint_balance_metrics(
    *,
    stage152_dominant_outer: np.ndarray,
    stage152_parent_outer: np.ndarray,
    stage152_combined_outer: np.ndarray,
    stage150_dominant_outer: np.ndarray,
    stage150_parent_outer: np.ndarray,
    stage150_fine_components: np.ndarray,
    stage150_coarse_components: np.ndarray,
) -> dict:
    d152 = np.asarray(stage152_dominant_outer, dtype=float)
    p152 = np.asarray(stage152_parent_outer, dtype=float)
    c152 = np.asarray(stage152_combined_outer, dtype=float)
    d150 = np.asarray(stage150_dominant_outer, dtype=float)
    p150 = np.asarray(stage150_parent_outer, dtype=float)
    fine = np.asarray(stage150_fine_components, dtype=float)
    coarse = np.asarray(stage150_coarse_components, dtype=float)
    for name, arr in {
        "stage152_dominant_outer": d152,
        "stage152_parent_outer": p152,
        "stage152_combined_outer": c152,
        "stage150_dominant_outer": d150,
        "stage150_parent_outer": p150,
        "stage150_fine_components": fine,
        "stage150_coarse_components": coarse,
    }.items():
        if arr.shape != (2,) or not np.isfinite(arr).all():
            raise ValueError(f"Stage 153 requires finite length-two vector {name}")

    outer_provenance = float(
        max(
            np.max(np.abs(d152 - d150)),
            np.max(np.abs(p152 - p150)),
            np.max(np.abs(c152 - (d150 + p150))),
        )
    )
    fine_metrics = _endpoint_pair_metrics(fine)
    coarse_metrics = _endpoint_pair_metrics(coarse)
    combined_fine = float(fine_metrics["combined"])
    combined_coarse = float(coarse_metrics["combined"])
    tiny = np.finfo(float).tiny
    endpoint_span = coarse - fine
    expected_span = np.array([d150.sum(), p150.sum()], dtype=float)
    span_closure = float(np.max(np.abs(endpoint_span - expected_span)))
    combined_span_closure = float(
        abs((combined_coarse - combined_fine) - float(c152.sum()))
    )
    maximum_error = max(outer_provenance, span_closure, combined_span_closure)

    return {
        "fine": fine_metrics,
        "coarse": coarse_metrics,
        "dominant_endpoint_sign_transition": bool(fine[0] * coarse[0] < 0.0),
        "parent_endpoint_sign_transition": bool(fine[1] * coarse[1] < 0.0),
        "combined_coarse_to_fine_magnitude_ratio": float(
            abs(combined_coarse) / max(abs(combined_fine), tiny)
        ),
        "coarse_minus_fine_components": endpoint_span.tolist(),
        "expected_component_scale_increments": expected_span.tolist(),
        "component_span_closure_error": span_closure,
        "combined_span_closure_error": combined_span_closure,
        "outer_vector_provenance_error": outer_provenance,
        "maximum_identity_or_provenance_error": maximum_error,
    }


def classify_left_channel_endpoint_balance(
    *,
    metrics: dict,
    stage152_record_ok=True,
    stage150_record_ok=True,
    parent_route_ok=True,
    finite=True,
) -> str:
    coarse = metrics.get("coarse", {})
    numeric = [
        coarse.get("cross_channel_cancellation_fraction", np.nan),
        coarse.get("stronger_absolute_share", np.nan),
        metrics.get("maximum_identity_or_provenance_error", np.nan),
    ]
    if not finite or not np.isfinite(numeric).all():
        return NONFINITE
    if not stage152_record_ok:
        return STAGE152_RECORD_BLOCKER
    if not stage150_record_ok:
        return STAGE150_RECORD_BLOCKER
    if not parent_route_ok:
        return PARENT_ROUTE_BLOCKER
    if float(metrics["maximum_identity_or_provenance_error"]) > PROVENANCE_MATCH_MAX:
        return PROVENANCE_BLOCKER
    if (
        int(metrics["fine"]["channel_sign_product"]) > 0
        and int(coarse["channel_sign_product"]) < 0
        and float(coarse["cross_channel_cancellation_fraction"])
        >= COARSE_ENDPOINT_CANCELLATION_MIN
    ):
        if float(coarse["stronger_absolute_share"]) >= SINGLE_CHANNEL_DOMINANCE_MIN:
            return COARSE_SINGLE_CHANNEL
        return COARSE_CANCELLATION
    return DISTRIBUTED_ENDPOINT_BALANCE


def run_stage153(
    stage152_dir: Path,
    stage152_record: Path,
    stage150_dir: Path,
    stage150_record: Path,
    output_dir: Path,
) -> dict:
    validate_stage153_design()
    summary152 = _load_json(stage152_dir / "summary.json")
    record152 = _load_json(stage152_record)
    summary150 = _load_json(stage150_dir / "summary.json")
    record150 = _load_json(stage150_record)
    stage152_record_ok = _check_stage152_record(record152)
    stage150_record_ok = _check_stage150_record(record150)
    parent_route_ok = bool(
        summary152.get("stage") == 152
        and summary152.get("decision") == EXPECTED_STAGE152_DECISION
        and summary152.get("aggregate", {}).get("left_channel_sign_product") == -1
        and summary152.get("aggregate", {}).get("left_cancellation_fraction", 0.0) >= 0.50
        and summary152.get("aggregate", {}).get("left_parent_absolute_share", 1.0) < 0.75
        and summary150.get("stage") == 150
        and summary150.get("decision") == EXPECTED_STAGE150_DECISION
    )

    with np.load(stage152_dir / "left_outer_channel_balance.npz") as data152, np.load(
        stage150_dir / "channel_sign_transition_localization.npz"
    ) as data150:
        metrics = left_channel_endpoint_balance_metrics(
            stage152_dominant_outer=data152["dominant_outer_increments"],
            stage152_parent_outer=data152["parent_outer_increments"],
            stage152_combined_outer=data152["combined_outer_increments"],
            stage150_dominant_outer=data150["dominant_outer_increments"],
            stage150_parent_outer=data150["parent_outer_increments"],
            stage150_fine_components=data150["fine_components"],
            stage150_coarse_components=data150["coarse_components"],
        )

    finite = bool(
        np.isfinite(
            [
                metrics["fine"]["cross_channel_cancellation_fraction"],
                metrics["coarse"]["cross_channel_cancellation_fraction"],
                metrics["combined_coarse_to_fine_magnitude_ratio"],
                metrics["maximum_identity_or_provenance_error"],
            ]
        ).all()
    )
    decision = classify_left_channel_endpoint_balance(
        metrics=metrics,
        stage152_record_ok=stage152_record_ok,
        stage150_record_ok=stage150_record_ok,
        parent_route_ok=parent_route_ok,
        finite=finite,
    )

    if decision == COARSE_CANCELLATION:
        conclusion = (
            "The mixed Stage-152 left balance is associated with a fixed fine-to-coarse endpoint transition: the two "
            "channels reinforce at the fine endpoint but oppose strongly at the coarse endpoint, where cross-channel "
            "cancellation exceeds the preregistered 75% guard. Neither coarse component reaches the fixed 75% "
            "single-channel dominance guard. The next artifact-only question is whether this coarse cancellation has "
            "broad support in the inherited five-point profile or is concentrated in one sampled location."
        )
    elif decision == COARSE_SINGLE_CHANNEL:
        conclusion = (
            "The fine-to-coarse endpoint transition becomes cancellation-dominated at the coarse endpoint and one fixed "
            "channel also exceeds the 75% absolute-share guard. A continuity audit of that channel is justified, but no "
            "solver change is implied."
        )
    else:
        conclusion = (
            "The endpoint decomposition does not isolate a preregistered coarse-cancellation route. The remaining "
            "five-point endpoint structure must remain descriptive and artifact-only."
        )

    config = {
        "grid": [64, 64],
        "interior_grid": [56, 56],
        "kn0": 10.0,
        "cold_hot_ratio": 0.1,
        "rule": [40, 96],
        "radial_scale": 2.0,
        "limiter": "minmod",
        "boundary_slope": "zero",
        "source_relaxation": 1.0,
        "correction_floor": 0.05,
        "witness_node": 9,
        "pair_sectors": [5, 6],
        "dominant_mirrored_sector": 6,
        "coarse_endpoint_cancellation_min": COARSE_ENDPOINT_CANCELLATION_MIN,
        "single_channel_dominance_min": SINGLE_CHANNEL_DOMINANCE_MIN,
        "endpoint_metrics_used_for_solver": False,
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
    payload = {
        "stage": STAGE,
        "decision": decision,
        "finite": finite,
        "configuration": config,
        "parents": {
            "stage152_source_head": EXPECTED_STAGE152_SOURCE_HEAD,
            "stage152_run_id": EXPECTED_STAGE152_RUN_ID,
            "stage152_job_id": EXPECTED_STAGE152_JOB_ID,
            "stage152_artifact_id": EXPECTED_STAGE152_ARTIFACT_ID,
            "stage150_source_head": EXPECTED_STAGE150_SOURCE_HEAD,
            "stage150_run_id": EXPECTED_STAGE150_RUN_ID,
            "stage150_job_id": EXPECTED_STAGE150_JOB_ID,
            "stage150_artifact_id": EXPECTED_STAGE150_ARTIFACT_ID,
        },
        "aggregate": {
            "stage152_record_ok": stage152_record_ok,
            "stage150_record_ok": stage150_record_ok,
            "parent_route_ok": parent_route_ok,
            "fine_channel_sign_product": metrics["fine"]["channel_sign_product"],
            "coarse_channel_sign_product": metrics["coarse"]["channel_sign_product"],
            "coarse_cross_channel_cancellation_fraction": metrics["coarse"][
                "cross_channel_cancellation_fraction"
            ],
            "coarse_stronger_absolute_share": metrics["coarse"]["stronger_absolute_share"],
            "maximum_identity_or_provenance_error": metrics[
                "maximum_identity_or_provenance_error"
            ],
        },
        "metrics": metrics,
        "scientific_conclusion": conclusion,
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. Stage 153 is an artifact-only "
            "left-channel endpoint-balance audit; endpoint shares, sign products, cancellation fractions, and retention "
            "ratios are diagnostics, not solver parameters. No physical, collision/source, floor, wall, reconstruction, "
            "transport, limiter, normalization, source-relaxation, or velocity-quadrature parameter is retuned; no measured "
            "endpoint balance is fed back into the solver, no solver endpoint or cross-Knudsen extension is advanced, and "
            "no benchmark or validation claim is permitted."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        output_dir / "left_channel_endpoint_balance.npz",
        fine_components=np.asarray(
            [metrics["fine"]["dominant"], metrics["fine"]["parent"]], dtype=float
        ),
        coarse_components=np.asarray(
            [metrics["coarse"]["dominant"], metrics["coarse"]["parent"]], dtype=float
        ),
        combined_endpoints=np.asarray(
            [metrics["fine"]["combined"], metrics["coarse"]["combined"]], dtype=float
        ),
        endpoint_cancellation_fractions=np.asarray(
            [
                metrics["fine"]["cross_channel_cancellation_fraction"],
                metrics["coarse"]["cross_channel_cancellation_fraction"],
            ],
            dtype=float,
        ),
        endpoint_stronger_absolute_shares=np.asarray(
            [metrics["fine"]["stronger_absolute_share"], metrics["coarse"]["stronger_absolute_share"]],
            dtype=float,
        ),
        component_scale_increments=np.asarray(metrics["coarse_minus_fine_components"], dtype=float),
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 153 fixed left-channel endpoint-balance audit")
    parser.add_argument("--stage152-dir", type=Path, required=True)
    parser.add_argument("--stage152-record", type=Path, required=True)
    parser.add_argument("--stage150-dir", type=Path, required=True)
    parser.add_argument("--stage150-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage153(
                args.stage152_dir,
                args.stage152_record,
                args.stage150_dir,
                args.stage150_record,
                args.output_dir,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
