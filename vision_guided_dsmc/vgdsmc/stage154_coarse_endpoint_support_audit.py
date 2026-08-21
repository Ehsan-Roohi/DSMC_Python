from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = 154

EXPECTED_STAGE153_SOURCE_HEAD = "1ed9276166fc4cad2fdf2b1ebd5e2a9e0a7c6c2c"
EXPECTED_STAGE153_RUN_ID = 32476131806
EXPECTED_STAGE153_JOB_ID = 96752746805
EXPECTED_STAGE153_ARTIFACT_ID = 9457019335
EXPECTED_STAGE153_ARTIFACT_SHA256 = "d60c5c764ffd4b57eb8417c6d5de79a19a7ad9d9f79f3bfad1d04b917cd724c6"
EXPECTED_STAGE153_SUMMARY_SHA256 = "e48fee36cad86162c69309863970064db329bfc1412e91ec537f347da97de3c9"
EXPECTED_STAGE153_PAYLOAD_SHA256 = "23ce0124dad79d32bbc4d26cfc5fed147f5390df6433130060541ff4e2db8dad"
EXPECTED_STAGE153_DECISION = (
    "stage153_coarse_endpoint_cancellation_without_single_channel_dominance_"
    "stage154_coarse_endpoint_support_audit"
)

EXPECTED_STAGE147_SOURCE_HEAD = "3dd94ff4b773ee21358a88a25e660c776988406d"
EXPECTED_STAGE147_RUN_ID = 32358081943
EXPECTED_STAGE147_JOB_ID = 96391517821
EXPECTED_STAGE147_ARTIFACT_ID = 9410888827
EXPECTED_STAGE147_ARTIFACT_SHA256 = "444949e992230e54f224aa254fcf1f4c1743b535aaacd39e38d43d5ef20709c4"
EXPECTED_STAGE147_SUMMARY_SHA256 = "bf8fa6cc06e510fa2f4fbd76726db8e03faf5237bc95cc897fbfa7d42cf0ba5b"
EXPECTED_STAGE147_PAYLOAD_SHA256 = "a77b8bc80a291fe9e440d30f7c09e768964e86b9e873a9ab9890074caf8d6c2d"
EXPECTED_STAGE147_DECISION = (
    "stage147_material_bilateral_curvature_sign_reversal_"
    "stage148_sample_scale_curvature_alternation_audit"
)

PROVENANCE_MATCH_MAX = 1.0e-12
COARSE_ENDPOINT_CANCELLATION_MIN = 0.75
SINGLE_SAMPLE_SUPPORT_DOMINANCE_MIN = 0.75
MIN_BROAD_OPPOSED_SUPPORT_COUNT = 2

NONFINITE = "stage154_nonfinite_blocker"
STAGE153_RECORD_BLOCKER = "stage154_stage153_record_blocker"
STAGE147_RECORD_BLOCKER = "stage154_stage147_record_blocker"
PARENT_ROUTE_BLOCKER = "stage154_parent_route_blocker"
PROVENANCE_BLOCKER = "stage154_parent_provenance_blocker"
BROAD_SUPPORT = (
    "stage154_coarse_cancellation_broad_support_"
    "stage155_support_sign_geometry_audit"
)
SINGLE_SAMPLE_SUPPORT = (
    "stage154_coarse_cancellation_single_sample_support_"
    "stage155_single_sample_continuity_audit"
)
NONLOCAL_OPPOSITION = (
    "stage154_coarse_cancellation_not_broadly_pointwise_opposed_"
    "stage155_cross_node_balance_audit"
)


def validate_stage154_design(
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
    single_sample_support_dominance_min=SINGLE_SAMPLE_SUPPORT_DOMINANCE_MIN,
    min_broad_opposed_support_count=MIN_BROAD_OPPOSED_SUPPORT_COUNT,
    support_metrics_used_for_solver=False,
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
        "single_sample_support_dominance_min": SINGLE_SAMPLE_SUPPORT_DOMINANCE_MIN,
        "min_broad_opposed_support_count": MIN_BROAD_OPPOSED_SUPPORT_COUNT,
        "support_metrics_used_for_solver": False,
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
                f"Stage 154 frozen-design violation: {key}={got[key]!r}, expected {value!r}"
            )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_stage153_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 153
        and record.get("source_head") == EXPECTED_STAGE153_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE153_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE153_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE153_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE153_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE153_SUMMARY_SHA256
        and record.get("left_channel_endpoint_balance_sha256")
        == EXPECTED_STAGE153_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE153_DECISION
        and record.get("workflow_status") == "completed"
        and record.get("workflow_conclusion") == "success"
    )


def _check_stage147_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 147
        and record.get("source_head") == EXPECTED_STAGE147_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE147_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE147_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE147_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE147_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE147_SUMMARY_SHA256
        and record.get("dual_channel_neighborhood_sha256")
        == EXPECTED_STAGE147_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE147_DECISION
        and record.get("workflow_status") == "completed"
        and record.get("workflow_conclusion") == "success"
    )


def coarse_endpoint_support_metrics(
    *,
    depth: np.ndarray,
    dominant_signed: np.ndarray,
    parent_signed: np.ndarray,
    stage153_coarse_components: np.ndarray,
) -> dict:
    depth = np.asarray(depth, dtype=float)
    dominant = np.asarray(dominant_signed, dtype=float)
    parent = np.asarray(parent_signed, dtype=float)
    coarse153 = np.asarray(stage153_coarse_components, dtype=float)
    if depth.shape != (5,) or dominant.shape != (5,) or parent.shape != (5,):
        raise ValueError("Stage 154 requires finite inherited five-point profiles")
    if coarse153.shape != (2,):
        raise ValueError("Stage 154 requires the exact two-component Stage-153 coarse endpoint")
    if not (
        np.isfinite(depth).all()
        and np.isfinite(dominant).all()
        and np.isfinite(parent).all()
        and np.isfinite(coarse153).all()
    ):
        raise ValueError("Stage 154 requires finite parent arrays")

    support_indices = np.asarray([0, 2, 4], dtype=int)
    support_depth = depth[support_indices]

    # Stage 150/153 definitions:
    # dominant coarse curvature = -(0.5*q0 + 0.5*q4 - q2)
    # parent coarse curvature   = +(0.5*q0 + 0.5*q4 - q2)
    dominant_coeff = np.asarray([-0.5, 1.0, -0.5], dtype=float)
    parent_coeff = -dominant_coeff
    dominant_support = dominant_coeff * dominant[support_indices]
    parent_support = parent_coeff * parent[support_indices]
    combined_support = dominant_support + parent_support

    l1_support = np.abs(dominant_support) + np.abs(parent_support)
    cancellation_magnitude = l1_support - np.abs(combined_support)
    cancellation_fraction = np.divide(
        cancellation_magnitude,
        l1_support,
        out=np.zeros_like(cancellation_magnitude),
        where=l1_support > 0.0,
    )
    total_cancellation_support = float(cancellation_magnitude.sum())
    cancellation_share = np.divide(
        cancellation_magnitude,
        total_cancellation_support,
        out=np.zeros_like(cancellation_magnitude),
        where=total_cancellation_support > 0.0,
    )
    channel_sign_product = np.sign(dominant_support * parent_support).astype(int)
    opposed_count = int(np.count_nonzero(channel_sign_product < 0))
    max_share = float(cancellation_share.max()) if cancellation_share.size else 0.0
    max_index_local = int(np.argmax(cancellation_share)) if cancellation_share.size else -1

    reconstructed = np.asarray(
        [dominant_support.sum(), parent_support.sum()], dtype=float
    )
    endpoint_error = float(np.max(np.abs(reconstructed - coarse153)))
    combined_error = float(abs(combined_support.sum() - coarse153.sum()))
    maximum_error = max(endpoint_error, combined_error)

    weights = cancellation_share[cancellation_share > 0.0]
    effective_support = (
        float(1.0 / np.sum(weights * weights)) if weights.size else 0.0
    )

    return {
        "support_indices": support_indices.tolist(),
        "support_depth": support_depth.tolist(),
        "dominant_support_contributions": dominant_support.tolist(),
        "parent_support_contributions": parent_support.tolist(),
        "combined_support_contributions": combined_support.tolist(),
        "channel_sign_products": channel_sign_product.tolist(),
        "node_cross_channel_cancellation_fraction": cancellation_fraction.tolist(),
        "node_cross_channel_cancellation_magnitude": cancellation_magnitude.tolist(),
        "node_cancellation_support_share": cancellation_share.tolist(),
        "opposed_support_count": opposed_count,
        "maximum_single_sample_cancellation_support_share": max_share,
        "maximum_support_sample_index": int(support_indices[max_index_local])
        if max_index_local >= 0
        else -1,
        "maximum_support_sample_depth": float(support_depth[max_index_local])
        if max_index_local >= 0
        else np.nan,
        "effective_cancellation_support_count": effective_support,
        "reconstructed_coarse_components": reconstructed.tolist(),
        "stage153_coarse_components": coarse153.tolist(),
        "coarse_component_reconstruction_error": endpoint_error,
        "coarse_combined_reconstruction_error": combined_error,
        "maximum_identity_or_provenance_error": maximum_error,
    }


def classify_coarse_endpoint_support(
    *,
    metrics: dict,
    stage153_record_ok=True,
    stage147_record_ok=True,
    parent_route_ok=True,
    finite=True,
) -> str:
    numeric = [
        metrics.get("maximum_single_sample_cancellation_support_share", np.nan),
        metrics.get("effective_cancellation_support_count", np.nan),
        metrics.get("maximum_identity_or_provenance_error", np.nan),
    ]
    if not finite or not np.isfinite(numeric).all():
        return NONFINITE
    if not stage153_record_ok:
        return STAGE153_RECORD_BLOCKER
    if not stage147_record_ok:
        return STAGE147_RECORD_BLOCKER
    if not parent_route_ok:
        return PARENT_ROUTE_BLOCKER
    if float(metrics["maximum_identity_or_provenance_error"]) > PROVENANCE_MATCH_MAX:
        return PROVENANCE_BLOCKER

    max_share = float(metrics["maximum_single_sample_cancellation_support_share"])
    opposed_count = int(metrics["opposed_support_count"])
    if max_share >= SINGLE_SAMPLE_SUPPORT_DOMINANCE_MIN:
        return SINGLE_SAMPLE_SUPPORT
    if opposed_count >= MIN_BROAD_OPPOSED_SUPPORT_COUNT:
        return BROAD_SUPPORT
    return NONLOCAL_OPPOSITION


def run_stage154(
    stage153_dir: Path,
    stage153_record: Path,
    stage147_dir: Path,
    stage147_record: Path,
    output_dir: Path,
) -> dict:
    validate_stage154_design()
    summary153 = _load_json(stage153_dir / "summary.json")
    record153 = _load_json(stage153_record)
    summary147 = _load_json(stage147_dir / "summary.json")
    record147 = _load_json(stage147_record)
    stage153_record_ok = _check_stage153_record(record153)
    stage147_record_ok = _check_stage147_record(record147)
    parent_route_ok = bool(
        summary153.get("stage") == 153
        and summary153.get("decision") == EXPECTED_STAGE153_DECISION
        and summary153.get("aggregate", {}).get(
            "coarse_cross_channel_cancellation_fraction", 0.0
        )
        >= COARSE_ENDPOINT_CANCELLATION_MIN
        and summary153.get("aggregate", {}).get(
            "coarse_stronger_absolute_share", 1.0
        )
        < SINGLE_SAMPLE_SUPPORT_DOMINANCE_MIN
        and summary147.get("stage") == 147
        and summary147.get("decision") == EXPECTED_STAGE147_DECISION
    )

    with np.load(stage153_dir / "left_channel_endpoint_balance.npz") as data153, np.load(
        stage147_dir / "dual_channel_neighborhood.npz"
    ) as data147:
        metrics = coarse_endpoint_support_metrics(
            depth=data147["five_point_depth"],
            dominant_signed=data147["five_point_dominant_signed"],
            parent_signed=data147["five_point_parent_signed"],
            stage153_coarse_components=data153["coarse_components"],
        )

    finite = bool(
        np.isfinite(
            [
                metrics["maximum_single_sample_cancellation_support_share"],
                metrics["effective_cancellation_support_count"],
                metrics["maximum_identity_or_provenance_error"],
            ]
        ).all()
    )
    decision = classify_coarse_endpoint_support(
        metrics=metrics,
        stage153_record_ok=stage153_record_ok,
        stage147_record_ok=stage147_record_ok,
        parent_route_ok=parent_route_ok,
        finite=finite,
    )

    if decision == BROAD_SUPPORT:
        conclusion = (
            "The Stage-153 coarse cross-channel cancellation is not concentrated at one inherited coarse-stencil sample. "
            "At least two fixed support samples are pointwise opposed and the largest single-sample share of the local "
            "cross-channel cancellation magnitude remains below the preregistered 75% guard. This supports a descriptive "
            "sign-geometry audit of the broad support only; it does not establish a physical mode, limiter causality, or "
            "solver instability."
        )
    elif decision == SINGLE_SAMPLE_SUPPORT:
        conclusion = (
            "The Stage-153 coarse cross-channel cancellation is materially concentrated at one fixed coarse-stencil sample "
            "under the preregistered 75% support-share guard. The next artifact-only question is whether that one-sample "
            "feature persists under its inherited neighboring support; no solver retuning is justified."
        )
    else:
        conclusion = (
            "The Stage-153 coarse endpoint cancellation is not reproduced as broad pointwise opposition on the inherited "
            "coarse-stencil samples. The remaining balance must be treated as a cross-node identity rather than a localized "
            "sample mechanism."
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
        "single_sample_support_dominance_min": SINGLE_SAMPLE_SUPPORT_DOMINANCE_MIN,
        "min_broad_opposed_support_count": MIN_BROAD_OPPOSED_SUPPORT_COUNT,
        "support_metrics_used_for_solver": False,
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
            "stage153_source_head": EXPECTED_STAGE153_SOURCE_HEAD,
            "stage153_run_id": EXPECTED_STAGE153_RUN_ID,
            "stage153_job_id": EXPECTED_STAGE153_JOB_ID,
            "stage153_artifact_id": EXPECTED_STAGE153_ARTIFACT_ID,
            "stage147_source_head": EXPECTED_STAGE147_SOURCE_HEAD,
            "stage147_run_id": EXPECTED_STAGE147_RUN_ID,
            "stage147_job_id": EXPECTED_STAGE147_JOB_ID,
            "stage147_artifact_id": EXPECTED_STAGE147_ARTIFACT_ID,
        },
        "aggregate": {
            "stage153_record_ok": stage153_record_ok,
            "stage147_record_ok": stage147_record_ok,
            "parent_route_ok": parent_route_ok,
            "opposed_support_count": metrics["opposed_support_count"],
            "maximum_single_sample_cancellation_support_share": metrics[
                "maximum_single_sample_cancellation_support_share"
            ],
            "effective_cancellation_support_count": metrics[
                "effective_cancellation_support_count"
            ],
            "maximum_identity_or_provenance_error": metrics[
                "maximum_identity_or_provenance_error"
            ],
        },
        "metrics": metrics,
        "scientific_conclusion": conclusion,
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. Stage 154 is an artifact-only "
            "coarse-endpoint support audit; pointwise channel opposition, cancellation-support shares, and effective-support "
            "counts are diagnostics, not solver parameters. No physical, collision/source, floor, wall, reconstruction, "
            "transport, limiter, normalization, source-relaxation, or velocity-quadrature parameter is retuned; no measured "
            "support scale is fed back into the solver, no solver endpoint or cross-Knudsen extension is advanced, and no "
            "benchmark or validation claim is permitted."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(
        output_dir / "coarse_endpoint_support.npz",
        support_indices=np.asarray(metrics["support_indices"], dtype=int),
        support_depth=np.asarray(metrics["support_depth"], dtype=float),
        dominant_support_contributions=np.asarray(
            metrics["dominant_support_contributions"], dtype=float
        ),
        parent_support_contributions=np.asarray(
            metrics["parent_support_contributions"], dtype=float
        ),
        combined_support_contributions=np.asarray(
            metrics["combined_support_contributions"], dtype=float
        ),
        channel_sign_products=np.asarray(metrics["channel_sign_products"], dtype=int),
        node_cross_channel_cancellation_fraction=np.asarray(
            metrics["node_cross_channel_cancellation_fraction"], dtype=float
        ),
        node_cross_channel_cancellation_magnitude=np.asarray(
            metrics["node_cross_channel_cancellation_magnitude"], dtype=float
        ),
        node_cancellation_support_share=np.asarray(
            metrics["node_cancellation_support_share"], dtype=float
        ),
        reconstructed_coarse_components=np.asarray(
            metrics["reconstructed_coarse_components"], dtype=float
        ),
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 154 fixed coarse-endpoint support audit"
    )
    parser.add_argument("--stage153-dir", type=Path, required=True)
    parser.add_argument("--stage153-record", type=Path, required=True)
    parser.add_argument("--stage147-dir", type=Path, required=True)
    parser.add_argument("--stage147-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage154(
                args.stage153_dir,
                args.stage153_record,
                args.stage147_dir,
                args.stage147_record,
                args.output_dir,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
