from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = 155

EXPECTED_STAGE154_SOURCE_HEAD = "3f28467704bcb17eee10039fb2f611ebd48676d6"
EXPECTED_STAGE154_RUN_ID = 32512066958
EXPECTED_STAGE154_JOB_ID = 96865265125
EXPECTED_STAGE154_ARTIFACT_ID = 9463275200
EXPECTED_STAGE154_ARTIFACT_SHA256 = "3b8eb8f1d9fd026d992af714c7a1e1560aaa79dfc95e69e2bedf7a24784bb8c0"
EXPECTED_STAGE154_SUMMARY_SHA256 = "8ee64f72a6cc973166c591fef332c2c6de65d880ed2ce8dba7fc6bdbea64a410"
EXPECTED_STAGE154_PAYLOAD_SHA256 = "20f81281d21be1d9f27d9d6d9fea8d33e59f2d58b03bb6e5f0831779de98c144"
EXPECTED_STAGE154_DECISION = (
    "stage154_coarse_cancellation_broad_support_"
    "stage155_support_sign_geometry_audit"
)

PROVENANCE_MATCH_MAX = 1.0e-12
RAW_SAME_SIGN_FRACTION_MIN = 1.0
COEFFICIENT_EXPLAINED_OPPOSITION_MIN = 1.0

NONFINITE = "stage155_nonfinite_blocker"
STAGE154_RECORD_BLOCKER = "stage155_stage154_record_blocker"
PARENT_ROUTE_BLOCKER = "stage155_parent_route_blocker"
PROVENANCE_BLOCKER = "stage155_parent_provenance_blocker"
COEFFICIENT_IMPOSED = (
    "stage155_opposition_coefficient_imposed_"
    "stage156_support_amplitude_balance_audit"
)
RAW_SIGN_STRUCTURE = (
    "stage155_raw_sign_structure_material_"
    "stage156_raw_sign_continuity_audit"
)


def validate_stage155_design(
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
    dominant_coefficients=(-0.5, 1.0, -0.5),
    parent_coefficients=(0.5, -1.0, 0.5),
    raw_same_sign_fraction_min=RAW_SAME_SIGN_FRACTION_MIN,
    coefficient_explained_opposition_min=COEFFICIENT_EXPLAINED_OPPOSITION_MIN,
    sign_geometry_used_for_solver=False,
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
        "dominant_coefficients": (-0.5, 1.0, -0.5),
        "parent_coefficients": (0.5, -1.0, 0.5),
        "raw_same_sign_fraction_min": RAW_SAME_SIGN_FRACTION_MIN,
        "coefficient_explained_opposition_min": COEFFICIENT_EXPLAINED_OPPOSITION_MIN,
        "sign_geometry_used_for_solver": False,
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
                f"Stage 155 frozen-design violation: {key}={got[key]!r}, expected {value!r}"
            )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_stage154_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 154
        and record.get("source_head") == EXPECTED_STAGE154_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE154_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE154_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE154_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE154_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE154_SUMMARY_SHA256
        and record.get("coarse_endpoint_support_sha256") == EXPECTED_STAGE154_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE154_DECISION
        and record.get("workflow_status") == "completed"
        and record.get("workflow_conclusion") == "success"
    )


def support_sign_geometry_metrics(
    *,
    support_depth: np.ndarray,
    dominant_support_contributions: np.ndarray,
    parent_support_contributions: np.ndarray,
    stage154_cancellation_fraction: np.ndarray,
) -> dict:
    depth = np.asarray(support_depth, dtype=float)
    dominant_contrib = np.asarray(dominant_support_contributions, dtype=float)
    parent_contrib = np.asarray(parent_support_contributions, dtype=float)
    observed_cancel = np.asarray(stage154_cancellation_fraction, dtype=float)
    if (
        depth.shape != (3,)
        or dominant_contrib.shape != (3,)
        or parent_contrib.shape != (3,)
        or observed_cancel.shape != (3,)
    ):
        raise ValueError("Stage 155 requires the exact three-node Stage-154 support arrays")
    if not (
        np.isfinite(depth).all()
        and np.isfinite(dominant_contrib).all()
        and np.isfinite(parent_contrib).all()
        and np.isfinite(observed_cancel).all()
    ):
        raise ValueError("Stage 155 requires finite parent arrays")

    dominant_coeff = np.asarray([-0.5, 1.0, -0.5], dtype=float)
    parent_coeff = -dominant_coeff
    dominant_raw = dominant_contrib / dominant_coeff
    parent_raw = parent_contrib / parent_coeff

    raw_sign_product = np.sign(dominant_raw * parent_raw).astype(int)
    coefficient_sign_product = np.sign(dominant_coeff * parent_coeff).astype(int)
    contribution_sign_product = np.sign(dominant_contrib * parent_contrib).astype(int)

    opposed = contribution_sign_product < 0
    coefficient_explained = opposed & (raw_sign_product > 0) & (coefficient_sign_product < 0)
    opposed_count = int(np.count_nonzero(opposed))
    explained_fraction = (
        float(np.count_nonzero(coefficient_explained) / opposed_count)
        if opposed_count
        else 0.0
    )
    raw_same_sign_fraction = float(np.mean(raw_sign_product > 0))

    dominant_raw_reversals = int(np.count_nonzero(np.sign(dominant_raw[1:]) != np.sign(dominant_raw[:-1])))
    parent_raw_reversals = int(np.count_nonzero(np.sign(parent_raw[1:]) != np.sign(parent_raw[:-1])))
    dominant_contribution_reversals = int(np.count_nonzero(np.sign(dominant_contrib[1:]) != np.sign(dominant_contrib[:-1])))
    parent_contribution_reversals = int(np.count_nonzero(np.sign(parent_contrib[1:]) != np.sign(parent_contrib[:-1])))

    dmag = np.abs(dominant_raw)
    pmag = np.abs(parent_raw)
    denom = float(np.linalg.norm(dmag) * np.linalg.norm(pmag))
    raw_magnitude_cosine = float(np.dot(dmag, pmag) / denom) if denom > 0.0 else 0.0
    amplitude_ratio_parent_to_dominant = np.divide(
        pmag, dmag, out=np.full_like(pmag, np.nan), where=dmag > 0.0
    )
    predicted_cancel = np.divide(
        2.0 * np.minimum(dmag, pmag),
        dmag + pmag,
        out=np.zeros_like(dmag),
        where=(dmag + pmag) > 0.0,
    )
    cancellation_reconstruction_error = float(np.max(np.abs(predicted_cancel - observed_cancel)))

    return {
        "support_depth": depth.tolist(),
        "dominant_coefficients": dominant_coeff.tolist(),
        "parent_coefficients": parent_coeff.tolist(),
        "dominant_raw_support": dominant_raw.tolist(),
        "parent_raw_support": parent_raw.tolist(),
        "raw_cross_channel_sign_products": raw_sign_product.tolist(),
        "coefficient_cross_channel_sign_products": coefficient_sign_product.tolist(),
        "contribution_cross_channel_sign_products": contribution_sign_product.tolist(),
        "raw_same_sign_fraction": raw_same_sign_fraction,
        "coefficient_explained_opposition_fraction": explained_fraction,
        "dominant_raw_sign_reversal_count": dominant_raw_reversals,
        "parent_raw_sign_reversal_count": parent_raw_reversals,
        "dominant_contribution_sign_reversal_count": dominant_contribution_reversals,
        "parent_contribution_sign_reversal_count": parent_contribution_reversals,
        "raw_magnitude_cosine": raw_magnitude_cosine,
        "amplitude_ratio_parent_to_dominant": amplitude_ratio_parent_to_dominant.tolist(),
        "predicted_cancellation_fraction": predicted_cancel.tolist(),
        "observed_cancellation_fraction": observed_cancel.tolist(),
        "cancellation_fraction_reconstruction_error": cancellation_reconstruction_error,
        "maximum_identity_or_provenance_error": cancellation_reconstruction_error,
    }


def classify_support_sign_geometry(
    *,
    metrics: dict,
    stage154_record_ok=True,
    parent_route_ok=True,
    finite=True,
) -> str:
    numeric = [
        metrics.get("raw_same_sign_fraction", np.nan),
        metrics.get("coefficient_explained_opposition_fraction", np.nan),
        metrics.get("cancellation_fraction_reconstruction_error", np.nan),
    ]
    if not finite or not np.isfinite(numeric).all():
        return NONFINITE
    if not stage154_record_ok:
        return STAGE154_RECORD_BLOCKER
    if not parent_route_ok:
        return PARENT_ROUTE_BLOCKER
    if float(metrics["maximum_identity_or_provenance_error"]) > PROVENANCE_MATCH_MAX:
        return PROVENANCE_BLOCKER

    if (
        float(metrics["raw_same_sign_fraction"]) >= RAW_SAME_SIGN_FRACTION_MIN
        and float(metrics["coefficient_explained_opposition_fraction"]) >= COEFFICIENT_EXPLAINED_OPPOSITION_MIN
        and int(metrics["dominant_raw_sign_reversal_count"]) == 0
        and int(metrics["parent_raw_sign_reversal_count"]) == 0
    ):
        return COEFFICIENT_IMPOSED
    return RAW_SIGN_STRUCTURE


def run_stage155(stage154_dir: Path, stage154_record: Path, output_dir: Path) -> dict:
    validate_stage155_design()
    summary154 = _load_json(stage154_dir / "summary.json")
    record154 = _load_json(stage154_record)
    stage154_record_ok = _check_stage154_record(record154)
    parent_route_ok = bool(
        summary154.get("stage") == 154
        and summary154.get("decision") == EXPECTED_STAGE154_DECISION
        and summary154.get("aggregate", {}).get("opposed_support_count") == 3
        and summary154.get("aggregate", {}).get("maximum_single_sample_cancellation_support_share", 1.0) < 0.75
    )

    with np.load(stage154_dir / "coarse_endpoint_support.npz") as data:
        metrics = support_sign_geometry_metrics(
            support_depth=data["support_depth"],
            dominant_support_contributions=data["dominant_support_contributions"],
            parent_support_contributions=data["parent_support_contributions"],
            stage154_cancellation_fraction=data["node_cross_channel_cancellation_fraction"],
        )

    finite = bool(np.isfinite([
        metrics["raw_same_sign_fraction"],
        metrics["coefficient_explained_opposition_fraction"],
        metrics["raw_magnitude_cosine"],
        metrics["cancellation_fraction_reconstruction_error"],
    ]).all())
    decision = classify_support_sign_geometry(
        metrics=metrics,
        stage154_record_ok=stage154_record_ok,
        parent_route_ok=parent_route_ok,
        finite=finite,
    )

    if decision == COEFFICIENT_IMPOSED:
        conclusion = (
            "The broad pointwise opposition seen in Stage 154 is not an independent raw-profile sign mode. "
            "The recovered dominant and parent raw support profiles keep the same sign at all three coarse nodes, "
            "while the fixed second-difference coefficient vectors are exact negatives of one another. Those coefficient "
            "signs therefore explain all opposed support nodes; the remaining nontrivial structure is the depth-dependent "
            "raw amplitude balance."
        )
    else:
        conclusion = (
            "The Stage-154 support opposition cannot be reduced completely to the fixed coarse-curvature coefficient signs "
            "under the preregistered exact sign guards. A raw-sign continuity audit is therefore required before any narrower "
            "interpretation. No solver mechanism is established."
        )

    negative_guard = (
        "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. Stage 155 is an artifact-only "
        "sign-geometry audit; recovered raw signs, fixed operator-coefficient signs, amplitude ratios, and cancellation "
        "reconstructions are diagnostics, not solver parameters. No physical, collision/source, floor, wall, reconstruction, "
        "transport, limiter, normalization, source-relaxation, or velocity-quadrature parameter is retuned; no measured sign "
        "or amplitude feature is fed back into the solver, no solver endpoint or cross-Knudsen extension is advanced, and no "
        "benchmark or validation claim is permitted."
    )

    summary = {
        "stage": STAGE,
        "decision": decision,
        "finite": finite,
        "configuration": {
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
            "dominant_coefficients": [-0.5, 1.0, -0.5],
            "parent_coefficients": [0.5, -1.0, 0.5],
            "raw_same_sign_fraction_min": RAW_SAME_SIGN_FRACTION_MIN,
            "coefficient_explained_opposition_min": COEFFICIENT_EXPLAINED_OPPOSITION_MIN,
            "sign_geometry_used_for_solver": False,
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
        },
        "parents": {
            "stage154_source_head": EXPECTED_STAGE154_SOURCE_HEAD,
            "stage154_run_id": EXPECTED_STAGE154_RUN_ID,
            "stage154_job_id": EXPECTED_STAGE154_JOB_ID,
            "stage154_artifact_id": EXPECTED_STAGE154_ARTIFACT_ID,
        },
        "aggregate": {
            "stage154_record_ok": stage154_record_ok,
            "parent_route_ok": parent_route_ok,
            "raw_same_sign_fraction": metrics["raw_same_sign_fraction"],
            "coefficient_explained_opposition_fraction": metrics["coefficient_explained_opposition_fraction"],
            "raw_magnitude_cosine": metrics["raw_magnitude_cosine"],
            "cancellation_fraction_reconstruction_error": metrics["cancellation_fraction_reconstruction_error"],
            "maximum_identity_or_provenance_error": metrics["maximum_identity_or_provenance_error"],
        },
        "metrics": metrics,
        "scientific_conclusion": conclusion,
        "negative_result_guard": negative_guard,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    np.savez(
        output_dir / "support_sign_geometry.npz",
        support_depth=np.asarray(metrics["support_depth"], dtype=float),
        dominant_coefficients=np.asarray(metrics["dominant_coefficients"], dtype=float),
        parent_coefficients=np.asarray(metrics["parent_coefficients"], dtype=float),
        dominant_raw_support=np.asarray(metrics["dominant_raw_support"], dtype=float),
        parent_raw_support=np.asarray(metrics["parent_raw_support"], dtype=float),
        raw_cross_channel_sign_products=np.asarray(metrics["raw_cross_channel_sign_products"], dtype=int),
        coefficient_cross_channel_sign_products=np.asarray(metrics["coefficient_cross_channel_sign_products"], dtype=int),
        contribution_cross_channel_sign_products=np.asarray(metrics["contribution_cross_channel_sign_products"], dtype=int),
        amplitude_ratio_parent_to_dominant=np.asarray(metrics["amplitude_ratio_parent_to_dominant"], dtype=float),
        predicted_cancellation_fraction=np.asarray(metrics["predicted_cancellation_fraction"], dtype=float),
        observed_cancellation_fraction=np.asarray(metrics["observed_cancellation_fraction"], dtype=float),
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage154-dir", type=Path, required=True)
    parser.add_argument("--stage154-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_stage155(args.stage154_dir, args.stage154_record, args.output_dir)


if __name__ == "__main__":
    main()
