from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = 148
EXPECTED_STAGE147_SOURCE_HEAD = "3dd94ff4b773ee21358a88a25e660c776988406d"
EXPECTED_STAGE147_RUN_ID = 32358081943
EXPECTED_STAGE147_JOB_ID = 96391517821
EXPECTED_STAGE147_ARTIFACT_ID = 9410888827
EXPECTED_STAGE147_ARTIFACT_SHA256 = "444949e992230e54f224aa254fcf1f4c1743b535aaacd39e38d43d5ef20709c4"
EXPECTED_STAGE147_SUMMARY_SHA256 = "bf8fa6cc06e510fa2f4fbd76726db8e03faf5237bc95cc897fbfa7d42cf0ba5b"
EXPECTED_STAGE147_PAYLOAD_SHA256 = "a77b8bc80a291fe9e440d30f7c09e768964e86b9e873a9ab9890074caf8d6c2d"
EXPECTED_STAGE147_DECISION = "stage147_material_bilateral_curvature_sign_reversal_stage148_sample_scale_curvature_alternation_audit"

PROVENANCE_MATCH_MAX = 1.0e-12
IDENTITY_CLOSURE_MAX = 1.0e-12
ALTERNATING_ENERGY_SHARE_MIN = 0.75
COARSE_CENTER_RETENTION_MAX = 0.50

NONFINITE = "stage148_nonfinite_blocker"
STAGE147_RECORD_BLOCKER = "stage148_stage147_record_blocker"
PARENT_ROUTE_BLOCKER = "stage148_parent_route_blocker"
PROVENANCE_BLOCKER = "stage148_parent_provenance_blocker"
IDENTITY_BLOCKER = "stage148_channel_identity_closure_blocker"
SAMPLE_SCALE = "stage148_sample_scale_alternation_dominant_stage149_channel_scale_separation_audit"
MULTISCALE = "stage148_multiscale_alternation_persists_stage149_extended_scale_audit"
WEAK_ALTERNATION = "stage148_weak_alternating_mode_stage149_shape_residual_audit"


def validate_stage148_design(
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
    alternating_energy_share_min=ALTERNATING_ENERGY_SHARE_MIN,
    coarse_center_retention_max=COARSE_CENTER_RETENTION_MAX,
    solver_rerun=False,
    solver_endpoint_advanced=False,
    cross_knudsen_extension_permitted=False,
    benchmark_or_validation_claim_permitted=False,
    curvature_scale_used_for_solver=False,
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
        "alternating_energy_share_min": ALTERNATING_ENERGY_SHARE_MIN,
        "coarse_center_retention_max": COARSE_CENTER_RETENTION_MAX,
        "solver_rerun": False,
        "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False,
        "benchmark_or_validation_claim_permitted": False,
        "curvature_scale_used_for_solver": False,
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
            raise ValueError(f"Stage 148 frozen-design violation: {key}={got[key]!r}, expected {value!r}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _check_stage147_record(record: dict) -> bool:
    return bool(
        record.get("stage") == 147
        and record.get("source_head") == EXPECTED_STAGE147_SOURCE_HEAD
        and record.get("workflow_run_id") == EXPECTED_STAGE147_RUN_ID
        and record.get("workflow_job_id") == EXPECTED_STAGE147_JOB_ID
        and record.get("artifact_id") == EXPECTED_STAGE147_ARTIFACT_ID
        and record.get("artifact_sha256") == EXPECTED_STAGE147_ARTIFACT_SHA256
        and record.get("summary_sha256") == EXPECTED_STAGE147_SUMMARY_SHA256
        and record.get("dual_channel_neighborhood_sha256") == EXPECTED_STAGE147_PAYLOAD_SHA256
        and record.get("decision") == EXPECTED_STAGE147_DECISION
        and record.get("workflow_status") == "completed"
        and record.get("workflow_conclusion") == "success"
    )


def _secant(xl: float, yl: float, xr: float, yr: float, xc: float) -> float:
    return float(yl + (yr - yl) * (xc - xl) / (xr - xl))


def _alternating_mode_metrics(values: np.ndarray) -> dict:
    q = np.asarray(values, dtype=float)
    if q.shape != (3,) or not np.isfinite(q).all():
        raise ValueError("Stage 148 alternating-mode input must be finite shape (3,)")
    template = np.array([-1.0, 1.0, -1.0])
    coeff = float(np.dot(q, template) / np.dot(template, template))
    fit = coeff * template
    denom = float(np.dot(q, q))
    if denom <= 0.0:
        return {
            "coefficient": 0.0,
            "energy_share": 0.0,
            "relative_l2_residual": 0.0,
            "sign_agreement_fraction": 1.0,
            "fit": fit.tolist(),
        }
    return {
        "coefficient": coeff,
        "energy_share": float(np.dot(fit, fit) / denom),
        "relative_l2_residual": float(np.linalg.norm(q - fit) / np.linalg.norm(q)),
        "sign_agreement_fraction": float(np.mean(np.sign(q) == np.sign(fit))),
        "fit": fit.tolist(),
    }


def sample_scale_curvature_metrics(
    depth: np.ndarray,
    dominant_signed: np.ndarray,
    parent_signed: np.ndarray,
    complement_signed: np.ndarray,
    inherited_dominant_curvature: np.ndarray,
    inherited_parent_curvature: np.ndarray,
    inherited_complement_deficit: np.ndarray,
) -> dict:
    x = np.asarray(depth, dtype=float)
    dominant = np.asarray(dominant_signed, dtype=float)
    parent = np.asarray(parent_signed, dtype=float)
    complement = np.asarray(complement_signed, dtype=float)
    inherited_d = np.asarray(inherited_dominant_curvature, dtype=float)
    inherited_p = np.asarray(inherited_parent_curvature, dtype=float)
    inherited_c = np.asarray(inherited_complement_deficit, dtype=float)
    if not (x.shape == dominant.shape == parent.shape == complement.shape == (5,)):
        raise ValueError("Stage 148 requires the exact five-point Stage-147 profiles")
    if not (inherited_d.shape == inherited_p.shape == inherited_c.shape == (3,)):
        raise ValueError("Stage 148 requires the exact three-point Stage-147 curvature arrays")
    if not all(np.isfinite(a).all() for a in (x, dominant, parent, complement, inherited_d, inherited_p, inherited_c)):
        raise ValueError("Stage 148 requires finite inputs")
    dx = np.diff(x)
    if not np.all(dx > 0.0):
        raise ValueError("Stage 148 depth samples must be strictly increasing")
    if float(np.max(np.abs(dx - dx[0]))) > PROVENANCE_MATCH_MAX:
        raise ValueError("Stage 148 requires the inherited equal-depth sampling")

    idx = np.array([1, 2, 3], dtype=int)
    fine_d = []
    fine_p = []
    fine_c = []
    for i in idx:
        dsec = _secant(x[i - 1], dominant[i - 1], x[i + 1], dominant[i + 1], x[i])
        psec = _secant(x[i - 1], parent[i - 1], x[i + 1], parent[i + 1], x[i])
        csec = _secant(x[i - 1], complement[i - 1], x[i + 1], complement[i + 1], x[i])
        fine_d.append(-(dsec - dominant[i]))
        fine_p.append(psec - parent[i])
        fine_c.append(csec - complement[i])
    fine_d = np.asarray(fine_d, dtype=float)
    fine_p = np.asarray(fine_p, dtype=float)
    fine_c = np.asarray(fine_c, dtype=float)

    inherited_match_error = float(max(
        np.max(np.abs(fine_d - inherited_d)),
        np.max(np.abs(fine_p - inherited_p)),
        np.max(np.abs(fine_c - inherited_c)),
    ))
    fine_identity_closure = float(np.max(np.abs((fine_d + fine_p) - fine_c)))

    center = 2
    dcoarse = -(_secant(x[0], dominant[0], x[4], dominant[4], x[center]) - dominant[center])
    pcoarse = _secant(x[0], parent[0], x[4], parent[4], x[center]) - parent[center]
    ccoarse = _secant(x[0], complement[0], x[4], complement[4], x[center]) - complement[center]
    coarse_identity_closure = float(abs((dcoarse + pcoarse) - ccoarse))

    tiny = np.finfo(float).tiny
    d_retention = float(abs(dcoarse) / max(abs(fine_d[1]), tiny))
    p_retention = float(abs(pcoarse) / max(abs(fine_p[1]), tiny))
    c_retention = float(abs(ccoarse) / max(abs(fine_c[1]), tiny))

    d_alt = _alternating_mode_metrics(fine_d)
    p_alt = _alternating_mode_metrics(fine_p)
    c_alt = _alternating_mode_metrics(fine_c)

    return {
        "fine_depths": x[idx].tolist(),
        "fine_dominant_projected_curvature": fine_d.tolist(),
        "fine_parent_projected_curvature": fine_p.tolist(),
        "fine_complement_secant_deficit": fine_c.tolist(),
        "fine_curvature_sign_sequence": np.sign(fine_c).astype(int).tolist(),
        "inherited_curvature_match_error": inherited_match_error,
        "fine_identity_closure": fine_identity_closure,
        "dominant_alternating_mode": d_alt,
        "parent_alternating_mode": p_alt,
        "complement_alternating_mode": c_alt,
        "coarse_center_dominant_projected_curvature": float(dcoarse),
        "coarse_center_parent_projected_curvature": float(pcoarse),
        "coarse_center_complement_secant_deficit": float(ccoarse),
        "dominant_coarse_to_fine_center_retention": d_retention,
        "parent_coarse_to_fine_center_retention": p_retention,
        "complement_coarse_to_fine_center_retention": c_retention,
        "coarse_center_identity_closure": coarse_identity_closure,
        "coarse_center_same_sign_as_fine_center": bool(ccoarse * fine_c[1] > 0.0),
        "maximum_identity_or_provenance_error": max(inherited_match_error, fine_identity_closure, coarse_identity_closure),
    }


def classify_sample_scale_curvature(
    *,
    metrics: dict,
    stage147_record_ok: bool = True,
    parent_route_ok: bool = True,
    finite: bool = True,
) -> str:
    numeric = [
        metrics.get("inherited_curvature_match_error", np.nan),
        metrics.get("maximum_identity_or_provenance_error", np.nan),
        metrics.get("complement_coarse_to_fine_center_retention", np.nan),
        metrics.get("complement_alternating_mode", {}).get("energy_share", np.nan),
        metrics.get("complement_alternating_mode", {}).get("sign_agreement_fraction", np.nan),
    ]
    if not finite or not np.isfinite(numeric).all():
        return NONFINITE
    if not stage147_record_ok:
        return STAGE147_RECORD_BLOCKER
    if not parent_route_ok:
        return PARENT_ROUTE_BLOCKER
    if float(metrics["inherited_curvature_match_error"]) > PROVENANCE_MATCH_MAX:
        return PROVENANCE_BLOCKER
    if float(metrics["maximum_identity_or_provenance_error"]) > IDENTITY_CLOSURE_MAX:
        return IDENTITY_BLOCKER
    alt = metrics["complement_alternating_mode"]
    strong_alt = (
        float(alt["energy_share"]) >= ALTERNATING_ENERGY_SHARE_MIN
        and float(alt["sign_agreement_fraction"]) == 1.0
    )
    if strong_alt and float(metrics["complement_coarse_to_fine_center_retention"]) <= COARSE_CENTER_RETENTION_MAX:
        return SAMPLE_SCALE
    if strong_alt:
        return MULTISCALE
    return WEAK_ALTERNATION


def run_stage148(stage147_dir: Path, stage147_record: Path, output_dir: Path) -> dict:
    validate_stage148_design()
    summary147 = _load_json(stage147_dir / "summary.json")
    record147 = _load_json(stage147_record)
    stage147_record_ok = _check_stage147_record(record147)
    parent_route_ok = bool(
        summary147.get("stage") == 147
        and summary147.get("decision") == EXPECTED_STAGE147_DECISION
        and summary147.get("aggregate", {}).get("bilateral_material_channel_sign_reversal") is True
    )

    with np.load(stage147_dir / "dual_channel_neighborhood.npz") as data:
        metrics = sample_scale_curvature_metrics(
            data["five_point_depth"],
            data["five_point_dominant_signed"],
            data["five_point_parent_signed"],
            data["five_point_complement_signed"],
            data["dominant_projected_curvature"],
            data["parent_projected_curvature"],
            data["complement_secant_deficit"],
        )

    finite = bool(np.isfinite([
        metrics["inherited_curvature_match_error"],
        metrics["maximum_identity_or_provenance_error"],
        metrics["complement_coarse_to_fine_center_retention"],
        metrics["complement_alternating_mode"]["energy_share"],
    ]).all())
    decision = classify_sample_scale_curvature(
        metrics=metrics,
        stage147_record_ok=stage147_record_ok,
        parent_route_ok=parent_route_ok,
        finite=finite,
    )

    if decision == SAMPLE_SCALE:
        conclusion = (
            "The inherited [-,+,-] complement-curvature sequence is dominated by the one-sample alternating mode, "
            "while the fixed two-cell center secant retains less than half of the one-cell center deficit. This supports "
            "a sample-scale curvature alternation in the frozen artifact rather than a broad resolved sub-lobe. The "
            "channel-specific coarse retentions remain diagnostic and justify a next artifact-only channel-scale separation audit."
        )
    elif decision == MULTISCALE:
        conclusion = (
            "The alternating mode is strong, but the fixed two-cell center secant retains more than half of the one-cell "
            "center deficit. The curvature therefore persists beyond a purely one-sample feature and requires an extended-scale audit."
        )
    elif decision == WEAK_ALTERNATION:
        conclusion = (
            "The three inherited curvature samples do not place at least 75% of their energy in the fixed [-,+,-] alternating mode. "
            "The apparent sign alternation should therefore be treated as shape residual structure rather than a dominant sampled mode."
        )
    else:
        conclusion = "Stage 148 is blocked by frozen provenance, identity, or finite-value guards; no scientific interpretation is promoted."

    configuration = {
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
        "alternating_energy_share_min": ALTERNATING_ENERGY_SHARE_MIN,
        "coarse_center_retention_max": COARSE_CENTER_RETENTION_MAX,
        "curvature_scale_used_for_solver": False,
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
    result = {
        "stage": STAGE,
        "finite": finite,
        "decision": decision,
        "configuration": configuration,
        "parents": {
            "stage147_source_head": EXPECTED_STAGE147_SOURCE_HEAD,
            "stage147_run_id": EXPECTED_STAGE147_RUN_ID,
            "stage147_job_id": EXPECTED_STAGE147_JOB_ID,
            "stage147_artifact_id": EXPECTED_STAGE147_ARTIFACT_ID,
        },
        "aggregate": {
            "stage147_record_ok": stage147_record_ok,
            "parent_route_ok": parent_route_ok,
            "inherited_curvature_match_error": metrics["inherited_curvature_match_error"],
            "maximum_identity_or_provenance_error": metrics["maximum_identity_or_provenance_error"],
            "complement_alternating_energy_share": metrics["complement_alternating_mode"]["energy_share"],
            "complement_coarse_to_fine_center_retention": metrics["complement_coarse_to_fine_center_retention"],
        },
        "metrics": metrics,
        "scientific_conclusion": conclusion,
        "negative_result_guard": (
            "Stage 28 remains a failed MUSCL endpoint and Stage 90 remains nonconverged. Stage 148 is an artifact-only "
            "sample-scale curvature audit; alternating-mode energy and one-cell/two-cell secant ratios are diagnostics, not solver "
            "parameters. No physical, collision/source, floor, wall, reconstruction, transport, limiter, normalization, "
            "source-relaxation, or velocity-quadrature parameter is retuned; no measured curvature scale is fed back into the solver, "
            "no solver endpoint or cross-Knudsen extension is advanced, and no benchmark or validation claim is permitted."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        output_dir / "sample_scale_curvature.npz",
        fine_depth=np.asarray(metrics["fine_depths"], dtype=float),
        fine_dominant_curvature=np.asarray(metrics["fine_dominant_projected_curvature"], dtype=float),
        fine_parent_curvature=np.asarray(metrics["fine_parent_projected_curvature"], dtype=float),
        fine_complement_deficit=np.asarray(metrics["fine_complement_secant_deficit"], dtype=float),
        dominant_alternating_fit=np.asarray(metrics["dominant_alternating_mode"]["fit"], dtype=float),
        parent_alternating_fit=np.asarray(metrics["parent_alternating_mode"]["fit"], dtype=float),
        complement_alternating_fit=np.asarray(metrics["complement_alternating_mode"]["fit"], dtype=float),
        coarse_center_components=np.asarray([
            metrics["coarse_center_dominant_projected_curvature"],
            metrics["coarse_center_parent_projected_curvature"],
            metrics["coarse_center_complement_secant_deficit"],
        ], dtype=float),
        coarse_to_fine_retention=np.asarray([
            metrics["dominant_coarse_to_fine_center_retention"],
            metrics["parent_coarse_to_fine_center_retention"],
            metrics["complement_coarse_to_fine_center_retention"],
        ], dtype=float),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage147-dir", type=Path, required=True)
    parser.add_argument("--stage147-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_stage148(args.stage147_dir, args.stage147_record, args.output_dir)
    print(json.dumps({
        "stage": result["stage"],
        "decision": result["decision"],
        "aggregate": result["aggregate"],
        "metrics": result["metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
