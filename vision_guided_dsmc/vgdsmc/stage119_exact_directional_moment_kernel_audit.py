from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import stage110_same_sign_slope_asymmetry_audit as s110
from . import stage114_wall_distance_conditioned_velocity_quadrature_audit as s114

STAGE118_RUN_ID = 31781982668
STAGE118_JOB_ID = 94709528690
STAGE118_ARTIFACT_ID = 9220901709
STAGE118_ARTIFACT_SHA256 = "812a82293d519639b43a3eaf419b14809fbe89e1bd41c81b05f4f7530a242a2d"
STAGE118_SUMMARY_SHA256 = "6ee1207ecf95ee3d945e63163ff826415bc14b81c3fef688a361a99a81b96a37"
STAGE118_PROFILES_SHA256 = "9710dea5a44e9c0e93c32a92b68b5953534da15a530fe71cccb73f4c36f41a64"
STAGE118_SOURCE_HEAD = "bf7f626f984390f64146c709bf2a453103aa0ec1"
STAGE118_DECISION = "stage118_energy_role_weighting_incomplete_stage119_exact_directional_moment_kernel_audit"

GRID = s114.GRID
KNUDSEN = s114.KNUDSEN
COLD_HOT_RATIO = s114.COLD_HOT_RATIO
RULE = s114.RULE
RADIAL_SCALE = s114.RADIAL_SCALE
LIMITER = s114.LIMITER
BOUNDARY_SLOPE = s114.BOUNDARY_SLOPE
SOURCE_RELAXATION = s114.SOURCE_RELAXATION
TOLERANCE = s114.TOLERANCE
CORRECTION_FLOOR = s114.CORRECTION_FLOOR
DIAGNOSTIC_STEPS = s114.DIAGNOSTIC_STEPS
WALL_BAND_CELLS = s114.WALL_BAND_CELLS
DOMINANT_RADIAL_SHELL = s114.DOMINANT_RADIAL_SHELL
RADIAL_SHELL_COUNT = s114.RADIAL_SHELL_COUNT
RADIAL_NODES_PER_SHELL = s114.RADIAL_NODES_PER_SHELL
ANGULAR_SECTORS = s114.ANGULAR_SECTORS
PAIR_SECTORS = (5, 6)
BANDS = ("near_1_4", "mid_5_14", "inner_15_28")
RAW_PARENT_CLOSURE_TOLERANCE = 1.0e-12
COMMON_COSINE_MIN = 0.95
COMMON_OVERLAP_MIN = 0.90

ALIGNED = "stage119_exact_x_heat_flux_kernel_aligns_stage120_role_weighted_spatial_colocation_audit"
INCOMPLETE = "stage119_exact_x_heat_flux_kernel_incomplete_stage120_kernel_residual_velocity_cell_audit"
NONFINITE = "stage119_nonfinite_exact_directional_kernel_blocker_without_retuning"
CLOSURE_BLOCKER = "stage119_stage118_raw_profile_reconstruction_blocker_without_retuning"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_stage119_design(**overrides: object) -> None:
    frozen = {
        "grid": GRID,
        "kn0": KNUDSEN,
        "cold_hot_ratio": COLD_HOT_RATIO,
        "rule": RULE,
        "radial_scale": RADIAL_SCALE,
        "limiter": LIMITER,
        "boundary_slope": BOUNDARY_SLOPE,
        "source_relaxation": SOURCE_RELAXATION,
        "tolerance": TOLERANCE,
        "correction_floor": CORRECTION_FLOOR,
        "diagnostic_steps": DIAGNOSTIC_STEPS,
        "wall_band_cells": WALL_BAND_CELLS,
        "dominant_radial_shell": DOMINANT_RADIAL_SHELL,
        "radial_shell_count": RADIAL_SHELL_COUNT,
        "radial_nodes_per_shell": RADIAL_NODES_PER_SHELL,
        "angular_sectors": ANGULAR_SECTORS,
        "pair_sectors": PAIR_SECTORS,
        "raw_parent_closure_tolerance": RAW_PARENT_CLOSURE_TOLERANCE,
        "common_cosine_min": COMMON_COSINE_MIN,
        "common_overlap_min": COMMON_OVERLAP_MIN,
        "stage67_run_id": s110.STAGE67_RUN_ID,
        "stage111_run_id": s114.STAGE111_RUN_ID,
        "stage118_run_id": STAGE118_RUN_ID,
    }
    if any(k not in frozen or frozen[k] != v for k, v in overrides.items()):
        raise ValueError(
            "Stage 119 is fixed to the exact x-directed reduced heat-flux moment kernel and may not retune retained or failed physical/numerical parameters"
        )
    if RULE != (40, 96) or RADIAL_NODES_PER_SHELL != 10 or PAIR_SECTORS != (5, 6):
        raise ValueError("Stage 119 requires the exact 40x96 rule, ten shell-1 radial nodes, and sectors 5+6")


def _load_stage118(parent_dir: str | Path, record_path: str | Path):
    root = Path(parent_dir)
    expected = {
        "summary.json": STAGE118_SUMMARY_SHA256,
        "distribution_role_weighted_profiles.npz": STAGE118_PROFILES_SHA256,
    }
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Stage-118 checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    if summary.get("stage") != 118 or summary.get("decision") != STAGE118_DECISION or summary.get("finite") is not True:
        raise ValueError("Stage-118 artifact does not authorize Stage 119")
    checks = (
        record.get("stage") == 118,
        record.get("decision") == STAGE118_DECISION,
        record.get("source_head") == STAGE118_SOURCE_HEAD,
        record.get("workflow_status") == "completed",
        record.get("workflow_conclusion") == "success",
        record.get("workflow_run_id") == STAGE118_RUN_ID,
        record.get("workflow_job_id") == STAGE118_JOB_ID,
        record.get("artifact_id") == STAGE118_ARTIFACT_ID,
        record.get("artifact_sha256") == STAGE118_ARTIFACT_SHA256,
        record.get("summary_sha256") == STAGE118_SUMMARY_SHA256,
        record.get("distribution_role_weighted_profiles_sha256") == STAGE118_PROFILES_SHA256,
        record.get("tests", {}).get("passed") == 4,
        record.get("tests", {}).get("failed") == 0,
    )
    if not all(checks):
        raise ValueError("Committed Stage-118 provenance does not authorize Stage 119")
    with np.load(root / "distribution_role_weighted_profiles.npz") as data:
        needed = {"raw_phi", "raw_psi", "phi_energy_role", "psi_energy_role", "node_speed_mean"}
        if not needed.issubset(data.files):
            raise ValueError("Stage-118 profile payload is incomplete")
        arrays = {name: np.asarray(data[name], dtype=np.float64).copy() for name in needed}
    for name in ("raw_phi", "raw_psi", "phi_energy_role", "psi_energy_role"):
        if arrays[name].shape != (3, 10) or not np.isfinite(arrays[name]).all() or np.any(arrays[name] < 0.0):
            raise ValueError(f"Stage-118 array {name} is invalid")
    return summary, record, arrays


def radial_node_indices_within_shell(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    speed = np.hypot(np.asarray(vx, dtype=np.float64), np.asarray(vy, dtype=np.float64))
    if speed.shape != (960,) or not np.isfinite(speed).all():
        raise ValueError("Stage 119 requires the exact 960-point shell-1 support")
    order = np.argsort(speed, kind="stable")
    labels = np.empty(960, dtype=np.int16)
    labels[order] = np.repeat(np.arange(10, dtype=np.int16), 96)
    for j in range(10):
        q = speed[labels == j]
        if q.size != 96 or float(q.max() - q.min()) > 1.0e-12 * max(float(q.mean()), 1.0):
            raise ValueError("Stage-119 radial-node grouping mixed distinct speeds")
    return labels


def _x_same_sign_change_pointwise(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    w = WALL_BAND_CELLS
    center = f[w:-w, w:-w]
    left = center - f[w:-w, w - 1 : -w - 1]
    right = f[w:-w, w + 1 : -w + 1] - center
    same = ((left > 0.0) & (right > 0.0)) | ((left < 0.0) & (right < 0.0))
    return np.where(same, 0.5 * np.abs(np.abs(left) - np.abs(right)), 0.0)


def _normalize(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64)
    total = float(x.sum())
    if x.shape != (10,) or not np.isfinite(x).all() or np.any(x < 0.0) or total <= 0.0:
        raise ValueError("Invalid Stage-119 radial profile")
    return x / total


def _profile_metrics(phi: np.ndarray, psi: np.ndarray) -> dict[str, float | list[int]]:
    p = _normalize(phi)
    q = _normalize(psi)
    d = p - q
    return {
        "profile_cosine": float(np.dot(p, q) / max(float(np.linalg.norm(p) * np.linalg.norm(q)), 1.0e-300)),
        "overlap_coefficient": float(np.minimum(p, q).sum()),
        "total_variation_distance": float(0.5 * np.abs(d).sum()),
        "transition_boundaries": [int(j) for j in range(9) if d[j] != 0.0 and d[j + 1] != 0.0 and d[j] * d[j + 1] < 0.0],
        "phi_centroid_node": float(np.dot(p, np.arange(10, dtype=np.float64))),
        "psi_centroid_node": float(np.dot(q, np.arange(10, dtype=np.float64))),
    }


def stage119_decision(metrics: dict[str, dict[str, object]], finite: bool, raw_closure: float) -> str:
    if not finite:
        return NONFINITE
    if raw_closure > RAW_PARENT_CLOSURE_TOLERANCE:
        return CLOSURE_BLOCKER
    aligned = all(
        float(metrics[band]["profile_cosine"]) >= COMMON_COSINE_MIN
        and float(metrics[band]["overlap_coefficient"]) >= COMMON_OVERLAP_MIN
        for band in BANDS
    )
    return ALIGNED if aligned else INCOMPLETE


def run(stage67_dir: str | Path, stage111_dir: str | Path, stage118_dir: str | Path, stage118_record_path: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage119_design(**design)
    s67_summary, distributions = s110._load_stage67(stage67_dir)
    s111_summary, maps = s114._load_stage111(stage111_dir)
    s118_summary, s118_record, parent = _load_stage118(stage118_dir, stage118_record_path)

    profiles: dict[str, list[np.ndarray]] = {"raw_phi": [], "raw_psi": [], "exact_phi": [], "exact_psi": []}
    bands = s114.wall_distance_band_masks()
    with np.load(distributions) as data:
        vx = np.asarray(data["vx"], dtype=np.float64)
        vy = np.asarray(data["vy"], dtype=np.float64)
        weight = np.asarray(data["weight"], dtype=np.float64)
        shell = s110._radial_shell_indices(vx, vy) == DOMINANT_RADIAL_SHELL
        svx, svy, sw = vx[shell], vy[shell], weight[shell]
        sector = s114.angular_sector_indices(svx, svy)
        node = radial_node_indices_within_shell(svx, svy)
        pair = np.isin(sector, PAIR_SECTORS)
        if any(np.count_nonzero((node == j) & pair) != 24 for j in range(10)):
            raise ValueError("Each Stage-119 radial node must contain exactly 24 points in sectors 5+6")
        exact_kernel = {
            "phi": np.abs(svx) * (svx * svx + svy * svy),
            "psi": np.abs(svx),
        }
        for name in ("phi", "psi"):
            change = _x_same_sign_change_pointwise(np.asarray(data[name], dtype=np.float64)[..., shell])
            density = change * sw[None, None, :] * maps[f"{name}_growth_amplitude"][..., None]
            for band in BANDS:
                cells = density[bands[band]]
                raw = np.array([cells[:, (node == j) & pair].sum() for j in range(10)], dtype=np.float64)
                exact = np.array([
                    np.sum(cells[:, (node == j) & pair] * exact_kernel[name][(node == j) & pair][None, :])
                    for j in range(10)
                ], dtype=np.float64)
                profiles[f"raw_{name}"].append(_normalize(raw))
                profiles[f"exact_{name}"].append(_normalize(exact))

    raw_phi, raw_psi = np.asarray(profiles["raw_phi"]), np.asarray(profiles["raw_psi"])
    exact_phi, exact_psi = np.asarray(profiles["exact_phi"]), np.asarray(profiles["exact_psi"])
    raw_closures = {
        "phi": float(np.linalg.norm(raw_phi - parent["raw_phi"]) / max(float(np.linalg.norm(parent["raw_phi"])), 1.0e-300)),
        "psi": float(np.linalg.norm(raw_psi - parent["raw_psi"]) / max(float(np.linalg.norm(parent["raw_psi"])), 1.0e-300)),
    }
    max_raw_closure = max(raw_closures.values())

    metrics: dict[str, dict[str, object]] = {}
    for i, band in enumerate(BANDS):
        exact = _profile_metrics(exact_phi[i], exact_psi[i])
        approx = _profile_metrics(parent["phi_energy_role"][i], parent["psi_energy_role"][i])
        exact["stage118_role_weighted_profile_cosine"] = float(approx["profile_cosine"])
        exact["stage118_role_weighted_total_variation"] = float(approx["total_variation_distance"])
        exact["cosine_gain_over_stage118_role_weighting"] = float(exact["profile_cosine"] - approx["profile_cosine"])
        exact["tv_reduction_fraction_over_stage118_role_weighting"] = float((approx["total_variation_distance"] - exact["total_variation_distance"]) / max(float(approx["total_variation_distance"]), 1.0e-300))
        metrics[band] = exact

    finite = bool(np.isfinite(raw_phi).all() and np.isfinite(raw_psi).all() and np.isfinite(exact_phi).all() and np.isfinite(exact_psi).all() and np.isfinite(max_raw_closure))
    decision = stage119_decision(metrics, finite, max_raw_closure)
    aggregate = {
        "minimum_exact_profile_cosine": float(min(float(metrics[b]["profile_cosine"]) for b in BANDS)),
        "minimum_exact_overlap": float(min(float(metrics[b]["overlap_coefficient"]) for b in BANDS)),
        "maximum_exact_total_variation": float(max(float(metrics[b]["total_variation_distance"]) for b in BANDS)),
        "minimum_cosine_gain_over_stage118_role_weighting": float(min(float(metrics[b]["cosine_gain_over_stage118_role_weighting"]) for b in BANDS)),
        "minimum_tv_reduction_fraction_over_stage118_role_weighting": float(min(float(metrics[b]["tv_reduction_fraction_over_stage118_role_weighting"]) for b in BANDS)),
        "maximum_stage118_raw_profile_reconstruction_relative_l2": float(max_raw_closure),
    }

    cfg = {
        "grid": list(GRID), "kn0": KNUDSEN, "cold_hot_ratio": COLD_HOT_RATIO, "rule": list(RULE),
        "radial_scale": RADIAL_SCALE, "limiter": LIMITER, "boundary_slope": BOUNDARY_SLOPE,
        "source_relaxation": SOURCE_RELAXATION, "tolerance": TOLERANCE, "correction_floor": CORRECTION_FLOOR,
        "diagnostic_steps": DIAGNOSTIC_STEPS, "wall_band_cells": WALL_BAND_CELLS,
        "dominant_radial_shell": DOMINANT_RADIAL_SHELL, "radial_shell_count": RADIAL_SHELL_COUNT,
        "radial_nodes_per_shell": RADIAL_NODES_PER_SHELL, "angular_sectors": ANGULAR_SECTORS,
        "pair_sectors": list(PAIR_SECTORS), "phi_exact_directional_kernel": "abs(c_x)*(c_x^2+c_y^2)",
        "psi_exact_directional_kernel": "abs(c_x)", "common_cosine_min": COMMON_COSINE_MIN,
        "common_overlap_min": COMMON_OVERLAP_MIN, "raw_parent_closure_tolerance": RAW_PARENT_CLOSURE_TOLERANCE,
        "stage67_run_id": s110.STAGE67_RUN_ID, "stage111_run_id": s114.STAGE111_RUN_ID,
        "stage118_run_id": STAGE118_RUN_ID, "full_solver_endpoint_rerun": False,
        "physical_parameter_retuning": False, "collision_parameter_retuning": False,
        "correction_floor_retuning": False, "positivity_floor_retuning": False,
        "source_relaxation_retuning": False, "transport_parameter_retuning": False,
        "wall_model_retuning": False, "normalization_retuning": False, "limiter_retuning": False,
        "velocity_quadrature_retuning": False, "failed_muscl_endpoint_rehabilitated": False,
        "one_sided_boundary_slope_promoted": False, "cross_knudsen_extension_permitted": False,
        "validation_claim_permitted": False, "solver_endpoint_claim_permitted": False,
    }
    summary = {
        "stage": 119,
        "configuration": cfg,
        "stage67_authorization": {"stage": s67_summary["stage"], "decision": s67_summary["decision"]},
        "stage111_authorization": {"stage": s111_summary["stage"], "decision": s111_summary["decision"]},
        "stage118_authorization": {"decision": s118_summary["decision"], "workflow_run_id": STAGE118_RUN_ID, "workflow_job_id": STAGE118_JOB_ID, "artifact_id": STAGE118_ARTIFACT_ID, "record_source_head": s118_record["source_head"]},
        "finite": finite,
        "raw_profile_reconstruction_relative_l2": raw_closures,
        "metrics": metrics,
        "aggregate": aggregate,
        "decision": decision,
        "scientific_conclusion": "Stage 119 replaces the Stage-118 radial-speed proxy with the exact pointwise magnitude kernel of the x-directed reduced heat-flux moment: |c_x|(c_x^2+c_y^2) for phi and |c_x| for psi, up to the common factor 1/2. Alignment would support a kinematic moment-kernel explanation; incomplete alignment would show that even the exact directional energy-flux kernel is insufficient. Neither outcome establishes MUSCL-instability causality, solver convergence, heat-flux improvement, benchmark accuracy, or validation.",
        "negative_result_guard": "No model, wall, reconstruction, source, transport, floor, normalization, velocity-grid, or failed MUSCL parameter is changed. The absolute c_x factor is fixed by the x-directed moment magnitude rather than fitted. Stage 90 remains nonconverged, Stage 28 remains failed, and Stage 89 remains unpromoted."
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(out / "exact_directional_moment_kernel_profiles.npz", raw_phi=raw_phi, raw_psi=raw_psi, exact_phi=exact_phi, exact_psi=exact_psi, stage118_phi_energy_role=parent["phi_energy_role"], stage118_psi_energy_role=parent["psi_energy_role"], node_speed_mean=parent["node_speed_mean"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage67-dir", required=True)
    parser.add_argument("--stage111-dir", required=True)
    parser.add_argument("--stage118-dir", required=True)
    parser.add_argument("--stage118-record", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run(args.stage67_dir, args.stage111_dir, args.stage118_dir, args.stage118_record, args.output_dir)


if __name__ == "__main__":
    main()
