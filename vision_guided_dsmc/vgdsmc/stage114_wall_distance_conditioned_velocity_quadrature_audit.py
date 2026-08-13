from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import stage110_same_sign_slope_asymmetry_audit as s110
from . import stage111_axis_conditioned_asymmetry_audit as s111

STAGE111_RUN_ID = 31590035358
STAGE111_JOB_ID = 94092631513
STAGE111_ARTIFACT_ID = 9149082510
STAGE111_ARTIFACT_SHA256 = "a83c09039e956a47f4d18a86db5f2541fe622b9405bb9b09266c4c2d54f1cd7e"
STAGE111_SUMMARY_SHA256 = "93cff62b97e29e7ad5600b7b0c26b5b39bd3f71517e70c15447c95be239f5a77"
STAGE111_MAPS_SHA256 = "78b8173d81eaf2523791201690d6464295b8aa65413d9ccb3eb2c2215ac85407"
STAGE111_DECISION = "stage111_x_axis_asymmetry_coupled_stage112_axis_specific_spatial_audit"

STAGE113_RUN_ID = 31637768178
STAGE113_JOB_ID = 94252318752
STAGE113_ARTIFACT_ID = 9163768111
STAGE113_ARTIFACT_SHA256 = "d1a404eec5f7131b93a6bcc65cc35778f294a83f81abcda483b15eabac67ff5c"
STAGE113_SUMMARY_SHA256 = "1a84edaa0a9e9d7cd3f38038289cccdd39283a4364a9415c863c0cf32113f2b8"
STAGE113_PROFILES_SHA256 = "52d5d39e01dd29e249cfdd006a276272864db4ab53f99bea28f15e33e15e222f"
STAGE113_DECISION = "stage113_broad_x_wall_distance_profile_stage114_wall_distance_conditioned_velocity_quadrature_audit"

GRID = s110.GRID
KNUDSEN = s110.KNUDSEN
COLD_HOT_RATIO = s110.COLD_HOT_RATIO
RULE = s110.RULE
RADIAL_SCALE = s110.RADIAL_SCALE
LIMITER = s110.LIMITER
BOUNDARY_SLOPE = s110.BOUNDARY_SLOPE
SOURCE_RELAXATION = s110.SOURCE_RELAXATION
TOLERANCE = s110.TOLERANCE
CORRECTION_FLOOR = s110.CORRECTION_FLOOR
DIAGNOSTIC_STEPS = s110.DIAGNOSTIC_STEPS
WALL_BAND_CELLS = s110.WALL_BAND_CELLS
DOMINANT_RADIAL_SHELL = s110.DOMINANT_RADIAL_SHELL
RADIAL_SHELL_COUNT = s110.RADIAL_SHELL_COUNT
RADIAL_NODES_PER_SHELL = s110.RADIAL_NODES_PER_SHELL
INTERIOR_EXTENT = s110.INTERIOR_EXTENT

ANGULAR_SECTORS = 8
SECTOR_WIDTH_DEGREES = 45.0
NEAR_WALL_DEPTH = 4
BROAD_WALL_DEPTH = 14
SECTOR_LOCALIZATION_SHARE_GUARD = 0.30
SECTOR_DIFFUSE_SHARE_GUARD = 0.25
DIFFUSE_EFFECTIVE_SECTOR_MIN = 4.0
CLOSURE_TOLERANCE = 1.0e-12


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_stage114_design(**overrides: object) -> None:
    frozen = {
        "grid": GRID, "kn0": KNUDSEN, "cold_hot_ratio": COLD_HOT_RATIO, "rule": RULE,
        "radial_scale": RADIAL_SCALE, "limiter": LIMITER, "boundary_slope": BOUNDARY_SLOPE,
        "source_relaxation": SOURCE_RELAXATION, "tolerance": TOLERANCE,
        "correction_floor": CORRECTION_FLOOR, "diagnostic_steps": DIAGNOSTIC_STEPS,
        "wall_band_cells": WALL_BAND_CELLS, "dominant_radial_shell": DOMINANT_RADIAL_SHELL,
        "radial_shell_count": RADIAL_SHELL_COUNT, "radial_nodes_per_shell": RADIAL_NODES_PER_SHELL,
        "angular_sectors": ANGULAR_SECTORS, "sector_width_degrees": SECTOR_WIDTH_DEGREES,
        "near_wall_depth": NEAR_WALL_DEPTH, "broad_wall_depth": BROAD_WALL_DEPTH,
        "sector_localization_share_guard": SECTOR_LOCALIZATION_SHARE_GUARD,
        "sector_diffuse_share_guard": SECTOR_DIFFUSE_SHARE_GUARD,
        "diffuse_effective_sector_min": DIFFUSE_EFFECTIVE_SECTOR_MIN,
        "closure_tolerance": CLOSURE_TOLERANCE, "stage67_run_id": s110.STAGE67_RUN_ID,
        "stage111_run_id": STAGE111_RUN_ID, "stage113_run_id": STAGE113_RUN_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError("Stage 114 is frozen to the exact completed Stage-67 velocity rule, Stage-111 x-axis same-sign decomposition, and Stage-113 broad wall-distance result. Physics, collision/source treatment, floors, source relaxation, transport, wall treatment, normalization, limiter, velocity quadrature, diagnostic window, decision guards, and failed MUSCL parameters may not be retuned.")
    if RULE != (40, 96) or ANGULAR_SECTORS != 8 or RULE[1] % ANGULAR_SECTORS != 0:
        raise ValueError("Stage 114 requires the exact 40x96 rule and eight fixed angular sectors")
    if INTERIOR_EXTENT != 56 or not (0 < NEAR_WALL_DEPTH < BROAD_WALL_DEPTH < INTERIOR_EXTENT // 2 + 1):
        raise ValueError("Stage 114 requires the exact 56-column interior and fixed 4/14-cell wall bands")
    if not (0.0 < SECTOR_DIFFUSE_SHARE_GUARD < SECTOR_LOCALIZATION_SHARE_GUARD < 1.0):
        raise ValueError("Stage-114 sector guards are invalid")
    if DIFFUSE_EFFECTIVE_SECTOR_MIN <= 1.0 or CLOSURE_TOLERANCE != 1.0e-12:
        raise ValueError("Stage-114 diffuse/closure guards are invalid")


def _load_stage111(root: str | Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    root = Path(root)
    expected = {"summary.json": STAGE111_SUMMARY_SHA256, "axis_conditioned_asymmetry_maps.npz": STAGE111_MAPS_SHA256}
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Stage-111 checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 111 or summary.get("decision") != STAGE111_DECISION or summary.get("finite") is not True:
        raise ValueError("Stage-111 artifact does not authorize Stage 114")
    if float(summary.get("max_parent_closure_relative_l2", np.inf)) > CLOSURE_TOLERANCE:
        raise ValueError("Stage-111 decomposition closure is not admissible")
    needed = {"phi_x_same_sign_change_weighted_abs", "psi_x_same_sign_change_weighted_abs", "phi_growth_amplitude", "psi_growth_amplitude", "joint_growth_amplitude"}
    with np.load(root / "axis_conditioned_asymmetry_maps.npz") as data:
        if not needed.issubset(data.files):
            raise ValueError("Stage-111 payload misses required x-axis or growth maps")
        arrays = {name: np.asarray(data[name], dtype=np.float64).copy() for name in needed}
    for name, value in arrays.items():
        if value.shape != (INTERIOR_EXTENT, INTERIOR_EXTENT) or not np.isfinite(value).all() or np.any(value < 0.0):
            raise ValueError(f"Stage-111 map {name} is invalid")
    return summary, arrays


def _load_stage113(root: str | Path, record_path: str | Path) -> tuple[dict[str, object], dict[str, object]]:
    root = Path(root)
    expected = {"summary.json": STAGE113_SUMMARY_SHA256, "x_wall_distance_profiles.npz": STAGE113_PROFILES_SHA256}
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Stage-113 checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    if summary.get("stage") != 113 or summary.get("decision") != STAGE113_DECISION or summary.get("finite") is not True:
        raise ValueError("Stage-113 artifact does not authorize Stage 114")
    if record.get("stage") != 113 or record.get("decision") != STAGE113_DECISION:
        raise ValueError("Committed Stage-113 record does not authorize Stage 114")
    if record.get("workflow_status") != "completed" or record.get("workflow_conclusion") != "success":
        raise ValueError("Committed Stage-113 record is not a successful completed workflow")
    if int(record.get("workflow_run_id", -1)) != STAGE113_RUN_ID or int(record.get("workflow_job_id", -1)) != STAGE113_JOB_ID:
        raise ValueError("Committed Stage-113 workflow provenance mismatch")
    if int(record.get("artifact_id", -1)) != STAGE113_ARTIFACT_ID or record.get("artifact_sha256") != STAGE113_ARTIFACT_SHA256:
        raise ValueError("Committed Stage-113 artifact provenance mismatch")
    if record.get("summary_sha256") != STAGE113_SUMMARY_SHA256 or record.get("x_wall_distance_profiles_sha256") != STAGE113_PROFILES_SHA256:
        raise ValueError("Committed Stage-113 file digest mismatch")
    tests = record.get("tests", {})
    if not isinstance(tests, dict) or tests.get("passed") != 216 or tests.get("failed") != 0:
        raise ValueError("Committed Stage-113 test record is not the exact successful endpoint")
    common = summary.get("metrics", {}).get("common", {})
    if float(common.get("first_4_cumulative_share", 1.0)) >= 0.50:
        raise ValueError("Stage 114 broad route cannot consume a thin Stage-113 wall profile")
    if float(common.get("first_14_cumulative_share", 0.0)) < 0.75 or int(common.get("half_mass_depth_cells", 0)) < 5:
        raise ValueError("Stage 114 requires the completed broad Stage-113 wall profile")
    return summary, record


def angular_sector_indices(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    x = np.asarray(vx, dtype=np.float64); y = np.asarray(vy, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Stage-114 velocity components must be matching vectors")
    angle = np.mod(np.arctan2(y, x) + 2.0 * np.pi, 2.0 * np.pi)
    sector_width = 2.0 * np.pi / ANGULAR_SECTORS
    scaled = angle / sector_width
    nearest_boundary = np.rint(scaled)
    boundary_tolerance = 64.0 * np.finfo(np.float64).eps
    scaled = np.where(
        np.isclose(scaled, nearest_boundary, rtol=0.0, atol=boundary_tolerance),
        nearest_boundary,
        scaled,
    )
    labels = np.floor(scaled).astype(np.int64)
    return np.mod(labels, ANGULAR_SECTORS)


def wall_distance_band_masks() -> dict[str, np.ndarray]:
    x = np.arange(INTERIOR_EXTENT); d = np.minimum(x, INTERIOR_EXTENT - 1 - x) + 1
    near_x = d <= NEAR_WALL_DEPTH; mid_x = (d > NEAR_WALL_DEPTH) & (d <= BROAD_WALL_DEPTH); inner_x = d > BROAD_WALL_DEPTH
    masks = {name: np.broadcast_to(sel[None, :], (INTERIOR_EXTENT, INTERIOR_EXTENT)).copy() for name, sel in (("near_1_4", near_x), ("mid_5_14", mid_x), ("inner_15_28", inner_x))}
    if not np.all(masks["near_1_4"] | masks["mid_5_14"] | masks["inner_15_28"]):
        raise RuntimeError("Stage-114 wall-distance bands do not cover the interior")
    return masks


def _x_sector_change_maps(distribution: np.ndarray, velocity_weight: np.ndarray, sector_index: np.ndarray) -> np.ndarray:
    f = np.asarray(distribution, dtype=np.float64); w = np.asarray(velocity_weight, dtype=np.float64); s = np.asarray(sector_index, dtype=np.int64)
    if f.ndim != 3 or f.shape[:2] != GRID or f.shape[-1] != w.size or s.shape != (w.size,):
        raise ValueError("Stage-114 distribution/weight/sector shapes are inconsistent")
    wb = WALL_BAND_CELLS; ys = slice(wb, f.shape[0] - wb); xs = slice(wb, f.shape[1] - wb); center = f[ys, xs]
    left = center - f[ys, slice(wb - 1, f.shape[1] - wb - 1)]; right = f[ys, slice(wb + 1, f.shape[1] - wb + 1)] - center
    same_sign = ((left > 0.0) & (right > 0.0)) | ((left < 0.0) & (right < 0.0)); base = np.where(same_sign, 0.5 * np.abs(np.abs(left) - np.abs(right)), 0.0)
    out = np.empty((ANGULAR_SECTORS, INTERIOR_EXTENT, INTERIOR_EXTENT), dtype=np.float64)
    for k in range(ANGULAR_SECTORS):
        mask = s == k
        if not np.any(mask): raise ValueError(f"Stage-114 angular sector {k} is empty")
        out[k] = np.sum(base[..., mask] * w[mask][None, None, :], axis=-1)
    return out


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(np.asarray(b, dtype=np.float64)))
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) / max(denom, 1.0e-300))


def _band_sector_metrics(sector_maps: np.ndarray, growth: np.ndarray) -> dict[str, object]:
    maps = np.asarray(sector_maps, dtype=np.float64); g = np.asarray(growth, dtype=np.float64); density = maps * g[None, :, :]
    total_all = float(np.sum(density))
    if not np.isfinite(total_all) or total_all <= 0.0: raise ValueError("Stage-114 conditioned density must have positive finite mass")
    out: dict[str, object] = {}
    for name, mask in wall_distance_band_masks().items():
        raw = np.array([float(np.sum(density[k][mask])) for k in range(ANGULAR_SECTORS)], dtype=np.float64); band_total = float(np.sum(raw)); share = raw / max(band_total, 1.0e-300)
        out[name] = {"total_conditioned_mass": band_total, "total_conditioned_mass_share": float(band_total / total_all), "sector_share": share.tolist(), "maximum_sector_index": int(np.argmax(share)), "maximum_sector_share": float(np.max(share)), "effective_sector_count": float(1.0 / np.sum(share * share))}
    return out


def _same_dominant_sector(phi: dict[str, object], psi: dict[str, object], band: str) -> tuple[bool, int]:
    pb = phi[band]; qb = psi[band]; assert isinstance(pb, dict) and isinstance(qb, dict)
    pi = int(pb["maximum_sector_index"]); qi = int(qb["maximum_sector_index"])
    strong = pi == qi and float(pb["maximum_sector_share"]) >= SECTOR_LOCALIZATION_SHARE_GUARD and float(qb["maximum_sector_share"]) >= SECTOR_LOCALIZATION_SHARE_GUARD
    return strong, pi if pi == qi else -1


def stage114_decision(metrics: dict[str, object], finite: bool, max_closure_relative_l2: float) -> str:
    if not finite or not np.isfinite(max_closure_relative_l2): return "stage114_nonfinite_velocity_quadrature_conditioning_blocker_without_retuning"
    if max_closure_relative_l2 > CLOSURE_TOLERANCE: return "stage114_velocity_sector_closure_blocker_without_retuning"
    phi = metrics["phi"]; psi = metrics["psi"]; assert isinstance(phi, dict) and isinstance(psi, dict)
    near_strong, near_sector = _same_dominant_sector(phi, psi, "near_1_4"); mid_strong, mid_sector = _same_dominant_sector(phi, psi, "mid_5_14")
    if near_strong and mid_strong and near_sector == mid_sector: return "stage114_common_wall_distance_sector_localization_stage115_sector_resolved_radial_node_audit"
    if near_strong or mid_strong: return "stage114_wall_distance_dependent_sector_structure_stage115_sector_transition_audit"
    diffuse = True
    for dist in (phi, psi):
        for band in ("near_1_4", "mid_5_14"):
            b = dist[band]; assert isinstance(b, dict)
            if float(b["maximum_sector_share"]) >= SECTOR_DIFFUSE_SHARE_GUARD or float(b["effective_sector_count"]) < DIFFUSE_EFFECTIVE_SECTOR_MIN: diffuse = False
    if diffuse: return "stage114_angularly_diffuse_within_broad_wall_profile_stage115_radial_node_conditioning_audit"
    return "stage114_mixed_velocity_quadrature_structure_stage115_distribution_specific_audit"


def run_stage114(stage67_artifact_dir: str | Path, stage111_artifact_dir: str | Path, stage113_artifact_dir: str | Path, stage113_record_path: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage114_design(**design)
    stage67_summary, stage67_distributions = s110._load_stage67(stage67_artifact_dir); stage111_summary, stage111_maps = _load_stage111(stage111_artifact_dir); stage113_summary, stage113_record = _load_stage113(stage113_artifact_dir, stage113_record_path)
    with np.load(stage67_distributions) as saved:
        vx = np.asarray(saved["vx"], dtype=np.float64); vy = np.asarray(saved["vy"], dtype=np.float64); weight = np.asarray(saved["weight"], dtype=np.float64); shell_index = s110._radial_shell_indices(vx, vy); shell_mask = shell_index == DOMINANT_RADIAL_SHELL
        shell_vx = vx[shell_mask]; shell_vy = vy[shell_mask]; shell_weight = weight[shell_mask]; sectors = angular_sector_indices(shell_vx, shell_vy)
        sector_point_count = [int(np.count_nonzero(sectors == k)) for k in range(ANGULAR_SECTORS)]
        expected_count = RADIAL_NODES_PER_SHELL * (RULE[1] // ANGULAR_SECTORS)
        if sector_point_count != [expected_count] * ANGULAR_SECTORS: raise ValueError("Stage-114 shell-1 angular sectors do not contain the expected fixed point counts")
        sector_maps: dict[str, np.ndarray] = {}; metrics: dict[str, object] = {}; closures: list[float] = []
        for dist in ("phi", "psi"):
            full = np.asarray(saved[dist], dtype=np.float64); maps = _x_sector_change_maps(full[..., shell_mask], shell_weight, sectors); parent = stage111_maps[f"{dist}_x_same_sign_change_weighted_abs"]; closure = _relative_l2(np.sum(maps, axis=0), parent); closures.append(closure); sector_maps[dist] = maps
            block = _band_sector_metrics(maps, stage111_maps[f"{dist}_growth_amplitude"]); block["parent_x_change_closure_relative_l2"] = closure; metrics[dist] = block
    max_closure = float(max(closures)); finite = all(np.isfinite(v).all() for v in sector_maps.values()) and np.isfinite(max_closure); decision = stage114_decision(metrics, bool(finite), max_closure)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True); sector_centers_deg = (np.arange(ANGULAR_SECTORS, dtype=np.float64) + 0.5) * SECTOR_WIDTH_DEGREES
    np.savez_compressed(out / "wall_distance_velocity_sector_maps.npz", phi_x_sector_change=sector_maps["phi"], psi_x_sector_change=sector_maps["psi"], phi_growth_amplitude=stage111_maps["phi_growth_amplitude"], psi_growth_amplitude=stage111_maps["psi_growth_amplitude"], joint_growth_amplitude=stage111_maps["joint_growth_amplitude"], sector_index=sectors, sector_center_degrees=sector_centers_deg, shell_vx=shell_vx, shell_vy=shell_vy, shell_weight=shell_weight)
    summary = {
        "stage": 114,
        "description": "Frozen wall-distance-conditioned audit of the Stage-111 shell-1 x-axis same-sign change over the unchanged 10x96 shell-1 velocity support. Eight fixed 45-degree sectors are reconstructed and required to close the Stage-111 parent x-change maps before wall-distance interpretation.",
        "configuration": {"grid": list(GRID), "kn0": KNUDSEN, "cold_hot_ratio": COLD_HOT_RATIO, "rule": list(RULE), "radial_scale": RADIAL_SCALE, "limiter": LIMITER, "boundary_slope": BOUNDARY_SLOPE, "source_relaxation": SOURCE_RELAXATION, "tolerance": TOLERANCE, "correction_floor": CORRECTION_FLOOR, "diagnostic_steps": DIAGNOSTIC_STEPS, "wall_band_cells": WALL_BAND_CELLS, "dominant_radial_shell": DOMINANT_RADIAL_SHELL, "radial_shell_count": RADIAL_SHELL_COUNT, "radial_nodes_per_shell": RADIAL_NODES_PER_SHELL, "interior_extent": INTERIOR_EXTENT, "angular_sectors": ANGULAR_SECTORS, "sector_width_degrees": SECTOR_WIDTH_DEGREES, "near_wall_depth": NEAR_WALL_DEPTH, "broad_wall_depth": BROAD_WALL_DEPTH, "sector_localization_share_guard": SECTOR_LOCALIZATION_SHARE_GUARD, "sector_diffuse_share_guard": SECTOR_DIFFUSE_SHARE_GUARD, "diffuse_effective_sector_min": DIFFUSE_EFFECTIVE_SECTOR_MIN, "closure_tolerance": CLOSURE_TOLERANCE, "stage67_run_id": s110.STAGE67_RUN_ID, "stage67_job_id": s110.STAGE67_JOB_ID, "stage67_artifact_id": s110.STAGE67_ARTIFACT_ID, "stage111_run_id": STAGE111_RUN_ID, "stage111_job_id": STAGE111_JOB_ID, "stage111_artifact_id": STAGE111_ARTIFACT_ID, "stage113_run_id": STAGE113_RUN_ID, "stage113_job_id": STAGE113_JOB_ID, "stage113_artifact_id": STAGE113_ARTIFACT_ID, "full_solver_endpoint_rerun": False, "physical_parameter_retuning": False, "collision_parameter_retuning": False, "correction_floor_retuning": False, "positivity_floor_retuning": False, "source_relaxation_retuning": False, "transport_parameter_retuning": False, "wall_model_retuning": False, "normalization_retuning": False, "limiter_retuning": False, "velocity_quadrature_retuning": False, "failed_muscl_endpoint_rehabilitated": False, "one_sided_boundary_slope_promoted": False, "cross_knudsen_extension_permitted": False, "validation_claim_permitted": False, "solver_endpoint_claim_permitted": False},
        "stage67_authorization": {"stage": stage67_summary["stage"], "decision": stage67_summary["decision"]},
        "stage111_authorization": {"decision": stage111_summary["decision"], "workflow_run_id": STAGE111_RUN_ID, "workflow_job_id": STAGE111_JOB_ID, "artifact_id": STAGE111_ARTIFACT_ID},
        "stage113_authorization": {"decision": stage113_summary["decision"], "workflow_run_id": STAGE113_RUN_ID, "workflow_job_id": STAGE113_JOB_ID, "artifact_id": STAGE113_ARTIFACT_ID, "tests_passed": stage113_record["tests"]["passed"], "tests_failed": stage113_record["tests"]["failed"], "common_first_4_cumulative_share": stage113_summary["metrics"]["common"]["first_4_cumulative_share"], "common_first_14_cumulative_share": stage113_summary["metrics"]["common"]["first_14_cumulative_share"], "common_half_mass_depth_cells": stage113_summary["metrics"]["common"]["half_mass_depth_cells"]},
        "sector_point_count": sector_point_count, "finite": bool(finite), "max_parent_x_change_closure_relative_l2": max_closure, "metrics": metrics, "decision": decision,
        "scientific_conclusion": "Stage 114 tests whether the broad Stage-113 sidewall-distance organization hides a wall-distance-specific angular concentration inside the already dominant shell-1 velocity support. The audit uses the existing quadrature only; sector localization or diffuseness is diagnostic attribution, not a proposal to alter the quadrature.",
        "negative_result_guard": "Stage 114 is an artifact-only velocity-sector attribution of the frozen Stage-111/113 association lineage. It cannot establish limiter causality, nonlinear MUSCL stability, endpoint convergence, heat-flux improvement, benchmark improvement, or external validation. Stage 113 remains a broad wall-distance localization surrogate; Stage 111 remains association rather than causal isolation; Stage 110 remains confounded by same-sign gradient strength; Stage 99 remains a negative cross-run reproducibility result; Stage 98 remains a negative cross-run replay result; Stage 90 remains nonconverged in both reconstruction arms; Stage 28 remains a failed MUSCL endpoint; the Stage-89 one-sided boundary slope remains unpromoted. No failed parameter or velocity quadrature is retuned, no cross-Knudsen MUSCL extension is permitted, and no stability, accuracy, benchmark, heat-flux-improvement, or validation claim is authorized."
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"); return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Stage-114 wall-distance-conditioned velocity-quadrature audit")
    parser.add_argument("--stage67-artifact-dir", required=True); parser.add_argument("--stage111-artifact-dir", required=True); parser.add_argument("--stage113-artifact-dir", required=True); parser.add_argument("--stage113-record-path", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); run_stage114(args.stage67_artifact_dir, args.stage111_artifact_dir, args.stage113_artifact_dir, args.stage113_record_path, args.output_dir)


if __name__ == "__main__": main()
