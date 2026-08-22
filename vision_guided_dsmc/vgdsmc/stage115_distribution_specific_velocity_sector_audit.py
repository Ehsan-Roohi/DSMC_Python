from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import stage114_wall_distance_conditioned_velocity_quadrature_audit as s114

STAGE114_RUN_ID = 31666806262
STAGE114_JOB_ID = 94343074558
STAGE114_ARTIFACT_ID = 9170744603
STAGE114_ARTIFACT_SHA256 = "b2b308e6db640dd1cd369c2ab6d3b76f372f27127585329cc1c552711deaf944"
STAGE114_SUMMARY_SHA256 = "fa3cd4aa577a7bf25036b949a23e07aa284cb34a52afb7e29eddb1cc479a4532"
STAGE114_MAPS_SHA256 = "948f7747dd31e33daaec311da01f5b315e0dbb5b06cd90768fec30e23bf7bbc9"
STAGE114_DECISION = "stage114_mixed_velocity_quadrature_structure_stage115_distribution_specific_audit"
STAGE114_SOURCE_HEAD = "eaca67d8b2f8bcda1ebb48271279ca61e55a511c"
STAGE114_TESTS_PASSED = 226

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
INTERIOR_EXTENT = s114.INTERIOR_EXTENT
ANGULAR_SECTORS = s114.ANGULAR_SECTORS
SECTOR_WIDTH_DEGREES = s114.SECTOR_WIDTH_DEGREES
NEAR_WALL_DEPTH = s114.NEAR_WALL_DEPTH
BROAD_WALL_DEPTH = s114.BROAD_WALL_DEPTH

BANDS = ("near_1_4", "mid_5_14", "inner_15_28")
PROFILE_COSINE_COMMON_MIN = 0.95
PROFILE_OVERLAP_COMMON_MIN = 0.90
COMMON_PAIR_SHARE_MIN = 0.50
DIVERGENCE_COSINE_MAX = 0.90
DIVERGENCE_OVERLAP_MAX = 0.80
PROFILE_RECONSTRUCTION_TOLERANCE = 1.0e-12


def validate_stage115_design(**overrides: object) -> None:
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
        "sector_width_degrees": SECTOR_WIDTH_DEGREES,
        "near_wall_depth": NEAR_WALL_DEPTH,
        "broad_wall_depth": BROAD_WALL_DEPTH,
        "profile_cosine_common_min": PROFILE_COSINE_COMMON_MIN,
        "profile_overlap_common_min": PROFILE_OVERLAP_COMMON_MIN,
        "common_pair_share_min": COMMON_PAIR_SHARE_MIN,
        "divergence_cosine_max": DIVERGENCE_COSINE_MAX,
        "divergence_overlap_max": DIVERGENCE_OVERLAP_MAX,
        "profile_reconstruction_tolerance": PROFILE_RECONSTRUCTION_TOLERANCE,
        "stage114_run_id": STAGE114_RUN_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 115 is frozen to the exact completed Stage-114 mixed-structure record. "
            "Physics, grid, collision/source treatment, floors, wall treatment, normalization, limiter, "
            "velocity quadrature, diagnostic window, failed MUSCL parameters, and interpretation guards may not be retuned."
        )
    if RULE != (40, 96) or ANGULAR_SECTORS != 8 or INTERIOR_EXTENT != 56:
        raise ValueError("Stage 115 requires the exact 40x96 rule, eight sectors, and 56x56 interior")
    if not (0.0 < DIVERGENCE_OVERLAP_MAX < PROFILE_OVERLAP_COMMON_MIN < 1.0):
        raise ValueError("Stage-115 overlap guards are invalid")
    if not (0.0 < DIVERGENCE_COSINE_MAX < PROFILE_COSINE_COMMON_MIN <= 1.0):
        raise ValueError("Stage-115 cosine guards are invalid")
    if not (0.0 < COMMON_PAIR_SHARE_MIN < 1.0):
        raise ValueError("Stage-115 pair-share guard is invalid")
    if PROFILE_RECONSTRUCTION_TOLERANCE != 1.0e-12:
        raise ValueError("Stage-115 reconstruction tolerance is frozen to 1e-12")


def _load_stage114_record(path: str | Path) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("stage") != 114 or record.get("decision") != STAGE114_DECISION or record.get("finite") is not True:
        raise ValueError("Committed Stage-114 record does not authorize Stage 115")
    if record.get("source_head") != STAGE114_SOURCE_HEAD:
        raise ValueError("Committed Stage-114 source head mismatch")
    if record.get("workflow_status") != "completed" or record.get("workflow_conclusion") != "success":
        raise ValueError("Committed Stage-114 workflow is not a successful completed run")
    if int(record.get("workflow_run_id", -1)) != STAGE114_RUN_ID or int(record.get("workflow_job_id", -1)) != STAGE114_JOB_ID:
        raise ValueError("Committed Stage-114 workflow provenance mismatch")
    if int(record.get("artifact_id", -1)) != STAGE114_ARTIFACT_ID or record.get("artifact_sha256") != STAGE114_ARTIFACT_SHA256:
        raise ValueError("Committed Stage-114 artifact provenance mismatch")
    if record.get("summary_sha256") != STAGE114_SUMMARY_SHA256 or record.get("wall_distance_velocity_sector_maps_sha256") != STAGE114_MAPS_SHA256:
        raise ValueError("Committed Stage-114 payload digest mismatch")
    tests = record.get("tests", {})
    if not isinstance(tests, dict) or tests.get("passed") != STAGE114_TESTS_PASSED or tests.get("failed") != 0:
        raise ValueError("Committed Stage-114 test record is not the exact successful endpoint")
    if float(record.get("max_parent_x_change_closure_relative_l2", np.inf)) > PROFILE_RECONSTRUCTION_TOLERANCE:
        raise ValueError("Stage-114 parent-map closure is not admissible")

    phi = []
    psi = []
    max_integrity_error = 0.0
    for band in BANDS:
        pblock = record["metrics"]["phi"][band]
        qblock = record["metrics"]["psi"][band]
        p = np.asarray(pblock["sector_share"], dtype=np.float64)
        q = np.asarray(qblock["sector_share"], dtype=np.float64)
        if p.shape != (ANGULAR_SECTORS,) or q.shape != (ANGULAR_SECTORS,):
            raise ValueError(f"Stage-114 sector profile shape is invalid in {band}")
        if not np.isfinite(p).all() or not np.isfinite(q).all() or np.any(p < 0.0) or np.any(q < 0.0):
            raise ValueError(f"Stage-114 sector profile is nonfinite or negative in {band}")
        max_integrity_error = max(
            max_integrity_error,
            abs(float(np.sum(p)) - 1.0),
            abs(float(np.sum(q)) - 1.0),
            abs(float(np.max(p)) - float(pblock["maximum_sector_share"])),
            abs(float(np.max(q)) - float(qblock["maximum_sector_share"])),
        )
        if int(np.argmax(p)) != int(pblock["maximum_sector_index"]) or int(np.argmax(q)) != int(qblock["maximum_sector_index"]):
            raise ValueError(f"Stage-114 recorded maximum-sector index is inconsistent in {band}")
        phi.append(p)
        psi.append(q)
    record["stage115_record_profile_integrity_abs_error"] = float(max_integrity_error)
    return record, np.asarray(phi), np.asarray(psi)


def _band_profiles(sector_maps: np.ndarray, growth: np.ndarray) -> np.ndarray:
    maps = np.asarray(sector_maps, dtype=np.float64)
    g = np.asarray(growth, dtype=np.float64)
    if maps.shape != (ANGULAR_SECTORS, INTERIOR_EXTENT, INTERIOR_EXTENT) or g.shape != (INTERIOR_EXTENT, INTERIOR_EXTENT):
        raise ValueError("Stage-115 profile inputs have invalid shapes")
    density = maps * g[None, :, :]
    masks = s114.wall_distance_band_masks()
    profiles = []
    for band in BANDS:
        raw = np.array([float(np.sum(density[k][masks[band]])) for k in range(ANGULAR_SECTORS)], dtype=np.float64)
        total = float(np.sum(raw))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(f"Stage-115 conditioned mass is invalid in band {band}")
        profiles.append(raw / total)
    out = np.asarray(profiles, dtype=np.float64)
    if not np.isfinite(out).all() or np.any(out < 0.0):
        raise ValueError("Stage-115 normalized profiles are nonfinite or negative")
    return out


def _circular_adjacent(pair: np.ndarray | list[int] | tuple[int, int]) -> bool:
    a, b = sorted(int(v) for v in pair)
    delta = b - a
    return delta == 1 or delta == ANGULAR_SECTORS - 1


def _top2(profile: np.ndarray) -> np.ndarray:
    p = np.asarray(profile, dtype=np.float64)
    if p.shape != (ANGULAR_SECTORS,):
        raise ValueError("Stage-115 top-two input must be one eight-sector profile")
    return np.argsort(p, kind="stable")[-2:][::-1]


def _profile_metrics(phi: np.ndarray, psi: np.ndarray) -> dict[str, object]:
    p = np.asarray(phi, dtype=np.float64)
    q = np.asarray(psi, dtype=np.float64)
    if p.shape != (ANGULAR_SECTORS,) or q.shape != (ANGULAR_SECTORS,):
        raise ValueError("Stage-115 profile comparison requires two eight-sector vectors")
    if not np.isfinite(p).all() or not np.isfinite(q).all() or np.any(p < 0.0) or np.any(q < 0.0):
        raise ValueError("Stage-115 profiles must be finite and nonnegative")
    p = p / float(np.sum(p)); q = q / float(np.sum(q))
    denom = float(np.linalg.norm(p) * np.linalg.norm(q))
    cosine = float(np.dot(p, q) / max(denom, 1.0e-300))
    tv = float(0.5 * np.sum(np.abs(p - q)))
    overlap = float(np.sum(np.minimum(p, q)))
    m = 0.5 * (p + q)
    mp = p > 0.0; mq = q > 0.0
    js_bits = 0.5 * (
        float(np.sum(p[mp] * np.log2(p[mp] / m[mp])))
        + float(np.sum(q[mq] * np.log2(q[mq] / m[mq])))
    )
    phi_top2 = _top2(p); psi_top2 = _top2(q); joint_top2 = _top2(m)
    phi_set = set(int(v) for v in phi_top2); psi_set = set(int(v) for v in psi_top2); joint_set = set(int(v) for v in joint_top2)
    return {
        "profile_cosine": cosine,
        "total_variation_distance": tv,
        "overlap_coefficient": overlap,
        "jensen_shannon_bits": float(js_bits),
        "phi_top2_sector_index": [int(v) for v in phi_top2],
        "psi_top2_sector_index": [int(v) for v in psi_top2],
        "joint_top2_sector_index": [int(v) for v in joint_top2],
        "phi_top2_share": float(np.sum(p[phi_top2])),
        "psi_top2_share": float(np.sum(q[psi_top2])),
        "joint_top2_share": float(np.sum(m[joint_top2])),
        "phi_psi_top2_sets_match": bool(phi_set == psi_set),
        "joint_top2_matches_both": bool(joint_set == phi_set == psi_set),
        "joint_top2_is_circularly_adjacent": bool(_circular_adjacent(joint_top2)),
    }


def stage115_decision(metrics: dict[str, dict[str, object]], finite: bool, max_profile_reconstruction_error: float) -> str:
    if not finite or not np.isfinite(max_profile_reconstruction_error):
        return "stage115_nonfinite_distribution_profile_blocker_without_retuning"
    if max_profile_reconstruction_error > PROFILE_RECONSTRUCTION_TOLERANCE:
        return "stage115_stage114_profile_reconstruction_blocker_without_retuning"
    broad = [metrics["near_1_4"], metrics["mid_5_14"]]
    pair_sets = []
    common = True
    for block in broad:
        pair_sets.append(tuple(sorted(int(v) for v in block["joint_top2_sector_index"])))
        if float(block["profile_cosine"]) < PROFILE_COSINE_COMMON_MIN:
            common = False
        if float(block["overlap_coefficient"]) < PROFILE_OVERLAP_COMMON_MIN:
            common = False
        if not bool(block["joint_top2_matches_both"]) or not bool(block["joint_top2_is_circularly_adjacent"]):
            common = False
        if float(block["phi_top2_share"]) < COMMON_PAIR_SHARE_MIN or float(block["psi_top2_share"]) < COMMON_PAIR_SHARE_MIN:
            common = False
    if common and pair_sets[0] == pair_sets[1]:
        return "stage115_common_adjacent_pair_support_stage116_pair_resolved_radial_node_audit"
    if any(float(block["profile_cosine"]) <= DIVERGENCE_COSINE_MAX or float(block["overlap_coefficient"]) <= DIVERGENCE_OVERLAP_MAX for block in broad):
        return "stage115_distribution_specific_angular_divergence_stage116_distribution_contrast_audit"
    return "stage115_partial_common_angular_support_stage116_band_specific_pair_audit"


def run_stage115(stage114_record_path: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage115_design(**design)
    record, phi_profiles, psi_profiles = _load_stage114_record(stage114_record_path)
    joint_profiles = 0.5 * (phi_profiles + psi_profiles)
    integrity_error = float(record["stage115_record_profile_integrity_abs_error"])
    metrics: dict[str, dict[str, object]] = {}
    for i, band in enumerate(BANDS):
        block = _profile_metrics(phi_profiles[i], psi_profiles[i])
        block["phi_sector_share"] = phi_profiles[i].tolist()
        block["psi_sector_share"] = psi_profiles[i].tolist()
        block["joint_sector_share"] = joint_profiles[i].tolist()
        metrics[band] = block
    finite = bool(np.isfinite(phi_profiles).all() and np.isfinite(psi_profiles).all() and np.isfinite(integrity_error))
    decision = stage115_decision(metrics, finite, integrity_error)

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "distribution_specific_sector_profiles.npz",
        phi_sector_share=phi_profiles,
        psi_sector_share=psi_profiles,
        joint_sector_share=joint_profiles,
        sector_center_degrees=(np.arange(ANGULAR_SECTORS, dtype=np.float64) + 0.5) * SECTOR_WIDTH_DEGREES,
    )
    summary = {
        "stage": 115,
        "description": "Frozen distribution-specific comparison of the completed Stage-114 phi and psi eight-sector profiles. The audit distinguishes genuine distribution-specific angular divergence from a common adjacent-pair support hidden by the earlier single-sector threshold.",
        "configuration": {
            "grid": list(GRID), "kn0": KNUDSEN, "cold_hot_ratio": COLD_HOT_RATIO,
            "rule": list(RULE), "radial_scale": RADIAL_SCALE, "limiter": LIMITER,
            "boundary_slope": BOUNDARY_SLOPE, "source_relaxation": SOURCE_RELAXATION,
            "tolerance": TOLERANCE, "correction_floor": CORRECTION_FLOOR,
            "diagnostic_steps": DIAGNOSTIC_STEPS, "wall_band_cells": WALL_BAND_CELLS,
            "dominant_radial_shell": DOMINANT_RADIAL_SHELL, "radial_shell_count": RADIAL_SHELL_COUNT,
            "radial_nodes_per_shell": RADIAL_NODES_PER_SHELL, "interior_extent": INTERIOR_EXTENT,
            "angular_sectors": ANGULAR_SECTORS, "sector_width_degrees": SECTOR_WIDTH_DEGREES,
            "near_wall_depth": NEAR_WALL_DEPTH, "broad_wall_depth": BROAD_WALL_DEPTH,
            "profile_cosine_common_min": PROFILE_COSINE_COMMON_MIN,
            "profile_overlap_common_min": PROFILE_OVERLAP_COMMON_MIN,
            "common_pair_share_min": COMMON_PAIR_SHARE_MIN,
            "divergence_cosine_max": DIVERGENCE_COSINE_MAX,
            "divergence_overlap_max": DIVERGENCE_OVERLAP_MAX,
            "profile_reconstruction_tolerance": PROFILE_RECONSTRUCTION_TOLERANCE,
            "stage114_run_id": STAGE114_RUN_ID, "stage114_job_id": STAGE114_JOB_ID,
            "stage114_artifact_id": STAGE114_ARTIFACT_ID,
            "full_solver_endpoint_rerun": False, "physical_parameter_retuning": False,
            "collision_parameter_retuning": False, "correction_floor_retuning": False,
            "positivity_floor_retuning": False, "source_relaxation_retuning": False,
            "transport_parameter_retuning": False, "wall_model_retuning": False,
            "normalization_retuning": False, "limiter_retuning": False,
            "velocity_quadrature_retuning": False, "failed_muscl_endpoint_rehabilitated": False,
            "one_sided_boundary_slope_promoted": False, "cross_knudsen_extension_permitted": False,
            "validation_claim_permitted": False, "solver_endpoint_claim_permitted": False,
        },
        "stage114_authorization": {
            "decision": record["decision"], "workflow_run_id": STAGE114_RUN_ID,
            "workflow_job_id": STAGE114_JOB_ID, "artifact_id": STAGE114_ARTIFACT_ID,
            "artifact_sha256": STAGE114_ARTIFACT_SHA256, "summary_sha256": STAGE114_SUMMARY_SHA256,
            "maps_sha256": STAGE114_MAPS_SHA256, "tests_passed": STAGE114_TESTS_PASSED,
            "tests_failed": 0, "max_parent_x_change_closure_relative_l2": record["max_parent_x_change_closure_relative_l2"],
        },
        "finite": finite,
        "max_stage114_profile_reconstruction_abs_error": integrity_error,
        "metrics": metrics,
        "decision": decision,
        "scientific_conclusion": "Stage 115 compares the already completed Stage-114 normalized phi/psi angular profiles only. A common-pair outcome means shared diagnostic support, not causal sensitivity, solver stability, accuracy, or validation.",
        "negative_result_guard": "Stage 115 is an artifact-record-only comparison of frozen Stage-114 phi/psi angular profiles. It cannot establish limiter causality, nonlinear MUSCL stability, endpoint convergence, heat-flux improvement, benchmark improvement, or external validation. Stage 114 did not support a single common sector above its preregistered threshold; Stage 113 remains a broad wall-distance localization surrogate; Stage 111 remains association rather than causal isolation; Stage 110 remains confounded by same-sign gradient strength; Stage 99 remains a negative cross-run reproducibility result; Stage 98 remains a negative cross-run replay result; Stage 90 remains nonconverged in both reconstruction arms; Stage 28 remains a failed MUSCL endpoint; the Stage-89 one-sided boundary slope remains unpromoted. No failed parameter or velocity quadrature is retuned, no cross-Knudsen MUSCL extension is permitted, and no stability, accuracy, benchmark, heat-flux-improvement, or validation claim is authorized.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Stage-115 distribution-specific velocity-sector audit")
    parser.add_argument("--stage114-record-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_stage115(args.stage114_record_path, args.output_dir)


if __name__ == "__main__":
    main()
