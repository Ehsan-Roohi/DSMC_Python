from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STAGE112_RUN_ID = 31618371834
STAGE112_JOB_ID = 94186713428
STAGE112_ARTIFACT_ID = 9156769170
STAGE112_ARTIFACT_SHA256 = "ec23d93839ead2882dd4471095b7e8f2504a1bc5a0b47301f65a9fb5eac997c0"
STAGE112_SUMMARY_SHA256 = "bc30e51ebceb57cff5e65d11f7dd66d446d00679f2d1745b8c535de1faa9ec00"
STAGE112_MAPS_SHA256 = "d51424d8a011083ae55fcfe084201d70b69c85cabf4a07c0b5deb7d203eae07e"
STAGE112_DECISION = "stage112_symmetric_outer_x_quarter_localization_stage113_x_wall_distance_profile_audit"

GRID = (64, 64)
KNUDSEN = 10.0
COLD_HOT_RATIO = 0.1
RULE = (40, 96)
RADIAL_SCALE = 2.0
LIMITER = "minmod"
BOUNDARY_SLOPE = "zero"
SOURCE_RELAXATION = 1.0
TOLERANCE = 2.0e-5
CORRECTION_FLOOR = 0.05
DIAGNOSTIC_STEPS = 25
WALL_BAND_CELLS = 4
DOMINANT_RADIAL_SHELL = 1
RADIAL_SHELL_COUNT = 4
RADIAL_NODES_PER_SHELL = 10
INTERIOR_EXTENT = 56
PROFILE_BINS = INTERIOR_EXTENT // 2

THIN_FIRST4_CUMULATIVE_GUARD = 0.50
THIN_HALF_MASS_DEPTH_MAX = 4
BROAD_FIRST14_CUMULATIVE_GUARD = 0.75
BROAD_HALF_MASS_DEPTH_MIN = 5
BROAD_EFFECTIVE_BINS_MIN = 12.0


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_stage113_design(**overrides: object) -> None:
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
        "interior_extent": INTERIOR_EXTENT,
        "profile_bins": PROFILE_BINS,
        "thin_first4_cumulative_guard": THIN_FIRST4_CUMULATIVE_GUARD,
        "thin_half_mass_depth_max": THIN_HALF_MASS_DEPTH_MAX,
        "broad_first14_cumulative_guard": BROAD_FIRST14_CUMULATIVE_GUARD,
        "broad_half_mass_depth_min": BROAD_HALF_MASS_DEPTH_MIN,
        "broad_effective_bins_min": BROAD_EFFECTIVE_BINS_MIN,
        "stage112_run_id": STAGE112_RUN_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 113 is frozen to the exact completed Stage-112 artifact. "
            "Physics, collision/source treatment, floors, source relaxation, transport, walls, "
            "normalization, limiter, velocity quadrature, diagnostic window, and decision guards "
            "may not be retuned."
        )
    if INTERIOR_EXTENT != 56 or PROFILE_BINS != 28:
        raise ValueError("Stage 113 requires the exact 56-column Stage-112 interior")
    if not (0.0 < THIN_FIRST4_CUMULATIVE_GUARD < 1.0):
        raise ValueError("invalid thin-layer guard")
    if not (0.0 < BROAD_FIRST14_CUMULATIVE_GUARD < 1.0):
        raise ValueError("invalid broad-layer guard")
    if BROAD_HALF_MASS_DEPTH_MIN <= THIN_HALF_MASS_DEPTH_MAX:
        raise ValueError("broad half-mass guard must exceed the thin guard")
    if BROAD_EFFECTIVE_BINS_MIN <= 1.0:
        raise ValueError("broad effective-bin guard must exceed one bin")


def _load_stage112(root: str | Path, record_path: str | Path):
    root = Path(root)
    expected = {
        "summary.json": STAGE112_SUMMARY_SHA256,
        "x_axis_spatial_localization_maps.npz": STAGE112_MAPS_SHA256,
    }
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Stage-112 checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    if summary.get("stage") != 112 or summary.get("decision") != STAGE112_DECISION or summary.get("finite") is not True:
        raise ValueError("Stage-112 artifact does not authorize Stage 113")
    if record.get("stage") != 112 or record.get("decision") != STAGE112_DECISION:
        raise ValueError("Committed Stage-112 record does not authorize Stage 113")
    if record.get("workflow_status") != "completed" or record.get("workflow_conclusion") != "success":
        raise ValueError("Committed Stage-112 record is not a successful completed workflow")
    if int(record.get("workflow_run_id", -1)) != STAGE112_RUN_ID:
        raise ValueError("Committed Stage-112 record has wrong workflow run")
    if int(record.get("workflow_job_id", -1)) != STAGE112_JOB_ID:
        raise ValueError("Committed Stage-112 record has wrong workflow job")
    if int(record.get("artifact_id", -1)) != STAGE112_ARTIFACT_ID:
        raise ValueError("Committed Stage-112 record has wrong artifact")
    if record.get("artifact_sha256") != STAGE112_ARTIFACT_SHA256:
        raise ValueError("Committed Stage-112 artifact digest mismatch")
    if record.get("summary_sha256") != STAGE112_SUMMARY_SHA256:
        raise ValueError("Committed Stage-112 summary digest mismatch")
    if record.get("x_axis_spatial_localization_maps_sha256") != STAGE112_MAPS_SHA256:
        raise ValueError("Committed Stage-112 map digest mismatch")
    tests = record.get("tests", {})
    if not isinstance(tests, dict) or tests.get("passed") != 204 or tests.get("failed") != 0:
        raise ValueError("Committed Stage-112 test record is not the exact successful endpoint")
    cfg = summary.get("configuration", {})
    expected_cfg = {
        "grid": list(GRID),
        "kn0": KNUDSEN,
        "cold_hot_ratio": COLD_HOT_RATIO,
        "rule": list(RULE),
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
        "interior_extent": INTERIOR_EXTENT,
    }
    if not isinstance(cfg, dict) or any(cfg.get(k) != v for k, v in expected_cfg.items()):
        raise ValueError("Stage-112 configuration does not match frozen Stage-113 design")
    for key, value in cfg.items():
        if key.endswith("_retuning") and value is not False:
            raise ValueError("Stage-112 parent reports forbidden retuning")
    if cfg.get("failed_muscl_endpoint_rehabilitated") is not False:
        raise ValueError("Stage 113 cannot consume a rehabilitated MUSCL endpoint")
    if cfg.get("cross_knudsen_extension_permitted") is not False:
        raise ValueError("Stage 113 cannot consume a cross-Knudsen MUSCL extension")
    needed = {
        "phi_x_asymmetry_growth_product_share",
        "psi_x_asymmetry_growth_product_share",
        "common_x_asymmetry_growth_product_share",
    }
    with np.load(root / "x_axis_spatial_localization_maps.npz") as data:
        if not needed.issubset(data.files):
            raise ValueError("Stage-112 map payload misses required product-share fields")
        arrays = {name: np.asarray(data[name], dtype=np.float64).copy() for name in needed}
    for name, value in arrays.items():
        if value.shape != (INTERIOR_EXTENT, INTERIOR_EXTENT):
            raise ValueError(f"{name} has wrong shape")
        if not np.isfinite(value).all() or np.any(value < 0.0):
            raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isclose(float(np.sum(value)), 1.0, rtol=0.0, atol=5e-13):
            raise ValueError(f"{name} must be normalized")
    return summary, arrays, record


def wall_distance_profile(share_map: np.ndarray) -> np.ndarray:
    a = np.asarray(share_map, dtype=np.float64)
    if a.shape != (INTERIOR_EXTENT, INTERIOR_EXTENT):
        raise ValueError("Stage-113 share map must be 56x56")
    if not np.isfinite(a).all() or np.any(a < 0.0):
        raise ValueError("Stage-113 share map must be finite and nonnegative")
    total = float(np.sum(a))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Stage-113 share map must have positive mass")
    a = a / total
    profile = np.array(
        [float(np.sum(a[:, d]) + np.sum(a[:, INTERIOR_EXTENT - 1 - d])) for d in range(PROFILE_BINS)],
        dtype=np.float64,
    )
    profile /= float(np.sum(profile))
    return profile


def profile_metrics(profile: np.ndarray) -> dict[str, object]:
    p = np.asarray(profile, dtype=np.float64)
    if p.shape != (PROFILE_BINS,) or not np.isfinite(p).all() or np.any(p < 0.0):
        raise ValueError("Stage-113 wall-distance profile is invalid")
    total = float(np.sum(p))
    if total <= 0.0:
        raise ValueError("Stage-113 wall-distance profile has no mass")
    p = p / total
    cumulative = np.cumsum(p)
    centers = np.arange(PROFILE_BINS, dtype=np.float64) + 0.5

    def depth(frac: float) -> int:
        idx = np.flatnonzero(cumulative >= frac - 1.0e-14)
        return int(idx[0] + 1)

    first4 = float(np.sum(p[:4]))
    next4 = float(np.sum(p[4:8]))
    return {
        "profile_share": p.tolist(),
        "cumulative_share": cumulative.tolist(),
        "first_1_cumulative_share": float(cumulative[0]),
        "first_2_cumulative_share": float(cumulative[1]),
        "first_4_cumulative_share": float(cumulative[3]),
        "first_7_cumulative_share": float(cumulative[6]),
        "first_14_cumulative_share": float(cumulative[13]),
        "half_mass_depth_cells": depth(0.50),
        "three_quarter_mass_depth_cells": depth(0.75),
        "ninety_percent_mass_depth_cells": depth(0.90),
        "mean_cell_center_distance": float(np.sum(centers * p)),
        "rms_cell_center_distance": float(np.sqrt(np.sum(centers * centers * p))),
        "effective_profile_bin_count": float(1.0 / np.sum(p * p)),
        "peak_distance_bin": int(np.argmax(p)),
        "peak_share": float(np.max(p)),
        "first4_to_next4_ratio": float(first4 / max(next4, 1.0e-300)),
        "monotonic_increase_count": int(np.count_nonzero(np.diff(p) > 0.0)),
    }


def stage113_decision(common: dict[str, object], finite: bool = True) -> str:
    if not finite:
        return "stage113_nonfinite_wall_distance_profile_blocker_without_retuning"
    first4 = float(common["first_4_cumulative_share"])
    first14 = float(common["first_14_cumulative_share"])
    d50 = int(common["half_mass_depth_cells"])
    eff = float(common["effective_profile_bin_count"])
    if first4 >= THIN_FIRST4_CUMULATIVE_GUARD and d50 <= THIN_HALF_MASS_DEPTH_MAX:
        return "stage113_thin_x_wall_layer_stage114_near_wall_velocity_quadrature_audit"
    if first14 >= BROAD_FIRST14_CUMULATIVE_GUARD and d50 >= BROAD_HALF_MASS_DEPTH_MIN and eff >= BROAD_EFFECTIVE_BINS_MIN:
        return "stage113_broad_x_wall_distance_profile_stage114_wall_distance_conditioned_velocity_quadrature_audit"
    return "stage113_intermediate_x_wall_profile_stage114_wall_distance_decay_audit"


def run_stage113(stage112_artifact_dir: str | Path, stage112_record_path: str | Path, output_dir: str | Path) -> dict[str, object]:
    validate_stage113_design()
    parent, arrays, record = _load_stage112(stage112_artifact_dir, stage112_record_path)
    profiles: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, object]] = {}
    mapping = {
        "phi": "phi_x_asymmetry_growth_product_share",
        "psi": "psi_x_asymmetry_growth_product_share",
        "common": "common_x_asymmetry_growth_product_share",
    }
    for label, name in mapping.items():
        profiles[label] = wall_distance_profile(arrays[name])
        metrics[label] = profile_metrics(profiles[label])
    finite = all(np.isfinite(p).all() for p in profiles.values())
    decision = stage113_decision(metrics["common"], finite=finite)
    summary = {
        "stage": 113,
        "description": "Frozen symmetric x-wall-distance profile of the completed Stage-112 association surrogate.",
        "configuration": {
            "grid": list(GRID), "kn0": KNUDSEN, "cold_hot_ratio": COLD_HOT_RATIO,
            "rule": list(RULE), "radial_scale": RADIAL_SCALE, "limiter": LIMITER,
            "boundary_slope": BOUNDARY_SLOPE, "source_relaxation": SOURCE_RELAXATION,
            "tolerance": TOLERANCE, "correction_floor": CORRECTION_FLOOR,
            "diagnostic_steps": DIAGNOSTIC_STEPS, "wall_band_cells": WALL_BAND_CELLS,
            "dominant_radial_shell": DOMINANT_RADIAL_SHELL, "radial_shell_count": RADIAL_SHELL_COUNT,
            "radial_nodes_per_shell": RADIAL_NODES_PER_SHELL, "interior_extent": INTERIOR_EXTENT,
            "profile_bins": PROFILE_BINS,
            "thin_first4_cumulative_guard": THIN_FIRST4_CUMULATIVE_GUARD,
            "thin_half_mass_depth_max": THIN_HALF_MASS_DEPTH_MAX,
            "broad_first14_cumulative_guard": BROAD_FIRST14_CUMULATIVE_GUARD,
            "broad_half_mass_depth_min": BROAD_HALF_MASS_DEPTH_MIN,
            "broad_effective_bins_min": BROAD_EFFECTIVE_BINS_MIN,
            "stage112_run_id": STAGE112_RUN_ID, "stage112_job_id": STAGE112_JOB_ID,
            "stage112_artifact_id": STAGE112_ARTIFACT_ID,
            "full_solver_endpoint_rerun": False, "failed_muscl_endpoint_rehabilitated": False,
            "one_sided_boundary_slope_promoted": False, "cross_knudsen_extension_permitted": False,
            "validation_claim_permitted": False, "solver_endpoint_claim_permitted": False,
            "physical_parameter_retuning": False, "collision_parameter_retuning": False,
            "correction_floor_retuning": False, "positivity_floor_retuning": False,
            "source_relaxation_retuning": False, "wall_model_retuning": False,
            "normalization_retuning": False, "limiter_retuning": False,
            "transport_parameter_retuning": False, "velocity_quadrature_retuning": False,
        },
        "stage112_authorization": {
            "decision": parent["decision"], "source_head": record["source_head"],
            "workflow_run_id": STAGE112_RUN_ID, "workflow_job_id": STAGE112_JOB_ID,
            "artifact_id": STAGE112_ARTIFACT_ID, "tests_passed": 204, "tests_failed": 0,
        },
        "metrics": metrics,
        "finite": bool(finite),
        "decision": decision,
        "negative_result_guard": (
            "Stage 113 is an artifact-only wall-distance localization audit of the frozen Stage-112 x-axis association surrogate. "
            "A broad or thin profile is not a causal sensitivity, adjoint, solver correction, convergence measure, heat-flux improvement, "
            "benchmark improvement, or validation. Stage 111 remains association rather than causal isolation; Stage 110 remains confounded "
            "by same-sign gradient strength; Stage 99 remains a negative cross-run reproducibility result; Stage 90 remains nonconverged in "
            "both reconstruction arms; Stage 28 remains a failed MUSCL endpoint; the Stage-89 one-sided boundary slope remains unpromoted. "
            "No failed parameter is retuned and no cross-Knudsen MUSCL extension is permitted."
        ),
        "scientific_conclusion": (
            f"The frozen common x-axis association has first-4-pair cumulative share {metrics['common']['first_4_cumulative_share']:.12g}, "
            f"half-mass wall distance {metrics['common']['half_mass_depth_cells']} cells, first-14-pair cumulative share "
            f"{metrics['common']['first_14_cumulative_share']:.12g}, and effective profile support "
            f"{metrics['common']['effective_profile_bin_count']:.12g} bins. This resolves whether the Stage-112 outer-quarter signal is a thin "
            "sidewall layer or a broad wall-distance structure without rerunning or retuning the solver. The next route, if broad, conditions "
            "the established signal on the frozen non-Cartesian velocity quadrature rather than changing that quadrature."
        ),
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        out / "x_wall_distance_profiles.npz",
        wall_distance_cell_centers=np.arange(PROFILE_BINS, dtype=np.float64) + 0.5,
        phi_profile=profiles["phi"], psi_profile=profiles["psi"], common_profile=profiles["common"],
        phi_cumulative=np.cumsum(profiles["phi"]), psi_cumulative=np.cumsum(profiles["psi"]),
        common_cumulative=np.cumsum(profiles["common"]),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage112-artifact-dir", required=True)
    parser.add_argument("--stage112-record-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = run_stage113(args.stage112_artifact_dir, args.stage112_record_path, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
