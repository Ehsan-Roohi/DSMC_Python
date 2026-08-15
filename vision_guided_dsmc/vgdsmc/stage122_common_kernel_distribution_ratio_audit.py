from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import stage121_radial_kernel_factor_decomposition_audit as s121

STAGE121_RUN_ID = 31849920665
STAGE121_JOB_ID = 94923642748
STAGE121_ARTIFACT_ID = 9241880532
STAGE121_ARTIFACT_SHA256 = "b37c499cd4e9c6696acc29ab99cc261f6bc4d02b31f9bb71b1fec0c51b9b112f"
STAGE121_SUMMARY_SHA256 = "8c2135d85984ad4a3e0902d35203bbadfe8e1c8c9ba942022a8744d824c34fe1"
STAGE121_PROFILES_SHA256 = "a5df5efbc9b50d6b97638419e32bd69d99d24b5a963d267b3a284b530e8ec5a0"
STAGE121_SOURCE_HEAD = "cc7d6f2be053866719e128de7c40c05fdfdd24c4"
STAGE121_DECISION = s121.MATERIAL

GRID = s121.GRID
KNUDSEN = s121.KNUDSEN
COLD_HOT_RATIO = s121.COLD_HOT_RATIO
RULE = s121.RULE
RADIAL_SCALE = s121.RADIAL_SCALE
LIMITER = s121.LIMITER
BOUNDARY_SLOPE = s121.BOUNDARY_SLOPE
SOURCE_RELAXATION = s121.SOURCE_RELAXATION
TOLERANCE = s121.TOLERANCE
CORRECTION_FLOOR = s121.CORRECTION_FLOOR
DIAGNOSTIC_STEPS = s121.DIAGNOSTIC_STEPS
WALL_BAND_CELLS = s121.WALL_BAND_CELLS
DOMINANT_RADIAL_SHELL = s121.DOMINANT_RADIAL_SHELL
RADIAL_NODES_PER_SHELL = s121.RADIAL_NODES_PER_SHELL
PAIR_SECTORS = s121.PAIR_SECTORS
BANDS = s121.BANDS
PARENT_CLOSURE_TOLERANCE = 1.0e-12
MIN_CENTERED_LOG_RATIO_COSINE = 0.99
MAX_NODE_RATIO_RELATIVE_RANGE = 0.20
MIN_LOO_TEMPLATE_PROFILE_COSINE = 0.995
MAX_LOO_TEMPLATE_TOTAL_VARIATION = 0.03

STABLE = "stage122_common_kernel_ratio_wall_distance_stable_stage123_cellwise_ratio_persistence_audit"
UNSTABLE = "stage122_common_kernel_ratio_wall_distance_dependent_stage123_band_resolved_ratio_origin_audit"
NONFINITE = "stage122_nonfinite_common_kernel_ratio_blocker_without_retuning"
CLOSURE_BLOCKER = "stage122_stage121_payload_reconstruction_blocker_without_retuning"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_stage122_design(**overrides: object) -> None:
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
        "radial_nodes_per_shell": RADIAL_NODES_PER_SHELL,
        "pair_sectors": PAIR_SECTORS,
        "parent_closure_tolerance": PARENT_CLOSURE_TOLERANCE,
        "min_centered_log_ratio_cosine": MIN_CENTERED_LOG_RATIO_COSINE,
        "max_node_ratio_relative_range": MAX_NODE_RATIO_RELATIVE_RANGE,
        "min_loo_template_profile_cosine": MIN_LOO_TEMPLATE_PROFILE_COSINE,
        "max_loo_template_total_variation": MAX_LOO_TEMPLATE_TOTAL_VARIATION,
        "stage121_run_id": STAGE121_RUN_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 122 is fixed to the completed Stage-121 common-|c_x| radial profiles and may not retune "
            "physics, wall/collision/source treatment, reconstruction, transport, floors, normalization, "
            "source relaxation, velocity quadrature, diagnostic window, decision guards, or failed MUSCL parameters"
        )
    if GRID != (64, 64) or RULE != (40, 96) or RADIAL_NODES_PER_SHELL != 10 or PAIR_SECTORS != (5, 6):
        raise ValueError("Stage 122 requires the exact retained 64x64, 40x96, ten-node, sectors-5+6 design")


def _normalize10(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64)
    total = float(x.sum())
    if x.shape != (10,) or not np.isfinite(x).all() or np.any(x <= 0.0) or total <= 0.0:
        raise ValueError("Invalid strictly-positive Stage-122 radial profile")
    return x / total


def profile_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    p, q = _normalize10(a), _normalize10(b)
    denom = max(float(np.linalg.norm(p) * np.linalg.norm(q)), 1.0e-300)
    return {
        "profile_cosine": float(np.dot(p, q) / denom),
        "overlap": float(np.minimum(p, q).sum()),
        "total_variation": float(0.5 * np.abs(p - q).sum()),
    }


def centered_log_ratio(phi_common: np.ndarray, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p, q = _normalize10(phi_common), _normalize10(psi)
    ratio = p / q
    log_ratio = np.log(ratio)
    return ratio, log_ratio - float(log_ratio.mean())


def leave_one_out_template(ratios: np.ndarray, held_out: int) -> np.ndarray:
    x = np.asarray(ratios, dtype=np.float64)
    if x.shape != (3, 10) or held_out not in (0, 1, 2) or not np.isfinite(x).all() or np.any(x <= 0.0):
        raise ValueError("Invalid Stage-122 ratio array")
    keep = [i for i in range(3) if i != held_out]
    return np.exp(np.mean(np.log(x[keep]), axis=0))


def _load_stage121(root: str | Path, record_path: str | Path):
    root = Path(root)
    expected = {
        "summary.json": STAGE121_SUMMARY_SHA256,
        "radial_kernel_factor_profiles.npz": STAGE121_PROFILES_SHA256,
    }
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Stage-121 checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    if summary.get("stage") != 121 or summary.get("decision") != STAGE121_DECISION or summary.get("finite") is not True:
        raise ValueError("Stage-121 artifact does not authorize Stage 122")
    checks = (
        record.get("stage") == 121,
        record.get("decision") == STAGE121_DECISION,
        record.get("source_head") == STAGE121_SOURCE_HEAD,
        record.get("workflow_status") == "completed",
        record.get("workflow_conclusion") == "success",
        record.get("workflow_run_id") == STAGE121_RUN_ID,
        record.get("workflow_job_id") == STAGE121_JOB_ID,
        record.get("artifact_id") == STAGE121_ARTIFACT_ID,
        record.get("artifact_sha256") == STAGE121_ARTIFACT_SHA256,
        record.get("summary_sha256") == STAGE121_SUMMARY_SHA256,
        record.get("radial_kernel_factor_profiles_sha256") == STAGE121_PROFILES_SHA256,
        record.get("tests", {}).get("passed") == 5,
        record.get("tests", {}).get("failed") == 0,
    )
    if not all(checks):
        raise ValueError("Committed Stage-121 provenance does not authorize Stage 122")
    with np.load(root / "radial_kernel_factor_profiles.npz") as data:
        needed = {"phi_common_abs_cx", "psi_exact_radial", "node_speed_squared"}
        if not needed.issubset(data.files):
            raise ValueError("Stage-121 profile payload is incomplete")
        phi_common = np.asarray(data["phi_common_abs_cx"], dtype=np.float64).copy()
        psi = np.asarray(data["psi_exact_radial"], dtype=np.float64).copy()
        speed2 = np.asarray(data["node_speed_squared"], dtype=np.float64).copy()
    if phi_common.shape != (3, 10) or psi.shape != (3, 10) or speed2.shape != (10,):
        raise ValueError("Stage-121 profile payload has the wrong shape")
    if not np.isfinite(phi_common).all() or not np.isfinite(psi).all() or np.any(phi_common <= 0.0) or np.any(psi <= 0.0):
        raise ValueError("Stage-121 common-kernel profiles are invalid")
    return summary, record, phi_common, psi, speed2


def stage122_decision(
    *,
    finite: bool,
    parent_closure: float,
    min_centered_cosine: float,
    max_node_relative_range: float,
    identical_crossing: bool,
    min_loo_cosine: float,
    max_loo_tv: float,
) -> str:
    if not finite:
        return NONFINITE
    if parent_closure > PARENT_CLOSURE_TOLERANCE:
        return CLOSURE_BLOCKER
    stable = (
        min_centered_cosine >= MIN_CENTERED_LOG_RATIO_COSINE
        and max_node_relative_range <= MAX_NODE_RATIO_RELATIVE_RANGE
        and identical_crossing
        and min_loo_cosine >= MIN_LOO_TEMPLATE_PROFILE_COSINE
        and max_loo_tv <= MAX_LOO_TEMPLATE_TOTAL_VARIATION
    )
    return STABLE if stable else UNSTABLE


def run(stage121_dir: str | Path, stage121_record_path: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage122_design(**design)
    parent, _, phi_common, psi, speed2 = _load_stage121(stage121_dir, stage121_record_path)
    ratios = np.empty((3, 10), dtype=np.float64)
    centered = np.empty((3, 10), dtype=np.float64)
    crossing_indices: list[int] = []
    band_metrics: dict[str, dict[str, float]] = {}
    parent_closure = 0.0

    for i, band in enumerate(BANDS):
        ratio, centered_log = centered_log_ratio(phi_common[i], psi[i])
        ratios[i], centered[i] = ratio, centered_log
        above = np.flatnonzero(ratio > 1.0)
        crossing = int(above.max()) if above.size else -1
        crossing_indices.append(crossing)
        common = profile_metrics(phi_common[i], psi[i])
        parent_metric = parent["metrics"][band]
        parent_closure = max(
            parent_closure,
            abs(common["profile_cosine"] - float(parent_metric["common_abs_cx_profile_cosine"])),
            abs(common["total_variation"] - float(parent_metric["common_abs_cx_total_variation"])),
        )
        positive = np.maximum(_normalize10(phi_common[i]) - _normalize10(psi[i]), 0.0)
        band_metrics[band] = {
            "common_profile_cosine": common["profile_cosine"],
            "common_overlap": common["overlap"],
            "common_total_variation": common["total_variation"],
            "last_ratio_above_one_node": crossing,
            "low_nodes_0_2_positive_excess": float(positive[:3].sum()),
            "full_positive_excess": float(positive.sum()),
            "ratio_node0": float(ratio[0]),
            "ratio_node2": float(ratio[2]),
            "ratio_node3": float(ratio[3]),
            "ratio_node9": float(ratio[9]),
        }

    pair_cosines: list[float] = []
    for i in range(3):
        for j in range(i + 1, 3):
            denom = max(float(np.linalg.norm(centered[i]) * np.linalg.norm(centered[j])), 1.0e-300)
            pair_cosines.append(float(np.dot(centered[i], centered[j]) / denom))

    ratio_mean = ratios.mean(axis=0)
    node_relative_range = (ratios.max(axis=0) - ratios.min(axis=0)) / np.maximum(ratio_mean, 1.0e-300)
    loo_predictions = np.empty_like(phi_common)
    loo_cosines: list[float] = []
    loo_tvs: list[float] = []
    for i, band in enumerate(BANDS):
        template = leave_one_out_template(ratios, i)
        pred = _normalize10(psi[i] * template)
        loo_predictions[i] = pred
        m = profile_metrics(phi_common[i], pred)
        loo_cosines.append(m["profile_cosine"])
        loo_tvs.append(m["total_variation"])
        band_metrics[band]["leave_one_out_template_profile_cosine"] = m["profile_cosine"]
        band_metrics[band]["leave_one_out_template_overlap"] = m["overlap"]
        band_metrics[band]["leave_one_out_template_total_variation"] = m["total_variation"]

    finite = bool(
        np.isfinite(ratios).all()
        and np.isfinite(centered).all()
        and np.isfinite(node_relative_range).all()
        and np.isfinite(loo_predictions).all()
        and np.isfinite(parent_closure)
    )
    identical_crossing = len(set(crossing_indices)) == 1
    aggregate = {
        "minimum_pairwise_centered_log_ratio_cosine": float(min(pair_cosines)),
        "maximum_node_ratio_relative_range": float(node_relative_range.max()),
        "identical_last_ratio_above_one_node_across_bands": bool(identical_crossing),
        "common_last_ratio_above_one_node": int(crossing_indices[0]) if identical_crossing else None,
        "minimum_leave_one_out_template_profile_cosine": float(min(loo_cosines)),
        "maximum_leave_one_out_template_total_variation": float(max(loo_tvs)),
        "minimum_leave_one_out_template_overlap": float(1.0 - max(loo_tvs)),
        "maximum_stage121_metric_reconstruction_absolute_error": float(parent_closure),
    }
    decision = stage122_decision(
        finite=finite,
        parent_closure=parent_closure,
        min_centered_cosine=aggregate["minimum_pairwise_centered_log_ratio_cosine"],
        max_node_relative_range=aggregate["maximum_node_ratio_relative_range"],
        identical_crossing=identical_crossing,
        min_loo_cosine=aggregate["minimum_leave_one_out_template_profile_cosine"],
        max_loo_tv=aggregate["maximum_leave_one_out_template_total_variation"],
    )
    summary: dict[str, object] = {
        "stage": 122,
        "finite": finite,
        "decision": decision,
        "parent_stage121": {
            "run_id": STAGE121_RUN_ID,
            "job_id": STAGE121_JOB_ID,
            "artifact_id": STAGE121_ARTIFACT_ID,
            "source_head": STAGE121_SOURCE_HEAD,
        },
        "parent_reconstruction_absolute_error": float(parent_closure),
        "configuration": {
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
            "radial_nodes_per_shell": RADIAL_NODES_PER_SHELL,
            "pair_sectors": list(PAIR_SECTORS),
            "common_kernel": "abs(c_x)",
            "ratio_definition": "normalized_phi_common_abs_cx / normalized_psi_abs_cx",
            "template_definition": "leave-one-band-out geometric mean ratio from the other two fixed wall-distance bands",
            "min_centered_log_ratio_cosine": MIN_CENTERED_LOG_RATIO_COSINE,
            "max_node_ratio_relative_range": MAX_NODE_RATIO_RELATIVE_RANGE,
            "min_loo_template_profile_cosine": MIN_LOO_TEMPLATE_PROFILE_COSINE,
            "max_loo_template_total_variation": MAX_LOO_TEMPLATE_TOTAL_VARIATION,
            "model_retuning": False,
            "wall_retuning": False,
            "source_retuning": False,
            "reconstruction_retuning": False,
            "transport_retuning": False,
            "floor_retuning": False,
            "normalization_retuning": False,
            "velocity_grid_retuning": False,
            "solver_endpoint_advanced": False,
            "failed_muscl_endpoint_rehabilitated": False,
            "stage89_one_sided_boundary_promoted": False,
            "cross_knudsen_extension_permitted": False,
            "benchmark_or_validation_claim_permitted": False,
        },
        "metrics": band_metrics,
        "aggregate": aggregate,
        "node_speed_squared": [float(v) for v in speed2],
        "scientific_conclusion": (
            "Stage 122 tests whether the residual phi/psi difference under the same |c_x| kernel is described by a "
            "wall-distance-stable radial distribution ratio. Stability is evaluated without fitting to the held-out band: "
            "each band is predicted from the geometric-mean ratio of the other two. A positive result supports a reproducible "
            "distribution-specific radial structure and justifies a cellwise persistence audit; it does not establish limiter "
            "causality, MUSCL stability, endpoint convergence, q_av improvement, benchmark accuracy, or validation."
        ),
        "negative_result_guard": (
            "No model, wall, collision/source, reconstruction, transport, positivity/correction floor, normalization, "
            "source-relaxation, velocity-grid, or failed MUSCL parameter is changed. Stage 90 remains nonconverged, Stage 28 "
            "remains failed, Stage 89 remains unpromoted, and no solver endpoint or cross-Knudsen extension is attempted."
        ),
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "common_kernel_distribution_ratio.npz",
        ratio=ratios,
        centered_log_ratio=centered,
        node_relative_range=node_relative_range,
        leave_one_out_predictions=loo_predictions,
        node_speed_squared=speed2,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 122 fixed common-kernel distribution-ratio audit")
    parser.add_argument("--stage121-dir", required=True)
    parser.add_argument("--stage121-record", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run(args.stage121_dir, args.stage121_record, args.output_dir)


if __name__ == "__main__":
    main()
