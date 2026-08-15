from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import stage110_same_sign_slope_asymmetry_audit as s110
from . import stage114_wall_distance_conditioned_velocity_quadrature_audit as s114
from . import stage119_exact_directional_moment_kernel_audit as s119
from . import stage122_common_kernel_distribution_ratio_audit as s122

STAGE122_RUN_ID = 31865788577
STAGE122_JOB_ID = 94966518942
STAGE122_ARTIFACT_ID = 9245886921
STAGE122_ARTIFACT_SHA256 = "9102aec398c865639945190c24b3368ceb10fbe02ee2b3308e05dab288fb9c00"
STAGE122_SUMMARY_SHA256 = "aea388dc498b1f78693fa215c8996bef6f71c949a4f2ffc8e169d0603c1e4e13"
STAGE122_PROFILES_SHA256 = "bc7829374fd43056c74ee101bac87d045760e76f6b520125749a2b017bf3591e"
STAGE122_SOURCE_HEAD = "8725bd56cebaa4d036f7eb741be3ba231480e621"
STAGE122_DECISION = s122.STABLE

GRID = s122.GRID
KNUDSEN = s122.KNUDSEN
COLD_HOT_RATIO = s122.COLD_HOT_RATIO
RULE = s122.RULE
RADIAL_SCALE = s122.RADIAL_SCALE
LIMITER = s122.LIMITER
BOUNDARY_SLOPE = s122.BOUNDARY_SLOPE
SOURCE_RELAXATION = s122.SOURCE_RELAXATION
TOLERANCE = s122.TOLERANCE
CORRECTION_FLOOR = s122.CORRECTION_FLOOR
DIAGNOSTIC_STEPS = s122.DIAGNOSTIC_STEPS
WALL_BAND_CELLS = s122.WALL_BAND_CELLS
DOMINANT_RADIAL_SHELL = s122.DOMINANT_RADIAL_SHELL
RADIAL_NODES_PER_SHELL = s122.RADIAL_NODES_PER_SHELL
PAIR_SECTORS = s122.PAIR_SECTORS
BANDS = s122.BANDS

# Stage 123 changes no solver parameter.  These are preregistered diagnostic
# guards for asking whether the Stage-122 cross-band ratio survives at the
# individual spatial-cell level.  The profile thresholds are intentionally
# fixed before the Stage-123 artifact is generated.
PARENT_CLOSURE_TOLERANCE = 1.0e-12
CELL_PREDICTION_COSINE_MIN = 0.95
CELL_PREDICTION_TV_MAX = 0.15
MIN_VALID_CELL_FRACTION = 0.95
MIN_CELL_PASS_FRACTION = 0.75

PERSISTENT = "stage123_cellwise_ratio_persistent_stage124_spatial_transition_coherence_audit"
AGGREGATION_ONLY = "stage123_band_ratio_aggregation_only_stage124_within_band_cancellation_audit"
NONFINITE = "stage123_nonfinite_cellwise_ratio_blocker_without_retuning"
CLOSURE_BLOCKER = "stage123_stage122_parent_profile_closure_blocker_without_retuning"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_stage123_design(**overrides: object) -> None:
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
        "cell_prediction_cosine_min": CELL_PREDICTION_COSINE_MIN,
        "cell_prediction_tv_max": CELL_PREDICTION_TV_MAX,
        "min_valid_cell_fraction": MIN_VALID_CELL_FRACTION,
        "min_cell_pass_fraction": MIN_CELL_PASS_FRACTION,
        "stage67_run_id": s110.STAGE67_RUN_ID,
        "stage111_run_id": s114.STAGE111_RUN_ID,
        "stage122_run_id": STAGE122_RUN_ID,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 123 is fixed to the completed Stage-122 leave-one-band-out ratio templates and the exact "
            "Stage-67/111 frozen diagnostic lineage; it may not retune physics, wall/collision/source treatment, "
            "reconstruction, transport, floors, normalization, source relaxation, velocity quadrature, diagnostic "
            "window, decision guards, or any failed MUSCL parameter"
        )
    if GRID != (64, 64) or RULE != (40, 96) or RADIAL_NODES_PER_SHELL != 10 or PAIR_SECTORS != (5, 6):
        raise ValueError("Stage 123 requires the exact retained 64x64, 40x96, ten-node, sectors-5+6 design")


def _load_stage122(root: str | Path, record_path: str | Path):
    root = Path(root)
    expected = {
        "summary.json": STAGE122_SUMMARY_SHA256,
        "common_kernel_distribution_ratio.npz": STAGE122_PROFILES_SHA256,
    }
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Stage-122 checksum mismatch: {name}")

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    if summary.get("stage") != 122 or summary.get("decision") != STAGE122_DECISION or summary.get("finite") is not True:
        raise ValueError("Stage-122 artifact does not authorize Stage 123")
    checks = (
        record.get("stage") == 122,
        record.get("decision") == STAGE122_DECISION,
        record.get("source_head") == STAGE122_SOURCE_HEAD,
        record.get("workflow_status") == "completed",
        record.get("workflow_conclusion") == "success",
        record.get("workflow_run_id") == STAGE122_RUN_ID,
        record.get("workflow_job_id") == STAGE122_JOB_ID,
        record.get("artifact_id") == STAGE122_ARTIFACT_ID,
        record.get("artifact_sha256") == STAGE122_ARTIFACT_SHA256,
        record.get("summary_sha256") == STAGE122_SUMMARY_SHA256,
        record.get("common_kernel_distribution_ratio_sha256") == STAGE122_PROFILES_SHA256,
        record.get("tests", {}).get("passed") == 6,
        record.get("tests", {}).get("failed") == 0,
    )
    if not all(checks):
        raise ValueError("Committed Stage-122 provenance does not authorize Stage 123")

    with np.load(root / "common_kernel_distribution_ratio.npz") as data:
        needed = {"phi_common", "psi_common", "loo_templates"}
        if not needed.issubset(data.files):
            raise ValueError("Stage-122 ratio payload is incomplete")
        arrays = {name: np.asarray(data[name], dtype=np.float64).copy() for name in needed}
    for name, value in arrays.items():
        if value.shape != (3, 10) or not np.isfinite(value).all() or np.any(value <= 0.0):
            raise ValueError(f"Stage-122 array {name} is invalid")
    return summary, record, arrays


def _normalize_last_axis(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(a, dtype=np.float64)
    if x.ndim < 2 or x.shape[-1] != RADIAL_NODES_PER_SHELL or not np.isfinite(x).all() or np.any(x < 0.0):
        raise ValueError("Invalid Stage-123 radial-cell payload")
    total = x.sum(axis=-1)
    out = np.zeros_like(x)
    valid = total > 0.0
    out[valid] = x[valid] / total[valid, None]
    return out, valid


def cell_prediction_metrics(phi: np.ndarray, psi: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p, pvalid = _normalize_last_axis(phi)
    q, qvalid = _normalize_last_axis(psi)
    t = np.asarray(template, dtype=np.float64)
    if t.shape != (RADIAL_NODES_PER_SHELL,) or not np.isfinite(t).all() or np.any(t <= 0.0):
        raise ValueError("Invalid Stage-123 held-out ratio template")
    pred_raw = q * t
    pred, predvalid = _normalize_last_axis(pred_raw)
    valid = pvalid & qvalid & predvalid
    cosine = np.full(p.shape[:-1], np.nan, dtype=np.float64)
    overlap = np.full(p.shape[:-1], np.nan, dtype=np.float64)
    tv = np.full(p.shape[:-1], np.nan, dtype=np.float64)
    if np.any(valid):
        pv = p[valid]
        rv = pred[valid]
        denom = np.linalg.norm(pv, axis=-1) * np.linalg.norm(rv, axis=-1)
        cosine[valid] = np.sum(pv * rv, axis=-1) / np.maximum(denom, 1.0e-300)
        overlap[valid] = np.minimum(pv, rv).sum(axis=-1)
        tv[valid] = 0.5 * np.abs(pv - rv).sum(axis=-1)
    return cosine, overlap, tv, valid


def stage123_decision(*, finite: bool, parent_closure: float, valid_fractions: list[float], pass_fractions: list[float]) -> str:
    if not finite:
        return NONFINITE
    if parent_closure > PARENT_CLOSURE_TOLERANCE:
        return CLOSURE_BLOCKER
    if all(v >= MIN_VALID_CELL_FRACTION for v in valid_fractions) and all(p >= MIN_CELL_PASS_FRACTION for p in pass_fractions):
        return PERSISTENT
    return AGGREGATION_ONLY


def _build_common_kernel_cell_profiles(distributions: Path, maps: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(distributions) as data:
        vx = np.asarray(data["vx"], dtype=np.float64)
        vy = np.asarray(data["vy"], dtype=np.float64)
        weight = np.asarray(data["weight"], dtype=np.float64)
        shell = s110._radial_shell_indices(vx, vy) == DOMINANT_RADIAL_SHELL
        svx, svy, sw = vx[shell], vy[shell], weight[shell]
        sector = s114.angular_sector_indices(svx, svy)
        node = s119.radial_node_indices_within_shell(svx, svy)
        support = np.isin(sector, np.asarray(PAIR_SECTORS, dtype=int))
        if int(np.count_nonzero(support)) != RADIAL_NODES_PER_SHELL * len(PAIR_SECTORS) * 12:
            raise ValueError("Stage-123 shell-1 sectors-5+6 support does not contain the expected 240 ordinates")

        out: dict[str, np.ndarray] = {}
        common_kernel = np.abs(svx)
        for name in ("phi", "psi"):
            change = s119._x_same_sign_change_pointwise(np.asarray(data[name], dtype=np.float64)[..., shell])
            growth = np.asarray(maps[f"{name}_growth_amplitude"], dtype=np.float64)
            if growth.shape != GRID or not np.isfinite(growth).all() or np.any(growth < 0.0):
                raise ValueError(f"Invalid Stage-111 {name} growth-amplitude map")
            density = change * sw[None, None, :] * common_kernel[None, None, :] * growth[..., None]
            profile = np.zeros((*GRID, RADIAL_NODES_PER_SHELL), dtype=np.float64)
            for j in range(RADIAL_NODES_PER_SHELL):
                mask = support & (node == j)
                if int(np.count_nonzero(mask)) != len(PAIR_SECTORS) * 12:
                    raise ValueError("Each Stage-123 paired radial node must contain exactly 24 ordinates")
                profile[..., j] = np.sum(density[..., mask], axis=-1)
            scale = max(float(np.max(np.abs(profile))), 1.0)
            if not np.isfinite(profile).all() or float(np.min(profile)) < -1.0e-14 * scale:
                raise ValueError(f"Nonfinite or materially negative Stage-123 {name} cell profile")
            profile[profile < 0.0] = 0.0
            out[name] = profile
    return out["phi"], out["psi"]


def _quantile(values: np.ndarray, q: float) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else float("nan")


def run(
    stage67_dir: str | Path,
    stage111_dir: str | Path,
    stage122_dir: str | Path,
    stage122_record_path: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage123_design(**design)
    _, distributions = s110._load_stage67(stage67_dir)
    _, maps = s114._load_stage111(stage111_dir)
    parent_summary, _, parent = _load_stage122(stage122_dir, stage122_record_path)

    phi_raw, psi_raw = _build_common_kernel_cell_profiles(distributions, maps)
    band_masks = s114.wall_distance_band_masks()
    cosine_map = np.full(GRID, np.nan, dtype=np.float64)
    overlap_map = np.full(GRID, np.nan, dtype=np.float64)
    tv_map = np.full(GRID, np.nan, dtype=np.float64)
    valid_map = np.zeros(GRID, dtype=bool)
    pass_map = np.zeros(GRID, dtype=bool)
    band_index = np.full(GRID, -1, dtype=np.int8)

    metrics: dict[str, dict[str, float | int]] = {}
    valid_fractions: list[float] = []
    pass_fractions: list[float] = []
    parent_closure = 0.0

    for i, band in enumerate(BANDS):
        mask = np.asarray(band_masks[band], dtype=bool)
        if mask.shape != GRID or not np.any(mask):
            raise ValueError(f"Invalid Stage-123 wall-distance mask {band}")
        band_index[mask] = i
        phi_cells = phi_raw[mask]
        psi_cells = psi_raw[mask]
        cosine, overlap, tv, valid = cell_prediction_metrics(phi_cells, psi_cells, parent["loo_templates"][i])
        passed = valid & (cosine >= CELL_PREDICTION_COSINE_MIN) & (tv <= CELL_PREDICTION_TV_MAX)

        cosine_map[mask] = cosine
        overlap_map[mask] = overlap
        tv_map[mask] = tv
        valid_map[mask] = valid
        pass_map[mask] = passed

        count = int(mask.sum())
        valid_count = int(valid.sum())
        valid_fraction = float(valid_count / count)
        pass_fraction = float(passed.sum() / max(valid_count, 1))
        valid_fractions.append(valid_fraction)
        pass_fractions.append(pass_fraction)

        phi_agg, phi_agg_valid = _normalize_last_axis(phi_cells.sum(axis=0, keepdims=True))
        psi_agg, psi_agg_valid = _normalize_last_axis(psi_cells.sum(axis=0, keepdims=True))
        if not bool(phi_agg_valid[0] and psi_agg_valid[0]):
            raise ValueError(f"Stage-123 {band} aggregate profile is empty")
        phi_closure = float(np.linalg.norm(phi_agg[0] - parent["phi_common"][i]) / max(float(np.linalg.norm(parent["phi_common"][i])), 1.0e-300))
        psi_closure = float(np.linalg.norm(psi_agg[0] - parent["psi_common"][i]) / max(float(np.linalg.norm(parent["psi_common"][i])), 1.0e-300))
        band_closure = max(phi_closure, psi_closure)
        parent_closure = max(parent_closure, band_closure)

        metrics[band] = {
            "cell_count": count,
            "valid_cell_count": valid_count,
            "valid_cell_fraction": valid_fraction,
            "passing_cell_count": int(passed.sum()),
            "passing_cell_fraction_of_valid": pass_fraction,
            "median_prediction_cosine": _quantile(cosine[valid], 0.50),
            "p10_prediction_cosine": _quantile(cosine[valid], 0.10),
            "median_prediction_overlap": _quantile(overlap[valid], 0.50),
            "median_prediction_total_variation": _quantile(tv[valid], 0.50),
            "p90_prediction_total_variation": _quantile(tv[valid], 0.90),
            "phi_parent_profile_closure_rel_l2": phi_closure,
            "psi_parent_profile_closure_rel_l2": psi_closure,
        }

    finite = bool(
        np.isfinite(phi_raw).all()
        and np.isfinite(psi_raw).all()
        and np.isfinite(parent_closure)
        and all(np.isfinite([float(v) for v in m.values()]).all() for m in metrics.values())
    )
    decision = stage123_decision(
        finite=finite,
        parent_closure=parent_closure,
        valid_fractions=valid_fractions,
        pass_fractions=pass_fractions,
    )

    aggregate = {
        "minimum_valid_cell_fraction": float(min(valid_fractions)),
        "minimum_passing_cell_fraction_of_valid": float(min(pass_fractions)),
        "minimum_band_median_prediction_cosine": float(min(float(metrics[b]["median_prediction_cosine"]) for b in BANDS)),
        "minimum_band_p10_prediction_cosine": float(min(float(metrics[b]["p10_prediction_cosine"]) for b in BANDS)),
        "minimum_band_median_prediction_overlap": float(min(float(metrics[b]["median_prediction_overlap"]) for b in BANDS)),
        "maximum_band_median_prediction_total_variation": float(max(float(metrics[b]["median_prediction_total_variation"]) for b in BANDS)),
        "maximum_band_p90_prediction_total_variation": float(max(float(metrics[b]["p90_prediction_total_variation"]) for b in BANDS)),
        "maximum_stage122_parent_profile_closure_rel_l2": float(parent_closure),
    }

    if decision == PERSISTENT:
        scientific_conclusion = (
            "The Stage-122 common-|c_x| distribution ratio persists at the individual spatial-cell level under a noncircular test: "
            "each cell is predicted with the leave-one-band-out ratio template built only from the other two wall-distance bands. "
            "This supports a genuine cellwise distribution-specific radial structure rather than a purely band-aggregation artifact and "
            "justifies one fixed spatial-transition-coherence audit. It does not establish limiter causality, MUSCL stability, endpoint "
            "convergence, q_av improvement, benchmark accuracy, or validation."
        )
    elif decision == AGGREGATION_ONLY:
        scientific_conclusion = (
            "The Stage-122 band-level common-|c_x| ratio does not satisfy the preregistered individual-cell persistence guards when "
            "predicted from other wall-distance bands. The stable band aggregate must therefore not be interpreted as a cellwise "
            "mechanism; one fixed within-band aggregation/cancellation audit is justified. No solver parameter is changed."
        )
    else:
        scientific_conclusion = (
            "Stage 123 is blocked by nonfinite data or failure to reconstruct the checksum-verified Stage-122 parent profiles. No "
            "mechanistic interpretation or parameter change is justified."
        )

    configuration = {
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
        "cell_template": "Stage-122 leave-one-band-out geometric-mean radial ratio from the other two fixed bands",
        "cell_prediction_cosine_min": CELL_PREDICTION_COSINE_MIN,
        "cell_prediction_total_variation_max": CELL_PREDICTION_TV_MAX,
        "minimum_valid_cell_fraction": MIN_VALID_CELL_FRACTION,
        "minimum_passing_cell_fraction": MIN_CELL_PASS_FRACTION,
        "model_retuning": False,
        "wall_retuning": False,
        "source_retuning": False,
        "reconstruction_retuning": False,
        "transport_retuning": False,
        "floor_retuning": False,
        "normalization_retuning": False,
        "velocity_grid_retuning": False,
        "solver_endpoint_advanced": False,
        "cross_knudsen_extension_permitted": False,
        "benchmark_or_validation_claim_permitted": False,
    }

    summary = {
        "stage": 123,
        "parent_stage122": {
            "run_id": STAGE122_RUN_ID,
            "job_id": STAGE122_JOB_ID,
            "artifact_id": STAGE122_ARTIFACT_ID,
            "source_head": STAGE122_SOURCE_HEAD,
            "decision": parent_summary["decision"],
        },
        "configuration": configuration,
        "metrics": metrics,
        "aggregate": aggregate,
        "finite": finite,
        "decision": decision,
        "scientific_conclusion": scientific_conclusion,
        "negative_result_guard": (
            "No model, wall, collision/source, reconstruction, transport, positivity/correction floor, normalization, source-relaxation, "
            "velocity-grid, or failed MUSCL parameter is changed. Stage 90 remains nonconverged, Stage 28 remains failed, Stage 89 "
            "remains unpromoted, and no solver endpoint or cross-Knudsen extension is attempted."
        ),
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        output / "cellwise_ratio_persistence.npz",
        phi_common_cell_profiles=phi_raw,
        psi_common_cell_profiles=psi_raw,
        prediction_cosine=cosine_map,
        prediction_overlap=overlap_map,
        prediction_total_variation=tv_map,
        valid_mask=valid_map,
        pass_mask=pass_map,
        band_index=band_index,
        loo_templates=parent["loo_templates"],
        parent_phi_common=parent["phi_common"],
        parent_psi_common=parent["psi_common"],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 123 fixed cellwise common-kernel ratio-persistence audit")
    parser.add_argument("--stage67-dir", required=True)
    parser.add_argument("--stage111-dir", required=True)
    parser.add_argument("--stage122-dir", required=True)
    parser.add_argument("--stage122-record", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run(args.stage67_dir, args.stage111_dir, args.stage122_dir, args.stage122_record, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
