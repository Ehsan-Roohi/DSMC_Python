from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import stage79_dominant_moment_radial_angular_gradient_audit as stage79
from . import stage80_dominant_shell_radial_node_angular_attribution_audit as stage80
from . import stage81_dominant_node_individual_ordinate_audit as stage81

STAGE67_COMPLETED_ENDPOINT = stage81.STAGE67_COMPLETED_ENDPOINT
STAGE86_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31253211394,
    "workflow_job_id": 93092657977,
    "workflow_conclusion": "success",
    "tests_passed": 300,
    "tests_failed": 0,
    "artifact_id": 9020634310,
    "artifact_size_bytes": 133526,
    "artifact_sha256": "a96bfb0d4696ebf425834081c4795a3f10c59ef20b995bf0098da99ca452095b",
    "source_head_sha": "d305dbd5250fa9f170ef5e06827d30adfe9cdc59",
    "summary_sha256": "a1dc308b502ebb66cba48ee77241e00516e930572f02f73c2153dcbfff4d59cf",
    "maps_sha256": "dba61681d96ca286dd57979c6e169eaf48a289c30cda90af2afdef39ec82f2d7",
    "decision": "stage86_boundary_zero_slope_stencil_primary_frozen_suppression_stage87_one_sided_boundary_slope_counterfactual_audit",
}

GRID = (64, 64)
KNUDSEN = 10.0
COLD_HOT_RATIO = 0.1
RULE = (40, 96)
POINT_COUNT = RULE[0] * RULE[1]
RADIAL_SCALE = 2.0
CHUNK_SIZE = 128
LIMITER = "minmod"
DOMINANT_MOMENT = "transverse_kinetic"
DOMINANT_GLOBAL_RADIAL_NODE = 21
VERTICAL_OBLIQUE_BINS = (1, 2, 5, 6)
OPPOSITE_SECTOR_PAIRS = ((1, 5), (2, 6))
BASELINE_CLOSURE_GUARD = 1.0e-10
BOUNDARY_JUMP_RECOVERY_PRIMARY_GUARD = 0.50
BOUNDARY_JUMP_RECOVERY_PARTIAL_GUARD = 0.10


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_stage87_design(**overrides: object) -> None:
    frozen = {
        "grid": GRID,
        "kn0": KNUDSEN,
        "cold_hot_ratio": COLD_HOT_RATIO,
        "rule": RULE,
        "radial_scale": RADIAL_SCALE,
        "chunk_size": CHUNK_SIZE,
        "limiter": LIMITER,
        "dominant_moment": DOMINANT_MOMENT,
        "dominant_global_radial_node": DOMINANT_GLOBAL_RADIAL_NODE,
        "vertical_oblique_bins": VERTICAL_OBLIQUE_BINS,
        "opposite_sector_pairs": OPPOSITE_SECTOR_PAIRS,
        "baseline_closure_guard": BASELINE_CLOSURE_GUARD,
        "boundary_jump_recovery_primary_guard": BOUNDARY_JUMP_RECOVERY_PRIMARY_GUARD,
        "boundary_jump_recovery_partial_guard": BOUNDARY_JUMP_RECOVERY_PARTIAL_GUARD,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 87 is frozen to the exact completed Stage-67 distributions and Stage-86 diagnostic endpoint; "
            "the only counterfactual is a one-sided boundary-cell slope at the two wall-adjacent interior x faces."
        )


def _validate_stage67(root: str | Path) -> dict[str, object]:
    root = Path(root)
    files = {
        "summary.json": "summary_sha256",
        "converged_full_distributions.npz": "distributions_sha256",
    }
    for name, key in files.items():
        path = root / name
        if not path.is_file() or _sha256(path) != str(STAGE67_COMPLETED_ENDPOINT[key]):
            raise ValueError(f"Stage-67 checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 67 or summary.get("decision") != STAGE67_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage-67 completed endpoint mismatch")
    return summary


def _validate_stage86(root: str | Path) -> dict[str, object]:
    root = Path(root)
    files = {
        "summary.json": "summary_sha256",
        "boundary_reconstruction_stencil_activation_maps.npz": "maps_sha256",
    }
    for name, key in files.items():
        path = root / name
        if not path.is_file() or _sha256(path) != str(STAGE86_COMPLETED_ENDPOINT[key]):
            raise ValueError(f"Stage-86 checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 86 or summary.get("decision") != STAGE86_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage-86 completed endpoint mismatch")
    return summary


def one_sided_boundary_slopes_x(distribution: np.ndarray) -> np.ndarray:
    """Retain minmod internally and use the sole one-sided difference at boundary cells."""
    distribution = np.asarray(distribution, dtype=np.float64)
    if distribution.ndim != 3 or distribution.shape[1] < 3:
        raise ValueError("Stage 87 requires (ny,nx,nv) data with at least three x cells")
    slope = stage79.limited_slopes_x(distribution)
    slope[:, 0] = distribution[:, 1] - distribution[:, 0]
    slope[:, -1] = distribution[:, -1] - distribution[:, -2]
    return slope


def counterfactual_interior_x_face_flux_difference_chunk(distribution: np.ndarray, vx: np.ndarray) -> np.ndarray:
    distribution = np.asarray(distribution, dtype=np.float64)
    vx = np.asarray(vx, dtype=np.float64)
    slope = one_sided_boundary_slopes_x(distribution)
    delta = np.zeros((distribution.shape[0], distribution.shape[1] - 1, distribution.shape[2]), dtype=np.float64)
    positive = vx > 0.0
    negative = vx < 0.0
    if np.any(positive):
        delta[..., positive] = 0.5 * vx[positive][None, None, :] * slope[:, :-1, positive]
    if np.any(negative):
        delta[..., negative] = -0.5 * vx[negative][None, None, :] * slope[:, 1:, negative]
    return delta


def _norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64).ravel()))


def _divergence(face: np.ndarray) -> np.ndarray:
    face = np.asarray(face, dtype=np.float64)
    cell = np.zeros(GRID, dtype=np.float64)
    cell[:, :-1] -= face
    cell[:, 1:] += face
    return cell


def dominant_node_pair_maps(
    phi: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if phi.shape != (GRID[0], GRID[1], POINT_COUNT):
        raise ValueError("Stage 87 requires exact 64x64x3840 Stage-67 phi")
    v = stage79.macroscopic_v(phi, vy, weight, CHUNK_SIZE)
    v_mid = 0.5 * (v[:, :-1] + v[:, 1:])
    node_labels = stage80.radial_node_indices(vx, vy)
    angular_labels = stage79.angular_bin_indices(vx, vy)
    baseline = np.zeros((2, GRID[0], GRID[1] - 1), dtype=np.float64)
    counterfactual = np.zeros_like(baseline)
    dx = 1.0 / GRID[1]

    for start in range(0, POINT_COUNT, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, POINT_COUNT)
        sl = slice(start, stop)
        d0 = stage79.interior_x_face_flux_difference_chunk(phi[..., sl], vx[sl])
        d1 = counterfactual_interior_x_face_flux_difference_chunk(phi[..., sl], vx[sl])
        B = vy[sl][None, None, :] - v_mid[..., None]
        factor = (0.5 * B * B * B) * weight[sl][None, None, :] / dx
        w0 = d0 * factor
        w1 = d1 * factor
        nodes = node_labels[sl]
        bins = angular_labels[sl]
        for pidx, pair in enumerate(OPPOSITE_SECTOR_PAIRS):
            selected = (nodes == DOMINANT_GLOBAL_RADIAL_NODE) & np.isin(bins, pair)
            if np.any(selected):
                baseline[pidx] += np.sum(w0[..., selected], axis=-1)
                counterfactual[pidx] += np.sum(w1[..., selected], axis=-1)

    baseline_cell = np.stack([_divergence(face) for face in baseline])
    counterfactual_cell = np.stack([_divergence(face) for face in counterfactual])
    return baseline, counterfactual, baseline_cell, counterfactual_cell


def counterfactual_metrics(
    baseline_face: np.ndarray,
    counterfactual_face: np.ndarray,
    baseline_cell: np.ndarray,
    counterfactual_cell: np.ndarray,
    retained_stage86_face: np.ndarray,
) -> dict[str, object]:
    closure = _norm(baseline_face - retained_stage86_face) / max(_norm(retained_stage86_face), 1.0e-300)
    rows: list[dict[str, object]] = []
    recovery_values: list[float] = []
    for pidx, pair in enumerate(OPPOSITE_SECTOR_PAIRS):
        for side, wi, fi in (("left", 0, 1), ("right", -1, -2)):
            base_wall = baseline_face[pidx, :, wi]
            cf_wall = counterfactual_face[pidx, :, wi]
            first = baseline_face[pidx, :, fi]
            jump = first - base_wall
            change = cf_wall - base_wall
            recovery = _norm(change) / max(_norm(jump), 1.0e-300)
            recovery_values.append(recovery)
            rows.append({
                "angular_bin_pair": list(pair),
                "side": side,
                "baseline_wall_l2": _norm(base_wall),
                "counterfactual_wall_l2": _norm(cf_wall),
                "first_interior_l2": _norm(first),
                "baseline_wall_to_first_l2_ratio": _norm(base_wall) / max(_norm(first), 1.0e-300),
                "counterfactual_wall_to_first_l2_ratio": _norm(cf_wall) / max(_norm(first), 1.0e-300),
                "counterfactual_change_l2": _norm(change),
                "wall_to_first_jump_l2": _norm(jump),
                "boundary_jump_recovery_fraction": recovery,
            })
    face_change = counterfactual_face - baseline_face
    cell_change = counterfactual_cell - baseline_cell
    interior_leak = float(np.max(np.abs(face_change[:, :, 1:-1])))
    return {
        "finite": bool(
            np.all(np.isfinite(baseline_face)) and np.all(np.isfinite(counterfactual_face))
            and np.all(np.isfinite(baseline_cell)) and np.all(np.isfinite(counterfactual_cell))
        ),
        "baseline_stage86_relative_l2_closure_error": closure,
        "maximum_counterfactual_change_away_from_wall_adjacent_faces": interior_leak,
        "maximum_boundary_jump_recovery_fraction": max(recovery_values),
        "minimum_boundary_jump_recovery_fraction": min(recovery_values),
        "mean_boundary_jump_recovery_fraction": float(np.mean(recovery_values)),
        "counterfactual_face_change_relative_l2": _norm(face_change) / max(_norm(baseline_face), 1.0e-300),
        "counterfactual_cell_divergence_change_relative_l2": _norm(cell_change) / max(_norm(baseline_cell), 1.0e-300),
        "rows": rows,
    }


def stage87_decision(metrics: dict[str, object]) -> str:
    if not bool(metrics["finite"]):
        return "stage87_nonfinite_counterfactual_blocker"
    if float(metrics["baseline_stage86_relative_l2_closure_error"]) > BASELINE_CLOSURE_GUARD:
        return "stage87_baseline_reconstruction_closure_blocker"
    recovery = float(metrics["mean_boundary_jump_recovery_fraction"])
    if recovery >= BOUNDARY_JUMP_RECOVERY_PRIMARY_GUARD:
        return "stage87_one_sided_boundary_slope_large_frozen_effect_stage88_full_moment_boundary_counterfactual_audit"
    if recovery >= BOUNDARY_JUMP_RECOVERY_PARTIAL_GUARD:
        return "stage87_one_sided_boundary_slope_partial_frozen_effect_stage88_remaining_direction_counterfactual_decomposition"
    return "stage87_one_sided_boundary_slope_weak_frozen_effect_stage88_boundary_distribution_curvature_audit"


def run_stage87(stage67_artifact_dir: str | Path, stage86_artifact_dir: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage87_design(**design)
    retained67 = _validate_stage67(stage67_artifact_dir)
    retained86 = _validate_stage86(stage86_artifact_dir)
    with np.load(Path(stage67_artifact_dir) / "converged_full_distributions.npz") as z:
        phi = np.asarray(z["phi"], dtype=np.float64)
        vx = np.asarray(z["vx"], dtype=np.float64)
        vy = np.asarray(z["vy"], dtype=np.float64)
        weight = np.asarray(z["weight"], dtype=np.float64)
    with np.load(Path(stage86_artifact_dir) / "boundary_reconstruction_stencil_activation_maps.npz") as z:
        retained_face = np.asarray(z["retained_pair_face_groups"], dtype=np.float64)

    base_face, cf_face, base_cell, cf_cell = dominant_node_pair_maps(phi, vx, vy, weight)
    metrics = counterfactual_metrics(base_face, cf_face, base_cell, cf_cell, retained_face)
    decision = stage87_decision(metrics)
    if decision.endswith("full_moment_boundary_counterfactual_audit"):
        next_scope = (
            "Propagate the same frozen one-sided boundary-slope counterfactual across all velocity nodes and retained moment components "
            "offline, checking conservation/moment closure before any solver experiment."
        )
    elif decision.endswith("remaining_direction_counterfactual_decomposition"):
        next_scope = (
            "Decompose the partial frozen response into boundary-upwind and remaining velocity directions using the same fixed one-sided slope; "
            "do not advance the solver."
        )
    else:
        next_scope = (
            "Audit boundary-cell distribution curvature and one-sided difference geometry at the frozen endpoint; do not modify or rerun the solver."
        )

    summary = {
        "stage": 87,
        "description": "Offline one-sided boundary-slope counterfactual at only the two wall-adjacent interior x faces.",
        "finite": bool(metrics["finite"]),
        "configuration": {
            "grid": [64, 64], "kn0": 10.0, "cold_hot_ratio": 0.1,
            "radial_nodes": 40, "angular_nodes": 96, "point_count": 3840, "radial_scale": 2.0,
            "limiter": "minmod", "counterfactual_boundary_slope": "one_sided_first_difference",
            "counterfactual_face_scope": "wall_adjacent_interior_x_faces_only",
            "dominant_moment": DOMINANT_MOMENT, "dominant_global_radial_node": DOMINANT_GLOBAL_RADIAL_NODE,
            "vertical_oblique_bins": list(VERTICAL_OBLIQUE_BINS),
            "opposite_sector_pairs": [list(p) for p in OPPOSITE_SECTOR_PAIRS],
            "baseline_closure_guard": BASELINE_CLOSURE_GUARD,
            "boundary_jump_recovery_primary_guard": BOUNDARY_JUMP_RECOVERY_PRIMARY_GUARD,
            "boundary_jump_recovery_partial_guard": BOUNDARY_JUMP_RECOVERY_PARTIAL_GUARD,
            "solver_rerun_count": 0, "physical_parameter_retuning": False,
            "collision_parameter_retuning": False, "correction_floor_retuning": False,
            "source_relaxation_retuning": False, "transport_parameter_retuning": False,
            "wall_model_retuning": False, "normalization_retuning": False,
            "velocity_quadrature_retuning": False, "failed_muscl_endpoint_rehabilitated": False,
            "cross_knudsen_extension_permitted": False, "validation_claim_permitted": False,
        },
        "retained_stage67_decision": retained67["decision"],
        "retained_stage86_decision": retained86["decision"],
        "counterfactual_metrics": metrics,
        "decision": decision,
        "positive_findings": [
            "The exact Stage-67 frozen distribution is used without time advancement, and the Stage-87 baseline reproduces the exact Stage-86 retained pair maps before applying the counterfactual.",
            "Only the boundary-cell slope feeding the wall-adjacent interior x face is replaced by the unique one-sided first difference; all interior minmod slopes and all physical parameters remain unchanged."
        ],
        "negative_findings": [
            "This is an offline frozen-state counterfactual, not a stable solver endpoint, not an error correction, and not evidence that q_av would improve in a rerun.",
            "The one-sided boundary difference is a diagnostic perturbation rather than an adopted boundary reconstruction.",
            "The failed Stage-28 MUSCL endpoint remains unrecovered, cross-Knudsen extension is prohibited, and no validation claim is permitted.",
            "No physical, collision, correction-floor, source-relaxation, transport, wall, normalization, or velocity-quadrature parameter is retuned."
        ],
        "scientifically_justified_next_scope": next_scope,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        output / "one_sided_boundary_slope_counterfactual_maps.npz",
        baseline_pair_face_groups=base_face,
        counterfactual_pair_face_groups=cf_face,
        face_change_groups=cf_face - base_face,
        baseline_pair_cell_divergence_groups=base_cell,
        counterfactual_pair_cell_divergence_groups=cf_cell,
        cell_divergence_change_groups=cf_cell - base_cell,
        opposite_sector_pairs=np.asarray(OPPOSITE_SECTOR_PAIRS, dtype=np.int16),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage86-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage87(args.stage67_artifact_dir, args.stage86_artifact_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
