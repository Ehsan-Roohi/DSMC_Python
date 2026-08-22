from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import stage76_local_velocity_frame_jump_audit as stage76

STAGE67_COMPLETED_ENDPOINT = stage76.STAGE67_COMPLETED_ENDPOINT
STAGE87_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31254047473,
    "workflow_job_id": 93094637633,
    "workflow_conclusion": "success",
    "tests_passed": 305,
    "tests_failed": 0,
    "artifact_id": 9022285440,
    "artifact_size_bytes": 255092,
    "artifact_sha256": "d12bb5d76c47c0506952f191e381759cf0bd27f4eb45f79621a2fdd1e16d50e5",
    "source_head_sha": "6c97dc185b985743e75400241c509f4f308ecb02",
    "summary_sha256": "c36126bcc089f2c02f6943a2680a213f6c1cd9b08299d40befeab38d22dd53b8",
    "maps_sha256": "125812e60c5bbae6c853a3a22263eac19abcba8c81084fbe27f57e77cce0010b",
    "decision": "stage87_one_sided_boundary_slope_large_frozen_effect_stage88_full_moment_boundary_counterfactual_audit",
}

GRID = (64, 64)
KNUDSEN = 10.0
COLD_HOT_RATIO = 0.1
RULE = (40, 96)
POINT_COUNT = RULE[0] * RULE[1]
RADIAL_SCALE = 2.0
CHUNK_SIZE = 128
LIMITER = "minmod"
MOMENT_NAMES = ("streamwise_kinetic", "transverse_kinetic", "reduced_internal")
BASELINE_CLOSURE_GUARD = 1.0e-10
CONSERVATION_GUARD = 1.0e-12
MOMENT_SUM_CLOSURE_GUARD = 1.0e-12
BOUNDARY_JUMP_RECOVERY_PRIMARY_GUARD = 0.50
BOUNDARY_JUMP_RECOVERY_PARTIAL_GUARD = 0.10


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_stage88_design(**overrides: object) -> None:
    frozen = {
        "grid": GRID,
        "kn0": KNUDSEN,
        "cold_hot_ratio": COLD_HOT_RATIO,
        "rule": RULE,
        "radial_scale": RADIAL_SCALE,
        "chunk_size": CHUNK_SIZE,
        "limiter": LIMITER,
        "moment_names": MOMENT_NAMES,
        "baseline_closure_guard": BASELINE_CLOSURE_GUARD,
        "conservation_guard": CONSERVATION_GUARD,
        "moment_sum_closure_guard": MOMENT_SUM_CLOSURE_GUARD,
        "boundary_jump_recovery_primary_guard": BOUNDARY_JUMP_RECOVERY_PRIMARY_GUARD,
        "boundary_jump_recovery_partial_guard": BOUNDARY_JUMP_RECOVERY_PARTIAL_GUARD,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 88 is frozen to the exact completed Stage-67 distributions and Stage-87 counterfactual endpoint, "
            "all 3840 velocity nodes, the three fixed common-frame heat-flux moments, and the preregistered closure, "
            "conservation, and boundary-jump guards; no solver rerun or parameter retuning is permitted."
        )


def _validate_stage67(root: str | Path) -> dict[str, object]:
    return stage76._validate_artifact(
        root,
        STAGE67_COMPLETED_ENDPOINT,
        {
            "summary.json": "summary_sha256",
            "converged_full_distributions.npz": "distributions_sha256",
            "steady_residual_moment_maps.npz": "residual_maps_sha256",
        },
        67,
    )


def _validate_stage87(root: str | Path) -> dict[str, object]:
    root = Path(root)
    files = {
        "summary.json": "summary_sha256",
        "one_sided_boundary_slope_counterfactual_maps.npz": "maps_sha256",
    }
    for name, key in files.items():
        path = root / name
        if not path.is_file() or _sha256(path) != str(STAGE87_COMPLETED_ENDPOINT[key]):
            raise ValueError(f"Stage-87 checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 87 or summary.get("decision") != STAGE87_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage-87 completed endpoint mismatch")
    return summary


def one_sided_boundary_slopes_x(distribution: np.ndarray) -> np.ndarray:
    distribution = np.asarray(distribution, dtype=np.float64)
    if distribution.ndim != 3 or distribution.shape[1] < 3:
        raise ValueError("Stage 88 requires (ny,nx,nv) data with at least three x cells")
    slope = stage76.limited_slopes_x(distribution)
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


def divergence_from_interior_faces(face: np.ndarray) -> np.ndarray:
    face = np.asarray(face, dtype=np.float64)
    if face.shape != (GRID[0], GRID[1] - 1):
        raise ValueError("Stage 88 requires a 64x63 interior x-face map")
    cell = np.zeros(GRID, dtype=np.float64)
    cell[:, :-1] -= face
    cell[:, 1:] += face
    return cell


def component_divergence(face_components: np.ndarray) -> np.ndarray:
    face_components = np.asarray(face_components, dtype=np.float64)
    if face_components.shape != (len(MOMENT_NAMES), GRID[0], GRID[1] - 1):
        raise ValueError("Stage 88 requires three 64x63 moment face maps")
    return np.stack([divergence_from_interior_faces(face) for face in face_components], axis=0)


def full_moment_face_maps(
    phi: np.ndarray,
    psi: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if phi.shape != psi.shape or phi.shape != (GRID[0], GRID[1], POINT_COUNT):
        raise ValueError("Stage 88 requires exact 64x64x3840 Stage-67 phi and psi")
    if vx.shape != (POINT_COUNT,) or vy.shape != (POINT_COUNT,) or weight.shape != (POINT_COUNT,):
        raise ValueError("Stage 88 requires the exact 3840-point Stage-67 velocity rule")

    _, u, v = stage76.macroscopic_velocity(phi, vx, vy, weight, CHUNK_SIZE)
    u_mid = 0.5 * (u[:, :-1] + u[:, 1:])
    v_mid = 0.5 * (v[:, :-1] + v[:, 1:])
    baseline = np.zeros((len(MOMENT_NAMES), GRID[0], GRID[1] - 1), dtype=np.float64)
    counterfactual = np.zeros_like(baseline)
    dx = 1.0 / GRID[1]

    for start in range(0, POINT_COUNT, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, POINT_COUNT)
        sl = slice(start, stop)
        dphi0 = stage76.interior_x_face_flux_difference_chunk(phi[..., sl], vx[sl])
        dpsi0 = stage76.interior_x_face_flux_difference_chunk(psi[..., sl], vx[sl])
        dphi1 = counterfactual_interior_x_face_flux_difference_chunk(phi[..., sl], vx[sl])
        dpsi1 = counterfactual_interior_x_face_flux_difference_chunk(psi[..., sl], vx[sl])
        A = vx[sl][None, None, :] - u_mid[..., None]
        B = vy[sl][None, None, :] - v_mid[..., None]
        w = weight[sl][None, None, :] / dx
        k_stream = 0.5 * B * A * A
        k_transverse = 0.5 * B * B * B
        k_internal = 0.5 * B

        baseline[0] += np.sum(dphi0 * k_stream * w, axis=-1)
        baseline[1] += np.sum(dphi0 * k_transverse * w, axis=-1)
        baseline[2] += np.sum(dpsi0 * k_internal * w, axis=-1)
        counterfactual[0] += np.sum(dphi1 * k_stream * w, axis=-1)
        counterfactual[1] += np.sum(dphi1 * k_transverse * w, axis=-1)
        counterfactual[2] += np.sum(dpsi1 * k_internal * w, axis=-1)

    return baseline, counterfactual


def _conservation_ratio(cell_map: np.ndarray) -> float:
    a = np.asarray(cell_map, dtype=np.float64)
    return abs(float(np.sum(a))) / max(float(np.sum(np.abs(a))), 1.0e-300)


def _boundary_row(name: str, baseline_face: np.ndarray, counterfactual_face: np.ndarray, side: str) -> dict[str, object]:
    if side == "left":
        wall_index, first_index = 0, 1
    elif side == "right":
        wall_index, first_index = -1, -2
    else:
        raise ValueError("side must be left or right")
    baseline_wall = baseline_face[:, wall_index]
    counterfactual_wall = counterfactual_face[:, wall_index]
    first = baseline_face[:, first_index]
    jump = first - baseline_wall
    change = counterfactual_wall - baseline_wall
    return {
        "moment": name,
        "side": side,
        "baseline_wall_l2": _norm(baseline_wall),
        "counterfactual_wall_l2": _norm(counterfactual_wall),
        "first_interior_l2": _norm(first),
        "baseline_wall_to_first_l2_ratio": _norm(baseline_wall) / max(_norm(first), 1.0e-300),
        "counterfactual_wall_to_first_l2_ratio": _norm(counterfactual_wall) / max(_norm(first), 1.0e-300),
        "counterfactual_change_l2": _norm(change),
        "wall_to_first_jump_l2": _norm(jump),
        "boundary_jump_recovery_fraction": _norm(change) / max(_norm(jump), 1.0e-300),
    }


def full_moment_metrics(
    baseline_face: np.ndarray,
    counterfactual_face: np.ndarray,
    retained_baseline_face: np.ndarray,
) -> dict[str, object]:
    baseline_face = np.asarray(baseline_face, dtype=np.float64)
    counterfactual_face = np.asarray(counterfactual_face, dtype=np.float64)
    retained_baseline_face = np.asarray(retained_baseline_face, dtype=np.float64)
    baseline_cell = component_divergence(baseline_face)
    counterfactual_cell = component_divergence(counterfactual_face)
    face_change = counterfactual_face - baseline_face
    cell_change = counterfactual_cell - baseline_cell

    retained_rel = _norm(baseline_face - retained_baseline_face) / max(_norm(retained_baseline_face), 1.0e-300)
    interior_leak = float(np.max(np.abs(face_change[:, :, 1:-1])))

    baseline_total_face = np.sum(baseline_face, axis=0)
    counterfactual_total_face = np.sum(counterfactual_face, axis=0)
    baseline_total_cell = divergence_from_interior_faces(baseline_total_face)
    counterfactual_total_cell = divergence_from_interior_faces(counterfactual_total_face)
    summed_baseline_cell = np.sum(baseline_cell, axis=0)
    summed_counterfactual_cell = np.sum(counterfactual_cell, axis=0)
    baseline_moment_sum_rel = _norm(summed_baseline_cell - baseline_total_cell) / max(_norm(baseline_total_cell), 1.0e-300)
    counterfactual_moment_sum_rel = _norm(summed_counterfactual_cell - counterfactual_total_cell) / max(_norm(counterfactual_total_cell), 1.0e-300)

    conservation_ratios = []
    for maps in (baseline_cell, counterfactual_cell, cell_change):
        conservation_ratios.extend(_conservation_ratio(maps[i]) for i in range(len(MOMENT_NAMES)))
    conservation_ratios.extend(
        [
            _conservation_ratio(baseline_total_cell),
            _conservation_ratio(counterfactual_total_cell),
            _conservation_ratio(counterfactual_total_cell - baseline_total_cell),
        ]
    )

    component_rows = []
    component_change_abs = np.sum(np.abs(cell_change), axis=(1, 2))
    component_change_abs_total = max(float(np.sum(component_change_abs)), 1.0e-300)
    per_component = {}
    for i, name in enumerate(MOMENT_NAMES):
        left = _boundary_row(name, baseline_face[i], counterfactual_face[i], "left")
        right = _boundary_row(name, baseline_face[i], counterfactual_face[i], "right")
        component_rows.extend([left, right])
        per_component[name] = {
            "face_change_relative_l2": _norm(face_change[i]) / max(_norm(baseline_face[i]), 1.0e-300),
            "cell_divergence_change_relative_l2": _norm(cell_change[i]) / max(_norm(baseline_cell[i]), 1.0e-300),
            "cell_change_absolute_share": float(component_change_abs[i] / component_change_abs_total),
            "left_boundary_jump_recovery_fraction": float(left["boundary_jump_recovery_fraction"]),
            "right_boundary_jump_recovery_fraction": float(right["boundary_jump_recovery_fraction"]),
        }

    total_rows = [
        _boundary_row("total", baseline_total_face, counterfactual_total_face, "left"),
        _boundary_row("total", baseline_total_face, counterfactual_total_face, "right"),
    ]
    total_recovery = [float(row["boundary_jump_recovery_fraction"]) for row in total_rows]
    total_face_change = counterfactual_total_face - baseline_total_face
    total_cell_change = counterfactual_total_cell - baseline_total_cell
    intercomponent_face_change_retention = float(
        np.sum(np.abs(total_face_change)) / max(float(np.sum(np.abs(face_change))), 1.0e-300)
    )
    intercomponent_cell_change_retention = float(
        np.sum(np.abs(total_cell_change)) / max(float(np.sum(np.abs(cell_change))), 1.0e-300)
    )
    dominant_index = int(np.argmax(component_change_abs))

    finite = bool(
        np.all(np.isfinite(baseline_face))
        and np.all(np.isfinite(counterfactual_face))
        and np.all(np.isfinite(baseline_cell))
        and np.all(np.isfinite(counterfactual_cell))
        and np.all(np.isfinite(np.asarray(conservation_ratios)))
        and np.isfinite(retained_rel)
        and np.isfinite(baseline_moment_sum_rel)
        and np.isfinite(counterfactual_moment_sum_rel)
    )

    return {
        "finite": finite,
        "baseline_stage76_component_relative_l2_closure_error": retained_rel,
        "maximum_counterfactual_change_away_from_wall_adjacent_faces": interior_leak,
        "maximum_global_conservation_ratio": max(conservation_ratios),
        "baseline_moment_sum_cell_relative_l2_error": baseline_moment_sum_rel,
        "counterfactual_moment_sum_cell_relative_l2_error": counterfactual_moment_sum_rel,
        "total_face_change_relative_l2": _norm(total_face_change) / max(_norm(baseline_total_face), 1.0e-300),
        "total_cell_divergence_change_relative_l2": _norm(total_cell_change) / max(_norm(baseline_total_cell), 1.0e-300),
        "minimum_total_boundary_jump_recovery_fraction": min(total_recovery),
        "maximum_total_boundary_jump_recovery_fraction": max(total_recovery),
        "mean_total_boundary_jump_recovery_fraction": float(np.mean(total_recovery)),
        "intercomponent_face_change_retention_ratio": intercomponent_face_change_retention,
        "intercomponent_cell_change_retention_ratio": intercomponent_cell_change_retention,
        "dominant_cell_change_component": MOMENT_NAMES[dominant_index],
        "dominant_cell_change_absolute_share": float(component_change_abs[dominant_index] / component_change_abs_total),
        "per_component": per_component,
        "component_boundary_rows": component_rows,
        "total_boundary_rows": total_rows,
    }


def stage88_decision(metrics: dict[str, object]) -> str:
    if not bool(metrics["finite"]):
        return "stage88_nonfinite_full_moment_counterfactual_blocker"
    if float(metrics["baseline_stage76_component_relative_l2_closure_error"]) > BASELINE_CLOSURE_GUARD:
        return "stage88_baseline_common_frame_component_closure_blocker"
    if float(metrics["maximum_counterfactual_change_away_from_wall_adjacent_faces"]) != 0.0:
        return "stage88_counterfactual_scope_leakage_blocker"
    if float(metrics["maximum_global_conservation_ratio"]) > CONSERVATION_GUARD:
        return "stage88_counterfactual_conservation_blocker"
    if max(
        float(metrics["baseline_moment_sum_cell_relative_l2_error"]),
        float(metrics["counterfactual_moment_sum_cell_relative_l2_error"]),
    ) > MOMENT_SUM_CLOSURE_GUARD:
        return "stage88_moment_sum_closure_blocker"
    recovery = float(metrics["minimum_total_boundary_jump_recovery_fraction"])
    if recovery >= BOUNDARY_JUMP_RECOVERY_PRIMARY_GUARD:
        return "stage88_full_moment_boundary_effect_large_stage89_boundary_reconstruction_admissibility_audit"
    if recovery >= BOUNDARY_JUMP_RECOVERY_PARTIAL_GUARD:
        return "stage88_full_moment_boundary_effect_partial_stage89_momentwise_cancellation_audit"
    return "stage88_full_moment_boundary_effect_weak_stage89_dominant_subspace_reconciliation_audit"


def run_stage88(
    stage67_artifact_dir: str | Path,
    stage87_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage88_design(**design)
    retained67 = _validate_stage67(stage67_artifact_dir)
    retained87 = _validate_stage87(stage87_artifact_dir)

    with np.load(Path(stage67_artifact_dir) / "converged_full_distributions.npz") as data:
        phi = np.asarray(data["phi"], dtype=np.float64)
        psi = np.asarray(data["psi"], dtype=np.float64)
        vx = np.asarray(data["vx"], dtype=np.float64)
        vy = np.asarray(data["vy"], dtype=np.float64)
        weight = np.asarray(data["weight"], dtype=np.float64)

    retained_maps = stage76.evaluate_frame_jump_maps(phi, psi, vx, vy, weight, CHUNK_SIZE)
    retained_baseline_face = np.asarray(retained_maps["midpoint_common_frame_face_components"], dtype=np.float64)
    baseline_face, counterfactual_face = full_moment_face_maps(phi, psi, vx, vy, weight)
    metrics = full_moment_metrics(baseline_face, counterfactual_face, retained_baseline_face)
    decision = stage88_decision(metrics)

    baseline_cell = component_divergence(baseline_face)
    counterfactual_cell = component_divergence(counterfactual_face)
    face_change = counterfactual_face - baseline_face
    cell_change = counterfactual_cell - baseline_cell
    baseline_total_face = np.sum(baseline_face, axis=0)
    counterfactual_total_face = np.sum(counterfactual_face, axis=0)
    baseline_total_cell = divergence_from_interior_faces(baseline_total_face)
    counterfactual_total_cell = divergence_from_interior_faces(counterfactual_total_face)

    if decision.endswith("boundary_reconstruction_admissibility_audit"):
        next_scope = (
            "Before any solver experiment, audit the frozen one-sided wall-adjacent reconstructed phi/psi states over all velocity nodes "
            "for nonnegativity, neighbor-bound overshoot, and moment consistency; a large frozen response alone does not justify adopting the stencil."
        )
    elif decision.endswith("momentwise_cancellation_audit"):
        next_scope = (
            "Decompose the partial full-moment response into signed cross-moment cancellation and side-specific contributions while retaining the same frozen endpoint; do not rerun the solver."
        )
    else:
        next_scope = (
            "Reconcile why the dominant Stage-87 velocity subspace showed a large response while the all-node/all-moment response is weak, using only frozen velocity-space attribution."
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "full_moment_boundary_counterfactual_maps.npz",
        moment_names=np.asarray(MOMENT_NAMES),
        baseline_face_components=baseline_face,
        counterfactual_face_components=counterfactual_face,
        face_change_components=face_change,
        baseline_cell_divergence_components=baseline_cell,
        counterfactual_cell_divergence_components=counterfactual_cell,
        cell_divergence_change_components=cell_change,
        baseline_total_face=baseline_total_face,
        counterfactual_total_face=counterfactual_total_face,
        baseline_total_cell_divergence=baseline_total_cell,
        counterfactual_total_cell_divergence=counterfactual_total_cell,
        retained_stage76_baseline_face_components=retained_baseline_face,
    )

    summary = {
        "stage": 88,
        "description": "Frozen all-velocity/all-moment propagation of the Stage-87 one-sided boundary-slope counterfactual.",
        "finite": bool(metrics["finite"]),
        "configuration": {
            "grid": list(GRID),
            "kn0": KNUDSEN,
            "cold_hot_ratio": COLD_HOT_RATIO,
            "radial_nodes": RULE[0],
            "angular_nodes": RULE[1],
            "point_count": POINT_COUNT,
            "radial_scale": RADIAL_SCALE,
            "chunk_size": CHUNK_SIZE,
            "limiter": LIMITER,
            "moment_names": list(MOMENT_NAMES),
            "counterfactual_boundary_slope": "one_sided_first_difference",
            "counterfactual_face_scope": "wall_adjacent_interior_x_faces_only_all_velocity_nodes",
            "baseline_closure_guard": BASELINE_CLOSURE_GUARD,
            "conservation_guard": CONSERVATION_GUARD,
            "moment_sum_closure_guard": MOMENT_SUM_CLOSURE_GUARD,
            "boundary_jump_recovery_primary_guard": BOUNDARY_JUMP_RECOVERY_PRIMARY_GUARD,
            "boundary_jump_recovery_partial_guard": BOUNDARY_JUMP_RECOVERY_PARTIAL_GUARD,
            "solver_rerun_count": 0,
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "correction_floor_retuning": False,
            "source_relaxation_retuning": False,
            "transport_parameter_retuning": False,
            "wall_model_retuning": False,
            "normalization_retuning": False,
            "velocity_quadrature_retuning": False,
            "failed_muscl_endpoint_rehabilitated": False,
            "cross_knudsen_extension_permitted": False,
            "validation_claim_permitted": False,
        },
        "retained_stage67_decision": retained67["decision"],
        "retained_stage87_decision": retained87["decision"],
        "full_moment_counterfactual_metrics": metrics,
        "decision": decision,
        "positive_findings": [
            "All 3840 fixed velocity nodes and all three common-frame heat-flux moment components are retained; no Stage-87 velocity subspace is privileged in the Stage-88 total.",
            "The independently accumulated Stage-88 baseline is closed against the exact Stage-76 common-frame component construction before applying the one-sided boundary-slope diagnostic."
        ],
        "negative_findings": [
            "This remains an offline frozen-state counterfactual, not a stable solver endpoint, not an error correction, and not evidence that q_av, convergence, or physical accuracy would improve after a rerun.",
            "A large all-moment response would still not establish that the one-sided reconstruction is admissible or should be adopted; admissibility must be audited separately before any solver experiment.",
            "The failed Stage-28 MUSCL endpoint remains unrecovered, cross-Knudsen extension is prohibited, and no validation claim is permitted.",
            "No physical, collision, correction-floor, source-relaxation, transport, wall, normalization, or velocity-quadrature parameter is retuned, and no cavity solver is rerun."
        ],
        "scientifically_justified_next_scope": next_scope,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage87-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage88(args.stage67_artifact_dir, args.stage87_artifact_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
