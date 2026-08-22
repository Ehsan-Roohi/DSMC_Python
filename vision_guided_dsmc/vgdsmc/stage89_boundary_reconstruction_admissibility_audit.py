from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import stage76_local_velocity_frame_jump_audit as stage76
from . import stage88_full_moment_boundary_counterfactual_audit as stage88

STAGE88_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31260169074,
    "workflow_job_id": 93109642208,
    "workflow_conclusion": "success",
    "tests_passed": 310,
    "tests_failed": 0,
    "artifact_id": 9025958614,
    "artifact_size_bytes": 442641,
    "artifact_sha256": "f1a33a9a5bbd6d82672e715c4ee2d89672e8d27a5b6bd6e415535646d7ed54b2",
    "source_head_sha": "72bf29dd2c86dd8e61c591e8893c3618886ecb84",
    "summary_sha256": "83eda4cca1cdda5112bab21dbb57a61bed8ff1ec5e8dbb28b08e5f3754c2ea76",
    "maps_sha256": "05d722d15aac6ffa6b43649297bb3b504dc6bded60b58e3d1d8f44fd3c92415b",
    "decision": "stage88_full_moment_boundary_effect_large_stage89_boundary_reconstruction_admissibility_audit",
}

GRID = (64, 64)
KNUDSEN = 10.0
COLD_HOT_RATIO = 0.1
RULE = (40, 96)
POINT_COUNT = RULE[0] * RULE[1]
RADIAL_SCALE = 2.0
LIMITER = "minmod"
COUNTERFACTUAL_BOUNDARY_SLOPE = "one_sided_first_difference"
NEGATIVITY_REL_GUARD = 1.0e-12
NEIGHBOR_OVERSHOOT_REL_GUARD = 1.0e-12
MIDPOINT_CLOSURE_GUARD = 1.0e-12
MOMENT_LINEARITY_GUARD = 1.0e-12
UNMODIFIED_SCOPE_GUARD = 0.0
MOMENT_NAMES = (
    "density_like",
    "x_momentum_like",
    "y_momentum_like",
    "translational_energy_like",
    "reduced_internal_energy_like",
)


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_stage89_design(**overrides: object) -> None:
    frozen = {
        "grid": GRID,
        "kn0": KNUDSEN,
        "cold_hot_ratio": COLD_HOT_RATIO,
        "rule": RULE,
        "radial_scale": RADIAL_SCALE,
        "limiter": LIMITER,
        "counterfactual_boundary_slope": COUNTERFACTUAL_BOUNDARY_SLOPE,
        "negativity_rel_guard": NEGATIVITY_REL_GUARD,
        "neighbor_overshoot_rel_guard": NEIGHBOR_OVERSHOOT_REL_GUARD,
        "midpoint_closure_guard": MIDPOINT_CLOSURE_GUARD,
        "moment_linearity_guard": MOMENT_LINEARITY_GUARD,
        "unmodified_scope_guard": UNMODIFIED_SCOPE_GUARD,
        "moment_names": MOMENT_NAMES,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 89 is frozen to the exact completed Stage-67 distributions and Stage-88 endpoint, the 40x96 velocity rule, "
            "the one-sided boundary-slope counterfactual, and preregistered local admissibility guards; no solver rerun or parameter retuning is permitted."
        )


def _validate_stage67(root: str | Path) -> dict[str, object]:
    return stage88._validate_stage67(root)


def _validate_stage88(root: str | Path) -> dict[str, object]:
    root = Path(root)
    files = {
        "summary.json": "summary_sha256",
        "full_moment_boundary_counterfactual_maps.npz": "maps_sha256",
    }
    for name, key in files.items():
        path = root / name
        if not path.is_file() or _sha256(path) != str(STAGE88_COMPLETED_ENDPOINT[key]):
            raise ValueError(f"Stage-88 checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 88 or summary.get("decision") != STAGE88_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage-88 completed endpoint mismatch")
    return summary


def wall_adjacent_face_states(
    distribution: np.ndarray,
    vx: np.ndarray,
    side: str,
) -> dict[str, np.ndarray]:
    distribution = np.asarray(distribution, dtype=np.float64)
    vx = np.asarray(vx, dtype=np.float64)
    if distribution.ndim != 3 or distribution.shape[1] < 3 or distribution.shape[2] != vx.size:
        raise ValueError("Stage 89 requires (ny,nx,nv) distributions with nx>=3 and matching vx")

    retained_slope = stage76.limited_slopes_x(distribution)
    counterfactual_slope = stage88.one_sided_boundary_slopes_x(distribution)
    positive = vx > 0.0
    negative = vx < 0.0
    used = positive | negative

    baseline = np.zeros((distribution.shape[0], vx.size), dtype=np.float64)
    counterfactual = np.zeros_like(baseline)
    if side == "left":
        boundary = distribution[:, 0]
        neighbor = distribution[:, 1]
        baseline[:, positive] = boundary[:, positive] + 0.5 * retained_slope[:, 0, positive]
        counterfactual[:, positive] = boundary[:, positive] + 0.5 * counterfactual_slope[:, 0, positive]
        baseline[:, negative] = neighbor[:, negative] - 0.5 * retained_slope[:, 1, negative]
        counterfactual[:, negative] = neighbor[:, negative] - 0.5 * counterfactual_slope[:, 1, negative]
        modified = positive
    elif side == "right":
        boundary = distribution[:, -1]
        neighbor = distribution[:, -2]
        baseline[:, positive] = neighbor[:, positive] + 0.5 * retained_slope[:, -2, positive]
        counterfactual[:, positive] = neighbor[:, positive] + 0.5 * counterfactual_slope[:, -2, positive]
        baseline[:, negative] = boundary[:, negative] - 0.5 * retained_slope[:, -1, negative]
        counterfactual[:, negative] = boundary[:, negative] - 0.5 * counterfactual_slope[:, -1, negative]
        modified = negative
    else:
        raise ValueError("side must be left or right")

    zero = ~used
    if np.any(zero):
        midpoint = 0.5 * (boundary[:, zero] + neighbor[:, zero])
        baseline[:, zero] = midpoint
        counterfactual[:, zero] = midpoint

    return {
        "baseline": baseline,
        "counterfactual": counterfactual,
        "boundary": boundary,
        "neighbor": neighbor,
        "used_mask": used,
        "modified_mask": modified,
        "zero_mask": zero,
    }


def _norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64).ravel()))


def _state_admissibility_row(field_name: str, side: str, state: dict[str, np.ndarray]) -> dict[str, object]:
    baseline = state["baseline"]
    counterfactual = state["counterfactual"]
    boundary = state["boundary"]
    neighbor = state["neighbor"]
    used = np.asarray(state["used_mask"], dtype=bool)
    modified = np.asarray(state["modified_mask"], dtype=bool)
    unmodified = used & ~modified

    endpoint_scale = max(
        float(np.max(np.abs(boundary[:, used]))),
        float(np.max(np.abs(neighbor[:, used]))),
        1.0e-300,
    )
    lower = np.minimum(boundary, neighbor)
    upper = np.maximum(boundary, neighbor)
    candidate_used = counterfactual[:, used]
    lower_used = lower[:, used]
    upper_used = upper[:, used]
    overshoot = np.maximum(np.maximum(lower_used - candidate_used, candidate_used - upper_used), 0.0)
    candidate_negativity = max(0.0, -float(np.min(candidate_used))) / endpoint_scale
    bound_overshoot = float(np.max(overshoot)) / endpoint_scale

    midpoint = 0.5 * (boundary[:, modified] + neighbor[:, modified])
    midpoint_error = _norm(counterfactual[:, modified] - midpoint) / max(_norm(midpoint), 1.0e-300)
    unmodified_change = (
        float(np.max(np.abs(counterfactual[:, unmodified] - baseline[:, unmodified]))) if np.any(unmodified) else 0.0
    )
    threshold = NEGATIVITY_REL_GUARD * endpoint_scale
    inherited_negative_count = int(
        np.count_nonzero(boundary[:, used] < -threshold) + np.count_nonzero(neighbor[:, used] < -threshold)
    )
    candidate_negative_count = int(np.count_nonzero(candidate_used < -threshold))
    new_negative_count = int(
        np.count_nonzero(
            (candidate_used < -threshold)
            & (lower_used >= -threshold)
        )
    )

    return {
        "field": field_name,
        "side": side,
        "used_velocity_nodes": int(np.count_nonzero(used)),
        "modified_velocity_nodes": int(np.count_nonzero(modified)),
        "zero_velocity_nodes": int(np.count_nonzero(~used)),
        "endpoint_scale": endpoint_scale,
        "minimum_boundary_value": float(np.min(boundary[:, used])),
        "minimum_neighbor_value": float(np.min(neighbor[:, used])),
        "minimum_counterfactual_face_value": float(np.min(candidate_used)),
        "candidate_negativity_normalized": candidate_negativity,
        "neighbor_bound_overshoot_normalized": bound_overshoot,
        "modified_halfspace_midpoint_relative_l2_error": midpoint_error,
        "maximum_change_on_unmodified_halfspace": unmodified_change,
        "inherited_endpoint_negative_count": inherited_negative_count,
        "candidate_negative_count": candidate_negative_count,
        "new_negative_count": new_negative_count,
        "counterfactual_vs_baseline_relative_l2": _norm(counterfactual[:, used] - baseline[:, used])
        / max(_norm(baseline[:, used]), 1.0e-300),
    }


def reduced_moment_array(
    phi_state: np.ndarray,
    psi_state: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    phi = np.asarray(phi_state, dtype=np.float64)[:, mask]
    psi = np.asarray(psi_state, dtype=np.float64)[:, mask]
    vx_m = np.asarray(vx, dtype=np.float64)[mask]
    vy_m = np.asarray(vy, dtype=np.float64)[mask]
    w = np.asarray(weight, dtype=np.float64)[mask]
    return np.stack(
        [
            np.sum(phi * w[None, :], axis=1),
            np.sum(phi * (vx_m * w)[None, :], axis=1),
            np.sum(phi * (vy_m * w)[None, :], axis=1),
            0.5 * np.sum(phi * ((vx_m * vx_m + vy_m * vy_m) * w)[None, :], axis=1),
            0.5 * np.sum(psi * w[None, :], axis=1),
        ],
        axis=1,
    )


def side_moment_metrics(
    side: str,
    phi_state: dict[str, np.ndarray],
    psi_state: dict[str, np.ndarray],
    vx: np.ndarray,
    vy: np.ndarray,
    weight: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    used = np.asarray(phi_state["used_mask"], dtype=bool)
    modified = np.asarray(phi_state["modified_mask"], dtype=bool)
    baseline_full = reduced_moment_array(phi_state["baseline"], psi_state["baseline"], vx, vy, weight, used)
    counterfactual_full = reduced_moment_array(
        phi_state["counterfactual"], psi_state["counterfactual"], vx, vy, weight, used
    )
    candidate_active = reduced_moment_array(
        phi_state["counterfactual"], psi_state["counterfactual"], vx, vy, weight, modified
    )
    boundary_active = reduced_moment_array(
        phi_state["boundary"], psi_state["boundary"], vx, vy, weight, modified
    )
    neighbor_active = reduced_moment_array(
        phi_state["neighbor"], psi_state["neighbor"], vx, vy, weight, modified
    )
    reference_active = 0.5 * (boundary_active + neighbor_active)
    linearity = _norm(candidate_active - reference_active) / max(_norm(reference_active), 1.0e-300)
    shift = _norm(counterfactual_full - baseline_full) / max(_norm(baseline_full), 1.0e-300)
    return (
        {
            "side": side,
            "modified_halfspace_moment_linearity_relative_l2_error": linearity,
            "full_face_reduced_moment_shift_relative_l2": shift,
            "maximum_absolute_full_face_moment_change": float(np.max(np.abs(counterfactual_full - baseline_full))),
        },
        {
            "baseline_full": baseline_full,
            "counterfactual_full": counterfactual_full,
            "candidate_active": candidate_active,
            "reference_active": reference_active,
        },
    )


def stage89_metrics(
    phi: np.ndarray,
    psi: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    weight: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    state_rows: list[dict[str, object]] = []
    moment_rows: list[dict[str, object]] = []
    profiles: dict[str, np.ndarray] = {}
    all_finite = bool(
        np.all(np.isfinite(phi))
        and np.all(np.isfinite(psi))
        and np.all(np.isfinite(vx))
        and np.all(np.isfinite(vy))
        and np.all(np.isfinite(weight))
    )

    for side in ("left", "right"):
        phi_state = wall_adjacent_face_states(phi, vx, side)
        psi_state = wall_adjacent_face_states(psi, vx, side)
        state_rows.append(_state_admissibility_row("phi", side, phi_state))
        state_rows.append(_state_admissibility_row("psi", side, psi_state))
        moment_row, side_profiles = side_moment_metrics(side, phi_state, psi_state, vx, vy, weight)
        moment_rows.append(moment_row)
        for name, value in side_profiles.items():
            profiles[f"{side}_{name}_moments"] = value
        profiles[f"{side}_phi_counterfactual_min_by_row"] = np.min(
            phi_state["counterfactual"][:, phi_state["used_mask"]], axis=1
        )
        profiles[f"{side}_psi_counterfactual_min_by_row"] = np.min(
            psi_state["counterfactual"][:, psi_state["used_mask"]], axis=1
        )

    all_finite = all_finite and all(
        np.isfinite(float(row[key]))
        for row in state_rows
        for key in (
            "endpoint_scale",
            "minimum_counterfactual_face_value",
            "candidate_negativity_normalized",
            "neighbor_bound_overshoot_normalized",
            "modified_halfspace_midpoint_relative_l2_error",
            "maximum_change_on_unmodified_halfspace",
        )
    )
    all_finite = all_finite and all(
        np.isfinite(float(row[key]))
        for row in moment_rows
        for key in (
            "modified_halfspace_moment_linearity_relative_l2_error",
            "full_face_reduced_moment_shift_relative_l2",
        )
    )

    metrics = {
        "finite": bool(all_finite),
        "maximum_candidate_negativity_normalized": max(float(row["candidate_negativity_normalized"]) for row in state_rows),
        "maximum_neighbor_bound_overshoot_normalized": max(float(row["neighbor_bound_overshoot_normalized"]) for row in state_rows),
        "maximum_modified_halfspace_midpoint_relative_l2_error": max(
            float(row["modified_halfspace_midpoint_relative_l2_error"]) for row in state_rows
        ),
        "maximum_change_on_unmodified_halfspace": max(float(row["maximum_change_on_unmodified_halfspace"]) for row in state_rows),
        "maximum_modified_halfspace_moment_linearity_relative_l2_error": max(
            float(row["modified_halfspace_moment_linearity_relative_l2_error"]) for row in moment_rows
        ),
        "maximum_full_face_reduced_moment_shift_relative_l2": max(
            float(row["full_face_reduced_moment_shift_relative_l2"]) for row in moment_rows
        ),
        "total_inherited_endpoint_negative_count": int(sum(int(row["inherited_endpoint_negative_count"]) for row in state_rows)),
        "total_candidate_negative_count": int(sum(int(row["candidate_negative_count"]) for row in state_rows)),
        "total_new_negative_count": int(sum(int(row["new_negative_count"]) for row in state_rows)),
        "state_rows": state_rows,
        "moment_rows": moment_rows,
    }
    profiles["moment_names"] = np.asarray(MOMENT_NAMES)
    return metrics, profiles


def stage89_decision(metrics: dict[str, object]) -> str:
    if not bool(metrics["finite"]):
        return "stage89_nonfinite_reconstruction_admissibility_blocker"
    if float(metrics["maximum_change_on_unmodified_halfspace"]) > UNMODIFIED_SCOPE_GUARD:
        return "stage89_counterfactual_halfspace_scope_leakage_blocker"
    if float(metrics["maximum_candidate_negativity_normalized"]) > NEGATIVITY_REL_GUARD:
        return "stage89_candidate_nonnegativity_blocker_stage90_negativity_origin_audit"
    if float(metrics["maximum_neighbor_bound_overshoot_normalized"]) > NEIGHBOR_OVERSHOOT_REL_GUARD:
        return "stage89_neighbor_bound_overshoot_blocker_stage90_overshoot_origin_audit"
    if float(metrics["maximum_modified_halfspace_midpoint_relative_l2_error"]) > MIDPOINT_CLOSURE_GUARD:
        return "stage89_midpoint_reconstruction_closure_blocker"
    if float(metrics["maximum_modified_halfspace_moment_linearity_relative_l2_error"]) > MOMENT_LINEARITY_GUARD:
        return "stage89_reduced_moment_linearity_closure_blocker"
    return "stage89_local_admissibility_closes_stage90_single_condition_reconstruction_solver_ab_audit"


def run_stage89(
    stage67_artifact_dir: str | Path,
    stage88_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage89_design(**design)
    retained67 = _validate_stage67(stage67_artifact_dir)
    retained88 = _validate_stage88(stage88_artifact_dir)

    with np.load(Path(stage67_artifact_dir) / "converged_full_distributions.npz") as data:
        phi = np.asarray(data["phi"], dtype=np.float64)
        psi = np.asarray(data["psi"], dtype=np.float64)
        vx = np.asarray(data["vx"], dtype=np.float64)
        vy = np.asarray(data["vy"], dtype=np.float64)
        weight = np.asarray(data["weight"], dtype=np.float64)

    if phi.shape != (GRID[0], GRID[1], POINT_COUNT) or psi.shape != phi.shape:
        raise ValueError("Stage 89 requires the exact 64x64x3840 Stage-67 phi/psi distributions")
    if vx.shape != (POINT_COUNT,) or vy.shape != vx.shape or weight.shape != vx.shape:
        raise ValueError("Stage 89 requires the exact 3840-point Stage-67 velocity rule")

    metrics, profiles = stage89_metrics(phi, psi, vx, vy, weight)
    decision = stage89_decision(metrics)

    if decision.endswith("single_condition_reconstruction_solver_ab_audit"):
        next_scope = (
            "At the single inherited Kn=10, cold/hot ratio 0.1 condition only, perform a preregistered solver A/B endpoint audit comparing the retained zero-boundary-slope reconstruction with only the locally admissible one-sided boundary slope. Keep initialization, collision/source treatment, relaxation, grid, velocity rule, wall model, tolerances and stopping rules identical; report convergence, finiteness and observables without tuning, cross-Knudsen extension or validation claims."
        )
    elif "nonnegativity_blocker" in decision:
        next_scope = (
            "Remain offline and locate whether the negative reconstructed values are inherited from the frozen Stage-67 endpoint or introduced by the face reconstruction, with velocity-node and wall-row attribution; do not rerun the solver."
        )
    elif "overshoot_blocker" in decision:
        next_scope = (
            "Remain offline and attribute neighbor-bound violations by field, sidewall, row and velocity node before considering any solver experiment; do not change limiter or reconstruction parameters."
        )
    else:
        next_scope = (
            "Treat the failed local closure/scope check as a genuine blocker and reconcile the exact frozen reconstruction algebra before any solver experiment."
        )

    if decision.endswith("single_condition_reconstruction_solver_ab_audit"):
        scientific_conclusion = (
            "Within the exact frozen Stage-67 state, the one-sided wall-adjacent face reconstruction satisfies the preregistered local nonnegativity, neighbor-bound, halfspace-scope, midpoint and reduced-moment-linearity guards. This establishes only local frozen-state admissibility; it does not establish nonlinear solver stability, wall-model correctness, q_av improvement, or physical validation."
        )
    else:
        scientific_conclusion = (
            "The frozen one-sided boundary reconstruction fails at least one preregistered local admissibility or closure guard. The failure is retained as a blocker and does not authorize a solver rerun or parameter adjustment."
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "boundary_reconstruction_admissibility_profiles.npz", **profiles)

    summary = {
        "stage": 89,
        "description": "Frozen local admissibility audit of the Stage-88 one-sided wall-adjacent boundary reconstruction over all fixed velocity nodes.",
        "finite": bool(metrics["finite"]),
        "configuration": {
            "grid": list(GRID),
            "kn0": KNUDSEN,
            "cold_hot_ratio": COLD_HOT_RATIO,
            "radial_nodes": RULE[0],
            "angular_nodes": RULE[1],
            "point_count": POINT_COUNT,
            "radial_scale": RADIAL_SCALE,
            "limiter": LIMITER,
            "counterfactual_boundary_slope": COUNTERFACTUAL_BOUNDARY_SLOPE,
            "negativity_rel_guard": NEGATIVITY_REL_GUARD,
            "neighbor_overshoot_rel_guard": NEIGHBOR_OVERSHOOT_REL_GUARD,
            "midpoint_closure_guard": MIDPOINT_CLOSURE_GUARD,
            "moment_linearity_guard": MOMENT_LINEARITY_GUARD,
            "unmodified_scope_guard": UNMODIFIED_SCOPE_GUARD,
            "moment_names": list(MOMENT_NAMES),
            "solver_rerun_count": 0,
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "correction_floor_retuning": False,
            "source_relaxation_retuning": False,
            "transport_parameter_retuning": False,
            "wall_model_retuning": False,
            "normalization_retuning": False,
            "velocity_quadrature_retuning": False,
            "limiter_retuning": False,
            "failed_muscl_endpoint_rehabilitated": False,
            "cross_knudsen_extension_permitted": False,
            "validation_claim_permitted": False,
        },
        "retained_stage67_decision": retained67.get("decision"),
        "retained_stage88_decision": retained88.get("decision"),
        "admissibility_metrics": metrics,
        "decision": decision,
        "scientific_conclusion": scientific_conclusion,
        "positive_findings": [
            "The audit evaluates phi and psi on both wall-adjacent interior x-faces using the exact 3840-node Stage-67 velocity rule; zero-vx nodes, if present, are identified separately because they carry no x-face transport.",
            "The modified boundary-upwind halfspace and the unchanged interior-upwind halfspace are audited separately, so a local admissibility result cannot hide scope leakage or inherited reconstruction behavior."
        ],
        "negative_findings": [
            "Local frozen-state nonnegativity and neighbor-boundedness are necessary diagnostics only; passing them does not establish nonlinear solver stability, wall-model correctness, or that the one-sided reconstruction should be adopted.",
            "Reduced-moment linearity closure is an algebraic/quadrature consistency check, not an adjoint sensitivity, a validation result, or evidence that q_av will improve.",
            "The failed Stage-28 MUSCL endpoint remains unrecovered; no failed parameter is retuned, no cross-Knudsen extension is permitted, and no validation claim is authorized.",
            "No cavity solver is rerun in Stage 89 and no physical, collision, correction-floor, source-relaxation, transport, wall, normalization, limiter, or velocity-quadrature parameter is changed."
        ],
        "scientifically_justified_next_scope": next_scope,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 89 frozen boundary-reconstruction admissibility audit")
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage88-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_stage89(args.stage67_artifact_dir, args.stage88_artifact_dir, args.output_dir)


if __name__ == "__main__":
    main()
