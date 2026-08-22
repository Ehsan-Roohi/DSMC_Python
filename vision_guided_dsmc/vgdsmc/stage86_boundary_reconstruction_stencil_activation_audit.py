from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import stage79_dominant_moment_radial_angular_gradient_audit as stage79
from . import stage82_within_sector_angular_coherence_audit as stage82

STAGE85_ENDPOINT = {
    "summary_sha256": "eadee947344ee83d370675fe260da570b301e09d03553f80d50a3cbc518bba11",
    "maps_sha256": "db179b1d1644f33f747ece8e5eb743f6f451474cd45490eee5a6fc389d1bb481",
    "decision": "stage85_abrupt_near_wall_face_suppression_stage86_boundary_reconstruction_stencil_activation_audit",
}
GRID = (64, 64)
KNUDSEN = 10.0
COLD_HOT_RATIO = 0.1
RADIAL_NODES = 40
ANGULAR_NODES = 96
RADIAL_SCALE = 2.0
LIMITER = "minmod"
VERTICAL_OBLIQUE_BINS = (1, 2, 5, 6)
OPPOSITE_SECTOR_PAIRS = ((1, 5), (2, 6))
PAIR_CLOSURE_GUARD = 1.0e-12
BOUNDARY_UPWIND_WALL_ABS_GUARD = 1.0e-14
BOUNDARY_UPWIND_FIRST_SHARE_GUARD = 0.95
REMAINING_WALL_FIRST_RATIO_MIN = 0.50
VX_SIGN_COS_TOL = 1.0e-12


def validate_stage86_design(**overrides: object) -> None:
    frozen = {
        "grid": GRID,
        "kn0": KNUDSEN,
        "cold_hot_ratio": COLD_HOT_RATIO,
        "radial_nodes": RADIAL_NODES,
        "angular_nodes": ANGULAR_NODES,
        "radial_scale": RADIAL_SCALE,
        "limiter": LIMITER,
        "vertical_oblique_bins": VERTICAL_OBLIQUE_BINS,
        "opposite_sector_pairs": OPPOSITE_SECTOR_PAIRS,
        "pair_closure_guard": PAIR_CLOSURE_GUARD,
        "boundary_upwind_wall_abs_guard": BOUNDARY_UPWIND_WALL_ABS_GUARD,
        "boundary_upwind_first_share_guard": BOUNDARY_UPWIND_FIRST_SHARE_GUARD,
        "remaining_wall_first_ratio_min": REMAINING_WALL_FIRST_RATIO_MIN,
        "vx_sign_cos_tol": VX_SIGN_COS_TOL,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError("Stage 86 is frozen to completed Stages 81/85 and fixed diagnostic guards; no retuning is permitted")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _validate_stage85_artifact(root: str | Path) -> dict[str, object]:
    root = Path(root)
    summary_path = root / "summary.json"
    maps_path = root / "near_wall_face_amplitude_suppression_maps.npz"
    if not summary_path.is_file() or not maps_path.is_file():
        raise FileNotFoundError("Stage 86 requires the exact completed Stage-85 summary and maps")
    if _sha256(summary_path) != STAGE85_ENDPOINT["summary_sha256"] or _sha256(maps_path) != STAGE85_ENDPOINT["maps_sha256"]:
        raise ValueError("Stage-85 checksum mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("stage") != 85 or summary.get("decision") != STAGE85_ENDPOINT["decision"] or summary.get("finite") is not True:
        raise ValueError("Stage-85 completed endpoint mismatch")
    cfg = summary["configuration"]
    expected = {
        "grid": [64, 64], "kn0": 10.0, "radial_nodes": 40, "angular_nodes": 96,
        "point_count": 3840, "radial_scale": 2.0, "limiter": "minmod",
        "dominant_moment": "transverse_kinetic", "dominant_radial_shell": 2,
        "dominant_local_radial_node": 1, "dominant_global_radial_node": 21,
        "vertical_oblique_bins": [1, 2, 5, 6], "opposite_sector_pairs": [[1, 5], [2, 6]],
        "solver_rerun_count": 0, "failed_muscl_endpoint_rehabilitated": False,
        "cross_knudsen_extension_permitted": False, "validation_claim_permitted": False,
    }
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError("Stage-85 frozen configuration mismatch")
    if any(v is not False for k, v in cfg.items() if k.endswith("_retuning")):
        raise ValueError("Stage-85 endpoint contains prohibited retuning")
    return summary


def reconstruction_implementation_invariants() -> dict[str, bool]:
    field = np.arange(40.0).reshape(2, 5, 4)
    slopes = stage79.limited_slopes_x(field)
    vx = np.asarray([1.0, -1.0, 0.5, -0.5])
    delta = stage79.interior_x_face_flux_difference_chunk(field, vx)
    return {
        "boundary_cell_slopes_exactly_zero": bool(np.all(slopes[:, 0] == 0.0) and np.all(slopes[:, -1] == 0.0)),
        "left_wall_adjacent_positive_vx_correction_exactly_zero": bool(np.all(delta[:, 0, vx > 0.0] == 0.0)),
        "right_wall_adjacent_negative_vx_correction_exactly_zero": bool(np.all(delta[:, -1, vx < 0.0] == 0.0)),
        "interior_slope_activity_present": bool(np.any(slopes[:, 1:-1] != 0.0)),
    }


def _norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64).ravel()))


def boundary_stencil_metrics(
    face: np.ndarray,
    bins: np.ndarray,
    angles_degrees: np.ndarray,
    retained_pairs: np.ndarray,
    pairs: np.ndarray,
) -> dict[str, object]:
    face = np.asarray(face, dtype=np.float64)
    bins = np.asarray(bins, dtype=np.int16)
    angles = np.asarray(angles_degrees, dtype=np.float64)
    retained = np.asarray(retained_pairs, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int16)
    if face.shape != (96, 64, 63) or bins.shape != (96,) or angles.shape != (96,):
        raise ValueError("Stage 86 requires exact Stage-81 96-ordinate face maps")
    if retained.shape != (2, 64, 63) or not np.array_equal(pairs, [[1, 5], [2, 6]]):
        raise ValueError("Stage 86 requires exact Stage-85 opposite-sector pair maps")
    if not (np.all(np.isfinite(face)) and np.all(np.isfinite(angles)) and np.all(np.isfinite(retained))):
        raise ValueError("Stage 86 requires finite maps")

    cos_angle = np.cos(np.deg2rad(angles))
    positive = cos_angle > VX_SIGN_COS_TOL
    negative = cos_angle < -VX_SIGN_COS_TOL
    near_zero = ~(positive | negative)
    closure: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    boundary_vectors = np.zeros((2, 2, 2, 64))
    remaining_vectors = np.zeros_like(boundary_vectors)

    for pidx, pair in enumerate(pairs.tolist()):
        selected = np.isin(bins, pair)
        reconstructed = np.sum(face[selected], axis=0)
        delta = reconstructed - retained[pidx]
        closure.append({
            "angular_bin_pair": pair,
            "maximum_absolute_error": float(np.max(np.abs(delta))),
            "relative_l2_error": _norm(delta) / max(_norm(retained[pidx]), 1.0e-300),
        })
        for sidx, (side, wi, fi, boundary_sign) in enumerate((("left", 0, 1, positive), ("right", -1, -2, negative))):
            boundary = selected & boundary_sign
            remaining = selected & (~boundary_sign) & (~near_zero)
            zero = selected & near_zero
            bw_ind = face[boundary, :, wi]
            bw = np.sum(bw_ind, axis=0)
            bf = np.sum(face[boundary, :, fi], axis=0)
            rw = np.sum(face[remaining, :, wi], axis=0)
            rf = np.sum(face[remaining, :, fi], axis=0)
            zw = np.sum(face[zero, :, wi], axis=0)
            zf = np.sum(face[zero, :, fi], axis=0)
            total_w = np.sum(face[selected, :, wi], axis=0)
            total_f = np.sum(face[selected, :, fi], axis=0)
            boundary_vectors[pidx, sidx] = np.stack([bw, bf])
            remaining_vectors[pidx, sidx] = np.stack([rw, rf])
            bf_l2, rf_l2, zf_l2 = _norm(bf), _norm(rf), _norm(zf)
            rows.append({
                "angular_bin_pair": pair, "side": side,
                "boundary_upwind_ordinate_count": int(np.sum(boundary)),
                "remaining_nonzero_vx_ordinate_count": int(np.sum(remaining)),
                "near_zero_vx_ordinate_count": int(np.sum(zero)),
                "boundary_upwind_wall_individual_max_abs": 0.0 if bw_ind.size == 0 else float(np.max(np.abs(bw_ind))),
                "boundary_upwind_wall_group_l2": _norm(bw),
                "boundary_upwind_first_interior_group_l2": bf_l2,
                "remaining_wall_group_l2": _norm(rw),
                "remaining_first_interior_group_l2": rf_l2,
                "near_zero_vx_wall_group_l2": _norm(zw),
                "near_zero_vx_first_interior_group_l2": zf_l2,
                "boundary_upwind_first_component_norm_share": bf_l2 / max(bf_l2 + rf_l2 + zf_l2, 1.0e-300),
                "remaining_direction_wall_to_first_l2_ratio": _norm(rw) / max(rf_l2, 1.0e-300),
                "total_wall_group_l2": _norm(total_w),
                "total_first_interior_group_l2": _norm(total_f),
                "total_wall_to_first_l2_ratio": _norm(total_w) / max(_norm(total_f), 1.0e-300),
            })

    return {
        "pair_reconstruction_closure": closure,
        "rows": rows,
        "maximum_pair_reconstruction_relative_l2_error": max(float(r["relative_l2_error"]) for r in closure),
        "maximum_boundary_upwind_wall_individual_absolute_value": max(float(r["boundary_upwind_wall_individual_max_abs"]) for r in rows),
        "minimum_boundary_upwind_first_component_norm_share": min(float(r["boundary_upwind_first_component_norm_share"]) for r in rows),
        "minimum_remaining_direction_wall_to_first_l2_ratio": min(float(r["remaining_direction_wall_to_first_l2_ratio"]) for r in rows),
        "positive_vx_ordinate_count": int(np.sum(positive)), "negative_vx_ordinate_count": int(np.sum(negative)),
        "near_zero_vx_ordinate_count": int(np.sum(near_zero)), "finite": True,
        "boundary_component_vectors": boundary_vectors, "remaining_component_vectors": remaining_vectors,
    }


def stage86_decision(finite: bool, invariants: bool, closure: float, wall_zero: float, first_share: float, remaining_ratio: float) -> str:
    if not finite:
        return "stage86_nonfinite_boundary_stencil_audit_blocker"
    if not invariants or closure > PAIR_CLOSURE_GUARD:
        return "stage86_reconstruction_or_pair_closure_blocker"
    if wall_zero <= BOUNDARY_UPWIND_WALL_ABS_GUARD and first_share >= BOUNDARY_UPWIND_FIRST_SHARE_GUARD and remaining_ratio >= REMAINING_WALL_FIRST_RATIO_MIN:
        return "stage86_boundary_zero_slope_stencil_primary_frozen_suppression_stage87_one_sided_boundary_slope_counterfactual_audit"
    if wall_zero <= BOUNDARY_UPWIND_WALL_ABS_GUARD:
        return "stage86_boundary_zero_slope_stencil_contributes_but_not_dominant_stage87_remaining_direction_localization_audit"
    return "stage86_no_structural_boundary_zero_explanation_stage87_frozen_distribution_transition_audit"


def run_stage86(stage81_artifact_dir: str | Path, stage85_artifact_dir: str | Path, output_dir: str | Path, **design: object) -> dict[str, object]:
    validate_stage86_design(**design)
    retained81 = stage82._validate_stage81_artifact(stage81_artifact_dir)
    retained85 = _validate_stage85_artifact(stage85_artifact_dir)
    with np.load(Path(stage81_artifact_dir) / "dominant_node_individual_ordinate_attribution_maps.npz") as z:
        face, bins = np.asarray(z["face_groups"], float), np.asarray(z["ordinate_to_angular_bin"], np.int16)
    angles = np.asarray(retained81["ordinate_metrics"]["ordinate_angles_degrees"], float)
    with np.load(Path(stage85_artifact_dir) / "near_wall_face_amplitude_suppression_maps.npz") as z:
        retained_face = np.asarray(z["pair_face_groups"], float)
        retained_cell = np.asarray(z["pair_cell_divergence_groups"], float)
        pairs = np.asarray(z["opposite_sector_pairs"], np.int16)
    invariants = reconstruction_implementation_invariants()
    metrics = boundary_stencil_metrics(face, bins, angles, retained_face, pairs)
    decision = stage86_decision(
        bool(metrics["finite"]), all(invariants.values()), float(metrics["maximum_pair_reconstruction_relative_l2_error"]),
        float(metrics["maximum_boundary_upwind_wall_individual_absolute_value"]),
        float(metrics["minimum_boundary_upwind_first_component_norm_share"]),
        float(metrics["minimum_remaining_direction_wall_to_first_l2_ratio"]),
    )
    if decision.endswith("one_sided_boundary_slope_counterfactual_audit"):
        next_scope = "Use the exact Stage-67 frozen distributions to construct an offline one-sided boundary-slope counterfactual only at wall-adjacent interior x faces; compare frozen face/cell residual changes without advancing or retuning the solver and without claiming q_av improvement or validation."
    elif decision.endswith("remaining_direction_localization_audit"):
        next_scope = "Localize the surviving non-boundary-upwind contribution in y and angle before any counterfactual; no solver rerun or retuning is authorized."
    else:
        next_scope = "Audit the frozen distribution transition before any reconstruction counterfactual; no solver rerun or retuning is authorized."
    public = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
    cfg = {
        "grid": [64, 64], "kn0": 10.0, "cold_hot_ratio": 0.1, "radial_nodes": 40, "angular_nodes": 96,
        "point_count": 3840, "radial_scale": 2.0, "limiter": "minmod", "dominant_moment": "transverse_kinetic",
        "dominant_radial_shell": 2, "dominant_local_radial_node": 1, "dominant_global_radial_node": 21,
        "vertical_oblique_bins": [1, 2, 5, 6], "opposite_sector_pairs": [[1, 5], [2, 6]],
        "pair_closure_guard": PAIR_CLOSURE_GUARD, "boundary_upwind_wall_abs_guard": BOUNDARY_UPWIND_WALL_ABS_GUARD,
        "boundary_upwind_first_share_guard": BOUNDARY_UPWIND_FIRST_SHARE_GUARD,
        "remaining_wall_first_ratio_min": REMAINING_WALL_FIRST_RATIO_MIN, "vx_sign_cos_tol": VX_SIGN_COS_TOL,
        "solver_rerun_count": 0, "physical_parameter_retuning": False, "collision_parameter_retuning": False,
        "correction_floor_retuning": False, "source_relaxation_retuning": False, "transport_parameter_retuning": False,
        "wall_model_retuning": False, "normalization_retuning": False, "velocity_quadrature_retuning": False,
        "failed_muscl_endpoint_rehabilitated": False, "cross_knudsen_extension_permitted": False, "validation_claim_permitted": False,
    }
    summary = {
        "stage": 86, "description": "Frozen boundary-reconstruction stencil-activation audit of Stage-85 near-wall suppression.",
        "finite": bool(metrics["finite"] and all(invariants.values())), "configuration": cfg,
        "retained_stage81_decision": retained81["decision"], "retained_stage85_decision": retained85["decision"],
        "reconstruction_implementation_invariants": invariants, "boundary_stencil_metrics": public, "decision": decision,
        "positive_findings": [
            "The exact Stage-81 individual-ordinate maps are recombined into the exact Stage-85 opposite-sector pair maps before stencil attribution.",
            "Boundary-upwind and remaining directions are separated without generating a new cavity state, while the frozen minmod reconstruction is checked for its boundary-slope invariant.",
        ],
        "negative_findings": [
            "A structural zero-boundary-slope contribution can explain a frozen correction pattern without proving that the boundary treatment is numerically wrong or that a changed solver would improve q_av.",
            "Non-boundary-upwind directions are retained explicitly; surviving wall-adjacent contributions are not discarded.",
            "The failed Stage-28 MUSCL endpoint remains unrecovered and is not rehabilitated or extended across Knudsen number.",
            "No physical, collision, correction-floor, source-relaxation, transport, wall, normalization, or velocity-quadrature parameter is retuned, and no cavity solver is rerun.",
        ],
        "scientifically_justified_next_scope": next_scope,
    }
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "boundary_reconstruction_stencil_activation_maps.npz", opposite_sector_pairs=pairs,
        retained_pair_face_groups=retained_face, retained_pair_cell_divergence_groups=retained_cell,
        ordinate_to_angular_bin=bins, ordinate_angles_degrees=angles,
        boundary_component_vectors=metrics["boundary_component_vectors"], remaining_component_vectors=metrics["remaining_component_vectors"])
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage81-artifact-dir", required=True); parser.add_argument("--stage85-artifact-dir", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); print(json.dumps(run_stage86(args.stage81_artifact_dir, args.stage85_artifact_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
