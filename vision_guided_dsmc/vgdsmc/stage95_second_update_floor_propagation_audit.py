from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from .stage34_velocity_scale_consistency import local_relaxation_time
from .stage41_projected_polar_operator_audit import (
    STAGE41_CORRECTION_FLOOR,
    mapped_polar_quadrature,
    projected_macroscopic,
    projected_shakhov_equilibrium,
)
from .stage42_projected_polar_heated_cavity_pilot import _upwind_neighbors, projected_wall_incoming
from .stage58_conservative_solver_64x64_confirmation import build_stage58_config
from .stage90_single_condition_reconstruction_solver_ab_audit import (
    COLD_HOT_RATIO,
    GRID,
    KNUDSEN,
    LIMITER,
    RADIAL_SCALE,
    RULE,
    SOURCE_RELAXATION,
    TOLERANCE,
    _validate_stage67,
    muscl_correction_divergence,
)
from .stage94_floor_moment_perturbation_audit import _field_difference


STAGE94_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31314965092,
    "workflow_job_id": 93248625477,
    "artifact_id": 9041472096,
    "artifact_sha256": "a92353337c0f84f6581e00258645af65d451758cccace5aa0ee9933170d4a5a0",
    "summary_sha256": "2f21e50651400f6c2ca7e9d3d0868e6773ee49c2e00fd6ca801378d405787ec3",
    "maps_sha256": "cfbfc8b6b9ffd01c270936f42d09d57c6481ae5232e50845e68d2ff2659530e6",
    "decision": "stage94_floor_moment_perturbation_negligible_stage95_unclipped_second_update_propagation_audit",
}
MATERIAL_PROPAGATION_GUARD = 1.0e-6
SECOND_UPDATE_INDEX = 2


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_stage95_design(**overrides: object) -> None:
    frozen = {
        "grid": GRID,
        "kn0": KNUDSEN,
        "cold_hot_ratio": COLD_HOT_RATIO,
        "rule": RULE,
        "radial_scale": RADIAL_SCALE,
        "limiter": LIMITER,
        "source_relaxation": SOURCE_RELAXATION,
        "tolerance": TOLERANCE,
        "correction_floor": STAGE41_CORRECTION_FLOOR,
        "second_update_index": SECOND_UPDATE_INDEX,
        "material_propagation_guard": MATERIAL_PROPAGATION_GUARD,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 95 is a frozen two-update propagation audit of the already observed Stage-94 "
            "positivity-floor perturbation. It may not retune physics, source relaxation, limiter, "
            "quadrature, wall model, positivity/correction floors, tolerance, update count, or the "
            "fixed materiality reporting guard."
        )


def _validate_stage94(root: str | Path) -> dict[str, object]:
    root = Path(root)
    expected = {
        "summary.json": STAGE94_COMPLETED_ENDPOINT["summary_sha256"],
        "floor_moment_perturbation_maps.npz": STAGE94_COMPLETED_ENDPOINT["maps_sha256"],
    }
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Stage-94 checksum mismatch: {name}")

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 94 or summary.get("decision") != STAGE94_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage-94 completed endpoint mismatch")
    cfg = summary.get("configuration", {})
    required = {
        "grid": [64, 64],
        "kn0": 10.0,
        "cold_hot_ratio": 0.1,
        "radial_nodes": 40,
        "angular_nodes": 96,
        "point_count": 3840,
        "radial_scale": 2.0,
        "limiter": "minmod",
        "source_relaxation": 1.0,
        "tolerance": 2.0e-5,
        "positivity_floor": 1.0e-30,
        "correction_floor": 0.05,
        "material_perturbation_guard": 1.0e-6,
        "failed_muscl_endpoint_rehabilitated": False,
        "cross_knudsen_extension_permitted": False,
        "validation_claim_permitted": False,
        "solver_endpoint_claim_permitted": False,
    }
    if any(cfg.get(key) != value for key, value in required.items()):
        raise ValueError("Stage-94 frozen design mismatch")
    for distribution in ("phi", "psi"):
        observed = summary.get(distribution, {})
        if not bool(observed.get("finite")) or float(observed.get("strict_negative_fraction", 1.0)) != 0.0:
            raise ValueError("Stage 95 requires the finite nonnegative Stage-94 first-order candidate")
    maximum_macro = max(
        float(stats["relative_l2"])
        for stats in summary.get("macroscopic_floor_perturbation", {}).values()
    )
    if maximum_macro > MATERIAL_PROPAGATION_GUARD:
        raise ValueError("Stage-94 floor perturbation is not negligible under the preregistered Stage-95 guard")
    return summary


def _candidate_update(
    phi: np.ndarray,
    psi: np.ndarray,
    cfg,
    quadrature,
    *,
    include_muscl: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    phi = np.asarray(phi, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    incoming = projected_wall_incoming(phi, psi, cfg, quadrature)
    left_phi, left_psi, right_phi, right_psi, bottom_phi, bottom_psi, top_phi, top_psi = incoming
    fields = projected_macroscopic(phi, psi, quadrature)
    equilibrium_phi, equilibrium_psi, clipping = projected_shakhov_equilibrium(
        fields,
        quadrature,
        prandtl=cfg.prandtl,
        correction_floor=STAGE41_CORRECTION_FLOOR,
    )
    tau = local_relaxation_time(fields["rho"], fields["T"], cfg, mapping="paper_consistent_c0")
    nu = 1.0 / np.maximum(tau, 1.0e-14)
    dx, dy = 1.0 / cfg.nx, 1.0 / cfg.ny
    ax = np.abs(quadrature.vx) / dx
    ay = np.abs(quadrature.vy) / dy
    denominator = nu[..., None] + ax[None, None, :] + ay[None, None, :]

    phi_x, phi_y = _upwind_neighbors(phi, left_phi, right_phi, bottom_phi, top_phi, quadrature)
    candidate_phi = (
        nu[..., None] * equilibrium_phi
        + ax[None, None, :] * phi_x
        + ay[None, None, :] * phi_y
    )
    del phi_x, phi_y, equilibrium_phi
    if include_muscl:
        candidate_phi -= muscl_correction_divergence(
            phi,
            quadrature.vx,
            quadrature.vy,
            dx,
            dy,
            one_sided_x_boundary=False,
        )
    candidate_phi /= denominator

    psi_x, psi_y = _upwind_neighbors(psi, left_psi, right_psi, bottom_psi, top_psi, quadrature)
    candidate_psi = (
        nu[..., None] * equilibrium_psi
        + ax[None, None, :] * psi_x
        + ay[None, None, :] * psi_y
    )
    del psi_x, psi_y, equilibrium_psi
    if include_muscl:
        candidate_psi -= muscl_correction_divergence(
            psi,
            quadrature.vx,
            quadrature.vy,
            dx,
            dy,
            one_sided_x_boundary=False,
        )
    candidate_psi /= denominator

    diagnostics = {
        "maximum_phi_equilibrium_clipped_weight_fraction": float(np.max(clipping["phi_clipped_weight_fraction"])),
        "maximum_psi_equilibrium_clipped_weight_fraction": float(np.max(clipping["psi_clipped_weight_fraction"])),
    }
    return candidate_phi, candidate_psi, diagnostics


def _floor_mask_stats(values: np.ndarray, positivity_floor: float) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    mask = values < positivity_floor
    return {
        "finite": bool(np.isfinite(values).all()),
        "activation_fraction_by_count": float(np.mean(mask)),
        "strict_negative_fraction": float(np.mean(values < 0.0)),
        "minimum_candidate": float(np.min(values)),
        "mask": mask,
    }


def _mask_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 1.0
    return int(np.count_nonzero(a & b)) / union


def _distribution_pair_stats(
    unclipped: np.ndarray,
    clipped_seed: np.ndarray,
    positivity_floor: float,
    weight: np.ndarray,
    speed_squared: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    unclipped = np.asarray(unclipped, dtype=np.float64)
    clipped_seed = np.asarray(clipped_seed, dtype=np.float64)
    if unclipped.shape != clipped_seed.shape or unclipped.ndim != 3 or unclipped.shape[-1] != weight.size:
        raise ValueError("Stage-95 distribution pair shape mismatch")
    delta = clipped_seed - unclipped
    cell_abs_m0 = np.sum(np.abs(delta) * weight, axis=-1)
    baseline_abs_m0 = float(np.sum(np.abs(unclipped) * weight))
    baseline_abs_m2 = float(np.sum(np.abs(unclipped) * speed_squared * weight))
    abs_delta_m0 = float(np.sum(cell_abs_m0))
    abs_delta_m2 = float(np.sum(np.abs(delta) * speed_squared * weight))
    raw_denominator = max(float(np.linalg.norm(unclipped.ravel())), 1.0e-300)

    a = _floor_mask_stats(unclipped, positivity_floor)
    b = _floor_mask_stats(clipped_seed, positivity_floor)
    a_mask = a.pop("mask")
    b_mask = b.pop("mask")
    return {
        "finite": bool(a["finite"] and b["finite"]),
        "raw_relative_l2": float(np.linalg.norm(delta.ravel()) / raw_denominator),
        "maximum_absolute_delta": float(np.max(np.abs(delta))),
        "weighted_m0_relative_absolute_difference": abs_delta_m0 / max(baseline_abs_m0, 1.0e-300),
        "weighted_speed_squared_relative_absolute_difference": abs_delta_m2 / max(baseline_abs_m2, 1.0e-300),
        "unclipped_seed": a,
        "clipped_seed": b,
        "floor_mask_jaccard": _mask_jaccard(a_mask, b_mask),
        "floor_mask_symmetric_difference_fraction": float(np.mean(a_mask ^ b_mask)),
    }, cell_abs_m0


def _macroscopic_pair_stats(
    phi_unclipped: np.ndarray,
    psi_unclipped: np.ndarray,
    phi_clipped_seed: np.ndarray,
    psi_clipped_seed: np.ndarray,
    quadrature,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    before = projected_macroscopic(phi_unclipped, psi_unclipped, quadrature)
    after = projected_macroscopic(phi_clipped_seed, psi_clipped_seed, quadrature)
    names = ("rho", "u", "v", "T", "qx", "qy", "total_internal_moment")
    stats = {name: _field_difference(before[name], after[name]) for name in names}
    maps = {name: np.asarray(after[name] - before[name], dtype=np.float64) for name in ("rho", "u", "v", "T", "qx", "qy")}
    return stats, maps


def _operator_pair_audit(
    first_unclipped_phi: np.ndarray,
    first_unclipped_psi: np.ndarray,
    first_clipped_phi: np.ndarray,
    first_clipped_psi: np.ndarray,
    cfg,
    quadrature,
    *,
    include_muscl: bool,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    phi_a, psi_a, diag_a = _candidate_update(
        first_unclipped_phi,
        first_unclipped_psi,
        cfg,
        quadrature,
        include_muscl=include_muscl,
    )
    phi_b, psi_b, diag_b = _candidate_update(
        first_clipped_phi,
        first_clipped_psi,
        cfg,
        quadrature,
        include_muscl=include_muscl,
    )

    speed_squared = quadrature.vx * quadrature.vx + quadrature.vy * quadrature.vy
    phi_stats, phi_map = _distribution_pair_stats(
        phi_a,
        phi_b,
        cfg.positivity_floor,
        quadrature.weight,
        speed_squared,
    )
    psi_stats, psi_map = _distribution_pair_stats(
        psi_a,
        psi_b,
        cfg.positivity_floor,
        quadrature.weight,
        speed_squared,
    )
    macro_stats, macro_maps = _macroscopic_pair_stats(phi_a, psi_a, phi_b, psi_b, quadrature)
    result = {
        "operator": "baseline_zero_boundary_slope_muscl" if include_muscl else "first_order_source_upwind",
        "second_candidate_floor_application": False,
        "phi": phi_stats,
        "psi": psi_stats,
        "macroscopic_difference": macro_stats,
        "unclipped_seed_equilibrium_clipping": diag_a,
        "clipped_seed_equilibrium_clipping": diag_b,
    }
    maps = {
        "phi_cell_abs_delta_m0": phi_map,
        "psi_cell_abs_delta_m0": psi_map,
        **macro_maps,
    }
    del phi_a, psi_a, phi_b, psi_b
    gc.collect()
    return result, maps


def stage95_decision(first_order: dict[str, object], muscl: dict[str, object]) -> str:
    for operator in (first_order, muscl):
        if not bool(operator["phi"]["finite"]) or not bool(operator["psi"]["finite"]):
            return "stage95_nonfinite_second_update_blocker_without_retuning"

    maximum_macro_relative = max(
        float(stats["relative_l2"])
        for operator in (first_order, muscl)
        for stats in operator["macroscopic_difference"].values()
    )
    maximum_weighted_relative = max(
        float(operator[distribution][metric])
        for operator in (first_order, muscl)
        for distribution in ("phi", "psi")
        for metric in (
            "weighted_m0_relative_absolute_difference",
            "weighted_speed_squared_relative_absolute_difference",
        )
    )
    if max(maximum_macro_relative, maximum_weighted_relative) <= MATERIAL_PROPAGATION_GUARD:
        return "stage95_floor_propagation_negligible_stage96_muscl_correction_growth_audit"
    return "stage95_floor_propagation_material_stage96_floor_sensitive_operator_localization_audit"


def run_stage95(
    stage67_artifact_dir: str | Path,
    stage94_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage95_design(**design)
    stage67_summary = _validate_stage67(stage67_artifact_dir)
    stage94_summary = _validate_stage94(stage94_artifact_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = build_stage58_config()
    quadrature = mapped_polar_quadrature(*RULE, radial_scale=RADIAL_SCALE)
    with np.load(Path(stage67_artifact_dir) / "converged_full_distributions.npz") as saved:
        initial_phi = np.asarray(saved["phi"], dtype=np.float64)
        initial_psi = np.asarray(saved["psi"], dtype=np.float64)
        for name, actual, expected in (
            ("vx", np.asarray(saved["vx"]), np.asarray(quadrature.vx)),
            ("vy", np.asarray(saved["vy"]), np.asarray(quadrature.vy)),
            ("weight", np.asarray(saved["weight"]), np.asarray(quadrature.weight)),
        ):
            if not np.array_equal(actual, expected):
                raise ValueError(f"Stage-67 {name} does not exactly match the frozen Stage-95 quadrature")

        first_phi, first_psi, first_diag = _candidate_update(
            initial_phi,
            initial_psi,
            cfg,
            quadrature,
            include_muscl=False,
        )
    del initial_phi, initial_psi
    gc.collect()

    first_phi_floor_fraction = float(np.mean(first_phi < cfg.positivity_floor))
    first_psi_floor_fraction = float(np.mean(first_psi < cfg.positivity_floor))
    if abs(first_phi_floor_fraction - float(stage94_summary["phi"]["activation_fraction_by_count"])) > 1.0e-15:
        raise ValueError("Stage-95 first phi candidate does not reproduce Stage 94")
    if abs(first_psi_floor_fraction - float(stage94_summary["psi"]["activation_fraction_by_count"])) > 1.0e-15:
        raise ValueError("Stage-95 first psi candidate does not reproduce Stage 94")
    if np.any(first_phi < 0.0) or np.any(first_psi < 0.0):
        raise ValueError("Stage-95 first-order seed unexpectedly contains strict negative values")

    clipped_phi = np.maximum(first_phi, cfg.positivity_floor)
    clipped_psi = np.maximum(first_psi, cfg.positivity_floor)
    first_macro = _macroscopic_pair_stats(first_phi, first_psi, clipped_phi, clipped_psi, quadrature)[0]
    for name, stats in first_macro.items():
        observed = stage94_summary["macroscopic_floor_perturbation"][name]
        if abs(float(stats["relative_l2"]) - float(observed["relative_l2"])) > 1.0e-12:
            raise ValueError(f"Stage-95 first-state {name} floor perturbation does not reproduce Stage 94")

    first_order, first_order_maps = _operator_pair_audit(
        first_phi,
        first_psi,
        clipped_phi,
        clipped_psi,
        cfg,
        quadrature,
        include_muscl=False,
    )
    muscl, muscl_maps = _operator_pair_audit(
        first_phi,
        first_psi,
        clipped_phi,
        clipped_psi,
        cfg,
        quadrature,
        include_muscl=True,
    )

    np.savez_compressed(
        out / "second_update_floor_propagation_maps.npz",
        first_order_phi_cell_abs_delta_m0=first_order_maps["phi_cell_abs_delta_m0"],
        first_order_psi_cell_abs_delta_m0=first_order_maps["psi_cell_abs_delta_m0"],
        first_order_rho_delta=first_order_maps["rho"],
        first_order_T_delta=first_order_maps["T"],
        first_order_qx_delta=first_order_maps["qx"],
        first_order_qy_delta=first_order_maps["qy"],
        muscl_phi_cell_abs_delta_m0=muscl_maps["phi_cell_abs_delta_m0"],
        muscl_psi_cell_abs_delta_m0=muscl_maps["psi_cell_abs_delta_m0"],
        muscl_rho_delta=muscl_maps["rho"],
        muscl_T_delta=muscl_maps["T"],
        muscl_qx_delta=muscl_maps["qx"],
        muscl_qy_delta=muscl_maps["qy"],
    )

    decision = stage95_decision(first_order, muscl)
    maximum_propagated_macro_relative_l2 = max(
        float(stats["relative_l2"])
        for operator in (first_order, muscl)
        for stats in operator["macroscopic_difference"].values()
    )
    maximum_propagated_weighted_relative_difference = max(
        float(operator[distribution][metric])
        for operator in (first_order, muscl)
        for distribution in ("phi", "psi")
        for metric in (
            "weighted_m0_relative_absolute_difference",
            "weighted_speed_squared_relative_absolute_difference",
        )
    )

    summary = {
        "stage": 95,
        "description": (
            "Frozen two-update propagation audit of the negligible Stage-94 positivity-floor insertion. "
            "The exact first-order candidate is propagated once without flooring and once after applying the "
            "retained 1e-30 floor; the second candidate is inspected before any floor is applied."
        ),
        "retained_stage67_decision": stage67_summary["decision"],
        "retained_stage94_decision": stage94_summary["decision"],
        "configuration": {
            "grid": list(GRID),
            "kn0": KNUDSEN,
            "cold_hot_ratio": COLD_HOT_RATIO,
            "radial_nodes": RULE[0],
            "angular_nodes": RULE[1],
            "point_count": RULE[0] * RULE[1],
            "radial_scale": RADIAL_SCALE,
            "limiter": LIMITER,
            "source_relaxation": SOURCE_RELAXATION,
            "tolerance": TOLERANCE,
            "positivity_floor": cfg.positivity_floor,
            "correction_floor": STAGE41_CORRECTION_FLOOR,
            "second_update_index": SECOND_UPDATE_INDEX,
            "material_propagation_guard": MATERIAL_PROPAGATION_GUARD,
            "boundary_reconstruction": "retained zero-boundary-slope baseline for the MUSCL channel",
            "second_candidate_floor_application": False,
            "initialization": "exact completed Stage-67 converged phi/psi",
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "correction_floor_retuning": False,
            "positivity_floor_retuning": False,
            "source_relaxation_retuning": False,
            "transport_parameter_retuning": False,
            "wall_model_retuning": False,
            "normalization_retuning": False,
            "limiter_retuning": False,
            "velocity_quadrature_retuning": False,
            "failed_muscl_endpoint_rehabilitated": False,
            "cross_knudsen_extension_permitted": False,
            "validation_claim_permitted": False,
            "solver_endpoint_claim_permitted": False,
        },
        "stage94_first_state_context": {
            "phi_floor_activation_fraction_by_count": first_phi_floor_fraction,
            "psi_floor_activation_fraction_by_count": first_psi_floor_fraction,
            "maximum_first_state_macroscopic_relative_l2": max(float(v["relative_l2"]) for v in first_macro.values()),
            "maximum_phi_equilibrium_clipped_weight_fraction": first_diag["maximum_phi_equilibrium_clipped_weight_fraction"],
            "maximum_psi_equilibrium_clipped_weight_fraction": first_diag["maximum_psi_equilibrium_clipped_weight_fraction"],
        },
        "second_update_first_order": first_order,
        "second_update_baseline_muscl": muscl,
        "maximum_propagated_macroscopic_relative_l2": maximum_propagated_macro_relative_l2,
        "maximum_propagated_weighted_relative_difference": maximum_propagated_weighted_relative_difference,
        "decision": decision,
        "scientific_conclusion": (
            "Stage 95 tests whether the numerically tiny first-update positivity-floor insertion can be amplified by "
            "one additional nonlinear source/upwind update or by the retained baseline MUSCL reconstruction. It does "
            "not remove or retune the floor and does not seek a converged endpoint. A negligible propagated difference "
            "would rule this particular first-update floor insertion out as a direct explanation of the Stage-90 "
            "nonconvergence at this state; a material difference would require localization before any solver change."
        ),
        "negative_result_guard": (
            "Stage 90 remains nonconverged in both reconstruction arms and Stage 28 remains a failed MUSCL endpoint. "
            "The Stage-89 one-sided boundary slope is not promoted here. Stage 95 is a two-update diagnostic only: no "
            "physics, collision, source-relaxation, transport, wall, limiter, positivity/correction floor, normalization, "
            "tolerance, or velocity-quadrature parameter is changed; no cross-Knudsen extension, accuracy-improvement, "
            "stable-solver, benchmark, or validation claim is permitted."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage94-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage95(
                args.stage67_artifact_dir,
                args.stage94_artifact_dir,
                args.output_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
