from __future__ import annotations

import argparse
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
from .stage42_projected_polar_heated_cavity_pilot import projected_wall_incoming
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
)
from .stage93_first_order_axis_component_balance_audit import _distribution_components


STAGE93_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31309184991,
    "workflow_job_id": 93234231556,
    "artifact_id": 9038287844,
    "artifact_sha256": "dd8cbdd4f85c9196a1b2b622e8e6c9b2e77227a69ad67d6bbda3732040f541cb",
    "summary_sha256": "eb4978115321f98cd7da9f1ed1180553ce1c21959b4be9aa3619025c6fc12d71",
    "maps_sha256": "59d47380a4ac71c7d5b72f8567ae2e09d18c831bb1053b01e40582c2849410de",
    "decision": "stage93_axis_component_balance_complete_stage94_floor_moment_perturbation_audit",
}
ONSET_STEP = 1
MATERIAL_PERTURBATION_GUARD = 1.0e-6


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_stage94_design(**overrides: object) -> None:
    frozen = {
        "grid": GRID,
        "kn0": KNUDSEN,
        "cold_hot_ratio": COLD_HOT_RATIO,
        "rule": RULE,
        "radial_scale": RADIAL_SCALE,
        "limiter": LIMITER,
        "source_relaxation": SOURCE_RELAXATION,
        "tolerance": TOLERANCE,
        "onset_step": ONSET_STEP,
        "correction_floor": STAGE41_CORRECTION_FLOOR,
        "material_perturbation_guard": MATERIAL_PERTURBATION_GUARD,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 94 is a frozen first-update positivity-floor moment-perturbation audit. "
            "It may not retune physics, source relaxation, limiter, quadrature, wall model, "
            "positivity/correction floors, tolerance, the diagnostic update, or the fixed "
            "materiality reporting guard."
        )


def _validate_stage93(root: str | Path) -> dict[str, object]:
    root = Path(root)
    expected = {
        "summary.json": STAGE93_COMPLETED_ENDPOINT["summary_sha256"],
        "first_order_axis_component_maps.npz": STAGE93_COMPLETED_ENDPOINT["maps_sha256"],
    }
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Stage-93 checksum mismatch: {name}")

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 93 or summary.get("decision") != STAGE93_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage-93 completed endpoint mismatch")

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
        "onset_step": 1,
        "failed_muscl_endpoint_rehabilitated": False,
        "cross_knudsen_extension_permitted": False,
        "validation_claim_permitted": False,
        "solver_endpoint_claim_permitted": False,
    }
    if any(cfg.get(key) != value for key, value in required.items()):
        raise ValueError("Stage-93 frozen design mismatch")

    for distribution in ("phi", "psi"):
        result = summary.get(distribution, {})
        if not bool(result.get("finite")):
            raise ValueError("Stage-93 nonfinite first-order candidate blocks Stage 94")
        if float(result.get("floor_activation_fraction_by_count", 0.0)) <= 0.0:
            raise ValueError("Stage-94 requires the observed Stage-93 first-order floor set")
        if float(result.get("strict_negative_fraction", 1.0)) != 0.0:
            raise ValueError("Stage-94 is preregistered for the observed nonnegative subfloor set")
    return summary


def _weighted_reduce(values: np.ndarray, multiplier: np.ndarray, weight: np.ndarray) -> float:
    if values.ndim != 3 or values.shape[-1] != weight.size:
        raise ValueError("Stage-94 distribution/quadrature shape mismatch")
    if multiplier.shape != weight.shape:
        raise ValueError("Stage-94 multiplier/quadrature shape mismatch")
    velocity_sum = np.sum(values, axis=(0, 1))
    return float(np.sum(velocity_sum * multiplier * weight))


def _floor_perturbation(
    candidate: np.ndarray,
    positivity_floor: float,
    weight: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    candidate = np.asarray(candidate, dtype=np.float64)
    if candidate.ndim != 3 or candidate.shape[-1] != weight.size:
        raise ValueError("Stage-94 candidate shape mismatch")
    if not (weight.shape == vx.shape == vy.shape):
        raise ValueError("Stage-94 quadrature arrays must share shape")
    if positivity_floor <= 0.0:
        raise ValueError("Stage-94 positivity floor must be positive")

    active = candidate < positivity_floor
    delta = np.where(active, positivity_floor - candidate, 0.0)
    finite = bool(np.isfinite(candidate).all() and np.isfinite(delta).all())
    strict_negative = candidate < 0.0
    exact_zero = candidate == 0.0

    multipliers = {
        "m0": np.ones_like(vx),
        "vx": vx,
        "vy": vy,
        "abs_vx": np.abs(vx),
        "abs_vy": np.abs(vy),
        "speed_squared": vx * vx + vy * vy,
    }
    moment_stats: dict[str, dict[str, float]] = {}
    for name, multiplier in multipliers.items():
        signed_delta = _weighted_reduce(delta, multiplier, weight)
        baseline_abs = _weighted_reduce(np.abs(candidate), np.abs(multiplier), weight)
        delta_abs = _weighted_reduce(delta, np.abs(multiplier), weight)
        moment_stats[name] = {
            "signed_delta": signed_delta,
            "absolute_delta": delta_abs,
            "baseline_absolute_scale": baseline_abs,
            "relative_absolute_perturbation": delta_abs / max(baseline_abs, 1.0e-300),
        }

    cell_added_m0 = np.sum(delta * weight, axis=-1)
    cell_baseline_m0 = np.sum(candidate * weight, axis=-1)
    cell_relative_m0 = cell_added_m0 / np.maximum(np.abs(cell_baseline_m0), 1.0e-300)

    active_count = int(np.count_nonzero(active))
    delta_weighted = float(np.sum(cell_added_m0))
    zero_delta_weighted = float(np.sum(np.where(exact_zero, delta, 0.0) * weight))

    stats: dict[str, object] = {
        "finite": finite,
        "positivity_floor": float(positivity_floor),
        "activation_fraction_by_count": active_count / candidate.size,
        "strict_negative_fraction": int(np.count_nonzero(strict_negative)) / candidate.size,
        "exact_zero_fraction_of_activations": int(np.count_nonzero(exact_zero & active)) / max(active_count, 1),
        "positive_subfloor_fraction_of_activations": int(np.count_nonzero((candidate > 0.0) & active)) / max(active_count, 1),
        "added_weighted_value": delta_weighted,
        "exact_zero_share_of_added_weighted_value": zero_delta_weighted / max(delta_weighted, 1.0e-300),
        "maximum_cell_relative_m0_perturbation": float(np.max(cell_relative_m0)),
        "mean_cell_relative_m0_perturbation": float(np.mean(cell_relative_m0)),
        "moment_perturbations": moment_stats,
    }
    return stats, delta, cell_added_m0


def _field_difference(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    if before.shape != after.shape:
        raise ValueError("Stage-94 macroscopic field shape mismatch")
    delta = after - before
    return {
        "relative_l2": float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(before.ravel()), 1.0e-300)),
        "maximum_absolute_delta": float(np.max(np.abs(delta))),
        "relative_to_baseline_max": float(np.max(np.abs(delta)) / max(np.max(np.abs(before)), 1.0e-300)),
        "mean_absolute_delta": float(np.mean(np.abs(delta))),
    }


def stage94_decision(
    phi: dict[str, object],
    psi: dict[str, object],
    macroscopic: dict[str, dict[str, float]],
) -> str:
    if not bool(phi.get("finite")) or not bool(psi.get("finite")):
        return "stage94_nonfinite_floor_perturbation_blocker_without_retuning"
    if float(phi.get("strict_negative_fraction", 0.0)) > 0.0 or float(psi.get("strict_negative_fraction", 0.0)) > 0.0:
        return "stage94_unexpected_negative_candidate_blocker_without_retuning"
    maximum_relative_l2 = max(float(stats["relative_l2"]) for stats in macroscopic.values())
    if maximum_relative_l2 <= MATERIAL_PERTURBATION_GUARD:
        return "stage94_floor_moment_perturbation_negligible_stage95_unclipped_second_update_propagation_audit"
    return "stage94_floor_moment_perturbation_material_stage95_clipped_unclipped_second_update_propagation_audit"


def run_stage94(
    stage67_artifact_dir: str | Path,
    stage93_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage94_design(**design)
    stage67_summary = _validate_stage67(stage67_artifact_dir)
    stage93_summary = _validate_stage93(stage93_artifact_dir)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = build_stage58_config()
    quadrature = mapped_polar_quadrature(*RULE, radial_scale=RADIAL_SCALE)

    with np.load(Path(stage67_artifact_dir) / "converged_full_distributions.npz") as saved:
        phi = np.asarray(saved["phi"], dtype=np.float64)
        psi = np.asarray(saved["psi"], dtype=np.float64)
        for name, actual, expected in (
            ("vx", np.asarray(saved["vx"]), np.asarray(quadrature.vx)),
            ("vy", np.asarray(saved["vy"]), np.asarray(quadrature.vy)),
            ("weight", np.asarray(saved["weight"]), np.asarray(quadrature.weight)),
        ):
            if not np.array_equal(actual, expected):
                raise ValueError(f"Stage-67 {name} does not exactly match the frozen Stage-94 quadrature")

        incoming = projected_wall_incoming(phi, psi, cfg, quadrature)
        (
            left_phi,
            left_psi,
            right_phi,
            right_psi,
            bottom_phi,
            bottom_psi,
            top_phi,
            top_psi,
        ) = incoming
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

        phi_components = _distribution_components(
            phi,
            equilibrium_phi,
            (left_phi, right_phi, bottom_phi, top_phi),
            nu,
            denominator,
            ax,
            ay,
            cfg,
            quadrature,
        )
        candidate_phi = phi_components[0] + phi_components[1] + phi_components[2]
        del phi_components

        psi_components = _distribution_components(
            psi,
            equilibrium_psi,
            (left_psi, right_psi, bottom_psi, top_psi),
            nu,
            denominator,
            ax,
            ay,
            cfg,
            quadrature,
        )
        candidate_psi = psi_components[0] + psi_components[1] + psi_components[2]
        del psi_components

    phi_result, delta_phi, phi_cell_added_m0 = _floor_perturbation(
        candidate_phi,
        cfg.positivity_floor,
        quadrature.weight,
        quadrature.vx,
        quadrature.vy,
    )
    psi_result, delta_psi, psi_cell_added_m0 = _floor_perturbation(
        candidate_psi,
        cfg.positivity_floor,
        quadrature.weight,
        quadrature.vx,
        quadrature.vy,
    )

    for distribution, result in (("phi", phi_result), ("psi", psi_result)):
        observed = stage93_summary[distribution]
        if abs(float(result["activation_fraction_by_count"]) - float(observed["floor_activation_fraction_by_count"])) > 1.0e-15:
            raise ValueError(f"Stage-94 {distribution} floor set does not reproduce Stage 93")
        if abs(float(result["strict_negative_fraction"]) - float(observed["strict_negative_fraction"])) > 1.0e-15:
            raise ValueError(f"Stage-94 {distribution} negative set does not reproduce Stage 93")

    before_fields = projected_macroscopic(candidate_phi, candidate_psi, quadrature)
    candidate_phi += delta_phi
    candidate_psi += delta_psi
    after_fields = projected_macroscopic(candidate_phi, candidate_psi, quadrature)

    field_names = ("rho", "u", "v", "T", "qx", "qy", "total_internal_moment")
    macroscopic = {
        name: _field_difference(before_fields[name], after_fields[name])
        for name in field_names
    }

    np.savez_compressed(
        out / "floor_moment_perturbation_maps.npz",
        phi_cell_added_m0=phi_cell_added_m0,
        psi_cell_added_m0=psi_cell_added_m0,
        rho_delta=after_fields["rho"] - before_fields["rho"],
        u_delta=after_fields["u"] - before_fields["u"],
        v_delta=after_fields["v"] - before_fields["v"],
        T_delta=after_fields["T"] - before_fields["T"],
        qx_delta=after_fields["qx"] - before_fields["qx"],
        qy_delta=after_fields["qy"] - before_fields["qy"],
    )

    decision = stage94_decision(phi_result, psi_result, macroscopic)
    summary = {
        "stage": 94,
        "description": (
            "Frozen first-update audit of the actual moment perturbation created by applying the "
            "retained 1e-30 positivity floor to the Stage-93 first-order source/upwind candidate."
        ),
        "retained_stage67_decision": stage67_summary["decision"],
        "retained_stage93_decision": stage93_summary["decision"],
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
            "onset_step": ONSET_STEP,
            "positivity_floor": cfg.positivity_floor,
            "correction_floor": STAGE41_CORRECTION_FLOOR,
            "material_perturbation_guard": MATERIAL_PERTURBATION_GUARD,
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
        "stage93_context": {
            "phi_floor_activation_fraction_by_count": float(stage93_summary["phi"]["floor_activation_fraction_by_count"]),
            "psi_floor_activation_fraction_by_count": float(stage93_summary["psi"]["floor_activation_fraction_by_count"]),
            "phi_floor_activation_fraction_by_quadrature_weight": float(stage93_summary["phi"]["floor_activation_fraction_by_quadrature_weight"]),
            "psi_floor_activation_fraction_by_quadrature_weight": float(stage93_summary["psi"]["floor_activation_fraction_by_quadrature_weight"]),
            "phi_masked_value_fraction": float(stage93_summary["phi"]["masked_quadrature_value_fraction_of_global_first_order"]),
            "psi_masked_value_fraction": float(stage93_summary["psi"]["masked_quadrature_value_fraction_of_global_first_order"]),
        },
        "equilibrium_clipping": {
            "maximum_phi_clipped_weight_fraction": float(np.max(clipping["phi_clipped_weight_fraction"])),
            "maximum_psi_clipped_weight_fraction": float(np.max(clipping["psi_clipped_weight_fraction"])),
        },
        "phi": phi_result,
        "psi": psi_result,
        "macroscopic_floor_perturbation": macroscopic,
        "decision": decision,
        "scientific_conclusion": (
            "Stage 94 measures the perturbation produced by the retained positivity floor rather than "
            "inferring importance from the size of the binary floor mask. It is a frozen operator diagnostic. "
            "A material perturbation would justify tracing clipped versus unclipped propagation without "
            "changing the floor; a negligible perturbation would rule the first-update floor insertion out "
            "as a direct moment-scale explanation at this state. Neither outcome validates or repairs the solver."
        ),
        "negative_result_guard": (
            "Stage 90 remains nonconverged in both reconstruction arms and Stage 28 remains a failed MUSCL "
            "endpoint. Stage 94 does not change or remove the positivity floor, does not rerun the cavity solver, "
            "does not retune physics, collision, wall, quadrature, source-relaxation, limiter, tolerance, or "
            "normalization parameters, and does not permit cross-Knudsen extension, validation, accuracy-improvement, "
            "or stable-solver claims."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage93-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage94(
                args.stage67_artifact_dir,
                args.stage93_artifact_dir,
                args.output_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
