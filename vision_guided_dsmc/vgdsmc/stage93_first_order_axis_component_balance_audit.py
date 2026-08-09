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
from .stage42_projected_polar_heated_cavity_pilot import (
    _upwind_neighbors,
    projected_wall_incoming,
)
from .stage58_conservative_solver_64x64_confirmation import build_stage58_config
from .stage90_single_condition_reconstruction_solver_ab_audit import (
    COLD_HOT_RATIO,
    GRID,
    KNUDSEN,
    LIMITER,
    RADIAL_SCALE,
    RULE,
    SOURCE_RELAXATION,
    STAGE67_COMPLETED_ENDPOINT,
    TOLERANCE,
    _validate_stage67,
)

STAGE92_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31301735307,
    "workflow_job_id": 93215501286,
    "artifact_id": 9036454653,
    "artifact_sha256": "c11011c08aaa53ecbd445a5b7e46c39fa8372f9aea9792bd6f539029d43d3352",
    "summary_sha256": "2ce4e4bfb8057e3aa52baf10554059239752119ac9d92534cfe1adc0e8842156",
    "maps_sha256": "7f7ddb448fdef302ab6ea7e757d4259afa468b0149e3cfdcd7851ddc87210d9f",
    "decision": "stage92_mixed_floor_onset_localization_stage93_axis_component_balance_audit",
}
ONSET_STEP = 1


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_stage93_design(**overrides: object) -> None:
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
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 93 is a frozen first-update axis-component audit of the Stage-92 "
            "first-order floor set. It may not retune physics, source relaxation, limiter, "
            "quadrature, wall model, positivity/correction floors, tolerance, or move the "
            "diagnostic away from the first update."
        )


def _validate_stage92(root: str | Path) -> dict[str, object]:
    root = Path(root)
    expected = {
        "summary.json": STAGE92_COMPLETED_ENDPOINT["summary_sha256"],
        "candidate_update_localization_maps.npz": STAGE92_COMPLETED_ENDPOINT["maps_sha256"],
    }
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Stage-92 checksum mismatch: {name}")

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 92 or summary.get("decision") != STAGE92_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage-92 completed endpoint mismatch")

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
        raise ValueError("Stage-92 frozen design mismatch")

    for distribution in ("phi", "psi"):
        result = summary.get(distribution, {})
        baseline = result.get("zero_boundary_slope", {})
        counter = result.get("one_sided_boundary_slope", {})
        overlap = result.get("paired_floor_mask_comparison", {}).get("jaccard_overlap")
        if overlap is None or float(overlap) < 0.99:
            raise ValueError("Stage-93 axis audit requires the strongly shared Stage-92 A/B floor topology")
        for arm in (baseline, counter):
            if not bool(arm.get("finite")):
                raise ValueError("Stage-92 nonfinite result blocks Stage 93")
            if float(arm.get("first_order_floor_fraction", -1.0)) <= 0.0:
                raise ValueError("Stage-93 requires the Stage-92 first-order floor set")
    return summary


def _weighted_sum(values: np.ndarray, qweight: np.ndarray) -> float:
    if values.ndim != 3 or values.shape[-1] != qweight.size:
        raise ValueError("Stage-93 weighted-sum shape mismatch")
    return float(np.sum(np.sum(values, axis=(0, 1)) * qweight))


def _positive_moment_fractions(
    first_order: np.ndarray,
    mask: np.ndarray,
    qweight: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
) -> dict[str, float]:
    multipliers = {
        "m0": np.ones_like(vx),
        "abs_vx": np.abs(vx),
        "abs_vy": np.abs(vy),
        "speed_squared": vx * vx + vy * vy,
    }
    out: dict[str, float] = {}
    counts = np.sum(first_order, axis=(0, 1))
    masked_counts = np.sum(np.where(mask, first_order, 0.0), axis=(0, 1))
    for name, multiplier in multipliers.items():
        denom = float(np.sum(counts * qweight * multiplier))
        numer = float(np.sum(masked_counts * qweight * multiplier))
        out[name] = numer / max(denom, 1.0e-300)
    return out


def _component_balance(
    source: np.ndarray,
    x_transport: np.ndarray,
    y_transport: np.ndarray,
    positivity_floor: float,
    qweight: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    if not (source.shape == x_transport.shape == y_transport.shape):
        raise ValueError("Stage-93 source/x/y components must share shape")
    if source.ndim != 3 or source.shape[-1] != qweight.size:
        raise ValueError("Stage-93 component/quadrature shapes are inconsistent")
    if qweight.shape != vx.shape or vx.shape != vy.shape:
        raise ValueError("Stage-93 quadrature arrays must share shape")

    first_order = source + x_transport + y_transport
    finite = bool(
        np.isfinite(source).all()
        and np.isfinite(x_transport).all()
        and np.isfinite(y_transport).all()
        and np.isfinite(first_order).all()
    )
    mask = first_order < positivity_floor
    active = int(np.count_nonzero(mask))
    total = first_order.size
    ny, nx, _ = first_order.shape

    velocity_active_counts = np.count_nonzero(mask, axis=(0, 1)).astype(np.float64)
    weighted_active = float(np.sum(velocity_active_counts * qweight))
    weighted_total = float(ny * nx * np.sum(qweight))

    component_arrays = {
        "source": source,
        "x_transport": x_transport,
        "y_transport": y_transport,
    }
    masked_first_value = _weighted_sum(np.where(mask, first_order, 0.0), qweight)
    global_first_value = _weighted_sum(first_order, qweight)

    component_stats: dict[str, dict[str, float]] = {}
    for name, component in component_arrays.items():
        masked_value = _weighted_sum(np.where(mask, component, 0.0), qweight)
        component_stats[name] = {
            "minimum": float(np.min(component)),
            "maximum": float(np.max(component)),
            "exact_zero_fraction_of_activations": (
                int(np.count_nonzero((component == 0.0) & mask)) / max(active, 1)
            ),
            "masked_quadrature_value_share": masked_value / max(masked_first_value, 1.0e-300),
            "masked_value_fraction_of_global_first_order": masked_value / max(global_first_value, 1.0e-300),
            "maximum_component_to_floor_on_activations": (
                float(np.max(component[mask] / positivity_floor)) if active else 0.0
            ),
        }

    stack = np.stack((source, x_transport, y_transport), axis=-1)
    component_sum = np.sum(stack, axis=-1)
    nonzero_mask = mask & (component_sum > 0.0)
    all_zero = mask & (component_sum == 0.0)
    dominant = np.argmax(stack, axis=-1)
    dominant_labels = ("source", "x_transport", "y_transport")
    dominant_fractions: dict[str, float] = {}
    nonzero_active = int(np.count_nonzero(nonzero_mask))
    for idx, name in enumerate(dominant_labels):
        dominant_fractions[name] = (
            int(np.count_nonzero((dominant == idx) & nonzero_mask)) / max(nonzero_active, 1)
        )

    closure = first_order - component_sum
    closure_rel_l2 = float(
        np.linalg.norm(closure.ravel()) / max(np.linalg.norm(first_order.ravel()), 1.0e-300)
    )

    axis_mask = np.abs(vx) >= np.abs(vy)
    horizontal_weighted_active = float(np.sum(velocity_active_counts[axis_mask] * qweight[axis_mask]))
    vertical_weighted_active = weighted_active - horizontal_weighted_active

    stats: dict[str, object] = {
        "finite": finite,
        "positivity_floor": float(positivity_floor),
        "floor_activation_fraction_by_count": active / total,
        "floor_activation_fraction_by_quadrature_weight": weighted_active / max(weighted_total, 1.0e-300),
        "strict_negative_fraction": int(np.count_nonzero(first_order < 0.0)) / total,
        "exact_zero_fraction_of_activations": int(np.count_nonzero((first_order == 0.0) & mask)) / max(active, 1),
        "all_components_exact_zero_fraction_of_activations": int(np.count_nonzero(all_zero)) / max(active, 1),
        "masked_quadrature_value_fraction_of_global_first_order": (
            masked_first_value / max(global_first_value, 1.0e-300)
        ),
        "positive_reduced_moment_fractions_from_floor_set": _positive_moment_fractions(
            first_order,
            mask,
            qweight,
            vx,
            vy,
        ),
        "horizontal_velocity_quadrature_weight_share_of_activations": (
            horizontal_weighted_active / max(weighted_active, 1.0e-300)
        ),
        "vertical_velocity_quadrature_weight_share_of_activations": (
            vertical_weighted_active / max(weighted_active, 1.0e-300)
        ),
        "dominant_component_fraction_among_nonzero_activations": dominant_fractions,
        "component_statistics": component_stats,
        "component_sum_closure_relative_l2": closure_rel_l2,
    }

    cell_map = np.mean(mask, axis=2)
    velocity_map = np.mean(mask, axis=(0, 1))
    return stats, cell_map, velocity_map


def stage93_decision(phi: dict[str, object], psi: dict[str, object]) -> str:
    if not bool(phi.get("finite")) or not bool(psi.get("finite")):
        return "stage93_nonfinite_first_order_component_blocker_without_retuning"
    return "stage93_axis_component_balance_complete_stage94_floor_moment_perturbation_audit"


def _distribution_components(
    distribution: np.ndarray,
    equilibrium: np.ndarray,
    wall_values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    nu: np.ndarray,
    denominator: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    cfg,
    quadrature,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, right, bottom, top = wall_values
    upstream_x, upstream_y = _upwind_neighbors(
        distribution,
        left,
        right,
        bottom,
        top,
        quadrature,
    )
    source = (nu[..., None] * equilibrium) / denominator
    x_transport = (ax[None, None, :] * upstream_x) / denominator
    y_transport = (ay[None, None, :] * upstream_y) / denominator
    return source, x_transport, y_transport


def run_stage93(
    stage67_artifact_dir: str | Path,
    stage92_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage93_design(**design)
    stage67_summary = _validate_stage67(stage67_artifact_dir)
    stage92_summary = _validate_stage92(stage92_artifact_dir)

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
                raise ValueError(f"Stage-67 {name} does not exactly match the frozen Stage-93 quadrature")

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
        phi_result, phi_cell_map, phi_velocity_map = _component_balance(
            *phi_components,
            cfg.positivity_floor,
            quadrature.weight,
            quadrature.vx,
            quadrature.vy,
        )
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
        psi_result, psi_cell_map, psi_velocity_map = _component_balance(
            *psi_components,
            cfg.positivity_floor,
            quadrature.weight,
            quadrature.vx,
            quadrature.vy,
        )
        del psi_components

    for distribution, result in (("phi", phi_result), ("psi", psi_result)):
        retained = float(
            stage92_summary[distribution]["zero_boundary_slope"]["first_order_floor_fraction"]
        )
        reproduced = float(result["floor_activation_fraction_by_count"])
        if abs(retained - reproduced) > 1.0e-15:
            raise ValueError(
                f"Stage-93 {distribution} first-order floor set does not reproduce Stage 92: "
                f"{reproduced} versus {retained}"
            )

    np.savez_compressed(
        out / "first_order_axis_component_maps.npz",
        phi_cell_floor_fraction=phi_cell_map,
        psi_cell_floor_fraction=psi_cell_map,
        phi_velocity_floor_fraction=phi_velocity_map,
        psi_velocity_floor_fraction=psi_velocity_map,
    )

    decision = stage93_decision(phi_result, psi_result)
    summary = {
        "stage": 93,
        "description": (
            "Frozen first-update axis-component balance audit of the first-order source/upwind "
            "candidate identified by Stage 92 as the dominant floor-activation set."
        ),
        "retained_stage67_decision": stage67_summary["decision"],
        "retained_stage92_decision": stage92_summary["decision"],
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
        "stage92_context": {
            "phi_first_order_floor_fraction": float(
                stage92_summary["phi"]["zero_boundary_slope"]["first_order_floor_fraction"]
            ),
            "psi_first_order_floor_fraction": float(
                stage92_summary["psi"]["zero_boundary_slope"]["first_order_floor_fraction"]
            ),
            "phi_ab_floor_mask_jaccard": float(
                stage92_summary["phi"]["paired_floor_mask_comparison"]["jaccard_overlap"]
            ),
            "psi_ab_floor_mask_jaccard": float(
                stage92_summary["psi"]["paired_floor_mask_comparison"]["jaccard_overlap"]
            ),
            "phi_baseline_correction_induced_fraction_of_activations": float(
                stage92_summary["phi"]["zero_boundary_slope"]["correction_induced_fraction_of_activations"]
            ),
            "psi_baseline_correction_induced_fraction_of_activations": float(
                stage92_summary["psi"]["zero_boundary_slope"]["correction_induced_fraction_of_activations"]
            ),
            "phi_one_sided_correction_induced_fraction_of_activations": float(
                stage92_summary["phi"]["one_sided_boundary_slope"]["correction_induced_fraction_of_activations"]
            ),
            "psi_one_sided_correction_induced_fraction_of_activations": float(
                stage92_summary["psi"]["one_sided_boundary_slope"]["correction_induced_fraction_of_activations"]
            ),
        },
        "equilibrium_clipping": {
            "maximum_phi_clipped_weight_fraction": float(np.max(clipping["phi_clipped_weight_fraction"])),
            "maximum_psi_clipped_weight_fraction": float(np.max(clipping["psi_clipped_weight_fraction"])),
        },
        "phi": phi_result,
        "psi": psi_result,
        "decision": decision,
        "scientific_conclusion": (
            "Stage 93 decomposes the already-observed first-order floor set into source, x-upwind, "
            "and y-upwind contributions and reports both node-count and quadrature-weighted/moment "
            "footprints. It is an operator-balance diagnostic only. It does not identify a parameter "
            "that should be tuned and does not establish solver stability, accuracy improvement, or validation."
        ),
        "negative_result_guard": (
            "Stage 90 remains nonconverged in both reconstruction arms and Stage 28 remains a failed "
            "MUSCL endpoint. Stage 92 showed that the one-sided boundary counterfactual leaves the "
            "floor topology almost unchanged and that most floor activations are already present in "
            "the first-order source/upwind candidate. No failed parameter is retuned, no solver is "
            "rerun, no cross-Knudsen extension is made, and no benchmark, validation, accuracy-improvement, "
            "or stable-solver claim is authorized."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage92-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage93(
                args.stage67_artifact_dir,
                args.stage92_artifact_dir,
                args.output_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
