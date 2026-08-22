from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    muscl_correction_divergence,
)

STAGE91_COMPLETED_ENDPOINT = {
    "workflow_run_id": 31297331158,
    "workflow_job_id": 93204469193,
    "artifact_id": 9034582171,
    "artifact_sha256": "b0c3baec5f59025f1898a34c757091e4f3bb5d29f93065525cf23f3b013f13c6",
    "summary_sha256": "e8a0ec372ec934b05c935daccd2a1949db4bf18977827bb3f38ad006248e60d4",
    "histories_sha256": "153bc27c9ac4c5d77465ec63690ce13b0ef5a585e1ed600a1a7341ca521b3aaa",
    "decision": "stage91_both_arms_activate_positivity_floor_within_fixed_onset_window_stage92_candidate_update_localization_audit",
}
ONSET_STEP = 1
ANGULAR_SECTORS = 8


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_stage92_design(**overrides: object) -> None:
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
        "angular_sectors": ANGULAR_SECTORS,
        "correction_floor": STAGE41_CORRECTION_FLOOR,
    }
    if any(key not in frozen or frozen[key] != value for key, value in overrides.items()):
        raise ValueError(
            "Stage 92 is a frozen first-update localization audit of the Stage-91 onset. "
            "It may not retune physics, source relaxation, limiter, quadrature, wall model, "
            "positivity/correction floors, tolerance, or move the diagnostic away from the "
            "first update where both Stage-90 arms already activated the positivity floor."
        )


def _validate_stage91(root: str | Path) -> dict[str, object]:
    root = Path(root)
    expected = {
        "summary.json": STAGE91_COMPLETED_ENDPOINT["summary_sha256"],
        "onset_histories.npz": STAGE91_COMPLETED_ENDPOINT["histories_sha256"],
    }
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Stage-91 checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 91 or summary.get("decision") != STAGE91_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage-91 completed endpoint mismatch")
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
        "diagnostic_steps": 25,
        "failed_muscl_endpoint_rehabilitated": False,
        "cross_knudsen_extension_permitted": False,
        "validation_claim_permitted": False,
    }
    if any(cfg.get(key) != value for key, value in required.items()):
        raise ValueError("Stage-91 frozen design mismatch")
    for arm_name in ("zero_boundary_slope", "one_sided_boundary_slope"):
        arm = summary.get(arm_name, {})
        if arm.get("first_phi_floor_activation_step") != 1 or arm.get("first_psi_floor_activation_step") != 1:
            raise ValueError("Stage-92 is only justified because both Stage-91 arms activate both floors at step 1")
    return summary


def _layer_index(ny: int, nx: int) -> np.ndarray:
    yy = np.arange(ny, dtype=np.int64)[:, None]
    xx = np.arange(nx, dtype=np.int64)[None, :]
    return np.minimum.reduce(
        [
            np.broadcast_to(yy, (ny, nx)),
            np.broadcast_to(ny - 1 - yy, (ny, nx)),
            np.broadcast_to(xx, (ny, nx)),
            np.broadcast_to(nx - 1 - xx, (ny, nx)),
        ]
    )


def _x_layer_index(nx: int) -> np.ndarray:
    xx = np.arange(nx, dtype=np.int64)
    return np.minimum(xx, nx - 1 - xx)


def _layer_rates(mask: np.ndarray, layer: np.ndarray) -> dict[str, dict[str, float]]:
    ny, nx, nq = mask.shape
    total_active = int(np.count_nonzero(mask))
    output: dict[str, dict[str, float]] = {}
    for label, selector in (
        ("0", layer == 0),
        ("1", layer == 1),
        ("2", layer == 2),
        ("3", layer == 3),
        ("4plus", layer >= 4),
    ):
        cells = int(np.count_nonzero(selector))
        active = int(np.count_nonzero(mask[selector])) if cells else 0
        output[label] = {
            "activation_rate_within_layer": active / max(cells * nq, 1),
            "fraction_of_all_activations": active / max(total_active, 1),
        }
    return output


def _angular_sector_rates(mask: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> list[dict[str, float]]:
    counts = np.count_nonzero(mask, axis=(0, 1)).astype(np.float64)
    total = float(np.sum(counts))
    theta = np.mod(np.arctan2(vy, vx), 2.0 * np.pi)
    sectors = np.minimum((theta / (2.0 * np.pi) * ANGULAR_SECTORS).astype(int), ANGULAR_SECTORS - 1)
    output: list[dict[str, float]] = []
    for sector in range(ANGULAR_SECTORS):
        selected = sectors == sector
        active = float(np.sum(counts[selected]))
        output.append(
            {
                "sector": int(sector),
                "theta_start_degrees": 360.0 * sector / ANGULAR_SECTORS,
                "theta_end_degrees": 360.0 * (sector + 1) / ANGULAR_SECTORS,
                "fraction_of_all_activations": active / max(total, 1.0),
            }
        )
    return output


def _activation_statistics(
    first_order: np.ndarray,
    candidate: np.ndarray,
    correction_over_denominator: np.ndarray,
    positivity_floor: float,
    vx: np.ndarray,
    vy: np.ndarray,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    if first_order.shape != candidate.shape or candidate.shape != correction_over_denominator.shape:
        raise ValueError("Stage-92 candidate arrays must share shape")
    if candidate.ndim != 3 or candidate.shape[-1] != vx.size or vx.shape != vy.shape:
        raise ValueError("Stage-92 candidate/quadrature shapes are inconsistent")

    floor_mask = candidate < positivity_floor
    first_order_floor = first_order < positivity_floor
    correction_induced = floor_mask & ~first_order_floor
    correction_rescued = first_order_floor & ~floor_mask
    negative_mask = candidate < 0.0

    ny, nx, nq = candidate.shape
    total = candidate.size
    active = int(np.count_nonzero(floor_mask))
    induced = int(np.count_nonzero(correction_induced))
    first_active = int(np.count_nonzero(first_order_floor))
    rescued = int(np.count_nonzero(correction_rescued))

    wall_layer = _layer_index(ny, nx)
    x_layer_1d = _x_layer_index(nx)
    x_layer = np.broadcast_to(x_layer_1d[None, :], (ny, nx))

    ratio = np.abs(correction_over_denominator) / np.maximum(np.abs(first_order), 1.0e-300)
    active_ratio = ratio[floor_mask]
    horizontally_dominant = np.abs(vx) >= np.abs(vy)
    velocity_counts = np.count_nonzero(floor_mask, axis=(0, 1)).astype(np.float64)
    horizontal_active = float(np.sum(velocity_counts[horizontally_dominant]))

    stats: dict[str, object] = {
        "finite": bool(np.isfinite(candidate).all() and np.isfinite(first_order).all()),
        "positivity_floor": float(positivity_floor),
        "floor_activation_fraction": active / total,
        "strict_negative_fraction": int(np.count_nonzero(negative_mask)) / total,
        "first_order_floor_fraction": first_active / total,
        "correction_induced_fraction_of_activations": induced / max(active, 1),
        "correction_rescued_fraction_of_first_order_activations": rescued / max(first_active, 1),
        "minimum_candidate": float(np.min(candidate)),
        "minimum_first_order_candidate": float(np.min(first_order)),
        "maximum_abs_correction_to_first_order_ratio_on_activations": (
            float(np.max(active_ratio)) if active_ratio.size else 0.0
        ),
        "mean_abs_correction_to_first_order_ratio_on_activations": (
            float(np.mean(active_ratio)) if active_ratio.size else 0.0
        ),
        "wall_distance_layers": _layer_rates(floor_mask, wall_layer),
        "x_wall_distance_layers": _layer_rates(floor_mask, x_layer),
        "horizontal_velocity_dominant_fraction_of_activations": horizontal_active / max(float(active), 1.0),
        "angular_sector_activation_fractions": _angular_sector_rates(floor_mask, vx, vy),
    }
    cell_map = np.mean(floor_mask, axis=2)
    velocity_map = np.mean(floor_mask, axis=(0, 1))
    return stats, floor_mask, cell_map, velocity_map


def _mask_overlap(baseline: np.ndarray, one_sided: np.ndarray) -> dict[str, float]:
    if baseline.shape != one_sided.shape:
        raise ValueError("Stage-92 A/B floor masks must share shape")
    intersection = int(np.count_nonzero(baseline & one_sided))
    union = int(np.count_nonzero(baseline | one_sided))
    base = int(np.count_nonzero(baseline))
    counter = int(np.count_nonzero(one_sided))
    added = int(np.count_nonzero(one_sided & ~baseline))
    removed = int(np.count_nonzero(baseline & ~one_sided))
    total = baseline.size
    return {
        "jaccard_overlap": intersection / max(union, 1),
        "shared_fraction_of_baseline_activations": intersection / max(base, 1),
        "shared_fraction_of_one_sided_activations": intersection / max(counter, 1),
        "one_sided_added_fraction_of_all_entries": added / total,
        "one_sided_removed_fraction_of_all_entries": removed / total,
    }


def _distribution_audit(
    distribution: np.ndarray,
    equilibrium: np.ndarray,
    wall_values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    nu: np.ndarray,
    denominator: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    cfg,
    quadrature,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    left, right, bottom, top = wall_values
    upstream_x, upstream_y = _upwind_neighbors(
        distribution,
        left,
        right,
        bottom,
        top,
        quadrature,
    )
    numerator = (
        nu[..., None] * equilibrium
        + ax[None, None, :] * upstream_x
        + ay[None, None, :] * upstream_y
    )
    first_order = numerator / denominator
    del numerator, upstream_x, upstream_y

    results: dict[str, object] = {}
    maps: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}

    dx, dy = 1.0 / cfg.nx, 1.0 / cfg.ny
    for arm_name, one_sided in (
        ("zero_boundary_slope", False),
        ("one_sided_boundary_slope", True),
    ):
        correction = muscl_correction_divergence(
            distribution,
            quadrature.vx,
            quadrature.vy,
            dx,
            dy,
            one_sided,
        )
        correction_over_denominator = correction / denominator
        candidate = first_order - correction_over_denominator
        stats, mask, cell_map, velocity_map = _activation_statistics(
            first_order,
            candidate,
            correction_over_denominator,
            cfg.positivity_floor,
            quadrature.vx,
            quadrature.vy,
        )
        results[arm_name] = stats
        masks[arm_name] = mask
        maps[f"{arm_name}_cell_floor_fraction"] = cell_map
        maps[f"{arm_name}_velocity_floor_fraction"] = velocity_map
        del correction, correction_over_denominator, candidate

    results["paired_floor_mask_comparison"] = _mask_overlap(
        masks["zero_boundary_slope"],
        masks["one_sided_boundary_slope"],
    )
    del masks, first_order
    return results, maps


def stage92_decision(phi: dict[str, object], psi: dict[str, object]) -> str:
    arm_stats = [
        phi["zero_boundary_slope"],
        phi["one_sided_boundary_slope"],
        psi["zero_boundary_slope"],
        psi["one_sided_boundary_slope"],
    ]
    if not all(bool(arm["finite"]) for arm in arm_stats):
        return "stage92_nonfinite_first_update_blocker_without_retuning"
    induced_min = min(float(arm["correction_induced_fraction_of_activations"]) for arm in arm_stats)
    overlap_min = min(
        float(phi["paired_floor_mask_comparison"]["jaccard_overlap"]),
        float(psi["paired_floor_mask_comparison"]["jaccard_overlap"]),
    )
    if induced_min >= 0.99 and overlap_min >= 0.90:
        return "stage92_shared_correction_induced_floor_onset_stage93_axis_component_balance_audit"
    if overlap_min < 0.50:
        return "stage92_boundary_counterfactual_changes_floor_topology_stage93_boundary_specific_balance_audit"
    return "stage92_mixed_floor_onset_localization_stage93_axis_component_balance_audit"


def run_stage92(
    stage67_artifact_dir: str | Path,
    stage91_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage92_design(**design)
    stage67_summary = _validate_stage67(stage67_artifact_dir)
    stage91_summary = _validate_stage91(stage91_artifact_dir)

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
                raise ValueError(f"Stage-67 {name} does not exactly match the frozen Stage-92 quadrature")

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

        phi_result, phi_maps = _distribution_audit(
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
        psi_result, psi_maps = _distribution_audit(
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

    np.savez_compressed(
        out / "candidate_update_localization_maps.npz",
        **{f"phi_{key}": value for key, value in phi_maps.items()},
        **{f"psi_{key}": value for key, value in psi_maps.items()},
    )

    decision = stage92_decision(phi_result, psi_result)
    summary = {
        "stage": 92,
        "description": (
            "Frozen first-update candidate localization audit following Stage 91, where both "
            "Stage-90 reconstruction arms activated the positivity floor for phi and psi at step 1."
        ),
        "retained_stage67_decision": stage67_summary["decision"],
        "retained_stage91_decision": stage91_summary["decision"],
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
            "angular_sectors": ANGULAR_SECTORS,
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
        "equilibrium_clipping": {
            "maximum_phi_clipped_weight_fraction": float(np.max(clipping["phi_clipped_weight_fraction"])),
            "maximum_psi_clipped_weight_fraction": float(np.max(clipping["psi_clipped_weight_fraction"])),
        },
        "phi": phi_result,
        "psi": psi_result,
        "decision": decision,
        "scientific_conclusion": (
            "Stage 92 localizes the already-observed first-update floor activation in physical and "
            "velocity space and separates entries that were already below the floor under the "
            "first-order source/upwind candidate from entries driven below the floor only "
            "after the frozen MUSCL correction. This is an operator diagnostic, not evidence that "
            "the limiter, positivity floor, wall treatment, transport model, or physical parameters "
            "should be changed."
        ),
        "negative_result_guard": (
            "Stage 90 remains nonconverged in both arms, Stage 28 remains a failed MUSCL endpoint, "
            "and Stage 91 already showed floor activation at the first update in both arms. "
            "No failed parameter is retuned, no cross-Knudsen extension is made, and no benchmark, "
            "validation, accuracy-improvement, or stable-solver claim is authorized."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage67-artifact-dir", required=True)
    parser.add_argument("--stage91-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage92(
                args.stage67_artifact_dir,
                args.stage91_artifact_dir,
                args.output_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
