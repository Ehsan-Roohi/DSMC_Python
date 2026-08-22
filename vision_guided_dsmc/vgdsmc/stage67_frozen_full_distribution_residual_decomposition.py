from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math
from typing import Mapping

import numpy as np

from .stage34_velocity_scale_consistency import local_relaxation_time
from .stage41_projected_polar_operator_audit import (
    STAGE41_CORRECTION_FLOOR,
    mapped_polar_quadrature,
    projected_macroscopic,
    projected_shakhov_equilibrium,
)
from . import stage42_projected_polar_heated_cavity_pilot as stage42
from .stage58_conservative_solver_64x64_confirmation import (
    STAGE58_CHECK_INTERVAL,
    STAGE58_GRID,
    STAGE58_KNUDSEN,
    STAGE58_MAX_ITERATIONS,
    STAGE58_MINIMUM_ITERATIONS,
    STAGE58_RADIAL_SCALE,
    STAGE58_RATIO,
    STAGE58_RULE,
    STAGE58_SOURCE_RELAXATION,
    STAGE58_TOLERANCE,
    build_stage58_config,
    validate_stage58_design,
)
from .stage66_frozen_collision_source_observable_audit import (
    signed_summary,
    wall_distance_layers,
    wall_layer_stratification,
)

STAGE66_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30976588772,
    "workflow_job_id": 92211738864,
    "workflow_conclusion": "success",
    "tests_passed": 54,
    "tests_failed": 0,
    "test_duration_seconds": 0.47,
    "artifact_id": 8923406335,
    "artifact_size_bytes": 293758,
    "artifact_sha256": "226aa6fb853836ab305ce555ecb5227a68a1c59e72e368d4093496219850d15a",
    "source_head_sha": "ebd96666185a78653b5f6a2ba288a2c86e38d807",
    "summary_sha256": "71726050b1b41ad4d14616f7f78212156343edba6de670b716498a6cfb40fa68",
    "maps_sha256": "c4948595303ed14beb7df7edf70597d45e4f17b3f7993c042ef95e67e8eb69d0",
    "decision": (
        "stage66_clipping_source_bias_opposes_heat_flux_overprediction_"
        "stage67_frozen_full_distribution_residual_decomposition"
    ),
}
STAGE58_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30864287564,
    "workflow_job_id": 91852659651,
    "workflow_conclusion": "success",
    "tests_passed": 72,
    "tests_failed": 0,
    "artifact_id": 8882191150,
    "artifact_size_bytes": 265675,
    "artifact_sha256": "c1e1cb40d35439d44a93bae091f732f35e1790c2c4e3f1180b16d7f8fb54e8f6",
    "source_head_sha": "448ce586344052de1cf5dd0fd86e3ffb4b6a52be",
    "summary_sha256": "b91921a631bd92c2696c7bdd18668afb4994c909bbc7aa82cca1d0afc25ffced",
    "baseline_fields_sha256": "9aa53136e05917236f87fb9279c2ecc4e29d6056ca5ec34ca3bcb4d8f66aa822",
    "decision": (
        "stage58_conservative_confirmation_stable_but_observables_degrade_"
        "requires_review_without_retuning"
    ),
}

STAGE67_REPLAY_ABSOLUTE_TOLERANCE = 1.0e-11
STAGE67_REPLAY_RELATIVE_TOLERANCE = 1.0e-10
STAGE67_RESIDUAL_BALANCE_GUARD = 5.0e-2
STAGE67_CHUNK_SIZE = 128
STAGE67_WALL_BAND_LAYERS = 4


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage67_design(
    grid: tuple[int, int] = STAGE58_GRID,
    kn0: float = STAGE58_KNUDSEN,
    cold_hot_ratio: float = STAGE58_RATIO,
    rule: tuple[int, int] = STAGE58_RULE,
    radial_scale: float = STAGE58_RADIAL_SCALE,
    source_relaxation: float = STAGE58_SOURCE_RELAXATION,
    max_iterations: int = STAGE58_MAX_ITERATIONS,
    minimum_iterations: int = STAGE58_MINIMUM_ITERATIONS,
    check_interval: int = STAGE58_CHECK_INTERVAL,
    tolerance: float = STAGE58_TOLERANCE,
    correction_floor: float = STAGE41_CORRECTION_FLOOR,
) -> None:
    validate_stage58_design(
        grid=grid,
        kn0=kn0,
        cold_hot_ratio=cold_hot_ratio,
        rule=rule,
        radial_scale=radial_scale,
        source_relaxation=source_relaxation,
        max_iterations=max_iterations,
        minimum_iterations=minimum_iterations,
        check_interval=check_interval,
        tolerance=tolerance,
        correction_floor=correction_floor,
    )


def _validate_artifact(
    root: str | Path,
    stage: int,
    decision: str,
    files: Mapping[str, str],
) -> dict[str, object]:
    root = Path(root)
    for name, expected in files.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Stage {stage} artifact checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != stage or summary.get("decision") != decision:
        raise ValueError(f"Stage {stage} artifact endpoint mismatch")
    return summary


def replay_retained_baseline(cfg, quadrature) -> dict[str, object]:
    """Run the exact retained solver while capturing its final distributions."""
    captured: dict[str, np.ndarray] = {}
    original = stage42.steady_source_iteration_step

    def capture_step(phi, psi, config, rule, source_relaxation=1.0):
        next_phi, next_psi, clipping = original(
            phi, psi, config, rule, source_relaxation
        )
        captured["phi"] = next_phi
        captured["psi"] = next_psi
        return next_phi, next_psi, clipping

    stage42.steady_source_iteration_step = capture_step
    try:
        result = stage42.solve_stage42_pilot(
            cfg, quadrature, STAGE58_SOURCE_RELAXATION
        )
    finally:
        stage42.steady_source_iteration_step = original
    if "phi" not in captured or "psi" not in captured:
        raise RuntimeError("frozen replay did not execute a source iteration")
    phi = captured["phi"]
    psi = captured["psi"]
    result["phi"] = phi
    result["psi"] = psi
    result["fields_internal"] = projected_macroscopic(phi, psi, quadrature)
    result["incoming"] = stage42.projected_wall_incoming(phi, psi, cfg, quadrature)
    return result


def compare_replay_to_stage58(
    replay: Mapping[str, object], retained_path: str | Path
) -> dict[str, object]:
    keys = (
        "T", "rho", "u", "v", "qx", "qy", "left_wall_velocity",
        "table_velocity", "bottom_heat_flux", "residual_history",
    )
    per_field: dict[str, dict[str, float]] = {}
    with np.load(retained_path) as retained:
        if set(retained.files) != set(keys):
            raise ValueError("Stage 58 retained baseline field contract mismatch")
        for key in keys:
            actual = np.asarray(replay[key], dtype=np.float64)
            expected = np.asarray(retained[key], dtype=np.float64)
            if actual.shape != expected.shape:
                raise ValueError(f"Stage 58 replay shape mismatch: {key}")
            delta = actual - expected
            per_field[key] = {
                "maximum_absolute_error": float(np.max(np.abs(delta))),
                "relative_l2_error": float(
                    np.linalg.norm(delta.ravel())
                    / max(float(np.linalg.norm(expected.ravel())), 1.0e-300)
                ),
            }
    max_abs = max(row["maximum_absolute_error"] for row in per_field.values())
    max_rel = max(row["relative_l2_error"] for row in per_field.values())
    return {
        "per_field": per_field,
        "maximum_absolute_error": max_abs,
        "maximum_relative_l2_error": max_rel,
        "within_frozen_tolerance": bool(
            max_abs <= STAGE67_REPLAY_ABSOLUTE_TOLERANCE
            and max_rel <= STAGE67_REPLAY_RELATIVE_TOLERANCE
        ),
    }


def split_upwind_transport_chunk(
    distribution: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate interior-neighbor and diffuse-wall terms in the retained stencil."""
    distribution = np.asarray(distribution, dtype=np.float64)
    interior = np.zeros_like(distribution)
    wall = np.zeros_like(distribution)
    for k, (vx_k, vy_k) in enumerate(zip(vx, vy, strict=True)):
        ax = abs(float(vx_k)) / dx
        ay = abs(float(vy_k)) / dy
        if vx_k > 0.0:
            interior[:, 1:, k] += ax * (
                distribution[:, :-1, k] - distribution[:, 1:, k]
            )
            wall[:, 0, k] += ax * (left[:, k] - distribution[:, 0, k])
        elif vx_k < 0.0:
            interior[:, :-1, k] += ax * (
                distribution[:, 1:, k] - distribution[:, :-1, k]
            )
            wall[:, -1, k] += ax * (right[:, k] - distribution[:, -1, k])
        if vy_k > 0.0:
            interior[1:, :, k] += ay * (
                distribution[:-1, :, k] - distribution[1:, :, k]
            )
            wall[0, :, k] += ay * (bottom[:, k] - distribution[0, :, k])
        elif vy_k < 0.0:
            interior[:-1, :, k] += ay * (
                distribution[1:, :, k] - distribution[:-1, :, k]
            )
            wall[-1, :, k] += ay * (top[:, k] - distribution[-1, :, k])
    return interior, wall


def _moment_maps(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {
        name: np.zeros(shape, dtype=np.float64)
        for name in ("mass", "momentum_x", "momentum_y", "energy", "qx", "qy")
    }


def _accumulate(
    output: dict[str, np.ndarray],
    residual_phi: np.ndarray,
    residual_psi: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    weight: np.ndarray,
    local_u: np.ndarray,
    local_v: np.ndarray,
) -> None:
    vx3 = vx[None, None, :]
    vy3 = vy[None, None, :]
    w3 = weight[None, None, :]
    cx = vx3 - local_u[..., None]
    cy = vy3 - local_v[..., None]
    absolute_energy = (vx3 * vx3 + vy3 * vy3) * residual_phi + residual_psi
    frozen_energy = (cx * cx + cy * cy) * residual_phi + residual_psi
    output["mass"] += np.sum(residual_phi * w3, axis=-1)
    output["momentum_x"] += np.sum(vx3 * residual_phi * w3, axis=-1)
    output["momentum_y"] += np.sum(vy3 * residual_phi * w3, axis=-1)
    output["energy"] += 0.5 * np.sum(absolute_energy * w3, axis=-1)
    output["qx"] += 0.5 * np.sum(cx * frozen_energy * w3, axis=-1)
    output["qy"] += 0.5 * np.sum(cy * frozen_energy * w3, axis=-1)


def decompose_steady_residual(replay, cfg, quadrature):
    phi = np.asarray(replay["phi"], dtype=np.float64)
    psi = np.asarray(replay["psi"], dtype=np.float64)
    fields = replay["fields_internal"]
    incoming = replay["incoming"]
    equilibrium_phi, equilibrium_psi, _ = projected_shakhov_equilibrium(
        fields, quadrature, prandtl=cfg.prandtl,
        correction_floor=STAGE41_CORRECTION_FLOOR,
    )
    frequency = 1.0 / np.maximum(
        local_relaxation_time(
            fields["rho"], fields["T"], cfg, mapping="paper_consistent_c0"
        ),
        1.0e-14,
    )
    components = {
        name: _moment_maps(phi.shape[:2])
        for name in ("collision", "interior_transport", "diffuse_wall", "total")
    }
    (
        left_phi, left_psi, right_phi, right_psi,
        bottom_phi, bottom_psi, top_phi, top_psi,
    ) = incoming
    local_u = np.asarray(fields["u"], dtype=np.float64)
    local_v = np.asarray(fields["v"], dtype=np.float64)
    dx, dy = 1.0 / cfg.nx, 1.0 / cfg.ny

    for start in range(0, quadrature.point_count, STAGE67_CHUNK_SIZE):
        stop = min(start + STAGE67_CHUNK_SIZE, quadrature.point_count)
        sl = slice(start, stop)
        vx, vy, weight = quadrature.vx[sl], quadrature.vy[sl], quadrature.weight[sl]
        p, s = phi[..., sl], psi[..., sl]
        collision_phi = frequency[..., None] * (equilibrium_phi[..., sl] - p)
        collision_psi = frequency[..., None] * (equilibrium_psi[..., sl] - s)
        interior_phi, wall_phi = split_upwind_transport_chunk(
            p, left_phi[..., sl], right_phi[..., sl], bottom_phi[..., sl],
            top_phi[..., sl], vx, vy, dx, dy,
        )
        interior_psi, wall_psi = split_upwind_transport_chunk(
            s, left_psi[..., sl], right_psi[..., sl], bottom_psi[..., sl],
            top_psi[..., sl], vx, vy, dx, dy,
        )
        pairs = {
            "collision": (collision_phi, collision_psi),
            "interior_transport": (interior_phi, interior_psi),
            "diffuse_wall": (wall_phi, wall_psi),
            "total": (
                collision_phi + interior_phi + wall_phi,
                collision_psi + interior_psi + wall_psi,
            ),
        }
        for name, (rphi, rpsi) in pairs.items():
            _accumulate(
                components[name], rphi, rpsi, vx, vy, weight, local_u, local_v
            )
    return components


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values)))


def summarize_components(components, stage66_maps_path: str | Path):
    summary = {
        component: {moment: signed_summary(values) for moment, values in maps.items()}
        for component, maps in components.items()
    }
    component_names = ("collision", "interior_transport", "diffuse_wall")
    q_denominator = sum(_rms(components[name]["qy"]) for name in component_names)
    conserved = ("mass", "momentum_x", "momentum_y", "energy")
    c_denominator = sum(
        _rms(components[name][moment])
        for name in component_names for moment in conserved
    )
    total_conserved = sum(_rms(components["total"][moment]) for moment in conserved)
    qy_rms = {name: _rms(components[name]["qy"]) for name in component_names}
    with np.load(stage66_maps_path) as maps:
        source_bias = np.asarray(maps["source_bias_qy_paper"], dtype=np.float64)
    collision_qy = np.asarray(components["collision"]["qy"], dtype=np.float64)
    correlation = float(np.corrcoef(source_bias.ravel(), collision_qy.ravel())[0, 1])
    distance = wall_distance_layers(collision_qy.shape)
    summary["balance"] = {
        "normal_heat_flux_residual_ratio": _rms(components["total"]["qy"])
        / max(q_denominator, 1.0e-300),
        "conserved_residual_ratio": total_conserved / max(c_denominator, 1.0e-300),
        "normal_component_rms": qy_rms,
        "dominant_normal_component": max(qy_rms, key=qy_rms.get),
        "stage66_source_bias_vs_active_collision_qy_correlation": correlation,
        "collision_qy_wall_distance": wall_layer_stratification(
            collision_qy, distance, STAGE67_WALL_BAND_LAYERS
        ),
        "diffuse_wall_qy_wall_distance": wall_layer_stratification(
            components["diffuse_wall"]["qy"], distance, STAGE67_WALL_BAND_LAYERS
        ),
    }
    return summary


def stage67_decision(metrics: Mapping[str, object]) -> str:
    if not bool(metrics.get("finite", False)):
        return "stage67_nonfinite_full_distribution_replay_blocker"
    if not bool(metrics.get("converged", False)):
        return "stage67_frozen_replay_nonconverged_blocker_without_retuning"
    if not bool(metrics.get("replay_within_tolerance", False)):
        return "stage67_stage58_replay_mismatch_blocker"
    if max(
        float(metrics.get("normal_heat_flux_residual_ratio", math.inf)),
        float(metrics.get("conserved_residual_ratio", math.inf)),
    ) > STAGE67_RESIDUAL_BALANCE_GUARD:
        return "stage67_distribution_fixed_point_residual_blocker"
    return (
        "stage67_frozen_replay_and_residual_balance_close_"
        "stage68_independent_transport_operator_residual_audit"
    )


def run_stage67(
    stage66_artifact_dir: str | Path,
    stage58_artifact_dir: str | Path,
    output_dir: str | Path,
    **design,
) -> dict[str, object]:
    validate_stage67_design(**design)
    stage66 = _validate_artifact(
        stage66_artifact_dir,
        66,
        STAGE66_COMPLETED_ENDPOINT["decision"],
        {
            "summary.json": STAGE66_COMPLETED_ENDPOINT["summary_sha256"],
            "source_observable_maps.npz": STAGE66_COMPLETED_ENDPOINT["maps_sha256"],
        },
    )
    stage58 = _validate_artifact(
        stage58_artifact_dir,
        58,
        STAGE58_COMPLETED_ENDPOINT["decision"],
        {
            "summary.json": STAGE58_COMPLETED_ENDPOINT["summary_sha256"],
            "baseline_clipped_fields_and_profiles.npz": (
                STAGE58_COMPLETED_ENDPOINT["baseline_fields_sha256"]
            ),
        },
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = build_stage58_config()
    quadrature = mapped_polar_quadrature(
        *STAGE58_RULE, radial_scale=STAGE58_RADIAL_SCALE
    )
    replay = replay_retained_baseline(cfg, quadrature)
    replay_comparison = compare_replay_to_stage58(
        replay,
        Path(stage58_artifact_dir) / "baseline_clipped_fields_and_profiles.npz",
    )
    components = decompose_steady_residual(replay, cfg, quadrature)
    component_summary = summarize_components(
        components, Path(stage66_artifact_dir) / "source_observable_maps.npz"
    )
    balance = component_summary["balance"]
    metrics = {
        "finite": bool(replay["finite"]) and all(
            np.isfinite(values).all()
            for maps in components.values() for values in maps.values()
        ),
        "converged": bool(replay["converged"]),
        "replay_within_tolerance": bool(replay_comparison["within_frozen_tolerance"]),
        "normal_heat_flux_residual_ratio": float(
            balance["normal_heat_flux_residual_ratio"]
        ),
        "conserved_residual_ratio": float(balance["conserved_residual_ratio"]),
    }
    decision = stage67_decision(metrics)

    np.savez_compressed(
        out / "converged_full_distributions.npz",
        phi=np.asarray(replay["phi"], dtype=np.float64),
        psi=np.asarray(replay["psi"], dtype=np.float64),
        vx=np.asarray(quadrature.vx, dtype=np.float64),
        vy=np.asarray(quadrature.vy, dtype=np.float64),
        weight=np.asarray(quadrature.weight, dtype=np.float64),
    )
    np.savez_compressed(
        out / "steady_residual_moment_maps.npz",
        **{
            f"{component}_{moment}": np.asarray(values, dtype=np.float64)
            for component, maps in components.items()
            for moment, values in maps.items()
        },
    )

    gap = stage66["audit"]["observed_heat_flux_gap"]
    summary = {
        "stage": 67,
        "description": (
            "One exact frozen replay of the retained Stage-58 64x64 Kn0=10 clipped "
            "projected-Shakhov arm, saving phi/psi and decomposing its local discrete "
            "steady residual into collision, interior-upwind and diffuse-wall parts."
        ),
        "retained_stage66_endpoint": STAGE66_COMPLETED_ENDPOINT,
        "retained_stage66_decision": stage66["decision"],
        "retained_stage58_endpoint": STAGE58_COMPLETED_ENDPOINT,
        "retained_stage58_decision": stage58["decision"],
        "configuration": {
            "grid": list(STAGE58_GRID),
            "kn0": STAGE58_KNUDSEN,
            "cold_hot_ratio": STAGE58_RATIO,
            "radial_nodes": STAGE58_RULE[0],
            "angular_nodes": STAGE58_RULE[1],
            "point_count": quadrature.point_count,
            "radial_scale": STAGE58_RADIAL_SCALE,
            "retained_correction_floor": STAGE41_CORRECTION_FLOOR,
            "source_relaxation": STAGE58_SOURCE_RELAXATION,
            "max_iterations": STAGE58_MAX_ITERATIONS,
            "minimum_iterations": STAGE58_MINIMUM_ITERATIONS,
            "check_interval": STAGE58_CHECK_INTERVAL,
            "tolerance": STAGE58_TOLERANCE,
            "replay_absolute_tolerance": STAGE67_REPLAY_ABSOLUTE_TOLERANCE,
            "replay_relative_tolerance": STAGE67_REPLAY_RELATIVE_TOLERANCE,
            "residual_balance_guard": STAGE67_RESIDUAL_BALANCE_GUARD,
            "chunk_size": STAGE67_CHUNK_SIZE,
            "wall_band_layers": STAGE67_WALL_BAND_LAYERS,
            "solver_replay_count": 1,
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "transport_retuning": False,
            "wall_model_retuning": False,
            "normalization_retuning": False,
            "correction_floor_retuning": False,
            "source_relaxation_retuning": False,
            "velocity_quadrature_retuning": False,
            "stopping_rule_retuning": False,
            "conservative_projection_adopted": False,
            "failed_muscl_endpoint_rehabilitated": False,
            "cross_knudsen_extension_permitted": False,
        },
        "replay": {
            "iterations": int(replay["iterations"]),
            "converged": bool(replay["converged"]),
            "final_change": float(replay["final_change"]),
            "finite": bool(replay["finite"]),
            "predicted_qav": float(replay["predicted_qav"]),
            "published_dsmc_qav": float(gap["published_dsmc_qav"]),
            "signed_qav_excess": float(replay["predicted_qav"])
            - float(gap["published_dsmc_qav"]),
            "relative_qav_excess": (
                float(replay["predicted_qav"]) - float(gap["published_dsmc_qav"])
            ) / float(gap["published_dsmc_qav"]),
            "wall_mass_balance_relative_error": float(
                replay["wall_mass_balance_relative_error"]
            ),
            "minimum_phi": float(replay["minimum_phi"]),
            "minimum_psi": float(replay["minimum_psi"]),
            "maximum_phi_clipped_weight_fraction": float(
                replay["maximum_phi_clipped_weight_fraction"]
            ),
            "maximum_psi_clipped_weight_fraction": float(
                replay["maximum_psi_clipped_weight_fraction"]
            ),
            "stage58_field_comparison": replay_comparison,
        },
        "residual_decomposition": component_summary,
        "decision_metrics": metrics,
        "decision": decision,
        "positive_findings": [
            "The exact frozen replay preserves the retained Stage-58 algorithm and writes the previously unavailable converged full phi/psi distributions.",
            "Collision, interior-upwind and diffuse-wall residual moments are retained separately as signed local maps.",
            "Replay integrity is checked against every saved Stage-58 baseline field, wall profile and residual-history array before interpretation.",
        ],
        "negative_findings": [
            "The replay intentionally reproduces the previously confirmed positive heat-flux discrepancy; reproducing it is not validation.",
            "Opposing residual components may cancel at the discrete fixed point, so component magnitude or sign alone is not a causal sensitivity of the converged wall heat flux.",
            "The audit does not rehabilitate MUSCL, adopt the conservative projection, justify parameter retuning or authorize cross-Knudsen extension.",
        ],
        "interpretation_guard": (
            "Residual moments are evaluated in the converged local velocity frame and "
            "their algebraic sum is the retained discrete steady equation. They are not "
            "adjoint sensitivities, perturbation responses or proof that any one term "
            "causes the published-reference heat-flux discrepancy."
        ),
        "scientifically_justified_next_scope": (
            "If replay integrity and residual balance close, evaluate an independent "
            "second-order control-volume transport operator on the saved frozen phi/psi "
            "fields without solving or tuning. This tests whether finite-Kn transport-"
            "collision coupling leaves a material higher-order transport residual while "
            "preserving all current physics, walls, quadrature and collision settings."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage66-artifact-dir", required=True)
    parser.add_argument("--stage58-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage67(
        args.stage66_artifact_dir, args.stage58_artifact_dir, args.output_dir
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
