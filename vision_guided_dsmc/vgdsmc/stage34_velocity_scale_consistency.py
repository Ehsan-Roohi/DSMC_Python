from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import numpy as np

from .linear_sidewall_validation import (
    LinearSidewallConfig,
    TABLE3_UY_RATIO_0P1,
    TABLE6_QAV_RATIO_0P1,
    sidewall_temperature_profile,
)
from .reduced_spherical_solver import (
    bottom_heat_flux,
    discrete_maxwellian,
    left_wall_tangential_velocity,
    macroscopic,
    shakhov_equilibrium,
    wall_incoming,
    wall_mass_balance_error,
)
from .stage32_near_continuum_observable_audit import (
    observable_metrics,
    wall_observable_profiles,
)
from .velocity_quadrature_audit import VelocityQuadrature, spherical_product


STAGE34_KNUDSEN = (0.1, 1.0, 10.0)
STAGE34_GRID = (12, 12)
STAGE34_RATIO = 0.1
STAGE34_QUADRATURE = "spherical_matched_r16_mu12_phi24"

# Exact spherical results previously obtained with the legacy c0-scale mapping
# tau = 2 Kn0/sqrt(pi) * T^(omega-1)/n. They are retained verbatim so Stage 34
# changes only the nondimensional velocity-scale conversion and does not rerun or
# silently replace the negative historical results.
LEGACY_SPHERICAL_BASELINES = {
    0.1: {
        "iterations": 1400,
        "converged": True,
        "final_change": 1.9574297504987292e-5,
        "predicted_qav": 0.09166984195087928,
        "literature_qav": 0.072,
        "qav_relative_error": 0.27319224931776787,
        "wall_velocity_relative_rms": 3.1935337629690843,
        "wall_velocity_sign_agreement": 0.2,
        "source_stage": 31,
    },
    1.0: {
        "iterations": 1700,
        "converged": True,
        "final_change": 2.699530439931319e-5,
        "predicted_qav": 0.16078706648046737,
        "literature_qav": 0.148,
        "qav_relative_error": 0.08639909784099577,
        "wall_velocity_relative_rms": 0.2733045144741894,
        "wall_velocity_sign_agreement": 1.0,
        "source_stage": 30,
    },
    10.0: {
        "iterations": 4500,
        "converged": True,
        "final_change": 2.7755827372466513e-5,
        "predicted_qav": 0.18468389772143382,
        "literature_qav": 0.178,
        "qav_relative_error": 0.03754998719906642,
        "wall_velocity_relative_rms": 0.14467929367009996,
        "wall_velocity_sign_agreement": 1.0,
        "source_stage": 31,
    },
}


def paper_zeta_tau_prefactor(kn0: float) -> float:
    """Return the paper's tau prefactor in zeta=xi/sqrt(2 k T0/m) units.

    From Eq. (6) of Vargas et al., Phys. Fluids 26, 057101 (2014),
    zeta.grad(g) = sqrt(pi)/(2 Kn0) n T^(1-omega) (gS-g).
    """
    if kn0 <= 0.0:
        raise ValueError("kn0 must be positive")
    return 2.0 * float(kn0) / math.sqrt(math.pi)


def paper_consistent_c0_tau_prefactor(kn0: float) -> float:
    """Convert the paper relaxation time to c0=sqrt(k T0/m) velocity units.

    The reduced solver uses c=sqrt(2)*zeta. Therefore c.grad(g) has a collision
    coefficient sqrt(2) larger than the paper zeta equation and its relaxation
    time is smaller by sqrt(2): tau_c0=sqrt(2) Kn0/sqrt(pi).
    """
    return paper_zeta_tau_prefactor(kn0) / math.sqrt(2.0)


def legacy_c0_tau_prefactor(kn0: float) -> float:
    """Return the historical mapping retained only for reproducibility."""
    return paper_zeta_tau_prefactor(kn0)


def local_relaxation_time(
    density: np.ndarray,
    temperature: np.ndarray,
    cfg: LinearSidewallConfig,
    mapping: str = "paper_consistent_c0",
) -> np.ndarray:
    density = np.maximum(np.asarray(density, dtype=np.float64), 1.0e-12)
    temperature = np.maximum(np.asarray(temperature, dtype=np.float64), 1.0e-10)
    if mapping == "paper_consistent_c0":
        prefactor = paper_consistent_c0_tau_prefactor(cfg.kn0)
    elif mapping == "legacy_c0":
        prefactor = legacy_c0_tau_prefactor(cfg.kn0)
    else:
        raise ValueError("unknown relaxation mapping")
    return prefactor * temperature ** (cfg.viscosity_exponent - 1.0) / density


def validate_stage34_design(
    knudsen_numbers: tuple[float, ...],
    grid: tuple[int, int],
    max_steps: int,
    tolerance: float,
) -> None:
    if tuple(float(value) for value in knudsen_numbers) != STAGE34_KNUDSEN:
        raise ValueError("Stage 34 is fixed to Kn0=(0.1,1,10)")
    if grid != STAGE34_GRID:
        raise ValueError("Stage 34 is fixed to a 12x12 spatial grid")
    if max_steps <= 0 or tolerance <= 0.0:
        raise ValueError("max_steps and tolerance must be positive")


def solve_reduced_case_with_mapping(
    cfg: LinearSidewallConfig,
    quadrature: VelocityQuadrature,
    mapping: str = "paper_consistent_c0",
) -> dict[str, object]:
    """Solve the reduced spherical Shakhov cavity with an explicit tau mapping."""
    if cfg.nx < 3 or cfg.ny < 3:
        raise ValueError("nx and ny must be at least three")
    if cfg.kn0 <= 0.0 or not 0.0 < cfg.cold_hot_ratio < 1.0:
        raise ValueError("invalid physical configuration")

    rho = np.ones((cfg.ny, cfg.nx))
    zero = np.zeros_like(rho)
    initial_temperature = np.repeat(
        sidewall_temperature_profile(cfg)[:, None], cfg.nx, axis=1
    )
    distribution = discrete_maxwellian(
        rho, zero, zero, zero, initial_temperature, quadrature
    )
    dx, dy = 1.0 / cfg.nx, 1.0 / cfg.ny
    maximum_speed = max(
        float(np.max(np.abs(quadrature.vx))),
        float(np.max(np.abs(quadrature.vy))),
    )
    dt = cfg.cfl * min(dx / maximum_speed, dy / maximum_speed)
    positive_x = (quadrature.vx > 0.0)[None, None, :]
    positive_y = (quadrature.vy > 0.0)[None, None, :]
    previous = macroscopic(distribution, quadrature)
    previous_temperature = previous["T"].copy()
    previous_velocity = np.stack([previous["u"], previous["v"]], axis=-1)
    previous_heat_flux = np.stack([previous["qx"], previous["qy"]], axis=-1)
    residual_history: list[float] = []
    converged = False

    for step in range(cfg.max_steps):
        left, right, bottom, top = wall_incoming(distribution, cfg, quadrature)
        left_neighbor = np.empty_like(distribution)
        right_neighbor = np.empty_like(distribution)
        bottom_neighbor = np.empty_like(distribution)
        top_neighbor = np.empty_like(distribution)
        left_neighbor[:, 1:] = distribution[:, :-1]
        left_neighbor[:, 0] = left
        right_neighbor[:, :-1] = distribution[:, 1:]
        right_neighbor[:, -1] = right
        bottom_neighbor[1:] = distribution[:-1]
        bottom_neighbor[0] = bottom
        top_neighbor[:-1] = distribution[1:]
        top_neighbor[-1] = top
        dfdx = np.where(
            positive_x,
            (distribution - left_neighbor) / dx,
            (right_neighbor - distribution) / dx,
        )
        dfdy = np.where(
            positive_y,
            (distribution - bottom_neighbor) / dy,
            (top_neighbor - distribution) / dy,
        )
        transported = np.maximum(
            distribution
            - dt
            * (
                quadrature.vx[None, None, :] * dfdx
                + quadrature.vy[None, None, :] * dfdy
            ),
            cfg.positivity_floor,
        )
        fields = macroscopic(transported, quadrature)
        equilibrium = shakhov_equilibrium(fields, quadrature, cfg.prandtl)
        tau = local_relaxation_time(fields["rho"], fields["T"], cfg, mapping)
        fraction = np.minimum(dt / tau, 1.0)[..., None]
        distribution = np.maximum(
            transported + fraction * (equilibrium - transported),
            cfg.positivity_floor,
        )
        if (step + 1) % cfg.check_interval == 0:
            fields = macroscopic(distribution, quadrature)
            velocity = np.stack([fields["u"], fields["v"]], axis=-1)
            heat_flux = np.stack([fields["qx"], fields["qy"]], axis=-1)
            change = max(
                float(np.max(np.abs(fields["T"] - previous_temperature))),
                float(np.max(np.abs(velocity - previous_velocity))),
                float(np.max(np.abs(heat_flux - previous_heat_flux))),
            )
            residual_history.append(change)
            previous_temperature = fields["T"].copy()
            previous_velocity = velocity.copy()
            previous_heat_flux = heat_flux.copy()
            if step + 1 >= cfg.minimum_steps and change < cfg.tolerance:
                converged = True
                break

    left, right, bottom, top = wall_incoming(distribution, cfg, quadrature)
    fields = macroscopic(distribution, quadrature)
    wall_velocity = left_wall_tangential_velocity(distribution, left, quadrature)
    y_centers = (np.arange(cfg.ny, dtype=np.float64) + 0.5) / cfg.ny
    from .linear_sidewall_validation import TABLE3_Y

    table_velocity = np.interp(TABLE3_Y, y_centers, wall_velocity)
    wall_balance = max(
        wall_mass_balance_error(
            distribution[:, 0], left, quadrature.vx, quadrature.vx > 0.0, quadrature
        ),
        wall_mass_balance_error(
            distribution[:, -1], right, -quadrature.vx, quadrature.vx < 0.0, quadrature
        ),
        wall_mass_balance_error(
            distribution[0], bottom, quadrature.vy, quadrature.vy > 0.0, quadrature
        ),
        wall_mass_balance_error(
            distribution[-1], top, -quadrature.vy, quadrature.vy < 0.0, quadrature
        ),
    )
    return {
        "T": np.asarray(fields["T"]),
        "rho": np.asarray(fields["rho"]) / np.mean(fields["rho"]),
        "u": np.asarray(fields["u"]) / math.sqrt(2.0),
        "v": np.asarray(fields["v"]) / math.sqrt(2.0),
        "qx": np.asarray(fields["qx"]) / math.sqrt(2.0),
        "qy": np.asarray(fields["qy"]) / math.sqrt(2.0),
        "left_wall_velocity": wall_velocity,
        "table_velocity": table_velocity,
        "bottom_heat_flux": bottom_heat_flux(distribution, bottom, quadrature),
        "residual_history": np.asarray(residual_history),
        "iterations": step + 1,
        "converged": converged,
        "dt": dt,
        "wall_mass_balance_relative_error": wall_balance,
        "minimum_distribution": float(np.min(distribution)),
    }


def corrected_case_metrics(
    result: dict[str, object],
    cfg: LinearSidewallConfig,
    quadrature: VelocityQuadrature,
) -> dict[str, object]:
    reference_velocity = TABLE3_UY_RATIO_0P1[cfg.kn0]
    reference_qav = TABLE6_QAV_RATIO_0P1[cfg.kn0]
    profiles = wall_observable_profiles(result, cfg)
    qav = float(np.mean(np.asarray(result["bottom_heat_flux"], dtype=np.float64)))
    residual = np.asarray(result["residual_history"], dtype=np.float64)
    return {
        "kn0": cfg.kn0,
        "mapping": "paper_consistent_c0",
        "point_count": quadrature.point_count,
        "iterations": int(result["iterations"]),
        "converged": bool(result["converged"]),
        "final_change": float(residual[-1]) if residual.size else float("nan"),
        "predicted_qav": qav,
        "literature_qav": reference_qav,
        "qav_relative_error": abs(qav - reference_qav) / reference_qav,
        "observable_metrics": {
            name: observable_metrics(profile, reference_velocity)
            for name, profile in profiles.items()
        },
        "wall_mass_balance_relative_error": float(
            result["wall_mass_balance_relative_error"]
        ),
        "minimum_distribution": float(result["minimum_distribution"]),
        "minimum_temperature": float(np.min(np.asarray(result["T"]))),
        "maximum_temperature": float(np.max(np.asarray(result["T"]))),
        "work_proxy": int(result["iterations"])
        * cfg.nx
        * cfg.ny
        * quadrature.point_count,
    }


def stage34_decision(comparisons: list[dict[str, object]]) -> str:
    materially_better = 0
    materially_worse = 0
    for comparison in comparisons:
        q_ratio = float(comparison["qav_error_ratio_corrected_to_legacy"])
        v_ratio = float(comparison["boundary_velocity_error_ratio_corrected_to_legacy"])
        sign_change = float(comparison["boundary_sign_agreement_change"])
        if q_ratio <= 0.90 and v_ratio <= 0.90 and sign_change >= 0.0:
            materially_better += 1
        if q_ratio >= 1.25 or v_ratio >= 1.25 or sign_change < -0.2:
            materially_worse += 1
    if materially_better >= 2 and materially_worse == 0:
        return "adopt_paper_consistent_velocity_scale_and_repeat_high_resolution_endpoint"
    if materially_worse >= 2:
        return "paper_consistent_mapping_is_required_but_exposes_additional_implementation_discrepancy"
    return "mixed_cross_kn_effect_audit_reduced_equations_and_benchmark_observables"


def run_stage34(
    output_dir: str | Path,
    *,
    knudsen_numbers: tuple[float, ...] = STAGE34_KNUDSEN,
    grid: tuple[int, int] = STAGE34_GRID,
    max_steps: int = 9000,
    tolerance: float = 3.0e-5,
) -> dict[str, object]:
    validate_stage34_design(knudsen_numbers, grid, max_steps, tolerance)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quadrature = spherical_product(16, 12, 24, 5.0, STAGE34_QUADRATURE)
    rows: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}

    for kn0 in knudsen_numbers:
        cfg = LinearSidewallConfig(
            nx=grid[0],
            ny=grid[1],
            nv=19,
            velocity_extent=5.0,
            kn0=kn0,
            cold_hot_ratio=STAGE34_RATIO,
            max_steps=max_steps,
            tolerance=tolerance,
            check_interval=100,
            minimum_steps=1200,
        )
        result = solve_reduced_case_with_mapping(cfg, quadrature)
        metrics = corrected_case_metrics(result, cfg, quadrature)
        rows.append(metrics)
        legacy = LEGACY_SPHERICAL_BASELINES[kn0]
        boundary = metrics["observable_metrics"]["boundary_mixture"]
        comparison = {
            "kn0": kn0,
            "legacy_source_stage": legacy["source_stage"],
            "legacy_qav_relative_error": legacy["qav_relative_error"],
            "corrected_qav_relative_error": metrics["qav_relative_error"],
            "qav_error_ratio_corrected_to_legacy": (
                metrics["qav_relative_error"] / legacy["qav_relative_error"]
            ),
            "legacy_boundary_velocity_relative_rms": legacy[
                "wall_velocity_relative_rms"
            ],
            "corrected_boundary_velocity_relative_rms": boundary["relative_rms"],
            "boundary_velocity_error_ratio_corrected_to_legacy": (
                boundary["relative_rms"] / legacy["wall_velocity_relative_rms"]
            ),
            "legacy_boundary_sign_agreement": legacy[
                "wall_velocity_sign_agreement"
            ],
            "corrected_boundary_sign_agreement": boundary["sign_agreement"],
            "boundary_sign_agreement_change": (
                boundary["sign_agreement"]
                - legacy["wall_velocity_sign_agreement"]
            ),
        }
        comparisons.append(comparison)
        key = str(kn0).replace(".", "p")
        for name in (
            "T",
            "rho",
            "u",
            "v",
            "qx",
            "qy",
            "bottom_heat_flux",
            "residual_history",
        ):
            arrays[f"{name}_kn{key}"] = np.asarray(result[name])
        for name, profile in wall_observable_profiles(result, cfg).items():
            arrays[f"table_velocity_{name}_kn{key}"] = profile

    summary = {
        "stage": 34,
        "description": (
            "Exact nondimensional velocity-scale consistency audit of the "
            "spherical reduced Shakhov cavity"
        ),
        "configuration": {
            "grid": list(grid),
            "kn0_sequence": list(knudsen_numbers),
            "cold_hot_ratio": STAGE34_RATIO,
            "quadrature": STAGE34_QUADRATURE,
            "velocity_point_count": quadrature.point_count,
            "transport": "first-order upwind",
            "collision": "Shakhov",
            "max_steps": max_steps,
            "tolerance": tolerance,
            "legacy_solver_velocity_scale": "c0=sqrt(k*T0/m)",
            "paper_velocity_scale": "v0=sqrt(2*k*T0/m)",
            "legacy_tau_prefactor": "2*Kn0/sqrt(pi)",
            "paper_consistent_c0_tau_prefactor": "sqrt(2)*Kn0/sqrt(pi)",
            "physical_parameter_retuning": False,
        },
        "legacy_spherical_baselines": LEGACY_SPHERICAL_BASELINES,
        "paper_consistent_rows": rows,
        "comparisons": comparisons,
        "all_corrected_cases_converged": all(bool(row["converged"]) for row in rows),
        "decision": stage34_decision(comparisons),
        "interpretation_guard": (
            "Stage 34 does not fit or retune Kn0. It applies the algebraic change "
            "of molecular-velocity units between the paper's zeta scale and the "
            "solver's c0 scale. Historical legacy metrics remain explicit."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(output_dir / "fields_and_profiles.npz", **arrays)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 34 exact velocity-scale consistency audit"
    )
    parser.add_argument("--output-dir", default="outputs/stage34_velocity_scale")
    parser.add_argument("--max-steps", type=int, default=9000)
    parser.add_argument("--tolerance", type=float, default=3.0e-5)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage34(
                args.output_dir,
                max_steps=args.max_steps,
                tolerance=args.tolerance,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
