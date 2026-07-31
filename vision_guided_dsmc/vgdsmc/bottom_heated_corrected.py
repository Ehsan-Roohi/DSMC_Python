from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import numpy as np

from .bottom_heated_benchmark import (
    _bottom_boundary_heat_flux,
    _cold_wall_incoming,
    bottom_wall_temperature_profile,
    paper_kn0_to_solver_relaxation_scale,
)
from .dvm_shakhov import ShakhovReferenceConfig, _macroscopic, _velocity_grid
from .dvm_shakhov_corrected import (
    _discrete_maxwellian,
    _shakhov_equilibrium,
    _unit_wall_maxwellian,
)


@dataclass(frozen=True)
class CorrectedBottomHeatedConfig:
    """Corrected bottom-heated benchmark configuration for Stage 25.

    The hot-wall temperature is the reference temperature, so TH/T0=1 and
    TC/T0=TC/TH. An odd Cartesian velocity count is mandatory so that zero
    velocity is present and low wall temperatures remain representable.
    """

    nx: int = 18
    ny: int = 18
    nv: int = 17
    velocity_extent: float = 5.0
    kn0: float = 0.10
    cold_hot_ratio: float = 0.10
    viscosity_exponent: float = 0.5
    corner_ramp_fraction: float = 0.05
    prandtl: float = 2.0 / 3.0
    max_steps: int = 4200
    cfl: float = 0.30
    tolerance: float = 2.0e-5
    check_interval: int = 100
    minimum_steps: int = 1200
    positivity_floor: float = 1.0e-30

    @property
    def hot_temperature(self) -> float:
        return 1.0

    @property
    def cold_temperature(self) -> float:
        return self.cold_hot_ratio


def validate_corrected_config(cfg: CorrectedBottomHeatedConfig) -> None:
    if cfg.nx < 3 or cfg.ny < 3 or cfg.nv < 7:
        raise ValueError("require nx, ny >= 3 and nv >= 7")
    if cfg.nv % 2 == 0:
        raise ValueError("Stage 25 requires an odd nv so the grid contains zero velocity")
    if not 0.0 < cfg.cold_hot_ratio < 1.0:
        raise ValueError("cold_hot_ratio must lie between zero and one")
    if cfg.viscosity_exponent != 0.5:
        raise ValueError("Stage 25 is fixed to the hard-sphere omega=1/2 benchmark")
    if cfg.max_steps <= 0 or cfg.check_interval <= 0:
        raise ValueError("time-stepping parameters must be positive")


def _velocity_quadrature(cfg: CorrectedBottomHeatedConfig):
    velocity_cfg = ShakhovReferenceConfig(
        nx=cfg.nx,
        ny=cfg.ny,
        nv=cfg.nv,
        velocity_extent=cfg.velocity_extent,
    )
    return _velocity_grid(velocity_cfg)


def discrete_wall_temperature(
    target_temperature: float,
    cfg: CorrectedBottomHeatedConfig,
) -> float:
    """Return the actual second-moment temperature represented on the grid."""
    validate_corrected_config(cfg)
    vx, vy, vz, dv = _velocity_quadrature(cfg)
    wall = _unit_wall_maxwellian(target_temperature, vx, vy, vz, dv)
    fields = _macroscopic(wall[None, None], vx, vy, vz, dv)
    return float(fields["T"][0, 0])


def initial_temperature_field(cfg: CorrectedBottomHeatedConfig) -> np.ndarray:
    """Warm-start with a vertical conductive profile between hot and cold walls."""
    y = (np.arange(cfg.ny, dtype=np.float64) + 0.5) / cfg.ny
    profile = cfg.cold_temperature + (
        cfg.hot_temperature - cfg.cold_temperature
    ) * (1.0 - y)
    return np.repeat(profile[:, None], cfg.nx, axis=1)


def local_relaxation_time(
    density: np.ndarray,
    temperature: np.ndarray,
    cfg: CorrectedBottomHeatedConfig,
) -> np.ndarray:
    density = np.maximum(np.asarray(density, dtype=np.float64), 1.0e-12)
    temperature = np.maximum(np.asarray(temperature, dtype=np.float64), 1.0e-10)
    return (
        paper_kn0_to_solver_relaxation_scale(cfg.kn0)
        * temperature ** (cfg.viscosity_exponent - 1.0)
        / density
    )


def _profiled_wall_incoming(
    distribution: np.ndarray,
    cfg: CorrectedBottomHeatedConfig,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cold = _unit_wall_maxwellian(cfg.cold_temperature, vx, vy, vz, dv)
    positive_x, negative_x = vx > 0.0, vx < 0.0
    positive_y, negative_y = vy > 0.0, vy < 0.0

    left = _cold_wall_incoming(
        distribution[:, 0], vx, positive_x, negative_x, cold, dv
    )
    right = _cold_wall_incoming(
        distribution[:, -1], -vx, negative_x, positive_x, cold, dv
    )
    top = _cold_wall_incoming(
        distribution[-1], -vy, negative_y, positive_y, cold, dv
    )

    bottom_temperature = bottom_wall_temperature_profile(
        cfg.nx,
        cfg.cold_temperature,
        cfg.hot_temperature,
        cfg.corner_ramp_fraction,
    )
    bottom_wall = np.stack(
        [
            _unit_wall_maxwellian(float(temperature), vx, vy, vz, dv)
            for temperature in bottom_temperature
        ],
        axis=0,
    )
    measure = dv**3
    outgoing_flux = np.sum(
        vy[None] * distribution[0] * negative_y[None],
        axis=(-3, -2, -1),
    ) * measure
    incoming_unit = np.sum(
        vy[None] * bottom_wall * positive_y[None],
        axis=(-3, -2, -1),
    ) * measure
    scale = -outgoing_flux / np.maximum(incoming_unit, 1.0e-14)
    bottom = scale[:, None, None, None] * bottom_wall
    return left, right, bottom, top, bottom_temperature


def _fixed_scale_change(
    temperature: np.ndarray,
    previous_temperature: np.ndarray,
    velocity: np.ndarray,
    previous_velocity: np.ndarray,
    heat_flux: np.ndarray,
    previous_heat_flux: np.ndarray,
) -> tuple[float, float, float, float]:
    temperature_change = float(np.max(np.abs(temperature - previous_temperature)))
    velocity_change = float(np.max(np.abs(velocity - previous_velocity)))
    heat_flux_change = float(np.max(np.abs(heat_flux - previous_heat_flux)))
    combined = max(temperature_change, velocity_change, heat_flux_change)
    return temperature_change, velocity_change, heat_flux_change, combined


def solve_corrected_bottom_heated_case(
    cfg: CorrectedBottomHeatedConfig,
) -> dict[str, np.ndarray | float | int | bool]:
    validate_corrected_config(cfg)
    vx, vy, vz, dv = _velocity_quadrature(cfg)
    rho = np.ones((cfg.ny, cfg.nx))
    zero = np.zeros_like(rho)
    initial_temperature = initial_temperature_field(cfg)
    distribution = _discrete_maxwellian(
        rho,
        zero,
        zero,
        zero,
        initial_temperature,
        vx,
        vy,
        vz,
        dv,
    )

    dx, dy = 1.0 / cfg.nx, 1.0 / cfg.ny
    maximum_speed = max(float(np.max(np.abs(vx))), float(np.max(np.abs(vy))))
    dt = cfg.cfl * min(dx / maximum_speed, dy / maximum_speed)
    positive_x = (vx > 0.0)[None, None]
    positive_y = (vy > 0.0)[None, None]
    previous = _macroscopic(distribution, vx, vy, vz, dv)
    previous_temperature = previous["T"].copy()
    previous_velocity = np.stack([previous["u"], previous["v"]], axis=-1)
    previous_heat_flux = np.stack([previous["qx"], previous["qy"]], axis=-1)
    residual_history: list[float] = []
    component_history: list[tuple[float, float, float]] = []
    bottom_incoming = np.empty((cfg.nx, cfg.nv, cfg.nv, cfg.nv))
    bottom_temperature = np.empty(cfg.nx)
    converged = False

    for step in range(cfg.max_steps):
        left, right, bottom_incoming, top, bottom_temperature = _profiled_wall_incoming(
            distribution, cfg, vx, vy, vz, dv
        )
        left_neighbor = np.empty_like(distribution)
        right_neighbor = np.empty_like(distribution)
        bottom_neighbor = np.empty_like(distribution)
        top_neighbor = np.empty_like(distribution)
        left_neighbor[:, 1:] = distribution[:, :-1]
        left_neighbor[:, 0] = left
        right_neighbor[:, :-1] = distribution[:, 1:]
        right_neighbor[:, -1] = right
        bottom_neighbor[1:] = distribution[:-1]
        bottom_neighbor[0] = bottom_incoming
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
        transported = distribution - dt * (
            vx[None, None] * dfdx + vy[None, None] * dfdy
        )
        transported = np.maximum(transported, cfg.positivity_floor)
        fields = _macroscopic(transported, vx, vy, vz, dv)
        equilibrium = _shakhov_equilibrium(
            fields, vx, vy, vz, dv, cfg.prandtl
        )
        relaxation_time = local_relaxation_time(fields["rho"], fields["T"], cfg)
        fraction = np.minimum(dt / relaxation_time, 1.0)[..., None, None, None]
        distribution = transported + fraction * (equilibrium - transported)
        distribution = np.maximum(distribution, cfg.positivity_floor)

        if (step + 1) % cfg.check_interval == 0:
            fields = _macroscopic(distribution, vx, vy, vz, dv)
            velocity = np.stack([fields["u"], fields["v"]], axis=-1)
            heat_flux = np.stack([fields["qx"], fields["qy"]], axis=-1)
            changes = _fixed_scale_change(
                fields["T"],
                previous_temperature,
                velocity,
                previous_velocity,
                heat_flux,
                previous_heat_flux,
            )
            component_history.append(changes[:3])
            residual_history.append(changes[3])
            previous_temperature = fields["T"].copy()
            previous_velocity = velocity.copy()
            previous_heat_flux = heat_flux.copy()
            if step + 1 >= cfg.minimum_steps and changes[3] < cfg.tolerance:
                converged = True
                break

    fields = _macroscopic(distribution, vx, vy, vz, dv)
    density = fields["rho"] / max(float(np.mean(fields["rho"])), 1.0e-14)
    paper_velocity_scale = math.sqrt(2.0)
    bottom_heat_flux = _bottom_boundary_heat_flux(
        distribution, bottom_incoming, vx, vy, vz, dv
    )
    return {
        "T": np.asarray(fields["T"]),
        "rho": density,
        "u": np.asarray(fields["u"]) / paper_velocity_scale,
        "v": np.asarray(fields["v"]) / paper_velocity_scale,
        "qx": np.asarray(fields["qx"]) / paper_velocity_scale,
        "qy": np.asarray(fields["qy"]) / paper_velocity_scale,
        "bottom_wall_temperature": bottom_temperature,
        "bottom_heat_flux": bottom_heat_flux,
        "residual_history": np.asarray(residual_history),
        "component_change_history": np.asarray(component_history),
        "iterations": step + 1,
        "dt": dt,
        "converged": converged,
    }


def summarize_corrected_case(
    result: dict[str, np.ndarray | float | int | bool],
    cfg: CorrectedBottomHeatedConfig,
) -> dict[str, float | int | bool]:
    vertical_velocity = np.asarray(result["v"], dtype=np.float64)
    side_velocity = np.concatenate([vertical_velocity[:, 0], vertical_velocity[:, -1]])
    scale = max(float(np.max(np.abs(side_velocity))), 1.0e-14)
    positive_fraction = float(np.mean(side_velocity > 1.0e-8 * scale))
    residual = np.asarray(result["residual_history"], dtype=np.float64)
    cold_actual = discrete_wall_temperature(cfg.cold_temperature, cfg)
    hot_actual = discrete_wall_temperature(cfg.hot_temperature, cfg)
    return {
        "kn0": cfg.kn0,
        "cold_hot_ratio": cfg.cold_hot_ratio,
        "iterations": int(result["iterations"]),
        "converged": bool(result["converged"]),
        "final_fixed_scale_change": float(residual[-1]) if residual.size else float("nan"),
        "cold_wall_target_temperature": cfg.cold_temperature,
        "cold_wall_discrete_temperature": cold_actual,
        "cold_wall_relative_representation_error": abs(cold_actual - cfg.cold_temperature)
        / cfg.cold_temperature,
        "hot_wall_discrete_temperature": hot_actual,
        "hot_wall_relative_representation_error": abs(hot_actual - 1.0),
        "positive_lateral_velocity_fraction": positive_fraction,
        "lateral_hot_to_cold_majority": bool(positive_fraction > 0.5),
        "average_bottom_heat_flux": float(np.mean(result["bottom_heat_flux"])),
        "maximum_dimensionless_speed": float(
            np.max(np.hypot(result["u"], result["v"]))
        ),
        "mean_temperature": float(np.mean(result["T"])),
        "minimum_temperature": float(np.min(result["T"])),
        "maximum_temperature": float(np.max(result["T"])),
    }


def _nondecreasing(values: list[float], relative_tolerance: float = 0.03) -> bool:
    return all(
        current >= previous * (1.0 - relative_tolerance) - 1.0e-14
        for previous, current in zip(values, values[1:])
    )


def run_corrected_feasibility_matrix(
    output_dir: str | Path,
    base_cfg: CorrectedBottomHeatedConfig = CorrectedBottomHeatedConfig(),
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = (
        (0.1, 0.1),
        (1.0, 0.1),
        (10.0, 0.1),
        (1.0, 0.5),
        (10.0, 0.5),
    )
    rows: list[dict[str, float | int | bool]] = []
    arrays: dict[str, np.ndarray] = {}

    for kn0, ratio in conditions:
        cfg = CorrectedBottomHeatedConfig(
            nx=base_cfg.nx,
            ny=base_cfg.ny,
            nv=base_cfg.nv,
            velocity_extent=base_cfg.velocity_extent,
            kn0=kn0,
            cold_hot_ratio=ratio,
            viscosity_exponent=0.5,
            corner_ramp_fraction=base_cfg.corner_ramp_fraction,
            prandtl=base_cfg.prandtl,
            max_steps=base_cfg.max_steps,
            cfl=base_cfg.cfl,
            tolerance=base_cfg.tolerance,
            check_interval=base_cfg.check_interval,
            minimum_steps=base_cfg.minimum_steps,
            positivity_floor=base_cfg.positivity_floor,
        )
        result = solve_corrected_bottom_heated_case(cfg)
        rows.append(summarize_corrected_case(result, cfg))
        key = f"ratio{ratio:.1f}_kn{kn0:g}".replace(".", "p")
        for name in (
            "T",
            "rho",
            "u",
            "v",
            "qx",
            "qy",
            "bottom_wall_temperature",
            "bottom_heat_flux",
            "residual_history",
            "component_change_history",
        ):
            arrays[f"{name}_{key}"] = np.asarray(result[name])
        arrays[f"lateral_velocity_{key}"] = 0.5 * (
            np.asarray(result["v"])[:, 0] + np.asarray(result["v"])[:, -1]
        )

    ratio01 = sorted(
        [row for row in rows if row["cold_hot_ratio"] == 0.1],
        key=lambda row: float(row["kn0"]),
    )
    q01 = [float(row["average_bottom_heat_flux"]) for row in ratio01]
    high_kn_lateral = [
        row for row in ratio01 if float(row["kn0"]) in (1.0, 10.0)
    ]
    preregistered_checks = {
        "all_wall_temperatures_represented_within_2pct": all(
            float(row["cold_wall_relative_representation_error"]) < 0.02
            and float(row["hot_wall_relative_representation_error"]) < 0.02
            for row in rows
        ),
        "all_cases_fixed_scale_converged": all(bool(row["converged"]) for row in rows),
        "ratio_0p1_bottom_heat_flux_nondecreasing_with_kn0": _nondecreasing(q01),
        "ratio_0p1_high_kn_lateral_hot_to_cold_majority": all(
            bool(row["lateral_hot_to_cold_majority"])
            for row in high_kn_lateral
        ),
    }

    summary: dict[str, object] = {
        "stage": 25,
        "description": (
            "Corrected bottom-heated benchmark feasibility study with TH as the "
            "reference temperature and an odd zero-containing velocity grid"
        ),
        "configuration": {
            "grid": [base_cfg.nx, base_cfg.ny],
            "nv": base_cfg.nv,
            "velocity_extent": base_cfg.velocity_extent,
            "reference_temperature_convention": "T0=TH",
            "cold_wall_temperature": "TC/T0=TC/TH",
            "kn_mapping": "tau_ref=Kn0*sqrt(2/pi) in c0=sqrt(kT0/m) scale",
            "max_steps": base_cfg.max_steps,
            "tolerance": base_cfg.tolerance,
            "check_interval": base_cfg.check_interval,
            "conditions": [list(condition) for condition in conditions],
        },
        "rows": rows,
        "ratio_0p1_heat_flux_sequence": {
            "kn0": [float(row["kn0"]) for row in ratio01],
            "average_bottom_heat_flux": q01,
        },
        "preregistered_checks": preregistered_checks,
        "success_count": int(sum(preregistered_checks.values())),
        "check_count": len(preregistered_checks),
        "stage24_correction": (
            "Stage 24 used T0=(TH+TC)/2 and even Nv=12, whose nearest velocities "
            "were +/-0.5 for extent 6. The TC/TH=0.1 wall was therefore not "
            "representable and the Stage-24 structural comparison is diagnostic only."
        ),
        "interpretation_guard": (
            "Passing structural checks is not quantitative validation. Exact literature "
            "curves or independent raw DSMC/DVM data are still required."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(output_dir / "fields_and_profiles.npz", **arrays)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 25 corrected bottom-heated benchmark study"
    )
    parser.add_argument("--output-dir", default="outputs/stage25_bottom_heated_corrected")
    parser.add_argument("--nx", type=int, default=18)
    parser.add_argument("--ny", type=int, default=18)
    parser.add_argument("--nv", type=int, default=17)
    parser.add_argument("--velocity-extent", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=4200)
    parser.add_argument("--tolerance", type=float, default=2.0e-5)
    args = parser.parse_args()
    summary = run_corrected_feasibility_matrix(
        args.output_dir,
        CorrectedBottomHeatedConfig(
            nx=args.nx,
            ny=args.ny,
            nv=args.nv,
            velocity_extent=args.velocity_extent,
            max_steps=args.max_steps,
            tolerance=args.tolerance,
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
