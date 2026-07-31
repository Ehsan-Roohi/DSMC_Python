from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import numpy as np

from .dvm_shakhov import ShakhovReferenceConfig, _macroscopic, _velocity_grid
from .dvm_shakhov_corrected import (
    _discrete_maxwellian,
    _shakhov_equilibrium,
    _unit_wall_maxwellian,
)


@dataclass(frozen=True)
class BottomHeatedBenchmarkConfig:
    """Bottom-heated square-cavity benchmark following Tatsios et al. (2014).

    The conference paper specifies the temperature ratio but does not explicitly
    state the numerical value chosen for its reference temperature in the parsed
    text. This implementation sets T0=(TH+TC)/2, so the dimensionless wall
    temperatures are TH/T0=2/(1+r) and TC/T0=2r/(1+r). Results are reported in
    the paper's velocity and heat-flux scales.
    """

    nx: int = 18
    ny: int = 18
    nv: int = 12
    velocity_extent: float = 6.0
    kn0: float = 0.10
    cold_hot_ratio: float = 0.5
    viscosity_exponent: float = 0.5
    corner_ramp_fraction: float = 0.05
    prandtl: float = 2.0 / 3.0
    max_steps: int = 2400
    cfl: float = 0.30
    tolerance: float = 3.0e-6
    check_interval: int = 50
    minimum_steps: int = 500
    positivity_floor: float = 1.0e-30

    @property
    def hot_temperature(self) -> float:
        return 2.0 / (1.0 + self.cold_hot_ratio)

    @property
    def cold_temperature(self) -> float:
        return 2.0 * self.cold_hot_ratio / (1.0 + self.cold_hot_ratio)


def paper_kn0_to_solver_relaxation_scale(kn0: float) -> float:
    """Convert the paper's Kn0 to the solver's c0=sqrt(kT0/m) time scale."""
    if kn0 <= 0.0:
        raise ValueError("kn0 must be positive")
    return float(kn0 * math.sqrt(2.0 / math.pi))


def bottom_wall_temperature_profile(
    nx: int,
    cold_temperature: float,
    hot_temperature: float,
    ramp_fraction: float = 0.05,
) -> np.ndarray:
    if nx < 2:
        raise ValueError("nx must be at least two")
    if not 0.0 < cold_temperature < hot_temperature:
        raise ValueError("require 0 < cold temperature < hot temperature")
    if not 0.0 < ramp_fraction < 0.5:
        raise ValueError("ramp fraction must lie between zero and one half")
    x = (np.arange(nx, dtype=np.float64) + 0.5) / nx
    edge_distance = np.minimum(x, 1.0 - x)
    ramp_weight = np.clip(edge_distance / ramp_fraction, 0.0, 1.0)
    return cold_temperature + (hot_temperature - cold_temperature) * ramp_weight


def _cold_wall_incoming(
    outgoing_distribution: np.ndarray,
    normal_velocity: np.ndarray,
    incoming_mask: np.ndarray,
    outgoing_mask: np.ndarray,
    wall_maxwellian: np.ndarray,
    dv: float,
) -> np.ndarray:
    measure = dv**3
    outgoing_flux = np.sum(
        normal_velocity[None]
        * outgoing_distribution
        * outgoing_mask[None],
        axis=(-3, -2, -1),
    ) * measure
    incoming_unit = float(
        np.sum(normal_velocity * wall_maxwellian * incoming_mask) * measure
    )
    scale = -outgoing_flux / max(incoming_unit, 1.0e-14)
    return scale[:, None, None, None] * wall_maxwellian[None]


def _profiled_wall_incoming(
    distribution: np.ndarray,
    cfg: BottomHeatedBenchmarkConfig,
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


def local_relaxation_time(
    density: np.ndarray,
    temperature: np.ndarray,
    cfg: BottomHeatedBenchmarkConfig,
) -> np.ndarray:
    density = np.maximum(np.asarray(density, dtype=np.float64), 1.0e-12)
    temperature = np.maximum(np.asarray(temperature, dtype=np.float64), 1.0e-10)
    return (
        paper_kn0_to_solver_relaxation_scale(cfg.kn0)
        * temperature ** (cfg.viscosity_exponent - 1.0)
        / density
    )


def _bottom_boundary_heat_flux(
    distribution: np.ndarray,
    bottom_incoming: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float,
) -> np.ndarray:
    """Dimensionless energy/heat flux leaving the stationary bottom wall.

    The wall mass flux is zero by construction, so total energy flux equals heat
    flux. Division by sqrt(2) converts the solver c0 scale to the paper's
    v0=sqrt(2 k T0/m) normalization.
    """
    incoming_mask = vy > 0.0
    boundary = np.where(
        incoming_mask[None],
        bottom_incoming,
        distribution[0],
    )
    speed2 = vx * vx + vy * vy + vz * vz
    solver_flux = 0.5 * np.sum(
        boundary * vy[None] * speed2[None],
        axis=(-3, -2, -1),
    ) * dv**3
    return solver_flux / math.sqrt(2.0)


def solve_bottom_heated_case(
    cfg: BottomHeatedBenchmarkConfig,
) -> dict[str, np.ndarray | float | int]:
    if cfg.nx < 3 or cfg.ny < 3 or cfg.nv < 6:
        raise ValueError("require nx, ny >= 3 and nv >= 6")
    if not 0.0 < cfg.cold_hot_ratio < 1.0:
        raise ValueError("cold_hot_ratio must lie between zero and one")
    if cfg.viscosity_exponent != 0.5:
        raise ValueError("Stage 24 is preregistered to the paper's hard-sphere omega=1/2")

    velocity_cfg = ShakhovReferenceConfig(
        nx=cfg.nx,
        ny=cfg.ny,
        nv=cfg.nv,
        velocity_extent=cfg.velocity_extent,
    )
    vx, vy, vz, dv = _velocity_grid(velocity_cfg)
    rho = np.ones((cfg.ny, cfg.nx))
    zero = np.zeros_like(rho)
    distribution = _discrete_maxwellian(
        rho,
        zero,
        zero,
        zero,
        np.ones_like(rho),
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
    bottom_incoming = np.empty((cfg.nx, cfg.nv, cfg.nv, cfg.nv))
    bottom_temperature = np.empty(cfg.nx)

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
            temperature_residual = float(
                np.max(np.abs(fields["T"] - previous_temperature))
                / max(float(np.max(np.abs(fields["T"]))), 1.0e-14)
            )
            velocity_residual = float(
                np.max(np.abs(velocity - previous_velocity))
                / max(float(np.max(np.abs(velocity))), 1.0e-12)
            )
            heat_flux_residual = float(
                np.max(np.abs(heat_flux - previous_heat_flux))
                / max(float(np.max(np.abs(heat_flux))), 1.0e-12)
            )
            residual = max(temperature_residual, velocity_residual, heat_flux_residual)
            residual_history.append(residual)
            previous_temperature = fields["T"].copy()
            previous_velocity = velocity.copy()
            previous_heat_flux = heat_flux.copy()
            if step + 1 >= cfg.minimum_steps and residual < cfg.tolerance:
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
        "iterations": step + 1,
        "dt": dt,
    }


def summarize_literature_case(
    result: dict[str, np.ndarray | float | int],
    cfg: BottomHeatedBenchmarkConfig,
) -> dict[str, float | int | bool]:
    vertical_velocity = np.asarray(result["v"], dtype=np.float64)
    side_velocity = np.concatenate(
        [vertical_velocity[:, 0], vertical_velocity[:, -1]]
    )
    scale = max(float(np.max(np.abs(side_velocity))), 1.0e-14)
    positive_fraction = float(np.mean(side_velocity > 1.0e-8 * scale))
    bottom_heat_flux = np.asarray(result["bottom_heat_flux"], dtype=np.float64)
    residual = np.asarray(result["residual_history"], dtype=np.float64)
    return {
        "kn0": cfg.kn0,
        "cold_hot_ratio": cfg.cold_hot_ratio,
        "iterations": int(result["iterations"]),
        "final_residual": float(residual[-1]) if residual.size else float("nan"),
        "positive_lateral_velocity_fraction": positive_fraction,
        "lateral_hot_to_cold_majority": bool(positive_fraction > 0.5),
        "average_bottom_heat_flux": float(np.mean(bottom_heat_flux)),
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


def run_literature_validation_matrix(
    output_dir: str | Path,
    base_cfg: BottomHeatedBenchmarkConfig = BottomHeatedBenchmarkConfig(),
    kn0_values: tuple[float, ...] = (0.1, 1.0, 10.0),
    ratios: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | bool]] = []
    arrays: dict[str, np.ndarray] = {}

    for ratio in ratios:
        for kn0 in kn0_values:
            cfg = BottomHeatedBenchmarkConfig(
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
            result = solve_bottom_heated_case(cfg)
            row = summarize_literature_case(result, cfg)
            rows.append(row)
            key = f"ratio{ratio:.1f}_kn{kn0:g}".replace(".", "p")
            for name in (
                "T", "rho", "u", "v", "qx", "qy",
                "bottom_wall_temperature", "bottom_heat_flux",
            ):
                arrays[f"{name}_{key}"] = np.asarray(result[name])
            arrays[f"lateral_velocity_{key}"] = 0.5 * (
                np.asarray(result["v"])[:, 0] + np.asarray(result["v"])[:, -1]
            )

    by_ratio: dict[str, dict[str, object]] = {}
    for ratio in ratios:
        subset = sorted(
            [row for row in rows if row["cold_hot_ratio"] == ratio],
            key=lambda row: float(row["kn0"]),
        )
        q_values = [float(row["average_bottom_heat_flux"]) for row in subset]
        by_ratio[f"ratio_{ratio:.1f}"] = {
            "kn0": [float(row["kn0"]) for row in subset],
            "average_bottom_heat_flux": q_values,
            "heat_flux_nondecreasing_with_kn0_3pct_tolerance": _nondecreasing(q_values),
            "all_lateral_profiles_hot_to_cold_majority": all(
                bool(row["lateral_hot_to_cold_majority"]) for row in subset
            ),
        }

    heat_flux_by_case = {
        (float(row["cold_hot_ratio"]), float(row["kn0"])):
        float(row["average_bottom_heat_flux"])
        for row in rows
    }
    ratio_ordering = {
        f"kn0_{kn0:g}": bool(
            heat_flux_by_case[(0.5, kn0)] > heat_flux_by_case[(0.1, kn0)]
        )
        for kn0 in kn0_values
        if kn0 > 0.5
    }
    structural_checks = {
        "lateral_hot_to_cold_majority_all_cases": all(
            bool(row["lateral_hot_to_cold_majority"]) for row in rows
        ),
        "bottom_heat_flux_increases_with_kn0_all_ratios": all(
            bool(summary["heat_flux_nondecreasing_with_kn0_3pct_tolerance"])
            for summary in by_ratio.values()
        ),
        "q_ratio_0p5_exceeds_0p1_for_kn0_gt_0p5": all(ratio_ordering.values()),
    }

    summary: dict[str, object] = {
        "stage": 24,
        "description": (
            "Structural cross-literature validation against the bottom-heated "
            "square-microcavity Shakhov/DSMC benchmark of Tatsios et al."
        ),
        "source": {
            "title": "Non-equilibrium gas flow and heat transfer in a bottom heated square microcavity",
            "authors": "Tatsios, Vargas, Stefanov, Valougeorgis",
            "conference": "4th Micro and Nano Flows Conference, 2014",
            "published_conditions": {
                "kn0": list(kn0_values),
                "cold_hot_ratios": list(ratios),
                "viscosity_exponent": 0.5,
                "diffuse_walls": True,
                "bottom_corner_ramp_fraction": 0.05,
            },
        },
        "configuration": {
            "grid": [base_cfg.nx, base_cfg.ny],
            "nv": base_cfg.nv,
            "velocity_extent": base_cfg.velocity_extent,
            "reference_temperature_convention": "T0=(TH+TC)/2",
            "kn_mapping": "tau_ref=Kn0*sqrt(2/pi) in solver c0=sqrt(kT0/m) scale",
            "max_steps": base_cfg.max_steps,
            "tolerance": base_cfg.tolerance,
        },
        "rows": rows,
        "by_ratio": by_ratio,
        "ratio_ordering_checks": ratio_ordering,
        "structural_checks": structural_checks,
        "structural_success_count": int(sum(structural_checks.values())),
        "structural_check_count": len(structural_checks),
        "validation_scope": (
            "Qualitative/structural cross-literature validation. Exact figure values were not "
            "digitized, and the present solver uses Cartesian three-velocity quadrature and "
            "first-order upwind transport rather than the paper's polar quadrature and "
            "second-order finite-volume discretization."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(output_dir / "fields_and_profiles.npz", **arrays)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 24 bottom-heated square-cavity literature validation"
    )
    parser.add_argument("--output-dir", default="outputs/stage24_bottom_heated")
    parser.add_argument("--nx", type=int, default=18)
    parser.add_argument("--ny", type=int, default=18)
    parser.add_argument("--nv", type=int, default=12)
    parser.add_argument("--velocity-extent", type=float, default=6.0)
    parser.add_argument("--max-steps", type=int, default=2400)
    parser.add_argument("--tolerance", type=float, default=3.0e-6)
    parser.add_argument("--kn0", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.5, 0.9])
    args = parser.parse_args()
    summary = run_literature_validation_matrix(
        args.output_dir,
        BottomHeatedBenchmarkConfig(
            nx=args.nx,
            ny=args.ny,
            nv=args.nv,
            velocity_extent=args.velocity_extent,
            max_steps=args.max_steps,
            tolerance=args.tolerance,
        ),
        tuple(args.kn0),
        tuple(args.ratios),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
