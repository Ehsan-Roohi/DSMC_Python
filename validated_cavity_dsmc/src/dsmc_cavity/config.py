"""Configuration and DSMC resolution safeguards."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


K_B = 1.380649e-23
MASS_ARGON = 6.63e-26


@dataclass
class SimulationConfig:
    model: str = "ntc-prescan"
    backend: str = "cpu"
    kn: float = 0.1
    length: float = 1.0e-6
    nx: int = 32
    ny: int = 32
    particles_per_cell: int = 16
    wall_temperature: float = 300.0
    lid_velocity: float = 100.0
    mass: float = MASS_ARGON
    diameter_ref: float = 4.17e-10
    temperature_ref: float = 273.0
    viscosity_index: float = 0.81
    dt: float | None = None
    steps: int = 1800
    warmup_steps: int = 600
    sample_stride: int = 5
    seed: int = 20260803
    gbt_fraction: float = 0.5
    tas_target_particles: int = 8
    probability_target: float = 0.20
    strict_probability: bool = True
    output_dir: str = "results/run"
    label: str = "cavity"

    @property
    def dx(self) -> float:
        return self.length / self.nx

    @property
    def dy(self) -> float:
        return self.length / self.ny

    @property
    def cell_volume(self) -> float:
        # Unit span is chosen equal to L, as in the conventional 2-D DSMC model.
        return self.dx * self.dy * self.length

    @property
    def mean_free_path(self) -> float:
        return self.kn * self.length

    @property
    def number_density(self) -> float:
        return 1.0 / (
            math.sqrt(2.0) * math.pi * self.diameter_ref**2 * self.mean_free_path
        )

    @property
    def particle_weight(self) -> float:
        return self.number_density * self.cell_volume / self.particles_per_cell

    @property
    def thermal_speed_mp(self) -> float:
        return math.sqrt(2.0 * K_B * self.wall_temperature / self.mass)

    @property
    def mean_collision_time(self) -> float:
        return self.mean_free_path / self.thermal_speed_mp

    def selected_particles(self, occupancy: int) -> int:
        return max(1, min(occupancy - 1, int(round(self.gbt_fraction * occupancy))))

    def bt_multiplier_bound(self, model: str, occupancy: int) -> float:
        n = max(2, occupancy)
        name = model.lower()
        if name in {"sbt", "sbt-tas"}:
            return float(n - 1)
        if name == "gbt" or name == "gbt-tas":
            ns = self.selected_particles(n)
            corr = n * (n - 1.0) / (ns * (2.0 * n - ns - 1.0))
            return corr * (n - 1.0)
        if name == "ssbt":
            return 0.5 * (n - 1.0)
        if name == "sgbt":
            ns = self.selected_particles(n)
            return n * (n - 1.0) / (2.0 * ns)
        return 0.0

    def conservative_sigma_g(self) -> float:
        from .physics import vhs_sigma_g_scalar

        g_bound = 8.0 * math.sqrt(K_B * self.wall_temperature / self.mass)
        g_bound += 2.0 * abs(self.lid_velocity)
        return vhs_sigma_g_scalar(
            g_bound,
            self.diameter_ref,
            self.temperature_ref,
            self.viscosity_index,
            self.mass,
        )

    def recommended_dt(self) -> tuple[float, dict[str, float]]:
        move = 0.25 * min(self.dx, self.dy) / (
            4.0 * self.thermal_speed_mp + abs(self.lid_velocity)
        )
        collision = 0.10 * self.mean_collision_time
        limits = {"movement": move, "collision": collision}
        model = self.model.lower()
        if model in {"sbt", "gbt", "ssbt", "sgbt", "sbt-tas", "gbt-tas"}:
            occupancy_bound = max(4, 2 * self.particles_per_cell)
            mult = self.bt_multiplier_bound(model, occupancy_bound)
            bt = (
                self.probability_target
                * self.cell_volume
                / (self.particle_weight * self.conservative_sigma_g() * mult)
            )
            # TAS subcells have smaller volumes.  Use the configured target occupancy.
            if model.endswith("-tas"):
                nsub = max(1, int(math.sqrt(occupancy_bound / self.tas_target_particles)))
                # A staggered corner subcell has one-quarter of a regular
                # subcell volume and is therefore the conservative bound.
                bt /= 4.0 * nsub * nsub
            limits["bernoulli_probability"] = bt
        return min(limits.values()), limits

    def resolved_dt(self) -> tuple[float, dict[str, float]]:
        recommended, limits = self.recommended_dt()
        if self.dt is None:
            return recommended, limits
        if self.dt > 1.001 * recommended and self.strict_probability:
            detail = ", ".join(f"{key}={value:.3e}" for key, value in limits.items())
            raise ValueError(
                f"Requested dt={self.dt:.3e} exceeds the conservative limit "
                f"{recommended:.3e} for {self.model}. Limits: {detail}. "
                "Bernoulli-trial schemes require the smaller probability-based dt."
            )
        return self.dt, limits

    def validate(self) -> None:
        if self.nx < 2 or self.ny < 2:
            raise ValueError("nx and ny must both be at least 2")
        if self.particles_per_cell < 2:
            raise ValueError("particles_per_cell must be at least 2")
        if not (0.0 < self.kn):
            raise ValueError("kn must be positive")
        if self.steps <= self.warmup_steps:
            raise ValueError("steps must be greater than warmup_steps")
        if self.sample_stride < 1:
            raise ValueError("sample_stride must be positive")
        if not (0.0 < self.gbt_fraction <= 1.0):
            raise ValueError("gbt_fraction must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        dt, limits = self.resolved_dt()
        data.update(
            {
                "resolved_dt": dt,
                "dt_limits": limits,
                "mean_free_path": self.mean_free_path,
                "number_density": self.number_density,
                "particle_weight": self.particle_weight,
                "dx_over_lambda": self.dx / self.mean_free_path,
                "dy_over_lambda": self.dy / self.mean_free_path,
                "dt_over_collision_time": dt / self.mean_collision_time,
            }
        )
        return data
