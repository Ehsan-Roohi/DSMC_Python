from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .vhs_model import (
    KB,
    PhysicalCavityConfig,
    PhysicalParticleState,
    _cell_xy,
)


@dataclass
class PhysicalMomentAccumulator:
    """Accumulate raw DSMC moments before constructing central moments.

    Forming temperature and heat flux after accumulation avoids the
    finite-particle bias caused by averaging nonlinear instantaneous cell
    estimates.  Particle weights include the physical FNUM conversion.
    """

    cfg: PhysicalCavityConfig
    samples: int = 0
    simulated_count: np.ndarray = field(init=False)
    m0: np.ndarray = field(init=False)
    m1: np.ndarray = field(init=False)
    m2: np.ndarray = field(init=False)
    energy: np.ndarray = field(init=False)
    energy_velocity: np.ndarray = field(init=False)
    speed4: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        ncell = self.cfg.nx * self.cfg.ny
        self.simulated_count = np.zeros(ncell, dtype=np.float64)
        self.m0 = np.zeros(ncell, dtype=np.float64)
        self.m1 = np.zeros((ncell, 3), dtype=np.float64)
        self.m2 = np.zeros((ncell, 3, 3), dtype=np.float64)
        self.energy = np.zeros(ncell, dtype=np.float64)
        self.energy_velocity = np.zeros((ncell, 3), dtype=np.float64)
        # Fourth raw speed moment.  This is the minimum extra DSMC-native
        # statistic needed to evaluate the sampling variance of temperature
        # without a local-equilibrium closure.  Third-moment heat-flux
        # variances still require sixth-order raw moments and are therefore
        # estimated from independent sampling blocks.
        self.speed4 = np.zeros(ncell, dtype=np.float64)

    def add(
        self,
        state: PhysicalParticleState,
        *,
        return_instantaneous: bool = False,
    ) -> dict[str, np.ndarray] | None:
        """Accumulate one state and optionally return its basic cell fields.

        The optional result reuses the raw reductions required by the
        accumulator.  This avoids a second cell lookup and six duplicate
        ``bincount`` passes when NTC temporal-scatter diagnostics are enabled.
        """
        ix, iy = _cell_xy(state.pos, self.cfg)
        cell = iy * self.cfg.nx + ix
        ncell = self.cfg.nx * self.cfg.ny
        represented = (
            self.cfg.real_particles_per_sim_particle * state.weight
        )
        speed2 = np.sum(state.vel**2, axis=1)

        sample_count = np.bincount(cell, minlength=ncell)
        sample_m0 = np.bincount(
            cell,
            weights=represented,
            minlength=ncell,
        )
        sample_m1 = np.empty((ncell, 3), dtype=np.float64)
        self.simulated_count += sample_count
        self.m0 += sample_m0
        for i in range(3):
            sample_m1[:, i] = np.bincount(
                cell,
                weights=represented * state.vel[:, i],
                minlength=ncell,
            )
            self.m1[:, i] += sample_m1[:, i]
            self.energy_velocity[:, i] += np.bincount(
                cell,
                weights=represented * speed2 * state.vel[:, i],
                minlength=ncell,
            )
            for j in range(3):
                self.m2[:, i, j] += np.bincount(
                    cell,
                    weights=(
                        represented * state.vel[:, i] * state.vel[:, j]
                    ),
                    minlength=ncell,
                )
        sample_energy = np.bincount(
            cell,
            weights=represented * speed2,
            minlength=ncell,
        )
        self.energy += sample_energy
        self.speed4 += np.bincount(
            cell,
            weights=represented * speed2**2,
            minlength=ncell,
        )
        self.samples += 1
        if not return_instantaneous:
            return None

        safe = np.maximum(sample_m0, 1.0e-200)
        velocity = sample_m1 / safe[:, None]
        peculiar2 = np.maximum(
            sample_energy / safe
            - velocity[:, 0] ** 2
            - velocity[:, 1] ** 2
            - velocity[:, 2] ** 2,
            0.0,
        )
        shape = (self.cfg.ny, self.cfg.nx)
        return {
            "u": velocity[:, 0].reshape(shape),
            "v": velocity[:, 1].reshape(shape),
            "w": velocity[:, 2].reshape(shape),
            "T": (
                self.cfg.vhs.mass * peculiar2 / (3.0 * KB)
            ).reshape(shape),
        }

    def finalize(self) -> dict[str, np.ndarray]:
        if self.samples <= 0:
            raise ValueError("No physical samples were accumulated")
        if np.any(self.m0 <= 0.0):
            empty = int(np.count_nonzero(self.m0 <= 0.0))
            raise ValueError(f"Raw-moment accumulator has {empty} empty cells")

        safe = self.m0
        mean_velocity = self.m1 / safe[:, None]
        second = self.m2 / safe[:, None, None]
        mean_speed2 = self.energy / safe
        mean_energy_velocity = self.energy_velocity / safe[:, None]
        mean_velocity2 = np.sum(mean_velocity**2, axis=1)
        peculiar2 = np.maximum(mean_speed2 - mean_velocity2, 0.0)
        temperature = self.cfg.vhs.mass * peculiar2 / (3.0 * KB)

        # <|c|^2 c_i> expressed solely through accumulated raw moments.
        second_times_mean = np.einsum(
            "nij,nj->ni",
            second,
            mean_velocity,
        )
        central_energy_velocity = (
            mean_energy_velocity
            - mean_velocity * mean_speed2[:, None]
            - 2.0 * second_times_mean
            + 2.0 * mean_velocity * mean_velocity2[:, None]
        )
        number_density = (
            self.m0 / float(self.samples) / self.cfg.cell_volume
        )
        heat_flux = (
            0.5
            * self.cfg.vhs.mass
            * number_density[:, None]
            * central_energy_velocity
        )

        shape = (self.cfg.ny, self.cfg.nx)
        density_mean = max(float(number_density.mean()), 1.0e-300)
        return {
            "number_density": number_density.reshape(shape),
            "rho": (number_density / density_mean).reshape(shape),
            "u": mean_velocity[:, 0].reshape(shape),
            "v": mean_velocity[:, 1].reshape(shape),
            "w": mean_velocity[:, 2].reshape(shape),
            "T": temperature.reshape(shape),
            "qx": heat_flux[:, 0].reshape(shape),
            "qy": heat_flux[:, 1].reshape(shape),
            "qz": heat_flux[:, 2].reshape(shape),
            "count": (
                self.simulated_count / float(self.samples)
            ).reshape(shape),
            "represented": (
                self.m0 / float(self.samples)
            ).reshape(shape),
        }
