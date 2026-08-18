from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .vhs_model import KB, PhysicalCavityConfig


@dataclass
class LidWallEventAccumulator:
    """Accumulate incident and reflected lid events for PRE Eqs. (7) and (8).

    The source method counts each wall collision twice: once with the incoming
    velocity and once with the diffusely reflected velocity.
    """

    cfg: PhysicalCavityConfig
    event_count: np.ndarray = field(init=False)
    inverse_flux_weight: np.ndarray = field(init=False)
    weighted_slip: np.ndarray = field(init=False)
    weighted_relative_speed2: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.event_count = np.zeros(self.cfg.nx, dtype=np.int64)
        self.inverse_flux_weight = np.zeros(self.cfg.nx, dtype=np.float64)
        self.weighted_slip = np.zeros(self.cfg.nx, dtype=np.float64)
        self.weighted_relative_speed2 = np.zeros(self.cfg.nx, dtype=np.float64)

    def add(
        self,
        wall: str,
        tangential_position: np.ndarray,
        incoming_velocity: np.ndarray,
        relative_particle_weight: np.ndarray,
        wall_velocity: np.ndarray,
    ) -> None:
        if wall != "top" or len(tangential_position) == 0:
            return
        position = np.asarray(tangential_position, dtype=np.float64)
        incoming = np.asarray(incoming_velocity, dtype=np.float64)
        particle_weight = np.asarray(relative_particle_weight, dtype=np.float64)
        wall_velocity = np.asarray(wall_velocity, dtype=np.float64)
        bins = np.clip(
            (position / self.cfg.length * self.cfg.nx).astype(np.int64),
            0,
            self.cfg.nx - 1,
        )
        relative = incoming - wall_velocity[None, :]
        normal_speed = np.abs(relative[:, 1])
        valid = (
            np.isfinite(normal_speed)
            & (normal_speed > 1.0e-14)
            & np.isfinite(particle_weight)
            & (particle_weight > 0.0)
        )
        if not np.any(valid):
            return
        bins = bins[valid]
        relative = relative[valid]
        # Wall-hit events are sampled in proportion to |v_n|.  The inverse
        # normal-speed factor reconstructs the surface distribution in the
        # direct microscopic estimator used by Mohammadzadeh et al.
        inverse_flux = particle_weight[valid] / normal_speed[valid]
        slip = -relative[:, 0]
        speed2 = np.sum(relative**2, axis=1)
        self.event_count += np.bincount(bins, minlength=self.cfg.nx)
        self.inverse_flux_weight += np.bincount(
            bins,
            weights=inverse_flux,
            minlength=self.cfg.nx,
        )
        self.weighted_slip += np.bincount(
            bins,
            weights=inverse_flux * slip,
            minlength=self.cfg.nx,
        )
        self.weighted_relative_speed2 += np.bincount(
            bins,
            weights=inverse_flux * speed2,
            minlength=self.cfg.nx,
        )

    def finalize(self) -> dict[str, np.ndarray]:
        safe = self.inverse_flux_weight
        valid = safe > 0.0
        slip = np.full(self.cfg.nx, np.nan, dtype=np.float64)
        temperature = np.full(self.cfg.nx, np.nan, dtype=np.float64)
        slip[valid] = self.weighted_slip[valid] / safe[valid]
        mean_speed2 = np.zeros(self.cfg.nx, dtype=np.float64)
        mean_speed2[valid] = self.weighted_relative_speed2[valid] / safe[valid]
        gas_constant = KB / self.cfg.vhs.mass
        # PRE Eq. (8) prints ``T_gas - T_wall`` on the left, but using that
        # literally would return about 600 K for a 300 K equilibrium wall,
        # contradicting Fig. 5.  The kinetic expression on the right is the
        # absolute incident-gas temperature and is used as such here.
        temperature[valid] = (
            mean_speed2[valid] - slip[valid] ** 2
        ) / (3.0 * gas_constant)
        normalized_slip = (
            slip / self.cfg.lid_velocity_x
            if self.cfg.lid_velocity_x != 0.0
            else np.full_like(slip, np.nan)
        )
        return {
            "microscopic_lid_slip": slip,
            "microscopic_lid_slip_over_uwall": normalized_slip,
            "microscopic_lid_T": temperature,
            "microscopic_lid_event_count": self.event_count.astype(float),
        }
