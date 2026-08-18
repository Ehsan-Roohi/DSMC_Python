from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .vhs_model import PhysicalCavityConfig


WALLS = ("left", "right", "bottom", "top")
_NORMAL_AXIS = {"left": 0, "right": 0, "bottom": 1, "top": 1}
_INWARD_SIGN = {"left": 1.0, "right": -1.0, "bottom": 1.0, "top": -1.0}


def _relative_imbalance(incoming: np.ndarray, outgoing: np.ndarray) -> np.ndarray:
    """Return ``|out-in|`` divided by the two-sided mean throughput."""
    scale = 0.5 * (incoming + outgoing)
    difference = np.abs(outgoing - incoming)
    return np.divide(
        difference,
        scale,
        out=np.where(difference == 0.0, 0.0, np.inf),
        where=scale > 0.0,
    )


@dataclass
class WallBalanceAccumulator:
    """Audit particle balance using the existing diffuse-wall callback.

    ``apply_diffuse_walls`` calls its event handler once before and once after
    each reflection.  This accumulator deliberately does not depend on call
    order: an event is classified from the sign of velocity relative to the
    wall along the inward normal.  The event weights reported here are
    physical represented-particle counts (relative particle weight times
    FNUM), not the inverse-flux weights used by microscopic wall estimators.
    Consequently this audit can be composed with ``LidWallEventAccumulator``
    without changing that estimator's required incident/reflected
    double-counting.
    """

    cfg: PhysicalCavityConfig
    incoming_count: np.ndarray = field(init=False)
    outgoing_count: np.ndarray = field(init=False)
    incoming_represented_weight: np.ndarray = field(init=False)
    outgoing_represented_weight: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.incoming_count = np.zeros(len(WALLS), dtype=np.int64)
        self.outgoing_count = np.zeros(len(WALLS), dtype=np.int64)
        self.incoming_represented_weight = np.zeros(len(WALLS), dtype=np.float64)
        self.outgoing_represented_weight = np.zeros(len(WALLS), dtype=np.float64)

    def add(
        self,
        wall: str,
        tangential_position: np.ndarray,
        velocity: np.ndarray,
        relative_particle_weight: np.ndarray,
        wall_velocity: np.ndarray,
    ) -> None:
        """Consume one batch from ``apply_diffuse_walls``' event handler."""
        if wall not in _NORMAL_AXIS:
            raise ValueError(f"unknown wall {wall!r}")
        position = np.asarray(tangential_position, dtype=np.float64)
        particle_velocity = np.asarray(velocity, dtype=np.float64)
        particle_weight = np.asarray(relative_particle_weight, dtype=np.float64)
        wall_velocity = np.asarray(wall_velocity, dtype=np.float64)

        if position.ndim != 1:
            raise ValueError("tangential_position must be one-dimensional")
        if particle_velocity.shape != (len(position), 3):
            raise ValueError("velocity must have shape (event_count, 3)")
        if particle_weight.shape != (len(position),):
            raise ValueError("relative_particle_weight must match event_count")
        if wall_velocity.shape != (3,):
            raise ValueError("wall_velocity must contain three components")
        if not (
            np.all(np.isfinite(position))
            and np.all(np.isfinite(particle_velocity))
            and np.all(np.isfinite(particle_weight))
            and np.all(np.isfinite(wall_velocity))
        ):
            raise ValueError("wall-event inputs must be finite")
        if np.any(particle_weight <= 0.0):
            raise ValueError("relative particle weights must be positive")
        if len(position) == 0:
            return

        axis = _NORMAL_AXIS[wall]
        inward_normal_velocity = (
            (particle_velocity[:, axis] - wall_velocity[axis])
            * _INWARD_SIGN[wall]
        )
        incoming = inward_normal_velocity < 0.0
        outgoing = inward_normal_velocity > 0.0
        if not np.all(incoming | outgoing):
            raise ValueError("a wall event has zero wall-relative normal velocity")

        represented = particle_weight * self.cfg.real_particles_per_sim_particle
        index = WALLS.index(wall)
        self.incoming_count[index] += int(np.count_nonzero(incoming))
        self.outgoing_count[index] += int(np.count_nonzero(outgoing))
        self.incoming_represented_weight[index] += float(
            np.sum(represented[incoming])
        )
        self.outgoing_represented_weight[index] += float(
            np.sum(represented[outgoing])
        )

    def finalize(self) -> dict[str, object]:
        """Return per-wall and global mass-balance diagnostics.

        Relative imbalance is the absolute incoming/outgoing difference
        divided by their arithmetic-mean throughput.  It is zero for an empty
        audit and infinite when events are observed in only one direction.
        Signed net represented weight is also retained so a caller can inspect
        the direction of any imbalance.
        """
        incoming_weight = self.incoming_represented_weight.copy()
        outgoing_weight = self.outgoing_represented_weight.copy()
        incoming_count = self.incoming_count.copy()
        outgoing_count = self.outgoing_count.copy()
        per_wall_relative = _relative_imbalance(
            incoming_weight,
            outgoing_weight,
        )
        total_incoming = float(np.sum(incoming_weight))
        total_outgoing = float(np.sum(outgoing_weight))
        relative_total = float(
            _relative_imbalance(
                np.asarray(total_incoming),
                np.asarray(total_outgoing),
            )
        )
        return {
            "wall_order": WALLS,
            "incoming_count": incoming_count,
            "outgoing_count": outgoing_count,
            "incoming_represented_weight": incoming_weight,
            "outgoing_represented_weight": outgoing_weight,
            "net_represented_weight": outgoing_weight - incoming_weight,
            "per_wall_relative_net_mass_imbalance": per_wall_relative,
            "total_incoming_count": int(np.sum(incoming_count)),
            "total_outgoing_count": int(np.sum(outgoing_count)),
            "total_incoming_represented_weight": total_incoming,
            "total_outgoing_represented_weight": total_outgoing,
            "total_net_represented_weight": total_outgoing - total_incoming,
            "relative_net_mass_imbalance": relative_total,
        }
