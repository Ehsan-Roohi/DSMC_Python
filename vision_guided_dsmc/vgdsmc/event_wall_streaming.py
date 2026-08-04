"""Event-driven free flight with diffuse reflection at square-cavity walls.

This module is intentionally opt-in.  It does not replace or modify the
legacy ``pos += vel * dt; apply_diffuse_walls(...)`` path.  In contrast with
that overshoot-and-fold path, :func:`stream_with_diffuse_walls` advances each
particle to its earliest wall impact, samples the diffuse reflection there,
and uses the reflected velocity for the unspent part of the time step.

Both supported callbacks receive batches.  ``wall_event_handler`` has the
same five-argument contract as the legacy wall handler and is called once for
the incident state and once for the reflected state.  The separate
``phase_event_handler`` adds an explicit ``"incoming"``/``"outgoing"``
argument, which lets balance audits pair events without inferring the phase
from a velocity sign::

    wall_event_handler(wall, position, velocity, weight, wall_velocity)
    phase_event_handler(
        wall, phase, position, velocity, weight, wall_velocity
    )

Machine-precision corner ties are resolved deterministically in favour of the x wall.
If the sampled tangential velocity still points through the coincident y
wall, that zero-flight-time impact is handled on the next event iteration.
The per-particle event cap converts pathological corner cycling into an
explicit error; positions are never silently clipped back into the domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from .vhs_model import (
    PhysicalCavityConfig,
    PhysicalParticleState,
    _diffuse_wall,
)


WallPhase = Literal["incoming", "outgoing"]
LegacyWallEventHandler = Callable[
    [str, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    None,
]
PhaseWallEventHandler = Callable[
    [str, WallPhase, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    None,
]

WALL_ORDER = ("left", "right", "bottom", "top")
_WALL_INDEX = {wall: index for index, wall in enumerate(WALL_ORDER)}
_TEMPERATURE_ATTRIBUTE = {
    "left": "t_left",
    "right": "t_right",
    "bottom": "t_bottom",
    "top": "t_top",
}
_NORMAL_AXIS = {"left": 0, "right": 0, "bottom": 1, "top": 1}
_INWARD_SIGN = {"left": 1.0, "right": -1.0, "bottom": 1.0, "top": -1.0}


@dataclass(frozen=True)
class EventWallStreamingDiagnostics:
    """Checkpoint-friendly invariants returned by one streaming call."""

    particle_count: int
    total_wall_hits: int
    maximum_hits_on_one_particle: int
    particles_with_multiple_hits: int
    zero_time_wall_hits: int
    wall_hits: tuple[int, int, int, int]
    incident_counts: tuple[int, int, int, int]
    reflected_counts: tuple[int, int, int, int]
    incident_relative_weight: tuple[float, float, float, float]
    reflected_relative_weight: tuple[float, float, float, float]
    exact_corner_ties: int
    incident_sign_violations: int
    reflected_sign_violations: int
    nonmonotone_hit_time: int
    fallback_clip_count: int
    cap_exhaustion: int
    particle_count_delta: int
    relative_weight_delta: float

    def as_dict(self) -> dict[str, object]:
        """Return JSON-safe primitives for run summaries and checkpoints."""
        return {
            "particle_count": self.particle_count,
            "total_wall_hits": self.total_wall_hits,
            "maximum_hits_on_one_particle": self.maximum_hits_on_one_particle,
            "particles_with_multiple_hits": self.particles_with_multiple_hits,
            "multi_hit_particle_fraction": (
                self.particles_with_multiple_hits / self.particle_count
                if self.particle_count
                else 0.0
            ),
            "zero_time_wall_hits": self.zero_time_wall_hits,
            "wall_order": list(WALL_ORDER),
            "wall_hits": list(self.wall_hits),
            "incident_counts": list(self.incident_counts),
            "reflected_counts": list(self.reflected_counts),
            "incident_relative_weight": list(self.incident_relative_weight),
            "reflected_relative_weight": list(self.reflected_relative_weight),
            "exact_corner_ties": self.exact_corner_ties,
            "incident_sign_violations": self.incident_sign_violations,
            "reflected_sign_violations": self.reflected_sign_violations,
            "nonmonotone_hit_time": self.nonmonotone_hit_time,
            "fallback_clip_count": self.fallback_clip_count,
            "cap_exhaustion": self.cap_exhaustion,
            "particle_count_delta": self.particle_count_delta,
            "relative_weight_delta": self.relative_weight_delta,
        }


def _empty_diagnostics(particle_count: int) -> EventWallStreamingDiagnostics:
    zero_int = (0, 0, 0, 0)
    zero_float = (0.0, 0.0, 0.0, 0.0)
    return EventWallStreamingDiagnostics(
        particle_count=particle_count,
        total_wall_hits=0,
        maximum_hits_on_one_particle=0,
        particles_with_multiple_hits=0,
        zero_time_wall_hits=0,
        wall_hits=zero_int,
        incident_counts=zero_int,
        reflected_counts=zero_int,
        incident_relative_weight=zero_float,
        reflected_relative_weight=zero_float,
        exact_corner_ties=0,
        incident_sign_violations=0,
        reflected_sign_violations=0,
        nonmonotone_hit_time=0,
        fallback_clip_count=0,
        cap_exhaustion=0,
        particle_count_delta=0,
        relative_weight_delta=0.0,
    )


def _validate_inputs(
    state: PhysicalParticleState,
    cfg: PhysicalCavityConfig,
    dt: float,
    max_events_per_particle: int,
) -> None:
    position = np.asarray(state.pos)
    velocity = np.asarray(state.vel)
    weight = np.asarray(state.weight)
    if position.ndim != 2 or position.shape[1:] != (2,):
        raise ValueError("state.pos must have shape (particle_count, 2)")
    particle_count = len(position)
    if velocity.shape != (particle_count, 3):
        raise ValueError("state.vel must have shape (particle_count, 3)")
    if weight.shape != (particle_count,):
        raise ValueError("state.weight must have shape (particle_count,)")
    if not (
        np.issubdtype(position.dtype, np.floating)
        and np.issubdtype(velocity.dtype, np.floating)
    ):
        raise ValueError("state.pos and state.vel must be floating-point arrays")
    if not (
        np.all(np.isfinite(position))
        and np.all(np.isfinite(velocity))
        and np.all(np.isfinite(weight))
    ):
        raise ValueError("particle state must be finite")
    if np.any(weight <= 0.0):
        raise ValueError("relative particle weights must be positive")
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")
    if (
        isinstance(max_events_per_particle, (bool, np.bool_))
        or not isinstance(max_events_per_particle, (int, np.integer))
        or max_events_per_particle <= 0
    ):
        raise ValueError("max_events_per_particle must be a positive integer")
    if not np.isfinite(cfg.length) or cfg.length <= 0.0:
        raise ValueError("cfg.length must be finite and positive")
    temperatures = np.asarray(
        [cfg.t_left, cfg.t_right, cfg.t_bottom, cfg.t_top],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(temperatures)) or np.any(temperatures <= 0.0):
        raise ValueError("wall temperatures must be finite and positive")
    if not np.isfinite(cfg.vhs.mass) or cfg.vhs.mass <= 0.0:
        raise ValueError("VHS molecular mass must be finite and positive")
    if np.any(position < 0.0) or np.any(position > cfg.length):
        raise ValueError(
            "event-driven streaming requires initial positions inside the domain"
        )


def _notify(
    wall: str,
    phase: WallPhase,
    tangential_position: np.ndarray,
    velocity: np.ndarray,
    weight: np.ndarray,
    wall_velocity: np.ndarray,
    wall_event_handler: LegacyWallEventHandler | None,
    phase_event_handler: PhaseWallEventHandler | None,
) -> None:
    """Dispatch immutable snapshots so later state updates cannot alter audits."""
    if wall_event_handler is not None:
        wall_event_handler(
            wall,
            tangential_position.copy(),
            velocity.copy(),
            weight.copy(),
            wall_velocity.copy(),
        )
    if phase_event_handler is not None:
        phase_event_handler(
            wall,
            phase,
            tangential_position.copy(),
            velocity.copy(),
            weight.copy(),
            wall_velocity.copy(),
        )


def stream_with_diffuse_walls(
    state: PhysicalParticleState,
    cfg: PhysicalCavityConfig,
    dt: float,
    rng: np.random.Generator,
    wall_event_handler: LegacyWallEventHandler | None = None,
    *,
    phase_event_handler: PhaseWallEventHandler | None = None,
    max_events_per_particle: int = 32,
) -> EventWallStreamingDiagnostics:
    """Advance ``state`` through ``dt`` using exact wall-impact times.

    Parameters
    ----------
    state:
        Mutable physical particle state.  Particle arrays keep their shape and
        order; only positions and velocities are changed.
    cfg:
        Square-cavity geometry, temperatures, wall velocities, and VHS model.
    dt:
        Free-flight duration in seconds.  A hit exactly at ``dt`` is reflected
        and reported, even though no post-impact flight time remains.
    rng:
        Random generator used by the existing diffuse-wall sampler.
    wall_event_handler:
        Optional legacy five-argument callback, invoked for both phases.
    phase_event_handler:
        Optional six-argument callback with an explicit phase as its second
        argument.
    max_events_per_particle:
        Hard safety limit.  An attempted additional impact raises
        ``RuntimeError`` instead of clipping or discarding remaining flight.

    Notes
    -----
    A hard-abort exception is deliberately non-transactional: particles
    processed before the violated invariant may already be advanced.  The
    caller must discard that state rather than resume from it.
    """
    _validate_inputs(state, cfg, dt, max_events_per_particle)
    wall_velocity = {
        wall: cfg.resolved_wall_velocity(wall) for wall in WALL_ORDER
    }
    particle_count = len(state.pos)
    if particle_count == 0 or dt == 0.0:
        return _empty_diagnostics(particle_count)

    original_particle_count = particle_count
    original_relative_weight = float(np.sum(state.weight, dtype=np.float64))
    remaining = np.full(particle_count, float(dt), dtype=np.float64)
    event_count = np.zeros(particle_count, dtype=np.int64)
    last_hit_time = np.full(particle_count, -np.inf, dtype=np.float64)
    wall_hits = np.zeros(len(WALL_ORDER), dtype=np.int64)
    incident_counts = np.zeros(len(WALL_ORDER), dtype=np.int64)
    reflected_counts = np.zeros(len(WALL_ORDER), dtype=np.int64)
    incident_weight = np.zeros(len(WALL_ORDER), dtype=np.float64)
    reflected_weight = np.zeros(len(WALL_ORDER), dtype=np.float64)
    corner_ties = 0
    zero_time_hits = 0
    incident_sign_violations = 0
    reflected_sign_violations = 0
    nonmonotone_hit_time = 0
    length = float(cfg.length)

    while True:
        active = np.flatnonzero(remaining > 0.0)
        if len(active) == 0:
            break

        # Compute event times in float64 even for float32 state arrays.  If the
        # division were performed in float32 first, physically unequal wall
        # times could quantize to the same value and be mistaken for a corner.
        position = np.asarray(state.pos[active], dtype=np.float64)
        velocity = np.asarray(state.vel[active], dtype=np.float64)
        vx = velocity[:, 0]
        vy = velocity[:, 1]

        time_x = np.full(len(active), np.inf, dtype=np.float64)
        moving_left = vx < 0.0
        moving_right = vx > 0.0
        time_x[moving_left] = -position[moving_left, 0] / vx[moving_left]
        time_x[moving_right] = (
            length - position[moving_right, 0]
        ) / vx[moving_right]

        time_y = np.full(len(active), np.inf, dtype=np.float64)
        moving_bottom = vy < 0.0
        moving_top = vy > 0.0
        time_y[moving_bottom] = -position[moving_bottom, 1] / vy[moving_bottom]
        time_y[moving_top] = (
            length - position[moving_top, 1]
        ) / vy[moving_top]

        if np.any(time_x < 0.0) or np.any(time_y < 0.0):
            raise RuntimeError(
                "negative wall-hit time encountered; state was not clipped"
            )

        hit_time = np.minimum(time_x, time_y)
        has_hit = hit_time <= remaining[active]

        free_ids = active[~has_hit]
        if len(free_ids):
            state.pos[free_ids] += (
                state.vel[free_ids, :2] * remaining[free_ids, None]
            )
            remaining[free_ids] = 0.0

        if not np.any(has_hit):
            continue

        hit_local = np.flatnonzero(has_hit)
        hit_ids = active[hit_local]
        capped = event_count[hit_ids] >= max_events_per_particle
        if np.any(capped):
            particle = int(hit_ids[np.flatnonzero(capped)[0]])
            raise RuntimeError(
                "maximum wall events exceeded for particle "
                f"{particle}: cap={max_events_per_particle}; "
                "remaining flight was not clipped or discarded"
            )

        flight = hit_time[hit_local]
        zero_time_hits += int(np.count_nonzero(flight == 0.0))
        absolute_hit_time = float(dt) - remaining[hit_ids] + flight
        monotonic_tolerance = (
            64.0
            * np.finfo(np.float64).eps
            * max(float(dt), np.finfo(np.float64).tiny)
        )
        nonmonotone = absolute_hit_time < (
            last_hit_time[hit_ids] - monotonic_tolerance
        )
        nonmonotone_hit_time += int(np.count_nonzero(nonmonotone))
        if np.any(nonmonotone):
            raise RuntimeError("non-monotone per-particle wall-hit time")
        last_hit_time[hit_ids] = absolute_hit_time
        state.pos[hit_ids] += state.vel[hit_ids, :2] * flight[:, None]
        remaining[hit_ids] -= flight

        hit_time_x = time_x[hit_local]
        hit_time_y = time_y[hit_local]
        finite_pair = np.isfinite(hit_time_x) & np.isfinite(hit_time_y)
        time_scale = np.maximum(np.abs(hit_time_x), np.abs(hit_time_y))
        time_close = finite_pair & (
            np.abs(hit_time_x - hit_time_y)
            <= 64.0 * np.finfo(np.float64).eps * time_scale
        )
        spatial_tolerance = (
            64.0 * np.finfo(np.float64).eps * length
            + abs(float(np.spacing(length)))
        )
        x_boundary_residual = np.minimum(
            np.abs(state.pos[hit_ids, 0]),
            np.abs(state.pos[hit_ids, 0] - length),
        )
        y_boundary_residual = np.minimum(
            np.abs(state.pos[hit_ids, 1]),
            np.abs(state.pos[hit_ids, 1] - length),
        )
        geometric_corner = (
            (x_boundary_residual <= spatial_tolerance)
            & (y_boundary_residual <= spatial_tolerance)
        )
        corner = time_close & geometric_corner
        corner_ties += int(np.count_nonzero(corner))

        # For a true simultaneous hit, assigning both analytically known
        # boundary coordinates avoids manufacturing an O(ulp) overshoot.  It
        # is not a fallback clip: both the time and geometric tie tests above
        # must pass, including when the computed hit times compare equal.
        corner_ids = hit_ids[corner]
        if len(corner_ids):
            corner_vx = vx[hit_local][corner]
            corner_vy = vy[hit_local][corner]
            state.pos[corner_ids, 0] = np.where(corner_vx < 0.0, 0.0, length)
            state.pos[corner_ids, 1] = np.where(corner_vy < 0.0, 0.0, length)

        hit_x = (hit_time_x <= hit_time_y) | corner
        hit_wall = np.empty(len(hit_ids), dtype=object)
        hit_wall[hit_x & (vx[hit_local] < 0.0)] = "left"
        hit_wall[hit_x & (vx[hit_local] > 0.0)] = "right"
        hit_wall[~hit_x & (vy[hit_local] < 0.0)] = "bottom"
        hit_wall[~hit_x & (vy[hit_local] > 0.0)] = "top"

        # Batch in a stable wall/particle order.  Fix only the analytically
        # hit normal coordinate; tangential coordinates are never clipped.
        for wall in WALL_ORDER:
            selected = hit_wall == wall
            if not np.any(selected):
                continue
            ids = hit_ids[selected]
            axis = _NORMAL_AXIS[wall]
            state.pos[ids, axis] = (
                0.0 if wall in {"left", "bottom"} else length
            )
            tangent_axis = 1 - axis
            tangent = state.pos[ids, tangent_axis]
            if np.any(tangent < 0.0) or np.any(tangent > length):
                raise RuntimeError(
                    f"{wall} impact has an out-of-domain tangential coordinate; "
                    "position was not clipped"
                )

            incoming = state.vel[ids].copy()
            weights = state.weight[ids].copy()
            wall_u = wall_velocity[wall]
            inward = _INWARD_SIGN[wall]
            incoming_normal = (
                incoming[:, axis] - wall_u[axis]
            ) * inward
            bad_incoming = incoming_normal >= 0.0
            incident_sign_violations += int(np.count_nonzero(bad_incoming))
            if np.any(bad_incoming):
                raise RuntimeError(
                    f"{wall} incident event lacks negative wall-relative normal speed"
                )
            _notify(
                wall,
                "incoming",
                tangent,
                incoming,
                weights,
                wall_u,
                wall_event_handler,
                phase_event_handler,
            )
            reflected = _diffuse_wall(
                incoming,
                float(getattr(cfg, _TEMPERATURE_ATTRIBUTE[wall])),
                normal_axis=axis,
                inward_sign=_INWARD_SIGN[wall],
                model=cfg.vhs,
                rng=rng,
                wall_velocity=wall_u,
            )
            state.vel[ids] = reflected
            reflected_normal = (
                reflected[:, axis] - wall_u[axis]
            ) * inward
            bad_reflected = reflected_normal <= 0.0
            reflected_sign_violations += int(np.count_nonzero(bad_reflected))
            if np.any(bad_reflected):
                raise RuntimeError(
                    f"{wall} reflected event lacks positive wall-relative normal speed"
                )
            _notify(
                wall,
                "outgoing",
                tangent,
                reflected,
                weights,
                wall_u,
                wall_event_handler,
                phase_event_handler,
            )
            wall_index = _WALL_INDEX[wall]
            batch_count = len(ids)
            batch_weight = float(np.sum(weights, dtype=np.float64))
            wall_hits[wall_index] += batch_count
            incident_counts[wall_index] += batch_count
            reflected_counts[wall_index] += batch_count
            incident_weight[wall_index] += batch_weight
            reflected_weight[wall_index] += batch_weight

        event_count[hit_ids] += 1

    if not np.all(np.isfinite(state.pos)) or not np.all(np.isfinite(state.vel)):
        raise RuntimeError("streaming produced non-finite particle state")
    if np.any(state.pos < 0.0) or np.any(state.pos > length):
        raise RuntimeError(
            "event-driven streaming ended outside the domain; position was not clipped"
        )

    final_particle_count = len(state.pos)
    final_relative_weight = float(np.sum(state.weight, dtype=np.float64))
    return EventWallStreamingDiagnostics(
        particle_count=particle_count,
        total_wall_hits=int(np.sum(wall_hits)),
        maximum_hits_on_one_particle=int(np.max(event_count, initial=0)),
        particles_with_multiple_hits=int(np.count_nonzero(event_count > 1)),
        zero_time_wall_hits=zero_time_hits,
        wall_hits=tuple(int(value) for value in wall_hits),
        incident_counts=tuple(int(value) for value in incident_counts),
        reflected_counts=tuple(int(value) for value in reflected_counts),
        incident_relative_weight=tuple(float(value) for value in incident_weight),
        reflected_relative_weight=tuple(float(value) for value in reflected_weight),
        exact_corner_ties=corner_ties,
        incident_sign_violations=incident_sign_violations,
        reflected_sign_violations=reflected_sign_violations,
        nonmonotone_hit_time=nonmonotone_hit_time,
        fallback_clip_count=0,
        cap_exhaustion=0,
        particle_count_delta=final_particle_count - original_particle_count,
        relative_weight_delta=final_relative_weight - original_relative_weight,
    )
