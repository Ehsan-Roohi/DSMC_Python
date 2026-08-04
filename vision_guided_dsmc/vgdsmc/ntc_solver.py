from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .moment_sampling import PhysicalMomentAccumulator
from .wall_sampling import LidWallEventAccumulator
from .vhs_model import (
    PhysicalCavityConfig,
    PhysicalParticleState,
    VHSModel,
    _cell_xy,
    apply_diffuse_walls,
    initialize_physical_state,
)


@dataclass(frozen=True)
class NTCCollisionDiagnostics:
    candidate_collisions: int = 0
    accepted_collisions: int = 0
    majorant_violations: int = 0
    max_acceptance_ratio: float = 0.0


def _equal_mass_elastic_scatter(
    velocity: np.ndarray,
    a: int,
    b: int,
    rng: np.random.Generator,
) -> None:
    """Scatter an equal-mass pair isotropically in its center-of-mass frame."""
    center = 0.5 * (velocity[a] + velocity[b])
    relative = velocity[a] - velocity[b]
    speed = float(np.linalg.norm(relative))
    if speed <= 0.0:
        return
    cos_chi = 2.0 * rng.random() - 1.0
    sin_chi = np.sqrt(max(0.0, 1.0 - cos_chi**2))
    phi = 2.0 * np.pi * rng.random()
    relative_new = speed * np.array(
        [
            sin_chi * np.cos(phi),
            sin_chi * np.sin(phi),
            cos_chi,
        ]
    )
    velocity[a] = center + 0.5 * relative_new
    velocity[b] = center - 0.5 * relative_new


def _stochastic_count(value: float, rng: np.random.Generator) -> int:
    base = int(np.floor(value))
    return base + int(rng.random() < value - base)


def _cell_collision_statistics(
    state: PhysicalParticleState,
    cfg: PhysicalCavityConfig,
    cell: np.ndarray,
    order: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return count, reference weight, and centered energy for every cell.

    Computing these quantities one cell at a time creates tens of thousands of
    tiny temporary arrays on the publication grid.  The reductions below are
    algebraically identical, while retaining the stable two-pass centered
    energy used by the scalar implementation.  The collision loop still visits
    cells and consumes random numbers in canonical cell order.
    """
    ncell = cfg.nx * cfg.ny
    counts = ends - starts
    reference_weights = np.ones(ncell, dtype=np.float64)

    # The locked benchmark always follows this fast path.  Retain the more
    # general per-cell check so callers with different (but cellwise uniform)
    # weights keep the previous behavior and error semantics.
    globally_equal = (
        len(state.weight) == 0
        or bool(np.all(state.weight == state.weight[0]))
    )
    if len(state.weight) and globally_equal:
        reference_weights.fill(float(state.weight[0]))
    elif not globally_equal:
        for start, end in zip(starts, ends):
            if start == end:
                continue
            ids = order[start:end]
            relative_weights = state.weight[ids]
            reference_weight = float(relative_weights[0])
            if not np.allclose(
                relative_weights,
                reference_weight,
                rtol=1.0e-12,
                atol=1.0e-14,
            ):
                raise ValueError(
                    "The locked NTC benchmark requires equal weights in each cell"
                )
            reference_weights[int(cell[ids[0]])] = reference_weight

    inverse_count = np.divide(
        1.0,
        counts,
        out=np.zeros(ncell, dtype=np.float64),
        where=counts > 0,
    )
    vx, vy, vz = state.vel.T
    mean_x = np.bincount(cell, weights=vx, minlength=ncell) * inverse_count
    mean_y = np.bincount(cell, weights=vy, minlength=ncell) * inverse_count
    mean_z = np.bincount(cell, weights=vz, minlength=ncell) * inverse_count
    centered_speed2 = (
        (vx - mean_x[cell]) ** 2
        + (vy - mean_y[cell]) ** 2
        + (vz - mean_z[cell]) ** 2
    )
    energy_bound = np.bincount(
        cell,
        weights=centered_speed2,
        minlength=ncell,
    )
    return counts, reference_weights, energy_bound


def collide_vhs_ntc(
    state: PhysicalParticleState,
    cfg: PhysicalCavityConfig,
    rng: np.random.Generator,
    *,
    strict_majorant: bool = True,
) -> NTCCollisionDiagnostics:
    """Apply Bird's no-time-counter collision selection in every cell.

    The benchmark path deliberately requires equal particle weights.  A safe
    per-cell majorant is formed from the cell's centered kinetic energy:
    ``g <= sqrt(2 * sum(|v - mean(v)|**2))``.  Elastic collisions preserve
    that bound throughout the collision phase, so acceptance probabilities
    must not be clipped.  Any violation is either raised immediately or
    reported.
    """
    if not 0.0 < cfg.vhs.omega <= 1.0:
        raise ValueError(
            "The energy-bound NTC majorant requires 0 < VHS omega <= 1"
        )
    ix, iy = _cell_xy(state.pos, cfg)
    cell = iy * cfg.nx + ix
    order = np.argsort(cell)
    sorted_cell = cell[order]
    starts = np.searchsorted(
        sorted_cell,
        np.arange(cfg.nx * cfg.ny),
        side="left",
    )
    ends = np.searchsorted(
        sorted_cell,
        np.arange(cfg.nx * cfg.ny),
        side="right",
    )

    counts, reference_weights, energy_bounds = _cell_collision_statistics(
        state,
        cfg,
        cell,
        order,
        starts,
        ends,
    )
    relative_speed_bounds = np.sqrt(2.0 * energy_bounds)
    majorants = (
        cfg.vhs.cross_section(relative_speed_bounds)
        * relative_speed_bounds
    )
    expected_candidates = (
        0.5
        * counts
        * (counts - 1)
        * cfg.real_particles_per_sim_particle
        * reference_weights
        * cfg.dt
        * majorants
        / cfg.cell_volume
    )

    candidates = 0
    accepted = 0
    violations = 0
    max_ratio = 0.0

    active_cells = np.flatnonzero((counts >= 2) & (energy_bounds > 0.0))
    for cid in active_cells:
        start, end = int(starts[cid]), int(ends[cid])
        ids = order[start:end]
        count = int(counts[cid])
        majorant = float(majorants[cid])
        cell_candidates = _stochastic_count(
            float(expected_candidates[cid]),
            rng,
        )
        candidates += cell_candidates

        for _ in range(cell_candidates):
            local_a = int(rng.integers(0, count))
            local_b = int(rng.integers(0, count - 1))
            if local_b >= local_a:
                local_b += 1
            a = int(ids[local_a])
            b = int(ids[local_b])
            relative_speed = float(np.linalg.norm(state.vel[a] - state.vel[b]))
            if relative_speed <= 1.0e-14:
                continue
            collision_rate = (
                cfg.vhs.cross_section(relative_speed) * relative_speed
            )
            ratio = float(collision_rate / majorant)
            max_ratio = max(max_ratio, ratio)
            if ratio > 1.0 + 1.0e-12:
                violations += 1
                if strict_majorant:
                    raise RuntimeError(
                        "NTC majorant was violated; refusing probability clipping"
                    )
            if rng.random() < min(ratio, 1.0):
                _equal_mass_elastic_scatter(state.vel, a, b, rng)
                accepted += 1

    return NTCCollisionDiagnostics(
        candidate_collisions=candidates,
        accepted_collisions=accepted,
        majorant_violations=violations,
        max_acceptance_ratio=max_ratio,
    )


def advance_physical_state_ntc(
    state: PhysicalParticleState,
    cfg: PhysicalCavityConfig,
    steps: int,
    sample_start: int = 0,
    seed: int | None = None,
) -> tuple[dict[str, np.ndarray], PhysicalParticleState, dict[str, float]]:
    if steps <= 0 or not 0 <= sample_start < steps:
        raise ValueError("Require steps > 0 and 0 <= sample_start < steps")
    rng = np.random.default_rng(cfg.seed + 130363 if seed is None else seed)
    moments = PhysicalMomentAccumulator(cfg)
    wall_events = LidWallEventAccumulator(cfg)
    temporal_sums: dict[str, np.ndarray] = {}
    sums2: dict[str, np.ndarray] = {}
    nsamples = 0
    candidates = 0
    accepted = 0
    violations = 0
    max_ratio = 0.0

    for step in range(steps):
        state.pos += state.vel[:, :2] * cfg.dt
        apply_diffuse_walls(
            state,
            cfg,
            rng,
            wall_event_handler=(wall_events.add if step >= sample_start else None),
        )
        collision = collide_vhs_ntc(state, cfg, rng)
        candidates += collision.candidate_collisions
        accepted += collision.accepted_collisions
        violations += collision.majorant_violations
        max_ratio = max(max_ratio, collision.max_acceptance_ratio)
        if step >= sample_start:
            instantaneous = moments.add(
                state,
                return_instantaneous=True,
            )
            assert instantaneous is not None
            # These instantaneous estimates are retained only to describe
            # temporal scatter.  Reported means and heat fluxes come from the
            # raw-moment accumulator below.
            for key in ("T", "u", "v", "w"):
                value = instantaneous[key]
                temporal_sums[key] = (
                    temporal_sums.get(key, np.zeros_like(value)) + value
                )
                sums2[key] = sums2.get(key, np.zeros_like(value)) + value**2
            nsamples += 1

    out = moments.finalize()
    out.update(wall_events.finalize())
    for key in ("T", "u", "v", "w"):
        # This is the RMS instantaneous fluctuation around the long-sample
        # raw-moment mean.  It is not divided by sqrt(N); independent-seed
        # uncertainty is handled by the benchmark runner.
        out[f"sigma_{key}"] = np.sqrt(
            np.maximum(
                sums2[key] / nsamples
                - (temporal_sums[key] / nsamples) ** 2,
                0.0,
            )
        )
    diagnostics = {
        "candidate_collisions": float(candidates),
        "accepted_collisions": float(accepted),
        "acceptance_fraction": float(accepted / max(candidates, 1)),
        "majorant_violations": float(violations),
        "max_acceptance_ratio": float(max_ratio),
        "collisions_per_particle_step": float(
            accepted / (len(state.pos) * steps)
        ),
        "dt": cfg.dt,
        "number_density": cfg.number_density,
        "mean_free_path": cfg.vhs.mean_free_path(
            cfg.number_density,
            cfg.t0,
        ),
    }
    return out, state, diagnostics


def run_physical_cavity_ntc(
    cfg: PhysicalCavityConfig,
    initial_state: PhysicalParticleState | None = None,
    return_state: bool = False,
):
    state = (
        initialize_physical_state(cfg)
        if initial_state is None
        else initial_state.copy()
    )
    fields, state, diagnostics = advance_physical_state_ntc(
        state,
        cfg,
        cfg.steps,
        cfg.sample_start,
        seed=cfg.seed + 29,
    )
    if return_state:
        return fields, state, diagnostics
    return fields


def conserved_quantities_ntc(
    state: PhysicalParticleState,
    model: VHSModel = VHSModel(),
) -> tuple[float, np.ndarray, float]:
    mass = float(np.sum(state.weight))
    momentum = np.sum(state.weight[:, None] * state.vel, axis=0)
    energy = 0.5 * model.mass * float(
        np.sum(state.weight * np.sum(state.vel**2, axis=1))
    )
    return mass, momentum, energy
