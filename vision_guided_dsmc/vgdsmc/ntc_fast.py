from __future__ import annotations

import numpy as np

from .ntc_solver import (
    NTCCollisionDiagnostics,
    _cell_collision_statistics,
)
from .vhs_model import PhysicalCavityConfig, PhysicalParticleState, _cell_xy


def _scatter_disjoint_pairs(
    velocity: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    relative_speed: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Scatter pairwise-disjoint equal-mass pairs in one vector operation."""
    pair_count = len(first)
    if pair_count == 0:
        return

    center = 0.5 * (velocity[first] + velocity[second])
    cos_chi = 2.0 * rng.random(pair_count) - 1.0
    sin_chi = np.sqrt(np.maximum(0.0, 1.0 - cos_chi**2))
    phi = 2.0 * np.pi * rng.random(pair_count)
    relative_new = relative_speed[:, None] * np.column_stack(
        (
            sin_chi * np.cos(phi),
            sin_chi * np.sin(phi),
            cos_chi,
        )
    )
    velocity[first] = center + 0.5 * relative_new
    velocity[second] = center - 0.5 * relative_new


def collide_vhs_ntc_fast(
    state: PhysicalParticleState,
    cfg: PhysicalCavityConfig,
    rng: np.random.Generator,
    *,
    strict_majorant: bool = True,
) -> NTCCollisionDiagnostics:
    """Vectorized Bird NTC collisions with the reference solver's physics.

    Candidate counts use the same stochastic-rounding formula as
    :func:`vgdsmc.ntc_solver.collide_vhs_ntc`.  Candidate ``k`` from every
    active cell is processed in the same vectorized round.  There is at most
    one pair per cell in a round, so all pairs are disjoint and their elastic
    scatters commute.  Later rounds see the velocities produced by earlier
    rounds, retaining the sequential NTC semantics for cells with multiple
    candidates.

    The random stream is deterministic for this backend but deliberately not
    bitwise-identical to the scalar reference backend: random draws are batched
    by operation and round rather than interleaved cell by cell.
    """
    if not 0.0 < cfg.vhs.omega <= 1.0:
        raise ValueError(
            "The energy-bound NTC majorant requires 0 < VHS omega <= 1"
        )

    ix, iy = _cell_xy(state.pos, cfg)
    cell = iy * cfg.nx + ix
    order = np.argsort(cell)
    sorted_cell = cell[order]
    cell_ids = np.arange(cfg.nx * cfg.ny)
    starts = np.searchsorted(sorted_cell, cell_ids, side="left")
    ends = np.searchsorted(sorted_cell, cell_ids, side="right")

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

    active_cells = np.flatnonzero((counts >= 2) & (energy_bounds > 0.0))
    expected_active = expected_candidates[active_cells]
    if not np.all(np.isfinite(expected_active)) or np.any(expected_active < 0.0):
        raise ValueError("NTC expected candidate counts must be finite and nonnegative")
    if np.any(expected_active > np.iinfo(np.int64).max):
        raise OverflowError("NTC expected candidate count exceeds int64 capacity")

    base_counts = np.floor(expected_active).astype(np.int64)
    candidate_counts = base_counts + (
        rng.random(len(active_cells)) < (expected_active - base_counts)
    )
    candidate_total = int(np.sum(candidate_counts, dtype=np.int64))
    accepted_total = 0
    violation_total = 0
    max_ratio = 0.0

    max_rounds = int(candidate_counts.max(initial=0))
    for round_index in range(max_rounds):
        round_cells = active_cells[candidate_counts > round_index]
        round_size = len(round_cells)
        round_counts = counts[round_cells]

        local_first = rng.integers(0, round_counts, size=round_size)
        local_second = rng.integers(0, round_counts - 1, size=round_size)
        local_second += local_second >= local_first
        first = order[starts[round_cells] + local_first]
        second = order[starts[round_cells] + local_second]

        relative = state.vel[first] - state.vel[second]
        relative_speed = np.sqrt(
            np.einsum("ij,ij->i", relative, relative)
        )
        collision_rates = (
            cfg.vhs.cross_section(relative_speed) * relative_speed
        )
        ratios = collision_rates / majorants[round_cells]
        if len(ratios):
            max_ratio = max(max_ratio, float(np.max(ratios)))

        violated = ratios > 1.0 + 1.0e-12
        violations = int(np.count_nonzero(violated))
        violation_total += violations
        if strict_majorant and violations:
            raise RuntimeError(
                "NTC majorant was violated; refusing probability clipping"
            )

        eligible = relative_speed > 1.0e-14
        accepted = np.zeros(round_size, dtype=bool)
        eligible_count = int(np.count_nonzero(eligible))
        accepted[eligible] = rng.random(eligible_count) < np.minimum(
            ratios[eligible],
            1.0,
        )
        accepted_total += int(np.count_nonzero(accepted))
        _scatter_disjoint_pairs(
            state.vel,
            first[accepted],
            second[accepted],
            relative_speed[accepted],
            rng,
        )

    return NTCCollisionDiagnostics(
        candidate_collisions=candidate_total,
        accepted_collisions=accepted_total,
        majorant_violations=violation_total,
        max_acceptance_ratio=max_ratio,
    )
