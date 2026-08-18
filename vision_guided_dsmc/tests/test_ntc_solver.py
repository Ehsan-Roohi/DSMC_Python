import numpy as np

from vgdsmc.ntc_solver import (
    _cell_collision_statistics,
    collide_vhs_ntc,
    conserved_quantities_ntc,
    run_physical_cavity_ntc,
)
from vgdsmc.vhs_model import (
    PhysicalCavityConfig,
    PhysicalParticleState,
    initialize_physical_state,
)


def test_ntc_collisions_conserve_momentum_and_energy_without_clipping():
    cfg = PhysicalCavityConfig(
        nx=1,
        ny=1,
        particles_per_cell=96,
        knudsen=0.05,
        dt_safety=0.4,
        seed=71,
    )
    state = initialize_physical_state(cfg)
    before = conserved_quantities_ntc(state)
    diagnostics = collide_vhs_ntc(
        state,
        cfg,
        np.random.default_rng(72),
    )
    after = conserved_quantities_ntc(state)

    assert diagnostics.candidate_collisions > 0
    assert diagnostics.accepted_collisions > 0
    assert diagnostics.majorant_violations == 0
    assert diagnostics.max_acceptance_ratio <= 1.0 + 1.0e-12
    assert np.allclose(after[1], before[1], rtol=1.0e-12, atol=1.0e-10)
    assert np.isclose(after[2], before[2], rtol=1.0e-12, atol=0.0)


def test_ntc_rejects_unequal_weights_in_locked_benchmark():
    cfg = PhysicalCavityConfig(nx=1, ny=1, particles_per_cell=8)
    state = initialize_physical_state(cfg)
    state.weight[0] = 2.0
    with np.testing.assert_raises_regex(ValueError, "equal weights"):
        collide_vhs_ntc(state, cfg, np.random.default_rng(11))


def test_small_ntc_cavity_runs_with_auditable_majorant():
    cfg = PhysicalCavityConfig(
        nx=4,
        ny=4,
        particles_per_cell=16,
        steps=30,
        sample_start=15,
        knudsen=0.05,
        seed=73,
    )
    fields, state, diagnostics = run_physical_cavity_ntc(
        cfg,
        return_state=True,
    )

    assert fields["T"].shape == (4, 4)
    assert np.isfinite(fields["T"]).all()
    assert fields["T"].mean() > 0.0
    assert diagnostics["candidate_collisions"] > 0
    assert diagnostics["accepted_collisions"] > 0
    assert diagnostics["majorant_violations"] == 0
    assert diagnostics["max_acceptance_ratio"] <= 1.0 + 1.0e-12
    assert len(state.pos) == 4 * 4 * 16


def test_ntc_conserved_quantity_helper_handles_manual_state():
    state = PhysicalParticleState(
        pos=np.zeros((2, 2)),
        vel=np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]]),
        weight=np.ones(2),
    )
    mass, momentum, energy = conserved_quantities_ntc(state)
    assert mass == 2.0
    assert np.array_equal(momentum, np.array([0.0, 2.0, 7.0]))
    assert energy > 0.0


def test_ntc_collision_is_galilean_invariant_with_centered_majorant():
    cfg = PhysicalCavityConfig(
        nx=1,
        ny=1,
        particles_per_cell=64,
        knudsen=0.05,
        dt_safety=0.4,
        seed=99,
    )
    base = initialize_physical_state(cfg)
    shifted = base.copy()
    drift = np.array([800.0, -350.0, 125.0])
    shifted.vel += drift
    first = collide_vhs_ntc(base, cfg, np.random.default_rng(100))
    second = collide_vhs_ntc(shifted, cfg, np.random.default_rng(100))
    assert first == second
    assert np.allclose(shifted.vel - base.vel, drift, rtol=0.0, atol=1.0e-12)


def test_stratified_initialization_starts_with_exact_ppc_per_cell():
    cfg = PhysicalCavityConfig(
        nx=5,
        ny=4,
        particles_per_cell=7,
        stratified_initialization=True,
        seed=101,
    )
    state = initialize_physical_state(cfg)
    ix = (state.pos[:, 0] / cfg.length * cfg.nx).astype(int)
    iy = (state.pos[:, 1] / cfg.length * cfg.ny).astype(int)
    counts = np.bincount(iy * cfg.nx + ix, minlength=cfg.nx * cfg.ny)
    assert np.array_equal(counts, np.full(cfg.nx * cfg.ny, 7))


def test_vectorized_ntc_cell_energy_matches_direct_centered_reduction():
    cfg = PhysicalCavityConfig(
        nx=5,
        ny=4,
        particles_per_cell=7,
        stratified_initialization=True,
        seed=131,
    )
    state = initialize_physical_state(cfg)
    # Exercise unequal occupancy while retaining uniform weights in every cell.
    state.pos[:9, 0] = 0.99 * cfg.length
    ix = (state.pos[:, 0] / cfg.length * cfg.nx).astype(int)
    iy = (state.pos[:, 1] / cfg.length * cfg.ny).astype(int)
    cell = iy * cfg.nx + ix
    order = np.argsort(cell)
    sorted_cell = cell[order]
    cells = np.arange(cfg.nx * cfg.ny)
    starts = np.searchsorted(sorted_cell, cells, side="left")
    ends = np.searchsorted(sorted_cell, cells, side="right")

    counts, reference_weights, energy = _cell_collision_statistics(
        state,
        cfg,
        cell,
        order,
        starts,
        ends,
    )
    direct = np.zeros_like(energy)
    for cid, (start, end) in enumerate(zip(starts, ends)):
        velocity = state.vel[order[start:end]]
        if len(velocity):
            direct[cid] = np.sum((velocity - velocity.mean(axis=0)) ** 2)

    assert np.array_equal(counts, ends - starts)
    assert np.array_equal(reference_weights, np.ones_like(reference_weights))
    assert np.allclose(energy, direct, rtol=5.0e-15, atol=1.0e-10)
