import numpy as np

from vgdsmc.physical_adaptive import (
    adaptation_target,
    conservative_reallocate,
    exact_budget_ppc,
    gradient_priority,
)
from vgdsmc.sbt_solver import (
    _weighted_elastic_scatter,
    conserved_quantities,
    equalize_cell_weights,
    run_physical_cavity,
)
from vgdsmc.vhs_model import PhysicalCavityConfig, PhysicalParticleState, VHSModel


def test_vhs_cross_section_decreases_with_speed():
    model = VHSModel()
    speeds = np.array([100.0, 500.0, 1500.0])
    sigma = model.cross_section(speeds)
    assert np.all(np.diff(sigma) < 0.0)
    assert np.all(sigma > 0.0)
    assert np.isclose(model.gamma_factor, 0.9067818160205746, rtol=1.0e-13)


def test_weighted_scatter_conserves_momentum_and_energy():
    rng = np.random.default_rng(3)
    velocity = rng.normal(size=(2, 3)) * 400.0
    weight = np.array([0.4, 2.3])
    state = PhysicalParticleState(
        np.zeros((2, 2)),
        velocity.copy(),
        weight.copy(),
    )
    before = conserved_quantities(state)
    _weighted_elastic_scatter(state.vel, state.weight, 0, 1, rng)
    after = conserved_quantities(state)
    assert np.allclose(after[1], before[1], rtol=1.0e-12, atol=1.0e-12)
    assert np.isclose(after[2], before[2], rtol=1.0e-12, atol=0.0)


def test_physical_cavity_runs_and_accepts_collisions():
    cfg = PhysicalCavityConfig(
        nx=5,
        ny=5,
        particles_per_cell=8,
        steps=30,
        sample_start=15,
        knudsen=0.05,
        seed=5,
    )
    fields, state, diagnostics = run_physical_cavity(cfg, return_state=True)
    assert fields["T"].shape == (5, 5)
    assert np.isfinite(fields["T"]).all()
    assert fields["T"].mean() > 0.0
    assert diagnostics["accepted_collisions"] > 0
    assert len(state.pos) == 5 * 5 * 8


def test_conservative_reallocation_and_exact_budget():
    cfg = PhysicalCavityConfig(
        nx=4,
        ny=4,
        particles_per_cell=8,
        steps=10,
        sample_start=5,
        seed=9,
    )
    _, state, _ = run_physical_cavity(cfg, return_state=True)
    before = conserved_quantities(state)
    fields = {
        "T": np.arange(16, dtype=float).reshape(4, 4) + 300.0,
        "sigma_T": np.ones((4, 4)),
        "rho": np.ones((4, 4)),
    }
    target = exact_budget_ppc(
        gradient_priority(fields),
        base_ppc=8,
        budget_ratio=1.25,
    )
    new_state = conservative_reallocate(state, cfg, target, seed=18)
    after = conserved_quantities(new_state)
    assert len(new_state.pos) == int(target.sum())
    assert np.isclose(after[0], before[0], rtol=1.0e-12)
    assert np.allclose(after[1], before[1], rtol=1.0e-10, atol=1.0e-8)
    assert np.isclose(after[2], before[2], rtol=1.0e-10)


def test_noise_gate_returns_uniform_allocation():
    cfg = PhysicalCavityConfig(nx=3, ny=3, particles_per_cell=10)
    fields = {
        "T": np.full((3, 3), 300.0),
        "sigma_T": np.full((3, 3), 150.0),
        "rho": np.ones((3, 3)),
    }
    target, decision = adaptation_target(fields, cfg)
    assert not decision["adapted"]
    assert np.all(target == 10)


def test_cell_weight_equalization_conserves_invariants():
    cfg = PhysicalCavityConfig(nx=1, ny=1, particles_per_cell=4)
    rng = np.random.default_rng(12)
    state = PhysicalParticleState(
        pos=rng.random((4, 2)) * cfg.length,
        vel=rng.normal(size=(4, 3)) * 300.0,
        weight=np.array([0.2, 0.7, 1.4, 2.1]),
    )
    before = conserved_quantities(state)
    equalize_cell_weights(state, cfg, rng)
    after = conserved_quantities(state)
    assert np.allclose(state.weight, state.weight[0])
    assert np.isclose(after[0], before[0], rtol=1.0e-12)
    assert np.allclose(after[1], before[1], rtol=1.0e-10, atol=1.0e-9)
    assert np.isclose(after[2], before[2], rtol=1.0e-10)
