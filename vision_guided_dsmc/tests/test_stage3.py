import numpy as np

from vgdsmc.adaptive import conservative_reallocate_state
from vgdsmc.model import build_unet
from vgdsmc.simulator import CavityConfig, ParticleState, sample_state


def test_unet_accepts_non_multiple_of_four():
    import torch

    model = build_unet()
    output = model(torch.zeros(1, 4, 6, 10))
    assert output.shape == (1, 3, 6, 10)


def test_conservative_weighted_reallocation():
    cfg = CavityConfig(nx=2, ny=2)
    rng = np.random.default_rng(5)
    positions = []
    velocities = []
    weights = []
    for j in range(2):
        for i in range(2):
            positions.append(
                np.column_stack(((i + rng.random(8)) / 2, (j + rng.random(8)) / 2))
            )
            velocities.append(rng.normal(size=(8, 2)))
            weights.append(rng.uniform(0.5, 1.5, 8))
    state = ParticleState(np.vstack(positions), np.vstack(velocities), np.concatenate(weights))
    target = np.array([[4, 12], [6, 10]])
    new_state, report = conservative_reallocate_state(state, target, rng)
    assert len(new_state.pos) == int(target.sum())
    assert report.empty_cells == 0
    assert report.max_mass_relative_error < 1.0e-12
    assert report.max_momentum_absolute_error < 1.0e-10
    assert report.max_energy_relative_error < 1.0e-10
    old = sample_state(state, cfg)
    new = sample_state(new_state, cfg)
    np.testing.assert_allclose(old["mass"], new["mass"], rtol=1.0e-12, atol=1.0e-12)


def test_score_allocation_exact_budget():
    from vgdsmc.adaptive import score_to_target_ppc

    score = np.arange(12, dtype=float).reshape(3, 4)
    target = score_to_target_ppc(score, base_ppc=20, budget_ratio=1.25, alpha=0.25)
    assert int(target.sum()) == 300
    assert target[-1, -1] > target[0, 0]


def test_physics_vision_score_range():
    from vgdsmc.vision import physics_vision_score

    shape = (6, 7)
    fields = {
        "T": np.tile(np.linspace(0.9, 1.1, shape[1]), (shape[0], 1)),
        "u": np.zeros(shape),
        "v": np.zeros(shape),
        "rho": np.ones(shape),
        "sigma_T": np.full(shape, 0.02),
    }
    score = physics_vision_score(fields, "temperature_gradient")
    assert score.shape == shape
    assert np.all((score >= 0.0) & (score <= 1.0))


def test_tiny_vision_closed_loop(tmp_path):
    from vgdsmc.closed_loop import run_vision_closed_loop

    cfg = CavityConfig(nx=4, ny=4, particles_per_cell=4, steps=12, sample_start=6, seed=4)
    result = run_vision_closed_loop(
        cfg,
        reference_ppc=8,
        continuation_steps=8,
        budget_ratio=1.25,
        output=tmp_path / "tiny.json",
    )
    assert result["score_source"].startswith("physics_vision")
    assert result["conservation"]["max_energy_relative_error"] < 1.0e-10
    assert (tmp_path / "tiny.json").exists()
