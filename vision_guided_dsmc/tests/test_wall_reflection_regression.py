import numpy as np

from vgdsmc.simulator import CavityConfig, ParticleState, _apply_walls
from vgdsmc.vhs_model import (
    KB,
    MASS_AR,
    PhysicalCavityConfig,
    PhysicalParticleState,
    VHSModel,
    _diffuse_wall,
    apply_diffuse_walls,
)


def test_nondimensional_advanced_index_wall_updates_velocity():
    cfg = CavityConfig(t_left=4.0)
    state = ParticleState(
        pos=np.array([[-0.01, 0.5]]),
        vel=np.array([[-9.0, 9.0]]),
        weight=np.ones(1),
    )
    _apply_walls(state, cfg, np.random.default_rng(1))
    assert state.vel[0, 0] > 0.0
    assert not np.allclose(state.vel[0], [-9.0, 9.0])


def test_physical_advanced_index_wall_updates_velocity():
    cfg = PhysicalCavityConfig(t_left=500.0)
    state = PhysicalParticleState(
        pos=np.array([[-1.0e-9, 0.5 * cfg.length]]),
        vel=np.array([[-1000.0, 1000.0, 1000.0]]),
        weight=np.ones(1),
    )
    apply_diffuse_walls(state, cfg, np.random.default_rng(2))
    assert state.vel[0, 0] > 0.0
    assert not np.allclose(
        state.vel[0],
        [-1000.0, 1000.0, 1000.0],
    )


def test_stationary_diffuse_wall_preserves_legacy_sampling_sequence():
    count = 4096
    temperature = 375.0
    model = VHSModel()
    expected_rng = np.random.default_rng(171)
    sigma = np.sqrt(KB * temperature / MASS_AR)
    expected = np.empty((count, 3))
    expected[:, [0, 2]] = expected_rng.normal(
        0.0,
        sigma,
        size=(count, 2),
    )
    uniform = np.maximum(expected_rng.random(count), 1.0e-14)
    expected[:, 1] = -sigma * np.sqrt(-2.0 * np.log(uniform))

    reflected = _diffuse_wall(
        np.zeros((count, 3)),
        temperature,
        normal_axis=1,
        inward_sign=-1.0,
        model=model,
        rng=np.random.default_rng(171),
    )

    assert np.array_equal(reflected, expected)


def test_moving_wall_half_range_is_sampled_in_wall_frame():
    count = 200_000
    temperature = 300.0
    model = VHSModel()
    wall_velocity = np.array([125.0, 0.0, -35.0])
    reflected = _diffuse_wall(
        np.zeros((count, 3)),
        temperature,
        normal_axis=1,
        inward_sign=-1.0,
        model=model,
        rng=np.random.default_rng(8128),
        wall_velocity=wall_velocity,
    )
    relative = reflected - wall_velocity
    sigma = np.sqrt(KB * temperature / model.mass)

    assert np.all(relative[:, 1] < 0.0)
    assert np.allclose(
        reflected[:, [0, 2]].mean(axis=0),
        wall_velocity[[0, 2]],
        atol=2.5,
    )
    assert np.isclose(relative[:, 0].var(), sigma**2, rtol=0.01)
    assert np.isclose(relative[:, 2].var(), sigma**2, rtol=0.01)
    assert np.isclose(
        np.mean(relative[:, 1]),
        -sigma * np.sqrt(np.pi / 2.0),
        rtol=0.01,
    )
    assert np.isclose(np.mean(relative[:, 1] ** 2), 2.0 * sigma**2, rtol=0.01)


def test_lid_velocity_alias_resolves_to_top_wall_x_component():
    cfg = PhysicalCavityConfig(
        lid_velocity_x=100.0,
        left_wall_velocity=(0.0, 4.0, -2.0),
    )
    assert np.array_equal(
        cfg.resolved_wall_velocity("top"),
        np.array([100.0, 0.0, 0.0]),
    )
    assert np.array_equal(
        cfg.resolved_wall_velocity("left"),
        np.array([0.0, 4.0, -2.0]),
    )


def test_apply_diffuse_walls_passes_lid_velocity_to_top_reflection():
    cfg = PhysicalCavityConfig(lid_velocity_x=100.0)
    state = PhysicalParticleState(
        pos=np.array([[0.5 * cfg.length, cfg.length + 1.0e-9]]),
        vel=np.array([[0.0, 500.0, 0.0]]),
        weight=np.ones(1),
    )
    expected = _diffuse_wall(
        state.vel,
        cfg.t_top,
        normal_axis=1,
        inward_sign=-1.0,
        model=cfg.vhs,
        rng=np.random.default_rng(991),
        wall_velocity=np.array([100.0, 0.0, 0.0]),
    )

    apply_diffuse_walls(state, cfg, np.random.default_rng(991))

    assert np.array_equal(state.vel, expected)
    assert 0.0 <= state.pos[0, 1] < cfg.length
