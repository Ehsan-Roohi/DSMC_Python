import numpy as np

from vgdsmc.simulator import CavityConfig, ParticleState, _apply_walls
from vgdsmc.vhs_model import (
    PhysicalCavityConfig,
    PhysicalParticleState,
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
