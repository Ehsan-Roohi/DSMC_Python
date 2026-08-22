import numpy as np

from vgdsmc.simulator import (
    CavityConfig,
    ParticleState,
    _apply_walls,
)


def test_nondimensional_diffuse_wall_updates_original_velocity_array():
    cfg = CavityConfig(nx=1, ny=1, particles_per_cell=1)
    state = ParticleState(
        pos=np.array([[-0.1, 0.5]]),
        vel=np.zeros((1, 2)),
        weight=np.ones(1),
    )
    _apply_walls(state, cfg, np.random.default_rng(19))
    assert state.pos[0, 0] >= 0.0
    assert state.vel[0, 0] > 0.0
    assert np.linalg.norm(state.vel[0]) > 0.0
