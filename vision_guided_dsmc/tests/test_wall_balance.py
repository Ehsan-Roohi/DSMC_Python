import numpy as np

from vgdsmc.vhs_model import (
    PhysicalCavityConfig,
    PhysicalParticleState,
    apply_diffuse_walls,
)
from vgdsmc.wall_balance import WALLS, WallBalanceAccumulator


def test_wall_balance_classifies_all_normals_and_uses_represented_weight():
    cfg = PhysicalCavityConfig(nx=2, ny=2, particles_per_cell=1)
    audit = WallBalanceAccumulator(cfg)
    wall_velocity = np.zeros(3)
    cases = {
        "left": ([-2.0, -3.0], [4.0, 5.0]),
        "right": ([2.0, 3.0], [-4.0, -5.0]),
        "bottom": ([-6.0, -7.0], [8.0, 9.0]),
        "top": ([6.0, 7.0], [-8.0, -9.0]),
    }
    for wall, (incoming_normal, outgoing_normal) in cases.items():
        axis = 0 if wall in {"left", "right"} else 1
        incoming = np.zeros((2, 3))
        outgoing = np.zeros((2, 3))
        incoming[:, axis] = incoming_normal
        outgoing[:, axis] = outgoing_normal
        audit.add(wall, np.array([0.2, 0.8]), incoming, np.array([1.0, 2.0]), wall_velocity)
        audit.add(wall, np.array([0.2, 0.8]), outgoing, np.array([1.0, 2.0]), wall_velocity)

    summary = audit.finalize()
    represented = 3.0 * cfg.real_particles_per_sim_particle
    assert summary["wall_order"] == WALLS
    assert np.array_equal(summary["incoming_count"], np.full(4, 2))
    assert np.array_equal(summary["outgoing_count"], np.full(4, 2))
    assert np.allclose(summary["incoming_represented_weight"], represented)
    assert np.allclose(summary["outgoing_represented_weight"], represented)
    assert np.array_equal(summary["per_wall_relative_net_mass_imbalance"], np.zeros(4))
    assert summary["relative_net_mass_imbalance"] == 0.0


def test_wall_balance_integrates_with_current_double_callback():
    cfg = PhysicalCavityConfig(nx=2, ny=2, particles_per_cell=1, seed=730)
    state = PhysicalParticleState(
        pos=np.array(
            [
                [-0.01, 0.2],
                [1.01, 0.3],
                [0.4, -0.01],
                [0.6, 1.01],
            ]
        )
        * cfg.length,
        vel=np.array(
            [
                [-100.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [0.0, -100.0, 0.0],
                [0.0, 100.0, 0.0],
            ]
        ),
        weight=np.array([0.5, 1.0, 1.5, 2.0]),
    )
    audit = WallBalanceAccumulator(cfg)
    apply_diffuse_walls(state, cfg, np.random.default_rng(731), audit.add)
    summary = audit.finalize()

    assert np.array_equal(summary["incoming_count"], np.ones(4))
    assert np.array_equal(summary["outgoing_count"], np.ones(4))
    assert np.array_equal(summary["net_represented_weight"], np.zeros(4))
    assert summary["total_incoming_count"] == 4
    assert summary["total_outgoing_count"] == 4
    assert summary["relative_net_mass_imbalance"] == 0.0


def test_wall_balance_reports_one_sided_imbalance_and_rejects_ambiguous_event():
    cfg = PhysicalCavityConfig(nx=1, ny=1, particles_per_cell=1)
    audit = WallBalanceAccumulator(cfg)
    audit.add(
        "top",
        np.array([0.5 * cfg.length]),
        np.array([[0.0, 10.0, 0.0]]),
        np.ones(1),
        np.zeros(3),
    )
    assert audit.finalize()["relative_net_mass_imbalance"] == 2.0

    with np.testing.assert_raises_regex(ValueError, "zero wall-relative"):
        audit.add(
            "top",
            np.array([0.5 * cfg.length]),
            np.array([[1.0, 0.0, 0.0]]),
            np.ones(1),
            np.zeros(3),
        )
