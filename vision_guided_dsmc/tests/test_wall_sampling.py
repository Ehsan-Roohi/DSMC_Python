import numpy as np

from vgdsmc.vhs_model import (
    KB,
    PhysicalCavityConfig,
    PhysicalParticleState,
    apply_diffuse_walls,
)
from vgdsmc.wall_sampling import LidWallEventAccumulator


def test_wall_handler_receives_incoming_velocity_and_impact_location():
    cfg = PhysicalCavityConfig(
        nx=10,
        ny=10,
        lid_velocity_x=100.0,
    )
    # This particle is 0.1 microseconds of its trajectory beyond the lid.
    incoming = np.array([[20.0, 200.0, -5.0]])
    after = 2.0e-10
    impact_x = 0.37 * cfg.length
    state = PhysicalParticleState(
        pos=np.array([[impact_x + incoming[0, 0] * after, cfg.length + incoming[0, 1] * after]]),
        vel=incoming.copy(),
        weight=np.ones(1),
    )
    captured = []

    def handler(wall, position, velocity, weight, wall_velocity):
        captured.append((wall, position.copy(), velocity.copy(), weight.copy(), wall_velocity.copy()))

    apply_diffuse_walls(state, cfg, np.random.default_rng(92), handler)
    assert len(captured) == 2
    wall, position, velocity, weight, wall_velocity = captured[0]
    assert wall == "top"
    assert np.isclose(position[0], impact_x, rtol=0.0, atol=1.0e-18)
    assert np.array_equal(velocity, incoming)
    assert np.array_equal(weight, np.ones(1))
    assert np.array_equal(wall_velocity, np.array([100.0, 0.0, 0.0]))
    reflected = captured[1]
    assert reflected[0] == "top"
    assert np.isclose(reflected[1][0], impact_x, rtol=0.0, atol=1.0e-18)
    assert reflected[2][0, 1] < 0.0


def test_microscopic_lid_estimator_matches_pre_equations_for_synthetic_events():
    cfg = PhysicalCavityConfig(
        nx=2,
        ny=2,
        lid_velocity_x=100.0,
        t_top=300.0,
    )
    accumulator = LidWallEventAccumulator(cfg)
    relative = np.array(
        [
            [-20.0, 100.0, 30.0],
            [-40.0, 200.0, -10.0],
        ]
    )
    incoming = relative + np.array([100.0, 0.0, 0.0])
    accumulator.add(
        "top",
        np.array([0.25, 0.25]) * cfg.length,
        incoming,
        np.ones(2),
        np.array([100.0, 0.0, 0.0]),
    )
    reflected_relative = np.array(
        [
            [0.0, -120.0, 20.0],
            [0.0, -180.0, -20.0],
        ]
    )
    accumulator.add(
        "top",
        np.array([0.25, 0.25]) * cfg.length,
        reflected_relative + np.array([100.0, 0.0, 0.0]),
        np.ones(2),
        np.array([100.0, 0.0, 0.0]),
    )
    result = accumulator.finalize()
    all_relative = np.vstack((relative, reflected_relative))
    inverse = 1.0 / np.abs(all_relative[:, 1])
    expected_slip = np.sum(inverse * -all_relative[:, 0]) / np.sum(inverse)
    expected_speed2 = np.sum(inverse * np.sum(all_relative**2, axis=1)) / np.sum(inverse)
    expected_temperature = (
        expected_speed2 - expected_slip**2
    ) / (3.0 * (KB / cfg.vhs.mass))
    assert np.isclose(result["microscopic_lid_slip"][0], expected_slip)
    assert np.isclose(result["microscopic_lid_T"][0], expected_temperature)
    assert result["microscopic_lid_event_count"][0] == 4.0
    assert np.isnan(result["microscopic_lid_T"][1])


def test_surface_estimator_recovers_equilibrium_wall_temperature():
    cfg = PhysicalCavityConfig(
        nx=1,
        ny=1,
        particles_per_cell=1,
        lid_velocity_x=100.0,
        t_top=300.0,
    )
    rng = np.random.default_rng(404)
    count = 120_000
    sigma = np.sqrt(KB * cfg.t_top / cfg.vhs.mass)
    relative = rng.normal(0.0, sigma, size=(2 * count, 3))
    normal_magnitude = sigma * np.sqrt(
        -2.0 * np.log(np.maximum(rng.random(2 * count), 1.0e-14))
    )
    relative[:count, 1] = normal_magnitude[:count]
    relative[count:, 1] = -normal_magnitude[count:]
    accumulator = LidWallEventAccumulator(cfg)
    accumulator.add(
        "top",
        np.full(2 * count, 0.5 * cfg.length),
        relative + np.array([cfg.lid_velocity_x, 0.0, 0.0]),
        np.ones(2 * count),
        np.array([cfg.lid_velocity_x, 0.0, 0.0]),
    )
    result = accumulator.finalize()
    assert abs(result["microscopic_lid_slip"][0]) < 3.0
    assert np.isclose(result["microscopic_lid_T"][0], 300.0, rtol=0.015)
