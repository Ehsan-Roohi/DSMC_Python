import numpy as np

from vgdsmc.moment_sampling import PhysicalMomentAccumulator
from vgdsmc.sbt_solver import sample_physical_state
from vgdsmc.vhs_model import (
    KB,
    PhysicalCavityConfig,
    PhysicalParticleState,
)


def _one_cell_state(velocity: np.ndarray) -> PhysicalParticleState:
    return PhysicalParticleState(
        pos=np.full((len(velocity), 2), 0.5e-6),
        vel=velocity.astype(float),
        weight=np.ones(len(velocity)),
    )


def test_raw_moment_temperature_and_heat_flux_for_symmetric_distribution():
    cfg = PhysicalCavityConfig(nx=1, ny=1, particles_per_cell=8)
    speed = np.sqrt(KB * 320.0 / cfg.vhs.mass)
    signs = np.array(
        [
            [sx, sy, sz]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
    )
    drift = np.array([91.0, -13.0, 7.0])
    state = _one_cell_state(drift + speed * signs)
    accumulator = PhysicalMomentAccumulator(cfg)
    accumulator.add(state)
    fields = accumulator.finalize()

    assert np.isclose(fields["T"][0, 0], 320.0, rtol=1.0e-12)
    assert np.allclose(
        [fields["u"][0, 0], fields["v"][0, 0], fields["w"][0, 0]],
        drift,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert np.allclose(
        [fields["qx"][0, 0], fields["qy"][0, 0], fields["qz"][0, 0]],
        0.0,
        rtol=0.0,
        atol=2.0e-8,
    )


def test_raw_moment_heat_flux_matches_direct_central_moment():
    cfg = PhysicalCavityConfig(nx=1, ny=1, particles_per_cell=128)
    rng = np.random.default_rng(881)
    velocity = rng.normal(size=(128, 3)) * np.array([300.0, 240.0, 180.0])
    velocity[:, 0] += 0.0008 * velocity[:, 1] ** 2
    state = _one_cell_state(velocity)
    accumulator = PhysicalMomentAccumulator(cfg)
    accumulator.add(state)
    accumulator.add(state)
    fields = accumulator.finalize()

    mean = velocity.mean(axis=0)
    peculiar = velocity - mean
    direct_central = np.mean(
        np.sum(peculiar**2, axis=1)[:, None] * peculiar,
        axis=0,
    )
    number_density = fields["number_density"][0, 0]
    expected = 0.5 * cfg.vhs.mass * number_density * direct_central
    actual = np.array(
        [fields["qx"][0, 0], fields["qy"][0, 0], fields["qz"][0, 0]]
    )
    assert np.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-10)


def test_raw_moment_accumulator_rejects_empty_cells():
    cfg = PhysicalCavityConfig(nx=2, ny=1, particles_per_cell=1)
    state = PhysicalParticleState(
        pos=np.array([[0.25 * cfg.length, 0.5 * cfg.length]]),
        vel=np.zeros((1, 3)),
        weight=np.ones(1),
    )
    accumulator = PhysicalMomentAccumulator(cfg)
    accumulator.add(state)
    with np.testing.assert_raises_regex(ValueError, "empty cells"):
        accumulator.finalize()


def test_accumulator_reuses_reductions_for_instantaneous_fields():
    cfg = PhysicalCavityConfig(
        nx=4,
        ny=3,
        particles_per_cell=8,
        stratified_initialization=True,
        seed=883,
    )
    rng = np.random.default_rng(884)
    state = PhysicalParticleState(
        pos=rng.random((96, 2)) * cfg.length,
        vel=rng.normal(0.0, 250.0, size=(96, 3)),
        weight=np.ones(96),
    )
    accumulator = PhysicalMomentAccumulator(cfg)
    reused = accumulator.add(state, return_instantaneous=True)
    direct = sample_physical_state(state, cfg)

    assert reused is not None
    for key in ("T", "u", "v", "w"):
        assert np.array_equal(reused[key], direct[key])
