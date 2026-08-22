import numpy as np

from vgdsmc.vhs_sbt import scatter_equal_mass, sbt_collide_cell, vhs_cross_section


def test_vhs_cross_section_decreases_with_relative_speed():
    cross_sections = vhs_cross_section(np.array([100.0, 500.0, 2000.0]))
    assert np.all(cross_sections[:-1] > cross_sections[1:])
    assert np.all(cross_sections > 0.0)


def test_equal_mass_scattering_conserves_pair_invariants():
    rng = np.random.default_rng(3)
    first = np.array([300.0, -50.0, 20.0])
    second = np.array([-100.0, 40.0, -10.0])
    momentum_before = first + second
    energy_before = np.dot(first, first) + np.dot(second, second)

    first_after, second_after = scatter_equal_mass(first, second, rng)

    assert np.allclose(first_after + second_after, momentum_before, atol=1.0e-12)
    assert np.isclose(
        np.dot(first_after, first_after) + np.dot(second_after, second_after),
        energy_before,
        rtol=1.0e-13,
    )


def test_sbt_cell_collision_conserves_global_invariants():
    rng = np.random.default_rng(5)
    velocities = rng.normal(0.0, 400.0, (80, 3))
    momentum_before = velocities.sum(axis=0)
    energy_before = np.sum(velocities**2)

    updated, accepted = sbt_collide_cell(
        velocities,
        fnum=1.0e8,
        dt=1.0e-7,
        cell_volume=1.0e-18,
        rng=rng,
    )

    assert accepted > 0
    assert np.allclose(updated.sum(axis=0), momentum_before, atol=1.0e-9)
    assert np.isclose(np.sum(updated**2), energy_before, rtol=1.0e-13)
