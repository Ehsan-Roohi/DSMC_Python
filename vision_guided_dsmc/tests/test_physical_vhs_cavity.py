import numpy as np

from vgdsmc.physical_vhs_cavity import (
    PhysicalCavityConfig,
    run_physical_vhs_cavity,
)


def test_isothermal_physical_cavity_remains_near_wall_temperature():
    cfg = PhysicalCavityConfig(
        nx=4,
        ny=4,
        particles_per_cell=12,
        steps=40,
        sample_start=20,
        t_left=270.0,
        t_right=270.0,
        t_top=270.0,
        t_bottom=270.0,
    )
    temperature, accepted = run_physical_vhs_cavity(cfg)

    assert accepted > 0
    assert abs(float(np.mean(temperature)) - 270.0) < 55.0


def test_hot_left_and_cold_right_generate_correct_temperature_ordering():
    cfg = PhysicalCavityConfig(
        nx=6,
        ny=4,
        particles_per_cell=20,
        steps=100,
        sample_start=50,
        t_left=330.0,
        t_right=210.0,
        t_top=270.0,
        t_bottom=270.0,
    )
    temperature, accepted = run_physical_vhs_cavity(cfg)

    assert accepted > 0
    assert float(np.mean(temperature[:, 0])) > float(np.mean(temperature[:, -1]))
