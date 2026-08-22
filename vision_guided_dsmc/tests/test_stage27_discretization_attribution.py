import math

from vgdsmc.stage27_discretization_attribution import (
    DEFAULT_CASES,
    computational_work_proxy,
)


def test_default_cases_isolate_spatial_and_velocity_refinement():
    by_name = {case.name: case for case in DEFAULT_CASES}
    baseline = by_name["baseline_20x20_nv17"]
    spatial = by_name["spatial_24x24_nv17"]
    velocity = by_name["velocity_20x20_nv19"]
    assert spatial.nv == baseline.nv
    assert spatial.nx > baseline.nx and spatial.ny > baseline.ny
    assert velocity.nx == baseline.nx and velocity.ny == baseline.ny
    assert velocity.nv > baseline.nv


def test_work_proxy_scales_with_each_resolution_axis():
    baseline = computational_work_proxy(20, 20, 17, 100)
    spatial = computational_work_proxy(24, 24, 17, 100)
    velocity = computational_work_proxy(20, 20, 19, 100)
    assert spatial > baseline
    assert velocity > baseline
    assert math.isclose(spatial / baseline, (24 / 20) ** 2)
    assert math.isclose(velocity / baseline, (19 / 17) ** 3)
