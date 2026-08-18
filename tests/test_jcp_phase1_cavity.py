from __future__ import annotations

import inspect

import numpy as np

from vgdsmc import jcp_phase1_cavity as jcp2
from vgdsmc.mohammadzadeh_production import _moment_payload
from vgdsmc.moment_sampling import PhysicalMomentAccumulator
from vgdsmc.vhs_model import KB, PhysicalCavityConfig, PhysicalParticleState


def test_locked_task_matrix_and_disjoint_budgets():
    assert len(jcp2.group_seeds("evaluation")) == 12
    assert len(jcp2.group_seeds("reference")) == 25
    assert not set(jcp2.group_seeds("evaluation")) & set(
        jcp2.group_seeds("reference")
    )
    assert jcp2.task_from_index("evaluation", 0) == 26082101
    assert jcp2.task_from_index("evaluation", 11) == 26082112
    assert jcp2.task_from_index("reference", 24) == 26082225
    assert set(jcp2.OBSERVATION_BLOCKS).isdisjoint(jcp2.COMPARATOR_BLOCKS)


def test_protocol_freezes_primary_S2_and_full_hierarchy():
    protocol = jcp2.load_protocol()
    active = [item for item in protocol["conditions"] if item["active_in_this_phase"]]
    assert [item["id"] for item in active] == ["S2_kn0p085_u350"]
    assert protocol["endpoints"]["primary"] == "qy at S2"
    assert tuple(protocol["estimator_contract"]["low_order_fields"]) == (
        "rho",
        "u",
        "v",
        "T",
    )
    assert tuple(protocol["estimator_contract"]["high_order_fields"]) == (
        "Pxy",
        "Pxx_minus_Pyy",
        "qx",
        "qy",
    )


def test_prediction_interface_cannot_accept_a_reference_path():
    parameters = inspect.signature(jcp2.predict).parameters
    assert "run_root" in parameters
    assert "reference_root" not in parameters
    assert "reference" not in parameters


def test_additive_merge_preserves_new_fourth_moment():
    base = {
        "samples": 3,
        "simulated_count": np.array([5.0, 7.0]),
        "m0": np.array([10.0, 11.0]),
        "m1": np.ones((2, 3)),
        "m2": np.ones((2, 3, 3)),
        "energy": np.array([20.0, 21.0]),
        "energy_velocity": np.ones((2, 3)) * 4.0,
        "speed4": np.array([100.0, 121.0]),
    }
    merged = jcp2.merge_payloads((base, base, base))
    assert merged["samples"] == 9
    for key in jcp2.MOMENT_KEYS:
        np.testing.assert_array_equal(merged[key], 3.0 * base[key])


def test_temperature_delta_variance_is_finite_and_nonnegative():
    cfg = PhysicalCavityConfig(nx=1, ny=1, particles_per_cell=8)
    rng = np.random.default_rng(22)
    state = PhysicalParticleState(
        pos=np.full((128, 2), 0.5 * cfg.length),
        vel=rng.normal(0.0, 300.0, size=(128, 3)),
        weight=np.ones(128),
    )
    accumulator = PhysicalMomentAccumulator(cfg)
    accumulator.add(state)
    variance = jcp2._temperature_delta_variance(
        _moment_payload(accumulator), cfg, KB
    )
    assert variance.shape == (1, 1)
    assert np.all(np.isfinite(variance))
    assert np.all(variance >= 0.0)


def test_effective_block_accounting_is_bounded():
    rng = np.random.default_rng(31)
    blocks = rng.normal(size=(20, 13, 8, 3, 4))
    effective, autocorrelation = jcp2._effective_blocks(blocks)
    assert effective.shape == (8,)
    assert autocorrelation.shape == (8, 13)
    assert np.all(effective >= 20.0)
    assert np.all(effective <= 260.0)
    np.testing.assert_array_equal(autocorrelation[:, 0], np.ones(8))


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"{len(tests)} JCP2 tests passed")
