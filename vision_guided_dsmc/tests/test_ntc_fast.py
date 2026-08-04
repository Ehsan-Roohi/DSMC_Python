import numpy as np

from vgdsmc.ntc_fast import collide_vhs_ntc_fast
from vgdsmc.ntc_solver import collide_vhs_ntc, conserved_quantities_ntc
from vgdsmc.vhs_model import PhysicalCavityConfig, initialize_physical_state


def _collision_config(seed: int = 401) -> PhysicalCavityConfig:
    return PhysicalCavityConfig(
        nx=4,
        ny=3,
        particles_per_cell=64,
        knudsen=0.05,
        dt_safety=0.4,
        seed=seed,
        stratified_initialization=True,
    )


def test_fast_ntc_is_deterministic_for_its_own_seed():
    cfg = _collision_config()
    first = initialize_physical_state(cfg)
    second = first.copy()

    first_diagnostics = collide_vhs_ntc_fast(
        first,
        cfg,
        np.random.default_rng(402),
    )
    second_diagnostics = collide_vhs_ntc_fast(
        second,
        cfg,
        np.random.default_rng(402),
    )

    assert first_diagnostics == second_diagnostics
    assert np.array_equal(first.vel, second.vel)


def test_fast_ntc_conserves_momentum_and_energy_without_clipping():
    cfg = _collision_config()
    state = initialize_physical_state(cfg)
    before = conserved_quantities_ntc(state)
    diagnostics = collide_vhs_ntc_fast(
        state,
        cfg,
        np.random.default_rng(403),
    )
    after = conserved_quantities_ntc(state)

    assert diagnostics.candidate_collisions > 0
    assert diagnostics.accepted_collisions > 0
    assert diagnostics.majorant_violations == 0
    assert diagnostics.max_acceptance_ratio <= 1.0 + 1.0e-12
    assert np.allclose(after[1], before[1], rtol=1.0e-12, atol=1.0e-9)
    assert np.isclose(after[2], before[2], rtol=1.0e-12, atol=0.0)


def test_fast_ntc_is_galilean_invariant():
    cfg = _collision_config()
    base = initialize_physical_state(cfg)
    shifted = base.copy()
    drift = np.array([800.0, -350.0, 125.0])
    shifted.vel += drift

    first = collide_vhs_ntc_fast(base, cfg, np.random.default_rng(404))
    second = collide_vhs_ntc_fast(shifted, cfg, np.random.default_rng(404))

    assert first == second
    assert np.allclose(shifted.vel - base.vel, drift, rtol=0.0, atol=2.0e-12)


def test_fast_ntc_matches_reference_collision_statistics_over_many_seeds():
    cfg = _collision_config(seed=405)
    initial = initialize_physical_state(cfg)
    reference_counts = []
    fast_counts = []

    for seed in range(500, 564):
        reference = initial.copy()
        fast = initial.copy()
        reference_diagnostics = collide_vhs_ntc(
            reference,
            cfg,
            np.random.default_rng(seed),
        )
        fast_diagnostics = collide_vhs_ntc_fast(
            fast,
            cfg,
            np.random.default_rng(seed),
        )
        reference_counts.append(
            (
                reference_diagnostics.candidate_collisions,
                reference_diagnostics.accepted_collisions,
            )
        )
        fast_counts.append(
            (
                fast_diagnostics.candidate_collisions,
                fast_diagnostics.accepted_collisions,
            )
        )
        assert fast_diagnostics.majorant_violations == 0

    reference_mean = np.mean(reference_counts, axis=0)
    fast_mean = np.mean(fast_counts, axis=0)
    # Both backends draw from the same NTC candidate and acceptance laws, but
    # consume RNG values in different orders.  This bound is intentionally a
    # statistical comparison rather than a bitwise trajectory comparison.
    assert np.allclose(fast_mean, reference_mean, rtol=0.025, atol=1.0)


def test_fast_ntc_scatter_has_reference_isotropy_statistics():
    cfg = _collision_config(seed=407)
    initial = initialize_physical_state(cfg)
    speed = np.linalg.norm(initial.vel, axis=1)
    sign = np.where(np.arange(len(speed)) % 2, 1.0, -1.0)
    initial.vel[:] = 0.0
    initial.vel[:, 0] = sign * speed

    component_energy = {"reference": [], "fast": []}
    for seed in range(600, 728):
        for name, backend in (
            ("reference", collide_vhs_ntc),
            ("fast", collide_vhs_ntc_fast),
        ):
            state = initial.copy()
            backend(state, cfg, np.random.default_rng(seed))
            component_energy[name].append(np.sum(state.vel**2, axis=0))

    reference_mean = np.mean(component_energy["reference"], axis=0)
    fast_mean = np.mean(component_energy["fast"], axis=0)
    assert np.allclose(fast_mean, reference_mean, rtol=0.035, atol=0.0)
    assert abs(fast_mean[1] - fast_mean[2]) / np.mean(fast_mean[1:]) < 0.04


def test_fast_ntc_rejects_unequal_weights_like_reference_backend():
    cfg = _collision_config()
    state = initialize_physical_state(cfg)
    state.weight[0] = 2.0
    with np.testing.assert_raises_regex(ValueError, "equal weights"):
        collide_vhs_ntc_fast(state, cfg, np.random.default_rng(406))
