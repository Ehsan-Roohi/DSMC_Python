from __future__ import annotations

import numpy as np
import pytest

import vgdsmc.event_wall_streaming as event_streaming
from vgdsmc.event_wall_streaming import (
    WALL_ORDER,
    stream_with_diffuse_walls,
)
from vgdsmc.vhs_model import (
    KB,
    PhysicalCavityConfig,
    PhysicalParticleState,
    VHSModel,
    _diffuse_wall,
)


def _unit_thermal_cfg(**overrides) -> PhysicalCavityConfig:
    # With mass=k_B and T=1, every wall has unit thermal standard deviation.
    values = {
        "nx": 1,
        "ny": 1,
        "particles_per_cell": 1,
        "length": 1.0,
        "t_left": 1.0,
        "t_right": 1.0,
        "t_bottom": 1.0,
        "t_top": 1.0,
        "vhs": VHSModel(mass=KB),
    }
    values.update(overrides)
    return PhysicalCavityConfig(**values)


def _state(position, velocity, weight=None) -> PhysicalParticleState:
    pos = np.asarray(position, dtype=np.float64)
    vel = np.asarray(velocity, dtype=np.float64)
    if weight is None:
        weight = np.ones(len(pos), dtype=np.float64)
    return PhysicalParticleState(pos, vel, np.asarray(weight, dtype=np.float64))


def _forced_corner_reflection(
    velocity,
    temperature,
    normal_axis,
    inward_sign,
    model,
    rng,
    wall_velocity=None,
):
    """Drive left-corner input through a legitimate left->bottom sequence."""
    reflected = np.zeros_like(velocity, dtype=np.float64)
    reflected[:, normal_axis] = inward_sign
    if normal_axis == 0:
        reflected[:, 1] = -1.0
    else:
        reflected[:, 0] = 1.0
    if wall_velocity is not None:
        reflected += np.asarray(wall_velocity)[None, :]
    return reflected


def test_ballistic_no_hit_is_exact_free_flight():
    cfg = _unit_thermal_cfg()
    state = _state(
        [[0.20, 0.30], [0.70, 0.80]],
        [[0.10, -0.20, 4.0], [-0.20, -0.10, -3.0]],
        [0.25, 1.75],
    )
    original_velocity = state.vel.copy()
    original_weight = state.weight.copy()

    diagnostics = stream_with_diffuse_walls(
        state, cfg, 0.25, np.random.default_rng(11)
    )

    expected = np.array([[0.225, 0.25], [0.65, 0.775]])
    assert np.allclose(state.pos, expected, rtol=0.0, atol=1.0e-15)
    assert np.array_equal(state.vel, original_velocity)
    assert np.array_equal(state.weight, original_weight)
    assert diagnostics.total_wall_hits == 0
    assert diagnostics.wall_hits == (0, 0, 0, 0)


def test_exact_impact_uses_reflected_velocity_for_remaining_time():
    cfg = _unit_thermal_cfg()
    incoming = np.array([[-1.0, 0.0, 0.0]])
    state = _state([[0.25, 0.50]], incoming)
    expected_reflection = _diffuse_wall(
        incoming,
        cfg.t_left,
        normal_axis=0,
        inward_sign=1.0,
        model=cfg.vhs,
        rng=np.random.default_rng(22),
        wall_velocity=np.zeros(3),
    )

    diagnostics = stream_with_diffuse_walls(
        state, cfg, 0.50, np.random.default_rng(22)
    )

    expected_position = np.array([[0.0, 0.50]])
    expected_position += expected_reflection[:, :2] * 0.25
    assert np.array_equal(state.vel, expected_reflection)
    assert np.allclose(state.pos, expected_position, rtol=0.0, atol=1.0e-15)
    assert diagnostics.total_wall_hits == 1
    assert diagnostics.wall_hits == (1, 0, 0, 0)


def test_exact_corner_tie_and_zero_time_followup_are_both_handled(monkeypatch):
    cfg = _unit_thermal_cfg()
    state = _state([[0.50, 0.50]], [[-1.0, -1.0, 0.0]])
    phases = []
    monkeypatch.setattr(
        event_streaming, "_diffuse_wall", _forced_corner_reflection
    )

    def phase_handler(wall, phase, position, velocity, weight, wall_velocity):
        phases.append((wall, phase, position.copy(), velocity.copy()))

    diagnostics = stream_with_diffuse_walls(
        state,
        cfg,
        0.60,
        np.random.default_rng(4),
        phase_event_handler=phase_handler,
        max_events_per_particle=8,
    )

    assert diagnostics.exact_corner_ties == 1
    assert diagnostics.total_wall_hits == 2
    assert diagnostics.zero_time_wall_hits == 1
    assert diagnostics.incident_sign_violations == 0
    assert diagnostics.reflected_sign_violations == 0
    assert diagnostics.nonmonotone_hit_time == 0
    assert diagnostics.fallback_clip_count == 0
    assert [entry[:2] for entry in phases] == [
        ("left", "incoming"),
        ("left", "outgoing"),
        ("bottom", "incoming"),
        ("bottom", "outgoing"),
    ]
    assert np.all((state.pos >= 0.0) & (state.pos <= cfg.length))
    assert state.vel[0, 0] > 0.0
    assert state.vel[0, 1] > 0.0


def test_event_cap_raises_instead_of_clipping_remaining_corner_flight(monkeypatch):
    cfg = _unit_thermal_cfg()
    state = _state([[0.50, 0.50]], [[-1.0, -1.0, 0.0]])
    monkeypatch.setattr(
        event_streaming, "_diffuse_wall", _forced_corner_reflection
    )

    with pytest.raises(RuntimeError, match="maximum wall events exceeded"):
        stream_with_diffuse_walls(
            state,
            cfg,
            0.60,
            np.random.default_rng(4),
            max_events_per_particle=1,
        )


def test_nonrepresentable_exact_corner_is_analytically_snapped(monkeypatch):
    cfg = _unit_thermal_cfg()
    coordinate = 0.22715759353337972
    speed = -6.961387926450331
    state = _state([[coordinate, coordinate]], [[speed, speed, 0.0]])
    monkeypatch.setattr(
        event_streaming, "_diffuse_wall", _forced_corner_reflection
    )

    diagnostics = stream_with_diffuse_walls(
        state,
        cfg,
        0.04,
        np.random.default_rng(91),
        max_events_per_particle=8,
    )

    assert diagnostics.exact_corner_ties == 1
    assert diagnostics.fallback_clip_count == 0
    assert np.all(state.pos >= 0.0)
    assert np.all(state.pos <= cfg.length)


def test_unequal_time_two_hit_trajectory_has_exact_chronology(monkeypatch):
    cfg = _unit_thermal_cfg()
    state = _state([[0.20, 0.40]], [[-2.0, 0.0, 0.0]])
    events = []

    def forced_reflection(
        velocity,
        temperature,
        normal_axis,
        inward_sign,
        model,
        rng,
        wall_velocity=None,
    ):
        reflected = np.zeros_like(velocity)
        if normal_axis == 0 and inward_sign > 0.0:  # left at t=0.1
            reflected[:] = [2.0, 0.25, 0.0]
        elif normal_axis == 0:  # right at t=0.6
            reflected[:] = [-1.0, -0.50, 0.0]
        else:  # pragma: no cover - an unexpected wall fails via chronology
            raise AssertionError("unexpected wall in forced trajectory")
        return reflected

    def phase_handler(wall, phase, position, velocity, weight, wall_velocity):
        if phase == "incoming":
            events.append((wall, position.copy(), velocity.copy()))

    monkeypatch.setattr(event_streaming, "_diffuse_wall", forced_reflection)
    diagnostics = stream_with_diffuse_walls(
        state,
        cfg,
        1.0,
        np.random.default_rng(7),
        phase_event_handler=phase_handler,
    )

    assert [event[0] for event in events] == ["left", "right"]
    assert np.isclose(events[0][1][0], 0.40)
    assert np.isclose(events[1][1][0], 0.525)
    assert np.allclose(state.pos, [[0.60, 0.325]], rtol=0.0, atol=1.0e-15)
    assert np.array_equal(state.vel, [[-1.0, -0.50, 0.0]])
    assert diagnostics.total_wall_hits == 2
    assert diagnostics.zero_time_wall_hits == 0


def test_many_particle_multi_hit_stream_ends_in_bounds_without_clipping():
    cfg = _unit_thermal_cfg()
    rng = np.random.default_rng(810)
    count = 800
    state = _state(
        rng.random((count, 2)),
        rng.normal(0.0, 3.0, size=(count, 3)),
        rng.uniform(0.1, 2.0, size=count),
    )

    diagnostics = stream_with_diffuse_walls(
        state,
        cfg,
        3.0,
        np.random.default_rng(811),
        max_events_per_particle=64,
    )

    assert diagnostics.total_wall_hits > count
    assert diagnostics.maximum_hits_on_one_particle > 1
    assert np.all(state.pos >= 0.0)
    assert np.all(state.pos <= cfg.length)
    assert np.all(np.isfinite(state.pos))
    assert np.all(np.isfinite(state.vel))


def test_stationary_wall_emission_recovers_equilibrium_half_range_moments():
    cfg = _unit_thermal_cfg()
    count = 60_000
    x = (np.arange(count, dtype=np.float64) + 0.5) / count
    state = _state(
        np.column_stack((x, np.full(count, 0.90))),
        np.column_stack((np.zeros(count), np.ones(count), np.zeros(count))),
    )
    outgoing = []

    def phase_handler(wall, phase, position, velocity, weight, wall_velocity):
        if wall == "top" and phase == "outgoing":
            outgoing.append(velocity.copy())

    stream_with_diffuse_walls(
        state,
        cfg,
        0.10,
        np.random.default_rng(917),
        phase_event_handler=phase_handler,
    )

    emitted = np.vstack(outgoing)
    assert len(emitted) == count
    assert np.all(emitted[:, 1] < 0.0)
    assert np.allclose(emitted[:, [0, 2]].mean(axis=0), 0.0, atol=0.015)
    assert np.allclose(emitted[:, [0, 2]].var(axis=0), 1.0, rtol=0.02)
    assert np.isclose(emitted[:, 1].mean(), -np.sqrt(np.pi / 2.0), rtol=0.01)
    assert np.isclose(np.mean(emitted[:, 1] ** 2), 2.0, rtol=0.015)


def test_moving_lid_is_sampled_in_its_frame():
    wall_velocity = np.array([3.0, 0.0, -2.0])
    cfg = _unit_thermal_cfg(top_wall_velocity=tuple(wall_velocity))
    count = 40_000
    state = _state(
        np.column_stack(
            (
                np.full(count, 0.50),
                np.full(count, 0.95),
            )
        ),
        np.column_stack((np.zeros(count), np.ones(count), np.zeros(count))),
    )
    outgoing = []

    def phase_handler(wall, phase, position, velocity, weight, event_wall_velocity):
        if phase == "outgoing":
            assert np.array_equal(event_wall_velocity, wall_velocity)
            outgoing.append(velocity.copy())

    stream_with_diffuse_walls(
        state,
        cfg,
        0.06,
        np.random.default_rng(918),
        phase_event_handler=phase_handler,
    )

    emitted = np.vstack(outgoing)
    relative = emitted - wall_velocity
    assert np.all(relative[:, 1] < 0.0)
    assert np.allclose(relative[:, [0, 2]].mean(axis=0), 0.0, atol=0.02)
    assert np.allclose(emitted[:, [0, 2]].mean(axis=0), [3.0, -2.0], atol=0.02)
    assert np.allclose(state.pos[:, 0], 0.50 + 0.01 * emitted[:, 0])
    assert np.allclose(state.pos[:, 1], 1.00 + 0.01 * emitted[:, 1])


def test_identical_rng_and_state_produce_bitwise_identical_trajectory_and_events():
    cfg = _unit_thermal_cfg(lid_velocity_x=0.5)
    initial = _state(
        [[0.1, 0.2], [0.8, 0.9], [0.4, 0.6], [0.7, 0.3]],
        [[-2.0, 1.0, 0.3], [0.5, 2.0, -0.2], [3.0, -1.0, 0.0], [-1.0, -2.0, 1.0]],
        [0.5, 1.0, 1.5, 2.0],
    )

    def run_once():
        state = initial.copy()
        events = []

        def phase_handler(wall, phase, position, velocity, weight, wall_velocity):
            events.append(
                (
                    wall,
                    phase,
                    position.copy(),
                    velocity.copy(),
                    weight.copy(),
                    wall_velocity.copy(),
                )
            )

        diagnostics = stream_with_diffuse_walls(
            state,
            cfg,
            1.25,
            np.random.default_rng(4321),
            phase_event_handler=phase_handler,
        )
        return state, events, diagnostics

    first, first_events, first_diagnostics = run_once()
    second, second_events, second_diagnostics = run_once()
    assert np.array_equal(first.pos, second.pos)
    assert np.array_equal(first.vel, second.vel)
    assert np.array_equal(first.weight, second.weight)
    assert first_diagnostics == second_diagnostics
    assert len(first_events) == len(second_events)
    for left, right in zip(first_events, second_events, strict=True):
        assert left[:2] == right[:2]
        for left_array, right_array in zip(left[2:], right[2:], strict=True):
            assert np.array_equal(left_array, right_array)


def test_legacy_and_phase_callbacks_report_paired_counts_and_weights():
    cfg = _unit_thermal_cfg()
    weights = np.array([0.5, 1.0, 1.5, 2.0])
    state = _state(
        [[0.1, 0.5], [0.9, 0.5], [0.4, 0.1], [0.6, 0.9]],
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
        weights,
    )
    legacy_calls = []
    phase_calls = []

    # Deliberately has exactly the five-argument legacy signature.
    def legacy_handler(wall, position, velocity, weight, wall_velocity):
        legacy_calls.append((wall, len(position), float(np.sum(weight))))

    def phase_handler(wall, phase, position, velocity, weight, wall_velocity):
        phase_calls.append((wall, phase, len(position), float(np.sum(weight))))

    diagnostics = stream_with_diffuse_walls(
        state,
        cfg,
        0.10,
        np.random.default_rng(74),
        legacy_handler,
        phase_event_handler=phase_handler,
    )

    assert diagnostics.wall_hits == (1, 1, 1, 1)
    assert diagnostics.incident_counts == diagnostics.reflected_counts
    assert diagnostics.incident_relative_weight == diagnostics.reflected_relative_weight
    assert diagnostics.particle_count_delta == 0
    assert diagnostics.relative_weight_delta == 0.0
    assert diagnostics.as_dict()["multi_hit_particle_fraction"] == 0.0
    assert len(legacy_calls) == 2 * len(WALL_ORDER)
    for wall, weight in zip(WALL_ORDER, weights, strict=True):
        matching = [entry for entry in phase_calls if entry[0] == wall]
        assert [entry[1] for entry in matching] == ["incoming", "outgoing"]
        assert [entry[2] for entry in matching] == [1, 1]
        assert np.allclose([entry[3] for entry in matching], weight)
    assert sum(call[3] for call in phase_calls if call[1] == "incoming") == pytest.approx(
        np.sum(weights)
    )
    assert sum(call[3] for call in phase_calls if call[1] == "outgoing") == pytest.approx(
        np.sum(weights)
    )


def test_particle_count_order_and_weights_are_conserved():
    cfg = _unit_thermal_cfg()
    rng = np.random.default_rng(628)
    count = 1_000
    state = _state(
        rng.random((count, 2)),
        rng.normal(size=(count, 3)),
        np.linspace(0.1, 2.0, count),
    )
    original_weight = state.weight.copy()
    original_particle_count = len(state.pos)

    stream_with_diffuse_walls(
        state, cfg, 2.0, np.random.default_rng(629), max_events_per_particle=64
    )

    assert state.pos.shape == (original_particle_count, 2)
    assert state.vel.shape == (original_particle_count, 3)
    assert state.weight.shape == (original_particle_count,)
    assert np.array_equal(state.weight, original_weight)


def test_callbacks_receive_isolated_snapshots(monkeypatch):
    cfg = _unit_thermal_cfg()
    state = _state([[0.10, 0.40]], [[-1.0, 0.25, 0.0]], [1.5])
    phase_snapshots = []

    def forced_reflection(*args, **kwargs):
        return np.array([[2.0, -0.5, 0.25]])

    def legacy_handler(wall, position, velocity, weight, wall_velocity):
        position[:] = 99.0
        velocity[:] = 99.0
        weight[:] = 99.0
        wall_velocity[:] = 99.0

    def phase_handler(wall, phase, position, velocity, weight, wall_velocity):
        phase_snapshots.append(
            (phase, position.copy(), velocity.copy(), weight.copy(), wall_velocity.copy())
        )
        position[:] = -99.0
        velocity[:] = -99.0
        weight[:] = -99.0
        wall_velocity[:] = -99.0

    monkeypatch.setattr(event_streaming, "_diffuse_wall", forced_reflection)
    stream_with_diffuse_walls(
        state,
        cfg,
        0.10,
        np.random.default_rng(8),
        legacy_handler,
        phase_event_handler=phase_handler,
    )

    assert [snapshot[0] for snapshot in phase_snapshots] == ["incoming", "outgoing"]
    assert np.array_equal(phase_snapshots[0][2], [[-1.0, 0.25, 0.0]])
    assert np.array_equal(phase_snapshots[1][2], [[2.0, -0.5, 0.25]])
    assert np.allclose(state.pos, [[0.0, 0.425]], rtol=0.0, atol=1.0e-15)
    assert np.array_equal(state.vel, [[2.0, -0.5, 0.25]])
    assert np.array_equal(state.weight, [1.5])


def test_float32_ballistic_state_is_supported_without_dtype_promotion():
    cfg = _unit_thermal_cfg()
    state = PhysicalParticleState(
        pos=np.array([[0.25, 0.50]], dtype=np.float32),
        vel=np.array([[0.50, -0.25, 3.0]], dtype=np.float32),
        weight=np.ones(1, dtype=np.float32),
    )

    diagnostics = stream_with_diffuse_walls(
        state, cfg, 0.10, np.random.default_rng(9)
    )

    assert state.pos.dtype == np.float32
    assert state.vel.dtype == np.float32
    assert np.allclose(state.pos, [[0.30, 0.475]])
    assert diagnostics.total_wall_hits == 0


def test_float32_unequal_hit_times_are_not_silently_snapped_to_corner(monkeypatch):
    cfg = _unit_thermal_cfg()
    state = PhysicalParticleState(
        pos=np.array([[0.3, 0.9000027]], dtype=np.float32),
        vel=np.array([[-1.0, -3.0000088, 0.0]], dtype=np.float32),
        weight=np.ones(1, dtype=np.float32),
    )
    phases = []

    def phase_handler(wall, phase, position, velocity, weight, wall_velocity):
        if phase == "incoming":
            phases.append(wall)

    monkeypatch.setattr(
        event_streaming, "_diffuse_wall", _forced_corner_reflection
    )
    diagnostics = stream_with_diffuse_walls(
        state,
        cfg,
        0.31,
        np.random.default_rng(12),
        phase_event_handler=phase_handler,
    )

    assert phases == ["left", "bottom"]
    assert diagnostics.exact_corner_ties == 0
    assert diagnostics.zero_time_wall_hits == 0
    assert diagnostics.fallback_clip_count == 0


def test_invalid_wall_physics_fails_before_mutation_or_callback():
    cfg = _unit_thermal_cfg(t_left=-1.0)
    state = _state([[0.10, 0.50]], [[-1.0, 0.0, 0.0]])
    original = state.copy()
    callbacks = []

    with pytest.raises(ValueError, match="wall temperatures"):
        stream_with_diffuse_walls(
            state,
            cfg,
            0.20,
            np.random.default_rng(10),
            lambda *args: callbacks.append(args),
        )

    assert np.array_equal(state.pos, original.pos)
    assert np.array_equal(state.vel, original.vel)
    assert callbacks == []


def test_out_of_domain_input_is_rejected_not_silently_clipped():
    cfg = _unit_thermal_cfg()
    state = _state([[-np.finfo(float).eps, 0.5]], [[1.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="initial positions inside"):
        stream_with_diffuse_walls(state, cfg, 0.1, np.random.default_rng(1))
    assert state.pos[0, 0] < 0.0
