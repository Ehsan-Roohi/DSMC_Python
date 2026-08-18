from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vgdsmc.moment_sampling import PhysicalMomentAccumulator
from vgdsmc.ntc_checkpoint import (
    CHECKPOINT_FORMAT,
    NTCCheckpoint,
    NTCCheckpointConfigMismatchError,
    NTCCheckpointCorruptionError,
    config_fingerprint,
    load_ntc_checkpoint,
    save_ntc_checkpoint,
)
from vgdsmc.vhs_model import (
    PhysicalCavityConfig,
    PhysicalParticleState,
    initialize_physical_state,
)
from vgdsmc.wall_sampling import LidWallEventAccumulator


def _runtime_checkpoint(
    cfg: PhysicalCavityConfig,
    rng: np.random.Generator,
) -> NTCCheckpoint:
    state = initialize_physical_state(cfg)
    moments = PhysicalMomentAccumulator(cfg)
    moments.add(state)
    moments.add(state)
    wall = LidWallEventAccumulator(cfg)
    wall.add(
        "top",
        np.array([0.15, 0.65]) * cfg.length,
        np.array([[20.0, -50.0, 4.0], [30.0, -60.0, -2.0]]),
        np.array([1.0, 0.75]),
        cfg.resolved_wall_velocity("top"),
    )
    rng.random(19)
    temporal_sums = {
        "T": np.arange(cfg.nx * cfg.ny, dtype=np.float64).reshape(
            cfg.ny, cfg.nx
        ),
        "u": np.full((cfg.ny, cfg.nx), -2.5),
    }
    temporal_sums2 = {
        key: value**2 + 0.25 for key, value in temporal_sums.items()
    }
    return NTCCheckpoint(
        state=state,
        rng_state=rng.bit_generator.state,
        step_index=37,
        diagnostics={
            "candidate_collisions": 91,
            "accepted_collisions": 43,
            "max_acceptance_ratio": 0.8125,
        },
        moments=moments,
        wall_events=wall,
        temporal_sums=temporal_sums,
        temporal_sums2=temporal_sums2,
        temporal_nsamples=2,
        block_accumulators={
            "block_000": {
                "sums": np.arange(6, dtype=np.float64).reshape(2, 3),
                "sums2": np.linspace(0.0, 1.0, 6).reshape(2, 3),
                "nsamples": 11,
            },
            "complete": False,
        },
        metadata={"stage": "M1", "seed": cfg.seed},
    )


def _assert_runtime_equal(left: NTCCheckpoint, right: NTCCheckpoint) -> None:
    assert right.step_index == left.step_index
    assert right.temporal_nsamples == left.temporal_nsamples
    assert right.diagnostics == left.diagnostics
    np.testing.assert_array_equal(right.state.pos, left.state.pos)
    np.testing.assert_array_equal(right.state.vel, left.state.vel)
    np.testing.assert_array_equal(right.state.weight, left.state.weight)
    for name in (
        "simulated_count",
        "m0",
        "m1",
        "m2",
        "energy",
        "energy_velocity",
        "speed4",
    ):
        np.testing.assert_array_equal(
            getattr(right.moments, name), getattr(left.moments, name)
        )
    assert right.moments.samples == left.moments.samples
    for name in (
        "event_count",
        "inverse_flux_weight",
        "weighted_slip",
        "weighted_relative_speed2",
    ):
        np.testing.assert_array_equal(
            getattr(right.wall_events, name), getattr(left.wall_events, name)
        )
    for key in left.temporal_sums:
        np.testing.assert_array_equal(
            right.temporal_sums[key], left.temporal_sums[key]
        )
        np.testing.assert_array_equal(
            right.temporal_sums2[key], left.temporal_sums2[key]
        )
    assert right.block_accumulators is not None
    assert left.block_accumulators is not None
    assert right.block_accumulators["complete"] is False
    assert right.block_accumulators["block_000"]["nsamples"] == 11
    np.testing.assert_array_equal(
        right.block_accumulators["block_000"]["sums"],
        left.block_accumulators["block_000"]["sums"],
    )


def test_checkpoint_round_trip_is_complete_deterministic_and_pickle_free(
    tmp_path: Path,
) -> None:
    cfg = PhysicalCavityConfig(
        nx=3,
        ny=2,
        particles_per_cell=4,
        lid_velocity_x=100.0,
        stratified_initialization=True,
        seed=91001,
    )
    rng = np.random.default_rng(4821)
    checkpoint = _runtime_checkpoint(cfg, rng)
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"

    manifest = save_ntc_checkpoint(first, cfg, checkpoint)
    save_ntc_checkpoint(second, cfg, checkpoint)

    assert first.read_bytes() == second.read_bytes()
    assert not list(tmp_path.glob(".*.tmp"))
    assert manifest["format"] == CHECKPOINT_FORMAT
    assert manifest["config_fingerprint"] == config_fingerprint(cfg)
    assert len(manifest["manifest_sha256"]) == 64
    assert all(
        len(descriptor["sha256"]) == 64
        for descriptor in manifest["arrays"].values()
    )
    with np.load(first, allow_pickle=False) as archive:
        assert all(archive[key].dtype != object for key in archive.files)
        embedded = json.loads(archive["__manifest__"].tobytes())
    assert embedded == manifest

    loaded = load_ntc_checkpoint(first, cfg)
    _assert_runtime_equal(checkpoint, loaded)
    assert loaded.metadata == checkpoint.metadata


@pytest.mark.parametrize(
    "bit_generator",
    [np.random.PCG64DXSM, np.random.MT19937],
)
def test_checkpoint_restores_exact_next_rng_draws(
    tmp_path: Path,
    bit_generator: type[np.random.BitGenerator],
) -> None:
    cfg = PhysicalCavityConfig(nx=2, ny=2, particles_per_cell=3, seed=71)
    rng = np.random.Generator(bit_generator(9981))
    checkpoint = _runtime_checkpoint(cfg, rng)
    path = tmp_path / "rng.npz"
    save_ntc_checkpoint(path, cfg, checkpoint)

    expected_float = rng.random(32)
    expected_integer = rng.integers(0, 2**31, size=32, dtype=np.int64)
    loaded = load_ntc_checkpoint(path, cfg)
    resumed = np.random.Generator(bit_generator())
    loaded.restore_rng(resumed)

    np.testing.assert_array_equal(resumed.random(32), expected_float)
    np.testing.assert_array_equal(
        resumed.integers(0, 2**31, size=32, dtype=np.int64),
        expected_integer,
    )


def test_checkpoint_rejects_config_mismatch(tmp_path: Path) -> None:
    cfg = PhysicalCavityConfig(nx=2, ny=2, particles_per_cell=3, seed=71)
    checkpoint = _runtime_checkpoint(cfg, np.random.default_rng(2))
    path = tmp_path / "config.npz"
    save_ntc_checkpoint(path, cfg, checkpoint)

    with pytest.raises(NTCCheckpointConfigMismatchError, match="fingerprint"):
        load_ntc_checkpoint(
            path,
            PhysicalCavityConfig(nx=2, ny=2, particles_per_cell=4, seed=71),
        )


def test_checkpoint_rejects_array_tampering(tmp_path: Path) -> None:
    cfg = PhysicalCavityConfig(nx=2, ny=2, particles_per_cell=3, seed=71)
    checkpoint = _runtime_checkpoint(cfg, np.random.default_rng(2))
    path = tmp_path / "tampered.npz"
    save_ntc_checkpoint(path, cfg, checkpoint)

    with np.load(path, allow_pickle=False) as archive:
        records = {key: np.array(archive[key], copy=True) for key in archive.files}
    array_key = sorted(key for key in records if key.startswith("array_"))[0]
    records[array_key].reshape(-1)[0] += 1.0
    np.savez_compressed(path, **records)

    with pytest.raises(NTCCheckpointCorruptionError, match="hash mismatch"):
        load_ntc_checkpoint(path, cfg)
