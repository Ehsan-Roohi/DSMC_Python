"""Preregistered M1R wall-streaming repair runner.

This module is deliberately separate from :mod:`mohammadzadeh_production`.
The completed M1 checkpoints fingerprint that legacy module and its source
bundle, so changing it would invalidate their provenance.  M1R reuses the
frozen collision and sampling implementations while replacing only free
flight plus wall handling with chronological, event-driven streaming.

Only the preregistered ``P0_event_mechanics`` and ``S1R_mechanics_32`` stages
are accepted here.  P0 never reads the digitized external profiles.  S1R
verifies their hashes and evaluates them only after a complete trajectory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .event_wall_streaming import (
    EventWallStreamingDiagnostics,
    WALL_ORDER,
    stream_with_diffuse_walls,
)
from .mohammadzadeh_production import (
    M1_SOURCE_FILES,
    _atomic_save_npz,
    _atomic_write_json,
    _block_index,
    _finish_block_fields,
    _finish_fields,
    _lid_payload,
    _moment_payload,
    _restore_lid,
    _restore_moment,
    _stationarity_report,
    _strict_json_ready,
)
from .mohammadzadeh_validation import (
    _profile_at_y,
    _zero_crossings,
    evaluate_mohammadzadeh_fields,
    mohammadzadeh_config,
    reference_directory,
)
from .moment_sampling import PhysicalMomentAccumulator
from .ntc_checkpoint import NTCCheckpoint, load_ntc_checkpoint, save_ntc_checkpoint
from .ntc_solver import collide_vhs_ntc
from .vhs_model import (
    PhysicalCavityConfig,
    PhysicalParticleState,
    initialize_physical_state,
)
from .wall_sampling import LidWallEventAccumulator


M1R_PROTOCOL_FILE = "m1r_repair_protocol.json"
M1R_SEED_FILE = "m1r_seed_bank.json"
M1R_LOCK_FILE = "m1r_lock_manifest.json"
M1R_LOCK_STATUS = "locked_before_first_M1R_physical_trajectory"
M1R_ALLOWED_STAGES = ("P0_event_mechanics", "S1R_mechanics_32")

# Importing the frozen M1 helpers above executes the legacy module, whose
# imports are exactly described by M1_SOURCE_FILES.  Fingerprint that complete
# frozen bundle plus the two new physical-source files rather than selecting a
# convenient subset after outcomes are known.
M1R_SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            "mohammadzadeh_production_m1r.py",
            "event_wall_streaming.py",
            *M1_SOURCE_FILES,
        )
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_bundle_sha256() -> str:
    source_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in M1R_SOURCE_FILES:
        path = source_dir / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _seed_group(seed_bank: Mapping[str, Any], path: str) -> list[int]:
    value: Any = seed_bank
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"unknown M1R seed group {path!r}")
        value = value[component]
    if not isinstance(value, list) or not value:
        raise ValueError(f"M1R seed group {path!r} is empty")
    result = [int(seed) for seed in value]
    if len(result) != len(set(result)):
        raise ValueError(f"M1R seed group {path!r} contains duplicates")
    return result


def _validate_seed_bank(seed_bank: Mapping[str, Any]) -> None:
    development = seed_bank.get("development")
    excluded = seed_bank.get("prior_or_excluded_seeds")
    if not isinstance(development, Mapping) or not isinstance(excluded, Mapping):
        raise ValueError("M1R seed bank lacks development/exclusion groups")
    development_groups = {
        name: [int(seed) for seed in value]
        for name, value in development.items()
        if isinstance(value, list)
    }
    if set(development_groups) != {
        "P0_event_mechanics",
        "P1_event_backend_parity",
        "S1R_mechanics_32",
    }:
        raise ValueError("M1R seed bank has unexpected development groups")
    flattened = [seed for group in development_groups.values() for seed in group]
    if len(flattened) != len(set(flattened)):
        raise ValueError("M1R development seed groups overlap")
    prior = {
        int(seed)
        for value in excluded.values()
        if isinstance(value, list)
        for seed in value
    }
    if set(flattened) & prior:
        raise ValueError("M1R seeds overlap M1 prior/reserved seeds")


def _attested_reference_bundle_sha256(
    protocol: Mapping[str, Any],
    m1_lock: Mapping[str, Any],
    m1_lock_sha256: str,
) -> str:
    """Hash names and frozen attestations without reading profile contents.

    This is used while configuring P0, whose protocol explicitly forbids
    reading or evaluating external profiles.  Before S1R evaluation, the
    attested file hashes are compared with the actual files separately.
    """
    recorded = m1_lock.get("files")
    if not isinstance(recorded, Mapping):
        raise ValueError("M1 lock manifest has no file attestations")
    digest = hashlib.sha256()
    for raw_name in protocol.get("reference_bundle", []):
        name = str(raw_name)
        if name == "m1_lock_manifest.json":
            attestation = m1_lock_sha256
        else:
            key = f"reference_data/mohammadzadeh_2012/{name}"
            attestation = recorded.get(key)
            if not isinstance(attestation, str) or len(attestation) != 64:
                raise ValueError(f"M1 lock does not attest {name!r}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(attestation.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _verify_m1r_lock_manifest(
    manifest: Mapping[str, Any],
    expected_hashes: Mapping[str, str],
) -> None:
    """Require a pretrajectory manifest for the four mutable lock inputs.

    The manifest intentionally does not attest its own bytes.  Its file hash
    is added to each checkpoint only after these independent attestations
    match, avoiding a self-referential lock construction.
    """
    if manifest.get("status") != M1R_LOCK_STATUS:
        raise ValueError("M1R lock manifest is not in pretrajectory locked status")
    recorded = manifest.get("hashes")
    if not isinstance(recorded, Mapping):
        raise ValueError("M1R lock manifest has no hash attestations")
    required = (
        "repair_protocol_sha256",
        "seed_bank_sha256",
        "m1_lock_manifest_sha256",
        "source_bundle_sha256",
    )
    if set(recorded) != set(required):
        raise ValueError("M1R lock manifest hash keys differ from the required set")
    for name in required:
        expected = expected_hashes.get(name)
        actual = recorded.get(name)
        if not isinstance(actual, str) or len(actual) != 64:
            raise ValueError(f"M1R lock manifest has malformed {name}")
        if actual != expected:
            raise ValueError(f"M1R lock manifest {name} mismatch")


def _load_m1r_locks() -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    ref_dir = reference_directory()
    protocol_path = ref_dir / M1R_PROTOCOL_FILE
    seed_path = ref_dir / M1R_SEED_FILE
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    seed_bank = json.loads(seed_path.read_text(encoding="utf-8"))
    if protocol.get("status") != (
        "locked_after_M1_postmortem_and_before_any_M1R_physical_trajectory"
    ):
        raise ValueError("M1R protocol is not in the locked preregistered state")
    if protocol.get("anti_circularity", {}).get("seed_bank") != M1R_SEED_FILE:
        raise ValueError("M1R protocol points to an unexpected seed bank")
    if protocol.get("scientific_threshold_changes") != "none":
        raise ValueError("M1R scientific thresholds are not frozen")
    if not protocol.get("isolated_source_change", {}).get(
        "legacy_files_are_immutable"
    ):
        raise ValueError("M1R protocol does not protect legacy M1 sources")
    _validate_seed_bank(seed_bank)

    m1_lock_path = ref_dir / "m1_lock_manifest.json"
    m1_lock_sha256 = _sha256(m1_lock_path)
    m1_lock = json.loads(m1_lock_path.read_text(encoding="utf-8"))
    composite = m1_lock.get("composite_hashes")
    if not isinstance(composite, Mapping):
        raise ValueError("M1 lock manifest has no composite hashes")
    recorded_files = m1_lock.get("files")
    if not isinstance(recorded_files, Mapping):
        raise ValueError("M1 lock manifest has no file attestations")
    frozen_source_dir = Path(__file__).resolve().parent
    for name in M1_SOURCE_FILES:
        expected = recorded_files.get(f"vgdsmc/{name}")
        actual = _sha256(frozen_source_dir / name)
        if expected != actual:
            raise ValueError(
                f"frozen M1 source hash mismatch for {name}; "
                "M1R cannot continue"
            )
    reference_bundle = composite.get("reference_bundle_sha256")
    if not isinstance(reference_bundle, str) or len(reference_bundle) != 64:
        raise ValueError("M1 reference bundle attestation is malformed")
    lock_hashes = {
        "repair_protocol_sha256": _sha256(protocol_path),
        "seed_bank_sha256": _sha256(seed_path),
        "m1_lock_manifest_sha256": m1_lock_sha256,
        "m1_reference_bundle_sha256": reference_bundle,
        "repair_reference_attestation_sha256": (
            _attested_reference_bundle_sha256(
                protocol,
                m1_lock,
                m1_lock_sha256,
            )
        ),
        "source_bundle_sha256": _source_bundle_sha256(),
    }
    pretrajectory_path = ref_dir / M1R_LOCK_FILE
    if not pretrajectory_path.is_file():
        raise FileNotFoundError(
            f"required pretrajectory M1R lock is missing: {pretrajectory_path}"
        )
    pretrajectory_manifest = json.loads(
        pretrajectory_path.read_text(encoding="utf-8")
    )
    if not isinstance(pretrajectory_manifest, Mapping):
        raise ValueError("M1R lock manifest must be a JSON object")
    _verify_m1r_lock_manifest(pretrajectory_manifest, lock_hashes)
    lock_hashes["m1r_lock_manifest_sha256"] = _sha256(pretrajectory_path)
    return protocol, seed_bank, lock_hashes


def stage_configuration(
    stage: str,
    seed: int,
) -> tuple[PhysicalCavityConfig, dict[str, Any], dict[str, Any], dict[str, str]]:
    protocol, seed_bank, lock_hashes = _load_m1r_locks()
    if stage not in M1R_ALLOWED_STAGES:
        raise ValueError(
            f"M1R runner accepts only {M1R_ALLOWED_STAGES}, not {stage!r}"
        )
    specification = protocol.get("stages", {}).get(stage)
    if not isinstance(specification, dict):
        raise ValueError(f"unknown M1R stage {stage!r}")
    if specification.get("backend") != "reference":
        raise ValueError(f"{stage} is not locked to the reference backend")
    allowed_seeds = _seed_group(seed_bank, str(specification["seed_group"]))
    if seed not in allowed_seeds:
        raise ValueError(
            f"seed {seed} is not preregistered for {stage}; "
            f"allowed={allowed_seeds}"
        )
    cfg = mohammadzadeh_config(
        grid=int(specification["grid"]),
        particles_per_cell=int(protocol["primary_case"]["particles_per_cell"]),
        steps=int(specification["steps"]),
        sample_start=int(specification["sample_start"]),
        seed=seed,
        dt_safety=float(specification["dt_safety"]),
    )
    return cfg, protocol, specification, lock_hashes


@dataclass
class EventDiagnosticsAccumulator:
    """Additive/checkpointable diagnostics from chronological wall streaming."""

    initial_particle_count: int
    initial_relative_weight: float
    stream_calls: int
    particle_count_mismatch_calls: int
    particle_count_delta_abs_max: int
    relative_weight_delta_abs_max: float
    total_wall_hits: int
    maximum_hits_on_one_particle: int
    particles_with_multiple_hits: int
    zero_time_wall_hits: int
    wall_hits: np.ndarray
    incident_counts: np.ndarray
    reflected_counts: np.ndarray
    incident_relative_weight: np.ndarray
    reflected_relative_weight: np.ndarray
    exact_corner_ties: int
    incident_sign_violations: int
    reflected_sign_violations: int
    nonmonotone_hit_time: int
    fallback_clip_count: int
    cap_exhaustion: int

    @classmethod
    def create(
        cls,
        initial_particle_count: int,
        initial_relative_weight: float,
    ) -> "EventDiagnosticsAccumulator":
        return cls(
            initial_particle_count=int(initial_particle_count),
            initial_relative_weight=float(initial_relative_weight),
            stream_calls=0,
            particle_count_mismatch_calls=0,
            particle_count_delta_abs_max=0,
            relative_weight_delta_abs_max=0.0,
            total_wall_hits=0,
            maximum_hits_on_one_particle=0,
            particles_with_multiple_hits=0,
            zero_time_wall_hits=0,
            wall_hits=np.zeros(len(WALL_ORDER), dtype=np.int64),
            incident_counts=np.zeros(len(WALL_ORDER), dtype=np.int64),
            reflected_counts=np.zeros(len(WALL_ORDER), dtype=np.int64),
            incident_relative_weight=np.zeros(len(WALL_ORDER)),
            reflected_relative_weight=np.zeros(len(WALL_ORDER)),
            exact_corner_ties=0,
            incident_sign_violations=0,
            reflected_sign_violations=0,
            nonmonotone_hit_time=0,
            fallback_clip_count=0,
            cap_exhaustion=0,
        )

    def add(self, diagnostics: EventWallStreamingDiagnostics) -> None:
        self.stream_calls += 1
        if diagnostics.particle_count != self.initial_particle_count:
            self.particle_count_mismatch_calls += 1
        self.particle_count_delta_abs_max = max(
            self.particle_count_delta_abs_max,
            abs(int(diagnostics.particle_count_delta)),
        )
        self.relative_weight_delta_abs_max = max(
            self.relative_weight_delta_abs_max,
            abs(float(diagnostics.relative_weight_delta)),
        )
        self.total_wall_hits += int(diagnostics.total_wall_hits)
        self.maximum_hits_on_one_particle = max(
            self.maximum_hits_on_one_particle,
            int(diagnostics.maximum_hits_on_one_particle),
        )
        self.particles_with_multiple_hits += int(
            diagnostics.particles_with_multiple_hits
        )
        self.zero_time_wall_hits += int(diagnostics.zero_time_wall_hits)
        self.wall_hits += np.asarray(diagnostics.wall_hits, dtype=np.int64)
        self.incident_counts += np.asarray(
            diagnostics.incident_counts, dtype=np.int64
        )
        self.reflected_counts += np.asarray(
            diagnostics.reflected_counts, dtype=np.int64
        )
        self.incident_relative_weight += np.asarray(
            diagnostics.incident_relative_weight, dtype=np.float64
        )
        self.reflected_relative_weight += np.asarray(
            diagnostics.reflected_relative_weight, dtype=np.float64
        )
        self.exact_corner_ties += int(diagnostics.exact_corner_ties)
        self.incident_sign_violations += int(
            diagnostics.incident_sign_violations
        )
        self.reflected_sign_violations += int(
            diagnostics.reflected_sign_violations
        )
        self.nonmonotone_hit_time += int(diagnostics.nonmonotone_hit_time)
        self.fallback_clip_count += int(diagnostics.fallback_clip_count)
        self.cap_exhaustion += int(diagnostics.cap_exhaustion)

    def payload(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def restore(cls, payload: Mapping[str, Any]) -> "EventDiagnosticsAccumulator":
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise ValueError("M1R checkpoint event diagnostic keys differ")
        arrays = {
            "wall_hits": np.int64,
            "incident_counts": np.int64,
            "reflected_counts": np.int64,
            "incident_relative_weight": np.float64,
            "reflected_relative_weight": np.float64,
        }
        values = dict(payload)
        for name, dtype in arrays.items():
            array = np.asarray(values[name], dtype=dtype).copy()
            if array.shape != (len(WALL_ORDER),):
                raise ValueError(f"M1R checkpoint {name} has invalid shape")
            values[name] = array
        int_fields = expected - set(arrays) - {
            "initial_relative_weight",
            "relative_weight_delta_abs_max",
        }
        for name in int_fields:
            values[name] = int(values[name])
        values["initial_relative_weight"] = float(
            values["initial_relative_weight"]
        )
        values["relative_weight_delta_abs_max"] = float(
            values["relative_weight_delta_abs_max"]
        )
        return cls(**values)


@dataclass
class M1RRuntime:
    state: PhysicalParticleState
    rng: np.random.Generator
    step_index: int
    moments: PhysicalMomentAccumulator
    lid_events: LidWallEventAccumulator
    block_moments: list[PhysicalMomentAccumulator]
    block_lid_events: list[LidWallEventAccumulator]
    temporal_sums: dict[str, np.ndarray]
    temporal_sums2: dict[str, np.ndarray]
    temporal_nsamples: int
    collision_diagnostics: dict[str, int | float]
    event_diagnostics: EventDiagnosticsAccumulator


def _new_collision_diagnostics() -> dict[str, int | float]:
    return {
        "candidate_collisions": 0,
        "accepted_collisions": 0,
        "majorant_violations": 0,
        "max_acceptance_ratio": 0.0,
    }


def _new_runtime(cfg: PhysicalCavityConfig, block_count: int) -> M1RRuntime:
    state = initialize_physical_state(cfg)
    return M1RRuntime(
        state=state,
        rng=np.random.default_rng(cfg.seed + 29),
        step_index=0,
        moments=PhysicalMomentAccumulator(cfg),
        lid_events=LidWallEventAccumulator(cfg),
        block_moments=[PhysicalMomentAccumulator(cfg) for _ in range(block_count)],
        block_lid_events=[LidWallEventAccumulator(cfg) for _ in range(block_count)],
        temporal_sums={},
        temporal_sums2={},
        temporal_nsamples=0,
        collision_diagnostics=_new_collision_diagnostics(),
        event_diagnostics=EventDiagnosticsAccumulator.create(
            len(state.pos),
            float(np.sum(state.weight, dtype=np.float64)),
        ),
    )


def _runtime_blocks_payload(runtime: M1RRuntime) -> dict[str, Any]:
    return {
        "block_moments": {
            f"{index:03d}": _moment_payload(accumulator)
            for index, accumulator in enumerate(runtime.block_moments)
        },
        "block_lid_events": {
            f"{index:03d}": _lid_payload(accumulator)
            for index, accumulator in enumerate(runtime.block_lid_events)
        },
        "event_diagnostics": runtime.event_diagnostics.payload(),
    }


def _checkpoint_metadata(
    *,
    stage: str,
    seed: int,
    sample_stride: int,
    block_count: int,
    max_events_per_particle: int,
    lock_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "runner": "M1R_event_wall_streaming",
        "stage": stage,
        "seed": seed,
        "backend": "reference",
        "sample_stride": sample_stride,
        "block_count": block_count,
        "max_events_per_particle": max_events_per_particle,
        "numpy_version": np.__version__,
        **lock_hashes,
    }


def _save_runtime_checkpoint(
    path: Path,
    cfg: PhysicalCavityConfig,
    runtime: M1RRuntime,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = NTCCheckpoint(
        state=runtime.state,
        rng_state=runtime.rng.bit_generator.state,
        step_index=runtime.step_index,
        diagnostics=runtime.collision_diagnostics,
        moments=runtime.moments,
        wall_events=runtime.lid_events,
        temporal_sums=runtime.temporal_sums,
        temporal_sums2=runtime.temporal_sums2,
        temporal_nsamples=runtime.temporal_nsamples,
        block_accumulators=_runtime_blocks_payload(runtime),
        metadata=metadata,
    )
    return save_ntc_checkpoint(path, cfg, checkpoint)


def _load_runtime_checkpoint(
    path: Path,
    cfg: PhysicalCavityConfig,
    expected_metadata: Mapping[str, Any],
) -> M1RRuntime:
    checkpoint = load_ntc_checkpoint(path, cfg)
    if dict(checkpoint.metadata) != dict(expected_metadata):
        raise ValueError("checkpoint M1R metadata differs from the locked run")
    payload = checkpoint.block_accumulators
    if not isinstance(payload, Mapping):
        raise ValueError("M1R checkpoint has no block payload")
    moment_payloads = payload.get("block_moments")
    lid_payloads = payload.get("block_lid_events")
    event_payload = payload.get("event_diagnostics")
    if not all(
        isinstance(value, Mapping)
        for value in (moment_payloads, lid_payloads, event_payload)
    ):
        raise ValueError("M1R checkpoint block payload is malformed")
    assert isinstance(moment_payloads, Mapping)
    assert isinstance(lid_payloads, Mapping)
    assert isinstance(event_payload, Mapping)
    if sorted(moment_payloads) != sorted(lid_payloads):
        raise ValueError("M1R checkpoint block moment/lid keys differ")
    rng = np.random.default_rng()
    checkpoint.restore_rng(rng)
    return M1RRuntime(
        state=checkpoint.state,
        rng=rng,
        step_index=checkpoint.step_index,
        moments=checkpoint.moments,
        lid_events=checkpoint.wall_events,
        block_moments=[
            _restore_moment(cfg, moment_payloads[key])
            for key in sorted(moment_payloads)
        ],
        block_lid_events=[
            _restore_lid(cfg, lid_payloads[key]) for key in sorted(lid_payloads)
        ],
        temporal_sums={
            key: np.asarray(value).copy()
            for key, value in checkpoint.temporal_sums.items()
        },
        temporal_sums2={
            key: np.asarray(value).copy()
            for key, value in checkpoint.temporal_sums2.items()
        },
        temporal_nsamples=checkpoint.temporal_nsamples,
        collision_diagnostics=dict(checkpoint.diagnostics),
        event_diagnostics=EventDiagnosticsAccumulator.restore(event_payload),
    )


def _tree_equal(first: Any, second: Any) -> bool:
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return isinstance(first, np.ndarray) and isinstance(
            second, np.ndarray
        ) and np.array_equal(first, second, equal_nan=True)
    if isinstance(first, Mapping) or isinstance(second, Mapping):
        return (
            isinstance(first, Mapping)
            and isinstance(second, Mapping)
            and set(first) == set(second)
            and all(_tree_equal(first[key], second[key]) for key in first)
        )
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        return (
            type(first) is type(second)
            and len(first) == len(second)
            and all(_tree_equal(a, b) for a, b in zip(first, second))
        )
    if isinstance(first, float) and isinstance(second, float):
        return first == second or (np.isnan(first) and np.isnan(second))
    return bool(first == second)


def _runtime_checkpoint_roundtrip_identity(
    first: M1RRuntime,
    second: M1RRuntime,
) -> bool:
    return all(
        (
            np.array_equal(first.state.pos, second.state.pos),
            np.array_equal(first.state.vel, second.state.vel),
            np.array_equal(first.state.weight, second.state.weight),
            first.step_index == second.step_index,
            _tree_equal(
                first.rng.bit_generator.state,
                second.rng.bit_generator.state,
            ),
            _tree_equal(
                first.collision_diagnostics,
                second.collision_diagnostics,
            ),
            _tree_equal(_moment_payload(first.moments), _moment_payload(second.moments)),
            _tree_equal(_lid_payload(first.lid_events), _lid_payload(second.lid_events)),
            _tree_equal(
                _runtime_blocks_payload(first),
                _runtime_blocks_payload(second),
            ),
            _tree_equal(first.temporal_sums, second.temporal_sums),
            _tree_equal(first.temporal_sums2, second.temporal_sums2),
            first.temporal_nsamples == second.temporal_nsamples,
        )
    )


def _relative_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    difference = np.abs(first - second)
    scale = 0.5 * (np.abs(first) + np.abs(second))
    return np.divide(
        difference,
        scale,
        out=np.where(difference == 0.0, 0.0, np.inf),
        where=scale > 0.0,
    )


def event_mechanics_report(
    accumulator: EventDiagnosticsAccumulator,
    state: PhysicalParticleState,
    cfg: PhysicalCavityConfig,
    gate_specification: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate exactly the preregistered M1R event-mechanics gates."""
    final_count = len(state.pos)
    final_weight = float(np.sum(state.weight, dtype=np.float64))
    particle_count_delta = final_count - accumulator.initial_particle_count
    weight_delta = final_weight - accumulator.initial_relative_weight
    weight_scale = max(abs(accumulator.initial_relative_weight), 1.0e-300)
    relative_weight_delta_abs_max = max(
        abs(weight_delta) / weight_scale,
        accumulator.relative_weight_delta_abs_max / weight_scale,
    )
    count_difference = accumulator.reflected_counts - accumulator.incident_counts
    weight_relative_difference = _relative_difference(
        accumulator.incident_relative_weight,
        accumulator.reflected_relative_weight,
    )
    out_of_domain = int(
        np.count_nonzero(
            np.any(
                (state.pos < 0.0) | (state.pos > cfg.length),
                axis=1,
            )
        )
    )
    finite_state = bool(
        np.all(np.isfinite(state.pos))
        and np.all(np.isfinite(state.vel))
        and np.all(np.isfinite(state.weight))
    )
    checks = {
        "particle_count_delta": particle_count_delta
        == int(gate_specification["particle_count_delta"]),
        "relative_particle_weight_delta_abs_max": relative_weight_delta_abs_max
        <= float(
            gate_specification["relative_particle_weight_delta_abs_max"]
        ),
        "incident_reflected_count_difference_per_wall": bool(
            np.all(
                count_difference
                == int(
                    gate_specification[
                        "incident_reflected_count_difference_per_wall"
                    ]
                )
            )
        ),
        "incident_reflected_weight_relative_difference_per_wall_max": bool(
            np.all(
                weight_relative_difference
                <= float(
                    gate_specification[
                        "incident_reflected_weight_relative_difference_per_wall_max"
                    ]
                )
            )
        ),
        "incident_wall_relative_sign_violations": (
            accumulator.incident_sign_violations
            == int(gate_specification["incident_wall_relative_sign_violations"])
        ),
        "reflected_wall_relative_sign_violations": (
            accumulator.reflected_sign_violations
            == int(gate_specification["reflected_wall_relative_sign_violations"])
        ),
        "nonmonotone_hit_times": accumulator.nonmonotone_hit_time
        == int(gate_specification["nonmonotone_hit_times"]),
        "out_of_domain_final_positions": out_of_domain
        == int(gate_specification["out_of_domain_final_positions"]),
        "fallback_position_clips": accumulator.fallback_clip_count
        == int(gate_specification["fallback_position_clips"]),
        "event_cap_exhaustions": accumulator.cap_exhaustion
        == int(gate_specification["event_cap_exhaustions"]),
    }
    accounting_consistent = bool(
        accumulator.particle_count_mismatch_calls == 0
        and accumulator.particle_count_delta_abs_max == 0
        and np.array_equal(accumulator.wall_hits, accumulator.incident_counts)
        and np.array_equal(accumulator.wall_hits, accumulator.reflected_counts)
        and accumulator.total_wall_hits == int(np.sum(accumulator.wall_hits))
    )
    metrics = {
        "stream_calls": accumulator.stream_calls,
        "initial_particle_count": accumulator.initial_particle_count,
        "final_particle_count": final_count,
        "particle_count_delta": particle_count_delta,
        "particle_count_mismatch_calls": accumulator.particle_count_mismatch_calls,
        "particle_count_delta_abs_max_per_call": (
            accumulator.particle_count_delta_abs_max
        ),
        "initial_relative_particle_weight": accumulator.initial_relative_weight,
        "final_relative_particle_weight": final_weight,
        "relative_particle_weight_delta_abs_max": (
            relative_weight_delta_abs_max
        ),
        "wall_order": list(WALL_ORDER),
        "total_wall_hits": accumulator.total_wall_hits,
        "wall_hits": accumulator.wall_hits,
        "incident_counts": accumulator.incident_counts,
        "reflected_counts": accumulator.reflected_counts,
        "incident_reflected_count_difference": count_difference,
        "incident_relative_weight": accumulator.incident_relative_weight,
        "reflected_relative_weight": accumulator.reflected_relative_weight,
        "incident_reflected_weight_relative_difference": (
            weight_relative_difference
        ),
        "maximum_hits_on_one_particle_per_step": (
            accumulator.maximum_hits_on_one_particle
        ),
        "particles_with_multiple_hits_summed_over_steps": (
            accumulator.particles_with_multiple_hits
        ),
        "zero_time_wall_hits": accumulator.zero_time_wall_hits,
        "exact_corner_ties": accumulator.exact_corner_ties,
        "incident_sign_violations": accumulator.incident_sign_violations,
        "reflected_sign_violations": accumulator.reflected_sign_violations,
        "nonmonotone_hit_times": accumulator.nonmonotone_hit_time,
        "out_of_domain_final_positions": out_of_domain,
        "fallback_position_clips": accumulator.fallback_clip_count,
        "event_cap_exhaustions": accumulator.cap_exhaustion,
        "finite_particle_state": finite_state,
        "accounting_consistent": accounting_consistent,
    }
    return {
        "metrics": metrics,
        "checks": checks,
        "accounting_consistent": accounting_consistent,
        "all_passed": bool(all(checks.values()) and accounting_consistent),
    }


def _verify_reference_bundle_for_s1r(protocol: Mapping[str, Any]) -> None:
    """Verify actual external inputs immediately before S1R evaluation."""
    ref_dir = reference_directory()
    m1_lock = json.loads(
        (ref_dir / "m1_lock_manifest.json").read_text(encoding="utf-8")
    )
    recorded = m1_lock.get("files")
    if not isinstance(recorded, Mapping):
        raise ValueError("M1 lock manifest has no file attestations")
    for raw_name in protocol.get("reference_bundle", []):
        name = str(raw_name)
        if name == "m1_lock_manifest.json":
            continue
        key = f"reference_data/mohammadzadeh_2012/{name}"
        expected = recorded.get(key)
        if expected != _sha256(ref_dir / name):
            raise ValueError(f"S1R external reference hash mismatch for {name}")


def _heat_flux_zero_crossing_counts(
    fields: Mapping[str, np.ndarray],
    cfg: PhysicalCavityConfig,
    evaluation: Mapping[str, Any],
) -> dict[str, int]:
    x_centers = (np.arange(cfg.nx, dtype=np.float64) + 0.5) / cfg.nx
    qy = _profile_at_y(np.asarray(fields["qy"]), 0.8)
    positive_scale = float(np.max(qy))
    normalized = (
        qy / positive_scale
        if positive_scale > 0.0
        else np.full_like(qy, np.nan)
    )
    arrays = evaluation.get("comparison_arrays")
    if not isinstance(arrays, Mapping):
        raise ValueError("S1R evaluation lacks comparison arrays")
    reference_x = np.asarray(arrays["reference_qy_x"])
    reference_qy = np.asarray(arrays["reference_qy_normalized"])
    return {
        "simulated": int(len(_zero_crossings(x_centers, normalized))),
        "reference": int(len(_zero_crossings(reference_x, reference_qy))),
    }


def _guard_output_directory(output_dir: Path) -> None:
    legacy = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "mohammadzadeh_2012"
        / "m1_s1_mechanics_32"
    ).resolve()
    destination = output_dir.resolve()
    if destination == legacy or destination.is_relative_to(legacy):
        raise ValueError("M1R may not overwrite or enter the frozen M1 output tree")


def run_m1r_seed(
    *,
    stage: str,
    seed: int,
    output_dir: Path,
    backend: str = "reference",
    resume: bool = True,
    stop_after_step: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Run one preregistered M1R seed with exact checkpoint/resume."""
    _guard_output_directory(output_dir)
    cfg, protocol, specification, lock_hashes = stage_configuration(stage, seed)
    if backend != "reference" or specification.get("backend") != "reference":
        raise ValueError(f"{stage} locks backend='reference', not {backend!r}")
    contract = protocol["runtime_contract"]
    sample_stride = int(contract["sample_stride"])
    block_count = int(contract["nonoverlapping_sampling_blocks"])
    checkpoint_interval = int(contract["checkpoint_interval_steps"])
    max_events = int(contract["maximum_events_per_particle_per_step"])
    metadata = _checkpoint_metadata(
        stage=stage,
        seed=seed,
        sample_stride=sample_stride,
        block_count=block_count,
        max_events_per_particle=max_events,
        lock_hashes=lock_hashes,
    )
    checkpoint_path = output_dir / "checkpoint.npz"
    final_names = (
        "fields.npz",
        "block_fields.npz",
        "summary.json",
        "artifact_manifest.json",
    )
    existing_final = [name for name in final_names if (output_dir / name).exists()]
    if not resume and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("--no-resume requires an empty output directory")
    if resume and not checkpoint_path.exists() and existing_final:
        raise FileExistsError(
            "final M1R artifacts exist without their checkpoint; use a new directory"
        )
    if resume and checkpoint_path.exists():
        runtime = _load_runtime_checkpoint(checkpoint_path, cfg, metadata)
    else:
        runtime = _new_runtime(cfg, block_count)
    if runtime.step_index < cfg.steps and existing_final:
        raise ValueError("incomplete M1R checkpoint has stale final artifacts")
    if len(runtime.block_moments) != block_count:
        raise ValueError("checkpoint block count differs from M1R lock")
    target_step = cfg.steps if stop_after_step is None else min(
        cfg.steps, stop_after_step
    )
    if target_step < runtime.step_index:
        raise ValueError("stop_after_step precedes the M1R checkpoint step")

    while runtime.step_index < target_step:
        step = runtime.step_index
        block_index = (
            _block_index(step, cfg, block_count)
            if step >= cfg.sample_start
            else None
        )

        def wall_handler(
            wall: str,
            tangential_position: np.ndarray,
            velocity: np.ndarray,
            weight: np.ndarray,
            wall_velocity: np.ndarray,
        ) -> None:
            assert block_index is not None
            runtime.lid_events.add(
                wall,
                tangential_position,
                velocity,
                weight,
                wall_velocity,
            )
            runtime.block_lid_events[block_index].add(
                wall,
                tangential_position,
                velocity,
                weight,
                wall_velocity,
            )

        event = stream_with_diffuse_walls(
            runtime.state,
            cfg,
            cfg.dt,
            runtime.rng,
            wall_event_handler=(wall_handler if step >= cfg.sample_start else None),
            max_events_per_particle=max_events,
        )
        runtime.event_diagnostics.add(event)
        collision = collide_vhs_ntc(runtime.state, cfg, runtime.rng)
        runtime.collision_diagnostics["candidate_collisions"] = int(
            runtime.collision_diagnostics["candidate_collisions"]
        ) + collision.candidate_collisions
        runtime.collision_diagnostics["accepted_collisions"] = int(
            runtime.collision_diagnostics["accepted_collisions"]
        ) + collision.accepted_collisions
        runtime.collision_diagnostics["majorant_violations"] = int(
            runtime.collision_diagnostics["majorant_violations"]
        ) + collision.majorant_violations
        runtime.collision_diagnostics["max_acceptance_ratio"] = max(
            float(runtime.collision_diagnostics["max_acceptance_ratio"]),
            collision.max_acceptance_ratio,
        )

        if step >= cfg.sample_start and (step - cfg.sample_start) % sample_stride == 0:
            assert block_index is not None
            instantaneous = runtime.moments.add(
                runtime.state,
                return_instantaneous=True,
            )
            assert instantaneous is not None
            runtime.block_moments[block_index].add(runtime.state)
            for key in ("T", "u", "v", "w"):
                value = instantaneous[key]
                runtime.temporal_sums[key] = (
                    runtime.temporal_sums.get(key, np.zeros_like(value)) + value
                )
                runtime.temporal_sums2[key] = (
                    runtime.temporal_sums2.get(key, np.zeros_like(value)) + value**2
                )
            runtime.temporal_nsamples += 1

        runtime.step_index += 1
        if runtime.step_index % checkpoint_interval == 0:
            _save_runtime_checkpoint(checkpoint_path, cfg, runtime, metadata)
            if progress is not None:
                progress(runtime.step_index, cfg.steps)

    manifest = _save_runtime_checkpoint(checkpoint_path, cfg, runtime, metadata)
    if runtime.step_index < cfg.steps:
        return {
            "stage": stage,
            "seed": seed,
            "status": "checkpointed_incomplete",
            "step_index": runtime.step_index,
            "target_steps": cfg.steps,
            "checkpoint_manifest_sha256": manifest["manifest_sha256"],
        }

    roundtrip = _load_runtime_checkpoint(checkpoint_path, cfg, metadata)
    checkpoint_roundtrip_identity = _runtime_checkpoint_roundtrip_identity(
        runtime, roundtrip
    )
    fields = _finish_fields(runtime)
    block_fields = _finish_block_fields(runtime)
    event_report = event_mechanics_report(
        runtime.event_diagnostics,
        runtime.state,
        cfg,
        protocol["event_mechanics_gates"],
    )
    finite_particle_state = bool(event_report["metrics"]["finite_particle_state"])
    accepted = int(runtime.collision_diagnostics["accepted_collisions"])
    candidates = int(runtime.collision_diagnostics["candidate_collisions"])
    diagnostics = {
        **runtime.collision_diagnostics,
        "acceptance_fraction": accepted / max(candidates, 1),
        "collisions_per_particle_step": accepted
        / (len(runtime.state.pos) * cfg.steps),
        "dt": cfg.dt,
        "number_density": cfg.number_density,
        "mean_free_path": cfg.vhs.mean_free_path(cfg.number_density, cfg.t0),
        "volume_sample_count": runtime.temporal_nsamples,
        "sample_stride": sample_stride,
        "backend": "reference",
    }
    evaluation_summary: dict[str, Any]
    stationarity: dict[str, Any] | None
    zero_crossings: dict[str, int] | None
    minimum_lid_events = int(np.min(fields["microscopic_lid_event_count"]))
    core_finite = all(
        np.all(np.isfinite(fields[key]))
        for key in ("T", "rho", "u", "v", "qx", "qy")
    )
    if stage == "P0_event_mechanics":
        stationarity = None
        zero_crossings = None
        evaluation_summary = {
            "performed": False,
            "external_profile_files_read": False,
            "reason": "P0 is preregistered for event mechanics only",
        }
        stage_checks = {
            "all_event_mechanics_gates_pass": bool(event_report["all_passed"]),
            "majorant_violations_equal_zero": int(
                runtime.collision_diagnostics["majorant_violations"]
            )
            == 0,
            "finite_particle_state": finite_particle_state,
            "checkpoint_roundtrip_bitwise_identity": (
                checkpoint_roundtrip_identity
            ),
        }
        decision = (
            "pass_P0_event_mechanics"
            if all(stage_checks.values())
            else "hold_P0_event_mechanics"
        )
    else:
        stationarity = _stationarity_report(
            block_fields,
            cfg,
            fields,
            z_limit=2.0,
            minimum_finite_per_half=int(
                protocol["stationarity_contract"][
                    "minimum_finite_blocks_per_half"
                ]
            ),
        )
        _verify_reference_bundle_for_s1r(protocol)
        evaluation = evaluate_mohammadzadeh_fields(fields, cfg)
        zero_crossings = _heat_flux_zero_crossing_counts(fields, cfg, evaluation)
        evaluation_summary = {
            key: value for key, value in evaluation.items() if key != "comparison_arrays"
        }
        stage_checks = {
            "all_event_mechanics_gates_pass": bool(event_report["all_passed"]),
            "majorant_violations_equal_zero": int(
                runtime.collision_diagnostics["majorant_violations"]
            )
            == 0,
            "finite_nonempty_fields": bool(
                core_finite and runtime.moments.samples > 0
            ),
            "complete_lid_event_bin_coverage": minimum_lid_events
            >= int(contract["minimum_wall_event_coverage_per_lid_bin"]),
            "stationarity_pass": bool(stationarity["all_passed"]),
            "heat_flux_sign_pass": bool(
                evaluation["checks"]["fig9_qy_sign"]
            ),
            "heat_flux_zero_crossing_pass": bool(
                evaluation["checks"]["fig9_qy_zero_crossing"]
            ),
            "checkpoint_roundtrip_bitwise_identity": (
                checkpoint_roundtrip_identity
            ),
        }
        decision = (
            "seed_passes_S1R_mechanics_awaiting_two_seed_aggregation"
            if all(stage_checks.values())
            else "seed_fails_S1R_mechanics_awaiting_two_seed_aggregation"
        )

    summary = {
        "stage": stage,
        "purpose": specification["purpose"],
        "status": "complete_development_seed_awaiting_stage_aggregation",
        "repair_scope": (
            "event-driven wall streaming only; not external validation"
        ),
        "seed": seed,
        "config": asdict(cfg),
        "lock_hashes": lock_hashes,
        "source_files": list(M1R_SOURCE_FILES),
        "checkpoint_manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_roundtrip_bitwise_identity": checkpoint_roundtrip_identity,
        "diagnostics": diagnostics,
        "event_mechanics": event_report,
        "minimum_lid_events_per_bin": minimum_lid_events,
        "macroscopic_wall_adjacent_profile_location": {
            "sampling": "top_cell_center",
            "y_over_l": 1.0 - 0.5 / cfg.ny,
        },
        "stationarity": stationarity,
        "heat_flux_zero_crossing_count": zero_crossings,
        "mechanical_checks": stage_checks,
        "evaluation": evaluation_summary,
        "decision": decision,
    }
    _atomic_save_npz(output_dir / "fields.npz", fields)
    _atomic_save_npz(output_dir / "block_fields.npz", block_fields)
    _atomic_write_json(output_dir / "summary.json", summary)
    artifact_manifest = {
        "stage": stage,
        "seed": seed,
        "status": summary["status"],
        "files": {
            name: {
                "sha256": _sha256(output_dir / name),
                "size_bytes": (output_dir / name).stat().st_size,
            }
            for name in (
                "checkpoint.npz",
                "fields.npz",
                "block_fields.npz",
                "summary.json",
            )
        },
        "lock_hashes": lock_hashes,
    }
    _atomic_write_json(output_dir / "artifact_manifest.json", artifact_manifest)
    return _strict_json_ready(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=M1R_ALLOWED_STAGES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("reference",), default="reference")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    summary = run_m1r_seed(
        stage=args.stage,
        seed=args.seed,
        output_dir=args.output_dir,
        backend=args.backend,
        resume=not args.no_resume,
        stop_after_step=args.stop_after_step,
        progress=lambda step, total: print(
            json.dumps({"step": step, "total": total}), flush=True
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
