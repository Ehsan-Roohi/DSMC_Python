from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

import numpy as np

from .mohammadzadeh_validation import (
    _profile_at_y,
    evaluate_mohammadzadeh_fields,
    mohammadzadeh_config,
    reference_directory,
)
from .moment_sampling import PhysicalMomentAccumulator
from .ntc_checkpoint import (
    NTCCheckpoint,
    load_ntc_checkpoint,
    save_ntc_checkpoint,
)
from .ntc_fast import collide_vhs_ntc_fast
from .ntc_solver import collide_vhs_ntc
from .vhs_model import (
    PhysicalCavityConfig,
    PhysicalParticleState,
    apply_diffuse_walls,
    initialize_physical_state,
)
from .wall_balance import WallBalanceAccumulator
from .wall_sampling import LidWallEventAccumulator


M1_PROTOCOL_FILE = "m1_execution_protocol.json"
M1_SEED_FILE = "m1_seed_bank.json"
M1_SOURCE_FILES = (
    "mohammadzadeh_production.py",
    "mohammadzadeh_statistics.py",
    "mohammadzadeh_validation.py",
    "moment_sampling.py",
    "ntc_checkpoint.py",
    "ntc_fast.py",
    "ntc_solver.py",
    "vhs_model.py",
    "wall_balance.py",
    "wall_sampling.py",
)


@dataclass
class M1Runtime:
    state: PhysicalParticleState
    rng: np.random.Generator
    step_index: int
    moments: PhysicalMomentAccumulator
    lid_events: LidWallEventAccumulator
    wall_balance: WallBalanceAccumulator
    block_moments: list[PhysicalMomentAccumulator]
    block_lid_events: list[LidWallEventAccumulator]
    temporal_sums: dict[str, np.ndarray]
    temporal_sums2: dict[str, np.ndarray]
    temporal_nsamples: int
    diagnostics: dict[str, int | float]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_bundle_sha256() -> str:
    source_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in M1_SOURCE_FILES:
        path = source_dir / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _named_bundle_sha256(directory: Path, names: list[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        path = directory / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_m1_locks() -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    ref_dir = reference_directory()
    protocol_path = ref_dir / M1_PROTOCOL_FILE
    seed_path = ref_dir / M1_SEED_FILE
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    seeds = json.loads(seed_path.read_text(encoding="utf-8"))
    if protocol["anti_circularity"]["seed_bank"] != M1_SEED_FILE:
        raise ValueError("M1 protocol points to an unexpected seed bank")
    return protocol, seeds, {
        "execution_protocol_sha256": _sha256(protocol_path),
        "seed_bank_sha256": _sha256(seed_path),
        "reference_bundle_sha256": _named_bundle_sha256(
            ref_dir,
            [str(name) for name in protocol["reference_bundle"]],
        ),
        "source_bundle_sha256": _source_bundle_sha256(),
    }


def _seed_group(seed_bank: dict[str, Any], path: str) -> list[int]:
    value: Any = seed_bank
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise ValueError(f"unknown M1 seed group {path!r}")
        value = value[component]
    if not isinstance(value, list) or not value:
        raise ValueError(f"M1 seed group {path!r} is empty")
    result = [int(seed) for seed in value]
    if len(set(result)) != len(result):
        raise ValueError(f"M1 seed group {path!r} contains duplicates")
    return result


def stage_configuration(
    stage: str,
    seed: int,
) -> tuple[PhysicalCavityConfig, dict[str, Any], dict[str, Any], dict[str, str]]:
    protocol, seed_bank, lock_hashes = _load_m1_locks()
    stages = protocol.get("stages", {})
    if stage not in stages:
        raise ValueError(f"unknown M1 stage {stage!r}")
    specification = stages[stage]
    allowed_seeds = _seed_group(seed_bank, specification["seed_group"])
    if seed not in allowed_seeds:
        raise ValueError(
            f"seed {seed} is not preregistered for {stage}; allowed={allowed_seeds}"
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


def _new_diagnostics() -> dict[str, int | float]:
    return {
        "candidate_collisions": 0,
        "accepted_collisions": 0,
        "majorant_violations": 0,
        "max_acceptance_ratio": 0.0,
    }


def _new_runtime(
    cfg: PhysicalCavityConfig,
    block_count: int,
) -> M1Runtime:
    return M1Runtime(
        state=initialize_physical_state(cfg),
        rng=np.random.default_rng(cfg.seed + 29),
        step_index=0,
        moments=PhysicalMomentAccumulator(cfg),
        lid_events=LidWallEventAccumulator(cfg),
        wall_balance=WallBalanceAccumulator(cfg),
        block_moments=[PhysicalMomentAccumulator(cfg) for _ in range(block_count)],
        block_lid_events=[LidWallEventAccumulator(cfg) for _ in range(block_count)],
        temporal_sums={},
        temporal_sums2={},
        temporal_nsamples=0,
        diagnostics=_new_diagnostics(),
    )


def _moment_payload(accumulator: PhysicalMomentAccumulator) -> dict[str, Any]:
    return {
        "samples": accumulator.samples,
        "simulated_count": accumulator.simulated_count,
        "m0": accumulator.m0,
        "m1": accumulator.m1,
        "m2": accumulator.m2,
        "energy": accumulator.energy,
        "energy_velocity": accumulator.energy_velocity,
    }


def _restore_moment(
    cfg: PhysicalCavityConfig,
    payload: dict[str, Any],
) -> PhysicalMomentAccumulator:
    accumulator = PhysicalMomentAccumulator(cfg)
    accumulator.samples = int(payload["samples"])
    for name in (
        "simulated_count",
        "m0",
        "m1",
        "m2",
        "energy",
        "energy_velocity",
    ):
        setattr(accumulator, name, np.asarray(payload[name]).copy())
    return accumulator


def _lid_payload(accumulator: LidWallEventAccumulator) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(getattr(accumulator, name)).copy()
        for name in (
            "event_count",
            "inverse_flux_weight",
            "weighted_slip",
            "weighted_relative_speed2",
        )
    }


def _restore_lid(
    cfg: PhysicalCavityConfig,
    payload: dict[str, Any],
) -> LidWallEventAccumulator:
    accumulator = LidWallEventAccumulator(cfg)
    for name in (
        "event_count",
        "inverse_flux_weight",
        "weighted_slip",
        "weighted_relative_speed2",
    ):
        setattr(accumulator, name, np.asarray(payload[name]).copy())
    return accumulator


def _balance_payload(accumulator: WallBalanceAccumulator) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(getattr(accumulator, name)).copy()
        for name in (
            "incoming_count",
            "outgoing_count",
            "incoming_represented_weight",
            "outgoing_represented_weight",
        )
    }


def _restore_balance(
    cfg: PhysicalCavityConfig,
    payload: dict[str, Any],
) -> WallBalanceAccumulator:
    accumulator = WallBalanceAccumulator(cfg)
    for name in (
        "incoming_count",
        "outgoing_count",
        "incoming_represented_weight",
        "outgoing_represented_weight",
    ):
        setattr(accumulator, name, np.asarray(payload[name]).copy())
    return accumulator


def _runtime_blocks_payload(runtime: M1Runtime) -> dict[str, Any]:
    return {
        "wall_balance": _balance_payload(runtime.wall_balance),
        "block_moments": {
            f"{index:03d}": _moment_payload(accumulator)
            for index, accumulator in enumerate(runtime.block_moments)
        },
        "block_lid_events": {
            f"{index:03d}": _lid_payload(accumulator)
            for index, accumulator in enumerate(runtime.block_lid_events)
        },
    }


def _checkpoint_metadata(
    *,
    stage: str,
    seed: int,
    backend: str,
    sample_stride: int,
    block_count: int,
    lock_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "seed": seed,
        "backend": backend,
        "sample_stride": sample_stride,
        "block_count": block_count,
        "numpy_version": np.__version__,
        **lock_hashes,
    }


def _save_runtime_checkpoint(
    path: Path,
    cfg: PhysicalCavityConfig,
    runtime: M1Runtime,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = NTCCheckpoint(
        state=runtime.state,
        rng_state=runtime.rng.bit_generator.state,
        step_index=runtime.step_index,
        diagnostics=runtime.diagnostics,
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
    expected_metadata: dict[str, Any],
) -> M1Runtime:
    checkpoint = load_ntc_checkpoint(path, cfg)
    if dict(checkpoint.metadata) != expected_metadata:
        raise ValueError("checkpoint M1 metadata differs from the locked run")
    payload = checkpoint.block_accumulators
    if not isinstance(payload, dict):
        raise ValueError("M1 checkpoint has no block accumulator payload")
    moment_payloads = payload.get("block_moments")
    lid_payloads = payload.get("block_lid_events")
    balance_payload = payload.get("wall_balance")
    if not all(
        isinstance(value, dict)
        for value in (moment_payloads, lid_payloads, balance_payload)
    ):
        raise ValueError("M1 checkpoint block payload is malformed")
    assert isinstance(moment_payloads, dict)
    assert isinstance(lid_payloads, dict)
    assert isinstance(balance_payload, dict)
    if sorted(moment_payloads) != sorted(lid_payloads):
        raise ValueError("M1 checkpoint block moment/lid keys differ")
    rng = np.random.default_rng()
    checkpoint.restore_rng(rng)
    return M1Runtime(
        state=checkpoint.state,
        rng=rng,
        step_index=checkpoint.step_index,
        moments=checkpoint.moments,
        lid_events=checkpoint.wall_events,
        wall_balance=_restore_balance(cfg, balance_payload),
        block_moments=[
            _restore_moment(cfg, moment_payloads[key])
            for key in sorted(moment_payloads)
        ],
        block_lid_events=[
            _restore_lid(cfg, lid_payloads[key])
            for key in sorted(lid_payloads)
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
        diagnostics={key: value for key, value in checkpoint.diagnostics.items()},
    )


def _block_index(step: int, cfg: PhysicalCavityConfig, block_count: int) -> int:
    if step < cfg.sample_start:
        raise ValueError("sampling block requested before sample_start")
    span = cfg.steps - cfg.sample_start
    return min(
        block_count - 1,
        (step - cfg.sample_start) * block_count // span,
    )


def _collision_backend(name: str) -> Callable[..., Any]:
    if name == "reference":
        return collide_vhs_ntc
    if name == "numpy_fast":
        return collide_vhs_ntc_fast
    raise ValueError("backend must be 'reference' or 'numpy_fast'")


def _finish_fields(runtime: M1Runtime) -> dict[str, np.ndarray]:
    fields = runtime.moments.finalize()
    fields.update(runtime.lid_events.finalize())
    if runtime.temporal_nsamples <= 0:
        raise ValueError("M1 run completed without temporal samples")
    for key in ("T", "u", "v", "w"):
        fields[f"sigma_{key}"] = np.sqrt(
            np.maximum(
                runtime.temporal_sums2[key] / runtime.temporal_nsamples
                - (runtime.temporal_sums[key] / runtime.temporal_nsamples) ** 2,
                0.0,
            )
        )
    return fields


def _finish_block_fields(runtime: M1Runtime) -> dict[str, np.ndarray]:
    blocks: list[dict[str, np.ndarray]] = []
    for moments, lid in zip(runtime.block_moments, runtime.block_lid_events):
        fields = moments.finalize()
        fields.update(lid.finalize())
        blocks.append(fields)
    keys = sorted(set.intersection(*(set(block) for block in blocks)))
    return {key: np.stack([block[key] for block in blocks]) for key in keys}


def _tracked_block_quantities(
    block_fields: dict[str, np.ndarray],
    cfg: PhysicalCavityConfig,
    overall_fields: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    center = (cfg.nx - 1) / 2.0

    def center_value(values: np.ndarray) -> np.ndarray:
        left = int(np.floor(center))
        right = int(np.ceil(center))
        return 0.5 * (values[:, left] + values[:, right])

    macro_slip = 1.0 - center_value(block_fields["u"][:, -1, :]) / cfg.lid_velocity_x
    micro_slip = center_value(block_fields["microscopic_lid_slip_over_uwall"])
    temperature = block_fields["T"]
    q_profiles = np.stack(
        [_profile_at_y(field, 0.8) for field in block_fields["qy"]]
    )
    overall_q_profile = _profile_at_y(overall_fields["qy"], 0.8)
    positive_scale = float(np.max(overall_q_profile))
    normalized = np.divide(
        q_profiles,
        positive_scale,
        out=np.full_like(q_profiles, np.nan),
        where=positive_scale > 0.0,
    )
    return {
        "macroscopic_lid_slip_center": macro_slip,
        "microscopic_lid_slip_center": micro_slip,
        "temperature_min_K": np.min(temperature, axis=(1, 2)),
        "temperature_max_K": np.max(temperature, axis=(1, 2)),
        "qy_profile_min_normalized": np.min(normalized, axis=1),
        "qy_profile_max_normalized": np.max(normalized, axis=1),
    }


def _stationarity_report(
    block_fields: dict[str, np.ndarray],
    cfg: PhysicalCavityConfig,
    overall_fields: dict[str, np.ndarray],
    *,
    z_limit: float = 2.0,
    minimum_finite_per_half: int = 3,
) -> dict[str, Any]:
    tracked = _tracked_block_quantities(block_fields, cfg, overall_fields)
    summaries: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    half_count = len(next(iter(tracked.values()))) // 2
    for name, values in tracked.items():
        finite = np.isfinite(values)
        first_finite = int(np.count_nonzero(finite[:half_count]))
        second_finite = int(np.count_nonzero(finite[half_count:]))
        coverage = (
            first_finite >= minimum_finite_per_half
            and second_finite >= minimum_finite_per_half
        )
        if coverage:
            first = values[:half_count][finite[:half_count]]
            second = values[half_count:][finite[half_count:]]
            first_mean = float(np.mean(first))
            second_mean = float(np.mean(second))
            difference = second_mean - first_mean
            standard_error = float(
                np.sqrt(
                    np.var(first, ddof=1) / len(first)
                    + np.var(second, ddof=1) / len(second)
                )
            )
            if standard_error > 0.0:
                z_score = difference / standard_error
            elif difference == 0.0:
                z_score = 0.0
            else:
                z_score = float(np.copysign(np.inf, difference))
            midpoint = 0.5 * (first_mean + second_mean)
            relative_drift = (
                abs(difference) / abs(midpoint)
                if midpoint != 0.0
                else (0.0 if difference == 0.0 else float("inf"))
            )
            summary = {
                "block_count": int(len(values)),
                "first_half_finite": first_finite,
                "second_half_finite": second_finite,
                "first_half_mean": first_mean,
                "second_half_mean": second_mean,
                "drift": difference,
                "drift_standard_error": standard_error,
                "drift_z_score": z_score,
                "relative_drift": relative_drift,
                "max_abs_drift_z_score": abs(z_score),
            }
            passed = float(summary["max_abs_drift_z_score"]) <= z_limit
        else:
            summary = {
                "block_count": int(len(values)),
                "first_half_finite": first_finite,
                "second_half_finite": second_finite,
                "max_abs_drift_z_score": float("inf"),
            }
            passed = False
        summaries[name] = summary
        checks[name] = bool(passed)
    return {
        "z_limit": z_limit,
        "minimum_finite_blocks_per_half": minimum_finite_per_half,
        "tracked": summaries,
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                _strict_json_ready(value),
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _strict_json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _strict_json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _strict_json_ready(value.item())
    if isinstance(value, dict):
        return {str(key): _strict_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            return "nan"
        return "+inf" if value > 0.0 else "-inf"
    return value


def run_m1_seed(
    *,
    stage: str,
    seed: int,
    output_dir: Path,
    backend: str = "reference",
    resume: bool = True,
    stop_after_step: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    cfg, protocol, specification, lock_hashes = stage_configuration(stage, seed)
    locked_backend = specification.get("backend")
    if locked_backend is not None and backend != locked_backend:
        raise ValueError(
            f"{stage} locks backend={locked_backend!r}, not {backend!r}"
        )
    contract = protocol["runtime_contract"]
    sample_stride = int(contract["sample_stride"])
    block_count = int(contract["nonoverlapping_sampling_blocks"])
    checkpoint_interval = int(contract["checkpoint_interval_steps"])
    metadata = _checkpoint_metadata(
        stage=stage,
        seed=seed,
        backend=backend,
        sample_stride=sample_stride,
        block_count=block_count,
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
        raise FileExistsError(
            "--no-resume requires an empty output directory"
        )
    if resume and not checkpoint_path.exists() and existing_final:
        raise FileExistsError(
            "final artifacts exist without their checkpoint; use a new output directory"
        )
    if resume and checkpoint_path.exists():
        runtime = _load_runtime_checkpoint(checkpoint_path, cfg, metadata)
    else:
        runtime = _new_runtime(cfg, block_count)
    if runtime.step_index < cfg.steps and existing_final:
        raise ValueError(
            "incomplete checkpoint is accompanied by stale final artifacts"
        )
    if len(runtime.block_moments) != block_count:
        raise ValueError("checkpoint block count differs from M1 lock")
    collide = _collision_backend(backend)
    target_step = cfg.steps if stop_after_step is None else min(cfg.steps, stop_after_step)
    if target_step < runtime.step_index:
        raise ValueError("stop_after_step precedes the checkpoint step")

    while runtime.step_index < target_step:
        step = runtime.step_index
        block_index = (
            _block_index(step, cfg, block_count)
            if step >= cfg.sample_start
            else None
        )

        def wall_handler(
            wall: str,
            position: np.ndarray,
            velocity: np.ndarray,
            weight: np.ndarray,
            wall_velocity: np.ndarray,
        ) -> None:
            runtime.lid_events.add(wall, position, velocity, weight, wall_velocity)
            runtime.wall_balance.add(wall, position, velocity, weight, wall_velocity)
            assert block_index is not None
            runtime.block_lid_events[block_index].add(
                wall, position, velocity, weight, wall_velocity
            )

        runtime.state.pos += runtime.state.vel[:, :2] * cfg.dt
        apply_diffuse_walls(
            runtime.state,
            cfg,
            runtime.rng,
            wall_event_handler=(wall_handler if step >= cfg.sample_start else None),
        )
        collision = collide(runtime.state, cfg, runtime.rng)
        runtime.diagnostics["candidate_collisions"] = int(
            runtime.diagnostics["candidate_collisions"]
        ) + collision.candidate_collisions
        runtime.diagnostics["accepted_collisions"] = int(
            runtime.diagnostics["accepted_collisions"]
        ) + collision.accepted_collisions
        runtime.diagnostics["majorant_violations"] = int(
            runtime.diagnostics["majorant_violations"]
        ) + collision.majorant_violations
        runtime.diagnostics["max_acceptance_ratio"] = max(
            float(runtime.diagnostics["max_acceptance_ratio"]),
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

    fields = _finish_fields(runtime)
    block_fields = _finish_block_fields(runtime)
    stationarity = _stationarity_report(
        block_fields,
        cfg,
        fields,
        z_limit=2.0,
        minimum_finite_per_half=int(
            protocol["stationarity_contract"]["minimum_finite_blocks_per_half"]
        ),
    )
    wall = runtime.wall_balance.finalize()
    evaluation = evaluate_mohammadzadeh_fields(fields, cfg)
    core_finite = all(
        np.all(np.isfinite(fields[key]))
        for key in ("T", "rho", "u", "v", "qx", "qy")
    )
    minimum_lid_events = int(np.min(fields["microscopic_lid_event_count"]))
    mechanical_checks = {
        "majorant_violations_equal_zero": int(
            runtime.diagnostics["majorant_violations"]
        ) == 0,
        "finite_nonempty_fields": bool(core_finite and runtime.moments.samples > 0),
        "complete_lid_event_bin_coverage": minimum_lid_events
        >= int(contract["minimum_wall_event_coverage_per_lid_bin"]),
        "wall_mass_imbalance_below_locked_limit": float(
            wall["relative_net_mass_imbalance"]
        ) <= float(contract["wall_relative_net_mass_imbalance_max"]),
        "stationarity_pass": bool(stationarity["all_passed"]),
        "heat_flux_sign_pass": bool(evaluation["checks"]["fig9_qy_sign"]),
        "heat_flux_zero_crossing_pass": bool(
            evaluation["checks"]["fig9_qy_zero_crossing"]
        ),
    }
    if stage == "S1_mechanics_32":
        decision = (
            "seed_passes_S1_mechanics_awaiting_stage_aggregation"
            if all(mechanical_checks.values())
            else "seed_fails_S1_mechanics_awaiting_stage_aggregation"
        )
    else:
        decision = "complete_seed_awaiting_stage_aggregation"
    completion_status = (
        "complete_confirmatory_seed_awaiting_eight_seed_aggregation"
        if stage == "S5_confirmatory_200"
        else "complete_development_seed_awaiting_stage_aggregation"
    )
    accepted = int(runtime.diagnostics["accepted_collisions"])
    candidates = int(runtime.diagnostics["candidate_collisions"])
    diagnostics = {
        **runtime.diagnostics,
        "acceptance_fraction": accepted / max(candidates, 1),
        "collisions_per_particle_step": accepted
        / (len(runtime.state.pos) * cfg.steps),
        "dt": cfg.dt,
        "number_density": cfg.number_density,
        "mean_free_path": cfg.vhs.mean_free_path(cfg.number_density, cfg.t0),
        "volume_sample_count": runtime.temporal_nsamples,
        "sample_stride": sample_stride,
        "backend": backend,
    }
    summary = {
        "stage": stage,
        "purpose": specification["purpose"],
        "status": completion_status,
        "seed": seed,
        "config": asdict(cfg),
        "lock_hashes": lock_hashes,
        "checkpoint_manifest_sha256": manifest["manifest_sha256"],
        "diagnostics": diagnostics,
        "wall_balance": wall,
        "minimum_lid_events_per_bin": minimum_lid_events,
        "stationarity": stationarity,
        "mechanical_checks": mechanical_checks,
        "evaluation": {
            key: value
            for key, value in evaluation.items()
            if key != "comparison_arrays"
        },
        "decision": decision,
    }
    _atomic_save_npz(output_dir / "fields.npz", fields)
    _atomic_save_npz(output_dir / "block_fields.npz", block_fields)
    _atomic_write_json(output_dir / "summary.json", summary)
    artifact_manifest = {
        "stage": stage,
        "seed": seed,
        "status": completion_status,
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
    parser.add_argument("--stage", default="S1_mechanics_32")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("reference", "numpy_fast"),
        default="reference",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    summary = run_m1_seed(
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
