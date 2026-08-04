"""Locked spatial-refinement runner for the repaired Mohammadzadeh cavity."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .event_wall_streaming import stream_with_diffuse_walls
from .mohammadzadeh_production import (
    _atomic_save_npz,
    _atomic_write_json,
    _finish_block_fields,
    _finish_fields,
    _stationarity_report,
    _strict_json_ready,
)
from .mohammadzadeh_production_m1r import (
    _heat_flux_zero_crossing_counts,
    _load_runtime_checkpoint,
    _new_runtime,
    _runtime_checkpoint_roundtrip_identity,
    _save_runtime_checkpoint,
    _verify_reference_bundle_for_s1r,
    event_mechanics_report,
)
from .mohammadzadeh_validation import (
    evaluate_mohammadzadeh_fields,
    mohammadzadeh_config,
    reference_directory,
)
from .ntc_fast import collide_vhs_ntc_fast
from .vhs_model import PhysicalCavityConfig


PROTOCOL_FILE = "m2_spatial_refinement_protocol.json"
SEED_FILE = "m2_refinement_seed_bank.json"
LOCK_FILE = "m2_spatial_refinement_lock.json"
LOCK_STATUS = "locked_before_any_M2_spatial_trajectory"
SOURCE_FILES = (
    "vgdsmc/mohammadzadeh_spatial_refinement.py",
    "tests/test_mohammadzadeh_spatial_refinement.py",
    "vgdsmc/event_wall_streaming.py",
    "vgdsmc/ntc_fast.py",
    "vgdsmc/mohammadzadeh_production_m1r.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_steps(cfg: PhysicalCavityConfig, sample_count: int) -> np.ndarray:
    if sample_count < 2 or sample_count > cfg.steps - cfg.sample_start:
        raise ValueError("invalid M2 spatial sample count")
    steps = np.rint(
        np.linspace(cfg.sample_start, cfg.steps - 1, sample_count)
    ).astype(np.int64)
    if len(np.unique(steps)) != sample_count:
        raise ValueError("M2 spatial sampling schedule is not unique")
    return steps


def _block_index(step: int, cfg: PhysicalCavityConfig, block_count: int) -> int:
    if not cfg.sample_start <= step < cfg.steps:
        raise ValueError("M2 block index requested outside production window")
    relative = step - cfg.sample_start
    duration = cfg.steps - cfg.sample_start
    return min(block_count - 1, relative * block_count // duration)


def expected_lock_hashes(reference_dir: Path | None = None) -> dict[str, str]:
    ref_dir = reference_directory() if reference_dir is None else Path(reference_dir)
    root = Path(__file__).resolve().parents[1]
    hashes = {
        "m2_protocol_sha256": _sha256(ref_dir / PROTOCOL_FILE),
        "m2_seed_bank_sha256": _sha256(ref_dir / SEED_FILE),
        "m1r_protocol_sha256": _sha256(ref_dir / "m1r_repair_protocol.json"),
        "m1r_seed_bank_sha256": _sha256(ref_dir / "m1r_seed_bank.json"),
        "m1r_lock_sha256": _sha256(ref_dir / "m1r_lock_manifest.json"),
        "p1_lock_sha256": _sha256(ref_dir / "p1_event_backend_parity_lock.json"),
        "p1_summary_sha256": _sha256(
            root
            / "results"
            / "mohammadzadeh_2012"
            / "p1_event_backend_parity"
            / "summary.json"
        ),
    }
    hashes.update(
        {f"source::{name}": _sha256(root / name) for name in SOURCE_FILES}
    )
    return hashes


def _seed_group(seed_bank: Mapping[str, Any], path: str) -> list[int]:
    value: Any = seed_bank
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"unknown M2 seed group {path!r}")
        value = value[component]
    if not isinstance(value, list) or not value:
        raise ValueError(f"empty M2 seed group {path!r}")
    seeds = [int(seed) for seed in value]
    if len(seeds) != len(set(seeds)):
        raise ValueError("M2 seed group contains duplicates")
    return seeds


def _load_locked_inputs(
    reference_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    ref_dir = reference_directory() if reference_dir is None else Path(reference_dir)
    protocol = json.loads((ref_dir / PROTOCOL_FILE).read_text(encoding="utf-8"))
    seed_bank = json.loads((ref_dir / SEED_FILE).read_text(encoding="utf-8"))
    m1r_seed_bank = json.loads(
        (ref_dir / "m1r_seed_bank.json").read_text(encoding="utf-8")
    )
    m1r_protocol = json.loads(
        (ref_dir / "m1r_repair_protocol.json").read_text(encoding="utf-8")
    )
    if protocol.get("status") != LOCK_STATUS:
        raise ValueError("M2 protocol is not locked before trajectories")
    if protocol.get("anti_circularity", {}).get("seed_bank") != SEED_FILE:
        raise ValueError("M2 protocol points to an unexpected seed bank")
    groups = seed_bank.get("development")
    if not isinstance(groups, Mapping) or set(groups) != {
        "R50_spatial", "R100_spatial", "R200_spatial"
    }:
        raise ValueError("M2 seed bank groups differ from the protocol")
    all_seeds = [int(seed) for group in groups.values() for seed in group]
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("M2 seed groups overlap")
    prior_seeds = {
        int(seed)
        for section in ("development", "prior_or_excluded_seeds")
        for group in m1r_seed_bank.get(section, {}).values()
        if isinstance(group, list)
        for seed in group
    }
    if set(all_seeds) & prior_seeds:
        raise ValueError("M2 seeds overlap M1/M1R/reserved seeds")
    stages = protocol.get("stages", {})
    authorized = [
        name for name, spec in stages.items()
        if isinstance(spec, Mapping) and spec.get("authorized")
    ]
    if authorized != ["R50_spatial"]:
        raise ValueError("only R50 may be authorized by the initial M2 lock")
    baseline = protocol.get("baseline", {})
    baseline_summary = (
        Path(__file__).resolve().parents[1]
        / str(baseline.get("directory", ""))
        / "summary.json"
    )
    if _sha256(baseline_summary) != baseline.get("summary_sha256"):
        raise ValueError("locked 32x32 baseline summary hash mismatch")
    p1_summary_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "mohammadzadeh_2012"
        / "p1_event_backend_parity"
        / "summary.json"
    )
    p1_summary = json.loads(p1_summary_path.read_text(encoding="utf-8"))
    if (
        p1_summary.get("decision")
        != "numpy_fast_eligible_for_later_development_pilots"
        or not p1_summary.get("all_passed")
        or p1_summary.get("wall_streamer") != "chronological_event_driven"
    ):
        raise ValueError("event-streaming P1 does not authorize numpy_fast")
    lock_path = ref_dir / LOCK_FILE
    if not lock_path.is_file():
        raise FileNotFoundError(f"M2 pretrajectory lock is missing: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected = expected_lock_hashes(ref_dir)
    if lock.get("status") != LOCK_STATUS or lock.get("hashes") != expected:
        raise ValueError("M2 pretrajectory lock does not match inputs/sources")
    hashes = {**expected, "m2_lock_manifest_sha256": _sha256(lock_path)}
    return protocol, seed_bank, m1r_protocol, hashes


def stage_configuration(
    stage: str,
    seed: int,
    *,
    reference_dir: Path | None = None,
) -> tuple[PhysicalCavityConfig, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    protocol, seed_bank, m1r_protocol, hashes = _load_locked_inputs(reference_dir)
    specification = protocol.get("stages", {}).get(stage)
    if not isinstance(specification, Mapping):
        raise ValueError(f"unknown M2 spatial stage {stage!r}")
    if not specification.get("authorized"):
        raise ValueError(f"{stage} is not yet authorized by the locked progression")
    allowed = _seed_group(seed_bank, str(specification["seed_group"]))
    if seed not in allowed:
        raise ValueError(f"seed {seed} is not preregistered for {stage}")
    cfg = mohammadzadeh_config(
        grid=int(specification["grid"]),
        particles_per_cell=int(protocol["physical_case"]["particles_per_cell"]),
        steps=int(specification["steps"]),
        sample_start=int(specification["sample_start"]),
        seed=seed,
        dt_safety=float(protocol["physical_case"]["dt_safety"]),
    )
    return cfg, protocol, dict(specification), m1r_protocol, hashes


def run_refinement_seed(
    *,
    stage: str,
    seed: int,
    output_dir: Path,
    resume: bool = True,
    stop_after_step: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    cfg, protocol, specification, m1r_protocol, lock_hashes = stage_configuration(
        stage, seed
    )
    block_count = int(protocol["runtime_contract"]["nonoverlapping_sampling_blocks"])
    sample_count = int(protocol["runtime_contract"]["sample_count"])
    max_events = int(
        protocol["runtime_contract"]["maximum_events_per_particle_per_step"]
    )
    checkpoint_interval = int(specification["checkpoint_interval_steps"])
    sample_steps = _sample_steps(cfg, sample_count)
    sample_step_set = set(int(step) for step in sample_steps)
    metadata = {
        "runner": "M2_spatial_refinement_event_streaming",
        "stage": stage,
        "seed": seed,
        "backend": "numpy_fast",
        "block_count": block_count,
        "sample_count": sample_count,
        "sample_steps_sha256": hashlib.sha256(sample_steps.tobytes()).hexdigest(),
        "max_events_per_particle": max_events,
        "numpy_version": np.__version__,
        **lock_hashes,
    }
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "checkpoint.npz"
    final_names = ("fields.npz", "block_fields.npz", "summary.json", "artifact_manifest.json")
    existing_final = [name for name in final_names if (output_dir / name).exists()]
    if not resume and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("--no-resume requires an empty output directory")
    if resume and not checkpoint_path.exists() and existing_final:
        raise FileExistsError("M2 final artifacts exist without a checkpoint")
    runtime = (
        _load_runtime_checkpoint(checkpoint_path, cfg, metadata)
        if resume and checkpoint_path.exists()
        else _new_runtime(cfg, block_count)
    )
    if runtime.step_index < cfg.steps and existing_final:
        raise ValueError("incomplete M2 checkpoint has stale final artifacts")
    target = cfg.steps if stop_after_step is None else min(cfg.steps, stop_after_step)
    if target < runtime.step_index:
        raise ValueError("stop_after_step precedes the M2 checkpoint")

    while runtime.step_index < target:
        step = runtime.step_index
        block = _block_index(step, cfg, block_count) if step >= cfg.sample_start else None

        def wall_handler(
            wall: str,
            tangential_position: np.ndarray,
            velocity: np.ndarray,
            weight: np.ndarray,
            wall_velocity: np.ndarray,
        ) -> None:
            assert block is not None
            runtime.lid_events.add(wall, tangential_position, velocity, weight, wall_velocity)
            runtime.block_lid_events[block].add(
                wall, tangential_position, velocity, weight, wall_velocity
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
        collision = collide_vhs_ntc_fast(runtime.state, cfg, runtime.rng)
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

        if step in sample_step_set:
            assert block is not None
            instantaneous = runtime.moments.add(runtime.state, return_instantaneous=True)
            assert instantaneous is not None
            runtime.block_moments[block].add(runtime.state)
            for key in ("T", "u", "v", "w"):
                value = instantaneous[key]
                runtime.temporal_sums[key] = runtime.temporal_sums.get(
                    key, np.zeros_like(value)
                ) + value
                runtime.temporal_sums2[key] = runtime.temporal_sums2.get(
                    key, np.zeros_like(value)
                ) + value**2
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
        }

    roundtrip = _load_runtime_checkpoint(checkpoint_path, cfg, metadata)
    roundtrip_identity = _runtime_checkpoint_roundtrip_identity(runtime, roundtrip)
    fields = _finish_fields(runtime)
    block_fields = _finish_block_fields(runtime)
    event_report = event_mechanics_report(
        runtime.event_diagnostics,
        runtime.state,
        cfg,
        m1r_protocol["event_mechanics_gates"],
    )
    stationarity = _stationarity_report(
        block_fields,
        cfg,
        fields,
        z_limit=2.0,
        minimum_finite_per_half=int(
            m1r_protocol["stationarity_contract"]["minimum_finite_blocks_per_half"]
        ),
    )
    _verify_reference_bundle_for_s1r(m1r_protocol)
    evaluation = evaluate_mohammadzadeh_fields(fields, cfg)
    crossings = _heat_flux_zero_crossing_counts(fields, cfg, evaluation)
    accepted = int(runtime.collision_diagnostics["accepted_collisions"])
    candidates = int(runtime.collision_diagnostics["candidate_collisions"])
    minimum_lid_events = int(np.min(fields["microscopic_lid_event_count"]))
    checks = {
        "all_event_mechanics_gates_pass": bool(event_report["all_passed"]),
        "majorant_violations_equal_zero": int(
            runtime.collision_diagnostics["majorant_violations"]
        ) == 0,
        "finite_nonempty_fields": bool(
            runtime.temporal_nsamples == sample_count
            and all(np.all(np.isfinite(fields[key])) for key in ("T", "rho", "u", "v", "qx", "qy"))
        ),
        "complete_lid_event_bin_coverage": minimum_lid_events >= 1,
        "stationarity_pass": bool(stationarity["all_passed"]),
        "checkpoint_roundtrip_bitwise_identity": roundtrip_identity,
    }
    summary = {
        "stage": stage,
        "status": "complete_M2_spatial_development_seed",
        "external_validation_claim": False,
        "seed": seed,
        "config": asdict(cfg),
        "backend": "numpy_fast",
        "wall_streamer": "chronological_event_driven",
        "sample_steps": sample_steps.tolist(),
        "sample_count": runtime.temporal_nsamples,
        "lock_hashes": lock_hashes,
        "checkpoint_manifest_sha256": manifest["manifest_sha256"],
        "diagnostics": {
            **runtime.collision_diagnostics,
            "acceptance_fraction": accepted / max(candidates, 1),
            "dt": cfg.dt,
            "total_physical_time": cfg.steps * cfg.dt,
            "burn_in_physical_time": cfg.sample_start * cfg.dt,
        },
        "event_mechanics": event_report,
        "stationarity": stationarity,
        "minimum_lid_events_per_bin": minimum_lid_events,
        "mechanical_checks": checks,
        "evaluation": {
            key: value for key, value in evaluation.items() if key != "comparison_arrays"
        },
        "heat_flux_zero_crossing_count": crossings,
        "decision": (
            "complete_R50_awaiting_locked_two_seed_aggregation"
            if all(checks.values())
            else "hold_R50_for_mechanical_or_stationarity_diagnosis"
        ),
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
            for name in ("checkpoint.npz", "fields.npz", "block_fields.npz", "summary.json")
        },
        "lock_hashes": lock_hashes,
    }
    _atomic_write_json(output_dir / "artifact_manifest.json", artifact_manifest)
    return _strict_json_ready(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    summary = run_refinement_seed(
        stage=args.stage,
        seed=args.seed,
        output_dir=args.output_dir,
        resume=not args.no_resume,
        stop_after_step=args.stop_after_step,
        progress=lambda step, total: print(
            json.dumps({"step": step, "total": total}), flush=True
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
