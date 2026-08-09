"""Locked confirmatory DSMC references for the Mohammadzadeh MV5 study.

The four condition combinations and sixteen seeds in this module are new
confirmatory data.  They are deliberately absent from MV3/MV4 development.
The numerical trajectory contract is inherited exactly from the locked MV3
reference protocol; heat flux remains outside the scientific scope.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import mohammadzadeh_mv3_reference as mv3ref
from . import mohammadzadeh_vision_mv3 as mv3
from . import mohammadzadeh_spatial_refinement as engine
from .mohammadzadeh_production import _atomic_write_json, _strict_json_ready


STAGE = "MV5_confirmatory_reference"
PROTOCOL_FILE = "mv5_confirmatory_protocol.json"
LOCK_STATUS = "locked_before_MV5_confirmatory_reference_or_model_outcomes"
COMPLETE_STATUS = "complete_MV5_confirmatory_reference_seed"


def protocol_path() -> Path:
    return mv3.reference_directory() / PROTOCOL_FILE


def locked_protocol() -> dict[str, Any]:
    value = json.loads(protocol_path().read_text(encoding="utf-8"))
    if (
        value.get("stage")
        != "MV5_Mohammadzadeh_confirmatory_selector_benchmark"
        or value.get("status") != LOCK_STATUS
    ):
        raise ValueError("MV5 confirmatory protocol is absent or unlocked")
    conditions = value.get("confirmatory_conditions", [])
    if len(conditions) != 4 or len({item["id"] for item in conditions}) != 4:
        raise ValueError("MV5 requires four unique confirmatory conditions")
    tasks = [
        (item["id"], int(seed))
        for item in conditions
        for seed in item["evaluation_seeds"]
    ]
    if (
        len(tasks)
        != int(value["reference_contract"]["new_trajectory_count"])
        or len({seed for _, seed in tasks}) != len(tasks)
    ):
        raise ValueError("MV5 reference task/seed contract differs from protocol")
    source = value["source_contract"]
    if engine._sha256(mv3.protocol_path()) != source["mv3_protocol_sha256"]:
        raise ValueError("MV3 protocol hash differs from the MV5 preregistration")
    return value


def condition_map(protocol: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    value = locked_protocol() if protocol is None else protocol
    return {
        str(item["id"]): dict(item) for item in value["confirmatory_conditions"]
    }


def reference_tasks() -> list[tuple[str, int]]:
    return [
        (condition_id, int(seed))
        for condition_id, item in condition_map().items()
        for seed in item["evaluation_seeds"]
    ]


def task_from_index(index: int) -> tuple[str, int]:
    tasks = reference_tasks()
    if not 0 <= index < len(tasks):
        raise ValueError("MV5 reference task index is outside the locked array")
    return tasks[index]


def _condition(protocol: Mapping[str, Any], condition_id: str) -> dict[str, Any]:
    matches = [
        item
        for item in protocol["confirmatory_conditions"]
        if item["id"] == condition_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown MV5 condition {condition_id!r}")
    return dict(matches[0])


def stage_configuration(
    stage: str,
    seed: int,
    *,
    reference_dir: Path | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    del reference_dir
    if not stage.startswith(f"{STAGE}::"):
        raise ValueError(f"MV5 reference runner accepts only {STAGE}::<condition>")
    condition_id = stage.split("::", 1)[1]
    protocol = locked_protocol()
    condition = _condition(protocol, condition_id)
    if int(seed) not in [int(item) for item in condition["evaluation_seeds"]]:
        raise ValueError("MV5 seed is not preregistered for this condition")

    mv3_protocol = mv3.locked_protocol()
    contract = mv3_protocol["reference_contract"]
    base = mv3ref.mohammadzadeh_config(
        grid=int(contract["grid"]),
        particles_per_cell=int(contract["particles_per_cell"]),
        steps=int(contract["steps"]),
        sample_start=int(contract["sample_start"]),
        seed=int(seed),
        dt_safety=float(contract["dt_safety"]),
    )
    cfg = replace(
        base,
        knudsen=float(condition["knudsen"]),
        lid_velocity_x=float(condition["lid_speed_m_per_s"]),
    )
    engine_protocol = dict(mv3_protocol)
    engine_protocol["stage"] = STAGE
    engine_protocol["runtime_contract"] = {
        "nonoverlapping_sampling_blocks": int(
            contract["nonoverlapping_sampling_blocks"]
        ),
        "sample_count": int(contract["sample_count"]),
        "maximum_events_per_particle_per_step": int(
            contract["maximum_events_per_particle_per_step"]
        ),
    }
    specification = {
        "name": stage,
        "condition_id": condition_id,
        "grid": cfg.nx,
        "steps": cfg.steps,
        "sample_start": cfg.sample_start,
        "checkpoint_interval_steps": int(contract["checkpoint_interval_steps"]),
    }
    m1r_protocol = json.loads(
        (mv3.reference_directory() / "m1r_repair_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    hashes = {
        "mv5_protocol_sha256": engine._sha256(protocol_path()),
        "mv3_protocol_sha256": engine._sha256(mv3.protocol_path()),
    }
    return cfg, engine_protocol, specification, m1r_protocol, hashes


def verify_lock() -> dict[str, Any]:
    configured = [
        stage_configuration(f"{STAGE}::{condition_id}", seed)
        for condition_id, seed in reference_tasks()
    ]
    return {
        "status": "MV5_confirmatory_lock_verified_without_running_trajectories",
        "task_count": len(configured),
        "tasks": [
            {"condition_id": condition, "seed": seed}
            for condition, seed in reference_tasks()
        ],
        "grid": configured[0][0].nx,
        "steps": configured[0][0].steps,
        "protocol_sha256": engine._sha256(protocol_path()),
    }


def run_reference_seed(
    *,
    condition_id: str,
    seed: int,
    output_dir: Path,
    resume: bool = True,
    stop_after_step: int | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Reuse the verified MV3 trajectory engine under the MV5 lock."""
    original_configuration = mv3ref.stage_configuration
    original_stage = mv3ref.STAGE
    mv3ref.stage_configuration = stage_configuration
    mv3ref.STAGE = STAGE
    try:
        result = mv3ref.run_reference_seed(
            condition_id=condition_id,
            seed=seed,
            output_dir=output_dir,
            resume=resume,
            stop_after_step=stop_after_step,
            progress=progress,
        )
    finally:
        mv3ref.stage_configuration = original_configuration
        mv3ref.STAGE = original_stage
    if result.get("status") != "complete_MV3_reference_seed":
        return result

    path = Path(output_dir)
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    mechanical_values = [
        bool(value)
        for key, value in summary.get("mechanical_checks", {}).items()
        if key != "stationarity_pass"
    ]
    stationarity_values = [
        bool(value)
        for key, value in summary.get("stationarity", {}).get("checks", {}).items()
        if not mv3._is_heat_flux_stationarity_key(str(key))
    ]
    accepted = bool(mechanical_values) and all(mechanical_values) and bool(stationarity_values) and all(stationarity_values)
    summary.update(
        {
            "stage": STAGE,
            "condition_id": condition_id,
            "status": COMPLETE_STATUS,
            "scientific_scope": "confirmatory_T_and_u_only_heat_flux_excluded",
            "decision": "accept_MV5_confirmatory_reference_seed"
            if accepted
            else "hold_MV5_confirmatory_reference_seed",
            "mv5_protocol_sha256": engine._sha256(protocol_path()),
        }
    )
    _atomic_write_json(path / "summary.json", summary)
    manifest = json.loads((path / "artifact_manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "stage": STAGE,
            "condition_id": condition_id,
            "status": COMPLETE_STATUS,
        }
    )
    manifest["files"]["summary.json"] = {
        "sha256": engine._sha256(path / "summary.json"),
        "size_bytes": (path / "summary.json").stat().st_size,
    }
    _atomic_write_json(path / "artifact_manifest.json", manifest)
    return _strict_json_ready(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--condition-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--verify-lock-only", action="store_true")
    args = parser.parse_args()
    if args.verify_lock_only:
        print(json.dumps(verify_lock(), indent=2, sort_keys=True, allow_nan=False))
        return
    if args.task_index is not None:
        condition_id, seed = task_from_index(args.task_index)
    elif args.condition_id is not None and args.seed is not None:
        condition_id, seed = args.condition_id, args.seed
    else:
        parser.error("run mode requires --task-index or --condition-id and --seed")
    if args.output_root is None:
        parser.error("run mode requires --output-root")
    output = args.output_root / "references" / condition_id / f"seed_{seed}"
    result = run_reference_seed(
        condition_id=condition_id,
        seed=seed,
        output_dir=output,
        resume=not args.no_resume,
        stop_after_step=args.stop_after_step,
        progress=lambda step, total: print(
            json.dumps({"step": step, "total": total}), flush=True
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
