"""Locked cross-condition DSMC references for the Mohammadzadeh MV3 study.

Only twelve preregistered trajectories are new.  The Kn=0.05, U=100 m/s
condition is reused from M3 and all heat-flux quantities are excluded from
the scientific claims of this stage.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import mohammadzadeh_spatial_refinement as engine
from .mohammadzadeh_production import _atomic_write_json, _strict_json_ready
from .mohammadzadeh_validation import mohammadzadeh_config, reference_directory


STAGE = "MV3_cross_condition_reference"
PROTOCOL_FILE = "mv3_cross_condition_protocol.json"
SEED_FILE = "mv3_cross_condition_seed_bank.json"
LOCK_FILE = "mv3_cross_condition_lock.json"
LOCK_STATUS = "locked_before_any_MV3_reference_trajectory_or_model_outcome"
SOURCE_FILES = (
    "vgdsmc/mohammadzadeh_mv3_reference.py",
    "vgdsmc/mohammadzadeh_vision_mv3.py",
    "vgdsmc/mohammadzadeh_spatial_refinement.py",
    "vgdsmc/event_wall_streaming.py",
    "vgdsmc/ntc_fast.py",
    "vgdsmc/mohammadzadeh_production_m1r.py",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def protocol_path(reference_dir: Path | None = None) -> Path:
    directory = reference_directory() if reference_dir is None else Path(reference_dir)
    return directory / PROTOCOL_FILE


def load_protocol(reference_dir: Path | None = None) -> dict[str, Any]:
    value = _load_json(protocol_path(reference_dir))
    if (
        value.get("stage") != "MV3_Mohammadzadeh_condition_heldout_benchmark"
        or value.get("status") != LOCK_STATUS
    ):
        raise ValueError("MV3 cross-condition protocol is not locked")
    conditions = value.get("conditions", [])
    if len(conditions) != 4 or len({item["id"] for item in conditions}) != 4:
        raise ValueError("MV3 must contain four unique conditions")
    new_count = sum(
        len(item["evaluation_seeds"])
        for item in conditions
        if item["source"] == "new_MV3_reference"
    )
    if new_count != int(value["reference_contract"]["new_trajectory_count"]):
        raise ValueError("MV3 new-reference count differs from the protocol")
    return value


def new_reference_tasks(reference_dir: Path | None = None) -> list[tuple[str, int]]:
    protocol = load_protocol(reference_dir)
    return [
        (str(condition["id"]), int(seed))
        for condition in protocol["conditions"]
        if condition["source"] == "new_MV3_reference"
        for seed in condition["evaluation_seeds"]
    ]


def task_from_index(index: int, reference_dir: Path | None = None) -> tuple[str, int]:
    tasks = new_reference_tasks(reference_dir)
    if not 0 <= index < len(tasks):
        raise ValueError("MV3 reference task index is outside the locked array")
    return tasks[index]


def expected_lock_hashes(reference_dir: Path | None = None) -> dict[str, str]:
    directory = reference_directory() if reference_dir is None else Path(reference_dir)
    root = Path(__file__).resolve().parents[1]
    hashes = {
        "protocol_sha256": engine._sha256(directory / PROTOCOL_FILE),
        "seed_bank_sha256": engine._sha256(directory / SEED_FILE),
        "m1r_protocol_sha256": engine._sha256(directory / "m1r_repair_protocol.json"),
    }
    hashes.update(
        {f"source::{name}": engine._sha256(root / name) for name in SOURCE_FILES}
    )
    return hashes


def _condition(protocol: Mapping[str, Any], condition_id: str) -> dict[str, Any]:
    matches = [item for item in protocol["conditions"] if item["id"] == condition_id]
    if len(matches) != 1:
        raise ValueError(f"unknown MV3 condition {condition_id!r}")
    return dict(matches[0])


def stage_configuration(
    stage: str,
    seed: int,
    *,
    reference_dir: Path | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    if not stage.startswith(f"{STAGE}::"):
        raise ValueError(f"MV3 reference runner accepts only {STAGE}::<condition>")
    condition_id = stage.split("::", 1)[1]
    directory = reference_directory() if reference_dir is None else Path(reference_dir)
    protocol = load_protocol(directory)
    condition = _condition(protocol, condition_id)
    if condition["source"] != "new_MV3_reference":
        raise ValueError("the existing M3 condition must not be rerun")
    if int(seed) not in [int(item) for item in condition["evaluation_seeds"]]:
        raise ValueError("MV3 seed is not preregistered for this condition")

    seed_bank = _load_json(directory / SEED_FILE)
    if [int(item) for item in seed_bank["new_reference_conditions"][condition_id]] != [
        int(item) for item in condition["evaluation_seeds"]
    ]:
        raise ValueError("MV3 seed bank differs from the condition protocol")
    all_seeds = [seed for _, seed in new_reference_tasks(directory)]
    existing = [
        int(item)
        for group in seed_bank["existing_kn0p05_u100"].values()
        for item in group
    ]
    if len(all_seeds) != len(set(all_seeds)) or set(all_seeds) & set(existing):
        raise ValueError("MV3 reference seeds overlap")

    lock_path = directory / LOCK_FILE
    if not lock_path.is_file():
        raise FileNotFoundError(f"MV3 lock is missing: {lock_path}")
    lock = _load_json(lock_path)
    expected = expected_lock_hashes(directory)
    if lock.get("status") != LOCK_STATUS or lock.get("hashes") != expected:
        raise ValueError("MV3 lock does not match its source bundle")

    contract = protocol["reference_contract"]
    base = mohammadzadeh_config(
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
    engine_protocol = dict(protocol)
    engine_protocol["runtime_contract"] = {
        "nonoverlapping_sampling_blocks": int(contract["nonoverlapping_sampling_blocks"]),
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
    m1r_protocol = _load_json(directory / "m1r_repair_protocol.json")
    hashes = {**expected, "mv3_lock_sha256": engine._sha256(lock_path)}
    return cfg, engine_protocol, specification, m1r_protocol, hashes


def verify_lock(reference_dir: Path | None = None) -> dict[str, Any]:
    tasks = new_reference_tasks(reference_dir)
    configured = [
        stage_configuration(f"{STAGE}::{condition_id}", seed, reference_dir=reference_dir)
        for condition_id, seed in tasks
    ]
    return {
        "status": "MV3_cross_condition_lock_verified_without_running_trajectories",
        "task_count": len(tasks),
        "tasks": [{"condition_id": condition, "seed": seed} for condition, seed in tasks],
        "grid": configured[0][0].nx,
        "steps": configured[0][0].steps,
        "lock_hashes": configured[0][4],
    }


def _read_profile(path: Path, knudsen: float, value_name: str) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row["sampling"] == "macroscopic"
            and np.isclose(float(row["kn"]), knudsen, rtol=0.0, atol=1.0e-12)
        ]
    return (
        np.asarray([float(row["x_over_l"]) for row in rows]),
        np.asarray([float(row[value_name]) for row in rows]),
    )


def _external_profile_evaluation(fields: Mapping[str, np.ndarray], cfg: Any) -> dict[str, Any]:
    """Report available PRE profiles without making a hard MV3 acceptance claim."""
    directory = reference_directory()
    if not np.isclose(cfg.lid_velocity_x, 100.0):
        return {
            "status": "not_available_for_this_lid_speed",
            "external_validation_claim": False,
            "comparison_arrays": {},
        }
    x = (np.arange(cfg.nx, dtype=float) + 0.5) / cfg.nx
    result: dict[str, Any] = {
        "status": "article_profiles_reported_not_used_as_MV3_gate",
        "external_validation_claim": False,
        "comparison_arrays": {},
        "metrics": {},
    }
    for name, filename, column, simulated in (
        (
            "macroscopic_lid_slip",
            "fig4_wall_slip_profiles.csv",
            "u_slip_over_uwall",
            1.0 - np.asarray(fields["u"])[-1] / cfg.lid_velocity_x,
        ),
        (
            "macroscopic_lid_temperature",
            "fig5_wall_temperature_profiles.csv",
            "temperature_K",
            np.asarray(fields["T"])[-1],
        ),
    ):
        reference_x, reference_value = _read_profile(
            directory / filename, cfg.knudsen, column
        )
        if len(reference_x) == 0:
            continue
        interpolated = np.interp(reference_x, x, simulated)
        denominator = max(
            float(np.linalg.norm(reference_value - reference_value.mean())), 1.0e-12
        )
        result["metrics"][f"{name}_nrmse"] = float(
            np.linalg.norm(interpolated - reference_value) / denominator
        )
    if not result["metrics"]:
        result["status"] = "not_available_for_this_knudsen"
    return result


def run_reference_seed(
    *,
    condition_id: str,
    seed: int,
    output_dir: Path,
    resume: bool = True,
    stop_after_step: int | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    original_stage = engine.stage_configuration
    original_evaluate = engine.evaluate_mohammadzadeh_fields
    original_crossings = engine._heat_flux_zero_crossing_counts
    engine.stage_configuration = stage_configuration
    engine.evaluate_mohammadzadeh_fields = _external_profile_evaluation
    engine._heat_flux_zero_crossing_counts = lambda fields, cfg, evaluation: {
        "status": "not_evaluated_heat_flux_excluded_from_MV3"
    }
    try:
        summary = engine.run_refinement_seed(
            stage=f"{STAGE}::{condition_id}",
            seed=seed,
            output_dir=Path(output_dir),
            resume=resume,
            stop_after_step=stop_after_step,
            progress=progress,
        )
    finally:
        engine.stage_configuration = original_stage
        engine.evaluate_mohammadzadeh_fields = original_evaluate
        engine._heat_flux_zero_crossing_counts = original_crossings
    if summary.get("status") != "complete_M2_spatial_development_seed":
        return summary
    summary["stage"] = STAGE
    summary["condition_id"] = condition_id
    summary["status"] = "complete_MV3_reference_seed"
    summary["scientific_scope"] = "T_and_u_reference_only_heat_flux_excluded"
    summary["decision"] = (
        "accept_MV3_reference_seed"
        if all(summary.get("mechanical_checks", {}).values())
        else "hold_MV3_reference_seed"
    )
    _atomic_write_json(Path(output_dir) / "summary.json", summary)
    manifest_path = Path(output_dir) / "artifact_manifest.json"
    manifest = _load_json(manifest_path)
    manifest["stage"] = STAGE
    manifest["condition_id"] = condition_id
    manifest["status"] = summary["status"]
    manifest["files"]["summary.json"] = {
        "sha256": engine._sha256(Path(output_dir) / "summary.json"),
        "size_bytes": (Path(output_dir) / "summary.json").stat().st_size,
    }
    _atomic_write_json(manifest_path, manifest)
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
