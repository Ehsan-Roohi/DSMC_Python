"""Locked late-window repairs for three MV5 T/u reference seeds.

The repair was locked after reference diagnostics and before any MV5 or MV6
model outcome.  It preserves physics, grid, particles, sample count, block
count, and the 93,750-step sampling horizon.  Only burn-in is extended.  The
original sixteen reference directories remain immutable.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from . import mohammadzadeh_mv5_reference as base
from . import mohammadzadeh_vision_mv3 as mv3
from . import mohammadzadeh_vision_mv5 as mv5
from . import mohammadzadeh_spatial_refinement as engine
from .mohammadzadeh_production import _atomic_write_json, _strict_json_ready


STAGE = "MV5_reference_stability_repair"
PROTOCOL_FILE = "mv5_reference_stability_repair_protocol.json"
LOCK_STATUS = (
    "locked_after_MV5_reference_diagnostics_before_any_MV5_or_MV6_model_outcome"
)
REPAIR_TASKS = (
    ("kn0p075_u150", 94003),
    ("kn0p1_u200", 94201),
    ("kn0p1_u400", 94301),
)
_BASE_STAGE_CONFIGURATION = base.stage_configuration


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def protocol_path() -> Path:
    return base.protocol_path().parent / PROTOCOL_FILE


def load_protocol() -> dict[str, Any]:
    value = json.loads(protocol_path().read_text(encoding="utf-8"))
    if value.get("stage") != STAGE or value.get("status") != LOCK_STATUS:
        raise ValueError("MV5 stability-repair protocol is absent or unlocked")
    diagnostics = value["diagnostic_basis"]
    tasks = tuple(
        (str(item["condition_id"]), int(item["seed"]))
        for item in diagnostics["failed_references"]
    )
    contract = value["repair_contract"]
    if (
        tasks != REPAIR_TASKS
        or int(contract["task_count"]) != len(REPAIR_TASKS)
        or int(contract["steps"]) != 153125
        or int(contract["sample_start"]) != 59375
        or int(contract["sample_count"]) != 3000
        or int(contract["nonoverlapping_sampling_blocks"]) != 10
        or int(contract["sampling_horizon_steps"]) != 93750
        or float(contract["stationarity_z_limit"]) != 2.0
    ):
        raise ValueError("MV5 stability-repair schedule differs from its lock")
    expected_guards = {
        "original_16_reference_directories_are_immutable": True,
        "only_the_three_preidentified_failed_seeds_are_repaired": True,
        "all_other_13_references_are_reused_byte_for_byte": True,
        "heat_flux_excluded_from_MV5_and_MV6_claims": True,
        "physics_changed": False,
        "grid_particles_sample_count_block_count_or_sampling_horizon_changed": False,
        "stationarity_gate_changed": False,
        "model_architecture_training_split_baselines_or_metrics_changed": False,
        "failed_model_attempts_created_no_model_outcomes": True,
    }
    if value.get("scope_guards") != expected_guards:
        raise ValueError("MV5 stability-repair scope guards differ from the lock")
    return value


def task_from_index(index: int) -> tuple[str, int]:
    if not 0 <= index < len(REPAIR_TASKS):
        raise ValueError("MV5 repair task index is outside the locked array")
    return REPAIR_TASKS[index]


def _diagnostic(condition_id: str, seed: int) -> dict[str, Any]:
    matches = [
        item
        for item in load_protocol()["diagnostic_basis"]["failed_references"]
        if item["condition_id"] == condition_id and int(item["seed"]) == int(seed)
    ]
    if len(matches) != 1:
        raise ValueError("repair task is absent from the diagnostic lock")
    return matches[0]


def stage_configuration(
    stage: str,
    seed: int,
    *,
    reference_dir: Path | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    del reference_dir
    if not stage.startswith(f"{base.STAGE}::"):
        raise ValueError("MV5 repair received a non-MV5 reference stage")
    condition_id = stage.split("::", 1)[1]
    if (condition_id, int(seed)) not in REPAIR_TASKS:
        raise ValueError("MV5 repair is locked to three diagnosed seeds")
    cfg, protocol, specification, m1r_protocol, hashes = _BASE_STAGE_CONFIGURATION(
        stage, seed
    )
    contract = load_protocol()["repair_contract"]
    cfg = replace(
        cfg,
        steps=int(contract["steps"]),
        sample_start=int(contract["sample_start"]),
    )
    specification = {
        **specification,
        "name": f"{STAGE}::{condition_id}",
        "steps": cfg.steps,
        "sample_start": cfg.sample_start,
        "checkpoint_interval_steps": int(contract["checkpoint_interval_steps"]),
    }
    protocol = {
        **protocol,
        "stage": STAGE,
        "runtime_contract": {
            **protocol["runtime_contract"],
            "sample_count": int(contract["sample_count"]),
            "nonoverlapping_sampling_blocks": int(
                contract["nonoverlapping_sampling_blocks"]
            ),
        },
    }
    hashes = {
        **hashes,
        "mv5_reference_stability_repair_protocol_sha256": _sha256(protocol_path()),
    }
    return cfg, protocol, specification, m1r_protocol, hashes


def _verify_manifest_file(directory: Path, name: str) -> Path:
    manifest = json.loads(
        (directory / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    path = directory / name
    record = manifest.get("files", {}).get(name, {})
    if (
        not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise ValueError(f"artifact verification failed: {path}")
    return path


def _in_scope_failed(summary: Mapping[str, Any]) -> set[str]:
    return {
        str(key)
        for key, passed in summary.get("stationarity", {}).get("checks", {}).items()
        if not bool(passed) and not mv3._is_heat_flux_stationarity_key(str(key))
    }


def validate_original_failure(
    output_root: Path, condition_id: str, seed: int
) -> dict[str, Any]:
    directory = output_root / "references" / condition_id / f"seed_{seed}"
    summary_path = _verify_manifest_file(directory, "summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    diagnostic = _diagnostic(condition_id, seed)
    expected_failed = set(diagnostic["failed_T_u_observables"])
    mechanics = [
        bool(value)
        for key, value in summary.get("mechanical_checks", {}).items()
        if key != "stationarity_pass"
    ]
    if (
        summary.get("status") != base.COMPLETE_STATUS
        or not mechanics
        or not all(mechanics)
        or _in_scope_failed(summary) != expected_failed
    ):
        raise ValueError(f"original failure differs from repair lock: {directory}")
    tracked = summary["stationarity"]["tracked"]
    for name, expected in diagnostic["failed_T_u_observables"].items():
        actual = tracked[name]
        if (
            abs(float(actual["drift_z_score"]) - float(expected["drift_z_score"]))
            > 1.0e-12
            or abs(float(actual["relative_drift"]) - float(expected["relative_drift"]))
            > 1.0e-12
        ):
            raise ValueError(f"diagnostic values changed for {directory}: {name}")
    return {
        "directory": str(directory),
        "summary_sha256": _sha256(summary_path),
        "failed_T_u_stationarity_checks": sorted(expected_failed),
    }


def run_repair(
    output_root: Path,
    *,
    condition_id: str,
    seed: int,
    resume: bool = True,
) -> dict[str, Any]:
    original = validate_original_failure(output_root, condition_id, seed)
    output = (
        output_root
        / "reference_stability_repair"
        / condition_id
        / f"seed_{seed}"
    )
    original_configuration = base.stage_configuration
    base.stage_configuration = stage_configuration
    try:
        result = base.run_reference_seed(
            condition_id=condition_id,
            seed=seed,
            output_dir=output,
            resume=resume,
            progress=lambda step, total: print(
                json.dumps({"step": step, "total": total}), flush=True
            ),
        )
    finally:
        base.stage_configuration = original_configuration
    if result.get("status") != base.COMPLETE_STATUS:
        return result
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "producer_stage": STAGE,
            "repair_protocol_sha256": _sha256(protocol_path()),
            "repair_contract": load_protocol()["repair_contract"],
            "superseded_original": original,
        }
    )
    mechanics = [
        bool(value)
        for key, value in summary.get("mechanical_checks", {}).items()
        if key != "stationarity_pass"
    ]
    accepted = bool(mechanics) and all(mechanics) and not _in_scope_failed(summary)
    summary["decision"] = (
        "accept_MV5_reference_stability_repair_seed"
        if accepted
        else "hold_MV5_reference_stability_repair_seed"
    )
    _atomic_write_json(summary_path, summary)
    manifest_path = output / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"producer_stage": STAGE})
    manifest["files"]["summary.json"] = {
        "sha256": _sha256(summary_path),
        "size_bytes": summary_path.stat().st_size,
    }
    _atomic_write_json(manifest_path, manifest)
    if not accepted:
        return _strict_json_ready(summary)
    mv5._verify_reference(output)
    return _strict_json_ready(summary)


def _selected_source(
    original_root: Path, condition_id: str, seed: int
) -> tuple[Path, str]:
    if (condition_id, seed) in REPAIR_TASKS:
        return (
            original_root
            / "reference_stability_repair"
            / condition_id
            / f"seed_{seed}",
            "late_window_repair",
        )
    return (
        original_root / "references" / condition_id / f"seed_{seed}",
        "immutable_original",
    )


def assemble_reference_tree(original_root: Path, assembled_root: Path) -> dict[str, Any]:
    load_protocol()
    entries = []
    for condition_id, seed in base.reference_tasks():
        source, source_kind = _selected_source(original_root, condition_id, seed)
        mv5._verify_reference(source)
        destination = assembled_root / "references" / condition_id / f"seed_{seed}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise ValueError(f"existing assembly link has wrong target: {destination}")
        elif destination.exists():
            raise ValueError(f"refusing to replace existing assembly path: {destination}")
        else:
            os.symlink(source.resolve(), destination, target_is_directory=True)
        entries.append(
            {
                "condition_id": condition_id,
                "seed": seed,
                "source_kind": source_kind,
                "source_directory": str(source.resolve()),
                "summary_sha256": _sha256(source / "summary.json"),
            }
        )
    report = {
        "stage": STAGE,
        "status": "complete_MV5_repaired_reference_tree",
        "repair_protocol_sha256": _sha256(protocol_path()),
        "entry_count": len(entries),
        "repair_entry_count": sum(
            item["source_kind"] == "late_window_repair" for item in entries
        ),
        "original_entry_count": sum(
            item["source_kind"] == "immutable_original" for item in entries
        ),
        "entries": entries,
        "checks": {
            "sixteen_entries": len(entries) == 16,
            "three_repairs": sum(
                item["source_kind"] == "late_window_repair" for item in entries
            )
            == 3,
            "thirteen_immutable_originals": sum(
                item["source_kind"] == "immutable_original" for item in entries
            )
            == 13,
        },
    }
    if not all(report["checks"].values()):
        raise ValueError("repaired reference-tree assembly failed")
    _atomic_write_json(assembled_root / "repair_assembly.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("repair", "assemble", "verify-lock"), required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--assembled-root", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "verify-lock":
        print(json.dumps(load_protocol(), indent=2, sort_keys=True))
        return
    if args.output_root is None:
        parser.error("repair and assemble modes require --output-root")
    if args.mode == "assemble":
        if args.assembled_root is None:
            parser.error("assemble mode requires --assembled-root")
        result = assemble_reference_tree(args.output_root, args.assembled_root)
    else:
        if args.task_index is None:
            parser.error("repair mode requires --task-index")
        condition_id, seed = task_from_index(args.task_index)
        result = run_repair(
            args.output_root,
            condition_id=condition_id,
            seed=seed,
            resume=not args.no_resume,
        )
        if result.get("decision") != "accept_MV5_reference_stability_repair_seed":
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            raise SystemExit(3)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
