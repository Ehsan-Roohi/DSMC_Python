"""Locked late-window stability repair for one MV3 T/u reference seed.

This stage is authorized after reference diagnostics and before any MV3 model
outcome.  It reruns only Kn=0.1, U_lid=100 m/s, seed 93202 with a later burn-in
and the same sampling horizon, sample count, block count, grid, and particles.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from . import mohammadzadeh_mv3_reference as base
from . import mohammadzadeh_spatial_refinement as engine
from .mohammadzadeh_production import _atomic_write_json, _strict_json_ready
from .mohammadzadeh_validation import reference_directory


STAGE = "MV3_reference_stability_repair"
CONDITION_ID = "kn0p1_u100"
SEED = 93202
PROTOCOL_FILE = "mv3_reference_stability_repair_protocol.json"
LOCK_FILE = "mv3_reference_stability_repair_lock.json"
LOCK_STATUS = "locked_after_reference_diagnostics_before_any_MV3_model_outcome"
SOURCE_FILES = (
    "vgdsmc/mohammadzadeh_mv3_reference_stability_repair.py",
    "vgdsmc/mohammadzadeh_mv3_reference.py",
    "vgdsmc/mohammadzadeh_spatial_refinement.py",
    "vgdsmc/event_wall_streaming.py",
    "vgdsmc/ntc_fast.py",
    "vgdsmc/mohammadzadeh_production_m1r.py",
)
_BASE_STAGE_CONFIGURATION = base.stage_configuration


def _sha256(path: Path) -> str:
    return engine._sha256(path)


def load_protocol() -> dict[str, Any]:
    value = json.loads(
        (reference_directory() / PROTOCOL_FILE).read_text(encoding="utf-8")
    )
    if value.get("status") != LOCK_STATUS or value.get("stage") != STAGE:
        raise ValueError("MV3 stability-repair protocol is not locked")
    contract = value.get("repair_contract", {})
    if (
        contract.get("condition_id") != CONDITION_ID
        or int(contract.get("seed", -1)) != SEED
        or int(contract.get("steps", -1)) != 153125
        or int(contract.get("sample_start", -1)) != 59375
        or int(contract.get("sample_count", -1)) != 3000
        or int(contract.get("nonoverlapping_sampling_blocks", -1)) != 10
    ):
        raise ValueError("MV3 stability-repair schedule differs from its lock")
    return value


def expected_lock_hashes() -> dict[str, str]:
    directory = reference_directory()
    root = Path(__file__).resolve().parents[1]
    hashes = {
        "protocol_sha256": _sha256(directory / PROTOCOL_FILE),
        "mv3_cross_condition_lock_sha256": _sha256(
            directory / base.LOCK_FILE
        ),
    }
    hashes.update(
        {f"source::{name}": _sha256(root / name) for name in SOURCE_FILES}
    )
    return hashes


def verify_lock() -> dict[str, Any]:
    protocol = load_protocol()
    lock_path = reference_directory() / LOCK_FILE
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected = expected_lock_hashes()
    if lock.get("status") != LOCK_STATUS or lock.get("hashes") != expected:
        raise ValueError("MV3 stability-repair lock is stale")
    return {
        "status": "MV3_reference_stability_repair_lock_verified",
        "condition_id": CONDITION_ID,
        "seed": SEED,
        "repair_contract": protocol["repair_contract"],
        "lock_hashes": expected,
    }


def stage_configuration(
    stage: str, seed: int
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    expected_stage = f"{base.STAGE}::{CONDITION_ID}"
    if stage != expected_stage or int(seed) != SEED:
        raise ValueError("stability repair is locked to kn0p1_u100 seed 93202")
    lock_report = verify_lock()
    cfg, protocol, specification, m1r_protocol, base_hashes = (
        _BASE_STAGE_CONFIGURATION(stage, seed)
    )
    contract = lock_report["repair_contract"]
    cfg = replace(
        cfg,
        steps=int(contract["steps"]),
        sample_start=int(contract["sample_start"]),
    )
    specification = {
        **specification,
        "name": STAGE,
        "steps": cfg.steps,
        "sample_start": cfg.sample_start,
        "checkpoint_interval_steps": int(contract["checkpoint_interval_steps"]),
    }
    protocol = {
        **protocol,
        "runtime_contract": {
            **protocol["runtime_contract"],
            "sample_count": int(contract["sample_count"]),
            "nonoverlapping_sampling_blocks": int(
                contract["nonoverlapping_sampling_blocks"]
            ),
        },
    }
    hashes = {
        **base_hashes,
        **{f"stability_repair::{key}": value for key, value in lock_report["lock_hashes"].items()},
        "stability_repair_lock_sha256": _sha256(reference_directory() / LOCK_FILE),
    }
    return cfg, protocol, specification, m1r_protocol, hashes


def _validate_original_failure(output_root: Path) -> dict[str, Any]:
    directory = output_root / "references" / CONDITION_ID / f"seed_{SEED}"
    summary_path = directory / "summary.json"
    manifest = json.loads(
        (directory / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    record = manifest.get("files", {}).get("summary.json", {})
    if (
        not summary_path.is_file()
        or summary_path.stat().st_size != record.get("size_bytes")
        or _sha256(summary_path) != record.get("sha256")
    ):
        raise ValueError("original 93202 summary failed artifact verification")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks = summary.get("mechanical_checks", {})
    relevant_mechanics = [
        value for key, value in checks.items() if key != "stationarity_pass"
    ]
    stationarity = summary.get("stationarity", {}).get("checks", {})
    failed_T_u = {
        key
        for key, value in stationarity.items()
        if not value and not str(key).lower().startswith(("qx_", "qy_"))
    }
    if (
        summary.get("status") != "complete_MV3_reference_seed"
        or summary.get("decision") != "hold_MV3_reference_seed"
        or not relevant_mechanics
        or not all(relevant_mechanics)
        or failed_T_u != {"temperature_min_K"}
    ):
        raise ValueError("original 93202 failure differs from the repair authorization")
    return {
        "summary_sha256": _sha256(summary_path),
        "failed_T_u_stationarity_checks": sorted(failed_T_u),
        "temperature_min_K": summary["stationarity"]["tracked"]["temperature_min_K"],
    }


def run_repair(output_root: Path, *, resume: bool = True) -> dict[str, Any]:
    original_failure = _validate_original_failure(output_root)
    output = (
        output_root
        / "reference_stability_repair"
        / CONDITION_ID
        / f"seed_{SEED}"
    )
    original_configuration = base.stage_configuration
    base.stage_configuration = stage_configuration
    try:
        result = base.run_reference_seed(
            condition_id=CONDITION_ID,
            seed=SEED,
            output_dir=output,
            resume=resume,
            progress=lambda step, total: print(
                json.dumps({"step": step, "total": total}), flush=True
            ),
        )
    finally:
        base.stage_configuration = original_configuration
    if result.get("status") != "complete_MV3_reference_seed":
        return result
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    summary.update(
        {
            "stage": STAGE,
            "status": "complete_MV3_reference_stability_repair_seed",
            "repair_contract": load_protocol()["repair_contract"],
            "superseded_original": original_failure,
        }
    )
    summary["decision"] = (
        "accept_MV3_reference_stability_repair_seed"
        if all(summary.get("mechanical_checks", {}).values())
        else "hold_MV3_reference_stability_repair_seed"
    )
    _atomic_write_json(output / "summary.json", summary)
    manifest_path = output / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"stage": STAGE, "status": summary["status"]})
    manifest["files"]["summary.json"] = {
        "sha256": _sha256(output / "summary.json"),
        "size_bytes": (output / "summary.json").stat().st_size,
    }
    _atomic_write_json(manifest_path, manifest)
    return _strict_json_ready(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verify-lock-only", action="store_true")
    args = parser.parse_args()
    if args.verify_lock_only:
        print(json.dumps(verify_lock(), indent=2, sort_keys=True, allow_nan=False))
        return
    if args.output_root is None:
        parser.error("run mode requires --output-root")
    result = run_repair(args.output_root, resume=not args.no_resume)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if result.get("decision") != "accept_MV3_reference_stability_repair_seed":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
