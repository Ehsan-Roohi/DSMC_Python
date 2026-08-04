"""Locked M3 heat-flux precision runner for the R100 Mohammadzadeh case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from . import mohammadzadeh_spatial_refinement as engine
from .mohammadzadeh_production import _atomic_write_json, _strict_json_ready
from .mohammadzadeh_validation import mohammadzadeh_config, reference_directory


PROTOCOL_FILE = "m3_qy_precision_protocol.json"
SEED_FILE = "m3_qy_precision_seed_bank.json"
BASIS_FILE = "m3_r100_basis.json"
LOCK_FILE = "m3_qy_precision_lock.json"
LOCK_STATUS = "locked_after_R100_and_before_any_M3_trajectory"
STAGE = "QY100_precision"
SOURCE_FILES = (
    "vgdsmc/mohammadzadeh_qy_precision.py",
    "vgdsmc/mohammadzadeh_spatial_refinement.py",
    "vgdsmc/event_wall_streaming.py",
    "vgdsmc/ntc_fast.py",
    "vgdsmc/mohammadzadeh_production_m1r.py",
)


def _seed_group(seed_bank: Mapping[str, Any], path: str) -> list[int]:
    value: Any = seed_bank
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"unknown M3 seed group {path!r}")
        value = value[component]
    if not isinstance(value, list) or not value:
        raise ValueError("M3 seed group is empty")
    seeds = [int(item) for item in value]
    if len(seeds) != len(set(seeds)):
        raise ValueError("M3 seed group contains duplicates")
    return seeds


def expected_lock_hashes(reference_dir: Path | None = None) -> dict[str, str]:
    ref_dir = reference_directory() if reference_dir is None else Path(reference_dir)
    root = Path(__file__).resolve().parents[1]
    hashes = {
        "protocol_sha256": engine._sha256(ref_dir / PROTOCOL_FILE),
        "seed_bank_sha256": engine._sha256(ref_dir / SEED_FILE),
        "r100_basis_sha256": engine._sha256(ref_dir / BASIS_FILE),
        "validation_protocol_sha256": engine._sha256(
            ref_dir / "validation_protocol.json"
        ),
    }
    hashes.update(
        {f"source::{name}": engine._sha256(root / name) for name in SOURCE_FILES}
    )
    return hashes


def stage_configuration(
    stage: str,
    seed: int,
    *,
    reference_dir: Path | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    if stage != STAGE:
        raise ValueError(f"M3 precision runner accepts only {STAGE}")
    ref_dir = reference_directory() if reference_dir is None else Path(reference_dir)
    protocol = json.loads((ref_dir / PROTOCOL_FILE).read_text(encoding="utf-8"))
    seed_bank = json.loads((ref_dir / SEED_FILE).read_text(encoding="utf-8"))
    basis = json.loads((ref_dir / BASIS_FILE).read_text(encoding="utf-8"))
    m1r_protocol = json.loads(
        (ref_dir / "m1r_repair_protocol.json").read_text(encoding="utf-8")
    )
    if protocol.get("status") != LOCK_STATUS:
        raise ValueError("M3 protocol has the wrong lock status")
    if basis.get("status") != (
        "attested_after_complete_R100_and_before_any_M3_trajectory"
    ):
        raise ValueError("M3 R100 basis has the wrong status")
    if basis.get("decision") != "authorize_QY100_precision_before_any_R200_trajectory":
        raise ValueError("R100 basis does not authorize M3")
    observed = basis.get("observed_R100", {})
    if observed.get("macroscopic_slip_nrmse", 1.0) > 0.10:
        raise ValueError("R100 slip did not pass the locked NRMSE gate")
    if observed.get("normalized_qy_profile_rse", 0.0) <= 0.20:
        raise ValueError("M3 is unnecessary because R100 qy precision already passed")
    specification = protocol.get("stage")
    if not isinstance(specification, Mapping) or not specification.get("authorized"):
        raise ValueError("M3 stage is not authorized")
    if (
        specification.get("name") != STAGE
        or specification.get("grid") != 100
        or specification.get("steps") != 106250
        or specification.get("sample_start") != 12500
    ):
        raise ValueError("M3 precision specification differs from the lock")
    seeds = _seed_group(seed_bank, str(specification["seed_group"]))
    if seeds != list(range(91901, 91909)) or seed not in seeds:
        raise ValueError("M3 seed is not preregistered")
    prior = json.loads(
        (ref_dir / "m2_refinement_seed_bank.json").read_text(encoding="utf-8")
    )
    prior_seeds = {
        int(item)
        for group in prior.get("development", {}).values()
        if isinstance(group, list)
        for item in group
    }
    if set(seeds) & prior_seeds:
        raise ValueError("M3 seeds overlap M2 seeds")
    lock_path = ref_dir / LOCK_FILE
    if not lock_path.is_file():
        raise FileNotFoundError(f"M3 lock is missing: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected = expected_lock_hashes(ref_dir)
    if lock.get("status") != LOCK_STATUS or lock.get("hashes") != expected:
        raise ValueError("M3 lock does not match its inputs and source bundle")
    physical = protocol["physical_case"]
    cfg = mohammadzadeh_config(
        grid=100,
        particles_per_cell=int(physical["particles_per_cell"]),
        steps=int(specification["steps"]),
        sample_start=int(specification["sample_start"]),
        seed=seed,
        dt_safety=float(physical["dt_safety"]),
    )
    hashes = {**expected, "m3_lock_sha256": engine._sha256(lock_path)}
    return cfg, protocol, dict(specification), m1r_protocol, hashes


def verify_lock() -> dict[str, Any]:
    configurations = [stage_configuration(STAGE, seed) for seed in range(91901, 91909)]
    first = configurations[0]
    return {
        "status": "M3_lock_verified_without_running_trajectories",
        "stage": STAGE,
        "seeds": list(range(91901, 91909)),
        "grid": first[0].nx,
        "particles_per_seed": first[0].nx * first[0].ny * first[0].particles_per_cell,
        "steps": first[0].steps,
        "sample_start": first[0].sample_start,
        "sample_count": first[1]["runtime_contract"]["sample_count"],
        "lock_hashes": first[4],
    }


def run_precision_seed(
    *,
    seed: int,
    output_dir: Path,
    resume: bool = True,
    stop_after_step: int | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    original = engine.stage_configuration
    engine.stage_configuration = stage_configuration
    try:
        summary = engine.run_refinement_seed(
            stage=STAGE,
            seed=seed,
            output_dir=output_dir,
            resume=resume,
            stop_after_step=stop_after_step,
            progress=progress,
        )
    finally:
        engine.stage_configuration = original
    if summary.get("status") != "complete_M2_spatial_development_seed":
        return summary
    summary["status"] = "complete_M3_qy_precision_seed"
    checks = summary.get("mechanical_checks", {})
    mechanics = all(value for key, value in checks.items() if key != "stationarity_pass")
    summary["decision"] = (
        "complete_M3_seed_awaiting_eight_seed_aggregation"
        if mechanics
        else "hold_M3_for_mechanical_diagnosis"
    )
    _atomic_write_json(Path(output_dir) / "summary.json", summary)
    manifest_path = Path(output_dir) / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = summary["status"]
    manifest["files"]["summary.json"] = {
        "sha256": engine._sha256(Path(output_dir) / "summary.json"),
        "size_bytes": (Path(output_dir) / "summary.json").stat().st_size,
    }
    _atomic_write_json(manifest_path, manifest)
    return _strict_json_ready(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--verify-lock-only", action="store_true")
    args = parser.parse_args()
    if args.verify_lock_only:
        print(json.dumps(verify_lock(), indent=2, sort_keys=True, allow_nan=False))
        return
    if args.seed is None or args.output_dir is None:
        parser.error("--seed and --output-dir are required unless --verify-lock-only is used")
    summary = run_precision_seed(
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
