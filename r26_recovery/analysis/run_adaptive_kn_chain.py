#!/usr/bin/env python3
"""Fail-closed adaptive Kn_Gu continuation around the single-stage R26 driver.

The single-stage driver is the scientific authority: this controller only
selects the next Knudsen number, invokes that driver, and promotes a restart
when ``run_summary.json`` says ``termination == \"target_accepted\"`` and a
new ``last_accepted_state.npz`` is present.  Rejected stages are retained as
evidence and are never used as restarts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRIVER = ROOT / "analysis" / "run_jfm_observability_continuation.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def resolve_result_dir(path: Path) -> Path:
    path = path.resolve()
    nested = path / "result"
    if (nested / "run_summary.json").is_file():
        return nested
    return path


def inspect_restart(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {"state", "x", "y", "lid_velocity", "kn_input"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"restart {path} lacks fields {sorted(missing)}")
        if "accepted" in archive and not bool(np.asarray(archive["accepted"]).item()):
            raise ValueError(f"restart {path} is explicitly marked rejected")
        state = np.asarray(archive["state"], dtype=float)
        x = np.asarray(archive["x"], dtype=float)
        y = np.asarray(archive["y"], dtype=float)
        lid = float(np.asarray(archive["lid_velocity"]).item())
        kn = float(np.asarray(archive["kn_input"]).item())
        convention = (
            str(np.asarray(archive["kn_convention"]).item())
            if "kn_convention" in archive
            else None
        )
    if state.ndim != 3 or state.shape[:2] != (y.size, x.size):
        raise ValueError(f"restart {path} has inconsistent grid/state dimensions")
    if not np.isfinite(state).all() or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"restart {path} contains non-finite data")
    if not np.isfinite(kn) or kn <= 0.0 or not np.isfinite(lid) or lid < 0.0:
        raise ValueError(f"restart {path} has invalid Kn or lid velocity")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "kn": kn,
        "lid": lid,
        "convention": convention,
        "nodes": int(x.size),
        "x": x,
        "y": y,
    }


def validate_pair(previous: dict[str, Any], current: dict[str, Any], nodes: int) -> None:
    if previous["nodes"] != nodes or current["nodes"] != nodes:
        raise ValueError("restart grid size does not match --nodes")
    if not np.array_equal(previous["x"], current["x"]) or not np.array_equal(
        previous["y"], current["y"]
    ):
        raise ValueError("previous/current restart grids differ")
    if not math.isclose(previous["lid"], current["lid"], rel_tol=0.0, abs_tol=1.0e-14):
        raise ValueError("previous/current restart lid velocities differ")
    if previous["convention"] != current["convention"]:
        raise ValueError("previous/current Kn conventions differ")
    if not previous["kn"] < current["kn"]:
        raise ValueError("restart Kn values must satisfy previous < current")


def accepted_stage(result_dir: Path, expected_kn: float) -> tuple[bool, dict[str, Any]]:
    summary_path = result_dir / "run_summary.json"
    state_path = result_dir / "last_accepted_state.npz"
    if not summary_path.is_file():
        return False, {"reason": "missing_run_summary"}
    summary = load_json(summary_path)
    case = summary.get("case", {})
    case_kn = case.get("kn_input") if isinstance(case, dict) else None
    accepted_attempt = any(
        isinstance(item, dict) and item.get("accepted") is True
        for item in summary.get("attempts", [])
    )
    checks = {
        "termination": summary.get("termination") == "target_accepted",
        "accepted_attempt": accepted_attempt,
        "state_present": state_path.is_file(),
        "case_kn": bool(
            case_kn is not None
            and math.isclose(float(case_kn), expected_kn, rel_tol=0.0, abs_tol=1.0e-14)
        ),
    }
    return all(checks.values()), {
        "checks": checks,
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "state": str(state_path) if state_path.is_file() else None,
        "state_sha256": sha256(state_path) if state_path.is_file() else None,
    }


def tag(value: float) -> str:
    return f"{value:.12f}".replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-stage-dir", type=Path, required=True)
    parser.add_argument("--previous-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--driver", type=Path, default=DEFAULT_DRIVER)
    parser.add_argument("--nodes", type=int, default=40)
    parser.add_argument("--target-kn-gu", type=float, default=0.2)
    parser.add_argument("--initial-step", type=float)
    parser.add_argument("--maximum-step", type=float, default=0.01)
    parser.add_argument("--minimum-step", type=float, default=0.0001)
    parser.add_argument("--growth-factor", type=float, default=1.35)
    parser.add_argument("--max-stages", type=int, default=100)
    parser.add_argument("--max-nfev", type=int, default=120)
    args = parser.parse_args()

    driver = args.driver.resolve()
    if not driver.is_file():
        raise FileNotFoundError(driver)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse nonempty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    initial_result = resolve_result_dir(args.initial_stage_dir)
    initial_summary = load_json(initial_result / "run_summary.json")
    case = initial_summary.get("case", {})
    current_kn = float(case.get("kn_input", -1.0))
    initial_ok, initial_evidence = accepted_stage(initial_result, current_kn)
    if not initial_ok:
        raise ValueError(f"initial stage is not strictly accepted: {initial_evidence}")

    current_state = initial_result / "last_accepted_state.npz"
    previous_state = args.previous_state.resolve()
    previous = inspect_restart(previous_state)
    current = inspect_restart(current_state)
    validate_pair(previous, current, args.nodes)
    if not math.isclose(current["kn"], current_kn, rel_tol=0.0, abs_tol=1.0e-14):
        raise ValueError("initial summary/restart Kn mismatch")
    if not current_kn < args.target_kn_gu:
        raise ValueError("initial Kn must be below target")

    inferred_step = current_kn - previous["kn"]
    step = inferred_step if args.initial_step is None else args.initial_step
    if not (
        np.isfinite(step)
        and np.isfinite(args.minimum_step)
        and np.isfinite(args.maximum_step)
        and 0.0 < args.minimum_step <= step <= args.maximum_step
    ):
        raise ValueError("invalid adaptive step controls")
    if not np.isfinite(args.growth_factor) or args.growth_factor <= 1.0:
        raise ValueError("growth factor must exceed one")

    attempts: list[dict[str, Any]] = []
    status = "running"
    summary_path = args.output_dir / "adaptive_chain_summary.json"

    for attempt in range(1, args.max_stages + 1):
        if current_kn >= args.target_kn_gu - 1.0e-14:
            status = "target_accepted"
            break
        proposed = min(args.target_kn_gu, current_kn + step)
        proposed = float(f"{proposed:.15g}")
        stage = args.output_dir / f"stage_{attempt:03d}_KnGu_{tag(proposed)}"
        stage.mkdir()
        result_dir = stage / "result"
        stdout_path = stage / "driver.out"
        stderr_path = stage / "driver.err"
        command = [
            sys.executable,
            str(driver),
            "--nodes",
            str(args.nodes),
            "--case-family",
            "jfm-observability",
            "--kn-gu",
            repr(proposed),
            "--initial-state",
            str(current_state),
            "--previous-state",
            str(previous_state),
            "--reconcile-initial",
            "--max-nfev",
            str(args.max_nfev),
            "--output-dir",
            str(result_dir),
        ]
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        accepted, evidence = accepted_stage(result_dir, proposed)
        record = {
            "attempt": attempt,
            "from_kn": current_kn,
            "proposed_kn": proposed,
            "step": step,
            "returncode": completed.returncode,
            "accepted": accepted,
            "stage": str(stage),
            "evidence": evidence,
        }
        attempts.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        write_json_atomic(
            summary_path,
            {
                "status": "running",
                "target_kn_gu": args.target_kn_gu,
                "current_kn_gu": current_kn,
                "strict_stage_gates_unchanged": True,
                "driver": str(driver),
                "driver_sha256": sha256(driver),
                "initial_stage_evidence": initial_evidence,
                "attempts": attempts,
            },
        )

        if completed.returncode != 0 or not (result_dir / "run_summary.json").is_file():
            status = "driver_error"
            break

        if accepted:
            previous_state = current_state
            previous = current
            current_state = result_dir / "last_accepted_state.npz"
            current = inspect_restart(current_state)
            validate_pair(previous, current, args.nodes)
            current_kn = proposed
            step = min(args.maximum_step, args.growth_factor * step)
        else:
            step *= 0.5
            if step < args.minimum_step:
                status = "minimum_step_rejected"
                break
    else:
        status = "maximum_stages_reached"

    if current_kn >= args.target_kn_gu - 1.0e-14:
        status = "target_accepted"
    final = {
        "status": status,
        "target_kn_gu": args.target_kn_gu,
        "current_kn_gu": current_kn,
        "strict_stage_gates_unchanged": True,
        "driver": str(driver),
        "driver_sha256": sha256(driver),
        "initial_stage_evidence": initial_evidence,
        "attempts": attempts,
    }
    if status == "target_accepted":
        shutil.copy2(current_state, args.output_dir / "last_accepted_state.npz")
        final["final_state"] = str(args.output_dir / "last_accepted_state.npz")
        final["final_state_sha256"] = sha256(args.output_dir / "last_accepted_state.npz")
    write_json_atomic(summary_path, final)
    return 0 if status == "target_accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
