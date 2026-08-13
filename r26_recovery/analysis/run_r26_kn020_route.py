#!/usr/bin/env python3
"""Run one fail-closed recovery route to the JFM R26 Kn_Gu=0.2 target.

The route controller never promotes a rejected nonlinear state.  It invokes
the audited single-case driver in fresh directories, verifies the driver's
strict acceptance evidence, and only then uses ``last_accepted_state.npz``
for the next Kn or grid.  Three independent routes are supported so a poor
nonlinear basin in one method does not block the other methods.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


TARGET_KN = 0.2
RAW_TOLERANCE = 1.0e-8
ROUTES = ("direct_colored", "direct_trf", "kn_ladder")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def explicit_true(value: Any) -> bool:
    return value is True or (
        isinstance(value, int) and not isinstance(value, bool) and value == 1
    )


def accepted_result(result_dir: Path, *, expected_kn: float, expected_nodes: int) -> tuple[bool, dict[str, Any]]:
    summary_path = result_dir / "run_summary.json"
    state_path = result_dir / "last_accepted_state.npz"
    evidence: dict[str, Any] = {
        "summary": str(summary_path),
        "state": str(state_path),
    }
    if not summary_path.is_file():
        evidence["reason"] = "missing_run_summary"
        return False, evidence
    summary = load_json(summary_path)
    case = summary.get("case", {})
    attempts = summary.get("attempts", [])
    accepted_attempts = [
        item for item in attempts
        if isinstance(item, dict) and explicit_true(item.get("accepted"))
    ]
    last_accepted = accepted_attempts[-1] if accepted_attempts else {}
    raw_gate = float(last_accepted.get("raw_acceptance_gate", math.inf))
    checks = {
        "termination": summary.get("termination") == "target_accepted",
        "accepted_attempt": bool(accepted_attempts),
        "raw_gate": bool(np.isfinite(raw_gate) and raw_gate <= RAW_TOLERANCE),
        "case_kn": bool(
            isinstance(case, dict)
            and case.get("kn_input") is not None
            and math.isclose(float(case["kn_input"]), expected_kn, rel_tol=0.0, abs_tol=1.0e-14)
        ),
        "case_nodes": bool(
            isinstance(case, dict) and int(case.get("nodes", -1)) == expected_nodes
        ),
        "state_present": state_path.is_file(),
    }
    evidence.update(
        {
            "checks": checks,
            "raw_acceptance_gate": raw_gate,
            "summary_sha256": sha256(summary_path),
        }
    )
    if not all(checks.values()):
        return False, evidence
    try:
        with np.load(state_path, allow_pickle=False) as archive:
            state = np.asarray(archive["state"], dtype=float)
            x = np.asarray(archive["x"], dtype=float)
            y = np.asarray(archive["y"], dtype=float)
            kn = float(np.asarray(archive["kn_input"]).item())
            convention = str(np.asarray(archive["kn_convention"]).item())
            lid = float(np.asarray(archive["lid_velocity"]).item())
        state_checks = {
            "shape": state.shape == (expected_nodes, expected_nodes, 17),
            "coordinates": x.shape == (expected_nodes,) and y.shape == (expected_nodes,),
            "finite": bool(np.isfinite(state).all()),
            "rho_positive": bool(np.all(state[..., 0] > 0.0)),
            "temperature_positive": bool(np.all(state[..., 3] > 0.0)),
            "kn": math.isclose(kn, expected_kn, rel_tol=0.0, abs_tol=1.0e-14),
            "kn_convention": convention == "gu_lambda_over_L",
            "lid": bool(np.isfinite(lid) and lid > 0.0),
        }
        evidence["state_checks"] = state_checks
        evidence["state_sha256"] = sha256(state_path)
        return all(state_checks.values()), evidence
    except (KeyError, OSError, ValueError) as exc:
        evidence["state_error"] = f"{type(exc).__name__}: {exc}"
        return False, evidence


class RouteController:
    def __init__(
        self,
        *,
        route: str,
        repo_root: Path,
        output_dir: Path,
        winner_dir: Path,
        refine_nodes: tuple[int, ...],
    ) -> None:
        self.route = route
        self.repo_root = repo_root.resolve()
        self.driver = self.repo_root / "r26_recovery" / "analysis" / "run_jfm_observability_continuation.py"
        self.code_dir = self.repo_root / "r26_recovery" / "code"
        self.output_dir = output_dir.resolve()
        self.winner_dir = winner_dir.resolve()
        self.refine_nodes = refine_nodes
        self.records: list[dict[str, Any]] = []
        self.stage_index = 0
        self.started_utc = utc_now()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        if not self.driver.is_file() or not self.code_dir.is_dir():
            raise FileNotFoundError("R26 driver/code directory is missing from the checkout")

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "route_summary.json"

    def other_winner_exists(self) -> bool:
        return self.winner_dir.exists()

    def save_summary(self, status: str, **extra: Any) -> None:
        payload: dict[str, Any] = {
            "route": self.route,
            "status": status,
            "started_utc": self.started_utc,
            "updated_utc": utc_now(),
            "target_kn_gu": TARGET_KN,
            "raw_tolerance": RAW_TOLERANCE,
            "refine_nodes": list(self.refine_nodes),
            "driver": str(self.driver),
            "driver_sha256": sha256(self.driver),
            "records": self.records,
        }
        payload.update(extra)
        write_json_atomic(self.summary_path, payload)

    def run_stage(
        self,
        *,
        label: str,
        nodes: int,
        kn: float,
        solver: str,
        max_nfev: int,
        initial_state: Path | None = None,
        smoke_lid: float = 0.001,
        lid_step: float = 0.04,
        minimum_lid_step: float = 0.00125,
    ) -> Path | None:
        self.stage_index += 1
        stage_dir = self.output_dir / f"stage_{self.stage_index:03d}_{label}"
        result_dir = stage_dir / "result"
        stage_dir.mkdir()
        command = [
            sys.executable,
            str(self.driver),
            "--nodes", str(nodes),
            "--case-family", "jfm-observability",
            "--kn-gu", repr(float(kn)),
            "--lid-speed-m-s", "100.0",
            "--wall-temperature-k", "300.0",
            "--vhs-omega", "0.81",
            "--beta", "2.5",
            "--closure-mode", "jfm2009",
            "--solver", solver,
            "--raw-tolerance", repr(RAW_TOLERANCE),
            "--solver-tolerance", "1.0e-9",
            "--smoke-lid", repr(smoke_lid),
            "--initial-step", repr(lid_step),
            "--minimum-step", repr(minimum_lid_step),
            "--max-nfev", str(max_nfev),
            "--output-dir", str(result_dir),
        ]
        if initial_state is not None:
            command.extend(["--initial-state", str(initial_state), "--reconcile-initial"])
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(self.code_dir) + os.pathsep + environment.get("PYTHONPATH", "")
        started = utc_now()
        with (stage_dir / "driver.out").open("w") as stdout, (stage_dir / "driver.err").open("w") as stderr:
            completed = subprocess.run(
                command,
                cwd=self.repo_root / "r26_recovery",
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        accepted, evidence = accepted_result(result_dir, expected_kn=kn, expected_nodes=nodes)
        record = {
            "stage": self.stage_index,
            "label": label,
            "nodes": nodes,
            "kn_gu": kn,
            "solver": solver,
            "max_nfev": max_nfev,
            "initial_state": str(initial_state) if initial_state is not None else None,
            "started_utc": started,
            "finished_utc": utc_now(),
            "returncode": completed.returncode,
            "accepted": accepted,
            "evidence": evidence,
            "command": command,
        }
        self.records.append(record)
        self.save_summary("running")
        print(json.dumps(record, sort_keys=True), flush=True)
        return result_dir / "last_accepted_state.npz" if accepted else None

    def direct_target(self, solver: str) -> Path | None:
        max_nfev = 220 if solver == "colored_newton" else 500
        return self.run_stage(
            label=f"direct_Kn0p2_N20_{solver}",
            nodes=20,
            kn=TARGET_KN,
            solver=solver,
            max_nfev=max_nfev,
            smoke_lid=0.001,
            lid_step=0.04 if solver == "colored_newton" else 0.05,
            minimum_lid_step=0.000625,
        )

    def kn_ladder(self) -> Path | None:
        current_kn = 0.05
        current_state = self.run_stage(
            label="ladder_base_Kn0p05_N20",
            nodes=20,
            kn=current_kn,
            solver="colored_newton",
            max_nfev=220,
            smoke_lid=0.001,
            lid_step=0.04,
            minimum_lid_step=0.000625,
        )
        if current_state is None:
            return None
        step = 0.025
        minimum_step = 0.0015625
        attempts = 0
        while current_kn < TARGET_KN - 1.0e-14:
            if self.other_winner_exists():
                self.save_summary("stopped_by_other_winner", current_kn_gu=current_kn)
                return None
            attempts += 1
            if attempts > 50:
                self.save_summary("kn_ladder_attempt_limit", current_kn_gu=current_kn)
                return None
            proposed = min(TARGET_KN, current_kn + step)
            proposed = float(f"{proposed:.15g}")
            tag = f"{proposed:.8f}".replace(".", "p")
            candidate = self.run_stage(
                label=f"ladder_Kn{tag}_N20",
                nodes=20,
                kn=proposed,
                solver="colored_newton",
                max_nfev=260,
                initial_state=current_state,
            )
            if candidate is not None:
                current_state = candidate
                current_kn = proposed
                step = min(0.025, 1.35 * step)
            else:
                step *= 0.5
                if step < minimum_step:
                    self.save_summary(
                        "kn_ladder_minimum_step_rejected",
                        current_kn_gu=current_kn,
                        rejected_step=step,
                    )
                    return None
        return current_state

    def refine(self, state: Path) -> Path | None:
        current = state
        primary = "least_squares" if self.route == "direct_trf" else "colored_newton"
        fallback = "colored_newton" if primary == "least_squares" else "least_squares"
        for nodes in self.refine_nodes:
            if nodes <= 20:
                continue
            if self.other_winner_exists():
                self.save_summary("stopped_by_other_winner", last_state=str(current))
                return None
            candidate = self.run_stage(
                label=f"refine_N{nodes}_{primary}",
                nodes=nodes,
                kn=TARGET_KN,
                solver=primary,
                max_nfev=300 if primary == "colored_newton" else 600,
                initial_state=current,
            )
            if candidate is None:
                candidate = self.run_stage(
                    label=f"refine_N{nodes}_{fallback}_fallback",
                    nodes=nodes,
                    kn=TARGET_KN,
                    solver=fallback,
                    max_nfev=360 if fallback == "colored_newton" else 700,
                    initial_state=current,
                )
            if candidate is None:
                self.save_summary("grid_refinement_rejected", failed_nodes=nodes)
                return None
            current = candidate
        return current

    def declare_winner(self, state: Path) -> bool:
        try:
            self.winner_dir.mkdir(parents=False)
        except FileExistsError:
            self.save_summary("accepted_but_other_winner_exists", final_state=str(state))
            return False
        payload = {
            "route": self.route,
            "created_utc": utc_now(),
            "target_kn_gu": TARGET_KN,
            "final_nodes": self.refine_nodes[-1],
            "final_state": str(state),
            "final_state_sha256": sha256(state),
            "route_summary": str(self.summary_path),
        }
        write_json_atomic(self.winner_dir / "winner.json", payload)
        self.save_summary("target_accepted", **payload)
        return True

    def run(self) -> int:
        self.save_summary("starting")
        if self.other_winner_exists():
            self.save_summary("stopped_by_other_winner")
            return 0
        if self.route == "direct_colored":
            state = self.direct_target("colored_newton")
        elif self.route == "direct_trf":
            state = self.direct_target("least_squares")
        else:
            state = self.kn_ladder()
        if state is None:
            status = str(load_json(self.summary_path).get("status"))
            if status == "running":
                self.save_summary("discovery_rejected")
                status = "discovery_rejected"
            return 0 if status == "stopped_by_other_winner" else 2
        final_state = self.refine(state)
        if final_state is None:
            status = str(load_json(self.summary_path).get("status"))
            return 0 if status == "stopped_by_other_winner" else 2
        self.declare_winner(final_state)
        return 0


def parse_nodes(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(value < 20 for value in values) or tuple(sorted(set(values))) != values:
        raise argparse.ArgumentTypeError("refine nodes must be unique increasing integers >=20")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--winner-dir", type=Path, required=True)
    parser.add_argument("--refine-nodes", type=parse_nodes, default=(24, 28, 32, 36, 40))
    args = parser.parse_args()
    controller = RouteController(
        route=args.route,
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        winner_dir=args.winner_dir,
        refine_nodes=args.refine_nodes,
    )
    return controller.run()


if __name__ == "__main__":
    raise SystemExit(main())
