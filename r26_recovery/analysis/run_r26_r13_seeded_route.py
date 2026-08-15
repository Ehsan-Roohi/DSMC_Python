#!/usr/bin/env python3
"""Recover the JFM R26 Kn_Gu=0.2 branch from the accepted R13/N60 state.

R13 and the planar R26 implementation use the same ordered 17-component
state.  The only transfer required is therefore an explicit coordinate-grid
transfer.  The transferred field is always treated as an *unaccepted seed*;
the existing fail-closed R26 driver must reconcile it before promotion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator


TARGET_KN = 0.2
RAW_TOLERANCE = 1.0e-8
STATE_ORDER = (
    "rho", "vx", "vy", "theta", "qx", "qy", "sigma_xx", "sigma_xy",
    "sigma_yy", "R_xx", "R_xy", "R_yy", "m_xxx", "m_xxy", "m_xyy",
    "m_yyy", "Delta",
)


@dataclass(frozen=True)
class RouteSpec:
    amplitude: float
    initial_nodes: int
    initial_beta: float
    solver: str = "colored_newton"


ROUTES = {
    "r13_full_uniform": RouteSpec(1.0, 20, 0.0),
    "r13_half_stretched": RouteSpec(0.5, 16, 2.5),
    "r13_quarter_stretched": RouteSpec(0.25, 12, 2.5),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_r13_source(result_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    state_path = result_dir / "state.npy"
    report_path = result_dir / "report.json"
    if not state_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("accepted R13 result needs state.npy and report.json")
    report = load_object(report_path)
    configuration = report.get("configuration", {})
    provenance = report.get("execution_provenance", {})
    expected_hash = provenance.get("output_state_sha256")
    checks = {
        "private_gates": report.get("passed_private_run_gates") is True,
        "metrics_valid": report.get("metrics_valid") is True,
        "accepted_semantics": report.get("state_semantics")
        == "accepted_private_physical_solution",
        "n60": configuration.get("nx") == 60 and configuration.get("ny") == 60,
        "equivalent_kn": math.isclose(
            float(configuration.get("kn", math.nan)),
            TARGET_KN * math.sqrt(2.0 / math.pi),
            rel_tol=0.0,
            abs_tol=1.0e-14,
        ),
        "state_hash": isinstance(expected_hash, str) and digest(state_path) == expected_hash,
    }
    if not all(checks.values()):
        raise ValueError(f"R13/N60 source verification failed: {checks}")
    state = np.asarray(np.load(state_path, allow_pickle=False), dtype=float)
    if state.shape != (60, 60, len(STATE_ORDER)) or not np.isfinite(state).all():
        raise ValueError(f"unexpected R13 state shape/content: {state.shape}")
    if np.any(state[..., 0] <= 0.0) or np.any(state[..., 3] <= 0.0):
        raise ValueError("accepted R13 source violates rho/theta positivity")
    lid = float(configuration.get("lid_velocity", math.nan))
    if not np.isfinite(lid) or lid <= 0.0:
        raise ValueError("R13 report has no finite positive lid velocity")
    return state, {
        "checks": checks,
        "state": str(state_path.resolve()),
        "state_sha256": digest(state_path),
        "report": str(report_path.resolve()),
        "report_sha256": digest(report_path),
        "source_lid_velocity": lid,
        "state_order": list(STATE_ORDER),
    }


def transfer_and_blend(
    source: np.ndarray,
    *,
    target_x: np.ndarray,
    target_y: np.ndarray,
    amplitude: float,
    mass_weights: np.ndarray,
) -> np.ndarray:
    if not 0.0 < amplitude <= 1.0:
        raise ValueError("amplitude must lie in (0,1]")
    source_ny, source_nx = source.shape[:2]
    source_x = np.arange(1, source_nx + 1, dtype=float) / (source_nx + 1.0)
    source_y = np.arange(1, source_ny + 1, dtype=float) / (source_ny + 1.0)
    yy, xx = np.meshgrid(target_y, target_x, indexing="ij")
    points = np.column_stack((yy.ravel(), xx.ravel()))
    transferred = np.empty((target_y.size, target_x.size, source.shape[-1]))
    for component in range(source.shape[-1]):
        values = source[..., component]
        logarithmic = component in (0, 3)
        if logarithmic:
            values = np.log(values)
        interpolator = RegularGridInterpolator(
            (source_y, source_x),
            values,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        values_on_target = interpolator(points).reshape(target_y.size, target_x.size)
        if logarithmic:
            # Logarithmic amplitude blending preserves positivity exactly.
            transferred[..., component] = np.exp(amplitude * values_on_target)
        else:
            transferred[..., component] = amplitude * values_on_target
    if not np.isfinite(transferred).all():
        raise ValueError("R13-to-R26 transfer produced non-finite data")
    if np.any(transferred[..., 0] <= 0.0) or np.any(transferred[..., 3] <= 0.0):
        raise ValueError("R13-to-R26 transfer violated positivity")
    weights = np.asarray(mass_weights, dtype=float)
    weights = weights / np.sum(weights)
    transferred[..., 0] /= float(np.sum(weights * transferred[..., 0]))
    return transferred


class Controller:
    def __init__(self, args: argparse.Namespace) -> None:
        self.route = args.route
        self.spec = ROUTES[self.route]
        self.repo_root = args.repo_root.resolve()
        self.output_dir = args.output_dir.resolve()
        self.winner_dir = args.winner_dir.resolve()
        self.r13_result = args.r13_result.resolve()
        self.refine_nodes = args.refine_nodes
        self.reconcile_timeout = args.reconcile_timeout
        self.continuation_timeout = args.continuation_timeout
        self.refinement_timeout = args.refinement_timeout
        self.driver = self.repo_root / "r26_recovery/analysis/run_jfm_observability_continuation.py"
        self.code_dir = self.repo_root / "r26_recovery/code"
        self.records: list[dict[str, Any]] = []
        self.stage = 0
        self.started = utc_now()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        if not self.driver.is_file() or not self.code_dir.is_dir():
            raise FileNotFoundError("R26 driver/code directory is missing")
        sys.path.insert(0, str(self.code_dir))
        from r26_cases import jfm_observability_cavity_case
        from r26_fv_backend import wall_bounded_control_volume_weights

        self.case_factory = jfm_observability_cavity_case
        self.mass_weights = wall_bounded_control_volume_weights

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "route_summary.json"

    def save(self, status: str, **extra: Any) -> None:
        payload: dict[str, Any] = {
            "route": self.route,
            "status": status,
            "started_utc": self.started,
            "updated_utc": utc_now(),
            "target_kn_gu": TARGET_KN,
            "raw_tolerance": RAW_TOLERANCE,
            "route_spec": self.spec.__dict__,
            "records": self.records,
        }
        payload.update(extra)
        write_atomic(self.summary_path, payload)

    def build_seed(self) -> tuple[Path, float]:
        source, provenance = verify_r13_source(self.r13_result)
        case = self.case_factory(
            self.spec.initial_nodes,
            kn=TARGET_KN,
            lid_speed_m_per_s=100.0,
            wall_temperature_K=300.0,
            viscosity_exponent=0.81,
            grid_stretch_beta=self.spec.initial_beta,
        )
        weights = self.mass_weights(case.x, case.y)
        state = transfer_and_blend(
            source,
            target_x=case.x,
            target_y=case.y,
            amplitude=self.spec.amplitude,
            mass_weights=weights,
        )
        source_lid = float(provenance["source_lid_velocity"])
        seed_lid = self.spec.amplitude * source_lid
        if seed_lid > case.lid_velocity + 1.0e-14:
            raise ValueError("R13-derived seed lid exceeds the R26 target lid")
        seed = self.output_dir / "r13_derived_unaccepted_seed.npz"
        np.savez_compressed(
            seed,
            state=state,
            x=case.x,
            y=case.y,
            lid_velocity=seed_lid,
            kn_input=case.kn,
            kn_convention=case.kn_convention.value,
            mu_equilibrium=case.mu_equilibrium,
            beta=case.grid_stretch_beta,
        )
        manifest = {
            "semantics": "unaccepted_R13_derived_initial_guess_requires_R26_reconciliation",
            "route": self.route,
            "amplitude": self.spec.amplitude,
            "target_nodes": self.spec.initial_nodes,
            "target_beta": self.spec.initial_beta,
            "seed_lid_velocity": seed_lid,
            "target_lid_velocity": case.lid_velocity,
            "seed": str(seed),
            "seed_sha256": digest(seed),
            "source": provenance,
            "finite": bool(np.isfinite(state).all()),
            "rho_min": float(np.min(state[..., 0])),
            "theta_min": float(np.min(state[..., 3])),
        }
        write_atomic(self.output_dir / "r13_seed_manifest.json", manifest)
        return seed, seed_lid

    def accepted_result(self, result: Path, nodes: int, kn: float) -> tuple[bool, dict[str, Any]]:
        summary_path = result / "run_summary.json"
        state_path = result / "last_accepted_state.npz"
        evidence: dict[str, Any] = {"summary": str(summary_path), "state": str(state_path)}
        if not summary_path.is_file():
            evidence["reason"] = "missing_run_summary"
            return False, evidence
        summary = load_object(summary_path)
        attempts = summary.get("attempts", [])
        accepted = [item for item in attempts if isinstance(item, dict) and item.get("accepted") is True]
        last = accepted[-1] if accepted else {}
        raw_gate = float(last.get("raw_acceptance_gate", math.inf))
        case = summary.get("case", {})
        checks = {
            "termination": summary.get("termination") == "target_accepted",
            "accepted_attempt": bool(accepted),
            "raw_gate": np.isfinite(raw_gate) and raw_gate <= RAW_TOLERANCE,
            "case_kn": math.isclose(float(case.get("kn_input", math.nan)), kn, rel_tol=0.0, abs_tol=1.0e-14),
            "case_nodes": case.get("nodes") == nodes,
            "state_present": state_path.is_file(),
        }
        evidence.update({"checks": checks, "raw_acceptance_gate": raw_gate})
        if not all(checks.values()):
            return False, evidence
        with np.load(state_path, allow_pickle=False) as archive:
            state = np.asarray(archive["state"], dtype=float)
            state_checks = {
                "shape": state.shape == (nodes, nodes, len(STATE_ORDER)),
                "finite": bool(np.isfinite(state).all()),
                "rho_positive": bool(np.all(state[..., 0] > 0.0)),
                "theta_positive": bool(np.all(state[..., 3] > 0.0)),
            }
        evidence["state_checks"] = state_checks
        evidence["state_sha256"] = digest(state_path)
        return all(state_checks.values()), evidence

    def run_stage(
        self,
        *,
        label: str,
        nodes: int,
        beta: float,
        initial_state: Path,
        timeout: int,
        reconcile: bool,
        target_lid: float | None = None,
        initial_step: float = 0.005,
        max_nfev: int = 220,
    ) -> Path | None:
        if self.winner_dir.exists():
            self.save("stopped_by_other_winner")
            return None
        self.stage += 1
        stage_dir = self.output_dir / f"stage_{self.stage:03d}_{label}"
        result = stage_dir / "result"
        stage_dir.mkdir()
        command = [
            sys.executable, str(self.driver),
            "--nodes", str(nodes),
            "--case-family", "jfm-observability",
            "--kn-gu", repr(TARGET_KN),
            "--lid-speed-m-s", "100.0",
            "--wall-temperature-k", "300.0",
            "--vhs-omega", "0.81",
            "--beta", repr(beta),
            "--closure-mode", "jfm2009",
            "--solver", self.spec.solver,
            "--raw-tolerance", repr(RAW_TOLERANCE),
            "--solver-tolerance", "1.0e-9",
            "--initial-step", repr(initial_step),
            "--minimum-step", "0.0003125",
            "--max-nfev", str(max_nfev),
            "--initial-state", str(initial_state),
            "--output-dir", str(result),
        ]
        if reconcile:
            command.append("--reconcile-initial")
        if target_lid is not None:
            command.extend(["--target-lid", repr(target_lid)])
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(self.code_dir) + os.pathsep + environment.get("PYTHONPATH", "")
        status_path = stage_dir / "stage_status.json"
        write_atomic(status_path, {"status": "running", "started_utc": utc_now(), "command": command})
        started_utc = utc_now()
        started = time.monotonic()
        returncode: int | None = None
        timed_out = False
        with (stage_dir / "driver.out").open("w") as stdout, (stage_dir / "driver.err").open("w") as stderr:
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.repo_root / "r26_recovery",
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                    timeout=timeout,
                )
                returncode = completed.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
        accepted, evidence = self.accepted_result(result, nodes, TARGET_KN)
        record = {
            "stage": self.stage,
            "label": label,
            "nodes": nodes,
            "beta": beta,
            "initial_state": str(initial_state),
            "solver": self.spec.solver,
            "timeout_seconds": timeout,
            "timed_out": timed_out,
            "returncode": returncode,
            "started_utc": started_utc,
            "finished_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "accepted": accepted,
            "evidence": evidence,
            "command": command,
        }
        self.records.append(record)
        write_atomic(status_path, record)
        self.save("running")
        print(json.dumps(record, sort_keys=True), flush=True)
        return result / "last_accepted_state.npz" if accepted else None

    def run(self) -> int:
        self.save("starting")
        seed, seed_lid = self.build_seed()
        state = self.run_stage(
            label="r13_seed_reconciliation",
            nodes=self.spec.initial_nodes,
            beta=self.spec.initial_beta,
            initial_state=seed,
            timeout=self.reconcile_timeout,
            reconcile=True,
            target_lid=seed_lid,
            initial_step=0.0025,
        )
        if state is None:
            if load_object(self.summary_path).get("status") == "stopped_by_other_winner":
                return 0
            self.save("r13_seed_reconciliation_rejected")
            return 2
        state = self.run_stage(
            label="lid_to_physical_target",
            nodes=self.spec.initial_nodes,
            beta=self.spec.initial_beta,
            initial_state=state,
            timeout=self.continuation_timeout,
            reconcile=False,
            initial_step=0.005,
            max_nfev=180,
        )
        if state is None:
            if load_object(self.summary_path).get("status") == "stopped_by_other_winner":
                return 0
            self.save("physical_lid_rejected")
            return 2
        if self.spec.initial_beta != 2.5:
            state = self.run_stage(
                label="stretch_to_beta2p5",
                nodes=self.spec.initial_nodes,
                beta=2.5,
                initial_state=state,
                timeout=self.refinement_timeout,
                reconcile=True,
                target_lid=None,
            )
            if state is None:
                self.save("stretch_reconciliation_rejected")
                return 2
        current_nodes = self.spec.initial_nodes
        for nodes in self.refine_nodes:
            if nodes <= current_nodes:
                continue
            state = self.run_stage(
                label=f"refine_N{nodes}",
                nodes=nodes,
                beta=2.5,
                initial_state=state,
                timeout=self.refinement_timeout,
                reconcile=True,
                max_nfev=260,
            )
            if state is None:
                if load_object(self.summary_path).get("status") == "stopped_by_other_winner":
                    return 0
                self.save("grid_refinement_rejected", failed_nodes=nodes)
                return 2
            current_nodes = nodes
        try:
            self.winner_dir.mkdir()
        except FileExistsError:
            self.save("accepted_but_other_winner_exists", final_state=str(state))
            return 0
        winner = {
            "route": self.route,
            "created_utc": utc_now(),
            "target_kn_gu": TARGET_KN,
            "final_nodes": current_nodes,
            "final_state": str(state),
            "final_state_sha256": digest(state),
            "route_summary": str(self.summary_path),
        }
        write_atomic(self.winner_dir / "winner.json", winner)
        shutil.copy2(state, self.winner_dir / "R26_KnGu0p2_N40_state.npz")
        self.save("target_accepted", **winner)
        return 0


def parse_nodes(text: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in text.split(",") if value)
    if not values or tuple(sorted(set(values))) != values:
        raise argparse.ArgumentTypeError("refine nodes must be unique and increasing")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=tuple(ROUTES), required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--r13-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--winner-dir", type=Path, required=True)
    parser.add_argument("--refine-nodes", type=parse_nodes, default=(16, 20, 24, 28, 32, 36, 40))
    parser.add_argument("--reconcile-timeout", type=int, default=10800)
    parser.add_argument("--continuation-timeout", type=int, default=21600)
    parser.add_argument("--refinement-timeout", type=int, default=14400)
    args = parser.parse_args()
    return Controller(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
