#!/usr/bin/env python3
"""Validate one completed numerical Maxwell R13 target."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--expected-kn-rana", type=float, required=True)
    parser.add_argument("--expected-nodes", type=int, required=True)
    args = parser.parse_args()
    report_path = args.result / "report.json"
    state_path = args.result / "state.npy"
    if not report_path.is_file() or not state_path.is_file():
        raise SystemExit("R13_MAXWELL_OUTPUT_MISSING")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = report.get("configuration", {})
    solver = report.get("solver", {})
    provenance = report.get("execution_provenance", {})
    checks = {
        "collision_model": report.get("collision_model") == "Maxwell molecules",
        "viscosity_law": report.get("viscosity_law") == "mu/mu0 = theta/theta0",
        "production_prefactor": report.get("production_prefactor") == "rho/Kn_Rana",
        "kn": math.isclose(float(config.get("kn", -1.0)), args.expected_kn_rana, rel_tol=0.0, abs_tol=2.0e-15),
        "nodes": config.get("nx") == args.expected_nodes and config.get("ny") == args.expected_nodes,
        "converged": solver.get("converged") is True,
        "residual": float(solver.get("core_relative_residual_without_mass_border", 1.0)) <= 1.0e-8,
        "rho": float(solver.get("rho_min", -1.0)) > 0.0,
        "theta": float(solver.get("theta_min", -1.0)) > 0.0,
        "mass": abs(float(solver.get("mass_error", 1.0))) <= 1.0e-8,
        "source_stable": provenance.get("source_unchanged_during_execution") is True,
        "gates": report.get("passed_private_run_gates") is True and report.get("metrics_valid") is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("R13_MAXWELL_VALIDATION_FAILED: " + ",".join(failed))
    print("R13_MAXWELL_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
