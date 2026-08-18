#!/usr/bin/env python3
"""Run the portable validation suite for the publication release."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, cwd: Path | None = None) -> None:
    display = " ".join(args)
    print(f"\n$ {display}", flush=True)
    environment = dict(os.environ)
    environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    subprocess.run(args, cwd=cwd or ROOT, env=environment, check=True)


def check_reduced_tables() -> None:
    required_rows = {
        "dsmc_sensitivity_figure_data.csv": 5,
    }
    for name, expected in required_rows.items():
        path = ROOT / "data" / "reduced" / name
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != expected:
            raise RuntimeError(f"{path}: expected {expected} rows, found {len(rows)}")
    print("REDUCED_TABLE_VALIDATION_PASS")


def check_pure_maxwell_products() -> None:
    data_dir = ROOT / "data" / "pure_maxwell"
    required_rows = {
        "field_metrics.csv": 4,
        "anti_fourier_metrics.csv": 4,
        "processing_sensitivity.csv": 80,
        "centerline_profiles.csv": 960,
    }
    for name, expected in required_rows.items():
        with (data_dir / name).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != expected:
            raise RuntimeError(f"{name}: expected {expected} rows, found {len(rows)}")
    for line in (data_dir / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        path = data_dir / name.strip()
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(f"pure-Maxwell checksum mismatch: {path}")
    figures = ROOT / "figures" / "pure_maxwell"
    stems = (
        "primary_k20",
        "centerlines_k20",
        "antifourier_k20",
        "antifourier_k05",
        "antifourier_atlas",
        "model_metrics",
    )
    for stem in stems:
        for suffix in (".pdf", ".png"):
            path = figures / f"{stem}{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"missing pure-Maxwell figure: {path}")
    print("PURE_MAXWELL_PRODUCTS_VALIDATION_PASS")


def check_release_hygiene() -> None:
    forbidden_names = {
        "rana_original_coefficients.py",
        "rana_original_reference_solver.py",
    }
    forbidden_suffixes = {".pyc", ".pyo"}
    for path in ROOT.rglob("*"):
        if path.name in forbidden_names:
            raise RuntimeError(f"excluded source present: {path}")
        if path.is_file() and path.suffix in forbidden_suffixes:
            raise RuntimeError(f"bytecode cache present: {path}")
        if path.is_dir() and path.name in {"__pycache__", ".pytest_cache"}:
            raise RuntimeError(f"cache directory present: {path}")
    analyzer = (ROOT / "analysis" / "pure_maxwell" / "analyze_pure_maxwell.py").read_text(
        encoding="utf-8"
    )
    for token in ("rana_original_reference_solver", "import_r13_solver"):
        if token in analyzer:
            raise RuntimeError(f"public analyzer retains excluded dependency: {token}")
    print("PUBLIC_RELEASE_HYGIENE_PASS")


def main() -> int:
    py = sys.executable
    run(
        py,
        "dsmc/scripts/validate_jfm_maxwell_kngu_case.py",
        "data/dsmc/kn005",
        "--kn-gu",
        "0.05",
        "--require-final",
    )
    run(
        py,
        "dsmc/scripts/validate_jfm_maxwell_kngu_case.py",
        "data/dsmc/kn020",
        "--kn-gu",
        "0.20",
        "--require-final",
    )
    run(py, "r13/tests/test_r13_equation_contract.py")
    run(
        py,
        "r13/tests/validate_r13_maxwell_run.py",
        "data/r13/kn005_N60",
        "--expected-kn-rana",
        "0.039894228040143274",
        "--expected-nodes",
        "60",
    )
    run(
        py,
        "r13/tests/validate_r13_maxwell_run.py",
        "data/r13/kn020_N60",
        "--expected-kn-rana",
        "0.1595769121605731",
        "--expected-nodes",
        "60",
    )
    run(py, "run_tests.py", cwd=ROOT / "r26" / "source" / "r26" / "code")
    run(py, "r26/tests/test_maxwell_contract.py")
    run(
        py,
        "r26/tests/validate_r26_maxwell_run.py",
        "data/r26/kn005_N40",
        "--expected-kn",
        "0.05",
        "--expected-nodes",
        "40",
    )
    run(
        py,
        "r26/tests/validate_r26_maxwell_run.py",
        "data/r26/kn020_N20",
        "--expected-kn",
        "0.20",
        "--expected-nodes",
        "20",
    )
    run(py, "analysis/pure_maxwell/analyze_pure_maxwell.py")
    check_pure_maxwell_products()
    check_reduced_tables()
    check_release_hygiene()
    print("\nPUBLIC_RELEASE_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
