"""Run the same cavity configuration with every pair-selection model."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsmc_cavity.collisions import SUPPORTED_MODELS
from dsmc_cavity.config import SimulationConfig
from dsmc_cavity.solver import CavitySolver


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("cpu", "gpu", "auto"), default="cpu")
    p.add_argument("--kn", type=float, default=0.1)
    p.add_argument("--nx", type=int, default=16)
    p.add_argument("--ppc", type=int, default=8)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260803)
    p.add_argument("--output", type=Path, default=ROOT / "results" / "model_comparison")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = SimulationConfig(
        backend=args.backend,
        kn=args.kn,
        nx=args.nx,
        ny=args.nx,
        particles_per_cell=args.ppc,
        steps=args.steps,
        warmup_steps=args.warmup,
        sample_stride=5,
        strict_probability=True,
    )
    reference_dt, _ = base.resolved_dt()
    target_end_time = args.steps * reference_dt
    target_warmup_time = args.warmup * reference_dt
    target_sample_interval = base.sample_stride * reference_dt
    profiles = {}
    rows = []
    for model in SUPPORTED_MODELS:
        probe = replace(base, model=model)
        model_dt, _ = probe.resolved_dt()
        model_warmup = max(1, math.ceil(target_warmup_time / model_dt))
        model_steps = max(model_warmup + 1, math.ceil(target_end_time / model_dt))
        model_stride = max(1, round(target_sample_interval / model_dt))
        config = replace(
            probe,
            steps=model_steps,
            warmup_steps=model_warmup,
            sample_stride=model_stride,
            output_dir=str(args.output / model),
            seed=args.seed,
        )
        result = CavitySolver(config).run(progress=False)
        profiles[model] = result
        rows.append(
            {
                "model": model,
                "dt_seconds": model_dt,
                "steps": model_steps,
                "warmup_steps": model_warmup,
                "end_time_seconds": model_steps * model_dt,
                "sample_interval_seconds": model_stride * model_dt,
                "runtime_seconds": result["metadata"]["runtime_seconds"],
                "selected": result["metadata"]["collision_statistics"]["selected"],
                "accepted": result["metadata"]["collision_statistics"]["accepted"],
                "max_probability": result["metadata"]["collision_statistics"]["max_probability"],
                "probability_exceedances": result["metadata"]["collision_statistics"]["probability_exceedances"],
            }
        )
        print(model, rows[-1])
    reference = profiles["ntc-prescan"]
    for row in rows:
        model = row["model"]
        current = profiles[model]
        row["lid_slip_rmse_vs_ntc_prescan"] = float(
            np.sqrt(np.mean((current["macro_slip"] - reference["macro_slip"]) ** 2))
        )
        u = current["fields"]["u"]
        u_ref = reference["fields"]["u"]
        row["u_relative_l2_vs_ntc_prescan"] = float(
            np.linalg.norm(u - u_ref) / max(np.linalg.norm(u_ref), 1e-30)
        )
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
