from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .closed_loop import run_vision_closed_loop
from .simulator import CavityConfig


def benchmark_cases() -> list[CavityConfig]:
    return [
        CavityConfig(nx=12, ny=12, particles_per_cell=20, steps=100, sample_start=50, t_left=1.10, t_right=0.90, collision_rate=0.16, seed=41),
        CavityConfig(nx=12, ny=12, particles_per_cell=20, steps=100, sample_start=50, t_left=1.15, t_right=0.85, collision_rate=0.18, seed=52),
        CavityConfig(nx=12, ny=12, particles_per_cell=20, steps=100, sample_start=50, t_left=1.20, t_right=0.80, collision_rate=0.20, seed=63),
    ]


def run_benchmark(output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, cfg in enumerate(benchmark_cases()):
        result = run_vision_closed_loop(
            cfg,
            reference_ppc=60,
            continuation_steps=150,
            vision_mode="temperature_gradient",
            budget_ratio=1.25,
            allocation_alpha=0.50,
            relaxation_fraction=2.0 / 3.0,
            output=output_dir / f"case_{index}.json",
        )
        rows.append(result)
    ratios = [row["mean_error_ratio"] for row in rows]
    summary = {
        "solver_status": "educational weighted DSMC-like pilot",
        "policy": "temperature-gradient image score",
        "budget_ratio": 1.25,
        "cases": len(rows),
        "improved_cases": int(sum(ratio < 1.0 for ratio in ratios)),
        "mean_error_ratio": float(np.mean(ratios)),
        "mean_error_reduction_percent": float(100.0 * (1.0 - np.mean(ratios))),
        "case_error_ratios": ratios,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage-3 vision-guided DSMC pilot benchmark")
    parser.add_argument("--output", default="outputs/stage3_benchmark")
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.output), indent=2))


if __name__ == "__main__":
    main()
