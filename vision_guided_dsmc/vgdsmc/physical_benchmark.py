from __future__ import annotations

from dataclasses import replace
import argparse
import json
from pathlib import Path

import numpy as np

from .physical_adaptive import adaptation_target, conservative_reallocate, field_error
from .sbt_solver import advance_physical_state, run_physical_cavity
from .vhs_model import PhysicalCavityConfig


def run_case(
    cfg: PhysicalCavityConfig,
    budget_ratio: float = 1.25,
) -> dict[str, float | bool]:
    warm_steps = max(40, cfg.steps // 2)
    continuation_steps = max(60, cfg.steps // 2)
    warm_cfg = replace(
        cfg,
        steps=warm_steps,
        sample_start=warm_steps // 2,
    )
    warm_fields, warm_state, warm_diagnostics = run_physical_cavity(
        warm_cfg,
        return_state=True,
    )

    target, decision = adaptation_target(warm_fields, cfg, budget_ratio)
    adaptive_state = (
        conservative_reallocate(warm_state, cfg, target, seed=cfg.seed + 700)
        if decision["adapted"]
        else warm_state.copy()
    )

    uniform_fields, _, uniform_diagnostics = advance_physical_state(
        warm_state.copy(),
        cfg,
        continuation_steps,
        continuation_steps // 2,
        seed=cfg.seed + 800,
    )
    adaptive_fields, _, adaptive_diagnostics = advance_physical_state(
        adaptive_state,
        cfg,
        continuation_steps,
        continuation_steps // 2,
        seed=cfg.seed + 800,
    )

    reference_cfg = replace(
        cfg,
        particles_per_cell=cfg.particles_per_cell * 4,
        steps=warm_steps + continuation_steps,
        sample_start=warm_steps,
    )
    reference = run_physical_cavity(reference_cfg)
    uniform_error = field_error(uniform_fields, reference)
    adaptive_error = field_error(adaptive_fields, reference)
    return {
        "knudsen": cfg.knudsen,
        "temperature_ratio": cfg.t_left / cfg.t_right,
        "uniform_error": uniform_error,
        "adaptive_error": adaptive_error,
        "error_ratio": adaptive_error / uniform_error,
        "improvement_percent": 100.0 * (1.0 - adaptive_error / uniform_error),
        "particle_ratio": float(
            target.sum() / (target.size * cfg.particles_per_cell)
        ),
        "adapted": bool(decision["adapted"]),
        "relative_temperature_noise": float(
            decision["relative_temperature_noise"]
        ),
        "uniform_collisions": uniform_diagnostics["accepted_collisions"],
        "adaptive_collisions": adaptive_diagnostics["accepted_collisions"],
        "warm_collisions": warm_diagnostics["accepted_collisions"],
        "dt": cfg.dt,
    }


def run_benchmark(output: str | Path) -> dict[str, object]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cases = [
        PhysicalCavityConfig(
            nx=8,
            ny=8,
            particles_per_cell=10,
            knudsen=0.05,
            t_left=330.0,
            t_right=270.0,
            steps=120,
            seed=11,
        ),
        PhysicalCavityConfig(
            nx=8,
            ny=8,
            particles_per_cell=10,
            knudsen=0.10,
            t_left=340.0,
            t_right=260.0,
            steps=120,
            seed=22,
        ),
        PhysicalCavityConfig(
            nx=8,
            ny=8,
            particles_per_cell=10,
            knudsen=0.20,
            t_left=320.0,
            t_right=280.0,
            steps=120,
            seed=33,
        ),
    ]
    rows = [run_case(case) for case in cases]
    ratios = np.array([row["error_ratio"] for row in rows])
    summary = {
        "kernel": "Argon VHS + SBT/TAS adaptive subcells",
        "source_kernel": "SBT/TAS probability adapted from Parallel_TAS.py",
        "vhs_normalization": (
            "Corrected Bird form with reduced mass and Gamma(5/2-omega)"
        ),
        "cases": rows,
        "improved_cases": int(np.sum(ratios < 1.0 - 1.0e-12)),
        "non_worse_cases": int(np.sum(ratios <= 1.0 + 1.0e-12)),
        "mean_error_ratio": float(ratios.mean()),
        "mean_improvement_percent": float(100.0 * (1.0 - ratios.mean())),
        "mean_particle_ratio": float(
            np.mean([row["particle_ratio"] for row in rows])
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/stage4_physical")
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.output), indent=2))


if __name__ == "__main__":
    main()
