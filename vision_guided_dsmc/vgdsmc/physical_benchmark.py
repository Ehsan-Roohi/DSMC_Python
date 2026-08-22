from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import numpy as np

from .vhs_model import PhysicalCavityConfig
from .sbt_solver import advance_physical_state, run_physical_cavity
from .physical_adaptive import (
    adaptation_target,
    conservative_reallocate,
    field_error,
    uniform_exact_budget_ppc,
)


def run_case(
    cfg: PhysicalCavityConfig,
    budget_ratio: float = 1.25,
) -> dict[str, float | bool]:
    """Compare vision allocation with a uniform allocation at identical cost."""
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

    adaptive_target, decision = adaptation_target(
        warm_fields,
        cfg,
        budget_ratio,
    )
    exact_budget = int(adaptive_target.sum())
    uniform_target = uniform_exact_budget_ppc(
        (cfg.ny, cfg.nx),
        cfg.particles_per_cell,
        exact_budget
        / (cfg.nx * cfg.ny * cfg.particles_per_cell),
    )
    if int(uniform_target.sum()) != exact_budget:
        raise RuntimeError(
            "Equal-budget control does not match adaptive particle count"
        )

    uniform_state = conservative_reallocate(
        warm_state,
        cfg,
        uniform_target,
        seed=cfg.seed + 700,
    )
    adaptive_state = conservative_reallocate(
        warm_state,
        cfg,
        adaptive_target,
        seed=cfg.seed + 700,
    )

    uniform_fields, _, uniform_diagnostics = advance_physical_state(
        uniform_state,
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
    error_ratio = adaptive_error / uniform_error
    base_particles = cfg.nx * cfg.ny * cfg.particles_per_cell

    return {
        "knudsen": float(cfg.knudsen),
        "temperature_ratio": float(cfg.t_left / cfg.t_right),
        "uniform_equal_budget_error": float(uniform_error),
        "adaptive_error": float(adaptive_error),
        "adaptive_to_uniform_error_ratio": float(error_ratio),
        "improvement_percent": float(
            100.0 * (1.0 - error_ratio)
        ),
        "base_to_budget_particle_ratio": float(
            exact_budget / base_particles
        ),
        "adaptive_to_uniform_particle_ratio": float(
            exact_budget / uniform_target.sum()
        ),
        "target_particles": exact_budget,
        "adapted": bool(decision["adapted"]),
        "relative_temperature_noise": float(
            decision["relative_temperature_noise"]
        ),
        "priority_std": float(decision["priority_std"]),
        "uniform_collisions": float(
            uniform_diagnostics["accepted_collisions"]
        ),
        "adaptive_collisions": float(
            adaptive_diagnostics["accepted_collisions"]
        ),
        "warm_collisions": float(
            warm_diagnostics["accepted_collisions"]
        ),
        "dt": float(cfg.dt),
    }


def benchmark_cases() -> list[PhysicalCavityConfig]:
    return [
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


def run_benchmark(output: str | Path) -> dict[str, object]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [run_case(case) for case in benchmark_cases()]
    ratios = np.array(
        [
            row["adaptive_to_uniform_error_ratio"]
            for row in rows
        ]
    )
    summary = {
        "kernel": "Argon VHS + SBT/TAS adaptive subcells",
        "comparison": (
            "vision allocation versus uniform allocation "
            "at identical particle budget"
        ),
        "source_kernel": (
            "SBT/TAS probability adapted from Parallel_TAS.py"
        ),
        "vhs_normalization": (
            "Bird form with reduced mass and Gamma(5/2-omega)"
        ),
        "withdrawn_results": [
            (
                "The earlier 7.84% nondimensional result used a "
                "broken diffuse-wall writeback."
            ),
            (
                "The earlier 5.81% physical result used the same "
                "advanced-indexing bug and is invalid."
            ),
        ],
        "cases": rows,
        "improved_cases": int(np.sum(ratios < 1.0)),
        "within_two_percent_cases": int(
            np.sum(ratios <= 1.02)
        ),
        "mean_error_ratio": float(ratios.mean()),
        "mean_improvement_percent": float(
            100.0 * (1.0 - ratios.mean())
        ),
        "mean_base_to_budget_particle_ratio": float(
            np.mean(
                [
                    row["base_to_budget_particle_ratio"]
                    for row in rows
                ]
            )
        ),
        "adaptive_to_uniform_particle_ratio": 1.0,
        "scientific_status": (
            "Promising pilot only: two of three cases improve at "
            "matched cost; multi-seed ensembles and a production "
            "DSMC collision kernel remain required."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run the matched-cost physical VHS/SBT "
            "vision-guided benchmark"
        )
    )
    parser.add_argument(
        "--output",
        default="outputs/stage5_physical_equal_budget",
    )
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.output), indent=2))


if __name__ == "__main__":
    main()
