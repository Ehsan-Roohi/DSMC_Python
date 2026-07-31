from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np

from .lowfreq_closed_loop import (
    ClosedLoopConfig,
    _paired_statistics,
    run_closed_loop_benchmark,
)


CONFIRMATORY_SEEDS = (177, 188, 199, 211, 222, 233, 244, 255, 266, 277)
CONFIRMATORY_CONDITIONS = (
    (0.10, 20.0),
    (0.10, 40.0),
    (0.10, 60.0),
)


def _condition_summaries(rows: list[dict[str, object]]) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for delta_temperature in sorted({float(row["delta_temperature"]) for row in rows}):
        selected = [
            row
            for row in rows
            if float(row["delta_temperature"]) == delta_temperature
        ]
        ratios = np.asarray([float(row["error_ratio"]) for row in selected])
        statistics = _paired_statistics(ratios)
        output[f"deltaT_{delta_temperature:.0f}"] = {
            **statistics,
            "run_count": len(selected),
            "improved_runs": int(np.sum(ratios < 1.0)),
            "mean_improvement_percent": 100.0 * (1.0 - statistics["mean"]),
        }
    return output


def run_confirmatory_benchmark(
    model_path: str | Path,
    output_dir: str | Path,
    *,
    seeds: tuple[int, ...] = CONFIRMATORY_SEEDS,
    amplitude: float = 0.05,
    nx: int = 6,
    ny: int = 6,
    particles_per_cell: int = 20,
    warm_steps: int = 40,
    continuation_steps: int = 60,
    nv: int = 6,
    dvm_max_steps: int = 900,
) -> dict[str, object]:
    summary = run_closed_loop_benchmark(
        model_path,
        output_dir,
        ClosedLoopConfig(
            nx=nx,
            ny=ny,
            particles_per_cell=particles_per_cell,
            warm_steps=warm_steps,
            continuation_steps=continuation_steps,
            nv=nv,
            dvm_max_steps=dvm_max_steps,
            amplitudes=(amplitude,),
            seeds=seeds,
            conditions=CONFIRMATORY_CONDITIONS,
        ),
    )
    amplitude_key = f"{amplitude:.3f}"
    primary = summary["amplitude_summaries"][amplitude_key]
    condition_summaries = _condition_summaries(summary["rows"])
    improving_conditions = sum(
        float(metrics["mean"]) < 1.0
        for metrics in condition_summaries.values()
    )
    primary_success = (
        float(primary["ci95_high"]) < 1.0
        and improving_conditions >= 2
    )

    summary["stage"] = 19
    summary["description"] = (
        "Confirmatory five-percent closed loop at Kn=0.10 across three temperature differences"
    )
    summary["confirmatory_design"] = {
        "selection_basis": (
            "Kn=0.10 was the strongest exploratory condition in Stage 18; "
            "all Stage 19 seeds are new and were not used in model training, "
            "validation, Stage 17, or Stage 18 selection."
        ),
        "primary_endpoint": "mean paired adaptive-to-uniform field-error ratio",
        "success_rule": (
            "The upper 95% normal-approximation confidence bound must be below "
            "one and at least two of the three temperature-difference means must "
            "be below one."
        ),
        "amplitude": amplitude,
        "knudsen": 0.10,
        "temperature_differences": [20.0, 40.0, 60.0],
        "seeds": list(seeds),
    }
    summary["condition_summaries"] = condition_summaries
    summary["improving_condition_count"] = improving_conditions
    summary["primary_success"] = bool(primary_success)
    summary["scientific_guard"] = (
        "The result is reported according to the stored success rule; failure "
        "does not trigger amplitude, seed, or condition retuning in Stage 19."
    )

    output_dir = Path(output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 19 confirmatory matched-budget closed-loop evaluation"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="outputs/stage19_confirmatory")
    parser.add_argument("--amplitude", type=float, default=0.05)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(CONFIRMATORY_SEEDS))
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--ppc", type=int, default=20)
    parser.add_argument("--warm-steps", type=int, default=40)
    parser.add_argument("--continuation-steps", type=int, default=60)
    parser.add_argument("--nv", type=int, default=6)
    parser.add_argument("--dvm-max-steps", type=int, default=900)
    args = parser.parse_args()
    summary = run_confirmatory_benchmark(
        args.model,
        args.output_dir,
        seeds=tuple(args.seeds),
        amplitude=args.amplitude,
        nx=args.nx,
        ny=args.ny,
        particles_per_cell=args.ppc,
        warm_steps=args.warm_steps,
        continuation_steps=args.continuation_steps,
        nv=args.nv,
        dvm_max_steps=args.dvm_max_steps,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
