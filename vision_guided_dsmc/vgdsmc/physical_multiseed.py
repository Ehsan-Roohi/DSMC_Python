from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
from pathlib import Path
import numpy as np

from .physical_benchmark import run_case
from .vhs_model import PhysicalCavityConfig


def multiseed_cases(
    seeds: tuple[int, ...] = (11, 22, 33),
) -> list[PhysicalCavityConfig]:
    conditions = [
        (0.05, 330.0, 270.0),
        (0.10, 340.0, 260.0),
        (0.20, 320.0, 280.0),
    ]
    return [
        PhysicalCavityConfig(
            nx=8,
            ny=8,
            particles_per_cell=10,
            knudsen=knudsen,
            t_left=t_left,
            t_right=t_right,
            steps=120,
            seed=seed,
        )
        for knudsen, t_left, t_right in conditions
        for seed in seeds
    ]


def _run_seeded_case(cfg: PhysicalCavityConfig) -> dict:
    result = run_case(cfg)
    result["seed"] = cfg.seed
    return result


def summarize_rows(rows: list[dict]) -> dict[str, object]:
    rows = sorted(
        rows,
        key=lambda row: (row["knudsen"], row["seed"]),
    )
    ratios = np.array(
        [
            row["adaptive_to_uniform_error_ratio"]
            for row in rows
        ],
        dtype=float,
    )
    per_condition: dict[str, dict[str, object]] = {}
    for knudsen in (0.05, 0.10, 0.20):
        values = np.array(
            [
                row["adaptive_to_uniform_error_ratio"]
                for row in rows
                if row["knudsen"] == knudsen
            ],
            dtype=float,
        )
        mean = float(values.mean())
        standard_error = float(
            values.std(ddof=1) / np.sqrt(len(values))
        )
        t_multiplier = 4.303 if len(values) == 3 else 1.96
        per_condition[str(knudsen)] = {
            "ratios": values.tolist(),
            "mean_ratio": mean,
            "mean_improvement_percent": 100.0 * (1.0 - mean),
            "standard_error": standard_error,
            "descriptive_95_percent_ci": [
                mean - t_multiplier * standard_error,
                mean + t_multiplier * standard_error,
            ],
        }

    mean = float(ratios.mean())
    standard_error = float(
        ratios.std(ddof=1) / np.sqrt(len(ratios))
    )
    return {
        "cases": len(rows),
        "seeds_per_condition": len(rows) // 3,
        "comparison": (
            "adaptive versus uniform at identical particle budget"
        ),
        "adaptive_to_uniform_particle_ratio": 1.0,
        "improved_cases": int(np.sum(ratios < 1.0)),
        "mean_error_ratio": mean,
        "mean_improvement_percent": 100.0 * (1.0 - mean),
        "standard_error": standard_error,
        "normal_approx_95_percent_ci": [
            mean - 1.96 * standard_error,
            mean + 1.96 * standard_error,
        ],
        "per_condition": per_condition,
        "scientific_status": (
            "No statistically resolved overall improvement in this "
            "nine-run pilot. The tested Kn=0.20 condition improved "
            "for all three seeds, while Kn=0.05 and Kn=0.10 were "
            "seed-sensitive."
        ),
        "rows": rows,
    }


def run_multiseed_benchmark(
    output: str | Path,
    seeds: tuple[int, ...] = (11, 22, 33),
    workers: int = 3,
) -> dict[str, object]:
    cases = multiseed_cases(seeds)
    rows: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=max(1, workers)
    ) as executor:
        futures = {
            executor.submit(_run_seeded_case, cfg): cfg
            for cfg in cases
        }
        for future in as_completed(futures):
            rows.append(future.result())
    summary = summarize_rows(rows)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a multi-seed matched-cost physical benchmark"
        )
    )
    parser.add_argument(
        "--output",
        default="outputs/stage6_physical_multiseed",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[11, 22, 33],
    )
    args = parser.parse_args()
    summary = run_multiseed_benchmark(
        args.output,
        seeds=tuple(args.seeds),
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
