from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
import argparse
import json
import math
import numpy as np

from .physical_adaptive import (
    adaptation_target,
    conservative_reallocate,
    field_error,
    uniform_exact_budget_ppc,
)
from .sbt_solver import advance_physical_state, run_physical_cavity
from .vhs_model import PhysicalCavityConfig


def _mean_fields(
    fields: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    common = set.intersection(*(set(item) for item in fields))
    return {
        key: np.mean([item[key] for item in fields], axis=0)
        for key in common
    }


def _run_reference(
    cfg: PhysicalCavityConfig,
) -> dict[str, np.ndarray]:
    return run_physical_cavity(cfg)


def build_reference(
    cfg: PhysicalCavityConfig,
    reference_seeds: tuple[int, ...],
    workers: int = 4,
) -> dict[str, np.ndarray]:
    reference_cfgs = [
        replace(
            cfg,
            particles_per_cell=cfg.particles_per_cell * 4,
            steps=cfg.steps,
            sample_start=cfg.steps // 2,
            seed=seed,
        )
        for seed in reference_seeds
    ]
    with ProcessPoolExecutor(
        max_workers=min(max(1, workers), len(reference_cfgs))
    ) as executor:
        fields = list(executor.map(_run_reference, reference_cfgs))
    return _mean_fields(fields)


def run_cluster(
    base_seed: int,
    reference: dict[str, np.ndarray],
    repetitions: int = 3,
    budget_ratio: float = 1.25,
) -> dict[str, object]:
    """Run one independent warm state with paired continuation replicas."""
    cfg = PhysicalCavityConfig(
        nx=8,
        ny=8,
        particles_per_cell=10,
        knudsen=0.20,
        t_left=320.0,
        t_right=280.0,
        steps=120,
        seed=base_seed,
    )
    warm_steps = cfg.steps // 2
    continuation_steps = cfg.steps // 2
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
    target_particles = int(adaptive_target.sum())
    uniform_target = uniform_exact_budget_ppc(
        (cfg.ny, cfg.nx),
        cfg.particles_per_cell,
        target_particles
        / (cfg.nx * cfg.ny * cfg.particles_per_cell),
    )
    if int(uniform_target.sum()) != target_particles:
        raise RuntimeError("paired control particle budget mismatch")

    uniform_state = conservative_reallocate(
        warm_state,
        cfg,
        uniform_target,
        seed=base_seed + 700,
    )
    adaptive_state = conservative_reallocate(
        warm_state,
        cfg,
        adaptive_target,
        seed=base_seed + 700,
    )

    pairs: list[dict[str, float | int]] = []
    for repetition in range(repetitions):
        continuation_seed = 100_000 + 100 * base_seed + repetition
        uniform_fields, _, uniform_diagnostics = advance_physical_state(
            uniform_state.copy(),
            cfg,
            continuation_steps,
            continuation_steps // 2,
            seed=continuation_seed,
        )
        adaptive_fields, _, adaptive_diagnostics = advance_physical_state(
            adaptive_state.copy(),
            cfg,
            continuation_steps,
            continuation_steps // 2,
            seed=continuation_seed,
        )
        uniform_error = field_error(uniform_fields, reference)
        adaptive_error = field_error(adaptive_fields, reference)
        pairs.append(
            {
                "repetition": repetition,
                "uniform_error": float(uniform_error),
                "adaptive_error": float(adaptive_error),
                "ratio": float(adaptive_error / uniform_error),
                "difference": float(adaptive_error - uniform_error),
                "uniform_collisions": float(
                    uniform_diagnostics["accepted_collisions"]
                ),
                "adaptive_collisions": float(
                    adaptive_diagnostics["accepted_collisions"]
                ),
            }
        )

    uniform_mean = float(
        np.mean([pair["uniform_error"] for pair in pairs])
    )
    adaptive_mean = float(
        np.mean([pair["adaptive_error"] for pair in pairs])
    )
    return {
        "base_seed": base_seed,
        "adapted": bool(decision["adapted"]),
        "relative_temperature_noise": float(
            decision["relative_temperature_noise"]
        ),
        "priority_std": float(decision["priority_std"]),
        "target_particles": target_particles,
        "warm_collisions": float(
            warm_diagnostics["accepted_collisions"]
        ),
        "pairs": pairs,
        "mean_uniform_error": uniform_mean,
        "mean_adaptive_error": adaptive_mean,
        "cluster_ratio": float(adaptive_mean / uniform_mean),
        "cluster_difference": float(adaptive_mean - uniform_mean),
    }


def _two_sided_sign_p(improved: int, total: int) -> float:
    tail = min(improved, total - improved)
    probability = (
        sum(math.comb(total, k) for k in range(tail + 1))
        / 2**total
    )
    return float(min(1.0, 2.0 * probability))


def summarize_clusters(
    clusters: list[dict[str, object]],
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 17,
) -> dict[str, object]:
    """Summarize at the independent warm-seed level, not the replica level."""
    clusters = sorted(
        clusters,
        key=lambda item: int(item["base_seed"]),
    )
    ratios = np.array(
        [float(item["cluster_ratio"]) for item in clusters]
    )
    differences = np.array(
        [float(item["cluster_difference"]) for item in clusters]
    )
    count = len(clusters)
    if count < 2:
        raise ValueError(
            "at least two independent warm seeds are required"
        )
    t_multiplier = {
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
    }.get(count, 1.96)
    ratio_se = float(ratios.std(ddof=1) / np.sqrt(count))
    difference_se = float(
        differences.std(ddof=1) / np.sqrt(count)
    )

    rng = np.random.default_rng(bootstrap_seed)
    samples = rng.integers(
        0,
        count,
        size=(bootstrap_samples, count),
    )
    bootstrap_ratio_means = ratios[samples].mean(axis=1)
    bootstrap_difference_means = differences[samples].mean(axis=1)
    improved = int(np.sum(ratios < 1.0))

    return {
        "independent_warm_seeds": count,
        "paired_continuations_per_seed": len(clusters[0]["pairs"]),
        "paired_comparisons": int(
            sum(len(item["pairs"]) for item in clusters)
        ),
        "adaptive_to_uniform_particle_ratio": 1.0,
        "improved_warm_seeds": improved,
        "cluster_ratios": ratios.tolist(),
        "mean_cluster_ratio": float(ratios.mean()),
        "median_cluster_ratio": float(np.median(ratios)),
        "mean_improvement_percent": float(
            100.0 * (1.0 - ratios.mean())
        ),
        "ratio_standard_error": ratio_se,
        "t_95_ci_ratio": [
            float(ratios.mean() - t_multiplier * ratio_se),
            float(ratios.mean() + t_multiplier * ratio_se),
        ],
        "cluster_bootstrap_95_ci_ratio": np.percentile(
            bootstrap_ratio_means,
            [2.5, 97.5],
        ).tolist(),
        "mean_paired_error_difference": float(differences.mean()),
        "difference_standard_error": difference_se,
        "t_95_ci_difference": [
            float(
                differences.mean() - t_multiplier * difference_se
            ),
            float(
                differences.mean() + t_multiplier * difference_se
            ),
        ],
        "cluster_bootstrap_95_ci_difference": np.percentile(
            bootstrap_difference_means,
            [2.5, 97.5],
        ).tolist(),
        "two_sided_sign_test_p": _two_sided_sign_p(
            improved,
            count,
        ),
        "clusters": clusters,
    }


def run_paired_ensemble(
    output: str | Path,
    base_seeds: tuple[int, ...] = tuple(range(301, 311)),
    reference_seeds: tuple[int, ...] = (9001, 9002, 9003, 9004),
    repetitions: int = 3,
    workers: int = 5,
) -> dict[str, object]:
    cfg = PhysicalCavityConfig(
        nx=8,
        ny=8,
        particles_per_cell=10,
        knudsen=0.20,
        t_left=320.0,
        t_right=280.0,
        steps=120,
        seed=base_seeds[0],
    )
    reference = build_reference(
        cfg,
        reference_seeds,
        workers=min(workers, len(reference_seeds)),
    )
    clusters: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=max(1, workers)
    ) as executor:
        futures = {
            executor.submit(
                run_cluster,
                seed,
                reference,
                repetitions,
            ): seed
            for seed in base_seeds
        }
        for future in as_completed(futures):
            clusters.append(future.result())

    summary = summarize_clusters(clusters)
    summary.update(
        {
            "method": (
                "four-reference ensemble; ten independent warm seeds; "
                "three common-random-number continuation pairs per seed"
            ),
            "condition": {
                "knudsen": 0.20,
                "t_left": 320.0,
                "t_right": 280.0,
            },
            "base_seeds": list(base_seeds),
            "reference_seeds": list(reference_seeds),
            "statistical_unit": "independent warm seed (cluster)",
        }
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the clustered paired Kn=0.20 ensemble"
    )
    parser.add_argument(
        "--output",
        default="outputs/stage7_kn02_paired_ensemble",
    )
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--base-seeds",
        type=int,
        nargs="+",
        default=list(range(301, 311)),
    )
    parser.add_argument(
        "--reference-seeds",
        type=int,
        nargs="+",
        default=[9001, 9002, 9003, 9004],
    )
    args = parser.parse_args()
    result = run_paired_ensemble(
        args.output,
        base_seeds=tuple(args.base_seeds),
        reference_seeds=tuple(args.reference_seeds),
        repetitions=args.repetitions,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
