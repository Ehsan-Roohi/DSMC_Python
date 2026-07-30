from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
import argparse
import json
import math
import numpy as np

from .physical_adaptive import (
    conservative_reallocate,
    exact_budget_ppc,
    field_error,
    uniform_exact_budget_ppc,
)
from .physical_paired_ensemble import build_reference
from .sbt_solver import advance_physical_state, run_physical_cavity
from .vhs_model import PhysicalCavityConfig


def _smooth3(field: np.ndarray) -> np.ndarray:
    padded = np.pad(field, 1, mode="edge")
    result = np.zeros_like(field, dtype=float)
    for j in range(3):
        for i in range(3):
            result += padded[
                j : j + field.shape[0],
                i : i + field.shape[1],
            ]
    return result / 9.0


def _robust_scale(field: np.ndarray) -> np.ndarray:
    low, high = np.percentile(field, [5.0, 95.0])
    return np.clip(
        (field - low) / max(float(high - low), 1.0e-30),
        0.0,
        1.0,
    )


def priority_candidates(
    fields: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return reference-free candidate priority images for development."""
    temperature = _smooth3(fields["T"])
    density = _smooth3(fields["rho"])
    noise = _smooth3(fields["sigma_T"])

    gy_t, gx_t = np.gradient(temperature)
    grad_t = np.hypot(gx_t, gy_t)
    gy_rho, gx_rho = np.gradient(density)
    grad_rho = np.hypot(gx_rho, gy_rho)
    laplacian_t = np.abs(
        np.gradient(np.gradient(temperature, axis=0), axis=0)
        + np.gradient(np.gradient(temperature, axis=1), axis=1)
    )

    ny, nx = temperature.shape
    x_index = np.arange(nx)
    side_distance = np.minimum(x_index, nx - 1 - x_index)
    sidewall = (
        np.exp(-side_distance / max(1.0, nx / 6.0))[None, :]
        * np.ones((ny, 1))
    )

    scaled_grad_t = _robust_scale(grad_t)
    scaled_grad_rho = _robust_scale(grad_rho)
    scaled_noise = _robust_scale(noise)
    scaled_curvature = _robust_scale(laplacian_t)
    scaled_sidewall = _robust_scale(sidewall)
    snr_grad_t = _robust_scale(
        grad_t
        / (
            noise
            + float(np.percentile(noise, 25.0))
            + 1.0e-12
        )
    )

    return {
        "current": (
            0.65 * scaled_grad_t
            + 0.20 * scaled_grad_rho
            + 0.15 * scaled_noise
        ),
        "grad_t": scaled_grad_t,
        "grad_rho": scaled_grad_rho,
        "noise": scaled_noise,
        "grad_t_noise": (
            0.60 * scaled_grad_t + 0.40 * scaled_noise
        ),
        "snr_grad_t": snr_grad_t,
        "curvature_t": scaled_curvature,
        "sidewall": scaled_sidewall,
        "sidewall_grad_t": (
            0.50 * scaled_sidewall + 0.50 * scaled_grad_t
        ),
    }


def run_policy_cluster(
    base_seed: int,
    reference: dict[str, np.ndarray],
    policy_names: tuple[str, ...],
    repetitions: int,
    continuation_seed_offset: int,
    budget_ratio: float = 1.25,
) -> dict[str, object]:
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
    warm_cfg = replace(cfg, steps=60, sample_start=30)
    warm_fields, warm_state, _ = run_physical_cavity(
        warm_cfg,
        return_state=True,
    )
    candidate_maps = priority_candidates(warm_fields)
    missing = set(policy_names) - set(candidate_maps)
    if missing:
        raise ValueError(f"unknown priority policies: {sorted(missing)}")

    targets = {
        name: exact_budget_ppc(
            candidate_maps[name],
            cfg.particles_per_cell,
            budget_ratio,
            min_ppc=8,
            max_ppc=20,
        )
        for name in policy_names
    }
    target_particles = int(next(iter(targets.values())).sum())
    if any(int(target.sum()) != target_particles for target in targets.values()):
        raise RuntimeError("candidate policies do not share an exact budget")

    uniform_target = uniform_exact_budget_ppc(
        (cfg.ny, cfg.nx),
        cfg.particles_per_cell,
        target_particles
        / (cfg.nx * cfg.ny * cfg.particles_per_cell),
    )
    states = {
        "uniform": conservative_reallocate(
            warm_state,
            cfg,
            uniform_target,
            seed=base_seed + 700,
        )
    }
    for name, target in targets.items():
        states[name] = conservative_reallocate(
            warm_state,
            cfg,
            target,
            seed=base_seed + 700,
        )

    errors = {name: [] for name in states}
    for repetition in range(repetitions):
        continuation_seed = (
            continuation_seed_offset + 100 * base_seed + repetition
        )
        for name, state in states.items():
            fields, _, _ = advance_physical_state(
                state.copy(),
                cfg,
                60,
                30,
                seed=continuation_seed,
            )
            errors[name].append(field_error(fields, reference))

    uniform_mean = float(np.mean(errors["uniform"]))
    return {
        "base_seed": base_seed,
        "uniform_error": uniform_mean,
        "target_particles": target_particles,
        "policies": {
            name: {
                "mean_error": float(np.mean(errors[name])),
                "ratio": float(np.mean(errors[name]) / uniform_mean),
                "replicate_ratios": (
                    np.asarray(errors[name])
                    / np.asarray(errors["uniform"])
                ).tolist(),
            }
            for name in policy_names
        },
    }


def _sign_test_p(improved: int, total: int) -> float:
    tail = min(improved, total - improved)
    probability = (
        sum(math.comb(total, index) for index in range(tail + 1))
        / 2**total
    )
    return float(min(1.0, 2.0 * probability))


def summarize_policy(
    ratios: list[float],
    bootstrap_seed: int = 99,
) -> dict[str, object]:
    values = np.asarray(ratios, dtype=float)
    count = len(values)
    if count < 2:
        raise ValueError("at least two independent seeds are required")
    t_multiplier = 2.262 if count == 10 else 2.776 if count == 5 else 1.96
    standard_error = float(values.std(ddof=1) / np.sqrt(count))
    improved = int(np.sum(values < 1.0))
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, count, size=(20_000, count))
    bootstrap_means = values[indices].mean(axis=1)
    return {
        "ratios": values.tolist(),
        "mean_ratio": float(values.mean()),
        "median_ratio": float(np.median(values)),
        "mean_improvement_percent": float(
            100.0 * (1.0 - values.mean())
        ),
        "improved_seeds": improved,
        "standard_error": standard_error,
        "t_95_ci": [
            float(values.mean() - t_multiplier * standard_error),
            float(values.mean() + t_multiplier * standard_error),
        ],
        "bootstrap_95_ci": np.percentile(
            bootstrap_means,
            [2.5, 97.5],
        ).tolist(),
        "two_sided_sign_test_p": _sign_test_p(improved, count),
        "worst_ratio": float(values.max()),
    }


def _run_phase(
    seeds: tuple[int, ...],
    reference_seeds: tuple[int, ...],
    policy_names: tuple[str, ...],
    repetitions: int,
    workers: int,
    continuation_seed_offset: int,
) -> dict[str, object]:
    cfg = PhysicalCavityConfig(
        nx=8,
        ny=8,
        particles_per_cell=10,
        knudsen=0.20,
        t_left=320.0,
        t_right=280.0,
        steps=120,
        seed=seeds[0],
    )
    reference = build_reference(
        cfg,
        reference_seeds,
        workers=min(workers, len(reference_seeds)),
    )
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(
                run_policy_cluster,
                seed,
                reference,
                policy_names,
                repetitions,
                continuation_seed_offset,
            )
            for seed in seeds
        ]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: int(row["base_seed"]))
    return {
        "seeds": list(seeds),
        "reference_seeds": list(reference_seeds),
        "repetitions": repetitions,
        "statistical_unit": "independent warm seed",
        "policies": {
            name: summarize_policy(
                [float(row["policies"][name]["ratio"]) for row in rows]
            )
            for name in policy_names
        },
        "rows": rows,
    }


def run_policy_study(
    output: str | Path,
    workers: int = 5,
) -> dict[str, object]:
    all_policies = tuple(
        priority_candidates(
            {
                "T": np.arange(64, dtype=float).reshape(8, 8),
                "rho": np.ones((8, 8)),
                "sigma_T": np.ones((8, 8)),
            }
        )
    )
    development = _run_phase(
        seeds=(401, 402, 403, 404, 405),
        reference_seeds=(9101, 9102),
        policy_names=all_policies,
        repetitions=2,
        workers=workers,
        continuation_seed_offset=200_000,
    )
    ranking = sorted(
        development["policies"],
        key=lambda name: development["policies"][name]["mean_ratio"],
    )
    finalists = tuple(
        dict.fromkeys(("current", *ranking[:3]))
    )
    validation = _run_phase(
        seeds=tuple(range(501, 511)),
        reference_seeds=(9201, 9202, 9203, 9204),
        policy_names=finalists,
        repetitions=3,
        workers=workers,
        continuation_seed_offset=300_000,
    )
    result = {
        "design": (
            "policy selection on five development warm seeds; final evaluation "
            "on ten disjoint validation warm seeds with four reference runs"
        ),
        "condition": {
            "knudsen": 0.20,
            "t_left": 320.0,
            "t_right": 280.0,
        },
        "development": development,
        "selected_finalists": list(finalists),
        "validation": validation,
        "scientific_status": (
            "No tested reference-free policy achieved a statistically resolved "
            "held-out improvement over equal-budget uniform allocation."
        ),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Develop and validate physical priority images"
    )
    parser.add_argument(
        "--output",
        default="outputs/stage8_policy_study",
    )
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            run_policy_study(args.output, workers=args.workers),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
