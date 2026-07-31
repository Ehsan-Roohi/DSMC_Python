from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import argparse
import json
import math
import numpy as np

from .dvm_shakhov_corrected import (
    ShakhovReferenceConfig,
    solve_shakhov_reference,
)
from .ensemble_score_regression import _augment_features
from .physical_adaptive import (
    conservative_reallocate,
    field_error,
    uniform_exact_budget_ppc,
)
from .sbt_solver import advance_physical_state, run_physical_cavity
from .vhs_model import PhysicalCavityConfig


@dataclass(frozen=True)
class ClosedLoopConfig:
    nx: int = 6
    ny: int = 6
    particles_per_cell: int = 20
    warm_steps: int = 40
    continuation_steps: int = 60
    nv: int = 6
    dvm_max_steps: int = 900
    amplitudes: tuple[float, ...] = (0.05, 0.10, 0.20)
    seeds: tuple[int, ...] = (66, 77, 88)
    conditions: tuple[tuple[float, float], ...] = (
        (0.05, 40.0),
        (0.10, 40.0),
        (0.20, 40.0),
    )


def exact_low_amplitude_ppc(
    priority: np.ndarray,
    base_ppc: int,
    amplitude: float,
) -> np.ndarray:
    """Convert a rank image to a bounded exact-budget integer PPC map.

    The continuous proposal is ``base_ppc * (1 + amplitude * centered_rank)``.
    Centering guarantees the unrounded total equals the uniform budget.  A
    largest-remainder correction then enforces the exact integer total without
    violating the requested local amplitude bounds.
    """
    priority = np.asarray(priority, dtype=np.float64)
    if priority.ndim != 2 or priority.size == 0:
        raise ValueError("priority must be a non-empty two-dimensional array")
    if not np.isfinite(priority).all():
        raise ValueError("priority must be finite")
    if base_ppc < 2:
        raise ValueError("base_ppc must be at least two")
    if not 0.0 <= amplitude < 1.0:
        raise ValueError("amplitude must satisfy 0 <= amplitude < 1")

    target_total = int(priority.size * base_ppc)
    if amplitude == 0.0 or float(np.ptp(priority)) <= 1.0e-14:
        return np.full(priority.shape, base_ppc, dtype=np.int64)

    centered = priority - float(np.mean(priority))
    scale = max(float(np.max(np.abs(centered))), 1.0e-14)
    centered /= scale
    raw = base_ppc * (1.0 + amplitude * centered)

    minimum = max(2, int(math.floor(base_ppc * (1.0 - amplitude))))
    maximum = max(minimum, int(math.ceil(base_ppc * (1.0 + amplitude))))
    raw = np.clip(raw, minimum, maximum)
    flat_raw = raw.ravel()
    allocation = np.floor(flat_raw).astype(np.int64)
    allocation = np.clip(allocation, minimum, maximum)
    difference = target_total - int(allocation.sum())
    residual = flat_raw - np.floor(flat_raw)

    while difference > 0:
        eligible = np.flatnonzero(allocation < maximum)
        if eligible.size == 0:
            raise RuntimeError("No capacity remains for exact particle budget")
        order = eligible[np.argsort(residual[eligible], kind="stable")[::-1]]
        take = min(difference, len(order))
        allocation[order[:take]] += 1
        difference -= take

    while difference < 0:
        eligible = np.flatnonzero(allocation > minimum)
        if eligible.size == 0:
            raise RuntimeError("No removable particles remain for exact budget")
        order = eligible[np.argsort(residual[eligible], kind="stable")]
        take = min(-difference, len(order))
        allocation[order[:take]] -= 1
        difference += take

    result = allocation.reshape(priority.shape)
    if int(result.sum()) != target_total:
        raise RuntimeError("Low-amplitude allocation failed exact-budget check")
    if int(result.min()) < minimum or int(result.max()) > maximum:
        raise RuntimeError("Low-amplitude allocation violated local bounds")
    return result


def _build_low_frequency_model(input_channels: int):
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError("Install optional ML dependencies: pip install -e '.[ml]'") from exc

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int, dilation: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(8, channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(8, channels),
            )
            self.activation = nn.GELU()

        def forward(self, values):
            return self.activation(values + self.layers(values))

    class LowFrequencyCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(input_channels, 48, 3, padding=1),
                nn.GroupNorm(8, 48),
                nn.GELU(),
            )
            self.body = nn.Sequential(
                ResidualBlock(48, 1),
                ResidualBlock(48, 2),
                ResidualBlock(48, 3),
            )
            self.head = nn.Sequential(
                nn.Conv2d(48, 24, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(24, 1, 1),
                nn.Sigmoid(),
            )

        def forward(self, values):
            return self.head(self.body(self.stem(values))).squeeze(1)

    return LowFrequencyCNN(), torch


def load_low_frequency_model(model_path: str | Path):
    model_path = Path(model_path)
    model, torch = _build_low_frequency_model(10)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    input_channels = int(checkpoint["input_channels"])
    if input_channels != 10:
        model, torch = _build_low_frequency_model(input_channels)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    return model, torch, mean, std


def predict_rank_priority(
    fields: dict[str, np.ndarray],
    cfg: PhysicalCavityConfig,
    model_bundle,
) -> np.ndarray:
    model, torch, mean, std = model_bundle
    sigma_t = np.asarray(
        fields.get("sigma_T", np.zeros((cfg.ny, cfg.nx))),
        dtype=np.float32,
    )
    base_features = np.stack(
        [fields["T"], fields["u"], fields["v"], sigma_t],
        axis=0,
    ).astype(np.float32)
    context = np.array(
        [
            cfg.knudsen,
            (cfg.t_left - cfg.t_right) / max(cfg.t0, 1.0e-12),
        ],
        dtype=np.float32,
    )
    features = _augment_features(base_features, context)
    normalized = ((features[None] - mean) / std).astype(np.float32)
    with torch.no_grad():
        prediction = model(torch.from_numpy(normalized)).numpy()[0]
    return np.clip(np.asarray(prediction, dtype=np.float64), 0.0, 1.0)


def _paired_statistics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    standard_error = (
        float(np.std(values, ddof=1) / np.sqrt(len(values)))
        if len(values) > 1
        else 0.0
    )
    return {
        "mean": mean,
        "median": float(np.median(values)),
        "standard_error": standard_error,
        "ci95_low": mean - 1.96 * standard_error,
        "ci95_high": mean + 1.96 * standard_error,
    }


def run_closed_loop_benchmark(
    model_path: str | Path,
    output_dir: str | Path,
    cfg: ClosedLoopConfig = ClosedLoopConfig(),
) -> dict[str, object]:
    if not cfg.amplitudes or not cfg.seeds or not cfg.conditions:
        raise ValueError("amplitudes, seeds, and conditions must be non-empty")
    model_bundle = load_low_frequency_model(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    references: dict[tuple[float, float], dict[str, np.ndarray]] = {}
    reference_diagnostics: dict[str, dict[str, float]] = {}
    for knudsen, delta_temperature in cfg.conditions:
        reference_cfg = ShakhovReferenceConfig(
            nx=cfg.nx,
            ny=cfg.ny,
            nv=cfg.nv,
            knudsen=knudsen,
            t_left=300.0 + delta_temperature,
            t_right=300.0 - delta_temperature,
            t_top=300.0,
            t_bottom=300.0,
            max_steps=cfg.dvm_max_steps,
        )
        reference = solve_shakhov_reference(reference_cfg)
        references[(knudsen, delta_temperature)] = reference
        reference_diagnostics[f"kn{knudsen:.3f}_dt{delta_temperature:.0f}"] = {
            "iterations": float(reference["iterations"]),
            "final_residual": float(reference["residual_history"][-1]),
            "mean_temperature": float(np.mean(reference["T"])),
        }

    rows: list[dict[str, object]] = []
    saved_maps: dict[str, np.ndarray] = {}
    for knudsen, delta_temperature in cfg.conditions:
        reference = references[(knudsen, delta_temperature)]
        for seed in cfg.seeds:
            warm_cfg = PhysicalCavityConfig(
                nx=cfg.nx,
                ny=cfg.ny,
                particles_per_cell=cfg.particles_per_cell,
                knudsen=knudsen,
                t_left=300.0 + delta_temperature,
                t_right=300.0 - delta_temperature,
                t_top=300.0,
                t_bottom=300.0,
                steps=cfg.warm_steps,
                sample_start=max(1, cfg.warm_steps // 2),
                seed=seed,
            )
            warm_fields, warm_state, warm_diagnostics = run_physical_cavity(
                warm_cfg,
                return_state=True,
            )
            priority = predict_rank_priority(warm_fields, warm_cfg, model_bundle)
            case_key = f"kn{knudsen:.3f}_dt{delta_temperature:.0f}_s{seed}"
            saved_maps[f"priority_{case_key}"] = priority.astype(np.float32)

            uniform_target = uniform_exact_budget_ppc(
                (cfg.ny, cfg.nx),
                cfg.particles_per_cell,
                1.0,
            )
            uniform_state = conservative_reallocate(
                warm_state,
                warm_cfg,
                uniform_target,
                seed=seed + 10000,
            )
            uniform_fields, _, uniform_diagnostics = advance_physical_state(
                uniform_state,
                warm_cfg,
                cfg.continuation_steps,
                cfg.continuation_steps // 2,
                seed=seed + 20000,
            )
            uniform_error = field_error(uniform_fields, reference)

            for amplitude in cfg.amplitudes:
                target = exact_low_amplitude_ppc(
                    priority,
                    cfg.particles_per_cell,
                    amplitude,
                )
                saved_maps[
                    f"target_a{amplitude:.3f}_{case_key}"
                ] = target.astype(np.int16)
                adaptive_state = conservative_reallocate(
                    warm_state,
                    warm_cfg,
                    target,
                    seed=seed + 10000,
                )
                adaptive_fields, _, adaptive_diagnostics = advance_physical_state(
                    adaptive_state,
                    warm_cfg,
                    cfg.continuation_steps,
                    cfg.continuation_steps // 2,
                    seed=seed + 20000,
                )
                adaptive_error = field_error(adaptive_fields, reference)
                fractional_change = target / cfg.particles_per_cell - 1.0
                rows.append(
                    {
                        "knudsen": knudsen,
                        "delta_temperature": delta_temperature,
                        "seed": seed,
                        "amplitude": amplitude,
                        "uniform_error": uniform_error,
                        "adaptive_error": adaptive_error,
                        "error_ratio": adaptive_error / max(uniform_error, 1.0e-14),
                        "improvement_percent": 100.0
                        * (1.0 - adaptive_error / max(uniform_error, 1.0e-14)),
                        "uniform_particles": int(uniform_target.sum()),
                        "adaptive_particles": int(target.sum()),
                        "particle_ratio": float(target.sum() / uniform_target.sum()),
                        "target_min_ppc": int(target.min()),
                        "target_max_ppc": int(target.max()),
                        "mean_absolute_fractional_change": float(
                            np.mean(np.abs(fractional_change))
                        ),
                        "maximum_fractional_change": float(
                            np.max(np.abs(fractional_change))
                        ),
                        "priority_std": float(np.std(priority)),
                        "warm_collisions": float(
                            warm_diagnostics["accepted_collisions"]
                        ),
                        "uniform_collisions": float(
                            uniform_diagnostics["accepted_collisions"]
                        ),
                        "adaptive_collisions": float(
                            adaptive_diagnostics["accepted_collisions"]
                        ),
                    }
                )

    amplitude_summaries: dict[str, dict[str, object]] = {}
    for amplitude in cfg.amplitudes:
        selected = [row for row in rows if row["amplitude"] == amplitude]
        ratios = np.asarray([row["error_ratio"] for row in selected], dtype=float)
        statistics = _paired_statistics(ratios)
        amplitude_summaries[f"{amplitude:.3f}"] = {
            **statistics,
            "run_count": len(selected),
            "improved_runs": int(np.sum(ratios < 1.0)),
            "non_worse_runs": int(np.sum(ratios <= 1.0)),
            "mean_improvement_percent": 100.0 * (1.0 - statistics["mean"]),
            "mean_realized_absolute_fractional_change": float(
                np.mean(
                    [row["mean_absolute_fractional_change"] for row in selected]
                )
            ),
            "maximum_realized_fractional_change": float(
                np.max([row["maximum_fractional_change"] for row in selected])
            ),
            "mean_collision_ratio": float(
                np.mean(
                    [
                        row["adaptive_collisions"]
                        / max(row["uniform_collisions"], 1.0)
                        for row in selected
                    ]
                )
            ),
        }

    summary: dict[str, object] = {
        "stage": 17,
        "description": (
            "Low-amplitude matched-budget paired closed loop on unseen seeds"
        ),
        "model_path": str(model_path),
        "configuration": {
            "nx": cfg.nx,
            "ny": cfg.ny,
            "particles_per_cell": cfg.particles_per_cell,
            "warm_steps": cfg.warm_steps,
            "continuation_steps": cfg.continuation_steps,
            "nv": cfg.nv,
            "dvm_max_steps": cfg.dvm_max_steps,
            "amplitudes": list(cfg.amplitudes),
            "seeds": list(cfg.seeds),
            "conditions": [list(value) for value in cfg.conditions],
        },
        "reference_diagnostics": reference_diagnostics,
        "rows": rows,
        "amplitude_summaries": amplitude_summaries,
        "scientific_guard": (
            "No closed-loop benefit is claimed unless the paired 95% interval "
            "for an amplitude lies entirely below an error ratio of one."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(output_dir / "allocation_maps.npz", **saved_maps)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a low-frequency score model in a paired matched-budget DSMC loop"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="outputs/stage17_closed_loop")
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--ppc", type=int, default=20)
    parser.add_argument("--warm-steps", type=int, default=40)
    parser.add_argument("--continuation-steps", type=int, default=60)
    parser.add_argument("--nv", type=int, default=6)
    parser.add_argument("--dvm-max-steps", type=int, default=900)
    parser.add_argument("--amplitudes", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--seeds", nargs="+", type=int, default=[66, 77, 88])
    args = parser.parse_args()
    summary = run_closed_loop_benchmark(
        args.model,
        args.output_dir,
        ClosedLoopConfig(
            nx=args.nx,
            ny=args.ny,
            particles_per_cell=args.ppc,
            warm_steps=args.warm_steps,
            continuation_steps=args.continuation_steps,
            nv=args.nv,
            dvm_max_steps=args.dvm_max_steps,
            amplitudes=tuple(args.amplitudes),
            seeds=tuple(args.seeds),
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
