from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import json
import numpy as np

from .adaptive import conservative_reallocate_state, score_to_target_ppc
from .dataset import make_label
from .simulator import CavityConfig, ParticleState, advance_state, run_cavity
from .vision import physics_vision_score


def relative_l2(a: np.ndarray, reference: np.ndarray, floor: float = 1.0e-12) -> float:
    return float(np.linalg.norm(a - reference) / max(np.linalg.norm(reference), floor))


def field_errors(fields: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "rho_l2": relative_l2(fields["rho"], reference["rho"]),
        "T_l2": relative_l2(fields["T"], reference["T"]),
        "speed_l2": relative_l2(
            np.hypot(fields["u"], fields["v"]),
            np.hypot(reference["u"], reference["v"]),
        ),
    }


def _continue_and_compare(
    cfg: CavityConfig,
    coarse_state: ParticleState,
    score: np.ndarray,
    score_source: str,
    reference_ppc: int,
    continuation_steps: int,
    budget_ratio: float,
    allocation_alpha: float,
    relaxation_fraction: float,
    output: str | Path | None,
    label: np.ndarray | None = None,
) -> dict:
    if not 0.0 <= relaxation_fraction < 1.0:
        raise ValueError("relaxation_fraction must lie in [0, 1)")
    sample_start = min(
        continuation_steps - 1,
        int(round(continuation_steps * relaxation_fraction)),
    )
    target = score_to_target_ppc(
        score,
        base_ppc=cfg.particles_per_cell,
        budget_ratio=budget_ratio,
        alpha=allocation_alpha,
    )
    adaptive_state, conservation = conservative_reallocate_state(
        coarse_state.copy(), target, np.random.default_rng(cfg.seed + 2000)
    )
    baseline_fields, _ = advance_state(
        coarse_state.copy(),
        cfg,
        steps=continuation_steps,
        sample_start=sample_start,
        seed=cfg.seed + 3000,
    )
    adaptive_fields, _ = advance_state(
        adaptive_state,
        cfg,
        steps=continuation_steps,
        sample_start=sample_start,
        seed=cfg.seed + 3000,
    )
    reference_cfg = replace(
        cfg,
        particles_per_cell=reference_ppc,
        seed=cfg.seed + 1000,
        steps=cfg.steps + continuation_steps,
        sample_start=cfg.steps + sample_start,
    )
    reference_final = run_cavity(reference_cfg)
    baseline_errors = field_errors(baseline_fields, reference_final)
    adaptive_errors = field_errors(adaptive_fields, reference_final)
    baseline_mean = float(np.mean(list(baseline_errors.values())))
    adaptive_mean = float(np.mean(list(adaptive_errors.values())))

    result = {
        "config": asdict(cfg),
        "score_source": score_source,
        "reference_ppc": reference_ppc,
        "continuation_steps": continuation_steps,
        "continuation_sample_start": sample_start,
        "budget_ratio": budget_ratio,
        "allocation_alpha": allocation_alpha,
        "target_particles": int(target.sum()),
        "uniform_particles": int(score.size * cfg.particles_per_cell),
        "continuation_cost_ratio": float(
            target.sum() / (score.size * cfg.particles_per_cell)
        ),
        "baseline_errors": baseline_errors,
        "adaptive_errors": adaptive_errors,
        "mean_error_ratio": adaptive_mean / baseline_mean,
        "conservation": asdict(conservation),
    }
    if label is not None:
        result["class_counts"] = np.bincount(label.ravel(), minlength=3).tolist()

    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        arrays = {
            "score": score,
            "target_ppc": target,
            **{f"baseline_{key}": value for key, value in baseline_fields.items()},
            **{f"adaptive_{key}": value for key, value in adaptive_fields.items()},
            **{f"reference_{key}": value for key, value in reference_final.items()},
        }
        if label is not None:
            arrays["label"] = label
        np.savez_compressed(output_path.with_suffix(".npz"), **arrays)
    return result


def run_vision_closed_loop(
    cfg: CavityConfig,
    reference_ppc: int,
    continuation_steps: int,
    vision_mode: str = "temperature_gradient",
    budget_ratio: float = 1.25,
    allocation_alpha: float = 0.50,
    relaxation_fraction: float = 2.0 / 3.0,
    output: str | Path | None = None,
) -> dict:
    coarse_fields, coarse_state = run_cavity(cfg, return_state=True)
    score = physics_vision_score(coarse_fields, mode=vision_mode)
    return _continue_and_compare(
        cfg,
        coarse_state,
        score,
        score_source=f"physics_vision:{vision_mode}",
        reference_ppc=reference_ppc,
        continuation_steps=continuation_steps,
        budget_ratio=budget_ratio,
        allocation_alpha=allocation_alpha,
        relaxation_fraction=relaxation_fraction,
        output=output,
    )


def run_oracle_closed_loop(
    cfg: CavityConfig,
    reference_ppc: int,
    continuation_steps: int,
    low_threshold: float = 0.15,
    high_threshold: float = 0.35,
    output: str | Path | None = None,
    budget_ratio: float = 1.25,
    allocation_alpha: float = 0.25,
    relaxation_fraction: float = 2.0 / 3.0,
) -> dict:
    coarse_fields, coarse_state = run_cavity(cfg, return_state=True)
    reference_warm = run_cavity(
        replace(cfg, particles_per_cell=reference_ppc, seed=cfg.seed + 1000)
    )
    score, label = make_label(
        coarse_fields,
        reference_warm,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    return _continue_and_compare(
        cfg,
        coarse_state,
        score,
        score_source="reference_error_oracle",
        reference_ppc=reference_ppc,
        continuation_steps=continuation_steps,
        budget_ratio=budget_ratio,
        allocation_alpha=allocation_alpha,
        relaxation_fraction=relaxation_fraction,
        output=output,
        label=label,
    )
