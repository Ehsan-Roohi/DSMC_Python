from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import numpy as np

from .simulator import CavityConfig, run_cavity


def relative_error(a: np.ndarray, b: np.ndarray, floor: float) -> np.ndarray:
    return np.abs(a - b) / np.maximum(np.abs(b), floor)


def make_label(
    coarse: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    low_threshold: float = 0.15,
    high_threshold: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Build reduce/retain/increase labels from a combined local error score."""
    if not 0.0 < low_threshold < high_threshold:
        raise ValueError("Require 0 < low_threshold < high_threshold")
    e_t = relative_error(coarse["T"], reference["T"], 0.05)
    speed_c = np.hypot(coarse["u"], coarse["v"])
    speed_r = np.hypot(reference["u"], reference["v"])
    e_u = np.abs(speed_c - speed_r) / 0.15
    e_rho = relative_error(coarse["rho"], reference["rho"], 0.10)
    score = 0.45 * e_t + 0.35 * e_u + 0.20 * e_rho
    label = np.zeros_like(score, dtype=np.int64)
    label[(score >= low_threshold) & (score < high_threshold)] = 1
    label[score >= high_threshold] = 2
    return score, label


def generate_case(
    output: str | Path,
    cfg: CavityConfig,
    reference_ppc: int = 120,
    low_threshold: float = 0.15,
    high_threshold: float = 0.35,
) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    coarse = run_cavity(cfg)
    reference = run_cavity(
        replace(cfg, particles_per_cell=reference_ppc, seed=cfg.seed + 1000)
    )
    score, label = make_label(coarse, reference, low_threshold, high_threshold)
    x = np.stack(
        [coarse["T"], coarse["u"], coarse["v"], coarse["sigma_T"]], axis=0
    ).astype(np.float32)
    path = output / "case.npz"
    np.savez_compressed(
        path,
        x=x,
        label=label,
        score=score,
        thresholds=np.array([low_threshold, high_threshold]),
        **{f"coarse_{key}": value for key, value in coarse.items()},
        **{f"reference_{key}": value for key, value in reference.items()},
    )
    return path
