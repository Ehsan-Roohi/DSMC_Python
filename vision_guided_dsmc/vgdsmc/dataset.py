from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import numpy as np
from .simulator import CavityConfig, run_cavity


def relative_error(a: np.ndarray, b: np.ndarray, floor: float) -> np.ndarray:
    return np.abs(a - b) / np.maximum(np.abs(b), floor)


def make_label(coarse: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    e_t = relative_error(coarse["T"], reference["T"], 0.05)
    speed_c = np.hypot(coarse["u"], coarse["v"])
    speed_r = np.hypot(reference["u"], reference["v"])
    e_u = np.abs(speed_c - speed_r) / 0.15
    e_rho = relative_error(coarse["rho"], reference["rho"], 0.1)
    score = 0.45 * e_t + 0.35 * e_u + 0.20 * e_rho
    label = np.zeros_like(score, dtype=np.int64)
    label[(score >= 0.08) & (score < 0.20)] = 1
    label[score >= 0.20] = 2
    return score, label


def generate_case(output: str | Path, cfg: CavityConfig, reference_ppc: int = 120) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    coarse = run_cavity(cfg)
    reference = run_cavity(replace(cfg, particles_per_cell=reference_ppc, seed=cfg.seed + 1000))
    score, label = make_label(coarse, reference)
    x = np.stack([coarse["T"], coarse["u"], coarse["v"], coarse["sigma_T"]], axis=0).astype(np.float32)
    np.savez_compressed(output / "case.npz", x=x, label=label, score=score, **{f"coarse_{k}": v for k, v in coarse.items()}, **{f"reference_{k}": v for k, v in reference.items()})
    return output / "case.npz"
