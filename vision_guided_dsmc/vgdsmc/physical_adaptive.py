from __future__ import annotations

import numpy as np

from .vhs_model import (
    KB, MASS_AR, PhysicalCavityConfig, PhysicalParticleState,
    _cell_xy, _thermal_velocities,
)


def _smooth3(field: np.ndarray) -> np.ndarray:
    padded = np.pad(field, 1, mode="edge")
    out = np.zeros_like(field, dtype=float)
    for j in range(3):
        for i in range(3):
            out += padded[j:j+field.shape[0], i:i+field.shape[1]]
    return out / 9.0


def gradient_priority(fields: dict[str, np.ndarray]) -> np.ndarray:
    temperature = _smooth3(fields["T"])
    density = _smooth3(fields.get("rho", np.ones_like(temperature)))
    gy_t, gx_t = np.gradient(temperature)
    gy_r, gx_r = np.gradient(density)
    grad_t = np.hypot(gx_t, gy_t)
    grad_r = np.hypot(gx_r, gy_r)
    noise = _smooth3(fields.get("sigma_T", np.zeros_like(grad_t)))

    def robust_scale(a: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(a, [5.0, 95.0])
        return np.clip((a - lo) / max(hi - lo, 1.0e-30), 0.0, 1.0)

    return (
        0.65 * robust_scale(grad_t)
        + 0.20 * robust_scale(grad_r)
        + 0.15 * robust_scale(noise)
    )


def exact_budget_ppc(
    priority: np.ndarray,
    base_ppc: int,
    budget_ratio: float,
    min_ppc: int = 4,
    max_ppc: int = 100,
) -> np.ndarray:
    target_total = int(round(priority.size * base_ppc * budget_ratio))
    target_total = int(
        np.clip(
            target_total,
            priority.size * min_ppc,
            priority.size * max_ppc,
        )
    )
    score = np.maximum(priority.astype(float), 0.0) + 0.05
    raw = score / score.sum() * target_total
    ppc = np.clip(np.floor(raw).astype(int), min_ppc, max_ppc)
    difference = target_total - int(ppc.sum())
    residual = raw - np.floor(raw)
    if difference > 0:
        for index in np.argsort(residual.ravel())[::-1]:
            j, i = np.unravel_index(index, ppc.shape)
            if ppc[j, i] < max_ppc:
                ppc[j, i] += 1
                difference -= 1
                if difference == 0:
                    break
    elif difference < 0:
        for index in np.argsort(residual.ravel()):
            j, i = np.unravel_index(index, ppc.shape)
            if ppc[j, i] > min_ppc:
                ppc[j, i] -= 1
                difference += 1
                if difference == 0:
                    break
    if int(ppc.sum()) != target_total:
        raise RuntimeError("Exact physical particle-budget correction failed")
    return ppc


def adaptation_target(
    fields: dict[str, np.ndarray],
    cfg: PhysicalCavityConfig,
    budget_ratio: float = 1.25,
    noise_limit: float = 0.42,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Return a confidence-gated vision allocation map."""
    relative_noise = float(
        np.mean(fields.get("sigma_T", 0.0))
        / max(np.mean(fields["T"]), 1.0e-30)
    )
    priority = gradient_priority(fields)
    adapted = (
        relative_noise <= noise_limit
        and float(np.std(priority)) >= 0.05
    )
    if adapted:
        target = exact_budget_ppc(
            priority,
            cfg.particles_per_cell,
            budget_ratio,
            min_ppc=max(4, int(round(0.8 * cfg.particles_per_cell))),
            max_ppc=max(
                cfg.particles_per_cell + 1,
                int(round(2.0 * cfg.particles_per_cell)),
            ),
        )
    else:
        target = np.full(
            (cfg.ny, cfg.nx),
            cfg.particles_per_cell,
            dtype=int,
        )
    return target, {
        "adapted": bool(adapted),
        "relative_temperature_noise": relative_noise,
        "priority_std": float(np.std(priority)),
    }


def conservative_reallocate(
    state: PhysicalParticleState,
    cfg: PhysicalCavityConfig,
    target_ppc: np.ndarray,
    seed: int = 0,
) -> PhysicalParticleState:
    """Resample each cell while preserving represented mass, momentum, and energy."""
    rng = np.random.default_rng(seed)
    ix, iy = _cell_xy(state.pos, cfg)
    new_pos: list[np.ndarray] = []
    new_vel: list[np.ndarray] = []
    new_weight: list[np.ndarray] = []
    dx, dy = cfg.length / cfg.nx, cfg.length / cfg.ny

    for j in range(cfg.ny):
        for i in range(cfg.nx):
            ids = np.flatnonzero((ix == i) & (iy == j))
            target = int(target_ppc[j, i])
            if target < 2:
                raise ValueError("target PPC must be at least two")
            if len(ids) == 0:
                positions = np.column_stack(
                    (
                        (i + rng.random(target)) * dx,
                        (j + rng.random(target)) * dy,
                    )
                )
                velocities = _thermal_velocities(
                    target,
                    cfg.t0,
                    cfg.vhs,
                    rng,
                )
                weights = np.full(target, 1.0e-12)
            else:
                old_weights = state.weight[ids]
                total_weight = float(old_weights.sum())
                probabilities = old_weights / total_weight
                chosen = rng.choice(
                    ids,
                    size=target,
                    replace=True,
                    p=probabilities,
                )
                positions = state.pos[chosen].copy()
                positions[:, 0] = np.clip(
                    positions[:, 0]
                    + rng.normal(0.0, 0.03 * dx, target),
                    i * dx,
                    np.nextafter((i + 1) * dx, i * dx),
                )
                positions[:, 1] = np.clip(
                    positions[:, 1]
                    + rng.normal(0.0, 0.03 * dy, target),
                    j * dy,
                    np.nextafter((j + 1) * dy, j * dy),
                )
                velocities = state.vel[chosen].copy()
                weights = np.full(target, total_weight / target)

                old_mean = (
                    np.sum(
                        old_weights[:, None] * state.vel[ids],
                        axis=0,
                    )
                    / total_weight
                )
                old_centered = state.vel[ids] - old_mean
                old_thermal = float(
                    np.sum(
                        old_weights
                        * np.sum(old_centered**2, axis=1)
                    )
                )
                sampled_mean = np.mean(velocities, axis=0)
                centered = velocities - sampled_mean
                sampled_thermal = float(
                    np.sum(
                        weights * np.sum(centered**2, axis=1)
                    )
                )
                if old_thermal <= 1.0e-30:
                    velocities[:] = old_mean
                elif sampled_thermal > 1.0e-30:
                    velocities = (
                        old_mean
                        + centered
                        * np.sqrt(old_thermal / sampled_thermal)
                    )
                else:
                    perturbation = rng.normal(size=velocities.shape)
                    perturbation -= perturbation.mean(axis=0)
                    denominator = float(
                        np.sum(
                            weights
                            * np.sum(perturbation**2, axis=1)
                        )
                    )
                    velocities = (
                        old_mean
                        + perturbation
                        * np.sqrt(
                            old_thermal / max(denominator, 1.0e-30)
                        )
                    )
            new_pos.append(positions)
            new_vel.append(velocities)
            new_weight.append(weights)

    return PhysicalParticleState(
        np.vstack(new_pos),
        np.vstack(new_vel),
        np.concatenate(new_weight),
    )


def field_error(
    fields: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> float:
    temperature_error = (
        np.mean(np.abs(fields["T"] - reference["T"]))
        / max(np.mean(reference["T"]), 1.0e-30)
    )
    speed = np.hypot(fields["u"], fields["v"])
    reference_speed = np.hypot(reference["u"], reference["v"])
    thermal_speed = np.sqrt(
        2.0 * KB * np.mean(reference["T"]) / MASS_AR
    )
    speed_error = (
        np.mean(np.abs(speed - reference_speed))
        / max(thermal_speed, 1.0e-30)
    )
    density_error = np.mean(
        np.abs(fields["rho"] - reference["rho"])
    )
    return float(
        0.45 * temperature_error
        + 0.25 * speed_error
        + 0.30 * density_error
    )


def uniform_exact_budget_ppc(
    shape: tuple[int, int],
    base_ppc: int,
    budget_ratio: float = 1.0,
) -> np.ndarray:
    """Return the most uniform integer PPC map with an exact global budget."""
    if base_ppc <= 0 or budget_ratio <= 0.0:
        raise ValueError("base_ppc and budget_ratio must be positive")
    total_cells = int(np.prod(shape))
    budget = int(round(total_cells * base_ppc * budget_ratio))
    quotient, remainder = divmod(budget, total_cells)
    target = np.full(total_cells, quotient, dtype=np.int64)
    target[:remainder] += 1
    return target.reshape(shape)
