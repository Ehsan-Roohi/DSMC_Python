from __future__ import annotations

import numpy as np

from .vhs_model import (
    KB,
    MASS_AR,
    PhysicalCavityConfig,
    PhysicalParticleState,
    _cell_xy,
    _thermal_velocities,
)


def _smooth3(field: np.ndarray) -> np.ndarray:
    padded = np.pad(field, 1, mode="edge")
    out = np.zeros_like(field, dtype=float)
    for j in range(3):
        for i in range(3):
            out += padded[
                j : j + field.shape[0],
                i : i + field.shape[1],
            ]
    return out / 9.0


def gradient_priority(fields: dict[str, np.ndarray]) -> np.ndarray:
    """Build a reference-free physical priority image."""
    temperature = _smooth3(fields["T"])
    density = _smooth3(
        fields.get("rho", np.ones_like(temperature))
    )
    gy_t, gx_t = np.gradient(temperature)
    gy_r, gx_r = np.gradient(density)
    grad_t = np.hypot(gx_t, gy_t)
    grad_r = np.hypot(gx_r, gy_r)
    noise = _smooth3(
        fields.get("sigma_T", np.zeros_like(grad_t))
    )

    def robust_scale(array: np.ndarray) -> np.ndarray:
        low, high = np.percentile(array, [5.0, 95.0])
        return np.clip(
            (array - low) / max(float(high - low), 1.0e-30),
            0.0,
            1.0,
        )

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
    """Allocate an exact bounded integer particle budget.

    The previous one-pass largest-remainder correction could fail after cells
    were clipped at ``min_ppc`` or ``max_ppc``. This bounded apportionment starts
    every cell at the minimum and repeatedly distributes the remaining budget
    over cells that still have capacity, so the requested global sum is exact.
    """
    priority = np.asarray(priority, dtype=np.float64)
    if priority.size == 0:
        raise ValueError("priority must contain at least one cell")
    if base_ppc <= 0 or budget_ratio <= 0.0:
        raise ValueError("base_ppc and budget_ratio must be positive")
    if min_ppc < 2 or max_ppc < min_ppc:
        raise ValueError("Require 2 <= min_ppc <= max_ppc")

    cells = priority.size
    requested = int(round(cells * base_ppc * budget_ratio))
    target_total = int(
        np.clip(requested, cells * min_ppc, cells * max_ppc)
    )

    flat_priority = np.maximum(priority.ravel(), 0.0) + 0.05
    allocation = np.full(cells, min_ppc, dtype=np.int64)
    capacity = np.full(cells, max_ppc - min_ppc, dtype=np.int64)
    remaining = target_total - int(allocation.sum())

    while remaining > 0:
        eligible = np.flatnonzero(capacity > 0)
        if eligible.size == 0:
            raise RuntimeError(
                "Exact physical particle-budget correction exhausted capacity"
            )

        weights = flat_priority[eligible]
        weight_sum = float(weights.sum())
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            weights = np.ones_like(weights)
            weight_sum = float(weights.sum())

        quota = remaining * weights / weight_sum
        whole = np.minimum(
            np.floor(quota).astype(np.int64),
            capacity[eligible],
        )
        distributed = int(whole.sum())

        if distributed > 0:
            allocation[eligible] += whole
            capacity[eligible] -= whole
            remaining -= distributed
            continue

        # All fractional quotas are below one. Give one particle to the
        # highest-priority residual cells, then repeat if more budget remains.
        order = np.argsort(quota, kind="stable")[::-1]
        take = min(remaining, eligible.size)
        chosen = eligible[order[:take]]
        allocation[chosen] += 1
        capacity[chosen] -= 1
        remaining -= take

    result = allocation.reshape(priority.shape)
    if int(result.sum()) != target_total:
        raise RuntimeError("Exact physical particle-budget correction failed")
    if result.min() < min_ppc or result.max() > max_ppc:
        raise RuntimeError("Physical particle allocation violated PPC bounds")
    return result


def adaptation_target(
    fields: dict[str, np.ndarray],
    cfg: PhysicalCavityConfig,
    budget_ratio: float = 1.25,
    noise_limit: float = 0.42,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Return a confidence-gated physical allocation map."""
    relative_noise = float(
        np.mean(fields.get("sigma_T", 0.0))
        / max(float(np.mean(fields["T"])), 1.0e-30)
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
            min_ppc=max(
                4,
                int(round(0.8 * cfg.particles_per_cell)),
            ),
            max_ppc=max(
                cfg.particles_per_cell + 1,
                int(round(2.0 * cfg.particles_per_cell)),
            ),
        )
    else:
        target = np.full(
            (cfg.ny, cfg.nx),
            cfg.particles_per_cell,
            dtype=np.int64,
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
    """Resample each cell while preserving represented moments."""
    rng = np.random.default_rng(seed)
    ix, iy = _cell_xy(state.pos, cfg)
    new_pos: list[np.ndarray] = []
    new_vel: list[np.ndarray] = []
    new_weight: list[np.ndarray] = []
    dx = cfg.length / cfg.nx
    dy = cfg.length / cfg.ny

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
                    np.sum(weights * np.sum(centered**2, axis=1))
                )

                if old_thermal <= 1.0e-30:
                    velocities[:] = old_mean
                elif sampled_thermal > 1.0e-30:
                    velocities = old_mean + centered * np.sqrt(
                        old_thermal / sampled_thermal
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
                    velocities = old_mean + perturbation * np.sqrt(
                        old_thermal / max(denominator, 1.0e-30)
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
        / max(float(np.mean(reference["T"])), 1.0e-30)
    )
    speed = np.hypot(fields["u"], fields["v"])
    reference_speed = np.hypot(reference["u"], reference["v"])
    thermal_speed = np.sqrt(
        2.0 * KB * np.mean(reference["T"]) / MASS_AR
    )
    speed_error = (
        np.mean(np.abs(speed - reference_speed))
        / max(float(thermal_speed), 1.0e-30)
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
