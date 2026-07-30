from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .simulator import ParticleState

CLASS_MULTIPLIERS = np.array([0.5, 1.0, 3.0], dtype=np.float64)


@dataclass(frozen=True)
class ConservationReport:
    populated_cells: int
    empty_cells: int
    max_mass_relative_error: float
    max_momentum_absolute_error: float
    max_energy_relative_error: float


def label_to_target_ppc(
    label: np.ndarray,
    base_ppc: int = 20,
    min_ppc: int = 4,
    max_ppc: int = 200,
) -> np.ndarray:
    if np.any((label < 0) | (label >= len(CLASS_MULTIPLIERS))):
        raise ValueError("label must contain only classes 0, 1, and 2")
    target = np.rint(base_ppc * CLASS_MULTIPLIERS[label]).astype(np.int64)
    return np.clip(target, min_ppc, max_ppc)


def _moments(velocity: np.ndarray, weight: np.ndarray) -> tuple[float, np.ndarray, float]:
    mass = float(np.sum(weight))
    if mass <= 0.0:
        return 0.0, np.zeros(2), 0.0
    momentum = np.sum(weight[:, None] * velocity, axis=0)
    energy = float(0.5 * np.sum(weight * np.sum(velocity**2, axis=1)))
    return mass, momentum, energy


def conservative_reallocate_state(
    state: ParticleState,
    target_ppc: np.ndarray,
    rng: np.random.Generator | None = None,
    jitter_fraction: float = 0.02,
) -> tuple[ParticleState, ConservationReport]:
    """Resample each cell while preserving represented mass, momentum, and energy.

    Every populated cell receives equal post-resampling weights. Velocities are
    shifted and scaled so weighted first and second moments match the source cell.
    """
    rng = np.random.default_rng() if rng is None else rng
    ny, nx = target_ppc.shape
    ix = np.clip((state.pos[:, 0] * nx).astype(np.int64), 0, nx - 1)
    iy = np.clip((state.pos[:, 1] * ny).astype(np.int64), 0, ny - 1)
    new_pos: list[np.ndarray] = []
    new_vel: list[np.ndarray] = []
    new_weight: list[np.ndarray] = []
    mass_errors: list[float] = []
    momentum_errors: list[float] = []
    energy_errors: list[float] = []
    populated_cells = 0
    empty_cells = 0

    for j in range(ny):
        for i in range(nx):
            ids = np.flatnonzero((ix == i) & (iy == j))
            target = int(target_ppc[j, i])
            if target < 2:
                raise ValueError("target PPC must be at least 2 for energy preservation")

            if len(ids) == 0:
                empty_cells += 1
                positions = np.column_stack(
                    ((i + rng.random(target)) / nx, (j + rng.random(target)) / ny)
                )
                velocities = rng.normal(0.0, 1.0, (target, 2))
                weights = np.ones(target)
            else:
                populated_cells += 1
                source_weight = state.weight[ids]
                probability = source_weight / np.sum(source_weight)
                chosen = rng.choice(ids, size=target, replace=target > len(ids), p=probability)
                positions = state.pos[chosen].copy()
                positions += rng.normal(
                    0.0, jitter_fraction / max(nx, ny), positions.shape
                )
                positions[:, 0] = np.clip(
                    positions[:, 0], i / nx + 1.0e-12, (i + 1) / nx - 1.0e-12
                )
                positions[:, 1] = np.clip(
                    positions[:, 1], j / ny + 1.0e-12, (j + 1) / ny - 1.0e-12
                )
                velocities = state.vel[chosen].copy()

                old_mass, old_momentum, old_energy = _moments(
                    state.vel[ids], source_weight
                )
                weights = np.full(target, old_mass / target)
                old_mean = old_momentum / old_mass
                sampled_mean = np.average(velocities, axis=0, weights=weights)
                fluctuation = velocities - sampled_mean
                sampled_thermal = float(
                    0.5 * np.sum(weights * np.sum(fluctuation**2, axis=1))
                )
                target_thermal = max(
                    old_energy - 0.5 * old_mass * float(np.dot(old_mean, old_mean)),
                    0.0,
                )
                if sampled_thermal <= 1.0e-14 and target_thermal > 0.0:
                    fluctuation = rng.normal(0.0, 1.0, velocities.shape)
                    fluctuation -= np.average(fluctuation, axis=0, weights=weights)
                    sampled_thermal = float(
                        0.5 * np.sum(weights * np.sum(fluctuation**2, axis=1))
                    )
                scale = (
                    np.sqrt(target_thermal / sampled_thermal)
                    if sampled_thermal > 1.0e-14
                    else 0.0
                )
                velocities = old_mean + scale * fluctuation

                new_mass, new_momentum, new_energy = _moments(velocities, weights)
                mass_errors.append(abs(new_mass - old_mass) / max(abs(old_mass), 1.0e-14))
                momentum_errors.append(float(np.max(np.abs(new_momentum - old_momentum))))
                energy_errors.append(
                    abs(new_energy - old_energy) / max(abs(old_energy), 1.0e-14)
                )

            new_pos.append(positions)
            new_vel.append(velocities)
            new_weight.append(weights)

    report = ConservationReport(
        populated_cells=populated_cells,
        empty_cells=empty_cells,
        max_mass_relative_error=max(mass_errors, default=0.0),
        max_momentum_absolute_error=max(momentum_errors, default=0.0),
        max_energy_relative_error=max(energy_errors, default=0.0),
    )
    return (
        ParticleState(np.vstack(new_pos), np.vstack(new_vel), np.concatenate(new_weight)),
        report,
    )


def reallocate_particles(
    pos: np.ndarray,
    vel: np.ndarray,
    target_ppc: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    state = ParticleState(pos.copy(), vel.copy(), np.ones(len(pos)))
    new_state, _ = conservative_reallocate_state(state, target_ppc, rng)
    return new_state.pos, new_state.vel


def allocation_summary(label: np.ndarray, base_ppc: int = 20) -> dict[str, float | int]:
    target = label_to_target_ppc(label, base_ppc)
    uniform = label.size * base_ppc
    return {
        "uniform_particles": int(uniform),
        "adaptive_particles": int(target.sum()),
        "ratio": float(target.sum() / uniform),
    }


def score_to_target_ppc(
    score: np.ndarray,
    base_ppc: int,
    budget_ratio: float = 1.0,
    alpha: float = 0.25,
    min_ppc: int = 4,
    max_ppc: int = 200,
) -> np.ndarray:
    """Convert a continuous need score into an exact-budget PPC map.

    A robust standardized score is exponentiated into a positive priority. A
    bisection scale and largest-remainder correction then enforce the requested
    global particle budget exactly.
    """
    if base_ppc <= 0 or budget_ratio <= 0.0 or alpha < 0.0:
        raise ValueError("base_ppc and budget_ratio must be positive; alpha nonnegative")
    if min_ppc < 2 or max_ppc < min_ppc:
        raise ValueError("Require 2 <= min_ppc <= max_ppc")
    score = np.asarray(score, dtype=np.float64)
    q25, median, q75 = np.quantile(score, [0.25, 0.50, 0.75])
    robust_scale = max(float(q75 - q25), 1.0e-12)
    standardized = (score - median) / robust_scale
    priority = np.exp(np.clip(alpha * standardized, -4.0, 4.0))
    budget = int(round(score.size * base_ppc * budget_ratio))
    feasible_min = score.size * min_ppc
    feasible_max = score.size * max_ppc
    if not feasible_min <= budget <= feasible_max:
        raise ValueError(
            f"Requested budget {budget} is outside feasible range "
            f"[{feasible_min}, {feasible_max}]"
        )

    low, high = 0.0, max_ppc / max(float(priority.min()), 1.0e-12)
    for _ in range(80):
        midpoint = 0.5 * (low + high)
        continuous = np.clip(midpoint * priority, min_ppc, max_ppc)
        if float(continuous.sum()) > budget:
            high = midpoint
        else:
            low = midpoint

    continuous = np.clip(low * priority, min_ppc, max_ppc)
    target = np.floor(continuous).astype(np.int64)
    difference = int(budget - target.sum())
    flat_target = target.ravel()
    flat_fraction = (continuous - target).ravel()
    if difference > 0:
        eligible = np.flatnonzero(flat_target < max_ppc)
        order = eligible[np.argsort(flat_fraction[eligible])[::-1]]
        flat_target[order[:difference]] += 1
    elif difference < 0:
        eligible = np.flatnonzero(flat_target > min_ppc)
        order = eligible[np.argsort(flat_fraction[eligible])]
        flat_target[order[: -difference]] -= 1
    if int(target.sum()) != budget:
        raise RuntimeError("Exact particle-budget correction failed")
    return target
