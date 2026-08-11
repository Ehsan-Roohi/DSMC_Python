"""Collision-pair selection algorithms for the cavity solver.

All methods call the same VHS acceptance and isotropic elastic scattering
kernel.  Only the statistical selection of candidate pairs differs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .physics import vhs_sigma_g


SUPPORTED_MODELS = (
    "ntc",
    "ntc-prescan",
    "mfs",
    "sbt",
    "gbt",
    "ssbt",
    "sgbt",
    "sbt-tas",
    "gbt-tas",
)


@dataclass
class CollisionStats:
    selected: int = 0
    accepted: int = 0
    probability_exceedances: int = 0
    max_probability: float = 0.0
    conflict_rounds: int = 0

    def add(self, other: "CollisionStats") -> None:
        self.selected += other.selected
        self.accepted += other.accepted
        self.probability_exceedances += other.probability_exceedances
        self.max_probability = max(self.max_probability, other.max_probability)
        self.conflict_rounds += other.conflict_rounds


@dataclass
class TrialBatch:
    first: np.ndarray
    second: np.ndarray
    multiplier: np.ndarray
    cell: np.ndarray
    volumes: np.ndarray
    majorant: np.ndarray | None = None

    @classmethod
    def empty(cls) -> "TrialBatch":
        z_i = np.empty(0, dtype=np.int64)
        z_f = np.empty(0, dtype=np.float64)
        return cls(z_i, z_i.copy(), z_f, z_i.copy(), z_f.copy(), None)


def _unique_unordered_pair(
    seen: set[tuple[int, int]], first: int, second: int
) -> bool:
    key = (first, second) if first < second else (second, first)
    if key in seen:
        return False
    seen.add(key)
    return True


def _append_bt_trials(
    model: str,
    ids: np.ndarray,
    volume: float,
    cell_id: int,
    config,
    rng: np.random.Generator,
    first: list[int],
    second: list[int],
    multiplier: list[float],
    cells: list[int],
    volumes: list[float],
) -> None:
    n = len(ids)
    if n < 2:
        return
    local = ids.copy()
    rng.shuffle(local)
    if model in {"sbt", "sbt-tas"}:
        for i in range(n - 1):
            j = int(rng.integers(i + 1, n))
            first.append(int(local[i]))
            second.append(int(local[j]))
            multiplier.append(float(n - i - 1))
            cells.append(cell_id)
            volumes.append(volume)
        return

    if model in {"gbt", "gbt-tas"}:
        nsel = config.selected_particles(n)
        if nsel >= n - 1:
            return _append_bt_trials(
                "sbt", ids, volume, cell_id, config, rng,
                first, second, multiplier, cells, volumes,
            )
        corr = n * (n - 1.0) / (nsel * (2.0 * n - nsel - 1.0))
        for i in range(nsel):
            j = int(rng.integers(i + 1, n))
            first.append(int(local[i]))
            second.append(int(local[j]))
            multiplier.append(float(corr * (n - i - 1)))
            cells.append(cell_id)
            volumes.append(volume)
        return

    if model == "ssbt":
        factor = 0.5 * (n - 1.0)
        for i in range(n):
            j = int(rng.integers(0, n - 1))
            if j >= i:
                j += 1
            first.append(int(local[i]))
            second.append(int(local[j]))
            multiplier.append(float(factor))
            cells.append(cell_id)
            volumes.append(volume)
        return

    if model == "sgbt":
        nsel = config.selected_particles(n)
        if nsel >= n - 1:
            return _append_bt_trials(
                "sbt", ids, volume, cell_id, config, rng,
                first, second, multiplier, cells, volumes,
            )
        factor = n * (n - 1.0) / (2.0 * nsel)
        seen: set[tuple[int, int]] = set()
        for i in range(nsel):
            # Duplicate-pair avoidance is part of SGBT, including reversed pairs.
            for _ in range(max(8, 2 * n)):
                j = int(rng.integers(0, n - 1))
                if j >= i:
                    j += 1
                if _unique_unordered_pair(seen, int(local[i]), int(local[j])):
                    break
            else:
                continue
            first.append(int(local[i]))
            second.append(int(local[j]))
            multiplier.append(float(factor))
            cells.append(cell_id)
            volumes.append(volume)
        return
    raise ValueError(f"Unsupported Bernoulli model {model}")


def _adaptive_subcell_groups(
    ids: np.ndarray,
    positions: np.ndarray,
    main_cell: int,
    config,
) -> Iterable[tuple[np.ndarray, float, int]]:
    """Yield TAS groups with a deterministic half-cell staggered second pass.

    The two passes use half the physical time step each.  Boundary subcells in
    the staggered pass are clipped rather than periodically wrapped.
    """
    n = len(ids)
    if n < 2:
        return
    ix = main_cell % config.nx
    iy = main_cell // config.nx
    nsub = max(1, int(math.sqrt(n / max(1, config.tas_target_particles))))
    x0, y0 = ix * config.dx, iy * config.dy
    local_x = np.clip((positions[ids, 0] - x0) / config.dx, 0.0, 1.0 - 1e-12)
    local_y = np.clip((positions[ids, 1] - y0) / config.dy, 0.0, 1.0 - 1e-12)
    for pass_id in (0, 1):
        if pass_id == 0:
            sx = np.floor(local_x * nsub).astype(int)
            sy = np.floor(local_y * nsub).astype(int)
            stride = nsub
        else:
            # Half-cell shift gives nsub+1 bins. The two boundary bins have
            # half width, matching the environmental-subcell correction in
            # the supplied Fortran TAS implementation.
            sx = np.floor(local_x * nsub + 0.5).astype(int)
            sy = np.floor(local_y * nsub + 0.5).astype(int)
            stride = nsub + 1
        keys = sx + stride * sy
        for key in np.unique(keys):
            group = ids[keys == key]
            if len(group) >= 2:
                if pass_id == 0:
                    volume = config.cell_volume / (nsub * nsub)
                else:
                    key_x = int(key) % stride
                    key_y = int(key) // stride
                    width_x = 0.5 / nsub if key_x in (0, nsub) else 1.0 / nsub
                    width_y = 0.5 / nsub if key_y in (0, nsub) else 1.0 / nsub
                    volume = config.cell_volume * width_x * width_y
                synthetic_id = main_cell * (2 * stride * stride) + pass_id * stride * stride + int(key)
                yield group, volume, synthetic_id


def generate_bt_trials(
    model: str,
    groups: list[np.ndarray],
    positions: np.ndarray,
    config,
    rng: np.random.Generator,
) -> TrialBatch:
    first: list[int] = []
    second: list[int] = []
    multiplier: list[float] = []
    cells: list[int] = []
    volumes: list[float] = []
    if model.endswith("-tas"):
        for cell_id, ids in enumerate(groups):
            for sub_ids, volume, sub_id in _adaptive_subcell_groups(
                ids, positions, cell_id, config
            ):
                before = len(first)
                _append_bt_trials(
                    model, sub_ids, volume, sub_id, config, rng,
                    first, second, multiplier, cells, volumes,
                )
                # Two staggered passes each represent dt/2.
                for k in range(before, len(multiplier)):
                    multiplier[k] *= 0.5
    else:
        for cell_id, ids in enumerate(groups):
            _append_bt_trials(
                model, ids, config.cell_volume, cell_id, config, rng,
                first, second, multiplier, cells, volumes,
            )
    if not first:
        return TrialBatch.empty()
    return TrialBatch(
        np.asarray(first, dtype=np.int64),
        np.asarray(second, dtype=np.int64),
        np.asarray(multiplier, dtype=np.float64),
        np.asarray(cells, dtype=np.int64),
        np.asarray(volumes, dtype=np.float64),
    )


def _random_pairs(ids: np.ndarray, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = len(ids)
    a = rng.integers(0, n, size=count)
    b = rng.integers(0, n - 1, size=count)
    b = b + (b >= a)
    return ids[a], ids[b]


def _prescan_majorants(
    groups: list[np.ndarray], velocities, config, backend, rng, prior: np.ndarray
) -> np.ndarray:
    xp = backend.xp
    updated = prior.copy()
    for cell, ids in enumerate(groups):
        if len(ids) < 2:
            continue
        count = min(32, max(4, len(ids)))
        p, q = _random_pairs(ids, count, rng)
        pp = xp.asarray(p)
        qq = xp.asarray(q)
        rel = velocities[pp] - velocities[qq]
        speed = xp.sqrt(xp.sum(rel * rel, axis=1))
        values = vhs_sigma_g(speed, config, xp)
        local = float(backend.asnumpy(xp.max(values)))
        updated[cell] = max(updated[cell], 1.05 * local)
    return updated


def generate_majorant_trials(
    model: str,
    groups: list[np.ndarray],
    velocities,
    config,
    backend,
    rng: np.random.Generator,
    majorants: np.ndarray,
    dt: float,
) -> tuple[TrialBatch, np.ndarray]:
    if model == "ntc-prescan":
        majorants = _prescan_majorants(groups, velocities, config, backend, rng, majorants)
    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    cells: list[np.ndarray] = []
    majors: list[np.ndarray] = []
    volumes: list[np.ndarray] = []
    for cell, ids in enumerate(groups):
        n = len(ids)
        if n < 2:
            continue
        majorant = max(float(majorants[cell]), config.conservative_sigma_g() * 0.25)
        expectation = (
            0.5
            * n
            * (n - 1)
            * config.particle_weight
            * majorant
            * dt
            / config.cell_volume
        )
        if model == "mfs":
            count = int(rng.poisson(expectation))
        else:
            count = int(math.floor(expectation + rng.random()))
        if count < 1:
            continue
        p, q = _random_pairs(ids, count, rng)
        first.append(p)
        second.append(q)
        cells.append(np.full(count, cell, dtype=np.int64))
        majors.append(np.full(count, majorant, dtype=np.float64))
        volumes.append(np.full(count, config.cell_volume, dtype=np.float64))
    if not first:
        return TrialBatch.empty(), majorants
    count = sum(len(x) for x in first)
    batch = TrialBatch(
        np.concatenate(first),
        np.concatenate(second),
        np.ones(count, dtype=np.float64),
        np.concatenate(cells),
        np.concatenate(volumes),
        np.concatenate(majors),
    )
    return batch, majorants


def _conflict_rounds(first: np.ndarray, second: np.ndarray) -> list[np.ndarray]:
    """Greedy edge coloring so a particle appears at most once per GPU batch."""
    rounds: list[list[int]] = []
    occupied: list[set[int]] = []
    for trial, (p, q) in enumerate(zip(first.tolist(), second.tolist())):
        for k, used in enumerate(occupied):
            if p not in used and q not in used:
                rounds[k].append(trial)
                used.add(p)
                used.add(q)
                break
        else:
            rounds.append([trial])
            occupied.append({p, q})
    return [np.asarray(item, dtype=np.int64) for item in rounds]


def apply_trials(batch: TrialBatch, velocities, config, backend, dt: float) -> CollisionStats:
    stats = CollisionStats(selected=len(batch.first))
    if len(batch.first) == 0:
        return stats
    xp = backend.xp
    p = xp.asarray(batch.first)
    q = xp.asarray(batch.second)
    rel = velocities[p] - velocities[q]
    speed = xp.sqrt(xp.sum(rel * rel, axis=1))
    sigma_g = vhs_sigma_g(speed, config, xp)
    if batch.majorant is None:
        probability = (
            xp.asarray(batch.multiplier)
            * config.particle_weight
            * dt
            * sigma_g
            / xp.asarray(batch.volumes)
        )
    else:
        probability = sigma_g / xp.asarray(batch.majorant)
    max_probability = float(backend.asnumpy(xp.max(probability)))
    stats.max_probability = max_probability
    probability_np = backend.asnumpy(probability)
    stats.probability_exceedances = int(np.count_nonzero(probability_np > 1.0))
    if stats.probability_exceedances and config.strict_probability:
        raise RuntimeError(
            f"Collision probability exceeded unity for {stats.probability_exceedances} "
            f"trials (max={max_probability:.3f}). Reduce dt; Bernoulli-trial "
            "models normally require the smaller probability-limited step."
        )
    accept = backend.uniform(len(batch.first)) < xp.minimum(probability, 1.0)
    accepted = np.flatnonzero(backend.asnumpy(accept))
    stats.accepted = len(accepted)
    if not len(accepted):
        return stats
    rounds = _conflict_rounds(batch.first[accepted], batch.second[accepted])
    stats.conflict_rounds = len(rounds)
    for local_round in rounds:
        chosen = accepted[local_round]
        pp = xp.asarray(batch.first[chosen])
        qq = xp.asarray(batch.second[chosen])
        v1 = velocities[pp].copy()
        v2 = velocities[qq].copy()
        center = 0.5 * (v1 + v2)
        relative = v1 - v2
        magnitude = xp.sqrt(xp.sum(relative * relative, axis=1))
        u1 = backend.uniform(len(chosen))
        u2 = backend.uniform(len(chosen))
        cos_theta = 2.0 * u1 - 1.0
        sin_theta = xp.sqrt(xp.maximum(0.0, 1.0 - cos_theta * cos_theta))
        phi = 2.0 * math.pi * u2
        direction = xp.stack(
            (sin_theta * xp.cos(phi), sin_theta * xp.sin(phi), cos_theta), axis=1
        )
        scattered = magnitude[:, None] * direction
        velocities[pp] = center + 0.5 * scattered
        velocities[qq] = center - 0.5 * scattered
    return stats


class CollisionEngine:
    def __init__(self, config, backend, rng: np.random.Generator):
        model = config.model.lower()
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown collision model {model}; choose from {SUPPORTED_MODELS}")
        self.model = model
        self.config = config
        self.backend = backend
        self.rng = rng
        self.majorants = np.full(config.nx * config.ny, config.conservative_sigma_g())
        self.total = CollisionStats()

    def collide(self, groups: list[np.ndarray], positions, velocities, dt: float) -> CollisionStats:
        positions_np = self.backend.asnumpy(positions)
        if self.model in {"ntc", "ntc-prescan", "mfs"}:
            batch, self.majorants = generate_majorant_trials(
                self.model,
                groups,
                velocities,
                self.config,
                self.backend,
                self.rng,
                self.majorants,
                dt,
            )
        else:
            batch = generate_bt_trials(
                self.model, groups, positions_np, self.config, self.rng
            )
        # Record any newly observed majorant before scattering changes the
        # relative velocity.  It cannot repair the already-selected trial
        # count for this step, so strict mode still rejects an invalid
        # probability; it does make the persistent bound valid for later
        # steps when the observed value remains below the active bound.
        if batch.majorant is not None and len(batch.first):
            xp = self.backend.xp
            p = xp.asarray(batch.first)
            q = xp.asarray(batch.second)
            rel = velocities[p] - velocities[q]
            sg = self.backend.asnumpy(
                vhs_sigma_g(xp.sqrt(xp.sum(rel * rel, axis=1)), self.config, xp)
            )
            for cell, value in zip(batch.cell, sg):
                if value > self.majorants[cell]:
                    self.majorants[cell] = 1.05 * value
        stats = apply_trials(batch, velocities, self.config, self.backend, dt)
        self.total.add(stats)
        return stats
