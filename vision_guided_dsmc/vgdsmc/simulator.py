from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CavityConfig:
    nx: int = 24
    ny: int = 24
    particles_per_cell: int = 20
    steps: int = 300
    sample_start: int = 120
    dt: float = 2.5e-3
    t_left: float = 1.10
    t_right: float = 0.90
    t_top: float = 1.00
    t_bottom: float = 1.00
    collision_rate: float = 0.20
    seed: int = 7


def _wall_temperature(x: np.ndarray, wall: str, cfg: CavityConfig) -> np.ndarray:
    if wall == "left":
        return np.full_like(x, cfg.t_left)
    if wall == "right":
        return np.full_like(x, cfg.t_right)
    if wall == "top":
        return np.full_like(x, cfg.t_top)
    return np.full_like(x, cfg.t_bottom)


def _diffuse_reflect(v: np.ndarray, temperature: np.ndarray, normal_axis: int, sign: float, rng: np.random.Generator) -> None:
    sigma = np.sqrt(temperature)
    tangential_axis = 1 - normal_axis
    v[:, tangential_axis] = rng.normal(0.0, sigma)
    v[:, normal_axis] = sign * sigma * np.sqrt(-2.0 * np.log(np.maximum(rng.random(len(v)), 1e-12)))


def _apply_walls(pos: np.ndarray, vel: np.ndarray, cfg: CavityConfig, rng: np.random.Generator) -> None:
    masks = {"left": pos[:, 0] < 0.0, "right": pos[:, 0] > 1.0, "bottom": pos[:, 1] < 0.0, "top": pos[:, 1] > 1.0}
    for wall, mask in masks.items():
        if not np.any(mask):
            continue
        ids = np.flatnonzero(mask)
        if wall == "left":
            pos[ids, 0] *= -1.0
            _diffuse_reflect(vel[ids], _wall_temperature(pos[ids, 1], wall, cfg), 0, +1.0, rng)
        elif wall == "right":
            pos[ids, 0] = 2.0 - pos[ids, 0]
            _diffuse_reflect(vel[ids], _wall_temperature(pos[ids, 1], wall, cfg), 0, -1.0, rng)
        elif wall == "bottom":
            pos[ids, 1] *= -1.0
            _diffuse_reflect(vel[ids], _wall_temperature(pos[ids, 0], wall, cfg), 1, +1.0, rng)
        else:
            pos[ids, 1] = 2.0 - pos[ids, 1]
            _diffuse_reflect(vel[ids], _wall_temperature(pos[ids, 0], wall, cfg), 1, -1.0, rng)


def _collide(pos: np.ndarray, vel: np.ndarray, cfg: CavityConfig, rng: np.random.Generator) -> None:
    ix = np.clip((pos[:, 0] * cfg.nx).astype(int), 0, cfg.nx - 1)
    iy = np.clip((pos[:, 1] * cfg.ny).astype(int), 0, cfg.ny - 1)
    cell = iy * cfg.nx + ix
    order = np.argsort(cell)
    sorted_cell = cell[order]
    starts = np.searchsorted(sorted_cell, np.arange(cfg.nx * cfg.ny), side="left")
    ends = np.searchsorted(sorted_cell, np.arange(cfg.nx * cfg.ny), side="right")
    for start, end in zip(starts, ends):
        ids = order[start:end]
        if len(ids) < 2:
            continue
        shuffled = rng.permutation(ids)
        for k in range(len(shuffled) // 2):
            if rng.random() > cfg.collision_rate:
                continue
            a, b = shuffled[2 * k], shuffled[2 * k + 1]
            vcm = 0.5 * (vel[a] + vel[b])
            rel = vel[a] - vel[b]
            speed = np.linalg.norm(rel)
            theta = 2.0 * np.pi * rng.random()
            rel_new = speed * np.array([np.cos(theta), np.sin(theta)])
            vel[a] = vcm + 0.5 * rel_new
            vel[b] = vcm - 0.5 * rel_new


def _sample(pos: np.ndarray, vel: np.ndarray, cfg: CavityConfig) -> dict[str, np.ndarray]:
    ix = np.clip((pos[:, 0] * cfg.nx).astype(int), 0, cfg.nx - 1)
    iy = np.clip((pos[:, 1] * cfg.ny).astype(int), 0, cfg.ny - 1)
    cell = iy * cfg.nx + ix
    ncell = cfg.nx * cfg.ny
    count = np.bincount(cell, minlength=ncell).astype(float)
    sx = np.bincount(cell, weights=vel[:, 0], minlength=ncell)
    sy = np.bincount(cell, weights=vel[:, 1], minlength=ncell)
    s2 = np.bincount(cell, weights=np.sum(vel**2, axis=1), minlength=ncell)
    safe = np.maximum(count, 1.0)
    u = sx / safe
    v = sy / safe
    temp = np.maximum(0.5 * (s2 / safe - u**2 - v**2), 0.0)
    shape = (cfg.ny, cfg.nx)
    return {"rho": (count / np.mean(count)).reshape(shape), "u": u.reshape(shape), "v": v.reshape(shape), "T": temp.reshape(shape), "count": count.reshape(shape)}


def run_cavity(cfg: CavityConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.nx * cfg.ny * cfg.particles_per_cell
    pos = rng.random((n, 2))
    vel = rng.normal(0.0, 1.0, (n, 2))
    sums: dict[str, np.ndarray] = {}
    sums2: dict[str, np.ndarray] = {}
    nsamples = 0
    for step in range(cfg.steps):
        pos += vel * cfg.dt
        _apply_walls(pos, vel, cfg, rng)
        _collide(pos, vel, cfg, rng)
        if step >= cfg.sample_start:
            fields = _sample(pos, vel, cfg)
            for key, value in fields.items():
                sums[key] = sums.get(key, np.zeros_like(value)) + value
                sums2[key] = sums2.get(key, np.zeros_like(value)) + value**2
            nsamples += 1
    if nsamples == 0:
        raise ValueError("sample_start must be less than steps")
    out = {key: value / nsamples for key, value in sums.items()}
    for key in ("T", "u", "v"):
        mean = out[key]
        out[f"sigma_{key}"] = np.sqrt(np.maximum(sums2[key] / nsamples - mean**2, 0.0))
    return out
