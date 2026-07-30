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


@dataclass
class ParticleState:
    pos: np.ndarray
    vel: np.ndarray
    weight: np.ndarray

    def copy(self) -> "ParticleState":
        return ParticleState(
            self.pos.copy(),
            self.vel.copy(),
            self.weight.copy(),
        )


def initialize_state(cfg: CavityConfig) -> ParticleState:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.nx * cfg.ny * cfg.particles_per_cell
    return ParticleState(
        pos=rng.random((n, 2)),
        vel=rng.normal(0.0, 1.0, (n, 2)),
        weight=np.ones(n, dtype=np.float64),
    )


def _wall_temperature(
    x: np.ndarray,
    wall: str,
    cfg: CavityConfig,
) -> np.ndarray:
    if wall == "left":
        return np.full_like(x, cfg.t_left)
    if wall == "right":
        return np.full_like(x, cfg.t_right)
    if wall == "top":
        return np.full_like(x, cfg.t_top)
    return np.full_like(x, cfg.t_bottom)


def _diffuse_reflect(
    velocity: np.ndarray,
    temperature: np.ndarray,
    normal_axis: int,
    sign: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return reflected velocities; fancy indexing is not a write view."""
    reflected = velocity.copy()
    sigma = np.sqrt(temperature)
    tangential_axis = 1 - normal_axis
    reflected[:, tangential_axis] = rng.normal(0.0, sigma)
    reflected[:, normal_axis] = sign * sigma * np.sqrt(
        -2.0
        * np.log(
            np.maximum(rng.random(len(reflected)), 1.0e-12)
        )
    )
    return reflected


def _apply_walls(
    state: ParticleState,
    cfg: CavityConfig,
    rng: np.random.Generator,
) -> None:
    pos, vel = state.pos, state.vel
    for _ in range(4):
        masks = {
            "left": pos[:, 0] < 0.0,
            "right": pos[:, 0] >= 1.0,
            "bottom": pos[:, 1] < 0.0,
            "top": pos[:, 1] >= 1.0,
        }
        if not any(np.any(mask) for mask in masks.values()):
            break
        for wall, mask in masks.items():
            if not np.any(mask):
                continue
            ids = np.flatnonzero(mask)
            if wall == "left":
                pos[ids, 0] *= -1.0
                vel[ids] = _diffuse_reflect(
                    vel[ids],
                    _wall_temperature(pos[ids, 1], wall, cfg),
                    0,
                    +1.0,
                    rng,
                )
            elif wall == "right":
                pos[ids, 0] = 2.0 - pos[ids, 0]
                vel[ids] = _diffuse_reflect(
                    vel[ids],
                    _wall_temperature(pos[ids, 1], wall, cfg),
                    0,
                    -1.0,
                    rng,
                )
            elif wall == "bottom":
                pos[ids, 1] *= -1.0
                vel[ids] = _diffuse_reflect(
                    vel[ids],
                    _wall_temperature(pos[ids, 0], wall, cfg),
                    1,
                    +1.0,
                    rng,
                )
            else:
                pos[ids, 1] = 2.0 - pos[ids, 1]
                vel[ids] = _diffuse_reflect(
                    vel[ids],
                    _wall_temperature(pos[ids, 0], wall, cfg),
                    1,
                    -1.0,
                    rng,
                )
    pos[:] = np.clip(pos, 0.0, 1.0 - 1.0e-12)


def _cell_ids(pos: np.ndarray, cfg: CavityConfig) -> np.ndarray:
    ix = np.clip(
        (pos[:, 0] * cfg.nx).astype(np.int64),
        0,
        cfg.nx - 1,
    )
    iy = np.clip(
        (pos[:, 1] * cfg.ny).astype(np.int64),
        0,
        cfg.ny - 1,
    )
    return iy * cfg.nx + ix


def _collide(
    state: ParticleState,
    cfg: CavityConfig,
    rng: np.random.Generator,
) -> None:
    """Simplified weighted pair collision conserving pair invariants."""
    cell = _cell_ids(state.pos, cfg)
    order = np.argsort(cell)
    sorted_cell = cell[order]
    starts = np.searchsorted(
        sorted_cell,
        np.arange(cfg.nx * cfg.ny),
        side="left",
    )
    ends = np.searchsorted(
        sorted_cell,
        np.arange(cfg.nx * cfg.ny),
        side="right",
    )
    global_mean_weight = max(
        float(np.mean(state.weight)),
        1.0e-12,
    )

    for start, end in zip(starts, ends):
        ids = order[start:end]
        if len(ids) < 2:
            continue
        shuffled = rng.permutation(ids)
        for k in range(len(shuffled) // 2):
            a = int(shuffled[2 * k])
            b = int(shuffled[2 * k + 1])
            rate_scale = max(
                state.weight[a],
                state.weight[b],
            ) / global_mean_weight
            if rng.random() > min(
                1.0,
                cfg.collision_rate * rate_scale,
            ):
                continue
            wa = float(state.weight[a])
            wb = float(state.weight[b])
            total_weight = wa + wb
            center = (
                wa * state.vel[a] + wb * state.vel[b]
            ) / total_weight
            relative = state.vel[a] - state.vel[b]
            speed = float(np.linalg.norm(relative))
            theta = 2.0 * np.pi * rng.random()
            relative_new = speed * np.array(
                [np.cos(theta), np.sin(theta)]
            )
            state.vel[a] = (
                center + (wb / total_weight) * relative_new
            )
            state.vel[b] = (
                center - (wa / total_weight) * relative_new
            )


def sample_state(
    state: ParticleState,
    cfg: CavityConfig,
) -> dict[str, np.ndarray]:
    cell = _cell_ids(state.pos, cfg)
    ncell = cfg.nx * cfg.ny
    mass = np.bincount(
        cell,
        weights=state.weight,
        minlength=ncell,
    )
    count = np.bincount(
        cell,
        minlength=ncell,
    ).astype(np.float64)
    sx = np.bincount(
        cell,
        weights=state.weight * state.vel[:, 0],
        minlength=ncell,
    )
    sy = np.bincount(
        cell,
        weights=state.weight * state.vel[:, 1],
        minlength=ncell,
    )
    s2 = np.bincount(
        cell,
        weights=state.weight * np.sum(state.vel**2, axis=1),
        minlength=ncell,
    )
    safe_mass = np.maximum(mass, 1.0e-14)
    u = sx / safe_mass
    v = sy / safe_mass
    temperature = np.maximum(
        0.5 * (s2 / safe_mass - u**2 - v**2),
        0.0,
    )
    mean_mass = max(float(np.mean(mass)), 1.0e-14)
    shape = (cfg.ny, cfg.nx)
    return {
        "rho": (mass / mean_mass).reshape(shape),
        "u": u.reshape(shape),
        "v": v.reshape(shape),
        "T": temperature.reshape(shape),
        "count": count.reshape(shape),
        "mass": mass.reshape(shape),
    }


def advance_state(
    state: ParticleState,
    cfg: CavityConfig,
    steps: int,
    sample_start: int = 0,
    seed: int | None = None,
) -> tuple[dict[str, np.ndarray], ParticleState]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0 <= sample_start < steps:
        raise ValueError(
            "sample_start must satisfy 0 <= sample_start < steps"
        )
    rng = np.random.default_rng(
        cfg.seed + 7919 if seed is None else seed
    )
    sums: dict[str, np.ndarray] = {}
    sums2: dict[str, np.ndarray] = {}
    nsamples = 0

    for step in range(steps):
        state.pos += state.vel * cfg.dt
        _apply_walls(state, cfg, rng)
        _collide(state, cfg, rng)
        if step >= sample_start:
            fields = sample_state(state, cfg)
            for key, value in fields.items():
                sums[key] = (
                    sums.get(key, np.zeros_like(value)) + value
                )
                sums2[key] = (
                    sums2.get(key, np.zeros_like(value))
                    + value**2
                )
            nsamples += 1

    out = {key: value / nsamples for key, value in sums.items()}
    for key in ("T", "u", "v"):
        mean = out[key]
        out[f"sigma_{key}"] = np.sqrt(
            np.maximum(
                sums2[key] / nsamples - mean**2,
                0.0,
            )
        )
    return out, state


def run_cavity(
    cfg: CavityConfig,
    initial_state: ParticleState | None = None,
    return_state: bool = False,
):
    state = (
        initialize_state(cfg)
        if initial_state is None
        else initial_state.copy()
    )
    fields, state = advance_state(
        state,
        cfg,
        steps=cfg.steps,
        sample_start=cfg.sample_start,
        seed=cfg.seed + 17,
    )
    return (fields, state) if return_state else fields
