from __future__ import annotations

import numpy as np

from .vhs_model import (
    KB,
    PhysicalCavityConfig,
    PhysicalParticleState,
    VHSModel,
    _cell_xy,
    apply_diffuse_walls,
    initialize_physical_state,
)


def _weighted_elastic_scatter(
    vel: np.ndarray,
    weight: np.ndarray,
    a: int,
    b: int,
    rng: np.random.Generator,
) -> None:
    wa, wb = float(weight[a]), float(weight[b])
    total = wa + wb
    center = (wa * vel[a] + wb * vel[b]) / total
    relative = vel[a] - vel[b]
    speed = float(np.linalg.norm(relative))
    if speed <= 0.0:
        return
    cos_chi = 2.0 * rng.random() - 1.0
    sin_chi = np.sqrt(max(0.0, 1.0 - cos_chi**2))
    phi = 2.0 * np.pi * rng.random()
    rel_new = speed * np.array([
        sin_chi * np.cos(phi),
        sin_chi * np.sin(phi),
        cos_chi,
    ])
    vel[a] = center + (wb / total) * rel_new
    vel[b] = center - (wa / total) * rel_new


def equalize_cell_weights(
    state: PhysicalParticleState,
    cfg: PhysicalCavityConfig,
    rng: np.random.Generator,
) -> None:
    """Make weights uniform inside each collision cell while preserving moments."""
    ix, iy = _cell_xy(state.pos, cfg)
    cell = iy * cfg.nx + ix
    for cid in range(cfg.nx * cfg.ny):
        ids = np.flatnonzero(cell == cid)
        n = len(ids)
        if n < 2:
            continue
        old_w = state.weight[ids].copy()
        if np.max(old_w) - np.min(old_w) <= 1.0e-12 * max(float(np.mean(old_w)), 1.0):
            continue
        total_w = float(old_w.sum())
        old_mean = np.sum(old_w[:, None] * state.vel[ids], axis=0) / total_w
        centered_old = state.vel[ids] - old_mean
        old_thermal = float(np.sum(old_w * np.sum(centered_old**2, axis=1)))
        probs = old_w / total_w
        chosen = rng.choice(ids, size=n, replace=True, p=probs)
        velocity = state.vel[chosen].copy()
        new_w = total_w / n
        center = velocity - velocity.mean(axis=0)
        new_thermal = float(new_w * np.sum(center**2))
        if old_thermal <= 1.0e-30:
            velocity[:] = old_mean
        elif new_thermal > 1.0e-30:
            velocity = old_mean + center * np.sqrt(old_thermal / new_thermal)
        else:
            perturb = rng.normal(size=velocity.shape)
            perturb -= perturb.mean(axis=0)
            denom = float(new_w * np.sum(perturb**2))
            velocity = old_mean + perturb * np.sqrt(old_thermal / max(denom, 1.0e-30))
        state.vel[ids] = velocity
        state.weight[ids] = new_w


def collide_vhs_sbt(
    state: PhysicalParticleState,
    cfg: PhysicalCavityConfig,
    rng: np.random.Generator,
) -> int:
    """Run VHS collisions with SBT pairing and adaptive two-dimensional subcells.

    The candidate-pair probability is the spatially two-dimensional extension of
    the repository's ``Parallel_TAS.py`` expression:
    ``remaining * FNUM * dt * sigma(g) * g / subcell_volume``.
    """
    equalize_cell_weights(state, cfg, rng)
    ix, iy = _cell_xy(state.pos, cfg)
    cell = iy * cfg.nx + ix
    order = np.argsort(cell)
    sorted_cell = cell[order]
    starts = np.searchsorted(sorted_cell, np.arange(cfg.nx * cfg.ny), side="left")
    ends = np.searchsorted(sorted_cell, np.arange(cfg.nx * cfg.ny), side="right")
    dx, dy = cfg.length / cfg.nx, cfg.length / cfg.ny
    base_fnum = cfg.real_particles_per_sim_particle
    accepted = 0

    for cid, (start, end) in enumerate(zip(starts, ends)):
        ids = order[start:end]
        count = len(ids)
        if count < 2:
            continue
        subdivisions = int(np.ceil(np.sqrt(count / max(cfg.target_ppc_subcell, 1))))
        subdivisions = int(np.clip(subdivisions, 1, cfg.max_subdivisions))
        cx, cy = cid % cfg.nx, cid // cfg.nx
        x0, y0 = cx * dx, cy * dy
        sx = np.clip(
            ((state.pos[ids, 0] - x0) / dx * subdivisions).astype(int),
            0,
            subdivisions - 1,
        )
        sy = np.clip(
            ((state.pos[ids, 1] - y0) / dy * subdivisions).astype(int),
            0,
            subdivisions - 1,
        )
        subid = sy * subdivisions + sx
        sub_volume = cfg.cell_volume / subdivisions**2

        for sid in range(subdivisions**2):
            group = ids[subid == sid]
            n = len(group)
            if n < 2:
                continue
            perm = rng.permutation(group)
            for i in range(n - 1):
                a = int(perm[i])
                j = i + 1 + int(rng.integers(0, n - i - 1))
                b = int(perm[j])
                relative = state.vel[a] - state.vel[b]
                g = float(np.linalg.norm(relative))
                if g <= 1.0e-12:
                    continue
                sigma = cfg.vhs.cross_section(g)
                remaining = float(n - i - 1)
                pair_fnum = base_fnum * 0.5 * (
                    float(state.weight[a]) + float(state.weight[b])
                )
                probability = remaining * pair_fnum * cfg.dt * sigma * g / sub_volume
                if rng.random() < min(1.0, probability):
                    _weighted_elastic_scatter(state.vel, state.weight, a, b, rng)
                    accepted += 1
    return accepted


def sample_physical_state(
    state: PhysicalParticleState,
    cfg: PhysicalCavityConfig,
) -> dict[str, np.ndarray]:
    ix, iy = _cell_xy(state.pos, cfg)
    cell = iy * cfg.nx + ix
    ncell = cfg.nx * cfg.ny
    represented = cfg.real_particles_per_sim_particle * state.weight
    number = np.bincount(cell, weights=represented, minlength=ncell)
    count = np.bincount(cell, minlength=ncell).astype(float)
    sx = np.bincount(cell, weights=represented * state.vel[:, 0], minlength=ncell)
    sy = np.bincount(cell, weights=represented * state.vel[:, 1], minlength=ncell)
    sz = np.bincount(cell, weights=represented * state.vel[:, 2], minlength=ncell)
    s2 = np.bincount(
        cell,
        weights=represented * np.sum(state.vel**2, axis=1),
        minlength=ncell,
    )
    safe = np.maximum(number, 1.0e-200)
    u, v, w = sx / safe, sy / safe, sz / safe
    peculiar2 = np.maximum(s2 / safe - u**2 - v**2 - w**2, 0.0)
    temperature = cfg.vhs.mass * peculiar2 / (3.0 * KB)
    shape = (cfg.ny, cfg.nx)
    return {
        "number_density": (number / cfg.cell_volume).reshape(shape),
        "rho": (number / max(float(number.mean()), 1.0e-200)).reshape(shape),
        "u": u.reshape(shape),
        "v": v.reshape(shape),
        "w": w.reshape(shape),
        "T": temperature.reshape(shape),
        "count": count.reshape(shape),
        "represented": number.reshape(shape),
    }


def advance_physical_state(
    state: PhysicalParticleState,
    cfg: PhysicalCavityConfig,
    steps: int,
    sample_start: int = 0,
    seed: int | None = None,
) -> tuple[dict[str, np.ndarray], PhysicalParticleState, dict[str, float]]:
    if steps <= 0 or not 0 <= sample_start < steps:
        raise ValueError("Require steps > 0 and 0 <= sample_start < steps")
    rng = np.random.default_rng(cfg.seed + 104729 if seed is None else seed)
    sums: dict[str, np.ndarray] = {}
    sums2: dict[str, np.ndarray] = {}
    nsamples = 0
    collisions = 0
    for step in range(steps):
        state.pos += state.vel[:, :2] * cfg.dt
        apply_diffuse_walls(state, cfg, rng)
        collisions += collide_vhs_sbt(state, cfg, rng)
        if step >= sample_start:
            fields = sample_physical_state(state, cfg)
            for key, value in fields.items():
                sums[key] = sums.get(key, np.zeros_like(value)) + value
                sums2[key] = sums2.get(key, np.zeros_like(value)) + value**2
            nsamples += 1
    out = {key: value / nsamples for key, value in sums.items()}
    for key in ("T", "u", "v", "w"):
        out[f"sigma_{key}"] = np.sqrt(
            np.maximum(sums2[key] / nsamples - out[key] ** 2, 0.0)
        )
    diagnostics = {
        "accepted_collisions": float(collisions),
        "collisions_per_particle_step": float(collisions / (len(state.pos) * steps)),
        "dt": cfg.dt,
        "number_density": cfg.number_density,
        "mean_free_path": cfg.vhs.mean_free_path(cfg.number_density, cfg.t0),
    }
    return out, state, diagnostics


def run_physical_cavity(
    cfg: PhysicalCavityConfig,
    initial_state: PhysicalParticleState | None = None,
    return_state: bool = False,
):
    state = initialize_physical_state(cfg) if initial_state is None else initial_state.copy()
    fields, state, diagnostics = advance_physical_state(
        state,
        cfg,
        cfg.steps,
        cfg.sample_start,
        seed=cfg.seed + 17,
    )
    if return_state:
        return fields, state, diagnostics
    return fields


def conserved_quantities(
    state: PhysicalParticleState,
    model: VHSModel = VHSModel(),
) -> tuple[float, np.ndarray, float]:
    mass = float(np.sum(state.weight))
    momentum = np.sum(state.weight[:, None] * state.vel, axis=0)
    energy = 0.5 * model.mass * float(
        np.sum(state.weight * np.sum(state.vel**2, axis=1))
    )
    return mass, momentum, energy
