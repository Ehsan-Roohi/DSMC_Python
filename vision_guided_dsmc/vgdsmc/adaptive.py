from __future__ import annotations

import numpy as np

CLASS_MULTIPLIERS = np.array([0.5, 1.0, 3.0], dtype=np.float32)


def label_to_target_ppc(
    label: np.ndarray,
    base_ppc: int = 20,
    min_ppc: int = 4,
    max_ppc: int = 200,
) -> np.ndarray:
    """Convert three allocation classes into a target particles-per-cell map."""
    target = np.rint(base_ppc * CLASS_MULTIPLIERS[label]).astype(np.int64)
    return np.clip(target, min_ppc, max_ppc)


def reallocate_particles(
    pos: np.ndarray,
    vel: np.ndarray,
    target_ppc: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split or merge particles cell by cell to match a target PPC map.

    Cloned particles retain the parent velocity and receive a small position
    perturbation. This is a pilot resampling method; conservation-aware weighted
    particles will replace it in the high-fidelity DSMC coupling stage.
    """
    rng = np.random.default_rng() if rng is None else rng
    ny, nx = target_ppc.shape
    ix = np.clip((pos[:, 0] * nx).astype(int), 0, nx - 1)
    iy = np.clip((pos[:, 1] * ny).astype(int), 0, ny - 1)
    new_pos, new_vel = [], []

    for j in range(ny):
        for i in range(nx):
            ids = np.flatnonzero((ix == i) & (iy == j))
            target = int(target_ppc[j, i])
            if len(ids) == 0:
                p = np.column_stack(
                    ((i + rng.random(target)) / nx, (j + rng.random(target)) / ny)
                )
                v = rng.normal(0.0, 1.0, (target, 2))
            else:
                chosen = rng.choice(ids, size=target, replace=target > len(ids))
                p = pos[chosen].copy()
                v = vel[chosen].copy()
                if target > len(ids):
                    p += rng.normal(0.0, 0.02 / min(nx, ny), p.shape)
                    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0 - 1.0e-12)
                    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0 - 1.0e-12)
            new_pos.append(p)
            new_vel.append(v)

    return np.vstack(new_pos), np.vstack(new_vel)


def allocation_summary(label: np.ndarray, base_ppc: int = 20) -> dict[str, float | int]:
    target = label_to_target_ppc(label, base_ppc)
    uniform = label.size * base_ppc
    return {
        "uniform_particles": int(uniform),
        "adaptive_particles": int(target.sum()),
        "ratio": float(target.sum() / uniform),
    }
