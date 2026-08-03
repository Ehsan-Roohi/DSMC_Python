"""Two-dimensional lid-driven cavity DSMC solver."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

from .backend import ArrayBackend
from .collisions import CollisionEngine
from .config import K_B, SimulationConfig


class MomentAccumulator:
    def __init__(self, n_cells: int, xp: Any):
        self.xp = xp
        self.samples = 0
        self.names = (
            "count", "vx", "vy", "vz", "vxx", "vyy", "vzz",
            "vxy", "vxz", "vyz", "v2vx", "v2vy",
        )
        self.data = {name: xp.zeros(n_cells, dtype=xp.float64) for name in self.names}

    def sample(self, cell_id, velocities) -> None:
        xp = self.xp
        n_cells = len(self.data["count"])
        vx, vy, vz = velocities[:, 0], velocities[:, 1], velocities[:, 2]
        v2 = vx * vx + vy * vy + vz * vz
        values = {
            "count": xp.ones_like(vx),
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "vxx": vx * vx,
            "vyy": vy * vy,
            "vzz": vz * vz,
            "vxy": vx * vy,
            "vxz": vx * vz,
            "vyz": vy * vz,
            "v2vx": v2 * vx,
            "v2vy": v2 * vy,
        }
        for name, value in values.items():
            self.data[name] += xp.bincount(cell_id, weights=value, minlength=n_cells)
        self.samples += 1

    def finalize(self, config: SimulationConfig, backend: ArrayBackend) -> dict[str, np.ndarray]:
        raw = {name: backend.asnumpy(value) for name, value in self.data.items()}
        count = np.maximum(raw["count"], 1.0)
        mean = {name: raw[name] / count for name in self.names if name != "count"}
        u, v, w = mean["vx"], mean["vy"], mean["vz"]
        v2 = mean["vxx"] + mean["vyy"] + mean["vzz"]
        c2 = np.maximum(0.0, v2 - u * u - v * v - w * w)
        temperature = config.mass * c2 / (3.0 * K_B)
        number_density = (
            config.particle_weight * raw["count"] / (self.samples * config.cell_volume)
        )
        rho = config.mass * number_density
        pxy = config.mass * number_density * (mean["vxy"] - u * v)
        # Raw-to-central third-moment conversion for q_i = 1/2 m n <c^2 c_i>.
        cv2cx = (
            mean["v2vx"]
            - u * v2
            - 2.0 * (u * mean["vxx"] + v * mean["vxy"] + w * mean["vxz"])
            + 2.0 * u * (u * u + v * v + w * w)
        )
        cv2cy = (
            mean["v2vy"]
            - v * v2
            - 2.0 * (u * mean["vxy"] + v * mean["vyy"] + w * mean["vyz"])
            + 2.0 * v * (u * u + v * v + w * w)
        )
        qx = 0.5 * config.mass * number_density * cv2cx
        qy = 0.5 * config.mass * number_density * cv2cy
        return {
            "sample_count": raw["count"],
            "number_density": number_density,
            "rho": rho,
            "u": u,
            "v": v,
            "w": w,
            "temperature": temperature,
            "pxy": pxy,
            "qx": qx,
            "qy": qy,
        }


class CavitySolver:
    def __init__(self, config: SimulationConfig):
        config.validate()
        self.config = config
        self.dt, self.dt_limits = config.resolved_dt()
        self.backend = ArrayBackend.create(config.backend, config.seed)
        self.xp = self.backend.xp
        self.pair_rng = np.random.default_rng(config.seed + 7919)
        self.collision = CollisionEngine(config, self.backend, self.pair_rng)
        self.positions, self.velocities = self._initialize_particles()
        self.moments = MomentAccumulator(config.nx * config.ny, self.xp)
        self.wall_hits = {"left": 0, "right": 0, "bottom": 0, "top": 0}
        self._top_weight = np.zeros(config.nx)
        self._top_u_weighted = np.zeros(config.nx)
        self._top_v2_weighted = np.zeros(config.nx)
        self.history: list[dict[str, float]] = []

    def _initialize_particles(self):
        c = self.config
        xp = self.xp
        n = c.nx * c.ny * c.particles_per_cell
        ids = np.arange(n, dtype=np.int64)
        cell = ids // c.particles_per_cell
        ix = cell % c.nx
        iy = cell // c.nx
        jitter = self.pair_rng.random((n, 2))
        position_np = np.column_stack(((ix + jitter[:, 0]) * c.dx, (iy + jitter[:, 1]) * c.dy))
        sigma = math.sqrt(K_B * c.wall_temperature / c.mass)
        velocity_np = self.pair_rng.normal(0.0, sigma, size=(n, 3))
        return xp.asarray(position_np), xp.asarray(velocity_np)

    def _cell_ids(self):
        xp = self.xp
        c = self.config
        ix = xp.clip((self.positions[:, 0] / c.dx).astype(xp.int64), 0, c.nx - 1)
        iy = xp.clip((self.positions[:, 1] / c.dy).astype(xp.int64), 0, c.ny - 1)
        return ix + c.nx * iy

    def _groups(self, cell_id) -> list[np.ndarray]:
        cell_np = self.backend.asnumpy(cell_id).astype(np.int64, copy=False)
        order = np.argsort(cell_np, kind="stable")
        counts = np.bincount(cell_np, minlength=self.config.nx * self.config.ny)
        cuts = np.cumsum(counts)
        starts = np.r_[0, cuts[:-1]]
        return [order[start:stop] for start, stop in zip(starts, cuts)]

    def _diffuse(self, mask, normal_axis: int, sign: float, tangential_mean: tuple[float, float]):
        xp = self.xp
        c = self.config
        indices = xp.flatnonzero(mask)
        count = int(indices.size)
        if count == 0:
            return indices
        sigma = math.sqrt(K_B * c.wall_temperature / c.mass)
        half = sigma * xp.sqrt(-2.0 * xp.log(xp.maximum(self.backend.uniform(count), 1e-15)))
        tangential = sigma * self.backend.normal((count, 2))
        if normal_axis == 0:
            self.velocities[indices, 0] = sign * half
            self.velocities[indices, 1] = tangential_mean[0] + tangential[:, 0]
            self.velocities[indices, 2] = tangential_mean[1] + tangential[:, 1]
        else:
            self.velocities[indices, 1] = sign * half
            self.velocities[indices, 0] = tangential_mean[0] + tangential[:, 0]
            self.velocities[indices, 2] = tangential_mean[1] + tangential[:, 1]
        return indices

    def _move_and_reflect(self, collect_wall: bool) -> None:
        xp = self.xp
        c = self.config
        remaining = xp.full(len(self.positions), self.dt, dtype=xp.float64)
        tolerance = 32.0 * np.finfo(float).eps * self.dt

        # Event-driven wall motion preserves the post-collision flight during
        # the remainder of the step.  The short DSMC step normally gives zero
        # or one event; the loop also resolves rare corner/multiple hits in
        # their true temporal order.
        for _ in range(8):
            vx = self.velocities[:, 0]
            vy = self.velocities[:, 1]
            speed_x = xp.maximum(xp.abs(vx), 1e-30)
            speed_y = xp.maximum(xp.abs(vy), 1e-30)
            tx_left = xp.where(vx < 0.0, self.positions[:, 0] / speed_x, xp.inf)
            tx_right = xp.where(
                vx > 0.0, (c.length - self.positions[:, 0]) / speed_x, xp.inf
            )
            ty_bottom = xp.where(vy < 0.0, self.positions[:, 1] / speed_y, xp.inf)
            ty_top = xp.where(
                vy > 0.0, (c.length - self.positions[:, 1]) / speed_y, xp.inf
            )
            tx = xp.minimum(tx_left, tx_right)
            ty = xp.minimum(ty_bottom, ty_top)
            hit_time = xp.minimum(tx, ty)
            collides = hit_time <= remaining + tolerance
            travel = xp.where(collides, xp.maximum(hit_time, 0.0), remaining)
            self.positions += self.velocities[:, :2] * travel[:, None]
            remaining = xp.where(collides, xp.maximum(0.0, remaining - travel), 0.0)

            if not bool(self.backend.asnumpy(xp.any(collides))):
                break

            hit_x = collides & (tx <= ty)
            left = hit_x & (vx < 0.0)
            right = hit_x & (vx > 0.0)
            bottom = collides & ~hit_x & (vy < 0.0)
            top = collides & ~hit_x & (vy > 0.0)

            self.positions[left, 0] = 0.0
            ids = self._diffuse(left, 0, +1.0, (0.0, 0.0))
            self.wall_hits["left"] += int(ids.size)

            self.positions[right, 0] = c.length
            ids = self._diffuse(right, 0, -1.0, (0.0, 0.0))
            self.wall_hits["right"] += int(ids.size)

            self.positions[bottom, 1] = 0.0
            ids = self._diffuse(bottom, 1, +1.0, (0.0, 0.0))
            self.wall_hits["bottom"] += int(ids.size)

            top_indices = xp.flatnonzero(top)
            if collect_wall and int(top_indices.size):
                incoming = self.backend.asnumpy(self.velocities[top_indices])
                x_hit = self.backend.asnumpy(self.positions[top_indices, 0])
                bins = np.clip((x_hit / c.dx).astype(int), 0, c.nx - 1)
                weight = 1.0 / np.maximum(np.abs(incoming[:, 1]), 1e-12)
                self._top_weight += np.bincount(bins, weights=weight, minlength=c.nx)
                self._top_u_weighted += np.bincount(
                    bins, weights=weight * incoming[:, 0], minlength=c.nx
                )
                self._top_v2_weighted += np.bincount(
                    bins,
                    weights=weight * np.sum(incoming * incoming, axis=1),
                    minlength=c.nx,
                )
            self.positions[top, 1] = c.length
            ids = self._diffuse(top, 1, -1.0, (c.lid_velocity, 0.0))
            self.wall_hits["top"] += int(ids.size)
        else:
            if bool(self.backend.asnumpy(xp.any(remaining > tolerance))):
                raise RuntimeError(
                    "A particle had more than eight wall events in one step; reduce dt."
                )

        invalid = xp.any((self.positions < -c.length * 1e-12) | (self.positions > c.length * (1.0 + 1e-12)))
        if bool(self.backend.asnumpy(invalid)):
            raise RuntimeError("Particle left the cavity after wall processing; reduce dt.")
        self.positions[:] = xp.clip(self.positions, 0.0, c.length)

    def _diagnostic(self, step: int, stats) -> None:
        xp = self.xp
        mean_v = xp.mean(self.velocities, axis=0)
        mean_t = self.config.mass * xp.mean(
            xp.sum((self.velocities - mean_v) ** 2, axis=1)
        ) / (3.0 * K_B)
        self.history.append(
            {
                "step": int(step),
                "time": float(step * self.dt),
                "temperature": float(self.backend.asnumpy(mean_t)),
                "mean_u": float(self.backend.asnumpy(mean_v[0])),
                "selected": int(stats.selected),
                "accepted": int(stats.accepted),
                "max_probability": float(stats.max_probability),
            }
        )

    def run(self, progress: bool = True) -> dict[str, Any]:
        c = self.config
        started = time.perf_counter()
        report_every = max(1, c.steps // 10)
        for step in range(1, c.steps + 1):
            collecting = step > c.warmup_steps
            self._move_and_reflect(collecting)
            cell_id = self._cell_ids()
            stats = self.collision.collide(
                self._groups(cell_id), self.positions, self.velocities, self.dt
            )
            if collecting and (step - c.warmup_steps) % c.sample_stride == 0:
                self.moments.sample(cell_id, self.velocities)
            if step == 1 or step % report_every == 0 or step == c.steps:
                self._diagnostic(step, stats)
                if progress:
                    print(
                        f"[{c.model}/{self.backend.name}] {step:>7}/{c.steps} "
                        f"T={self.history[-1]['temperature']:.2f} K "
                        f"Pmax={stats.max_probability:.3f}"
                    )
        self.backend.synchronize()
        runtime = time.perf_counter() - started
        if self.moments.samples < 1:
            raise RuntimeError("No samples were collected; adjust warmup_steps/sample_stride.")
        fields = self.moments.finalize(c, self.backend)
        return self._save(fields, runtime)

    def _top_microscopic(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.config
        weight = np.maximum(self._top_weight, 1e-30)
        gas_u = self._top_u_weighted / weight
        mean_v2 = self._top_v2_weighted / weight
        slip = (c.lid_velocity - gas_u) / c.lid_velocity
        temperature = c.mass * np.maximum(0.0, mean_v2 - gas_u * gas_u) / (3.0 * K_B)
        slip[self._top_weight == 0] = np.nan
        temperature[self._top_weight == 0] = np.nan
        return slip, temperature

    def _save(self, fields: dict[str, np.ndarray], runtime: float) -> dict[str, Any]:
        c = self.config
        out = Path(c.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        shape = (c.ny, c.nx)
        reshaped = {key: value.reshape(shape) for key, value in fields.items()}
        du_dy, du_dx = np.gradient(reshaped["u"], c.dy, c.dx)
        dv_dy, dv_dx = np.gradient(reshaped["v"], c.dy, c.dx)
        vorticity = dv_dx - du_dy
        x = (np.arange(c.nx) + 0.5) / c.nx
        y = (np.arange(c.ny) + 0.5) / c.ny
        xx, yy = np.meshgrid(x, y)
        rho_ratio = reshaped["rho"] / (c.mass * c.number_density)

        np.savez_compressed(
            out / "fields.npz",
            x=xx,
            y=yy,
            vorticity=vorticity,
            **reshaped,
        )
        columns = [
            xx.ravel(), yy.ravel(), rho_ratio.ravel(), reshaped["u"].ravel(),
            reshaped["v"].ravel(), reshaped["w"].ravel(),
            reshaped["temperature"].ravel(), reshaped["qx"].ravel(),
            reshaped["qy"].ravel(), reshaped["pxy"].ravel(),
            vorticity.ravel(), reshaped["sample_count"].ravel(),
        ]
        np.savetxt(
            out / "grid.csv",
            np.column_stack(columns),
            delimiter=",",
            header="x_over_L,y_over_L,rho_over_rho0,u,v,w,T,qx,qy,pxy,vorticity,sample_count",
            comments="",
        )
        top_u = reshaped["u"][-1]
        top_t = reshaped["temperature"][-1]
        macro_slip = (c.lid_velocity - top_u) / c.lid_velocity
        micro_slip, micro_t = self._top_microscopic()
        np.savetxt(
            out / "lid_profile.csv",
            np.column_stack((x, macro_slip, top_t, micro_slip, micro_t)),
            delimiter=",",
            header="x_over_L,macro_slip_over_Uwall,macro_T_K,micro_slip_over_Uwall,micro_T_K",
            comments="",
        )
        with (out / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.history[0].keys())
            writer.writeheader()
            writer.writerows(self.history)
        metadata = {
            "schema_version": 1,
            "config": c.to_dict(),
            "backend_resolved": self.backend.name,
            "runtime_seconds": runtime,
            "samples": self.moments.samples,
            "wall_hits": self.wall_hits,
            "collision_statistics": self.collision.total.__dict__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        }
        with (out / "metadata.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
        return {
            "output_dir": str(out.resolve()),
            "metadata": metadata,
            "fields": reshaped,
            "lid_x": x,
            "macro_slip": macro_slip,
            "macro_temperature": top_t,
        }
