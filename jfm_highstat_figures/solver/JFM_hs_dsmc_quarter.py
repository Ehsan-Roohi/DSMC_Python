#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hard-sphere DSMC production solver for the JFM reviewer calculation.

Case represented by the defaults:
  * paper Knudsen number: Kn = 30
  * wall-temperature ratio: Tc/Th = 0.2
  * one-quarter cavity: x,y in [0,L/2]
  * specular symmetry planes: x=0 and y=0
  * physical diffuse walls: x=L/2 (cold), y=L/2 (hot)
  * 22,000,000 simulator particles

The paper's Knudsen-number definition is used literally:

    Kn = 1 / (sqrt(2) * n0 * pi*d^2 * L).

No spatial filtering or projection is applied to any exported field.  The
full cavity is reconstructed from the sampled quadrant using exact parity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import cupy as cp
import numpy as np
from numba import cuda


SOLVER_VERSION = "2026-07-26-jfm-hs-dsmc-quarter-22m-v1"

MASS = 6.63e-26
KB = 1.380649e-23
D_REF = 4.17e-10
SIGMA = math.pi * D_REF**2

L = 1.0e-3
LQ = 0.5 * L
T_HOT = 1000.0

NC = 100
DX = LQ / NC
INV_DX = 1.0 / DX
CELL_VOL = DX * DX
N_CELLS = NC * NC

VMP_HOT = math.sqrt(2.0 * KB * T_HOT / MASS)
DT_DEFAULT = 0.15 * DX / VMP_HOT
THREADS = 128


@cuda.jit(device=True, inline=True)
def xorshift64star(state):
    state ^= state >> np.uint64(12)
    state ^= (state << np.uint64(25)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    state ^= state >> np.uint64(27)
    state = (
        state * np.uint64(2685821657736338717)
    ) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return state


@cuda.jit(device=True, inline=True)
def rng_uniform(state):
    state = xorshift64star(state)
    value = np.float32(
        ((state >> np.uint64(40)) & np.uint64(0xFFFFFF))
        / np.float32(16777216.0)
    )
    if value < np.float32(1.0e-7):
        value = np.float32(1.0e-7)
    return state, value


@cuda.jit(device=True, inline=True)
def rng_normal(state):
    state, u1 = rng_uniform(state)
    state, u2 = rng_uniform(state)
    radius = math.sqrt(-2.0 * math.log(u1))
    angle = 2.0 * math.pi * u2
    return state, np.float32(radius * math.cos(angle))


@cuda.jit(device=True, inline=True)
def diffuse_velocity(state, temperature, mass, kb, nx, ny):
    """Sample a diffuse-reflection velocity directed along inward (nx,ny)."""
    thermal_sigma = math.sqrt(kb * temperature / mass)
    state, uniform = rng_uniform(state)
    normal_speed = thermal_sigma * math.sqrt(-2.0 * math.log(uniform))
    state, tangent_gaussian = rng_normal(state)
    state, z_gaussian = rng_normal(state)

    tx = -ny
    ty = nx
    vx = normal_speed * nx + thermal_sigma * tangent_gaussian * tx
    vy = normal_speed * ny + thermal_sigma * tangent_gaussian * ty
    vz = thermal_sigma * z_gaussian
    return state, np.float32(vx), np.float32(vy), np.float32(vz)


@cuda.jit(fastmath=True)
def initialize_particles(x, y, vx, vy, vz, particle_rng, lq, t0, mass, kb):
    i = cuda.grid(1)
    if i >= x.size:
        return

    state = particle_rng[i]
    state, u1 = rng_uniform(state)
    state, u2 = rng_uniform(state)
    x[i] = u1 * lq
    y[i] = u2 * lq

    thermal_sigma = math.sqrt(kb * t0 / mass)
    state, g1 = rng_normal(state)
    state, g2 = rng_normal(state)
    state, g3 = rng_normal(state)
    vx[i] = thermal_sigma * g1
    vy[i] = thermal_sigma * g2
    vz[i] = thermal_sigma * g3
    particle_rng[i] = state


@cuda.jit(fastmath=True)
def advance_event_driven(
    x,
    y,
    vx,
    vy,
    vz,
    particle_rng,
    lq,
    dt,
    t_hot,
    t_cold,
    mass,
    kb,
):
    """
    Advance through the complete time step.

    Each wall hit is handled at its time of impact and the particle then
    advances for the remaining sub-step.  This avoids the wall artefact caused
    by reflecting an already-overshot particle only once per global step.
    """
    i = cuda.grid(1)
    if i >= x.size:
        return

    px = x[i]
    py = y[i]
    ux = vx[i]
    uy = vy[i]
    uz = vz[i]
    state = particle_rng[i]
    remaining = dt
    huge = np.float32(1.0e30)
    eps = np.float32(2.0e-7) * lq

    for _ in range(8):
        if remaining <= 0.0:
            break

        tx = huge
        ty = huge
        if ux > 0.0:
            tx = (lq - px) / ux
        elif ux < 0.0:
            tx = -px / ux

        if uy > 0.0:
            ty = (lq - py) / uy
        elif uy < 0.0:
            ty = -py / uy

        t_hit = tx
        hit_x = True
        if ty < tx:
            t_hit = ty
            hit_x = False

        if t_hit < 0.0 or t_hit >= remaining or t_hit == huge:
            px += ux * remaining
            py += uy * remaining
            remaining = 0.0
            break

        px += ux * t_hit
        py += uy * t_hit
        remaining -= t_hit

        if hit_x:
            if ux < 0.0:
                # x=0: specular symmetry plane.
                px = eps
                ux = -ux
            else:
                # x=L/2: cold diffuse wall; inward normal is -x.
                px = lq - eps
                state, ux, uy, uz = diffuse_velocity(
                    state, t_cold, mass, kb, -1.0, 0.0
                )
        else:
            if uy < 0.0:
                # y=0: specular symmetry plane.
                py = eps
                uy = -uy
            else:
                # y=L/2: hot diffuse wall; inward normal is -y.
                py = lq - eps
                state, ux, uy, uz = diffuse_velocity(
                    state, t_hot, mass, kb, 0.0, -1.0
                )

    # Extremely unlikely high-speed fallback after eight hits.
    if remaining > 0.0:
        px += ux * remaining
        py += uy * remaining

    if px < 0.0:
        px = -px
        ux = -ux
    elif px >= lq:
        px = lq - eps
        state, ux, uy, uz = diffuse_velocity(
            state, t_cold, mass, kb, -1.0, 0.0
        )

    if py < 0.0:
        py = -py
        uy = -uy
    elif py >= lq:
        py = lq - eps
        state, ux, uy, uz = diffuse_velocity(
            state, t_hot, mass, kb, 0.0, -1.0
        )

    x[i] = px
    y[i] = py
    vx[i] = ux
    vy[i] = uy
    vz[i] = uz
    particle_rng[i] = state


@cuda.jit(fastmath=True)
def assign_cells(x, y, cell_id, inv_dx, nc):
    i = cuda.grid(1)
    if i >= x.size:
        return
    ix = int(x[i] * inv_dx)
    iy = int(y[i] * inv_dx)
    if ix < 0:
        ix = 0
    elif ix >= nc:
        ix = nc - 1
    if iy < 0:
        iy = 0
    elif iy >= nc:
        iy = nc - 1
    cell_id[i] = iy * nc + ix


@cuda.jit(fastmath=True)
def collide_hard_sphere_ntc(
    vx,
    vy,
    vz,
    starts,
    counts,
    sigma_g_majorant,
    cell_rng,
    candidate_pairs,
    accepted_collisions,
    majorant_updates,
    dt,
    cell_vol,
    fnum,
    sigma_hs,
):
    """One-thread-per-cell no-time-counter hard-sphere DSMC collision step."""
    cid = cuda.grid(1)
    if cid >= counts.size:
        return

    count = counts[cid]
    if count < 2:
        return

    start = starts[cid]
    state = cell_rng[cid]
    local_majorant = sigma_g_majorant[cid]
    if local_majorant < 1.0e-30:
        local_majorant = np.float32(1.0e-30)

    expected = (
        0.5
        * count
        * (count - 1)
        * fnum
        * local_majorant
        * dt
        / cell_vol
    )
    n_candidates = int(expected)
    state, uniform = rng_uniform(state)
    if uniform < expected - n_candidates:
        n_candidates += 1

    local_accepted = 0
    local_updates = 0

    for _ in range(n_candidates):
        state, u1 = rng_uniform(state)
        state, u2 = rng_uniform(state)
        local_p1 = min(int(u1 * count), count - 1)
        # Draw the second member from the remaining count-1 particles, so a
        # nominal NTC candidate is never silently discarded as a self-pair.
        local_p2 = min(int(u2 * (count - 1)), count - 2)
        if local_p2 >= local_p1:
            local_p2 += 1
        p1 = start + local_p1
        p2 = start + local_p2

        gx = vx[p1] - vx[p2]
        gy = vy[p1] - vy[p2]
        gz = vz[p1] - vz[p2]
        relative_speed = math.sqrt(gx * gx + gy * gy + gz * gz)
        sigma_g = sigma_hs * relative_speed

        if sigma_g > local_majorant:
            local_majorant = sigma_g
            local_updates += 1

        state, accept_u = rng_uniform(state)
        if accept_u * local_majorant >= sigma_g:
            continue

        state, cos_u = rng_uniform(state)
        cos_chi = 2.0 * cos_u - 1.0
        sin_chi = math.sqrt(max(0.0, 1.0 - cos_chi * cos_chi))
        state, phi_u = rng_uniform(state)
        phi = 2.0 * math.pi * phi_u

        vcmx = 0.5 * (vx[p1] + vx[p2])
        vcmy = 0.5 * (vy[p1] + vy[p2])
        vcmz = 0.5 * (vz[p1] + vz[p2])
        half_g = 0.5 * relative_speed
        dx = half_g * sin_chi * math.cos(phi)
        dy = half_g * sin_chi * math.sin(phi)
        dz = half_g * cos_chi

        vx[p1] = vcmx + dx
        vy[p1] = vcmy + dy
        vz[p1] = vcmz + dz
        vx[p2] = vcmx - dx
        vy[p2] = vcmy - dy
        vz[p2] = vcmz - dz
        local_accepted += 1

    sigma_g_majorant[cid] = local_majorant
    cell_rng[cid] = state
    candidate_pairs[cid] += n_candidates
    accepted_collisions[cid] += local_accepted
    majorant_updates[cid] += local_updates


@cuda.jit(fastmath=True)
def sample_sorted_cells(vx, vy, vz, starts, counts, accumulators, block_id, ncells):
    """
    Accumulate raw cell moments with one thread per sorted cell.

    This avoids tens of millions of atomic operations at each sampling time.
    """
    cid = cuda.grid(1)
    if cid >= counts.size:
        return

    count = counts[cid]
    start = starts[cid]
    sum_u = 0.0
    sum_v = 0.0
    sum_w = 0.0
    sum_v2 = 0.0
    for p in range(start, start + count):
        up = float(vx[p])
        vp = float(vy[p])
        wp = float(vz[p])
        sum_u += up
        sum_v += vp
        sum_w += wp
        sum_v2 += up * up + vp * vp + wp * wp

    offset = block_id * 5 * ncells + cid
    accumulators[offset] += count
    accumulators[offset + ncells] += sum_u
    accumulators[offset + 2 * ncells] += sum_v
    accumulators[offset + 3 * ncells] += sum_w
    accumulators[offset + 4 * ncells] += sum_v2


def reconstruct_scalar(quadrant):
    top = np.concatenate((np.fliplr(quadrant), quadrant), axis=1)
    return np.concatenate((np.flipud(top), top), axis=0)


def reconstruct_velocity(u_quadrant, v_quadrant):
    u_top = np.concatenate((-np.fliplr(u_quadrant), u_quadrant), axis=1)
    v_top = np.concatenate((np.fliplr(v_quadrant), v_quadrant), axis=1)
    u_full = np.concatenate((np.flipud(u_top), u_top), axis=0)
    v_full = np.concatenate((-np.flipud(v_top), v_top), axis=0)
    return u_full, v_full


def moments_to_fields(moment_sum, sample_count, fnum, n0):
    sn, su, sv, sw, sv2 = moment_sum
    safe = np.maximum(sn, 1.0)
    uq = (su / safe).reshape(NC, NC)
    vq = (sv / safe).reshape(NC, NC)
    wq = (sw / safe).reshape(NC, NC)
    mean_v2 = (sv2 / safe).reshape(NC, NC)
    thermal = np.maximum(mean_v2 - uq * uq - vq * vq - wq * wq, 1.0e-12)
    tq = MASS * thermal / (3.0 * KB)

    mean_particles_per_sample = sn / max(sample_count, 1)
    number_density = (
        mean_particles_per_sample * fnum / CELL_VOL
    ).reshape(NC, NC)
    rhoq = number_density / n0

    u_full, v_full = reconstruct_velocity(uq, vq)
    return (
        u_full / VMP_HOT,
        v_full / VMP_HOT,
        reconstruct_scalar(tq) / T_HOT,
        reconstruct_scalar(rhoq),
    )


def write_tecplot(path, x, y, ux, uy, temperature, density):
    with path.open("w", encoding="utf-8") as handle:
        handle.write('TITLE="JFM hard-sphere DSMC raw unfiltered field"\n')
        handle.write('VARIABLES="x","y","u_x","u_y","T","rho","Umag"\n')
        handle.write(f"ZONE I={x.size}, J={y.size}, F=POINT\n")
        for j, y_value in enumerate(y):
            for i, x_value in enumerate(x):
                speed = math.hypot(ux[j, i], uy[j, i])
                handle.write(
                    f"{x_value:.10e} {y_value:.10e} "
                    f"{ux[j, i]:.10e} {uy[j, i]:.10e} "
                    f"{temperature[j, i]:.10e} {density[j, i]:.10e} "
                    f"{speed:.10e}\n"
                )


def sample_total(steps, sample_start, sample_every):
    return 1 + (steps - 1 - sample_start) // sample_every


def memory_gib(number_bytes):
    return number_bytes / 1024.0**3


class HardSphereQuarterCavity:
    def __init__(self, args):
        self.args = args
        self.n_particles = args.particles
        self.t_cold = args.rt * T_HOT
        self.dt = args.dt
        self.n0 = 1.0 / (
            math.sqrt(2.0) * SIGMA * args.kn * L
        )
        self.fnum = self.n0 * LQ * LQ / self.n_particles
        self.n_sampling_times = sample_total(
            args.steps, args.sample_start, args.sample_every
        )
        self.block_sample_counts = np.zeros(args.time_blocks, dtype=np.int64)

        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        self.gpu_free_gib_at_start = memory_gib(free_bytes)
        self.gpu_total_gib = memory_gib(total_bytes)
        if self.gpu_free_gib_at_start < args.require_free_gb:
            raise RuntimeError(
                f"Only {self.gpu_free_gib_at_start:.2f} GiB GPU memory is free; "
                f"--require-free-gb={args.require_free_gb:.2f}."
            )

        # Float32 particle state keeps the 22-million-particle run comfortably
        # inside an 11-GiB RTX 2080 Ti.  All sampled moment sums are float64.
        self.x = cp.empty(self.n_particles, dtype=cp.float32)
        self.y = cp.empty(self.n_particles, dtype=cp.float32)
        self.vx = cp.empty(self.n_particles, dtype=cp.float32)
        self.vy = cp.empty(self.n_particles, dtype=cp.float32)
        self.vz = cp.empty(self.n_particles, dtype=cp.float32)

        seed_u64 = np.uint64(args.seed)
        seed_offset = (
            seed_u64 * np.uint64(1442695040888963407)
            + np.uint64(6364136223846793005)
        )
        self.particle_rng = (
            cp.arange(self.n_particles, dtype=cp.uint64)
            * cp.uint64(2862933555777941757)
            + cp.uint64(seed_offset)
        )
        self.cell_rng = (
            cp.arange(N_CELLS, dtype=cp.uint64)
            * cp.uint64(3202034522624059733)
            + cp.uint64(seed_offset ^ np.uint64(0x9E3779B97F4A7C15))
        )
        self.cell_id = cp.empty(self.n_particles, dtype=cp.int32)

        initial_relative_speed_majorant = 4.5 * VMP_HOT
        self.sigma_g_majorant = cp.full(
            N_CELLS,
            SIGMA * initial_relative_speed_majorant,
            dtype=cp.float32,
        )
        self.candidate_pairs = cp.zeros(N_CELLS, dtype=cp.int64)
        self.accepted_collisions = cp.zeros(N_CELLS, dtype=cp.int64)
        self.majorant_updates = cp.zeros(N_CELLS, dtype=cp.int64)
        self.accumulators = cp.zeros(
            args.time_blocks * 5 * N_CELLS, dtype=cp.float64
        )

        blocks_particles = (self.n_particles + THREADS - 1) // THREADS
        t0 = 0.5 * (T_HOT + self.t_cold)
        initialize_particles[blocks_particles, THREADS](
            self.x,
            self.y,
            self.vx,
            self.vy,
            self.vz,
            self.particle_rng,
            LQ,
            t0,
            MASS,
            KB,
        )
        cuda.synchronize()

    def sort_by_cell(self):
        blocks_particles = (self.n_particles + THREADS - 1) // THREADS
        assign_cells[blocks_particles, THREADS](
            self.x, self.y, self.cell_id, INV_DX, NC
        )
        counts = cp.bincount(self.cell_id, minlength=N_CELLS).astype(
            cp.int32, copy=False
        )
        order = cp.argsort(self.cell_id)
        self.x = self.x[order]
        self.y = self.y[order]
        self.vx = self.vx[order]
        self.vy = self.vy[order]
        self.vz = self.vz[order]
        self.particle_rng = self.particle_rng[order]
        starts = cp.empty(N_CELLS, dtype=cp.int64)
        starts[0] = 0
        cp.cumsum(counts[:-1], dtype=cp.int64, out=starts[1:])
        return starts, counts

    def run(self):
        args = self.args
        blocks_particles = (self.n_particles + THREADS - 1) // THREADS
        blocks_cells = (N_CELLS + THREADS - 1) // THREADS
        print(
            f"--- HS DSMC quarter cavity: Kn={args.kn:g}, RT={args.rt:g}, "
            f"N={self.n_particles}, seed={args.seed}, NCq={NC} ---",
            flush=True,
        )
        print(
            f"solver={SOLVER_VERSION}; dt={self.dt:.8e} s; "
            f"steps={args.steps}; sample={args.sample_start}:{args.sample_every}",
            flush=True,
        )
        print(
            f"n0={self.n0:.8e} 1/m^3; FNUM={self.fnum:.8e}; "
            f"GPU memory free/total at start="
            f"{self.gpu_free_gib_at_start:.2f}/{self.gpu_total_gib:.2f} GiB",
            flush=True,
        )

        run_start = time.time()
        sample_index = 0
        for step in range(args.steps):
            advance_event_driven[blocks_particles, THREADS](
                self.x,
                self.y,
                self.vx,
                self.vy,
                self.vz,
                self.particle_rng,
                LQ,
                self.dt,
                T_HOT,
                self.t_cold,
                MASS,
                KB,
            )
            starts, counts = self.sort_by_cell()
            collide_hard_sphere_ntc[blocks_cells, THREADS](
                self.vx,
                self.vy,
                self.vz,
                starts,
                counts,
                self.sigma_g_majorant,
                self.cell_rng,
                self.candidate_pairs,
                self.accepted_collisions,
                self.majorant_updates,
                self.dt,
                CELL_VOL,
                self.fnum,
                np.float32(SIGMA),
            )

            if (
                step >= args.sample_start
                and (step - args.sample_start) % args.sample_every == 0
            ):
                block_id = min(
                    args.time_blocks - 1,
                    sample_index * args.time_blocks // self.n_sampling_times,
                )
                sample_sorted_cells[blocks_cells, THREADS](
                    self.vx,
                    self.vy,
                    self.vz,
                    starts,
                    counts,
                    self.accumulators,
                    block_id,
                    N_CELLS,
                )
                self.block_sample_counts[block_id] += 1
                sample_index += 1

            if step % args.progress_every == 0:
                elapsed_hours = (time.time() - run_start) / 3600.0
                print(
                    f"step {step}/{args.steps}; samples={sample_index}; "
                    f"elapsed={elapsed_hours:.3f} h",
                    flush=True,
                )

        cuda.synchronize()
        self.wall_seconds = time.time() - run_start
        if sample_index != self.n_sampling_times:
            raise RuntimeError(
                f"Expected {self.n_sampling_times} samples, obtained {sample_index}"
            )

        raw = cp.asnumpy(self.accumulators).reshape(
            args.time_blocks, 5, N_CELLS
        )
        total_moments = raw.sum(axis=0)
        self.fields = moments_to_fields(
            total_moments, sample_index, self.fnum, self.n0
        )
        block_fields = [
            moments_to_fields(
                raw[b],
                int(self.block_sample_counts[b]),
                self.fnum,
                self.n0,
            )
            for b in range(args.time_blocks)
        ]
        self.block_fields = tuple(
            np.stack([entry[k] for entry in block_fields], axis=0)
            for k in range(4)
        )

        self.diagnostics = {
            "candidate_pairs": int(cp.asnumpy(self.candidate_pairs).sum()),
            "accepted_collisions": int(
                cp.asnumpy(self.accepted_collisions).sum()
            ),
            "majorant_updates": int(cp.asnumpy(self.majorant_updates).sum()),
            "final_sigma_g_majorant_min": float(
                cp.asnumpy(self.sigma_g_majorant).min()
            ),
            "final_sigma_g_majorant_mean": float(
                cp.asnumpy(self.sigma_g_majorant).mean()
            ),
            "final_sigma_g_majorant_max": float(
                cp.asnumpy(self.sigma_g_majorant).max()
            ),
        }
        candidates = self.diagnostics["candidate_pairs"]
        self.diagnostics["acceptance_fraction"] = (
            self.diagnostics["accepted_collisions"] / candidates
            if candidates
            else 0.0
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="JFM Kn=30 hard-sphere DSMC quarter-cavity production run"
    )
    parser.add_argument("--kn", type=float, default=30.0)
    parser.add_argument("--rt", type=float, default=0.2)
    parser.add_argument("--particles", type=int, default=22_000_000)
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--sample-start", type=int, default=400_000)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--time-blocks", type=int, default=4)
    parser.add_argument("--dt", type=float, default=DT_DEFAULT)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--require-free-gb", type=float, default=8.5)
    parser.add_argument(
        "--output", default="results_hs_dsmc_kn30_22m"
    )
    return parser.parse_args()


def validate_args(args):
    if args.kn <= 0.0:
        raise ValueError("--kn must be positive")
    if not 0.0 < args.rt < 1.0:
        raise ValueError("--rt must satisfy 0 < RT < 1")
    if args.particles < 100_000:
        raise ValueError("--particles must be at least 100000")
    if args.steps <= 0 or not 0 <= args.sample_start < args.steps:
        raise ValueError("Require 0 <= sample-start < steps")
    if args.sample_every <= 0 or args.time_blocks <= 0:
        raise ValueError("sample-every and time-blocks must be positive")
    if args.time_blocks > sample_total(
        args.steps, args.sample_start, args.sample_every
    ):
        raise ValueError("More time blocks than sampling times")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")


def main():
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rt_tag = str(args.rt).replace(".", "p")
    stem = output_dir / (
        f"ThermalCavity_HS_DSMC_Kn{args.kn:g}_RT{rt_tag}"
        f"_quarter_seed{args.seed}"
    )

    total_start = time.time()
    solver = HardSphereQuarterCavity(args)
    solver.run()
    ux, uy, temperature, density = solver.fields
    ux_blocks, uy_blocks, t_blocks, rho_blocks = solver.block_fields

    nf = 2 * NC
    coordinate = -0.5 + (np.arange(nf) + 0.5) / nf
    np.savez_compressed(
        str(stem) + "_raw.npz",
        x=coordinate,
        y=coordinate,
        ux=ux,
        uy=uy,
        T=temperature,
        rho=density,
        ux_time_blocks=ux_blocks,
        uy_time_blocks=uy_blocks,
        T_time_blocks=t_blocks,
        rho_time_blocks=rho_blocks,
        samples_per_time_block=solver.block_sample_counts,
        seed=np.int64(args.seed),
        Kn_paper=np.float64(args.kn),
        RT=np.float64(args.rt),
    )
    write_tecplot(
        Path(str(stem) + "_raw.dat"),
        coordinate,
        coordinate,
        ux,
        uy,
        temperature,
        density,
    )

    speed = np.hypot(ux, uy)
    dx_nd = 1.0 / nf
    kinetic_energy = float(
        np.sum(density * speed * speed) * dx_nd * dx_nd
    )
    block_speed_max = [
        float(np.hypot(ux_blocks[b], uy_blocks[b]).max())
        for b in range(args.time_blocks)
    ]
    last_block_velocity_rmse = float(
        np.sqrt(
            np.mean(
                (ux_blocks[-1] - ux) ** 2
                + (uy_blocks[-1] - uy) ** 2
            )
        )
    )

    metrics = {
        "solver_version": SOLVER_VERSION,
        "mode": "HARD_SPHERE_DSMC",
        "collision_model": "hard-sphere no-time-counter DSMC",
        "seed": args.seed,
        "Kn_paper": args.kn,
        "RT": args.rt,
        "kn_definition": "1/(sqrt(2)*n0*pi*d^2*L)",
        "quarter_domain": True,
        "specular_planes": ["x=0", "y=0"],
        "physical_diffuse_walls": {
            "x=L/2": "cold",
            "y=L/2": "hot",
        },
        "wall_advance_scheme": (
            "event-driven time-of-impact with residual-time advance"
        ),
        "quadrant_cells": [NC, NC],
        "full_reconstructed_cells": [nf, nf],
        "particles": args.particles,
        "mean_particles_per_quadrant_cell": args.particles / N_CELLS,
        "particle_storage_dtype": "float32",
        "moment_accumulator_dtype": "float64",
        "steps": args.steps,
        "sample_start": args.sample_start,
        "sample_every": args.sample_every,
        "profile_samples": solver.n_sampling_times,
        "time_blocks": args.time_blocks,
        "samples_per_time_block": solver.block_sample_counts.tolist(),
        "dt_seconds": args.dt,
        "n0_per_m3": solver.n0,
        "FNUM": solver.fnum,
        "max_speed_nondimensional": float(speed.max()),
        "kinetic_energy_nondimensional": kinetic_energy,
        "time_block_max_speeds_nondimensional": block_speed_max,
        "last_block_velocity_rmse_vs_all_samples": (
            last_block_velocity_rmse
        ),
        "gpu_free_GiB_at_start": solver.gpu_free_gib_at_start,
        "gpu_total_GiB": solver.gpu_total_gib,
        "solver_wall_clock_seconds": solver.wall_seconds,
        "total_wall_clock_seconds": time.time() - total_start,
        "dsmc_diagnostics": solver.diagnostics,
        "quantitative_fields_are_unfiltered": True,
        "spatial_smoothing_applied": False,
        "velocity_projection_applied": False,
    }
    Path(str(stem) + "_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"[OK] Raw unfiltered outputs: {stem}_raw.*", flush=True)


if __name__ == "__main__":
    main()
