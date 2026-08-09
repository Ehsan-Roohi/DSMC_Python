#!/usr/bin/env python3
"""Build direct chemistry OFF/ON/ON-OFF Gate-5 comparisons."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

P0 = 5.0e5
FLOW_NAME = "QK_PRODUCTION_FLOW_FIELD.dat"
SPECIES_NAME = "QK_PRODUCTION_SPECIES_FIELD.dat"
REACTION_NAME = "QK_PRODUCTION_REACTION_FIELD.dat"


def load_table(path: Path, columns: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.loadtxt(path, skiprows=1)
    data = np.atleast_2d(data)
    if data.shape != (4200, columns):
        raise ValueError(f"{path}: expected (4200,{columns}), found {data.shape}")
    return data


def locate_cases(result: Path) -> tuple[Path, Path]:
    roots = [result / "cases", result / "gate5_work" / "cases", result]
    for root in roots:
        off = root / "p5_r027_t4000_chem_off"
        on = root / "p5_r027_t4000_chem_on"
        if (off / FLOW_NAME).is_file() and (on / FLOW_NAME).is_file():
            return off, on
    raise FileNotFoundError(f"OFF/ON case pair not found below {result}")


def load_case(path: Path) -> dict[str, np.ndarray]:
    flow = load_table(path / FLOW_NAME, 14)
    species = load_table(path / SPECIES_NAME, 19)
    reaction = load_table(path / REACTION_NAME, 8)
    if not np.allclose(flow[:, 0], species[:, 0]) or not np.allclose(flow[:, 0], reaction[:, 0]):
        raise ValueError(f"cell numbering mismatch in {path}")
    if not np.allclose(flow[:, 2:4], species[:, 1:3]) or not np.allclose(flow[:, 2:4], reaction[:, 1:3]):
        raise ValueError(f"coordinate mismatch in {path}")
    return {"flow": flow, "species": species, "reaction": reaction}


def mesh(flow: np.ndarray, start: int, ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    block = flow[start : start + nx * ny]
    return block[:, 2].reshape(ny, nx) * 1.0e6, block[:, 3].reshape(ny, nx) * 1.0e6


def draw(ax, flow: np.ndarray, values: np.ndarray, *, cmap: str, vmin=None, vmax=None, norm=None):
    artist = None
    for start, ny, nx in ((0, 30, 100), (3000, 40, 30)):
        x, y = mesh(flow, start, ny, nx)
        z = values[start : start + nx * ny].reshape(ny, nx)
        artist = ax.pcolormesh(
            x, y, z, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax,
            norm=norm, rasterized=True,
        )
    ax.set(xlim=(0.0, 266.5), ylim=(0.0, 94.0), xlabel=r"$x\;(\mu\mathrm{m})$", ylabel=r"$y\;(\mu\mathrm{m})$")
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(direction="in", top=True, right=True)
    return artist


def finite_limits(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    combined = np.concatenate((a[np.isfinite(a)], b[np.isfinite(b)]))
    low, high = np.percentile(combined, [0.5, 99.5])
    delta_max = float(np.percentile(np.abs(b - a), 99.5))
    if not np.isfinite(delta_max) or delta_max <= 0.0:
        delta_max = 1.0
    return float(low), float(high), delta_max


def centerline(flow: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices: dict[float, int] = {}
    for i, row in enumerate(flow):
        cell, zone, x, y = int(row[0]), int(row[1]), float(row[2]), float(row[3])
        if zone != 1 or cell > 3000:
            continue
        old = indices.get(x)
        if old is None or y > flow[old, 3]:
            indices[x] = i
    order = [indices[x] for x in sorted(indices)]
    return flow[order, 2] * 1.0e6, values[order]


def shock_x(flow: np.ndarray) -> float | None:
    x, mach = centerline(flow, flow[:, 11])
    _, rho = centerline(flow, flow[:, 4])
    _, pressure = centerline(flow, flow[:, 12])
    _, velocity = centerline(flow, flow[:, 8])
    best: tuple[float, int] | None = None
    for i in range(3, len(x) - 4):
        if x[i] <= x[0] + 0.25 * (x[-1] - x[0]) or x[i] >= x[0] + 0.97 * (x[-1] - x[0]):
            continue
        rj = (rho[i + 2] - rho[i - 2]) / max(0.5 * (abs(rho[i + 2]) + abs(rho[i - 2])), 1.0e-300)
        pj = (pressure[i + 2] - pressure[i - 2]) / max(0.5 * (abs(pressure[i + 2]) + abs(pressure[i - 2])), 1.0e-300)
        ud = (velocity[i - 2] - velocity[i + 2]) / max(abs(velocity[i - 2]), 1.0e-300)
        score = max(rj, 0.0) + max(pj, 0.0) + max(ud, 0.0)
        if best is None or score > best[0]:
            best = (score, i)
    if best is None:
        return None
    return float(x[best[1]])


def make_contours(outdir: Path, off: dict[str, np.ndarray], on: dict[str, np.ndarray]) -> None:
    fields = [
        ("Mach", off["flow"][:, 11], on["flow"][:, 11], "viridis"),
        (r"$p/p_0$", off["flow"][:, 12] / P0, on["flow"][:, 12] / P0, "cividis"),
        (r"$T_{tr}$ (K)", off["flow"][:, 5], on["flow"][:, 5], "inferno"),
        (r"$X_{H_2O}$", off["species"][:, 16], on["species"][:, 16], "magma"),
        (r"$X_{OH}$", off["species"][:, 15], on["species"][:, 15], "plasma"),
    ]
    fig, axes = plt.subplots(len(fields), 3, figsize=(16.5, 14.5), constrained_layout=True)
    for row, (label, a, b, cmap) in enumerate(fields):
        low, high, dmax = finite_limits(a, b)
        for col, (values, title) in enumerate(((a, "chemistry OFF"), (b, "chemistry ON"))):
            artist = draw(axes[row, col], off["flow"], values, cmap=cmap, vmin=low, vmax=high)
            axes[row, col].set_title(f"{label} — {title}", loc="left", fontsize=10)
            fig.colorbar(artist, ax=axes[row, col], pad=0.01, shrink=0.82)
        norm = TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax)
        artist = draw(axes[row, 2], off["flow"], b - a, cmap="coolwarm", norm=norm)
        axes[row, 2].set_title(f"{label} — ON − OFF", loc="left", fontsize=10)
        fig.colorbar(artist, ax=axes[row, 2], pad=0.01, shrink=0.82)
    fig.suptitle("Gate 5 nozzle: direct chemistry OFF / ON / ON−OFF comparison", fontsize=15)
    fig.savefig(outdir / "QK_GATE5_ON_OFF_CONTOUR_COMPARISON.png", dpi=250, bbox_inches="tight")
    fig.savefig(outdir / "QK_GATE5_ON_OFF_CONTOUR_COMPARISON.pdf", bbox_inches="tight")
    plt.close(fig)


def make_centerlines(outdir: Path, off: dict[str, np.ndarray], on: dict[str, np.ndarray]) -> None:
    fields = [
        ("Mach", off["flow"][:, 11], on["flow"][:, 11]),
        (r"$p/p_0$", off["flow"][:, 12] / P0, on["flow"][:, 12] / P0),
        (r"$T_{tr}$ (K)", off["flow"][:, 5], on["flow"][:, 5]),
        (r"$X_{H_2O}$", off["species"][:, 16], on["species"][:, 16]),
        (r"$X_{OH}$", off["species"][:, 15], on["species"][:, 15]),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 10.0), constrained_layout=True)
    for ax, (label, a, b) in zip(axes.flat, fields):
        x, av = centerline(off["flow"], a)
        _, bv = centerline(on["flow"], b)
        ax.plot(x, av, label="OFF", lw=1.7)
        ax.plot(x, bv, label="ON", lw=1.7)
        ax.set(xlabel=r"$x\;(\mu\mathrm{m})$", ylabel=label)
        ax.grid(alpha=0.25)
        ax.legend()
    ax = axes.flat[-1]
    x, off_m = centerline(off["flow"], off["flow"][:, 11])
    _, on_m = centerline(on["flow"], on["flow"][:, 11])
    ax.plot(x, on_m - off_m, color="tab:red", lw=1.7)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set(xlabel=r"$x\;(\mu\mathrm{m})$", ylabel=r"$\Delta$Mach (ON−OFF)")
    ax.grid(alpha=0.25)
    fig.suptitle("Gate 5 nozzle centerline comparison", fontsize=14)
    fig.savefig(outdir / "QK_GATE5_ON_OFF_CENTERLINE_COMPARISON.png", dpi=250, bbox_inches="tight")
    fig.savefig(outdir / "QK_GATE5_ON_OFF_CENTERLINE_COMPARISON.pdf", bbox_inches="tight")
    plt.close(fig)


def write_delta(outdir: Path, off: dict[str, np.ndarray], on: dict[str, np.ndarray]) -> dict[str, object]:
    delta_fields = {
        "delta_rho": on["flow"][:, 4] - off["flow"][:, 4],
        "delta_Ttr_K": on["flow"][:, 5] - off["flow"][:, 5],
        "delta_Mach": on["flow"][:, 11] - off["flow"][:, 11],
        "delta_p_over_p0": (on["flow"][:, 12] - off["flow"][:, 12]) / P0,
        "delta_XH2O": on["species"][:, 16] - off["species"][:, 16],
        "delta_XOH": on["species"][:, 15] - off["species"][:, 15],
    }
    path = outdir / "QK_GATE5_ON_MINUS_OFF_FIELD.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell", "zone", "x_m", "y_m", *delta_fields])
        for i in range(4200):
            writer.writerow([
                int(off["flow"][i, 0]), int(off["flow"][i, 1]),
                off["flow"][i, 2], off["flow"][i, 3],
                *(delta_fields[name][i] for name in delta_fields),
            ])
    off_shock = shock_x(off["flow"])
    on_shock = shock_x(on["flow"])
    metrics: dict[str, object] = {
        "scope": "Gate5 direct chemistry OFF/ON comparison",
        "cells": 4200,
        "shock_x_um": {"off": off_shock, "on": on_shock},
        "shock_shift_on_minus_off_um": None if off_shock is None or on_shock is None else on_shock - off_shock,
        "reaction_events_in_sampled_field": {
            "off": int(np.rint(off["reaction"][:, 3:6].sum())),
            "on": int(np.rint(on["reaction"][:, 3:6].sum())),
        },
        "field_differences": {},
    }
    for name, values in delta_fields.items():
        metrics["field_differences"][name] = {
            "mean": float(np.mean(values)),
            "rms": float(np.sqrt(np.mean(values * values))),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    (outdir / "QK_GATE5_ON_OFF_COMPARISON.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True, help="Gate-5 job result directory")
    parser.add_argument("--out", type=Path, help="output directory (default: RESULT_DIR/comparison)")
    args = parser.parse_args()
    outdir = args.out or args.result / "comparison"
    outdir.mkdir(parents=True, exist_ok=True)
    off_path, on_path = locate_cases(args.result)
    off, on = load_case(off_path), load_case(on_path)
    off_events = float(off["reaction"][:, 3:6].sum())
    on_events = float(on["reaction"][:, 3:6].sum())
    if off_events != 0.0 or on_events <= 0.0:
        raise ValueError(f"case identity check failed: OFF events={off_events}, ON events={on_events}")
    make_contours(outdir, off, on)
    make_centerlines(outdir, off, on)
    metrics = write_delta(outdir, off, on)
    print(json.dumps(metrics, indent=2))
    print(f"QK_GATE5_ON_OFF_COMPARISON_PASS out={outdir}")


if __name__ == "__main__":
    main()
