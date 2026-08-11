"""Create publication-ready overview plots from a saved cavity run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fields", type=Path, help="Path to fields.npz")
    p.add_argument("--output", type=Path, default=Path("cavity_overview.png"))
    args = p.parse_args()
    data = np.load(args.fields)
    x, y = data["x"], data["y"]
    u, v = data["u"], data["v"]
    speed = np.sqrt(u * u + v * v)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.4), constrained_layout=True)
    items = (
        (speed, "Velocity magnitude [m/s]", "viridis"),
        (data["temperature"], "Temperature [K]", "inferno"),
        (data["rho"] / np.nanmean(data["rho"]), r"$\rho/\langle\rho\rangle$", "coolwarm"),
        (data["vorticity"], "Vorticity [1/s]", "RdBu_r"),
    )
    for ax, (field, title, cmap) in zip(axes.ravel(), items):
        contour = ax.contourf(x, y, field, levels=35, cmap=cmap)
        fig.colorbar(contour, ax=ax, shrink=0.86)
        ax.set(xlabel="x/L", ylabel="y/L", title=title, aspect="equal")
    axes[0, 0].streamplot(x[0], y[:, 0], u, v, color="white", density=1.1, linewidth=0.6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
