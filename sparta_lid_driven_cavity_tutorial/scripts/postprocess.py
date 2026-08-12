#!/usr/bin/env python3
"""Create standalone SPARTA cavity profiles, figures, and run statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np


def latest_dump(run_dir: Path) -> Path:
    candidates = list(run_dir.glob("grid.final.*"))
    if not candidates:
        raise FileNotFoundError(f"No grid.final.* dump was found in {run_dir}")

    def step(path: Path) -> int:
        match = re.search(r"(\d+)$", path.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=step)


def read_last_snapshot(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    headers = [i for i, line in enumerate(lines) if line.startswith("ITEM: CELLS")]
    if not headers:
        raise ValueError(f"No ITEM: CELLS block in {path}")
    start = headers[-1]
    columns = lines[start].split()[2:]
    rows: list[list[float]] = []
    for line in lines[start + 1 :]:
        if line.startswith("ITEM:"):
            break
        if line.strip():
            rows.append([float(value) for value in line.split()])
    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(columns):
        raise ValueError(f"Malformed grid block in {path}")
    return columns, data


def moving_average(values: np.ndarray, window: int = 11) -> np.ndarray:
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def grid_arrays(
    columns: list[str], values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    index = {name: i for i, name in enumerate(columns)}
    required = ["xc", "yc", "f_flowavg[1]", "f_flowavg[2]", "f_flowavg[3]", "f_flowavg[5]"]
    missing = [name for name in required if name not in index]
    if missing:
        raise ValueError(f"Missing columns in SPARTA dump: {missing}; got {columns}")
    x = np.unique(values[:, index["xc"]])
    y = np.unique(values[:, index["yc"]])
    xi = np.searchsorted(x, values[:, index["xc"]])
    yi = np.searchsorted(y, values[:, index["yc"]])
    fields: dict[str, np.ndarray] = {}
    for label, column in {
        "number_density": "f_flowavg[1]",
        "u": "f_flowavg[2]",
        "v": "f_flowavg[3]",
        "temperature": "f_flowavg[5]",
    }.items():
        field = np.full((len(y), len(x)), np.nan)
        field[yi, xi] = values[:, index[column]]
        fields[label] = field
    return x, y, fields


def runtime_from_log(path: Path) -> dict[str, float | int]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    loops = re.findall(
        r"Loop time of\s+([0-9.eE+-]+)\s+on\s+(\d+)\s+procs\s+for\s+(\d+)\s+steps\s+with\s+(\d+)\s+particles",
        text,
    )
    perf = re.findall(
        r"Performance:\s+([0-9.eE+-]+)\s+timesteps/s,\s+([0-9.eE+-]+)\s+Mparticle-step/s",
        text,
    )
    collisions = re.findall(r"Collide occurs\s*=\s*(\d+)", text)
    stuck = re.findall(r"Particles stuck\s*=\s*(\d+)", text)
    report: dict[str, float | int] = {}
    if loops:
        warmup = loops[0]
        sample = loops[-1]
        report.update(
            {
                "mpi_ranks": int(sample[1]),
                "simulator_particles": int(sample[3]),
                "warmup_loop_seconds": float(warmup[0]),
                "sampling_loop_seconds": float(sample[0]),
            }
        )
    if perf:
        report["sampling_timesteps_per_second"] = float(perf[-1][0])
        report["sampling_mparticle_steps_per_second"] = float(perf[-1][1])
    if collisions:
        report["sampling_collisions"] = int(collisions[-1])
    if stuck:
        report["particles_stuck"] = int(stuck[-1])
    return report


def load_run(run_dir: Path) -> dict[str, object]:
    metadata = json.loads((run_dir / "case_metadata.json").read_text(encoding="utf-8"))
    dump = latest_dump(run_dir)
    columns, values = read_last_snapshot(dump)
    x, y, fields = grid_arrays(columns, values)
    return {
        "metadata": metadata,
        "dump": dump,
        "x": x,
        "y": y,
        "fields": fields,
        "runtime": runtime_from_log(run_dir / "log.cavity"),
    }


def process(run_dir: Path, window: int = 11) -> dict[str, object]:
    loaded = load_run(run_dir)
    metadata = loaded["metadata"]
    assert isinstance(metadata, dict)
    x = np.asarray(loaded["x"])
    y = np.asarray(loaded["y"])
    fields = loaded["fields"]
    assert isinstance(fields, dict)
    u = np.asarray(fields["u"])
    v = np.asarray(fields["v"])
    temperature = np.asarray(fields["temperature"])
    length = float(metadata["length_m"])
    lid_speed = float(metadata["lid_velocity_m_per_s"])
    lid_u = u[-1, :]
    lid_temperature = temperature[-1, :]
    lid_slip = (lid_speed - lid_u) / lid_speed
    smooth_u = moving_average(lid_u, window)
    smooth_temperature = moving_average(lid_temperature, window)
    smooth_slip = moving_average(lid_slip, window)

    raw_path = run_dir / "lid_profile_raw.csv"
    np.savetxt(
        raw_path,
        np.column_stack((x / length, lid_u / lid_speed, lid_slip, lid_temperature)),
        delimiter=",",
        header="x_over_L,u_over_Ulid,slip_over_Ulid,temperature_K",
        comments="",
    )
    smooth_path = run_dir / f"lid_profile_{window}cell.csv"
    np.savetxt(
        smooth_path,
        np.column_stack((x / length, smooth_u / lid_speed, smooth_slip, smooth_temperature)),
        delimiter=",",
        header="x_over_L,u_over_Ulid,slip_over_Ulid,temperature_K",
        comments="",
    )

    speed = np.sqrt(u**2 + v**2)
    runtime = loaded["runtime"]
    assert isinstance(runtime, dict)
    summary: dict[str, object] = {
        "case": metadata.get("case", "two-dimensional rarefied lid-driven cavity"),
        "seed": int(metadata["seed"]),
        "kn": float(metadata["kn"]),
        "grid": [int(metadata["nx"]), int(metadata["ny"])],
        "particles_per_cell": int(metadata["particles_per_cell"]),
        "final_dump": Path(loaded["dump"]).name,
        "lid_temperature_raw_min_K": float(np.min(lid_temperature)),
        "lid_temperature_raw_max_K": float(np.max(lid_temperature)),
        "lid_temperature_raw_mean_K": float(np.mean(lid_temperature)),
        "lid_temperature_11cell_min_K": float(np.min(smooth_temperature)),
        "lid_temperature_11cell_max_K": float(np.max(smooth_temperature)),
        "domain_temperature_min_K": float(np.min(temperature)),
        "domain_temperature_max_K": float(np.max(temperature)),
        "domain_speed_max_m_per_s": float(np.max(speed)),
        "profile_filter": f"centred {window}-cell moving average, display only",
        "runtime": runtime,
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    try:
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter

        fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), constrained_layout=True)
        axes[0].plot(x / length, lid_slip, color="0.70", lw=0.75, label="Raw cell values")
        axes[0].plot(x / length, smooth_slip, color="#176B87", lw=2.0, label=f"{window}-cell display average")
        axes[0].set(xlabel=r"$x/L$", ylabel=r"$(U_{lid}-u)/U_{lid}$")
        axes[1].plot(x / length, lid_temperature, color="0.70", lw=0.75, label="Raw cell values")
        axes[1].plot(x / length, smooth_temperature, color="#B7472A", lw=2.0, label=f"{window}-cell display average")
        axes[1].set(xlabel=r"$x/L$", ylabel="Temperature [K]")
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.legend(frameon=False, fontsize=8)
        fig.savefig(run_dir / "lid_profiles.png", dpi=240)
        plt.close(fig)

        sigma = 1.0
        temp_display = gaussian_filter(temperature, sigma=sigma, mode="nearest")
        u_stream = gaussian_filter(u, sigma=sigma, mode="nearest")
        v_stream = gaussian_filter(v, sigma=sigma, mode="nearest")
        xx, yy = np.meshgrid(x / length, y / length)
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.25), constrained_layout=True)
        image = axes[0].pcolormesh(xx, yy, temp_display, shading="auto", cmap="inferno")
        axes[0].streamplot(x / length, y / length, u_stream, v_stream, color="white", density=1.0, linewidth=0.45)
        fig.colorbar(image, ax=axes[0], label="Temperature [K]")
        speed_display = np.sqrt(u_stream**2 + v_stream**2)
        image = axes[1].pcolormesh(xx, yy, speed_display, shading="auto", cmap="viridis")
        skip = max(1, len(x) // 24)
        axes[1].quiver(xx[::skip, ::skip], yy[::skip, ::skip], u_stream[::skip, ::skip], v_stream[::skip, ::skip], color="white", alpha=0.75, scale=800)
        fig.colorbar(image, ax=axes[1], label="Speed [m/s]")
        for axis in axes:
            axis.set(xlabel=r"$x/L$", ylabel=r"$y/L$", aspect="equal")
        fig.savefig(run_dir / "fields.png", dpi=240)
        plt.close(fig)
    except ImportError:
        summary["plot_note"] = "Install requirements.txt to create PNG figures."
        (run_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--window", type=int, default=11)
    args = parser.parse_args()
    process(args.run_dir.resolve(), args.window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

