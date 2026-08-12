#!/usr/bin/env python3
"""Combine independent SPARTA cavity seeds into ensemble statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from postprocess import load_run, moving_average, process


def t_critical_95(n: int) -> float:
    if n < 2:
        return 0.0
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, n - 1))
    except ImportError:
        return 1.96


def confidence_halfwidth(stack: np.ndarray) -> np.ndarray:
    n = stack.shape[0]
    if n < 2:
        return np.zeros(stack.shape[1:], dtype=float)
    return t_critical_95(n) * np.std(stack, axis=0, ddof=1) / np.sqrt(n)


def ensemble(run_dirs: list[Path], output: Path, window: int = 11) -> dict[str, object]:
    if len(run_dirs) < 2:
        raise ValueError("At least two independent run directories are required")
    output.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, object]] = []
    for run_dir in run_dirs:
        if not (run_dir / "run_summary.json").exists():
            process(run_dir, window)
        runs.append(load_run(run_dir))

    first_meta = runs[0]["metadata"]
    assert isinstance(first_meta, dict)
    length = float(first_meta["length_m"])
    lid_speed = float(first_meta["lid_velocity_m_per_s"])
    x = np.asarray(runs[0]["x"])
    y = np.asarray(runs[0]["y"])
    seeds: list[int] = []
    runtime_by_seed: dict[str, object] = {}
    u_stack: list[np.ndarray] = []
    v_stack: list[np.ndarray] = []
    temp_stack: list[np.ndarray] = []

    for run in runs:
        meta = run["metadata"]
        fields = run["fields"]
        assert isinstance(meta, dict) and isinstance(fields, dict)
        if int(meta["nx"]) != int(first_meta["nx"]) or int(meta["ny"]) != int(first_meta["ny"]):
            raise ValueError("All ensemble runs must use the same grid")
        if not np.allclose(np.asarray(run["x"]), x) or not np.allclose(np.asarray(run["y"]), y):
            raise ValueError("All ensemble runs must use identical cell centres")
        seed = int(meta["seed"])
        seeds.append(seed)
        runtime_by_seed[str(seed)] = run["runtime"]
        u_stack.append(np.asarray(fields["u"]))
        v_stack.append(np.asarray(fields["v"]))
        temp_stack.append(np.asarray(fields["temperature"]))

    u_values = np.stack(u_stack)
    v_values = np.stack(v_stack)
    temp_values = np.stack(temp_stack)
    u_mean = np.mean(u_values, axis=0)
    v_mean = np.mean(v_values, axis=0)
    temp_mean = np.mean(temp_values, axis=0)
    lid_u_values = u_values[:, -1, :]
    lid_temp_values = temp_values[:, -1, :]
    lid_slip_values = (lid_speed - lid_u_values) / lid_speed
    lid_u_mean = np.mean(lid_u_values, axis=0)
    lid_temp_mean = np.mean(lid_temp_values, axis=0)
    lid_slip_mean = np.mean(lid_slip_values, axis=0)
    lid_u_ci = confidence_halfwidth(lid_u_values)
    lid_temp_ci = confidence_halfwidth(lid_temp_values)
    lid_slip_ci = confidence_halfwidth(lid_slip_values)
    lid_u_smooth = moving_average(lid_u_mean, window)
    lid_temp_smooth = moving_average(lid_temp_mean, window)
    lid_slip_smooth = moving_average(lid_slip_mean, window)

    csv_path = output / "ensemble_lid_profile.csv"
    np.savetxt(
        csv_path,
        np.column_stack(
            (
                x / length,
                lid_u_mean / lid_speed,
                lid_u_ci / lid_speed,
                lid_slip_mean,
                lid_slip_ci,
                lid_temp_mean,
                lid_temp_ci,
                lid_u_smooth / lid_speed,
                lid_slip_smooth,
                lid_temp_smooth,
            )
        ),
        delimiter=",",
        header=(
            "x_over_L,u_mean_over_Ulid,u_CI95_halfwidth_over_Ulid,"
            "slip_mean_over_Ulid,slip_CI95_halfwidth,temperature_mean_K,"
            "temperature_CI95_halfwidth_K,u_mean_11cell_over_Ulid,"
            "slip_mean_11cell_over_Ulid,temperature_mean_11cell_K"
        ),
        comments="",
    )

    speed_mean = np.sqrt(u_mean**2 + v_mean**2)
    summary: dict[str, object] = {
        "case": "two-dimensional rarefied lid-driven cavity",
        "source_job_array": 62778322,
        "seeds": seeds,
        "independent_seed_count": len(seeds),
        "kn": float(first_meta["kn"]),
        "grid": [int(first_meta["nx"]), int(first_meta["ny"])],
        "particles_per_cell": int(first_meta["particles_per_cell"]),
        "nominal_simulator_particles": int(first_meta["nparticles"]),
        "wall_temperature_K": float(first_meta["wall_temperature_K"]),
        "lid_velocity_m_per_s": lid_speed,
        "warmup_steps": int(first_meta["warmup_steps"]),
        "sample_steps": int(first_meta["sample_steps"]),
        "sample_stride": int(first_meta["sample_stride"]),
        "lid_temperature_mean_K": float(np.mean(lid_temp_mean)),
        "lid_temperature_11cell_min_K": float(np.min(lid_temp_smooth)),
        "lid_temperature_11cell_max_K": float(np.max(lid_temp_smooth)),
        "domain_temperature_mean_min_K": float(np.min(temp_mean)),
        "domain_temperature_mean_max_K": float(np.max(temp_mean)),
        "domain_mean_speed_max_m_per_s": float(np.max(speed_mean)),
        "mean_cellwise_seed_standard_deviation_temperature_K": float(
            np.mean(np.std(temp_values, axis=0, ddof=1))
        ),
        "mean_lid_seed_standard_deviation_temperature_K": float(
            np.mean(np.std(lid_temp_values, axis=0, ddof=1))
        ),
        "uncertainty": "two-sided 95% Student-t interval across independent seeds",
        "display_filter": f"centred {window}-cell lid average and sigma=1-cell field Gaussian; raw values retained in CSV",
        "runtime_by_seed": runtime_by_seed,
    }
    (output / "ensemble_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    xx = x / length
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    axes[0].fill_between(xx, lid_slip_mean - lid_slip_ci, lid_slip_mean + lid_slip_ci, color="#176B87", alpha=0.16, linewidth=0, label="95% seed interval")
    axes[0].plot(xx, lid_slip_mean, color="0.68", lw=0.75, label="Raw ensemble mean")
    axes[0].plot(xx, lid_slip_smooth, color="#176B87", lw=2.1, label=f"{window}-cell display average")
    axes[0].set(xlabel=r"$x/L$", ylabel=r"$(U_{lid}-u)/U_{lid}$")
    axes[1].fill_between(xx, lid_temp_mean - lid_temp_ci, lid_temp_mean + lid_temp_ci, color="#B7472A", alpha=0.16, linewidth=0, label="95% seed interval")
    axes[1].plot(xx, lid_temp_mean, color="0.68", lw=0.75, label="Raw ensemble mean")
    axes[1].plot(xx, lid_temp_smooth, color="#B7472A", lw=2.1, label=f"{window}-cell display average")
    axes[1].set(xlabel=r"$x/L$", ylabel="Temperature [K]")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    fig.savefig(output / "ensemble_lid_profiles.png", dpi=260)
    fig.savefig(output / "ensemble_lid_profiles.pdf")
    plt.close(fig)

    sigma = 1.0
    temp_display = gaussian_filter(temp_mean, sigma=sigma, mode="nearest")
    u_display = gaussian_filter(u_mean, sigma=sigma, mode="nearest")
    v_display = gaussian_filter(v_mean, sigma=sigma, mode="nearest")
    speed_display = np.sqrt(u_display**2 + v_display**2)
    xnorm = x / length
    ynorm = y / length
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 4.35),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    field_specs = (
        (temp_display, "inferno", "Temperature [K]"),
        (speed_display, "viridis", "Speed [m/s]"),
    )
    for axis, (field, cmap, colorbar_label) in zip(axes, field_specs):
        # The dump coordinates are cell centers.  A fixed image extent fills the
        # complete physical cavity and avoids artificial white strips at x/L=1
        # or y/L=1.  Both panels show the same continuous streamline topology.
        image = axis.imshow(
            field,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            interpolation="nearest",
            cmap=cmap,
            aspect="equal",
        )
        stream = axis.streamplot(
            xnorm,
            ynorm,
            u_display,
            v_display,
            color="white",
            density=1.08,
            linewidth=0.52,
            arrowsize=0.72,
            minlength=0.08,
            maxlength=4.0,
            zorder=3,
        )
        stream.lines.set_alpha(0.88)
        stream.arrows.set_alpha(0.88)
        axis.set(
            xlabel=r"$x/L$",
            ylabel=r"$y/L$",
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            aspect="equal",
        )
        axis.set_xticks(np.linspace(0.0, 1.0, 6))
        axis.set_yticks(np.linspace(0.0, 1.0, 6))
        axis.tick_params(labelleft=True)
        fig.colorbar(image, ax=axis, label=colorbar_label, fraction=0.047, pad=0.035)
    fig.savefig(output / "ensemble_fields.png", dpi=250, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output / "ensemble_fields.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window", type=int, default=11)
    args = parser.parse_args()
    ensemble(
        [path.resolve() for path in args.run_dirs],
        args.output.resolve(),
        args.window,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
