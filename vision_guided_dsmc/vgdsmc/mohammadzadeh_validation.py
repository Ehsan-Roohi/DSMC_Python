from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .ntc_solver import run_physical_cavity_ntc
from .vhs_model import PhysicalCavityConfig


REFERENCE_SUBDIR = Path("reference_data") / "mohammadzadeh_2012"
SOURCE_PDF_SHA256 = (
    "9582052a1c9a7a6ab7df93decd8ccfa6d61c13788f9338f31bde1894ead0e93c"
)


def reference_directory() -> Path:
    return Path(__file__).resolve().parents[1] / REFERENCE_SUBDIR


def mohammadzadeh_config(
    *,
    grid: int,
    particles_per_cell: int,
    steps: int,
    sample_start: int,
    seed: int,
    dt_safety: float = 0.20,
) -> PhysicalCavityConfig:
    return PhysicalCavityConfig(
        nx=grid,
        ny=grid,
        particles_per_cell=particles_per_cell,
        length=1.0e-6,
        knudsen=0.05,
        t_left=300.0,
        t_right=300.0,
        t_top=300.0,
        t_bottom=300.0,
        steps=steps,
        sample_start=sample_start,
        dt_safety=dt_safety,
        seed=seed,
        lid_velocity_x=100.0,
        strict_sbt_probability=True,
        stratified_initialization=True,
    )


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in rows[0]
        if key not in {"sampling"}
    } | (
        {"sampling": np.asarray([row["sampling"] for row in rows])}
        if "sampling" in rows[0]
        else {}
    )


def _profile_at_x(field: np.ndarray, x_over_l: float) -> np.ndarray:
    nx = field.shape[1]
    centers = (np.arange(nx, dtype=float) + 0.5) / nx
    return np.asarray(
        [np.interp(x_over_l, centers, row) for row in field],
        dtype=np.float64,
    )


def _profile_at_y(field: np.ndarray, y_over_l: float) -> np.ndarray:
    ny = field.shape[0]
    centers = (np.arange(ny, dtype=float) + 0.5) / ny
    return np.asarray(
        [np.interp(y_over_l, centers, field[:, column]) for column in range(field.shape[1])],
        dtype=np.float64,
    )


def _nrmse_over_reference_rms(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean((actual - reference) ** 2))
        / max(float(np.sqrt(np.mean(reference**2))), 1.0e-300)
    )


def _zero_crossings(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    crossings: list[float] = []
    for index in range(len(x) - 1):
        left, right = float(y[index]), float(y[index + 1])
        if left == 0.0:
            crossings.append(float(x[index]))
        elif left * right < 0.0:
            fraction = -left / (right - left)
            crossings.append(float(x[index] + fraction * (x[index + 1] - x[index])))
    return np.asarray(crossings, dtype=np.float64)


def _nearest_crossing_error(
    actual_x: np.ndarray,
    actual_y: np.ndarray,
    reference_x: np.ndarray,
    reference_y: np.ndarray,
) -> float:
    actual = _zero_crossings(actual_x, actual_y)
    reference = _zero_crossings(reference_x, reference_y)
    if len(actual) == 0 or len(reference) == 0:
        return float("inf")
    return float(np.min(np.abs(actual[:, None] - reference[None, :])))


def evaluate_mohammadzadeh_fields(
    fields: dict[str, np.ndarray],
    cfg: PhysicalCavityConfig,
    *,
    reference_dir: Path | None = None,
) -> dict[str, Any]:
    """Compare a field ensemble with the locked Kn=0.05 PRE profiles."""
    ref_dir = reference_directory() if reference_dir is None else reference_dir
    protocol = json.loads((ref_dir / "validation_protocol.json").read_text())
    if not np.isclose(cfg.knudsen, 0.05) or not np.isclose(cfg.lid_velocity_x, 100.0):
        raise ValueError("The locked primary case is Kn=0.05 and U_wall=100 m/s")

    x_centers = (np.arange(cfg.nx, dtype=float) + 0.5) / cfg.nx
    y_centers = (np.arange(cfg.ny, dtype=float) + 0.5) / cfg.ny
    slip_reference = _read_csv(ref_dir / "fig4_wall_slip_profiles.csv")
    slip_selection = (
        (slip_reference["sampling"] == "macroscopic")
        & np.isclose(slip_reference["kn"], 0.05)
    )
    slip_x = slip_reference["x_over_l"][slip_selection]
    slip_y = slip_reference["u_slip_over_uwall"][slip_selection]
    # The plotted slip is the lid-to-gas tangential speed difference.
    simulated_slip = 1.0 - fields["u"][-1] / cfg.lid_velocity_x
    slip_actual = np.interp(slip_x, x_centers, simulated_slip)
    central = (slip_x >= 0.05) & (slip_x <= 0.95)
    slip_nrmse = _nrmse_over_reference_rms(
        slip_actual[central],
        slip_y[central],
    )
    slip_max = float(np.max(np.abs(slip_actual[central] - slip_y[central])))
    microscopic_slip_selection = (
        (slip_reference["sampling"] == "microscopic")
        & np.isclose(slip_reference["kn"], 0.05)
    )
    microscopic_slip_x = slip_reference["x_over_l"][microscopic_slip_selection]
    microscopic_slip_y = slip_reference["u_slip_over_uwall"][microscopic_slip_selection]
    microscopic_slip_actual = np.interp(
        microscopic_slip_x,
        x_centers,
        fields.get(
            "microscopic_lid_slip_over_uwall",
            np.full(cfg.nx, np.nan),
        ),
    )
    microscopic_slip_central = (
        (microscopic_slip_x >= 0.05)
        & (microscopic_slip_x <= 0.95)
        & np.isfinite(microscopic_slip_actual)
    )
    microscopic_slip_nrmse = (
        _nrmse_over_reference_rms(
            microscopic_slip_actual[microscopic_slip_central],
            microscopic_slip_y[microscopic_slip_central],
        )
        if np.any(microscopic_slip_central)
        else float("inf")
    )
    microscopic_slip_max = (
        float(
            np.max(
                np.abs(
                    microscopic_slip_actual[microscopic_slip_central]
                    - microscopic_slip_y[microscopic_slip_central]
                )
            )
        )
        if np.any(microscopic_slip_central)
        else float("inf")
    )

    lid_temperature_reference = _read_csv(
        ref_dir / "fig5_wall_temperature_profiles.csv"
    )
    lid_selection = (
        (lid_temperature_reference["sampling"] == "macroscopic")
        & np.isclose(lid_temperature_reference["kn"], 0.05)
    )
    lid_x = lid_temperature_reference["x_over_l"][lid_selection]
    lid_t = lid_temperature_reference["temperature_K"][lid_selection]
    lid_actual = np.interp(lid_x, x_centers, fields["T"][-1])
    lid_central = (lid_x >= 0.05) & (lid_x <= 0.95)
    lid_mae = float(np.mean(np.abs(lid_actual[lid_central] - lid_t[lid_central])))
    lid_max = float(np.max(np.abs(lid_actual[lid_central] - lid_t[lid_central])))
    microscopic_lid_selection = (
        (lid_temperature_reference["sampling"] == "microscopic")
        & np.isclose(lid_temperature_reference["kn"], 0.05)
    )
    microscopic_lid_x = lid_temperature_reference["x_over_l"][microscopic_lid_selection]
    microscopic_lid_t = lid_temperature_reference["temperature_K"][microscopic_lid_selection]
    microscopic_lid_actual = np.interp(
        microscopic_lid_x,
        x_centers,
        fields.get("microscopic_lid_T", np.full(cfg.nx, np.nan)),
    )
    microscopic_lid_central = (
        (microscopic_lid_x >= 0.05)
        & (microscopic_lid_x <= 0.95)
        & np.isfinite(microscopic_lid_actual)
    )
    microscopic_lid_mae = (
        float(
            np.mean(
                np.abs(
                    microscopic_lid_actual[microscopic_lid_central]
                    - microscopic_lid_t[microscopic_lid_central]
                )
            )
        )
        if np.any(microscopic_lid_central)
        else float("inf")
    )
    microscopic_lid_max = (
        float(
            np.max(
                np.abs(
                    microscopic_lid_actual[microscopic_lid_central]
                    - microscopic_lid_t[microscopic_lid_central]
                )
            )
        )
        if np.any(microscopic_lid_central)
        else float("inf")
    )

    vertical_reference = _read_csv(
        ref_dir / "fig9b_dsmc_temperature_profile_x08.csv"
    )
    vertical_actual_field = _profile_at_x(fields["T"], 0.8)
    vertical_y = vertical_reference["y_over_l"]
    vertical_t = vertical_reference["temperature_K"]
    vertical_actual = np.interp(vertical_y, y_centers, vertical_actual_field)
    vertical_mae = float(np.mean(np.abs(vertical_actual - vertical_t)))

    q_reference = _read_csv(ref_dir / "fig9b_dsmc_qy_profile_y08.csv")
    q_actual_field = _profile_at_y(fields["qy"], 0.8)
    positive_max = float(np.max(q_actual_field))
    if positive_max <= 0.0:
        q_normalized = np.full_like(q_actual_field, np.nan)
    else:
        q_normalized = q_actual_field / positive_max
    q_x = q_reference["x_over_l"]
    q_ref = q_reference["qy_over_q0"]
    q_actual = np.interp(q_x, x_centers, q_normalized)
    finite_q = np.isfinite(q_actual)
    q_nrmse = (
        _nrmse_over_reference_rms(q_actual[finite_q], q_ref[finite_q])
        if np.any(finite_q)
        else float("inf")
    )
    sign_mask = finite_q & (np.abs(q_ref) >= 0.05)
    sign_agreement = (
        float(np.mean(np.sign(q_actual[sign_mask]) == np.sign(q_ref[sign_mask])))
        if np.any(sign_mask)
        else 0.0
    )
    crossing_error = _nearest_crossing_error(
        x_centers,
        q_normalized,
        q_x,
        q_ref,
    )

    gates = protocol["gates"]
    checks = {
        "fig4_slip_nrmse": slip_nrmse
        <= gates["fig4_macroscopic_slip"]["nrmse_over_reference_rms_max"],
        "fig4_slip_max_abs": slip_max
        <= gates["fig4_macroscopic_slip"]["maximum_absolute_error_max"],
        "fig4_microscopic_slip_nrmse": microscopic_slip_nrmse
        <= gates["fig4_microscopic_slip"]["nrmse_over_reference_rms_max"],
        "fig4_microscopic_slip_max_abs": microscopic_slip_max
        <= gates["fig4_microscopic_slip"]["maximum_absolute_error_max"],
        "fig5_temperature_mae": lid_mae
        <= gates["fig5_macroscopic_lid_temperature"]["mae_K_max"],
        "fig5_temperature_max_abs": lid_max
        <= gates["fig5_macroscopic_lid_temperature"]["maximum_absolute_error_K_max"],
        "fig5_microscopic_temperature_mae": microscopic_lid_mae
        <= gates["fig5_microscopic_lid_temperature"]["mae_K_max"],
        "fig5_microscopic_temperature_max_abs": microscopic_lid_max
        <= gates["fig5_microscopic_lid_temperature"]["maximum_absolute_error_K_max"],
        "fig9_temperature_mae": vertical_mae
        <= gates["fig9_temperature_x08"]["mae_K_max"],
        "fig9_qy_nrmse": q_nrmse
        <= gates["fig9_qy_y08"]["nrmse_over_reference_rms_max"],
        "fig9_qy_sign": sign_agreement
        >= gates["fig9_qy_y08"]["sign_agreement_min"],
        "fig9_qy_zero_crossing": crossing_error
        <= gates["fig9_qy_y08"]["zero_crossing_error_x_over_l_max"],
        "fig6_temperature_min": abs(float(np.min(fields["T"])) - 297.0)
        <= gates["fig6e_temperature_contour"]["extreme_error_K_max"],
        "fig6_temperature_max": abs(float(np.max(fields["T"])) - 309.0)
        <= gates["fig6e_temperature_contour"]["extreme_error_K_max"],
    }
    eligible = (
        cfg.nx == 200
        and cfg.ny == 200
        and cfg.particles_per_cell == 32
        and cfg.steps >= 200_000
        and cfg.sample_start >= 80_000
    )
    return {
        "source_pdf_sha256": SOURCE_PDF_SHA256,
        "protocol_version": protocol["protocol_version"],
        "eligible_single_run_geometry_and_duration": eligible,
        "confirmatory_ensemble_required": True,
        "metrics": {
            "fig4_slip_nrmse_over_reference_rms": slip_nrmse,
            "fig4_slip_max_absolute_error": slip_max,
            "fig4_microscopic_slip_nrmse_over_reference_rms": microscopic_slip_nrmse,
            "fig4_microscopic_slip_max_absolute_error": microscopic_slip_max,
            "fig5_lid_temperature_mae_K": lid_mae,
            "fig5_lid_temperature_max_absolute_error_K": lid_max,
            "fig5_microscopic_lid_temperature_mae_K": microscopic_lid_mae,
            "fig5_microscopic_lid_temperature_max_absolute_error_K": microscopic_lid_max,
            "fig9_temperature_x08_mae_K": vertical_mae,
            "fig9_qy_nrmse_over_reference_rms": q_nrmse,
            "fig9_qy_sign_agreement": sign_agreement,
            "fig9_qy_zero_crossing_error_x_over_l": crossing_error,
            "temperature_min_K": float(np.min(fields["T"])),
            "temperature_max_K": float(np.max(fields["T"])),
        },
        "checks": checks,
        "comparison_arrays": {
            "x_centers": x_centers,
            "y_centers": y_centers,
            "simulated_slip": simulated_slip,
            "reference_slip_x": slip_x,
            "reference_slip": slip_y,
            "reference_microscopic_slip_x": microscopic_slip_x,
            "reference_microscopic_slip": microscopic_slip_y,
            "simulated_microscopic_slip": fields.get(
                "microscopic_lid_slip_over_uwall",
                np.full(cfg.nx, np.nan),
            ),
            "reference_lid_temperature_x": lid_x,
            "reference_lid_temperature_K": lid_t,
            "reference_microscopic_lid_temperature_x": microscopic_lid_x,
            "reference_microscopic_lid_temperature_K": microscopic_lid_t,
            "simulated_microscopic_lid_temperature_K": fields.get(
                "microscopic_lid_T",
                np.full(cfg.nx, np.nan),
            ),
            "simulated_vertical_temperature_K": vertical_actual_field,
            "reference_vertical_temperature_y": vertical_y,
            "reference_vertical_temperature_K": vertical_t,
            "simulated_qy_normalized": q_normalized,
            "reference_qy_x": q_x,
            "reference_qy_normalized": q_ref,
        },
        "decision": (
            "not_eligible_for_validation"
            if not eligible
            else "single_run_only_requires_eight_seed_confirmatory_ensemble"
        ),
    }


def plot_mohammadzadeh_comparison(
    fields: dict[str, np.ndarray],
    report: dict[str, Any],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    arrays = report["comparison_arrays"]
    x = arrays["x_centers"]
    y = arrays["y_centers"]
    xx, yy = np.meshgrid(x, y)
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), constrained_layout=True)

    contour = axes[0, 0].contourf(xx, yy, fields["T"], levels=21, cmap="coolwarm")
    skip = max(1, fields["T"].shape[0] // 20)
    axes[0, 0].quiver(
        xx[::skip, ::skip],
        yy[::skip, ::skip],
        fields["qx"][::skip, ::skip],
        fields["qy"][::skip, ::skip],
        color="black",
        alpha=0.55,
        angles="xy",
    )
    figure.colorbar(contour, ax=axes[0, 0], label="T (K)")
    axes[0, 0].set(title="Temperature contour and conductive heat flux", xlabel="x/L", ylabel="y/L", aspect="equal")

    axes[0, 1].plot(
        arrays["reference_slip_x"],
        arrays["reference_slip"],
        "o",
        ms=3.5,
        label="PRE Fig. 4(b), DSMC macro",
    )
    axes[0, 1].plot(x, arrays["simulated_slip"], lw=1.8, label="Current NTC macro")
    axes[0, 1].plot(
        arrays["reference_microscopic_slip_x"],
        arrays["reference_microscopic_slip"],
        "x",
        ms=3.5,
        label="PRE Fig. 4(b), DSMC micro",
    )
    axes[0, 1].plot(
        x,
        arrays["simulated_microscopic_slip"],
        "--",
        lw=1.6,
        label="Current NTC micro",
    )
    axes[0, 1].set(title="Driven-lid slip profile", xlabel="x/L", ylabel=r"$u_{slip}/U_{wall}$", xlim=(0, 1))
    axes[0, 1].legend(fontsize=8)

    axes[0, 2].plot(
        arrays["reference_lid_temperature_x"],
        arrays["reference_lid_temperature_K"],
        "o",
        ms=3.5,
        label="PRE Fig. 5(b), DSMC macro",
    )
    axes[0, 2].plot(x, fields["T"][-1], lw=1.8, label="Current NTC macro")
    axes[0, 2].plot(
        arrays["reference_microscopic_lid_temperature_x"],
        arrays["reference_microscopic_lid_temperature_K"],
        "x",
        ms=3.5,
        label="PRE Fig. 5(b), DSMC micro",
    )
    axes[0, 2].plot(
        x,
        arrays["simulated_microscopic_lid_temperature_K"],
        "--",
        lw=1.6,
        label="Current NTC micro",
    )
    axes[0, 2].set(title="Gas temperature next to driven lid", xlabel="x/L", ylabel="T (K)", xlim=(0, 1))
    axes[0, 2].legend(fontsize=8)

    axes[1, 0].contour(xx, yy, fields["T"], levels=14, colors="black", linewidths=0.65)
    axes[1, 0].streamplot(x, y, fields["qx"], fields["qy"], density=0.8, linewidth=0.7, color="tab:blue")
    axes[1, 0].set(title="Heat-flux topology on temperature isolines", xlabel="x/L", ylabel="y/L", aspect="equal")

    axes[1, 1].plot(
        arrays["reference_vertical_temperature_K"],
        arrays["reference_vertical_temperature_y"],
        lw=2.0,
        label="PRE Fig. 9(b), DSMC",
    )
    axes[1, 1].plot(
        arrays["simulated_vertical_temperature_K"],
        y,
        lw=1.8,
        label="Current NTC",
    )
    axes[1, 1].set(title="Temperature profile at x/L = 0.8", xlabel="T (K)", ylabel="y/L", ylim=(0, 1))
    axes[1, 1].legend(fontsize=8)

    axes[1, 2].plot(
        arrays["reference_qy_x"],
        arrays["reference_qy_normalized"],
        lw=2.0,
        label="PRE Fig. 9(b), DSMC",
    )
    axes[1, 2].plot(x, arrays["simulated_qy_normalized"], lw=1.8, label="Current NTC")
    axes[1, 2].axhline(0.0, color="0.35", lw=0.7)
    axes[1, 2].set(title="Vertical heat flux at y/L = 0.8", xlabel="x/L", ylabel=r"$q_y/q_0$", xlim=(0, 1))
    axes[1, 2].legend(fontsize=8)

    figure.suptitle(
        "Mohammadzadeh et al. (2012), Kn=0.05 — infrastructure smoke test, not validation"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--particles-per-cell", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--sample-start", type=int, default=800)
    parser.add_argument("--seed", type=int, default=91001)
    args = parser.parse_args()

    cfg = mohammadzadeh_config(
        grid=args.grid,
        particles_per_cell=args.particles_per_cell,
        steps=args.steps,
        sample_start=args.sample_start,
        seed=args.seed,
    )
    fields, _, diagnostics = run_physical_cavity_ntc(cfg, return_state=True)
    report = evaluate_mohammadzadeh_fields(fields, cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "fields.npz", **fields)
    summary = {
        "stage": "infrastructure_smoke_not_validation",
        "config": _json_ready(asdict(cfg)),
        "diagnostics": _json_ready(diagnostics),
        "evaluation": {
            key: value
            for key, value in report.items()
            if key != "comparison_arrays"
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_mohammadzadeh_comparison(
        fields,
        report,
        args.output_dir / "comparison.png",
    )
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
