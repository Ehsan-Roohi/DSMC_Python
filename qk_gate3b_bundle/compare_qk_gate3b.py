#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

SPECIES = ["H2", "H", "O2", "O", "OH", "H2O", "HO2", "Ar"]
SPECIES_PEAK_NRMSE_LIMIT = {
    "H2": 0.08, "H": 0.08, "O2": 0.08, "O": 0.08,
    "OH": 0.12, "H2O": 0.12, "HO2": 0.30, "Ar": 0.08,
}
EVENT_REL_LIMIT = {
    "final_particles": 0.01, "accepted_collisions": 0.02,
    "exchange_reactions": 0.06, "recombinations": 0.30, "dissociations": 0.04,
}
EVENT_COUNT_FLOOR = {
    "final_particles": 2.0, "accepted_collisions": 10.0,
    "exchange_reactions": 2.0, "recombinations": 2.0, "dissociations": 2.0,
}


def parse_summary(path: Path) -> dict:
    output = {}
    for line in path.read_text().splitlines():
        if "=" not in line or line.startswith("R"):
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        try:
            output[key] = int(value)
        except ValueError:
            try:
                output[key] = float(value)
            except ValueError:
                output[key] = value
    return output


def read_history(path: Path) -> list[dict]:
    with path.open() as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def interpolated_crossing(x, y, threshold, direction="up") -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    crossed = (
        (lambda value: value >= threshold)
        if direction == "up"
        else (lambda value: value <= threshold)
    )
    if crossed(y[0]):
        return float(x[0])
    for index in range(1, len(x)):
        if not crossed(y[index]):
            continue
        y0, y1 = y[index - 1], y[index]
        if y1 == y0:
            return float(x[index])
        fraction = min(max(float((threshold - y0) / (y1 - y0)), 0.0), 1.0)
        return float(x[index - 1] + fraction * (x[index] - x[index - 1]))
    return -1.0


def nrmse(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(
        np.sqrt(np.mean((a - b) ** 2)) / max(np.mean(np.abs(b)), 1.0e-300)
    )


def peak_nrmse(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(
        np.sqrt(np.mean((a - b) ** 2)) / max(np.max(np.abs(b)), 1.0e-12)
    )


def relative_difference(a, b) -> float:
    return float(abs(a - b) / max(abs(b), 1.0e-300))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortran-root", type=Path, required=True)
    parser.add_argument("--python-json", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--min-seeds", type=int, default=32)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    fortran_dirs = sorted(
        path for path in args.fortran_root.iterdir()
        if path.is_dir() and (path / "qk_gate3_summary.txt").exists()
    )
    if len(fortran_dirs) < args.min_seeds:
        raise SystemExit(
            f"Need at least {args.min_seeds} Fortran runs; found {len(fortran_dirs)}"
        )
    fortran_summary = [
        parse_summary(path / "qk_gate3_summary.txt") for path in fortran_dirs
    ]
    fortran_history = [
        read_history(path / "qk_gate3_history.csv") for path in fortran_dirs
    ]
    python_data = json.loads(args.python_json.read_text())
    python_runs = python_data.get("runs")
    if python_runs is None:
        raise SystemExit(
            "Gate 3B requires a live Python oracle; frozen oracle is not accepted"
        )
    if len(python_runs) != len(fortran_dirs):
        raise SystemExit(
            f"Fortran/Python run count mismatch: "
            f"{len(fortran_dirs)} versus {len(python_runs)}"
        )

    steps = np.array([int(row["step"]) for row in fortran_history[0]])
    distance = np.array(
        [row["distance_m"] for row in fortran_history[0]], dtype=float
    )
    for history in fortran_history:
        if np.any(np.array([int(row["step"]) for row in history]) != steps):
            raise SystemExit("Fortran history grid mismatch")
    for run in python_runs:
        if np.any(
            np.array([int(row["step"]) for row in run["history"]]) != steps
        ):
            raise SystemExit("Python history grid mismatch")

    fortran_profiles = {}
    python_profiles = {}
    for key in ["Ttr_K", "Trot_K", "Tvib_K"]:
        fortran_profiles[key] = np.mean(
            [[row[key] for row in history] for history in fortran_history],
            axis=0,
        )
        python_profiles[key] = np.mean(
            [[row[key] for row in run["history"]] for run in python_runs],
            axis=0,
        )
    fortran_species = np.mean(
        [
            [
                [row[f"X_{species}"] for species in SPECIES]
                for row in history
            ]
            for history in fortran_history
        ],
        axis=0,
    )
    python_species = np.mean(
        [
            [row["mole_fractions"] for row in run["history"]]
            for run in python_runs
        ],
        axis=0,
    )

    metrics = {
        "seed_count": len(fortran_dirs),
        "profile_Ttr_nrmse": nrmse(
            fortran_profiles["Ttr_K"], python_profiles["Ttr_K"]
        ),
        "profile_Trot_nrmse": nrmse(
            fortran_profiles["Trot_K"], python_profiles["Trot_K"]
        ),
        "profile_Tvib_nrmse": nrmse(
            fortran_profiles["Tvib_K"], python_profiles["Tvib_K"]
        ),
    }
    metrics["species_max_abs_diff"] = {
        species: float(
            np.max(
                np.abs(
                    fortran_species[:, index] - python_species[:, index]
                )
            )
        )
        for index, species in enumerate(SPECIES)
    }
    metrics["species_peak_nrmse"] = {
        species: peak_nrmse(
            fortran_species[:, index], python_species[:, index]
        )
        for index, species in enumerate(SPECIES)
    }
    metrics["max_species_abs_diff"] = max(
        metrics["species_max_abs_diff"].values()
    )

    event_names = [
        "final_particles", "accepted_collisions", "exchange_reactions",
        "recombinations", "dissociations",
    ]
    event_metrics = {}
    event_consistency = {}
    for name in event_names:
        fortran_values = np.array(
            [float(row[name]) for row in fortran_summary]
        )
        python_values = np.array(
            [float(row[name]) for row in python_runs]
        )
        standard_error = math.sqrt(
            fortran_values.var(ddof=1) / len(fortran_values)
            + python_values.var(ddof=1) / len(python_values)
        )
        mean_difference = float(
            abs(fortran_values.mean() - python_values.mean())
        )
        statistical_tolerance = float(
            3.0 * standard_error + EVENT_COUNT_FLOOR[name]
        )
        rel = relative_difference(
            fortran_values.mean(), python_values.mean()
        )
        event_metrics[name] = {
            "fortran": float(fortran_values.mean()),
            "python": float(python_values.mean()),
            "relative_difference": rel,
            "fortran_std": float(fortran_values.std(ddof=1)),
            "python_std": float(python_values.std(ddof=1)),
            "combined_standard_error": standard_error,
            "mean_difference": mean_difference,
            "three_sigma_plus_floor_tolerance": statistical_tolerance,
        }
        event_consistency[name] = bool(
            mean_difference <= statistical_tolerance
            and rel < EVENT_REL_LIMIT[name]
        )
    metrics["event_means"] = event_metrics

    dx = float(distance[1] - distance[0])
    oh_fortran = interpolated_crossing(
        distance, fortran_species[:, 4], 0.01, "up"
    )
    oh_python = interpolated_crossing(
        distance, python_species[:, 4], 0.01, "up"
    )
    h2_fortran = interpolated_crossing(
        distance, fortran_species[:, 0], 0.95 * 0.2, "down"
    )
    h2_python = interpolated_crossing(
        distance, python_species[:, 0], 0.95 * 0.2, "down"
    )
    metrics["induction_distances_m"] = {
        "OH_1pct_fortran_interpolated": oh_fortran,
        "OH_1pct_python_interpolated": oh_python,
        "H2_5pct_fortran_interpolated": h2_fortran,
        "H2_5pct_python_interpolated": h2_python,
        "sampling_dx": dx,
    }
    metrics["OH_1pct_relative_difference"] = relative_difference(
        oh_fortran, oh_python
    )
    metrics["H2_5pct_relative_difference"] = relative_difference(
        h2_fortran, h2_python
    )

    max_fortran_errors = {
        "mass": max(float(row["relative_mass_error"]) for row in fortran_summary),
        "momentum": max(
            float(row["relative_momentum_error"]) for row in fortran_summary
        ),
        "energy": max(
            float(row["relative_total_energy_error"]) for row in fortran_summary
        ),
        "H_atom": max(
            float(row["relative_H_atom_error"]) for row in fortran_summary
        ),
        "O_atom": max(
            float(row["relative_O_atom_error"]) for row in fortran_summary
        ),
        "Ar_atom": max(
            float(row["relative_Ar_atom_error"]) for row in fortran_summary
        ),
        "RH_mass": max(
            float(row["rankine_hugoniot_mass_residual"])
            for row in fortran_summary
        ),
        "RH_momentum": max(
            float(row["rankine_hugoniot_momentum_residual"])
            for row in fortran_summary
        ),
        "RH_energy": max(
            float(row["rankine_hugoniot_energy_residual"])
            for row in fortran_summary
        ),
    }
    max_python_errors = {
        "mass": max(float(row["relative_mass_error"]) for row in python_runs),
        "momentum": max(
            float(row["relative_momentum_error"]) for row in python_runs
        ),
        "energy": max(
            float(row["relative_total_energy_error"]) for row in python_runs
        ),
        "H_atom": max(
            float(row["relative_atom_errors"][0]) for row in python_runs
        ),
        "O_atom": max(
            float(row["relative_atom_errors"][1]) for row in python_runs
        ),
        "Ar_atom": max(
            float(row["relative_atom_errors"][2]) for row in python_runs
        ),
    }
    metrics["max_fortran_errors"] = max_fortran_errors
    metrics["max_python_errors"] = max_python_errors

    recombination_fraction_fortran = float(
        np.mean([int(row["recombinations"]) > 0 for row in fortran_summary])
    )
    recombination_fraction_python = float(
        np.mean([int(row["recombinations"]) > 0 for row in python_runs])
    )
    metrics["recombination_nonzero_run_fraction"] = {
        "fortran": recombination_fraction_fortran,
        "python": recombination_fraction_python,
    }
    gates = {
        "live_python_oracle": True,
        "minimum_32_fortran_and_python_seeds": (
            len(fortran_dirs) >= args.min_seeds
            and len(python_runs) >= args.min_seeds
        ),
        "rankine_hugoniot_jump": max(
            max_fortran_errors["RH_mass"],
            max_fortran_errors["RH_momentum"],
            max_fortran_errors["RH_energy"],
        ) < 1.0e-12,
        "fortran_mass_conservation": max_fortran_errors["mass"] < 2.0e-11,
        "fortran_momentum_conservation": (
            max_fortran_errors["momentum"] < 2.0e-11
        ),
        "fortran_total_energy_conservation": (
            max_fortran_errors["energy"] < 2.0e-8
        ),
        "fortran_atom_conservation": max(
            max_fortran_errors["H_atom"],
            max_fortran_errors["O_atom"],
            max_fortran_errors["Ar_atom"],
        ) < 1.0e-14,
        "python_mass_conservation": max_python_errors["mass"] < 2.0e-11,
        "python_momentum_conservation": (
            max_python_errors["momentum"] < 2.0e-11
        ),
        "python_total_energy_conservation": (
            max_python_errors["energy"] < 2.0e-8
        ),
        "python_atom_conservation": max(
            max_python_errors["H_atom"],
            max_python_errors["O_atom"],
            max_python_errors["Ar_atom"],
        ) < 1.0e-14,
        "fortran_particle_identity": all(
            int(row["particle_identity_error"]) == 0
            for row in fortran_summary
        ),
        "python_particle_identity": all(
            int(row["particle_identity_error"]) == 0
            for row in python_runs
        ),
        "exchange_and_dissociation_observed_in_every_run": all(
            int(row["exchange_reactions"]) > 0
            and int(row["dissociations"]) > 0
            for row in [*fortran_summary, *python_runs]
        ),
        "rare_recombination_observed_in_at_least_90pct_of_runs": (
            recombination_fraction_fortran >= 0.90
            and recombination_fraction_python >= 0.90
        ),
        "Ttr_profile_matches_live_oracle": (
            metrics["profile_Ttr_nrmse"] < 0.02
        ),
        "Trot_profile_matches_live_oracle": (
            metrics["profile_Trot_nrmse"] < 0.025
        ),
        "Tvib_profile_matches_live_oracle": (
            metrics["profile_Tvib_nrmse"] < 0.025
        ),
        "species_absolute_profiles_match": (
            metrics["max_species_abs_diff"] < 0.0075
        ),
        "species_peak_normalized_profiles_match": all(
            metrics["species_peak_nrmse"][species]
            < SPECIES_PEAK_NRMSE_LIMIT[species]
            for species in SPECIES
        ),
        **{
            f"{name}_count_statistically_consistent": value
            for name, value in event_consistency.items()
        },
        "OH_interpolated_induction_location_matches": (
            oh_fortran > 0.0
            and oh_python > 0.0
            and metrics["OH_1pct_relative_difference"] < 0.10
        ),
        "H2_interpolated_consumption_location_matches": (
            h2_fortran > 0.0
            and h2_python > 0.0
            and metrics["H2_5pct_relative_difference"] < 0.10
        ),
    }
    gates = {key: bool(value) for key, value in gates.items()}
    passed = bool(all(gates.values()))

    output = {
        "scope": (
            "Gate 3B live 32-seed prescribed post-shock "
            "Q-K induction equivalence"
        ),
        "fortran_run_dirs": [str(path) for path in fortran_dirs],
        "python_reference": str(args.python_json),
        "python_reference_mode": "live_independent_runs",
        "metrics": metrics,
        "limits": {
            "temperature_nrmse": {
                "Ttr": 0.02, "Trot": 0.025, "Tvib": 0.025
            },
            "species_max_abs_diff": 0.0075,
            "species_peak_nrmse": SPECIES_PEAK_NRMSE_LIMIT,
            "event_relative_difference": EVENT_REL_LIMIT,
            "induction_relative_difference": 0.10,
        },
        "gates": gates,
        "pass": passed,
    }
    (args.outdir / "qk_gate3b_comparison.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )

    with (args.outdir / "qk_gate3b_mean_profiles.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step", "distance_m", "F_Ttr_K", "P_Ttr_K",
                "F_Trot_K", "P_Trot_K", "F_Tvib_K", "P_Tvib_K",
            ]
            + [f"F_X_{species}" for species in SPECIES]
            + [f"P_X_{species}" for species in SPECIES]
        )
        for index, step in enumerate(steps):
            writer.writerow(
                [
                    int(step), distance[index],
                    fortran_profiles["Ttr_K"][index],
                    python_profiles["Ttr_K"][index],
                    fortran_profiles["Trot_K"][index],
                    python_profiles["Trot_K"][index],
                    fortran_profiles["Tvib_K"][index],
                    python_profiles["Tvib_K"][index],
                    *fortran_species[index], *python_species[index],
                ]
            )

    lines = [
        "Bird-QK Gate 3B: live 32-seed cross-language induction validation",
        "=" * 72,
        "",
        f"Fortran seeds={len(fortran_dirs)}; live Python seeds={len(python_runs)}",
        "No frozen oracle is accepted in Gate 3B.",
        "Crossing locations use linear interpolation between history points.",
        "",
        f"Ttr profile NRMSE={metrics['profile_Ttr_nrmse']:.3%}",
        f"Trot profile NRMSE={metrics['profile_Trot_nrmse']:.3%}",
        f"Tvib profile NRMSE={metrics['profile_Tvib_nrmse']:.3%}",
        (
            "maximum species absolute difference="
            f"{metrics['max_species_abs_diff']:.6e}"
        ),
        "",
        "Species peak-normalized NRMSE",
    ]
    lines.extend(
        (
            f"{species:4s}: {metrics['species_peak_nrmse'][species]:.3%} "
            f"(limit {SPECIES_PEAK_NRMSE_LIMIT[species]:.1%})"
        )
        for species in SPECIES
    )
    lines.extend(["", "Event-count ensemble comparisons"])
    for name in event_names:
        item = event_metrics[name]
        lines.append(
            f"{name:22s}: {item['fortran']:12.3f} "
            f"vs {item['python']:12.3f}; "
            f"rel.diff={item['relative_difference']:.3%}; "
            f"|delta|={item['mean_difference']:.3f}; "
            f"3SE+floor={item['three_sigma_plus_floor_tolerance']:.3f}"
        )
    lines.extend(
        [
            "",
            (
                f"Interpolated OH 1% distance: Fortran={oh_fortran:.6e} m "
                f"Python={oh_python:.6e} m "
                f"rel.diff={metrics['OH_1pct_relative_difference']:.3%}"
            ),
            (
                f"Interpolated H2 5% distance: Fortran={h2_fortran:.6e} m "
                f"Python={h2_python:.6e} m "
                f"rel.diff={metrics['H2_5pct_relative_difference']:.3%}"
            ),
            "",
            "Gate results",
        ]
    )
    lines.extend(
        f"{key}: {'PASS' if value else 'FAIL'}"
        for key, value in gates.items()
    )
    lines.extend(["", f"OVERALL={'PASS' if passed else 'FAIL'}"])
    report = "\n".join(lines) + "\n"
    (args.outdir / "QK_GATE3B_VALIDATION_REPORT.txt").write_text(report)
    print(report, end="")
    print(
        "QK_GATE3B_LIVE_CROSS_LANGUAGE_PASS"
        if passed
        else "QK_GATE3B_LIVE_CROSS_LANGUAGE_FAIL"
    )
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
