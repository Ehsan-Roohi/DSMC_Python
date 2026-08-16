#!/usr/bin/env python3
"""Generate a shock-fixed SPARTA normal-shock input deck."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil


K_B = 1.380649e-23
MASS = 6.63e-26
DIAMETER_REF = 4.17e-10
TEMPERATURE_REF = 273.0
VISCOSITY_INDEX = 0.81
GAMMA = 5.0 / 3.0

PRESETS = {
    "smoke": {
        "nx": 80,
        "ppc": 8,
        "half_span_lambda": 8.0,
        "warmup": 200,
        "sample": 400,
        "sample_stride": 5,
        "dump_frequency": 400,
        "restart_frequency": 0,
    },
    "pilot": {
        "nx": 320,
        "ppc": 32,
        "half_span_lambda": 12.0,
        "warmup": 12000,
        "sample": 40000,
        "sample_stride": 10,
        "dump_frequency": 10000,
        "restart_frequency": 10000,
    },
    "production": {
        "nx": 600,
        "ppc": 64,
        "half_span_lambda": 15.0,
        "warmup": 40000,
        "sample": 160000,
        "sample_stride": 10,
        "dump_frequency": 40000,
        "restart_frequency": 40000,
    },
}


def rankine_hugoniot(mach: float, temperature_1: float, number_density_1: float) -> dict[str, float]:
    """Return ideal monatomic normal-shock reservoir states."""
    density_ratio = ((GAMMA + 1.0) * mach**2) / ((GAMMA - 1.0) * mach**2 + 2.0)
    pressure_ratio = 1.0 + 2.0 * GAMMA * (mach**2 - 1.0) / (GAMMA + 1.0)
    temperature_ratio = pressure_ratio / density_ratio
    sound_speed_1 = math.sqrt(GAMMA * K_B * temperature_1 / MASS)
    velocity_1 = mach * sound_speed_1
    return {
        "gamma": GAMMA,
        "mach_1": mach,
        "number_density_1": number_density_1,
        "temperature_1": temperature_1,
        "velocity_1": velocity_1,
        "pressure_1": number_density_1 * K_B * temperature_1,
        "density_ratio": density_ratio,
        "pressure_ratio": pressure_ratio,
        "temperature_ratio": temperature_ratio,
        "number_density_2": number_density_1 * density_ratio,
        "temperature_2": temperature_1 * temperature_ratio,
        "velocity_2": velocity_1 / density_ratio,
        "pressure_2": number_density_1 * K_B * temperature_1 * pressure_ratio,
    }


def physical_parameters(
    mach: float,
    temperature_1: float,
    mean_free_path_1: float,
    nx: int,
    ppc: int,
    half_span_lambda: float,
) -> dict[str, float]:
    number_density_1 = 1.0 / (
        math.sqrt(2.0) * math.pi * DIAMETER_REF**2 * mean_free_path_1
    )
    state = rankine_hugoniot(mach, temperature_1, number_density_1)
    length = 2.0 * half_span_lambda * mean_free_path_1
    transverse_width = mean_free_path_1
    dx = length / nx
    # SPARTA dimension=2 uses unit depth.  The transverse one-cell width is
    # explicit, so fnum gives the requested upstream particles per cell.
    fnum = number_density_1 * dx * transverse_width / ppc
    most_probable_2 = math.sqrt(2.0 * K_B * state["temperature_2"] / MASS)
    maximum_characteristic_speed = state["velocity_1"] + 4.0 * most_probable_2
    collision_time_1 = mean_free_path_1 / math.sqrt(2.0 * K_B * temperature_1 / MASS)
    dt_transport = 0.20 * dx / maximum_characteristic_speed
    dt_collision = 0.10 * collision_time_1
    dt = min(dt_transport, dt_collision)
    return {
        **state,
        "mean_free_path_1": mean_free_path_1,
        "number_density_definition": "1/(sqrt(2)*pi*d_ref^2*lambda_1)",
        "domain_length": length,
        "transverse_width": transverse_width,
        "half_span_lambda": half_span_lambda,
        "dx": dx,
        "dx_over_lambda_1": dx / mean_free_path_1,
        "fnum": fnum,
        "target_upstream_particles_per_cell": ppc,
        "estimated_downstream_particles_per_cell": ppc * state["density_ratio"],
        "collision_time_1": collision_time_1,
        "timestep": dt,
        "dt_over_collision_time_1": dt / collision_time_1,
        "transport_cfl_bound": dt * maximum_characteristic_speed / dx,
    }


def write_case(
    output: Path,
    level: str,
    mach: float,
    seed: int,
    temperature_1: float = 300.0,
    mean_free_path_1: float = 1.0e-7,
) -> dict[str, object]:
    preset = PRESETS[level]
    nx = int(preset["nx"])
    values = physical_parameters(
        mach,
        temperature_1,
        mean_free_path_1,
        nx,
        int(preset["ppc"]),
        float(preset["half_span_lambda"]),
    )
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    for name in ("argon.species", "argon.vss"):
        shutil.copy2(root / "data" / name, output / name)

    xlo = -0.5 * values["domain_length"]
    xhi = 0.5 * values["domain_length"]
    width = values["transverse_width"]
    warmup = int(preset["warmup"])
    sample = int(preset["sample"])
    stride = int(preset["sample_stride"])
    dump_frequency = int(preset["dump_frequency"])
    restart_frequency = int(preset["restart_frequency"])
    restart_line = (
        f"restart              {restart_frequency} restart.shock.1 restart.shock.2\n"
        if restart_frequency
        else ""
    )

    deck = f"""# Generated by scripts/generate_case.py; do not hand-edit constants.
# Shock-fixed monatomic-argon normal shock.  xlo is upstream, xhi downstream.

units                si
seed                 {seed}
dimension            2
boundary             o p p

create_box           {xlo:.16e} {xhi:.16e} 0.0 {width:.16e} -0.5 0.5
create_grid          {nx} 1 1
balance_grid         rcb cell

region               upstream_region block {xlo:.16e} 0.0 0.0 {width:.16e} -0.5 0.5
region               downstream_region block 0.0 {xhi:.16e} 0.0 {width:.16e} -0.5 0.5

global               nrho {values['number_density_1']:.16e} fnum {values['fnum']:.16e} temp {temperature_1:.12g}
species              argon.species Ar
mixture              all Ar
mixture              upstream Ar nrho {values['number_density_1']:.16e} temp {temperature_1:.12g} vstream {values['velocity_1']:.16e} 0.0 0.0
mixture              downstream Ar nrho {values['number_density_2']:.16e} temp {values['temperature_2']:.16e} vstream {values['velocity_2']:.16e} 0.0 0.0

create_particles     upstream n 0 region upstream_region twopass
create_particles     downstream n 0 region downstream_region twopass
balance_grid         rcb part
collide              vss all argon.vss
fix                  inject_left emit/face upstream xlo twopass
fix                  inject_right emit/face downstream xhi twopass

timestep             {values['timestep']:.16e}
stats                {max(20, min(2000, warmup // 20))}
stats_style          step cpu np nattempt ncoll
run                  {warmup}

reset_timestep       0
compute              macro grid all all nrho u v w
compute              thermal thermal/grid all all temp press
compute              pflux pflux/grid all all momxx momyy momzz
compute              eflux eflux/grid all all heatx
fix                  avg ave/grid all {stride} 1 {stride} c_macro[*] c_thermal[*] c_pflux[*] c_eflux[*] ave running
dump                 profile grid all {dump_frequency} profile.final.* id xc f_avg[*]
dump_modify          profile pad 8
{restart_line}run                  {sample}
"""
    (output / "in.shock").write_text(deck, encoding="utf-8")

    metadata: dict[str, object] = {
        "schema_version": 1,
        "case_kind": "steady_shock_fixed_normal_shock",
        "solver": "SPARTA",
        "gas": "argon",
        "level": level,
        "seed": seed,
        "nx": nx,
        "ny": 1,
        "warmup_steps": warmup,
        "sample_steps": sample,
        "sample_stride": stride,
        "dump_frequency_steps": dump_frequency,
        "restart_frequency_steps": restart_frequency,
        "argon_mass_kg": MASS,
        "diameter_ref_m": DIAMETER_REF,
        "temperature_ref_K": TEMPERATURE_REF,
        "viscosity_index": VISCOSITY_INDEX,
        "vss_alpha": 1.0,
        "dump_columns_after_id_xc": [
            "number_density",
            "u",
            "v",
            "w",
            "translational_temperature",
            "pressure",
            "Pxx",
            "Pyy",
            "Pzz",
            "qx",
        ],
        "alignment": "translate each realization to its unsmoothed density midpoint",
        "smoothing": "none",
        **values,
    }
    (output / "case_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=PRESETS, default="pilot")
    parser.add_argument("--mach", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--temperature-1", type=float, default=300.0)
    parser.add_argument("--lambda-1", type=float, default=1.0e-7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mach <= 1.0:
        parser.error("--mach must be greater than one")
    if args.temperature_1 <= 0.0 or args.lambda_1 <= 0.0:
        parser.error("--temperature-1 and --lambda-1 must be positive")
    metadata = write_case(
        args.output.resolve(), args.level, args.mach, args.seed,
        args.temperature_1, args.lambda_1,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
