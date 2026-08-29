#!/usr/bin/env python3
"""Design a shock-triggered ignition window before launching more DSMC.

The scan combines a frozen, calorically-imperfect normal-shock calculation
with Cantera constant-volume induction delays.  It searches stagnation state,
upstream Mach number, post-shock duct length, and argon dilution while enforcing
three independent requirements:

* negligible pre-shock reaction (Da_pre < 0.1),
* dynamically relevant post-shock reaction (Da_post >= 0.3, with >=1 strong),
* a local mean-free-path screen based on the nozzle throat height.

The result is a physics design map, not a replacement for DSMC or a reacting
quasi-1D nozzle solution.  Its purpose is to identify a small, defensible set
of conditions and geometry targets for the next expensive stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from gate5_physics_prescreen import (
    MECHANISM,
    bisect_root,
    equilibrium_cj_screen,
    ignition_delay,
    import_cantera,
    safe_ratio,
)


ALL_CASES_NAME = "QK_GATE5_MACH_RESIDENCE_ALL_CASES.csv"
CANDIDATES_NAME = "QK_GATE5_MACH_RESIDENCE_CANDIDATES.csv"
DESIGN_JSON_NAME = "QK_GATE5_MACH_RESIDENCE_DESIGN.json"
DESIGN_SUMMARY_NAME = "QK_GATE5_MACH_RESIDENCE_DESIGN.txt"


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(not math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("expected a comma-separated finite numeric list")
    return values


def bracketed_root_from_scan(
    function: Callable[[float], float], values: Sequence[float]
) -> Optional[float]:
    previous_x = values[0]
    previous_f = function(previous_x)
    for current_x in values[1:]:
        current_f = function(current_x)
        if (
            math.isfinite(previous_f) and math.isfinite(current_f)
            and previous_f * current_f <= 0.0
        ):
            return bisect_root(function, previous_x, current_x)
        previous_x, previous_f = current_x, current_f
    return None


def frozen_isentropic_static_state(
    ct: Any, temperature0_k: float, pressure0_pa: float, mach: float, mixture: str
) -> dict[str, float]:
    gas = ct.Solution(MECHANISM)
    gas.TPX = temperature0_k, pressure0_pa, mixture
    h0 = float(gas.enthalpy_mass)
    s0 = float(gas.entropy_mass)

    def energy_residual(temperature: float) -> float:
        gas.TPX = temperature, pressure0_pa, mixture
        return float(gas.enthalpy_mass + 0.5 * (mach * gas.sound_speed) ** 2 - h0)

    temperature1 = bisect_root(energy_residual, max(80.0, 0.05 * temperature0_k), temperature0_k)
    if temperature1 is None:
        raise ValueError("unable to solve frozen isentropic static temperature")

    def entropy_residual(log_pressure: float) -> float:
        gas.TPX = temperature1, math.exp(log_pressure), mixture
        return float(gas.entropy_mass - s0)

    log_pressure1 = bisect_root(
        entropy_residual,
        math.log(max(1.0e-8 * pressure0_pa, 1.0e-3)),
        math.log(pressure0_pa),
    )
    if log_pressure1 is None:
        raise ValueError("unable to solve frozen isentropic static pressure")
    pressure1 = math.exp(log_pressure1)
    gas.TPX = temperature1, pressure1, mixture
    return {
        "T_K": temperature1,
        "pressure_Pa": pressure1,
        "rho_kg_m3": float(gas.density),
        "sound_speed_m_s": float(gas.sound_speed),
        "velocity_m_s": mach * float(gas.sound_speed),
        "Mach": mach,
        "h_J_kg": float(gas.enthalpy_mass),
        "gamma": float(gas.cp_mass / gas.cv_mass),
    }


def frozen_normal_shock(
    ct: Any, upstream: dict[str, float], mixture: str
) -> dict[str, float]:
    gas = ct.Solution(MECHANISM)
    rho1 = upstream["rho_kg_m3"]
    pressure1 = upstream["pressure_Pa"]
    velocity1 = upstream["velocity_m_s"]
    total_energy = upstream["h_J_kg"] + 0.5 * velocity1**2
    gas.TPX = upstream["T_K"], pressure1, mixture
    gas_constant = ct.gas_constant / gas.mean_molecular_weight

    def energy_residual(density_ratio: float) -> float:
        velocity2 = velocity1 / density_ratio
        pressure2 = pressure1 + rho1 * velocity1**2 * (1.0 - 1.0 / density_ratio)
        rho2 = density_ratio * rho1
        temperature2 = pressure2 / (rho2 * gas_constant)
        if temperature2 <= 0.0 or pressure2 <= 0.0:
            return math.nan
        gas.TPX = temperature2, pressure2, mixture
        return float(gas.enthalpy_mass + 0.5 * velocity2**2 - total_energy)

    ratios = [1.02 + index * (9.98 / 250.0) for index in range(251)]
    density_ratio = bracketed_root_from_scan(energy_residual, ratios)
    if density_ratio is None:
        raise ValueError("unable to solve non-trivial frozen normal shock")
    velocity2 = velocity1 / density_ratio
    pressure2 = pressure1 + rho1 * velocity1**2 * (1.0 - 1.0 / density_ratio)
    rho2 = density_ratio * rho1
    temperature2 = pressure2 / (rho2 * gas_constant)
    gas.TPX = temperature2, pressure2, mixture
    return {
        "T_K": temperature2,
        "pressure_Pa": pressure2,
        "rho_kg_m3": rho2,
        "sound_speed_m_s": float(gas.sound_speed),
        "velocity_m_s": velocity2,
        "Mach": velocity2 / float(gas.sound_speed),
        "density_ratio": density_ratio,
        "pressure_ratio": pressure2 / pressure1,
        "temperature_ratio": temperature2 / upstream["T_K"],
        "gamma": float(gas.cp_mass / gas.cv_mass),
    }


def mean_free_path_screen(
    ct: Any, temperature_k: float, pressure_pa: float, mixture: str
) -> float:
    gas = ct.Solution(MECHANISM)
    gas.TPX = temperature_k, pressure_pa, mixture
    gas_constant = ct.gas_constant / gas.mean_molecular_weight
    return float(gas.viscosity / pressure_pa * math.sqrt(math.pi * gas_constant * temperature_k / 2.0))


def evaluate_design_point(
    ct: Any,
    temperature0_k: float,
    pressure0_pa: float,
    mach1: float,
    post_length_m: float,
    pre_length_m: float,
    throat_height_m: float,
    argon_moles: float,
    ignition_horizon_s: float,
) -> dict[str, Any]:
    mixture = f"H2:2,O2:1,AR:{argon_moles:g}" if argon_moles > 0.0 else "H2:2,O2:1"
    upstream = frozen_isentropic_static_state(ct, temperature0_k, pressure0_pa, mach1, mixture)
    sonic = frozen_isentropic_static_state(ct, temperature0_k, pressure0_pa, 1.0, mixture)
    downstream = frozen_normal_shock(ct, upstream, mixture)
    area_ratio = (
        sonic["rho_kg_m3"] * sonic["velocity_m_s"]
        / (upstream["rho_kg_m3"] * upstream["velocity_m_s"])
    )
    # The expanding pre-shock stream is slower than u1 over part of the path;
    # 0.7*u1 is a transparent conservative approximation for pre-ignition risk.
    pre_residence = pre_length_m / (0.7 * upstream["velocity_m_s"])
    post_residence = post_length_m / downstream["velocity_m_s"]
    horizon = max(ignition_horizon_s, 100.0 * pre_residence, 100.0 * post_residence)
    pre_ignition = ignition_delay(
        ct, upstream["T_K"], upstream["pressure_Pa"], horizon, mixture=mixture
    )
    post_ignition = ignition_delay(
        ct, downstream["T_K"], downstream["pressure_Pa"], horizon, mixture=mixture
    )
    tau_pre = pre_ignition.get("tau_s")
    tau_post = post_ignition.get("tau_s")
    da_pre = safe_ratio(pre_residence, tau_pre)
    da_post = safe_ratio(post_residence, tau_post)
    lambda1 = mean_free_path_screen(ct, upstream["T_K"], upstream["pressure_Pa"], mixture)
    lambda2 = mean_free_path_screen(ct, downstream["T_K"], downstream["pressure_Pa"], mixture)
    kn1 = lambda1 / throat_height_m
    kn2 = lambda2 / throat_height_m
    pre_safe = da_pre is None or da_pre < 0.10
    post_relevant = da_post is not None and da_post >= 0.30
    post_strong = da_post is not None and da_post >= 1.0
    rarefied_screen = max(kn1, kn2) >= 0.005
    return {
        "T0_K": temperature0_k,
        "p0_Pa": pressure0_pa,
        "M1_target": mach1,
        "required_area_ratio_A_over_Astar": area_ratio,
        "required_height_at_shock_m": throat_height_m * area_ratio,
        "post_length_m": post_length_m,
        "pre_length_m": pre_length_m,
        "throat_height_m": throat_height_m,
        "argon_moles_per_2H2_1O2": argon_moles,
        "mixture": mixture,
        "T1_K": upstream["T_K"],
        "p1_Pa": upstream["pressure_Pa"],
        "rho1_kg_m3": upstream["rho_kg_m3"],
        "u1_m_s": upstream["velocity_m_s"],
        "gamma1": upstream["gamma"],
        "T2_K": downstream["T_K"],
        "p2_Pa": downstream["pressure_Pa"],
        "rho2_kg_m3": downstream["rho_kg_m3"],
        "u2_m_s": downstream["velocity_m_s"],
        "M2": downstream["Mach"],
        "shock_density_ratio": downstream["density_ratio"],
        "shock_pressure_ratio": downstream["pressure_ratio"],
        "pre_residence_s": pre_residence,
        "post_residence_s": post_residence,
        "tau_pre_s": tau_pre,
        "tau_post_s": tau_post,
        "Da_pre": da_pre,
        "Da_post": da_post,
        "pre_criterion": pre_ignition.get("criterion"),
        "post_criterion": post_ignition.get("criterion"),
        "lambda1_m": lambda1,
        "lambda2_m": lambda2,
        "Kn_height_upstream": kn1,
        "Kn_height_downstream": kn2,
        "pre_safe": pre_safe,
        "post_relevant": post_relevant,
        "post_strong": post_strong,
        "rarefied_screen": rarefied_screen,
        "shock_triggered_candidate": pre_safe and post_relevant and rarefied_screen,
        "strong_shock_triggered_candidate": pre_safe and post_strong and rarefied_screen,
    }


def candidate_rank(row: dict[str, Any]) -> tuple[float, ...]:
    da_post = float(row.get("Da_post") or 0.0)
    da_pre = float(row.get("Da_pre") or 0.0)
    kn = max(float(row["Kn_height_upstream"]), float(row["Kn_height_downstream"]))
    return (
        0.0 if row["strong_shock_triggered_candidate"] else 1.0,
        0.0 if row["shock_triggered_candidate"] else 1.0,
        abs(math.log10(max(da_post, 1.0e-20))),
        da_pre,
        -kn,
        float(row["post_length_m"]),
    )


def select_diverse_candidates(rows: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row["shock_triggered_candidate"]]
    selected: list[dict[str, Any]] = []
    seen_rows: set[tuple[float, float, float, float, float]] = set()

    def add(row: dict[str, Any]) -> None:
        signature = (
            float(row["M1_target"]), float(row["T0_K"]), float(row["p0_Pa"]),
            float(row["argon_moles_per_2H2_1O2"]), float(row["post_length_m"]),
        )
        if signature not in seen_rows and len(selected) < limit:
            seen_rows.add(signature)
            selected.append(dict(row))

    # Preserve the existing 2H2+O2+3Ar chemistry as the primary scientific
    # branch and include its best point at every Mach number before controls.
    if eligible:
        primary_argon = max(float(row["argon_moles_per_2H2_1O2"]) for row in eligible)
        primary = [
            row for row in eligible
            if float(row["argon_moles_per_2H2_1O2"]) == primary_argon
        ]
        for mach in sorted({float(row["M1_target"]) for row in primary}):
            group = [row for row in primary if float(row["M1_target"]) == mach]
            add(min(group, key=candidate_rank))
        # Retain one high-Mach, short-residence primary-mixture point because
        # this is the branch most likely to satisfy the independent CJ screen.
        maximum_mach = max(float(row["M1_target"]) for row in primary)
        high_mach_strong = [
            row for row in primary
            if float(row["M1_target"]) == maximum_mach
            and row["strong_shock_triggered_candidate"]
        ]
        if high_mach_strong:
            add(min(
                high_mach_strong,
                key=lambda row: (
                    float(row["post_length_m"]),
                    abs(math.log10(max(float(row["Da_post"]), 1.0e-20))),
                    -float(row["T0_K"]),
                ),
            ))

    # Add the best point for each Mach/dilution control, then fill globally.
    groups: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in eligible:
        key = (float(row["argon_moles_per_2H2_1O2"]), float(row["M1_target"]))
        groups.setdefault(key, []).append(row)
    for key in sorted(groups, key=lambda item: (-item[0], item[1])):
        add(min(groups[key], key=candidate_rank))
    for row in sorted(eligible, key=candidate_rank):
        add(row)
        if len(selected) >= limit:
            break
    return selected


def add_cj_screens(ct: Any, candidates: list[dict[str, Any]]) -> None:
    cache: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in candidates:
        key = (round(row["T1_K"]), round(row["p1_Pa"]), row["mixture"])
        if key not in cache:
            cache[key] = equilibrium_cj_screen(
                ct, row["T1_K"], row["p1_Pa"], mixture=row["mixture"]
            )
        cj = cache[key]
        row["CJ_speed_screening_m_s"] = cj.get("speed_m_s")
        row["u1_over_CJ"] = safe_ratio(row["u1_m_s"], cj.get("speed_m_s"))
        row["stationary_normal_detonation_screen"] = bool(
            row["u1_over_CJ"] is not None and row["u1_over_CJ"] >= 1.0
        )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def self_test(ct: Any) -> None:
    upstream = frozen_isentropic_static_state(ct, 2000.0, 500000.0, 3.0, "H2:2,O2:1,AR:3")
    downstream = frozen_normal_shock(ct, upstream, "H2:2,O2:1,AR:3")
    assert upstream["Mach"] == 3.0
    assert downstream["Mach"] < 1.0
    assert downstream["pressure_Pa"] > upstream["pressure_Pa"]
    assert downstream["T_K"] > upstream["T_K"]
    total1 = upstream["h_J_kg"] + 0.5 * upstream["velocity_m_s"] ** 2
    gas = ct.Solution(MECHANISM)
    gas.TPX = downstream["T_K"], downstream["pressure_Pa"], "H2:2,O2:1,AR:3"
    total2 = gas.enthalpy_mass + 0.5 * downstream["velocity_m_s"] ** 2
    assert abs(total2 - total1) / abs(total1) < 1.0e-9
    assert mean_free_path_screen(
        ct, downstream["T_K"], downstream["pressure_Pa"], "H2:2,O2:1,AR:3"
    ) > 0.0
    high_temperature = ignition_delay(
        ct, 3500.0, 68399.58, 0.02, mixture="H2:2,O2:1,AR:3"
    )
    assert high_temperature["ignited"]
    assert high_temperature["chemical_event"]
    assert not high_temperature["thermal_event"]
    assert high_temperature["tau_s"] is not None
    print("QK_GATE5_MACH_RESIDENCE_DESIGN_SELF_TEST_PASS")


def make_summary(result: dict[str, Any]) -> str:
    lines = [
        "QK Gate-5 Mach-residence design",
        f"Generated UTC: {result['generated_utc']}",
        f"Grid points evaluated: {result['evaluated_case_count']}",
        f"Shock-triggered candidates: {result['candidate_count']}",
        f"Strong candidates (Da_post >= 1): {result['strong_candidate_count']}",
        f"Selected DSMC design points: {len(result['selected_candidates'])}",
        "",
    ]
    if result["selected_candidates"]:
        lines.append("Selected candidates:")
        for index, row in enumerate(result["selected_candidates"], 1):
            lines.append(
                "  {i}: M1={M:g}, T0={T:g} K, p0={p:.3g} Pa, Ar={Ar:g}, "
                "A/A*={area:.3g}, Lpost={L:.3g} m, Da_pre={Dpre}, "
                "Da_post={Dpost}, Kn={Kn:.3g}, u1/CJ={cj}".format(
                    i=index, M=row["M1_target"], T=row["T0_K"], p=row["p0_Pa"],
                    Ar=row["argon_moles_per_2H2_1O2"], L=row["post_length_m"],
                    area=row["required_area_ratio_A_over_Astar"],
                    Dpre=row["Da_pre"], Dpost=row["Da_post"],
                    Kn=max(row["Kn_height_upstream"], row["Kn_height_downstream"]),
                    cj=row.get("u1_over_CJ"),
                )
            )
    else:
        lines.append("No point met all pre-ignition, post-ignition, and rarefaction constraints.")
    lines.extend([
        "",
        "Guardrails:",
        "  Frozen isentropic/normal-shock states are a design approximation.",
        "  Constant-area Lpost/u2 is a residence-time approximation.",
        "  Kn uses a viscosity-based mean-free-path estimate and the 15 um throat height.",
        "  Selected points require a new geometry/preflight before reacting DSMC production.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--temperatures-k", type=parse_float_list, default=parse_float_list("1500,1750,2000,2250,2500,2750"))
    parser.add_argument("--pressures-pa", type=parse_float_list, default=parse_float_list("500000,1000000,2000000,5000000"))
    parser.add_argument("--mach", type=parse_float_list, default=parse_float_list("2,2.5,3,3.5,4"))
    parser.add_argument("--post-lengths-m", type=parse_float_list, default=parse_float_list("0.0001,0.00025,0.0005,0.001,0.002"))
    parser.add_argument("--argon-moles", type=parse_float_list, default=parse_float_list("0,1,3"))
    parser.add_argument("--pre-length-m", type=float, default=100.0e-6)
    parser.add_argument("--throat-height-m", type=float, default=15.0e-6)
    parser.add_argument("--ignition-horizon-s", type=float, default=0.02)
    parser.add_argument("--candidate-limit", type=int, default=12)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    ct = import_cantera()
    if args.self_test:
        self_test(ct)
        return
    if args.out_dir is None:
        parser.error("--out-dir is required unless --self-test is used")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for argon in args.argon_moles:
        for pressure in args.pressures_pa:
            for temperature in args.temperatures_k:
                for mach in args.mach:
                    for length in args.post_lengths_m:
                        try:
                            rows.append(evaluate_design_point(
                                ct, temperature, pressure, mach, length,
                                args.pre_length_m, args.throat_height_m, argon,
                                args.ignition_horizon_s,
                            ))
                        except Exception as exc:
                            errors.append({
                                "T0_K": temperature, "p0_Pa": pressure, "M1": mach,
                                "post_length_m": length, "argon_moles": argon,
                                "error": str(exc),
                            })
    selected = select_diverse_candidates(rows, args.candidate_limit)
    add_cj_screens(ct, selected)
    candidates = sorted(
        [row for row in rows if row["shock_triggered_candidate"]], key=candidate_rank
    )
    result = {
        "scope": "Mach-pressure-residence-dilution design for shock-triggered nozzle ignition",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mechanism": MECHANISM,
        "grid": {
            "temperatures_K": args.temperatures_k,
            "pressures_Pa": args.pressures_pa,
            "Mach1": args.mach,
            "post_lengths_m": args.post_lengths_m,
            "argon_moles_per_2H2_1O2": args.argon_moles,
            "pre_length_m": args.pre_length_m,
            "throat_height_m": args.throat_height_m,
        },
        "criteria": {
            "pre_safe": "Da_pre < 0.1 or no ignition within horizon",
            "post_relevant": "Da_post >= 0.3",
            "post_strong": "Da_post >= 1",
            "rarefied_screen": "max(Kn_height_upstream,Kn_height_downstream) >= 0.005",
        },
        "evaluated_case_count": len(rows),
        "failed_case_count": len(errors),
        "candidate_count": sum(row["shock_triggered_candidate"] for row in rows),
        "strong_candidate_count": sum(row["strong_shock_triggered_candidate"] for row in rows),
        "selected_candidates": selected,
        "errors": errors,
        "limitations": [
            "Frozen quasi-1D states omit boundary layers, shock motion, and area variation after the shock.",
            "Homogeneous ignition omits finite-rate shock structure and transport gradients.",
            "The mean-free-path and CJ values are screening estimates.",
        ],
    }
    write_csv(out_dir / ALL_CASES_NAME, rows)
    write_csv(out_dir / CANDIDATES_NAME, candidates)
    (out_dir / DESIGN_JSON_NAME).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (out_dir / DESIGN_SUMMARY_NAME).write_text(make_summary(result))
    print(json.dumps({
        "output_dir": str(out_dir),
        "evaluated_case_count": len(rows),
        "failed_case_count": len(errors),
        "candidate_count": result["candidate_count"],
        "strong_candidate_count": result["strong_candidate_count"],
        "selected_candidate_count": len(selected),
    }, indent=2))
    print("QK_GATE5_MACH_RESIDENCE_DESIGN_PASS")


if __name__ == "__main__":
    main()
