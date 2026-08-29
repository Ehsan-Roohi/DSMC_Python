#!/usr/bin/env python3
"""Physics prescreen for the QK Gate-5 reacting-nozzle DSMC campaign.

The program consumes the JSON and centerline CSV files produced by
``validate_gate5.py``.  It measures the actual states around every detected
shock, estimates available residence times, evaluates homogeneous adiabatic
constant-volume ignition delays with Cantera, and makes a deliberately
conservative recommendation for the next DSMC matrix.

This is a screening tool, not a detonation solver.  In particular, the CJ
speed reported here is the minimum of an equilibrium Hugoniot/Rayleigh scan.
It should be replaced by a validated CJ package for publication-quality
numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


MIXTURE = "H2:2,O2:1,AR:3"
MECHANISM = "h2o2.yaml"
REPORT_NAME = "QK_GATE5_SHOCK_IGNITION_SCREEN_REPORT.json"
CASE_TABLE_NAME = "QK_GATE5_PHYSICS_CASES.csv"
MATRIX_NAME = "QK_GATE5_NEXT_MATRIX.csv"
OUTPUT_JSON_NAME = "QK_GATE5_PHYSICS_PRESCREEN.json"
SUMMARY_NAME = "QK_GATE5_PHYSICS_PRESCREEN.txt"


@dataclass(frozen=True)
class ShockState:
    index: int
    x_m: float
    grid_spacing_m: float
    thickness_m: float
    throat_x_m: float
    upstream_index: int
    downstream_index: int
    upstream: dict[str, float]
    downstream: dict[str, float]
    pre_shock_time_s: Optional[float]
    post_shock_time_s: Optional[float]
    post_shock_length_m: float


def finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fmean(values: Iterable[float]) -> float:
    selected = [float(value) for value in values if math.isfinite(float(value))]
    if not selected:
        return math.nan
    return statistics.fmean(selected)


def read_centerline(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"centerline file not found: {path}")
    required = {
        "x_m", "rho", "Ttr", "T", "ux", "Mach", "pressure", "Kn"
    }
    rows: list[dict[str, float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for raw in reader:
            try:
                row = {key: float(value) for key, value in raw.items() if key and value != ""}
            except ValueError:
                continue
            if all(math.isfinite(row.get(key, math.nan)) for key in required):
                rows.append(row)
    rows.sort(key=lambda item: item["x_m"])
    if len(rows) < 12:
        raise ValueError(f"{path}: only {len(rows)} usable centerline rows")
    return rows


def nearest_index(rows: Sequence[dict[str, float]], x_m: float) -> int:
    return min(range(len(rows)), key=lambda index: abs(rows[index]["x_m"] - x_m))


def median_spacing(rows: Sequence[dict[str, float]]) -> float:
    spacings = [
        rows[index + 1]["x_m"] - rows[index]["x_m"]
        for index in range(len(rows) - 1)
        if rows[index + 1]["x_m"] > rows[index]["x_m"]
    ]
    if not spacings:
        raise ValueError("centerline has no positive axial spacing")
    return statistics.median(spacings)


def average_state(rows: Sequence[dict[str, float]], indices: range) -> dict[str, float]:
    keys = ("rho", "Ttr", "Trot", "T", "ux", "Mach", "pressure", "Kn")
    selected = [rows[index] for index in indices if 0 <= index < len(rows)]
    if not selected:
        raise ValueError("empty state-averaging window")
    state: dict[str, float] = {}
    for key in keys:
        values = [row[key] for row in selected if key in row and math.isfinite(row[key])]
        if values:
            state[key] = statistics.median(values)
    state["x_m"] = statistics.fmean(row["x_m"] for row in selected)
    return state


def integrate_travel_time(
    rows: Sequence[dict[str, float]], start: int, stop: int
) -> Optional[float]:
    """Integrate dx/|u| from start to stop (both indices lie on the path)."""
    if start == stop:
        return 0.0
    direction = 1 if stop > start else -1
    total = 0.0
    for left in range(start, stop, direction):
        right = left + direction
        dx = abs(rows[right]["x_m"] - rows[left]["x_m"])
        speed = 0.5 * (abs(rows[right]["ux"]) + abs(rows[left]["ux"]))
        if dx <= 0.0 or speed <= 1.0e-12 or not math.isfinite(speed):
            return None
        total += dx / speed
    return total


def crossing_position(
    rows: Sequence[dict[str, float]], indices: Sequence[int], normalized: Sequence[float], level: float
) -> Optional[float]:
    for pair in range(len(indices) - 1):
        a, b = normalized[pair], normalized[pair + 1]
        if (a - level) * (b - level) > 0.0 or a == b:
            continue
        xa = rows[indices[pair]]["x_m"]
        xb = rows[indices[pair + 1]]["x_m"]
        fraction = (level - a) / (b - a)
        return xa + fraction * (xb - xa)
    return None


def shock_thickness(
    rows: Sequence[dict[str, float]], shock_index: int, rho_up: float, rho_down: float
) -> float:
    spacing = median_spacing(rows)
    if abs(rho_down - rho_up) <= 1.0e-30:
        return spacing
    lo = max(0, shock_index - 12)
    hi = min(len(rows), shock_index + 13)
    indices = list(range(lo, hi))
    normalized = [
        max(0.0, min(1.0, (rows[index]["rho"] - rho_up) / (rho_down - rho_up)))
        for index in indices
    ]
    x10 = crossing_position(rows, indices, normalized, 0.10)
    x90 = crossing_position(rows, indices, normalized, 0.90)
    if x10 is None or x90 is None:
        return spacing
    return max(abs(x90 - x10), spacing)


def extract_shock_state(
    rows: Sequence[dict[str, float]], shock_x_m: float, window: int = 3, gap: int = 2
) -> ShockState:
    index = nearest_index(rows, shock_x_m)
    upstream_stop = max(1, index - gap)
    upstream_start = max(0, upstream_stop - window)
    downstream_start = min(len(rows) - 1, index + gap + 1)
    downstream_stop = min(len(rows), downstream_start + window)
    if upstream_stop - upstream_start < 1 or downstream_stop - downstream_start < 1:
        raise ValueError("shock is too near a centerline boundary")
    upstream = average_state(rows, range(upstream_start, upstream_stop))
    downstream = average_state(rows, range(downstream_start, downstream_stop))
    length = rows[-1]["x_m"] - rows[0]["x_m"]
    throat_x = rows[0]["x_m"] + 0.25 * length
    throat_index = nearest_index(rows, throat_x)
    upstream_index = max(upstream_start, upstream_stop - 1)
    downstream_index = downstream_start
    pre_time = integrate_travel_time(rows, throat_index, upstream_index)
    post_time = integrate_travel_time(rows, downstream_index, len(rows) - 1)
    thickness = shock_thickness(rows, index, upstream["rho"], downstream["rho"])
    return ShockState(
        index=index,
        x_m=rows[index]["x_m"],
        grid_spacing_m=median_spacing(rows),
        thickness_m=thickness,
        throat_x_m=throat_x,
        upstream_index=upstream_index,
        downstream_index=downstream_index,
        upstream=upstream,
        downstream=downstream,
        pre_shock_time_s=pre_time,
        post_shock_time_s=post_time,
        post_shock_length_m=max(0.0, rows[-1]["x_m"] - downstream["x_m"]),
    )


def import_cantera() -> Any:
    try:
        import cantera as ct  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Cantera is required for a full run. Install cantera>=3.0 or use --self-test."
        ) from exc
    return ct


def positive_rate_peak(
    history: Sequence[dict[str, float]], key: str, sign: float = 1.0
) -> tuple[Optional[float], Optional[float]]:
    peak_rate = -math.inf
    peak_time: Optional[float] = None
    for previous, current in zip(history, history[1:]):
        dt = current["time_s"] - previous["time_s"]
        if dt <= 0.0:
            continue
        rate = sign * (current[key] - previous[key]) / dt
        if math.isfinite(rate) and rate > peak_rate:
            peak_rate = rate
            peak_time = current["time_s"]
    if peak_time is None or not math.isfinite(peak_rate):
        return None, None
    return peak_rate, peak_time


def classify_ignition_history(history: Sequence[dict[str, float]]) -> dict[str, Any]:
    """Classify ignition without assuming that high-T chemistry raises T.

    A thermal event uses the conventional maximum dT/dt after a 50 K rise.
    A high-temperature chemical event instead requires at least 1% H2
    consumption together with radical or product formation.  All individual
    delay definitions are retained so the criterion sensitivity is auditable.
    """
    if len(history) < 2:
        return {"ignited": False, "tau_s": None, "status": "insufficient_history"}
    initial, final = history[0], history[-1]
    max_temperature = max(row["T_K"] for row in history)
    max_oh = max(row["X_OH"] for row in history)
    max_h2o = max(row["X_H2O"] for row in history)
    oh_peak_row = max(history, key=lambda row: row["X_OH"])
    initial_h2 = max(initial["X_H2"], 1.0e-300)
    h2_conversion = max(0.0, (initial["X_H2"] - final["X_H2"]) / initial_h2)
    h2o_gain = max(0.0, max_h2o - initial["X_H2O"])
    max_temperature_rise = max_temperature - initial["T_K"]
    final_temperature_change = final["T_K"] - initial["T_K"]

    dtdt, tau_dtdt = positive_rate_peak(history, "T_K")
    dohdt, tau_dohdt = positive_rate_peak(history, "X_OH")
    dh2odt, tau_dh2odt = positive_rate_peak(history, "X_H2O")
    h2_consumption_rate, tau_h2 = positive_rate_peak(history, "X_H2", sign=-1.0)

    thermal_event = bool(max_temperature_rise >= 50.0 and (dtdt or 0.0) > 0.0)
    chemical_event = bool(
        h2_conversion >= 0.01
        and (max_oh >= 1.0e-5 or h2o_gain >= 1.0e-4)
        and ((h2_consumption_rate or 0.0) > 0.0)
    )
    if thermal_event:
        tau, criterion = tau_dtdt, "maximum_dT_dt_after_50_K_rise"
    elif chemical_event:
        tau, criterion = tau_h2, "maximum_H2_consumption_rate_high_T"
    else:
        tau, criterion = None, "no_resolved_thermal_or_chemical_event"
    ignited = bool(tau is not None and (thermal_event or chemical_event))
    return {
        "ignited": ignited,
        "tau_s": tau,
        "criterion": criterion,
        "thermal_event": thermal_event,
        "chemical_event": chemical_event,
        "tau_dTdt_s": tau_dtdt,
        "tau_dOHdt_s": tau_dohdt,
        "tau_dH2Odt_s": tau_dh2odt,
        "tau_H2_consumption_s": tau_h2,
        "tau_OH_peak_s": oh_peak_row["time_s"],
        "max_dT_dt_K_s": dtdt,
        "max_dX_OH_dt_1_s": dohdt,
        "max_dX_H2O_dt_1_s": dh2odt,
        "max_H2_consumption_rate_1_s": h2_consumption_rate,
        "max_X_OH": max_oh,
        "H2_conversion": h2_conversion,
        "H2O_gain": h2o_gain,
        "max_delta_T_K": max_temperature_rise,
        "final_delta_T_K": final_temperature_change,
        "integrated_to_s": final["time_s"],
        "steps": len(history) - 1,
        "status": "ignited" if ignited else "no_ignition_within_horizon",
    }


def ignition_delay(
    ct: Any,
    temperature_k: float,
    pressure_pa: float,
    max_time_s: float,
    mechanism: str = MECHANISM,
    mixture: str = MIXTURE,
) -> dict[str, Any]:
    """Return homogeneous, adiabatic, constant-volume ignition diagnostics."""
    if temperature_k <= 0.0 or pressure_pa <= 0.0 or max_time_s <= 0.0:
        return {"ignited": False, "tau_s": None, "status": "invalid_state"}
    gas = ct.Solution(mechanism)
    gas.TPX = temperature_k, pressure_pa, mixture
    reactor = ct.IdealGasReactor(gas, energy="on")
    network = ct.ReactorNet([reactor])
    phase = reactor.thermo
    oh_index = phase.species_index("OH")
    h2_index = phase.species_index("H2")
    h2o_index = phase.species_index("H2O")
    history = [{
        "time_s": 0.0,
        "T_K": float(phase.T),
        "X_OH": float(phase.X[oh_index]),
        "X_H2": float(phase.X[h2_index]),
        "X_H2O": float(phase.X[h2o_index]),
    }]
    sample_count = 800
    minimum_time = min(1.0e-12, max_time_s * 1.0e-8)
    log_lo = math.log(max(minimum_time, 1.0e-30))
    log_hi = math.log(max_time_s)
    for sample in range(1, sample_count + 1):
        fraction = sample / sample_count
        target = math.exp(log_lo + fraction * (log_hi - log_lo))
        network.advance(target)
        now = float(network.time)
        history.append({
            "time_s": now,
            "T_K": float(phase.T),
            "X_OH": float(phase.X[oh_index]),
            "X_H2": float(phase.X[h2_index]),
            "X_H2O": float(phase.X[h2o_index]),
        })
        if history[-1]["T_K"] - history[0]["T_K"] >= 1000.0:
            break
    return classify_ignition_history(history)


def bisect_root(function: Any, lo: float, hi: float, iterations: int = 48) -> Optional[float]:
    flo, fhi = function(lo), function(hi)
    if not (math.isfinite(flo) and math.isfinite(fhi)) or flo * fhi > 0.0:
        return None
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        fmid = function(mid)
        if not math.isfinite(fmid):
            return None
        if flo * fmid <= 0.0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


def equilibrium_cj_screen(
    ct: Any,
    temperature_k: float,
    pressure_pa: float,
    mechanism: str = MECHANISM,
    mixture: str = MIXTURE,
) -> dict[str, Any]:
    """Approximate CJ speed from an equilibrium Hugoniot/Rayleigh scan.

    For each candidate density ratio, temperature is solved on the equilibrium
    Hugoniot and the corresponding wave speed is recovered from the Rayleigh
    relation.  The minimum sampled speed is the screening CJ estimate.
    """
    initial = ct.Solution(mechanism)
    initial.TPX = temperature_k, pressure_pa, mixture
    h1 = float(initial.enthalpy_mass)
    rho1 = float(initial.density)
    v1 = 1.0 / rho1
    composition = initial.X
    work = ct.Solution(mechanism)
    candidates: list[tuple[float, float, float, float]] = []
    ratios = [1.05 + index * (6.95 / 55.0) for index in range(56)]
    for ratio in ratios:
        rho2 = ratio * rho1
        v2 = 1.0 / rho2

        def hugoniot(temperature: float) -> float:
            work.TDX = temperature, rho2, composition
            work.equilibrate("TV")
            return float(work.enthalpy_mass - h1 - 0.5 * (work.P - pressure_pa) * (v1 + v2))

        bracket = None
        temperatures = [250.0, 400.0, 700.0, 1100.0, 1700.0, 2600.0, 4000.0, 6000.0, 9000.0]
        last_t, last_f = temperatures[0], hugoniot(temperatures[0])
        for trial_t in temperatures[1:]:
            trial_f = hugoniot(trial_t)
            if math.isfinite(last_f) and math.isfinite(trial_f) and last_f * trial_f <= 0.0:
                bracket = (last_t, trial_t)
                break
            last_t, last_f = trial_t, trial_f
        if bracket is None:
            continue
        root_t = bisect_root(hugoniot, bracket[0], bracket[1])
        if root_t is None:
            continue
        work.TDX = root_t, rho2, composition
        work.equilibrate("TV")
        p2 = float(work.P)
        denominator = v1 - v2
        if p2 <= pressure_pa or denominator <= 0.0:
            continue
        mass_flux_squared = (p2 - pressure_pa) / denominator
        speed = math.sqrt(mass_flux_squared) * v1
        candidates.append((speed, ratio, root_t, p2))
    if not candidates:
        return {"available": False, "speed_m_s": None, "method": "equilibrium_hugoniot_scan"}
    speed, ratio, product_t, product_p = min(candidates)
    return {
        "available": True,
        "speed_m_s": speed,
        "density_ratio": ratio,
        "equilibrium_product_T_K": product_t,
        "equilibrium_product_p_Pa": product_p,
        "sample_count": len(candidates),
        "method": "minimum_equilibrium_hugoniot_rayleigh_speed_screening",
        "publication_warning": "Replace with a validated CJ solver before quantitative use.",
    }


def safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def serialize_state(prefix: str, state: dict[str, float], target: dict[str, Any]) -> None:
    mapping = {
        "x_m": "x_m", "rho": "rho_kg_m3", "Ttr": "Ttr_K", "T": "T_K",
        "ux": "ux_m_s", "Mach": "Mach", "pressure": "pressure_Pa", "Kn": "Kn",
    }
    for source, suffix in mapping.items():
        if source in state:
            target[f"{prefix}_{suffix}"] = state[source]


def analyze_case(
    ct: Any,
    case: dict[str, Any],
    result_dir: Path,
    ignition_horizon_s: float,
    cj_cache: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    name = str(case.get("case", "unknown"))
    condition = case.get("condition") or {}
    shock = case.get("shock") or {}
    through = case.get("through_flow") or {}
    result: dict[str, Any] = {
        "case": name,
        "T0_K": finite(condition.get("T0_K")),
        "pb_over_p0": finite(condition.get("pb_over_p0")),
        "p0_Pa": finite(condition.get("p0_Pa")),
        "numerical_pass": bool(case.get("numerical_pass")),
        "shock_detected": bool(shock.get("detected")),
        "steady": bool(through.get("steady")),
        "mass_flux_imbalance_last20": finite(through.get("mass_flux_imbalance_last20")),
        "eligible": False,
    }
    if not result["numerical_pass"]:
        result["skip_reason"] = "numerical_fail"
        return result
    if not result["shock_detected"] or finite(shock.get("x_m")) is None:
        result["skip_reason"] = "no_detected_internal_shock"
        return result
    centerline_path = result_dir / f"{name}_centerline.csv"
    try:
        rows = read_centerline(centerline_path)
        measured = extract_shock_state(rows, float(shock["x_m"]))
    except (FileNotFoundError, ValueError) as exc:
        result["skip_reason"] = f"centerline_error: {exc}"
        return result
    result["eligible"] = True
    result.update({
        "shock_x_m": measured.x_m,
        "throat_x_m": measured.throat_x_m,
        "grid_spacing_m": measured.grid_spacing_m,
        "shock_thickness_10_90_m": measured.thickness_m,
        "pre_shock_time_s": measured.pre_shock_time_s,
        "post_shock_time_s": measured.post_shock_time_s,
        "post_shock_length_m": measured.post_shock_length_m,
    })
    serialize_state("upstream", measured.upstream, result)
    serialize_state("downstream", measured.downstream, result)
    horizon = max(
        ignition_horizon_s,
        100.0 * (measured.pre_shock_time_s or 0.0),
        100.0 * (measured.post_shock_time_s or 0.0),
    )
    upstream_ignition = ignition_delay(
        ct, measured.upstream["Ttr"], measured.upstream["pressure"], horizon
    )
    downstream_ignition = ignition_delay(
        ct, measured.downstream["Ttr"], measured.downstream["pressure"], horizon
    )
    result["upstream_ignition"] = upstream_ignition
    result["downstream_ignition"] = downstream_ignition
    tau_up = finite(upstream_ignition.get("tau_s"))
    tau_down = finite(downstream_ignition.get("tau_s"))
    result["Da_pre"] = safe_ratio(measured.pre_shock_time_s, tau_up)
    result["Da_post"] = safe_ratio(measured.post_shock_time_s, tau_down)
    result["induction_length_m"] = (
        abs(measured.downstream["ux"]) * tau_down if tau_down is not None else None
    )
    result["induction_to_shock_thickness"] = safe_ratio(
        finite(result["induction_length_m"]), measured.thickness_m
    )
    cache_key = (round(measured.upstream["Ttr"]), round(measured.upstream["pressure"]))
    if cache_key not in cj_cache:
        cj_cache[cache_key] = equilibrium_cj_screen(
            ct, measured.upstream["Ttr"], measured.upstream["pressure"]
        )
    cj = cj_cache[cache_key]
    result["cj_screen"] = cj
    cj_speed = finite(cj.get("speed_m_s"))
    result["standing_CJ_velocity_margin"] = safe_ratio(abs(measured.upstream["ux"]), cj_speed)
    result["stationary_normal_detonation_kinematically_possible_screen"] = bool(
        result["standing_CJ_velocity_margin"] is not None
        and result["standing_CJ_velocity_margin"] >= 1.0
    )
    result["quality_for_fit"] = bool(
        result["steady"]
        and (result["mass_flux_imbalance_last20"] is None or result["mass_flux_imbalance_last20"] <= 0.10)
    )
    return result


def linear_fit(xs: Sequence[float], ys: Sequence[float]) -> Optional[tuple[float, float]]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    xbar, ybar = statistics.fmean(xs), statistics.fmean(ys)
    denominator = sum((x - xbar) ** 2 for x in xs)
    if denominator <= 0.0:
        return None
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denominator
    return slope, ybar - slope * xbar


def round_to(value: float, increment: float) -> float:
    return increment * round(value / increment)


def find_target_temperature(
    ct: Any, pressure_pa: float, residence_s: float, horizon_s: float
) -> tuple[Optional[float], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    for temperature in range(800, 3501, 50):
        ignition = ignition_delay(ct, float(temperature), pressure_pa, horizon_s)
        tau = finite(ignition.get("tau_s"))
        samples.append({
            "T_K": temperature,
            "tau_s": tau,
            "criterion": ignition.get("criterion"),
            "H2_conversion": ignition.get("H2_conversion"),
            "max_X_OH": ignition.get("max_X_OH"),
            "max_delta_T_K": ignition.get("max_delta_T_K"),
        })
        if tau is not None and tau <= residence_s:
            return float(temperature), samples
    return None, samples


def evaluate_recommended_condition(
    ct: Any,
    temperature0_k: float,
    pressure0_pa: float,
    default_pressure0_pa: float,
    t1_fit: tuple[float, float],
    t2_fit: tuple[float, float],
    upstream_pressure_pa: float,
    downstream_pressure_pa: float,
    pre_residence_s: float,
    post_residence_s: float,
    ignition_horizon_s: float,
) -> dict[str, Any]:
    pressure_scale = pressure0_pa / default_pressure0_pa
    predicted_t1 = t1_fit[0] * temperature0_k + t1_fit[1]
    predicted_t2 = t2_fit[0] * temperature0_k + t2_fit[1]
    predicted_p1 = upstream_pressure_pa * pressure_scale
    predicted_p2 = downstream_pressure_pa * pressure_scale
    horizon = max(ignition_horizon_s, 100.0 * pre_residence_s, 100.0 * post_residence_s)
    pre_ignition = ignition_delay(ct, predicted_t1, predicted_p1, horizon)
    post_ignition = ignition_delay(ct, predicted_t2, predicted_p2, horizon)
    tau_pre = finite(pre_ignition.get("tau_s"))
    tau_post = finite(post_ignition.get("tau_s"))
    da_pre = safe_ratio(pre_residence_s, tau_pre)
    da_post = safe_ratio(post_residence_s, tau_post)
    pre_safe = da_pre is None or da_pre < 0.10
    post_relevant = da_post is not None and da_post >= 0.10
    if not pre_safe:
        status = "pre_shock_ignition_risk"
    elif not post_relevant:
        status = "postshock_chemistry_too_slow"
    else:
        status = "shock_triggered_window_candidate"
    return {
        "p0_Pa": pressure0_pa,
        "T0_K": temperature0_k,
        "predicted_upstream_T_K": predicted_t1,
        "predicted_downstream_T_K": predicted_t2,
        "predicted_upstream_pressure_Pa": predicted_p1,
        "predicted_downstream_pressure_Pa": predicted_p2,
        "predicted_tau_pre_s": tau_pre,
        "predicted_tau_post_s": tau_post,
        "predicted_Da_pre": da_pre,
        "predicted_Da_post": da_post,
        "pre_ignition_criterion": pre_ignition.get("criterion"),
        "post_ignition_criterion": post_ignition.get("criterion"),
        "screen_status": status,
        "passes_shock_triggered_window": pre_safe and post_relevant,
    }


def recommend_matrix(
    ct: Any,
    analyzed: Sequence[dict[str, Any]],
    default_p0_pa: float,
    ignition_horizon_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    recommendations: list[dict[str, Any]] = []
    branch_diagnostics: list[dict[str, Any]] = []
    eligible = [row for row in analyzed if row.get("eligible") and row.get("steady")]
    ratios = sorted({float(row["pb_over_p0"]) for row in eligible if row.get("pb_over_p0") is not None})
    preferred = [ratio for ratio in ratios if ratio >= 0.18] or ratios
    for ratio in preferred:
        group = [row for row in eligible if abs(float(row["pb_over_p0"]) - ratio) < 1.0e-9]
        high_quality = [row for row in group if row.get("quality_for_fit")]
        fit_rows = high_quality if len(high_quality) >= 2 else group
        xs = [float(row["T0_K"]) for row in fit_rows if row.get("T0_K") is not None]
        t2_values = [float(row["downstream_Ttr_K"]) for row in fit_rows if row.get("T0_K") is not None]
        t1_values = [float(row["upstream_Ttr_K"]) for row in fit_rows if row.get("T0_K") is not None]
        t2_fit = linear_fit(xs, t2_values)
        t1_fit = linear_fit(xs, t1_values)
        post_residence_values = [
            float(row["post_shock_time_s"])
            for row in fit_rows
            if finite(row.get("post_shock_time_s")) is not None
        ]
        pre_residence_values = [
            float(row["pre_shock_time_s"])
            for row in fit_rows
            if finite(row.get("pre_shock_time_s")) is not None
        ]
        downstream_pressure_values = [
            float(row["downstream_pressure_Pa"])
            for row in fit_rows
            if finite(row.get("downstream_pressure_Pa")) is not None
        ]
        upstream_pressure_values = [
            float(row["upstream_pressure_Pa"])
            for row in fit_rows
            if finite(row.get("upstream_pressure_Pa")) is not None
        ]
        diagnostic: dict[str, Any] = {
            "pb_over_p0": ratio,
            "fit_case_count": len(fit_rows),
            "used_high_quality_subset": len(high_quality) >= 2,
        }
        if (
            t2_fit is None or t1_fit is None
            or not post_residence_values or not pre_residence_values
            or not downstream_pressure_values or not upstream_pressure_values
        ):
            diagnostic["status"] = "insufficient_steady_shock_data"
            branch_diagnostics.append(diagnostic)
            continue
        t2_slope, t2_intercept = t2_fit
        pre_residence = statistics.median(pre_residence_values)
        post_residence = statistics.median(post_residence_values)
        downstream_pressure = statistics.median(downstream_pressure_values)
        upstream_pressure = statistics.median(upstream_pressure_values)
        horizon = max(ignition_horizon_s, 100.0 * pre_residence, 100.0 * post_residence)
        target_t2, ignition_scan = find_target_temperature(
            ct, downstream_pressure, post_residence, horizon
        )
        diagnostic.update({
            "T1_fit_slope_K_per_K": t1_fit[0],
            "T1_fit_intercept_K": t1_fit[1],
            "T2_fit_slope_K_per_K": t2_slope,
            "T2_fit_intercept_K": t2_intercept,
            "representative_pre_shock_residence_s": pre_residence,
            "representative_post_shock_residence_s": post_residence,
            "representative_upstream_pressure_Pa": upstream_pressure,
            "representative_downstream_pressure_Pa": downstream_pressure,
            "target_post_shock_T_K_for_Da_ge_1": target_t2,
            "ignition_scan": ignition_scan,
        })
        if target_t2 is None or t2_slope <= 0.0:
            diagnostic["status"] = "no_Da_one_target_below_3500_K"
            branch_diagnostics.append(diagnostic)
            continue
        required_t0 = (target_t2 - t2_intercept) / t2_slope
        required_t0 = max(1750.0, min(3500.0, required_t0))
        candidates = sorted({
            max(1750.0, min(3500.0, round_to(required_t0 * factor, 25.0)))
            for factor in (0.90, 1.00, 1.10)
        })
        evaluated: list[dict[str, Any]] = []
        for temperature in candidates:
            screen = evaluate_recommended_condition(
                ct, temperature, default_p0_pa, default_p0_pa,
                t1_fit, t2_fit, upstream_pressure, downstream_pressure,
                pre_residence, post_residence, ignition_horizon_s,
            )
            screen["pb_over_p0"] = ratio
            evaluated.append(screen)
            if screen["passes_shock_triggered_window"]:
                recommendations.append({
                "p0_Pa": default_p0_pa,
                "T0_K": temperature,
                "pb_over_p0": ratio,
                "chemistry": "ON",
                "purpose": "bracket_postshock_Da_ind_near_one",
                "basis": "linear_T2_vs_T0_fit_plus_Cantera_constant_volume_ignition",
                "predicted_Da_pre": screen["predicted_Da_pre"],
                "predicted_Da_post": screen["predicted_Da_post"],
                })
        pressure_screen = evaluate_recommended_condition(
            ct, round_to(required_t0, 25.0), 2.0 * default_p0_pa, default_p0_pa,
            t1_fit, t2_fit, upstream_pressure, downstream_pressure,
            pre_residence, post_residence, ignition_horizon_s,
        )
        pressure_screen["pb_over_p0"] = ratio
        evaluated.append(pressure_screen)
        if pressure_screen["passes_shock_triggered_window"]:
            recommendations.append({
                "p0_Pa": 2.0 * default_p0_pa,
                "T0_K": round_to(required_t0, 25.0),
                "pb_over_p0": ratio,
                "chemistry": "ON",
                "purpose": "pressure_sensitivity_at_predicted_transition",
                "basis": "kinetic_sensitivity_control; DSMC similarity must be recomputed",
                "predicted_Da_pre": pressure_screen["predicted_Da_pre"],
                "predicted_Da_post": pressure_screen["predicted_Da_post"],
            })
        diagnostic["estimated_T0_K_for_Da_one"] = required_t0
        diagnostic["evaluated_candidates"] = evaluated
        diagnostic["status"] = (
            "recommended_screen_branch"
            if any(item["passes_shock_triggered_window"] for item in evaluated)
            else "no_pre_safe_shock_triggered_window_in_candidate_set"
        )
        branch_diagnostics.append(diagnostic)
    unique: list[dict[str, Any]] = []
    keys: set[tuple[float, float, float]] = set()
    for row in recommendations:
        key = (float(row["p0_Pa"]), float(row["T0_K"]), float(row["pb_over_p0"]))
        if key not in keys:
            keys.add(key)
            unique.append(row)
    return unique, branch_diagnostics


def flatten_case(row: dict[str, Any]) -> dict[str, Any]:
    flat = {key: value for key, value in row.items() if not isinstance(value, dict)}
    for prefix in ("upstream_ignition", "downstream_ignition"):
        nested = row.get(prefix) or {}
        flat[f"{prefix}_tau_s"] = nested.get("tau_s")
        flat[f"{prefix}_status"] = nested.get("status")
    cj = row.get("cj_screen") or {}
    flat["cj_speed_screening_m_s"] = cj.get("speed_m_s")
    flat["cj_screen_method"] = cj.get("method")
    return flat


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
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_summary(result: dict[str, Any]) -> str:
    lines = [
        "QK Gate-5 physics prescreen",
        f"Generated UTC: {result['generated_utc']}",
        f"Source report: {result['source_report']}",
        f"Shock cases analyzed: {result['shock_cases_analyzed']}",
        f"Recommended next cases: {len(result['recommended_matrix'])}",
        "",
        "Case diagnostics:",
    ]
    for row in result["cases"]:
        if not row.get("eligible"):
            lines.append(f"  {row['case']}: skipped ({row.get('skip_reason', 'ineligible')})")
            continue
        tau = (row.get("downstream_ignition") or {}).get("tau_s")
        lines.append(
            "  {case}: M1={mach:.3g}, T2={temp:.1f} K, tpost={tpost}, "
            "tau_ind={tau}, Da_post={da}, u1/DCJ={margin}".format(
                case=row["case"],
                mach=float(row.get("upstream_Mach", math.nan)),
                temp=float(row.get("downstream_Ttr_K", math.nan)),
                tpost=row.get("post_shock_time_s"),
                tau=tau,
                da=row.get("Da_post"),
                margin=row.get("standing_CJ_velocity_margin"),
            )
        )
    lines.extend([
        "",
        "Interpretation guardrails:",
        "  Da_post >= 1 is a homogeneous-kinetics screening condition, not proof of nozzle ignition.",
        "  u1/DCJ >= 1 is only a kinematic screen for a stationary normal detonation.",
        "  The CJ value is an equilibrium Hugoniot scan and must be independently validated.",
        "  Recommended cases still require DSMC resolution, particle statistics, and conservation audits.",
    ])
    return "\n".join(lines) + "\n"


def load_report(result_dir: Path, explicit: Optional[Path]) -> tuple[Path, dict[str, Any]]:
    candidates = [explicit] if explicit is not None else [
        result_dir / REPORT_NAME,
        result_dir / "QK_GATE5_SHOCK_IGNITION_SCREEN.json",
    ]
    for path in candidates:
        if path is not None and path.is_file():
            return path, json.loads(path.read_text())
    raise FileNotFoundError(
        "Gate-5 report not found; tried: " + ", ".join(str(path) for path in candidates if path)
    )


def self_test() -> None:
    rows: list[dict[str, float]] = []
    for index in range(101):
        x = index / 100.0
        transition = 1.0 / (1.0 + math.exp(-(x - 0.60) / 0.008))
        rows.append({
            "x_m": x,
            "rho": 1.0 + transition,
            "Ttr": 800.0 + 400.0 * transition,
            "T": 800.0 + 400.0 * transition,
            "Trot": 800.0 + 400.0 * transition,
            "ux": 1000.0 - 500.0 * transition,
            "Mach": 2.0 - 1.2 * transition,
            "pressure": 1.0e5 + 2.0e5 * transition,
            "Kn": 0.01,
        })
    state = extract_shock_state(rows, 0.60)
    assert state.upstream["rho"] < state.downstream["rho"]
    assert state.upstream["ux"] > state.downstream["ux"]
    assert state.pre_shock_time_s is not None and state.pre_shock_time_s > 0.0
    assert state.post_shock_time_s is not None and state.post_shock_time_s > 0.0
    assert state.thickness_m >= state.grid_spacing_m
    fit = linear_fit([1000.0, 1500.0, 2000.0], [600.0, 900.0, 1200.0])
    assert fit is not None and abs(fit[0] - 0.6) < 1.0e-12
    assert abs((fit[1])) < 1.0e-9
    assert safe_ratio(2.0, 4.0) == 0.5
    thermal_history = [
        {"time_s": 0.0, "T_K": 1000.0, "X_OH": 0.0, "X_H2": 0.33, "X_H2O": 0.0},
        {"time_s": 1.0e-6, "T_K": 1010.0, "X_OH": 1.0e-6, "X_H2": 0.329, "X_H2O": 1.0e-5},
        {"time_s": 2.0e-6, "T_K": 1200.0, "X_OH": 1.0e-3, "X_H2": 0.25, "X_H2O": 0.05},
    ]
    thermal = classify_ignition_history(thermal_history)
    assert thermal["ignited"] and thermal["thermal_event"]
    assert thermal["criterion"] == "maximum_dT_dt_after_50_K_rise"
    high_temperature_history = [
        {"time_s": 0.0, "T_K": 3500.0, "X_OH": 0.0, "X_H2": 0.33, "X_H2O": 0.0},
        {"time_s": 1.0e-8, "T_K": 3495.0, "X_OH": 2.0e-4, "X_H2": 0.30, "X_H2O": 2.0e-3},
        {"time_s": 2.0e-8, "T_K": 3480.0, "X_OH": 1.0e-3, "X_H2": 0.20, "X_H2O": 0.04},
    ]
    high_temperature = classify_ignition_history(high_temperature_history)
    assert high_temperature["ignited"] and high_temperature["chemical_event"]
    assert not high_temperature["thermal_event"]
    assert high_temperature["criterion"] == "maximum_H2_consumption_rate_high_T"
    inert_history = [
        {"time_s": 0.0, "T_K": 800.0, "X_OH": 0.0, "X_H2": 0.33, "X_H2O": 0.0},
        {"time_s": 1.0e-3, "T_K": 800.001, "X_OH": 1.0e-12, "X_H2": 0.3299999, "X_H2O": 1.0e-10},
    ]
    inert = classify_ignition_history(inert_history)
    assert not inert["ignited"] and inert["tau_s"] is None
    print("QK_GATE5_PHYSICS_PRESCREEN_SELF_TEST_PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, help="completed Gate-5 screen result directory")
    parser.add_argument("--report", type=Path, help="explicit Gate-5 JSON report")
    parser.add_argument("--out-dir", type=Path, help="output directory; defaults to RESULT_DIR/physics_prescreen")
    parser.add_argument("--ignition-horizon-s", type=float, default=0.02)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.result_dir is None:
        parser.error("--result-dir is required unless --self-test is used")
    result_dir = args.result_dir.resolve()
    out_dir = (args.out_dir or (result_dir / "physics_prescreen")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path, report = load_report(result_dir, args.report)
    ct = import_cantera()
    cases = report.get("cases") or []
    if not isinstance(cases, list):
        raise SystemExit("report field 'cases' is not a list")
    cj_cache: dict[tuple[int, int], dict[str, Any]] = {}
    analyzed = [
        analyze_case(ct, case, result_dir, args.ignition_horizon_s, cj_cache)
        for case in cases
    ]
    p0 = finite((report.get("conditions") or {}).get("p0_Pa")) or 500000.0
    matrix, branches = recommend_matrix(ct, analyzed, p0, args.ignition_horizon_s)
    result = {
        "scope": "physics-first prescreen for shock-triggered H2/O2/Ar reaction in a C-D nozzle",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_report": str(report_path),
        "source_result_dir": str(result_dir),
        "mechanism": MECHANISM,
        "mixture": MIXTURE,
        "ignition_model": "adiabatic homogeneous constant-volume Cantera reactor",
        "ignition_horizon_s": args.ignition_horizon_s,
        "shock_cases_analyzed": sum(bool(row.get("eligible")) for row in analyzed),
        "cj_method": "minimum equilibrium Hugoniot/Rayleigh speed screening",
        "cases": analyzed,
        "recommendation_branches": branches,
        "recommended_matrix": matrix,
        "limitations": [
            "Homogeneous ignition delay omits gradients, wall losses, diffusion, and finite-rate shock structure.",
            "Equilibrium CJ scan is a screening estimate and is not a validated CJ calculation.",
            "A recommended condition is not evidence of ignition until the DSMC reaction and conservation audits pass.",
        ],
    }
    (out_dir / OUTPUT_JSON_NAME).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_csv(out_dir / CASE_TABLE_NAME, [flatten_case(row) for row in analyzed])
    write_csv(out_dir / MATRIX_NAME, matrix)
    (out_dir / SUMMARY_NAME).write_text(make_summary(result))
    print(json.dumps({
        "output_dir": str(out_dir),
        "shock_cases_analyzed": result["shock_cases_analyzed"],
        "recommended_case_count": len(matrix),
        "case_table": str(out_dir / CASE_TABLE_NAME),
        "next_matrix": str(out_dir / MATRIX_NAME),
    }, indent=2))
    print("QK_GATE5_PHYSICS_PRESCREEN_PASS")


if __name__ == "__main__":
    main()
