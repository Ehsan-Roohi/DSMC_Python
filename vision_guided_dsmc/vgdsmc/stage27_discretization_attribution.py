from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import numpy as np

from .linear_sidewall_validation import (
    LinearSidewallConfig,
    TABLE3_UY_RATIO_0P1,
    TABLE6_QAV_RATIO_0P1,
    solve_linear_sidewall_case,
)


@dataclass(frozen=True)
class AttributionCase:
    name: str
    nx: int
    ny: int
    nv: int


DEFAULT_CASES = (
    AttributionCase("baseline_20x20_nv17", 20, 20, 17),
    AttributionCase("spatial_24x24_nv17", 24, 24, 17),
    AttributionCase("velocity_20x20_nv19", 20, 20, 19),
)


def _relative_rms(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    denominator = max(float(np.sqrt(np.mean(reference**2))), 1.0e-14)
    return float(np.sqrt(np.mean((candidate - reference) ** 2)) / denominator)


def computational_work_proxy(nx: int, ny: int, nv: int, iterations: int) -> int:
    """Simple operation-count proxy proportional to cells * velocity nodes * steps."""
    if min(nx, ny, nv, iterations) <= 0:
        raise ValueError("all work-proxy inputs must be positive")
    return int(nx * ny * nv**3 * iterations)


def run_stage27(
    output_dir: str | Path,
    cases: tuple[AttributionCase, ...] = DEFAULT_CASES,
    kn0: float = 1.0,
    max_steps: int = 6500,
    tolerance: float = 2.0e-5,
) -> dict[str, object]:
    if len(cases) < 3:
        raise ValueError("Stage 27 requires baseline, spatial, and velocity cases")
    if len({case.name for case in cases}) != len(cases):
        raise ValueError("case names must be unique")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    literature_velocity = TABLE3_UY_RATIO_0P1[float(kn0)]
    literature_qav = TABLE6_QAV_RATIO_0P1[float(kn0)]

    rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    for case in cases:
        cfg = LinearSidewallConfig(
            nx=case.nx,
            ny=case.ny,
            nv=case.nv,
            velocity_extent=5.0,
            kn0=kn0,
            cold_hot_ratio=0.1,
            viscosity_exponent=0.5,
            prandtl=2.0 / 3.0,
            max_steps=max_steps,
            cfl=0.30,
            tolerance=tolerance,
            check_interval=100,
            minimum_steps=1500,
            positivity_floor=1.0e-30,
        )
        result = solve_linear_sidewall_case(cfg)
        predicted_velocity = np.asarray(result["table_velocity"], dtype=np.float64)
        predicted_qav = float(np.mean(result["bottom_heat_flux"]))
        iterations = int(result["iterations"])
        q_error = abs(predicted_qav - literature_qav) / literature_qav
        velocity_error = _relative_rms(predicted_velocity, literature_velocity)
        sign_agreement = float(
            np.mean(np.sign(predicted_velocity) == np.sign(literature_velocity))
        )
        rows.append(
            {
                "case": case.name,
                "grid": [case.nx, case.ny],
                "nv": case.nv,
                "iterations": iterations,
                "converged": bool(result["converged"]),
                "final_change": float(np.asarray(result["residual_history"])[-1]),
                "predicted_qav": predicted_qav,
                "literature_qav": literature_qav,
                "qav_relative_error": q_error,
                "wall_velocity_relative_rms": velocity_error,
                "wall_velocity_sign_agreement": sign_agreement,
                "work_proxy": computational_work_proxy(
                    case.nx, case.ny, case.nv, iterations
                ),
            }
        )
        for name in (
            "T",
            "rho",
            "u",
            "v",
            "qx",
            "qy",
            "left_wall_velocity",
            "table_velocity",
            "bottom_heat_flux",
            "residual_history",
        ):
            arrays[f"{name}_{case.name}"] = np.asarray(result[name])

    row_by_name = {str(row["case"]): row for row in rows}
    baseline = row_by_name["baseline_20x20_nv17"]
    spatial = row_by_name["spatial_24x24_nv17"]
    velocity = row_by_name["velocity_20x20_nv19"]

    def fractional_change(candidate: dict[str, object], metric: str) -> float:
        base_value = float(baseline[metric])
        return (float(candidate[metric]) - base_value) / max(abs(base_value), 1.0e-14)

    attribution = {
        "spatial_refinement": {
            "qav_error_fractional_change": fractional_change(
                spatial, "qav_relative_error"
            ),
            "velocity_error_fractional_change": fractional_change(
                spatial, "wall_velocity_relative_rms"
            ),
            "sign_agreement_change": float(
                spatial["wall_velocity_sign_agreement"]
            )
            - float(baseline["wall_velocity_sign_agreement"]),
            "work_ratio": float(spatial["work_proxy"]) / float(baseline["work_proxy"]),
        },
        "velocity_refinement": {
            "qav_error_fractional_change": fractional_change(
                velocity, "qav_relative_error"
            ),
            "velocity_error_fractional_change": fractional_change(
                velocity, "wall_velocity_relative_rms"
            ),
            "sign_agreement_change": float(
                velocity["wall_velocity_sign_agreement"]
            )
            - float(baseline["wall_velocity_sign_agreement"]),
            "work_ratio": float(velocity["work_proxy"]) / float(baseline["work_proxy"]),
        },
    }
    best_q = min(rows, key=lambda row: float(row["qav_relative_error"]))
    best_velocity = min(rows, key=lambda row: float(row["wall_velocity_relative_rms"]))
    summary: dict[str, object] = {
        "stage": 27,
        "description": (
            "Fixed-physics attribution of quantitative benchmark error to "
            "spatial versus Cartesian velocity-grid refinement"
        ),
        "configuration": {
            "kn0": kn0,
            "cold_hot_ratio": 0.1,
            "max_steps": max_steps,
            "tolerance": tolerance,
            "physics_frozen": True,
            "transport_order": "first-order upwind",
            "cases": [case.__dict__ for case in cases],
        },
        "literature": {
            "table6_qav": literature_qav,
            "table3_velocity": literature_velocity.tolist(),
        },
        "rows": rows,
        "attribution": attribution,
        "best_qav_case": str(best_q["case"]),
        "best_velocity_case": str(best_velocity["case"]),
        "all_converged": bool(all(bool(row["converged"]) for row in rows)),
        "decision_rule": (
            "If neither isolated refinement materially lowers both quantitative "
            "errors or repairs the velocity sign, Stage 28 must change the "
            "transport/quadrature formulation rather than retune Kn0."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(output_dir / "fields_and_profiles.npz", **arrays)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 27 spatial/velocity discretization attribution"
    )
    parser.add_argument("--output-dir", default="outputs/stage27_attribution")
    parser.add_argument("--max-steps", type=int, default=6500)
    parser.add_argument("--tolerance", type=float, default=2.0e-5)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage27(
                args.output_dir,
                max_steps=args.max_steps,
                tolerance=args.tolerance,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
