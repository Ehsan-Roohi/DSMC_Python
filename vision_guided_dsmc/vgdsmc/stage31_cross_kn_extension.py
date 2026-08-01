from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np

from .linear_sidewall_validation import LinearSidewallConfig
from .reduced_spherical_solver import _case_metrics, solve_reduced_case
from .velocity_quadrature_audit import cartesian_midpoint, spherical_product


STAGE31_KNUDSEN_SEQUENCE = (0.1, 10.0)
STAGE31_QUADRATURE_NAMES = (
    "cartesian_midpoint_nv19",
    "spherical_matched_r16_mu12_phi24",
)


def validate_stage31_design(
    nx: int,
    ny: int,
    max_steps: int,
    tolerance: float,
) -> None:
    if nx < 3 or ny < 3:
        raise ValueError("nx and ny must be at least three")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")


def _materially_better(spherical: dict[str, object], cartesian: dict[str, object]) -> bool:
    return bool(
        float(spherical["qav_relative_error"])
        <= 0.85 * float(cartesian["qav_relative_error"])
        and float(spherical["wall_velocity_relative_rms"])
        <= 0.85 * float(cartesian["wall_velocity_relative_rms"])
        and float(spherical["wall_velocity_sign_agreement"])
        >= float(cartesian["wall_velocity_sign_agreement"])
    )


def run_stage31(
    output_dir: str | Path,
    *,
    nx: int = 12,
    ny: int = 12,
    max_steps: int = 9000,
    tolerance: float = 3.0e-5,
) -> dict[str, object]:
    """Extend the fixed Stage-30 comparison to Kn0=0.1 and 10.

    Both Cartesian and spherical rules are run at each Knudsen number so that
    quadrature is the only changed numerical ingredient. No failed physical
    parameter is retuned.
    """
    validate_stage31_design(nx, ny, max_steps, tolerance)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quadratures = (
        cartesian_midpoint(19, 5.0),
        spherical_product(
            16, 12, 24, 5.0, "spherical_matched_r16_mu12_phi24"
        ),
    )
    rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}

    for kn0 in STAGE31_KNUDSEN_SEQUENCE:
        cfg = LinearSidewallConfig(
            nx=nx,
            ny=ny,
            nv=19,
            velocity_extent=5.0,
            kn0=kn0,
            cold_hot_ratio=0.1,
            max_steps=max_steps,
            tolerance=tolerance,
            check_interval=100,
            minimum_steps=1200,
        )
        for quadrature in quadratures:
            result = solve_reduced_case(cfg, quadrature)
            metrics = _case_metrics(result, cfg, quadrature)
            metrics["kn0"] = kn0
            rows.append(metrics)
            key = f"kn{str(kn0).replace('.', 'p')}_{quadrature.name}"
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
                arrays[f"{name}_{key}"] = np.asarray(result[name])

    comparisons: list[dict[str, object]] = []
    for kn0 in STAGE31_KNUDSEN_SEQUENCE:
        selected = [row for row in rows if float(row["kn0"]) == kn0]
        by_scheme = {str(row["scheme"]): row for row in selected}
        cartesian = by_scheme["cartesian_midpoint_nv19"]
        spherical = by_scheme["spherical_matched_r16_mu12_phi24"]
        comparisons.append(
            {
                "kn0": kn0,
                "cartesian_converged": bool(cartesian["converged"]),
                "spherical_converged": bool(spherical["converged"]),
                "spherical_qav_error_ratio_to_cartesian": float(
                    spherical["qav_relative_error"]
                )
                / max(float(cartesian["qav_relative_error"]), 1.0e-14),
                "spherical_velocity_error_ratio_to_cartesian": float(
                    spherical["wall_velocity_relative_rms"]
                )
                / max(float(cartesian["wall_velocity_relative_rms"]), 1.0e-14),
                "spherical_sign_agreement_change": float(
                    spherical["wall_velocity_sign_agreement"]
                )
                - float(cartesian["wall_velocity_sign_agreement"]),
                "spherical_materially_better_on_both_literature_metrics": _materially_better(
                    spherical, cartesian
                ),
            }
        )

    all_converged = all(bool(row["converged"]) for row in rows)
    supported_at_both = all(
        bool(item["spherical_materially_better_on_both_literature_metrics"])
        for item in comparisons
    )
    decision = (
        "spherical_quadrature_supported_across_kn0_0p1_1_10"
        if all_converged and supported_at_both
        else "record_partial_or_negative_cross_kn_extension_and_audit_remaining_model_error"
    )
    summary = {
        "stage": 31,
        "description": (
            "Fixed-physics cross-Knudsen extension of Cartesian and matched "
            "spherical-product reduced Shakhov cavity solvers"
        ),
        "configuration": {
            "grid": [nx, ny],
            "kn0_sequence": list(STAGE31_KNUDSEN_SEQUENCE),
            "cold_hot_ratio": 0.1,
            "max_steps": max_steps,
            "tolerance": tolerance,
            "transport": "first-order upwind",
            "collision": "Shakhov",
            "wall_model": "bottom hot, top cold, side walls linear hot-to-cold",
            "quadratures": list(STAGE31_QUADRATURE_NAMES),
            "physical_parameter_retuning": False,
        },
        "rows": rows,
        "comparisons": comparisons,
        "all_cases_converged": all_converged,
        "spherical_materially_better_at_both_new_knudsen_numbers": supported_at_both,
        "decision": decision,
        "interpretation_guard": (
            "Stage 31 changes only velocity quadrature at each fixed Knudsen number. "
            "No Knudsen, collision, wall, viscosity, Prandtl, or normalization "
            "parameter is fitted or retuned, and nonconvergence or degraded metrics "
            "remain explicit."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(output_dir / "fields_and_profiles.npz", **arrays)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 31 fixed-physics cross-Knudsen quadrature extension"
    )
    parser.add_argument("--output-dir", default="outputs/stage31_cross_kn")
    parser.add_argument("--nx", type=int, default=12)
    parser.add_argument("--ny", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=9000)
    parser.add_argument("--tolerance", type=float, default=3.0e-5)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage31(
                args.output_dir,
                nx=args.nx,
                ny=args.ny,
                max_steps=args.max_steps,
                tolerance=args.tolerance,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
