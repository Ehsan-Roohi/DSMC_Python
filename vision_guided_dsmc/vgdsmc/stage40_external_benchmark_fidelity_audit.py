from __future__ import annotations

from pathlib import Path
import argparse
import json
import math

from .linear_sidewall_validation import (
    TABLE3_UY_RATIO_0P1 as REPOSITORY_TABLE3_UY_RATIO_0P1,
    TABLE3_Y as REPOSITORY_TABLE3_Y,
    TABLE6_QAV_RATIO_0P1 as REPOSITORY_TABLE6_QAV_RATIO_0P1,
)


SOURCE = {
    "title": "Nonequilibrium Gas Flow and Heat Transfer in a Heated Square Microcavity",
    "authors": [
        "Giorgos Tatsios",
        "Manuel H. Vargas",
        "Stefan K. Stefanov",
        "Dimitris Valougeorgis",
    ],
    "journal": "Heat Transfer Engineering",
    "volume": 37,
    "issue": "13-14",
    "pages": "1085-1095",
    "year": 2016,
    "doi": "10.1080/01457632.2015.1111079",
    "source_full_text_accessed_utc_date": "2026-08-01",
}

SOURCE_TABLE3_Y = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
SOURCE_TABLE3_SHAKHOV_RATIO_0P1 = {
    0.1: [1.7e-3, 8.8e-4, -1.5e-4, -1.1e-3, -1.8e-3,
          -2.5e-3, -2.8e-3, -2.8e-3, -2.0e-3, -5.9e-5],
    1.0: [5.4e-3, 5.8e-3, 5.6e-3, 5.0e-3, 4.4e-3,
          3.6e-3, 2.8e-3, 1.9e-3, 1.2e-3, 7.7e-4],
    10.0: [1.3e-3, 1.3e-3, 1.2e-3, 1.1e-3, 9.2e-4,
           7.4e-4, 5.7e-4, 3.9e-4, 2.5e-4, 1.3e-4],
}
SOURCE_TABLE6_SHAKHOV_RATIO_0P1 = {
    0.01: 1.33e-2,
    0.1: 7.20e-2,
    1.0: 1.48e-1,
    10.0: 1.78e-1,
}
SOURCE_TABLE6_DSMC_RATIO_0P1 = {
    0.01: 1.38e-2,
    0.1: 7.16e-2,
    1.0: 1.49e-1,
    10.0: 1.79e-1,
}

PUBLISHED_DETERMINISTIC_METHOD = {
    "distribution_representation": "projected_phi_psi_in_two_dimensional_molecular_velocity",
    "velocity_coordinates": "polar",
    "radial_quadrature": "mapped_Gauss_Legendre",
    "angular_quadrature": "trapezoidal",
    "radial_nodes_M": 80,
    "angular_nodes_N": 400,
    "velocity_vector_count": 32000,
    "physical_cells_I": 400,
    "physical_cells_J_for_square": 400,
    "transport": "second_order_control_volume",
    "reported_convergence_tolerance": 1.0e-10,
    "velocity_scale": "v0=sqrt(2*k_B*T0/m)",
    "heat_flux_scale": "P0*v0",
    "table3_location": "left_lateral_wall_x=-0.5_by_symmetry",
}
PUBLISHED_DSMC_METHOD = {
    "physical_cells_x": 400,
    "physical_cells_y": 400,
    "particles_per_cell": 100,
}

STAGE39_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30718896816,
    "workflow_job_id": 91419210631,
    "workflow_conclusion": "success",
    "tests_passed": 64,
    "tests_failed": 0,
    "artifact_id": 8825367919,
    "artifact_size_bytes": 41954,
    "artifact_sha256": "c2d44bb727009bb6f7d15255606fad2270e0d80bec5f1ce60643a95a01f43656",
    "head_sha": "c315e4822016a2bf0f1beb663031bfb9858290c2",
    "configuration": {
        "kn0": 0.1,
        "cold_hot_ratio": 0.1,
        "grid": [24, 24],
        "velocity_point_count": 4608,
        "quadrature": "spherical_matched_r16_mu12_phi24",
        "distribution_representation": "full_three_dimensional_molecular_velocity",
        "transport": "first_order_upwind",
        "coupling": "strang_explicit_half_collision",
        "tolerance": 2.0e-5,
        "physical_parameter_retuning": False,
    },
    "shakhov": {
        "iterations": 5500,
        "converged": True,
        "final_change": 1.3511402530752559e-5,
        "predicted_qav": 0.0726173364328985,
        "table3_direct_relative_rms": 1.0091453484689894,
        "table3_direct_sign_agreement": 0.9,
        "wall_mass_balance_relative_error": 2.3230252114543756e-16,
    },
    "bgk": {
        "iterations": 5300,
        "converged": True,
        "predicted_qav": 0.05892478163650692,
        "table3_direct_relative_rms": 1.0750354910948237,
        "table3_direct_sign_agreement": 0.8,
    },
    "decision": "mixed_collision_model_sensitivity_stage40_external_benchmark_audit",
}

STAGE40_EXTERNAL_HEAT_FLUX_AGREEMENT_LIMIT = 0.02


def relative_error(candidate: float, reference: float) -> float:
    if not math.isfinite(candidate) or not math.isfinite(reference):
        raise ValueError("candidate and reference must be finite")
    if reference == 0.0:
        raise ValueError("reference must be nonzero")
    return abs(candidate - reference) / abs(reference)


def _exact_sequence_equal(left: object, right: object) -> bool:
    return list(left) == list(right)


def source_transcription_audit() -> dict[str, object]:
    table3_y_exact = _exact_sequence_equal(REPOSITORY_TABLE3_Y, SOURCE_TABLE3_Y)
    table3_rows = {
        str(kn): _exact_sequence_equal(
            REPOSITORY_TABLE3_UY_RATIO_0P1[kn],
            SOURCE_TABLE3_SHAKHOV_RATIO_0P1[kn],
        )
        for kn in (0.1, 1.0, 10.0)
    }
    table6_rows = {
        str(kn): float(REPOSITORY_TABLE6_QAV_RATIO_0P1[kn])
        == SOURCE_TABLE6_SHAKHOV_RATIO_0P1[kn]
        for kn in (0.1, 1.0, 10.0)
    }
    return {
        "table3_y_exact": table3_y_exact,
        "table3_shakhov_rows_exact": table3_rows,
        "table6_shakhov_rows_exact": table6_rows,
        "all_exact": bool(
            table3_y_exact
            and all(table3_rows.values())
            and all(table6_rows.values())
        ),
    }


def source_table6_model_spread() -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for kn0 in (0.01, 0.1, 1.0, 10.0):
        shakhov = SOURCE_TABLE6_SHAKHOV_RATIO_0P1[kn0]
        dsmc = SOURCE_TABLE6_DSMC_RATIO_0P1[kn0]
        rows[str(kn0)] = {
            "shakhov": shakhov,
            "dsmc": dsmc,
            "absolute_difference": abs(shakhov - dsmc),
            "relative_to_dsmc": relative_error(shakhov, dsmc),
            "symmetric_relative_difference": (
                abs(shakhov - dsmc) / (0.5 * (abs(shakhov) + abs(dsmc)))
            ),
        }
    return rows


def method_fidelity_gap() -> dict[str, object]:
    stage39 = STAGE39_COMPLETED_ENDPOINT["configuration"]
    source = PUBLISHED_DETERMINISTIC_METHOD
    stage_grid = stage39["grid"]
    return {
        "physical_cells_per_direction_ratio_source_to_stage39": (
            source["physical_cells_I"] / stage_grid[0]
        ),
        "physical_cell_count_ratio_source_to_stage39": (
            source["physical_cells_I"]
            * source["physical_cells_J_for_square"]
            / (stage_grid[0] * stage_grid[1])
        ),
        "velocity_vector_count_ratio_source_to_stage39": (
            source["velocity_vector_count"] / stage39["velocity_point_count"]
        ),
        "convergence_tolerance_ratio_stage39_to_source": (
            stage39["tolerance"] / source["reported_convergence_tolerance"]
        ),
        "transport_order_matches": False,
        "distribution_representation_matches": False,
        "velocity_quadrature_architecture_matches": False,
        "source_transport": source["transport"],
        "stage39_transport": stage39["transport"],
        "source_distribution_representation": source["distribution_representation"],
        "stage39_distribution_representation": stage39[
            "distribution_representation"
        ],
        "interpretation": (
            "Stage 39 is a controlled reduced solver, not a source-faithful reproduction "
            "of the published 400x400 projected polar-velocity second-order DVM."
        ),
    }


def stage40_decision(
    transcription_ok: bool,
    agrees_with_shakhov: bool,
    agrees_with_dsmc: bool,
    independent_table3_available: bool,
) -> str:
    if not transcription_ok:
        return "stage40_source_transcription_blocker"
    if agrees_with_shakhov and agrees_with_dsmc and not independent_table3_available:
        return "heat_flux_independently_supported_stage41_projected_polar_dvm"
    if not agrees_with_shakhov or not agrees_with_dsmc:
        return "heat_flux_external_mismatch_stage41_normalization_and_flux_audit"
    return "external_benchmarks_resolved_stage41_source_faithful_confirmation"


def run_stage40(output_dir: str | Path) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transcription = source_transcription_audit()
    source_spread = source_table6_model_spread()
    stage39_qav = STAGE39_COMPLETED_ENDPOINT["shakhov"]["predicted_qav"]
    source_shakhov = SOURCE_TABLE6_SHAKHOV_RATIO_0P1[0.1]
    source_dsmc = SOURCE_TABLE6_DSMC_RATIO_0P1[0.1]
    q_error_shakhov = relative_error(stage39_qav, source_shakhov)
    q_error_dsmc = relative_error(stage39_qav, source_dsmc)
    agrees_shakhov = q_error_shakhov <= STAGE40_EXTERNAL_HEAT_FLUX_AGREEMENT_LIMIT
    agrees_dsmc = q_error_dsmc <= STAGE40_EXTERNAL_HEAT_FLUX_AGREEMENT_LIMIT

    table3_reference_status = {
        "source_table": 3,
        "source_values_are_shakhov_only": True,
        "independent_dsmc_column_present": False,
        "independent_table3_reference_available": False,
        "stage39_direct_relative_rms": STAGE39_COMPLETED_ENDPOINT["shakhov"][
            "table3_direct_relative_rms"
        ],
        "stage39_direct_sign_agreement": STAGE39_COMPLETED_ENDPOINT["shakhov"][
            "table3_direct_sign_agreement"
        ],
        "conclusion": (
            "Table 3 cannot provide an independent collision-model validation because "
            "the published tabulation contains the deterministic Shakhov values only."
        ),
    }

    decision = stage40_decision(
        bool(transcription["all_exact"]),
        agrees_shakhov,
        agrees_dsmc,
        bool(table3_reference_status["independent_table3_reference_available"]),
    )
    summary = {
        "stage": 40,
        "description": (
            "External source-table, independent DSMC heat-flux, and published-method "
            "fidelity audit following the mixed Stage 39 collision-model result"
        ),
        "source": SOURCE,
        "source_normalization": {
            "length_scale": "W",
            "velocity_scale": PUBLISHED_DETERMINISTIC_METHOD["velocity_scale"],
            "heat_flux_scale": PUBLISHED_DETERMINISTIC_METHOD["heat_flux_scale"],
            "knudsen_definition": "Kn0=mu0*v0*sqrt(pi)/(2*P0*W)",
            "cold_hot_ratio": 0.1,
        },
        "source_table3": {
            "y": SOURCE_TABLE3_Y,
            "shakhov_ratio_0p1": {
                str(kn): values
                for kn, values in SOURCE_TABLE3_SHAKHOV_RATIO_0P1.items()
            },
        },
        "source_table6": {
            "shakhov_ratio_0p1": {
                str(kn): value
                for kn, value in SOURCE_TABLE6_SHAKHOV_RATIO_0P1.items()
            },
            "dsmc_ratio_0p1": {
                str(kn): value
                for kn, value in SOURCE_TABLE6_DSMC_RATIO_0P1.items()
            },
            "model_spread": source_spread,
        },
        "published_deterministic_method": PUBLISHED_DETERMINISTIC_METHOD,
        "published_dsmc_method": PUBLISHED_DSMC_METHOD,
        "retained_stage39_endpoint": STAGE39_COMPLETED_ENDPOINT,
        "repository_source_transcription": transcription,
        "stage39_external_heat_flux_comparison": {
            "predicted_qav": stage39_qav,
            "source_shakhov_qav": source_shakhov,
            "source_dsmc_qav": source_dsmc,
            "relative_error_to_source_shakhov": q_error_shakhov,
            "relative_error_to_source_dsmc": q_error_dsmc,
            "agreement_limit": STAGE40_EXTERNAL_HEAT_FLUX_AGREEMENT_LIMIT,
            "agrees_with_source_shakhov": agrees_shakhov,
            "agrees_with_independent_dsmc": agrees_dsmc,
            "source_shakhov_dsmc_relative_spread_to_dsmc": source_spread["0.1"][
                "relative_to_dsmc"
            ],
        },
        "table3_reference_status": table3_reference_status,
        "method_fidelity_gap": method_fidelity_gap(),
        "decision": decision,
        "physical_parameter_retuning": False,
        "scientific_conclusion": (
            "The Stage 39 Shakhov heat flux is within 0.8574117123590345% of the "
            "published Shakhov value and within 1.4208609398023778% of the independently "
            "reported DSMC value at Kn0=0.1. Heat flux is therefore externally supported "
            "at the reduced solver's present resolution. The wall-velocity discrepancy is "
            "not independently adjudicated by Table 3, which tabulates Shakhov values only. "
            "The published deterministic calculation used projected phi/psi distributions, "
            "a 32000-vector polar quadrature, a 400x400 physical grid, second-order control "
            "volumes, and a 1e-10 convergence criterion; Stage 39 used a 24x24 full-3D "
            "4608-point first-order reduced solver. The next justified stage is a "
            "source-faithful projected polar-velocity DVM implementation/audit, not "
            "retuning of Knudsen number or collision parameters."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 40 external source and method-fidelity audit"
    )
    parser.add_argument(
        "--output-dir", default="outputs/stage40_external_benchmark_fidelity_audit"
    )
    args = parser.parse_args()
    print(json.dumps(run_stage40(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
