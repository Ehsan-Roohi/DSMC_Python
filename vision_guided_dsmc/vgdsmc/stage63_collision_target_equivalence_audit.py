from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np

from .stage41_projected_polar_operator_audit import (
    mapped_polar_quadrature,
    projected_shakhov_equilibrium,
)


STAGE62_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30919066526,
    "workflow_job_id": 92024594771,
    "workflow_conclusion": "success",
    "tests_passed": 23,
    "tests_failed": 0,
    "test_duration_seconds": 0.35,
    "artifact_id": 8903379367,
    "artifact_size_bytes": 2583,
    "artifact_sha256": "560afeb69a6a1a8c1c055c68b6e44029a74432294ff952625141f60a664541ea",
    "source_head_sha": "8a67dccbcba01c2752e8129c80f8dbab34b4b943",
    "summary_sha256": "c01131443fa1188cbf604f340aa2598b493ed629ac8be4aa66e91318c310a026",
    "decision": (
        "stage62_finite_kn_relaxation_and_normalization_close_"
        "stage63_collision_target_equivalence_audit"
    ),
}

STAGE63_KNUDSEN = 10.0
STAGE63_COLD_HOT_RATIO = 0.1
STAGE63_VISCOSITY_EXPONENT = 0.5
STAGE63_RULE = (40, 96)
STAGE63_RADIAL_SCALE = 2.0
STAGE63_PRANDTL = 2.0 / 3.0
STAGE63_CORRECTION_FLOOR = 0.05
STAGE63_FORMULA_TOLERANCE = 1.0e-12
STAGE63_IMPLEMENTATION_TOLERANCE = 1.0e-12
STAGE63_DIAGNOSTIC_TOLERANCE = 1.0e-12
STAGE63_UNCLIPPED_CONSERVED_TOLERANCE = 1.0e-7
STAGE63_UNCLIPPED_HEAT_FLUX_TOLERANCE = 1.0e-6
STAGE63_MATERIAL_DEFECT_THRESHOLD = 1.0e-2
STAGE63_MINIMUM_CLIPPED_WEIGHT_FRACTION = 1.0e-4

FROZEN_PAPER_STATES = (
    {
        "name": "equilibrium_control",
        "rho": 1.0,
        "T": 0.8,
        "u_zeta": 0.08,
        "v_zeta": -0.03,
        "qx_zeta": 0.0,
        "qy_zeta": 0.0,
    },
    {
        "name": "cold_axial_moderate",
        "rho": 1.0,
        "T": 0.1,
        "u_zeta": 0.03,
        "v_zeta": 0.02,
        "qx_zeta": 0.03,
        "qy_zeta": 0.0,
    },
    {
        "name": "cold_diagonal_strong",
        "rho": 0.85,
        "T": 0.1,
        "u_zeta": -0.05,
        "v_zeta": 0.04,
        "qx_zeta": 0.03,
        "qy_zeta": -0.02,
    },
    {
        "name": "intermediate_oblique",
        "rho": 1.2,
        "T": 0.3,
        "u_zeta": 0.08,
        "v_zeta": 0.02,
        "qx_zeta": 0.3,
        "qy_zeta": 0.1,
    },
    {
        "name": "warm_transverse",
        "rho": 0.9,
        "T": 0.7,
        "u_zeta": -0.06,
        "v_zeta": 0.07,
        "qx_zeta": 0.3,
        "qy_zeta": -0.25,
    },
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage62_artifact(root: str | Path) -> dict[str, object]:
    summary_path = Path(root) / "summary.json"
    if not summary_path.is_file():
        raise ValueError("Stage 62 summary is missing")
    if sha256_file(summary_path) != STAGE62_COMPLETED_ENDPOINT["summary_sha256"]:
        raise ValueError("Stage 62 summary checksum mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("stage") != 62:
        raise ValueError("Stage 62 artifact stage mismatch")
    if summary.get("decision") != STAGE62_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage 62 artifact decision mismatch")
    return summary


def validate_stage63_design(
    kn0: float = STAGE63_KNUDSEN,
    cold_hot_ratio: float = STAGE63_COLD_HOT_RATIO,
    viscosity_exponent: float = STAGE63_VISCOSITY_EXPONENT,
    rule: tuple[int, int] = STAGE63_RULE,
    radial_scale: float = STAGE63_RADIAL_SCALE,
    prandtl: float = STAGE63_PRANDTL,
    correction_floor: float = STAGE63_CORRECTION_FLOOR,
) -> None:
    actual = (
        kn0,
        cold_hot_ratio,
        viscosity_exponent,
        rule,
        radial_scale,
        prandtl,
        correction_floor,
    )
    expected = (
        STAGE63_KNUDSEN,
        STAGE63_COLD_HOT_RATIO,
        STAGE63_VISCOSITY_EXPONENT,
        STAGE63_RULE,
        STAGE63_RADIAL_SCALE,
        STAGE63_PRANDTL,
        STAGE63_CORRECTION_FLOOR,
    )
    if actual != expected:
        raise ValueError(
            "Stage 63 is frozen to Kn0=10, Tc/Th=0.1, omega=0.5, the 40x96 "
            "mapped-polar rule at radial scale 2.0, Pr=2/3 and correction floor "
            "0.05; it is a collision-target audit, not a retuning stage."
        )


def _state_in_c_coordinates(state: dict[str, float | str]) -> dict[str, float]:
    scale = math.sqrt(2.0)
    return {
        "rho": float(state["rho"]),
        "T": float(state["T"]),
        "u": scale * float(state["u_zeta"]),
        "v": scale * float(state["v_zeta"]),
        "qx": scale * float(state["qx_zeta"]),
        "qy": scale * float(state["qy_zeta"]),
    }


def _weighted_relative_l1(
    actual: np.ndarray,
    expected: np.ndarray,
    weight: np.ndarray,
) -> float:
    numerator = float(np.sum(np.abs(actual - expected) * weight))
    denominator = float(np.sum(np.abs(expected) * weight))
    return numerator / max(denominator, 1.0e-300)


def _relative_to_peak(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(actual - expected))) / max(
        float(np.max(np.abs(expected))), 1.0e-300
    )


def independent_paper_target(
    state: dict[str, float | str],
    quadrature,
    *,
    apply_retained_clipping: bool,
) -> dict[str, object]:
    fields = _state_in_c_coordinates(state)
    rho = fields["rho"]
    temperature = fields["T"]
    cx = quadrature.vx - fields["u"]
    cy = quadrature.vy - fields["v"]
    c_parallel2 = cx * cx + cy * cy
    raw_phi_m = np.exp(-c_parallel2 / (2.0 * temperature)) / (
        2.0 * math.pi * temperature
    )
    discrete_mass = float(np.sum(raw_phi_m * quadrature.weight))
    phi_m = rho * raw_phi_m / max(discrete_mass, 1.0e-300)
    psi_m = temperature * phi_m

    zeta_x = cx / math.sqrt(2.0)
    zeta_y = cy / math.sqrt(2.0)
    zeta2 = zeta_x * zeta_x + zeta_y * zeta_y
    q_dot_zeta = (
        float(state["qx_zeta"]) * zeta_x
        + float(state["qy_zeta"]) * zeta_y
    )
    paper_coefficient = (
        (1.0 - STAGE63_PRANDTL)
        * 4.0
        / (5.0 * rho * temperature**2)
    )
    paper_phi_factor = 1.0 + paper_coefficient * q_dot_zeta * (
        zeta2 / temperature - 2.0
    )
    paper_psi_factor = 1.0 + paper_coefficient * q_dot_zeta * (
        zeta2 / temperature - 1.0
    )

    c_dot_q = cx * fields["qx"] + cy * fields["qy"]
    c_coefficient = (1.0 - STAGE63_PRANDTL) / (
        5.0 * rho * temperature**2
    )
    c_phi_factor = 1.0 + c_coefficient * c_dot_q * (
        c_parallel2 / temperature - 4.0
    )
    c_psi_factor = 1.0 + c_coefficient * c_dot_q * (
        c_parallel2 / temperature - 2.0
    )

    phi_factor = paper_phi_factor
    psi_factor = paper_psi_factor
    if apply_retained_clipping:
        phi_factor = np.maximum(phi_factor, STAGE63_CORRECTION_FLOOR)
        psi_factor = np.maximum(psi_factor, STAGE63_CORRECTION_FLOOR)
    phi = phi_m * phi_factor
    psi = psi_m * psi_factor
    if apply_retained_clipping:
        density_scale = rho / max(float(np.sum(phi * quadrature.weight)), 1.0e-300)
        phi *= density_scale
        psi *= density_scale
    phi_mass = float(np.sum(phi_m * quadrature.weight))
    psi_mass = float(np.sum(psi_m * quadrature.weight))
    return {
        "phi": phi,
        "psi": psi,
        "phi_m": phi_m,
        "psi_m": psi_m,
        "paper_phi_factor": paper_phi_factor,
        "paper_psi_factor": paper_psi_factor,
        "c_phi_factor": c_phi_factor,
        "c_psi_factor": c_psi_factor,
        "phi_clipped_weight_fraction": float(
            np.sum(
                phi_m
                * quadrature.weight
                * (paper_phi_factor < STAGE63_CORRECTION_FLOOR)
            )
            / max(phi_mass, 1.0e-300)
        ),
        "psi_clipped_weight_fraction": float(
            np.sum(
                psi_m
                * quadrature.weight
                * (paper_psi_factor < STAGE63_CORRECTION_FLOOR)
            )
            / max(psi_mass, 1.0e-300)
        ),
        "minimum_paper_phi_factor": float(np.min(paper_phi_factor)),
        "minimum_paper_psi_factor": float(np.min(paper_psi_factor)),
    }


def _moment_defects(
    phi: np.ndarray,
    psi: np.ndarray,
    state: dict[str, float | str],
    quadrature,
) -> dict[str, float]:
    fields = _state_in_c_coordinates(state)
    rho = fields["rho"]
    temperature = fields["T"]
    weight = quadrature.weight
    mass = float(np.sum(phi * weight))
    mx = float(np.sum(phi * quadrature.vx * weight))
    my = float(np.sum(phi * quadrature.vy * weight))
    energy = float(
        np.sum(
            ((quadrature.vx**2 + quadrature.vy**2) * phi + psi) * weight
        )
    )
    target_mx = rho * fields["u"]
    target_my = rho * fields["v"]
    target_energy = rho * (
        fields["u"] ** 2 + fields["v"] ** 2 + 3.0 * temperature
    )
    conserved = {
        "mass": abs(mass - rho) / max(rho, 1.0e-300),
        "x_momentum": abs(mx - target_mx) / max(rho * math.sqrt(temperature), 1.0e-300),
        "y_momentum": abs(my - target_my) / max(rho * math.sqrt(temperature), 1.0e-300),
        "total_energy": abs(energy - target_energy) / max(
            abs(target_energy), rho * temperature, 1.0e-300
        ),
    }
    safe_mass = max(mass, 1.0e-300)
    u = mx / safe_mass
    v = my / safe_mass
    cx = quadrature.vx - u
    cy = quadrature.vy - v
    c_parallel2 = cx * cx + cy * cy
    qx = 0.5 * float(np.sum(cx * (c_parallel2 * phi + psi) * weight))
    qy = 0.5 * float(np.sum(cy * (c_parallel2 * phi + psi) * weight))
    expected_q = (1.0 - STAGE63_PRANDTL) * np.asarray(
        [fields["qx"], fields["qy"]]
    )
    actual_q = np.asarray([qx, qy])
    q_scale = float(np.linalg.norm(expected_q))
    if q_scale > 1.0e-12:
        heat_flux_error = float(np.linalg.norm(actual_q - expected_q)) / q_scale
    else:
        heat_flux_error = float(np.linalg.norm(actual_q - expected_q)) / max(
            rho * temperature**1.5, 1.0e-300
        )
    return {
        "mass_defect": conserved["mass"],
        "x_momentum_defect": conserved["x_momentum"],
        "y_momentum_defect": conserved["y_momentum"],
        "total_energy_defect": conserved["total_energy"],
        "maximum_conserved_moment_defect": max(conserved.values()),
        "heat_flux_closure_error": heat_flux_error,
    }


def audit_frozen_state(
    state: dict[str, float | str],
    quadrature,
) -> dict[str, object]:
    fields = _state_in_c_coordinates(state)
    actual_phi, actual_psi, diagnostics = projected_shakhov_equilibrium(
        fields,
        quadrature,
        prandtl=STAGE63_PRANDTL,
        correction_floor=STAGE63_CORRECTION_FLOOR,
    )
    direct_clipped = independent_paper_target(
        state, quadrature, apply_retained_clipping=True
    )
    direct_unclipped = independent_paper_target(
        state, quadrature, apply_retained_clipping=False
    )
    formula_phi_error = _relative_to_peak(
        direct_unclipped["paper_phi_factor"], direct_unclipped["c_phi_factor"]
    )
    formula_psi_error = _relative_to_peak(
        direct_unclipped["paper_psi_factor"], direct_unclipped["c_psi_factor"]
    )
    implementation_phi_l1 = _weighted_relative_l1(
        actual_phi, direct_clipped["phi"], quadrature.weight
    )
    implementation_psi_l1 = _weighted_relative_l1(
        actual_psi, direct_clipped["psi"], quadrature.weight
    )
    implementation_phi_peak = _relative_to_peak(actual_phi, direct_clipped["phi"])
    implementation_psi_peak = _relative_to_peak(actual_psi, direct_clipped["psi"])
    diagnostic_error = max(
        abs(
            float(np.asarray(diagnostics["phi_clipped_weight_fraction"]))
            - float(direct_clipped["phi_clipped_weight_fraction"])
        ),
        abs(
            float(np.asarray(diagnostics["psi_clipped_weight_fraction"]))
            - float(direct_clipped["psi_clipped_weight_fraction"])
        ),
        abs(
            float(np.asarray(diagnostics["minimum_raw_phi_factor"]))
            - float(direct_clipped["minimum_paper_phi_factor"])
        )
        / max(abs(float(direct_clipped["minimum_paper_phi_factor"])), 1.0),
        abs(
            float(np.asarray(diagnostics["minimum_raw_psi_factor"]))
            - float(direct_clipped["minimum_paper_psi_factor"])
        )
        / max(abs(float(direct_clipped["minimum_paper_psi_factor"])), 1.0),
    )
    return {
        "name": str(state["name"]),
        "paper_state": dict(state),
        "c_coordinate_state": fields,
        "formula_transform": {
            "phi_relative_to_peak_error": formula_phi_error,
            "psi_relative_to_peak_error": formula_psi_error,
        },
        "retained_implementation_match": {
            "phi_weighted_relative_l1": implementation_phi_l1,
            "psi_weighted_relative_l1": implementation_psi_l1,
            "phi_relative_to_peak": implementation_phi_peak,
            "psi_relative_to_peak": implementation_psi_peak,
            "diagnostic_error": diagnostic_error,
        },
        "clipping": {
            "phi_weighted_fraction": direct_clipped["phi_clipped_weight_fraction"],
            "psi_weighted_fraction": direct_clipped["psi_clipped_weight_fraction"],
            "minimum_phi_factor": direct_clipped["minimum_paper_phi_factor"],
            "minimum_psi_factor": direct_clipped["minimum_paper_psi_factor"],
        },
        "unclipped_moment_closure": _moment_defects(
            direct_unclipped["phi"], direct_unclipped["psi"], state, quadrature
        ),
        "retained_clipped_moment_closure": _moment_defects(
            direct_clipped["phi"], direct_clipped["psi"], state, quadrature
        ),
    }


def aggregate_rows(rows: list[dict[str, object]]) -> dict[str, float | int]:
    return {
        "state_count": len(rows),
        "states_with_material_clipping": sum(
            max(
                float(row["clipping"]["phi_weighted_fraction"]),
                float(row["clipping"]["psi_weighted_fraction"]),
            )
            >= STAGE63_MINIMUM_CLIPPED_WEIGHT_FRACTION
            for row in rows
        ),
        "maximum_formula_transform_error": max(
            max(row["formula_transform"].values()) for row in rows
        ),
        "maximum_implementation_match_error": max(
            max(row["retained_implementation_match"].values()) for row in rows
        ),
        "maximum_phi_clipped_weight_fraction": max(
            float(row["clipping"]["phi_weighted_fraction"]) for row in rows
        ),
        "maximum_psi_clipped_weight_fraction": max(
            float(row["clipping"]["psi_weighted_fraction"]) for row in rows
        ),
        "maximum_unclipped_conserved_moment_defect": max(
            float(row["unclipped_moment_closure"]["maximum_conserved_moment_defect"])
            for row in rows
        ),
        "maximum_unclipped_heat_flux_closure_error": max(
            float(row["unclipped_moment_closure"]["heat_flux_closure_error"])
            for row in rows
        ),
        "maximum_clipped_conserved_moment_defect": max(
            float(row["retained_clipped_moment_closure"]["maximum_conserved_moment_defect"])
            for row in rows
        ),
        "maximum_clipped_heat_flux_closure_error": max(
            float(row["retained_clipped_moment_closure"]["heat_flux_closure_error"])
            for row in rows
        ),
    }


def stage63_decision(aggregate: dict[str, float | int]) -> str:
    if not all(np.isfinite(float(value)) for value in aggregate.values()):
        return "stage63_nonfinite_collision_target_audit_blocker"
    if float(aggregate["maximum_formula_transform_error"]) > STAGE63_FORMULA_TOLERANCE:
        return "stage63_independent_paper_formula_transform_blocker"
    if (
        float(aggregate["maximum_implementation_match_error"])
        > STAGE63_IMPLEMENTATION_TOLERANCE
    ):
        return "stage63_retained_collision_target_implementation_blocker"
    if int(aggregate["states_with_material_clipping"]) < 3:
        return "stage63_frozen_state_clipping_coverage_blocker"
    if (
        float(aggregate["maximum_unclipped_conserved_moment_defect"])
        > STAGE63_UNCLIPPED_CONSERVED_TOLERANCE
        or float(aggregate["maximum_unclipped_heat_flux_closure_error"])
        > STAGE63_UNCLIPPED_HEAT_FLUX_TOLERANCE
    ):
        return "stage63_unclipped_paper_target_quadrature_closure_blocker"
    material = max(
        float(aggregate["maximum_clipped_conserved_moment_defect"]),
        float(aggregate["maximum_clipped_heat_flux_closure_error"]),
    ) >= STAGE63_MATERIAL_DEFECT_THRESHOLD
    if material:
        return (
            "stage63_collision_target_matches_independent_paper_equations_"
            "clipping_defects_material_stage64_source_step_consistency_audit"
        )
    return (
        "stage63_collision_target_matches_independent_paper_equations_"
        "clipping_defects_submaterial_stage64_source_step_consistency_audit"
    )


def run_stage63(
    stage62_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage63_design(**design)
    retained62 = validate_stage62_artifact(stage62_artifact_dir)
    quadrature = mapped_polar_quadrature(
        *STAGE63_RULE, radial_scale=STAGE63_RADIAL_SCALE
    )
    rows = [audit_frozen_state(state, quadrature) for state in FROZEN_PAPER_STATES]
    aggregate = aggregate_rows(rows)
    decision = stage63_decision(aggregate)
    summary = {
        "stage": 63,
        "description": (
            "Independent direct evaluation of the paper's reduced projected-Shakhov "
            "collision target on a frozen nonlinear state suite, including the exact "
            "retained 0.05 clipping and density-renormalization path."
        ),
        "retained_stage62_endpoint": STAGE62_COMPLETED_ENDPOINT,
        "retained_stage62_decision": retained62["decision"],
        "configuration": {
            "kn0_investigation_scope": STAGE63_KNUDSEN,
            "cold_hot_ratio": STAGE63_COLD_HOT_RATIO,
            "viscosity_exponent": STAGE63_VISCOSITY_EXPONENT,
            "velocity_rule": list(STAGE63_RULE),
            "radial_scale": STAGE63_RADIAL_SCALE,
            "prandtl": STAGE63_PRANDTL,
            "retained_correction_floor": STAGE63_CORRECTION_FLOOR,
            "frozen_state_count": len(FROZEN_PAPER_STATES),
            "solver_rerun": False,
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "correction_floor_retuning": False,
            "transport_retuning": False,
            "wall_model_retuning": False,
            "normalization_retuning": False,
            "velocity_quadrature_retuning": False,
            "cross_knudsen_extension_permitted": False,
            "conservative_projection_adopted": False,
        },
        "thresholds": {
            "formula_transform": STAGE63_FORMULA_TOLERANCE,
            "retained_implementation_match": STAGE63_IMPLEMENTATION_TOLERANCE,
            "diagnostic_match": STAGE63_DIAGNOSTIC_TOLERANCE,
            "unclipped_conserved_moment": STAGE63_UNCLIPPED_CONSERVED_TOLERANCE,
            "unclipped_heat_flux_closure": STAGE63_UNCLIPPED_HEAT_FLUX_TOLERANCE,
            "material_clipping_defect": STAGE63_MATERIAL_DEFECT_THRESHOLD,
            "minimum_clipped_weight_fraction": STAGE63_MINIMUM_CLIPPED_WEIGHT_FRACTION,
        },
        "rows": rows,
        "aggregate": aggregate,
        "decision": decision,
        "positive_findings": [
            "The c-coordinate implementation is compared with a separately coded zeta-coordinate evaluation of the paper's reduced phi/psi equations across five frozen states.",
            "The suite contains an equilibrium control and axial, diagonal, oblique and transverse non-equilibrium states with preregistered clipping coverage.",
            "The retained max-factor and density-renormalization path is reconstructed independently rather than inferred from benchmark agreement.",
        ],
        "negative_findings": [
            "Agreement with the independently reconstructed clipped target establishes implementation fidelity, not physical correctness of the clipping policy or external cavity validation.",
            "Material clipping-induced invariant or heat-flux defects are retained explicitly and are not removed by changing the floor, Knudsen number, quadrature, relaxation, walls or normalization.",
            "The previously tested conservative projection is not adopted because its frozen 64x64 benchmark endpoint did not improve the unresolved cross-Knudsen discrepancy.",
            "The failed second-order MUSCL endpoint remains negative and cross-Knudsen extension remains prohibited.",
        ],
        "interpretation_guard": (
            "Stage 63 is a frozen collision-target identity and defect audit. It does "
            "not rerun the cavity solver, select a favorable target, validate Table 3 "
            "or Table 6, or authorize retuning."
        ),
        "scientifically_justified_next_scope": (
            "Audit a homogeneous collision-only source update against independently "
            "integrated moment-relaxation identities on the same frozen states, using "
            "the retained clipped target as the active arm and preserving the unclipped "
            "and conservative alternatives as labeled diagnostics only."
        ),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage62-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage63(args.stage62_artifact_dir, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
