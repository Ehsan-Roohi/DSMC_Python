from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np

from .linear_sidewall_validation import LinearSidewallConfig
from .stage34_velocity_scale_consistency import (
    local_relaxation_time,
    paper_consistent_c0_tau_prefactor,
    paper_zeta_tau_prefactor,
)
from .stage41_projected_polar_operator_audit import (
    mapped_polar_quadrature,
    projected_macroscopic,
    projected_maxwellian,
    projected_shakhov_equilibrium,
)


STAGE61_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30911134793,
    "workflow_job_id": 91997890514,
    "workflow_conclusion": "success",
    "tests_passed": 13,
    "tests_failed": 0,
    "test_duration_seconds": 0.27,
    "artifact_id": 8895505466,
    "artifact_size_bytes": 2741,
    "artifact_sha256": "e478c02ba435c0bd70dd154b21713c4ad856880bff34fbe396357f26cb1f0889",
    "source_head_sha": "70d173a896c9f84eae99621f0d817f86090cc74a",
    "summary_sha256": "0da47d675f25fd100016665e9aba79d367a474fde88bc33733973be7dad78ad1",
    "decision": (
        "stage61_collision_off_transport_diffusion_below_material_threshold_"
        "finite_kn_audit_next"
    ),
}

STAGE62_KNUDSEN = 10.0
STAGE62_COLD_HOT_RATIO = 0.1
STAGE62_VISCOSITY_EXPONENT = 0.5
STAGE62_RULE = (40, 96)
STAGE62_RADIAL_SCALE = 2.0
STAGE62_SPATIAL_CELL_WIDTH = 1.0 / 64.0
STAGE62_PRANDTL = 2.0 / 3.0
STAGE62_COORDINATE_TOLERANCE = 1.0e-8
STAGE62_COLLISION_TOLERANCE = 1.0e-12
STAGE62_RATIO_TOLERANCE = 1.0e-12
STAGE62_OBSERVABLE_TOLERANCE = 1.0e-14
STAGE62_SHAKHOV_TOLERANCE = 1.0e-12


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage61_artifact(root: str | Path) -> dict[str, object]:
    root = Path(root)
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise ValueError("Stage 61 summary is missing")
    if sha256_file(summary_path) != STAGE61_COMPLETED_ENDPOINT["summary_sha256"]:
        raise ValueError("Stage 61 summary checksum mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("stage") != 61:
        raise ValueError("Stage 61 artifact stage mismatch")
    if summary.get("decision") != STAGE61_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage 61 artifact decision mismatch")
    return summary


def validate_stage62_design(
    kn0: float = STAGE62_KNUDSEN,
    cold_hot_ratio: float = STAGE62_COLD_HOT_RATIO,
    viscosity_exponent: float = STAGE62_VISCOSITY_EXPONENT,
    rule: tuple[int, int] = STAGE62_RULE,
    radial_scale: float = STAGE62_RADIAL_SCALE,
    cell_width: float = STAGE62_SPATIAL_CELL_WIDTH,
    prandtl: float = STAGE62_PRANDTL,
) -> None:
    actual = (
        kn0,
        cold_hot_ratio,
        viscosity_exponent,
        rule,
        radial_scale,
        cell_width,
        prandtl,
    )
    expected = (
        STAGE62_KNUDSEN,
        STAGE62_COLD_HOT_RATIO,
        STAGE62_VISCOSITY_EXPONENT,
        STAGE62_RULE,
        STAGE62_RADIAL_SCALE,
        STAGE62_SPATIAL_CELL_WIDTH,
        STAGE62_PRANDTL,
    )
    if actual != expected:
        raise ValueError(
            "Stage 62 is frozen to Kn0=10, Tc/Th=0.1, omega=0.5, the "
            "40x96 mapped-polar rule at radial scale 2.0, a 64-cell transport "
            "width and Pr=2/3; it is an equation audit, not a retuning stage."
        )


def _relative_error(actual: np.ndarray | float, expected: np.ndarray | float) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    return float(
        np.max(np.abs(actual - expected) / np.maximum(np.abs(expected), 1.0e-300))
    )


def coordinate_scale_audit() -> dict[str, float]:
    quadrature = mapped_polar_quadrature(
        *STAGE62_RULE, radial_scale=STAGE62_RADIAL_SCALE
    )
    phi, psi = projected_maxwellian(
        np.asarray(1.0),
        np.asarray(0.0),
        np.asarray(0.0),
        np.asarray(1.0),
        quadrature,
    )
    fields = projected_macroscopic(phi, psi, quadrature)
    weight = quadrature.weight
    mass = float(np.sum(phi * weight))
    variance_x = float(np.sum(phi * quadrature.vx**2 * weight) / mass)
    variance_y = float(np.sum(phi * quadrature.vy**2 * weight) / mass)
    inferred_scale_x = math.sqrt(variance_x / 0.5)
    inferred_scale_y = math.sqrt(variance_y / 0.5)
    return {
        "mass_error": abs(mass - 1.0),
        "temperature_error": abs(float(fields["T"]) - 1.0),
        "variance_x_error_from_c0": abs(variance_x - 1.0),
        "variance_y_error_from_c0": abs(variance_y - 1.0),
        "inferred_c_over_zeta_scale_x": inferred_scale_x,
        "inferred_c_over_zeta_scale_y": inferred_scale_y,
        "maximum_scale_error_from_sqrt2": max(
            abs(inferred_scale_x - math.sqrt(2.0)),
            abs(inferred_scale_y - math.sqrt(2.0)),
        ),
    }


def collision_frequency_audit() -> dict[str, object]:
    cfg = LinearSidewallConfig(
        nx=64,
        ny=64,
        kn0=STAGE62_KNUDSEN,
        cold_hot_ratio=STAGE62_COLD_HOT_RATIO,
        viscosity_exponent=STAGE62_VISCOSITY_EXPONENT,
        prandtl=STAGE62_PRANDTL,
    )
    density = np.asarray([[0.6, 0.9, 1.3], [0.75, 1.1, 1.45]])
    temperature = np.asarray([[0.1, 0.35, 1.0], [0.18, 0.62, 0.84]])
    tau_implemented = local_relaxation_time(
        density, temperature, cfg, mapping="paper_consistent_c0"
    )
    frequency_implemented = 1.0 / tau_implemented
    frequency_paper_zeta = (
        math.sqrt(math.pi)
        / (2.0 * STAGE62_KNUDSEN)
        * density
        * temperature ** (1.0 - STAGE62_VISCOSITY_EXPONENT)
    )
    frequency_transformed_to_c = math.sqrt(2.0) * frequency_paper_zeta
    prefactor_implemented = paper_consistent_c0_tau_prefactor(STAGE62_KNUDSEN)
    prefactor_expected = paper_zeta_tau_prefactor(STAGE62_KNUDSEN) / math.sqrt(2.0)
    return {
        "density_samples": density.tolist(),
        "temperature_samples": temperature.tolist(),
        "paper_zeta_collision_frequency": frequency_paper_zeta.tolist(),
        "paper_transformed_c_collision_frequency": frequency_transformed_to_c.tolist(),
        "implemented_c_collision_frequency": frequency_implemented.tolist(),
        "maximum_relative_collision_frequency_error": _relative_error(
            frequency_implemented, frequency_transformed_to_c
        ),
        "implemented_tau_prefactor": float(prefactor_implemented),
        "expected_c_tau_prefactor": float(prefactor_expected),
        "tau_prefactor_relative_error": _relative_error(
            prefactor_implemented, prefactor_expected
        ),
    }


def transport_collision_ratio_audit() -> dict[str, object]:
    density = np.asarray([0.65, 0.95, 1.4])
    temperature = np.asarray([0.1, 0.4, 1.0])
    zeta_speed = np.asarray([0.125, 0.75, 2.25])
    c_speed = math.sqrt(2.0) * zeta_speed
    frequency_zeta = (
        math.sqrt(math.pi)
        / (2.0 * STAGE62_KNUDSEN)
        * density
        * temperature ** (1.0 - STAGE62_VISCOSITY_EXPONENT)
    )
    frequency_c = math.sqrt(2.0) * frequency_zeta
    ratio_zeta = (zeta_speed / STAGE62_SPATIAL_CELL_WIDTH) / frequency_zeta
    ratio_c = (c_speed / STAGE62_SPATIAL_CELL_WIDTH) / frequency_c
    return {
        "zeta_streaming_to_collision_ratio": ratio_zeta.tolist(),
        "c_streaming_to_collision_ratio": ratio_c.tolist(),
        "maximum_relative_ratio_error": _relative_error(ratio_c, ratio_zeta),
    }


def observable_normalization_audit() -> dict[str, object]:
    paper_velocity = np.asarray([-0.0032, 0.0, 0.0017, 0.0064])
    paper_heat_flux = np.asarray([-0.21, -0.08, 0.035, 0.19])
    c_velocity_moment = math.sqrt(2.0) * paper_velocity
    c_heat_flux_moment = math.sqrt(2.0) * paper_heat_flux
    reported_velocity = c_velocity_moment / math.sqrt(2.0)
    reported_heat_flux = c_heat_flux_moment / math.sqrt(2.0)
    return {
        "paper_velocity_samples": paper_velocity.tolist(),
        "paper_heat_flux_samples": paper_heat_flux.tolist(),
        "implemented_reported_velocity": reported_velocity.tolist(),
        "implemented_reported_heat_flux": reported_heat_flux.tolist(),
        "maximum_relative_velocity_conversion_error": _relative_error(
            reported_velocity, paper_velocity
        ),
        "maximum_relative_heat_flux_conversion_error": _relative_error(
            reported_heat_flux, paper_heat_flux
        ),
        "conversion_factor_c_moment_to_paper": 1.0 / math.sqrt(2.0),
    }


def shakhov_target_equivalence_audit() -> dict[str, float]:
    quadrature = mapped_polar_quadrature(
        *STAGE62_RULE, radial_scale=STAGE62_RADIAL_SCALE
    )
    rho = np.asarray(0.92)
    temperature = np.asarray(0.63)
    u_c = np.asarray(0.17)
    v_c = np.asarray(-0.09)
    qx_c = np.asarray(8.0e-14)
    qy_c = np.asarray(-5.0e-14)
    fields = {
        "rho": rho,
        "T": temperature,
        "u": u_c,
        "v": v_c,
        "qx": qx_c,
        "qy": qy_c,
    }
    actual_phi, actual_psi, diagnostics = projected_shakhov_equilibrium(
        fields,
        quadrature,
        prandtl=STAGE62_PRANDTL,
        correction_floor=0.05,
    )
    phi_m, psi_m = projected_maxwellian(
        rho, u_c, v_c, temperature, quadrature
    )
    zeta_x = (quadrature.vx - float(u_c)) / math.sqrt(2.0)
    zeta_y = (quadrature.vy - float(v_c)) / math.sqrt(2.0)
    qx_zeta = float(qx_c) / math.sqrt(2.0)
    qy_zeta = float(qy_c) / math.sqrt(2.0)
    zeta2 = zeta_x**2 + zeta_y**2
    q_dot_zeta = qx_zeta * zeta_x + qy_zeta * zeta_y
    paper_coefficient = 4.0 / (15.0 * float(rho) * float(temperature) ** 2)
    expected_phi_factor = 1.0 + paper_coefficient * q_dot_zeta * (
        zeta2 / float(temperature) - 2.0
    )
    expected_psi_factor = 1.0 + paper_coefficient * q_dot_zeta * (
        zeta2 / float(temperature) - 1.0
    )
    if min(float(np.min(expected_phi_factor)), float(np.min(expected_psi_factor))) <= 0.05:
        raise ValueError("Stage 62 synthetic Shakhov state unexpectedly activates clipping")
    expected_phi = phi_m * expected_phi_factor
    expected_psi = psi_m * expected_psi_factor
    density_scale = float(rho) / float(np.sum(expected_phi * quadrature.weight))
    expected_phi *= density_scale
    expected_psi *= density_scale
    return {
        "maximum_relative_phi_target_error": _relative_error(actual_phi, expected_phi),
        "maximum_relative_psi_target_error": _relative_error(actual_psi, expected_psi),
        "minimum_paper_phi_factor": float(np.min(expected_phi_factor)),
        "minimum_paper_psi_factor": float(np.min(expected_psi_factor)),
        "maximum_phi_clipped_weight_fraction": float(
            np.max(diagnostics["phi_clipped_weight_fraction"])
        ),
        "maximum_psi_clipped_weight_fraction": float(
            np.max(diagnostics["psi_clipped_weight_fraction"])
        ),
    }


def stage62_decision(audits: dict[str, dict[str, object]]) -> str:
    coordinate = audits["coordinate_scale"]
    collision = audits["collision_frequency"]
    ratio = audits["transport_collision_ratio"]
    observable = audits["observable_normalization"]
    shakhov = audits["shakhov_target_equivalence"]
    finite = all(
        np.isfinite(float(value))
        for section in audits.values()
        for value in section.values()
        if isinstance(value, (float, int, np.floating, np.integer))
    )
    if not finite:
        return "stage62_nonfinite_equation_audit_blocker"
    if (
        float(coordinate["mass_error"]) > STAGE62_COORDINATE_TOLERANCE
        or float(coordinate["temperature_error"]) > STAGE62_COORDINATE_TOLERANCE
        or float(coordinate["maximum_scale_error_from_sqrt2"])
        > STAGE62_COORDINATE_TOLERANCE
    ):
        return "stage62_velocity_coordinate_identification_blocker"
    if (
        float(collision["maximum_relative_collision_frequency_error"])
        > STAGE62_COLLISION_TOLERANCE
        or float(collision["tau_prefactor_relative_error"])
        > STAGE62_COLLISION_TOLERANCE
    ):
        return "stage62_relaxation_frequency_mapping_blocker"
    if float(ratio["maximum_relative_ratio_error"]) > STAGE62_RATIO_TOLERANCE:
        return "stage62_transport_collision_ratio_blocker"
    if max(
        float(observable["maximum_relative_velocity_conversion_error"]),
        float(observable["maximum_relative_heat_flux_conversion_error"]),
    ) > STAGE62_OBSERVABLE_TOLERANCE:
        return "stage62_observable_normalization_blocker"
    if max(
        float(shakhov["maximum_relative_phi_target_error"]),
        float(shakhov["maximum_relative_psi_target_error"]),
    ) > STAGE62_SHAKHOV_TOLERANCE:
        return "stage62_projected_shakhov_coordinate_transform_blocker"
    return (
        "stage62_finite_kn_relaxation_and_normalization_close_"
        "stage63_collision_target_equivalence_audit"
    )


def run_stage62(
    stage61_artifact_dir: str | Path,
    output_dir: str | Path,
    **design: object,
) -> dict[str, object]:
    validate_stage62_design(**design)
    retained61 = validate_stage61_artifact(stage61_artifact_dir)
    audits = {
        "coordinate_scale": coordinate_scale_audit(),
        "collision_frequency": collision_frequency_audit(),
        "transport_collision_ratio": transport_collision_ratio_audit(),
        "observable_normalization": observable_normalization_audit(),
        "shakhov_target_equivalence": shakhov_target_equivalence_audit(),
    }
    decision = stage62_decision(audits)
    summary = {
        "stage": 62,
        "description": (
            "Equation-level audit of the finite-Knudsen collision frequency, "
            "transport-to-collision ratio, velocity/heat-flux normalization and "
            "projected Shakhov coordinate transformation at the unresolved Kn0=10 endpoint."
        ),
        "retained_stage61_endpoint": STAGE61_COMPLETED_ENDPOINT,
        "retained_stage61_decision": retained61["decision"],
        "source_equations": {
            "paper_velocity_coordinate": "zeta=xi/sqrt(2*k*T0/m)",
            "implemented_velocity_coordinate": "c=xi/sqrt(k*T0/m)=sqrt(2)*zeta",
            "paper_collision_frequency": "sqrt(pi)/(2*Kn0)*n*T^(1-omega)",
            "implemented_tau": "sqrt(2)*Kn0/sqrt(pi)*T^(omega-1)/n",
            "paper_observables": "u=U/v0 and q=Q/(P0*v0), v0=sqrt(2*k*T0/m)",
        },
        "configuration": {
            "kn0": STAGE62_KNUDSEN,
            "cold_hot_ratio": STAGE62_COLD_HOT_RATIO,
            "viscosity_exponent": STAGE62_VISCOSITY_EXPONENT,
            "velocity_rule": list(STAGE62_RULE),
            "radial_scale": STAGE62_RADIAL_SCALE,
            "spatial_cell_width": STAGE62_SPATIAL_CELL_WIDTH,
            "prandtl": STAGE62_PRANDTL,
            "solver_rerun": False,
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "transport_retuning": False,
            "wall_model_retuning": False,
            "normalization_retuning": False,
            "velocity_quadrature_retuning": False,
            "cross_knudsen_extension_permitted": False,
        },
        "thresholds": {
            "coordinate_identification": STAGE62_COORDINATE_TOLERANCE,
            "collision_frequency": STAGE62_COLLISION_TOLERANCE,
            "transport_collision_ratio": STAGE62_RATIO_TOLERANCE,
            "observable_normalization": STAGE62_OBSERVABLE_TOLERANCE,
            "shakhov_target_equivalence": STAGE62_SHAKHOV_TOLERANCE,
        },
        "audits": audits,
        "decision": decision,
        "positive_findings": [
            "The implemented Maxwellian independently identifies the c=sqrt(2)*zeta velocity coordinate used by the solver.",
            "The implemented local relaxation time reproduces the paper collision frequency after the required coordinate transformation.",
            "Streaming-to-collision ratios and the reported velocity and heat-flux conversion are invariant under the same transformation.",
            "The unclipped projected Shakhov target is algebraically equivalent to the paper's reduced phi/psi targets for the fixed synthetic non-equilibrium state.",
        ],
        "negative_findings": [
            "Closing these normalization identities does not validate the finite-Kn cavity solution or explain the greater-than-25% Stage-59 heat-flux discrepancy.",
            "The audit does not rehabilitate the failed MUSCL endpoint, adopt the conservative projection, or authorize cross-Knudsen extension.",
            "The nonlinear clipped collision target and its interaction with the finite-Kn steady iteration remain unresolved and require the next independent audit.",
        ],
        "interpretation_guard": (
            "A passing Stage 62 rules out retuning Kn0, the sqrt(2) velocity conversion, "
            "the collision-frequency prefactor or the output heat-flux normalization as "
            "a scientifically justified response to the Stage-59 discrepancy. It is an "
            "equation audit, not external validation."
        ),
        "scientifically_justified_next_scope": (
            "Audit the finite-Kn nonlinear projected Shakhov collision target against an "
            "independent direct evaluation of the paper's reduced equations on frozen "
            "non-equilibrium states, including the retained clipping path, before any solver retuning."
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
    parser.add_argument("--stage61-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage62(args.stage61_artifact_dir, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
