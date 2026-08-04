from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np

from .stage41_projected_polar_operator_audit import (
    mapped_polar_quadrature,
    projected_macroscopic,
    projected_maxwellian,
    projected_shakhov_equilibrium,
)
from .stage56_conservative_projection_pilot import (
    _moment_basis,
    _retained_clipped_lower_bounds,
    _target_moments,
    bounded_conservative_projection,
)
from .stage63_collision_target_equivalence_audit import (
    FROZEN_PAPER_STATES,
    STAGE63_COLD_HOT_RATIO,
    STAGE63_CORRECTION_FLOOR,
    STAGE63_KNUDSEN,
    STAGE63_PRANDTL,
    STAGE63_RADIAL_SCALE,
    STAGE63_RULE,
    STAGE63_VISCOSITY_EXPONENT,
    _state_in_c_coordinates,
    independent_paper_target,
)


STAGE63_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30938577712,
    "workflow_job_id": 92090852960,
    "workflow_conclusion": "success",
    "tests_passed": 34,
    "tests_failed": 0,
    "test_duration_seconds": 0.43,
    "artifact_id": 8908675572,
    "artifact_size_bytes": 3846,
    "artifact_sha256": "461ece7b7b9d70bbf6638f0dc042d7bde164fdce28859dc4556561e909987ac6",
    "source_head_sha": "4ec7ab5ae6cd01866447834994d73d39854eef5c",
    "summary_sha256": "b49cfd6af534b777a48b8316fe841f7b6beddbf199a7a25b10c1df7afe70e5be",
    "decision": (
        "stage63_collision_target_matches_independent_paper_equations_"
        "clipping_defects_material_stage64_source_step_consistency_audit"
    ),
}

STAGE64_SOURCE_FRACTIONS = (0.05, 0.25, 0.5, 1.0)
STAGE64_IDENTITY_TOLERANCE = 1.0e-12
STAGE64_INITIAL_MOMENT_TOLERANCE = 1.0e-6
STAGE64_UNCLIPPED_CONSERVED_TOLERANCE = 1.0e-6
STAGE64_UNCLIPPED_HEAT_FLUX_TOLERANCE = 1.0e-5
STAGE64_CONSERVATIVE_CONSERVED_TOLERANCE = 1.0e-9
STAGE64_CONSERVATIVE_HEAT_FLUX_TOLERANCE = 1.0e-8
STAGE64_MATERIAL_SOURCE_DEFECT = 1.0e-2


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage63_artifact(root: str | Path) -> dict[str, object]:
    summary_path = Path(root) / "summary.json"
    if not summary_path.is_file():
        raise ValueError("Stage 63 summary is missing")
    if sha256_file(summary_path) != STAGE63_COMPLETED_ENDPOINT["summary_sha256"]:
        raise ValueError("Stage 63 summary checksum mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("stage") != 63:
        raise ValueError("Stage 63 artifact stage mismatch")
    if summary.get("decision") != STAGE63_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage 63 artifact decision mismatch")
    return summary


def validate_stage64_design(
    kn0: float = STAGE63_KNUDSEN,
    cold_hot_ratio: float = STAGE63_COLD_HOT_RATIO,
    viscosity_exponent: float = STAGE63_VISCOSITY_EXPONENT,
    rule: tuple[int, int] = STAGE63_RULE,
    radial_scale: float = STAGE63_RADIAL_SCALE,
    prandtl: float = STAGE63_PRANDTL,
    correction_floor: float = STAGE63_CORRECTION_FLOOR,
    source_fractions: tuple[float, ...] = STAGE64_SOURCE_FRACTIONS,
) -> None:
    actual = (
        kn0,
        cold_hot_ratio,
        viscosity_exponent,
        rule,
        radial_scale,
        prandtl,
        correction_floor,
        source_fractions,
    )
    expected = (
        STAGE63_KNUDSEN,
        STAGE63_COLD_HOT_RATIO,
        STAGE63_VISCOSITY_EXPONENT,
        STAGE63_RULE,
        STAGE63_RADIAL_SCALE,
        STAGE63_PRANDTL,
        STAGE63_CORRECTION_FLOOR,
        STAGE64_SOURCE_FRACTIONS,
    )
    if actual != expected:
        raise ValueError(
            "Stage 64 is frozen to the completed Stage 63 states, Kn0=10, "
            "Tc/Th=0.1, omega=0.5, the 40x96 mapped-polar rule at radial "
            "scale 2.0, Pr=2/3, correction floor 0.05, and the preregistered "
            "source fractions; it is not a retuning stage."
        )


def _raw_moments(phi: np.ndarray, psi: np.ndarray, quadrature) -> np.ndarray:
    weight = quadrature.weight
    speed2 = quadrature.vx**2 + quadrature.vy**2
    energy_density = speed2 * phi + psi
    return np.asarray(
        [
            np.sum(phi * weight),
            np.sum(quadrature.vx * phi * weight),
            np.sum(quadrature.vy * phi * weight),
            np.sum(energy_density * weight),
            0.5 * np.sum(quadrature.vx * energy_density * weight),
            0.5 * np.sum(quadrature.vy * energy_density * weight),
        ],
        dtype=np.float64,
    )


def _relative_vector_error(actual: np.ndarray, expected: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(expected))), 1.0e-14)
    return float(np.max(np.abs(actual - expected))) / scale


def _state_error(macroscopic: dict[str, np.ndarray], fields: dict[str, float]) -> float:
    rho = float(fields["rho"])
    temperature = float(fields["T"])
    velocity_scale = math.sqrt(max(temperature, 1.0e-14))
    return max(
        abs(float(macroscopic["rho"]) - rho) / max(rho, 1.0e-14),
        abs(float(macroscopic["u"]) - float(fields["u"])) / velocity_scale,
        abs(float(macroscopic["v"]) - float(fields["v"])) / velocity_scale,
        abs(float(macroscopic["T"]) - temperature) / max(temperature, 1.0e-14),
    )


def _heat_flux_law_error(
    macroscopic: dict[str, np.ndarray],
    fields: dict[str, float],
    source_fraction: float,
) -> float:
    initial_q = np.asarray([fields["qx"], fields["qy"]], dtype=np.float64)
    expected = (1.0 - source_fraction * STAGE63_PRANDTL) * initial_q
    actual = np.asarray(
        [float(macroscopic["qx"]), float(macroscopic["qy"])], dtype=np.float64
    )
    scale = max(float(np.linalg.norm(initial_q)), fields["rho"] * fields["T"] ** 1.5, 1.0e-14)
    return float(np.linalg.norm(actual - expected)) / scale


def _manufactured_initial_distribution(state: dict[str, float | str], quadrature):
    amplified = dict(state)
    amplification = 1.0 / (1.0 - STAGE63_PRANDTL)
    amplified["qx_zeta"] = amplification * float(state["qx_zeta"])
    amplified["qy_zeta"] = amplification * float(state["qy_zeta"])
    target = independent_paper_target(
        amplified, quadrature, apply_retained_clipping=False
    )
    return np.asarray(target["phi"]), np.asarray(target["psi"])


def _conservative_diagnostic_target(
    fields: dict[str, float],
    clipped_phi: np.ndarray,
    clipped_psi: np.ndarray,
    quadrature,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    phi_m, psi_m = projected_maxwellian(
        fields["rho"], fields["u"], fields["v"], fields["T"], quadrature
    )
    phi_lower, psi_lower = _retained_clipped_lower_bounds(
        fields["rho"],
        fields["u"],
        fields["v"],
        fields["T"],
        fields["qx"],
        fields["qy"],
        phi_m,
        psi_m,
        quadrature,
    )
    phi_basis, psi_basis = _moment_basis(
        quadrature.vx, quadrature.vy, fields["u"], fields["v"]
    )
    target_moments = _target_moments(
        fields["rho"],
        fields["u"],
        fields["v"],
        fields["T"],
        fields["qx"],
        fields["qy"],
    )
    return bounded_conservative_projection(
        clipped_phi,
        clipped_psi,
        phi_lower,
        psi_lower,
        phi_basis,
        psi_basis,
        quadrature.weight,
        target_moments,
    )


def _source_arm_rows(
    arm: str,
    initial_phi: np.ndarray,
    initial_psi: np.ndarray,
    target_phi: np.ndarray,
    target_psi: np.ndarray,
    fields: dict[str, float],
    quadrature,
) -> list[dict[str, object]]:
    initial_raw = _raw_moments(initial_phi, initial_psi, quadrature)
    target_raw = _raw_moments(target_phi, target_psi, quadrature)
    rows = []
    for fraction in STAGE64_SOURCE_FRACTIONS:
        phi = (1.0 - fraction) * initial_phi + fraction * target_phi
        psi = (1.0 - fraction) * initial_psi + fraction * target_psi
        actual_raw = _raw_moments(phi, psi, quadrature)
        expected_raw = (1.0 - fraction) * initial_raw + fraction * target_raw
        macroscopic = projected_macroscopic(phi, psi, quadrature)
        rows.append(
            {
                "arm": arm,
                "source_fraction": fraction,
                "raw_moment_identity_error": _relative_vector_error(
                    actual_raw, expected_raw
                ),
                "conserved_state_drift": _state_error(macroscopic, fields),
                "ideal_shakhov_heat_flux_law_error": _heat_flux_law_error(
                    macroscopic, fields, fraction
                ),
                "rho": float(macroscopic["rho"]),
                "u": float(macroscopic["u"]),
                "v": float(macroscopic["v"]),
                "T": float(macroscopic["T"]),
                "qx": float(macroscopic["qx"]),
                "qy": float(macroscopic["qy"]),
                "minimum_phi": float(np.min(phi)),
                "minimum_psi": float(np.min(psi)),
            }
        )
    return rows


def audit_frozen_state(state: dict[str, float | str], quadrature) -> dict[str, object]:
    fields = _state_in_c_coordinates(state)
    initial_phi, initial_psi = _manufactured_initial_distribution(state, quadrature)
    initial_macroscopic = projected_macroscopic(initial_phi, initial_psi, quadrature)
    initial_state_error = _state_error(initial_macroscopic, fields)
    initial_q_error = _heat_flux_law_error(initial_macroscopic, fields, 0.0)

    active_phi, active_psi, active_diagnostics = projected_shakhov_equilibrium(
        fields,
        quadrature,
        prandtl=STAGE63_PRANDTL,
        correction_floor=STAGE63_CORRECTION_FLOOR,
    )
    unclipped = independent_paper_target(
        state, quadrature, apply_retained_clipping=False
    )
    conservative_phi, conservative_psi, projection_diagnostics = (
        _conservative_diagnostic_target(
            fields, active_phi, active_psi, quadrature
        )
    )

    arms = {
        "retained_clipped_active": (active_phi, active_psi),
        "unclipped_paper_diagnostic": (
            np.asarray(unclipped["phi"]), np.asarray(unclipped["psi"])
        ),
        "bounded_conservative_diagnostic": (
            conservative_phi, conservative_psi
        ),
    }
    source_rows = []
    for arm, (target_phi, target_psi) in arms.items():
        source_rows.extend(
            _source_arm_rows(
                arm,
                initial_phi,
                initial_psi,
                target_phi,
                target_psi,
                fields,
                quadrature,
            )
        )
    return {
        "name": str(state["name"]),
        "paper_state": dict(state),
        "c_coordinate_state": fields,
        "manufactured_initial": {
            "description": (
                "Signed unclipped manufactured distribution obtained by scaling "
                "the paper heat flux by 1/(1-Pr), so its recovered heat flux "
                "equals the frozen state before the source step."
            ),
            "state_error": initial_state_error,
            "heat_flux_error": initial_q_error,
            "minimum_phi": float(np.min(initial_phi)),
            "minimum_psi": float(np.min(initial_psi)),
        },
        "active_clipping": {
            key: float(np.asarray(value))
            for key, value in active_diagnostics.items()
        },
        "conservative_projection_diagnostic": projection_diagnostics,
        "source_rows": source_rows,
    }


def aggregate_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    by_arm: dict[str, list[dict[str, object]]] = {}
    initial_errors = []
    initial_q_errors = []
    projection_failures = 0
    materially_clipped_states = 0
    for row in rows:
        initial_errors.append(float(row["manufactured_initial"]["state_error"]))
        initial_q_errors.append(float(row["manufactured_initial"]["heat_flux_error"]))
        projection = row["conservative_projection_diagnostic"]
        if not bool(projection["converged"]):
            projection_failures += 1
        clipping = row["active_clipping"]
        if max(
            float(clipping["phi_clipped_weight_fraction"]),
            float(clipping["psi_clipped_weight_fraction"]),
        ) >= 1.0e-4:
            materially_clipped_states += 1
        for source_row in row["source_rows"]:
            by_arm.setdefault(str(source_row["arm"]), []).append(source_row)

    aggregate: dict[str, object] = {
        "state_count": len(rows),
        "source_fraction_count": len(STAGE64_SOURCE_FRACTIONS),
        "maximum_initial_state_error": max(initial_errors),
        "maximum_initial_heat_flux_error": max(initial_q_errors),
        "materially_clipped_state_count": materially_clipped_states,
        "conservative_projection_failure_count": projection_failures,
        "arms": {},
    }
    for arm, arm_rows in by_arm.items():
        aggregate["arms"][arm] = {
            "maximum_raw_moment_identity_error": max(
                float(item["raw_moment_identity_error"]) for item in arm_rows
            ),
            "maximum_conserved_state_drift": max(
                float(item["conserved_state_drift"]) for item in arm_rows
            ),
            "maximum_ideal_shakhov_heat_flux_law_error": max(
                float(item["ideal_shakhov_heat_flux_law_error"])
                for item in arm_rows
            ),
            "minimum_phi": min(float(item["minimum_phi"]) for item in arm_rows),
            "minimum_psi": min(float(item["minimum_psi"]) for item in arm_rows),
        }
    return aggregate


def stage64_decision(aggregate: dict[str, object]) -> str:
    values = [
        float(aggregate["maximum_initial_state_error"]),
        float(aggregate["maximum_initial_heat_flux_error"]),
    ]
    for arm in aggregate["arms"].values():
        values.extend(
            [
                float(arm["maximum_raw_moment_identity_error"]),
                float(arm["maximum_conserved_state_drift"]),
                float(arm["maximum_ideal_shakhov_heat_flux_law_error"]),
            ]
        )
    if not np.all(np.isfinite(values)):
        return "stage64_nonfinite_source_step_audit_blocker"
    if max(
        float(arm["maximum_raw_moment_identity_error"])
        for arm in aggregate["arms"].values()
    ) >= STAGE64_IDENTITY_TOLERANCE:
        return "stage64_source_step_raw_moment_identity_blocker"
    if (
        float(aggregate["maximum_initial_state_error"])
        >= STAGE64_INITIAL_MOMENT_TOLERANCE
        or float(aggregate["maximum_initial_heat_flux_error"])
        >= STAGE64_INITIAL_MOMENT_TOLERANCE
    ):
        return "stage64_manufactured_initial_state_blocker"
    if int(aggregate["conservative_projection_failure_count"]) != 0:
        return "stage64_conservative_diagnostic_projection_blocker"

    unclipped = aggregate["arms"]["unclipped_paper_diagnostic"]
    conservative = aggregate["arms"]["bounded_conservative_diagnostic"]
    active = aggregate["arms"]["retained_clipped_active"]
    if (
        float(unclipped["maximum_conserved_state_drift"])
        >= STAGE64_UNCLIPPED_CONSERVED_TOLERANCE
        or float(unclipped["maximum_ideal_shakhov_heat_flux_law_error"])
        >= STAGE64_UNCLIPPED_HEAT_FLUX_TOLERANCE
    ):
        return "stage64_unclipped_source_step_consistency_blocker"
    if (
        float(conservative["maximum_conserved_state_drift"])
        >= STAGE64_CONSERVATIVE_CONSERVED_TOLERANCE
        or float(conservative["maximum_ideal_shakhov_heat_flux_law_error"])
        >= STAGE64_CONSERVATIVE_HEAT_FLUX_TOLERANCE
    ):
        return "stage64_conservative_diagnostic_source_step_blocker"
    if max(
        float(active["maximum_conserved_state_drift"]),
        float(active["maximum_ideal_shakhov_heat_flux_law_error"]),
    ) >= STAGE64_MATERIAL_SOURCE_DEFECT:
        return (
            "stage64_source_update_algebraically_consistent_retained_clipping_"
            "injects_material_collision_defects_stage65_local_activation_map_audit"
        )
    return "stage64_no_material_source_defect_stop"


def run_stage64(
    stage63_artifact_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    validate_stage64_design()
    retained = validate_stage63_artifact(stage63_artifact_dir)
    quadrature = mapped_polar_quadrature(
        *STAGE63_RULE, radial_scale=STAGE63_RADIAL_SCALE
    )
    rows = [audit_frozen_state(state, quadrature) for state in FROZEN_PAPER_STATES]
    aggregate = aggregate_rows(rows)
    decision = stage64_decision(aggregate)
    summary = {
        "stage": 64,
        "description": (
            "Frozen homogeneous collision-only source-step audit. The active arm "
            "uses the exact retained clipped projected-Shakhov target; unclipped "
            "paper and bounded-conservative targets are diagnostics only."
        ),
        "retained_stage63_endpoint": dict(STAGE63_COMPLETED_ENDPOINT),
        "retained_stage63_decision": retained["decision"],
        "configuration": {
            "kn0_investigation_scope": STAGE63_KNUDSEN,
            "cold_hot_ratio": STAGE63_COLD_HOT_RATIO,
            "viscosity_exponent": STAGE63_VISCOSITY_EXPONENT,
            "velocity_rule": list(STAGE63_RULE),
            "radial_scale": STAGE63_RADIAL_SCALE,
            "prandtl": STAGE63_PRANDTL,
            "retained_correction_floor": STAGE63_CORRECTION_FLOOR,
            "frozen_state_count": len(FROZEN_PAPER_STATES),
            "source_fractions": list(STAGE64_SOURCE_FRACTIONS),
            "active_arm": "retained_clipped_active",
            "diagnostic_arms": [
                "unclipped_paper_diagnostic",
                "bounded_conservative_diagnostic",
            ],
            "solver_rerun": False,
            "signed_manufactured_initial_state": True,
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "correction_floor_retuning": False,
            "normalization_retuning": False,
            "transport_retuning": False,
            "velocity_quadrature_retuning": False,
            "wall_model_retuning": False,
            "source_fraction_retuning": False,
            "cross_knudsen_extension_permitted": False,
            "conservative_projection_adopted": False,
        },
        "thresholds": {
            "raw_moment_identity": STAGE64_IDENTITY_TOLERANCE,
            "manufactured_initial_moments": STAGE64_INITIAL_MOMENT_TOLERANCE,
            "unclipped_conserved": STAGE64_UNCLIPPED_CONSERVED_TOLERANCE,
            "unclipped_heat_flux": STAGE64_UNCLIPPED_HEAT_FLUX_TOLERANCE,
            "conservative_conserved": STAGE64_CONSERVATIVE_CONSERVED_TOLERANCE,
            "conservative_heat_flux": STAGE64_CONSERVATIVE_HEAT_FLUX_TOLERANCE,
            "material_source_defect": STAGE64_MATERIAL_SOURCE_DEFECT,
        },
        "aggregate": aggregate,
        "rows": rows,
        "decision": decision,
        "positive_findings": [
            "The distribution update is checked against independent raw-moment linear-relaxation identities at four frozen source fractions.",
            "The unclipped paper target and bounded-conservative projection are retained as labeled diagnostics rather than silently replacing the active solver target.",
            "All physical, quadrature, clipping, normalization, wall and cross-Knudsen settings remain frozen."
        ],
        "negative_findings": [
            "This signed manufactured-state source audit is not a positivity, stability, cavity-flow or external-validation result.",
            "Any conserved-state or ideal heat-flux relaxation drift generated by the retained clipped active target is reported rather than removed by floor retuning.",
            "The conservative diagnostic remains non-adopted because the completed full-cavity confirmation did not improve the unresolved benchmark endpoint.",
            "The failed MUSCL endpoint remains negative and cross-Knudsen extension remains prohibited."
        ],
        "interpretation_guard": (
            "Stage 64 audits one frozen homogeneous source step. It does not rerun "
            "transport or walls, validate Table 3 or Table 6, establish positivity "
            "of the manufactured initial distributions, or authorize retuning."
        ),
        "scientifically_justified_next_scope": (
            "If the active clipped source step is algebraically correct but materially "
            "non-conservative, quantify where and how often clipping activates in an "
            "exact retained full-cavity endpoint and map the local source defects, "
            "without rerunning or retuning the solver."
        ),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage63-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = run_stage64(args.stage63_artifact_dir, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
