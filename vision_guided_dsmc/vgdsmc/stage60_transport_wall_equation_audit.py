from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np

from .linear_sidewall_validation import LinearSidewallConfig, sidewall_temperature_profile
from .stage41_projected_polar_operator_audit import (
    PolarQuadrature,
    mapped_polar_quadrature,
    projected_maxwellian,
    wall_mass_balance_error,
)
from .stage42_projected_polar_heated_cavity_pilot import (
    _profile_diffuse_incoming,
    _upwind_neighbors,
    bottom_wall_heat_flux,
    projected_wall_incoming,
)


STAGE59_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30884941157,
    "workflow_job_id": 91913958890,
    "workflow_conclusion": "success",
    "tests_passed": 14,
    "tests_failed": 0,
    "artifact_id": 8885375771,
    "artifact_size_bytes": 2135,
    "artifact_sha256": "416dc86d095e4dae745d246e31912d47e57e0474878f153712021cb899b548db",
    "source_head_sha": "5b21563f2786cb627bd2ed185637774fc372a21e",
    "summary_sha256": "d562563b58c5cb08aea638278cc2486b03e2ea526cd89e369956c6bb39b0929b",
    "decision": (
        "stage59_independent_dsmc_heat_flux_confirms_discrepancy_"
        "projection_not_adopted_transport_wall_audit_next"
    ),
}

STAGE60_GRID = (7, 5)
STAGE60_RULE = (40, 96)
STAGE60_RADIAL_SCALE = 2.0
STAGE60_KNUDSEN = 10.0
STAGE60_COLD_HOT_RATIO = 0.1
STAGE60_HALF_SPACE_TEMPERATURES = (0.1, 0.35, 1.0)
STAGE60_OUTGOING_TEMPERATURE = 0.55
STAGE60_TRANSPORT_BALANCE_TOLERANCE = 1.0e-12
STAGE60_ISOTHERMAL_FIXED_POINT_TOLERANCE = 1.0e-12
STAGE60_WALL_MASS_TOLERANCE = 1.0e-12
STAGE60_HALF_SPACE_ENERGY_TOLERANCE = 1.0e-8
STAGE60_OBSERVABLE_IDENTITY_TOLERANCE = 1.0e-12


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage59_artifact(root: str | Path) -> dict[str, object]:
    root = Path(root)
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise ValueError("Stage 59 summary is missing")
    if sha256_file(summary_path) != STAGE59_COMPLETED_ENDPOINT["summary_sha256"]:
        raise ValueError("Stage 59 summary checksum mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("stage") != 59:
        raise ValueError("Stage 59 artifact stage mismatch")
    if summary.get("decision") != STAGE59_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage 59 artifact decision mismatch")
    return summary


def validate_stage60_design(
    grid: tuple[int, int],
    rule: tuple[int, int],
    radial_scale: float,
    kn0: float,
    cold_hot_ratio: float,
) -> None:
    if grid != STAGE60_GRID:
        raise ValueError("Stage 60 uses the frozen non-square 7x5 equation-audit grid")
    if rule != STAGE60_RULE:
        raise ValueError("Stage 60 retains the Stage 58 40x96 velocity rule")
    if radial_scale != STAGE60_RADIAL_SCALE:
        raise ValueError("Stage 60 retains radial mapping scale 2.0")
    if kn0 != STAGE60_KNUDSEN:
        raise ValueError("Stage 60 remains scoped to Kn0=10")
    if cold_hot_ratio != STAGE60_COLD_HOT_RATIO:
        raise ValueError("Stage 60 retains Tcold/Thot=0.1")


def _relative_pair_error(left: float, right: float) -> float:
    return abs(float(left) - float(right)) / max(
        abs(float(left)) + abs(float(right)), 1.0e-14
    )


def _boundary_flux_per_velocity(
    distribution: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
    quadrature: PolarQuadrature,
    dx: float,
    dy: float,
) -> np.ndarray:
    left_face = np.where(
        (quadrature.vx > 0.0)[None, :], left, distribution[:, 0]
    )
    right_face = np.where(
        (quadrature.vx < 0.0)[None, :], right, distribution[:, -1]
    )
    bottom_face = np.where(
        (quadrature.vy > 0.0)[None, :], bottom, distribution[0]
    )
    top_face = np.where(
        (quadrature.vy < 0.0)[None, :], top, distribution[-1]
    )
    x_flux = dy * np.sum(
        -quadrature.vx[None, :] * left_face
        + quadrature.vx[None, :] * right_face,
        axis=0,
    )
    y_flux = dx * np.sum(
        -quadrature.vy[None, :] * bottom_face
        + quadrature.vy[None, :] * top_face,
        axis=0,
    )
    return x_flux + y_flux


def _streaming_residual(
    distribution: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
    quadrature: PolarQuadrature,
    dx: float,
    dy: float,
) -> np.ndarray:
    x_upwind, y_upwind = _upwind_neighbors(
        distribution, left, right, bottom, top, quadrature
    )
    return (
        np.abs(quadrature.vx)[None, None, :] * (distribution - x_upwind) / dx
        + np.abs(quadrature.vy)[None, None, :] * (distribution - y_upwind) / dy
    )


def _deterministic_positive_state(
    cfg: LinearSidewallConfig,
    quadrature: PolarQuadrature,
) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(cfg.nx, dtype=np.float64) + 0.5) / cfg.nx
    y = (np.arange(cfg.ny, dtype=np.float64) + 0.5) / cfg.ny
    yy, xx = np.meshgrid(y, x, indexing="ij")
    rho = 0.85 + 0.15 * np.sin(math.pi * xx) * np.cos(math.pi * yy)
    u = 0.03 * np.sin(2.0 * math.pi * yy)
    v = -0.02 * np.cos(2.0 * math.pi * xx)
    temperature = (
        cfg.cold_temperature
        + (cfg.hot_temperature - cfg.cold_temperature) * (1.0 - yy)
        + 0.05 * np.sin(math.pi * xx) * np.sin(math.pi * yy)
    )
    phi, psi = projected_maxwellian(rho, u, v, temperature, quadrature)
    phi_factor = (
        1.0
        + 0.05 * np.sin(2.0 * math.pi * xx)[..., None]
        * np.tanh(quadrature.vx)[None, None, :]
        + 0.04 * np.cos(2.0 * math.pi * yy)[..., None]
        * np.tanh(quadrature.vy)[None, None, :]
    )
    psi_factor = (
        1.0
        + 0.03 * np.cos(2.0 * math.pi * xx)[..., None]
        * np.tanh(quadrature.vy)[None, None, :]
        - 0.02 * np.sin(2.0 * math.pi * yy)[..., None]
        * np.tanh(quadrature.vx)[None, None, :]
    )
    phi *= phi_factor
    psi *= psi_factor
    if float(np.min(phi)) < 0.0 or float(np.min(psi)) < 0.0:
        raise ValueError("manufactured Stage 60 distributions must remain nonnegative")
    return phi, psi


def transport_conservation_audit(
    cfg: LinearSidewallConfig,
    quadrature: PolarQuadrature,
) -> dict[str, float]:
    phi, psi = _deterministic_positive_state(cfg, quadrature)
    incoming = projected_wall_incoming(phi, psi, cfg, quadrature)
    (
        left_phi, left_psi, right_phi, right_psi,
        bottom_phi, bottom_psi, top_phi, top_psi,
    ) = incoming
    dx = 1.0 / cfg.nx
    dy = 1.0 / cfg.ny
    residual_phi = _streaming_residual(
        phi, left_phi, right_phi, bottom_phi, top_phi, quadrature, dx, dy
    )
    residual_psi = _streaming_residual(
        psi, left_psi, right_psi, bottom_psi, top_psi, quadrature, dx, dy
    )
    cell_phi = np.sum(residual_phi, axis=(0, 1)) * dx * dy
    cell_psi = np.sum(residual_psi, axis=(0, 1)) * dx * dy
    boundary_phi = _boundary_flux_per_velocity(
        phi, left_phi, right_phi, bottom_phi, top_phi, quadrature, dx, dy
    )
    boundary_psi = _boundary_flux_per_velocity(
        psi, left_psi, right_psi, bottom_psi, top_psi, quadrature, dx, dy
    )
    weight = quadrature.weight
    phi_relative = float(
        np.sum(np.abs(cell_phi - boundary_phi) * weight)
        / max(np.sum((np.abs(cell_phi) + np.abs(boundary_phi)) * weight), 1.0e-14)
    )
    psi_relative = float(
        np.sum(np.abs(cell_psi - boundary_psi) * weight)
        / max(np.sum((np.abs(cell_psi) + np.abs(boundary_psi)) * weight), 1.0e-14)
    )
    cell_mass = float(np.sum(cell_phi * weight))
    boundary_mass = float(np.sum(boundary_phi * weight))
    speed2 = quadrature.vx**2 + quadrature.vy**2
    cell_energy = float(0.5 * np.sum((speed2 * cell_phi + cell_psi) * weight))
    boundary_energy = float(
        0.5 * np.sum((speed2 * boundary_phi + boundary_psi) * weight)
    )
    return {
        "phi_telescoping_relative_error": phi_relative,
        "psi_telescoping_relative_error": psi_relative,
        "mass_balance_identity_relative_error":
            _relative_pair_error(cell_mass, boundary_mass),
        "energy_balance_identity_relative_error":
            _relative_pair_error(cell_energy, boundary_energy),
        "cell_integrated_mass_residual": cell_mass,
        "boundary_mass_flux": boundary_mass,
        "cell_integrated_energy_residual": cell_energy,
        "boundary_energy_flux": boundary_energy,
    }


def isothermal_collision_off_audit(
    quadrature: PolarQuadrature,
) -> dict[str, float]:
    cfg = LinearSidewallConfig(
        nx=STAGE60_GRID[0],
        ny=STAGE60_GRID[1],
        kn0=STAGE60_KNUDSEN,
        cold_hot_ratio=1.0,
    )
    rho = np.full((cfg.ny, cfg.nx), 0.73)
    zero = np.zeros_like(rho)
    temperature = np.ones_like(rho)
    phi, psi = projected_maxwellian(rho, zero, zero, temperature, quadrature)
    incoming = projected_wall_incoming(phi, psi, cfg, quadrature)
    (
        left_phi, left_psi, right_phi, right_psi,
        bottom_phi, bottom_psi, top_phi, top_psi,
    ) = incoming
    phi_x, phi_y = _upwind_neighbors(
        phi, left_phi, right_phi, bottom_phi, top_phi, quadrature
    )
    psi_x, psi_y = _upwind_neighbors(
        psi, left_psi, right_psi, bottom_psi, top_psi, quadrature
    )
    dx = 1.0 / cfg.nx
    dy = 1.0 / cfg.ny
    ax = np.abs(quadrature.vx) / dx
    ay = np.abs(quadrature.vy) / dy
    denominator = ax + ay
    candidate_phi = (
        ax[None, None, :] * phi_x + ay[None, None, :] * phi_y
    ) / denominator[None, None, :]
    candidate_psi = (
        ax[None, None, :] * psi_x + ay[None, None, :] * psi_y
    ) / denominator[None, None, :]
    phi_error = float(np.max(np.abs(candidate_phi - phi)) / np.max(phi))
    psi_error = float(np.max(np.abs(candidate_psi - psi)) / np.max(psi))
    wall_balance = max(
        wall_mass_balance_error(
            phi[:, 0], left_phi, quadrature.vx, quadrature.vx > 0.0, quadrature
        ),
        wall_mass_balance_error(
            phi[:, -1], right_phi, -quadrature.vx, quadrature.vx < 0.0, quadrature
        ),
        wall_mass_balance_error(
            phi[0], bottom_phi, quadrature.vy, quadrature.vy > 0.0, quadrature
        ),
        wall_mass_balance_error(
            phi[-1], top_phi, -quadrature.vy, quadrature.vy < 0.0, quadrature
        ),
    )
    heat_flux = bottom_wall_heat_flux(
        phi, psi, bottom_phi, bottom_psi, quadrature
    )
    return {
        "phi_fixed_point_relative_error": phi_error,
        "psi_fixed_point_relative_error": psi_error,
        "maximum_wall_mass_balance_error": float(wall_balance),
        "maximum_absolute_bottom_wall_heat_flux": float(np.max(np.abs(heat_flux))),
    }


def _orientation_rows(
    quadrature: PolarQuadrature,
) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    return (
        ("left", quadrature.vx, quadrature.vx > 0.0),
        ("right", -quadrature.vx, quadrature.vx < 0.0),
        ("bottom", quadrature.vy, quadrature.vy > 0.0),
        ("top", -quadrature.vy, quadrature.vy < 0.0),
    )


def diffuse_wall_half_space_audit(
    quadrature: PolarQuadrature,
) -> dict[str, object]:
    gas_phi, gas_psi = projected_maxwellian(
        np.asarray(0.8),
        np.asarray(0.0),
        np.asarray(0.0),
        np.asarray(STAGE60_OUTGOING_TEMPERATURE),
        quadrature,
    )
    speed2 = quadrature.vx**2 + quadrature.vy**2
    rows: list[dict[str, float | str]] = []
    for wall_temperature in STAGE60_HALF_SPACE_TEMPERATURES:
        for orientation, normal_velocity, incoming_mask in _orientation_rows(quadrature):
            incoming_phi, incoming_psi = _profile_diffuse_incoming(
                gas_phi[None, :],
                gas_psi[None, :],
                normal_velocity,
                incoming_mask,
                np.asarray([wall_temperature]),
                quadrature,
            )
            boundary_phi = np.where(incoming_mask, incoming_phi[0], gas_phi)
            boundary_psi = np.where(incoming_mask, incoming_psi[0], gas_psi)
            net_mass = float(np.sum(
                normal_velocity * boundary_phi * quadrature.weight
            ))
            absolute_mass_flux = float(np.sum(
                np.abs(normal_velocity * boundary_phi) * quadrature.weight
            ))
            incoming_mass = float(np.sum(
                normal_velocity * incoming_phi[0] * incoming_mask * quadrature.weight
            ))
            incoming_energy = float(0.5 * np.sum(
                normal_velocity
                * (speed2 * incoming_phi[0] + incoming_psi[0])
                * incoming_mask
                * quadrature.weight
            ))
            net_energy = float(0.5 * np.sum(
                normal_velocity
                * (speed2 * boundary_phi + boundary_psi)
                * quadrature.weight
            ))
            expected_incoming_energy = 2.0 * wall_temperature * incoming_mass
            expected_net_energy = (
                2.0
                * (wall_temperature - STAGE60_OUTGOING_TEMPERATURE)
                * incoming_mass
            )
            observable_error = 0.0
            if orientation == "bottom":
                observed = float(bottom_wall_heat_flux(
                    gas_phi[None, None, :],
                    gas_psi[None, None, :],
                    incoming_phi,
                    incoming_psi,
                    quadrature,
                )[0])
                observable_error = _relative_pair_error(
                    observed, net_energy / math.sqrt(2.0)
                )
            rows.append({
                "orientation": orientation,
                "wall_temperature": float(wall_temperature),
                "relative_mass_balance_error": abs(net_mass)
                    / max(absolute_mass_flux, 1.0e-14),
                "incoming_energy_per_mass_relative_error":
                    _relative_pair_error(incoming_energy, expected_incoming_energy),
                "net_energy_exchange_relative_error":
                    _relative_pair_error(net_energy, expected_net_energy),
                "bottom_observable_identity_relative_error": observable_error,
                "incoming_mass_flux": incoming_mass,
                "net_energy_flux": net_energy,
            })
    return {
        "rows": rows,
        "maximum_relative_mass_balance_error": max(
            float(row["relative_mass_balance_error"]) for row in rows
        ),
        "maximum_incoming_energy_per_mass_relative_error": max(
            float(row["incoming_energy_per_mass_relative_error"]) for row in rows
        ),
        "maximum_net_energy_exchange_relative_error": max(
            float(row["net_energy_exchange_relative_error"]) for row in rows
        ),
        "maximum_bottom_observable_identity_relative_error": max(
            float(row["bottom_observable_identity_relative_error"]) for row in rows
        ),
    }


def evaluate_stage60() -> dict[str, object]:
    validate_stage60_design(
        STAGE60_GRID,
        STAGE60_RULE,
        STAGE60_RADIAL_SCALE,
        STAGE60_KNUDSEN,
        STAGE60_COLD_HOT_RATIO,
    )
    quadrature = mapped_polar_quadrature(
        STAGE60_RULE[0], STAGE60_RULE[1], STAGE60_RADIAL_SCALE
    )
    cfg = LinearSidewallConfig(
        nx=STAGE60_GRID[0],
        ny=STAGE60_GRID[1],
        kn0=STAGE60_KNUDSEN,
        cold_hot_ratio=STAGE60_COLD_HOT_RATIO,
    )
    transport = transport_conservation_audit(cfg, quadrature)
    isothermal = isothermal_collision_off_audit(quadrature)
    wall = diffuse_wall_half_space_audit(quadrature)

    transport_pass = max(
        float(transport["phi_telescoping_relative_error"]),
        float(transport["psi_telescoping_relative_error"]),
        float(transport["mass_balance_identity_relative_error"]),
        float(transport["energy_balance_identity_relative_error"]),
    ) <= STAGE60_TRANSPORT_BALANCE_TOLERANCE
    isothermal_pass = max(
        float(isothermal["phi_fixed_point_relative_error"]),
        float(isothermal["psi_fixed_point_relative_error"]),
        float(isothermal["maximum_absolute_bottom_wall_heat_flux"]),
    ) <= STAGE60_ISOTHERMAL_FIXED_POINT_TOLERANCE
    wall_mass_pass = (
        float(wall["maximum_relative_mass_balance_error"])
        <= STAGE60_WALL_MASS_TOLERANCE
    )
    wall_energy_pass = max(
        float(wall["maximum_incoming_energy_per_mass_relative_error"]),
        float(wall["maximum_net_energy_exchange_relative_error"]),
    ) <= STAGE60_HALF_SPACE_ENERGY_TOLERANCE
    observable_pass = (
        float(wall["maximum_bottom_observable_identity_relative_error"])
        <= STAGE60_OBSERVABLE_IDENTITY_TOLERANCE
    )

    if not transport_pass:
        decision = "stage60_transport_conservation_blocker"
        next_scope = "Review the first-order upwind residual and boundary-face sign conventions before any further solver run."
    elif not isothermal_pass:
        decision = "stage60_collision_off_isothermal_fixed_point_blocker"
        next_scope = "Review collision-off source iteration and isothermal diffuse-wall reconstruction before any further solver run."
    elif not wall_mass_pass or not wall_energy_pass or not observable_pass:
        decision = "stage60_diffuse_wall_mass_or_energy_flux_blocker"
        next_scope = "Review projected diffuse-wall mass scaling, half-space energy moments and wall heat-flux extraction before any further solver run."
    else:
        decision = (
            "stage60_transport_and_diffuse_wall_equations_close_"
            "discrepancy_not_explained_characteristic_audit_next"
        )
        next_scope = (
            "Construct an independent characteristic-based collision-off, non-isothermal cavity "
            "reference on the same frozen velocity quadrature and compare it with the first-order "
            "upwind discrete solution to quantify transport diffusion without parameter retuning."
        )

    return {
        "stage": 60,
        "description": (
            "Equation-level audit of the frozen Stage 58 projected-polar first-order transport, "
            "diffuse-wall mass/energy fluxes and collision-off isothermal fixed point."
        ),
        "retained_stage59_endpoint": STAGE59_COMPLETED_ENDPOINT,
        "configuration": {
            "diagnostic_grid": list(STAGE60_GRID),
            "velocity_rule": list(STAGE60_RULE),
            "radial_scale": STAGE60_RADIAL_SCALE,
            "kn0_scope": STAGE60_KNUDSEN,
            "cold_hot_ratio": STAGE60_COLD_HOT_RATIO,
            "transport_operator": "steady_first_order_upwind_jacobi",
            "wall_operator": "projected_diffuse_fully_accommodating_zero_mass_flux",
            "solver_rerun": False,
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "transport_retuning": False,
            "wall_model_retuning": False,
            "normalization_retuning": False,
            "correction_floor_retuning": False,
            "source_relaxation_retuning": False,
            "velocity_quadrature_retuning": False,
            "cross_knudsen_extension_permitted": False,
        },
        "thresholds": {
            "transport_balance": STAGE60_TRANSPORT_BALANCE_TOLERANCE,
            "isothermal_fixed_point": STAGE60_ISOTHERMAL_FIXED_POINT_TOLERANCE,
            "wall_mass": STAGE60_WALL_MASS_TOLERANCE,
            "half_space_energy": STAGE60_HALF_SPACE_ENERGY_TOLERANCE,
            "wall_observable_identity": STAGE60_OBSERVABLE_IDENTITY_TOLERANCE,
        },
        "transport_conservation": transport,
        "collision_off_isothermal": isothermal,
        "diffuse_wall_half_space": wall,
        "checks": {
            "transport_conservation_pass": bool(transport_pass),
            "collision_off_isothermal_fixed_point_pass": bool(isothermal_pass),
            "diffuse_wall_mass_pass": bool(wall_mass_pass),
            "diffuse_wall_energy_pass": bool(wall_energy_pass),
            "wall_heat_flux_observable_identity_pass": bool(observable_pass),
        },
        "decision": decision,
        "positive_findings": [
            "The finite-volume first-order upwind residual telescopes to the boundary flux for both projected distributions and for mass and total energy moments.",
            "A uniform isothermal Maxwellian is a collision-off fixed point with zero wall heat flux and machine-level wall-mass balance.",
            "Diffuse re-emission satisfies zero mass flux and the monatomic half-space energy identity Ein/Jin=2Tw in all four wall orientations.",
            "The implemented bottom-wall observable equals the equation-level boundary energy flux with the retained sqrt(2) normalization.",
        ] if all((transport_pass, isothermal_pass, wall_mass_pass, wall_energy_pass, observable_pass)) else [
            "Stage 60 retains every passing transport or wall identity even if another preregistered check fails."
        ],
        "negative_findings": [
            "These exact identities do not explain or remove the greater-than-25% Kn0=10 heat-flux discrepancy confirmed in Stage 59.",
            "Equation closure does not establish accuracy of the first-order upwind transport for the strongly non-isothermal free-molecular limit.",
            "No independent collision-off non-isothermal characteristic solution or independent DSMC wall-velocity profile is yet available.",
        ],
        "interpretation_guard": (
            "Passing equation-level conservation and wall-moment checks rules out basic sign, "
            "telescoping and half-space-flux inconsistencies at the audited quadrature. It does not "
            "validate the cavity solver, justify the conservative projection, authorize retuning, "
            "or permit cross-Knudsen extension."
        ),
        "scientifically_justified_next_scope": next_scope,
    }


def run_stage60(stage59_artifact_dir: str | Path, output_dir: str | Path) -> dict[str, object]:
    validate_stage59_artifact(stage59_artifact_dir)
    summary = evaluate_stage60()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage59-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(
        run_stage60(args.stage59_artifact_dir, args.output_dir),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
