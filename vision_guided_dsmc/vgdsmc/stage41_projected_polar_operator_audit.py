from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import numpy as np

STAGE40_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30725610349,
    "workflow_job_id": 91436612348,
    "workflow_conclusion": "success",
    "tests_passed": 74,
    "tests_failed": 0,
    "artifact_id": 8826814764,
    "artifact_sha256": "ca7a493fc5c17e416ba06905fdbdc19598535cb5d8aecd7adae6a883c0ee1eb7",
    "head_sha": "39034139a6864947c0ccefc9b5d2fb48ca79e070",
    "predicted_qav": 0.0726173364328985,
    "source_shakhov_qav": 0.072,
    "source_dsmc_qav": 0.0716,
    "relative_error_to_source_shakhov": 0.008574117123590345,
    "relative_error_to_source_dsmc": 0.014208609398023778,
    "table3_independent_reference_available": False,
    "decision": "heat_flux_independently_supported_stage41_projected_polar_dvm",
}

SOURCE_ARCHITECTURE = {
    "distribution_representation": "projected_phi_psi_in_two_dimensional_molecular_velocity",
    "velocity_coordinates": "polar",
    "radial_quadrature": "mapped_Gauss_Legendre",
    "angular_quadrature": "trapezoidal",
    "radial_nodes_M": 80,
    "angular_nodes_N": 400,
    "velocity_vector_count": 32000,
    "physical_grid": [400, 400],
    "transport": "second_order_control_volume",
    "reported_convergence_tolerance": 1.0e-10,
}

STAGE41_COARSE_RULE = (16, 48)
STAGE41_FINE_RULE = (32, 96)
STAGE41_RADIAL_SCALE = 1.0
STAGE41_PRANDTL = 2.0 / 3.0
STAGE41_CORRECTION_FLOOR = 0.05


@dataclass(frozen=True)
class PolarQuadrature:
    vx: np.ndarray
    vy: np.ndarray
    weight: np.ndarray
    radius: np.ndarray
    angle: np.ndarray
    radial_nodes: int
    angular_nodes: int
    radial_scale: float

    @property
    def point_count(self) -> int:
        return int(self.weight.size)


def mapped_polar_quadrature(
    radial_nodes: int,
    angular_nodes: int,
    radial_scale: float = STAGE41_RADIAL_SCALE,
) -> PolarQuadrature:
    """Mapped Gauss-Legendre radius on [0,inf) and periodic trapezoidal angle."""
    if radial_nodes < 4 or angular_nodes < 8:
        raise ValueError("insufficient polar quadrature resolution")
    if angular_nodes % 4 != 0:
        raise ValueError("angular_nodes must be divisible by four")
    if radial_scale <= 0.0:
        raise ValueError("radial_scale must be positive")

    abscissa, radial_weight = np.polynomial.legendre.leggauss(radial_nodes)
    radius = radial_scale * (1.0 + abscissa) / (1.0 - abscissa)
    jacobian = 2.0 * radial_scale / (1.0 - abscissa) ** 2
    angle = 2.0 * math.pi * np.arange(angular_nodes, dtype=np.float64) / angular_nodes
    rr, tt = np.meshgrid(radius, angle, indexing="ij")
    weight = (
        (radial_weight * jacobian * radius)[:, None]
        * np.full((1, angular_nodes), 2.0 * math.pi / angular_nodes)
    )
    return PolarQuadrature(
        vx=(rr * np.cos(tt)).ravel(),
        vy=(rr * np.sin(tt)).ravel(),
        weight=weight.ravel(),
        radius=rr.ravel(),
        angle=tt.ravel(),
        radial_nodes=radial_nodes,
        angular_nodes=angular_nodes,
        radial_scale=radial_scale,
    )


def _matching_fields(*values: np.ndarray | float) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in values)
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays[1:]):
        raise ValueError("macroscopic fields must have matching shapes")
    return arrays


def projected_maxwellian(
    rho: np.ndarray | float,
    u: np.ndarray | float,
    v: np.ndarray | float,
    temperature: np.ndarray | float,
    quadrature: PolarQuadrature,
) -> tuple[np.ndarray, np.ndarray]:
    """Return phi=int f dc_z and psi=int c_z^2 f dc_z for a 3-D Maxwellian."""
    rho, u, v, temperature = _matching_fields(rho, u, v, temperature)
    temperature = np.maximum(temperature, 1.0e-12)
    cx = quadrature.vx - u[..., None]
    cy = quadrature.vy - v[..., None]
    c_parallel2 = cx * cx + cy * cy
    raw_phi = np.exp(-c_parallel2 / (2.0 * temperature[..., None]))
    raw_phi /= 2.0 * math.pi * temperature[..., None]
    discrete_mass = np.sum(raw_phi * quadrature.weight, axis=-1)
    phi = rho[..., None] * raw_phi / np.maximum(discrete_mass[..., None], 1.0e-300)
    psi = temperature[..., None] * phi
    return phi, psi


def projected_macroscopic(
    phi: np.ndarray,
    psi: np.ndarray,
    quadrature: PolarQuadrature,
) -> dict[str, np.ndarray]:
    """Recover 3-D monatomic moments from the two projected distributions."""
    phi = np.asarray(phi, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    if phi.shape != psi.shape or phi.shape[-1] != quadrature.point_count:
        raise ValueError("phi, psi, and quadrature dimensions must match")
    weight = quadrature.weight
    rho = np.sum(phi * weight, axis=-1)
    safe_rho = np.maximum(rho, 1.0e-14)
    u = np.sum(phi * quadrature.vx * weight, axis=-1) / safe_rho
    v = np.sum(phi * quadrature.vy * weight, axis=-1) / safe_rho
    cx = quadrature.vx - u[..., None]
    cy = quadrature.vy - v[..., None]
    c_parallel2 = cx * cx + cy * cy
    total_internal_moment = np.sum(
        (c_parallel2 * phi + psi) * weight,
        axis=-1,
    )
    temperature = total_internal_moment / (3.0 * safe_rho)
    qx = 0.5 * np.sum(
        cx * (c_parallel2 * phi + psi) * weight,
        axis=-1,
    )
    qy = 0.5 * np.sum(
        cy * (c_parallel2 * phi + psi) * weight,
        axis=-1,
    )
    return {
        "rho": rho,
        "u": u,
        "v": v,
        "T": temperature,
        "qx": qx,
        "qy": qy,
        "total_internal_moment": total_internal_moment,
    }


def projected_shakhov_equilibrium(
    fields: dict[str, np.ndarray | float],
    quadrature: PolarQuadrature,
    prandtl: float = STAGE41_PRANDTL,
    correction_floor: float = STAGE41_CORRECTION_FLOOR,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Projected Shakhov equilibrium.

    Integrating the 3-D Shakhov correction over c_z gives
      phi_S = phi_M [1 + A (c_parallel.q) (c_parallel^2/T - 4)]
      psi_S = T phi_M [1 + A (c_parallel.q) (c_parallel^2/T - 2)]
    with A=(1-Pr)/(5 p T). The -4 and -2 constants are the exact Gaussian
    c_z projections of the original 3-D factor (c^2/T - 5).
    """
    if not 0.0 < prandtl <= 1.0:
        raise ValueError("prandtl must lie in (0,1]")
    if not 0.0 < correction_floor <= 1.0:
        raise ValueError("correction_floor must lie in (0,1]")

    rho, u, v, temperature, qx, qy = _matching_fields(
        fields["rho"], fields["u"], fields["v"], fields["T"],
        fields["qx"], fields["qy"],
    )
    temperature = np.maximum(temperature, 1.0e-12)
    rho = np.maximum(rho, 1.0e-14)
    phi_m, psi_m = projected_maxwellian(rho, u, v, temperature, quadrature)
    cx = quadrature.vx - u[..., None]
    cy = quadrature.vy - v[..., None]
    c_parallel2 = cx * cx + cy * cy
    c_dot_q = cx * qx[..., None] + cy * qy[..., None]
    coefficient = (1.0 - prandtl) / (
        5.0 * rho[..., None] * temperature[..., None] ** 2
    )
    raw_phi_factor = 1.0 + coefficient * c_dot_q * (
        c_parallel2 / temperature[..., None] - 4.0
    )
    raw_psi_factor = 1.0 + coefficient * c_dot_q * (
        c_parallel2 / temperature[..., None] - 2.0
    )
    phi = phi_m * np.maximum(raw_phi_factor, correction_floor)
    psi = psi_m * np.maximum(raw_psi_factor, correction_floor)
    density = np.sum(phi * quadrature.weight, axis=-1)
    density_scale = rho / np.maximum(density, 1.0e-14)
    phi *= density_scale[..., None]
    psi *= density_scale[..., None]

    phi_mass = np.sum(phi_m * quadrature.weight, axis=-1)
    psi_mass = np.sum(psi_m * quadrature.weight, axis=-1)
    diagnostics = {
        "phi_clipped_weight_fraction": np.sum(
            phi_m * quadrature.weight * (raw_phi_factor < correction_floor),
            axis=-1,
        ) / np.maximum(phi_mass, 1.0e-300),
        "psi_clipped_weight_fraction": np.sum(
            psi_m * quadrature.weight * (raw_psi_factor < correction_floor),
            axis=-1,
        ) / np.maximum(psi_mass, 1.0e-300),
        "minimum_raw_phi_factor": np.min(raw_phi_factor, axis=-1),
        "minimum_raw_psi_factor": np.min(raw_psi_factor, axis=-1),
    }
    return phi, psi, diagnostics


def unit_projected_wall_maxwellian(
    temperature: float,
    quadrature: PolarQuadrature,
) -> tuple[np.ndarray, np.ndarray]:
    if temperature <= 0.0:
        raise ValueError("wall temperature must be positive")
    return projected_maxwellian(
        np.asarray(1.0), np.asarray(0.0), np.asarray(0.0),
        np.asarray(temperature), quadrature,
    )


def diffuse_wall_incoming(
    outgoing_phi: np.ndarray,
    outgoing_psi: np.ndarray,
    normal_velocity: np.ndarray,
    incoming_mask: np.ndarray,
    wall_temperature: float,
    quadrature: PolarQuadrature,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Diffuse incoming phi/psi with one density scale fixed by zero mass flux."""
    outgoing_phi = np.asarray(outgoing_phi, dtype=np.float64)
    outgoing_psi = np.asarray(outgoing_psi, dtype=np.float64)
    normal_velocity = np.asarray(normal_velocity, dtype=np.float64)
    incoming_mask = np.asarray(incoming_mask, dtype=bool)
    if outgoing_phi.shape != outgoing_psi.shape:
        raise ValueError("outgoing phi and psi must match")
    if outgoing_phi.shape[-1] != quadrature.point_count:
        raise ValueError("distribution and quadrature sizes must match")
    if normal_velocity.shape != quadrature.weight.shape:
        raise ValueError("normal velocity must match quadrature")
    if incoming_mask.shape != quadrature.weight.shape:
        raise ValueError("incoming mask must match quadrature")

    wall_phi, wall_psi = unit_projected_wall_maxwellian(
        wall_temperature, quadrature
    )
    outgoing_mask = (~incoming_mask) & (np.abs(normal_velocity) > 0.0)
    outgoing_flux = np.sum(
        normal_velocity * outgoing_phi * outgoing_mask * quadrature.weight,
        axis=-1,
    )
    incoming_unit_flux = np.sum(
        normal_velocity * wall_phi * incoming_mask * quadrature.weight,
        axis=-1,
    )
    scale = -outgoing_flux / np.where(
        np.abs(incoming_unit_flux) > 1.0e-14,
        incoming_unit_flux,
        np.copysign(1.0e-14, incoming_unit_flux + 1.0e-300),
    )
    return (
        scale[..., None] * wall_phi,
        scale[..., None] * wall_psi,
        float(np.max(np.asarray(scale))),
    )


def wall_mass_balance_error(
    outgoing_phi: np.ndarray,
    incoming_phi: np.ndarray,
    normal_velocity: np.ndarray,
    incoming_mask: np.ndarray,
    quadrature: PolarQuadrature,
) -> float:
    boundary_phi = np.where(incoming_mask, incoming_phi, outgoing_phi)
    net = np.sum(
        normal_velocity * boundary_phi * quadrature.weight,
        axis=-1,
    )
    scale = np.maximum(
        np.sum(
            np.abs(normal_velocity * boundary_phi) * quadrature.weight,
            axis=-1,
        ),
        1.0e-14,
    )
    return float(np.max(np.abs(net) / scale))


REPRESENTATIVE_STATES = (
    {
        "name": "hot_weak_flow",
        "rho": 1.0, "u": 0.10, "v": -0.05, "T": 1.0,
        "qx": 0.020, "qy": -0.010,
    },
    {
        "name": "intermediate",
        "rho": 0.9, "u": -0.06, "v": 0.02, "T": 0.35,
        "qx": 0.005, "qy": -0.003,
    },
    {
        "name": "cold_weak_flow",
        "rho": 1.1, "u": 0.04, "v": -0.03, "T": 0.10,
        "qx": 8.0e-4, "qy": -4.0e-4,
    },
)


def _state_fields(state: dict[str, float | str]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(float(state[key]))
        for key in ("rho", "u", "v", "T", "qx", "qy")
    }


def _moment_error(
    recovered: dict[str, np.ndarray],
    state: dict[str, float | str],
) -> float:
    return float(max(
        abs(float(recovered["rho"]) - float(state["rho"])) / float(state["rho"]),
        abs(float(recovered["u"]) - float(state["u"])),
        abs(float(recovered["v"]) - float(state["v"])),
        abs(float(recovered["T"]) - float(state["T"])) / float(state["T"]),
    ))


def quadrature_state_audit(quadrature: PolarQuadrature) -> dict[str, object]:
    rows = []
    for state in REPRESENTATIVE_STATES:
        fields = _state_fields(state)
        phi_m, psi_m = projected_maxwellian(
            fields["rho"], fields["u"], fields["v"], fields["T"], quadrature
        )
        maxwellian_moments = projected_macroscopic(phi_m, psi_m, quadrature)
        phi_s, psi_s, clipping = projected_shakhov_equilibrium(
            fields, quadrature
        )
        shakhov_moments = projected_macroscopic(phi_s, psi_s, quadrature)
        expected_q = (1.0 - STAGE41_PRANDTL) * np.array(
            [float(state["qx"]), float(state["qy"])]
        )
        recovered_q = np.array([
            float(shakhov_moments["qx"]),
            float(shakhov_moments["qy"]),
        ])
        q_scale = max(float(np.linalg.norm(expected_q)), 1.0e-14)
        rows.append({
            "state": state["name"],
            "maxwellian_moment_error": _moment_error(maxwellian_moments, state),
            "shakhov_invariant_error": _moment_error(shakhov_moments, state),
            "shakhov_heat_flux_closure_error": float(
                np.linalg.norm(recovered_q - expected_q) / q_scale
            ),
            "expected_equilibrium_q": expected_q.tolist(),
            "recovered_equilibrium_q": recovered_q.tolist(),
            "phi_clipped_weight_fraction": float(
                clipping["phi_clipped_weight_fraction"]
            ),
            "psi_clipped_weight_fraction": float(
                clipping["psi_clipped_weight_fraction"]
            ),
            "minimum_raw_phi_factor": float(
                clipping["minimum_raw_phi_factor"]
            ),
            "minimum_raw_psi_factor": float(
                clipping["minimum_raw_psi_factor"]
            ),
        })
    return {
        "radial_nodes": quadrature.radial_nodes,
        "angular_nodes": quadrature.angular_nodes,
        "point_count": quadrature.point_count,
        "rows": rows,
        "max_maxwellian_moment_error": max(
            row["maxwellian_moment_error"] for row in rows
        ),
        "max_shakhov_invariant_error": max(
            row["shakhov_invariant_error"] for row in rows
        ),
        "max_shakhov_heat_flux_closure_error": max(
            row["shakhov_heat_flux_closure_error"] for row in rows
        ),
        "max_phi_clipped_weight_fraction": max(
            row["phi_clipped_weight_fraction"] for row in rows
        ),
        "max_psi_clipped_weight_fraction": max(
            row["psi_clipped_weight_fraction"] for row in rows
        ),
    }


def diffuse_wall_audit(quadrature: PolarQuadrature) -> dict[str, float]:
    outgoing_phi, outgoing_psi = projected_maxwellian(
        np.asarray(1.1), np.asarray(-0.08), np.asarray(0.03),
        np.asarray(0.6), quadrature,
    )
    incoming_mask = quadrature.vx > 0.0
    incoming_phi, incoming_psi, scale = diffuse_wall_incoming(
        outgoing_phi, outgoing_psi, quadrature.vx, incoming_mask,
        0.2, quadrature,
    )
    return {
        "incoming_density_scale": scale,
        "mass_balance_relative_error": wall_mass_balance_error(
            outgoing_phi, incoming_phi, quadrature.vx,
            incoming_mask, quadrature,
        ),
        "minimum_incoming_phi": float(np.min(incoming_phi)),
        "minimum_incoming_psi": float(np.min(incoming_psi)),
    }


def homogeneous_relaxation_audit(
    quadrature: PolarQuadrature,
) -> dict[str, float]:
    phi_m, psi_m = projected_maxwellian(
        np.asarray(1.0), np.asarray(0.03), np.asarray(-0.02),
        np.asarray(0.5), quadrature,
    )
    anisotropy = np.cos(2.0 * quadrature.angle) * (
        quadrature.radius**2 / (1.0 + quadrature.radius**2)
    )
    phi = phi_m * (1.0 + 0.20 * anisotropy)
    psi = psi_m * (1.0 - 0.10 * anisotropy)
    phi *= 1.0 / np.sum(phi * quadrature.weight)
    before = projected_macroscopic(phi, psi, quadrature)
    phi_eq, psi_eq, _ = projected_shakhov_equilibrium(before, quadrature)
    fraction = 0.25
    phi_after = phi + fraction * (phi_eq - phi)
    psi_after = psi + fraction * (psi_eq - psi)
    after = projected_macroscopic(phi_after, psi_after, quadrature)

    before_vector = np.array([
        float(before["rho"]),
        float(before["rho"] * before["u"]),
        float(before["rho"] * before["v"]),
        float(before["total_internal_moment"]),
    ])
    after_vector = np.array([
        float(after["rho"]),
        float(after["rho"] * after["u"]),
        float(after["rho"] * after["v"]),
        float(after["total_internal_moment"]),
    ])
    normalization = np.array([
        max(abs(before_vector[0]), 1.0e-14),
        1.0,
        1.0,
        max(abs(before_vector[3]), 1.0e-14),
    ])
    relative_change = np.abs(after_vector - before_vector) / normalization
    return {
        "density_change": float(relative_change[0]),
        "x_momentum_change": float(relative_change[1]),
        "y_momentum_change": float(relative_change[2]),
        "internal_energy_change": float(relative_change[3]),
        "maximum_conserved_moment_change": float(np.max(relative_change)),
        "heat_flux_norm_before": float(math.hypot(
            float(before["qx"]), float(before["qy"])
        )),
        "heat_flux_norm_after": float(math.hypot(
            float(after["qx"]), float(after["qy"])
        )),
    }


def stage41_decision(
    coarse: dict[str, object],
    fine: dict[str, object],
    wall: dict[str, float],
    relaxation: dict[str, float],
) -> str:
    trend_ok = (
        float(fine["max_maxwellian_moment_error"])
        < float(coarse["max_maxwellian_moment_error"])
        and float(fine["max_shakhov_invariant_error"])
        < float(coarse["max_shakhov_invariant_error"])
        and float(fine["max_shakhov_heat_flux_closure_error"])
        < float(coarse["max_shakhov_heat_flux_closure_error"])
    )
    fine_ok = (
        float(fine["max_maxwellian_moment_error"]) <= 1.0e-6
        and float(fine["max_shakhov_invariant_error"]) <= 1.0e-6
        and float(fine["max_shakhov_heat_flux_closure_error"]) <= 5.0e-4
    )
    wall_ok = wall["mass_balance_relative_error"] <= 1.0e-12
    relaxation_ok = relaxation["maximum_conserved_moment_change"] <= 1.0e-6
    if trend_ok and fine_ok and wall_ok and relaxation_ok:
        return "projected_polar_operators_pass_stage42_heated_cavity_pilot"
    return "projected_polar_operator_blocker"


def run_self_tests() -> dict[str, float | bool]:
    fine = mapped_polar_quadrature(*STAGE41_FINE_RULE)
    phi, psi = projected_maxwellian(
        np.asarray(1.0), np.asarray(0.1), np.asarray(-0.05),
        np.asarray(1.0), fine,
    )
    moments = projected_macroscopic(phi, psi, fine)
    assert abs(float(moments["rho"]) - 1.0) < 1.0e-12
    assert abs(float(moments["u"]) - 0.1) < 1.0e-6
    assert abs(float(moments["v"]) + 0.05) < 1.0e-6
    assert abs(float(moments["T"]) - 1.0) < 1.0e-6

    fields = {
        "rho": np.asarray(1.0), "u": np.asarray(0.1),
        "v": np.asarray(-0.05), "T": np.asarray(1.0),
        "qx": np.asarray(0.02), "qy": np.asarray(-0.01),
    }
    phi_s, psi_s, _ = projected_shakhov_equilibrium(fields, fine)
    shakhov = projected_macroscopic(phi_s, psi_s, fine)
    expected = (1.0 - STAGE41_PRANDTL) * np.array([0.02, -0.01])
    recovered = np.array([float(shakhov["qx"]), float(shakhov["qy"])])
    closure_error = float(
        np.linalg.norm(recovered - expected) / np.linalg.norm(expected)
    )
    assert closure_error < 5.0e-4

    wall = diffuse_wall_audit(fine)
    assert wall["mass_balance_relative_error"] < 1.0e-12
    relaxation = homogeneous_relaxation_audit(fine)
    assert relaxation["maximum_conserved_moment_change"] < 1.0e-6
    return {
        "passed": True,
        "shakhov_heat_flux_closure_error": closure_error,
        "wall_mass_balance_relative_error": wall[
            "mass_balance_relative_error"
        ],
        "relaxation_maximum_conserved_moment_change": relaxation[
            "maximum_conserved_moment_change"
        ],
    }


def run_stage41(output_dir: str | Path) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coarse_rule = mapped_polar_quadrature(*STAGE41_COARSE_RULE)
    fine_rule = mapped_polar_quadrature(*STAGE41_FINE_RULE)
    coarse = quadrature_state_audit(coarse_rule)
    fine = quadrature_state_audit(fine_rule)
    wall = diffuse_wall_audit(fine_rule)
    relaxation = homogeneous_relaxation_audit(fine_rule)
    self_tests = run_self_tests()
    decision = stage41_decision(coarse, fine, wall, relaxation)

    summary = {
        "stage": 41,
        "description": (
            "Projected phi/psi polar-velocity operator audit before any heated-cavity "
            "claim or source-resolution reproduction"
        ),
        "retained_stage40_endpoint": STAGE40_COMPLETED_ENDPOINT,
        "source_architecture": SOURCE_ARCHITECTURE,
        "configuration": {
            "coarse_rule": {
                "radial_nodes": STAGE41_COARSE_RULE[0],
                "angular_nodes": STAGE41_COARSE_RULE[1],
                "point_count": int(np.prod(STAGE41_COARSE_RULE)),
            },
            "fine_rule": {
                "radial_nodes": STAGE41_FINE_RULE[0],
                "angular_nodes": STAGE41_FINE_RULE[1],
                "point_count": int(np.prod(STAGE41_FINE_RULE)),
            },
            "radial_mapping": "r=s*(1+x)/(1-x)",
            "radial_scale": STAGE41_RADIAL_SCALE,
            "angular_rule": "periodic_trapezoidal",
            "prandtl": STAGE41_PRANDTL,
            "shakhov_correction_floor": STAGE41_CORRECTION_FLOOR,
            "physical_parameter_retuning": False,
            "heated_cavity_solved": False,
        },
        "projection_formulas": {
            "phi": "integral f dc_z",
            "psi": "integral c_z^2 f dc_z",
            "phi_shakhov_polynomial_constant": -4.0,
            "psi_shakhov_polynomial_constant": -2.0,
            "expected_equilibrium_heat_flux": "(1-Pr)*q",
        },
        "coarse_quadrature_audit": coarse,
        "fine_quadrature_audit": fine,
        "diffuse_wall_audit": wall,
        "homogeneous_relaxation_audit": relaxation,
        "self_tests": self_tests,
        "source_to_stage41_velocity_point_ratio": (
            SOURCE_ARCHITECTURE["velocity_vector_count"] / fine_rule.point_count
        ),
        "decision": decision,
        "interpretation_guard": (
            "Stage 41 verifies the projected-distribution formulas, mapped polar "
            "quadrature convergence, diffuse-wall mass balance, and homogeneous "
            "collision conservation. It does not solve the heated cavity, does not "
            "match the source 400x400/32000-vector resolution, and is not external "
            "validation of the Table 3 wall velocity."
        ),
        "scientific_conclusion": (
            "A source-architecture operator audit is the required precursor to an "
            "expensive projected-polar cavity calculation. Positive and negative "
            "operator findings are retained. No Knudsen number, Prandtl number, "
            "collision parameter, wall condition, or failed transport parameter is "
            "retuned."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "quadrature_and_operator_samples.npz",
        fine_vx=fine_rule.vx,
        fine_vy=fine_rule.vy,
        fine_weight=fine_rule.weight,
        fine_radius=fine_rule.radius,
        fine_angle=fine_rule.angle,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="results/stage41_projected_polar_operator_audit",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(run_self_tests(), indent=2))
        return
    print(json.dumps(run_stage41(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
