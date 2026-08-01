from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import numpy as np


TEMPERATURES = (0.1, 0.5, 1.0)
TANGENTIAL_SHIFTS = (-0.005, 0.005)


@dataclass(frozen=True)
class VelocityQuadrature:
    name: str
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    weight: np.ndarray
    family: str

    @property
    def point_count(self) -> int:
        return int(self.weight.size)


def cartesian_midpoint(order: int, extent: float = 5.0) -> VelocityQuadrature:
    if order < 3 or order % 2 == 0:
        raise ValueError("Cartesian midpoint order must be odd and at least three")
    if extent <= 0.0:
        raise ValueError("extent must be positive")
    dv = 2.0 * extent / order
    values = -extent + dv * (np.arange(order, dtype=np.float64) + 0.5)
    vx, vy, vz = np.meshgrid(values, values, values, indexing="ij")
    weight = np.full(vx.size, dv**3, dtype=np.float64)
    return VelocityQuadrature(
        name=f"cartesian_midpoint_nv{order}",
        vx=vx.ravel(),
        vy=vy.ravel(),
        vz=vz.ravel(),
        weight=weight,
        family="cartesian_midpoint",
    )


def tensor_gauss_hermite(order: int, reference_temperature: float = 1.0) -> VelocityQuadrature:
    """Fixed tensor Gauss-Hermite rule converted to Lebesgue integration weights."""
    if order < 4:
        raise ValueError("Gauss-Hermite order must be at least four")
    if reference_temperature <= 0.0:
        raise ValueError("reference_temperature must be positive")
    nodes, hermite_weights = np.polynomial.hermite.hermgauss(order)
    scale = math.sqrt(2.0 * reference_temperature)
    values = scale * nodes
    one_d_weight = scale * hermite_weights * np.exp(nodes * nodes)
    vx, vy, vz = np.meshgrid(values, values, values, indexing="ij")
    wx, wy, wz = np.meshgrid(one_d_weight, one_d_weight, one_d_weight, indexing="ij")
    return VelocityQuadrature(
        name=f"gauss_hermite_order{order}_tref{reference_temperature:g}",
        vx=vx.ravel(),
        vy=vy.ravel(),
        vz=vz.ravel(),
        weight=(wx * wy * wz).ravel(),
        family="tensor_gauss_hermite",
    )


def spherical_product(
    radial_order: int,
    polar_order: int,
    azimuthal_order: int,
    radius: float = 5.0,
    label: str = "spherical_product",
) -> VelocityQuadrature:
    """Positive radial-angular quadrature in spherical velocity coordinates."""
    if min(radial_order, polar_order, azimuthal_order) < 4:
        raise ValueError("all spherical-product orders must be at least four")
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    radial_nodes, radial_weights = np.polynomial.legendre.leggauss(radial_order)
    radii = 0.5 * radius * (radial_nodes + 1.0)
    wr = 0.5 * radius * radial_weights * radii * radii
    mu, wmu = np.polynomial.legendre.leggauss(polar_order)
    phi = 2.0 * math.pi * (np.arange(azimuthal_order, dtype=np.float64) + 0.5) / azimuthal_order
    wphi = 2.0 * math.pi / azimuthal_order
    r3, mu3, phi3 = np.meshgrid(radii, mu, phi, indexing="ij")
    wr3, wmu3, _ = np.meshgrid(wr, wmu, phi, indexing="ij")
    transverse = np.sqrt(np.maximum(1.0 - mu3 * mu3, 0.0))
    vx = r3 * mu3
    vy = r3 * transverse * np.cos(phi3)
    vz = r3 * transverse * np.sin(phi3)
    weight = wr3 * wmu3 * wphi
    return VelocityQuadrature(
        name=label,
        vx=vx.ravel(),
        vy=vy.ravel(),
        vz=vz.ravel(),
        weight=weight.ravel(),
        family="spherical_product",
    )


def maxwellian(
    quadrature: VelocityQuadrature,
    temperature: float,
    tangential_velocity: float = 0.0,
) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    speed2 = (
        quadrature.vx * quadrature.vx
        + (quadrature.vy - tangential_velocity) ** 2
        + quadrature.vz * quadrature.vz
    )
    return np.exp(-speed2 / (2.0 * temperature)) / (2.0 * math.pi * temperature) ** 1.5


def integrate(quadrature: VelocityQuadrature, values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != quadrature.weight.shape:
        raise ValueError("values and quadrature weights must have matching shapes")
    return float(np.sum(quadrature.weight * values))


def _relative_error(value: float, reference: float, floor: float = 1.0e-14) -> float:
    return abs(value - reference) / max(abs(reference), floor)


def audit_condition(
    quadrature: VelocityQuadrature,
    temperature: float,
) -> dict[str, float]:
    f = maxwellian(quadrature, temperature)
    rho = integrate(quadrature, f)
    ux = integrate(quadrature, f * quadrature.vx) / max(rho, 1.0e-300)
    uy = integrate(quadrature, f * quadrature.vy) / max(rho, 1.0e-300)
    uz = integrate(quadrature, f * quadrature.vz) / max(rho, 1.0e-300)
    cx = quadrature.vx - ux
    cy = quadrature.vy - uy
    cz = quadrature.vz - uz
    c2 = cx * cx + cy * cy + cz * cz
    measured_temperature = integrate(quadrature, f * c2) / (3.0 * max(rho, 1.0e-300))
    fourth = np.array(
        [
            integrate(quadrature, f * cx**4) / max(rho, 1.0e-300),
            integrate(quadrature, f * cy**4) / max(rho, 1.0e-300),
            integrate(quadrature, f * cz**4) / max(rho, 1.0e-300),
        ]
    )
    mixed = np.array(
        [
            integrate(quadrature, f * cx * cx * cy * cy) / max(rho, 1.0e-300),
            integrate(quadrature, f * cx * cx * cz * cz) / max(rho, 1.0e-300),
            integrate(quadrature, f * cy * cy * cz * cz) / max(rho, 1.0e-300),
        ]
    )
    positive = quadrature.vx > 0.0
    speed2 = quadrature.vx**2 + quadrature.vy**2 + quadrature.vz**2
    mass_flux = integrate(quadrature, f * quadrature.vx * positive)
    energy_flux = integrate(quadrature, 0.5 * f * speed2 * quadrature.vx * positive)
    exact_mass_flux = math.sqrt(temperature / (2.0 * math.pi))
    exact_energy_flux = 2.0 * temperature * exact_mass_flux
    return {
        "temperature": temperature,
        "density_relative_error": _relative_error(rho, 1.0),
        "mean_velocity_absolute_error": float(max(abs(ux), abs(uy), abs(uz))),
        "temperature_relative_error": _relative_error(measured_temperature, temperature),
        "fourth_moment_relative_error": float(
            np.max(np.abs(fourth - 3.0 * temperature**2)) / (3.0 * temperature**2)
        ),
        "mixed_moment_relative_error": float(
            np.max(np.abs(mixed - temperature**2)) / (temperature**2)
        ),
        "fourth_moment_isotropy": float(
            (np.max(fourth) - np.min(fourth)) / max(3.0 * temperature**2, 1.0e-14)
        ),
        "mixed_moment_isotropy": float(
            (np.max(mixed) - np.min(mixed)) / max(temperature**2, 1.0e-14)
        ),
        "half_mass_flux_relative_error": _relative_error(mass_flux, exact_mass_flux),
        "half_energy_flux_relative_error": _relative_error(energy_flux, exact_energy_flux),
    }


def audit_tangential_response(
    quadrature: VelocityQuadrature,
    temperature: float,
    tangential_velocity: float,
) -> dict[str, float | bool]:
    f = maxwellian(quadrature, temperature, tangential_velocity)
    positive = quadrature.vx > 0.0
    mass_flux_exact = math.sqrt(temperature / (2.0 * math.pi))
    exact = tangential_velocity * mass_flux_exact
    predicted = integrate(
        quadrature,
        f * quadrature.vx * quadrature.vy * positive,
    )
    return {
        "temperature": temperature,
        "tangential_velocity": tangential_velocity,
        "predicted_half_tangential_momentum_flux": predicted,
        "exact_half_tangential_momentum_flux": exact,
        "relative_error": _relative_error(predicted, exact),
        "sign_correct": bool(np.sign(predicted) == np.sign(exact)),
    }


def audit_quadrature(quadrature: VelocityQuadrature) -> dict[str, object]:
    equilibrium_rows = [audit_condition(quadrature, temperature) for temperature in TEMPERATURES]
    shifted_rows = [
        audit_tangential_response(quadrature, temperature, shift)
        for temperature in TEMPERATURES
        for shift in TANGENTIAL_SHIFTS
    ]
    scalar_error_keys = (
        "density_relative_error",
        "mean_velocity_absolute_error",
        "temperature_relative_error",
        "fourth_moment_relative_error",
        "mixed_moment_relative_error",
        "fourth_moment_isotropy",
        "mixed_moment_isotropy",
        "half_mass_flux_relative_error",
        "half_energy_flux_relative_error",
    )
    max_equilibrium_error = max(
        float(row[key]) for row in equilibrium_rows for key in scalar_error_keys
    )
    max_shifted_error = max(float(row["relative_error"]) for row in shifted_rows)
    sign_agreement = float(np.mean([bool(row["sign_correct"]) for row in shifted_rows]))
    cold_row = next(row for row in equilibrium_rows if row["temperature"] == 0.1)
    composite = max(max_equilibrium_error, max_shifted_error)
    return {
        "name": quadrature.name,
        "family": quadrature.family,
        "point_count": quadrature.point_count,
        "positive_weights": bool(np.all(quadrature.weight > 0.0)),
        "weight_sum": float(np.sum(quadrature.weight)),
        "equilibrium_rows": equilibrium_rows,
        "shifted_rows": shifted_rows,
        "max_equilibrium_error": max_equilibrium_error,
        "max_shifted_tangential_error": max_shifted_error,
        "shifted_sign_agreement": sign_agreement,
        "cold_wall_mass_flux_relative_error": float(cold_row["half_mass_flux_relative_error"]),
        "cold_wall_energy_flux_relative_error": float(cold_row["half_energy_flux_relative_error"]),
        "composite_max_error": composite,
    }


def build_stage29_quadratures() -> list[VelocityQuadrature]:
    return [
        cartesian_midpoint(17, 5.0),
        cartesian_midpoint(19, 5.0),
        tensor_gauss_hermite(12, 1.0),
        spherical_product(12, 10, 16, 5.0, "spherical_reduced_r12_mu10_phi16"),
        spherical_product(16, 12, 24, 5.0, "spherical_matched_r16_mu12_phi24"),
    ]


def run_stage29(output_dir: str | Path) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [audit_quadrature(quadrature) for quadrature in build_stage29_quadratures()]
    by_name = {row["name"]: row for row in rows}
    baseline = by_name["cartesian_midpoint_nv19"]
    non_cartesian = [row for row in rows if row["family"] != "cartesian_midpoint"]
    best = min(non_cartesian, key=lambda row: float(row["composite_max_error"]))
    materially_better = bool(
        float(best["composite_max_error"]) <= 0.75 * float(baseline["composite_max_error"])
        and float(best["shifted_sign_agreement"]) >= float(baseline["shifted_sign_agreement"])
    )
    summary = {
        "stage": 29,
        "description": "Fixed-physics reduced and non-Cartesian velocity-quadrature audit",
        "conditions": {
            "temperatures": list(TEMPERATURES),
            "tangential_shifts": list(TANGENTIAL_SHIFTS),
            "velocity_extent": 5.0,
            "no_physical_parameter_retuning": True,
        },
        "schemes": rows,
        "cartesian_baseline": baseline["name"],
        "best_non_cartesian": best["name"],
        "best_non_cartesian_materially_better": materially_better,
        "decision": (
            "integrate_best_non_cartesian_rule_into_reduced_kinetic_solver"
            if materially_better
            else "audit_wall_observable_and_sign_convention_before_solver_integration"
        ),
        "interpretation_guard": (
            "This stage audits quadrature moments and half-space wall observables only. "
            "It does not claim cavity validation and does not tune Knudsen, collision, wall, or normalization parameters."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        output_dir / "quadrature_metrics.npz",
        names=np.asarray([row["name"] for row in rows]),
        point_count=np.asarray([row["point_count"] for row in rows], dtype=np.int64),
        composite_max_error=np.asarray([row["composite_max_error"] for row in rows]),
        cold_mass_flux_error=np.asarray([row["cold_wall_mass_flux_relative_error"] for row in rows]),
        cold_energy_flux_error=np.asarray([row["cold_wall_energy_flux_relative_error"] for row in rows]),
        shifted_sign_agreement=np.asarray([row["shifted_sign_agreement"] for row in rows]),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 29 velocity quadrature audit")
    parser.add_argument("--output-dir", default="outputs/stage29_velocity_quadrature")
    args = parser.parse_args()
    print(json.dumps(run_stage29(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
