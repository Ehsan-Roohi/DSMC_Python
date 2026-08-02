from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import numpy as np

from .bottom_heated_benchmark import _bottom_boundary_heat_flux
from .dvm_shakhov import ShakhovReferenceConfig, _macroscopic, _velocity_grid
from .dvm_shakhov_corrected import (
    _discrete_maxwellian,
    _shakhov_equilibrium,
    _unit_wall_maxwellian,
)


# Source-table coordinates are stored explicitly rather than generated with
# np.arange, whose accumulated binary rounding prevents bitwise transcription
# checks even though the intended decimal points are the same.
TABLE3_Y = np.array(
    [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
    dtype=np.float64,
)
TABLE3_UY_RATIO_0P1 = {
    0.1: np.array([1.7e-3, 8.8e-4, -1.5e-4, -1.1e-3, -1.8e-3,
                   -2.5e-3, -2.8e-3, -2.8e-3, -2.0e-3, -5.9e-5]),
    1.0: np.array([5.4e-3, 5.8e-3, 5.6e-3, 5.0e-3, 4.4e-3,
                   3.6e-3, 2.8e-3, 1.9e-3, 1.2e-3, 7.7e-4]),
    10.0: np.array([1.3e-3, 1.3e-3, 1.2e-3, 1.1e-3, 9.2e-4,
                    7.4e-4, 5.7e-4, 3.9e-4, 2.5e-4, 1.3e-4]),
}
TABLE6_QAV_RATIO_0P1 = {0.1: 7.20e-2, 1.0: 1.48e-1, 10.0: 1.78e-1}


@dataclass(frozen=True)
class LinearSidewallConfig:
    nx: int = 20
    ny: int = 20
    nv: int = 17
    velocity_extent: float = 5.0
    kn0: float = 0.1
    cold_hot_ratio: float = 0.1
    viscosity_exponent: float = 0.5
    prandtl: float = 2.0 / 3.0
    max_steps: int = 6000
    cfl: float = 0.30
    tolerance: float = 2.0e-5
    check_interval: int = 100
    minimum_steps: int = 1500
    positivity_floor: float = 1.0e-30

    @property
    def hot_temperature(self) -> float:
        return 1.0

    @property
    def cold_temperature(self) -> float:
        return self.cold_hot_ratio


def paper_relaxation_scale(kn0: float) -> float:
    """Return 2 Kn0/sqrt(pi), obtained from Eq. (1) and Eq. (6)."""
    if kn0 <= 0.0:
        raise ValueError("kn0 must be positive")
    return 2.0 * float(kn0) / math.sqrt(math.pi)


def sidewall_temperature_profile(cfg: LinearSidewallConfig) -> np.ndarray:
    y = (np.arange(cfg.ny, dtype=np.float64) + 0.5) / cfg.ny
    return cfg.hot_temperature - (
        cfg.hot_temperature - cfg.cold_temperature
    ) * y


def local_relaxation_time(
    density: np.ndarray,
    temperature: np.ndarray,
    cfg: LinearSidewallConfig,
) -> np.ndarray:
    density = np.maximum(np.asarray(density, dtype=np.float64), 1.0e-12)
    temperature = np.maximum(np.asarray(temperature, dtype=np.float64), 1.0e-10)
    return (
        paper_relaxation_scale(cfg.kn0)
        * temperature ** (cfg.viscosity_exponent - 1.0)
        / density
    )


def _profile_wall_incoming(
    outgoing_distribution: np.ndarray,
    normal_velocity: np.ndarray,
    incoming_mask: np.ndarray,
    outgoing_mask: np.ndarray,
    wall_maxwellians: np.ndarray,
    dv: float,
) -> np.ndarray:
    measure = dv**3
    outgoing_flux = np.sum(
        normal_velocity[None] * outgoing_distribution * outgoing_mask[None],
        axis=(-3, -2, -1),
    ) * measure
    incoming_unit = np.sum(
        normal_velocity[None] * wall_maxwellians * incoming_mask[None],
        axis=(-3, -2, -1),
    ) * measure
    scale = -outgoing_flux / np.maximum(incoming_unit, 1.0e-14)
    return scale[:, None, None, None] * wall_maxwellians


def _wall_incoming(
    distribution: np.ndarray,
    cfg: LinearSidewallConfig,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    side_temperature = sidewall_temperature_profile(cfg)
    side_wall = np.stack([
        _unit_wall_maxwellian(float(t), vx, vy, vz, dv)
        for t in side_temperature
    ])
    bottom_wall = np.repeat(
        _unit_wall_maxwellian(cfg.hot_temperature, vx, vy, vz, dv)[None],
        cfg.nx,
        axis=0,
    )
    top_wall = np.repeat(
        _unit_wall_maxwellian(cfg.cold_temperature, vx, vy, vz, dv)[None],
        cfg.nx,
        axis=0,
    )
    px, nx = vx > 0.0, vx < 0.0
    py, ny = vy > 0.0, vy < 0.0
    left = _profile_wall_incoming(distribution[:, 0], vx, px, nx, side_wall, dv)
    right = _profile_wall_incoming(distribution[:, -1], -vx, nx, px, side_wall, dv)
    bottom = _profile_wall_incoming(distribution[0], vy, py, ny, bottom_wall, dv)
    top = _profile_wall_incoming(distribution[-1], -vy, ny, py, top_wall, dv)
    return left, right, bottom, top


def _left_wall_tangential_velocity(
    distribution: np.ndarray,
    left_incoming: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    dv: float,
) -> np.ndarray:
    boundary = np.where((vx > 0.0)[None], left_incoming, distribution[:, 0])
    fields = _macroscopic(boundary[:, None], vx, vy, vz, dv)
    return np.asarray(fields["v"][:, 0]) / math.sqrt(2.0)


def solve_linear_sidewall_case(cfg: LinearSidewallConfig) -> dict[str, object]:
    if cfg.nv % 2 == 0:
        raise ValueError("an odd velocity count is required")
    if not 0.0 < cfg.cold_hot_ratio < 1.0:
        raise ValueError("cold_hot_ratio must lie in (0,1)")
    velocity_cfg = ShakhovReferenceConfig(
        nx=cfg.nx, ny=cfg.ny, nv=cfg.nv, velocity_extent=cfg.velocity_extent
    )
    vx, vy, vz, dv = _velocity_grid(velocity_cfg)
    rho = np.ones((cfg.ny, cfg.nx))
    zero = np.zeros_like(rho)
    initial_temperature = np.repeat(
        sidewall_temperature_profile(cfg)[:, None], cfg.nx, axis=1
    )
    distribution = _discrete_maxwellian(
        rho, zero, zero, zero, initial_temperature, vx, vy, vz, dv
    )
    dx, dy = 1.0 / cfg.nx, 1.0 / cfg.ny
    maximum_speed = max(float(np.max(np.abs(vx))), float(np.max(np.abs(vy))))
    dt = cfg.cfl * min(dx / maximum_speed, dy / maximum_speed)
    positive_x = (vx > 0.0)[None, None]
    positive_y = (vy > 0.0)[None, None]
    previous = _macroscopic(distribution, vx, vy, vz, dv)
    previous_T = previous["T"].copy()
    previous_u = np.stack([previous["u"], previous["v"]], axis=-1)
    previous_q = np.stack([previous["qx"], previous["qy"]], axis=-1)
    residual_history: list[float] = []
    converged = False
    left = right = bottom = top = None

    for step in range(cfg.max_steps):
        left, right, bottom, top = _wall_incoming(
            distribution, cfg, vx, vy, vz, dv
        )
        ln = np.empty_like(distribution); rn = np.empty_like(distribution)
        bn = np.empty_like(distribution); tn = np.empty_like(distribution)
        ln[:, 1:] = distribution[:, :-1]; ln[:, 0] = left
        rn[:, :-1] = distribution[:, 1:]; rn[:, -1] = right
        bn[1:] = distribution[:-1]; bn[0] = bottom
        tn[:-1] = distribution[1:]; tn[-1] = top
        dfdx = np.where(positive_x, (distribution - ln) / dx, (rn - distribution) / dx)
        dfdy = np.where(positive_y, (distribution - bn) / dy, (tn - distribution) / dy)
        transported = np.maximum(
            distribution - dt * (vx[None, None] * dfdx + vy[None, None] * dfdy),
            cfg.positivity_floor,
        )
        fields = _macroscopic(transported, vx, vy, vz, dv)
        equilibrium = _shakhov_equilibrium(fields, vx, vy, vz, dv, cfg.prandtl)
        tau = local_relaxation_time(fields["rho"], fields["T"], cfg)
        fraction = np.minimum(dt / tau, 1.0)[..., None, None, None]
        distribution = np.maximum(
            transported + fraction * (equilibrium - transported),
            cfg.positivity_floor,
        )
        if (step + 1) % cfg.check_interval == 0:
            fields = _macroscopic(distribution, vx, vy, vz, dv)
            velocity = np.stack([fields["u"], fields["v"]], axis=-1)
            heat_flux = np.stack([fields["qx"], fields["qy"]], axis=-1)
            change = max(
                float(np.max(np.abs(fields["T"] - previous_T))),
                float(np.max(np.abs(velocity - previous_u))),
                float(np.max(np.abs(heat_flux - previous_q))),
            )
            residual_history.append(change)
            previous_T = fields["T"].copy(); previous_u = velocity.copy(); previous_q = heat_flux.copy()
            if step + 1 >= cfg.minimum_steps and change < cfg.tolerance:
                converged = True
                break

    assert left is not None and bottom is not None
    fields = _macroscopic(distribution, vx, vy, vz, dv)
    wall_velocity = _left_wall_tangential_velocity(
        distribution, left, vx, vy, vz, dv
    )
    y_centers = (np.arange(cfg.ny, dtype=np.float64) + 0.5) / cfg.ny
    table_velocity = np.interp(TABLE3_Y, y_centers, wall_velocity)
    bottom_heat_flux = _bottom_boundary_heat_flux(
        distribution, bottom, vx, vy, vz, dv
    )
    return {
        "T": np.asarray(fields["T"]),
        "rho": np.asarray(fields["rho"]) / np.mean(fields["rho"]),
        "u": np.asarray(fields["u"]) / math.sqrt(2.0),
        "v": np.asarray(fields["v"]) / math.sqrt(2.0),
        "qx": np.asarray(fields["qx"]) / math.sqrt(2.0),
        "qy": np.asarray(fields["qy"]) / math.sqrt(2.0),
        "left_wall_velocity": wall_velocity,
        "table_y": TABLE3_Y.copy(),
        "table_velocity": table_velocity,
        "bottom_heat_flux": bottom_heat_flux,
        "residual_history": np.asarray(residual_history),
        "iterations": step + 1,
        "converged": converged,
    }


def _relative_rms(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.sqrt(np.mean(reference**2))), 1.0e-14)
    return float(np.sqrt(np.mean((candidate - reference) ** 2)) / denominator)


def run_stage26(output_dir: str | Path, base: LinearSidewallConfig) -> dict[str, object]:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    rows = []; arrays: dict[str, np.ndarray] = {}
    for kn0 in (0.1, 1.0, 10.0):
        cfg = LinearSidewallConfig(**{**base.__dict__, "kn0": kn0})
        result = solve_linear_sidewall_case(cfg)
        predicted_u = np.asarray(result["table_velocity"])
        reference_u = TABLE3_UY_RATIO_0P1[kn0]
        predicted_q = float(np.mean(result["bottom_heat_flux"]))
        reference_q = TABLE6_QAV_RATIO_0P1[kn0]
        key = str(kn0).replace(".", "p")
        rows.append({
            "kn0": kn0,
            "iterations": int(result["iterations"]),
            "converged": bool(result["converged"]),
            "final_change": float(np.asarray(result["residual_history"])[-1]),
            "predicted_qav": predicted_q,
            "literature_qav": reference_q,
            "qav_relative_error": abs(predicted_q - reference_q) / reference_q,
            "wall_velocity_relative_rms": _relative_rms(predicted_u, reference_u),
            "wall_velocity_sign_agreement": float(np.mean(np.sign(predicted_u) == np.sign(reference_u))),
        })
        for name in ("T", "rho", "u", "v", "qx", "qy", "left_wall_velocity", "table_velocity", "bottom_heat_flux", "residual_history"):
            arrays[f"{name}_kn{key}"] = np.asarray(result[name])
    summary = {
        "stage": 26,
        "description": "Quantitative validation for the journal linear-sidewall cavity using Tables 3 and 6",
        "configuration": {
            "grid": [base.nx, base.ny], "nv": base.nv,
            "velocity_extent": base.velocity_extent,
            "cold_hot_ratio": base.cold_hot_ratio,
            "wall_model": "bottom hot, top cold, side walls linear hot-to-cold",
            "relaxation_scale": "2*Kn0/sqrt(pi)",
            "heat_flux_scale": "Q/(P0*v0), v0=sqrt(2*k*T0/m)",
        },
        "literature": {
            "table3_y": TABLE3_Y.tolist(),
            "table6_qav": {str(k): v for k, v in TABLE6_QAV_RATIO_0P1.items()},
        },
        "rows": rows,
        "mean_qav_relative_error": float(np.mean([row["qav_relative_error"] for row in rows])),
        "mean_velocity_relative_rms": float(np.mean([row["wall_velocity_relative_rms"] for row in rows])),
        "all_converged": bool(all(row["converged"] for row in rows)),
        "interpretation_guard": "This is quantitative cross-literature validation. Failure is reported and not tuned away.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(output_dir / "fields_and_profiles.npz", **arrays)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 26 linear-sidewall quantitative validation")
    parser.add_argument("--output-dir", default="outputs/stage26_linear_sidewall")
    parser.add_argument("--nx", type=int, default=20); parser.add_argument("--ny", type=int, default=20)
    parser.add_argument("--nv", type=int, default=17); parser.add_argument("--velocity-extent", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=6000); parser.add_argument("--tolerance", type=float, default=2e-5)
    args = parser.parse_args()
    summary = run_stage26(args.output_dir, LinearSidewallConfig(
        nx=args.nx, ny=args.ny, nv=args.nv, velocity_extent=args.velocity_extent,
        max_steps=args.max_steps, tolerance=args.tolerance,
    ))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
