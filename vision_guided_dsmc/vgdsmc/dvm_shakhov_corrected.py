from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import json
import numpy as np

from .dvm_shakhov import (
    ShakhovReferenceConfig,
    _discrete_maxwellian,
    _macroscopic,
    _velocity_grid,
    solve_shakhov_reference as _solve_raw,
)


@lru_cache(maxsize=32)
def _temperature_response(nv: int, velocity_extent: float) -> tuple[np.ndarray, np.ndarray]:
    """Map Maxwellian parameter temperature to its measured discrete moment.

    A low-order velocity quadrature does not reproduce the continuous Maxwellian
    second moment exactly.  This monotone lookup is used to reconstruct the
    physical temperature represented by the discrete distribution.
    """
    cfg = ShakhovReferenceConfig(nv=nv, velocity_extent=velocity_extent)
    vx, vy, vz, dv = _velocity_grid(cfg)
    parameter = np.geomspace(0.15, 3.0, 512)
    measured = np.empty_like(parameter)
    one = np.ones((1, 1))
    zero = np.zeros((1, 1))
    for index, theta in enumerate(parameter):
        distribution = _discrete_maxwellian(
            one, zero, zero, zero, np.full((1, 1), theta), vx, vy, vz, dv
        )
        measured[index] = float(_macroscopic(distribution, vx, vy, vz, dv)["T"][0, 0])
    order = np.argsort(measured)
    measured = measured[order]
    parameter = parameter[order]
    keep = np.concatenate(([True], np.diff(measured) > 1.0e-12))
    return measured[keep], parameter[keep]


def reconstruct_temperature(
    measured_temperature: np.ndarray,
    cfg: ShakhovReferenceConfig,
) -> np.ndarray:
    measured_curve, parameter_curve = _temperature_response(cfg.nv, cfg.velocity_extent)
    nondimensional = np.asarray(measured_temperature, dtype=np.float64) / cfg.reference_temperature
    reconstructed = np.interp(
        nondimensional,
        measured_curve,
        parameter_curve,
        left=parameter_curve[0],
        right=parameter_curve[-1],
    )
    return reconstructed * cfg.reference_temperature


def solve_shakhov_reference(
    cfg: ShakhovReferenceConfig,
) -> dict[str, np.ndarray | float | int]:
    result = _solve_raw(cfg)
    corrected = dict(result)
    corrected["T_raw_quadrature"] = np.asarray(result["T"], dtype=np.float64)
    corrected["T"] = reconstruct_temperature(corrected["T_raw_quadrature"], cfg)
    return corrected


def save_shakhov_reference(output_path: str | Path, cfg: ShakhovReferenceConfig) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = solve_shakhov_reference(cfg)
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    arrays["iterations"] = np.int64(result["iterations"])
    arrays["dt"] = np.float64(result["dt"])
    np.savez_compressed(output_path, **arrays)
    metadata = {
        "model": "deterministic_2d_space_3d_velocity_shakhov_dvm",
        "temperature_reconstruction": "inverse_discrete_Maxwellian_second_moment",
        "status": "reference_pilot",
        "config": asdict(cfg),
        "iterations": int(result["iterations"]),
        "final_residual": (
            float(result["residual_history"][-1])
            if len(result["residual_history"])
            else None
        ),
        "field_contract": ["T", "rho", "u", "v", "qx", "qy"],
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path


__all__ = [
    "ShakhovReferenceConfig",
    "reconstruct_temperature",
    "solve_shakhov_reference",
    "save_shakhov_reference",
]
