"""Traceable interior-node grid transfer for R13 restart guesses."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def interior_coordinates(nodes: int) -> np.ndarray:
    if nodes < 3:
        raise ValueError("at least three interior nodes are required")
    return np.arange(1, nodes + 1, dtype=float) / float(nodes + 1)


def interpolate_interior_nodal_state(
    state: np.ndarray,
    *,
    target_nx: int,
    target_ny: int,
    rho_index: int = 0,
    theta_index: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Linearly transfer an interior nodal state to another tensor grid.

    The source and target nodes use ``i/(N+1)``.  Target nodes immediately
    outside the source interior-node interval are linearly extrapolated by
    less than one source spacing.  The returned array is only an initial
    guess; acceptance remains the responsibility of the nonlinear solver.
    """

    source = np.asarray(state, dtype=float)
    if source.ndim != 3:
        raise ValueError("state must have shape (ny, nx, nvar)")
    source_ny, source_nx, variables = source.shape
    if variables < 4:
        raise ValueError("state does not contain the required R13 fields")
    if not np.isfinite(source).all():
        raise ValueError("source state contains non-finite data")
    if target_nx < 3 or target_ny < 3:
        raise ValueError("target grid is too small")

    source_x = interior_coordinates(source_nx)
    source_y = interior_coordinates(source_ny)
    target_x = interior_coordinates(target_nx)
    target_y = interior_coordinates(target_ny)
    target_xx, target_yy = np.meshgrid(target_x, target_y)
    points = np.column_stack((target_yy.ravel(), target_xx.ravel()))
    transferred = np.empty((target_ny, target_nx, variables), dtype=float)

    for variable in range(variables):
        interpolator = RegularGridInterpolator(
            (source_y, source_x),
            source[..., variable],
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        transferred[..., variable] = interpolator(points).reshape(
            target_ny, target_nx
        )

    if not np.isfinite(transferred).all():
        raise ValueError("grid transfer produced non-finite data")
    rho_min = float(np.min(transferred[..., rho_index]))
    theta_min = float(np.min(transferred[..., theta_index]))
    if rho_min <= 0.0 or theta_min <= 0.0:
        raise ValueError("grid transfer violated rho/theta positivity")

    metadata: dict[str, Any] = {
        "applied": True,
        "semantics": "initial_guess_only_not_an_accepted_solution",
        "method": "tensor_product_linear_on_interior_nodes",
        "source_shape": list(source.shape),
        "target_shape": list(transferred.shape),
        "source_coordinate_rule": "i/(N+1)",
        "target_coordinate_rule": "i/(N+1)",
        "bounded_extrapolation_near_walls": bool(
            target_x[0] < source_x[0]
            or target_x[-1] > source_x[-1]
            or target_y[0] < source_y[0]
            or target_y[-1] > source_y[-1]
        ),
        "finite": True,
        "rho_min": rho_min,
        "theta_min": theta_min,
    }
    return transferred, metadata
