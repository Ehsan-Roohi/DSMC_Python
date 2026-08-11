from __future__ import annotations

import numpy as np

from r13_grid_transfer import interior_coordinates, interpolate_interior_nodal_state


def test_linear_fields_transfer_exactly_between_interior_grids() -> None:
    source_x = interior_coordinates(6)
    source_y = interior_coordinates(5)
    xx, yy = np.meshgrid(source_x, source_y)
    state = np.empty((5, 6, 4))
    state[..., 0] = 1.0 + 0.1 * xx - 0.05 * yy
    state[..., 1] = xx + 2.0 * yy
    state[..., 2] = -0.3 * xx + 0.4 * yy
    state[..., 3] = 0.9 + 0.02 * xx + 0.03 * yy

    transferred, metadata = interpolate_interior_nodal_state(
        state, target_nx=9, target_ny=8
    )
    target_x = interior_coordinates(9)
    target_y = interior_coordinates(8)
    target_xx, target_yy = np.meshgrid(target_x, target_y)

    assert transferred.shape == (8, 9, 4)
    assert np.allclose(
        transferred[..., 0], 1.0 + 0.1 * target_xx - 0.05 * target_yy
    )
    assert np.allclose(transferred[..., 1], target_xx + 2.0 * target_yy)
    assert np.allclose(transferred[..., 2], -0.3 * target_xx + 0.4 * target_yy)
    assert np.allclose(
        transferred[..., 3], 0.9 + 0.02 * target_xx + 0.03 * target_yy
    )
    assert metadata["applied"] is True
    assert metadata["bounded_extrapolation_near_walls"] is True
    assert metadata["semantics"] == "initial_guess_only_not_an_accepted_solution"


def test_transfer_rejects_nonfinite_and_nonpositive_thermodynamic_fields() -> None:
    valid = np.zeros((5, 5, 4))
    valid[..., 0] = 1.0
    valid[..., 3] = 1.0

    nonfinite = valid.copy()
    nonfinite[2, 2, 1] = np.nan
    try:
        interpolate_interior_nodal_state(nonfinite, target_nx=7, target_ny=7)
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("non-finite source was accepted")

    nonpositive = valid.copy()
    nonpositive[..., 0] = -1.0
    try:
        interpolate_interior_nodal_state(nonpositive, target_nx=7, target_ny=7)
    except ValueError as error:
        assert "positivity" in str(error)
    else:
        raise AssertionError("nonpositive density was accepted")
