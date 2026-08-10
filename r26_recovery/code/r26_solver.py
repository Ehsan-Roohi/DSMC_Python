#!/usr/bin/env python3
"""Stateless nonlinear solver utilities for the private node-grid R26 BVP."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import OptimizeResult, least_squares, root
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import lsmr, splu

from r26_cases import CavityCase
from r26_discretization import R26NodeBVP, ResidualDiagnostics, trapezoidal_node_weights
from r26_fv_backend import fv_absolute_difference_step
from r26_state import NVAR, validate_planar_state


@dataclass(frozen=True)
class LogStateTransform:
    """Logarithmic rho/T coordinates and raw coordinates for all other fields."""

    shape: tuple[int, int, int]
    maximum_log_magnitude: float = 50.0

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or self.shape[-1] != NVAR:
            raise ValueError("transform shape must be (ny,nx,17)")
        if self.maximum_log_magnitude <= 0.0:
            raise ValueError("maximum log magnitude must be positive")

    def encode(self, state: np.ndarray) -> np.ndarray:
        u = validate_planar_state(state)
        if u.shape != self.shape:
            raise ValueError(f"state shape must be {self.shape}")
        encoded = u.copy()
        encoded[..., 0] = np.log(u[..., 0])
        encoded[..., 3] = np.log(u[..., 3])
        return encoded.ravel()

    def decode(self, vector: np.ndarray) -> np.ndarray:
        x = np.asarray(vector, dtype=float)
        if x.shape != (int(np.prod(self.shape)),) or not np.isfinite(x).all():
            raise ValueError("encoded state has incorrect shape or non-finite values")
        encoded = x.reshape(self.shape)
        logs = encoded[..., (0, 3)]
        if np.max(np.abs(logs), initial=0.0) > self.maximum_log_magnitude:
            raise FloatingPointError("rho/T log coordinate exceeded the declared solver domain")
        state = encoded.copy()
        state[..., 0] = np.exp(encoded[..., 0])
        state[..., 3] = np.exp(encoded[..., 3])
        return validate_planar_state(state)

    def least_squares_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(self.shape, -np.inf)
        upper = np.full(self.shape, np.inf)
        lower[..., 0] = lower[..., 3] = -self.maximum_log_magnitude
        upper[..., 0] = upper[..., 3] = self.maximum_log_magnitude
        return lower.ravel(), upper.ravel()


@dataclass(frozen=True)
class SolveOptions:
    method: str = "krylov"
    residual_tolerance: float = 1.0e-9
    held_out_continuity_tolerance: float = 1.0e-7
    max_iterations: int = 200
    max_function_evaluations: int = 5000
    line_search: str = "armijo"
    display: bool = False
    invalid_penalty: float = 1.0e8
    armijo_coefficient: float = 1.0e-4
    minimum_line_search_alpha: float = 2.0**-24
    lsmr_max_iterations: int = 240
    regularization_factors: tuple[float, ...] = (0.0, 1.0e-6, 1.0e-4, 1.0e-2, 1.0)

    def __post_init__(self) -> None:
        if self.method not in {"krylov", "least_squares", "colored_newton"}:
            raise ValueError("method must be krylov, least_squares, or colored_newton")
        if self.residual_tolerance <= 0.0 or self.held_out_continuity_tolerance <= 0.0:
            raise ValueError("solver tolerances must be positive")
        if self.max_iterations < 1 or self.max_function_evaluations < 1:
            raise ValueError("solver iteration limits must be positive")
        if not 0.0 < self.armijo_coefficient < 1.0:
            raise ValueError("Armijo coefficient must lie strictly between zero and one")
        if not 0.0 < self.minimum_line_search_alpha <= 1.0:
            raise ValueError("minimum line-search alpha must lie in (0,1]")
        if self.lsmr_max_iterations < 1:
            raise ValueError("LSMR iteration limit must be positive")
        if not self.regularization_factors:
            raise ValueError("at least one LSMR regularization factor is required")
        if any((not np.isfinite(value)) or value < 0.0 for value in self.regularization_factors):
            raise ValueError("LSMR regularization factors must be finite and nonnegative")


@dataclass(frozen=True)
class R26SolveResult:
    case: CavityCase
    state: np.ndarray
    encoded_state: np.ndarray
    diagnostics: ResidualDiagnostics
    converged: bool
    scipy_success: bool
    message: str
    iterations: int
    function_evaluations: int
    invalid_evaluations: int
    last_invalid_error: str | None
    solver_method: str


class EncodedR26Objective:
    """Pure vector objective plus a finite guard for rejected line-search steps."""

    def __init__(self, problem: R26NodeBVP, transform: LogStateTransform, penalty: float) -> None:
        self.problem = problem
        self.transform = transform
        self.penalty = float(penalty)
        self.invalid_evaluations = 0
        self.last_invalid_error: str | None = None

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        try:
            state = self.transform.decode(vector)
            return self.problem.residual(state)
        except (FloatingPointError, ValueError, OverflowError) as exc:
            self.invalid_evaluations += 1
            self.last_invalid_error = f"{type(exc).__name__}: {exc}"
            x = np.asarray(vector, dtype=float)
            sign = np.where(np.isfinite(x) & (x < 0.0), -1.0, 1.0)
            magnitude = 1.0 + np.minimum(np.nan_to_num(np.abs(x), nan=100.0, posinf=100.0), 100.0)
            return self.penalty * sign * magnitude

    def jvp(self, vector: np.ndarray, direction: np.ndarray, *, relative_step: float = 1.0e-6) -> np.ndarray:
        """Centered finite-difference Jacobian-vector product of the full BVP."""

        x = np.asarray(vector, dtype=float)
        v = np.asarray(direction, dtype=float)
        if x.shape != v.shape or not np.isfinite(x).all() or not np.isfinite(v).all():
            raise ValueError("JVP vectors must have the same finite shape")
        norm_v = np.linalg.norm(v)
        if norm_v == 0.0:
            return np.zeros_like(x)
        step = relative_step * (1.0 + np.linalg.norm(x)) / norm_v
        return (self(x + step * v) - self(x - step * v)) / (2.0 * step)


class EncodedR26MassContinuityObjective:
    """Return all physical BVP rows plus the independent mass constraint.

    ``R26NodeBVP`` keeps a square system by replacing one interior continuity
    row with the global mass border.  For an overdetermined consistency solve,
    this objective restores that held continuity row at its original index and
    appends the mass equation as row ``unknown_count``.  No physical equation
    is removed, duplicated, or converted into a penalty.
    """

    def __init__(self, problem: R26NodeBVP, transform: LogStateTransform, penalty: float) -> None:
        self.problem = problem
        self.transform = transform
        self.penalty = float(penalty)
        self.invalid_evaluations = 0
        self.last_invalid_error: str | None = None

    @property
    def equation_count(self) -> int:
        return self.problem.unknown_count + 1

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        try:
            state = self.transform.decode(vector)
            evaluation = self.problem.evaluate(state)
            values = evaluation.residual.ravel().copy()
            mass_index = int(np.ravel_multi_index(evaluation.mass_row, self.problem.shape))
            values[mass_index] = (
                evaluation.diagnostics.held_out_continuity
                / self.problem.case.scaling.bulk[0]
            )
            appended_mass = (
                evaluation.diagnostics.mass_error / self.problem.case.scaling.mass
            )
            return np.concatenate((values, np.asarray((appended_mass,), dtype=float)))
        except (FloatingPointError, ValueError, OverflowError) as exc:
            self.invalid_evaluations += 1
            self.last_invalid_error = f"{type(exc).__name__}: {exc}"
            x = np.asarray(vector, dtype=float)
            sign = np.where(np.isfinite(x) & (x < 0.0), -1.0, 1.0)
            magnitude = 1.0 + np.minimum(
                np.nan_to_num(np.abs(x), nan=100.0, posinf=100.0), 100.0
            )
            guarded = self.penalty * sign * magnitude
            return np.concatenate((guarded, np.asarray((self.penalty,), dtype=float)))


class EncodedR26RawMassContinuityObjective:
    """Return every *unscaled* physical row plus the raw mass constraint.

    This is the fail-closed merit function for final nonlinear reconciliation.
    Component/family scaling can improve a predictor or Newton step, but a
    small scaled residual does not imply that every dimensionaless raw moment
    balance is small.  As in :class:`EncodedR26MassContinuityObjective`, the
    displaced local continuity row is restored at its original index and mass
    is appended as an independent equation.  No physical row is omitted and
    no penalty/regularization row is added for a valid state.
    """

    def __init__(self, problem: R26NodeBVP, transform: LogStateTransform, penalty: float) -> None:
        self.problem = problem
        self.transform = transform
        self.penalty = float(penalty)
        self.invalid_evaluations = 0
        self.last_invalid_error: str | None = None

    @property
    def equation_count(self) -> int:
        return self.problem.unknown_count + 1

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        try:
            state = self.transform.decode(vector)
            evaluation = self.problem.evaluate(state)
            values = evaluation.unscaled_residual.ravel().copy()
            mass_index = int(np.ravel_multi_index(evaluation.mass_row, self.problem.shape))
            values[mass_index] = evaluation.diagnostics.held_out_continuity
            return np.concatenate(
                (values, np.asarray((evaluation.diagnostics.mass_error,), dtype=float))
            )
        except (FloatingPointError, ValueError, OverflowError) as exc:
            self.invalid_evaluations += 1
            self.last_invalid_error = f"{type(exc).__name__}: {exc}"
            x = np.asarray(vector, dtype=float)
            sign = np.where(np.isfinite(x) & (x < 0.0), -1.0, 1.0)
            magnitude = 1.0 + np.minimum(
                np.nan_to_num(np.abs(x), nan=100.0, posinf=100.0), 100.0
            )
            guarded = self.penalty * sign * magnitude
            return np.concatenate((guarded, np.asarray((self.penalty,), dtype=float)))


def residual_family_row_scales(
    problem: R26NodeBVP,
    jacobian: np.ndarray,
    *,
    relative_floor: float = 1.0e-10,
    absolute_floor: float = 1.0e-12,
) -> np.ndarray:
    """Equilibrate Jacobian rows without changing a physical R26 equation.

    A single global norm is unsafe for this BVP: collision-dominated bulk
    moments, wall relations, numerical boundary completion, corners, and the
    bordered mass equation have very different Jacobian magnitudes.  This
    helper assigns one robust median row norm to each component *within* each
    residual family.  The scales are fixed for a nonlinear solve and final
    acceptance remains based on the original raw residual.

    The mass-border row is treated separately because it couples every
    density degree of freedom.  Floors only prevent division by zero for an
    exactly inactive row in a diagnostic/mock problem; they do not add a row
    or alter the BVP.
    """

    matrix = np.asarray(jacobian, dtype=float)
    expected = (problem.unknown_count, problem.unknown_count)
    if matrix.shape != expected or not np.isfinite(matrix).all():
        raise ValueError(f"jacobian must be finite with shape {expected}")
    if not np.isfinite(relative_floor) or relative_floor <= 0.0:
        raise ValueError("relative_floor must be finite and positive")
    if not np.isfinite(absolute_floor) or absolute_floor <= 0.0:
        raise ValueError("absolute_floor must be finite and positive")

    norms = np.linalg.norm(matrix, axis=1).reshape(problem.shape)
    scales = np.ones(problem.shape)

    interior = norms[1:-1, 1:-1].copy()
    interior[problem.mass_j - 1, problem.mass_i - 1, 0] = np.nan
    scales[1:-1, 1:-1] = np.nanmedian(interior, axis=(0, 1))

    wall = np.median(
        np.stack([norms[node.j, node.i, :11] for node in problem.boundary_nodes]),
        axis=0,
    )
    extrapolation = np.median(
        np.stack([norms[node.j, node.i, 11:] for node in problem.boundary_nodes]),
        axis=0,
    )
    for node in problem.boundary_nodes:
        scales[node.j, node.i, :11] = wall
        scales[node.j, node.i, 11:] = extrapolation

    corners = ((0, 0), (0, -1), (-1, 0), (-1, -1))
    corner = np.median(np.stack([norms[j, i] for j, i in corners]), axis=0)
    for j, i in corners:
        scales[j, i] = corner

    scales[problem.mass_j, problem.mass_i, 0] = norms[
        problem.mass_j, problem.mass_i, 0
    ]
    maximum = float(np.nanmax(scales, initial=0.0))
    floor = max(float(absolute_floor), float(relative_floor) * maximum)
    return np.maximum(np.nan_to_num(scales.ravel(), nan=floor), floor)


def jacobian_sparsity(problem: R26NodeBVP, *, stencil_radius: int = 2) -> csr_matrix:
    """Conservative dependency pattern for colored finite differences.

    The raw R26 rows contain a derivative of a closure that itself contains
    first derivatives, hence a radius-two dependency is required.  The mass
    border row additionally depends on every density degree of freedom.  This
    is a sparsity declaration only; it does not approximate or omit a term in
    the residual.
    """

    if stencil_radius < 2:
        raise ValueError("R26 closure derivatives require stencil_radius >= 2")
    ny, nx, nv = problem.shape
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    for j in range(ny):
        j0, j1 = max(0, j - stencil_radius), min(ny, j + stencil_radius + 1)
        for i in range(nx):
            i0, i1 = max(0, i - stencil_radius), min(nx, i + stencil_radius + 1)
            output_rows = (np.ravel_multi_index((j, i, 0), problem.shape) + np.arange(nv))
            neighbour_columns: list[int] = []
            for jj in range(j0, j1):
                for ii in range(i0, i1):
                    base = int(np.ravel_multi_index((jj, ii, 0), problem.shape))
                    neighbour_columns.extend(range(base, base + nv))
            cols = np.asarray(neighbour_columns, dtype=np.int64)
            rows.append(np.repeat(output_rows, cols.size))
            columns.append(np.tile(cols, output_rows.size))
    # The continuity row replaced by the integral mass constraint sees every
    # rho node, including all four explicitly modelled corners.
    mass_row = int(np.ravel_multi_index((problem.mass_j, problem.mass_i, 0), problem.shape))
    rho_columns = np.arange(0, problem.unknown_count, nv, dtype=np.int64)
    rows.append(np.full(rho_columns.shape, mass_row, dtype=np.int64))
    columns.append(rho_columns)
    row = np.concatenate(rows)
    column = np.concatenate(columns)
    data = np.ones(row.size, dtype=bool)
    return coo_matrix((data, (row, column)), shape=(problem.unknown_count, problem.unknown_count)).tocsr()


def _armijo_backtracking(
    objective: Callable[[np.ndarray], np.ndarray],
    encoded: np.ndarray,
    residual: np.ndarray,
    direction: np.ndarray,
    jacobian: csr_matrix,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    coefficient: float,
    minimum_alpha: float,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Return one physical residual-decreasing trial or ``None``.

    The objective itself maps invalid physical states (including non-positive
    effective wall pressure) to a large finite penalty.  This routine therefore
    remains fail-closed while backtracking away from such trials.  The Armijo
    right-hand side uses the actual clipped step rather than assuming an exact
    Newton direction, which is essential for regularized LSMR and gradient
    fallbacks.
    """

    x = np.asarray(encoded, dtype=float)
    r = np.asarray(residual, dtype=float)
    d = np.asarray(direction, dtype=float)
    if x.shape != r.shape or x.shape != d.shape:
        raise ValueError("line-search vectors must have identical shapes")
    if not np.isfinite(d).all():
        return None
    merit = 0.5 * float(np.dot(r, r))
    alpha = 1.0
    while alpha >= minimum_alpha:
        trial = np.clip(x + alpha * d, lower, upper)
        actual_step = trial - x
        if not np.any(actual_step):
            alpha *= 0.5
            continue
        slope_step = float(np.dot(r, jacobian @ actual_step))
        if not np.isfinite(slope_step) or slope_step >= 0.0:
            alpha *= 0.5
            continue
        trial_residual = np.asarray(objective(trial), dtype=float)
        trial_merit = 0.5 * float(np.dot(trial_residual, trial_residual))
        if np.isfinite(trial_merit) and trial_merit <= merit + coefficient * slope_step:
            return trial, trial_residual, alpha
        alpha *= 0.5
    return None


def _regularized_lsmr_directions(
    jacobian: csr_matrix,
    residual: np.ndarray,
    *,
    regularization_factors: tuple[float, ...],
    max_iterations: int,
    include_undamped: bool,
) -> tuple[tuple[str, np.ndarray], ...]:
    """Build deterministic Gauss--Newton/LM fallback directions."""

    data = np.asarray(jacobian.data, dtype=float)
    jacobian_scale = max(float(np.linalg.norm(data) / np.sqrt(max(data.size, 1))), 1.0e-14)
    candidates: list[tuple[str, np.ndarray]] = []
    for factor in regularization_factors:
        if factor == 0.0 and not include_undamped:
            continue
        damping = float(factor * jacobian_scale)
        direction = lsmr(
            jacobian,
            -residual,
            damp=damping,
            atol=1.0e-11,
            btol=1.0e-11,
            maxiter=max_iterations,
        )[0]
        label = "lsmr" if factor == 0.0 else f"lm-lsmr({factor:.0e})"
        candidates.append((label, np.asarray(direction, dtype=float)))

    gradient = np.asarray(jacobian.T @ residual, dtype=float)
    gradient_norm = float(np.linalg.norm(gradient))
    if np.isfinite(gradient_norm) and gradient_norm > 0.0:
        # Normalization makes alpha meaningful without changing the guaranteed
        # descent sign r^T J (-J^T r) < 0.
        candidates.append(("scaled-steepest-descent", -gradient / gradient_norm))
    return tuple(candidates)


def solve_r26_bvp(
    problem: R26NodeBVP,
    initial_state: np.ndarray,
    *,
    options: SolveOptions | None = None,
) -> R26SolveResult:
    """Solve one fixed case from one explicit initial state.

    Algebraic acceptance additionally requires the continuity equation removed
    for the mass border to remain small; optimizer success by itself is never
    reported as R26 convergence.
    """

    options = SolveOptions() if options is None else options
    transform = LogStateTransform(problem.shape)
    x0 = transform.encode(initial_state)
    objective = EncodedR26Objective(problem, transform, options.invalid_penalty)

    if options.method == "krylov":
        scipy_result: OptimizeResult = root(
            objective,
            x0,
            method="krylov",
            options={
                "fatol": options.residual_tolerance,
                "maxiter": options.max_iterations,
                "line_search": options.line_search,
                "disp": options.display,
            },
        )
        encoded = np.asarray(scipy_result.x, dtype=float)
        iterations = int(getattr(scipy_result, "nit", 0))
        evaluations = int(getattr(scipy_result, "nfev", 0))
    elif options.method == "least_squares":
        lower, upper = transform.least_squares_bounds()
        # On a very small grid the radius-two pattern is effectively dense
        # and a direct dense trust-region solve is faster/more accurate.  From
        # N=7 onward colored finite differences avoid the quadratic number of
        # residual calls.
        sparsity = None if problem.case.nodes <= 6 else jacobian_sparsity(problem)
        result = least_squares(
            objective,
            x0,
            bounds=(lower, upper),
            method="trf",
            jac="2-point",
            jac_sparsity=sparsity,
            x_scale="jac",
            ftol=options.residual_tolerance,
            xtol=options.residual_tolerance,
            gtol=options.residual_tolerance,
            max_nfev=options.max_function_evaluations,
            verbose=2 if options.display else 0,
        )
        scipy_result = result
        encoded = np.asarray(result.x, dtype=float)
        iterations = int(getattr(result, "njev", 0) or 0)
        evaluations = int(result.nfev)
    else:
        # Near equilibrium the steady R26 cavity system is almost linear.
        # One explicitly formed, conservatively colored finite-difference
        # Jacobian plus sparse Newton is substantially more deterministic
        # than an unpreconditioned Krylov solve.  The Jacobian is refreshed
        # after every accepted nonlinear step; a backtracking merit search
        # prevents silently accepting an invalid or residual-growing update.
        lower, upper = transform.least_squares_bounds()
        pattern = jacobian_sparsity(problem)
        encoded = x0.copy()
        evaluations = 0

        def counted(vector: np.ndarray) -> np.ndarray:
            nonlocal evaluations
            evaluations += 1
            return objective(vector)

        residual = counted(encoded)
        success = False
        message = "colored sparse Newton iteration limit reached"
        iterations = 0
        jacobian = None
        factorization = None
        chord_steps = 0
        globalization_fallbacks = 0
        for iteration in range(1, options.max_iterations + 1):
            iterations = iteration
            residual_linf = float(np.max(np.abs(residual), initial=0.0))
            if residual_linf <= options.residual_tolerance:
                success = True
                message = "colored sparse Newton residual tolerance reached"
                break
            if factorization is None or chord_steps >= 3:
                jacobian = approx_derivative(
                    counted,
                    encoded,
                    method="2-point",
                    # R26 high-order moments can be many orders of magnitude
                    # smaller than unity.  A relative-only perturbation then
                    # rounds away and corrupts the colored Jacobian precisely
                    # in the wall-layer rows that control grid reconciliation.
                    # Use the already audited absolute floor while retaining
                    # proportional growth for large encoded coordinates.
                    abs_step=fv_absolute_difference_step(encoded),
                    bounds=(lower, upper),
                    sparsity=pattern,
                ).tocsc()
                try:
                    factorization = splu(jacobian)
                except RuntimeError:
                    factorization = None
                chord_steps = 0
            try:
                if factorization is None:
                    raise RuntimeError("sparse LU unavailable")
                direction = factorization.solve(-residual)
                linear_solver = "splu"
            except RuntimeError:
                assert jacobian is not None
                direction = lsmr(
                    jacobian,
                    -residual,
                    atol=1.0e-11,
                    btol=1.0e-11,
                    maxiter=options.lsmr_max_iterations,
                )[0]
                linear_solver = "lsmr-fallback"
            if not np.isfinite(direction).all():
                message = f"{linear_solver} produced a non-finite Newton direction"
                break
            assert jacobian is not None
            merit = 0.5 * float(np.dot(residual, residual))
            outcome = _armijo_backtracking(
                counted,
                encoded,
                residual,
                direction,
                jacobian,
                lower,
                upper,
                coefficient=options.armijo_coefficient,
                minimum_alpha=options.minimum_line_search_alpha,
            )
            if outcome is not None:
                encoded, residual, _ = outcome
                chord_steps += 1
                trial_merit = 0.5 * float(np.dot(residual, residual))
                if trial_merit > 0.25 * merit:
                    factorization = None
                continue

            # A failed chord direction should first trigger an exact Jacobian
            # refresh.  Only a failure against that fresh Jacobian activates
            # the more expensive globalization ladder.
            if chord_steps > 0:
                factorization = None
                chord_steps = 0
                continue

            fallback_outcome: tuple[np.ndarray, np.ndarray, float] | None = None
            fallback_label = ""
            for candidate_label, candidate_direction in _regularized_lsmr_directions(
                jacobian,
                residual,
                regularization_factors=options.regularization_factors,
                max_iterations=options.lsmr_max_iterations,
                include_undamped=(linear_solver != "lsmr-fallback"),
            ):
                fallback_outcome = _armijo_backtracking(
                    counted,
                    encoded,
                    residual,
                    candidate_direction,
                    jacobian,
                    lower,
                    upper,
                    coefficient=options.armijo_coefficient,
                    minimum_alpha=options.minimum_line_search_alpha,
                )
                if fallback_outcome is not None:
                    fallback_label = candidate_label
                    break

            if fallback_outcome is not None:
                encoded, residual, _ = fallback_outcome
                globalization_fallbacks += 1
                factorization = None
                chord_steps = 0
                if options.display:
                    print(
                        f"R26 colored Newton globalization accepted {fallback_label} "
                        f"at iteration {iteration}",
                        flush=True,
                    )
                continue

            message = (
                "colored sparse Newton globalization failed after "
                f"{linear_solver}, regularized LSMR, and gradient descent"
            )
            break
        else:
            residual_linf = float(np.max(np.abs(residual), initial=0.0))
            if residual_linf <= options.residual_tolerance:
                success = True
                message = "colored sparse Newton residual tolerance reached"
        if success and globalization_fallbacks:
            message += f"; globalization fallbacks accepted={globalization_fallbacks}"
        scipy_result = OptimizeResult(
            x=encoded,
            success=success,
            message=message,
            nit=iterations,
            nfev=evaluations,
        )

    try:
        state = transform.decode(encoded)
        evaluation = problem.evaluate(state)
        diagnostics = evaluation.diagnostics
        physical_final = True
    except (FloatingPointError, ValueError) as exc:
        state = np.asarray(initial_state, dtype=float).copy()
        diagnostics = problem.evaluate(state).diagnostics
        physical_final = False
        objective.last_invalid_error = f"final state invalid: {type(exc).__name__}: {exc}"

    converged = bool(
        physical_final
        and diagnostics.total_linf <= options.residual_tolerance
        and abs(diagnostics.held_out_continuity) <= options.held_out_continuity_tolerance
    )
    message = str(getattr(scipy_result, "message", ""))
    if bool(getattr(scipy_result, "success", False)) and not converged:
        message += (
            "; optimizer stopped but strict R26 acceptance failed "
            f"(residual={diagnostics.total_linf:.3e}, "
            f"held-out continuity={diagnostics.held_out_continuity:.3e})"
        )
    return R26SolveResult(
        case=problem.case,
        state=state,
        encoded_state=encoded,
        diagnostics=diagnostics,
        converged=converged,
        scipy_success=bool(getattr(scipy_result, "success", False)),
        message=message,
        iterations=iterations,
        function_evaluations=evaluations,
        invalid_evaluations=objective.invalid_evaluations,
        last_invalid_error=objective.last_invalid_error,
        solver_method=options.method,
    )


def interpolate_state_grid(
    state: np.ndarray,
    new_nodes: int,
    *,
    target_mean_density: float = 1.0,
    mass_weights: np.ndarray | None = None,
    old_x: np.ndarray | None = None,
    old_y: np.ndarray | None = None,
    new_x: np.ndarray | None = None,
    new_y: np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate a square restart, preserving rho/T positivity and mass.

    Coordinate arrays make refinement between uniform and wall-stretched
    grids explicit.  Omitting all four arrays retains the legacy unit-square
    uniform behavior.
    """

    old = validate_planar_state(state)
    if old.ndim != 3 or old.shape[0] != old.shape[1]:
        raise ValueError("restart interpolation expects a square state grid")
    if new_nodes < 5:
        raise ValueError("new grid needs at least five nodes")
    old_nodes = old.shape[0]
    old_xv = np.linspace(0.0, 1.0, old_nodes) if old_x is None else np.asarray(old_x, dtype=float)
    old_yv = np.linspace(0.0, 1.0, old_nodes) if old_y is None else np.asarray(old_y, dtype=float)
    new_xv = np.linspace(0.0, 1.0, new_nodes) if new_x is None else np.asarray(new_x, dtype=float)
    new_yv = np.linspace(0.0, 1.0, new_nodes) if new_y is None else np.asarray(new_y, dtype=float)
    for name, coordinate, size in (
        ("old_x", old_xv, old_nodes),
        ("old_y", old_yv, old_nodes),
        ("new_x", new_xv, new_nodes),
        ("new_y", new_yv, new_nodes),
    ):
        if coordinate.shape != (size,) or not np.isfinite(coordinate).all() or np.any(np.diff(coordinate) <= 0.0):
            raise ValueError(f"{name} must be finite, increasing, and have length {size}")
    if new_xv[0] < old_xv[0] or new_xv[-1] > old_xv[-1] or new_yv[0] < old_yv[0] or new_yv[-1] > old_yv[-1]:
        raise ValueError("new grid must remain within the old interpolation domain")
    yy, xx = np.meshgrid(new_yv, new_xv, indexing="ij")
    points = np.column_stack((yy.ravel(), xx.ravel()))
    result = np.empty((new_nodes, new_nodes, NVAR))
    for component in range(NVAR):
        values = old[..., component]
        logarithmic = component in (0, 3)
        if logarithmic:
            values = np.log(values)
        interpolator = RegularGridInterpolator(
            (old_yv, old_xv), values, method="linear", bounds_error=True
        )
        interpolated = interpolator(points).reshape(new_nodes, new_nodes)
        result[..., component] = np.exp(interpolated) if logarithmic else interpolated
    if mass_weights is None:
        weights = trapezoidal_node_weights(new_nodes)
    else:
        weights = np.asarray(mass_weights, dtype=float)
        if weights.shape != (new_nodes, new_nodes):
            raise ValueError("mass_weights must match the refined node grid")
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("mass_weights must be finite and nonnegative")
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0:
            raise ValueError("mass_weights must have positive total weight")
        weights = weights / total_weight
    current_mass = float(np.sum(weights * result[..., 0]))
    result[..., 0] *= target_mean_density / current_mass
    return validate_planar_state(result)


def secant_predict_state(
    previous_state: np.ndarray,
    current_state: np.ndarray,
    factor: float,
    *,
    target_mean_density: float = 1.0,
    mass_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Extrapolate two accepted states while preserving rho/T positivity.

    ``factor`` is the requested parameter increment divided by the separation
    of the two accepted parameter values.  Density and temperature are
    extrapolated in logarithmic coordinates; all remaining R26 moments use a
    conventional secant predictor.  Density is finally renormalized with the
    same declared quadrature used by the mass equation.
    """

    previous = validate_planar_state(previous_state)
    current = validate_planar_state(current_state)
    if previous.shape != current.shape or current.ndim != 3:
        raise ValueError("secant states must have the same (ny,nx,17) shape")
    if not np.isfinite(factor) or factor < 0.0:
        raise ValueError("secant factor must be finite and nonnegative")
    if not np.isfinite(target_mean_density) or target_mean_density <= 0.0:
        raise ValueError("target mean density must be finite and positive")

    transform = LogStateTransform(current.shape)
    previous_encoded = transform.encode(previous)
    current_encoded = transform.encode(current)
    predicted = transform.decode(
        current_encoded + float(factor) * (current_encoded - previous_encoded)
    )

    nodes_y, nodes_x = current.shape[:2]
    if mass_weights is None:
        if nodes_y != nodes_x:
            raise ValueError("default secant mass weights require a square grid")
        weights = trapezoidal_node_weights(nodes_x)
    else:
        weights = np.asarray(mass_weights, dtype=float)
        if weights.shape != (nodes_y, nodes_x):
            raise ValueError("mass_weights must match the secant state grid")
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("mass_weights must be finite and nonnegative")
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0:
            raise ValueError("mass_weights must have positive total weight")
        weights = weights / total_weight
    current_mass = float(np.sum(weights * predicted[..., 0]))
    if not np.isfinite(current_mass) or current_mass <= 0.0:
        raise FloatingPointError("secant predictor produced invalid density mass")
    predicted[..., 0] *= target_mean_density / current_mass
    return validate_planar_state(predicted)


def solve_lid_continuation(
    case: CavityCase,
    *,
    steps: int,
    problem_factory: Callable[[CavityCase], R26NodeBVP] = R26NodeBVP,
    initial_state: np.ndarray | None = None,
    options: SolveOptions | None = None,
) -> tuple[R26SolveResult, ...]:
    """Run independent stateless solves along a monotone lid-speed ladder."""

    if steps < 1:
        raise ValueError("continuation steps must be positive")
    state = case.equilibrium_state() if initial_state is None else validate_planar_state(initial_state)
    speeds = np.linspace(0.0, case.lid_velocity, steps + 1)[1:]
    results: list[R26SolveResult] = []
    for index, speed in enumerate(speeds, start=1):
        step_case = replace(case, name=f"{case.name}-cont{index:03d}", lid_velocity=float(speed))
        result = solve_r26_bvp(problem_factory(step_case), state, options=options)
        results.append(result)
        if not result.converged:
            break
        state = result.state.copy()
    return tuple(results)


def save_restart(path: str | Path, result: R26SolveResult) -> Path:
    """Write a private NPZ restart with enough metadata to reject wrong cases."""

    output = Path(path)
    metadata = {
        "case_name": result.case.name,
        "nodes": result.case.nodes,
        "kn": result.case.kn,
        "kn_convention": result.case.kn_convention.value,
        "lid_velocity": result.case.lid_velocity,
        "grid_stretch_beta": result.case.grid_stretch_beta,
        "x": result.case.x.tolist(),
        "y": result.case.y.tolist(),
        "wall_temperature": result.case.wall_temperature,
        "converged": result.converged,
        "solver_method": result.solver_method,
    }
    np.savez_compressed(output, state=result.state, metadata=np.asarray(json.dumps(metadata)))
    return output


def load_restart(path: str | Path) -> tuple[np.ndarray, dict[str, object]]:
    """Load and validate a private NPZ restart without hidden solver history."""

    with np.load(Path(path), allow_pickle=False) as archive:
        state = validate_planar_state(np.asarray(archive["state"], dtype=float))
        metadata = json.loads(str(np.asarray(archive["metadata"]).item()))
    if state.ndim != 3 or state.shape[-1] != NVAR:
        raise ValueError("restart contains an invalid R26 state shape")
    return state, metadata


__all__ = [
    "EncodedR26Objective",
    "EncodedR26MassContinuityObjective",
    "EncodedR26RawMassContinuityObjective",
    "LogStateTransform",
    "R26SolveResult",
    "SolveOptions",
    "interpolate_state_grid",
    "jacobian_sparsity",
    "load_restart",
    "residual_family_row_scales",
    "save_restart",
    "secant_predict_state",
    "solve_lid_continuation",
    "solve_r26_bvp",
]
