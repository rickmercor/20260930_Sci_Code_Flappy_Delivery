#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np

def scaled_starting_matrix(A: np.ndarray, p: int) -> np.ndarray:
    """Reference starting matrix of the scaled coupled pair."""
    if isinstance(p, bool) or not isinstance(p, (int, np.integer)):
        raise ValueError("p must be an integer")
    if int(p) < 1:
        raise ValueError("p must be at least one")
    order = int(p)
    matrix = np.asarray(A)
    if np.iscomplexobj(matrix):
        raise ValueError("A must be real")
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("A must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("A must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError("A must be symmetric")
    projected = matrix.copy()
    for row in range(matrix.shape[0]):
        for column in range(row + 1, matrix.shape[1]):
            left = float(matrix[row, column])
            right = float(matrix[column, row])
            if np.signbit(left) == np.signbit(right):
                midpoint = left + 0.5 * (right - left)
            else:
                midpoint = 0.5 * (left + right)
            projected[row, column] = midpoint
            projected[column, row] = midpoint
    matrix = projected
    magnitude = float(np.max(np.abs(matrix)))
    if magnitude == 0.0:
        raise ValueError("the symmetric part of A must be nonzero")
    scaled = matrix / magnitude
    scaled_norm = float(np.linalg.norm(scaled, ord="fro"))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        frobenius = float(np.linalg.norm(matrix, ord="fro"))
    if (
        np.isfinite(frobenius)
        and frobenius > 0.0
        and frobenius <= np.finfo(float).max / 2.0
    ):
        scale = float((2.0 * frobenius / (order + 1.0)) ** (1.0 / order))
        return matrix / scale**order
    return (scaled / scaled_norm) * ((order + 1.0) / 2.0)

import numpy as np

def sketched_loss_coefficients(
    R: np.ndarray,
    S: np.ndarray,
    p: int,
) -> np.ndarray:
    """Reference sketched loss, assembled from scalar trace moments."""
    if isinstance(p, bool) or not isinstance(p, (int, np.integer)):
        raise ValueError("p must be an integer")
    if int(p) < 1:
        raise ValueError("p must be at least one")
    order = int(p)
    residual = np.asarray(R)
    sketch = np.asarray(S)
    if np.iscomplexobj(residual) or np.iscomplexobj(sketch):
        raise ValueError("R and S must be real")
    residual = np.asarray(residual, dtype=float)
    sketch = np.asarray(sketch, dtype=float)
    if (
        residual.ndim != 2
        or residual.shape[0] != residual.shape[1]
        or residual.shape[0] == 0
    ):
        raise ValueError("R must be a nonempty square matrix")
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(sketch)):
        raise ValueError("R and S must contain only finite values")
    if not np.allclose(residual, residual.T, rtol=0.0, atol=1e-12):
        raise ValueError("R must be symmetric")
    if sketch.ndim != 2 or sketch.shape[0] == 0 or sketch.shape[1] != residual.shape[0]:
        raise ValueError("S must have shape (m, n) with m positive")
    projected = residual.copy()
    for row in range(residual.shape[0]):
        for column in range(row + 1, residual.shape[1]):
            left = float(residual[row, column])
            right = float(residual[column, row])
            if np.signbit(left) == np.signbit(right):
                midpoint = left + 0.5 * (right - left)
            else:
                midpoint = 0.5 * (left + right)
            projected[row, column] = midpoint
            projected[column, row] = midpoint
    residual = projected
    width = order + 2
    expansion = np.zeros((order + 1, width), dtype=float)
    expansion[0, 1] = 1.0
    weight = 1.0
    for index in range(1, order + 1):
        weight = weight * (order - index + 1) / index
        expansion[index, index] -= weight
        expansion[index, index + 1] += weight
    moments = np.empty(2 * (width - 1) + 1, dtype=float)
    rolling = sketch.copy()
    for power in range(moments.size):
        moments[power] = float(np.tensordot(rolling, sketch, axes=([0, 1], [0, 1])))
        rolling = rolling @ residual
    gram = np.array(
        [[moments[a + b] for b in range(width)] for a in range(width)],
        dtype=float,
    )
    coefficients = np.zeros(2 * order + 1, dtype=float)
    for left in range(order + 1):
        for right in range(order + 1):
            coefficients[left + right] += float(
                expansion[left] @ gram @ expansion[right]
            )
    return coefficients

import numpy as np

def fit_bounded_coefficient(
    c: np.ndarray,
    lower: float,
    upper: float,
) -> float:
    """Reference bounded minimisation under the published conventions."""
    coefficients = np.asarray(c)
    if np.iscomplexobj(coefficients):
        raise ValueError("c must be real")
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 1 or coefficients.size < 2:
        raise ValueError("c must be one-dimensional with at least two entries")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("c must contain only finite values")
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("bounds must be finite and satisfy lower < upper")
    coefficient_scale = float(np.max(np.abs(coefficients)))
    extreme_scale = coefficient_scale != 0.0 and (
        coefficient_scale < 1e-150 or coefficient_scale > 1e150
    )
    working = coefficients / coefficient_scale if extreme_scale else coefficients
    ascending = np.array(
        [(j + 1) * working[j + 1] for j in range(working.size - 1)],
        dtype=float,
    )
    descending = ascending[::-1]
    nonzero = np.flatnonzero(descending != 0.0)
    found = [lower, upper]
    if nonzero.size and descending[nonzero[0] :].size > 1:
        for root in np.roots(descending[nonzero[0] :]):
            if abs(root.imag) > 1e-10:
                continue
            point = float(root.real)
            if abs(point - lower) <= 1e-12:
                point = lower
            if abs(point - upper) <= 1e-12:
                point = upper
            if lower <= point <= upper:
                found.append(point)
    found.sort()
    kept: list[float] = []
    for point in found:
        if not kept or abs(point - kept[-1]) > 1e-12:
            kept.append(point)
    best = float(kept[0])
    best_value = 0.0
    for coefficient in working[::-1]:
        best_value = best_value * best + float(coefficient)
    score_tolerance = float(1e-14 / coefficient_scale) if extreme_scale else 1e-14
    for point in kept[1:]:
        value = 0.0
        for coefficient in working[::-1]:
            value = value * float(point) + float(coefficient)
        if value < best_value - score_tolerance:
            best = float(point)
            best_value = value
    return float(best)

import numpy as np

def advance_inverse_newton(M: np.ndarray, alpha: float, p: int) -> np.ndarray:
    """Reference accelerated inverse Newton step."""
    carried = np.asarray(M)
    if np.iscomplexobj(carried):
        raise ValueError("M must be real")
    carried = np.asarray(carried, dtype=float)
    if (
        carried.ndim != 2
        or carried.shape[0] != carried.shape[1]
        or carried.shape[0] == 0
    ):
        raise ValueError("M must be a nonempty square matrix")
    if not np.all(np.isfinite(carried)):
        raise ValueError("M must contain only finite values")
    alpha = float(alpha)
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if isinstance(p, bool) or not isinstance(p, (int, np.integer)):
        raise ValueError("p must be an integer")
    if int(p) < 1:
        raise ValueError("p must be at least one")
    order = int(p)
    identity = np.eye(carried.shape[0], dtype=float)
    factor = identity + alpha * (identity - carried)
    return np.linalg.matrix_power(factor, order) @ carried

import numpy as np

def coupled_partner_factor(
    A: np.ndarray,
    alphas: np.ndarray,
    p: int,
) -> np.ndarray:
    """Reference partner iterate, advanced in lockstep with the earlier oracles."""
    if isinstance(p, bool) or not isinstance(p, (int, np.integer)):
        raise ValueError("p must be an integer")
    if int(p) < 1:
        raise ValueError("p must be at least one")
    order = int(p)
    coefficients = np.asarray(alphas)
    if np.iscomplexobj(coefficients):
        raise ValueError("alphas must be real")
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 1 or coefficients.size == 0:
        raise ValueError("alphas must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("alphas must contain only finite values")
    carried = scaled_starting_matrix(A, order)  # noqa: F821
    matrix = np.asarray(A, dtype=float)
    projected = matrix.copy()
    for row in range(matrix.shape[0]):
        for column in range(row + 1, matrix.shape[1]):
            left = float(matrix[row, column])
            right = float(matrix[column, row])
            if np.signbit(left) == np.signbit(right):
                midpoint = left + 0.5 * (right - left)
            else:
                midpoint = 0.5 * (left + right)
            projected[row, column] = midpoint
            projected[column, row] = midpoint
    matrix = projected
    magnitude = float(np.max(np.abs(matrix)))
    scaled = matrix / magnitude
    scaled_norm = float(np.linalg.norm(scaled, ord="fro"))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        frobenius = float(np.linalg.norm(matrix, ord="fro"))
    if (
        np.isfinite(frobenius)
        and frobenius > 0.0
        and frobenius <= np.finfo(float).max / 2.0
    ):
        scale = float((2.0 * frobenius / (order + 1.0)) ** (1.0 / order))
        inverse_scale = 1.0 / scale
    else:
        log_scale = (
            np.log(2.0 / (order + 1.0)) + np.log(magnitude) + np.log(scaled_norm)
        ) / order
        inverse_scale = float(np.exp(-log_scale))
    identity = np.eye(carried.shape[0], dtype=float)
    partner = identity * inverse_scale
    for alpha in coefficients:
        partner = partner @ (identity + float(alpha) * (identity - carried))
        carried = advance_inverse_newton(carried, float(alpha), order)  # noqa: F821
    return partner

import numpy as np

def fitted_coefficient_sequence(
    A: np.ndarray,
    S: np.ndarray,
    p: int,
    lower: float,
    upper: float,
    iterations: int,
) -> np.ndarray:
    """Reference coefficient trajectory composed from the four earlier oracles."""
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)):
        raise ValueError("iterations must be an integer")
    if int(iterations) < 1:
        raise ValueError("iterations must be at least one")
    steps = int(iterations)
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("bounds must be finite and satisfy lower < upper")
    sketch = np.asarray(S)
    if np.iscomplexobj(sketch):
        raise ValueError("S must be real")
    sketch = np.asarray(sketch, dtype=float)
    if sketch.ndim != 2 or sketch.shape[0] == 0:
        raise ValueError("S must be a nonempty two-dimensional matrix")
    if not np.all(np.isfinite(sketch)):
        raise ValueError("S must contain only finite values")
    carried = scaled_starting_matrix(A, p)  # noqa: F821
    if sketch.shape[1] != carried.shape[0]:
        raise ValueError("S width must equal the dimension of A")
    order = int(p)
    identity = np.eye(carried.shape[0], dtype=float)
    selected = np.empty(steps, dtype=float)
    for index in range(steps):
        residual = identity - carried
        coefficients = sketched_loss_coefficients(  # noqa: F821
            residual, sketch, order
        )
        alpha = fit_bounded_coefficient(coefficients, lower, upper)  # noqa: F821
        selected[index] = alpha
        carried = advance_inverse_newton(carried, alpha, order)  # noqa: F821
    return selected

import numpy as np

def residual_trajectory(
    A: np.ndarray,
    S: np.ndarray,
    p: int,
    lower: float,
    upper: float,
    iterations: int,
) -> np.ndarray:
    """Reference residual path, replayed from the earlier oracles."""
    selected = fitted_coefficient_sequence(  # noqa: F821
        A, S, p, lower, upper, iterations
    )
    carried = scaled_starting_matrix(A, p)  # noqa: F821
    order = int(p)
    identity = np.eye(carried.shape[0], dtype=float)
    path = np.empty(selected.size + 1, dtype=float)
    path[0] = float(np.linalg.norm(identity - carried, ord="fro"))
    for index, alpha in enumerate(selected):
        carried = advance_inverse_newton(carried, float(alpha), order)  # noqa: F821
        path[index + 1] = float(np.linalg.norm(identity - carried, ord="fro"))
    if not np.all(np.isfinite(path)):
        raise ValueError("residual trajectory must be finite")
    return path

import numpy as np

def compute_inverse_root_residual(
    A: np.ndarray,
    S: np.ndarray,
    p: int,
    lower: float,
    upper: float,
    iterations: int,
) -> float:
    """Reference end-to-end solver composed from the seven earlier oracles."""
    path = residual_trajectory(A, S, p, lower, upper, iterations)  # noqa: F821
    selected = fitted_coefficient_sequence(  # noqa: F821
        A, S, p, lower, upper, iterations
    )
    if selected.size + 1 != path.size:
        raise ValueError("coefficient and residual trajectories must agree in length")
    order = int(p)
    carried = scaled_starting_matrix(A, order)  # noqa: F821
    identity = np.eye(carried.shape[0], dtype=float)
    for alpha in selected:
        residual = identity - carried
        replayed = fit_bounded_coefficient(  # noqa: F821
            sketched_loss_coefficients(residual, S, order),  # noqa: F821
            lower,
            upper,
        )
        if not np.isfinite(replayed):
            raise ValueError("replayed coefficient must be finite")
        carried = advance_inverse_newton(carried, float(alpha), order)  # noqa: F821
    partner = coupled_partner_factor(A, selected, order)  # noqa: F821
    tie = float(
        np.linalg.norm(
            carried
            - np.linalg.matrix_power(partner, order) @ np.asarray(A, dtype=float),
            ord="fro",
        )
    )
    span = max(1.0, float(np.linalg.norm(carried, ord="fro")))
    if not np.isfinite(tie) or tie > 1e-6 * span:
        raise ValueError("coupled members must stay tied through the run")
    value = float(np.linalg.norm(identity - carried, ord="fro"))
    if not np.isfinite(value):
        raise ValueError("final residual must be finite")
    return float(np.round(value, 12))
SCICODE_GOLD_EOF
