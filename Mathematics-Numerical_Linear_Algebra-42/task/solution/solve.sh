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

def compute_glr_laguerre_state(
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference modified GLR predictor/corrector construction."""
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least one")

    machine_epsilon = np.finfo(float).eps

    def _evaluate_weighted_laguerre(x: float) -> tuple[float, float]:
        exponential = np.exp(-0.5 * x)
        laguerre = (1.0 - x) * exponential
        difference = -x * exponential
        derivative = -0.5 * laguerre - exponential
        for degree in range(1, n):
            next_difference = (degree * difference - x * laguerre) / (degree + 1.0)
            next_laguerre = laguerre + next_difference
            next_derivative = derivative - 0.5 * (laguerre + next_laguerre)
            laguerre = next_laguerre
            difference = next_difference
            derivative = next_derivative
        return float(laguerre), float(derivative)

    def _phase_integrate(theta: float, target: float, x: float) -> float:
        step_size = (target - theta) / 10.0
        for _ in range(10):
            phase_term = n + 0.5 - 0.25 * x
            first = -step_size / (
                np.sqrt(phase_term / x)
                + 0.25 * (1.0 / x - 0.25 / phase_term) * np.sin(2.0 * theta)
            )
            theta += step_size
            x += first
            phase_term = n + 0.5 - 0.25 * x
            second = -step_size / (
                np.sqrt(phase_term / x)
                + 0.25 * (1.0 / x - 0.25 / phase_term) * np.sin(2.0 * theta)
            )
            x += 0.5 * (second - first)
        return float(x)

    def _refine_initial_root(seed: float) -> tuple[float, float]:
        value, derivative = _evaluate_weighted_laguerre(seed)
        phase = np.arctan(np.sqrt(seed / (n + 0.5 - 0.25 * seed)) * derivative / value)
        root = _phase_integrate(float(phase), -0.5 * np.pi, seed)
        step = np.inf
        for _ in range(200):
            if abs(step) <= machine_epsilon and abs(value) <= machine_epsilon:
                break
            value, derivative = _evaluate_weighted_laguerre(root)
            step = value / derivative
            root -= step
        _, derivative = _evaluate_weighted_laguerre(root)
        return float(root), float(derivative)

    initial_count = min(20, n)
    roots = np.zeros(n, dtype=float)
    derivatives = np.zeros(n, dtype=float)
    seed = 1.0 / (2.0 * n + 1.0)
    for index in range(initial_count):
        seed, derivatives[index] = _refine_initial_root(seed)
        roots[index] = seed
        seed *= 1.1

    root = roots[initial_count - 1]
    order = 60 if n < 30 else 30
    for current in range(initial_count - 1, n - 1):
        if current == n - 6:
            order = 60

        displacement = _phase_integrate(0.5 * np.pi, -0.5 * np.pi, root) - root
        inverse_scale = 1.0 / displacement
        scale2 = inverse_scale**2
        scale3 = inverse_scale**3
        scale4 = inverse_scale**4
        recurrence_term = root * (n + 0.5 - 0.25 * root)
        root2 = root**2

        coefficients = np.zeros(order + 1, dtype=float)
        derivative_coefficients = np.zeros(order + 1, dtype=float)
        coefficients[1] = derivatives[current] / inverse_scale
        coefficients[2] = -0.5 * coefficients[1] / (inverse_scale * root)
        coefficients[3] = (
            -coefficients[2] / (inverse_scale * root)
            + (-(1.0 + recurrence_term) * coefficients[1] / (6.0 * scale2)) / root2
        )
        derivative_coefficients[:3] = (
            coefficients[1],
            2.0 * coefficients[2] * inverse_scale,
            3.0 * coefficients[3] * inverse_scale,
        )
        for k in range(2, order - 1):
            coefficients[k + 2] = (
                -root
                * (2.0 * k + 1.0)
                * (k + 1.0)
                * coefficients[k + 1]
                / inverse_scale
                - (k * k + recurrence_term) * coefficients[k] / scale2
                - (n + 0.5 - 0.5 * root) * coefficients[k - 1] / scale3
                + 0.25 * coefficients[k - 2] / scale4
            ) / (root2 * (k + 2.0) * (k + 1.0))
            derivative_coefficients[k + 1] = (
                (k + 2.0) * coefficients[k + 2] * inverse_scale
            )

        coefficients = coefficients[::-1]
        derivative_coefficients = derivative_coefficients[::-1]
        powers = np.ones(order + 1, dtype=float)
        powers[-1] = inverse_scale
        newton_step = np.inf
        for _ in range(10):
            if abs(newton_step) <= machine_epsilon:
                break
            newton_step = np.dot(coefficients, powers) / np.dot(
                derivative_coefficients, powers
            )
            displacement -= newton_step
            scaled_displacement = inverse_scale * displacement
            powers = np.concatenate(
                ([inverse_scale], np.cumprod(np.full(order, scaled_displacement)))
            )[::-1]

        root += displacement
        roots[current + 1] = root
        derivatives[current + 1] = np.dot(derivative_coefficients, powers)

    if (
        not np.all(np.isfinite(roots))
        or np.any(roots <= 0.0)
        or np.any(np.diff(roots) <= 0.0)
        or not np.all(np.isfinite(derivatives))
        or np.any(derivatives == 0.0)
    ):
        raise ValueError("modified GLR construction failed")
    nodes = np.concatenate(([0.0], roots))
    return nodes, derivatives

import numpy as np
from scipy import special

def build_truncated_collocation_state(
    n: int,
    alpha: float,
    nodes: np.ndarray,
    weighted_derivatives: np.ndarray,
    selected_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Reference augmented Table 2/Table 3 construction."""
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha <= -1.0:
        raise ValueError("alpha must be finite and greater than -1")
    n = int(n)
    alpha = float(alpha)
    x = np.asarray(nodes, dtype=float)
    derivatives = np.asarray(weighted_derivatives, dtype=float)
    if n < 1 or x.shape != (n + 1,) or derivatives.shape != (n,):
        raise ValueError("grid state has incompatible shapes")
    if (
        not np.all(np.isfinite(x))
        or x[0] != 0.0
        or np.any(x[1:] <= 0.0)
        or np.any(np.diff(x) <= 0.0)
    ):
        raise ValueError("nodes must start at zero and increase")
    expected_signs = np.where(np.arange(n) % 2 == 0, -1.0, 1.0)
    if (
        not np.all(np.isfinite(derivatives))
        or np.any(derivatives == 0.0)
        or not np.array_equal(np.sign(derivatives), expected_signs)
    ):
        raise ValueError("weighted derivatives have invalid signs or values")

    endpoint_log = (
        special.gammaln(n + alpha + 1.0)
        - special.gammaln(alpha + 1.0)
        - special.gammaln(n + 1.0)
    )
    endpoint_coefficient = float(np.exp(endpoint_log))
    coefficients = np.concatenate(([endpoint_coefficient], x[1:] * derivatives))
    if not np.all(np.isfinite(coefficients)) or np.any(coefficients == 0.0):
        raise ValueError("scaled coefficients must be finite and nonzero")

    total = n + 1
    if selected_indices is None:
        indices = np.arange(total, dtype=int)
    else:
        raw_indices = np.asarray(selected_indices)
        if (
            raw_indices.ndim != 1
            or raw_indices.size < 1
            or raw_indices.dtype.kind not in "iu"
        ):
            raise ValueError("selected_indices must be a nonempty integer vector")
        indices = raw_indices.astype(int, copy=False)
        if (
            np.any(indices < 0)
            or np.any(indices >= total)
            or np.any(np.diff(indices) <= 0)
        ):
            raise ValueError("selected_indices must increase within the full grid")

    selected_x = x[indices]
    selected_coefficients = coefficients[indices]
    count = indices.size
    differences = selected_x[:, None] - selected_x[None, :]
    ratios = selected_coefficients[:, None] / selected_coefficients[None, :]
    off_diagonal = ~np.eye(count, dtype=bool)
    safe_differences = np.where(off_diagonal, differences, 1.0)

    first = ratios / safe_differences
    first_diagonal = np.empty(count, dtype=float)
    at_endpoint = indices == 0
    first_diagonal[at_endpoint] = -0.5 - n / (alpha + 1.0)
    first_diagonal[~at_endpoint] = (1.0 - alpha) / (2.0 * selected_x[~at_endpoint])
    first[np.arange(count), np.arange(count)] = first_diagonal

    second = 2.0 / safe_differences * (ratios * first_diagonal[:, None] - first)
    second_diagonal = np.empty(count, dtype=float)
    second_diagonal[at_endpoint] = 0.25 + n * (n + alpha + 1.0) / (
        (alpha + 1.0) * (alpha + 2.0)
    )
    positive = selected_x[~at_endpoint]
    correction = 4.0 * (alpha + 1.0) * (alpha - 1.0)
    second_diagonal[~at_endpoint] = 1.0 / 12.0 - (
        2.0 * (2.0 * n + alpha + 1.0) * positive - correction
    ) / (12.0 * positive**2)
    second[np.arange(count), np.arange(count)] = second_diagonal

    blocks = np.stack((first, second))
    if not np.all(np.isfinite(blocks)):
        raise ValueError("derivative blocks are not finite")
    return blocks

import numpy as np

def scale_and_reduce_second_order(
    nodes: np.ndarray, second_order: np.ndarray, beta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reference coordinate scaling and endpoint deletion."""
    x = np.asarray(nodes, dtype=float)
    matrix = np.asarray(second_order, dtype=float)
    if x.ndim != 1 or x.size < 2 or matrix.shape != (x.size, x.size):
        raise ValueError("nodes and second_order have incompatible shapes")
    if (
        not np.all(np.isfinite(x))
        or x[0] != 0.0
        or np.any(x[1:] <= 0.0)
        or np.any(np.diff(x) <= 0.0)
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("nodes and second_order must be finite and ordered")
    if not np.isscalar(beta) or not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be positive and finite")
    beta = float(beta)
    physical_nodes = x[1:] / beta
    reduced_second_order = beta**2 * matrix[1:, 1:]
    if not np.all(np.isfinite(reduced_second_order)):
        raise ValueError("scaled operator is not finite")
    return physical_nodes, reduced_second_order

import numpy as np
from scipy import special

def sample_woods_saxon_profile(
    physical_nodes: np.ndarray, radius: float, diffuseness: float
) -> np.ndarray:
    """Reference stable Woods--Saxon sampling."""
    x = np.asarray(physical_nodes, dtype=float)
    if (
        x.ndim != 1
        or x.size == 0
        or not np.all(np.isfinite(x))
        or np.any(x <= 0.0)
        or np.any(np.diff(x) <= 0.0)
    ):
        raise ValueError("physical_nodes must be finite, positive, and increasing")
    if not np.isscalar(radius) or not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")
    if (
        not np.isscalar(diffuseness)
        or not np.isfinite(diffuseness)
        or diffuseness <= 0.0
    ):
        raise ValueError("diffuseness must be positive and finite")

    potential = special.expit(-(x - float(radius)) / float(diffuseness))
    if (
        not np.all(np.isfinite(potential))
        or np.any(potential <= 0.0)
        or np.any(potential >= 1.0)
    ):
        raise ValueError("profile samples must lie strictly between zero and one")
    return potential

import numpy as np

def assemble_generalized_pencil(
    reduced_second_order: np.ndarray, potential: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reference assembly of ``A = -D2 + I`` and ``Q = diag(q)``."""
    matrix = np.asarray(reduced_second_order, dtype=float)
    q = np.asarray(potential, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[0] != matrix.shape[1]
        or q.ndim != 1
        or q.size != matrix.shape[0]
    ):
        raise ValueError("operator and potential have incompatible shapes")
    if (
        not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(q))
        or np.any(q <= 0.0)
    ):
        raise ValueError("operator must be finite and potential must be positive")

    a_matrix = -matrix + np.eye(matrix.shape[0], dtype=float)
    q_matrix = np.diag(q)
    return a_matrix, q_matrix

import numpy as np
from scipy import linalg

def compute_finite_spectral_magnitudes(
    a_matrix: np.ndarray, q_matrix: np.ndarray
) -> np.ndarray:
    """Reference homogeneous generalized-eigenvalue filtering."""
    a = np.asarray(a_matrix, dtype=float)
    q = np.asarray(q_matrix, dtype=float)
    if (
        a.ndim != 2
        or q.ndim != 2
        or a.shape != q.shape
        or a.shape[0] == 0
        or a.shape[0] != a.shape[1]
    ):
        raise ValueError("pencil matrices must be matching nonempty squares")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(q)):
        raise ValueError("pencil matrices must be finite")

    homogeneous = linalg.eigvals(a, q, homogeneous_eigvals=True, check_finite=True)
    numerators = homogeneous[0]
    denominators = homogeneous[1]
    mask = np.isfinite(numerators) & np.isfinite(denominators) & (denominators != 0.0)
    eigenvalues = numerators[mask] / denominators[mask]
    eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
    if eigenvalues.size == 0:
        raise ValueError("the pencil has no finite generalized eigenvalues")
    magnitudes = np.sort(np.abs(eigenvalues).astype(float, copy=False))
    if not np.all(np.isfinite(magnitudes)):
        raise ValueError("finite generalized eigenvalues could not be resolved")
    return magnitudes

import numpy as np

def select_one_based_mode(magnitudes: np.ndarray, mode_index: int) -> float:
    """Reference validation and one-based selection."""
    values = np.asarray(magnitudes, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(np.diff(values) < 0.0)
    ):
        raise ValueError("magnitudes must be finite, nonnegative, and sorted")
    if isinstance(mode_index, (bool, np.bool_)) or not isinstance(
        mode_index, (int, np.integer)
    ):
        raise ValueError("mode_index must be a non-boolean integer")
    index = int(mode_index)
    if index < 1 or index > values.size:
        raise ValueError("mode_index lies outside the available spectrum")
    return float(values[index - 1])

import numpy as np

def compute_laguerre_woods_saxon_mode(
    n: int,
    beta: float,
    radius: float,
    diffuseness: float,
    mode_index: int,
) -> float:
    """Reference end-to-end composition of the seven earlier oracles."""
    if isinstance(mode_index, (bool, np.bool_)) or not isinstance(
        mode_index, (int, np.integer)
    ):
        raise ValueError("mode_index must be an integer")
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    n = int(n)
    mode_index = int(mode_index)
    if n < 1 or mode_index < 1 or mode_index > n:
        raise ValueError("mode_index must lie between one and n")

    alpha = 0.0
    nodes, derivatives = compute_glr_laguerre_state(n)  # noqa: F821
    collocation = build_truncated_collocation_state(  # noqa: F821
        n, alpha, nodes, derivatives
    )
    physical_nodes, reduced_d2 = scale_and_reduce_second_order(  # noqa: F821
        nodes, collocation[1], beta
    )
    potential = sample_woods_saxon_profile(  # noqa: F821
        physical_nodes, radius, diffuseness
    )
    a_matrix, q_matrix = assemble_generalized_pencil(  # noqa: F821
        reduced_d2, potential
    )
    magnitudes = compute_finite_spectral_magnitudes(  # noqa: F821
        a_matrix, q_matrix
    )
    return select_one_based_mode(magnitudes, mode_index)  # noqa: F821
SCICODE_GOLD_EOF
