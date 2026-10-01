"""
Certify the global bounded optimum of the nonsmooth two-channel objective.



The certificate joins the finite candidate construction to the original

three-term error identity.  Besides the minimizer, it records one-sided

derivatives, the separation from the next distinct candidate, reconstruction

residuals on every smooth cell, and preservation of the unquantized product.

Returns
-------
finite float np.ndarray of shape (14,), global optimum and verification certificate
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def certify_bounded_shared_scaling(
    A: np.ndarray,
    B: np.ndarray,
    bits_a: int,
    bits_b: int,
    bound: float,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Return a numerical global-optimality certificate on the closed bound.

    The reported switch count is the size of the bounded partition, which
    always includes both closed-bound endpoints ``-bound`` and ``+bound`` in
    addition to every distinct in-bound range tie.

    Raises ``ValueError`` unless the identity-transform expected error is
    finite and strictly positive, as well as for invalid arguments.

    Parameters
    ----------
    A : np.ndarray
        Finite left factor of shape ``(m, 2)``.
    B : np.ndarray
        Finite right factor of shape ``(2, n)``.
    bits_a : int
        Non-Boolean signed bit width for the left factor, at least 2.
    bits_b : int
        Non-Boolean signed bit width for the right factor, at least 2.
    bound : float
        Finite positive log-scale bound.
    tolerance : float, optional
        Finite positive switch and tie tolerance, below ``bound``.

    Returns
    -------
    np.ndarray
        A finite shape-``(14,)`` array containing ``x_star``, reciprocal
        scales, optimized and identity errors, switch and candidate counts,
        candidate kind and index, left and right derivatives, second-candidate
        gap, maximum branch residual, and product-preservation residual.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _certificate_inputs(
    A: np.ndarray,
    B: np.ndarray,
    bits_a: int,
    bits_b: int,
    bound: float,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    left = np.asarray(A, dtype=float)
    right = np.asarray(B, dtype=float)
    if left.ndim != 2 or left.shape[0] < 1 or left.shape[1] != 2:
        raise ValueError("A must have shape (m, 2) with m positive")
    if right.ndim != 2 or right.shape[0] != 2 or right.shape[1] < 1:
        raise ValueError("B must have shape (2, n) with n positive")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("matrix entries must be finite")
    if isinstance(bound, (bool, np.bool_)) or not np.isscalar(bound):
        raise ValueError("bound must be a finite positive scalar")
    if isinstance(tolerance, (bool, np.bool_)) or not np.isscalar(tolerance):
        raise ValueError("tolerance must be finite, positive, and below bound")
    try:
        limit = float(bound)
        tol = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("bound and tolerance must be real scalars") from exc
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("bound must be a finite positive scalar")
    if not np.isfinite(tol) or tol <= 0.0 or tol >= limit:
        raise ValueError("tolerance must be finite, positive, and below bound")
    return (
        left,
        right,
        _oracle_compute_dither_variance_coefficient(bits_a),  # noqa: F821 - step 01
        _oracle_compute_dither_variance_coefficient(bits_b),  # noqa: F821 - step 01
        limit,
        tol,
    )


def _cell_branch(
    left: np.ndarray,
    right: np.ndarray,
    bits_a: int,
    bits_b: int,
    lower: float,
    upper: float,
    probe: float,
    tolerance: float,
) -> np.ndarray:
    """Return one smooth cell's ``(P, Q, C)`` branch from the step-04 oracle."""
    tie_tolerance = min(float(tolerance), 0.25 * (float(upper) - float(lower)))
    branch = _oracle_reconstruct_active_error_branch(  # noqa: F821 - step 04
        left, right, bits_a, bits_b, float(probe), tie_tolerance
    )
    return np.array([branch[0], branch[1], branch[2]], dtype=float)


def _certificate_error(
    left: np.ndarray,
    right: np.ndarray,
    coefficient_a: float,
    coefficient_b: float,
    x: float,
) -> float:
    scales = np.exp(np.array([x, -x], dtype=float))
    transformed_left = left * scales
    transformed_right = right / scales[:, None]
    variance_a = coefficient_a * np.max(np.abs(transformed_left), axis=1) ** 2
    variance_b = coefficient_b * np.max(np.abs(transformed_right), axis=0) ** 2
    field_a = np.repeat(variance_a[:, None], 2, axis=1)
    field_b = np.repeat(variance_b[None, :], 2, axis=0)
    return float(
        _oracle_compute_expected_product_error(  # noqa: F821 - step 06
            transformed_left, transformed_right, field_a, field_b
        )[3]
    )


def _branch_value(coefficients: np.ndarray, x: float) -> float:
    return float(
        coefficients[0] * np.exp(4.0 * x)
        + coefficients[1] * np.exp(-4.0 * x)
        + coefficients[2]
    )


def _branch_derivative(coefficients: np.ndarray, x: float) -> float:
    return float(
        4.0 * (coefficients[0] * np.exp(4.0 * x) - coefficients[1] * np.exp(-4.0 * x))
    )


def _oracle_certify_bounded_shared_scaling(
    A: np.ndarray,
    B: np.ndarray,
    bits_a: int,
    bits_b: int,
    bound: float,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Reference global certificate from active cells and finite candidates."""
    left, right, coefficient_a, coefficient_b, limit, tol = _certificate_inputs(
        A, B, bits_a, bits_b, bound, tolerance
    )
    identity_error = _certificate_error(left, right, coefficient_a, coefficient_b, 0.0)
    if not np.isfinite(identity_error) or identity_error <= 0.0:
        raise ValueError("identity-transform error must be finite and positive")

    partition = _oracle_construct_two_channel_switch_partition(  # noqa: F821 - step 02
        left, right, limit, tol
    )
    count = int(round(float(partition[0])))
    switches = np.asarray(partition[1 : 1 + count], dtype=float)
    probes = np.asarray(partition[1 + count :], dtype=float)
    branches = [
        _cell_branch(left, right, bits_a, bits_b, lower, upper, probe, tol)
        for lower, upper, probe in zip(switches[:-1], switches[1:], probes)
    ]
    table = _oracle_enumerate_bounded_error_candidates(  # noqa: F821 - step 05
        left, right, bits_a, bits_b, limit, tol
    )
    records = np.asarray(table[1:], dtype=float).reshape(int(round(float(table[0]))), 8)
    candidates = records[:, [0, 1, 2, 6]]
    minimum = float(np.min(candidates[:, 3]))
    energy_tol = tol * max(1.0, abs(minimum))
    eligible = np.flatnonzero(candidates[:, 3] <= minimum + energy_tol)
    best = int(eligible[np.argmin(candidates[eligible, 0])])
    x_star, kind, cell_value, optimized_error = candidates[best]
    cell = int(cell_value)

    switch_distance = np.abs(switches - x_star)
    switch_index = int(np.argmin(switch_distance))
    if switch_distance[switch_index] <= tol:
        left_cell = max(0, switch_index - 1)
        right_cell = min(len(branches) - 1, switch_index)
    else:
        left_cell = right_cell = cell
    left_derivative = _branch_derivative(branches[left_cell], float(x_star))
    right_derivative = _branch_derivative(branches[right_cell], float(x_star))

    distinct = np.flatnonzero(np.abs(candidates[:, 0] - x_star) > tol)
    gap = (
        float(np.min(candidates[distinct, 3]) - optimized_error)
        if distinct.size
        else 0.0
    )
    gap = max(0.0, gap)

    max_branch_residual = 0.0
    for (lower, upper), coefficients in zip(zip(switches[:-1], switches[1:]), branches):
        for fraction in (0.25, 0.75):
            point = float(lower + fraction * (upper - lower))
            direct = _certificate_error(
                left, right, coefficient_a, coefficient_b, point
            )
            max_branch_residual = max(
                max_branch_residual,
                abs(direct - _branch_value(coefficients, point)),
            )

    scales = np.exp(np.array([x_star, -x_star], dtype=float))
    transformed_left = left * scales
    transformed_right = right / scales[:, None]
    product_residual = float(
        np.max(np.abs(transformed_left @ transformed_right - left @ right))
    )
    result = np.array(
        [
            x_star,
            scales[0],
            scales[1],
            optimized_error,
            identity_error,
            float(switches.size),
            float(candidates.shape[0]),
            kind,
            float(best),
            left_derivative,
            right_derivative,
            gap,
            max_branch_residual,
            product_residual,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("global certificate must be finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return interior, switch, symmetric, boundary, mixed-bit, and invalid cases."""
    return [
        {
            "setup": """import numpy as np
A = np.array([[12.0,0.5],[7.0,-0.4],[0.6,9.0],[0.8,6.0],[1.2,0.7]])
B = np.array([[0.2,0.5,-0.3,0.4],[6.0,-4.0,5.0,3.0]])
bits_a = bits_b = 2
bound = np.log(4.0)
""",
            "call": "certify_bounded_shared_scaling(A, B, bits_a, bits_b, bound)",
            "gold_call": "_oracle_certify_bounded_shared_scaling(A, B, bits_a, bits_b, bound)",
            "tol": 1e-9,
        },
        {
            "setup": """import numpy as np
A = np.array([[4.0,-4.0],[-4.0,-1.0],[2.0,4.0]])
B = np.array([[-5.0,1.0,-1.0],[-4.0,-4.0,-4.0]])
bits_a, bits_b = 2, 3
bound = 1.0
""",
            "call": "certify_bounded_shared_scaling(A, B, bits_a, bits_b, bound)",
            "gold_call": "_oracle_certify_bounded_shared_scaling(A, B, bits_a, bits_b, bound)",
            "tol": 1e-9,
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0,1.0],[1.0,2.0],[-2.0,1.0],[1.0,-2.0]])
B = np.array([[2.0,1.0,-2.0,1.0],[1.0,2.0,1.0,-2.0]])
bits_a = bits_b = 2
bound = np.log(4.0)
""",
            "call": "certify_bounded_shared_scaling(A, B, bits_a, bits_b, bound)",
            "gold_call": "_oracle_certify_bounded_shared_scaling(A, B, bits_a, bits_b, bound)",
            "tol": 1e-9,
        },
        {
            "setup": """import numpy as np
A = np.array([[100.0,1.0],[80.0,2.0],[60.0,1.0]])
B = np.array([[0.001,0.002],[4.0,5.0]])
bits_a = bits_b = 2
bound = np.log(4.0)
""",
            "call": "certify_bounded_shared_scaling(A, B, bits_a, bits_b, bound)",
            "gold_call": "_oracle_certify_bounded_shared_scaling(A, B, bits_a, bits_b, bound)",
            "tol": 1e-8,
        },
        {
            "setup": """import numpy as np
A = np.array([[-8.0,0.2],[0.4,7.0],[3.0,-2.0],[0.1,5.0]])
B = np.array([[5.0,-0.3,1.0,-2.0],[-0.2,6.0,4.0,0.5]])
bits_a, bits_b = 5, 3
bound = 0.83
""",
            "call": "certify_bounded_shared_scaling(A, B, bits_a, bits_b, bound)",
            "gold_call": "_oracle_certify_bounded_shared_scaling(A, B, bits_a, bits_b, bound)",
            "tol": 1e-9,
        },
        {
            "setup": """import numpy as np
A = np.zeros((2, 2))
B = np.ones((2, 2))
bits_a = bits_b = 2
bound = 1.0
def run_model():
    try:
        certify_bounded_shared_scaling(A, B, bits_a, bits_b, bound)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_certify_bounded_shared_scaling(A, B, bits_a, bits_b, bound)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
            "tol": 0.0,
        },
    ]
