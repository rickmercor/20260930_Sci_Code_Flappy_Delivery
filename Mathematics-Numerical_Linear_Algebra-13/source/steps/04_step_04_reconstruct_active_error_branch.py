"""
Reconstruct one smooth branch of the exact product-error objective.



Away from a range tie, every max-defined variance selects one channel.  The

three-term product error on that cell is exactly

`$P*exp(4*x) + Q*exp(-4*x) + C$`.  This step recovers the coefficients and a

numerical certificate at a supplied interior probe without sampling or curve

fitting.

Returns
-------
finite float np.ndarray of shape (8+m+n,), branch coefficients and active indices
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def reconstruct_active_error_branch(
    A: np.ndarray,
    B: np.ndarray,
    bits_a: int,
    bits_b: int,
    probe: float,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Recover the exact exponential branch active at ``probe``.

    Raises ``ValueError`` unless ``A`` and ``B`` are finite two-dimensional
    arrays with shapes ``(m, 2)`` and ``(2, n)`` for positive ``m`` and ``n``;
    both bit widths are non-Boolean integers of at least 2; ``tolerance`` is a
    non-Boolean scalar that is finite and strictly positive; and ``probe`` is
    a finite non-Boolean scalar whose distance to every nonzero row and column
    range tie is strictly greater than ``tolerance``.

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
    probe : float
        Finite log scale that is farther than ``tolerance`` from every
        nonzero range tie.
    tolerance : float, optional
        Finite positive tie-distance tolerance.

    Returns
    -------
    np.ndarray
        ``[P, Q, C, E, E_prime, E_second, m, n, active_A, active_B]``.
        The active arrays contain channel indices 0 or 1.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _validated_branch_problem(
    A: np.ndarray,
    B: np.ndarray,
    bits_a: int,
    bits_b: int,
    probe: float,
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
    if isinstance(probe, (bool, np.bool_)) or not np.isscalar(probe):
        raise ValueError("probe must be a finite real scalar")
    if isinstance(tolerance, (bool, np.bool_)) or not np.isscalar(tolerance):
        raise ValueError("tolerance must be a finite positive scalar")
    try:
        point = float(probe)
        tol = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("probe and tolerance must be real scalars") from exc
    if not np.isfinite(point):
        raise ValueError("probe must be finite")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    return (
        left,
        right,
        _oracle_compute_dither_variance_coefficient(bits_a),  # noqa: F821 - step 01
        _oracle_compute_dither_variance_coefficient(bits_b),  # noqa: F821 - step 01
        point,
        tol,
    )


def _reject_nonzero_ties(
    left: np.ndarray,
    right: np.ndarray,
    probe: float,
    tolerance: float,
) -> None:
    for first, second in np.abs(left):
        if first > 0.0 and second > 0.0:
            if abs(probe - 0.5 * np.log(second / first)) <= tolerance:
                raise ValueError("probe lies on a left-factor range tie")
    for first, second in np.abs(right.T):
        if first > 0.0 and second > 0.0:
            if abs(probe - 0.5 * np.log(first / second)) <= tolerance:
                raise ValueError("probe lies on a right-factor range tie")


def _oracle_reconstruct_active_error_branch(
    A: np.ndarray,
    B: np.ndarray,
    bits_a: int,
    bits_b: int,
    probe: float,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Reference algebraic reconstruction of one active error branch."""
    left, right, coefficient_a, coefficient_b, point, tol = _validated_branch_problem(
        A, B, bits_a, bits_b, probe, tolerance
    )
    _reject_nonzero_ties(left, right, point, tol)

    scales = np.exp(np.array([point, -point], dtype=float))
    active_a = np.argmax(np.abs(left * scales), axis=1)
    active_b = np.argmax(np.abs(right / scales[:, None]), axis=0)
    amplitudes_a = left[np.arange(left.shape[0]), active_a] ** 2
    amplitudes_b = right[active_b, np.arange(right.shape[1])] ** 2
    exponents_a = np.where(active_a == 0, 2, -2)
    exponents_b = np.where(active_b == 0, -2, 2)
    energy_a = np.sum(left**2, axis=0)
    energy_b = np.sum(right**2, axis=1)

    coefficients = {-4: 0.0, 0: 0.0, 4: 0.0}
    for amplitude, exponent in zip(amplitudes_a, exponents_a):
        coefficients[int(exponent - 2)] += coefficient_a * amplitude * energy_b[0]
        coefficients[int(exponent + 2)] += coefficient_a * amplitude * energy_b[1]
    for amplitude, exponent in zip(amplitudes_b, exponents_b):
        coefficients[int(exponent + 2)] += coefficient_b * amplitude * energy_a[0]
        coefficients[int(exponent - 2)] += coefficient_b * amplitude * energy_a[1]

    cross_scale = 2.0 * coefficient_a * coefficient_b
    for amplitude_a, exponent_a in zip(amplitudes_a, exponents_a):
        for amplitude_b, exponent_b in zip(amplitudes_b, exponents_b):
            coefficients[int(exponent_a + exponent_b)] += (
                cross_scale * amplitude_a * amplitude_b
            )

    positive = coefficients[4]
    negative = coefficients[-4]
    constant = coefficients[0]
    exp_positive = np.exp(4.0 * point)
    exp_negative = np.exp(-4.0 * point)
    value = positive * exp_positive + negative * exp_negative + constant
    derivative = 4.0 * (positive * exp_positive - negative * exp_negative)
    curvature = 16.0 * (positive * exp_positive + negative * exp_negative)
    result = np.concatenate(
        (
            [
                positive,
                negative,
                constant,
                value,
                derivative,
                curvature,
                float(left.shape[0]),
                float(right.shape[1]),
            ],
            active_a.astype(float),
            active_b.astype(float),
        )
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("branch reconstruction must be finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return mixed-bit, sparse, asymmetric, near-tie, constant, and invalid cases."""
    return [
        {
            "setup": """import numpy as np
A = np.array([[12.0,0.5],[7.0,-0.4],[0.6,9.0],[0.8,6.0],[1.2,0.7]])
B = np.array([[0.2,0.5,-0.3,0.4],[6.0,-4.0,5.0,3.0]])
bits_a, bits_b, probe = 2, 2, -0.8
""",
            "call": "reconstruct_active_error_branch(A, B, bits_a, bits_b, probe)",
            "gold_call": "_oracle_reconstruct_active_error_branch(A, B, bits_a, bits_b, probe)",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
A = np.array([[3.0,-0.25],[0.0,2.0],[1.5,0.0]])
B = np.array([[0.0,4.0,-1.0],[2.0,0.0,3.0]])
bits_a, bits_b, probe = 3, 5, 0.37
""",
            "call": "reconstruct_active_error_branch(A, B, bits_a, bits_b, probe)",
            "gold_call": "_oracle_reconstruct_active_error_branch(A, B, bits_a, bits_b, probe)",
            "tol": 1e-11,
        },
        {
            "setup": """import numpy as np
A = np.array([[-8.0,1.0],[0.2,-6.0],[3.0,2.0],[1.0,5.0]])
B = np.array([[5.0,-0.4,2.0],[-1.0,7.0,0.3]])
bits_a, bits_b, probe = 4, 2, -0.19
""",
            "call": "reconstruct_active_error_branch(A, B, bits_a, bits_b, probe)",
            "gold_call": "_oracle_reconstruct_active_error_branch(A, B, bits_a, bits_b, probe)",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
A = np.array([[1.0,np.exp(2e-7)],[4.0,0.25]])
B = np.array([[1.0,2.0],[np.exp(-6e-7),0.1]])
bits_a, bits_b, probe = 2, 3, 2e-7
tolerance = 1e-12
""",
            "call": "reconstruct_active_error_branch(A, B, bits_a, bits_b, probe, tolerance)",
            "gold_call": "_oracle_reconstruct_active_error_branch(A, B, bits_a, bits_b, probe, tolerance)",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
A = np.zeros((2, 2))
B = np.array([[2.0,-1.0],[0.5,3.0]])
bits_a, bits_b, probe = 2, 2, 0.4
""",
            "call": "reconstruct_active_error_branch(A, B, bits_a, bits_b, probe)",
            "gold_call": "_oracle_reconstruct_active_error_branch(A, B, bits_a, bits_b, probe)",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np
A = np.array([[1.0,1.0],[2.0,0.5]])
B = np.array([[1.0,3.0],[1.0,0.2]])
bits_a = bits_b = 2
probe = 0.0
def run_model():
    try:
        reconstruct_active_error_branch(A, B, bits_a, bits_b, probe)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_reconstruct_active_error_branch(A, B, bits_a, bits_b, probe)
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
