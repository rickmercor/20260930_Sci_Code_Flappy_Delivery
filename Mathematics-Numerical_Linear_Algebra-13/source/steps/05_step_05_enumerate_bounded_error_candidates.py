"""
Enumerate a finite global-minimization certificate on the bounded gauge.



All nonsmooth points come from active-range switches.  Within an open cell,

the exact objective is a convex three-term exponential branch, so that cell

contributes at most one interior stationary candidate.  The returned table

contains every endpoint, every distinct switch, and every admissible

stationary point together with its branch data and derivative.

Returns
-------
finite float np.ndarray of shape (1 + 8*N,), complete sorted candidate table
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def enumerate_bounded_error_candidates(
    A: np.ndarray,
    B: np.ndarray,
    bits_a: int,
    bits_b: int,
    bound: float,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Construct the complete finite candidate table for the bounded problem.

    Raises ``ValueError`` unless ``A`` and ``B`` are finite two-dimensional
    arrays with shapes ``(m, 2)`` and ``(2, n)`` for positive ``m`` and ``n``;
    both bit widths are non-Boolean integers of at least 2; ``bound`` is a
    non-Boolean scalar that is finite and strictly positive; and ``tolerance``
    is a non-Boolean scalar that is finite, strictly positive, and smaller
    than ``bound``.

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
        Finite positive switch-clustering and candidate-deduplication distance,
        strictly smaller than ``bound``.

    Returns
    -------
    np.ndarray
        Packed vector ``[N, records]`` with eight values per sorted record:
        ``[x, kind, cell, P, Q, C, E, E_prime]``.  Kind is -1 for the lower
        endpoint, 0 for an interior switch, 1 for the upper endpoint, and 2
        for an interior stationary point.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _candidate_inputs(
    A: np.ndarray,
    B: np.ndarray,
    bits_a: int,
    bits_b: int,
    bound: float,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
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
    _oracle_compute_dither_variance_coefficient(bits_a)  # noqa: F821 - step 01
    _oracle_compute_dither_variance_coefficient(bits_b)  # noqa: F821 - step 01
    return left, right, limit, tol


def _cell_branch(
    left: np.ndarray,
    right: np.ndarray,
    bits_a: int,
    bits_b: int,
    lower: float,
    upper: float,
    probe: float,
    tolerance: float,
) -> tuple[float, float, float]:
    """Return one smooth cell's ``(P, Q, C)`` branch from the step-04 oracle."""
    tie_tolerance = min(float(tolerance), 0.25 * (float(upper) - float(lower)))
    branch = _oracle_reconstruct_active_error_branch(  # noqa: F821 - step 04
        left, right, bits_a, bits_b, float(probe), tie_tolerance
    )
    return float(branch[0]), float(branch[1]), float(branch[2])


def _candidate_record(
    x: float,
    kind: float,
    cell: int,
    coefficients: tuple[float, float, float],
) -> np.ndarray:
    positive, negative, constant = coefficients
    exp_positive = np.exp(4.0 * x)
    exp_negative = np.exp(-4.0 * x)
    value = positive * exp_positive + negative * exp_negative + constant
    derivative = 4.0 * (positive * exp_positive - negative * exp_negative)
    return np.array(
        [x, kind, float(cell), positive, negative, constant, value, derivative],
        dtype=float,
    )


def _oracle_enumerate_bounded_error_candidates(
    A: np.ndarray,
    B: np.ndarray,
    bits_a: int,
    bits_b: int,
    bound: float,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Reference enumeration of switches and cell stationary points."""
    left, right, limit, tol = _candidate_inputs(A, B, bits_a, bits_b, bound, tolerance)
    partition = _oracle_construct_two_channel_switch_partition(  # noqa: F821 - step 02
        left, right, limit, tol
    )
    count = int(round(float(partition[0])))
    switches = np.asarray(partition[1 : 1 + count], dtype=float)
    probes = np.asarray(partition[1 + count :], dtype=float)
    branch_coefficients = [
        _cell_branch(left, right, bits_a, bits_b, lower, upper, probe, tol)
        for lower, upper, probe in zip(switches[:-1], switches[1:], probes)
    ]

    records = []
    last_cell = len(branch_coefficients) - 1
    for index, point in enumerate(switches):
        cell = min(index, last_cell)
        kind = -1.0 if index == 0 else (1.0 if index == switches.size - 1 else 0.0)
        records.append(
            _candidate_record(float(point), kind, cell, branch_coefficients[cell])
        )

    for cell, ((lower, upper), coefficients) in enumerate(
        zip(zip(switches[:-1], switches[1:]), branch_coefficients)
    ):
        positive, negative, _constant = coefficients
        if positive > 0.0 and negative > 0.0:
            stationary = float(0.125 * np.log(negative / positive))
            if lower + tol < stationary < upper - tol:
                records.append(_candidate_record(stationary, 2.0, cell, coefficients))

    records.sort(key=lambda record: (record[0], record[1]))
    unique = []
    for record in records:
        if unique and abs(record[0] - unique[-1][0]) <= tol:
            if record[1] != 2.0 and unique[-1][1] == 2.0:
                unique[-1] = record
        else:
            unique.append(record)
    table = np.vstack(unique)
    result = np.concatenate(([float(table.shape[0])], table.ravel()))
    if not np.all(np.isfinite(result)):
        raise ValueError("candidate table must be finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return interior, switch-rich, mixed-bit, boundary, zero, and invalid cases."""
    return [
        {
            "setup": """import numpy as np
A = np.array([[12.0,0.5],[7.0,-0.4],[0.6,9.0],[0.8,6.0],[1.2,0.7]])
B = np.array([[0.2,0.5,-0.3,0.4],[6.0,-4.0,5.0,3.0]])
bits_a = bits_b = 2
bound = np.log(4.0)
""",
            "call": "enumerate_bounded_error_candidates(A, B, bits_a, bits_b, bound)",
            "gold_call": "_oracle_enumerate_bounded_error_candidates(A, B, bits_a, bits_b, bound)",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
A = np.array([[4.0,-4.0],[-4.0,-1.0],[2.0,4.0]])
B = np.array([[-5.0,1.0,-1.0],[-4.0,-4.0,-4.0]])
bits_a, bits_b = 2, 3
bound = 1.0
""",
            "call": "enumerate_bounded_error_candidates(A, B, bits_a, bits_b, bound)",
            "gold_call": "_oracle_enumerate_bounded_error_candidates(A, B, bits_a, bits_b, bound)",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
A = np.array([[-8.0,0.2],[0.4,7.0],[3.0,-2.0]])
B = np.array([[5.0,-0.3,1.0],[-0.2,6.0,4.0]])
bits_a, bits_b = 5, 3
bound = 0.83
""",
            "call": "enumerate_bounded_error_candidates(A, B, bits_a, bits_b, bound)",
            "gold_call": "_oracle_enumerate_bounded_error_candidates(A, B, bits_a, bits_b, bound)",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
A = np.array([[100.0,1.0],[80.0,2.0],[60.0,1.0]])
B = np.array([[0.001,0.002],[4.0,5.0]])
bits_a = bits_b = 2
bound = np.log(4.0)
""",
            "call": "enumerate_bounded_error_candidates(A, B, bits_a, bits_b, bound)",
            "gold_call": "_oracle_enumerate_bounded_error_candidates(A, B, bits_a, bits_b, bound)",
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
A = np.zeros((3, 2))
B = np.array([[1.0,0.0],[0.0,2.0]])
bits_a = bits_b = 2
bound = 0.7
""",
            "call": "enumerate_bounded_error_candidates(A, B, bits_a, bits_b, bound)",
            "gold_call": "_oracle_enumerate_bounded_error_candidates(A, B, bits_a, bits_b, bound)",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np
A = np.ones((2, 2))
B = np.ones((2, 2))
bits_a = bits_b = 2
bound = 0.0
def run_model():
    try:
        enumerate_bounded_error_candidates(A, B, bits_a, bits_b, bound)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_enumerate_bounded_error_candidates(A, B, bits_a, bits_b, bound)
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
