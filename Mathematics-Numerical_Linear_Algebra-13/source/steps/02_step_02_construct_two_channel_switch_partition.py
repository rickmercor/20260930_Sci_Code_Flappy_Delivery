"""
Construct the exact active-range partition for a bounded two-channel gauge.



For `$H(x) = diag(exp(x), exp(-x))$`, a row range of ``A H`` or a

column range of `$H^{-1} B$` is a maximum of two exponentials.  The full

error is smooth only between their tie points.  This step returns every

distinct in-bound tie, the two endpoints, and one probe in each open cell.

Returns
-------
finite float np.ndarray of shape (2*s,), [s, sorted switches, cell midpoints]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def construct_two_channel_switch_partition(
    A: np.ndarray,
    B: np.ndarray,
    bound: float,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Build the bounded partition induced by all row and column range ties.

    Raises ``ValueError`` unless ``A`` and ``B`` are finite two-dimensional
    arrays with shapes ``(m, 2)`` and ``(2, n)`` for positive ``m`` and ``n``;
    ``bound`` is a non-Boolean scalar that is finite and strictly positive;
    and ``tolerance`` is a non-Boolean scalar that is finite, strictly
    positive, and smaller than ``bound``.

    Parameters
    ----------
    A : np.ndarray
        Finite left factor of shape ``(m, 2)``, with ``m >= 1``.
    B : np.ndarray
        Finite right factor of shape ``(2, n)``, with ``n >= 1``.
    bound : float
        Finite positive log-scale bound; the domain is ``[-bound, bound]``.
    tolerance : float, optional
        Finite positive distance used to cluster coincident tie points.  It
        must be smaller than ``bound``.

    Returns
    -------
    np.ndarray
        A finite vector ``[s, z_0, ..., z_{s-1}, p_0, ..., p_{s-2}]``.
        The ``z`` values are strictly increasing partition points and each
        ``p_j`` is the midpoint of ``(z_j, z_{j+1})``.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _validated_partition_inputs(
    A: np.ndarray,
    B: np.ndarray,
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
        raise ValueError("tolerance must be a finite positive scalar below bound")
    try:
        limit = float(bound)
        tol = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("bound and tolerance must be real scalars") from exc
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("bound must be a finite positive scalar")
    if not np.isfinite(tol) or tol <= 0.0 or tol >= limit:
        raise ValueError("tolerance must be finite, positive, and below bound")
    return left, right, limit, tol


def _cluster_switches(points: list[float], tolerance: float) -> np.ndarray:
    points.sort()
    clusters: list[list[float]] = [[points[0]]]
    for point in points[1:]:
        if point - clusters[-1][-1] <= tolerance:
            clusters[-1].append(point)
        else:
            clusters.append([point])
    return np.array([sum(group) / len(group) for group in clusters], dtype=float)


def _oracle_construct_two_channel_switch_partition(
    A: np.ndarray,
    B: np.ndarray,
    bound: float,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Reference construction of the finite active-range partition."""
    left, right, limit, tol = _validated_partition_inputs(A, B, bound, tolerance)
    points = [-limit, limit]

    for first, second in np.abs(left):
        if first > 0.0 and second > 0.0:
            switch = float(0.5 * np.log(second / first))
            if -limit <= switch <= limit:
                points.append(float(np.clip(switch, -limit, limit)))

    for first, second in np.abs(right.T):
        if first > 0.0 and second > 0.0:
            switch = float(0.5 * np.log(first / second))
            if -limit <= switch <= limit:
                points.append(float(np.clip(switch, -limit, limit)))

    switches = _cluster_switches(points, tol)
    switches[0] = -limit
    switches[-1] = limit
    if switches.size < 2 or np.any(np.diff(switches) <= 0.0):
        raise ValueError("merged partition must contain increasing endpoints")
    probes = 0.5 * (switches[:-1] + switches[1:])
    result = np.concatenate(([float(switches.size)], switches, probes))
    if not np.all(np.isfinite(result)):
        raise ValueError("partition must be finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return ordinary, coincident, zero-entry, boundary, and invalid cases."""
    return [
        {
            "setup": """import numpy as np
A = np.array([[12.0,0.5],[7.0,-0.4],[0.6,9.0],[0.8,6.0],[1.2,0.7]])
B = np.array([[0.2,0.5,-0.3,0.4],[6.0,-4.0,5.0,3.0]])
bound = np.log(4.0)
""",
            "call": "construct_two_channel_switch_partition(A, B, bound)",
            "gold_call": "_oracle_construct_two_channel_switch_partition(A, B, bound)",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np
A = np.array([[1.0,1.0],[2.0,2.0],[-3.0,3.0]])
B = np.array([[1.0,-2.0,4.0],[1.0,2.0,-4.0]])
bound = np.log(3.0)
""",
            "call": "construct_two_channel_switch_partition(A, B, bound)",
            "gold_call": "_oracle_construct_two_channel_switch_partition(A, B, bound)",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np
A = np.array([[0.0,2.0],[3.0,0.0],[0.0,0.0],[1.0,4.0]])
B = np.array([[0.0,5.0,0.0],[2.0,0.0,0.0]])
bound = 0.9
""",
            "call": "construct_two_channel_switch_partition(A, B, bound)",
            "gold_call": "_oracle_construct_two_channel_switch_partition(A, B, bound)",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np
eps = 4e-13
A = np.array([[1.0, np.exp(2.0 * eps)], [1.0, 4.0]])
B = np.array([[1.0, 2.0], [np.exp(-2.0 * eps), 0.5]])
bound = np.log(2.0)
tolerance = 1e-12
""",
            "call": "construct_two_channel_switch_partition(A, B, bound, tolerance)",
            "gold_call": "_oracle_construct_two_channel_switch_partition(A, B, bound, tolerance)",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np
A = np.array([[1.0, 4.0], [4.0, 1.0]])
B = np.array([[1.0, 4.0], [4.0, 1.0]])
bound = np.log(2.0)
""",
            "call": "construct_two_channel_switch_partition(A, B, bound)",
            "gold_call": "_oracle_construct_two_channel_switch_partition(A, B, bound)",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np
A = np.ones((2, 3))
B = np.ones((2, 2))
bound = 1.0
def run_model():
    try:
        construct_two_channel_switch_partition(A, B, bound)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_construct_two_channel_switch_partition(A, B, bound)
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
