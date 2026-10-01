"""
The total discrepancy is the unnormalised sum of Anderson-Darling distances over all matching replicate-time pairs.

Compute the hierarchical Anderson-Darling snapshot discrepancy. Each observed replicate-time snapshot defines an interpolated empirical CDF.

Returns
-------
float, the unnormalised replicate-time discrepancy sum.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_snapshot_discrepancy(
    observed: np.ndarray,
    simulated: np.ndarray,
    n_grid: int,
) -> float:
    """Compute the unnormalised replicate-time Anderson-Darling sum.

    Parameters
    ----------
    observed : np.ndarray
        Observed snapshots with shape (time, replicate, cell).
    simulated : np.ndarray
        Simulated snapshots with matching time and replicate dimensions.
    n_grid : int
        Number of points in each interpolated empirical-CDF grid.

    Returns
    -------
    result : float
        Sum of all replicate-time Anderson-Darling distances.

    Raises
    ------
    ValueError
        If either array is not three-dimensional with matching time and
        replicate dimensions, an observed snapshot has fewer than two finite
        values, a simulated snapshot has no finite values, or n_grid is below 2.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _make_cdf_grid(x, n_grid):
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2 or np.any(~np.isfinite(x)):
        raise ValueError("each observed snapshot must contain at least two finite values")
    if int(n_grid) < 2:
        raise ValueError("n_grid must be at least 2")
    xp = np.linspace(float(np.min(x)), float(np.max(x)) + 1.0e-3, int(n_grid))
    p = np.searchsorted(np.sort(x), xp, side="left") / float(x.size)
    return xp, p


def _prepare_discrepancy(observed, n_grid):
    observed = np.asarray(observed, dtype=float)
    if observed.ndim != 3:
        raise ValueError("observed must have shape (time, replicate, cell)")
    nt, m = observed.shape[:2]
    prepared = []
    for i in range(nt):
        row = []
        for j in range(m):
            row.append(_make_cdf_grid(observed[i, j], n_grid))
        prepared.append(row)
    return prepared


def _anderson_darling_distance(cdf, y):
    xp, p = cdf
    y = np.asarray(y, dtype=float).ravel()
    if y.size < 1 or np.any(~np.isfinite(y)):
        raise ValueError("simulated snapshot must contain finite values")
    f = np.interp(np.sort(y), xp, p, left=p[0], right=p[-1])
    f = np.clip(f, 1.0e-9, 1.0 - 1.0e-9)
    n = y.size
    i = np.arange(1, n + 1, dtype=float)
    value = -n - np.sum(
        ((2.0 * i - 1.0) / n) * (np.log(f) + np.log(1.0 - f[::-1]))
    )
    return float(np.sqrt(max(value, 0.0)))


def _compute_prepared_discrepancy(simulated, prepared):
    simulated = np.asarray(simulated, dtype=float)
    if simulated.ndim != 3:
        raise ValueError("simulated must have shape (time, replicate, cell)")
    if simulated.shape[0] != len(prepared) or simulated.shape[1] != len(prepared[0]):
        raise ValueError("observed and simulated time-replicate dimensions must match")
    total = 0.0
    for i in range(simulated.shape[0]):
        for j in range(simulated.shape[1]):
            total += _anderson_darling_distance(prepared[i][j], simulated[i, j])
    return float(total)


# =============================================================================
# ORACLE SOLUTION
# =============================================================================

import math
import numpy as np
from scipy.special import ndtr, ndtri

def _oracle_compute_snapshot_discrepancy(
    observed: np.ndarray,
    simulated: np.ndarray,
    n_grid: int,
) -> float:
    if np.asarray(observed).ndim != 3 or np.asarray(simulated).ndim != 3:
        raise ValueError("observed and simulated must be three-dimensional")
    prepared = _prepare_discrepancy(observed, n_grid)
    return float(_compute_prepared_discrepancy(simulated, prepared))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic tests for the finite-sample discrepancy."""
    return [
        {
            "setup": """import numpy as np
observed = np.array([[[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]],
                     [[1.5, 2.5, 3.5, 4.5], [3.0, 5.0, 7.0, 9.0]]])
simulated = observed.copy()
""",
            "call": "compute_snapshot_discrepancy(observed, simulated, 200)",
            "gold_call": "_oracle_compute_snapshot_discrepancy(observed, simulated, 200)",
        },
        {
            "setup": """import numpy as np
observed = np.array([[[0.0, 0.5, 1.0, 1.5, 2.0]]])
simulated = np.array([[[1.0, 1.5, 2.0, 2.5, 3.0]]])
""",
            "call": "compute_snapshot_discrepancy(observed, simulated, 101)",
            "gold_call": "_oracle_compute_snapshot_discrepancy(observed, simulated, 101)",
        },
        {
            "setup": """import numpy as np
observed = np.array([[[0.2, 0.4, 0.8, 1.6], [1.0, 2.0, 4.0, 8.0]]])
simulated = np.array([[[0.1, 0.3, 0.5, 0.7, 0.9, 1.1],
                       [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]]])
""",
            "call": "compute_snapshot_discrepancy(observed, simulated, 257)",
            "gold_call": "_oracle_compute_snapshot_discrepancy(observed, simulated, 257)",
        },
        {
            "setup": """import numpy as np
def value_error_code(fn):
    try:
        fn()
    except ValueError:
        return 1.0
    return 0.0
observed = np.array([[1.0, 2.0, 3.0]])
simulated = np.array([[[1.0, 2.0, 3.0]]])
""",
            "call": "value_error_code(lambda: compute_snapshot_discrepancy(observed, simulated, 101))",
            "gold_call": "value_error_code(lambda: _oracle_compute_snapshot_discrepancy(observed, simulated, 101))",
        },
    ]
