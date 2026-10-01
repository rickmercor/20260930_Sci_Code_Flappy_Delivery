"""
Build the reference (prior) joint law over (S1, V1, S2, V2, S3) by interleaving independent lognormal transition kernels.

Before any constraints are enforced, the SPX-VIX calibration scheme needs a starting ("reference") measure over the five-coordinate state (S1, V1, S2, V2, S3): a joint law obtained by treating each transition as an independent lognormal step (mean zero in log-return, variance equal to the current volatility state squared times the accrual fraction), chained across the two post-inception dates. This reference tensor is the input the augmented-Bregman calibration scheme (a later step) reconciles toward the target marginals and conditional constraints; it is not itself required to satisfy any of those constraints.

Returns
-------
pi : np.ndarray, (nS, nV, nS, nV, nS) probability tensor over (S1, V1, S2, V2, S3), nonnegative entries summing to 1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_reference_tensor(S_grid: np.ndarray, V_grid: np.ndarray, tau: float) -> np.ndarray:
    """Build the reference (prior) joint law over (S1, V1, S2, V2, S3) by
    interleaving independent lognormal transition kernels.

    Parameters
    ----------
    S_grid : np.ndarray
        1D array of nS candidate index levels.
    V_grid : np.ndarray
        1D array of nV candidate volatility levels.
    tau : float
        Accrual fraction per step (e.g. 30/365).

    Returns
    -------
    pi : np.ndarray
        (nS, nV, nS, nV, nS) array, a valid probability tensor (nonnegative,
        sums to 1) over (S1, V1, S2, V2, S3). S1's law is uniform over the
        grid; V1, V2 are drawn independently and uniformly; S2 | S1, V1 and
        S3 | S2, V2 each follow a discretized lognormal kernel with mean
        zero in log-return and variance V_i^2 * tau, renormalized over the
        S_grid.

    Raises
    ------
    ValueError
        If S_grid or V_grid is not a 1D array of length >= 1, or tau <= 0.
    """
    nS = len(S_grid)
    pi = np.ones((nS, len(V_grid), nS, len(V_grid), nS))
    pi = pi / pi.sum()  # placeholder
    return pi

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_reference_tensor(S_grid: np.ndarray, V_grid: np.ndarray, tau: float) -> np.ndarray:
    import numpy as np
    S_grid = np.asarray(S_grid, dtype=float)
    V_grid = np.asarray(V_grid, dtype=float)
    if S_grid.ndim != 1 or len(S_grid) < 1 or V_grid.ndim != 1 or len(V_grid) < 1:
        raise ValueError("S_grid and V_grid must be 1D arrays of length >= 1")
    if tau <= 0:
        raise ValueError("tau must be positive")

    def lognormal_kernel(Si, Vi):
        import numpy as np
        m = -0.5 * Vi ** 2 * tau
        s2 = Vi ** 2 * tau
        x = np.log(S_grid / Si)
        dens = np.exp(-(x - m) ** 2 / (2 * s2)) / np.sqrt(2 * np.pi * s2)
        return dens / dens.sum()

    nS, nV = len(S_grid), len(V_grid)
    pi = np.zeros((nS, nV, nS, nV, nS))
    for i0 in range(nS):
        for iv1 in range(nV):
            k1 = lognormal_kernel(S_grid[i0], V_grid[iv1])
            for i1 in range(nS):
                for iv2 in range(nV):
                    k2 = lognormal_kernel(S_grid[i1], V_grid[iv2])
                    pi[i0, iv1, i1, iv2, :] = k1[i1] * k2
    return pi / pi.sum()

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal case: a small toy 2x1 grid, exact expected tensor ---
        {
            "setup": """import numpy as np
S_grid = np.array([0.9, 1.1])
V_grid = np.array([0.2])
tau = 30 / 365
def run_model():
    return build_reference_tensor(S_grid, V_grid, tau).ravel()
def run_gold():
    return _oracle_build_reference_tensor(S_grid, V_grid, tau).ravel()
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
            "tol": 1e-6,
        },
        # --- Boundary case: single (S,V) point -- tensor must be a single 1.0 ---
        {
            "setup": """import numpy as np
S_grid = np.array([1.0])
V_grid = np.array([0.2])
tau = 30 / 365
def run_model():
    return build_reference_tensor(S_grid, V_grid, tau).ravel()
def run_gold():
    return _oracle_build_reference_tensor(S_grid, V_grid, tau).ravel()
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
            "tol": 1e-9,
        },
        # --- Edge case: tau <= 0 should raise ---
        {
            "setup": """import numpy as np
S_grid = np.array([0.9, 1.0, 1.1])
V_grid = np.array([0.2])
tau = 0.0
def run_model():
    try:
        build_reference_tensor(S_grid, V_grid, tau)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_reference_tensor(S_grid, V_grid, tau)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
