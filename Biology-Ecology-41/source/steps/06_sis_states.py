"""
Compute the equilibrium-proxy node states of the deterministic SIS network model across an infection-rate sweep.

The resulting state matrix supplies the test-dynamics activity curve on which the training sentinel set is evaluated.

Returns
-------
return out
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sis_states(A: "np.ndarray", lambda_grid: "np.ndarray", T: float) -> "np.ndarray":
    '''Return terminal node infection probabilities for the deterministic SIS model.

    For each infection rate λ, integrate the deterministic SIS equations

    dx_i/dt = -x_i + λ (1 - x_i) Σ_j A_ij x_j,

    with x_i(0) = 0.01 for every node i. Return the node-state vector at
    terminal time T for every value in lambda_grid.

    Parameters
    ----------
    A : np.ndarray
        Square symmetric binary adjacency matrix of a connected simple graph.
    lambda_grid : np.ndarray
        Strictly increasing one-dimensional infection-rate grid in [0, 1].
    T : float
        Positive terminal time; the benchmark uses T = 15.

    Returns
    -------
    states : np.ndarray
        Array of shape (len(lambda_grid), N), one terminal node-state vector
        per infection-rate value.

    Raises
    ------
    ValueError
        If the graph, grid, or terminal time violates the stated domain.
    '''
    return states

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp


def _oracle_sis_states(
    A: "np.ndarray",
    lambda_grid: "np.ndarray",
    T: float,
) -> "np.ndarray":
    A = np.asarray(A, dtype=float)
    lambda_grid = np.asarray(lambda_grid, dtype=float)
    _oracle_network_summary(A)

    if (
        lambda_grid.ndim != 1
        or lambda_grid.size < 2
        or not np.all(np.isfinite(lambda_grid))
    ):
        raise ValueError(
            "lambda_grid must be a finite one-dimensional grid with at least two values"
        )
    if not np.all(np.diff(lambda_grid) > 0.0):
        raise ValueError("lambda_grid must be strictly increasing")
    if lambda_grid[0] < 0.0 or lambda_grid[-1] > 1.0:
        raise ValueError("lambda_grid must lie in [0,1]")
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T must be positive and finite")

    N = A.shape[0]
    initial = np.full(N, 0.01, dtype=float)
    out = np.empty((lambda_grid.size, N), dtype=float)

    def _rhs(_t, x, lam):
        return -x + lam * (1.0 - x) * (A @ x)

    for k, lam in enumerate(lambda_grid):
        sol = solve_ivp(
            _rhs,
            (0.0, float(T)),
            initial,
            args=(float(lam),),
            method="DOP853",
            rtol=1e-11,
            atol=1e-13,
            max_step=0.1,
        )
        if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
            raise ValueError("SIS integration failed")
        out[k] = sol.y[:, -1]

    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return representative SIS integration cases."""
    return [
        {
            "setup": """import numpy as np
A = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=float)
lambda_grid = np.linspace(0.0, 1.0, 5)
T = 15.0
""",
            "call": "sis_states(A, lambda_grid, T)",
            "gold_call": "_oracle_sis_states(A, lambda_grid, T)",
        },
        {
            "setup": """import numpy as np
A = np.array([[0,1],[1,0]], dtype=float)
lambda_grid = np.array([0.0, 1.0], dtype=float)
T = 8.0
""",
            "call": "sis_states(A, lambda_grid, T)",
            "gold_call": "_oracle_sis_states(A, lambda_grid, T)",
        },
        {
            "setup": """import numpy as np
A = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]], dtype=float)
lambda_grid = np.linspace(0.0, 1.0, 9)
T = 15.0
""",
            "call": "sis_states(A, lambda_grid, T)",
            "gold_call": "_oracle_sis_states(A, lambda_grid, T)",
        },
    ]
