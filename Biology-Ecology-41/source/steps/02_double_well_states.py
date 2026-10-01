"""
Compute the equilibrium-proxy node states of the coupled double-well network model across a control-parameter sweep.

Each control value produces one state vector, and the collection of vectors is the training trajectory used by the sentinel reduction. For each D, integrate the node-state equations



dx_i/dt = -(x_i - 1)(x_i - 3)(x_i - 5)

          + D * sum_j A_ij x_j,



with x_i(0) = 1 for every node i.

Returns
-------
return out
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def double_well_states(A: "np.ndarray", D_grid: "np.ndarray", T: float) -> "np.ndarray":
    '''Return node states at the terminal integration time for the coupled double-well system.

    For each control value D, integrate the coupled double-well equations

    dx_i/dt = -(x_i - 1)(x_i - 3)(x_i - 5)
              + D Σ_j A_ij x_j,

    with x_i(0) = 1 for every node i. Return the node-state vector at
    terminal time T for every value in D_grid.

    Parameters
    ----------
    A : np.ndarray
        Square symmetric binary adjacency matrix of a connected simple graph.
    D_grid : np.ndarray
        Strictly increasing one-dimensional control-parameter grid in [0, 1].
    T : float
        Positive terminal time; the benchmark uses T = 15.

    Returns
    -------
    states : np.ndarray
        Array of shape (len(D_grid), N), with one terminal node-state vector
        per control value.

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


def _oracle_double_well_states(
    A: "np.ndarray",
    D_grid: "np.ndarray",
    T: float,
) -> "np.ndarray":
    A = np.asarray(A, dtype=float)
    D_grid = np.asarray(D_grid, dtype=float)
    _oracle_network_summary(A)

    if D_grid.ndim != 1 or D_grid.size < 2 or not np.all(np.isfinite(D_grid)):
        raise ValueError("D_grid must be a finite one-dimensional grid with at least two values")
    if not np.all(np.diff(D_grid) > 0.0):
        raise ValueError("D_grid must be strictly increasing")
    if D_grid[0] < 0.0 or D_grid[-1] > 1.0:
        raise ValueError("D_grid must lie in [0,1]")
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T must be positive and finite")

    N = A.shape[0]
    initial = np.ones(N, dtype=float)
    out = np.empty((D_grid.size, N), dtype=float)

    def _rhs(_t, x, D):
        return -(x - 1.0) * (x - 3.0) * (x - 5.0) + D * (A @ x)

    for k, D in enumerate(D_grid):
        sol = solve_ivp(
            _rhs,
            (0.0, float(T)),
            initial,
            args=(float(D),),
            method="DOP853",
            rtol=1e-11,
            atol=1e-13,
            max_step=0.1,
        )
        if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
            raise ValueError("double-well integration failed")
        out[k] = sol.y[:, -1]

    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return representative double-well integration cases."""
    return [
        {
            "setup": """import numpy as np
A = np.array([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]], dtype=float)
D_grid = np.linspace(0.0, 1.0, 5)
T = 15.0
""",
            "call": "double_well_states(A, D_grid, T)",
            "gold_call": "_oracle_double_well_states(A, D_grid, T)",
        },
        {
            "setup": """import numpy as np
A = np.array([[0,1],[1,0]], dtype=float)
D_grid = np.array([0.0, 1.0], dtype=float)
T = 8.0
""",
            "call": "double_well_states(A, D_grid, T)",
            "gold_call": "_oracle_double_well_states(A, D_grid, T)",
        },
        {
            "setup": """import numpy as np
A = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]], dtype=float)
D_grid = np.linspace(0.0, 1.0, 9)
T = 15.0
""",
            "call": "double_well_states(A, D_grid, T)",
            "gold_call": "_oracle_double_well_states(A, D_grid, T)",
        },
    ]
