"""
Compute mass-balance residuals for aligned steady-state flux vectors.

For a stoichiometric matrix S with shape (n_metabolites, n_reactions) and an aligned matrix of net flux vectors V with shape (n_states, n_reactions), the steady-state mass-balance residuals are R = V S^T. The returned array therefore has shape (n_states, n_metabolites). A zero residual for a metabolite indicates exact steady-state mass balance for that metabolite in the corresponding state. stoichiometric_matrix and net_fluxes must be finite two-dimensional numerical arrays with at least one row and one column. Their reaction dimensions must agree. Raise ValueError if these requirements are not satisfied.

Returns
-------
2D NumPy float array of shape (n_states, n_metabolites) containing the steady-state mass-balance residuals.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_steady_state_residuals(
    stoichiometric_matrix: np.ndarray,
    net_fluxes: np.ndarray,
) -> np.ndarray:
    """
    Compute steady-state mass-balance residuals for aligned net-flux vectors.

    Parameters
    ----------
    stoichiometric_matrix : np.ndarray
        Finite array of shape (n_metabolites, n_reactions).
    net_fluxes : np.ndarray
        Finite array of shape (n_states, n_reactions).

    Returns
    -------
    np.ndarray
        Residual array of shape (n_states, n_metabolites).
    """
    return np.empty((0, 0), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_steady_state_residuals(
    stoichiometric_matrix: np.ndarray,
    net_fluxes: np.ndarray,
) -> np.ndarray:
    import numpy as np

    S = np.asarray(stoichiometric_matrix, dtype=float)
    V = np.asarray(net_fluxes, dtype=float)

    if S.ndim != 2 or S.shape[0] < 1 or S.shape[1] < 1:
        raise ValueError(
            "stoichiometric_matrix must be a non-empty two-dimensional array"
        )

    if V.ndim != 2 or V.shape[0] < 1 or V.shape[1] < 1:
        raise ValueError(
            "net_fluxes must be a non-empty two-dimensional array"
        )

    if V.shape[1] != S.shape[1]:
        raise ValueError(
            "stoichiometric_matrix and net_fluxes must have the same reaction dimension"
        )

    if not np.all(np.isfinite(S)):
        raise ValueError(
            "stoichiometric_matrix must contain only finite values"
        )

    if not np.all(np.isfinite(V)):
        raise ValueError(
            "net_fluxes must contain only finite values"
        )

    residuals = V @ S.T

    if not np.all(np.isfinite(residuals)):
        raise ValueError("computed residuals must be finite")

    return residuals.astype(float, copy=False)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for compute_steady_state_residuals."""
    return [
        {
            "setup": """import numpy as np
stoichiometric_matrix = np.array(
    [
        [-1.0,  1.0,  0.0,  0.0],
        [ 0.0, -1.0,  1.0,  0.0],
        [ 0.0,  0.0, -1.0,  1.0],
    ],
    dtype=float,
)
net_fluxes = np.array(
    [
        [3.0, 3.0, 3.0, 3.0],
        [5.0, 2.0, 2.0, 2.0],
    ],
    dtype=float,
)
""",
            "call": "compute_steady_state_residuals(stoichiometric_matrix, net_fluxes)",
            "gold_call": "_oracle_compute_steady_state_residuals(stoichiometric_matrix, net_fluxes)",
        },
        {
            "setup": """import numpy as np
stoichiometric_matrix = np.array(
    [
        [ 1.0, -2.0,  0.0,  1.0],
        [ 0.0,  1.0, -1.0, -1.0],
    ],
    dtype=float,
)
net_fluxes = np.array(
    [
        [ 1.5, -0.5, 2.0, 0.25],
        [-1.0,  3.0, 0.5, 2.00],
        [ 0.0,  0.0, 0.0, 0.00],
    ],
    dtype=float,
)

permutation = np.array([2, 0, 3, 1], dtype=int)
stoichiometric_matrix = stoichiometric_matrix[:, permutation]
net_fluxes = net_fluxes[:, permutation]
""",
            "call": "compute_steady_state_residuals(stoichiometric_matrix, net_fluxes)",
            "gold_call": "_oracle_compute_steady_state_residuals(stoichiometric_matrix, net_fluxes)",
        },
        {
            "setup": """import numpy as np
stoichiometric_matrix = np.array(
    [
        [-2.0, 1.0, 0.0],
        [ 0.0, -1.0, 2.0],
    ],
    dtype=float,
)
net_fluxes = np.zeros((3, 3), dtype=float)
""",
            "call": "compute_steady_state_residuals(stoichiometric_matrix, net_fluxes)",
            "gold_call": "_oracle_compute_steady_state_residuals(stoichiometric_matrix, net_fluxes)",
        },
        {
            "setup": """import numpy as np
stoichiometric_matrix = np.array(
    [
        [-2.0, 1.0],
    ],
    dtype=float,
)
net_fluxes = np.array(
    [
        [ 0.5,  1.0],
        [-1.5, -3.0],
    ],
    dtype=float,
)
""",
            "call": "compute_steady_state_residuals(stoichiometric_matrix, net_fluxes)",
            "gold_call": "_oracle_compute_steady_state_residuals(stoichiometric_matrix, net_fluxes)",
        },
    ]
