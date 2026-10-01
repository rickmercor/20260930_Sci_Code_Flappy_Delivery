"""
Compute the complete sentinel-node transfer diagnostic for a network and two nonlinear dynamics sweeps.

The returned scalar is the dimensionless test-to-training approximation-error ratio for the lowest-error sentinel set visited by the training-dynamics search. Exact-error ties are resolved lexicographically.

Returns
-------
float     The dimensionless test-to-training approximation-error ratio, evaluated at     the lowest-error sentinel set visited by the training-dynamics search.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sentinel_transfer_pipeline(A: "np.ndarray", D_grid: "np.ndarray", lambda_grid: "np.ndarray", seed: int = 123) -> float:
    '''Return the deterministic transfer-penalty ratio for the supplied network and sweeps.

    Parameters
    ----------
    A : np.ndarray
        Square symmetric binary adjacency matrix of a connected simple graph.
    D_grid : np.ndarray
        Strictly increasing double-well control grid in [0,1].
    lambda_grid : np.ndarray
        Strictly increasing SIS infection-rate grid in [0,1].
    seed : int, optional
        Non-negative random seed for the sentinel selection; the benchmark
        canonical value is 123.

    Returns
    -------
    ratio : float
        Test-dynamics approximation error divided by training-dynamics
        approximation error for the selected sentinel set.

    Raises
    ------
    ValueError
        If the graph, grids, or seed are outside the supported domain or if
        the training normalization produces a zero or negative denominator.
    '''
    return ratio

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sentinel_transfer_pipeline(A: "np.ndarray", D_grid: "np.ndarray", lambda_grid: "np.ndarray", seed: int = 123) -> float:
    summary = _oracle_network_summary(A)
    N = int(round(summary[0]))
    n = int(np.floor(np.log(N)))
    if n < 1:
        raise ValueError("network is too small for a sentinel set")

    training_states = _oracle_double_well_states(A, D_grid, 15.0)
    training_mean = _oracle_network_average(training_states)
    sentinel = _oracle_select_sentinels(training_states, training_mean, n, seed)
    training_error = _oracle_sentinel_error(training_states, training_mean, sentinel)

    test_states = _oracle_sis_states(A, lambda_grid, 15.0)
    test_error = _oracle_transfer_error(test_states, sentinel)
    return _oracle_transfer_penalty(training_error, test_error)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return whole-pipeline cases with distinct graph or parameter configurations."""
    return [
        {
            "setup": """import numpy as np
A = np.array([
[0,1,1,0,1,0],[1,0,1,1,0,0],[1,1,0,1,1,0],
[0,1,1,0,1,1],[1,0,1,1,0,1],[0,0,0,1,1,0]
], dtype=float)
D_grid = np.linspace(0.0, 1.0, 9)
lambda_grid = np.linspace(0.0, 1.0, 9)
seed = 123
""",
            "call": "sentinel_transfer_pipeline(A, D_grid, lambda_grid, seed)",
            "gold_call": "_oracle_sentinel_transfer_pipeline(A, D_grid, lambda_grid, seed)",
        },
        {
            "setup": """import numpy as np
A = np.array([[0,1,0,1,0],[1,0,1,0,1],[0,1,0,1,0],[1,0,1,0,1],[0,1,0,1,0]], dtype=float)
D_grid = np.linspace(0.0, 1.0, 7)
lambda_grid = np.linspace(0.0, 1.0, 7)
seed = 7
""",
            "call": "sentinel_transfer_pipeline(A, D_grid, lambda_grid, seed)",
            "gold_call": "_oracle_sentinel_transfer_pipeline(A, D_grid, lambda_grid, seed)",
        },
        {
            "setup": """import numpy as np
A = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]], dtype=float)
D_grid = np.linspace(0.0, 1.0, 5)
lambda_grid = np.linspace(0.0, 1.0, 5)
seed = 42
""",
            "call": "sentinel_transfer_pipeline(A, D_grid, lambda_grid, seed)",
            "gold_call": "_oracle_sentinel_transfer_pipeline(A, D_grid, lambda_grid, seed)",
        },
    ]
