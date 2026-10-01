"""
Compute the scenario loss vector induced by a portfolio weight vector and a scenario-return matrix.

Portfolio losses use the upper-tail risk convention and retain the original ordering of the scenarios for all downstream calculations.

Returns
-------
The returned entries preserve the row order of the scenario data.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_loss_vector(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return one loss value per return scenario.

    Parameters
    ----------
    returns : np.ndarray
        Array of shape (S, n) containing scenario returns.
    weights : np.ndarray
        Portfolio weights of length n.

    Returns
    -------
    losses : np.ndarray
        Vector of S scenario losses in the same row order as returns.

    Raises
    ------
    ValueError
        If the dimensions are incompatible or inputs contain non-finite values.
    """
    return losses

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_loss_vector(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    R = np.asarray(returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    if R.ndim != 2 or w.ndim != 1 or R.shape[1] != w.size or R.shape[0] == 0:
        raise ValueError("incompatible return and weight dimensions")
    if np.any(~np.isfinite(R)) or np.any(~np.isfinite(w)):
        raise ValueError("inputs must be finite")
    return -R @ w

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and edge cases."""
    return [
        {"setup": "import numpy as np\nR=np.array([[1.,2.],[3.,4.],[-1.,0.]])\nw=np.array([0.4,0.6])", "call": "tuple(compute_loss_vector(R,w))", "gold_call": "tuple(_oracle_compute_loss_vector(R,w))", "tol": 1e-12},
        {"setup": "import numpy as np\nR=np.array([[0.1,-0.2]])\nw=np.array([1.,0.])", "call": "tuple(compute_loss_vector(R,w))", "gold_call": "tuple(_oracle_compute_loss_vector(R,w))", "tol": 1e-12},
        {"setup": "import numpy as np\nR=np.zeros((6,3)); w=np.array([1/3,1/3,1/3])", "call": "tuple(compute_loss_vector(R,w))", "gold_call": "tuple(_oracle_compute_loss_vector(R,w))", "tol": 1e-12},
    ]
