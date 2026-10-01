"""
Return the symmetric matrix of Bray-Curtis dissimilarities between all pairs of rows of states, d(x, y) = sum_s |x_s - y_s| / sum_s (x_s + y_s), with zeros on the diagonal.

The state space of a dynamic regime is defined by the pairwise dissimilarities between all observed states; for relative-abundance data the Bray-Curtis dissimilarity is the usual choice and lies between 0 and 1.

Returns
-------
numpy.ndarray of float64 with shape (n, n): the Bray-Curtis dissimilarity matrix.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bray_curtis_matrix(states: "numpy.ndarray") -> "numpy.ndarray":
    """Return the symmetric matrix of Bray-Curtis dissimilarities between all pairs of rows of states, d(x, y) = sum_s |x_s - y_s| / sum_s (x_s + y_s), with zeros on the diagonal.

    Parameters
    ----------
    states : numpy.ndarray
        Array of shape (n, S) of nonnegative state variables; every row must have a positive total.

    Returns
    -------
    dissimilarity : numpy.ndarray
        Array of shape (n, n), symmetric with zero diagonal (float64).

    Raises
    ------
    ValueError
        If states is not a two-dimensional array with at least one row and one column, contains a non-finite or negative value, or a pair of rows has a zero total abundance.
    """
    return dissimilarity

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_bray_curtis_matrix(states: "numpy.ndarray") -> "numpy.ndarray":
    """Pairwise Bray-Curtis dissimilarity sum|x - y| / sum(x + y) between the rows of states."""
    X = np.asarray(states, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 1:
        raise ValueError("states must be a 2-D array with at least one row and one column")
    if not np.all(np.isfinite(X)) or np.any(X < 0.0):
        raise ValueError("states must be finite and nonnegative")
    num = np.abs(X[:, None, :] - X[None, :, :]).sum(axis=2)
    den = (X[:, None, :] + X[None, :, :]).sum(axis=2)
    if np.any(den <= 0.0):
        raise ValueError("every pair of rows must have a positive total abundance")
    D = num / den
    np.fill_diagonal(D, 0.0)
    return D

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nstates = _oracle_regime_states(36, 8, base, 0.35, r, K, A)\n",
            "call": "bray_curtis_matrix(states)",
            "gold_call": "_oracle_bray_curtis_matrix(states)",
        },
        {
            "setup": "import numpy as np\nstates = np.array([[0.5, 0.5, 0.0], [0.2, 0.3, 0.5], [1.0, 0.0, 0.0], [0.1, 0.1, 0.8]])\n",
            "call": "bray_curtis_matrix(states)",
            "gold_call": "_oracle_bray_curtis_matrix(states)",
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nstates = _oracle_regime_states(12, 6, base, 0.2, r, K, A)\n",
            "call": "bray_curtis_matrix(states)",
            "gold_call": "_oracle_bray_curtis_matrix(states)",
        },
        {
            "setup": "import numpy as np\nstates = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 0.0]])\ndef run_model():\n    try:\n        bray_curtis_matrix(states)\n        return 0\n    except ValueError:\n        return 1\ndef _fx_safe(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "_fx_safe(lambda: _oracle_bray_curtis_matrix(states))",
        },
    ]
