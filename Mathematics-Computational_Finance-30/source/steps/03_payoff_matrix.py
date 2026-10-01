"""
Build the square map the source uses in Section 4.1 to read a marginal density's own discrete call values off the strike ladder. Row index is the strike being valued, column index the ladder point carrying mass. Section 4.1 states which valuation this map applies, and it is not the one step 2 uses.

The source needs these values to compare adjacent expiries. Section 4.1 defines the map for the toy model; the production model later writes the same map with a switch between this valuation and a priced one, and the source states that strict absence of arbitrage in time needs the one Section 4.1 states.

Returns
-------
A float64 square matrix of side equal to the ladder length.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def payoff_matrix(strikes: "np.ndarray") -> "np.ndarray":
    """Build the square map from a marginal density to that density's own discrete call values on the ladder.

    Parameters
    ----------
    strikes : numpy.ndarray
        One-dimensional strike ladder, strictly increasing and strictly positive.

    Returns
    -------
    ordering_map : numpy.ndarray
        float64 square matrix of side equal to the ladder length.

    Raises
    ------
    ValueError
        If the ladder is not one-dimensional, not strictly increasing, not strictly positive, or contains a non-finite value.
    """
    return ordering_map

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _ladder(strikes):
    """Validate and return a strike ladder. Shared by several steps."""
    K = np.asarray(strikes, dtype=np.float64)
    if K.ndim != 1 or K.size == 0:
        raise ValueError("strikes must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(K)):
        raise ValueError("strikes must all be finite")
    if np.any(K <= 0.0):
        raise ValueError("strikes must be strictly positive")
    if K.size > 1 and not np.all(np.diff(K) > 0.0):
        raise ValueError("strikes must be strictly increasing")
    return K


def _oracle_payoff_matrix(strikes: "np.ndarray") -> "np.ndarray":
    """U[l,i] = (K^i - K^l)+ . Intrinsic payoff, ALL rows. Shape (N, N)."""
    K = _ladder(strikes)
    U = np.maximum(K[None, :] - K[:, None], 0.0).astype(np.float64)
    if U.shape[0] != U.shape[1]:
        raise ValueError("the ordering map must be square on the ladder")
    return U

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            'setup': 'import numpy as np\nK = np.array([0.8, 0.9, 1.0, 1.1, 1.2])\n',
            'call': 'payoff_matrix(K)',
            'gold_call': '_oracle_payoff_matrix(K)',
        },
        {
            'setup': 'import numpy as np\nK = np.array([0.5, 1.0, 1.5])\n',
            'call': 'payoff_matrix(K)',
            'gold_call': '_oracle_payoff_matrix(K)',
        },
        {
            'setup': 'import numpy as np\nK = np.array([1.0])\n',
            'call': 'payoff_matrix(K)',
            'gold_call': '_oracle_payoff_matrix(K)',
        },
        {
            'setup': 'import numpy as np\nK = np.linspace(0.6, 1.4, 9)\n',
            'call': 'payoff_matrix(K)',
            'gold_call': '_oracle_payoff_matrix(K)',
        },
        {
            'setup': 'import numpy as np\ndef run_model():\n    try:\n        payoff_matrix(np.array([1.0, 0.5]))\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_payoff_matrix(np.array([1.0, 0.5]))\n        return 0\n    except ValueError:\n        return 1\n',
            'call': 'run_model()',
            'gold_call': 'run_oracle()',
        },
    ]
