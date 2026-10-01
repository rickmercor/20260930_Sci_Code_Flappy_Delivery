"""
Build the matrix that sends a marginal density on the strike ladder to model prices at the ladder's quoted strikes for a single expiry, as the source assembles it around its Eq. (31). Each entry is the step 1 pricing function evaluated with the row's strike, the column's strike, and the expiry dispersion after the source's scaling by the third argument. The row set is the one Eq. (31) specifies, which is not every strike on the ladder.

Section 4.1 of the source builds this map for its homogeneous-strike toy model. Its row count and its column count differ, and Eq. (31) states the index range for each. The scaling applied to the expiry dispersion is the parameter the source introduces in Section 3.1.

Returns
-------
A float64 matrix whose column count is the ladder length and whose row count is the one Eq. (31) prescribes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def model_price_matrix(strikes: "np.ndarray", atm_var: float, eta: float) -> "np.ndarray":
    """Build the single-expiry map from a marginal density on the strike ladder to model prices at the quoted strikes.

    Parameters
    ----------
    strikes : numpy.ndarray
        One-dimensional strike ladder, strictly increasing and strictly positive, at least three points.
    atm_var : float
        Node dispersion of the expiry, in the source's Section 2 parameterisation; non-negative.
    eta : float
        The source's scaling factor applied to the node dispersion (Section 3.1); non-negative.

    Returns
    -------
    price_map : numpy.ndarray
        float64 matrix with one column per ladder point and the row set Eq. (31) prescribes.

    Raises
    ------
    ValueError
        If the ladder is not one-dimensional, strictly increasing and strictly positive, if it has fewer than three points, or if atm_var or eta is negative or not finite.
    """
    return price_map

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.stats import norm


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


def _oracle_model_price_matrix(strikes: "np.ndarray", atm_var: float, eta: float) -> "np.ndarray":
    """C[l,i] = Call(K^i, K^l, eta*V) for INTERIOR rows l = 1..N-2. Shape (N-2, N)."""
    K = _ladder(strikes)
    if K.size < 3:
        raise ValueError("the ladder needs at least three strikes to have an interior")
    av, e = float(atm_var), float(eta)
    if not np.isfinite(av) or av < 0.0:
        raise ValueError("atm_var must be finite and non-negative")
    if not np.isfinite(e) or e < 0.0:
        raise ValueError("eta must be finite and non-negative")
    rows = np.arange(1, K.size - 1)
    C = np.empty((rows.size, K.size), dtype=np.float64)
    for a, l in enumerate(rows):
        C[a, :] = _oracle_bs_call_price(K, K[l], e * av)
    return C

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            'setup': 'import numpy as np\nK = np.array([0.8, 0.9, 1.0, 1.1, 1.2])\n',
            'call': 'model_price_matrix(K, 0.05, 0.3)',
            'gold_call': '_oracle_model_price_matrix(K, 0.05, 0.3)',
        },
        {
            'setup': 'import numpy as np\nK = np.array([0.7, 0.85, 1.0, 1.15, 1.3, 1.45])\n',
            'call': 'model_price_matrix(K, 0.12, 0.4)',
            'gold_call': '_oracle_model_price_matrix(K, 0.12, 0.4)',
        },
        {
            'setup': 'import numpy as np\nK = np.array([0.9, 1.0, 1.1])\n',
            'call': 'model_price_matrix(K, 0.02, 0.6)',
            'gold_call': '_oracle_model_price_matrix(K, 0.02, 0.6)',
        },
        {
            'setup': 'import numpy as np\nK = np.array([0.8, 0.9, 1.0, 1.1, 1.2])\n',
            'call': 'model_price_matrix(K, 0.05, 0.0)',
            'gold_call': '_oracle_model_price_matrix(K, 0.05, 0.0)',
        },
        {
            'setup': 'import numpy as np\ndef run_model():\n    try:\n        model_price_matrix(np.array([1.0, 0.5]), 0.05, 0.3)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_model_price_matrix(np.array([1.0, 0.5]), 0.05, 0.3)\n        return 0\n    except ValueError:\n        return 1\n',
            'call': 'run_model()',
            'gold_call': 'run_oracle()',
        },
    ]
