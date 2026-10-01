"""
Determine the numerically resolvable pole rank of the moment sequence at the requested order from the singular-value spectrum of its Hankel moment matrix, using the source's relative criterion with threshold tau.

The Hankel matrix of power moments is severely ill-conditioned, so a rule that keeps every nominal pole reproduces round-off as spurious spectral structure. The resolvable rank is the number of singular values that stand above the noise floor, and it is the order actually used for the quadrature rule.

Returns
-------
int, Resolvable rank N* with 1 <= N* <= order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def resolvable_rank(moments: "np.ndarray", order: int, tau: float) -> int:
    """Determine the numerically resolvable pole rank of the moment sequence at the requested order from the singular-value spectrum of its Hankel moment matrix, using the source's relative criterion with threshold tau.

    Parameters
    ----------
    moments : numpy.ndarray
        Finite 1-D array holding at least 2*order-1 moments mu_0, mu_1, ...
    order : int
        Positive nominal pole order N.
    tau : float
        Dimensionless threshold in (0, 1).

    Returns
    -------
    n_star : int
        Resolvable rank N* with 1 <= N* <= order.

    Raises
    ------
    ValueError
        If order is not a positive integer, moments is too short or not finite, or tau is outside (0, 1).
    """
    return n_star

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_resolvable_rank(moments: "np.ndarray", order: int, tau: float) -> int:
    """Eq (14): N* = max{n : sigma_n / sigma_1 > tau} from the SVD of the N x N Hankel matrix."""
    moments = np.asarray(moments, dtype=np.float64)
    if int(order) != order or order < 1:
        raise ValueError("order must be a positive integer")
    order = int(order)
    if moments.ndim != 1 or moments.size < 2 * order - 1 or not np.all(np.isfinite(moments)):
        raise ValueError("moments must be a finite 1-D array with at least 2*order-1 entries")
    if not (np.isfinite(tau) and 0.0 < tau < 1.0):
        raise ValueError("tau must lie in (0, 1)")
    hankel = np.array([[moments[i + j] for j in range(order)] for i in range(order)])
    sigma = np.linalg.svd(hankel, compute_uv=False)
    kept = np.nonzero(sigma / sigma[0] > tau)[0]
    return int(kept.max() + 1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nmoments = np.array([1.0, 0.0, 0.2836, 0.0, 0.0889, 0.0])\norder, tau = 3, 1e-8\n",
            "call": "resolvable_rank(moments, order, tau)",
            "gold_call": "_oracle_resolvable_rank(moments, order, tau)",
        },
        {
            "setup": "import numpy as np\nmoments = np.array([1.0, 0.0, 0.25, 0.0, 0.0625, 0.0])\norder, tau = 3, 1e-8\n",
            "call": "resolvable_rank(moments, order, tau)",
            "gold_call": "_oracle_resolvable_rank(moments, order, tau)",
        },
        {
            "setup": "import numpy as np\nmoments = np.array([1.0, 0.3, 0.5])\norder, tau = 2, 1e-6\n",
            "call": "resolvable_rank(moments, order, tau)",
            "gold_call": "_oracle_resolvable_rank(moments, order, tau)",
        },
        {
            "setup": "import numpy as np\nmoments = np.array([1.0, 0.0, 0.25])\norder, tau = 3, 1e-8\ndef run_model():\n    try:\n        resolvable_rank(moments, order, tau)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_resolvable_rank(moments, order, tau)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
