"""
Nuclear-norm residual certificate from the operator-monotone log bound.

Error bounds for low-rank approximations of matrix functions are available when the function is operator monotone, as the matrix logarithm is. In that regime the approximation error is controlled in the nuclear norm by the trailing spectrum beyond the target rank, inflated by a factor depending on the oversampling. This yields an a priori certificate accompanying the estimate.

Returns
-------
float, nuclear-norm residual certificate R = (1+k/(p-1)) * sum_i log(1+lambda_i) as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def nuclear_residual_certificate(lam: np.ndarray, r: int, p: int) -> float:
    """Return the nuclear-norm residual certificate R for f(x) = log(1+x).

    The target rank is determined by the accepted Nyström rank r and the
    oversampling p. Eigenvalues are sorted internally into nonincreasing order.

    Parameters
    ----------
    lam : ndarray, shape (n,)
        Eigenvalues of A.
    r : int
        Nyström rank used by the accepted detective branch.
    p : int
        Oversampling parameter (>= 2).

    Returns
    -------
    float
        Nuclear-norm residual certificate R.

    Raises
    ------
    ValueError
        If ``p < 2``; if ``r < p``; or if ``lam`` has fewer than ``r``
        entries.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_nuclear_residual_certificate(lam: np.ndarray, r: int, p: int) -> float:
    

    lam = np.asarray(lam, dtype=float).reshape(-1)
    if p < 2:
        raise ValueError("p must be >= 2")
    if r < p:
        raise ValueError("r must be at least p")
    if lam.size < r:
        raise ValueError("lam must have length at least r")
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    k = r - p
    tail = float(np.sum(np.log1p(np.maximum(lam[k:], 0.0))))
    return float((1.0 + k / (p - 1)) * tail)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                "import numpy as np\n"
                "lam = (np.arange(1, 33, dtype=float) ** (-2)) / 1e-2"
            ),
            "call": "nuclear_residual_certificate(lam, 16, 2)",
            "gold_call": "_oracle_nuclear_residual_certificate(lam, 16, 2)",
        },
        {
            "setup": "import numpy as np\nlam = np.array([10.0, 1.0, 0.1, 0.01])",
            "call": "nuclear_residual_certificate(lam, 3, 2)",
            "gold_call": "_oracle_nuclear_residual_certificate(lam, 3, 2)",
        },
        {
            "setup": "import numpy as np\nlam = np.zeros(5)",
            "call": "nuclear_residual_certificate(lam, 2, 2)",
            "gold_call": "_oracle_nuclear_residual_certificate(lam, 2, 2)",
        },
    ]
