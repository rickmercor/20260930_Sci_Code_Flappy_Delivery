"""
Return the product coupling of two prescribed marginals, normalised to unit total mass. This is the starting iterate the source prescribes for every scheme it compares.

The product of two marginals is the unique coupling under which the two coordinates are independent, and it meets both marginal families exactly at the outset.

Returns
-------
ndarray of shape (n, m), float64: the normalised product coupling.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def product_coupling(w_source, w_target):
    """Return the product coupling of two prescribed marginals, normalised to unit total mass. This is the starting iterate the source prescribes for every scheme it compares.

    Returns
    -------
    ndarray of shape (n, m), float64: the normalised product coupling.
    """
    return np.zeros((np.asarray(w_source, dtype=float).size,
                     np.asarray(w_target, dtype=float).size), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_product_coupling(w_source, w_target):
    a = np.asarray(w_source, dtype=float)
    b = np.asarray(w_target, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("both marginals must be 1-D")
    if np.any(a < 0.0) or np.any(b < 0.0):
        raise ValueError("marginals must be non-negative")
    if not (float(a.sum()) > 0.0 and float(b.sum()) > 0.0):
        raise ValueError("both marginals must carry positive mass")
    P = np.outer(a / a.sum(), b / b.sum())
    return P / float(P.sum())

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
        {
            "setup": "import numpy as np\nx = 0.25 + 0.0375*np.arange(41)\na = np.exp(-((x-1.0)**2)/(2*0.15**2)); a/=a.sum()\nb = np.exp(-((x-1.0)**2)/(2*(0.15*np.sqrt(0.70))**2)); b/=b.sum()",
            "call": "product_coupling(a, b)",
            "gold_call": "_oracle_product_coupling(a, b)",
        },
        {
            "setup": "import numpy as np\na = np.array([1.0]); b = np.array([1.0, 3.0])",
            "call": "product_coupling(a, b)",
            "gold_call": "_oracle_product_coupling(a, b)",
        },
        {
            "setup": "import numpy as np\na = np.array([1.0, 0.0, 0.0]); b = np.array([0.0, 1.0, 0.0])",
            "call": "product_coupling(a, b)",
            "gold_call": "_oracle_product_coupling(a, b)",
        },
    ]
