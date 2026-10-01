"""
Evaluate the complete quadratic monomial basis of Eq (49) at each row of the (n, 2) array xi of bond vectors, with both coordinates scaled by the horizon delta before any power is taken. Return an (n, 6) array in the source's ordering: constant, x, y, x squared, xy, y squared. Raise ValueError if xi is not a non-empty (n, 2) array or if delta is not positive.

The nodal influence weight is approximated within each neighbourhood as a polynomial field in the bond vector. The same monomials serve as the test functions against which the discrete operators are later matched.

Returns
-------
ndarray of float64 with shape (n, 6).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def scaled_monomial_basis(xi, delta):
    """ndarray of float64 with shape (n, 6)."""
    return np.zeros((np.shape(xi)[0], 6), dtype=np.float64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_scaled_monomial_basis(xi, delta):
    """Eq (49): the complete quadratic monomial basis with coordinates scaled by delta.

    Returns an (n, 6) array with columns [1, x/d, y/d, x^2/d^2, xy/d^2, y^2/d^2].
    """
    import numpy as np
    xi = np.asarray(xi, dtype=np.float64)
    if xi.ndim != 2 or xi.shape[1] != 2 or xi.shape[0] < 1:
        raise ValueError("xi must be a non-empty (n, 2) array")
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    x = xi[:, 0] / float(delta)
    y = xi[:, 1] / float(delta)
    return np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nxi = np.array([[0.25, 0.0], [0.0, -0.25], [0.125, 0.125]])\ndelta = 0.5",
            "call": "scaled_monomial_basis(xi, delta)",
            "gold_call": "_oracle_scaled_monomial_basis(xi, delta)",
        },
        {
            "setup": "import numpy as np\nxi = np.zeros((1, 2))\ndelta = 0.3",
            "call": "scaled_monomial_basis(xi, delta)",
            "gold_call": "_oracle_scaled_monomial_basis(xi, delta)",
        },
        {
            "setup": "import numpy as np\nxi = np.array([[0.75, -0.5], [-0.25, 0.625]])\ndelta = 1.25",
            "call": "scaled_monomial_basis(xi, delta)",
            "gold_call": "_oracle_scaled_monomial_basis(xi, delta)",
        },
    ]
