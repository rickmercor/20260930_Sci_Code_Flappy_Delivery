"""
Evaluate the nested two-asset normalized polynomial basis, optionally after pathwise sorting.

The source permits arbitrary non-martingale basis functions and, for its two-asset min-put experiment, improves the regression signal by sorting asset prices pathwise with the larger price first before basis evaluation. The task uses a compact nested total-degree monomial family so basis enrichment can be tracked deterministically.

Returns
-------
A NumPy matrix with `(degree + 1)(degree + 2)/2` normalized monomial columns.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def min_put_polynomial_basis(states: "np.ndarray", strike: float, degree: int, sort_state: bool = False) -> "np.ndarray":
    """Return normalized total-degree monomials for two-asset states.

    Parameters
    ----------
    states : np.ndarray
        Array of shape ``(n, 2)`` containing asset prices.
    strike : float
        Strike used to normalize moneyness as ``z_i = S_i / strike - 1``.
    degree : int
        Maximum total polynomial degree.
    sort_state : bool
        If True, sort each two-asset state in descending price order (larger
        price first) before evaluating the basis.

    Returns
    -------
    Phi : np.ndarray
        Matrix with ``(degree + 1)(degree + 2)/2`` columns. Columns are ordered
        by increasing total degree and, within a total degree, decreasing
        exponent of the first state variable.
    """
    return Phi

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _basis_exponents(degree: int):
    exponents = []
    for total in range(int(degree) + 1):
        for a in range(total, -1, -1):
            exponents.append((a, total - a))
    return exponents


def _oracle_min_put_polynomial_basis(states: "np.ndarray", strike: float, degree: int, sort_state: bool = False) -> "np.ndarray":
    x = np.asarray(states, dtype=float)
    if bool(sort_state):
        x = np.sort(x, axis=1)[:, ::-1]
    z = x / float(strike) - 1.0
    return np.column_stack([(z[:, 0] ** a) * (z[:, 1] ** b) for a, b in _basis_exponents(int(degree))])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three polynomial-basis cases."""
    return [
        {
            "setup": "import numpy as np\nstates=np.array([[90.,105.],[100.,100.],[115.,80.]]); strike=100.; degree=1; sort_state=False",
            "call": "min_put_polynomial_basis(states,strike,degree,sort_state)",
            "gold_call": "_oracle_min_put_polynomial_basis(states,strike,degree,sort_state)",
            "tol": 1e-14,
        },
        {
            "setup": "import numpy as np\nstates=np.array([[105.,90.],[80.,120.],[99.,101.]]); strike=100.; degree=2; sort_state=True",
            "call": "min_put_polynomial_basis(states,strike,degree,sort_state)",
            "gold_call": "_oracle_min_put_polynomial_basis(states,strike,degree,sort_state)",
            "tol": 1e-14,
        },
        {
            "setup": "import numpy as np\nstates=np.array([[70.,130.],[130.,70.],[100.,85.],[112.,112.]]); strike=95.; degree=4; sort_state=False",
            "call": "min_put_polynomial_basis(states,strike,degree,sort_state)",
            "gold_call": "_oracle_min_put_polynomial_basis(states,strike,degree,sort_state)",
            "tol": 1e-13,
        },
    ]
