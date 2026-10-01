"""
Retain the independent mass balance equations.

Internal mass balance can contain redundant metabolite equations. Retaining an equivalent independent system prevents a redundant constraint from obstructing the numerical state calculation.

Returns
-------
A finite ndarray of shape (rank,N+1), with retained rows in input order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def balance_rows(stoichiometry: "np.ndarray", demand: "np.ndarray") -> "np.ndarray":
    """
    stoichiometry is a finite nonempty (M,N) array, with products positive and reactants
    negative. demand is a finite length-M vector in S v = demand. Scan rows in input
    order and retain a row exactly when it increases rank, using an absolute singular-
    value threshold of 1e-10. Return retained rows with demand as the last column, shape
    (rank,N+1). Raise ValueError for invalid inputs or inconsistent equations at that
    rank threshold.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_balance_rows(
    stoichiometry: "np.ndarray", demand: "np.ndarray"
) -> "np.ndarray":
    import numpy as np

    a = np.asarray(stoichiometry, dtype=float)
    b = np.asarray(demand, dtype=float)
    if (
        a.ndim != 2
        or min(a.shape) == 0
        or b.shape != (a.shape[0],)
        or (not np.isfinite(a).all())
        or (not np.isfinite(b).all())
    ):
        raise ValueError("Finite balance matrix and aligned demand required")
    selected = []
    rank = 0
    for i in range(len(a)):
        r = np.linalg.matrix_rank(a[selected + [i]], tol=1e-10)
        if r > rank:
            selected.append(i)
            rank = r
    if np.linalg.matrix_rank(np.c_[a, b], tol=1e-10) > rank:
        raise ValueError("Inconsistent balance equations")
    return np.c_[a[selected], b[selected]].reshape(rank, a.shape[1] + 1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\n"
            "a = np.array([[-1.0, -1.0, 1.0], [1.0, 1.0, -1.0], [0.0, 0.0, 0.0]])\n"
            "b = np.array([-4.0, 4.0, 0.0])",
            "call": "balance_rows(a,b)",
            "gold_call": "_oracle_balance_rows(a,b)",
        },
        {
            "setup": "import numpy as np\na = np.eye(2)\nb = np.array([0.0, 2.0])",
            "call": "balance_rows(a,b)",
            "gold_call": "_oracle_balance_rows(a,b)",
        },
        {
            "setup": "import numpy as np\n"
            "a = np.array([[0.0, 0.0], [2.0, 4.0], [1.0, 2.0]])\n"
            "b = np.array([0.0, 6.0, 3.0])",
            "call": "balance_rows(a,b)",
            "gold_call": "_oracle_balance_rows(a,b)",
        },
        {
            "setup": "import numpy as np\n"
            "a = np.array([[1.0, 2.0], [2.0, 4.0]])\n"
            "b = np.array([1.0, 3.0])\n"
            "\n"
            "def _error_check(fn, args):\n"
            "    try:\n"
            "        fn(*args)\n"
            "    except ValueError:\n"
            "        return 1.0\n"
            "    return 0.0",
            "call": "_error_check(balance_rows, (a,b,))",
            "gold_call": "_error_check(_oracle_balance_rows, (a,b,))",
            "tol": 0.0,
        },
    ]
