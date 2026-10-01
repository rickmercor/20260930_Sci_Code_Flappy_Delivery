"""
Compute the anticommutation graph of the measured observables.

Two Pauli observables either commute or anticommute. Their phases do not enter this test: for binary vectors $(\boldsymbol a_i,\boldsymbol b_i)$,

$$

\Omega_{ij}=\boldsymbol a_i\cdot\boldsymbol b_j+ \boldsymbol b_i\cdot\boldsymbol a_j\pmod 2.

$$

An edge joins a pair precisely when $\Omega_{ij}=1$. Independent sets of this graph are simultaneous measurement contexts; retaining every maximal context will be necessary for constructing the projected convex set.

Returns
-------
Symmetric integer array of shape $(m,m)$; one means anticommutation and the diagonal is zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_frustration_matrix(encoded: "np.ndarray") -> "np.ndarray":
    r"""Compute the anticommutation graph of the measured observables.

    Parameters
    ----------
    encoded : np.ndarray
        Integer array of shape $(m,2n+1)$ from the encoding step, with $1\le n\le6$ and $m\ge1$.

    Returns
    -------
    adjacency : np.ndarray
        Symmetric integer array of shape $(m,m)$; one means anticommutation and the diagonal is zero.

    Raises
    ------
    ValueError
        If the array has invalid dimensions, noninteger entries, nonbinary bit blocks, or a phase inconsistent with an unsigned Hermitian Pauli word.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _checked_encoding(encoded):
    values = np.asarray(encoded)
    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] not in range(3, 14, 2)
    ):
        raise ValueError("encoded must have shape (m, 2*n+1)")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("encoding must be integral")
    n = (values.shape[1] - 1) // 2
    a, b = values[:, 1 : n + 1], values[:, n + 1 :]
    if not np.all((values[:, 1:] == 0) | (values[:, 1:] == 1)):
        raise ValueError("Pauli bits must be binary")
    if not np.array_equal(values[:, 0], np.sum(a * b, axis=1) % 4):
        raise ValueError("phase does not describe the unsigned Hermitian word")
    return values.astype(np.int64, copy=True), n


def _oracle_build_frustration_matrix(encoded: "np.ndarray") -> "np.ndarray":
    values, n = _checked_encoding(encoded)
    a, b = values[:, 1 : n + 1], values[:, n + 1 :]
    return (a @ b.T + b @ a.T) % 2

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical cases for this step."""
    return [
        {
            "setup": """import numpy as np
encoded = np.array([[0,1,0],[1,1,1],[0,0,1]])
""",
            "call": "build_frustration_matrix(encoded.copy())",
            "gold_call": "_oracle_build_frustration_matrix(encoded.copy())",
            "tol": 0,
        },
        {
            "setup": """import numpy as np
encoded = np.array([[0,1,0,0,0],[0,0,1,0,0],[0,1,1,0,0]])
""",
            "call": "build_frustration_matrix(encoded.copy())",
            "gold_call": "_oracle_build_frustration_matrix(encoded.copy())",
            "tol": 0,
        },
        {
            "setup": """import numpy as np
encoded = np.array([[0,1,0,0,0],[0,0,0,1,0],[0,0,1,0,0]])
""",
            "call": "build_frustration_matrix(encoded.copy())",
            "gold_call": "_oracle_build_frustration_matrix(encoded.copy())",
            "tol": 0,
        },
        {
            "setup": """import numpy as np
encoded = np.array([[0,1,1]])

def _raises_value_error(fn):
    try:
        fn(encoded.copy())
    except ValueError:
        return 1
    return 0
""",
            "call": "_raises_value_error(build_frustration_matrix)",
            "gold_call": "_raises_value_error(_oracle_build_frustration_matrix)",
            "tol": 0.0,
        },
    ]
