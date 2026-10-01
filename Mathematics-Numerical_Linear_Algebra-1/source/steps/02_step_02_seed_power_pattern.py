"""
Form the a-priori sparsity pattern of one column of the approximate inverse from the low powers of the matrix.

By the Cayley-Hamilton theorem the inverse is a polynomial in the matrix, $A^{-1} = \sum_{i=0}^{m-1} c_i A^i$, so the nonzero structure of its low powers indicates where a column of the inverse carries its largest entries.

Returns
-------
np.ndarray: sorted integer positions where column $k$ of $I$, $A$ or $A^2$ is nonzero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def seed_power_pattern(matrix: "np.ndarray", column: int) -> "np.ndarray":
    r"""Return the a-priori sparsity pattern of one column of the approximate inverse.

    Write $A$ for ``matrix`` and $k$ for ``column``. The pattern of column $k$
    is the set of row positions at which column $k$ of the identity $I$,
    column $k$ of $A$, or column $k$ of the square $A^2$ is nonzero, that is
    $J_k^0 = \{\, i : I_{ik} \ne 0 \ \text{or}\ A_{ik} \ne 0 \ \text{or}\ (A^2)_{ik} \ne 0 \,\}$.
    The square is formed in floating point, so a position at which its
    products cancel exactly to zero does not belong to the pattern unless the
    identity or $A$ itself puts it there.

    Parameters
    ----------
    matrix : np.ndarray
        Float array of shape ``(n, n)``, ``n >= 1``, with finite entries.
    column : int
        Column index ``k`` with ``0 <= k < n``.

    Returns
    -------
    np.ndarray
        Sorted 1-D integer array of the distinct pattern positions.

    Raises
    ------
    ValueError
        If ``matrix`` is not a nonempty square 2-D array of finite numbers, or
        ``column`` is not an integer in ``[0, n)`` (booleans are rejected).
    """
    return pattern

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_seed_power_pattern(matrix: "np.ndarray", column: int) -> "np.ndarray":
    """Reference implementation (union of the column supports of I, A and A^2)."""
    import numpy as np

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] < 1:
        raise ValueError("matrix must be a nonempty square 2-D array")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix must have finite entries")
    size = array.shape[0]
    if not (_is_integer(column) and 0 <= column < size):
        raise ValueError("column must be an integer index of the matrix")
    k = int(column)
    first = array[:, k]
    # Column k of A @ A is A applied to column k of A.
    second = array @ first
    mask = first != 0.0
    mask |= second != 0.0
    mask[k] = True
    return np.flatnonzero(mask).astype(int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    code = (
        "import numpy as np\n"
        "def _code(p):\n"
        "    p = np.asarray(p)\n"
        "    if p.ndim != 1 or (p.size and not np.issubdtype(p.dtype, np.integer)):\n"
        "        return -1.0\n"
        "    return float(1000.0 * p.size + np.sum((np.arange(p.size) + 1.0) * (p + 1.0)))\n"
    )
    status = (
        "import numpy as np\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        {
            "setup": code + (
                "A = np.zeros((10, 10))\n"
                "for i in range(10):\n"
                "    A[i, i] = 4.0 + 0.1 * i\n"
                "    A[(i + 1) % 10, i] = -1.5\n"
                "    A[i, (i + 3) % 10] = 0.75\n"
            ),
            "call": "_code(seed_power_pattern(A.copy(), 4))",
            "gold_call": "_code(_oracle_seed_power_pattern(A.copy(), 4))",
        },
        {
            "setup": code + (
                "A = np.array([[0.0, 2.0, 0.0, 0.0, 0.0, 0.0],\n"
                "              [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],\n"
                "              [1.0, 0.0, 0.0, 0.0, 0.0, 5.0],\n"
                "              [0.0, 0.0, 0.0, 2.0, 1.0, 0.0],\n"
                "              [0.0, 0.0, 0.0, 0.0, 0.0, 4.0],\n"
                "              [0.0, 1.0, 0.0, 6.0, 0.0, 1.0]])\n"
            ),
            "call": "_code(seed_power_pattern(A.copy(), 0))",
            "gold_call": "_code(_oracle_seed_power_pattern(A.copy(), 0))",
        },
        {
            "setup": code + (
                "A = np.array([[2.0, 0.0, 0.0, 0.0, 0.0],\n"
                "              [1.0, 3.0, 0.0, 0.0, 0.0],\n"
                "              [1.0, 0.0, 3.0, 0.0, 0.0],\n"
                "              [0.0, 1.0, -1.0, 2.0, 0.0],\n"
                "              [0.0, 1.0, 1.0, 0.0, 5.0]])\n"
            ),
            "call": "_code(seed_power_pattern(A.copy(), 0))",
            "gold_call": "_code(_oracle_seed_power_pattern(A.copy(), 0))",
        },
        {
            "setup": code + "A = np.diag([1.0, -2.0, 3.0, 4.0])\n",
            "call": "_code(seed_power_pattern(A.copy(), 2))",
            "gold_call": "_code(_oracle_seed_power_pattern(A.copy(), 2))",
        },
        {
            "setup": code + "A = np.array([[-7.5]])\n",
            "call": "_code(seed_power_pattern(A.copy(), 0))",
            "gold_call": "_code(_oracle_seed_power_pattern(A.copy(), 0))",
        },
        {
            "setup": code + (
                "A = np.zeros((12, 12))\n"
                "for i in range(12):\n"
                "    A[i, i] = 3.0\n"
                "    if i >= 1:\n"
                "        A[i, i - 1] = -1.0\n"
                "    if i >= 4:\n"
                "        A[i, i - 4] = -0.5\n"
                "    if i + 2 < 12:\n"
                "        A[i, i + 2] = 0.25\n"
            ),
            "call": "_code(seed_power_pattern(A.copy(), 5))",
            "gold_call": "_code(_oracle_seed_power_pattern(A.copy(), 5))",
        },
        {
            "setup": status,
            "call": "_status(lambda: seed_power_pattern(np.eye(3), 3))",
            "gold_call": "_status(lambda: _oracle_seed_power_pattern(np.eye(3), 3))",
        },
        {
            "setup": status,
            "call": "_status(lambda: seed_power_pattern(np.ones((2, 3)), 0))",
            "gold_call": "_status(lambda: _oracle_seed_power_pattern(np.ones((2, 3)), 0))",
        },
    ]
