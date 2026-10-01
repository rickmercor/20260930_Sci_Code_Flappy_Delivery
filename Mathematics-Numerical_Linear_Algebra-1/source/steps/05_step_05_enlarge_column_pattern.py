"""
Enlarge the sparsity pattern of a column so that the least-squares solution can act on the residual entries admitted in the current pass.

A residual entry can only be reduced through unknowns that couple to it, so the admitted residual entries $\hat{D}$ dictate which new positions the column pattern must acquire: every position $j$ outside $J_k$ for which $A(i, j)$ is nonzero at some admitted row $i$.

Returns
-------
np.ndarray: sorted positions of $J_k \cup \hat{J}$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def enlarge_column_pattern(matrix: "np.ndarray", pattern: "np.ndarray", selected: "np.ndarray") -> "np.ndarray":
    r"""Return the column pattern enlarged toward the selected residual entries.

    Write $A$ for ``matrix``, $J$ for ``pattern`` and $\hat{D}$ for
    ``selected``. For a column $m$ of length $n$ whose entries vanish outside
    $J$, the residual of the column is $r = A m - e_k$. Return the sorted
    union of $J$ with every position $j$ such that letting $m(j)$ be nonzero
    changes at least one residual entry $r(i)$ with $i \in \hat{D}$, that is
    $J \cup \{\, j : A(i, j) \ne 0 \ \text{for some}\ i \in \hat{D} \,\}$. An
    empty $\hat{D}$ leaves the pattern unchanged.

    Parameters
    ----------
    matrix : np.ndarray
        Float array of shape ``(n, n)``, ``n >= 1``, with finite entries.
    pattern : np.ndarray
        1-D integer array of distinct positions in ``[0, n)``, in any order.
    selected : np.ndarray
        1-D integer array of residual indices in ``[0, n)``; may be empty.

    Returns
    -------
    np.ndarray
        Sorted 1-D integer array of the distinct positions of the enlarged
        pattern.

    Raises
    ------
    ValueError
        If ``matrix`` is not a nonempty square 2-D array of finite numbers, or
        ``pattern`` or ``selected`` is not a 1-D integer array of indices in
        ``[0, n)``, or ``pattern`` repeats a position.
    """
    return enlarged

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_enlarge_column_pattern(matrix: "np.ndarray", pattern: "np.ndarray", selected: "np.ndarray") -> "np.ndarray":
    """Reference implementation (row patterns of the selected residual rows)."""
    import numpy as np

    def _index_array(values, size, name):
        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError(f"{name} must be a 1-D array")
        if values.size == 0:
            return np.zeros(0, dtype=int)
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must hold integer indices")
        values = values.astype(int)
        if values.min() < 0 or values.max() >= size:
            raise ValueError(f"{name} indices must lie in [0, n)")
        return values

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] < 1:
        raise ValueError("matrix must be a nonempty square 2-D array")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix must have finite entries")
    size = array.shape[0]
    current = _index_array(pattern, size, "pattern")
    if np.unique(current).size != current.size:
        raise ValueError("pattern must not repeat a position")
    chosen = _index_array(selected, size, "selected")
    if chosen.size == 0:
        return np.sort(current).astype(int)
    # Residual entry i equals sum_j matrix[i, j] m[j] - delta_ik, so m[j]
    # reaches entry i exactly when matrix[i, j] is nonzero: row patterns.
    reach = np.flatnonzero(np.any(array[chosen, :] != 0.0, axis=0))
    return np.union1d(current, reach).astype(int)

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
        "def _upper(n):\n"
        "    A = np.zeros((n, n))\n"
        "    for i in range(n):\n"
        "        A[i, i] = 2.0 + 0.5 * i\n"
        "        if i + 1 < n:\n"
        "            A[i, i + 1] = -1.0\n"
        "        if i + 3 < n:\n"
        "            A[i, i + 3] = 0.5\n"
        "        if i >= 5:\n"
        "            A[i, i - 5] = -0.25\n"
        "    return A\n"
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
            "setup": code + "A = _upper(12)\n",
            "call": "_code(enlarge_column_pattern(A.copy(), np.array([6, 5, 3]), np.array([2, 7])))",
            "gold_call": "_code(_oracle_enlarge_column_pattern(A.copy(), np.array([6, 5, 3]), np.array([2, 7])))",
        },
        {
            "setup": code + "A = _upper(10)\n",
            "call": "_code(enlarge_column_pattern(A.copy(), np.array([9, 0]), np.array([8])))",
            "gold_call": "_code(_oracle_enlarge_column_pattern(A.copy(), np.array([9, 0]), np.array([8])))",
        },
        {
            "setup": code + "A = _upper(8)\n",
            "call": "_code(enlarge_column_pattern(A.copy(), np.array([4, 1, 2]), np.array([], dtype=int)))",
            "gold_call": "_code(_oracle_enlarge_column_pattern(A.copy(), np.array([4, 1, 2]), np.array([], dtype=int)))",
        },
        {
            "setup": code + "A = _upper(9)\n",
            "call": "_code(enlarge_column_pattern(A.copy(), np.array([3, 4, 6, 7]), np.array([3, 4])))",
            "gold_call": "_code(_oracle_enlarge_column_pattern(A.copy(), np.array([3, 4, 6, 7]), np.array([3, 4])))",
        },
        {
            "setup": code + (
                "A = np.array([[1.0, 0.0, 0.0, 4.0, 0.0],\n"
                "              [0.0, 2.0, 0.0, 0.0, 0.0],\n"
                "              [3.0, 0.0, 1.0, 0.0, 0.0],\n"
                "              [0.0, 0.0, 0.0, 1.0, 0.0],\n"
                "              [0.0, 5.0, 0.0, 0.0, 1.0]])\n"
            ),
            "call": "_code(enlarge_column_pattern(A.copy(), np.array([1]), np.array([0, 2])))",
            "gold_call": "_code(_oracle_enlarge_column_pattern(A.copy(), np.array([1]), np.array([0, 2])))",
        },
        {
            "setup": code + "A = np.array([[3.0]])\n",
            "call": "_code(enlarge_column_pattern(A.copy(), np.array([0]), np.array([0])))",
            "gold_call": "_code(_oracle_enlarge_column_pattern(A.copy(), np.array([0]), np.array([0])))",
        },
        {
            "setup": status,
            "call": "_status(lambda: enlarge_column_pattern(np.eye(4), np.array([0, 5]), np.array([1])))",
            "gold_call": "_status(lambda: _oracle_enlarge_column_pattern(np.eye(4), np.array([0, 5]), np.array([1])))",
        },
        {
            "setup": status,
            "call": "_status(lambda: enlarge_column_pattern(np.eye(4), np.array([1, 1]), np.array([2])))",
            "gold_call": "_status(lambda: _oracle_enlarge_column_pattern(np.eye(4), np.array([1, 1]), np.array([2])))",
        },
    ]
