"""
Carry out the pattern update of one enlargement pass of a column: pick the admissible residual rows of the current reduced problem, grow the pattern through them and record them as admitted.

Each pass looks only at the candidate set $D_k^l$, the rows of the current reduced problem $I_k^l$ that the admission history $R_k^l$ does not already contain, so the residual steers the pattern toward the couplings it still lacks without revisiting rows already exploited.

Returns
-------
tuple: (sorted enlarged pattern $J_k^{l+1}$, sorted admitted-row set $R_k^{l+1}$ after the pass).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def advance_column_pattern(matrix: "np.ndarray", pattern: "np.ndarray", residual: "np.ndarray", used: "np.ndarray", threshold: float, cap: int) -> tuple:
    r"""Return the support and admission history of one adaptive transition.

    Write $A$ for ``matrix``, $J$ for ``pattern``, $r$ for ``residual``, $U$
    for ``used``, $\delta$ for ``threshold`` and $c$ for ``cap``. The residual
    is $r = A m - e_k$ for a column supported on $J$. The reduced problem's
    row set is $I = \{\, i : A(i, J) \ne 0 \,\}$, the rows on which the linear
    map $A[:, J]$ can be nonzero. Admissions are drawn from $I \setminus U$.
    Their residual magnitudes must satisfy
    $\lvert r(i) \rvert \ge \delta \lVert r \rVert_2$, where the norm includes
    every component of $r$. The admission set $\hat{D}$ contains the first $c$
    qualifying rows in descending residual magnitude, with smaller global row
    index breaking ties, or all of them if fewer qualify.

    The returned column support is the smallest superset of $J$ containing
    every coordinate of $m$ that can affect $r$ at an index in $\hat{D}$, that
    is $J \cup \{\, j : A(i, j) \ne 0 \ \text{for some}\ i \in \hat{D} \,\}$.
    The returned history is $U \cup \hat{D}$. History records admission
    independently of support growth. Thus an empty $\hat{D}$ leaves both sets
    unchanged, while a nonempty $\hat{D}$ is recorded even when the support
    stays the same. The row indices in $I$, $\hat{D}$ and $U$ are indices of
    $A$, not positions within a reduced vector. Both output sets are
    represented in increasing index order.

    Parameters
    ----------
    matrix : np.ndarray
        Float array of shape ``(n, n)``, ``n >= 1``, with finite entries.
    pattern : np.ndarray
        1-D integer array of distinct positions in ``[0, n)``, at least one.
    residual : np.ndarray
        Float array of length ``n`` with finite entries.
    used : np.ndarray
        1-D integer array of distinct row indices in ``[0, n)``; may be empty.
    threshold : float
        Finite nonnegative relative admission threshold.
    cap : int
        Maximum number of rows admitted, at least 1.

    Returns
    -------
    tuple
        ``(new_pattern, new_used)``: two sorted 1-D integer arrays of distinct
        indices.

    Raises
    ------
    ValueError
        If ``matrix`` is not a nonempty square 2-D array of finite numbers,
        ``pattern`` or ``used`` is not a 1-D integer array of distinct indices
        in ``[0, n)``, ``pattern`` is empty, ``residual`` does not have length
        ``n`` or has a non-finite entry, ``threshold`` is not a finite
        nonnegative number, or ``cap`` is not an integer of at least 1
        (booleans are rejected).
    """
    return (new_pattern, new_used)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_advance_column_pattern(matrix: "np.ndarray", pattern: "np.ndarray", residual: "np.ndarray", used: "np.ndarray", threshold: float, cap: int) -> tuple:
    """Reference implementation (select on the reduced rows, enlarge, record)."""
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
        if values.min() < 0 or values.max() >= size or np.unique(values).size != values.size:
            raise ValueError(f"{name} must hold distinct indices in [0, n)")
        return values

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] < 1:
        raise ValueError("matrix must be a nonempty square 2-D array")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix must have finite entries")
    size = array.shape[0]
    current = _index_array(pattern, size, "pattern")
    if current.size == 0:
        raise ValueError("pattern must not be empty")
    previous = _index_array(used, size, "used")
    vector = np.asarray(residual, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError("residual must be a finite vector of length n")
    # Candidate rows are the rows of the current reduced problem (Eq. 11).
    rows = np.flatnonzero(np.any(array[:, current] != 0.0, axis=1))
    chosen = _oracle_select_residual_rows(vector, rows, previous, threshold, cap)
    if chosen.size == 0:
        return np.sort(current).astype(int), np.sort(previous).astype(int)
    enlarged = _oracle_enlarge_column_pattern(array, current, chosen)
    admitted = np.union1d(previous, chosen).astype(int)
    return enlarged, admitted

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    common = (
        "import numpy as np\n"
        "def _code(p):\n"
        "    p = np.asarray(p)\n"
        "    if p.ndim != 1 or (p.size and not np.issubdtype(p.dtype, np.integer)):\n"
        "        return -1.0e6\n"
        "    return 1000.0 * p.size + np.sum((np.arange(p.size) + 1.0) * (p + 1.0))\n"
        "def _two(t):\n"
        "    if not isinstance(t, tuple) or len(t) != 2:\n"
        "        return -1.0\n"
        "    return float(_code(t[0]) / 1000.0 + _code(t[1]) / 1.0e6)\n"
        "def _net(n, offsets, seed):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    A = np.diag(3.0 + rng.random(n))\n"
        "    for d in offsets:\n"
        "        for i in range(n):\n"
        "            j = i + d\n"
        "            if 0 <= j < n:\n"
        "                A[i, j] = rng.uniform(-1.5, 1.5)\n"
        "    return A\n"
        "def _state(A, k, J):\n"
        "    I = np.flatnonzero(np.any(A[:, J] != 0, axis=1))\n"
        "    m = np.zeros(A.shape[0])\n"
        "    m[J] = np.linalg.lstsq(A[np.ix_(I, J)], (I == k).astype(float), rcond=None)[0]\n"
        "    r = A @ m\n"
        "    r[k] -= 1.0\n"
        "    return r\n"
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
            "setup": common + (
                "A = _net(24, (-1, 2, -5, 7), 11)\n"
                "J = np.array([2, 3, 8, 9, 10, 11, 16])\n"
                "r = _state(A, 9, J)\n"
            ),
            "call": "_two(advance_column_pattern(A.copy(), J.copy(), r.copy(), np.array([], dtype=int), 0.2, 3))",
            "gold_call": "_two(_oracle_advance_column_pattern(A.copy(), J.copy(), r.copy(), np.array([], dtype=int), 0.2, 3))",
        },
        {
            "setup": common + (
                "A = _net(24, (-1, 2, -5, 7), 11)\n"
                "J = np.array([2, 3, 8, 9, 10, 11, 16])\n"
                "r = _state(A, 9, J)\n"
                "U = np.array([1, 4, 8, 9])\n"
            ),
            "call": "_two(advance_column_pattern(A.copy(), J.copy(), r.copy(), U.copy(), 0.1, 2))",
            "gold_call": "_two(_oracle_advance_column_pattern(A.copy(), J.copy(), r.copy(), U.copy(), 0.1, 2))",
        },
        {
            "setup": common + (
                "A = _net(20, (1, -3, 4), 5)\n"
                "J = np.array([0, 1, 3, 4, 5])\n"
                "r = _state(A, 0, J)\n"
            ),
            "call": "_two(advance_column_pattern(A.copy(), J.copy(), r.copy(), np.array([], dtype=int), 0.9, 3))",
            "gold_call": "_two(_oracle_advance_column_pattern(A.copy(), J.copy(), r.copy(), np.array([], dtype=int), 0.9, 3))",
        },
        {
            "setup": common + (
                "A = np.array([[4.0, 0.0, 0.0, 0.0, 0.0, 0.0],\n"
                "              [-1.0, 4.0, 0.0, 0.0, 0.0, 0.0],\n"
                "              [0.0, -1.0, 4.0, 0.0, 0.0, 0.0],\n"
                "              [0.0, 0.0, -1.0, 4.0, 0.0, 0.0],\n"
                "              [0.0, 0.0, 0.0, -1.0, 4.0, 0.0],\n"
                "              [0.0, 0.0, 0.0, 0.0, -1.0, 4.0]])\n"
                "J = np.array([0, 1, 2])\n"
                "r = np.array([0.9, 0.1, 0.05, 0.02, 0.0, 0.0])\n"
            ),
            "call": "_two(advance_column_pattern(A.copy(), J.copy(), r.copy(), np.array([], dtype=int), 0.5, 1))",
            "gold_call": "_two(_oracle_advance_column_pattern(A.copy(), J.copy(), r.copy(), np.array([], dtype=int), 0.5, 1))",
        },
        {
            "setup": common + (
                "A = np.array([[4.0, 0.0, 0.0, 0.0, 0.0, 0.0],\n"
                "              [-1.0, 4.0, 0.0, 0.0, 0.0, 0.0],\n"
                "              [0.0, -1.0, 4.0, 0.0, 0.0, 0.0],\n"
                "              [0.0, 0.0, -1.0, 4.0, 0.0, 0.0],\n"
                "              [0.0, 0.0, 0.0, -1.0, 4.0, 0.0],\n"
                "              [0.0, 0.0, 0.0, 0.0, -1.0, 4.0]])\n"
                "J = np.array([0, 1, 2])\n"
                "r = np.array([3.0, 0.0, 0.0, -4.0, 0.0, 0.0])\n"
            ),
            "call": "_two(advance_column_pattern(A.copy(), J.copy(), r.copy(), np.array([1], dtype=int), 0.6, 3))",
            "gold_call": "_two(_oracle_advance_column_pattern(A.copy(), J.copy(), r.copy(), np.array([1], dtype=int), 0.6, 3))",
        },
        {
            "setup": common + (
                "A = np.zeros((8, 8))\n"
                "for i in range(8):\n"
                "    A[(i + 1) % 8, i] = 2.0 + 0.1 * i\n"
                "J = np.array([3, 4, 5])\n"
                "r = _state(A, 3, J)\n"
            ),
            "call": "_two(advance_column_pattern(A.copy(), J.copy(), r.copy(), np.array([], dtype=int), 0.2, 3))",
            "gold_call": "_two(_oracle_advance_column_pattern(A.copy(), J.copy(), r.copy(), np.array([], dtype=int), 0.2, 3))",
        },
        {
            "setup": common + (
                "A = _net(16, (-1, 1, 4, -4), 29)\n"
                "J = np.array([6, 7, 8, 11])\n"
                "r = _state(A, 7, J)\n"
                "U = np.array([2, 3])\n"
            ),
            "call": "_two(advance_column_pattern(A.copy(), J.copy(), r.copy(), U.copy(), 0.0, 4))",
            "gold_call": "_two(_oracle_advance_column_pattern(A.copy(), J.copy(), r.copy(), U.copy(), 0.0, 4))",
        },
        {
            "setup": status,
            "call": "_status(lambda: advance_column_pattern(np.eye(4), np.array([], dtype=int), np.ones(4), np.array([], dtype=int), 0.2, 2))",
            "gold_call": "_status(lambda: _oracle_advance_column_pattern(np.eye(4), np.array([], dtype=int), np.ones(4), np.array([], dtype=int), 0.2, 2))",
        },
        {
            "setup": status,
            "call": "_status(lambda: advance_column_pattern(np.eye(4), np.array([1]), np.ones(3), np.array([], dtype=int), 0.2, 2))",
            "gold_call": "_status(lambda: _oracle_advance_column_pattern(np.eye(4), np.array([1]), np.ones(3), np.array([], dtype=int), 0.2, 2))",
        },
    ]
