"""
Choose which residual entries drive one enlargement pass of a column, using a threshold relative to the residual norm and a cap on the number admitted.

In residual-based pattern selection the largest residual components dominate the column residual norm, and a norm-relative threshold $|r_k(i)| \ge \delta \|r_k\|_2$ lets the number of indices admitted per pass adapt to how concentrated the residual is.

Returns
-------
np.ndarray: admitted row indices, largest $|r_k(i)|$ first, at most $c$ of them.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_residual_rows(residual: "np.ndarray", rows: "np.ndarray", used: "np.ndarray", threshold: float, cap: int) -> "np.ndarray":
    r"""Return the residual rows admitted in one enlargement pass.

    Write $r$ for ``residual``, $\delta$ for ``threshold`` and $c$ for ``cap``.
    The candidates are the entries of ``rows`` that do not appear in ``used``.
    A candidate $i$ qualifies when $\lvert r(i) \rvert \ge \delta \lVert r \rVert_2$,
    the norm being taken over the whole vector $r$. Of the qualifying
    candidates, at most $c$ are admitted: those with the largest
    $\lvert r(i) \rvert$. The admitted rows are returned in order of
    decreasing $\lvert r(i) \rvert$, ties going to the smaller row index
    first. The result may be empty.

    Parameters
    ----------
    residual : np.ndarray
        1-D float array of length ``n >= 1`` with finite entries.
    rows : np.ndarray
        1-D integer array of distinct row indices in ``[0, n)``; may be
        empty.
    used : np.ndarray
        1-D integer array of row indices in ``[0, n)`` already admitted in
        earlier passes; may be empty.
    threshold : float
        Finite nonnegative relative threshold.
    cap : int
        Maximum number of rows admitted, at least 1.

    Returns
    -------
    np.ndarray
        1-D integer array of admitted row indices, ordered as described.

    Raises
    ------
    ValueError
        If ``residual`` is not a nonempty 1-D array of finite numbers, ``rows``
        or ``used`` is not a 1-D integer array of indices in ``[0, n)``,
        ``rows`` repeats an index, ``threshold`` is not a finite nonnegative
        number, or ``cap`` is not an integer of at least 1 (booleans are
        rejected).
    """
    return admitted

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_select_residual_rows(residual: "np.ndarray", rows: "np.ndarray", used: "np.ndarray", threshold: float, cap: int) -> "np.ndarray":
    """Reference implementation (norm-relative threshold with a per-pass cap)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

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

    vector = np.asarray(residual, dtype=float)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("residual must be a nonempty 1-D array of finite numbers")
    size = vector.size
    candidates = _index_array(rows, size, "rows")
    if np.unique(candidates).size != candidates.size:
        raise ValueError("rows must not repeat an index")
    previous = _index_array(used, size, "used")
    if not (_is_number(threshold) and threshold >= 0.0):
        raise ValueError("threshold must be a finite nonnegative number")
    if not (_is_integer(cap) and cap >= 1):
        raise ValueError("cap must be an integer of at least 1")
    candidates = candidates[~np.isin(candidates, previous)]
    if candidates.size == 0:
        return np.zeros(0, dtype=int)
    level = float(threshold) * float(np.linalg.norm(vector))
    magnitudes = np.abs(vector[candidates])
    keep = magnitudes >= level
    candidates, magnitudes = candidates[keep], magnitudes[keep]
    # Primary key: decreasing magnitude; secondary key: increasing index.
    order = np.lexsort((candidates, -magnitudes))
    return candidates[order][: int(cap)].astype(int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    code = (
        "import numpy as np\n"
        "def _code(s):\n"
        "    s = np.asarray(s)\n"
        "    if s.ndim != 1 or (s.size and not np.issubdtype(s.dtype, np.integer)):\n"
        "        return -1.0\n"
        "    return float(1000.0 * s.size + np.sum((np.arange(s.size) + 1.0) * (s + 1.0) ** 2))\n"
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
                "r = np.array([0.05, -0.31, 0.12, 0.44, -0.02, 0.27, -0.38, 0.09, 0.21, -0.15, 0.33, 0.01])\n"
                "rows = np.array([1, 2, 3, 5, 6, 8, 9, 10])\n"
                "used = np.array([3])\n"
            ),
            "call": "_code(select_residual_rows(r.copy(), rows.copy(), used.copy(), 0.2, 3))",
            "gold_call": "_code(_oracle_select_residual_rows(r.copy(), rows.copy(), used.copy(), 0.2, 3))",
        },
        {
            "setup": code + (
                "r = np.array([0.05, -0.31, 0.12, 0.44, -0.02, 0.27, -0.38, 0.09, 0.21, -0.15, 0.33, 0.01])\n"
                "rows = np.arange(12)\n"
                "used = np.array([], dtype=int)\n"
            ),
            "call": "_code(select_residual_rows(r.copy(), rows.copy(), used.copy(), 0.35, 4))",
            "gold_call": "_code(_oracle_select_residual_rows(r.copy(), rows.copy(), used.copy(), 0.35, 4))",
        },
        {
            "setup": code + (
                "r = np.array([0.5, -0.1, -0.5, 0.2, 0.5, 0.0, -0.3])\n"
                "rows = np.array([6, 4, 2, 0, 3])\n"
                "used = np.array([], dtype=int)\n"
            ),
            "call": "_code(select_residual_rows(r.copy(), rows.copy(), used.copy(), 0.1, 2))",
            "gold_call": "_code(_oracle_select_residual_rows(r.copy(), rows.copy(), used.copy(), 0.1, 2))",
        },
        {
            "setup": code + (
                "r = np.array([0.0, 3.0, 0.0, -4.0, 0.0])\n"
                "rows = np.array([1, 3])\n"
                "used = np.array([], dtype=int)\n"
            ),
            "call": "_code(select_residual_rows(r.copy(), rows.copy(), used.copy(), 0.8, 3))",
            "gold_call": "_code(_oracle_select_residual_rows(r.copy(), rows.copy(), used.copy(), 0.8, 3))",
        },
        {
            "setup": code + (
                "r = np.array([-1.0, 0.0, 0.12, 0.0, -0.09, 0.2, 0.0, 0.15])\n"
                "rows = np.array([2, 4, 5, 7])\n"
                "used = np.array([], dtype=int)\n"
            ),
            "call": "_code(select_residual_rows(r.copy(), rows.copy(), used.copy(), 0.13, 3))",
            "gold_call": "_code(_oracle_select_residual_rows(r.copy(), rows.copy(), used.copy(), 0.13, 3))",
        },
        {
            "setup": code + (
                "r = np.array([0.2, -0.2, 0.2, 0.1])\n"
                "rows = np.array([0, 1, 2, 3])\n"
                "used = np.array([0, 1, 2, 3])\n"
            ),
            "call": "_code(select_residual_rows(r.copy(), rows.copy(), used.copy(), 0.0, 2))",
            "gold_call": "_code(_oracle_select_residual_rows(r.copy(), rows.copy(), used.copy(), 0.0, 2))",
        },
        {
            "setup": code + (
                "r = np.array([0.3, -0.25, 0.28, 0.02, -0.29, 0.05])\n"
                "rows = np.array([0, 1, 2, 3, 4, 5])\n"
                "used = np.array([2])\n"
            ),
            "call": "_code(select_residual_rows(r.copy(), rows.copy(), used.copy(), 0.45, 5))",
            "gold_call": "_code(_oracle_select_residual_rows(r.copy(), rows.copy(), used.copy(), 0.45, 5))",
        },
        {
            "setup": status,
            "call": "_status(lambda: select_residual_rows(np.ones(4), np.array([0, 1]), np.array([], dtype=int), 0.2, 0))",
            "gold_call": "_status(lambda: _oracle_select_residual_rows(np.ones(4), np.array([0, 1]), np.array([], dtype=int), 0.2, 0))",
        },
        {
            "setup": status,
            "call": "_status(lambda: select_residual_rows(np.ones(4), np.array([0, 4]), np.array([], dtype=int), 0.2, 2))",
            "gold_call": "_status(lambda: _oracle_select_residual_rows(np.ones(4), np.array([0, 4]), np.array([], dtype=int), 0.2, 2))",
        },
    ]
