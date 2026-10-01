"""
Compute the natural-log forward weights of every partial global alignment of a coded sequence x with a coded sequence y under a three-state pair model.

A pair model sums the weights of all alignments of two sequences with a dynamic program over prefix pairs, and for sequences of thousands of bases, those weights leave the range of double-precision numbers unless they are carried as logarithms.

Returns
-------
np.ndarray: shape (3, L + 1, K + 1), natural-log forward weights for states M, X, Y.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_log_forward_tables(
    x: "np.ndarray",
    y: "np.ndarray",
    match_factors: "np.ndarray",
    x_gap_factors: "np.ndarray",
    y_gap_factors: "np.ndarray",
    gap_open: float,
    gap_extend: float,
) -> "np.ndarray":
    """Return natural-log forward weights of the pair model on every lattice point.

    Bases are coded 0 to 3. An alignment column is in state M (base ``a``
    of ``x`` against base ``b`` of ``y``, factor ``match_factors[a, b]``),
    X (base ``a`` of ``x`` against a gap, factor ``x_gap_factors[a]``) or Y
    (base ``b`` of ``y`` against a gap, factor ``y_gap_factors[b]``). An M
    column carries only its emission factor whatever the previous state. An
    X or Y column carries ``gap_open`` times its emission factor when the
    previous column is M and ``gap_extend`` times it when the previous column
    is the same gap state; X never follows Y and Y never follows X. The first
    column is scored as though it followed M, every base of both sequences is
    used, and an alignment may end in any state.

    Entry ``[s, i, j]`` is the natural log of the summed weight of all
    partial alignments of the first ``i`` bases of ``x`` with the first
    ``j`` bases of ``y`` whose last column is in state ``s`` (0 = M, 1 = X,
    2 = Y), or ``-inf`` when there are none. The empty alignment counts as
    an M column: entry ``[0, 0, 0]`` is 0 and entries ``[1, 0, 0]`` and
    ``[2, 0, 0]`` are ``-inf``. Values must stay exact to double precision
    even when the summed weights are far below the smallest positive float.

    Parameters
    ----------
    x, y : np.ndarray
        Non-empty one-dimensional integer arrays of codes 0 to 3, lengths
        ``L`` and ``K``.
    match_factors : np.ndarray
        Shape ``(4, 4)``, finite and positive.
    x_gap_factors, y_gap_factors : np.ndarray
        Shape ``(4,)``, finite and positive.
    gap_open, gap_extend : float
        Finite positive transition factors.

    Returns
    -------
    np.ndarray
        Float array of shape ``(3, L + 1, K + 1)``.

    Raises
    ------
    ValueError
        If a sequence is empty, not one-dimensional, not of integer type or
        holds a code outside 0 to 3, if a factor array has the wrong shape,
        or if any factor is not finite and positive.
    """
    return log_forward

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_log_forward_tables(
    x: "np.ndarray",
    y: "np.ndarray",
    match_factors: "np.ndarray",
    x_gap_factors: "np.ndarray",
    y_gap_factors: "np.ndarray",
    gap_open: float,
    gap_extend: float,
) -> "np.ndarray":
    """Reference implementation sweeping anti-diagonals in the log domain."""
    import numpy as np

    xs, ys, log_m, log_x, log_y, log_open, log_extend = _validate_pair_model(
        x, y, match_factors, x_gap_factors, y_gap_factors, gap_open, gap_extend)
    n, m = xs.size, ys.size
    table = np.full((3, n + 1, m + 1), -np.inf)
    fm, fx, fy = table  # views into the returned table
    fm[0, 0] = 0.0
    # Every cell on anti-diagonal d depends only on diagonals d - 1 and d - 2.
    for d in range(1, n + m + 1):
        i = np.arange(max(0, d - m), min(n, d) + 1)
        j = d - i
        keep = (i > 0) & (j > 0)
        a, b = i[keep], j[keep]
        fm[a, b] = log_m[a - 1, b - 1] + np.logaddexp(
            np.logaddexp(fm[a - 1, b - 1], fx[a - 1, b - 1]), fy[a - 1, b - 1])
        keep = i > 0
        a, b = i[keep], j[keep]
        fx[a, b] = log_x[a - 1] + np.logaddexp(log_open + fm[a - 1, b], log_extend + fx[a - 1, b])
        keep = j > 0
        a, b = i[keep], j[keep]
        fy[a, b] = log_y[b - 1] + np.logaddexp(log_open + fm[a, b - 1], log_extend + fy[a, b - 1])
    return table


def _validate_pair_model(x, y, match_factors, x_gap_factors, y_gap_factors, gap_open, gap_extend):
    """Return validated codes and the natural-log factors of the pair model."""
    import numpy as np

    xs, ys = (np.asarray(s) for s in (x, y))
    for name, a in (("x", xs), ("y", ys)):
        if a.ndim != 1 or a.size == 0 or not np.issubdtype(a.dtype, np.integer) or a.min() < 0 or a.max() > 3:
            raise ValueError(f"{name} must be a non-empty 1-D integer array of codes 0 to 3")
    em, ex, ey = (np.asarray(v, dtype=float) for v in (match_factors, x_gap_factors, y_gap_factors))
    if (em.shape, ex.shape, ey.shape) != ((4, 4), (4,), (4,)):
        raise ValueError("factor arrays must have shapes (4, 4), (4,) and (4,)")
    every = np.concatenate([em.ravel(), ex, ey, np.array([gap_open, gap_extend], dtype=float)])
    if not np.all(np.isfinite(every)) or np.any(every <= 0.0):
        raise ValueError("every factor must be finite and positive")
    xs, ys = xs.astype(np.int64), ys.astype(np.int64)
    return (xs, ys, np.log(em)[xs][:, ys], np.log(ex)[xs], np.log(ey)[ys],
            float(np.log(every[-2])), float(np.log(every[-1])))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    model = (
        "import numpy as np\n"
        "M = np.array([[0.16, 0.03, 0.05, 0.03], [0.03, 0.16, 0.03, 0.05],\n"
        "              [0.05, 0.03, 0.16, 0.03], [0.03, 0.05, 0.03, 0.16]])\n"
        "EX = np.array([0.12, 0.38, 0.38, 0.12])\n"
        "EY = np.array([0.16, 0.34, 0.34, 0.16])\n"
        "def _pin(table):\n"
        "    a = np.asarray(table, dtype=float)\n"
        "    if a.ndim != 3:\n"
        "        return -1.0\n"
        "    f = np.where(np.isfinite(a), a, -9000.0).ravel()\n"
        "    r = np.arange(1.0, f.size + 1.0)\n"
        "    shape = a.shape[0] * 1.0e6 + a.shape[1] * 1.0e3 + a.shape[2]\n"
        "    return float(shape + np.sum(np.sin(0.37 * r) * f) + np.sum(np.cos(0.11 * r) * f) / 7.0)\n"
    )

    def _status(args, extra=""):
        return {
            "setup": model + extra + (
                "def _candidate():\n"
                "    try:\n"
                f"        compute_log_forward_tables({args})\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "def _reference():\n"
                "    try:\n"
                f"        _oracle_compute_log_forward_tables({args})\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        }

    return [
        {  # Normal pair-HMM lattice.
            "setup": model,
            "call": "_pin(compute_log_forward_tables(np.array([0, 2, 1, 3, 3, 0, 1]), np.array([0, 2, 2, 3, 0, 1]), M.copy(), EX.copy(), EY.copy(), 0.04, 0.4))",
            "gold_call": "_pin(_oracle_compute_log_forward_tables(np.array([0, 2, 1, 3, 3, 0, 1]), np.array([0, 2, 2, 3, 0, 1]), M.copy(), EX.copy(), EY.copy(), 0.04, 0.4))",
        },
        {  # Numerically extreme but finite weights.
            "setup": model,
            "call": "_pin(compute_log_forward_tables(np.array([2, 2, 1, 0, 3]), np.array([2, 1, 1, 0, 3, 3]), M * 1e-150, EX * 1e-150, EY * 1e-150, 1e-120, 0.5))",
            "gold_call": "_pin(_oracle_compute_log_forward_tables(np.array([2, 2, 1, 0, 3]), np.array([2, 1, 1, 0, 3, 3]), M * 1e-150, EX * 1e-150, EY * 1e-150, 1e-120, 0.5))",
        },
        {  # Boundary: one residue in each sequence.
            "setup": model,
            "call": "_pin(compute_log_forward_tables(np.array([2]), np.array([1]), M.copy(), EX.copy(), EY.copy(), 0.3, 0.6))",
            "gold_call": "_pin(_oracle_compute_log_forward_tables(np.array([2]), np.array([1]), M.copy(), EX.copy(), EY.copy(), 0.3, 0.6))",
        },
        {  # Edge: strongly unbalanced sequence lengths.
            "setup": model,
            "call": "_pin(compute_log_forward_tables(np.array([1]), np.array([0, 1, 2, 3, 3]), M.copy(), EX.copy(), EY.copy(), 0.2, 0.9))",
            "gold_call": "_pin(_oracle_compute_log_forward_tables(np.array([1]), np.array([0, 1, 2, 3, 3]), M.copy(), EX.copy(), EY.copy(), 0.2, 0.9))",
        },
        {  # Edge: asymmetric emission scaling.
            "setup": model,
            "call": "_pin(compute_log_forward_tables(np.array([3, 0, 0, 1, 2, 2, 1, 0]), np.array([3, 0]), M.copy(), (EX * 2.0).copy(), EY.copy(), 0.5, 0.25))",
            "gold_call": "_pin(_oracle_compute_log_forward_tables(np.array([3, 0, 0, 1, 2, 2, 1, 0]), np.array([3, 0]), M.copy(), (EX * 2.0).copy(), EY.copy(), 0.5, 0.25))",
        },
        _status("np.array([0, 4, 1]), np.array([0, 1]), M.copy(), EX.copy(), EY.copy(), 0.04, 0.4"),
        _status("np.array([0, 1]), np.array([0, 1]), M.copy(), EX.copy(), EY.copy(), 0.04, 0.0"),
    ]
