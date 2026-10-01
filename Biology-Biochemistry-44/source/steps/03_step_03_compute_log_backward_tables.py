"""
Compute the natural logs of the derivatives of the total alignment weight with respect to every forward weight of the pair model.

The weight of all completions of a partial alignment is the sensitivity of the total weight to that partial alignment's forward weight, so the backward table reuses the forward computation in reverse and supplies the context against which any local change can be scored.

Returns
-------
np.ndarray: shape (3, L + 1, K + 1), natural-log derivatives of the total weight for states M, X, Y.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_log_backward_tables(
    x: "np.ndarray",
    y: "np.ndarray",
    match_factors: "np.ndarray",
    x_gap_factors: "np.ndarray",
    y_gap_factors: "np.ndarray",
    gap_open: float,
    gap_extend: float,
) -> "np.ndarray":
    """Return natural-log derivatives of the total weight with respect to forward weights.

    The model, the coding of bases and states and the forward weights
    ``F[s, i, j]`` are those of ``compute_log_forward_tables``. Let ``Z``
    be the summed weight of all complete alignments, the sum of the three
    forward weights at ``(L, K)``. Treat ``Z`` as a function of the forward
    weights through the forward recursion applied at every lattice point,
    with predecessors outside the lattice contributing nothing. Entry
    ``[s, i, j]`` is the natural log of the derivative of ``Z`` with respect
    to ``F[s, i, j]``, and ``-inf`` where that derivative is zero. The three
    entries at ``(L, K)`` are therefore 0, and entry ``[0, 0, 0]`` equals the
    natural log of ``Z``. Values must stay exact to double precision even
    when the weights are far below the smallest positive float.

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
    return log_backward

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_log_backward_tables(
    x: "np.ndarray",
    y: "np.ndarray",
    match_factors: "np.ndarray",
    x_gap_factors: "np.ndarray",
    y_gap_factors: "np.ndarray",
    gap_open: float,
    gap_extend: float,
) -> "np.ndarray":
    """Reference implementation sweeping anti-diagonals backward in the log domain."""
    import numpy as np

    xs, ys, log_m, log_x, log_y, log_open, log_extend = _validate_pair_model(
        x, y, match_factors, x_gap_factors, y_gap_factors, gap_open, gap_extend)
    n, m = xs.size, ys.size
    table = np.full((3, n + 1, m + 1), -np.inf)
    bm, bx, by = table  # views into the returned table
    table[:, n, m] = 0.0
    # Cells on anti-diagonal d depend only on diagonals d + 1 and d + 2.
    for d in range(n + m - 1, -1, -1):
        i = np.arange(max(0, d - m), min(n, d) + 1)
        j = d - i
        to_match = np.full(i.size, -np.inf)
        to_xgap = np.full(i.size, -np.inf)
        to_ygap = np.full(i.size, -np.inf)
        keep = (i < n) & (j < m)
        to_match[keep] = log_m[i[keep], j[keep]] + bm[i[keep] + 1, j[keep] + 1]
        keep = i < n
        to_xgap[keep] = log_x[i[keep]] + bx[i[keep] + 1, j[keep]]
        keep = j < m
        to_ygap[keep] = log_y[j[keep]] + by[i[keep], j[keep] + 1]
        bm[i, j] = np.logaddexp(to_match, np.logaddexp(log_open + to_xgap, log_open + to_ygap))
        bx[i, j] = np.logaddexp(to_match, log_extend + to_xgap)
        by[i, j] = np.logaddexp(to_match, log_extend + to_ygap)
    return table

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
        "    return float(shape + np.sum(np.sin(0.29 * r) * f) + np.sum(np.cos(0.13 * r) * f) / 5.0)\n"
    )

    def _status(args):
        return {
            "setup": model + (
                "def _candidate():\n"
                "    try:\n"
                f"        compute_log_backward_tables({args})\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "def _reference():\n"
                "    try:\n"
                f"        _oracle_compute_log_backward_tables({args})\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        }

    return [
        {  # Normal pair-HMM lattice.
            "setup": model,
            "call": "_pin(compute_log_backward_tables(np.array([0, 2, 1, 3, 3, 0, 1]), np.array([0, 2, 2, 3, 0, 1]), M.copy(), EX.copy(), EY.copy(), 0.04, 0.4))",
            "gold_call": "_pin(_oracle_compute_log_backward_tables(np.array([0, 2, 1, 3, 3, 0, 1]), np.array([0, 2, 2, 3, 0, 1]), M.copy(), EX.copy(), EY.copy(), 0.04, 0.4))",
        },
        {  # Numerically extreme but finite weights.
            "setup": model,
            "call": "_pin(compute_log_backward_tables(np.array([2, 2, 1, 0, 3]), np.array([2, 1, 1, 0, 3, 3]), M * 1e-150, EX * 1e-150, EY * 1e-150, 1e-120, 0.5))",
            "gold_call": "_pin(_oracle_compute_log_backward_tables(np.array([2, 2, 1, 0, 3]), np.array([2, 1, 1, 0, 3, 3]), M * 1e-150, EX * 1e-150, EY * 1e-150, 1e-120, 0.5))",
        },
        {  # Boundary: one residue in each sequence.
            "setup": model,
            "call": "_pin(compute_log_backward_tables(np.array([2]), np.array([1]), M.copy(), EX.copy(), EY.copy(), 0.3, 0.6))",
            "gold_call": "_pin(_oracle_compute_log_backward_tables(np.array([2]), np.array([1]), M.copy(), EX.copy(), EY.copy(), 0.3, 0.6))",
        },
        {  # Edge: strongly unbalanced sequence lengths.
            "setup": model,
            "call": "_pin(compute_log_backward_tables(np.array([1]), np.array([0, 1, 2, 3, 3]), M.copy(), EX.copy(), EY.copy(), 0.2, 0.9))",
            "gold_call": "_pin(_oracle_compute_log_backward_tables(np.array([1]), np.array([0, 1, 2, 3, 3]), M.copy(), EX.copy(), EY.copy(), 0.2, 0.9))",
        },
        {  # Edge: asymmetric emission scaling.
            "setup": model,
            "call": "_pin(compute_log_backward_tables(np.array([3, 0, 0, 1, 2, 2, 1, 0]), np.array([3, 0]), M.copy(), (EX * 2.0).copy(), EY.copy(), 0.5, 0.25))",
            "gold_call": "_pin(_oracle_compute_log_backward_tables(np.array([3, 0, 0, 1, 2, 2, 1, 0]), np.array([3, 0]), M.copy(), (EX * 2.0).copy(), EY.copy(), 0.5, 0.25))",
        },
        _status("np.array([[0, 1], [1, 0]]), np.array([0, 1]), M.copy(), EX.copy(), EY.copy(), 0.04, 0.4"),
        _status("np.array([0, 1]), np.array([0, 1]), M[:3].copy(), EX.copy(), EY.copy(), 0.04, 0.4"),
    ]
