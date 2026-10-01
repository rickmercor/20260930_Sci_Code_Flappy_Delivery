"""
Obtain the exact natural-log total alignment weight after every double substitution of two neighbouring bases of x from the reference forward and backward tables.

Neighbouring residues are aligned jointly, so replacing two adjacent bases is a genuinely two-site edit whose effect on the ensemble cannot be read off the two single replacements.

Returns
-------
np.ndarray: shape (L - 1, 4, 4), natural-log total weight after setting bases i and i + 1 of x to codes c and d.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_neighbour_double_log_weights(
    x: "np.ndarray",
    y: "np.ndarray",
    match_factors: "np.ndarray",
    x_gap_factors: "np.ndarray",
    y_gap_factors: "np.ndarray",
    gap_open: float,
    gap_extend: float,
    log_forward: "np.ndarray",
    log_backward: "np.ndarray",
) -> "np.ndarray":
    """Return the natural-log total weight of x after each neighbouring double substitution.

    The pair model and the tables are those of ``compute_log_forward_tables``
    and ``compute_log_backward_tables`` evaluated for the reference ``x``
    and ``y``. Entry ``[i, c, d]`` is the natural log of the summed weight of
    all complete alignments of ``y`` with ``x`` after base ``i`` (0-based) is
    set to code ``c`` and base ``i + 1`` to code ``d``, every other factor of
    the model unchanged. A code equal to the base already present leaves that
    base unchanged, so entry ``[i, x[i], x[i + 1]]`` is the reference value.
    Results must agree with a full recomputation for the substituted sequence
    to double precision, including when the weights are far below the
    smallest positive float.

    Parameters
    ----------
    x, y : np.ndarray
        Reference sequences, one-dimensional integer arrays of codes 0 to 3,
        lengths ``L >= 2`` and ``K >= 1``.
    match_factors : np.ndarray
        Shape ``(4, 4)``, finite and positive.
    x_gap_factors, y_gap_factors : np.ndarray
        Shape ``(4,)``, finite and positive.
    gap_open, gap_extend : float
        Finite positive transition factors.
    log_forward, log_backward : np.ndarray
        Reference tables of shape ``(3, L + 1, K + 1)``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(L - 1, 4, 4)``.

    Raises
    ------
    ValueError
        If a sequence or factor is invalid as in ``compute_log_forward_tables``,
        if ``x`` has fewer than two bases, if a table has the wrong shape or
        holds NaN or ``+inf``, or if the reference total weight is zero.
    """
    return double_log_weights

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_neighbour_double_log_weights(
    x: "np.ndarray",
    y: "np.ndarray",
    match_factors: "np.ndarray",
    x_gap_factors: "np.ndarray",
    y_gap_factors: "np.ndarray",
    gap_open: float,
    gap_extend: float,
    log_forward: "np.ndarray",
    log_backward: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation: outside context, joint two-row block, inside context."""
    import numpy as np

    xs, ys, log_m, log_x, log_y, log_open, log_extend = _validate_pair_model(
        x, y, match_factors, x_gap_factors, y_gap_factors, gap_open, gap_extend)
    n, m = xs.size, ys.size
    if n < 2:
        raise ValueError("x must have at least two bases")
    f, b = _validate_log_tables(log_forward, log_backward, n, m)
    new_match = np.log(np.asarray(match_factors, dtype=float))[:, ys]
    new_xgap = np.log(np.asarray(x_gap_factors, dtype=float))
    result = np.empty((n - 1, 4, 4))
    for start in range(0, n - 1, 256):
        i = np.arange(start, min(n - 1, start + 256))
        # Weight up to and including base i placed in M (columns 1..m) or X (0..m).
        into_i_match = np.full((i.size, 4, m + 1), -np.inf)
        into_i_match[:, :, 1:] = (f[0, i + 1, 1:] - log_m[i])[:, None, :] + new_match[None]
        into_i_xgap = (f[1, i + 1, :] - log_x[i][:, None])[:, None, :] + new_xgap[None, :, None]
        # Columns of y placed against gaps between the two bases follow only M.
        y_run = np.full((i.size, 4, m + 1), -np.inf)
        for col in range(1, m + 1):
            y_run[:, :, col] = log_y[col - 1] + np.logaddexp(
                log_open + into_i_match[:, :, col - 1], log_extend + y_run[:, :, col - 1])
        before_match = np.logaddexp(np.logaddexp(into_i_match[:, :, :-1], into_i_xgap[:, :, :-1]),
                                    y_run[:, :, :-1])
        before_xgap = np.logaddexp(log_open + into_i_match, log_extend + into_i_xgap)
        after_match, after_xgap = b[0, i + 2, 1:], b[1, i + 2, :]
        for code in range(4):
            via_match = np.logaddexp.reduce(
                before_match + (new_match[code] + after_match)[:, None, :], axis=2)
            via_xgap = np.logaddexp.reduce(
                before_xgap + (new_xgap[code] + after_xgap)[:, None, :], axis=2)
            result[i, :, code] = np.logaddexp(via_match, via_xgap)
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    model = (
        "import math\n"
        "import numpy as np\n"
        "M = np.array([[0.16, 0.03, 0.05, 0.03], [0.03, 0.16, 0.03, 0.05],\n"
        "              [0.05, 0.03, 0.16, 0.03], [0.03, 0.05, 0.03, 0.16]])\n"
        "EX = np.array([0.12, 0.38, 0.38, 0.12])\n"
        "EY = np.array([0.16, 0.34, 0.34, 0.16])\n"
        "def _lae(*v):\n"
        "    top = max(v)\n"
        "    return top if top == -math.inf else top + math.log(sum(math.exp(t - top) for t in v))\n"
        "def _tables(x, y, em, ex, ey, go, ge):\n"
        "    n, m, ninf = len(x), len(y), -math.inf\n"
        "    lo, le = math.log(go), math.log(ge)\n"
        "    F = np.full((3, n + 1, m + 1), ninf)\n"
        "    F[0, 0, 0] = 0.0\n"
        "    for i in range(n + 1):\n"
        "        for j in range(m + 1):\n"
        "            if i and j:\n"
        "                F[0, i, j] = math.log(em[x[i-1], y[j-1]]) + _lae(*F[:, i-1, j-1])\n"
        "            if i:\n"
        "                F[1, i, j] = math.log(ex[x[i-1]]) + _lae(lo + F[0, i-1, j], le + F[1, i-1, j])\n"
        "            if j:\n"
        "                F[2, i, j] = math.log(ey[y[j-1]]) + _lae(lo + F[0, i, j-1], le + F[2, i, j-1])\n"
        "    B = np.full((3, n + 1, m + 1), ninf)\n"
        "    B[:, n, m] = 0.0\n"
        "    for i in range(n, -1, -1):\n"
        "        for j in range(m, -1, -1):\n"
        "            if i == n and j == m:\n"
        "                continue\n"
        "            mm = math.log(em[x[i], y[j]]) + B[0, i+1, j+1] if i < n and j < m else ninf\n"
        "            gx = math.log(ex[x[i]]) + B[1, i+1, j] if i < n else ninf\n"
        "            gy = math.log(ey[y[j]]) + B[2, i, j+1] if j < m else ninf\n"
        "            B[0, i, j] = _lae(mm, lo + gx, lo + gy)\n"
        "            B[1, i, j] = _lae(mm, le + gx)\n"
        "            B[2, i, j] = _lae(mm, le + gy)\n"
        "    return F, B\n"
        "def _pin(values):\n"
        "    a = np.asarray(values, dtype=float)\n"
        "    if a.ndim != 3 or a.shape[1:] != (4, 4):\n"
        "        return -1.0\n"
        "    f = np.where(np.isfinite(a), a, -9000.0).ravel()\n"
        "    r = np.arange(1.0, f.size + 1.0)\n"
        "    return float(a.shape[0] * 1.0e3 + np.sum(np.sin(0.43 * r) * f) + np.sum(np.cos(0.19 * r) * f) / 3.0)\n"
    )
    args = "x.copy(), y.copy(), em.copy(), ex.copy(), ey.copy(), go, ge, F.copy(), B.copy()"
    instances = [
        ("x = np.array([0, 2, 1, 3, 3, 0, 1]); y = np.array([0, 2, 2, 3, 0, 1])\n"
         "em, ex, ey, go, ge = M, EX, EY, 0.04, 0.4\n"),
        ("x = np.array([2, 2, 1, 0, 3]); y = np.array([2, 1, 1, 0, 3, 3])\n"
         "em, ex, ey, go, ge = M * 1e-150, EX * 1e-150, EY * 1e-150, 1e-120, 0.5\n"),
        ("x = np.array([3, 1]); y = np.array([1])\n"
         "em, ex, ey, go, ge = M, EX, EY, 0.3, 0.6\n"),
        ("x = np.array([1, 3, 2]); y = np.array([0, 1, 2, 3, 3, 2, 1])\n"
         "em, ex, ey, go, ge = M, np.array([0.05, 0.9, 0.4, 0.02]), np.array([0.7, 0.1, 0.3, 0.8]), 0.2, 0.9\n"),
        ("x = np.array([3, 0, 0, 1, 2, 2, 1, 0]); y = np.array([3, 0])\n"
         "em, ex, ey, go, ge = M, EX * 2.0, EY, 0.5, 0.25\n"),
    ]
    cases = [
        {  # Normal lattice.
            "setup": model + instances[0] + "F, B = _tables(x, y, em, ex, ey, go, ge)\n",
            "call": f"_pin(compute_neighbour_double_log_weights({args}))",
            "gold_call": f"_pin(_oracle_compute_neighbour_double_log_weights({args}))",
        },
        {  # Numerically extreme but finite weights.
            "setup": model + instances[1] + "F, B = _tables(x, y, em, ex, ey, go, ge)\n",
            "call": f"_pin(compute_neighbour_double_log_weights({args}))",
            "gold_call": f"_pin(_oracle_compute_neighbour_double_log_weights({args}))",
        },
        {  # Boundary: the shortest sequence with one neighbouring pair.
            "setup": model + instances[2] + "F, B = _tables(x, y, em, ex, ey, go, ge)\n",
            "call": f"_pin(compute_neighbour_double_log_weights({args}))",
            "gold_call": f"_pin(_oracle_compute_neighbour_double_log_weights({args}))",
        },
        {  # Edge: strongly unbalanced lengths and skewed emissions.
            "setup": model + instances[3] + "F, B = _tables(x, y, em, ex, ey, go, ge)\n",
            "call": f"_pin(compute_neighbour_double_log_weights({args}))",
            "gold_call": f"_pin(_oracle_compute_neighbour_double_log_weights({args}))",
        },
        {  # Edge: asymmetric gap-emission scaling.
            "setup": model + instances[4] + "F, B = _tables(x, y, em, ex, ey, go, ge)\n",
            "call": f"_pin(compute_neighbour_double_log_weights({args}))",
            "gold_call": f"_pin(_oracle_compute_neighbour_double_log_weights({args}))",
        },
    ]
    cases.append({
        "setup": model + "x = np.array([2]); y = np.array([2, 1])\n"
                 "em, ex, ey, go, ge = M, EX, EY, 0.04, 0.4\nF, B = _tables(x, y, em, ex, ey, go, ge)\n" + (
            "def _candidate():\n"
            "    try:\n"
            f"        compute_neighbour_double_log_weights({args})\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "def _reference():\n"
            "    try:\n"
            f"        _oracle_compute_neighbour_double_log_weights({args})\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"),
        "call": "_candidate()",
        "gold_call": "_reference()",
    })
    return cases
