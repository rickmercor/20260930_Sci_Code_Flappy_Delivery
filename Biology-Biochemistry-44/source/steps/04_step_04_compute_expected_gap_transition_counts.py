"""
Compute the posterior expected numbers of gap-opening and gap-extension columns in the alignment ensemble of x with y from the reference forward and backward tables.

When every alignment is weighted in proportion to the product of factors, the expected number of times a transition is used is a sensitivity of the logarithm of the total weight, which is what links a fitted pair model to the transition counts observed in curated alignments.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_expected_gap_transition_counts(
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
    """Return the expected numbers of gap-opening and gap-extension columns.

    The pair model and the tables are those of ``compute_log_forward_tables``
    and ``compute_log_backward_tables`` for the same arguments. Each complete
    alignment of ``x`` with ``y`` is drawn with probability equal to its
    weight divided by the summed weight ``Z`` of all complete alignments. A
    gap-opening column is an X or Y column whose previous column is M, the
    first column counting as following M; a gap-extension column is an X or
    Y column whose previous column is the same gap state. Return the expected
    number of each kind of column, exact to double precision.

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
    log_forward, log_backward : np.ndarray
        Tables of shape ``(3, L + 1, K + 1)`` for these arguments.

    Returns
    -------
    np.ndarray
        Float array ``[expected_openings, expected_extensions]``.

    Raises
    ------
    ValueError
        If a sequence or factor is invalid as in ``compute_log_forward_tables``,
        if a table has the wrong shape or holds NaN or ``+inf``, or if the
        total weight ``Z`` is zero.
    """
    return expected_counts

# EXPECTED RETURN
# np.ndarray: [expected number of gap-opening columns, expected number of gap-extension columns].

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_expected_gap_transition_counts(
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
    """Reference implementation summing local-event posteriors of the gap transitions."""
    import numpy as np

    xs, ys, _, log_x, log_y, log_open, log_extend = _validate_pair_model(
        x, y, match_factors, x_gap_factors, y_gap_factors, gap_open, gap_extend)
    n, m = xs.size, ys.size
    f, b = _validate_log_tables(log_forward, log_backward, n, m)
    log_z = np.logaddexp.reduce(f[:, n, m])
    # A column in X at (i, j) follows lattice point (i - 1, j); in Y at (i, j) it follows (i, j - 1).
    into_x = log_x[:, None] + b[1, 1:, :]
    into_y = log_y[None, :] + b[2, :, 1:]
    opened = np.logaddexp(np.logaddexp.reduce(log_open + f[0, :-1, :] + into_x, axis=None),
                          np.logaddexp.reduce(log_open + f[0, :, :-1] + into_y, axis=None))
    extended = np.logaddexp(np.logaddexp.reduce(log_extend + f[1, :-1, :] + into_x, axis=None),
                            np.logaddexp.reduce(log_extend + f[2, :, :-1] + into_y, axis=None))
    return np.exp(np.array([opened, extended]) - log_z)


def _validate_log_tables(log_forward, log_backward, n, m):
    """Return validated float tables of shape (3, n + 1, m + 1)."""
    import numpy as np

    tables = [np.asarray(t, dtype=float) for t in (log_forward, log_backward)]
    for t in tables:
        if t.shape != (3, n + 1, m + 1) or np.any(np.isnan(t)) or np.any(t == np.inf):
            raise ValueError("log tables must have shape (3, L + 1, K + 1) without NaN or +inf")
    if not np.isfinite(np.logaddexp.reduce(tables[0][:, n, m])):
        raise ValueError("the reference total weight must be positive")
    return tables[0], tables[1]

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
        "def _pin(counts):\n"
        "    c = np.asarray(counts, dtype=float)\n"
        "    if c.shape != (2,):\n"
        "        return -1.0\n"
        "    return float(1.0 + 1.0e3 * c[0] + 1.0e2 * c[1])\n"
    )
    args = "x.copy(), y.copy(), em.copy(), ex.copy(), ey.copy(), go, ge, F.copy(), B.copy()"
    instances = [
        ("x = np.array([0, 2, 1, 3, 3, 0, 1]); y = np.array([0, 2, 2, 3, 0, 1])\n"
         "em, ex, ey, go, ge = M, EX, EY, 0.04, 0.4\n"),
        ("x = np.array([2, 2, 1, 0, 3]); y = np.array([2, 1, 1, 0, 3, 3])\n"
         "em, ex, ey, go, ge = M * 1e-150, EX * 1e-150, EY * 1e-150, 1e-120, 0.5\n"),
        ("x = np.array([1, 3]); y = np.array([1, 3, 3, 2, 0, 0, 2])\n"
         "em, ex, ey, go, ge = M, EX, EY, 0.2, 0.9\n"),
        ("x = np.array([3, 0, 0, 1, 2, 2, 1, 0, 3]); y = np.array([3, 0, 2])\n"
         "em, ex, ey, go, ge = M, EX * 2.0, EY, 0.5, 0.25\n"),
        ("x = np.array([2, 1, 1, 0, 2, 3]); y = np.array([2, 1, 0, 0, 2, 3])\n"
         "em, ex, ey, go, ge = M, np.array([0.05, 0.9, 0.4, 0.02]), np.array([0.7, 0.1, 0.3, 0.8]), 0.3, 0.6\n"),
    ]
    cases = [
        {  # Normal lattice.
            "setup": model + instances[0] + "F, B = _tables(x, y, em, ex, ey, go, ge)\n",
            "call": f"_pin(compute_expected_gap_transition_counts({args}))",
            "gold_call": f"_pin(_oracle_compute_expected_gap_transition_counts({args}))",
        },
        {  # Numerically extreme but finite weights.
            "setup": model + instances[1] + "F, B = _tables(x, y, em, ex, ey, go, ge)\n",
            "call": f"_pin(compute_expected_gap_transition_counts({args}))",
            "gold_call": f"_pin(_oracle_compute_expected_gap_transition_counts({args}))",
        },
        {  # Boundary: short, length-imbalanced sequences.
            "setup": model + instances[2] + "F, B = _tables(x, y, em, ex, ey, go, ge)\n",
            "call": f"_pin(compute_expected_gap_transition_counts({args}))",
            "gold_call": f"_pin(_oracle_compute_expected_gap_transition_counts({args}))",
        },
        {  # Edge: asymmetric sequence lengths and emissions.
            "setup": model + instances[3] + "F, B = _tables(x, y, em, ex, ey, go, ge)\n",
            "call": f"_pin(compute_expected_gap_transition_counts({args}))",
            "gold_call": f"_pin(_oracle_compute_expected_gap_transition_counts({args}))",
        },
        {  # Edge: residue-skewed gap emissions.
            "setup": model + instances[4] + "F, B = _tables(x, y, em, ex, ey, go, ge)\n",
            "call": f"_pin(compute_expected_gap_transition_counts({args}))",
            "gold_call": f"_pin(_oracle_compute_expected_gap_transition_counts({args}))",
        },
    ]
    cases.append({
        "setup": model + instances[0] + "F, B = _tables(x, y, em, ex, ey, go, ge)\nB[1, 2, 3] = np.nan\n" + (
            "def _candidate():\n"
            "    try:\n"
            f"        compute_expected_gap_transition_counts({args})\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "def _reference():\n"
            "    try:\n"
            f"        _oracle_compute_expected_gap_transition_counts({args})\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"),
        "call": "_candidate()",
        "gold_call": "_reference()",
    })
    return cases
