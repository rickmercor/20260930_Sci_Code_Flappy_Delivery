"""
Obtain the exact natural-log total alignment weight after every single-base substitution of x from the reference forward and backward tables.

Mutational scans score every possible base replacement against one reference sequence, so the value of a reference ensemble computation lies in how much of it can be reused when a single residue is changed by a finite amount.

Returns
-------
np.ndarray: shape (L, 4), natural-log total weight after setting base i of x to each code.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_single_substitution_log_weights(
    x: "np.ndarray",
    y: "np.ndarray",
    match_factors: "np.ndarray",
    x_gap_factors: "np.ndarray",
    log_forward: "np.ndarray",
    log_backward: "np.ndarray",
) -> "np.ndarray":
    """Return the natural-log total weight of x after each single-base substitution.

    The pair model and the tables are those of ``compute_log_forward_tables``
    and ``compute_log_backward_tables`` evaluated for the reference ``x``
    and ``y``. Entry ``[i, c]`` is the natural log of the summed weight of all
    complete alignments of ``y`` with ``x`` after base ``i`` (0-based) is set
    to code ``c``, every other factor of the model unchanged; for ``c ==
    x[i]`` it is the reference value. Results must agree with a full
    recomputation for the substituted sequence to double precision, including
    when the weights are far below the smallest positive float.

    Parameters
    ----------
    x, y : np.ndarray
        Reference sequences, non-empty one-dimensional integer arrays of
        codes 0 to 3, lengths ``L`` and ``K``.
    match_factors : np.ndarray
        Shape ``(4, 4)``, finite and positive.
    x_gap_factors : np.ndarray
        Shape ``(4,)``, finite and positive.
    log_forward, log_backward : np.ndarray
        Reference tables of shape ``(3, L + 1, K + 1)``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(L, 4)``.

    Raises
    ------
    ValueError
        If a sequence or factor array is invalid as in
        ``compute_log_forward_tables``, if a table has the wrong shape or
        holds NaN or ``+inf``, or if the reference total weight is zero.
    """
    return single_log_weights

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_evaluate_single_substitution_log_weights(
    x: "np.ndarray",
    y: "np.ndarray",
    match_factors: "np.ndarray",
    x_gap_factors: "np.ndarray",
    log_forward: "np.ndarray",
    log_backward: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation contracting per-base coefficients with new factors."""
    import numpy as np

    # The y-gap and transition factors do not enter this contraction, so
    # placeholders stand in for them during validation.
    xs, ys, log_m, log_x, _, _, _ = _validate_pair_model(
        x, y, match_factors, x_gap_factors, x_gap_factors, 1.0, 1.0)
    f, b = _validate_log_tables(log_forward, log_backward, xs.size, ys.size)
    new_match = np.log(np.asarray(match_factors, dtype=float))[:, ys]
    new_xgap = np.log(np.asarray(x_gap_factors, dtype=float))
    # Each complete alignment uses base i once: in M at some column j, or in X.
    match_coefficient = f[0, 1:, 1:] + b[0, 1:, 1:] - log_m
    xgap_coefficient = np.logaddexp.reduce(f[1, 1:, :] + b[1, 1:, :], axis=1) - log_x
    result = np.empty((xs.size, 4))
    for code in range(4):
        through_match = np.logaddexp.reduce(match_coefficient + new_match[code], axis=1)
        result[:, code] = np.logaddexp(through_match, xgap_coefficient + new_xgap[code])
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
        "    if a.ndim != 2 or a.shape[1] != 4:\n"
        "        return -1.0\n"
        "    f = np.where(np.isfinite(a), a, -9000.0).ravel()\n"
        "    r = np.arange(1.0, f.size + 1.0)\n"
        "    return float(a.shape[0] * 1.0e3 + np.sum(np.sin(0.41 * r) * f) + np.sum(np.cos(0.17 * r) * f) / 3.0)\n"
    )
    instances = [
        ("x = np.array([0, 2, 1, 3, 3, 0, 1]); y = np.array([0, 2, 2, 3, 0, 1])\n"
         "em, ex, ey, go, ge = M, EX, EY, 0.04, 0.4\n"),
        ("x = np.array([2, 2, 1, 0, 3]); y = np.array([2, 1, 1, 0, 3, 3])\n"
         "em, ex, ey, go, ge = M * 1e-150, EX * 1e-150, EY * 1e-150, 1e-120, 0.5\n"),
        ("x = np.array([3]); y = np.array([1])\n"
         "em, ex, ey, go, ge = M, EX, EY, 0.3, 0.6\n"),
        ("x = np.array([1, 3, 2]); y = np.array([0, 1, 2, 3, 3, 2, 1])\n"
         "em, ex, ey, go, ge = M, np.array([0.05, 0.9, 0.4, 0.02]), EY, 0.2, 0.9\n"),
        ("x = np.array([3, 0, 0, 1, 2, 2, 1, 0]); y = np.array([3, 0])\n"
         "em, ex, ey, go, ge = M, EX * 2.0, EY, 0.5, 0.25\n"),
    ]
    cases = [
        {  # Normal lattice.
            "setup": model + instances[0] + "F, B = _tables(x, y, em, ex, ey, go, ge)\n",
            "call": "_pin(evaluate_single_substitution_log_weights(x.copy(), y.copy(), em.copy(), ex.copy(), F.copy(), B.copy()))",
            "gold_call": "_pin(_oracle_evaluate_single_substitution_log_weights(x.copy(), y.copy(), em.copy(), ex.copy(), F.copy(), B.copy()))",
        },
        {  # Numerically extreme but finite weights.
            "setup": model + instances[1] + "F, B = _tables(x, y, em, ex, ey, go, ge)\n",
            "call": "_pin(evaluate_single_substitution_log_weights(x.copy(), y.copy(), em.copy(), ex.copy(), F.copy(), B.copy()))",
            "gold_call": "_pin(_oracle_evaluate_single_substitution_log_weights(x.copy(), y.copy(), em.copy(), ex.copy(), F.copy(), B.copy()))",
        },
        {  # Boundary: a single substitutable residue.
            "setup": model + instances[2] + "F, B = _tables(x, y, em, ex, ey, go, ge)\n",
            "call": "_pin(evaluate_single_substitution_log_weights(x.copy(), y.copy(), em.copy(), ex.copy(), F.copy(), B.copy()))",
            "gold_call": "_pin(_oracle_evaluate_single_substitution_log_weights(x.copy(), y.copy(), em.copy(), ex.copy(), F.copy(), B.copy()))",
        },
        {  # Edge: strongly unbalanced lengths and skewed emissions.
            "setup": model + instances[3] + "F, B = _tables(x, y, em, ex, ey, go, ge)\n",
            "call": "_pin(evaluate_single_substitution_log_weights(x.copy(), y.copy(), em.copy(), ex.copy(), F.copy(), B.copy()))",
            "gold_call": "_pin(_oracle_evaluate_single_substitution_log_weights(x.copy(), y.copy(), em.copy(), ex.copy(), F.copy(), B.copy()))",
        },
        {  # Edge: asymmetric gap-emission scaling.
            "setup": model + instances[4] + "F, B = _tables(x, y, em, ex, ey, go, ge)\n",
            "call": "_pin(evaluate_single_substitution_log_weights(x.copy(), y.copy(), em.copy(), ex.copy(), F.copy(), B.copy()))",
            "gold_call": "_pin(_oracle_evaluate_single_substitution_log_weights(x.copy(), y.copy(), em.copy(), ex.copy(), F.copy(), B.copy()))",
        },
    ]
    cases.append({
        "setup": model + instances[0] + "F, B = _tables(x, y, em, ex, ey, go, ge)\n" + (
            "def _candidate():\n"
            "    try:\n"
            "        evaluate_single_substitution_log_weights(x.copy(), y.copy(), em.copy(), ex.copy(), F[:, :-1, :].copy(), B.copy())\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "def _reference():\n"
            "    try:\n"
            "        _oracle_evaluate_single_substitution_log_weights(x.copy(), y.copy(), em.copy(), ex.copy(), F[:, :-1, :].copy(), B.copy())\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"),
        "call": "_candidate()",
        "gold_call": "_reference()",
    })
    return cases
