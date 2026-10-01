"""
Tabulate log ensemble weights for every interval of an RNA sequence under the five-production grammar.

RNA parse-tree sums can span many orders of magnitude. Carrying interval weights in logarithmic form preserves the same sum-product grammar while avoiding overflow and underflow in long or strongly weighted sequences.

Returns
-------
np.ndarray: float log-inside table of shape (n + 1, n + 1), with -inf for zero weights.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_log_inside_table(sequence: str, rule_weights: "np.ndarray",
                             unpaired: "np.ndarray", pair_factors: "np.ndarray") -> "np.ndarray":
    """Return the natural logarithm of every interval's inside weight.

    The grammar is the five-production grammar in the problem statement. Array
    layouts are ``(t_P, t_L, t_R, t_B, t_E)``, left/right unpaired factors in
    A, C, G, U order, and ordered pair factors with the 5' base as the row.
    Entry ``[i, k]`` represents ``sequence[i:k]``. Empty intervals have weight
    ``t_E``; bifurcation children are non-empty; a pair may join the endpoints
    of any interval of length at least two. Store ``-inf`` for a zero weight and
    for entries below the diagonal.

    Parameters
    ----------
    sequence : str
        Non-empty RNA over A, C, G, U.
    rule_weights : np.ndarray
        Non-negative finite array of shape (5,), with positive t_E.
    unpaired : np.ndarray
        Non-negative finite array of shape (2, 4), left then right.
    pair_factors : np.ndarray
        Non-negative finite array of shape (4, 4), with nonzero entries only
        for A-U, U-A, C-G, G-C, G-U and U-G.

    Returns
    -------
    np.ndarray
        Float log-inside table of shape (n + 1, n + 1).

    Raises
    ------
    ValueError
        If the sequence or a grammar array violates the contract above.
    """
    return log_inside  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _validate_rna_grammar(sequence, rule_weights, unpaired, pair_factors):
    if not isinstance(sequence, str) or not sequence or set(sequence) - set("ACGU"):
        raise ValueError("sequence must be a non-empty string over A, C, G, U")
    w, u, pf = (np.asarray(a, dtype=float) for a in (rule_weights, unpaired, pair_factors))
    if w.shape != (5,) or u.shape != (2, 4) or pf.shape != (4, 4):
        raise ValueError("grammar arrays have the wrong shape")
    allowed = np.zeros((4, 4), dtype=bool)
    allowed[[0, 3, 1, 2, 2, 3], [3, 0, 2, 1, 3, 2]] = True
    if any(not np.all(np.isfinite(a)) or np.any(a < 0.0) for a in (w, u, pf)) \
            or w[4] <= 0.0 or np.any(pf[~allowed] != 0.0):
        raise ValueError("factors must be finite and non-negative, t_E positive, and disallowed pairs zero")
    x = np.fromiter(("ACGU".index(ch) for ch in sequence), dtype=int, count=len(sequence))
    return x, w, u, pf

def _log_factor(values):
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, -np.inf, dtype=float)
    positive = values > 0.0
    result[positive] = np.log(values[positive])
    return result

def _logsumexp(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return -np.inf
    maximum = float(np.max(values))
    if np.isneginf(maximum):
        return -np.inf
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))

def _oracle_compute_log_inside_table(sequence: str, rule_weights: "np.ndarray",
                                     unpaired: "np.ndarray", pair_factors: "np.ndarray") -> "np.ndarray":
    x, w, u, pf = _validate_rna_grammar(sequence, rule_weights, unpaired, pair_factors)
    n = len(x)
    lw, lu, lpf = (_log_factor(a) for a in (w, u, pf))
    table = np.full((n + 1, n + 1), -np.inf, dtype=float)
    np.fill_diagonal(table, lw[4])
    for span in range(1, n + 1):
        for i in range(n - span + 1):
            k = i + span
            terms = [
                lw[1] + lu[0, x[i]] + table[i + 1, k],
                lw[2] + lu[1, x[k - 1]] + table[i, k - 1],
            ]
            if span >= 2:
                terms.append(lw[0] + lpf[x[i], x[k - 1]] + table[i + 1, k - 1])
                split_terms = table[i, i + 1:k] + table[i + 1:k, k]
                terms.append(lw[3] + _logsumexp(split_terms))
            table[i, k] = _logsumexp(terms)
    return table

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    base = (
        "import numpy as np\nw = np.array([0.7, 0.44, 0.36, 0.24, 0.8]); u = np.array([[0.28, 0.19, 0.27, 0.26], [0.24, 0.26, 0.21, 0.29]])\n"
        "P = np.zeros((4, 4)); P[0, 3] = P[3, 0] = 1.1; P[1, 2] = P[2, 1] = 1.9; P[2, 3] = 0.5; P[3, 2] = 0.7\n"
        "def _pin(v):\n    a = np.asarray(v, dtype=float); n = a.shape[0] - 1\n    if a.shape != (n + 1, n + 1) or not np.all(np.isneginf(a[np.tril_indices(n + 1, -1)])):\n        return -1.0\n    z = a[np.triu_indices(n + 1)]; finite = np.isfinite(z); q = np.where(finite, z, -700.0); r = np.arange(1.0, q.size + 1.0)\n    return float(1000 * finite.sum() + np.sum(np.sin(0.37 * r) * q) + np.sum(np.cos(0.19 * r) * q * q) / q.size)\n"
        "def _status(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
    )
    return [
        {"setup": base, "call": "_pin(compute_log_inside_table('GCAUGGUACGU', w.copy(), u.copy(), P.copy()))",
         "gold_call": "_pin(_oracle_compute_log_inside_table('GCAUGGUACGU', w.copy(), u.copy(), P.copy()))"},
        {"setup": base + "w = np.array([1e120, 1e150, 1e140, 1e130, 1e110])\n",
         "call": "_pin(compute_log_inside_table('GGUUCCACAGUA', w.copy(), u.copy(), P.copy()))",
         "gold_call": "_pin(_oracle_compute_log_inside_table('GGUUCCACAGUA', w.copy(), u.copy(), P.copy()))"},
        {"setup": base + "u[0, 2] = 0.0; u[1, 2] = 0.0; P[2, :] = 0.0; P[:, 2] = 0.0\n",
         "call": "_pin(compute_log_inside_table('G', w.copy(), u.copy(), P.copy()))",
         "gold_call": "_pin(_oracle_compute_log_inside_table('G', w.copy(), u.copy(), P.copy()))"},
        {"setup": base + "P[0, 2] = 0.4\n", "call": "_status(lambda: compute_log_inside_table('ACGU', w, u, P))",
         "gold_call": "_status(lambda: _oracle_compute_log_inside_table('ACGU', w, u, P))"},
    ]
