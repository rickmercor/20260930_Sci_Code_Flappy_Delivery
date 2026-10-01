"""
Tabulate log outside adjoints for every interval of the five-production RNA grammar.

Outside quantities collect all parse-tree contexts surrounding an interval. Logarithmic reverse accumulation preserves contributions whose magnitudes differ too much for an ordinary floating-point sum.

Returns
-------
np.ndarray: float log-outside table of shape (n + 1, n + 1), with -inf for zero adjoints.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_log_outside_table(sequence: str, rule_weights: "np.ndarray", unpaired: "np.ndarray",
                              pair_factors: "np.ndarray", log_inside: "np.ndarray") -> "np.ndarray":
    """Return natural logarithms of the outside adjoints.

    ``log_inside`` follows ``compute_log_inside_table``. Entry ``[i, k]`` is
    the logarithm of the derivative of the root weight with respect to the
    linear inside weight of ``sequence[i:k]``. The root entry is therefore
    zero. Store ``-inf`` for a zero adjoint and below the diagonal. Reverse
    propagation follows the same empty, pair and non-empty-bifurcation
    conventions as the inside table.

    Parameters
    ----------
    sequence : str
        Non-empty RNA over A, C, G, U.
    rule_weights, unpaired, pair_factors : np.ndarray
        Grammar arrays in the layouts accepted by ``compute_log_inside_table``.
    log_inside : np.ndarray
        Log-inside table of shape (n + 1, n + 1), with finite root weight.

    Returns
    -------
    np.ndarray
        Float log-outside table of shape (n + 1, n + 1).

    Raises
    ------
    ValueError
        If the grammar is invalid or ``log_inside`` has the wrong shape,
        contains NaN or positive infinity, or has a non-finite root entry.
    """
    return log_outside  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_log_outside_table(sequence: str, rule_weights: "np.ndarray", unpaired: "np.ndarray",
                                      pair_factors: "np.ndarray", log_inside: "np.ndarray") -> "np.ndarray":
    x, w, u, pf = _validate_rna_grammar(sequence, rule_weights, unpaired, pair_factors)
    n = len(x)
    table = np.asarray(log_inside, dtype=float)
    if table.shape != (n + 1, n + 1) or np.any(np.isnan(table)) or np.any(np.isposinf(table)) \
            or not np.isfinite(table[0, n]):
        raise ValueError("log_inside must have the right shape, no NaN or positive infinity, and a finite root")
    lw, lu, lpf = (_log_factor(a) for a in (w, u, pf))
    outside = np.full((n + 1, n + 1), -np.inf, dtype=float)
    outside[0, n] = 0.0
    for span in range(n, 0, -1):
        for i in range(n - span + 1):
            k = i + span
            parent = outside[i, k]
            if np.isneginf(parent):
                continue
            outside[i + 1, k] = np.logaddexp(
                outside[i + 1, k], parent + lw[1] + lu[0, x[i]]
            )
            outside[i, k - 1] = np.logaddexp(
                outside[i, k - 1], parent + lw[2] + lu[1, x[k - 1]]
            )
            if span >= 2:
                outside[i + 1, k - 1] = np.logaddexp(
                    outside[i + 1, k - 1], parent + lw[0] + lpf[x[i], x[k - 1]]
                )
                split = np.arange(i + 1, k)
                outside[i, split] = np.logaddexp(
                    outside[i, split], parent + lw[3] + table[split, k]
                )
                outside[split, k] = np.logaddexp(
                    outside[split, k], parent + lw[3] + table[i, split]
                )
    return outside

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    base = (
        "import numpy as np\nw = np.array([0.7, 0.44, 0.36, 0.24, 0.8]); u = np.array([[0.28, 0.19, 0.27, 0.26], [0.24, 0.26, 0.21, 0.29]])\n"
        "P = np.zeros((4, 4)); P[0, 3] = P[3, 0] = 1.1; P[1, 2] = P[2, 1] = 1.9; P[2, 3] = 0.5; P[3, 2] = 0.7\n"
        "def _inside(s):\n    x = np.array(['ACGU'.index(ch) for ch in s]); n = len(x); t = np.diag(np.full(n + 1, w[4]))\n    for d in range(1, n + 1):\n        for i in range(n - d + 1):\n            k = i + d; t[i, k] = w[1]*u[0,x[i]]*t[i+1,k] + w[2]*u[1,x[k-1]]*t[i,k-1]\n            if d > 1: t[i,k] += w[0]*P[x[i],x[k-1]]*t[i+1,k-1] + w[3]*float(np.dot(t[i,i+1:k],t[i+1:k,k]))\n    z = np.full_like(t, -np.inf); mask = t > 0; z[mask] = np.log(t[mask]); return z\n"
        "def _pin(v):\n    a = np.asarray(v, dtype=float); n = a.shape[0] - 1\n    if a.shape != (n + 1, n + 1) or not np.all(np.isneginf(a[np.tril_indices(n + 1, -1)])):\n        return -1.0\n    z = a[np.triu_indices(n + 1)]; q = np.where(np.isfinite(z), z, -700.0); r = np.arange(1.0, q.size + 1.0)\n    return float(1000*np.isfinite(z).sum() + np.sum(np.sin(.29*r)*q) + np.sum(np.cos(.11*r)*q*q)/q.size)\n"
        "def _status(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
    )
    args = "(s, w.copy(), u.copy(), P.copy(), L.copy())"
    return [
        {"setup": base + "s = 'GCAUGGUACGU'; L = _inside(s)\n",
         "call": "_pin(compute_log_outside_table" + args + ")", "gold_call": "_pin(_oracle_compute_log_outside_table" + args + ")"},
        {"setup": base + "w=np.array([0.0,0.0,1e200,0.0,1e110]); P[:]=0.0; s='AAAAAAAA'; n=len(s); a=np.log(w[2]*u[1,0]); L=np.full((n+1,n+1),-np.inf)\nfor i in range(n+1):\n    for k in range(i,n+1): L[i,k]=np.log(w[4])+(k-i)*a\n",
         "call": "_pin(compute_log_outside_table" + args + ")", "gold_call": "_pin(_oracle_compute_log_outside_table" + args + ")"},
        {"setup": base + "s = 'A'; L = _inside(s)\n",
         "call": "_pin(compute_log_outside_table" + args + ")", "gold_call": "_pin(_oracle_compute_log_outside_table" + args + ")"},
        {"setup": base + "s = 'ACGU'; L = np.zeros((4, 4))\n",
         "call": "_status(lambda: compute_log_outside_table" + args + ")",
         "gold_call": "_status(lambda: _oracle_compute_log_outside_table" + args + ")"},
    ]
