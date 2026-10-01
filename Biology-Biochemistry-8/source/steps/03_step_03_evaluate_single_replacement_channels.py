"""
Evaluate every finite one-base replacement and resolve its exact weight among the four emission roles.

Every parse tree emits a position exactly once. Cutting that unique occurrence partitions the mutant ensemble into four mutually exclusive role classes whose log weights can be reconstructed from reference inside and outside quantities without rerunning the full grammar for every candidate.

Returns
-------
np.ndarray: shape (n, 4, 4), exact log weights by position, candidate base and emission role.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_single_replacement_channels(sequence: str, rule_weights: "np.ndarray",
                                         unpaired: "np.ndarray", pair_factors: "np.ndarray",
                                         log_inside: "np.ndarray",
                                         log_outside: "np.ndarray") -> "np.ndarray":
    """Return log weights for all candidates and the four emission roles.

    For each 0-based position and candidate base in A, C, G, U order, resolve
    the exact finite-replacement endpoint into parse trees where that position
    is emitted as the 5' base of a pair, the 3' base of a pair, left-unpaired,
    or right-unpaired, in that axis-2 order. The four classes are mutually
    exclusive and exhaustive; log-summing them gives the complete mutant
    weight. Use the reference log tables from the first two steps. Store
    ``-inf`` when a role has zero weight.

    Parameters
    ----------
    sequence : str
        Non-empty RNA over A, C, G, U.
    rule_weights, unpaired, pair_factors : np.ndarray
        Grammar arrays in the layouts accepted by ``compute_log_inside_table``.
    log_inside, log_outside : np.ndarray
        Reference log tables of shape (n + 1, n + 1).

    Returns
    -------
    np.ndarray
        Float array of shape (n, 4, 4): position, candidate base, role.

    Raises
    ------
    ValueError
        If the grammar is invalid, a log table has the wrong shape, contains
        NaN or positive infinity, or the reference root weight is non-finite.
    """
    return channel_logs  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_evaluate_single_replacement_channels(sequence: str, rule_weights: "np.ndarray",
                                                 unpaired: "np.ndarray", pair_factors: "np.ndarray",
                                                 log_inside: "np.ndarray",
                                                 log_outside: "np.ndarray") -> "np.ndarray":
    x, w, u, pf = _validate_rna_grammar(sequence, rule_weights, unpaired, pair_factors)
    n = len(x)
    inside, outside = (np.asarray(a, dtype=float) for a in (log_inside, log_outside))
    if any(a.shape != (n + 1, n + 1) or np.any(np.isnan(a)) or np.any(np.isposinf(a))
           for a in (inside, outside)) or not np.isfinite(inside[0, n]):
        raise ValueError("log tables must have the right shape, no NaN or positive infinity, and a finite root")
    lw, lu, lpf = (_log_factor(a) for a in (w, u, pf))
    result = np.full((n, 4, 4), -np.inf, dtype=float)
    for p in range(n):
        right_ends = np.arange(p + 2, n + 1)
        left_starts = np.arange(0, p)
        right_bounds = np.arange(p + 1, n + 1)
        left_bounds = np.arange(0, p + 1)
        p5_core = outside[p, right_ends] + lw[0] + inside[p + 1, right_ends - 1]
        p3_core = outside[left_starts, p + 1] + lw[0] + inside[left_starts + 1, p]
        left_core = outside[p, right_bounds] + inside[p + 1, right_bounds]
        right_core = outside[left_bounds, p + 1] + inside[left_bounds, p]
        for candidate in range(4):
            result[p, candidate, 0] = _logsumexp(
                p5_core + lpf[candidate, x[right_ends - 1]]
            )
            result[p, candidate, 1] = _logsumexp(
                p3_core + lpf[x[left_starts], candidate]
            )
            result[p, candidate, 2] = lw[1] + lu[0, candidate] + _logsumexp(left_core)
            result[p, candidate, 3] = lw[2] + lu[1, candidate] + _logsumexp(right_core)
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    base = (
        "import numpy as np\nw = np.array([0.7, 0.44, 0.36, 0.24, 0.8]); u = np.array([[0.28, 0.19, 0.27, 0.26], [0.24, 0.26, 0.21, 0.29]])\n"
        "P = np.zeros((4, 4)); P[0, 3] = P[3, 0] = 1.1; P[1, 2] = P[2, 1] = 1.9; P[2, 3] = 0.5; P[3, 2] = 0.7\n"
        "def _tables(s):\n    x=np.array(['ACGU'.index(c) for c in s]); n=len(s); t=np.diag(np.full(n+1,w[4])); o=np.zeros_like(t); o[0,n]=1.0\n    for d in range(1,n+1):\n        for i in range(n-d+1):\n            k=i+d; t[i,k]=w[1]*u[0,x[i]]*t[i+1,k]+w[2]*u[1,x[k-1]]*t[i,k-1]\n            if d>1: t[i,k]+=w[0]*P[x[i],x[k-1]]*t[i+1,k-1]+w[3]*float(np.dot(t[i,i+1:k],t[i+1:k,k]))\n    for d in range(n,0,-1):\n        for i in range(n-d+1):\n            k=i+d; b=o[i,k]; o[i+1,k]+=b*w[1]*u[0,x[i]]; o[i,k-1]+=b*w[2]*u[1,x[k-1]]\n            if d>1: o[i+1,k-1]+=b*w[0]*P[x[i],x[k-1]]; o[i,i+1:k]+=b*w[3]*t[i+1:k,k]; o[i+1:k,k]+=b*w[3]*t[i,i+1:k]\n    lt=np.full_like(t,-np.inf); lo=np.full_like(o,-np.inf); lt[t>0]=np.log(t[t>0]); lo[o>0]=np.log(o[o>0]); return lt,lo\n"
        "def _pin(v):\n    a=np.asarray(v,dtype=float); q=np.where(np.isfinite(a),a,-700.0).ravel(); m=np.isfinite(a).ravel(); r=np.arange(1.0,q.size+1.0)\n    return float(1000*m.sum()+np.sum(np.sin(.17*r)*q)+np.sum(np.cos(.07*r)*q*q)/q.size)\n"
        "def _status(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
    )
    args = "(s, w.copy(), u.copy(), P.copy(), L.copy(), O.copy())"
    return [
        {"setup": base + "s='GCAUGGUACGU'; L,O=_tables(s)\n",
         "call": "_pin(evaluate_single_replacement_channels" + args + ")",
         "gold_call": "_pin(_oracle_evaluate_single_replacement_channels" + args + ")"},
        {"setup": base + "w=np.array([0.0,0.0,1e200,0.0,1e110]); P[:]=0.0; s='AAAAAAAA'; n=len(s); a=np.log(w[2]*u[1,0]); L=np.full((n+1,n+1),-np.inf); O=np.full_like(L,-np.inf)\nfor i in range(n+1):\n    for k in range(i,n+1): L[i,k]=np.log(w[4])+(k-i)*a\nfor k in range(n+1): O[0,k]=(n-k)*a\n",
         "call": "_pin(evaluate_single_replacement_channels" + args + ")",
         "gold_call": "_pin(_oracle_evaluate_single_replacement_channels" + args + ")"},
        {"setup": base + "s='A'; L,O=_tables(s)\n",
         "call": "_pin(evaluate_single_replacement_channels" + args + ")",
         "gold_call": "_pin(_oracle_evaluate_single_replacement_channels" + args + ")"},
        {"setup": base + "s='ACGU'; L,O=_tables(s); O=O[:-1]\n",
         "call": "_status(lambda: evaluate_single_replacement_channels" + args + ")",
         "gold_call": "_status(lambda: _oracle_evaluate_single_replacement_channels" + args + ")"},
    ]
