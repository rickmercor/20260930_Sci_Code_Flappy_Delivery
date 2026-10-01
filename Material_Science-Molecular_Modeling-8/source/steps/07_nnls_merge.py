"""
Perform the full rate-preserving merge: stack the scaled moment block and the rate rows, apply the column scaling, solve the nonnegative least-squares system, recover the relative weights as the source prescribes, retain particles whose relative weight reaches the threshold eps, and return the survivors in ascending original index order with the weights, velocities and positions the source assigns them.

How the solution vector turns back into particle weights, which particles survive, and which coordinates the survivors keep, are all fixed by the source's algorithm listing.

Returns
-------
return (M, 5) float64: merged particles [weight, v(3), x]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def nnls_merge(w, v, x, w2, v2, eps):
    """w, v, x: merged-species particles; w2, v2: fixed background species;
    eps: retention threshold on relative weights. Returns (M, 5) float64:
    one row per surviving particle, [weight, vx, vy, vz, x], in ascending
    original index order, assembled exactly as the source's algorithm
    prescribes. Raises ValueError on an invalid threshold."""
    return np.zeros((1, 5))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 7: the full rate-preserving NNLS merge."""

import numpy as np


def _oracle_nnls_merge(w, v, x, w2, v2, eps):
    w = np.asarray(w, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("invalid threshold")
    n = w.shape[0]
    wS = float(np.sum(w))
    M = _oracle_scaled_moment_system(w, v, x)
    R = _oracle_rate_rows(w, v, w2, v2)
    A = np.vstack([M[:, :n], R[:, :n]])
    b = np.concatenate([M[:, n], R[:, n]])
    s = _oracle_column_scaling(A)
    sol = _oracle_nnls_lawson_hanson(A * s[None, :], b)
    wh = s * sol
    keep = [i for i in range(n) if wh[i] >= eps]
    rows = [[wS * wh[i], v[i, 0], v[i, 1], v[i, 2], x[i]] for i in keep]
    return np.asarray(rows, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\nI=_n.arange(24)\nw=0.5+(((I+1)*(I+3)+3)%7)/10.0\nv=_n.stack([((((I+1)*(I+2)+(c+2)*(I+5)+3)%47)/23.5-1.0) for c in range(3)],axis=1)\nx=(((I+2)*(I+4)*3+1)%41)/41.0\nK=_n.arange(10)\nw2=0.8+(((K+1)*(K+2)+3)%5)/10.0\nv2=_n.stack([((((K+2)*(K+3)+(c+3)*(K+1)+2*3)%31)/25.0-0.6) for c in range(3)],axis=1)', "call": "nnls_merge(w, v, x, w2, v2, 1e-8)", "gold_call": "_oracle_nnls_merge(w, v, x, w2, v2, 1e-8)", "tol": 1e-08},
        {"setup": 'import numpy as _n\nI=_n.arange(24)\nw=0.5+(((I+1)*(I+3)+5)%7)/10.0\nv=_n.stack([((((I+1)*(I+2)+(c+2)*(I+5)+5)%47)/23.5-1.0) for c in range(3)],axis=1)\nx=(((I+2)*(I+4)*5+1)%41)/41.0\nK=_n.arange(10)\nw2=0.8+(((K+1)*(K+2)+5)%5)/10.0\nv2=_n.stack([((((K+2)*(K+3)+(c+3)*(K+1)+2*5)%31)/25.0-0.6) for c in range(3)],axis=1)', "call": "nnls_merge(w, v, x, w2, v2, 1e-8)", "gold_call": "_oracle_nnls_merge(w, v, x, w2, v2, 1e-8)", "tol": 1e-08},
        {"setup": 'import numpy as _n\nI=_n.arange(24)\nw=0.5+(((I+1)*(I+3)+4)%7)/10.0\nv=_n.stack([((((I+1)*(I+2)+(c+2)*(I+5)+4)%47)/23.5-1.0) for c in range(3)],axis=1)\nx=(((I+2)*(I+4)*4+1)%41)/41.0\nK=_n.arange(10)\nw2=0.8+(((K+1)*(K+2)+4)%5)/10.0\nv2=_n.stack([((((K+2)*(K+3)+(c+3)*(K+1)+2*4)%31)/25.0-0.6) for c in range(3)],axis=1)', "call": "nnls_merge(w, v, x, w2, v2, 1e-8)", "gold_call": "_oracle_nnls_merge(w, v, x, w2, v2, 1e-8)", "tol": 1e-08},
        {"setup": 'import numpy as _n\nI=_n.arange(24)\nw=0.5+(((I+1)*(I+3)+3)%7)/10.0\nv=_n.stack([((((I+1)*(I+2)+(c+2)*(I+5)+3)%47)/23.5-1.0) for c in range(3)],axis=1)\nx=(((I+2)*(I+4)*3+1)%41)/41.0\nK=_n.arange(10)\nw2=0.8+(((K+1)*(K+2)+3)%5)/10.0\nv2=_n.stack([((((K+2)*(K+3)+(c+3)*(K+1)+2*3)%31)/25.0-0.6) for c in range(3)],axis=1)', "call": "nnls_merge(w, v, x, w2, v2, 0.03)", "gold_call": "_oracle_nnls_merge(w, v, x, w2, v2, 0.03)", "tol": 1e-08},
        {"setup": 'import numpy as _n\nw=_n.array([0.7,1.1])\nv=_n.array([[0.2,-0.4,0.1],[-0.3,0.5,-0.2]])\nx=_n.array([0.25,0.75])\nw2=_n.array([1.0])\nv2=_n.array([[0.1,0.0,-0.1]])', "call": "nnls_merge(w, v, x, w2, v2, 1e-8)", "gold_call": "_oracle_nnls_merge(w, v, x, w2, v2, 1e-8)", "tol": 1e-08},
    ]
