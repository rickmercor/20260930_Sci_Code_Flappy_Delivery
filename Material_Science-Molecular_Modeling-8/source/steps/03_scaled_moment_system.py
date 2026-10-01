"""
Assemble the scaled moment block of the merging system for the declared multi-index sets: 10 velocity moment rows for the multi-indices (0,0,0), (1,0,0), (0,1,0), (0,0,1), (2,0,0), (0,2,0), (0,0,2), (1,1,0), (1,0,1), (0,1,1) in that order, then 2 spatial rows for powers 1 and 2, built from the particles transformed exactly as the source prescribes, with the right-hand side constructed consistently as the source's algorithm states, appended as the last column.

The source stabilizes the Vandermonde moment system by transforming the particle coordinates before building monomials; which transformation, and how the right-hand side is formed from it, follow the source's algorithm listing.

Returns
-------
return (12, N+1) float64: scaled moment rows plus right-hand side
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def scaled_moment_system(w, v, x):
    """w: (N,) weights; v: (N, 3) velocities; x: (N,) positions.
    Returns (12, N+1) float64: the scaled moment rows for the declared
    velocity multi-index set and spatial powers, in the declared order, with
    the consistently constructed right-hand side as the final column.
    Raises ValueError on degenerate scales."""
    return np.zeros((12, np.asarray(w).shape[0] + 1))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 3: scaled moment block with consistent right-hand side."""

import numpy as np

_IV = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
       (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
_IX = [1, 2]


def _oracle_scaled_moment_system(w, v, x):
    w = np.asarray(w, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = w.shape[0]
    agg = _oracle_bin_aggregates(w, v, x)
    ref = _oracle_reference_scales(w, v, x)
    wS, vS, xS = agg[0], agg[1:4], agg[4]
    vref, xref = ref[0:3], ref[3]
    vh = (v - vS) / vref
    xh = (x - xS) / xref
    out = np.empty((len(_IV) + len(_IX), n + 1))
    for j, (mx, my, mz) in enumerate(_IV):
        out[j, :n] = (vh[:, 0] ** mx) * (vh[:, 1] ** my) * (vh[:, 2] ** mz)
    for k, mx_ in enumerate(_IX):
        out[len(_IV) + k, :n] = xh ** mx_
    out[:, n] = out[:, :n] @ (w / wS)
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\nI=_n.arange(24)\nw=0.5+(((I+1)*(I+3)+3)%7)/10.0\nv=_n.stack([((((I+1)*(I+2)+(c+2)*(I+5)+3)%47)/23.5-1.0) for c in range(3)],axis=1)\nx=(((I+2)*(I+4)*3+1)%41)/41.0\nK=_n.arange(10)\nw2=0.8+(((K+1)*(K+2)+3)%5)/10.0\nv2=_n.stack([((((K+2)*(K+3)+(c+3)*(K+1)+2*3)%31)/25.0-0.6) for c in range(3)],axis=1)', "call": "scaled_moment_system(w, v, x)", "gold_call": "_oracle_scaled_moment_system(w, v, x)", "tol": 1e-09},
        {"setup": 'import numpy as _n\nI=_n.arange(24)\nw=0.5+(((I+1)*(I+3)+5)%7)/10.0\nv=_n.stack([((((I+1)*(I+2)+(c+2)*(I+5)+5)%47)/23.5-1.0) for c in range(3)],axis=1)\nx=(((I+2)*(I+4)*5+1)%41)/41.0\nK=_n.arange(10)\nw2=0.8+(((K+1)*(K+2)+5)%5)/10.0\nv2=_n.stack([((((K+2)*(K+3)+(c+3)*(K+1)+2*5)%31)/25.0-0.6) for c in range(3)],axis=1)', "call": "scaled_moment_system(w, v, x)", "gold_call": "_oracle_scaled_moment_system(w, v, x)", "tol": 1e-09},
        {"setup": 'import numpy as _n\nI=_n.arange(24)\nw=0.5+(((I+1)*(I+3)+4)%7)/10.0\nv=_n.stack([((((I+1)*(I+2)+(c+2)*(I+5)+4)%47)/23.5-1.0) for c in range(3)],axis=1)\nx=(((I+2)*(I+4)*4+1)%41)/41.0\nK=_n.arange(10)\nw2=0.8+(((K+1)*(K+2)+4)%5)/10.0\nv2=_n.stack([((((K+2)*(K+3)+(c+3)*(K+1)+2*4)%31)/25.0-0.6) for c in range(3)],axis=1)', "call": "scaled_moment_system(w, v, x)", "gold_call": "_oracle_scaled_moment_system(w, v, x)", "tol": 1e-09},
        {"setup": 'import numpy as _n\nw=_n.array([0.7,1.1])\nv=_n.array([[0.2,-0.4,0.1],[-0.3,0.5,-0.2]])\nx=_n.array([0.25,0.75])', "call": "scaled_moment_system(w, v, x)", "gold_call": "_oracle_scaled_moment_system(w, v, x)", "tol": 1e-09},
        {"setup": 'import numpy as _n\nw=_n.array([1.0,0.8,1.0,0.6])\nv=_n.array([[0.3,-0.2,0.5],[0.3,-0.2,0.5],[-0.4,0.6,-0.1],[0.1,0.1,0.2]])\nx=_n.array([0.2,0.2,0.8,0.5])', "call": "scaled_moment_system(w, v, x)", "gold_call": "_oracle_scaled_moment_system(w, v, x)", "tol": 1e-09},
    ]
