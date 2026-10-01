"""
Compute the bin aggregates of the merge group: the total weight, the weighted mean velocity and the weighted mean position, packed as a length-5 vector.

Every later stage of the source's merging pipeline is expressed relative to these aggregates.

Returns
-------
return (5,) float64: [w_sum, v_mean(3), x_mean]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def bin_aggregates(w, v, x):
    """w: (N,) positive weights; v: (N, 3) velocities; x: (N,) positions.
    Returns (5,) float64: the total weight, the three components of the
    weighted mean velocity, and the weighted mean position, in that order.
    Raises ValueError on shape mismatch, nonpositive weights or nonfinite
    input."""
    return np.zeros(5)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 1: bin aggregates of the merge group."""

import numpy as np


def _oracle_bin_aggregates(w, v, x):
    w = np.asarray(w, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if w.ndim != 1 or v.shape != (w.shape[0], 3) or x.shape != w.shape:
        raise ValueError("shape mismatch")
    if not (np.all(np.isfinite(w)) and np.all(np.isfinite(v)) and np.all(np.isfinite(x))):
        raise ValueError("nonfinite input")
    if np.any(w <= 0):
        raise ValueError("nonpositive weight")
    wS = float(np.sum(w))
    vS = (w @ v) / wS
    xS = float(w @ x) / wS
    return np.concatenate([[wS], vS, [xS]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\nI=_n.arange(24)\nw=0.5+(((I+1)*(I+3)+3)%7)/10.0\nv=_n.stack([((((I+1)*(I+2)+(c+2)*(I+5)+3)%47)/23.5-1.0) for c in range(3)],axis=1)\nx=(((I+2)*(I+4)*3+1)%41)/41.0\nK=_n.arange(10)\nw2=0.8+(((K+1)*(K+2)+3)%5)/10.0\nv2=_n.stack([((((K+2)*(K+3)+(c+3)*(K+1)+2*3)%31)/25.0-0.6) for c in range(3)],axis=1)', "call": "bin_aggregates(w, v, x)", "gold_call": "_oracle_bin_aggregates(w, v, x)", "tol": 1e-09},
        {"setup": 'import numpy as _n\nI=_n.arange(24)\nw=0.5+(((I+1)*(I+3)+5)%7)/10.0\nv=_n.stack([((((I+1)*(I+2)+(c+2)*(I+5)+5)%47)/23.5-1.0) for c in range(3)],axis=1)\nx=(((I+2)*(I+4)*5+1)%41)/41.0\nK=_n.arange(10)\nw2=0.8+(((K+1)*(K+2)+5)%5)/10.0\nv2=_n.stack([((((K+2)*(K+3)+(c+3)*(K+1)+2*5)%31)/25.0-0.6) for c in range(3)],axis=1)', "call": "bin_aggregates(w, v, x)", "gold_call": "_oracle_bin_aggregates(w, v, x)", "tol": 1e-09},
        {"setup": 'import numpy as _n\nI=_n.arange(24)\nw=0.5+(((I+1)*(I+3)+4)%7)/10.0\nv=_n.stack([((((I+1)*(I+2)+(c+2)*(I+5)+4)%47)/23.5-1.0) for c in range(3)],axis=1)\nx=(((I+2)*(I+4)*4+1)%41)/41.0\nK=_n.arange(10)\nw2=0.8+(((K+1)*(K+2)+4)%5)/10.0\nv2=_n.stack([((((K+2)*(K+3)+(c+3)*(K+1)+2*4)%31)/25.0-0.6) for c in range(3)],axis=1)', "call": "bin_aggregates(w, v, x)", "gold_call": "_oracle_bin_aggregates(w, v, x)", "tol": 1e-09},
        {"setup": 'import numpy as _n\nw=_n.array([1e-12,1.3,2e-12,0.7])\nv=_n.array([[0.4,-0.2,0.1],[-0.1,0.3,-0.4],[0.2,0.2,-0.3],[-0.5,0.1,0.6]])\nx=_n.array([0.9,0.2,0.5,0.7])', "call": "bin_aggregates(w, v, x)", "gold_call": "_oracle_bin_aggregates(w, v, x)", "tol": 1e-09},
        {"setup": 'import numpy as _n\nw=_n.array([0.85])\nv=_n.array([[0.3,-0.6,0.2]])\nx=_n.array([0.4])', "call": "bin_aggregates(w, v, x)", "gold_call": "_oracle_bin_aggregates(w, v, x)", "tol": 1e-09},
    ]
