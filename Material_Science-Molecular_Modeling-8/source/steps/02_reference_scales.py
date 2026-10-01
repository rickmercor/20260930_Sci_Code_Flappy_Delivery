"""
Compute the reference velocity and position scales of the merge group exactly as the source defines them for its scaled system, packed as a length-4 vector.

The source scales its ill-conditioned moment system with reference scales derived from the particle group itself; how those scales are normalized is the source's choice.

Returns
-------
return (4,) float64: [v_ref(3), x_ref]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def reference_scales(w, v, x):
    """w: (N,) weights; v: (N, 3) velocities; x: (N,) positions.
    Returns (4,) float64: the three components of the reference velocity
    scale and the reference position scale, exactly as the source defines
    them. Raises ValueError if any scale degenerates to zero."""
    return np.ones(4)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 2: reference scales for the scaled moment system."""

import numpy as np


def _oracle_reference_scales(w, v, x):
    w = np.asarray(w, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    agg = _oracle_bin_aggregates(w, v, x)
    wS, vS, xS = agg[0], agg[1:4], agg[4]
    vref = np.sqrt((w @ (v - vS) ** 2) / wS)
    xref = float(np.sqrt(w @ (x - xS) ** 2 / wS))
    if np.any(vref <= 0) or xref <= 0:
        raise ValueError("degenerate reference scale")
    return np.concatenate([vref, [xref]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\nI=_n.arange(24)\nw=0.5+(((I+1)*(I+3)+3)%7)/10.0\nv=_n.stack([((((I+1)*(I+2)+(c+2)*(I+5)+3)%47)/23.5-1.0) for c in range(3)],axis=1)\nx=(((I+2)*(I+4)*3+1)%41)/41.0\nK=_n.arange(10)\nw2=0.8+(((K+1)*(K+2)+3)%5)/10.0\nv2=_n.stack([((((K+2)*(K+3)+(c+3)*(K+1)+2*3)%31)/25.0-0.6) for c in range(3)],axis=1)', "call": "reference_scales(w, v, x)", "gold_call": "_oracle_reference_scales(w, v, x)", "tol": 1e-09},
        {"setup": 'import numpy as _n\nI=_n.arange(24)\nw=0.5+(((I+1)*(I+3)+5)%7)/10.0\nv=_n.stack([((((I+1)*(I+2)+(c+2)*(I+5)+5)%47)/23.5-1.0) for c in range(3)],axis=1)\nx=(((I+2)*(I+4)*5+1)%41)/41.0\nK=_n.arange(10)\nw2=0.8+(((K+1)*(K+2)+5)%5)/10.0\nv2=_n.stack([((((K+2)*(K+3)+(c+3)*(K+1)+2*5)%31)/25.0-0.6) for c in range(3)],axis=1)', "call": "reference_scales(w, v, x)", "gold_call": "_oracle_reference_scales(w, v, x)", "tol": 1e-09},
        {"setup": 'import numpy as _n\nI=_n.arange(24)\nw=0.5+(((I+1)*(I+3)+4)%7)/10.0\nv=_n.stack([((((I+1)*(I+2)+(c+2)*(I+5)+4)%47)/23.5-1.0) for c in range(3)],axis=1)\nx=(((I+2)*(I+4)*4+1)%41)/41.0\nK=_n.arange(10)\nw2=0.8+(((K+1)*(K+2)+4)%5)/10.0\nv2=_n.stack([((((K+2)*(K+3)+(c+3)*(K+1)+2*4)%31)/25.0-0.6) for c in range(3)],axis=1)', "call": "reference_scales(w, v, x)", "gold_call": "_oracle_reference_scales(w, v, x)", "tol": 1e-09},
        {"setup": 'import numpy as _n\nb=_n.array([0.2,-0.3,0.5])\nw=_n.array([1.0,0.8,1.2,0.6])\nv=b[None,:]+1e-6*_n.array([[1.0,-1.0,0.5],[-1.0,0.5,-1.0],[0.5,1.0,1.0],[-0.5,-0.5,-0.5]])\nx=0.4+1e-6*_n.array([1.0,-1.0,0.5,-0.5])', "call": "reference_scales(w, v, x)", "gold_call": "_oracle_reference_scales(w, v, x)", "tol": 1e-09},
        {"setup": 'import numpy as _n\nw=_n.array([0.7,1.1])\nv=_n.array([[0.2,-0.4,0.1],[-0.3,0.5,-0.2]])\nx=_n.array([0.25,0.75])', "call": "reference_scales(w, v, x)", "gold_call": "_oracle_reference_scales(w, v, x)", "tol": 1e-09},
    ]
