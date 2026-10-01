"""
Assemble the two rate-preservation rows against the fixed background species for the declared cross sections sigma_1(g) = 1/(1+g^2) and sigma_2(g) = g/(1+g), with reference rate coefficients equal to one, using the particle velocities in the form the source requires for rate rows and the source's row scaling, with the consistent right-hand side as the final column, expressed in the same unknown as the scaled moment block.

The source extends the moment system with collision-rate rows; in which coordinates those rows are evaluated, and how each row is scaled, are stated explicitly in the source and differ from the moment block.

Returns
-------
return (2, N+1) float64: rate rows plus right-hand side
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def rate_rows(w, v, w2, v2):
    """w, v: (N,) weights and (N, 3) velocities of the merged species;
    w2, v2: (M,) weights and (M, 3) velocities of the fixed background
    species. Returns (2, N+1) float64: one row per declared cross section,
    built and scaled exactly as the source prescribes for rate preservation,
    with the consistent right-hand side as the final column. Raises
    ValueError on shape mismatch."""
    return np.zeros((2, np.asarray(v).shape[0] + 1))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 4: rate-preservation rows against the fixed background species."""

import numpy as np


def _sigma(r, g):
    if r == 0:
        return 1.0 / (1.0 + g * g)
    return g / (1.0 + g)


def _oracle_rate_rows(w, v, w2, v2):
    w = np.asarray(w, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w2 = np.asarray(w2, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3 or w.shape[0] != v.shape[0] or v2.shape != (w2.shape[0], 3):
        raise ValueError("shape mismatch")
    n = v.shape[0]
    wS = float(np.sum(w))
    wS2 = float(np.sum(w2))
    out = np.empty((2, n + 1))
    for r in range(2):
        c = np.empty(n)
        for i in range(n):
            g = np.linalg.norm(v[i] - v2, axis=1)
            c[i] = float(np.sum(w2 * g * _sigma(r, g)))
        m_r = float(np.sum(w * c))
        scale = wS * wS2
        out[r, :n] = (wS * c) / scale
        out[r, n] = m_r / scale
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\nI=_n.arange(24)\nw=0.5+(((I+1)*(I+3)+3)%7)/10.0\nv=_n.stack([((((I+1)*(I+2)+(c+2)*(I+5)+3)%47)/23.5-1.0) for c in range(3)],axis=1)\nx=(((I+2)*(I+4)*3+1)%41)/41.0\nK=_n.arange(10)\nw2=0.8+(((K+1)*(K+2)+3)%5)/10.0\nv2=_n.stack([((((K+2)*(K+3)+(c+3)*(K+1)+2*3)%31)/25.0-0.6) for c in range(3)],axis=1)', "call": "rate_rows(w, v, w2, v2)", "gold_call": "_oracle_rate_rows(w, v, w2, v2)", "tol": 1e-09},
        {"setup": 'import numpy as _n\nI=_n.arange(24)\nw=0.5+(((I+1)*(I+3)+5)%7)/10.0\nv=_n.stack([((((I+1)*(I+2)+(c+2)*(I+5)+5)%47)/23.5-1.0) for c in range(3)],axis=1)\nx=(((I+2)*(I+4)*5+1)%41)/41.0\nK=_n.arange(10)\nw2=0.8+(((K+1)*(K+2)+5)%5)/10.0\nv2=_n.stack([((((K+2)*(K+3)+(c+3)*(K+1)+2*5)%31)/25.0-0.6) for c in range(3)],axis=1)', "call": "rate_rows(w, v, w2, v2)", "gold_call": "_oracle_rate_rows(w, v, w2, v2)", "tol": 1e-09},
        {"setup": 'import numpy as _n\nI=_n.arange(24)\nw=0.5+(((I+1)*(I+3)+4)%7)/10.0\nv=_n.stack([((((I+1)*(I+2)+(c+2)*(I+5)+4)%47)/23.5-1.0) for c in range(3)],axis=1)\nx=(((I+2)*(I+4)*4+1)%41)/41.0\nK=_n.arange(10)\nw2=0.8+(((K+1)*(K+2)+4)%5)/10.0\nv2=_n.stack([((((K+2)*(K+3)+(c+3)*(K+1)+2*4)%31)/25.0-0.6) for c in range(3)],axis=1)', "call": "rate_rows(w, v, w2, v2)", "gold_call": "_oracle_rate_rows(w, v, w2, v2)", "tol": 1e-09},
        {"setup": 'import numpy as _n\nw=_n.array([0.9,1.4,0.6])\nv=_n.array([[0.2,-0.1,0.3],[-0.4,0.5,-0.2],[0.1,0.3,-0.5]])\nw2=_n.array([1.2])\nv2=_n.array([[0.05,-0.15,0.1]])', "call": "rate_rows(w, v, w2, v2)", "gold_call": "_oracle_rate_rows(w, v, w2, v2)", "tol": 1e-09},
        {"setup": 'import numpy as _n\nw=_n.array([1.0,0.5])\nv=_n.array([[0.1,0.0,-0.1],[0.4,-0.3,0.2]])\nw2=_n.array([0.9,1.2])\nv2=_n.array([[0.1,0.0,-0.1],[-0.2,0.5,0.3]])', "call": "rate_rows(w, v, w2, v2)", "gold_call": "_oracle_rate_rows(w, v, w2, v2)", "tol": 1e-09},
    ]
