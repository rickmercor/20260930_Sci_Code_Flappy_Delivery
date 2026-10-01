"""
Compute the per-column scaling factors of the stacked constraint matrix exactly as the source defines them, guarding against zero columns.

The source applies a second, column-wise scaling on top of the row transformations before solving; its exact definition and the stage at which it is computed follow the source.

Returns
-------
return (N,) float64: per-column scaling factors
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def column_scaling(A):
    """A: (m, N) stacked constraint matrix. Returns (N,) float64: the
    per-column scaling factors exactly as the source defines them.
    Raises ValueError if any column is identically zero."""
    return np.ones(np.asarray(A).shape[1])

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 5: column scaling of the stacked constraint matrix."""

import numpy as np


def _oracle_column_scaling(A):
    A = np.asarray(A, dtype=np.float64)
    nrm2 = np.sum(A ** 2, axis=0)
    if np.any(nrm2 <= 0):
        raise ValueError("zero column in scaled system")
    return nrm2 ** -0.5

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\nI=_n.arange(24)\nw=0.5+(((I+1)*(I+3)+3)%7)/10.0\nv=_n.stack([((((I+1)*(I+2)+(c+2)*(I+5)+3)%47)/23.5-1.0) for c in range(3)],axis=1)\nx=(((I+2)*(I+4)*3+1)%41)/41.0\nK=_n.arange(10)\nw2=0.8+(((K+1)*(K+2)+3)%5)/10.0\nv2=_n.stack([((((K+2)*(K+3)+(c+3)*(K+1)+2*3)%31)/25.0-0.6) for c in range(3)],axis=1)\nM=scaled_moment_system(w, v, x)\nR=rate_rows(w, v, w2, v2)\nA=_n.vstack([M[:, :24], R[:, :24]])', "call": "column_scaling(A)", "gold_call": "_oracle_column_scaling(A)", "tol": 1e-09},
        {"setup": 'import numpy as _n\nI=_n.arange(24)\nw=0.5+(((I+1)*(I+3)+5)%7)/10.0\nv=_n.stack([((((I+1)*(I+2)+(c+2)*(I+5)+5)%47)/23.5-1.0) for c in range(3)],axis=1)\nx=(((I+2)*(I+4)*5+1)%41)/41.0\nK=_n.arange(10)\nw2=0.8+(((K+1)*(K+2)+5)%5)/10.0\nv2=_n.stack([((((K+2)*(K+3)+(c+3)*(K+1)+2*5)%31)/25.0-0.6) for c in range(3)],axis=1)\nM=scaled_moment_system(w, v, x)\nA=M[:, :24]', "call": "column_scaling(A)", "gold_call": "_oracle_column_scaling(A)", "tol": 1e-09},
        {"setup": 'import numpy as _n\nI=_n.arange(24)\nw=0.5+(((I+1)*(I+3)+4)%7)/10.0\nv=_n.stack([((((I+1)*(I+2)+(c+2)*(I+5)+4)%47)/23.5-1.0) for c in range(3)],axis=1)\nx=(((I+2)*(I+4)*4+1)%41)/41.0\nK=_n.arange(10)\nw2=0.8+(((K+1)*(K+2)+4)%5)/10.0\nv2=_n.stack([((((K+2)*(K+3)+(c+3)*(K+1)+2*4)%31)/25.0-0.6) for c in range(3)],axis=1)\nR=rate_rows(w, v, w2, v2)\nA=R[:, :24]', "call": "column_scaling(A)", "gold_call": "_oracle_column_scaling(A)", "tol": 1e-09},
        {"setup": 'import numpy as _n\nA=_n.array([[0.3],[-1.2],[0.5],[2.0]])', "call": "column_scaling(A)", "gold_call": "_oracle_column_scaling(A)", "tol": 1e-09},
        {"setup": 'import numpy as _n\nA=_n.array([[1e-8,-2.0,3e6,0.04]])', "call": "column_scaling(A)", "gold_call": "_oracle_column_scaling(A)", "tol": 1e-09},
    ]
