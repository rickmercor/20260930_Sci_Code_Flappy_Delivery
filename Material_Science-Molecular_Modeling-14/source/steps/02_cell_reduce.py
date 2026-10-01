"""
Reduce a batch of momentum vectors into the reciprocal unit cell of a layer exactly by the source's reduction map for its truncation criterion.

The source measures distances to the Dirac point only after mapping shifted momenta back into the layer's reciprocal cell; which reduction map it uses is stated in the source.

Returns
-------
return (n, 2) float64: vectors reduced into the reciprocal cell
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def cell_reduce(X, B):
    """X: (n, 2) momentum vectors (a single vector may be passed as (2,));
    B: the layer reciprocal matrix, passed flattened (4,) row-major or as
    (2, 2). Returns (n, 2) float64: each vector reduced into the layer's
    reciprocal unit cell exactly by the source's reduction map."""
    return np.atleast_2d(np.asarray(X, dtype=np.float64))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 2: reduction into the layer reciprocal cell."""

import numpy as np


def _oracle_cell_reduce(X, B):
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    B = np.asarray(B, dtype=np.float64).reshape(2, 2)
    Binv = np.linalg.inv(B)
    n = np.floor(X @ Binv.T)
    return X - n @ B.T

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\ngeom=layer_geometry(_n.array([-0.06,0.0,0.11]))\nq=geom[1,8:10]+_n.array([0.03,0.02])\nX=_n.array([[3.1,-2.4],[0.2,5.5],[-4.0,1.7]])', "call": "cell_reduce(X, geom[0,4:8])", "gold_call": "_oracle_cell_reduce(X, geom[0,4:8])", "tol": 1e-09},
        {"setup": 'import numpy as _n\ngeom=layer_geometry(_n.array([-0.09,0.0,0.07]))\nq=geom[1,10:12]+_n.array([0.03,0.02])\nX=_n.array([[1.9,2.2],[-3.3,-0.7]])', "call": "cell_reduce(X, geom[2,4:8])", "gold_call": "_oracle_cell_reduce(X, geom[2,4:8])", "tol": 1e-09},
        {"setup": 'import numpy as _n\ngeom=layer_geometry(_n.array([-0.05,0.0,0.09]))\nq=geom[1,8:10]+_n.array([0.03,0.02])\nX=q[None,:]+_n.array([[0.0,0.0],[2.9,-1.2]])', "call": "cell_reduce(X, geom[1,4:8])", "gold_call": "_oracle_cell_reduce(X, geom[1,4:8])", "tol": 1e-09},
    ]
