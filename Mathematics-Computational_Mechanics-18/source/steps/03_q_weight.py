"""
Return the nodal values of the weight (virtual crack extension) function used by the domain-form J-integral, for local nodes given by their position relative to the crack tip at the origin (the paper's Eqs. 21-23). A node takes the value one when it lies within the source's declared J-integration radius of the tip and zero outside. Recover the exact radius rule from the paper (it is a fixed multiple of the local element size).

The equivalent-domain-integral form of the J-integral replaces the contour by a weight function that is one on an inner region around the tip and zero on the outer boundary; its gradient localises the integral to a ring of elements.

Returns
-------
return (N,) float64: nodal weight-function values, 1.0 inside the J-domain
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def q_weight(node_xy, h_L):
    """node_xy: (N, 2) local node coordinates relative to the crack tip at the origin;
    h_L: local element size. Returns a float64 array (N,): 1.0 for a node inside the
    source's J-integration radius, else 0.0 (paper Eqs. 21-23). Raises ValueError on a
    non-positive or nonfinite h_L."""
    return np.zeros(np.asarray(node_xy).shape[0])

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 3: J-integral weight-function nodal values (Eqs. 21-23)."""

import numpy as np


def _oracle_q_weight(node_xy, h_L):
    xy = np.asarray(node_xy, dtype=np.float64)
    if not np.isfinite(h_L) or h_L <= 0:
        raise ValueError("h_L must be positive and finite")
    R_J = 1.5 * h_L
    r = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    return (r <= R_J).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\nnode_xy=_n.array([[0.,0.],[0.5,0.],[1.0,1.0],[1.5,1.5],[-1.5,0.5]]);h_L=1.0', "call": 'q_weight(node_xy, h_L)', "gold_call": '_oracle_q_weight(node_xy, h_L)', "tol": 1e-12},
        {"setup": 'import numpy as _n\nnode_xy=_n.array([[0.2,-0.3],[0.9,0.9],[1.2,0.4],[-1.0,-1.0],[0.6,1.3]]);h_L=0.8', "call": 'q_weight(node_xy, h_L)', "gold_call": '_oracle_q_weight(node_xy, h_L)', "tol": 1e-12},
        {"setup": 'import numpy as _n\nnode_xy=_n.array([[0.,0.],[2.0,0.],[0.7,0.7],[-0.4,1.1],[1.4,-1.4]]);h_L=1.2', "call": 'q_weight(node_xy, h_L)', "gold_call": '_oracle_q_weight(node_xy, h_L)', "tol": 1e-12},
    ]
