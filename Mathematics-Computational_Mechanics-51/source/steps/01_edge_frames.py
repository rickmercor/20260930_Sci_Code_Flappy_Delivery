"""
Given the node coordinates of a spatial network and its list of edges, return one row per edge holding, in order, the three components of that edge's unit tangent vector, the edge length, and then one entry per node giving the source's orientation sign of that edge with respect to that node, or zero when the node is not an endpoint of the edge. The sign convention is the source's and is fixed by the direction of the edge tangent relative to the node; recover it from the paper. Raise ValueError on an edge of zero length.

A spatial network is a graph embedded in space whose edges carry one-dimensional equations. Each edge is parametrised in a fixed direction, so quantities meeting at a shared node arrive with edge-dependent orientations that must be reconciled before they can be combined.

Returns
-------
return (E, 4 + n_nodes) float64: tangent, length and the source's orientation signs
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def edge_frames(nodes, edges):
    """nodes: (n_nodes, 3) coordinates; edges: list of (a, b) node index pairs.
    Returns (E, 4 + n_nodes) float64 with the unit tangent in columns 0-2, the
    length in column 3, and the orientation sign of the edge with respect to
    every node in columns 4 onwards (zero for non-endpoints)."""
    return np.zeros((len(edges), 4 + len(nodes)))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Gold oracle: edge_frames."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")




def _oracle_edge_frames(nodes, edges):
    """Per-edge unit tangent, length, and the source's orientation signs.

    Returns (E, 4 + n_nodes): columns 0-2 the unit tangent, column 3 the length,
    columns 4.. the sign nu(e, n) for every node n (0 when n is not an endpoint).
    """
    X = np.asarray(nodes, dtype=np.float64)
    nn = X.shape[0]
    out = np.zeros((len(edges), 4 + nn), dtype=np.float64)
    for k, (a, b) in enumerate(edges):
        a = int(a); b = int(b)
        d = X[b] - X[a]; L = float(np.linalg.norm(d))
        if L <= 0.0:
            raise ValueError("edge of zero length")
        out[k, :3] = d / L
        out[k, 3] = L
        out[k, 4 + b] = 1.0      # tangent points TOWARD b
        out[k, 4 + a] = -1.0     # and away from a
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nnodes = [[0.0,0.0,0.0],[1.0,0.2,-0.3],[0.4,1.1,0.5],[-0.6,0.7,1.2],[0.9,-0.8,0.6]]\nedges = [(0,1),(1,2),(2,3),(0,4),(4,2)]\n', "call": 'edge_frames(nodes, edges)', "gold_call": '_oracle_edge_frames(nodes, edges)', "tol": 1e-12},
        {"setup": 'import numpy as np\nnodes = [[0.0,0.0,0.0],[1.0,0.2,-0.3],[0.4,1.1,0.5],[-0.6,0.7,1.2],[0.9,-0.8,0.6]]\nedges = [(1,0),(2,1),(3,2)]\n', "call": 'edge_frames(nodes, edges)', "gold_call": '_oracle_edge_frames(nodes, edges)', "tol": 1e-12},
        {"setup": 'import numpy as np\nnodes=[[0,0,0],[2.5,0,0],[0,-1.5,0]]\nedges=[(0,1),(2,0)]\n', "call": 'edge_frames(nodes, edges)', "gold_call": '_oracle_edge_frames(nodes, edges)', "tol": 1e-12},
    ]
