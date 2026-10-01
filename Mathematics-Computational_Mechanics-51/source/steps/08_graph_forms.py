"""
Assemble the source's two graph-level bilinear forms on the free nodes and return them stacked, the mass-type form first and the weighted graph Laplacian-type form second. Both are built by summing a contribution from every edge of the network, each written in terms of the values the nodal function takes at that edge's two endpoints and weighted by that edge's length. One form sees those two endpoint values separately, the other sees only their difference, and the two carry opposite powers of the edge length together with a common numerical prefactor. The exact weights, the prefactor and which power goes with which form are the source's convention; recover them from the paper rather than assuming an unweighted graph Laplacian. Free nodes are those not in the Dirichlet set, ordered by increasing node index, with the six components of a node occupying consecutive rows; endpoints that are Dirichlet nodes contribute nothing. Raise ValueError on an edge of zero length.

Preconditioning a network problem rests on comparing the operator actually solved against much simpler operators posed directly on the graph. The comparison is only meaningful if those graph operators carry the same scaling in the edge lengths as the problem they stand in for.

Returns
-------
return (2, 6*nf, 6*nf) float64: the source's two graph-level bilinear forms on the free nodes
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def graph_forms(nodes, edges, dirichlet):
    """nodes: (n_nodes, 3) coordinates; edges: list of (a, b) node index pairs;
    dirichlet: set of fixed node indices. Returns (2, 6*nf, 6*nf) float64 with the
    mass-type form first and the weighted graph Laplacian-type form second."""
    return np.zeros((2, 6, 6))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Gold oracle: graph_forms."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")




def _oracle_graph_forms(nodes, edges, dirichlet):
    """The source's mass-type and weighted graph Laplacian-type forms (Eq 6.1).

    Returns (2, 6*nf, 6*nf) stacked as [M_G, L_G] on the free nodes.
    """
    X = np.asarray(nodes, dtype=np.float64)
    ND = set(dirichlet)
    fr = [j for j in range(X.shape[0]) if j not in ND]
    pos = {n: i for i, n in enumerate(fr)}
    nf = len(fr)
    MG = np.zeros((6 * nf, 6 * nf)); LG = np.zeros((6 * nf, 6 * nf))
    I6 = np.eye(6)
    for (a, b) in edges:
        a = int(a); b = int(b)
        he = float(np.linalg.norm(X[b] - X[a]))
        if he <= 0.0:
            raise ValueError("edge of zero length")
        for nd in (a, b):
            if nd in ND:
                continue
            r = pos[nd]
            MG[6 * r:6 * r + 6, 6 * r:6 * r + 6] += 0.5 * he * I6
        for (u, sgu) in ((a, 1.0), (b, -1.0)):
            if u in ND:
                continue
            ru = pos[u]
            for (v, sgv) in ((a, 1.0), (b, -1.0)):
                if v in ND:
                    continue
                rv = pos[v]
                LG[6 * ru:6 * ru + 6, 6 * rv:6 * rv + 6] += 0.5 / he * sgu * sgv * I6
    return np.stack([MG, LG])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nnodes = [[0.0,0.0,0.0],[1.0,0.2,-0.3],[0.4,1.1,0.5],[-0.6,0.7,1.2],[0.9,-0.8,0.6]]\nedges = [(0,1),(1,2),(2,3),(0,4),(4,2)]\ndirichlet={0,3}', "call": 'graph_forms(nodes, edges, dirichlet)', "gold_call": '_oracle_graph_forms(nodes, edges, dirichlet)', "tol": 1e-10},
        {"setup": 'import numpy as np\nnodes = [[0.0,0.0,0.0],[1.0,0.2,-0.3],[0.4,1.1,0.5],[-0.6,0.7,1.2],[0.9,-0.8,0.6]]\nedges = [(0,1),(1,2),(2,3),(0,4),(4,2)]\ndirichlet={1,4}', "call": 'graph_forms(nodes, edges, dirichlet)', "gold_call": '_oracle_graph_forms(nodes, edges, dirichlet)', "tol": 1e-10},
        {"setup": 'import numpy as np\nnodes=[[0,0,0],[1.5,0,0],[0,2.0,0],[0,0,2.5]]\nedges=[(0,1),(1,2),(2,3),(3,0)]\ndirichlet={0}', "call": 'graph_forms(nodes, edges, dirichlet)', "gold_call": '_oracle_graph_forms(nodes, edges, dirichlet)', "tol": 1e-10},
    ]
