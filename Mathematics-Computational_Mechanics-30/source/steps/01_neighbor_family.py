"""
Return the discrete family of node i on a uniform lattice of spacing dx: every other node whose distance from node i is positive and at most delta, nodes at exactly distance delta included, decided exactly (on the integer lattice offsets, not on rounded floating-point distances). Return an (n_F, 5) array whose columns are the neighbour's node index (stored as a float), the bond vector components xi_x and xi_y, the bond length, and the quadrature volume V_j assigned to that neighbour. Assign the volume the way the source's numerical study does on a uniform grid; note that it names its quadrature choice. Order rows by increasing node index. Raise ValueError if coords is not an (N, 2) array with N >= 2, if i is not a valid index, if delta or dx is not positive, if coords do not lie on a lattice of spacing dx, or if the node has no neighbours inside the horizon.

Peridynamic operators are sums over a node's neighbourhood. Near a free surface the neighbourhood is truncated, which is the whole phenomenon this method corrects, so the family must be built exactly and its volumes assigned exactly as the source's numerical study does.

Returns
-------
ndarray of float64 with shape (n_F, 5): columns node index, xi_x, xi_y, bond length, V_j.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def neighbor_family(coords, i, delta, dx):
    """ndarray of float64 with shape (n_F, 5): columns node index, xi_x, xi_y, bond length, V_j."""
    return np.zeros((1, 5), dtype=np.float64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_neighbor_family(coords, i, delta, dx):
    """Discrete family of node i on a uniform grid, with unmodified midpoint volumes.

    Returns an (n_F, 5) float64 array with columns [j, xi_x, xi_y, |xi|, V_j] for every
    neighbour j with 0 < |xi_j| <= delta, in increasing node-index order (j stored as float).
    V_j = dx^2 for every neighbour: the source's numerical study uses unmodified midpoint
    quadrature, with NO partial-volume correction for bonds cut by the horizon.
    """
    import numpy as np
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] < 2:
        raise ValueError("coords must be an (N, 2) array with N >= 2")
    if int(i) != i or not (0 <= i < coords.shape[0]):
        raise ValueError("i must be a valid node index")
    if delta <= 0.0 or dx <= 0.0:
        raise ValueError("delta and dx must be positive")
    xi = coords - coords[int(i)]
    # Membership is decided on the INTEGER lattice offsets, not on float distances: on a
    # uniform grid F_i = {j : 0 < |xi_j| <= delta} is exactly {(a,b) : 0 < a^2+b^2 <= m^2}
    # with m = delta/dx, and a node sitting exactly on the horizon (e.g. (m,0) or (3,4)
    # for m = 5) must not drop in or out with the last bit of a hypot().
    off = np.rint(xi / float(dx))
    if not np.allclose(off * float(dx), xi, rtol=0.0, atol=1e-9 * float(dx)):
        raise ValueError("coords must lie on a uniform lattice of spacing dx")
    a2b2 = off[:, 0] ** 2 + off[:, 1] ** 2
    m2 = (float(delta) / float(dx)) ** 2
    mask = (a2b2 > 0.0) & (a2b2 <= m2 * (1.0 + 1e-9))
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        raise ValueError("node has no neighbours inside the horizon")
    r = np.hypot(xi[idx, 0], xi[idx, 1])
    V = np.full(idx.size, float(dx) * float(dx))
    return np.column_stack([idx.astype(np.float64), xi[idx, 0], xi[idx, 1], r, V])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nX, Y = np.meshgrid(np.arange(9) * 0.25, np.arange(9) * 0.25, indexing='ij')\ncoords = np.column_stack([X.ravel(), Y.ravel()])\ni, delta, dx = 40, 0.75, 0.25",
            "call": "neighbor_family(coords, i, delta, dx)",
            "gold_call": "_oracle_neighbor_family(coords, i, delta, dx)",
        },
        {
            "setup": "import numpy as np\nX, Y = np.meshgrid(np.arange(9) * 0.25, np.arange(9) * 0.25, indexing='ij')\ncoords = np.column_stack([X.ravel(), Y.ravel()])\ni, delta, dx = 0, 0.75, 0.25",
            "call": "neighbor_family(coords, i, delta, dx)",
            "gold_call": "_oracle_neighbor_family(coords, i, delta, dx)",
        },
        {
            "setup": "import numpy as np\nX, Y = np.meshgrid(np.arange(11) * 0.125, np.arange(11) * 0.125, indexing='ij')\ncoords = np.column_stack([X.ravel(), Y.ravel()])\ni, delta, dx = 0, 0.625, 0.125",
            "call": "neighbor_family(coords, i, delta, dx)",
            "gold_call": "_oracle_neighbor_family(coords, i, delta, dx)",
        },
    ]
