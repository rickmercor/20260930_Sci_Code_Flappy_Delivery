"""
Build the element stiffness matrix of one trilinear hexahedron of the displacement space the source uses, Eq. (14a), for an isotropic linear elastic solid. The element is a rectangular box aligned with the axes. Use the tensor-product Gauss rule that integrates this element exactly, and order the twenty four degrees of freedom node by node with the three components contiguous within each node. Take the eight local nodes in the counter-clockwise order (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1) around the lower face, followed by the same four corners on the upper face. That local ordering belongs to the element itself and is not the lexicographic ordering used to number the nodes of a block.

The source discretises both subdomains with hexahedra carrying trilinear nodal displacements. This element is completely standard; it is the block on which everything downstream is assembled, and getting its degree of freedom ordering right is what lets the interface operators be gathered by index later.

Returns
-------
A (24, 24) float64 symmetric element stiffness matrix.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def hex_stiffness(E, nu, hx, hy, hz):
    """Build the element stiffness matrix of one trilinear hexahedron of the displacement space
    the source uses, Eq. (14a), for an isotropic linear elastic solid. A (24, 24) float64
    symmetric element stiffness matrix."""
    return np.zeros((24, 24))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


_GP = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])

_CORNER = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)

def _oracle_hex_stiffness(E, nu, hx, hy, hz):
    if not np.isfinite(E) or E <= 0.0:
        raise ValueError("E must be a positive finite modulus")
    if not np.isfinite(nu) or nu <= -1.0 or nu >= 0.5:
        raise ValueError("nu must lie in (-1, 0.5)")
    if min(hx, hy, hz) <= 0.0:
        raise ValueError("element sizes must be positive")
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    Jd = np.array([hx / 2.0, hy / 2.0, hz / 2.0])
    detJ = float(np.prod(Jd))
    K = np.zeros((24, 24))
    s = _CORNER
    for xi in _GP:
        for eta in _GP:
            for zt in _GP:
                dN = np.empty((8, 3))
                dN[:, 0] = 0.125 * s[:, 0] * (1 + s[:, 1] * eta) * (1 + s[:, 2] * zt)
                dN[:, 1] = 0.125 * s[:, 1] * (1 + s[:, 0] * xi) * (1 + s[:, 2] * zt)
                dN[:, 2] = 0.125 * s[:, 2] * (1 + s[:, 0] * xi) * (1 + s[:, 1] * eta)
                g = dN / Jd[None, :]
                B = np.zeros((6, 24))
                for a in range(8):
                    B[0, 3 * a + 0] = g[a, 0]
                    B[1, 3 * a + 1] = g[a, 1]
                    B[2, 3 * a + 2] = g[a, 2]
                    B[3, 3 * a + 1] = g[a, 2]
                    B[3, 3 * a + 2] = g[a, 1]
                    B[4, 3 * a + 0] = g[a, 2]
                    B[4, 3 * a + 2] = g[a, 0]
                    B[5, 3 * a + 0] = g[a, 1]
                    B[5, 3 * a + 1] = g[a, 0]
                K += (B.T @ C @ B) * detJ
    return K

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nE = 1.0\nnu = 0.0\nhx = 0.25\nhy = 0.25\nhz = 1.0\n',
         'call': 'hex_stiffness(E, nu, hx, hy, hz)',
         'gold_call': '_oracle_hex_stiffness(E, nu, hx, hy, hz)'},
        {'setup': 'import numpy as np\nE = 1000.0\nnu = 0.25\nhx = 0.5\nhy = 0.125\nhz = 0.75\n',
         'call': 'hex_stiffness(E, nu, hx, hy, hz)',
         'gold_call': '_oracle_hex_stiffness(E, nu, hx, hy, hz)'},
        {'setup': 'import numpy as np\nE = 15000.0\nnu = 0.4\nhx = 1.0\nhy = 1.0\nhz = 1.0\n',
         'call': 'hex_stiffness(E, nu, hx, hy, hz)',
         'gold_call': '_oracle_hex_stiffness(E, nu, hx, hy, hz)'},
    ]
