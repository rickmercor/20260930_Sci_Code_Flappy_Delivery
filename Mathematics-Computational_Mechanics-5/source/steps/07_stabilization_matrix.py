"""
Assemble the global stabilising operator of Eq. (16) by the procedure of Algorithm 1. Sweep the internal nodes of the mortar interface grid; on each patch form the local approximation and derive from it the second order tensor that scales the contribution of every internal edge of the patch, per Eq. (20); then place that contribution into the global operator through the jump structure the bilinear form implies. Take an internal edge to be one shared by two non-mortar faces that both belong to the patch and that are neighbours across a coordinate direction. The sweep visits some edges more than once, and Algorithm 1 says what to do about that.

The stabilisation acts on the difference of the multiplier across an interior edge of the non-mortar interface, which is what makes it vanish on the traction fields the formulation must reproduce exactly while removing the spurious ones. Its strength is set locally, from the patch operator, rather than by any user chosen parameter, which is what the source claims as an advantage.

Returns
-------
A (3*n1*n1, 3*n1*n1) float64 symmetric positive semi-definite array.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def stabilization_matrix(n1, n2, A1, A2, D, M):
    """Assemble the global stabilising operator of Eq. (16) by the procedure of Algorithm 1. A
    (3*n1*n1, 3*n1*n1) float64 symmetric positive semi-definite array."""
    return np.zeros((3, 3))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _face_id(n, i, j):
    return j * n + i

def _oracle_stabilization_matrix(n1, n2, A1, A2, D, M):
    if n1 < 2 or n2 < 2:
        raise ValueError("both interface grids need at least two cells per side")
    if np.asarray(D).shape[0] != 3 * n1 * n1 or np.asarray(M).shape[0] != 3 * n1 * n1:
        raise ValueError("both interface operators must have one row block per non-mortar face")
    nt = n1 * n1
    H = np.zeros((3 * nt, 3 * nt))
    for j2 in range(1, n2):
        for i2 in range(1, n2):
            St = _oracle_local_scaling(n1, n2, i2, j2, A1, A2, D, M)
            mask = _oracle_macroelement_masks(n1, n2, i2, j2)
            ncells = [(i, j) for j in range(n1) for i in range(n1) if mask[_face_id(n1, i, j)] > 0.0]
            loc = {c: k for k, c in enumerate(ncells)}
            for (i, j) in ncells:
                for di, dj in ((1, 0), (0, 1)):
                    nb = (i + di, j + dj)
                    if nb not in loc:
                        continue
                    K, L = loc[(i, j)], loc[nb]
                    SE = 0.5 * (St[3 * K:3 * K + 3, 3 * K:3 * K + 3]
                                + St[3 * L:3 * L + 3, 3 * L:3 * L + 3])
                    gK = np.array([3 * _face_id(n1, i, j) + d for d in range(3)])
                    gL = np.array([3 * _face_id(n1, nb[0], nb[1]) + d for d in range(3)])
                    H[np.ix_(gK, gK)] += SE
                    H[np.ix_(gL, gL)] += SE
                    H[np.ix_(gK, gL)] -= SE
                    H[np.ix_(gL, gK)] -= SE
    return H

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nn1 = 2\nn2 = 2\nA1 = _oracle_block_stiffness(n1, _oracle_hex_stiffness(1.0, 0.0, 1.0/n1, 1.0/n1, 1.0))\nA2 = _oracle_block_stiffness(n2, _oracle_hex_stiffness(1.0, 0.0, 1.0/n2, 1.0/n2, 1.0))\nD = _oracle_mortar_mass(n1)\nM = _oracle_mortar_coupling(n1, n2)\n',
         'call': 'stabilization_matrix(n1, n2, A1, A2, D, M)',
         'gold_call': '_oracle_stabilization_matrix(n1, n2, A1, A2, D, M)'},
        {'setup': 'import numpy as np\nn1 = 3\nn2 = 2\nA1 = _oracle_block_stiffness(n1, _oracle_hex_stiffness(1.0, 0.0, 1.0/n1, 1.0/n1, 1.0))\nA2 = _oracle_block_stiffness(n2, _oracle_hex_stiffness(1.0, 0.0, 1.0/n2, 1.0/n2, 1.0))\nD = _oracle_mortar_mass(n1)\nM = _oracle_mortar_coupling(n1, n2)\n',
         'call': 'stabilization_matrix(n1, n2, A1, A2, D, M)',
         'gold_call': '_oracle_stabilization_matrix(n1, n2, A1, A2, D, M)'},
        {'setup': 'import numpy as np\nn1 = 3\nn2 = 3\nA1 = _oracle_block_stiffness(n1, _oracle_hex_stiffness(1.0, 0.0, 1.0/n1, 1.0/n1, 1.0))\nA2 = _oracle_block_stiffness(n2, _oracle_hex_stiffness(1.0, 0.0, 1.0/n2, 1.0/n2, 1.0))\nD = _oracle_mortar_mass(n1)\nM = _oracle_mortar_coupling(n1, n2)\n',
         'call': 'stabilization_matrix(n1, n2, A1, A2, D, M)',
         'gold_call': '_oracle_stabilization_matrix(n1, n2, A1, A2, D, M)'},
    ]
