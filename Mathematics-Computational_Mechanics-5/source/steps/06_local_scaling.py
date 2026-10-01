"""
Form the local approximation of Eq. (19) on the patch of one mortar node. Gather the two blocks' stiffness entries for the patch displacement degrees of freedom and the two interface operators for its multiplier degrees of freedom, then combine them into an operator on the patch multiplier space. The source states exactly which part of the gathered stiffness it retains and why, immediately below the equation, and that choice is what fixes the magnitude of everything downstream.

The stabilisation must be scaled so its entries sit at the same order as the object it is added to. The source obtains that scale from a cheap local surrogate of that object, cheap precisely because of what it discards from the gathered stiffness. Patch degrees of freedom are ordered non-mortar side first, then mortar side, each in ascending global index.

Returns
-------
A (3*m, 3*m) float64 symmetric array on the patch multiplier space, m being the number of non-mortar faces in the patch.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def local_scaling(n1, n2, i2, j2, A1, A2, D, M):
    """Form the local approximation of Eq. (19) on the patch of one mortar node. A (3*m, 3*m)
    float64 symmetric array on the patch multiplier space, m being the number of non-mortar
    faces in the patch."""
    return np.zeros((3, 3))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


_QUAD = ((0, 0), (1, 0), (1, 1), (0, 1))

def _node_id(n, nz, i, j, k):
    return (k * (n + 1) + j) * (n + 1) + i

def _face_id(n, i, j):
    return j * n + i

def _oracle_local_scaling(n1, n2, i2, j2, A1, A2, D, M):
    mask = _oracle_macroelement_masks(n1, n2, i2, j2)
    ncells = [(i, j) for j in range(n1) for i in range(n1) if mask[_face_id(n1, i, j)] > 0.0]
    mcells = [(a, b) for b in range(n2) for a in range(n2) if mask[n1 * n1 + _face_id(n2, a, b)] > 0.0]
    It = np.array([3 * _face_id(n1, i, j) + d for (i, j) in ncells for d in range(3)])
    n1set = sorted({_node_id(n1, 1, i + di, j + dj, 1) for (i, j) in ncells for di, dj in _QUAD})
    n2set = sorted({_node_id(n2, 1, a + da, b + db, 0) for (a, b) in mcells for da, db in _QUAD})
    Iu1 = np.array([3 * p + d for p in n1set for d in range(3)])
    Iu2 = np.array([3 * p + d for p in n2set for d in range(3)])
    dA = np.concatenate([np.diag(A1[np.ix_(Iu1, Iu1)]), np.diag(A2[np.ix_(Iu2, Iu2)])])
    if np.any(dA <= 0.0):
        raise ValueError("the gathered local stiffness diagonal must be positive")
    Bh = np.hstack([D[np.ix_(It, Iu1)], -M[np.ix_(It, Iu2)]])
    return Bh @ (Bh / dA[None, :]).T

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nn1 = 2\nn2 = 2\ni2 = 1\nj2 = 1\nA1 = _oracle_block_stiffness(n1, _oracle_hex_stiffness(1.0, 0.0, 1.0/n1, 1.0/n1, 1.0))\nA2 = _oracle_block_stiffness(n2, _oracle_hex_stiffness(1.0, 0.0, 1.0/n2, 1.0/n2, 1.0))\nD = _oracle_mortar_mass(n1)\nM = _oracle_mortar_coupling(n1, n2)\n',
         'call': 'local_scaling(n1, n2, i2, j2, A1, A2, D, M)',
         'gold_call': '_oracle_local_scaling(n1, n2, i2, j2, A1, A2, D, M)'},
        {'setup': 'import numpy as np\nn1 = 3\nn2 = 2\ni2 = 1\nj2 = 1\nA1 = _oracle_block_stiffness(n1, _oracle_hex_stiffness(1000.0, 0.25, 1.0/n1, 1.0/n1, 0.5))\nA2 = _oracle_block_stiffness(n2, _oracle_hex_stiffness(1000.0, 0.25, 1.0/n2, 1.0/n2, 0.5))\nD = _oracle_mortar_mass(n1)\nM = _oracle_mortar_coupling(n1, n2)\n',
         'call': 'local_scaling(n1, n2, i2, j2, A1, A2, D, M)',
         'gold_call': '_oracle_local_scaling(n1, n2, i2, j2, A1, A2, D, M)'},
        {'setup': 'import numpy as np\nn1 = 3\nn2 = 3\ni2 = 2\nj2 = 1\nA1 = _oracle_block_stiffness(n1, _oracle_hex_stiffness(1.0, 0.0, 1.0/n1, 1.0/n1, 1.0))\nA2 = _oracle_block_stiffness(n2, _oracle_hex_stiffness(1.0, 0.0, 1.0/n2, 1.0/n2, 1.0))\nD = _oracle_mortar_mass(n1)\nM = _oracle_mortar_coupling(n1, n2)\n',
         'call': 'local_scaling(n1, n2, i2, j2, A1, A2, D, M)',
         'gold_call': '_oracle_local_scaling(n1, n2, i2, j2, A1, A2, D, M)'},
    ]
