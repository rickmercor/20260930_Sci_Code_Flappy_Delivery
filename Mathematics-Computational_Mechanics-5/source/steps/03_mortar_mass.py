"""
Assemble the first of the two interface operators of Eq. (11), pairing every multiplier basis function with the displacement basis functions of the non-mortar side over the interface. The multipliers are the vector piecewise constants of Eq. (14b), one per non-mortar interface face; the displacements are the traces of the trilinear block functions. Each of the three components pairs only with its own component.

The multipliers live on the non-mortar side, so this pairing is an integral of a constant against a bilinear trace over a face that both functions share. The source calls the result a standard mass matrix. It is the half of the constraint that carries no non-conformity, and it is exact.

Returns
-------
A (3*n1*n1, 3*N1) float64 array, N1 being the number of nodes of the non-mortar block.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def mortar_mass(n1):
    """Assemble the first of the two interface operators of Eq. (11), pairing every multiplier
    basis function with the displacement basis functions of the non-mortar side over the
    interface. A (3*n1*n1, 3*N1) float64 array, N1 being the number of nodes of the non-
    mortar block."""
    return np.zeros((3, 3))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


_QUAD = ((0, 0), (1, 0), (1, 1), (0, 1))

def _edges(n):
    return np.linspace(0.0, 1.0, n + 1)

def _hat_integral(a, b, c0, c1, at_lo):
    """Integral over [a,b] of the 1D hat on [c0,c1] equal to one at c0 (at_lo) or at c1."""
    L = c1 - c0
    if at_lo:
        return ((c1 - a) ** 2 - (c1 - b) ** 2) / (2.0 * L)
    return ((b - c0) ** 2 - (a - c0) ** 2) / (2.0 * L)

def _node_id(n, nz, i, j, k):
    return (k * (n + 1) + j) * (n + 1) + i

def _face_id(n, i, j):
    return j * n + i

def _oracle_mortar_mass(n1):
    if n1 < 1:
        raise ValueError("n1 must be at least one")
    e = _edges(n1)
    nn = (n1 + 1) * (n1 + 1) * 2
    D = np.zeros((3 * n1 * n1, 3 * nn))
    for j in range(n1):
        for i in range(n1):
            f = _face_id(n1, i, j)
            x0, x1 = e[i], e[i + 1]
            y0, y1 = e[j], e[j + 1]
            for di, dj in _QUAD:
                node = _node_id(n1, 1, i + di, j + dj, 1)
                v = _hat_integral(x0, x1, x0, x1, di == 0) * _hat_integral(y0, y1, y0, y1, dj == 0)
                for d in range(3):
                    D[3 * f + d, 3 * node + d] += v
    return D

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nn1 = 2\n',
         'call': 'mortar_mass(n1)',
         'gold_call': '_oracle_mortar_mass(n1)'},
        {'setup': 'import numpy as np\nn1 = 4\n',
         'call': 'mortar_mass(n1)',
         'gold_call': '_oracle_mortar_mass(n1)'},
        {'setup': 'import numpy as np\nn1 = 5\n',
         'call': 'mortar_mass(n1)',
         'gold_call': '_oracle_mortar_mass(n1)'},
    ]
