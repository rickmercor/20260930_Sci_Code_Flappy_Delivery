"""
Assemble the second interface operator of Eq. (11), pairing every multiplier with the displacement basis functions of the mortar side after they have been mapped onto the non-mortar interface. The two interface grids do not match, so the integrand is discontinuous over a non-mortar face; the source names this the major algorithmic challenge of the method and it must be integrated as the source intends rather than approximated.

The operator carries the projection of Eq. (2), which for a planar interface with a constant normal reduces to the in-plane identity, so the difficulty is purely one of integrating across two unrelated grids. The mortar and non-mortar cell edges are axis aligned here, so every region over which both integrands are smooth is a rectangle.

Returns
-------
A (3*n1*n1, 3*N2) float64 array, N2 being the number of nodes of the mortar block.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def mortar_coupling(n1, n2):
    """Assemble the second interface operator of Eq. (11), pairing every multiplier with the
    displacement basis functions of the mortar side after they have been mapped onto the
    non-mortar interface. A (3*n1*n1, 3*N2) float64 array, N2 being the number of nodes of
    the mortar block."""
    return np.zeros((3, 3))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


_QUAD = ((0, 0), (1, 0), (1, 1), (0, 1))

def _edges(n):
    return np.linspace(0.0, 1.0, n + 1)

def _overlap(a0, a1, b0, b1):
    lo, hi = max(a0, b0), min(a1, b1)
    return max(0.0, hi - lo), lo, hi

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

def _oracle_mortar_coupling(n1, n2):
    if n1 < 1 or n2 < 1:
        raise ValueError("n1 and n2 must be at least one")
    e1, e2 = _edges(n1), _edges(n2)
    nn2 = (n2 + 1) * (n2 + 1) * 2
    M = np.zeros((3 * n1 * n1, 3 * nn2))
    for j in range(n1):
        for i in range(n1):
            f = _face_id(n1, i, j)
            X0, X1, Y0, Y1 = e1[i], e1[i + 1], e1[j], e1[j + 1]
            for j2 in range(n2):
                ly, y0, y1 = _overlap(Y0, Y1, e2[j2], e2[j2 + 1])
                if ly <= 0.0:
                    continue
                for i2 in range(n2):
                    lx, x0, x1 = _overlap(X0, X1, e2[i2], e2[i2 + 1])
                    if lx <= 0.0:
                        continue
                    for di, dj in _QUAD:
                        node = _node_id(n2, 1, i2 + di, j2 + dj, 0)
                        v = (_hat_integral(x0, x1, e2[i2], e2[i2 + 1], di == 0)
                             * _hat_integral(y0, y1, e2[j2], e2[j2 + 1], dj == 0))
                        for d in range(3):
                            M[3 * f + d, 3 * node + d] += v
    return M

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nn1 = 4\nn2 = 2\n',
         'call': 'mortar_coupling(n1, n2)',
         'gold_call': '_oracle_mortar_coupling(n1, n2)'},
        {'setup': 'import numpy as np\nn1 = 6\nn2 = 3\n',
         'call': 'mortar_coupling(n1, n2)',
         'gold_call': '_oracle_mortar_coupling(n1, n2)'},
        {'setup': 'import numpy as np\nn1 = 5\nn2 = 3\n',
         'call': 'mortar_coupling(n1, n2)',
         'gold_call': '_oracle_mortar_coupling(n1, n2)'},
    ]
