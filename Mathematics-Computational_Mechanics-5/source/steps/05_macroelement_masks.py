"""
Identify the local patch the source builds around one internal node of the mortar interface grid, as the configuration defines it: the mortar faces touching that node, together with every non-mortar face whose area overlaps the region they cover. Return the membership of both sets as a single indicator vector, mortar faces after non-mortar faces, each face numbered with the x index fastest.

The scaling of the stabilisation is derived locally rather than globally, so the method needs a neighbourhood around each interface node on which a local problem can be posed. Because the two interface grids are non-conforming, the non-mortar faces that belong to the patch are found by overlap rather than by connectivity.

Returns
-------
A (n1*n1 + n2*n2,) float64 indicator vector holding 1.0 for a member face and 0.0 otherwise.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def macroelement_masks(n1, n2, i2, j2):
    """Identify the local patch the source builds around one internal node of the mortar
    interface grid, as the configuration defines it: the mortar faces touching that node,
    together with every non-mortar face whose area overlaps the region they cover. A (n1*n1
    + n2*n2,) float64 indicator vector holding 1.0 for a member face and 0.0 otherwise."""
    return np.zeros(2)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _edges(n):
    return np.linspace(0.0, 1.0, n + 1)

def _overlap(a0, a1, b0, b1):
    lo, hi = max(a0, b0), min(a1, b1)
    return max(0.0, hi - lo), lo, hi

def _face_id(n, i, j):
    return j * n + i

def _oracle_macroelement_masks(n1, n2, i2, j2):
    if not (1 <= i2 <= n2 - 1 and 1 <= j2 <= n2 - 1):
        raise ValueError("(i2, j2) must be an internal node of the mortar interface grid")
    mortar = [(a, b) for a in (i2 - 1, i2) for b in (j2 - 1, j2)
              if 0 <= a < n2 and 0 <= b < n2]
    e1, e2 = _edges(n1), _edges(n2)
    X0 = min(e2[a] for a, _ in mortar)
    X1 = max(e2[a + 1] for a, _ in mortar)
    Y0 = min(e2[b] for _, b in mortar)
    Y1 = max(e2[b + 1] for _, b in mortar)
    m = np.zeros(n1 * n1 + n2 * n2)
    for j in range(n1):
        for i in range(n1):
            lx, _, _ = _overlap(e1[i], e1[i + 1], X0, X1)
            ly, _, _ = _overlap(e1[j], e1[j + 1], Y0, Y1)
            if lx > 1e-14 and ly > 1e-14:
                m[_face_id(n1, i, j)] = 1.0
    for a, b in mortar:
        m[n1 * n1 + _face_id(n2, a, b)] = 1.0
    return m

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nn1 = 4\nn2 = 2\ni2 = 1\nj2 = 1\n',
         'call': 'macroelement_masks(n1, n2, i2, j2)',
         'gold_call': '_oracle_macroelement_masks(n1, n2, i2, j2)'},
        {'setup': 'import numpy as np\nn1 = 6\nn2 = 3\ni2 = 1\nj2 = 2\n',
         'call': 'macroelement_masks(n1, n2, i2, j2)',
         'gold_call': '_oracle_macroelement_masks(n1, n2, i2, j2)'},
        {'setup': 'import numpy as np\nn1 = 8\nn2 = 4\ni2 = 3\nj2 = 1\n',
         'call': 'macroelement_masks(n1, n2, i2, j2)',
         'gold_call': '_oracle_macroelement_masks(n1, n2, i2, j2)'},
    ]
