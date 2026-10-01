"""
Assemble the global stiffness matrix of one of the two subdomains. The block occupies the unit square in x and y, is meshed with n by n elements in plan and a single element through the thickness, and every element is the box whose stiffness matrix is supplied. Number the nodes with x fastest, then y, then z, and give each node three consecutive degrees of freedom. When an element is gathered into the global matrix its eight nodes are taken in the element's own counter-clockwise local order, not in the global lexicographic order.

The source keeps the two subdomains separate and couples them only weakly through the interface, so each block is assembled on its own and no stiffness entry ever links a node on one side to a node on the other. The node numbering fixed here is the one every later interface operator indexes into.

Returns
-------
A (3*N, 3*N) float64 symmetric stiffness matrix, N being the number of nodes of the block.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def block_stiffness(n, Ke):
    """Assemble the global stiffness matrix of one of the two subdomains. A (3*N, 3*N) float64
    symmetric stiffness matrix, N being the number of nodes of the block."""
    return np.zeros((6, 6))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _node_id(n, nz, i, j, k):
    return (k * (n + 1) + j) * (n + 1) + i

def _oracle_block_stiffness(n, Ke):
    if n < 1:
        raise ValueError("n must be at least one element per side")
    Ke = np.asarray(Ke, dtype=float)
    if Ke.shape != (24, 24) or not np.all(np.isfinite(Ke)):
        raise ValueError("Ke must be a finite (24, 24) element stiffness matrix")
    nn = (n + 1) * (n + 1) * 2
    A = np.zeros((3 * nn, 3 * nn))
    for j in range(n):
        for i in range(n):
            nodes = [_node_id(n, 1, i + a, j + b, c)
                     for a, b, c in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                                     (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))]
            dofs = np.array([3 * p + d for p in nodes for d in range(3)])
            A[np.ix_(dofs, dofs)] += Ke
    return A

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nn = 2\nKe = _oracle_hex_stiffness(1.0, 0.0, 0.5, 0.5, 1.0)\n',
         'call': 'block_stiffness(n, Ke)',
         'gold_call': '_oracle_block_stiffness(n, Ke)'},
        {'setup': 'import numpy as np\nn = 3\nKe = _oracle_hex_stiffness(1000.0, 0.25, 1.0/3, 1.0/3, 1.5)\n',
         'call': 'block_stiffness(n, Ke)',
         'gold_call': '_oracle_block_stiffness(n, Ke)'},
        {'setup': 'import numpy as np\nn = 3\nKe = _oracle_hex_stiffness(5000.0, 0.3, 1.0/3, 1.0/3, 0.5)\n',
         'call': 'block_stiffness(n, Ke)',
         'gold_call': '_oracle_block_stiffness(n, Ke)'},
    ]
