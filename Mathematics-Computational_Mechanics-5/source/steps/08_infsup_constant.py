"""
Evaluate the discrete inf-sup constant of the saddle point problem through the characterisation the source derives, Eq. (24). Build the constrained operator on the multiplier space from the given stiffness and constraint operators, add the given stabilising operator to it, and extract the constant from the extreme generalised eigenvalue against the given scaled multiplier metric. Return the constant itself, not its square, and never a negative number.

Eq. (13) is the solvability condition of the mixed formulation. The source weakens it to Eq. (22) to avoid the fractional norm, then rewrites it in matrix form so it can be evaluated on a sequence of meshes. Without the stabilising term the constrained operator has a kernel or a vanishing extreme eigenvalue, which is exactly the instability the method is designed to remove.

Returns
-------
A Python float, the inf-sup constant.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def infsup_constant(A, B, Q, H, h):
    """Evaluate the discrete inf-sup constant of the saddle point problem through the
    characterisation the source derives, Eq. (24). A Python float, the inf-sup constant."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _node_id(n, nz, i, j, k):
    return (k * (n + 1) + j) * (n + 1) + i

def _far_face(n, k):
    out = []
    for j in range(n + 1):
        for i in range(n + 1):
            p = _node_id(n, 1, i, j, k)
            out += [3 * p, 3 * p + 1, 3 * p + 2]
    return np.array(sorted(set(out)))

def _oracle_infsup_constant(A, B, Q, H, h):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    H = np.asarray(H, dtype=float)
    if A.shape[0] != A.shape[1] or B.shape[0] != A.shape[0]:
        raise ValueError("A must be square and B must have as many rows as A")
    if Q.shape != H.shape or Q.shape[0] != B.shape[1]:
        raise ValueError("Q and H must be square of size equal to the number of tractions")
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("h must be a positive finite mesh size")
    import scipy.linalg as _sla
    S = B.T @ np.linalg.solve(A, B)
    w = _sla.eigvalsh(S + H, h * Q)
    return float(np.sqrt(max(float(w[0]), 0.0)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nn1 = 2\nn2 = 2\nA1 = _oracle_block_stiffness(n1, _oracle_hex_stiffness(1.0, 0.0, 1.0/n1, 1.0/n1, 1.0))\nA2 = _oracle_block_stiffness(n2, _oracle_hex_stiffness(1.0, 0.0, 1.0/n2, 1.0/n2, 1.0))\nD = _oracle_mortar_mass(n1)\nM = _oracle_mortar_coupling(n1, n2)\nimport scipy.linalg as sla\nH = _oracle_stabilization_matrix(n1, n2, A1, A2, D, M)\nS1 = np.setdiff1d(np.arange(A1.shape[0]), _far_face(n1, 0))\nS2 = np.setdiff1d(np.arange(A2.shape[0]), _far_face(n2, 1))\nA = sla.block_diag(A1[np.ix_(S1, S1)], A2[np.ix_(S2, S2)])\nB = np.hstack([D[:, S1], -M[:, S2]]).T\nQ = np.diag(np.repeat(1.0 / (n1 * n1), 3 * n1 * n1))\nh = 1.0 / n1\n',
         'call': 'infsup_constant(A, B, Q, H, h)',
         'gold_call': '_oracle_infsup_constant(A, B, Q, H, h)'},
        {'setup': 'import numpy as np\nn1 = 3\nn2 = 2\nA1 = _oracle_block_stiffness(n1, _oracle_hex_stiffness(1.0, 0.0, 1.0/n1, 1.0/n1, 1.0))\nA2 = _oracle_block_stiffness(n2, _oracle_hex_stiffness(1.0, 0.0, 1.0/n2, 1.0/n2, 1.0))\nD = _oracle_mortar_mass(n1)\nM = _oracle_mortar_coupling(n1, n2)\nimport scipy.linalg as sla\nH = _oracle_stabilization_matrix(n1, n2, A1, A2, D, M)\nS1 = np.setdiff1d(np.arange(A1.shape[0]), _far_face(n1, 0))\nS2 = np.setdiff1d(np.arange(A2.shape[0]), _far_face(n2, 1))\nA = sla.block_diag(A1[np.ix_(S1, S1)], A2[np.ix_(S2, S2)])\nB = np.hstack([D[:, S1], -M[:, S2]]).T\nQ = np.diag(np.repeat(1.0 / (n1 * n1), 3 * n1 * n1))\nh = 1.0 / n1\n',
         'call': 'infsup_constant(A, B, Q, H, h)',
         'gold_call': '_oracle_infsup_constant(A, B, Q, H, h)'},
        {'setup': 'import numpy as np\nn1 = 3\nn2 = 3\nA1 = _oracle_block_stiffness(n1, _oracle_hex_stiffness(1.0, 0.0, 1.0/n1, 1.0/n1, 1.0))\nA2 = _oracle_block_stiffness(n2, _oracle_hex_stiffness(1.0, 0.0, 1.0/n2, 1.0/n2, 1.0))\nD = _oracle_mortar_mass(n1)\nM = _oracle_mortar_coupling(n1, n2)\nimport scipy.linalg as sla\nH = _oracle_stabilization_matrix(n1, n2, A1, A2, D, M)\nS1 = np.setdiff1d(np.arange(A1.shape[0]), _far_face(n1, 0))\nS2 = np.setdiff1d(np.arange(A2.shape[0]), _far_face(n2, 1))\nA = sla.block_diag(A1[np.ix_(S1, S1)], A2[np.ix_(S2, S2)])\nB = np.hstack([D[:, S1], -M[:, S2]]).T\nQ = np.diag(np.repeat(1.0 / (n1 * n1), 3 * n1 * n1))\nh = 1.0 / n1\n',
         'call': 'infsup_constant(A, B, Q, H, h)',
         'gold_call': '_oracle_infsup_constant(A, B, Q, H, h)'},
    ]
