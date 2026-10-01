"""
Solve the stabilised saddle point system of Eq. (17) for the interface multipliers under the given load, and return only the multipliers. The stabilising operator enters the constraint block of the system, and it enters with the sign that Eq. (17) gives it. There is no load on the constraint equations.

This is the linear system solved at each Newton and active set iteration once the whole interface is taken to be in the tied regime, Eq. (12), with the stabilising term added. Its multiplier part is the interface traction field, which is the quantity the source examines when it checks that a constant traction is transferred without oscillation.

Returns
-------
A (3*n1*n1,) float64 array of interface multiplier values, three components per non-mortar face.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def interface_tractions(A, B, H, f):
    """Solve the stabilised saddle point system of Eq. (17) for the interface multipliers under
    the given load, and return only the multipliers. A (3*n1*n1,) float64 array of interface
    multiplier values, three components per non-mortar face."""
    return np.zeros(3)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


_QUAD = ((0, 0), (1, 0), (1, 1), (0, 1))

def _node_id(n, nz, i, j, k):
    return (k * (n + 1) + j) * (n + 1) + i

def _far_face(n, k):
    out = []
    for j in range(n + 1):
        for i in range(n + 1):
            p = _node_id(n, 1, i, j, k)
            out += [3 * p, 3 * p + 1, 3 * p + 2]
    return np.array(sorted(set(out)))

def _top_load(n2, varying):
    f = np.zeros(3 * (n2 + 1) * (n2 + 1) * 2)
    h = 1.0 / n2
    a = h * h / 4.0
    for j in range(n2):
        for i in range(n2):
            for di, dj in _QUAD:
                p = _node_id(n2, 1, i + di, j + dj, 1)
                x, y = (i + di) * h, (j + dj) * h
                if varying:
                    f[3 * p + 0] += a * (0.30 * y)
                    f[3 * p + 1] += a * (0.20 * x)
                    f[3 * p + 2] += a * (-(1.0 + 0.5 * x + 0.25 * y))
                else:
                    f[3 * p + 2] += -a
    return f

def _oracle_interface_tractions(A, B, H, f):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    H = np.asarray(H, dtype=float)
    f = np.asarray(f, dtype=float)
    if f.shape[0] != A.shape[0]:
        raise ValueError("f must have one entry per displacement degree of freedom")
    nt = B.shape[1]
    K = np.block([[A, B], [B.T, -H]])
    rhs = np.concatenate([f, np.zeros(nt)])
    sol = np.linalg.solve(K, rhs)
    return sol[A.shape[0]:]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nn1 = 2\nn2 = 2\nA1 = _oracle_block_stiffness(n1, _oracle_hex_stiffness(1.0, 0.0, 1.0/n1, 1.0/n1, 1.0))\nA2 = _oracle_block_stiffness(n2, _oracle_hex_stiffness(1.0, 0.0, 1.0/n2, 1.0/n2, 1.0))\nD = _oracle_mortar_mass(n1)\nM = _oracle_mortar_coupling(n1, n2)\nimport scipy.linalg as sla\nH = _oracle_stabilization_matrix(n1, n2, A1, A2, D, M)\nL1 = np.setdiff1d(np.arange(A1.shape[0]), _far_face(n1, 0))\nL2 = np.arange(A2.shape[0])\nA = sla.block_diag(A1[np.ix_(L1, L1)], A2[np.ix_(L2, L2)])\nB = np.hstack([D[:, L1], -M[:, L2]]).T\nf = np.concatenate([np.zeros(len(L1)), _top_load(n2, True)[L2]])\n',
         'call': 'interface_tractions(A, B, H, f)',
         'gold_call': '_oracle_interface_tractions(A, B, H, f)'},
        {'setup': 'import numpy as np\nn1 = 3\nn2 = 2\nA1 = _oracle_block_stiffness(n1, _oracle_hex_stiffness(1.0, 0.0, 1.0/n1, 1.0/n1, 1.0))\nA2 = _oracle_block_stiffness(n2, _oracle_hex_stiffness(1.0, 0.0, 1.0/n2, 1.0/n2, 1.0))\nD = _oracle_mortar_mass(n1)\nM = _oracle_mortar_coupling(n1, n2)\nimport scipy.linalg as sla\nH = _oracle_stabilization_matrix(n1, n2, A1, A2, D, M)\nL1 = np.setdiff1d(np.arange(A1.shape[0]), _far_face(n1, 0))\nL2 = np.arange(A2.shape[0])\nA = sla.block_diag(A1[np.ix_(L1, L1)], A2[np.ix_(L2, L2)])\nB = np.hstack([D[:, L1], -M[:, L2]]).T\nf = np.concatenate([np.zeros(len(L1)), _top_load(n2, True)[L2]])\n',
         'call': 'interface_tractions(A, B, H, f)',
         'gold_call': '_oracle_interface_tractions(A, B, H, f)'},
        {'setup': 'import numpy as np\nn1 = 3\nn2 = 2\nA1 = _oracle_block_stiffness(n1, _oracle_hex_stiffness(1000.0, 0.25, 1.0/n1, 1.0/n1, 0.5))\nA2 = _oracle_block_stiffness(n2, _oracle_hex_stiffness(1000.0, 0.25, 1.0/n2, 1.0/n2, 0.5))\nD = _oracle_mortar_mass(n1)\nM = _oracle_mortar_coupling(n1, n2)\nimport scipy.linalg as sla\nH = _oracle_stabilization_matrix(n1, n2, A1, A2, D, M)\nL1 = np.setdiff1d(np.arange(A1.shape[0]), _far_face(n1, 0))\nL2 = np.arange(A2.shape[0])\nA = sla.block_diag(A1[np.ix_(L1, L1)], A2[np.ix_(L2, L2)])\nB = np.hstack([D[:, L1], -M[:, L2]]).T\nf = np.concatenate([np.zeros(len(L1)), _top_load(n2, True)[L2]])\n',
         'call': 'interface_tractions(A, B, H, f)',
         'gold_call': '_oracle_interface_tractions(A, B, H, f)'},
    ]
