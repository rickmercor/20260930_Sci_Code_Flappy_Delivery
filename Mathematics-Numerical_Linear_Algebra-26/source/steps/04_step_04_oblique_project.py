"""
Implement oblique_project for a rank-r matrix state.

Given orthonormal factors U and V of Y = U Sigma V* and a same-sized matrix Z, return the complex oblique projection of Z onto the tangent space at Y. The data-sparse interpolation formula must be recovered from the literature on interpolatory dynamical low-rank approximation.

An oblique tangent-space projector represents the local rank-r dynamics using interpolated rows and columns rather than a dense orthogonal projection. The orthonormal factors U and V define the tangent point, while the selected basis submatrices determine how Z is reconstructed from sampled entries.

Returns
-------
np.ndarray as specified by the function Returns section.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def oblique_project(U: np.ndarray, V: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Compute the oblique tangent-space projection P_Y^angle[Z] using DEIM
    interpolation indices.

    Parameters
    ----------
    U : np.ndarray, shape (n, r)
        Left orthonormal factor of Y = U Sigma V*.
    V : np.ndarray, shape (n, r)
        Right orthonormal factor of Y = U Sigma V*.
    Z : np.ndarray, shape (n, n)
        Matrix to project.

    Returns
    -------
    P_Z : np.ndarray, shape (n, n)
        Oblique projection of Z onto T_Y M_r.
    """
    P_Z = np.zeros_like(Z, dtype=complex)
    return P_Z

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_oblique_project(U: np.ndarray, V: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    U = np.asarray(U, dtype=complex)
    V = np.asarray(V, dtype=complex)
    Z = np.asarray(Z, dtype=complex)
    n, r = U.shape
    I_U = _oracle_qdeim_select(U)
    I_V = _oracle_qdeim_select(V)
    L_U = np.linalg.inv(U[I_U, :])
    R_V = np.linalg.solve(V[I_V, :].conj().T, V.conj().T)
    Z_rows = Z[I_U, :]
    Z_cols = Z[:, I_V]
    Z_sub = Z[np.ix_(I_U, I_V)]
    return U @ L_U @ (Z_rows - Z_sub @ R_V) + Z_cols @ R_V

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Case 1
        {
            "setup": """import numpy as np
np.random.seed(42)
n, r = 6, 2
G1 = np.random.randn(n, r) + 1j * np.random.randn(n, r)
G2 = np.random.randn(n, r) + 1j * np.random.randn(n, r)
U, _ = np.linalg.qr(G1, mode='reduced')
V, _ = np.linalg.qr(G2, mode='reduced')
Z = np.random.randn(n, n) + 1j * np.random.randn(n, n)
""",
            "call": "np.round(oblique_project(U, V, Z), 8).tolist()",
            "gold_call": "np.round(_oracle_oblique_project(U, V, Z), 8).tolist()",
        },
        # Case 2
        {
            "setup": """import numpy as np
np.random.seed(31)
n, r = 8, 3
G1 = np.random.randn(n, r) + 1j * np.random.randn(n, r)
G2 = np.random.randn(n, r) + 1j * np.random.randn(n, r)
U, _ = np.linalg.qr(G1, mode='reduced')
V, _ = np.linalg.qr(G2, mode='reduced')
M = np.random.randn(r, r) + 1j * np.random.randn(r, r)
T = U @ M @ V.conj().T
""",
            "call": "np.round(oblique_project(U, V, T), 8).tolist()",
            "gold_call": "np.round(_oracle_oblique_project(U, V, T), 8).tolist()",
        },
        # Case 3
        {
            "setup": """import numpy as np
n = 5
U = np.zeros((n, 1), dtype=complex); U[2, 0] = 1.0
V = np.zeros((n, 1), dtype=complex); V[0, 0] = 1.0
Z = np.ones((n, n), dtype=complex)
""",
            "call": "np.round(oblique_project(U, V, Z), 10).tolist()",
            "gold_call": "np.round(_oracle_oblique_project(U, V, Z), 10).tolist()",
        },
        # Case 4
        {
            "setup": """import numpy as np
np.random.seed(88)
n, r = 12, 4
G1 = np.random.randn(n, r) + 1j * np.random.randn(n, r)
G2 = np.random.randn(n, r) + 1j * np.random.randn(n, r)
U, _ = np.linalg.qr(G1, mode='reduced')
V, _ = np.linalg.qr(G2, mode='reduced')
Z = np.random.randn(n, n) + 1j * np.random.randn(n, n)
""",
            "call": "np.round(oblique_project(U, V, Z), 8).tolist()",
            "gold_call": "np.round(_oracle_oblique_project(U, V, Z), 8).tolist()",
        },
        # Case 5
        {
            "setup": """import numpy as np
np.random.seed(200)
n, r = 8, 3
G1 = np.random.randn(n, r) + 1j * np.random.randn(n, r)
G2 = np.random.randn(n, r) + 1j * np.random.randn(n, r)
U, _ = np.linalg.qr(G1, mode='reduced')
V, _ = np.linalg.qr(G2, mode='reduced')
Z = 1j * np.random.randn(n, n)
""",
            "call": "np.round(oblique_project(U, V, Z), 8).tolist()",
            "gold_call": "np.round(_oracle_oblique_project(U, V, Z), 8).tolist()",
        },
        # Case 6
        {
            "setup": """import numpy as np
np.random.seed(505)
n, r = 10, 3
G1 = np.random.randn(n, r) + 1j * np.random.randn(n, r)
G2 = np.random.randn(n, r) + 1j * np.random.randn(n, r)
U, _ = np.linalg.qr(G1, mode='reduced')
V, _ = np.linalg.qr(G2, mode='reduced')
Z = np.random.randn(n, n) + 1j * np.random.randn(n, n)
""",
            "call": "np.round(oblique_project(U, V, oblique_project(U, V, Z)), 7).tolist()",
            "gold_call": "np.round(_oracle_oblique_project(U, V, _oracle_oblique_project(U, V, Z)), 7).tolist()",
        },
        # Case 7
        {
            "setup": """import numpy as np
np.random.seed(777)
n, r = 14, 5
G1 = np.random.randn(n, r) + 1j * np.random.randn(n, r)
G2 = np.random.randn(n, r) + 1j * np.random.randn(n, r)
U, _ = np.linalg.qr(G1, mode='reduced')
V, _ = np.linalg.qr(G2, mode='reduced')
Z = np.random.randn(n, n) + 1j * np.random.randn(n, n)
def proj_norm(P_Z):
    return round(float(np.linalg.norm(P_Z, 'fro')), 8)
""",
            "call": "proj_norm(oblique_project(U, V, Z))",
            "gold_call": "proj_norm(_oracle_oblique_project(U, V, Z))",
        },
        # Case 8
        {
            "setup": """import numpy as np
np.random.seed(111)
n, r = 10, 3
G1 = np.random.randn(n, r) + 1j * np.random.randn(n, r)
G2 = np.random.randn(n, r) + 1j * np.random.randn(n, r)
U, _ = np.linalg.qr(G1, mode='reduced')
V, _ = np.linalg.qr(G2, mode='reduced')
Z = np.random.randn(n, n) + 1j * np.random.randn(n, n)
def full_matrix_check(P_Z):
    return tuple(np.round(P_Z.ravel()[:10], 8).tolist())
""",
            "call": "full_matrix_check(oblique_project(U, V, Z))",
            "gold_call": "full_matrix_check(_oracle_oblique_project(U, V, Z))",
        },
        # Case 9
        {
            "setup": """import numpy as np
np.random.seed(222)
n, r = 8, 2
G1 = np.random.randn(n, r) + 1j * np.random.randn(n, r)
G2 = np.random.randn(n, r) + 1j * np.random.randn(n, r)
U, _ = np.linalg.qr(G1, mode='reduced')
V, _ = np.linalg.qr(G2, mode='reduced')
Z = np.random.randn(n, n) + 1j * np.random.randn(n, n)
def element_check(P_Z):
    return tuple(np.round(P_Z.ravel()[-10:], 8).tolist())
""",
            "call": "element_check(oblique_project(U, V, Z))",
            "gold_call": "element_check(_oracle_oblique_project(U, V, Z))",
        },
    ]
