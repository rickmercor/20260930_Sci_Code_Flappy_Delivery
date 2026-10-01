"""
Assemble the quadratic operator associated with the reduced discrete Hamiltonian of the mixed Cosserat-rod formulation. The reduced state is ordered as the translational velocity, stacked director velocity, force-stress vector, and moment-stress vector. Place the supplied translational mass, director mass, force-compliance, and moment-compliance blocks into the appropriate state partitions without inverting, rescaling, or mixing the blocks. Return the resulting 18-by-18 symmetric NumPy array. All supplied blocks must have the prescribed dimensions, contain finite values, and be symmetric.

The structure-preserving finite-element formulation retains a quadratic discrete Hamiltonian in the mixed velocity-stress variables. For the reduced state used in this task, translational velocity, director velocity, force stress, and moment stress occupy distinct coordinate partitions and are associated with their corresponding reduced mass or compliance operators. Constructing the global quadratic operator separately makes the state ordering explicit and allows the same operator to be reused for both endpoint Hamiltonian evaluations.

Returns
-------
return K
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def assemble_discrete_hamiltonian_operator(
    M_phi: np.ndarray,
    M_d: np.ndarray,
    C_N: np.ndarray,
    C_M: np.ndarray,
) -> np.ndarray:
    """Return the 18-by-18 quadratic Hamiltonian operator."""
    return np.zeros((18, 18), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_assemble_discrete_hamiltonian_operator(
    M_phi: np.ndarray,
    M_d: np.ndarray,
    C_N: np.ndarray,
    C_M: np.ndarray,
) -> np.ndarray:
    import numpy as np

    M_phi = np.asarray(M_phi, dtype=float)
    M_d = np.asarray(M_d, dtype=float)
    C_N = np.asarray(C_N, dtype=float)
    C_M = np.asarray(C_M, dtype=float)

    if M_phi.shape != (3, 3):
        raise ValueError("M_phi must have shape (3, 3).")
    if M_d.shape != (9, 9):
        raise ValueError("M_d must have shape (9, 9).")
    if C_N.shape != (3, 3):
        raise ValueError("C_N must have shape (3, 3).")
    if C_M.shape != (3, 3):
        raise ValueError("C_M must have shape (3, 3).")

    blocks = [M_phi, M_d, C_N, C_M]

    if not all(np.all(np.isfinite(A)) for A in blocks):
        raise ValueError("All operator blocks must be finite.")

    if not all(
        np.allclose(A, A.T, rtol=1e-12, atol=1e-12)
        for A in blocks
    ):
        raise ValueError("All operator blocks must be symmetric.")

    K = np.zeros((18, 18), dtype=float)

    K[0:3, 0:3] = M_phi
    K[3:12, 3:12] = M_d
    K[12:15, 12:15] = C_N
    K[15:18, 15:18] = C_M

    return K

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np

M_phi = np.diag([1.3, 1.1, 0.9])

M_d = np.diag([
    0.42, 0.42, 0.42,
    0.37, 0.37, 0.37,
    0.0, 0.0, 0.0,
])

C_N = np.diag([0.55, 0.48, 0.62])
C_M = np.diag([0.31, 0.27, 0.35])
""",
            "call": """
assemble_discrete_hamiltonian_operator(
    M_phi,
    M_d,
    C_N,
    C_M,
)
""",
            "gold_call": """
_oracle_assemble_discrete_hamiltonian_operator(
    M_phi,
    M_d,
    C_N,
    C_M,
)
""",
        },
        {
            "setup": """
import numpy as np

M_phi = 2.0 * np.eye(3)
M_d = 0.5 * np.eye(9)
C_N = 0.25 * np.eye(3)
C_M = 0.75 * np.eye(3)
""",
            "call": """
assemble_discrete_hamiltonian_operator(
    M_phi,
    M_d,
    C_N,
    C_M,
)
""",
            "gold_call": """
_oracle_assemble_discrete_hamiltonian_operator(
    M_phi,
    M_d,
    C_N,
    C_M,
)
""",
        },
        {
            "setup": """
import numpy as np

M_phi = np.array([
    [2.0, 0.2, -0.1],
    [0.2, 1.5, 0.3],
    [-0.1, 0.3, 1.2],
], dtype=float)

M_d = np.diag([
    0.8, 0.7, 0.6,
    0.5, 0.4, 0.3,
    0.2, 0.1, 0.05,
])

C_N = np.array([
    [0.6, 0.08, -0.02],
    [0.08, 0.5, 0.04],
    [-0.02, 0.04, 0.7],
], dtype=float)

C_M = np.array([
    [0.4, -0.05, 0.03],
    [-0.05, 0.35, 0.02],
    [0.03, 0.02, 0.45],
], dtype=float)
""",
            "call": """
assemble_discrete_hamiltonian_operator(
    M_phi,
    M_d,
    C_N,
    C_M,
)
""",
            "gold_call": """
_oracle_assemble_discrete_hamiltonian_operator(
    M_phi,
    M_d,
    C_N,
    C_M,
)
""",
        },
    ]
