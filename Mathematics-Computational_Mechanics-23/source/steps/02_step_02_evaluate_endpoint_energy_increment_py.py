"""
Evaluate the discrete Hamiltonian at two consecutive endpoint states using the preassembled 18-by-18 quadratic operator returned by the preceding subproblem. Pack each endpoint state in the prescribed order consisting of translational velocity, stacked director velocity, force stress, and moment stress. Evaluate the quadratic Hamiltonian at each endpoint and retain the signed ordering of the time levels. Return a three-component NumPy array containing H^n, H^(n+1), and H^(n+1)-H^n in that exact order. Do not reconstruct or modify the Hamiltonian operator inside this function.

The midpoint power balance compares a Hamiltonian increment between two endpoint states with work evaluated at the temporal midpoint. The Hamiltonian operator itself is constant for the reduced data supplied here, so it can be assembled once and applied independently to the two state vectors. Keeping the state packing and endpoint evaluation in a dedicated stage prevents midpoint boundary data from entering the endpoint energy calculation.

Returns
-------
return np.array([H_n, H_np1, delta_H], dtype=float)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def evaluate_endpoint_energy_increment(
    v_phi_n: np.ndarray,
    v_d_n: np.ndarray,
    N_n: np.ndarray,
    M_n: np.ndarray,
    v_phi_np1: np.ndarray,
    v_d_np1: np.ndarray,
    N_np1: np.ndarray,
    M_np1: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Return [H_n, H_np1, H_np1_minus_H_n]."""
    return np.zeros(3, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_evaluate_endpoint_energy_increment(
    v_phi_n: np.ndarray,
    v_d_n: np.ndarray,
    N_n: np.ndarray,
    M_n: np.ndarray,
    v_phi_np1: np.ndarray,
    v_d_np1: np.ndarray,
    N_np1: np.ndarray,
    M_np1: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    import numpy as np

    v_phi_n = np.asarray(v_phi_n, dtype=float)
    v_d_n = np.asarray(v_d_n, dtype=float)
    N_n = np.asarray(N_n, dtype=float)
    M_n = np.asarray(M_n, dtype=float)

    v_phi_np1 = np.asarray(v_phi_np1, dtype=float)
    v_d_np1 = np.asarray(v_d_np1, dtype=float)
    N_np1 = np.asarray(N_np1, dtype=float)
    M_np1 = np.asarray(M_np1, dtype=float)

    K = np.asarray(K, dtype=float)

    vectors = [
        (v_phi_n, (3,)),
        (v_d_n, (9,)),
        (N_n, (3,)),
        (M_n, (3,)),
        (v_phi_np1, (3,)),
        (v_d_np1, (9,)),
        (N_np1, (3,)),
        (M_np1, (3,)),
    ]

    if any(v.shape != shape for v, shape in vectors):
        raise ValueError("Endpoint state vector has invalid shape.")

    if K.shape != (18, 18):
        raise ValueError("K must have shape (18, 18).")

    if not all(np.all(np.isfinite(v)) for v, _ in vectors):
        raise ValueError("Endpoint states must be finite.")

    if not np.all(np.isfinite(K)):
        raise ValueError("K must be finite.")

    z_n = np.concatenate([
        v_phi_n,
        v_d_n,
        N_n,
        M_n,
    ])

    z_np1 = np.concatenate([
        v_phi_np1,
        v_d_np1,
        N_np1,
        M_np1,
    ])

    H_n = 0.5 * float(z_n @ K @ z_n)
    H_np1 = 0.5 * float(z_np1 @ K @ z_np1)
    delta_H = H_np1 - H_n

    result = np.array(
        [H_n, H_np1, delta_H],
        dtype=float,
    )

    if not np.all(np.isfinite(result)):
        raise ValueError("Endpoint energies must be finite.")

    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np

v_phi_n = np.array([0.18, -0.11, 0.07])
v_d_n = np.array([
    0.02, -0.03, 0.01,
    -0.04, 0.015, 0.025,
    0.03, -0.02, 0.01,
])
N_n = np.array([0.42, -0.18, 0.11])
M_n = np.array([-0.09, 0.14, 0.05])

v_phi_np1 = np.array([0.23, -0.08, 0.095])
v_d_np1 = np.array([
    0.028, -0.025, 0.014,
    -0.035, 0.021, 0.03,
    0.034, -0.016, 0.013,
])
N_np1 = np.array([0.46, -0.15, 0.13])
M_np1 = np.array([-0.075, 0.16, 0.062])

weights = np.array([
    1.3, 1.1, 0.9,
    0.42, 0.42, 0.42,
    0.37, 0.37, 0.37,
    0.0, 0.0, 0.0,
    0.55, 0.48, 0.62,
    0.31, 0.27, 0.35,
])

K = np.diag(weights)
""",
            "call": """
evaluate_endpoint_energy_increment(
    v_phi_n,
    v_d_n,
    N_n,
    M_n,
    v_phi_np1,
    v_d_np1,
    N_np1,
    M_np1,
    K,
)
""",
            "gold_call": """
_oracle_evaluate_endpoint_energy_increment(
    v_phi_n,
    v_d_n,
    N_n,
    M_n,
    v_phi_np1,
    v_d_np1,
    N_np1,
    M_np1,
    K,
)
""",
        },
        {
            "setup": """
import numpy as np

v_phi_n = np.array([0.3, -0.2, 0.1])
v_d_n = np.arange(1.0, 10.0) / 10.0
N_n = np.array([0.2, -0.3, 0.4])
M_n = np.array([-0.1, 0.15, 0.25])

v_phi_np1 = v_phi_n.copy()
v_d_np1 = v_d_n.copy()
N_np1 = N_n.copy()
M_np1 = M_n.copy()

K = np.eye(18)
""",
            "call": """
evaluate_endpoint_energy_increment(
    v_phi_n,
    v_d_n,
    N_n,
    M_n,
    v_phi_np1,
    v_d_np1,
    N_np1,
    M_np1,
    K,
)
""",
            "gold_call": """
_oracle_evaluate_endpoint_energy_increment(
    v_phi_n,
    v_d_n,
    N_n,
    M_n,
    v_phi_np1,
    v_d_np1,
    N_np1,
    M_np1,
    K,
)
""",
        },
        {
            "setup": """
import numpy as np

v_phi_n = np.array([1.0, -0.5, 0.2])
v_d_n = np.array([
    0.2, -0.1, 0.3,
    0.4, -0.2, 0.1,
    0.5, -0.4, 0.6,
])
N_n = np.array([-0.3, 0.7, 0.2])
M_n = np.array([0.4, -0.2, 0.5])

v_phi_np1 = np.array([0.7, -0.2, 0.4])
v_d_np1 = np.array([
    -0.1, 0.3, 0.2,
    0.5, -0.4, 0.2,
    -0.3, 0.6, 0.1,
])
N_np1 = np.array([0.1, 0.4, -0.2])
M_np1 = np.array([-0.3, 0.5, 0.2])

A = np.arange(1.0, 325.0).reshape(18, 18) / 200.0
K = A.T @ A + 0.5 * np.eye(18)
""",
            "call": """
evaluate_endpoint_energy_increment(
    v_phi_n,
    v_d_n,
    N_n,
    M_n,
    v_phi_np1,
    v_d_np1,
    N_np1,
    M_np1,
    K,
)
""",
            "gold_call": """
_oracle_evaluate_endpoint_energy_increment(
    v_phi_n,
    v_d_n,
    N_n,
    M_n,
    v_phi_np1,
    v_d_np1,
    N_np1,
    M_np1,
    K,
)
""",
        },
    ]
