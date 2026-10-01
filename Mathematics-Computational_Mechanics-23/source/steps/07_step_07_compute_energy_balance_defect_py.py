"""
Compute the signed energy-balance defect for the complete reduced Cosserat-rod state transition by orchestrating all six preceding subproblems. Assemble the discrete Hamiltonian operator, evaluate the endpoint Hamiltonians and their signed increment, construct the paper-specific director kinematic matrix, assemble the generalized mechanical boundary effort, evaluate the midpoint mechanical boundary port, and finally form the one-step discrete power-balance result. Return only the signed energy-balance defect as a finite floating-point scalar. The public implementation must call and genuinely use the outputs of assemble_discrete_hamiltonian_operator, evaluate_endpoint_energy_increment, build_director_kinematic_map, assemble_mechanical_boundary_effort, evaluate_midpoint_boundary_port, and compute_discrete_power_balance_defect rather than duplicating their calculations.

The complete verification quantity combines the mixed discrete Hamiltonian, the paper-specific director kinematics, the generalized mechanical boundary effort, and the implicit-midpoint power balance. Keeping these operations in separate subproblems preserves the structure of the formulation and makes each convention independently testable. The final orchestrator therefore performs no alternative reconstruction of these quantities; it composes the outputs of the preceding steps and returns the resulting signed one-step energy-balance defect.

Returns
-------
A finite float containing the signed one-step energy-balance defect.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_energy_balance_defect(
    M_phi: np.ndarray,
    M_d: np.ndarray,
    C_N: np.ndarray,
    C_M: np.ndarray,
    v_phi_n: np.ndarray,
    v_d_n: np.ndarray,
    N_n: np.ndarray,
    M_n: np.ndarray,
    v_phi_np1: np.ndarray,
    v_d_np1: np.ndarray,
    N_np1: np.ndarray,
    M_np1: np.ndarray,
    directors: np.ndarray,
    v_d_boundary: np.ndarray,
    v_phi_boundary: np.ndarray,
    n_boundary: np.ndarray,
    m_boundary: np.ndarray,
    h: float,
) -> float:
    """Return the signed one-step energy-balance defect for the complete pipeline."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_energy_balance_defect(
    M_phi: np.ndarray,
    M_d: np.ndarray,
    C_N: np.ndarray,
    C_M: np.ndarray,
    v_phi_n: np.ndarray,
    v_d_n: np.ndarray,
    N_n: np.ndarray,
    M_n: np.ndarray,
    v_phi_np1: np.ndarray,
    v_d_np1: np.ndarray,
    N_np1: np.ndarray,
    M_np1: np.ndarray,
    directors: np.ndarray,
    v_d_boundary: np.ndarray,
    v_phi_boundary: np.ndarray,
    n_boundary: np.ndarray,
    m_boundary: np.ndarray,
    h: float,
) -> float:
    import numpy as np

    K = _oracle_assemble_discrete_hamiltonian_operator(
        M_phi,
        M_d,
        C_N,
        C_M,
    )

    energy_summary = _oracle_evaluate_endpoint_energy_increment(
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

    T = _oracle_build_director_kinematic_map(
        directors,
    )

    boundary_effort = _oracle_assemble_mechanical_boundary_effort(
        T,
        n_boundary,
        m_boundary,
    )

    port_summary = _oracle_evaluate_midpoint_boundary_port(
        T,
        boundary_effort,
        v_d_boundary,
        v_phi_boundary,
    )

    balance_summary = _oracle_compute_discrete_power_balance_defect(
        energy_summary,
        port_summary,
        h,
    )

    defect = float(balance_summary[1])

    if not np.isfinite(defect):
        raise ValueError("The energy-balance defect must be finite.")

    return defect

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return test cases for the complete energy-balance pipeline."""
    return [
        {
            "setup": """import numpy as np

d = 0.03
E = 6.0e5
G = 2.0e5
rho = 1080.0
r_k = 0.065
L = 0.1755
h_paper = 0.05
t_end = 4.0
f_max = -50.0
t_1 = 0.5
t_2 = 3.5

a = d / L
b = r_k / L
c = G / E
r = rho / 1000.0
q = (t_2 - t_1) / t_end
u = abs(f_max) / 100.0

M_phi = np.array([
    [
        1.0 + 0.4 * r + 0.2 * q,
        0.20 * a + 0.03 * u,
        0.05 * b,
    ],
    [
        0.20 * a + 0.03 * u,
        0.8 + 0.35 * r + 0.15 * u,
        -0.12 * c - 0.02 * q,
    ],
    [
        0.05 * b,
        -0.12 * c - 0.02 * q,
        0.7 + 0.30 * r + 0.10 * q,
    ],
])

M_d = np.diag([
    0.25 + 0.18 * b + 0.05 * u,
    0.25 + 0.18 * b + 0.05 * u,
    0.25 + 0.18 * b + 0.05 * u,
    0.24 + 0.16 * a + 0.04 * q,
    0.24 + 0.16 * a + 0.04 * q,
    0.24 + 0.16 * a + 0.04 * q,
    0.0,
    0.0,
    0.0,
])

C_N = np.array([
    [
        0.36 + 0.25 * c + 0.08 * u,
        0.06 * b + 0.01 * q,
        0.04 * a,
    ],
    [
        0.06 * b + 0.01 * q,
        0.40 + 0.10 * r + 0.05 * q,
        -0.03 * c,
    ],
    [
        0.04 * a,
        -0.03 * c,
        0.46 + 0.16 * a + 0.04 * u,
    ],
])

C_M = np.array([
    [
        0.20 + 0.20 * c + 0.05 * q,
        -0.04 * a,
        0.03 * b,
    ],
    [
        -0.04 * a,
        0.22 + 0.16 * b + 0.04 * u,
        0.025 * c,
    ],
    [
        0.03 * b,
        0.025 * c,
        0.24 + 0.08 * r + 0.03 * q,
    ],
])

n_boundary = np.array([
    0.25 + 0.40 * b + 0.15 * a + 0.05 * u,
    -0.08 - 0.22 * c - 0.08 * b - 0.03 * q,
    0.06 + 0.18 * a + 0.04 * r + 0.02 * u,
])

m_boundary = np.array([
    -0.03 + 0.12 * b - 0.04 * c + 0.01 * u,
    0.08 + 0.15 * a + 0.02 * r + 0.03 * q,
    0.04 + 0.10 * c + 0.03 * b + 0.02 * u,
])

h = h_paper * (
    0.55
    + 0.20 * b
    + 0.10 * c
    + 0.05 * q
)

v_phi_n = np.array([
    0.11,
    -0.07,
    0.05,
])

v_d_n = np.array([
    0.01, -0.02, 0.015,
    -0.025, 0.012, 0.018,
    0.02, -0.01, 0.005,
])

N_n = np.array([
    0.31,
    -0.12,
    0.09,
])

M_n = np.array([
    -0.06,
    0.10,
    0.035,
])

v_phi_np1 = np.array([
    0.16,
    -0.045,
    0.072,
])

v_d_np1 = np.array([
    0.016, -0.017, 0.021,
    -0.020, 0.018, 0.024,
    0.025, -0.006, 0.009,
])

N_np1 = np.array([
    0.36,
    -0.095,
    0.115,
])

M_np1 = np.array([
    -0.048,
    0.125,
    0.047,
])

directors = np.array([
    [0.866025403784, -0.5, 0.0],
    [0.5, 0.866025403784, 0.0],
    [0.0, 0.0, 1.0],
])

v_phi_boundary = np.array([
    0.17,
    -0.06,
    0.09,
])

v_d_boundary = np.array([
    0.055,
    0.095262794416,
    0.007942286341,
    -0.095262794416,
    0.055,
    0.166243556530,
    -0.09,
    -0.14,
    0.0,
])
""",

            "call": """compute_energy_balance_defect(
    M_phi,
    M_d,
    C_N,
    C_M,
    v_phi_n,
    v_d_n,
    N_n,
    M_n,
    v_phi_np1,
    v_d_np1,
    N_np1,
    M_np1,
    directors,
    v_d_boundary,
    v_phi_boundary,
    n_boundary,
    m_boundary,
    h,
)""",

            "gold_call": """_oracle_compute_energy_balance_defect(
    M_phi,
    M_d,
    C_N,
    C_M,
    v_phi_n,
    v_d_n,
    N_n,
    M_n,
    v_phi_np1,
    v_d_np1,
    N_np1,
    M_np1,
    directors,
    v_d_boundary,
    v_phi_boundary,
    n_boundary,
    m_boundary,
    h,
)""",
        },

        {
            "setup": """import numpy as np

M_phi = np.array([
    [1.30, 0.05, 0.00],
    [0.05, 1.10, -0.02],
    [0.00, -0.02, 0.90],
])

M_d = np.diag([
    0.40, 0.40, 0.40,
    0.35, 0.35, 0.35,
    0.00, 0.00, 0.00,
])

C_N = np.array([
    [0.52, 0.03, -0.01],
    [0.03, 0.47, 0.02],
    [-0.01, 0.02, 0.61],
])

C_M = np.array([
    [0.31, 0.01, -0.02],
    [0.01, 0.27, 0.012],
    [-0.02, 0.012, 0.36],
])

v_phi_n = np.array([0.07, -0.04, 0.03])

v_d_n = np.array([
    0.012, -0.018, 0.009,
    -0.021, 0.014, 0.016,
    0.010, -0.007, 0.004,
])

N_n = np.array([0.25, -0.11, 0.08])
M_n = np.array([-0.04, 0.085, 0.028])

v_phi_np1 = np.array([0.13, -0.02, 0.055])

v_d_np1 = np.array([
    0.018, -0.012, 0.015,
    -0.015, 0.020, 0.021,
    0.018, -0.003, 0.008,
])

N_np1 = np.array([0.33, -0.07, 0.105])
M_np1 = np.array([-0.025, 0.11, 0.042])

theta = np.pi / 4.0

directors = np.array([
    [np.cos(theta), -np.sin(theta), 0.0],
    [np.sin(theta),  np.cos(theta), 0.0],
    [0.0,            0.0,           1.0],
])

omega_b = np.array([0.09, -0.13, 0.07])

v_d_boundary = np.concatenate([
    np.cross(omega_b, directors[0]),
    np.cross(omega_b, directors[1]),
    np.cross(omega_b, directors[2]),
])

v_phi_boundary = np.array([0.12, -0.08, 0.04])

n_boundary = np.array([0.46, -0.24, 0.18])
m_boundary = np.array([-0.06, 0.15, 0.09])

h = 0.06
""",

            "call": """compute_energy_balance_defect(
    M_phi,
    M_d,
    C_N,
    C_M,
    v_phi_n,
    v_d_n,
    N_n,
    M_n,
    v_phi_np1,
    v_d_np1,
    N_np1,
    M_np1,
    directors,
    v_d_boundary,
    v_phi_boundary,
    n_boundary,
    m_boundary,
    h,
)""",

            "gold_call": """_oracle_compute_energy_balance_defect(
    M_phi,
    M_d,
    C_N,
    C_M,
    v_phi_n,
    v_d_n,
    N_n,
    M_n,
    v_phi_np1,
    v_d_np1,
    N_np1,
    M_np1,
    directors,
    v_d_boundary,
    v_phi_boundary,
    n_boundary,
    m_boundary,
    h,
)""",
        },

        {
            "setup": """import numpy as np

M_phi = np.eye(3)

M_d = np.diag([
    0.30, 0.30, 0.30,
    0.30, 0.30, 0.30,
    0.00, 0.00, 0.00,
])

C_N = 0.50 * np.eye(3)
C_M = 0.25 * np.eye(3)

v_phi_n = np.array([0.10, -0.05, 0.02])

v_d_n = np.array([
    0.01, -0.01, 0.02,
    -0.015, 0.008, 0.012,
    0.004, -0.003, 0.002,
])

N_n = np.array([0.20, -0.10, 0.05])
M_n = np.array([0.03, 0.04, -0.02])

v_phi_np1 = v_phi_n.copy()
v_d_np1 = v_d_n.copy()
N_np1 = N_n.copy()
M_np1 = M_n.copy()

directors = np.eye(3)

omega_b = np.array([0.20, 0.00, 0.00])

v_d_boundary = np.concatenate([
    np.cross(omega_b, directors[0]),
    np.cross(omega_b, directors[1]),
    np.cross(omega_b, directors[2]),
])

v_phi_boundary = np.zeros(3)

n_boundary = np.array([0.40, -0.20, 0.10])
m_boundary = np.array([1.31, 0.00, 0.00])

h = 0.50
""",

            "call": """compute_energy_balance_defect(
    M_phi,
    M_d,
    C_N,
    C_M,
    v_phi_n,
    v_d_n,
    N_n,
    M_n,
    v_phi_np1,
    v_d_np1,
    N_np1,
    M_np1,
    directors,
    v_d_boundary,
    v_phi_boundary,
    n_boundary,
    m_boundary,
    h,
)""",

            "gold_call": """_oracle_compute_energy_balance_defect(
    M_phi,
    M_d,
    C_N,
    C_M,
    v_phi_n,
    v_d_n,
    N_n,
    M_n,
    v_phi_np1,
    v_d_np1,
    N_np1,
    M_np1,
    directors,
    v_d_boundary,
    v_phi_boundary,
    n_boundary,
    m_boundary,
    h,
)""",
        },

        {
            "setup": """import numpy as np

M_phi = np.eye(3)
M_d = np.eye(9)
C_N = np.eye(3)
C_M = np.eye(3)

v_phi_n = np.zeros(3)
v_d_n = np.zeros(9)
N_n = np.zeros(3)
M_n = np.zeros(3)

v_phi_np1 = np.array([2.0, 0.0, 0.0])
v_d_np1 = np.zeros(9)
N_np1 = np.zeros(3)
M_np1 = np.zeros(3)

directors = np.eye(3)

omega_b = np.array([0.10, 0.20, -0.10])

v_d_boundary = np.concatenate([
    np.cross(omega_b, directors[0]),
    np.cross(omega_b, directors[1]),
    np.cross(omega_b, directors[2]),
])

v_phi_boundary = np.zeros(3)

n_boundary = np.zeros(3)
m_boundary = np.zeros(3)

h = 0.25
""",

            "call": """compute_energy_balance_defect(
    M_phi,
    M_d,
    C_N,
    C_M,
    v_phi_n,
    v_d_n,
    N_n,
    M_n,
    v_phi_np1,
    v_d_np1,
    N_np1,
    M_np1,
    directors,
    v_d_boundary,
    v_phi_boundary,
    n_boundary,
    m_boundary,
    h,
)""",

            "gold_call": """_oracle_compute_energy_balance_defect(
    M_phi,
    M_d,
    C_N,
    C_M,
    v_phi_n,
    v_d_n,
    N_n,
    M_n,
    v_phi_np1,
    v_d_np1,
    N_np1,
    M_np1,
    directors,
    v_d_boundary,
    v_phi_boundary,
    n_boundary,
    m_boundary,
    h,
)""",
        },
    ]
