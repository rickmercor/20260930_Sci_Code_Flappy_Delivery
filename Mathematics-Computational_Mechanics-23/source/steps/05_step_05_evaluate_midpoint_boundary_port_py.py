"""
Evaluate the midpoint mechanical boundary port using both the paper-specific director kinematic matrix and the generalized boundary-effort vector returned by the preceding subproblem. Recover the spatial angular velocity from the stacked midpoint director-rate vector using the supplied 9-by-3 director map. Evaluate translational power by pairing the centerline boundary velocity with the first three components of the generalized effort, and evaluate rotational power by pairing the stacked director-rate vector with its remaining nine components. Return a six-component NumPy array containing, in order, the three angular-velocity components, translational power, rotational power, and total boundary power. Do not reconstruct the generalized boundary effort inside this function.

The mixed port-Hamiltonian boundary variables can be expressed either through the physical translational and rotational pairing or through the equivalent generalized pairing in centerline and director-rate coordinates. Once the boundary moment has been transferred to director-rate coordinates by the preceding subproblem, the rotational contribution to boundary power is obtained directly from the stacked director velocity and the corresponding generalized effort. The same director kinematic map independently reconstructs the spatial angular velocity, allowing the generalized power pairing and the physical rotational quantity to remain consistent with the paper-specific director representation.

Returns
-------
np.ndarray of shape (6,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def evaluate_midpoint_boundary_port(
    T: np.ndarray,
    boundary_effort: np.ndarray,
    v_d_boundary: np.ndarray,
    v_phi_boundary: np.ndarray,
) -> np.ndarray:
    """Return the midpoint angular velocity and mechanical boundary-power components."""
    return np.zeros(6, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_evaluate_midpoint_boundary_port(
    T: np.ndarray,
    boundary_effort: np.ndarray,
    v_d_boundary: np.ndarray,
    v_phi_boundary: np.ndarray,
) -> np.ndarray:
    import numpy as np

    T = np.asarray(T, dtype=float)
    boundary_effort = np.asarray(boundary_effort, dtype=float)
    v_d_boundary = np.asarray(v_d_boundary, dtype=float)
    v_phi_boundary = np.asarray(v_phi_boundary, dtype=float)

    if T.shape != (9, 3):
        raise ValueError("T must have shape (9, 3).")

    if boundary_effort.shape != (12,):
        raise ValueError("boundary_effort must have shape (12,).")

    if v_d_boundary.shape != (9,):
        raise ValueError("v_d_boundary must have shape (9,).")

    if v_phi_boundary.shape != (3,):
        raise ValueError("v_phi_boundary must have shape (3,).")

    if not (
        np.all(np.isfinite(T))
        and np.all(np.isfinite(boundary_effort))
        and np.all(np.isfinite(v_d_boundary))
        and np.all(np.isfinite(v_phi_boundary))
    ):
        raise ValueError("Boundary-port inputs must be finite.")

    omega = T.T @ v_d_boundary

    translational_effort = boundary_effort[:3]
    director_effort = boundary_effort[3:]

    p_trans = float(v_phi_boundary @ translational_effort)
    p_rot = float(v_d_boundary @ director_effort)
    p_total = p_trans + p_rot

    return np.array(
        [
            omega[0],
            omega[1],
            omega[2],
            p_trans,
            p_rot,
            p_total,
        ],
        dtype=float,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np

sqrt3 = np.sqrt(3.0)

T = np.array([
    [0.0, 0.0, -0.25],
    [0.0, 0.0, sqrt3 / 4.0],
    [0.25, -sqrt3 / 4.0, 0.0],

    [0.0, 0.0, -sqrt3 / 4.0],
    [0.0, 0.0, -0.25],
    [sqrt3 / 4.0, 0.25, 0.0],

    [0.0, 0.5, 0.0],
    [-0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0],
], dtype=float)

boundary_effort = np.array([
    0.7,
    -0.25,
    0.15,
    0.0225,
    -0.03897114317029974,
    -0.04794228634059948,
    0.03897114317029974,
    0.0225,
    0.09696152422706632,
    0.09,
    -0.06,
    0.0,
], dtype=float)

v_d_boundary = np.array([
    -0.025,
    0.025 * sqrt3,
    0.06 + 0.04 * sqrt3,
    -0.025 * sqrt3,
    -0.025,
    0.06 * sqrt3 - 0.04,
    -0.08,
    -0.12,
    0.0,
], dtype=float)

v_phi_boundary = np.array(
    [0.205, -0.095, 0.082],
    dtype=float,
)
""",
            "call": """
evaluate_midpoint_boundary_port(
    T,
    boundary_effort,
    v_d_boundary,
    v_phi_boundary,
)
""",
            "gold_call": """
_oracle_evaluate_midpoint_boundary_port(
    T,
    boundary_effort,
    v_d_boundary,
    v_phi_boundary,
)
""",
        },

        {
            "setup": """
import numpy as np

T = np.array([
    [0.0, 0.0, -0.5],
    [0.0, 0.0, 0.5],
    [0.5, -0.5, 0.0],

    [0.0, 0.0, -0.25],
    [0.0, 0.0, 0.0],
    [0.25, 0.0, 0.0],

    [0.0, 0.5, 0.0],
    [-0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0],
], dtype=float)

boundary_effort = np.array([
    0.4,
    -0.3,
    0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
], dtype=float)

v_d_boundary = np.zeros(9, dtype=float)

v_phi_boundary = np.array(
    [0.2, -0.1, 0.3],
    dtype=float,
)
""",
            "call": """
evaluate_midpoint_boundary_port(
    T,
    boundary_effort,
    v_d_boundary,
    v_phi_boundary,
)
""",
            "gold_call": """
_oracle_evaluate_midpoint_boundary_port(
    T,
    boundary_effort,
    v_d_boundary,
    v_phi_boundary,
)
""",
        },

        {
            "setup": """
import numpy as np

T = np.array([
    [0.0, 0.2, -0.1],
    [-0.2, 0.0, 0.3],
    [0.1, -0.3, 0.0],

    [0.0, -0.4, 0.2],
    [0.4, 0.0, -0.1],
    [-0.2, 0.1, 0.0],

    [0.0, 0.5, 0.25],
    [-0.5, 0.0, -0.15],
    [-0.25, 0.15, 0.0],
], dtype=float)

boundary_effort = np.array([
    0.0,
    0.0,
    0.0,
    0.10,
    -0.04,
    0.07,
    -0.02,
    0.09,
    -0.05,
    0.03,
    -0.08,
    0.06,
], dtype=float)

v_d_boundary = np.array([
    0.1,
    -0.2,
    0.05,
    -0.1,
    0.04,
    0.08,
    0.03,
    -0.06,
    0.02,
], dtype=float)

v_phi_boundary = np.zeros(3, dtype=float)
""",
            "call": """
evaluate_midpoint_boundary_port(
    T,
    boundary_effort,
    v_d_boundary,
    v_phi_boundary,
)
""",
            "gold_call": """
_oracle_evaluate_midpoint_boundary_port(
    T,
    boundary_effort,
    v_d_boundary,
    v_phi_boundary,
)
""",
        },
    ]
