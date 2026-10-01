"""
Assemble the 12-component generalized mechanical boundary effort by concatenating the boundary force $n_boundary$ with the director-rate moment $T @ m_boundary$. Return that vector.

In the port-Hamiltonian Cosserat-rod formulation, the mechanical boundary port pairs translational velocity with force and the director-based rotational velocity with moment. Because the rotational velocity is represented through the director kinematic map, the moment must be transferred consistently to the stacked director-rate coordinates before a single generalized velocity-effort pairing can be formed. This produces a boundary effort living in the same reduced coordinate space as the centerline and director boundary rates and provides an additional formulation-specific link between the director kinematics and the mechanical power port.

Returns
-------
return boundary_effort
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def assemble_mechanical_boundary_effort(
    T: np.ndarray,
    n_boundary: np.ndarray,
    m_boundary: np.ndarray,
) -> np.ndarray:
    """Return the 12-component generalized mechanical boundary effort."""
    return np.zeros(12, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_assemble_mechanical_boundary_effort(
    T: np.ndarray,
    n_boundary: np.ndarray,
    m_boundary: np.ndarray,
) -> np.ndarray:
    import numpy as np

    T = np.asarray(T, dtype=float)
    n_boundary = np.asarray(n_boundary, dtype=float)
    m_boundary = np.asarray(m_boundary, dtype=float)

    if T.shape != (9, 3):
        raise ValueError("T must have shape (9, 3).")

    if n_boundary.shape != (3,):
        raise ValueError("n_boundary must have shape (3,).")

    if m_boundary.shape != (3,):
        raise ValueError("m_boundary must have shape (3,).")

    if not (
        np.all(np.isfinite(T))
        and np.all(np.isfinite(n_boundary))
        and np.all(np.isfinite(m_boundary))
    ):
        raise ValueError("Boundary-effort inputs must be finite.")

    director_effort = T @ m_boundary

    boundary_effort = np.concatenate([
        n_boundary,
        director_effort,
    ])

    return boundary_effort

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

n_boundary = np.array(
    [0.7, -0.25, 0.15],
    dtype=float,
)

m_boundary = np.array(
    [0.12, 0.18, -0.09],
    dtype=float,
)
""",
            "call": """
assemble_mechanical_boundary_effort(
    T,
    n_boundary,
    m_boundary,
)
""",
            "gold_call": """
_oracle_assemble_mechanical_boundary_effort(
    T,
    n_boundary,
    m_boundary,
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

n_boundary = np.array(
    [0.4, -0.3, 0.2],
    dtype=float,
)

m_boundary = np.zeros(
    3,
    dtype=float,
)
""",
            "call": """
assemble_mechanical_boundary_effort(
    T,
    n_boundary,
    m_boundary,
)
""",
            "gold_call": """
_oracle_assemble_mechanical_boundary_effort(
    T,
    n_boundary,
    m_boundary,
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

n_boundary = np.zeros(
    3,
    dtype=float,
)

m_boundary = np.array(
    [-0.6, 0.4, 0.8],
    dtype=float,
)
""",
            "call": """
assemble_mechanical_boundary_effort(
    T,
    n_boundary,
    m_boundary,
)
""",
            "gold_call": """
_oracle_assemble_mechanical_boundary_effort(
    T,
    n_boundary,
    m_boundary,
)
""",
        },
    ]
