"""
Construct the paper-specific director-dependent kinematic matrix T(d) from an ordered orthonormal director triad. The three supplied director vectors are stored as the rows of a 3-by-3 array in the order d_1, d_2, d_3. Return the 9-by-3 matrix used by the paper to map the stacked director-rate vector to spatial angular velocity. Preserve the paper's exact sign, scaling, block orientation, and director ordering. Do not replace this construction by a generic rotation-vector, quaternion, or finite-difference approximation.

The director formulation represents finite rotation using an orthonormal triad rather than rotational coordinates. The paper introduces a director-dependent linear transformation that connects variations and rates of the stacked directors to spatial virtual rotation and angular velocity. The orientation and scaling of this transformation follow from the orthonormality relations of the director triad and are essential to the port-Hamiltonian boundary variables. This subproblem asks for that transformation itself, rather than only its action on one particular director-rate vector.

Returns
-------
return T
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_director_kinematic_map(
    directors: np.ndarray,
) -> np.ndarray:
    """Return the paper-specific 9-by-3 director kinematic matrix."""
    return np.zeros((9, 3), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_director_kinematic_map(
    directors: np.ndarray,
) -> np.ndarray:
    import numpy as np

    directors = np.asarray(
        directors,
        dtype=float,
    )

    if directors.shape != (3, 3):
        raise ValueError(
            "directors must have shape (3, 3)."
        )

    if not np.all(np.isfinite(directors)):
        raise ValueError(
            "directors must contain only finite values."
        )

    gram = directors @ directors.T

    if not np.allclose(
        gram,
        np.eye(3),
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ValueError(
            "directors must form an orthonormal triad."
        )

    if not np.isclose(
        np.linalg.det(directors),
        1.0,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ValueError(
            "directors must form a proper right-handed triad."
        )

    def skew(a):
        x, y, z = a

        return np.array([
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ], dtype=float)

    T = -0.5 * np.vstack([
        skew(directors[0]),
        skew(directors[1]),
        skew(directors[2]),
    ])

    return T

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return test cases for the director kinematic map."""
    return [
        {
            "setup": """import numpy as np

theta = np.pi / 6.0

directors = np.array([
    [np.cos(theta),  np.sin(theta), 0.0],
    [-np.sin(theta), np.cos(theta), 0.0],
    [0.0,            0.0,           1.0],
])
""",
            "call": """build_director_kinematic_map(
    directors
)""",
            "gold_call": """_oracle_build_director_kinematic_map(
    directors
)""",
        },

        {
            "setup": """import numpy as np

directors = np.eye(3)
""",
            "call": """build_director_kinematic_map(
    directors
)""",
            "gold_call": """_oracle_build_director_kinematic_map(
    directors
)""",
        },

        {
            "setup": """import numpy as np

theta = np.pi / 4.0

directors = np.array([
    [np.cos(theta), 0.0, -np.sin(theta)],
    [0.0,           1.0,  0.0],
    [np.sin(theta), 0.0,  np.cos(theta)],
])
""",
            "call": """build_director_kinematic_map(
    directors
)""",
            "gold_call": """_oracle_build_director_kinematic_map(
    directors
)""",
        },
    ]
