"""
Construct the anisotropic thermal-conductivity tensor of each omnidirectional cell. Rotate the two principal conductivities from the local gradient-aligned basis into the common coordinate basis using the rotation recovered from the design-state temperature gradient.

For each cell, the conductivity tensor in its local principal basis is



$$

\boldsymbol{\kappa}'_{D,i}=

\begin{pmatrix}

\kappa_{\parallel,i} & 0\\

0 & \kappa_{\perp,i}

\end{pmatrix}.

$$



Using the gradient-aligned rotation matrix $\mathbf R_i$, express the conductivity tensor in the common coordinate basis as

$$

\boldsymbol{\kappa}_{D,i}

=\mathbf R_i

\boldsymbol{\kappa}'_{D,i}

\mathbf R_i^T.

$$

Because $\mathbf R_i$ is orthogonal, this rotation preserves the two principal conductivities while introducing off-diagonal components when the principal axes are not aligned with the coordinate axes.

Returns
-------
numerical NumPy array with shape (N, 2, 2), containing the transformed anisotropic thermal-conductivity tensor for each cell
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def construct_conductivity_tensors(
    rotations: "np.ndarray",
    principal_conductivities: "np.ndarray",
) -> "np.ndarray":
    """
    Rotate local principal conductivities into the common coordinate basis.

    Parameters
    ----------
    rotations : np.ndarray
        Gradient-aligned rotation matrices with shape (N, 2, 2).
    principal_conductivities : np.ndarray
        Positive principal conductivity pairs with shape (N, 2).
        Column 0 contains the conductivity parallel to the local
        design-state gradient and column 1 contains the orthogonal
        conductivity.

    Returns
    -------
    conductivity_tensors : np.ndarray
        Rotated anisotropic conductivity tensors with shape (N, 2, 2).

    Raises
    ------
    ValueError
        If any principal conductivity is non-finite or non-positive, or if
        any supplied rotation matrix is not orthogonal with determinant +1.
    """
    return conductivity_tensors

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_construct_conductivity_tensors(
    rotations: "np.ndarray",
    principal_conductivities: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation of the conductivity-tensor rotation."""
    rotations = np.asarray(rotations, dtype=float)
    principal_conductivities = np.asarray(
        principal_conductivities,
        dtype=float,
    )

    if rotations.ndim != 3 or rotations.shape[1:] != (2, 2):
        raise ValueError("rotations must have shape (N, 2, 2).")

    n_cells = rotations.shape[0]

    if n_cells < 1:
        raise ValueError("rotations must contain at least one cell.")

    if principal_conductivities.shape != (n_cells, 2):
        raise ValueError(
            "principal_conductivities must have shape (N, 2)."
        )

    if not np.all(np.isfinite(rotations)):
        raise ValueError("rotations must contain only finite values.")

    if not np.all(np.isfinite(principal_conductivities)):
        raise ValueError(
            "principal_conductivities must contain only finite values."
        )

    if np.any(principal_conductivities <= 0.0):
        raise ValueError(
            "principal conductivities must be positive."
        )

    identity = np.eye(2, dtype=float)

    for rotation in rotations:
        if not np.allclose(
            rotation.T @ rotation,
            identity,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError("Each rotation matrix must be orthogonal.")

        if not np.isclose(
            np.linalg.det(rotation),
            1.0,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError(
                "Each rotation matrix must have determinant +1."
            )

    conductivity_tensors = np.empty(
        (n_cells, 2, 2),
        dtype=float,
    )

    for i in range(n_cells):
        principal_tensor = np.diag(
            principal_conductivities[i]
        )

        conductivity_tensors[i] = (
            rotations[i]
            @ principal_tensor
            @ rotations[i].T
        )

    return conductivity_tensors

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test cases for construct_conductivity_tensors."""
    return [
        {
            "setup": """import numpy as np

gradients = np.array([
    [0.34, 0.79],
    [0.72, 0.31],
    [0.46, -0.88],
], dtype=float)

magnitudes = np.linalg.norm(gradients, axis=1)

rotations = np.empty((3, 2, 2), dtype=float)
rotations[:, 0, 0] = gradients[:, 0] / magnitudes
rotations[:, 0, 1] = -gradients[:, 1] / magnitudes
rotations[:, 1, 0] = gradients[:, 1] / magnitudes
rotations[:, 1, 1] = gradients[:, 0] / magnitudes

principal_conductivities = np.array([
    [0.22, 1.0 / 0.22],
    [1.85, 1.0 / 1.85],
    [4.60, 1.0 / 4.60],
], dtype=float)
""",
            "call": "construct_conductivity_tensors(rotations.copy(), principal_conductivities.copy())",
            "gold_call": "_oracle_construct_conductivity_tensors(rotations.copy(), principal_conductivities.copy())",
        },
        {
            "setup": """import numpy as np

rotations = np.array([
    [[1.0, 0.0],
     [0.0, 1.0]],

    [[0.0, -1.0],
     [1.0,  0.0]],
], dtype=float)

principal_conductivities = np.array([
    [2.0, 2.0],
    [3.0, 1.0],
], dtype=float)
""",
            "call": "construct_conductivity_tensors(rotations.copy(), principal_conductivities.copy())",
            "gold_call": "_oracle_construct_conductivity_tensors(rotations.copy(), principal_conductivities.copy())",
        },
        {
            "setup": """import numpy as np

theta = np.deg2rad(-45.0)

rotations = np.array([
    [[np.cos(theta), -np.sin(theta)],
     [np.sin(theta),  np.cos(theta)]]
], dtype=float)

principal_conductivities = np.array([
    [0.02, 50.0],
], dtype=float)
""",
            "call": "construct_conductivity_tensors(rotations.copy(), principal_conductivities.copy())",
            "gold_call": "_oracle_construct_conductivity_tensors(rotations.copy(), principal_conductivities.copy())",
        },
        {
            "setup": """import numpy as np

rotations = np.array([
    [[1.0, 0.0],
     [0.0, 1.0]]
], dtype=float)

principal_conductivities = np.array([
    [1.0, 0.0],
], dtype=float)

def run_model():
    try:
        construct_conductivity_tensors(
            rotations.copy(),
            principal_conductivities.copy(),
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_construct_conductivity_tensors(
            rotations.copy(),
            principal_conductivities.copy(),
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
