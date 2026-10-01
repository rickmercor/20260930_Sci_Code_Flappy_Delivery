"""
Construct the local geometric transformation used by the field-derived thermal-metamaterial method. For each cell, combine the recovered scale factor and gradient-aligned rotation with the conductivity-dependent transverse stretch, then calculate the determinant of the local Jacobian.

For each cell, the transverse stretching matrix is

$$

\mathbf S_i=\begin{pmatrix}

1 & 0\\

0 & \kappa_0/\kappa_{P,i}

\end{pmatrix}.

$$

Using the scale factor $\lambda_i$ and the gradient-aligned rotation matrix $\mathbf R_i$, construct the local Jacobian as

$$

\boldsymbol{\Lambda}_i=\lambda_i\mathbf R_i\mathbf S_i.

$$

The local Jacobian determinant is

$$

J_i=\det\boldsymbol{\Lambda}_i.

$$

For positive $\lambda_i$, $\kappa_0$, and $\kappa_{P,i}$, the field-derived transformation is orientation preserving and therefore requires $J_i>0$.

Returns
-------
tuple of three numerical NumPy arrays: stretching matrices with shape (N, 2, 2), local Jacobian matrices with shape (N, 2, 2), and Jacobian determinants with shape (N,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_dual_jacobian(
    scale_factors: "np.ndarray",
    rotations: "np.ndarray",
    kappa_0: float,
    kappa_p: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """
    Construct the local stretching matrices and field-derived Jacobians.

    Parameters
    ----------
    scale_factors : np.ndarray
        Positive local geometric scale factors with shape (N,).
    rotations : np.ndarray
        Gradient-aligned rotation matrices with shape (N, 2, 2).
    kappa_0 : float
        Positive background thermal conductivity.
    kappa_p : np.ndarray
        Positive local isotropic conductivities with shape (N,).

    Returns
    -------
    stretches : np.ndarray
        Local stretching matrices with shape (N, 2, 2).
    jacobians : np.ndarray
        Local Jacobian matrices with shape (N, 2, 2).
    determinants : np.ndarray
        Local Jacobian determinants with shape (N,).

    Raises
    ------
    ValueError
        If `scale_factors`, `kappa_0`, or `kappa_p` contains a non-finite or
        non-positive value, or if any constructed Jacobian has a non-finite
        or non-positive determinant.
    """
    return stretches, jacobians, determinants

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_build_dual_jacobian(
    scale_factors: "np.ndarray",
    rotations: "np.ndarray",
    kappa_0: float,
    kappa_p: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Reference implementation of the local Jacobian construction."""
    scale_factors = np.asarray(scale_factors, dtype=float)
    rotations = np.asarray(rotations, dtype=float)
    kappa_p = np.asarray(kappa_p, dtype=float)

    if scale_factors.ndim != 1 or scale_factors.size < 1:
        raise ValueError("scale_factors must have shape (N,) with N >= 1.")

    n_cells = scale_factors.size

    if rotations.shape != (n_cells, 2, 2):
        raise ValueError("rotations must have shape (N, 2, 2).")

    if kappa_p.shape != (n_cells,):
        raise ValueError("kappa_p must have shape (N,).")

    if not np.isscalar(kappa_0) or not np.isfinite(float(kappa_0)):
        raise ValueError("kappa_0 must be a finite scalar.")

    kappa_0 = float(kappa_0)

    if kappa_0 <= 0.0:
        raise ValueError("kappa_0 must be positive.")

    if not np.all(np.isfinite(scale_factors)):
        raise ValueError("scale_factors must contain only finite values.")

    if not np.all(np.isfinite(rotations)):
        raise ValueError("rotations must contain only finite values.")

    if not np.all(np.isfinite(kappa_p)):
        raise ValueError("kappa_p must contain only finite values.")

    if np.any(scale_factors <= 0.0):
        raise ValueError("scale_factors must be positive.")

    if np.any(kappa_p <= 0.0):
        raise ValueError("kappa_p must be positive.")

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

    stretches = np.zeros((n_cells, 2, 2), dtype=float)
    stretches[:, 0, 0] = 1.0
    stretches[:, 1, 1] = kappa_0 / kappa_p

    jacobians = (
        scale_factors[:, None, None]
        * np.matmul(rotations, stretches)
    )

    determinants = np.linalg.det(jacobians)

    if np.any(determinants <= 0.0):
        raise ValueError(
            "Every local Jacobian must have positive determinant."
        )

    return stretches, jacobians, determinants

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test cases for build_dual_jacobian."""
    return [
        {
            "setup": """import numpy as np

gradients = np.array([
    [0.34, 0.79],
    [0.72, 0.31],
    [0.46, -0.88],
], dtype=float)

magnitudes = np.linalg.norm(gradients, axis=1)
scale_factors = 1.0 / magnitudes

rotations = np.empty((3, 2, 2), dtype=float)
rotations[:, 0, 0] = gradients[:, 0] / magnitudes
rotations[:, 0, 1] = -gradients[:, 1] / magnitudes
rotations[:, 1, 0] = gradients[:, 1] / magnitudes
rotations[:, 1, 1] = gradients[:, 0] / magnitudes

kappa_0 = 1.0
kappa_p = np.array([0.22, 1.85, 4.60], dtype=float)
""",
            "call": "build_dual_jacobian(scale_factors.copy(), rotations.copy(), kappa_0, kappa_p.copy())",
            "gold_call": "_oracle_build_dual_jacobian(scale_factors.copy(), rotations.copy(), kappa_0, kappa_p.copy())",
        },
        {
            "setup": """import numpy as np

scale_factors = np.array([1.0, 2.0], dtype=float)

rotations = np.array([
    [[1.0, 0.0],
     [0.0, 1.0]],

    [[0.0, -1.0],
     [1.0,  0.0]],
], dtype=float)

kappa_0 = 2.0
kappa_p = np.array([2.0, 2.0], dtype=float)
""",
            "call": "build_dual_jacobian(scale_factors.copy(), rotations.copy(), kappa_0, kappa_p.copy())",
            "gold_call": "_oracle_build_dual_jacobian(scale_factors.copy(), rotations.copy(), kappa_0, kappa_p.copy())",
        },
        {
            "setup": """import numpy as np

theta = np.deg2rad(-60.0)

scale_factors = np.array([0.75], dtype=float)

rotations = np.array([
    [[np.cos(theta), -np.sin(theta)],
     [np.sin(theta),  np.cos(theta)]]
], dtype=float)

kappa_0 = 1.0
kappa_p = np.array([0.05], dtype=float)
""",
            "call": "build_dual_jacobian(scale_factors.copy(), rotations.copy(), kappa_0, kappa_p.copy())",
            "gold_call": "_oracle_build_dual_jacobian(scale_factors.copy(), rotations.copy(), kappa_0, kappa_p.copy())",
        },
        {
            "setup": """import numpy as np

scale_factors = np.array([1.0], dtype=float)
rotations = np.array([
    [[1.0, 0.0],
     [0.0, 1.0]]
], dtype=float)

kappa_0 = 1.0
kappa_p = np.array([0.0], dtype=float)

def run_model():
    try:
        build_dual_jacobian(
            scale_factors.copy(), rotations.copy(), kappa_0, kappa_p.copy()
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_build_dual_jacobian(
            scale_factors.copy(), rotations.copy(), kappa_0, kappa_p.copy()
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
