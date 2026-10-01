"""
Recover the local geometric information used by the field-derived transformation. For each nonzero design-state temperature gradient, calculate its magnitude, the reference-to-local gradient scale factor, and the rotation matrix whose first column is aligned with the local gradient direction.

For a reference temperature gradient $\mathbf G_0$ and a local design-state temperature gradient $\mathbf G_i=(G_{x,i},G_{y,i})^T$, calculate the local gradient magnitude as

$$

|\mathbf G_i|=\sqrt{G_{x,i}^2+G_{y,i}^2}, \qquad \lambda_i=\frac{|\mathbf G_0|}{|\mathbf G_i|}.

$$

The local rotation matrix is

$$

\mathbf R_i=\frac{1}{|\mathbf G_i|}

\begin{pmatrix}

G_{x,i} & -G_{y,i}\\

G_{y,i} & G_{x,i}

\end{pmatrix}.

$$

Returns
-------
tuple of three numerical NumPy arrays: local gradient magnitudes with shape (N,), scale factors with shape (N,), and rotation matrices with shape (N, 2, 2)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def recover_field_geometry(
    g0: "np.ndarray",
    gradients: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """
    Recover local field geometry from reference and design-state gradients.

    Parameters
    ----------
    g0 : np.ndarray
        Reference temperature-gradient vector with shape (2,).
    gradients : np.ndarray
        Local temperature-gradient vectors with shape (N, 2).

    Returns
    -------
    magnitudes : np.ndarray
        Euclidean magnitudes of the local gradients with shape (N,).
    scale_factors : np.ndarray
        Reference-to-local gradient scale factors with shape (N,).
    rotations : np.ndarray
        Gradient-aligned rotation matrices with shape (N, 2, 2).

    Raises
    ------
    ValueError
        If the reference gradient or any local gradient contains a non-finite
        component, or if the reference gradient or any local gradient has
        zero magnitude.
    """
    return magnitudes, scale_factors, rotations

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_recover_field_geometry(
    g0: "np.ndarray",
    gradients: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Reference implementation of the field-geometry recovery."""
    g0 = np.asarray(g0, dtype=float)
    gradients = np.asarray(gradients, dtype=float)

    if g0.shape != (2,):
        raise ValueError("g0 must have shape (2,).")

    if gradients.ndim != 2 or gradients.shape[1] != 2:
        raise ValueError("gradients must have shape (N, 2).")

    if gradients.shape[0] < 1:
        raise ValueError("gradients must contain at least one local gradient.")

    if not np.all(np.isfinite(g0)) or not np.all(np.isfinite(gradients)):
        raise ValueError("All gradient values must be finite.")

    g0_norm = float(np.linalg.norm(g0))

    if g0_norm <= 0.0:
        raise ValueError("The reference gradient must be nonzero.")

    magnitudes = np.linalg.norm(gradients, axis=1)

    if np.any(magnitudes <= 0.0):
        raise ValueError("Every local gradient must be nonzero.")

    scale_factors = g0_norm / magnitudes

    rotations = np.empty((gradients.shape[0], 2, 2), dtype=float)

    rotations[:, 0, 0] = gradients[:, 0] / magnitudes
    rotations[:, 0, 1] = -gradients[:, 1] / magnitudes
    rotations[:, 1, 0] = gradients[:, 1] / magnitudes
    rotations[:, 1, 1] = gradients[:, 0] / magnitudes

    return magnitudes, scale_factors, rotations

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test cases for recover_field_geometry."""
    return [
        {
            "setup": """import numpy as np
g0 = np.array([1.0, 0.0], dtype=float)
gradients = np.array([
    [0.34, 0.79],
    [0.72, 0.31],
    [0.46, -0.88],
], dtype=float)
""",
            "call": "recover_field_geometry(g0.copy(), gradients.copy())",
            "gold_call": "_oracle_recover_field_geometry(g0.copy(), gradients.copy())",
        },
        {
            "setup": """import numpy as np
g0 = np.array([2.0, 0.0], dtype=float)
gradients = np.array([
    [1.0, 0.0],
    [0.0, 2.0],
], dtype=float)
""",
            "call": "recover_field_geometry(g0.copy(), gradients.copy())",
            "gold_call": "_oracle_recover_field_geometry(g0.copy(), gradients.copy())",
        },
        {
            "setup": """import numpy as np
g0 = np.array([1.0, 0.0], dtype=float)
gradients = np.array([
    [-0.6, -0.8],
    [0.6, -0.8],
], dtype=float)
""",
            "call": "recover_field_geometry(g0.copy(), gradients.copy())",
            "gold_call": "_oracle_recover_field_geometry(g0.copy(), gradients.copy())",
        },
        {
            "setup": """import numpy as np
g0 = np.array([1.0, 0.0], dtype=float)
gradients = np.array([[0.0, 0.0]], dtype=float)

def run_model():
    try:
        recover_field_geometry(g0.copy(), gradients.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_recover_field_geometry(g0.copy(), gradients.copy())
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
