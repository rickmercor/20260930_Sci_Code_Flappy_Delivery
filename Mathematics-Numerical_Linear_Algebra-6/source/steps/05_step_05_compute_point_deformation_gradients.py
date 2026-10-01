"""
Recover the nonlocal deformation gradient at every material point.

With displacement $U_k=x_k-X_k$ and stored volume-weighted shape gradients,

the first-order reconstruction is



$$

\bar F_k=I+\sum_n (U_n-U_k)\otimes\nabla\phi_{kn}.

$$



The identity term is essential: the summation reconstructs the displacement

gradient, not the deformation gradient itself. Directed shape gradients allow

different kinematic neighborhoods at the two ends of a damaged bond.

Returns
-------
A finite float array of shape (N, 3, 3) containing dimensionless point gradients.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_point_deformation_gradients(
    displacements: np.ndarray,
    shape_gradients: np.ndarray,
) -> np.ndarray:
    r"""Compute all nonlocal point deformation gradients $\bar F_k$.

    Raises ``ValueError`` unless ``displacements`` is a finite array of shape
    $(N,3)$ with $N\geq2$, ``shape_gradients`` is a finite array of shape
    $(N,N,3)$, and every diagonal shape-gradient vector is zero.

    Parameters
    ----------
    displacements : np.ndarray
        Point displacements of shape $(N,3)$ in mm.
    shape_gradients : np.ndarray
        Directed volume-weighted gradients of shape $(N,N,3)$ in mm$^{-1}$.

    Returns
    -------
    np.ndarray
        Dimensionless deformation gradients of shape $(N,3,3)$.
    """
    return NotImplemented

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_point_deformation_gradients(
    displacements: np.ndarray,
    shape_gradients: np.ndarray,
) -> np.ndarray:
    """Reference first-order nonlocal reconstruction."""
    displacements = np.asarray(displacements, dtype=float)
    gradients = np.asarray(shape_gradients, dtype=float)
    if (
        displacements.ndim != 2
        or displacements.shape[1] != 3
        or displacements.shape[0] < 2
    ):
        raise ValueError("displacements must have shape (N, 3) with N at least two")
    point_count = displacements.shape[0]
    if gradients.shape != (point_count, point_count, 3):
        raise ValueError("shape_gradients must have shape (N, N, 3)")
    if not np.all(np.isfinite(displacements)) or not np.all(np.isfinite(gradients)):
        raise ValueError("displacements and shape_gradients must be finite")
    if not np.allclose(
        gradients[np.arange(point_count), np.arange(point_count)],
        0.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("self-bond shape gradients must be zero")
    deformation_gradients = np.repeat(np.eye(3)[None, :, :], point_count, axis=0)
    for k in range(point_count):
        for n in range(point_count):
            deformation_gradients[k] += np.outer(
                displacements[n] - displacements[k], gradients[k, n]
            )
    if not np.all(np.isfinite(deformation_gradients)):
        raise ValueError("deformation gradients must be finite")
    return deformation_gradients

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return nonaffine-cube, zero-motion, and invalid-diagonal cases."""
    return [
        {
            "setup": """import numpy as np
reference_positions = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]], dtype=float)
volumes = np.array([1.00,0.95,1.05,1.10,0.90,1.08,0.98,1.02]); previous_phase = np.zeros((8,8))
horizon = 2.0; kernel_coefficients = np.array([1.0,-2.0,1.0]); kinematic_threshold = 0.8
moments = _oracle_build_kinematic_moments(reference_positions, volumes, previous_phase, horizon, kernel_coefficients, kinematic_threshold)
shape_gradients = _oracle_compute_bond_shape_gradients(reference_positions, volumes, previous_phase, moments, horizon, kernel_coefficients, kinematic_threshold)
displacements = np.column_stack((0.004*reference_positions[:,0] + 0.002*reference_positions[:,1]*reference_positions[:,2], -0.001*reference_positions[:,1] + 0.0015*reference_positions[:,0]*reference_positions[:,2], 0.002*reference_positions[:,2] - 0.001*reference_positions[:,0]*reference_positions[:,1]))
displacements[7] += np.array([0.0012,-0.0007,0.0009])
""",
            "call": "compute_point_deformation_gradients(displacements, shape_gradients)",
            "gold_call": "_oracle_compute_point_deformation_gradients(displacements, shape_gradients)",
        },
        {
            "setup": """import numpy as np
displacements = np.zeros((2,3)); shape_gradients = np.zeros((2,2,3))
""",
            "call": "compute_point_deformation_gradients(displacements, shape_gradients)",
            "gold_call": "_oracle_compute_point_deformation_gradients(displacements, shape_gradients)",
        },
        {
            "setup": """import numpy as np
displacements = np.zeros((2,3)); shape_gradients = np.zeros((2,2,3)); shape_gradients[0,0,0] = 1.0
def run_model():
    try:
        compute_point_deformation_gradients(displacements, shape_gradients)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_point_deformation_gradients(displacements, shape_gradients)
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
