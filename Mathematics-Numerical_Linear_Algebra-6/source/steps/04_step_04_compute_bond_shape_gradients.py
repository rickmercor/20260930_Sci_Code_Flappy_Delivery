"""
Compute the volume-weighted first-order shape gradients of all active bonds.

The discrete shape derivative for an active directed bond $(k,n)$ is



$$

\nabla\phi_{kn}=\omega(r_{kn})h(s^n_{kn})M_k^{-1}\Delta X_{kn}V_n.

$$



Solving the linear system is numerically preferable to forming $M_k^{-1}$.

The factor $V_n$ belongs to the stored gradient; this convention is

load-bearing in the later deformation-gradient and force-state contractions.

Returns
-------
A float array of shape (N, N, 3) containing directed shape gradients in mm^-1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_bond_shape_gradients(
    reference_positions: np.ndarray,
    volumes: np.ndarray,
    previous_phase: np.ndarray,
    moments: np.ndarray,
    horizon: float,
    kernel_coefficients: np.ndarray,
    kinematic_threshold: float,
) -> np.ndarray:
    r"""Compute $\nabla\phi_{kn}$ for every directed bond.

    Raises ``ValueError`` unless the point, volume, phase, horizon, kernel, and
    threshold inputs satisfy the moment-builder contract; ``moments`` has shape
    $(N,3,3)$ with finite symmetric positive-definite matrices; and every
    active kernel value is nonnegative.

    Parameters
    ----------
    reference_positions : np.ndarray
        Reference coordinates of shape $(N,3)$ in mm.
    volumes : np.ndarray
        Positive nodal volumes of shape $(N,)$ in mm$^3$.
    previous_phase : np.ndarray
        Symmetric old-time phase field of shape $(N,N)$.
    moments : np.ndarray
        Moment matrices of shape $(N,3,3)$ in mm$^5$.
    horizon : float
        Inclusive neighborhood radius $\delta$ in mm.
    kernel_coefficients : np.ndarray
        Ascending coefficients of $\omega(r/\delta)$.
    kinematic_threshold : float
        Kinematic degradation threshold $s_c$ in $[0,1)$.

    Returns
    -------
    np.ndarray
        Shape-gradient array of shape $(N,N,3)$ in mm$^{-1}$, zero off bonds.
    """
    return NotImplemented

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_bond_shape_gradients(
    reference_positions: np.ndarray,
    volumes: np.ndarray,
    previous_phase: np.ndarray,
    moments: np.ndarray,
    horizon: float,
    kernel_coefficients: np.ndarray,
    kinematic_threshold: float,
) -> np.ndarray:
    """Reference system-solve construction of directed shape gradients."""
    positions = np.asarray(reference_positions, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    phase = np.asarray(previous_phase, dtype=float)
    moments = np.asarray(moments, dtype=float)
    coefficients = np.asarray(kernel_coefficients, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 4:
        raise ValueError(
            "reference_positions must have shape (N, 3) with N at least four"
        )
    point_count = positions.shape[0]
    if volumes.shape != (point_count,) or np.any(volumes <= 0.0):
        raise ValueError("volumes must be positive with shape (N,)")
    if (
        phase.shape != (point_count, point_count)
        or np.any(phase < 0.0)
        or np.any(phase > 1.0)
    ):
        raise ValueError("previous_phase must have shape (N, N) with values in [0, 1]")
    if not np.allclose(phase, phase.T, rtol=0.0, atol=1e-12) or not np.allclose(
        np.diag(phase), 0.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("previous_phase must be symmetric with zero diagonal")
    if moments.shape != (point_count, 3, 3) or not np.all(np.isfinite(moments)):
        raise ValueError("moments must be a finite array of shape (N, 3, 3)")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(volumes)):
        raise ValueError("positions and volumes must be finite")
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be positive and finite")
    if (
        coefficients.ndim != 1
        or coefficients.size == 0
        or not np.all(np.isfinite(coefficients))
    ):
        raise ValueError("kernel_coefficients must be a nonempty finite vector")
    if not np.isfinite(kinematic_threshold) or not 0.0 <= kinematic_threshold < 1.0:
        raise ValueError("kinematic_threshold must lie in [0, 1)")
    gradients = np.zeros((point_count, point_count, 3), dtype=float)
    for k in range(point_count):
        if not np.allclose(moments[k], moments[k].T, rtol=0.0, atol=1e-12):
            raise ValueError("each moment must be symmetric")
        eigenvalues = np.linalg.eigvalsh(moments[k])
        if eigenvalues[0] <= 100.0 * np.finfo(float).eps * max(1.0, eigenvalues[-1]):
            raise ValueError("each moment must be positive definite")
        for n in range(point_count):
            difference = positions[n] - positions[k]
            distance = float(np.linalg.norm(difference))
            if 0.0 < distance <= horizon:
                weight = np.polynomial.polynomial.polyval(
                    distance / horizon, coefficients
                )
                if weight < -1e-12:
                    raise ValueError("the kernel is negative on an active bond")
                factor = 1.0
                if phase[k, n] > kinematic_threshold:
                    factor = ((1.0 - phase[k, n]) / (1.0 - kinematic_threshold)) ** 2
                gradients[k, n] = (
                    weight
                    * factor
                    * np.linalg.solve(moments[k], difference)
                    * volumes[n]
                )
    return gradients

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return asymmetric-volume, threshold-boundary, and indefinite-moment cases."""
    return [
        {
            "setup": """import numpy as np
reference_positions = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1]], dtype=float)
volumes = np.array([1.0,0.8,1.2,0.9,1.1])
previous_phase = np.zeros((5,5)); previous_phase[0,4] = previous_phase[4,0] = 0.9
horizon = 2.0; kernel_coefficients = np.array([1.0,-2.0,1.0]); kinematic_threshold = 0.8
moments = _oracle_build_kinematic_moments(reference_positions, volumes, previous_phase, horizon, kernel_coefficients, kinematic_threshold)
""",
            "call": "compute_bond_shape_gradients(reference_positions, volumes, previous_phase, moments, horizon, kernel_coefficients, kinematic_threshold)",
            "gold_call": "_oracle_compute_bond_shape_gradients(reference_positions, volumes, previous_phase, moments, horizon, kernel_coefficients, kinematic_threshold)",
        },
        {
            "setup": """import numpy as np
reference_positions = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
volumes = np.ones(4); previous_phase = np.full((4,4),0.8); np.fill_diagonal(previous_phase,0.0)
horizon = np.sqrt(2.0); kernel_coefficients = np.array([1.0]); kinematic_threshold = 0.8
moments = _oracle_build_kinematic_moments(reference_positions, volumes, previous_phase, horizon, kernel_coefficients, kinematic_threshold)
""",
            "call": "compute_bond_shape_gradients(reference_positions, volumes, previous_phase, moments, horizon, kernel_coefficients, kinematic_threshold)",
            "gold_call": "_oracle_compute_bond_shape_gradients(reference_positions, volumes, previous_phase, moments, horizon, kernel_coefficients, kinematic_threshold)",
        },
        {
            "setup": """import numpy as np
reference_positions = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
volumes = np.ones(4); previous_phase = np.zeros((4,4)); horizon = 2.0
kernel_coefficients = np.array([1.0]); kinematic_threshold = 0.8
moments = np.repeat(np.eye(3)[None,:,:],4,axis=0); moments[2,2,2] = -1.0
def run_model():
    try:
        compute_bond_shape_gradients(reference_positions, volumes, previous_phase, moments, horizon, kernel_coefficients, kinematic_threshold)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_bond_shape_gradients(reference_positions, volumes, previous_phase, moments, horizon, kernel_coefficients, kinematic_threshold)
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
