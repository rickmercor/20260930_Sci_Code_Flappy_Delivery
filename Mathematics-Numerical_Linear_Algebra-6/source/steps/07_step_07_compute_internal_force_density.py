"""
Assemble the bond-associated force state and internal force density.

Let $e_{kn}=\Delta X_{kn}/r_{kn}$,

$\omega_k^0=\sum_n\omega_{kn}V_n$, and

$\bar\omega_{kn}=\tfrac12(\omega_{kn}/\omega_k^0+

\omega_{kn}/\omega_n^0)$. The stabilization tensor and directed force are



$$

Z_k=\sum_n\bar\omega_{kn}\widetilde P_{kn}

(I-e_{kn}\otimes e_{kn})V_n,

$$



$$

t_{kn}=\bar\omega_{kn}(\widetilde P_{kn}e_{kn})

+Z_k\frac{\nabla\phi_{kn}}{V_n},\qquad

B_k^{\mathrm{int}}=\sum_n(t_{kn}-t_{nk})V_n.

$$



The antisymmetric difference enforces global linear-momentum closure under

the nodal-volume quadrature.

Returns
-------
A finite float array of shape (N, 3) containing internal force density in N/mm^3.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_internal_force_density(
    reference_positions: np.ndarray,
    volumes: np.ndarray,
    horizon: float,
    kernel_coefficients: np.ndarray,
    shape_gradients: np.ndarray,
    degraded_piola: np.ndarray,
) -> np.ndarray:
    r"""Assemble $B_k^{\mathrm{int}}$ from directed bond force states.

    Raises ``ValueError`` unless reference positions have finite shape $(N,3)$
    with $N\geq2$; volumes are finite and positive with shape $(N,)$; the
    horizon is positive and finite; kernel coefficients are a nonempty finite
    vector; shape gradients and degraded stresses have shapes $(N,N,3)$ and
    $(N,N,3,3)$ with finite entries and zero diagonals; active kernel values
    are nonnegative; and every point has positive kernel volume $\omega_k^0$.

    Parameters
    ----------
    reference_positions : np.ndarray
        Reference coordinates of shape $(N,3)$ in mm.
    volumes : np.ndarray
        Positive nodal volumes of shape $(N,)$ in mm$^3$.
    horizon : float
        Inclusive neighborhood radius $\delta$ in mm.
    kernel_coefficients : np.ndarray
        Ascending coefficients of $\omega(r/\delta)$.
    shape_gradients : np.ndarray
        Directed volume-weighted gradients of shape $(N,N,3)$ in mm$^{-1}$.
    degraded_piola : np.ndarray
        Directed bond stresses of shape $(N,N,3,3)$ in MPa.

    Returns
    -------
    np.ndarray
        Internal force densities of shape $(N,3)$ in N/mm$^3$.
    """
    return NotImplemented

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_internal_force_density(
    reference_positions: np.ndarray,
    volumes: np.ndarray,
    horizon: float,
    kernel_coefficients: np.ndarray,
    shape_gradients: np.ndarray,
    degraded_piola: np.ndarray,
) -> np.ndarray:
    """Reference force-state assembly with normalized symmetric weights."""
    positions = np.asarray(reference_positions, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    coefficients = np.asarray(kernel_coefficients, dtype=float)
    gradients = np.asarray(shape_gradients, dtype=float)
    stresses = np.asarray(degraded_piola, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 2:
        raise ValueError(
            "reference_positions must have shape (N, 3) with N at least two"
        )
    point_count = positions.shape[0]
    if volumes.shape != (point_count,) or np.any(volumes <= 0.0):
        raise ValueError("volumes must be positive with shape (N,)")
    if gradients.shape != (point_count, point_count, 3):
        raise ValueError("shape_gradients must have shape (N, N, 3)")
    if stresses.shape != (point_count, point_count, 3, 3):
        raise ValueError("degraded_piola must have shape (N, N, 3, 3)")
    if (
        not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(volumes))
        or not np.all(np.isfinite(gradients))
        or not np.all(np.isfinite(stresses))
    ):
        raise ValueError("all numerical arrays must be finite")
    if not np.allclose(
        gradients[np.arange(point_count), np.arange(point_count)],
        0.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("self-bond shape gradients must be zero")
    if not np.allclose(
        stresses[np.arange(point_count), np.arange(point_count)],
        0.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("self-bond stresses must be zero")
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be positive and finite")
    if (
        coefficients.ndim != 1
        or coefficients.size == 0
        or not np.all(np.isfinite(coefficients))
    ):
        raise ValueError("kernel_coefficients must be a nonempty finite vector")
    distances = np.linalg.norm(positions[None, :, :] - positions[:, None, :], axis=2)
    active = (distances > 0.0) & (distances <= horizon)
    weights = np.zeros((point_count, point_count), dtype=float)
    weights[active] = np.polynomial.polynomial.polyval(
        distances[active] / horizon, coefficients
    )
    if np.any(weights[active] < -1e-12):
        raise ValueError("the kernel is negative on an active bond")
    kernel_volumes = weights @ volumes
    if np.any(kernel_volumes <= 0.0) or not np.all(np.isfinite(kernel_volumes)):
        raise ValueError("each point must have positive finite kernel volume")
    normalized = (
        0.5 * weights * (1.0 / kernel_volumes[:, None] + 1.0 / kernel_volumes[None, :])
    )
    stabilization = np.zeros((point_count, 3, 3), dtype=float)
    identity = np.eye(3)
    for k in range(point_count):
        for n in range(point_count):
            if active[k, n]:
                direction = (positions[n] - positions[k]) / distances[k, n]
                stabilization[k] += (
                    normalized[k, n]
                    * stresses[k, n]
                    @ (identity - np.outer(direction, direction))
                    * volumes[n]
                )
    force_state = np.zeros((point_count, point_count, 3), dtype=float)
    for k in range(point_count):
        for n in range(point_count):
            if active[k, n]:
                direction = (positions[n] - positions[k]) / distances[k, n]
                bond_shape = gradients[k, n] / volumes[n]
                force_state[k, n] = (
                    normalized[k, n] * (stresses[k, n] @ direction)
                    + stabilization[k] @ bond_shape
                )
    internal_force = np.zeros((point_count, 3), dtype=float)
    for k in range(point_count):
        for n in range(point_count):
            internal_force[k] += (force_state[k, n] - force_state[n, k]) * volumes[n]
    return internal_force

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return nonsymmetric-state, zero-stress, and isolated-point cases."""
    return [
        {
            "setup": """import numpy as np
reference_positions = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float); volumes = np.array([1.0,0.8,1.1,0.9])
horizon = 2.0; kernel_coefficients = np.array([1.0,-2.0,1.0]); shape_gradients = np.zeros((4,4,3)); degraded_piola = np.zeros((4,4,3,3))
for k in range(4):
    for n in range(4):
        if k != n:
            shape_gradients[k,n] = 0.1*(reference_positions[n]-reference_positions[k])
            degraded_piola[k,n] = np.diag([10.0+k,8.0+n,6.0])
""",
            "call": "compute_internal_force_density(reference_positions, volumes, horizon, kernel_coefficients, shape_gradients, degraded_piola)",
            "gold_call": "_oracle_compute_internal_force_density(reference_positions, volumes, horizon, kernel_coefficients, shape_gradients, degraded_piola)",
        },
        {
            "setup": """import numpy as np
reference_positions = np.array([[0,0,0],[1,0,0]], dtype=float); volumes = np.ones(2); horizon = 1.0
kernel_coefficients = np.array([1.0]); shape_gradients = np.zeros((2,2,3)); degraded_piola = np.zeros((2,2,3,3))
""",
            "call": "compute_internal_force_density(reference_positions, volumes, horizon, kernel_coefficients, shape_gradients, degraded_piola)",
            "gold_call": "_oracle_compute_internal_force_density(reference_positions, volumes, horizon, kernel_coefficients, shape_gradients, degraded_piola)",
        },
        {
            "setup": """import numpy as np
reference_positions = np.array([[0,0,0],[1,0,0],[9,9,9]], dtype=float); volumes = np.ones(3); horizon = 1.1
kernel_coefficients = np.array([1.0]); shape_gradients = np.zeros((3,3,3)); degraded_piola = np.zeros((3,3,3,3))
def run_model():
    try:
        compute_internal_force_density(reference_positions, volumes, horizon, kernel_coefficients, shape_gradients, degraded_piola)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_internal_force_density(reference_positions, volumes, horizon, kernel_coefficients, shape_gradients, degraded_piola)
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
