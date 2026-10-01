"""
Assemble the phase-weighted first-order moment matrix at every material point.

For the neighbor set $H_k=\{n:0<\lVert\Delta X_{kn}\rVert\leq\delta\}$,

the first-order moment is



$$

M_k=\sum_{n\in H_k}\omega(r_{kn})h(s^n_{kn})

\Delta X_{kn}\otimes\Delta X_{kn}V_n.

$$



The kinematic degradation is delayed relative to the current damage update:

$h(s)=1$ for $s\leq s_c$ and

$h(s)=((1-s)/(1-s_c))^2$ above $s_c$. Positive-definite moments are required

to reproduce affine fields in three dimensions.

Returns
-------
A float array of shape (N, 3, 3) containing positive-definite moments in mm^5.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_kinematic_moments(
    reference_positions: np.ndarray,
    volumes: np.ndarray,
    previous_phase: np.ndarray,
    horizon: float,
    kernel_coefficients: np.ndarray,
    kinematic_threshold: float,
) -> np.ndarray:
    r"""Build all phase-weighted moment matrices $M_k$.

    Raises ``ValueError`` unless positions have shape $(N,3)$ with $N\geq4$,
    volumes have shape $(N,)$ and are positive, ``previous_phase`` is a finite
    symmetric $(N,N)$ array in $[0,1]$ with zero diagonal, ``horizon`` is
    positive and finite, the coefficient array is finite and one-dimensional,
    $0\leq s_c<1$, and every assembled moment is positive definite.

    Parameters
    ----------
    reference_positions : np.ndarray
        Reference coordinates of shape $(N,3)$ in mm.
    volumes : np.ndarray
        Positive nodal volumes of shape $(N,)$ in mm$^3$.
    previous_phase : np.ndarray
        Symmetric old-time phase field of shape $(N,N)$.
    horizon : float
        Neighborhood radius $\delta$ in mm, inclusive at its boundary.
    kernel_coefficients : np.ndarray
        Ascending coefficients of $\omega(r/\delta)$.
    kinematic_threshold : float
        Threshold $s_c$ separating intact and degraded kinematics.

    Returns
    -------
    np.ndarray
        Positive-definite moment matrices of shape $(N,3,3)$ in mm$^5$.
    """
    return NotImplemented

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _kinematic_factor(phase, threshold):
    factor = np.ones_like(phase, dtype=float)
    mask = phase > threshold
    factor[mask] = ((1.0 - phase[mask]) / (1.0 - threshold)) ** 2
    return factor
def _kernel_value(distance, horizon, coefficients):
    return np.polynomial.polynomial.polyval(distance / horizon, coefficients)
def _oracle_build_kinematic_moments(
    reference_positions: np.ndarray,
    volumes: np.ndarray,
    previous_phase: np.ndarray,
    horizon: float,
    kernel_coefficients: np.ndarray,
    kinematic_threshold: float,
) -> np.ndarray:
    """Reference moment assembly using the old-time phase field."""
    positions = np.asarray(reference_positions, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    phase = np.asarray(previous_phase, dtype=float)
    coefficients = np.asarray(kernel_coefficients, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 4:
        raise ValueError(
            "reference_positions must have shape (N, 3) with N at least four"
        )
    point_count = positions.shape[0]
    if volumes.shape != (point_count,) or np.any(volumes <= 0.0):
        raise ValueError("volumes must have shape (N,) and be strictly positive")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(volumes)):
        raise ValueError("positions and volumes must be finite")
    if phase.shape != (point_count, point_count) or not np.all(np.isfinite(phase)):
        raise ValueError("previous_phase must be a finite (N, N) array")
    if np.any(phase < 0.0) or np.any(phase > 1.0):
        raise ValueError("previous_phase must lie in [0, 1]")
    if not np.allclose(phase, phase.T, rtol=0.0, atol=1e-12):
        raise ValueError("previous_phase must be symmetric")
    if not np.allclose(np.diag(phase), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("previous_phase must have a zero diagonal")
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
    factors = _kinematic_factor(phase, float(kinematic_threshold))
    moments = np.zeros((point_count, 3, 3), dtype=float)
    for k in range(point_count):
        for n in range(point_count):
            difference = positions[n] - positions[k]
            distance = float(np.linalg.norm(difference))
            if 0.0 < distance <= horizon:
                weight = _kernel_value(distance, horizon, coefficients)
                if weight < -1e-12:
                    raise ValueError("the kernel is negative on an active bond")
                moments[k] += (
                    weight
                    * factors[k, n]
                    * np.outer(difference, difference)
                    * volumes[n]
                )
        eigenvalues = np.linalg.eigvalsh(moments[k])
        if eigenvalues[0] <= 100.0 * np.finfo(float).eps * max(1.0, eigenvalues[-1]):
            raise ValueError("each moment matrix must be positive definite")
    return moments

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return degraded-cube, intact-tetrahedron, and singular-neighborhood cases."""
    return [
        {
            "setup": """import numpy as np
reference_positions = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]], dtype=float)
volumes = np.array([1.00,0.95,1.05,1.10,0.90,1.08,0.98,1.02])
previous_phase = np.zeros((8,8)); previous_phase[0,7] = previous_phase[7,0] = 0.9
horizon = 2.0
kernel_coefficients = np.array([1.0,-2.0,1.0])
kinematic_threshold = 0.8
""",
            "call": "build_kinematic_moments(reference_positions, volumes, previous_phase, horizon, kernel_coefficients, kinematic_threshold)",
            "gold_call": "_oracle_build_kinematic_moments(reference_positions, volumes, previous_phase, horizon, kernel_coefficients, kinematic_threshold)",
        },
        {
            "setup": """import numpy as np
reference_positions = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
volumes = np.ones(4)
previous_phase = np.zeros((4,4))
horizon = np.sqrt(2.0)
kernel_coefficients = np.array([1.0])
kinematic_threshold = 0.0
""",
            "call": "build_kinematic_moments(reference_positions, volumes, previous_phase, horizon, kernel_coefficients, kinematic_threshold)",
            "gold_call": "_oracle_build_kinematic_moments(reference_positions, volumes, previous_phase, horizon, kernel_coefficients, kinematic_threshold)",
        },
        {
            "setup": """import numpy as np
reference_positions = np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]], dtype=float)
volumes = np.ones(4)
previous_phase = np.zeros((4,4))
horizon = 3.0
kernel_coefficients = np.array([1.0])
kinematic_threshold = 0.8
def run_model():
    try:
        build_kinematic_moments(reference_positions, volumes, previous_phase, horizon, kernel_coefficients, kinematic_threshold)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_kinematic_moments(reference_positions, volumes, previous_phase, horizon, kernel_coefficients, kinematic_threshold)
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
