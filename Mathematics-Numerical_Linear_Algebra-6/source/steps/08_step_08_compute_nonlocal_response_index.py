"""
Run the complete nonlocal damage-and-force pipeline and return one response index.

The orchestrator composes kernel normalization, the old-phase kinematic

moments, directed shape gradients, point deformation gradients, corrected

bond constitutive states, the irreversible phase update, and internal forces.

The dimensionless endpoint is the horizon-scaled volume-weighted RMS force

density



$$

R=\frac{\delta}{E_Y}

\sqrt{\frac{\sum_kV_k\lVert B_k^{\mathrm{int}}\rVert_2^2}

{\sum_kV_k}}.

$$



Kinematics use the phase implied by the previous history, whereas energetic

stress degradation uses the newly updated phase. Reversing that time ordering

defines a different numerical method.

Returns
-------
One finite dimensionless float equal to the horizon-scaled weighted RMS force density.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_nonlocal_response_index(
    reference_positions: np.ndarray,
    current_positions: np.ndarray,
    volumes: np.ndarray,
    previous_history: np.ndarray,
    horizon: float,
    kernel_coefficients: np.ndarray,
    youngs_modulus: float,
    poisson_ratio: float,
    fracture_energy: float,
    kinematic_threshold: float,
) -> float:
    r"""Return the deterministic dimensionless response index $R$.

    Raises ``ValueError`` when any earlier stage rejects its inputs, when
    ``current_positions`` and ``reference_positions`` do not share shape
    $(N,3)$, or when the final weighted RMS or response index is nonfinite.
    The old phase is computed from ``previous_history`` and $Y_c$ before any
    current-step damage update; no randomness is used.

    Parameters
    ----------
    reference_positions, current_positions : np.ndarray
        Reference and current coordinates of shape $(N,3)$ in mm.
    volumes : np.ndarray
        Positive nodal volumes of shape $(N,)$ in mm$^3$.
    previous_history : np.ndarray
        Symmetric previous bond history of shape $(N,N)$ in MPa.
    horizon : float
        Inclusive neighborhood radius $\delta$ in mm.
    kernel_coefficients : np.ndarray
        Ascending coefficients of $\omega(r/\delta)$.
    youngs_modulus : float
        Young's modulus $E_Y$ in MPa.
    poisson_ratio : float
        Poisson ratio $\nu$ in $(-1,0.5)$.
    fracture_energy : float
        Critical energy release rate $G_c$ in N/mm.
    kinematic_threshold : float
        Kinematic degradation threshold $s_c$ in $[0,1)$.

    Returns
    -------
    float
        Finite dimensionless response index $R$.
    """
    return NotImplemented

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_nonlocal_response_index(
    reference_positions: np.ndarray,
    current_positions: np.ndarray,
    volumes: np.ndarray,
    previous_history: np.ndarray,
    horizon: float,
    kernel_coefficients: np.ndarray,
    youngs_modulus: float,
    poisson_ratio: float,
    fracture_energy: float,
    kinematic_threshold: float,
) -> float:
    """Reference end-to-end composition of the seven preceding oracles."""
    reference = np.asarray(reference_positions, dtype=float)
    current = np.asarray(current_positions, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    history = np.asarray(previous_history, dtype=float)
    if current.shape != reference.shape:
        raise ValueError(
            "current_positions and reference_positions must have equal shapes"
        )
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be positive and finite")
    if not np.isfinite(fracture_energy) or fracture_energy <= 0.0:
        raise ValueError("fracture_energy must be positive and finite")
    kernel_normalization = globals()["_oracle_compute_kernel_normalization"](
        kernel_coefficients
    )
    critical = fracture_energy / (2.0 * kernel_normalization * horizon)
    _, previous_phase = globals()["_oracle_update_bond_phase_history"](
        np.zeros_like(history), history, critical
    )
    moments = globals()["_oracle_build_kinematic_moments"](
        reference,
        volumes,
        previous_phase,
        horizon,
        kernel_coefficients,
        kinematic_threshold,
    )
    shape_gradients = globals()["_oracle_compute_bond_shape_gradients"](
        reference,
        volumes,
        previous_phase,
        moments,
        horizon,
        kernel_coefficients,
        kinematic_threshold,
    )
    point_gradients = globals()["_oracle_compute_point_deformation_gradients"](
        current - reference, shape_gradients
    )
    degraded_piola, _, _ = globals()["_oracle_update_bond_constitutive_state"](
        reference,
        current,
        point_gradients,
        history,
        kernel_normalization,
        horizon,
        youngs_modulus,
        poisson_ratio,
        fracture_energy,
    )
    internal_force = globals()["_oracle_compute_internal_force_density"](
        reference,
        volumes,
        horizon,
        kernel_coefficients,
        shape_gradients,
        degraded_piola,
    )
    weighted_mean_square = np.sum(volumes * np.sum(internal_force**2, axis=1)) / np.sum(
        volumes
    )
    result = float(horizon / youngs_modulus * np.sqrt(weighted_mean_square))
    if not np.isfinite(result):
        raise ValueError("the response index must be finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return baseline, stronger-loading, and reduced-history pipeline cases."""
    return [
        {
            "setup": """import numpy as np
reference_positions = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]], dtype=float)
volumes = np.array([1.00,0.95,1.05,1.10,0.90,1.08,0.98,1.02])
displacements = np.column_stack((0.004*reference_positions[:,0] + 0.002*reference_positions[:,1]*reference_positions[:,2], -0.001*reference_positions[:,1] + 0.0015*reference_positions[:,0]*reference_positions[:,2], 0.002*reference_positions[:,2] - 0.001*reference_positions[:,0]*reference_positions[:,1]))
displacements[7] += np.array([0.0012,-0.0007,0.0009]); current_positions = reference_positions + displacements
previous_history = np.zeros((8,8))
for i in range(8):
    for j in range(i+1,8):
        previous_history[i,j] = previous_history[j,i] = 0.015 + 0.01*((i+j)%3)
for i,j,value in [(0,7,0.5),(1,6,0.3),(2,5,0.12),(3,4,0.08)]:
    previous_history[i,j] = previous_history[j,i] = value
horizon = 2.0; kernel_coefficients = np.array([1.0,-2.0,1.0]); youngs_modulus = 32000.0
poisson_ratio = 0.25; fracture_energy = 0.1; kinematic_threshold = 0.8
""",
            "call": "compute_nonlocal_response_index(reference_positions, current_positions, volumes, previous_history, horizon, kernel_coefficients, youngs_modulus, poisson_ratio, fracture_energy, kinematic_threshold)",
            "gold_call": "_oracle_compute_nonlocal_response_index(reference_positions, current_positions, volumes, previous_history, horizon, kernel_coefficients, youngs_modulus, poisson_ratio, fracture_energy, kinematic_threshold)",
        },
        {
            "setup": """import numpy as np
reference_positions = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]], dtype=float)
volumes = np.array([1.00,0.95,1.05,1.10,0.90,1.08,0.98,1.02])
displacements = np.column_stack((0.004*reference_positions[:,0] + 0.002*reference_positions[:,1]*reference_positions[:,2], -0.001*reference_positions[:,1] + 0.0015*reference_positions[:,0]*reference_positions[:,2], 0.002*reference_positions[:,2] - 0.001*reference_positions[:,0]*reference_positions[:,1]))
displacements[7] += np.array([0.0012,-0.0007,0.0009]); current_positions = reference_positions + 1.35*displacements
previous_history = np.zeros((8,8))
for i in range(8):
    for j in range(i+1,8):
        previous_history[i,j] = previous_history[j,i] = 0.015 + 0.01*((i+j)%3)
for i,j,value in [(0,7,0.5),(1,6,0.3),(2,5,0.12),(3,4,0.08)]:
    previous_history[i,j] = previous_history[j,i] = value
horizon = 2.0; kernel_coefficients = np.array([1.0,-2.0,1.0]); youngs_modulus = 32000.0
poisson_ratio = 0.25; fracture_energy = 0.1; kinematic_threshold = 0.8
""",
            "call": "compute_nonlocal_response_index(reference_positions, current_positions, volumes, previous_history, horizon, kernel_coefficients, youngs_modulus, poisson_ratio, fracture_energy, kinematic_threshold)",
            "gold_call": "_oracle_compute_nonlocal_response_index(reference_positions, current_positions, volumes, previous_history, horizon, kernel_coefficients, youngs_modulus, poisson_ratio, fracture_energy, kinematic_threshold)",
        },
        {
            "setup": """import numpy as np
reference_positions = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]], dtype=float)
volumes = np.array([1.00,0.95,1.05,1.10,0.90,1.08,0.98,1.02])
displacements = np.column_stack((0.004*reference_positions[:,0] + 0.002*reference_positions[:,1]*reference_positions[:,2], -0.001*reference_positions[:,1] + 0.0015*reference_positions[:,0]*reference_positions[:,2], 0.002*reference_positions[:,2] - 0.001*reference_positions[:,0]*reference_positions[:,1]))
displacements[7] += np.array([0.0012,-0.0007,0.0009]); current_positions = reference_positions + displacements
previous_history = np.zeros((8,8))
for i in range(8):
    for j in range(i+1,8):
        previous_history[i,j] = previous_history[j,i] = 0.25*(0.015 + 0.01*((i+j)%3))
for i,j,value in [(0,7,0.125),(1,6,0.075),(2,5,0.03),(3,4,0.02)]:
    previous_history[i,j] = previous_history[j,i] = value
horizon = 2.0; kernel_coefficients = np.array([1.0,-2.0,1.0]); youngs_modulus = 32000.0
poisson_ratio = 0.25; fracture_energy = 0.1; kinematic_threshold = 0.8
""",
            "call": "compute_nonlocal_response_index(reference_positions, current_positions, volumes, previous_history, horizon, kernel_coefficients, youngs_modulus, poisson_ratio, fracture_energy, kinematic_threshold)",
            "gold_call": "_oracle_compute_nonlocal_response_index(reference_positions, current_positions, volumes, previous_history, horizon, kernel_coefficients, youngs_modulus, poisson_ratio, fracture_energy, kinematic_threshold)",
        },
    ]
