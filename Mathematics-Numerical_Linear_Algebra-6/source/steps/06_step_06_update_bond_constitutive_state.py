"""
Apply the bond-associated correction, elastic law, tensile driving force, and damage update.

The endpoint-average point gradient is corrected so that each bond deformation

is reproduced exactly:



$$

\widetilde F_{kn}=\frac{\bar F_k+\bar F_n}{2}

+\frac{(\Delta x_{kn}-\bar F^{\mathrm{av}}_{kn}\Delta X_{kn})

\otimes\Delta X_{kn}}{\lVert\Delta X_{kn}\rVert^2}.

$$



Saint-Venant--Kirchhoff elasticity gives

$E_G=(\widetilde F^T\widetilde F-I)/2$,

$S=\lambda\operatorname{tr}(E_G)I+2\mu E_G$, and $P_0=\widetilde F S$.

The trial force is $Y=\langle\sigma_1\rangle_+^2/(2E_Y)$, with $\sigma_1$

the maximum eigenvalue of $P_0\widetilde F^T/\det\widetilde F$; after the

irreversible update, energetic degradation uses $\widetilde P=(1-s)^2P_0$.

Returns
-------
Three float arrays: degraded Piola stress (N, N, 3, 3), history (N, N), and phase (N, N).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def update_bond_constitutive_state(
    reference_positions: np.ndarray,
    current_positions: np.ndarray,
    point_gradients: np.ndarray,
    previous_history: np.ndarray,
    kernel_normalization: float,
    horizon: float,
    youngs_modulus: float,
    poisson_ratio: float,
    fracture_energy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Update stress, history, and phase for every active bond.

    Raises ``ValueError`` unless both position arrays have the same finite
    shape $(N,3)$ with $N\geq2$; point gradients have finite shape $(N,3,3)$;
    previous history is finite, nonnegative, symmetric, and zero-diagonal with
    shape $(N,N)$; $c_0$, $\delta$, $E_Y$, and $G_c$ are positive and finite;
    $-1<\nu<0.5$; and every active corrected gradient has positive determinant.

    Parameters
    ----------
    reference_positions, current_positions : np.ndarray
        Reference and current coordinates of shape $(N,3)$ in mm.
    point_gradients : np.ndarray
        Dimensionless point gradients of shape $(N,3,3)$.
    previous_history : np.ndarray
        Previous symmetric bond history of shape $(N,N)$ in MPa.
    kernel_normalization : float
        Positive dimensionless constant $c_0$.
    horizon : float
        Inclusive bond horizon $\delta$ in mm.
    youngs_modulus : float
        Young's modulus $E_Y$ in MPa.
    poisson_ratio : float
        Poisson ratio $\nu$.
    fracture_energy : float
        Critical energy release rate $G_c$ in N/mm.

    Returns
    -------
    degraded_piola : np.ndarray
        Bond stresses of shape $(N,N,3,3)$ in MPa, zero off active bonds.
    history_new : np.ndarray
        Updated symmetric history of shape $(N,N)$ in MPa.
    phase_new : np.ndarray
        Updated symmetric phase field of shape $(N,N)$ in $[0,1]$.
    """
    return NotImplemented

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_update_bond_constitutive_state(
    reference_positions: np.ndarray,
    current_positions: np.ndarray,
    point_gradients: np.ndarray,
    previous_history: np.ndarray,
    kernel_normalization: float,
    horizon: float,
    youngs_modulus: float,
    poisson_ratio: float,
    fracture_energy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference corrected-gradient and damage update."""
    reference = np.asarray(reference_positions, dtype=float)
    current = np.asarray(current_positions, dtype=float)
    gradients = np.asarray(point_gradients, dtype=float)
    history = np.asarray(previous_history, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 3 or reference.shape[0] < 2:
        raise ValueError(
            "reference_positions must have shape (N, 3) with N at least two"
        )
    point_count = reference.shape[0]
    if current.shape != reference.shape or gradients.shape != (point_count, 3, 3):
        raise ValueError("current positions or point gradients have invalid shapes")
    if history.shape != (point_count, point_count):
        raise ValueError("previous_history must have shape (N, N)")
    if (
        not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(current))
        or not np.all(np.isfinite(gradients))
    ):
        raise ValueError("position and gradient inputs must be finite")
    if not np.all(np.isfinite(history)) or np.any(history < 0.0):
        raise ValueError("previous_history must be finite and nonnegative")
    if not np.allclose(history, history.T, rtol=0.0, atol=1e-12) or not np.allclose(
        np.diag(history), 0.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("previous_history must be symmetric with zero diagonal")
    positive_parameters = (
        kernel_normalization,
        horizon,
        youngs_modulus,
        fracture_energy,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in positive_parameters):
        raise ValueError(
            "c0, horizon, Young's modulus, and fracture energy must be positive and finite"
        )
    if not np.isfinite(poisson_ratio) or not -1.0 < poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must lie in (-1, 0.5)")
    lame_lambda = (
        youngs_modulus
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    critical = fracture_energy / (2.0 * kernel_normalization * horizon)
    trial = np.zeros((point_count, point_count), dtype=float)
    piola_undamaged = np.zeros((point_count, point_count, 3, 3), dtype=float)
    identity = np.eye(3)
    for k in range(point_count):
        for n in range(k + 1, point_count):
            reference_bond = reference[n] - reference[k]
            distance = float(np.linalg.norm(reference_bond))
            if 0.0 < distance <= horizon:
                current_bond = current[n] - current[k]
                average = 0.5 * (gradients[k] + gradients[n])
                correction = (
                    np.outer(current_bond - average @ reference_bond, reference_bond)
                    / distance**2
                )
                corrected = average + correction
                determinant = float(np.linalg.det(corrected))
                if not np.isfinite(determinant) or determinant <= 0.0:
                    raise ValueError(
                        "every active corrected gradient must have positive determinant"
                    )
                green = 0.5 * (corrected.T @ corrected - identity)
                second_piola = (
                    lame_lambda * np.trace(green) * identity
                    + 2.0 * shear_modulus * green
                )
                first_piola = corrected @ second_piola
                cauchy = first_piola @ corrected.T / determinant
                maximum_principal = float(
                    np.linalg.eigvalsh(0.5 * (cauchy + cauchy.T))[-1]
                )
                driving_force = max(0.0, maximum_principal) ** 2 / (
                    2.0 * youngs_modulus
                )
                trial[k, n] = trial[n, k] = driving_force
                piola_undamaged[k, n] = piola_undamaged[n, k] = first_piola
    history_updater = globals()["_oracle_update_bond_phase_history"]
    history_new, phase_new = history_updater(trial, history, critical)
    degraded = (1.0 - phase_new[..., None, None]) ** 2 * piola_undamaged
    return degraded, history_new, phase_new

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return mixed-history, rigid-motion, and inverted-bond cases."""
    return [
        {
            "setup": """import numpy as np
reference_positions = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
current_positions = reference_positions @ np.array([[1.01,0.01,0.0],[0.0,0.995,0.0],[0.0,0.0,1.002]]).T
point_gradients = np.repeat(np.eye(3)[None,:,:],4,axis=0); point_gradients[:,0,0] = 1.01
previous_history = np.zeros((4,4)); previous_history[0,1] = previous_history[1,0] = 0.2
kernel_normalization = 0.1; horizon = 2.0; youngs_modulus = 32000.0; poisson_ratio = 0.25; fracture_energy = 0.1
""",
            "call": "update_bond_constitutive_state(reference_positions, current_positions, point_gradients, previous_history, kernel_normalization, horizon, youngs_modulus, poisson_ratio, fracture_energy)",
            "gold_call": "_oracle_update_bond_constitutive_state(reference_positions, current_positions, point_gradients, previous_history, kernel_normalization, horizon, youngs_modulus, poisson_ratio, fracture_energy)",
        },
        {
            "setup": """import numpy as np
reference_positions = np.array([[0,0,0],[1,0,0]], dtype=float); current_positions = reference_positions + np.array([2.0,-1.0,0.5])
point_gradients = np.repeat(np.eye(3)[None,:,:],2,axis=0); previous_history = np.zeros((2,2))
kernel_normalization = 0.1; horizon = 1.0; youngs_modulus = 32000.0; poisson_ratio = 0.25; fracture_energy = 0.1
""",
            "call": "update_bond_constitutive_state(reference_positions, current_positions, point_gradients, previous_history, kernel_normalization, horizon, youngs_modulus, poisson_ratio, fracture_energy)",
            "gold_call": "_oracle_update_bond_constitutive_state(reference_positions, current_positions, point_gradients, previous_history, kernel_normalization, horizon, youngs_modulus, poisson_ratio, fracture_energy)",
        },
        {
            "setup": """import numpy as np
reference_positions = np.array([[0,0,0],[1,0,0]], dtype=float); current_positions = np.array([[0,0,0],[-1,0,0]], dtype=float)
point_gradients = np.repeat((-np.eye(3))[None,:,:],2,axis=0); previous_history = np.zeros((2,2))
kernel_normalization = 0.1; horizon = 1.0; youngs_modulus = 32000.0; poisson_ratio = 0.25; fracture_energy = 0.1
def run_model():
    try:
        update_bond_constitutive_state(reference_positions, current_positions, point_gradients, previous_history, kernel_normalization, horizon, youngs_modulus, poisson_ratio, fracture_energy)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_update_bond_constitutive_state(reference_positions, current_positions, point_gradients, previous_history, kernel_normalization, horizon, youngs_modulus, poisson_ratio, fracture_energy)
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
