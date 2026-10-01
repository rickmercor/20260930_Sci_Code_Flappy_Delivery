"""
Convert overlap measure and its derivative into contact energy and force.



For $w(V_c)=kV_c^m$, the generalized contact force is the negative state

gradient of this energy. Return the energy, force, and consistent force

Jacobian assembled from the overlap gradient and Hessian. The no-contact state

is defined to have zero energy, force, and Jacobian. Evaluate every power-law

coefficient without overflowing an avoidable intermediate.

The scalar contact energy is



$$

E(q)=k\,V(q)^m,

\qquad k>0,\quad m\geq 1,

$$



where $V$ is the overlap area. Stiffness and exponent remain fixed

when differentiating with respect to the generalized state $q$.



The generalized force is the negative state gradient of $E$. The consistent

force Jacobian is the derivative of that force with respect to $q$. Its sign

is therefore the sign of the force derivative, not the energy Hessian.

Compute these quantities using the supplied overlap gradient and Hessian.



The signature specifies how energy, force, and force Jacobian share the

symmetric output array. At exactly zero overlap, the entire output is defined

as zero, including the linear-exponent case. For positive overlap, use the

energy law above. Preserve finite results when evaluating an isolated raw

power would overflow. The signature also defines validation and input layout.

Returns
-------
symmetric float64 np.ndarray of shape (p + 1, p + 1), with energy at [0, 0], force in row/column 0, and the force Jacobian in the lower-right block
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_contact_energy_force(
    overlap_area: float,
    overlap_sensitivities: np.ndarray,
    stiffness: float,
    exponent: float = 1.0,
) -> np.ndarray:
    """Return overlap energy, generalized force, and force Jacobian.

    Parameters
    ----------
    overlap_area : float
        Nonnegative contact area in square metres.
    overlap_sensitivities : np.ndarray
        Finite array with shape ``(p + 1, p)``. Row zero is the overlap
        gradient and rows 1 through ``p`` are its symmetric Hessian.
    stiffness : float
        Strictly positive energy coefficient in units compatible with the
        selected exponent.
    exponent : float, optional
        Finite exponent at least one.

    Returns
    -------
    np.ndarray
        Symmetric float64 array of shape ``(p + 1, p + 1)``. Entry ``[0, 0]``
        is the energy, row and column zero contain the generalized force, and
        the lower-right block is its symmetric Jacobian with respect to state.
        Finite mathematical results must not overflow merely because a raw
        power of ``overlap_area`` does so as an intermediate.

    Raises
    ------
    ValueError
        If any input is non-finite, the overlap Hessian is not symmetric,
        ``overlap_area`` is negative, ``stiffness`` is not positive, or
        ``exponent`` is below one.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_contact_energy_force(
    overlap_area: float,
    overlap_sensitivities: np.ndarray,
    stiffness: float,
    exponent: float = 1.0,
) -> np.ndarray:
    """Reference energy and negative-gradient force."""
    sensitivities = np.asarray(overlap_sensitivities, dtype=np.float64)
    values = np.asarray([overlap_area, stiffness, exponent], dtype=np.float64)
    if (
        sensitivities.ndim != 2
        or sensitivities.shape[1] < 1
        or sensitivities.shape[0] != sensitivities.shape[1] + 1
    ):
        raise ValueError("overlap_sensitivities must have shape (p + 1, p)")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(sensitivities)):
        raise ValueError("all inputs must be finite")
    gradient = sensitivities[0]
    hessian = sensitivities[1:]
    if not np.allclose(hessian, hessian.T, rtol=1e-10, atol=1e-12):
        raise ValueError("overlap Hessian must be symmetric")
    if overlap_area < 0.0:
        raise ValueError("overlap_area must be nonnegative")
    if stiffness <= 0.0 or exponent < 1.0:
        raise ValueError("stiffness must be positive and exponent must be at least one")
    if overlap_area == 0.0:
        return np.zeros((gradient.size + 1, gradient.size + 1), dtype=np.float64)
    log_area = np.log(overlap_area)
    log_energy = np.log(stiffness) + exponent * log_area
    log_max = np.log(np.finfo(np.float64).max)
    if log_energy > log_max:
        raise ValueError("energy and force must be finite")
    energy = float(np.exp(log_energy))
    log_first = (
        np.log(stiffness) + np.log(exponent) + (exponent - 1.0) * log_area
    )
    if log_first > log_max:
        raise ValueError("energy and force must be finite")
    first_factor = float(np.exp(log_first))
    if exponent == 1.0:
        second_factor = 0.0
    else:
        log_second = (
            np.log(stiffness)
            + np.log(exponent)
            + np.log(exponent - 1.0)
            + (exponent - 2.0) * log_area
        )
        if log_second > log_max:
            raise ValueError("force Jacobian must be finite")
        second_factor = float(np.exp(log_second))
    force = -first_factor * gradient
    force_jacobian = -(
        second_factor * np.outer(gradient, gradient) + first_factor * hessian
    )
    result = np.empty((gradient.size + 1, gradient.size + 1), dtype=np.float64)
    result[0, 0] = energy
    result[0, 1:] = force
    result[1:, 0] = force
    result[1:, 1:] = force_jacobian
    if not np.all(np.isfinite(result)):
        raise ValueError("energy and force must be finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return tangent, scaled-power, no-contact, and invalid cases."""
    return [
        {
            "setup": """
import numpy as np
overlap_area = 0.42
overlap_sensitivities = np.array([
    [-0.31, 0.14, -0.05],
    [0.20, -0.04, 0.03],
    [-0.04, -0.10, 0.02],
    [0.03, 0.02, 0.15],
])
stiffness = 480.0
exponent = 1.0
""",
            "call": "compute_contact_energy_force(overlap_area, overlap_sensitivities, stiffness, exponent)",
            "gold_call": "_oracle_compute_contact_energy_force(overlap_area, overlap_sensitivities, stiffness, exponent)",
        },
        {
            "setup": """
import numpy as np
overlap_area = 1.0e200
overlap_sensitivities = np.array([
    [1.0e100, -5.0e99],
    [2.0e-100, -1.0e-100],
    [-1.0e-100, 3.0e-100],
])
stiffness = 1.0e-300
exponent = 2.0
""",
            "call": "compute_contact_energy_force(overlap_area, overlap_sensitivities, stiffness, exponent)",
            "gold_call": "_oracle_compute_contact_energy_force(overlap_area, overlap_sensitivities, stiffness, exponent)",
        },
        {
            "setup": """
import numpy as np
overlap_area = 0.0
overlap_sensitivities = np.vstack([
    np.array([-4.0, 2.0, 0.0, 7.0]),
    np.diag([0.2, -0.1, 0.3, 0.4]),
])
stiffness = 700.0
exponent = 1.0
""",
            "call": "compute_contact_energy_force(overlap_area, overlap_sensitivities, stiffness, exponent)",
            "gold_call": "_oracle_compute_contact_energy_force(overlap_area, overlap_sensitivities, stiffness, exponent)",
        },
        {
            "setup": """
import numpy as np
overlap_area = -0.1
overlap_sensitivities = np.array([
    [0.2, -0.1], [0.3, 0.0], [0.0, -0.2]
])
stiffness = 10.0
exponent = 1.0
def run_model():
    try:
        compute_contact_energy_force(overlap_area, overlap_sensitivities, stiffness, exponent)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_contact_energy_force(overlap_area, overlap_sensitivities, stiffness, exponent)
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
