"""
Return the traction the interface carries for the given opening and state. The source splits the law at the threshold from the previous step and gives a different expression on each side; reproduce both, and note the tangential weighting used here is not the one used to form the scalar opening measure.

Below the threshold the regularisation replaces the unbounded stiffness; above it the law is the ordinary degrading response. The two expressions agree exactly at the threshold, which is a useful check.

Returns
-------
ndarray of shape (n_interfaces, 1 + n_tangential): the traction vector per interface.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def cohesive_traction(delta_n, delta_t, beta, damage, cohesive_strength, critical_opening, damage_threshold):
    """Return the traction the interface carries for the given opening and state. The source splits the law at the threshold from the previous step and gives a different expression on each side; reproduce both, and note the tangential weighting used here is not the one used to form the scalar opening measure.

    Returns
    -------
    ndarray of shape (n_interfaces, 1 + n_tangential): the traction vector per interface.

    Raises
    ------
    ValueError: if damage lies outside [0, 1], or cohesive_strength or critical_opening is not positive.
    """
    return np.zeros(1)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_cohesive_traction(delta_n, delta_t, beta, damage, cohesive_strength,
                              critical_opening, damage_threshold):
    dn = np.asarray(delta_n, dtype=float)
    dt = np.atleast_2d(np.asarray(delta_t, dtype=float))
    d = np.asarray(damage, dtype=float)
    if np.any(d < 0.0) or np.any(d > 1.0):
        raise ValueError("damage must lie in [0, 1]")
    if float(critical_opening) <= 0.0 or float(cohesive_strength) <= 0.0:
        raise ValueError("cohesive_strength and critical_opening must be positive")
    b = float(beta); sig = float(cohesive_strength); dc = float(critical_opening)
    dtil = float(damage_threshold)
    # The vector the traction points along: beta SQUARED multiplies delta_t here, so its
    # norm carries beta to the FOURTH power - deliberately not the effective opening of
    # step 09. Getting these two weightings the same way round is the trap.
    vec = np.concatenate([dn[:, None], (b * b) * dt], axis=1)
    vnorm = np.sqrt(np.sum(vec * vec, axis=1))
    d_eff = _oracle_cohesive_effective_opening(dn, dt, b)
    out = np.zeros_like(vec)
    safe_v = np.where(vnorm > 0.0, vnorm, 1.0)
    safe_d = np.where(d > 0.0, d, 1.0)
    # CONVENTION (paper, regularised law): below the damage threshold the traction is
    # CONSTANT in magnitude, sigma_c (1 - d), and only its DIRECTION varies. At or above
    # the threshold it is the secant law of eq. 20.
    below = (d < dtil)
    mag_below = sig * (1.0 - d)
    # t = k(d) * (delta_n n + beta^2 delta_t) with the Camacho-Ortiz secant stiffness
    # k(d) = ((1-d)/d)(sigma_c/delta_c). The effective opening cancels in the chain rule,
    # so there is NO 1/delta factor here - checking the units is what catches that.
    scale_above = ((1.0 - d) / safe_d) * (sig / dc)
    for i in range(vec.shape[0]):
        if vnorm[i] <= 0.0:
            continue
        if below[i]:
            out[i] = mag_below[i] * vec[i] / safe_v[i]
        else:
            out[i] = scale_above[i] * vec[i]
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "cohesive_traction(np.array([3e-4]), np.array([[2e-5]]), 1.4, np.array([0.5]), 6.0e8, 6e-4, 0.006)",
         "gold_call": "_oracle_cohesive_traction(np.array([3e-4]), np.array([[2e-5]]), 1.4, np.array([0.5]), 6.0e8, 6e-4, 0.006)"},   # normal
        {"setup": "import numpy as np",
         "call": "cohesive_traction(np.array([0.0]), np.array([[0.0]]), 1.4, np.array([0.0]), 6.0e8, 6e-4, 0.006)",
         "gold_call": "_oracle_cohesive_traction(np.array([0.0]), np.array([[0.0]]), 1.4, np.array([0.0]), 6.0e8, 6e-4, 0.006)"},   # boundary
        {"setup": "import numpy as np",
         "call": "cohesive_traction(np.array([5.9e-4]), np.array([[2e-5]]), 1.4, np.array([0.98]), 6.0e8, 6e-4, 0.006)",
         "gold_call": "_oracle_cohesive_traction(np.array([5.9e-4]), np.array([[2e-5]]), 1.4, np.array([0.98]), 6.0e8, 6e-4, 0.006)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        cohesive_traction(np.array([1e-4]), np.array([[1e-5]]), 1.4, np.array([1.5]), 6.0e8, 6e-4, 0.006)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_cohesive_traction(np.array([1e-4]), np.array([[1e-5]]), 1.4, np.array([1.5]), 6.0e8, 6e-4, 0.006)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
