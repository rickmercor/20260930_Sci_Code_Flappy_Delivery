"""
Compute the ordered parameters of one finite-interval CIR transition.

A square-root variance factor has a scaled noncentral-chi-square conditional law over a finite interval.

Returns
-------
np.ndarray, [scale, degrees of freedom, noncentrality multiplier] as a float array with shape (3,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_cir_transition_parameters(
    kappa: float, theta: float, gamma: float, dt: float
) -> "np.ndarray":
    r"""Return the scale, degrees of freedom, and noncentrality multiplier.

    Parameters
    ----------
    kappa, theta, gamma : float
        Positive mean reversion, long-run variance, and vol-of-vol.
    dt : float
        Positive transition interval.

    Returns
    -------
    parameters : numpy.ndarray
        Three finite transition parameters in source-defined order.

    Raises
    ------
    ValueError
        If any input is nonfinite or nonpositive.
    """
    return parameters

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_cir_transition_parameters(
    kappa: float, theta: float, gamma: float, dt: float
) -> "np.ndarray":
    values = np.asarray([kappa, theta, gamma, dt], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("all CIR inputs must be finite and positive")
    decay_loss = -np.expm1(-kappa * dt)
    decay = np.exp(-kappa * dt)
    scale = gamma ** 2 * decay_loss / (4.0 * kappa)
    degrees = 4.0 * kappa * theta / gamma ** 2
    multiplier = 4.0 * kappa * decay / (gamma ** 2 * decay_loss)
    return np.array([scale, degrees, multiplier], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and edge test specifications."""
    return [
        {
            "setup": """import numpy as np
w=np.array([1.0,0.01,1e-4])
""",
            "call": "float(np.dot(compute_cir_transition_parameters(.9,.1,.1,.25/6),w))",
            "gold_call": "float(np.dot(_oracle_compute_cir_transition_parameters(.9,.1,.1,.25/6),w))",
        },
        {
            "setup": """import numpy as np
w=np.array([10.0,0.1,1e-6])
""",
            "call": "float(np.dot(compute_cir_transition_parameters(2.0,.3,.5,1e-6),w))",
            "gold_call": "float(np.dot(_oracle_compute_cir_transition_parameters(2.0,.3,.5,1e-6),w))",
        },
        {
            "setup": """import numpy as np
w=np.array([100.0,0.001,1e-8])
""",
            "call": "float(np.dot(compute_cir_transition_parameters(.05,.001,2.0,3.0),w))",
            "gold_call": "float(np.dot(_oracle_compute_cir_transition_parameters(.05,.001,2.0,3.0),w))",
        },
    ]
