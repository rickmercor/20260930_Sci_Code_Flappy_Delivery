"""
Compute the seven ordered coefficients in one endpoint-conditioned double-Heston log-price update.

The almost-exact construction replaces each correlated price shock by variance endpoints, leaving only orthogonal Gaussian shocks.

Returns
-------
np.ndarray, [constant, current_v1, current_v2, next_v1, next_v2, residual_v1, residual_v2] as a float array with shape (7,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_double_heston_aes_coefficients(
    dt: float,
    rate: float,
    kappa: "np.ndarray",
    theta: "np.ndarray",
    gamma: "np.ndarray",
    rho: "np.ndarray",
) -> "np.ndarray":
    r"""Return the ordered coefficient vector for one time interval.

    Parameters
    ----------
    dt, rate : float
        Positive interval length and finite risk-free rate.
    kappa, theta, gamma, rho : numpy.ndarray
        Length-two factor parameters; scales are positive and correlations
        lie in [-1, 1].

    Returns
    -------
    coefficients : numpy.ndarray
        ``[constant, current_v1, current_v2, next_v1, next_v2,
        residual_v1, residual_v2]`` for the log update.

    Raises
    ------
    ValueError
        If ``dt`` is not positive, inputs are nonfinite or not length two,
        CIR scales are not positive, or a correlation is outside [-1, 1].
    """
    return coefficients

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_double_heston_aes_coefficients(
    dt: float, rate: float, kappa: "np.ndarray", theta: "np.ndarray",
    gamma: "np.ndarray", rho: "np.ndarray",
) -> "np.ndarray":
    import numpy as np
    arrays = [np.asarray(x, dtype=float) for x in (kappa, theta, gamma, rho)]
    if not np.isfinite(dt) or dt <= 0.0 or not np.isfinite(rate):
        raise ValueError("dt must be positive and rate must be finite")
    if any(x.shape != (2,) or not np.all(np.isfinite(x)) for x in arrays):
        raise ValueError("factor parameters must be finite length-two arrays")
    kappa, theta, gamma, rho = arrays
    if np.any(kappa <= 0.0) or np.any(theta <= 0.0) or np.any(gamma <= 0.0):
        raise ValueError("CIR scale parameters must be positive")
    if np.any(np.abs(rho) > 1.0):
        raise ValueError("correlations must lie in [-1, 1]")
    current = (rho * kappa / gamma - 0.5) * dt - rho / gamma
    endpoint = rho / gamma
    residual = (1.0 - rho ** 2) * dt
    drift = (rate - np.sum(rho * kappa * theta / gamma)) * dt
    return np.array([drift, *current, *endpoint, *residual], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and asymmetric specifications."""
    return [
        {"setup": "import numpy as np\nk=np.array([.9,1.2]); t=np.array([.1,.15]); g=np.array([.1,.2]); q=np.array([-.5,-.5]); w=np.arange(1.,8.)",
         "call": "float(np.dot(compute_double_heston_aes_coefficients(.25/6,.03,k,t,g,q),w))",
         "gold_call": "float(np.dot(_oracle_compute_double_heston_aes_coefficients(.25/6,.03,k,t,g,q),w))"},
        {"setup": "import numpy as np\nk=np.array([1.,2.]); t=np.array([.2,.3]); g=np.array([.4,.5]); q=np.zeros(2); w=np.arange(1,8)",
         "call": "float(np.dot(compute_double_heston_aes_coefficients(.5,0.,k,t,g,q),w))",
         "gold_call": "float(np.dot(_oracle_compute_double_heston_aes_coefficients(.5,0.,k,t,g,q),w))"},
        {"setup": "import numpy as np\nk=np.array([.7,1.3]); t=np.array([.08,.12]); g=np.array([.3,.4]); q=np.array([-1.,1.]); w=np.arange(1.,8.)",
         "call": "float(np.dot(compute_double_heston_aes_coefficients(.2,-.01,k,t,g,q),w))",
         "gold_call": "float(np.dot(_oracle_compute_double_heston_aes_coefficients(.2,-.01,k,t,g,q),w))"},
        {"setup": "import numpy as np\nk=np.array([.7,3.]); t=np.array([.04,.8]); g=np.array([.3,1.1]); q=np.array([-.9,.65]); w=np.array([1.,-2.,3.,-4.,5.,-6.,7.])",
         "call": "float(np.dot(compute_double_heston_aes_coefficients(.07,.015,k,t,g,q),w))",
         "gold_call": "float(np.dot(_oracle_compute_double_heston_aes_coefficients(.07,.015,k,t,g,q),w))"},
    ]
