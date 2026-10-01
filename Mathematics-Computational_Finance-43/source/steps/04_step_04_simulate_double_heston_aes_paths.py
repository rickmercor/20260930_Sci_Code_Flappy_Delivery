"""
Simulate asset paths from two variance paths with the endpoint-conditioned log-price update.

Variance endpoints carry correlated shocks while orthogonal price integrals retain the scheme's left-endpoint approximation.

Returns
-------
np.ndarray, positive asset paths as a float array with the same shape as each variance-path input.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def simulate_double_heston_aes_paths(
    initial_spot: float, coefficients: "np.ndarray",
    variance_one: "np.ndarray",
    variance_two: "np.ndarray", seed: int,
) -> "np.ndarray":
    r"""Return asset paths implied by two supplied variance-path arrays.

    Parameters
    ----------
    initial_spot : float
        Finite positive initial asset price.
    coefficients : numpy.ndarray
        Seven coefficients from the endpoint-conditioned construction.
    variance_one, variance_two : numpy.ndarray
        Equal-shaped nonnegative variance paths including time zero.
    seed : int
        Seed for factor-one then factor-two normal vectors at each step.

    Returns
    -------
    spots : numpy.ndarray
        Positive asset paths with the same shape as each variance array.
    Raises
    ------
    ValueError
        If any state, coefficient, initial value, or seed is invalid.
    """
    return spots

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_simulate_double_heston_aes_paths(
    initial_spot: float, coefficients: "np.ndarray", variance_one: "np.ndarray",
    variance_two: "np.ndarray", seed: int,
) -> "np.ndarray":
    coeff = np.asarray(coefficients, dtype=float)
    v1 = np.asarray(variance_one, dtype=float); v2 = np.asarray(variance_two, dtype=float)
    if coeff.shape != (7,) or not np.all(np.isfinite(coeff)):
        raise ValueError("coefficients must be a finite length-seven array")
    if v1.ndim != 2 or v1.shape != v2.shape or v1.shape[0] < 2:
        raise ValueError("variance arrays must have one equal two-dimensional shape")
    if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v2)):
        raise ValueError("variance paths must be finite")
    if np.any(v1 < 0.0) or np.any(v2 < 0.0) or np.any(coeff[5:] < 0.0):
        raise ValueError("variance paths and residual coefficients must be nonnegative")
    if initial_spot <= 0.0 or not np.isfinite(initial_spot):
        raise ValueError("initial_spot must be finite and positive")
    if (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    log_spot = np.empty_like(v1); log_spot[0] = np.log(initial_spot)
    for step in range(v1.shape[0] - 1):
        z1 = rng.standard_normal(v1.shape[1]); z2 = rng.standard_normal(v1.shape[1])
        log_spot[step + 1] = (
            log_spot[step] + coeff[0] + coeff[1] * v1[step]
            + coeff[2] * v2[step] + coeff[3] * v1[step + 1]
            + coeff[4] * v2[step + 1] + np.sqrt(coeff[5] * v1[step]) * z1
            + np.sqrt(coeff[6] * v2[step]) * z2
        )
    return np.exp(log_spot)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and edge test specifications."""
    return [
        {"setup": "import numpy as np\nc=np.array([.01,.2,-.1,.3,-.2,.04,.03]); v1=np.full((4,6),.2); v2=np.full((4,6),.1)",
         "call": "float(np.sum(simulate_double_heston_aes_paths(50,c,v1,v2,13)))",
         "gold_call": "float(np.sum(_oracle_simulate_double_heston_aes_paths(50,c,v1,v2,13)))"},
        {"setup": "import numpy as np\nc=np.array([.02,.1,.1,-.1,-.1,0,0.]); v1=np.full((2,3),.2); v2=np.full((2,3),.3)",
         "call": "float(np.sum(simulate_double_heston_aes_paths(1,c,v1,v2,0)))",
         "gold_call": "float(np.sum(_oracle_simulate_double_heston_aes_paths(1,c,v1,v2,0)))"},
        {"setup": "import numpy as np\nc=np.array([0,0,0,0,0,.01,.02]); v1=np.array([[0.],[.1]]); v2=np.array([[0.],[.2]])",
         "call": "float(np.sum(simulate_double_heston_aes_paths(7,c,v1,v2,99)))",
         "gold_call": "float(np.sum(_oracle_simulate_double_heston_aes_paths(7,c,v1,v2,99)))"},
    ]
