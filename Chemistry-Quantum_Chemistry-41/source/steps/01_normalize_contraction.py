"""
Implement normalize_contraction, which converts the tabulated contraction coefficients of one
contracted Gaussian shell into weights of unnormalized radial primitives that give a
unit-norm contracted radial function.

Gaussian basis sets are tabulated as primitive exponents and contraction coefficients for
each contracted shell.

Returns
-------
np.ndarray of shape (n,): weights w_k of the unnormalized radial primitives r^l exp(-beta_k r^2) giving a unit-norm contracted radial function
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def normalize_contraction(l: int, exponents: "np.ndarray", coefficients: "np.ndarray") -> "np.ndarray":
    '''Weights of unnormalized radial primitives for a unit-norm contracted Gaussian shell.

    Parameters
    ----------
    l : int
        Angular momentum of the shell, a non-negative integer.
    exponents : np.ndarray
        Primitive exponents beta_k (bohr^-2), shape (n,), n >= 1, all positive.
    coefficients : np.ndarray
        Tabulated contraction coefficients c_k, shape (n,), of either sign; zeros are allowed
        but not all of them may vanish. c_k multiplies the normalized radial primitive
        N_k r^l exp(-beta_k r^2), where N_k makes int_0^inf (N_k r^l exp(-beta_k r^2))^2 r^2 dr = 1.

    Returns
    -------
    weights : np.ndarray
        Shape (n,). Weights w_k such that the radial function
        R(r) = sum_k w_k r^l exp(-beta_k r^2) is proportional to sum_k c_k N_k r^l exp(-beta_k r^2)
        with a positive factor and satisfies int_0^inf R(r)^2 r^2 dr = 1.

    Raises
    ------
    ValueError
        If l is negative, exponents and coefficients are empty or of different lengths, an
        exponent is not positive, or every coefficient is zero.
    '''
    return weights

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import gamma


def _oracle_normalize_contraction(l: int, exponents: "np.ndarray", coefficients: "np.ndarray") -> "np.ndarray":
    if int(l) != l or l < 0:
        raise ValueError("l must be a non-negative integer")
    l = int(l)
    beta = np.asarray(exponents, dtype=float).ravel()
    coef = np.asarray(coefficients, dtype=float).ravel()
    if beta.size == 0 or beta.size != coef.size:
        raise ValueError("exponents and coefficients must be non-empty and of equal length")
    if np.any(beta <= 0.0):
        raise ValueError("exponents must be positive")
    if not np.any(coef != 0.0):
        raise ValueError("at least one contraction coefficient must be non-zero")
    g = gamma(l + 1.5)
    prim_norm = np.sqrt(2.0 * (2.0 * beta) ** (l + 1.5) / g)
    weights = coef * prim_norm
    radial_overlap = g / (2.0 * (beta[:, None] + beta[None, :]) ** (l + 1.5))
    return weights / np.sqrt(weights @ radial_overlap @ weights)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    invalid = """
def run_model():
    try:
        normalize_contraction(l, exponents.copy(), coefficients.copy())
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_normalize_contraction(l, exponents.copy(), coefficients.copy())
        return 0
    except ValueError:
        return 1
"""
    return [
        # --- Typical: nitrogen diffuse s augmentation, strongly cancelling coefficients ---
        {
            "setup": """import numpy as np
l = 0
exponents = np.array([0.09, 0.1846836552, 0.402911141, 0.9277239437])
coefficients = np.array([-155.7296559966, 61.8126687524, -69.8698010234, -5.5545900093])
""",
            "call": "normalize_contraction(l, exponents.copy(), coefficients.copy())",
            "gold_call": "_oracle_normalize_contraction(l, exponents.copy(), coefficients.copy())",
            "tol": 1e-10,
        },
        # --- Typical: nitrogen d augmentation (l = 2) ---
        {
            "setup": """import numpy as np
l = 2
exponents = np.array([0.09, 0.1846836552, 0.402911141, 0.9277239437])
coefficients = np.array([-0.0966051188, -0.131498833, 0.0681453767, -0.1191458075])
""",
            "call": "normalize_contraction(l, exponents.copy(), coefficients.copy())",
            "gold_call": "_oracle_normalize_contraction(l, exponents.copy(), coefficients.copy())",
            "tol": 1e-10,
        },
        # --- Typical: a p contraction whose primitive list contains zero coefficients ---
        {
            "setup": """import numpy as np
l = 1
exponents = np.array([0.0654284286, 0.100112428, 0.2430767471, 0.6259552659, 1.822142904])
coefficients = np.array([0.4762021876, -0.1110297174, 5.00489e-05, 0.0, 0.0])
""",
            "call": "normalize_contraction(l, exponents.copy(), coefficients.copy())",
            "gold_call": "_oracle_normalize_contraction(l, exponents.copy(), coefficients.copy())",
            "tol": 1e-10,
        },
        # --- Boundary: a single primitive with a negative coefficient (weight is the positive
        #     primitive normalization) ---
        {
            "setup": """import numpy as np
l = 2
exponents = np.array([0.8])
coefficients = np.array([-2.5])
""",
            "call": "normalize_contraction(l, exponents.copy(), coefficients.copy())",
            "gold_call": "_oracle_normalize_contraction(l, exponents.copy(), coefficients.copy())",
            "tol": 1e-10,
        },
        # --- Edge: a tight six-primitive core contraction ---
        {
            "setup": """import numpy as np
l = 0
exponents = np.array([1027.828458, 188.4512226, 52.72186097, 18.11138217, 7.033179691, 2.896651794])
coefficients = np.array([0.0091635963, 0.0493614929, 0.1685383049, 0.3705627997, 0.4164915298, 0.1303340841])
""",
            "call": "normalize_contraction(l, exponents.copy(), coefficients.copy())",
            "gold_call": "_oracle_normalize_contraction(l, exponents.copy(), coefficients.copy())",
            "tol": 1e-10,
        },
        # --- Invalid: exponents and coefficients of different lengths ---
        {
            "setup": """import numpy as np
l = 0
exponents = np.array([1.0, 0.2])
coefficients = np.array([1.0])
""" + invalid,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: every coefficient is zero ---
        {
            "setup": """import numpy as np
l = 1
exponents = np.array([1.0, 0.2])
coefficients = np.array([0.0, 0.0])
""" + invalid,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
