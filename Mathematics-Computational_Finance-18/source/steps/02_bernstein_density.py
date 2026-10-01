"""
Evaluate the density of the representing measure of the SAME regularized kernel used in the previous step, at strictly positive frequencies x. This is the density whose mixture of decaying exponentials reproduces that kernel; its form follows from the regularization the source fixed. Raise ValueError if alpha is outside (0, 1/2), if delta_star is not positive, if x is not a non-empty 1-D array, or if any x is not strictly positive.

A completely monotone kernel is a mixture of decaying exponentials. The mixing density is what makes a finite-dimensional Markovian realization possible, and its form is inherited from whichever regularization the source fixed.

Returns
-------
ndarray of float64, the density evaluated elementwise on x, same shape as x.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bernstein_density(alpha: float, delta_star: float, x: "np.ndarray") -> "np.ndarray":
    """Evaluate the density of the representing measure of the regularized kernel of the
    previous step at strictly positive frequencies.

    Parameters
    ----------
    alpha : float
        Roughness exponent of the fractional kernel, 0 < alpha < 1/2.
    delta_star : float
        Resolution scale of the regularization, positive.
    x : np.ndarray
        Non-empty 1-D array of strictly positive frequencies.

    Returns
    -------
    result : np.ndarray
        ndarray of float64, the density evaluated elementwise on x, same shape as x.

    Raises
    ------
    ValueError
        If alpha is outside (0, 1/2).
        If delta_star is not positive.
        If x is not a non-empty 1-D array.
        If any x is not strictly positive.
    """
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import gamma, roots_jacobi, roots_legendre, roots_laguerre


def _oracle_bernstein_density(alpha: float, delta_star: float, x: "np.ndarray") -> "np.ndarray":
    """Density of the representing measure of the TRANSLATED kernel.

    K_{a,d*}(t) = int_0^inf exp(-x t) w(x) dx  with
        w(x) = exp(-x d*) x^{a-1} / (Gamma(a) Gamma(1-a)).
    The exp(-x d*) factor is exactly what the translation contributes.
    """
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha must satisfy 0 < alpha < 1/2")
    if delta_star <= 0.0:
        raise ValueError("delta_star must be positive")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size < 1:
        raise ValueError("x must be a non-empty 1-D array")
    if np.any(x <= 0.0):
        raise ValueError("x must be strictly positive")
    return np.exp(-x * delta_star) * x ** (alpha - 1.0) / (gamma(alpha) * gamma(1.0 - alpha))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nalpha, delta_star = 0.35, 0.05\nx = np.array([0.1, 1.0, 10.0])\nx_ref = x.copy()",
            "call": "bernstein_density(alpha, delta_star, x)",
            "gold_call": "_oracle_bernstein_density(alpha, delta_star, x_ref)",
        },
        {
            "setup": "import numpy as np\nalpha, delta_star = 0.35, 0.05\nx = np.array([1e-6])\nx_ref = x.copy()",
            "call": "bernstein_density(alpha, delta_star, x)",
            "gold_call": "_oracle_bernstein_density(alpha, delta_star, x_ref)",
        },
        {
            "setup": "import numpy as np\nalpha, delta_star = 0.45, 0.5\nx = np.array([1e-3, 1.0, 1e2])\nx_ref = x.copy()",
            "call": "bernstein_density(alpha, delta_star, x)",
            "gold_call": "_oracle_bernstein_density(alpha, delta_star, x_ref)",
        },
        {
            "setup": "import numpy as np\n# invalid input: a frequency that is not strictly positive must raise ValueError\ndef run_model():\n    try:\n        bernstein_density(0.35, 0.05, np.array([0.0, 1.0]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_bernstein_density(0.35, 0.05, np.array([0.0, 1.0]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
