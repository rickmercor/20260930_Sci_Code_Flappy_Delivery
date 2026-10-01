"""
Evaluate the regularized fractional kernel of Eq (19) on a 1-D grid of nonnegative lags. Apply the regularization the source prescribes at its stated resolution scale; it must leave the kernel finite at zero lag and preserve the fractional decay at resolved lags. Raise ValueError if alpha is outside (0, 1/2), if delta_star is not positive, if t is not a non-empty 1-D array, or if any lag is negative.

The benchmark weakly singular fractional kernel cannot drive a jump channel directly: a jump would produce an unbounded response at zero lag. The benchmark kernel is K_alpha(t) = t^(-alpha) / Gamma(1 - alpha) with 0 < alpha < 1/2; the source replaces it with a finite-resolution variant at the resolution scale delta_star, which is finite at zero lag and keeps the fractional decay at lags long compared with delta_star.

Returns
-------
ndarray of float64, the kernel evaluated elementwise on t, same shape as t.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def regularized_kernel(alpha: float, delta_star: float, t: "np.ndarray") -> "np.ndarray":
    """Evaluate the regularized fractional kernel of Eq (19) on a 1-D grid of nonnegative lags.

    Parameters
    ----------
    alpha : float
        Roughness exponent of the fractional kernel, 0 < alpha < 1/2.
    delta_star : float
        Resolution scale of the regularization, positive.
    t : np.ndarray
        Non-empty 1-D array of nonnegative lags.

    Returns
    -------
    result : np.ndarray
        ndarray of float64, the kernel evaluated elementwise on t, same shape as t.

    Raises
    ------
    ValueError
        If alpha is outside (0, 1/2).
        If delta_star is not positive.
        If t is not a non-empty 1-D array.
        If any lag is negative.
    """
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import gamma, roots_jacobi, roots_legendre, roots_laguerre


def _oracle_regularized_kernel(alpha: float, delta_star: float, t: "np.ndarray") -> "np.ndarray":
    """Eq (19): translated fractional kernel K_{a,d*}(t) = (t + d*)^{-a} / Gamma(1-a)."""
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha must satisfy 0 < alpha < 1/2")
    if delta_star <= 0.0:
        raise ValueError("delta_star must be positive")
    t = np.asarray(t, dtype=np.float64)
    if t.ndim != 1 or t.size < 1:
        raise ValueError("t must be a non-empty 1-D array")
    if np.any(t < 0.0):
        raise ValueError("t must be nonnegative")
    # TRANSLATION, not clipping: the argument is shifted, which keeps the kernel
    # completely monotone with a nonnegative representing measure while making the
    # zero-lag response finite. Clipping and truncation also make it finite and are
    # the natural guesses, but neither retains that representation.
    return (t + delta_star) ** (-alpha) / gamma(1.0 - alpha)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nalpha, delta_star = 0.35, 0.05\nt = np.linspace(0.0, 1.0, 7)\nt_ref = t.copy()",
            "call": "regularized_kernel(alpha, delta_star, t)",
            "gold_call": "_oracle_regularized_kernel(alpha, delta_star, t_ref)",
        },
        {
            "setup": "import numpy as np\nalpha, delta_star = 0.35, 0.05\nt = np.array([0.0])\nt_ref = t.copy()",
            "call": "regularized_kernel(alpha, delta_star, t)",
            "gold_call": "_oracle_regularized_kernel(alpha, delta_star, t_ref)",
        },
        {
            "setup": "import numpy as np\nalpha, delta_star = 0.05, 1e-3\nt = np.array([0.0, 1e-8, 5.0])\nt_ref = t.copy()",
            "call": "regularized_kernel(alpha, delta_star, t)",
            "gold_call": "_oracle_regularized_kernel(alpha, delta_star, t_ref)",
        },
        {
            "setup": "import numpy as np\n# invalid input: an alpha outside (0, 1/2) must raise ValueError\ndef run_model():\n    try:\n        regularized_kernel(0.6, 0.05, np.array([0.0, 1.0]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_regularized_kernel(0.6, 0.05, np.array([0.0, 1.0]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
