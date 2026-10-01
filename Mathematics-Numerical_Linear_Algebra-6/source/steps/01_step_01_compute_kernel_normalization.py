"""
Compute the spherical-kernel normalization used by the fracture threshold.

For an ascending polynomial kernel $\omega_s(\rho)=\sum_{j=0}^p a_j\rho^j$

on $0\leq\rho\leq1$, the normalization is the ratio



$$

c_0=\frac{\int_0^1\omega_s(\rho)\rho^3\,d\rho}

{2\int_0^1\omega_s(\rho)\rho^2\,d\rho}.

$$



Analytic integration avoids quadrature error and preserves the ascending

coefficient convention used throughout the task.

Returns
-------
A positive finite float equal to the exact polynomial-integral ratio.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_kernel_normalization(kernel_coefficients: np.ndarray) -> float:
    r"""Return the spherical-kernel normalization $c_0$.

    Raises ``ValueError`` unless ``kernel_coefficients`` is a nonempty,
    one-dimensional, finite array; the kernel is nonnegative at 1001 equally
    spaced points on $[0,1]$; and both defining integrals and the resulting
    normalization are strictly positive and finite.

    Parameters
    ----------
    kernel_coefficients : np.ndarray
        Ascending coefficients $(a_0,a_1,\ldots,a_p)$ of $\omega_s$.

    Returns
    -------
    float
        Positive dimensionless normalization $c_0$.
    """
    return NotImplemented

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_kernel_normalization(kernel_coefficients: np.ndarray) -> float:
    """Reference analytic evaluation of the normalization ratio."""
    coefficients = np.asarray(kernel_coefficients, dtype=float)
    if coefficients.ndim != 1 or coefficients.size == 0:
        raise ValueError("kernel_coefficients must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("kernel_coefficients must be finite")
    rho = np.linspace(0.0, 1.0, 1001)
    if np.min(np.polynomial.polynomial.polyval(rho, coefficients)) < -1e-12:
        raise ValueError("the kernel must be nonnegative on [0, 1]")
    powers = np.arange(coefficients.size, dtype=float)
    numerator = np.sum(coefficients / (powers + 4.0))
    denominator = 2.0 * np.sum(coefficients / (powers + 3.0))
    if numerator <= 0.0 or denominator <= 0.0:
        raise ValueError("the defining kernel moments must be positive")
    result = float(numerator / denominator)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("the normalization must be positive and finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return quadratic-kernel, constant-kernel, and invalid-kernel cases."""
    return [
        {
            "setup": """import numpy as np
kernel_coefficients = np.array([1.0, -2.0, 1.0])
""",
            "call": "compute_kernel_normalization(kernel_coefficients)",
            "gold_call": "_oracle_compute_kernel_normalization(kernel_coefficients)",
        },
        {
            "setup": """import numpy as np
kernel_coefficients = np.array([1.0])
""",
            "call": "compute_kernel_normalization(kernel_coefficients)",
            "gold_call": "_oracle_compute_kernel_normalization(kernel_coefficients)",
        },
        {
            "setup": """import numpy as np
kernel_coefficients = np.array([1.0, -3.0])
def run_model():
    try:
        compute_kernel_normalization(kernel_coefficients)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_kernel_normalization(kernel_coefficients)
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
