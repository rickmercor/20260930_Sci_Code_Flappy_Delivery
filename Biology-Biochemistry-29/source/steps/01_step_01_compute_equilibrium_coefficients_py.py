"""
Evaluate the equilibrium parametrization from a complete vector of fourteen kinetic rate constants. Return the five coefficients multiplying Q, the rational coefficient beta, and the constant concentration Yp. Validate that the input and all returned coefficients are finite and strictly positive.

The equilibrium representation separates five Q-linear concentrations from Xp=beta Q/Y and a rate-dependent constant Yp. For k=(k1,...,k14), define D=k9k11(k13+k14)+k1k3k12k14(k10+k11)/(k2*(k4+k5)). The coefficients are c_X=k8*(k4+k5)/(k3k5), c_XD=k2k8*(k4+k5)/(k1k3k5), c_XDYp=k8k9(k13+k14)/D, c_XT=k8/k5, c_XTYp=k1k3k8k12(k10+k11)/(k2*(k4+k5)D), beta=(k7+k8)/k6, and Yp=k1k3k5(k10+k11)(k13+k14)/(k2(k4+k5)*D). Preserve the full rational expressions; no coefficient may be set to zero merely because it is small.

Returns
-------
np.ndarray of shape (7,), containing finite, strictly positive coefficients.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_equilibrium_coefficients(
    rate_constants: np.ndarray,
) -> np.ndarray:
    """Return [c_X, c_XD, c_XDYp, c_XT, c_XTYp, beta, Yp].

    The input has shape (14,) in k1,...,k14 order.
    Evaluate the complete equilibrium parametrization without
    setting small positive coefficients to zero.

    Returns:
        np.ndarray: Seven finite, strictly positive coefficients.

    Raises:
        ValueError: If the rate vector has an invalid shape,
            contains nonfinite or nonpositive values, or produces
            nonfinite or nonpositive coefficients.
    """
    return np.empty(7, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _checked_array(value, shape, name, positive=False):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must have shape {shape} and be finite")
    if positive and np.any(a <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return a

def _oracle_compute_equilibrium_coefficients(rate_constants):
    import numpy as np

    k = _checked_array(rate_constants, (14,), "rate_constants", True)

    with np.errstate(all="ignore"):
        D = (
            k[8]*k[10]*(k[12]+k[13])
            + k[0]*k[2]*k[11]*k[13]*(k[9]+k[10])
            / (k[1]*(k[3]+k[4]))
        )

        c = np.array([
            k[7]*(k[3]+k[4])/(k[2]*k[4]),
            k[1]*k[7]*(k[3]+k[4])/(k[0]*k[2]*k[4]),
            k[7]*k[8]*(k[12]+k[13])/D,
            k[7]/k[4],
            k[0]*k[2]*k[7]*k[11]*(k[9]+k[10])
            / (k[1]*(k[3]+k[4])*D),
            (k[6]+k[7])/k[5],
            k[0]*k[2]*k[4]*(k[9]+k[10])*(k[12]+k[13])
            / (k[1]*(k[3]+k[4])*D)
        ])

    return _checked_array(c, (7,), "coefficients", True)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common = """import numpy as np

def capture(fn, *args):
    try:
        fn(*args)
        return 0.0
    except ValueError:
        return 1.0
    except Exception:
        return 2.0

k = np.array([
    1.7, 0.83, 1.2, 0.47, 1.11, 2.7,
    0.62, 0.8, 1.37, 0.58, 1.43, 1.0,
    0.66, 1.21
], dtype=float)
"""

    return [
        {
            "description": "Normal independent synthetic kinetic rates.",
            "setup": common,
            "call": "compute_equilibrium_coefficients(k)",
            "gold_call": "_oracle_compute_equilibrium_coefficients(k)",
            "tol": 1e-9
        },
        {
            "description": "Large finite k6 retains a nonzero beta.",
            "setup": common + "\nk[5] = 1e13\n",
            "call": "compute_equilibrium_coefficients(k)",
            "gold_call": "_oracle_compute_equilibrium_coefficients(k)",
            "tol": 1e-9
        },
        {
            "description": "A zero rate is rejected.",
            "setup": common + "\nk[2] = 0.0\n",
            "call": "capture(compute_equilibrium_coefficients, k)",
            "gold_call": "capture(_oracle_compute_equilibrium_coefficients, k)",
            "tol": 1e-9
        }
    ]
