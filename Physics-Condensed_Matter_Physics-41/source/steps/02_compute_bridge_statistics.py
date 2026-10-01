"""
Compute the harmonic and free fixed-endpoint Gaussian bridge moments for one bead between two fixed neighboring beads.

Obtain the one-bead harmonic bridge by conditioning the exact harmonic-oscillator density matrix on the two fixed neighboring beads, after shifting all coordinates by the local minimum. Return its conditional mean and variance together with the corresponding free-particle bridge mean and variance. The implementation must use an algebraically equivalent numerically stable form: it must remain finite for both very small and very large $a=\tau\hbar\omega$, including values for which direct evaluation of $\sinh(a)$ or $\cosh(a)$ would overflow. The same stability requirement applies to the affine endpoint combinations: do not form an overflowing shifted endpoint sum when the exact conditional and free means are finite.

Returns
-------
np.ndarray, a length-4 float array [harmonic_mean, harmonic_variance, free_mean, free_variance]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

from numbers import Real

import numpy as np


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def compute_bridge_statistics(
    left: float,
    right: float,
    minimum: float,
    tau: float,
    mass: float,
    omega: float,
    hbar: float,
) -> np.ndarray:
    """Compute exact harmonic and free fixed-endpoint bridge statistics.

    Parameters
    ----------
    left, right : float
        Fixed neighboring bead positions.
    minimum : float
        Position of the local harmonic minimum.
    tau : float
        Positive imaginary-time step.
    mass : float
        Positive particle mass.
    omega : float
        Positive harmonic reference frequency.
    hbar : float
        Positive reduced Planck constant.

    Returns
    -------
    statistics : numpy.ndarray
        Length-4 float array ``[harmonic_mean, harmonic_variance,
        free_mean, free_variance]``. Both the hyperbolic factors and the
        affine endpoint combinations must be evaluated without overflow
        for finite inputs whose four exact outputs are finite.

    Raises
    ------
    ValueError
        If any scalar input is not a finite real value; if ``tau``,
        ``mass``, ``omega``, or ``hbar`` is not strictly positive; or if
        the resulting bridge statistics are non-finite or either variance
        is not strictly positive.
    """
    return np.full(4, np.nan, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_bridge_statistics(
    left: float,
    right: float,
    minimum: float,
    tau: float,
    mass: float,
    omega: float,
    hbar: float,
) -> np.ndarray:
    """Reference implementation with stable hyperbolic and affine evaluation."""
    import math
    import numbers

    import numpy as np

    values = {
        "left": left,
        "right": right,
        "minimum": minimum,
        "tau": tau,
        "mass": mass,
        "omega": omega,
        "hbar": hbar,
    }
    converted: dict[str, float] = {}
    for name, value in values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be a finite real scalar")
        value_f = float(value)
        if not math.isfinite(value_f):
            raise ValueError(f"{name} must be finite")
        converted[name] = value_f

    left_f = converted["left"]
    right_f = converted["right"]
    minimum_f = converted["minimum"]
    tau_f = converted["tau"]
    mass_f = converted["mass"]
    omega_f = converted["omega"]
    hbar_f = converted["hbar"]
    if tau_f <= 0.0 or mass_f <= 0.0 or omega_f <= 0.0 or hbar_f <= 0.0:
        raise ValueError("tau, mass, omega, and hbar must be > 0")

    a = tau_f * hbar_f * omega_f
    if not math.isfinite(a) or a <= 0.0:
        raise ValueError("tau*hbar*omega must be finite and > 0")

    tanh_a = math.tanh(a)
    if a < 20.0:
        sech_a = 1.0 / math.cosh(a)
    else:
        exp_minus_a = math.exp(-a)
        sech_a = 2.0 * exp_minus_a / (1.0 + exp_minus_a * exp_minus_a)

    # Preserve the ordinary-regime operation order, but fall back to a
    # scaled convex combination if shifted coordinates overflow.
    q_left = left_f - minimum_f
    q_right = right_f - minimum_f
    q_sum = q_left + q_right
    harmonic_mean = minimum_f + 0.5 * q_sum * sech_a
    if not math.isfinite(harmonic_mean):
        scale = max(abs(minimum_f), abs(left_f), abs(right_f))
        if scale == 0.0:
            harmonic_mean = 0.0
        else:
            harmonic_mean = scale * math.fsum(
                [
                    (1.0 - sech_a) * (minimum_f / scale),
                    0.5 * sech_a * (left_f / scale),
                    0.5 * sech_a * (right_f / scale),
                ]
            )

    free_sum = left_f + right_f
    free_mean = 0.5 * free_sum
    if not math.isfinite(free_mean):
        scale = max(abs(left_f), abs(right_f))
        if scale == 0.0:
            free_mean = 0.0
        else:
            free_mean = 0.5 * scale * math.fsum(
                [left_f / scale, right_f / scale]
            )

    harmonic_variance = hbar_f * tanh_a / (2.0 * mass_f * omega_f)
    free_variance = hbar_f * hbar_f * tau_f / (2.0 * mass_f)

    result = np.array(
        [harmonic_mean, harmonic_variance, free_mean, free_variance],
        dtype=float,
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("bridge statistics must be finite")
    if harmonic_variance <= 0.0 or free_variance <= 0.0:
        raise ValueError("bridge variances must be > 0")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, asymptotic, boundary, invalid, and affine-overflow cases."""
    return [
        {
            "setup": """left = 0.22\nright = 0.03\nminimum = -0.15\ntau = 0.6\nmass = 1.3\nomega = (3.2 / 1.3) ** 0.5\nhbar = 0.7""",
            "call": "compute_bridge_statistics(left, right, minimum, tau, mass, omega, hbar)",
            "gold_call": "_oracle_compute_bridge_statistics(left, right, minimum, tau, mass, omega, hbar)",
        },
        {
            "setup": """left = 2.0\nright = -1.0\nminimum = 0.3\ntau = 800.0\nmass = 1.0\nomega = 1.0\nhbar = 1.0""",
            "call": "compute_bridge_statistics(left, right, minimum, tau, mass, omega, hbar)",
            "gold_call": "_oracle_compute_bridge_statistics(left, right, minimum, tau, mass, omega, hbar)",
        },
        {
            "setup": """left = 1.2\nright = -0.7\nminimum = 0.1\ntau = 1.0e-12\nmass = 2.0\nomega = 3.0\nhbar = 0.5""",
            "call": "compute_bridge_statistics(left, right, minimum, tau, mass, omega, hbar)",
            "gold_call": "_oracle_compute_bridge_statistics(left, right, minimum, tau, mass, omega, hbar)",
        },
        {
            "setup": """left = 0.0\nright = 0.0\nminimum = 0.0\ntau = 0.0\nmass = 1.0\nomega = 1.0\nhbar = 1.0\ndef run_model():\n    try:\n        compute_bridge_statistics(left, right, minimum, tau, mass, omega, hbar)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_compute_bridge_statistics(left, right, minimum, tau, mass, omega, hbar)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """left = 0.0\nright = 0.0\nminimum = 0.0\ntau = 1.0\nmass = 1.0\nomega = 1.0\nhbar = float('nan')\ndef run_model():\n    try:\n        compute_bridge_statistics(left, right, minimum, tau, mass, omega, hbar)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_compute_bridge_statistics(left, right, minimum, tau, mass, omega, hbar)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """left = 1.0e308\nright = -1.0e308\nminimum = 1.0e308\ntau = 800.0\nmass = 1.0\nomega = 1.0\nhbar = 1.0""",
            "call": "compute_bridge_statistics(left, right, minimum, tau, mass, omega, hbar)",
            "gold_call": "_oracle_compute_bridge_statistics(left, right, minimum, tau, mass, omega, hbar)",
        },
    ]
