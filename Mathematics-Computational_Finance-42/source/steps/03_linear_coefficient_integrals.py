"""
Build the linear coefficient of the trial potential at one average point and return it together with the three weighted time integrals of it that every later stage consumes. The linear coefficient comes from matching the smeared first derivative of the true potential; the integrals weight it against the two hyperbolic solutions of the trial equation of motion, which is how the endpoints of the interval feel a force applied in its interior.

Each weighted integral is a response function. One weights the force by the solution that vanishes at the start of the interval, one by the solution that vanishes at the end, and one is a nested double integral that pairs the force with itself at two ordered times. The last is what produces the correction to the action rather than to the endpoints, and it is the only quantity here that is quadratic in the force.

Only the combinations in which these integrals divided by a hyperbolic sine of the full interval appear downstream, and those combinations stay bounded while the integrals themselves grow exponentially with the frequency-horizon product. Returning the reduced combinations rather than the raw integrals is therefore not a convenience: the raw ones overflow double precision well inside the integration window used later.

When the mean-reversion level is constant in time the linear coefficient does not depend on time either, and all three reduced combinations collapse to closed forms in the hyperbolic tangent of the half-product. The nested one keeps a term linear in the horizon, which is the signature of the two integrations not commuting.

Returns
-------
np.ndarray of shape (5,) holding the linear coefficient, its plain time integral, and the three reduced combinations in the order sum, start-weighted, nested. At an average point of -3.352407217493 with frequency 0.370154368438 over a horizon of 5 the entries are 0.144984665608, 0.724923328042, 0.570637399292, 0.285318699646 and 0.030215898683. The sum combination is exactly twice the start-weighted one whenever the coefficient is constant in time.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def linear_coefficient_integrals(xbar, omega, T, k, sigma, theta):
    """Linear trial coefficient and its reduced weighted time integrals.

    The linear coefficient of the trial quadratic potential at the average
    point xbar follows from matching the smeared first derivative of the true
    potential of the lognormal intensity model.  With a mean-reversion level
    constant in time it is itself constant in time.  The three reduced
    combinations returned are the two endpoint response integrals and the
    nested double integral, each already divided by the hyperbolic sine of
    the full frequency-horizon product so that nothing overflows.

    Args:
        xbar (float): average point.
        omega (float): trial frequency at that average point, strictly
            positive.
        T (float): horizon, strictly positive.
        k (float): mean-reversion speed.
        sigma (float): diffusion coefficient, strictly positive.
        theta (float): mean-reversion level, constant in time.

    Expected return:
        np.ndarray of shape (5,) packed as [gamma, gamma_hat, r_sum, r_zero,
        r_cross]: the linear coefficient, its plain time integral, and the
        sum, start-weighted and nested reduced combinations.

    Raises:
        ValueError: if omega <= 0, if T <= 0, or if sigma <= 0.
    """
    return np.zeros(5)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_linear_coefficient_integrals(xbar, omega, T, k, sigma, theta):
    xbar, omega, T = float(xbar), float(omega), float(T)
    k, sigma, theta = float(k), float(sigma), float(theta)
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    if T <= 0.0:
        raise ValueError("T must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    g = ((omega * omega - k * k) / sigma ** 2
         + k * k * (xbar - theta) / sigma ** 2)
    th = np.tanh(0.5 * omega * T)
    ghat = g * T
    r_sum = 2.0 * g / omega * th
    r_zero = g / omega * th
    r_cross = g * g / omega * (0.5 * T - th / omega)
    return np.array([g, ghat, r_sum, r_zero, r_cross])

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": (
                "v = linear_coefficient_integrals(-3.352407217493, 0.370154368438, "
                "5.0, 0.35, 0.62, -3.688879454114)"
            ),
            "call": "np.round(v, 12)",
            "gold_call": ("np.array([0.144984665608, 0.724923328042, 0.570637399292, "
                          "0.285318699646, 0.030215898683])"),
            "tol": 1e-10,
        },
        {
            "setup": (
                "v = linear_coefficient_integrals(-1.0, 0.9, 2.0, 0.35, 0.62, "
                "-3.688879454114)"
            ),
            "call": "np.round(v, 12)",
            "gold_call": ("np.array([2.645389524269, 5.290779048538, 4.210859737957, "
                          "2.105429868978, 1.587115128481])"),
            "tol": 1e-10,
        },
        {
            "setup": (
                "v = linear_coefficient_integrals(-2.0, 0.7, 3.0, 0.35, 0.62, "
                "-3.688879454114)"
            ),
            "call": "np.round(np.array([v[1] - v[0] * 3.0, v[2] - 2.0 * v[3]]), 12)",
            "gold_call": "np.zeros(2)",
        },
        {
            "setup": (
                "k, sig, th = 0.35, 0.62, -3.688879454114\n"
                "xb = th + (0.35 ** 2 - 0.9 ** 2) / 0.35 ** 2\n"
                "v = linear_coefficient_integrals(xb, 0.9, 2.0, k, sig, th)"
            ),
            "call": "np.round(v, 12)",
            "gold_call": "np.zeros(5)",
        },
        {
            "setup": (
                "big = linear_coefficient_integrals(2.0, 40.0, 5.0, 0.35, 0.62, "
                "-3.688879454114)"
            ),
            "call": "bool(np.all(np.isfinite(big)))",
            "gold_call": "True",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        linear_coefficient_integrals(-1.0, -0.9, 2.0, 0.35, 0.62, -3.6)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_linear_coefficient_integrals(-1.0, -0.9, 2.0, 0.35, "
                "0.62, -3.6)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        linear_coefficient_integrals(-1.0, 0.9, -2.0, 0.35, 0.62, -3.6)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_linear_coefficient_integrals(-1.0, 0.9, -2.0, 0.35, "
                "0.62, -3.6)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
