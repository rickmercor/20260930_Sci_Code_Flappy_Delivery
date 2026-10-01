"""
Carry out the integral over the terminal state analytically and return the quadratic, linear and constant coefficients it is performed with, the point the remaining payoff is evaluated at, and the logarithm of the resulting weight for this average point. The terminal state appears quadratically in three places at once: in the constrained propagator between the endpoints, in the propagator's dependence on the average, and in the drift-removal factor that turned the mean-reverting process into a driftless one. Collecting all three is what makes the integral Gaussian and therefore closed-form.

The drift-removal factor is the part that is easy to lose. Moving from the mean-reverting process to a driftless one costs an exponential of a quadratic in the endpoints plus a linear term in the mean-reversion level and a constant proportional to the horizon; those pieces belong to the coefficients here, not to the potential. An implementation that builds the coefficients from the propagator alone gets a quadratic coefficient short of the mean-reversion term and a constant short of the horizon term, and the resulting weight is not a probability density in any limit.

The linear coefficient is defined so that the Gaussian is centred at the initial state minus half the linear over the quadratic, which is why it is not simply the coefficient of the terminal state in the exponent. That centre is where the payoff is evaluated after the terminal integration, and it moves with the average point.

The constant coefficient contains the square of the same endpoint gap that appears in the linear one, and the cross term with the summed response integral. Both survive to the final weight, and the weight is the object the quadrature over the average point is applied to.

Returns
-------
np.ndarray of shape (5,) holding the quadratic coefficient, the linear coefficient, the constant coefficient, the payoff evaluation point and the logarithm of the weight for this average point, in that order. At an average point equal to the initial state -3.352407217493 over a horizon of 5 the entries are 1.609678810745, 0.306361297652, 0.850299589464, -3.447569462161 and -0.401267317263.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def endpoint_coefficients(xbar, x0, delta, reduced, alpha, omega, log_N, T, k,
                          sigma, theta):
    """Coefficients of the analytic terminal-state integration.

    Collects every dependence on the terminal state - from the constrained
    propagator, from its dependence on the average point, and from the
    drift-removal factor that maps the mean-reverting process onto a driftless
    one - into a quadratic, a linear and a constant coefficient, then returns
    them together with the point at which the remaining payoff is evaluated
    and the logarithm of the weight attached to this average point.

    Args:
        xbar (float): average point.
        x0 (float): initial state.
        delta (float): endpoint displacement from the previous step.
        reduced: (5,) array of reduced quantities from the third step.
        alpha (float): fluctuation width, strictly positive.
        omega (float): trial frequency, strictly positive.
        log_N (float): logarithm of the trial normalisation.
        T (float): horizon.
        k (float): mean-reversion speed.
        sigma (float): diffusion coefficient, strictly positive.
        theta (float): mean-reversion level, constant in time.

    Expected return:
        np.ndarray of shape (5,) packed as [A, B, C, centre, log_weight],
        where centre = x0 - B/(2*A).

    Raises:
        ValueError: if reduced does not hold exactly five entries, if
        alpha <= 0, if omega <= 0, or if sigma <= 0.
    """
    return np.zeros(5)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_endpoint_coefficients(xbar, x0, delta, reduced, alpha, omega,
                                  log_N, T, k, sigma, theta):
    v = np.asarray(reduced, dtype=float).reshape(-1)
    if v.size != 5:
        raise ValueError("reduced must hold five entries")
    xbar, x0, delta = float(xbar), float(x0), float(delta)
    alpha, omega, log_N = float(alpha), float(omega), float(log_N)
    T, k, sigma, theta = float(T), float(k), float(sigma), float(theta)
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    f = 0.5 * omega * T
    coth = (1.0 + np.exp(-2.0 * f)) / (1.0 - np.exp(-2.0 * f))
    r_sum, r_zero = v[2], v[3]
    A = (1.0 / (8.0 * alpha) + omega * coth / (4.0 * sigma ** 2)
         + k / (2.0 * sigma ** 2))
    B = ((x0 - xbar + delta) / (2.0 * alpha) + r_zero
         + k * (x0 - theta) / sigma ** 2)
    C = (-(x0 - xbar + delta) ** 2 / (2.0 * alpha)
         - (x0 - xbar) * r_sum + k * T / 2.0)
    log_weight = 0.5 * np.log(np.pi / A) + log_N + C + B * B / (4.0 * A)
    return np.array([A, B, C, x0 - B / (2.0 * A), log_weight])

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

RED_A = ("red = np.array([0.144984665608, 0.724923328042, 0.570637399292, "
         "0.285318699646, 0.030215898683])\n")
RED_B = ("red = np.array([0.464902342200, 1.394707026601, 1.251932859103, "
         "0.625966429551, 0.084142891892])\n")


def test_cases():
    return [
        {
            "setup": (
                RED_A
                + "v = endpoint_coefficients(-3.352407217493, -3.352407217493, "
                  "-0.086571299276, red, 0.151709823758, 0.370154368438, "
                  "-1.600491532702, 5.0, 0.35, 0.62, -3.688879454114)"
            ),
            "call": "np.round(v, 12)",
            "gold_call": ("np.array([1.609678810745, 0.306361297652, 0.850299589464, "
                          "-3.447569462161, -0.401267317263])"),
            "tol": 1e-10,
        },
        {
            "setup": (
                RED_B
                + "v = endpoint_coefficients(-2.5, -3.352407217493, -0.117593652245, "
                  "red, 0.093929679528, 0.394424551990, -1.632227758862, 3.0, 0.35, "
                  "0.62, -3.688879454114)"
            ),
            "call": "np.round(v, 12)",
            "gold_call": ("np.array([2.269060177520, -4.231114087022, -3.416386445726, "
                          "-2.420057752075, -2.913493646610])"),
            "tol": 1e-10,
        },
        {
            "setup": (
                RED_A
                + "v = endpoint_coefficients(-3.0, -3.352407217493, -0.08, red, 0.15, "
                  "0.37, -1.6, 5.0, 0.35, 0.62, -3.688879454114)\n"
                  "resid = v[3] - (-3.352407217493 - v[1] / (2.0 * v[0]))"
            ),
            "call": "round(float(resid), 12)",
            "gold_call": "0.0",
        },
        {
            "setup": (
                RED_A
                + "a = endpoint_coefficients(-3.0, -3.352407217493, -0.08, red, 0.15, "
                  "0.37, -1.6, 5.0, 0.35, 0.62, -3.688879454114)\n"
                  "b = endpoint_coefficients(-3.0, -3.352407217493, -0.08, red, 0.15, "
                  "0.37, -0.6, 5.0, 0.35, 0.62, -3.688879454114)"
            ),
            "call": "np.round(np.array([b[4] - a[4] - 1.0, b[0] - a[0]]), 12)",
            "gold_call": "np.zeros(2)",
        },
        {
            "setup": (
                "red = np.zeros(5)\n"
                "v = endpoint_coefficients(-3.352407217493, -3.352407217493, 0.0, red, "
                "0.15, 0.37, 0.0, 5.0, 0.35, 0.62, -3.688879454114)"
            ),
            "call": "round(float(v[2]), 12)",
            "gold_call": "0.875",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        endpoint_coefficients(-3.0, -3.3, 0.0, np.zeros(3), 0.15, 0.37, "
                "0.0, 5.0, 0.35, 0.62, -3.6)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_endpoint_coefficients(-3.0, -3.3, 0.0, np.zeros(3), 0.15, "
                "0.37, 0.0, 5.0, 0.35, 0.62, -3.6)\n"
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
                "        endpoint_coefficients(-3.0, -3.3, 0.0, np.zeros(5), 0.0, 0.37, "
                "0.0, 5.0, 0.35, 0.62, -3.6)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_endpoint_coefficients(-3.0, -3.3, 0.0, np.zeros(5), 0.0, "
                "0.37, 0.0, 5.0, 0.35, 0.62, -3.6)\n"
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
