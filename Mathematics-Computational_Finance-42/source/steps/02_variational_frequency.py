"""
Solve the variational condition that fixes the trial frequency at one average point and return that frequency. The condition matches the curvature of the trial quadratic potential to the smeared curvature of the true one, which makes the frequency and the fluctuation width mutually dependent and the problem a scalar fixed point rather than a formula.

The true potential of a lognormal intensity model is a quadratic mean-reversion term plus an exponential. The quadratic part contributes a curvature that does not move; the exponential part contributes a curvature equal to itself, and it is that self-reference that makes the scheme variational rather than perturbative. Smearing the exponential over the Gaussian fluctuation multiplies it by the exponential of half the width, so the width enters the frequency and the frequency enters the width.

The fixed point is a contraction because raising the frequency narrows the fluctuation, which lowers the smeared curvature, which lowers the frequency again. Starting from the mean-reversion speed it converges in a handful of iterations at every average point in the integration window, and the limit does not depend on the starting value. That stability is what makes the method usable inside a quadrature loop: no bracketing, no derivative, no failure mode to trap.

At a vanishing intensity weight the exponential contribution disappears and the frequency collapses to the mean-reversion speed, which is the exactly-solvable Ornstein-Uhlenbeck case and the cheapest check on an implementation.

Only the frequency is returned. The width that belongs to it is recovered by feeding the frequency back through the previous step, which keeps one definition of the width in the pipeline rather than two.

Returns
-------
float, the trial frequency. At an average point of -3.352407217493 over a horizon of 5, with mean reversion 0.35, diffusion 0.62 and unit weight, the value is 0.370154368438; the width it induces is 0.151709823758. At zero weight the frequency is the mean-reversion speed exactly. The frequency is at least the mean-reversion speed and increases with the average point.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def variational_frequency(xbar, T, k, sigma, lam):
    """Trial frequency at one average point.

    The frequency solves the variational curvature-matching condition of the
    source method for a lognormal intensity h = exp(x) under

        dX_t = k*(theta_t - X_t)*dt + sigma*dW_t,

    weighted by exp(-lam * integral of h).  The condition couples the
    frequency to the fluctuation width of the previous step, so it is solved
    as a scalar fixed point started from omega = k and iterated to a relative
    tolerance of 1e-15.

    Args:
        xbar (float): average point at which the condition is imposed.
        T (float): horizon, strictly positive.
        k (float): mean-reversion speed, strictly positive.
        sigma (float): diffusion coefficient, strictly positive.
        lam (float): intensity weight, non-negative.

    Expected return:
        float: the trial frequency omega.  With lam = 0 it is exactly k.  The
        fluctuation width that belongs to it is obtained by passing omega back
        through the previous step.

    Raises:
        ValueError: if T <= 0, if k <= 0, if sigma <= 0, or if lam < 0.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_variational_frequency(xbar, T, k, sigma, lam):
    def _width(omega, T, sigma):
        f = 0.5 * omega * T
        if f < 1.0e-3:
            br = f / 3.0 - f ** 3 / 45.0 + 2.0 * f ** 5 / 945.0
        else:
            br = 1.0 / np.tanh(f) - 1.0 / f
        return sigma * sigma / (2.0 * omega) * br

    xbar, T, k = float(xbar), float(T), float(k)
    sigma, lam = float(sigma), float(lam)
    if T <= 0.0:
        raise ValueError("T must be positive")
    if k <= 0.0:
        raise ValueError("k must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if lam < 0.0:
        raise ValueError("lam must be non-negative")
    omega = k
    for _ in range(10000):
        a = _width(omega, T, sigma)
        new = np.sqrt(k * k + sigma * sigma * lam * np.exp(0.5 * a + xbar))
        if abs(new - omega) <= 1.0e-15 * max(1.0, abs(new)):
            omega = new
            break
        omega = new
    return float(omega)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": "",
            "call": ("round(float(variational_frequency(-3.352407217493, 5.0, 0.35, "
                     "0.62, 1.0)), 12)"),
            "gold_call": "0.370154368438",
            "tol": 1e-10,
        },
        {
            "setup": "",
            "call": "round(float(variational_frequency(-1.0, 2.0, 0.35, 0.62, 1.0)), 12)",
            "gold_call": "0.518106668279",
            "tol": 1e-10,
        },
        {
            "setup": "",
            "call": ("round(float(variational_frequency(-3.352407217493, 5.0, 0.35, "
                     "0.62, 0.0)), 12)"),
            "gold_call": "0.35",
            "tol": 1e-12,
        },
        {
            "setup": (
                "xb, T, k, sig, lam = -2.0, 4.0, 0.35, 0.62, 1.0\n"
                "om = float(variational_frequency(xb, T, k, sig, lam))\n"
                "f = 0.5 * om * T\n"
                "al = sig ** 2 / (2.0 * om) * (1.0 / np.tanh(f) - 1.0 / f)\n"
                "res = om ** 2 - k ** 2 - sig ** 2 * lam * np.exp(0.5 * al + xb)"
            ),
            "call": "bool(abs(float(res)) < 1e-12)",
            "gold_call": "True",
        },
        {
            "setup": (
                "ws = [float(variational_frequency(xb, 3.0, 0.35, 0.62, 1.0)) "
                "for xb in (-6.0, -4.0, -2.0, 0.0)]"
            ),
            "call": "bool(all(ws[i] < ws[i + 1] for i in range(3)) and ws[0] > 0.35)",
            "gold_call": "True",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        variational_frequency(-3.0, 0.0, 0.35, 0.62, 1.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_variational_frequency(-3.0, 0.0, 0.35, 0.62, 1.0)\n"
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
                "        variational_frequency(-3.0, 5.0, 0.35, 0.62, -1.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_variational_frequency(-3.0, 5.0, 0.35, 0.62, -1.0)\n"
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
