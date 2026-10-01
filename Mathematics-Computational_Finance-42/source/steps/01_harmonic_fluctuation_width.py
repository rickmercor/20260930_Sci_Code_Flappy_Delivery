"""
Return the width of the Gaussian fluctuations that a path makes about its own time average, for a trial harmonic action of a given frequency over a given horizon. This single scalar is what the smearing operator of the variational scheme integrates against, so every quantity built later in the pipeline inherits it. It depends only on the frequency, the horizon and the diffusion coefficient, never on the state or on the default intensity.

The classification of paths by their average point is what separates this construction from an ordinary saddle-point expansion. Once the average is fixed, the remaining fluctuation is a bridge-like object whose variance is smaller than the free variance, because the constraint removes the zero mode that carries most of the spread. The correction that removes it is the reason the bracket below is a difference of two terms rather than a single hyperbolic cotangent, and dropping the second term is the commonest way to produce a width that is too large at every frequency.

Two limits are worth holding on to. At small frequency-horizon product the bracket behaves like one third of that product, so the width tends to the diffusion coefficient times the horizon over twelve, which is the variance of a Brownian bridge-like deviation from its own mean rather than the free variance. At large frequency the bracket tends to one and the width decays like the reciprocal of the frequency, so stiff nodes contribute a narrow smearing.

The difference of the two terms in the bracket cancels severely when the product is small, so an implementation that evaluates it naively loses precision there; the series expansion of the bracket is the stable route.

Returns
-------
float. For a frequency of 0.370154368438 over a horizon of 5 with a diffusion coefficient of 0.62 the value is 0.151709823758. The width is positive for every positive frequency, decreasing in the frequency, and tends to the diffusion coefficient squared times the horizon divided by twelve as the frequency tends to zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def harmonic_fluctuation_width(omega, T, sigma):
    """Width of the constrained Gaussian fluctuation of the trial action.

    For a trial harmonic action of frequency omega on [0, T] with diffusion
    coefficient sigma, paths are classified by their time average and the
    residual fluctuation about that average is Gaussian.  This returns its
    variance, written through the half-product f = omega*T/2.

    Args:
        omega (float): trial harmonic frequency, strictly positive.
        T (float): horizon, strictly positive.
        sigma (float): diffusion coefficient of the state process, strictly
            positive.

    Expected return:
        float: the fluctuation width, strictly positive.  As omega*T tends to
        zero the value tends to sigma**2 * T / 12.

    Raises:
        ValueError: if omega <= 0, if T <= 0, or if sigma <= 0.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_harmonic_fluctuation_width(omega, T, sigma):
    omega, T, sigma = float(omega), float(T), float(sigma)
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    if T <= 0.0:
        raise ValueError("T must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    f = 0.5 * omega * T
    if f < 1.0e-3:
        bracket = f / 3.0 - f ** 3 / 45.0 + 2.0 * f ** 5 / 945.0
    else:
        bracket = 1.0 / np.tanh(f) - 1.0 / f
    return float(sigma * sigma / (2.0 * omega) * bracket)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": "",
            "call": "round(harmonic_fluctuation_width(0.370154368438, 5.0, 0.62), 12)",
            "gold_call": "0.151709823758",
            "tol": 1e-10,
        },
        {
            "setup": "",
            "call": "round(harmonic_fluctuation_width(1.0, 1.0, 1.0), 12)",
            "gold_call": "0.081976706869",
            "tol": 1e-10,
        },
        {
            "setup": "",
            "call": "round(harmonic_fluctuation_width(2.0, 3.0, 0.8), 12)",
            "gold_call": "0.107461838397",
            "tol": 1e-10,
        },
        {
            "setup": (
                "a = harmonic_fluctuation_width(1.0e-4, 2.0, 0.5)\n"
                "limit = 0.5 ** 2 * 2.0 / 12.0"
            ),
            "call": "bool(abs(a - limit) < 1e-9)",
            "gold_call": "True",
        },
        {
            "setup": (
                "base = harmonic_fluctuation_width(1.3, 2.0, 0.7)\n"
                "scaled = harmonic_fluctuation_width(1.3, 2.0, 1.4)"
            ),
            "call": "round(scaled - 4.0 * base, 12)",
            "gold_call": "0.0",
        },
        {
            "setup": (
                "vals = [harmonic_fluctuation_width(w, 2.0, 0.9) "
                "for w in (0.25, 0.5, 1.0, 2.0, 4.0)]"
            ),
            "call": "bool(all(vals[i] > vals[i + 1] > 0.0 for i in range(4)))",
            "gold_call": "True",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        harmonic_fluctuation_width(0.0, 5.0, 0.62)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_harmonic_fluctuation_width(0.0, 5.0, 0.62)\n"
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
                "        harmonic_fluctuation_width(0.5, 5.0, -0.62)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_harmonic_fluctuation_width(0.5, 5.0, -0.62)\n"
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
