"""
Assemble the logarithm of the normalisation of the constrained trial propagator at one average point. Three ingredients enter: the Gaussian prefactor of the constrained measure, the ratio of the half-product to its hyperbolic sine, and the exponential of the action correction less the time integral of the constant part of the trial potential. The logarithm is returned rather than the value because the exponent runs over many tens across the integration window and the value itself underflows long before the contribution to the integral becomes negligible.

The constant part of the trial potential is fixed by matching the smeared value of the true potential, not by any free choice, and it carries the exponential term in its smeared form. That smearing multiplies the intensity by the exponential of half the fluctuation width, exactly as it did in the curvature condition, so the same width appears in the frequency and in the level. A level built from the unsmeared intensity gives a normalisation that is too large at every average point, and the error grows with the horizon.

The prefactor is the product of two square roots, one carrying the fluctuation width and one carrying the horizon and the diffusion coefficient. The first is the normalisation of the constrained fluctuation, the second that of the free measure over the average point itself; keeping only one of them leaves a result that is not a density in the average point at all.

The hyperbolic ratio is the usual fluctuation determinant of a harmonic action. Written naively it overflows for a stiff node; written through the exponential of minus the half-product it is stable everywhere in the window.

Returns
-------
float. At an average point of -3.352407217493 with frequency 0.370154368438, fluctuation width 0.151709823758 and action correction 0.024700410536, over a horizon of 5 with mean reversion 0.35, diffusion 0.62, unit weight and level -3.688879454114, the value is -1.600491532702.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def log_trial_normalisation(xbar, omega, alpha, corr, T, k, sigma, lam, theta):
    """Logarithm of the constrained trial propagator normalisation.

    Combines the Gaussian prefactor of the constrained measure, the harmonic
    fluctuation determinant and the exponential of the action correction less
    the time integral of the constant part of the trial potential.  The
    constant part is fixed by matching the smeared value of the true
    potential of the lognormal intensity model, so it carries the intensity in
    its smeared form.

    Args:
        xbar (float): average point.
        omega (float): trial frequency, strictly positive.
        alpha (float): fluctuation width, strictly positive.
        corr (float): additive action correction from the previous step.
        T (float): horizon, strictly positive.
        k (float): mean-reversion speed.
        sigma (float): diffusion coefficient, strictly positive.
        lam (float): intensity weight.
        theta (float): mean-reversion level, constant in time.

    Expected return:
        float: the natural logarithm of the normalisation.

    Raises:
        ValueError: if omega <= 0, if alpha <= 0, if T <= 0, or if sigma <= 0.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_log_trial_normalisation(xbar, omega, alpha, corr, T, k, sigma,
                                    lam, theta):
    xbar, omega, alpha, corr = float(xbar), float(omega), float(alpha), float(corr)
    T, k, sigma = float(T), float(k), float(sigma)
    lam, theta = float(lam), float(theta)
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if T <= 0.0:
        raise ValueError("T must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    f = 0.5 * omega * T
    w_int = T * (k * k * (theta - xbar) ** 2 / (2.0 * sigma ** 2)
                 + lam * np.exp(0.5 * alpha + xbar)
                 + (k * k - omega * omega) / (2.0 * sigma ** 2) * alpha)
    log_fsinh = np.log(2.0 * f) - f - np.log1p(-np.exp(-2.0 * f))
    return float(-0.5 * np.log(2.0 * np.pi * alpha)
                 - 0.5 * np.log(2.0 * np.pi * T * sigma ** 2)
                 + log_fsinh - w_int + corr)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": "",
            "call": ("round(log_trial_normalisation(-3.352407217493, 0.370154368438, "
                     "0.151709823758, 0.024700410536, 5.0, 0.35, 0.62, 1.0, "
                     "-3.688879454114), 12)"),
            "gold_call": "-1.600491532702",
            "tol": 1e-10,
        },
        {
            "setup": "",
            "call": ("round(log_trial_normalisation(-2.5, 0.394424551990, "
                     "0.093929679528, 0.073609678634, 3.0, 0.35, 0.62, 1.0, "
                     "-3.688879454114), 12)"),
            "gold_call": "-1.632227758862",
            "tol": 1e-10,
        },
        {
            "setup": (
                "a = log_trial_normalisation(-3.0, 0.4, 0.12, 0.0, 4.0, 0.35, 0.62, "
                "1.0, -3.688879454114)\n"
                "b = log_trial_normalisation(-3.0, 0.4, 0.12, 0.7, 4.0, 0.35, 0.62, "
                "1.0, -3.688879454114)"
            ),
            "call": "round(b - a - 0.7, 12)",
            "gold_call": "0.0",
        },
        {
            "setup": (
                "a = log_trial_normalisation(-3.0, 0.4, 0.12, 0.0, 4.0, 0.35, 0.62, "
                "0.0, -3.688879454114)\n"
                "b = log_trial_normalisation(-3.0, 0.4, 0.12, 0.0, 4.0, 0.35, 0.62, "
                "1.0, -3.688879454114)\n"
                "want = 4.0 * np.exp(0.5 * 0.12 - 3.0)"
            ),
            "call": "round(a - b - want, 12)",
            "gold_call": "0.0",
        },
        {
            "setup": (
                "v = log_trial_normalisation(1.0, 40.0, 0.006, 0.0, 5.0, 0.35, 0.62, "
                "1.0, -3.688879454114)"
            ),
            "call": "bool(np.isfinite(v))",
            "gold_call": "True",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        log_trial_normalisation(-3.0, 0.4, 0.0, 0.0, 4.0, 0.35, 0.62, "
                "1.0, -3.6)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_log_trial_normalisation(-3.0, 0.4, 0.0, 0.0, 4.0, 0.35, "
                "0.62, 1.0, -3.6)\n"
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
                "        log_trial_normalisation(-3.0, -0.4, 0.12, 0.0, 4.0, 0.35, 0.62, "
                "1.0, -3.6)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_log_trial_normalisation(-3.0, -0.4, 0.12, 0.0, 4.0, 0.35, "
                "0.62, 1.0, -3.6)\n"
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
