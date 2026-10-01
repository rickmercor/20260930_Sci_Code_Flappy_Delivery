"""
Return the factor that replaces the standard exponential weight of the Gauss-Laguerre rule by one matched to the integrand at hand. The factor combines a correlation-dependent prefactor with a bracket holding a term linear in maturity and a term carrying the fractional order through a Gamma function. It depends on the model and the maturity only, not on the strike, the damping or the discretization level.

A Gauss-Laguerre rule places its nodes assuming the integrand decays like the weight it is built from. The standard weight assumes a decay rate of one. Real Fourier integrands decay at a rate set by the model and the maturity, and when the two disagree the rule either wastes points far out in the tail where the integrand has already vanished, or stops too early and truncates a tail that is still contributing.

The rate used here comes from the asymptotic behaviour of the characteristic function along the contour. As the Fourier variable grows the Riccati solution becomes asymptotically linear in it, with a slope fixed by the correlation and the product of mean reversion and volatility of volatility, and the exponent inherits a linear growth whose real part is the decay rate of the characteristic function. Both terms of the bracket survive that limit: one from the long-run variance level over the whole interval, one from the initial variance weighted by a fractional power of maturity. Keeping only the first is the natural simplification for anyone reasoning from the classical model, and it is wrong here because the fractional term is the larger of the two at this maturity.

The payoff transform decays only algebraically, so it does not enter the rate. At perfect correlation the prefactor vanishes and the construction degenerates, which is why the endpoint is excluded rather than handled.

Returns
-------
float. For the task configuration the value is $1.674240093130$, the product of a prefactor $22.123366$ and a bracket $0.075677$. It is symmetric in the sign of the correlation and reduces to the prefactor times the mean-reversion term when the initial variance is zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def laguerre_scaling(T, alpha, gam, nu, rho, theta, V0):
    """Scaling factor of the scaled Gauss-Laguerre weight.

    The Fourier integrand is integrated against the weight exp(-sigma*u) on
    (0, inf) instead of the standard exp(-u).  The factor is chosen to match
    the estimated large-frequency exponential decay rate of the rough Heston
    characteristic function along the integration contour, using the decay
    rate the source paper adopts for this purpose.  The payoff transform
    contributes only an algebraic factor and does not enter.

    Args:
        T (float): maturity.
        alpha (float): fractional order H + 1/2.
        gam, nu, rho, theta, V0 (float): rough Heston parameters.

    Expected return:
        float: the scaling factor sigma, strictly positive for |rho| < 1.
        Raises:
        ValueError: if abs(rho) >= 1, if T <= 0, or if gam*nu == 0.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_laguerre_scaling(T, alpha, gam, nu, rho, theta, V0):
    def _gamma(x):
        g = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
             -176.61502916214059, 12.507343278686905, -0.13857109526572012,
             9.9843695780195716e-6, 1.5056327351493116e-7]
        if x < 0.5:
            return np.pi / (np.sin(np.pi * x) * _gamma(1.0 - x))
        x -= 1.0
        a = 0.99999999999980993
        t = x + 7.5
        for i, c in enumerate(g):
            a += c / (x + i + 1.0)
        return np.sqrt(2.0 * np.pi) * t ** (x + 0.5) * np.exp(-t) * a

    T, alpha = float(T), float(alpha)
    gam, nu, rho = float(gam), float(nu), float(rho)
    theta, V0 = float(theta), float(V0)
    if abs(rho) >= 1.0:
        raise ValueError("the scale degenerates at |rho| = 1")
    if T <= 0.0:
        raise ValueError("T must be positive")
    if gam * nu == 0.0:
        raise ValueError("gam*nu must be non-zero")
    pref = np.sqrt(1.0 - rho * rho) / (gam * nu)
    bracket = gam * theta * T + V0 * T ** (1.0 - alpha) / _gamma(2.0 - alpha)
    return float(pref * bracket)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": "",
            "call": ("round(laguerre_scaling(1.0, 0.62, 0.1, 0.331, -0.681, "
                     "0.3156, 0.0392), 12)"),
            "gold_call": "1.674240093130",
            "tol": 1e-9,
        },
        {
            "setup": "",
            "call": ("round(laguerre_scaling(2.0, 0.62, 0.1, 0.331, -0.681, "
                     "0.3156, 0.0392), 12)"),
            "gold_call": "2.666571209748",
            "tol": 1e-9,
        },
        {
            "setup": (
                "a = laguerre_scaling(1.0, 0.62, 0.1, 0.331, -0.681, 0.3156, 0.0)\n"
                "want = np.sqrt(1.0 - 0.681 ** 2) / (0.1 * 0.331) * 0.1 * 0.3156"
            ),
            "call": "round(a - want, 12)",
            "gold_call": "0.0",
        },
        {
            "setup": (
                "a = laguerre_scaling(1.0, 0.7151, 1.8967, 0.6144356, -0.6704, "
                "0.03848, 0.06246)\n"
                "b = laguerre_scaling(1.0, 0.7151, 1.8967, 0.6144356, 0.6704, "
                "0.03848, 0.06246)"
            ),
            "call": "round(a - b, 12)",
            "gold_call": "0.0",
        },
        {
            "setup": "",
            "call": ("round(laguerre_scaling(1.0, 0.7151, 1.8967, 0.6144356, "
                     "-0.6704, 0.03848, 0.06246), 12)"),
            "gold_call": "0.090661005412",
            "tol": 1e-9,
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        laguerre_scaling(1.0, 0.62, 0.1, 0.331, -1.0, 0.3156, "
                "0.0392)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_laguerre_scaling(1.0, 0.62, 0.1, 0.331, -1.0, "
                "0.3156, 0.0392)\n"
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
