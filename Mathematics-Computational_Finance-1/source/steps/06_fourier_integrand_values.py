"""
Combine the characteristic exponent with the payoff transform along the damped contour and return the real integrand of the Fourier valuation formula at every node. The exponent arrives already evaluated, as two real arrays, so this step does no model work and is a pure assembly of the pricing formula. The discount factor and the constant of the inverse transform are included here rather than left to the caller.

Fourier pricing rests on Parseval: the expectation of a payoff is the integral of the product of the characteristic function and the payoff's Fourier transform. Neither factor is integrable on the real line for a call, whose payoff grows without bound, so the contour is shifted into the complex plane by a damping parameter. The shift makes the payoff transform decay and costs nothing in exact arithmetic, since the integrand is analytic in the strip between the two constraints.

The call transform has a pole where the shift vanishes and another at the origin, and only one side of the strip is available in closed form here: the payoff side, which requires the shift to sit strictly below minus one. The model side depends on where the moment generating function of the log price stops being finite, and for rough Heston that boundary has no usable characterisation, which is why the damping is chosen numerically and tabulated rather than derived.

Because the integrand is even in the Fourier variable, only the half-line is needed and the result is doubled. The doubling is absorbed into the quadrature functional rather than applied here, so this step returns the integrand itself.

Returns
-------
np.ndarray of shape (n,). With a vanishing exponent the values reduce to the real part of the payoff transform over two pi. A strongly negative exponent drives the values to zero, and an empty node array returns an empty result.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def fourier_integrand(u, G_re, G_im, R, K, r, T):
    """Fourier pricing integrand of a European call at the damped contour.

    With xi = u + 1j*R and Phi(xi) = exp(G(xi)), the integrand of the Fourier
    valuation formula is

        g(u) = exp(-r*T) / (2*pi) * Re[ Phi(xi) * Phat(xi) ],

    where Phat is the Fourier transform of the call payoff,

        Phat(xi) = -K**(1 - 1j*xi) / (xi**2 + 1j*xi).

    Payoff admissibility requires R < -1; the value R = -1 is the pole of Phat
    at xi = -1j.

    Args:
        u: (n,) array of non-negative Fourier variables.
        G_re, G_im: (n,) arrays holding the characteristic exponent at
            xi = u + 1j*R, as produced by the exponent step.
        R (float): damping parameter, strictly less than -1.
        K (float): strike.
        r (float): risk-free rate.
        T (float): maturity.

    Expected return:
        np.ndarray of shape (n,): the integrand values g(u).
        Raises:
        ValueError: if u, G_re and G_im differ in length, if R >= -1,
        or if K <= 0.
    """
    return np.zeros(np.asarray(u, dtype=float).size)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_fourier_integrand(u, G_re, G_im, R, K, r, T):
    def _payoff_hat(xi, K):
        return -(K ** (1.0 - 1j * xi)) / (xi * xi + 1j * xi)

    u = np.asarray(u, dtype=float).reshape(-1)
    gr = np.asarray(G_re, dtype=float).reshape(-1)
    gi = np.asarray(G_im, dtype=float).reshape(-1)
    if u.size != gr.size or u.size != gi.size:
        raise ValueError("u, G_re and G_im must have the same length")
    R, K, r, T = float(R), float(K), float(r), float(T)
    if R >= -1.0:
        raise ValueError("payoff admissibility requires R < -1")
    if K <= 0.0:
        raise ValueError("K must be positive")
    xi = u + 1j * R
    return np.exp(-r * T) / (2.0 * np.pi) * np.real(
        np.exp(gr + 1j * gi) * _payoff_hat(xi, K))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": (
                "u = np.array([0.0])\n"
                "g = fourier_integrand(u, np.array([0.0]), np.array([0.0]), "
                "-1.5, 100.0, 0.0, 1.0)\n"
                "xi = -1.5j\n"
                "want = np.real(-(100.0 ** (1.0 - 1j * xi)) / (xi * xi + 1j * xi)) "
                "/ (2.0 * np.pi)"
            ),
            "call": "round(float(g[0] - want), 12)",
            "gold_call": "0.0",
        },
        {
            "setup": (
                "u = np.array([0.5, 1.0, 2.0])\n"
                "z = np.zeros(3)\n"
                "a = fourier_integrand(u, z, z, -2.0, 100.0, 0.0, 1.0)\n"
                "b = fourier_integrand(u, z, z, -2.0, 100.0, 0.05, 1.0)"
            ),
            "call": "np.round(b - a * np.exp(-0.05), 12)",
            "gold_call": "np.zeros(3)",
        },
        {
            "setup": (
                "u = np.array([0.3, 1.7])\n"
                "gr = np.array([-0.4, -1.1])\n"
                "gi = np.array([0.2, -0.7])"
            ),
            "call": ("np.round(fourier_integrand(u, gr, gi, -1.5, 1000.0, 0.0, "
                     "1.0), 8)"),
            "gold_call": "np.array([-0.00326061, -0.00017709])",
            "tol": 1e-6,
        },
        {
            "setup": (
                "u = np.array([1.0])\n"
                "big = fourier_integrand(u, np.array([-50.0]), np.array([0.0]), "
                "-1.5, 1000.0, 0.0, 1.0)"
            ),
            "call": "bool(abs(float(big[0])) < 1e-15)",
            "gold_call": "True",
        },
        {
            "setup": (
                "u = np.zeros(0)\n"
                "g = fourier_integrand(u, u, u, -1.5, 100.0, 0.0, 1.0)"
            ),
            "call": "int(np.asarray(g).size)",
            "gold_call": "0",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        fourier_integrand(np.array([1.0]), np.array([0.0]), "
                "np.array([0.0]), -1.0, 100.0, 0.0, 1.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_fourier_integrand(np.array([1.0]), "
                "np.array([0.0]), np.array([0.0]), -1.0, 100.0, 0.0, 1.0)\n"
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
                "        fourier_integrand(np.array([1.0, 2.0]), np.array([0.0]), "
                "np.array([0.0]), -1.5, 100.0, 0.0, 1.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_fourier_integrand(np.array([1.0, 2.0]), "
                "np.array([0.0]), np.array([0.0]), -1.5, 100.0, 0.0, 1.0)\n"
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
