"""
Turn the reduced response integrals into the two displacements and the one action correction that the linear part of the trial potential generates. A constant force applied to a harmonic path does three things: it shifts the whole path, it shifts the point about which the constrained fluctuation is centred, and it lowers the action by an amount quadratic in the force. The three outputs here are exactly those three effects.

The first displacement enters the propagator between the two endpoints and therefore appears in the coefficients of the endpoint integral later. The second is the offset between the average point and the centre of the diagonal density, and it is the argument at which the smearing operator of the variational conditions is evaluated, which is why the frequency condition of the earlier step already referred to it. They are different quantities built from the same three integrals and are easy to conflate.

The action correction subtracts a term proportional to the square of the gap between the two ways of integrating the force, which is the part that would otherwise be double counted between the endpoint displacement and the normalisation. It is the only place where the nested double integral is used, and omitting it leaves a normalisation that is wrong by a state-dependent factor rather than by a constant, so it does not cancel in any ratio.

There is a structural check worth keeping. When the force does not vary in time, the two ways of weighting it against the interval agree and the second displacement vanishes identically, while the first does not. An implementation that returns a non-zero second displacement for a time-constant mean-reversion level has mixed up the two hyperbolic weights.

Returns
-------
np.ndarray of shape (3,) holding the endpoint displacement, the action correction and the centre displacement, in that order. For the reduced integrals at an average point of -3.352407217493 with frequency 0.370154368438 over a horizon of 5 and diffusion 0.62 the entries are -0.086571299276, 0.024700410536 and 0.0. The third entry is zero to machine precision for any time-constant linear coefficient.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def displacement_and_correction(reduced, omega, T, sigma):
    """Endpoint displacement, action correction and centre displacement.

    Consumes the five reduced quantities of the previous step and returns the
    three scalars the linear part of the trial potential contributes: the
    displacement that enters the propagator between the endpoints, the
    additive correction to the trial action, and the displacement of the
    centre of the constrained diagonal density away from the average point.

    Args:
        reduced: (5,) array [gamma, gamma_hat, r_sum, r_zero, r_cross] as
            produced by the previous step.
        omega (float): trial frequency, strictly positive.
        T (float): horizon, strictly positive.
        sigma (float): diffusion coefficient.

    Expected return:
        np.ndarray of shape (3,) packed as [delta, correction, delta_gamma].
        The third entry vanishes when the linear coefficient is constant in
        time.

    Raises:
        ValueError: if reduced does not hold exactly five entries, if
        omega <= 0, or if T <= 0.
    """
    return np.zeros(3)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_displacement_and_correction(reduced, omega, T, sigma):
    v = np.asarray(reduced, dtype=float).reshape(-1)
    omega, T, sigma = float(omega), float(T), float(sigma)
    if v.size != 5:
        raise ValueError("reduced must hold five entries")
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    if T <= 0.0:
        raise ValueError("T must be positive")
    ghat, r_sum, r_cross = v[1], v[2], v[4]
    f = 0.5 * omega * T
    gap = r_sum - ghat
    delta = sigma ** 2 / (2.0 * omega * f) * gap
    corr = sigma ** 2 / omega * (r_cross - 0.25 / f * gap * gap)
    d_gamma = sigma ** 2 / (2.0 * omega) * (r_sum / np.tanh(f) - ghat / f)
    return np.array([delta, corr, d_gamma])

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

RED_A = ("red = np.array([0.144984665608, 0.724923328042, 0.570637399292, "
         "0.285318699646, 0.030215898683])\n")
RED_B = ("red = np.array([2.645389524269, 5.290779048538, 4.210859737957, "
         "2.105429868978, 1.587115128481])\n")


def test_cases():
    return [
        {
            "setup": RED_A + "v = displacement_and_correction(red, 0.370154368438, 5.0, 0.62)",
            "call": "np.round(v, 12)",
            "gold_call": "np.array([-0.086571299276, 0.024700410536, 0.0])",
            "tol": 1e-10,
        },
        {
            "setup": RED_B + "v = displacement_and_correction(red, 0.9, 2.0, 0.62)",
            "call": "np.round(v, 12)",
            "gold_call": "np.array([-0.256247520363, 0.539511183223, 0.0])",
            "tol": 1e-10,
        },
        {
            "setup": (
                "red = np.zeros(5)\n"
                "v = displacement_and_correction(red, 0.7, 3.0, 0.62)"
            ),
            "call": "np.round(v, 12)",
            "gold_call": "np.zeros(3)",
        },
        {
            "setup": (
                "g = 0.4\n"
                "om, T, sig = 0.8, 2.5, 0.62\n"
                "th = np.tanh(0.5 * om * T)\n"
                "red = np.array([g, g * T, 2.0 * g / om * th, g / om * th,\n"
                "                g * g / om * (0.5 * T - th / om)])\n"
                "v = displacement_and_correction(red, om, T, sig)"
            ),
            "call": "bool(abs(float(v[2])) < 1e-14 and float(v[0]) < 0.0)",
            "gold_call": "True",
        },
        {
            "setup": (
                RED_A
                + "a = displacement_and_correction(red, 0.370154368438, 5.0, 0.62)\n"
                  "b = displacement_and_correction(red, 0.370154368438, 5.0, 1.24)"
            ),
            "call": "np.round(b[:2] - 4.0 * a[:2], 12)",
            "gold_call": "np.zeros(2)",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        displacement_and_correction(np.zeros(4), 0.9, 2.0, 0.62)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_displacement_and_correction(np.zeros(4), 0.9, 2.0, 0.62)\n"
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
                "        displacement_and_correction(np.zeros(5), 0.0, 2.0, 0.62)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_displacement_and_correction(np.zeros(5), 0.0, 2.0, 0.62)\n"
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
