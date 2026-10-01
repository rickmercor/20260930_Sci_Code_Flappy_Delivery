"""
Apply the smearing operator to the payoff and sum the per-node weights into the two expectations the pricing needs: the one for a payoff identically equal to one, and the one for a payoff equal to the intensity itself. Both are produced from the same per-node data, because the terminal integration has already been carried out and all that remains is a Gaussian smearing of the payoff about the point that integration left behind.

The smearing is not optional decoration. Integrating the terminal state out of a Gaussian leaves a payoff averaged over the residual terminal uncertainty, and the width of that averaging is the reciprocal of twice the quadratic coefficient. For a payoff identically one the averaging does nothing, which is why the survival side looks like a plain sum; for an exponential payoff it multiplies by the exponential of a quarter of the reciprocal quadratic coefficient, and dropping that factor biases the intensity-weighted expectation low at every node and by a different amount at each.

The two outputs are related by a derivative in the horizon: the intensity-weighted expectation is minus the rate of change of the survival expectation. That identity is not used to compute either one, but it is the sharpest available check that the smearing has been applied with the right width and at the right point.

The sum is taken in logarithms with the largest exponent factored out, because the per-node weights span many orders of magnitude across the window and the extreme nodes underflow before their contribution is negligible.

Returns
-------
np.ndarray of shape (2,) holding the expectation for the unit payoff and the expectation for the exponential payoff, in that order. For a single node with zero log weight, zero evaluation point, unit quadratic coefficient and unit quadrature weight the pair is (1.0, 1.284025416688), the second entry being the exponential of one quarter.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def assemble_expectation(log_weight, centre, A, quad_weight):
    """Smear the payoff and sum the per-node contributions.

    Each node of the average-point grid carries a log weight, the point at
    which the payoff is evaluated after the terminal state has been integrated
    out, and the quadratic coefficient of that integration.  The payoff is
    smeared over the residual Gaussian uncertainty left by the terminal
    integration and the results are summed against the quadrature weights.
    Two payoffs are evaluated: the constant payoff one, and the exponential
    payoff exp(x).

    Args:
        log_weight: (n,) array of per-node log weights.
        centre: (n,) array of payoff evaluation points.
        A: (n,) array of quadratic coefficients, all strictly positive.
        quad_weight: (n,) array of quadrature weights.

    Expected return:
        np.ndarray of shape (2,) packed as [unit payoff expectation,
        exponential payoff expectation].

    Raises:
        ValueError: if the four inputs do not all have the same length, if
        they are empty, or if any entry of A is not positive.
    """
    return np.zeros(2)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_assemble_expectation(log_weight, centre, A, quad_weight):
    ln = np.asarray(log_weight, dtype=float).reshape(-1)
    ce = np.asarray(centre, dtype=float).reshape(-1)
    aa = np.asarray(A, dtype=float).reshape(-1)
    ww = np.asarray(quad_weight, dtype=float).reshape(-1)
    if not (ln.size == ce.size == aa.size == ww.size):
        raise ValueError("all four inputs must have the same length")
    if ln.size == 0:
        raise ValueError("inputs must be non-empty")
    if np.any(aa <= 0.0):
        raise ValueError("entries of A must be positive")

    def _total(logs):
        mx = logs.max()
        return float(np.exp(mx) * np.sum(ww * np.exp(logs - mx)))

    return np.array([_total(ln), _total(ln + ce + 1.0 / (4.0 * aa))])

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": ("v = assemble_expectation(np.array([0.0]), np.array([0.0]), "
                      "np.array([1.0]), np.array([1.0]))"),
            "call": "np.round(v, 12)",
            "gold_call": "np.array([1.0, 1.284025416688])",
            "tol": 1e-10,
        },
        {
            "setup": (
                "v = assemble_expectation(np.array([-0.4, -1.2]), "
                "np.array([-3.4, -2.9]), np.array([1.6, 2.0]), "
                "np.array([0.3, 0.7]))"
            ),
            "call": "np.round(v, 12)",
            "gold_call": "np.array([0.411931962149, 0.020991735656])",
            "tol": 1e-10,
        },
        {
            "setup": (
                "ln = np.array([-0.4, -1.2, -2.0])\n"
                "ce = np.array([-3.4, -2.9, -3.1])\n"
                "aa = np.array([1.6, 2.0, 1.1])\n"
                "ww = np.array([0.3, 0.7, 0.2])\n"
                "a = assemble_expectation(ln, ce, aa, ww)\n"
                "b = assemble_expectation(ln, ce, aa, 3.0 * ww)"
            ),
            "call": "np.round(b - 3.0 * a, 12)",
            "gold_call": "np.zeros(2)",
        },
        {
            "setup": (
                "ln = np.array([-0.4, -1.2])\n"
                "ce = np.array([-3.4, -2.9])\n"
                "aa = np.array([1.6, 2.0])\n"
                "ww = np.array([0.3, 0.7])\n"
                "v = assemble_expectation(ln, ce, aa, ww)\n"
                "manual = float(np.sum(ww * np.exp(ln + ce + 1.0 / (4.0 * aa))))"
            ),
            "call": "round(float(v[1]) - manual, 12)",
            "gold_call": "0.0",
        },
        {
            "setup": (
                "ln = np.array([-5.0, -400.0])\n"
                "v = assemble_expectation(ln, np.array([-3.0, -3.0]), "
                "np.array([1.0, 1.0]), np.array([1.0, 1.0]))"
            ),
            "call": "bool(abs(float(v[0]) - np.exp(-5.0)) < 1e-18 and float(v[0]) > 0.0)",
            "gold_call": "True",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        assemble_expectation(np.zeros(3), np.zeros(2), np.ones(3), "
                "np.ones(3))\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_assemble_expectation(np.zeros(3), np.zeros(2), "
                "np.ones(3), np.ones(3))\n"
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
                "        assemble_expectation(np.zeros(2), np.zeros(2), np.zeros(2), "
                "np.ones(2))\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_assemble_expectation(np.zeros(2), np.zeros(2), "
                "np.zeros(2), np.ones(2))\n"
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
