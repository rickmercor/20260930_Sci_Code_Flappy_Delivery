"""
Choose the finest level of the time-step hierarchy from the first level difference and the discretization share of the tolerance. The rule extrapolates the observed convergence rate from a single difference between two consecutive levels, converts it into a predicted error at any level, and returns the smallest level at which the prediction meets the target. The level is never returned below one, since that is where the indicator is defined.

The quantity the level is supposed to control is not computable: it is the distance between the discrete price and the exact one. What is computable is the gap between two consecutive discrete prices, and under an assumption that the leading error term does not cancel after integration, that gap is proportional to the error with a known constant. The constant is the difference of two consecutive powers of two raised to the convergence rate, which is where the denominator in the rule comes from.

Evaluating the indicator at every level would defeat the purpose, since each evaluation costs a pair of solves at that level. Instead one difference is computed at the bottom of the hierarchy with a fixed diagnostic point count, and the rate is used to extrapolate it upwards. The unit offset in the rule is the bookkeeping that accounts for the indicator being anchored at level one rather than level zero.

The convergence rate itself is empirical. No proof is available for the fractional Adams scheme in this setting; the observed slope matches one plus the fractional order across the tested configurations, and that is what the rule uses. Substituting a guessed integer order changes the selected level and therefore the answer.

Returns
-------
float holding an integer level, at least one. For the task the first level difference 0.008803790 with a discretization tolerance of 1e-3 and a rate of 1.62 gives 3; the same difference with a rate of 2 gives 2, and a vanishing difference gives 1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def select_discretization_level(D1, eps_disc, p):
    """Finest discretization level of the time-step hierarchy.

    The hierarchy is dt_l = 2**(-l) * dt_0.  The discretization error is not
    computable directly, so the level is chosen from the asymptotic
    Richardson-type indicator the source paper derives, which extrapolates the
    observed rate from the first level difference

        D1 = | V_{Nbar, 1} - V_{Nbar, 0} |

    computed with a fixed number of quadrature points.  The rule returns the
    smallest admissible level whose indicator does not exceed the
    discretization part of the tolerance, and it never returns a level below
    the one at which the indicator is defined.

    Args:
        D1 (float): magnitude of the first level difference, positive.
        eps_disc (float): discretization part of the prescribed tolerance.
        p (float): convergence order of the Fourier integrand under refinement
            of the fractional Riccati time discretization.

    Expected return:
        float: the selected level L, an integer value returned as a float.
        Raises:
        ValueError: if D1 <= 0, if eps_disc <= 0, or if p <= 0.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_select_discretization_level(D1, eps_disc, p):
    D1, eps_disc, p = float(D1), float(eps_disc), float(p)
    if D1 <= 0.0:
        raise ValueError("D1 must be positive")
    if eps_disc <= 0.0:
        raise ValueError("eps_disc must be positive")
    if p <= 0.0:
        raise ValueError("p must be positive")
    inner = D1 / ((2.0 ** p - 1.0) * eps_disc)
    return float(max(1, int(np.ceil(1.0 + np.log2(inner) / p))))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": "",
            "call": ("float(select_discretization_level(0.00880379037602097, "
                     "1e-3, 1.62))"),
            "gold_call": "3.0",
        },
        {
            "setup": "",
            "call": ("float(select_discretization_level(0.00880379037602097, "
                     "1e-3, 2.0))"),
            "gold_call": "2.0",
        },
        {
            "setup": "",
            "call": "float(select_discretization_level(1e-12, 1e-2, 1.62))",
            "gold_call": "1.0",
        },
        {
            "setup": (
                "vals = [float(select_discretization_level(0.0088, e, 1.62)) "
                "for e in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)]"
            ),
            "call": "np.array(vals)",
            "gold_call": "np.array([1.0, 1.0, 3.0, 5.0, 7.0])",
        },
        {
            "setup": (
                "a = float(select_discretization_level(0.0088, 1e-3, 1.62))\n"
                "b = float(select_discretization_level(0.0176, 1e-3, 1.62))"
            ),
            "call": "bool(b >= a)",
            "gold_call": "True",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        select_discretization_level(-1.0, 1e-3, 1.62)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_select_discretization_level(-1.0, 1e-3, 1.62)\n"
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
                "        select_discretization_level(0.0088, 0.0, 1.62)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_select_discretization_level(0.0088, 0.0, 1.62)\n"
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
