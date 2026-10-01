"""
Return the 13 sub-step durations of one DS4 step, in the operator order C B A B C B A B C B A B C. C entries are physical times and A/B entries are scaled times tau = eps*weight*h.

The symmetric second-order map C(h/2) B(eps h/2) A(eps h) B(eps h/2) C(h/2) is raised to fourth order by Yoshida's triple-jump with weights (lambda, 1 - 2*lambda, lambda), lambda = 1/(2 - 2^(1/3)). Adjacent C flows merge, which leaves 13 operators with two negative Kepler durations (1 - lambda)*h/2 and a negative middle A/B block.

Returns
-------
np.ndarray of shape (13,): sub-step durations in operator order C B A B C B A B C B A B C.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def ds4_schedule(h, eps):
    """Sub-step durations of the fourth-order DS4 composition for one step of size h.

    Operators are applied in the fixed order C B A B C B A B C B A B C (13 entries).
    C entries are physical times for the doubled Kepler flow; A and B entries are the
    scaled durations tau = eps * (weight) * h passed to the cross-coupled flows.
    Returns a flat array of length 13.
    Raises ValueError for h <= 0 or eps < 0.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_ds4_schedule(h, eps):
    # Oracle: Yoshida triple-jump of the symmetric map C(h/2) B(eps h/2) A(eps h) B(eps h/2) C(h/2)
    # with weights (lam, 1 - 2 lam, lam), lam = 1/(2 - 2**(1/3)); adjacent C flows merged.
    if h <= 0.0:
        raise ValueError("step size h must be positive")
    if eps < 0.0:
        raise ValueError("eps must be non-negative")
    lam = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    w = [lam, 1.0 - 2.0 * lam, lam]
    out = [w[0] * h / 2.0]
    for i, wi in enumerate(w):
        out.extend([wi * eps * h / 2.0, wi * eps * h, wi * eps * h / 2.0])
        nxt = w[i + 1] if i < 2 else 0.0
        out.append((wi + nxt) * h / 2.0)
    return np.array(out, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    return [
        {
            "name": (
                "normal_task_step"
            ),
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "ds4_schedule(2.0, 0.6309573444801932)\n"
            ),
            "gold_call": (
                "_oracle_ds4_schedule(2.0, 0.6309573444801932)\n"
            ),
            "tol": 1e-13,
        },
        {
            "name": (
                "boundary_eps_zero_pure_kepler"
            ),
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "ds4_schedule(2.0, 0.0)\n"
            ),
            "gold_call": (
                "_oracle_ds4_schedule(2.0, 0.0)\n"
            ),
            "tol": 1e-13,
        },
        {
            "name": (
                "edge_small_step_weak_field"
            ),
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "ds4_schedule(0.5, 0.1)\n"
            ),
            "gold_call": (
                "_oracle_ds4_schedule(0.5, 0.1)\n"
            ),
            "tol": 1e-13,
        },
        {
            "name": (
                "invalid_negative_step_raises_valueerror"
            ),
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def run_model():\n"
                "    try:\n"
                "        ds4_schedule(-1.0, 0.1)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "\n"
                "\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_ds4_schedule(-1.0, 0.1)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
            ),
            "call": (
                "run_model()\n"
            ),
            "gold_call": (
                "run_gold()\n"
            ),
            "tol": 0,
        },
    ]
