"""
Return the beam width at which the reduced description predicts no width acceleration, for the given power, local Kerr coefficient, confinement strength and nonlocal length. Locate it by bisection on the bracket from lo to hi, halving the bracket two hundred times and returning its midpoint. Raise ValueError if the bracket is not ordered and positive, or if the predicted acceleration does not change sign across it.

This width is the stationary point of the effective potential: a beam launched there keeps its width, and a beam launched near it oscillates around it. When the medium parameters are changed slowly the beam follows this width adiabatically, which is the reference against which any accelerated protocol is judged.

Returns
-------
float: the width at which the predicted width acceleration vanishes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def equilibrium_width(power: float, gamma: float, alpha: float, sigma: float, lo: float = 1e-3, hi: float = 50.0) -> float:
    """Return the beam width at which the reduced description predicts no width acceleration, for the given power, local Kerr coefficient, confinement strength and nonlocal length. Locate it by bisection on the bracket from lo to hi, halving the bracket two hundred times and returning its midpoint. Raise ValueError if the bracket is not ordered and positive, or if the predicted acceleration does not change sign across it.

    Returns
    -------
    float: the width at which the predicted width acceleration vanishes.

    Raises
    ------
    ValueError
        If lo, hi do not satisfy 0 < lo < hi, or the predicted acceleration has the same sign at both ends.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_equilibrium_width(power: float, gamma: float, alpha: float, sigma: float,
                              lo: float = 1e-3, hi: float = 50.0) -> float:
    lo = float(lo); hi = float(hi)
    if not (0.0 < lo < hi):
        raise ValueError("require 0 < lo < hi")
    f = lambda a: _oracle_width_acceleration(a, power, gamma, alpha, sigma)
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0.0:
        raise ValueError("no sign change for the equilibrium condition on [lo, hi]")
    for _ in range(200):                      # bisection: deterministic, no scipy
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if flo * fm <= 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
    return float(0.5 * (lo + hi))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
            {
                    "setup": "import numpy as np",
                    "call": "equilibrium_width(12.0, 0.2, 0.15, 4.0)",
                    "gold_call": "_oracle_equilibrium_width(12.0, 0.2, 0.15, 4.0)"
            },
            {
                    "setup": "import numpy as np",
                    "call": "equilibrium_width(20.0, 0.1, 0.1, 0.5)",
                    "gold_call": "_oracle_equilibrium_width(20.0, 0.1, 0.1, 0.5)"
            },
            {
                    "setup": "import numpy as np",
                    "call": "equilibrium_width(12.0, 0.2, 0.15, 0.8, 1e-2, 5.0)",
                    "gold_call": "_oracle_equilibrium_width(12.0, 0.2, 0.15, 0.8, 1e-2, 5.0)"
            },
            {
                    "setup": "import numpy as np\ndef residual(fn, acc):\n    a = fn(12.0, 0.2, 0.15, 4.0)\n    return float(abs(acc(a, 12.0, 0.2, 0.15, 4.0)) < 1e-9)",
                    "call": "residual(equilibrium_width, width_acceleration)",
                    "gold_call": "residual(_oracle_equilibrium_width, _oracle_width_acceleration)"
            },
            {
                    "setup": "import numpy as np\ndef probe_public():\n    try:\n        equilibrium_width(12.0, 0.2, 0.15, 4.0, 5.0, 1.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\ndef probe_gold():\n    try:\n        _oracle_equilibrium_width(12.0, 0.2, 0.15, 4.0, 5.0, 1.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
                    "call": "probe_public()",
                    "gold_call": "probe_gold()"
            }
    ]
