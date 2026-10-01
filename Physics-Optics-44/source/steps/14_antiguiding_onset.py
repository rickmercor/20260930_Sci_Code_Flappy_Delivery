"""
Return the shortest compression protocol that stays everywhere guiding: the protocol length in the closed interval from lo to hi at which the smallest squared confinement strength demanded anywhere along the protocol is exactly zero. Protocols shorter than this demand a locally antiguiding profile, longer ones do not. The protocol is the same one the previous quantity describes, built from the same two equilibrium widths and the same minimum-jerk interpolation. Locate the crossing to an absolute accuracy of 1e-10 in the protocol length. Raise ValueError unless lo and hi satisfy 0 < lo < hi, or if the interval does not bracket a change of sign.

Accelerating a compression buys speed at the cost of control amplitude, and for the confinement knob that cost is bounded below by the requirement that the guiding structure remain guiding. The crossing therefore marks a physical limit on the achievable acceleration for this control strategy, separate from any limit set by how faithfully the final profile is reproduced. It is a property of the medium and the compression target alone, so it is independent of any propagation grid.

Returns
-------
float: the shortest protocol length that remains everywhere guiding.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def antiguiding_onset(power: float, gamma: float, alpha: float, sigma_i: float, sigma_f: float, lo: float = 1e-2, hi: float = 20.0) -> float:
    """Return the shortest compression protocol that stays everywhere guiding: the protocol length in the closed interval from lo to hi at which the smallest squared confinement strength demanded anywhere along the protocol is exactly zero. Protocols shorter than this demand a locally antiguiding profile, longer ones do not. The protocol is the same one the previous quantity describes, built from the same two equilibrium widths and the same minimum-jerk interpolation. Locate the crossing to an absolute accuracy of 1e-10 in the protocol length. Raise ValueError unless lo and hi satisfy 0 < lo < hi, or if the interval does not bracket a change of sign.

    Returns
    -------
    float: the shortest protocol length that remains everywhere guiding.

    Raises
    ------
    ValueError
        If lo and hi do not satisfy 0 < lo < hi, or if the interval brackets no change of sign.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_antiguiding_onset(power: float, gamma: float, alpha: float, sigma_i: float,
                              sigma_f: float, lo: float = 1e-2, hi: float = 20.0) -> float:
    """Shortest shortcut length whose confinement profile stays everywhere guiding."""
    lo = float(lo); hi = float(hi)
    if not (0.0 < lo < hi):
        raise ValueError("require 0 < lo < hi")
    f = lambda z: _oracle_confinement_floor(z, power, gamma, alpha, sigma_i, sigma_f)
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0.0:
        raise ValueError("no antiguiding onset is bracketed by [lo, hi]")
    for _ in range(200):                      # bisection: deterministic, no scipy
        if hi - lo <= 1e-12:
            break
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
                    "call": "antiguiding_onset(12.0, 0.2, 0.15, 4.0, 0.8)",
                    "gold_call": "_oracle_antiguiding_onset(12.0, 0.2, 0.15, 4.0, 0.8)",
                    "tol": 1e-08
            },
            {
                    "setup": "import numpy as np",
                    "call": "antiguiding_onset(20.0, 0.1, 0.1, 5.0, 0.5)",
                    "gold_call": "_oracle_antiguiding_onset(20.0, 0.1, 0.1, 5.0, 0.5)",
                    "tol": 1e-08
            },
            {
                    "setup": "import numpy as np",
                    "call": "antiguiding_onset(12.0, 0.2, 0.15, 4.0, 0.8, 0.5, 3.0)",
                    "gold_call": "_oracle_antiguiding_onset(12.0, 0.2, 0.15, 4.0, 0.8, 0.5, 3.0)",
                    "tol": 1e-08
            },
            {
                    "setup": "import numpy as np\ndef bracket_free(fn):\n    return float(fn(12.0, 0.2, 0.15, 4.0, 0.8, 1e-3, 30.0))",
                    "call": "bracket_free(antiguiding_onset)",
                    "gold_call": "bracket_free(_oracle_antiguiding_onset)",
                    "tol": 1e-08
            },
            {
                    "setup": "import numpy as np\ndef probe_nobracket_public():\n    try:\n        antiguiding_onset(12.0, 0.2, 0.15, 4.0, 0.8, 2.0, 20.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\ndef probe_nobracket_gold():\n    try:\n        _oracle_antiguiding_onset(12.0, 0.2, 0.15, 4.0, 0.8, 2.0, 20.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
                    "call": "probe_nobracket_public()",
                    "gold_call": "probe_nobracket_gold()"
            },
            {
                    "setup": "import numpy as np\ndef probe_public():\n    try:\n        antiguiding_onset(12.0, 0.2, 0.15, 4.0, 0.8, 5.0, 1.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\ndef probe_gold():\n    try:\n        _oracle_antiguiding_onset(12.0, 0.2, 0.15, 4.0, 0.8, 5.0, 1.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
                    "call": "probe_public()",
                    "gold_call": "probe_gold()"
            }
    ]
