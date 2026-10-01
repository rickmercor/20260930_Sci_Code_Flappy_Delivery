"""
Given the final weight assigned to the first stage, recover the tableau with that first weight among the admissible tableaux of the robust two-stage construction, and return it as the stage abscissa followed by the two final weights. A tableau is admissible only if its internal stage is evaluated inside the step it belongs to, strictly after the start of the step and no later than its end. Raise ValueError if b1 is not given as an integer or floating-point number (a string such as "0" must be rejected, not converted), if it is not finite, or if no admissible tableau has that first weight.

Order is raised by supplying the practical splitting calculation with increments assembled from several evaluations of the vector field, as an explicit Runge-Kutta method assembles its own. In the common-base construction every stage, including the final one, is applied to the factors and the row space held at the beginning of the step.

Returns
-------
np.ndarray of shape (3,), holding the stage abscissa followed by the two final weights, dtype float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def two_stage_tableau(b1: float) -> "np.ndarray":
    '''Recover the admissible two-stage tableau from its first final weight.

    Parameters
    ----------
    b1 : float
        Final weight assigned to the first stage.

    Returns
    -------
    tableau : np.ndarray
        (3,) array holding the stage abscissa, the first final weight and the
        second final weight, in that order, float64.
    '''
    return tableau  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_two_stage_tableau(b1: float) -> "np.ndarray":
    if isinstance(b1, bool) or not isinstance(b1, (int, float, np.integer, np.floating)):
        raise ValueError("b1 must be a real number")
    v = float(b1)
    if not np.isfinite(v):
        raise ValueError("b1 must be finite")
    b2 = 1.0 - v
    if b2 == 0.0:
        raise ValueError("b1 = 1 leaves no second-stage weight; no admissible tableau")
    a = 0.5 / b2
    if not (0.0 < a <= 1.0):
        raise ValueError("no admissible tableau: the stage abscissa must satisfy 0 < a <= 1")
    return np.array([a, v, b2], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the production tableau, with a negative first weight ---
        {
            "setup": """import numpy as np
b1 = -1.0
""",
            "call": "two_stage_tableau(b1)",
            "gold_call": "_oracle_two_stage_tableau(b1)",
        },
        # --- normal: the member printed as the reference algorithm ---
        {
            "setup": """import numpy as np
b1 = 0.0
""",
            "call": "two_stage_tableau(b1)",
            "gold_call": "_oracle_two_stage_tableau(b1)",
        },
        # --- boundary: the largest admissible first weight, abscissa exactly one ---
        {
            "setup": """import numpy as np
b1 = 0.5
""",
            "call": "two_stage_tableau(b1)",
            "gold_call": "_oracle_two_stage_tableau(b1)",
        },
        # --- edge: first weight one ulp beyond the boundary ---
        {
            "setup": """import numpy as np
b1 = 0.5 + 1e-16
def run_model():
    try:
        two_stage_tableau(b1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_two_stage_tableau(b1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- edge: extreme negative weight, abscissa near zero ---
        {
            "setup": """import numpy as np
b1 = -1e6
""",
            "call": "two_stage_tableau(b1)",
            "gold_call": "_oracle_two_stage_tableau(b1)",
        },
        # --- edge: abscissa with no exact binary representation ---
        {
            "setup": """import numpy as np
b1 = -0.5
""",
            "call": "two_stage_tableau(b1)",
            "gold_call": "_oracle_two_stage_tableau(b1)",
        },
        # --- boundary: acceptance across the family, as integers, with the
        #     order conditions verified on each accepted member ---
        {
            "setup": """import numpy as np
vals = [-1e6, -3.0, -1.0, -0.5, 0.0, 0.25, 0.5, 0.5 + 1e-16, 0.75, 1.0, 2.0]
def probe(fn):
    out = []
    for v in vals:
        try:
            t = np.asarray(fn(v), dtype=float)
            ok = (abs(t[1] + t[2] - 1.0) < 1e-12
                  and abs(t[0]*t[2] - 0.5) < 1e-12
                  and 0.0 < t[0] <= 1.0)
            out.append(1 if ok else 0)
        except ValueError:
            out.append(-1)
        except Exception:
            out.append(-2)
    return out
def run_model():
    return probe(two_stage_tableau)
def run_gold():
    return probe(_oracle_two_stage_tableau)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: first weight leaving no second-stage weight ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        two_stage_tableau(1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_two_stage_tableau(1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: non-finite input ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        two_stage_tableau(np.nan)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_two_stage_tableau(np.nan)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: non-numeric input ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        two_stage_tableau("0")
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_two_stage_tableau("0")
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
