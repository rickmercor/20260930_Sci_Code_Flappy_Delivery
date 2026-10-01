"""
Select the number of internal stages for each stabilized explicit diffusion sub-step.

For each supplied abscissa increment, the corresponding sub-step advances over an interval of length H = delta_c h, and the selected count is the smallest number of internal stages for which the second-order Runge-Kutta-Legendre method is linearly stable over the whole segment of the negative real axis reaching out to H lam_eff. The method requires at least two internal stages, so the selected count is never below 2.

The sub-step lengths are not the full step size: each entry of delta_c is the abscissa increment of one super-time-stepping stage, and the counts are returned in the order the increments are supplied.

The eigenvalue magnitude passed here is the safeguarded one; the estimate has already been multiplied by its safety factor before reaching this function.

The function raises ValueError when h is not a finite positive scalar; when lam_eff is not a finite scalar; when lam_eff is negative; when delta_c is not a non-empty one-dimensional array; when delta_c contains a non-finite value; or when any entry of delta_c is not positive.

The defining property of super-time-stepping methods is that stability along the negative real axis is bought with stages rather than with implicitness. For a classical explicit Runge-Kutta method the stable step size shrinks in proportion to the reciprocal of the dominant eigenvalue magnitude, which for a diffusion operator means proportional to the square of the grid spacing. For a stabilized method the stability interval instead grows quadratically in the stage count, so doubling the number of right-hand-side evaluations quadruples the reachable step size. The work required to cross a fixed time interval then scales like the square root of the stiffness rather than linearly with it, which is what makes these methods competitive with implicit solvers on parabolic problems while requiring no linear or nonlinear solves.

Different stabilized families realize this scaling with different constants and different damping behaviour. Chebyshev-based constructions extend their intervals slightly further for a given stage count but require a damping parameter to keep the stability function bounded away from one at interior points, and their stage-count rules carry the corresponding constants. Legendre-based constructions have an exactly quadratic interval with no damping parameter, so the admissible count follows from inverting a simple quadratic. The rules are not interchangeable: applying one family's stage-count rule while running the other family's recurrence gives a scheme whose stability was never established.

Two features of the selection deserve attention. First, the count is an integer obtained by rounding upward, so nearby eigenvalue estimates can select the same count or adjacent counts depending on where they fall relative to an integer boundary; the resulting scheme is a different member of a discrete family in each case. Second, when the method is embedded as an inner solver inside a partitioned outer method, the relevant interval length is the outer stage's abscissa increment times the outer step size, not the outer step size itself. Stages with different increments therefore receive different counts within the same step.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_rkl_stage_counts(h: float, delta_c: np.ndarray, lam_eff: float) -> np.ndarray:
    '''Select minimum stable Runge-Kutta-Legendre stage counts for diffusion sub-steps.

    Parameters
    ----------
    h : float
        Outer step size, positive.
    delta_c : np.ndarray
        Abscissa increments of the super-time-stepping stages, shape (m,), every
        entry positive. Stage k advances over an interval of length delta_c[k] * h.
    lam_eff : float
        Safeguarded dominant eigenvalue magnitude of the diffusion operator,
        non-negative.

    Returns
    -------
    counts : np.ndarray
        Array of shape (m,) holding the selected stage count for each sub-step,
        as floats, in the order the abscissa increments were supplied.
    '''
    return counts  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_select_rkl_stage_counts(h: float, delta_c: np.ndarray,
                                    lam_eff: float) -> np.ndarray:
    """Reference implementation."""
    delta_c = np.asarray(delta_c, dtype=float)
    if not (np.isscalar(h) or np.ndim(h) == 0) or not np.isfinite(float(h)) \
            or float(h) <= 0.0:
        raise ValueError("h must be a finite positive scalar")
    if not (np.isscalar(lam_eff) or np.ndim(lam_eff) == 0) \
            or not np.isfinite(float(lam_eff)):
        raise ValueError("lam_eff must be a finite scalar")
    if float(lam_eff) < 0.0:
        raise ValueError("lam_eff must be a non-negative magnitude")
    if delta_c.ndim != 1 or delta_c.size < 1:
        raise ValueError("delta_c must be a non-empty 1D array")
    if not np.isfinite(delta_c).all():
        raise ValueError("delta_c must contain only finite values")
    if np.any(delta_c <= 0.0):
        raise ValueError("every entry of delta_c must be positive")
    h = float(h)
    lam_eff = float(lam_eff)

    counts = []
    for dc in delta_c:
        arg = 0.5 * (np.sqrt(9.0 + 8.0 * float(dc) * h * lam_eff) - 1.0)
        counts.append(max(2, int(np.ceil(arg))))

    return np.array(counts, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: locked configuration, both super-time-stepping stages ---
        {
            "setup": """import numpy as np
gamma = (2.0 - np.sqrt(2.0)) / 2.0
h = 1.5 / 73.0
delta_c = np.array([2.0 * gamma, 1.0 - 2.0 * gamma])
lam_eff = 3288.763196088486
""",
            "call": "select_rkl_stage_counts(h, delta_c, lam_eff)",
            "gold_call": "_oracle_select_rkl_stage_counts(h, delta_c, lam_eff)",
        },
        # --- Normal: same increments, halved step size ---
        {
            "setup": """import numpy as np
gamma = (2.0 - np.sqrt(2.0)) / 2.0
h = 1.5 / 146.0
delta_c = np.array([2.0 * gamma, 1.0 - 2.0 * gamma])
lam_eff = 3288.763196088486
""",
            "call": "select_rkl_stage_counts(h, delta_c, lam_eff)",
            "gold_call": "_oracle_select_rkl_stage_counts(h, delta_c, lam_eff)",
        },
        # --- Normal: unsafeguarded magnitude on the locked step size ---
        {
            "setup": """import numpy as np
gamma = (2.0 - np.sqrt(2.0)) / 2.0
h = 1.5 / 73.0
delta_c = np.array([2.0 * gamma, 1.0 - 2.0 * gamma])
lam_eff = 2989.7847237168053
""",
            "call": "select_rkl_stage_counts(h, delta_c, lam_eff)",
            "gold_call": "_oracle_select_rkl_stage_counts(h, delta_c, lam_eff)",
        },
        # --- Boundary: product places the count argument exactly on an integer ---
        {
            "setup": """import numpy as np
h = 1.0
delta_c = np.array([1.0])
lam_eff = 14.0
""",
            "call": "select_rkl_stage_counts(h, delta_c, lam_eff)",
            "gold_call": "_oracle_select_rkl_stage_counts(h, delta_c, lam_eff)",
        },
        # --- Boundary: same product perturbed just above the integer ---
        {
            "setup": """import numpy as np
h = 1.0
delta_c = np.array([1.0])
lam_eff = 14.0000001
""",
            "call": "select_rkl_stage_counts(h, delta_c, lam_eff)",
            "gold_call": "_oracle_select_rkl_stage_counts(h, delta_c, lam_eff)",
        },
        # --- Edge: zero eigenvalue magnitude activates the two-stage floor ---
        {
            "setup": """import numpy as np
h = 0.01
delta_c = np.array([0.5, 0.25])
lam_eff = 0.0
""",
            "call": "select_rkl_stage_counts(h, delta_c, lam_eff)",
            "gold_call": "_oracle_select_rkl_stage_counts(h, delta_c, lam_eff)",
        },
        # --- Edge: single increment with a large eigenvalue magnitude ---
        {
            "setup": """import numpy as np
h = 0.02
delta_c = np.array([1.0])
lam_eff = 1.0e6
""",
            "call": "select_rkl_stage_counts(h, delta_c, lam_eff)",
            "gold_call": "_oracle_select_rkl_stage_counts(h, delta_c, lam_eff)",
        },
        # --- Edge: several increments of differing size in one call ---
        {
            "setup": """import numpy as np
h = 0.05
delta_c = np.array([0.1, 0.3, 0.6, 1.0])
lam_eff = 4210.5
""",
            "call": "select_rkl_stage_counts(h, delta_c, lam_eff)",
            "gold_call": "_oracle_select_rkl_stage_counts(h, delta_c, lam_eff)",
        },
        # --- Invalid: non-positive step size ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        select_rkl_stage_counts(0.0, np.array([0.5]), 100.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_select_rkl_stage_counts(0.0, np.array([0.5]), 100.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative eigenvalue magnitude ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        select_rkl_stage_counts(0.1, np.array([0.5]), -5.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_select_rkl_stage_counts(0.1, np.array([0.5]), -5.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: zero abscissa increment ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        select_rkl_stage_counts(0.1, np.array([0.5, 0.0]), 100.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_select_rkl_stage_counts(0.1, np.array([0.5, 0.0]), 100.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: empty increment array ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        select_rkl_stage_counts(0.1, np.array([]), 100.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_select_rkl_stage_counts(0.1, np.array([]), 100.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-finite eigenvalue magnitude ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        select_rkl_stage_counts(0.1, np.array([0.5]), np.inf)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_select_rkl_stage_counts(0.1, np.array([0.5]), np.inf)
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
