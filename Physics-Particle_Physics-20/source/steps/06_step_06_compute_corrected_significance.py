"""
Combine the signed root r(0) and the auxiliary statistic u(0) into the higher-order corrected discovery significance of one channel.

The higher-order refinement replaces the signed root r(0) by a corrected root whose sampling distribution is more accurately standard normal at small event yields; the corrected discovery statistic is the square of the positive part of the corrected root, and the channel significance is Z = max{0, corrected root}. The rule by which r(0) and u(0) combine into the corrected root is the scheme's prescription and must be taken from the browsing sources; the corrected root reverts to r(0) in the large-sample limit u(0) -> r(0). Boundary rule: whenever r(0) = 0 or u(0) = 0 the correction is undefined and the corrected root is taken equal to r(0). No continuity or discreteness adjustment is applied in the two-measurement setting.

Returns
-------
float, the corrected channel significance Z = max{0, corrected root}
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_corrected_significance(r0: float, u0: float) -> float:
    '''Higher-order corrected discovery significance from r(0) and u(0).

    Parameters
    ----------
    r0 : float
        Signed likelihood-ratio root r(0); must be finite.
    u0 : float
        Auxiliary statistic u(0); must be finite.

    Raises
    ------
    ValueError
        If r0 or u0 is non-finite.

    Returns
    -------
    z : float
        The channel significance Z = max{0, corrected root}, where the
        corrected root equals r0 whenever r0 = 0 or u0 = 0, as a native
        Python float.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_corrected_significance(r0: float, u0: float) -> float:
    if not (np.isfinite(float(r0)) and np.isfinite(float(u0))):
        raise ValueError("r0 and u0 must be finite")
    r0, u0 = float(r0), float(u0)
    if r0 == 0.0 or u0 == 0.0:
        rstar = r0
    else:
        rstar = r0 + (1.0 / r0) * np.log(abs(u0 / r0))
    return float(max(0.0, rstar))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: normal, small-count channel (correction lowers r0) ---
        {
            "setup": """import numpy as np
r0, u0 = 2.8061669282, 2.2043944471
""",
            "call": "compute_corrected_significance(r0, u0)",
            "gold_call": "_oracle_compute_corrected_significance(r0, u0)",
        },
        # --- Valid: boundary, u0 = 0 invokes the boundary rule ---
        {
            "setup": """import numpy as np
r0, u0 = 1.7, 0.0
""",
            "call": "compute_corrected_significance(r0, u0)",
            "gold_call": "_oracle_compute_corrected_significance(r0, u0)",
        },
        # --- Valid: edge, negative corrected root is clipped to zero ---
        {
            "setup": """import numpy as np
r0, u0 = -1.0, -0.5
""",
            "call": "compute_corrected_significance(r0, u0)",
            "gold_call": "_oracle_compute_corrected_significance(r0, u0)",
        },
        # --- Valid: edge, u0 = r0 leaves the root unchanged ---
        {
            "setup": """import numpy as np
r0, u0 = 2.25, 2.25
""",
            "call": "compute_corrected_significance(r0, u0)",
            "gold_call": "_oracle_compute_corrected_significance(r0, u0)",
        },
        # --- Invalid: non-finite input ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_corrected_significance(float('inf'), 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_corrected_significance(float('inf'), 1.0)
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
