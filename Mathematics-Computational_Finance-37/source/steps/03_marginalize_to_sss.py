"""
Marginalize the calibrated (S1,V1,S2,V2,S3) joint law over V1 and V2 to obtain the (S1,S2,S3) joint law the downstream contract valuation step consumes.

The calibrated (S1,V1,S2,V2,S3) joint law from the previous step carries the volatility-index coordinates V1 and V2 only because they were needed during calibration to enforce the conditional martingale/dispersion constraints at each transition date. Once calibration is complete, the equity-index contract this joint law will drive (the surrenderable contract's fund) depends only on the index path (S1,S2,S3); V1 and V2 play no further role. Marginalizing the calibrated tensor over V1 and V2 discards those two axes by summation, leaving the (S1,S2,S3) joint law that the downstream contract valuation step actually consumes.

Returns
-------
joint123 : np.ndarray, (nS, nS, nS) joint law over (S1, S2, S3), summing pi over its V1 (axis 1) and V2 (axis 3) axes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def marginalize_to_sss(pi: np.ndarray) -> np.ndarray:
    """Marginalize the calibrated (S1,V1,S2,V2,S3) joint law over V1 and V2
    to obtain the joint law over (S1,S2,S3) alone.

    Parameters
    ----------
    pi : np.ndarray
        (nS, nV, nS, nV, nS) calibrated probability tensor over
        (S1, V1, S2, V2, S3).

    Returns
    -------
    joint123 : np.ndarray
        (nS, nS, nS) joint law over (S1, S2, S3), summing pi over its V1
        (axis 1) and V2 (axis 3) axes.

    Raises
    ------
    ValueError
        If pi is not a 5D array, or if its S1, S2, S3 axes (0, 2, 4) do not
        all share the same length.
    """
    nS = pi.shape[0] if pi.ndim >= 1 else 0
    joint123 = np.zeros((nS, nS, nS))  # placeholder
    return joint123

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_marginalize_to_sss(pi: np.ndarray) -> np.ndarray:
    import numpy as np
    pi = np.asarray(pi, dtype=float)
    if pi.ndim != 5:
        raise ValueError("pi must be a 5D array (nS, nV, nS, nV, nS)")
    if pi.shape[0] != pi.shape[2] or pi.shape[0] != pi.shape[4]:
        raise ValueError("pi's S1, S2, S3 axes (0, 2, 4) must share the same length")
    return pi.sum(axis=(1, 3))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal case: a small synthetic 2x2 grid tensor with a
        # non-trivial (>1) volatility axis, so summing over V1/V2 actually
        # aggregates multiple values rather than being a no-op reshape ---
        {
            "setup": """import numpy as np
pi = np.array([0.026875750897640258, 0.06822009068755779, 0.05252544613711316, 0.0429577379949394, 0.011195377757724962, 0.01119364697927388, 0.004167886464797783, 0.06215391397580111, 0.043134010226088614, 0.05080892876539047, 0.0014770747196599795, 0.06959749908793092, 0.05973331006455814, 0.015236747032542896, 0.013047153775323945, 0.01316049649232231, 0.02183140956830895, 0.037654772956993016, 0.030994935224103643, 0.020897632675997223, 0.043904524928236745, 0.010009614623319594, 0.020963326503644797, 0.026288905095302344, 0.03272606236530567, 0.05634161065083038, 0.014327925260964476, 0.036899749784680455, 0.04250969543627922, 0.0033331268361774602, 0.04359539413053129, 0.01223624290065907]).reshape(2, 2, 2, 2, 2)
def run_model():
    return marginalize_to_sss(pi).ravel()
def run_gold():
    return _oracle_marginalize_to_sss(pi).ravel()
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
            "tol": 1e-9,
        },
        # --- Boundary case: nS=1 (single index level) -- joint123 must
        # collapse to the single value [[[1.0]]] ---
        {
            "setup": """import numpy as np
pi = np.ones((1, 3, 1, 3, 1))
pi = pi / pi.sum()
def run_model():
    return marginalize_to_sss(pi).ravel()
def run_gold():
    return _oracle_marginalize_to_sss(pi).ravel()
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
            "tol": 1e-9,
        },
        # --- Edge case: wrong number of dimensions should raise ---
        {
            "setup": """import numpy as np
pi = np.ones((2, 2, 2))
def run_model():
    try:
        marginalize_to_sss(pi)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_marginalize_to_sss(pi)
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
