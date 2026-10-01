"""
Apply the source second-order jackknife correction to an ordered estimator triplet.

A nonlinear estimator can contain a leading finite-sample bias. The source jackknife-2 construction combines the estimate from a complete sample block with the estimates from its two consecutive halves. Return both the corrected estimate and the inferred leading-bias contribution.

Returns
-------
Return a two-element tuple containing the bias-corrected estimate followed by the leading-bias estimate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def jackknife2_mtp(estimates: np.ndarray) -> tuple[float, float]:
    """Return the source bias-corrected estimate and its leading-bias estimate from the ordered three-value input."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_jackknife2_mtp(estimates: np.ndarray) -> tuple[float, float]:
    import numpy as np

    values = np.asarray(estimates, dtype=float)
    if values.shape != (3,) or np.any(~np.isfinite(values)):
        raise ValueError("three finite estimates are required")
    full, first_half, second_half = map(float, values)
    half_average = 0.5 * (first_half + second_half)
    leading_bias = half_average - full
    corrected = full - leading_bias
    return float(corrected), float(leading_bias)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nq=np.array([12.,14.,16.])",
            "call": "np.asarray(jackknife2_mtp(q))",
            "gold_call": "np.asarray(_oracle_jackknife2_mtp(q))",
        },
        {
            "setup": "import numpy as np\nq=np.array([5.,5.,5.])",
            "call": "np.asarray(jackknife2_mtp(q))",
            "gold_call": "np.asarray(_oracle_jackknife2_mtp(q))",
        },
        {
            "setup": "import numpy as np\nq=np.array([1e-8,4e-8,2e-8])",
            "call": "np.asarray(jackknife2_mtp(q))",
            "gold_call": "np.asarray(_oracle_jackknife2_mtp(q))",
        },
    ]
