"""
Carrier transport in real crystals is anisotropic, so the mean free path varies by crystallographic direction. A single representative value must weight each direction's contribution by how much carrier density actually flows along it, since directions carrying more current are more relevant to the effective transport behavior -- an unweighted average would misrepresent this transport-relevant length scale.

Carrier transport in real crystals is anisotropic, so the mean free path varies by crystallographic direction. A single representative value must weight each direction's contribution by how much carrier density actually flows along it, since directions carrying more current are more relevant to the effective transport behavior. An unweighted average would misrepresent this transport-relevant length scale, since it would treat all directions as equally important regardless of how much current they actually carry.

Returns
-------
float, the weighted average MFP
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def weighted_mfp_average(MFP: list[float], weights: list[float]) -> float:
    '''Compute the carrier-density-weighted average mean free path.

    Parameters
    ----------
    MFP : list[float]
        1D array of per-direction mean free path values (nm).
    weights : list[float]
        1D array of per-direction carrier-density weights, same length as MFP.

    Returns
    -------
    MFP_avg : float
        The weighted average MFP, as a native Python float.

    Raises
    ------
    ValueError
        If MFP and weights are not equal-length non-empty 1D arrays, if
        any weight is negative, or if the weights sum to zero.
    '''
    return MFP_avg  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_weighted_mfp_average(MFP: list[float], weights: list[float]) -> float:
    """Reference implementation."""
    MFP = np.asarray(MFP, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if MFP.ndim != 1 or weights.ndim != 1 or MFP.shape[0] != weights.shape[0]:
        raise ValueError("MFP and weights must be 1D arrays of the same length")
    if MFP.shape[0] < 1:
        raise ValueError("MFP must be non-empty")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    if np.sum(weights) == 0:
        raise ValueError("sum of weights must not be zero")
    return float(np.sum(MFP * weights) / np.sum(weights))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                "import numpy as np\n"
                "rng_mfp = np.random.default_rng(421)\n"
                "MFP = rng_mfp.uniform(20, 150, 6)\n"
                "rng_w = np.random.default_rng(433)\n"
                "weights = rng_w.uniform(0.5, 2.0, 6)"
            ),
            "call": "weighted_mfp_average(MFP, weights)",
            "gold_call": "_oracle_weighted_mfp_average(MFP, weights)",
        },
        {"setup": "MFP = [1.0, 2.0, 3.0]\nweights = [1.0, 1.0, 1.0]", "call": "weighted_mfp_average(MFP, weights)", "gold_call": "_oracle_weighted_mfp_average(MFP, weights)"},
        {"setup": "MFP = [50.0]\nweights = [1.0]", "call": "weighted_mfp_average(MFP, weights)", "gold_call": "_oracle_weighted_mfp_average(MFP, weights)"},
        {
            "setup": (
                "MFP = [1.0, 2.0]\nweights = [0.0, 0.0]\n"
                "def run_model():\n    try:\n        weighted_mfp_average(MFP, weights)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
                "def run_gold():\n    try:\n        _oracle_weighted_mfp_average(MFP, weights)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2"
            ),
            "call": "run_model()", "gold_call": "run_gold()",
        },
    ]
