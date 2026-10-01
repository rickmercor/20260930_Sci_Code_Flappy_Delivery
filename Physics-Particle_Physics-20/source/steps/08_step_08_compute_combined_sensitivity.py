"""
Run the full multi-channel pipeline: generate the channel ensemble, evaluate each channel at its Asimov data, compute the higher-order corrected significance and the posterior signal probability per channel, select channels by the Bayesian criterion, and combine the selected significances in quadrature.

The combined sensitivity of the ensemble is obtained channel by channel. For channel i with parameters (s_i, b_i, tau_i) the Asimov counts are n_i = s_i + b_i and m_i = tau_i * b_i. The channel significance Z_i is the higher-order corrected significance of the frequentist test of s = 0, and the channel enters the combination if and only if its posterior signal probability satisfies P_i > threshold (strict inequality). The combined sensitivity is the quadrature sum Z_comb = sqrt( sum over selected channels of Z_i**2 ); if no channel is selected, Z_comb = 0.

Returns
-------
float, the combined sensitivity Z_comb over the selected channels
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_combined_sensitivity(n_channels: int, seed: int, threshold: float) -> float:
    '''Combined multi-channel sensitivity Z_comb of the ensemble.

    Parameters
    ----------
    n_channels : int
        Number of on/off channels; must be a positive integer.
    seed : int
        Seed of the single numpy.random.default_rng draw; must be an
        integer (booleans are not accepted).
    threshold : float
        Selection threshold on the posterior signal probability; must be
        finite and satisfy 0 < threshold < 1. Selection uses the strict
        inequality P > threshold.

    Raises
    ------
    ValueError
        If n_channels is not a positive integer, or seed is not an
        integer, or either of those arguments is a boolean, or threshold
        is non-finite or not strictly between 0 and 1.

    Returns
    -------
    z_comb : float
        The combined sensitivity Z_comb = sqrt(sum of squared channel
        significances over the selected channels), 0.0 if no channel is
        selected, as a native Python float.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy import special


def _oracle_compute_combined_sensitivity(n_channels: int, seed: int, threshold: float) -> float:
    if isinstance(n_channels, bool) or not isinstance(n_channels, (int, np.integer)):
        raise ValueError("n_channels must be a positive integer")
    if n_channels < 1:
        raise ValueError("n_channels must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    t = float(threshold)
    if not np.isfinite(t) or not (0.0 < t < 1.0):
        raise ValueError("threshold must be finite and strictly between 0 and 1")
    params = _oracle_generate_channel_parameters(n_channels, seed)
    counts = _oracle_compute_asimov_counts(params)
    total = 0.0
    for i in range(params.shape[0]):
        tau_i = float(params[i, 2])
        n_i, m_i = float(counts[i, 0]), float(counts[i, 1])
        p_i = _oracle_compute_signal_probability(n_i, m_i, tau_i)
        if p_i > t:
            r0_i = _oracle_compute_signed_root(n_i, m_i, tau_i)
            u0_i = _oracle_compute_auxiliary_statistic(n_i, m_i, tau_i)
            z_i = _oracle_compute_corrected_significance(r0_i, u0_i)
            total += z_i * z_i
    return float(np.sqrt(total))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: normal, the full task configuration ---
        {
            "setup": """import numpy as np
n_channels, seed, threshold = 120, 21, 0.95
""",
            "call": "compute_combined_sensitivity(n_channels, seed, threshold)",
            "gold_call": "_oracle_compute_combined_sensitivity(n_channels, seed, threshold)",
        },
        # --- Valid: boundary, small ensemble with a different seed ---
        {
            "setup": """import numpy as np
n_channels, seed, threshold = 10, 3, 0.95
""",
            "call": "compute_combined_sensitivity(n_channels, seed, threshold)",
            "gold_call": "_oracle_compute_combined_sensitivity(n_channels, seed, threshold)",
        },
        # --- Valid: edge, threshold high enough to empty the selection ---
        {
            "setup": """import numpy as np
n_channels, seed, threshold = 5, 7, 0.999999
""",
            "call": "compute_combined_sensitivity(n_channels, seed, threshold)",
            "gold_call": "_oracle_compute_combined_sensitivity(n_channels, seed, threshold)",
        },
        # --- Invalid: threshold outside (0, 1) ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_combined_sensitivity(10, 3, 1.2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_combined_sensitivity(10, 3, 1.2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive n_channels ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_combined_sensitivity(-5, 3, 0.95)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_combined_sensitivity(-5, 3, 0.95)
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
