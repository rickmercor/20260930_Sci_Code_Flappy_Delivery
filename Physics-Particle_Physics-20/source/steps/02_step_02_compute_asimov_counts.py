"""
Replace each channel's observable counts by their expectation values under the nominal signal hypothesis (the Asimov data of the channel).

The median discovery significance of a channel is approximated by evaluating the test statistic on the data set in which every observed count is replaced by its expectation value under the assumed signal: the on-region count n by s + b and the control count m by tau*b. These Asimov counts are real-valued and are used in place of integer data in all subsequent channel-level computations. Each s, b and tau is restricted to at most 1e12, the supported domain of the counting model, so that n and m remain finite and exactly representable sums.

Returns
-------
numpy.ndarray of shape (N, 2), float64, columns [n, m] = [s + b, tau * b]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_asimov_counts(params):
    '''Compute the Asimov counts (n, m) for every channel.

    Parameters
    ----------
    params : numpy.ndarray
        Array of shape (N, 3) with columns [s, b, tau]; all entries must
        be finite, strictly positive and at most 1e12.

    Raises
    ------
    ValueError
        If params is not a two-dimensional array with exactly 3 columns,
        or contains a non-finite entry, or any s, b or tau is not
        strictly positive, or any entry exceeds 1e12, which is the
        supported domain of the counting model.

    Returns
    -------
    counts : numpy.ndarray
        Array of shape (N, 2), float64, columns [n, m] with n = s + b
        and m = tau * b.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_asimov_counts(params):
    p = np.asarray(params, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("params must be a 2-D array with 3 columns")
    if not np.all(np.isfinite(p)):
        raise ValueError("params must contain only finite values")
    if not np.all(p > 0.0):
        raise ValueError("all s, b, tau must be strictly positive")
    if np.any(p > 1.0e12):
        raise ValueError("all s, b, tau must not exceed 1e12")
    s, b, tau = p[:, 0], p[:, 1], p[:, 2]
    n = s + b
    m = tau * b
    return np.column_stack([n, m]).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: normal three-channel array ---
        {
            "setup": """import numpy as np
params = np.array([[4.0806938730, 3.7533280686, 2.2745029760],
                   [1.2, 0.4, 0.9],
                   [3.5, 5.1, 1.7]], dtype=float)
""",
            "call": "compute_asimov_counts(params.copy())",
            "gold_call": "_oracle_compute_asimov_counts(params.copy())",
        },
        # --- Valid: boundary, single channel at small yields ---
        {
            "setup": """import numpy as np
params = np.array([[0.8, 0.3, 0.5]], dtype=float)
""",
            "call": "compute_asimov_counts(params.copy())",
            "gold_call": "_oracle_compute_asimov_counts(params.copy())",
        },
        # --- Valid: edge, large exposure ratio ---
        {
            "setup": """import numpy as np
params = np.array([[2.0, 1.0, 3.0], [5.0, 6.0, 2.99]], dtype=float)
""",
            "call": "compute_asimov_counts(params.copy())",
            "gold_call": "_oracle_compute_asimov_counts(params.copy())",
        },
        # --- Invalid: entry beyond the supported domain ---
        {
            "setup": """import numpy as np
params = np.array([[1e200, 1e200, 1.0]], dtype=float)
def run_model():
    try:
        compute_asimov_counts(params.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_asimov_counts(params.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: wrong shape ---
        {
            "setup": """import numpy as np
params = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
def run_model():
    try:
        compute_asimov_counts(params.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_asimov_counts(params.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive background ---
        {
            "setup": """import numpy as np
params = np.array([[2.0, 0.0, 1.5]], dtype=float)
def run_model():
    try:
        compute_asimov_counts(params.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_asimov_counts(params.copy())
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
