"""
Enumerate conditional partner modes along a one-dimensional factor.

At a fixed factor value f, conditional independence makes the modal partner

allele x_j*(f) = 1{b_j f > tau_j}. Every nonzero loading contributes one

breakpoint t_j = tau_j / b_j. Sorting the distinct breakpoints partitions the

factor line into intervals on which the full modal configuration is constant,

so one interior representative per interval enumerates the candidate path.

Inputs

------

partner_loadings : one-factor loadings for partner variants

partner_thresholds : fixed thresholds for the same partners

Returns

-------

mode_candidates : unique binary modal configurations in lexicographic order

Returns
-------
np.ndarray of shape (m, k), unique binary factor-mode candidates as uint8
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def enumerate_factor_modes(
    partner_loadings: np.ndarray, partner_thresholds: np.ndarray
) -> np.ndarray:
    '''Enumerate every partner configuration that is modal on a factor interval.

    Parameters
    ----------
    partner_loadings : np.ndarray
        One-dimensional partner loadings.
    partner_thresholds : np.ndarray
        One-dimensional partner thresholds in the same order.

    Returns
    -------
    mode_candidates : np.ndarray
        Binary array with one unique modal configuration per row.

    Raises
    ------
    ValueError
        If `partner_loadings` is not a nonempty finite one-dimensional vector,
        or if `partner_thresholds` is not finite with the same shape as
        `partner_loadings`.
    '''
    return mode_candidates  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_enumerate_factor_modes(
    partner_loadings: np.ndarray, partner_thresholds: np.ndarray
) -> np.ndarray:
    """Reference implementation of one-factor breakpoint enumeration."""
    loadings = np.asarray(partner_loadings, dtype=float)
    thresholds = np.asarray(partner_thresholds, dtype=float)
    if loadings.ndim != 1 or loadings.size < 1 or not np.isfinite(loadings).all():
        raise ValueError("partner_loadings must be a nonempty finite vector")
    if thresholds.shape != loadings.shape or not np.isfinite(thresholds).all():
        raise ValueError("partner_thresholds must be finite and match partner_loadings")

    active = loadings != 0.0
    if np.any(active):
        breakpoints = np.unique(np.sort(thresholds[active] / loadings[active]))
        representatives = [float(breakpoints[0] - 1.0)]
        representatives.extend(
            float(0.5 * (left + right))
            for left, right in zip(breakpoints[:-1], breakpoints[1:])
        )
        representatives.append(float(breakpoints[-1] + 1.0))
    else:
        representatives = [0.0]
    rows = [
        (loadings * factor > thresholds).astype(np.uint8)
        for factor in representatives
    ]
    return np.unique(np.asarray(rows, dtype=np.uint8), axis=0)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """partner_loadings = np.array([0.8619342151577695, -0.8849182223819824, 0.7808688094430303, -0.8402966482242443, 0.6476484200955405, -0.7546055221635047, 0.8232127859153063])
partner_thresholds = np.array([0.7332357810248108, -0.5813930077385457, 0.5104216363344174, -0.7332357810248108, 1.2278262639421003, -0.18445243845167293, 0.31060942561200097])
""",
            "call": "enumerate_factor_modes(partner_loadings, partner_thresholds).tolist()",
            "gold_call": "_oracle_enumerate_factor_modes(partner_loadings, partner_thresholds).tolist()",
        },
        {
            "setup": """partner_loadings = np.array([0.0, 0.0])
partner_thresholds = np.array([-0.2, 0.3])
""",
            "call": "enumerate_factor_modes(partner_loadings, partner_thresholds).tolist()",
            "gold_call": "_oracle_enumerate_factor_modes(partner_loadings, partner_thresholds).tolist()",
        },
        {
            "setup": """partner_loadings = np.array([0.5, -0.5])
partner_thresholds = np.array([0.0])
def run_model():
    try:
        enumerate_factor_modes(partner_loadings, partner_thresholds)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_enumerate_factor_modes(partner_loadings, partner_thresholds)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        {
            "setup": """partner_loadings = np.array([1.0, 2.0, -1.0, 0.0])
partner_thresholds = np.array([0.5, 1.0, -0.5, -0.2])
""",
            "call": "enumerate_factor_modes(partner_loadings, partner_thresholds).tolist()",
            "gold_call": "_oracle_enumerate_factor_modes(partner_loadings, partner_thresholds).tolist()",
        },
    ]
