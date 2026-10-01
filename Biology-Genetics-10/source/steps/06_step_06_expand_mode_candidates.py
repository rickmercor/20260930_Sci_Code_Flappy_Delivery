"""
Expand the deterministic factor-mode set around its leading configuration.

The one-factor breakpoint path contains only configurations that are modal at

some factor value. After these modes are scored for one lead state, the paper's

local repair selects the highest-probability mode and adjoins every configuration

obtained by flipping exactly one partner allele. The union removes duplicates and

is ordered lexicographically before final probability scoring.

Inputs

------

mode_candidates : binary configurations from factor-space enumeration

mode_probabilities : conditioned probabilities for those configurations

Returns

-------

expanded_candidates : mode set plus all one-allele neighbors of its leader

Returns
-------
np.ndarray of shape (m_expanded, k), unique expanded candidates as uint8
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def expand_mode_candidates(
    mode_candidates: np.ndarray, mode_probabilities: np.ndarray
) -> np.ndarray:
    '''Add every one-allele neighbor of the highest-probability factor mode.

    Parameters
    ----------
    mode_candidates : np.ndarray
        Unique binary factor-mode configurations.
    mode_probabilities : np.ndarray
        Conditional probabilities in corresponding row order.

    Returns
    -------
    expanded_candidates : np.ndarray
        Lexicographically ordered unique binary configurations.

    Raises
    ------
    ValueError
        If `mode_candidates` is not a nonempty two-dimensional binary array, or
        if `mode_probabilities` is not a finite vector with one nonnegative
        entry per candidate row.
    '''
    return expanded_candidates  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811

def _oracle_expand_mode_candidates(
    mode_candidates: np.ndarray, mode_probabilities: np.ndarray
) -> np.ndarray:
    """Reference implementation of the one-allele neighborhood expansion."""
    candidates = np.asarray(mode_candidates)
    probabilities = np.asarray(mode_probabilities, dtype=float)
    if candidates.ndim != 2 or candidates.shape[0] < 1 or candidates.shape[1] < 1:
        raise ValueError("mode_candidates must be a nonempty two-dimensional array")
    if not np.isin(candidates, (0, 1)).all():
        raise ValueError("mode_candidates must be binary")
    if probabilities.shape != (candidates.shape[0],) or not np.isfinite(probabilities).all():
        raise ValueError("mode_probabilities must be finite and match the candidate rows")
    if np.any(probabilities < 0.0):
        raise ValueError("mode_probabilities must be nonnegative")

    candidates = candidates.astype(np.uint8)
    leading_index = min(
        range(candidates.shape[0]),
        key=lambda index: (-float(probabilities[index]), tuple(int(v) for v in candidates[index])),
    )
    leading = candidates[leading_index]
    neighbors = np.repeat(leading[None, :], leading.size, axis=0)
    neighbors[np.arange(leading.size), np.arange(leading.size)] ^= 1
    return np.unique(np.vstack([candidates, neighbors]), axis=0).astype(np.uint8)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """mode_candidates = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.uint8)
mode_probabilities = np.array([0.1, 0.5, 0.2])
""",
            "call": "expand_mode_candidates(mode_candidates, mode_probabilities).tolist()",
            "gold_call": "_oracle_expand_mode_candidates(mode_candidates, mode_probabilities).tolist()",
        },
        {
            "setup": """mode_candidates = np.array([[0], [1]], dtype=np.uint8)
mode_probabilities = np.array([0.5, 0.5])
""",
            "call": "expand_mode_candidates(mode_candidates, mode_probabilities).tolist()",
            "gold_call": "_oracle_expand_mode_candidates(mode_candidates, mode_probabilities).tolist()",
        },
        {
            "setup": """mode_candidates = np.array([[0, 1], [1, 0]], dtype=np.uint8)
mode_probabilities = np.array([0.2, -0.1])
def run_model():
    try:
        expand_mode_candidates(mode_candidates, mode_probabilities)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_expand_mode_candidates(mode_candidates, mode_probabilities)
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
            "setup": """mode_candidates = np.array([[1, 1, 1], [0, 0, 0]], dtype=np.uint8)
mode_probabilities = np.array([0.5, 0.5])
""",
            "call": "expand_mode_candidates(mode_candidates, mode_probabilities).tolist()",
            "gold_call": "_oracle_expand_mode_candidates(mode_candidates, mode_probabilities).tolist()",
        },
    ]
