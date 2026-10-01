"""
The system-centered probability vector F_p has entries indexed by the no-state outcome and richness values 0 through N. If a system has no retained state, all probability mass is assigned to the no-state entry. Otherwise, the no-state entry is zero and each richness entry is S_p^k/S_p, so every system with states contributes one normalized distribution regardless of its multiplicity.

Returns
-------
Array whose rows are (F_p^empty, F_p^0, ..., F_p^N).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_system_probability_vectors(system_summaries: np.ndarray) -> np.ndarray:
    '''Normalize richness counts within each ecological system.

    Parameters
    ----------
    system_summaries : np.ndarray
        Non-negative integer array of shape (M, N + 2), with each row equal
        to (S_p, S_p^0, ..., S_p^N).

    Raises
    ------
    ValueError
        If the input is not a two-dimensional non-negative integer array with
        at least one richness column, or S_p differs from the richness-count sum.

    Returns
    -------
    probability_vectors : np.ndarray
        Float array of shape (M, N + 2), with columns for no state followed by
        richness values 0 through N.
    '''
    return probability_vectors  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_build_system_probability_vectors(system_summaries: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    system_summaries = np.asarray(system_summaries)
    if system_summaries.ndim != 2 or system_summaries.shape[0] == 0:
        raise ValueError("system_summaries must be a non-empty two-dimensional array")
    if system_summaries.shape[1] < 3:
        raise ValueError("system_summaries must include S_p and richness counts")
    if not np.all(np.isfinite(system_summaries)):
        raise ValueError("system_summaries must be finite")
    if not np.all(system_summaries == np.floor(system_summaries)):
        raise ValueError("system_summaries must contain integers")
    if np.any(system_summaries < 0):
        raise ValueError("system_summaries must be non-negative")
    summaries = system_summaries.astype(int)
    if not np.array_equal(summaries[:, 0], np.sum(summaries[:, 1:], axis=1)):
        raise ValueError("S_p must equal the sum of richness counts")

    probability_vectors = np.zeros(summaries.shape, dtype=float)
    has_states = summaries[:, 0] > 0
    probability_vectors[~has_states, 0] = 1.0
    probability_vectors[has_states, 1:] = (
        summaries[has_states, 1:] / summaries[has_states, 0, None]
    )
    return probability_vectors

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": '''system_summaries = np.array([
    [3, 0, 2, 1],
    [0, 0, 0, 0],
    [1, 0, 0, 1],
])
''',
            "call": "build_system_probability_vectors(system_summaries).tolist()",
            "gold_call": "_oracle_build_system_probability_vectors(system_summaries).tolist()",
        },
        {
            "setup": "system_summaries = np.array([[0, 0, 0]])\n",
            "call": "build_system_probability_vectors(system_summaries).tolist()",
            "gold_call": "_oracle_build_system_probability_vectors(system_summaries).tolist()",
        },
        {
            "setup": '''system_summaries = np.array([[2, 0, 1, 0]])
def run_model():
    try:
        build_system_probability_vectors(system_summaries)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_build_system_probability_vectors(system_summaries)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
''',
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
