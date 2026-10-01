"""
For one parameterized ecological system p, the stable-state multiplicity S_p is the total number of stable equilibria, and S_p^k counts those equilibria whose support contains exactly k species. These counts retain the identity of the system that generated the states. The compact summary [S_p, S_p^0, ..., S_p^N] is the deterministic input to the system-centered probability construction.

Returns
-------
Integer vector [S_p, S_p^0, ..., S_p^N].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def summarize_system_richness(
    support_masks: np.ndarray,
    stable_flags: np.ndarray,
) -> np.ndarray:
    '''Count stable equilibria by support richness for one GLV system.

    Parameters
    ----------
    support_masks : np.ndarray
        Binary matrix with one species support per row.
    stable_flags : np.ndarray
        Binary vector marking the stable states.

    Raises
    ------
    ValueError
        If support_masks is not a nonempty binary matrix, or stable_flags does
        not contain one binary value per support.

    Returns
    -------
    system_summary : np.ndarray
        Integer vector [S_p, S_p^0, ..., S_p^N].
    '''
    return system_summary  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_summarize_system_richness(
    support_masks: np.ndarray,
    stable_flags: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    support_masks = np.asarray(support_masks)
    stable_flags = np.asarray(stable_flags)
    if support_masks.ndim != 2 or support_masks.shape[0] == 0 or support_masks.shape[1] == 0:
        raise ValueError("support_masks must be a nonempty two-dimensional array")
    if not np.all((support_masks == 0) | (support_masks == 1)):
        raise ValueError("support_masks must be binary")
    if stable_flags.shape != (support_masks.shape[0],):
        raise ValueError("stable_flags must have one entry per support")
    if not np.all((stable_flags == 0) | (stable_flags == 1)):
        raise ValueError("stable_flags must be binary")

    n_species = support_masks.shape[1]
    richness = np.sum(support_masks, axis=1, dtype=int)
    selected = stable_flags.astype(bool)
    counts = np.bincount(richness[selected], minlength=n_species + 1)
    multiplicity = int(np.sum(stable_flags))
    return np.concatenate((np.array([multiplicity], dtype=int), counts.astype(int)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": '''support_masks = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]],
    dtype=int,
)
stable_flags = np.array([0, 1, 1, 0, 1, 0], dtype=int)
''',
            "call": "summarize_system_richness(support_masks, stable_flags).tolist()",
            "gold_call": "_oracle_summarize_system_richness(support_masks, stable_flags).tolist()",
        },
        {
            "setup": '''support_masks = np.array([[0], [1]], dtype=int)
stable_flags = np.array([0, 0], dtype=int)
''',
            "call": "summarize_system_richness(support_masks, stable_flags).tolist()",
            "gold_call": "_oracle_summarize_system_richness(support_masks, stable_flags).tolist()",
        },
        {
            "setup": '''support_masks = np.array([[0, 0], [1, 0]], dtype=int)
stable_flags = np.array([1], dtype=int)
def run_model():
    try:
        summarize_system_richness(support_masks, stable_flags)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_summarize_system_richness(support_masks, stable_flags)
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
