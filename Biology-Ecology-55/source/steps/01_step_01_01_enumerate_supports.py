"""
A generalized Lotka-Volterra system with N species can have an equilibrium on any subset of the species pool. Exhaustive state characterization therefore begins with the 2^N binary support masks, including the all-zero extinction support. Ordering masks first by richness and then lexicographically makes the enumeration deterministic while preserving the one-to-one relation between a row and a candidate subcommunity.

Returns
-------
Binary array of shape (2^N, N), ordered by richness and then lexicographically.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def enumerate_supports(n_species: int) -> np.ndarray:
    '''Enumerate every species support, including extinction.

    Parameters
    ----------
    n_species : int
        Positive number of species in the system.

    Raises
    ------
    ValueError
        If n_species is not a positive integer.

    Returns
    -------
    support_masks : np.ndarray
        Binary array of shape (2**n_species, n_species), ordered by richness
        and then lexicographically.
    '''
    return support_masks  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from itertools import combinations

import numpy as np  # noqa: E402, F811


def _oracle_enumerate_supports(n_species: int) -> np.ndarray:
    """Reference implementation."""
    if isinstance(n_species, (bool, np.bool_)) or not isinstance(
        n_species, (int, np.integer)
    ):
        raise ValueError("n_species must be a positive integer")
    n_species = int(n_species)
    if n_species <= 0:
        raise ValueError("n_species must be a positive integer")

    rows = []
    for richness in range(n_species + 1):
        for support in combinations(range(n_species), richness):
            mask = np.zeros(n_species, dtype=int)
            mask[list(support)] = 1
            rows.append(mask)
    return np.vstack(rows)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "n_species = 4\n",
            "call": "enumerate_supports(n_species).tolist()",
            "gold_call": "_oracle_enumerate_supports(n_species).tolist()",
        },
        {
            "setup": "n_species = 1\n",
            "call": "enumerate_supports(n_species).tolist()",
            "gold_call": "_oracle_enumerate_supports(n_species).tolist()",
        },
        {
            "setup": '''def run_model():
    try:
        enumerate_supports(0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_enumerate_supports(0)
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
