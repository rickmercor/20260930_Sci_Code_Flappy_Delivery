"""
Compute the aligned net moiety-count change between the supplied products and reactants.

reactant_moiety_counts and product_moiety_counts are aligned one-dimensional arrays of non-negative integer-valued moiety counts. Compute product counts minus reactant counts and preserve moiety order. Raise ValueError if either array is empty, misaligned, non-finite, negative, or non-integer-valued.

Returns
-------
1D NumPy float array aligned with the moiety rows and containing product counts minus reactant counts.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_overall_moiety_change(
    reactant_moiety_counts: np.ndarray,
    product_moiety_counts: np.ndarray,
) -> np.ndarray:
    """Return the aligned product-minus-reactant moiety change."""
    return np.empty(0, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_overall_moiety_change(reactant_moiety_counts, product_moiety_counts):
    import numpy as np

    reactants = np.asarray(reactant_moiety_counts, dtype=float)
    products = np.asarray(product_moiety_counts, dtype=float)

    if reactants.ndim != 1 or reactants.size < 1:
        raise ValueError('reactant_moiety_counts must be a non-empty one-dimensional array')
    if products.shape != reactants.shape:
        raise ValueError('product_moiety_counts must align with reactant_moiety_counts')
    if not np.all(np.isfinite(reactants)) or not np.all(np.isfinite(products)):
        raise ValueError('moiety counts must contain only finite values')
    if np.any(reactants < 0.0) or np.any(products < 0.0):
        raise ValueError('moiety counts must be non-negative')
    if not np.all(reactants == np.floor(reactants)) or not np.all(products == np.floor(products)):
        raise ValueError('moiety counts must be integer-valued')

    return products - reactants

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential tests for this public function."""
    return [
        {
            "setup": """import numpy as np
reactant_moiety_counts = np.array([2, 0, 1, 3], dtype=float)
product_moiety_counts = np.array([0, 2, 1, 1], dtype=float)
""",
            "call": 'compute_overall_moiety_change(reactant_moiety_counts.copy(), product_moiety_counts.copy())',
            "gold_call": '_oracle_compute_overall_moiety_change(reactant_moiety_counts.copy(), product_moiety_counts.copy())',
        },
        {
            "setup": """import numpy as np
reactant_moiety_counts = np.array([0, 5, 0], dtype=int)
product_moiety_counts = np.array([4, 0, 2], dtype=int)
""",
            "call": 'compute_overall_moiety_change(reactant_moiety_counts.copy(), product_moiety_counts.copy())',
            "gold_call": '_oracle_compute_overall_moiety_change(reactant_moiety_counts.copy(), product_moiety_counts.copy())',
        },
        {
            "setup": """import numpy as np
reactant_moiety_counts = np.array([3, 1, 0, 2, 4], dtype=float)
product_moiety_counts = np.array([1, 4, 2, 2, 0], dtype=float)
order = np.array([4, 1, 3, 0, 2], dtype=int)
reactant_moiety_counts = reactant_moiety_counts[order]
product_moiety_counts = product_moiety_counts[order]
""",
            "call": 'compute_overall_moiety_change(reactant_moiety_counts.copy(), product_moiety_counts.copy())',
            "gold_call": '_oracle_compute_overall_moiety_change(reactant_moiety_counts.copy(), product_moiety_counts.copy())',
        },
        {
            "setup": """import numpy as np
reactant_moiety_counts = np.array([1.0, -1.0], dtype=float)
product_moiety_counts = np.array([0.0, 2.0], dtype=float)

def candidate_wrapper():
    try:
        compute_overall_moiety_change(reactant_moiety_counts, product_moiety_counts)
    except ValueError:
        return 1.0
    return 0.0

def gold_wrapper():
    try:
        _oracle_compute_overall_moiety_change(reactant_moiety_counts, product_moiety_counts)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": 'candidate_wrapper()',
            "gold_call": 'gold_wrapper()',
        },
    ]
