"""
Measure the symbolic complexity of a candidate reaction mechanism.

SISR penalizes extra kinetic terms, higher-order reactant products, and heavier stoichiometric expressions.

Returns
-------
Float mechanism complexity score.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def expression_complexity(reactions: np.ndarray) -> float:
    """Measure the symbolic complexity of a candidate reaction mechanism.

    Parameters
    ----------
    reactions : np.ndarray
        Integer rows in [reactants | products] form.

    Returns
    -------
    float
        Complexity score for the mechanism.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_expression_complexity(reactions: np.ndarray) -> float:
    """Reference expression-complexity score."""
    r = np.asarray(reactions, dtype=float)

    if r.ndim != 2 or r.shape[0] < 1 or r.shape[1] % 2 != 0:
        raise ValueError("reactions must have shape (n_reactions, 2*n_species)")
    if not np.all(np.isfinite(r)) or np.any(r < 0) or not np.all(r == np.floor(r)):
        raise ValueError("reaction coefficients must be nonnegative integers")

    n_species = r.shape[1] // 2
    reactants = r[:, :n_species]
    products = r[:, n_species:]

    order_penalty = np.maximum(np.sum(reactants, axis=1) - 1.0, 0.0)
    stoich_weight = np.sum(reactants + products, axis=1)
    return float(np.sum(1.0 + order_penalty + 0.25 * stoich_weight))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
reactions = np.array([[1, 0, 0, 1]])""",
            "call": "expression_complexity(reactions)",
            "gold_call": "_oracle_expression_complexity(reactions)",
        },
        {
            "setup": """import numpy as np
reactions = np.array([[1, 1, 0, 0, 0, 1], [1, 0, 0, 0, 1, 0]])""",
            "call": "expression_complexity(reactions)",
            "gold_call": "_oracle_expression_complexity(reactions)",
        },
        {
            "setup": """import numpy as np
reactions = np.array([[2, 0, 0, 1], [0, 1, 1, 0], [1, 1, 0, 2]])""",
            "call": "expression_complexity(reactions)",
            "gold_call": "_oracle_expression_complexity(reactions)",
        },
    ]
