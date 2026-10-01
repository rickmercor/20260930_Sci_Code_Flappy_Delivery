"""
Identify candidates that are not dominated in loss-complexity space.

Mechanism selection should compare predictive fit and interpretability rather than optimize one number blindly.

Returns
-------
Boolean array with one entry per candidate.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pareto_front(losses: np.ndarray, complexities: np.ndarray) -> np.ndarray:
    """Identify candidates that are not dominated in loss-complexity space.

    Parameters
    ----------
    losses : np.ndarray
        Concentration-trajectory losses for candidate mechanisms.
    complexities : np.ndarray
        Complexity scores for the same candidates.

    Returns
    -------
    np.ndarray
        Boolean mask marking Pareto candidates.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_pareto_front(losses: np.ndarray, complexities: np.ndarray) -> np.ndarray:
    """Reference nondominated-front calculation."""
    loss = np.asarray(losses, dtype=float)
    comp = np.asarray(complexities, dtype=float)

    if loss.ndim != 1 or comp.ndim != 1 or loss.shape != comp.shape or loss.size < 1:
        raise ValueError("losses and complexities must be matching one-dimensional arrays")
    if not np.all(np.isfinite(loss)) or not np.all(np.isfinite(comp)):
        raise ValueError("inputs must be finite")

    keep = np.ones(loss.size, dtype=bool)
    for i in range(loss.size):
        dominated = (
            (loss <= loss[i])
            & (comp <= comp[i])
            & ((loss < loss[i]) | (comp < comp[i]))
        )
        if np.any(dominated):
            keep[i] = False
    return keep

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
losses = np.array([0.40, 0.22, 0.18, 0.18])
complexities = np.array([3.0, 4.0, 7.0, 9.0])""",
            "call": "pareto_front(losses, complexities)",
            "gold_call": "_oracle_pareto_front(losses, complexities)",
        },
        {
            "setup": """import numpy as np
losses = np.array([0.1, 0.2, 0.3])
complexities = np.array([5.0, 4.0, 3.0])""",
            "call": "pareto_front(losses, complexities)",
            "gold_call": "_oracle_pareto_front(losses, complexities)",
        },
        {
            "setup": """import numpy as np
losses = np.array([0.5, 0.5, 0.4, 0.7])
complexities = np.array([2.0, 3.0, 4.0, 1.0])""",
            "call": "pareto_front(losses, complexities)",
            "gold_call": "_oracle_pareto_front(losses, complexities)",
        },
    ]
