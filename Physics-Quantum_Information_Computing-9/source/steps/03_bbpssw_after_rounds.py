"""
Propagate a Werner-state fidelity through a specified number of ideal symmetric BBPSSW rounds.

Reconstruct the symmetric-input recurrence adopted by the source paper. The round count is exact, each iterate uses the preceding full-precision fidelity, and no stochastic success multiplier belongs in this planning calculation.

Returns
-------
return float(fidelity)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

def bbpssw_after_rounds(initial_fidelity, rounds):
    """Return the fidelity after exactly the requested BBPSSW rounds.

    Returns:
        float: The full-precision post-round fidelity.
    """
    return float(fidelity)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_bbpssw_after_rounds(initial_fidelity: float, rounds: int) -> float:
    import math
    fidelity = float(initial_fidelity)
    rounds = int(rounds)
    if not math.isfinite(fidelity) or fidelity <= 0.5 or fidelity > 1.0:
        raise ValueError("initial_fidelity must be finite and in (0.5, 1]")
    if rounds < 0:
        raise ValueError("rounds must be nonnegative")
    for _ in range(rounds):
        complement = 1.0 - fidelity
        fidelity = (fidelity * fidelity + complement * complement / 9.0) / (
            fidelity * fidelity
            + 2.0 * fidelity * complement / 3.0
            + 5.0 * complement * complement / 9.0
        )
    return float(fidelity)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "", "call": "bbpssw_after_rounds(0.91,0)", "gold_call": "_oracle_bbpssw_after_rounds(0.91,0)"},
        {"setup": "", "call": "bbpssw_after_rounds(0.91,1)", "gold_call": "_oracle_bbpssw_after_rounds(0.91,1)"},
        {"setup": "", "call": "bbpssw_after_rounds(0.84,2)", "gold_call": "_oracle_bbpssw_after_rounds(0.84,2)"},
        {"setup": "", "call": "bbpssw_after_rounds(0.73,4)", "gold_call": "_oracle_bbpssw_after_rounds(0.73,4)"},
        {"setup": "", "call": "bbpssw_after_rounds(0.501,3)", "gold_call": "_oracle_bbpssw_after_rounds(0.501,3)"},
        {"setup": "", "call": "bbpssw_after_rounds(0.985,5)", "gold_call": "_oracle_bbpssw_after_rounds(0.985,5)"},
        {"setup": "", "call": "bbpssw_after_rounds(1.0,7)", "gold_call": "_oracle_bbpssw_after_rounds(1.0,7)"}
    ]
