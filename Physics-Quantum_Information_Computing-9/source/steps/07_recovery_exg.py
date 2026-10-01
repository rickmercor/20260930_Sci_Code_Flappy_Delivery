"""
Evaluate the paper's local expected-goodput score for one feasible recovery span.

Reconstruct Phase 4's EXG metric, not the exploratory Q-GUARD-FP path score. The span has one fewer intermediate swap than hops. Use the paper's analytic raw-pair cost per planned round and the supplied bottleneck availability factor. Preserve full precision.

Returns
-------
return float(exg)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

def recovery_exg(width, swap_probability, rounds, availability_factor):
    """Return the expected goodput of one feasible span.

    Returns:
        float: The full-precision EXG score.
    """
    return float(exg)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_recovery_exg(width: int, swap_probability: float, rounds, availability_factor: float):
    import math
    width = int(width)
    swap_probability = float(swap_probability)
    rounds = [int(value) for value in rounds]
    availability_factor = float(availability_factor)
    if width < 1 or not rounds or any(value < 0 for value in rounds):
        raise ValueError("width and rounds must describe a feasible nonempty span")
    if not 0.0 <= swap_probability <= 1.0:
        raise ValueError("swap_probability must be in [0, 1]")
    if not 0.0 <= availability_factor <= 1.0:
        raise ValueError("availability_factor must be in [0, 1]")
    if any(2**value > width for value in rounds):
        raise ValueError("purification cost exceeds width")
    denominator = 1.0 + sum(2**value - 1 for value in rounds)
    swaps = len(rounds) - 1
    return float(width * swap_probability**swaps / denominator * availability_factor)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "", "call": "recovery_exg(10,.9,[0],.8)", "gold_call": "_oracle_recovery_exg(10,.9,[0],.8)"},
        {"setup": "", "call": "recovery_exg(20,.91,[1,2],.784)", "gold_call": "_oracle_recovery_exg(20,.91,[1,2],.784)"},
        {"setup": "", "call": "recovery_exg(28,.91,[4,1,2,3],.738)", "gold_call": "_oracle_recovery_exg(28,.91,[4,1,2,3],.738)"},
        {"setup": "", "call": "recovery_exg(16,0.0,[1,1],.9)", "gold_call": "_oracle_recovery_exg(16,0.0,[1,1],.9)"},
        {"setup": "", "call": "recovery_exg(16,.8,[1,1],0.0)", "gold_call": "_oracle_recovery_exg(16,.8,[1,1],0.0)"},
        {"setup": "", "call": "recovery_exg(8,1.0,[3,0,1],1.0)", "gold_call": "_oracle_recovery_exg(8,1.0,[3,0,1],1.0)"},
        {"setup": "", "call": "recovery_exg(32,.77,[2,4,1,3,2],.66)", "gold_call": "_oracle_recovery_exg(32,.77,[2,4,1,3,2],.66)"},
        {"setup": "", "call": "recovery_exg(20,.91,[1,2],.784)-recovery_exg(28,.91,[2,3],.859)", "gold_call": "_oracle_recovery_exg(20,.91,[1,2],.784)-_oracle_recovery_exg(28,.91,[2,3],.859)"}
    ]
