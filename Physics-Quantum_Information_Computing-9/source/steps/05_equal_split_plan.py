"""
Plan the minimum per-hop purification rounds for base Q-GUARD's equal target allocation.

Reconstruct base Q-GUARD's recovery-plan convention from the source paper. Return one minimum round count per hop; use -1 exactly for a hop that cannot reach its assigned target within both the round and width limits.

Returns
-------
return rounds
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

def equal_split_plan(initial_fidelities, segment_budget, r_max, width):
    """Return base Q-GUARD's ordered per-hop round plan.

    Returns:
        list[int]: One round count per input hop, with -1 marking an unreachable hop.
    """
    return rounds

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_equal_split_plan(initial_fidelities, segment_budget: float, r_max: int, width: int):
    import math
    fidelities = [float(value) for value in initial_fidelities]
    segment_budget = float(segment_budget)
    r_max = int(r_max)
    width = int(width)
    if not fidelities or any(not math.isfinite(value) or value <= 0.5 or value > 1.0 for value in fidelities):
        raise ValueError("initial_fidelities must be a nonempty sequence in (0.5, 1]")
    if not math.isfinite(segment_budget) or not 0.0 < segment_budget <= 1.0:
        raise ValueError("segment_budget must be in (0, 1]")
    if r_max < 0 or width < 1:
        raise ValueError("r_max must be nonnegative and width positive")
    hop_target = segment_budget ** (1.0 / len(fidelities))
    rounds = []
    for initial in fidelities:
        selected = -1
        for count in range(r_max + 1):
            if 2**count > width:
                break
            if _oracle_werner_parameter(_oracle_bbpssw_after_rounds(initial, count)) + 1e-12 >= hop_target:
                selected = count
                break
        rounds.append(selected)
    return rounds

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "", "call": "equal_split_plan([.99],.8,4,8)", "gold_call": "_oracle_equal_split_plan([.99],.8,4,8)"},
        {"setup": "", "call": "equal_split_plan([.96,.94],.88,4,16)", "gold_call": "_oracle_equal_split_plan([.96,.94],.88,4,16)"},
        {"setup": "", "call": "equal_split_plan([.84,.92,.95],.82,5,32)", "gold_call": "_oracle_equal_split_plan([.84,.92,.95],.82,5,32)"},
        {"setup": "", "call": "equal_split_plan([.73,.76],.96,2,8)", "gold_call": "_oracle_equal_split_plan([.73,.76],.96,2,8)"},
        {"setup": "", "call": "equal_split_plan([.91,.93],.90,5,2)", "gold_call": "_oracle_equal_split_plan([.91,.93],.90,5,2)"},
        {"setup": "", "call": "equal_split_plan([.951,.926],.915197221965416,4,20)", "gold_call": "_oracle_equal_split_plan([.951,.926],.915197221965416,4,20)"},
        {"setup": "", "call": "equal_split_plan([.94,.97,.91,.95],.86,4,16)", "gold_call": "_oracle_equal_split_plan([.94,.97,.91,.95],.86,4,16)"}
    ]
