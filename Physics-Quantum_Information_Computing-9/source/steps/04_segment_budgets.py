"""
Compute the base and hardware-weighted Werner budgets assigned to a replaced major-path segment.

Compare the base recovery allocation with Q-GUARD-WS's depolarization-burden allocation in the source paper. Return [threshold Werner parameter, base segment budget, weighted segment budget, replaced-segment burden, total-path burden]. Indices are unique, zero-based major-path link indices.

Returns
-------
return [threshold_w, base_budget, weighted_budget, segment_burden, total_burden]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

def segment_budgets(fidelity_threshold, major_qualities, replaced_indices):
    """Return the two segment budgets and their burden diagnostics.

    Returns:
        list[float]: Five values in the documented order.
    """
    return [threshold_w, base_budget, weighted_budget, segment_burden, total_burden]

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_segment_budgets(fidelity_threshold: float, major_qualities, replaced_indices):
    import math
    qualities = [float(value) for value in major_qualities]
    indices = [int(value) for value in replaced_indices]
    if not qualities or any(not math.isfinite(value) or value <= 0.0 for value in qualities):
        raise ValueError("major_qualities must be a nonempty positive finite sequence")
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("replaced_indices must be nonempty and unique")
    if any(index < 0 or index >= len(qualities) for index in indices):
        raise ValueError("replaced index out of range")
    threshold_w = _oracle_werner_parameter(fidelity_threshold)
    base_budget = threshold_w ** (len(indices) / len(qualities))
    all_burden = sum(1.0 / value for value in qualities)
    segment_burden = sum(1.0 / qualities[index] for index in indices)
    weighted_budget = threshold_w ** (segment_burden / all_burden)
    return [threshold_w, base_budget, weighted_budget, segment_burden, all_burden]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "", "call": "segment_budgets(.8,[5.2,9,4.8,11.5,6,10,7.5],[2,3])", "gold_call": "_oracle_segment_budgets(.8,[5.2,9,4.8,11.5,6,10,7.5],[2,3])"},
        {"setup": "", "call": "segment_budgets(.75,[8,8,8,8],[0])", "gold_call": "_oracle_segment_budgets(.75,[8,8,8,8],[0])"},
        {"setup": "", "call": "segment_budgets(.9,[4,20,20,20],[0])", "gold_call": "_oracle_segment_budgets(.9,[4,20,20,20],[0])"},
        {"setup": "", "call": "segment_budgets(.7,[20,4,20,4],[1,3])", "gold_call": "_oracle_segment_budgets(.7,[20,4,20,4],[1,3])"},
        {"setup": "", "call": "segment_budgets(.85,[6,7,8,9,10],[0,4])", "gold_call": "_oracle_segment_budgets(.85,[6,7,8,9,10],[0,4])"},
        {"setup": "", "call": "segment_budgets(.6,[3,30,9],[0,1,2])", "gold_call": "_oracle_segment_budgets(.6,[3,30,9],[0,1,2])"},
        {"setup": "", "call": "segment_budgets(1.0,[5,10],[1])", "gold_call": "_oracle_segment_budgets(1.0,[5,10],[1])"}
    ]
