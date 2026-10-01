"""
Allocate Q-GUARD-WS purification rounds greedily across heterogeneous detour links.

Reconstruct the complete Q-GUARD-WS greedy effort distribution from Section V-A of the source paper, including its feasibility and resource rules. Resolve an exact marginal-loss tie by lower hop index. Return one count per hop or an all--1 vector when the span is infeasible.

Returns
-------
return rounds
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

def weighted_split_plan(qualities, segment_budget, r_max, width):
    """Return Q-GUARD-WS's ordered nonuniform round plan.

    Returns:
        list[int]: One count per quality, or all -1 when infeasible.
    """
    return rounds

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_weighted_split_plan(qualities, segment_budget: float, r_max: int, width: int):
    import math
    qualities = [float(value) for value in qualities]
    segment_budget = float(segment_budget)
    r_max = int(r_max)
    width = int(width)
    if not qualities or any(not math.isfinite(value) or value <= 0.0 for value in qualities):
        raise ValueError("qualities must be a nonempty positive finite sequence")
    if not math.isfinite(segment_budget) or not 0.0 < segment_budget <= 1.0:
        raise ValueError("segment_budget must be in (0, 1]")
    if r_max < 0 or width < 1:
        raise ValueError("r_max must be nonnegative and width positive")
    initials = [_oracle_hardware_initial_fidelity(value) for value in qualities]
    if any(value <= 0.5 for value in initials):
        return [-1 for _ in initials]
    uniform = -1
    for count in range(r_max + 1):
        if 2**count > width:
            break
        product = math.prod(
            _oracle_werner_parameter(_oracle_bbpssw_after_rounds(initial, count)) for initial in initials
        )
        if product + 1e-12 >= segment_budget:
            uniform = count
            break
    if uniform < 0:
        return [-1 for _ in initials]
    rounds = [uniform for _ in initials]
    while True:
        options = []
        for index, count in enumerate(rounds):
            if count == 0:
                continue
            trial = list(rounds)
            trial[index] -= 1
            product = math.prod(
                _oracle_werner_parameter(_oracle_bbpssw_after_rounds(initial, value))
                for initial, value in zip(initials, trial)
            )
            if product + 1e-12 >= segment_budget:
                before = _oracle_werner_parameter(_oracle_bbpssw_after_rounds(initials[index], count))
                after = _oracle_werner_parameter(_oracle_bbpssw_after_rounds(initials[index], count - 1))
                options.append((before - after, index, trial))
        if not options:
            break
        _, _, rounds = min(options, key=lambda row: (row[0], row[1]))
    return rounds

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "", "call": "weighted_split_plan([15,15],.75,4,16)", "gold_call": "_oracle_weighted_split_plan([15,15],.75,4,16)"},
        {"setup": "", "call": "weighted_split_plan([14.68,12.63],.9123751497822746,4,20)", "gold_call": "_oracle_weighted_split_plan([14.68,12.63],.9123751497822746,4,20)"},
        {"setup": "", "call": "weighted_split_plan([6,18,10],.84,5,32)", "gold_call": "_oracle_weighted_split_plan([6,18,10],.84,5,32)"},
        {"setup": "", "call": "weighted_split_plan([4,5],.98,2,8)", "gold_call": "_oracle_weighted_split_plan([4,5],.98,2,8)"},
        {"setup": "", "call": "weighted_split_plan([12,9,15],.90,5,4)", "gold_call": "_oracle_weighted_split_plan([12,9,15],.90,5,4)"},
        {"setup": "", "call": "weighted_split_plan([20],.7,0,1)", "gold_call": "_oracle_weighted_split_plan([20],.7,0,1)"},
        {"setup": "", "call": "weighted_split_plan([7,11,16,9],.78,5,32)", "gold_call": "_oracle_weighted_split_plan([7,11,16,9],.78,5,32)"},
        {"setup": "", "call": "weighted_split_plan([.5,10],.70,4,16)", "gold_call": "_oracle_weighted_split_plan([.5,10],.70,4,16)"}
    ]
