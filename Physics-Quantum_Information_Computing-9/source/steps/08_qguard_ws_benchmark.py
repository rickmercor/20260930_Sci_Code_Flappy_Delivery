"""
Run the complete recovery comparison and return the selected-EXG gain of Q-GUARD-WS over base Q-GUARD.

This is the final orchestrator. Compose every preceding public function to reproduce the two paper-defined recovery modes, apply their feasibility and EXG selection contracts, and return Python round(weighted selected EXG - base selected EXG, 6). Require two eligible candidates per mode, retain earlier input order on an exact EXG tie, use every called output, and do not duplicate prior scientific formulas locally.

Returns
-------
return float(result)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

def qguard_ws_benchmark(fidelity_threshold=0.80, major_qualities=None, replaced_indices=None, swap_probability=0.91, r_max=4, candidates=None):
    """Return the six-decimal selected-EXG gain for one recovery instance; omitted arguments use the prompt benchmark.

    Returns:
        float: Q-GUARD-WS selected EXG minus base selected EXG, rounded once.
    """
    return float(result)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_qguard_ws_benchmark(
    fidelity_threshold=0.80,
    major_qualities=None,
    replaced_indices=None,
    swap_probability=0.91,
    r_max=4,
    candidates=None,
):
    import math
    """Return the rounded gain in selected EXG from Q-GUARD-WS over base Q-GUARD."""
    if major_qualities is None:
        major_qualities = [5.2, 9.0, 4.8, 11.5, 6.0, 10.0, 7.5]
    if replaced_indices is None:
        replaced_indices = [2, 3]
    if candidates is None:
        candidates = [
            {"width": 20, "availability": 0.623, "fidelities": [0.963, 0.927, 0.911, 0.900], "qualities": [14.56, 12.46, 6.22, 8.94]},
            {"width": 28, "availability": 0.738, "fidelities": [0.933, 0.979, 0.969, 0.962], "qualities": [9.47, 16.68, 14.42, 14.78]},
            {"width": 20, "availability": 0.784, "fidelities": [0.951, 0.926], "qualities": [14.68, 12.63]},
            {"width": 28, "availability": 0.859, "fidelities": [0.951, 0.912], "qualities": [10.51, 7.96]},
        ]
    threshold_w = _oracle_werner_parameter(fidelity_threshold)
    budgets = _oracle_segment_budgets(fidelity_threshold, major_qualities, replaced_indices)
    if abs(threshold_w - budgets[0]) > 1e-12:
        raise RuntimeError("inconsistent threshold conversion")
    base_rows = []
    weighted_rows = []
    for index, candidate in enumerate(candidates):
        width = int(candidate["width"])
        availability = float(candidate["availability"])
        realized = [float(value) for value in candidate["fidelities"]]
        qualities = [float(value) for value in candidate["qualities"]]
        if not realized or len(realized) != len(qualities):
            raise ValueError("candidate fidelity and quality arrays must be nonempty and aligned")
        predicted = [_oracle_hardware_initial_fidelity(value) for value in qualities]
        base_rounds = _oracle_equal_split_plan(realized, budgets[1], r_max, width)
        weighted_rounds = _oracle_weighted_split_plan(qualities, budgets[2], r_max, width)
        for output, initials, rounds, budget in (
            (base_rows, realized, base_rounds, budgets[1]),
            (weighted_rows, predicted, weighted_rounds, budgets[2]),
        ):
            feasible = all(value >= 0 for value in rounds)
            post = (
                [_oracle_bbpssw_after_rounds(initial, count) for initial, count in zip(initials, rounds)]
                if feasible
                else []
            )
            product = math.prod(_oracle_werner_parameter(value) for value in post) if feasible else 0.0
            margin = product - budget
            feasible = feasible and margin >= -1e-12
            score = _oracle_recovery_exg(width, swap_probability, rounds, availability) if feasible else 0.0
            output.append([float(index), float(feasible), score, margin, float(sum(rounds))])

    def best(rows):
        ranked = [(row[2], -int(row[0]), row) for row in rows if bool(row[1])]
        if len(ranked) < 2:
            raise ValueError("each allocation method requires at least two feasible candidates")
        ranked.sort(reverse=True)
        return ranked[0][2], ranked[1][2]

    base_best, _ = best(base_rows)
    weighted_best, _ = best(weighted_rows)
    return float(round(weighted_best[2] - base_best[2], 6))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "", "call": "qguard_ws_benchmark()", "gold_call": "_oracle_qguard_ws_benchmark()"},
        {"setup": "fidelity_threshold = 0.80\nmajor_qualities = [5.2, 9.0, 4.8, 11.5, 6.0, 10.0, 7.5]\nreplaced_indices = [2, 3]\nswap_probability = 0.91\nr_max = 4\ncandidates = [\n    {\"width\":20, \"availability\":0.623,\n     \"fidelities\":[0.963,0.927,0.911,0.900],\n     \"qualities\":[14.56,12.46,6.22,8.94]},\n    {\"width\":28, \"availability\":0.738,\n     \"fidelities\":[0.933,0.979,0.969,0.962],\n     \"qualities\":[9.47,16.68,14.42,14.78]},\n    {\"width\":20, \"availability\":0.784,\n     \"fidelities\":[0.951,0.926],\n     \"qualities\":[14.68,12.63]},\n    {\"width\":28, \"availability\":0.859,\n     \"fidelities\":[0.951,0.912],\n     \"qualities\":[10.51,7.96]},\n]\nswap_probability=1.0", "call": "qguard_ws_benchmark(fidelity_threshold,major_qualities,replaced_indices,swap_probability,r_max,candidates)", "gold_call": "_oracle_qguard_ws_benchmark(fidelity_threshold,major_qualities,replaced_indices,swap_probability,r_max,candidates)"},
        {"setup": "fidelity_threshold = 0.80\nmajor_qualities = [5.2, 9.0, 4.8, 11.5, 6.0, 10.0, 7.5]\nreplaced_indices = [2, 3]\nswap_probability = 0.91\nr_max = 4\ncandidates = [\n    {\"width\":20, \"availability\":0.623,\n     \"fidelities\":[0.963,0.927,0.911,0.900],\n     \"qualities\":[14.56,12.46,6.22,8.94]},\n    {\"width\":28, \"availability\":0.738,\n     \"fidelities\":[0.933,0.979,0.969,0.962],\n     \"qualities\":[9.47,16.68,14.42,14.78]},\n    {\"width\":20, \"availability\":0.784,\n     \"fidelities\":[0.951,0.926],\n     \"qualities\":[14.68,12.63]},\n    {\"width\":28, \"availability\":0.859,\n     \"fidelities\":[0.951,0.912],\n     \"qualities\":[10.51,7.96]},\n]\nswap_probability=0.83", "call": "qguard_ws_benchmark(fidelity_threshold,major_qualities,replaced_indices,swap_probability,r_max,candidates)", "gold_call": "_oracle_qguard_ws_benchmark(fidelity_threshold,major_qualities,replaced_indices,swap_probability,r_max,candidates)"},
        {"setup": "fidelity_threshold = 0.80\nmajor_qualities = [5.2, 9.0, 4.8, 11.5, 6.0, 10.0, 7.5]\nreplaced_indices = [2, 3]\nswap_probability = 0.91\nr_max = 4\ncandidates = [\n    {\"width\":20, \"availability\":0.623,\n     \"fidelities\":[0.963,0.927,0.911,0.900],\n     \"qualities\":[14.56,12.46,6.22,8.94]},\n    {\"width\":28, \"availability\":0.738,\n     \"fidelities\":[0.933,0.979,0.969,0.962],\n     \"qualities\":[9.47,16.68,14.42,14.78]},\n    {\"width\":20, \"availability\":0.784,\n     \"fidelities\":[0.951,0.926],\n     \"qualities\":[14.68,12.63]},\n    {\"width\":28, \"availability\":0.859,\n     \"fidelities\":[0.951,0.912],\n     \"qualities\":[10.51,7.96]},\n]\ncandidates[1]['availability']=0.55", "call": "qguard_ws_benchmark(fidelity_threshold,major_qualities,replaced_indices,swap_probability,r_max,candidates)", "gold_call": "_oracle_qguard_ws_benchmark(fidelity_threshold,major_qualities,replaced_indices,swap_probability,r_max,candidates)"},
        {"setup": "fidelity_threshold = 0.80\nmajor_qualities = [5.2, 9.0, 4.8, 11.5, 6.0, 10.0, 7.5]\nreplaced_indices = [2, 3]\nswap_probability = 0.91\nr_max = 4\ncandidates = [\n    {\"width\":20, \"availability\":0.623,\n     \"fidelities\":[0.963,0.927,0.911,0.900],\n     \"qualities\":[14.56,12.46,6.22,8.94]},\n    {\"width\":28, \"availability\":0.738,\n     \"fidelities\":[0.933,0.979,0.969,0.962],\n     \"qualities\":[9.47,16.68,14.42,14.78]},\n    {\"width\":20, \"availability\":0.784,\n     \"fidelities\":[0.951,0.926],\n     \"qualities\":[14.68,12.63]},\n    {\"width\":28, \"availability\":0.859,\n     \"fidelities\":[0.951,0.912],\n     \"qualities\":[10.51,7.96]},\n]\nfidelity_threshold=0.78", "call": "qguard_ws_benchmark(fidelity_threshold,major_qualities,replaced_indices,swap_probability,r_max,candidates)", "gold_call": "_oracle_qguard_ws_benchmark(fidelity_threshold,major_qualities,replaced_indices,swap_probability,r_max,candidates)"},
        {"setup": "fidelity_threshold = 0.80\nmajor_qualities = [5.2, 9.0, 4.8, 11.5, 6.0, 10.0, 7.5]\nreplaced_indices = [2, 3]\nswap_probability = 0.91\nr_max = 4\ncandidates = [\n    {\"width\":20, \"availability\":0.623,\n     \"fidelities\":[0.963,0.927,0.911,0.900],\n     \"qualities\":[14.56,12.46,6.22,8.94]},\n    {\"width\":28, \"availability\":0.738,\n     \"fidelities\":[0.933,0.979,0.969,0.962],\n     \"qualities\":[9.47,16.68,14.42,14.78]},\n    {\"width\":20, \"availability\":0.784,\n     \"fidelities\":[0.951,0.926],\n     \"qualities\":[14.68,12.63]},\n    {\"width\":28, \"availability\":0.859,\n     \"fidelities\":[0.951,0.912],\n     \"qualities\":[10.51,7.96]},\n]\nreplaced_indices=[0,1]", "call": "qguard_ws_benchmark(fidelity_threshold,major_qualities,replaced_indices,swap_probability,r_max,candidates)", "gold_call": "_oracle_qguard_ws_benchmark(fidelity_threshold,major_qualities,replaced_indices,swap_probability,r_max,candidates)"},
        {"setup": "fidelity_threshold = 0.80\nmajor_qualities = [5.2, 9.0, 4.8, 11.5, 6.0, 10.0, 7.5]\nreplaced_indices = [2, 3]\nswap_probability = 0.91\nr_max = 4\ncandidates = [\n    {\"width\":20, \"availability\":0.623,\n     \"fidelities\":[0.963,0.927,0.911,0.900],\n     \"qualities\":[14.56,12.46,6.22,8.94]},\n    {\"width\":28, \"availability\":0.738,\n     \"fidelities\":[0.933,0.979,0.969,0.962],\n     \"qualities\":[9.47,16.68,14.42,14.78]},\n    {\"width\":20, \"availability\":0.784,\n     \"fidelities\":[0.951,0.926],\n     \"qualities\":[14.68,12.63]},\n    {\"width\":28, \"availability\":0.859,\n     \"fidelities\":[0.951,0.912],\n     \"qualities\":[10.51,7.96]},\n]\nmajor_qualities=[8.0]*7", "call": "qguard_ws_benchmark(fidelity_threshold,major_qualities,replaced_indices,swap_probability,r_max,candidates)", "gold_call": "_oracle_qguard_ws_benchmark(fidelity_threshold,major_qualities,replaced_indices,swap_probability,r_max,candidates)"}
    ]
