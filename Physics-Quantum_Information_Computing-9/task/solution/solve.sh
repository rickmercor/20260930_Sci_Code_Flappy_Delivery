#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def werner_parameter(fidelity: float) -> float:
    import math
    fidelity = float(fidelity)
    if not math.isfinite(fidelity) or fidelity < 0.25 or fidelity > 1.0:
        raise ValueError("fidelity must be finite and in [0.25, 1]")
    return (4.0 * fidelity - 1.0) / 3.0

def hardware_initial_fidelity(quality: float) -> float:
    import math
    quality = float(quality)
    if not math.isfinite(quality) or quality <= 0.0:
        raise ValueError("quality must be finite and positive")
    return 0.25 + 0.75 * math.exp(-1.0 / quality)

def bbpssw_after_rounds(initial_fidelity: float, rounds: int) -> float:
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

def segment_budgets(fidelity_threshold: float, major_qualities, replaced_indices):
    import math
    qualities = [float(value) for value in major_qualities]
    indices = [int(value) for value in replaced_indices]
    if not qualities or any(not math.isfinite(value) or value <= 0.0 for value in qualities):
        raise ValueError("major_qualities must be a nonempty positive finite sequence")
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("replaced_indices must be nonempty and unique")
    if any(index < 0 or index >= len(qualities) for index in indices):
        raise ValueError("replaced index out of range")
    threshold_w = werner_parameter(fidelity_threshold)
    base_budget = threshold_w ** (len(indices) / len(qualities))
    all_burden = sum(1.0 / value for value in qualities)
    segment_burden = sum(1.0 / qualities[index] for index in indices)
    weighted_budget = threshold_w ** (segment_burden / all_burden)
    return [threshold_w, base_budget, weighted_budget, segment_burden, all_burden]

def equal_split_plan(initial_fidelities, segment_budget: float, r_max: int, width: int):
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
            if werner_parameter(bbpssw_after_rounds(initial, count)) + 1e-12 >= hop_target:
                selected = count
                break
        rounds.append(selected)
    return rounds

def weighted_split_plan(qualities, segment_budget: float, r_max: int, width: int):
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
    initials = [hardware_initial_fidelity(value) for value in qualities]
    if any(value <= 0.5 for value in initials):
        return [-1 for _ in initials]
    uniform = -1
    for count in range(r_max + 1):
        if 2**count > width:
            break
        product = math.prod(
            werner_parameter(bbpssw_after_rounds(initial, count)) for initial in initials
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
                werner_parameter(bbpssw_after_rounds(initial, value))
                for initial, value in zip(initials, trial)
            )
            if product + 1e-12 >= segment_budget:
                before = werner_parameter(bbpssw_after_rounds(initials[index], count))
                after = werner_parameter(bbpssw_after_rounds(initials[index], count - 1))
                options.append((before - after, index, trial))
        if not options:
            break
        _, _, rounds = min(options, key=lambda row: (row[0], row[1]))
    return rounds

def recovery_exg(width: int, swap_probability: float, rounds, availability_factor: float):
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

def qguard_ws_benchmark(
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
    threshold_w = werner_parameter(fidelity_threshold)
    budgets = segment_budgets(fidelity_threshold, major_qualities, replaced_indices)
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
        predicted = [hardware_initial_fidelity(value) for value in qualities]
        base_rounds = equal_split_plan(realized, budgets[1], r_max, width)
        weighted_rounds = weighted_split_plan(qualities, budgets[2], r_max, width)
        for output, initials, rounds, budget in (
            (base_rows, realized, base_rounds, budgets[1]),
            (weighted_rows, predicted, weighted_rounds, budgets[2]),
        ):
            feasible = all(value >= 0 for value in rounds)
            post = (
                [bbpssw_after_rounds(initial, count) for initial, count in zip(initials, rounds)]
                if feasible
                else []
            )
            product = math.prod(werner_parameter(value) for value in post) if feasible else 0.0
            margin = product - budget
            feasible = feasible and margin >= -1e-12
            score = recovery_exg(width, swap_probability, rounds, availability) if feasible else 0.0
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
SCICODE_GOLD_EOF
