"""
Verify compatibility and unique maximum-base-pair folding, then calculate GC percentage.

Allowed pairs are AU, UA, GC, CG, GU and UG. Interval dynamic programming finds the largest number of noncrossing allowed pairs and counts optimal structures. The target is unique only when it is compatible, reaches the maximum score and is the sole optimum. GC percentage is (N_G + N_C) / n × 100.

Returns
-------
return (compatible, target_pairs, maximum_pairs, optimal_structure_count, unique_target, N_G, N_C, gc_percentage)  # tuple[int, int, int, int, int, int, int, float]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_design(
    sequence: tuple[int, ...],
    structure: str,
    minimum_span: int = 0,
    gc_decimals: int = 4,
) -> tuple[int, int, int, int, int, int, int, float]:
    """Return documented numeric verification values and rounded GC percentage."""
    return None  # TODO: replace with the candidate implementation

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_evaluate_design(
    sequence: tuple[int, ...],
    structure: str,
    minimum_span: int = 0,
    gc_decimals: int = 4,
) -> tuple[int, int, int, int, int, int, int, float]:
    pairs, _ = _oracle_parse_target(structure)
    if not sequence or len(sequence) != len(structure) or any(base not in {0, 1, 2, 3} for base in sequence):
        raise ValueError("Sequence must match target length and use codes 0, 1, 2, 3")
    if not isinstance(minimum_span, int) or minimum_span < 0:
        raise ValueError("minimum_span must be a nonnegative integer")

    allowed = {(2, 1), (1, 2), (0, 3), (3, 0), (2, 3), (3, 2)}
    n = len(sequence)
    score = [[0] * n for _ in range(n)]
    count = [[1] * n for _ in range(n)]

    def _entry(matrix, left, right, empty):
        return matrix[left][right] if left <= right else empty

    for width in range(2, n + 1):
        for left in range(n - width + 1):
            right = left + width - 1
            best = _entry(score, left, right - 1, 0)
            ways = _entry(count, left, right - 1, 1)
            for partner in range(left, right - minimum_span):
                if (sequence[partner], sequence[right]) not in allowed:
                    continue
                candidate = _entry(score, left, partner - 1, 0) + _entry(score, partner + 1, right - 1, 0) + 1
                new_ways = _entry(count, left, partner - 1, 1) * _entry(count, partner + 1, right - 1, 1)
                if candidate > best:
                    best, ways = candidate, new_ways
                elif candidate == best:
                    ways += new_ways
            score[left][right], count[left][right] = best, ways

    compatible = int(all(
        (sequence[left], sequence[right]) in allowed and right - left - 1 >= minimum_span
        for left, right in pairs
    ))
    maximum_pairs = score[0][n - 1]
    optimal_count = count[0][n - 1]
    unique_target = int(compatible == 1 and len(pairs) == maximum_pairs and optimal_count == 1)
    n_g = sequence.count(2)
    n_c = sequence.count(1)
    gc_percentage = round(100.0 * (n_g + n_c) / n, gc_decimals)
    return (compatible, len(pairs), maximum_pairs, optimal_count, unique_target, n_g, n_c, gc_percentage)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    return [
        {"setup": 'sequence = (2,0,2,0,0,0,1,3,1); structure = "(((...)))"; minimum_span = 0; gc_decimals = 4', "call": "evaluate_design(sequence, structure, minimum_span, gc_decimals)", "gold_call": "_oracle_evaluate_design(sequence, structure, minimum_span, gc_decimals)"},
        {"setup": 'sequence = (0,); structure = "."; minimum_span = 0; gc_decimals = 4', "call": "evaluate_design(sequence, structure, minimum_span, gc_decimals)", "gold_call": "_oracle_evaluate_design(sequence, structure, minimum_span, gc_decimals)"},
        {"setup": 'sequence = (2,1); structure = "()"; minimum_span = 1; gc_decimals = 2', "call": "evaluate_design(sequence, structure, minimum_span, gc_decimals)", "gold_call": "_oracle_evaluate_design(sequence, structure, minimum_span, gc_decimals)"},
    ]
