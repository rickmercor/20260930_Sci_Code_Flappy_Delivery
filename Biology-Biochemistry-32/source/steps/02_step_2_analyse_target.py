"""
Measure maximal helices and count the forbidden local motifs m5 and m3-bullet.

A helix is a maximal run of consecutively nested pairs. For a loop, exposed pairs comprise its enclosing pair, except at the exterior root, and its direct paired children. The forbidden patterns are five exposed pairs (m5), or at least three exposed pairs together with an attached unpaired position (m3-bullet).

Returns
-------
return (helix_count, h_min, m5_count, m3_bullet_count)  # tuple[int, int, int, int]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def analyse_target(structure: str) -> tuple[int, int, int, int]:
    """Return four numeric structural-summary values."""
    return None  # TODO: replace with the candidate implementation

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_analyse_target(structure: str) -> tuple[int, int, int, int]:
    pairs, unpaired = _oracle_parse_target(structure)
    pair_set = set(pairs)
    lengths: list[int] = []
    for left, right in pairs:
        if (left - 1, right + 1) in pair_set:
            continue
        length = 0
        while (left, right) in pair_set:
            length += 1
            left, right = left + 1, right - 1
        lengths.append(length)

    nodes: tuple[tuple[int, int] | None, ...] = (None,) + pairs
    children = {node: [] for node in nodes}
    leaves = {node: [] for node in nodes}
    for pair in pairs:
        enclosing = [p for p in pairs if p[0] < pair[0] and pair[1] < p[1]]
        parent = min(enclosing, key=lambda p: p[1] - p[0]) if enclosing else None
        children[parent].append(pair)
    for position in unpaired:
        enclosing = [p for p in pairs if p[0] < position < p[1]]
        parent = min(enclosing, key=lambda p: p[1] - p[0]) if enclosing else None
        leaves[parent].append(position)

    m5_count = 0
    m3_bullet_count = 0
    for node in nodes:
        exposed = len(children[node]) + int(node is not None)
        m5_count += int(exposed >= 5)
        m3_bullet_count += int(exposed >= 3 and bool(leaves[node]))
    return (len(lengths), min(lengths) if lengths else 0, m5_count, m3_bullet_count)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    return [
        {"setup": 'structure = "(((...)))"', "call": "analyse_target(structure)", "gold_call": "_oracle_analyse_target(structure)"},
        {"setup": 'structure = "..."', "call": "analyse_target(structure)", "gold_call": "_oracle_analyse_target(structure)"},
        {"setup": 'structure = "(.()())"', "call": "analyse_target(structure)", "gold_call": "_oracle_analyse_target(structure)"},
    ]
