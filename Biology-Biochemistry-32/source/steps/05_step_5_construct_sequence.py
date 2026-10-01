"""
Convert the ordered numeric colouring into a numerically encoded RNA sequence.

Use A = 0, C = 1, G = 2 and U = 3. Black maps to G-C, white to C-G, and grey to A-U or U-A. Grey siblings require opposite orientations, adjacent grey parent-child pairs require the same orientation, and unpaired positions become A. Within each set of grey pairs linked by these two rules, the pair with the smallest left coordinate is oriented A-U, with A at its left position, and the rest of the set follows.

Returns
-------
return sequence  # tuple[int, ...], using A=0, C=1, G=2, U=3
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def construct_sequence(structure: str, coloring: tuple[int, ...]) -> tuple[int, ...]:
    """Return the encoded RNA sequence induced by the ordered colour codes."""
    return None  # TODO: replace with the candidate implementation

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_construct_sequence(structure: str, coloring: tuple[int, ...]) -> tuple[int, ...]:
    pairs, _ = _oracle_parse_target(structure)
    if len(coloring) != len(pairs) or any(color not in {0, 1, 2} for color in coloring):
        raise ValueError("Supply exactly one valid colour code per ordered pair")
    color_by_pair = dict(zip(pairs, coloring))
    children = {pair: [] for pair in pairs}
    children[None] = []
    for pair in pairs:
        enclosing = [p for p in pairs if p[0] < pair[0] and pair[1] < p[1]]
        parent = min(enclosing, key=lambda p: p[1] - p[0]) if enclosing else None
        children[parent].append(pair)

    grey = [pair for pair in pairs if color_by_pair[pair] == 2]
    graph = {pair: [] for pair in grey}
    for parent, direct_children in children.items():
        siblings = [child for child in direct_children if child in graph]
        for index, first in enumerate(siblings):
            for second in siblings[index + 1:]:
                graph[first].append((second, 1))
                graph[second].append((first, 1))
        if parent in graph:
            for child in siblings:
                graph[parent].append((child, 0))
                graph[child].append((parent, 0))

    orientation = {}
    for start in grey:
        if start in orientation:
            continue
        orientation[start] = 0
        pending = [start]
        while pending:
            pair = pending.pop()
            for neighbor, flip in graph[pair]:
                expected = orientation[pair] ^ flip
                if neighbor not in orientation:
                    orientation[neighbor] = expected
                    pending.append(neighbor)
                elif orientation[neighbor] != expected:
                    raise ValueError("Conflicting grey-pair orientation constraints")

    sequence = [0] * len(structure)
    for pair, color in color_by_pair.items():
        left, right = pair
        if color == 0:
            bases = (2, 1)
        elif color == 1:
            bases = (1, 2)
        else:
            bases = (0, 3) if orientation[pair] == 0 else (3, 0)
        sequence[left], sequence[right] = bases
    return tuple(sequence)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    return [
        {"setup": 'structure = "(((...)))"; coloring = (0, 2, 0)', "call": "construct_sequence(structure, coloring)", "gold_call": "_oracle_construct_sequence(structure, coloring)"},
        {"setup": 'structure = "..."; coloring = ()', "call": "construct_sequence(structure, coloring)", "gold_call": "_oracle_construct_sequence(structure, coloring)"},
        {"setup": 'structure = "()"; coloring = (1,)', "call": "construct_sequence(structure, coloring)", "gold_call": "_oracle_construct_sequence(structure, coloring)"},
    ]
