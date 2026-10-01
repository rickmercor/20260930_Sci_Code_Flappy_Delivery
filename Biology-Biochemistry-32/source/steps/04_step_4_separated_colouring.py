"""
Find the deterministic locally proper modulo-m separated colouring of the target.

Black changes the level by +1, white by -1 and grey by 0, modulo m. Grey-pair residues and unpaired-leaf residues must be disjoint. To make the answer reproducible, unpaired positions are fixed at residue 0, pairs are processed by increasing left coordinate, and colours are tried in the order black, white, grey.

Returns
-------
return coloring  # tuple[int, ...] aligned with Step 1 pairs, or None
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def find_separated_coloring(structure: str, m: int = 2) -> tuple[int, ...] | None:
    """Return deterministic colour codes aligned with the ordered base pairs."""
    return None  # TODO: replace with the candidate implementation

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from functools import lru_cache
from itertools import product


def _oracle_find_separated_coloring(structure: str, m: int = 2) -> tuple[int, ...] | None:
    if not isinstance(m, int) or m < 2:
        raise ValueError("m must be an integer of at least 2")
    pairs, unpaired = _oracle_parse_target(structure)
    _, _, m5_count, m3_count = _oracle_analyse_target(structure)
    if m5_count or m3_count:
        return None

    nodes = (None,) + pairs
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

    colors = (0, 1, 2)
    delta = {0: 1, 1: -1, 2: 0}
    leaf_residue = 0
    choices = {}

    @lru_cache(None)
    def _solve(node, color, level):
        if color == 2 and level == leaf_residue:
            return False
        below = (level + delta[color]) % m
        if leaves[node] and below != leaf_residue:
            return False
        for assigned in product(colors, repeat=len(children[node])):
            if not _oracle_check_local_coloring(color, assigned):
                continue
            if all(_solve(child, child_color, below)
                   for child, child_color in zip(children[node], assigned)):
                choices[(node, color, level)] = assigned
                return True
        return False

    roots = tuple(children[None])
    for root_colors in product(colors, repeat=len(roots)):
        if not _oracle_check_local_coloring(-1, root_colors):
            continue
        if not all(_solve(node, color, 0) for node, color in zip(roots, root_colors)):
            continue
        coloring = {}
        work = [(node, color, 0) for node, color in reversed(tuple(zip(roots, root_colors)))]
        while work:
            node, color, level = work.pop()
            coloring[node] = color
            below = (level + delta[color]) % m
            assigned = choices[(node, color, level)]
            work.extend((child, child_color, below)
                        for child, child_color in reversed(tuple(zip(children[node], assigned))))
        return tuple(coloring[pair] for pair in pairs)
    return None

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    pin = (
        "import math\n"
        "def _pin(result):\n"
        "    if result is None:\n"
        "        return -1.0\n"
        "    return (sum(math.sin(0.31 * (i + 1)) * v for i, v in enumerate(result))\n"
        "            + 10000.0 * len(result))\n"
    )
    return [
        {"setup": pin + 'structure = "((((((...)))(((...)))(((...))))))"; m = 2',
         "call": "_pin(find_separated_coloring(structure, m))",
         "gold_call": "_pin(_oracle_find_separated_coloring(structure, m))"},
        {"setup": pin + 'structure = "..."; m = 2',
         "call": "_pin(find_separated_coloring(structure, m))",
         "gold_call": "_pin(_oracle_find_separated_coloring(structure, m))"},
        {"setup": pin + 'structure = "()()"; m = 3',
         "call": "_pin(find_separated_coloring(structure, m))",
         "gold_call": "_pin(_oracle_find_separated_coloring(structure, m))"},
        {"setup": pin + 'structure = "(.)."; m = 2',
         "call": "_pin(find_separated_coloring(structure, m))",
         "gold_call": "_pin(_oracle_find_separated_coloring(structure, m))"},
        {"setup": pin + 'structure = "((...))((...))((...))((...))((...))"; m = 2',
         "call": "_pin(find_separated_coloring(structure, m))",
         "gold_call": "_pin(_oracle_find_separated_coloring(structure, m))"},
    ]
