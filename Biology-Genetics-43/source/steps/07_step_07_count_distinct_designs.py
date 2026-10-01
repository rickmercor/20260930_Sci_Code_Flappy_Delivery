"""
Count the distinct proper sequences whose unpaired positions and A-U-type pairs occupy disjoint level residues modulo m, and total their G+C content.

The separated sequences of a structure form the design family a uniform sampler draws from, so their number and composition set both the sampler's efficiency and the expected nucleotide content of what it returns.

Returns
-------
np.ndarray: integer array [number of distinct separated sequences, their total G+C count].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def count_distinct_designs(tree: np.ndarray, modulus: int) -> np.ndarray:
    """Return the number of distinct separated sequences and their total G+C count.

    Sequences, levels and residues are as in ``count_restricted_designs``:
    A at every unpaired position, one content per pair, every loop proper,
    and levels taken modulo ``modulus``. A sequence is separated modulo
    ``modulus`` when no residue is shared by an unpaired position and an
    A-U or U-A pair, i.e. the set of level residues of its unpaired
    positions and the set of level residues of its A-U-type pairs are
    disjoint. Count every separated sequence exactly once and total the
    number of G and C nucleotides over them.

    Parameters
    ----------
    tree : np.ndarray
        Integer loop table of shape ``(P + 1, 4)`` as returned by
        ``build_loop_tree``.
    modulus : int
        Positive modulus applied to levels.

    Returns
    -------
    np.ndarray
        Integer array ``[count, gc_total]`` over the distinct separated
        sequences.

    Raises
    ------
    ValueError
        If ``modulus`` is not a positive integer (booleans are rejected) or
        ``tree`` is not a valid loop table in the sense of
        ``count_restricted_designs``.
    """
    return counts

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_count_distinct_designs(tree: np.ndarray, modulus: int) -> np.ndarray:
    """Reference implementation (inclusion-exclusion over disjoint allowed residue sets)."""
    import itertools
    import numpy as np

    if isinstance(modulus, bool) or not isinstance(modulus, (int, np.integer)) or modulus < 1:
        raise ValueError("modulus must be a positive integer")
    m = int(modulus)
    # Each residue is allowed for unpaired positions (0), for A-U-type pairs
    # (1) or for neither (2). A sequence whose occupied residue sets are L and
    # G (disjoint) is counted by every labelling with L inside the first class
    # and G inside the second; the residues it leaves free each contribute
    # +1 + 1 - 1 = 1, so the signed sum counts it exactly once, while
    # sequences with L and G overlapping are counted by no labelling at all.
    count, gc_total = 0, 0
    for labels in itertools.product(range(3), repeat=m):
        leaf = [r for r in range(m) if labels[r] == 0]
        gray = [r for r in range(m) if labels[r] == 1]
        sign = -1 if (m - len(leaf) - len(gray)) % 2 else 1
        c, g = _oracle_count_restricted_designs(tree, m, leaf, gray)
        count += sign * int(c)
        gc_total += sign * int(g)
    return np.array([count, gc_total], dtype=np.int64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    tree = (
        "import numpy as np\n"
        "def _tree(s):\n"
        "    rows, stack = [[-1, 0, len(s) + 1, 0]], [0]\n"
        "    for p, ch in enumerate(s, start=1):\n"
        "        if ch == '(':\n"
        "            rows.append([stack[-1], p, 0, 0]); stack.append(len(rows) - 1)\n"
        "        elif ch == ')':\n"
        "            rows[stack.pop()][2] = p\n"
        "        else:\n"
        "            rows[stack[-1]][3] += 1\n"
        "    return np.array(rows, dtype=np.int64)\n"
        "def _pair(a):\n"
        "    a = np.asarray(a)\n"
        "    if a.shape != (2,) or not np.issubdtype(a.dtype, np.integer):\n"
        "        return -1.0\n"
        "    return float(a[0]) / 1000.0 + float(a[1]) / 1.0e7\n"
    )
    status = (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        {
            "setup": tree,
            "call": "_pair(count_distinct_designs(_tree('((...))'), 3))",
            "gold_call": "_pair(_oracle_count_distinct_designs(_tree('((...))'), 3))",
        },
        {
            "setup": tree,
            "call": "_pair(count_distinct_designs(_tree('((((...))((....))))'), 4))",
            "gold_call": "_pair(_oracle_count_distinct_designs(_tree('((((...))((....))))'), 4))",
        },
        {
            "setup": tree,
            "call": "_pair(count_distinct_designs(_tree('((.....))(((..((((...))((....)))).)))..'), 4))",
            "gold_call": "_pair(_oracle_count_distinct_designs(_tree('((.....))(((..((((...))((....)))).)))..'), 4))",
        },
        {
            "setup": tree,
            "call": "_pair(count_distinct_designs(_tree('..(((((....)))(((...)))))..(....)..'), 5))",
            "gold_call": "_pair(_oracle_count_distinct_designs(_tree('..(((((....)))(((...)))))..(....)..'), 5))",
        },
        {
            "setup": tree,
            "call": "_pair(count_distinct_designs(_tree('((((...))((...))((...))))'), 2))",
            "gold_call": "_pair(_oracle_count_distinct_designs(_tree('((((...))((...))((...))))'), 2))",
        },
        {
            "setup": tree,
            "call": "_pair(count_distinct_designs(_tree('.....'), 1))",
            "gold_call": "_pair(_oracle_count_distinct_designs(_tree('.....'), 1))",
        },
        {
            "setup": tree + status,
            "call": "_status(lambda: count_distinct_designs(_tree('((...))'), -2))",
            "gold_call": "_status(lambda: _oracle_count_distinct_designs(_tree('((...))'), -2))",
        },
    ]
