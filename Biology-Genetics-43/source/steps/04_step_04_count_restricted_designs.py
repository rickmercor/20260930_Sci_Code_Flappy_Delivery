"""
Count the proper sequences of a loop tree whose unpaired positions and A-U-type pairs fall on prescribed residue classes of their levels, together with their total G+C content.

Every G-C or C-G pair shifts the G/C balance of the region it encloses, and keeping A-U-type pairs off the balance levels of unpaired A's removes the only base-pair-conserving refoldings a proper sequence could still undergo.

Returns
-------
np.ndarray: integer array [count, gc_total] for the restricted proper sequences.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def count_restricted_designs(
    tree: np.ndarray,
    modulus: int,
    leaf_residues: "Sequence[int]",
    gray_residues: "Sequence[int]",
) -> np.ndarray:
    """Return the number of restricted proper sequences and their total G+C count.

    ``tree`` is a loop table with rows ``[parent, i, j, u]`` as returned by
    ``build_loop_tree`` (row 0 the exterior loop, row ``k >= 1`` a pair whose
    innermost enclosing pair is row ``parent``, 0 for none, and which
    directly encloses ``u`` unpaired positions). A sequence places A at every
    unpaired position and one content on every pair (``0`` = G-C, ``1`` =
    C-G, ``2`` = A-U, ``3`` = U-A), such that every loop is proper in the
    sense of ``list_proper_child_contents``.

    A pair shifts the level of everything it encloses by ``+1`` if it is
    G-C, ``-1`` if it is C-G and ``0`` if it is A-U or U-A. The level of a
    pair is the sum of the shifts of the pairs strictly enclosing it; the
    level of an unpaired position is the sum of the shifts of all pairs
    enclosing it (0 in the exterior loop). A sequence is counted when every
    unpaired position has its level modulo ``modulus`` in ``leaf_residues``
    and every A-U or U-A pair has its level modulo ``modulus`` in
    ``gray_residues``.

    Parameters
    ----------
    tree : np.ndarray
        Integer loop table of shape ``(P + 1, 4)``.
    modulus : int
        Positive modulus applied to levels (Python ``%`` convention, so
        residues lie in ``[0, modulus)``).
    leaf_residues : sequence of int
        Residues allowed for unpaired positions; may be empty.
    gray_residues : sequence of int
        Residues allowed for A-U and U-A pairs; may be empty.

    Returns
    -------
    np.ndarray
        Integer array ``[count, gc_total]``: the number of counted sequences
        and the number of G and C nucleotides summed over all of them.

    Raises
    ------
    ValueError
        If ``tree`` is not a two-dimensional integer array with four columns
        whose row 0 has parent ``-1``, whose every row has a nonnegative
        unpaired count, and whose every row ``k >= 1`` has a parent in
        ``[0, k - 1]``, if ``modulus`` is not a positive integer, or if a
        residue is not an integer in ``[0, modulus)`` (booleans are rejected
        throughout).
    """
    return counts

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_count_restricted_designs(
    tree: np.ndarray,
    modulus: int,
    leaf_residues: "Sequence[int]",
    gray_residues: "Sequence[int]",
) -> np.ndarray:
    """Reference implementation (memoized tree recursion over pair, content and residue)."""
    import numpy as np

    def _is_int(value):
        return not isinstance(value, bool) and isinstance(value, (int, np.integer))

    table = np.asarray(tree)
    if (table.ndim != 2 or table.shape[1] != 4 or table.shape[0] < 1
            or not np.issubdtype(table.dtype, np.integer)):
        raise ValueError("tree must be a two-dimensional integer array with four columns")
    parent = [int(x) for x in table[:, 0]]
    unpaired = [int(x) for x in table[:, 3]]
    if parent[0] != -1 or unpaired[0] < 0:
        raise ValueError("row 0 must describe the exterior loop")
    n_pairs = table.shape[0] - 1
    children = [[] for _ in range(n_pairs + 1)]
    for k in range(1, n_pairs + 1):
        if not (0 <= parent[k] < k) or unpaired[k] < 0:
            raise ValueError("every pair needs an earlier parent row and a nonnegative count")
        children[parent[k]].append(k)
    if not (_is_int(modulus) and modulus >= 1):
        raise ValueError("modulus must be a positive integer")
    m = int(modulus)
    allowed = []
    for residues in (leaf_residues, gray_residues):
        values = list(residues)
        if not all(_is_int(r) and 0 <= r < m for r in values):
            raise ValueError("residues must be integers in [0, modulus)")
        allowed.append({int(r) for r in values})
    leaf_ok, gray_ok = allowed
    shift = (1, -1, 0, 0)
    gc_pair = (2, 2, 0, 0)
    tuples, memo = {}, {}

    def _proper(content, count):
        key = (content, count)
        if key not in tuples:
            rows = _oracle_list_proper_child_contents(content, count)
            tuples[key] = [tuple(int(x) for x in row) for row in rows]
        return tuples[key]

    def _loop(v, content, inner):
        # Sequences of the pairs directly enclosed by loop v (and below), given
        # the closing content and the residue ``inner`` of its interior.
        if unpaired[v] > 0 and inner not in leaf_ok:
            return 0, 0
        total, gc_total = 0, 0
        for combo in _proper(content, len(children[v])):
            count, gc_sum = 1, 0
            for kid, kid_content in zip(children[v], combo):
                c, g = _pair(kid, kid_content, inner)
                count, gc_sum = count * c, gc_sum * c + count * g
                if count == 0:
                    break
            total += count
            gc_total += gc_sum
        return total, gc_total

    def _pair(k, content, residue):
        key = (k, content, residue)
        if key not in memo:
            if content >= 2 and residue not in gray_ok:
                memo[key] = (0, 0)
            else:
                c, g = _loop(k, content, (residue + shift[content]) % m)
                memo[key] = (c, g + gc_pair[content] * c)
        return memo[key]

    count, gc_total = _loop(0, -1, 0)
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
            "call": "_pair(count_restricted_designs(_tree('((((...))((....))))'), 3, [1, 2], [0]))",
            "gold_call": "_pair(_oracle_count_restricted_designs(_tree('((((...))((....))))'), 3, [1, 2], [0]))",
        },
        {
            "setup": tree,
            "call": "_pair(count_restricted_designs(_tree('(((((...))((...))))).((((....))))'), 4, [1, 3], [0, 2]))",
            "gold_call": "_pair(_oracle_count_restricted_designs(_tree('(((((...))((...))))).((((....))))'), 4, [1, 3], [0, 2]))",
        },
        {
            "setup": tree,
            "call": "_pair(count_restricted_designs(_tree('..((...))..((....))..'), 2, [0, 1], []))",
            "gold_call": "_pair(_oracle_count_restricted_designs(_tree('..((...))..((....))..'), 2, [0, 1], []))",
        },
        {
            "setup": tree,
            "call": "_pair(count_restricted_designs(_tree('((((.....))((...))((....))))'), 5, [1, 2, 4], [0, 3]))",
            "gold_call": "_pair(_oracle_count_restricted_designs(_tree('((((.....))((...))((....))))'), 5, [1, 2, 4], [0, 3]))",
        },
        {
            "setup": tree,
            "call": "_pair(count_restricted_designs(_tree('(((...)))((...))((.((...)).))((...))'), 3, (1, 2), (0,)))",
            "gold_call": "_pair(_oracle_count_restricted_designs(_tree('(((...)))((...))((.((...)).))((...))'), 3, (1, 2), (0,)))",
        },
        {
            "setup": tree,
            "call": "_pair(count_restricted_designs(_tree('()'), 1, [0], [0]))",
            "gold_call": "_pair(_oracle_count_restricted_designs(_tree('()'), 1, [0], [0]))",
        },
        {
            "setup": tree,
            "call": "_pair(count_restricted_designs(_tree('.((((...))((...)))).'), 3, [2], [0, 1]))",
            "gold_call": "_pair(_oracle_count_restricted_designs(_tree('.((((...))((...)))).'), 3, [2], [0, 1]))",
        },
        {
            "setup": tree + status,
            "call": "_status(lambda: count_restricted_designs(_tree('((...))'), 3, [3], [0]))",
            "gold_call": "_status(lambda: _oracle_count_restricted_designs(_tree('((...))'), 3, [3], [0]))",
        },
        {
            "setup": tree + status,
            "call": "_status(lambda: count_restricted_designs(_tree('((...))'), 0, [], []))",
            "gold_call": "_status(lambda: _oracle_count_restricted_designs(_tree('((...))'), 0, [], []))",
        },
    ]
