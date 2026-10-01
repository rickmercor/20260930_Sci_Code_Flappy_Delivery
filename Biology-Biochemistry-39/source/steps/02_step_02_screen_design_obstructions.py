"""
Count the two loop motifs that make a target undesignable under base-pair maximization and report the shortest helix of the loop tree.

Loops that expose too many pairs, or several pairs beside an unpaired base, can always refold locally without losing pairs, while long helices leave room to offset the nesting of pairs.

Returns
-------
np.ndarray: integer [five-pair motif count, three-pair-with-unpaired motif count, shortest helix length].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def screen_design_obstructions(tree: "np.ndarray") -> "np.ndarray":
    """Return the forbidden-motif counts and the shortest helix length.

    ``tree`` is a loop tree as returned by ``build_loop_tree``: rows
    ``[i, j, parent, unpaired]`` for the ``k`` base pairs in 5' order, then
    the exterior-loop row ``[-1, n, -1, unpaired]``. The children of a row
    are the base-pair rows naming it as ``parent``. Return
    ``[five_pair, three_pair_unpaired, shortest_helix]``:

    * ``five_pair`` counts pair-closed loops with four or more children and
      exterior loops with five or more children;
    * ``three_pair_unpaired`` counts pair-closed loops with two or more
      children and exterior loops with three or more children when the
      loop's ``unpaired`` flag is 1;
    * ``shortest_helix`` is the smallest number of base pairs in a helix,
      a maximal chain of base pairs in which every pair except the last has
      exactly one child and an ``unpaired`` flag of 0; it is 0 when ``k`` is
      0.

    Parameters
    ----------
    tree : np.ndarray
        Integer loop tree of shape ``(k + 1, 4)``.

    Returns
    -------
    np.ndarray
        Integer array ``[five_pair, three_pair_unpaired, shortest_helix]``.

    Raises
    ------
    ValueError
        If ``tree`` is not a two-dimensional integer array with four columns
        and at least one row, if its last row does not have ``-1`` in columns
        0 and 2, if an ``unpaired`` flag is not 0 or 1, or if a base-pair row
        names a parent that is neither an earlier base-pair row nor the
        exterior row.
    """
    return screen

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_screen_design_obstructions(tree: "np.ndarray") -> "np.ndarray":
    """Reference implementation (child counts and helix lengths in 5' order)."""
    import numpy as np

    table = np.asarray(tree)
    if table.ndim != 2 or table.shape[1] != 4 or table.shape[0] < 1:
        raise ValueError("tree must be a two-dimensional array with four columns")
    if not np.issubdtype(table.dtype, np.integer):
        raise ValueError("tree must hold integers")
    exterior = table.shape[0] - 1
    if table[exterior, 0] != -1 or table[exterior, 2] != -1:
        raise ValueError("the last row must be the exterior loop")
    if not np.all((table[:, 3] == 0) | (table[:, 3] == 1)):
        raise ValueError("unpaired flags must be 0 or 1")
    parents = table[:exterior, 2]
    rows = np.arange(exterior)
    if not np.all(((parents >= 0) & (parents < rows)) | (parents == exterior)):
        raise ValueError("every base pair must name an earlier pair or the exterior loop")

    children = np.bincount(parents, minlength=exterior + 1) if exterior else np.zeros(1, dtype=np.int64)
    unpaired = table[:, 3]
    five_pair = int(np.sum(children[:exterior] >= 4)) + int(children[exterior] >= 5)
    three_pair_unpaired = (
        int(np.sum((children[:exterior] >= 2) & (unpaired[:exterior] == 1)))
        + int(children[exterior] >= 3 and unpaired[exterior] == 1)
    )

    # A pair continues its parent's helix when the parent is a base pair with
    # a single child and no unpaired position; parents precede children.
    stacked = (children == 1) & (unpaired == 0)
    length = np.zeros(exterior, dtype=np.int64)
    shortest = 0
    for row in range(exterior):
        parent = int(parents[row])
        length[row] = length[parent] + 1 if parent != exterior and stacked[parent] else 1
        if not stacked[row]:
            shortest = int(length[row]) if shortest == 0 else min(shortest, int(length[row]))
    return np.array([five_pair, three_pair_unpaired, shortest], dtype=np.int64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    build = (
        "import numpy as np\n"
        "def _tree(db):\n"
        "    stack, rows = [], {}\n"
        "    opens = [p for p, s in enumerate(db) if s == '(']\n"
        "    index = {p: r for r, p in enumerate(opens)}\n"
        "    k = len(opens)\n"
        "    t = np.zeros((k + 1, 4), dtype=np.int64)\n"
        "    t[k] = (-1, len(db), -1, 0)\n"
        "    owners = [k]\n"
        "    for p, s in enumerate(db):\n"
        "        if s == '.':\n"
        "            t[owners[-1], 3] = 1\n"
        "        elif s == '(':\n"
        "            stack.append(p)\n"
        "            t[index[p], 0] = p\n"
        "            t[index[p], 2] = owners[-1]\n"
        "            owners.append(index[p])\n"
        "        else:\n"
        "            t[index[stack.pop()], 1] = p\n"
        "            owners.pop()\n"
        "    return t\n"
        "def _code(a):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (3,):\n"
        "        return -1.0\n"
        "    return float(a[0] + 10.0 * a[1] + 100.0 * a[2]) / 100.0\n"
    )
    status = build + (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
    )
    return [
        {
            "setup": build,
            "call": "_code(screen_design_obstructions(_tree('((((((((....)))))...))(((..(......)...))))..(((((...(((...)))))(((.....))(((......)))(((...)))))))')))",
            "gold_call": "_code(_oracle_screen_design_obstructions(_tree('((((((((....)))))...))(((..(......)...))))..(((((...(((...)))))(((.....))(((......)))(((...)))))))')))",
        },
        {
            "setup": build,
            "call": "_code(screen_design_obstructions(_tree('((((((...)))((...))((...))((...)))))..(((.((...))((...)))))')))",
            "gold_call": "_code(_oracle_screen_design_obstructions(_tree('((((((...)))((...))((...))((...)))))..(((.((...))((...)))))')))",
        },
        {
            "setup": build,
            "call": "_code(screen_design_obstructions(_tree('(((((....)))))..((((...))))')))",
            "gold_call": "_code(_oracle_screen_design_obstructions(_tree('(((((....)))))..((((...))))')))",
        },
        {
            "setup": build,
            "call": "_code(screen_design_obstructions(_tree('......')))",
            "gold_call": "_code(_oracle_screen_design_obstructions(_tree('......')))",
        },
        {
            "setup": build,
            "call": "_code(screen_design_obstructions(_tree('((..((((...))))(((...)))))((...)).(((((...)))((...))((...))))')))",
            "gold_call": "_code(_oracle_screen_design_obstructions(_tree('((..((((...))))(((...)))))((...)).(((((...)))((...))((...))))')))",
        },
        {
            "setup": build,
            "call": "_code(screen_design_obstructions(_tree('((((..((...))..))))')))",
            "gold_call": "_code(_oracle_screen_design_obstructions(_tree('((((..((...))..))))')))",
        },
        {
            "setup": build,
            "call": "_code(screen_design_obstructions(_tree('(((...)))(((...)))(((...)))(((...)))')))",
            "gold_call": "_code(_oracle_screen_design_obstructions(_tree('(((...)))(((...)))(((...)))(((...)))')))",
        },
        {
            "setup": build,
            "call": "_code(screen_design_obstructions(_tree('()()()()()')))",
            "gold_call": "_code(_oracle_screen_design_obstructions(_tree('()()()()()')))",
        },
        {
            "setup": build,
            "call": "_code(screen_design_obstructions(_tree('().()()')))",
            "gold_call": "_code(_oracle_screen_design_obstructions(_tree('().()()')))",
        },
        {
            "setup": status,
            "call": "_status(lambda: screen_design_obstructions(np.array([[0, 5, 1, 1], [-1, 6, -1, 1]])[::-1].copy()))",
            "gold_call": "_status(lambda: _oracle_screen_design_obstructions(np.array([[0, 5, 1, 1], [-1, 6, -1, 1]])[::-1].copy()))",
        },
        {
            "setup": status,
            "call": "_status(lambda: screen_design_obstructions(np.array([[0, 5, 1, 2], [-1, 6, -1, 0]])))",
            "gold_call": "_status(lambda: _oracle_screen_design_obstructions(np.array([[0, 5, 1, 2], [-1, 6, -1, 0]])))",
        },
    ]
