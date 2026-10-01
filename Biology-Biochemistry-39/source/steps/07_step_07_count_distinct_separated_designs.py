"""
Count the distinct designs that are separated for at least one modular assignment, resolved by the number of strong pairs.

A design leaving some residues occupied by neither unpaired positions nor weak pairs is separated for several assignments at once, so distinct designs cannot be counted by adding assignments.

Returns
-------
np.ndarray: integer counts of distinct separated designs, indexed by the number of G-C and C-G pairs (length k + 1).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def count_distinct_separated_designs(tree: "np.ndarray", modulus: int) -> "np.ndarray":
    """Return the number of distinct separated designs by number of strong pairs.

    A proper design (as defined for ``count_level_constrained_designs``) is
    separated at ``modulus`` when no residue is shared by an unpaired
    position and an A-U or U-A pair, that is, when it is counted in at least
    one row of ``tabulate_assignment_counts(tree, modulus)``. Entry ``g`` of
    the result is the number of distinct separated designs with exactly
    ``g`` pairs coded G-C or C-G, each design counted once.

    Parameters
    ----------
    tree : np.ndarray
        Integer loop tree of shape ``(k + 1, 4)`` from ``build_loop_tree``.
    modulus : int
        Level modulus, from 1 to 8.

    Returns
    -------
    np.ndarray
        Integer array of length ``k + 1``.

    Raises
    ------
    ValueError
        If ``modulus`` is not an integer from 1 to 8 (booleans are rejected)
        or if ``count_level_constrained_designs`` rejects ``tree``.
    """
    return distinct

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_count_distinct_separated_designs(tree: "np.ndarray", modulus: int) -> "np.ndarray":
    """Reference implementation (Moebius inversion over the occupied unpaired residues)."""
    import numpy as np

    if isinstance(modulus, bool) or not isinstance(modulus, (int, np.integer)):
        raise ValueError("modulus must be an integer from 1 to 8")
    if not 1 <= int(modulus) <= 8:
        raise ValueError("modulus must be an integer from 1 to 8")
    size = int(modulus)
    everything = 2 ** size - 1
    total = None
    # Classify each separated design by the exact set S of residues its
    # unpaired positions occupy; its weak pairs must then avoid S. Designs
    # with unpaired residues inside S are counted by subset constraints, and
    # inclusion-exclusion over the subsets A of S keeps those occupying S
    # exactly.
    for exact in range(2 ** size):
        subset = exact
        while True:
            sign = -1 if bin(exact ^ subset).count("1") % 2 else 1
            counts = _oracle_count_level_constrained_designs(tree, size, subset, everything ^ exact)
            contribution = [sign * int(value) for value in counts]
            total = contribution if total is None else [a + b for a, b in zip(total, contribution)]
            if subset == 0:
                break
            subset = (subset - 1) & exact
    return np.array(total, dtype=np.int64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    build = (
        "import numpy as np\n"
        "def _tree(db):\n"
        "    stack = []\n"
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
        "def _poly(a, db):\n"
        "    width = db.count('(') + 1\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (width,):\n"
        "        return -1.0\n"
        "    return float(np.dot(a, np.cos(np.arange(1, width + 1))) / 100.0 + np.sum(a) / 1000.0)\n"
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
            "call": "_poly(count_distinct_separated_designs(_tree('((..))'), 3), '((..))')",
            "gold_call": "_poly(_oracle_count_distinct_separated_designs(_tree('((..))'), 3), '((..))')",
        },
        {
            "setup": build,
            "call": "_poly(count_distinct_separated_designs(_tree('((((((((....)))))...))(((..(......)...))))'), 4), '((((((((....)))))...))(((..(......)...))))')",
            "gold_call": "_poly(_oracle_count_distinct_separated_designs(_tree('((((((((....)))))...))(((..(......)...))))'), 4), '((((((((....)))))...))(((..(......)...))))')",
        },
        {
            "setup": build,
            "call": "_poly(count_distinct_separated_designs(_tree('((((...))((....))(((...)))))..'), 5), '((((...))((....))(((...)))))..')",
            "gold_call": "_poly(_oracle_count_distinct_separated_designs(_tree('((((...))((....))(((...)))))..'), 5), '((((...))((....))(((...)))))..')",
        },
        {
            "setup": build,
            "call": "_poly(count_distinct_separated_designs(_tree('(((((....)))(((...))))).((((....))))'), 4), '(((((....)))(((...))))).((((....))))')",
            "gold_call": "_poly(_oracle_count_distinct_separated_designs(_tree('(((((....)))(((...))))).((((....))))'), 4), '(((((....)))(((...))))).((((....))))')",
        },
        {
            "setup": build,
            "call": "_poly(count_distinct_separated_designs(_tree('.....'), 2), '.....')",
            "gold_call": "_poly(_oracle_count_distinct_separated_designs(_tree('.....'), 2), '.....')",
        },
        {
            "setup": build,
            "call": "_poly(count_distinct_separated_designs(_tree('((((((..((...))..))((....))))))'), 5), '((((((..((...))..))((....))))))')",
            "gold_call": "_poly(_oracle_count_distinct_separated_designs(_tree('((((((..((...))..))((....))))))'), 5), '((((((..((...))..))((....))))))')",
        },
        {
            "setup": build,
            "call": "_poly(count_distinct_separated_designs(_tree('(((...)))'), 3), '(((...)))')",
            "gold_call": "_poly(_oracle_count_distinct_separated_designs(_tree('(((...)))'), 3), '(((...)))')",
        },
        {
            "setup": status,
            "call": "_status(lambda: count_distinct_separated_designs(_tree('((...))'), True))",
            "gold_call": "_status(lambda: _oracle_count_distinct_separated_designs(_tree('((...))'), True))",
        },
    ]
