"""
Tabulate, for every modular assignment of residues to unpaired positions, the proper designs whose weak pairs avoid those residues, resolved by the number of strong pairs.

Once the residues reserved for unpaired positions are fixed and weak pairs are kept off them, no competing pair between an unpaired A and a paired U can reach the maximum pair count.

Returns
-------
np.ndarray: integer (2**modulus, k + 1) table of separated design counts per residue assignment and number of strong pairs.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def tabulate_assignment_counts(tree: "np.ndarray", modulus: int) -> "np.ndarray":
    """Return the separated design counts of every modular assignment.

    For each assignment ``s`` in ``0 .. 2**modulus - 1``, read as a set of
    residues through its bits, row ``s`` of the result is
    ``count_level_constrained_designs(tree, modulus, s, gray)`` with
    ``gray`` the complementary set ``2**modulus - 1 - s``: the proper
    designs whose unpaired positions all have residues in ``s`` and whose
    A-U and U-A pairs all have residues outside ``s``, indexed by the number
    of G-C and C-G pairs.

    Parameters
    ----------
    tree : np.ndarray
        Integer loop tree of shape ``(k + 1, 4)`` from ``build_loop_tree``.
    modulus : int
        Level modulus, from 1 to 8.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(2**modulus, k + 1)``.

    Raises
    ------
    ValueError
        If ``modulus`` is not an integer from 1 to 8 (booleans are rejected)
        or if ``count_level_constrained_designs`` rejects ``tree``.
    """
    return table

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_tabulate_assignment_counts(tree: "np.ndarray", modulus: int) -> "np.ndarray":
    """Reference implementation (one constrained count per residue assignment)."""
    import numpy as np

    if isinstance(modulus, bool) or not isinstance(modulus, (int, np.integer)):
        raise ValueError("modulus must be an integer from 1 to 8")
    if not 1 <= int(modulus) <= 8:
        raise ValueError("modulus must be an integer from 1 to 8")
    size = int(modulus)
    everything = 2 ** size - 1
    rows = [
        _oracle_count_level_constrained_designs(tree, size, assignment, everything ^ assignment)
        for assignment in range(2 ** size)
    ]
    return np.vstack(rows).astype(np.int64)

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
        "def _table(a, db, modulus):\n"
        "    shape = (2 ** modulus, db.count('(') + 1)\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != shape:\n"
        "        return -1.0\n"
        "    rows = np.cos(np.arange(1, shape[0] + 1))[:, None]\n"
        "    cols = np.sin(np.arange(1, shape[1] + 1))[None, :]\n"
        "    return float(np.sum(a * rows * cols) / 100.0 + np.sum(a) / 1000.0)\n"
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
            "call": "_table(tabulate_assignment_counts(_tree('((((...))((....))(((...)))))..'), 4), '((((...))((....))(((...)))))..', 4)",
            "gold_call": "_table(_oracle_tabulate_assignment_counts(_tree('((((...))((....))(((...)))))..'), 4), '((((...))((....))(((...)))))..', 4)",
        },
        {
            "setup": build,
            "call": "_table(tabulate_assignment_counts(_tree('((((((((....)))))...))(((..(......)...))))'), 3), '((((((((....)))))...))(((..(......)...))))', 3)",
            "gold_call": "_table(_oracle_tabulate_assignment_counts(_tree('((((((((....)))))...))(((..(......)...))))'), 3), '((((((((....)))))...))(((..(......)...))))', 3)",
        },
        {
            "setup": build,
            "call": "_table(tabulate_assignment_counts(_tree('(((((....)))(((...))))).((((....))))'), 3), '(((((....)))(((...))))).((((....))))', 3)",
            "gold_call": "_table(_oracle_tabulate_assignment_counts(_tree('(((((....)))(((...))))).((((....))))'), 3), '(((((....)))(((...))))).((((....))))', 3)",
        },
        {
            "setup": build,
            "call": "_table(tabulate_assignment_counts(_tree('....'), 2), '....', 2)",
            "gold_call": "_table(_oracle_tabulate_assignment_counts(_tree('....'), 2), '....', 2)",
        },
        {
            "setup": build,
            "call": "_table(tabulate_assignment_counts(_tree('((..))'), 3), '((..))', 3)",
            "gold_call": "_table(_oracle_tabulate_assignment_counts(_tree('((..))'), 3), '((..))', 3)",
        },
        {
            "setup": build,
            "call": "_table(tabulate_assignment_counts(_tree('(((((...(((...)))))(((.....))(((......)))(((...)))))))'), 4), '(((((...(((...)))))(((.....))(((......)))(((...)))))))', 4)",
            "gold_call": "_table(_oracle_tabulate_assignment_counts(_tree('(((((...(((...)))))(((.....))(((......)))(((...)))))))'), 4), '(((((...(((...)))))(((.....))(((......)))(((...)))))))', 4)",
        },
        {
            "setup": status,
            "call": "_status(lambda: tabulate_assignment_counts(_tree('((...))'), 9))",
            "gold_call": "_status(lambda: _oracle_tabulate_assignment_counts(_tree('((...))'), 9))",
        },
    ]
