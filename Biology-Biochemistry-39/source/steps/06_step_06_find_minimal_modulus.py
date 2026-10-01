"""
Find the smallest level modulus at which the target admits at least one separated design.

Reducing levels modulo a small number keeps the design search fast, but short helices can make small moduli infeasible, so the modulus is raised until a separated design first exists.

Returns
-------
int: the smallest modulus from 1 to max_modulus whose assignment table has a nonzero entry.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def find_minimal_modulus(tree: "np.ndarray", max_modulus: int) -> int:
    """Return the smallest modulus with a nonzero separated design count.

    Scan ``modulus = 1, 2, ..., max_modulus`` and return the first modulus
    for which ``tabulate_assignment_counts(tree, modulus)`` has a nonzero
    entry.

    Parameters
    ----------
    tree : np.ndarray
        Integer loop tree of shape ``(k + 1, 4)`` from ``build_loop_tree``.
    max_modulus : int
        Largest modulus to try, from 1 to 8.

    Returns
    -------
    int
        The smallest admissible modulus.

    Raises
    ------
    ValueError
        If ``max_modulus`` is not an integer from 1 to 8 (booleans are
        rejected), if no modulus up to ``max_modulus`` admits a separated
        design, or if ``tabulate_assignment_counts`` rejects ``tree``.
    """
    return modulus

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_find_minimal_modulus(tree: "np.ndarray", max_modulus: int) -> int:
    """Reference implementation (increasing scan of assignment tables)."""
    import numpy as np

    if isinstance(max_modulus, bool) or not isinstance(max_modulus, (int, np.integer)):
        raise ValueError("max_modulus must be an integer from 1 to 8")
    if not 1 <= int(max_modulus) <= 8:
        raise ValueError("max_modulus must be an integer from 1 to 8")
    for modulus in range(1, int(max_modulus) + 1):
        table = _oracle_tabulate_assignment_counts(tree, modulus)
        if np.any(table > 0):
            return modulus
    raise ValueError("no modulus up to max_modulus admits a separated design")

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
        "def _status(fn):\n"
        "    try:\n"
        "        return float(fn())\n"
        "    except ValueError:\n"
        "        return -1.0\n"
    )
    return [
        {
            "setup": build,
            "call": "_status(lambda: find_minimal_modulus(_tree('((((((((....)))))...))(((..(......)...))))..'), 8))",
            "gold_call": "_status(lambda: _oracle_find_minimal_modulus(_tree('((((((((....)))))...))(((..(......)...))))..'), 8))",
        },
        {
            "setup": build,
            "call": "_status(lambda: find_minimal_modulus(_tree('((((...))((....))(((...)))))..'), 8))",
            "gold_call": "_status(lambda: _oracle_find_minimal_modulus(_tree('((((...))((....))(((...)))))..'), 8))",
        },
        {
            "setup": build,
            "call": "_status(lambda: find_minimal_modulus(_tree('(((((....)))(((...))))).((((....))))'), 8))",
            "gold_call": "_status(lambda: _oracle_find_minimal_modulus(_tree('(((((....)))(((...))))).((((....))))'), 8))",
        },
        {
            "setup": build,
            "call": "_status(lambda: find_minimal_modulus(_tree('((.((...)).))'), 8))",
            "gold_call": "_status(lambda: _oracle_find_minimal_modulus(_tree('((.((...)).))'), 8))",
        },
        {
            "setup": build,
            "call": "_status(lambda: find_minimal_modulus(_tree('(((((...(((...)))))(((.....))(((......)))(((...)))))))'), 8))",
            "gold_call": "_status(lambda: _oracle_find_minimal_modulus(_tree('(((((...(((...)))))(((.....))(((......)))(((...)))))))'), 8))",
        },
        {
            "setup": build,
            "call": "_status(lambda: find_minimal_modulus(_tree('((((((((....)))))...))(((..(......)...))))..'), 3))",
            "gold_call": "_status(lambda: _oracle_find_minimal_modulus(_tree('((((((((....)))))...))(((..(......)...))))..'), 3))",
        },
        {
            "setup": build,
            "call": "_status(lambda: find_minimal_modulus(_tree('(((((...(((...)))))(((.....))(((......)))(((...)))))))'), 1))",
            "gold_call": "_status(lambda: _oracle_find_minimal_modulus(_tree('(((((...(((...)))))(((.....))(((......)))(((...)))))))'), 1))",
        },
        {
            "setup": build,
            "call": "_status(lambda: find_minimal_modulus(_tree('(((..((...))((...)))))'), 8))",
            "gold_call": "_status(lambda: _oracle_find_minimal_modulus(_tree('(((..((...))((...)))))'), 8))",
        },
        {
            "setup": build,
            "call": "_status(lambda: find_minimal_modulus(_tree('((...))'), 0))",
            "gold_call": "_status(lambda: _oracle_find_minimal_modulus(_tree('((...))'), 0))",
        },
    ]
