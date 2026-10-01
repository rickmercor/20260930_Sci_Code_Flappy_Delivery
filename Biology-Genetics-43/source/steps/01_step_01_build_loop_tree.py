"""
Convert a dot-bracket secondary structure into its loop table, one row for the exterior loop and one row per base pair.

A pseudoknot-free secondary structure is a tree whose internal nodes are base pairs and whose leaves are unpaired nucleotides, so each loop is one closing pair together with the pairs and unpaired positions it directly encloses.

Returns
-------
np.ndarray: integer loop table of shape (P + 1, 4), rows [parent, i, j, u].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_loop_tree(structure: str, min_hairpin: int = 0) -> np.ndarray:
    """Return the loop table of a dot-bracket secondary structure.

    Positions are numbered 1 to ``n``. Row 0 describes the exterior loop and
    rows ``1..P`` describe the ``P`` base pairs in increasing order of their
    5' position. Row ``k`` holds ``[parent, i, j, u]``: the 5' and 3'
    positions ``i < j`` of the pair, the row ``parent`` of the innermost pair
    enclosing it (0 when no pair encloses it), and the number ``u`` of
    unpaired positions whose innermost enclosing pair is this one. Row 0 is
    ``[-1, 0, n + 1, u0]`` with ``u0`` the number of unpaired positions
    enclosed by no pair.

    Parameters
    ----------
    structure : str
        Non-empty string over ``'('``, ``')'`` and ``'.'``.
    min_hairpin : int
        Nonnegative minimum number of positions strictly between the two
        positions of every pair.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(P + 1, 4)``.

    Raises
    ------
    ValueError
        If ``structure`` is not a non-empty string over the three symbols,
        if its brackets are unbalanced, if ``min_hairpin`` is not a
        nonnegative integer (booleans are rejected), or if some pair encloses
        fewer than ``min_hairpin`` positions.
    """
    return table

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_build_loop_tree(structure: str, min_hairpin: int = 0) -> np.ndarray:
    """Reference implementation (one left-to-right stack pass)."""
    import numpy as np

    def _is_int(value):
        return not isinstance(value, bool) and isinstance(value, (int, np.integer))

    if not isinstance(structure, str) or len(structure) == 0:
        raise ValueError("structure must be a non-empty string")
    if set(structure) - set("()."):
        raise ValueError("structure may only contain '(', ')' and '.'")
    if not (_is_int(min_hairpin) and min_hairpin >= 0):
        raise ValueError("min_hairpin must be a nonnegative integer")
    n = len(structure)
    rows = [[-1, 0, n + 1, 0]]
    open_rows = [0]  # rows whose loops are open at the current position
    for position, symbol in enumerate(structure, start=1):
        current = open_rows[-1]
        if symbol == "(":
            rows.append([current, position, 0, 0])
            open_rows.append(len(rows) - 1)
        elif symbol == ")":
            if current == 0:
                raise ValueError("unbalanced brackets: ')' without a matching '('")
            if position - rows[current][1] - 1 < min_hairpin:
                raise ValueError("a pair encloses fewer than min_hairpin positions")
            rows[current][2] = position
            open_rows.pop()
        else:
            rows[current][3] += 1
    if len(open_rows) != 1:
        raise ValueError("unbalanced brackets: '(' without a matching ')'")
    return np.array(rows, dtype=np.int64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    reduce = (
        "import numpy as np\n"
        "def _sig(t):\n"
        "    t = np.asarray(t)\n"
        "    if t.ndim != 2 or t.shape[1] != 4 or not np.issubdtype(t.dtype, np.integer):\n"
        "        return -1.0\n"
        "    w = np.cos(np.arange(t.size, dtype=float) + 1.0)\n"
        "    flat = t.astype(float).ravel()\n"
        "    return float(t.shape[0] + (np.sum(np.abs(flat)) + flat @ w) / 1000.0)\n"
    )
    status = (
        "import numpy as np\n"
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
            "setup": reduce,
            "call": "_sig(build_loop_tree('((..((...))..))((.....))..'))",
            "gold_call": "_sig(_oracle_build_loop_tree('((..((...))..))((.....))..'))",
        },
        {
            "setup": reduce,
            "call": "_sig(build_loop_tree('.((((...))((....))((..))))...(.)', 1))",
            "gold_call": "_sig(_oracle_build_loop_tree('.((((...))((....))((..))))...(.)', 1))",
        },
        {
            "setup": reduce,
            "call": "_sig(build_loop_tree('()(())'))",
            "gold_call": "_sig(_oracle_build_loop_tree('()(())'))",
        },
        {
            "setup": reduce,
            "call": "_sig(build_loop_tree('......'))",
            "gold_call": "_sig(_oracle_build_loop_tree('......'))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "int(build_loop_tree('((.((....)).))((...))')[4, 0])",
            "gold_call": "int(_oracle_build_loop_tree('((.((....)).))((...))')[4, 0])",
        },
        {
            "setup": status,
            "call": "_status(lambda: build_loop_tree('((...)).)'))",
            "gold_call": "_status(lambda: _oracle_build_loop_tree('((...)).)'))",
        },
        {
            "setup": status,
            "call": "_status(lambda: build_loop_tree('((.))', 2))",
            "gold_call": "_status(lambda: _oracle_build_loop_tree('((.))', 2))",
        },
        {
            "setup": status,
            "call": "_status(lambda: build_loop_tree('(.[.])'))",
            "gold_call": "_status(lambda: _oracle_build_loop_tree('(.[.])'))",
        },
    ]
