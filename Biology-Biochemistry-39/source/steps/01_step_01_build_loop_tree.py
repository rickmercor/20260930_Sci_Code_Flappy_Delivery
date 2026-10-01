"""
Convert a dot-bracket secondary structure into the loop tree of its base pairs and exterior loop, flagging every loop that holds unpaired positions.

A pseudoknot-free secondary structure is a tree whose nodes are base pairs nested inside one another, and each loop, closed by a pair or by the exterior, may hold unpaired nucleotides.

Returns
-------
np.ndarray: integer (k + 1, 4) loop tree, base-pair rows [i, j, parent, unpaired] in 5' order, then the exterior row.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_loop_tree(dot_bracket: str) -> "np.ndarray":
    """Return the loop tree of a pseudoknot-free secondary structure.

    Positions are 0-based and ``n`` is the length of ``dot_bracket``. With
    ``k`` base pairs the result has ``k + 1`` rows. Rows ``0 .. k - 1``
    describe the base pairs in increasing order of their 5' position, each
    as ``[i, j, parent, unpaired]``: ``i < j`` are the paired positions,
    ``parent`` is the row of the closest pair enclosing ``(i, j)`` or ``k``
    when no pair encloses it, and ``unpaired`` is 1 when some unpaired
    position lies strictly between ``i`` and ``j`` but inside no pair nested
    in ``(i, j)``, else 0. Row ``k`` is the exterior loop
    ``[-1, n, -1, unpaired]``, whose flag is 1 when some unpaired position
    lies inside no pair, else 0.

    Parameters
    ----------
    dot_bracket : str
        Structure written with ``(``, ``)`` and ``.``.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(k + 1, 4)``.

    Raises
    ------
    ValueError
        If ``dot_bracket`` is not a non-empty string, contains a character
        other than ``(``, ``)`` and ``.``, or has unbalanced brackets.
    """
    return tree

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_build_loop_tree(dot_bracket: str) -> "np.ndarray":
    """Reference implementation (one bracket-matching pass, one loop-ownership pass)."""
    import numpy as np

    if not isinstance(dot_bracket, str) or not dot_bracket:
        raise ValueError("dot_bracket must be a non-empty string")
    if set(dot_bracket) - set("()."):
        raise ValueError("dot_bracket may contain only '(', ')' and '.'")
    partner = [-1] * len(dot_bracket)
    stack = []
    for position, symbol in enumerate(dot_bracket):
        if symbol == "(":
            stack.append(position)
        elif symbol == ")":
            if not stack:
                raise ValueError("unbalanced closing bracket")
            opening = stack.pop()
            partner[opening] = position
            partner[position] = opening
    if stack:
        raise ValueError("unbalanced opening bracket")

    openings = [position for position, symbol in enumerate(dot_bracket) if symbol == "("]
    row_of = {position: row for row, position in enumerate(openings)}
    exterior = len(openings)
    tree = np.zeros((exterior + 1, 4), dtype=np.int64)
    tree[exterior] = (-1, len(dot_bracket), -1, 0)
    # The innermost loop still open at each position owns that position.
    owners = [exterior]
    for position, symbol in enumerate(dot_bracket):
        owner = owners[-1]
        if symbol == ".":
            tree[owner, 3] = 1
        elif symbol == "(":
            row = row_of[position]
            tree[row] = (position, partner[position], owner, 0)
            owners.append(row)
        else:
            owners.pop()
    return tree

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    digest = (
        "import numpy as np\n"
        "def _digest(a):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.ndim != 2 or a.shape[1] != 4:\n"
        "        return -1.0\n"
        "    rows = np.cos(np.arange(1, a.shape[0] + 1))[:, None]\n"
        "    cols = np.array([1.0, 0.37, 2.11, 5.3])\n"
        "    return float(np.sum(a * cols * rows) / 10.0 + a.shape[0])\n"
    )
    status = (
        "import numpy as np\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
    )
    return [
        {
            "setup": digest,
            "call": "_digest(build_loop_tree('((((((((....)))))...))(((..(......)...))))..'))",
            "gold_call": "_digest(_oracle_build_loop_tree('((((((((....)))))...))(((..(......)...))))..'))",
        },
        {
            "setup": digest,
            "call": "_digest(build_loop_tree('((((...))((....))(((...)))))'))",
            "gold_call": "_digest(_oracle_build_loop_tree('((((...))((....))(((...)))))'))",
        },
        {
            "setup": digest,
            "call": "_digest(build_loop_tree('(())((.))'))",
            "gold_call": "_digest(_oracle_build_loop_tree('(())((.))'))",
        },
        {
            "setup": digest,
            "call": "_digest(build_loop_tree('.....'))",
            "gold_call": "_digest(_oracle_build_loop_tree('.....'))",
        },
        {
            "setup": digest,
            "call": "_digest(build_loop_tree('.((..(...)..((....)).))..(.)'))",
            "gold_call": "_digest(_oracle_build_loop_tree('.((..(...)..((....)).))..(.)'))",
        },
        {
            "setup": status,
            "call": "_status(lambda: build_loop_tree('((..)'))",
            "gold_call": "_status(lambda: _oracle_build_loop_tree('((..)'))",
        },
        {
            "setup": status,
            "call": "_status(lambda: build_loop_tree('((.x))'))",
            "gold_call": "_status(lambda: _oracle_build_loop_tree('((.x))'))",
        },
        {
            "setup": status,
            "call": "_status(lambda: build_loop_tree(')(..'))",
            "gold_call": "_status(lambda: _oracle_build_loop_tree(')(..'))",
        },
    ]
