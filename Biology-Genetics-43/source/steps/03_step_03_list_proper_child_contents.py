"""
List every assignment of pair contents to the pairs directly enclosed by one loop that keeps the loop proper, given the content of its closing pair.

A loop whose pairs can exchange partners locally without losing a pair cannot be part of a design, so the pair contents inside every loop must be chosen jointly with the closing pair.

Returns
-------
np.ndarray: integer array (K, n_children) of proper content tuples in lexicographic order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def list_proper_child_contents(parent_content: int, n_children: int) -> np.ndarray:
    """Return the proper content tuples for the pairs directly enclosed by a loop.

    Pair contents are coded ``0`` = G-C, ``1`` = C-G, ``2`` = A-U and
    ``3`` = U-A, the first base being the 5' base of the pair. A pair is
    black for G-C, white for C-G and gray for A-U or U-A; the complement of
    black is white, of white is black and of gray is gray. ``parent_content``
    is the content of the loop's closing pair, or ``-1`` for the exterior
    loop, which has no closing pair.

    The loop is proper when two conditions hold. First, the list made of the
    complement of the closing pair's color followed by the colors of the
    enclosed pairs (for the exterior loop, only the enclosed pairs' colors)
    holds at most one black, at most one white and at most two gray entries.
    Second, a gray enclosed pair has the same content as a gray closing pair,
    and two gray enclosed pairs have different contents.

    Parameters
    ----------
    parent_content : int
        Content code of the closing pair in ``{0, 1, 2, 3}``, or ``-1``.
    n_children : int
        Nonnegative number of pairs directly enclosed by the loop.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(K, n_children)`` holding every content
        tuple of the enclosed pairs (in 5'-to-3' order) that makes the loop
        proper, one per row, rows in increasing lexicographic order. When
        ``n_children`` is 0 it holds one empty row; when no tuple is proper
        it has no rows.

    Raises
    ------
    ValueError
        If ``parent_content`` is not an integer in ``{-1, 0, 1, 2, 3}`` or
        ``n_children`` is not a nonnegative integer (booleans are rejected).
    """
    return contents

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_list_proper_child_contents(parent_content: int, n_children: int) -> np.ndarray:
    """Reference implementation (enumeration under the color-count and gray-content rules)."""
    import itertools
    import numpy as np

    def _is_int(value):
        return not isinstance(value, bool) and isinstance(value, (int, np.integer))

    if not (_is_int(parent_content) and -1 <= parent_content <= 3):
        raise ValueError("parent_content must be an integer in {-1, 0, 1, 2, 3}")
    if not (_is_int(n_children) and n_children >= 0):
        raise ValueError("n_children must be a nonnegative integer")
    parent_content, n_children = int(parent_content), int(n_children)
    color = ("black", "white", "gray", "gray")
    complement = {"black": "white", "white": "black", "gray": "gray"}
    # At most one black, one white and two gray entries fit in the list.
    capacity = 4 - (1 if parent_content >= 0 else 0)
    if n_children > capacity:
        return np.zeros((0, n_children), dtype=np.int64)
    rows = []
    for combo in itertools.product(range(4), repeat=n_children):
        colors = [color[c] for c in combo]
        if parent_content >= 0:
            colors.append(complement[color[parent_content]])
        if colors.count("black") > 1 or colors.count("white") > 1 or colors.count("gray") > 2:
            continue
        grays = [c for c in combo if c >= 2]
        if len(set(grays)) != len(grays):
            continue  # two gray enclosed pairs must differ
        if parent_content >= 2 and any(c != parent_content for c in grays):
            continue  # a gray enclosed pair must match a gray closing pair
        rows.append(combo)
    return np.array(rows, dtype=np.int64).reshape(len(rows), n_children)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    reduce = (
        "import numpy as np\n"
        "def _rows(a, t):\n"
        "    a = np.asarray(a)\n"
        "    if a.ndim != 2 or a.shape[1] != t or not np.issubdtype(a.dtype, np.integer):\n"
        "        return -1.0\n"
        "    codes = a.astype(float) @ (4.0 ** np.arange(t - 1, -1, -1)) if t else np.zeros(len(a))\n"
        "    w = np.cos(np.arange(len(a), dtype=float) + 1.0)\n"
        "    return float(len(a) + codes @ w / 1000.0)\n"
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
            "call": "_rows(list_proper_child_contents(0, 2), 2)",
            "gold_call": "_rows(_oracle_list_proper_child_contents(0, 2), 2)",
        },
        {
            "setup": reduce,
            "call": "_rows(list_proper_child_contents(2, 1), 1)",
            "gold_call": "_rows(_oracle_list_proper_child_contents(2, 1), 1)",
        },
        {
            "setup": reduce,
            "call": "_rows(list_proper_child_contents(3, 2), 2)",
            "gold_call": "_rows(_oracle_list_proper_child_contents(3, 2), 2)",
        },
        {
            "setup": reduce,
            "call": "_rows(list_proper_child_contents(-1, 4), 4)",
            "gold_call": "_rows(_oracle_list_proper_child_contents(-1, 4), 4)",
        },
        {
            "setup": reduce,
            "call": "_rows(list_proper_child_contents(1, 3), 3)",
            "gold_call": "_rows(_oracle_list_proper_child_contents(1, 3), 3)",
        },
        {
            "setup": reduce,
            "call": "_rows(list_proper_child_contents(-1, 2), 2)",
            "gold_call": "_rows(_oracle_list_proper_child_contents(-1, 2), 2)",
        },
        {
            "setup": reduce,
            "call": "_rows(list_proper_child_contents(3, 0), 0)",
            "gold_call": "_rows(_oracle_list_proper_child_contents(3, 0), 0)",
        },
        {
            "setup": reduce,
            "call": "_rows(list_proper_child_contents(0, 4), 4)",
            "gold_call": "_rows(_oracle_list_proper_child_contents(0, 4), 4)",
        },
        {
            "setup": status,
            "call": "_status(lambda: list_proper_child_contents(4, 1))",
            "gold_call": "_status(lambda: _oracle_list_proper_child_contents(4, 1))",
        },
        {
            "setup": status,
            "call": "_status(lambda: list_proper_child_contents(True, 2))",
            "gold_call": "_status(lambda: _oracle_list_proper_child_contents(True, 2))",
        },
    ]
