"""
List every assignment of pair identities to the child pairs of one loop that leaves the loop locally proper for a given closing pair.

Inside a single loop, two pairs of the same kind, or a closing pair and a child of the complementary kind, can swap partners and refold without losing a pair, so a design must forbid those local combinations.

Returns
-------
np.ndarray: integer (r, n_children) proper child identity rows in increasing lexicographic order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def enumerate_proper_loop_assignments(closing_code: int, n_children: int) -> "np.ndarray":
    """Return the proper pair identities for the children of one loop.

    Pair identities are coded by their 5' nucleotide first: 0 is G-C,
    1 is C-G, 2 is A-U and 3 is U-A. Codes 0 and 1 are the two strong
    colours and codes 2 and 3 share the weak colour; the complement of a
    colour exchanges the two strong colours and keeps the weak one.
    ``closing_code`` is the identity of the pair closing the loop, or ``-1``
    for the exterior loop, which has no closing pair.

    An assignment of identities to the ``n_children`` child pairs, listed in
    5' order, is proper when both conditions hold:

    * the colour list made of the complement of the closing pair's colour
      (omitted for the exterior loop) and the colours of the children holds
      at most one G-C, at most one C-G and at most two weak entries;
    * two weak children have different identities, and a weak child of a
      weak closing pair has the same identity as the closing pair.

    Return all proper assignments as rows in increasing lexicographic order,
    the first child being the most significant.

    Parameters
    ----------
    closing_code : int
        Identity 0, 1, 2 or 3 of the closing pair, or -1 for the exterior
        loop.
    n_children : int
        Number of child pairs, between 0 and 8 inclusive.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(r, n_children)``; ``r`` is 1 when
        ``n_children`` is 0 and may be 0 when no assignment is proper.

    Raises
    ------
    ValueError
        If ``closing_code`` is not an integer in ``{-1, 0, 1, 2, 3}`` or
        ``n_children`` is not an integer from 0 to 8 (booleans are rejected
        for both).
    """
    return assignments

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_enumerate_proper_loop_assignments(closing_code: int, n_children: int) -> "np.ndarray":
    """Reference implementation (filtered lexicographic product)."""
    import itertools

    import numpy as np

    def _is_integer(value):
        return not isinstance(value, bool) and isinstance(value, (int, np.integer))

    if not (_is_integer(closing_code) and int(closing_code) in (-1, 0, 1, 2, 3)):
        raise ValueError("closing_code must be -1, 0, 1, 2 or 3")
    if not (_is_integer(n_children) and 0 <= int(n_children) <= 8):
        raise ValueError("n_children must be an integer from 0 to 8")
    closing = int(closing_code)
    count = int(n_children)
    colour = (0, 1, 2, 2)
    complement = (1, 0, 2, 2)

    rows = []
    for combo in itertools.product(range(4), repeat=count):
        colours = [colour[code] for code in combo]
        if closing >= 0:
            colours.append(complement[closing])
        if colours.count(0) > 1 or colours.count(1) > 1 or colours.count(2) > 2:
            continue
        weak = [code for code in combo if code >= 2]
        if len(weak) == 2 and weak[0] == weak[1]:
            continue
        # An opposite weak child would let the stacked A and U swap partners.
        if closing >= 2 and any(code != closing for code in weak):
            continue
        rows.append(combo)
    return np.array(rows, dtype=np.int64).reshape(len(rows), count)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    rows = (
        "import numpy as np\n"
        "def _rows(a, width):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.ndim != 2 or a.shape[1] != width:\n"
        "        return -1.0\n"
        "    if a.shape[0] == 0:\n"
        "        return 0.5\n"
        "    place = np.cos(np.arange(1, a.shape[0] + 1))[:, None]\n"
        "    column = np.arange(1, width + 1, dtype=float) if width else np.zeros(0)\n"
        "    return float(a.shape[0] + np.sum((a + 1.0) * column * place) / 10.0)\n"
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
            "setup": rows,
            "call": "_rows(enumerate_proper_loop_assignments(2, 1), 1)",
            "gold_call": "_rows(_oracle_enumerate_proper_loop_assignments(2, 1), 1)",
        },
        {
            "setup": rows,
            "call": "_rows(enumerate_proper_loop_assignments(0, 2), 2)",
            "gold_call": "_rows(_oracle_enumerate_proper_loop_assignments(0, 2), 2)",
        },
        {
            "setup": rows,
            "call": "_rows(enumerate_proper_loop_assignments(3, 3), 3)",
            "gold_call": "_rows(_oracle_enumerate_proper_loop_assignments(3, 3), 3)",
        },
        {
            "setup": rows,
            "call": "_rows(enumerate_proper_loop_assignments(-1, 4), 4)",
            "gold_call": "_rows(_oracle_enumerate_proper_loop_assignments(-1, 4), 4)",
        },
        {
            "setup": rows,
            "call": "_rows(enumerate_proper_loop_assignments(1, 0), 0)",
            "gold_call": "_rows(_oracle_enumerate_proper_loop_assignments(1, 0), 0)",
        },
        {
            "setup": rows,
            "call": "_rows(enumerate_proper_loop_assignments(1, 4), 4)",
            "gold_call": "_rows(_oracle_enumerate_proper_loop_assignments(1, 4), 4)",
        },
        {
            "setup": status,
            "call": "_status(lambda: enumerate_proper_loop_assignments(4, 2))",
            "gold_call": "_status(lambda: _oracle_enumerate_proper_loop_assignments(4, 2))",
        },
        {
            "setup": status,
            "call": "_status(lambda: enumerate_proper_loop_assignments(0, True))",
            "gold_call": "_status(lambda: _oracle_enumerate_proper_loop_assignments(0, True))",
        },
    ]
