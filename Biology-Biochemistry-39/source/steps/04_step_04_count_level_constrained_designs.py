"""
Count the locally proper designs of a loop tree whose unpaired positions and weak pairs fall on prescribed level residues, resolved by the number of strong pairs.

Each G-C or C-G pair shifts a nesting level up or down, and an unpaired A can only steal the U of an A-U or U-A pair at the same level, so constraining level residues confines the competing structures.

Returns
-------
np.ndarray: integer counts of level-constrained proper designs, indexed by the number of G-C and C-G pairs (length k + 1).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def count_level_constrained_designs(
    tree: "np.ndarray",
    modulus: int,
    leaf_mask: int,
    gray_mask: int,
) -> "np.ndarray":
    """Return level-constrained proper design counts by number of strong pairs.

    ``tree`` is a loop tree from ``build_loop_tree`` with ``k`` base pairs.
    A design writes A at every unpaired position and gives every base pair
    an identity code (0 G-C, 1 C-G, 2 A-U, 3 U-A, 5' nucleotide first). It
    is proper when, in every loop including the exterior loop, the
    identities of the child pairs in 5' order form one of the rows that
    ``enumerate_proper_loop_assignments`` returns for the identity of the
    loop's closing pair (``-1`` for the exterior loop).

    The level of a base pair is the number of G-C pairs minus the number of
    C-G pairs among the pairs strictly enclosing it. The level of an
    unpaired position is that difference over all pairs enclosing the
    position. The residue of a level is its value reduced into
    ``[0, modulus)``, so a level of -1 has residue ``modulus - 1``.

    Count the proper designs in which every unpaired position has a residue
    ``r`` whose bit ``2**r`` is set in ``leaf_mask`` and every A-U or U-A
    pair has a residue whose bit is set in ``gray_mask``. Entry ``g`` of the
    result is the number of those designs with exactly ``g`` pairs coded
    G-C or C-G.

    Parameters
    ----------
    tree : np.ndarray
        Integer loop tree of shape ``(k + 1, 4)`` with ``k`` at most 30.
    modulus : int
        Level modulus, from 1 to 8.
    leaf_mask : int
        Allowed residues of unpaired positions, in ``[0, 2**modulus)``.
    gray_mask : int
        Allowed residues of A-U and U-A pairs, in ``[0, 2**modulus)``.

    Returns
    -------
    np.ndarray
        Integer array of length ``k + 1``.

    Raises
    ------
    ValueError
        If ``modulus`` is not an integer from 1 to 8, if a mask is not an
        integer in ``[0, 2**modulus)`` (booleans are rejected), if ``tree``
        has more than 30 base pairs, if the tree contains a five-pair loop
        or a three-pair loop with an unpaired position, or if
        ``screen_design_obstructions`` rejects the tree representation.
    """
    return counts

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_count_level_constrained_designs(
    tree: "np.ndarray",
    modulus: int,
    leaf_mask: int,
    gray_mask: int,
) -> "np.ndarray":
    """Reference implementation (memoised loop recursion over pair, identity and residue)."""
    from functools import lru_cache

    import numpy as np

    def _is_integer(value):
        return not isinstance(value, bool) and isinstance(value, (int, np.integer))

    if not (_is_integer(modulus) and 1 <= int(modulus) <= 8):
        raise ValueError("modulus must be an integer from 1 to 8")
    size = int(modulus)
    for name, mask in (("leaf_mask", leaf_mask), ("gray_mask", gray_mask)):
        if not (_is_integer(mask) and 0 <= int(mask) < 2 ** size):
            raise ValueError(f"{name} must be an integer in [0, 2**modulus)")
    obstructions = _oracle_screen_design_obstructions(tree)
    if int(obstructions[0]) or int(obstructions[1]):
        raise ValueError("the tree contains a loop motif that no sequence can design")
    table = np.asarray(tree, dtype=np.int64)
    exterior = table.shape[0] - 1
    if exterior > 30:
        raise ValueError("at most 30 base pairs are supported")
    children = [[] for _ in range(exterior + 1)]
    for row in range(exterior):
        children[int(table[row, 2])].append(row)
    unpaired = [bool(flag) for flag in table[:, 3]]
    leaves, weak = int(leaf_mask), int(gray_mask)
    shift = (1, -1, 0, 0)
    proper = {}

    def _times(first, second):
        product = [0] * (len(first) + len(second) - 1)
        for i, a in enumerate(first):
            if a:
                for j, b in enumerate(second):
                    product[i + j] += a * b
        return product

    def _loop(row, closing, residue):
        key = (closing, len(children[row]))
        if key not in proper:
            proper[key] = _oracle_enumerate_proper_loop_assignments(*key)
        total = [0]
        for combo in proper[key]:
            product = [1]
            for child, code in zip(children[row], combo):
                product = _times(product, _pair(child, int(code), residue))
                if not any(product):
                    break
            total += [0] * max(0, len(product) - len(total))
            for g, value in enumerate(product):
                total[g] += value
        return total

    @lru_cache(maxsize=None)
    def _pair(row, code, residue):
        if code >= 2 and not (weak >> residue) & 1:
            return (0,)
        inner = (residue + shift[code]) % size
        if unpaired[row] and not (leaves >> inner) & 1:
            return (0,)
        body = _loop(row, code, inner)
        return tuple([0] + body) if code <= 1 else tuple(body)

    if unpaired[exterior] and not leaves & 1:
        counts = [0]
    else:
        counts = _loop(exterior, -1, 0)
    result = np.zeros(exterior + 1, dtype=np.int64)
    for g, value in enumerate(counts[: exterior + 1]):
        result[g] = value
    return result

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
    cases = [
        ("((((((((....)))))...))(((..(......)...))))", 4, 3, 12),
        ("((((...))((....))(((...)))))..", 4, 9, 6),
        ("(((.((....))..)))((((...))))", 3, 7, 7),
        ("..((...))", 2, 2, 3),
        ("((((....))))((((....))))", 1, 1, 0),
        ("(((((....)))(((...))))).((((....))))", 3, 1, 6),
        ("(((((((....)))..)))((((...)))))..((((....))))", 4, 5, 10),
        ("(((((....)))(((...))))).((((....))))", 3, 5, 7),
    ]
    specs = []
    for db, modulus, leaves, weak in cases:
        specs.append({
            "setup": build,
            "call": f"_poly(count_level_constrained_designs(_tree('{db}'), {modulus}, {leaves}, {weak}), '{db}')",
            "gold_call": f"_poly(_oracle_count_level_constrained_designs(_tree('{db}'), {modulus}, {leaves}, {weak}), '{db}')",
        })
    specs.append({
        "setup": status,
        "call": "_status(lambda: count_level_constrained_designs(_tree('((...))'), 4, 16, 3))",
        "gold_call": "_status(lambda: _oracle_count_level_constrained_designs(_tree('((...))'), 4, 16, 3))",
    })
    specs.append({
        "setup": status,
        "call": "_status(lambda: count_level_constrained_designs(_tree('((...))'), 0, 0, 0))",
        "gold_call": "_status(lambda: _oracle_count_level_constrained_designs(_tree('((...))'), 0, 0, 0))",
    })
    specs.append({
        "setup": status,
        "call": "_status(lambda: count_level_constrained_designs(_tree('()()()()()'), 4, 1, 14))",
        "gold_call": "_status(lambda: _oracle_count_level_constrained_designs(_tree('()()()()()'), 4, 1, 14))",
    })
    return specs
