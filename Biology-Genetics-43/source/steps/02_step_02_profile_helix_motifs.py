"""
Measure the helix lengths of a loop tree and count its loops that carry either of the two locally undesignable motifs of base-pair maximization.

Under base-pair maximization a loop exposing too many pairs, or several pairs next to an unpaired base, can always refold without losing a pair, while long helices give a designer room to adjust the coloring.

Returns
-------
np.ndarray: integer array [shortest helix, helix count, five-pair motif loops, three-pair motif loops].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def profile_helix_motifs(tree: np.ndarray) -> np.ndarray:
    """Return the helix and forbidden-motif profile of a loop tree.

    ``tree`` is a loop table with rows ``[parent, i, j, u]`` as returned by
    ``build_loop_tree``: row 0 is the exterior loop, row ``k >= 1`` a base
    pair whose innermost enclosing pair is row ``parent`` (0 for none) and
    which directly encloses ``u`` unpaired positions. The pairs directly
    enclosed by a loop are the rows whose ``parent`` is that loop.

    A helix is a maximal chain of pairs ``v_1, ..., v_L`` in which every
    ``v_a`` with ``a < L`` directly encloses exactly one pair, ``v_{a+1}``,
    and no unpaired position; ``L`` is its length. A loop's pair count is the
    number of pairs it directly encloses, plus one for its closing pair when
    it is not the exterior loop. A loop carries the five-pair motif when its
    pair count is at least 5, and the three-pair motif when its pair count is
    at least 3 and it directly encloses at least one unpaired position.

    Parameters
    ----------
    tree : np.ndarray
        Integer loop table of shape ``(P + 1, 4)``.

    Returns
    -------
    np.ndarray
        Integer array ``[shortest helix length, number of helices, number of
        loops with the five-pair motif, number of loops with the three-pair
        motif]``; the shortest helix length is 0 when ``P = 0``.

    Raises
    ------
    ValueError
        If ``tree`` is not a two-dimensional integer array with four columns
        whose row 0 has parent ``-1``, whose every row has a nonnegative
        unpaired count, and whose every row ``k >= 1`` has a parent in
        ``[0, k - 1]``.
    """
    return profile

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_profile_helix_motifs(tree: np.ndarray) -> np.ndarray:
    """Reference implementation (helix chains followed from their first pair)."""
    import numpy as np

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

    def _continues(v):
        # A pair extends its helix into the single pair it encloses.
        return v >= 1 and len(children[v]) == 1 and unpaired[v] == 0

    lengths = []
    for k in range(1, n_pairs + 1):
        if _continues(parent[k]):
            continue  # k is inside a helix that started higher up
        length, v = 1, k
        while _continues(v):
            v = children[v][0]
            length += 1
        lengths.append(length)
    five_pair, three_pair = 0, 0
    for v in range(n_pairs + 1):
        pair_count = len(children[v]) + (1 if v >= 1 else 0)
        if pair_count >= 5:
            five_pair += 1
        if pair_count >= 3 and unpaired[v] >= 1:
            three_pair += 1
    shortest = min(lengths) if lengths else 0
    return np.array([shortest, len(lengths), five_pair, three_pair], dtype=np.int64)

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
        "def _code(a):\n"
        "    a = np.asarray(a)\n"
        "    if a.shape != (4,):\n"
        "        return -1.0\n"
        "    return float(a[0] + a[1] / 100.0 + a[2] / 10000.0 + a[3] / 1000000.0)\n"
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
            "call": "_code(profile_helix_motifs(_tree('(((((...))((((....))))))).((((...)))).')))",
            "gold_call": "_code(_oracle_profile_helix_motifs(_tree('(((((...))((((....))))))).((((...)))).')))",
        },
        {
            "setup": tree,
            "call": "_code(profile_helix_motifs(_tree('((.((...)).))(((.....)))..(.((...)))')))",
            "gold_call": "_code(_oracle_profile_helix_motifs(_tree('((.((...)).))(((.....)))..(.((...)))')))",
        },
        {
            "setup": tree,
            "call": "_code(profile_helix_motifs(_tree('((((...))(...)((...)).))(((...))((...))((...))((...)))')))",
            "gold_call": "_code(_oracle_profile_helix_motifs(_tree('((((...))(...)((...)).))(((...))((...))((...))((...)))')))",
        },
        {
            "setup": tree,
            "call": "_code(profile_helix_motifs(_tree('.((...))((...))((...)).((...))((...))')))",
            "gold_call": "_code(_oracle_profile_helix_motifs(_tree('.((...))((...))((...)).((...))((...))')))",
        },
        {
            "setup": tree,
            "call": "_code(profile_helix_motifs(_tree('((((...))((...)).))((.((....))))')))",
            "gold_call": "_code(_oracle_profile_helix_motifs(_tree('((((...))((...)).))((.((....))))')))",
        },
        {
            "setup": tree,
            "call": "_code(profile_helix_motifs(_tree('.....')))",
            "gold_call": "_code(_oracle_profile_helix_motifs(_tree('.....')))",
        },
        {
            "setup": tree + status,
            "call": "_status(lambda: profile_helix_motifs(np.array([[-1, 0, 5, 0], [2, 1, 4, 2], [0, 2, 3, 0]])))",
            "gold_call": "_status(lambda: _oracle_profile_helix_motifs(np.array([[-1, 0, 5, 0], [2, 1, 4, 2], [0, 2, 3, 0]])))",
        },
        {
            "setup": tree + status,
            "call": "_status(lambda: profile_helix_motifs(np.array([[-1.0, 0.0, 3.0, 2.0]])))",
            "gold_call": "_status(lambda: _oracle_profile_helix_motifs(np.array([[-1.0, 0.0, 3.0, 2.0]])))",
        },
    ]
