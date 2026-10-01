"""
Find the smallest modulus of at least 2 at which a loop tree admits a sequence whose A-U-type pairs and unpaired positions occupy disjoint level residues.

Separation modulo m is solvable in time linear in the structure length for each fixed m, so the smallest workable modulus is found by trying m = 2, 3, and so on, after ruling out the locally undesignable motifs.

Returns
-------
int: the smallest modulus m >= 2 with a positive per-residue-set sequence total.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def find_minimal_modulus(tree: np.ndarray, helix_profile: np.ndarray, max_modulus: int) -> int:
    """Return the smallest workable modulus of a loop tree.

    ``helix_profile`` is the array ``[shortest helix length, number of
    helices, loops with the five-pair motif, loops with the three-pair
    motif]`` returned by ``profile_helix_motifs(tree)``. If it reports any
    loop with either motif, the structure has no design and ``ValueError``
    is raised. If there is at least one helix and every helix has at least
    three pairs, a sequence is guaranteed at modulus 2 and 2 is returned
    without further computation. Otherwise return the smallest ``m`` in
    ``[2, max_modulus]`` for which the total returned by
    ``compute_sampler_normalizer(tree, m)`` is positive.

    Parameters
    ----------
    tree : np.ndarray
        Integer loop table of shape ``(P + 1, 4)`` as returned by
        ``build_loop_tree``.
    helix_profile : np.ndarray
        Length-4 array of nonnegative integers from ``profile_helix_motifs``.
    max_modulus : int
        Largest modulus tried; at least 2.

    Returns
    -------
    int
        The smallest workable modulus.

    Raises
    ------
    ValueError
        If ``helix_profile`` is not a length-4 array of nonnegative
        integers, if ``max_modulus`` is not an integer of at least 2
        (booleans are rejected), if the profile reports a loop with either
        motif, if no modulus up to ``max_modulus`` admits a sequence, or,
        when the modulus scan runs, if ``tree`` is not a valid loop table in
        the sense of ``count_restricted_designs``.
    """
    return modulus

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_find_minimal_modulus(tree: np.ndarray, helix_profile: np.ndarray, max_modulus: int) -> int:
    """Reference implementation (motif guard, helix-length guarantee, then an increasing scan)."""
    import numpy as np

    def _is_int(value):
        return not isinstance(value, bool) and isinstance(value, (int, np.integer))

    profile = np.asarray(helix_profile)
    if (profile.shape != (4,) or not np.issubdtype(profile.dtype, np.integer)
            or np.any(profile < 0)):
        raise ValueError("helix_profile must be a length-4 array of nonnegative integers")
    if not (_is_int(max_modulus) and max_modulus >= 2):
        raise ValueError("max_modulus must be an integer of at least 2")
    shortest, n_helices, five_pair, three_pair = (int(x) for x in profile)
    if five_pair + three_pair > 0:
        raise ValueError("a loop carries a locally undesignable motif")
    if n_helices > 0 and shortest >= 3:
        return 2  # every helix has three or more pairs: modulus 2 always works
    for m in range(2, int(max_modulus) + 1):
        if int(_oracle_compute_sampler_normalizer(tree, m)[0]) > 0:
            return m
    raise ValueError("no modulus up to max_modulus admits a separated sequence")

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
            "call": "find_minimal_modulus(_tree('((((...))((...))((...))))'), np.array([2, 4, 0, 0]), 9)",
            "gold_call": "_oracle_find_minimal_modulus(_tree('((((...))((...))((...))))'), np.array([2, 4, 0, 0]), 9)",
        },
        {
            "setup": tree,
            "call": "find_minimal_modulus(_tree('(...((.(......))).)(....)((.....))'), np.array([1, 5, 0, 0]), 9)",
            "gold_call": "_oracle_find_minimal_modulus(_tree('(...((.(......))).)(....)((.....))'), np.array([1, 5, 0, 0]), 9)",
        },
        {
            "setup": tree,
            "call": "find_minimal_modulus(_tree('((...))((.(....)))(...(((......))))'), np.array([1, 5, 0, 0]), 7)",
            "gold_call": "_oracle_find_minimal_modulus(_tree('((...))((.(....)))(...(((......))))'), np.array([1, 5, 0, 0]), 7)",
        },
        {
            "setup": tree,
            "call": "find_minimal_modulus(_tree('..(((((....)))(((...)))))..(....)..'), np.array([1, 4, 0, 0]), 6)",
            "gold_call": "_oracle_find_minimal_modulus(_tree('..(((((....)))(((...)))))..(....)..'), np.array([1, 4, 0, 0]), 6)",
        },
        {
            "setup": tree,
            "call": "find_minimal_modulus(_tree('((((((...)))((((....))))))).((((...))))'), np.array([3, 4, 0, 0]), 2)",
            "gold_call": "_oracle_find_minimal_modulus(_tree('((((((...)))((((....))))))).((((...))))'), np.array([3, 4, 0, 0]), 2)",
        },
        {
            "setup": tree,
            "call": "find_minimal_modulus(_tree('......'), np.array([0, 0, 0, 0]), 2)",
            "gold_call": "_oracle_find_minimal_modulus(_tree('......'), np.array([0, 0, 0, 0]), 2)",
        },
        {
            "setup": tree + status,
            "call": "_status(lambda: find_minimal_modulus(_tree('(...((.(......))).)(....)((.....))'), np.array([1, 5, 0, 0]), 4))",
            "gold_call": "_status(lambda: _oracle_find_minimal_modulus(_tree('(...((.(......))).)(....)((.....))'), np.array([1, 5, 0, 0]), 4))",
        },
        {
            "setup": tree + status,
            "call": "_status(lambda: find_minimal_modulus(_tree('(((...))((...))(...).)'), np.array([1, 4, 0, 1]), 9))",
            "gold_call": "_status(lambda: _oracle_find_minimal_modulus(_tree('(((...))((...))(...).)'), np.array([1, 4, 0, 1]), 9))",
        },
        {
            "setup": tree + status,
            "call": "_status(lambda: find_minimal_modulus(_tree('((((...))((...))((...))))'), np.array([2, 4, 1, 0]), 9))",
            "gold_call": "_status(lambda: _oracle_find_minimal_modulus(_tree('((((...))((...))((...))))'), np.array([2, 4, 1, 0]), 9))",
        },
        {
            "setup": tree + status,
            "call": "_status(lambda: find_minimal_modulus(_tree('((...))'), np.array([2, 1, 0]), 9))",
            "gold_call": "_status(lambda: _oracle_find_minimal_modulus(_tree('((...))'), np.array([2, 1, 0]), 9))",
        },
    ]
