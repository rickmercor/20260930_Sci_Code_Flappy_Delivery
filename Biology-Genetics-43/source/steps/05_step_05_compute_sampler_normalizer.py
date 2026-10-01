"""
Sum, over every set of residues reserved for unpaired positions, the number of proper sequences whose A-U-type pairs use only the remaining residues, and count the sets that admit at least one sequence.

Fixing which residues of the level modulo m belong to unpaired positions turns the separation requirement into local checks, so a sampler for a fixed modulus can first pick such a set in proportion to its number of sequences.

Returns
-------
np.ndarray: integer array [sum of per-set sequence counts, number of sets with a sequence].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_sampler_normalizer(tree: np.ndarray, modulus: int) -> np.ndarray:
    """Return the per-residue-set sequence total and the number of admissible sets.

    For a residue set ``S`` contained in ``{0, ..., modulus - 1}``, let
    ``F(S)`` be the count returned by ``count_restricted_designs(tree,
    modulus, S, T)`` with ``T`` the complement of ``S`` in ``{0, ...,
    modulus - 1}``: the number of proper sequences whose unpaired positions
    all have level residues in ``S`` and whose A-U and U-A pairs all have
    level residues outside ``S``. Return the sum of ``F(S)`` over all
    ``2 ** modulus`` sets ``S`` (the empty set and the full set included)
    and the number of sets with ``F(S) > 0``.

    Parameters
    ----------
    tree : np.ndarray
        Integer loop table of shape ``(P + 1, 4)`` as returned by
        ``build_loop_tree``.
    modulus : int
        Positive modulus applied to levels.

    Returns
    -------
    np.ndarray
        Integer array ``[total, admissible]``.

    Raises
    ------
    ValueError
        If ``modulus`` is not a positive integer (booleans are rejected) or
        ``tree`` is not a valid loop table in the sense of
        ``count_restricted_designs``.
    """
    return normalizer

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_sampler_normalizer(tree: np.ndarray, modulus: int) -> np.ndarray:
    """Reference implementation (one restricted count per residue set)."""
    import numpy as np

    if isinstance(modulus, bool) or not isinstance(modulus, (int, np.integer)) or modulus < 1:
        raise ValueError("modulus must be a positive integer")
    m = int(modulus)
    total, admissible = 0, 0
    for mask in range(2 ** m):
        leaf = [r for r in range(m) if (mask >> r) & 1]
        gray = [r for r in range(m) if not (mask >> r) & 1]
        count = int(_oracle_count_restricted_designs(tree, m, leaf, gray)[0])
        total += count
        admissible += 1 if count > 0 else 0
    return np.array([total, admissible], dtype=np.int64)

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
        "def _pair(a):\n"
        "    a = np.asarray(a)\n"
        "    if a.shape != (2,) or not np.issubdtype(a.dtype, np.integer):\n"
        "        return -1.0\n"
        "    return float(a[0]) / 1000.0 + float(a[1]) / 1.0e6\n"
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
            "call": "_pair(compute_sampler_normalizer(_tree('((((...))((....))))'), 3))",
            "gold_call": "_pair(_oracle_compute_sampler_normalizer(_tree('((((...))((....))))'), 3))",
        },
        {
            "setup": tree,
            "call": "_pair(compute_sampler_normalizer(_tree('((.....))(((..((((...))((....)))).)))..'), 4))",
            "gold_call": "_pair(_oracle_compute_sampler_normalizer(_tree('((.....))(((..((((...))((....)))).)))..'), 4))",
        },
        {
            "setup": tree,
            "call": "_pair(compute_sampler_normalizer(_tree('((((...))((...))((...))))'), 2))",
            "gold_call": "_pair(_oracle_compute_sampler_normalizer(_tree('((((...))((...))((...))))'), 2))",
        },
        {
            "setup": tree,
            "call": "_pair(compute_sampler_normalizer(_tree('.((.((....)).))(((...)))'), 5))",
            "gold_call": "_pair(_oracle_compute_sampler_normalizer(_tree('.((.((....)).))(((...)))'), 5))",
        },
        {
            "setup": tree,
            "call": "_pair(compute_sampler_normalizer(_tree('....'), 1))",
            "gold_call": "_pair(_oracle_compute_sampler_normalizer(_tree('....'), 1))",
        },
        {
            "setup": tree + status,
            "call": "_status(lambda: compute_sampler_normalizer(_tree('((...))'), 0))",
            "gold_call": "_status(lambda: _oracle_compute_sampler_normalizer(_tree('((...))'), 0))",
        },
    ]
