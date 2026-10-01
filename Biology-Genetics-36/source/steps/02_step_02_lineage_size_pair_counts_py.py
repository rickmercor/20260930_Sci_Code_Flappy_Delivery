"""
Expected numbers of ordered pairs of lineages of given sizes in pairs of coalescent states.

A mutation on a branch of the genealogy is inherited by every sampled sequence

that descends from that branch, so the number of sequences carrying the derived

allele at a segregating site, its derived count, is the size of the branch on

which the mutation arose. The probability of a mutation pattern therefore

depends on how the branch sizes are distributed across the coalescent states,

and for patterns of two mutations on how the sizes of two branches, possibly in

different states, are distributed jointly. Sizes in different states are not

independent: a lineage in an older state is the ancestor of some of the

lineages in a younger state, and its size is the sum of theirs. This step

provides the joint expectations of the numbers of lineages of given sizes in

pairs of states, which every two-mutation calculation needs.

Returns
-------
np.ndarray of shape (n - 1, n - 1, n - 1, n - 1), the expected numbers of ordered pairs of lineages, counts[a, b, i - 1, j - 1] = E[l_k(i) l_k'(j)] for the states with k = a + 2 and k' = b + 2 lineages
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def lineage_size_pair_counts(sample_size: int) -> "np.ndarray":
    '''Expected numbers of ordered pairs of lineages of given sizes in pairs of coalescent states.

    The genealogy of sample_size sequences follows the standard coalescent
    topology: at every coalescence a uniformly chosen pair of the lineages then
    present merges, independently of the waiting times and of all earlier
    merges. A lineage in the state with k ancestral lineages is of size i when
    exactly i of the sampled sequences descend from it, so the sizes in that
    state are positive and sum to sample_size. Let l_k(i) be the number of
    lineages of size i in the state with k lineages. Return the expected
    products E[l_k(i) l_k'(j)] for every pair of states and every pair of
    sizes: the expected number of ordered pairs (a lineage of size i in the
    state with k lineages, a lineage of size j in the state with k' lineages),
    in which the pair of a lineage with itself is counted when the two states
    coincide and i = j.

    Parameters
    ----------
    sample_size : int
        Number of sampled sequences n, >= 2.

    Returns
    -------
    counts : np.ndarray
        Shape (n - 1, n - 1, n - 1, n - 1). counts[a, b, i - 1, j - 1] is
        E[l_k(i) l_k'(j)] for the states with k = a + 2 and k' = b + 2 lineages
        and the sizes i and j from 1 to n - 1. The array is symmetric under the
        exchange of (a, i) with (b, j). Every entry is an exact rational number
        represented to a relative error below 1e-12.

    Raises
    ------
    ValueError
        If sample_size is not an integer >= 2.
    '''
    return counts  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _compositions(total: int, parts: int) -> int:
    """Number of ways to write total as an ordered sum of `parts` positive integers."""
    import math
    if parts < 0 or total < 0:
        return 0
    if parts == 0:
        return 1 if total == 0 else 0
    if total < parts:
        return 0
    return math.comb(total - 1, parts - 1)


def _oracle_lineage_size_pair_counts(sample_size: int) -> "np.ndarray":
    if isinstance(sample_size, bool) or not isinstance(sample_size, (int, np.integer)) or int(sample_size) < 2:
        raise ValueError("sample_size must be an integer >= 2")
    n = int(sample_size)
    counts = np.zeros((n - 1, n - 1, n - 1, n - 1))
    # in a state with k lineages the sizes are uniform over the ordered compositions of n into k parts
    for k in range(2, n + 1):
        ck = _compositions(n, k)
        for i in range(1, n):
            for j in range(1, n):
                value = k * (k - 1) * _compositions(n - i - j, k - 2) / ck
                if i == j:
                    value += k * _compositions(n - i, k - 1) / ck
                counts[k - 2, k - 2, i - 1, j - 1] = value
    # two states k < k': the grouping of the k' lineages into their k ancestors is independent of the
    # sizes at k'; an ancestor has m descendant lineages at k' with the size probability of a sample of k'
    for k in range(2, n + 1):
        for kp in range(k + 1, n + 1):
            ckp = _compositions(n, kp)
            for i in range(1, n):
                for j in range(1, n):
                    total = 0.0
                    for m in range(1, kp - k + 2):
                        pm = _compositions(kp - m, k - 1) / _compositions(kp, k)
                        # the lineage of size j at k' descends from the lineage of size i at k
                        total += k * m * pm * _compositions(i - j, m - 1) * _compositions(n - i, kp - m) / ckp
                        # the lineage of size j at k' descends from another ancestor
                        total += k * (kp - m) * pm * _compositions(i, m) * _compositions(n - i - j, kp - m - 1) / ckp
                    counts[k - 2, kp - 2, i - 1, j - 1] = total
                    counts[kp - 2, k - 2, j - 1, i - 1] = total
    return counts

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: fourteen sequences ---
        {
            "setup": "import numpy as np\n",
            "call": "lineage_size_pair_counts(14)",
            "gold_call": "_oracle_lineage_size_pair_counts(14)",
        },
        # --- boundary: three sequences, two states, every entry checkable by hand ---
        {
            "setup": "import numpy as np\n",
            "call": "lineage_size_pair_counts(3)",
            "gold_call": "_oracle_lineage_size_pair_counts(3)",
        },
        # --- edge: a larger sample, where nested and disjoint configurations across distant states dominate ---
        {
            "setup": "import numpy as np\n",
            "call": "lineage_size_pair_counts(18)",
            "gold_call": "_oracle_lineage_size_pair_counts(18)",
        },
    ]
