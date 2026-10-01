"""
Joint probability of a three-locus observation under two size histories, and the decision between them.

Patterns of segregating sites at short loci are informative about the history

of the population that produced them, because a change in size while a given

number of ancestral lineages existed reshapes the lengths of the branches of

particular sizes. When several unlinked loci are sequenced in the same sample,

their genealogies are independent draws from the same history, and the

probability of the whole observation under a history is the product of the

loci's pattern probabilities. Comparing that joint probability between candidate

histories is the likelihood comparison on which model choice rests. This step

carries out that comparison for three loci, one showing two segregating sites

and two showing a single site each.

Returns
-------
np.ndarray of shape (3,), the joint probability of the three-locus observation under sizes_null, under sizes_alt, and the decision (1.0 for sizes_alt, 0.0 for sizes_null)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def history_decision(sizes_null: "np.ndarray", sizes_alt: "np.ndarray", theta_a: float, count_a1: int,
                     count_a2: int, theta_b: float, count_b: int, theta_c: float, count_c: int) -> "np.ndarray":
    '''Joint probability of a three-locus observation under two size histories, and the decision between them.

    Three unlinked non-recombining loci are sequenced in the same sample of
    n = len(sizes_null) + 1 sequences, so their genealogies are independent
    draws from the same history. Locus A shows exactly two segregating sites
    with derived counts count_a1 and count_a2 (an unordered pair) and has the
    mutation parameter theta_a; locus B shows exactly one segregating site
    with derived count count_b and has the mutation parameter theta_b; locus C
    shows exactly one segregating site with derived count count_c and has the
    mutation parameter theta_c. The size histories sizes_null and sizes_alt
    follow the convention of the earlier steps (entry j is the relative size
    while j + 2 lineages exist), as do the coalescent, its time unit and the
    mutation process. Return the probability of the joint observation under
    each history, and the decision: 1.0 when the joint probability under
    sizes_alt is strictly larger than under sizes_null, 0.0 otherwise.

    Parameters
    ----------
    sizes_null : np.ndarray
        Shape (n - 1,), n >= 2; finite entries > 0.
    sizes_alt : np.ndarray
        Shape (n - 1,), the same length as sizes_null; finite entries > 0.
    theta_a : float
        Mutation parameter of locus A, a finite number > 0.
    count_a1 : int
        Derived count of one site of locus A, from 1 to n - 1.
    count_a2 : int
        Derived count of the other site of locus A, from 1 to n - 1.
    theta_b : float
        Mutation parameter of locus B, a finite number > 0.
    count_b : int
        Derived count of the site of locus B, from 1 to n - 1.
    theta_c : float
        Mutation parameter of locus C, a finite number > 0.
    count_c : int
        Derived count of the site of locus C, from 1 to n - 1.

    Returns
    -------
    result : np.ndarray
        Shape (3,). result[0] and result[1] are the joint probabilities of the
        observation under sizes_null and sizes_alt, accurate to a relative
        error below 1e-10; result[2] is the decision, 1.0 for sizes_alt and
        0.0 for sizes_null.

    Raises
    ------
    ValueError
        If sizes_null or sizes_alt is not a one-dimensional array with at
        least one finite entry > 0, if their lengths differ, if theta_a,
        theta_b or theta_c is not a finite number > 0, or if count_a1,
        count_a2, count_b or count_c is not an integer from 1 to n - 1.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_history_decision(sizes_null: "np.ndarray", sizes_alt: "np.ndarray", theta_a: float, count_a1: int,
                             count_a2: int, theta_b: float, count_b: int, theta_c: float, count_c: int) -> "np.ndarray":
    v0 = _check_sizes(sizes_null)
    v1 = _check_sizes(sizes_alt)
    if v0.size != v1.size:
        raise ValueError("sizes_null and sizes_alt must have the same length")
    n = v0.size + 1
    _check_count(count_b, n, "count_b")
    _check_count(count_c, n, "count_c")
    joint = []
    for v in (v0, v1):
        pattern = _oracle_two_site_pattern_probability(v, theta_a, count_a1, count_a2)
        single_b = _oracle_single_site_probability(v, theta_b, count_b)[0]
        single_c = _oracle_single_site_probability(v, theta_c, count_c)[0]
        joint.append(pattern * single_b * single_c)       # independent loci: the probabilities multiply
    decision = 1.0 if joint[1] > joint[0] else 0.0
    return np.array([joint[0], joint[1], decision])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the shipped comparison, decided for the reduced-size history ---
        {
            "setup": ("import numpy as np\nnull = np.ones(13)\n"
                      "alt = np.array([1.0, 0.15, 0.15, 0.15] + [1.0] * 9)\n"),
            "call": "history_decision(null, alt, 0.44, 6, 8, 0.32, 12, 0.40, 1)",
            "gold_call": "_oracle_history_decision(null, alt, 0.44, 6, 8, 0.32, 12, 0.40, 1)",
        },
        # --- boundary: two identical histories, a tie, decided for the null by the tie rule ---
        {
            "setup": ("import numpy as np\nnull = np.ones(13)\n"
                      "alt = np.ones(13)\n"),
            "call": "history_decision(null, alt, 0.44, 6, 8, 0.32, 12, 0.40, 1)",
            "gold_call": "_oracle_history_decision(null, alt, 0.44, 6, 8, 0.32, 12, 0.40, 1)",
        },
        # --- edge: a recent bottleneck against growth in a sample of ten, two singletons at locus A,
        #     decided for the null ---
        {
            "setup": ("import numpy as np\nnull = np.array([1.0] * 6 + [0.1] * 3)\n"
                      "alt = np.sqrt(np.arange(2, 11, dtype=float))\n"),
            "call": "history_decision(null, alt, 0.5, 1, 1, 0.25, 5, 0.3, 9)",
            "gold_call": "_oracle_history_decision(null, alt, 0.5, 1, 1, 0.25, 5, 0.3, 9)",
        },
    ]
