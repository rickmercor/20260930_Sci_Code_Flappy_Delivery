"""
Probability that a non-recombining locus shows exactly two segregating sites with given derived counts.

Under the infinite-sites model every mutation creates a new segregating site,

and conditional on the genealogy the mutations fall on its branches as a Poisson

process. The probability that a locus shows a particular pattern of segregating

sites, a given number of them with given derived counts, is therefore an

expectation over the random genealogy of a product of branch lengths and an

exponential factor in the total length. For a single segregating site this

expectation involves the length of the branches of one size; for two sites it

involves the lengths of branches of two sizes jointly, together with the sizes

being distributed across the states as the previous step describes. This step

computes the probability of a two-site pattern for a size history indexed by the

number of ancestral lineages.

Returns
-------
float, the probability that the locus shows exactly two segregating sites with the derived counts count_a and count_b as an unordered pair, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def two_site_pattern_probability(relative_sizes: "np.ndarray", theta: float, count_a: int, count_b: int) -> float:
    '''Probability that a non-recombining locus shows exactly two segregating sites with given derived counts.

    The sample holds n = len(relative_sizes) + 1 sequences whose genealogy
    follows the coalescent of the previous steps: relative_sizes[j] is the
    effective size while j + 2 ancestral lineages exist, relative to N_ref;
    time is in units of 4 N_ref generations, so that with k lineages the
    waiting time to the next coalescence is exponential with rate
    k (k - 1) / relative_sizes[k - 2]; and at every coalescence a uniformly
    chosen pair of lineages merges. Mutations arise along the genealogy as a
    Poisson process of rate theta per unit branch length, theta being
    4 N_ref mu for the whole locus, under the infinite-sites model, so that
    every mutation is a segregating site whose derived allele is carried by the
    sequences descending from the branch on which it arose. Return the
    probability that the locus shows exactly two segregating sites and that
    their derived counts are count_a and count_b as an unordered pair: when
    count_a != count_b, one site has derived count count_a and the other
    count_b, in either assignment; when count_a == count_b, both sites have
    that derived count.

    Parameters
    ----------
    relative_sizes : np.ndarray
        Shape (n - 1,), n >= 2; finite entries > 0.
    theta : float
        Mutation parameter of the locus, 4 N_ref mu, a finite number > 0.
    count_a : int
        Derived count of one site, from 1 to n - 1.
    count_b : int
        Derived count of the other site, from 1 to n - 1.

    Returns
    -------
    probability : float
        The pattern probability as a native Python float, accurate to a
        relative error below 1e-10.

    Raises
    ------
    ValueError
        If relative_sizes is not a one-dimensional array with at least one
        finite entry > 0, if theta is not a finite number > 0, or if count_a or
        count_b is not an integer from 1 to n - 1.
    '''
    return probability  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_count(value, n: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 1 <= int(value) <= n - 1:
        raise ValueError(f"{name} must be an integer from 1 to n - 1")
    return int(value)


def _check_theta(theta) -> float:
    if isinstance(theta, bool) or not isinstance(theta, (int, float, np.integer, np.floating)):
        raise ValueError("theta must be a finite number > 0")
    if not np.isfinite(float(theta)) or float(theta) <= 0.0:
        raise ValueError("theta must be a finite number > 0")
    return float(theta)


def _pair_time_factors(v: "np.ndarray", theta: float) -> "np.ndarray":
    """T[a, b] = E[t_k t_k' exp(-theta L)] for the states k = a + 2, k' = b + 2, with t_k the waiting time with k lineages."""
    n = v.size + 1
    k = np.arange(2, n + 1, dtype=float)
    lam = (k - 1.0) / v                                  # rates of the scaled times k t_k
    fac = lam / (lam + theta)                            # E[exp(-theta k t_k)]
    allprod = float(np.prod(fac))
    T = np.zeros((n - 1, n - 1))
    for a in range(n - 1):
        la, ka = lam[a], k[a]
        T[a, a] = 2.0 * la / (la + theta) ** 3 * allprod / fac[a] / ka ** 2
        for b in range(a + 1, n - 1):
            lb, kb = lam[b], k[b]
            T[a, b] = T[b, a] = (la / (la + theta) ** 2) * (lb / (lb + theta) ** 2) * allprod / fac[a] / fac[b] / (ka * kb)
    return T


def _oracle_two_site_pattern_probability(relative_sizes: "np.ndarray", theta: float, count_a: int, count_b: int) -> float:
    v = _check_sizes(relative_sizes)
    th = _check_theta(theta)
    n = v.size + 1
    i = _check_count(count_a, n, "count_a")
    j = _check_count(count_b, n, "count_b")
    counts = _oracle_lineage_size_pair_counts(n)
    T = _pair_time_factors(v, th)
    # E[L_i L_j exp(-theta L)] = sum over pairs of states of E[l_k(i) l_k'(j)] E[t_k t_k' exp(-theta L)]
    mixed = float(np.sum(counts[:, :, i - 1, j - 1] * T))
    # two sites of the same count are two mutations of one Poisson class: (theta L_i)^2 / 2
    return th ** 2 * mixed / (2.0 if i == j else 1.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the reduced-size history, derived counts 6 and 8 in fourteen sequences ---
        {
            "setup": "import numpy as np\nsizes = np.array([1.0, 0.15, 0.15, 0.15] + [1.0] * 9)\n",
            "call": "two_site_pattern_probability(sizes, 0.44, 6, 8)",
            "gold_call": "_oracle_two_site_pattern_probability(sizes, 0.44, 6, 8)",
        },
        # --- boundary: two sites of the same derived count in a constant population ---
        {
            "setup": "import numpy as np\nsizes = np.ones(4)\n",
            "call": "two_site_pattern_probability(sizes, 0.3, 2, 2)",
            "gold_call": "_oracle_two_site_pattern_probability(sizes, 0.3, 2, 2)",
        },
        # --- edge: a singleton and a site carried by all but one sequence, in a growing population
        #     with a large mutation parameter ---
        {
            "setup": "import numpy as np\nsizes = np.sqrt(np.arange(2, 13, dtype=float))\n",
            "call": "two_site_pattern_probability(sizes, 0.9, 1, 11)",
            "gold_call": "_oracle_two_site_pattern_probability(sizes, 0.9, 1, 11)",
        },
    ]
