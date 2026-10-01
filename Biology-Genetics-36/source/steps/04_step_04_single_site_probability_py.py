"""
Probability that a non-recombining locus shows exactly one segregating site, with a given derived count and with any count.

The simplest pattern of variation at a short locus is a single segregating site,

and the derived count of that site carries information about the history of the

population: the branches on which a mutation of a given size can arise sit at

particular depths of the genealogy, and a change in population size while a

given number of lineages existed lengthens or shortens exactly those branches.

The probability of observing one site with a given derived count is an

expectation over the genealogy that couples the length of the branches of that

size with the exponential factor in the total length, and its ratio to the

probability of observing one site of any count is the exact conditional

distribution of the count. This step computes both probabilities.

Returns
-------
np.ndarray of shape (2,), the probability of exactly one segregating site of derived count `count` and the probability of exactly one segregating site of any count
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def single_site_probability(relative_sizes: "np.ndarray", theta: float, count: int) -> "np.ndarray":
    '''Probability that a non-recombining locus shows exactly one segregating site, with a given derived count and with any count.

    The sample of n = len(relative_sizes) + 1 sequences, its genealogy, the
    size history relative_sizes (entry j for the state with j + 2 lineages,
    relative to N_ref), the time unit of 4 N_ref generations with the
    coalescence rate k (k - 1) / relative_sizes[k - 2] while k lineages exist,
    the uniform merging of lineages, and the infinite-sites mutation process of
    rate theta per unit branch length are those of the previous steps. Return
    the probability that the locus shows exactly one segregating site whose
    derived allele is carried by exactly `count` sequences, and the probability
    that it shows exactly one segregating site of any derived count.

    Parameters
    ----------
    relative_sizes : np.ndarray
        Shape (n - 1,), n >= 2; finite entries > 0.
    theta : float
        Mutation parameter of the locus, 4 N_ref mu, a finite number > 0.
    count : int
        Derived count of the site, from 1 to n - 1.

    Returns
    -------
    probabilities : np.ndarray
        Shape (2,). probabilities[0] is the probability of exactly one
        segregating site of derived count `count`; probabilities[1] is the
        probability of exactly one segregating site. Both accurate to a
        relative error below 1e-10.

    Raises
    ------
    ValueError
        If relative_sizes is not a one-dimensional array with at least one
        finite entry > 0, if theta is not a finite number > 0, or if count is
        not an integer from 1 to n - 1.
    '''
    return probabilities  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _site_count_distribution(v: "np.ndarray", theta: float, max_sites: int) -> "np.ndarray":
    """P(K = 0), ..., P(K = max_sites): the numbers of mutations in the states are independent geometric variables."""
    k = np.arange(2, v.size + 2, dtype=float)
    lam = (k - 1.0) / v
    dist = np.zeros(max_sites + 1)
    dist[0] = 1.0
    for rate in lam:
        q = theta / (rate + theta)
        geo = (1.0 - q) * q ** np.arange(max_sites + 1)
        dist = np.convolve(dist, geo)[: max_sites + 1]
    return dist


def _oracle_single_site_probability(relative_sizes: "np.ndarray", theta: float, count: int) -> "np.ndarray":
    v = _check_sizes(relative_sizes)
    th = _check_theta(theta)
    n = v.size + 1
    i = _check_count(count, n, "count")
    k = np.arange(2, n + 1, dtype=float)
    lam = (k - 1.0) / v
    fac = lam / (lam + th)
    allprod = float(np.prod(fac))
    # theta E[L_i exp(-theta L)]: a random lineage in the state with k lineages has size i with
    # probability C(n-i-1, k-2)/C(n-1, k-1) (the compositions of n into k parts), and there are k of them
    total = 0.0
    for a in range(n - 1):
        kk = a + 2
        p_size = _compositions(n - i, kk - 1) / _compositions(n, kk)
        total += p_size * lam[a] / (lam[a] + th) ** 2 * allprod / fac[a]
    p_count = th * total
    p_one = float(_site_count_distribution(v, th, 1)[1])
    return np.array([p_count, p_one])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the reduced-size history, a site carried by 12 of 14 sequences ---
        {
            "setup": "import numpy as np\nsizes = np.array([1.0, 0.15, 0.15, 0.15] + [1.0] * 9)\n",
            "call": "single_site_probability(sizes, 0.32, 12)",
            "gold_call": "_oracle_single_site_probability(sizes, 0.32, 12)",
        },
        # --- boundary: three sequences in a constant population, a singleton ---
        {
            "setup": "import numpy as np\nsizes = np.ones(2)\n",
            "call": "single_site_probability(sizes, 0.1, 1)",
            "gold_call": "_oracle_single_site_probability(sizes, 0.1, 1)",
        },
        # --- edge: a bottleneck while the last three lineages exist, a site carried by all but one of
        #     twenty sequences, a large mutation parameter ---
        {
            "setup": "import numpy as np\nsizes = np.array([0.05, 0.05, 0.05] + [1.0] * 16)\n",
            "call": "single_site_probability(sizes, 0.7, 19)",
            "gold_call": "_oracle_single_site_probability(sizes, 0.7, 19)",
        },
    ]
