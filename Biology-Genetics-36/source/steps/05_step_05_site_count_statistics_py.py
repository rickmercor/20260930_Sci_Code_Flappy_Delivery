"""
Exact probability of a given number of segregating sites, the accuracy of its series evaluation, and a central moment of the number of sites.

Conditional on the genealogy, the number of segregating sites at a locus is a

Poisson count whose mean is the mutation parameter times the total branch

length, so its unconditional distribution and all of its moments are set by the

distribution of the length. The power moments of the length computed in the

first step give the moments of the number of sites directly, and they also give

its probabilities through an alternating series obtained by expanding the

exponential factor of the Poisson probability: the series converges, but the

number of terms it needs grows with the mutation parameter and with the number

of sites, so its accuracy has to be judged against an exact evaluation. This

step returns the exact probability of a given number of sites, the accuracy

behaviour of the series for it, and a central moment of the number of sites.

Returns
-------
np.ndarray of shape (4,), the exact probability that K = n_sites, the smallest series partial-sum index reaching the tolerance (as a float), that partial sum, and the central moment of K of order moment_order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def site_count_statistics(relative_sizes: "np.ndarray", theta: float, n_sites: int, tolerance: float,
                          moment_order: int) -> "np.ndarray":
    '''Exact probability of a given number of segregating sites, the accuracy of its series evaluation, and a central moment of the number of sites.

    The sample of n = len(relative_sizes) + 1 sequences, its genealogy, the
    size history relative_sizes (entry j for the state with j + 2 lineages,
    relative to N_ref), the time unit of 4 N_ref generations with the
    coalescence rate k (k - 1) / relative_sizes[k - 2] while k lineages exist,
    and the infinite-sites mutation process of rate theta per unit branch
    length are those of the previous steps; K denotes the number of
    segregating sites of the locus. Three quantities are returned. First, the
    exact probability that K = n_sites. Second, the accuracy behaviour of the
    series evaluation of that probability: expanding the exponential factor of
    the Poisson probability of n_sites mutations in powers of theta times the
    total length and taking the expectation term by term expresses the
    probability as an alternating series in the power moments of the length,
    whose partial sum of index h keeps the terms of order theta^n_sites to
    theta^(n_sites + h); return the smallest h >= 0 at which that partial sum
    differs from the exact probability by less than `tolerance` in relative
    terms, and the partial sum itself. Third, the central moment of K of order
    moment_order, E[(K - E[K])^moment_order].

    Parameters
    ----------
    relative_sizes : np.ndarray
        Shape (n - 1,), n >= 2; finite entries > 0.
    theta : float
        Mutation parameter of the locus, 4 N_ref mu, a finite number > 0.
    n_sites : int
        Number of segregating sites whose probability is returned, >= 0.
    tolerance : float
        Relative accuracy required of the series partial sum, a finite number
        in (0, 1).
    moment_order : int
        Order of the central moment of K to return, >= 0.

    Returns
    -------
    statistics : np.ndarray
        Shape (4,). statistics[0] is the exact probability that K = n_sites;
        statistics[1] is the smallest partial-sum index h that reaches the
        tolerance, as a float holding an integer value; statistics[2] is the
        partial sum of that index; statistics[3] is the central moment of K of
        order moment_order. The exact probability and the central moment are
        accurate to a relative error below 1e-10.

    Raises
    ------
    ValueError
        If relative_sizes is not a one-dimensional array with at least one
        finite entry > 0, if theta is not a finite number > 0, if n_sites is
        not an integer >= 0, if tolerance is not a finite number in (0, 1), if
        moment_order is not an integer >= 0, or if the partial sums have not
        reached the tolerance by index 150.
    '''
    return statistics  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _stirling_second(m: int, k: int) -> int:
    """Stirling number of the second kind S(m, k)."""
    import math
    return sum((-1) ** (k - r) * math.comb(k, r) * r ** m for r in range(k + 1)) // math.factorial(k)


def _oracle_site_count_statistics(relative_sizes: "np.ndarray", theta: float, n_sites: int, tolerance: float,
                                  moment_order: int) -> "np.ndarray":
    import math
    v = _check_sizes(relative_sizes)
    th = _check_theta(theta)
    if isinstance(n_sites, bool) or not isinstance(n_sites, (int, np.integer)) or int(n_sites) < 0:
        raise ValueError("n_sites must be an integer >= 0")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float, np.integer, np.floating)):
        raise ValueError("tolerance must be a finite number in (0, 1)")
    if not np.isfinite(float(tolerance)) or not 0.0 < float(tolerance) < 1.0:
        raise ValueError("tolerance must be a finite number in (0, 1)")
    if isinstance(moment_order, bool) or not isinstance(moment_order, (int, np.integer)) or int(moment_order) < 0:
        raise ValueError("moment_order must be an integer >= 0")
    K, tol, order = int(n_sites), float(tolerance), int(moment_order)
    exact = float(_site_count_distribution(v, th, K)[K])
    # the series: P(K) = theta^K / K! sum_h (-1)^h theta^h mu_{h+K} / h!
    h_cap = 150
    moments = _oracle_tree_length_moments(v, h_cap + K)
    prefactor = th ** K / math.factorial(K)
    partial, h_min, value = 0.0, -1, 0.0
    for h in range(h_cap + 1):
        partial += (-1) ** h * th ** h / math.factorial(h) * moments[h + K]
        value = prefactor * partial
        if abs(value - exact) < tol * exact:
            h_min = h
            break
    if h_min < 0:
        raise ValueError("the series partial sums have not reached the tolerance by index 150")
    # power moments of K by conditioning on the length (K | L is Poisson with mean theta L), then the
    # central moment by the binomial expansion about E[K]
    power = [sum(_stirling_second(m, r) * moments[r] * th ** r for r in range(m + 1)) for m in range(order + 1)]
    mean = th * moments[1]
    central = sum((-1) ** r * math.comb(order, r) * mean ** r * power[order - r] for r in range(order + 1))
    return np.array([exact, float(h_min), value, float(central)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: two sites at the reduced-size history, one per cent accuracy, fifth central moment ---
        {
            "setup": "import numpy as np\nsizes = np.array([1.0, 0.15, 0.15, 0.15] + [1.0] * 9)\n",
            "call": "site_count_statistics(sizes, 0.44, 2, 0.01, 5)",
            "gold_call": "_oracle_site_count_statistics(sizes, 0.44, 2, 0.01, 5)",
        },
        # --- boundary: no segregating site, a small mutation parameter, the variance ---
        {
            "setup": "import numpy as np\nsizes = np.ones(7)\n",
            "call": "site_count_statistics(sizes, 0.2, 0, 1e-3, 2)",
            "gold_call": "_oracle_site_count_statistics(sizes, 0.2, 0, 1e-3, 2)",
        },
        # --- edge: four sites in a slowly growing population at a large mutation parameter and a tight
        #     tolerance, where the series needs about fifty terms ---
        {
            "setup": "import numpy as np\nsizes = np.arange(2, 31, dtype=float) ** 0.25\n",
            "call": "site_count_statistics(sizes, 0.5, 4, 1e-4, 4)",
            "gold_call": "_oracle_site_count_statistics(sizes, 0.5, 4, 1e-4, 4)",
        },
    ]
