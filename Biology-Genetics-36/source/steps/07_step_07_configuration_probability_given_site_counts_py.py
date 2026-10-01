"""
Probability of the observed derived counts at three loci given their numbers of segregating sites, under the history the observation supports.

This step returns the deliverable: for three unlinked loci sequenced in the same

sample, one showing two segregating sites and two showing a single site each, it

decides between a constant-size history and a history with a reduced size while

a stated range of ancestral lineages existed, and returns, under the decided

history, the probability of the observed derived counts given the observed

numbers of segregating sites at the loci. That conditional probability isolates

the information carried by the frequencies of the variants from the information

carried by the amount of variation.

Returns
-------
float, the product over the three loci of the probability of the observed derived counts given the number of segregating sites, under the decided history, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def configuration_probability_given_site_counts(sample_size: int, reduced_fraction: float, reduced_from: int,
                                                reduced_to: int, theta_per_site: float, length_a: float,
                                                count_a1: int, count_a2: int, length_b: float, count_b: int,
                                                length_c: float, count_c: int) -> float:
    '''Probability of the observed derived counts at three loci given their numbers of segregating sites, under the history the observation supports.

    A sample of sample_size sequences is sequenced at three unlinked
    non-recombining loci of length_a, length_b and length_c base pairs, whose
    genealogies are independent draws from the same history. Locus A shows
    exactly two segregating sites with derived counts count_a1 and count_a2
    (an unordered pair); locus B shows exactly one segregating site with
    derived count count_b; locus C shows exactly one segregating site with
    derived count count_c. The mutation parameter is theta_per_site =
    4 N_ref mu per site, N_ref being the present-day effective size, so a
    locus of length L has the mutation parameter theta_per_site * L. Two
    histories are compared, both in the model of the earlier steps in which
    the effective size is constant while a given number of ancestral lineages
    exists: history 1 keeps the size N_ref throughout; history 2 has the size
    reduced_fraction * N_ref while k lineages exist for every k from
    reduced_from to reduced_to inclusive, and N_ref otherwise. The history is
    decided as in the previous step, by the larger joint probability of the
    three loci's patterns, history 1 on a tie. Return, under the decided
    history, the product of three conditional probabilities: the probability
    of the derived counts count_a1 and count_a2 at locus A given that it shows
    exactly two segregating sites, the probability of the derived count
    count_b at locus B given that it shows exactly one, and the probability of
    the derived count count_c at locus C given that it shows exactly one.

    Parameters
    ----------
    sample_size : int
        Number of sampled sequences n, >= 3.
    reduced_fraction : float
        Size of history 2 while reduced_from to reduced_to lineages exist,
        relative to N_ref; a finite number > 0.
    reduced_from : int
        First number of lineages of the reduced range, from 2 to reduced_to.
    reduced_to : int
        Last number of lineages of the reduced range, from reduced_from to n.
    theta_per_site : float
        Mutation parameter per site, 4 N_ref mu, a finite number > 0.
    length_a : float
        Length of locus A in base pairs, a finite number > 0.
    count_a1 : int
        Derived count of one site of locus A, from 1 to n - 1.
    count_a2 : int
        Derived count of the other site of locus A, from 1 to n - 1.
    length_b : float
        Length of locus B in base pairs, a finite number > 0.
    count_b : int
        Derived count of the site of locus B, from 1 to n - 1.
    length_c : float
        Length of locus C in base pairs, a finite number > 0.
    count_c : int
        Derived count of the site of locus C, from 1 to n - 1.

    Returns
    -------
    probability : float
        The product of the three conditional probabilities under the decided
        history, as a native Python float, accurate to a relative error below
        1e-9.

    Raises
    ------
    ValueError
        If sample_size is not an integer >= 3, if reduced_fraction,
        theta_per_site, length_a, length_b or length_c is not a finite number
        > 0, if reduced_from and reduced_to are not integers with
        2 <= reduced_from <= reduced_to <= sample_size, or if count_a1,
        count_a2, count_b or count_c is not an integer from 1 to n - 1.
        Conditions raised by the earlier steps propagate unchanged; in
        particular the series evaluation of the site-count step raises if its
        partial sums have not reached one per cent by index 150, which needs a
        locus mutation parameter far above those of this problem.
    '''
    return probability  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_configuration_probability_given_site_counts(sample_size: int, reduced_fraction: float, reduced_from: int,
                                                        reduced_to: int, theta_per_site: float, length_a: float,
                                                        count_a1: int, count_a2: int, length_b: float, count_b: int,
                                                        length_c: float, count_c: int) -> float:
    if isinstance(sample_size, bool) or not isinstance(sample_size, (int, np.integer)) or int(sample_size) < 3:
        raise ValueError("sample_size must be an integer >= 3")
    n = int(sample_size)
    for name, value in (("reduced_fraction", reduced_fraction), ("theta_per_site", theta_per_site),
                        ("length_a", length_a), ("length_b", length_b), ("length_c", length_c)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a finite number > 0")
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite number > 0")
    for name, value in (("reduced_from", reduced_from), ("reduced_to", reduced_to)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("reduced_from and reduced_to must be integers with 2 <= reduced_from <= reduced_to <= sample_size")
    k_lo, k_hi = int(reduced_from), int(reduced_to)
    if not 2 <= k_lo <= k_hi <= n:
        raise ValueError("reduced_from and reduced_to must be integers with 2 <= reduced_from <= reduced_to <= sample_size")
    theta_a = float(theta_per_site) * float(length_a)
    theta_b = float(theta_per_site) * float(length_b)
    theta_c = float(theta_per_site) * float(length_c)
    sizes_null = np.ones(n - 1)
    sizes_alt = np.ones(n - 1)
    sizes_alt[k_lo - 2:k_hi - 1] = float(reduced_fraction)     # entry k - 2 holds the state with k lineages
    decision = _oracle_history_decision(sizes_null, sizes_alt, theta_a, count_a1, count_a2, theta_b, count_b,
                                        theta_c, count_c)
    sizes = sizes_alt if decision[2] > 0.5 else sizes_null
    pattern = _oracle_two_site_pattern_probability(sizes, theta_a, count_a1, count_a2)
    # the site-count step also reports the series accuracy at one per cent and the fifth central moment,
    # which the reasoning asks for; only the exact probability enters the deliverable
    two_sites = _oracle_site_count_statistics(sizes, theta_a, 2, 0.01, 5)[0]
    single_b = _oracle_single_site_probability(sizes, theta_b, count_b)
    single_c = _oracle_single_site_probability(sizes, theta_c, count_c)
    return float(pattern / two_sites * single_b[0] / single_b[1] * single_c[0] / single_c[1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the shipped configuration; this is the reported answer ---
        {
            "setup": "import numpy as np\n",
            "call": "configuration_probability_given_site_counts(14, 0.15, 3, 5, 8e-4, 550.0, 6, 8, 400.0, 12, 500.0, 1)",
            "gold_call": "_oracle_configuration_probability_given_site_counts(14, 0.15, 3, 5, 8e-4, 550.0, 6, 8, 400.0, 12, 500.0, 1)",
        },
        # --- boundary: intermediate counts at loci B and C, where the constant history is decided ---
        {
            "setup": "import numpy as np\n",
            "call": "configuration_probability_given_site_counts(14, 0.15, 3, 5, 8e-4, 550.0, 6, 8, 400.0, 4, 500.0, 5)",
            "gold_call": "_oracle_configuration_probability_given_site_counts(14, 0.15, 3, 5, 8e-4, 550.0, 6, 8, 400.0, 4, 500.0, 5)",
        },
        # --- edge: a recent reduction covering the youngest states of a sample of ten, two singletons at
        #     locus A and intermediate counts at loci B and C ---
        {
            "setup": "import numpy as np\n",
            "call": "configuration_probability_given_site_counts(10, 0.1, 8, 10, 1e-3, 500.0, 1, 1, 300.0, 5, 250.0, 4)",
            "gold_call": "_oracle_configuration_probability_given_site_counts(10, 0.1, 8, 10, 1e-3, 500.0, 1, 1, 300.0, 5, 250.0, 4)",
        },
    ]
