"""
Standing genetic variance for fitness at mutation-selection-drift balance, reduced by the recurrent fixation of deleterious mutations.

Background selection draws its strength from the standing variance in fitness that

deleterious mutations maintain along a chromosome. In an infinitely large population that

variance is fixed by the balance between mutation and selection alone, but in a finite

population selection is not perfect: from time to time a deleterious mutation drifts to

fixation, mean fitness drops by its effect, and the variance that selection can act on is

smaller than the infinite-population value. How often such fixations occur depends on the

effective size of the population, which background selection itself reduces, so the

standing variance, the interval between fixations and the asymptotic effective size have

to be found together. This step solves that joint balance for one census size.

Returns
-------
float, the standing genetic variance for relative fitness V_w, in (0, mutation_rate * selection], as a native Python float accurate to a relative error below 1e-11
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fitness_variance_under_ratchet(census: float, mutation_rate: float, selection: float,
                                   chromosome_length: float) -> float:
    '''Standing genetic variance for fitness at mutation-selection-drift balance, reduced by the recurrent fixation of deleterious mutations.

    A randomly mating Wright-Fisher population of census diploid individuals
    (2 * census haploid genomes) carries a chromosome of map length
    chromosome_length Morgans on which deleterious mutations arise at a total
    rate of mutation_rate per chromosome per generation, each with a
    multiplicative fitness effect of selection in heterozygotes. Write U for
    mutation_rate, s for selection, L for chromosome_length and N for census.
    In an infinitely large population the standing genetic variance for
    relative fitness would be U s. In the finite population deleterious
    mutations fix at an expected interval of T generations, each fixation
    lowers mean fitness by s, and the decline of mean fitness per generation,
    -s/T, equals the loss through new mutations, -U s, plus the response to
    selection, which is the variance itself, so the standing variance is
    V_w = U s - s/T. The interval between fixations is
    T = (exp(2 s n) - 1) / (2 U s n), where n is the asymptotic effective
    number of haploid genomes at a neutral position of the chromosome. That
    asymptotic number is n = 2 N exp(-V_w * Qbar), where
    Qbar = (2/L) * integral from 0 to L/2 of Q(x)^2 dx,
    Q(x) = 1 / (1 - (1 - c(x)) (1 - V_M/V_w)) is the cumulative effect of an
    association of unlimited age with a position at map distance x,
    c(x) = (1 - exp(-2 x))/2 is Haldane's recombination fraction, and
    V_M = U s^2 is the mutational input of fitness variance per generation.
    The three relations determine n, T and V_w uniquely: as n increases T
    increases, V_w increases towards U s, V_M/V_w decreases, Q(x) and Qbar
    increase, and 2 N exp(-V_w Qbar) decreases, so n = 2 N exp(-V_w Qbar) has
    exactly one solution in (0, 2 N), and V_w is positive there because
    T > 1/U for every n > 0. Solve for n to a relative precision of 1e-12 or
    better, evaluating s/T in the form 2 U s^2 n exp(-2 s n) / (1 - exp(-2 s n))
    so that large arguments do not overflow, and return V_w.

    Parameters
    ----------
    census : float
        Number of breeding diploid individuals, >= 1.
    mutation_rate : float
        Deleterious mutation rate per chromosome per generation, > 0.
    selection : float
        Multiplicative fitness effect of a deleterious mutation in
        heterozygotes, in (0, 1).
    chromosome_length : float
        Map length of the chromosome in Morgans, > 0.

    Returns
    -------
    variance : float
        The standing genetic variance for relative fitness V_w, in
        (0, mutation_rate * selection], as a native Python float accurate to a
        relative error below 1e-11.

    Raises
    ------
    ValueError
        If census is not a finite number >= 1, if mutation_rate is not a finite
        number > 0, if selection is not a finite number in (0, 1), or if
        chromosome_length is not a finite number > 0.
    '''
    return variance  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_fitness_variance_under_ratchet(census: float, mutation_rate: float, selection: float,
                                           chromosome_length: float) -> float:
    from numpy.polynomial.legendre import leggauss

    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(float(v)))

    if not _num(census) or float(census) < 1.0:
        raise ValueError("census must be a finite number >= 1")
    if not _num(mutation_rate) or float(mutation_rate) <= 0.0:
        raise ValueError("mutation_rate must be a finite number > 0")
    if not _num(selection) or not 0.0 < float(selection) < 1.0:
        raise ValueError("selection must be a finite number in (0, 1)")
    if not _num(chromosome_length) or float(chromosome_length) <= 0.0:
        raise ValueError("chromosome_length must be a finite number > 0")
    big_n, u, sel, length = float(census), float(mutation_rate), float(selection), float(chromosome_length)
    # quadrature over the map distances to half the chromosome, split near x = 0 where the integrand varies on the
    # scale of the renewal fraction
    nodes, weights = leggauss(96)
    half = 0.5 * length
    split = min(0.05, half)
    x1, w1 = 0.5 * split * (nodes + 1.0), 0.5 * split * weights
    x2, w2 = split + 0.5 * (half - split) * (nodes + 1.0), 0.5 * (half - split) * weights
    x = np.concatenate([x1, x2])
    w = np.concatenate([w1, w2])
    recomb = 0.5 * (1.0 - np.exp(-2.0 * x))                # Haldane's map function

    def _variance(n):
        """V_w = U s - s/T at the asymptotic haploid number n, with s/T written so that large 2 s n cannot overflow."""
        return u * sel - 2.0 * u * sel * sel * n * np.exp(-2.0 * sel * n) / (-np.expm1(-2.0 * sel * n))

    def _asymptotic(n):
        """2 N exp(-V_w Qbar) for the variance that n implies."""
        variance = _variance(n)
        renewal = u * sel * sel / variance
        q = 1.0 / (1.0 - (1.0 - recomb) * (1.0 - renewal))
        return 2.0 * big_n * np.exp(-variance * (2.0 / length) * ((q * q) @ w))

    # the map n -> 2 N exp(-V_w Qbar) is strictly decreasing, so g(n) = map(n) - n has one root in (0, 2 N):
    # bisection to far below the required precision
    lo, hi = 1.0e-9, 2.0 * big_n
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _asymptotic(mid) > mid:
            lo = mid
        else:
            hi = mid
    return float(_variance(0.5 * (lo + hi)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the shipped chromosome at the bottleneck census, where fixations are frequent ---
        {
            "setup": "import numpy as np\n",
            "call": "fitness_variance_under_ratchet(150, 0.5, 0.02, 1.0)",
            "gold_call": "_oracle_fitness_variance_under_ratchet(150, 0.5, 0.02, 1.0)",
        },
        # --- boundary: the shipped chromosome at the recent census, where the ratchet barely turns ---
        {
            "setup": "import numpy as np\n",
            "call": "fitness_variance_under_ratchet(600, 0.5, 0.02, 1.0)",
            "gold_call": "_oracle_fitness_variance_under_ratchet(600, 0.5, 0.02, 1.0)",
        },
        # --- edge: a tiny population under strong selection on a short chromosome, deep in the ratchet regime ---
        {
            "setup": "import numpy as np\n",
            "call": "fitness_variance_under_ratchet(12, 0.8, 0.1, 0.25)",
            "gold_call": "_oracle_fitness_variance_under_ratchet(12, 0.8, 0.1, 0.25)",
        },
        # --- edge: a large census, where the fixation interval is astronomically long and V_w returns U s ---
        {
            "setup": "import numpy as np\n",
            "call": "fitness_variance_under_ratchet(8000, 0.5, 0.02, 1.0)",
            "gold_call": "_oracle_fitness_variance_under_ratchet(8000, 0.5, 0.02, 1.0)",
        },
    ]
