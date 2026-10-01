"""
Factor by which background selection reduces the effective size of a neutral position after a given number of generations of association.

Selection at linked sites accelerates the drift of a neutral position: a neutral copy

that starts on a background of higher or lower fitness carries that association into

its descendants until recombination separates it from the selected sites and until the

fitness differences themselves are renewed by mutation. The strength of the extra drift

therefore grows with the age of the association, and the effective size that a neutral

position experiences is a decreasing series in the number of generations since the

association began, not a single constant. This step evaluates that series for a

chromosome under background selection, with the standing fitness variance of the

previous step for the census in question.

Returns
-------
np.ndarray, shape (k,): factor[j] is F(generations[j]), in (0, 1]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def selection_reduction_factor(generations: "np.ndarray", mutation_rate: float, selection: float,
                               chromosome_length: float, census: float) -> "np.ndarray":
    '''Factor by which background selection reduces the effective size of a neutral position after a given number of generations of association.

    Deleterious mutations arise along a chromosome of map length
    chromosome_length Morgans at a total rate of mutation_rate per chromosome
    per generation, each with a multiplicative fitness effect of selection in
    heterozygotes, in a population of census breeding diploid individuals at
    mutation-selection-drift balance, where the standing genetic variance for
    fitness V_w is the value of the previous step for this census (the
    infinite-population value mutation_rate * selection reduced by the
    recurrent fixation of deleterious mutations) and the mutational input of
    variance per generation is V_M = mutation_rate * selection^2. A neutral copy at map distance x Morgans from a position under
    selection loses its association with the fitness deviation of its
    background in each generation by the recombination fraction c(x) between the
    two positions, given by Haldane's map function c(x) = (1 - exp(-2 x))/2, and
    by the fraction V_M/V_w of the fitness variance that mutation renews, so its
    cumulative expected contribution over g generations of association is
    Q_x(g) = sum from i = 0 to g - 1 of [(1 - c(x)) (1 - V_M/V_w)]^i. Under
    multiplicative fitness the effective size after g generations of
    association is the census size times
    F(g) = exp(-(2/chromosome_length) * integral from 0 to chromosome_length/2
    of V_w * Q_x(g)^2 dx), the average over the map distances from the focal
    position to the positions of a chromosome of that length. Return F(g) for
    each entry of generations.

    Parameters
    ----------
    generations : np.ndarray
        Shape (k,): numbers of generations of association, integers >= 1.
    mutation_rate : float
        Deleterious mutation rate per chromosome per generation, > 0.
    selection : float
        Multiplicative fitness effect of a deleterious mutation in
        heterozygotes, in (0, 1).
    chromosome_length : float
        Map length of the chromosome in Morgans, > 0.
    census : float
        Number of breeding diploid individuals whose standing variance
        applies, >= 1.

    Returns
    -------
    factor : np.ndarray
        Shape (k,): factor[j] is F(generations[j]), in (0, 1]. Every entry is
        accurate to an absolute error below 1e-12.

    Raises
    ------
    ValueError
        If generations is not a one-dimensional array of integers >= 1 with at
        least one entry, if mutation_rate is not a finite number > 0, if
        selection is not a finite number in (0, 1), if chromosome_length is
        not a finite number > 0, or if census is not a finite number >= 1.
    '''
    return factor  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_selection_reduction_factor(generations: "np.ndarray", mutation_rate: float, selection: float,
                                       chromosome_length: float, census: float) -> "np.ndarray":
    from numpy.polynomial.legendre import leggauss

    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(float(v)))

    gg = np.asarray(generations) if not isinstance(generations, (str, bytes)) else None
    if (gg is None or gg.ndim != 1 or gg.size < 1 or gg.dtype.kind == "b"
            or not np.all(np.isfinite(gg.astype(float))) or np.any(gg.astype(float) != np.floor(gg.astype(float)))
            or np.any(gg.astype(float) < 1)):
        raise ValueError("generations must be a one-dimensional array of integers >= 1 with at least one entry")
    if not _num(mutation_rate) or float(mutation_rate) <= 0.0:
        raise ValueError("mutation_rate must be a finite number > 0")
    if not _num(selection) or not 0.0 < float(selection) < 1.0:
        raise ValueError("selection must be a finite number in (0, 1)")
    if not _num(chromosome_length) or float(chromosome_length) <= 0.0:
        raise ValueError("chromosome_length must be a finite number > 0")
    if not _num(census) or float(census) < 1.0:
        raise ValueError("census must be a finite number >= 1")
    u, sel, length = float(mutation_rate), float(selection), float(chromosome_length)
    variance = _oracle_fitness_variance_under_ratchet(census, u, sel, length)   # V_w for this census
    renewal = u * sel * sel / variance                     # V_M / V_w
    # the integrand varies on the scale of the selection coefficient near x = 0, so the range is split there
    nodes, weights = leggauss(96)
    half = 0.5 * length
    split = min(0.05, half)
    x1, w1 = 0.5 * split * (nodes + 1.0), 0.5 * split * weights
    x2, w2 = split + 0.5 * (half - split) * (nodes + 1.0), 0.5 * (half - split) * weights
    x = np.concatenate([x1, x2])
    w = np.concatenate([w1, w2])
    recomb = 0.5 * (1.0 - np.exp(-2.0 * x))                # Haldane's map function
    ratio = (1.0 - recomb) * (1.0 - renewal)               # the per-generation survival of the association, < 1
    g = gg.astype(float)[:, None]
    q = (1.0 - ratio ** g) / (1.0 - ratio)                 # sum of the geometric series with g terms
    exponent = (2.0 / length) * variance * ((q * q) @ w)
    return np.exp(-exponent)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the shipped chromosome at the bottleneck census, across recent to old associations ---
        {
            "setup": "import numpy as np\ng = np.array([1, 5, 10, 26, 61, 200, 3000])\n",
            "call": "selection_reduction_factor(g, 0.5, 0.02, 1.0, 150)",
            "gold_call": "_oracle_selection_reduction_factor(g, 0.5, 0.02, 1.0, 150)",
        },
        # --- boundary: one generation of association at the recent census, where only the immediate variance acts ---
        {
            "setup": "import numpy as np\ng = np.array([1, 30])\n",
            "call": "selection_reduction_factor(g, 1.2, 0.05, 2.0, 600)",
            "gold_call": "_oracle_selection_reduction_factor(g, 1.2, 0.05, 2.0, 600)",
        },
        # --- edge: a short chromosome under strong selection in a small population, near the asymptote ---
        {
            "setup": "import numpy as np\ng = np.array([40, 400, 4000])\n",
            "call": "selection_reduction_factor(g, 0.05, 0.1, 0.08, 500)",
            "gold_call": "_oracle_selection_reduction_factor(g, 0.05, 0.1, 0.08, 500)",
        },
    ]
