"""
Survival function and density of one side of the run of homozygosity around a focal position.

Around any position at which two haplotypes descend from a common ancestral copy,

they share a segment of identity by descent whose ends are set by the recombination,

mutation and gene-conversion events that accumulated along both lineages since the

common ancestor. Genotyping does not see those events: a run of homozygosity ends at

typed markers, so its observable boundaries depend on how the markers are spaced and

how often they are heterozygous, and they do not coincide with the ends of the

identity segment. This step describes one side of the run around a focal position,

for a pair of haplotypes whose lineages coalesced a given number of generations ago,

under the construction of the observable boundary stated in the contract below.

Returns
-------
np.ndarray, shape (2, n): the probability that the run extends at least x[i] on the given side (row 0) and the density of that side's length at x[i] (row 1)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def flank_survival_and_density(x: "np.ndarray", t: int, break_rate: float, marker_spacing: float,
                               heterozygosity: float) -> "np.ndarray":
    '''Survival function and density of one side of the run of homozygosity around a focal position.

    Two haplotypes of a randomly mating population coalesced at the focal
    position t generations ago. Along each lineage, in each generation, identity
    is broken by recombination at rate 1 per Morgan and by de novo mutation and
    gene conversion at a combined rate of break_rate per Morgan, all as
    independent Poisson events on an unbounded chromosome. Typed markers lie at
    random along the chromosome with mean spacing marker_spacing, and each is
    heterozygous between two random haplotypes with probability heterozygosity,
    independently of the others. On each lineage, in each generation, only the
    first recombination breakpoint on a side counts: it is observed at the first
    heterozygous marker beyond it, a displacement that is exponentially
    distributed with rate heterozygosity/marker_spacing per Morgan and drawn
    independently for each lineage and generation, whereas mutation and
    gene-conversion breaks are observed where they occur. The run extends on a
    side as far as no observable break of any lineage in any generation has
    occurred. For each distance in x, return the probability that the run
    extends at least that far beyond the focal position on a given side, and
    the density of that side's length at that distance.

    Parameters
    ----------
    x : np.ndarray
        Shape (n,): distances from the focal position in Morgans, each >= 0.
    t : int
        Number of generations since the two lineages coalesced, >= 1.
    break_rate : float
        Combined rate of mutation and gene-conversion breaks per Morgan per
        generation along one lineage, >= 0.
    marker_spacing : float
        Mean spacing of the typed markers in Morgans, in [0, heterozygosity).
        With marker_spacing = 0 every break is observed where it occurs.
    heterozygosity : float
        Probability that a typed marker is heterozygous between two random
        haplotypes, in (0, 1].

    Returns
    -------
    out : np.ndarray
        Shape (2, n). out[0, i] is the probability that the run extends at least
        x[i] Morgans on the given side; out[1, i] is the density of that side's
        length at x[i], per Morgan. Every entry is accurate to an absolute error
        below 1e-12.

    Raises
    ------
    ValueError
        If x is not a one-dimensional array of finite numbers >= 0 with at least
        one entry, if t is not an integer >= 1, if break_rate is not a finite
        number >= 0, if heterozygosity is not a finite number in (0, 1], or if
        marker_spacing is not a finite number in [0, heterozygosity).
    '''
    return out  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_flank_survival_and_density(x: "np.ndarray", t: int, break_rate: float, marker_spacing: float,
                                       heterozygosity: float) -> "np.ndarray":
    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(float(v)))

    xx = np.asarray(x, dtype=float) if not isinstance(x, (str, bytes)) else None
    if xx is None or xx.ndim != 1 or xx.size < 1 or not np.all(np.isfinite(xx)) or np.any(xx < 0.0):
        raise ValueError("x must be a one-dimensional array of finite numbers >= 0 with at least one entry")
    if isinstance(t, bool) or not isinstance(t, (int, np.integer)) or int(t) < 1:
        raise ValueError("t must be an integer >= 1")
    if not _num(break_rate) or float(break_rate) < 0.0:
        raise ValueError("break_rate must be a finite number >= 0")
    if not _num(heterozygosity) or not 0.0 < float(heterozygosity) <= 1.0:
        raise ValueError("heterozygosity must be a finite number in (0, 1]")
    if not _num(marker_spacing) or not 0.0 <= float(marker_spacing) < float(heterozygosity):
        raise ValueError("marker_spacing must be a finite number in [0, heterozygosity)")
    m, d, H = float(break_rate), float(marker_spacing), float(heterozygosity)
    n_lineage_generations = 2 * int(t)          # two lineages, t generations each
    decay = np.exp(-m * xx)                     # mutation and conversion breaks, observed where they occur
    if d == 0.0:
        p = np.exp(-xx) * decay
        dp = -(1.0 + m) * np.exp(-xx) * decay
    else:
        # a recombination breakpoint is observed at the first heterozygous marker beyond it: the
        # observable distance is the sum of an exponential(1) and an exponential(H/d) variable
        r = H / d
        c1 = H / (H - d)
        c2 = d / (H - d)
        rec = c1 * np.exp(-xx) - c2 * np.exp(-r * xx)
        drec = -c1 * np.exp(-xx) + c2 * r * np.exp(-r * xx)
        p = rec * decay
        dp = (drec - m * rec) * decay
    survival = p ** n_lineage_generations
    density = -n_lineage_generations * p ** (n_lineage_generations - 1) * dp
    return np.array([survival, density])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the shipped marker panel and break rate, 30 generations since coalescence ---
        {
            "setup": "import numpy as np\nx = np.array([0.0, 0.002, 0.005, 0.01, 0.02, 0.04])\n",
            "call": "flank_survival_and_density(x, 30, 1.5, 0.001, 0.30)",
            "gold_call": "_oracle_flank_survival_and_density(x, 30, 1.5, 0.001, 0.30)",
        },
        # --- boundary: markers so dense that every break is observed where it occurs ---
        {
            "setup": "import numpy as np\nx = np.array([0.001, 0.01, 0.05])\n",
            "call": "flank_survival_and_density(x, 12, 0.8, 0.0, 0.5)",
            "gold_call": "_oracle_flank_survival_and_density(x, 12, 0.8, 0.0, 0.5)",
        },
        # --- edge: a sparse panel one generation after coalescence, out to half a Morgan ---
        {
            "setup": "import numpy as np\nx = np.array([0.0, 0.01, 0.05, 0.1, 0.25, 0.5])\n",
            "call": "flank_survival_and_density(x, 1, 2.5, 0.02, 0.25)",
            "gold_call": "_oracle_flank_survival_and_density(x, 1, 2.5, 0.02, 0.25)",
        },
        # --- edge: an old coalescence, where only very short sides survive ---
        {
            "setup": "import numpy as np\nx = np.array([1e-05, 0.0001, 0.0005, 0.002])\n",
            "call": "flank_survival_and_density(x, 800, 1.5, 0.001, 0.30)",
            "gold_call": "_oracle_flank_survival_and_density(x, 800, 1.5, 0.001, 0.30)",
        },
    ]
