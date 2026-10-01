"""
Variant-aware diversity of a presence/absence segment relative to its neutral flanks, at the polarity decided from the segment's spectrum.

This step returns the deliverable: the diversity of a presence/absence segment relative

to its neutral flanks, estimated in a way that accounts for the variant. A study

reports the folded spectrum of the sites inside the segment among the haplotypes that

carry it, and the polarised spectrum of a flanking neutral region in the same

haplotypes, but no outgroup aligns to the segment, so the polarity of the variant is

unknown. The step estimates the mutation parameter from the flanks, decides the

polarity from the segment's spectrum, applies the class-wise variant-aware correction

with pairwise weights to the folded segment spectrum under the decided polarity, and

reports the ratio of that estimate to the flanking value.

Returns
-------
float, the variant-aware estimate of theta per site inside the segment divided by the flanking estimate, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sv_aware_relative_diversity(n: int, carriers: int, folded_counts: "np.ndarray", segment_length: float,
                                flank_counts: "np.ndarray", flank_length: float) -> float:
    '''Variant-aware diversity of a presence/absence segment relative to its neutral flanks, at the polarity decided from the segment's spectrum.

    n haplotypes are sampled from a neutrally evolving population that
    follows the standard neutral coalescent, with no recombination inside the
    region, mutation under the infinite-sites model, and a segment of
    segment_length base pairs present on `carriers` of the haplotypes and
    absent from the others, the presence/absence variant being neutral,
    biallelic, of single origin and in complete linkage with every site
    inside the segment. Sites inside the segment are unpolarised, so
    folded_counts holds their spectrum among the carriers in minor-allele
    classes 1 to floor(carriers / 2); flank_counts holds the polarised
    spectrum of a neutral flanking region of flank_length base pairs in the
    same n haplotypes, by derived count 1 to n - 1. Take the mutation
    parameter per site to be Watterson's estimate from the flanking spectrum
    (the statistics step, divided by flank_length). Decide the polarity of the
    variant from the Poisson log-likelihoods of the segment's folded spectrum
    under the two polarities at that mutation parameter (the polarity step):
    the absent state is derived when its log-likelihood is at least as large
    as the other's. Then compute the variant-aware estimate of the mutation
    parameter per site from the folded segment spectrum under the decided
    polarity, with pairwise weights and correction 1 of the estimation step,
    and return it divided by the flanking estimate.

    Parameters
    ----------
    n : int
        Number of haplotypes in the sample, >= 4.
    carriers : int
        Number of haplotypes carrying the segment, from 2 to n - 1.
    folded_counts : np.ndarray
        Shape (carriers // 2,): the observed number of sites inside the
        segment of minor count j among the carriers, at index j - 1, finite
        entries >= 0.
    segment_length : float
        Length of the segment in base pairs, > 0.
    flank_counts : np.ndarray
        Shape (n - 1,): the observed number of sites of derived count k in the
        flanking region, at index k - 1, finite entries >= 0, with at least 2
        sites in total.
    flank_length : float
        Length of the flanking region in base pairs, > 0.

    Returns
    -------
    relative_diversity : float
        The variant-aware estimate of theta per site inside the segment
        divided by the flanking estimate, as a native Python float.

    Raises
    ------
    ValueError
        If flank_length is not a finite number > 0, or on any condition raised
        by the earlier steps for the given or derived inputs.
    '''
    return relative_diversity  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sv_aware_relative_diversity(n: int, carriers: int, folded_counts: "np.ndarray", segment_length: float,
                                        flank_counts: "np.ndarray", flank_length: float) -> float:
    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(float(v)))

    if not _num(flank_length) or float(flank_length) <= 0.0:
        raise ValueError("flank_length must be a finite number > 0")
    theta_flank = float(_oracle_spectrum_statistics(flank_counts, n)[1]) / float(flank_length)
    log_likelihoods = _oracle_polarity_log_likelihoods(folded_counts, n, carriers, theta_flank, segment_length)
    absence_derived = bool(log_likelihoods[0] >= log_likelihoods[1])
    theta_segment = _oracle_sv_aware_theta_folded(folded_counts, n, carriers, absence_derived, segment_length, "pairwise", 1)
    return float(theta_segment / theta_flank)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the shipped configuration; this is the reported answer ---
        {
            "setup": "import numpy as np\nfolded = np.array([29, 14, 9, 11, 3, 6])\nflank = np.array([292, 159, 91, 71, 57, 62, 41, 35, 37, 31, 25, 20, 15, 24, 15, 19, 26, 14, 17, 13, 7, 11, 12, 16, 12, 6, 11, 16, 12, 7, 7, 14, 13])\n",
            "call": "sv_aware_relative_diversity(34, 13, folded, 9000.0, flank, 60000.0)",
            "gold_call": "_oracle_sv_aware_relative_diversity(34, 13, folded, 9000.0, flank, 60000.0)",
        },
        # --- edge: a sparse segment spectrum for which the present state is decided to be derived ---
        {
            "setup": "import numpy as np\nfolded = np.array([12, 7, 5, 4, 6, 1])\nflank = np.array([292, 159, 91, 71, 57, 62, 41, 35, 37, 31, 25, 20, 15, 24, 15, 19, 26, 14, 17, 13, 7, 11, 12, 16, 12, 6, 11, 16, 12, 7, 7, 14, 13])\n",
            "call": "sv_aware_relative_diversity(34, 13, folded, 9000.0, flank, 60000.0)",
            "gold_call": "_oracle_sv_aware_relative_diversity(34, 13, folded, 9000.0, flank, 60000.0)",
        },
        # --- boundary: a small sample with three carriers, one folded class, and a short flank ---
        {
            "setup": "import numpy as np\nfolded = np.array([7.0])\nflank = np.array([40, 18, 12, 9, 6, 7, 4, 5, 6])\n",
            "call": "sv_aware_relative_diversity(10, 3, folded, 3000.0, flank, 15000.0)",
            "gold_call": "_oracle_sv_aware_relative_diversity(10, 3, folded, 3000.0, flank, 15000.0)",
        },
        # --- edge: an even number of carriers in a sample of 20, with a middle folded class ---
        {
            "setup": "import numpy as np\nfolded = np.array([18, 9, 4, 6])\nflank = np.array([120, 63, 40, 31, 22, 20, 16, 15, 12, 11, 9, 8, 10, 7, 6, 8, 5, 7, 6])\n",
            "call": "sv_aware_relative_diversity(20, 8, folded, 12000.0, flank, 50000.0)",
            "gold_call": "_oracle_sv_aware_relative_diversity(20, 8, folded, 12000.0, flank, 50000.0)",
        },
    ]
