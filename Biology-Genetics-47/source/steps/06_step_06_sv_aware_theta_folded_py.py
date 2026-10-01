"""
Variant-aware estimate of the population-scaled mutation rate per site from the folded spectrum of a presence/absence segment.

A linear estimator of the mutation parameter divides each class of the spectrum by

the class's expected value under the standard neutral model and combines the classes

with weights that sum to one; Watterson's estimator and the pairwise diversity are the

two classical weightings. Inside a segment linked to a structural variant the standard

expectations do not hold, and the source adjusts such an estimator in one of two ways:

by replacing each class's baseline expectation with its expectation conditional on the

variant, or by rescaling the whole standard estimator by the ratio between its

conditional and its baseline expectation. Both adjustments are unbiased under the

conditional model. With unpolarised sites the classes merge in pairs, and the

adjustment is carried out on the folded spectrum. This step evaluates the adjusted

estimators from folded data.

Returns
-------
float, the variant-aware estimate of theta per site, as a native Python float, accurate to a relative error below 1e-12
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sv_aware_theta_folded(folded_counts: "np.ndarray", n: int, carriers: int, absence_derived: bool,
                          segment_length: float, weighting: str, correction: int) -> float:
    '''Variant-aware estimate of the population-scaled mutation rate per site from the folded spectrum of a presence/absence segment.

    The segment of segment_length base pairs is present on `carriers` of the
    n haplotypes, with the polarity given by absence_derived, and its sites
    are unpolarised, so folded_counts holds their spectrum among the carriers
    in minor-allele classes 1 to floor(carriers / 2) as in the folding step.
    The estimator starts from a linear estimator of theta over the derived
    counts 1 to carriers - 1 of the carriers' sample: the sum over counts k of
    a weight times the number of sites of count k divided by the baseline
    expectation of that number per unit theta, which is segment_length / k,
    the weights summing to one; weighting "watterson" uses the weights of
    Watterson's estimator and weighting "pairwise" those of the average
    pairwise difference. With correction 1 the baseline expectation of every
    class is replaced by the class's expectation under the indel spectrum of
    the earlier step for this polarity (the expected number of sites per unit
    theta L times segment_length), and the estimate is formed on the folded
    classes: the weight of a minor-allele class is the sum of the weights of
    the two derived counts it merges (the weight of the single derived count
    for the class of equal counts), and its expectation is the indel spectrum
    folded the same way. With correction 2 the standard estimator, evaluated
    on the folded counts, is divided by the ratio of its expectation under the
    indel spectrum to its expectation under the baseline. Return the estimate
    per site.

    Parameters
    ----------
    folded_counts : np.ndarray
        Shape (carriers // 2,): the observed number of sites of minor count j
        at index j - 1, finite entries >= 0.
    n : int
        Number of haplotypes in the sample, >= 3.
    carriers : int
        Number of haplotypes carrying the segment, from 2 to n - 1.
    absence_derived : bool
        True if the absent state is derived (a deletion), False if the present
        state is derived (an insertion).
    segment_length : float
        Length of the segment in base pairs, > 0.
    weighting : str
        "watterson" or "pairwise".
    correction : int
        1 for the class-wise replacement of the baseline expectations, 2 for
        the rescaling of the standard estimator.

    Returns
    -------
    theta_hat : float
        The variant-aware estimate of theta per site, as a native Python
        float, accurate to a relative error below 1e-12.

    Raises
    ------
    ValueError
        If folded_counts is not a one-dimensional array of finite numbers >= 0
        of length carriers // 2, if segment_length is not a finite number > 0,
        if weighting is not "watterson" or "pairwise", if correction is not 1
        or 2, or on any condition raised by the earlier steps for n, carriers
        and absence_derived.
    '''
    return theta_hat  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sv_aware_theta_folded(folded_counts: "np.ndarray", n: int, carriers: int, absence_derived: bool,
                                  segment_length: float, weighting: str, correction: int) -> float:
    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(float(v)))

    if isinstance(carriers, bool) or not isinstance(carriers, (int, np.integer)):
        raise ValueError("carriers must be an integer from 2 to n - 1")
    c = int(carriers)
    eta = np.asarray(folded_counts, dtype=float) if not isinstance(folded_counts, (str, bytes)) else None
    if eta is None or eta.ndim != 1 or eta.shape[0] != c // 2 or not np.all(np.isfinite(eta)) or np.any(eta < 0.0):
        raise ValueError("folded_counts must be a one-dimensional array of finite numbers >= 0 of length carriers // 2")
    if not _num(segment_length) or float(segment_length) <= 0.0:
        raise ValueError("segment_length must be a finite number > 0")
    if weighting not in ("watterson", "pairwise"):
        raise ValueError('weighting must be "watterson" or "pairwise"')
    if isinstance(correction, bool) or correction not in (1, 2):
        raise ValueError("correction must be 1 or 2")
    length = float(segment_length)
    null = _oracle_indel_null_spectrum(n, c, absence_derived)       # per unit theta L, derived counts 1 .. c - 1
    k = np.arange(1, c, dtype=float)
    if weighting == "watterson":
        w = 1.0 / (k * float(np.sum(1.0 / k)))
    else:
        w = 2.0 * (c - k) / (c * (c - 1.0))
    if correction == 1:
        w_folded = _oracle_fold_spectrum(w, c)
        null_folded = _oracle_fold_spectrum(null, c)
        return float(np.sum(w_folded * eta / (length * null_folded)))
    coef = w * k                                                     # symmetric under k <-> c - k for both weightings
    standard = float(np.sum(coef[:c // 2] * eta)) / length           # the standard estimator from the folded classes
    return float(standard / np.sum(coef * null))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the shipped segment, absence derived, pairwise weights, class-wise correction ---
        {
            "setup": "import numpy as np\nfolded = np.array([29, 14, 9, 11, 3, 6])\n",
            "call": "sv_aware_theta_folded(folded, 34, 13, True, 9000.0, 'pairwise', 1)",
            "gold_call": "_oracle_sv_aware_theta_folded(folded, 34, 13, True, 9000.0, 'pairwise', 1)",
        },
        # --- normal: the same data under the other polarity, Watterson weights ---
        {
            "setup": "import numpy as np\nfolded = np.array([29, 14, 9, 11, 3, 6])\n",
            "call": "sv_aware_theta_folded(folded, 34, 13, False, 9000.0, 'watterson', 1)",
            "gold_call": "_oracle_sv_aware_theta_folded(folded, 34, 13, False, 9000.0, 'watterson', 1)",
        },
        # --- normal: the rescaling correction on the shipped data ---
        {
            "setup": "import numpy as np\nfolded = np.array([29, 14, 9, 11, 3, 6])\n",
            "call": "sv_aware_theta_folded(folded, 34, 13, True, 9000.0, 'pairwise', 2)",
            "gold_call": "_oracle_sv_aware_theta_folded(folded, 34, 13, True, 9000.0, 'pairwise', 2)",
        },
        # --- boundary: three carriers, a single folded class ---
        {
            "setup": "import numpy as np\nfolded = np.array([7.0])\n",
            "call": "sv_aware_theta_folded(folded, 10, 3, True, 3000.0, 'watterson', 1)",
            "gold_call": "_oracle_sv_aware_theta_folded(folded, 10, 3, True, 3000.0, 'watterson', 1)",
        },
        # --- edge: an even number of carriers, whose middle class carries a single weight ---
        {
            "setup": "import numpy as np\nfolded = np.array([10, 6, 4, 3])\n",
            "call": "sv_aware_theta_folded(folded, 20, 8, False, 5000.0, 'pairwise', 1)",
            "gold_call": "_oracle_sv_aware_theta_folded(folded, 20, 8, False, 5000.0, 'pairwise', 1)",
        },
    ]
