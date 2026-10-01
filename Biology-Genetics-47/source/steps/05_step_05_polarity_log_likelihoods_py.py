"""
Poisson log-likelihoods of the observed folded spectrum of a presence/absence segment under each polarity of the variant.

Whether the absent or the present state of a presence/absence variant is the derived

one is often unknown: the region may not align to any outgroup, and the variant may

be too old for its frequency alone to tell. The two polarities imply different

expected spectra for the sites inside the segment, and different expected numbers of

sites, because the two backgrounds have different genealogical histories. Given the

mutation parameter from neutral flanking sequence, each polarity therefore predicts

the expected count of every class of the folded spectrum, and the observed spectrum

can be weighed against the two predictions. This step computes that comparison.

Returns
-------
np.ndarray, shape (2,): the Poisson log-likelihood of the folded spectrum with the absent state derived (entry 0) and with the present state derived (entry 1)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def polarity_log_likelihoods(folded_counts: "np.ndarray", n: int, carriers: int, theta_per_site: float,
                             segment_length: float) -> "np.ndarray":
    '''Poisson log-likelihoods of the observed folded spectrum of a presence/absence segment under each polarity of the variant.

    The segment of segment_length base pairs is present on `carriers` of the n
    haplotypes; the population-scaled mutation rate per site is
    theta_per_site; the sites inside the segment are unpolarised, so their
    spectrum among the carriers is folded into minor-allele classes 1 to
    floor(carriers / 2) as in the folding step. Under each polarity the
    expected number of sites in class j is theta_per_site times segment_length
    times the expected spectrum of the segment for that polarity (the indel
    spectrum of the earlier step) folded the same way, and the classes are
    treated as independent Poisson counts. Return the log-likelihood of
    folded_counts under each polarity as the sum over classes of the observed
    count times the logarithm of the expected count, minus the expected count;
    the term in the factorials of the observed counts is the same for both
    polarities and is omitted.

    Parameters
    ----------
    folded_counts : np.ndarray
        Shape (carriers // 2,): the observed number of sites of minor count j
        at index j - 1, finite entries >= 0.
    n : int
        Number of haplotypes in the sample, >= 3.
    carriers : int
        Number of haplotypes carrying the segment, from 2 to n - 1.
    theta_per_site : float
        Population-scaled mutation rate per site, > 0.
    segment_length : float
        Length of the segment in base pairs, > 0.

    Returns
    -------
    log_likelihoods : np.ndarray
        Shape (2,): entry 0 for the absent state derived (a deletion), entry 1
        for the present state derived (an insertion). Accurate to an absolute
        error below 1e-9.

    Raises
    ------
    ValueError
        If folded_counts is not a one-dimensional array of finite numbers >= 0
        of length carriers // 2, if theta_per_site or segment_length is not a
        finite number > 0, or on any condition raised by the earlier steps for
        n and carriers.
    '''
    return log_likelihoods  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_polarity_log_likelihoods(folded_counts: "np.ndarray", n: int, carriers: int, theta_per_site: float,
                                     segment_length: float) -> "np.ndarray":
    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(float(v)))

    if isinstance(carriers, bool) or not isinstance(carriers, (int, np.integer)):
        raise ValueError("carriers must be an integer from 2 to n - 1")
    c = int(carriers)
    eta = np.asarray(folded_counts, dtype=float) if not isinstance(folded_counts, (str, bytes)) else None
    if eta is None or eta.ndim != 1 or eta.shape[0] != c // 2 or not np.all(np.isfinite(eta)) or np.any(eta < 0.0):
        raise ValueError("folded_counts must be a one-dimensional array of finite numbers >= 0 of length carriers // 2")
    if not _num(theta_per_site) or float(theta_per_site) <= 0.0 or not _num(segment_length) or float(segment_length) <= 0.0:
        raise ValueError("theta_per_site and segment_length must be finite numbers > 0")
    scale = float(theta_per_site) * float(segment_length)
    out = np.zeros(2)
    for idx, absence_derived in enumerate((True, False)):
        lam = scale * _oracle_fold_spectrum(_oracle_indel_null_spectrum(n, c, absence_derived), c)
        out[idx] = float(np.sum(eta * np.log(lam) - lam))
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the shipped segment's folded spectrum at the flanking mutation parameter ---
        {
            "setup": "import numpy as np\nfolded = np.array([29, 14, 9, 11, 3, 6])\ntheta = 1208.0 / (np.sum(1.0 / np.arange(1, 34)) * 60000.0)\n",
            "call": "polarity_log_likelihoods(folded, 34, 13, theta, 9000.0)",
            "gold_call": "_oracle_polarity_log_likelihoods(folded, 34, 13, theta, 9000.0)",
        },
        # --- edge: a sparse spectrum of the same segment that favours the present state as derived ---
        {
            "setup": "import numpy as np\nfolded = np.array([12, 7, 5, 4, 6, 1])\ntheta = 1208.0 / (np.sum(1.0 / np.arange(1, 34)) * 60000.0)\n",
            "call": "polarity_log_likelihoods(folded, 34, 13, theta, 9000.0)",
            "gold_call": "_oracle_polarity_log_likelihoods(folded, 34, 13, theta, 9000.0)",
        },
        # --- boundary: three carriers, a single folded class ---
        {
            "setup": "import numpy as np\nfolded = np.array([5.0])\n",
            "call": "polarity_log_likelihoods(folded, 10, 3, 0.01, 2000.0)",
            "gold_call": "_oracle_polarity_log_likelihoods(folded, 10, 3, 0.01, 2000.0)",
        },
        # --- edge: an even number of carriers, whose last folded class is the middle class ---
        {
            "setup": "import numpy as np\nfolded = np.array([18, 9, 4, 6])\n",
            "call": "polarity_log_likelihoods(folded, 20, 8, 0.006, 12000.0)",
            "gold_call": "_oracle_polarity_log_likelihoods(folded, 20, 8, 0.006, 12000.0)",
        },
    ]
