"""
Number of sites, Watterson's estimator, pairwise diversity, Fay and Wu's estimator and Tajima's D of an unfolded spectrum, as region totals.

Watterson's estimator, Tajima's pairwise diversity and Fay and Wu's estimator are

linear functions of the site frequency spectrum, each weighting the derived-count

classes in its own way, and Tajima's D contrasts the first two on the scale of the

variance they would have under the standard neutral model. Applied to an observed

spectrum they give the classical summaries of a region; applied to an expected

spectrum, with the expectation substituted for every observed quantity, they give

the value each summary is expected to take under the model behind that spectrum.

This step evaluates them for either kind of input.

Returns
-------
np.ndarray, shape (5,): [S, theta_W, pi, theta_H, D], the first four as region totals and D dimensionless
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spectrum_statistics(counts: "np.ndarray", m: int) -> "np.ndarray":
    '''Number of sites, Watterson's estimator, pairwise diversity, Fay and Wu's estimator and Tajima's D of an unfolded spectrum, as region totals.

    counts gives the number of sites of derived count k in a sample of m
    sequences; it may hold expected (non-integer) numbers. Return the total
    number of sites S; Watterson's estimator of theta for the region, S over
    the harmonic number of m - 1 terms; the average number of pairwise
    differences pi; Fay and Wu's theta_H, the estimator that weights each
    class by the square of its derived count; and Tajima's D, the difference
    between pi and Watterson's estimator divided by the square root of its
    variance estimate as defined by Tajima (1989), with S(S - 1) in the
    second-order term, every quantity evaluated from counts as given. The
    three estimators of theta are totals over the region, not per site.

    Parameters
    ----------
    counts : np.ndarray
        Shape (m - 1,): the number of sites of derived count k at index k - 1,
        finite entries >= 0.
    m : int
        Number of sequences in the sample, >= 4.

    Returns
    -------
    stats : np.ndarray
        Shape (5,): [S, theta_W, pi, theta_H, D], with the first four as region
        totals and D dimensionless. Accurate to a relative error below 1e-12.

    Raises
    ------
    ValueError
        If m is not an integer >= 4, if counts is not a one-dimensional array
        of finite numbers >= 0 of length m - 1, or if the total number of sites
        is below 2, for which D is not defined.
    '''
    return stats  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_spectrum_statistics(counts: "np.ndarray", m: int) -> "np.ndarray":
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or int(m) < 4:
        raise ValueError("m must be an integer >= 4")
    m = int(m)
    x = np.asarray(counts, dtype=float) if not isinstance(counts, (str, bytes)) else None
    if x is None or x.ndim != 1 or x.shape[0] != m - 1 or not np.all(np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError("counts must be a one-dimensional array of finite numbers >= 0 of length m - 1")
    k = np.arange(1, m, dtype=float)
    s_total = float(np.sum(x))
    if s_total < 2.0:
        raise ValueError("Tajima's D needs at least 2 sites")
    a1 = float(np.sum(1.0 / k))
    a2 = float(np.sum(1.0 / k ** 2))
    theta_w = s_total / a1
    pi = float(np.sum(2.0 * k * (m - k) / (m * (m - 1.0)) * x))
    theta_h = float(np.sum(2.0 * k * k / (m * (m - 1.0)) * x))
    b1 = (m + 1.0) / (3.0 * (m - 1.0))
    b2 = 2.0 * (m * m + m + 3.0) / (9.0 * m * (m - 1.0))
    c1 = b1 - 1.0 / a1
    c2 = b2 - (m + 2.0) / (a1 * m) + a2 / a1 ** 2
    e1, e2 = c1 / a1, c2 / (a1 ** 2 + a2)
    d = (pi - theta_w) / np.sqrt(e1 * s_total + e2 * s_total * (s_total - 1.0))
    return np.array([s_total, theta_w, pi, theta_h, d])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the shipped flanking spectrum of 34 haplotypes ---
        {
            "setup": "import numpy as np\ncounts = np.array([292, 159, 91, 71, 57, 62, 41, 35, 37, 31, 25, 20, 15, 24, 15, 19, 26, 14, 17, 13, 7, 11, 12, 16, 12, 6, 11, 16, 12, 7, 7, 14, 13])\n",
            "call": "spectrum_statistics(counts, 34)",
            "gold_call": "_oracle_spectrum_statistics(counts, 34)",
        },
        # --- edge: an expected spectrum with non-integer entries (the shipped segment's null times theta L) ---
        {
            "setup": "import numpy as np\ncounts = 44.316 * _oracle_indel_null_spectrum(34, 13, True)\n",
            "call": "spectrum_statistics(counts, 13)",
            "gold_call": "_oracle_spectrum_statistics(counts, 13)",
        },
        # --- boundary: the smallest sample size for which D is defined, with an excess of intermediate counts ---
        {
            "setup": "import numpy as np\ncounts = np.array([1.0, 5.0, 2.0])\n",
            "call": "spectrum_statistics(counts, 4)",
            "gold_call": "_oracle_spectrum_statistics(counts, 4)",
        },
        # --- edge: a spectrum dominated by high-frequency derived alleles in a sample of 20 ---
        {
            "setup": "import numpy as np\ncounts = np.array([3, 1, 0, 2, 0, 1, 0, 0, 1, 0, 2, 1, 0, 3, 4, 6, 9, 12, 20])\n",
            "call": "spectrum_statistics(counts, 20)",
            "gold_call": "_oracle_spectrum_statistics(counts, 20)",
        },
    ]
