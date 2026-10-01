"""
Expected spectrum of neutral sites completely linked to a focal neutral variant of known sample count, split by the relation of each site to the variant.

The site frequency spectrum of a sample counts the polymorphic sites at each

derived-allele count, and its expectation under the standard neutral model, inversely

proportional to the count, is the baseline against which the classical diversity

estimators and neutrality tests are read. A structural variant that segregates in the

sample and suppresses recombination across its region conditions the genealogy of

every linked site: the haplotypes fall into carriers and non-carriers of the variant,

and the expected spectrum of linked neutral sites, given the variant's sample count,

departs from the baseline even under strict neutrality. Under the standard neutral

coalescent that conditional spectrum has an exact finite-sample form, which separates

into components by the relation between a site's carrier set and the variant's

carrier set. This step computes those components, the machinery of every later step.

Returns
-------
np.ndarray, shape (5, n - 1): the expected number of sites of derived count k (column k - 1) per unit theta L in each relation to the focal variant (rows: strictly nested, co-occurring, enclosing, complementary, strictly disjoint)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def linked_site_spectrum_components(n: int, i: int) -> "np.ndarray":
    '''Expected spectrum of neutral sites completely linked to a focal neutral variant of known sample count, split by the relation of each site to the variant.

    A sample of n haplotypes is drawn from a population at mutation-drift
    equilibrium that follows the standard neutral coalescent, with no
    recombination anywhere in the region and mutation under the infinite-sites
    model. One focal neutral mutation is carried by exactly i of the n
    haplotypes. Every other polymorphic site in the region is classified by
    the set of haplotypes carrying its derived allele, relative to the focal
    carrier set: strictly nested (a proper subset of the focal set),
    co-occurring (the same set), enclosing (a proper superset), complementary
    (exactly the haplotypes outside the focal set), or strictly disjoint (no
    haplotype in common with the focal set, and not complementary). Return, for
    each relation and each derived count k, the expected number of such sites
    averaged over genealogies, in units of theta L (the population-scaled
    mutation rate per site times the region length), scaled so that without
    any conditioning the expected number of sites of derived count k is
    theta L / k. The expectations are exact for the finite sample, not a
    large-sample or diffusion limit.

    Parameters
    ----------
    n : int
        Number of haplotypes in the sample, >= 2.
    i : int
        Number of haplotypes carrying the focal variant, from 1 to n - 1.

    Returns
    -------
    components : np.ndarray
        Shape (5, n - 1). Row 0 strictly nested, row 1 co-occurring, row 2
        enclosing, row 3 complementary, row 4 strictly disjoint; column k - 1
        holds the expected number of sites of derived count k (k from 1 to
        n - 1) per unit theta L. Entries outside a relation's admissible counts
        are 0. Every entry is accurate to a relative error below 1e-12.

    Raises
    ------
    ValueError
        If n is not an integer >= 2, or if i is not an integer from 1 to n - 1.
    '''
    return components  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _harmonic(m: int) -> float:
    """a_m = sum over j = 1 .. m - 1 of 1/j, with a_1 = 0."""
    return float(np.sum(1.0 / np.arange(1, m))) if m > 1 else 0.0


def _oracle_linked_site_spectrum_components(n: int, i: int) -> "np.ndarray":
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 2:
        raise ValueError("n must be an integer >= 2")
    if isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 1 <= int(i) <= int(n) - 1:
        raise ValueError("i must be an integer from 1 to n - 1")
    n, i = int(n), int(i)
    a = np.array([_harmonic(m) for m in range(n + 2)])            # a[m] = a_m
    b = np.zeros(n + 2)                                              # b[j] = beta_n(j), Fu's second-moment coefficient
    for j in range(1, n):
        b[j] = 2.0 * n * (a[n + 1] - a[j]) / ((n - j + 1.0) * (n - j)) - 2.0 / (n - j)
    out = np.zeros((5, n - 1))
    for k in range(1, n):
        if k < i:
            out[0, k - 1] = i * (b[k] - b[k + 1]) / 2.0
        elif k == i:
            out[1, k - 1] = i * b[i]
        else:
            out[2, k - 1] = i * (b[i] - b[i + 1]) / 2.0
        if k == n - i:
            out[3, k - 1] = i * ((a[n] - a[k]) / (n - k) + (a[n] - a[i]) / (n - i) - (b[k] + b[i]) / 2.0)
        if k + i < n:
            out[4, k - 1] = 1.0 / k - i * (b[k] - b[k + 1] + b[i] - b[i + 1]) / 2.0
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the shipped sample size, a variant carried by 21 of 34 haplotypes ---
        {
            "setup": "import numpy as np\n",
            "call": "linked_site_spectrum_components(34, 21)",
            "gold_call": "_oracle_linked_site_spectrum_components(34, 21)",
        },
        # --- boundary: a singleton focal variant, so nothing can be strictly nested in it ---
        {
            "setup": "import numpy as np\n",
            "call": "linked_site_spectrum_components(12, 1)",
            "gold_call": "_oracle_linked_site_spectrum_components(12, 1)",
        },
        # --- edge: a large sample with the focal variant one copy short of fixation ---
        {
            "setup": "import numpy as np\n",
            "call": "linked_site_spectrum_components(80, 79)",
            "gold_call": "_oracle_linked_site_spectrum_components(80, 79)",
        },
        # --- edge: a small sample with the focal variant at half frequency, where every relation is populated ---
        {
            "setup": "import numpy as np\n",
            "call": "linked_site_spectrum_components(6, 3)",
            "gold_call": "_oracle_linked_site_spectrum_components(6, 3)",
        },
    ]
