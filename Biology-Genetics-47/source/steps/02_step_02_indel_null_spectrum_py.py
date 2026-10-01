"""
Expected spectrum of neutral sites inside a presence/absence segment, among the haplotypes that carry the segment, for a stated polarity of the variant.

An insertion or deletion polymorphism differs from an inversion or a point mutation in

one respect that matters for the spectrum: the sequence inside the variable segment

exists on one background only. Sites inside a deleted segment can be observed only on

the haplotypes that still carry the segment, the ancestral background; sites inside an

inserted segment exist only on the haplotypes that carry the insertion, the derived

background. In either case the analysed sample is the set of carriers of the segment,

and the expected spectrum among them follows from the conditional spectrum of the

previous step once the polarity of the variant, which state is derived, is fixed.

This step assembles that expected spectrum for either polarity.

Returns
-------
np.ndarray, shape (carriers - 1,): the expected number of sites inside the segment whose derived allele is carried by exactly k of the carriers (entry k - 1), per unit theta L
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def indel_null_spectrum(n: int, carriers: int, absence_derived: bool) -> "np.ndarray":
    '''Expected spectrum of neutral sites inside a presence/absence segment, among the haplotypes that carry the segment, for a stated polarity of the variant.

    The sample holds n haplotypes; a segment is present on exactly `carriers`
    of them and absent from the others, the presence/absence variant being a
    single-origin, neutral, biallelic variant in complete linkage with every
    site inside the segment, under the model of the previous step. Sites
    inside the segment exist only on the carriers, so they are analysed in the
    sample of `carriers` sequences and only sites that are polymorphic among
    the carriers are counted. If absence_derived is True the absent state
    arose once by a deletion, so the carriers are the ancestral background;
    otherwise the present state arose once by an insertion, so the carriers
    are the derived background. Return the expected number of polymorphic
    sites inside the segment classified by the number k of carriers that hold
    the derived allele of the site (k from 1 to carriers - 1), the derived
    allele of a site being the allele that arose by mutation, per unit
    theta L as in the previous step.

    Parameters
    ----------
    n : int
        Number of haplotypes in the sample, >= 3.
    carriers : int
        Number of haplotypes carrying the segment, from 2 to n - 1.
    absence_derived : bool
        True if the absent state is the derived state (a deletion), False if
        the present state is the derived state (an insertion).

    Returns
    -------
    spectrum : np.ndarray
        Shape (carriers - 1,). Entry k - 1 is the expected number of sites
        inside the segment whose derived allele is carried by exactly k of the
        carriers, per unit theta L. Every entry is accurate to a relative error
        below 1e-12.

    Raises
    ------
    ValueError
        If n is not an integer >= 3, if carriers is not an integer from 2 to
        n - 1, or if absence_derived is not a bool.
    '''
    return spectrum  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_indel_null_spectrum(n: int, carriers: int, absence_derived: bool) -> "np.ndarray":
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 3:
        raise ValueError("n must be an integer >= 3")
    if isinstance(carriers, bool) or not isinstance(carriers, (int, np.integer)) or not 2 <= int(carriers) <= int(n) - 1:
        raise ValueError("carriers must be an integer from 2 to n - 1")
    if not isinstance(absence_derived, (bool, np.bool_)):
        raise ValueError("absence_derived must be a bool")
    n, c = int(n), int(carriers)
    k = np.arange(1, c)
    if absence_derived:
        i = n - c                                                    # sample count of the deletion, the derived state
        comp = _oracle_linked_site_spectrum_components(n, i)
        # a site polymorphic among the survivors is either disjoint from the deletion (count k in the whole
        # sample) or encloses it (count k + i in the whole sample, its copies on the deleted background unseen)
        return comp[4, k - 1] + comp[2, k + i - 1]
    comp = _oracle_linked_site_spectrum_components(n, c)             # the insertion itself is the focal variant
    return comp[0, k - 1]                                            # every site inside it is strictly nested

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the shipped segment, present on 13 of 34 haplotypes, absence derived ---
        {
            "setup": "import numpy as np\n",
            "call": "indel_null_spectrum(34, 13, True)",
            "gold_call": "_oracle_indel_null_spectrum(34, 13, True)",
        },
        # --- normal: the same segment with the presence derived ---
        {
            "setup": "import numpy as np\n",
            "call": "indel_null_spectrum(34, 13, False)",
            "gold_call": "_oracle_indel_null_spectrum(34, 13, False)",
        },
        # --- boundary: two carriers, so the spectrum has a single class ---
        {
            "setup": "import numpy as np\n",
            "call": "indel_null_spectrum(10, 2, True)",
            "gold_call": "_oracle_indel_null_spectrum(10, 2, True)",
        },
        # --- edge: an insertion carried by all but one haplotype of a large sample ---
        {
            "setup": "import numpy as np\n",
            "call": "indel_null_spectrum(60, 59, False)",
            "gold_call": "_oracle_indel_null_spectrum(60, 59, False)",
        },
    ]
