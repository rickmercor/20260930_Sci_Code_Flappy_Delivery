"""
Fold a spectrum over derived counts 1 to m - 1 into minor-allele classes 1 to floor(m / 2).

When the ancestral state of a site cannot be assigned, because no outgroup sequence

aligns to the region, the two alleles of a site can only be told apart as minor and

major, and the spectrum is folded: in a sample of m sequences the class of derived

count k and the class of derived count m - k merge into one class of minor count k.

The class whose two counts coincide exists only for an even sample size, is a single

class, and is counted once. This step folds any spectrum over derived counts, observed

counts and expected counts alike.

Returns
-------
np.ndarray, shape (m // 2,): the number of sites of minor count j at entry j - 1, the class of equal counts counted once
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fold_spectrum(spectrum: "np.ndarray", m: int) -> "np.ndarray":
    '''Fold a spectrum over derived counts 1 to m - 1 into minor-allele classes 1 to floor(m / 2).

    Parameters
    ----------
    spectrum : np.ndarray
        Shape (m - 1,): the number (or expected number) of sites of derived
        count k at index k - 1, finite entries.
    m : int
        Number of sequences in the sample the spectrum refers to, >= 2.

    Returns
    -------
    folded : np.ndarray
        Shape (m // 2,). Entry j - 1 is the number of sites of minor count j:
        the sum of the entries for derived counts j and m - j when j < m / 2,
        and the entry for derived count m / 2 alone, counted once, when m is
        even and j = m / 2.

    Raises
    ------
    ValueError
        If m is not an integer >= 2, or if spectrum is not a one-dimensional
        array of finite numbers of length m - 1.
    '''
    return folded  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_fold_spectrum(spectrum: "np.ndarray", m: int) -> "np.ndarray":
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or int(m) < 2:
        raise ValueError("m must be an integer >= 2")
    m = int(m)
    x = np.asarray(spectrum, dtype=float) if not isinstance(spectrum, (str, bytes)) else None
    if x is None or x.ndim != 1 or x.shape[0] != m - 1 or not np.all(np.isfinite(x)):
        raise ValueError("spectrum must be a one-dimensional array of finite numbers of length m - 1")
    out = np.zeros(m // 2)
    for j in range(1, m // 2 + 1):
        out[j - 1] = x[j - 1] if 2 * j == m else x[j - 1] + x[m - j - 1]
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the expected spectrum of the shipped segment (13 carriers, absence derived), an odd sample ---
        {
            "setup": "import numpy as np\nspec = _oracle_indel_null_spectrum(34, 13, True)\n",
            "call": "fold_spectrum(spec, 13)",
            "gold_call": "_oracle_fold_spectrum(spec, 13)",
        },
        # --- boundary: an even sample, whose middle class is counted once ---
        {
            "setup": "import numpy as np\nspec = np.array([15.0, 8.0, 6.0, 5.0, 2.0, 3.0, 4.0])\n",
            "call": "fold_spectrum(spec, 8)",
            "gold_call": "_oracle_fold_spectrum(spec, 8)",
        },
        # --- edge: two sequences, a single class that is its own mirror image ---
        {
            "setup": "import numpy as np\nspec = np.array([7.0])\n",
            "call": "fold_spectrum(spec, 2)",
            "gold_call": "_oracle_fold_spectrum(spec, 2)",
        },
        # --- edge: the polarised flanking spectrum of the shipped sample folded over 34 sequences ---
        {
            "setup": "import numpy as np\nspec = np.array([292, 159, 91, 71, 57, 62, 41, 35, 37, 31, 25, 20, 15, 24, 15, 19, 26, 14, 17, 13, 7, 11, 12, 16, 12, 6, 11, 16, 12, 7, 7, 14, 13])\n",
            "call": "fold_spectrum(spec, 34)",
            "gold_call": "_oracle_fold_spectrum(spec, 34)",
        },
    ]
