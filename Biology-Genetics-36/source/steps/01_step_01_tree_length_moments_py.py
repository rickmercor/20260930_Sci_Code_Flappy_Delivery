"""
Power moments of the total branch length of a sample genealogy under a size history indexed by the number of ancestral lineages.

The genealogy of a sample of DNA sequences from a non-recombining locus is a tree

whose branches carry the mutations that separate the sequences. Almost every

statistic of the sample's variation is governed by the total branch length of

that tree: the number of mutations it carries is a Poisson count with mean

proportional to that length, so the distribution of the number of segregating

sites, and every probability of a particular pattern of mutations, rests on the

distribution of the length. The length itself is a random variable, because the

waiting times between successive coalescences of ancestral lineages are random

and, when the population size has changed through the history of the sample,

each waiting time has its own scale. This step computes the power moments of the

total length for a history in which the effective size is constant while a

given number of ancestral lineages exists.

Returns
-------
np.ndarray of shape (max_order + 1,), the power moments E[L^m] of the total branch length for m = 0 to max_order, moments[0] = 1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def tree_length_moments(relative_sizes: "np.ndarray", max_order: int) -> "np.ndarray":
    '''Power moments of the total branch length of a sample genealogy under a size history indexed by the number of ancestral lineages.

    The sample holds n = len(relative_sizes) + 1 sequences. Their genealogy
    passes through the states with n, n - 1, ..., 2 ancestral lineages.
    relative_sizes[j] is the effective population size while j + 2 ancestral
    lineages exist, relative to a reference size N_ref, so the entry for the
    state with two lineages comes first and the entry for the state with n
    lineages last. Time is measured in units of 4 N_ref generations: while k
    lineages exist, the waiting time until two of them coalesce is
    exponentially distributed with rate k (k - 1) / relative_sizes[k - 2],
    independently of the waiting times of the other states. The total branch
    length L of the genealogy is the sum over the states of k times the waiting
    time spent with k lineages. Return the power moments E[L^m] for
    m = 0, 1, ..., max_order.

    Parameters
    ----------
    relative_sizes : np.ndarray
        Shape (n - 1,), n >= 2; finite entries > 0, one per coalescent state as
        described above.
    max_order : int
        Highest moment order to return, >= 0.

    Returns
    -------
    moments : np.ndarray
        Shape (max_order + 1,). moments[m] = E[L^m]; moments[0] = 1. Every entry
        is accurate to a relative error below 1e-12.

    Raises
    ------
    ValueError
        If relative_sizes is not a one-dimensional array with at least one
        entry, if any of its entries is not a finite number > 0, or if
        max_order is not an integer >= 0.
    '''
    return moments  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_sizes(relative_sizes) -> "np.ndarray":
    """Validate a size vector and return it as a float array."""
    v = np.asarray(relative_sizes)
    if v.ndim != 1 or v.size < 1:
        raise ValueError("relative_sizes must be a one-dimensional array with at least one entry")
    if not np.issubdtype(v.dtype, np.number) or np.issubdtype(v.dtype, np.bool_):
        raise ValueError("relative_sizes must hold finite numbers > 0")
    v = v.astype(float)
    if not np.all(np.isfinite(v)) or np.any(v <= 0.0):
        raise ValueError("relative_sizes must hold finite numbers > 0")
    return v


def _oracle_tree_length_moments(relative_sizes: "np.ndarray", max_order: int) -> "np.ndarray":
    v = _check_sizes(relative_sizes)
    if isinstance(max_order, bool) or not isinstance(max_order, (int, np.integer)) or int(max_order) < 0:
        raise ValueError("max_order must be an integer >= 0")
    m_max = int(max_order)
    # L is a sum of independent exponential variables k t_k with rates (k - 1) / v_{k-1};
    # its cumulants are kappa_m = (m - 1)! sum_k (v_{k-1} / (k - 1))^m, and the power
    # moments follow from the cumulants by the standard recursion
    scales = v / np.arange(1, v.size + 1, dtype=float)          # v_{k-1} / (k - 1), k = 2..n
    moments = np.empty(m_max + 1)
    moments[0] = 1.0
    for m in range(1, m_max + 1):
        total = 0.0
        coeff = 1.0                                              # (m - 1)_{i-1} = (m-1)(m-2)...(m-i+1)
        for i in range(1, m + 1):
            total += coeff * float(np.sum(scales ** i)) * moments[m - i]
            coeff *= (m - i)
        moments[m] = total
    return moments

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: fourteen sequences, the size reduced to 0.15 while 3, 4 or 5 lineages exist ---
        {
            "setup": "import numpy as np\nsizes = np.array([1.0, 0.15, 0.15, 0.15] + [1.0] * 9)\n",
            "call": "tree_length_moments(sizes, 12)",
            "gold_call": "_oracle_tree_length_moments(sizes, 12)",
        },
        # --- boundary: two sequences, a single coalescent state, so L is one exponential variable ---
        {
            "setup": "import numpy as np\nsizes = np.array([1.0])\n",
            "call": "tree_length_moments(sizes, 6)",
            "gold_call": "_oracle_tree_length_moments(sizes, 6)",
        },
        # --- edge: a growing population (size proportional to the square root of the number of
        #     lineages) with high moment orders, where the moments span many orders of magnitude ---
        {
            "setup": "import numpy as np\nsizes = np.sqrt(np.arange(2, 26, dtype=float))\n",
            "call": "tree_length_moments(sizes, 30)",
            "gold_call": "_oracle_tree_length_moments(sizes, 30)",
        },
    ]
