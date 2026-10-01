"""
Pseudo-replication parameter rho of the mean r^2 over all pairs of SNPs on different chromosomes.

In a genomic panel with few chromosomes, the unlinked pairs used by the

linkage-disequilibrium method are formed by pairing every SNP on one chromosome

with every SNP on another, so there are far more unlinked pairs than independent

pieces of information. Their r^2 values are correlated: pairs can share a SNP,

and SNPs on the same chromosome are themselves in drift-induced linkage

disequilibrium. The source summarises this pseudo-replication in one parameter

that measures how much the variance of the mean unlinked r^2 exceeds its value

under independence, and it expresses the correlations through the drift-induced

LD along each chromosome. A sample reports a raw r^2 for a pair of SNPs on the

same chromosome that carries a sampling component as well, the same component an

unlinked pair carries.

Returns
-------
float, the pseudo-replication parameter, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pseudo_replication_rho(raw_same_chromosome_r2: "np.ndarray", n_snps: "np.ndarray", sample_size: int) -> float:
    '''Pseudo-replication parameter rho of the mean r^2 over all pairs of SNPs on different chromosomes.

    The panel has C chromosomes, with L_c SNPs on chromosome c, genotyped in a
    sample of S diploid individuals. The unlinked pairs are all N = sum over
    a < b of L_a L_b pairs of SNPs on different chromosomes, and their r^2
    values are averaged. rho is the source's parameter: the sum, over all
    unordered pairs of distinct unlinked pairs, of the correlation between
    their r^2 values, divided by N, so that the variance of the mean is
    1 + 2 rho times its value if the N r^2 values were independent. The
    correlations follow the source's model, which expresses them through the
    drift-induced r^2 among SNPs that lie on the same chromosome;
    drift-induced LD between SNPs on different chromosomes is zero.

    Entry c of raw_same_chromosome_r2 is the sample's mean raw r^2 over all
    L_c (L_c - 1) / 2 unordered pairs of distinct SNPs on chromosome c. Its
    sampling component has the same expectation as for an unlinked pair,
    1/S + 3.19/S^2, and the rest is the drift-induced part.

    Parameters
    ----------
    raw_same_chromosome_r2 : np.ndarray
        Shape (C,), C >= 2; the sample's mean raw r^2 over the pairs of
        distinct SNPs on each chromosome, each a finite number between
        1/S + 3.19/S^2 and 1.
    n_snps : np.ndarray
        Shape (C,); integer numbers of SNPs L_c, each from 2 to 10**6.
    sample_size : int
        Number of diploid individuals S, an integer from 30 to 100000.

    Returns
    -------
    rho : float
        The pseudo-replication parameter, as a native Python float.

    Raises
    ------
    ValueError
        If the arrays are not one-dimensional with the same length C >= 2, if
        n_snps holds a value that is not an integer from 2 to 10**6, if
        sample_size is not an integer from 30 to 100000, or if an entry of
        raw_same_chromosome_r2 is not finite, is below the sampling
        expectation or is above 1.
    '''
    return rho  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_pseudo_replication_rho(raw_same_chromosome_r2: "np.ndarray", n_snps: "np.ndarray", sample_size: int) -> float:
    raw = np.asarray(raw_same_chromosome_r2, dtype=float)
    L = np.asarray(n_snps)
    if raw.ndim != 1 or L.ndim != 1 or raw.shape != L.shape or L.size < 2:
        raise ValueError("raw_same_chromosome_r2 and n_snps must be 1-D arrays of the same length C >= 2")
    if not np.issubdtype(L.dtype, np.integer) or np.any(L < 2) or np.any(L > 10 ** 6):
        raise ValueError("n_snps must hold integers from 2 to 10**6")
    if isinstance(sample_size, bool) or not isinstance(sample_size, (int, np.integer)) \
            or not 30 <= int(sample_size) <= 100000:
        raise ValueError("sample_size must be an integer from 30 to 100000")
    s = int(sample_size)
    e_samp = 1.0 / s + 3.19 / (s * s)
    if not np.all(np.isfinite(raw)) or np.any(raw < e_samp) or np.any(raw > 1.0):
        raise ValueError("each raw mean must be finite, at least the sampling expectation and at most 1")
    Lf = L.astype(float)
    A = (raw - e_samp) * Lf * (Lf - 1.0) / 2.0        # drift-induced r^2 summed over the pairs on each chromosome
    total_pairs = 0.0
    corr_sum = 0.0
    C = L.size
    for a in range(C):
        for b in range(a + 1, C):
            total_pairs += Lf[a] * Lf[b]
            # pairs of unlinked (a, b) pairs sharing a SNP on a or on b, then pairs sharing none
            corr_sum += Lf[a] * A[b] + Lf[b] * A[a] + 2.0 * A[a] * A[b]
    return float(corr_sum / total_pairs)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the pilot's three chromosomes, raw means near 0.035 in a sample of 45 ---
        {
            "setup": "import numpy as np\nraw = np.array([0.0357, 0.0352, 0.0345])\nL = np.array([3076, 2273, 1955])\n",
            "call": "pseudo_replication_rho(raw, L, 45)",
            "gold_call": "_oracle_pseudo_replication_rho(raw, L, 45)",
        },
        # --- boundary: two chromosomes, the only pair of chromosomes, in a larger sample ---
        {
            "setup": "import numpy as np\nraw = np.array([0.0265, 0.0258])\nL = np.array([5368, 3552])\n",
            "call": "pseudo_replication_rho(raw, L, 68)",
            "gold_call": "_oracle_pseudo_replication_rho(raw, L, 68)",
        },
        # --- edge: four chromosomes at the smallest allowed sample, one of them with no drift-induced LD at
        #     all (its raw mean is exactly the sampling expectation) and one carrying only two SNPs ---
        {
            "setup": "import numpy as np\nS = 30\nes = 1 / S + 3.19 / S ** 2\n"
                     "raw = np.array([es + 0.4, es, es + 0.05, es + 0.12])\nL = np.array([2, 800, 1500, 40])\n",
            "call": "pseudo_replication_rho(raw, L, 30)",
            "gold_call": "_oracle_pseudo_replication_rho(raw, L, 30)",
        },
    ]
