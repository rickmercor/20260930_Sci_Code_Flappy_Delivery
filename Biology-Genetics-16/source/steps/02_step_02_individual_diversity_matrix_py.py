"""
'''Covariance of per-individual reference-allele PROPORTION across individuals.




    Individual k owns haplotype rows 2k and 2k+1, so its allele content at locus i

    is c[k, i] = (haplotypes[2k, i] + haplotypes[2k+1, i]) / 2, taking values in

    {0, 1/2, 1}. Return the population covariance of c across individuals, i.e.

    normalised by the number of individuals (ddof = 0), not by n - 1.




    Parameters

    ----------

    haplotypes : np.ndarray

        (2N, n_loci) array of 0/1 values with N >= 2 individuals.




    Returns

    -------

    L : np.ndarray

        (n_loci, n_loci) symmetric covariance matrix.




    Raises

    ------

    ValueError

        If haplotypes is not a 2D array, if it has fewer than 4 rows, if its row

        count is odd, if it has fewer than 1 column, or if it contains a value

        other than 0 or 1.

    '''

Fisher's average effects for relative fitness are the partial regression

coefficients of relative fitness on the per-individual allele content at every

locus simultaneously. The matrix that has to be inverted to obtain them, and that

the additive genetic variance is later contracted with, is therefore the

covariance matrix of that allele content computed across individuals.




The scale of this matrix is fixed by the units of the allele-content variable and

is not free. The variable is the PROPORTION of reference-allele copies carried by

an individual - one of 0, 1/2 or 1 - so that under Hardy-Weinberg equilibrium the

diagonal is p*q/2, i.e. one quarter of the gene diversity 2pq. Equivalently the

matrix factorises as L = B R B with R the correlation matrix and B diagonal

holding HALF the square root of the genetic diversities. Coding allele content as

the 0/1/2 dosage instead inflates every element of L fourfold and deflates the

final variance by the same factor.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def individual_diversity_matrix(haplotypes: np.ndarray) -> np.ndarray:
    '''Covariance of per-individual reference-allele PROPORTION across individuals.

    Individual k owns haplotype rows 2k and 2k+1, so its allele content at locus i
    is c[k, i] = (haplotypes[2k, i] + haplotypes[2k+1, i]) / 2, taking values in
    {0, 1/2, 1}. Return the population covariance of c across individuals, i.e.
    normalised by the number of individuals (ddof = 0), not by n - 1.

    Parameters
    ----------
    haplotypes : np.ndarray
        (2N, n_loci) array of 0/1 values with N >= 2 individuals.

    Returns
    -------
    L : np.ndarray
        (n_loci, n_loci) symmetric covariance matrix.

    Raises
    ------
    ValueError
        If haplotypes is not a 2D array, if it has fewer than 4 rows, if its row
        count is odd, if it has fewer than 1 column, or if it contains a value
        other than 0 or 1.
    '''
    return L  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_individual_diversity_matrix(haplotypes: np.ndarray) -> np.ndarray:
    H = np.asarray(haplotypes)

    if H.ndim != 2:
        raise ValueError("haplotypes must be a 2D array")
    if H.shape[0] < 4:
        raise ValueError(
            "haplotypes must contain at least 2 individuals (4 rows)"
        )
    if H.shape[0] % 2 != 0:
        raise ValueError("haplotypes must have an even number of rows")
    if H.shape[1] < 1:
        raise ValueError("haplotypes must have at least one locus")

    Hf = np.asarray(H, dtype=float)

    if not np.all((Hf == 0.0) | (Hf == 1.0)):
        raise ValueError("haplotypes must contain only 0 and 1")

    C = 0.5 * (Hf[0::2] + Hf[1::2])
    Cc = C - C.mean(axis=0, keepdims=True)

    return (Cc.T @ Cc) / C.shape[0]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the shipped base population ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(2043)
Z = rng.standard_normal((4000, 10))
X = np.empty_like(Z); X[:, 0] = Z[:, 0]
s = np.sqrt(1 - 0.90 ** 2)
for i in range(1, 10):
    X[:, i] = 0.90 * X[:, i - 1] + s * Z[:, i]
thr = np.array([ 1.0364334,  0.7461852,  0.5084881,  0.2967378,  0.0976349,
                -0.0976349, -0.2967378, -0.5084881, -0.7461852, -1.0364334])
H = (X > thr[None, :]).astype(int)
""",
            "call": "individual_diversity_matrix(H)",
            "gold_call": "_oracle_individual_diversity_matrix(H)",
        },
        # --- boundary: two loci in perfect gametic-phase disequilibrium, so L
        #     has non-zero entries but is exactly rank 1 ---
        {
            "setup": """import numpy as np
H = np.array([[1, 1], [1, 1], [0, 0], [0, 0], [1, 1], [0, 0]], dtype=int)
""",
            "call": "individual_diversity_matrix(H)",
            "gold_call": "_oracle_individual_diversity_matrix(H)",
        },
        # --- edge: a monomorphic locus alongside a polymorphic one, giving an
        #     exactly zero row and column ---
        {
            "setup": """import numpy as np
H = np.array([[1, 0], [1, 0], [1, 1], [1, 1], [1, 0], [1, 1]], dtype=int)
""",
            "call": "individual_diversity_matrix(H)",
            "gold_call": "_oracle_individual_diversity_matrix(H)",
        },
    ]
