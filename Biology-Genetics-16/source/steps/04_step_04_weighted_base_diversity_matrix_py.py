"""
'''Base-population matrix that the forward projection of diversity acts on.




    Split the allele-content covariance of the base population into its

    gametic-phase component (pairs of alleles on the same gamete) and its

    non-gametic-phase component (pairs of alleles on the two different gametes of

    an individual). Combine them into the matrix L_tilde defined by this property:

    in the absence of selection and drift, one round of random mating produces a

    generation whose expected gametic-phase component is

    (1 - recombination) * L_tilde, elementwise.




    Parameters

    ----------

    haplotypes : np.ndarray

        (2N, n_loci) array of 0/1 values, individual k owning rows 2k and 2k+1.

    recombination : np.ndarray

        (n_loci, n_loci) symmetric matrix of pairwise recombination probabilities

        with a zero diagonal and entries in [0, 0.5).




    Returns

    -------

    L_tilde : np.ndarray

        (n_loci, n_loci) symmetric matrix.




    Raises

    ------

    ValueError

        If haplotypes is not a 2D array with an even number of rows >= 4, if it

        contains a value other than 0 or 1, if recombination is not a square 2D

        array whose size matches the locus count, if recombination is not

        symmetric within atol=1e-12, or if any entry of recombination is outside

        [0, 0.5).

    '''

The covariance of per-individual allele content splits into two parts with

different evolutionary dynamics. The gametic-phase part L' is the contribution

from pairs of alleles carried on the SAME gamete of an individual; its

off-diagonals are gametic-phase (ordinary) linkage disequilibrium. The

non-gametic-phase part L'' is the contribution from pairs of alleles carried on

the two DIFFERENT gametes of an individual; its diagonal is proportional to the

Hardy-Weinberg disequilibrium and its off-diagonals are non-gametic-phase

disequilibrium. The disequilibria in both parts vanish in a randomly mating,

infinitely large population; neither vanishes in a finite one.




The two parts do not persist alike, because meiosis reshuffles which alleles share

a gamete. Projecting the structure of the base population forward therefore needs

a single matrix that already accounts for how each part feeds the gametic-phase

disequilibrium of the next generation.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def weighted_base_diversity_matrix(haplotypes: np.ndarray,
                                   recombination: np.ndarray) -> np.ndarray:
    '''Base-population matrix that the forward projection of diversity acts on.

    Split the allele-content covariance of the base population into its
    gametic-phase component (pairs of alleles on the same gamete) and its
    non-gametic-phase component (pairs of alleles on the two different gametes of
    an individual). Combine them into the matrix L_tilde defined by this property:
    in the absence of selection and drift, one round of random mating produces a
    generation whose expected gametic-phase component is
    (1 - recombination) * L_tilde, elementwise.

    Parameters
    ----------
    haplotypes : np.ndarray
        (2N, n_loci) array of 0/1 values, individual k owning rows 2k and 2k+1.
    recombination : np.ndarray
        (n_loci, n_loci) symmetric matrix of pairwise recombination probabilities
        with a zero diagonal and entries in [0, 0.5).

    Returns
    -------
    L_tilde : np.ndarray
        (n_loci, n_loci) symmetric matrix.

    Raises
    ------
    ValueError
        If haplotypes is not a 2D array with an even number of rows >= 4, if it
        contains a value other than 0 or 1, if recombination is not a square 2D
        array whose size matches the locus count, if recombination is not
        symmetric within atol=1e-12, or if any entry of recombination is outside
        [0, 0.5).
    '''
    return L_tilde  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# HELPER FUNCTIONS

def _phase_components(haplotypes: np.ndarray):
    """Split the allele-content covariance into gametic and non-gametic parts.

    With a and b the two gamete sets of the population, both centred on the
    POOLED reference-allele frequency (the labels a and b are arbitrary, so
    centring them separately would leak a finite-sample artefact into the split):

        L'  = (1/4) [Cov(a_i, a_j) + Cov(b_i, b_j)]
        L'' = (1/4) [Cov(a_i, b_j) + Cov(b_i, a_j)]

    so that L' + L'' is exactly the covariance of the allele proportions, and
    L'[i, i] = p_i q_i / 2 exactly.
    """
    Hf = np.asarray(haplotypes, dtype=float)
    p = Hf.mean(axis=0, keepdims=True)
    a, b = Hf[0::2] - p, Hf[1::2] - p
    n = a.shape[0]
    gametic = 0.25 * (a.T @ a + b.T @ b) / n
    cross = 0.25 * (a.T @ b + b.T @ a) / n
    return gametic, cross


# ORACLE FUNCTION

def _oracle_weighted_base_diversity_matrix(haplotypes: np.ndarray,
                                           recombination: np.ndarray) -> np.ndarray:
    H = np.asarray(haplotypes)
    if H.ndim != 2 or H.shape[0] < 4 or H.shape[0] % 2 != 0:
        raise ValueError("haplotypes must be 2D with an even number of rows >= 4")
    Hf = np.asarray(H, dtype=float)
    if not np.all((Hf == 0.0) | (Hf == 1.0)):
        raise ValueError("haplotypes must contain only 0 and 1")
    R = np.asarray(recombination, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("recombination must be a square 2D array")
    if R.shape[0] != Hf.shape[1]:
        raise ValueError("recombination size must match the number of loci")
    if not np.allclose(R, R.T, rtol=0.0, atol=1e-12):
        raise ValueError("recombination must be symmetric within atol=1e-12")
    if not np.all(np.isfinite(R)) or np.any(R < 0.0) or np.any(R >= 0.5):
        raise ValueError("recombination entries must lie in [0, 0.5)")

    gametic, cross = _phase_components(Hf)
    return gametic + (R / (1.0 - R)) * cross

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    _SETUP_SHIPPED = """import numpy as np
rng = np.random.default_rng(2043)
Z = rng.standard_normal((4000, 10))
X = np.empty_like(Z); X[:, 0] = Z[:, 0]
s = np.sqrt(1 - 0.90 ** 2)
for i in range(1, 10):
    X[:, i] = 0.90 * X[:, i - 1] + s * Z[:, i]
thr = np.array([ 1.0364334,  0.7461852,  0.5084881,  0.2967378,  0.0976349,
                -0.0976349, -0.2967378, -0.5084881, -0.7461852, -1.0364334])
H = (X > thr[None, :]).astype(int)
d = np.concatenate([[0.0], np.cumsum(-0.5 * np.log1p(-2.0 * np.full(9, 0.05)))])
R = 0.5 * (1.0 - np.exp(-2.0 * np.abs(d[:, None] - d[None, :])))
"""
    return [
        # --- normal: the shipped base population and map ---
        {
            "setup": _SETUP_SHIPPED,
            "call": "weighted_base_diversity_matrix(H, R)",
            "gold_call": "_oracle_weighted_base_diversity_matrix(H, R)",
        },
        # --- boundary: complete linkage, R = 0, so L'' is discarded entirely ---
        {
            "setup": """import numpy as np
H = np.array([[1, 1], [0, 1], [1, 0], [0, 0], [1, 1], [0, 0]], dtype=int)
R = np.zeros((2, 2))
""",
            "call": "weighted_base_diversity_matrix(H, R)",
            "gold_call": "_oracle_weighted_base_diversity_matrix(H, R)",
        },
        # --- edge: gametic-phase and non-gametic-phase associations cancel
        #     exactly, so the allele-content covariance is identically zero while
        #     L' and L'' are individually large and of opposite sign. Every entry
        #     of the returned matrix is then produced by the r/(1-r) weighting
        #     alone, which a route that returns L' + L'' cannot reproduce. ---
        {
            "setup": """import numpy as np
H = np.array([[1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0]], dtype=int)
R = np.array([[0.0, 0.02], [0.02, 0.0]])
""",
            "call": "weighted_base_diversity_matrix(H, R)",
            "gold_call": "_oracle_weighted_base_diversity_matrix(H, R)",
        },
    ]
