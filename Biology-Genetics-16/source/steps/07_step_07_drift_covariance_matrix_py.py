"""
'''Covariance across loci of the drift contribution to observed frequency change.




    Return the covariance, taken over the evolutionary process, of the part of the

    allele-frequency change between the two measurements of a replicate that is

    due to genetic drift. The replicate history is the one described for

    expected_change_operator: founded from the offspring of one round of random

    mating in the base population, measured in that founding generation and again

    n_generations generations later, with the structure carried by base_matrix

    eroded by recombination and by drift, drift acting at founding_effective_size

    in the round of random mating that produced the founders and at

    inbreeding_effective_size in every later round. The sampling of allele

    frequencies in each generation of the window is governed by the variance

    effective size n_effective.




    Parameters

    ----------

    base_matrix : np.ndarray

        (n_loci, n_loci) symmetric weighted base-population matrix, as returned

        by weighted_base_diversity_matrix.

    recombination : np.ndarray

        (n_loci, n_loci) symmetric matrix of pairwise recombination probabilities

        with a zero diagonal and entries in [0, 0.5).

    n_generations : int

        Number of generations between the two measurements, >= 1.

    founding_effective_size : float

        Inbreeding effective size of the round of random mating that produced the

        replicate's founders. Must be > 0.5.

    inbreeding_effective_size : float

        Inbreeding effective size of the replicate in every round of mating after

        the founding round. Must be > 0.5.

    n_effective : float

        Variance effective size governing the sampling variance of allele

        frequency. Must be > 0.




    Returns

    -------

    covariance : np.ndarray

        (n_loci, n_loci) symmetric matrix.




    Raises

    ------

    ValueError

        If base_matrix or recombination is not a square 2D array, if their shapes

        disagree, if either is not symmetric within atol=1e-12, if any entry of

        recombination is outside [0, 0.5), if n_generations is not an integer >= 1,

        if founding_effective_size or inbreeding_effective_size is not a finite

        number > 0.5, or if n_effective is not a finite number > 0.

    '''

Observed allele-frequency change is the sum of a predictable response to selection

and a random contribution from drift. The two can only be separated across

replicates if the covariance structure of the drift contribution is known, because

that structure is what the replicate observations must be weighted by when the

response to selection is fitted. Drift is not independent across loci: linked loci

drift together, so the covariance matrix is not diagonal.




The sampling variance of allele frequency in each generation is set by the

VARIANCE effective size rather than the inbreeding effective size, and every

generation of the observation window contributes its own round of sampling.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def drift_covariance_matrix(base_matrix: np.ndarray, recombination: np.ndarray,
                            n_generations: int, founding_effective_size: float,
                            inbreeding_effective_size: float,
                            n_effective: float) -> np.ndarray:
    '''Covariance across loci of the drift contribution to observed frequency change.

    Return the covariance, taken over the evolutionary process, of the part of the
    allele-frequency change between the two measurements of a replicate that is
    due to genetic drift. The replicate history is the one described for
    expected_change_operator: founded from the offspring of one round of random
    mating in the base population, measured in that founding generation and again
    n_generations generations later, with the structure carried by base_matrix
    eroded by recombination and by drift, drift acting at founding_effective_size
    in the round of random mating that produced the founders and at
    inbreeding_effective_size in every later round. The sampling of allele
    frequencies in each generation of the window is governed by the variance
    effective size n_effective.

    Parameters
    ----------
    base_matrix : np.ndarray
        (n_loci, n_loci) symmetric weighted base-population matrix, as returned
        by weighted_base_diversity_matrix.
    recombination : np.ndarray
        (n_loci, n_loci) symmetric matrix of pairwise recombination probabilities
        with a zero diagonal and entries in [0, 0.5).
    n_generations : int
        Number of generations between the two measurements, >= 1.
    founding_effective_size : float
        Inbreeding effective size of the round of random mating that produced the
        replicate's founders. Must be > 0.5.
    inbreeding_effective_size : float
        Inbreeding effective size of the replicate in every round of mating after
        the founding round. Must be > 0.5.
    n_effective : float
        Variance effective size governing the sampling variance of allele
        frequency. Must be > 0.

    Returns
    -------
    covariance : np.ndarray
        (n_loci, n_loci) symmetric matrix.

    Raises
    ------
    ValueError
        If base_matrix or recombination is not a square 2D array, if their shapes
        disagree, if either is not symmetric within atol=1e-12, if any entry of
        recombination is outside [0, 0.5), if n_generations is not an integer >= 1,
        if founding_effective_size or inbreeding_effective_size is not a finite
        number > 0.5, or if n_effective is not a finite number > 0.
    '''
    return covariance  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_drift_covariance_matrix(base_matrix: np.ndarray, recombination: np.ndarray,
                                    n_generations: int, founding_effective_size: float,
                                    inbreeding_effective_size: float,
                                    n_effective: float) -> np.ndarray:
    L = np.asarray(base_matrix, dtype=float)
    R = np.asarray(recombination, dtype=float)
    for name, A in (("base_matrix", L), ("recombination", R)):
        if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
            raise ValueError(f"{name} must be a square 2D array")
        if not np.allclose(A, A.T, rtol=0.0, atol=1e-12):
            raise ValueError(f"{name} must be symmetric within atol=1e-12")
    if L.shape != R.shape:
        raise ValueError("base_matrix and recombination must have the same shape")
    if not np.all(np.isfinite(R)) or np.any(R < 0.0) or np.any(R >= 0.5):
        raise ValueError("recombination entries must lie in [0, 0.5)")
    if isinstance(n_generations, bool) or not isinstance(n_generations, (int, np.integer)):
        raise ValueError("n_generations must be an integer >= 1")
    if int(n_generations) < 1:
        raise ValueError("n_generations must be an integer >= 1")
    sizes = []
    for name, value in (("founding_effective_size", founding_effective_size),
                        ("inbreeding_effective_size", inbreeding_effective_size)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
            raise ValueError(f"{name} must be a finite number > 0.5")
        if not np.isfinite(float(value)) or float(value) <= 0.5:
            raise ValueError(f"{name} must be a finite number > 0.5")
        sizes.append(float(value))
    ne0, ne = sizes
    if isinstance(n_effective, bool) or not isinstance(
            n_effective, (int, float, np.floating, np.integer)):
        raise ValueError("n_effective must be a finite number > 0")
    nE = float(n_effective)
    if not np.isfinite(nE) or nE <= 0.0:
        raise ValueError("n_effective must be a finite number > 0")

    # Same window and drift factor as expected_change_operator: generations
    # t = 1 .. n_generations, founding bottleneck in round k = 0.
    founding = 1.0 - 1.0 / (2.0 * ne0)
    drift = 1.0 - 1.0 / (2.0 * ne)
    total = np.zeros_like(L)
    for t in range(1, int(n_generations) + 1):
        total = total + ((1.0 - R) / nE) * ((1.0 - R) ** t) * founding * (drift ** (t - 1))
    return L * total

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    _SETUP = """import numpy as np
rng = np.random.default_rng(2043)
Z = rng.standard_normal((4000, 10))
X = np.empty_like(Z); X[:, 0] = Z[:, 0]
s = np.sqrt(1 - 0.90 ** 2)
for i in range(1, 10):
    X[:, i] = 0.90 * X[:, i - 1] + s * Z[:, i]
thr = np.array([ 1.0364334,  0.7461852,  0.5084881,  0.2967378,  0.0976349,
                -0.0976349, -0.2967378, -0.5084881, -0.7461852, -1.0364334])
H = (X > thr[None, :]).astype(float)
d = np.concatenate([[0.0], np.cumsum(-0.5 * np.log1p(-2.0 * np.full(9, 0.05)))])
R = 0.5 * (1.0 - np.exp(-2.0 * np.abs(d[:, None] - d[None, :])))
p = H.mean(axis=0, keepdims=True)
a, b = H[0::2] - p, H[1::2] - p
n = a.shape[0]
Lg = 0.25 * (a.T @ a + b.T @ b) / n
Lx = 0.25 * (a.T @ b + b.T @ a) / n
L = Lg + (R / (1.0 - R)) * Lx
"""
    return [
        # --- normal: the smallest shipped replicate, N = 150, V_o = 6 -> N_E = 75 ---
        {
            "setup": _SETUP,
            "call": "drift_covariance_matrix(L, R, 2, 50.0, 150.0, 75.0)",
            "gold_call": "_oracle_drift_covariance_matrix(L, R, 2, 50.0, 150.0, 75.0)",
        },
        # --- boundary: a single recorded generation, where the diagonal is the
        #     Wright-Fisher variance p*q / (2 N_E) reduced by the diversity lost in
        #     the founding bottleneck, a factor 1 - 1/(2 * 50) ---
        {
            "setup": _SETUP,
            "call": "drift_covariance_matrix(L, R, 1, 50.0, 2000.0, 1000.0)",
            "gold_call": "_oracle_drift_covariance_matrix(L, R, 1, 50.0, 2000.0, 1000.0)",
        },
        # --- edge: complete linkage, a founding size and N_E below 1, so drift
        #     dominates ---
        {
            "setup": """import numpy as np
L = np.array([[0.12, 0.05], [0.05, 0.10]])
R = np.zeros((2, 2))
""",
            "call": "drift_covariance_matrix(L, R, 4, 0.6, 0.8, 0.5)",
            "gold_call": "_oracle_drift_covariance_matrix(L, R, 4, 0.6, 0.8, 0.5)",
        },
    ]
