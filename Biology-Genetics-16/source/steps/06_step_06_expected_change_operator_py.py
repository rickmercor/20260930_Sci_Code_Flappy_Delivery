"""
'''Operator mapping base-population average effects onto expected frequency change.




    Return the matrix L_m such that L_m @ alpha_bar is the expected

    allele-frequency change due to selection between the two measurements of a

    replicate, where alpha_bar holds the mean average effects for relative

    fitness. The replicate was founded from the offspring of one round of random

    mating in the base population; the first measurement is taken in that

    founding generation and the second n_generations generations later. From the

    base population onward, the structure carried by base_matrix is eroded by

    recombination and by drift. Drift acts at founding_effective_size in the round

    of random mating that produced the founders and at inbreeding_effective_size in

    every later round.




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




    Returns

    -------

    operator : np.ndarray

        (n_loci, n_loci) symmetric matrix.




    Raises

    ------

    ValueError

        If base_matrix or recombination is not a square 2D array, if their shapes

        disagree, if either is not symmetric within atol=1e-12, if any entry of

        recombination is outside [0, 0.5), if n_generations is not an integer >= 1,

        or if founding_effective_size or inbreeding_effective_size is not a finite

        number > 0.5.

    '''

The average effects for relative fitness are properties of the base population,

but the allele-frequency change actually observed accumulated over several

generations during which the diversity/disequilibrium structure was itself

changing. The operator that maps average effects onto expected frequency change

over the experiment is therefore neither the base-population matrix nor that

matrix multiplied by the number of generations: every generation in the

observation window contributes the response to selection acting on the structure

the population carries in that generation.




The replicate populations are not the base population itself. Each was founded

from the offspring of one round of random mating in the base population, and its

allele frequencies were first recorded in that founding generation. Recombination

and drift both erode the structure inherited from the base, drift at a rate set by

the inbreeding effective size of each round of mating; the round that produced the

founders passed through a bottleneck, so that rate is not the same in every round.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def expected_change_operator(base_matrix: np.ndarray, recombination: np.ndarray,
                             n_generations: int, founding_effective_size: float,
                             inbreeding_effective_size: float) -> np.ndarray:
    '''Operator mapping base-population average effects onto expected frequency change.

    Return the matrix L_m such that L_m @ alpha_bar is the expected
    allele-frequency change due to selection between the two measurements of a
    replicate, where alpha_bar holds the mean average effects for relative
    fitness. The replicate was founded from the offspring of one round of random
    mating in the base population; the first measurement is taken in that
    founding generation and the second n_generations generations later. From the
    base population onward, the structure carried by base_matrix is eroded by
    recombination and by drift. Drift acts at founding_effective_size in the round
    of random mating that produced the founders and at inbreeding_effective_size in
    every later round.

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

    Returns
    -------
    operator : np.ndarray
        (n_loci, n_loci) symmetric matrix.

    Raises
    ------
    ValueError
        If base_matrix or recombination is not a square 2D array, if their shapes
        disagree, if either is not symmetric within atol=1e-12, if any entry of
        recombination is outside [0, 0.5), if n_generations is not an integer >= 1,
        or if founding_effective_size or inbreeding_effective_size is not a finite
        number > 0.5.
    '''
    return operator  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_expected_change_operator(base_matrix: np.ndarray, recombination: np.ndarray,
                                     n_generations: int, founding_effective_size: float,
                                     inbreeding_effective_size: float) -> np.ndarray:
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

    # The window opens in the founding generation, one round of mating after the
    # base, so it spans t = 1 .. n_generations; the prediction N_t o L_tilde of
    # the paper's Eq. 7 holds only for t > 0. Its drift factor is a product over
    # the rounds k < t, and round k = 0 is the founding bottleneck.
    founding = 1.0 - 1.0 / (2.0 * ne0)
    drift = 1.0 - 1.0 / (2.0 * ne)
    out = np.zeros_like(L)
    for t in range(1, int(n_generations) + 1):
        out = out + ((1.0 - R) ** t) * founding * (drift ** (t - 1)) * L
    return out

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
        # --- normal: the shipped configuration, 2 generations, largest replicate ---
        {
            "setup": _SETUP,
            "call": "expected_change_operator(L, R, 2, 50.0, 2000.0)",
            "gold_call": "_oracle_expected_change_operator(L, R, 2, 50.0, 2000.0)",
        },
        # --- boundary: a single recorded generation, where only the founding round
        #     has acted, so the census size plays no part ---
        {
            "setup": _SETUP,
            "call": "expected_change_operator(L, R, 1, 50.0, 150.0)",
            "gold_call": "_oracle_expected_change_operator(L, R, 1, 50.0, 150.0)",
        },
        # --- edge: a severe founding bottleneck and many generations, where the
        #     founding factor must enter every term exactly once ---
        {
            "setup": """import numpy as np
L = np.array([[0.12, 0.05], [0.05, 0.10]])
R = np.array([[0.0, 0.02], [0.02, 0.0]])
""",
            "call": "expected_change_operator(L, R, 25, 0.75, 40.0)",
            "gold_call": "_oracle_expected_change_operator(L, R, 25, 0.75, 40.0)",
        },
    ]
