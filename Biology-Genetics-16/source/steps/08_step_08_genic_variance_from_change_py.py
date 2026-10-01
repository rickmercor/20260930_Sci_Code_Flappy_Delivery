"""
'''Additive GENIC variance for relative fitness, assuming linkage equilibrium.




    Sum the squared expected per-locus allele-frequency change divided by the

    corresponding per-locus entry on the diagonal of the diversity matrix. Loci

    whose diagonal entry is zero are monomorphic and contribute nothing; they must

    be skipped rather than producing a division by zero.




    Parameters

    ----------

    delta_p : np.ndarray

        (n_loci,) expected allele-frequency change attributable to selection.

    diversity_matrix : np.ndarray

        (n_loci, n_loci) base-population diversity matrix; only its diagonal is

        used here.




    Returns

    -------

    genic_variance : float

        Native Python float, >= 0.




    Raises

    ------

    ValueError

        If delta_p is not a finite 1D array, if diversity_matrix is not a square

        2D array whose size matches delta_p, if any diagonal entry of

        diversity_matrix is negative, or if a locus has a zero diagonal entry but

        a non-zero delta_p.

    '''

Two additive variances for relative fitness have to be distinguished, and the

distinction is the whole reason a matrix appears in this analysis at all.




The additive GENIC variance is the contribution of the per-locus diversities

alone; it is what you obtain by treating each locus in isolation, so under an

assumption of linkage equilibrium it is recovered from expected frequency change

by summing the squared per-locus change divided by the per-locus diversity. The

additive GENETIC variance, which is the quantity Fisher's theorem equates with the

rate of adaptation, additionally contains every contribution from disequilibrium

between loci, and the two coincide only when signed disequilibrium among selected

loci is zero.




This step computes the genic variance, which is reported alongside the genetic

variance so that the size of the disequilibrium contribution is visible. It is a

diagnostic, not the target quantity.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def genic_variance_from_change(delta_p: np.ndarray,
                               diversity_matrix: np.ndarray) -> float:
    '''Additive GENIC variance for relative fitness, assuming linkage equilibrium.

    Sum the squared expected per-locus allele-frequency change divided by the
    corresponding per-locus entry on the diagonal of the diversity matrix. Loci
    whose diagonal entry is zero are monomorphic and contribute nothing; they must
    be skipped rather than producing a division by zero.

    Parameters
    ----------
    delta_p : np.ndarray
        (n_loci,) expected allele-frequency change attributable to selection.
    diversity_matrix : np.ndarray
        (n_loci, n_loci) base-population diversity matrix; only its diagonal is
        used here.

    Returns
    -------
    genic_variance : float
        Native Python float, >= 0.

    Raises
    ------
    ValueError
        If delta_p is not a finite 1D array, if diversity_matrix is not a square
        2D array whose size matches delta_p, if any diagonal entry of
        diversity_matrix is negative, or if a locus has a zero diagonal entry but
        a non-zero delta_p.
    '''
    return genic_variance  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_genic_variance_from_change(delta_p: np.ndarray,
                                       diversity_matrix: np.ndarray) -> float:
    dp = np.asarray(delta_p, dtype=float)
    L = np.asarray(diversity_matrix, dtype=float)
    if dp.ndim != 1 or dp.size < 1:
        raise ValueError("delta_p must be a non-empty 1D array")
    if not np.all(np.isfinite(dp)):
        raise ValueError("delta_p must be finite")
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("diversity_matrix must be a square 2D array")
    if L.shape[0] != dp.size:
        raise ValueError("diversity_matrix size must match delta_p")
    d = np.diag(L)
    if np.any(d < 0.0):
        raise ValueError("diagonal entries of diversity_matrix must be >= 0")
    zero = d == 0.0
    if np.any(zero & (dp != 0.0)):
        raise ValueError("a monomorphic locus cannot have a non-zero delta_p")
    keep = ~zero
    return float(np.sum(dp[keep] ** 2 / d[keep]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: replicate-mean change against the shipped diversity matrix ---
        {
            "setup": """import numpy as np
dp = np.array([-0.082062, -0.103579, -0.097155, -0.076363, -0.043564, 0.011792, 0.034409, 0.081252, 0.069469, 0.064476])
d = np.array([0.066508, 0.092087, 0.108340, 0.123244, 0.128911, 0.125736, 0.124076, 0.107344, 0.086860, 0.062986])
L = np.diag(d) + 0.01 * (np.ones((10, 10)) - np.eye(10))
""",
            "call": "genic_variance_from_change(dp, L)",
            "gold_call": "_oracle_genic_variance_from_change(dp, L)",
        },
        # --- boundary: no change at all -> exactly zero ---
        {
            "setup": """import numpy as np
dp = np.zeros(3)
L = np.diag([0.1, 0.2, 0.05])
""",
            "call": "genic_variance_from_change(dp, L)",
            "gold_call": "_oracle_genic_variance_from_change(dp, L)",
        },
        # --- edge: a monomorphic locus (zero diagonal, zero change) is skipped ---
        {
            "setup": """import numpy as np
dp = np.array([0.02, 0.0, -0.01])
L = np.diag([0.1, 0.0, 0.05])
""",
            "call": "genic_variance_from_change(dp, L)",
            "gold_call": "_oracle_genic_variance_from_change(dp, L)",
        },
    ]
