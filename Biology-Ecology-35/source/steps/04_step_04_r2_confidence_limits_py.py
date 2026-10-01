"""
Two-sided confidence limits for the expected mean r^2 over unlinked pairs, allowing for pseudo-replication.

The classical confidence interval for the linkage-disequilibrium estimate of

effective size treats the r^2 values of the unlinked pairs as independent, which

makes it far too narrow for genomic panels. The source replaces it with an

interval that allows for pseudo-replication through its parameter rho, and it

builds that interval for r^2 first; converting it to effective size is a separate

step.

Returns
-------
np.ndarray, shape (2,), float; the lower and the upper confidence limit for the expected mean r^2, in that order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def r2_confidence_limits(r2_mean: float, rho: float, n_snps: "np.ndarray", z: float) -> "np.ndarray":
    '''Two-sided confidence limits for the expected mean r^2 over unlinked pairs, allowing for pseudo-replication.

    The observed mean r2_mean is taken over all N = sum over a < b of
    L_a L_b pairs of SNPs on different chromosomes, and rho is the source's
    pseudo-replication parameter of that mean for the given panel. The source
    takes r2_mean / E(r^2) to have mean 1 and variance 2 (1 + 2 rho) / N, and
    inverts 1 - w <= r2_mean / E(r^2) <= 1 + w, where
    w = z sqrt(2 (1 + 2 rho) / N): the lower limit is r2_mean / (1 + w), and
    the upper limit is r2_mean / (1 - w), or infinite when w >= 1. Each limit
    is clamped to the range 0 to 1 of r^2.

    Parameters
    ----------
    r2_mean : float
        Observed mean r^2 over the N unlinked pairs, a finite number between
        0 and 1, not 0.
    rho : float
        The panel's pseudo-replication parameter, a finite number that is not
        negative.
    n_snps : np.ndarray
        Shape (C,), C >= 2; integer numbers of SNPs L_c on each chromosome,
        each from 2 to 10**6.
    z : float
        Standard normal quantile, a finite number from 0.5 to 5.

    Returns
    -------
    limits : np.ndarray
        Shape (2,), float; the lower and the upper confidence limit for the
        expected mean r^2, in that order.

    Raises
    ------
    ValueError
        If r2_mean is not a finite number in (0, 1], if rho is not a finite
        number that is at least 0, if z is not a finite number from 0.5 to 5,
        or if n_snps is not a one-dimensional array of at least two integers
        from 2 to 10**6.
    '''
    return limits  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np


def _oracle_r2_confidence_limits(r2_mean: float, rho: float, n_snps: "np.ndarray", z: float) -> "np.ndarray":
    r2 = float(r2_mean)
    if not math.isfinite(r2) or not 0.0 < r2 <= 1.0:
        raise ValueError("r2_mean must be a finite number in (0, 1]")
    r = float(rho)
    if not math.isfinite(r) or r < 0.0:
        raise ValueError("rho must be a finite number that is at least 0")
    zz = float(z)
    if not math.isfinite(zz) or not 0.5 <= zz <= 5.0:
        raise ValueError("z must be a finite number from 0.5 to 5")
    L = np.asarray(n_snps)
    if L.ndim != 1 or L.size < 2 or not np.issubdtype(L.dtype, np.integer) \
            or np.any(L < 2) or np.any(L > 10 ** 6):
        raise ValueError("n_snps must be a 1-D array of at least two integers from 2 to 10**6")
    Lf = L.astype(float)
    total_pairs = float(sum(Lf[a] * Lf[b] for a in range(Lf.size) for b in range(a + 1, Lf.size)))
    half_width = zz * math.sqrt(2.0 * (1.0 + 2.0 * r) / total_pairs)
    upper = r2 / (1.0 - half_width) if half_width < 1.0 else math.inf
    return np.array([min(max(r2 / (1.0 + half_width), 0.0), 1.0), min(max(upper, 0.0), 1.0)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the pilot's mean over three chromosomes at 95 per cent ---
        {
            "setup": "import numpy as np\nL = np.array([3076, 2273, 1955])\n",
            "call": "r2_confidence_limits(0.0242, 419.2392200655806, L, 1.96)",
            "gold_call": "_oracle_r2_confidence_limits(0.0242, 419.2392200655806, L, 1.96)",
        },
        # --- boundary: no pseudo-replication at all on a two-chromosome panel ---
        {
            "setup": "import numpy as np\nL = np.array([5368, 3552])\n",
            "call": "r2_confidence_limits(0.01566, 0.0, L, 1.96)",
            "gold_call": "_oracle_r2_confidence_limits(0.01566, 0.0, L, 1.96)",
        },
        # --- edge: a tiny panel whose half-width exceeds 1, so the upper limit is infinite before clamping ---
        {
            "setup": "import numpy as np\nL = np.array([2, 2])\n",
            "call": "r2_confidence_limits(0.08, 1.5, L, 5.0)",
            "gold_call": "_oracle_r2_confidence_limits(0.08, 1.5, L, 5.0)",
        },
        # --- edge: a strongly pseudo-replicated panel at a wide z, the lower limit still positive ---
        {
            "setup": "import numpy as np\nL = np.array([400, 300, 250])\n",
            "call": "r2_confidence_limits(0.0402, 3804.0, L, 2.5)",
            "gold_call": "_oracle_r2_confidence_limits(0.0402, 3804.0, L, 2.5)",
        },
    ]
