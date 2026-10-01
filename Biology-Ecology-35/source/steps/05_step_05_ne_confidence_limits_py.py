"""
Two-sided confidence limits for the effective population size from a mean r^2 over unlinked pairs, allowing for pseudo-replication.

A confidence interval for the expected mean unlinked r^2 becomes a confidence

interval for the effective size by passing each limit through the same relation

that turns an observed mean into an estimate. Because a larger r^2 means stronger

drift and so a smaller population, the interval reverses under the conversion,

and when the lower r^2 limit falls to the sampling expectation or below it the

data no longer bound the effective size from above.

Returns
-------
np.ndarray, shape (2,), float; the lower and the upper confidence limit for the effective population size, in that order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ne_confidence_limits(r2_mean: float, sample_size: int, rho: float, n_snps: "np.ndarray", z: float) -> "np.ndarray":
    '''Two-sided confidence limits for the effective population size from a mean r^2 over unlinked pairs, allowing for pseudo-replication.

    The confidence limits for the expected mean unlinked r^2 are those of the
    source's interval for the given panel and pseudo-replication parameter (as
    in the r^2 limits step), and each is converted to an effective size with
    Waples' relations for S diploid individuals (as in the estimate step): the
    sampling expectation is 1/S + 3.19/S^2 and
    Ne = (1/3 + sqrt(1/9 - 2.76 x)) / (2 x) for the drift part x of the limit.

    Parameters
    ----------
    r2_mean : float
        Observed mean r^2 over all pairs of SNPs on different chromosomes, a
        finite number between 0 and 1.
    sample_size : int
        Number of diploid individuals S, an integer from 30 to 100000.
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
        effective population size, in that order.

    Raises
    ------
    ValueError
        If any argument is invalid as in the r^2 limits step or the estimate
        step, or if either limit is not a finite effective size: the drift
        part of the lower r^2 limit must be at least 1e-6, and the drift part
        of the upper r^2 limit at most 1/(9 * 2.76).
    '''
    return limits  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_ne_confidence_limits(r2_mean: float, sample_size: int, rho: float, n_snps: "np.ndarray", z: float) -> "np.ndarray":
    r2_limits = _oracle_r2_confidence_limits(r2_mean, rho, n_snps, z)
    ne_low = _oracle_waples_ne_estimate(float(r2_limits[1]), sample_size)    # the upper r^2 limit bounds Ne from below
    ne_high = _oracle_waples_ne_estimate(float(r2_limits[0]), sample_size)
    return np.array([ne_low, ne_high])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: three chromosomes and 81 individuals, the mean r^2 expected for Ne = 826 ---
        {
            "setup": "import numpy as np\nL = np.array([3076, 2273, 1955])\ns = 81\n"
                     "r2 = 1 / s + 3.19 / s ** 2 + 1 / (3 * 826.0) - 0.69 / 826.0 ** 2\n",
            "call": "ne_confidence_limits(r2, s, 419.2392200655806, L, 1.96)",
            "gold_call": "_oracle_ne_confidence_limits(r2, s, 419.2392200655806, L, 1.96)",
        },
        # --- boundary: no pseudo-replication, so only the number of unlinked pairs sets the width ---
        {
            "setup": "import numpy as np\nL = np.array([1200, 900])\n",
            "call": "ne_confidence_limits(0.0381, 30, 0.0, L, 1.96)",
            "gold_call": "_oracle_ne_confidence_limits(0.0381, 30, 0.0, L, 1.96)",
        },
        # --- edge: a small, strongly drifting population with a sparse, strongly linked panel at 99 per cent ---
        {
            "setup": "import numpy as np\nL = np.array([60, 45, 30])\n",
            "call": "ne_confidence_limits(0.0152, 200, 24.6, L, 2.5758293035489)",
            "gold_call": "_oracle_ne_confidence_limits(0.0152, 200, 24.6, L, 2.5758293035489)",
        },
    ]
