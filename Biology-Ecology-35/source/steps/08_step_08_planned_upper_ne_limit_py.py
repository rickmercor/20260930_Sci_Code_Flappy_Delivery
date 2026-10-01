"""
Expected upper confidence limit for the effective size at the smallest sample assured of showing it exceeds a threshold.

A pilot sample gives a first estimate of the effective size and, through the raw

r^2 among SNPs on the same chromosome, the drift-induced LD that makes the

unlinked r^2 values of the panel pseudo-replicated. A monitoring programme then

sizes the full survey: taking the pilot estimate as the true effective size and

keeping the panel, it finds the smallest sample that would report a lower

confidence limit above a conservation threshold with a stated probability, and

reports how far the corrected interval is expected to reach upwards at that

sample.

Returns
-------
float, the expected upper confidence limit for the effective size at the assured sample size, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def planned_upper_ne_limit(pilot_r2_mean: float, pilot_size: int, raw_same_chromosome_r2: "np.ndarray", n_snps: "np.ndarray", ne_threshold: float, assurance: float, z: float, max_sample: int) -> float:
    '''Expected upper confidence limit for the effective size at the smallest sample assured of showing it exceeds a threshold.

    The pilot sample of pilot_size diploid individuals has mean r^2
    pilot_r2_mean over all pairs of SNPs on different chromosomes and, on
    chromosome c, mean raw r^2 raw_same_chromosome_r2[c] over all pairs of its
    n_snps[c] SNPs. The pilot's point estimate of Ne (as in the estimate step)
    is taken as the true effective size, and the panel's pseudo-replication
    parameter (as in the rho step, for the pilot's own sample size) as fixed.
    The smallest assured sample is found as in the assured-sample step with
    this Ne, ne_threshold, assurance, the panel, z and max_sample. At that
    sample size the mean r^2 over unlinked pairs is set to its expectation (as
    in the expected r^2 step), and the confidence limits for the effective
    size follow as in the Ne-limits step. Return the upper limit.

    Parameters
    ----------
    pilot_r2_mean : float
        The pilot's mean r^2 over unlinked pairs, a finite number between 0
        and 1.
    pilot_size : int
        Number of diploid individuals in the pilot, an integer from 30 to
        100000.
    raw_same_chromosome_r2 : np.ndarray
        Shape (C,), C >= 2; the pilot's mean raw r^2 over pairs of SNPs on
        each chromosome, each between the pilot's sampling expectation and 1.
    n_snps : np.ndarray
        Shape (C,); integer numbers of SNPs L_c on each chromosome, each from
        2 to 10**6.
    ne_threshold : float
        Threshold the lower limit must exceed, a finite number from 5 to the
        pilot estimate.
    assurance : float
        Required probability of certifying, a finite number from 0.5 to 0.999.
    z : float
        Standard normal quantile, a finite number from 0.5 to 5.
    max_sample : int
        Largest sample size to consider, an integer from 30 to 100000.

    Returns
    -------
    upper_limit : float
        The expected upper confidence limit for the effective size at the
        assured sample size, as a native Python float.

    Raises
    ------
    ValueError
        If any argument is invalid as in the steps it is passed to, if the
        pilot's point estimate exceeds 300000, if no sample size up to
        max_sample reaches the assurance, or if the upper confidence limit at
        that sample size is not finite.
    '''
    return upper_limit  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_planned_upper_ne_limit(pilot_r2_mean: float, pilot_size: int, raw_same_chromosome_r2: "np.ndarray", n_snps: "np.ndarray", ne_threshold: float, assurance: float, z: float, max_sample: int) -> float:
    ne_hat = _oracle_waples_ne_estimate(pilot_r2_mean, pilot_size)
    rho = _oracle_pseudo_replication_rho(raw_same_chromosome_r2, n_snps, pilot_size)
    s_assured = _oracle_smallest_assured_sample(ne_hat, ne_threshold, assurance, rho, n_snps, z, max_sample)
    r2_assured = _oracle_expected_unlinked_r2(ne_hat, s_assured)
    limits = _oracle_ne_confidence_limits(r2_assured, s_assured, rho, n_snps, z)
    return float(limits[1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the pilot of 45 individuals on three chromosomes, threshold 500 at 95 per cent limits
        #     and a nine-in-ten assurance ---
        {
            "setup": "import numpy as np\nraw = np.array([0.0357, 0.0352, 0.0345])\nL = np.array([3076, 2273, 1955])\n",
            "call": "planned_upper_ne_limit(0.0242, 45, raw, L, 500.0, 0.9, 1.96, 2000)",
            "gold_call": "_oracle_planned_upper_ne_limit(0.0242, 45, raw, L, 500.0, 0.9, 1.96, 2000)",
        },
        # --- boundary: the same pilot with only chromosomes 1 and 2, the two-chromosome panel ---
        {
            "setup": "import numpy as np\nraw = np.array([0.0357, 0.0352])\nL = np.array([3076, 2273])\n",
            "call": "planned_upper_ne_limit(0.0242, 45, raw, L, 500.0, 0.9, 1.96, 2000)",
            "gold_call": "_oracle_planned_upper_ne_limit(0.0242, 45, raw, L, 500.0, 0.9, 1.96, 2000)",
        },
        # --- edge: a larger pilot on five chromosomes, a lower threshold, 90 per cent limits and an assurance
        #     of one half, where the assured sample is the one that just qualifies in expectation ---
        {
            "setup": "import numpy as np\nraw = np.array([0.0302, 0.0288, 0.0311, 0.0265, 0.0297])\nL = np.array([1800, 1500, 2200, 900, 1300])\n",
            "call": "planned_upper_ne_limit(0.0188, 60, raw, L, 250.0, 0.5, 1.6448536269514722, 2000)",
            "gold_call": "_oracle_planned_upper_ne_limit(0.0188, 60, raw, L, 250.0, 0.5, 1.6448536269514722, 2000)",
        },
    ]
