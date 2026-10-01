"""
Smallest number of diploid individuals whose probability of showing the effective size to exceed a threshold reaches an assurance level.

Planning a survey means choosing how many individuals to genotype before any of

them are collected. Because the mean unlinked r^2 the survey will return is a

draw rather than its expectation, a sample size that clears a conservation

threshold on average leaves the programme with roughly even odds of reporting a

lower confidence limit above it. Sizing the survey to a stated chance of success

instead asks for the smallest sample whose probability of certifying the

population reaches that level.

Returns
-------
int, the smallest qualifying S, as a native Python int
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def smallest_assured_sample(ne: float, ne_threshold: float, assurance: float, rho: float, n_snps: "np.ndarray", z: float, max_sample: int) -> int:
    '''Smallest number of diploid individuals whose probability of showing the effective size to exceed a threshold reaches an assurance level.

    For each integer S from 30 upwards, take the probability that a future
    sample of S individuals genotyped at the given panel certifies a
    population of effective size ne against ne_threshold, as the
    certification-probability step defines it for the same rho and z. Return
    the first S at which that probability is greater than or equal to
    assurance.

    Parameters
    ----------
    ne : float
        The population's effective size, a finite number from 5 to 300000.
    ne_threshold : float
        Threshold the lower limit must exceed, a finite number from 5 to ne.
    assurance : float
        Required probability of certifying, a finite number from 0.5 to 0.999.
    rho : float
        The panel's pseudo-replication parameter, a finite number that is not
        negative.
    n_snps : np.ndarray
        Shape (C,), C >= 2; integer numbers of SNPs L_c on each chromosome,
        each from 2 to 10**6.
    z : float
        Standard normal quantile of the confidence limits, a finite number
        from 0.5 to 5.
    max_sample : int
        Largest sample size to consider, an integer from 30 to 100000.

    Returns
    -------
    sample_size : int
        The smallest qualifying S, as a native Python int.

    Raises
    ------
    ValueError
        If assurance or max_sample are outside their ranges, if any other
        argument is invalid as for the certification-probability step, or if
        no S up to max_sample qualifies.
    '''
    return sample_size  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


import math


def _oracle_smallest_assured_sample(ne: float, ne_threshold: float, assurance: float, rho: float, n_snps: "np.ndarray", z: float, max_sample: int) -> int:
    a = float(assurance)
    if not math.isfinite(a) or not 0.5 <= a <= 0.999:
        raise ValueError("assurance must be a finite number from 0.5 to 0.999")
    if isinstance(max_sample, bool) or not isinstance(max_sample, int) or not 30 <= max_sample <= 100000:
        raise ValueError("max_sample must be an integer from 30 to 100000")
    for s in range(30, max_sample + 1):
        if _oracle_certification_probability(ne, ne_threshold, s, rho, n_snps, z) >= a:
            return s
    raise ValueError("no sample size up to max_sample qualifies")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the pilot's panel and population, sized for a nine-in-ten chance of certifying ---
        {
            "setup": "import numpy as np\nL = np.array([3076, 2273, 1955])\n",
            "call": "smallest_assured_sample(826.1457, 500.0, 0.9, 419.2392200655806, L, 1.96, 2000)",
            "gold_call": "_oracle_smallest_assured_sample(826.1457, 500.0, 0.9, 419.2392200655806, L, 1.96, 2000)",
        },
        # --- boundary: an assurance of exactly one half, the weakest the contract allows ---
        {
            "setup": "import numpy as np\nL = np.array([3076, 2273, 1955])\n",
            "call": "smallest_assured_sample(826.1457, 500.0, 0.5, 419.2392200655806, L, 1.96, 2000)",
            "gold_call": "_oracle_smallest_assured_sample(826.1457, 500.0, 0.5, 419.2392200655806, L, 1.96, 2000)",
        },
        # --- edge: a two-chromosome panel with no pseudo-replication at a demanding assurance ---
        {
            "setup": "import numpy as np\nL = np.array([5368, 3552])\n",
            "call": "smallest_assured_sample(1258.0, 500.0, 0.99, 0.0, L, 1.96, 2000)",
            "gold_call": "_oracle_smallest_assured_sample(1258.0, 500.0, 0.99, 0.0, L, 1.96, 2000)",
        },
        # --- edge: a small, strongly pseudo-replicated panel and a threshold well below the population ---
        {
            "setup": "import numpy as np\nL = np.array([400, 300, 250])\n",
            "call": "smallest_assured_sample(120.0, 50.0, 0.95, 1500.0, L, 2.5, 2000)",
            "gold_call": "_oracle_smallest_assured_sample(120.0, 50.0, 0.95, 1500.0, L, 2.5, 2000)",
        },
    ]
