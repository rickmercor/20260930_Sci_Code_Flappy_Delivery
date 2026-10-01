"""
Probability that a future sample shows the effective population size to exceed a threshold.

A survey that is planned on expectations alone succeeds only about half the time,

because the mean r^2 it will observe is itself a random quantity: the source

draws the sampling distribution of the estimate under a normal approximation

whose spread is set by the same pseudo-replication that widens the confidence

interval. A monitoring programme that has to report whether a population is above

a conservation threshold therefore needs the chance that the survey it is about

to run will return a lower confidence limit above that threshold.

Returns
-------
float, the probability that the future sample certifies the population, a native Python float between 0 and 1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def certification_probability(ne: float, ne_threshold: float, sample_size: int, rho: float, n_snps: "np.ndarray", z: float) -> float:
    '''Probability that a future sample shows the effective population size to exceed a threshold.

    A future sample of S diploid individuals is genotyped at the given panel,
    and the corrected two-sided confidence limits for the effective size are
    computed from its own mean r^2 over the panel's unlinked pairs, at the
    same sample size, rho and z (as in the Ne limits step). The sample
    certifies the population when that lower limit is strictly greater than
    ne_threshold. A sample whose upper confidence limit for the expected mean
    r^2 has a drift part at or below zero leaves the effective size unbounded
    from above and so certifies; one whose upper limit has a drift part above
    1/(9 * 2.76), where Waples' relation has no solution, does not.

    The population's effective size is ne. Under the source's model the future
    sample's mean r^2 over the unlinked pairs is normally distributed, with
    mean the expected mean unlinked r^2 for ne and S (as in the expected r^2
    step) and with standard deviation that expectation times
    sqrt(2 (1 + 2 rho) / N), where N is the panel's number of unlinked pairs.
    Return the probability, under that distribution of the sample's mean r^2,
    that the sample certifies the population.

    Parameters
    ----------
    ne : float
        The population's effective size, a finite number from 5 to 300000.
    ne_threshold : float
        Threshold the lower limit must exceed, a finite number from 5 to ne.
    sample_size : int
        Number of diploid individuals S in the future sample, an integer from
        30 to 100000.
    rho : float
        The panel's pseudo-replication parameter, a finite number that is not
        negative.
    n_snps : np.ndarray
        Shape (C,), C >= 2; integer numbers of SNPs L_c on each chromosome,
        each from 2 to 10**6.
    z : float
        Standard normal quantile of the confidence limits, a finite number
        from 0.5 to 5.

    Returns
    -------
    probability : float
        The probability that the future sample certifies the population, a
        native Python float between 0 and 1.

    Raises
    ------
    ValueError
        If ne, ne_threshold, sample_size or z are outside their ranges, if rho
        or n_snps are invalid as for the r^2 limits step, or if
        z sqrt(2 (1 + 2 rho) / N) is 1 or more, where the upper confidence
        limit carries no information.
    '''
    return probability  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np
from scipy import stats


def _oracle_certification_probability(ne: float, ne_threshold: float, sample_size: int, rho: float, n_snps: "np.ndarray", z: float) -> float:
    n = float(ne)
    if not math.isfinite(n) or not 5.0 <= n <= 300000.0:
        raise ValueError("ne must be a finite number from 5 to 300000")
    t = float(ne_threshold)
    if not math.isfinite(t) or not 5.0 <= t <= n:
        raise ValueError("ne_threshold must be a finite number from 5 to ne")
    if isinstance(sample_size, bool) or not isinstance(sample_size, (int, np.integer)) \
            or not 30 <= int(sample_size) <= 100000:
        raise ValueError("sample_size must be an integer from 30 to 100000")
    s = int(sample_size)
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
    relative_sd = math.sqrt(2.0 * (1.0 + 2.0 * r) / total_pairs)
    half_width = zz * relative_sd
    if half_width >= 1.0:
        raise ValueError("z sqrt(2 (1 + 2 rho) / N) must be below 1")
    mean = _oracle_expected_unlinked_r2(n, s)
    # Waples' estimate falls as r^2 rises, so the lower limit clears the threshold exactly while the sample's
    # own mean stays below the mean expected at the threshold, shrunk by the interval's factor 1 - w
    critical = (1.0 - half_width) * _oracle_expected_unlinked_r2(t, s)
    return float(stats.norm.cdf((critical - mean) / (mean * relative_sd)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: a repeat of the pilot's own design on the pilot's panel ---
        {
            "setup": "import numpy as np\nL = np.array([3076, 2273, 1955])\n",
            "call": "certification_probability(826.1457, 500.0, 45, 419.2392200655806, L, 1.96)",
            "gold_call": "_oracle_certification_probability(826.1457, 500.0, 45, 419.2392200655806, L, 1.96)",
        },
        # --- normal: the sample whose expected lower limit just clears the threshold ---
        {
            "setup": "import numpy as np\nL = np.array([3076, 2273, 1955])\n",
            "call": "certification_probability(826.1457, 500.0, 81, 419.2392200655806, L, 1.96)",
            "gold_call": "_oracle_certification_probability(826.1457, 500.0, 81, 419.2392200655806, L, 1.96)",
        },
        # --- boundary: no pseudo-replication and a sample far larger than the threshold requires ---
        {
            "setup": "import numpy as np\nL = np.array([5368, 3552])\n",
            "call": "certification_probability(1258.0, 500.0, 400, 0.0, L, 1.96)",
            "gold_call": "_oracle_certification_probability(1258.0, 500.0, 400, 0.0, L, 1.96)",
        },
        # --- edge: a threshold equal to the population's own size, at the smallest allowed sample and a wide z,
        #     where the interval alone makes certification very unlikely ---
        {
            "setup": "import numpy as np\nL = np.array([400, 300, 250])\n",
            "call": "certification_probability(120.0, 120.0, 30, 3800.0, L, 2.5)",
            "gold_call": "_oracle_certification_probability(120.0, 120.0, 30, 3800.0, L, 2.5)",
        },
    ]
