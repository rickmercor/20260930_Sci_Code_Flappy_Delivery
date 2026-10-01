"""
Effective population size from a mean r^2 over unlinked pairs of loci, by Waples' relations for 30 or more individuals.

The linkage-disequilibrium method estimates the contemporary effective size of a

population from the mean squared correlation r^2 between alleles at pairs of

unlinked loci in a sample of diploid individuals. Part of that mean is produced by

sampling alone and part by genetic drift in the population; the sampling part is

removed with an empirical expectation that depends only on the sample size, and

the remaining drift part is converted to an effective size with an empirical

relation calibrated by simulation. For samples of at least 30 individuals the

standard relations are those of Waples (2006), used here as stated in the

docstring.

Returns
-------
float, the estimated effective population size, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def waples_ne_estimate(r2_mean: float, sample_size: int) -> float:
    '''Effective population size from a mean r^2 over unlinked pairs of loci, by Waples' relations for 30 or more individuals.

    The expected r^2 from sampling alone in a sample of S diploid
    individuals is 1/S + 3.19/S^2. The drift part x of a mean r^2 is the
    mean minus that expectation, and the effective size is
    Ne = (1/3 + sqrt(1/9 - 2.76 x)) / (2 x), which is the root of
    x = 1/(3 Ne) - 0.69/Ne^2 on the branch that grows as x falls.

    Parameters
    ----------
    r2_mean : float
        Mean r^2 over pairs of unlinked loci, between 0 and 1.
    sample_size : int
        Number of diploid individuals S, an integer from 30 to 100000.

    Returns
    -------
    ne : float
        The estimated effective population size, as a native Python float.

    Raises
    ------
    ValueError
        If sample_size is not an integer from 30 to 100000, if r2_mean is
        not a finite number between 0 and 1, if the drift part x is below
        1e-6 (the estimate would be larger than about 333,000 or infinite),
        or if x exceeds 1/(9 * 2.76), where the relation has no real
        solution.
    '''
    return ne  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


import math


def _oracle_waples_ne_estimate(r2_mean: float, sample_size: int) -> float:
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or not 30 <= sample_size <= 100000:
        raise ValueError("sample_size must be an integer from 30 to 100000")
    r2 = float(r2_mean)
    if not math.isfinite(r2) or not 0.0 <= r2 <= 1.0:
        raise ValueError("r2_mean must be a finite number between 0 and 1")
    s = float(sample_size)
    x = r2 - (1.0 / s + 3.19 / (s * s))
    if x < 1e-6:
        raise ValueError("the drift part of r2_mean must be at least 1e-6")
    disc = 1.0 / 9.0 - 2.76 * x
    if disc < 0.0:
        raise ValueError("the drift part of r2_mean exceeds 1/(9 * 2.76)")
    return float((1.0 / 3.0 + math.sqrt(disc)) / (2.0 * x))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: a pilot of 45 individuals with a mean unlinked r^2 of 0.0242 ---
        {
            "setup": "",
            "call": "waples_ne_estimate(0.0242, 45)",
            "gold_call": "_oracle_waples_ne_estimate(0.0242, 45)",
        },
        # --- boundary: the smallest sample the relations cover, a drift part of 2e-5 (a large population) ---
        {
            "setup": "s = 30\nr2 = 1.0 / s + 3.19 / s ** 2 + 2e-5\n",
            "call": "waples_ne_estimate(r2, s)",
            "gold_call": "_oracle_waples_ne_estimate(r2, s)",
        },
        # --- edge: a strongly drifting population, drift part 0.03, close to where the relation has no solution ---
        {
            "setup": "s = 250\nr2 = 1.0 / s + 3.19 / s ** 2 + 0.03\n",
            "call": "waples_ne_estimate(r2, s)",
            "gold_call": "_oracle_waples_ne_estimate(r2, s)",
        },
    ]
