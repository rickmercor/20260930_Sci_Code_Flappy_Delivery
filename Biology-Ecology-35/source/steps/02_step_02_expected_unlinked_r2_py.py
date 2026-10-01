"""
Expected mean r^2 over unlinked pairs of loci for a population of effective size Ne sampled with S diploid individuals.

Planning a genetic monitoring survey starts from the reverse question: for a

population of a given effective size, what mean r^2 over unlinked pairs of loci

should a sample of a given number of individuals show? The expectation adds the

sampling part, which shrinks as the sample grows, to the drift part, which is set

by the effective size alone, using the same empirical relations of Waples (2006)

that turn an observed mean into an estimate.

Returns
-------
float, the expected mean r^2 over unlinked pairs, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def expected_unlinked_r2(ne: float, sample_size: int) -> float:
    '''Expected mean r^2 over unlinked pairs of loci for a population of effective size Ne sampled with S diploid individuals.

    The expectation is the sampling part 1/S + 3.19/S^2 plus the drift part
    1/(3 Ne) - 0.69/Ne^2, so that Waples' estimator applied to the result
    returns Ne.

    Parameters
    ----------
    ne : float
        Effective population size, a finite number from 5 to 300000.
    sample_size : int
        Number of diploid individuals S, an integer from 30 to 100000.

    Returns
    -------
    r2 : float
        The expected mean r^2 over unlinked pairs, as a native Python float.

    Raises
    ------
    ValueError
        If ne is not a finite number from 5 to 300000, or if sample_size is
        not an integer from 30 to 100000.
    '''
    return r2  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


import math


def _oracle_expected_unlinked_r2(ne: float, sample_size: int) -> float:
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or not 30 <= sample_size <= 100000:
        raise ValueError("sample_size must be an integer from 30 to 100000")
    n = float(ne)
    if not math.isfinite(n) or not 5.0 <= n <= 300000.0:
        raise ValueError("ne must be a finite number from 5 to 300000")
    s = float(sample_size)
    return float(1.0 / s + 3.19 / (s * s) + 1.0 / (3.0 * n) - 0.69 / (n * n))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: a planning value of Ne = 826 sampled with 81 individuals ---
        {
            "setup": "",
            "call": "expected_unlinked_r2(826.0, 81)",
            "gold_call": "_oracle_expected_unlinked_r2(826.0, 81)",
        },
        # --- boundary: the smallest sample the relations cover, a small population where the second-order term matters ---
        {
            "setup": "",
            "call": "expected_unlinked_r2(12.5, 30)",
            "gold_call": "_oracle_expected_unlinked_r2(12.5, 30)",
        },
        # --- edge: a large population and a large sample, where both parts are small ---
        {
            "setup": "",
            "call": "expected_unlinked_r2(250000.0, 20000)",
            "gold_call": "_oracle_expected_unlinked_r2(250000.0, 20000)",
        },
    ]
