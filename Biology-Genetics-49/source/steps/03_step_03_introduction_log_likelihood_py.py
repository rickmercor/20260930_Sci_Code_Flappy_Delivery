"""
Natural logarithm of the probability of a count of fixations among independent single-copy introductions with a common fixation probability.

Establishment experiments measure fixation directly: a single mutant individual is

introduced into many independent replicate populations, and the number of

replicates in which its lineage takes over is counted. Each introduction is an

independent trial with the same fixation probability, so the count carries a

likelihood for any model that predicts that probability, and competing models of

the same population can be compared through it. This step computes the natural

logarithm of that likelihood for one strain.

Returns
-------
float, the natural-log binomial probability, as a native Python float, accurate to a relative error below 1e-12
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def introduction_log_likelihood(fixations: int, introductions: int, probability: float) -> float:
    '''Natural logarithm of the probability of a count of fixations among independent single-copy introductions with a common fixation probability.

    Each of `introductions` independent introductions fixes with probability
    `probability`. Return the natural logarithm of the binomial probability of
    exactly `fixations` fixations, the binomial coefficient included.

    Parameters
    ----------
    fixations : int
        Number of introductions that fixed, an integer from 0 to introductions.
    introductions : int
        Number of independent introductions, an integer >= 1.
    probability : float
        Fixation probability of one introduction, a finite number with
        0 < probability < 1.

    Returns
    -------
    log_likelihood : float
        The natural-log binomial probability, as a native Python float,
        accurate to a relative error below 1e-12.

    Raises
    ------
    ValueError
        If introductions is not an integer >= 1, if fixations is not an
        integer from 0 to introductions, or if probability is not a finite
        number with 0 < probability < 1.
    '''
    return log_likelihood  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np
from scipy import special


def _check_count(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer, float, np.floating)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    v = float(value)
    if not np.isfinite(v) or v != math.floor(v) or v < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(v)


def _oracle_introduction_log_likelihood(fixations: int, introductions: int, probability: float) -> float:
    n = _check_count("introductions", introductions, 1)
    k = _check_count("fixations", fixations, 0)
    if k > n:
        raise ValueError("fixations must be an integer from 0 to introductions")
    if isinstance(probability, bool) or not isinstance(probability, (int, float, np.integer, np.floating)):
        raise ValueError("probability must be a finite number with 0 < probability < 1")
    q = float(probability)
    if not np.isfinite(q) or not 0.0 < q < 1.0:
        raise ValueError("probability must be a finite number with 0 < probability < 1")
    log_choose = special.gammaln(n + 1.0) - special.gammaln(k + 1.0) - special.gammaln(n - k + 1.0)
    return float(log_choose + k * math.log(q) + (n - k) * math.log1p(-q))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: hundreds of fixations among two hundred thousand introductions ---
        {
            "setup": "import numpy as np\n",
            "call": "introduction_log_likelihood(888, 200000, 0.0045429111)",
            "gold_call": "_oracle_introduction_log_likelihood(888, 200000, 0.0045429111)",
        },
        # --- boundary: no fixation at all ---
        {
            "setup": "import numpy as np\n",
            "call": "introduction_log_likelihood(0, 50, 0.1)",
            "gold_call": "_oracle_introduction_log_likelihood(0, 50, 0.1)",
        },
        # --- edge: every introduction fixed ---
        {
            "setup": "import numpy as np\n",
            "call": "introduction_log_likelihood(4, 4, 0.75)",
            "gold_call": "_oracle_introduction_log_likelihood(4, 4, 0.75)",
        },
    ]
