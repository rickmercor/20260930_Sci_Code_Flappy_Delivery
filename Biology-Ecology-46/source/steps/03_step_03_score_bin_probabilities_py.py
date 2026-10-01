"""
Probability that the age score of an animal of each age 0..max_age falls in the bin (lower, upper].

The age score is read with an error that grows with the animal's age, as band counts and

methylation scores do, and it is recorded only as the bin of the score scale into which it

falls. For each possible age, what enters the calibration is the probability that a score

falls in the recorded bin under the score's error distribution.

Returns
-------
np.ndarray, shape (max_age + 1,): the bin probability at each age 0, 1, ..., max_age
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def score_bin_probabilities(lower: float, upper: float, alpha: float, beta: float, dispersion: float, max_age: int) -> "np.ndarray":
    '''Probability that the age score of an animal of each age 0..max_age falls in the bin (lower, upper].

    The score of an animal aged a whole years is gamma distributed with mean
    alpha + beta * a and coefficient of variation sqrt(dispersion), the same
    at every age. Return, for a = 0, 1, ..., max_age, the probability that
    the score lies in (lower, upper].

    Parameters
    ----------
    lower : float
        Lower bin edge, finite, 0 <= lower < upper.
    upper : float
        Upper bin edge, finite.
    alpha : float
        Mean score at age 0, positive and finite.
    beta : float
        Increase of the mean score per year of age, positive and finite.
    dispersion : float
        Squared coefficient of variation of the score, positive and finite.
    max_age : int
        Oldest age, an integer from 0 to 400.

    Returns
    -------
    probabilities : np.ndarray
        Shape (max_age + 1,): the bin probability at each age 0, 1, ..., max_age.

    Raises
    ------
    ValueError
        If the edges are not finite with 0 <= lower < upper, if alpha, beta or
        dispersion is not positive and finite, or if max_age is not an integer
        from 0 to 400.
    '''
    return probabilities  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import gammainc, gammaincc


def _oracle_score_bin_probabilities(lower: float, upper: float, alpha: float, beta: float, dispersion: float, max_age: int) -> "np.ndarray":
    if not (np.isfinite(lower) and np.isfinite(upper) and 0.0 <= lower < upper):
        raise ValueError("the bin edges must be finite with 0 <= lower < upper")
    for v in (alpha, beta, dispersion):
        if not (np.isfinite(v) and v > 0.0):
            raise ValueError("alpha, beta and dispersion must be positive and finite")
    if isinstance(max_age, bool) or not isinstance(max_age, (int, np.integer)) or not 0 <= max_age <= 400:
        raise ValueError("max_age must be an integer from 0 to 400")
    shape = 1.0 / dispersion
    scale = dispersion * (alpha + beta * np.arange(max_age + 1, dtype=float))
    x_lo, x_hi = lower / scale, upper / scale
    lower_side = gammainc(shape, x_hi) - gammainc(shape, x_lo)
    upper_side = gammaincc(shape, x_lo) - gammaincc(shape, x_hi)   # the same mass, without cancellation above the mode
    return np.where(x_lo > shape, upper_side, lower_side)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: a young-animal bin under a precise score ---
        {
            "setup": "",
            "call": "score_bin_probabilities(3.0, 5.0, 1.49, 0.82, 0.009, 150)",
            "gold_call": "_oracle_score_bin_probabilities(3.0, 5.0, 1.49, 0.82, 0.009, 150)",
        },
        # --- boundary: the bin starts at zero, so it holds every score up to its upper edge ---
        {
            "setup": "",
            "call": "score_bin_probabilities(0.0, 2.0, 1.5, 0.8, 0.01, 20)",
            "gold_call": "_oracle_score_bin_probabilities(0.0, 2.0, 1.5, 0.8, 0.01, 20)",
        },
        # --- edge: a very noisy score and a wide bin far above the mean at birth ---
        {
            "setup": "",
            "call": "score_bin_probabilities(30.0, 40.0, 1.0, 1.2, 0.4, 60)",
            "gold_call": "_oracle_score_bin_probabilities(30.0, 40.0, 1.0, 1.2, 0.4, 60)",
        },
    ]
