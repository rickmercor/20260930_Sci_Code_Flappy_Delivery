"""
Maximum-likelihood intercept and dispersion of a gamma age score, from the scores of newborns.

An epigenetic age score has to be tied to chronological age before it can date the births

that close-kin probabilities depend on, and animals of known age are rarely available in the

wild. Newborns are the exception: an animal caught in its first year is known to be of age

zero, so the scores of newborns fix the level of the score at birth and the size of its error.

The score is strictly positive and its error grows with its mean, so it is modelled with a

constant coefficient of variation, and the newborns' unbinned scores are used at full

precision.

Returns
-------
np.ndarray, shape (2,): the maximum-likelihood alpha and dispersion, in that order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def newborn_calibration(scores: "np.ndarray") -> "np.ndarray":
    '''Maximum-likelihood intercept and dispersion of a gamma age score, from the scores of newborns.

    Each newborn's score is an independent gamma variable with mean alpha and
    coefficient of variation sqrt(dispersion), so that its shape is
    1 / dispersion and its scale is dispersion * alpha. Return the pair
    (alpha, dispersion) that maximises the joint likelihood of the scores,
    each entry to a relative accuracy of 1e-11.

    Parameters
    ----------
    scores : np.ndarray
        Shape (K,), K >= 2; the newborns' scores, positive and finite, not all
        equal.

    Returns
    -------
    calibration : np.ndarray
        Shape (2,): the maximum-likelihood alpha and dispersion, in that order.

    Raises
    ------
    ValueError
        If scores is not a one-dimensional array of at least two positive,
        finite values, or if all its values are equal.
    '''
    return calibration  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq
from scipy.special import digamma


def _oracle_newborn_calibration(scores: "np.ndarray") -> "np.ndarray":
    x = np.asarray(scores, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)) or np.any(x <= 0.0):
        raise ValueError("scores must be a one-dimensional array of at least two positive, finite values")
    if np.all(x == x[0]):
        raise ValueError("the scores must not all be equal")
    mean = float(np.mean(x))
    gap = float(np.log(mean) - np.mean(np.log(x)))         # log of arithmetic over geometric mean, > 0
    if not gap > 0.0:
        raise ValueError("the scores are too nearly equal for a finite maximum-likelihood dispersion")

    def _shape_equation(log_shape: float) -> float:
        shape = np.exp(log_shape)
        return float(log_shape - digamma(shape) - gap)     # decreasing in the shape; its root is the MLE

    lo, hi = -30.0, 60.0
    log_shape = brentq(_shape_equation, lo, hi, xtol=1e-15, rtol=4.0 * np.finfo(float).eps, maxiter=500)
    return np.array([mean, float(np.exp(-log_shape))])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: twelve newborns with scores near 1.5 ---
        {
            "setup": "import numpy as np\n"
                     "scores = np.array([1.59, 1.54, 1.51, 1.6, 1.5, 1.69, 1.61, 1.59, 1.35, 1.26, 1.26, 1.36])\n",
            "call": "newborn_calibration(scores)",
            "gold_call": "_oracle_newborn_calibration(scores)",
        },
        # --- boundary: the smallest sample, two newborns ---
        {
            "setup": "import numpy as np\nscores = np.array([1.2, 1.8])\n",
            "call": "newborn_calibration(scores)",
            "gold_call": "_oracle_newborn_calibration(scores)",
        },
        # --- edge: a very noisy score, coefficient of variation near one ---
        {
            "setup": "import numpy as np\nscores = np.array([0.35, 2.4, 0.9, 1.6, 0.12, 3.3, 0.7])\n",
            "call": "newborn_calibration(scores)",
            "gold_call": "_oracle_newborn_calibration(scores)",
        },
    ]
