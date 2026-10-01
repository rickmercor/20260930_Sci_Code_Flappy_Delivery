"""
Return the source's specialization-averaged weight of every given candidate prey pair of one predator (the pairs of step 03 at sigma = sigma_max, with their thresholds of step 04, in the same order) when the trophic specialization is unknown: the source's Bayesian weight of a pair at specialization sigma - its prior 1 / n(sigma), uniform over the n(sigma) pairs that are candidates at sigma, times its likelihood, the product of the Gaussian densities of mean zero and standard deviation sigma at the two prey offsets c - y_lower and y_upper - c, that is exp(-((c - y_lower)^2 + (y_upper - c)^2) / (2 sigma^2)) / (2 pi sigma^2) - averaged over sigma uniformly distributed on [0, sigma_max], that is integrated over sigma and divided by sigma_max, a pair contributing only above its threshold and the prior being evaluated at each sigma. Evaluate the average exactly, not by numerical quadrature. The weights are not normalised; the pair probabilities are their normalisation over the candidate pairs.

The trophic specialization of a real predator is unknown; the source removes this parameter by averaging the plausibility of each prey pair over all specializations up to a maximum, so that narrow and wide diets are both represented.

Returns
-------
numpy.ndarray of float64 with shape (n_pairs,): the sigma-averaged unnormalised pair weights, in the order of the pairs.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def marginal_pair_weights(positions: "numpy.ndarray", predator: int, pairs: "numpy.ndarray",
                          thresholds: "numpy.ndarray", sigma_max: float) -> "numpy.ndarray":
    """Return the source's specialization-averaged weight of every given candidate prey pair of one predator (the pairs of step 03 at sigma = sigma_max, with their thresholds of step 04, in the same order) when the trophic specialization is unknown: the source's Bayesian weight of a pair at specialization sigma - its prior 1 / n(sigma), uniform over the n(sigma) pairs that are candidates at sigma, times its likelihood, the product of the Gaussian densities of mean zero and standard deviation sigma at the two prey offsets c - y_lower and y_upper - c, that is exp(-((c - y_lower)^2 + (y_upper - c)^2) / (2 sigma^2)) / (2 pi sigma^2) - averaged over sigma uniformly distributed on [0, sigma_max], that is integrated over sigma and divided by sigma_max, a pair contributing only above its threshold and the prior being evaluated at each sigma. Evaluate the average exactly, not by numerical quadrature. The weights are not normalised; the pair probabilities are their normalisation over the candidate pairs.

    Parameters
    ----------
    positions : numpy.ndarray
        One-dimensional finite array of the trophic positions of at least three species.
    predator : int
        Index of the predator species.
    pairs : numpy.ndarray
        Integer array of shape (n_pairs, 2): the candidate pairs of the predator at sigma_max (step 03).
    thresholds : numpy.ndarray
        One-dimensional array of length n_pairs: the admissibility thresholds of the pairs (step 04), all below sigma_max.
    sigma_max : float
        Positive upper bound of the trophic specialization.

    Returns
    -------
    weights : numpy.ndarray
        One-dimensional array of length n_pairs holding the averaged unnormalised weights (float64).

    Raises
    ------
    ValueError
        If sigma_max is not positive and finite, positions or predator are invalid, pairs is empty or malformed, or thresholds does not hold one finite value per pair in [0, sigma_max).
    """
    return weights

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np


def _oracle_marginal_pair_weights(positions: "numpy.ndarray", predator: int, pairs: "numpy.ndarray",
                                  thresholds: "numpy.ndarray", sigma_max: float) -> "numpy.ndarray":
    """Eq 29 evaluated exactly: the sigma-averaged Bayes numerator of every candidate pair on [0, sigma_max]."""
    y = np.asarray(positions, dtype=np.float64)
    P = np.asarray(pairs)
    thr = np.asarray(thresholds, dtype=np.float64)
    if not np.isfinite(sigma_max) or sigma_max <= 0.0:
        raise ValueError("sigma_max must be finite and positive")
    if y.ndim != 1 or y.size < 3 or not np.all(np.isfinite(y)):
        raise ValueError("positions must be a finite 1-D array of at least three species")
    if int(predator) != predator or not (0 <= predator < y.size):
        raise ValueError("predator must be a valid species index")
    if P.ndim != 2 or P.shape[1] != 2 or P.shape[0] == 0:
        raise ValueError("the predator has no candidate prey pair")
    if thr.shape != (P.shape[0],) or not np.all(np.isfinite(thr)) or np.any(thr < 0.0) or np.any(thr >= sigma_max):
        raise ValueError("thresholds must be one finite value per pair, nonnegative and below sigma_max")
    c = y[int(predator)] - 1.0
    smax = float(sigma_max)
    # the admissible set, and with it the prior 1/n_pairs(sigma), is piecewise constant between the thresholds
    knots = np.unique(np.concatenate([thr, [smax]]))
    weights = np.zeros(P.shape[0])
    for k, (d, u) in enumerate(P):
        a = 0.5 * ((c - y[d]) ** 2 + (y[u] - c) ** 2)            # numerator of the Gaussian exponent
        total = 0.0
        for lo, hi in zip(knots[:-1], knots[1:]):
            if hi <= thr[k]:
                continue                                         # the pair does not exist yet on this interval
            n_pairs = int(np.sum(thr <= lo + 1e-15))             # pairs admissible on (lo, hi)
            # int_lo^hi exp(-a / s^2) / (2 pi s^2) ds = [erfc(sqrt(a)/hi) - erfc(sqrt(a)/lo)] * sqrt(pi / a) / (4 pi)
            ra = math.sqrt(a)
            total += (math.erfc(ra / hi) - math.erfc(ra / lo)) * math.sqrt(math.pi / a) / (4.0 * math.pi) / n_pairs
        weights[k] = total / smax                                # uniform average over sigma in [0, sigma_max]
    return weights

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\npositions = _oracle_trophic_positions(true_diet)\npredator = 8\npairs = _oracle_candidate_pairs(positions, predator, 1.0 / 3.0)\nthresholds = _oracle_pair_thresholds(positions, predator, pairs)\nsigma_max = 1.0 / 3.0\n",
            "call": "marginal_pair_weights(positions, predator, pairs, thresholds, sigma_max)",
            "gold_call": "_oracle_marginal_pair_weights(positions, predator, pairs, thresholds, sigma_max)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\npositions = _oracle_trophic_positions(true_diet)\npredator = 6\npairs = _oracle_candidate_pairs(positions, predator, 1.0 / 3.0)\nthresholds = _oracle_pair_thresholds(positions, predator, pairs)\nsigma_max = 1.0 / 3.0\n",
            "call": "marginal_pair_weights(positions, predator, pairs, thresholds, sigma_max)",
            "gold_call": "_oracle_marginal_pair_weights(positions, predator, pairs, thresholds, sigma_max)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((8, 8))\ntrue_diet[1, 0] = true_diet[2, 0] = 1.0\ntrue_diet[3, [0, 1, 2]] = [0.2, 0.4, 0.4]\ntrue_diet[4, [1, 3]] = [0.6, 0.4]\ntrue_diet[5, [2, 3, 4]] = [0.3, 0.3, 0.4]\ntrue_diet[6, [3, 4, 5]] = [0.2, 0.5, 0.3]\ntrue_diet[7, [4, 5, 6]] = [0.3, 0.3, 0.4]\npositions = _oracle_trophic_positions(true_diet)\npredator = 7\npairs = _oracle_candidate_pairs(positions, predator, 0.4)\nthresholds = _oracle_pair_thresholds(positions, predator, pairs)\nsigma_max = 0.4\n",
            "call": "marginal_pair_weights(positions, predator, pairs, thresholds, sigma_max)",
            "gold_call": "_oracle_marginal_pair_weights(positions, predator, pairs, thresholds, sigma_max)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\npositions = _oracle_trophic_positions(true_diet)\npredator = 8\npairs = _oracle_candidate_pairs(positions, predator, 1.0 / 3.0)\nthresholds = _oracle_pair_thresholds(positions, predator, pairs)\nsigma_max = 0.0\ndef _exception_code(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "_exception_code(lambda: marginal_pair_weights(positions, predator, pairs, thresholds, sigma_max))",
            "gold_call": "_exception_code(lambda: _oracle_marginal_pair_weights(positions, predator, pairs, thresholds, sigma_max))",
        },
    ]
