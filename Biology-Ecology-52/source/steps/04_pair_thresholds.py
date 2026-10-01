"""
Return, for every given candidate prey pair of one predator (rows (lower prey index, upper prey index) as produced by step 03), the admissibility threshold of the pair: the trophic specialization above which both of its prey lie inside the source's admissibility window of step 03, so that the pair is a candidate exactly for sigma strictly above the threshold. Return one threshold per pair, in the order of the input.

Because the admissible window widens with the trophic specialization, each prey pair enters the candidate set at a specialization of its own; these thresholds partition the range of specializations into intervals on which the candidate set, and hence the number of candidate pairs, is constant.

Returns
-------
numpy.ndarray of float64 with shape (n_pairs,): the admissibility thresholds of the given pairs, in their order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pair_thresholds(positions: "numpy.ndarray", predator: int, pairs: "numpy.ndarray") -> "numpy.ndarray":
    """Return, for every given candidate prey pair of one predator (rows (lower prey index, upper prey index) as produced by step 03), the admissibility threshold of the pair: the trophic specialization above which both of its prey lie inside the source's admissibility window of step 03, so that the pair is a candidate exactly for sigma strictly above the threshold. Return one threshold per pair, in the order of the input.

    Parameters
    ----------
    positions : numpy.ndarray
        One-dimensional finite array of the trophic positions of at least three species.
    predator : int
        Index of the predator species.
    pairs : numpy.ndarray
        Integer array of shape (n_pairs, 2) of (lower prey index, upper prey index) pairs of the predator, every lower prey below and every upper prey above the predator's centre.

    Returns
    -------
    thresholds : numpy.ndarray
        One-dimensional array of length n_pairs holding the admissibility threshold of each pair (float64).

    Raises
    ------
    ValueError
        If positions is not a finite 1-D array of at least three entries, predator is not a valid index, pairs is not an integer array of shape (n_pairs, 2) with valid indices, or a pair does not bracket the predator's centre.
    """
    return thresholds

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np


def _oracle_pair_thresholds(positions: "numpy.ndarray", predator: int, pairs: "numpy.ndarray") -> "numpy.ndarray":
    """Admissibility threshold of every given pair: the pair is a candidate for sigma above max(d_b, d_a)/3."""
    y = np.asarray(positions, dtype=np.float64)
    P = np.asarray(pairs)
    if y.ndim != 1 or y.size < 3 or not np.all(np.isfinite(y)):
        raise ValueError("positions must be a finite 1-D array of at least three species")
    if int(predator) != predator or not (0 <= predator < y.size):
        raise ValueError("predator must be a valid species index")
    if P.ndim != 2 or P.shape[1] != 2 or P.size and (P.min() < 0 or P.max() >= y.size):
        raise ValueError("pairs must be an integer array of shape (n_pairs, 2) with valid species indices")
    c = y[int(predator)] - 1.0
    if P.shape[0] and (np.any(y[P[:, 0]] >= c) or np.any(y[P[:, 1]] <= c)):
        raise ValueError("every pair must hold one prey below and one prey above the predator's centre")
    # the strict 3-sigma window admits the lower prey for sigma > (c - y_b)/3 and the upper prey for sigma > (y_a - c)/3;
    # an empty pair set gives an empty threshold array
    return np.maximum(c - y[P[:, 0]], y[P[:, 1]] - c).astype(np.float64) / 3.0

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\npositions = _oracle_trophic_positions(true_diet)\npredator = 8\npairs = _oracle_candidate_pairs(positions, predator, 1.0 / 3.0)\n",
            "call": "pair_thresholds(positions, predator, pairs)",
            "gold_call": "_oracle_pair_thresholds(positions, predator, pairs)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\npositions = _oracle_trophic_positions(true_diet)\npredator = 6\npairs = _oracle_candidate_pairs(positions, predator, 1.0 / 3.0)\n",
            "call": "pair_thresholds(positions, predator, pairs)",
            "gold_call": "_oracle_pair_thresholds(positions, predator, pairs)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((8, 8))\ntrue_diet[1, 0] = true_diet[2, 0] = 1.0\ntrue_diet[3, [0, 1, 2]] = [0.2, 0.4, 0.4]\ntrue_diet[4, [1, 3]] = [0.6, 0.4]\ntrue_diet[5, [2, 3, 4]] = [0.3, 0.3, 0.4]\ntrue_diet[6, [3, 4, 5]] = [0.2, 0.5, 0.3]\ntrue_diet[7, [4, 5, 6]] = [0.3, 0.3, 0.4]\npositions = _oracle_trophic_positions(true_diet)\npredator = 7\npairs = _oracle_candidate_pairs(positions, predator, 0.4)\n",
            "call": "pair_thresholds(positions, predator, pairs)",
            "gold_call": "_oracle_pair_thresholds(positions, predator, pairs)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\npositions = _oracle_trophic_positions(true_diet)\npredator = 8\npairs = np.array([[6, 4]])\ndef _exception_code(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "_exception_code(lambda: pair_thresholds(positions, predator, pairs))",
            "gold_call": "_exception_code(lambda: _oracle_pair_thresholds(positions, predator, pairs))",
        },
    ]
