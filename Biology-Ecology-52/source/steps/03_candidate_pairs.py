"""
Return the candidate prey pairs of one predator at trophic specialization sigma according to the source's decomposition. The admissible prey of predator i are the other species whose trophic position lies inside the source's admissibility window around the centre c = positions[i] - 1, the window of half-width 3 sigma, from c - 3 sigma to c + 3 sigma (strict inequalities: a species exactly on the window's boundary or exactly at the centre is not admissible; the predator itself is never admissible); every admissible prey below the centre is paired with every admissible prey above it, so that the two-prey solution of step 02 applies to every pair. Return an integer array of shape (n_pairs, 2) whose rows are (index of the lower prey, index of the upper prey), ordered by lower-prey index and then by upper-prey index in increasing order of species index; the array has zero rows when no pair exists.

A predator with more than two prey has infinitely many diets consistent with its trophic position; the reconstruction resolves this by admitting only the prey that are plausible at a given trophic specialization and breaking them into pairs, each of which has a unique two-prey diet, and superposing the pairs later.

Returns
-------
numpy.ndarray of int64 with shape (n_pairs, 2): the candidate (lower prey, upper prey) pairs.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def candidate_pairs(positions: "numpy.ndarray", predator: int, sigma: float) -> "numpy.ndarray":
    """Return the candidate prey pairs of one predator at trophic specialization sigma according to the source's decomposition. The admissible prey of predator i are the other species whose trophic position lies inside the source's admissibility window around the centre c = positions[i] - 1, the window of half-width 3 sigma, from c - 3 sigma to c + 3 sigma (strict inequalities: a species exactly on the window's boundary or exactly at the centre is not admissible; the predator itself is never admissible); every admissible prey below the centre is paired with every admissible prey above it, so that the two-prey solution of step 02 applies to every pair. Return an integer array of shape (n_pairs, 2) whose rows are (index of the lower prey, index of the upper prey), ordered by lower-prey index and then by upper-prey index in increasing order of species index; the array has zero rows when no pair exists.

    Parameters
    ----------
    positions : numpy.ndarray
        One-dimensional finite array of the trophic positions of at least three species.
    predator : int
        Index of the predator species.
    sigma : float
        Nonnegative trophic specialization.

    Returns
    -------
    pairs : numpy.ndarray
        Integer array of shape (n_pairs, 2): rows (lower prey index, upper prey index) in the stated order.

    Raises
    ------
    ValueError
        If positions is not a finite 1-D array of at least three entries, predator is not a valid index, or sigma is negative or not finite.
    """
    return pairs

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np


def _oracle_candidate_pairs(positions: "numpy.ndarray", predator: int, sigma: float) -> "numpy.ndarray":
    """Source decomposition at trophic specialization sigma: all (below, above) pairs of admissible prey of the predator."""
    y = np.asarray(positions, dtype=np.float64)
    if y.ndim != 1 or y.size < 3 or not np.all(np.isfinite(y)):
        raise ValueError("positions must be a finite 1-D array of at least three species")
    if int(predator) != predator or not (0 <= predator < y.size):
        raise ValueError("predator must be a valid species index")
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma must be finite and nonnegative")
    i = int(predator)
    c = y[i] - 1.0
    idx = np.arange(y.size)
    # strict 3-sigma window on each side of the centre; the predator itself is never a prey (no cannibalism)
    below = idx[(idx != i) & (y < c) & (y > c - 3.0 * float(sigma))]
    above = idx[(idx != i) & (y > c) & (y < c + 3.0 * float(sigma))]
    # every prey below the centre paired with every prey above it (the two-prey solution needs one of each)
    pairs = np.array([(int(d), int(u)) for d in below for u in above], dtype=np.int64).reshape(-1, 2)
    return pairs

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\npositions = _oracle_trophic_positions(true_diet)\npredator, sigma = 8, 1.0 / 3.0\n",
            "call": "candidate_pairs(positions, predator, sigma)",
            "gold_call": "_oracle_candidate_pairs(positions, predator, sigma)",
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\npositions = _oracle_trophic_positions(true_diet)\npredator, sigma = 6, 0.2\n",
            "call": "candidate_pairs(positions, predator, sigma)",
            "gold_call": "_oracle_candidate_pairs(positions, predator, sigma)",
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\npositions = _oracle_trophic_positions(true_diet)\npredator, sigma = 11, 0.05\n",
            "call": "candidate_pairs(positions, predator, sigma)",
            "gold_call": "_oracle_candidate_pairs(positions, predator, sigma)",
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\npositions = _oracle_trophic_positions(true_diet)\npredator, sigma = 8, -0.1\ndef _exception_code(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "_exception_code(lambda: candidate_pairs(positions, predator, sigma))",
            "gold_call": "_exception_code(lambda: _oracle_candidate_pairs(positions, predator, sigma))",
        },
    ]
