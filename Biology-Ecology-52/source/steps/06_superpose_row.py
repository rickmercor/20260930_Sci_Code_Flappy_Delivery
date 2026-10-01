"""
Return the reconstructed diet row of one predator as the source's superposition of its candidate pairs: the pair weights (step 05) normalised to probabilities summing to one, and the row of length n_species whose entry for each prey is the probability-weighted sum, over the pairs containing that prey, of the prey's coefficient in the pair's two-prey diet (step 02). The row is zero outside the candidate prey.

Superposing the two-prey diets with their probabilities yields a full diet row that automatically satisfies both constraints of the inverse problem, the unit row sum and the predator's trophic position.

Returns
-------
numpy.ndarray of float64 with shape (n_species,): the reconstructed diet row.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def superpose_row(n_species: int, pairs: "numpy.ndarray", pair_diets: "numpy.ndarray",
                  weights: "numpy.ndarray") -> "numpy.ndarray":
    """Return the reconstructed diet row of one predator as the source's superposition of its candidate pairs: the pair weights (step 05) normalised to probabilities summing to one, and the row of length n_species whose entry for each prey is the probability-weighted sum, over the pairs containing that prey, of the prey's coefficient in the pair's two-prey diet (step 02). The row is zero outside the candidate prey.

    Parameters
    ----------
    n_species : int
        Number of species in the web, at least 3 (the length of the row).
    pairs : numpy.ndarray
        Integer array of shape (n_pairs, 2) of (lower prey index, upper prey index) pairs, non-empty.
    pair_diets : numpy.ndarray
        Array of shape (n_pairs, 2): for each pair the two-prey diet [coefficient of the lower prey, coefficient of the upper prey].
    weights : numpy.ndarray
        One-dimensional array of length n_pairs of nonnegative pair weights with a positive sum.

    Returns
    -------
    row : numpy.ndarray
        One-dimensional array of length n_species: the reconstructed diet coefficients of the predator (float64).

    Raises
    ------
    ValueError
        If n_species is not an integer of at least 3, pairs is empty or holds an index outside the web, pair_diets is not a finite nonnegative array of shape (n_pairs, 2), or weights is not one finite nonnegative value per pair with a positive sum.
    """
    return row

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np


def _oracle_superpose_row(n_species: int, pairs: "numpy.ndarray", pair_diets: "numpy.ndarray",
                          weights: "numpy.ndarray") -> "numpy.ndarray":
    """Superposition (Eqs 23-26): the diet row from the pair diets weighted by the normalised pair weights."""
    P = np.asarray(pairs)
    D = np.asarray(pair_diets, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if int(n_species) != n_species or n_species < 3:
        raise ValueError("n_species must be an integer of at least 3")
    n = int(n_species)
    if P.ndim != 2 or P.shape[1] != 2 or P.shape[0] == 0 or P.min() < 0 or P.max() >= n:
        raise ValueError("pairs must be a non-empty integer array of shape (n_pairs, 2) with valid species indices")
    if D.shape != (P.shape[0], 2) or not np.all(np.isfinite(D)) or np.any(D < 0.0):
        raise ValueError("pair_diets must be a finite nonnegative array of shape (n_pairs, 2)")
    if w.shape != (P.shape[0],) or not np.all(np.isfinite(w)) or np.any(w < 0.0) or w.sum() <= 0.0:
        raise ValueError("weights must be one finite nonnegative value per pair with a positive sum")
    probs = w / w.sum()                                           # law of total probability over the pairs
    row = np.zeros(n)
    for p, (d, u), t in zip(probs, P, D):
        row[d] += p * t[0]
        row[u] += p * t[1]
    return row

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\npositions = _oracle_trophic_positions(true_diet)\npredator = 8\nSMAX = 1.0 / 3.0\npairs = _oracle_candidate_pairs(positions, predator, 1.0 / 3.0)\nthresholds = _oracle_pair_thresholds(positions, predator, pairs)\ncentre = positions[predator] - 1.0\nn_species = positions.size\nweights = _oracle_marginal_pair_weights(positions, predator, pairs, thresholds, SMAX)\npair_diets = np.array([_oracle_pair_diet_coefficients(positions[d], positions[u], centre) for d, u in pairs])\n",
            "call": "superpose_row(n_species, pairs, pair_diets, weights)",
            "gold_call": "_oracle_superpose_row(n_species, pairs, pair_diets, weights)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\npositions = _oracle_trophic_positions(true_diet)\npredator = 11\nSMAX = 1.0 / 3.0\npairs = _oracle_candidate_pairs(positions, predator, SMAX)\nthresholds = _oracle_pair_thresholds(positions, predator, pairs)\ncentre = positions[predator] - 1.0\nn_species = positions.size\nweights = _oracle_marginal_pair_weights(positions, predator, pairs, thresholds, SMAX)\npair_diets = np.array([_oracle_pair_diet_coefficients(positions[d], positions[u], centre) for d, u in pairs])\n",
            "call": "superpose_row(n_species, pairs, pair_diets, weights)",
            "gold_call": "_oracle_superpose_row(n_species, pairs, pair_diets, weights)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((6, 6))\ntrue_diet[1, 0] = true_diet[2, 0] = 1.0\ntrue_diet[3, [0, 1]] = [0.3, 0.7]\ntrue_diet[4, [1, 2, 3]] = [0.2, 0.3, 0.5]\ntrue_diet[5, [2, 3, 4]] = [0.25, 0.25, 0.50]\npositions = _oracle_trophic_positions(true_diet)\npredator = 5\nSMAX = 1.0 / 3.0\npairs = _oracle_candidate_pairs(positions, predator, SMAX)\nthresholds = _oracle_pair_thresholds(positions, predator, pairs)\ncentre = positions[predator] - 1.0\nn_species = positions.size\nweights = _oracle_marginal_pair_weights(positions, predator, pairs, thresholds, SMAX)\npair_diets = np.array([_oracle_pair_diet_coefficients(positions[d], positions[u], centre) for d, u in pairs])\n",
            "call": "superpose_row(n_species, pairs, pair_diets, weights)",
            "gold_call": "_oracle_superpose_row(n_species, pairs, pair_diets, weights)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\npositions = _oracle_trophic_positions(true_diet)\npredator = 8\nSMAX = 1.0 / 3.0\npairs = _oracle_candidate_pairs(positions, predator, 1.0 / 3.0)\nthresholds = _oracle_pair_thresholds(positions, predator, pairs)\ncentre = positions[predator] - 1.0\nn_species = positions.size\npair_diets = np.array([_oracle_pair_diet_coefficients(positions[d], positions[u], centre) for d, u in pairs])\nweights = np.zeros(len(pairs))\ndef _exception_code(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "_exception_code(lambda: superpose_row(n_species, pairs, pair_diets, weights))",
            "gold_call": "_exception_code(lambda: _oracle_superpose_row(n_species, pairs, pair_diets, weights))",
        },
    ]
