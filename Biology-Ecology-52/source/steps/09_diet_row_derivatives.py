"""
Return the exact first and second derivatives of the reconstructed diet row of one predator with respect to the trophic position of one species, all other positions and sigma_max held fixed. The row is the one steps 02-06 build from the positions: the candidate pairs at sigma_max (step 03), their thresholds (step 04) and averaged weights (step 05), the two-prey diet of each pair (step 02) and the superposition (step 06); the pair list is held fixed (it persists under small displacements) while the pair weights (whose derivatives are step 08), their normalisation to probabilities and the two-prey diets all move with the position. Return the exact derivatives, analytically, as an array of shape (n_species, 2) whose columns are the first and the second derivative of the row (zero outside the candidate prey). Raise ValueError if the predator has no candidate pair at sigma_max, if another species sits at the predator's centre or on the admissibility window at sigma_max to within an absolute tolerance of 1e-12 (the row has no derivative there), or if step 08 raises.

The reconstructed diet coefficients respond to displaced positions through the pair diets, which the two-prey solution ties to the positions directly, and through the pair probabilities; combining both to second order gives the exact curvature of one row of the reconstructed web.

Returns
-------
numpy.ndarray of float64 with shape (n_species, 2): the exact first and second derivatives of the reconstructed diet row with respect to the species' position.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def diet_row_derivatives(positions: "numpy.ndarray", predator: int, sigma_max: float,
                         species: int) -> "numpy.ndarray":
    """Return the exact first and second derivatives of the reconstructed diet row of one predator with respect to the trophic position of one species, all other positions and sigma_max held fixed. The row is the one steps 02-06 build from the positions: the candidate pairs at sigma_max (step 03), their thresholds (step 04) and averaged weights (step 05), the two-prey diet of each pair (step 02) and the superposition (step 06); the pair list is held fixed (it persists under small displacements) while the pair weights (whose derivatives are step 08), their normalisation to probabilities and the two-prey diets all move with the position. Return the exact derivatives, analytically, as an array of shape (n_species, 2) whose columns are the first and the second derivative of the row (zero outside the candidate prey). Raise ValueError if the predator has no candidate pair at sigma_max, if another species sits at the predator's centre or on the admissibility window at sigma_max to within an absolute tolerance of 1e-12 (the row has no derivative there), or if step 08 raises.

    Parameters
    ----------
    positions : numpy.ndarray
        One-dimensional finite array of the trophic positions of at least three species.
    predator : int
        Index of the predator species.
    sigma_max : float
        Positive upper bound of the trophic specialization.
    species : int
        Index of the species whose trophic position is displaced.

    Returns
    -------
    derivatives : numpy.ndarray
        Array of shape (n_species, 2): column 0 the first and column 1 the second derivative of the predator's reconstructed diet row with respect to the position of the species (float64).

    Raises
    ------
    ValueError
        If sigma_max is not positive and finite, positions, predator or species are invalid, the predator has no candidate pair at sigma_max, a species sits at the predator's centre or on its admissibility window at sigma_max to within an absolute tolerance of 1e-12, or the weight derivatives of step 08 do not exist.
    """
    return derivatives

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np


def _oracle_diet_row_derivatives(positions: "numpy.ndarray", predator: int, sigma_max: float,
                                 species: int) -> "numpy.ndarray":
    """Exact first and second derivatives of the reconstructed diet row of one predator (steps 02-06) w.r.t. one position."""
    y = np.asarray(positions, dtype=np.float64)
    if not np.isfinite(sigma_max) or sigma_max <= 0.0:
        raise ValueError("sigma_max must be finite and positive")
    if y.ndim != 1 or y.size < 3 or not np.all(np.isfinite(y)):
        raise ValueError("positions must be a finite 1-D array of at least three species")
    if int(predator) != predator or not (0 <= predator < y.size):
        raise ValueError("predator must be a valid species index")
    if int(species) != species or not (0 <= species < y.size):
        raise ValueError("species must be a valid species index")
    i, m, n = int(predator), int(species), y.size
    c = y[i] - 1.0
    if np.any((np.arange(n) != i) & (np.abs(y - c) < 1e-12)):
        raise ValueError("a prey sits exactly at the predator's centre; the two-prey decomposition does not apply")
    if np.any((np.arange(n) != i) & (np.abs(np.abs(y - c) - 3.0 * float(sigma_max)) < 1e-12)):
        raise ValueError("a species sits exactly on the admissibility window at sigma_max; the row has no derivative there")
    pairs = _oracle_candidate_pairs(y, i, sigma_max)
    if pairs.shape[0] == 0:
        raise ValueError("the predator has no candidate prey pair at sigma_max")
    thresholds = _oracle_pair_thresholds(y, i, pairs)
    w = _oracle_marginal_pair_weights(y, i, pairs, thresholds, sigma_max)
    dw = _oracle_marginal_weight_derivatives(y, i, pairs, sigma_max, m)
    W, W1, W2 = w.sum(), dw[:, 0].sum(), dw[:, 1].sum()
    probs = w / W
    p1 = dw[:, 0] / W - w * W1 / W ** 2                           # normalisation over the pairs, first order
    p2 = dw[:, 1] / W - 2.0 * dw[:, 0] * W1 / W ** 2 - w * W2 / W ** 2 + 2.0 * w * W1 ** 2 / W ** 3
    dc = 1.0 if m == i else 0.0
    out = np.zeros((n, 2))
    for k, (b, a) in enumerate(pairs):
        diet = _oracle_pair_diet_coefficients(y[b], y[a], c)
        span = y[a] - y[b]
        dy_a, dy_b = float(a == m), float(b == m)
        # Eq 7: Q_b = (y_a - c) / (y_a - y_b), Q_a = 1 - Q_b; numerator and denominator are linear in the position
        dnum, dspan = dy_a - dc, dy_a - dy_b
        q1 = (dnum * span - (y[a] - c) * dspan) / span ** 2
        q2 = -2.0 * dspan * q1 / span
        out[b, 0] += p1[k] * diet[0] + probs[k] * q1
        out[a, 0] += p1[k] * diet[1] - probs[k] * q1
        out[b, 1] += p2[k] * diet[0] + 2.0 * p1[k] * q1 + probs[k] * q2
        out[a, 1] += p2[k] * diet[1] - 2.0 * p1[k] * q1 - probs[k] * q2
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\npositions = _oracle_trophic_positions(true_diet)\npredator, sigma_max, species = 8, 1.0 / 3.0, 8\n",
            "call": "diet_row_derivatives(positions, predator, sigma_max, species)",
            "gold_call": "_oracle_diet_row_derivatives(positions, predator, sigma_max, species)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\npositions = _oracle_trophic_positions(true_diet)\npredator, sigma_max, species = 11, 1.0 / 3.0, 10\n",
            "call": "diet_row_derivatives(positions, predator, sigma_max, species)",
            "gold_call": "_oracle_diet_row_derivatives(positions, predator, sigma_max, species)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\npositions = _oracle_trophic_positions(true_diet)\npredator, sigma_max, species = 6, 1.0 / 3.0, 6\n",
            "call": "diet_row_derivatives(positions, predator, sigma_max, species)",
            "gold_call": "_oracle_diet_row_derivatives(positions, predator, sigma_max, species)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((6, 6))\ntrue_diet[1, 0] = true_diet[2, 0] = 1.0\ntrue_diet[3, [0, 1]] = [0.3, 0.7]\ntrue_diet[4, [1, 2, 3]] = [0.2, 0.3, 0.5]\ntrue_diet[5, [2, 3, 4]] = [0.25, 0.25, 0.50]\npositions = _oracle_trophic_positions(true_diet)\npredator, sigma_max, species = 5, 1.0 / 3.0, 4\n",
            "call": "diet_row_derivatives(positions, predator, sigma_max, species)",
            "gold_call": "_oracle_diet_row_derivatives(positions, predator, sigma_max, species)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((8, 8))\ntrue_diet[1, 0] = true_diet[2, 0] = 1.0\ntrue_diet[3, [0, 1, 2]] = [0.2, 0.4, 0.4]\ntrue_diet[4, [1, 3]] = [0.6, 0.4]\ntrue_diet[5, [2, 3, 4]] = [0.3, 0.3, 0.4]\ntrue_diet[6, [3, 4, 5]] = [0.2, 0.5, 0.3]\ntrue_diet[7, [4, 5, 6]] = [0.3, 0.3, 0.4]\npositions = _oracle_trophic_positions(true_diet)\npredator, sigma_max, species = 7, 0.4, 7\n",
            "call": "diet_row_derivatives(positions, predator, sigma_max, species)",
            "gold_call": "_oracle_diet_row_derivatives(positions, predator, sigma_max, species)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\npositions = np.array([1.0, 2.0, 2.0, 3.9])\npredator, sigma_max, species = 3, 1.0 / 3.0, 3\ndef run_model():\n    try:\n        diet_row_derivatives(positions, predator, sigma_max, species)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_diet_row_derivatives(positions, predator, sigma_max, species)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
