"""
Return the trophic positions of every species of a food web from its diet-coefficient matrix Q, whose entry Q[i, j] is the proportion of prey j in the diet of predator i (rows of consumers sum to one, rows of primary producers are zero, the diagonal is zero because cannibalism is excluded). The trophic position of a predator is one plus the diet-weighted mean of the trophic positions of its prey, and primary producers have trophic position one; solve the resulting linear system y = (I - Q)^-1 1 exactly.

Trophic position ranks the species of a food web by how many feeding steps separate them from the primary producers; when the diet coefficients are known it follows from a linear system, and it is the only information the reconstruction method of this task starts from.

Returns
-------
numpy.ndarray of float64 with shape (n,): the trophic positions.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def trophic_positions(diet: "numpy.ndarray") -> "numpy.ndarray":
    """Return the trophic positions of every species of a food web from its diet-coefficient matrix Q, whose entry Q[i, j] is the proportion of prey j in the diet of predator i (rows of consumers sum to one, rows of primary producers are zero, the diagonal is zero because cannibalism is excluded). The trophic position of a predator is one plus the diet-weighted mean of the trophic positions of its prey, and primary producers have trophic position one; solve the resulting linear system y = (I - Q)^-1 1 exactly.

    Parameters
    ----------
    diet : numpy.ndarray
        Square array of shape (n, n) of finite nonnegative diet coefficients; consumer rows sum to one, producer rows to zero, the diagonal is zero.

    Returns
    -------
    positions : numpy.ndarray
        One-dimensional array of length n holding the trophic position of each species (float64).

    Raises
    ------
    ValueError
        If diet is not square, contains a non-finite or negative entry or a nonzero diagonal entry, has a row sum other than 0 or 1, or I - Q is singular.
    """
    return positions

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np


def _oracle_trophic_positions(diet: "numpy.ndarray") -> "numpy.ndarray":
    """Trophic positions y = (I - Q)^-1 1 of a diet-coefficient matrix Q (rows predators, columns prey)."""
    Q = np.asarray(diet, dtype=np.float64)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1] or Q.shape[0] < 1:
        raise ValueError("diet must be a square matrix")
    if not np.all(np.isfinite(Q)) or np.any(Q < 0.0):
        raise ValueError("diet coefficients must be finite and nonnegative")
    if np.any(np.abs(np.diag(Q)) > 0.0):
        raise ValueError("cannibalism is excluded: the diagonal must be zero")
    sums = Q.sum(axis=1)
    if not np.all((np.abs(sums) < 1e-12) | (np.abs(sums - 1.0) < 1e-12)):
        raise ValueError("every row must sum to 0 (producer) or 1 (consumer)")
    n = Q.shape[0]
    M = np.eye(n) - Q
    if abs(np.linalg.det(M)) < 1e-12:
        raise ValueError("I - Q is singular: the web has no producer or contains a closed feeding loop")
    return np.linalg.solve(M, np.ones(n))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\n",
            "call": "trophic_positions(true_diet)",
            "gold_call": "_oracle_trophic_positions(true_diet)",
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((6, 6))\ntrue_diet[1, 0] = true_diet[2, 0] = 1.0\ntrue_diet[3, [0, 1]] = [0.3, 0.7]\ntrue_diet[4, [1, 2, 3]] = [0.2, 0.3, 0.5]\ntrue_diet[5, [2, 3, 4]] = [0.25, 0.25, 0.50]\n",
            "call": "trophic_positions(true_diet)",
            "gold_call": "_oracle_trophic_positions(true_diet)",
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((8, 8))\ntrue_diet[1, 0] = true_diet[2, 0] = 1.0\ntrue_diet[3, [0, 1, 2]] = [0.2, 0.4, 0.4]\ntrue_diet[4, [1, 3]] = [0.6, 0.4]\ntrue_diet[5, [2, 3, 4]] = [0.3, 0.3, 0.4]\ntrue_diet[6, [3, 4, 5]] = [0.2, 0.5, 0.3]\ntrue_diet[7, [4, 5, 6]] = [0.3, 0.3, 0.4]\n",
            "call": "trophic_positions(true_diet)",
            "gold_call": "_oracle_trophic_positions(true_diet)",
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((4, 4))\ntrue_diet[1, 0] = 1.0\ntrue_diet[2, [0, 1]] = [0.5, 0.6]\ntrue_diet[3, [1, 2]] = [0.5, 0.5]\ndef run_model():\n    try:\n        trophic_positions(true_diet)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_trophic_positions(true_diet)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
