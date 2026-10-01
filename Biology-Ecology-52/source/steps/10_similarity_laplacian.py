"""
Orchestrator. For a true diet-coefficient matrix, compute the trophic positions of all species (step 01) and reconstruct the whole web from those positions alone: species whose position is 1 to within an absolute tolerance of 1e-12 are primary producers and get zero rows; species whose position is 2 to within the same tolerance are primary consumers and feed on the primary producers alone, in equal proportions; for every other consumer form its candidate pairs at sigma_max (step 03), their thresholds (step 04) and averaged weights (step 05), the two-prey diet of each pair (step 02), and its row by superposition (step 06). Then take the similarity of step 07 between the true and the reconstructed matrices (the sum of the entry-wise products divided by the product of the entry-wise L2 norms) as a function of the positions of the consumers reconstructed by pairs (the producers' and primary consumers' positions and rows are fixed), the reconstructed matrix moving with them through the row derivatives of step 09 while the true matrix is unchanged, and return its Laplacian with respect to those positions: the sum over them of the exact second partial derivatives of the similarity. Raise ValueError if the web has no producer, some other species sits at the centre of a consumer reconstructed by pairs to within an absolute tolerance of 1e-12 (the producers at a primary consumer's centre are its diet by the rule above, not an error), a consumer reconstructed by pairs has no candidate pair at sigma_max, or the derivatives do not exist (step 08 or step 09 raises). Call the earlier step functions rather than reimplementing them.

The source validates its reconstruction on known webs and propagates the measurement error of the positions through it by simulation; for small independent errors of standard deviation s the expected similarity deviates from the error-free value by s^2/2 times its Laplacian, so the Laplacian is the exact leading bias of that propagation and needs the full second-order response of the pair structure.

Returns
-------
float, the Laplacian of the similarity between the true and the reconstructed diet matrices with respect to the estimated trophic positions.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def similarity_laplacian(true_diet: "numpy.ndarray", sigma_max: float) -> float:
    """Orchestrator. For a true diet-coefficient matrix, compute the trophic positions of all species (step 01) and reconstruct the whole web from those positions alone: species whose position is 1 to within an absolute tolerance of 1e-12 are primary producers and get zero rows; species whose position is 2 to within the same tolerance are primary consumers and feed on the primary producers alone, in equal proportions; for every other consumer form its candidate pairs at sigma_max (step 03), their thresholds (step 04) and averaged weights (step 05), the two-prey diet of each pair (step 02), and its row by superposition (step 06). Then take the similarity of step 07 between the true and the reconstructed matrices (the sum of the entry-wise products divided by the product of the entry-wise L2 norms) as a function of the positions of the consumers reconstructed by pairs (the producers' and primary consumers' positions and rows are fixed), the reconstructed matrix moving with them through the row derivatives of step 09 while the true matrix is unchanged, and return its Laplacian with respect to those positions: the sum over them of the exact second partial derivatives of the similarity. Raise ValueError if the web has no producer, some other species sits at the centre of a consumer reconstructed by pairs to within an absolute tolerance of 1e-12 (the producers at a primary consumer's centre are its diet by the rule above, not an error), a consumer reconstructed by pairs has no candidate pair at sigma_max, or the derivatives do not exist (step 08 or step 09 raises). Call the earlier step functions rather than reimplementing them.

    Parameters
    ----------
    true_diet : numpy.ndarray
        Square array of shape (n, n) of finite nonnegative true diet coefficients (consumer rows sum to one).
    sigma_max : float
        Positive upper bound of the trophic specialization.

    Returns
    -------
    laplacian : float
        The sum of the second partial derivatives of the similarity with respect to the positions of the consumers reconstructed by pairs, as a native Python float.

    Raises
    ------
    ValueError
        If true_diet is invalid for step 01, sigma_max is not positive and finite, the web has no producer, a species sits at the centre of a consumer reconstructed by pairs to within an absolute tolerance of 1e-12, such a consumer has no candidate pair at sigma_max, or the derivatives do not exist (equal prey offsets in a pair, a species on an admissibility window, each to within 1e-12).
    """
    return laplacian

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np


def _oracle_similarity_laplacian(true_diet: "numpy.ndarray", sigma_max: float) -> float:
    """ORCHESTRATOR: positions of the true web, reconstruction from the positions alone, and the Laplacian of the
    similarity (Eq 39) with respect to the positions of the consumers reconstructed by pairs (the sum of the second
    partial derivatives), which fixes the leading bias of the expected similarity under small position errors."""
    Qt = np.asarray(true_diet, dtype=np.float64)
    if not np.isfinite(sigma_max) or sigma_max <= 0.0:
        raise ValueError("sigma_max must be finite and positive")
    y = _oracle_trophic_positions(Qt)
    n = y.size
    producers = np.where(np.abs(y - 1.0) < 1e-12)[0]
    if producers.size == 0:
        raise ValueError("the web has no primary producer")
    estimated = []                                                # consumers whose rows come from the pair decomposition
    Q = np.zeros((n, n))
    for i in range(n):
        if abs(y[i] - 1.0) < 1e-12:
            continue                                              # producer: feeds on nothing
        if abs(y[i] - 2.0) < 1e-12:
            Q[i, producers] = 1.0 / producers.size               # primary consumer: producers only (source convention)
            continue
        c = y[i] - 1.0
        if np.any((np.arange(n) != i) & (np.abs(y - c) < 1e-12)):
            raise ValueError("a prey sits exactly at the centre of predator %d; the two-prey decomposition does not apply" % i)
        pairs = _oracle_candidate_pairs(y, i, sigma_max)
        if pairs.shape[0] == 0:
            raise ValueError("predator %d has no candidate prey pair at sigma_max" % i)
        thresholds = _oracle_pair_thresholds(y, i, pairs)
        weights = _oracle_marginal_pair_weights(y, i, pairs, thresholds, sigma_max)
        diets = np.array([_oracle_pair_diet_coefficients(y[d], y[u], c) for d, u in pairs])
        Q[i] = _oracle_superpose_row(n, pairs, diets, weights)
        estimated.append(i)
    metrics = _oracle_reconstruction_metrics(Qt, Q)
    theta = metrics[8]
    norm_true, norm_rec = np.linalg.norm(Qt), np.linalg.norm(Q)
    laplacian = 0.0
    for m in estimated:
        dQ, d2Q = np.zeros((n, n)), np.zeros((n, n))
        for i in estimated:
            der = _oracle_diet_row_derivatives(y, i, sigma_max, m)   # rows of producers and primary consumers are fixed
            dQ[i], d2Q[i] = der[:, 0], der[:, 1]
        # Eq 39: theta = N / (|Qt| |Q|) with N = <Q, Qt>; only the reconstructed matrix moves with the positions
        num1, num2 = np.sum(dQ * Qt), np.sum(d2Q * Qt)
        r1 = np.sum(Q * dQ) / norm_rec
        r2 = (np.sum(dQ * dQ) + np.sum(Q * d2Q)) / norm_rec - np.sum(Q * dQ) ** 2 / norm_rec ** 3
        num0 = theta * norm_true * norm_rec
        laplacian += (num2 / norm_rec - 2.0 * num1 * r1 / norm_rec ** 2 - num0 * r2 / norm_rec ** 2
                      + 2.0 * num0 * r1 ** 2 / norm_rec ** 3) / norm_true
    return float(laplacian)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\nsigma_max = 1.0 / 3.0\n",
            "call": "similarity_laplacian(true_diet, sigma_max)",
            "gold_call": "_oracle_similarity_laplacian(true_diet, sigma_max)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((8, 8))\ntrue_diet[1, 0] = true_diet[2, 0] = 1.0\ntrue_diet[3, [0, 1, 2]] = [0.2, 0.4, 0.4]\ntrue_diet[4, [1, 3]] = [0.6, 0.4]\ntrue_diet[5, [2, 3, 4]] = [0.3, 0.3, 0.4]\ntrue_diet[6, [3, 4, 5]] = [0.2, 0.5, 0.3]\ntrue_diet[7, [4, 5, 6]] = [0.3, 0.3, 0.4]\nsigma_max = 0.4\n",
            "call": "similarity_laplacian(true_diet, sigma_max)",
            "gold_call": "_oracle_similarity_laplacian(true_diet, sigma_max)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\nsigma_max = 0.5\n",
            "call": "similarity_laplacian(true_diet, sigma_max)",
            "gold_call": "_oracle_similarity_laplacian(true_diet, sigma_max)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((8, 8))\ntrue_diet[1, 0] = true_diet[2, 0] = 1.0\ntrue_diet[3, [0, 1, 2]] = [0.2, 0.4, 0.4]\ntrue_diet[4, [1, 3]] = [0.6, 0.4]\ntrue_diet[5, [2, 3, 4]] = [0.3, 0.3, 0.4]\ntrue_diet[6, [3, 4, 5]] = [0.2, 0.5, 0.3]\ntrue_diet[7, [4, 5, 6]] = [0.3, 0.3, 0.4]\nsigma_max = 1.0 / 3.0\n",
            "call": "similarity_laplacian(true_diet, sigma_max)",
            "gold_call": "_oracle_similarity_laplacian(true_diet, sigma_max)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((6, 6))\ntrue_diet[1, 0] = true_diet[2, 0] = 1.0\ntrue_diet[3, [0, 1]] = [0.3, 0.7]\ntrue_diet[4, [1, 2, 3]] = [0.2, 0.3, 0.5]\ntrue_diet[5, [2, 3, 4]] = [0.25, 0.25, 0.50]\nsigma_max = 1.0 / 3.0\ndef run_model():\n    try:\n        similarity_laplacian(true_diet, sigma_max)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_similarity_laplacian(true_diet, sigma_max)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
