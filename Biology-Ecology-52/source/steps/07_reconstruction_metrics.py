"""
Return the reconstruction metrics comparing a reconstructed diet matrix with the true one, as the array [TP, FN, TN, FP, TPR, FPR, BA, mean error, similarity]. A link is a strictly positive coefficient and the diagonal is never a link: TP counts the links present in both webs, FN the true links missing from the reconstruction, TN the off-diagonal non-links of both, FP the reconstructed links absent from the true web; TPR = TP / L_true, FPR = (L - TP) / (n (n - 1) - L_true) with L_true and L the numbers of true and reconstructed links, BA = (TPR + 1 - FPR) / 2; the mean error is the sum of the absolute differences of all n^2 entries divided by n^2, and the similarity is the sum of the entry-wise products divided by the product of the two entry-wise (Frobenius) L2 norms.

Reconstruction quality is judged first on the existence of links (recall, fallout and balanced accuracy) and then on the closeness of the diet coefficients themselves (a mean absolute error and a cosine-type similarity of the two matrices).

Returns
-------
numpy.ndarray of float64 with shape (9,): [TP, FN, TN, FP, TPR, FPR, BA, mean error, similarity].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reconstruction_metrics(true_diet: "numpy.ndarray", diet: "numpy.ndarray") -> "numpy.ndarray":
    """Return the reconstruction metrics comparing a reconstructed diet matrix with the true one, as the array [TP, FN, TN, FP, TPR, FPR, BA, mean error, similarity]. A link is a strictly positive coefficient and the diagonal is never a link: TP counts the links present in both webs, FN the true links missing from the reconstruction, TN the off-diagonal non-links of both, FP the reconstructed links absent from the true web; TPR = TP / L_true, FPR = (L - TP) / (n (n - 1) - L_true) with L_true and L the numbers of true and reconstructed links, BA = (TPR + 1 - FPR) / 2; the mean error is the sum of the absolute differences of all n^2 entries divided by n^2, and the similarity is the sum of the entry-wise products divided by the product of the two entry-wise (Frobenius) L2 norms.

    Parameters
    ----------
    true_diet : numpy.ndarray
        Square array of shape (n, n) of finite nonnegative true diet coefficients.
    diet : numpy.ndarray
        Square array of shape (n, n) of finite nonnegative reconstructed diet coefficients.

    Returns
    -------
    metrics : numpy.ndarray
        Array of shape (9,): [TP, FN, TN, FP, TPR, FPR, BA, mean error, similarity] (float64).

    Raises
    ------
    ValueError
        If the matrices are not square of the same size with at least two species, contain a non-finite or negative entry, the true web has no link or no absent link, or the reconstructed matrix is entirely zero.
    """
    return metrics

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np


def _oracle_reconstruction_metrics(true_diet: "numpy.ndarray", diet: "numpy.ndarray") -> "numpy.ndarray":
    """Metrics of Eqs 30-39: [TP, FN, TN, FP, TPR, FPR, BA, mean error, similarity] of the reconstruction against the true web."""
    Qt = np.asarray(true_diet, dtype=np.float64)
    Q = np.asarray(diet, dtype=np.float64)
    if Qt.ndim != 2 or Qt.shape[0] != Qt.shape[1] or Q.shape != Qt.shape or Qt.shape[0] < 2:
        raise ValueError("both diet matrices must be square, of the same size, with at least two species")
    if not (np.all(np.isfinite(Qt)) and np.all(np.isfinite(Q))) or np.any(Qt < 0.0) or np.any(Q < 0.0):
        raise ValueError("diet coefficients must be finite and nonnegative")
    n = Qt.shape[0]
    At = (Qt > 0.0).astype(np.float64)                            # adjacency (Eq 30)
    A = (Q > 0.0).astype(np.float64)
    off = np.ones((n, n)) - np.eye(n)                            # I_ij = 1 - delta_ij
    TP = float(np.sum(At * A))
    FN = float(np.sum(At * (off - A)))
    TN = float(np.sum((off - At) * (off - A)))
    FP = float(np.sum((off - At) * A))
    L_true, L = float(At.sum()), float(A.sum())
    if L_true == 0.0 or n * (n - 1) - L_true == 0.0:
        raise ValueError("the true web must have at least one link and at least one absent link")
    nt, nr = np.linalg.norm(Qt), np.linalg.norm(Q)
    if nr == 0.0:
        raise ValueError("the reconstructed matrix must have at least one nonzero entry")
    TPR = TP / L_true                                             # recall (Eq 35)
    FPR = (L - TP) / (n * (n - 1) - L_true)                       # fallout (Eq 36)
    BA = 0.5 * (TPR + 1.0 - FPR)                                  # balanced accuracy (Eq 37)
    mean_error = float(np.sum(np.abs(Qt - Q))) / (n * n)         # Eq 38
    similarity = float(np.sum(Qt * Q)) / (nt * nr)               # Eq 39 (entry-wise product over Frobenius norms)
    return np.array([TP, FN, TN, FP, TPR, FPR, BA, mean_error, similarity])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((12, 12))\ntrue_diet[1, 0] = true_diet[2, 0] = true_diet[3, 0] = 1.0\ntrue_diet[4, [0, 1, 2]] = [0.25, 0.45, 0.30]\ntrue_diet[5, [1, 3, 4]] = [0.30, 0.30, 0.40]\ntrue_diet[6, [2, 4, 5]] = [0.50, 0.20, 0.30]\ntrue_diet[7, [4, 5, 6]] = [0.40, 0.35, 0.25]\ntrue_diet[8, [3, 5, 6, 7]] = [0.15, 0.25, 0.30, 0.30]\ntrue_diet[9, [6, 7, 8]] = [0.35, 0.40, 0.25]\ntrue_diet[10, [7, 8, 9]] = [0.30, 0.30, 0.40]\ntrue_diet[11, [8, 9, 10]] = [0.20, 0.35, 0.45]\nreconstructed = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.19999999999999987, 0.19999999999999987, 0.19999999999999987, 0.40000000000000036, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.09629784384887809, 0.09629784384887809, 0.09629784384887809, 0.6989789254352279, 0.012127543018137753, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.3461183997760055, 0.5273602674039838, 0.12652133282001068, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0007982656079984527, 0.4614633881044116, 0.5371019413286272, 0.0006364049589628214, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.023669151744345272, 0.25382044015863003, 0.6033950047737433, 0.11911540332328124, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11269865734204665, 0.5817280116996429, 0.3055733309583104, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.010177190644722436, 0.7309458891180116, 0.25887692023726594, 0.0]])\n",
            "call": "reconstruction_metrics(true_diet, reconstructed)",
            "gold_call": "_oracle_reconstruction_metrics(true_diet, reconstructed)",
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((6, 6))\ntrue_diet[1, 0] = true_diet[2, 0] = 1.0\ntrue_diet[3, [0, 1]] = [0.3, 0.7]\ntrue_diet[4, [1, 2, 3]] = [0.2, 0.3, 0.5]\ntrue_diet[5, [2, 3, 4]] = [0.25, 0.25, 0.50]\nreconstructed = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2999999999999998, 0.3500000000000001, 0.3500000000000001, 0.0, 0.0, 0.0], [0.0, 0.25, 0.25, 0.5, 0.0, 0.0], [0.0, 0.0020677171515869385, 0.0020677171515869385, 0.760641790293409, 0.23522277540341724, 0.0]])\n",
            "call": "reconstruction_metrics(true_diet, reconstructed)",
            "gold_call": "_oracle_reconstruction_metrics(true_diet, reconstructed)",
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.4, 0.6, 0.0]])\nreconstructed = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])\n",
            "call": "reconstruction_metrics(true_diet, reconstructed)",
            "gold_call": "_oracle_reconstruction_metrics(true_diet, reconstructed)",
        },
        {
            "setup": "import numpy as np\ntrue_diet = np.zeros((3, 3))\nreconstructed = np.eye(3)\ndef run_model():\n    try:\n        reconstruction_metrics(true_diet, reconstructed)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_reconstruction_metrics(true_diet, reconstructed)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
