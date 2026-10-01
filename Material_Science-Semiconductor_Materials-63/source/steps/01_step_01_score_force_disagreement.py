"""
Score per-frame force-model disagreement for defective-supercell active learning.

Machine-learned force fields for defective semiconductors are refined by active learning: ensemble spread among force predictors marks which MD frames most need DFT labels

Returns
-------
np.ndarray shape (n_configs,), force disagreement scores in eV/Å
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_force_disagreement_scores(force_ensembles: np.ndarray) -> np.ndarray:
    """Return per-configuration force-ensemble disagreement scores.

    Parameters
    ----------
    force_ensembles : np.ndarray
        Shape (n_configs, n_models, n_atoms, 3). Forces in eV/Å.

    Returns
    -------
    scores : np.ndarray
        Shape (n_configs,). Disagreement scores in eV/Å.
    """
    return np.asarray([], dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_force_disagreement_scores(force_ensembles: np.ndarray) -> np.ndarray:
    import numpy as np
    arr = np.asarray(force_ensembles, dtype=float)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError("force_ensembles must have shape (n_configs, n_models, n_atoms, 3)")
    if arr.shape[1] < 2:
        raise ValueError("at least two ensemble members are required")
    if arr.shape[0] < 1 or arr.shape[2] < 1:
        raise ValueError("n_configs and n_atoms must be positive")
    std = np.std(arr, axis=1, ddof=0)
    return np.max(np.linalg.norm(std, axis=-1), axis=-1).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                "import numpy as np\n"
                "force_ensembles = np.zeros((2, 3, 2, 3))\n"
                "force_ensembles[1, 1] = [[0.03, 0.0, 0.0], [0.0, 0.0, 0.0]]\n"
                "force_ensembles[1, 2] = [[-0.03, 0.0, 0.0], [0.0, 0.0, 0.0]]\n"
            ),
            "call": "compute_force_disagreement_scores(force_ensembles)",
            "gold_call": "_oracle_compute_force_disagreement_scores(force_ensembles)",
        },
        {
            "setup": "import numpy as np\nforce_ensembles = np.ones((1, 2, 1, 3)) * 0.1\n",
            "call": "compute_force_disagreement_scores(force_ensembles)",
            "gold_call": "_oracle_compute_force_disagreement_scores(force_ensembles)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "force_ensembles = np.zeros((1, 3, 3, 3))\n"
                "force_ensembles[0, 0, 2] = [1.0, 0.0, 0.0]\n"
                "force_ensembles[0, 1, 2] = [0.0, 1.0, 0.0]\n"
                "force_ensembles[0, 2, 2] = [0.0, 0.0, 1.0]\n"
            ),
            "call": "compute_force_disagreement_scores(force_ensembles)",
            "gold_call": "_oracle_compute_force_disagreement_scores(force_ensembles)",
        },
    ]
