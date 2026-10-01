"""
Score per-structure Hamiltonian-model disagreement for electronic AL

After force-field MD, electronic properties of defects are learned from DFT Hamiltonians. Ensemble disagreement among Hamiltonian models is used to decide which defective geometries most need new labels

Returns
-------
np.ndarray shape (n_configs,), Hamiltonian disagreement scores in eV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_hamiltonian_disagreement_scores(h_ensembles: np.ndarray) -> np.ndarray:
    """Return per-structure Hamiltonian ensemble disagreement scores.

    

    Parameters
    ----------
    h_ensembles : np.ndarray
        Shape (n_configs, n_models, n_orb, n_orb). Hamiltonians in eV.

    Returns
    -------
    scores : np.ndarray
        Shape (n_configs,). Disagreement scores in eV.
    """
    return np.asarray([], dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_hamiltonian_disagreement_scores(h_ensembles: np.ndarray) -> np.ndarray:
    import numpy as np
    arr = np.asarray(h_ensembles, dtype=float)
    if arr.ndim != 4:
        raise ValueError("h_ensembles must have shape (n_configs, n_models, n_orb, n_orb)")
    if arr.shape[1] < 2:
        raise ValueError("at least two ensemble members are required")
    if arr.shape[0] < 1 or arr.shape[2] != arr.shape[3] or arr.shape[2] < 1:
        raise ValueError("invalid Hamiltonian ensemble shape")
    std = np.std(arr, axis=1, ddof=0)
    return np.sqrt((std**2).mean(axis=(-2, -1))).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
           
            "setup": (
                "import numpy as np\n"
                "h = np.zeros((2, 3, 2, 2))\n"
                "h[1, 1] = [[0.03, 0.0], [0.0, -0.03]]\n"
                "h[1, 2] = [[-0.03, 0.0], [0.0, 0.03]]\n"
            ),
            "call": "compute_hamiltonian_disagreement_scores(h)",
            "gold_call": "_oracle_compute_hamiltonian_disagreement_scores(h)",
        },
        {
            "setup": "import numpy as np\nh = np.ones((1, 2, 1, 1)) * 0.5\n",
            "call": "compute_hamiltonian_disagreement_scores(h)",
            "gold_call": "_oracle_compute_hamiltonian_disagreement_scores(h)",
        },
        {
            
            "setup": (
                "import numpy as np\n"
                "h = np.zeros((1, 3, 2, 2))\n"
                "h[0, 0, 0, 1] = 1.0\n"
                "h[0, 0, 1, 0] = 1.0\n"
                "h[0, 1, 0, 1] = -1.0\n"
                "h[0, 1, 1, 0] = -1.0\n"
            ),
            "call": "compute_hamiltonian_disagreement_scores(h)",
            "gold_call": "_oracle_compute_hamiltonian_disagreement_scores(h)",
        },
        {
           
            "setup": (
                "import numpy as np\n"
                "h = np.zeros((1, 2, 1, 1))\n"
                "h[0, 0, 0, 0] = 0.0\n"
                "h[0, 1, 0, 0] = 0.02\n"
            ),
            "call": "compute_hamiltonian_disagreement_scores(h)",
            "gold_call": "_oracle_compute_hamiltonian_disagreement_scores(h)",
        },
        {
            
            "setup": (
                "import numpy as np\n"
                "h = np.zeros((3, 3, 2, 2))\n"
                "h[0, 0] = [[0.10, 0.02], [0.02, -0.04]]\n"
                "h[0, 1] = [[0.00, 0.00], [0.00, 0.00]]\n"
                "h[0, 2] = [[-0.10, -0.02], [-0.02, 0.04]]\n"
                "h[1, 0] = [[0.01, 0.00], [0.00, 0.01]]\n"
                "h[1, 1] = [[0.01, 0.00], [0.00, 0.01]]\n"
                "h[1, 2] = [[0.01, 0.00], [0.00, 0.01]]\n"
                "h[2, 0, 0, 1] = 0.05\n"
                "h[2, 0, 1, 0] = 0.05\n"
                "h[2, 2, 0, 1] = -0.05\n"
                "h[2, 2, 1, 0] = -0.05\n"
            ),
            "call": "compute_hamiltonian_disagreement_scores(h)",
            "gold_call": "_oracle_compute_hamiltonian_disagreement_scores(h)",
        },
    ]
