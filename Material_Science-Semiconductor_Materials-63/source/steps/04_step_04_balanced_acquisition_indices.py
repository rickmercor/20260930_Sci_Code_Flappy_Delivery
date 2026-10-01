"""
Acquire a balanced active-learning index set over defect and temperature classes

Electronic active learning on multi-defect, multi-temperature pools must avoid concentrating labels on one chemistry or one thermal regime.

Returns
-------
np.ndarray shape (n_selected,), selected pool indices as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_balanced_acquisition_indices(
    scores: np.ndarray,
    defect_ids: np.ndarray,
    temperatures: np.ndarray,
    n_per_group: int,
) -> np.ndarray:
    """Return selected pool indices for the balanced acquisition schedule.

    Parameters
    ----------
    scores : np.ndarray
        Shape (n_configs,). Hamiltonian disagreement scores.
    defect_ids : np.ndarray
        Shape (n_configs,). Integer defect labels.
    temperatures : np.ndarray
        Shape (n_configs,). Temperatures in K.
    n_per_group : int
        Per-class acquisition count for this instance. Equal scores within a
        class are ranked by ascending original pool index.

    Returns
    -------
    indices : np.ndarray
        1D float array of selected indices.
    """
    return np.asarray([], dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_balanced_acquisition_indices(
    
    scores: np.ndarray,
    defect_ids: np.ndarray,
    temperatures: np.ndarray,
    n_per_group: int,
) -> np.ndarray:
    u = np.asarray(scores, dtype=float).reshape(-1)
    d = np.asarray(defect_ids).reshape(-1)
    t = np.asarray(temperatures, dtype=float).reshape(-1)
    if not (u.size == d.size == t.size) or u.size < 1:
        raise ValueError("scores, defect_ids, temperatures length mismatch")
    if int(n_per_group) < 1:
        raise ValueError("n_per_group must be >= 1")
    if np.any(~np.isfinite(u)) or np.any(~np.isfinite(t)):
        raise ValueError("scores and temperatures must be finite")
    selected: list[int] = []
    for defect in sorted(set(d.tolist())):
        for temperature in sorted(set(t.tolist())):
            idxs = np.where((d == defect) & np.isclose(t, float(temperature)))[0]
            if idxs.size < int(n_per_group):
                raise ValueError("insufficient candidates in a (defect, temperature) group")
            order = idxs[np.argsort(-u[idxs], kind="mergesort")]
            selected.extend(int(i) for i in order[: int(n_per_group)])
    return np.asarray(selected, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            
            "setup": (
                "import numpy as np\n"
                "scores = np.array([1.0, 4.0, 2.0, 5.0, 3.0, 6.0])\n"
                "defect_ids = np.array([0, 0, 0, 1, 1, 1])\n"
                "temperatures = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])\n"
                "n_per_group = 2\n"
            ),
            "call": "compute_balanced_acquisition_indices(scores, defect_ids, temperatures, n_per_group)",
            "gold_call": "_oracle_compute_balanced_acquisition_indices(scores, defect_ids, temperatures, n_per_group)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "scores = np.array([0.9, 0.1, 0.8, 0.2])\n"
                "defect_ids = np.array([0, 0, 0, 0])\n"
                "temperatures = np.array([100.0, 100.0, 500.0, 500.0])\n"
                "n_per_group = 1\n"
            ),
            "call": "compute_balanced_acquisition_indices(scores, defect_ids, temperatures, n_per_group)",
            "gold_call": "_oracle_compute_balanced_acquisition_indices(scores, defect_ids, temperatures, n_per_group)",
        },
        {
            
            "setup": (
                "import numpy as np\n"
                "scores = np.array([0.2, 0.2, 0.2, 0.2])\n"
                "defect_ids = np.array([2, 2, 2, 2])\n"
                "temperatures = np.array([300.0, 300.0, 300.0, 300.0])\n"
                "n_per_group = 3\n"
            ),
            "call": "compute_balanced_acquisition_indices(scores, defect_ids, temperatures, n_per_group)",
            "gold_call": "_oracle_compute_balanced_acquisition_indices(scores, defect_ids, temperatures, n_per_group)",
        },
        {
           
            "setup": (
                "import numpy as np\n"
                "scores = np.array([0.5, 9.0, 0.4, 8.0, 7.0, 0.3, 6.0, 0.2])\n"
                "defect_ids = np.array([1, 0, 1, 0, 1, 0, 1, 0])\n"
                "temperatures = np.array([500.0, 100.0, 100.0, 500.0, 500.0, 100.0, 100.0, 500.0])\n"
                "n_per_group = 1\n"
            ),
            "call": "compute_balanced_acquisition_indices(scores, defect_ids, temperatures, n_per_group)",
            "gold_call": "_oracle_compute_balanced_acquisition_indices(scores, defect_ids, temperatures, n_per_group)",
        },
        {
            
            "setup": (
                "import numpy as np\n"
                "scores = np.array([3.0, 5.0, 5.0, 4.0, 0.1, 0.2, 1.0, 2.0, 8.0, 7.0])\n"
                "defect_ids = np.array([0, 0, 0, 0, 0, 0, 3, 3, 3, 3])\n"
                "temperatures = np.array([100.0, 100.0, 100.0, 100.0, 200.0, 200.0, 100.0, 100.0, 200.0, 200.0])\n"
                "n_per_group = 2\n"
            ),
            "call": "compute_balanced_acquisition_indices(scores, defect_ids, temperatures, n_per_group)",
            "gold_call": "_oracle_compute_balanced_acquisition_indices(scores, defect_ids, temperatures, n_per_group)",
        },
    ]
