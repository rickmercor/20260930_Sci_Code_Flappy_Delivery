"""
Assemble the ordered arsenic-antisite CBM-depth series from acquired indices.

Arsenic antisite (As_Ga) is the technologically relevant mid-gap center retained after electronic acquisition. Temperature-resolved depth statistics require a reproducible ordering of the selected As_Ga frames before block averaging.

Returns
-------
np.ndarray shape (n_asga,), ordered As_Ga CBM-Ed depths in eV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_asga_ordered_depth_series(
    selected_indices: np.ndarray,
    defect_ids: np.ndarray,
    temperatures: np.ndarray,
    trained_hamiltonians: np.ndarray,
    asga_defect_id: int = 4,
) -> np.ndarray:
    """Return ordered As_Ga CBM-Ed depths for acquired structures.

    Parameters
    ----------
    selected_indices : np.ndarray
        Selected pool indices from balanced acquisition.
    defect_ids : np.ndarray
        Shape (n_pool,). Defect labels.
    temperatures : np.ndarray
        Shape (n_pool,). Temperatures in K.
    trained_hamiltonians : np.ndarray
        Shape (n_pool, 6, 6). Trained near-gap Hamiltonians in eV.
    asga_defect_id : int
        Arsenic-antisite label (default 4).

    Returns
    -------
    depths : np.ndarray
        1D array of CBM-Ed values in the required order.
    """
    return np.asarray([], dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_asga_ordered_depth_series(
    selected_indices: np.ndarray,
    defect_ids: np.ndarray,
    temperatures: np.ndarray,
    trained_hamiltonians: np.ndarray,
    asga_defect_id: int = 4,
) -> np.ndarray:
    sel = np.asarray(selected_indices, dtype=int).reshape(-1)
    d = np.asarray(defect_ids).reshape(-1)
    t = np.asarray(temperatures, dtype=float).reshape(-1)
    H = np.asarray(trained_hamiltonians, dtype=float)
    if H.ndim != 3 or H.shape[1:] != (6, 6):
        raise ValueError("trained_hamiltonians must have shape (n_pool, 6, 6)")
    if not (d.size == t.size == H.shape[0]):
        raise ValueError("defect_ids, temperatures, trained_hamiltonians length mismatch")
    if sel.size < 1:
        raise ValueError("selected_indices must be non-empty")
    if np.any(sel < 0) or np.any(sel >= H.shape[0]):
        raise ValueError("selected index out of range")

    keep = sel[d[sel] == int(asga_defect_id)]
    if keep.size < 1:
        raise ValueError("no As_Ga structures in the selection")

    order = np.lexsort((-keep, t[keep]))
    keep_ordered = keep[order]

    table = compute_gap_spectrum_table(H[keep_ordered])
    if table.ndim == 1:
        table = table.reshape(1, 4)
    return table[:, 3].astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                "import numpy as np\n"
                "selected_indices = np.array([0.0, 1.0, 2.0, 3.0])\n"
                "defect_ids = np.array([0, 4, 4, 1])\n"
                "temperatures = np.array([100.0, 500.0, 100.0, 100.0])\n"
                "eigs = np.array([\n"
                " [-2.0, -1.5, -1.0, 0.0, 0.7, 1.2],\n"
                " [-2.0, -1.5, -1.0, 0.1, 0.7, 1.2],\n"
                " [-2.0, -1.5, -1.0, 0.2, 0.7, 1.2],\n"
                " [-2.0, -1.5, -1.0, 0.05, 0.7, 1.2],\n"
                "])\n"
                "trained_hamiltonians = np.stack([np.diag(e) for e in eigs])\n"
                "asga_defect_id = 4\n"
            ),
            "call": "compute_asga_ordered_depth_series(selected_indices, defect_ids, temperatures, trained_hamiltonians, asga_defect_id)",
            "gold_call": "_oracle_compute_asga_ordered_depth_series(selected_indices, defect_ids, temperatures, trained_hamiltonians, asga_defect_id)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "selected_indices = np.array([1.0])\n"
                "defect_ids = np.array([4, 4])\n"
                "temperatures = np.array([100.0, 100.0])\n"
                "trained_hamiltonians = np.stack([\n"
                " np.diag([-2.0, -1.5, -1.0, -0.1, 0.5, 1.0]),\n"
                " np.diag([-2.0, -1.5, -1.0, 0.0, 0.5, 1.0]),\n"
                "])\n"
            ),
            "call": "compute_asga_ordered_depth_series(selected_indices, defect_ids, temperatures, trained_hamiltonians)",
            "gold_call": "_oracle_compute_asga_ordered_depth_series(selected_indices, defect_ids, temperatures, trained_hamiltonians)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "h = np.zeros((4, 3, 6, 6))\n"
                "base = np.array([\n"
                " [-2.0, -1.5, -1.0, 0.0, 0.5, 1.0],\n"
                " [-2.0, -1.5, -1.0, 0.1, 0.5, 1.0],\n"
                " [-2.0, -1.5, -1.0, -0.05, 0.5, 1.0],\n"
                " [-2.0, -1.5, -1.0, 0.05, 0.5, 1.0],\n"
                "])\n"
                "trained_hamiltonians = []\n"
                "for i, sc in enumerate([2.0, 1.0, 3.0, 1.5]):\n"
                "    Href = np.diag(base[i])\n"
                "    Href[3, 4] = Href[4, 3] = 0.05 * sc\n"
                "    trained_hamiltonians.append(Href)\n"
                "    for m, amp in enumerate((-1.0, 0.0, 1.0)):\n"
                "        h[i, m] = Href + amp * 0.01 * sc * np.eye(6)\n"
                "trained_hamiltonians = np.asarray(trained_hamiltonians, dtype=float)\n"
                "scores = compute_hamiltonian_disagreement_scores(h)\n"
                "defect_ids = np.array([4, 4, 4, 4])\n"
                "temperatures = np.array([100.0, 100.0, 500.0, 500.0])\n"
                "selected_indices = compute_balanced_acquisition_indices(scores, defect_ids, temperatures, 1)\n"
                "asga_defect_id = 4\n"
            ),
            "call": "compute_asga_ordered_depth_series(selected_indices, defect_ids, temperatures, trained_hamiltonians, asga_defect_id)",
            "gold_call": "_oracle_compute_asga_ordered_depth_series(selected_indices, defect_ids, temperatures, trained_hamiltonians, asga_defect_id)",
        },
    ]
