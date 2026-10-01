"""
Resolve valence, mid-gap, and conduction levels from truncated near-gap spectra.

Point-defect electronics in semiconductors are interpreted relative to the host gap. A truncated near-gap Hamiltonian for a defective supercell mixes bulk-like states with a localized mid-gap level; the observable of interest here is how deep that level sits below the conduction-band edge

Returns
-------
np.ndarray shape (4,) or (n, 4): [VBM, Ed, CBM, depth] in eV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_gap_spectrum_table(hamiltonians: np.ndarray) -> np.ndarray:
    """Map near-gap Hamiltonians to band edges, defect level, and depth.

    Parameters
    ----------
    hamiltonians : np.ndarray
        Shape (6, 6) or (n, 6, 6). Symmetric near-gap Hamiltonians in eV.

    Returns
    -------
    table : np.ndarray
        Shape (4,) or (n, 4) with columns [VBM, Ed, CBM, CBM-Ed] in eV.
    """
    return np.asarray([], dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_gap_spectrum_table(hamiltonians: np.ndarray) -> np.ndarray:
    arr = np.asarray(hamiltonians, dtype=float)
    single = arr.ndim == 2
    if single:
        if arr.shape != (6, 6):
            raise ValueError("single Hamiltonian must have shape (6, 6)")
        arr2 = arr.reshape(1, 6, 6)
    elif arr.ndim == 3:
        arr2 = arr
    else:
        raise ValueError("hamiltonians must have shape (6, 6) or (n, 6, 6)")
    if arr2.shape[1:] != (6, 6):
        raise ValueError("each Hamiltonian must be 6x6")
    if np.any(~np.isfinite(arr2)):
        raise ValueError("hamiltonians must be finite")
    if np.any(np.abs(arr2 - np.swapaxes(arr2, -1, -2)) > 1e-10):
        raise ValueError("hamiltonians must be symmetric")
    e = np.linalg.eigvalsh(arr2)
    e = np.sort(e, axis=1)
    vbm, ed, cbm = e[:, 2], e[:, 3], e[:, 4]
    if np.any(~((vbm < ed) & (ed < cbm))):
        raise ValueError("mid-gap level is not strictly inside the host gap")
    if np.any(cbm - vbm <= 0.0):
        raise ValueError("non-positive host gap")
    out = np.stack([vbm, ed, cbm, cbm - ed], axis=1)
    return out[0] if single else out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                "import numpy as np\n"
                "hamiltonians = np.diag([-2.0, -1.5, -1.0, 0.1, 0.7, 1.2])\n"
            ),
            "call": "compute_gap_spectrum_table(hamiltonians)",
            "gold_call": "_oracle_compute_gap_spectrum_table(hamiltonians)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "d = np.diag([0.7, -1.0, 1.2, -2.0, 0.0, -1.5])\n"
                "hamiltonians = d.reshape(1, 6, 6)\n"
            ),
            "call": "compute_gap_spectrum_table(hamiltonians)",
            "gold_call": "_oracle_compute_gap_spectrum_table(hamiltonians)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "hamiltonians = np.stack([\n"
                " np.diag([-2.0, -1.5, -1.0, -0.2, 0.5, 1.0]),\n"
                " np.diag([-2.0, -1.5, -1.0, 0.0, 0.5, 1.0]),\n"
                "])\n"
            ),
            "call": "compute_gap_spectrum_table(hamiltonians)",
            "gold_call": "_oracle_compute_gap_spectrum_table(hamiltonians)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "hamiltonians = np.diag([-2.0, -1.5, -1.0, 0.0, 0.5, 1.0]).astype(float)\n"
                "hamiltonians[2, 3] = hamiltonians[3, 2] = 0.22\n"
                "hamiltonians[3, 4] = hamiltonians[4, 3] = 0.18\n"
            ),
            "call": "compute_gap_spectrum_table(hamiltonians)",
            "gold_call": "_oracle_compute_gap_spectrum_table(hamiltonians)",
        },
    ]
