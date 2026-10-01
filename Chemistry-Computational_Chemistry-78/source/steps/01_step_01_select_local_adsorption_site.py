"""
Select the local adsorption site used by the source paper's GNN.

The local site is the union of cutoff spheres of radius 4.5 Å
centered on the adsorbate atoms that bind the surface. Every bonded
adsorbate atom is kept, and every other atom whose distance to at
least one bonded adsorbate atom is at most 4.5 Å is kept. The
result is a substructure: the GNN never sees the rest of the slab.

This stage only selects atoms. It does not build the neighbor list
and does not attach node features.

Returns
-------
return node_index
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def select_local_adsorption_site(
    atomic_numbers: np.ndarray,
    coordinates: np.ndarray,
    bonded_adsorbate: np.ndarray,
) -> np.ndarray:
    """
    Return the atom indices that form the local adsorption site.

    Parameters
    ----------
    atomic_numbers : np.ndarray
        Nuclear charges, shape (n_atoms,).
    coordinates : np.ndarray
        Cartesian coordinates in Angstrom, shape (n_atoms, 3).
    bonded_adsorbate : np.ndarray
        Indices of the adsorbate atoms that bind the surface.

    Returns
    -------
    np.ndarray
        Sorted unique local-site indices.

    Raises
    ------
    ValueError
        If coordinates are not (n_atoms, 3), bonded indices are
        invalid, distinct atoms coincide, or numerical values are
        invalid.
    """
    return node_index

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_select_local_adsorption_site(
    atomic_numbers: np.ndarray,
    coordinates: np.ndarray,
    bonded_adsorbate: np.ndarray,
) -> np.ndarray:
    cutoff = 4.5
    allowed_z = np.array(
        [1, 6, 7, 8, 13, 26, 28, 29, 30, 31, 47, 78, 79], dtype=int
    )
    z = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    r = np.asarray(coordinates, dtype=float)
    bonded = np.asarray(bonded_adsorbate, dtype=int).reshape(-1)
    if z.size == 0:
        raise ValueError("at least one atom is required")
    if r.ndim != 2 or r.shape[1] != 3 or r.shape[0] != z.size:
        raise ValueError("coordinates must have shape (n_atoms, 3)")
    if not np.all(np.isfinite(r)):
        raise ValueError("coordinates contain non-finite values")
    if not np.all(np.isin(z, allowed_z)):
        raise ValueError("atomic numbers must belong to the paper element set")
    if bonded.size == 0:
        raise ValueError("at least one bonded adsorbate atom is required")
    if np.any(bonded < 0) or np.any(bonded >= z.size):
        raise ValueError("bonded adsorbate indices are out of range")
    if np.unique(bonded).size != bonded.size:
        raise ValueError("bonded adsorbate indices must be unique")
    pair = np.linalg.norm(r[:, None, :] - r[None, :, :], axis=-1)
    if np.any(pair[np.triu_indices(z.size, 1)] < 1e-8):
        raise ValueError("distinct atoms must not coincide")
    keep = np.zeros(z.size, dtype=bool)
    keep[bonded] = True
    for idx in bonded:
        keep |= pair[:, idx] <= cutoff
    selected = np.flatnonzero(keep)
    if selected.size < 2:
        raise ValueError("local adsorption site must contain at least two atoms")
    return selected

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np

atomic_numbers = np.array([28, 28, 31, 28, 8, 6, 8, 1, 28])
coordinates = np.array([
    [0.00, 0.00, 0.00],
    [2.50, 0.00, 0.00],
    [1.25, 2.16, 0.00],
    [1.25, 0.72, -2.05],
    [0.15, 0.10, 1.82],
    [1.25, 0.10, 2.48],
    [2.35, 0.10, 1.82],
    [1.25, 0.10, 3.56],
    [8.50, 8.50, 8.50],
])
bonded_adsorbate = np.array([4, 6])

def run_model():
    return select_local_adsorption_site(
        atomic_numbers.copy(), coordinates.copy(), bonded_adsorbate.copy()
    )

def run_gold():
    return _oracle_select_local_adsorption_site(
        atomic_numbers.copy(), coordinates.copy(), bonded_adsorbate.copy()
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

atomic_numbers = np.array([28, 8, 28])
coordinates = np.array([
    [0.0, 0.0, 0.0],
    [1.8, 0.0, 0.0],
    [7.0, 0.0, 0.0],
])
bonded_adsorbate = np.array([1])

def run_model():
    return select_local_adsorption_site(
        atomic_numbers.copy(), coordinates.copy(), bonded_adsorbate.copy()
    )

def run_gold():
    return _oracle_select_local_adsorption_site(
        atomic_numbers.copy(), coordinates.copy(), bonded_adsorbate.copy()
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

atomic_numbers = np.array([28, 8])
coordinates = np.array([
    [0.0, 0.0, 0.0],
    [1.8, 0.0, 0.0],
])
bonded_adsorbate = np.array([4])

def run_model():
    try:
        select_local_adsorption_site(
            atomic_numbers.copy(), coordinates.copy(), bonded_adsorbate.copy()
        )
        return 0
    except ValueError:
        return 1

def run_gold():
    try:
        _oracle_select_local_adsorption_site(
            atomic_numbers.copy(), coordinates.copy(), bonded_adsorbate.copy()
        )
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

atomic_numbers = np.array([28, 8])
coordinates = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])
bonded_adsorbate = np.array([1])

def run_model():
    try:
        select_local_adsorption_site(
            atomic_numbers.copy(), coordinates.copy(), bonded_adsorbate.copy()
        )
        return 0
    except ValueError:
        return 1

def run_gold():
    try:
        _oracle_select_local_adsorption_site(
            atomic_numbers.copy(), coordinates.copy(), bonded_adsorbate.copy()
        )
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

atomic_numbers = np.array([28, 8, 28, 28])
coordinates = np.array([
    [0.00, 0.0, 0.0],
    [1.80, 0.0, 0.0],
    [1.80 + 4.49, 0.0, 0.0],
    [1.80 + 4.51, 0.0, 0.0],
])
bonded_adsorbate = np.array([1])

def run_model():
    return select_local_adsorption_site(
        atomic_numbers.copy(), coordinates.copy(), bonded_adsorbate.copy()
    )

def run_gold():
    return _oracle_select_local_adsorption_site(
        atomic_numbers.copy(), coordinates.copy(), bonded_adsorbate.copy()
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
