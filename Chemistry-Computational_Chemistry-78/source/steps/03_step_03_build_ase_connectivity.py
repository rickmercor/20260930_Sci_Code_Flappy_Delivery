"""
Build the undirected first-neighbor graph of the local adsorption site.

Two atoms are neighbors only if their distance is smaller than the Atomic Simulation Environment natural cutoff on the two covalent radii, that is the sum of their covalent radii multiplied by 1.1. Self-loops are not added.

Returns
-------
return edge_index
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_ase_connectivity(
    atomic_numbers: np.ndarray,
    coordinates: np.ndarray,
    node_index: np.ndarray,
) -> np.ndarray:
    """
    Return the bidirectional first-neighbor edge index.

    Parameters
    ----------
    atomic_numbers : np.ndarray
        Nuclear charges, shape (n_atoms,).
    coordinates : np.ndarray
        Cartesian coordinates in Angstrom, shape (n_atoms, 3).
    node_index : np.ndarray
        Local-site atom indices. Edge endpoints are numbered in this
        selected order, starting at 0.

    Returns
    -------
    np.ndarray
        Directed edges with shape (2, 2 n_undirected), stored as
        (src, dst).

    Raises
    ------
    ValueError
        If coordinates are not (n_atoms, 3), node indices are out of
        range, numerical values are invalid, or no pair satisfies
        the cutoff.
    """
    return edge_index

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_ase_connectivity(
    atomic_numbers: np.ndarray,
    coordinates: np.ndarray,
    node_index: np.ndarray,
) -> np.ndarray:
    ase_mult = 1.1
    covalent_radii = {
        1: 0.31,
        6: 0.76,
        7: 0.71,
        8: 0.66,
        13: 1.21,
        26: 1.32,
        28: 1.24,
        29: 1.32,
        30: 1.22,
        31: 1.22,
        47: 1.45,
        78: 1.36,
        79: 1.36,
    }
    z = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    r = np.asarray(coordinates, dtype=float)
    node_index = np.asarray(node_index, dtype=int).reshape(-1)
    if r.ndim != 2 or r.shape[1] != 3 or r.shape[0] != z.size:
        raise ValueError("coordinates must have shape (n_atoms, 3)")
    if not np.all(np.isfinite(r)):
        raise ValueError("coordinates contain non-finite values")
    if node_index.size < 2:
        raise ValueError("at least two local-site atoms are required")
    if np.any(node_index < 0) or np.any(node_index >= z.size):
        raise ValueError("node_index is out of range")
    if np.unique(node_index).size != node_index.size:
        raise ValueError("node_index must be unique")
    z_sel = z[node_index]
    r_sel = r[node_index]
    for zi in z_sel:
        if int(zi) not in covalent_radii:
            raise ValueError("atomic numbers must belong to the paper element set")
    radii = np.array([covalent_radii[int(zi)] for zi in z_sel], dtype=float)
    dist = np.linalg.norm(r_sel[:, None, :] - r_sel[None, :, :], axis=-1)
    undirected = []
    n = z_sel.size
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] < ase_mult * (radii[i] + radii[j]):
                undirected.append((i, j))
    if not undirected:
        raise ValueError("ASE neighbor list is empty")
    src = []
    dst = []
    for i, j in undirected:
        src.extend([i, j])
        dst.extend([j, i])
    return np.vstack([np.array(src, dtype=int), np.array(dst, dtype=int)])

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
node_index = np.arange(8)

def run_model():
    edges = build_ase_connectivity(atomic_numbers.copy(), coordinates.copy(), node_index.copy())
    pairs = {tuple(sorted((int(s), int(d)))) for s, d in zip(edges[0], edges[1])}
    return np.array(sorted(pairs)).reshape(-1)

def run_gold():
    edges = _oracle_build_ase_connectivity(atomic_numbers.copy(), coordinates.copy(), node_index.copy())
    pairs = {tuple(sorted((int(s), int(d)))) for s, d in zip(edges[0], edges[1])}
    return np.array(sorted(pairs)).reshape(-1)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

atomic_numbers = np.array([6, 1])
coordinates = np.array([
    [0.00, 0.0, 0.0],
    [1.08, 0.0, 0.0],
])
node_index = np.array([0, 1])

def run_model():
    edges = build_ase_connectivity(atomic_numbers.copy(), coordinates.copy(), node_index.copy())
    return np.array([edges.shape[1], int(np.any(edges[0] != edges[1]))])

def run_gold():
    edges = _oracle_build_ase_connectivity(atomic_numbers.copy(), coordinates.copy(), node_index.copy())
    return np.array([edges.shape[1], int(np.any(edges[0] != edges[1]))])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

atomic_numbers = np.array([28, 8])
coordinates = np.array([
    [0.0, 0.0, 0.0],
    [3.5, 0.0, 0.0],
])
node_index = np.array([0, 1])

def run_model():
    try:
        build_ase_connectivity(atomic_numbers.copy(), coordinates.copy(), node_index.copy())
        return 0
    except ValueError:
        return 1

def run_gold():
    try:
        _oracle_build_ase_connectivity(atomic_numbers.copy(), coordinates.copy(), node_index.copy())
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

atomic_numbers = np.array([28, 28, 8])
coordinates = np.array([
    [0.00, 0.0, 0.0],
    [2.50, 0.0, 0.0],
    [0.15, 0.0, 1.82],
])
node_index = np.array([0, 1, 2])

def run_model():
    edges = build_ase_connectivity(atomic_numbers.copy(), coordinates.copy(), node_index.copy())
    both = 0
    for i, j in ((0, 1), (0, 2)):
        fwd = np.any((edges[0] == i) & (edges[1] == j))
        rev = np.any((edges[0] == j) & (edges[1] == i))
        both += int(fwd and rev)
    self_loops = int(np.any(edges[0] == edges[1]))
    return np.array([both, self_loops, edges.shape[1]])

def run_gold():
    edges = _oracle_build_ase_connectivity(atomic_numbers.copy(), coordinates.copy(), node_index.copy())
    both = 0
    for i, j in ((0, 1), (0, 2)):
        fwd = np.any((edges[0] == i) & (edges[1] == j))
        rev = np.any((edges[0] == j) & (edges[1] == i))
        both += int(fwd and rev)
    self_loops = int(np.any(edges[0] == edges[1]))
    return np.array([both, self_loops, edges.shape[1]])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
