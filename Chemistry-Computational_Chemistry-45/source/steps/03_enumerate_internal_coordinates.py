"""
Enumerate canonical bond, angle, and dihedral coordinates from a molecular graph. The first column stores coordinate arity: 2 for a bond, 3 for an angle, and 4 for a dihedral.

Graph-derived coordinates convert molecular connectivity into local geometric descriptors without requiring a Cartesian alignment. Rows use [arity, a, b, c, d], with unused atom columns padded by -1.

Returns
-------
Integer matrix of canonical graph coordinates with five columns; the first is arity 2, 3, or 4 for a bond, angle, or dihedral.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def enumerate_internal_coordinates(bonds: np.ndarray, n_atoms: int) -> np.ndarray:
    """Enumerate canonical internal coordinates from graph edges.

    Parameters
    ----------
    bonds : np.ndarray
        Zero-based undirected graph edges with shape (n_bonds, 2).
    n_atoms : int
        Number of atoms in the graph.

    Returns
    -------
    np.ndarray
        Sorted rows [arity, a, b, c, d], where arity is 2 for a bond, 3 for an angle, or 4 for a dihedral; unused atom columns are padded with -1.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_enumerate_internal_coordinates(
    bonds: np.ndarray,
    n_atoms: int,
) -> np.ndarray:
    """Reference graph-path enumeration."""
    import numpy as np
    from itertools import combinations

    edges = np.asarray(bonds)

    if not isinstance(n_atoms, (int, np.integer)) or int(n_atoms) < 2:
        raise ValueError("n_atoms must be an integer of at least two")
    n_atoms = int(n_atoms)

    if edges.ndim != 2 or edges.shape[0] < 1 or edges.shape[1] != 2:
        raise ValueError("bonds must have shape (n_bonds, 2)")
    if not np.all(np.isfinite(edges)) or not np.all(edges == np.floor(edges)):
        raise ValueError("bond indices must be finite integers")

    edges = edges.astype(int)
    if (
        np.any(edges < 0)
        or np.any(edges >= n_atoms)
        or np.any(edges[:, 0] == edges[:, 1])
    ):
        raise ValueError("bond indices are out of range or self-connected")

    edge_set = {
        tuple(sorted(map(int, edge)))
        for edge in edges
    }
    neighbors = {atom: set() for atom in range(n_atoms)}

    for left, right in edge_set:
        neighbors[left].add(right)
        neighbors[right].add(left)

    rows = [
        (2, left, right, -1, -1)
        for left, right in edge_set
    ]

    for center in range(n_atoms):
        for left, right in combinations(sorted(neighbors[center]), 2):
            rows.append((3, left, center, right, -1))

    dihedrals = set()
    for center_left, center_right in edge_set:
        for left in neighbors[center_left] - {center_right}:
            for right in neighbors[center_right] - {center_left}:
                if left != right:
                    path = (left, center_left, center_right, right)
                    dihedrals.add(min(path, tuple(reversed(path))))

    rows.extend((4, *path) for path in dihedrals)
    return np.asarray(sorted(rows), dtype=int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
bonds = np.array([[0,1],[1,2],[2,3]])
n_atoms = 4""",
            "call": "enumerate_internal_coordinates(bonds, n_atoms)",
            "gold_call": "_oracle_enumerate_internal_coordinates(bonds, n_atoms)",
        },
        {
            "setup": """import numpy as np
bonds = np.array([[0,1],[1,2],[2,3],[3,0],[1,4],[4,5]])
n_atoms = 7""",
            "call": "enumerate_internal_coordinates(bonds, n_atoms)",
            "gold_call": "_oracle_enumerate_internal_coordinates(bonds, n_atoms)",
        },
        {
            "setup": """import numpy as np
bonds = np.array([[2,1],[1,0],[0,1],[1,2]])
n_atoms = 3""",
            "call": "enumerate_internal_coordinates(bonds, n_atoms)",
            "gold_call": "_oracle_enumerate_internal_coordinates(bonds, n_atoms)",
        },
    ]
