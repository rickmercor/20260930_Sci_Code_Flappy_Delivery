"""
Classify graph-coordinate changes using status 0 for unreported, 1 for a primary bond, 2 for a coupled-proton bond, 3 for an independent angle, and 4 for a representative independent dihedral.

GraphRC prioritizes bond rearrangements, links the second leg of proton transfer at a lower threshold, removes dependent coordinates, and reduces equivalent torsions to one representative. Every comparison is inclusive: a change equal to its threshold counts as significant, for all four thresholds. If the first screen is empty, halve the three primary thresholds once while leaving the coupled-proton threshold unchanged. For each dihedral axis, choose the largest summed atomic number, then largest change, then lexicographically smallest atom tuple, then lowest row index.

Returns
-------
Status vector using codes 0 (unreported), 1 (primary bond), 2 (coupled-proton bond), 3 (independent angle), and 4 (representative independent dihedral).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def screen_graph_changes(
    changes: np.ndarray,
    coordinates: np.ndarray,
    atomic_numbers: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Classify changed internal coordinates with graphRC hierarchy.

    Parameters
    ----------
    changes : np.ndarray
        Nonnegative change magnitude for each coordinate.
    coordinates : np.ndarray
        Encoded rows [order, a, b, c, d].
    atomic_numbers : np.ndarray
        Positive atomic numbers, one per atom.
    thresholds : np.ndarray
        Bond, angle, dihedral, and coupled-proton thresholds.

    Returns
    -------
    np.ndarray
        Integer status code for each coordinate using the documented 0-4 mapping, one-shot fallback, and representative rule.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_screen_graph_changes(
    changes: np.ndarray,
    coordinates: np.ndarray,
    atomic_numbers: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Reference hierarchical graph-coordinate screening."""
    import numpy as np

    magnitude = np.asarray(changes, dtype=float)
    coords = np.asarray(coordinates)
    numbers = np.asarray(atomic_numbers)
    limits = np.asarray(thresholds, dtype=float)

    if (
        magnitude.ndim != 1
        or magnitude.size < 1
        or coords.shape != (magnitude.size, 5)
    ):
        raise ValueError("changes and coordinates must align")
    if (
        not np.all(np.isfinite(magnitude))
        or np.any(magnitude < 0.0)
    ):
        raise ValueError("changes must be finite and nonnegative")
    if (
        not np.all(np.isfinite(coords))
        or not np.all(coords == np.floor(coords))
    ):
        raise ValueError(
            "coordinate encoding must contain finite integers"
        )
    if (
        numbers.ndim != 1
        or numbers.size < 2
        or not np.all(np.isfinite(numbers))
    ):
        raise ValueError(
            "atomic_numbers must contain one value per atom"
        )
    if (
        not np.all(numbers == np.floor(numbers))
        or np.any(numbers < 1)
    ):
        raise ValueError(
            "atomic_numbers must be positive integers"
        )
    if (
        limits.shape != (4,)
        or not np.all(np.isfinite(limits))
        or np.any(limits <= 0.0)
    ):
        raise ValueError(
            "thresholds must be four positive values"
        )

    coords = coords.astype(int)
    numbers = numbers.astype(int)

    for row in coords:
        order = int(row[0])
        if order not in (2, 3, 4):
            raise ValueError(
                "coordinate order must be 2, 3, or 4"
            )

        atoms = row[1 : order + 1]
        if (
            np.any(atoms < 0)
            or np.any(atoms >= numbers.size)
            or len(set(atoms.tolist())) != order
        ):
            raise ValueError(
                "invalid coordinate atom indices"
            )
        if not np.all(row[order + 1 :] == -1):
            raise ValueError(
                "unused coordinate entries must be -1"
            )

    selected = np.zeros(magnitude.size, dtype=int)

    for primary_limits in (limits[:3], limits[:3] * 0.5):
        status = np.zeros(magnitude.size, dtype=int)

        bond_rows = np.flatnonzero(coords[:, 0] == 2)
        primary = bond_rows[
            magnitude[bond_rows] >= primary_limits[0]
        ]
        status[primary] = 1

        active_hydrogens = set()
        for index in primary:
            atom_a, atom_b = coords[index, 1:3]
            if numbers[atom_a] == 1:
                active_hydrogens.add(int(atom_a))
            if numbers[atom_b] == 1:
                active_hydrogens.add(int(atom_b))

        for index in bond_rows:
            if (
                status[index] != 0
                or magnitude[index] < limits[3]
            ):
                continue

            atom_a, atom_b = coords[index, 1:3]
            if (
                int(atom_a) in active_hydrogens
                or int(atom_b) in active_hydrogens
            ):
                status[index] = 2

        changed_atoms = set()
        for index in np.flatnonzero(
            (status == 1) | (status == 2)
        ):
            changed_atoms.update(
                map(int, coords[index, 1:3])
            )

        for index in np.flatnonzero(coords[:, 0] == 3):
            atoms = set(map(int, coords[index, 1:4]))
            if (
                magnitude[index] >= primary_limits[1]
                and not atoms.intersection(changed_atoms)
            ):
                status[index] = 3

        groups = {}
        for index in np.flatnonzero(coords[:, 0] == 4):
            atoms = tuple(map(int, coords[index, 1:5]))

            if (
                magnitude[index] < primary_limits[2]
                or set(atoms).intersection(changed_atoms)
            ):
                continue

            axis = tuple(sorted(atoms[1:3]))
            mass_proxy = int(
                np.sum(numbers[list(atoms)])
            )
            candidate = (
                -mass_proxy,
                -float(magnitude[index]),
                atoms,
                int(index),
            )
            groups.setdefault(axis, []).append(candidate)

        for candidates in groups.values():
            status[min(candidates)[3]] = 4

        selected = status
        if np.any(selected):
            break

    return selected

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
changes = np.array([0.65,0.27,31.0,18.0,30.0,25.0,24.0])
coordinates = np.array([
    [2,0,1,-1,-1],
    [2,1,2,-1,-1],
    [3,0,1,2,-1],
    [3,3,4,5,-1],
    [4,0,1,2,3],
    [4,3,4,5,6],
    [4,7,4,5,8],
])
atomic_numbers = np.array([8,1,8,6,6,6,1,8,1])
thresholds = np.array([0.55,14.0,23.0,0.22])""",
            "call": "screen_graph_changes(changes, coordinates, atomic_numbers, thresholds)",
            "gold_call": "_oracle_screen_graph_changes(changes, coordinates, atomic_numbers, thresholds)",
        },
        {
            "setup": """import numpy as np
changes = np.array([0.55,0.22,14.0,23.0])
coordinates = np.array([
    [2,0,1,-1,-1],
    [2,1,2,-1,-1],
    [3,3,4,5,-1],
    [4,4,5,6,7],
])
atomic_numbers = np.array([8,1,8,6,6,6,6,1])
thresholds = np.array([0.55,14.0,23.0,0.22])""",
            "call": "screen_graph_changes(changes, coordinates, atomic_numbers, thresholds)",
            "gold_call": "_oracle_screen_graph_changes(changes, coordinates, atomic_numbers, thresholds)",
        },
        {
            "setup": """import numpy as np
changes = np.array([0.31,0.21,6.0,12.0,13.0])
coordinates = np.array([
    [2,0,1,-1,-1],
    [2,1,2,-1,-1],
    [3,3,4,5,-1],
    [4,3,4,5,6],
    [4,7,4,5,8],
])
atomic_numbers = np.array([8,1,8,6,6,6,6,8,4])
thresholds = np.array([0.55,14.0,23.0,0.22])""",
            "call": "screen_graph_changes(changes, coordinates, atomic_numbers, thresholds)",
            "gold_call": "_oracle_screen_graph_changes(changes, coordinates, atomic_numbers, thresholds)",
        },
    ]
