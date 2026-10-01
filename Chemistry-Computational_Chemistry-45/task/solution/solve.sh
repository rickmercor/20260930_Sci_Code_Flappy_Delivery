#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def generate_mode_trajectory(
    reference_positions: np.ndarray,
    mode_vector: np.ndarray,
    amplitudes: np.ndarray,
) -> np.ndarray:
    """Reference linear mode-displacement construction."""
    import numpy as np

    reference = np.asarray(reference_positions, dtype=float)
    mode = np.asarray(mode_vector, dtype=float)
    scale = np.asarray(amplitudes, dtype=float)

    if reference.ndim != 2 or reference.shape[0] < 2 or reference.shape[1] != 3:
        raise ValueError("reference_positions must have shape (n_atoms, 3)")
    if mode.shape != reference.shape:
        raise ValueError("mode_vector must match reference_positions")
    if scale.ndim != 1 or scale.size < 2:
        raise ValueError("amplitudes must contain at least two values")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(mode)):
        raise ValueError("coordinates and mode vectors must be finite")
    if not np.all(np.isfinite(scale)) or np.ptp(scale) <= 0.0:
        raise ValueError("amplitudes must be finite and nonconstant")

    return reference[None, :, :] + scale[:, None, None] * mode[None, :, :]

def select_diverse_pair(trajectory: np.ndarray) -> np.ndarray:
    """Reference unaligned Cartesian RMSD pair selection."""
    import numpy as np

    frames = np.asarray(trajectory, dtype=float)
    if (
        frames.ndim != 3
        or frames.shape[0] < 2
        or frames.shape[1] < 1
        or frames.shape[2] != 3
    ):
        raise ValueError("trajectory must have shape (n_frames, n_atoms, 3)")
    if not np.all(np.isfinite(frames)):
        raise ValueError("trajectory must be finite")

    best_pair = (0, 1)
    best_rmsd = -np.inf

    for first in range(frames.shape[0] - 1):
        for second in range(first + 1, frames.shape[0]):
            delta = frames[first] - frames[second]
            rmsd = float(
                np.sqrt(np.mean(np.sum(delta * delta, axis=1)))
            )
            if rmsd > best_rmsd + 1e-12:
                best_rmsd = rmsd
                best_pair = (first, second)

    return np.asarray(best_pair, dtype=int)

def enumerate_internal_coordinates(
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

def evaluate_internal_coordinates(
    trajectory: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Reference Cartesian-to-internal-coordinate evaluation."""
    import numpy as np

    frames = np.asarray(trajectory, dtype=float)
    coords = np.asarray(coordinates)

    if (
        frames.ndim != 3
        or frames.shape[0] < 1
        or frames.shape[1] < 2
        or frames.shape[2] != 3
    ):
        raise ValueError(
            "trajectory must have shape (n_frames, n_atoms, 3)"
        )
    if (
        coords.ndim != 2
        or coords.shape[0] < 1
        or coords.shape[1] != 5
    ):
        raise ValueError(
            "coordinates must have shape (n_coordinates, 5)"
        )
    if not np.all(np.isfinite(frames)) or not np.all(np.isfinite(coords)):
        raise ValueError("inputs must be finite")
    if not np.all(coords == np.floor(coords)):
        raise ValueError("coordinate encoding must contain integers")

    coords = coords.astype(int)
    values = np.empty(
        (frames.shape[0], coords.shape[0]),
        dtype=float,
    )

    for column, row in enumerate(coords):
        order = int(row[0])
        if order not in (2, 3, 4):
            raise ValueError("coordinate order must be 2, 3, or 4")

        atoms = row[1 : order + 1]
        if (
            np.any(atoms < 0)
            or np.any(atoms >= frames.shape[1])
            or len(set(atoms.tolist())) != order
        ):
            raise ValueError("invalid atom indices in coordinate row")
        if not np.all(row[order + 1 :] == -1):
            raise ValueError("unused coordinate entries must be -1")

        if order == 2:
            vector = frames[:, atoms[0]] - frames[:, atoms[1]]
            values[:, column] = np.linalg.norm(vector, axis=1)

        elif order == 3:
            left = frames[:, atoms[0]] - frames[:, atoms[1]]
            right = frames[:, atoms[2]] - frames[:, atoms[1]]
            denominator = (
                np.linalg.norm(left, axis=1)
                * np.linalg.norm(right, axis=1)
            )
            if np.any(denominator <= 1e-14):
                raise ValueError("angle contains a zero-length vector")

            cosine = np.sum(left * right, axis=1) / denominator
            values[:, column] = np.degrees(
                np.arccos(np.clip(cosine, -1.0, 1.0))
            )

        else:
            p0, p1, p2, p3 = (
                frames[:, atom]
                for atom in atoms
            )
            b0 = -(p1 - p0)
            b1 = p2 - p1
            b2 = p3 - p2

            b1_norm = np.linalg.norm(b1, axis=1)
            if np.any(b1_norm <= 1e-14):
                raise ValueError(
                    "dihedral contains a zero-length central bond"
                )

            axis = b1 / b1_norm[:, None]
            left_normal = (
                b0
                - np.sum(b0 * axis, axis=1)[:, None] * axis
            )
            right_normal = (
                b2
                - np.sum(b2 * axis, axis=1)[:, None] * axis
            )
            norm_product = (
                np.linalg.norm(left_normal, axis=1)
                * np.linalg.norm(right_normal, axis=1)
            )
            if np.any(norm_product <= 1e-14):
                raise ValueError(
                    "dihedral is undefined for collinear atoms"
                )

            x_value = np.sum(
                left_normal * right_normal,
                axis=1,
            )
            y_value = np.sum(
                np.cross(axis, left_normal) * right_normal,
                axis=1,
            )
            values[:, column] = np.degrees(
                np.arctan2(y_value, x_value)
            )

    return values

def coordinate_change_magnitudes(
    coordinate_values: np.ndarray,
    frame_pair: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Reference mixed linear-periodic change calculation."""
    import numpy as np

    values = np.asarray(coordinate_values, dtype=float)
    pair = np.asarray(frame_pair)
    coords = np.asarray(coordinates)

    if (
        values.ndim != 2
        or values.shape[0] < 2
        or values.shape[1] < 1
    ):
        raise ValueError(
            "coordinate_values must have at least two frames and one coordinate"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("coordinate_values must be finite")

    if (
        pair.shape != (2,)
        or not np.all(np.isfinite(pair))
        or not np.all(pair == np.floor(pair))
    ):
        raise ValueError("frame_pair must contain two integer indices")

    pair = pair.astype(int)
    if (
        pair[0] < 0
        or pair[1] >= values.shape[0]
        or pair[0] >= pair[1]
    ):
        raise ValueError(
            "frame_pair must be increasing and in range"
        )

    if (
        coords.shape != (values.shape[1], 5)
        or not np.all(np.isfinite(coords))
    ):
        raise ValueError(
            "coordinates must align with coordinate_values"
        )
    if (
        not np.all(coords == np.floor(coords))
        or not np.all(np.isin(coords[:, 0], [2, 3, 4]))
    ):
        raise ValueError(
            "coordinates must have valid integer orders"
        )

    delta = values[pair[1]] - values[pair[0]]
    changes = np.abs(delta)

    is_dihedral = coords[:, 0].astype(int) == 4
    changes[is_dihedral] = np.abs(
        (delta[is_dihedral] + 180.0) % 360.0 - 180.0
    )

    return changes.astype(float)

def screen_graph_changes(
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

def bond_change_f1(
    statuses: np.ndarray,
    coordinates: np.ndarray,
    expected_bonds: np.ndarray,
) -> float:
    """Reference set-based reactive-bond F1 score."""
    import numpy as np

    state = np.asarray(statuses)
    coords = np.asarray(coordinates)
    expected = np.asarray(expected_bonds)

    if (
        state.ndim != 1
        or state.size < 1
        or coords.shape != (state.size, 5)
    ):
        raise ValueError("statuses and coordinates must align")
    if (
        not np.all(np.isfinite(state))
        or not np.all(state == np.floor(state))
    ):
        raise ValueError("statuses must be finite integers")

    state = state.astype(int)
    if not np.all(np.isin(state, [0, 1, 2, 3, 4])):
        raise ValueError(
            "status codes must be between 0 and 4"
        )

    if (
        not np.all(np.isfinite(coords))
        or not np.all(coords == np.floor(coords))
    ):
        raise ValueError(
            "coordinate encoding must contain finite integers"
        )
    coords = coords.astype(int)

    if expected.ndim != 2 or expected.shape[1] != 2:
        raise ValueError(
            "expected_bonds must have shape (n_expected, 2)"
        )
    if (
        not np.all(np.isfinite(expected))
        or not np.all(expected == np.floor(expected))
    ):
        raise ValueError(
            "expected bond indices must be finite integers"
        )

    expected = expected.astype(int)
    if (
        np.any(expected < 0)
        or np.any(expected[:, 0] == expected[:, 1])
    ):
        raise ValueError(
            "expected bonds must contain distinct nonnegative indices"
        )

    detected = {
        tuple(sorted(map(int, coords[index, 1:3])))
        for index in np.flatnonzero(
            (state == 1) | (state == 2)
        )
        if coords[index, 0] == 2
    }

    target = {
        tuple(sorted(map(int, row)))
        for row in expected
    }

    if not detected and not target:
        return 1.0

    true_positive = len(detected.intersection(target))
    false_positive = len(detected - target)
    false_negative = len(target - detected)

    denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    return (
        0.0
        if denominator == 0
        else float(2 * true_positive / denominator)
    )

def identify_transition_mode(
    reference_positions: np.ndarray,
    mode_vectors: np.ndarray,
    amplitudes: np.ndarray,
    bonds: np.ndarray,
    atomic_numbers: np.ndarray,
    expected_bonds: np.ndarray,
) -> int:
    """Compose the seven graph-coordinate steps and select a mode."""
    import numpy as np

    reference = np.asarray(reference_positions, dtype=float)
    modes = np.asarray(mode_vectors, dtype=float)
    if reference.ndim != 2 or reference.shape[0] < 2 or reference.shape[1] != 3:
        raise ValueError("reference_positions must have shape (n_atoms, 3)")
    if modes.ndim != 3 or modes.shape[0] < 1 or modes.shape[1:] != reference.shape:
        raise ValueError("mode_vectors must have shape (n_modes, n_atoms, 3)")

    coordinates = enumerate_internal_coordinates(bonds, reference.shape[0])
    thresholds = np.array([0.4, 10.0, 20.0, 0.15], dtype=float)
    ranking = []
    for index, mode in enumerate(modes):
        trajectory = generate_mode_trajectory(reference, mode, amplitudes)
        pair = select_diverse_pair(trajectory)
        values = evaluate_internal_coordinates(trajectory, coordinates)
        changes = coordinate_change_magnitudes(values, pair, coordinates)
        statuses = screen_graph_changes(
            changes,
            coordinates,
            atomic_numbers,
            thresholds,
        )
        f1_score = bond_change_f1(statuses, coordinates, expected_bonds)
        independent_nonbond = int(np.count_nonzero((statuses == 3) | (statuses == 4)))
        ranking.append((-f1_score, independent_nonbond, index))

    return int(min(ranking)[2] + 1)
SCICODE_GOLD_EOF
