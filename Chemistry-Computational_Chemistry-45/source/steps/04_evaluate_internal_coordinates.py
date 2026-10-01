"""
Evaluate encoded bond lengths, angles, and signed dihedrals over a Cartesian trajectory.

Bond distances are measured in angstrom, while graph angles and oriented dihedrals are evaluated in degrees. For ordered atoms (a,b,c,d), set b0=r_a-r_b, b1=r_c-r_b, and b2=r_d-r_c; project b0 and b2 perpendicular to the unit b1 axis and return atan2((b1_hat cross v) dot w, v dot w) in degrees.

Returns
-------
Float matrix containing bond lengths in angstrom and angles/dihedrals in degrees using the stated oriented-dihedral sign convention.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def evaluate_internal_coordinates(
    trajectory: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Evaluate graph internal coordinates for every frame.

    Parameters
    ----------
    trajectory : np.ndarray
        Cartesian frames with shape (n_frames, n_atoms, 3).
    coordinates : np.ndarray
        Encoded coordinate rows [order, a, b, c, d].

    Returns
    -------
    np.ndarray
        Values with shape (n_frames, n_coordinates), using the oriented-dihedral sign convention stated in the step background.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_evaluate_internal_coordinates(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
trajectory = np.array([[[0.,0.,0.],[1.,0.,0.]], [[0.,0.,0.],[0.,2.,0.]]])
coordinates = np.array([[2,0,1,-1,-1]])""",
            "call": "evaluate_internal_coordinates(trajectory, coordinates)",
            "gold_call": "_oracle_evaluate_internal_coordinates(trajectory, coordinates)",
        },
        {
            "setup": """import numpy as np
trajectory = np.array([[[1.,0.,0.],[0.,0.,0.],[0.,1.,0.]], [[1.,0.,0.],[0.,0.,0.],[-1.,1.,0.]]])
coordinates = np.array([[2,0,1,-1,-1],[3,0,1,2,-1]])""",
            "call": "evaluate_internal_coordinates(trajectory, coordinates)",
            "gold_call": "_oracle_evaluate_internal_coordinates(trajectory, coordinates)",
        },
        {
            "setup": """import numpy as np
trajectory = np.array([[[0.,1.,0.],[0.,0.,0.],[1.,0.,0.],[1.,1.,1.]], [[0.,1.,1.],[0.,0.,0.],[1.,0.,0.],[1.,1.,0.]]])
coordinates = np.array([[3,0,1,2,-1],[4,0,1,2,3]])""",
            "call": "evaluate_internal_coordinates(trajectory, coordinates)",
            "gold_call": "_oracle_evaluate_internal_coordinates(trajectory, coordinates)",
        },
    ]
