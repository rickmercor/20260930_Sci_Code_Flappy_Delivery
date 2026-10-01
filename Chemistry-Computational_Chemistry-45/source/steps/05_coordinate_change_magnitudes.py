"""
Compute internal-coordinate changes between two frames with periodic dihedral wrapping.

Bond lengths are in angstrom, while angles and dihedrals are in degrees. Signed dihedrals have a 360-degree period, so their shortest angular separation must be used instead of a raw subtraction.

Returns
-------
Float vector of absolute changes, using shortest wrapped differences for dihedrals.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def coordinate_change_magnitudes(
    coordinate_values: np.ndarray,
    frame_pair: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Compute absolute coordinate changes for a selected frame pair.

    Parameters
    ----------
    coordinate_values : np.ndarray
        Evaluated values with shape (n_frames, n_coordinates): bond lengths
        in angstrom and angles and dihedrals in degrees, with a 360-degree
        period for dihedrals.
    frame_pair : np.ndarray
        Two increasing zero-based frame indices.
    coordinates : np.ndarray
        Encoded coordinate rows aligned with the value columns.

    Returns
    -------
    np.ndarray
        Nonnegative change magnitude for each coordinate.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_coordinate_change_magnitudes(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
coordinate_values = np.array([[1.0,90.0,179.0],[1.6,105.0,-179.0]])
frame_pair = np.array([0,1])
coordinates = np.array([[2,0,1,-1,-1],[3,0,1,2,-1],[4,0,1,2,3]])""",
            "call": "coordinate_change_magnitudes(coordinate_values, frame_pair, coordinates)",
            "gold_call": "_oracle_coordinate_change_magnitudes(coordinate_values, frame_pair, coordinates)",
        },
        {
            "setup": """import numpy as np
coordinate_values = np.array([
    [179.0,-179.0,0.0,1.2],
    [-1.0,1.0,180.0,1.8],
])
frame_pair = np.array([0,1])
coordinates = np.array([
    [4,0,1,2,3],
    [4,1,2,3,4],
    [4,2,3,4,5],
    [2,0,1,-1,-1],
])""",
            "call": "coordinate_change_magnitudes(coordinate_values, frame_pair, coordinates)",
            "gold_call": "_oracle_coordinate_change_magnitudes(coordinate_values, frame_pair, coordinates)",
        },
        {
            "setup": """import numpy as np
coordinate_values = np.array([
    [1.0,15.0,170.0,-170.0],
    [9.0,90.0,0.0,0.0],
    [4.0,175.0,-170.0,170.0],
])
frame_pair = np.array([0,2])
coordinates = np.array([
    [2,0,1,-1,-1],
    [3,0,1,2,-1],
    [4,0,1,2,3],
    [4,1,2,3,4],
])""",
            "call": "coordinate_change_magnitudes(coordinate_values, frame_pair, coordinates)",
            "gold_call": "_oracle_coordinate_change_magnitudes(coordinate_values, frame_pair, coordinates)",
        },
    ]
