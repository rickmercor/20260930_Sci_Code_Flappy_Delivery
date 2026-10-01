"""
Map every atomistic Cartesian frame to its coarse-grained bead coordinates using the supplied linear coordinate map.

A linear coarse-graining map contracts the atom axis while preserving the frame and Cartesian-component axes. For atomistic coordinates with shape (frames, atoms, components) and a bead-by-atom mapping matrix, the result has shape (frames, beads, components). The mapping must be finite, nonnegative, full row rank, and each row must sum to one.

Returns
-------
Return a NumPy array with shape (frames, beads, components).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def map_atomistic_coordinates(
    atomistic_positions: np.ndarray,
    mapping: np.ndarray,
) -> np.ndarray:
    """Map atomistic Cartesian positions to coarse-grained beads.

    Parameters
    ----------
    atomistic_positions
        Array with shape (frames, atoms, components).
    mapping
        Full-row-rank nonnegative coordinate map with shape (beads, atoms).
        Each row sums to one.

    Returns
    -------
    np.ndarray
        Mapped positions with shape (frames, beads, components).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_map_atomistic_coordinates(atomistic_positions, mapping):
    import numpy as np

    positions = np.asarray(atomistic_positions, dtype=float)
    coordinate_map = np.asarray(mapping, dtype=float)
    if positions.ndim != 3 or min(positions.shape) == 0:
        raise ValueError("atomistic_positions must be a nonempty 3D array")
    if (
        coordinate_map.ndim != 2
        or coordinate_map.shape[0] == 0
        or coordinate_map.shape[1] != positions.shape[1]
    ):
        raise ValueError("mapping has incompatible shape")
    if np.any(~np.isfinite(positions)) or np.any(~np.isfinite(coordinate_map)):
        raise ValueError("positions and mapping must be finite")
    if np.any(coordinate_map < 0):
        raise ValueError("mapping weights must be nonnegative")
    if np.linalg.matrix_rank(coordinate_map) != coordinate_map.shape[0]:
        raise ValueError("mapping must have full row rank")
    if not np.allclose(
        np.sum(coordinate_map, axis=1), 1.0, rtol=1e-12, atol=1e-12
    ):
        raise ValueError("each mapping row must sum to one")
    return np.einsum("qa,fac->fqc", coordinate_map, positions)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nr=np.arange(48,dtype=float).reshape(4,4,3)/10; "
            "M=np.array([[.6,.4,0,0],[0,0,.25,.75]])",
            "call": "map_atomistic_coordinates(r,M)",
            "gold_call": "_oracle_map_atomistic_coordinates(r,M)",
        },
        {
            "setup": "import numpy as np\nr=np.array([[[1.,-1.],[2.,3.],[4.,0.]]]); M=np.eye(3)",
            "call": "map_atomistic_coordinates(r,M)",
            "gold_call": "_oracle_map_atomistic_coordinates(r,M)",
        },
        {
            "setup": "import numpy as np\nr=np.arange(30,dtype=float).reshape(5,3,2); "
            "M=np.array([[.2,.3,.5]])",
            "call": "map_atomistic_coordinates(r,M)",
            "gold_call": "_oracle_map_atomistic_coordinates(r,M)",
        },
        {
            "setup": "import numpy as np\nr=np.array([[[1.],[-2.],[3.],[.5]],"
            "[[0.],[4.],[-1.],[2.]],[[2.],[1.],[.5],[-3.]]]); "
            "M=np.array([[.5,.25,.25,0.],[0.,.2,.3,.5]])",
            "call": "map_atomistic_coordinates(r,M)",
            "gold_call": "_oracle_map_atomistic_coordinates(r,M)",
        },
    ]
