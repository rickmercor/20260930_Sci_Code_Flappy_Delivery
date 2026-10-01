"""
Implement atom_centered_grid, which lays down the input real-space quadrature grid as a union of atom-centered shells built from six octahedral directions at a set of radii.

Quadrature grids in molecular electronic structure theory are almost always atom-centered, because the integrands are sums of functions that peak sharply at the nuclei and decay smoothly between them. The standard construction places a radial rule around every nucleus and distributes points over a spherical shell at each radius, so that the point density follows the nuclear cusps automatically. The six octahedral directions used here, the positive and negative Cartesian axes, form the smallest closed spherical rule that still integrates the low-order angular momentum components of a shell exactly, and adding the nuclear position itself captures the maximum of every primitive centered there.



A grid of this kind is deliberately redundant for a separable factorization of the two-electron integrals: it is constructed from the molecular geometry alone, without reference to which orbital products actually have to be integrated, so many of its points contribute almost nothing. Grid points are what every subsequent contraction scales with, which is why the redundancy is worth removing. This step produces only the point positions; weights are attached downstream, either kept uniform while points are selected or refitted from scratch.

Returns
-------
np.ndarray of shape (n_centers * (1 + 6 * n_radii), 3), float: the grid point coordinates in bohr, center block by center block
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def atom_centered_grid(centers: "np.ndarray", radii: "np.ndarray") -> "np.ndarray":
    '''Build the atom-centered octahedral input quadrature grid.

    For each center in order, the grid holds the nuclear position followed by, for each
    radius in order, the six points displaced from that center by the radius along the
    directions +x, -x, +y, -y, +z and -z, in that order. A center with n_radii radii
    therefore contributes 1 + 6 * n_radii points, and the point at position
    1 + 6 * i + d within a center's block lies at radius index i along direction index d.

    Parameters
    ----------
    centers : np.ndarray
        Array of shape (n_centers, 3) holding the center positions in bohr.
        Must contain at least one row.
    radii : np.ndarray
        Array of shape (n_radii,) holding the shell radii in bohr. Must contain at least
        one entry and every radius must be strictly positive.

    Returns
    -------
    grid_points : np.ndarray
        Array of shape (n_centers * (1 + 6 * n_radii), 3) of dtype float holding the grid
        point coordinates in bohr, ordered center block by center block.

    Raises
    ------
    ValueError
        If centers is not a two-dimensional array with three columns and at least one row,
        if radii is not a one-dimensional array with at least one entry, or if any radius
        is not strictly positive.
    '''
    return grid_points

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _octahedral_directions():
    """Unit vectors +x, -x, +y, -y, +z, -z in that order."""
    return np.array([[1.0, 0.0, 0.0],
                     [-1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, -1.0, 0.0],
                     [0.0, 0.0, 1.0],
                     [0.0, 0.0, -1.0]], dtype=float)


def _oracle_atom_centered_grid(centers: "np.ndarray", radii: "np.ndarray") -> "np.ndarray":
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] < 1:
        raise ValueError("centers must have shape (n_centers, 3) with n_centers >= 1")
    if radii.ndim != 1 or radii.shape[0] < 1:
        raise ValueError("radii must have shape (n_radii,) with n_radii >= 1")
    if not np.all(radii > 0.0):
        raise ValueError("every shell radius must be strictly positive")

    shell = (radii[:, None, None] * _octahedral_directions()[None, :, :]).reshape(-1, 3)
    offsets = np.vstack([np.zeros((1, 3), dtype=float), shell])
    grid_points = (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    return grid_points

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Typical: two centers with three radii, giving 2 * 19 points ---
        {
            "setup": """import numpy as np
centers = np.array([[0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0]], dtype=float)
radii = np.array([0.4, 0.9, 1.7], dtype=float)
""",
            "call": "atom_centered_grid(centers.copy(), radii.copy())",
            "gold_call": "_oracle_atom_centered_grid(centers.copy(), radii.copy())",
        },
        # --- Boundary: the smallest grid, one center and one radius, giving 7 points ---
        {
            "setup": """import numpy as np
centers = np.array([[1.0, -2.0, 0.5]], dtype=float)
radii = np.array([1.25], dtype=float)
""",
            "call": "atom_centered_grid(centers.copy(), radii.copy())",
            "gold_call": "_oracle_atom_centered_grid(centers.copy(), radii.copy())",
        },
        # --- Edge: the production geometry and radii, checking shell radii and ordering ---
        {
            "setup": """import numpy as np
centers = np.array([[0.0, 0.0, 0.0],
                    [1.9, 0.0, 0.2],
                    [-0.2, 2.0, 0.1],
                    [0.9, 1.1, 1.75]], dtype=float)
radii = np.array([0.4, 0.9, 1.7, 3.0], dtype=float)
def shape_and_radii(fn):
    points = np.asarray(fn(centers.copy(), radii.copy()), dtype=float)
    if points.shape != (100, 3):
        raise AssertionError("expected 100 grid points in three dimensions")
    block = points.reshape(4, 25, 3) - centers[:, None, :]
    distances = np.linalg.norm(block, axis=-1)
    return np.concatenate([points.reshape(-1), distances.reshape(-1)])
""",
            "call": "shape_and_radii(atom_centered_grid)",
            "gold_call": "shape_and_radii(_oracle_atom_centered_grid)",
        },
        # --- Invalid: a non-positive radius ---
        {
            "setup": """import numpy as np
centers = np.zeros((1, 3), dtype=float)
radii = np.array([0.5, 0.0], dtype=float)
def run_model():
    try:
        atom_centered_grid(centers.copy(), radii.copy())
        return 0.0
    except ValueError:
        return 1.0
def run_oracle():
    try:
        _oracle_atom_centered_grid(centers.copy(), radii.copy())
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
