"""
Rebuild every monomer of a cluster at the isolated-monomer reference geometry while preserving its centre of mass and its orientation, producing the zeroth-order point of the reference manifold.

A coarse-grained configuration records only the positions and orientations of the monomers; the internal geometry of each monomer has been integrated out. Reconstructing a point in the full atomic configuration space from such a record therefore requires attaching the distinguished internal geometry to each monomer in the orientation the monomer actually has. The set of all points built this way is the zeroth-order manifold, and it is the starting guess from which the relaxation departs. It is also, on its own, exactly the conventional rigidification: freezing every monomer at one fixed shape is what it means to stop here.

Paper I defines the monomer position as its first and heaviest atom, the oxygen for water. It defines the orientation from a Jacobi frame: the y-axis follows the vector from oxygen to the midpoint of the two hydrogens, the x-axis follows the component of H2-H1 perpendicular to that bisector, and the z-axis completes the right-handed frame. Rebuilding therefore keeps the oxygen fixed and carries the isolated reference geometry from its own Jacobi frame into the input monomer's Jacobi frame. No centre-of-mass fit or Eckart/Kabsch superposition enters this coordinate map.

The internal distortions carried by the input cluster are discarded by this step, and that is the intended behaviour rather than a loss: they are precisely the fast degrees of freedom the coarse-grained model does not represent, and the relaxation that follows will regenerate them from the intermolecular environment instead of inheriting them from the input.

Returns
-------
np.ndarray of shape (n_atoms, 3), float: the zeroth-order reference-manifold configuration of the cluster, in angstrom.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_reference_configuration(coords: np.ndarray,
                                  monomer_geometry: np.ndarray) -> np.ndarray:
    """Place the reference monomer geometry onto every monomer of a cluster.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (3 * n_monomers, 3) in angstrom holding the atomic
        positions of the input cluster, ordered O, H, H within each monomer.
    monomer_geometry : np.ndarray
        Array of shape (3, 3) in angstrom holding the reference monomer
        geometry. Its first row is the oxygen position; its Jacobi frame
        defines the body-frame orientation.

    Returns
    -------
    reference_coords : np.ndarray
        Array with the same shape as ``coords`` in angstrom, holding the
        cluster with every monomer rebuilt at the reference geometry in its
        original position and orientation.

    Raises
    ------
    ValueError
        If either coordinate array has the wrong shape or non-finite entries,
        if ``coords`` does not contain complete three-atom monomers, or if a
        monomer has a degenerate bisector or projected H2-H1 Jacobi axis.
    """
    return reference_coords  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_reference_configuration(coords: np.ndarray,
                                          monomer_geometry: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    coords = np.asarray(coords, dtype=float)
    monomer_geometry = np.asarray(monomer_geometry, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_atoms, 3)")
    if coords.shape[0] < 3 or coords.shape[0] % 3 != 0:
        raise ValueError("coords must hold a whole number of three-atom monomers")
    if monomer_geometry.shape != (3, 3):
        raise ValueError("monomer_geometry must have shape (3, 3)")
    if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(monomer_geometry)):
        raise ValueError("coords and monomer_geometry must contain only finite entries")

    def jacobi_frame(block):
        bisector = 0.5 * (block[1] + block[2]) - block[0]
        bisector_norm = float(np.linalg.norm(bisector))
        if bisector_norm <= 1.0e-14:
            raise ValueError("the oxygen-to-hydrogen-midpoint vector must be nonzero")
        axis_y = bisector / bisector_norm

        difference = block[2] - block[1]
        projected = difference - float(np.dot(difference, axis_y)) * axis_y
        projected_norm = float(np.linalg.norm(projected))
        if projected_norm <= 1.0e-14:
            raise ValueError("the projected H2-H1 vector must be nonzero")
        axis_x = projected / projected_norm
        axis_z = np.cross(axis_x, axis_y)
        return np.column_stack((axis_x, axis_y, axis_z))

    n_monomers = coords.shape[0] // 3
    reference_coords = np.zeros_like(coords)

    reference_origin = monomer_geometry[0]
    reference_relative = monomer_geometry - reference_origin
    reference_frame = jacobi_frame(monomer_geometry)

    for mono in range(n_monomers):
        block = coords[3 * mono: 3 * mono + 3]
        target_frame = jacobi_frame(block)
        body_coordinates = reference_relative @ reference_frame
        reference_coords[3 * mono: 3 * mono + 3] = (
            body_coordinates @ target_frame.T + block[0])

    return reference_coords

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the benchmark cyclic tetramer (normal scenario) ---
        {
            "setup": """import numpy as np
coords = np.array([
    [1.965757, 0.000000, 0.000000], [1.276254, 0.689503, 0.097837],
    [2.425271, 0.000868, 0.837181], [0.000000, 1.965757, 0.000000],
    [-0.680176, 1.285581, -0.077118], [0.004437, 2.439238, -0.852304],
    [-1.965757, 0.000000, 0.000000], [-1.270755, -0.695002, 0.118515],
    [-2.428412, 0.000015, 0.841160], [0.000000, -1.965757, 0.000000],
    [0.684659, -1.281098, -0.058165], [0.015394, -2.427969, -0.869683]])
half = np.deg2rad(107.4) / 2.0
geom = np.array([[0.0, 0.0, 0.0],
                 [-0.9419 * np.sin(half), 0.9419 * np.cos(half), 0.0],
                 [0.9419 * np.sin(half), 0.9419 * np.cos(half), 0.0]])
""",
            "call": "build_reference_configuration(coords, geom)",
            "gold_call": "_oracle_build_reference_configuration(coords, geom)",
        },
        # --- Valid: a single monomer already at the reference geometry, which
        #     must be returned unchanged up to round-off ---
        {
            "setup": """import numpy as np
half = np.deg2rad(107.4) / 2.0
geom = np.array([[0.0, 0.0, 0.0],
                 [-0.9419 * np.sin(half), 0.9419 * np.cos(half), 0.0],
                 [0.9419 * np.sin(half), 0.9419 * np.cos(half), 0.0]])
coords = geom + np.array([1.5, -2.0, 0.25])
""",
            "call": "build_reference_configuration(coords, geom)",
            "gold_call": "_oracle_build_reference_configuration(coords, geom)",
        },
        # --- Boundary: a nearly linear but non-degenerate monomer ---
        {
            "setup": """import numpy as np
half = np.deg2rad(107.4) / 2.0
geom = np.array([[0.0, 0.0, 0.0],
                 [-0.9419 * np.sin(half), 0.9419 * np.cos(half), 0.0],
                 [0.9419 * np.sin(half), 0.9419 * np.cos(half), 0.0]])
coords = np.array([[0.0, 0.0, 0.0], [-0.999, 0.022, 0.003],
                   [1.001, 0.021, -0.002]])
""",
            "call": "build_reference_configuration(coords, geom)",
            "gold_call": "_oracle_build_reference_configuration(coords, geom)",
        },
        # --- Edge: a heavily distorted monomer at a general orientation, so
        #     both Jacobi axes must be constructed explicitly ---
        {
            "setup": """import numpy as np
half = np.deg2rad(107.4) / 2.0
geom = np.array([[0.0, 0.0, 0.0],
                 [-0.9419 * np.sin(half), 0.9419 * np.cos(half), 0.0],
                 [0.9419 * np.sin(half), 0.9419 * np.cos(half), 0.0]])
coords = np.array([[0.10, -0.20, 0.35],
                   [1.21, 0.14, 0.02],
                   [-0.18, 0.63, 1.02]])
""",
            "call": "build_reference_configuration(coords, geom)",
            "gold_call": "_oracle_build_reference_configuration(coords, geom)",
        },
        # --- Invalid: oxygen coincides with the hydrogen midpoint ---
        {
            "setup": """import numpy as np
coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
geom = np.array([[0.0, 0.0, 0.0], [-0.7, 0.5, 0.0], [0.7, 0.5, 0.0]])
def run_model():
    try:
        build_reference_configuration(coords, geom)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_reference_configuration(coords, geom)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: H2-H1 is parallel to the bisector ---
        {
            "setup": """import numpy as np
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]])
geom = np.array([[0.0, 0.0, 0.0], [-0.7, 0.5, 0.0], [0.7, 0.5, 0.0]])
def run_model():
    try:
        build_reference_configuration(coords, geom)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_reference_configuration(coords, geom)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
