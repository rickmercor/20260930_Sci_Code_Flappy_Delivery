"""
Assemble the consistent nodal vector through which the convective patch drives the near field, before it is scaled by the ambient temperature history.

A convective boundary condition states that the flux leaving the surface is the film coefficient times the difference between the ambient temperature and the surface temperature. In the weak form the two halves of that difference behave completely differently. The half proportional to the unknown surface temperature is bilinear and joins the operator acting on the unknowns; the half proportional to the ambient temperature is linear and forms a load. Because the ambient temperature here is uniform over the patch and depends only on time, the load separates into a fixed spatial vector multiplied by a scalar history, and that spatial vector is what this step produces. Keeping it separate from the history is what allows the time integration that follows to build its propagators once and reuse them at every step, instead of reassembling a load vector at every quadrature point of every step.




The vector itself is the film coefficient times the integral of each shape function over the heated part of the surface. This is a consistent load, not a lumped one: the total it distributes equals the film coefficient times the patch area, but the shares are unequal, with interior nodes of the patch receiving more than nodes on its rim and corner nodes least of all. Replacing it by an equal split over the patch nodes would preserve the total energy input but move the flux distribution, and since the peak thermal stress in this problem sits at the rim of the heated region, where the constraint from the surrounding cold material is largest, that redistribution moves precisely the quantity being measured.

A detail worth stating explicitly is which surface elements count as heated. The patch is a centred square whose side need not be commensurate with the mesh, so a rule is needed for elements the patch edge crosses. Taking only those elements that lie entirely inside the patch keeps the heated region a union of whole element faces, which makes the assembled vector independent of any convention for partial faces and keeps the discrete patch area exactly equal to the square of the specified side whenever the mesh resolves it. A consequence, and a useful degenerate check, is that a patch narrower than one element contains no complete face and therefore drives the medium not at all.

Returns
-------
np.ndarray of shape (n_nodes,), float: the consistent convective load vector in W/K, to be multiplied by the ambient temperature history.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def assemble_patch_load_vector(width: float, depth: float, patch_width: float,
                               n_lateral: int, n_depth: int,
                               film_coefficient: float) -> np.ndarray:
    """Assemble the spatial part of the convective load over the heated patch.

    Parameters
    ----------
    width : float
        Lateral extent of the near-field box in metres (width > 0).
    depth : float
        Depth of the near-field box in metres (depth > 0).
    patch_width : float
        Side length of the centred square heated patch in metres
        (patch_width >= 0). The patch is resolved by the mesh: a top-surface
        element face carries the convective term when the whole face lies
        inside the patch and carries none otherwise, so a face is never
        partially heated.
    n_lateral : int
        Number of elements along each lateral direction (n_lateral >= 1).
    n_depth : int
        Number of elements through the depth, a multiple of four
        (n_depth >= 1).
    film_coefficient : float
        Convective film coefficient over the patch in W m^-2 K^-1
        (film_coefficient >= 0).

    Returns
    -------
    load : np.ndarray
        Vector of length n_nodes holding the film coefficient times the
        integral of each shape function over the heated patch, in W K^-1.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above.
    """
    return load  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_assemble_patch_load_vector(width: float, depth: float, patch_width: float,
                                       n_lateral: int, n_depth: int,
                                       film_coefficient: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    _QUAD_SIGNS = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

    _GAUSS_2PT = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))

    def _node_id(i, j, k, n_lateral):
        """Return the global index of the node at grid position (i, j, k)."""
        return i + (n_lateral + 1) * (j + (n_lateral + 1) * k)

    def _quad_shape(r, s):
        """Return shape functions and natural derivatives of the bilinear quadrilateral."""
        sr, ss = _QUAD_SIGNS[:, 0], _QUAD_SIGNS[:, 1]
        shape = 0.25 * (1.0 + sr * r) * (1.0 + ss * s)
        grad = np.empty((2, 4))
        grad[0] = 0.25 * sr * (1.0 + ss * s)
        grad[1] = 0.25 * (1.0 + sr * r) * ss
        return shape, grad

    def _build_mesh(width, depth, n_lateral, n_depth):
        """Return nodes, element connectivity and the stratum index of each element."""
        lateral = np.linspace(0.0, width, n_lateral + 1)
        vertical = np.linspace(0.0, depth, n_depth + 1)
        nodes = np.empty(((n_lateral + 1) ** 2 * (n_depth + 1), 3))
        for k in range(n_depth + 1):
            for j in range(n_lateral + 1):
                for i in range(n_lateral + 1):
                    nodes[_node_id(i, j, k, n_lateral)] = (lateral[i], lateral[j], vertical[k])
        elements = []
        for k in range(n_depth):
            for j in range(n_lateral):
                for i in range(n_lateral):
                    elements.append([
                        _node_id(i, j, k, n_lateral), _node_id(i + 1, j, k, n_lateral),
                        _node_id(i + 1, j + 1, k, n_lateral), _node_id(i, j + 1, k, n_lateral),
                        _node_id(i, j, k + 1, n_lateral), _node_id(i + 1, j, k + 1, n_lateral),
                        _node_id(i + 1, j + 1, k + 1, n_lateral), _node_id(i, j + 1, k + 1, n_lateral)])
        elements = np.array(elements, dtype=int)
        centre_depth = nodes[elements][:, :, 2].mean(axis=1)
        stratum = np.minimum((centre_depth / (depth / 4.0)).astype(int), 3)
        return nodes, elements, stratum

    def _patch_faces(nodes, width, patch_width, n_lateral):
        """Return the top-surface element faces that lie entirely inside the heated patch."""
        faces = []
        for j in range(n_lateral):
            for i in range(n_lateral):
                face = [_node_id(i, j, 0, n_lateral), _node_id(i + 1, j, 0, n_lateral),
                        _node_id(i + 1, j + 1, 0, n_lateral), _node_id(i, j + 1, 0, n_lateral)]
                corners = nodes[face][:, :2]
                if np.all(np.abs(corners - 0.5 * width) <= 0.5 * patch_width + 1.0e-9):
                    faces.append(face)
        return faces

    def _validate_mesh_arguments(width, depth, n_lateral, n_depth):
        """Raise ValueError when the near-field geometry or discretisation is inadmissible."""
        for name, value in (("width", width), ("depth", depth)):
            if not (isinstance(value, (int, float)) and np.isfinite(value) and float(value) > 0.0):
                raise ValueError(f"{name} must be a finite number > 0")
        for name, value in (("n_lateral", n_lateral), ("n_depth", n_depth)):
            if not (isinstance(value, (int, np.integer)) and not isinstance(value, bool)
                    and int(value) >= 1):
                raise ValueError(f"{name} must be an integer >= 1")
        if int(n_depth) % 4 != 0:
            raise ValueError("n_depth must be a multiple of 4 so that every element lies in one stratum")

    _validate_mesh_arguments(width, depth, n_lateral, n_depth)
    if not (isinstance(patch_width, (int, float)) and np.isfinite(patch_width)
            and float(patch_width) >= 0.0):
        raise ValueError("patch_width must be a finite number >= 0")
    if not (isinstance(film_coefficient, (int, float)) and np.isfinite(film_coefficient)
            and float(film_coefficient) >= 0.0):
        raise ValueError("film_coefficient must be a finite number >= 0")

    width = float(width)
    depth = float(depth)
    patch_width = float(patch_width)
    n_lateral = int(n_lateral)
    n_depth = int(n_depth)
    film_coefficient = float(film_coefficient)

    nodes, _, _ = _build_mesh(width, depth, n_lateral, n_depth)
    load = np.zeros(nodes.shape[0], dtype=float)
    for face in _patch_faces(nodes, width, patch_width, n_lateral):
        planar = nodes[face][:, :2]
        for r in _GAUSS_2PT:
            for s in _GAUSS_2PT:
                shape, natural = _quad_shape(r, s)
                detj = np.linalg.det(natural @ planar)
                load[face] += film_coefficient * shape * detj
    return load

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


def test_cases():
    """Return list of test case specifications."""
    # Every case returns a plain float: the valid cases reduce the returned array
    # through a position weighted digest, the invalid cases return a status code.
    return [
        # --- Valid: benchmark patch and mesh (normal scenario) ---
        {
            "setup": """import numpy as np
width, depth, patch_width = 120.0, 40.0, 60.0
n_lateral, n_depth = 4, 8
film_coefficient = 25.0

def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(assemble_patch_load_vector(width, depth, patch_width, n_lateral, n_depth, film_coefficient))",
            "gold_call": "digest(_oracle_assemble_patch_load_vector(width, depth, patch_width, n_lateral, n_depth, film_coefficient))",
        },
        # --- Valid: finer lateral mesh over the same patch ---
        {
            "setup": """import numpy as np
width, depth, patch_width = 120.0, 40.0, 60.0
n_lateral, n_depth = 8, 4
film_coefficient = 3000.0

def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(assemble_patch_load_vector(width, depth, patch_width, n_lateral, n_depth, film_coefficient))",
            "gold_call": "digest(_oracle_assemble_patch_load_vector(width, depth, patch_width, n_lateral, n_depth, film_coefficient))",
        },
        # --- Boundary: the patch covers the whole free surface ---
        {
            "setup": """import numpy as np
width, depth, patch_width = 120.0, 40.0, 120.0
n_lateral, n_depth = 4, 4
film_coefficient = 25.0

def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(assemble_patch_load_vector(width, depth, patch_width, n_lateral, n_depth, film_coefficient))",
            "gold_call": "digest(_oracle_assemble_patch_load_vector(width, depth, patch_width, n_lateral, n_depth, film_coefficient))",
        },
        # --- Edge: the free surface is unheated, so the load vanishes ---
        {
            "setup": """import numpy as np
width, depth, patch_width = 120.0, 40.0, 0.0
n_lateral, n_depth = 4, 4
film_coefficient = 25.0

def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(assemble_patch_load_vector(width, depth, patch_width, n_lateral, n_depth, film_coefficient))",
            "gold_call": "digest(_oracle_assemble_patch_load_vector(width, depth, patch_width, n_lateral, n_depth, film_coefficient))",
        },
        # --- Invalid: negative film coefficient ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        assemble_patch_load_vector(120.0, 40.0, 60.0, 4, 4, -25.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_patch_load_vector(120.0, 40.0, 60.0, 4, 4, -25.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive lateral extent ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        assemble_patch_load_vector(0.0, 40.0, 60.0, 4, 4, 25.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_patch_load_vector(0.0, 40.0, 60.0, 4, 4, 25.0)
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
