"""
Assemble the conductance and capacity operators of the layered near field, including the convective surface term of the heated patch.

Transient conduction in a heterogeneous solid is governed by the balance between the rate at which internal energy is stored and the divergence of the Fourier flux. Discretising the weak form of that balance with isoparametric elements produces two operators of the same sparsity pattern but very different physical content. The conductance operator is the quadratic form of the temperature gradient against the conductivity tensor, so it is a weighted graph Laplacian on the mesh: positive semi-definite, singular on constant temperature fields, and blind to how much heat the material can hold. The capacity operator is the quadratic form of the temperature itself against the volumetric heat capacity, the product of density and specific heat, so it is a mass matrix: positive definite and blind to how easily heat moves. Their ratio sets the diffusivity of the medium and therefore the depth a thermal front reaches in a given time, while the conductivity carried by the conductance operator alone is what relates a temperature gradient to a flux; both bear on this problem and they need not act in the same direction.

Keeping the capacity operator consistent rather than lumping it onto the diagonal matters here for a reason specific to what follows. The time integration is built on the matrix exponential of the operator obtained by inverting the capacity against the conductance, and the eigenvalues of that operator are what decide whether a given time step is admissible. Lumping shifts the whole spectrum, most strongly at its stiff end, so a lumped and a consistent assembly of the same mesh do not merely differ in accuracy, they differ in which time steps are usable at all.

The convective exchange over the heated patch is a Robin condition, and in the weak form it splits into two pieces that must not be confused. The part proportional to the surface temperature is a surface mass matrix scaled by the film coefficient, and it belongs in the conductance operator because it acts on the unknown; the part proportional to the ambient temperature is a load and belongs on the right-hand side. Placing both on the right-hand side would make the surface condition a prescribed flux rather than a convective exchange, and the resulting temperature field would then depend on the conductivity of the surface stratum in a way the convective condition largely removes. The film coefficient chosen here is large enough, measured against the heat capacity of the layer the thermal front occupies, that every stratification order reaches nearly the same surface temperature, which is what makes a comparison of stratifications a comparison of the interior rather than of the boundary.

Returns
-------
np.ndarray of shape (2, n_nodes, n_nodes), float: the symmetric conductance operator # including the convective surface term, stacked with the consistent capacity operator.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def assemble_near_field_operators(order, width: float, depth: float, patch_width: float,
                                  n_lateral: int, n_depth: int,
                                  film_coefficient: float) -> np.ndarray:
    """Assemble the thermal operators of the layered near field.

    Parameters
    ----------
    order : sequence of int
        The four material numbers, from the free surface downwards; entries lie
        between 1 and 4 and may repeat.
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
    operators : np.ndarray
        Array of shape (2, n_nodes, n_nodes); entry 0 is the conductance
        operator including the convective surface term and entry 1 is the
        consistent capacity operator, both in SI units.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above.
    """
    return operators  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_assemble_near_field_operators(order, width: float, depth: float, patch_width: float,
                                          n_lateral: int, n_depth: int,
                                          film_coefficient: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    # (thermal conductivity, density, Young's modulus, Poisson's ratio,
    #  thermal expansion coefficient, specific heat capacity) of materials 1 to 4.
    _MATERIALS = (
        (50.0, 1000.0, 10.0e6, 0.30, 1.0e-5, 10.0),
        (66.0, 1500.0, 15.0e6, 0.35, 5.0e-5, 15.0),
        (89.0, 2000.0, 20.0e6, 0.40, 1.0e-6, 20.0),
        (100.0, 2500.0, 25.0e6, 0.45, 5.0e-6, 25.0),
    )

    _HEX_SIGNS = np.array([[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
                           [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]])

    _QUAD_SIGNS = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

    _GAUSS_2PT = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))

    def _node_id(i, j, k, n_lateral):
        """Return the global index of the node at grid position (i, j, k)."""
        return i + (n_lateral + 1) * (j + (n_lateral + 1) * k)

    def _hex_shape(r, s, t):
        """Return shape functions and natural derivatives of the trilinear brick."""
        sr, ss, st = _HEX_SIGNS[:, 0], _HEX_SIGNS[:, 1], _HEX_SIGNS[:, 2]
        shape = 0.125 * (1.0 + sr * r) * (1.0 + ss * s) * (1.0 + st * t)
        grad = np.empty((3, 8))
        grad[0] = 0.125 * sr * (1.0 + ss * s) * (1.0 + st * t)
        grad[1] = 0.125 * (1.0 + sr * r) * ss * (1.0 + st * t)
        grad[2] = 0.125 * (1.0 + sr * r) * (1.0 + ss * s) * st
        return shape, grad

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

    def _validate_order(order):
        """Raise ValueError unless order lists four material numbers from one to four."""
        seq = list(order) if isinstance(order, (list, tuple, np.ndarray)) else None
        if seq is None or len(seq) != 4:
            raise ValueError("order must be a sequence of four material numbers")
        for value in seq:
            if not (isinstance(value, (int, np.integer)) and not isinstance(value, bool)
                    and 1 <= int(value) <= 4):
                raise ValueError("order entries must be integers between 1 and 4")
        return [int(value) for value in seq]

    material_order = _validate_order(order)
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

    nodes, elements, stratum = _build_mesh(width, depth, n_lateral, n_depth)
    n_nodes = nodes.shape[0]
    conductance = np.zeros((n_nodes, n_nodes), dtype=float)
    capacity = np.zeros((n_nodes, n_nodes), dtype=float)

    # Volume terms: a weighted Laplacian for conduction, a mass matrix for storage.
    for index, connectivity in enumerate(elements):
        conductivity, density, _, _, _, specific_heat = \
            _MATERIALS[material_order[stratum[index]] - 1]
        volumetric_capacity = density * specific_heat
        coordinates = nodes[connectivity]
        for r in _GAUSS_2PT:
            for s in _GAUSS_2PT:
                for t in _GAUSS_2PT:
                    shape, natural = _hex_shape(r, s, t)
                    jacobian = natural @ coordinates
                    detj = np.linalg.det(jacobian)
                    cartesian = np.linalg.solve(jacobian, natural)
                    conductance[np.ix_(connectivity, connectivity)] += \
                        conductivity * (cartesian.T @ cartesian) * detj
                    capacity[np.ix_(connectivity, connectivity)] += \
                        volumetric_capacity * np.outer(shape, shape) * detj

    # Robin surface term: the part acting on the unknown temperature.
    for face in _patch_faces(nodes, width, patch_width, n_lateral):
        planar = nodes[face][:, :2]
        for r in _GAUSS_2PT:
            for s in _GAUSS_2PT:
                shape, natural = _quad_shape(r, s)
                detj = np.linalg.det(natural @ planar)
                conductance[np.ix_(face, face)] += \
                    film_coefficient * np.outer(shape, shape) * detj

    return np.stack([conductance, capacity])

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    # Every case returns a plain float: the valid cases reduce the returned array
    # through a position weighted digest, the invalid cases return a status code.
    return [
        # --- Valid: benchmark near field, stratification (1, 2, 3, 4) (normal scenario) ---
        {
            "setup": """import numpy as np
order = [1, 2, 3, 4]
width, depth, patch_width = 120.0, 40.0, 60.0
n_lateral, n_depth = 4, 8
film_coefficient = 25.0

def digest(values):
    # Each returned block is reduced on its own scale and raised to its own
    # power, so that a relative error in one block moves the result by that
    # relative amount however small that block is against the others. The
    # weight is an outer product of two different position functions, which
    # makes the reduction sensitive to a transposed or reordered block.
    total = 1.0
    for index, block in enumerate(np.asarray(values, dtype=float)):
        entries = 1.0e6 * np.asarray(block, dtype=float)
        weight = np.outer(np.arange(1.0, entries.shape[0] + 1.0),
                          np.sqrt(np.arange(1.0, entries.shape[1] + 1.0)))
        weight = weight / weight.sum()
        moments = np.sqrt(np.sum(weight * entries ** 2)) + 0.5 * np.sum(weight * entries)
        total *= (entries.size + moments) ** (index + 1.0)
    return float(total)
""",
            "call": "digest(assemble_near_field_operators(order, width, depth, patch_width, n_lateral, n_depth, film_coefficient))",
            "gold_call": "digest(_oracle_assemble_near_field_operators(order, width, depth, patch_width, n_lateral, n_depth, film_coefficient))",
        },
        # --- Valid: inverted stratification on a coarser mesh ---
        {
            "setup": """import numpy as np
order = (4, 3, 2, 1)
width, depth, patch_width = 120.0, 40.0, 60.0
n_lateral, n_depth = 4, 4
film_coefficient = 25.0

def digest(values):
    # Each returned block is reduced on its own scale and raised to its own
    # power, so that a relative error in one block moves the result by that
    # relative amount however small that block is against the others. The
    # weight is an outer product of two different position functions, which
    # makes the reduction sensitive to a transposed or reordered block.
    total = 1.0
    for index, block in enumerate(np.asarray(values, dtype=float)):
        entries = 1.0e6 * np.asarray(block, dtype=float)
        weight = np.outer(np.arange(1.0, entries.shape[0] + 1.0),
                          np.sqrt(np.arange(1.0, entries.shape[1] + 1.0)))
        weight = weight / weight.sum()
        moments = np.sqrt(np.sum(weight * entries ** 2)) + 0.5 * np.sum(weight * entries)
        total *= (entries.size + moments) ** (index + 1.0)
    return float(total)
""",
            "call": "digest(assemble_near_field_operators(order, width, depth, patch_width, n_lateral, n_depth, film_coefficient))",
            "gold_call": "digest(_oracle_assemble_near_field_operators(order, width, depth, patch_width, n_lateral, n_depth, film_coefficient))",
        },
        # --- Boundary: the patch covers the whole free surface ---
        {
            "setup": """import numpy as np
order = [2, 3, 1, 4]
width, depth, patch_width = 120.0, 40.0, 120.0
n_lateral, n_depth = 4, 4
film_coefficient = 25.0

def digest(values):
    # Each returned block is reduced on its own scale and raised to its own
    # power, so that a relative error in one block moves the result by that
    # relative amount however small that block is against the others. The
    # weight is an outer product of two different position functions, which
    # makes the reduction sensitive to a transposed or reordered block.
    total = 1.0
    for index, block in enumerate(np.asarray(values, dtype=float)):
        entries = 1.0e6 * np.asarray(block, dtype=float)
        weight = np.outer(np.arange(1.0, entries.shape[0] + 1.0),
                          np.sqrt(np.arange(1.0, entries.shape[1] + 1.0)))
        weight = weight / weight.sum()
        moments = np.sqrt(np.sum(weight * entries ** 2)) + 0.5 * np.sum(weight * entries)
        total *= (entries.size + moments) ** (index + 1.0)
    return float(total)
""",
            "call": "digest(assemble_near_field_operators(order, width, depth, patch_width, n_lateral, n_depth, film_coefficient))",
            "gold_call": "digest(_oracle_assemble_near_field_operators(order, width, depth, patch_width, n_lateral, n_depth, film_coefficient))",
        },
        # --- Edge: the free surface is unheated, so no convective term appears
        #     anywhere in the conductance operator ---
        {
            "setup": """import numpy as np
order = [1, 1, 1, 1]
width, depth, patch_width = 120.0, 40.0, 0.0
n_lateral, n_depth = 4, 4
film_coefficient = 250.0

def digest(values):
    # Each returned block is reduced on its own scale and raised to its own
    # power, so that a relative error in one block moves the result by that
    # relative amount however small that block is against the others. The
    # weight is an outer product of two different position functions, which
    # makes the reduction sensitive to a transposed or reordered block.
    total = 1.0
    for index, block in enumerate(np.asarray(values, dtype=float)):
        entries = 1.0e6 * np.asarray(block, dtype=float)
        weight = np.outer(np.arange(1.0, entries.shape[0] + 1.0),
                          np.sqrt(np.arange(1.0, entries.shape[1] + 1.0)))
        weight = weight / weight.sum()
        moments = np.sqrt(np.sum(weight * entries ** 2)) + 0.5 * np.sum(weight * entries)
        total *= (entries.size + moments) ** (index + 1.0)
    return float(total)
""",
            "call": "digest(assemble_near_field_operators(order, width, depth, patch_width, n_lateral, n_depth, film_coefficient))",
            "gold_call": "digest(_oracle_assemble_near_field_operators(order, width, depth, patch_width, n_lateral, n_depth, film_coefficient))",
        },
        # --- Valid: one element across the plan, where the operators are small
        #     enough that a single wrong entry moves the digest well clear of the
        #     comparison tolerance ---
        {
            "setup": """import numpy as np
order = [3, 1, 4, 2]
width, depth, patch_width = 120.0, 40.0, 120.0
n_lateral, n_depth = 1, 4
film_coefficient = 25.0

def digest(values):
    # Each returned block is reduced on its own scale and raised to its own
    # power, so that a relative error in one block moves the result by that
    # relative amount however small that block is against the others. The
    # weight is an outer product of two different position functions, which
    # makes the reduction sensitive to a transposed or reordered block.
    total = 1.0
    for index, block in enumerate(np.asarray(values, dtype=float)):
        entries = 1.0e6 * np.asarray(block, dtype=float)
        weight = np.outer(np.arange(1.0, entries.shape[0] + 1.0),
                          np.sqrt(np.arange(1.0, entries.shape[1] + 1.0)))
        weight = weight / weight.sum()
        moments = np.sqrt(np.sum(weight * entries ** 2)) + 0.5 * np.sum(weight * entries)
        total *= (entries.size + moments) ** (index + 1.0)
    return float(total)
""",
            "call": "digest(assemble_near_field_operators(order, width, depth, patch_width, n_lateral, n_depth, film_coefficient))",
            "gold_call": "digest(_oracle_assemble_near_field_operators(order, width, depth, patch_width, n_lateral, n_depth, film_coefficient))",
        },
        # --- Invalid: a material number outside the available set ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        assemble_near_field_operators([1, 2, 3, 5], 120.0, 40.0, 60.0, 4, 4, 25.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_near_field_operators([1, 2, 3, 5], 120.0, 40.0, 60.0, 4, 4, 25.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: depth discretisation incompatible with four strata ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        assemble_near_field_operators([1, 2, 3, 4], 120.0, 40.0, 60.0, 4, 6, 25.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_near_field_operators([1, 2, 3, 4], 120.0, 40.0, 60.0, 4, 6, 25.0)
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
