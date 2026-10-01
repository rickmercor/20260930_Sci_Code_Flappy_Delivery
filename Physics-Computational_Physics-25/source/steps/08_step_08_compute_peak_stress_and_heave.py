"""
Recover the element-centroid stresses from the displacement and temperature fields and report the peak equivalent stress together with the peak heave of the free surface.

Stress is not a nodal quantity. It follows from the displacement field by differentiation, so in an element interpolation of a given order the stress is one order lower and is discontinuous across element boundaries. Sampling it at the element centroid is the natural choice for a trilinear brick, because the centroid is where the linear part of the displacement gradient is represented best and where the parasitic contributions that make such elements over-stiff in bending are smallest; sampling instead at the corners or averaging to the nodes gives systematically different peaks on the same solution. The temperature entering the constitutive correction must be evaluated at the same point, by interpolating the nodal values with the element shape functions rather than by taking an element average, so that the mechanical and thermal parts of the stress are consistent.




The constitutive law subtracts the thermal strain before the elasticity matrix is applied, so the stress is the elasticity matrix acting on the strain implied by the displacements minus the same matrix acting on the free thermal strain. The second term is what makes a fully restrained heated element carry stress even though it has not moved at all, and it is the dominant term wherever the restraint is strong. Reducing the six components to a single equivalent value is done with the von Mises invariant, which is built from the differences between the direct components and from the shear components, and which vanishes for a purely hydrostatic state. That is the right choice here because it measures the distortional part of the stress, the part that drives yielding and cracking, and because it is invariant under the choice of axes, so it does not depend on how the mesh happens to be oriented.




The heave of the free surface is the complementary measurement. Thermal strain that is realised as displacement is not available to be converted into stress, so within one material a compliant, weakly confined configuration produces large heave and small stress while a stiff, strongly confined one produces the reverse. Across configurations built from different materials the trade-off is no longer clean, because the expansion coefficient scales both quantities together while the stiffness scales them in opposite directions, so which effect wins is a quantitative question rather than a matter of principle, and it has to be settled by computing both.

Returns
-------
np.ndarray of shape (2,), float: the peak element-centroid von Mises stress in Pa and the peak magnitude of the free-surface vertical displacement in m.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def compute_peak_stress_and_heave(order, width: float, depth: float, n_lateral: int,
                                  n_depth: int, temperature: np.ndarray,
                                  displacement: np.ndarray) -> np.ndarray:
    """Report the peak equivalent stress and the peak heave of the free surface.

    Parameters
    ----------
    order : sequence of int
        The four material numbers, from the free surface downwards; entries lie
        between 1 and 4 and may repeat.
    width : float
        Lateral extent of the near-field box in metres (width > 0).
    depth : float
        Depth of the near-field box in metres (depth > 0).
    n_lateral : int
        Number of elements along each lateral direction (n_lateral >= 1).
    n_depth : int
        Number of elements through the depth, a multiple of four
        (n_depth >= 1).
    temperature : np.ndarray
        Vector of length n_nodes holding the nodal temperature rise above the
        stress-free reference state, in kelvin.
    displacement : np.ndarray
        Vector of length 3 * n_nodes holding the nodal displacements in metres,
        ordered x, y, z at each node.

    Returns
    -------
    measures : np.ndarray
        Array of two floats: the largest von Mises stress over the element
        centroids in pascals, and the largest magnitude of the vertical
        displacement over the free-surface nodes in metres.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above.
    """
    return measures  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_compute_peak_stress_and_heave(order, width: float, depth: float, n_lateral: int,
                                          n_depth: int, temperature: np.ndarray,
                                          displacement: np.ndarray) -> np.ndarray:
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

    def _elasticity_matrix(modulus, poisson):
        """Return the isotropic elasticity matrix in the ordering xx, yy, zz, yz, zx, xy."""
        matrix = np.zeros((6, 6))
        factor = modulus / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        matrix[:3, :3] = factor * poisson
        matrix[0, 0] = matrix[1, 1] = matrix[2, 2] = factor * (1.0 - poisson)
        shear = factor * (1.0 - 2.0 * poisson) / 2.0
        matrix[3, 3] = matrix[4, 4] = matrix[5, 5] = shear
        return matrix

    def _strain_matrix(cartesian):
        """Return the strain-displacement operator of the brick from Cartesian derivatives."""
        operator = np.zeros((6, 24))
        operator[0, 0::3] = cartesian[0]
        operator[1, 1::3] = cartesian[1]
        operator[2, 2::3] = cartesian[2]
        operator[3, 1::3] = cartesian[2]
        operator[3, 2::3] = cartesian[1]
        operator[4, 0::3] = cartesian[2]
        operator[4, 2::3] = cartesian[0]
        operator[5, 0::3] = cartesian[1]
        operator[5, 1::3] = cartesian[0]
        return operator

    def _von_mises(components):
        """Return the von Mises invariant of a stress vector in engineering ordering."""
        xx, yy, zz, yz, zx, xy = components
        deviatoric = 0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2)
        return float(np.sqrt(deviatoric + 3.0 * (yz ** 2 + zx ** 2 + xy ** 2)))

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

    width = float(width)
    depth = float(depth)
    n_lateral = int(n_lateral)
    n_depth = int(n_depth)

    nodes, elements, stratum = _build_mesh(width, depth, n_lateral, n_depth)
    field = np.asarray(temperature, dtype=float)
    motion = np.asarray(displacement, dtype=float)
    if field.ndim != 1 or field.shape[0] != nodes.shape[0]:
        raise ValueError("temperature must be a vector with one entry per mesh node")
    if motion.ndim != 1 or motion.shape[0] != 3 * nodes.shape[0]:
        raise ValueError("displacement must be a vector with three entries per mesh node")
    if not (np.all(np.isfinite(field)) and np.all(np.isfinite(motion))):
        raise ValueError("temperature and displacement must contain only finite entries")

    peak_stress = 0.0
    for index, connectivity in enumerate(elements):
        _, _, modulus, poisson, expansion, _ = _MATERIALS[material_order[stratum[index]] - 1]
        elasticity = _elasticity_matrix(modulus, poisson)
        thermal = np.array([expansion, expansion, expansion, 0.0, 0.0, 0.0])
        dofs = np.array([3 * node + component
                         for node in connectivity for component in range(3)])
        shape, natural = _hex_shape(0.0, 0.0, 0.0)
        cartesian = np.linalg.solve(natural @ nodes[connectivity], natural)
        operator = _strain_matrix(cartesian)
        stress = elasticity @ (operator @ motion[dofs]) \
            - (elasticity @ thermal) * (shape @ field[connectivity])
        peak_stress = max(peak_stress, _von_mises(stress))

    surface = np.abs(nodes[:, 2]) < 1.0e-9
    peak_heave = float(np.max(np.abs(motion[2::3][surface]))) if np.any(surface) else 0.0
    return np.array([peak_stress, peak_heave], dtype=float)

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
        # --- Valid: benchmark mesh with a depth-decaying field and a heaving surface ---
        {
            "setup": """import numpy as np
order = [1, 2, 3, 4]
width, depth, n_lateral, n_depth = 120.0, 40.0, 4, 8
levels = np.repeat(np.linspace(0.0, depth, n_depth + 1), (n_lateral + 1) ** 2)
temperature = 70.0 * np.exp(-levels / 12.0)
displacement = np.zeros(3 * levels.size)
displacement[2::3] = -2.0e-3 * np.exp(-levels / 18.0)
displacement[0::3] = 4.0e-4 * np.cos(levels / 9.0)

def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(compute_peak_stress_and_heave(order, width, depth, n_lateral, n_depth, temperature, displacement))",
            "gold_call": "digest(_oracle_compute_peak_stress_and_heave(order, width, depth, n_lateral, n_depth, temperature, displacement))",
        },
        # --- Valid: inverted stratification changes which stratum carries the peak ---
        {
            "setup": """import numpy as np
order = (4, 3, 2, 1)
width, depth, n_lateral, n_depth = 120.0, 40.0, 4, 8
levels = np.repeat(np.linspace(0.0, depth, n_depth + 1), (n_lateral + 1) ** 2)
temperature = 70.0 * np.exp(-levels / 12.0)
displacement = np.zeros(3 * levels.size)
displacement[2::3] = -2.0e-3 * np.exp(-levels / 18.0)

def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(compute_peak_stress_and_heave(order, width, depth, n_lateral, n_depth, temperature, displacement))",
            "gold_call": "digest(_oracle_compute_peak_stress_and_heave(order, width, depth, n_lateral, n_depth, temperature, displacement))",
        },
        # --- Boundary: the coarsest admissible mesh, one element across each
        #     lateral direction and one per stratum, so every element touches
        #     the free surface and the base at once ---
        {
            "setup": """import numpy as np
order = [2, 3, 1, 4]
width, depth, n_lateral, n_depth = 120.0, 40.0, 1, 4
levels = np.repeat(np.linspace(0.0, depth, n_depth + 1), (n_lateral + 1) ** 2)
temperature = 60.0 * np.exp(-levels / 15.0)
displacement = np.zeros(3 * levels.size)
displacement[2::3] = -1.5e-3 * np.exp(-levels / 20.0)

def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(compute_peak_stress_and_heave(order, width, depth, n_lateral, n_depth, temperature, displacement))",
            "gold_call": "digest(_oracle_compute_peak_stress_and_heave(order, width, depth, n_lateral, n_depth, temperature, displacement))",
        },
        # --- Edge: uniform heating, whose stress is purely hydrostatic and so
        #     invisible to the von Mises measure, superposed on a shearing motion
        #     that is the only thing the measure can see ---
        {
            "setup": """import numpy as np
order = [3, 3, 3, 3]
width, depth, n_lateral, n_depth = 60.0, 20.0, 2, 4
levels = np.repeat(np.linspace(0.0, depth, n_depth + 1), (n_lateral + 1) ** 2)
temperature = np.full(levels.size, 100.0)
displacement = np.zeros(3 * levels.size)
displacement[0::3] = 8.0e-4 * (1.0 - levels / depth)
displacement[2::3] = 3.0e-4 * (1.0 - levels / depth)

def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(compute_peak_stress_and_heave(order, width, depth, n_lateral, n_depth, temperature, displacement))",
            "gold_call": "digest(_oracle_compute_peak_stress_and_heave(order, width, depth, n_lateral, n_depth, temperature, displacement))",
        },
        # --- Invalid: displacement vector of the wrong length ---
        {
            "setup": """import numpy as np
count = 5 ** 2 * 5
temperature = np.zeros(count)
displacement = np.zeros(count)
def run_model():
    try:
        compute_peak_stress_and_heave([1, 2, 3, 4], 120.0, 40.0, 4, 4, temperature, displacement)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_peak_stress_and_heave([1, 2, 3, 4], 120.0, 40.0, 4, 4, temperature, displacement)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a material number outside the available set ---
        {
            "setup": """import numpy as np
count = 5 ** 2 * 5
temperature = np.zeros(count)
displacement = np.zeros(3 * count)
def run_model():
    try:
        compute_peak_stress_and_heave([0, 2, 3, 4], 120.0, 40.0, 4, 4, temperature, displacement)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_peak_stress_and_heave([0, 2, 3, 4], 120.0, 40.0, 4, 4, temperature, displacement)
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
