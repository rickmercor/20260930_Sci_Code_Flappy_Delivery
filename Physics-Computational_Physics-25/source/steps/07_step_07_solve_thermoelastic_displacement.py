"""
Solve the one-way coupled elastic problem driven by a given temperature field, with the base restrained and the vertical faces on normal rollers.

In a sequentially coupled thermo-elastic analysis the temperature field is computed first and then applied to the elastic model as a load, the displacement field being assumed not to feed back into the thermal field. The assumption is exact for the block of the coupled operator that would carry the feedback, which is identically zero when mechanical dissipation is negligible against the imposed thermal loading, and it is what allows a single elastic solve at the instant of interest rather than a stepwise mechanical march. Elastostatics has no memory, so the displacement at a given time depends only on the temperature field at that time and not on its history.




The load itself follows from splitting the total strain into an elastic part and a thermal part. Only the elastic part generates stress, so the constitutive law applied to the total strain must be corrected by the elasticity matrix acting on the thermal strain, and moving that correction to the right-hand side of the equilibrium statement produces a body-force-like load equal to the integral of the strain-displacement operator transposed against the elasticity matrix times the thermal expansion vector times the interpolated temperature. The thermal expansion vector has entries only in the three direct components, because free thermal expansion is isotropic and produces no shear. A consequence that is easy to overlook is that the load is not proportional to the temperature gradient but to the temperature itself weighted by the local product of stiffness and expansion coefficient, so two strata at the same temperature contribute very differently if their thermoelastic properties differ.




The restraints are what convert expansion into stress. A body free to expand in all directions under a uniform temperature rise develops displacement but no stress at all; stress appears only where expansion is prevented. Placing normal rollers on the four vertical faces suppresses lateral expansion while leaving the vertical direction free, which reproduces the confinement a heated region experiences from the cold material surrounding it, and fully restraining the base anchors the model against the far field. The free surface is left unloaded, so the heated region can heave upwards, and the competition between that heave and the lateral confinement is exactly the mechanism that decides how much of the thermal strain is realised as deformation and how much is converted into stress.

Returns
-------
np.ndarray of shape (3 * n_nodes,), float: the nodal displacement field in metres, with the restrained degrees of freedom left at zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def solve_thermoelastic_displacement(order, width: float, depth: float, n_lateral: int,
                                     n_depth: int, temperature: np.ndarray) -> np.ndarray:
    """Solve the restrained elastic problem driven by a temperature field.

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

    Returns
    -------
    displacement : np.ndarray
        Vector of length 3 * n_nodes holding the nodal displacements in
        metres, ordered x, y, z at each node.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above.
    """
    return displacement  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_solve_thermoelastic_displacement(order, width: float, depth: float, n_lateral: int,
                                             n_depth: int, temperature: np.ndarray) -> np.ndarray:
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

    def _restrained_dofs(nodes, width, depth):
        """Return the set of degrees of freedom removed by the mechanical restraints."""
        fixed = set()
        tolerance = 1.0e-9
        for index, (x, y, z) in enumerate(nodes):
            if abs(z - depth) < tolerance:
                fixed.update((3 * index, 3 * index + 1, 3 * index + 2))
                continue
            if abs(x) < tolerance or abs(x - width) < tolerance:
                fixed.add(3 * index)
            if abs(y) < tolerance or abs(y - width) < tolerance:
                fixed.add(3 * index + 1)
        return fixed

    material_order = _validate_order(order)
    _validate_mesh_arguments(width, depth, n_lateral, n_depth)

    width = float(width)
    depth = float(depth)
    n_lateral = int(n_lateral)
    n_depth = int(n_depth)

    nodes, elements, stratum = _build_mesh(width, depth, n_lateral, n_depth)
    field = np.asarray(temperature, dtype=float)
    if field.ndim != 1 or field.shape[0] != nodes.shape[0]:
        raise ValueError("temperature must be a vector with one entry per mesh node")
    if not np.all(np.isfinite(field)):
        raise ValueError("temperature must contain only finite entries")

    n_dof = 3 * nodes.shape[0]
    stiffness = np.zeros((n_dof, n_dof), dtype=float)
    load = np.zeros(n_dof, dtype=float)

    for index, connectivity in enumerate(elements):
        _, _, modulus, poisson, expansion, _ = _MATERIALS[material_order[stratum[index]] - 1]
        elasticity = _elasticity_matrix(modulus, poisson)
        thermal = np.array([expansion, expansion, expansion, 0.0, 0.0, 0.0])
        coordinates = nodes[connectivity]
        dofs = np.array([3 * node + component
                         for node in connectivity for component in range(3)])
        element_stiffness = np.zeros((24, 24))
        element_load = np.zeros(24)
        for r in _GAUSS_2PT:
            for s in _GAUSS_2PT:
                for t in _GAUSS_2PT:
                    shape, natural = _hex_shape(r, s, t)
                    jacobian = natural @ coordinates
                    detj = np.linalg.det(jacobian)
                    cartesian = np.linalg.solve(jacobian, natural)
                    operator = _strain_matrix(cartesian)
                    element_stiffness += (operator.T @ elasticity @ operator) * detj
                    element_load += (operator.T @ (elasticity @ thermal)) \
                        * (shape @ field[connectivity]) * detj
        stiffness[np.ix_(dofs, dofs)] += element_stiffness
        load[dofs] += element_load

    fixed = _restrained_dofs(nodes, width, depth)
    free = np.array([dof for dof in range(n_dof) if dof not in fixed], dtype=int)
    displacement = np.zeros(n_dof, dtype=float)
    if free.size:
        displacement[free] = np.linalg.solve(stiffness[np.ix_(free, free)], load[free])
    return displacement

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
        # --- Valid: benchmark mesh with an exponential depth profile (normal scenario) ---
        {
            "setup": """import numpy as np
order = [1, 2, 3, 4]
width, depth, n_lateral, n_depth = 120.0, 40.0, 4, 8
levels = np.linspace(0.0, depth, n_depth + 1)
temperature = 70.0 * np.exp(-np.repeat(levels, (n_lateral + 1) ** 2) / 12.0)

def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(solve_thermoelastic_displacement(order, width, depth, n_lateral, n_depth, temperature))",
            "gold_call": "digest(_oracle_solve_thermoelastic_displacement(order, width, depth, n_lateral, n_depth, temperature))",
        },
        # --- Valid: inverted stratification on the same profile ---
        {
            "setup": """import numpy as np
order = (4, 3, 2, 1)
width, depth, n_lateral, n_depth = 120.0, 40.0, 4, 4
levels = np.linspace(0.0, depth, n_depth + 1)
temperature = 70.0 * np.exp(-np.repeat(levels, (n_lateral + 1) ** 2) / 12.0)

def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(solve_thermoelastic_displacement(order, width, depth, n_lateral, n_depth, temperature))",
            "gold_call": "digest(_oracle_solve_thermoelastic_displacement(order, width, depth, n_lateral, n_depth, temperature))",
        },
        # --- Boundary: an isothermal field at the reference state gives no displacement ---
        {
            "setup": """import numpy as np
order = [2, 3, 1, 4]
width, depth, n_lateral, n_depth = 120.0, 40.0, 2, 4
temperature = np.zeros((n_lateral + 1) ** 2 * (n_depth + 1))

def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(solve_thermoelastic_displacement(order, width, depth, n_lateral, n_depth, temperature))",
            "gold_call": "digest(_oracle_solve_thermoelastic_displacement(order, width, depth, n_lateral, n_depth, temperature))",
        },
        # --- Edge: one material throughout, heated uniformly against the restraints ---
        {
            "setup": """import numpy as np
order = [3, 3, 3, 3]
width, depth, n_lateral, n_depth = 60.0, 20.0, 2, 4
temperature = np.full((n_lateral + 1) ** 2 * (n_depth + 1), 100.0)

def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(solve_thermoelastic_displacement(order, width, depth, n_lateral, n_depth, temperature))",
            "gold_call": "digest(_oracle_solve_thermoelastic_displacement(order, width, depth, n_lateral, n_depth, temperature))",
        },
        # --- Invalid: temperature vector of the wrong length ---
        {
            "setup": """import numpy as np
bad = np.zeros(7)
def run_model():
    try:
        solve_thermoelastic_displacement([1, 2, 3, 4], 120.0, 40.0, 4, 4, bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_thermoelastic_displacement([1, 2, 3, 4], 120.0, 40.0, 4, 4, bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: stratification of the wrong length ---
        {
            "setup": """import numpy as np
field = np.zeros(5 ** 2 * 5)
def run_model():
    try:
        solve_thermoelastic_displacement([1, 2, 3], 120.0, 40.0, 4, 4, field)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_thermoelastic_displacement([1, 2, 3], 120.0, 40.0, 4, 4, field)
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
