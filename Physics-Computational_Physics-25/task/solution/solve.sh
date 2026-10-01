#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

# ORACLE SOLUTION


def assemble_near_field_operators(order, width: float, depth: float, patch_width: float,
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

# ORACLE SOLUTION


def assemble_patch_load_vector(width: float, depth: float, patch_width: float,
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

# ORACLE SOLUTION


def assemble_scaling_surface_coefficients(width: float, depth: float, n_lateral: int,
                                                  far_conductivity: float, contraction: float,
                                                  radial_coordinate: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    _QUAD_SIGNS = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

    _GAUSS_2PT = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))

    def _quad_shape(r, s):
        """Return shape functions and natural derivatives of the bilinear quadrilateral."""
        sr, ss = _QUAD_SIGNS[:, 0], _QUAD_SIGNS[:, 1]
        shape = 0.25 * (1.0 + sr * r) * (1.0 + ss * s)
        grad = np.empty((2, 4))
        grad[0] = 0.25 * sr * (1.0 + ss * s)
        grad[1] = 0.25 * (1.0 + sr * r) * ss
        return shape, grad

    def _surface_meshes(width, depth, n_lateral, contraction):
        """Return the boundary-surface nodes, the scaling-surface nodes and the faces.

        The boundary surface is the base of the near-field box at z = depth, carrying
        the quadrilaterals the near-field mesh induces there. The scaling surface is
        that same square contracted about the vertical axis through the box centre
        and laid on the free surface z = 0. Surface nodes are numbered i + (n+1) j.
        """
        lateral = np.linspace(0.0, width, n_lateral + 1)
        count = (n_lateral + 1) ** 2
        boundary = np.empty((count, 3))
        for j in range(n_lateral + 1):
            for i in range(n_lateral + 1):
                boundary[i + (n_lateral + 1) * j] = (lateral[i], lateral[j], depth)
        scaling = boundary.copy()
        scaling[:, 0] = 0.5 * width + contraction * (boundary[:, 0] - 0.5 * width)
        scaling[:, 1] = 0.5 * width + contraction * (boundary[:, 1] - 0.5 * width)
        scaling[:, 2] = 0.0
        faces = []
        for j in range(n_lateral):
            for i in range(n_lateral):
                faces.append([i + (n_lateral + 1) * j, (i + 1) + (n_lateral + 1) * j,
                              (i + 1) + (n_lateral + 1) * (j + 1), i + (n_lateral + 1) * (j + 1)])
        return boundary, scaling, np.array(faces, dtype=int)

    for name, value in (("width", width), ("depth", depth),
                        ("far_conductivity", far_conductivity),
                        ("radial_coordinate", radial_coordinate)):
        if not (isinstance(value, (int, float)) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(n_lateral, (int, np.integer)) and not isinstance(n_lateral, bool)
            and int(n_lateral) >= 1):
        raise ValueError("n_lateral must be an integer >= 1")
    if not (isinstance(contraction, (int, float)) and np.isfinite(contraction)
            and 0.0 <= float(contraction) < 1.0):
        raise ValueError("contraction must be a finite number in the interval [0, 1)")

    width = float(width)
    depth = float(depth)
    n_lateral = int(n_lateral)
    far_conductivity = float(far_conductivity)
    contraction = float(contraction)
    radial_coordinate = float(radial_coordinate)

    boundary, scaling, faces = _surface_meshes(width, depth, n_lateral, contraction)
    count = boundary.shape[0]
    radial = np.zeros((count, count), dtype=float)
    cross = np.zeros((count, count), dtype=float)
    circumferential = np.zeros((count, count), dtype=float)

    for face in faces:
        offset = boundary[face] - scaling[face]
        for r in _GAUSS_2PT:
            for s in _GAUSS_2PT:
                shape, natural = _quad_shape(r, s)
                # Position of the swept surface at the frozen radial coordinate.
                swept = scaling[face] + radial_coordinate * offset
                jacobian = np.vstack([shape @ offset, natural[0] @ swept, natural[1] @ swept])
                detj = np.linalg.det(jacobian)
                inverse = np.linalg.inv(jacobian)
                # Columns of the inverse Jacobian are the gradients of the mapped
                # coordinates; the first drives the radial derivative and the other
                # two drive the circumferential derivatives.
                b_radial = np.outer(inverse[:, 0], shape)
                b_circ = (np.outer(inverse[:, 1], natural[0])
                          + np.outer(inverse[:, 2], natural[1]))
                radial[np.ix_(face, face)] += \
                    far_conductivity * (b_radial.T @ b_radial) * detj
                cross[np.ix_(face, face)] += \
                    far_conductivity * (b_circ.T @ b_radial) * detj
                circumferential[np.ix_(face, face)] += \
                    far_conductivity * (b_circ.T @ b_circ) * detj

    # Characteristic length of the mapping and its logarithmic radial derivative.
    root_boundary = np.sqrt(width ** 2)                    # square root of the boundary area
    root_scaling = np.sqrt((contraction * width) ** 2)     # square root of the scaling area
    length = root_scaling + (root_boundary - root_scaling) * radial_coordinate
    divergence = (root_boundary - root_scaling) / length
    if divergence <= 0.0:
        raise ValueError("the scaling and boundary surfaces must define a diverging exterior")

    return np.stack([divergence * radial, cross, circumferential / divergence])

# ORACLE SOLUTION


def solve_far_field_conduction_matrix(radial_matrix: np.ndarray,
                                              cross_matrix: np.ndarray,
                                              circumferential_matrix: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _validate_coefficient_block(name, matrix, order):
        """Raise ValueError unless matrix is a finite square array of the given order."""
        array = np.asarray(matrix, dtype=float)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError(f"{name} must be a square two-dimensional array")
        if order is not None and array.shape[0] != order:
            raise ValueError("the coefficient matrices must all have the same order")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite entries")
        return array

    radial = _validate_coefficient_block("radial_matrix", radial_matrix, None)
    order = radial.shape[0]
    cross = _validate_coefficient_block("cross_matrix", cross_matrix, order)
    circumferential = _validate_coefficient_block(
        "circumferential_matrix", circumferential_matrix, order)
    if np.linalg.matrix_rank(radial) < order:
        raise ValueError("radial_matrix must be non-singular")
    if np.min(np.linalg.eigvalsh(0.5 * (radial + radial.T))) <= 0.0:
        raise ValueError("radial_matrix must be positive definite")

    inverse_radial = np.linalg.inv(radial)
    identity = np.eye(order)
    # State matrix of the first-order radial system for interface temperatures
    # and interface fluxes. The half-identity shifts split the linear term that
    # the characteristic-length rescaling contributes to the matrix quadratic.
    state = np.block([
        [-inverse_radial @ cross.T + 0.5 * identity, -inverse_radial],
        [cross @ inverse_radial @ cross.T - circumferential,
         cross @ inverse_radial - 0.5 * identity]])

    values, vectors = np.linalg.eig(state)
    # The exterior is unbounded, so the admissible modes are the decaying half.
    decaying = np.argsort(values.real)[:order]
    if np.max(values.real[decaying]) >= 0.0:
        raise ValueError("the coefficient matrices do not admit a decaying half-spectrum")

    temperature_block = vectors[:order, decaying]
    flux_block = vectors[order:, decaying]
    conduction = (flux_block @ np.linalg.inv(temperature_block)).real
    return 0.5 * (conduction + conduction.T)

# ORACLE SOLUTION


def compute_precise_propagator(state_matrix: np.ndarray, interval: float,
                                       n_levels: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _validate_state_matrix(state_matrix):
        """Raise ValueError unless the argument is a finite square array."""
        array = np.asarray(state_matrix, dtype=float)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError("state_matrix must be a square two-dimensional array")
        if array.shape[0] == 0:
            raise ValueError("state_matrix must have at least one row")
        if not np.all(np.isfinite(array)):
            raise ValueError("state_matrix must contain only finite entries")
        return array

    array = _validate_state_matrix(state_matrix)
    if not (isinstance(interval, (int, float)) and np.isfinite(interval)
            and float(interval) >= 0.0):
        raise ValueError("interval must be a finite number >= 0")
    if not (isinstance(n_levels, (int, np.integer)) and not isinstance(n_levels, bool)
            and int(n_levels) >= 0):
        raise ValueError("n_levels must be an integer >= 0")

    interval = float(interval)
    n_levels = int(n_levels)

    # Exponential of one subdivided part, as an increment on the identity.
    part = array * (interval / 2.0 ** n_levels)
    square = part @ part
    increment = part + 0.5 * square + (square @ part) / 6.0

    # Doubling recursion carried out on the increment alone, so that a quantity
    # far smaller than unity is never added to the identity until the end.
    for _ in range(n_levels):
        increment = 2.0 * increment + increment @ increment

    return np.eye(array.shape[0]) + increment

# ORACLE SOLUTION


def integrate_temperature_field(step_propagator: np.ndarray, tail_propagators: list,
                                        load_influence: np.ndarray, ambient_amplitude: float,
                                        ramp_time: float, step: float,
                                        n_steps: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _validate_propagators(step_propagator, tail_propagators):
        """Raise ValueError unless the supplied propagators are square and consistent."""
        whole = np.asarray(step_propagator, dtype=float)
        if whole.ndim != 2 or whole.shape[0] != whole.shape[1] or whole.shape[0] == 0:
            raise ValueError("step_propagator must be a non-empty square two-dimensional array")
        try:
            tails = [np.asarray(tail, dtype=float) for tail in tail_propagators]
        except TypeError:
            raise ValueError("tail_propagators must be a sequence of arrays")
        if len(tails) < 1:
            raise ValueError("tail_propagators must hold at least one propagator")
        if any(tail.shape != whole.shape for tail in tails):
            raise ValueError("every tail propagator must have the shape of step_propagator")
        if not (np.all(np.isfinite(whole)) and all(np.all(np.isfinite(tail)) for tail in tails)):
            raise ValueError("the propagators must contain only finite entries")
        return whole, tails

    whole, tails = _validate_propagators(step_propagator, tail_propagators)
    load = np.asarray(load_influence, dtype=float)
    if load.ndim != 1 or load.shape[0] != whole.shape[0]:
        raise ValueError("load_influence must be a vector of the same order as the propagators")
    if not np.all(np.isfinite(load)):
        raise ValueError("load_influence must contain only finite entries")
    if not (isinstance(ambient_amplitude, (int, float)) and np.isfinite(ambient_amplitude)):
        raise ValueError("ambient_amplitude must be a finite number")
    for name, value in (("ramp_time", ramp_time), ("step", step)):
        if not (isinstance(value, (int, float)) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(n_steps, (int, np.integer)) and not isinstance(n_steps, bool)
            and int(n_steps) >= 0):
        raise ValueError("n_steps must be an integer >= 0")

    ambient_amplitude = float(ambient_amplitude)
    ramp_time = float(ramp_time)
    step = float(step)
    n_steps = int(n_steps)

    # The rule is fixed by how many tail propagators were supplied, one per
    # abscissa, so the caller and the march cannot disagree about its order.
    abscissae, weights = np.polynomial.legendre.leggauss(len(tails))
    driven = [tail @ load for tail in tails]

    temperature = np.zeros(whole.shape[0], dtype=float)
    for index in range(n_steps):
        start = index * step
        particular = np.zeros_like(temperature)
        for weight, node, contribution in zip(weights, abscissae, driven):
            sample = start + 0.5 * step * (1.0 + node)
            ambient = ambient_amplitude * (1.0 - np.exp(-sample / ramp_time))
            particular += weight * ambient * contribution
        temperature = whole @ temperature + 0.5 * step * particular
    return temperature

# ORACLE SOLUTION


def solve_thermoelastic_displacement(order, width: float, depth: float, n_lateral: int,
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

# ORACLE SOLUTION


def compute_peak_stress_and_heave(order, width: float, depth: float, n_lateral: int,
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

# ORACLE SOLUTION


def compute_surface_gradient(width: float, depth: float, n_lateral: int, n_depth: int,
                                     temperature: np.ndarray) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

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

    gradient = 0.0
    _, natural = _hex_shape(0.0, 0.0, 0.0)
    for index, connectivity in enumerate(elements):
        if stratum[index] != 0:
            continue
        cartesian = np.linalg.solve(natural @ nodes[connectivity], natural)
        gradient = max(gradient, abs(float((cartesian @ field[connectivity])[2])))
    return float(gradient)

# ORACLE SOLUTION


def run_constraint_dominance_pipeline(width: float = 120.0, depth: float = 40.0,
                                              patch_width: float = 60.0, n_lateral: int = 4,
                                              n_depth: int = 8, film_coefficient: float = 25.0,
                                              ambient_amplitude: float = 80.0,
                                              ramp_time: float = 1.0e5, step: float = 2500.0,
                                              n_steps: int = 80, n_gauss: int = 6,
                                              n_levels: int = 20, contraction: float = 0.5,
                                              radial_coordinate: float = 1.0) -> float:
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

    # The three stratifications compared, listed from the free surface downwards.
    _STRATIFICATIONS = ((1, 2, 3, 4), (2, 3, 1, 4), (4, 3, 2, 1))

    def _interface_nodes(n_lateral, n_depth):
        """Return the global indices of the base-face nodes, in surface-local order."""
        return [i + (n_lateral + 1) * (j + (n_lateral + 1) * n_depth)
                for j in range(n_lateral + 1) for i in range(n_lateral + 1)]

    # -- The earlier steps are called by their reference names, so that this
    #    value is produced by the reference chain alone and never by whatever
    #    implementation of an earlier step happens to be in scope.

    # -- Validate the orchestrator inputs.
    for name, value, floor in (("n_lateral", n_lateral, 1), ("n_depth", n_depth, 1),
                               ("n_steps", n_steps, 0), ("n_gauss", n_gauss, 1),
                               ("n_levels", n_levels, 0)):
        if not (isinstance(value, (int, np.integer)) and not isinstance(value, bool)
                and int(value) >= floor):
            raise ValueError(f"{name} must be an integer >= {floor}")
    for name, value in (("width", width), ("depth", depth), ("ramp_time", ramp_time),
                        ("step", step), ("radial_coordinate", radial_coordinate)):
        if not (isinstance(value, (int, float)) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    for name, value in (("patch_width", patch_width),
                        ("film_coefficient", film_coefficient)):
        if not (isinstance(value, (int, float)) and np.isfinite(value) and float(value) >= 0.0):
            raise ValueError(f"{name} must be a finite number >= 0")
    if not (isinstance(contraction, (int, float)) and np.isfinite(contraction)
            and 0.0 <= float(contraction) < 1.0):
        raise ValueError("contraction must be a finite number in the interval [0, 1)")
    if int(n_depth) % 4 != 0:
        raise ValueError("n_depth must be a multiple of 4 so that every element lies in one stratum")

    n_lateral = int(n_lateral)
    n_depth = int(n_depth)
    n_gauss = int(n_gauss)
    n_levels = int(n_levels)
    interface = _interface_nodes(n_lateral, n_depth)

    peak_stress = []
    peak_gradient = []
    for order in _STRATIFICATIONS:
        # -- Sub-problems 01-02: the near-field operators and the patch load.
        operators = assemble_near_field_operators(list(order), width, depth, patch_width,
                                                  n_lateral, n_depth, film_coefficient)
        conductance = np.array(operators[0], dtype=float)
        capacity = np.array(operators[1], dtype=float)
        load = assemble_patch_load_vector(width, depth, patch_width, n_lateral, n_depth,
                                          film_coefficient)

        # -- Sub-problems 03-04: the unbounded exterior, condensed onto the base.
        coefficients = assemble_scaling_surface_coefficients(width, depth, n_lateral,
                                                             _MATERIALS[order[3] - 1][0],
                                                             contraction, radial_coordinate)
        far_field = solve_far_field_conduction_matrix(coefficients[0], coefficients[1],
                                                      coefficients[2])
        conductance[np.ix_(interface, interface)] += far_field

        # -- Sub-problems 05-06: the semi-discrete system, the propagators the
        #    march repeats at every step, and the march itself. The propagator
        #    over a whole step and the one over the tail of each quadrature
        #    abscissa depend only on the state matrix, the step and the number
        #    of subdivision levels, so they are built here once per order and
        #    handed to the march.
        state = -np.linalg.solve(capacity, conductance)
        influence = np.linalg.solve(capacity, load)
        abscissae = np.polynomial.legendre.leggauss(n_gauss)[0]
        step_propagator = compute_precise_propagator(state, step, n_levels)
        tail_propagators = [compute_precise_propagator(state, 0.5 * step * (1.0 - node), n_levels)
                            for node in abscissae]
        temperature = integrate_temperature_field(step_propagator, tail_propagators, influence,
                                                  ambient_amplitude, ramp_time, step, n_steps)

        # -- Sub-problems 07-09: the mechanical response and the thermal front.
        displacement = solve_thermoelastic_displacement(list(order), width, depth,
                                                        n_lateral, n_depth, temperature)
        measures = compute_peak_stress_and_heave(list(order), width, depth, n_lateral,
                                                 n_depth, temperature, displacement)
        peak_stress.append(float(measures[0]))
        peak_gradient.append(float(compute_surface_gradient(width, depth, n_lateral,
                                                            n_depth, temperature)))

    stress = np.array(peak_stress, dtype=float)
    gradient = np.array(peak_gradient, dtype=float)
    if np.min(stress) <= 0.0 or np.min(gradient) <= 0.0:
        raise ValueError("the configuration produced a vanishing stress or gradient spread")
    return float((stress.max() / stress.min()) / (gradient.max() / gradient.min()))
SCICODE_GOLD_EOF
