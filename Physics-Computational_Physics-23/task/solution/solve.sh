#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np
def time_spectral_operator(num_nodes: int, time_step: float) -> tuple:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if isinstance(num_nodes, bool) or not isinstance(num_nodes, (int, np.integer)):
        raise ValueError("num_nodes must be an integer")
    if int(num_nodes) < 2:
        raise ValueError("num_nodes must be an integer >= 2")
    if isinstance(time_step, bool) or not isinstance(time_step, (int, float, np.integer, np.floating)):
        raise ValueError("time_step must be a real number")
    if not np.isfinite(float(time_step)) or float(time_step) <= 0.0:
        raise ValueError("time_step must be a finite number > 0")

    order = int(num_nodes)
    step = float(time_step)

    # Gauss-Legendre nodes and weights of the reference interval [-1, 1].
    nodes, weights = np.polynomial.legendre.leggauss(order)

    # Spectral integration matrix. The k-th Legendre coefficient of a function
    # sampled at the nodes is (1 + 2k)/2 times the quadrature sum of the
    # function against P_k, and the running integral of the expansion from -1
    # to each node collects those coefficients against the antiderivatives of
    # the Legendre polynomials.
    spectral = np.zeros((order, order), dtype=float)
    for degree in range(order):
        basis = np.zeros(degree + 1, dtype=float)
        basis[degree] = 1.0
        values = np.polynomial.legendre.legval(nodes, basis)
        primitive = np.polynomial.legendre.legint(basis)
        running = (np.polynomial.legendre.legval(nodes, primitive)
                   - float(np.polynomial.legendre.legval(-1.0, primitive)))
        spectral += (0.5 * (1.0 + 2.0 * degree)) * np.outer(running, weights * values)

    # The reference interval carries a factor of half the step length, so that
    # the running integral in physical time is step * spectral applied to the
    # nodal values.
    spectral *= 0.5

    times = (1.0 + nodes) * step / 2.0
    operator = np.linalg.inv(spectral) / step
    return (np.asarray(times, dtype=float), np.asarray(operator, dtype=float))

import numpy as np

def rectangular_layer_mesh(length: float, y_bottom: float, y_top: float,
                                   n_long: int, n_short: int, n_x: int, n_y: int) -> tuple:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    for name, value in (("length", length), ("y_bottom", y_bottom), ("y_top", y_top)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(length) <= 0.0:
        raise ValueError("length must be a finite number > 0")
    if float(y_top) <= float(y_bottom):
        raise ValueError("y_top must be greater than y_bottom")
    counts = {"n_long": n_long, "n_short": n_short, "n_x": n_x, "n_y": n_y}
    for name, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if int(value) < 1:
            raise ValueError(f"{name} must be an integer >= 1")

    span = float(length)
    low = float(y_bottom)
    high = float(y_top)

    def _face(start, end, count):
        # Subdivide one straight face into quadratic elements whose middle node
        # sits at the midpoint, keeping the traversal direction of the face.
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        out = []
        for index in range(int(count)):
            first = start + (end - start) * (index / float(count))
            second = start + (end - start) * ((index + 1) / float(count))
            out.append(np.array([first, 0.5 * (first + second), second], dtype=float))
        return out

    # Anticlockwise traversal of the rectangle keeps the swept area positive
    # when it is seen from any point inside the layer.
    corners = [(0.0, low), (span, low), (span, high), (0.0, high)]
    elements = []
    elements += _face(corners[0], corners[1], n_long)
    elements += _face(corners[1], corners[2], n_short)
    elements += _face(corners[2], corners[3], n_long)
    elements += _face(corners[3], corners[0], n_short)

    columns = (np.arange(int(n_x), dtype=float) + 0.5) * span / float(n_x)
    rows = low + (np.arange(int(n_y), dtype=float) + 0.5) * (high - low) / float(n_y)
    grid_x, grid_y = np.meshgrid(columns, rows, indexing="ij")
    interior = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    return (np.asarray(elements, dtype=float), np.asarray(interior, dtype=float))

import numpy as np
def layer_reference_fields(points: "np.ndarray", time: float, slope: float,
                                   conductivity: float) -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    sample = np.asarray(points, dtype=float)
    if sample.ndim != 2 or sample.shape[0] < 1 or sample.shape[1] != 2:
        raise ValueError("points must have shape (n, 2) with n >= 1")
    if not np.all(np.isfinite(sample)):
        raise ValueError("points must be finite")
    for name, value in (("time", time), ("slope", slope), ("conductivity", conductivity)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(conductivity) <= 0.0:
        raise ValueError("conductivity must be a finite number > 0")

    x_coord = sample[:, 0]
    y_coord = sample[:, 1]
    growth = np.exp(float(time))
    tilt = float(slope)

    # The benchmark field and the two derivatives that build the boundary flux.
    temperature = (tilt * y_coord + 3.0 - np.sin(x_coord)) * growth
    d_dx = -np.cos(x_coord) * growth
    d_dy = np.full_like(x_coord, tilt) * growth

    # The field grows like exp(t), so its time derivative equals the field
    # itself, and its Laplacian comes only from the sine in x. The source
    # density is what the transient balance leaves over.
    laplacian = np.sin(x_coord) * growth
    source = laplacian - temperature / float(conductivity)

    return np.column_stack([temperature, d_dx, d_dy, source])

import numpy as np
def near_singular_rule(source_point: "np.ndarray", element_nodes: "np.ndarray",
                               num_gauss: int, use_sinh: bool) -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    point = np.asarray(source_point, dtype=float)
    nodes = np.asarray(element_nodes, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("source_point must be a finite array of shape (2,)")
    if nodes.shape != (3, 2) or not np.all(np.isfinite(nodes)):
        raise ValueError("element_nodes must be a finite array of shape (3, 2)")
    if isinstance(num_gauss, bool) or not isinstance(num_gauss, (int, np.integer)):
        raise ValueError("num_gauss must be an integer")
    if int(num_gauss) < 1:
        raise ValueError("num_gauss must be an integer >= 1")
    if not isinstance(use_sinh, (bool, np.bool_)):
        raise ValueError("use_sinh must be a boolean")

    half = 0.5 * (nodes[2] - nodes[0])
    half_length = float(np.hypot(half[0], half[1]))
    if half_length <= 0.0:
        raise ValueError("element_nodes must describe an element of non-zero length")
    if float(np.hypot(*(nodes[1] - 0.5 * (nodes[0] + nodes[2])))) > 1.0e-9 * half_length:
        raise ValueError("the middle node must be the midpoint of the element")

    points, weights = np.polynomial.legendre.leggauss(int(num_gauss))
    if not bool(use_sinh):
        return np.column_stack([np.asarray(points, dtype=float),
                                np.asarray(weights, dtype=float)])

    # Foot of the perpendicular from the source point, expressed in the element
    # local coordinate, and the distance to it in units of the half length.
    foot_local = float(np.dot(point - nodes[1], half) / np.dot(half, half))
    foot = nodes[1] + foot_local * half
    offset = float(np.hypot(*(point - foot))) / half_length
    if offset <= 0.0:
        raise ValueError("source_point must not lie on the line carrying the element")

    # Requiring t = -1 and t = 1 to land on the two element ends fixes the two
    # constants of the change of variable.
    to_far = float(np.arcsinh((1.0 + foot_local) / offset))
    to_near = float(np.arcsinh((1.0 - foot_local) / offset))
    stretch = 0.5 * (to_far + to_near)
    shift = 0.5 * (to_far - to_near)

    argument = stretch * points - shift
    local = foot_local + offset * np.sinh(argument)
    scaled = weights * offset * stretch * np.cosh(argument)
    return np.column_stack([np.asarray(local, dtype=float), np.asarray(scaled, dtype=float)])

import numpy as np
def element_boundary_terms(source_point: "np.ndarray", element_nodes: "np.ndarray",
                                   rule: "np.ndarray") -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    point = np.asarray(source_point, dtype=float)
    nodes = np.asarray(element_nodes, dtype=float)
    table = np.asarray(rule, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("source_point must be a finite array of shape (2,)")
    if nodes.shape != (3, 2) or not np.all(np.isfinite(nodes)):
        raise ValueError("element_nodes must be a finite array of shape (3, 2)")
    if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] != 2:
        raise ValueError("rule must have shape (m, 2) with m >= 1")
    if not np.all(np.isfinite(table)):
        raise ValueError("rule must be finite")

    local = table[:, 0]
    weight = table[:, 1]

    # Quadratic shape functions with nodes at -1, 0 and 1, and their slopes.
    shape = np.column_stack([0.5 * local * (local - 1.0),
                             1.0 - local ** 2,
                             0.5 * local * (local + 1.0)])
    slope = np.column_stack([local - 0.5, -2.0 * local, local + 0.5])

    coords = shape @ nodes
    tangent = slope @ nodes
    jacobian = np.hypot(tangent[:, 0], tangent[:, 1])
    if np.any(jacobian <= 0.0):
        raise ValueError("the element tangent vanishes at a quadrature point")

    # Anticlockwise traversal puts the outward normal to the right of the
    # tangent.
    normal_x = tangent[:, 1] / jacobian
    normal_y = -tangent[:, 0] / jacobian

    offset_x = coords[:, 0] - point[0]
    offset_y = coords[:, 1] - point[1]
    distance2 = offset_x ** 2 + offset_y ** 2
    if np.any(distance2 <= 0.0):
        raise ValueError("source_point coincides with a quadrature point")

    # Fundamental solution of the Laplace operator and its normal derivative.
    fundamental = -0.25 / np.pi * np.log(distance2)
    flux_kernel = -0.5 / np.pi * (offset_x * normal_x + offset_y * normal_y) / distance2

    return np.column_stack([coords[:, 0], coords[:, 1],
                            weight * jacobian * fundamental,
                            weight * jacobian * flux_kernel,
                            normal_x, normal_y])

import numpy as np
def sct_domain_terms(source_point: "np.ndarray", element_nodes: "np.ndarray",
                             rule: "np.ndarray", num_radial: int) -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    point = np.asarray(source_point, dtype=float)
    nodes = np.asarray(element_nodes, dtype=float)
    table = np.asarray(rule, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("source_point must be a finite array of shape (2,)")
    if nodes.shape != (3, 2) or not np.all(np.isfinite(nodes)):
        raise ValueError("element_nodes must be a finite array of shape (3, 2)")
    if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] != 2:
        raise ValueError("rule must have shape (m, 2) with m >= 1")
    if not np.all(np.isfinite(table)):
        raise ValueError("rule must be finite")
    if isinstance(num_radial, bool) or not isinstance(num_radial, (int, np.integer)):
        raise ValueError("num_radial must be an integer")
    if int(num_radial) < 1:
        raise ValueError("num_radial must be an integer >= 1")

    local = table[:, 0]
    circ_weight = table[:, 1]

    shape = np.column_stack([0.5 * local * (local - 1.0),
                             1.0 - local ** 2,
                             0.5 * local * (local + 1.0)])
    slope = np.column_stack([local - 0.5, -2.0 * local, local + 0.5])

    # Boundary points and tangents measured from the source point.
    spoke = shape @ nodes - point
    tangent = slope @ nodes

    # The area swept per unit radial fraction and per unit local coordinate.
    sweep = spoke[:, 0] * tangent[:, 1] - spoke[:, 1] * tangent[:, 0]
    radius = np.hypot(spoke[:, 0], spoke[:, 1])
    if np.any(radius <= 0.0):
        raise ValueError("source_point coincides with a circumferential quadrature point")

    fraction, radial_weight = np.polynomial.legendre.leggauss(int(num_radial))
    fraction = 0.5 * (1.0 + fraction)
    radial_weight = 0.5 * radial_weight

    coords_x = fraction[:, None] * spoke[None, :, 0] + point[0]
    coords_y = fraction[:, None] * spoke[None, :, 1] + point[1]

    # The radial factor of the area element cancels the logarithm's singularity
    # at the source point, leaving a bounded integrand.
    log_distance = np.log(fraction[:, None] * radius[None, :])
    weight = (-0.5 / np.pi) * log_distance * fraction[:, None] * sweep[None, :] \
        * radial_weight[:, None] * circ_weight[None, :]

    return np.column_stack([coords_x.ravel(), coords_y.ravel(), weight.ravel()])

import numpy as np
def domain_integrand_values(operator: "np.ndarray", nodal_values: "np.ndarray",
                                    initial_values: "np.ndarray", source_values: "np.ndarray",
                                    conductivity: float) -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    matrix = np.asarray(operator, dtype=float)
    values = np.asarray(nodal_values, dtype=float)
    initial = np.asarray(initial_values, dtype=float)
    sources = np.asarray(source_values, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator must be a square array with at least one row")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("operator must be finite")
    order = matrix.shape[0]
    if values.ndim != 2 or values.shape[0] != order or values.shape[1] < 1:
        raise ValueError("nodal_values must have shape (p, n) with n >= 1")
    if sources.shape != values.shape:
        raise ValueError("source_values must have the same shape as nodal_values")
    if initial.shape != (values.shape[1],):
        raise ValueError("initial_values must have shape (n,)")
    for name, block in (("nodal_values", values), ("initial_values", initial),
                        ("source_values", sources)):
        if not np.all(np.isfinite(block)):
            raise ValueError(f"{name} must be finite")
    if isinstance(conductivity, bool) or not isinstance(conductivity, (int, float, np.integer, np.floating)):
        raise ValueError("conductivity must be a real number")
    if not np.isfinite(float(conductivity)) or float(conductivity) <= 0.0:
        raise ValueError("conductivity must be a finite number > 0")

    # The spectral operator sees the increment of the temperature over the step,
    # not the temperature itself.
    rate = matrix @ (values - initial[None, :])

    return np.asarray(rate / float(conductivity) + sources, dtype=float)

import numpy as np
def reconstruct_interior_value(boundary_terms: "np.ndarray", boundary_fields: "np.ndarray",
                                       domain_terms: "np.ndarray", domain_values: "np.ndarray") -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    terms = np.asarray(boundary_terms, dtype=float)
    fields = np.asarray(boundary_fields, dtype=float)
    sectors = np.asarray(domain_terms, dtype=float)
    density = np.asarray(domain_values, dtype=float)

    if terms.ndim != 2 or terms.shape[0] < 1 or terms.shape[1] != 6:
        raise ValueError("boundary_terms must have shape (m, 6) with m >= 1")
    if fields.shape != (terms.shape[0], 3):
        raise ValueError("boundary_fields must have shape (m, 3)")
    if sectors.ndim != 2 or sectors.shape[0] < 1 or sectors.shape[1] != 3:
        raise ValueError("domain_terms must have shape (q, 3) with q >= 1")
    if density.shape != (sectors.shape[0],):
        raise ValueError("domain_values must have shape (q,)")
    for name, block in (("boundary_terms", terms), ("boundary_fields", fields),
                        ("domain_terms", sectors), ("domain_values", density)):
        if not np.all(np.isfinite(block)):
            raise ValueError(f"{name} must be finite")

    # Normal derivative of the temperature at each boundary quadrature point.
    flux = fields[:, 1] * terms[:, 4] + fields[:, 2] * terms[:, 5]

    boundary = float(np.sum(terms[:, 2] * flux - terms[:, 3] * fields[:, 0]))
    domain = float(np.sum(sectors[:, 2] * density))

    return float(boundary - domain)

import numpy as np
def run_coating_benchmark(length: float, thickness: float, substrate_depth: float,
                                  conductivities: tuple, coating_mesh: tuple, substrate_mesh: tuple,
                                  num_nodes: int, time_step: float, num_gauss: int,
                                  num_radial: int) -> float:
    import numpy as np

    # -- Validate the orchestrator inputs.
    for name, value in (("length", length), ("thickness", thickness),
                        ("substrate_depth", substrate_depth), ("time_step", time_step)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite number > 0")
    if not isinstance(conductivities, (tuple, list)) or len(conductivities) != 2:
        raise ValueError("conductivities must be a sequence of exactly two numbers")
    for value in conductivities:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("conductivities must contain real numbers")
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError("conductivities must be finite numbers > 0")
    for name, spec in (("coating_mesh", coating_mesh), ("substrate_mesh", substrate_mesh)):
        if not isinstance(spec, (tuple, list)) or len(spec) != 4:
            raise ValueError(f"{name} must be a sequence of exactly four integers")
        for value in spec:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must contain integers only")
            if int(value) < 1:
                raise ValueError(f"{name} entries must be >= 1")
    for name, value in (("num_nodes", num_nodes), ("num_gauss", num_gauss),
                        ("num_radial", num_radial)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if int(num_nodes) < 2:
        raise ValueError("num_nodes must be an integer >= 2")
    if int(num_gauss) < 1 or int(num_radial) < 1:
        raise ValueError("num_gauss and num_radial must be integers >= 1")

    k_coating = float(conductivities[0])
    k_substrate = float(conductivities[1])

    # -- Sub-problem 01 reference: the Gauss node times of the step and the operator that
    # turns nodal temperatures into nodal time derivatives.
    times, operator = time_spectral_operator(int(num_nodes), float(time_step))
    times = np.asarray(times, dtype=float)

    # Interface continuity of temperature and of normal flux fixes the tilt of
    # the benchmark field in each layer.
    layers = (
        (0.0, float(thickness), tuple(coating_mesh), k_substrate / k_coating, k_coating),
        (-float(substrate_depth), 0.0, tuple(substrate_mesh), 1.0, k_substrate),
    )

    errors = {}
    for use_sinh in (True, False):
        computed = []
        reference = []
        for y_low, y_high, spec, tilt, conductivity in layers:
            # -- Sub-problem 02 reference: the outline and the interior sample points.
            elements, interior = rectangular_layer_mesh(
                float(length), y_low, y_high,
                int(spec[0]), int(spec[1]), int(spec[2]), int(spec[3]))
            elements = np.asarray(elements, dtype=float)
            interior = np.asarray(interior, dtype=float)

            for index in range(interior.shape[0]):
                source = interior[index]
                boundary_blocks = []
                domain_blocks = []
                for element in range(elements.shape[0]):
                    nodes = elements[element]
                    # -- Sub-problem 04 reference: the element quadrature rule.
                    rule = near_singular_rule(source, nodes, int(num_gauss), bool(use_sinh))
                    # -- Sub-problem 05 reference: the boundary quadrature of that element.
                    boundary_blocks.append(element_boundary_terms(source, nodes, rule))
                    # -- Sub-problem 06 reference: the domain quadrature of its sector.
                    domain_blocks.append(sct_domain_terms(source, nodes, rule, int(num_radial)))
                boundary = np.vstack([np.asarray(block, dtype=float) for block in boundary_blocks])
                domain = np.vstack([np.asarray(block, dtype=float) for block in domain_blocks])

                # -- Sub-problem 03 reference: the benchmark data the two quadratures need.
                domain_points = domain[:, :2]
                nodal = np.vstack([
                    np.asarray(layer_reference_fields(
                        domain_points, float(instant), tilt, conductivity),
                        dtype=float)[:, 0] for instant in times])
                sources = np.vstack([
                    np.asarray(layer_reference_fields(
                        domain_points, float(instant), tilt, conductivity),
                        dtype=float)[:, 3] for instant in times])
                initial = np.asarray(layer_reference_fields(
                    domain_points, 0.0, tilt, conductivity), dtype=float)[:, 0]

                # -- Sub-problem 07 reference: the density of the domain integral.
                density = np.asarray(domain_integrand_values(
                    operator, nodal, initial, sources, conductivity), dtype=float)

                for step in range(times.size):
                    fields = np.asarray(layer_reference_fields(
                        boundary[:, :2], float(times[step]), tilt, conductivity),
                        dtype=float)
                    # -- Sub-problem 08 reference: close the representation formula.
                    computed.append(reconstruct_interior_value(
                        boundary, fields[:, :3], domain, density[step]))
                    exact = np.asarray(layer_reference_fields(
                        source[None, :], float(times[step]), tilt, conductivity),
                        dtype=float)
                    reference.append(float(exact[0, 0]))

        computed = np.asarray(computed, dtype=float)
        reference = np.asarray(reference, dtype=float)
        errors[use_sinh] = float(np.linalg.norm(computed - reference)
                                 / np.linalg.norm(reference))

    if errors[True] <= 0.0:
        raise ValueError("the regularised reconstruction is exact, so no gain is defined")
    return float(errors[False] / errors[True])
SCICODE_GOLD_EOF
