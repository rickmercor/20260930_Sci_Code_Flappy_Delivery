#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def build_ddfv_mesh(nx: int, ny: int, distortion: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if not (isinstance(nx, (int, np.integer)) and not isinstance(nx, bool)
            and int(nx) >= 2):
        raise ValueError("nx must be an integer >= 2")
    if not (isinstance(ny, (int, np.integer)) and not isinstance(ny, bool)
            and int(ny) >= 2):
        raise ValueError("ny must be an integer >= 2")
    if not (isinstance(distortion, (int, float)) and np.isfinite(distortion)
            and 0.0 <= float(distortion) < 0.5):
        raise ValueError("distortion must be a finite number in [0, 0.5)")

    nx, ny, distortion = int(nx), int(ny), float(distortion)

    def _primal_mesh():
        """Vertex coordinates and triangle vertex table of the distorted mesh.

        The unit square is cut into nx by ny equal rectangles, each rectangle
        is split along its lower-left to upper-right diagonal, and every
        strictly interior vertex is displaced by a checkerboard-signed
        diagonal offset.
        """
        dx, dy = 1.0 / nx, 1.0 / ny
        coords = np.zeros(((nx + 1) * (ny + 1), 2), dtype=float)
        for j in range(ny + 1):
            for i in range(nx + 1):
                x, y = i * dx, j * dy
                if 0 < i < nx and 0 < j < ny:
                    sign = 1.0 if (i + j) % 2 == 0 else -1.0
                    x += distortion * dx * sign
                    y += distortion * dy * sign
                coords[j * (nx + 1) + i] = (x, y)

        cells = []
        for j in range(ny):
            for i in range(nx):
                v00 = j * (nx + 1) + i
                v01 = (j + 1) * (nx + 1) + i
                cells.append((v00, v00 + 1, v01 + 1))
                cells.append((v00, v01 + 1, v01))

        return coords, np.asarray(cells, dtype=int)

    def _edge_table(cells):
        """Map every unordered vertex pair to the triangles carrying it."""
        table = {}
        for t, (a, b, c) in enumerate(cells):
            for u, v in ((a, b), (b, c), (c, a)):
                table.setdefault((min(u, v), max(u, v)), []).append(t)
        return table

    xy, tris = _primal_mesh()
    edges = _edge_table(tris)

    interior = sorted(e for e, ts in edges.items() if len(ts) == 2)
    boundary = sorted(e for e, ts in edges.items() if len(ts) == 1)
    n_tri, n_bnd = len(tris), len(boundary)
    bnd_index = {e: k for k, e in enumerate(boundary)}
    bary = xy[tris].mean(axis=1)

    rows = []
    for edge in interior + boundary:
        ks, ls = edge
        touching = edges[edge]
        cell_k = touching[0]
        x_k = bary[cell_k]

        if len(touching) == 2:
            cell_l = touching[1]
            x_l = bary[touching[1]]
            on_boundary = 0.0
        else:
            cell_l = n_tri + bnd_index[edge]
            x_l = 0.5 * (xy[ks] + xy[ls])
            on_boundary = 1.0

        diag_star = x_l - x_k
        diag = xy[ls] - xy[ks]
        cross = diag_star[0] * diag[1] - diag_star[1] * diag[0]

        if cross < 0.0:
            ks, ls = ls, ks
            diag, cross = -diag, -cross
        if cross == 0.0:
            raise ValueError("degenerate diamond cell with zero area")

        area = 0.5 * abs(cross)
        len_sigma = float(np.hypot(diag[0], diag[1]))
        len_sigma_star = float(np.hypot(diag_star[0], diag_star[1]))

        normal_kl = np.array([-diag[1], diag[0]]) / len_sigma
        if normal_kl @ diag_star < 0.0:
            normal_kl = -normal_kl

        normal_ks = np.array(
            [-diag_star[1], diag_star[0]]
        ) / len_sigma_star
        if normal_ks @ diag < 0.0:
            normal_ks = -normal_ks

        rows.append([
            float(cell_k),
            float(cell_l),
            float(n_tri + n_bnd + ks),
            float(n_tri + n_bnd + ls),
            area,
            len_sigma,
            len_sigma_star,
            float(normal_kl @ normal_ks),
            on_boundary,
        ])

    return np.asarray(rows, dtype=float)

def build_ddfv_node_table(
    nx: int,
    ny: int,
    distortion: float,
) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if not (
        isinstance(nx, (int, np.integer))
        and not isinstance(nx, bool)
        and int(nx) >= 2
    ):
        raise ValueError("nx must be an integer >= 2")

    if not (
        isinstance(ny, (int, np.integer))
        and not isinstance(ny, bool)
        and int(ny) >= 2
    ):
        raise ValueError("ny must be an integer >= 2")

    if not (
        isinstance(distortion, (int, float))
        and np.isfinite(distortion)
        and 0.0 <= float(distortion) < 0.5
    ):
        raise ValueError(
            "distortion must be a finite number in [0, 0.5)"
        )

    nx = int(nx)
    ny = int(ny)
    distortion = float(distortion)

    def _primal_mesh():
        """Vertex coordinates and triangle table of the distorted mesh."""
        dx = 1.0 / nx
        dy = 1.0 / ny
        coords = np.zeros(
            ((nx + 1) * (ny + 1), 2),
            dtype=float,
        )

        for j in range(ny + 1):
            for i in range(nx + 1):
                x = i * dx
                y = j * dy

                if 0 < i < nx and 0 < j < ny:
                    sign = 1.0 if (i + j) % 2 == 0 else -1.0
                    x += distortion * dx * sign
                    y += distortion * dy * sign

                coords[j * (nx + 1) + i] = (x, y)

        cells = []
        for j in range(ny):
            for i in range(nx):
                v00 = j * (nx + 1) + i
                v01 = (j + 1) * (nx + 1) + i
                cells.append((v00, v00 + 1, v01 + 1))
                cells.append((v00, v01 + 1, v01))

        return coords, np.asarray(cells, dtype=int)

    def _edge_table(cells):
        """Map every vertex pair to the triangles carrying it."""
        table = {}

        for t, (a, b, c) in enumerate(cells):
            for u, v in ((a, b), (b, c), (c, a)):
                key = (min(u, v), max(u, v))
                table.setdefault(key, []).append(t)

        return table

    def _triangle_area(p, q, r):
        """Return the unsigned area of a triangle."""
        return 0.5 * abs(
            (q[0] - p[0]) * (r[1] - p[1])
            - (q[1] - p[1]) * (r[0] - p[0])
        )

    xy, tris = _primal_mesh()
    edges = _edge_table(tris)

    interior = sorted(
        edge for edge, touching in edges.items()
        if len(touching) == 2
    )
    boundary = sorted(
        edge for edge, touching in edges.items()
        if len(touching) == 1
    )

    n_tri = len(tris)
    n_bnd = len(boundary)
    n_vert = len(xy)
    n_nodes = n_tri + n_bnd + n_vert

    bary = xy[tris].mean(axis=1)
    bnd_index = {
        edge: index
        for index, edge in enumerate(boundary)
    }

    nodes = np.zeros((n_nodes, 5), dtype=float)
    nodes[:n_tri, :2] = bary

    for edge, index in bnd_index.items():
        nodes[n_tri + index, :2] = (
            0.5 * (xy[edge[0]] + xy[edge[1]])
        )

    nodes[n_tri + n_bnd:, :2] = xy

    # Primal measures are triangle areas.
    for t, (a, b, c) in enumerate(tris):
        nodes[t, 2] = _triangle_area(
            xy[a],
            xy[b],
            xy[c],
        )

    # Dual measures accumulate one triangle per incident diamond.
    for edge in interior + boundary:
        ks, ls = edge
        touching = edges[edge]
        x_k = bary[touching[0]]

        if len(touching) == 2:
            x_l = bary[touching[1]]
        else:
            x_l = 0.5 * (xy[ks] + xy[ls])

        nodes[n_tri + n_bnd + ks, 2] += _triangle_area(
            x_k,
            xy[ks],
            x_l,
        )
        nodes[n_tri + n_bnd + ls, 2] += _triangle_area(
            x_k,
            x_l,
            xy[ls],
        )

    # Equation classes and contact tags.
    for i in range(n_tri, n_tri + n_bnd):
        x, y = nodes[i, 0], nodes[i, 1]

        if y == 0.0:
            nodes[i, 3] = 1.0
            nodes[i, 4] = 1.0
        elif y == 1.0:
            nodes[i, 3] = 1.0
            nodes[i, 4] = 2.0
        else:
            nodes[i, 3] = 2.0

    for i in range(n_tri + n_bnd, n_nodes):
        x, y = nodes[i, 0], nodes[i, 1]

        if y == 0.0:
            nodes[i, 3] = 4.0
            nodes[i, 4] = 1.0
        elif y == 1.0:
            nodes[i, 3] = 4.0
            nodes[i, 4] = 2.0
        elif x == 0.0 or x == 1.0:
            nodes[i, 3] = 5.0
        else:
            nodes[i, 3] = 3.0

    return nodes

def evaluate_bernoulli(t: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    t = np.asarray(t, dtype=float)

    if not np.all(np.isfinite(t)):
        raise ValueError(
            "t must contain only finite values"
        )

    values = np.empty(
        t.shape,
        dtype=float,
    )

    small = np.abs(t) < 1.0e-8
    large_pos = t > 500.0
    large_neg = t < -500.0
    middle = ~(
        small
        | large_pos
        | large_neg
    )

    ts = t[small]
    values[small] = (
        1.0
        - 0.5 * ts
        + ts * ts / 12.0
    )

    values[large_pos] = (
        t[large_pos]
        * np.exp(-t[large_pos])
    )

    values[large_neg] = -t[large_neg]

    tm = t[middle]
    values[middle] = (
        tm / np.expm1(tm)
    )

    return values

# ORACLE SOLUTION


def build_junction_state(nodes: np.ndarray, anode_voltage: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 5 or nodes.shape[0] < 1:
        raise ValueError("nodes must be a 2D array with shape (n_nodes, 5)")
    if not np.all(np.isfinite(nodes)):
        raise ValueError("nodes must contain only finite values")
    if not (isinstance(anode_voltage, (int, float))
            and np.isfinite(anode_voltage)):
        raise ValueError("anode_voltage must be a finite number")

    # Silicon at 300 K, scaled by the reference doping and the thermal voltage.
    thermal_voltage = 0.025852                       # V
    scaled_intrinsic = 1.087386e10 / 1.0e15          # n_ie / N_0

    height = nodes[:, 1]
    contact = nodes[:, 4]

    # Abrupt junction: donor dominated below the junction line, acceptor
    # dominated above it, and the mean of the two exactly on it.
    doping = np.where(height < 0.5, 1.0, np.where(height > 0.5, -1.0, 0.0))

    # Charge neutrality together with the mass action law at an Ohmic contact.
    root = np.sqrt(doping * doping + 4.0 * scaled_intrinsic * scaled_intrinsic)
    electrons = 0.5 * (doping + root)
    holes = 0.5 * (-doping + root)

    applied = np.where(contact == 2.0, float(anode_voltage) / thermal_voltage, 0.0)
    potential = applied + np.log(electrons / scaled_intrinsic)

    return np.column_stack([doping, electrons, holes, potential])

def assemble_ddfv_laplacian(
    diamonds: np.ndarray,
    nodes: np.ndarray,
) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    diamonds = np.asarray(diamonds, dtype=float)
    nodes = np.asarray(nodes, dtype=float)

    if (
        diamonds.ndim != 2
        or diamonds.shape[1] != 9
        or diamonds.shape[0] < 1
    ):
        raise ValueError(
            "diamonds must be a 2D array with shape (n_diamonds, 9)"
        )

    if (
        nodes.ndim != 2
        or nodes.shape[1] != 5
        or nodes.shape[0] < 1
    ):
        raise ValueError(
            "nodes must be a 2D array with shape (n_nodes, 5)"
        )

    if np.any(diamonds[:, 4] <= 0.0):
        raise ValueError(
            "every diamond area must be strictly positive"
        )

    if (
        diamonds[:, :4].max() >= nodes.shape[0]
        or diamonds[:, :4].min() < 0.0
    ):
        raise ValueError(
            "diamond node indices must lie inside the node table"
        )

    def _scatter(primal_block, dual_block, boundary_term):
        """Scatter per-diamond blocks into the global matrix."""
        n_nodes = nodes.shape[0]
        matrix = np.zeros(
            (n_nodes, n_nodes),
            dtype=float,
        )

        code = nodes[:, 3]
        measure = nodes[:, 2]
        inverse = np.where(
            measure > 0.0,
            1.0 / np.where(measure > 0.0, measure, 1.0),
            0.0,
        )

        cell_k = diamonds[:, 0].astype(int)
        cell_l = diamonds[:, 1].astype(int)
        dual_k = diamonds[:, 2].astype(int)
        dual_l = diamonds[:, 3].astype(int)

        columns = np.column_stack([
            cell_k,
            cell_l,
            dual_k,
            dual_l,
        ])

        def _push(rows, mask, block, scale):
            if not np.any(mask):
                return

            target = np.repeat(rows[mask], 4)
            source = columns[mask].ravel()
            values = (
                block[mask] * scale[mask][:, None]
            ).ravel()

            np.add.at(
                matrix,
                (target, source),
                values,
            )

        unit = np.ones(
            diamonds.shape[0],
            dtype=float,
        )

        _push(
            cell_k,
            code[cell_k] == 0.0,
            primal_block,
            inverse[cell_k],
        )
        _push(
            cell_l,
            code[cell_l] == 0.0,
            -primal_block,
            inverse[cell_l],
        )
        _push(
            dual_k,
            np.isin(code[dual_k], (3.0, 5.0)),
            dual_block,
            inverse[dual_k],
        )
        _push(
            dual_l,
            np.isin(code[dual_l], (3.0, 5.0)),
            -dual_block,
            inverse[dual_l],
        )
        _push(
            cell_l,
            code[cell_l] == 2.0,
            primal_block,
            unit,
        )

        if boundary_term:
            on_boundary = diamonds[:, 8] > 0.5

            _push(
                dual_k,
                on_boundary
                & np.isin(code[dual_k], (3.0, 5.0)),
                0.5 * primal_block,
                inverse[dual_k],
            )
            _push(
                dual_l,
                on_boundary
                & np.isin(code[dual_l], (3.0, 5.0)),
                0.5 * primal_block,
                inverse[dual_l],
            )

        return matrix

    area = diamonds[:, 4]
    len_sigma = diamonds[:, 5]
    len_sigma_star = diamonds[:, 6]
    normal_dot = diamonds[:, 7]

    primal = (
        len_sigma * len_sigma
        / (2.0 * area)
    )
    cross = (
        len_sigma
        * len_sigma_star
        * normal_dot
        / (2.0 * area)
    )
    dual = (
        len_sigma_star * len_sigma_star
        / (2.0 * area)
    )

    primal_block = np.column_stack([
        -primal,
        primal,
        -cross,
        cross,
    ])
    dual_block = np.column_stack([
        -cross,
        cross,
        -dual,
        dual,
    ])

    return _scatter(
        primal_block,
        dual_block,
        True,
    )

def assemble_harmonic_flux_matrix(
    diamonds: np.ndarray,
    nodes: np.ndarray,
    potential: np.ndarray,
    diffusivity: float,
    is_hole: bool,
) -> np.ndarray:
    import numpy as np

    diamonds = np.asarray(diamonds, dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    potential = np.asarray(potential, dtype=float)

    if (
        diamonds.ndim != 2
        or diamonds.shape[1] != 9
        or diamonds.shape[0] < 1
    ):
        raise ValueError(
            "diamonds must be a 2D array with shape (n_diamonds, 9)"
        )

    if (
        nodes.ndim != 2
        or nodes.shape[1] != 5
        or nodes.shape[0] < 1
    ):
        raise ValueError(
            "nodes must be a 2D array with shape (n_nodes, 5)"
        )

    if potential.shape != (nodes.shape[0],):
        raise ValueError(
            "potential must have shape (n_nodes,)"
        )

    if not np.all(np.isfinite(potential)):
        raise ValueError(
            "potential must contain only finite values"
        )

    if not (
        isinstance(diffusivity, (int, float))
        and np.isfinite(diffusivity)
        and float(diffusivity) > 0.0
    ):
        raise ValueError(
            "diffusivity must be a finite number > 0"
        )

    if not isinstance(is_hole, (bool, np.bool_)):
        raise ValueError(
            "is_hole must be a boolean"
        )

    if np.any(diamonds[:, 4] <= 0.0):
        raise ValueError(
            "every diamond area must be strictly positive"
        )

    if (
        diamonds[:, :4].max() >= nodes.shape[0]
        or diamonds[:, :4].min() < 0.0
    ):
        raise ValueError(
            "diamond node indices must lie inside the node table"
        )

    bernoulli = evaluate_bernoulli

    def _scatter(primal_block, dual_block, boundary_term):
        """Scatter per-diamond blocks into the global matrix."""
        n_nodes = nodes.shape[0]
        matrix = np.zeros(
            (n_nodes, n_nodes),
            dtype=float,
        )

        code = nodes[:, 3]
        measure = nodes[:, 2]
        inverse = np.where(
            measure > 0.0,
            1.0 / np.where(measure > 0.0, measure, 1.0),
            0.0,
        )

        k = diamonds[:, 0].astype(int)
        l = diamonds[:, 1].astype(int)
        ks = diamonds[:, 2].astype(int)
        ls = diamonds[:, 3].astype(int)

        columns = np.column_stack([
            k,
            l,
            ks,
            ls,
        ])

        def _push(rows, mask, block, scale):
            if not np.any(mask):
                return

            target = np.repeat(rows[mask], 4)
            source = columns[mask].ravel()
            values = (
                block[mask] * scale[mask][:, None]
            ).ravel()

            np.add.at(
                matrix,
                (target, source),
                values,
            )

        unit = np.ones(
            diamonds.shape[0],
            dtype=float,
        )

        _push(
            k,
            code[k] == 0.0,
            primal_block,
            inverse[k],
        )
        _push(
            l,
            code[l] == 0.0,
            -primal_block,
            inverse[l],
        )
        _push(
            ks,
            np.isin(code[ks], (3.0, 5.0)),
            dual_block,
            inverse[ks],
        )
        _push(
            ls,
            np.isin(code[ls], (3.0, 5.0)),
            -dual_block,
            inverse[ls],
        )
        _push(
            l,
            code[l] == 2.0,
            primal_block,
            unit,
        )

        if boundary_term:
            on_boundary = diamonds[:, 8] > 0.5

            _push(
                ks,
                on_boundary
                & np.isin(code[ks], (3.0, 5.0)),
                0.5 * primal_block,
                inverse[ks],
            )
            _push(
                ls,
                on_boundary
                & np.isin(code[ls], (3.0, 5.0)),
                0.5 * primal_block,
                inverse[ls],
            )

        return matrix

    diffusivity = float(diffusivity)
    area = diamonds[:, 4]
    len_sigma = diamonds[:, 5]
    len_sigma_star = diamonds[:, 6]
    normal_dot = diamonds[:, 7]

    coeff_primal = (
        diffusivity
        * len_sigma
        * len_sigma
        / (2.0 * area)
    )
    coeff_cross = (
        diffusivity
        * len_sigma
        * len_sigma_star
        * normal_dot
        / (2.0 * area)
    )
    coeff_dual = (
        diffusivity
        * len_sigma_star
        * len_sigma_star
        / (2.0 * area)
    )

    cell_k = diamonds[:, 0].astype(int)
    cell_l = diamonds[:, 1].astype(int)
    dual_k = diamonds[:, 2].astype(int)
    dual_l = diamonds[:, 3].astype(int)

    sign = -1.0 if bool(is_hole) else 1.0

    drop_primal = sign * (
        potential[cell_k] - potential[cell_l]
    )
    drop_dual = sign * (
        potential[dual_k] - potential[dual_l]
    )

    b_primal = bernoulli(drop_primal)
    b_primal_rev = bernoulli(-drop_primal)
    b_dual = bernoulli(drop_dual)
    b_dual_rev = bernoulli(-drop_dual)

    primal_block = np.column_stack([
        coeff_primal * b_primal,
        -coeff_primal * b_primal_rev,
        coeff_cross * b_dual,
        -coeff_cross * b_dual_rev,
    ])

    dual_block = np.column_stack([
        coeff_cross * b_primal,
        -coeff_cross * b_primal_rev,
        coeff_dual * b_dual,
        -coeff_dual * b_dual_rev,
    ])

    return _scatter(
        primal_block,
        dual_block,
        False,
    )

# ORACLE SOLUTION


def assemble_coupled_residual(nodes: np.ndarray, state: np.ndarray,
                                      laplacian: np.ndarray,
                                      electron_matrix: np.ndarray,
                                      hole_matrix: np.ndarray,
                                      unknowns: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    nodes = np.asarray(nodes, dtype=float)
    state = np.asarray(state, dtype=float)
    laplacian = np.asarray(laplacian, dtype=float)
    electron_matrix = np.asarray(electron_matrix, dtype=float)
    hole_matrix = np.asarray(hole_matrix, dtype=float)
    unknowns = np.asarray(unknowns, dtype=float)

    if nodes.ndim != 2 or nodes.shape[1] != 5 or nodes.shape[0] < 1:
        raise ValueError("nodes must be a 2D array with shape (n_nodes, 5)")
    n_nodes = nodes.shape[0]
    if state.shape != (n_nodes, 4):
        raise ValueError("state must have shape (n_nodes, 4)")
    if unknowns.shape != (n_nodes, 3):
        raise ValueError("unknowns must have shape (n_nodes, 3)")
    for name, matrix in (("laplacian", laplacian),
                         ("electron_matrix", electron_matrix),
                         ("hole_matrix", hole_matrix)):
        if matrix.shape != (n_nodes, n_nodes):
            raise ValueError(f"{name} must have shape (n_nodes, n_nodes)")
    if not np.all(np.isfinite(unknowns)):
        raise ValueError("unknowns must contain only finite values")

    # Silicon at 300 K. Under the stated nondimensionalisation the whole
    # material data set collapses into this one dimensionless group, the
    # square of the Debye length in units of the length scale.
    elementary_charge = 1.602192e-19          # C
    permittivity = 1.035941e-12               # C / (V cm)
    thermal_voltage = 0.025852                # V
    reference_density = 1.0e15                # cm^-3
    length_unit = 1.0e-4                      # cm, that is 1 micrometre
    debye_group = (permittivity * thermal_voltage
                   / (elementary_charge * reference_density * length_unit ** 2))

    code = nodes[:, 3]
    doping, electrons_dir, holes_dir, potential_dir = state.T
    potential, electrons, holes = unknowns.T

    curvature = laplacian @ potential
    poisson = debye_group * curvature + (holes - electrons + doping)
    electron_balance = electron_matrix @ electrons
    hole_balance = hole_matrix @ holes

    # Contact-free boundary edges: the row already holds the raw normal
    # contraction, so only the space charge term is dropped from Poisson.
    neumann = code == 2.0
    poisson[neumann] = curvature[neumann]

    # Ohmic contacts, primal and dual alike: prescribe all three unknowns.
    dirichlet = np.isin(code, (1.0, 4.0))
    poisson[dirichlet] = potential[dirichlet] - potential_dir[dirichlet]
    electron_balance[dirichlet] = electrons[dirichlet] - electrons_dir[dirichlet]
    hole_balance[dirichlet] = holes[dirichlet] - holes_dir[dirichlet]

    return np.concatenate([poisson, electron_balance, hole_balance])

def solve_drift_diffusion(
    diamonds: np.ndarray,
    nodes: np.ndarray,
    anode_voltage: float,
    n_steps: int = 4,
    tolerance: float = 1.0e-10,
    max_iterations: int = 50,
) -> np.ndarray:
    import numpy as np

    diamonds = np.asarray(diamonds, dtype=float)
    nodes = np.asarray(nodes, dtype=float)

    if (
        diamonds.ndim != 2
        or diamonds.shape[1] != 9
        or diamonds.shape[0] < 1
    ):
        raise ValueError(
            "diamonds must be a 2D array with shape (n_diamonds, 9)"
        )

    if (
        nodes.ndim != 2
        or nodes.shape[1] != 5
        or nodes.shape[0] < 1
    ):
        raise ValueError(
            "nodes must be a 2D array with shape (n_nodes, 5)"
        )

    if not (
        isinstance(anode_voltage, (int, float))
        and np.isfinite(anode_voltage)
    ):
        raise ValueError(
            "anode_voltage must be a finite number"
        )

    if not (
        isinstance(n_steps, (int, np.integer))
        and not isinstance(n_steps, bool)
        and int(n_steps) >= 1
    ):
        raise ValueError(
            "n_steps must be an integer >= 1"
        )

    if not (
        isinstance(tolerance, (int, float))
        and np.isfinite(tolerance)
        and float(tolerance) > 0.0
    ):
        raise ValueError(
            "tolerance must be a finite number > 0"
        )

    if not (
        isinstance(max_iterations, (int, np.integer))
        and not isinstance(max_iterations, bool)
        and int(max_iterations) >= 1
    ):
        raise ValueError(
            "max_iterations must be an integer >= 1"
        )

    if np.any(diamonds[:, 4] <= 0.0):
        raise ValueError(
            "every diamond area must be strictly positive"
        )

    anode_voltage = float(anode_voltage)
    n_steps = int(n_steps)
    max_iterations = int(max_iterations)
    tolerance = float(tolerance)

    junction_state = build_junction_state
    build_laplacian = assemble_ddfv_laplacian
    flux_matrix = assemble_harmonic_flux_matrix
    coupled_residual = assemble_coupled_residual

    elementary_charge = 1.602192e-19
    permittivity = 1.035941e-12
    thermal_voltage = 0.025852
    reference_density = 1.0e15
    length_unit = 1.0e-4
    scaled_hole_diffusivity = 12.16336 / 36.63227

    debye_group = (
        permittivity
        * thermal_voltage
        / (
            elementary_charge
            * reference_density
            * length_unit ** 2
        )
    )

    density_floor = 1.0e-20

    n_nodes = nodes.shape[0]
    code = nodes[:, 3]
    measure = nodes[:, 2]
    dirichlet = np.isin(code, (1.0, 4.0))
    neumann = code == 2.0
    identity = np.eye(n_nodes)

    cell_k = diamonds[:, 0].astype(int)
    cell_l = diamonds[:, 1].astype(int)
    dual_k = diamonds[:, 2].astype(int)
    dual_l = diamonds[:, 3].astype(int)

    columns = np.column_stack([
        cell_k,
        cell_l,
        dual_k,
        dual_l,
    ])

    inverse_measure = np.where(
        measure > 0.0,
        1.0 / np.where(
            measure > 0.0,
            measure,
            1.0,
        ),
        0.0,
    )

    unit = np.ones(
        diamonds.shape[0],
        dtype=float,
    )

    def _bernoulli_derivative(t):
        """First derivative of the Bernoulli function."""
        t = np.asarray(t, dtype=float)
        values = np.empty(t.shape, dtype=float)

        small = np.abs(t) < 1.0e-6
        large_pos = t > 500.0
        large_neg = t < -500.0
        middle = ~(
            small
            | large_pos
            | large_neg
        )

        values[small] = (
            -0.5
            + t[small] / 6.0
        )
        values[large_pos] = (
            1.0 - t[large_pos]
        ) * np.exp(-t[large_pos])
        values[large_neg] = -1.0

        tm = t[middle]
        expm = np.expm1(tm)
        values[middle] = (
            expm
            - tm * np.exp(tm)
        ) / (expm * expm)

        return values

    def _scatter(primal_block, dual_block):
        """Scatter per-diamond blocks into the global matrix."""
        matrix = np.zeros(
            (n_nodes, n_nodes),
            dtype=float,
        )

        def _push(rows, mask, block, scale):
            if not np.any(mask):
                return

            target = np.repeat(
                rows[mask],
                4,
            )
            source = columns[mask].ravel()
            values = (
                block[mask]
                * scale[mask][:, None]
            ).ravel()

            np.add.at(
                matrix,
                (target, source),
                values,
            )

        _push(
            cell_k,
            code[cell_k] == 0.0,
            primal_block,
            inverse_measure[cell_k],
        )
        _push(
            cell_l,
            code[cell_l] == 0.0,
            -primal_block,
            inverse_measure[cell_l],
        )
        _push(
            dual_k,
            np.isin(
                code[dual_k],
                (3.0, 5.0),
            ),
            dual_block,
            inverse_measure[dual_k],
        )
        _push(
            dual_l,
            np.isin(
                code[dual_l],
                (3.0, 5.0),
            ),
            -dual_block,
            inverse_measure[dual_l],
        )
        _push(
            cell_l,
            code[cell_l] == 2.0,
            primal_block,
            unit,
        )

        return matrix

    def _geometric_coefficients(diffusivity):
        """Return the primal, cross and dual coefficients."""
        area = diamonds[:, 4]
        len_sigma = diamonds[:, 5]
        len_sigma_star = diamonds[:, 6]
        normal_dot = diamonds[:, 7]

        return (
            diffusivity
            * len_sigma
            * len_sigma
            / (2.0 * area),
            diffusivity
            * len_sigma
            * len_sigma_star
            * normal_dot
            / (2.0 * area),
            diffusivity
            * len_sigma_star
            * len_sigma_star
            / (2.0 * area),
        )

    def _flux_potential_jacobian(
        potential,
        density,
        diffusivity,
        is_hole,
    ):
        """Derivative of the flux balance with respect to potential."""
        primal, cross, dual = _geometric_coefficients(
            diffusivity
        )

        sign = (
            -1.0
            if is_hole
            else 1.0
        )

        drop_p = sign * (
            potential[cell_k]
            - potential[cell_l]
        )
        drop_d = sign * (
            potential[dual_k]
            - potential[dual_l]
        )

        grad_p = sign * (
            _bernoulli_derivative(drop_p)
            * density[cell_k]
            + _bernoulli_derivative(-drop_p)
            * density[cell_l]
        )
        grad_d = sign * (
            _bernoulli_derivative(drop_d)
            * density[dual_k]
            + _bernoulli_derivative(-drop_d)
            * density[dual_l]
        )

        primal_block = np.column_stack([
            primal * grad_p,
            -primal * grad_p,
            cross * grad_d,
            -cross * grad_d,
        ])

        dual_block = np.column_stack([
            cross * grad_p,
            -cross * grad_p,
            dual * grad_d,
            -dual * grad_d,
        ])

        return _scatter(
            primal_block,
            dual_block,
        )

    laplacian = build_laplacian(
        diamonds,
        nodes,
    )

    initial = junction_state(
        nodes,
        0.0,
    )

    potential = initial[:, 3].copy()
    electrons = initial[:, 1].copy()
    holes = initial[:, 2].copy()

    for step in range(1, n_steps + 1):
        state = junction_state(
            nodes,
            anode_voltage * step / n_steps,
        )

        potential[dirichlet] = state[
            dirichlet,
            3,
        ]
        electrons[dirichlet] = state[
            dirichlet,
            1,
        ]
        holes[dirichlet] = state[
            dirichlet,
            2,
        ]

        converged = False

        for _ in range(max_iterations):
            electron_matrix = flux_matrix(
                diamonds,
                nodes,
                potential,
                1.0,
                False,
            )
            hole_matrix = flux_matrix(
                diamonds,
                nodes,
                potential,
                scaled_hole_diffusivity,
                True,
            )

            current = coupled_residual(
                nodes,
                state,
                laplacian,
                electron_matrix,
                hole_matrix,
                np.column_stack([
                    potential,
                    electrons,
                    holes,
                ]),
            )

            if np.max(
                np.abs(current)
            ) < tolerance:
                converged = True
                break

            jacobian = np.zeros(
                (
                    3 * n_nodes,
                    3 * n_nodes,
                ),
                dtype=float,
            )

            jacobian[
                :n_nodes,
                :n_nodes,
            ] = (
                debye_group
                * laplacian
            )

            jacobian[
                :n_nodes,
                n_nodes:2 * n_nodes,
            ] = -identity

            jacobian[
                :n_nodes,
                2 * n_nodes:,
            ] = identity

            jacobian[
                :n_nodes,
                :n_nodes,
            ][neumann] = laplacian[neumann]

            jacobian[
                :n_nodes,
                n_nodes:2 * n_nodes,
            ][neumann] = 0.0

            jacobian[
                :n_nodes,
                2 * n_nodes:,
            ][neumann] = 0.0

            jacobian[
                n_nodes:2 * n_nodes,
                :n_nodes,
            ] = _flux_potential_jacobian(
                potential,
                electrons,
                1.0,
                False,
            )

            jacobian[
                n_nodes:2 * n_nodes,
                n_nodes:2 * n_nodes,
            ] = electron_matrix

            jacobian[
                2 * n_nodes:,
                :n_nodes,
            ] = _flux_potential_jacobian(
                potential,
                holes,
                scaled_hole_diffusivity,
                True,
            )

            jacobian[
                2 * n_nodes:,
                2 * n_nodes:,
            ] = hole_matrix

            rows = np.where(
                dirichlet
            )[0]

            for offset in (
                0,
                n_nodes,
                2 * n_nodes,
            ):
                jacobian[
                    rows + offset,
                    :,
                ] = 0.0

                jacobian[
                    rows + offset,
                    rows + offset,
                ] = 1.0

            try:
                update = np.linalg.solve(
                    jacobian,
                    -current,
                )
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "singular Newton Jacobian"
                ) from exc

            if not np.all(
                np.isfinite(update)
            ):
                raise ValueError(
                    "Newton update is not finite"
                )

            potential = (
                potential
                + update[:n_nodes]
            )

            electrons = np.maximum(
                electrons
                + update[
                    n_nodes:2 * n_nodes
                ],
                density_floor,
            )

            holes = np.maximum(
                holes
                + update[
                    2 * n_nodes:
                ],
                density_floor,
            )

        if not converged:
            raise ValueError(
                "Newton iteration failed to reach the tolerance"
            )

    return np.column_stack([
        potential,
        electrons,
        holes,
    ])

def compute_terminal_current(
    diamonds: np.ndarray,
    nodes: np.ndarray,
    solution: np.ndarray,
    contact_tag: int,
) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    diamonds = np.asarray(diamonds, dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    solution = np.asarray(solution, dtype=float)

    if (
        diamonds.ndim != 2
        or diamonds.shape[1] != 9
        or diamonds.shape[0] < 1
    ):
        raise ValueError(
            "diamonds must be a 2D array with shape (n_diamonds, 9)"
        )

    if (
        nodes.ndim != 2
        or nodes.shape[1] != 5
        or nodes.shape[0] < 1
    ):
        raise ValueError(
            "nodes must be a 2D array with shape (n_nodes, 5)"
        )

    if solution.shape != (nodes.shape[0], 3):
        raise ValueError(
            "solution must have shape (n_nodes, 3)"
        )

    if not np.all(np.isfinite(solution)):
        raise ValueError(
            "solution must contain only finite values"
        )

    if not (
        isinstance(contact_tag, (int, np.integer))
        and not isinstance(contact_tag, bool)
        and int(contact_tag) in (1, 2)
    ):
        raise ValueError(
            "contact_tag must be the integer 1 or 2"
        )

    if np.any(diamonds[:, 4] <= 0.0):
        raise ValueError(
            "every diamond area must be strictly positive"
        )

    def _bernoulli(t):
        """Bernoulli function B(t) = t / (exp(t) - 1)."""
        t = np.asarray(t, dtype=float)
        values = np.empty(t.shape, dtype=float)

        small = np.abs(t) < 1.0e-8
        large_pos = t > 500.0
        large_neg = t < -500.0
        middle = ~(
            small
            | large_pos
            | large_neg
        )

        ts = t[small]
        values[small] = (
            1.0
            - 0.5 * ts
            + ts * ts / 12.0
        )
        values[large_pos] = (
            t[large_pos]
            * np.exp(-t[large_pos])
        )
        values[large_neg] = -t[large_neg]

        tm = t[middle]
        values[middle] = (
            tm / np.expm1(tm)
        )

        return values

    scaled_hole_diffusivity = (
        12.16336 / 36.63227
    )

    contact_tag = int(contact_tag)
    potential, electrons, holes = solution.T

    cell_k = diamonds[:, 0].astype(int)
    cell_l = diamonds[:, 1].astype(int)
    dual_k = diamonds[:, 2].astype(int)
    dual_l = diamonds[:, 3].astype(int)

    selected = (
        (diamonds[:, 8] > 0.5)
        & (
            nodes[cell_l, 4]
            == float(contact_tag)
        )
    )

    if not np.any(selected):
        raise ValueError(
            "no boundary diamond belongs to the requested contact"
        )

    chosen = diamonds[selected]
    cell_k = cell_k[selected]
    cell_l = cell_l[selected]
    dual_k = dual_k[selected]
    dual_l = dual_l[selected]

    area = chosen[:, 4]
    len_sigma = chosen[:, 5]
    len_sigma_star = chosen[:, 6]
    normal_dot = chosen[:, 7]

    current = 0.0

    for diffusivity, is_hole, charge_sign in (
        (1.0, False, -1.0),
        (
            scaled_hole_diffusivity,
            True,
            1.0,
        ),
    ):
        coeff_primal = (
            diffusivity
            * len_sigma
            * len_sigma
            / (2.0 * area)
        )
        coeff_cross = (
            diffusivity
            * len_sigma
            * len_sigma_star
            * normal_dot
            / (2.0 * area)
        )

        density = (
            holes
            if is_hole
            else electrons
        )
        sign = (
            -1.0
            if is_hole
            else 1.0
        )

        drop_primal = sign * (
            potential[cell_k]
            - potential[cell_l]
        )
        drop_dual = sign * (
            potential[dual_k]
            - potential[dual_l]
        )

        flux = (
            coeff_primal
            * (
                _bernoulli(drop_primal)
                * density[cell_k]
                - _bernoulli(-drop_primal)
                * density[cell_l]
            )
            + coeff_cross
            * (
                _bernoulli(drop_dual)
                * density[dual_k]
                - _bernoulli(-drop_dual)
                * density[dual_l]
            )
        )

        current += (
            charge_sign
            * float(flux.sum())
        )

    return float(current)

def run_ddfv_ha_pipeline(
    nx: int = 6,
    ny: int = 12,
    distortion: float = 0.40,
    anode_voltage: float = 0.4,
    n_steps: int = 4,
    tolerance: float = 1.0e-10,
    max_iterations: int = 50,
    contact_tag: int = 1,
    drop_cross_term: bool = False,
) -> float:
    import numpy as np

    for name, value, floor in (
        ("nx", nx, 2),
        ("ny", ny, 2),
        ("n_steps", n_steps, 1),
        ("max_iterations", max_iterations, 1),
    ):
        if not (
            isinstance(value, (int, np.integer))
            and not isinstance(value, bool)
            and int(value) >= floor
        ):
            raise ValueError(
                f"{name} must be an integer >= {floor}"
            )

    if not (
        isinstance(distortion, (int, float))
        and np.isfinite(distortion)
        and 0.0 <= float(distortion) < 0.5
    ):
        raise ValueError(
            "distortion must be a finite number in [0, 0.5)"
        )

    if not (
        isinstance(anode_voltage, (int, float))
        and np.isfinite(anode_voltage)
    ):
        raise ValueError(
            "anode_voltage must be a finite number"
        )

    if not (
        isinstance(tolerance, (int, float))
        and np.isfinite(tolerance)
        and float(tolerance) > 0.0
    ):
        raise ValueError(
            "tolerance must be a finite number > 0"
        )

    if not (
        isinstance(contact_tag, (int, np.integer))
        and not isinstance(contact_tag, bool)
        and int(contact_tag) in (1, 2)
    ):
        raise ValueError(
            "contact_tag must be the integer 1 or 2"
        )

    if not isinstance(
        drop_cross_term,
        (bool, np.bool_),
    ):
        raise ValueError(
            "drop_cross_term must be a boolean"
        )

    # Earlier steps are available in Studio's concatenated namespace.
    build_mesh = build_ddfv_mesh
    build_nodes = build_ddfv_node_table
    _evaluate_bernoulli = evaluate_bernoulli
    junction_state = build_junction_state
    assemble_laplacian = assemble_ddfv_laplacian
    assemble_flux = assemble_harmonic_flux_matrix
    assemble_residual = assemble_coupled_residual
    solve_system = solve_drift_diffusion
    terminal_current = compute_terminal_current

    # Steps 01 and 02: construct the distorted mesh.
    diamonds = build_mesh(
        int(nx),
        int(ny),
        float(distortion),
    )

    nodes = build_nodes(
        int(nx),
        int(ny),
        float(distortion),
    )

    if bool(drop_cross_term):
        diamonds = np.array(
            diamonds,
            dtype=float,
            copy=True,
        )
        diamonds[:, 7] = 0.0

    # Steps 03 to 07: construct and verify one complete residual.
    bernoulli_probe = _evaluate_bernoulli(
        np.array([
            0.0,
            1.0,
            -1.0,
        ])
    )

    state = junction_state(
        nodes,
        float(anode_voltage),
    )

    potential = state[:, 3]

    laplacian = assemble_laplacian(
        diamonds,
        nodes,
    )

    electron_matrix = assemble_flux(
        diamonds,
        nodes,
        potential,
        1.0,
        False,
    )

    hole_matrix = assemble_flux(
        diamonds,
        nodes,
        potential,
        12.16336 / 36.63227,
        True,
    )

    initial_unknowns = np.column_stack([
        potential,
        state[:, 1],
        state[:, 2],
    ])

    residual = assemble_residual(
        nodes,
        state,
        laplacian,
        electron_matrix,
        hole_matrix,
        initial_unknowns,
    )

    dependency_outputs = (
        bernoulli_probe,
        state,
        laplacian,
        electron_matrix,
        hole_matrix,
        residual,
    )

    if not all(
        np.all(np.isfinite(value))
        for value in dependency_outputs
    ):
        raise ValueError(
            "an earlier pipeline step produced a non-finite output"
        )

    solution = solve_system(
        diamonds,
        nodes,
        float(anode_voltage),
        int(n_steps),
        float(tolerance),
        int(max_iterations),
    )

    return float(
        terminal_current(
            diamonds,
            nodes,
            solution,
            int(contact_tag),
        )
    )
SCICODE_GOLD_EOF
