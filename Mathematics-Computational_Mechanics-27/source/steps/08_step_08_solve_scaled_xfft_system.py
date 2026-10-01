"""
This step solves the equilibrium system for one prescribed macroscopic strain and returns the converged fluctuation together with the diagnostics that certify the solve: the iteration count, the history of the preconditioned residual, the threshold in force at the end, the volume-averaged Mandel stress and a convergence flag.

The stopping threshold is not a constant of the run. It is recomputed from the current volume-averaged stress at every iteration, so the test measures the residual against the quantity the calculation exists to produce rather than against the initial residual, and a configuration whose macroscopic stress is small is not thereby granted a looser solve.

The averaged stress is assembled from the integrated stiffness acting on the macroscopic strain plus the element stress maps acting on the fluctuation, and divided by the cell volume. It is therefore available at every iteration at the cost of one pass over the elements, which is what makes the running threshold affordable.

The discrete equilibrium problem is to find the displacement fluctuation whose strain $\epsilon_h = \overline{\epsilon} + \operatorname{sym} \operatorname{grad}(u_h)$ makes the weak form vanish for every admissible test field. In terms of the assembled element operators this is $\sum_e \Lambda_e^{T} [A_e u_e + S_e^{T} \overline{\epsilon}] = 0$, so the right-hand side carries a minus sign relative to the macroscopic contribution: the fluctuation must undo the out-of-balance traction that the uniform macroscopic strain would otherwise leave at every material interface.

The system is solved by linear conjugate gradients from a zero fluctuation, preconditioned by the block matrix that is the constant-coefficient operator on the standard block and the identity on the enriched block. Both the operator and the preconditioner are applied without ever forming a global matrix: the operator element by element, and the preconditioner by a forward transform, a pointwise multiplication by the Fourier multiplier, and an inverse transform. The monitored quantity is the preconditioned residual norm $\mathrm{res}_k = \sqrt{r_k^{T} \widetilde{P}^{-1} r_k}$, and the iteration stops when it falls at or below $\mathrm{tol} \| \sigma_{avg} \|_2$, where $\sigma_{avg}$ is the macroscopic stress, the volume average and not the volume integral over the cell.

Returns
-------
dict, the converged displacement fluctuation with the iteration count, residuals and mean stress.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_scaled_xfft_system(
    operators: dict,
    green_operator,
    macroscopic_strain,
    cell_volume: float,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Solve the preconditioned enriched equilibrium system by conjugate gradients.

    Parameters
    ----------
    operators : dict
        Assembled element operators.
    green_operator : array_like
        Complex Fourier multiplier of shape (n, n, n, 3, 3).
    macroscopic_strain : array_like
        Prescribed macroscopic strain of shape (6,) in Mandel order.
    cell_volume : float
        Volume of the periodic cell.
    tolerance : float
        Relative tolerance of the stopping test.
    max_iterations : int
        Maximum number of conjugate-gradient iterations.

    Returns
    -------
    dict
        Keys displacement, iterations, residuals, threshold, effective_stress and converged.

    Raises
    ------
    ValueError
        If macroscopic_strain does not have shape (6,), if cell_volume is not strictly positive, or if tolerance is not positive or max_iterations is below one.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _apply_operator(operators, vector):
    """Matrix-free application of the assembled system operator."""
    padded = np.concatenate([vector, [0.0]])
    dofs = operators["element_dofs"]
    local = np.einsum("eij,ej->ei", operators["element_stiffness"], padded[dofs])
    return np.bincount(dofs.ravel(), weights=local.ravel(),
                       minlength=operators["n_dof"] + 1)[:operators["n_dof"]]


def _load(operators, macroscopic_strain):
    """Right-hand side of the equilibrium system for a prescribed macroscopic strain."""
    dofs = operators["element_dofs"]
    local = -np.einsum("eij,i->ej", operators["element_stress_map"], macroscopic_strain)
    return np.bincount(dofs.ravel(), weights=local.ravel(),
                       minlength=operators["n_dof"] + 1)[:operators["n_dof"]]


def _effective_stress(operators, vector, macroscopic_strain, cell_volume):
    """Volume-averaged Mandel stress of the current displacement fluctuation."""
    padded = np.concatenate([vector, [0.0]])
    integrated = (operators["integrated_stiffness"] @ macroscopic_strain
                  + np.einsum("eij,ej->i", operators["element_stress_map"],
                              padded[operators["element_dofs"]]))
    return integrated / cell_volume


def _precondition(operators, green, vector):
    """Apply the inverse block preconditioner, the standard block through the transform."""
    n_voxels = green.shape[0]
    n_standard = 3 * operators["n_nodes"]
    field = vector[:n_standard].reshape(n_voxels, n_voxels, n_voxels, 3)
    transformed = np.fft.fftn(field, axes=(0, 1, 2))
    filtered = np.einsum("...ij,...j->...i", green, transformed)
    standard = np.fft.ifftn(filtered, axes=(0, 1, 2)).real.reshape(-1)
    return np.concatenate([standard, vector[n_standard:]])


def _solve(operators, green, macroscopic_strain, cell_volume, tolerance, max_iterations):
    """Preconditioned linear conjugate gradients from a zero displacement fluctuation."""
    macroscopic_strain = np.asarray(macroscopic_strain, dtype=float)
    right = _load(operators, macroscopic_strain)
    displacement = np.zeros(operators["n_dof"])
    residual_vector = right.copy()
    preconditioned = _precondition(operators, green, residual_vector)
    inner = residual_vector @ preconditioned
    residuals = [float(np.sqrt(inner))]
    threshold = tolerance * float(np.linalg.norm(
        _effective_stress(operators, displacement, macroscopic_strain, cell_volume)))
    direction = preconditioned.copy()
    iterations = 0
    while residuals[-1] > threshold and iterations < max_iterations:
        applied = _apply_operator(operators, direction)
        step = inner / (direction @ applied)
        displacement = displacement + step * direction
        residual_vector = residual_vector - step * applied
        preconditioned = _precondition(operators, green, residual_vector)
        inner_new = residual_vector @ preconditioned
        residuals.append(float(np.sqrt(inner_new)))
        iterations += 1
        threshold = tolerance * float(np.linalg.norm(
            _effective_stress(operators, displacement, macroscopic_strain, cell_volume)))
        if residuals[-1] <= threshold:
            break
        direction = preconditioned + (inner_new / inner) * direction
        inner = inner_new
    converged = residuals[-1] <= threshold
    return {
        "displacement": displacement,
        "iterations": iterations,
        "residuals": np.array(residuals),
        "threshold": threshold,
        "effective_stress": _effective_stress(operators, displacement,
                                              macroscopic_strain, cell_volume),
        "converged": bool(converged),
    }


def _oracle_solve_scaled_xfft_system(
    operators: dict,
    green_operator,
    macroscopic_strain,
    cell_volume: float,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Reference implementation."""
    macroscopic_strain = np.asarray(macroscopic_strain, dtype=float)
    if macroscopic_strain.shape != (6,):
        raise ValueError("macroscopic_strain must have shape (6,)")
    if float(cell_volume) <= 0.0:
        raise ValueError("cell_volume must be strictly positive")
    if float(tolerance) <= 0.0 or int(max_iterations) < 1:
        raise ValueError("tolerance must be positive and max_iterations at least one")
    return _solve(operators, np.asarray(green_operator), macroscopic_strain,
                  float(cell_volume), float(tolerance), int(max_iterations))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
def cell_mesh(n, size, centre, r_inner, r_outer):
    # the periodic three-phase mesh of stage 3 rebuilt inline from literals, so that this
    # case needs no upstream oracle
    corner = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                       [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    tetra = np.array([[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
                      [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]])
    spacing = float(size) / n
    middle = 0.5 * (r_inner + r_outer)
    axis = np.arange(n)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    place = (grid * spacing)[:, None, :] + spacing * corner[None, :, :]
    label = (grid[:, None, :] + corner[None, :, :].astype(int)) % n
    label = label[..., 0] * n * n + label[..., 1] * n + label[..., 2]
    span = np.linalg.norm(place - np.asarray(centre, dtype=float), axis=-1)
    signed = np.where(span >= middle, span - r_outer, r_inner - span)
    region = np.where(span >= r_outer, 0, np.where(span >= r_inner, 1, 2))
    vertices = place[:, tetra, :].reshape(-1, 4, 3)
    nodes = label[:, tetra].reshape(-1, 4)
    levels = signed[:, tetra].reshape(-1, 4)
    regions = region[:, tetra].reshape(-1, 4)
    cut = (levels.min(axis=1) < 0.0) & (levels.max(axis=1) > 0.0)
    outer = np.where((regions == 0).any(axis=1), 0,
                     np.where((regions == 2).any(axis=1), 2, 1))
    touched = np.unique(nodes[cut])
    slot = -np.ones(n ** 3, dtype=int)
    slot[touched] = np.arange(touched.size)
    return {"element_vertices": vertices, "element_nodes": nodes,
            "element_levels": levels, "is_cut": cut, "element_outer_phase": outer,
            "enriched_index": slot, "n_nodes": n ** 3, "n_elements": vertices.shape[0],
            "n_enriched": int(touched.size), "n_dof": 3 * n ** 3 + 3 * int(touched.size),
            "smallest_absolute_level": float(np.abs(levels).min())}
def cell_operators(mesh, stiffness):
    # the interface quadrature, the modified enrichment and the internally scaled assembly of
    # stages 4 to 6 rebuilt inline, so that this case needs no upstream oracle
    root2 = np.sqrt(2.0)
    near = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
    far = (5.0 - np.sqrt(5.0)) / 20.0
    rule = np.array([[near, far, far, far], [far, near, far, far],
                     [far, far, near, far], [far, far, far, near]])

    def mandel_column(gradient, direction):
        gx, gy, gz = gradient
        if direction == 0:
            return np.array([gx, 0.0, 0.0, 0.0, gz / root2, gy / root2])
        if direction == 1:
            return np.array([0.0, gy, 0.0, gz / root2, 0.0, gx / root2])
        return np.array([0.0, 0.0, gz, gy / root2, gx / root2, 0.0])

    def basis_gradients(vertices):
        return np.linalg.inv(np.column_stack([np.ones(4), vertices]))[1:, :].T

    def prism(lower, upper):
        return [(lower[0], lower[1], lower[2], upper[0]),
                (lower[1], lower[2], upper[0], upper[1]),
                (lower[2], upper[0], upper[1], upper[2])]

    def subcells(levels):
        below = [i for i in range(4) if levels[i] < 0.0]
        above = [i for i in range(4) if levels[i] > 0.0]
        if len(below) + len(above) != 4:
            raise ValueError("a nodal level set value vanishes")
        if not below or not above:
            return [(np.eye(4), -1 if not above else 1)]

        def vertex(index):
            point = np.zeros(4)
            point[index] = 1.0
            return point

        def crossing(i, j):
            fraction = levels[i] / (levels[i] - levels[j])
            point = np.zeros(4)
            point[i] = 1.0 - fraction
            point[j] = fraction
            return point

        cells = []
        if len(below) == 1 or len(above) == 1:
            lone = below[0] if len(below) == 1 else above[0]
            others = above if len(below) == 1 else below
            side = -1 if len(below) == 1 else 1
            cuts = [crossing(lone, other) for other in others]
            cells.append((np.array([vertex(lone), cuts[0], cuts[1], cuts[2]]), side))
            for cell in prism(cuts, [vertex(other) for other in others]):
                cells.append((np.array(cell), -side))
        else:
            first, second = below
            third, fourth = above
            cut13, cut14 = crossing(first, third), crossing(first, fourth)
            cut23, cut24 = crossing(second, third), crossing(second, fourth)
            for cell in prism([vertex(first), cut13, cut14], [vertex(second), cut23, cut24]):
                cells.append((np.array(cell), -1))
            for cell in prism([vertex(third), cut13, cut23], [vertex(fourth), cut14, cut24]):
                cells.append((np.array(cell), 1))
        return cells

    def element_rule(vertices, levels):
        bary, weights, sides = [], [], []
        for cell, side in subcells(levels):
            physical = cell @ vertices
            volume = abs(np.linalg.det((physical[1:] - physical[0]).T)) / 6.0
            bary.append(rule @ cell)
            weights.extend([volume / 4.0] * 4)
            sides.extend([side] * 4)
        return np.vstack(bary), np.array(weights), np.array(sides, dtype=int)

    def element_operator(vertices, levels, is_cut, bary, factor=None):
        gradients = basis_gradients(vertices)
        operator = np.zeros((bary.shape[0], 6, 24))
        for node in range(4):
            for direction in range(3):
                operator[:, :, 3 * node + direction] = mandel_column(gradients[node], direction)
        if is_cut:
            interpolated = bary @ levels
            sign = np.where(interpolated >= 0.0, 1.0, -1.0)
            rho = bary @ np.abs(levels) - np.abs(interpolated)
            grad_rho = ((np.abs(levels) @ gradients)[None, :]
                        - sign[:, None] * (levels @ gradients)[None, :])
            for node in range(4):
                gradient = (rho[:, None] * gradients[node][None, :]
                            + bary[:, node, None] * grad_rho)
                for direction in range(3):
                    column = np.array([mandel_column(g, direction) for g in gradient])
                    if factor is not None:
                        column = column / factor[node, direction]
                    operator[:, :, 12 + 3 * node + direction] = column
        return operator

    def element_dofs(nodes, index, is_cut, n_nodes, n_dof):
        dofs = np.empty(24, dtype=int)
        for local, node in enumerate(nodes):
            for direction in range(3):
                dofs[3 * local + direction] = 3 * int(node) + direction
        for local, node in enumerate(nodes):
            place = index[int(node)]
            for direction in range(3):
                dofs[12 + 3 * local + direction] = (
                    3 * n_nodes + 3 * place + direction if (is_cut and place >= 0) else n_dof)
        return dofs

    stiffness = np.asarray(stiffness, dtype=float)
    if stiffness.shape != (3, 6, 6):
        raise ValueError("stiffness must have shape (3, 6, 6)")
    n_elements = mesh["n_elements"]
    n_nodes = mesh["n_nodes"]
    n_dof = mesh["n_dof"]
    vertices = mesh["element_vertices"]
    levels = mesh["element_levels"]
    nodes = mesh["element_nodes"]
    is_cut = mesh["is_cut"]
    outer = mesh["element_outer_phase"]
    index = mesh["enriched_index"]

    quadrature = []
    for element in range(n_elements):
        bary, weights, sides = element_rule(vertices[element], levels[element])
        quadrature.append((bary, weights, np.where(sides < 0, 1, outer[element])))

    scale = np.zeros((mesh["n_enriched"], 3))
    for element in range(n_elements):
        if not is_cut[element]:
            continue
        bary, weights, _ = quadrature[element]
        operator = element_operator(vertices[element], levels[element], True, bary)
        for node in range(4):
            place = index[int(nodes[element, node])]
            for direction in range(3):
                column = operator[:, :, 12 + 3 * node + direction]
                scale[place, direction] += np.sum(weights * np.einsum("qj,qj->q", column, column))
    scale = np.sqrt(scale)

    element_stiffness = np.zeros((n_elements, 24, 24))
    element_stress_map = np.zeros((n_elements, 6, 24))
    dof_table = np.zeros((n_elements, 24), dtype=int)
    integrated = np.zeros((6, 6))
    points, all_weights, all_phase, all_operator, owner = [], [], [], [], []
    for element in range(n_elements):
        bary, weights, phase = quadrature[element]
        factor = scale[index[nodes[element]]] if is_cut[element] else None
        operator = element_operator(vertices[element], levels[element],
                                    bool(is_cut[element]), bary, factor)
        dofs = element_dofs(nodes[element], index, bool(is_cut[element]), n_nodes, n_dof)
        weighted = weights[:, None, None] * stiffness[phase]
        element_stiffness[element] = np.einsum("qji,qjk,qkl->il", operator, weighted, operator)
        element_stress_map[element] = np.einsum("qij,qjk->ik", weighted, operator)
        integrated += weighted.sum(axis=0)
        inactive = dofs >= n_dof
        element_stiffness[element][inactive, :] = 0.0
        element_stiffness[element][:, inactive] = 0.0
        element_stress_map[element][:, inactive] = 0.0
        dof_table[element] = dofs
        points.append(bary @ vertices[element])
        all_weights.append(weights)
        all_phase.append(phase)
        all_operator.append(operator)
        owner.append(np.full(weights.size, element))
    return {"element_stiffness": element_stiffness, "element_stress_map": element_stress_map,
            "element_dofs": dof_table, "scaling": scale, "integrated_stiffness": integrated,
            "quadrature_points": np.concatenate(points),
            "quadrature_weights": np.concatenate(all_weights),
            "quadrature_phase": np.concatenate(all_phase),
            "strain_operator": np.concatenate(all_operator),
            "quadrature_element": np.concatenate(owner), "n_dof": n_dof, "n_nodes": n_nodes,
            "n_quadrature": int(np.concatenate(all_weights).size)}

def cell_green(n, size):
    # the Fourier multiplier of stage 7 rebuilt inline, so that this case needs no upstream oracle
    root2 = np.sqrt(2.0)
    corner = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                       [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    tetra = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
             (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))
    spacing = float(size) / int(n)

    def mandel_column(gradient, direction):
        gx, gy, gz = gradient
        if direction == 0:
            return np.array([gx, 0.0, 0.0, 0.0, gz / root2, gy / root2])
        if direction == 1:
            return np.array([0.0, gy, 0.0, gz / root2, 0.0, gx / root2])
        return np.array([0.0, 0.0, gz, gy / root2, gx / root2, 0.0])

    voxel = np.zeros((24, 24))
    corners = spacing * corner
    for cell in tetra:
        vertices = corners[list(cell)]
        gradients = np.linalg.inv(np.column_stack([np.ones(4), vertices]))[1:, :].T
        volume = abs(np.linalg.det((vertices[1:] - vertices[0]).T)) / 6.0
        operator = np.zeros((6, 12))
        for node in range(4):
            for direction in range(3):
                operator[:, 3 * node + direction] = mandel_column(gradients[node], direction)
        local = volume * (operator.T @ operator)
        place = np.array([3 * cell[node] + direction
                          for node in range(4) for direction in range(3)])
        voxel[np.ix_(place, place)] += local
    green = np.zeros((n, n, n, 3, 3), dtype=complex)
    for k1 in range(n):
        for k2 in range(n):
            for k3 in range(n):
                if k1 == 0 and k2 == 0 and k3 == 0:
                    continue
                phase = np.exp(2j * np.pi * (corner @ np.array([k1, k2, k3])) / n)
                projector = np.zeros((24, 3), dtype=complex)
                for slot in range(8):
                    projector[3 * slot:3 * slot + 3, :] = phase[slot] * np.eye(3)
                green[k1, k2, k3] = np.linalg.inv(projector.conj().T @ voxel @ projector)
    return green
def stiffnesses(values, nu):
    out = []
    for bulk in values:
        lame = 3*bulk*nu/(1+nu)
        shear = 3*bulk*(1-2*nu)/(2*(1+nu))
        C = lame*np.ones((6, 6)); C[3:, :] = 0.0; C[:, 3:] = 0.0
        out.append(C + 2*shear*np.eye(6))
    return np.array(out)
MESH = cell_mesh(6, 3.0, (1.5, 1.5, 1.5), 0.42, 1.05)
OPS = cell_operators(MESH, stiffnesses([1.0, 0.808024, 8.080240], 0.25))
GREEN = cell_green(6, 3.0)
STRAIN = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
def summarize(data):
    return (data["iterations"], int(data["converged"]),
            round(float(data["residuals"][0]), 9),
            tuple(round(float(x), 10) for x in data["effective_stress"]))
""",
            "call": "summarize(solve_scaled_xfft_system(OPS, GREEN, STRAIN, 27.0, 1e-7, 200))",
            "gold_call": "summarize(_oracle_solve_scaled_xfft_system(OPS, GREEN, STRAIN, 27.0, 1e-7, 200))",
        },
        {
            "setup": """import numpy as np
def cell_mesh(n, size, centre, r_inner, r_outer):
    # the periodic three-phase mesh of stage 3 rebuilt inline from literals, so that this
    # case needs no upstream oracle
    corner = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                       [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    tetra = np.array([[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
                      [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]])
    spacing = float(size) / n
    middle = 0.5 * (r_inner + r_outer)
    axis = np.arange(n)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    place = (grid * spacing)[:, None, :] + spacing * corner[None, :, :]
    label = (grid[:, None, :] + corner[None, :, :].astype(int)) % n
    label = label[..., 0] * n * n + label[..., 1] * n + label[..., 2]
    span = np.linalg.norm(place - np.asarray(centre, dtype=float), axis=-1)
    signed = np.where(span >= middle, span - r_outer, r_inner - span)
    region = np.where(span >= r_outer, 0, np.where(span >= r_inner, 1, 2))
    vertices = place[:, tetra, :].reshape(-1, 4, 3)
    nodes = label[:, tetra].reshape(-1, 4)
    levels = signed[:, tetra].reshape(-1, 4)
    regions = region[:, tetra].reshape(-1, 4)
    cut = (levels.min(axis=1) < 0.0) & (levels.max(axis=1) > 0.0)
    outer = np.where((regions == 0).any(axis=1), 0,
                     np.where((regions == 2).any(axis=1), 2, 1))
    touched = np.unique(nodes[cut])
    slot = -np.ones(n ** 3, dtype=int)
    slot[touched] = np.arange(touched.size)
    return {"element_vertices": vertices, "element_nodes": nodes,
            "element_levels": levels, "is_cut": cut, "element_outer_phase": outer,
            "enriched_index": slot, "n_nodes": n ** 3, "n_elements": vertices.shape[0],
            "n_enriched": int(touched.size), "n_dof": 3 * n ** 3 + 3 * int(touched.size),
            "smallest_absolute_level": float(np.abs(levels).min())}
def cell_operators(mesh, stiffness):
    # the interface quadrature, the modified enrichment and the internally scaled assembly of
    # stages 4 to 6 rebuilt inline, so that this case needs no upstream oracle
    root2 = np.sqrt(2.0)
    near = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
    far = (5.0 - np.sqrt(5.0)) / 20.0
    rule = np.array([[near, far, far, far], [far, near, far, far],
                     [far, far, near, far], [far, far, far, near]])

    def mandel_column(gradient, direction):
        gx, gy, gz = gradient
        if direction == 0:
            return np.array([gx, 0.0, 0.0, 0.0, gz / root2, gy / root2])
        if direction == 1:
            return np.array([0.0, gy, 0.0, gz / root2, 0.0, gx / root2])
        return np.array([0.0, 0.0, gz, gy / root2, gx / root2, 0.0])

    def basis_gradients(vertices):
        return np.linalg.inv(np.column_stack([np.ones(4), vertices]))[1:, :].T

    def prism(lower, upper):
        return [(lower[0], lower[1], lower[2], upper[0]),
                (lower[1], lower[2], upper[0], upper[1]),
                (lower[2], upper[0], upper[1], upper[2])]

    def subcells(levels):
        below = [i for i in range(4) if levels[i] < 0.0]
        above = [i for i in range(4) if levels[i] > 0.0]
        if len(below) + len(above) != 4:
            raise ValueError("a nodal level set value vanishes")
        if not below or not above:
            return [(np.eye(4), -1 if not above else 1)]

        def vertex(index):
            point = np.zeros(4)
            point[index] = 1.0
            return point

        def crossing(i, j):
            fraction = levels[i] / (levels[i] - levels[j])
            point = np.zeros(4)
            point[i] = 1.0 - fraction
            point[j] = fraction
            return point

        cells = []
        if len(below) == 1 or len(above) == 1:
            lone = below[0] if len(below) == 1 else above[0]
            others = above if len(below) == 1 else below
            side = -1 if len(below) == 1 else 1
            cuts = [crossing(lone, other) for other in others]
            cells.append((np.array([vertex(lone), cuts[0], cuts[1], cuts[2]]), side))
            for cell in prism(cuts, [vertex(other) for other in others]):
                cells.append((np.array(cell), -side))
        else:
            first, second = below
            third, fourth = above
            cut13, cut14 = crossing(first, third), crossing(first, fourth)
            cut23, cut24 = crossing(second, third), crossing(second, fourth)
            for cell in prism([vertex(first), cut13, cut14], [vertex(second), cut23, cut24]):
                cells.append((np.array(cell), -1))
            for cell in prism([vertex(third), cut13, cut23], [vertex(fourth), cut14, cut24]):
                cells.append((np.array(cell), 1))
        return cells

    def element_rule(vertices, levels):
        bary, weights, sides = [], [], []
        for cell, side in subcells(levels):
            physical = cell @ vertices
            volume = abs(np.linalg.det((physical[1:] - physical[0]).T)) / 6.0
            bary.append(rule @ cell)
            weights.extend([volume / 4.0] * 4)
            sides.extend([side] * 4)
        return np.vstack(bary), np.array(weights), np.array(sides, dtype=int)

    def element_operator(vertices, levels, is_cut, bary, factor=None):
        gradients = basis_gradients(vertices)
        operator = np.zeros((bary.shape[0], 6, 24))
        for node in range(4):
            for direction in range(3):
                operator[:, :, 3 * node + direction] = mandel_column(gradients[node], direction)
        if is_cut:
            interpolated = bary @ levels
            sign = np.where(interpolated >= 0.0, 1.0, -1.0)
            rho = bary @ np.abs(levels) - np.abs(interpolated)
            grad_rho = ((np.abs(levels) @ gradients)[None, :]
                        - sign[:, None] * (levels @ gradients)[None, :])
            for node in range(4):
                gradient = (rho[:, None] * gradients[node][None, :]
                            + bary[:, node, None] * grad_rho)
                for direction in range(3):
                    column = np.array([mandel_column(g, direction) for g in gradient])
                    if factor is not None:
                        column = column / factor[node, direction]
                    operator[:, :, 12 + 3 * node + direction] = column
        return operator

    def element_dofs(nodes, index, is_cut, n_nodes, n_dof):
        dofs = np.empty(24, dtype=int)
        for local, node in enumerate(nodes):
            for direction in range(3):
                dofs[3 * local + direction] = 3 * int(node) + direction
        for local, node in enumerate(nodes):
            place = index[int(node)]
            for direction in range(3):
                dofs[12 + 3 * local + direction] = (
                    3 * n_nodes + 3 * place + direction if (is_cut and place >= 0) else n_dof)
        return dofs

    stiffness = np.asarray(stiffness, dtype=float)
    if stiffness.shape != (3, 6, 6):
        raise ValueError("stiffness must have shape (3, 6, 6)")
    n_elements = mesh["n_elements"]
    n_nodes = mesh["n_nodes"]
    n_dof = mesh["n_dof"]
    vertices = mesh["element_vertices"]
    levels = mesh["element_levels"]
    nodes = mesh["element_nodes"]
    is_cut = mesh["is_cut"]
    outer = mesh["element_outer_phase"]
    index = mesh["enriched_index"]

    quadrature = []
    for element in range(n_elements):
        bary, weights, sides = element_rule(vertices[element], levels[element])
        quadrature.append((bary, weights, np.where(sides < 0, 1, outer[element])))

    scale = np.zeros((mesh["n_enriched"], 3))
    for element in range(n_elements):
        if not is_cut[element]:
            continue
        bary, weights, _ = quadrature[element]
        operator = element_operator(vertices[element], levels[element], True, bary)
        for node in range(4):
            place = index[int(nodes[element, node])]
            for direction in range(3):
                column = operator[:, :, 12 + 3 * node + direction]
                scale[place, direction] += np.sum(weights * np.einsum("qj,qj->q", column, column))
    scale = np.sqrt(scale)

    element_stiffness = np.zeros((n_elements, 24, 24))
    element_stress_map = np.zeros((n_elements, 6, 24))
    dof_table = np.zeros((n_elements, 24), dtype=int)
    integrated = np.zeros((6, 6))
    points, all_weights, all_phase, all_operator, owner = [], [], [], [], []
    for element in range(n_elements):
        bary, weights, phase = quadrature[element]
        factor = scale[index[nodes[element]]] if is_cut[element] else None
        operator = element_operator(vertices[element], levels[element],
                                    bool(is_cut[element]), bary, factor)
        dofs = element_dofs(nodes[element], index, bool(is_cut[element]), n_nodes, n_dof)
        weighted = weights[:, None, None] * stiffness[phase]
        element_stiffness[element] = np.einsum("qji,qjk,qkl->il", operator, weighted, operator)
        element_stress_map[element] = np.einsum("qij,qjk->ik", weighted, operator)
        integrated += weighted.sum(axis=0)
        inactive = dofs >= n_dof
        element_stiffness[element][inactive, :] = 0.0
        element_stiffness[element][:, inactive] = 0.0
        element_stress_map[element][:, inactive] = 0.0
        dof_table[element] = dofs
        points.append(bary @ vertices[element])
        all_weights.append(weights)
        all_phase.append(phase)
        all_operator.append(operator)
        owner.append(np.full(weights.size, element))
    return {"element_stiffness": element_stiffness, "element_stress_map": element_stress_map,
            "element_dofs": dof_table, "scaling": scale, "integrated_stiffness": integrated,
            "quadrature_points": np.concatenate(points),
            "quadrature_weights": np.concatenate(all_weights),
            "quadrature_phase": np.concatenate(all_phase),
            "strain_operator": np.concatenate(all_operator),
            "quadrature_element": np.concatenate(owner), "n_dof": n_dof, "n_nodes": n_nodes,
            "n_quadrature": int(np.concatenate(all_weights).size)}

def cell_green(n, size):
    # the Fourier multiplier of stage 7 rebuilt inline, so that this case needs no upstream oracle
    root2 = np.sqrt(2.0)
    corner = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                       [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    tetra = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
             (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))
    spacing = float(size) / int(n)

    def mandel_column(gradient, direction):
        gx, gy, gz = gradient
        if direction == 0:
            return np.array([gx, 0.0, 0.0, 0.0, gz / root2, gy / root2])
        if direction == 1:
            return np.array([0.0, gy, 0.0, gz / root2, 0.0, gx / root2])
        return np.array([0.0, 0.0, gz, gy / root2, gx / root2, 0.0])

    voxel = np.zeros((24, 24))
    corners = spacing * corner
    for cell in tetra:
        vertices = corners[list(cell)]
        gradients = np.linalg.inv(np.column_stack([np.ones(4), vertices]))[1:, :].T
        volume = abs(np.linalg.det((vertices[1:] - vertices[0]).T)) / 6.0
        operator = np.zeros((6, 12))
        for node in range(4):
            for direction in range(3):
                operator[:, 3 * node + direction] = mandel_column(gradients[node], direction)
        local = volume * (operator.T @ operator)
        place = np.array([3 * cell[node] + direction
                          for node in range(4) for direction in range(3)])
        voxel[np.ix_(place, place)] += local
    green = np.zeros((n, n, n, 3, 3), dtype=complex)
    for k1 in range(n):
        for k2 in range(n):
            for k3 in range(n):
                if k1 == 0 and k2 == 0 and k3 == 0:
                    continue
                phase = np.exp(2j * np.pi * (corner @ np.array([k1, k2, k3])) / n)
                projector = np.zeros((24, 3), dtype=complex)
                for slot in range(8):
                    projector[3 * slot:3 * slot + 3, :] = phase[slot] * np.eye(3)
                green[k1, k2, k3] = np.linalg.inv(projector.conj().T @ voxel @ projector)
    return green
def stiffnesses(values, nu):
    out = []
    for bulk in values:
        lame = 3*bulk*nu/(1+nu)
        shear = 3*bulk*(1-2*nu)/(2*(1+nu))
        C = lame*np.ones((6, 6)); C[3:, :] = 0.0; C[:, 3:] = 0.0
        out.append(C + 2*shear*np.eye(6))
    return np.array(out)
MESH = cell_mesh(6, 3.0, (1.5, 1.5, 1.5), 0.42, 1.05)
OPS = cell_operators(MESH, stiffnesses([1.0, 0.808024, 8.080240], 0.25))
GREEN = cell_green(6, 3.0)
STRAIN = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
def equilibrium(fn):
    # at convergence the effective stress is stationary and tightening the tolerance by
    # three decades must not move it beyond the loose tolerance itself
    loose = fn(OPS, GREEN, STRAIN, 27.0, 1e-6, 400)
    tight = fn(OPS, GREEN, STRAIN, 27.0, 1e-9, 400)
    shift = np.abs(loose["effective_stress"] - tight["effective_stress"]).max()
    return (int(tight["iterations"] > loose["iterations"]),
            int(shift < 1e-5), round(float(np.trace(np.diag(tight["effective_stress"][:3]))), 8))
""",
            "call": "equilibrium(solve_scaled_xfft_system)",
            "gold_call": "equilibrium(_oracle_solve_scaled_xfft_system)",
        },
        {
            "setup": """import numpy as np
def cell_mesh(n, size, centre, r_inner, r_outer):
    # the periodic three-phase mesh of stage 3 rebuilt inline from literals, so that this
    # case needs no upstream oracle
    corner = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                       [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    tetra = np.array([[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
                      [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]])
    spacing = float(size) / n
    middle = 0.5 * (r_inner + r_outer)
    axis = np.arange(n)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    place = (grid * spacing)[:, None, :] + spacing * corner[None, :, :]
    label = (grid[:, None, :] + corner[None, :, :].astype(int)) % n
    label = label[..., 0] * n * n + label[..., 1] * n + label[..., 2]
    span = np.linalg.norm(place - np.asarray(centre, dtype=float), axis=-1)
    signed = np.where(span >= middle, span - r_outer, r_inner - span)
    region = np.where(span >= r_outer, 0, np.where(span >= r_inner, 1, 2))
    vertices = place[:, tetra, :].reshape(-1, 4, 3)
    nodes = label[:, tetra].reshape(-1, 4)
    levels = signed[:, tetra].reshape(-1, 4)
    regions = region[:, tetra].reshape(-1, 4)
    cut = (levels.min(axis=1) < 0.0) & (levels.max(axis=1) > 0.0)
    outer = np.where((regions == 0).any(axis=1), 0,
                     np.where((regions == 2).any(axis=1), 2, 1))
    touched = np.unique(nodes[cut])
    slot = -np.ones(n ** 3, dtype=int)
    slot[touched] = np.arange(touched.size)
    return {"element_vertices": vertices, "element_nodes": nodes,
            "element_levels": levels, "is_cut": cut, "element_outer_phase": outer,
            "enriched_index": slot, "n_nodes": n ** 3, "n_elements": vertices.shape[0],
            "n_enriched": int(touched.size), "n_dof": 3 * n ** 3 + 3 * int(touched.size),
            "smallest_absolute_level": float(np.abs(levels).min())}
def cell_operators(mesh, stiffness):
    # the interface quadrature, the modified enrichment and the internally scaled assembly of
    # stages 4 to 6 rebuilt inline, so that this case needs no upstream oracle
    root2 = np.sqrt(2.0)
    near = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
    far = (5.0 - np.sqrt(5.0)) / 20.0
    rule = np.array([[near, far, far, far], [far, near, far, far],
                     [far, far, near, far], [far, far, far, near]])

    def mandel_column(gradient, direction):
        gx, gy, gz = gradient
        if direction == 0:
            return np.array([gx, 0.0, 0.0, 0.0, gz / root2, gy / root2])
        if direction == 1:
            return np.array([0.0, gy, 0.0, gz / root2, 0.0, gx / root2])
        return np.array([0.0, 0.0, gz, gy / root2, gx / root2, 0.0])

    def basis_gradients(vertices):
        return np.linalg.inv(np.column_stack([np.ones(4), vertices]))[1:, :].T

    def prism(lower, upper):
        return [(lower[0], lower[1], lower[2], upper[0]),
                (lower[1], lower[2], upper[0], upper[1]),
                (lower[2], upper[0], upper[1], upper[2])]

    def subcells(levels):
        below = [i for i in range(4) if levels[i] < 0.0]
        above = [i for i in range(4) if levels[i] > 0.0]
        if len(below) + len(above) != 4:
            raise ValueError("a nodal level set value vanishes")
        if not below or not above:
            return [(np.eye(4), -1 if not above else 1)]

        def vertex(index):
            point = np.zeros(4)
            point[index] = 1.0
            return point

        def crossing(i, j):
            fraction = levels[i] / (levels[i] - levels[j])
            point = np.zeros(4)
            point[i] = 1.0 - fraction
            point[j] = fraction
            return point

        cells = []
        if len(below) == 1 or len(above) == 1:
            lone = below[0] if len(below) == 1 else above[0]
            others = above if len(below) == 1 else below
            side = -1 if len(below) == 1 else 1
            cuts = [crossing(lone, other) for other in others]
            cells.append((np.array([vertex(lone), cuts[0], cuts[1], cuts[2]]), side))
            for cell in prism(cuts, [vertex(other) for other in others]):
                cells.append((np.array(cell), -side))
        else:
            first, second = below
            third, fourth = above
            cut13, cut14 = crossing(first, third), crossing(first, fourth)
            cut23, cut24 = crossing(second, third), crossing(second, fourth)
            for cell in prism([vertex(first), cut13, cut14], [vertex(second), cut23, cut24]):
                cells.append((np.array(cell), -1))
            for cell in prism([vertex(third), cut13, cut23], [vertex(fourth), cut14, cut24]):
                cells.append((np.array(cell), 1))
        return cells

    def element_rule(vertices, levels):
        bary, weights, sides = [], [], []
        for cell, side in subcells(levels):
            physical = cell @ vertices
            volume = abs(np.linalg.det((physical[1:] - physical[0]).T)) / 6.0
            bary.append(rule @ cell)
            weights.extend([volume / 4.0] * 4)
            sides.extend([side] * 4)
        return np.vstack(bary), np.array(weights), np.array(sides, dtype=int)

    def element_operator(vertices, levels, is_cut, bary, factor=None):
        gradients = basis_gradients(vertices)
        operator = np.zeros((bary.shape[0], 6, 24))
        for node in range(4):
            for direction in range(3):
                operator[:, :, 3 * node + direction] = mandel_column(gradients[node], direction)
        if is_cut:
            interpolated = bary @ levels
            sign = np.where(interpolated >= 0.0, 1.0, -1.0)
            rho = bary @ np.abs(levels) - np.abs(interpolated)
            grad_rho = ((np.abs(levels) @ gradients)[None, :]
                        - sign[:, None] * (levels @ gradients)[None, :])
            for node in range(4):
                gradient = (rho[:, None] * gradients[node][None, :]
                            + bary[:, node, None] * grad_rho)
                for direction in range(3):
                    column = np.array([mandel_column(g, direction) for g in gradient])
                    if factor is not None:
                        column = column / factor[node, direction]
                    operator[:, :, 12 + 3 * node + direction] = column
        return operator

    def element_dofs(nodes, index, is_cut, n_nodes, n_dof):
        dofs = np.empty(24, dtype=int)
        for local, node in enumerate(nodes):
            for direction in range(3):
                dofs[3 * local + direction] = 3 * int(node) + direction
        for local, node in enumerate(nodes):
            place = index[int(node)]
            for direction in range(3):
                dofs[12 + 3 * local + direction] = (
                    3 * n_nodes + 3 * place + direction if (is_cut and place >= 0) else n_dof)
        return dofs

    stiffness = np.asarray(stiffness, dtype=float)
    if stiffness.shape != (3, 6, 6):
        raise ValueError("stiffness must have shape (3, 6, 6)")
    n_elements = mesh["n_elements"]
    n_nodes = mesh["n_nodes"]
    n_dof = mesh["n_dof"]
    vertices = mesh["element_vertices"]
    levels = mesh["element_levels"]
    nodes = mesh["element_nodes"]
    is_cut = mesh["is_cut"]
    outer = mesh["element_outer_phase"]
    index = mesh["enriched_index"]

    quadrature = []
    for element in range(n_elements):
        bary, weights, sides = element_rule(vertices[element], levels[element])
        quadrature.append((bary, weights, np.where(sides < 0, 1, outer[element])))

    scale = np.zeros((mesh["n_enriched"], 3))
    for element in range(n_elements):
        if not is_cut[element]:
            continue
        bary, weights, _ = quadrature[element]
        operator = element_operator(vertices[element], levels[element], True, bary)
        for node in range(4):
            place = index[int(nodes[element, node])]
            for direction in range(3):
                column = operator[:, :, 12 + 3 * node + direction]
                scale[place, direction] += np.sum(weights * np.einsum("qj,qj->q", column, column))
    scale = np.sqrt(scale)

    element_stiffness = np.zeros((n_elements, 24, 24))
    element_stress_map = np.zeros((n_elements, 6, 24))
    dof_table = np.zeros((n_elements, 24), dtype=int)
    integrated = np.zeros((6, 6))
    points, all_weights, all_phase, all_operator, owner = [], [], [], [], []
    for element in range(n_elements):
        bary, weights, phase = quadrature[element]
        factor = scale[index[nodes[element]]] if is_cut[element] else None
        operator = element_operator(vertices[element], levels[element],
                                    bool(is_cut[element]), bary, factor)
        dofs = element_dofs(nodes[element], index, bool(is_cut[element]), n_nodes, n_dof)
        weighted = weights[:, None, None] * stiffness[phase]
        element_stiffness[element] = np.einsum("qji,qjk,qkl->il", operator, weighted, operator)
        element_stress_map[element] = np.einsum("qij,qjk->ik", weighted, operator)
        integrated += weighted.sum(axis=0)
        inactive = dofs >= n_dof
        element_stiffness[element][inactive, :] = 0.0
        element_stiffness[element][:, inactive] = 0.0
        element_stress_map[element][:, inactive] = 0.0
        dof_table[element] = dofs
        points.append(bary @ vertices[element])
        all_weights.append(weights)
        all_phase.append(phase)
        all_operator.append(operator)
        owner.append(np.full(weights.size, element))
    return {"element_stiffness": element_stiffness, "element_stress_map": element_stress_map,
            "element_dofs": dof_table, "scaling": scale, "integrated_stiffness": integrated,
            "quadrature_points": np.concatenate(points),
            "quadrature_weights": np.concatenate(all_weights),
            "quadrature_phase": np.concatenate(all_phase),
            "strain_operator": np.concatenate(all_operator),
            "quadrature_element": np.concatenate(owner), "n_dof": n_dof, "n_nodes": n_nodes,
            "n_quadrature": int(np.concatenate(all_weights).size)}

def cell_green(n, size):
    # the Fourier multiplier of stage 7 rebuilt inline, so that this case needs no upstream oracle
    root2 = np.sqrt(2.0)
    corner = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                       [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    tetra = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
             (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))
    spacing = float(size) / int(n)

    def mandel_column(gradient, direction):
        gx, gy, gz = gradient
        if direction == 0:
            return np.array([gx, 0.0, 0.0, 0.0, gz / root2, gy / root2])
        if direction == 1:
            return np.array([0.0, gy, 0.0, gz / root2, 0.0, gx / root2])
        return np.array([0.0, 0.0, gz, gy / root2, gx / root2, 0.0])

    voxel = np.zeros((24, 24))
    corners = spacing * corner
    for cell in tetra:
        vertices = corners[list(cell)]
        gradients = np.linalg.inv(np.column_stack([np.ones(4), vertices]))[1:, :].T
        volume = abs(np.linalg.det((vertices[1:] - vertices[0]).T)) / 6.0
        operator = np.zeros((6, 12))
        for node in range(4):
            for direction in range(3):
                operator[:, 3 * node + direction] = mandel_column(gradients[node], direction)
        local = volume * (operator.T @ operator)
        place = np.array([3 * cell[node] + direction
                          for node in range(4) for direction in range(3)])
        voxel[np.ix_(place, place)] += local
    green = np.zeros((n, n, n, 3, 3), dtype=complex)
    for k1 in range(n):
        for k2 in range(n):
            for k3 in range(n):
                if k1 == 0 and k2 == 0 and k3 == 0:
                    continue
                phase = np.exp(2j * np.pi * (corner @ np.array([k1, k2, k3])) / n)
                projector = np.zeros((24, 3), dtype=complex)
                for slot in range(8):
                    projector[3 * slot:3 * slot + 3, :] = phase[slot] * np.eye(3)
                green[k1, k2, k3] = np.linalg.inv(projector.conj().T @ voxel @ projector)
    return green
def stiffnesses(values, nu):
    out = []
    for bulk in values:
        lame = 3*bulk*nu/(1+nu)
        shear = 3*bulk*(1-2*nu)/(2*(1+nu))
        C = lame*np.ones((6, 6)); C[3:, :] = 0.0; C[:, 3:] = 0.0
        out.append(C + 2*shear*np.eye(6))
    return np.array(out)
MESH = cell_mesh(6, 3.0, (1.5, 1.5, 1.5), 0.42, 1.05)
OPS = cell_operators(MESH, stiffnesses([1.0, 0.808024, 8.080240], 0.25))
GREEN = cell_green(6, 3.0)
STRAIN = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
def run(fn):
    codes = []
    for args in [(OPS, GREEN, np.zeros(5), 27.0, 1e-7, 200),
                 (OPS, GREEN, STRAIN, -1.0, 1e-7, 200),
                 (OPS, GREEN, STRAIN, 27.0, 1e-7, 0)]:
        try:
            fn(*args)
            codes.append(0)
        except ValueError:
            codes.append(1)
        except Exception:
            codes.append(2)
    return tuple(codes)
""",
            "call": "run(solve_scaled_xfft_system)",
            "gold_call": "run(_oracle_solve_scaled_xfft_system)",
        },
    ]
