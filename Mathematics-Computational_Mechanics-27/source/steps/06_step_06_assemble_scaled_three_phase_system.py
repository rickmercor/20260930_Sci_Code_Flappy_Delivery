"""
This step turns the mesh and the three phase stiffnesses into the element operators the solver will use, and into the single global array of quadrature points, weights and phases that the error functional will later read.

It performs two passes over the elements, and the order matters. The first pass evaluates the unscaled enriched columns and accumulates the reference energy of each enriched node and direction, because a scaling factor is an integral over every element that shares the node and cannot be known from one element alone. The second pass rebuilds the columns with the factors applied and forms the element stiffness and stress map.

The step calls the interface quadrature step and the enrichment step for every element rather than repeating their logic. Inactive columns, those of enriched slots on uncut elements, are zeroed and directed at a discard index, so the later matrix-free product may sum over all twenty-four local entries without testing whether each one exists.

Each cut element carries three extra displacement degrees of freedom per node, whose basis functions are $N_i rho^m$. Those functions are far smaller than the standard basis functions on an element that is barely cut, so the two blocks of the system sit on wildly different scales and an identity preconditioner on the enriched block is meaningless. The remedy is an internal scaling that gives every enriched basis vector unit reference energy. For enriched node $j$ and displacement direction $a$ the factor is $D_{ja} = \int_Y \| \operatorname{sym} \operatorname{grad} (N_j rho^m e_a) \|^2 dV$, and the corresponding enriched column of the strain operator is divided by $\sqrt{D_{ja}}$. The scaling is componentwise: it produces one factor per node and per direction, not one factor per node. The symmetrised gradient appears because the kinematic variable of the mechanical problem is the strain.

With the scaled operator $B_e$ in hand the element stiffness and the element stress map are $A_e = \sum_q w_q B_e(q)^{T} C(q) B_e(q)$ and $S_e = \sum_q w_q C(q) B_e(q)$, the sums running over the quadrature points of the element. The stiffness at a quadrature point is that of the coating where the interpolated level set is negative and that of the element's positive-side phase otherwise. Enriched columns of uncut elements vanish identically and are marked inactive.

Returns
-------
dict, the element stiffness and stress maps with the scaled assembly and quadrature operators.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_scaled_three_phase_system(mesh: dict, stiffness) -> dict:
    """Assemble the internally scaled element operators of the enriched system.

    Parameters
    ----------
    mesh : dict
        Mesh description as returned by the mesh construction step.
    stiffness : array_like
        Array of shape (3, 6, 6) of Mandel stiffness matrices ordered matrix, coating, inclusion.

    Returns
    -------
    dict
        Keys element_stiffness, element_stress_map, element_dofs, scaling, integrated_stiffness, quadrature_points, quadrature_weights, quadrature_phase, strain_operator, quadrature_element, n_dof, n_nodes and n_quadrature.

    Raises
    ------
    ValueError
        If stiffness does not have shape (3, 6, 6).
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

R2 = np.sqrt(2.0)
MATRIX, COATING, INCLUSION = 0, 1, 2

def _strain_column(gradient, direction):
    """Mandel strain of the vector field N e_direction whose scalar gradient is given."""
    gx, gy, gz = gradient
    if direction == 0:
        return np.array([gx, 0.0, 0.0, 0.0, gz / R2, gy / R2])
    if direction == 1:
        return np.array([0.0, gy, 0.0, gz / R2, 0.0, gx / R2])
    return np.array([0.0, 0.0, gz, gy / R2, gx / R2, 0.0])

def _shape_gradients(vertices):
    """Gradients of the four linear basis functions of a tetrahedron."""
    return np.linalg.inv(np.column_stack([np.ones(4), vertices]))[1:, :].T


def _element_strain_operator(vertices, levels, is_cut, barycentric, enriched_scale=None):
    """Mandel strain operator of one element at its quadrature points, shape (q, 6, 24).

    Columns 0 to 11 hold the standard linear basis and columns 12 to 23 the enriched basis of
    the same four nodes. The enriched columns vanish on uncut elements. When enriched_scale is
    supplied it holds one factor per node and displacement direction.
    """
    gradients = _shape_gradients(vertices)
    operator = np.zeros((barycentric.shape[0], 6, 24))
    for node in range(4):
        for direction in range(3):
            operator[:, :, 3 * node + direction] = _strain_column(gradients[node], direction)
    if is_cut:
        enrichment = _oracle_evaluate_modified_abs_enrichment(
            barycentric, levels, gradients)
        rho, grad_rho = enrichment["rho"], enrichment["grad_rho"]
        for node in range(4):
            gradient = (rho[:, None] * gradients[node][None, :]
                        + barycentric[:, node, None] * grad_rho)
            for direction in range(3):
                column = np.array([_strain_column(g, direction) for g in gradient])
                if enriched_scale is not None:
                    column = column / enriched_scale[node, direction]
                operator[:, :, 12 + 3 * node + direction] = column
    return operator


def _element_dofs(nodes, enriched_index, is_cut, n_nodes, n_dof):
    """Global degree-of-freedom indices of one element, inactive entries mapped to n_dof."""
    dofs = np.empty(24, dtype=int)
    for local, node in enumerate(nodes):
        for direction in range(3):
            dofs[3 * local + direction] = 3 * int(node) + direction
    for local, node in enumerate(nodes):
        slot = enriched_index[int(node)]
        for direction in range(3):
            dofs[12 + 3 * local + direction] = (
                3 * n_nodes + 3 * slot + direction if (is_cut and slot >= 0) else n_dof)
    return dofs


def _assemble(mesh, stiffness):
    """Internal scaling factors, element operators and quadrature of the whole cell."""
    n_elements = mesh["n_elements"]
    n_nodes = mesh["n_nodes"]
    n_dof = mesh["n_dof"]
    vertices = mesh["element_vertices"]
    levels = mesh["element_levels"]
    nodes = mesh["element_nodes"]
    is_cut = mesh["is_cut"]
    outer = mesh["element_outer_phase"]
    enriched_index = mesh["enriched_index"]

    quadrature = []
    for element in range(n_elements):
        rule = _oracle_build_interface_subcell_quadrature(
            vertices[element], levels[element])
        phase = np.where(rule["side"] < 0, COATING, outer[element])
        quadrature.append((rule["barycentric"], rule["weights"], phase))

    scale = np.zeros((mesh["n_enriched"], 3))
    for element in range(n_elements):
        if not is_cut[element]:
            continue
        bary, weights, _ = quadrature[element]
        operator = _element_strain_operator(vertices[element], levels[element], True, bary)
        for node in range(4):
            slot = enriched_index[int(nodes[element, node])]
            for direction in range(3):
                column = operator[:, :, 12 + 3 * node + direction]
                scale[slot, direction] += np.sum(weights * np.einsum("qj,qj->q", column, column))
    scale = np.sqrt(scale)

    element_stiffness = np.zeros((n_elements, 24, 24))
    element_stress_map = np.zeros((n_elements, 6, 24))
    element_dofs = np.zeros((n_elements, 24), dtype=int)
    integrated_stiffness = np.zeros((6, 6))
    points, all_weights, all_phase, all_operator, owner = [], [], [], [], []
    for element in range(n_elements):
        bary, weights, phase = quadrature[element]
        local_scale = scale[enriched_index[nodes[element]]] if is_cut[element] else None
        operator = _element_strain_operator(vertices[element], levels[element],
                                            bool(is_cut[element]), bary, local_scale)
        dofs = _element_dofs(nodes[element], enriched_index, bool(is_cut[element]),
                             n_nodes, n_dof)
        weighted = weights[:, None, None] * stiffness[phase]
        element_stiffness[element] = np.einsum("qji,qjk,qkl->il", operator, weighted, operator)
        element_stress_map[element] = np.einsum("qij,qjk->ik", weighted, operator)
        integrated_stiffness += weighted.sum(axis=0)
        inactive = dofs >= n_dof
        element_stiffness[element][inactive, :] = 0.0
        element_stiffness[element][:, inactive] = 0.0
        element_stress_map[element][:, inactive] = 0.0
        element_dofs[element] = dofs
        points.append(bary @ vertices[element])
        all_weights.append(weights)
        all_phase.append(phase)
        all_operator.append(operator)
        owner.append(np.full(weights.size, element))
    return {
        "element_stiffness": element_stiffness,
        "element_stress_map": element_stress_map,
        "element_dofs": element_dofs,
        "scaling": scale,
        "integrated_stiffness": integrated_stiffness,
        "quadrature_points": np.concatenate(points),
        "quadrature_weights": np.concatenate(all_weights),
        "quadrature_phase": np.concatenate(all_phase),
        "strain_operator": np.concatenate(all_operator),
        "quadrature_element": np.concatenate(owner),
        "n_dof": n_dof,
        "n_nodes": n_nodes,
        "n_quadrature": int(np.concatenate(all_weights).size),
    }


def _oracle_assemble_scaled_three_phase_system(mesh: dict, stiffness) -> dict:
    """Reference implementation."""
    stiffness = np.asarray(stiffness, dtype=float)
    if stiffness.shape != (3, 6, 6):
        raise ValueError("stiffness must have shape (3, 6, 6)")
    return _assemble(mesh, stiffness)

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
def stiffnesses(values, nu):
    out = []
    for bulk in values:
        lame = 3*bulk*nu/(1+nu)
        shear = 3*bulk*(1-2*nu)/(2*(1+nu))
        C = lame*np.ones((6, 6)); C[3:, :] = 0.0; C[:, 3:] = 0.0
        out.append(C + 2*shear*np.eye(6))
    return np.array(out)
MESH = cell_mesh(6, 3.0, (1.5, 1.5, 1.5), 0.42, 1.05)
STIFF = stiffnesses([1.0, 0.808024, 8.080240], 0.25)
def summarize(data):
    return (data["n_quadrature"], data["n_dof"],
            round(float(data["scaling"].min()), 11),
            round(float(data["scaling"].max()), 9),
            round(float(data["quadrature_weights"].sum()), 9),
            round(float(np.trace(data["integrated_stiffness"])), 8))
""",
            "call": "summarize(assemble_scaled_three_phase_system(MESH, STIFF))",
            "gold_call": "summarize(_oracle_assemble_scaled_three_phase_system(MESH, STIFF))",
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
def stiffnesses(values, nu):
    out = []
    for bulk in values:
        lame = 3*bulk*nu/(1+nu)
        shear = 3*bulk*(1-2*nu)/(2*(1+nu))
        C = lame*np.ones((6, 6)); C[3:, :] = 0.0; C[:, 3:] = 0.0
        out.append(C + 2*shear*np.eye(6))
    return np.array(out)
MESH = cell_mesh(6, 3.0, (1.5, 1.5, 1.5), 0.42, 1.05)
STIFF = stiffnesses([1.0, 0.808024, 8.080240], 0.25)
def properties(data, mesh):
    A = data["element_stiffness"]
    # every element stiffness is symmetric positive semi-definite, and a rigid translation
    # of the standard block produces no strain energy
    translation = np.zeros(24); translation[0::3][:4] = 1.0
    energy = np.einsum("i,eij,j->e", translation, A, translation)
    unit = np.abs(A - np.transpose(A, (0, 2, 1))).max()
    inactive = data["element_dofs"] >= data["n_dof"]
    return (round(float(unit), 12), round(float(np.abs(energy).max()), 10),
            int(inactive[~mesh["is_cut"]].sum()), int(inactive[mesh["is_cut"]].sum()))
""",
            "call": "properties(assemble_scaled_three_phase_system(MESH, STIFF), MESH)",
            "gold_call": "properties(_oracle_assemble_scaled_three_phase_system(MESH, STIFF), MESH)",
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
MESH = cell_mesh(6, 3.0, (1.5, 1.5, 1.5), 0.42, 1.05)
def run(fn):
    try:
        fn(MESH, np.zeros((2, 6, 6)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(assemble_scaled_three_phase_system)",
            "gold_call": "run(_oracle_assemble_scaled_three_phase_system)",
        },
    ]
