"""
Internal enrichment scaling and cached X-FEM element operators.




The displacement gradient is represented in orthonormal Voigt-Mandel coordinates. Each enriched displacement component is scaled by the inverse square root of its cell-integrated symmetric-gradient energy, so its unit-material energy norm is one. A second quadrature pass assembles scaled 24 by 24 local matrices, macroscopic-strain loads, and stress maps. These element arrays support later matrix-free gathering and scattering.




Inputs

------

mesh : dict

    Required fields are float64 arrays nodes (v, 3), vertices (e, 4, 3), and levels (e, 4); integer arrays node_ids (e, 4) and enriched_nodes (k,); and Boolean array cut (e,), where v, e, and k count nodes, elements, and enriched nodes.

quadrature : dict

    Required fields are integer arrays element_ptr (e + 1,) and phases (q,), float64 array barycentric (q, 4), and float64 array weights (q,), where q is the total number of quadrature points.

matrix_lame : array-like of shape (2,)

inclusion_lame : array-like of shape (2,)

macrostrain : np.ndarray of shape (6,)




Returns

-------

elements : dict

    Keys scaling_energy, element_matrices, element_rhs, element_stress, element_dofs, rhs, stress_macro, n_standard_dofs, and n_enriched_dofs, with shapes and dtypes specified below.

Returns
-------
dict with float64 arrays scaling_energy (k, 3), element_matrices (e, 24, 24), element_rhs (e, 24), element_stress (e, 6, 24), rhs (d,), and stress_macro (6,); integer array element_dofs (e, 24); and native ints n_standard_dofs and n_enriched_dofs, where e is the number of elements, k is the number of enriched nodes, and d is the sum of the two degree-of-freedom counts.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def scale_and_assemble_elements(
    mesh: dict,
    quadrature: dict,
    matrix_lame: np.ndarray,
    inclusion_lame: np.ndarray,
    macrostrain: np.ndarray,
) -> dict:
    """Apply internal scaling and assemble the matrix-free element cache.

    Parameters
    ----------
    mesh : dict
        Required fields are float64 arrays nodes (v, 3), vertices (e, 4, 3), and levels (e, 4); integer arrays node_ids (e, 4) and enriched_nodes (k,); and Boolean array cut (e,), where v, e, and k count nodes, elements, and enriched nodes.
    quadrature : dict
        Required fields are integer arrays element_ptr (e + 1,) and phases (q,), float64 array barycentric (q, 4), and float64 array weights (q,), where q is the total number of quadrature points.
    matrix_lame : np.ndarray
        Matrix Lamé pair (lambda, mu).
    inclusion_lame : np.ndarray
        Inclusion Lamé pair (lambda, mu).
    macrostrain : np.ndarray
        Prescribed Mandel strain vector.

    Returns
    -------
    elements : dict
        Keys scaling_energy, element_matrices, element_rhs, element_stress, element_dofs, rhs, stress_macro, n_standard_dofs, and n_enriched_dofs, with shapes and dtypes specified below.
    Raises
    ------
    ValueError
        If mesh or quadrature is not a dict carrying every required field, if
        mesh vertices is not shape (e, 4, 3) or node_ids, levels or cut do not
        align with it, if element_ptr is not shape (e + 1,), does not start at
        0 or does not end at the number of quadrature points, if either Lame
        pair is not shape (2,) with strictly positive entries, if macrostrain
        is not shape (6,), or if any enriched component has scaling energy at
        most 1e-14.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _shape_gradients(vertices):
    interpolation = np.column_stack([np.ones(4), vertices])
    return np.linalg.inv(interpolation)[1:, :].T


def _modified_abs_gradients(barycentric, levels, gradients):
    interpolated_level = barycentric @ levels
    signs = np.where(interpolated_level >= 0.0, 1.0, -1.0)
    rho = barycentric @ np.abs(levels) - np.abs(interpolated_level)
    grad_rho = (np.abs(levels) @ gradients)[None, :] - signs[:, None] * (
        levels @ gradients
    )[None, :]
    return (
        rho[:, None, None] * gradients[None, :, :]
        + barycentric[:, :, None] * grad_rho[:, None, :]
    )


def _mandel_column(gradient, component):
    gx, gy, gz = gradient
    root_two = np.sqrt(2.0)
    if component == 0:
        return np.array([gx, 0.0, 0.0, 0.0, gz / root_two, gy / root_two])
    if component == 1:
        return np.array([0.0, gy, 0.0, gz / root_two, 0.0, gx / root_two])
    return np.array([0.0, 0.0, gz, gy / root_two, gx / root_two, 0.0])


def _isotropic_elasticity(lame_pair):
    lame_lambda, lame_mu = lame_pair
    elasticity = np.zeros((6, 6), dtype=float)
    elasticity[:3, :3] = lame_lambda
    elasticity[np.arange(3), np.arange(3)] += 2.0 * lame_mu
    elasticity[3:, 3:] = 2.0 * lame_mu * np.eye(3)
    return elasticity


def _validate_inputs(mesh, quadrature, matrix_lame, inclusion_lame, macrostrain):
    mesh_fields = {"nodes", "vertices", "node_ids", "levels", "cut", "enriched_nodes"}
    quad_fields = {"element_ptr", "barycentric", "weights", "phases"}
    if not isinstance(mesh, dict) or not mesh_fields.issubset(mesh):
        raise ValueError("mesh is missing required fields")
    if not isinstance(quadrature, dict) or not quad_fields.issubset(quadrature):
        raise ValueError("quadrature is missing required fields")
    vertices = np.asarray(mesh["vertices"], dtype=float)
    node_ids = np.asarray(mesh["node_ids"], dtype=int)
    levels = np.asarray(mesh["levels"], dtype=float)
    cut = np.asarray(mesh["cut"], dtype=bool)
    if vertices.ndim != 3 or vertices.shape[1:] != (4, 3):
        raise ValueError("mesh vertices must have shape (e, 4, 3)")
    if node_ids.shape != vertices.shape[:2] or levels.shape != vertices.shape[:2]:
        raise ValueError("mesh connectivity and level arrays do not align")
    if cut.shape != (vertices.shape[0],):
        raise ValueError("cut flags do not align with elements")
    pointer = np.asarray(quadrature["element_ptr"], dtype=int)
    if pointer.shape != (vertices.shape[0] + 1,) or pointer[0] != 0:
        raise ValueError("element_ptr has an invalid shape")
    if pointer[-1] != len(quadrature["weights"]):
        raise ValueError("element_ptr does not cover all quadrature points")
    matrix_lame = np.asarray(matrix_lame, dtype=float)
    inclusion_lame = np.asarray(inclusion_lame, dtype=float)
    macrostrain = np.asarray(macrostrain, dtype=float)
    if matrix_lame.shape != (2,) or inclusion_lame.shape != (2,):
        raise ValueError("each Lamé pair must have shape (2,)")
    if np.any(matrix_lame <= 0.0) or np.any(inclusion_lame <= 0.0):
        raise ValueError("Lamé parameters must be positive")
    if macrostrain.shape != (6,):
        raise ValueError("macrostrain must have shape (6,)")
    return matrix_lame, inclusion_lame, macrostrain


def _oracle_scale_and_assemble_elements(
    mesh: dict,
    quadrature: dict,
    matrix_lame: np.ndarray,
    inclusion_lame: np.ndarray,
    macrostrain: np.ndarray,
) -> dict:
    """Reference implementation."""
    matrix_lame, inclusion_lame, macrostrain = _validate_inputs(
        mesh, quadrature, matrix_lame, inclusion_lame, macrostrain
    )
    n_nodes = len(mesh["nodes"])
    enriched_nodes = np.asarray(mesh["enriched_nodes"], dtype=int)
    enrichment_index = {int(node): index for index, node in enumerate(enriched_nodes)}
    scaling_energy = np.zeros((len(enriched_nodes), 3), dtype=float)

    for element, (vertices, node_ids, levels, is_cut) in enumerate(
        zip(mesh["vertices"], mesh["node_ids"], mesh["levels"], mesh["cut"])
    ):
        if not is_cut:
            continue
        start, stop = quadrature["element_ptr"][element : element + 2]
        barycentric = quadrature["barycentric"][start:stop]
        weights = quadrature["weights"][start:stop]
        gradients = _shape_gradients(vertices)
        enriched_gradients = _modified_abs_gradients(barycentric, levels, gradients)
        for point, weight in enumerate(weights):
            for local_node, global_node in enumerate(node_ids):
                enriched_node = enrichment_index[int(global_node)]
                for component in range(3):
                    column = _mandel_column(
                        enriched_gradients[point, local_node], component
                    )
                    scaling_energy[enriched_node, component] += weight * (
                        column @ column
                    )
    if scaling_energy.size and np.any(scaling_energy <= 1e-14):
        raise ValueError("an enriched component has zero scaling energy")

    n_elements = len(mesh["vertices"])
    element_matrices = np.zeros((n_elements, 24, 24), dtype=float)
    element_rhs = np.zeros((n_elements, 24), dtype=float)
    element_stress = np.zeros((n_elements, 6, 24), dtype=float)
    element_dofs = np.full((n_elements, 24), -1, dtype=int)
    stress_macro = np.zeros(6, dtype=float)
    materials = {
        -1: _isotropic_elasticity(inclusion_lame),
        1: _isotropic_elasticity(matrix_lame),
    }

    for element, (vertices, node_ids, levels, is_cut) in enumerate(
        zip(mesh["vertices"], mesh["node_ids"], mesh["levels"], mesh["cut"])
    ):
        start, stop = quadrature["element_ptr"][element : element + 2]
        barycentric = quadrature["barycentric"][start:stop]
        weights = quadrature["weights"][start:stop]
        phases = quadrature["phases"][start:stop]
        gradients = _shape_gradients(vertices)
        enriched_gradients = None
        if is_cut:
            enriched_gradients = _modified_abs_gradients(barycentric, levels, gradients)

        dofs = []
        for node in node_ids:
            dofs.extend([3 * int(node) + component for component in range(3)])
        for node in node_ids:
            if int(node) in enrichment_index:
                offset = 3 * n_nodes + 3 * enrichment_index[int(node)]
                dofs.extend([offset + component for component in range(3)])
            else:
                dofs.extend([-1, -1, -1])
        element_dofs[element] = dofs

        for point, (weight, phase) in enumerate(zip(weights, phases)):
            strain_matrix = np.zeros((6, 24), dtype=float)
            for local_node in range(4):
                for component in range(3):
                    strain_matrix[:, 3 * local_node + component] = _mandel_column(
                        gradients[local_node], component
                    )
                    if is_cut:
                        global_node = int(node_ids[local_node])
                        enriched_node = enrichment_index[global_node]
                        scale = np.sqrt(scaling_energy[enriched_node, component])
                        strain_matrix[:, 12 + 3 * local_node + component] = (
                            _mandel_column(
                                enriched_gradients[point, local_node], component
                            )
                            / scale
                        )
            elasticity = materials[int(phase)]
            element_matrices[element] += weight * (
                strain_matrix.T @ elasticity @ strain_matrix
            )
            element_rhs[element] -= weight * (
                strain_matrix.T @ elasticity @ macrostrain
            )
            element_stress[element] += weight * (elasticity @ strain_matrix)
            stress_macro += weight * (elasticity @ macrostrain)

    n_standard_dofs = 3 * n_nodes
    n_enriched_dofs = 3 * len(enriched_nodes)
    right_hand_side = np.zeros(n_standard_dofs + n_enriched_dofs, dtype=float)
    for dofs, local_rhs in zip(element_dofs, element_rhs):
        active = dofs >= 0
        np.add.at(right_hand_side, dofs[active], local_rhs[active])
    return {
        "scaling_energy": scaling_energy,
        "element_matrices": element_matrices,
        "element_rhs": element_rhs,
        "element_stress": element_stress,
        "element_dofs": element_dofs,
        "rhs": right_hand_side,
        "stress_macro": stress_macro,
        "n_standard_dofs": n_standard_dofs,
        "n_enriched_dofs": n_enriched_dofs,
    }

# =============================================================================
# TEST CASES
# =============================================================================

_PRECOMPUTED_CUT_INPUTS = """import numpy as np
vertices = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
mesh = {
    "nodes": vertices.copy(),
    "vertices": vertices[None, :, :],
    "node_ids": np.array([[0, 1, 2, 3]], dtype=int),
    "levels": np.array([[-0.35, 0.65, 0.45, 0.25]], dtype=float),
    "cut": np.array([True]),
    "enriched_nodes": np.array([0, 1, 2, 3], dtype=int),
}
barycentric = np.array([
    [0.6905452238123723, 0.09959220259843841, 0.12449025324804802, 0.08537232034114124],
    [0.6481531017389388, 0.060461012992192094, 0.27123221427147176, 0.020153670997397368],
    [0.8438090497701705, 0.060461012992192094, 0.07557626624024012, 0.020153670997397368],
    [0.6872842913451851, 0.21698577141717737, 0.07557626624024012, 0.020153670997397368],
    [0.6703915528149749, 0.09959220259843841, 0.06402924025585593, 0.16598700433073071],
    [0.6671306203477878, 0.21698577141717737, 0.015115253248048023, 0.10076835498698684],
    [0.8236553787727731, 0.060461012992192094, 0.015115253248048023, 0.10076835498698684],
    [0.5627807813977975, 0.060461012992192094, 0.015115253248048023, 0.3616429523619623],
    [0.6582993502165365, 0.05122339220468474, 0.12449025324804802, 0.1659870043307307],
    [0.5506885787993592, 0.012092202598438418, 0.07557626624024011, 0.36164295236196226],
    [0.8115631761743347, 0.012092202598438418, 0.07557626624024011, 0.10076835498698682],
    [0.6159072281431031, 0.012092202598438418, 0.2712322142714717, 0.10076835498698682],
    [0.6099305398227828, 0.09959220259843841, 0.12449025324804802, 0.16598700433073071],
    [0.5023197684056054, 0.060461012992192094, 0.07557626624024011, 0.3616429523619623],
    [0.6066696073555957, 0.2169857714171774, 0.07557626624024011, 0.10076835498698684],
    [0.5675384177493493, 0.060461012992192094, 0.2712322142714717, 0.10076835498698684],
    [0.32651850864210485, 0.3182827057593821, 0.20071553926692412, 0.1544832463315889],
    [0.45664524233271764, 0.21765964677189154, 0.2892265633762908, 0.036468547519099996],
    [0.495776431938964, 0.37418440519687685, 0.0935706153450592, 0.036468547519099996],
    [0.2050875948639913, 0.6648732422718495, 0.0935706153450592, 0.036468547519099996],
    [0.23669071791084806, 0.26991389536562843, 0.3389121403919346, 0.1544832463315889],
    [0.36681745160146084, 0.16929083637813785, 0.42742316450130136, 0.036468547519099996],
    [0.11525980413273446, 0.6165044318780959, 0.2317672164700697, 0.036468547519099996],
    [0.11525980413273446, 0.16929083637813785, 0.6789808119700277, 0.036468547519099996],
    [0.3063648376447075, 0.1800861046343716, 0.14025452627473203, 0.37329453144618885],
    [0.47562276094156664, 0.23598780407186631, 0.0331096023528671, 0.25527983263369997],
    [0.3712729219915764, 0.07946304564688103, 0.0331096023528671, 0.5161544300086754],
    [0.1849339238665939, 0.07946304564688103, 0.0331096023528671, 0.702493428133658],
    [0.24878292050928646, 0.3182827057593821, 0.14025452627473203, 0.2926798474565994],
    [0.4180408438061456, 0.37418440519687685, 0.0331096023528671, 0.1746651486441105],
    [0.1273520067311729, 0.21765964677189154, 0.0331096023528671, 0.6218787441440685],
    [0.1273520067311729, 0.6648732422718495, 0.0331096023528671, 0.1746651486441105],
    [0.2942726350462691, 0.13171729424061793, 0.3389121403919346, 0.23509793032117837],
    [0.359180719393138, 0.031094235253127364, 0.2317672164700697, 0.3779578288836649],
    [0.4243993687368819, 0.031094235253127364, 0.42742316450130124, 0.11708323150868943],
    [0.1728417212681555, 0.031094235253127364, 0.6789808119700277, 0.11708323150868943],
    [0.2165370469134507, 0.13171729424061793, 0.27845112739974254, 0.37329453144618885],
    [0.2814451312603196, 0.031094235253127364, 0.17130620347787762, 0.5161544300086754],
    [0.09510613313533708, 0.031094235253127364, 0.6185197989778356, 0.25527983263369997],
    [0.09510613313533708, 0.031094235253127364, 0.17130620347787762, 0.702493428133658],
    [0.15895512977802964, 0.26991389536562843, 0.27845112739974254, 0.2926798474565994],
    [0.037524215999916045, 0.6165044318780959, 0.17130620347787762, 0.1746651486441105],
    [0.037524215999916045, 0.16929083637813785, 0.17130620347787762, 0.6218787441440685],
    [0.037524215999916045, 0.16929083637813785, 0.6185197989778356, 0.1746651486441105],
    [0.3841004257775259, 0.1800861046343716, 0.20071553926692412, 0.23509793032117837],
    [0.4490085101243948, 0.07946304564688103, 0.09357061534505919, 0.37795782888366497],
    [0.553358349074385, 0.23598780407186634, 0.09357061534505919, 0.11708323150868946],
    [0.5142271594681387, 0.07946304564688103, 0.28922656337629077, 0.11708323150868946],
], dtype=float)
weights = np.array([
    0.0009304470486111112, 0.0009304470486111112, 0.0009304470486111112, 0.0009304470486111112,
    0.0009304470486111112, 0.0009304470486111112, 0.0009304470486111112, 0.0009304470486111112,
    0.0009304470486111112, 0.0009304470486111112, 0.0009304470486111112, 0.0009304470486111112,
    0.0009304470486111107, 0.0009304470486111107, 0.0009304470486111107, 0.0009304470486111107,
    0.003126808449074075, 0.003126808449074075, 0.003126808449074075, 0.003126808449074075,
    0.006184895833333333, 0.006184895833333333, 0.006184895833333333, 0.006184895833333333,
    0.0014558015046296296, 0.0014558015046296296, 0.0014558015046296296, 0.0014558015046296296,
    0.006488715277777779, 0.006488715277777779, 0.006488715277777779, 0.006488715277777779,
    0.0030761718750000003, 0.0030761718750000003, 0.0030761718750000003, 0.0030761718750000003,
    0.003906250000000001, 0.003906250000000001, 0.003906250000000001, 0.003906250000000001,
    0.011313657407407406, 0.011313657407407406, 0.011313657407407406, 0.011313657407407406,
    0.0023925781250000006, 0.0023925781250000006, 0.0023925781250000006, 0.0023925781250000006,
], dtype=float)
phases = np.array([
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
], dtype=int)
quadrature = {
    "element_ptr": np.array([0, 48], dtype=int),
    "barycentric": barycentric,
    "weights": weights,
    "phases": phases,
}
"""


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": _PRECOMPUTED_CUT_INPUTS + """
matrix_lame = np.array([3., 2.])
inclusion_lame = np.array([30., 20.])
macrostrain = np.array([1., 0., 0., 0., 0., 0.])
def summarize(result):
    return (
        result["element_matrices"].shape,
        result["element_dofs"].shape,
        result["rhs"].shape,
        round(float(np.min(result["scaling_energy"])), 14),
        round(float(np.max(result["scaling_energy"])), 14),
        round(float(np.linalg.norm(result["rhs"])), 12),
        np.round(result["stress_macro"], 12).tolist(),
    )
""",
            "call": "summarize(scale_and_assemble_elements(mesh, quadrature, matrix_lame, inclusion_lame, macrostrain))",
            "gold_call": "summarize(_oracle_scale_and_assemble_elements(mesh, quadrature, matrix_lame, inclusion_lame, macrostrain))",
        },
        {
            "setup": """import numpy as np
vertices = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
mesh = {"nodes": vertices[0], "vertices": vertices, "node_ids": np.arange(4)[None, :], "levels": np.ones((1, 4)), "cut": np.array([False]), "enriched_nodes": np.array([], dtype=int)}
high = (5. + 3. * np.sqrt(5.)) / 20.
low = (5. - np.sqrt(5.)) / 20.
quadrature = {"element_ptr": np.array([0, 4]), "barycentric": np.array([[high, low, low, low], [low, high, low, low], [low, low, high, low], [low, low, low, high]]), "weights": np.full(4, 1. / 24.), "phases": np.ones(4, dtype=int)}
matrix_lame = np.array([3., 2.])
inclusion_lame = np.array([30., 20.])
macrostrain = np.zeros(6)
def summarize(result):
    return (
        result["scaling_energy"].shape,
        result["rhs"].tolist(),
        result["n_enriched_dofs"],
    )
""",
            "call": "summarize(scale_and_assemble_elements(mesh, quadrature, matrix_lame, inclusion_lame, macrostrain))",
            "gold_call": "summarize(_oracle_scale_and_assemble_elements(mesh, quadrature, matrix_lame, inclusion_lame, macrostrain))",
        },
        {
            "setup": _PRECOMPUTED_CUT_INPUTS + """
matrix_lame = np.array([3., 2.])
inclusion_lame = np.array([30., 20.])
macrostrain = np.zeros(5)
def run_model():
    try:
        scale_and_assemble_elements(mesh, quadrature, matrix_lame, inclusion_lame, macrostrain)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_scale_and_assemble_elements(mesh, quadrature, matrix_lame, inclusion_lame, macrostrain)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
