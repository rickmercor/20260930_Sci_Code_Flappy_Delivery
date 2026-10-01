"""
Assemble resisting forces and their consistent node-major Jacobian under finite deformation.

The constitutive tensor is interpreted as Cauchy stress in the corotated frame. Rotating it into the spatial frame gives



$$

\\sigma_{spatial}=R\\sigma_{cr}R^T.

$$



Reference-volume force assembly requires the first Piola stress. With $F=RS$ and $J=\\det F>0$,



$$

P=J\\sigma_{spatial}F^{-T}

=JR\\sigma_{cr}S^{-1}.

$$



The Jacobian and inverse stretch are essential at finite deformation; using only $R\\sigma_{cr}$ is a small-strain approximation. For a linear tetrahedron, the local resisting force is



$$

f_{e,i}=-V_eP_e\\nabla N_{e,i},

$$



and the global force is assembled as



$$

f_i=\\sum_{e\\ni i}f_{e,i}.

$$



The supplied gradients are rows in local vertex order, and contributions at shared nodes are added. Since $V_e$ is a reference volume, it must not be multiplied by $J$ again. Partition of unity implies zero resultant force for a constant element stress, providing a useful assembly check.



The force Jacobian differentiates the entire finite-geometry map. For a nodal perturbation, $\\delta F=\\delta x_i\\otimes\\nabla N_i$; the kinematic Fréchet derivative supplies $\\delta S$, the material tangent supplies $\\delta\\sigma_{cr}$, and $\\delta R=(\\delta F-R\\delta S)S^{-1}$. The remaining identities are



$$

\\delta J=J\\operatorname{tr}(F^{-1}\\delta F),

\\qquad

\\delta S^{-1}=-S^{-1}(\\delta S)S^{-1}.

$$



Differentiate every factor of $P=JR\\sigma_{cr}S^{-1}$ and scatter each local derivative into node-major degrees of freedom $(x_0,y_0,z_0,x_1,\\ldots)$. Freezing $J$, $R$, or $S^{-1}$ gives a plausible material-only stiffness but not the requested finite-deformation force Jacobian.

Returns
-------
Return assembled resisting forces and their node-major position Jacobian with shapes (node_count, 3) and (3*node_count, 3*node_count).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def reference_forces(
    deformation: np.ndarray,
    rotation: np.ndarray,
    stretch: np.ndarray,
    stress: np.ndarray,
    material_tangent: np.ndarray,
    strain_jacobian: np.ndarray,
    volumes: np.ndarray,
    gradients: np.ndarray,
    cells: np.ndarray,
    node_count: int,
) -> tuple:
    r"""Map stress to nodal forces and consistently linearize with respect to positions.

    Inputs must be finite with the stated shapes and positive volumes and det(F).
    Connectivity must have distinct, in-range integer vertices. R must be proper
    orthogonal, S positive definite and symmetric, stress symmetric, F=RS, and local
    gradients sum to zero, all tensor consistency checks within atol=1e-10,
    rtol=1e-8. The material tangent is symmetric in Mandel order
    (xx, yy, zz, yz, xz, xy), and strain_jacobian maps row-major F increments to
    Mandel strain increments. Invalid inputs raise ValueError.

    Parameters
    ----------
    deformation : np.ndarray
        F, shape (n_elements, 3, 3), dimensionless.
    rotation : np.ndarray
        Proper R from polar decomposition, same shape.
    stretch : np.ndarray
        Symmetric positive-definite S, same shape.
    stress : np.ndarray
        Symmetric corotational Cauchy stress, same shape, MPa.
    material_tangent : np.ndarray
        Consistent corotational Mandel tangent, shape (n_elements, 6, 6), MPa.
    strain_jacobian : np.ndarray
        Mandel-strain derivative with respect to row-major F, shape
        (n_elements, 6, 9).
    volumes : np.ndarray
        Positive reference volumes, shape (n_elements,), mm cubed.
    gradients : np.ndarray
        Reference gradients, shape (n_elements, 4, 3), inverse mm.
    cells : np.ndarray
        Integer zero-based connectivity, shape (n_elements, 4).
    node_count : int
        Number of global nodes, at least four.

    Raises
    ------
    ValueError
        If inputs are nonfinite or malformed; volumes are nonpositive; connectivity is
        nonintegral, repeated, or out of range; node_count is invalid; F reverses
        orientation; R is not proper orthogonal; S is not symmetric positive definite;
        stress is nonsymmetric; F differs from RS; or local gradients do not sum to
        zero within the stated tolerances.

    Returns
    -------
    tuple
        (forces, stiffness): arrays of shapes (node_count, 3) and
        (3*node_count, 3*node_count), in N and N/mm. Stiffness is the Jacobian of
        resisting force with node-major displacement degrees of freedom.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _finite_array(value):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("numeric inputs must be finite")
    return array


def _matched_vectors(*values):
    arrays = tuple(_finite_array(value) for value in values)
    if not arrays or arrays[0].ndim != 1 or arrays[0].size == 0:
        raise ValueError("expected nonempty one-dimensional arrays")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("vector shapes must match")
    return arrays


def _symmetric_tensors(value, count):
    array = _finite_array(value)
    if array.shape != (count, 3, 3):
        raise ValueError("expected shape (n_elements, 3, 3)")
    if not np.allclose(array, array.swapaxes(1, 2), rtol=0.0, atol=1e-10):
        raise ValueError("tensors must be symmetric to absolute tolerance 1e-10")
    return array


def _mandel_tensor_from_vector(vector):
    root_two = np.sqrt(2.0)
    tensor = np.array(
        [
            [vector[0], vector[5] / root_two, vector[4] / root_two],
            [vector[5] / root_two, vector[1], vector[3] / root_two],
            [vector[4] / root_two, vector[3] / root_two, vector[2]],
        ]
    )
    return tensor


def _oracle_reference_forces(
    deformation: np.ndarray,
    rotation: np.ndarray,
    stretch: np.ndarray,
    stress: np.ndarray,
    material_tangent: np.ndarray,
    strain_jacobian: np.ndarray,
    volumes: np.ndarray,
    gradients: np.ndarray,
    cells: np.ndarray,
    node_count: int,
) -> tuple:
    volumes = _finite_array(volumes)
    if volumes.ndim != 1 or volumes.size == 0 or np.any(volumes <= 0):
        raise ValueError("volumes must be a nonempty positive vector")
    count = len(volumes)
    if (
        isinstance(node_count, (bool, np.bool_))
        or not isinstance(node_count, (int, np.integer))
        or node_count < 4
    ):
        raise ValueError("node_count must be an integer at least four")
    deform = _finite_array(deformation)
    rotation = _finite_array(rotation)
    stretch = _symmetric_tensors(stretch, count)
    stress = _symmetric_tensors(stress, count)
    tangent = _finite_array(material_tangent)
    strain_jacobian = _finite_array(strain_jacobian)
    gradients = _finite_array(gradients)
    cells = np.asarray(cells)
    if (
        deform.shape != (count, 3, 3)
        or rotation.shape != deform.shape
        or gradients.shape != (count, 4, 3)
        or tangent.shape != (count, 6, 6)
        or strain_jacobian.shape != (count, 6, 9)
    ):
        raise ValueError("invalid tensor or gradient shape")
    if cells.shape != (count, 4) or not np.issubdtype(cells.dtype, np.integer):
        raise ValueError("invalid connectivity shape or dtype")
    if (
        np.any(cells < 0)
        or np.any(cells >= node_count)
        or np.any(np.diff(np.sort(cells, axis=1), axis=1) == 0)
    ):
        raise ValueError("invalid connectivity indices")
    jacobian = np.linalg.det(deform)
    if np.any(jacobian <= 0) or np.any(np.linalg.eigvalsh(stretch) <= 0):
        raise ValueError("deformation and stretch must preserve orientation")
    checks = [
        (rotation @ rotation.swapaxes(1, 2), np.eye(3)),
        (np.linalg.det(rotation), np.ones(count)),
        (deform, rotation @ stretch),
        (gradients.sum(axis=1), np.zeros((count, 3))),
    ]
    if any(not np.allclose(a, b, atol=1e-10, rtol=1e-8) for a, b in checks):
        raise ValueError("inconsistent rotation, stretch, or reference gradients")
    if not np.allclose(tangent, tangent.swapaxes(1, 2), atol=1e-10, rtol=1e-8):
        raise ValueError("material tangent must be Mandel symmetric")
    piola = jacobian[:, None, None] * (rotation @ stress @ np.linalg.inv(stretch))
    local_forces = -volumes[:, None, None] * (gradients @ piola.swapaxes(1, 2))
    forces = np.zeros((node_count, 3))
    np.add.at(forces, cells.reshape(-1), local_forces.reshape(-1, 3))
    stiffness = np.zeros((3 * node_count, 3 * node_count))
    for element in range(count):
        inverse_deform = np.linalg.inv(deform[element])
        inverse_stretch = np.linalg.inv(stretch[element])
        base = rotation[element] @ stress[element] @ inverse_stretch
        for local_node, global_node in enumerate(cells[element]):
            for component in range(3):
                deform_increment = np.zeros((3, 3))
                deform_increment[component] = gradients[element, local_node]
                strain_mandel = strain_jacobian[element] @ deform_increment.reshape(-1)
                stretch_increment = _mandel_tensor_from_vector(strain_mandel)
                rotation_increment = (
                    deform_increment - rotation[element] @ stretch_increment
                ) @ inverse_stretch
                stress_increment = _mandel_tensor_from_vector(
                    tangent[element] @ strain_mandel
                )
                jacobian_increment = jacobian[element] * np.trace(
                    inverse_deform @ deform_increment
                )
                inverse_stretch_increment = (
                    -inverse_stretch @ stretch_increment @ inverse_stretch
                )
                piola_increment = jacobian_increment * base + jacobian[element] * (
                    rotation_increment @ stress[element] @ inverse_stretch
                    + rotation[element] @ stress_increment @ inverse_stretch
                    + rotation[element] @ stress[element] @ inverse_stretch_increment
                )
                force_increment = -volumes[element] * (
                    gradients[element] @ piola_increment.T
                )
                column = 3 * global_node + component
                for response_node, response_global in enumerate(cells[element]):
                    row = slice(3 * response_global, 3 * response_global + 3)
                    stiffness[row, column] += force_increment[response_node]
    if not np.all(np.isfinite(stiffness)):
        raise ValueError("force Jacobian exceeds finite arithmetic range")
    return forces, stiffness

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": """
import numpy as np
T = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
S = np.array([np.diag([1.2, .9, 1.1]), [[1.1, .04, 0.], [.04, .95, .01], [0., .01, 1.03]]])
R = np.array([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], np.eye(3)])
F = R @ S
stress = np.array([[[2., .4, .1], [.4, 1., 0.], [.1, 0., 3.]], np.diag([1., 2., 4.])])
V = np.array([.2, .3])
grad = np.array([[[-1., -1., -1.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], [[-.6, -.8, -1.], [.5, .1, 0.], [.1, .7, .2], [0., 0., .8]]])
# Fixed strain-map input for the two deformation gradients above.
Jstrain = np.array([[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -0.6363961030678927, 0.0, 0.0, 0.0, 0.0, 0.7778174593052024, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.7378505542816147, 0.6763630080914803, 0.0, 0.0], [-0.6060915267313265, 0.0, 0.0, 0.0, 0.8081220356417687, 0.0, 0.0, 0.0, 0.0]], [[0.9999999999999999, -0.01951264216300682, 9.164341639248195e-05, 0.019512642163006785, 3.2892474858558264e-19, -1.8513821493437883e-06, -9.164341639245427e-05, 1.8513821493430133e-06, 1.1967183450311348e-21], [-1.1711442876453445e-18, 0.019512179317469548, 3.2399187613510823e-06, -0.019512179317469493, 0.9999999999999999, -0.0050505705034092725, -3.2399187613515596e-06, 0.005050570503409251, 1.1171235350650311e-19], [3.276294357493557e-21, 4.6284553733576275e-07, -9.48833351537959e-05, -4.6284553733596233e-07, 1.0139534158906797e-18, 0.005052421885558416, 9.488333515383756e-05, -0.005052421885558617, 0.9999999999999999], [-5.950340601923167e-19, -6.741993092693477e-05, 0.01382108584003135, 6.741993092696961e-05, -1.4729484874819875e-16, 0.6782575963741614, -0.013821085840031316, 0.7359559659989335, -1.2916557668216046e-17], [2.7211717509888604e-17, -0.003561474215278687, 0.730102214132138, 0.003561474215278551, 2.804588790230447e-18, 0.013820431277595099, 0.684111348240957, -0.013820431277595123, 1.8883053570453477e-17], [3.131603927859297e-17, 0.7588636875780754, -0.0035640924650234187, 0.6553498747950196, -2.615075959732461e-17, 7.200186798025797e-05, 0.0035640924650234074, -7.200186798023577e-05, -9.426865652048757e-20]]], dtype=np.float64)
_m = np.array([1., 1., 1., 0., 0., 0.])
_Pdev = np.eye(6) - np.outer(_m, _m) / 3.
Cmat = np.array([17. * np.outer(_m, _m) + 12. * _Pdev, 23. * np.outer(_m, _m) + 19. * _Pdev])
Cmat[1, 3, 5] = Cmat[1, 5, 3] = 1.7
""",
            "call": "np.concatenate([np.asarray(x).reshape(-1) for x in reference_forces(F, R, S, stress, Cmat, Jstrain, V, grad, T, 5)])",
            "gold_call": "np.concatenate([np.asarray(x).reshape(-1) for x in _oracle_reference_forces(F, R, S, stress, Cmat, Jstrain, V, grad, T, 5)])",
        },
        {
            "setup": """
import numpy as np
T = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
S = np.array([np.diag([1.2, .9, 1.1]), [[1.1, .04, 0.], [.04, .95, .01], [0., .01, 1.03]]])
R = np.array([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], np.eye(3)])
F = R @ S
stress = np.array([[[2., .4, .1], [.4, 1., 0.], [.1, 0., 3.]], np.diag([1., 2., 4.])])
V = np.array([.2, .3])
grad = np.array([[[-1., -1., -1.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], [[-.6, -.8, -1.], [.5, .1, 0.], [.1, .7, .2], [0., 0., .8]]])
# Fixed strain-map input for the two deformation gradients above.
Jstrain = np.array([[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -0.6363961030678927, 0.0, 0.0, 0.0, 0.0, 0.7778174593052024, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.7378505542816147, 0.6763630080914803, 0.0, 0.0], [-0.6060915267313265, 0.0, 0.0, 0.0, 0.8081220356417687, 0.0, 0.0, 0.0, 0.0]], [[0.9999999999999999, -0.01951264216300682, 9.164341639248195e-05, 0.019512642163006785, 3.2892474858558264e-19, -1.8513821493437883e-06, -9.164341639245427e-05, 1.8513821493430133e-06, 1.1967183450311348e-21], [-1.1711442876453445e-18, 0.019512179317469548, 3.2399187613510823e-06, -0.019512179317469493, 0.9999999999999999, -0.0050505705034092725, -3.2399187613515596e-06, 0.005050570503409251, 1.1171235350650311e-19], [3.276294357493557e-21, 4.6284553733576275e-07, -9.48833351537959e-05, -4.6284553733596233e-07, 1.0139534158906797e-18, 0.005052421885558416, 9.488333515383756e-05, -0.005052421885558617, 0.9999999999999999], [-5.950340601923167e-19, -6.741993092693477e-05, 0.01382108584003135, 6.741993092696961e-05, -1.4729484874819875e-16, 0.6782575963741614, -0.013821085840031316, 0.7359559659989335, -1.2916557668216046e-17], [2.7211717509888604e-17, -0.003561474215278687, 0.730102214132138, 0.003561474215278551, 2.804588790230447e-18, 0.013820431277595099, 0.684111348240957, -0.013820431277595123, 1.8883053570453477e-17], [3.131603927859297e-17, 0.7588636875780754, -0.0035640924650234187, 0.6553498747950196, -2.615075959732461e-17, 7.200186798025797e-05, 0.0035640924650234074, -7.200186798023577e-05, -9.426865652048757e-20]]], dtype=np.float64)
_m = np.array([1., 1., 1., 0., 0., 0.])
_Pdev = np.eye(6) - np.outer(_m, _m) / 3.
Cmat = np.array([17. * np.outer(_m, _m) + 12. * _Pdev, 23. * np.outer(_m, _m) + 19. * _Pdev])
Cmat[1, 3, 5] = Cmat[1, 5, 3] = 1.7
S[:] = np.eye(3)
R[:] = np.eye(3)
F = R @ S
""",
            "call": "np.concatenate([np.asarray(x).reshape(-1) for x in reference_forces(F, R, S, stress, Cmat, Jstrain, V, grad, T, 5)])",
            "gold_call": "np.concatenate([np.asarray(x).reshape(-1) for x in _oracle_reference_forces(F, R, S, stress, Cmat, Jstrain, V, grad, T, 5)])",
        },
        {
            "setup": """
import numpy as np
T = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
S = np.array([np.diag([1.2, .9, 1.1]), [[1.1, .04, 0.], [.04, .95, .01], [0., .01, 1.03]]])
R = np.array([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], np.eye(3)])
F = R @ S
stress = np.array([[[2., .4, .1], [.4, 1., 0.], [.1, 0., 3.]], np.diag([1., 2., 4.])])
V = np.array([.2, .3])
grad = np.array([[[-1., -1., -1.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], [[-.6, -.8, -1.], [.5, .1, 0.], [.1, .7, .2], [0., 0., .8]]])
# Fixed strain-map input for the two deformation gradients above.
Jstrain = np.array([[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -0.6363961030678927, 0.0, 0.0, 0.0, 0.0, 0.7778174593052024, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.7378505542816147, 0.6763630080914803, 0.0, 0.0], [-0.6060915267313265, 0.0, 0.0, 0.0, 0.8081220356417687, 0.0, 0.0, 0.0, 0.0]], [[0.9999999999999999, -0.01951264216300682, 9.164341639248195e-05, 0.019512642163006785, 3.2892474858558264e-19, -1.8513821493437883e-06, -9.164341639245427e-05, 1.8513821493430133e-06, 1.1967183450311348e-21], [-1.1711442876453445e-18, 0.019512179317469548, 3.2399187613510823e-06, -0.019512179317469493, 0.9999999999999999, -0.0050505705034092725, -3.2399187613515596e-06, 0.005050570503409251, 1.1171235350650311e-19], [3.276294357493557e-21, 4.6284553733576275e-07, -9.48833351537959e-05, -4.6284553733596233e-07, 1.0139534158906797e-18, 0.005052421885558416, 9.488333515383756e-05, -0.005052421885558617, 0.9999999999999999], [-5.950340601923167e-19, -6.741993092693477e-05, 0.01382108584003135, 6.741993092696961e-05, -1.4729484874819875e-16, 0.6782575963741614, -0.013821085840031316, 0.7359559659989335, -1.2916557668216046e-17], [2.7211717509888604e-17, -0.003561474215278687, 0.730102214132138, 0.003561474215278551, 2.804588790230447e-18, 0.013820431277595099, 0.684111348240957, -0.013820431277595123, 1.8883053570453477e-17], [3.131603927859297e-17, 0.7588636875780754, -0.0035640924650234187, 0.6553498747950196, -2.615075959732461e-17, 7.200186798025797e-05, 0.0035640924650234074, -7.200186798023577e-05, -9.426865652048757e-20]]], dtype=np.float64)
_m = np.array([1., 1., 1., 0., 0., 0.])
_Pdev = np.eye(6) - np.outer(_m, _m) / 3.
Cmat = np.array([17. * np.outer(_m, _m) + 12. * _Pdev, 23. * np.outer(_m, _m) + 19. * _Pdev])
Cmat[1, 3, 5] = Cmat[1, 5, 3] = 1.7
stress[1] *= -2.
V *= 8
grad /= 2
""",
            "call": "np.concatenate([np.asarray(x).reshape(-1) for x in reference_forces(F, R, S, stress, Cmat, Jstrain, V, grad, T, 5)])",
            "gold_call": "np.concatenate([np.asarray(x).reshape(-1) for x in _oracle_reference_forces(F, R, S, stress, Cmat, Jstrain, V, grad, T, 5)])",
        },
        {
            "setup": """
import numpy as np
T = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
S = np.array([np.diag([1.2, .9, 1.1]), [[1.1, .04, 0.], [.04, .95, .01], [0., .01, 1.03]]])
R = np.array([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], np.eye(3)])
F = R @ S
stress = np.array([[[2., .4, .1], [.4, 1., 0.], [.1, 0., 3.]], np.diag([1., 2., 4.])])
V = np.array([.2, .3])
grad = np.array([[[-1., -1., -1.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], [[-.6, -.8, -1.], [.5, .1, 0.], [.1, .7, .2], [0., 0., .8]]])
# Fixed strain-map input for the two deformation gradients above.
Jstrain = np.array([[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -0.6363961030678927, 0.0, 0.0, 0.0, 0.0, 0.7778174593052024, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.7378505542816147, 0.6763630080914803, 0.0, 0.0], [-0.6060915267313265, 0.0, 0.0, 0.0, 0.8081220356417687, 0.0, 0.0, 0.0, 0.0]], [[0.9999999999999999, -0.01951264216300682, 9.164341639248195e-05, 0.019512642163006785, 3.2892474858558264e-19, -1.8513821493437883e-06, -9.164341639245427e-05, 1.8513821493430133e-06, 1.1967183450311348e-21], [-1.1711442876453445e-18, 0.019512179317469548, 3.2399187613510823e-06, -0.019512179317469493, 0.9999999999999999, -0.0050505705034092725, -3.2399187613515596e-06, 0.005050570503409251, 1.1171235350650311e-19], [3.276294357493557e-21, 4.6284553733576275e-07, -9.48833351537959e-05, -4.6284553733596233e-07, 1.0139534158906797e-18, 0.005052421885558416, 9.488333515383756e-05, -0.005052421885558617, 0.9999999999999999], [-5.950340601923167e-19, -6.741993092693477e-05, 0.01382108584003135, 6.741993092696961e-05, -1.4729484874819875e-16, 0.6782575963741614, -0.013821085840031316, 0.7359559659989335, -1.2916557668216046e-17], [2.7211717509888604e-17, -0.003561474215278687, 0.730102214132138, 0.003561474215278551, 2.804588790230447e-18, 0.013820431277595099, 0.684111348240957, -0.013820431277595123, 1.8883053570453477e-17], [3.131603927859297e-17, 0.7588636875780754, -0.0035640924650234187, 0.6553498747950196, -2.615075959732461e-17, 7.200186798025797e-05, 0.0035640924650234074, -7.200186798023577e-05, -9.426865652048757e-20]]], dtype=np.float64)
_m = np.array([1., 1., 1., 0., 0., 0.])
_Pdev = np.eye(6) - np.outer(_m, _m) / 3.
Cmat = np.array([17. * np.outer(_m, _m) + 12. * _Pdev, 23. * np.outer(_m, _m) + 19. * _Pdev])
Cmat[1, 3, 5] = Cmat[1, 5, 3] = 1.7
V[0] = 0
def _invalid_status(function):
    try:
        function(F, R, S, stress, Cmat, Jstrain, V, grad, T, 5)
    except ValueError:
        return 1.0
    except Exception:
        return 2.0
    return 0.0
""",
            "call": "_invalid_status(reference_forces)",
            "gold_call": "_invalid_status(_oracle_reference_forces)",
        },
    ]
