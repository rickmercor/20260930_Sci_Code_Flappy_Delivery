"""
Differentiate discrete path work while transporting the residual-state sensitivity through loading and unloading.

Let the reference geometry and all material parameters be fixed while a dimensionless parameter changes the imposed nodal path. With the same right-handed rotations and basis as the problem statement, define



$$

B=\\operatorname{diag}(1,-1/2,-1/2),\\qquad

F_n=R_z(\\theta_n)\\{I+a_n[R_y(\\phi_n)BR_y(\\phi_n)^T+0.1I]\\},

\\qquad x_{n,i}(\\lambda)=F_nX_i+\\lambda\\eta_{n,i}.

$$



The input perturbations are arbitrary nodal vectors, not necessarily affine within the full mesh. Each tetrahedron therefore has its own deformation-gradient variation even when the unperturbed path is affine. Coordinates and perturbations have units mm, and the initial perturbation is zero. Replay the maximum-history constitutive law from zero state, with no intermediate frames or equilibrium solves, and let the initial force and its sensitivity be exactly zero. The target is



$$

W(\\lambda)=-\\sum_{n=1}^{N_f-1}\\sum_i

\\dfrac{f_{n,i}(\\lambda)+f_{n-1,i}(\\lambda)}{2}\\cdot

[x_{n,i}(\\lambda)-x_{n-1,i}(\\lambda)],

\\qquad \\mathcal S=.\\dfrac{dW}{d\\lambda}|_0.

$$



The force Jacobian from the preceding step holds previous internal state fixed. The present derivative instead follows the entire perturbed loading history: a stored plastic tensor can be unchanged between two frames and still depend on the perturbation at an earlier frame. Differentiate both the force and displacement factors of the work pairing, and carry the selected state branch through the replay. Exact ties use the inactive-branch linearization without differentiating the activity indicator; this convention is not a claim that a two-sided derivative exists at a branch tie. The result has units N mm. It is the sensitivity of this discrete functional, not a thermodynamic energy difference. Any numerical method meeting the stated contract and tolerance is acceptable.

Returns
-------
Return the selected-branch directional derivative of discrete path work in N mm as one finite Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def prescribed_path_work(
    reference: np.ndarray,
    cells: np.ndarray,
    materials: np.ndarray,
    amplitudes: np.ndarray,
    rotations: np.ndarray,
    direction_angles: np.ndarray,
    perturbations: np.ndarray,
) -> float:
    r"""Return the directional sensitivity of discrete resisting-force work along a nodal path.

    Reference coordinates must be finite with shape (n_nodes, 3), n_nodes >= 4.
    Connectivity must be a nonempty integer (n_elements, 4) array with distinct, in-
    range vertices per cell. A cell is rejected if its absolute edge determinant is at
    most 1e-12 times the product of its edge lengths. Materials must satisfy E > 0, -1 <
    nu < 0.5, sigma_y > 0, H >= 0, beta > 0, C >= 0. All arrays are finite. Frame
    vectors have matching nonempty shapes, at least two frames, and initial amplitude
    and angles exactly zero. Each amplitude must satisfy 1 + 1.1*a > 0 and 1 - 0.4*a >
    0. No random seed is used. Invalid inputs raise ValueError.

    At each frame use x(lambda) = x(0) + lambda * perturbations, with lambda
    dimensionless and reference geometry and materials fixed. Differentiate the
    complete replay, including the state carried from earlier frames. At a branch
    tie return the selected inactive-branch linearization, without differentiating
    the activity flag; this is not asserted to be a two-sided derivative at a tie.
    Initial force, state, and their rates are exactly zero. Perturbations must have
    shape (n_frames, n_nodes, 3), be finite, and vanish exactly at frame zero.

    Parameters
    ----------
    reference : np.ndarray
        Reference points, shape (n_nodes, 3), in mm.
    cells : np.ndarray
        Zero-based connectivity, shape (n_elements, 4).
    materials : np.ndarray
        Rows (E, nu, sigma_y, H, beta, C), shape (n_elements, 6); E, sigma_y, H in MPa.
    amplitudes : np.ndarray
        Dimensionless amplitude vector, shape (n_frames,), in replay order.
    rotations : np.ndarray
        Right-handed z-axis angles theta in radians, same shape.
    direction_angles : np.ndarray
        Right-handed y-axis material-direction angles phi in radians, same shape.

    perturbations : np.ndarray
        Nodal path direction in mm per dimensionless lambda, shape
        (n_frames, n_nodes, 3); first frame must be exactly zero.

    Raises
    ------
    ValueError
        If mesh data are nonfinite, malformed, disconnected from valid indices, or
        degenerate; material rows have the wrong shape or violate E > 0, -1 < nu <
        0.5, sigma_y > 0, H >= 0, beta > 0, or C >= 0; frame arrays are nonfinite,
        mismatched, or shorter than two frames; the initial amplitude or angles are
        nonzero; a prescribed stretch is not positive definite; or the resulting work
        is nonfinite; or perturbations are nonfinite, incorrectly shaped, or nonzero
        at the initial frame.

    Returns
    -------
    float
        One finite Python float: selected-branch work sensitivity dW/dlambda in N mm.
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


def _mesh_arrays(reference, cells):
    points = _finite_array(reference)
    cells = np.asarray(cells)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("reference must have shape (n_nodes, 3), n_nodes >= 4")
    if cells.ndim != 2 or cells.shape[1] != 4 or len(cells) == 0:
        raise ValueError("cells must have shape (n_elements, 4)")
    if not np.issubdtype(cells.dtype, np.integer):
        raise ValueError("cell indices must be integers")
    if np.any(cells < 0) or np.any(cells >= len(points)):
        raise ValueError("cell index out of range")
    if np.any(np.diff(np.sort(cells, axis=1), axis=1) == 0):
        raise ValueError("each cell needs four distinct vertices")
    edges = (points[cells[:, 1:]] - points[cells[:, :1]]).transpose(0, 2, 1)
    lengths = np.linalg.norm(edges, axis=1)
    scale = np.prod(lengths, axis=1)
    determinant = np.linalg.det(edges)
    if np.any(scale == 0) or np.any(np.abs(determinant) <= 1e-12 * scale):
        raise ValueError(
            "reference tetrahedra are degenerate at relative tolerance 1e-12"
        )
    return points, cells.astype(int), edges


def _path_rotation(angle, axis):
    cosine, sine = np.cos(angle), np.sin(angle)
    if axis == "z":
        return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])


def _oracle_prescribed_path_work(
    reference: np.ndarray,
    cells: np.ndarray,
    materials: np.ndarray,
    amplitudes: np.ndarray,
    rotations: np.ndarray,
    direction_angles: np.ndarray,
    perturbations: np.ndarray,
) -> float:
    points, cells, _ = _mesh_arrays(reference, cells)
    material = _finite_array(materials)
    if material.shape != (len(cells), 6):
        raise ValueError("materials must have shape (n_elements, 6)")
    young, poisson, sy, hardening, beta, rate = material.T
    if (
        np.any(young <= 0)
        or np.any(poisson <= -1)
        or np.any(poisson >= 0.5)
        or np.any(sy <= 0)
        or np.any(hardening < 0)
        or np.any(beta <= 0)
        or np.any(rate < 0)
    ):
        raise ValueError("invalid material parameters")
    amplitude, angles, directions = _matched_vectors(
        amplitudes, rotations, direction_angles
    )
    if len(amplitude) < 2 or amplitude[0] != 0 or angles[0] != 0 or directions[0] != 0:
        raise ValueError(
            "path needs at least two frames and starts at zero amplitude and angles"
        )
    if np.any(1.0 + 1.1 * amplitude <= 0) or np.any(1.0 - 0.4 * amplitude <= 0):
        raise ValueError("prescribed stretch must be positive definite")

    path_rate = _finite_array(perturbations)
    if path_rate.shape != (len(amplitude), len(points), 3):
        raise ValueError("perturbations must have shape (n_frames, n_nodes, 3)")
    if np.any(path_rate[0] != 0.0):
        raise ValueError("the initial nodal perturbation must be zero")

    mu = young / (2.0 * (1.0 + poisson))
    bulk = young / (3.0 * (1.0 - 2.0 * poisson))
    volumes, gradients = _oracle_reference_geometry(points, cells)
    history = np.zeros(len(cells))
    plastic = np.zeros((len(cells), 3, 3))
    previous_points = points.copy()
    previous_forces = np.zeros_like(points)
    previous_force_rate = np.zeros_like(points)
    previous_point_rate = np.zeros_like(points)
    plastic_rate = np.zeros_like(plastic)
    basis = np.diag([1.0, -0.5, -0.5])
    total = 0.0

    for frame, (amplitude_n, angle, direction) in enumerate(
        zip(amplitude[1:], angles[1:], directions[1:]), start=1
    ):
        axes = _path_rotation(direction, "y")
        prescribed_stretch = np.eye(3) + amplitude_n * (
            axes @ basis @ axes.T + 0.1 * np.eye(3)
        )
        current = points @ (_path_rotation(angle, "z") @ prescribed_stretch).T
        deform, rotation, stretch, strain, deviator, equivalent, strain_jacobian = (
            _oracle_corotational_kinematics(points, current, cells)
        )
        candidate, slope, curvature = _oracle_normalized_activation(
            equivalent, mu, sy, beta
        )
        history, plastic, gp, active = _oracle_irreversible_state(
            deviator, equivalent, candidate, slope, history, plastic
        )
        _, _, derivative, derivative_h = _oracle_inelastic_work(
            history, sy, hardening, rate
        )
        stress, material_tangent = _oracle_branch_stress(
            strain,
            deviator,
            equivalent,
            plastic,
            history,
            gp,
            active,
            bulk,
            mu,
            derivative,
            np.where(active == 1.0, curvature, 0.0),
            derivative_h,
        )
        forces, stiffness = _oracle_reference_forces(
            deform,
            rotation,
            stretch,
            stress,
            material_tangent,
            strain_jacobian,
            volumes,
            gradients,
            cells,
            len(points),
        )
        point_rate = path_rate[frame]
        deformation_rate = np.einsum("eia,eib->eab", point_rate[cells], gradients)
        strain_rate_mandel = np.einsum(
            "eij,ej->ei", strain_jacobian, deformation_rate.reshape(len(cells), 9)
        )
        strain_rate = np.array(
            [_mandel_tensor_from_vector(vector) for vector in strain_rate_mandel]
        )
        deviator_rate = strain_rate - (np.trace(strain_rate, axis1=1, axis2=2) / 3.0)[
            :, None, None
        ] * np.eye(3)
        loading = active == 1.0
        # The local Jacobian holds prior state fixed; unloading retains its rate.
        memory_stress_rate = -2.0 * mu[:, None, None] * plastic_rate
        memory_stress_rate[loading] = 0.0
        memory_piola_rate = np.linalg.det(deform)[:, None, None] * (
            rotation @ memory_stress_rate @ np.linalg.inv(stretch)
        )
        memory_force_rate = np.zeros_like(points)
        local_memory_rate = -volumes[:, None, None] * np.einsum(
            "eab,eib->eia", memory_piola_rate, gradients
        )
        np.add.at(
            memory_force_rate, cells.reshape(-1), local_memory_rate.reshape(-1, 3)
        )
        force_rate = (stiffness @ point_rate.reshape(-1)).reshape(points.shape)
        force_rate += memory_force_rate
        if np.any(loading):
            q = equivalent[loading]
            e = deviator[loading]
            edot = deviator_rate[loading]
            qdot = (2.0 / (3.0 * q)) * np.sum(e * edot, axis=(1, 2))
            hdot = slope[loading] * qdot
            plastic_rate[loading] = (history[loading] / q)[:, None, None] * edot + (
                hdot / q - history[loading] * qdot / q**2
            )[:, None, None] * e
        total -= 0.5 * (
            np.sum((force_rate + previous_force_rate) * (current - previous_points))
            + np.sum((forces + previous_forces) * (point_rate - previous_point_rate))
        )
        previous_force_rate = force_rate
        previous_point_rate = point_rate
        previous_points = current
        previous_forces = forces

    if not np.isfinite(total):
        raise ValueError("path-work sensitivity is not finite")
    return float(total)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return full-path, rigid, history-pulse, endpoint, linearity, and invalid cases."""
    return [
        {
            "setup": "\nimport numpy as np\nX = np.array([[0., 0., 0.], [1., .1, 0.], [.2, 1.2, .1], [.1, .2, 1.1], [1.1, 1., 1.3]])\nT = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])\nM = np.array([[20., .3, 2., .5, 12., 2.2], [30., .25, 1.2, 2.4, 8., 4.4]])\na = np.array([0., .03, .09, .16, .07, .13, .18, .04, .12])\ntheta = np.array([0., .1, .2, .3, .2, .1, -.1, .15, .25])\nphi = .01 * np.arange(9)\nZ = np.array([[.03, -.02, .01], [-.01, .04, .02], [.02, .01, -.03], [-.04, .02, .01], [.01, -.03, .04]])\nb = np.array([0., .2, -.3, .4, -.1, .15, -.2, .3, -.4])\neta = b[:, None, None] * Z[None, :, :]\n",
            "call": "prescribed_path_work(X, T, M, a, theta, phi, eta)",
            "gold_call": "_oracle_prescribed_path_work(X, T, M, a, theta, phi, eta)",
        },
        {
            "setup": "\nimport numpy as np\nX = np.array([[0., 0., 0.], [1., .1, 0.], [.2, 1.2, .1], [.1, .2, 1.1], [1.1, 1., 1.3]])\nT = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])\nM = np.array([[20., .3, 2., .5, 12., 2.2], [30., .25, 1.2, 2.4, 8., 4.4]])\na = np.array([0., .03, .09, .16, .07, .13, .18, .04, .12])\ntheta = np.array([0., .1, .2, .3, .2, .1, -.1, .15, .25])\nphi = .01 * np.arange(9)\nZ = np.array([[.03, -.02, .01], [-.01, .04, .02], [.02, .01, -.03], [-.04, .02, .01], [.01, -.03, .04]])\nb = np.array([0., .2, -.3, .4, -.1, .15, -.2, .3, -.4])\neta = b[:, None, None] * Z[None, :, :]\na[:] = 0\n",
            "call": "prescribed_path_work(X, T, M, a, theta, phi, eta)",
            "gold_call": "_oracle_prescribed_path_work(X, T, M, a, theta, phi, eta)",
        },
        {
            "setup": "\nimport numpy as np\nX = np.array([[0., 0., 0.], [1., .1, 0.], [.2, 1.2, .1], [.1, .2, 1.1], [1.1, 1., 1.3]])\nT = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])\nM = np.array([[20., .3, 2., .5, 12., 2.2], [30., .25, 1.2, 2.4, 8., 4.4]])\na = np.array([0., .03, .09, .16, .07, .13, .18, .04, .12])\ntheta = np.array([0., .1, .2, .3, .2, .1, -.1, .15, .25])\nphi = .01 * np.arange(9)\nZ = np.array([[.03, -.02, .01], [-.01, .04, .02], [.02, .01, -.03], [-.04, .02, .01], [.01, -.03, .04]])\nb = np.array([0., .2, -.3, .4, -.1, .15, -.2, .3, -.4])\neta = b[:, None, None] * Z[None, :, :]\nM[:, 5] = 0\n",
            "call": "prescribed_path_work(X, T, M, a, theta, phi, eta)",
            "gold_call": "_oracle_prescribed_path_work(X, T, M, a, theta, phi, eta)",
        },
        {
            "setup": "\nimport numpy as np\nX = np.array([[0., 0., 0.], [1., .1, 0.], [.2, 1.2, .1], [.1, .2, 1.1], [1.1, 1., 1.3]])\nT = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])\nM = np.array([[20., .3, 2., .5, 12., 2.2], [30., .25, 1.2, 2.4, 8., 4.4]])\na = np.array([0., .03, .09, .16, .07, .13, .18, .04, .12])\ntheta = np.array([0., .1, .2, .3, .2, .1, -.1, .15, .25])\nphi = .01 * np.arange(9)\nZ = np.array([[.03, -.02, .01], [-.01, .04, .02], [.02, .01, -.03], [-.04, .02, .01], [.01, -.03, .04]])\nb = np.array([0., .2, -.3, .4, -.1, .15, -.2, .3, -.4])\neta = b[:, None, None] * Z[None, :, :]\n\neta[:] = 0.0\neta[3] = Z\n",
            "call": "prescribed_path_work(X, T, M, a, theta, phi, eta)",
            "gold_call": "_oracle_prescribed_path_work(X, T, M, a, theta, phi, eta)",
        },
        {
            "setup": "\nimport numpy as np\nX = np.array([[0., 0., 0.], [1., .1, 0.], [.2, 1.2, .1], [.1, .2, 1.1], [1.1, 1., 1.3]])\nT = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])\nM = np.array([[20., .3, 2., .5, 12., 2.2], [30., .25, 1.2, 2.4, 8., 4.4]])\na = np.array([0., .03, .09, .16, .07, .13, .18, .04, .12])\ntheta = np.array([0., .1, .2, .3, .2, .1, -.1, .15, .25])\nphi = .01 * np.arange(9)\nZ = np.array([[.03, -.02, .01], [-.01, .04, .02], [.02, .01, -.03], [-.04, .02, .01], [.01, -.03, .04]])\nb = np.array([0., .2, -.3, .4, -.1, .15, -.2, .3, -.4])\neta = b[:, None, None] * Z[None, :, :]\n\neta[:] = 0.0\neta[-1] = Z\n",
            "call": "prescribed_path_work(X, T, M, a, theta, phi, eta)",
            "gold_call": "_oracle_prescribed_path_work(X, T, M, a, theta, phi, eta)",
        },
        {
            "setup": "\nimport numpy as np\nX = np.array([[0., 0., 0.], [1., .1, 0.], [.2, 1.2, .1], [.1, .2, 1.1], [1.1, 1., 1.3]])\nT = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])\nM = np.array([[20., .3, 2., .5, 12., 2.2], [30., .25, 1.2, 2.4, 8., 4.4]])\na = np.array([0., .03, .09, .16, .07, .13, .18, .04, .12])\ntheta = np.array([0., .1, .2, .3, .2, .1, -.1, .15, .25])\nphi = .01 * np.arange(9)\nZ = np.array([[.03, -.02, .01], [-.01, .04, .02], [.02, .01, -.03], [-.04, .02, .01], [.01, -.03, .04]])\nb = np.array([0., .2, -.3, .4, -.1, .15, -.2, .3, -.4])\neta = b[:, None, None] * Z[None, :, :]\n\neta = -3.0 * eta\n",
            "call": "prescribed_path_work(X, T, M, a, theta, phi, eta)",
            "gold_call": "_oracle_prescribed_path_work(X, T, M, a, theta, phi, eta)",
        },
        {
            "setup": "\nimport numpy as np\nX = np.array([[0., 0., 0.], [1., .1, 0.], [.2, 1.2, .1], [.1, .2, 1.1], [1.1, 1., 1.3]])\nT = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])\nM = np.array([[20., .3, 2., .5, 12., 2.2], [30., .25, 1.2, 2.4, 8., 4.4]])\na = np.array([0., .03, .09, .16, .07, .13, .18, .04, .12])\ntheta = np.array([0., .1, .2, .3, .2, .1, -.1, .15, .25])\nphi = .01 * np.arange(9)\nZ = np.array([[.03, -.02, .01], [-.01, .04, .02], [.02, .01, -.03], [-.04, .02, .01], [.01, -.03, .04]])\nb = np.array([0., .2, -.3, .4, -.1, .15, -.2, .3, -.4])\neta = b[:, None, None] * Z[None, :, :]\na[0] = .01\ndef _invalid_status(function):\n    try:\n        function(X, T, M, a, theta, phi, eta)\n    except ValueError:\n        return 1.0\n    except Exception:\n        return 2.0\n    return 0.0\n",
            "call": "_invalid_status(prescribed_path_work)",
            "gold_call": "_invalid_status(_oracle_prescribed_path_work)",
        },
    ]
