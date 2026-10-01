"""
Transfer the material point state to the background grid, returning the internal

nodal force vector assembled from the particle stresses and the external nodal

force vector assembled from the applied point loads.

The discrete force balance of the material point method lives on the grid, and

the two force vectors that enter it are both obtained by particle to grid

transfer. The internal force is the material point quadrature of the internal

virtual work, so the contribution of point p to node v is the current particle

volume times the Cauchy stress acting on the spatial gradient of the basis

function, V_p sigma_p grad_x S_vp, where the current volume follows from the

reference volume through the Jacobian, V_p = J V_p^0. The gradient required here

is taken with respect to the current configuration, whereas the basis functions

are formed once from the reference material point positions and held fixed

through the load increment; the two are related by the chain rule through the

deformation gradient, so the spatial gradient is the pull of the reference

gradient by the inverse transpose of F. Making that distinction is what keeps

the discrete internal force the exact gradient of the stored energy of the

quadrature, and hence the tangent symmetric. The external force is simpler,

since prescribed point loads are carried by the material points and are

distributed to the nodes by the basis weights themselves, without any gradient.

Returns
-------
tuple (internal_forces, external_forces) of np.ndarray, both of shape (n_nodes, 2) and float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def assemble_nodal_forces(cauchy_stress: np.ndarray, deformation_gradients: np.ndarray,
                          volumes: np.ndarray, shape_values: np.ndarray,
                          shape_gradients: np.ndarray, point_loads: np.ndarray):
    """Assemble the internal and external nodal force vectors.

    Parameters
    ----------
    cauchy_stress : np.ndarray
        Cauchy stress at each material point, shape (n_points, 2, 2).
    deformation_gradients : np.ndarray
        Deformation gradients, shape (n_points, 2, 2).
    volumes : np.ndarray
        Reference volumes, shape (n_points,).
    shape_values : np.ndarray
        Basis weights, shape (n_points, n_nodes).
    shape_gradients : np.ndarray
        Reference basis gradients, shape (n_points, n_nodes, 2).
    point_loads : np.ndarray
        External force applied at each material point, shape (n_points, 2).

    Returns
    -------
    nodal_forces : tuple of np.ndarray
        The pair (internal_forces, external_forces), both of shape
        (n_nodes, 2).

    Raises
    ------
    ValueError
        If `shape_values` is not two dimensional, if the stress, deformation
        gradient, volume or point load arrays are not sized by n_points, if
        `shape_gradients` is not of shape (n_points, n_nodes, 2), or if any
        deformation gradient has a non-positive determinant.
    """
    return nodal_forces

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_assemble_nodal_forces(cauchy_stress: np.ndarray, deformation_gradients: np.ndarray,
                                  volumes: np.ndarray, shape_values: np.ndarray,
                                  shape_gradients: np.ndarray, point_loads: np.ndarray):
    """Reference implementation."""
    stress = np.asarray(cauchy_stress, dtype=float)
    gradients = np.asarray(deformation_gradients, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    values = np.asarray(shape_values, dtype=float)
    reference_gradients = np.asarray(shape_gradients, dtype=float)
    loads = np.asarray(point_loads, dtype=float)
    if values.ndim != 2:
        raise ValueError("shape_values must have shape (n_points, n_nodes)")
    n_points, n_nodes = values.shape
    if stress.shape != (n_points, 2, 2) or gradients.shape != (n_points, 2, 2):
        raise ValueError("stress and deformation gradient arrays must have shape (n_points, 2, 2)")
    if volumes.shape != (n_points,) or loads.shape != (n_points, 2):
        raise ValueError("volumes and point_loads must be sized by n_points")
    if reference_gradients.shape != (n_points, n_nodes, 2):
        raise ValueError("shape_gradients must have shape (n_points, n_nodes, 2)")
    internal_forces = np.zeros((n_nodes, 2))
    for p in range(n_points):
        determinant = float(np.linalg.det(gradients[p]))
        if determinant <= 0.0:
            raise ValueError("every deformation gradient must have a positive determinant")
        spatial_gradients = reference_gradients[p] @ np.linalg.inv(gradients[p])
        internal_forces += (determinant * volumes[p]) * (spatial_gradients @ stress[p].T)
    return internal_forces, values.T @ loads

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

_SHAPE_VALUES = np.array([[0.30, 0.20, 0.30, 0.20],
                          [0.20, 0.30, 0.20, 0.30]])
_SHAPE_GRADIENTS = np.array([[[-1.0, -1.2], [1.0, -0.8], [-1.0, 1.2], [1.0, 0.8]],
                             [[-1.0, -0.8], [1.0, -1.2], [-1.0, 0.8], [1.0, 1.2]]])
_STRESS = np.array([[[0.051469407641, -0.019953851179],
                     [-0.019953851179, 0.014577031322]],
                    [[-0.032517446129, 0.008914552037],
                     [0.008914552037, -0.005271993844]]])
_GRADIENTS = np.array([[[1.182159787136, 0.035865387354],
                        [-0.207004690418, 0.984255344907]],
                       [[0.964121773395, -0.111403618763],
                        [0.089720145522, 1.043118293701]]])


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal scenario: two loaded points transferring to four nodes ---
        {
            "setup": """import numpy as np
volumes = np.array([0.04, 0.04])
point_loads = np.array([[0.0, -0.0012], [0.0, -0.0012]])
""",
            "call": "[np.round(assemble_nodal_forces(_STRESS, _GRADIENTS, volumes, _SHAPE_VALUES,"
                    " _SHAPE_GRADIENTS, point_loads)[0], 10).tolist(),"
                    " np.round(assemble_nodal_forces(_STRESS, _GRADIENTS, volumes, _SHAPE_VALUES,"
                    " _SHAPE_GRADIENTS, point_loads)[1], 10).tolist()]",
            "gold_call": "[np.round(_oracle_assemble_nodal_forces(_STRESS, _GRADIENTS, volumes, _SHAPE_VALUES,"
                         " _SHAPE_GRADIENTS, point_loads)[0], 10).tolist(),"
                         " np.round(_oracle_assemble_nodal_forces(_STRESS, _GRADIENTS, volumes, _SHAPE_VALUES,"
                         " _SHAPE_GRADIENTS, point_loads)[1], 10).tolist()]",
        },
        # --- Boundary case: undeformed and unstressed points carrying no load ---
        {
            "setup": """import numpy as np
stress = np.zeros((2, 2, 2))
gradients = np.array([np.eye(2), np.eye(2)])
volumes = np.array([0.04, 0.04])
point_loads = np.zeros((2, 2))
""",
            "call": "[np.round(assemble_nodal_forces(stress, gradients, volumes, _SHAPE_VALUES,"
                    " _SHAPE_GRADIENTS, point_loads)[0], 10).tolist(),"
                    " np.round(assemble_nodal_forces(stress, gradients, volumes, _SHAPE_VALUES,"
                    " _SHAPE_GRADIENTS, point_loads)[1], 10).tolist()]",
            "gold_call": "[np.round(_oracle_assemble_nodal_forces(stress, gradients, volumes, _SHAPE_VALUES,"
                         " _SHAPE_GRADIENTS, point_loads)[0], 10).tolist(),"
                         " np.round(_oracle_assemble_nodal_forces(stress, gradients, volumes, _SHAPE_VALUES,"
                         " _SHAPE_GRADIENTS, point_loads)[1], 10).tolist()]",
        },
        # --- Edge case: a gradient array sized for the wrong node count ---
        {
            "setup": """import numpy as np
volumes = np.array([0.04, 0.04])
point_loads = np.zeros((2, 2))
bad_gradients = _SHAPE_GRADIENTS[:, :3, :].copy()
def run_model():
    try:
        assemble_nodal_forces(_STRESS, _GRADIENTS, volumes, _SHAPE_VALUES, bad_gradients, point_loads)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_assemble_nodal_forces(_STRESS, _GRADIENTS, volumes, _SHAPE_VALUES, bad_gradients, point_loads)
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
