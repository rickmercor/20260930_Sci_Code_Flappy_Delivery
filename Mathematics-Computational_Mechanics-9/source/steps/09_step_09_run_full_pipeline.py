"""
Chain every earlier stage into the full implicit material point analysis of the

loaded design and return the most negative entry of the compliance sensitivity

vector, the largest compliance reduction available per unit increase in the

pseudo-density of a single material point.

The stages compose in the order the mechanics dictates. The lattice of material

points fixes the quadrature of the design domain and carries the pseudo-density

field, and the generalized interpolation basis fixes how that quadrature

communicates with the background grid; both are formed once from the reference

configuration and held fixed, since the whole load is applied in a single

increment. The SIMP rule turns the pseudo-densities into the Lame parameters

that the constitutive law needs, and the Hencky law under plane stress turns the

current deformation gradient into the Cauchy stress that the particle to grid

transfer needs. The nodal residual formed from those stresses vanishes only at

equilibrium, and because the response is geometrically and materially nonlinear

it is driven to zero by Newton-Raphson, whose correction requires the consistent

tangent at the current state; the iteration starts from the undeformed state,

where the left Cauchy-Green tensor is the identity, so the derivative of the

matrix logarithm is evaluated at a repeated eigenvalue on the very first

correction. The converged displacement fixes the compliance, and the tangent at

that converged state fixes the adjoint field, which the implicit function

theorem requires in place of differentiating through the iteration history.

Contracting the adjoint with the design derivative of the residual gives one

sensitivity per material point, and the reported scalar is the minimum of that

vector, in the same force and length units as the applied load and the domain.

Returns
-------
float, the most negative entry of the compliance sensitivity vector as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_full_pipeline(total_load: float = 0.006, penalty: float = 3.0) -> float:
    '''Run the full analysis and return the most negative compliance sensitivity.

    Parameters
    ----------
    total_load : float
        Total force in the negative y direction shared equally by the material
        points on the loaded edge.
    penalty : float
        SIMP exponent q.

    Returns
    -------
    minimum_sensitivity : float
        The most negative entry of the compliance sensitivity vector.

    Raises
    ------
    ValueError
        If `total_load` or `penalty` is not a real number, if `total_load` is
        not positive, or if `penalty` is less than 1.
    '''
    return minimum_sensitivity

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

_DOMAIN = np.array([2.0, 1.0])
_POINT_SPACING = 0.2
_CELL_SIZE = 0.5
_POINT_DOMAIN_LENGTH = 0.2
_YOUNGS_MODULUS = 1.0
_POISSON_RATIO = 0.3
_NEWTON_TOLERANCE = 1e-10
_MAX_NEWTON_ITERATIONS = 50


def _oracle_run_full_pipeline(total_load: float = 0.006, penalty: float = 3.0) -> float:
    """Reference implementation chaining every earlier step."""
    try:
        load = float(total_load)
        exponent = float(penalty)
    except (TypeError, ValueError):
        raise ValueError("total_load and penalty must be real numbers") from None
    if load <= 0.0:
        raise ValueError("total_load must be > 0")
    if exponent < 1.0:
        raise ValueError("penalty must be >= 1")

    points, volumes = _oracle_initialize_material_points(_DOMAIN, _POINT_SPACING)
    n_x = round(float(_DOMAIN[0] / _CELL_SIZE)) + 1
    n_y = round(float(_DOMAIN[1] / _CELL_SIZE)) + 1
    node_x = np.arange(n_x) * _CELL_SIZE
    node_y = np.arange(n_y) * _CELL_SIZE
    nodes = np.stack([np.tile(node_x, n_y), np.repeat(node_y, n_x)], axis=1)
    shape_values, shape_gradients = _oracle_gimp_shape_functions(
        points, nodes, _CELL_SIZE, _POINT_DOMAIN_LENGTH)

    densities = 0.3 + np.abs(points[:, 1] - 0.5)
    lame = _oracle_simp_lame_parameters(densities, _YOUNGS_MODULUS, _POISSON_RATIO, exponent)

    loaded = np.abs(points[:, 0] - points[:, 0].max()) < 1e-12
    point_loads = np.zeros((points.shape[0], 2))
    point_loads[loaded, 1] = -load / int(loaded.sum())
    fixed_nodes = np.abs(nodes[:, 0]) < 1e-12
    constrained = np.repeat(fixed_nodes, 2)

    displacement = np.zeros((nodes.shape[0], 2))
    identity = np.eye(2)[None, :, :]
    for _ in range(_MAX_NEWTON_ITERATIONS):
        gradients = identity + np.einsum('vi,pvj->pij', displacement, shape_gradients)
        cauchy_stress, _jacobian = _oracle_hencky_cauchy_stress(gradients, lame)
        internal_forces, external_forces = _oracle_assemble_nodal_forces(
            cauchy_stress, gradients, volumes, shape_values, shape_gradients, point_loads)
        residual = internal_forces - external_forces
        residual.ravel()[constrained] = 0.0
        if float(np.linalg.norm(residual)) < _NEWTON_TOLERANCE:
            break
        stiffness = _oracle_assemble_tangent_stiffness(
            gradients, lame, volumes, shape_gradients, fixed_nodes)
        correction = np.linalg.solve(stiffness, -residual.ravel())
        displacement = displacement + correction.reshape(nodes.shape[0], 2)

    stiffness = _oracle_assemble_tangent_stiffness(
        gradients, lame, volumes, shape_gradients, fixed_nodes)
    _compliance, adjoint = _oracle_compliance_and_adjoint(
        stiffness, external_forces, displacement, fixed_nodes)
    sensitivity = _oracle_density_sensitivity(
        adjoint, cauchy_stress, gradients, volumes, shape_gradients, densities, exponent)
    return float(sensitivity.min())

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal scenario: the configuration given in the problem ---
        {
            "setup": "total_load = 0.006\npenalty = 3.0\n",
            "call": "round(float(run_full_pipeline(total_load=total_load, penalty=penalty)), 10)",
            "gold_call": "round(float(_oracle_run_full_pipeline(total_load=total_load, penalty=penalty)), 10)",
        },
        # --- Boundary case: no penalization, so the design enters linearly ---
        {
            "setup": "total_load = 0.004\npenalty = 1.0\n",
            "call": "round(float(run_full_pipeline(total_load=total_load, penalty=penalty)), 10)",
            "gold_call": "round(float(_oracle_run_full_pipeline(total_load=total_load, penalty=penalty)), 10)",
        },
        # --- Edge case: a non-positive load is not an admissible configuration ---
        {
            "setup": """def run_model():
    try:
        run_full_pipeline(total_load=0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_run_full_pipeline(total_load=0.0)
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
