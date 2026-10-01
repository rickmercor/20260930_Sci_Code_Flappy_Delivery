"""
Resolve the regular polynomial trajectory of the finite curve model.

Unit mobility is used. All scalar and vector response and test fields use this continuous spatial quadratic space and are linear in time, while geometry is quadratic in time; all variational pairings are exact physical space-time integrals on the evolving curve. Use the whole-slab dual formulation of surface diffusion whose vector motion minimizes the integrated squared surface gradient for each weak normal action, where that action pairs the vector field's normal component with every scalar test through the evolving curve measure. Its discrete curvature force is the exact dual representation of the negative length variation through this minimum-deformation relation on the full weak-normal action space. The normal H-minus-one norm is dual to the square root of the integrated squared scalar surface gradient on the whole scalar test space, on actions annihilating its kernel.

Returns
-------
return history
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def discrete_curve_trajectory(initial_nodes: "np.ndarray", dt: float) -> "np.ndarray":
    """Return the self-consistent curve path on the regular zero-slab branch.

    Parameters
    ----------
    initial_nodes : np.ndarray
        Finite array of shape (12,2) defining a closed counterclockwise curve.
    dt : float
        Finite positive slab length.

    Returns
    -------
    np.ndarray
        Coefficients of shape (3,12,2) in (1,tau,tau**2), where tau=t/dt.

    Raises
    ------
    ValueError
        If inputs are invalid, dt is not positive, the path is outside
        the stated domain, or the selected regular branch fails.

    Notes
    -----
    The curve uses quadratic elements (2e,2e+1,(2e+2)%12) at local
    nodes (-1,0,1), with outward normals. Restrict the trajectory domain
    to paths for which each element tangent has positive projection onto
    its initial endpoint chord direction over the entire element and slab.

    The path belongs to the model stated in the task: its physical
    derivative is its exact-dual surface-diffusion velocity. Select
    the regular continuation from the zero-slab limit, where geometry
    is the initial curve and velocity is its fixed-geometry response.
    Preserve node labels and connectivity.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

from scipy.optimize import root

def _cdf_history_from_velocity(nodes, velocity, dt):
    v = np.reshape(velocity, (2, 12, 2))
    return np.array([nodes, dt*(v[0]-v[1]), dt*v[1]])

def _oracle_discrete_curve_trajectory(initial_nodes: "np.ndarray", dt: float) -> "np.ndarray":
    nodes = _cdf_real_array(initial_nodes, (12, 2), 'initial_nodes')
    dt = _cdf_dt(dt)
    initial_history = np.array([nodes, np.zeros_like(nodes), np.zeros_like(nodes)])
    guess = _oracle_fixed_geometry_response(initial_history, dt).ravel()
    # A static path has exactly zero linear temporal response coefficient.
    # Remove roundoff there before MINPACK chooses relative difference steps.
    guess[24:] = 0.
    # Continue from a short regular slab using the same finite spaces.
    schedule = []
    slab = min(dt, .001)
    while slab < dt:
        schedule.append(slab)
        slab = min(dt, 2*slab)
    schedule.append(dt)
    for slab in schedule:
        def _residual(coefficients):
            path = _cdf_history_from_velocity(nodes, coefficients, slab)
            return coefficients-_oracle_fixed_geometry_response(path, slab).ravel()
        try:
            result = root(_residual, guess, method='hybr', options={'xtol': 2e-11})
            error = np.max(np.abs(_residual(result.x)))
        except (np.linalg.LinAlgError, FloatingPointError) as exc:
            raise ValueError('regular finite trajectory solve failed') from exc
        if not np.all(np.isfinite(result.x)) or error > 2e-8:
            raise ValueError('regular finite trajectory solve did not converge')
        guess = result.x
    return _cdf_history_from_velocity(nodes, guess, dt)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base = '''import numpy as np
theta = np.arange(12)*np.pi/6
radius = 1 + .16*np.cos(2*theta) + .07*np.sin(3*theta) + .04*np.cos(theta)
nodes = np.column_stack((radius*np.cos(theta), radius*np.sin(theta)))
history = np.zeros((3,12,2))
history[0] = nodes
dt = .001
'''
    def _exception_case(setup, function_name, argument_text):
        for check_name, target in (('_check_public', function_name),
                                   ('_check_gold', '_oracle_'+function_name)):
            setup += ('\ndef '+check_name+'():\n'
                      '    try:\n'
                      '        '+target+'('+argument_text+')\n'
                      '    except ValueError:\n'
                      '        return 1\n'
                      '    except Exception:\n'
                      '        return 2\n'
                      '    return 0\n')
        return {'setup': setup, 'call': '_check_public()', 'gold_call': '_check_gold()'}
    return [
        {'setup': base,
         'call': 'discrete_curve_trajectory(nodes.copy(), dt)',
         'gold_call': '_oracle_discrete_curve_trajectory(nodes.copy(), dt)'},
        {'setup': base,
         'call': 'discrete_curve_trajectory(nodes.copy(), .0001)',
         'gold_call': '_oracle_discrete_curve_trajectory(nodes.copy(), .0001)'},
        {'setup': base + 'nodes=nodes+np.array([.2,-.15])',
         'call': 'discrete_curve_trajectory(nodes.copy(), dt)',
         'gold_call': '_oracle_discrete_curve_trajectory(nodes.copy(), dt)'},
        _exception_case(base, 'discrete_curve_trajectory', 'nodes.copy(), -dt')]
