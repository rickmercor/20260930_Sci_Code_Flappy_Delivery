"""
Find a dynamical response while keeping the prescribed geometry fixed.

Unit mobility is used. All scalar and vector response and test fields use this continuous spatial quadratic space and are linear in time, while geometry is quadratic in time; all variational pairings are exact physical space-time integrals on the evolving curve. Use the whole-slab dual formulation of surface diffusion whose vector motion minimizes the integrated squared surface gradient for each weak normal action, where that action pairs the vector field's normal component with every scalar test through the evolving curve measure. Its discrete curvature force is the exact dual representation of the negative length variation through this minimum-deformation relation on the full weak-normal action space. The normal H-minus-one norm is dual to the square root of the integrated squared scalar surface gradient on the whole scalar test space, on actions annihilating its kernel.

Returns
-------
return velocity
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fixed_geometry_response(history: "np.ndarray", dt: float) -> "np.ndarray":
    """Return the velocity selected by the fixed-geometry curve model.

    Parameters
    ----------
    history : np.ndarray
        Shape (3,12,2) in powers (1,tau,tau**2), where tau=t/dt.
    dt : float
        Positive physical slab length.

    Returns
    -------
    np.ndarray
        Vector coefficients of shape (2,12,2) in (1,2*tau-1).

    Raises
    ------
    ValueError
        If inputs are invalid, dt is not positive, the history is outside
        the stated domain, or the response is undefined.

    Notes
    -----
    Use periodic quadratic elements (2e,2e+1,(2e+2)%12) with local
    nodes (-1,0,1), include the physical time measure, and use outward
    normals for counterclockwise node ordering. Each element tangent
    must have positive projection onto its initial endpoint chord
    direction over the entire element and slab.

    Return the velocity selected by the stated whole-slab exact-dual
    surface-diffusion model on this prescribed geometry path. Keep
    the prescribed geometry fixed while evaluating this response.
    The supplied history need not be a flow solution.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_fixed_geometry_response(history: "np.ndarray", dt: float) -> "np.ndarray":
    if np.shape(history) != (3, 12, 2):
        raise ValueError('history must have shape (3,12,2)')
    force = _oracle_normal_length_force(history, dt)
    _, _, scalar, _ = _cdf_forms(history, dt)
    action = -(scalar @ force.ravel()).reshape(2, 12)
    return _oracle_minimum_deformation_lift(history, action, dt)

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
        {'setup': base + 'history[1]=.012*nodes*np.sin(theta)[:,None]',
         'call': 'fixed_geometry_response(history.copy(), dt)',
         'gold_call': '_oracle_fixed_geometry_response(history.copy(), dt)'},
        {'setup': base,
         'call': 'fixed_geometry_response(history.copy(), dt)',
         'gold_call': '_oracle_fixed_geometry_response(history.copy(), dt)'},
        {'setup': base + 'history[0]=1.15*nodes',
         'call': 'fixed_geometry_response(history.copy(), dt)',
         'gold_call': '_oracle_fixed_geometry_response(history.copy(), dt)'},
        _exception_case(base, 'fixed_geometry_response', 'history.copy(), 0')]
