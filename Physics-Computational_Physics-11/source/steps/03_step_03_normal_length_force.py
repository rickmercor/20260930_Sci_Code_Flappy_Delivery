"""
Express geometric length work in weak normal coordinates.

Unit mobility is used. All scalar and vector response and test fields use this continuous spatial quadratic space and are linear in time, while geometry is quadratic in time; all variational pairings are exact physical space-time integrals on the evolving curve. Use the whole-slab dual formulation of surface diffusion whose vector motion minimizes the integrated squared surface gradient for each weak normal action, where that action pairs the vector field's normal component with every scalar test through the evolving curve measure. Its discrete curvature force is the exact dual representation of the negative length variation through this minimum-deformation relation on the full weak-normal action space. The normal H-minus-one norm is dual to the square root of the integrated squared scalar surface gradient on the whole scalar test space, on actions annihilating its kernel.

Returns
-------
return force
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def normal_length_force(history: "np.ndarray", dt: float) -> "np.ndarray":
    """Return the scalar normal representation of the model's length work.

    Parameters
    ----------
    history : np.ndarray
        Shape (3,12,2) in powers (1,tau,tau**2), where tau=t/dt.
    dt : float
        Positive physical slab length.

    Returns
    -------
    np.ndarray
        Scalar coefficients of shape (2,12) in (1,2*tau-1).

    Raises
    ------
    ValueError
        If data are invalid, dt is not positive, the history is outside
        the stated domain, or the force representation is not unique.

    Notes
    -----
    Use six periodic quadratic elements with connectivity
    (2e,2e+1,(2e+2)%12) and local nodes (-1,0,1), and outward normals
    for counterclockwise node ordering. Each element tangent must have
    positive projection onto its initial endpoint chord direction over
    the entire element and slab.

    The field represents length work on the full weak-normal action
    space through its minimum-deformation representatives, including
    area-changing actions, in the stated exact-dual model.
    Retain the spatially constant components. The sign is increasing
    length work. Geometry is held fixed while this force is evaluated.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_normal_length_force(history: "np.ndarray", dt: float) -> "np.ndarray":
    if np.shape(history) != (3, 12, 2):
        raise ValueError('history must have shape (3,12,2)')
    vector, normal, _, work = _cdf_forms(history, dt)
    auxiliary = _cdf_saddle(vector, normal, np.r_[-work, np.zeros(24)])
    return -auxiliary[48:].reshape(2, 12)

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
        {'setup': base + 'history[1]=.015*nodes*np.cos(theta)[:,None]',
         'call': 'normal_length_force(history.copy(), dt)',
         'gold_call': '_oracle_normal_length_force(history.copy(), dt)'},
        {'setup': base,
         'call': 'normal_length_force(history.copy(), dt)',
         'gold_call': '_oracle_normal_length_force(history.copy(), dt)'},
        {'setup': base + 'history[0]=1.2*nodes+np.array([.2,-.1])',
         'call': 'normal_length_force(history.copy(), dt)',
         'gold_call': '_oracle_normal_length_force(history.copy(), dt)'},
        _exception_case(base + 'history[:]=0', 'normal_length_force', 'history.copy(), dt')]
