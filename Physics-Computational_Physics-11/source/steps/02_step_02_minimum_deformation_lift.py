"""
Choose a vector representative for prescribed weak normal motion.

Unit mobility is used. All scalar and vector response and test fields use this continuous spatial quadratic space and are linear in time, while geometry is quadratic in time; all variational pairings are exact physical space-time integrals on the evolving curve. Use the whole-slab dual formulation of surface diffusion whose vector motion minimizes the integrated squared surface gradient for each weak normal action, where that action pairs the vector field's normal component with every scalar test through the evolving curve measure. Its discrete curvature force is the exact dual representation of the negative length variation through this minimum-deformation relation on the full weak-normal action space. The normal H-minus-one norm is dual to the square root of the integrated squared scalar surface gradient on the whole scalar test space, on actions annihilating its kernel.

Returns
-------
return velocity
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def minimum_deformation_lift(history: "np.ndarray", normal_action: "np.ndarray", dt: float) -> "np.ndarray":
    """Return the model's minimum-H1 representative of a normal action.

    Parameters
    ----------
    history : np.ndarray
        Shape (3,12,2) in temporal powers (1,tau,tau**2).
    normal_action : np.ndarray
        Shape (2,12), containing covector entries on the scalar nodal
        basis times (1,2*tau-1), including physical time and curve measures.
    dt : float
        Positive physical slab length, where tau=t/dt.

    Returns
    -------
    np.ndarray
        Vector coefficients of shape (2,12,2) in (1,2*tau-1).

    Raises
    ------
    ValueError
        If data are invalid, dt is not positive, the history is outside
        the stated domain, or the constrained minimizer does not exist uniquely.

    Notes
    -----
    Elements connect (2e,2e+1,(2e+2)%12) at local (-1,0,1) with
    quadratic interpolation. Use outward normals for counterclockwise
    node ordering. Each element tangent must have positive projection
    onto its initial endpoint chord direction over the entire element
    and slab.

    The returned coefficients minimize the physical space-time integral of the squared
    surface gradient of the full vector field while preserving the
    supplied weak normal action.
    Hold the geometry fixed. Area-changing actions are allowed here;
    do not impose the diffusion-compatible restriction.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_minimum_deformation_lift(history: "np.ndarray", normal_action: "np.ndarray", dt: float) -> "np.ndarray":
    if np.shape(normal_action) != (2, 12):
        raise ValueError('normal_action must have shape (2,12)')
    action = _cdf_real_array(normal_action, (2, 12), 'normal_action')
    vector, normal, _, _ = _cdf_forms(history, dt)
    solution = _cdf_saddle(vector, normal, np.r_[np.zeros(48), action.ravel()])
    return solution[:48].reshape(2, 12, 2)

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
        {'setup': base + 'action=np.zeros((2,12)); action[0]=.001*np.cos(theta); action[1]=.0003*np.sin(2*theta)',
         'call': 'minimum_deformation_lift(history.copy(), action.copy(), dt)',
         'gold_call': '_oracle_minimum_deformation_lift(history.copy(), action.copy(), dt)'},
        {'setup': base + 'action=np.zeros((2,12))',
         'call': 'minimum_deformation_lift(history.copy(), action.copy(), dt)',
         'gold_call': '_oracle_minimum_deformation_lift(history.copy(), action.copy(), dt)'},
        {'setup': base + 'action=np.zeros((2,12)); action[0]=dt/12',
         'call': 'minimum_deformation_lift(history.copy(), action.copy(), dt)',
         'gold_call': '_oracle_minimum_deformation_lift(history.copy(), action.copy(), dt)'},
        _exception_case(base + 'action=np.zeros((12,2))',
                        'minimum_deformation_lift', 'history.copy(), action.copy(), dt')]
