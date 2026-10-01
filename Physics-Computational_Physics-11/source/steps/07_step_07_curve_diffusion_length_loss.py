"""
FINAL ORCHESTRATOR: the finite curve-diffusion length-loss observable.

Unit mobility is used. All scalar and vector response and test fields use this continuous spatial quadratic space and are linear in time, while geometry is quadratic in time; all variational pairings are exact physical space-time integrals on the evolving curve. Use the whole-slab dual formulation of surface diffusion whose vector motion minimizes the integrated squared surface gradient for each weak normal action, where that action pairs the vector field's normal component with every scalar test through the evolving curve measure. Its discrete curvature force is the exact dual representation of the negative length variation through this minimum-deformation relation on the full weak-normal action space. The normal H-minus-one norm is dual to the square root of the integrated squared scalar surface gradient on the whole scalar test space, on actions annihilating its kernel.

Returns
-------
return loss
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def curve_diffusion_length_loss(initial_nodes: "np.ndarray", dt: float) -> float:
    """Compute the specified finite curve model's length loss in percent.

    Parameters
    ----------
    initial_nodes : np.ndarray
        Shape (12,2), defining six periodic quadratic elements.
    dt : float
        Finite positive slab length.

    Returns
    -------
    float
        One finite endpoint curve-length loss in percent.

    Raises
    ------
    ValueError
        If inputs are invalid or a finite flow is not well defined on
        the selected branch.

    Notes
    -----
    Elements connect (2e,2e+1,(2e+2)%12) at local coordinates (-1,0,1).
    Use the stated degree-two geometry, degree-one test fields,
    exact-integral dual surface-diffusion model, and regular branch continued
    from the zero-slab limit. Measure endpoint curved lengths with initial
    length as denominator. Use outward normals for counterclockwise ordering
    and require every element tangent to have positive projection on its
    initial endpoint chord direction throughout the slab. Use the earlier
    subproblem functions for their stated operations.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_curve_diffusion_length_loss(initial_nodes: "np.ndarray", dt: float) -> float:
    history = _oracle_discrete_curve_trajectory(initial_nodes, dt)
    response = _oracle_fixed_geometry_response(history, dt)
    force = _oracle_normal_length_force(history, dt)
    _, normal, _, _ = _cdf_forms(history, dt)
    action = (normal @ response.ravel()).reshape(2, 12)
    lifted = _oracle_minimum_deformation_lift(history, action, dt)
    dissipation = _oracle_normal_diffusion_cost(history, lifted, dt)
    answer = _oracle_relative_curve_length_loss(history)
    lengths = _cdf_lengths(history)
    physical_velocity = np.array([(history[1]+history[2])/dt, history[2]/dt])
    scale = max(1., float(np.max(np.abs(response))))
    if np.max(np.abs(physical_velocity-response)) > 3e-8*scale:
        raise ValueError('trajectory and response disagree')
    if np.max(np.abs(lifted-response)) > 3e-9*scale:
        raise ValueError('response is not its minimum-deformation representative')
    work = float(force.ravel() @ action.ravel())
    if abs(work+dissipation) > 2e-10*max(1., dissipation):
        raise ValueError('diffusion response and length work disagree')
    if abs(lengths[0]-lengths[1]-dissipation) > 2e-10*max(1., lengths[0]):
        raise ValueError('geometric and dissipative length losses disagree')
    return float(answer)

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
         'call': 'curve_diffusion_length_loss(nodes.copy(), dt)',
         'gold_call': '_oracle_curve_diffusion_length_loss(nodes.copy(), dt)'},
        {'setup': base,
         'call': 'curve_diffusion_length_loss(nodes.copy(), .0001)',
         'gold_call': '_oracle_curve_diffusion_length_loss(nodes.copy(), .0001)'},
        {'setup': base + 'nodes=nodes+np.array([-.25,.3])',
         'call': 'curve_diffusion_length_loss(nodes.copy(), dt)',
         'gold_call': '_oracle_curve_diffusion_length_loss(nodes.copy(), dt)'},
        _exception_case(base, 'curve_diffusion_length_loss', 'nodes.copy(), 0')]
