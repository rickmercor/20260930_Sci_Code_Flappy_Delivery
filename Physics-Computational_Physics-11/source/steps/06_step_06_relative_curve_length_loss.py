"""
Measure endpoint lengths of curved finite-element histories.

Unit mobility is used. All scalar and vector response and test fields use this continuous spatial quadratic space and are linear in time, while geometry is quadratic in time; all variational pairings are exact physical space-time integrals on the evolving curve. Use the whole-slab dual formulation of surface diffusion whose vector motion minimizes the integrated squared surface gradient for each weak normal action, where that action pairs the vector field's normal component with every scalar test through the evolving curve measure. Its discrete curvature force is the exact dual representation of the negative length variation through this minimum-deformation relation on the full weak-normal action space. The normal H-minus-one norm is dual to the square root of the integrated squared scalar surface gradient on the whole scalar test space, on actions annihilating its kernel.

Returns
-------
return loss
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def relative_curve_length_loss(history: "np.ndarray") -> float:
    """Return the relative endpoint curve-length loss in percent.

    Parameters
    ----------
    history : np.ndarray
        Shape (3,12,2), with coefficients of (1,tau,tau**2) for tau in [0,1].

    Returns
    -------
    float
        One finite relative endpoint length loss in percent.

    Raises
    ------
    ValueError
        If the shape or coefficients are invalid or an endpoint geometry
        is degenerate.

    Notes
    -----
    Elements connect nodes (2e,2e+1,(2e+2)%12) at local (-1,0,1),
    using quadratic rather than polygonal curves. Use initial curved-element
    length as the denominator. A constant path returns zero; an expanding
    path can return a negative loss. Do not assume the history solves a flow.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

from numpy.polynomial.legendre import leggauss

def _oracle_relative_curve_length_loss(history: "np.ndarray") -> float:
    lengths = _cdf_lengths(history)
    answer = float(100*(lengths[0]-lengths[1])/lengths[0])
    if not np.isfinite(answer):
        raise ValueError('curve-length loss is not finite')
    return answer

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
        {'setup': base + 'history[1]=.025*nodes*np.cos(2*theta)[:,None]',
         'call': 'relative_curve_length_loss(history.copy())',
         'gold_call': '_oracle_relative_curve_length_loss(history.copy())'},
        {'setup': base,
         'call': 'relative_curve_length_loss(history.copy())',
         'gold_call': '_oracle_relative_curve_length_loss(history.copy())'},
        {'setup': base + 'history[1]=.02*nodes',
         'call': 'relative_curve_length_loss(history.copy())',
         'gold_call': '_oracle_relative_curve_length_loss(history.copy())'},
        {'setup': base + 'history[1]=-16*nodes; history[2]=64*nodes',
         'call': 'relative_curve_length_loss(history.copy())',
         'gold_call': '_oracle_relative_curve_length_loss(history.copy())'},
        _exception_case(base + 'history[1]=-nodes', 'relative_curve_length_loss', 'history.copy()')]
