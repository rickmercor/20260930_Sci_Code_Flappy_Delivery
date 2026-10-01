"""
Determine the response derivatives in calibration-measurement coordinates.

The calibration map is locally invertible; all input derivatives refer to the same expansion point.

Returns
-------
return out  # out : float ndarray, shape (n+1,n)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def inverse_calibration_jet(calibration_jacobian: np.ndarray, calibration_hessians: np.ndarray, response_gradient: np.ndarray, response_hessian: np.ndarray) -> np.ndarray:
    """Transform the response jet through a locally invertible calibration map.

    Parameters
    ----------
    calibration_jacobian : float ndarray, shape (n,n)
        Nonsingular dF_i/dtheta_j; rows are measured observables.
    calibration_hessians : float ndarray, shape (n,n,n)
        Entry [i,j,k] is d2F_i/(dtheta_j*dtheta_k).
    response_gradient : float ndarray, shape (n,)
        Gradient of the scalar response B with respect to theta.
    response_hessian : float ndarray, shape (n,n)
        Symmetric Hessian of B with respect to theta.

    Returns
    -------
    out : float ndarray, shape (n+1,n)
        Row zero is the gradient of B composed with the local inverse of F;
        the remaining n rows are its Hessian, both in measurement coordinates.
        Units are response per measurement, and response per measurement squared.
    """
    return np.zeros((len(response_gradient) + 1, len(response_gradient)), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_inverse_calibration_jet(calibration_jacobian: np.ndarray, calibration_hessians: np.ndarray, response_gradient: np.ndarray, response_hessian: np.ndarray) -> np.ndarray:
    """Transform the response jet through a locally invertible calibration map.

    Parameters
    ----------
    calibration_jacobian : float ndarray, shape (n,n)
        Nonsingular dF_i/dtheta_j; rows are measured observables.
    calibration_hessians : float ndarray, shape (n,n,n)
        Entry [i,j,k] is d2F_i/(dtheta_j*dtheta_k).
    response_gradient : float ndarray, shape (n,)
        Gradient of the scalar response B with respect to theta.
    response_hessian : float ndarray, shape (n,n)
        Symmetric Hessian of B with respect to theta.

    Returns
    -------
    out : float ndarray, shape (n+1,n)
        Row zero is the gradient of B composed with the local inverse of F;
        the remaining n rows are its Hessian, both in measurement coordinates.
        Units are response per measurement, and response per measurement squared.
    """
    (j, f2, g, b2) = map(np.asarray, (calibration_jacobian, calibration_hessians, response_gradient, response_hessian))
    n = g.size
    if j.shape != (n, n) or f2.shape != (n, n, n) or b2.shape != (n, n):
        raise ValueError('Incompatible calibration and response jets')
    a = np.linalg.solve(j.T, g)
    reduced = b2 - np.einsum('i,ijk->jk', a, f2)
    right = np.linalg.solve(j.T, reduced.T).T
    hessian = np.linalg.solve(j.T, right)
    return np.vstack((a, (hessian + hessian.T) / 2))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'linear_nonsymmetric_coordinates',
      'setup': 'import numpy as np\n'
               'j=np.array([[1.2,.3,-.2],[.1,.9,.4],[-.3,.2,1.1]])\n'
               'f2=np.array([[[.7,.1,-.2],[.1,-.3,.4],[-.2,.4,.2]], [[-.5,.2,.3],[.2,.8,-.1],[.3,-.1,.4]], '
               '[[.6,-.4,.2],[-.4,.1,.5],[.2,.5,-.3]]])\n'
               'g=np.array([1.2,-.7,.4]);b2=np.array([[.4,.3,-.2],[.3,-.6,.7],[-.2,.7,.8]])\n'
               'f2*=0',
      'call': 'inverse_calibration_jet(j,f2,g,b2)',
      'gold_call': '_oracle_inverse_calibration_jet(j,f2,g,b2)'},
     {'name': 'nonlinear_mixed_inverse',
      'setup': 'import numpy as np\n'
               'j=np.array([[1.2,.3,-.2],[.1,.9,.4],[-.3,.2,1.1]])\n'
               'f2=np.array([[[.7,.1,-.2],[.1,-.3,.4],[-.2,.4,.2]], [[-.5,.2,.3],[.2,.8,-.1],[.3,-.1,.4]], '
               '[[.6,-.4,.2],[-.4,.1,.5],[.2,.5,-.3]]])\n'
               'g=np.array([1.2,-.7,.4]);b2=np.array([[.4,.3,-.2],[.3,-.6,.7],[-.2,.7,.8]])\n',
      'call': 'inverse_calibration_jet(j,f2,g,b2)',
      'gold_call': '_oracle_inverse_calibration_jet(j,f2,g,b2)'},
     {'name': 'exact_coordinate_cancellation',
      'setup': 'import numpy as np\n'
               'j=np.array([[1.2,.3,-.2],[.1,.9,.4],[-.3,.2,1.1]])\n'
               'f2=np.array([[[.7,.1,-.2],[.1,-.3,.4],[-.2,.4,.2]], [[-.5,.2,.3],[.2,.8,-.1],[.3,-.1,.4]], '
               '[[.6,-.4,.2],[-.4,.1,.5],[.2,.5,-.3]]])\n'
               'g=np.array([1.2,-.7,.4]);b2=np.array([[.4,.3,-.2],[.3,-.6,.7],[-.2,.7,.8]])\n'
               'g=j[1].copy();b2=f2[1].copy()',
      'call': 'inverse_calibration_jet(j,f2,g,b2)',
      'gold_call': '_oracle_inverse_calibration_jet(j,f2,g,b2)'},
     {'name': 'stationary_response',
      'setup': 'import numpy as np\n'
               'j=np.array([[1.2,.3,-.2],[.1,.9,.4],[-.3,.2,1.1]])\n'
               'f2=np.array([[[.7,.1,-.2],[.1,-.3,.4],[-.2,.4,.2]], [[-.5,.2,.3],[.2,.8,-.1],[.3,-.1,.4]], '
               '[[.6,-.4,.2],[-.4,.1,.5],[.2,.5,-.3]]])\n'
               'g=np.array([1.2,-.7,.4]);b2=np.array([[.4,.3,-.2],[.3,-.6,.7],[-.2,.7,.8]])\n'
               'g*=0',
      'call': 'inverse_calibration_jet(j,f2,g,b2)',
      'gold_call': '_oracle_inverse_calibration_jet(j,f2,g,b2)'},
     {'name': 'affine_response_curved_calibration',
      'setup': 'import numpy as np\n'
               'j=np.array([[1.2,.3,-.2],[.1,.9,.4],[-.3,.2,1.1]])\n'
               'f2=np.array([[[.7,.1,-.2],[.1,-.3,.4],[-.2,.4,.2]], [[-.5,.2,.3],[.2,.8,-.1],[.3,-.1,.4]], '
               '[[.6,-.4,.2],[-.4,.1,.5],[.2,.5,-.3]]])\n'
               'g=np.array([1.2,-.7,.4]);b2=np.array([[.4,.3,-.2],[.3,-.6,.7],[-.2,.7,.8]])\n'
               'b2*=0',
      'call': 'inverse_calibration_jet(j,f2,g,b2)',
      'gold_call': '_oracle_inverse_calibration_jet(j,f2,g,b2)'},
     {'name': 'ill_conditioned_nonsingular',
      'setup': 'import numpy as np\n'
               'j=np.array([[1.2,.3,-.2],[.1,.9,.4],[-.3,.2,1.1]])\n'
               'f2=np.array([[[.7,.1,-.2],[.1,-.3,.4],[-.2,.4,.2]], [[-.5,.2,.3],[.2,.8,-.1],[.3,-.1,.4]], '
               '[[.6,-.4,.2],[-.4,.1,.5],[.2,.5,-.3]]])\n'
               'g=np.array([1.2,-.7,.4]);b2=np.array([[.4,.3,-.2],[.3,-.6,.7],[-.2,.7,.8]])\n'
               'j=np.diag([.015,2.,.7])',
      'call': 'inverse_calibration_jet(j,f2,g,b2)',
      'gold_call': '_oracle_inverse_calibration_jet(j,f2,g,b2)'},
     {'name': 'permuted_measurement_rows',
      'setup': 'import numpy as np\n'
               'j=np.array([[1.2,.3,-.2],[.1,.9,.4],[-.3,.2,1.1]])\n'
               'f2=np.array([[[.7,.1,-.2],[.1,-.3,.4],[-.2,.4,.2]], [[-.5,.2,.3],[.2,.8,-.1],[.3,-.1,.4]], '
               '[[.6,-.4,.2],[-.4,.1,.5],[.2,.5,-.3]]])\n'
               'g=np.array([1.2,-.7,.4]);b2=np.array([[.4,.3,-.2],[.3,-.6,.7],[-.2,.7,.8]])\n'
               'perm=[2,0,1];j=j[perm];f2=f2[perm]',
      'call': 'inverse_calibration_jet(j,f2,g,b2)',
      'gold_call': '_oracle_inverse_calibration_jet(j,f2,g,b2)'},
     {'name': 'one_dimensional_inverse',
      'setup': 'import numpy as np\n'
               'j=np.array([[1.2,.3,-.2],[.1,.9,.4],[-.3,.2,1.1]])\n'
               'f2=np.array([[[.7,.1,-.2],[.1,-.3,.4],[-.2,.4,.2]], [[-.5,.2,.3],[.2,.8,-.1],[.3,-.1,.4]], '
               '[[.6,-.4,.2],[-.4,.1,.5],[.2,.5,-.3]]])\n'
               'g=np.array([1.2,-.7,.4]);b2=np.array([[.4,.3,-.2],[.3,-.6,.7],[-.2,.7,.8]])\n'
               'j=np.array([[2.]]);f2=np.array([[[3.]]]);g=np.array([4.]);b2=np.array([[5.]])',
      'call': 'inverse_calibration_jet(j,f2,g,b2)',
      'gold_call': '_oracle_inverse_calibration_jet(j,f2,g,b2)'}]
