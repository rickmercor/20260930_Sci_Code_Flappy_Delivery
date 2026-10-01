"""
Determine the uncertainty certificate for the prescribed quadratic response.

The response is the quadratic Taylor polynomial in the declared Gaussian measurement perturbation.

Returns
-------
return out  # out : float ndarray, shape (4,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def gaussian_quadratic_bound(value: float, gradient: np.ndarray, hessian: np.ndarray, covariance: np.ndarray, quantile: float) -> np.ndarray:
    """Evaluate Gaussian moments and a lower normal-equivalent bound of a quadratic.

    Parameters
    ----------
    value : float
        Response at zero perturbation, measured in percentage points.
    gradient : float ndarray, shape (n,)
        Linear coefficients with respect to calibration measurements.
    hessian : float ndarray, shape (n,n)
        Symmetric second derivatives with respect to those measurements.
    covariance : float ndarray, shape (n,n)
        Symmetric positive-semidefinite covariance of zero-mean Gaussian noise.
    quantile : float
        Nonnegative normal-equivalent standard-deviation multiplier.

    Returns
    -------
    out : float ndarray, shape (4,)
        Ordered [mean, standard deviation, lower bound, curvature bias].
        All four entries are in percentage points and refer to the
        quadratic Taylor polynomial defined in the problem.
    """
    return np.zeros((4,), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_gaussian_quadratic_bound(value: float, gradient: np.ndarray, hessian: np.ndarray, covariance: np.ndarray, quantile: float) -> np.ndarray:
    """Evaluate Gaussian moments and a lower normal-equivalent bound of a quadratic.

    Parameters
    ----------
    value : float
        Response at zero perturbation, measured in percentage points.
    gradient : float ndarray, shape (n,)
        Linear coefficients with respect to calibration measurements.
    hessian : float ndarray, shape (n,n)
        Symmetric second derivatives with respect to those measurements.
    covariance : float ndarray, shape (n,n)
        Symmetric positive-semidefinite covariance of zero-mean Gaussian noise.
    quantile : float
        Nonnegative normal-equivalent standard-deviation multiplier.

    Returns
    -------
    out : float ndarray, shape (4,)
        Ordered [mean, standard deviation, lower bound, curvature bias].
        All four entries are in percentage points. Moments are those of the
        complete quadratic Taylor polynomial, including its quadratic variance.
    """
    (g, h, s) = map(np.asarray, (gradient, hessian, covariance))
    n = g.size
    if h.shape != (n, n) or s.shape != (n, n) or quantile < 0:
        raise ValueError('Incompatible quadratic uncertainty inputs')
    bias = 0.5 * np.trace(h @ s)
    variance = g @ s @ g + 0.5 * np.trace(h @ s @ h @ s)
    sd = np.sqrt(max(0.0, float(variance)))
    mean = value + bias
    return np.array([mean, sd, mean - quantile * sd, bias])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'zero_uncertainty',
      'setup': 'import numpy as np\n'
               'b=1.7;g=np.array([.8,-.5,.3]);h=np.array([[.6,.4,-.2],[.4,-.3,.5],[-.2,.5,.9]])\n'
               's=np.array([[.4,.08,-.03],[.08,.3,.1],[-.03,.1,.2]]);z=1.645\n'
               's*=0',
      'call': 'gaussian_quadratic_bound(b,g,h,s,z)',
      'gold_call': '_oracle_gaussian_quadratic_bound(b,g,h,s,z)'},
     {'name': 'linear_gaussian_response',
      'setup': 'import numpy as np\n'
               'b=1.7;g=np.array([.8,-.5,.3]);h=np.array([[.6,.4,-.2],[.4,-.3,.5],[-.2,.5,.9]])\n'
               's=np.array([[.4,.08,-.03],[.08,.3,.1],[-.03,.1,.2]]);z=1.645\n'
               'h*=0',
      'call': 'gaussian_quadratic_bound(b,g,h,s,z)',
      'gold_call': '_oracle_gaussian_quadratic_bound(b,g,h,s,z)'},
     {'name': 'pure_quadratic_response',
      'setup': 'import numpy as np\n'
               'b=1.7;g=np.array([.8,-.5,.3]);h=np.array([[.6,.4,-.2],[.4,-.3,.5],[-.2,.5,.9]])\n'
               's=np.array([[.4,.08,-.03],[.08,.3,.1],[-.03,.1,.2]]);z=1.645\n'
               'g*=0',
      'call': 'gaussian_quadratic_bound(b,g,h,s,z)',
      'gold_call': '_oracle_gaussian_quadratic_bound(b,g,h,s,z)'},
     {'name': 'correlated_indefinite_curvature',
      'setup': 'import numpy as np\n'
               'b=1.7;g=np.array([.8,-.5,.3]);h=np.array([[.6,.4,-.2],[.4,-.3,.5],[-.2,.5,.9]])\n'
               's=np.array([[.4,.08,-.03],[.08,.3,.1],[-.03,.1,.2]]);z=1.645\n',
      'call': 'gaussian_quadratic_bound(b,g,h,s,z)',
      'gold_call': '_oracle_gaussian_quadratic_bound(b,g,h,s,z)'},
     {'name': 'rank_one_covariance',
      'setup': 'import numpy as np\n'
               'b=1.7;g=np.array([.8,-.5,.3]);h=np.array([[.6,.4,-.2],[.4,-.3,.5],[-.2,.5,.9]])\n'
               's=np.array([[.4,.08,-.03],[.08,.3,.1],[-.03,.1,.2]]);z=1.645\n'
               'v=np.array([.2,-.3,.4]);s=np.outer(v,v)',
      'call': 'gaussian_quadratic_bound(b,g,h,s,z)',
      'gold_call': '_oracle_gaussian_quadratic_bound(b,g,h,s,z)'},
     {'name': 'opposite_curvature_trace_cancellation',
      'setup': 'import numpy as np\n'
               'b=1.7;g=np.array([.8,-.5,.3]);h=np.array([[.6,.4,-.2],[.4,-.3,.5],[-.2,.5,.9]])\n'
               's=np.array([[.4,.08,-.03],[.08,.3,.1],[-.03,.1,.2]]);z=1.645\n'
               'g*=0;s=np.eye(3)*.2;h=np.diag([1.,-1.,0.])',
      'call': 'gaussian_quadratic_bound(b,g,h,s,z)',
      'gold_call': '_oracle_gaussian_quadratic_bound(b,g,h,s,z)'},
     {'name': 'off_diagonal_quadratic_variance',
      'setup': 'import numpy as np\n'
               'b=1.7;g=np.array([.8,-.5,.3]);h=np.array([[.6,.4,-.2],[.4,-.3,.5],[-.2,.5,.9]])\n'
               's=np.array([[.4,.08,-.03],[.08,.3,.1],[-.03,.1,.2]]);z=1.645\n'
               'g*=0;s=np.diag([.2,.3,.4]);h=np.array([[0.,.7,0.],[.7,0.,-.4],[0.,-.4,0.]])',
      'call': 'gaussian_quadratic_bound(b,g,h,s,z)',
      'gold_call': '_oracle_gaussian_quadratic_bound(b,g,h,s,z)'},
     {'name': 'mean_endpoint',
      'setup': 'import numpy as np\n'
               'b=1.7;g=np.array([.8,-.5,.3]);h=np.array([[.6,.4,-.2],[.4,-.3,.5],[-.2,.5,.9]])\n'
               's=np.array([[.4,.08,-.03],[.08,.3,.1],[-.03,.1,.2]]);z=1.645\n'
               'z=0',
      'call': 'gaussian_quadratic_bound(b,g,h,s,z)',
      'gold_call': '_oracle_gaussian_quadratic_bound(b,g,h,s,z)'}]
