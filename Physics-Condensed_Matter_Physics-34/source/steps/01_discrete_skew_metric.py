"""
Eq. (10): finite discrete beta=1 skew measure in supplied node order.

Eq. (10): finite discrete beta=1 skew measure in supplied node order.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def discrete_skew_metric(nodes, weights):
    """Construct the discrete beta=1 polynomial skew metric.

    Parameters
    ----------
    nodes : real array, (m,)
        Distinct dimensionless abscissae, m>=1, in any supplied order.
    weights : real array, (m,)
        Nonnegative dimensionless masses. Zero masses are permitted here.

    Returns
    -------
    float64 array, (m,m)
        B such that f.T@B@g is one half of the sum of
        f(x_i)g(x_j)sign(x_j-x_i)w_i*w_j over all ordered pairs.
        The diagonal sign is zero. Supplied node order is retained.

    Raises
    ------
    ValueError
        For nonfinite/nonreal inputs, wrong shapes, repeated nodes,
        an empty domain, or negative weights.
    """
    return np.zeros((len(nodes), len(nodes)))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

def _real(a, ndim):
    a=np.asarray(a)
    if a.ndim!=ndim or not np.isfinite(a).all() or np.any(a.imag):
        raise ValueError('Expected a finite real array of the declared dimension')
    return a.real.astype(float)

def _j(n):
    return np.kron(np.eye(n),np.array([[0.,1.],[-1.,0.]]))

def _oracle_discrete_skew_metric(nodes, weights):
    """Construct the discrete beta=1 polynomial skew metric.

    Parameters
    ----------
    nodes : real array, (m,)
        Distinct dimensionless abscissae, m>=1, in any supplied order.
    weights : real array, (m,)
        Nonnegative dimensionless masses. Zero masses are permitted here.

    Returns
    -------
    float64 array, (m,m)
        B such that f.T@B@g is one half of the sum of
        f(x_i)g(x_j)sign(x_j-x_i)w_i*w_j over all ordered pairs.
        The diagonal sign is zero. Supplied node order is retained.

    Raises
    ------
    ValueError
        For nonfinite/nonreal inputs, wrong shapes, repeated nodes,
        an empty domain, or negative weights.
    """
    x=_real(nodes,1); w=_real(weights,1)
    if len(x)==0 or w.shape!=x.shape or len(np.unique(x))!=len(x) or np.any(w<0):
        raise ValueError('Invalid discrete measure')
    return .5*np.sign(x[None,:]-x[:,None])*w[:,None]*w[None,:]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'two-node orientation',
      'setup': 'import numpy as np\n'
               'arg0=np.array([0.0, 2.0],dtype=float).reshape((2,))\n'
               'arg1=np.array([0.5, 3.0],dtype=float).reshape((2,))',
      'call': 'discrete_skew_metric(arg0,arg1)',
      'gold_call': '_oracle_discrete_skew_metric(arg0,arg1)'},
     {'name': 'singleton diagonal sign',
      'setup': 'import numpy as np\n'
               'arg0=np.array([1.0],dtype=float).reshape((1,))\n'
               'arg1=np.array([2.0],dtype=float).reshape((1,))',
      'call': 'discrete_skew_metric(arg0,arg1)',
      'gold_call': '_oracle_discrete_skew_metric(arg0,arg1)'},
     {'name': 'nonuniform abscissae',
      'setup': 'import numpy as np\n'
               'arg0=np.array([-2.2, -1.6, -0.85, -0.3, 0.15, 0.8, 1.5, 2.35],dtype=float).reshape((8,))\n'
               'arg1=np.array([0.2240244682127417, 0.42827061721265963, 0.7474222443980617, 0.9403529457394286, '
               '1.013971704553439, 0.9455391358903963, 0.6924633268086434, '
               '0.3412550957478586],dtype=float).reshape((8,))',
      'call': 'discrete_skew_metric(arg0,arg1)',
      'gold_call': '_oracle_discrete_skew_metric(arg0,arg1)'},
     {'name': 'zero support mass',
      'setup': 'import numpy as np\n'
               'arg0=np.array([-2.2, -1.6, -0.85, -0.3, 0.15, 0.8, 1.5, 2.35],dtype=float).reshape((8,))\n'
               'arg1=np.array([0.0, 0.42827061721265963, 0.7474222443980617, 0.9403529457394286, '
               '1.013971704553439, 0.9455391358903963, 0.6924633268086434, '
               '0.3412550957478586],dtype=float).reshape((8,))',
      'call': 'discrete_skew_metric(arg0,arg1)',
      'gold_call': '_oracle_discrete_skew_metric(arg0,arg1)'},
     {'name': 'unsorted labels',
      'setup': 'import numpy as np\n'
               'arg0=np.array([-0.3, -2.2, 1.5, -1.6, 2.35, -0.85, 0.8, 0.15],dtype=float).reshape((8,))\n'
               'arg1=np.array([0.9403529457394286, 0.2240244682127417, 0.6924633268086434, 0.42827061721265963, '
               '0.3412550957478586, 0.7474222443980617, 0.9455391358903963, '
               '1.013971704553439],dtype=float).reshape((8,))',
      'call': 'discrete_skew_metric(arg0,arg1)',
      'gold_call': '_oracle_discrete_skew_metric(arg0,arg1)'},
     {'name': 'reflection reverses skew sign',
      'setup': 'import numpy as np\n'
               'arg0=np.array([2.2, 1.6, 0.85, 0.3, -0.15, -0.8, -1.5, -2.35],dtype=float).reshape((8,))\n'
               'arg1=np.array([0.2240244682127417, 0.42827061721265963, 0.7474222443980617, 0.9403529457394286, '
               '1.013971704553439, 0.9455391358903963, 0.6924633268086434, '
               '0.3412550957478586],dtype=float).reshape((8,))',
      'call': 'discrete_skew_metric(arg0,arg1)',
      'gold_call': '_oracle_discrete_skew_metric(arg0,arg1)'},
     {'name': 'quadratic common-weight scaling',
      'setup': 'import numpy as np\n'
               'arg0=np.array([-2.2, -1.6, -0.85, -0.3, 0.15, 0.8, 1.5, 2.35],dtype=float).reshape((8,))\n'
               'arg1=np.array([0.6048660641744026, 1.156330666474181, 2.0180400598747665, 2.5389529534964574, '
               '2.737723602294286, 2.55295566690407, 1.8696509823833374, '
               '0.9213887585192183],dtype=float).reshape((8,))',
      'call': 'discrete_skew_metric(arg0,arg1)',
      'gold_call': '_oracle_discrete_skew_metric(arg0,arg1)'},
     {'name': 'separated mass scales',
      'setup': 'import numpy as np\n'
               'arg0=np.array([-2.2, -1.6, -0.85, -0.3, 0.15, 0.8, 1.5, 2.35],dtype=float).reshape((8,))\n'
               'arg1=np.array([1e-05, 0.001, 0.1, 1.0, 2.0, 0.03, 0.0001, 1e-06],dtype=float).reshape((8,))',
      'call': 'discrete_skew_metric(arg0,arg1)',
      'gold_call': '_oracle_discrete_skew_metric(arg0,arg1)'}]
