"""
Calibrate immigration while preserving the expected total number of jumps.



The activation process starts at q0. Arrivals have intensity lambda+alpha*Q

and increase Q by one; expirations have intensity beta*Q and decrease Q by

one. Choose the constant baseline lambda(alpha) so E[N(T)-N(0)]=target,

where N counts arrivals and alpha is the clustering rate. Return lambda

and its first two ordinary derivatives with respect to alpha. Hold T,

target, beta and q0 fixed. Derive the expectation from these birth and

death dynamics, and evaluate the derivatives analytically.

The first activation moment determines expected arrival count. The baseline rate changes with clustering when this count is held fixed.

Returns
-------
return np.zeros(3)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def matched_arrival_rate(T,target,alpha,beta,q0):
    """Return float shape (3,): lambda, lambda prime, lambda double-prime.

    T and target are positive finite scalars; beta > alpha > 0. q0 is a
    nonnegative integer. The calibrated baseline must be strictly positive.
    Raise ValueError for nonfinite inputs, invalid ranges or infeasible target.
    """
    return np.zeros(3)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_matched_arrival_rate(T,target,alpha,beta,q0):
    np=__import__('numpy')
    if not np.all(np.isfinite([T,target,alpha,beta])) or min(T,target)<=0 or not beta>alpha>0:
        raise ValueError('model inputs')
    if isinstance(q0,(bool,np.bool_)) or not np.isscalar(q0) or not np.isfinite(q0) or int(q0)!=q0 or q0<0:
        raise ValueError('q0')
    def c(x):return np.array([x,0.,0.])
    def mul(a,b):return np.array([a[0]*b[0],a[1]*b[0]+a[0]*b[1],a[2]*b[0]+2*a[1]*b[1]+a[0]*b[2]])
    def div(a,b):
        inverse=np.array([1/b[0],-b[1]/b[0]**2,2*b[1]**2/b[0]**3-b[2]/b[0]**2])
        return mul(a,inverse)
    def exp(a):
        y=np.exp(a[0]);return np.array([y,y*a[1],y*(a[2]+a[1]**2)])
    al=np.array([alpha,1.,0.]);delta=c(beta)-al
    survival_integral=div(c(1)-exp(-T*delta),delta)
    immigration_count=c(T)+div(mul(al,c(T)-survival_integral),delta)
    inherited_count=q0*mul(al,survival_integral)
    result=div(c(target)-inherited_count,immigration_count)
    if result[0]<=0:raise ValueError('target implies a nonpositive baseline')
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return explicit dictionaries for Studio differential-case preflight."""
    return [{'setup': 'args=(2.5, 2.4, 0.35, 0.9, 2)',
      'call': 'matched_arrival_rate(*args).tolist()',
      'gold_call': '_oracle_matched_arrival_rate(*args).tolist()'},
     {'setup': 'args=(3.2, 1.7, 0.2, 0.8, 0)',
      'call': 'matched_arrival_rate(*args).tolist()',
      'gold_call': '_oracle_matched_arrival_rate(*args).tolist()'},
     {'setup': 'args=(1.4, 1.9, 0.6, 1.1, 1)',
      'call': 'matched_arrival_rate(*args).tolist()',
      'gold_call': '_oracle_matched_arrival_rate(*args).tolist()'},
     {'setup': 'def check(f):\n'
               '    try:\n'
               '        f(2.5,.1,.35,.9,2)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n',
      'call': 'check(matched_arrival_rate)',
      'gold_call': 'check(_oracle_matched_arrival_rate)'}]
