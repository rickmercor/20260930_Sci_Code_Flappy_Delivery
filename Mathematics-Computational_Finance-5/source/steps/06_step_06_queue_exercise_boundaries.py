"""
Find the numerical exercise boundary separately for every activation state. For row q of C define V_q(x)=sum_k' Re(C[q,k]*exp(1j*omega[k]*(x-a_next))). The prime halves k=0. Solve V_q(x_q)=cost for every row on the supplied outer interval [a,b], using absolute and relative root tolerances 1e-12. Require V_q(a)<cost<V_q(b). Valid inputs have exactly one crossing and positive slope there; these properties are guaranteed apart from the explicitly checked endpoint bracket.

The investment decision compares continuation with a fixed cost separately in every activation state.

Returns
-------
return np.zeros(np.shape(C)[0])
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def queue_exercise_boundaries(C,omega,a_next,a,b,cost):
    """Return a float vector containing one root per row of C.

    C: finite complex shape (Q,N); omega: finite real shape (N,), starts
    at zero and strictly increases. Q,N are positive. a_next,a,b,cost are
    finite, b>a and cost>0. Raise ValueError for invalid shapes/ranges,
    nonfinite inputs, or failure of any strict endpoint bracket.
    """
    return np.zeros(np.shape(C)[0])

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_queue_exercise_boundaries(C,omega,a_next,a,b,cost):
    np=__import__('numpy');brentq=__import__('scipy.optimize',fromlist=['brentq']).brentq
    C=np.asarray(C,complex);w=np.asarray(omega,float)
    if C.ndim!=2 or min(C.shape)<1 or w.shape!=(C.shape[1],) or not np.all(np.isfinite(C)) or not np.all(np.isfinite(w)):
        raise ValueError('coefficient shapes')
    if w[0]!=0 or np.any(np.diff(w)<=0) or not np.all(np.isfinite([a_next,a,b,cost])) or b<=a or cost<=0:
        raise ValueError('ranges')
    prime=np.ones(len(w));prime[0]=.5;roots=[]
    for row in C:
        def f(x):return np.real(row*np.exp(1j*w*(x-a_next)))@prime-cost
        if not f(a)<0<f(b):raise ValueError('strict boundary bracket')
        roots.append(brentq(f,a,b,xtol=1e-12,rtol=1e-12))
    return np.array(roots)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return explicit dictionaries for Studio differential-case preflight."""
    return [{'setup': 'import numpy as np\n'
               'roots=np.array([0.7, 1.4, 2.2])\n'
               'w=np.array([0.,.8,1.6]);C=np.zeros((len(roots),3),complex)\n'
               'C[:,0]=2*(3.5+2*np.cos(.8*(roots+1)))\n'
               'C[:,1]=-2\n'
               'args=(C,w,-1.,0.,2.8,3.5)\n',
      'call': 'queue_exercise_boundaries(*args).tolist()',
      'gold_call': '_oracle_queue_exercise_boundaries(*args).tolist()'},
     {'setup': 'import numpy as np\n'
               'roots=np.array([1.1])\n'
               'w=np.array([0.,.8,1.6]);C=np.zeros((len(roots),3),complex)\n'
               'C[:,0]=2*(3.5+2*np.cos(.8*(roots+1)))\n'
               'C[:,1]=-2\n'
               'args=(C,w,-1.,0.,2.8,3.5)\n',
      'call': 'queue_exercise_boundaries(*args).tolist()',
      'gold_call': '_oracle_queue_exercise_boundaries(*args).tolist()'},
     {'setup': 'import numpy as np\n'
               'roots=np.array([0.3, 0.9, 1.8, 2.5])\n'
               'w=np.array([0.,.8,1.6]);C=np.zeros((len(roots),3),complex)\n'
               'C[:,0]=2*(3.5+2*np.cos(.8*(roots+1)))\n'
               'C[:,1]=-2\n'
               'args=(C,w,-1.,0.,2.8,3.5)\n',
      'call': 'queue_exercise_boundaries(*args).tolist()',
      'gold_call': '_oracle_queue_exercise_boundaries(*args).tolist()'},
     {'setup': 'import numpy as np\n'
               'def check(f):\n'
               '    try:\n'
               '        f(np.zeros((2,3)),[0.,1.,2.],-1.,0.,2.,3.5)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n',
      'call': 'check(queue_exercise_boundaries)',
      'gold_call': 'check(_oracle_queue_exercise_boundaries)'}]
