"""
Evaluate the polarizability of the charge-relaxed energy.

The molecular polarizability is the negative field Hessian of the minimized QEq energy. Field-dependent hardness changes both the explicit curvature and the relaxation correction.

Returns
-------
Real array alpha with shape (3,3).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def relaxed_polarizability(A1: 'np.ndarray', A2: 'np.ndarray', b1: 'np.ndarray', b2: 'np.ndarray', c2: 'np.ndarray', q: 'np.ndarray', dq: 'np.ndarray') -> 'np.ndarray':
    """Compute alpha_ab=-partial_a partial_b min_{sum q=Q} U.
    
    Parameters
    ----------
    A1, A2, b1, b2, c2, q, dq : numpy.ndarray
        Real finite coefficient derivatives, relaxed charges, and charge response.

    For U=c+b.T@q+q.T@A@q/2, the inputs are field derivatives at fixed
    geometry and the constrained charge response dq[b,i]=partial_b q_i.
    q has shape (n,), n>=1; A1 (3,n,n); A2 (3,3,n,n);
    b1 (3,n); b2 (3,3,n); c2 (3,3); dq (3,n). Inputs are finite and real.
    Compute H_ab=c2_ab+sum_i b2_ab,i*q_i
    +0.5*sum_ij q_i*A2_ab,ij*q_j
    +sum_i (b1_a,i+sum_j A1_a,ij*q_j)*dq_b,i.
    Return -(H+H.T)/2 as a (3,3) array. In consistent inputs H is symmetric.
    Do not append a charge-position dipole or freeze A's field dependence.
    
    Returns
    -------
    result
        Real array alpha with shape (3,3).
    
    Raises
    ------
    ValueError
        If any input contains complex values, including a complex dtype with
        zero imaginary part.
        If any stated shape or finiteness condition fails.
    
    Notes
    --------------------------
    Import required packages inside the function. NumPy and SciPy are installed.
    An import in test setup does not define a global in this function.
    Implement the public function only; the reference implementation is separate.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_relaxed_polarizability(A1: 'np.ndarray', A2: 'np.ndarray', b1: 'np.ndarray', b2: 'np.ndarray', c2: 'np.ndarray', q: 'np.ndarray', dq: 'np.ndarray') -> 'np.ndarray':
    import numpy as np

    for _name, _value in (('A1', A1), ('A2', A2), ('b1', b1), ('b2', b2), ('c2', c2), ('q', q), ('dq', dq),):
        if np.iscomplexobj(_value):
            raise ValueError(_name + " must be real")

    def _array(x, shape, name):
        try:
            x = np.asarray(x, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(name) from exc
        if x.shape != shape or not np.all(np.isfinite(x)):
            raise ValueError(name)
        return x

    try:
        q=np.asarray(q,dtype=float)
    except (ValueError,TypeError,OverflowError) as exc:
        raise ValueError('q') from exc
    if q.ndim!=1 or len(q)<1:
        raise ValueError('q')
    n=len(q); q=_array(q,(n,),'q')
    A1=_array(A1,(3,n,n),'A1'); A2=_array(A2,(3,3,n,n),'A2')
    b1=_array(b1,(3,n),'b1'); b2=_array(b2,(3,3,n),'b2')
    c2=_array(c2,(3,3),'c2'); dq=_array(dq,(3,n),'dq')
    tangent=b1+np.einsum('aij,j->ai',A1,q)
    H=c2+np.einsum('abi,i->ab',b2,q)+0.5*np.einsum('i,abij,j->ab',q,A2,q)+tangent@dq.T
    return -0.5*(H+H.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Numerical cases and explicitly declared invalid inputs."""
    return [{'setup': 'import numpy as np\n'
               'A=np.array([[2.,.3,-.1],[.3,1.5,.2],[-.1,.2,1.8]])\n'
               'A1=np.array([[[.2,.03,0],[.03,-.1,.02],[0,.02,.1]],[[.1,0,-.02],[0,.15,0],[-.02,0,-.05]],[[0,.04,0],[.04,.1,.03],[0,.03,.2]]])\n'
               'b=np.array([-.6,.3,.8]);b1=np.array([[.3,-.2,.1],[-.4,.1,.2],[.05,.15,-.2]])\n'
               'Q=.37\n'
               '\n'
               'K=np.block([[A,np.ones((3,1))],[np.ones((1,3)),np.zeros((1,1))]])\n'
               'q=np.linalg.solve(K,np.r_[-b,Q])[:3]\n'
               'dq=np.array([np.linalg.solve(K,np.r_[-A1[a]@q-b1[a],0.])[:3] for a in range(3)])\n'
               'A2=np.zeros((3,3,3,3));b2=np.zeros((3,3,3))\n'
               'for a in range(3):\n'
               ' for bidx in range(3):\n'
               '  A2[a,bidx]=np.diag(np.array([.08,-.03,.04])*(a+1)*(bidx+1))\n'
               '  b2[a,bidx]=np.array([.02,-.05,.03])*(1+a+bidx)\n'
               'c2=-np.array([[1.7,.2,-.1],[.2,1.5,.15],[-.1,.15,1.8]])\n',
      'call': 'relaxed_polarizability(A1,A2,b1,b2,c2,q,dq)',
      'gold_call': '_oracle_relaxed_polarizability(A1,A2,b1,b2,c2,q,dq)'},
     {'setup': 'import numpy as np\n'
               'A=np.array([[2.,.3,-.1],[.3,1.5,.2],[-.1,.2,1.8]])\n'
               'A1=np.array([[[.2,.03,0],[.03,-.1,.02],[0,.02,.1]],[[.1,0,-.02],[0,.15,0],[-.02,0,-.05]],[[0,.04,0],[.04,.1,.03],[0,.03,.2]]])\n'
               'b=np.array([-.6,.3,.8]);b1=np.array([[.3,-.2,.1],[-.4,.1,.2],[.05,.15,-.2]])\n'
               'Q=.37\n'
               '\n'
               'K=np.block([[A,np.ones((3,1))],[np.ones((1,3)),np.zeros((1,1))]])\n'
               'q=np.linalg.solve(K,np.r_[-b,Q])[:3]\n'
               'dq=np.array([np.linalg.solve(K,np.r_[-A1[a]@q-b1[a],0.])[:3] for a in range(3)])\n'
               'A2=np.zeros((3,3,3,3));b2=np.zeros((3,3,3))\n'
               'for a in range(3):\n'
               ' for bidx in range(3):\n'
               '  A2[a,bidx]=np.diag(np.array([.08,-.03,.04])*(a+1)*(bidx+1))\n'
               '  b2[a,bidx]=np.array([.02,-.05,.03])*(1+a+bidx)\n'
               'c2=-np.array([[1.7,.2,-.1],[.2,1.5,.15],[-.1,.15,1.8]])\n'
               'A1[:]=0;A2[:]=0\n',
      'call': 'relaxed_polarizability(A1,A2,b1,b2,c2,q,dq)',
      'gold_call': '_oracle_relaxed_polarizability(A1,A2,b1,b2,c2,q,dq)'},
     {'setup': 'import numpy as np\n'
               'A=np.array([[2.,.3,-.1],[.3,1.5,.2],[-.1,.2,1.8]])\n'
               'A1=np.array([[[.2,.03,0],[.03,-.1,.02],[0,.02,.1]],[[.1,0,-.02],[0,.15,0],[-.02,0,-.05]],[[0,.04,0],[.04,.1,.03],[0,.03,.2]]])\n'
               'b=np.array([-.6,.3,.8]);b1=np.array([[.3,-.2,.1],[-.4,.1,.2],[.05,.15,-.2]])\n'
               'Q=.37\n'
               '\n'
               'K=np.block([[A,np.ones((3,1))],[np.ones((1,3)),np.zeros((1,1))]])\n'
               'q=np.linalg.solve(K,np.r_[-b,Q])[:3]\n'
               'dq=np.array([np.linalg.solve(K,np.r_[-A1[a]@q-b1[a],0.])[:3] for a in range(3)])\n'
               'A2=np.zeros((3,3,3,3));b2=np.zeros((3,3,3))\n'
               'for a in range(3):\n'
               ' for bidx in range(3):\n'
               '  A2[a,bidx]=np.diag(np.array([.08,-.03,.04])*(a+1)*(bidx+1))\n'
               '  b2[a,bidx]=np.array([.02,-.05,.03])*(1+a+bidx)\n'
               'c2=-np.array([[1.7,.2,-.1],[.2,1.5,.15],[-.1,.15,1.8]])\n'
               'dq[:]=0\n',
      'call': 'relaxed_polarizability(A1,A2,b1,b2,c2,q,dq)',
      'gold_call': '_oracle_relaxed_polarizability(A1,A2,b1,b2,c2,q,dq)'},
     {'setup': 'import numpy as np\n'
               'A=np.array([[2.,.3,-.1],[.3,1.5,.2],[-.1,.2,1.8]])\n'
               'A1=np.array([[[.2,.03,0],[.03,-.1,.02],[0,.02,.1]],[[.1,0,-.02],[0,.15,0],[-.02,0,-.05]],[[0,.04,0],[.04,.1,.03],[0,.03,.2]]])\n'
               'b=np.array([-.6,.3,.8]);b1=np.array([[.3,-.2,.1],[-.4,.1,.2],[.05,.15,-.2]])\n'
               'Q=.37\n'
               '\n'
               'K=np.block([[A,np.ones((3,1))],[np.ones((1,3)),np.zeros((1,1))]])\n'
               'q=np.linalg.solve(K,np.r_[-b,Q])[:3]\n'
               'dq=np.array([np.linalg.solve(K,np.r_[-A1[a]@q-b1[a],0.])[:3] for a in range(3)])\n'
               'A2=np.zeros((3,3,3,3));b2=np.zeros((3,3,3))\n'
               'for a in range(3):\n'
               ' for bidx in range(3):\n'
               '  A2[a,bidx]=np.diag(np.array([.08,-.03,.04])*(a+1)*(bidx+1))\n'
               '  b2[a,bidx]=np.array([.02,-.05,.03])*(1+a+bidx)\n'
               'c2=-np.array([[1.7,.2,-.1],[.2,1.5,.15],[-.1,.15,1.8]])\n'
               'q[:]=0;dq[:]=0\n',
      'call': 'relaxed_polarizability(A1,A2,b1,b2,c2,q,dq)',
      'gold_call': '_oracle_relaxed_polarizability(A1,A2,b1,b2,c2,q,dq)'},
     {'setup': 'import numpy as np\n'
               'A=np.array([[2.,.3,-.1],[.3,1.5,.2],[-.1,.2,1.8]])\n'
               'A1=np.array([[[.2,.03,0],[.03,-.1,.02],[0,.02,.1]],[[.1,0,-.02],[0,.15,0],[-.02,0,-.05]],[[0,.04,0],[.04,.1,.03],[0,.03,.2]]])\n'
               'b=np.array([-.6,.3,.8]);b1=np.array([[.3,-.2,.1],[-.4,.1,.2],[.05,.15,-.2]])\n'
               'Q=.37\n'
               '\n'
               'K=np.block([[A,np.ones((3,1))],[np.ones((1,3)),np.zeros((1,1))]])\n'
               'q=np.linalg.solve(K,np.r_[-b,Q])[:3]\n'
               'dq=np.array([np.linalg.solve(K,np.r_[-A1[a]@q-b1[a],0.])[:3] for a in range(3)])\n'
               'A2=np.zeros((3,3,3,3));b2=np.zeros((3,3,3))\n'
               'for a in range(3):\n'
               ' for bidx in range(3):\n'
               '  A2[a,bidx]=np.diag(np.array([.08,-.03,.04])*(a+1)*(bidx+1))\n'
               '  b2[a,bidx]=np.array([.02,-.05,.03])*(1+a+bidx)\n'
               'c2=-np.array([[1.7,.2,-.1],[.2,1.5,.15],[-.1,.15,1.8]])\n'
               'c2=np.array([[0.,.4,-.2],[.4,0.,.3],[-.2,.3,0.]])\n',
      'call': 'relaxed_polarizability(A1,A2,b1,b2,c2,q,dq)',
      'gold_call': '_oracle_relaxed_polarizability(A1,A2,b1,b2,c2,q,dq)'},
     {'setup': 'import numpy as np\n'
               'A=np.array([[2.,.3,-.1],[.3,1.5,.2],[-.1,.2,1.8]])\n'
               'A1=np.array([[[.2,.03,0],[.03,-.1,.02],[0,.02,.1]],[[.1,0,-.02],[0,.15,0],[-.02,0,-.05]],[[0,.04,0],[.04,.1,.03],[0,.03,.2]]])\n'
               'b=np.array([-.6,.3,.8]);b1=np.array([[.3,-.2,.1],[-.4,.1,.2],[.05,.15,-.2]])\n'
               'Q=.37\n'
               '\n'
               'K=np.block([[A,np.ones((3,1))],[np.ones((1,3)),np.zeros((1,1))]])\n'
               'q=np.linalg.solve(K,np.r_[-b,Q])[:3]\n'
               'dq=np.array([np.linalg.solve(K,np.r_[-A1[a]@q-b1[a],0.])[:3] for a in range(3)])\n'
               'A2=np.zeros((3,3,3,3));b2=np.zeros((3,3,3))\n'
               'for a in range(3):\n'
               ' for bidx in range(3):\n'
               '  A2[a,bidx]=np.diag(np.array([.08,-.03,.04])*(a+1)*(bidx+1))\n'
               '  b2[a,bidx]=np.array([.02,-.05,.03])*(1+a+bidx)\n'
               'c2=-np.array([[1.7,.2,-.1],[.2,1.5,.15],[-.1,.15,1.8]])\n'
               '\n'
               'def _error_code(thunk):\n'
               '    try:thunk()\n'
               '    except ValueError:return 1\n'
               '    except Exception:return 2\n'
               '    return 0\n'
               'dq=dq[:,:2]\n',
      'call': '_error_code(lambda:relaxed_polarizability(A1,A2,b1,b2,c2,q,dq))',
      'gold_call': '_error_code(lambda:_oracle_relaxed_polarizability(A1,A2,b1,b2,c2,q,dq))'},
     {'setup': 'import numpy as np\n'
               'A=np.array([[2.,.3,-.1],[.3,1.5,.2],[-.1,.2,1.8]])\n'
               'A1=np.array([[[.2,.03,0],[.03,-.1,.02],[0,.02,.1]],[[.1,0,-.02],[0,.15,0],[-.02,0,-.05]],[[0,.04,0],[.04,.1,.03],[0,.03,.2]]])\n'
               'b=np.array([-.6,.3,.8]);b1=np.array([[.3,-.2,.1],[-.4,.1,.2],[.05,.15,-.2]])\n'
               'Q=.37\n'
               '\n'
               'K=np.block([[A,np.ones((3,1))],[np.ones((1,3)),np.zeros((1,1))]])\n'
               'q=np.linalg.solve(K,np.r_[-b,Q])[:3]\n'
               'dq=np.array([np.linalg.solve(K,np.r_[-A1[a]@q-b1[a],0.])[:3] for a in range(3)])\n'
               'A2=np.zeros((3,3,3,3));b2=np.zeros((3,3,3))\n'
               'for a in range(3):\n'
               ' for bidx in range(3):\n'
               '  A2[a,bidx]=np.diag(np.array([.08,-.03,.04])*(a+1)*(bidx+1))\n'
               '  b2[a,bidx]=np.array([.02,-.05,.03])*(1+a+bidx)\n'
               'c2=-np.array([[1.7,.2,-.1],[.2,1.5,.15],[-.1,.15,1.8]])\n'
               '\n'
               'q=q.astype(complex)+1j\n'
               '\n'
               'def _error_code(thunk):\n'
               '    try:\n'
               '        thunk()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_error_code(lambda: relaxed_polarizability(A1,A2,b1,b2,c2,q,dq))',
      'gold_call': '_error_code(lambda: _oracle_relaxed_polarizability(A1,A2,b1,b2,c2,q,dq))'}]
