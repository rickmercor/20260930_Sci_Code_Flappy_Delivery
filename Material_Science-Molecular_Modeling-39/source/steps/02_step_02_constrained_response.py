"""
Solve the fixed-charge stationary equations and their field derivatives.

Charge relaxation under a fixed total charge. Differentiate the constrained stationary point rather than freezing charges while differentiating the energy.

Returns
-------
Tuple (q,dq) of real arrays with shapes (n,) and (3,n).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def constrained_response(A: 'np.ndarray', A1: 'np.ndarray', b: 'np.ndarray', b1: 'np.ndarray', total_charge: float) -> 'tuple[np.ndarray, np.ndarray]':
    """Return q and dq for min_q b.T@q+q.T@A@q/2 subject to sum(q)=Q.
    
    Parameters
    ----------
    A, A1, b, b1 : numpy.ndarray
        Real finite energy coefficients and first field derivatives.
    total_charge : float
        Finite constrained total charge Q.

    A is a real finite symmetric positive-definite (n,n) matrix, n>=1.
    A1[a,i,j]=partial A_ij/partial field_a has shape (3,n,n), with each
    matrix symmetric. b has shape (n,), b1=partial_a b_i has shape (3,n).
    total_charge=Q is finite. Build K=[[A,ones],[ones.T,0]].
    K@[q,lambda]=[-b,Q]. For each a,
    K@[dq[a],dlambda[a]]=[-A1[a]@q-b1[a],0].
    Return (q,dq) with shapes (n,) and (3,n). Charges are re-equilibrated
    under the same total-charge constraint for every field derivative.
    
    Returns
    -------
    result
        Tuple (q,dq) of real arrays with shapes (n,) and (3,n).
    
    Raises
    ------
    ValueError
        If any input contains complex values, including a complex dtype with
        zero imaginary part.
        If the shapes/finiteness conditions fail, A is not positive definite,
        or A or any A1[a] is nonsymmetric beyond absolute 1e-12.
    
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

def _oracle_constrained_response(A: 'np.ndarray', A1: 'np.ndarray', b: 'np.ndarray', b1: 'np.ndarray', total_charge: float) -> 'tuple[np.ndarray, np.ndarray]':
    import numpy as np

    for _name, _value in (('A', A), ('A1', A1), ('b', b), ('b1', b1), ('total_charge', total_charge),):
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

    def _scalar(x, name):
        try:
            a = np.asarray(x, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(name) from exc
        if a.shape != () or not np.isfinite(a):
            raise ValueError(name)
        return float(a)
    try:
        A = np.asarray(A, dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('A') from exc
    if A.ndim!=2 or A.shape[0]<1 or A.shape[0]!=A.shape[1]:
        raise ValueError('A')
    n=len(A)
    A=_array(A,(n,n),'A'); A1=_array(A1,(3,n,n),'A1')
    b=_array(b,(n,),'b'); b1=_array(b1,(3,n),'b1')
    Q=_scalar(total_charge,'total_charge')
    if not np.allclose(A,A.T,atol=1e-12,rtol=0) or not np.allclose(A1,A1.swapaxes(1,2),atol=1e-12,rtol=0):
        raise ValueError('symmetry')
    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError as exc:
        raise ValueError('A must be positive definite') from exc
    K=np.zeros((n+1,n+1)); K[:n,:n]=A; K[n,:n]=1; K[:n,n]=1
    q=np.linalg.solve(K,np.r_[-b,Q])[:n]
    rhs=np.zeros((n+1,3)); rhs[:n]=-(np.einsum('aij,j->ai',A1,q)+b1).T
    dq=np.linalg.solve(K,rhs)[:n].T
    return q,dq

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Numerical cases and explicitly declared invalid inputs."""
    return [{'setup': 'import numpy as np\n'
               'A=np.array([[2.,.3,-.1],[.3,1.5,.2],[-.1,.2,1.8]])\n'
               'A1=np.array([[[.2,.03,0],[.03,-.1,.02],[0,.02,.1]],[[.1,0,-.02],[0,.15,0],[-.02,0,-.05]],[[0,.04,0],[.04,.1,.03],[0,.03,.2]]])\n'
               'b=np.array([-.6,.3,.8]);b1=np.array([[.3,-.2,.1],[-.4,.1,.2],[.05,.15,-.2]])\n'
               'Q=.37\n',
      'call': 'constrained_response(A,A1,b,b1,Q)',
      'gold_call': '_oracle_constrained_response(A,A1,b,b1,Q)'},
     {'setup': 'import numpy as np\n'
               'A=np.array([[2.,.3,-.1],[.3,1.5,.2],[-.1,.2,1.8]])\n'
               'A1=np.array([[[.2,.03,0],[.03,-.1,.02],[0,.02,.1]],[[.1,0,-.02],[0,.15,0],[-.02,0,-.05]],[[0,.04,0],[.04,.1,.03],[0,.03,.2]]])\n'
               'b=np.array([-.6,.3,.8]);b1=np.array([[.3,-.2,.1],[-.4,.1,.2],[.05,.15,-.2]])\n'
               'Q=.37\n'
               'Q=0.\n',
      'call': 'constrained_response(A,A1,b,b1,Q)',
      'gold_call': '_oracle_constrained_response(A,A1,b,b1,Q)'},
     {'setup': 'import numpy as np\n'
               'A=np.array([[2.]]);A1=np.ones((3,1,1));b=np.array([.7]);b1=np.ones((3,1));Q=-.4\n',
      'call': 'constrained_response(A,A1,b,b1,Q)',
      'gold_call': '_oracle_constrained_response(A,A1,b,b1,Q)'},
     {'setup': 'import numpy as np\n'
               'A=np.array([[2.,.3,-.1],[.3,1.5,.2],[-.1,.2,1.8]])\n'
               'A1=np.array([[[.2,.03,0],[.03,-.1,.02],[0,.02,.1]],[[.1,0,-.02],[0,.15,0],[-.02,0,-.05]],[[0,.04,0],[.04,.1,.03],[0,.03,.2]]])\n'
               'b=np.array([-.6,.3,.8]);b1=np.array([[.3,-.2,.1],[-.4,.1,.2],[.05,.15,-.2]])\n'
               'Q=.37\n'
               'A1[:]=0;b1[:]=0\n',
      'call': 'constrained_response(A,A1,b,b1,Q)',
      'gold_call': '_oracle_constrained_response(A,A1,b,b1,Q)'},
     {'setup': 'import numpy as np\n'
               'A=np.array([[2.,.3,-.1],[.3,1.5,.2],[-.1,.2,1.8]])\n'
               'A1=np.array([[[.2,.03,0],[.03,-.1,.02],[0,.02,.1]],[[.1,0,-.02],[0,.15,0],[-.02,0,-.05]],[[0,.04,0],[.04,.1,.03],[0,.03,.2]]])\n'
               'b=np.array([-.6,.3,.8]);b1=np.array([[.3,-.2,.1],[-.4,.1,.2],[.05,.15,-.2]])\n'
               'Q=.37\n'
               'A=np.diag([.003,.2,2.]);Q=-.1\n',
      'call': 'constrained_response(A,A1,b,b1,Q)',
      'gold_call': '_oracle_constrained_response(A,A1,b,b1,Q)'},
     {'setup': 'import numpy as np\n'
               'A=np.array([[2.,.3,-.1],[.3,1.5,.2],[-.1,.2,1.8]])\n'
               'A1=np.array([[[.2,.03,0],[.03,-.1,.02],[0,.02,.1]],[[.1,0,-.02],[0,.15,0],[-.02,0,-.05]],[[0,.04,0],[.04,.1,.03],[0,.03,.2]]])\n'
               'b=np.array([-.6,.3,.8]);b1=np.array([[.3,-.2,.1],[-.4,.1,.2],[.05,.15,-.2]])\n'
               'Q=.37\n'
               '\n'
               'def _error_code(thunk):\n'
               '    try:thunk()\n'
               '    except ValueError:return 1\n'
               '    except Exception:return 2\n'
               '    return 0\n'
               'A[0,0]=-1\n',
      'call': '_error_code(lambda:constrained_response(A,A1,b,b1,Q))',
      'gold_call': '_error_code(lambda:_oracle_constrained_response(A,A1,b,b1,Q))'},
     {'setup': 'import numpy as np\n'
               'A=np.array([[2.,.3,-.1],[.3,1.5,.2],[-.1,.2,1.8]])\n'
               'A1=np.array([[[.2,.03,0],[.03,-.1,.02],[0,.02,.1]],[[.1,0,-.02],[0,.15,0],[-.02,0,-.05]],[[0,.04,0],[.04,.1,.03],[0,.03,.2]]])\n'
               'b=np.array([-.6,.3,.8]);b1=np.array([[.3,-.2,.1],[-.4,.1,.2],[.05,.15,-.2]])\n'
               'Q=.37\n'
               '\n'
               'def _error_code(thunk):\n'
               '    try:thunk()\n'
               '    except ValueError:return 1\n'
               '    except Exception:return 2\n'
               '    return 0\n'
               'A1[0,0,1]+=.1\n',
      'call': '_error_code(lambda:constrained_response(A,A1,b,b1,Q))',
      'gold_call': '_error_code(lambda:_oracle_constrained_response(A,A1,b,b1,Q))'},
     {'setup': 'import numpy as np\n'
               'A=np.array([[2.,.3,-.1],[.3,1.5,.2],[-.1,.2,1.8]])\n'
               'A1=np.array([[[.2,.03,0],[.03,-.1,.02],[0,.02,.1]],[[.1,0,-.02],[0,.15,0],[-.02,0,-.05]],[[0,.04,0],[.04,.1,.03],[0,.03,.2]]])\n'
               'b=np.array([-.6,.3,.8]);b1=np.array([[.3,-.2,.1],[-.4,.1,.2],[.05,.15,-.2]])\n'
               'Q=.37\n'
               '\n'
               'A=A.astype(complex)+1j\n'
               '\n'
               'def _error_code(thunk):\n'
               '    try:\n'
               '        thunk()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_error_code(lambda: constrained_response(A,A1,b,b1,Q))',
      'gold_call': '_error_code(lambda: _oracle_constrained_response(A,A1,b,b1,Q))'}]
