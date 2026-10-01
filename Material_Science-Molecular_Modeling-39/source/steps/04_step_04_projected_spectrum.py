"""
Project the response tensors and compute the centered replica spectrum.

A fixed laboratory polarization channel provides one scalar polarizability series per independent replica. A rescaled two-sided periodogram estimates a one-sided autocorrelation integral.

Returns
-------
Tuple (f,y,nu) of real arrays, each with shape (floor(N/2),).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def projected_spectrum(alpha: 'np.ndarray', projector: 'np.ndarray', time_step: float) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]':
    """Project polarizability trajectories and return (f,I,nu).
    
    Parameters
    ----------
    alpha, projector : numpy.ndarray
        Real finite polarizability trajectories and projection matrix.
    time_step : float
        Positive finite sampling interval.

    alpha is real finite with shape (M,N,3,3), M>=1,N>=4;
    projector is a real finite (3,3) matrix; time_step=h is finite and >0.
    x[m,t]=sum_ab projector[a,b]*alpha[m,t,a,b]. Subtract the time mean
    separately from each replica. Use X[m,k]=sum_t x[m,t]*exp(-2*pi*i*k*t/N).
    Discard k=0. Return f=k/(N*h), I=h*sum_m(abs(X[m,k])**2)/(2*N*M)
    and nu=2*M, except nu=M at k=N/2 if N is even.
    k=1,...,floor(N/2). All three outputs have shape (floor(N/2),).
    The factor 1/2 belongs to the autocorrelation integral convention;
    do not double the positive-frequency bins of rfft.
    
    Returns
    -------
    result
        Tuple (f,y,nu) of real arrays, each with shape (floor(N/2),).
    
    Raises
    ------
    ValueError
        If any input contains complex values, including a complex dtype with
        zero imaginary part.
        If any stated shape/finiteness condition fails or h<=0.
    
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

def _oracle_projected_spectrum(alpha: 'np.ndarray', projector: 'np.ndarray', time_step: float) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]':
    import numpy as np

    for _name, _value in (('alpha', alpha), ('projector', projector), ('time_step', time_step),):
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
        alpha=np.asarray(alpha,dtype=float)
    except (ValueError,TypeError,OverflowError) as exc:
        raise ValueError('alpha') from exc
    if alpha.ndim!=4 or alpha.shape[2:]!=(3,3) or alpha.shape[0]<1 or alpha.shape[1]<4 or not np.isfinite(alpha).all():
        raise ValueError('alpha')
    projector=_array(projector,(3,3),'projector'); h=_scalar(time_step,'time_step')
    if h<=0: raise ValueError('time_step')
    M,N=alpha.shape[:2]
    x=np.einsum('mnab,ab->mn',alpha,projector)
    x=x-x.mean(axis=1,keepdims=True)
    X=np.fft.rfft(x,axis=1)[:,1:]
    I=h*np.mean(abs(X)**2,axis=0)/(2*N)
    f=np.arange(1,N//2+1)/(N*h)
    nu=np.full(len(f),2*M,dtype=float)
    if N%2==0:nu[-1]=M
    return f,I,nu

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Numerical cases and explicitly declared invalid inputs."""
    return [{'setup': 'import numpy as np\n'
               'M=3;N=8\n'
               'alpha=np.zeros((M,N,3,3))\n'
               'for m in range(M):\n'
               ' for t in range(N):\n'
               '  alpha[m,t,0,1]=alpha[m,t,1,0]=2+m+.3*np.cos(2*np.pi*t/N+.4*m)+.1*(-1.)**t\n'
               'P=np.array([[0.,1.,0.],[0.,0.,0.],[0.,0.,0.]])\n'
               'h=.2\n',
      'call': 'projected_spectrum(alpha,P,h)',
      'gold_call': '_oracle_projected_spectrum(alpha,P,h)'},
     {'setup': 'import numpy as np\n'
               'M=3;N=9\n'
               'alpha=np.zeros((M,N,3,3))\n'
               'for m in range(M):\n'
               ' for t in range(N):\n'
               '  alpha[m,t,0,1]=alpha[m,t,1,0]=2+m+.3*np.cos(2*np.pi*t/N+.4*m)+.1*(-1.)**t\n'
               'P=np.array([[0.,1.,0.],[0.,0.,0.],[0.,0.,0.]])\n'
               'h=.2\n',
      'call': 'projected_spectrum(alpha,P,h)',
      'gold_call': '_oracle_projected_spectrum(alpha,P,h)'},
     {'setup': 'import numpy as np\n'
               'M=3;N=8\n'
               'alpha=np.zeros((M,N,3,3))\n'
               'for m in range(M):\n'
               ' for t in range(N):\n'
               '  alpha[m,t,0,1]=alpha[m,t,1,0]=2+m+.3*np.cos(2*np.pi*t/N+.4*m)+.1*(-1.)**t\n'
               'P=np.array([[0.,1.,0.],[0.,0.,0.],[0.,0.,0.]])\n'
               'h=.2\n'
               'alpha[:]=2.\n',
      'call': 'projected_spectrum(alpha,P,h)',
      'gold_call': '_oracle_projected_spectrum(alpha,P,h)'},
     {'setup': 'import numpy as np\n'
               'M=1;N=4\n'
               'alpha=np.zeros((M,N,3,3))\n'
               'for m in range(M):\n'
               ' for t in range(N):\n'
               '  alpha[m,t,0,1]=alpha[m,t,1,0]=2+m+.3*np.cos(2*np.pi*t/N+.4*m)+.1*(-1.)**t\n'
               'P=np.array([[0.,1.,0.],[0.,0.,0.],[0.,0.,0.]])\n'
               'h=.2\n',
      'call': 'projected_spectrum(alpha,P,h)',
      'gold_call': '_oracle_projected_spectrum(alpha,P,h)'},
     {'setup': 'import numpy as np\n'
               'M=3;N=8\n'
               'alpha=np.zeros((M,N,3,3))\n'
               'for m in range(M):\n'
               ' for t in range(N):\n'
               '  alpha[m,t,0,1]=alpha[m,t,1,0]=2+m+.3*np.cos(2*np.pi*t/N+.4*m)+.1*(-1.)**t\n'
               'P=np.array([[0.,1.,0.],[0.,0.,0.],[0.,0.,0.]])\n'
               'h=.2\n'
               'P[:]=0.\n',
      'call': 'projected_spectrum(alpha,P,h)',
      'gold_call': '_oracle_projected_spectrum(alpha,P,h)'},
     {'setup': 'import numpy as np\n'
               'M=3;N=8\n'
               'alpha=np.zeros((M,N,3,3))\n'
               'for m in range(M):\n'
               ' for t in range(N):\n'
               '  alpha[m,t,0,1]=alpha[m,t,1,0]=2+m+.3*np.cos(2*np.pi*t/N+.4*m)+.1*(-1.)**t\n'
               'P=np.array([[0.,1.,0.],[0.,0.,0.],[0.,0.,0.]])\n'
               'h=.2\n'
               '\n'
               'def _error_code(thunk):\n'
               '    try:thunk()\n'
               '    except ValueError:return 1\n'
               '    except Exception:return 2\n'
               '    return 0\n'
               'h=0.\n',
      'call': '_error_code(lambda:projected_spectrum(alpha,P,h))',
      'gold_call': '_error_code(lambda:_oracle_projected_spectrum(alpha,P,h))'},
     {'setup': 'import numpy as np\n'
               'M=3;N=8\n'
               'alpha=np.zeros((M,N,3,3))\n'
               'for m in range(M):\n'
               ' for t in range(N):\n'
               '  alpha[m,t,0,1]=alpha[m,t,1,0]=2+m+.3*np.cos(2*np.pi*t/N+.4*m)+.1*(-1.)**t\n'
               'P=np.array([[0.,1.,0.],[0.,0.,0.],[0.,0.,0.]])\n'
               'h=.2\n'
               '\n'
               'alpha=alpha.astype(complex)+1j\n'
               '\n'
               'def _error_code(thunk):\n'
               '    try:\n'
               '        thunk()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_error_code(lambda: projected_spectrum(alpha,P,h))',
      'gold_call': '_error_code(lambda: _oracle_projected_spectrum(alpha,P,h))'}]
