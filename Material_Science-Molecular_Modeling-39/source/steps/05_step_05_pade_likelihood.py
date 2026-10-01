"""
Evaluate the weighted Gamma loss, gradient and observed Hessian.

Weighted Gamma likelihood for a rational low-frequency spectrum. Its observed Hessian contains the curvature of the rational model as well as a Jacobian product.

Returns
-------
Tuple (loss,gradient,hessian): native float, real array (3,), real array (3,3).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pade_likelihood(p: 'np.ndarray', frequency: 'np.ndarray', observed: 'np.ndarray', dof: 'np.ndarray', weights: 'np.ndarray') -> 'tuple[float, np.ndarray, np.ndarray]':
    """Return the weighted negative log-likelihood, gradient and observed Hessian.
    
    Parameters
    ----------
    p, frequency, observed, dof, weights : numpy.ndarray
        Real finite Padé parameters and spectral data with the constraints below.

    p=(p0,p2,q2) is finite, shape (3,). Frequency f is a finite nonnegative,
    strictly increasing vector of length K>=4. observed=y and dof=nu are
    positive finite vectors of length K. weights=w is finite, nonnegative,
    same length, with at least one positive entry.
    m_k=(p0+p2*f_k**2)/(1+q2*f_k**2); both numerator and denominator
    must be strictly positive at every supplied frequency, including w=0.
    L=sum_k w_k*nu_k/2*(log(m_k)+y_k/m_k). Constants independent of p
    are omitted. Return (float(L), dL/dp, d2L/dp2), shapes (), (3,), (3,3).
    Use the observed Hessian, not Fisher information or Gauss-Newton.
    
    Returns
    -------
    result
        Tuple (loss,gradient,hessian): native float, real array (3,), real array (3,3).
    
    Raises
    ------
    ValueError
        If any input contains complex values, including a complex dtype with
        zero imaginary part.
        If a stated shape/finiteness/positivity/order condition fails.
    
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

def _oracle_pade_likelihood(p: 'np.ndarray', frequency: 'np.ndarray', observed: 'np.ndarray', dof: 'np.ndarray', weights: 'np.ndarray') -> 'tuple[float, np.ndarray, np.ndarray]':
    import numpy as np

    for _name, _value in (('p', p), ('frequency', frequency), ('observed', observed), ('dof', dof), ('weights', weights),):
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


    def _spectrum(f, observed, dof):
        try:
            f = np.asarray(f, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError('frequency') from exc
        if f.ndim != 1 or len(f)<4 or not np.isfinite(f).all() or np.any(f<0) or np.any(np.diff(f)<=0):
            raise ValueError('frequency')
        y = _array(observed, f.shape, 'observed')
        nu = _array(dof, f.shape, 'dof')
        if np.any(y<=0) or np.any(nu<=0):
            raise ValueError('observed and dof must be positive')
        return f,y,nu

    def _pade_jet(p, f):
        x = f*f
        num = p[0]+p[1]*x
        den = 1+p[2]*x
        if np.any(num<=0) or np.any(den<=0):
            raise ValueError('nonpositive Pade numerator or denominator')
        m = num/den
        J = np.stack((1/den,x/den,-num*x/den**2),axis=1)
        D = np.zeros((len(f),3,3))
        D[:,0,2]=D[:,2,0]=-x/den**2
        D[:,1,2]=D[:,2,1]=-x*x/den**2
        D[:,2,2]=2*num*x*x/den**3
        return m,J,D
    f,y,nu=_spectrum(frequency,observed,dof)
    p=_array(p,(3,),'p'); w=_array(weights,f.shape,'weights')
    if np.any(w<0) or not np.any(w>0):raise ValueError('weights')
    m,J,D=_pade_jet(p,f)
    a=w*nu/2
    first=a*(m-y)/m**2
    second=a*(2*y-m)/m**3
    value=np.sum(a*(np.log(m)+y/m))
    gradient=J.T@first
    hessian=J.T@(second[:,None]*J)+np.einsum('k,kab->ab',first,D)
    return float(value),gradient,hessian

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Numerical cases and explicitly declared invalid inputs."""
    return [{'setup': 'import numpy as np\n'
               'f=np.linspace(.01,1.,48);p=np.array([1.4,.25,8.])\n'
               'y=(1.3+.3*f*f)/(1+7*f*f)*(1+.09*np.sin(np.arange(len(f))*1.7))\n'
               'nu=np.full(len(f),18.);nu[-1]=9.\n'
               'w=1/(1+(f/.45)**8)\n',
      'call': 'pade_likelihood(p,f,y,nu,w)',
      'gold_call': '_oracle_pade_likelihood(p,f,y,nu,w)'},
     {'setup': 'import numpy as np\n'
               'f=np.linspace(.01,1.,48);p=np.array([1.4,.25,8.])\n'
               'y=(1.3+.3*f*f)/(1+7*f*f)*(1+.09*np.sin(np.arange(len(f))*1.7))\n'
               'nu=np.full(len(f),18.);nu[-1]=9.\n'
               'w=1/(1+(f/.45)**8)\n'
               'y=(p[0]+p[1]*f*f)/(1+p[2]*f*f)\n',
      'call': 'pade_likelihood(p,f,y,nu,w)',
      'gold_call': '_oracle_pade_likelihood(p,f,y,nu,w)'},
     {'setup': 'import numpy as np\n'
               'f=np.linspace(.01,1.,48);p=np.array([1.4,.25,8.])\n'
               'y=(1.3+.3*f*f)/(1+7*f*f)*(1+.09*np.sin(np.arange(len(f))*1.7))\n'
               'nu=np.full(len(f),18.);nu[-1]=9.\n'
               'w=1/(1+(f/.45)**8)\n'
               'w[w<.01]=0\n',
      'call': 'pade_likelihood(p,f,y,nu,w)',
      'gold_call': '_oracle_pade_likelihood(p,f,y,nu,w)'},
     {'setup': 'import numpy as np\n'
               'f=np.linspace(.01,1.,48);p=np.array([1.4,.25,8.])\n'
               'y=(1.3+.3*f*f)/(1+7*f*f)*(1+.09*np.sin(np.arange(len(f))*1.7))\n'
               'nu=np.full(len(f),18.);nu[-1]=9.\n'
               'w=1/(1+(f/.45)**8)\n'
               'p=np.array([1.4,.25,-.5])\n',
      'call': 'pade_likelihood(p,f,y,nu,w)',
      'gold_call': '_oracle_pade_likelihood(p,f,y,nu,w)'},
     {'setup': 'import numpy as np\n'
               'f=np.linspace(.01,1.,48);p=np.array([1.4,.25,8.])\n'
               'y=(1.3+.3*f*f)/(1+7*f*f)*(1+.09*np.sin(np.arange(len(f))*1.7))\n'
               'nu=np.full(len(f),18.);nu[-1]=9.\n'
               'w=1/(1+(f/.45)**8)\n'
               'p=np.array([1.4,-.25,8.])\n',
      'call': 'pade_likelihood(p,f,y,nu,w)',
      'gold_call': '_oracle_pade_likelihood(p,f,y,nu,w)'},
     {'setup': 'import numpy as np\n'
               'f=np.linspace(.01,1.,48);p=np.array([1.4,.25,8.])\n'
               'y=(1.3+.3*f*f)/(1+7*f*f)*(1+.09*np.sin(np.arange(len(f))*1.7))\n'
               'nu=np.full(len(f),18.);nu[-1]=9.\n'
               'w=1/(1+(f/.45)**8)\n'
               'nu=np.linspace(3.,22.,len(f))\n',
      'call': 'pade_likelihood(p,f,y,nu,w)',
      'gold_call': '_oracle_pade_likelihood(p,f,y,nu,w)'},
     {'setup': 'import numpy as np\n'
               'f=np.linspace(.01,1.,48);p=np.array([1.4,.25,8.])\n'
               'y=(1.3+.3*f*f)/(1+7*f*f)*(1+.09*np.sin(np.arange(len(f))*1.7))\n'
               'nu=np.full(len(f),18.);nu[-1]=9.\n'
               'w=1/(1+(f/.45)**8)\n'
               '\n'
               'def _error_code(thunk):\n'
               '    try:thunk()\n'
               '    except ValueError:return 1\n'
               '    except Exception:return 2\n'
               '    return 0\n'
               'y[0]=0\n',
      'call': '_error_code(lambda:pade_likelihood(p,f,y,nu,w))',
      'gold_call': '_error_code(lambda:_oracle_pade_likelihood(p,f,y,nu,w))'},
     {'setup': 'import numpy as np\n'
               'f=np.linspace(.01,1.,48);p=np.array([1.4,.25,8.])\n'
               'y=(1.3+.3*f*f)/(1+7*f*f)*(1+.09*np.sin(np.arange(len(f))*1.7))\n'
               'nu=np.full(len(f),18.);nu[-1]=9.\n'
               'w=1/(1+(f/.45)**8)\n'
               '\n'
               'p=p.astype(complex)+1j\n'
               '\n'
               'def _error_code(thunk):\n'
               '    try:\n'
               '        thunk()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_error_code(lambda: pade_likelihood(p,f,y,nu,w))',
      'gold_call': '_error_code(lambda: _oracle_pade_likelihood(p,f,y,nu,w))'}]
