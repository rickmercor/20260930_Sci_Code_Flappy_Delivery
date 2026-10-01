"""
Evaluate correlated-window linear cross-validation for one fitted spectrum.

Linearized fits on overlapping low/high frequency windows share the same periodogram noise. Their difference must include this cross-covariance before the Gaussian score is evaluated.

Returns
-------
Tuple (criterion,d,Cd): native float, real array (3,), real array (3,3).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cv2l_score(p: 'np.ndarray', frequency: 'np.ndarray', observed: 'np.ndarray', dof: 'np.ndarray', cutoff: float) -> 'tuple[float, np.ndarray, np.ndarray]':
    """Return (criterion,d,Cd) for STACIE cross-validation with two linearized fits.
    
    Parameters
    ----------
    p, frequency, observed, dof : numpy.ndarray
        Real finite Padé parameters and spectral data.
    cutoff : float
        Positive finite frequency cutoff.

    p=(p0,p2,q2) is a finite real array of shape (3,). Frequency f is a
    finite real strictly increasing vector, length K>=4, with f>=0. observed=y
    and dof=nu are positive finite real vectors of length K. The numerator
    p0+p2*f**2 and denominator 1+q2*f**2 must be positive at every supplied f.
    cutoff=c is a finite positive scalar. Set
    w(f,c)=1/(1+(f/c)**8); w1=w(f,1.25*c/2);
    w2=w(f,1.25*c)-w1. Replace individual w1 or w2 values strictly below
    0.001 by zero; retain all supplied frequencies for this operation.
    Let m_k=(p0+p2*f_k**2)/(1+q2*f_k**2),
    J_k,a=partial m_k/partial p_a, V_k=2*m_k**2/nu_k.
    For h=1,2, define Bh=inv(J.T@diag(wh/V)@J)@J.T@diag(wh/V).
    D=B1-B2; d=D@(y-m); Cd=D@diag(V)@D.T.
    criterion=(3*log(2*pi)+log(det(Cd))+d.T@inv(Cd)@d)/2.
    Return native float criterion, d shape (3,), Cd shape (3,3).
    V uses the fitted spectrum, not observed squared. Account for both
    within-window noise and the covariance between windows. Do not refit
    nonlinear models to the windows. Use all three Pade coordinates as defined.
    
    Returns
    -------
    result
        Tuple (criterion,d,Cd): native float, real array (3,), real array (3,3).
    
    Raises
    ------
    ValueError
        If any input contains complex values, including a complex dtype with
        zero imaginary part.
        On a stated input-contract violation, a rank-deficient weighted
        Jacobian, or a nonpositive-definite covariance of the difference.
    
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

def _oracle_cv2l_score(p: 'np.ndarray', frequency: 'np.ndarray', observed: 'np.ndarray', dof: 'np.ndarray', cutoff: float) -> 'tuple[float, np.ndarray, np.ndarray]':
    import numpy as np

    for _name, _value in (('p', p), ('frequency', frequency), ('observed', observed), ('dof', dof), ('cutoff', cutoff),):
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
    p=_array(p,(3,),'p'); c=_scalar(cutoff,'cutoff')
    if c<=0:raise ValueError('cutoff')
    m,J,_=_pade_jet(p,f)
    # Column scaling preserves the specified parameterization of the result.
    scales=np.array([p[0],p[0]/c**2,1/c**2])
    JS=J*scales
    V=2*m*m/nu
    with np.errstate(over='ignore'):
        w1=1/(1+(f/(1.25*c/2))**8)
        w2=1/(1+(f/(1.25*c))**8)-w1
    w1[w1<0.001]=0; w2[w2<0.001]=0
    matrices=[]
    try:
        for w in (w1,w2):
            rhs=JS.T*(w/V)
            gram=rhs@JS
            np.linalg.cholesky(gram)
            matrices.append(np.linalg.solve(gram,rhs))
        D=matrices[0]-matrices[1]
        d=D@(y-m); Cd=(D*V)@D.T
        np.linalg.cholesky(Cd)
        d=d*scales; Cd=Cd*scales[:,None]*scales[None,:]
        sign,logdet=np.linalg.slogdet(Cd)
        if sign<=0:raise np.linalg.LinAlgError('covariance')
        criterion=0.5*(3*np.log(2*np.pi)+logdet+d@np.linalg.solve(Cd,d))
    except np.linalg.LinAlgError as exc:
        raise ValueError('singular or indefinite covariance') from exc
    return float(criterion),d,Cd

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
      'call': 'cv2l_score(p,f,y,nu,.45)',
      'gold_call': '_oracle_cv2l_score(p,f,y,nu,.45)'},
     {'setup': 'import numpy as np\n'
               'f=np.linspace(.01,1.,48);p=np.array([1.4,.25,8.])\n'
               'y=(1.3+.3*f*f)/(1+7*f*f)*(1+.09*np.sin(np.arange(len(f))*1.7))\n'
               'nu=np.full(len(f),18.);nu[-1]=9.\n'
               'w=1/(1+(f/.45)**8)\n'
               'y=(p[0]+p[1]*f*f)/(1+p[2]*f*f)\n',
      'call': 'cv2l_score(p,f,y,nu,.45)',
      'gold_call': '_oracle_cv2l_score(p,f,y,nu,.45)'},
     {'setup': 'import numpy as np\n'
               'f=np.linspace(.01,1.,48);p=np.array([1.4,.25,8.])\n'
               'y=(1.3+.3*f*f)/(1+7*f*f)*(1+.09*np.sin(np.arange(len(f))*1.7))\n'
               'nu=np.full(len(f),18.);nu[-1]=9.\n'
               'w=1/(1+(f/.45)**8)\n',
      'call': 'cv2l_score(p,f,y,nu,.22)',
      'gold_call': '_oracle_cv2l_score(p,f,y,nu,.22)'},
     {'setup': 'import numpy as np\n'
               'f=np.linspace(.01,1.,48);p=np.array([1.4,.25,8.])\n'
               'y=(1.3+.3*f*f)/(1+7*f*f)*(1+.09*np.sin(np.arange(len(f))*1.7))\n'
               'nu=np.full(len(f),18.);nu[-1]=9.\n'
               'w=1/(1+(f/.45)**8)\n'
               'nu=np.linspace(2.,20.,len(f))\n',
      'call': 'cv2l_score(p,f,y,nu,.45)',
      'gold_call': '_oracle_cv2l_score(p,f,y,nu,.45)'},
     {'setup': 'import numpy as np\n'
               'f=np.linspace(.01,1.,48);p=np.array([1.4,.25,8.])\n'
               'y=(1.3+.3*f*f)/(1+7*f*f)*(1+.09*np.sin(np.arange(len(f))*1.7))\n'
               'nu=np.full(len(f),18.);nu[-1]=9.\n'
               'w=1/(1+(f/.45)**8)\n'
               'p[:2]*=1e-3;y*=1e-3\n',
      'call': 'cv2l_score(p,f,y,nu,.45)',
      'gold_call': '_oracle_cv2l_score(p,f,y,nu,.45)'},
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
               'p[2]=-2\n',
      'call': '_error_code(lambda:cv2l_score(p,f,y,nu,.45))',
      'gold_call': '_error_code(lambda:_oracle_cv2l_score(p,f,y,nu,.45))'},
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
      'call': '_error_code(lambda: cv2l_score(p,f,y,nu,.45))',
      'gold_call': '_error_code(lambda: _oracle_cv2l_score(p,f,y,nu,.45))'}]
