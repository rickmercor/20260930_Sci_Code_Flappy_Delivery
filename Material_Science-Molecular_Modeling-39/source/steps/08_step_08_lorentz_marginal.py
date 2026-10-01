"""
Transform physical fits to Lorentz variables and marginalize the cutoffs.

Map Pade fits to a slow Lorentzian plus white background, reject unresolved relaxation times, then combine frequency cutoffs with CV2L and relative-uncertainty penalties.

Returns
-------
Tuple (mean,Cmix,weights,ratios,valid): shapes (4,), (4,4), (J,), (J,), (J,); valid contains integer 0 or 1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def lorentz_marginal(parameters: 'np.ndarray', covariance: 'np.ndarray', criteria: 'np.ndarray') -> 'tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]':
    """Return (mean,mixture_covariance,weights,ratios,valid) across cutoffs.
    
    Parameters
    ----------
    parameters, covariance, criteria : numpy.ndarray
        Real finite cutoff-fit parameters, covariances, and CV criteria.

    parameters has shape (J,3), J>=1, with rows (p0,p2,q2). covariance has
    shape (J,3,3); every matrix is finite, symmetric positive definite;
    criteria is a finite vector (J,). All parameters are finite.
    For a physical row require p0>0,q2>0,p0*q2>p2 (strict inequalities).
    Transform to v=(I,tau,C0,C1), where I=p0, tau=sqrt(q2)/(2*pi),
    C0=p2/q2, C1=pi*(p0-p2/q2)/sqrt(q2).
    At each physical row map the full covariance by T=(dv/dp): S=T@C@T.T.
    R=(sqrt(S[1,1])/tau)/(sqrt(S[0,0])/I).
    Retain iff R<=100, including equality. Negative C0 is allowed by this
    diagnostic contract. Invalid rows get ratio=0 and valid=0; physical rows
    with R>100 keep their R but get valid=0. Output valid is integer 0 or 1.
    Normalize weights exp(-criterion-R) over retained rows using log-sum-exp;
    rejected rows get zero. mean=sum_j Wj*vj; mixture covariance is
    sum_j Wj*(Sj+outer(vj-mean,vj-mean)). Transform each row before averaging.
    Shapes: (4,), (4,4), (J,), (J,), (J,). Only numeric outputs are returned.
    
    Returns
    -------
    result
        Tuple (mean,Cmix,weights,ratios,valid): shapes (4,), (4,4), (J,), (J,), (J,); valid contains integer 0 or 1.
    
    Raises
    ------
    ValueError
        If any input contains complex values, including a complex dtype with
        zero imaginary part.
        If any stated shape/finiteness condition fails, any covariance is not
        symmetric within absolute 1e-12 or is not positive definite, or no
        row satisfies both the physical conditions and R<=100.
    
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

def _oracle_lorentz_marginal(parameters: 'np.ndarray', covariance: 'np.ndarray', criteria: 'np.ndarray') -> 'tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]':
    import numpy as np

    for _name, _value in (('parameters', parameters), ('covariance', covariance), ('criteria', criteria),):
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
        p=np.asarray(parameters,dtype=float)
    except (ValueError,TypeError,OverflowError) as exc:
        raise ValueError('parameters') from exc
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<1 or not np.isfinite(p).all():
        raise ValueError('parameters')
    J=len(p); C=_array(covariance,(J,3,3),'covariance'); crit=_array(criteria,(J,),'criteria')
    if not np.allclose(C,C.swapaxes(1,2),atol=1e-12,rtol=0):raise ValueError('covariance symmetry')
    try:np.linalg.cholesky(C)
    except np.linalg.LinAlgError as exc:raise ValueError('covariance') from exc
    values=np.zeros((J,4)); cov=np.zeros((J,4,4)); ratios=np.zeros(J); valid=np.zeros(J,dtype=int)
    for j,(a,b,c) in enumerate(p):
        if a<=0 or c<=0 or a*c<=b:continue
        tau=np.sqrt(c)/(2*np.pi); C0=b/c; C1=np.pi*(a-C0)/np.sqrt(c)
        T=np.zeros((4,3)); T[0,0]=1; T[1,2]=1/(4*np.pi*np.sqrt(c))
        T[2,1]=1/c; T[2,2]=-b/c**2
        T[3]=[np.pi/np.sqrt(c),-np.pi/c**1.5,np.pi*(-0.5*a/c**1.5+1.5*b/c**2.5)]
        values[j]=[a,tau,C0,C1]; cov[j]=T@C[j]@T.T
        ratios[j]=(np.sqrt(cov[j,1,1])/tau)/(np.sqrt(cov[j,0,0])/a)
        valid[j]=int(ratios[j]<=100)
    ids=np.flatnonzero(valid)
    if len(ids)==0:raise ValueError('no admissible Lorentz fit')
    logs=-crit[ids]-ratios[ids]; w=np.zeros(J)
    w[ids]=np.exp(logs-logs.max()); w/=w.sum()
    mean=np.sum(w[:,None]*values,axis=0)
    centered=values-mean
    mixture=np.einsum('j,jab->ab',w,cov)+np.einsum('j,ja,jb->ab',w,centered,centered)
    return mean,mixture,w,ratios,valid

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Numerical cases and explicitly declared invalid inputs."""
    return [{'setup': 'import numpy as np\n'
               'p=np.array([[1.2,.3,4.],[1.1,.2,5.],[1.4,.4,6.]])\n'
               'L=np.array([[.10,0.,0.],[.02,.08,0.],[-.04,.03,.4]])\n'
               'C=np.stack([L@L.T,(L@L.T)*1.3,(L@L.T)*.8])\n'
               'crit=np.array([1.1,.6,1.4])\n',
      'call': 'lorentz_marginal(p,C,crit)',
      'gold_call': '_oracle_lorentz_marginal(p,C,crit)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([[1.2,.3,4.],[1.1,.2,5.],[1.4,.4,6.]])\n'
               'L=np.array([[.10,0.,0.],[.02,.08,0.],[-.04,.03,.4]])\n'
               'C=np.stack([L@L.T,(L@L.T)*1.3,(L@L.T)*.8])\n'
               'crit=np.array([1.1,.6,1.4])\n'
               'p=p[:1];C=C[:1];crit=crit[:1]\n',
      'call': 'lorentz_marginal(p,C,crit)',
      'gold_call': '_oracle_lorentz_marginal(p,C,crit)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([[1.2,.3,4.],[1.1,.2,5.],[1.4,.4,6.]])\n'
               'L=np.array([[.10,0.,0.],[.02,.08,0.],[-.04,.03,.4]])\n'
               'C=np.stack([L@L.T,(L@L.T)*1.3,(L@L.T)*.8])\n'
               'crit=np.array([1.1,.6,1.4])\n'
               'p[1,2]=-1;p[2,1]=p[2,0]*p[2,2]\n',
      'call': 'lorentz_marginal(p,C,crit)',
      'gold_call': '_oracle_lorentz_marginal(p,C,crit)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([[1.2,.3,4.],[1.1,.2,5.],[1.4,.4,6.]])\n'
               'L=np.array([[.10,0.,0.],[.02,.08,0.],[-.04,.03,.4]])\n'
               'C=np.stack([L@L.T,(L@L.T)*1.3,(L@L.T)*.8])\n'
               'crit=np.array([1.1,.6,1.4])\n'
               'p[1]=[1.,.1,1.];C[1]=np.diag([1e-4,.001,4.])\n',
      'call': 'lorentz_marginal(p,C,crit)',
      'gold_call': '_oracle_lorentz_marginal(p,C,crit)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([[1.2,.3,4.],[1.1,.2,5.],[1.4,.4,6.]])\n'
               'L=np.array([[.10,0.,0.],[.02,.08,0.],[-.04,.03,.4]])\n'
               'C=np.stack([L@L.T,(L@L.T)*1.3,(L@L.T)*.8])\n'
               'crit=np.array([1.1,.6,1.4])\n'
               'p[1]=[1.,.1,1.];C[1]=np.diag([1e-4,.001,4.001])\n',
      'call': 'lorentz_marginal(p,C,crit)',
      'gold_call': '_oracle_lorentz_marginal(p,C,crit)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([[1.2,.3,4.],[1.1,.2,5.],[1.4,.4,6.]])\n'
               'L=np.array([[.10,0.,0.],[.02,.08,0.],[-.04,.03,.4]])\n'
               'C=np.stack([L@L.T,(L@L.T)*1.3,(L@L.T)*.8])\n'
               'crit=np.array([1.1,.6,1.4])\n'
               'crit+=10000.\n',
      'call': 'lorentz_marginal(p,C,crit)',
      'gold_call': '_oracle_lorentz_marginal(p,C,crit)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([[1.2,.3,4.],[1.1,.2,5.],[1.4,.4,6.]])\n'
               'L=np.array([[.10,0.,0.],[.02,.08,0.],[-.04,.03,.4]])\n'
               'C=np.stack([L@L.T,(L@L.T)*1.3,(L@L.T)*.8])\n'
               'crit=np.array([1.1,.6,1.4])\n'
               'p[1,1]=-.2\n',
      'call': 'lorentz_marginal(p,C,crit)',
      'gold_call': '_oracle_lorentz_marginal(p,C,crit)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([[1.2,.3,4.],[1.1,.2,5.],[1.4,.4,6.]])\n'
               'L=np.array([[.10,0.,0.],[.02,.08,0.],[-.04,.03,.4]])\n'
               'C=np.stack([L@L.T,(L@L.T)*1.3,(L@L.T)*.8])\n'
               'crit=np.array([1.1,.6,1.4])\n'
               '\n'
               'def _error_code(thunk):\n'
               '    try:thunk()\n'
               '    except ValueError:return 1\n'
               '    except Exception:return 2\n'
               '    return 0\n'
               'p[:,2]=-1\n',
      'call': '_error_code(lambda:lorentz_marginal(p,C,crit))',
      'gold_call': '_error_code(lambda:_oracle_lorentz_marginal(p,C,crit))'},
     {'setup': 'import numpy as np\n'
               'p=np.array([[1.2,.3,4.],[1.1,.2,5.],[1.4,.4,6.]])\n'
               'L=np.array([[.10,0.,0.],[.02,.08,0.],[-.04,.03,.4]])\n'
               'C=np.stack([L@L.T,(L@L.T)*1.3,(L@L.T)*.8])\n'
               'crit=np.array([1.1,.6,1.4])\n'
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
      'call': '_error_code(lambda: lorentz_marginal(p,C,crit))',
      'gold_call': '_error_code(lambda: _oracle_lorentz_marginal(p,C,crit))'}]
