"""
Solve the feature-dependent detection subproblem for a prescribed number of ADMM iterations.

The bounded detection auxiliary and the feature projection can disagree before convergence. Stable quadratic minimization, row ordering, and the finite update order therefore matter.

Returns
-------
P, alpha_new, omega : tuple of ndarrays     P has shape (I,J); alpha_new shape (R,); omega shape (I*J,).     Reset omega=0 on entry. Each iteration uses b=Z@alpha-omega,     d=eta*b-replicates*intensity.ravel(order='C'), then     p=clip((d+sqrt(d*d+4*eta*S.ravel()))/(2*eta),0,1),     alpha=pinv(Z)@(p+omega), omega=omega+p-Z@alpha, in this order.     Use Moore-Penrose pinv with relative singular cutoff 1e-15.     Return auxiliary p, NOT Z@alpha; there is no convergence stopping.     Evaluate the quadratic root without catastrophic cancellation.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def detection_update(S: 'np.ndarray', intensity: 'np.ndarray', Z: 'np.ndarray', alpha: 'np.ndarray', replicates: int, steps: int = 80, penalty: float = 1.0) -> tuple:
    """Perform a finite constrained detection ADMM solve.

    Parameters
    ----------
    S, intensity : ndarrays, common shape (I,J), I,J >= 1
        Nonnegative finite aggregate counts (S may be real) and latent means.
        Positive S requires strictly positive intensity.
    Z : ndarray, shape (I*J,R), R >= 1
        Finite real design; row-major pair order. Rank deficiency is allowed.
    alpha : ndarray, shape (R,)
        Finite starting coefficients, with no sign constraint.
    replicates, steps : int
        Positive integers, booleans excluded. Execute exactly steps updates.
    penalty : float
        Finite strictly positive inner ADMM penalty eta.

    Returns
    -------
    P, alpha_new, omega : tuple of ndarrays
        P has shape (I,J); alpha_new shape (R,); omega shape (I*J,).
        Reset omega=0 on entry. Each iteration uses b=Z@alpha-omega,
        d=eta*b-replicates*intensity.ravel(order='C'), then
        p=clip((d+sqrt(d*d+4*eta*S.ravel()))/(2*eta),0,1),
        alpha=pinv(Z)@(p+omega), omega=omega+p-Z@alpha, in this order.
        Use Moore-Penrose pinv with relative singular cutoff 1e-15.
        Return auxiliary p, NOT Z@alpha; there is no convergence stopping.
        Evaluate the quadratic root without catastrophic cancellation.

    Raises
    ------
    ValueError
        Non-real/nonnumeric input; inconsistent or empty shapes; nonfinite
        entries; negative counts/intensities; zero intensity at positive count;
        invalid integer controls or penalty; failed pseudoinverse; or a
        nonfinite intermediate/result.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_detection_update(S: 'np.ndarray', intensity: 'np.ndarray', Z: 'np.ndarray', alpha: 'np.ndarray', replicates: int, steps: int = 80, penalty: float = 1.0) -> tuple:
    """Perform a finite constrained detection ADMM solve.

    Parameters
    ----------
    S, intensity : ndarrays, common shape (I,J), I,J >= 1
        Nonnegative finite aggregate counts (S may be real) and latent means.
        Positive S requires strictly positive intensity.
    Z : ndarray, shape (I*J,R), R >= 1
        Finite real design; row-major pair order. Rank deficiency is allowed.
    alpha : ndarray, shape (R,)
        Finite starting coefficients, with no sign constraint.
    replicates, steps : int
        Positive integers, booleans excluded. Execute exactly steps updates.
    penalty : float
        Finite strictly positive inner ADMM penalty eta.

    Returns
    -------
    P, alpha_new, omega : tuple of ndarrays
        P has shape (I,J); alpha_new shape (R,); omega shape (I*J,).
        Reset omega=0 on entry. Each iteration uses b=Z@alpha-omega,
        d=eta*b-replicates*intensity.ravel(order='C'), then
        p=clip((d+sqrt(d*d+4*eta*S.ravel()))/(2*eta),0,1),
        alpha=pinv(Z)@(p+omega), omega=omega+p-Z@alpha, in this order.
        Use Moore-Penrose pinv with relative singular cutoff 1e-15.
        Return auxiliary p, NOT Z@alpha; there is no convergence stopping.
        Evaluate the quadratic root without catastrophic cancellation.

    Raises
    ------
    ValueError
        Non-real/nonnumeric input; inconsistent or empty shapes; nonfinite
        entries; negative counts/intensities; zero intensity at positive count;
        invalid integer controls or penalty; failed pseudoinverse; or a
        nonfinite intermediate/result.
    """
    import numpy as np
    try:
        if any(np.iscomplexobj(a) for a in (S,intensity,Z,alpha,penalty)): raise ValueError('real inputs required')
        S=np.asarray(S,float); intensity=np.asarray(intensity,float); Z=np.asarray(Z,float); alpha=np.array(alpha,dtype=float,copy=True); penalty=float(penalty)
    except (TypeError,ValueError,OverflowError) as e: raise ValueError('numeric inputs required') from e
    if S.ndim!=2 or min(S.shape)<1 or intensity.shape!=S.shape or Z.ndim!=2 or Z.shape[0]!=S.size or Z.shape[1]<1 or alpha.shape!=(Z.shape[1],): raise ValueError('invalid shapes')
    if any(not np.isfinite(a).all() for a in (S,intensity,Z,alpha)) or np.any(S<0) or np.any(intensity<0) or np.any(intensity[S>0]<=0): raise ValueError('invalid data')
    if any(isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)) or n<1 for n in (replicates,steps)) or not np.isfinite(penalty) or penalty<=0: raise ValueError('invalid controls')
    try: inverse=np.linalg.pinv(Z,rcond=1e-15)
    except np.linalg.LinAlgError as e: raise ValueError('pseudoinverse failed') from e
    y=S.ravel(); omega=np.zeros(S.size)
    for _ in range(steps):
        d=penalty*(Z@alpha-omega)-replicates*intensity.ravel()
        root=np.hypot(d,2*np.sqrt(penalty)*np.sqrt(y))
        p=np.empty_like(y); positive=d>=0
        p[positive]=(d[positive]+root[positive])/(2*penalty)
        p[~positive]=2*y[~positive]/(root[~positive]-d[~positive])
        p=np.clip(p,0,1); alpha=inverse@(p+omega); omega=omega+p-Z@alpha
        if any(not np.isfinite(a).all() for a in (d,root,p,alpha,omega)): raise ValueError('nonfinite detection iterate')
    return p.reshape(S.shape),alpha,omega

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'Y=np.array([[[0,1,0],[3,5,4],[0,0,0]],[[2,1,3],[0,0,1],[4,3,5]],[[6,4,5],[1,0,1],[0,1,0]],[[0,0,0],[2,4,3],[5,6,4]]],dtype=float)\n'
               'S=Y.sum(axis=2)\n'
               'x=np.array([-.9,-.2,.4,1.]); z=np.array([-.8,.1,.9])\n'
               'Z=np.array([[1+.2*xi,.5+.1*zj] for xi in x for zj in z])\n'
               'alpha=np.array([.4,.3]); weights=np.array([.8,1.1,.9]); rho=np.array([.08,.06,.1])\n'
               'U=np.array([[1.,.4],[.2,1.1],[.8,.3],[.4,.7]])\n'
               'V=np.array([[.6,.8],[1.2,.2],[.3,1.]])\n'
               'P=np.full((4,3),.6)\n'
               'products=(U@U.T,U@V.T,V@V.T)\n'
               'A=tuple(a.copy() for a in products)\n'
               'W=tuple(np.zeros_like(a) for a in products)\n'
               'B=tuple(a+.2 for a in products)\n'
               '\n'
               'def _pack_numeric(value):\n'
               '    import numpy as np\n'
               '    if isinstance(value, (tuple, list)):\n'
               '        chunks = [np.array([2., float(len(value))])]\n'
               '        for item in value:\n'
               '            packed = _pack_numeric(item)\n'
               '            chunks.extend((np.array([float(packed.size)]), packed))\n'
               '        return np.concatenate(chunks)\n'
               '    array = np.asarray(value, dtype=float)\n'
               '    return np.concatenate((np.array([1., float(array.ndim)]),\n'
               '                           np.asarray(array.shape, dtype=float),\n'
               '                           array.reshape(-1)))\n',
      'call': '_pack_numeric(detection_update(S, U@V.T, Z, alpha, 3))',
      'gold_call': '_pack_numeric(_oracle_detection_update(S, U@V.T, Z, alpha, 3))',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'Y=np.array([[[0,1,0],[3,5,4],[0,0,0]],[[2,1,3],[0,0,1],[4,3,5]],[[6,4,5],[1,0,1],[0,1,0]],[[0,0,0],[2,4,3],[5,6,4]]],dtype=float)\n'
               'S=Y.sum(axis=2)\n'
               'x=np.array([-.9,-.2,.4,1.]); z=np.array([-.8,.1,.9])\n'
               'Z=np.array([[1+.2*xi,.5+.1*zj] for xi in x for zj in z])\n'
               'alpha=np.array([.4,.3]); weights=np.array([.8,1.1,.9]); rho=np.array([.08,.06,.1])\n'
               'U=np.array([[1.,.4],[.2,1.1],[.8,.3],[.4,.7]])\n'
               'V=np.array([[.6,.8],[1.2,.2],[.3,1.]])\n'
               'P=np.full((4,3),.6)\n'
               'products=(U@U.T,U@V.T,V@V.T)\n'
               'A=tuple(a.copy() for a in products)\n'
               'W=tuple(np.zeros_like(a) for a in products)\n'
               'B=tuple(a+.2 for a in products)\n'
               'S=np.zeros((4,3))\n'
               '\n'
               'def _pack_numeric(value):\n'
               '    import numpy as np\n'
               '    if isinstance(value, (tuple, list)):\n'
               '        chunks = [np.array([2., float(len(value))])]\n'
               '        for item in value:\n'
               '            packed = _pack_numeric(item)\n'
               '            chunks.extend((np.array([float(packed.size)]), packed))\n'
               '        return np.concatenate(chunks)\n'
               '    array = np.asarray(value, dtype=float)\n'
               '    return np.concatenate((np.array([1., float(array.ndim)]),\n'
               '                           np.asarray(array.shape, dtype=float),\n'
               '                           array.reshape(-1)))\n',
      'call': '_pack_numeric(detection_update(S, U@V.T, Z, alpha, 3))',
      'gold_call': '_pack_numeric(_oracle_detection_update(S, U@V.T, Z, alpha, 3))',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'Y=np.array([[[0,1,0],[3,5,4],[0,0,0]],[[2,1,3],[0,0,1],[4,3,5]],[[6,4,5],[1,0,1],[0,1,0]],[[0,0,0],[2,4,3],[5,6,4]]],dtype=float)\n'
               'S=Y.sum(axis=2)\n'
               'x=np.array([-.9,-.2,.4,1.]); z=np.array([-.8,.1,.9])\n'
               'Z=np.array([[1+.2*xi,.5+.1*zj] for xi in x for zj in z])\n'
               'alpha=np.array([.4,.3]); weights=np.array([.8,1.1,.9]); rho=np.array([.08,.06,.1])\n'
               'U=np.array([[1.,.4],[.2,1.1],[.8,.3],[.4,.7]])\n'
               'V=np.array([[.6,.8],[1.2,.2],[.3,1.]])\n'
               'P=np.full((4,3),.6)\n'
               'products=(U@U.T,U@V.T,V@V.T)\n'
               'A=tuple(a.copy() for a in products)\n'
               'W=tuple(np.zeros_like(a) for a in products)\n'
               'B=tuple(a+.2 for a in products)\n'
               'S=np.array([[1e4]])\n'
               'U=np.array([[1e12]])\n'
               'V=np.ones((1,1))\n'
               'Z=np.ones((1,1))\n'
               'alpha=np.zeros(1)\n'
               '\n'
               'def _pack_numeric(value):\n'
               '    import numpy as np\n'
               '    if isinstance(value, (tuple, list)):\n'
               '        chunks = [np.array([2., float(len(value))])]\n'
               '        for item in value:\n'
               '            packed = _pack_numeric(item)\n'
               '            chunks.extend((np.array([float(packed.size)]), packed))\n'
               '        return np.concatenate(chunks)\n'
               '    array = np.asarray(value, dtype=float)\n'
               '    return np.concatenate((np.array([1., float(array.ndim)]),\n'
               '                           np.asarray(array.shape, dtype=float),\n'
               '                           array.reshape(-1)))\n',
      'call': '_pack_numeric(detection_update(S, U@V.T, Z, alpha, 1, 1))',
      'gold_call': '_pack_numeric(_oracle_detection_update(S, U@V.T, Z, alpha, 1, 1))',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'Y=np.array([[[0,1,0],[3,5,4],[0,0,0]],[[2,1,3],[0,0,1],[4,3,5]],[[6,4,5],[1,0,1],[0,1,0]],[[0,0,0],[2,4,3],[5,6,4]]],dtype=float)\n'
               'S=Y.sum(axis=2)\n'
               'x=np.array([-.9,-.2,.4,1.]); z=np.array([-.8,.1,.9])\n'
               'Z=np.array([[1+.2*xi,.5+.1*zj] for xi in x for zj in z])\n'
               'alpha=np.array([.4,.3]); weights=np.array([.8,1.1,.9]); rho=np.array([.08,.06,.1])\n'
               'U=np.array([[1.,.4],[.2,1.1],[.8,.3],[.4,.7]])\n'
               'V=np.array([[.6,.8],[1.2,.2],[.3,1.]])\n'
               'P=np.full((4,3),.6)\n'
               'products=(U@U.T,U@V.T,V@V.T)\n'
               'A=tuple(a.copy() for a in products)\n'
               'W=tuple(np.zeros_like(a) for a in products)\n'
               'B=tuple(a+.2 for a in products)\n'
               'Z=np.column_stack((np.ones(12),np.ones(12)))\n'
               '\n'
               'def _pack_numeric(value):\n'
               '    import numpy as np\n'
               '    if isinstance(value, (tuple, list)):\n'
               '        chunks = [np.array([2., float(len(value))])]\n'
               '        for item in value:\n'
               '            packed = _pack_numeric(item)\n'
               '            chunks.extend((np.array([float(packed.size)]), packed))\n'
               '        return np.concatenate(chunks)\n'
               '    array = np.asarray(value, dtype=float)\n'
               '    return np.concatenate((np.array([1., float(array.ndim)]),\n'
               '                           np.asarray(array.shape, dtype=float),\n'
               '                           array.reshape(-1)))\n',
      'call': '_pack_numeric(detection_update(S, U@V.T, Z, alpha, 3, 1))',
      'gold_call': '_pack_numeric(_oracle_detection_update(S, U@V.T, Z, alpha, 3, 1))',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'Y=np.array([[[0,1,0],[3,5,4],[0,0,0]],[[2,1,3],[0,0,1],[4,3,5]],[[6,4,5],[1,0,1],[0,1,0]],[[0,0,0],[2,4,3],[5,6,4]]],dtype=float)\n'
               'S=Y.sum(axis=2)\n'
               'x=np.array([-.9,-.2,.4,1.]); z=np.array([-.8,.1,.9])\n'
               'Z=np.array([[1+.2*xi,.5+.1*zj] for xi in x for zj in z])\n'
               'alpha=np.array([.4,.3]); weights=np.array([.8,1.1,.9]); rho=np.array([.08,.06,.1])\n'
               'U=np.array([[1.,.4],[.2,1.1],[.8,.3],[.4,.7]])\n'
               'V=np.array([[.6,.8],[1.2,.2],[.3,1.]])\n'
               'P=np.full((4,3),.6)\n'
               'products=(U@U.T,U@V.T,V@V.T)\n'
               'A=tuple(a.copy() for a in products)\n'
               'W=tuple(np.zeros_like(a) for a in products)\n'
               'B=tuple(a+.2 for a in products)\n'
               '\n'
               'def _pack_numeric(value):\n'
               '    import numpy as np\n'
               '    if isinstance(value, (tuple, list)):\n'
               '        chunks = [np.array([2., float(len(value))])]\n'
               '        for item in value:\n'
               '            packed = _pack_numeric(item)\n'
               '            chunks.extend((np.array([float(packed.size)]), packed))\n'
               '        return np.concatenate(chunks)\n'
               '    array = np.asarray(value, dtype=float)\n'
               '    return np.concatenate((np.array([1., float(array.ndim)]),\n'
               '                           np.asarray(array.shape, dtype=float),\n'
               '                           array.reshape(-1)))\n',
      'call': '_pack_numeric(detection_update(S, .02*(U@V.T), Z, alpha, 5, 7, .3))',
      'gold_call': '_pack_numeric(_oracle_detection_update(S, .02*(U@V.T), Z, alpha, 5, 7, .3))',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'Y=np.array([[[0,1,0],[3,5,4],[0,0,0]],[[2,1,3],[0,0,1],[4,3,5]],[[6,4,5],[1,0,1],[0,1,0]],[[0,0,0],[2,4,3],[5,6,4]]],dtype=float)\n'
               'S=Y.sum(axis=2)\n'
               'x=np.array([-.9,-.2,.4,1.]); z=np.array([-.8,.1,.9])\n'
               'Z=np.array([[1+.2*xi,.5+.1*zj] for xi in x for zj in z])\n'
               'alpha=np.array([.4,.3]); weights=np.array([.8,1.1,.9]); rho=np.array([.08,.06,.1])\n'
               'U=np.array([[1.,.4],[.2,1.1],[.8,.3],[.4,.7]])\n'
               'V=np.array([[.6,.8],[1.2,.2],[.3,1.]])\n'
               'P=np.full((4,3),.6)\n'
               'products=(U@U.T,U@V.T,V@V.T)\n'
               'A=tuple(a.copy() for a in products)\n'
               'W=tuple(np.zeros_like(a) for a in products)\n'
               'B=tuple(a+.2 for a in products)\n'
               'S=np.zeros((4,3))\n'
               '\n'
               'def _pack_numeric(value):\n'
               '    import numpy as np\n'
               '    if isinstance(value, (tuple, list)):\n'
               '        chunks = [np.array([2., float(len(value))])]\n'
               '        for item in value:\n'
               '            packed = _pack_numeric(item)\n'
               '            chunks.extend((np.array([float(packed.size)]), packed))\n'
               '        return np.concatenate(chunks)\n'
               '    array = np.asarray(value, dtype=float)\n'
               '    return np.concatenate((np.array([1., float(array.ndim)]),\n'
               '                           np.asarray(array.shape, dtype=float),\n'
               '                           array.reshape(-1)))\n',
      'call': '_pack_numeric(detection_update(S, np.zeros((4,3)), Z, alpha, 3, 2))',
      'gold_call': '_pack_numeric(_oracle_detection_update(S, np.zeros((4,3)), Z, alpha, 3, 2))',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'Y=np.array([[[0,1,0],[3,5,4],[0,0,0]],[[2,1,3],[0,0,1],[4,3,5]],[[6,4,5],[1,0,1],[0,1,0]],[[0,0,0],[2,4,3],[5,6,4]]],dtype=float)\n'
               'S=Y.sum(axis=2)\n'
               'x=np.array([-.9,-.2,.4,1.]); z=np.array([-.8,.1,.9])\n'
               'Z=np.array([[1+.2*xi,.5+.1*zj] for xi in x for zj in z])\n'
               'alpha=np.array([.4,.3]); weights=np.array([.8,1.1,.9]); rho=np.array([.08,.06,.1])\n'
               'U=np.array([[1.,.4],[.2,1.1],[.8,.3],[.4,.7]])\n'
               'V=np.array([[.6,.8],[1.2,.2],[.3,1.]])\n'
               'P=np.full((4,3),.6)\n'
               'products=(U@U.T,U@V.T,V@V.T)\n'
               'A=tuple(a.copy() for a in products)\n'
               'W=tuple(np.zeros_like(a) for a in products)\n'
               'B=tuple(a+.2 for a in products)\n'
               '\n'
               'def _exception_code(function,*args,**kwargs):\n'
               '    try:\n'
               '        function(*args,**kwargs)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_exception_code(detection_update, S, np.zeros((4,3)), Z, alpha, 3)',
      'gold_call': '_exception_code(_oracle_detection_update, S, np.zeros((4,3)), Z, alpha, 3)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               '\n'
               'def _pack_numeric(value):\n'
               '    import numpy as np\n'
               '    if isinstance(value, (tuple, list)):\n'
               '        chunks = [np.array([2., float(len(value))])]\n'
               '        for item in value:\n'
               '            packed = _pack_numeric(item)\n'
               '            chunks.extend((np.array([float(packed.size)]), packed))\n'
               '        return np.concatenate(chunks)\n'
               '    array = np.asarray(value, dtype=float)\n'
               '    return np.concatenate((np.array([1., float(array.ndim)]),\n'
               '                           np.asarray(array.shape, dtype=float),\n'
               '                           array.reshape(-1)))\n',
      'call': '_pack_numeric(bool(abs(float(detection_update(np.array([[1e4]]),np.array([[1e12]]),np.ones((1,1)),np.zeros(1),1,1)[0][0,0])/1e-8-1.0)<1e-10))',
      'gold_call': '_pack_numeric(bool(abs(float(_oracle_detection_update(np.array([[1e4]]),np.array([[1e12]]),np.ones((1,1)),np.zeros(1),1,1)[0][0,0])/1e-8-1.0)<1e-10))',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               '\n'
               'def _pack_numeric(value):\n'
               '    import numpy as np\n'
               '    if isinstance(value, (tuple, list)):\n'
               '        chunks = [np.array([2., float(len(value))])]\n'
               '        for item in value:\n'
               '            packed = _pack_numeric(item)\n'
               '            chunks.extend((np.array([float(packed.size)]), packed))\n'
               '        return np.concatenate(chunks)\n'
               '    array = np.asarray(value, dtype=float)\n'
               '    return np.concatenate((np.array([1., float(array.ndim)]),\n'
               '                           np.asarray(array.shape, dtype=float),\n'
               '                           array.reshape(-1)))\n',
      'call': '_pack_numeric(bool(abs(float(detection_update(np.array([[1e120]]),np.array([[1e200]]),np.ones((1,1)),np.zeros(1),1,1)[0][0,0])/1e-80-1.0)<1e-10))',
      'gold_call': '_pack_numeric(bool(abs(float(_oracle_detection_update(np.array([[1e120]]),np.array([[1e200]]),np.ones((1,1)),np.zeros(1),1,1)[0][0,0])/1e-80-1.0)<1e-10))',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               '\n'
               'def _pack_numeric(value):\n'
               '    import numpy as np\n'
               '    if isinstance(value, (tuple, list)):\n'
               '        chunks = [np.array([2., float(len(value))])]\n'
               '        for item in value:\n'
               '            packed = _pack_numeric(item)\n'
               '            chunks.extend((np.array([float(packed.size)]), packed))\n'
               '        return np.concatenate(chunks)\n'
               '    array = np.asarray(value, dtype=float)\n'
               '    return np.concatenate((np.array([1., float(array.ndim)]),\n'
               '                           np.asarray(array.shape, dtype=float),\n'
               '                           array.reshape(-1)))\n',
      'call': '_pack_numeric(bool(np.allclose(detection_update(np.array([[0.,0.,1.]]),np.array([[0.,0.,1.]]),np.eye(3),np.array([-.5,.4,4.]),1,1)[0],np.array([[0.,.4,1.]]),rtol=0,atol=1e-12)))',
      'gold_call': '_pack_numeric(bool(np.allclose(_oracle_detection_update(np.array([[0.,0.,1.]]),np.array([[0.,0.,1.]]),np.eye(3),np.array([-.5,.4,4.]),1,1)[0],np.array([[0.,.4,1.]]),rtol=0,atol=1e-12)))',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               '\n'
               'def _pack_numeric(value):\n'
               '    import numpy as np\n'
               '    if isinstance(value, (tuple, list)):\n'
               '        chunks = [np.array([2., float(len(value))])]\n'
               '        for item in value:\n'
               '            packed = _pack_numeric(item)\n'
               '            chunks.extend((np.array([float(packed.size)]), packed))\n'
               '        return np.concatenate(chunks)\n'
               '    array = np.asarray(value, dtype=float)\n'
               '    return np.concatenate((np.array([1., float(array.ndim)]),\n'
               '                           np.asarray(array.shape, dtype=float),\n'
               '                           array.reshape(-1)))\n',
      'call': '_pack_numeric(detection_update(np.array([[1.,0.],[4.,2.]]),np.array([[.3,.8],[1.1,.4]]),np.array([[1.,2.],[-1.,-2.],[.5,1.],[2.,4.]]),np.array([.2,-.1]),4,3,.7))',
      'gold_call': '_pack_numeric(_oracle_detection_update(np.array([[1.,0.],[4.,2.]]),np.array([[.3,.8],[1.1,.4]]),np.array([[1.,2.],[-1.,-2.],[.5,1.],[2.,4.]]),np.array([.2,-.1]),4,3,.7))',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               '\n'
               'def _pack_numeric(value):\n'
               '    import numpy as np\n'
               '    if isinstance(value, (tuple, list)):\n'
               '        chunks = [np.array([2., float(len(value))])]\n'
               '        for item in value:\n'
               '            packed = _pack_numeric(item)\n'
               '            chunks.extend((np.array([float(packed.size)]), packed))\n'
               '        return np.concatenate(chunks)\n'
               '    array = np.asarray(value, dtype=float)\n'
               '    return np.concatenate((np.array([1., float(array.ndim)]),\n'
               '                           np.asarray(array.shape, dtype=float),\n'
               '                           array.reshape(-1)))\n',
      'call': '_pack_numeric(bool(np.allclose(detection_update(np.array([[.1875]]),np.array([[.25]]),np.ones((1,1)),np.zeros(1),2,1)[0],np.array([[.25]]),rtol=0,atol=1e-12)))',
      'gold_call': '_pack_numeric(bool(np.allclose(_oracle_detection_update(np.array([[.1875]]),np.array([[.25]]),np.ones((1,1)),np.zeros(1),2,1)[0],np.array([[.25]]),rtol=0,atol=1e-12)))',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               '\n'
               'def _pack_numeric(value):\n'
               '    import numpy as np\n'
               '    if isinstance(value, (tuple, list)):\n'
               '        chunks = [np.array([2., float(len(value))])]\n'
               '        for item in value:\n'
               '            packed = _pack_numeric(item)\n'
               '            chunks.extend((np.array([float(packed.size)]), packed))\n'
               '        return np.concatenate(chunks)\n'
               '    array = np.asarray(value, dtype=float)\n'
               '    return np.concatenate((np.array([1., float(array.ndim)]),\n'
               '                           np.asarray(array.shape, dtype=float),\n'
               '                           array.reshape(-1)))\n',
      'call': '_pack_numeric(bool(np.allclose(detection_update(np.array([[.5]]),np.array([[.75]]),np.ones((1,1)),np.array([.25]),2,1,2.)[0],np.array([[(np.sqrt(5.)-1.)/4.]]),rtol=1e-12,atol=0)))',
      'gold_call': '_pack_numeric(bool(np.allclose(_oracle_detection_update(np.array([[.5]]),np.array([[.75]]),np.ones((1,1)),np.array([.25]),2,1,2.)[0],np.array([[(np.sqrt(5.)-1.)/4.]]),rtol=1e-12,atol=0)))',
      'tol': 1e-09}]
