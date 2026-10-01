"""
Construct the scale-aware rank-limited initialization from replicated ecological counts.

Independent Poisson-thinned replicates contribute summed counts and a replicate multiplier to the likelihood. The initialization rescales observed counts by mean detection before taking absolute singular-vector factors.

Returns
-------
S, U, V : tuple of ndarrays     S=sum(Y,axis=2), shape (I,J). Take the descending SVD L,s,Rt     of S/(M*p0). U=abs(L[:,:rank])*sqrt(s[:rank]);     V=abs(Rt[:rank,:].T)*sqrt(s[:rank]). Do not fit an NMF or     rectify the reconstructed matrix instead of the singular vectors.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spectral_start(Y: 'np.ndarray', rank: int, p0: float) -> tuple:
    """Aggregate independent replicate counts and initialize nonnegative factors.

    Parameters
    ----------
    Y : ndarray, shape (I,J,M)
        Finite nonnegative integer counts; I,J,M >= 1. Last axis is replicate.
    rank : int
        Retained rank, 1 <= rank <= min(I,J); booleans are excluded.
    p0 : float
        Finite initial mean detection probability, 0 < p0 <= 1.

    Returns
    -------
    S, U, V : tuple of ndarrays
        S=sum(Y,axis=2), shape (I,J). Take the descending SVD L,s,Rt
        of S/(M*p0). U=abs(L[:,:rank])*sqrt(s[:rank]);
        V=abs(Rt[:rank,:].T)*sqrt(s[:rank]). Do not fit an NMF or
        rectify the reconstructed matrix instead of the singular vectors.

    Raises
    ------
    ValueError
        Non-real or nonnumeric input; invalid dimensions, counts, rank or p0;
        any retained singular value <= 1e-12*s[0], or adjacent singular
        values separated by <= 1e-12*s[0] when their upper index is retained
        (including the rank cutoff); failed SVD; or nonfinite result.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_spectral_start(Y: 'np.ndarray', rank: int, p0: float) -> tuple:
    """Aggregate independent replicate counts and initialize nonnegative factors.

    Parameters
    ----------
    Y : ndarray, shape (I,J,M)
        Finite nonnegative integer counts; I,J,M >= 1. Last axis is replicate.
    rank : int
        Retained rank, 1 <= rank <= min(I,J); booleans are excluded.
    p0 : float
        Finite initial mean detection probability, 0 < p0 <= 1.

    Returns
    -------
    S, U, V : tuple of ndarrays
        S=sum(Y,axis=2), shape (I,J). Take the descending SVD L,s,Rt
        of S/(M*p0). U=abs(L[:,:rank])*sqrt(s[:rank]);
        V=abs(Rt[:rank,:].T)*sqrt(s[:rank]). Do not fit an NMF or
        rectify the reconstructed matrix instead of the singular vectors.

    Raises
    ------
    ValueError
        Non-real or nonnumeric input; invalid dimensions, counts, rank or p0;
        any retained singular value <= 1e-12*s[0], or adjacent singular
        values separated by <= 1e-12*s[0] when their upper index is retained
        (including the rank cutoff); failed SVD; or nonfinite result.
    """
    import numpy as np
    try:
        if np.iscomplexobj(Y) or np.iscomplexobj(p0): raise ValueError('real inputs required')
        Y=np.asarray(Y,dtype=float); p0=float(p0)
    except (TypeError,ValueError,OverflowError) as e: raise ValueError('numeric inputs required') from e
    if Y.ndim!=3 or min(Y.shape)<1 or not np.isfinite(Y).all() or np.any(Y<0) or np.any(Y!=np.floor(Y)):
        raise ValueError('invalid counts')
    if isinstance(rank,(bool,np.bool_)) or not isinstance(rank,(int,np.integer)) or not 1<=rank<=min(Y.shape[:2]): raise ValueError('invalid rank')
    if not np.isfinite(p0) or not 0<p0<=1: raise ValueError('invalid p0')
    S=Y.sum(axis=2); proxy=S/(Y.shape[2]*p0)
    if not np.isfinite(proxy).all(): raise ValueError('nonfinite proxy')
    try: left,s,right=np.linalg.svd(proxy,full_matrices=False)
    except np.linalg.LinAlgError as e: raise ValueError('SVD failed') from e
    tol=1e-12*s[0]
    if np.any(s[:rank]<=tol) or np.any((s[:-1]-s[1:])[:rank]<=tol): raise ValueError('ambiguous or null spectral initialization')
    U=abs(left[:,:rank])*np.sqrt(s[:rank]); V=abs(right[:rank,:].T)*np.sqrt(s[:rank])
    if not np.isfinite(U).all() or not np.isfinite(V).all(): raise ValueError('nonfinite factors')
    return S,U,V

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
      'call': '_pack_numeric(spectral_start(Y, 2, .55))',
      'gold_call': '_pack_numeric(_oracle_spectral_start(Y, 2, .55))',
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
               'Y=np.array([[[0.,2.,1.]]])\n'
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
      'call': '_pack_numeric(spectral_start(Y, 1, 1.))',
      'gold_call': '_pack_numeric(_oracle_spectral_start(Y, 1, 1.))',
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
               'Y=np.array([[[10000.,9000.],[1.,0.]],[[0.,1.],[3.,4.]],[[2.,0.],[1.,3.]]])\n'
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
      'call': '_pack_numeric(spectral_start(Y, 2, .02))',
      'gold_call': '_pack_numeric(_oracle_spectral_start(Y, 2, .02))',
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
               'Y=np.array([[[3.],[0.],[2.],[1.]],[[1.],[5.],[0.],[2.]]])\n'
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
      'call': '_pack_numeric(spectral_start(Y, 2, .7))',
      'gold_call': '_pack_numeric(_oracle_spectral_start(Y, 2, .7))',
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
      'call': '_pack_numeric(spectral_start(Y, 1, .55))',
      'gold_call': '_pack_numeric(_oracle_spectral_start(Y, 1, .55))',
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
               'Y=np.eye(2)[:,:,None]\n'
               '\n'
               'def _exception_code(function,*args,**kwargs):\n'
               '    try:\n'
               '        function(*args,**kwargs)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_exception_code(spectral_start, Y, 1, .5)',
      'gold_call': '_exception_code(_oracle_spectral_start, Y, 1, .5)',
      'tol': 1e-09}]
