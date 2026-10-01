"""
Evaluate the Poisson factor objective and the derivatives of all three Gram-product constraints.

The within-group products contain the factor twice; full Frobenius norms count both off-diagonal entries. Zero observations do not justify deleting the mean-intensity contribution.

Returns
-------
value, grad_U, grad_V : tuple     Native float and arrays shaped like U,V. Evaluate     sum(replicates*P*L - S*log(L)) +     sum_X rho_X/2 * ||M_X-B_X||_F**2,     where L=U@V.T and M=(U@U.T,U@V.T,V@V.T).     Include all matrix entries, including diagonals and both symmetric     off-diagonal entries. Interpret 0*log(0) and 0/0 count ratios as zero.     Derive gradients with respect to the factors, including both     appearances of a factor in its within-group Gram matrix.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def factor_terms(S: 'np.ndarray', P: 'np.ndarray', U: 'np.ndarray', V: 'np.ndarray', B: tuple, rho: 'np.ndarray', replicates: int) -> tuple:
    """Evaluate the factor objective and both exact Euclidean gradients.

    Parameters
    ----------
    S, P : ndarrays, shape (I,J), I,J >= 1
        Finite aggregate counts S>=0 and auxiliary detection probabilities
        0<=P<=1; S may be real. P is fixed during differentiation.
    U, V : ndarrays, shapes (I,F), (J,F), F >= 1
        Finite nonnegative factors. Positive counts require (U@V.T)>0.
    B : tuple/list of three arrays
        Centers ordered UU,UV,VV with shapes (I,I),(I,J),(J,J).
        All finite; within-group centers must be exactly symmetric.
    rho : ndarray, shape (3,)
        Finite strictly positive penalties ordered UU,UV,VV.
    replicates : int
        Positive integer, booleans excluded.

    Returns
    -------
    value, grad_U, grad_V : tuple
        Native float and arrays shaped like U,V. Evaluate
        sum(replicates*P*L - S*log(L)) +
        sum_X rho_X/2 * ||M_X-B_X||_F**2,
        where L=U@V.T and M=(U@U.T,U@V.T,V@V.T).
        Include all matrix entries, including diagonals and both symmetric
        off-diagonal entries. Interpret 0*log(0) and 0/0 count ratios as zero.
        Derive gradients with respect to the factors, including both
        appearances of a factor in its within-group Gram matrix.

    Raises
    ------
    ValueError
        Non-real/nonnumeric input; malformed/nonfinite inputs; incompatible
        or empty dimensions; invalid probabilities, negative counts/factors,
        asymmetric within-group centers, nonpositive penalties, invalid
        replicate count, zero intensity at positive count, or nonfinite
        objective/gradient/product.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_factor_terms(S: 'np.ndarray', P: 'np.ndarray', U: 'np.ndarray', V: 'np.ndarray', B: tuple, rho: 'np.ndarray', replicates: int) -> tuple:
    """Evaluate the factor objective and both exact Euclidean gradients.

    Parameters
    ----------
    S, P : ndarrays, shape (I,J), I,J >= 1
        Finite aggregate counts S>=0 and auxiliary detection probabilities
        0<=P<=1; S may be real. P is fixed during differentiation.
    U, V : ndarrays, shapes (I,F), (J,F), F >= 1
        Finite nonnegative factors. Positive counts require (U@V.T)>0.
    B : tuple/list of three arrays
        Centers ordered UU,UV,VV with shapes (I,I),(I,J),(J,J).
        All finite; within-group centers must be exactly symmetric.
    rho : ndarray, shape (3,)
        Finite strictly positive penalties ordered UU,UV,VV.
    replicates : int
        Positive integer, booleans excluded.

    Returns
    -------
    value, grad_U, grad_V : tuple
        Native float and arrays shaped like U,V. Evaluate
        sum(replicates*P*L - S*log(L)) +
        sum_X rho_X/2 * ||M_X-B_X||_F**2,
        where L=U@V.T and M=(U@U.T,U@V.T,V@V.T).
        Include all matrix entries, including diagonals and both symmetric
        off-diagonal entries. Interpret 0*log(0) and 0/0 count ratios as zero.
        Derive gradients with respect to the factors, including both
        appearances of a factor in its within-group Gram matrix.

    Raises
    ------
    ValueError
        Non-real/nonnumeric input; malformed/nonfinite inputs; incompatible
        or empty dimensions; invalid probabilities, negative counts/factors,
        asymmetric within-group centers, nonpositive penalties, invalid
        replicate count, zero intensity at positive count, or nonfinite
        objective/gradient/product.
    """
    import numpy as np
    try:
        if any(np.iscomplexobj(a) for a in (S,P,U,V,rho)) or len(B)!=3 or any(np.iscomplexobj(a) for a in B): raise ValueError('real inputs and three centers required')
        S=np.asarray(S,float); P=np.asarray(P,float); U=np.asarray(U,float); V=np.asarray(V,float); rho=np.asarray(rho,float); B=tuple(np.asarray(a,float) for a in B)
    except (TypeError,ValueError,OverflowError) as e: raise ValueError('numeric inputs required') from e
    if S.ndim!=2 or min(S.shape)<1 or P.shape!=S.shape or U.ndim!=2 or V.ndim!=2 or U.shape[0]!=S.shape[0] or V.shape[0]!=S.shape[1] or U.shape[1]<1 or V.shape[1]!=U.shape[1] or rho.shape!=(3,): raise ValueError('invalid shapes')
    i,j=S.shape
    if tuple(a.shape for a in B)!=((i,i),(i,j),(j,j)): raise ValueError('invalid center shapes')
    if any(not np.isfinite(a).all() for a in (S,P,U,V,rho)+B) or np.any(S<0) or np.any(U<0) or np.any(V<0) or np.any(P<0) or np.any(P>1) or np.any(rho<=0): raise ValueError('invalid numeric data')
    if not np.array_equal(B[0],B[0].T) or not np.array_equal(B[2],B[2].T): raise ValueError('centers must be symmetric')
    if isinstance(replicates,(bool,np.bool_)) or not isinstance(replicates,(int,np.integer)) or replicates<1: raise ValueError('invalid replicate count')
    L=U@V.T
    if not np.isfinite(L).all() or np.any(L[S>0]<=0): raise ValueError('invalid intensity')
    residuals=(U@U.T-B[0],L-B[1],V@V.T-B[2])
    value=float(np.sum(replicates*P*L)-np.sum(S[S>0]*np.log(L[S>0]))+sum(r/2*np.sum(a*a) for r,a in zip(rho,residuals)))
    D=replicates*P-np.divide(S,L,out=np.zeros_like(S),where=L>0)
    gu=D@V+2*rho[0]*residuals[0]@U+rho[1]*residuals[1]@V
    gv=D.T@U+2*rho[2]*residuals[2]@V+rho[1]*residuals[1].T@U
    if not np.isfinite(value) or not np.isfinite(gu).all() or not np.isfinite(gv).all(): raise ValueError('nonfinite terms')
    return value,gu,gv

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
      'call': '_pack_numeric(factor_terms(S, P, U, V, B, rho, 3))',
      'gold_call': '_pack_numeric(_oracle_factor_terms(S, P, U, V, B, rho, 3))',
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
               'S=np.zeros((4,3)); U=np.zeros((4,2)); V=np.zeros((3,2))\n'
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
      'call': '_pack_numeric(factor_terms(S, P, U, V, B, rho, 3))',
      'gold_call': '_pack_numeric(_oracle_factor_terms(S, P, U, V, B, rho, 3))',
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
               'B=(B[0]-2,B[1]+.7,B[2]-.9)\n'
               'rho=np.array([2.,.003,.7])\n'
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
      'call': '_pack_numeric(factor_terms(S, P, U, V, B, rho, 3))',
      'gold_call': '_pack_numeric(_oracle_factor_terms(S, P, U, V, B, rho, 3))',
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
               'U*=1e-7; V*=1e-6\n'
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
      'call': '_pack_numeric(factor_terms(S, P, U, V, B, rho, 3))',
      'gold_call': '_pack_numeric(_oracle_factor_terms(S, P, U, V, B, rho, 3))',
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
               'P=np.zeros((4,3))\n'
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
      'call': '_pack_numeric(factor_terms(S, P, U, V, B, rho, 3))',
      'gold_call': '_pack_numeric(_oracle_factor_terms(S, P, U, V, B, rho, 3))',
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
               'B[0][0,1]+=.1\n'
               '\n'
               'def _exception_code(function,*args,**kwargs):\n'
               '    try:\n'
               '        function(*args,**kwargs)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_exception_code(factor_terms, S, P, U, V, B, rho, 3)',
      'gold_call': '_exception_code(_oracle_factor_terms, S, P, U, V, B, rho, 3)',
      'tol': 1e-09}]
