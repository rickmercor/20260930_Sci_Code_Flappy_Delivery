"""
Update three independent penalties and their scaled dual variables without altering the intended unscaled update.

Penalty adaptation changes the unit used to represent the scaled dual. Preserving H=rho*W requires rescaling after the residual increment, with a strict greater-than threshold.

Returns
-------
rho_new, duals_new, residual_norms : tuple     Penalty vector, tuple of three arrays, and length-3 Frobenius norms.     For each block, increase rho by gamma iff norm(residual)>tolerance;     equality leaves rho unchanged. Apply H_new=H+rho_old*residual,     then express H_new using the NEW penalty. Never mutate inputs.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def adapt_duals(products: tuple, auxiliaries: tuple, duals: tuple, rho: 'np.ndarray', gamma: float, tolerance: 'np.ndarray') -> tuple:
    """Adapt three penalties while preserving the unscaled dual update.

    Parameters
    ----------
    products, auxiliaries, duals : tuple/list of three finite real matrices
        Corresponding shapes (I,I),(I,J),(J,J), I,J>=1. Matrices may be
        signed and need not be symmetric for this algebraic operation.
        Residual = products - auxiliaries; duals contain W=H/rho.
    rho, tolerance : ndarray, shape (3,)
        Finite rho>0 and tolerance>=0, in UU,UV,VV order.
    gamma : float
        Finite multiplier strictly greater than one.

    Returns
    -------
    rho_new, duals_new, residual_norms : tuple
        Penalty vector, tuple of three arrays, and length-3 Frobenius norms.
        For each block, increase rho by gamma iff norm(residual)>tolerance;
        equality leaves rho unchanged. Apply H_new=H+rho_old*residual,
        then express H_new using the NEW penalty. Never mutate inputs.

    Raises
    ------
    ValueError
        Non-real/nonnumeric inputs; missing blocks, malformed or inconsistent
        shapes, nonfinite entries, nonpositive rho, negative tolerance,
        gamma<=1, or nonfinite computed norms, penalties or duals.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_adapt_duals(products: tuple, auxiliaries: tuple, duals: tuple, rho: 'np.ndarray', gamma: float, tolerance: 'np.ndarray') -> tuple:
    """Adapt three penalties while preserving the unscaled dual update.

    Parameters
    ----------
    products, auxiliaries, duals : tuple/list of three finite real matrices
        Corresponding shapes (I,I),(I,J),(J,J), I,J>=1. Matrices may be
        signed and need not be symmetric for this algebraic operation.
        Residual = products - auxiliaries; duals contain W=H/rho.
    rho, tolerance : ndarray, shape (3,)
        Finite rho>0 and tolerance>=0, in UU,UV,VV order.
    gamma : float
        Finite multiplier strictly greater than one.

    Returns
    -------
    rho_new, duals_new, residual_norms : tuple
        Penalty vector, tuple of three arrays, and length-3 Frobenius norms.
        For each block, increase rho by gamma iff norm(residual)>tolerance;
        equality leaves rho unchanged. Apply H_new=H+rho_old*residual,
        then express H_new using the NEW penalty. Never mutate inputs.

    Raises
    ------
    ValueError
        Non-real/nonnumeric inputs; missing blocks, malformed or inconsistent
        shapes, nonfinite entries, nonpositive rho, negative tolerance,
        gamma<=1, or nonfinite computed norms, penalties or duals.
    """
    import numpy as np
    try:
        if any(len(a)!=3 for a in (products,auxiliaries,duals)): raise ValueError('three blocks required')
        if any(np.iscomplexobj(a) for group in (products,auxiliaries,duals) for a in group) or any(np.iscomplexobj(a) for a in (rho,tolerance,gamma)): raise ValueError('real inputs required')
        products=tuple(np.asarray(a,float) for a in products); auxiliaries=tuple(np.asarray(a,float) for a in auxiliaries); duals=tuple(np.asarray(a,float) for a in duals)
        rho=np.asarray(rho,float); tolerance=np.asarray(tolerance,float); gamma=float(gamma)
    except (TypeError,ValueError,OverflowError) as e: raise ValueError('numeric inputs required') from e
    if any(a.ndim!=2 for a in products) or min(products[1].shape)<1: raise ValueError('invalid shapes')
    i,j=products[1].shape; shapes=((i,i),(i,j),(j,j))
    if any(tuple(a.shape for a in group)!=shapes for group in (products,auxiliaries,duals)) or rho.shape!=(3,) or tolerance.shape!=(3,): raise ValueError('inconsistent shapes')
    if any(not np.isfinite(a).all() for a in products+auxiliaries+duals+(rho,tolerance)) or np.any(rho<=0) or np.any(tolerance<0) or not np.isfinite(gamma) or gamma<=1: raise ValueError('invalid values')
    residuals=tuple(a-b for a,b in zip(products,auxiliaries)); norms=np.array([np.linalg.norm(a) for a in residuals])
    newrho=rho*np.where(norms>tolerance,gamma,1)
    newduals=tuple((old/new)*(w+r) for old,new,w,r in zip(rho,newrho,duals,residuals))
    if not np.isfinite(norms).all() or not np.isfinite(newrho).all() or any(not np.isfinite(a).all() for a in newduals): raise ValueError('nonfinite update')
    return newrho,newduals,norms

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
               'A=tuple(a-.1 for a in products)\n'
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
      'call': '_pack_numeric(adapt_duals(products, A, W, rho, 1.5, np.array([.2,.4,.3])))',
      'gold_call': '_pack_numeric(_oracle_adapt_duals(products, A, W, rho, 1.5, np.array([.2,.4,.3])))',
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
               'A=tuple(a-np.eye(*a.shape) for a in products)\n'
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
      'call': '_pack_numeric(adapt_duals(products, A, W, rho, 1.5, np.full(3,1.5)))',
      'gold_call': '_pack_numeric(_oracle_adapt_duals(products, A, W, rho, 1.5, np.full(3,1.5)))',
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
      'call': '_pack_numeric(adapt_duals(products, A, W, rho, 2., np.zeros(3)))',
      'gold_call': '_pack_numeric(_oracle_adapt_duals(products, A, W, rho, 2., np.zeros(3)))',
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
               'products=(np.array([[1.]]),np.array([[2.]]),np.array([[3.]])); '
               'A=tuple(np.zeros_like(a) for a in products); W=tuple(np.ones_like(a) for a in '
               'products)\n'
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
      'call': '_pack_numeric(adapt_duals(products, A, W, rho, 2., np.array([1.,1.,4.])))',
      'gold_call': '_pack_numeric(_oracle_adapt_duals(products, A, W, rho, 2., np.array([1.,1.,4.])))',
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
               'A=tuple(a+.3 for a in products); W=tuple(np.full_like(a,-.7) for a in products)\n'
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
      'call': '_pack_numeric(adapt_duals(products, A, W, rho, 1.7, np.array([0.,10.,0.])))',
      'gold_call': '_pack_numeric(_oracle_adapt_duals(products, A, W, rho, 1.7, '
                   'np.array([0.,10.,0.])))',
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
               'A[0][0,1]+=.8\n'
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
      'call': '_pack_numeric(adapt_duals(products, A, W, rho, 1.5, np.zeros(3)))',
      'gold_call': '_pack_numeric(_oracle_adapt_duals(products, A, W, rho, 1.5, np.zeros(3)))',
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
      'call': '_exception_code(adapt_duals, products, A, W, rho, 1., np.zeros(3))',
      'gold_call': '_exception_code(_oracle_adapt_duals, products, A, W, rho, 1., np.zeros(3))',
      'tol': 1e-09}]
