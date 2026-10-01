"""
Combine the detection, factor, nonconvex auxiliary, and adaptive-dual operations in the prescribed order.

The final detection auxiliary belongs to the pre-factor state. Recomputing it after updating factors, or using new penalties for the same iteration's proximal solve, changes the finite trajectory.

Returns
-------
U_new,V_new,alpha_new,A_new,W_new,rho_new,P : tuple     Reset detection omega, perform detection with penalty 1, then update     U and V with B=A-W, initial step .1 and Armijo coefficient 1e-4.     With updated factors, globally minimize all three auxiliary blocks     centered at M_X+W_X using weights and OLD rho. Then adapt penalties     and duals with gamma=1.5 and tolerance_X=.5/iteration**1.2.     P is the finite detection auxiliary from BEFORE the factor sweep;     do not recompute it using final factors. Inputs remain unchanged.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def outer_cycle(S: 'np.ndarray', replicates: int, Z: 'np.ndarray', U: 'np.ndarray', V: 'np.ndarray', alpha: 'np.ndarray', auxiliaries: tuple, duals: tuple, rho: 'np.ndarray', weights: 'np.ndarray', iteration: int, detection_steps: int = 80, factor_steps: int = 8) -> tuple:
    """Advance one prescribed sparse-network outer iteration.

    Parameters
    ----------
    S, replicates, Z, U, V, alpha
        Data and factors satisfying detection_update and factor_terms contracts.
    auxiliaries, duals : tuple/list of three arrays
        Shapes (I,I),(I,J),(J,J), finite real entries. Each within-group
        array must be exactly symmetric. Auxiliary entries may be negative.
    rho, weights : ndarray, shape (3,)
        Finite rho>0 and weights>=0, ordered UU,UV,VV.
    iteration : int
        Positive one-based outer index; booleans excluded.
    detection_steps : int
        Positive finite-update budget for detection_update, default 80.
    factor_steps : int
        Nonnegative budget per factor in factor_sweep, default 8.

    Returns
    -------
    U_new,V_new,alpha_new,A_new,W_new,rho_new,P : tuple
        Reset detection omega, perform detection with penalty 1, then update
        U and V with B=A-W, initial step .1 and Armijo coefficient 1e-4.
        With updated factors, globally minimize all three auxiliary blocks
        centered at M_X+W_X using weights and OLD rho. Then adapt penalties
        and duals with gamma=1.5 and tolerance_X=.5/iteration**1.2.
        P is the finite detection auxiliary from BEFORE the factor sweep;
        do not recompute it using final factors. Inputs remain unchanged.

    Raises
    ------
    ValueError
        Any violated earlier-step contract (including their numerical
        failures); invalid iteration; invalid weights; malformed auxiliary
        or dual blocks; or asymmetric within-group auxiliary/dual matrices.

    Inherited requirements (explicit)
    ---------------------------------
    S is nonempty (I,J), nonnegative; U,V are nonnegative (I,F),(J,F), F>=1. A,W each have shapes (I,I),(I,J),(J,J), with exactly symmetric first/last matrices. Z is (I*J,R), R>=1; alpha is (R,); rho and weights are (3,), rho>0 and weights>=0. All entries must be finite and real. Positive S requires positive U@V.T. Replicates, iteration and detection_steps are positive nonboolean integers; factor_steps is a nonnegative nonboolean integer. Violations, failed pseudoinverse, nonfinite computed terms, or a line search failing all 100 candidates raise ValueError.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_outer_cycle(S: 'np.ndarray', replicates: int, Z: 'np.ndarray', U: 'np.ndarray', V: 'np.ndarray', alpha: 'np.ndarray', auxiliaries: tuple, duals: tuple, rho: 'np.ndarray', weights: 'np.ndarray', iteration: int, detection_steps: int = 80, factor_steps: int = 8) -> tuple:
    """Advance one prescribed sparse-network outer iteration.

    Parameters
    ----------
    S, replicates, Z, U, V, alpha
        Data and factors satisfying detection_update and factor_terms contracts.
    auxiliaries, duals : tuple/list of three arrays
        Shapes (I,I),(I,J),(J,J), finite real entries. Each within-group
        array must be exactly symmetric. Auxiliary entries may be negative.
    rho, weights : ndarray, shape (3,)
        Finite rho>0 and weights>=0, ordered UU,UV,VV.
    iteration : int
        Positive one-based outer index; booleans excluded.
    detection_steps : int
        Positive finite-update budget for detection_update, default 80.
    factor_steps : int
        Nonnegative budget per factor in factor_sweep, default 8.

    Returns
    -------
    U_new,V_new,alpha_new,A_new,W_new,rho_new,P : tuple
        Reset detection omega, perform detection with penalty 1, then update
        U and V with B=A-W, initial step .1 and Armijo coefficient 1e-4.
        With updated factors, globally minimize all three auxiliary blocks
        centered at M_X+W_X using weights and OLD rho. Then adapt penalties
        and duals with gamma=1.5 and tolerance_X=.5/iteration**1.2.
        P is the finite detection auxiliary from BEFORE the factor sweep;
        do not recompute it using final factors. Inputs remain unchanged.

    Raises
    ------
    ValueError
        Any violated earlier-step contract (including their numerical
        failures); invalid iteration; invalid weights; malformed auxiliary
        or dual blocks; or asymmetric within-group auxiliary/dual matrices.
    """
    import numpy as np
    if isinstance(iteration,(bool,np.bool_)) or not isinstance(iteration,(int,np.integer)) or iteration<1: raise ValueError('invalid iteration')
    try:
        if np.iscomplexobj(weights) or len(auxiliaries)!=3 or len(duals)!=3 or any(np.iscomplexobj(a) for a in tuple(auxiliaries)+tuple(duals)): raise ValueError('real blocks required')
        weights=np.asarray(weights,float); auxiliaries=tuple(np.asarray(a,float) for a in auxiliaries); duals=tuple(np.asarray(a,float) for a in duals)
        if any(np.iscomplexobj(a) for a in (U,V)): raise ValueError('real factors required')
        U=np.asarray(U,float); V=np.asarray(V,float)
    except (TypeError,ValueError,OverflowError) as e: raise ValueError('numeric inputs required') from e
    if weights.shape!=(3,) or not np.isfinite(weights).all() or np.any(weights<0): raise ValueError('invalid weights')
    if U.ndim!=2 or V.ndim!=2 or U.shape[1]<1 or U.shape[1]!=V.shape[1]: raise ValueError('invalid factors')
    products=(U@U.T,U@V.T,V@V.T)
    _oracle_adapt_duals(products,auxiliaries,duals,rho,1.5,np.zeros(3))
    if any(not np.array_equal(group[j],group[j].T) for group in (auxiliaries,duals) for j in (0,2)): raise ValueError('asymmetric within-group blocks')
    B=tuple(a-w for a,w in zip(auxiliaries,duals))
    _oracle_factor_terms(S,np.zeros_like(products[1]),U,V,B,rho,replicates)
    P,newalpha,_=_oracle_detection_update(S,products[1],Z,alpha,replicates,detection_steps,1.)
    U,V=_oracle_factor_sweep(S,P,U,V,B,rho,replicates,factor_steps,.1,1e-4)
    products=(U@U.T,U@V.T,V@V.T)
    newA=tuple(_oracle_half_prox(m+w,l,r) for m,w,l,r in zip(products,duals,weights,rho))
    newrho,newW,_=_oracle_adapt_duals(products,newA,duals,rho,1.5,np.full(3,.5/iteration**1.2))
    return U,V,newalpha,newA,newW,newrho,P

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
      'call': '_pack_numeric(outer_cycle(S, 3, Z, U, V, alpha, A, W, rho, weights, 1))',
      'gold_call': '_pack_numeric(_oracle_outer_cycle(S, 3, Z, U, V, alpha, A, W, rho, weights, 1))',
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
      'call': '_pack_numeric(outer_cycle(S, 3, Z, U, V, alpha, A, W, rho, weights, 1, 2, 0))',
      'gold_call': '_pack_numeric(_oracle_outer_cycle(S, 3, Z, U, V, alpha, A, W, rho, weights, 1, 2, '
                   '0))',
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
               'W=tuple(np.full_like(a,.15) for a in products)\n'
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
      'call': '_pack_numeric(outer_cycle(S, 3, Z, U, V, alpha, A, W, rho, weights, 5, 7, 3))',
      'gold_call': '_pack_numeric(_oracle_outer_cycle(S, 3, Z, U, V, alpha, A, W, rho, weights, 5, 7, '
                   '3))',
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
               'W=tuple(np.full_like(a,.5) for a in products); weights=np.full(3,.001)\n'
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
      'call': '_pack_numeric(outer_cycle(S, 3, Z, U, V, alpha, A, W, rho, weights, 3, 4, 2))',
      'gold_call': '_pack_numeric(_oracle_outer_cycle(S, 3, Z, U, V, alpha, A, W, rho, weights, 3, 4, '
                   '2))',
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
               'weights=np.zeros(3)\n'
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
      'call': '_pack_numeric(outer_cycle(S, 3, Z, U, V, alpha, A, W, rho, weights, 2, 10, 4))',
      'gold_call': '_pack_numeric(_oracle_outer_cycle(S, 3, Z, U, V, alpha, A, W, rho, weights, 2, 10, '
                   '4))',
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
               'weights=np.array([.001,4.,.08])\n'
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
      'call': '_pack_numeric(outer_cycle(S, 3, Z, U, V, alpha, A, W, rho, weights, 9, 3, 2))',
      'gold_call': '_pack_numeric(_oracle_outer_cycle(S, 3, Z, U, V, alpha, A, W, rho, weights, 9, 3, '
                   '2))',
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
      'call': '_exception_code(outer_cycle, S, 3, Z, U, V, alpha, A, W, rho, weights, 0)',
      'gold_call': '_exception_code(_oracle_outer_cycle, S, 3, Z, U, V, alpha, A, W, rho, weights, 0)',
      'tol': 1e-09}]
