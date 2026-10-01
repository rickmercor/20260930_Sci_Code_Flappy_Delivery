"""
Apply ordered finite projected-gradient sweeps with the prescribed Armijo rule.

Projection changes the descent displacement; an unconstrained gradient-norm Armijo condition is not equivalent. Updated U must enter the V sweep, and trial likelihood-domain violations must be rejected.

Returns
-------
U_new, V_new : tuple of ndarrays     Perform steps updates of U with V frozen, then steps updates of V     with the newly updated U frozen. Freeze B,rho,P for the whole sweep.     Q'=maximum(Q-t*g,0), halving t until Phi(Q') <=     Phi(Q)+armijo*sum(g*(Q'-Q)). Use the projected displacement.     A trial with zero intensity at positive count or nonfinite terms     is rejected; do not floor intensities. No convergence stopping.     An unchanged projected candidate is accepted by the same inequality.     At most 100 candidate trials per gradient update, starting at t0.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def factor_sweep(S: 'np.ndarray', P: 'np.ndarray', U: 'np.ndarray', V: 'np.ndarray', B: tuple, rho: 'np.ndarray', replicates: int, steps: int = 8, initial_step: float = 0.1, armijo: float = 0.0001) -> tuple:
    """Update U then V by finite projected-gradient Armijo sweeps.

    Parameters
    ----------
    S, P, U, V, B, rho, replicates
        Exactly the shapes, domains, objective and gradient contract of
        factor_terms. Do not mutate any input.
    steps : int
        Nonnegative updates per block, booleans excluded; zero returns copies.
    initial_step : float
        Finite strictly positive starting t, reset at EVERY update.
    armijo : float
        Finite acceptance coefficient strictly between zero and one.

    Returns
    -------
    U_new, V_new : tuple of ndarrays
        Perform steps updates of U with V frozen, then steps updates of V
        with the newly updated U frozen. Freeze B,rho,P for the whole sweep.
        Q'=maximum(Q-t*g,0), halving t until Phi(Q') <=
        Phi(Q)+armijo*sum(g*(Q'-Q)). Use the projected displacement.
        A trial with zero intensity at positive count or nonfinite terms
        is rejected; do not floor intensities. No convergence stopping.
        An unchanged projected candidate is accepted by the same inequality.
        At most 100 candidate trials per gradient update, starting at t0.

    Raises
    ------
    ValueError
        Any violation of factor_terms' input contract; non-real/nonnumeric
        initial_step/armijo or values outside the ranges above; invalid steps;
        or no acceptable candidate in 100 trials.

    Inherited requirements (explicit)
    ---------------------------------
    S,P must have common nonempty shape (I,J); S>=0 and 0<=P<=1. U,V have shapes (I,F),(J,F), F>=1, and are nonnegative. B has shapes (I,I),(I,J),(J,J), with exactly symmetric first/last matrices; rho is length three and strictly positive. All entries are finite real numbers. Replicates is a positive nonboolean integer. A positive S entry requires positive U@V.T. Nonnumeric, non-real, nonfinite, malformed inputs or nonfinite initial objective/gradients raise ValueError.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_factor_sweep(S: 'np.ndarray', P: 'np.ndarray', U: 'np.ndarray', V: 'np.ndarray', B: tuple, rho: 'np.ndarray', replicates: int, steps: int = 8, initial_step: float = 0.1, armijo: float = 0.0001) -> tuple:
    """Update U then V by finite projected-gradient Armijo sweeps.

    Parameters
    ----------
    S, P, U, V, B, rho, replicates
        Exactly the shapes, domains, objective and gradient contract of
        factor_terms. Do not mutate any input.
    steps : int
        Nonnegative updates per block, booleans excluded; zero returns copies.
    initial_step : float
        Finite strictly positive starting t, reset at EVERY update.
    armijo : float
        Finite acceptance coefficient strictly between zero and one.

    Returns
    -------
    U_new, V_new : tuple of ndarrays
        Perform steps updates of U with V frozen, then steps updates of V
        with the newly updated U frozen. Freeze B,rho,P for the whole sweep.
        Q'=maximum(Q-t*g,0), halving t until Phi(Q') <=
        Phi(Q)+armijo*sum(g*(Q'-Q)). Use the projected displacement.
        A trial with zero intensity at positive count or nonfinite terms
        is rejected; do not floor intensities. No convergence stopping.
        An unchanged projected candidate is accepted by the same inequality.
        At most 100 candidate trials per gradient update, starting at t0.

    Raises
    ------
    ValueError
        Any violation of factor_terms' input contract; non-real/nonnumeric
        initial_step/armijo or values outside the ranges above; invalid steps;
        or no acceptable candidate in 100 trials.
    """
    import numpy as np
    _oracle_factor_terms(S,P,U,V,B,rho,replicates)
    try:
        if np.iscomplexobj(initial_step) or np.iscomplexobj(armijo): raise ValueError('real controls required')
        initial_step=float(initial_step); armijo=float(armijo)
    except (TypeError,ValueError,OverflowError) as e: raise ValueError('numeric controls required') from e
    if isinstance(steps,(bool,np.bool_)) or not isinstance(steps,(int,np.integer)) or steps<0 or not np.isfinite(initial_step) or initial_step<=0 or not np.isfinite(armijo) or not 0<armijo<1: raise ValueError('invalid controls')
    U=np.array(U,dtype=float,copy=True); V=np.array(V,dtype=float,copy=True)
    for block in (0,1):
        for _ in range(steps):
            value,gu,gv=_oracle_factor_terms(S,P,U,V,B,rho,replicates)
            old=U if block==0 else V; g=gu if block==0 else gv; t=initial_step
            for trial in range(100):
                candidate=np.maximum(old-t*g,0)
                try:
                    newvalue=_oracle_factor_terms(S,P,candidate if block==0 else U,V if block==0 else candidate,B,rho,replicates)[0]
                except ValueError: newvalue=np.inf
                if np.isfinite(newvalue) and newvalue<=value+armijo*np.sum(g*(candidate-old)): break
                t*=.5
            else: raise ValueError('line search exhausted')
            if block==0: U=candidate
            else: V=candidate
    return U,V

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
      'call': '_pack_numeric(factor_sweep(S, P, U, V, B, rho, 3))',
      'gold_call': '_pack_numeric(_oracle_factor_sweep(S, P, U, V, B, rho, 3))',
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
      'call': '_pack_numeric(factor_sweep(S, P, U, V, B, rho, 3, 0))',
      'gold_call': '_pack_numeric(_oracle_factor_sweep(S, P, U, V, B, rho, 3, 0))',
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
      'call': '_pack_numeric(factor_sweep(S, P, U, V, B, rho, 3, 3, 20.))',
      'gold_call': '_pack_numeric(_oracle_factor_sweep(S, P, U, V, B, rho, 3, 3, 20.))',
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
               'S=np.zeros((4,3)); U=np.zeros((4,2)); '
               'B=(np.zeros((4,4)),np.zeros((4,3)),np.zeros((3,3)))\n'
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
      'call': '_pack_numeric(factor_sweep(S, P, U, V, B, rho, 3, 2))',
      'gold_call': '_pack_numeric(_oracle_factor_sweep(S, P, U, V, B, rho, 3, 2))',
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
               'rho=np.array([3.,.02,1.5])\n'
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
      'call': '_pack_numeric(factor_sweep(S, P, U, V, B, rho, 3, 5))',
      'gold_call': '_pack_numeric(_oracle_factor_sweep(S, P, U, V, B, rho, 3, 5))',
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
      'call': '_pack_numeric(factor_sweep(S, P, U, V, B, rho, 7, 4, .2, .01))',
      'gold_call': '_pack_numeric(_oracle_factor_sweep(S, P, U, V, B, rho, 7, 4, .2, .01))',
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
      'call': '_exception_code(factor_sweep, S, P, U, V, B, rho, 3, 2, .1, 1.)',
      'gold_call': '_exception_code(_oracle_factor_sweep, S, P, U, V, B, rho, 3, 2, .1, 1.)',
      'tol': 1e-09}]
