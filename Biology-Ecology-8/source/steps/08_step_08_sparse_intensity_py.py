"""
Return the prescribed latent intensity by directly composing Steps 01–06 for initialization and the first cycle, then passing that updated state to Step 07 for all remaining cycles.

The first cycle exposes every primitive dependency. Step 03 supplies the augmented-objective values used to verify descent of the Step 04 factor sweep; Step 05 supplies the auxiliaries consumed by Step 06. Later cycles use Step 07 starting from the complete first-cycle state. Both paths implement the same finite update order, so the split must not restart any state or repeat the first cycle.

Returns
-------
result : float     (U@V.T)[target] after exactly outer_steps cycles. Initialize factors     with spectral_start, auxiliaries with the three factor products,     outer duals with zeros and alpha=alpha0. Cycle indices start at one;     use 80 detection updates and eight updates per factor each cycle.     For outer_steps=0 return the initialized intensity. No rounding,     detection multiplier, convergence stopping or global-optimum claim.     All input contracts are checked even if outer_steps=0.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sparse_intensity(Y: 'np.ndarray', Z: 'np.ndarray', rank: int, p0: float, alpha0: 'np.ndarray', weights: 'np.ndarray', rho0: 'np.ndarray', outer_steps: int = 12, target: tuple = (1, 2)) -> float:
    """Orchestrate the finite sparse ecological-network reconstruction.

    Parameters
    ----------
    Y, rank, p0
        spectral_start contract, with Y shape (I,J,M).
    Z, alpha0
        detection_update design and coefficient contracts; row-major pairs.
    weights, rho0 : ndarray, shape (3,)
        Finite weights>=0 and rho0>0, in UU,UV,VV order.
    outer_steps : int
        Nonnegative number of outer iterations; booleans excluded.
    target : tuple/list of two ints
        ZERO-based plant,pollinator indices, within (I,J); booleans excluded.

    Returns
    -------
    result : float
        (U@V.T)[target] after exactly outer_steps cycles. Initialize factors
        with spectral_start, auxiliaries with the three factor products,
        outer duals with zeros and alpha=alpha0. Cycle indices start at one;
        use 80 detection updates and eight updates per factor each cycle.
        For outer_steps=0 return the initialized intensity. No rounding,
        detection multiplier, convergence stopping or global-optimum claim.
        All input contracts are checked even if outer_steps=0.

    Raises
    ------
    ValueError
        Any violation of the earlier-step contracts or their numerical
        failure conditions; malformed/nonfinite weights or rho0; invalid
        outer_steps or target. Inputs must not be mutated.


    Composition requirements
    ------------------------
    Call spectral_start directly. When outer_steps>=1, implement cycle 1
    directly using detection_update, factor_terms, factor_sweep, half_prox,
    and adapt_duals, consuming their returned values in that order.
    Evaluate factor_terms before and after the factor_sweep with the same
    P, B=A-W and OLD rho. Raise ValueError if the final objective exceeds
    the initial objective by more than 1e-10*(1+abs(initial objective)).
    Then form all three products, compute all auxiliaries using OLD rho,
    and update rho and W. For cycle indices 2..outer_steps, pass the full
    evolving state into outer_cycle. Do not restart or duplicate cycle 1.
    The zero-cycle path still validates inherited input contracts.

    Inherited requirements (explicit)
    ---------------------------------
    Y is a nonempty (I,J,M) real finite nonnegative integer array. Rank is a nonboolean integer in [1,min(I,J)]; 0<p0<=1, finite. Retained singular values must exceed 1e-12*s[0]; adjacent gaps whose upper index is retained, including the cutoff, must exceed 1e-12*s[0]. Z is finite real (I*J,R), R>=1, and alpha0 is finite real (R,). Weights and rho0 are finite real (3,), weights>=0, rho0>0. Outer_steps is a nonnegative nonboolean integer; target contains two nonboolean integers in [0,I) and [0,J). Violations, failed SVD/pseudoinverse, nonfinite computed values, zero intensity at positive count in a current iterate, or exhaustion of 100 line-search trials raise ValueError. These validations also apply when outer_steps=0.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_sparse_intensity(Y: 'np.ndarray', Z: 'np.ndarray', rank: int, p0: float, alpha0: 'np.ndarray', weights: 'np.ndarray', rho0: 'np.ndarray', outer_steps: int = 12, target: tuple = (1, 2)) -> float:
    S, U, V = _oracle_spectral_start(Y, rank, p0)
    try:
        if any(np.iscomplexobj(a) for a in (Z, alpha0, weights, rho0)):
            raise ValueError('real inputs required')
        Z = np.asarray(Z, dtype=float)
        alpha = np.array(alpha0, dtype=float, copy=True)
        weights = np.asarray(weights, dtype=float)
        rho = np.array(rho0, dtype=float, copy=True)
        target = tuple(target)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('numeric inputs required') from exc
    if (isinstance(outer_steps, (bool, np.bool_))
            or not isinstance(outer_steps, (int, np.integer)) or outer_steps < 0):
        raise ValueError('invalid outer count')
    if len(target) != 2 or any(
        isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer))
        or t < 0 or t >= n for t, n in zip(target, S.shape)
    ):
        raise ValueError('invalid target')
    if weights.shape != (3,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('invalid weights')
    M = np.asarray(Y).shape[2]
    A = (U @ U.T, U @ V.T, V @ V.T)
    W = tuple(np.zeros_like(a) for a in A)
    # Validate penalties and block shapes even when no cycle is requested.
    _oracle_adapt_duals(A, A, W, rho, 1.5, np.zeros(3))
    if outer_steps == 0:
        _oracle_detection_update(S, U @ V.T, Z, alpha, M, 1, 1.)
        _oracle_factor_terms(S, np.zeros_like(S), U, V, A, rho, M)
    else:
        # Cycle 1: every primitive is a direct, operative dependency.
        P, alpha, _ = _oracle_detection_update(S, U @ V.T, Z, alpha, M, 80, 1.)
        B = tuple(a-w for a, w in zip(A, W))
        before, _, _ = _oracle_factor_terms(S, P, U, V, B, rho, M)
        U, V = _oracle_factor_sweep(S, P, U, V, B, rho, M, 8, .1, 1e-4)
        after, _, _ = _oracle_factor_terms(S, P, U, V, B, rho, M)
        if after > before + 1e-10*(1.0+abs(before)):
            raise ValueError('factor sweep increased the fixed-block objective')
        products = (U @ U.T, U @ V.T, V @ V.T)
        A = tuple(_oracle_half_prox(m+w, lam, r)
                  for m, w, lam, r in zip(products, W, weights, rho))
        rho, W, _ = _oracle_adapt_duals(products, A, W, rho, 1.5, np.full(3, .5))
        # Step 07 consumes the first cycle's factors, alpha, auxiliaries,
        # penalties and scaled duals; its returned state feeds the result.
        for k in range(2, outer_steps+1):
            U, V, alpha, A, W, rho, P = _oracle_outer_cycle(
                S, M, Z, U, V, alpha, A, W, rho, weights, k, 80, 8)
    result = float((U @ V.T)[target])
    if not np.isfinite(result):
        raise ValueError('nonfinite intensity')
    return result

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
               'B=tuple(a+.2 for a in products)\n',
      'call': 'sparse_intensity(Y, Z, 2, .55, alpha, weights, rho)',
      'gold_call': '_oracle_sparse_intensity(Y, Z, 2, .55, alpha, weights, rho)',
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
               'B=tuple(a+.2 for a in products)\n',
      'call': 'sparse_intensity(Y, Z, 2, .55, alpha, weights, rho, 0)',
      'gold_call': '_oracle_sparse_intensity(Y, Z, 2, .55, alpha, weights, rho, 0)',
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
               'B=tuple(a+.2 for a in products)\n',
      'call': 'sparse_intensity(Y, Z, 1, .35, alpha, weights, rho, 3, (3,0))',
      'gold_call': '_oracle_sparse_intensity(Y, Z, 1, .35, alpha, weights, rho, 3, (3,0))',
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
               'weights=np.array([.05,.2,.1]); rho=np.array([.2,.07,.12])\n',
      'call': 'sparse_intensity(Y, Z, 2, .55, alpha, weights, rho, 5, (0,1))',
      'gold_call': '_oracle_sparse_intensity(Y, Z, 2, .55, alpha, weights, rho, 5, (0,1))',
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
               'Y=Y[:3,:2,:]; Z=Z.reshape(4,3,2)[:3,:2,:].reshape(6,2)\n',
      'call': 'sparse_intensity(Y, Z, 2, .55, alpha, weights, rho, 2, (2,1))',
      'gold_call': '_oracle_sparse_intensity(Y, Z, 2, .55, alpha, weights, rho, 2, (2,1))',
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
      'call': '_exception_code(sparse_intensity, Y, Z, 2, .55, alpha, weights, rho, 2, (4,0))',
      'gold_call': '_exception_code(_oracle_sparse_intensity, Y, Z, 2, .55, alpha, weights, rho, 2, '
                   '(4,0))',
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
               'B=tuple(a+.2 for a in products)\n',
      'call': 'sparse_intensity(Y,Z,2,.55,alpha,weights,rho,1)',
      'gold_call': '_oracle_sparse_intensity(Y,Z,2,.55,alpha,weights,rho,1)',
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
               'B=tuple(a+.2 for a in products)\n',
      'call': 'sparse_intensity(Y,Z,2,.55,alpha,weights,rho,2)',
      'gold_call': '_oracle_sparse_intensity(Y,Z,2,.55,alpha,weights,rho,2)',
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
               'import numpy as np\n'
               'def _exception_code(function, *args):\n'
               '    try:\n'
               '        function(*args)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_exception_code(sparse_intensity,Y,Z,2,.55,alpha,weights,np.zeros(3),0)',
      'gold_call': '_exception_code(_oracle_sparse_intensity,Y,Z,2,.55,alpha,weights,np.zeros(3),0)',
      'tol': 1e-09}]
