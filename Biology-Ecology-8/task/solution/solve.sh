#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np

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

import numpy as np

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

import numpy as np

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

import numpy as np

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
    """
    import numpy as np
    factor_terms(S,P,U,V,B,rho,replicates)
    try:
        if np.iscomplexobj(initial_step) or np.iscomplexobj(armijo): raise ValueError('real controls required')
        initial_step=float(initial_step); armijo=float(armijo)
    except (TypeError,ValueError,OverflowError) as e: raise ValueError('numeric controls required') from e
    if isinstance(steps,(bool,np.bool_)) or not isinstance(steps,(int,np.integer)) or steps<0 or not np.isfinite(initial_step) or initial_step<=0 or not np.isfinite(armijo) or not 0<armijo<1: raise ValueError('invalid controls')
    U=np.array(U,dtype=float,copy=True); V=np.array(V,dtype=float,copy=True)
    for block in (0,1):
        for _ in range(steps):
            value,gu,gv=factor_terms(S,P,U,V,B,rho,replicates)
            old=U if block==0 else V; g=gu if block==0 else gv; t=initial_step
            for trial in range(100):
                candidate=np.maximum(old-t*g,0)
                try:
                    newvalue=factor_terms(S,P,candidate if block==0 else U,V if block==0 else candidate,B,rho,replicates)[0]
                except ValueError: newvalue=np.inf
                if np.isfinite(newvalue) and newvalue<=value+armijo*np.sum(g*(candidate-old)): break
                t*=.5
            else: raise ValueError('line search exhausted')
            if block==0: U=candidate
            else: V=candidate
    return U,V

import numpy as np

def half_prox(b: 'np.ndarray', weight: 'np.ndarray | float', penalty: 'np.ndarray | float') -> 'np.ndarray':
    try:
        if any(np.iscomplexobj(x) for x in (b, weight, penalty)):
            raise ValueError('real inputs required')
        b = np.asarray(b, dtype=float)
        weight = np.asarray(weight, dtype=float)
        penalty = np.asarray(penalty, dtype=float)
        weight = np.broadcast_to(weight, b.shape)
        penalty = np.broadcast_to(penalty, b.shape)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('numeric inputs broadcastable to b required') from exc
    if (b.size == 0 or any(not np.isfinite(x).all() for x in (b, weight, penalty))
            or np.any(weight < 0) or np.any(penalty <= 0)):
        raise ValueError('invalid inputs')
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        c = weight / penalty
        threshold = 1.5 * np.power(c, 2.0 / 3.0)
    if (not np.isfinite(c).all() or not np.isfinite(threshold).all()
            or np.any((weight > 0) & (c == 0))):
        raise ValueError('effective ratio outside numerical contract')
    flat = b.ravel()
    cf = c.ravel()
    wf = weight.ravel()
    result = np.zeros_like(flat)
    identity = wf == 0
    result[identity] = flat[identity]
    active = (~identity) & (np.abs(flat) > threshold.ravel())
    mag = np.abs(flat[active])
    # Dimensionless cubic: avoid mag**1.5 and squaring a large root.
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        q = (cf[active] / mag) / np.sqrt(mag)
        arg = np.clip(-(3.0 * np.sqrt(3.0) / 4.0) * q, -1.0, 1.0)
        fraction = (4.0 / 3.0) * np.cos(np.arccos(arg) / 3.0)**2
        result[active] = np.sign(flat[active]) * (mag * fraction)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite proximal result')
    return result.reshape(b.shape)

import numpy as np

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

import numpy as np

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
    adapt_duals(products,auxiliaries,duals,rho,1.5,np.zeros(3))
    if any(not np.array_equal(group[j],group[j].T) for group in (auxiliaries,duals) for j in (0,2)): raise ValueError('asymmetric within-group blocks')
    B=tuple(a-w for a,w in zip(auxiliaries,duals))
    factor_terms(S,np.zeros_like(products[1]),U,V,B,rho,replicates)
    P,newalpha,_=detection_update(S,products[1],Z,alpha,replicates,detection_steps,1.)
    U,V=factor_sweep(S,P,U,V,B,rho,replicates,factor_steps,.1,1e-4)
    products=(U@U.T,U@V.T,V@V.T)
    newA=tuple(half_prox(m+w,l,r) for m,w,l,r in zip(products,duals,weights,rho))
    newrho,newW,_=adapt_duals(products,newA,duals,rho,1.5,np.full(3,.5/iteration**1.2))
    return U,V,newalpha,newA,newW,newrho,P

import numpy as np

def sparse_intensity(Y: 'np.ndarray', Z: 'np.ndarray', rank: int, p0: float, alpha0: 'np.ndarray', weights: 'np.ndarray', rho0: 'np.ndarray', outer_steps: int = 12, target: tuple = (1, 2)) -> float:
    S, U, V = spectral_start(Y, rank, p0)
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
    adapt_duals(A, A, W, rho, 1.5, np.zeros(3))
    if outer_steps == 0:
        detection_update(S, U @ V.T, Z, alpha, M, 1, 1.)
        factor_terms(S, np.zeros_like(S), U, V, A, rho, M)
    else:
        # Cycle 1: every primitive is a direct, operative dependency.
        P, alpha, _ = detection_update(S, U @ V.T, Z, alpha, M, 80, 1.)
        B = tuple(a-w for a, w in zip(A, W))
        before, _, _ = factor_terms(S, P, U, V, B, rho, M)
        U, V = factor_sweep(S, P, U, V, B, rho, M, 8, .1, 1e-4)
        after, _, _ = factor_terms(S, P, U, V, B, rho, M)
        if after > before + 1e-10*(1.0+abs(before)):
            raise ValueError('factor sweep increased the fixed-block objective')
        products = (U @ U.T, U @ V.T, V @ V.T)
        A = tuple(half_prox(m+w, lam, r)
                  for m, w, lam, r in zip(products, W, weights, rho))
        rho, W, _ = adapt_duals(products, A, W, rho, 1.5, np.full(3, .5))
        # Step 07 consumes the first cycle's factors, alpha, auxiliaries,
        # penalties and scaled duals; its returned state feeds the result.
        for k in range(2, outer_steps+1):
            U, V, alpha, A, W, rho, P = outer_cycle(
                S, M, Z, U, V, alpha, A, W, rho, weights, k, 80, 8)
    result = float((U @ V.T)[target])
    if not np.isfinite(result):
        raise ValueError('nonfinite intensity')
    return result
SCICODE_GOLD_EOF
