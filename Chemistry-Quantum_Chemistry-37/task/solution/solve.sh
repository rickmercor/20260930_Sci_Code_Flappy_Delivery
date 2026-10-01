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

def model_jet(a: float = 0.0, b: float = 0.0) -> np.ndarray:
    """Construct the four-configuration Hamiltonian and its control derivatives.

    Parameters
    ----------
    a, b : float
        Finite real scalar controls. In units of |t|, K_alpha=3.2+a,
        K_beta=1.5+b, t=-1, t_prime=-1.5, U=3, U_prime=6.
        Basis order is (alpha, alpha_prime, beta, beta_prime).

    Returns
    -------
    result : np.ndarray
        Shape (4,4,4), with slots (H, H_a, H_b, H_ab). Slots contain
        actual derivatives, not Taylor coefficients. The diagonal is
        (0,0,3,6); both reference configurations couple to both perturbers
        with t and t_prime respectively.

    Raises
    ------
    ValueError
        If either control is not a finite real scalar convertible to float,
        or construction produces nonfinite entries.
    """
    try:
        x = np.asarray([a, b])
        if x.shape != (2,) or np.iscomplexobj(x):
            raise ValueError('real scalar controls required')
        a, b = np.asarray(x, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('invalid controls') from exc
    if not np.isfinite([a, b]).all():
        raise ValueError('finite controls required')
    result = np.zeros((4, 4, 4))
    result[0] = [[0,3.2+a,-1,-1], [3.2+a,0,-1.5,-1.5],
                 [-1,-1.5,3,1.5+b], [-1,-1.5,1.5+b,6]]
    result[1,0,1] = result[1,1,0] = 1
    result[2,2,3] = result[2,3,2] = 1
    if not np.isfinite(result).all():
        raise ValueError('nonfinite Hamiltonian')
    return result

import numpy as np

def select_space(H: np.ndarray, e: np.ndarray, k: int,
                         rho: float = 0.4, enrich: float = 0.6) -> np.ndarray:
    """Select a candidate model space with secondary enrichment closure.

    Parameters
    ----------
    H : np.ndarray
        Finite real symmetric (n,n) Hamiltonian, n>=1; symmetry atol=1e-12.
    e : np.ndarray
        Finite real reference energies of shape (n,). These need not equal
        diag(H). Indices below k are frozen and cannot enter the model space.
    k : int
        Candidate index, 0<=k<n; booleans are not accepted as integers.
    rho, enrich : float
        Finite real scalars satisfying 0<rho<enrich. Starting with k, include
        every unfrozen j whose |H[k,j]/(e[k]-e[j])| exceeds rho, including
        exact degeneracies regardless of coupling. Enrich to closure using
        the same pair ratio and threshold enrich. All tests are strict >;
        exact degeneracies always qualify. Never divide a zero denominator.

    Returns
    -------
    result : np.ndarray
        Sorted, distinct integer indices of the selected space, including k.

    Raises
    ------
    ValueError
        If arrays are not convertible to finite real arrays of the stated
        shapes, H is not symmetric within atol=1e-12 and rtol=0, k is invalid,
        or thresholds are not finite real scalars with 0<rho<enrich.
    """
    try:
        if np.iscomplexobj(H) or np.iscomplexobj(e):
            raise ValueError('complex arrays')
        H, e = np.asarray(H, float), np.asarray(e, float)
        t = np.asarray([rho, enrich])
        if t.shape != (2,) or np.iscomplexobj(t):
            raise ValueError('invalid thresholds')
        rho, enrich = np.asarray(t, float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('invalid numeric inputs') from exc
    if H.ndim != 2 or H.shape[0] < 1 or H.shape[0] != H.shape[1]:
        raise ValueError('invalid H shape')
    n = H.shape[0]
    if e.shape != (n,) or not np.isfinite(H).all() or not np.isfinite(e).all():
        raise ValueError('invalid energies or entries')
    if not np.allclose(H, H.T, atol=1e-12, rtol=0):
        raise ValueError('asymmetric H')
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 0 <= k < n:
        raise ValueError('invalid candidate')
    if not np.isfinite([rho, enrich]).all() or not 0 < rho < enrich:
        raise ValueError('invalid thresholds')
    d = np.abs(e[:,None] - e[None,:])
    r = np.full((n,n), np.inf)
    np.divide(np.abs(H), d, out=r, where=d != 0)
    np.fill_diagonal(r, 0)
    chosen = {k} | {j for j in range(k+1,n) if r[k,j] > rho}
    while True:
        expanded = chosen | {j for j in range(k,n) if j not in chosen
                             and any(r[i,j] > enrich for i in chosen)}
        if expanded == chosen:
            break
        chosen = expanded
    return np.array(sorted(chosen), dtype=int)

import numpy as np

def effective_jet(H: np.ndarray, e: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Construct the symmetrized second-order effective-Hamiltonian jet.

    Parameters
    ----------
    H : np.ndarray
        Finite real (4,n,n) jet, n>=1, each slot symmetric to atol=1e-12.
    e : np.ndarray
        Finite real reference-energy jet (4,n).
    P : np.ndarray
        Nonempty, strictly increasing integer index array in [0,n).
        Q is its complement. Slots are (value,a,b,ab), actual derivatives.
        For i,j in P the matrix is H_ij plus one half the sum over q in Q
        of H_iq H_qj [(e_i-e_q)^(-1)+(e_j-e_q)^(-1)]. Differentiate this
        expression through mixed order with P fixed. A full P gives H.

    Returns
    -------
    result : np.ndarray
        Shape (4,len(P),len(P)), the effective matrix and its derivatives.

    Raises
    ------
    ValueError
        If arrays cannot be converted to finite real arrays, shapes or P
        indices are invalid, any H slot fails symmetry at atol=1e-12,
        any baseline P-Q reference-energy separation is <=1e-10 in absolute
        value, or the calculated result is nonfinite.
    """
    try:
        if any(np.iscomplexobj(x) for x in (H,e,P)):
            raise ValueError('complex input')
        H, e, P = np.asarray(H,float), np.asarray(e,float), np.asarray(P)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('invalid arrays') from exc
    if H.ndim != 3 or H.shape[0] != 4 or H.shape[1] < 1 or H.shape[1] != H.shape[2]:
        raise ValueError('invalid H shape')
    n = H.shape[1]
    if e.shape != (4,n) or not np.isfinite(H).all() or not np.isfinite(e).all():
        raise ValueError('invalid energy jet')
    if not np.allclose(H,H.transpose(0,2,1),atol=1e-12,rtol=0):
        raise ValueError('asymmetric jet')
    if P.ndim != 1 or P.size == 0 or P.dtype.kind not in 'iu' or np.any(P<0) or np.any(P>=n) or np.any(np.diff(P.astype(int))<=0):
        raise ValueError('invalid P')
    Q = [q for q in range(n) if q not in P]
    # Product tensor for actual value/a/b/ab derivatives.
    C = np.zeros((4,4,4))
    for r,s,t in [(0,0,0),(1,1,0),(1,0,1),(2,2,0),(2,0,2),
                  (3,3,0),(3,1,2),(3,2,1),(3,0,3)]:
        C[r,s,t] = 1
    result = H[:,P,:][:,:,P].copy()
    for ii,i in enumerate(P):
        for jj,j in enumerate(P):
            for q in Q:
                d = e[:,j] - e[:,q]
                if abs(d[0]) <= 1e-10:
                    raise ValueError('singular reference denominator')
                reciprocal = np.array([1/d[0], -d[1]/d[0]**2,
                    -d[2]/d[0]**2, 2*d[1]*d[2]/d[0]**3-d[3]/d[0]**2])
                product = np.einsum('rst,s,t->r', C, H[:,i,q], H[:,q,j])
                result[:,ii,jj] += np.einsum('rst,s,t->r', C, product, reciprocal)
    result = (result + result.transpose(0,2,1))/2
    if not np.isfinite(result).all():
        raise ValueError('nonfinite effective jet')
    return result

import numpy as np

def eigensystem_jet(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate a simple symmetric eigensystem through mixed order.

    Parameters
    ----------
    A : np.ndarray
        Finite real symmetric jet (4,n,n), n>=1, in (value,a,b,ab) slots.
        Every slot must be symmetric to atol=1e-12, rtol=0. Baseline
        eigenvalues must have pairwise separations >1e-10. Eigenvectors
        are columns ordered by increasing eigenvalue. Fix each column's
        sign by making its largest-absolute baseline entry positive,
        choosing the lowest row index if tied. Derivatives use a smooth
        local continuation of this sign, not differentiation of argmax.

    Returns
    -------
    result : tuple[np.ndarray,np.ndarray]
        Eigenvalue jet (4,n) and normalized eigenvector jet (4,n,n).
        Include eigenvector response and mixed normalization terms.

    Raises
    ------
    ValueError
        If input is not convertible to a finite real jet of the stated
        shape and symmetry, baseline eigenvalues are separated by <=1e-10,
        the eigensolver fails, or the calculated result is nonfinite.
    """
    try:
        if np.iscomplexobj(A):
            raise ValueError('complex input')
        A = np.asarray(A,float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('invalid A') from exc
    if A.ndim != 3 or A.shape[0] != 4 or A.shape[1] < 1 or A.shape[1] != A.shape[2]:
        raise ValueError('invalid shape')
    if not np.isfinite(A).all() or not np.allclose(A,A.transpose(0,2,1),atol=1e-12,rtol=0):
        raise ValueError('nonfinite or asymmetric jet')
    try:
        lam,V = np.linalg.eigh(A[0])
    except np.linalg.LinAlgError as exc:
        raise ValueError('eigensolver failed') from exc
    if np.any(np.diff(lam)<=1e-10):
        raise ValueError('spectrum not simple')
    n = len(lam)
    for i in range(n):
        if V[np.argmax(np.abs(V[:,i])),i] < 0:
            V[:,i] *= -1
    L,R = np.zeros((4,n)), np.zeros((4,n,n))
    L[0],R[0] = lam,V
    for i in range(n):
        v = V[:,i]
        for d in (1,2):
            L[d,i] = v@A[d]@v
            for j in range(n):
                if j != i:
                    R[d,:,i] += V[:,j]*(V[:,j]@A[d]@v)/(lam[i]-lam[j])
        va,vb = R[1,:,i],R[2,:,i]
        L[3,i] = v@A[3]@v + va@A[2]@v + vb@A[1]@v
        z = A[3]@v + (A[1]-L[1,i]*np.eye(n))@vb + (A[2]-L[2,i]*np.eye(n))@va
        R[3,:,i] = -(va@vb)*v
        for j in range(n):
            if j != i:
                R[3,:,i] += V[:,j]*(V[:,j]@z)/(lam[i]-lam[j])
    if not np.isfinite(L).all() or not np.isfinite(R).all():
        raise ValueError('nonfinite eigensystem jet')
    return L,R

import numpy as np

def update_partition(H: np.ndarray, e: np.ndarray, P: np.ndarray,
                             L: np.ndarray, V: np.ndarray, frozen: int = 0
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a Hamiltonian jet and update its reference partition.

    Parameters
    ----------
    H, e : np.ndarray
        Finite real jets of shapes (4,n,n) and (4,n), n>=1. Each H slot
        is symmetric to atol=1e-12. Slots are (value,a,b,ab).
    P : np.ndarray
        Nonempty strictly increasing integer indices, all in [frozen,n).
    L, V : np.ndarray
        Finite real eigenvalue and eigenvector jets of shapes (4,p) and
        (4,p,p), p=len(P). V must be orthogonal through mixed order:
        the four jet slots of V.T@V equal (I,0,0,0) to atol=1e-9.
        Embed V on P and identity on its complement, transform H, replace
        e[:,P] with L, and retain every other reference energy unchanged.
    frozen : int
        Integer 0<=frozen<n (not bool). Preserve the prefix of this length;
        stably sort the remaining basis by updated baseline reference energy.

    Returns
    -------
    result : tuple[np.ndarray,np.ndarray]
        Transformed Hamiltonian jet (4,n,n) and reference jet (4,n), with
        one consistent permutation applied to every derivative slot.

    Raises
    ------
    ValueError
        If arrays are not convertible to finite real arrays of the stated
        shapes, H lacks the stated symmetry, P or frozen is invalid,
        V fails the stated derivative orthogonality test, or outputs
        contain nonfinite entries.
    """
    try:
        if any(np.iscomplexobj(x) for x in (H,e,P,L,V)):
            raise ValueError('complex input')
        H,e,L,V = (np.asarray(x,float) for x in (H,e,L,V))
        P = np.asarray(P)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('invalid arrays') from exc
    if H.ndim != 3 or H.shape[0] != 4 or H.shape[1] < 1 or H.shape[1] != H.shape[2]:
        raise ValueError('invalid H shape')
    n = H.shape[1]
    if isinstance(frozen,(bool,np.bool_)) or not isinstance(frozen,(int,np.integer)) or not 0<=frozen<n:
        raise ValueError('invalid frozen prefix')
    if P.ndim != 1 or P.size == 0 or P.dtype.kind not in 'iu' or np.any(P<frozen) or np.any(P>=n) or np.any(np.diff(P.astype(int))<=0):
        raise ValueError('invalid P')
    p = len(P)
    if e.shape != (4,n) or L.shape != (4,p) or V.shape != (4,p,p):
        raise ValueError('incompatible jet shapes')
    if not all(np.isfinite(x).all() for x in (H,e,L,V)) or not np.allclose(H,H.transpose(0,2,1),atol=1e-12,rtol=0):
        raise ValueError('nonfinite or asymmetric jet')
    C = np.zeros((4,4,4))
    for r,s,t in [(0,0,0),(1,1,0),(1,0,1),(2,2,0),(2,0,2),
                  (3,3,0),(3,1,2),(3,2,1),(3,0,3)]:
        C[r,s,t] = 1
    gram = np.einsum('rst,sji,tjk->rik',C,V,V)
    expected = np.zeros_like(gram); expected[0] = np.eye(p)
    if not np.allclose(gram,expected,atol=1e-9,rtol=0):
        raise ValueError('V is not an orthogonal jet')
    R = np.zeros_like(H); R[0] = np.eye(n)
    for i,ii in enumerate(P):
        for j,jj in enumerate(P):
            R[:,ii,jj] = V[:,i,j]
    HR = np.einsum('rst,sij,tjk->rik',C,H,R)
    transformed = np.einsum('rst,sji,tjk->rik',C,R,HR)
    transformed = (transformed+transformed.transpose(0,2,1))/2
    updated = e.copy(); updated[:,P] = L
    order = np.r_[np.arange(frozen), frozen+np.argsort(updated[0,frozen:],kind='stable')]
    transformed,updated = transformed[:,order,:][:,:,order],updated[:,order]
    if not np.isfinite(transformed).all() or not np.isfinite(updated).all():
        raise ValueError('nonfinite updated partition')
    return transformed,updated

import numpy as np

def bw_root_jet(H: np.ndarray, e: np.ndarray, k: int) -> np.ndarray:
    """Solve and analytically differentiate the selected second-order BW root.

    Parameters
    ----------
    H, e : np.ndarray
        Finite real jets (4,n,n) and (4,n), n>=1; each H slot symmetric
        to atol=1e-12. Slots contain actual derivatives (value,a,b,ab).
    k : int
        Integer index 0<=k<n, not bool. Solve
        F(z)=z-H_kk-sum_{j!=k} H_kj**2/(z-e_j)=0 in the open pole interval
        containing e_k. The interval may be unbounded; all e_j, j!=k,
        are treated as denominator poles even for zero coupling.
        Include derivatives of H, e and the implicit root. No finite
        differences. With n=1, return H[:,0,0].

    Returns
    -------
    result : np.ndarray
        Four-vector (root, root_a, root_b, root_ab).

    Raises
    ------
    ValueError
        If arrays are not convertible to finite real arrays of the stated
        shapes and symmetry, k is invalid, e_k lies within 1e-10 of another
        reference energy, the selected interval does not contain exactly
        one real root separated from both finite bounds by >1e-10,
        the baseline eigensolver fails, or the output is nonfinite.
    """
    try:
        if np.iscomplexobj(H) or np.iscomplexobj(e):
            raise ValueError('complex input')
        H,e = np.asarray(H,float),np.asarray(e,float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('invalid arrays') from exc
    if H.ndim != 3 or H.shape[0] != 4 or H.shape[1] < 1 or H.shape[1] != H.shape[2]:
        raise ValueError('invalid H shape')
    n = H.shape[1]
    if e.shape != (4,n) or not np.isfinite(H).all() or not np.isfinite(e).all() or not np.allclose(H,H.transpose(0,2,1),atol=1e-12,rtol=0):
        raise ValueError('invalid or asymmetric jet')
    if isinstance(k,(bool,np.bool_)) or not isinstance(k,(int,np.integer)) or not 0<=k<n:
        raise ValueError('invalid candidate')
    J = [j for j in range(n) if j != k]
    poles = e[:,J]
    if np.any(np.abs(e[0,k]-poles[0])<=1e-10):
        raise ValueError('reference at a pole')
    lo = max((p for p in poles[0] if p<e[0,k]),default=-np.inf)
    hi = min((p for p in poles[0] if p>e[0,k]),default=np.inf)
    arrow = np.diag(np.r_[H[0,k,k],poles[0]])
    arrow[0,1:] = H[0,k,J]; arrow[1:,0] = H[0,k,J]
    try:
        roots = np.linalg.eigvalsh(arrow)
    except np.linalg.LinAlgError as exc:
        raise ValueError('root eigensolver failed') from exc
    candidates = roots[(roots>lo+1e-10)&(roots<hi-1e-10)]
    if len(candidates) != 1:
        raise ValueError('no unique interior root')
    C = np.zeros((4,4,4))
    for r,s,t in [(0,0,0),(1,1,0),(1,0,1),(2,2,0),(2,0,2),
                  (3,3,0),(3,1,2),(3,2,1),(3,0,3)]:
        C[r,s,t] = 1
    w = H[:,k,J]
    c = np.einsum('rst,sj,tj->rj',C,w,w)
    result = np.array([candidates[0],0.,0.,0.])
    fz = 1+np.sum(c[0]/(result[0]-poles[0])**2)
    # First evaluate partial a,b terms, then the total mixed residual
    # with the known root_a/root_b but root_ab set to zero.
    for stage in range(2):
        d = result[:,None]-poles
        reciprocal = np.array([1/d[0],-d[1]/d[0]**2,-d[2]/d[0]**2,
                                2*d[1]*d[2]/d[0]**3-d[3]/d[0]**2])
        residual = result-H[:,k,k]-np.einsum('rst,sj,tj->rj',C,c,reciprocal).sum(axis=1)
        if stage == 0:
            result[1:3] = -residual[1:3]/fz
        else:
            result[3] = -residual[3]/fz
    if not np.isfinite(result).all():
        raise ValueError('nonfinite root jet')
    return result

import numpy as np

def ssrsbw_jet(H: np.ndarray, targets: int = 3,
                       rho: float = 0.4, enrich: float = 0.6,
                       max_updates: int = 50) -> np.ndarray:
    """Optimize states sequentially and return corrected energy derivatives.

    Parameters
    ----------
    H : np.ndarray
        Finite real symmetric Hamiltonian jet (4,n,n), n>=1, in actual
        (value,a,b,ab) derivatives. Symmetry atol=1e-12, rtol=0.
        Initially e is the diagonal jet of H. Stably order the initial
        basis by e[0], preserving input order for ties.
    targets : int
        Number of targeted states, 1<=targets<=n, not bool.
    rho, enrich : float
        Finite real scalars 0<rho<enrich, used by select_space.
    max_updates : int
        Positive integer, not bool; maximum transformations per target.
        For each target, use select_space, effective_jet, eigensystem_jet
        and update_partition until the selected space is a singleton.
        Compute bw_root_jet immediately, then freeze the optimized reference
        (not its corrected energy). Frozen states remain external perturbers.
        Finally sort the stored corrected energy jets by their baseline.
        Derivatives describe the selected fixed branch; do not differentiate
        the discrete selection decisions and do not use finite differences.

    Returns
    -------
    result : np.ndarray
        Shape (4,targets): sorted corrected energies and control derivatives.

    Raises
    ------
    ValueError
        If input shape, finiteness, reality, symmetry, integer controls or
        thresholds violate the conditions above; a required P-Q denominator
        has magnitude <=1e-10; an effective spectrum has separation <=1e-10;
        a rotation jet fails orthogonality at atol=1e-9; a BW reference lies
        within 1e-10 of a pole or lacks exactly one root >1e-10 from finite
        interval bounds; a numerical eigensolver fails; calculations become
        nonfinite; more than max_updates transformations are needed for a
        target; or final corrected energies have separation <=1e-10.
    """
    try:
        if np.iscomplexobj(H):
            raise ValueError('complex H')
        H = np.asarray(H,float).copy()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('invalid H') from exc
    if H.ndim != 3 or H.shape[0] != 4 or H.shape[1] < 1 or H.shape[1] != H.shape[2]:
        raise ValueError('invalid H shape')
    n = H.shape[1]
    if not np.isfinite(H).all() or not np.allclose(H,H.transpose(0,2,1),atol=1e-12,rtol=0):
        raise ValueError('invalid jet')
    if isinstance(targets,(bool,np.bool_)) or not isinstance(targets,(int,np.integer)) or not 1<=targets<=n:
        raise ValueError('invalid targets')
    if isinstance(max_updates,(bool,np.bool_)) or not isinstance(max_updates,(int,np.integer)) or max_updates<1:
        raise ValueError('invalid max_updates')
    e = np.array([np.diag(slot) for slot in H])
    order = np.argsort(e[0],kind='stable')
    H,e = H[:,order,:][:,:,order],e[:,order]
    corrected = []
    for k in range(targets):
        updates = 0
        while True:
            P = select_space(H[0],e[0],k,rho,enrich)
            if len(P) == 1:
                break
            if updates == max_updates:
                raise ValueError('transformation limit exceeded')
            A = effective_jet(H,e,P)
            L,V = eigensystem_jet(A)
            H,e = update_partition(H,e,P,L,V,frozen=k)
            updates += 1
        corrected.append(bw_root_jet(H,e,k))
    result = np.stack(corrected,axis=1)
    result = result[:,np.argsort(result[0],kind='stable')]
    if np.any(np.diff(result[0])<=1e-10):
        raise ValueError('corrected spectrum not simple')
    return result

import numpy as np

def gap_response(a: float = 0.0, b: float = 0.0) -> float:
    """Return the mixed response of the specified corrected excitation gap.

    Parameters
    ----------
    a, b : float
        Finite real scalar controls with |a|<=0.001 and |b|<=0.001. This
        neighborhood retains the benchmark's selection branch. Build the
        published four-state model with model_jet and use ssrsbw_jet with
        targets=3, rho=0.4, enrich=0.6, max_updates=50.

    Returns
    -------
    result : float
        Native Python float d_a d_b (epsilon_2-epsilon_1), in |t| units.
        Return the unrounded value. Use analytic derivative propagation.

    Raises
    ------
    ValueError
        If controls are not finite real scalars convertible to float or
        exceed the stated neighborhood; a numerical eigensolver fails;
        computations become nonfinite; a required reference denominator,
        effective spectral gap, or corrected spectral gap has magnitude
        <=1e-10; a rotation jet fails orthogonality at atol=1e-9; a BW
        reference is within 1e-10 of a pole or its interval lacks exactly
        one root >1e-10 from finite bounds; or a target needs more than
        50 transformations.
    """
    H = model_jet(a,b)
    if abs(float(a))>0.001 or abs(float(b))>0.001:
        raise ValueError('controls outside benchmark neighborhood')
    energies = ssrsbw_jet(H,targets=3,rho=0.4,enrich=0.6,max_updates=50)
    return float(energies[3,2]-energies[3,1])
SCICODE_GOLD_EOF
