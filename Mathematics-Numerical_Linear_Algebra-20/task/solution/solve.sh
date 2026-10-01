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

def construct_banded_component(n: int) -> np.ndarray:
    if not isinstance(n, (int, np.integer)) or int(n) < 1:
        raise ValueError("n must be a positive integer")
    n = int(n)
    bands = np.zeros((6, n), dtype=np.float64)
    i = np.arange(1, n + 1, dtype=np.float64)
    bands[2] = 5.0 + 0.2*np.sin(0.01*i)
    if n >= 2:
        bands[1,1:] = 0.6
        bands[3,:-1] = -0.8
    if n >= 3:
        bands[0,2:] = -0.1
        bands[4,:-2] = 0.15
    if n >= 4:
        bands[5,:-3] = -0.05
    return bands

import numpy as np

def construct_semiseparable_generators(n: int):
    if not isinstance(n, (int, np.integer)) or int(n) < 1:
        raise ValueError("n must be a positive integer")
    n = int(n)
    i = np.arange(1, n + 1, dtype=np.float64)
    c = np.float64(1.0/np.sqrt(np.float64(n)))
    U = c*np.column_stack((1+0.2*np.sin(0.017*i), 0.7+0.15*np.cos(0.011*i)))
    V = c*np.column_stack((0.9+0.1*np.cos(0.013*i), -0.6+0.12*np.sin(0.019*i)))
    W = c*np.column_stack((0.8+0.18*np.sin(0.023*i), 0.5+0.10*np.cos(0.029*i)))
    S = c*np.column_stack((-0.7+0.11*np.cos(0.031*i), 0.6+0.14*np.sin(0.037*i)))
    return tuple(np.asarray(x,dtype=np.float64) for x in (U,V,W,S))

import numpy as np

def _band_entry(bands: np.ndarray, i: int, j: int) -> np.float64:
    d = j - i
    if d < -2 or d > 3:
        return np.float64(0.0)
    return np.float64(bands[d + 2, i])

def precompute_bps_quantities(
    bands: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    S: np.ndarray,
) -> tuple[dict, dict]:
    """Reference implementation."""
    bands = np.asarray(bands, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)

    if bands.ndim != 2 or bands.shape[0] != 6 or bands.shape[1] < 1:
        raise ValueError("bands must have shape (6, n), n >= 1")
    n = bands.shape[1]
    if U.shape != (n, 2) or V.shape != (n, 2):
        raise ValueError("U and V must each have shape (n, 2)")
    if W.shape != (n, 2) or S.shape != (n, 2):
        raise ValueError("W and S must each have shape (n, 2)")
    if not all(np.all(np.isfinite(x)) for x in (bands, U, V, W, S)):
        raise ValueError("all inputs must contain finite values")

    # Backward suffix Gram lookup.
    UU_lookup = np.empty((n, 2, 2), dtype=np.float64)
    gram = np.zeros((2, 2), dtype=np.float64)
    for k in range(n - 1, -1, -1):
        gram = gram + np.outer(U[k], U[k])
        UU_lookup[k] = gram

    # Forward prefix U W^T lookup.
    UW_lookup = np.empty((n, 2, 2), dtype=np.float64)
    prefix = np.zeros((2, 2), dtype=np.float64)
    for k in range(n):
        prefix = prefix + np.outer(U[k], W[k])
        UW_lookup[k] = prefix

    # B^T U in O(n) because the number of band diagonals is fixed.
    BTU = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        j0 = max(0, i - 2)
        j1 = min(n - 1, i + 3)
        for j in range(j0, j1 + 1):
            BTU[j] += _band_entry(bands, i, j) * U[i]

    # A^T U = B^T U + lower-semiseparable contribution
    #                   + upper-semiseparable contribution.
    A_T_U = BTU.copy()
    for j in range(n):
        if j + 1 < n:
            suffix_after = UU_lookup[j + 1]
            A_T_U[j] += suffix_after @ V[j]
        if j > 0:
            prefix_before = UW_lookup[j - 1]
            A_T_U[j] += prefix_before @ S[j]

    S_tilde = np.concatenate((S, A_T_U), axis=1)

    precomp = {
        "A_T_U": A_T_U,
        "S_tilde": S_tilde,
        "UU_lookup": UU_lookup,
        "UW_lookup": UW_lookup,
    }
    state = {
        "J": np.zeros((2, 2), dtype=np.float64),
        "K": np.zeros((2, 2), dtype=np.float64),
        "E": np.zeros((2, 5), dtype=np.float64),
        "X": np.zeros((2, 2), dtype=np.float64),
        "Y": np.zeros((2, 2), dtype=np.float64),
        "Z": np.zeros((2, 5), dtype=np.float64),
        "k": 0,
    }
    return precomp, state

import numpy as np

def _band_entry(bands: np.ndarray, i: int, j: int) -> np.float64:
    d = j - i
    if d < -2 or d > 3:
        return np.float64(0.0)
    return np.float64(bands[d + 2, i])

def form_structured_householder(
    bands: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    S: np.ndarray,
    precomp: dict,
    state: dict,
) -> dict:
    """Reference implementation."""
    bands=np.asarray(bands,dtype=np.float64)
    U=np.asarray(U,dtype=np.float64)
    V=np.asarray(V,dtype=np.float64)
    W=np.asarray(W,dtype=np.float64)
    S=np.asarray(S,dtype=np.float64)

    if not isinstance(precomp,dict) or not isinstance(state,dict):
        raise ValueError("precomp and state must be dictionaries")
    reqp={"A_T_U","S_tilde","UU_lookup","UW_lookup"}
    reqs={"J","K","E","X","Y","Z","k"}
    if not reqp.issubset(precomp) or not reqs.issubset(state):
        raise ValueError("missing required precomputation or state fields")

    n=U.shape[0]
    if n<1 or bands.shape!=(6,n) or V.shape!=U.shape or W.shape!=U.shape or S.shape!=U.shape or U.shape[1]!=2:
        raise ValueError("incompatible BPS input shapes")

    UU=np.asarray(precomp["UU_lookup"],dtype=np.float64)
    if UU.shape!=(n,2,2):
        raise ValueError("UU_lookup must have shape (n,2,2)")

    J=np.asarray(state["J"],dtype=np.float64)
    K=np.asarray(state["K"],dtype=np.float64)
    E=np.asarray(state["E"],dtype=np.float64)
    X=np.asarray(state["X"],dtype=np.float64)
    Y=np.asarray(state["Y"],dtype=np.float64)
    Z=np.asarray(state["Z"],dtype=np.float64)
    if J.shape!=(2,2) or K.shape!=(2,2) or E.shape!=(2,5) or X.shape!=(2,2) or Y.shape!=(2,2) or Z.shape!=(2,5):
        raise ValueError("invalid state matrix shapes")

    k=int(state["k"])
    if not (0<=k<n):
        raise ValueError("state k must satisfy 0 <= k < n")

    qdim=n-k
    gram=UU[k]
    u0=U[k]; v0=V[k]; s0=S[k]
    support=min(3,qdim)

    bcol=np.array([_band_entry(bands,k+d,k) for d in range(support)],dtype=np.float64)

    # U[k:n]^T A[k:n,k] for the original BPS block.
    g=U[k:k+support].T@bcol
    if qdim>1:
        gram_after=UU[k+1]
        g=g+gram_after@v0

    qvec=v0+J@s0+K@g+E[:,0]

    c=bcol.copy()
    sparse_rows=min(2,qdim)
    if sparse_rows:
        c[:sparse_rows]+=X[:sparse_rows]@s0+Y[:sparse_rows]@g+Z[:sparse_rows,0]
    c[0]-=np.dot(u0,v0)

    x0=np.float64(c[0]+np.dot(u0,qvec))
    norm_sq=np.dot(c,c)+2.0*np.dot(c,U[k:k+support]@qvec)+qvec@gram@qvec
    rho=np.float64(np.sqrt(max(float(norm_sq),0.0)))

    if rho==0.0:
        kbar=np.zeros(2,dtype=np.float64)
        bhat=np.zeros(support,dtype=np.float64); bhat[0]=1.0
        tau=np.float64(0.0)
        rkk=np.float64(0.0)
    else:
        sign=np.float64(1.0 if x0>=0.0 else -1.0)
        alpha=np.float64(x0+sign*rho)
        if alpha==0.0:
            raise ValueError("sign-stable Householder normalization produced zero alpha")

        kbar=qvec/alpha
        bhat=np.zeros(support,dtype=np.float64)
        bhat[0]=1.0-np.dot(u0,kbar)
        if support>1:
            bhat[1:]=c[1:]/alpha

        ynorm=kbar@gram@kbar+2.0*np.dot(bhat,U[k:k+support]@kbar)+np.dot(bhat,bhat)
        tau=np.float64(2.0/ynorm)
        rkk=np.float64(-sign*rho)

    return {
        "kbar":np.asarray(kbar,dtype=np.float64),
        "bhat":np.asarray(bhat,dtype=np.float64),
        "tau":float(tau),
        "rkk":float(rkk),
        "qvec":np.asarray(qvec,dtype=np.float64),
        "c":np.asarray(c,dtype=np.float64),
        "rho":float(rho),
    }

import numpy as np

def _band_entry(bands: np.ndarray, i: int, j: int) -> np.float64:
    d=j-i
    if d < -2 or d > 3:
        return np.float64(0.0)
    return np.float64(bands[d+2,i])


def _a_entry(bands,U,V,W,S,i,j):
    value=_band_entry(bands,i,j)
    if i>j:
        value+=np.dot(U[i],V[j])
    elif i<j:
        value+=np.dot(W[i],S[j])
    return np.float64(value)


def _local_uT_A_col(bands,U,S,precomp,k,j):
    """Compute U[k:n].T @ A[k:n,j] from global lookup data."""
    A_T_U=np.asarray(precomp["A_T_U"],dtype=np.float64)
    UW=np.asarray(precomp["UW_lookup"],dtype=np.float64)

    g=A_T_U[j].copy()
    if k>0:
        g-=UW[k-1]@S[j]
        for t in range(max(0,k-3),k):
            g-=U[t]*_band_entry(bands,t,j)
    return g


def _current_entry(bands,U,V,W,S,precomp,state,k,j):
    """Evaluate one entry in row k of the current trailing matrix."""
    J=np.asarray(state["J"]); K=np.asarray(state["K"]); E=np.asarray(state["E"])
    X=np.asarray(state["X"]); Y=np.asarray(state["Y"]); Z=np.asarray(state["Z"])
    d=j-k
    g=_local_uT_A_col(bands,U,S,precomp,k,j)

    value=_a_entry(bands,U,V,W,S,k,j)
    value+=U[k]@J@S[j]
    value+=U[k]@K@g
    if 0<=d<5:
        value+=U[k]@E[:,d]
    if X.shape[0]>0:
        value+=X[0]@S[j]+Y[0]@g
        if 0<=d<5:
            value+=Z[0,d]
    return np.float64(value)

def compute_structured_row_update(
    bands: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    S: np.ndarray,
    precomp: dict,
    state: dict,
    reflector: dict,
) -> dict:
    """Reference implementation."""
    bands=np.asarray(bands,dtype=np.float64)
    U=np.asarray(U,dtype=np.float64)
    V=np.asarray(V,dtype=np.float64)
    W=np.asarray(W,dtype=np.float64)
    S=np.asarray(S,dtype=np.float64)

    if not isinstance(precomp,dict) or not isinstance(state,dict) or not isinstance(reflector,dict):
        raise ValueError("precomp, state, and reflector must be dictionaries")
    reqp={"A_T_U","S_tilde","UU_lookup","UW_lookup"}
    reqs={"J","K","E","X","Y","Z","k"}
    reqr={"kbar","bhat","tau","rkk"}
    if not reqp.issubset(precomp) or not reqs.issubset(state) or not reqr.issubset(reflector):
        raise ValueError("missing required fields")

    n=U.shape[0]
    if bands.shape!=(6,n) or V.shape!=U.shape or W.shape!=U.shape or S.shape!=U.shape:
        raise ValueError("incompatible BPS input shapes")

    k=int(state["k"])
    if not (0<=k<n):
        raise ValueError("invalid state index")

    J=np.asarray(state["J"],dtype=np.float64)
    K=np.asarray(state["K"],dtype=np.float64)
    E=np.asarray(state["E"],dtype=np.float64)
    X=np.asarray(state["X"],dtype=np.float64)
    Y=np.asarray(state["Y"],dtype=np.float64)
    Z=np.asarray(state["Z"],dtype=np.float64)
    UU=np.asarray(precomp["UU_lookup"],dtype=np.float64)
    A_T_U=np.asarray(precomp["A_T_U"],dtype=np.float64)
    S_tilde=np.asarray(precomp["S_tilde"],dtype=np.float64)
    UW=np.asarray(precomp["UW_lookup"],dtype=np.float64)

    kbar=np.asarray(reflector["kbar"],dtype=np.float64)
    bhat=np.asarray(reflector["bhat"],dtype=np.float64)
    tau=np.float64(reflector["tau"])
    rkk=np.float64(reflector["rkk"])

    if J.shape!=(2,2) or K.shape!=(2,2) or E.shape!=(2,5) or X.shape!=(2,2) or Y.shape!=(2,2) or Z.shape!=(2,5):
        raise ValueError("invalid state matrix shapes")
    if kbar.shape!=(2,) or bhat.ndim!=1 or bhat.size<1 or S_tilde.shape!=(n,4):
        raise ValueError("invalid reflector or precomputation shapes")

    qdim=n-k
    support=min(3,qdim)
    if bhat.size!=support:
        raise ValueError("bhat length is inconsistent with the active dimension")

    gram=UU[k]
    u0=U[k]; w0=W[k]

    # y = U_local @ kbar + bhat on its short support.
    y_head=U[k:k+support]@kbar+bhat
    uTy=gram@kbar+U[k:k+support].T@bhat
    y_sparse=y_head[:min(2,qdim)]

    # Coefficients of y^T times the current structured perturbation.
    wS_pert=uTy@J
    wU_pert=uTy@K
    if y_sparse.size:
        wS_pert+=y_sparse@X[:y_sparse.size]
        wU_pert+=y_sparse@Y[:y_sparse.size]

    sparse_pert_tail=np.zeros(5,dtype=np.float64)
    local_sparse_cols=min(5,qdim)
    if local_sparse_cols:
        s_local=uTy@E[:,:local_sparse_cols]
        if y_sparse.size:
            s_local+=y_sparse@Z[:y_sparse.size,:local_sparse_cols]
        if local_sparse_cols>1:
            sparse_pert_tail[:local_sparse_cols-1]=s_local[1:]

    # Coefficients of y^T A for the original BPS block.
    wU_yA=kbar+wU_pert
    scalar_u0=np.float64(np.dot(wU_yA,u0))
    bhat_w=bhat@W[k:k+support]
    wS_yA=wS_pert+scalar_u0*w0+bhat_w

    max_tail=min(5,qdim-1)
    sparse_from_bhat=np.zeros(5,dtype=np.float64)
    band_row_tail=np.zeros(5,dtype=np.float64)
    for d in range(1,max_tail+1):
        j=k+d
        direct=np.float64(0.0)
        for a in range(support):
            direct+=bhat[a]*_a_entry(bands,U,V,W,S,k+a,j)
        sparse_from_bhat[d-1]=direct-np.dot(bhat_w,S[j])
        band_row_tail[d-1]=_band_entry(bands,k,j)

    sparse_yA=sparse_from_bhat+scalar_u0*band_row_tail+sparse_pert_tail

    # Current first-row generator coefficients relative to [S, A^T U].
    coeff_U = u0@K + Y[0]
    coeff_S = w0 + u0@J + X[0]
    if k>0:
        coeff_S = coeff_S - coeff_U @ UW[k-1]

    # H = I - tau*y*y^T and y[0] = 1.
    w_factor_S = coeff_S - tau*wS_yA
    w_factor_U = coeff_U - tau*wU_yA
    w_tilde=np.concatenate((w_factor_S,w_factor_U))

    dlen=min(6,qdim)
    d_tilde=np.zeros(dlen,dtype=np.float64)
    d_tilde[0]=rkk
    for d in range(1,dlen):
        j=k+d
        transformed=_current_entry(bands,U,V,W,S,precomp,state,k,j)
        yTA=sparse_yA[d-1]+wS_yA@S[j]+wU_yA@A_T_U[j]
        transformed-=tau*yTA
        d_tilde[d]=transformed-w_tilde@S_tilde[j]

    return {
        "w_tilde":np.asarray(w_tilde,dtype=np.float64),
        "d_tilde":np.asarray(d_tilde,dtype=np.float64),
        "wS_yA":np.asarray(wS_yA,dtype=np.float64),
        "wU_yA":np.asarray(wU_yA,dtype=np.float64),
        "sparse_yA":np.asarray(sparse_yA,dtype=np.float64),
        "band_row_tail":np.asarray(band_row_tail,dtype=np.float64),
    }

import numpy as np

def update_structured_perturbation(
    U: np.ndarray,
    W: np.ndarray,
    state: dict,
    reflector: dict,
    row_update: dict,
) -> dict:
    """Reference implementation."""
    U=np.asarray(U,dtype=np.float64)
    W=np.asarray(W,dtype=np.float64)
    if U.ndim!=2 or U.shape[1]!=2 or W.shape!=U.shape or U.shape[0]<1:
        raise ValueError("U and W must have matching shape (n,2)")
    if not isinstance(state,dict) or not isinstance(reflector,dict) or not isinstance(row_update,dict):
        raise ValueError("state, reflector, and row_update must be dictionaries")

    reqs={"J","K","E","X","Y","Z","k"}
    reqr={"kbar","bhat","tau"}
    reqw={"wS_yA","wU_yA","sparse_yA","band_row_tail"}
    if not reqs.issubset(state) or not reqr.issubset(reflector) or not reqw.issubset(row_update):
        raise ValueError("missing required fields")

    k=int(state["k"])
    if not (0<=k<U.shape[0]):
        raise ValueError("invalid state index")

    J=np.asarray(state["J"],dtype=np.float64)
    K=np.asarray(state["K"],dtype=np.float64)
    E=np.asarray(state["E"],dtype=np.float64)
    X=np.asarray(state["X"],dtype=np.float64)
    Y=np.asarray(state["Y"],dtype=np.float64)
    Z=np.asarray(state["Z"],dtype=np.float64)

    kbar=np.asarray(reflector["kbar"],dtype=np.float64)
    bhat=np.asarray(reflector["bhat"],dtype=np.float64)
    tau=np.float64(reflector["tau"])
    wS=np.asarray(row_update["wS_yA"],dtype=np.float64)
    wU=np.asarray(row_update["wU_yA"],dtype=np.float64)
    sparse=np.asarray(row_update["sparse_yA"],dtype=np.float64)
    band_tail=np.asarray(row_update["band_row_tail"],dtype=np.float64)

    if J.shape!=(2,2) or K.shape!=(2,2) or E.shape!=(2,5) or X.shape!=(2,2) or Y.shape!=(2,2) or Z.shape!=(2,5):
        raise ValueError("invalid state matrix shapes")
    if kbar.shape!=(2,) or wS.shape!=(2,) or wU.shape!=(2,) or sparse.shape!=(5,) or band_tail.shape!=(5,):
        raise ValueError("invalid reflector or row-update shapes")

    u0=U[k]; w0=W[k]

    # Restrict the old perturbation to the next principal block.
    J0=J+np.outer(K@u0,w0)
    K0=K.copy()

    E0=np.zeros_like(E)
    X0=np.zeros_like(X)
    Y0=np.zeros_like(Y)
    Z0=np.zeros_like(Z)

    E0[:,:4]=E[:,1:]
    E0+=np.outer(K@u0,band_tail)

    X0[:1]=X[1:]
    Y0[:1]=Y[1:]
    restricted_y_u0=Y[1:]@u0
    X0[:1]+=np.outer(restricted_y_u0,w0)

    Z0[:1,:4]=Z[1:,1:]
    Z0[:1]+=np.outer(restricted_y_u0,band_tail)

    b_tail=np.zeros(2,dtype=np.float64)
    available=min(2,U.shape[0]-k-1)
    if available:
        if bhat.size < available+1:
            raise ValueError("bhat is too short for the active bandwidth")
        b_tail[:available]=bhat[1:1+available]

    return {
        "J":J0-tau*np.outer(kbar,wS),
        "K":K0-tau*np.outer(kbar,wU),
        "E":E0-tau*np.outer(kbar,sparse),
        "X":X0-tau*np.outer(b_tail,wS),
        "Y":Y0-tau*np.outer(b_tail,wU),
        "Z":Z0-tau*np.outer(b_tail,sparse),
        "k":k+1,
    }

import numpy as np

def compute_target_tau(
    n: int = 100000,
    target_index: int = 73129,
) -> float:
    """Reference end-to-end implementation composed from Steps 1-6."""
    if not isinstance(n,(int,np.integer)) or int(n)<1:
        raise ValueError("n must be a positive integer")
    if not isinstance(target_index,(int,np.integer)):
        raise ValueError("target_index must be an integer")
    n=int(n); target_index=int(target_index)
    if not (1<=target_index<=n):
        raise ValueError("target_index must lie in [1,n]")
    
    bands=construct_banded_component(n)
    U,V,W,S=construct_semiseparable_generators(n)
    precomp,state=precompute_bps_quantities(bands,U,V,W,S)

    tau_target=None
    for q in range(target_index):
        reflector=form_structured_householder(
            bands,U,V,W,S,precomp,state
        )
        row_update=compute_structured_row_update(
            bands,U,V,W,S,precomp,state,reflector
        )
        tau_target=float(reflector["tau"])

        if q<target_index-1:
            state=update_structured_perturbation(
                U,W,state,reflector,row_update
            )

    return float(tau_target)
SCICODE_GOLD_EOF
