"""
Compute the structured upper-row update produced by one Householder transformation.

This step computes the transformed factor row produced by applying the current Householder reflector while preserving the compact BPS representation. The product of the reflector vector with the current trailing matrix is decomposed into coefficients associated with the structured upper generators plus a short local banded contribution. Those quantities are combined with the current row to form the updated upper semiseparable generator row and its banded residual. The same decomposition also produces the information required to update the structured perturbation state in the next step.

Returns
-------
The row update dictionary containing w_tilde, d_tilde, and y^T @ A coefficients.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

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
    """
    Compute the upper semiseparable and banded row produced at one QR step.

    Parameters
    ----------
    bands : np.ndarray
        Compact banded component with shape (6, n).
    U, V, W, S : np.ndarray
        Semiseparable generators, each with shape (n, 2).
    precomp : dict
        Contains A_T_U, S_tilde, UU_lookup, and UW_lookup.
    state : dict
        Current structured perturbation state.
    reflector : dict
        Contains kbar, bhat, tau, and rkk from Step 4.

    Returns
    -------
    row_update : dict
        Contains w_tilde with shape (4,), d_tilde with length at most 6,
        and compact y.T @ A coefficients used by Step 6.

    Raises
    ------
    ValueError
        If the BPS inputs, precomputed quantities, structured state, or 
        reflector data are incomplete, dimensionally inconsistent, or 
        incompatible with the active QR step.
    """
    return row_update

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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

def _oracle_compute_structured_row_update(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid test case specifications."""
    return [
        {
            "setup": """import numpy as np
n=4
bands=np.zeros((6,n)); bands[2]=5.; bands[1,1:]=.6; bands[3,:-1]=-.8
U=np.ones((n,2))/np.sqrt(n); V=U.copy(); W=U.copy(); S=U.copy()
UU=np.empty((n,2,2)); g=np.zeros((2,2))
for k in range(n-1,-1,-1): g=g+np.outer(U[k],U[k]); UU[k]=g
UW=np.empty((n,2,2)); p=np.zeros((2,2))
for k in range(n): p=p+np.outer(U[k],W[k]); UW[k]=p
precomp={"A_T_U":np.zeros((n,2)),"S_tilde":np.concatenate((S,np.zeros((n,2))),axis=1),
         "UU_lookup":UU,"UW_lookup":UW}
state={"J":np.zeros((2,2)),"K":np.zeros((2,2)),"E":np.zeros((2,5)),
       "X":np.zeros((2,2)),"Y":np.zeros((2,2)),"Z":np.zeros((2,5)),"k":0}
reflector={"kbar":np.zeros(2),"bhat":np.array([1.,0.,0.]),"tau":0.,"rkk":-5.}""",
            "call":'(lambda o: np.concatenate([np.asarray(o["w_tilde"]).reshape(-1),np.asarray(o["d_tilde"]).reshape(-1),np.asarray(o["wS_yA"]).reshape(-1),np.asarray(o["wU_yA"]).reshape(-1),np.asarray(o["sparse_yA"]).reshape(-1),np.asarray(o["band_row_tail"]).reshape(-1)]).tolist())(compute_structured_row_update(bands,U,V,W,S,precomp,state,reflector))',
            "gold_call":'(lambda o: np.concatenate([np.asarray(o["w_tilde"]).reshape(-1),np.asarray(o["d_tilde"]).reshape(-1),np.asarray(o["wS_yA"]).reshape(-1),np.asarray(o["wU_yA"]).reshape(-1),np.asarray(o["sparse_yA"]).reshape(-1),np.asarray(o["band_row_tail"]).reshape(-1)]).tolist())(_oracle_compute_structured_row_update(bands,U,V,W,S,precomp,state,reflector))'},
        {
            "setup": """import numpy as np
n=4
bands=np.zeros((6,n)); bands[2]=5.; bands[1,1:]=.6; bands[3,:-1]=-.8
U=np.ones((n,2))/np.sqrt(n); V=U.copy(); W=U.copy(); S=U.copy()
UU=np.empty((n,2,2)); g=np.zeros((2,2))
for k in range(n-1,-1,-1): g=g+np.outer(U[k],U[k]); UU[k]=g
UW=np.empty((n,2,2)); p=np.zeros((2,2))
for k in range(n): p=p+np.outer(U[k],W[k]); UW[k]=p
precomp={"A_T_U":np.zeros((n,2)),"S_tilde":np.concatenate((S,np.zeros((n,2))),axis=1),
         "UU_lookup":UU,"UW_lookup":UW}
state={"J":np.zeros((2,2)),"K":np.zeros((2,2)),"E":np.zeros((2,5)),
       "X":np.zeros((2,2)),"Y":np.zeros((2,2)),"Z":np.zeros((2,5)),"k":0}
reflector={"kbar":np.zeros(2),"bhat":np.array([1.,0.,0.]),"tau":0.,"rkk":-5.}
bad={}
def run_model():
    try:
        compute_structured_row_update(bands,U,V,W,S,precomp,state,bad); return 0
    except ValueError: return 1
    except Exception: return 2
def run_oracle():
    try:
        _oracle_compute_structured_row_update(bands,U,V,W,S,precomp,state,bad); return 0
    except ValueError: return 1
    except Exception: return 2
""",
            "call":"run_model()","gold_call":"run_oracle()"
        },
        {
            "setup": """import numpy as np
n=4
bands=np.zeros((6,n)); bands[2]=5.; bands[1,1:]=.6; bands[3,:-1]=-.8
U=np.ones((n,2))/np.sqrt(n); V=U.copy(); W=U.copy(); S=U.copy()
UU=np.empty((n,2,2)); g=np.zeros((2,2))
for k in range(n-1,-1,-1): g=g+np.outer(U[k],U[k]); UU[k]=g
UW=np.empty((n,2,2)); p=np.zeros((2,2))
for k in range(n): p=p+np.outer(U[k],W[k]); UW[k]=p
precomp={"A_T_U":np.zeros((n,2)),"S_tilde":np.concatenate((S,np.zeros((n,2))),axis=1),
         "UU_lookup":UU,"UW_lookup":UW}
state={"J":np.zeros((2,2)),"K":np.zeros((2,2)),"E":np.zeros((2,5)),
       "X":np.zeros((2,2)),"Y":np.zeros((2,2)),"Z":np.zeros((2,5)),"k":0}
reflector={"kbar":np.zeros(2),"bhat":np.array([1.,0.,0.]),"tau":0.,"rkk":-5.}
state['k']=3
reflector['bhat']=np.array([1.])""",
            "call":'(lambda o: np.concatenate([np.asarray(o["w_tilde"]).reshape(-1),np.asarray(o["d_tilde"]).reshape(-1),np.asarray(o["wS_yA"]).reshape(-1),np.asarray(o["wU_yA"]).reshape(-1),np.asarray(o["sparse_yA"]).reshape(-1),np.asarray(o["band_row_tail"]).reshape(-1)]).tolist())(compute_structured_row_update(bands,U,V,W,S,precomp,state,reflector))',
            "gold_call":'(lambda o: np.concatenate([np.asarray(o["w_tilde"]).reshape(-1),np.asarray(o["d_tilde"]).reshape(-1),np.asarray(o["wS_yA"]).reshape(-1),np.asarray(o["wU_yA"]).reshape(-1),np.asarray(o["sparse_yA"]).reshape(-1),np.asarray(o["band_row_tail"]).reshape(-1)]).tolist())(_oracle_compute_structured_row_update(bands,U,V,W,S,precomp,state,reflector))'},
    ]
