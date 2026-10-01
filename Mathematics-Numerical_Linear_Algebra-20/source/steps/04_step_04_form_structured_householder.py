"""
Form one normalized Householder vector from the current structured BPS trailing state.

This step constructs the normalized Householder reflector that eliminates the subdiagonal entries of the current active QR column without explicitly forming the entire column. The active column is represented as a low-rank contribution plus a short-banded residual, and its norm is evaluated using the precomputed suffix Gram information. A sign-stable Householder construction is used to avoid subtractive cancellation, and the reflector vector is normalized so that its first active component equals 1. The step then computes the Householder scaling coefficient tau together with the diagonal entry that will appear in the transformed factor row.

Returns
-------
A dictionary with the reflector (k_bar, b_hat, tau, rkk, qvec, c), and the active-column norm.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def form_structured_householder(
    bands: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    S: np.ndarray,
    precomp: dict,
    state: dict,
) -> dict:
    """
    Form the next normalized Householder reflector from the compact BPS state.

    Parameters
    ----------
    bands : np.ndarray
        Compact banded component with shape (6, n).
    U, V, W, S : np.ndarray
        Semiseparable generators, each with shape (n, 2).
    precomp : dict
        Precomputed A_T_U, S_tilde, UU_lookup, and UW_lookup.
    state : dict
        Current matrices J, K, E, X, Y, Z and zero-based index k.

    Returns
    -------
    reflector : dict
        Contains kbar, bhat, tau, rkk, qvec, c, and the active-column norm.

    Raises
    ------
    ValueError
        If the BPS inputs, precomputed quantities, or structured state are incomplete, 
        dimensionally inconsistent, numerically invalid, or do not define a valid 
        active Householder step.
    """
    return reflector

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _band_entry(bands: np.ndarray, i: int, j: int) -> np.float64:
    d = j - i
    if d < -2 or d > 3:
        return np.float64(0.0)
    return np.float64(bands[d + 2, i])

def _oracle_form_structured_householder(
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
for k in range(n-1,-1,-1):
    g=g+np.outer(U[k],U[k]); UU[k]=g
UW=np.empty((n,2,2)); p=np.zeros((2,2))
for k in range(n):
    p=p+np.outer(U[k],W[k]); UW[k]=p
precomp={"A_T_U":np.zeros((n,2)),"S_tilde":np.zeros((n,4)),"UU_lookup":UU,"UW_lookup":UW}
state={"J":np.zeros((2,2)),"K":np.zeros((2,2)),"E":np.zeros((2,5)),
       "X":np.zeros((2,2)),"Y":np.zeros((2,2)),"Z":np.zeros((2,5)),"k":0}""",
            "call":'(lambda o: np.concatenate([np.asarray(o["kbar"]).reshape(-1),np.asarray(o["bhat"]).reshape(-1),np.asarray([o["tau"],o["rkk"]]),np.asarray(o["qvec"]).reshape(-1),np.asarray(o["c"]).reshape(-1),np.asarray([o["rho"]])]).tolist())(form_structured_householder(bands,U,V,W,S,precomp,state))',
            "gold_call":'(lambda o: np.concatenate([np.asarray(o["kbar"]).reshape(-1),np.asarray(o["bhat"]).reshape(-1),np.asarray([o["tau"],o["rkk"]]),np.asarray(o["qvec"]).reshape(-1),np.asarray(o["c"]).reshape(-1),np.asarray([o["rho"]])]).tolist())(_oracle_form_structured_householder(bands,U,V,W,S,precomp,state))'},
        {
            "setup": """import numpy as np
n=4
bands=np.zeros((6,n)); bands[2]=5.; bands[1,1:]=.6; bands[3,:-1]=-.8
U=np.ones((n,2))/np.sqrt(n); V=U.copy(); W=U.copy(); S=U.copy()
UU=np.empty((n,2,2)); g=np.zeros((2,2))
for k in range(n-1,-1,-1):
    g=g+np.outer(U[k],U[k]); UU[k]=g
UW=np.empty((n,2,2)); p=np.zeros((2,2))
for k in range(n):
    p=p+np.outer(U[k],W[k]); UW[k]=p
precomp={"A_T_U":np.zeros((n,2)),"S_tilde":np.zeros((n,4)),"UU_lookup":UU,"UW_lookup":UW}
state={"J":np.zeros((2,2)),"K":np.zeros((2,2)),"E":np.zeros((2,5)),
       "X":np.zeros((2,2)),"Y":np.zeros((2,2)),"Z":np.zeros((2,5)),"k":0}
state['k']=3""",
            "call":'(lambda o: np.concatenate([np.asarray(o["kbar"]).reshape(-1),np.asarray(o["bhat"]).reshape(-1),np.asarray([o["tau"],o["rkk"]]),np.asarray(o["qvec"]).reshape(-1),np.asarray(o["c"]).reshape(-1),np.asarray([o["rho"]])]).tolist())(form_structured_householder(bands,U,V,W,S,precomp,state))',
            "gold_call":'(lambda o: np.concatenate([np.asarray(o["kbar"]).reshape(-1),np.asarray(o["bhat"]).reshape(-1),np.asarray([o["tau"],o["rkk"]]),np.asarray(o["qvec"]).reshape(-1),np.asarray(o["c"]).reshape(-1),np.asarray([o["rho"]])]).tolist())(_oracle_form_structured_householder(bands,U,V,W,S,precomp,state))'},
        {
            "setup": """import numpy as np
n=4
bands=np.zeros((6,n)); bands[2]=5.; bands[1,1:]=.6; bands[3,:-1]=-.8
U=np.ones((n,2))/np.sqrt(n); V=U.copy(); W=U.copy(); S=U.copy()
UU=np.empty((n,2,2)); g=np.zeros((2,2))
for k in range(n-1,-1,-1):
    g=g+np.outer(U[k],U[k]); UU[k]=g
UW=np.empty((n,2,2)); p=np.zeros((2,2))
for k in range(n):
    p=p+np.outer(U[k],W[k]); UW[k]=p
precomp={"A_T_U":np.zeros((n,2)),"S_tilde":np.zeros((n,4)),"UU_lookup":UU,"UW_lookup":UW}
state={"J":np.zeros((2,2)),"K":np.zeros((2,2)),"E":np.zeros((2,5)),
       "X":np.zeros((2,2)),"Y":np.zeros((2,2)),"Z":np.zeros((2,5)),"k":0}
bad={}
def run_model():
    try:
        form_structured_householder(bands,U,V,W,S,precomp,bad); return 0
    except ValueError: return 1
    except Exception: return 2
def run_oracle():
    try:
        _oracle_form_structured_householder(bands,U,V,W,S,precomp,bad); return 0
    except ValueError: return 1
    except Exception: return 2
""",
            "call":"run_model()","gold_call":"run_oracle()"
        },
    ]
