"""
Advance the six compact perturbation matrices after one structured Householder step.

This step advances the compact representation of the transformed trailing matrix after one Householder operation. The existing perturbation is first restricted to the next principal submatrix. Then the effect of the current rank-one Householder correction is absorbed into the six structured state matrices J, K, E, X, Y, and Z. This preserves the same perturbation form for the next QR iteration, rather than allowing the trailing matrix to become an unrestricted dense matrix. Because the state dimensions depend only on the fixed ranks and bandwidths, each update requires only constant-size work with respect to n.

Returns
-------
A dictionary of the next state values for the J, K, E, X, Y, and Z matrices, as well as the incremented index, k.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def update_structured_perturbation(
    U: np.ndarray,
    W: np.ndarray,
    state: dict,
    reflector: dict,
    row_update: dict,
) -> dict:
    """
    Update the compact perturbation matrices for the next QR iteration.

    Parameters
    ----------
    U, W : np.ndarray
        Generator matrices with shape (n, 2).
    state : dict
        Current matrices J, K, E, X, Y, Z and zero-based index k.
    reflector : dict
        Householder data containing kbar, bhat, and tau.
    row_update : dict
        Contains wS_yA, wU_yA, sparse_yA, and band_row_tail.

    Returns
    -------
    next_state : dict
        Updated matrices J, K, E, X, Y, Z and incremented index k.

    Raises
    ------
    ValueError
        If the generator arrays, structured state, reflector data, or 
        row-update data are incomplete, dimensionally inconsistent, or 
        incompatible with the active QR step.
    """
    return next_state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_update_structured_perturbation(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid test case specifications."""
    return [
        {
            "setup": """import numpy as np
n=4
U=np.ones((n,2))/np.sqrt(n); W=U.copy()
state={"J":np.zeros((2,2)),"K":np.zeros((2,2)),"E":np.zeros((2,5)),
       "X":np.zeros((2,2)),"Y":np.zeros((2,2)),"Z":np.zeros((2,5)),"k":0}
reflector={"kbar":np.zeros(2),"bhat":np.array([1.,0.,0.]),"tau":0.}
row_update={"wS_yA":np.zeros(2),"wU_yA":np.zeros(2),
            "sparse_yA":np.zeros(5),"band_row_tail":np.zeros(5)}""",
            "call":'(lambda o: (np.concatenate([np.asarray(o["J"]).reshape(-1),np.asarray(o["K"]).reshape(-1),np.asarray(o["E"]).reshape(-1),np.asarray(o["X"]).reshape(-1),np.asarray(o["Y"]).reshape(-1),np.asarray(o["Z"]).reshape(-1)]).tolist()+[int(o["k"])]) )(update_structured_perturbation(U,W,state,reflector,row_update))',
            "gold_call":'(lambda o: (np.concatenate([np.asarray(o["J"]).reshape(-1),np.asarray(o["K"]).reshape(-1),np.asarray(o["E"]).reshape(-1),np.asarray(o["X"]).reshape(-1),np.asarray(o["Y"]).reshape(-1),np.asarray(o["Z"]).reshape(-1)]).tolist()+[int(o["k"])]) )(_oracle_update_structured_perturbation(U,W,state,reflector,row_update))'},
        {
            "setup": """import numpy as np
n=4
U=np.ones((n,2))/np.sqrt(n); W=U.copy()
state={"J":np.zeros((2,2)),"K":np.zeros((2,2)),"E":np.zeros((2,5)),
       "X":np.zeros((2,2)),"Y":np.zeros((2,2)),"Z":np.zeros((2,5)),"k":0}
reflector={"kbar":np.zeros(2),"bhat":np.array([1.,0.,0.]),"tau":0.}
row_update={"wS_yA":np.zeros(2),"wU_yA":np.zeros(2),
            "sparse_yA":np.zeros(5),"band_row_tail":np.zeros(5)}
state['k']=3
reflector['bhat']=np.array([1.])""",
            "call":'(lambda o: (np.concatenate([np.asarray(o["J"]).reshape(-1),np.asarray(o["K"]).reshape(-1),np.asarray(o["E"]).reshape(-1),np.asarray(o["X"]).reshape(-1),np.asarray(o["Y"]).reshape(-1),np.asarray(o["Z"]).reshape(-1)]).tolist()+[int(o["k"])]) )(update_structured_perturbation(U,W,state,reflector,row_update))',
            "gold_call":'(lambda o: (np.concatenate([np.asarray(o["J"]).reshape(-1),np.asarray(o["K"]).reshape(-1),np.asarray(o["E"]).reshape(-1),np.asarray(o["X"]).reshape(-1),np.asarray(o["Y"]).reshape(-1),np.asarray(o["Z"]).reshape(-1)]).tolist()+[int(o["k"])]) )(_oracle_update_structured_perturbation(U,W,state,reflector,row_update))'},
        {
            "setup": """import numpy as np
n=4
U=np.ones((n,2))/np.sqrt(n); W=U.copy()
state={"J":np.zeros((2,2)),"K":np.zeros((2,2)),"E":np.zeros((2,5)),
       "X":np.zeros((2,2)),"Y":np.zeros((2,2)),"Z":np.zeros((2,5)),"k":0}
reflector={"kbar":np.zeros(2),"bhat":np.array([1.,0.,0.]),"tau":0.}
row_update={"wS_yA":np.zeros(2),"wU_yA":np.zeros(2),
            "sparse_yA":np.zeros(5),"band_row_tail":np.zeros(5)}
bad={}
def run_model():
    try:
        update_structured_perturbation(U,W,state,reflector,bad); return 0
    except ValueError: return 1
    except Exception: return 2
def run_oracle():
    try:
        _oracle_update_structured_perturbation(U,W,state,reflector,bad); return 0
    except ValueError: return 1
    except Exception: return 2
""",
            "call":"run_model()","gold_call":"run_oracle()"
        },
    ]
