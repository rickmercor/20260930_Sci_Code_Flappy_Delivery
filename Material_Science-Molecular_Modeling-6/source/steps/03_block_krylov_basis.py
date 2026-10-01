"""
Build the block-Krylov sequence B, AB, ..., A^(v-1)B, flatten it block by block in that order, orthonormalize the columns left to right by modified Gram-Schmidt with one full re-orthogonalization sweep, drop any column whose residual norm falls below 1e-10, and sign-fix each kept column so its first component with absolute value above 1e-8 is positive.

The source computes all Krylov-subspace quantities in an orthonormal space generated from the probing block; the block ordering, drop tolerance and sign fixing pin the reported basis uniquely.

Returns
-------
return (n, k) float64: orthonormal block-Krylov basis
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def block_krylov_basis(A, B, v):
    """A: (n, n) symmetric; B: (n, m) block of starting vectors; v: int >= 1.
    Returns (n, k): orthonormal basis of the block-Krylov space
    span{B, AB, ..., A^(v-1)B}, built by flattening the blocks in that order,
    orthonormalizing columns left to right by modified Gram-Schmidt with one
    full re-orthogonalization sweep, dropping columns with residual norm
    below 1e-10, and sign-fixing each kept column so that its first component
    with absolute value above 1e-8 is positive. Raises ValueError on an
    invalid v or if no column survives."""
    return np.zeros((B.shape[0], 1))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 3: orthonormal block-Krylov basis with the declared conventions."""

import numpy as np


def _oracle_block_krylov_basis(A, B, v):
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
        raise ValueError("invalid Krylov dimension")
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if B.ndim == 1:
        B = B[:, None]
    if A.shape[0] != A.shape[1] or A.shape[0] != B.shape[0]:
        raise ValueError("shape mismatch")
    cols = []
    blk = B.copy()
    for _ in range(int(v)):
        for c in range(blk.shape[1]):
            cols.append(blk[:, c].copy())
        blk = A @ blk
    Q = []
    for w in cols:
        for q in Q:
            w = w - (q @ w) * q
        for q in Q:
            w = w - (q @ w) * q
        nrm = float(np.linalg.norm(w))
        if nrm < 1e-10:
            continue
        w = w / nrm
        for comp in w:
            if abs(comp) > 1e-8:
                if comp < 0.0:
                    w = -w
                break
        Q.append(w)
    if not Q:
        raise ValueError("no Krylov column survived")
    return np.column_stack(Q)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+3)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+6)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*3)%5)/25.0);S=(S+S.T)/2\nO=css_matrix(greedy_multicoloring(D, 5.0))', "call": "block_krylov_basis(S, O, 3)", "gold_call": "_oracle_block_krylov_basis(S, O, 3)", "tol": 1e-09},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+5)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+10)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*5)%5)/25.0);S=(S+S.T)/2\nO=css_matrix(greedy_multicoloring(D, 5.0))', "call": "block_krylov_basis(S, O, 4)", "gold_call": "_oracle_block_krylov_basis(S, O, 4)", "tol": 1e-09},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+4)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+8)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*4)%5)/25.0);S=(S+S.T)/2\nR=css_matrix(greedy_multicoloring(D, 3.0))', "call": "block_krylov_basis(H, R, 5)", "gold_call": "_oracle_block_krylov_basis(H, R, 5)", "tol": 1e-09},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+3)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+6)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*3)%5)/25.0);S=(S+S.T)/2\nO=css_matrix(greedy_multicoloring(D, 5.0))', "call": "block_krylov_basis(S, O, 1)", "gold_call": "_oracle_block_krylov_basis(S, O, 1)", "tol": 1e-09},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+3)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+6)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*3)%5)/25.0);S=(S+S.T)/2\nb0=-_n.ones(20)\nB=_n.column_stack([b0,b0])', "call": "block_krylov_basis(S, B, 2)", "gold_call": "_oracle_block_krylov_basis(S, B, 2)", "tol": 1e-09},
    ]
