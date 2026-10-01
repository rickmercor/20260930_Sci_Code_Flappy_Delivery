"""
Project the overlap onto the orthonormal Krylov basis, symmetrize, and return the inverse square root of the projected overlap computed by exact eigendecomposition, guarding positive definiteness.

The source evaluates matrix functions of the overlap by exact diagonalization inside the small Krylov space rather than by a polynomial expansion.

Returns
-------
return (k, k) float64: inverse square root of the projected overlap
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def projected_inv_sqrt(S, Q):
    """S: (n, n) symmetric positive definite overlap; Q: (n, k) orthonormal
    basis. Returns (k, k): the inverse square root of the symmetrized
    projected overlap Q^T S Q, computed by exact eigendecomposition.
    Raises ValueError if the projected overlap has an eigenvalue below
    1e-10."""
    return np.eye(Q.shape[1])

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 4: inverse square root of the projected overlap."""

import numpy as np


def _oracle_projected_inv_sqrt(S, Q):
    S = np.asarray(S, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if S.shape[0] != S.shape[1] or Q.shape[0] != S.shape[0]:
        raise ValueError("shape mismatch")
    SK = Q.T @ S @ Q
    SK = (SK + SK.T) / 2.0
    lam, U = np.linalg.eigh(SK)
    if float(lam.min()) < 1e-10:
        raise ValueError("projected overlap not positive definite")
    return U @ np.diag(lam ** -0.5) @ U.T

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+3)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+6)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*3)%5)/25.0);S=(S+S.T)/2\nQ=block_krylov_basis(S, css_matrix(greedy_multicoloring(D, 5.0)), 3)', "call": "projected_inv_sqrt(S, Q)", "gold_call": "_oracle_projected_inv_sqrt(S, Q)", "tol": 1e-09},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+5)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+10)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*5)%5)/25.0);S=(S+S.T)/2\nQ=block_krylov_basis(S, css_matrix(greedy_multicoloring(D, 5.0)), 4)', "call": "projected_inv_sqrt(S, Q)", "gold_call": "_oracle_projected_inv_sqrt(S, Q)", "tol": 1e-09},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+4)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+8)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*4)%5)/25.0);S=(S+S.T)/2\nQ=block_krylov_basis(S, css_matrix(greedy_multicoloring(D, 5.0)), 2)', "call": "projected_inv_sqrt(S, Q)", "gold_call": "_oracle_projected_inv_sqrt(S, Q)", "tol": 1e-09},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+3)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+6)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*3)%5)/25.0);S=(S+S.T)/2\nQ=_n.eye(20)[:, :1]', "call": "projected_inv_sqrt(S, Q)", "gold_call": "_oracle_projected_inv_sqrt(S, Q)", "tol": 1e-09},
        {"setup": 'import numpy as _n\nS=4.0*_n.eye(12)\nQ=_n.eye(12)[:, :3]', "call": "projected_inv_sqrt(S, Q)", "gold_call": "_oracle_projected_inv_sqrt(S, Q)", "tol": 1e-09},
    ]
