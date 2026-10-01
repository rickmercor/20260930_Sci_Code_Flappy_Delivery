"""
Compute the CSS density matrix at inverse temperature beta and chemical potential mu: transform the Hamiltonian with the CSS overlap inverse square root, color with the density cutoff, build and transform the density probing states, form the block-Krylov basis of the transformed Hamiltonian, apply the Fermi-Dirac function by exact diagonalization inside the Krylov space, assemble the projected density block in the source's arrangement, and extract.

The Fermi operator is treated exactly within the small Krylov space; where the overlap inverse square root enters the assembly, and on which side of the projected block it acts, follows the source.

Returns
-------
return (n, n) float64: CSS density matrix
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def css_density(H, S, D, dcut_rho, dcut_s, v, v_s, beta, mu):
    """H, S: (n, n) symmetric Hamiltonian and SPD overlap; D: (n, n)
    distances; dcut_rho, dcut_s: density and overlap cutoffs; v, v_s: Krylov
    dimensions; beta, mu: Fermi-Dirac inverse temperature and chemical
    potential. Returns (n, n): the CSS density matrix, assembled exactly as
    the source specifies and extracted on the density cutoff. Raises
    ValueError on invalid Krylov dimensions."""
    return np.zeros(H.shape)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 7: CSS density matrix via the transformed Hamiltonian."""

import numpy as np


def _oracle_css_density(H, S, D, dcut_rho, dcut_s, v, v_s, beta, mu):
    H = np.asarray(H, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    Shat = _oracle_css_inv_sqrt(S, D, dcut_s, v_s)
    Hp = Shat @ H @ Shat
    Hp = (Hp + Hp.T) / 2.0
    colors_r = _oracle_greedy_multicoloring(D, dcut_rho)
    R = _oracle_css_matrix(colors_r)
    Rt = Shat @ R
    Q = _oracle_block_krylov_basis(Hp, Rt, v)
    HK = Q.T @ Hp @ Q
    HK = (HK + HK.T) / 2.0
    eps, U = np.linalg.eigh(HK)
    f = 1.0 / (1.0 + np.exp(beta * (eps - mu)))
    rhoK = U @ np.diag(f) @ U.T
    rho_t = Shat @ (Q @ (rhoK @ (Q.T @ Rt)))
    return _oracle_css_extract(rho_t, R, colors_r, D, dcut_rho)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+3)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+6)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*3)%5)/25.0);S=(S+S.T)/2', "call": "css_density(H, S, D, 3.0, 5.0, 5, 3, 6.0, 0.15)", "gold_call": "_oracle_css_density(H, S, D, 3.0, 5.0, 5, 3, 6.0, 0.15)", "tol": 1e-08},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+5)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+10)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*5)%5)/25.0);S=(S+S.T)/2', "call": "css_density(H, S, D, 3.0, 5.0, 5, 3, 6.0, -0.1)", "gold_call": "_oracle_css_density(H, S, D, 3.0, 5.0, 5, 3, 6.0, -0.1)", "tol": 1e-08},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+4)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+8)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*4)%5)/25.0);S=(S+S.T)/2', "call": "css_density(H, S, D, 3.0, 4.0, 4, 3, 6.0, 0.0)", "gold_call": "_oracle_css_density(H, S, D, 3.0, 4.0, 4, 3, 6.0, 0.0)", "tol": 1e-08},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+3)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+6)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*3)%5)/25.0);S=(S+S.T)/2', "call": "css_density(H, S, D, 3.0, 5.0, 1, 1, 6.0, 0.15)", "gold_call": "_oracle_css_density(H, S, D, 3.0, 5.0, 1, 1, 6.0, 0.15)", "tol": 1e-08},
        {"setup": 'import numpy as _n\n_i,_j=_n.meshgrid(_n.arange(20),_n.arange(20),indexing="ij")\nD=_n.abs(_i-_j).astype(float)\nH=_n.where(_n.abs(_i-_j)<=3,(((_i+1)*(_j+1)+3)%13)/13.0-0.5,0.0);H=(H+H.T)/2\nS=_n.where((_n.abs(_i-_j)<=2)&(_i!=_j),(((_i+2)*(_j+2)+6)%7)/35.0-0.1,0.0)\nS=S+_n.diag(1.4+(((_n.arange(20)+1)*3)%5)/25.0);S=(S+S.T)/2', "call": "css_density(H, S, D, 25.0, 25.0, 2, 2, 6.0, 0.15)", "gold_call": "_oracle_css_density(H, S, D, 25.0, 25.0, 2, 2, 6.0, 0.15)", "tol": 1e-08},
    ]
