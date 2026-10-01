"""
Assemble the internal force density at every material point for the dual-horizon formulation with spatially varying horizons (the paper's Eq. 17, with the pairwise bond force of Eq. 24 and the dual set of Eq. 11). Because family membership is asymmetric, a point receives contributions both from the bonds it owns and from the bonds owned by other points that reach it; the source's expression accounts for both, and the horizon that enters the micro-modulus of a bond is fixed by the source's convention in Eq. 24. Use E = 1.0 and A = 1.0. Recover the exact assembly and the horizon convention from the paper; the symmetric single-horizon expression is a different formula.

With a spatially varying horizon the naive single-horizon assembly violates the balance of linear momentum and produces ghost forces; the source's variational treatment restores it exactly.

Returns
-------
return (N,) float64: internal force density at each point
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def internal_force(X, u, V, delta, H, mu):
    """X: (N,) reference coordinates; u: (N,) displacements; V: (N,) point
    volumes; delta: (N,) horizons; H: (N, N) family indicator; mu: (N, N) bond
    damage indicator. Returns (N,) float64 with the internal force density at
    each point for the source's dual-horizon formulation (paper Eq. 17 with
    Eq. 11 and Eq. 24), using E = 1.0 and A = 1.0."""
    return np.zeros(np.asarray(X).size)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 6: dual-horizon internal force density (Eq. 17, 11, 24)."""

import numpy as np

_E = 1.0
_A = 1.0


def _stretch(X, u):
    X = np.asarray(X, dtype=np.float64)
    x = X + np.asarray(u, dtype=np.float64)
    R = np.abs(X[None, :] - X[:, None])
    r = np.abs(x[None, :] - x[:, None])
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(R > 0, (r - R) / np.where(R > 0, R, 1.0), 0.0)
    return s


def _oracle_internal_force(X, u, V, delta, H, mu):
    X = np.asarray(X, dtype=np.float64)
    x = X + np.asarray(u, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    c = _oracle_micro_modulus(delta, _E, _A)
    Hb = (np.asarray(H, dtype=np.float64) > 0).astype(np.float64)
    Hd = Hb.T
    mu = np.asarray(mu, dtype=np.float64)
    s = _stretch(X, u)
    xhat = np.sign(x[None, :] - x[:, None])
    A1 = (Hb * mu * s * xhat) * (c[None, :] * V[None, :])
    A2 = (Hd * mu.T * s * xhat) * (V[None, :] * c[:, None])
    return A1.sum(axis=1) + A2.sum(axis=1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\nnC,nF,m,a,dC=4,4,2.515,3,0.1\ndF=dC/2.0\n_xs=[0.0]\nfor _ in range(nC-1): _xs.append(_xs[-1]+dC)\nfor _ in range(nF): _xs.append(_xs[-1]+dF)\nX=_n.array(_xs,dtype=float); N=X.size\nsp=_n.empty(N); sp[:nC]=dC; sp[nC:]=dF; sp[nC-1]=0.5*(dC+dF)\nV=sp*1.0; delta=m*sp\n_p=_n.array([(((i+1)*(i+2)+a*(i+3))%17)-8 for i in range(N)],dtype=float)\nu=0.010*X+0.0015*_p\nH=families(X, delta)\nmu=damage_state(bond_stretch(X, u), H, 0.16125)', "call": 'internal_force(X, u, V, delta, H, mu)', "gold_call": '_oracle_internal_force(X, u, V, delta, H, mu)', "tol": 1e-09},
        {"setup": 'import numpy as _n\nnC,nF,m,a,dC=3,5,2.515,5,0.08\ndF=dC/2.0\n_xs=[0.0]\nfor _ in range(nC-1): _xs.append(_xs[-1]+dC)\nfor _ in range(nF): _xs.append(_xs[-1]+dF)\nX=_n.array(_xs,dtype=float); N=X.size\nsp=_n.empty(N); sp[:nC]=dC; sp[nC:]=dF; sp[nC-1]=0.5*(dC+dF)\nV=sp*1.0; delta=m*sp\n_p=_n.array([(((i+1)*(i+2)+a*(i+3))%17)-8 for i in range(N)],dtype=float)\nu=0.010*X+0.0015*_p\nH=families(X, delta)\nmu=damage_state(bond_stretch(X, u), H, 0.1370625)', "call": 'internal_force(X, u, V, delta, H, mu)', "gold_call": '_oracle_internal_force(X, u, V, delta, H, mu)', "tol": 1e-09},
        {"setup": 'import numpy as _n\nnC,nF,m,a,dC=5,3,3.015,7,0.1\ndF=dC/2.0\n_xs=[0.0]\nfor _ in range(nC-1): _xs.append(_xs[-1]+dC)\nfor _ in range(nF): _xs.append(_xs[-1]+dF)\nX=_n.array(_xs,dtype=float); N=X.size\nsp=_n.empty(N); sp[:nC]=dC; sp[nC:]=dF; sp[nC-1]=0.5*(dC+dF)\nV=sp*1.0; delta=m*sp\n_p=_n.array([(((i+1)*(i+2)+a*(i+3))%17)-8 for i in range(N)],dtype=float)\nu=0.010*X+0.0015*_p\nH=families(X, delta)\nmu=damage_state(bond_stretch(X, u), H, 0.209625)', "call": 'internal_force(X, u, V, delta, H, mu)', "gold_call": '_oracle_internal_force(X, u, V, delta, H, mu)', "tol": 1e-09},
    ]
