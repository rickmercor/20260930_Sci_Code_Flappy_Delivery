"""
Reconstruct the three interior fields of one edge from the hybrid nodal data, by solving the edge-local block system of the source's fully discrete scheme. Return them stacked as rows in the order dual, primal, auxiliary. The block system couples the dual bilinear form, the mixed form and its transpose, the stabilised trace term and the two mass couplings that carry the time discretisation; its exact block structure and the way the hybrid data enters the two right-hand sides are the source's. The 12-vector of hybrid values holds the start-node value in its first six entries and the end-node value in the last six, and nu0 and nuL are that edge's orientation signs at the start and end nodes. Raise ValueError unless dt and tau are positive. Assemble by calling the earlier sub-problem functions.

Because every bilinear form is edgewise, the interior unknowns of an element can be eliminated in terms of the boundary unknown alone. That elimination is what reduces the global problem to a system posed only on the network nodes.

Returns
-------
return (3, N) float64 with N = 6*(p+1): the dual, primal and auxiliary fields inside one edge
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def edge_local_solver(p, L, i_hat, Cn, Cm, Cu, Cr, tau, dt, nu0, nuL, lam12, rhs_y, rhs_z):
    """p, L: degree and edge length; i_hat: (3,) tangent; Cn, Cm, Cu, Cr: (3, 3)
    blocks; tau: stabilisation; dt: time step; nu0, nuL: orientation signs at the
    two endpoints; lam12: (12,) hybrid nodal values; rhs_y, rhs_z: (6*(p+1),)
    right-hand sides. Returns (3, 6*(p+1)) float64 stacked as dual, primal,
    auxiliary."""
    return np.zeros((3, 6 * (p + 1)))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Gold oracle: edge_local_solver."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

_CACHE = {}


def _key(*a):
    return tuple(x.tobytes() if isinstance(x, np.ndarray) else x for x in a)




# Chain the reference oracles of the earlier sub-problems (project rule, 2026-08-28).
# When only the public name is bound, alias it so the reference path still resolves.
for _n in ('timoshenko_operators', 'hdg_local_matrices'):
    if "_oracle_" + _n not in globals() and _n in globals():
        globals()["_oracle_" + _n] = globals()[_n]

def _basis(p, L):
    ck = ("basis", p, float(L))
    if ck in _CACHE:
        return _CACHE[ck]
    nq = 2 * p + 6
    xg, wg = np.polynomial.legendre.leggauss(nq)
    x = 0.5 * L * (xg + 1.0); w = 0.5 * L * wg
    s = 2.0 * x / L - 1.0
    I = np.eye(p + 1)
    V = np.array([np.polynomial.legendre.legval(s, I[j]) for j in range(p + 1)])
    dV = np.array([np.polynomial.legendre.legval(
        s, np.polynomial.legendre.legder(I[j])) * (2.0 / L) for j in range(p + 1)])
    M = (V * w) @ V.T
    Dx = (V * w) @ dV.T
    e0 = np.array([np.polynomial.legendre.legval(-1.0, I[j]) for j in range(p + 1)])
    eL = np.array([np.polynomial.legendre.legval(1.0, I[j]) for j in range(p + 1)])
    _CACHE[ck] = (M, Dx, e0, eL, x, w, V)
    return _CACHE[ck]


def _traces(p, L):
    _M, _Dx, e0, eL, _x, _w, _V = _basis(p, L)
    I6 = np.eye(6)
    return np.kron(I6, e0.reshape(1, -1)), np.kron(I6, eL.reshape(1, -1))


def _oracle_edge_local_solver(p, L, i_hat, Cn, Cm, Cu, Cr, tau, dt, nu0, nuL, lam12, rhs_y, rhs_z):
    """Reconstruct (q, y, z) inside one edge from the hybrid nodal data (paper Sec 5.3).

    Returns (3, N) stacked as [q, y, z] with N = 6*(p+1).
    """
    if dt <= 0 or tau <= 0:
        raise ValueError("need dt > 0 and tau > 0")
    Ms = _oracle_hdg_local_matrices(p, L, i_hat, Cn, Cm, Cu, Cr)
    A, B, Mass, D, T = Ms[0], Ms[1], Ms[2], Ms[3], Ms[4]
    T0, TL = _traces(p, L)
    n = A.shape[0]; Zr = np.zeros((n, n))
    ck = ("K", p, float(L), float(tau), float(dt)) + _key(
        np.asarray(i_hat, float), np.asarray(Cn, float), np.asarray(Cm, float),
        np.asarray(Cu, float), np.asarray(Cr, float))
    if ck in _CACHE:
        K = _CACHE[ck]
    else:
        K = np.block([[A, -B.T, Zr],
                      [B, tau * T, (2.0 / dt) * Mass],
                      [Zr, -(2.0 / dt) * Mass, D]])
        _CACHE[ck] = K
    lam = np.asarray(lam12, dtype=np.float64)
    rq = -(nu0 * T0.T @ lam[:6] + nuL * TL.T @ lam[6:])
    ry = np.asarray(rhs_y, dtype=np.float64) + tau * (T0.T @ lam[:6] + TL.T @ lam[6:])
    sol = np.linalg.solve(K, np.concatenate([rq, ry, np.asarray(rhs_z, dtype=np.float64)]))
    return sol.reshape(3, n)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\np=2;L=1.3;i_hat=np.array([0.,0.,1.]);Cn=np.eye(3);Cm=np.eye(3);Cu=np.eye(3);Cr=np.eye(3)\ntau=1.0;dt=0.02;nu0=-1.0;nuL=1.0\nrng=np.random.default_rng(2)\nlam12=rng.standard_normal(12);rhs_y=rng.standard_normal(6*(p+1));rhs_z=rng.standard_normal(6*(p+1))', "call": 'edge_local_solver(p, L, i_hat, Cn, Cm, Cu, Cr, tau, dt, nu0, nuL, lam12, rhs_y, rhs_z)', "gold_call": '_oracle_edge_local_solver(p, L, i_hat, Cn, Cm, Cu, Cr, tau, dt, nu0, nuL, lam12, rhs_y, rhs_z)', "tol": 1e-08},
        {"setup": 'import numpy as np\np=3;L=2.0548046676563256;i_hat=np.array([1.,1.,1.])/np.sqrt(3)\nw=np.array([2.,3.,4.]);w=w/np.linalg.norm(w);O=np.outer(w,w)\nCn=np.eye(3)+0.6*O;Cm=np.eye(3)+0.9*O;Cu=np.eye(3)+0.4*O;Cr=np.eye(3)+0.7*O\ntau=1.0;dt=0.02;nu0=-1.0;nuL=1.0\nrng=np.random.default_rng(5)\nlam12=rng.standard_normal(12)*0.4;rhs_y=rng.standard_normal(6*(p+1))*3.0;rhs_z=rng.standard_normal(6*(p+1))*3.0', "call": 'edge_local_solver(p, L, i_hat, Cn, Cm, Cu, Cr, tau, dt, nu0, nuL, lam12, rhs_y, rhs_z)', "gold_call": '_oracle_edge_local_solver(p, L, i_hat, Cn, Cm, Cu, Cr, tau, dt, nu0, nuL, lam12, rhs_y, rhs_z)', "tol": 1e-08},
        {"setup": 'import numpy as np\np=1;L=0.9;i_hat=np.array([0.6,-0.8,0.])\nCn=np.diag([1.,2.,3.]);Cm=np.diag([2.,1.,4.]);Cu=np.diag([1.,1.,2.]);Cr=np.diag([3.,1.,1.])\ntau=0.5;dt=0.05;nu0=1.0;nuL=-1.0\nlam12=np.arange(12.0)*0.1;rhs_y=np.ones(6*(p+1));rhs_z=np.zeros(6*(p+1))', "call": 'edge_local_solver(p, L, i_hat, Cn, Cm, Cu, Cr, tau, dt, nu0, nuL, lam12, rhs_y, rhs_z)', "gold_call": '_oracle_edge_local_solver(p, L, i_hat, Cn, Cm, Cu, Cr, tau, dt, nu0, nuL, lam12, rhs_y, rhs_z)', "tol": 1e-08},
        {"setup": 'import numpy as np\np=3;L=1.6;i_hat=np.array([0.,1.,0.]);Cn=np.eye(3)*2;Cm=np.eye(3)*0.5;Cu=np.eye(3)*1.5;Cr=np.eye(3)*0.8\ntau=3.0;dt=0.004;nu0=-1.0;nuL=1.0\nrng=np.random.default_rng(11)\nlam12=rng.standard_normal(12);rhs_y=rng.standard_normal(6*(p+1));rhs_z=rng.standard_normal(6*(p+1))', "call": 'edge_local_solver(p, L, i_hat, Cn, Cm, Cu, Cr, tau, dt, nu0, nuL, lam12, rhs_y, rhs_z)', "gold_call": '_oracle_edge_local_solver(p, L, i_hat, Cn, Cm, Cu, Cr, tau, dt, nu0, nuL, lam12, rhs_y, rhs_z)', "tol": 1e-08},
    ]
