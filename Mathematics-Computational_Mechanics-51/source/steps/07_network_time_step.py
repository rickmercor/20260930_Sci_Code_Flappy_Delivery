"""
Advance the network state by one step of the source's energy-conservative implicit time discretisation and return the hybrid nodal values on the free nodes at the new time level. The two per-edge right-hand sides carry the previous dual, primal and auxiliary fields together with the previous hybrid nodal state, each with the weight the source's scheme gives it; the previous hybrid state is supplied per edge as a 12-vector, start node first. How the source treats quantities that sit between two time levels, and the weights that follow from it, must be recovered from the paper, and they are what make the discrete energy exactly conserved rather than merely bounded. Assemble by calling the earlier sub-problem functions.

An implicit centred scheme avoids the severe step-size restriction that an explicit method would inherit from the shortest edge in the network, and it is the choice that makes the fully discrete energy exactly conserved rather than merely bounded.

Returns
-------
return (6*nf,) float64: the hybrid nodal values on the free nodes at the new time level
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def network_time_step(nodes, edges, dirichlet, coeffs, p, tau, dt, Q, Y, Z, lam_prev, lamD):
    """Q, Y, Z: lists of (6*(p+1),) per-edge dual, primal and auxiliary fields at
    the previous time level; lam_prev: list of per-edge (12,) hybrid values at the
    previous level; other arguments as before. Returns (6*nf,) float64."""
    return np.zeros(6)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Gold oracle: network_time_step."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

_CACHE = {}


def _key(*a):
    return tuple(x.tobytes() if isinstance(x, np.ndarray) else x for x in a)




# Chain the reference oracles of the earlier sub-problems (project rule, 2026-08-28).
# When only the public name is bound, alias it so the reference path still resolves.
for _n in ('edge_frames', 'timoshenko_operators', 'hdg_local_matrices', 'edge_local_solver', 'condensed_node_system'):
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


def _oracle_network_time_step(nodes, edges, dirichlet, coeffs, p, tau, dt, Q, Y, Z, lam_prev, lamD):
    """One step of the source's energy-conservative implicit scheme (paper Eqs 5.1-5.2).

    lam_prev: list of per-edge 12-vectors holding the previous hybrid nodal state.
    Returns the (6*nf,) hybrid nodal values on the free nodes at the new time level.
    """
    if p < 0 or tau <= 0 or dt <= 0:
        raise ValueError("need p >= 0, tau > 0 and dt > 0")
    if not (len(Q) == len(Y) == len(Z) == len(lam_prev) == len(edges)):
        raise ValueError("one previous state per edge is required")
    FR = _oracle_edge_frames(nodes, edges)
    rhs_y, rhs_z = [], []
    for k, (a, b) in enumerate(edges):
        L = FR[k, 3]
        Ms = _oracle_hdg_local_matrices(p, L, FR[k, :3], *coeffs[k])
        B, Mass, D, T = Ms[1], Ms[2], Ms[3], Ms[4]
        T0, TL = _traces(p, L)
        lp = np.asarray(lam_prev[k], dtype=np.float64)
        Gl = tau * (T0.T @ lp[:6] + TL.T @ lp[6:])
        rhs_y.append(-B @ Q[k] - tau * T @ Y[k] + Gl + (2.0 / dt) * Mass @ Z[k])
        rhs_z.append(-(2.0 / dt) * Mass @ Y[k] - D @ Z[k])
    out = _oracle_condensed_node_system(nodes, edges, dirichlet, coeffs, p, tau, dt,
                                rhs_y, rhs_z, lamD)
    return np.linalg.solve(out[:, :-1], out[:, -1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nnodes = [[0.0,0.0,0.0],[1.0,0.2,-0.3],[0.4,1.1,0.5],[-0.6,0.7,1.2],[0.9,-0.8,0.6]]\nedges = [(0,1),(1,2),(2,3),(0,4),(4,2)]\nI3 = np.eye(3)\ndef _cf(k):\n    w = np.array([((k+1)%3)+1.0, ((k+2)%4)+1.0, ((k+3)%5)+1.0]); w = w/np.linalg.norm(w)\n    O = np.outer(w,w)\n    return [I3+0.6*O, I3+0.9*O, I3+0.4*O, I3+0.7*O]\ncoeffs = [_cf(k) for k in range(len(edges))]\np=2;tau=1.0;dt=0.02;n=6*(p+1)\ndirichlet={0,3}\nrng=np.random.default_rng(6)\nQ=[rng.standard_normal(n)*0.1 for _ in edges];Y=[rng.standard_normal(n)*0.1 for _ in edges];Z=[rng.standard_normal(n)*0.1 for _ in edges]\nlam_prev=[rng.standard_normal(12)*0.05 for _ in edges]\nlamD={0:np.zeros(6),3:np.zeros(6)}', "call": 'network_time_step(nodes, edges, dirichlet, coeffs, p, tau, dt, Q, Y, Z, lam_prev, lamD)', "gold_call": '_oracle_network_time_step(nodes, edges, dirichlet, coeffs, p, tau, dt, Q, Y, Z, lam_prev, lamD)', "tol": 1e-08},
        {"setup": 'import numpy as np\nnodes = [[0.0,0.0,0.0],[1.0,0.2,-0.3],[0.4,1.1,0.5],[-0.6,0.7,1.2],[0.9,-0.8,0.6]]\nedges = [(0,1),(1,2),(2,3),(0,4),(4,2)]\nI3 = np.eye(3)\ndef _cf(k):\n    w = np.array([((k+1)%3)+1.0, ((k+2)%4)+1.0, ((k+3)%5)+1.0]); w = w/np.linalg.norm(w)\n    O = np.outer(w,w)\n    return [I3+0.6*O, I3+0.9*O, I3+0.4*O, I3+0.7*O]\ncoeffs = [_cf(k) for k in range(len(edges))]\np=3;tau=1.0;dt=0.02;n=6*(p+1)\ndirichlet={0,3}\nrng=np.random.default_rng(8)\nQ=[rng.standard_normal(n)*0.3 for _ in edges];Y=[rng.standard_normal(n)*0.3 for _ in edges];Z=[rng.standard_normal(n)*0.3 for _ in edges]\nlam_prev=[np.zeros(12) for _ in edges]\nlamD={0:np.zeros(6),3:np.zeros(6)}', "call": 'network_time_step(nodes, edges, dirichlet, coeffs, p, tau, dt, Q, Y, Z, lam_prev, lamD)', "gold_call": '_oracle_network_time_step(nodes, edges, dirichlet, coeffs, p, tau, dt, Q, Y, Z, lam_prev, lamD)', "tol": 1e-08},
        {"setup": 'import numpy as np\nnodes = [[0.0,0.0,0.0],[1.0,0.2,-0.3],[0.4,1.1,0.5],[-0.6,0.7,1.2],[0.9,-0.8,0.6]]\nedges = [(0,1),(1,2),(2,3),(0,4),(4,2)]\nI3 = np.eye(3)\ndef _cf(k):\n    w = np.array([((k+1)%3)+1.0, ((k+2)%4)+1.0, ((k+3)%5)+1.0]); w = w/np.linalg.norm(w)\n    O = np.outer(w,w)\n    return [I3+0.6*O, I3+0.9*O, I3+0.4*O, I3+0.7*O]\ncoeffs = [_cf(k) for k in range(len(edges))]\np=1;tau=0.5;dt=0.1;n=6*(p+1)\ndirichlet={1,4}\nQ=[np.ones(n)*0.02 for _ in edges];Y=[np.ones(n)*0.05 for _ in edges];Z=[np.zeros(n) for _ in edges]\nlam_prev=[np.ones(12)*0.01 for _ in edges]\nlamD={1:np.zeros(6),4:np.zeros(6)}', "call": 'network_time_step(nodes, edges, dirichlet, coeffs, p, tau, dt, Q, Y, Z, lam_prev, lamD)', "gold_call": '_oracle_network_time_step(nodes, edges, dirichlet, coeffs, p, tau, dt, Q, Y, Z, lam_prev, lamD)', "tol": 1e-08},
    ]
