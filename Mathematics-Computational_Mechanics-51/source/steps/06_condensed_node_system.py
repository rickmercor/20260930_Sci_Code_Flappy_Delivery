"""
Eliminate every edge interior and assemble the source's condensed system on the free nodes, returning the operator with the load vector appended as a final column. Free nodes are those not in the Dirichlet set, ordered by increasing node index, with the six components of a node occupying consecutive rows. The condensed operator is obtained by substituting the edge-local reconstructions into the source's nodal condition, which is imposed on the numerical flux rather than on the dual variable directly, summed over the edges incident to each free node with each edge's own orientation sign. Contributions of Dirichlet nodes move to the load vector. The resulting operator is symmetric positive definite. Assemble by calling the earlier sub-problem functions.

After condensation the only remaining unknowns live on the network nodes, so the linear system solved at each time step has a size set by the graph rather than by the polynomial degree.

Returns
-------
return (6*nf, 6*nf + 1) float64: the condensed node operator with its load vector appended
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def condensed_node_system(nodes, edges, dirichlet, coeffs, p, tau, dt, rhs_y, rhs_z, lamD):
    """nodes: (n_nodes, 3); edges: list of (a, b); dirichlet: set of node indices;
    coeffs: per-edge [Cn, Cm, Cu, Cr]; p, tau, dt as before; rhs_y, rhs_z: lists of
    (6*(p+1),) per-edge right-hand sides; lamD: dict of prescribed nodal values.
    Returns (6*nf, 6*nf + 1) float64."""
    return np.zeros((6, 7))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Gold oracle: condensed_node_system."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

_CACHE = {}


def _key(*a):
    return tuple(x.tobytes() if isinstance(x, np.ndarray) else x for x in a)




# Chain the reference oracles of the earlier sub-problems (project rule, 2026-08-28).
# When only the public name is bound, alias it so the reference path still resolves.
for _n in ('edge_frames', 'timoshenko_operators', 'hdg_local_matrices', 'hdg_numerical_flux', 'edge_local_solver'):
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


def _oracle_condensed_node_system(nodes, edges, dirichlet, coeffs, p, tau, dt, rhs_y, rhs_z, lamD):
    """Static condensation onto the free nodes (paper Eqs 5.5-5.7).

    Returns (6*nf, 6*nf + 1): the condensed operator with the load vector appended.
    """
    if p < 0 or tau <= 0 or dt <= 0:
        raise ValueError("need p >= 0, tau > 0 and dt > 0")
    if len(coeffs) != len(edges):
        raise ValueError("one coefficient set per edge is required")
    X = np.asarray(nodes, dtype=np.float64)
    fr = [j for j in range(X.shape[0]) if j not in set(dirichlet)]
    pos = {n: k for k, n in enumerate(fr)}
    nf = len(fr)
    if nf == 0:
        raise ValueError("the network has no free nodes")
    Ahat = np.zeros((6 * nf, 6 * nf)); Fhat = np.zeros(6 * nf)
    FR = _oracle_edge_frames(nodes, edges)
    for k, (a, b) in enumerate(edges):
        a = int(a); b = int(b)
        i_hat, L = FR[k, :3], FR[k, 3]
        nu0, nuL = FR[k, 4 + a], FR[k, 4 + b]
        Cn, Cm, Cu, Cr = coeffs[k]
        cols = []
        for j in range(12):
            e = np.zeros(12); e[j] = 1.0
            cols.append(_oracle_edge_local_solver(p, L, i_hat, Cn, Cm, Cu, Cr, tau, dt,
                                          nu0, nuL, e, np.zeros(6 * (p + 1)),
                                          np.zeros(6 * (p + 1))).ravel())
        S_lam = np.stack(cols, axis=1)
        s_par = _oracle_edge_local_solver(p, L, i_hat, Cn, Cm, Cu, Cr, tau, dt, nu0, nuL,
                                  np.zeros(12), rhs_y[k], rhs_z[k]).ravel()
        n = 6 * (p + 1)
        lam_fixed = np.zeros(12)
        for j, nd in enumerate((a, b)):
            if nd in set(dirichlet):
                lam_fixed[6 * j:6 * j + 6] = lamD[nd]
        # the nodal condition is imposed on the numerical flux (paper Eq 4.2), so
        # build the rows by evaluating that flux on each hybrid basis response
        FL = np.zeros((2, 6, 12))
        for j in range(12):
            e = np.zeros(12); e[j] = 1.0
            FL[:, :, j] = _oracle_hdg_numerical_flux(S_lam[:n, j], S_lam[n:2 * n, j], e, p, L, tau)
        FLp = _oracle_hdg_numerical_flux(s_par[:n], s_par[n:2 * n], np.zeros(12), p, L, tau)
        for nd, fi, sel in ((a, 0, slice(0, 6)), (b, 1, slice(6, 12))):
            if nd in set(dirichlet):
                continue
            row = -FL[fi]
            rhsv = -FLp[fi] + row @ lam_fixed
            r = pos[nd]
            for other, osel in ((a, slice(0, 6)), (b, slice(6, 12))):
                if other in set(dirichlet):
                    continue
                c = pos[other]
                Ahat[6 * r:6 * r + 6, 6 * c:6 * c + 6] += row[:, osel]
            Fhat[6 * r:6 * r + 6] -= rhsv
    return np.hstack([Ahat, Fhat.reshape(-1, 1)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nnodes = [[0.0,0.0,0.0],[1.0,0.2,-0.3],[0.4,1.1,0.5],[-0.6,0.7,1.2],[0.9,-0.8,0.6]]\nedges = [(0,1),(1,2),(2,3),(0,4),(4,2)]\nI3 = np.eye(3)\ndef _cf(k):\n    w = np.array([((k+1)%3)+1.0, ((k+2)%4)+1.0, ((k+3)%5)+1.0]); w = w/np.linalg.norm(w)\n    O = np.outer(w,w)\n    return [I3+0.6*O, I3+0.9*O, I3+0.4*O, I3+0.7*O]\ncoeffs = [_cf(k) for k in range(len(edges))]\np=2;tau=1.0;dt=0.02;n=6*(p+1)\ndirichlet={0,3}\nrhs_y=[np.zeros(n) for _ in edges];rhs_z=[np.zeros(n) for _ in edges]\nlamD={0:np.zeros(6),3:np.zeros(6)}', "call": 'condensed_node_system(nodes, edges, dirichlet, coeffs, p, tau, dt, rhs_y, rhs_z, lamD)', "gold_call": '_oracle_condensed_node_system(nodes, edges, dirichlet, coeffs, p, tau, dt, rhs_y, rhs_z, lamD)', "tol": 1e-08},
        {"setup": 'import numpy as np\nnodes = [[0.0,0.0,0.0],[1.0,0.2,-0.3],[0.4,1.1,0.5],[-0.6,0.7,1.2],[0.9,-0.8,0.6]]\nedges = [(0,1),(1,2),(2,3),(0,4),(4,2)]\nI3 = np.eye(3)\ndef _cf(k):\n    w = np.array([((k+1)%3)+1.0, ((k+2)%4)+1.0, ((k+3)%5)+1.0]); w = w/np.linalg.norm(w)\n    O = np.outer(w,w)\n    return [I3+0.6*O, I3+0.9*O, I3+0.4*O, I3+0.7*O]\ncoeffs = [_cf(k) for k in range(len(edges))]\np=2;tau=1.0;dt=0.02;n=6*(p+1)\ndirichlet={0,3}\nrng=np.random.default_rng(9)\nrhs_y=[rng.standard_normal(n) for _ in edges];rhs_z=[rng.standard_normal(n) for _ in edges]\nlamD={0:np.arange(6.0)*0.1,3:np.zeros(6)}', "call": 'condensed_node_system(nodes, edges, dirichlet, coeffs, p, tau, dt, rhs_y, rhs_z, lamD)', "gold_call": '_oracle_condensed_node_system(nodes, edges, dirichlet, coeffs, p, tau, dt, rhs_y, rhs_z, lamD)', "tol": 1e-08},
        {"setup": 'import numpy as np\nnodes = [[0.0,0.0,0.0],[1.0,0.2,-0.3],[0.4,1.1,0.5],[-0.6,0.7,1.2],[0.9,-0.8,0.6]]\nedges = [(0,1),(1,2),(2,3),(0,4),(4,2)]\nI3 = np.eye(3)\ndef _cf(k):\n    w = np.array([((k+1)%3)+1.0, ((k+2)%4)+1.0, ((k+3)%5)+1.0]); w = w/np.linalg.norm(w)\n    O = np.outer(w,w)\n    return [I3+0.6*O, I3+0.9*O, I3+0.4*O, I3+0.7*O]\ncoeffs = [_cf(k) for k in range(len(edges))]\np=1;tau=2.0;dt=0.005;n=6*(p+1)\ndirichlet={1,4}\nrhs_y=[np.ones(n)*0.2 for _ in edges];rhs_z=[np.zeros(n) for _ in edges]\nlamD={1:np.zeros(6),4:np.ones(6)*0.05}', "call": 'condensed_node_system(nodes, edges, dirichlet, coeffs, p, tau, dt, rhs_y, rhs_z, lamD)', "gold_call": '_oracle_condensed_node_system(nodes, edges, dirichlet, coeffs, p, tau, dt, rhs_y, rhs_z, lamD)', "tol": 1e-08},
    ]
