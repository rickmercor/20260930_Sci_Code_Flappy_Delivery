"""
Run the whole scheme on the three declared network configurations and return one row each. A configuration with n_nodes nodes and integer parameter a places node j at [((3j+a) mod 7) - 3, ((5j+2a) mod 11) - 5, ((2j+3a) mod 13) - 6] divided by 3, joins j to j+1 for every j below n_nodes-1 and j to j+3 for every j below n_nodes-3, and holds nodes 0 and n_nodes-1 fixed with prescribed nodal values of zero. Edge k carries the coefficient blocks I + c outer(w, w) with w the normalisation of [((k+1) mod 3)+1, ((k+2) mod 4)+1, ((k+3) mod 5)+1] and c equal to 0.6, 0.9, 0.4 and 0.7 for the four blocks in the order used by the earlier sub-problems. The three configurations are (n_nodes, a) = (7, 3), (6, 5) and (8, 2), all at polynomial degree 3, stabilisation 1.0 and 24 steps of size 0.02*dt_scale. The initial primal field on edge k is the L2 projection onto the polynomial space of the function whose component c at arclength x is sin((c+1)x/2 + 0.3(k+1))/(c+2), and the initial velocity field is the projection of cos((c+2)x/3 + 0.2(k+1))/(c+3); the initial auxiliary field is obtained from that velocity as the source's first-order formulation prescribes, and the initial dual and hybrid states are NOT free: the source fixes them by its initial constitutive relation together with its nodal condition, and that system has to be solved before the first step. Each row has SEVEN entries: [initial discrete energy, final discrete energy, Euclidean norm of the final free-node hybrid vector, sum over free nodes of the norm of the displacement half of each nodal value, sum over free nodes of the norm of the full nodal value, and then the two constants of the source's spectral equivalence between the condensed node operator and a combination of the two graph forms of the previous sub-problem, ordered SMALLEST FIRST. Those two constants are the extremal generalised eigenvalues of the condensed operator with respect to that combination; the source scales one of the two graph forms by a power of the time step before combining them, and which form and which power is the source's convention, so recover it from the paper. Evaluate the condensed operator at the declared time step with vanishing right-hand sides. Raise ValueError unless dt_scale is positive and finite. Assemble by calling the earlier sub-problem functions, every one of them.

Auditing a scheme means checking the quantities it is supposed to preserve alongside the state it produces, so that a result which merely looks plausible can be separated from one that respects the structure of the underlying problem.

Returns
-------
return (3, 7) float64: one audit row per declared network configuration
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def beam_network_audit(dt_scale):
    """dt_scale: positive multiplier on the declared time step. Returns (3, 7)
    float64 with one row per configuration as described above."""
    return np.zeros((3, 7))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Gold oracle: beam_network_audit."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

P_DEG, TAU, NSTEP = 3, 1.0, 24
DT = 0.02
BASE = {1: (7, 3), 2: (6, 5), 3: (8, 2)}

_CACHE = {}


def _key(*a):
    return tuple(x.tobytes() if isinstance(x, np.ndarray) else x for x in a)




# Chain the reference oracles of the earlier sub-problems (project rule, 2026-08-28).
# When only the public name is bound, alias it so the reference path still resolves.
for _n in ('edge_frames', 'timoshenko_operators', 'hdg_local_matrices', 'hdg_numerical_flux', 'edge_local_solver', 'condensed_node_system', 'network_time_step', 'graph_forms'):
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


def _config(v):
    Nn, a = BASE[v]
    X = np.array([[((3 * j + a) % 7) - 3, ((5 * j + 2 * a) % 11) - 5,
                   ((2 * j + 3 * a) % 13) - 6] for j in range(Nn)], dtype=np.float64) / 3.0
    E = [(j, j + 1) for j in range(Nn - 1)] + [(j, j + 3) for j in range(Nn - 3)]
    co = []
    for k in range(len(E)):
        w = np.array([((k + 1) % 3) + 1.0, ((k + 2) % 4) + 1.0, ((k + 3) % 5) + 1.0])
        w = w / np.linalg.norm(w); O = np.outer(w, w); I3 = np.eye(3)
        co.append([I3 + 0.6 * O, I3 + 0.9 * O, I3 + 0.4 * O, I3 + 0.7 * O])
    return X, E, {0, Nn - 1}, co


def _project(p, L, k, which):
    """L2 projection onto V^p of the declared initial fields."""
    M, _Dx, _e0, _eL, x, w, V = _basis(p, L)
    out = np.zeros(6 * (p + 1))
    for c in range(6):
        if which == 0:
            g = np.sin((c + 1) * x / 2.0 + 0.3 * (k + 1)) / (c + 2.0)
        else:
            g = np.cos((c + 2) * x / 3.0 + 0.2 * (k + 1)) / (c + 3.0)
        out[c * (p + 1):(c + 1) * (p + 1)] = np.linalg.solve(M, V @ (w * g))
    return out


def _gather(edges, dirichlet, fr, pos, lam_free, lamD):
    out = []
    for (a, b) in edges:
        v = np.zeros(12)
        for j, nd in enumerate((int(a), int(b))):
            v[6 * j:6 * j + 6] = lamD[nd] if nd in set(dirichlet) else lam_free[6 * pos[nd]:6 * pos[nd] + 6]
        out.append(v)
    return out


def _initial(nodes, edges, dirichlet, coeffs, p, tau, Y0, lamD):
    """Paper Sec 5.1: q^0 and lambda^0 from the constitutive relation and balance."""
    X = np.asarray(nodes, dtype=np.float64)
    fr = [j for j in range(X.shape[0]) if j not in set(dirichlet)]
    pos = {n: i for i, n in enumerate(fr)}; nf = len(fr); n = 6 * (p + 1)
    FR = _oracle_edge_frames(nodes, edges)
    Ah = np.zeros((6 * nf, 6 * nf)); Fh = np.zeros(6 * nf); per = []
    for k, (a, b) in enumerate(edges):
        a = int(a); b = int(b); L = FR[k, 3]
        nu0, nuL = FR[k, 4 + a], FR[k, 4 + b]
        Ms = _oracle_hdg_local_matrices(p, L, FR[k, :3], *coeffs[k])
        Ainv = np.linalg.inv(Ms[0]); T0, TL = _traces(p, L)
        Rq = np.zeros((n, 12)); Rq[:, :6] = -nu0 * T0.T; Rq[:, 6:] = -nuL * TL.T
        Sq = Ainv @ Rq; qpar = Ainv @ (Ms[1].T @ Y0[k]); per.append((Sq, qpar))
        # the nodal condition is the source's numerical flux (Eq 4.2), evaluated on
        # each hybrid basis response and on the particular part
        FLs = np.zeros((2, 6, 12))
        for j in range(12):
            e = np.zeros(12); e[j] = 1.0
            FLs[:, :, j] = _oracle_hdg_numerical_flux(Sq[:, j], np.zeros(n), e, p, L, tau)
        FLp = _oracle_hdg_numerical_flux(qpar, Y0[k], np.zeros(12), p, L, tau)
        lam_fixed = np.zeros(12)
        for j, nd in enumerate((a, b)):
            if nd in set(dirichlet):
                lam_fixed[6 * j:6 * j + 6] = lamD[nd]
        for nd, fi, sel in ((a, 0, slice(0, 6)), (b, 1, slice(6, 12))):
            if nd in set(dirichlet):
                continue
            row = -FLs[fi]
            rhsv = -FLp[fi] + row @ lam_fixed
            r = pos[nd]
            for other, osel in ((a, slice(0, 6)), (b, slice(6, 12))):
                if other in set(dirichlet):
                    continue
                Ah[6 * r:6 * r + 6, 6 * pos[other]:6 * pos[other] + 6] += row[:, osel]
            Fh[6 * r:6 * r + 6] -= rhsv
    lam_free = np.linalg.solve(Ah, Fh)
    lam_e = _gather(edges, dirichlet, fr, pos, lam_free, lamD)
    Q0 = [per[k][0] @ lam_e[k] + per[k][1] for k in range(len(edges))]
    return Q0, lam_free, lam_e


def _energy(edges, coeffs, p, tau, FR, Q, Y, Z, lam_e):
    E = 0.0
    for k in range(len(edges)):
        L = FR[k, 3]
        Ms = _oracle_hdg_local_matrices(p, L, FR[k, :3], *coeffs[k])
        T0, TL = _traces(p, L)
        E += 0.5 * Q[k] @ Ms[0] @ Q[k] + 0.5 * Z[k] @ Ms[3] @ Z[k]
        for T, sel in ((T0, slice(0, 6)), (TL, slice(6, 12))):
            j = T @ Y[k] - lam_e[k][sel]
            E += 0.5 * tau * (j @ j)
    return float(E)


def _spectral_constants(Gforms, dt, Ahat):
    """Extremal generalised eigenvalues of Ahat against MG/dt^2 + LG (Eq 6.2)."""
    G = np.asarray(Gforms, dtype=np.float64)
    S = G[0] / (dt * dt) + G[1]
    Ssym = 0.5 * (S + S.T); Asym = 0.5 * (Ahat + Ahat.T)
    w, V = np.linalg.eigh(Ssym)
    if np.min(w) <= 0:
        raise ValueError("graph form is not positive definite")
    Wi = (V * (1.0 / np.sqrt(w))) @ V.T
    ev = np.linalg.eigvalsh(Wi @ Asym @ Wi)
    return float(ev.min()), float(ev.max())


def _step_rhs(E, co, p, tau, dt, FR, Q, Y, Z, lam_prev):
    """The per-edge right-hand sides of the source's time step; the same expressions
    the time-stepping sub-problem builds, kept here so the reconstruction after the
    solve uses exactly what went into it."""
    ry, rz = [], []
    for k in range(len(E)):
        L = FR[k, 3]
        Ms = _oracle_hdg_local_matrices(p, L, FR[k, :3], *co[k]); T0, TL = _traces(p, L)
        lp = np.asarray(lam_prev[k], dtype=np.float64)
        Gl = tau * (T0.T @ lp[:6] + TL.T @ lp[6:])
        ry.append(-Ms[1] @ Q[k] - tau * Ms[4] @ Y[k] + Gl + (2.0 / dt) * Ms[2] @ Z[k])
        rz.append(-(2.0 / dt) * Ms[2] @ Y[k] - Ms[3] @ Z[k])
    return ry, rz


def _oracle_beam_network_audit(dt_scale):
    if isinstance(dt_scale, bool) or not np.isfinite(dt_scale) or dt_scale <= 0:
        raise ValueError("dt_scale must be positive and finite")
    rows = []
    for v in (1, 2, 3):
        X, E, ND, co = _config(v)
        dt = DT * float(dt_scale)
        p, tau = P_DEG, TAU
        FR = _oracle_edge_frames(X, E)
        fr = [j for j in range(X.shape[0]) if j not in ND]
        pos = {n: i for i, n in enumerate(fr)}
        lamD = {nd: np.zeros(6) for nd in ND}
        Y = [_project(p, FR[k, 3], k, 0) for k in range(len(E))]
        Y1 = [_project(p, FR[k, 3], k, 1) for k in range(len(E))]
        Z = [np.kron(_oracle_timoshenko_operators(FR[k, :3], *co[k])[1], np.eye(p + 1)) @ Y1[k]
             for k in range(len(E))]
        Q, lam_free, lam_e = _initial(X, E, ND, co, p, tau, Y, lamD)
        E0 = _energy(E, co, p, tau, FR, Q, Y, Z, lam_e)
        for _ in range(NSTEP):
            ry, rz = _step_rhs(E, co, p, tau, dt, FR, Q, Y, Z, lam_e)
            lam_free = _oracle_network_time_step(X, E, ND, co, p, tau, dt, Q, Y, Z, lam_e, lamD)
            lam_e = _gather(E, ND, fr, pos, lam_free, lamD)
            newQ, newY, newZ = [], [], []
            for k in range(len(E)):
                sol = _oracle_edge_local_solver(p, FR[k, 3], FR[k, :3], *co[k], tau, dt,
                                        FR[k, 4 + int(E[k][0])], FR[k, 4 + int(E[k][1])],
                                        lam_e[k], ry[k], rz[k])
                newQ.append(sol[0]); newY.append(sol[1]); newZ.append(sol[2])
            Q, Y, Z = newQ, newY, newZ
        EK = _energy(E, co, p, tau, FR, Q, Y, Z, lam_e)
        u = sum(float(np.linalg.norm(lam_free[6 * i:6 * i + 3])) for i in range(len(fr)))
        s = sum(float(np.linalg.norm(lam_free[6 * i:6 * i + 6])) for i in range(len(fr)))
        out = _oracle_condensed_node_system(X, E, ND, co, p, tau, dt,
                                    [np.zeros(6 * (p + 1))] * len(E),
                                    [np.zeros(6 * (p + 1))] * len(E), lamD)
        Gforms = _oracle_graph_forms(X, E, ND)
        al, be = _spectral_constants(Gforms, dt, out[:, :-1])
        # the source's nodal balance: the fluxes meeting at a free node must cancel
        bal = {nd: np.zeros(6) for nd in fr}
        scale = 0.0
        for k in range(len(E)):
            fl = _oracle_hdg_numerical_flux(Q[k], Y[k], lam_e[k], p, FR[k, 3], tau)
            scale = max(scale, float(np.max(np.abs(fl))),
                        float(np.max(np.abs(_oracle_hdg_local_matrices(p, FR[k, 3], FR[k, :3], *co[k])[0]))))
            for j, nd in enumerate((int(E[k][0]), int(E[k][1]))):
                if nd in fr:
                    bal[nd] = bal[nd] + fl[j]
        resid = max(float(np.max(np.abs(bal[nd]))) for nd in fr) if fr else 0.0
        if not np.isfinite(resid) or resid > 1e-6 * max(1.0, scale):
            raise ValueError("the source's nodal balance is not satisfied at the final state")
        rows.append([E0, EK, float(np.linalg.norm(lam_free)), u, s, al, be])
    return np.asarray(rows, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": '', "call": 'beam_network_audit(1.0)', "gold_call": '_oracle_beam_network_audit(1.0)', "tol": 1e-07},
        {"setup": '', "call": 'beam_network_audit(0.5)', "gold_call": '_oracle_beam_network_audit(0.5)', "tol": 1e-07},
        {"setup": '', "call": 'beam_network_audit(2.0)', "gold_call": '_oracle_beam_network_audit(2.0)', "tol": 1e-07},
        {"setup": '', "call": 'beam_network_audit(0.25)', "gold_call": '_oracle_beam_network_audit(0.25)', "tol": 1e-07},
        {"setup": '', "call": 'beam_network_audit(1.75)', "gold_call": '_oracle_beam_network_audit(1.75)', "tol": 1e-07},
    ]
