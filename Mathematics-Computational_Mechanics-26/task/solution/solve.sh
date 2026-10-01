#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def build_cooks_mesh(n):
    out = np.zeros(2 * (n + 1) ** 2)
    for j in range(n + 1):
        for i in range(n + 1):
            xi, eta = i / n, j / n
            nd = j * (n + 1) + i
            out[2 * nd] = 480.0 * xi
            out[2 * nd + 1] = 440.0 * xi + eta * (440.0 - 280.0 * xi)
    return out

import numpy as np


def q1_shape_gradients(xe, xi, eta):
    dN = 0.25 * np.array([[-(1 - eta), -(1 - xi)],
                          [(1 - eta), -(1 + xi)],
                          [(1 + eta), (1 + xi)],
                          [-(1 + eta), (1 - xi)]])
    J = np.asarray(xe, dtype=float).T @ dN
    detJ = float(np.linalg.det(J))
    dNdX = dN @ np.linalg.inv(J)
    return np.concatenate(([detJ], dNdX.reshape(-1)))

import numpy as np


def constitutive_response(C, Ci):
    LAM, MU, LAMV, MUV = 30000.0, 7500.0, 30000.0, 7500.0
    V_DEV, V_VOL = 10000.0, 50000.0
    def _d_psi(A, m, l):
        Jd = np.sqrt(np.linalg.det(A))
        AinvT = np.linalg.inv(A).T
        return 0.5 * (m * (np.eye(3) - AinvT)
                      + l * (np.log(Jd) + Jd * (Jd - 1.0)) * AinvT)

    C = np.asarray(C, dtype=float)
    Ci = np.asarray(Ci, dtype=float)
    S_eq = 2.0 * _d_psi(C, MU, LAM)
    Ci_inv = np.linalg.inv(Ci)
    S_neq = 2.0 * _d_psi(C @ Ci_inv, MUV, LAMV) @ Ci_inv
    M = S_neq @ C
    return np.concatenate(((S_eq + S_neq).reshape(-1), M.reshape(-1)))

import numpy as np


def viscous_flow(M, Ci):
    LAM, MU, LAMV, MUV = 30000.0, 7500.0, 30000.0, 7500.0
    V_DEV, V_VOL = 10000.0, 50000.0

    M = np.asarray(M, dtype=float)
    Ci = np.asarray(Ci, dtype=float)
    A = M.T
    vol = np.trace(A) / 3.0 * np.eye(3)
    Vinv_A = vol / V_VOL + (A - vol) / (2.0 * V_DEV)
    return (2.0 * Ci @ Vinv_A).reshape(-1)

import numpy as np


def evolution_residual(C_mid, ci_new, ci_old, h):
    LAM, MU, LAMV, MUV = 30000.0, 7500.0, 30000.0, 7500.0
    V_DEV, V_VOL = 10000.0, 50000.0
    SYM = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]

    def _vec_to_sym(v):
        A = np.zeros((3, 3))
        for k, (i, j) in enumerate(SYM):
            A[i, j] = v[k]
            A[j, i] = v[k]
        return A

    def _sym_to_vec(A):
        return np.array([A[i, j] for (i, j) in SYM])
    def _d_psi(A, m, l):
        Jd = np.sqrt(np.linalg.det(A))
        AinvT = np.linalg.inv(A).T
        return 0.5 * (m * (np.eye(3) - AinvT)
                      + l * (np.log(Jd) + Jd * (Jd - 1.0)) * AinvT)
    def _constitutive(C, Ci):
        S_eq = 2.0 * _d_psi(C, MU, LAM)
        Ci_inv = np.linalg.inv(Ci)
        S_neq = 2.0 * _d_psi(C @ Ci_inv, MUV, LAMV) @ Ci_inv
        return S_eq + S_neq, S_neq @ C
    def _flow(M, Ci):
        A = M.T
        vol = np.trace(A) / 3.0 * np.eye(3)
        return 2.0 * Ci @ (vol / V_VOL + (A - vol) / (2.0 * V_DEV))

    Ci_new = _vec_to_sym(np.asarray(ci_new, dtype=float))
    Ci_old = _vec_to_sym(np.asarray(ci_old, dtype=float))
    Ci_mid = 0.5 * (Ci_new + Ci_old)
    _, M = _constitutive(np.asarray(C_mid, dtype=float), Ci_mid)
    R = (Ci_new - Ci_old) / h - _flow(M, Ci_mid)
    return _sym_to_vec(0.5 * (R + R.T))

import numpy as np


def assemble_momentum_residual(n, u, u_n, ci, ci_n, lam):
    LAM, MU, LAMV, MUV = 30000.0, 7500.0, 30000.0, 7500.0
    V_DEV, V_VOL = 10000.0, 50000.0
    SYM = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]

    def _vec_to_sym(v):
        A = np.zeros((3, 3))
        for k, (i, j) in enumerate(SYM):
            A[i, j] = v[k]
            A[j, i] = v[k]
        return A

    def _sym_to_vec(A):
        return np.array([A[i, j] for (i, j) in SYM])
    def _d_psi(A, m, l):
        Jd = np.sqrt(np.linalg.det(A))
        AinvT = np.linalg.inv(A).T
        return 0.5 * (m * (np.eye(3) - AinvT)
                      + l * (np.log(Jd) + Jd * (Jd - 1.0)) * AinvT)
    def _constitutive(C, Ci):
        S_eq = 2.0 * _d_psi(C, MU, LAM)
        Ci_inv = np.linalg.inv(Ci)
        S_neq = 2.0 * _d_psi(C @ Ci_inv, MUV, LAMV) @ Ci_inv
        return S_eq + S_neq, S_neq @ C
    def _mesh(n):
        out = np.zeros(2 * (n + 1) ** 2)
        for j in range(n + 1):
            for i in range(n + 1):
                xi, eta = i / n, j / n
                nd = j * (n + 1) + i
                out[2 * nd] = 480.0 * xi
                out[2 * nd + 1] = 440.0 * xi + eta * (440.0 - 280.0 * xi)
        return out

    def _conn(n):
        c = []
        for j in range(n):
            for i in range(n):
                n0 = j * (n + 1) + i
                c.append([n0, n0 + 1, n0 + n + 2, n0 + n + 1])
        return np.array(c)
    def _grads(xe, xi, eta):
        dN = 0.25 * np.array([[-(1 - eta), -(1 - xi)],
                              [(1 - eta), -(1 + xi)],
                              [(1 + eta), (1 + xi)],
                              [-(1 + eta), (1 - xi)]])
        J = np.asarray(xe).T @ dN
        return float(np.linalg.det(J)), dN @ np.linalg.inv(J)

    T_BAR = np.array([-750.0, 1000.0])
    GP1D = np.array([-1.0, 1.0]) / np.sqrt(3.0)

    coords = _mesh(n).reshape(-1, 2)
    conn = _conn(n)
    nn = coords.shape[0]
    u = np.asarray(u, dtype=float).reshape(nn, 2)
    u_n = np.asarray(u_n, dtype=float).reshape(nn, 2)
    ci = np.asarray(ci, dtype=float)
    ci_n = np.asarray(ci_n, dtype=float)
    u_mid = 0.5 * (u + u_n)

    R = np.zeros(2 * nn)
    g = 0
    for e in range(conn.shape[0]):
        nodes = conn[e]
        xe = coords[nodes]
        for a in GP1D:
            for b in GP1D:
                detJ, dNdX = _grads(xe, a, b)
                F = np.eye(3)
                F[:2, :2] = np.eye(2) + u_mid[nodes].T @ dNdX
                C_mid = F.T @ F
                Ci_mid = _vec_to_sym(0.5 * (ci[6 * g:6 * g + 6]
                                            + ci_n[6 * g:6 * g + 6]))
                S, _ = _constitutive(C_mid, Ci_mid)
                P = F @ S
                for k, nd in enumerate(nodes):
                    R[2 * nd:2 * nd + 2] += P[:2, :2] @ dNdX[k] * detJ
                g += 1

    t = lam * T_BAR
    for j in range(n):
        na = (j + 1) * (n + 1) - 1
        nb = (j + 2) * (n + 1) - 1
        L = float(np.linalg.norm(coords[nb] - coords[na]))
        for gp in GP1D:
            Nv = np.array([0.5 * (1 - gp), 0.5 * (1 + gp)])
            for k, nd in enumerate((na, nb)):
                R[2 * nd:2 * nd + 2] -= Nv[k] * t * (L / 2.0)
    return R

import numpy as np


def condensed_correction(Kqq, KqC, KCq, KCC, Rq, RC):
    Kqq = np.asarray(Kqq, dtype=float)
    KqC = np.asarray(KqC, dtype=float)
    KCq = np.asarray(KCq, dtype=float)
    KCC = np.asarray(KCC, dtype=float)
    Rq = np.asarray(Rq, dtype=float).reshape(-1)
    RC = np.asarray(RC, dtype=float).reshape(-1)
    W = np.linalg.solve(KCC, np.column_stack([KCq, RC.reshape(-1, 1)]))
    return np.linalg.solve(Kqq - KqC @ W[:, :-1], -(Rq - KqC @ W[:, -1]))

import numpy as np


def run_monolithic_pipeline(n, n_steps, p_mult, fd_step, newton_iters):
    SYM = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    GP1D = np.array([-1.0, 1.0]) / np.sqrt(3.0)
    T_END = 10.0

    def _vec_to_sym(v):
        A = np.zeros((3, 3))
        for k, (i, j) in enumerate(SYM):
            A[i, j] = v[k]
            A[j, i] = v[k]
        return A

    coords = build_cooks_mesh(n).reshape(-1, 2)
    nn = coords.shape[0]
    conn = []
    for j in range(n):
        for i in range(n):
            n0 = j * (n + 1) + i
            conn.append([n0, n0 + 1, n0 + n + 2, n0 + n + 1])
    conn = np.array(conn)
    ngp = 4 * conn.shape[0]

    shape = []
    for e in range(conn.shape[0]):
        xe = coords[conn[e]]
        for a in GP1D:
            for b in GP1D:
                shape.append((conn[e], q1_shape_gradients(xe, a, b)))

    fixed = set()
    for nd in range(nn):
        if abs(coords[nd, 0]) < 1e-12:
            fixed.add(2 * nd)
            fixed.add(2 * nd + 1)
    free = np.array([k for k in range(2 * nn) if k not in fixed])
    nq, nc = len(free), 6 * ngp

    def _cmid(u, u_n):
        um = 0.5 * (u.reshape(nn, 2) + u_n.reshape(nn, 2))
        out = []
        for nodes, sd in shape:
            F = np.eye(3)
            F[:2, :2] = np.eye(2) + um[nodes].T @ sd[1:].reshape(4, 2)
            out.append(F.T @ F)
        return out

    def _evo(u, u_n, ci, ci_n, h):
        out = np.zeros(nc)
        for g, C_mid in enumerate(_cmid(u, u_n)):
            out[6 * g:6 * g + 6] = evolution_residual(
                C_mid, ci[6 * g:6 * g + 6], ci_n[6 * g:6 * g + 6], h)
        return out

    u = np.zeros(2 * nn)
    ci = np.tile(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), ngp)
    h = T_END / n_steps
    answer = 0.0

    for step in range(1, n_steps + 1):
        lam = min((step - 0.5) * h / T_END, 1.0) * p_mult
        u_n, ci_n = u.copy(), ci.copy()
        for it in range(newton_iters):
            Rq = assemble_momentum_residual(n, u, u_n, ci, ci_n, lam)[free]
            RC = _evo(u, u_n, ci, ci_n, h)
            Kqq = np.zeros((nq, nq))
            KCq = np.zeros((nc, nq))
            KqC = np.zeros((nq, nc))
            KCC = np.zeros((nc, nc))
            for a in range(nq):
                up, um = u.copy(), u.copy()
                up[free[a]] += fd_step
                um[free[a]] -= fd_step
                Kqq[:, a] = (assemble_momentum_residual(n, up, u_n, ci, ci_n, lam)[free]
                             - assemble_momentum_residual(n, um, u_n, ci, ci_n, lam)[free]) / (2 * fd_step)
                KCq[:, a] = (_evo(up, u_n, ci, ci_n, h)
                             - _evo(um, u_n, ci, ci_n, h)) / (2 * fd_step)
            for a in range(nc):
                cp, cm = ci.copy(), ci.copy()
                cp[a] += fd_step
                cm[a] -= fd_step
                KqC[:, a] = (assemble_momentum_residual(n, u, u_n, cp, ci_n, lam)[free]
                             - assemble_momentum_residual(n, u, u_n, cm, ci_n, lam)[free]) / (2 * fd_step)
                KCC[:, a] = (_evo(u, u_n, cp, ci_n, h)
                             - _evo(u, u_n, cm, ci_n, h)) / (2 * fd_step)

            dq = condensed_correction(Kqq, KqC, KCq, KCC, Rq, RC)
            if it == 0 and step == n_steps:
                answer = float(np.linalg.norm(dq))
                _ = viscous_flow(constitutive_response(
                    _cmid(u, u_n)[0], _vec_to_sym(0.5 * (ci[:6] + ci_n[:6]))
                )[9:].reshape(3, 3), _vec_to_sym(0.5 * (ci[:6] + ci_n[:6])))
            W = np.linalg.solve(KCC, np.column_stack([KCq, RC.reshape(-1, 1)]))
            u[free] += dq
            ci += -W[:, -1] - W[:, :-1] @ dq
    return answer
SCICODE_GOLD_EOF
