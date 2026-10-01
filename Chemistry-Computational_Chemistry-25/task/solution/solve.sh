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
from scipy.linalg import eigh
from scipy.special import gammainc, gamma

BOHR_PER_ANGSTROM = 1.0/0.52917721092
HARTREE2KCAL = 627.5094740631


def _dfact(n):
    return 1.0 if n <= 0 else float(np.prod(np.arange(n, 0, -2)))


def basis_table(symbols: "list", basis_data: "dict") -> "np.ndarray":
    if len(symbols) != 2:
        raise ValueError("symbols must name exactly two atoms")
    for s in symbols:
        if s not in basis_data:
            raise ValueError(f"no basis entry for {s}")
    rows = []
    for ia, sym in enumerate(symbols):
        for l, e, c in basis_data[sym]:
            e = np.asarray(e, dtype=float)
            c = np.asarray(c, dtype=float)
            if e.size != 3 or c.size != 3:
                raise ValueError("every shell must hold exactly three primitives")
            comps = [(0, 0, 0)] if int(l) == 0 else [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
            for lv in comps:
                dd = _dfact(2*lv[0]-1)*_dfact(2*lv[1]-1)*_dfact(2*lv[2]-1)
                L = sum(lv)
                cn = c*np.power(2*e/np.pi, 0.75)*np.power(4*e, L/2.0)/np.sqrt(dd)
                p = e[:, None] + e[None, :]
                s = float(np.sum(cn[:, None]*cn[None, :]*np.power(np.pi/p, 1.5)*dd
                                 / np.power(2*p, L)))
                if not np.isfinite(s) or s <= 0.0:
                    raise ValueError("shell has no positive self-overlap")
                rows.append(np.concatenate([[float(ia)], np.array(lv, dtype=float),
                                            e, cn/np.sqrt(s)]))
    return np.array(rows)

import numpy as np
from scipy.linalg import eigh
from scipy.special import gammainc, gamma

BOHR_PER_ANGSTROM = 1.0/0.52917721092
HARTREE2KCAL = 627.5094740631


def _boys(nmax, T):
    T = np.asarray(T, dtype=float)
    out = np.empty((nmax + 1,) + T.shape)
    small = T < 1e-8
    Ts = np.where(small, 1.0, T)
    for n in range(nmax + 1):
        a = n + 0.5
        big = 0.5*gamma(a)*gammainc(a, Ts)*np.power(Ts, -a)
        s = np.zeros_like(T)
        fac = 1.0
        for k in range(8):
            s = s + ((-T)**k)/(fac*(2*n + 2*k + 1))
            fac *= (k + 1)
        out[n] = np.where(small, s, big)
    return out


def _hermite_E(i, j, t, Qx, a, b):
    p = a + b
    q = a*b/p
    if t < 0 or t > i + j:
        return 0.0
    if i == 0 and j == 0:
        return np.exp(-q*Qx*Qx) if t == 0 else 0.0
    if j == 0:
        return (1.0/(2*p))*_hermite_E(i-1, j, t-1, Qx, a, b) \
               - (q*Qx/a)*_hermite_E(i-1, j, t, Qx, a, b) \
               + (t+1)*_hermite_E(i-1, j, t+1, Qx, a, b)
    return (1.0/(2*p))*_hermite_E(i, j-1, t-1, Qx, a, b) \
           + (q*Qx/b)*_hermite_E(i, j-1, t, Qx, a, b) \
           + (t+1)*_hermite_E(i, j-1, t+1, Qx, a, b)


def _hermite_R(tmax, umax, vmax, p, PQ):
    T = p*np.sum(PQ*PQ, axis=-1)
    nmax = tmax + umax + vmax
    F = _boys(nmax, T)
    R = {(0, 0, 0, n): ((-2.0*p)**n)*F[n] for n in range(nmax + 1)}
    for tot in range(1, nmax + 1):
        for t in range(tot + 1):
            for u in range(tot - t + 1):
                v = tot - t - u
                if t > tmax or u > umax or v > vmax:
                    continue
                for n in range(nmax - tot + 1):
                    if t > 0:
                        val = PQ[..., 0]*R[(t-1, u, v, n+1)]
                        if t > 1:
                            val = val + (t-1)*R[(t-2, u, v, n+1)]
                    elif u > 0:
                        val = PQ[..., 1]*R[(t, u-1, v, n+1)]
                        if u > 1:
                            val = val + (u-1)*R[(t, u-2, v, n+1)]
                    else:
                        val = PQ[..., 2]*R[(t, u, v-1, n+1)]
                        if v > 1:
                            val = val + (v-1)*R[(t, u, v-2, n+1)]
                    R[(t, u, v, n)] = val
    return R


def _unpack(table):
    table = np.asarray(table, dtype=float)
    if table.ndim != 2 or table.shape[1] != 10:
        raise ValueError("basis table must have ten columns")
    return (table[:, 0].astype(int), table[:, 1:4].astype(int),
            table[:, 4:7], table[:, 7:10])


def _pair_data(centers, lxyz, exps, coefs, coords, i, j):
    A = coords[centers[i]]
    B = coords[centers[j]]
    ei, ej = exps[i], exps[j]
    a = ei[:, None]
    b = ej[None, :]
    p = (a + b).ravel()
    P = ((a[..., None]*A + b[..., None]*B)/(a + b)[..., None]).reshape(-1, 3)
    c = (coefs[i][:, None]*coefs[j][None, :]).ravel()
    ab = [(x, y) for x in ei for y in ej]
    E = []
    for d in range(3):
        tmax = int(lxyz[i][d] + lxyz[j][d])
        tab = np.zeros((tmax + 1, p.size))
        for k, (aa, bb) in enumerate(ab):
            for t in range(tmax + 1):
                tab[t, k] = _hermite_E(int(lxyz[i][d]), int(lxyz[j][d]), t, A[d]-B[d], aa, bb)
        E.append(tab)
    return dict(p=p, P=P, c=c, E=E, ab=ab, A=A, B=B,
                lsum=tuple(int(lxyz[i][d] + lxyz[j][d]) for d in range(3)))


def core_integrals(table: "np.ndarray", coords: "np.ndarray", charges: "list") -> "np.ndarray":
    centers, lxyz, exps, coefs = _unpack(table)
    coords = np.asarray(coords, dtype=float)
    Z = np.asarray(charges, dtype=float)
    n = len(centers)
    if n == 0:
        raise ValueError("empty basis")
    if coords.shape != (2, 3) or Z.size != 2:
        raise ValueError("coords must have shape (2, 3) and charges length two")
    S = np.zeros((n, n)); T = np.zeros((n, n)); V = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            d = _pair_data(centers, lxyz, exps, coefs, coords, i, j)
            p, c, E, ab = d["p"], d["c"], d["E"], d["ab"]
            Q = d["A"] - d["B"]
            bj = np.array([y for _, y in ab])

            def _ov(shift):
                tot = np.ones_like(p)
                for dax in range(3):
                    jj = int(lxyz[j][dax]) + shift[dax]
                    if jj < 0:
                        return np.zeros_like(p)
                    tot = tot*np.array([_hermite_E(int(lxyz[i][dax]), jj, 0, Q[dax], aa, bb)
                                        for aa, bb in ab])
                return tot*np.power(np.pi/p, 1.5)

            s0 = _ov((0, 0, 0))
            S[i, j] = S[j, i] = float(np.sum(c*s0))
            t = bj*(2*int(np.sum(lxyz[j])) + 3)*s0
            for dax in range(3):
                up = [0, 0, 0]; up[dax] = 2
                dn = [0, 0, 0]; dn[dax] = -2
                t = t - 2.0*bj*bj*_ov(tuple(up))
                if int(lxyz[j][dax]) >= 2:
                    t = t - 0.5*lxyz[j][dax]*(lxyz[j][dax]-1)*_ov(tuple(dn))
            T[i, j] = T[j, i] = float(np.sum(c*t))
            ls = d["lsum"]
            acc = np.zeros_like(p)
            for C, Zc in zip(coords, Z):
                R = _hermite_R(ls[0], ls[1], ls[2], p, d["P"] - C[None, :])
                sm = np.zeros_like(p)
                for t1 in range(ls[0] + 1):
                    for u1 in range(ls[1] + 1):
                        for v1 in range(ls[2] + 1):
                            sm = sm + E[0][t1]*E[1][u1]*E[2][v1]*R[(t1, u1, v1, 0)]
                acc = acc - Zc*sm
            V[i, j] = V[j, i] = float(np.sum(c*2.0*np.pi/p*acc))
    return np.stack([S, T, V])

import numpy as np
from scipy.linalg import eigh
from scipy.special import gammainc, gamma

BOHR_PER_ANGSTROM = 1.0/0.52917721092
HARTREE2KCAL = 627.5094740631


def _boys(nmax, T):
    T = np.asarray(T, dtype=float)
    out = np.empty((nmax + 1,) + T.shape)
    small = T < 1e-8
    Ts = np.where(small, 1.0, T)
    for n in range(nmax + 1):
        a = n + 0.5
        big = 0.5*gamma(a)*gammainc(a, Ts)*np.power(Ts, -a)
        s = np.zeros_like(T)
        fac = 1.0
        for k in range(8):
            s = s + ((-T)**k)/(fac*(2*n + 2*k + 1))
            fac *= (k + 1)
        out[n] = np.where(small, s, big)
    return out


def _hermite_E(i, j, t, Qx, a, b):
    p = a + b
    q = a*b/p
    if t < 0 or t > i + j:
        return 0.0
    if i == 0 and j == 0:
        return np.exp(-q*Qx*Qx) if t == 0 else 0.0
    if j == 0:
        return (1.0/(2*p))*_hermite_E(i-1, j, t-1, Qx, a, b) \
               - (q*Qx/a)*_hermite_E(i-1, j, t, Qx, a, b) \
               + (t+1)*_hermite_E(i-1, j, t+1, Qx, a, b)
    return (1.0/(2*p))*_hermite_E(i, j-1, t-1, Qx, a, b) \
           + (q*Qx/b)*_hermite_E(i, j-1, t, Qx, a, b) \
           + (t+1)*_hermite_E(i, j-1, t+1, Qx, a, b)


def _hermite_R(tmax, umax, vmax, p, PQ):
    T = p*np.sum(PQ*PQ, axis=-1)
    nmax = tmax + umax + vmax
    F = _boys(nmax, T)
    R = {(0, 0, 0, n): ((-2.0*p)**n)*F[n] for n in range(nmax + 1)}
    for tot in range(1, nmax + 1):
        for t in range(tot + 1):
            for u in range(tot - t + 1):
                v = tot - t - u
                if t > tmax or u > umax or v > vmax:
                    continue
                for n in range(nmax - tot + 1):
                    if t > 0:
                        val = PQ[..., 0]*R[(t-1, u, v, n+1)]
                        if t > 1:
                            val = val + (t-1)*R[(t-2, u, v, n+1)]
                    elif u > 0:
                        val = PQ[..., 1]*R[(t, u-1, v, n+1)]
                        if u > 1:
                            val = val + (u-1)*R[(t, u-2, v, n+1)]
                    else:
                        val = PQ[..., 2]*R[(t, u, v-1, n+1)]
                        if v > 1:
                            val = val + (v-1)*R[(t, u, v-2, n+1)]
                    R[(t, u, v, n)] = val
    return R


def _unpack(table):
    table = np.asarray(table, dtype=float)
    if table.ndim != 2 or table.shape[1] != 10:
        raise ValueError("basis table must have ten columns")
    return (table[:, 0].astype(int), table[:, 1:4].astype(int),
            table[:, 4:7], table[:, 7:10])


def _pair_data(centers, lxyz, exps, coefs, coords, i, j):
    A = coords[centers[i]]
    B = coords[centers[j]]
    ei, ej = exps[i], exps[j]
    a = ei[:, None]
    b = ej[None, :]
    p = (a + b).ravel()
    P = ((a[..., None]*A + b[..., None]*B)/(a + b)[..., None]).reshape(-1, 3)
    c = (coefs[i][:, None]*coefs[j][None, :]).ravel()
    ab = [(x, y) for x in ei for y in ej]
    E = []
    for d in range(3):
        tmax = int(lxyz[i][d] + lxyz[j][d])
        tab = np.zeros((tmax + 1, p.size))
        for k, (aa, bb) in enumerate(ab):
            for t in range(tmax + 1):
                tab[t, k] = _hermite_E(int(lxyz[i][d]), int(lxyz[j][d]), t, A[d]-B[d], aa, bb)
        E.append(tab)
    return dict(p=p, P=P, c=c, E=E, ab=ab, A=A, B=B,
                lsum=tuple(int(lxyz[i][d] + lxyz[j][d]) for d in range(3)))


def electron_repulsion(table: "np.ndarray", coords: "np.ndarray") -> "np.ndarray":
    centers, lxyz, exps, coefs = _unpack(table)
    coords = np.asarray(coords, dtype=float)
    n = len(centers)
    if n == 0:
        raise ValueError("empty basis")
    if coords.shape != (2, 3):
        raise ValueError("coords must have shape (2, 3)")
    keys = [(i, j) for i in range(n) for j in range(i + 1)]
    pr = {k: _pair_data(centers, lxyz, exps, coefs, coords, k[0], k[1]) for k in keys}
    out = np.zeros((n, n, n, n))
    for xa, (i, j) in enumerate(keys):
        d1 = pr[(i, j)]
        for (k, l) in keys[:xa + 1]:
            d2 = pr[(k, l)]
            p, q = d1["p"], d2["p"]
            alpha = p[:, None]*q[None, :]/(p[:, None] + q[None, :])
            PQ = d1["P"][:, None, :] - d2["P"][None, :, :]
            l1, l2 = d1["lsum"], d2["lsum"]
            R = _hermite_R(l1[0]+l2[0], l1[1]+l2[1], l1[2]+l2[2], alpha, PQ)
            acc = np.zeros_like(alpha)
            for t in range(l1[0] + 1):
                for u in range(l1[1] + 1):
                    for v in range(l1[2] + 1):
                        e1 = (d1["E"][0][t]*d1["E"][1][u]*d1["E"][2][v])[:, None]
                        for tt in range(l2[0] + 1):
                            for uu in range(l2[1] + 1):
                                for vv in range(l2[2] + 1):
                                    e2 = (d2["E"][0][tt]*d2["E"][1][uu]
                                          * d2["E"][2][vv])[None, :]
                                    acc = acc + ((-1)**(tt+uu+vv))*e1*e2 \
                                        * R[(t+tt, u+uu, v+vv, 0)]
            pref = 2.0*np.pi**2.5/(p[:, None]*q[None, :]*np.sqrt(p[:, None] + q[None, :]))
            val = float(np.sum(d1["c"][:, None]*d2["c"][None, :]*pref*acc))
            for a, b in ((i, j), (j, i)):
                for cq, dq in ((k, l), (l, k)):
                    out[a, b, cq, dq] = val
                    out[cq, dq, a, b] = val
    return out

import numpy as np
from scipy.linalg import eigh
from scipy.special import gammainc, gamma

BOHR_PER_ANGSTROM = 1.0/0.52917721092
HARTREE2KCAL = 627.5094740631


def _unpack(table):
    table = np.asarray(table, dtype=float)
    if table.ndim != 2 or table.shape[1] != 10:
        raise ValueError("basis table must have ten columns")
    return (table[:, 0].astype(int), table[:, 1:4].astype(int),
            table[:, 4:7], table[:, 7:10])


def descriptor_matrices(table: "np.ndarray", coords: "np.ndarray", eri: "np.ndarray") -> "np.ndarray":
    centers, lxyz, exps, coefs = _unpack(table)
    coords = np.asarray(coords, dtype=float)
    eri = np.asarray(eri, dtype=float)
    n = len(centers)
    if eri.shape != (n, n, n, n):
        raise ValueError("eri does not match the basis size")
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(coords[centers[i]] - coords[centers[j]])
    return np.stack([D, np.einsum("mmnn->mn", eri), np.einsum("mnmn->mn", eri),
                     np.einsum("mnnn->mn", eri)])

import numpy as np
from scipy.linalg import eigh
from scipy.special import gammainc, gamma

BOHR_PER_ANGSTROM = 1.0/0.52917721092
HARTREE2KCAL = 627.5094740631


def _sym_power(S, power):
    w, U = np.linalg.eigh(np.asarray(S, dtype=float))
    if np.min(w) <= 0.0:
        raise ValueError("overlap matrix is not positive definite")
    return (U*np.power(w, power)) @ U.T


def _safe_inverse(A):
    A = np.asarray(A, dtype=float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("operand is not square")
    try:
        B = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        raise ValueError("operand has no inverse")
    if not np.all(np.isfinite(B)):
        raise ValueError("operand has no finite inverse")
    return B


def operand_table(core: "np.ndarray", descriptors: "np.ndarray") -> "np.ndarray":
    core = np.asarray(core, dtype=float)
    descriptors = np.asarray(descriptors, dtype=float)
    if core.ndim != 3 or core.shape[0] != 3:
        raise ValueError("core must have shape (3, n, n)")
    n = core.shape[1]
    if descriptors.shape != (4, n, n):
        raise ValueError("descriptors must have shape (4, n, n)")
    S, T, V = core
    D, mmnn, mnmn, mnnn = descriptors
    Dinv = np.where(D == 0.0, 0.0, 1.0/np.where(D == 0.0, 1.0, D))
    return np.stack([S, T, V, _sym_power(S, -0.5), D, Dinv, _sym_power(S, 0.5),
                     mmnn, mnmn, mnnn, _safe_inverse(S)])

import numpy as np
from scipy.linalg import eigh
from scipy.special import gammainc, gamma

BOHR_PER_ANGSTROM = 1.0/0.52917721092
HARTREE2KCAL = 627.5094740631


def _safe_inverse(A):
    A = np.asarray(A, dtype=float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("operand is not square")
    try:
        B = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        raise ValueError("operand has no inverse")
    if not np.all(np.isfinite(B)):
        raise ValueError("operand has no finite inverse")
    return B


def _library():
    """(kind, payload) for operation ids 0..130, in the fixed order of the contract."""
    lib = []
    for op in "+-*/":
        for c in (0.5, 1.0, 3.0, 4.0):
            lib.append(("const", (op, c)))
    for op in "+-*/":
        for k in range(11):
            lib.append(("amat", (op, k)))
    for k in range(11):
        lib.append(("amul", k))
    for op in "+-*/":
        for k in range(11):
            lib.append(("ainv", (op, k)))
    for k in range(11):
        lib.append(("ainvmul", k))
    lib.append(("unary", "exp"))
    lib.append(("unary", "ln"))
    lib.append(("power", 0.5))
    lib.append(("power", 2.0))
    lib.append(("unary", "expneg"))
    return lib


def apply_operation(M: "np.ndarray", op_id: "int", operands: "np.ndarray") -> "np.ndarray":
    lib = _library()
    op_id = int(op_id)
    if op_id < 0 or op_id >= len(lib):
        raise ValueError("op_id outside the library")
    M = np.asarray(M, dtype=float)
    operands = np.asarray(operands, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix")
    if operands.shape != (11,) + M.shape:
        raise ValueError("operands must have shape (11, n, n) matching M")
    kind, pay = lib[op_id]
    with np.errstate(all="ignore"):
        if kind == "const":
            o, c = pay
            out = M + c if o == "+" else M - c if o == "-" else M*c if o == "*" else M/c
        elif kind == "amat":
            o, k = pay
            X = operands[k]
            out = M + X if o == "+" else M - X if o == "-" else M*X if o == "*" else M/X
        elif kind == "amul":
            out = M @ operands[pay]
        elif kind == "ainvmul":
            out = M @ _safe_inverse(operands[pay])
        elif kind == "ainv":
            o, k = pay
            X = _safe_inverse(operands[k])
            out = M + X if o == "+" else M - X if o == "-" else M*X if o == "*" else M/X
        elif kind == "unary":
            out = np.exp(M) if pay == "exp" else np.log(M) if pay == "ln" else np.exp(-M)
        else:
            out = np.power(M, pay)
    out = np.asarray(out, dtype=float)
    if not np.all(np.isfinite(out)):
        raise ValueError("operation produced a non-finite workspace matrix")
    return out

import numpy as np
from scipy.linalg import eigh
from scipy.special import gammainc, gamma

BOHR_PER_ANGSTROM = 1.0/0.52917721092
HARTREE2KCAL = 627.5094740631


def workspace_matrix(hcore: "np.ndarray", program: "list", operands: "np.ndarray") -> "np.ndarray":
    M = np.asarray(hcore, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("hcore must be a square matrix")
    for op_id in program:
        M = apply_operation(M, op_id, operands)
    L = np.tril(M)
    return L + L.T - np.diag(np.diag(L))

import numpy as np
from scipy.linalg import eigh
from scipy.special import gammainc, gamma

BOHR_PER_ANGSTROM = 1.0/0.52917721092
HARTREE2KCAL = 627.5094740631


def program_energy(M: "np.ndarray", S: "np.ndarray", hcore: "np.ndarray", eri: "np.ndarray", n_occ: "int") -> "float":
    M = np.asarray(M, dtype=float)
    S = np.asarray(S, dtype=float)
    hcore = np.asarray(hcore, dtype=float)
    eri = np.asarray(eri, dtype=float)
    n = M.shape[0]
    n_occ = int(n_occ)
    if n_occ < 1 or n_occ > n:
        raise ValueError("n_occ outside the basis")
    if S.shape != (n, n) or hcore.shape != (n, n) or eri.shape != (n, n, n, n):
        raise ValueError("inconsistent matrix shapes")
    try:
        w, C = eigh(M, S)
    except Exception:
        raise ValueError("the generalised eigenvalue problem could not be solved")
    Co = C[:, :n_occ]
    P = 2.0*Co @ Co.T
    J = np.einsum("mnls,ls->mn", eri, P)
    K = np.einsum("mlsn,ls->mn", eri, P)
    e = 0.5*float(np.sum(P*(hcore + hcore + J - 0.5*K)))
    if not np.isfinite(e):
        raise ValueError("non-finite electronic energy")
    return e

import numpy as np
from scipy.linalg import eigh
from scipy.special import gammainc, gamma

BOHR_PER_ANGSTROM = 1.0/0.52917721092
HARTREE2KCAL = 627.5094740631


def target_function(species: "list", e_pips: "list", e_ref: "list", e_atom: "dict") -> "float":
    e_pips = np.asarray(e_pips, dtype=float)
    e_ref = np.asarray(e_ref, dtype=float)
    species = [tuple(s) for s in species]
    if len(species) != e_pips.size or e_pips.size != e_ref.size or e_pips.size == 0:
        raise ValueError("species, e_pips and e_ref must have the same non-zero length")
    if not np.all(np.isfinite(e_pips)) or not np.all(np.isfinite(e_ref)):
        raise ValueError("non-finite energies")
    res = np.empty(e_pips.size)
    for s in dict.fromkeys(species):
        for a in s:
            if a not in e_atom:
                raise ValueError(f"no atomic energy for {a}")
        idx = [i for i, x in enumerate(species) if x == s]
        Di = float(e_atom[s[0]] + e_atom[s[1]])
        Dbar = float(np.mean(e_pips[idx] - e_ref[idx])) + Di
        for i in idx:
            res[i] = (e_pips[i] - Dbar) - (e_ref[i] - Di)
    return float(np.sqrt(np.mean(res*res)))*627.5094740631

import numpy as np
from scipy.linalg import eigh
from scipy.special import gammainc, gamma

BOHR_PER_ANGSTROM = 1.0/0.52917721092
HARTREE2KCAL = 627.5094740631


def best_two_operation_program(species: "list", bond_lengths: "list", e_ref: "list", e_atom: "dict",
                                       basis_data: "dict", charges: "dict") -> "float":
    species = [tuple(s) for s in species]
    bond_lengths = np.asarray(bond_lengths, dtype=float)
    e_ref = np.asarray(e_ref, dtype=float)
    if not (len(species) == bond_lengths.size == e_ref.size) or len(species) == 0:
        raise ValueError("species, bond_lengths and e_ref must have the same non-zero length")
    if not np.all(np.isfinite(bond_lengths)) or np.any(bond_lengths <= 0.0):
        raise ValueError("bond lengths must be positive and finite")

    geoms = []
    for (a, b), R in zip(species, bond_lengths):
        bohr = 1.0/0.52917721092
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, float(R)*bohr]])
        Z = [float(charges[a]), float(charges[b])]
        table = basis_table([a, b], basis_data)
        core = core_integrals(table, coords, Z)
        eri = electron_repulsion(table, coords)
        desc = descriptor_matrices(table, coords, eri)
        operands = operand_table(core, desc)
        n_elec = int(round(Z[0] + Z[1]))
        if n_elec % 2:
            raise ValueError("the molecule does not have an even number of electrons")
        geoms.append(dict(S=core[0], hcore=core[1] + core[2], eri=eri,
                          enuc=Z[0]*Z[1]/(float(R)*bohr),
                          operands=operands, n_occ=n_elec//2))

    matmul_ids = list(range(60, 71)) + list(range(115, 126))
    best = None
    for op1 in matmul_ids:
        try:
            for g in geoms:
                apply_operation(g["hcore"], op1, g["operands"])
        except ValueError:
            continue                  # this opening already fails: no need to try any second op
        for op2 in range(131):
            e_pips, ok = [], True
            for g in geoms:
                try:
                    M = workspace_matrix(g["hcore"], (op1, op2), g["operands"])
                    e_pips.append(program_energy(M, g["S"], g["hcore"], g["eri"],
                                                         g["n_occ"]) + g["enuc"])
                except ValueError:
                    ok = False
                    break
            if not ok:
                continue
            try:
                F = target_function(species, e_pips, e_ref, e_atom)
            except ValueError:
                continue
            if np.isfinite(F) and (best is None or (F, op1, op2) < best):
                best = (F, op1, op2)
    if best is None:
        raise ValueError("no candidate program produced a finite target function")
    return float(best[0])
SCICODE_GOLD_EOF
