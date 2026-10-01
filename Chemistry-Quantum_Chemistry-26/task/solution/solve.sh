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
from scipy.special import gamma, gammainc


def _cartesian_components(l):
    """Cartesian exponent triples of a shell; a p shell is ordered x, y, z."""
    return [(lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)]


def _boys(n_max, t):
    """Boys functions F_n(t) for n = 0..n_max; result has shape (n_max + 1,) + t.shape."""
    t = np.asarray(t, dtype=float)
    out = np.empty((n_max + 1,) + t.shape)
    small = t < 1e-8
    t_safe = np.where(small, 1.0, t)
    for n in range(n_max + 1):
        a = n + 0.5
        series = 1.0 / (2 * n + 1) - t / (2 * n + 3) + t * t / (2.0 * (2 * n + 5))
        out[n] = np.where(small, series, gamma(a) * gammainc(a, t_safe) / (2.0 * t_safe ** a))
    return out


def _normalized_shells(coords, shells):
    """Validate the shells and fold primitive and contraction normalisation into the coefficients."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_atoms, 3)")
    out = []
    for shell in shells:
        center, l = int(shell["center"]), int(shell["l"])
        exps = np.asarray(shell["exponents"], dtype=float)
        coefs = np.asarray(shell["coefficients"], dtype=float)
        if l not in (0, 1):
            raise ValueError("only s (l = 0) and p (l = 1) shells are supported")
        if not 0 <= center < coords.shape[0]:
            raise ValueError("shell centre index out of range")
        if exps.ndim != 1 or exps.size == 0 or exps.shape != coefs.shape or np.any(exps <= 0.0):
            raise ValueError("exponents and coefficients must be matching non-empty lists, exponents > 0")
        d = coefs * (2.0 * exps / np.pi) ** 0.75 * (4.0 * exps) ** (0.5 * l)
        p = exps[:, None] + exps[None, :]
        self_overlap = np.sum(d[:, None] * d[None, :] * (np.pi / p) ** 1.5 / (2.0 * p) ** l)
        out.append((coords[center].copy(), l, exps, d / np.sqrt(self_overlap)))
    return out


def _hermite_expansion(la, lb, a, b, q):
    """1D Hermite expansion coefficients E[i][j][t] for all primitive pairs (arrays a, b), q = A - B."""
    p = a + b
    xpa, xpb = -b / p * q, a / p * q
    e = [[None] * (lb + 1) for _ in range(la + 1)]
    e[0][0] = np.exp(-a * b / p * q * q)[None, :]
    for i in range(la + 1):
        for j in range(lb + 1):
            if i == 0 and j == 0:
                continue
            prev, xp = (e[i - 1][j], xpa) if i > 0 else (e[i][j - 1], xpb)
            n = i + j
            cur = np.zeros((n + 1, p.size))
            for t in range(n + 1):
                val = np.zeros(p.size)
                if t >= 1:
                    val = val + 0.5 / p * prev[t - 1]
                if t <= n - 1:
                    val = val + xp * prev[t]
                if t + 1 <= n - 1:
                    val = val + (t + 1) * prev[t + 1]
                cur[t] = val
            e[i][j] = cur
    return e


def _hermite_coulomb(l_max, alpha, x, y, z):
    """Hermite Coulomb integrals R_tuv(alpha, (x, y, z)) for t + u + v <= l_max; shape (L+1, L+1, L+1, ...)."""
    shape = np.broadcast(alpha, x, y, z).shape
    boys = _boys(l_max, alpha * (x * x + y * y + z * z))
    r = np.zeros((l_max + 1,) * 4 + shape)
    factor = np.ones(shape)
    for n in range(l_max + 1):
        r[n, 0, 0, 0] = factor * boys[n]
        factor = factor * (-2.0 * alpha)
    for order in range(1, l_max + 1):
        for t in range(order + 1):
            for u in range(order - t + 1):
                v = order - t - u
                for n in range(l_max - order + 1):
                    if t > 0:
                        val = x * r[n + 1, t - 1, u, v]
                        if t > 1:
                            val = val + (t - 1) * r[n + 1, t - 2, u, v]
                    elif u > 0:
                        val = y * r[n + 1, t, u - 1, v]
                        if u > 1:
                            val = val + (u - 1) * r[n + 1, t, u - 2, v]
                    else:
                        val = z * r[n + 1, t, u, v - 1]
                        if v > 1:
                            val = val + (v - 1) * r[n + 1, t, u, v - 2]
                    r[n, t, u, v] = val
    return r[0]


def _shell_pair(shell_a, shell_b, extra=0):
    """Primitive-pair data and Hermite expansion of the product of two shells."""
    center_a, la, exps_a, coefs_a = shell_a
    center_b, lb, exps_b, coefs_b = shell_b
    a = np.repeat(exps_a, exps_b.size)
    b = np.tile(exps_b, exps_a.size)
    p = a + b
    q = center_a - center_b
    e = [_hermite_expansion(la, lb + extra, a, b, np.full_like(a, q[k])) for k in range(3)]
    comps_a, comps_b = _cartesian_components(la), _cartesian_components(lb)
    l_sum = la + lb
    tuv = [(t, u, v) for t in range(l_sum + 1) for u in range(l_sum + 1 - t)
           for v in range(l_sum + 1 - t - u)]
    eh = np.zeros((len(comps_a) * len(comps_b), len(tuv), a.size))
    k = 0
    for ax, ay, az in comps_a:
        for bx, by, bz in comps_b:
            for m, (t, u, v) in enumerate(tuv):
                if t <= ax + bx and u <= ay + by and v <= az + bz:
                    eh[k, m] = e[0][ax][bx][t] * e[1][ay][by][u] * e[2][az][bz][v]
            k += 1
    return {"a": a, "b": b, "p": p, "lb": lb, "l_sum": l_sum, "tuv": tuv, "e": e, "eh": eh,
            "center": (a[:, None] * center_a[None, :] + b[:, None] * center_b[None, :]) / p[:, None],
            "cc": np.repeat(coefs_a, exps_b.size) * np.tile(coefs_b, exps_a.size),
            "comps_a": comps_a, "comps_b": comps_b}


def _shell_offsets(norm_shells):
    offsets = [0]
    for _, l, _, _ in norm_shells:
        offsets.append(offsets[-1] + (l + 1) * (l + 2) // 2)
    return offsets


def compute_one_electron_integrals(coords: "np.ndarray", nuclear_charges: "np.ndarray", shells: list) -> "np.ndarray":
    coords = np.asarray(coords, dtype=float)
    charges = np.asarray(nuclear_charges, dtype=float).ravel()
    norm_shells = _normalized_shells(coords, shells)
    if charges.size != coords.shape[0]:
        raise ValueError("one nuclear charge per atom is required")
    offsets = _shell_offsets(norm_shells)
    nbf = offsets[-1]
    s_mat, t_mat, v_mat = np.zeros((nbf, nbf)), np.zeros((nbf, nbf)), np.zeros((nbf, nbf))
    for i, shell_a in enumerate(norm_shells):
        for j, shell_b in enumerate(norm_shells[: i + 1]):
            pair = _shell_pair(shell_a, shell_b, extra=2)
            p, b, e = pair["p"], pair["b"], pair["e"]
            pref = pair["cc"] * (np.pi / p) ** 1.5
            for ia, (ax, ay, az) in enumerate(pair["comps_a"]):
                for ib, (bx, by, bz) in enumerate(pair["comps_b"]):
                    ov, kin = [], []
                    for d, (la_d, lb_d) in enumerate(((ax, bx), (ay, by), (az, bz))):
                        ov.append(e[d][la_d][lb_d][0])
                        kd = -2.0 * b * b * e[d][la_d][lb_d + 2][0] + b * (2 * lb_d + 1) * e[d][la_d][lb_d][0]
                        if lb_d >= 2:
                            kd = kd - 0.5 * lb_d * (lb_d - 1) * e[d][la_d][lb_d - 2][0]
                        kin.append(kd)
                    s_mat[offsets[i] + ia, offsets[j] + ib] = np.sum(pref * ov[0] * ov[1] * ov[2])
                    t_mat[offsets[i] + ia, offsets[j] + ib] = np.sum(
                        pref * (kin[0] * ov[1] * ov[2] + ov[0] * kin[1] * ov[2] + ov[0] * ov[1] * kin[2]))
            vals = np.zeros(pair["eh"].shape[0])
            for nucleus, z in zip(coords, charges):
                pc = pair["center"] - nucleus[None, :]
                r = _hermite_coulomb(pair["l_sum"], p, pc[:, 0], pc[:, 1], pc[:, 2])
                rv = np.array([r[t, u, v] for (t, u, v) in pair["tuv"]])
                vals -= z * np.einsum("ktp,tp,p->k", pair["eh"], rv, pair["cc"] * 2.0 * np.pi / p)
            nb = len(pair["comps_b"])
            for ia in range(len(pair["comps_a"])):
                for ib in range(nb):
                    v_mat[offsets[i] + ia, offsets[j] + ib] = vals[ia * nb + ib]
    lower = np.tril_indices(nbf, -1)
    for m in (s_mat, t_mat, v_mat):
        m[lower[1], lower[0]] = m[lower]
    return np.stack([s_mat, t_mat, v_mat])

import numpy as np


def compute_electron_repulsion_integrals(coords: "np.ndarray", shells: list) -> "np.ndarray":
    norm_shells = _normalized_shells(coords, shells)
    offsets = _shell_offsets(norm_shells)
    nbf = offsets[-1]
    eri = np.zeros((nbf, nbf, nbf, nbf))
    pairs = {}
    for i in range(len(norm_shells)):
        for j in range(i + 1):
            pairs[(i, j)] = _shell_pair(norm_shells[i], norm_shells[j])
    keys = sorted(pairs)
    for x, (i, j) in enumerate(keys):
        bra = pairs[(i, j)]
        for k, l in keys[: x + 1]:
            ket = pairs[(k, l)]
            p, q = bra["p"][:, None], ket["p"][None, :]
            alpha = p * q / (p + q)
            pq = bra["center"][:, None, :] - ket["center"][None, :, :]
            r = _hermite_coulomb(bra["l_sum"] + ket["l_sum"], alpha, pq[..., 0], pq[..., 1], pq[..., 2])
            rm = np.empty((len(bra["tuv"]), len(ket["tuv"])) + alpha.shape)
            for m, (t, u, v) in enumerate(bra["tuv"]):
                for mm, (tt, uu, vv) in enumerate(ket["tuv"]):
                    rm[m, mm] = (-1.0) ** (tt + uu + vv) * r[t + tt, u + uu, v + vv]
            weight = 2.0 * np.pi ** 2.5 / (p * q * np.sqrt(p + q)) * bra["cc"][:, None] * ket["cc"][None, :]
            block = np.einsum("atx,tsxy,bsy->ab", bra["eh"], rm * weight[None, None], ket["eh"], optimize=True)
            na, nb = len(bra["comps_a"]), len(bra["comps_b"])
            nc, nd = len(ket["comps_a"]), len(ket["comps_b"])
            block = block.reshape(na, nb, nc, nd)
            si = slice(offsets[i], offsets[i] + na)
            sj = slice(offsets[j], offsets[j] + nb)
            sk = slice(offsets[k], offsets[k] + nc)
            sl = slice(offsets[l], offsets[l] + nd)
            for b1, (s1, s2, s3, s4) in ((block, (si, sj, sk, sl)),
                                         (block.transpose(2, 3, 0, 1), (sk, sl, si, sj))):
                eri[s1, s2, s3, s4] = b1
                eri[s2, s1, s3, s4] = b1.transpose(1, 0, 2, 3)
                eri[s1, s2, s4, s3] = b1.transpose(0, 1, 3, 2)
                eri[s2, s1, s4, s3] = b1.transpose(1, 0, 3, 2)
    return eri

import numpy as np


def _symmetric_orthogonalizer(S):
    """Loewdin S^(-1/2); raises ValueError for a non-positive-definite overlap."""
    w, u = np.linalg.eigh(S)
    if w.min() <= 1e-12 * max(1.0, w.max()):
        raise ValueError("overlap matrix is not positive definite")
    return (u / np.sqrt(w)) @ u.T


def _coulomb_exchange(eri, P):
    """Coulomb and exchange matrices J[P], K[P] in the AO basis."""
    J = np.einsum("mnls,ls->mn", eri, P, optimize=True)
    K = np.einsum("mlns,ls->mn", eri, P, optimize=True)
    return J, K


def _diis_extrapolate(focks, errors):
    """Pulay DIIS combination of stored Fock matrices."""
    m = len(focks)
    if m < 2:
        return focks[-1]
    b = -np.ones((m + 1, m + 1))
    b[m, m] = 0.0
    for i in range(m):
        for j in range(m):
            b[i, j] = np.sum(errors[i] * errors[j])
    rhs = np.zeros(m + 1)
    rhs[m] = -1.0
    coef = np.linalg.lstsq(b, rhs, rcond=None)[0][:m]
    return sum(c * f for c, f in zip(coef, focks))


def solve_restricted_hartree_fock(S: "np.ndarray", h: "np.ndarray", eri: "np.ndarray", n_occ: int, e_nuc: float) -> tuple:
    S = np.asarray(S, dtype=float)
    h = np.asarray(h, dtype=float)
    eri = np.asarray(eri, dtype=float)
    nbf = S.shape[0]
    if S.shape != (nbf, nbf) or h.shape != (nbf, nbf) or eri.shape != (nbf,) * 4:
        raise ValueError("inconsistent array shapes")
    if not 1 <= int(n_occ) <= nbf:
        raise ValueError("n_occ must lie in [1, nbf]")
    n_occ = int(n_occ)
    X = _symmetric_orthogonalizer(S)
    eps, ct = np.linalg.eigh(X @ h @ X)
    C = X @ ct
    focks, errors = [], []
    energy_old = None
    for _ in range(500):
        P = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
        J, K = _coulomb_exchange(eri, P)
        F = h + J - 0.5 * K
        energy = 0.5 * np.sum(P * (h + F)) + float(e_nuc)
        err = X @ (F @ P @ S - S @ P @ F) @ X
        if energy_old is not None and abs(energy - energy_old) < 1e-12 and np.abs(err).max() < 1e-10:
            eps, ct = np.linalg.eigh(X @ F @ X)
            return float(energy), eps, X @ ct
        energy_old = energy
        focks.append(F)
        errors.append(err)
        if len(focks) > 8:
            focks.pop(0)
            errors.pop(0)
        eps, ct = np.linalg.eigh(X @ _diis_extrapolate(focks, errors) @ X)
        C = X @ ct
    raise RuntimeError("RHF did not converge")

import numpy as np


def compute_flow_regularized_weight(x: "np.ndarray", y: "np.ndarray", s: float) -> "np.ndarray":
    s = float(s)
    if not np.isfinite(s) or s < 0.0:
        raise ValueError("the flow parameter s must be finite and non-negative")
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    z = s * (x * x + y * y)
    small = z < 1e-10
    z_safe = np.where(small, 1.0, z)
    # (1 - exp(-z)) / z evaluated without cancellation; equals 1 in the limit z -> 0
    phi = np.where(small, 1.0 - 0.5 * z, -np.expm1(-z_safe) / z_safe)
    return (x + y) * s * phi

import numpy as np


def compute_static_self_energy(eps: "np.ndarray", eri_mo: "np.ndarray", n_occ: int, s: float, c_ss: float, c_os: float) -> "np.ndarray":
    eps = np.asarray(eps, dtype=float).ravel()
    g = np.asarray(eri_mo, dtype=float)
    n = eps.size
    if g.shape != (n, n, n, n):
        raise ValueError("eri_mo must have shape (n, n, n, n) with n = len(eps)")
    if not 1 <= int(n_occ) <= n:
        raise ValueError("n_occ must lie in [1, n]")
    n_occ = int(n_occ)
    occ, vir = slice(0, n_occ), slice(n_occ, n)
    e_o, e_v = eps[occ], eps[vir]
    c_same, c_opp = float(c_ss), float(c_os)

    # 2h1p term: A[p, i, a, j] = (pi|aj); exchange partner (qj|ai) = A[q, j, a, i]
    a_h = g[:, occ, vir, occ]
    b_h = (c_same + c_opp) * a_h - c_same * a_h.transpose(0, 3, 2, 1)
    d_h = (eps[:, None, None, None] - e_o[None, :, None, None]
           + e_v[None, None, :, None] - e_o[None, None, None, :])
    w_h = compute_flow_regularized_weight(d_h[:, None], d_h[None, :], s)
    sigma = np.einsum("pqiaj,piaj,qiaj->pq", w_h, a_h, b_h, optimize=True)

    # 2p1h term: A[p, a, i, b] = (pa|ib); exchange partner (qb|ia) = A[q, b, i, a]
    a_p = g[:, vir, occ, vir]
    b_p = (c_same + c_opp) * a_p - c_same * a_p.transpose(0, 3, 2, 1)
    d_p = (eps[:, None, None, None] - e_v[None, :, None, None]
           + e_o[None, None, :, None] - e_v[None, None, None, :])
    w_p = compute_flow_regularized_weight(d_p[:, None], d_p[None, :], s)
    sigma = sigma + np.einsum("pqaib,paib,qaib->pq", w_p, a_p, b_p, optimize=True)
    return 0.5 * (sigma + sigma.T)

import numpy as np


def build_quasiparticle_fock(S: "np.ndarray", h: "np.ndarray", eri: "np.ndarray", C: "np.ndarray", eps: "np.ndarray", n_occ: int, s: float, c_ss: float, c_os: float) -> "np.ndarray":
    S = np.asarray(S, dtype=float)
    h = np.asarray(h, dtype=float)
    eri = np.asarray(eri, dtype=float)
    C = np.asarray(C, dtype=float)
    nbf = S.shape[0]
    if (S.shape != (nbf, nbf) or h.shape != (nbf, nbf) or C.shape != (nbf, nbf)
            or eri.shape != (nbf,) * 4 or np.asarray(eps).size != nbf):
        raise ValueError("inconsistent array shapes")
    if not 1 <= int(n_occ) <= nbf:
        raise ValueError("n_occ must lie in [1, nbf]")
    c_occ = C[:, : int(n_occ)]
    J, K = _coulomb_exchange(eri, 2.0 * c_occ @ c_occ.T)
    eri_mo = np.einsum("mnls,mp,nq,lr,st->pqrt", eri, C, C, C, C, optimize=True)
    sigma = compute_static_self_energy(eps, eri_mo, n_occ, s, c_ss, c_os)
    sc = S @ C
    return h + J - 0.5 * K + sc @ sigma @ sc.T

import numpy as np


def iterate_quasiparticle_self_consistency(S: "np.ndarray", h: "np.ndarray", eri: "np.ndarray", n_occ: int, C0: "np.ndarray", eps0: "np.ndarray", s: float, c_ss: float, c_os: float, conv_tol: float = 1e-9, max_iter: int = 200) -> tuple:
    S = np.asarray(S, dtype=float)
    C = np.array(C0, dtype=float)
    eps = np.array(eps0, dtype=float).ravel()
    X = _symmetric_orthogonalizer(S)
    focks, errors = [], []
    for _ in range(int(max_iter)):
        fock = build_quasiparticle_fock(S, h, eri, C, eps, n_occ, s, c_ss, c_os)
        residual = C.T @ fock @ C - np.diag(eps)
        if np.abs(residual).max() < conv_tol:
            return eps, C
        u = X @ S @ C  # S^(1/2) C: maps the orbital basis to the Loewdin-orthonormal basis
        focks.append(fock)
        errors.append(u @ residual @ u.T)
        if len(focks) > 10:
            focks.pop(0)
            errors.pop(0)
        eps, ct = np.linalg.eigh(X @ _diis_extrapolate(focks, errors) @ X)
        C = X @ ct
    raise RuntimeError("quasiparticle self-consistency did not converge")

import numpy as np


def compute_homo_quasiparticle_energy(coords: "np.ndarray", nuclear_charges: "np.ndarray", shells: list, variant: str, charge: int = 0) -> float:
    # published (c_ss, c_os, s [hartree^-2]) of the source method's three parametrisations
    parameters = {"plain": (1.0, 1.0, 0.525), "scs": (0.6, 1.0, 0.7), "sos": (0.0, 1.0, 1.4)}
    if variant not in parameters:
        raise ValueError("variant must be 'plain', 'scs' or 'sos'")
    c_ss, c_os, s = parameters[variant]
    coords = np.asarray(coords, dtype=float)
    z = np.asarray(nuclear_charges, dtype=float).ravel()
    n_elec = int(round(z.sum())) - int(charge)
    if n_elec <= 0 or n_elec % 2:
        raise ValueError("a closed-shell reference needs a positive, even electron count")
    n_occ = n_elec // 2
    ints = compute_one_electron_integrals(coords, z, shells)
    S, h = ints[0], ints[1] + ints[2]
    if n_occ > S.shape[0]:
        raise ValueError("more occupied orbitals than basis functions")
    eri = compute_electron_repulsion_integrals(coords, shells)
    e_nuc = sum(z[a] * z[b] / np.linalg.norm(coords[a] - coords[b])
                for a in range(len(z)) for b in range(a))
    _, eps0, C0 = solve_restricted_hartree_fock(S, h, eri, n_occ, e_nuc)
    eps, _ = iterate_quasiparticle_self_consistency(
        S, h, eri, n_occ, C0, eps0, s, c_ss, c_os, conv_tol=1e-10, max_iter=300)
    return float(eps[n_occ - 1])
SCICODE_GOLD_EOF
