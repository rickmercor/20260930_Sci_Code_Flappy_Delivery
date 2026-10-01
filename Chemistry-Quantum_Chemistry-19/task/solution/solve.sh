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


def ppp_fock_matrix(bond_lengths, hoppings, site_energies, hubbard_u, n_occ):
    lengths = np.asarray(bond_lengths, dtype=float)
    t = np.asarray(hoppings, dtype=float)
    eps_site = np.asarray(site_energies, dtype=float)
    n = eps_site.shape[0]
    if eps_site.ndim != 1 or n < 2:
        raise ValueError("site_energies must be a vector with at least two entries")
    if lengths.shape != (n - 1,) or t.shape != (n - 1,):
        raise ValueError("bond_lengths and hoppings must have n - 1 entries")
    if not (np.all(np.isfinite(lengths)) and np.all(np.isfinite(t)) and np.all(np.isfinite(eps_site))):
        raise ValueError("inputs must be finite")
    if np.any(lengths <= 0.0):
        raise ValueError("bond lengths must be positive")
    if not np.isfinite(hubbard_u) or hubbard_u <= 0.0:
        raise ValueError("hubbard_u must be positive")
    if int(n_occ) != n_occ or n_occ < 1 or n_occ > n:
        raise ValueError("n_occ must be an integer between 1 and n")
    n_occ = int(n_occ)
    e2 = 14.397
    pos = np.concatenate([[0.0], np.cumsum(lengths)])
    r = np.abs(pos[:, None] - pos[None, :])
    V = e2 / np.sqrt((e2 / hubbard_u) ** 2 + r ** 2)
    h = np.zeros((n, n))
    for k in range(n - 1):
        h[k, k + 1] = t[k]
        h[k + 1, k] = t[k]
    for m in range(n):
        h[m, m] = eps_site[m] - (np.sum(V[m]) - V[m, m])
    _, C = np.linalg.eigh(h)
    P = 2.0 * np.dot(C[:, :n_occ], C[:, :n_occ].T)
    for _ in range(2000):
        F = h + np.diag(np.dot(V, np.diag(P))) - 0.5 * P * V
        _, C = np.linalg.eigh(F)
        P_new = 2.0 * np.dot(C[:, :n_occ], C[:, :n_occ].T)
        if np.max(np.abs(P_new - P)) < 1e-12:
            P = P_new
            break
        P = 0.5 * (P + P_new)
    F = h + np.diag(np.dot(V, np.diag(P))) - 0.5 * P * V
    return F

import numpy as np


def mo_eri_tensor(fock, ohno):
    F = np.asarray(fock, dtype=float)
    V = np.asarray(ohno, dtype=float)
    if F.ndim != 2 or F.shape[0] != F.shape[1] or F.shape != V.shape or F.shape[0] < 1:
        raise ValueError("fock and ohno must be square matrices of the same size")
    if not (np.all(np.isfinite(F)) and np.all(np.isfinite(V))):
        raise ValueError("fock and ohno must be finite")
    if np.max(np.abs(F - F.T)) > 1e-10 or np.max(np.abs(V - V.T)) > 1e-10:
        raise ValueError("fock and ohno must be symmetric")
    _, C = np.linalg.eigh(F)
    C = C.copy()
    for p in range(C.shape[1]):
        k = np.argmax(np.abs(C[:, p]))
        if C[k, p] < 0.0:
            C[:, p] *= -1.0
    return np.einsum("mp,nq,mr,ns,mn->pqrs", C, C, C, C, V, optimize=True)

import numpy as np


def _drpa_pairs(eps, eri, n_occ):
    n = eps.shape[0]
    pairs = [(i, a) for i in range(n_occ) for a in range(n_occ, n)]
    nov = len(pairs)
    A = np.zeros((nov, nov))
    B = np.zeros((nov, nov))
    for k, (i, a) in enumerate(pairs):
        for l, (j, b) in enumerate(pairs):
            A[k, l] = (eps[a] - eps[i]) * (1.0 if k == l else 0.0) + 2.0 * eri[a, j, i, b]
            B[k, l] = 2.0 * eri[a, b, i, j]
    return pairs, A, B


def drpa_excitation_energies(orbital_energies, eri, n_occ):
    eps = np.asarray(orbital_energies, dtype=float)
    g = np.asarray(eri, dtype=float)
    n = eps.shape[0]
    if eps.ndim != 1 or n < 2 or g.shape != (n, n, n, n):
        raise ValueError("orbital_energies must have n entries and eri shape (n, n, n, n)")
    if not (np.all(np.isfinite(eps)) and np.all(np.isfinite(g))):
        raise ValueError("inputs must be finite")
    if int(n_occ) != n_occ or n_occ < 1 or n_occ > n - 1:
        raise ValueError("n_occ must be an integer between 1 and n - 1")
    n_occ = int(n_occ)
    _, A, B = _drpa_pairs(eps, g, n_occ)
    AmB = A - B
    w_amb, U = np.linalg.eigh(0.5 * (AmB + AmB.T))
    if np.any(w_amb <= 0.0):
        raise ValueError("A - B is not positive definite")
    S = np.dot(U * np.sqrt(w_amb), U.T)
    w2 = np.linalg.eigvalsh(np.linalg.multi_dot([S, A + B, S]))
    if np.any(w2 <= 0.0):
        raise ValueError("dRPA problem has no real positive spectrum")
    return np.sort(np.sqrt(w2))

import numpy as np


def gw_effective_integrals(orbital_energies, eri, n_occ):
    eps = np.asarray(orbital_energies, dtype=float)
    g = np.asarray(eri, dtype=float)
    n = eps.shape[0]
    if eps.ndim != 1 or n < 2 or g.shape != (n, n, n, n):
        raise ValueError("orbital_energies must have n entries and eri shape (n, n, n, n)")
    if not (np.all(np.isfinite(eps)) and np.all(np.isfinite(g))):
        raise ValueError("inputs must be finite")
    if int(n_occ) != n_occ or n_occ < 1 or n_occ > n - 1:
        raise ValueError("n_occ must be an integer between 1 and n - 1")
    n_occ = int(n_occ)
    pairs = [(i, a) for i in range(n_occ) for a in range(n_occ, n)]
    nov = len(pairs)
    A = np.zeros((nov, nov))
    B = np.zeros((nov, nov))
    for k, (i, a) in enumerate(pairs):
        for l, (j, b) in enumerate(pairs):
            A[k, l] = (eps[a] - eps[i]) * (1.0 if k == l else 0.0) + 2.0 * g[a, j, i, b]
            B[k, l] = 2.0 * g[a, b, i, j]
    AmB = A - B
    w_amb, U = np.linalg.eigh(0.5 * (AmB + AmB.T))
    if np.any(w_amb <= 0.0):
        raise ValueError("A - B is not positive definite")
    S = np.dot(U * np.sqrt(w_amb), U.T)
    Sinv = np.dot(U / np.sqrt(w_amb), U.T)
    w2, Z = np.linalg.eigh(np.linalg.multi_dot([S, A + B, S]))
    if np.any(w2 <= 0.0):
        raise ValueError("dRPA problem has no real positive spectrum")
    order = np.argsort(w2)
    w2 = w2[order]
    Z = Z[:, order]
    Om = np.sqrt(w2)
    XpY = np.dot(S, Z) / np.sqrt(Om)
    XmY = np.dot(Sinv, Z) * np.sqrt(Om)
    for v in range(nov):
        k = np.argmax(np.abs(XpY[:, v]))
        if XpY[k, v] < 0.0:
            XpY[:, v] *= -1.0
            XmY[:, v] *= -1.0
    X = 0.5 * (XpY + XmY)
    Y = 0.5 * (XpY - XmY)
    M = np.zeros((n, n, nov))
    for k, (i, a) in enumerate(pairs):
        M += np.sqrt(2.0) * (g[:, a, :, i][:, :, None] * X[k][None, None, :] + g[:, i, :, a][:, :, None] * Y[k][None, None, :])
    return M

import numpy as np


def _check_block_inputs_sosex(orbital_energies, M, omega, n_occ, branch, eri=None):
    eps = np.asarray(orbital_energies, dtype=float)
    Mm = np.asarray(M, dtype=float)
    Om = np.asarray(omega, dtype=float)
    n = eps.shape[0]
    if eps.ndim != 1 or n < 2:
        raise ValueError("orbital_energies must be a vector with at least two entries")
    if Om.ndim != 1 or Mm.shape != (n, n, Om.shape[0]) or Om.shape[0] < 1:
        raise ValueError("M must have shape (n, n, n_modes) and omega n_modes entries")
    if eri is not None and np.asarray(eri).shape != (n, n, n, n):
        raise ValueError("eri must have shape (n, n, n, n)")
    if not (np.all(np.isfinite(eps)) and np.all(np.isfinite(Mm)) and np.all(np.isfinite(Om))):
        raise ValueError("inputs must be finite")
    if eri is not None and not np.all(np.isfinite(np.asarray(eri, dtype=float))):
        raise ValueError("inputs must be finite")
    if int(n_occ) != n_occ or n_occ < 1 or n_occ > n - 1:
        raise ValueError("n_occ must be an integer between 1 and n - 1")
    if branch not in ("hole", "particle"):
        raise ValueError("branch must be 'hole' or 'particle'")
    return eps, Mm, Om, int(n_occ)


def sosex_coupling_block(orbital_energies, eri, M, omega, n_occ, branch):
    eps, M, Om, n_occ = _check_block_inputs_sosex(orbital_energies, M, omega, n_occ, branch, eri=eri)
    g = np.asarray(eri, dtype=float)
    n = eps.shape[0]
    occ = range(n_occ)
    vir = range(n_occ, n)
    rows = list(occ) if branch == "hole" else list(vir)
    nov = Om.shape[0]
    U = np.zeros((len(rows) * nov, n))
    r = 0
    for p in rows:
        for v in range(nov):
            for q in range(n):
                s = 0.0
                for k in occ:
                    for c in vir:
                        if branch == "hole":
                            s += M[c, k, v] * g[p, c, k, q] / (eps[c] - eps[k] + Om[v])
                            s += M[k, c, v] * g[p, k, c, q] / (eps[c] - eps[k] - Om[v])
                        else:
                            s += M[c, k, v] * g[p, k, c, q] / (eps[c] - eps[k] + Om[v])
                            s += M[k, c, v] * g[p, c, k, q] / (eps[c] - eps[k] - Om[v])
                U[r, q] = s
            r += 1
    return U

import numpy as np


def _check_block_inputs(orbital_energies, M, omega, n_occ, branch):
    eps = np.asarray(orbital_energies, dtype=float)
    Mm = np.asarray(M, dtype=float)
    Om = np.asarray(omega, dtype=float)
    n = eps.shape[0]
    if eps.ndim != 1 or n < 2:
        raise ValueError("orbital_energies must be a vector with at least two entries")
    if Om.ndim != 1 or Mm.shape != (n, n, Om.shape[0]) or Om.shape[0] < 1:
        raise ValueError("M must have shape (n, n, n_modes) and omega n_modes entries")
    if not (np.all(np.isfinite(eps)) and np.all(np.isfinite(Mm)) and np.all(np.isfinite(Om))):
        raise ValueError("inputs must be finite")
    if int(n_occ) != n_occ or n_occ < 1 or n_occ > n - 1:
        raise ValueError("n_occ must be an integer between 1 and n - 1")
    if branch not in ("hole", "particle"):
        raise ValueError("branch must be 'hole' or 'particle'")
    return eps, Mm, Om, int(n_occ)


def adc3_diagonal_block(orbital_energies, M, omega, n_occ, branch):
    eps, M, Om, n_occ = _check_block_inputs(orbital_energies, M, omega, n_occ, branch)
    n = eps.shape[0]
    occ = list(range(n_occ))
    vir = list(range(n_occ, n))
    rows = occ if branch == "hole" else vir
    other = vir if branch == "hole" else occ
    nov = Om.shape[0]
    conf = [(p, v) for p in rows for v in range(nov)]
    C = np.zeros((len(conf), len(conf)))
    for r, (i, v) in enumerate(conf):
        for s_, (j, mu) in enumerate(conf):
            val = 0.0
            for c in other:
                if branch == "hole":
                    prod = M[i, c, mu] * M[j, c, v]
                    val += 0.5 * prod / (eps[i] - eps[c] + Om[mu])
                    val += 0.5 * prod / (eps[j] - eps[c] + Om[v])
                else:
                    prod = M[c, i, mu] * M[c, j, v]
                    val += 0.5 * prod / (eps[i] - eps[c] - Om[mu])
                    val += 0.5 * prod / (eps[j] - eps[c] - Om[v])
            C[r, s_] = val
    return C

import numpy as np


def _check_block_inputs(orbital_energies, M, omega, n_occ, branch):
    eps = np.asarray(orbital_energies, dtype=float)
    Mm = np.asarray(M, dtype=float)
    Om = np.asarray(omega, dtype=float)
    n = eps.shape[0]
    if eps.ndim != 1 or n < 2:
        raise ValueError("orbital_energies must be a vector with at least two entries")
    if Om.ndim != 1 or Mm.shape != (n, n, Om.shape[0]) or Om.shape[0] < 1:
        raise ValueError("M must have shape (n, n, n_modes) and omega n_modes entries")
    if not (np.all(np.isfinite(eps)) and np.all(np.isfinite(Mm)) and np.all(np.isfinite(Om))):
        raise ValueError("inputs must be finite")
    if int(n_occ) != n_occ or n_occ < 1 or n_occ > n - 1:
        raise ValueError("n_occ must be an integer between 1 and n - 1")
    if branch not in ("hole", "particle"):
        raise ValueError("branch must be 'hole' or 'particle'")
    return eps, Mm, Om, int(n_occ)


def third_order_coupling_block(orbital_energies, M, omega, n_occ, branch):
    eps, M, Om, n_occ = _check_block_inputs(orbital_energies, M, omega, n_occ, branch)
    n = eps.shape[0]
    occ = range(n_occ)
    vir = range(n_occ, n)
    rows = list(occ) if branch == "hole" else list(vir)
    nov = Om.shape[0]
    U = np.zeros((len(rows) * nov, n))
    r = 0
    for p in rows:
        for v in range(nov):
            for q in range(n):
                s = 0.0
                if branch == "hole":
                    i = p
                    for k in occ:
                        for c in vir:
                            for mu in range(nov):
                                s += 0.5 * M[i, c, mu] * M[k, c, v] * M[q, k, mu] / ((eps[c] - eps[k] - Om[v]) * (eps[c] - eps[i] - Om[mu]))
                                s -= M[c, i, mu] * M[k, c, v] * M[k, q, mu] / ((eps[c] - eps[k] - Om[v]) * (eps[c] - eps[i] + Om[mu]))
                                s -= M[k, i, mu] * M[c, k, v] * M[c, q, mu] / ((eps[c] - eps[i] + Om[v] + Om[mu]) * (eps[c] - eps[k] + Om[v]))
                    for c in vir:
                        for d in vir:
                            for mu in range(nov):
                                s += M[d, i, mu] * M[c, d, v] * M[c, q, mu] / ((eps[c] - eps[i] + Om[v] + Om[mu]) * (eps[d] - eps[i] + Om[mu]))
                else:
                    a = p
                    for k in occ:
                        for c in vir:
                            for mu in range(nov):
                                s += 0.5 * M[k, a, mu] * M[k, c, v] * M[c, q, mu] / ((eps[c] - eps[k] - Om[v]) * (eps[a] - eps[k] - Om[mu]))
                                s -= M[a, k, mu] * M[k, c, v] * M[q, c, mu] / ((eps[c] - eps[k] - Om[v]) * (eps[a] - eps[k] + Om[mu]))
                                s -= M[a, c, mu] * M[c, k, v] * M[q, k, mu] / ((eps[a] - eps[k] + Om[v] + Om[mu]) * (eps[c] - eps[k] + Om[v]))
                    for k in occ:
                        for l in occ:
                            for mu in range(nov):
                                s += M[a, l, mu] * M[l, k, v] * M[q, k, mu] / ((eps[a] - eps[k] + Om[v] + Om[mu]) * (eps[a] - eps[l] + Om[mu]))
                U[r, q] = s
            r += 1
    return U

import numpy as np


def _check_block_inputs(orbital_energies, M, omega, n_occ, branch):
    eps = np.asarray(orbital_energies, dtype=float)
    Mm = np.asarray(M, dtype=float)
    Om = np.asarray(omega, dtype=float)
    n = eps.shape[0]
    if eps.ndim != 1 or n < 2:
        raise ValueError("orbital_energies must be a vector with at least two entries")
    if Om.ndim != 1 or Mm.shape != (n, n, Om.shape[0]) or Om.shape[0] < 1:
        raise ValueError("M must have shape (n, n, n_modes) and omega n_modes entries")
    if not (np.all(np.isfinite(eps)) and np.all(np.isfinite(Mm)) and np.all(np.isfinite(Om))):
        raise ValueError("inputs must be finite")
    if int(n_occ) != n_occ or n_occ < 1 or n_occ > n - 1:
        raise ValueError("n_occ must be an integer between 1 and n - 1")
    if branch not in ("hole", "particle"):
        raise ValueError("branch must be 'hole' or 'particle'")
    return eps, Mm, Om, int(n_occ)


def three_body_row_block(orbital_energies, M, omega, n_occ, branch):
    eps, M, Om, n_occ = _check_block_inputs(orbital_energies, M, omega, n_occ, branch)
    n = eps.shape[0]
    occ = list(range(n_occ))
    vir = list(range(n_occ, n))
    rows = occ if branch == "hole" else vir
    other = vir if branch == "hole" else occ
    nov = Om.shape[0]
    conf3 = [(p, v, mu) for p in rows for v in range(nov) for mu in range(nov)]
    conf2 = [(p, v) for p in rows for v in range(nov)]
    U2 = np.zeros((len(conf3), n))
    C21 = np.zeros((len(conf3), len(conf2)))
    K2 = np.zeros((len(conf3), len(conf3)))
    sgn = -1.0 if branch == "hole" else 1.0
    for r, (p, v, mu) in enumerate(conf3):
        K2[r, r] = eps[p] + sgn * (Om[v] + Om[mu])
        for q in range(n):
            val = 0.0
            for c in other:
                if branch == "hole":
                    val -= M[c, p, mu] * M[q, c, v] / (eps[c] - eps[p] + Om[mu])
                else:
                    val += M[p, c, mu] * M[c, q, v] / (eps[p] - eps[c] + Om[mu])
            U2[r, q] = val
        for s_, (j, lam) in enumerate(conf2):
            if lam == v:
                C21[r, s_] = M[j, p, mu] if branch == "hole" else M[p, j, mu]
    return np.hstack([U2, C21, K2])

import numpy as np


def _principal_ip(H, orbital):
    w, v = np.linalg.eigh(H)
    k = int(np.argmax(v[orbital, :] ** 2))
    return -w[k]


def _assemble(f, blocks):
    """blocks[branch] = (U1, D1, U2, C21, K2); returns the symmetric Hamiltonian."""
    n = f.shape[0]
    Um, Dm, U2m, C21m, K2m = blocks["hole"]
    Up, Dp, U2p, C21p, K2p = blocks["particle"]
    dm, d2m, dp, d2p = Dm.shape[0], K2m.shape[0], Dp.shape[0], K2p.shape[0]
    o = np.cumsum([0, n, dm, d2m, dp, d2p])
    H = np.zeros((o[-1], o[-1]))
    H[o[0]:o[1], o[0]:o[1]] = f
    H[o[1]:o[2], o[0]:o[1]] = Um
    H[o[1]:o[2], o[1]:o[2]] = Dm
    H[o[2]:o[3], o[0]:o[1]] = U2m
    H[o[2]:o[3], o[1]:o[2]] = C21m
    H[o[2]:o[3], o[2]:o[3]] = K2m
    H[o[3]:o[4], o[0]:o[1]] = Up
    H[o[3]:o[4], o[3]:o[4]] = Dp
    H[o[4]:o[5], o[0]:o[1]] = U2p
    H[o[4]:o[5], o[3]:o[4]] = C21p
    H[o[4]:o[5], o[4]:o[5]] = K2p
    return np.tril(H) + np.tril(H, -1).T


def adc_g3w2_vertex_correction(bond_lengths, hoppings, site_energies, hubbard_u, n_occ):
    eps_site = np.asarray(site_energies, dtype=float)
    n = eps_site.shape[0]
    if int(n_occ) != n_occ or n_occ < 1 or n_occ > n - 1:
        raise ValueError("n_occ must be an integer between 1 and n - 1")
    n_occ = int(n_occ)
    lengths = np.asarray(bond_lengths, dtype=float)
    if lengths.shape != (n - 1,) or np.any(lengths <= 0.0):
        raise ValueError("bond_lengths must have n - 1 positive entries")
    e2 = 14.397
    pos = np.concatenate([[0.0], np.cumsum(lengths)])
    r = np.abs(pos[:, None] - pos[None, :])
    V = e2 / np.sqrt((e2 / float(hubbard_u)) ** 2 + r ** 2)
    F = ppp_fock_matrix(bond_lengths, hoppings, site_energies, hubbard_u, n_occ)
    eps = np.linalg.eigvalsh(F)
    eri = mo_eri_tensor(F, V)
    Om = drpa_excitation_energies(eps, eri, n_occ)
    M = gw_effective_integrals(eps, eri, n_occ)
    nov = Om.shape[0]
    f = np.diag(eps)
    gw = {}
    full = {}
    for branch in ("hole", "particle"):
        rows = list(range(n_occ)) if branch == "hole" else list(range(n_occ, n))
        sgn = -1.0 if branch == "hole" else 1.0
        K1 = np.diag([eps[p] + sgn * Om[v] for p in rows for v in range(nov)])
        if branch == "hole":
            U1 = np.array([[M[q, i, v] for q in range(n)] for i in rows for v in range(nov)])
        else:
            U1 = np.array([[M[a, q, v] for q in range(n)] for a in rows for v in range(nov)])
        d1 = K1.shape[0]
        gw[branch] = (U1, K1, np.zeros((0, n)), np.zeros((0, d1)), np.zeros((0, 0)))
        U1_full = (U1 + sosex_coupling_block(eps, eri, M, Om, n_occ, branch)
                   + third_order_coupling_block(eps, M, Om, n_occ, branch))
        D1 = K1 + adc3_diagonal_block(eps, M, Om, n_occ, branch)
        row = three_body_row_block(eps, M, Om, n_occ, branch)
        U2 = row[:, :n]
        C21 = row[:, n:n + d1]
        K2 = row[:, n + d1:]
        full[branch] = (U1_full, D1, U2, C21, K2)
    homo = n_occ - 1
    ip_gw = _principal_ip(_assemble(f, gw), homo)
    ip_full = _principal_ip(_assemble(f, full), homo)
    return float(ip_full - ip_gw)
SCICODE_GOLD_EOF
