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


def ppp_hamiltonian(t: float, delta: float, U: float, kappa: float, eps_site: "np.ndarray") -> "np.ndarray":
    """Open PPP chain: h_ii = eps_i - sum_{j!=i} V_ij (neutral +1 background per site),
    h_{i,i+1} = -t(1 + delta(-1)^i), Ohno V_ij = U / sqrt(1 + (U|i-j|/kappa)^2), V_ii = U.
    Returns array (2, K, K): [h, V]."""
    eps_site = np.asarray(eps_site, dtype=float)
    if eps_site.ndim != 1 or eps_site.size < 2:
        raise ValueError("eps_site must be a 1-D array with at least 2 sites")
    t, delta, U, kappa = float(t), float(delta), float(U), float(kappa)
    if t <= 0.0 or U <= 0.0 or kappa <= 0.0 or not (-1.0 < delta < 1.0):
        raise ValueError("need t > 0, U > 0, kappa > 0, |delta| < 1")
    K = eps_site.size
    idx = np.arange(K)
    V = U / np.sqrt(1.0 + (U * np.abs(idx[:, None] - idx[None, :]) / kappa) ** 2)
    h = np.zeros((K, K))
    for i in range(K - 1):
        h[i, i + 1] = h[i + 1, i] = -t * (1.0 + delta * (-1.0) ** i)
    h[idx, idx] = eps_site - (V.sum(axis=1) - np.diag(V))
    return np.stack([h, V])

import numpy as np


def _fix_phase(C):
    """Column sign convention: first component with |c| > 1e-8 positive."""
    C = np.array(C, dtype=float)
    for k in range(C.shape[1]):
        col = C[:, k]
        nz = np.where(np.abs(col) > 1e-8)[0]
        if nz.size and col[nz[0]] < 0.0:
            C[:, k] = -col
    return C


def rhf_orbitals(h: "np.ndarray", V: "np.ndarray", n_pairs: int) -> "np.ndarray":
    """Restricted Hartree-Fock for the density-density Hamiltonian (chemists' (pq|rs) = delta_pq delta_rs V_pr):
    F = h + diag(V D) - (1/2) V o D with D = 2 C_occ C_occ^T. Damped iteration to 1e-12 in D.
    Returns C (K, K), columns sorted by orbital energy, phase-fixed."""
    h = np.asarray(h, dtype=float); V = np.asarray(V, dtype=float); P = int(n_pairs)
    K = h.shape[0]
    if h.shape != (K, K) or V.shape != (K, K) or K < 2:
        raise ValueError("h and V must be (K, K) with K >= 2")
    if not (1 <= P < K):
        raise ValueError("need 1 <= n_pairs < K")
    if not (np.allclose(h, h.T, atol=1e-12) and np.allclose(V, V.T, atol=1e-12)):
        raise ValueError("h and V must be symmetric")
    C = np.linalg.eigh(h)[1]
    D = 2.0 * C[:, :P] @ C[:, :P].T
    for it in range(2000):
        F = h + np.diag(V @ np.diag(D)) - 0.5 * V * D
        eps, C = np.linalg.eigh(F)
        Dn = 2.0 * C[:, :P] @ C[:, :P].T
        if np.max(np.abs(Dn - D)) < 1e-12:
            D = Dn
            break
        D = 0.5 * D + 0.5 * Dn
    else:
        raise ValueError("RHF did not converge")
    F = h + np.diag(V @ np.diag(D)) - 0.5 * V * D
    eps, C = np.linalg.eigh(F)
    return _fix_phase(C)

import numpy as np


def mo_two_electron_integrals(V: "np.ndarray", C: "np.ndarray") -> "np.ndarray":
    """(pq|rs) = sum_ij C_ip C_iq V_ij C_jr C_js in the orbital basis C. Returns (K, K, K, K)."""
    V = np.asarray(V, dtype=float); C = np.asarray(C, dtype=float)
    K = V.shape[0]
    if V.shape != (K, K) or C.shape != (K, K):
        raise ValueError("V and C must be (K, K)")
    X = np.einsum('ip,iq->ipq', C, C)                       # X[i,p,q] = C_ip C_iq
    return np.einsum('ipq,ij,jrs->pqrs', X, V, X, optimize=True)

import numpy as np


def _pair_quantities(hm, Vm):
    """eps_p = 2 h_pp + (pp|pp); W_pq = 2(pp|qq) - (pq|pq) (p != q, W_pp = 0); K_pq = (pq|pq)."""
    J = np.einsum('ppqq->pq', Vm); Kx = np.einsum('pqpq->pq', Vm)
    eps = 2.0 * np.diag(hm) + np.diag(J)
    W = 2.0 * J - Kx
    np.fill_diagonal(W, 0.0)
    return eps, W, Kx


def _pccd_residual_jacobian(c, eps, W, Kx, P):
    """pCCD projected equations R_ia = <Phi_i^a| e^{-T} H e^{T} |Phi_0> (pair Hamiltonian form) and dR/dc."""
    K = eps.size; o = slice(0, P); v = slice(P, K); nv = K - P
    Wo = W[:, o].sum(axis=1)
    D = (eps[v][None, :] - eps[o][:, None]) + 2.0 * ((Wo[v][None, :] - W[v, o].T) - Wo[o][:, None])
    Kov = Kx[o, v]; Kvv = Kx[v, v].copy(); np.fill_diagonal(Kvv, 0.0); Koo = Kx[o, o].copy(); np.fill_diagonal(Koo, 0.0)
    q = (c * Kov).sum(axis=1)[:, None] + (c * Kov).sum(axis=0)[None, :] - c * Kov
    R = Kov + c * D + c @ Kvv + Koo @ c + c @ Kov.T @ c - 2.0 * c * q
    # Jacobian dR_ia/dc_jb
    Jm = np.zeros((P, nv, P, nv))
    I_o = np.eye(P); I_v = np.eye(nv)
    Jm += np.einsum('ij,ab,ia->iajb', I_o, I_v, D - 2.0 * q)
    Jm += np.einsum('ij,ba->iajb', I_o, Kvv)
    Jm += np.einsum('ab,ij->iajb', I_v, Koo)
    Jm += np.einsum('ij,kb,ka->iajb', I_o, Kov, c)          # d/dc_jb of sum_kc c_ic K_kc c_ka, c index = b, i = j
    Jm += np.einsum('ab,ic,jc->iajb', I_v, c, Kov)          # k = j, a = b
    # -2 c_ia dq_ia/dc_jb, dq_ia/dc_jb = delta_ij K_ib + delta_ab K_ja - delta_ij delta_ab K_ia
    Jm -= 2.0 * np.einsum('ia,ij,ib->iajb', c, I_o, Kov)
    Jm -= 2.0 * np.einsum('ia,ab,ja->iajb', c, I_v, Kov)
    Jm += 2.0 * np.einsum('ia,ij,ab,ia->iajb', c, I_o, I_v, Kov)
    return R, Jm.reshape(P * nv, P * nv)


def pccd_amplitudes(hm: "np.ndarray", Vm: "np.ndarray", n_pairs: int) -> "np.ndarray":
    """Solve the pCCD (AP1roG) projected equations by Newton from c = 0 to |R| < 1e-12. Returns c (P, K-P)."""
    hm = np.asarray(hm, dtype=float); Vm = np.asarray(Vm, dtype=float); P = int(n_pairs)
    K = hm.shape[0]
    if hm.shape != (K, K) or Vm.shape != (K, K, K, K) or not (1 <= P < K):
        raise ValueError("bad shapes or n_pairs")
    eps, W, Kx = _pair_quantities(hm, Vm)
    c = np.zeros((P, K - P))
    R, Jm = _pccd_residual_jacobian(c, eps, W, Kx, P)
    for it in range(200):
        if np.max(np.abs(R)) < 1e-12:
            return c
        step = -np.linalg.solve(Jm, R.ravel()).reshape(P, K - P)
        alpha, r0 = 1.0, np.linalg.norm(R)
        for _ in range(40):                                   # backtracking on the residual norm
            cn = c + alpha * step
            Rn, Jn = _pccd_residual_jacobian(cn, eps, W, Kx, P)
            if np.linalg.norm(Rn) < r0:
                break
            alpha *= 0.5
        else:
            raise ValueError("pCCD amplitude equations did not converge")
        c, R, Jm = cn, Rn, Jn
    raise ValueError("pCCD amplitude equations did not converge")

import numpy as np


def _pair_quantities(hm, Vm):
    """eps_p = 2 h_pp + (pp|pp); W_pq = 2(pp|qq) - (pq|pq) (p != q, W_pp = 0); K_pq = (pq|pq)."""
    J = np.einsum('ppqq->pq', Vm); Kx = np.einsum('pqpq->pq', Vm)
    eps = 2.0 * np.diag(hm) + np.diag(J)
    W = 2.0 * J - Kx
    np.fill_diagonal(W, 0.0)
    return eps, W, Kx


def _pccd_residual_jacobian(c, eps, W, Kx, P):
    """pCCD projected equations R_ia = <Phi_i^a| e^{-T} H e^{T} |Phi_0> (pair Hamiltonian form) and dR/dc."""
    K = eps.size; o = slice(0, P); v = slice(P, K); nv = K - P
    Wo = W[:, o].sum(axis=1)
    D = (eps[v][None, :] - eps[o][:, None]) + 2.0 * ((Wo[v][None, :] - W[v, o].T) - Wo[o][:, None])
    Kov = Kx[o, v]; Kvv = Kx[v, v].copy(); np.fill_diagonal(Kvv, 0.0); Koo = Kx[o, o].copy(); np.fill_diagonal(Koo, 0.0)
    q = (c * Kov).sum(axis=1)[:, None] + (c * Kov).sum(axis=0)[None, :] - c * Kov
    R = Kov + c * D + c @ Kvv + Koo @ c + c @ Kov.T @ c - 2.0 * c * q
    # Jacobian dR_ia/dc_jb
    Jm = np.zeros((P, nv, P, nv))
    I_o = np.eye(P); I_v = np.eye(nv)
    Jm += np.einsum('ij,ab,ia->iajb', I_o, I_v, D - 2.0 * q)
    Jm += np.einsum('ij,ba->iajb', I_o, Kvv)
    Jm += np.einsum('ab,ij->iajb', I_v, Koo)
    Jm += np.einsum('ij,kb,ka->iajb', I_o, Kov, c)          # d/dc_jb of sum_kc c_ic K_kc c_ka, c index = b, i = j
    Jm += np.einsum('ab,ic,jc->iajb', I_v, c, Kov)          # k = j, a = b
    # -2 c_ia dq_ia/dc_jb, dq_ia/dc_jb = delta_ij K_ib + delta_ab K_ja - delta_ij delta_ab K_ia
    Jm -= 2.0 * np.einsum('ia,ij,ib->iajb', c, I_o, Kov)
    Jm -= 2.0 * np.einsum('ia,ab,ja->iajb', c, I_v, Kov)
    Jm += 2.0 * np.einsum('ia,ij,ab,ia->iajb', c, I_o, I_v, Kov)
    return R, Jm.reshape(P * nv, P * nv)


def pccd_lambda_amplitudes(hm: "np.ndarray", Vm: "np.ndarray", c: "np.ndarray") -> "np.ndarray":
    """Lambda equations dL/dc_ia = 0 with L = E(c) + sum lambda_jb R_jb(c): J^T lambda = -K_ov. Returns lambda (P, K-P)."""
    hm = np.asarray(hm, dtype=float); Vm = np.asarray(Vm, dtype=float); c = np.asarray(c, dtype=float)
    K = hm.shape[0]; P = c.shape[0]
    if c.shape != (P, K - P) or hm.shape != (K, K) or Vm.shape != (K, K, K, K):
        raise ValueError("bad shapes")
    eps, W, Kx = _pair_quantities(hm, Vm)
    R, Jm = _pccd_residual_jacobian(c, eps, W, Kx, P)
    if np.max(np.abs(R)) > 1e-8:
        raise ValueError("amplitudes do not satisfy the pCCD equations")
    lam = np.linalg.solve(Jm.T, -Kx[:P, P:].ravel())
    return lam.reshape(P, K - P)

import numpy as np


def pccd_response_rdms(c: "np.ndarray", lam: "np.ndarray") -> "np.ndarray":
    """Response 1- and 2-RDMs of pCCD (per spin), Eqs 13-19 of the source with Eq 17 corrected.
    Returns (3, K, K): [0] diag(gamma_p); [1] Gamma^{p qbar}_{p qbar} (= Gamma^{pq}_{pq}; diagonal = gamma_p);
    [2] Gamma^{p pbar}_{q qbar} (pair-transfer block)."""
    c = np.asarray(c, dtype=float); lam = np.asarray(lam, dtype=float)
    if c.ndim != 2 or c.shape != lam.shape:
        raise ValueError("c and lambda must be 2-D arrays of the same shape")
    P, nv = c.shape; K = P + nv
    s_i = (c * lam).sum(axis=1); s_a = (c * lam).sum(axis=0)
    gam = np.concatenate([1.0 - s_i, s_a])
    G = np.zeros((K, K)); Gp = np.zeros((K, K))
    o = slice(0, P); v = slice(P, K)
    G[o, o] = 1.0 - s_i[:, None] - s_i[None, :]
    G[o, v] = s_a[None, :] - lam * c
    G[v, o] = G[o, v].T
    G[v, v] = 0.0
    np.fill_diagonal(G, gam)
    Gp[o, o] = c @ lam.T                                     # sum_c lambda_jc c_ic  -> [i, j]
    Gp[o, o] += np.diag(1.0 - 2.0 * s_i)
    Gp[o, v] = (c + 2.0 * lam * c * c - 2.0 * s_a[None, :] * c - 2.0 * s_i[:, None] * c + c @ lam.T @ c)
    Gp[v, o] = lam.T
    Gp[v, v] = lam.T @ c
    return np.stack([np.diag(gam), G, Gp])

import numpy as np


def generalized_fock(hm: "np.ndarray", Vm: "np.ndarray", rdms: "np.ndarray") -> "np.ndarray":
    """Eq 22: F_pq = h_pq gamma_q + sum_r [ <qr||pr> G_qr + <q rbar|p rbar> G_qr + <p qbar|r rbar> Gt_qr ],
    physicists' <pq|rs> = (pr|qs), Gt = (Gp + Gp^T)/2. Returns F (K, K)."""
    hm = np.asarray(hm, dtype=float); Vm = np.asarray(Vm, dtype=float); rdms = np.asarray(rdms, dtype=float)
    K = hm.shape[0]
    if rdms.shape != (3, K, K) or Vm.shape != (K, K, K, K):
        raise ValueError("bad shapes")
    gam = np.diag(rdms[0]); G = rdms[1].copy(); Gp = rdms[2]
    np.fill_diagonal(G, 0.0)
    Gt = 0.5 * (Gp + Gp.T)
    phys = np.einsum('prqs->pqrs', Vm)
    anti = phys - np.einsum('pqrs->pqsr', phys)
    F = hm * gam[None, :]
    F += np.einsum('qrpr,qr->pq', anti, G) + np.einsum('qrpr,qr->pq', phys, G)
    F += np.einsum('pqrr,qr->pq', phys, Gt)
    return F

import numpy as np


def _fix_phase(C):
    """Column sign convention: first component with |c| > 1e-8 positive."""
    C = np.array(C, dtype=float)
    for k in range(C.shape[1]):
        col = C[:, k]
        nz = np.where(np.abs(col) > 1e-8)[0]
        if nz.size and col[nz[0]] < 0.0:
            C[:, k] = -col
    return C


def _pair_quantities(hm, Vm):
    """eps_p = 2 h_pp + (pp|pp); W_pq = 2(pp|qq) - (pq|pq) (p != q, W_pp = 0); K_pq = (pq|pq)."""
    J = np.einsum('ppqq->pq', Vm); Kx = np.einsum('pqpq->pq', Vm)
    eps = 2.0 * np.diag(hm) + np.diag(J)
    W = 2.0 * J - Kx
    np.fill_diagonal(W, 0.0)
    return eps, W, Kx


def _pccd_gfm_at(h, V, C, P):
    hm = C.T @ h @ C
    Vm = mo_two_electron_integrals(V, C)
    c = pccd_amplitudes(hm, Vm, P)
    lam = pccd_lambda_amplitudes(hm, Vm, c)
    rd = pccd_response_rdms(c, lam)
    F = generalized_fock(hm, Vm, rd)
    eps, W, Kx = _pair_quantities(hm, Vm)
    E = eps[:P].sum() + W[:P, :P].sum() + (c * Kx[:P, P:]).sum()
    return E, F, np.diag(rd[0]), C


def _rotate(C, kv, iu):
    K = C.shape[0]; kap = np.zeros((K, K)); kap[iu] = kv; kap = kap - kap.T
    w, U = np.linalg.eigh(1j * kap)                          # exp(kap) via Hermitian i*kap
    R = (U * np.exp(1j * w)) @ U.conj().T                   # exp(-kap): kap = -i U diag(w) U^dag
    return C @ np.real(R)


def oo_pccd_orbitals(h: "np.ndarray", V: "np.ndarray", C0: "np.ndarray", n_pairs: int) -> "np.ndarray":
    """Variational orbital optimisation of pCCD: minimise the pCCD energy over all orbital rotations
    C -> C exp(-kappa); gradient 4(F^T - F) from the generalised Fock matrix, Newton steps with a
    finite-difference Hessian of that gradient, converged to max|g| < 1e-10. Returns the natural
    orbitals (columns sorted by occupation number descending, phase-fixed)."""
    h = np.asarray(h, dtype=float); V = np.asarray(V, dtype=float); C = np.array(C0, dtype=float); P = int(n_pairs)
    K = h.shape[0]
    if C.shape != (K, K) or V.shape != (K, K) or not (1 <= P < K):
        raise ValueError("bad shapes or n_pairs")
    iu = np.triu_indices(K, 1); n = iu[0].size

    def _grad_at(Cx):
        E, F, gam, _ = _pccd_gfm_at(h, V, Cx, P)
        return E, (4.0 * (F.T - F))[iu], gam

    E, g, gam = _grad_at(C)
    for it in range(100):
        if np.max(np.abs(g)) < 1e-10:
            break
        Hs = np.zeros((n, n)); dk = 1e-4
        for m in range(n):
            e = np.zeros(n); e[m] = dk
            Hs[:, m] = (_grad_at(_rotate(C, e, iu))[1] - _grad_at(_rotate(C, -e, iu))[1]) / (2.0 * dk)
        Hs = 0.5 * (Hs + Hs.T)
        w, Uh = np.linalg.eigh(Hs)
        w = np.where(w > 1e-6, w, np.maximum(np.abs(w), 1e-6))    # positive-definite shift for descent
        step = -Uh @ ((Uh.T @ g) / w)
        smax = np.max(np.abs(step))
        if smax > 0.5:                                        # trust region on the rotation angle
            step *= 0.5 / smax
        alpha = 1.0
        for _ in range(30):
            Cn = _rotate(C, alpha * step, iu)
            try:
                En, gn, gamn = _grad_at(Cn)
            except ValueError:
                alpha *= 0.5
                continue
            if En <= E + 1e-12:
                break
            alpha *= 0.5
        else:
            raise ValueError("oo-pCCD line search failed")
        C, E, g, gam = Cn, En, gn, gamn
    else:
        raise ValueError("oo-pCCD did not converge")
    order = np.argsort(-gam)
    return _fix_phase(C[:, order])

import numpy as np


def first_ionization_potential(t: float, delta: float, U: float, kappa: float, eps_site: "np.ndarray", cutoff: float) -> float:
    """EKT(oo-pCCD) first ionisation potential of the PPP chain at half filling (K sites, K electrons, K even).
    Steps: model -> RHF -> oo-pCCD natural orbitals -> amplitudes, Lambda, response RDMs, generalised Fock
    -> drop natural orbitals with occupation below cutoff -> F' = gamma^{-1/2} F gamma^{-1/2} (F symmetrised)
    -> eigenvalues e; the ionisation potentials are -e for eigenvectors with more than half their weight on
    the P strongly occupied natural orbitals; return the smallest of them."""
    eps_site = np.asarray(eps_site, dtype=float); K = eps_site.size
    if K % 2 or K < 4:
        raise ValueError("need an even number of sites, at least 4")
    cutoff = float(cutoff)
    if not (0.0 <= cutoff < 1.0):
        raise ValueError("cutoff must lie in [0, 1)")
    P = K // 2
    hV = ppp_hamiltonian(t, delta, U, kappa, eps_site); h, V = hV[0], hV[1]
    C0 = rhf_orbitals(h, V, P)
    C = oo_pccd_orbitals(h, V, C0, P)
    hm = C.T @ h @ C
    Vm = mo_two_electron_integrals(V, C)
    c = pccd_amplitudes(hm, Vm, P)
    lam = pccd_lambda_amplitudes(hm, Vm, c)
    rd = pccd_response_rdms(c, lam)
    F = generalized_fock(hm, Vm, rd)
    gam = np.diag(rd[0])
    keep = np.where(gam > cutoff)[0]
    Fk = 0.5 * (F[np.ix_(keep, keep)] + F[np.ix_(keep, keep)].T); gk = gam[keep]
    Fp = Fk / np.sqrt(gk)[:, None] / np.sqrt(gk)[None, :]
    e, Cp = np.linalg.eigh(Fp)
    occ_idx = np.where(keep < P)[0]
    weight = (Cp[occ_idx, :] ** 2).sum(axis=0)
    phys = e[weight > 0.5]
    if phys.size == 0:
        raise ValueError("no eigenvector with dominant occupied character")
    return float(-phys.max())
SCICODE_GOLD_EOF
