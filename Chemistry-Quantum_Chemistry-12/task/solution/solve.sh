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
    """Open PPP chain of K sites: h_ii = eps_i - sum_{j != i} V_ij (a +1 neutralising background on every site),
    h_{i,i+1} = h_{i+1,i} = -t (1 + delta (-1)^i) (0-based i), Ohno V_ij = U / sqrt(1 + (U |i - j| / kappa)^2),
    V_ii = U.  Returns array (2, K, K) = [h, V]."""
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
    """Column sign convention: the first component with |c| > 1e-8 is positive."""
    C = np.array(C, dtype=float)
    for k in range(C.shape[1]):
        nz = np.where(np.abs(C[:, k]) > 1e-8)[0]
        if nz.size and C[nz[0], k] < 0.0:
            C[:, k] = -C[:, k]
    return C


def rhf_reference(h: "np.ndarray", V: "np.ndarray", n_pairs: int) -> "np.ndarray":
    """Restricted closed-shell Hartree-Fock for the density-density Hamiltonian ((pq|rs) = delta_pq delta_rs V_pr
    in the site basis): F = h + diag(V D) - (1/2) V o D with D = 2 C_occ C_occ^T, started from the eigenvectors of
    h, damped iteration D <- (D + D_new)/2 until max|D_new - D| < 1e-12, then one final diagonalisation of F.
    Returns array (K + 1, K): row 0 = orbital energies ascending, rows 1..K = the orbital coefficient matrix C
    (columns = orbitals, phase-fixed so that the first component with |c| > 1e-8 is positive)."""
    h = np.asarray(h, dtype=float)
    V = np.asarray(V, dtype=float)
    P = int(n_pairs)
    K = h.shape[0]
    if h.shape != (K, K) or V.shape != (K, K) or K < 2:
        raise ValueError("h and V must be (K, K) with K >= 2")
    if not (1 <= P < K):
        raise ValueError("need 1 <= n_pairs < K")
    if not (np.allclose(h, h.T, atol=1e-12) and np.allclose(V, V.T, atol=1e-12)):
        raise ValueError("h and V must be symmetric")
    C = np.linalg.eigh(h)[1]
    D = 2.0 * C[:, :P] @ C[:, :P].T
    for _ in range(5000):
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
    return np.vstack([eps[None, :], _fix_phase(C)])

import numpy as np


def mo_two_electron_integrals(V: "np.ndarray", C: "np.ndarray") -> "np.ndarray":
    """Chemists' integrals (pq|rs) = sum_ij C_ip C_iq V_ij C_jr C_js in the orbital basis C. Returns (K, K, K, K)."""
    V = np.asarray(V, dtype=float)
    C = np.asarray(C, dtype=float)
    K = V.shape[0]
    if V.shape != (K, K) or C.shape != (K, K):
        raise ValueError("V and C must be (K, K)")
    X = np.einsum('ip,iq->ipq', C, C)
    return np.einsum('ipq,ij,jrs->pqrs', X, V, X, optimize=True)

import numpy as np


def static_screened_interaction(eps: "np.ndarray", eri: "np.ndarray", n_occ: int) -> "np.ndarray":
    """Statically screened interaction of the direct RPA in the orbital basis:
    W0_{pq,rs} = (pq|rs) + sum_{ia,jb} (pq|ia) chi_{ia,jb} (jb|rs), chi = (1 - chi0 J)^{-1} chi0 with
    chi0_{ia,jb} = -4 delta_ij delta_ab / (e_a - e_i) (spin-summed, real orbitals) and J_{ia,jb} = (ia|jb).
    Returns W0 with the same index layout as eri, (K, K, K, K)."""
    eps = np.asarray(eps, dtype=float).ravel()
    eri = np.asarray(eri, dtype=float)
    n_occ = int(n_occ)
    K = eps.size
    if eri.shape != (K, K, K, K) or not (1 <= n_occ < K):
        raise ValueError("eri must be (K, K, K, K) and 1 <= n_occ < K")
    pairs = [(i, a) for i in range(n_occ) for a in range(n_occ, K)]
    D = np.array([eps[a] - eps[i] for i, a in pairs])
    if np.any(D <= 0.0):
        raise ValueError("the reference must have a positive occupied-virtual gap")
    J = np.array([[eri[i, a, j, b] for (j, b) in pairs] for (i, a) in pairs])
    chi0 = np.diag(-4.0 / D)
    chi = np.linalg.solve(np.eye(len(pairs)) - chi0 @ J, chi0)
    vph = np.array([[eri[p, q, i, a] for (i, a) in pairs] for p in range(K) for q in range(K)]).reshape(K, K, len(pairs))
    return eri + np.einsum('pqm,mn,rsn->pqrs', vph, chi, vph, optimize=True)

import numpy as np


def bse_excitations(eps: "np.ndarray", eri: "np.ndarray", W0: "np.ndarray", n_occ: int, triplet: int) -> "np.ndarray":
    """Casida problem [[A, B], [-B, -A]] (X, Y)^T = Omega (X, Y)^T with the source's (v, W0) kernel (Eq 21-22):
    singlet A = D + 2J - K, B = 2J - K'; triplet A = D - K, B = -K', where D_{ia,jb} = (e_a - e_i) delta delta,
    J_{ia,jb} = (ia|jb), K_{ia,jb} = (ij|W0|ab), K'_{ia,jb} = (ib|W0|aj); pairs (i, a) ordered i-major, a-minor.
    Positive roots Omega_nu ascending; each (X, Y) normalised to sum(X^2 - Y^2) = 1 with the sign fixed so that
    the component of X + Y of largest magnitude is positive.  Returns array (2 n + 1, n), n = n_occ (K - n_occ):
    row 0 = Omega, rows 1..n = X (row = excitation nu, column = pair), rows n+1..2n = Y."""
    eps = np.asarray(eps, dtype=float).ravel()
    eri = np.asarray(eri, dtype=float)
    W0 = np.asarray(W0, dtype=float)
    n_occ, triplet = int(n_occ), int(triplet)
    K = eps.size
    if eri.shape != (K, K, K, K) or W0.shape != (K, K, K, K) or not (1 <= n_occ < K) or triplet not in (0, 1):
        raise ValueError("inconsistent shapes, n_occ out of range or triplet not in {0, 1}")
    pairs = [(i, a) for i in range(n_occ) for a in range(n_occ, K)]
    n = len(pairs)
    D = np.diag([eps[a] - eps[i] for i, a in pairs])
    J = np.array([[eri[i, a, j, b] for (j, b) in pairs] for (i, a) in pairs])
    Kx = np.array([[W0[i, j, a, b] for (j, b) in pairs] for (i, a) in pairs])
    Kp = np.array([[W0[i, b, a, j] for (j, b) in pairs] for (i, a) in pairs])
    if triplet:
        A, B = D - Kx, -Kp
    else:
        A, B = D + 2.0 * J - Kx, 2.0 * J - Kp
    w, Uv = np.linalg.eigh(A - B)
    if w.min() <= 0.0:
        raise ValueError("A - B is not positive definite (reference instability)")
    S = Uv @ np.diag(np.sqrt(w)) @ Uv.T
    O2, T = np.linalg.eigh(S @ (A + B) @ S)
    if O2.min() <= 0.0:
        raise ValueError("imaginary excitation energy")
    Om = np.sqrt(O2)
    XpY = S @ T / np.sqrt(Om)[None, :]
    XmY = np.linalg.solve(S, T) * np.sqrt(Om)[None, :]
    X, Y = 0.5 * (XpY + XmY), 0.5 * (XpY - XmY)
    for nu in range(n):
        k = int(np.argmax(np.abs(XpY[:, nu])))
        if XpY[k, nu] < 0.0:
            X[:, nu] = -X[:, nu]
            Y[:, nu] = -Y[:, nu]
    return np.vstack([Om[None, :], X.T, Y.T])

import numpy as np


def _half_diagrams(eri: "np.ndarray", W0: "np.ndarray", bse: "np.ndarray", n_occ: int) -> "np.ndarray":
    """Helper: the source's half-diagrams a and b (Eq 12) for every excitation nu and every intermediate orbital k:
    a[nu, k, p] = sqrt(2) sum_{jb} (pk|jb) (X + Y)^nu_{jb};
    for occupied k (hole intermediates, Eq 12b) b[nu, k, q] = -(1/sqrt 2) sum_{ia} [ (iq|W0|ka) X^nu_{ia} + (aq|W0|ki) Y^nu_{ia} ];
    for virtual k = c (particle intermediates) the particle-hole transform b[nu, c, q] = -(1/sqrt 2) sum_{ia} [ (aq|W0|ci) X^nu_{ia} + (iq|W0|ca) Y^nu_{ia} ].
    Returns array (2, n, K, K): [a, b] indexed [nu, k, orbital]."""
    eri = np.asarray(eri, dtype=float)
    W0 = np.asarray(W0, dtype=float)
    bse = np.asarray(bse, dtype=float)
    n_occ = int(n_occ)
    K = eri.shape[0]
    n = n_occ * (K - n_occ)
    if eri.shape != (K, K, K, K) or W0.shape != (K, K, K, K) or bse.shape != (2 * n + 1, n) or not (1 <= n_occ < K):
        raise ValueError("inconsistent shapes")
    pairs = [(i, a) for i in range(n_occ) for a in range(n_occ, K)]
    X, Y = bse[1:n + 1], bse[n + 1:]                       # [nu, pair]
    XpY = X + Y
    # a: (pk|jb) contracted with (X + Y)
    vjb = np.array([eri[:, :, j, b] for (j, b) in pairs])   # [pair, p, k]
    a = np.sqrt(2.0) * np.einsum('nm,mpk->nkp', XpY, vjb)
    # b: hole intermediates (occupied k) and particle intermediates (virtual c)
    b = np.zeros((n, K, K))
    for m, (i, aa) in enumerate(pairs):
        # (iq|W0|ka) -> W0[i, q, k, aa];  (aq|W0|ki) -> W0[aa, q, k, i]
        hole = np.einsum('n,qk->nkq', X[:, m], W0[i, :, :, aa]) + np.einsum('n,qk->nkq', Y[:, m], W0[aa, :, :, i])
        part = np.einsum('n,qk->nkq', X[:, m], W0[aa, :, :, i]) + np.einsum('n,qk->nkq', Y[:, m], W0[i, :, :, aa])
        b[:, :n_occ, :] += hole[:, :n_occ, :]
        b[:, n_occ:, :] += part[:, n_occ:, :]
    b *= -1.0 / np.sqrt(2.0)
    return np.stack([a, b])


def self_energy_residues(eri: "np.ndarray", W0: "np.ndarray", bse: "np.ndarray", n_occ: int) -> "np.ndarray":
    """Residue tensor of the source's minimal PSD self-energy: R[nu, k, p, q] = c_p c_q with the complete amplitude
    c[nu, k, p] = a[nu, k, p] + b[nu, k, p] of Eq 12 (a: bare interaction x spin-summed singlet transition density,
    sqrt(2) prefactor; b: screened exchange with X and Y, -1/sqrt(2) prefactor; particle intermediates with X and Y
    exchanged), so that Sigma_pq(omega) = sum_{nu,k} R[nu, k, p, q] / (omega - pole_{nu,k}).
    Returns array (n, K, K, K) indexed [nu, k, p, q]."""
    hd = _half_diagrams(eri, W0, bse, n_occ)
    c = hd[0] + hd[1]                                       # [nu, k, p]
    return np.einsum('nkp,nkq->nkpq', c, c)

import numpy as np


def psd_self_energy(eps: "np.ndarray", Om: "np.ndarray", res: "np.ndarray", n_occ: int, omega: float) -> "np.ndarray":
    """The source's minimal PSD self-energy (Eq 11/13) at real frequency omega (eta -> 0):
    Sigma_pq(omega) = sum_nu [ sum_{k occ} R[nu, k, p, q] / (omega - e_k + Omega_nu)
                             + sum_{c vir} R[nu, c, p, q] / (omega - e_c - Omega_nu) ],
    with the residues R = c c^T of step 06, singlet excitations only.  Returns the real symmetric (K, K) matrix."""
    eps = np.asarray(eps, dtype=float).ravel()
    Om = np.asarray(Om, dtype=float).ravel()
    res = np.asarray(res, dtype=float)
    n_occ, omega = int(n_occ), float(omega)
    K = eps.size
    if res.shape != (Om.size, K, K, K) or not (1 <= n_occ < K) or not np.isfinite(omega):
        raise ValueError("inconsistent shapes")
    den = np.empty((Om.size, K))
    den[:, :n_occ] = omega - eps[None, :n_occ] + Om[:, None]
    den[:, n_occ:] = omega - eps[None, n_occ:] - Om[:, None]
    if np.any(np.abs(den) < 1e-12):
        raise ValueError("omega coincides with a pole")
    return np.einsum('nkpq,nk->pq', res, 1.0 / den, optimize=True)

import numpy as np


def quasiparticle_energy(eps: "np.ndarray", Om: "np.ndarray", res: "np.ndarray", n_occ: int, p: int) -> float:
    """One-shot graphical solution of omega = e_p + Sigma_pp(omega) for orbital p (0-based) with the PSD-I
    self-energy of step 07 on the HF reference: the root nearest e_p, located by bisection on
    f(omega) = omega - e_p - Sigma_pp(omega) in the bracket [e_p - w, e_p + w] with w = 0.5 widened by 0.25 until
    f changes sign, to |bracket| < 1e-12.  Returns the quasiparticle energy."""
    eps = np.asarray(eps, dtype=float).ravel()
    p = int(p)
    K = eps.size
    if not (0 <= p < K):
        raise ValueError("p out of range")
    e0 = eps[p]

    def _f(w):
        return w - e0 - psd_self_energy(eps, Om, res, n_occ, w)[p, p]

    lo, hi = e0 - 0.5, e0 + 0.5
    for _ in range(40):
        if _f(lo) * _f(hi) <= 0.0:
            break
        lo -= 0.25
        hi += 0.25
    else:
        raise ValueError("no quasiparticle root bracketed")
    flo = _f(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = _f(mid)
        if flo * fm <= 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
        if hi - lo < 1e-12:
            break
    return float(0.5 * (lo + hi))

import numpy as np


def psd_ionization_potential(t: float, delta: float, U: float, kappa: float, eps_site: "np.ndarray", n_pairs: int) -> float:
    """ORCHESTRATOR: the first ionisation potential of the PPP chain at the PSD-I level, IP = -e_HOMO^QP, with
    the chain of steps 01-08 (singlet BSE excitations; the triplet manifold of step 05 is evaluated as a
    stability check; step 06 supplies the self-energy residues)."""
    n_pairs = int(n_pairs)
    hV = ppp_hamiltonian(t, delta, U, kappa, eps_site)
    h, V = hV[0], hV[1]
    ref = rhf_reference(h, V, n_pairs)
    eps, C = ref[0], ref[1:]
    eri = mo_two_electron_integrals(V, C)
    W0 = static_screened_interaction(eps, eri, n_pairs)
    bse_s = bse_excitations(eps, eri, W0, n_pairs, 0)
    bse_t = bse_excitations(eps, eri, W0, n_pairs, 1)
    if bse_t[0].min() <= 0.0 or bse_s[0].min() <= 0.0:
        raise ValueError("unstable reference")
    res = self_energy_residues(eri, W0, bse_s, n_pairs)
    e_qp = quasiparticle_energy(eps, bse_s[0], res, n_pairs, n_pairs - 1)
    sig = psd_self_energy(eps, bse_s[0], res, n_pairs, e_qp)
    if abs(e_qp - eps[n_pairs - 1] - sig[n_pairs - 1, n_pairs - 1]) > 1e-9:
        raise ValueError("quasiparticle equation not satisfied")
    return float(-e_qp)
SCICODE_GOLD_EOF
