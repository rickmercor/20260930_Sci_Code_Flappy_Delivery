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

_MK_SYSTEM = """import numpy as np

def _mk_system(seed, M):
    rng = np.random.default_rng(seed)
    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])
    dx = 0.10 + 0.05 * rng.random(M)
    x = np.empty(2 * M)
    x[0::2] = centers
    x[1::2] = centers + dx
    tau = 0.95 + 0.10 * rng.random(2 * M)
    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)
    m = 0.5 * (x[:, None] + x[None, :])
    V = (A[:, :, None, None] * A[None, None, :, :]
         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))
    h = np.zeros((2 * M, 2 * M))
    diag_b = -1.25 + 0.08 * rng.random(M)
    gap = 0.45 + 0.55 * rng.random(M)
    for a in range(M):
        h[2 * a, 2 * a] = diag_b[a]
        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]
    off = 0.02 * rng.standard_normal((2 * M, 2 * M))
    off = 0.5 * (off + off.T)
    np.fill_diagonal(off, 0.0)
    h = h + off
    return h, V
"""

def _validate_system(h, V):
    """Validate the integral arrays; return (h, V, M) with M bond units."""
    h = np.asarray(h, dtype=float)
    V = np.asarray(V, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("h must be a square matrix")
    n_orb = h.shape[0]
    if n_orb < 2 or n_orb % 2 != 0:
        raise ValueError("h must have even dimension 2M with M >= 1")
    if V.shape != (n_orb, n_orb, n_orb, n_orb):
        raise ValueError("V must have shape (2M, 2M, 2M, 2M) matching h")
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(V))):
        raise ValueError("integrals must be finite")
    if not np.allclose(h, h.T, rtol=0.0, atol=1e-10):
        raise ValueError("h must be symmetric")
    for perm in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        if not np.allclose(V, np.transpose(V, perm), rtol=0.0, atol=1e-10):
            raise ValueError("V must have the 8-fold real-orbital permutation symmetry")
    return h, V, n_orb // 2

def _validate_gaps(omega, M):
    omega = np.asarray(omega, dtype=float)
    if omega.shape != (M,):
        raise ValueError("omega must be a length-M vector of bond-unit gaps")
    if not np.all(np.isfinite(omega)):
        raise ValueError("omega must be finite")
    return omega

def _pair_integrals(V):
    """Direct J_pq = V_ppqq, exchange K_pq = V_pqqp, pair-transfer L_pq = V_pqpq,
    and the combination G = 2J - K."""
    idx = np.arange(V.shape[0])
    J = V[idx[:, None], idx[:, None], idx[None, :], idx[None, :]]
    K = V[idx[:, None], idx[None, :], idx[None, :], idx[:, None]]
    L = V[idx[:, None], idx[None, :], idx[:, None], idx[None, :]]
    return J, K, L, 2.0 * J - K

def _occupations(omega):
    """Occupations n_{a,mu} = 1 + (-1)^mu * w_a/eta_a, eta_a = sqrt(w_a^2 + 1)."""
    omega = np.asarray(omega, dtype=float)
    eta = np.sqrt(omega * omega + 1.0)
    n = np.empty(2 * omega.size)
    n[0::2] = 1.0 + omega / eta
    n[1::2] = 1.0 - omega / eta
    return n, eta

def _orbital_energies(h, L, G, n):
    """eps_p = h_pp + L_pp/2 + (1/2) sum_{q outside p's bond unit} G_pq n_q."""
    n_orb = h.shape[0]
    eps = np.empty(n_orb)
    for p in range(n_orb):
        s = h[p, p] + 0.5 * L[p, p]
        for q in range(n_orb):
            if q // 2 != p // 2:
                s += 0.5 * G[p, q] * n[q]
        eps[p] = s
    return eps

def _relax_bond_gaps(h, V, alpha=0.55, n_sweeps=5):
    """Exactly n_sweeps damped Jacobi sweeps of
    w_a <- w_a + alpha * ((eps_{a1} - eps_{a0}) / L_{a0,a1} - w_a) from w = 0."""
    _, _, L, G = _pair_integrals(V)
    M = h.shape[0] // 2
    l_intra = np.array([L[2 * a, 2 * a + 1] for a in range(M)])
    if np.any(np.abs(l_intra) < 1e-12):
        raise ValueError("an intra-unit pair-transfer integral L_{a0,a1} vanishes")
    omega = np.zeros(M)
    for _ in range(int(n_sweeps)):
        n, _ = _occupations(omega)
        eps = _orbital_energies(h, L, G, n)
        omega = omega + alpha * ((eps[1::2] - eps[0::2]) / l_intra - omega)
    return omega

def relax_bond_gaps(h, V, alpha=0.55, n_sweeps=5):
    """Reference implementation. Deterministic."""
    h, V, _ = _validate_system(h, V)
    alpha = float(alpha)
    if not (np.isfinite(alpha) and 0.0 < alpha <= 1.0):
        raise ValueError("alpha must lie in (0, 1]")
    if not float(n_sweeps).is_integer() or int(n_sweeps) < 1:
        raise ValueError("n_sweeps must be a positive integer")
    return _relax_bond_gaps(h, V, alpha=alpha, n_sweeps=int(n_sweeps))

import numpy as np

def _validate_system(h, V):
    """Validate the integral arrays; return (h, V, M) with M bond units."""
    h = np.asarray(h, dtype=float)
    V = np.asarray(V, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("h must be a square matrix")
    n_orb = h.shape[0]
    if n_orb < 2 or n_orb % 2 != 0:
        raise ValueError("h must have even dimension 2M with M >= 1")
    if V.shape != (n_orb, n_orb, n_orb, n_orb):
        raise ValueError("V must have shape (2M, 2M, 2M, 2M) matching h")
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(V))):
        raise ValueError("integrals must be finite")
    if not np.allclose(h, h.T, rtol=0.0, atol=1e-10):
        raise ValueError("h must be symmetric")
    for perm in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        if not np.allclose(V, np.transpose(V, perm), rtol=0.0, atol=1e-10):
            raise ValueError("V must have the 8-fold real-orbital permutation symmetry")
    return h, V, n_orb // 2

def _validate_gaps(omega, M):
    omega = np.asarray(omega, dtype=float)
    if omega.shape != (M,):
        raise ValueError("omega must be a length-M vector of bond-unit gaps")
    if not np.all(np.isfinite(omega)):
        raise ValueError("omega must be finite")
    return omega

def _pair_integrals(V):
    """Direct J_pq = V_ppqq, exchange K_pq = V_pqqp, pair-transfer L_pq = V_pqpq,
    and the combination G = 2J - K."""
    idx = np.arange(V.shape[0])
    J = V[idx[:, None], idx[:, None], idx[None, :], idx[None, :]]
    K = V[idx[:, None], idx[None, :], idx[None, :], idx[:, None]]
    L = V[idx[:, None], idx[None, :], idx[:, None], idx[None, :]]
    return J, K, L, 2.0 * J - K

def _occupations(omega):
    """Occupations n_{a,mu} = 1 + (-1)^mu * w_a/eta_a, eta_a = sqrt(w_a^2 + 1)."""
    omega = np.asarray(omega, dtype=float)
    eta = np.sqrt(omega * omega + 1.0)
    n = np.empty(2 * omega.size)
    n[0::2] = 1.0 + omega / eta
    n[1::2] = 1.0 - omega / eta
    return n, eta

def _orbital_energies(h, L, G, n):
    """eps_p = h_pp + L_pp/2 + (1/2) sum_{q outside p's bond unit} G_pq n_q."""
    n_orb = h.shape[0]
    eps = np.empty(n_orb)
    for p in range(n_orb):
        s = h[p, p] + 0.5 * L[p, p]
        for q in range(n_orb):
            if q // 2 != p // 2:
                s += 0.5 * G[p, q] * n[q]
        eps[p] = s
    return eps

def _reference_energy(h, V, omega):
    """E_ref = sum_p eps_p n_p - sum_a L_{a0,a1}/eta_a
    - (1/2) sum'_{p<q} G_pq n_p n_q  (' excludes same-unit pairs)."""
    _, _, L, G = _pair_integrals(V)
    M = h.shape[0] // 2
    n, eta = _occupations(omega)
    eps = _orbital_energies(h, L, G, n)
    energy = float(np.dot(eps, n))
    for a in range(M):
        energy -= L[2 * a, 2 * a + 1] / eta[a]
    for p in range(2 * M):
        for q in range(p + 1, 2 * M):
            if q // 2 != p // 2:
                energy -= 0.5 * G[p, q] * n[p] * n[q]
    return energy

def reference_pair_energy(h, V, omega):
    """Reference implementation. Deterministic."""
    h, V, M = _validate_system(h, V)
    omega = _validate_gaps(omega, M)
    return float(_reference_energy(h, V, omega))

import numpy as np

def _validate_system(h, V):
    """Validate the integral arrays; return (h, V, M) with M bond units."""
    h = np.asarray(h, dtype=float)
    V = np.asarray(V, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("h must be a square matrix")
    n_orb = h.shape[0]
    if n_orb < 2 or n_orb % 2 != 0:
        raise ValueError("h must have even dimension 2M with M >= 1")
    if V.shape != (n_orb, n_orb, n_orb, n_orb):
        raise ValueError("V must have shape (2M, 2M, 2M, 2M) matching h")
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(V))):
        raise ValueError("integrals must be finite")
    if not np.allclose(h, h.T, rtol=0.0, atol=1e-10):
        raise ValueError("h must be symmetric")
    for perm in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        if not np.allclose(V, np.transpose(V, perm), rtol=0.0, atol=1e-10):
            raise ValueError("V must have the 8-fold real-orbital permutation symmetry")
    return h, V, n_orb // 2

def _validate_gaps(omega, M):
    omega = np.asarray(omega, dtype=float)
    if omega.shape != (M,):
        raise ValueError("omega must be a length-M vector of bond-unit gaps")
    if not np.all(np.isfinite(omega)):
        raise ValueError("omega must be finite")
    return omega

def _pair_integrals(V):
    """Direct J_pq = V_ppqq, exchange K_pq = V_pqqp, pair-transfer L_pq = V_pqpq,
    and the combination G = 2J - K."""
    idx = np.arange(V.shape[0])
    J = V[idx[:, None], idx[:, None], idx[None, :], idx[None, :]]
    K = V[idx[:, None], idx[None, :], idx[None, :], idx[:, None]]
    L = V[idx[:, None], idx[None, :], idx[:, None], idx[None, :]]
    return J, K, L, 2.0 * J - K

def _occupations(omega):
    """Occupations n_{a,mu} = 1 + (-1)^mu * w_a/eta_a, eta_a = sqrt(w_a^2 + 1)."""
    omega = np.asarray(omega, dtype=float)
    eta = np.sqrt(omega * omega + 1.0)
    n = np.empty(2 * omega.size)
    n[0::2] = 1.0 + omega / eta
    n[1::2] = 1.0 - omega / eta
    return n, eta

def _orbital_energies(h, L, G, n):
    """eps_p = h_pp + L_pp/2 + (1/2) sum_{q outside p's bond unit} G_pq n_q."""
    n_orb = h.shape[0]
    eps = np.empty(n_orb)
    for p in range(n_orb):
        s = h[p, p] + 0.5 * L[p, p]
        for q in range(n_orb):
            if q // 2 != p // 2:
                s += 0.5 * G[p, q] * n[q]
        eps[p] = s
    return eps

def _generalized_fock(h, V, omega):
    """f_pq = h_pq n_p + (1/2) sum_r (2 V_rrpq - V_rqpr) D_rp + sum_r V_rprq P_rp,
    with the reference two-body data D_pp = 0, D within a unit = 0, D_rp = n_r n_p
    otherwise, P_pp = n_p, and P_{a0,a1} = P_{a1,a0} = -1/eta_a."""
    n_orb = h.shape[0]
    M = n_orb // 2
    n, eta = _occupations(omega)
    D = np.outer(n, n)
    for a in range(M):
        D[2 * a, 2 * a + 1] = D[2 * a + 1, 2 * a] = 0.0
    np.fill_diagonal(D, 0.0)
    P = np.zeros((n_orb, n_orb))
    np.fill_diagonal(P, n)
    for a in range(M):
        P[2 * a, 2 * a + 1] = P[2 * a + 1, 2 * a] = -1.0 / eta[a]
    f = np.zeros((n_orb, n_orb))
    for p in range(n_orb):
        for q in range(n_orb):
            s = h[p, q] * n[p]
            for r in range(n_orb):
                s += 0.5 * (2.0 * V[r, r, p, q] - V[r, q, p, r]) * D[r, p]
                s += V[r, p, r, q] * P[r, p]
            f[p, q] = s
    return f

def pair_generalized_fock(h, V, omega):
    """Reference implementation. Deterministic."""
    h, V, M = _validate_system(h, V)
    omega = _validate_gaps(omega, M)
    return _generalized_fock(h, V, omega)

import itertools

import numpy as np

def _validate_system(h, V):
    """Validate the integral arrays; return (h, V, M) with M bond units."""
    h = np.asarray(h, dtype=float)
    V = np.asarray(V, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("h must be a square matrix")
    n_orb = h.shape[0]
    if n_orb < 2 or n_orb % 2 != 0:
        raise ValueError("h must have even dimension 2M with M >= 1")
    if V.shape != (n_orb, n_orb, n_orb, n_orb):
        raise ValueError("V must have shape (2M, 2M, 2M, 2M) matching h")
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(V))):
        raise ValueError("integrals must be finite")
    if not np.allclose(h, h.T, rtol=0.0, atol=1e-10):
        raise ValueError("h must be symmetric")
    for perm in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        if not np.allclose(V, np.transpose(V, perm), rtol=0.0, atol=1e-10):
            raise ValueError("V must have the 8-fold real-orbital permutation symmetry")
    return h, V, n_orb // 2

def _validate_gaps(omega, M):
    omega = np.asarray(omega, dtype=float)
    if omega.shape != (M,):
        raise ValueError("omega must be a length-M vector of bond-unit gaps")
    if not np.all(np.isfinite(omega)):
        raise ValueError("omega must be finite")
    return omega

def _pair_integrals(V):
    """Direct J_pq = V_ppqq, exchange K_pq = V_pqqp, pair-transfer L_pq = V_pqpq,
    and the combination G = 2J - K."""
    idx = np.arange(V.shape[0])
    J = V[idx[:, None], idx[:, None], idx[None, :], idx[None, :]]
    K = V[idx[:, None], idx[None, :], idx[None, :], idx[:, None]]
    L = V[idx[:, None], idx[None, :], idx[:, None], idx[None, :]]
    return J, K, L, 2.0 * J - K

def _occupations(omega):
    """Occupations n_{a,mu} = 1 + (-1)^mu * w_a/eta_a, eta_a = sqrt(w_a^2 + 1)."""
    omega = np.asarray(omega, dtype=float)
    eta = np.sqrt(omega * omega + 1.0)
    n = np.empty(2 * omega.size)
    n[0::2] = 1.0 + omega / eta
    n[1::2] = 1.0 - omega / eta
    return n, eta

def _orbital_energies(h, L, G, n):
    """eps_p = h_pp + L_pp/2 + (1/2) sum_{q outside p's bond unit} G_pq n_q."""
    n_orb = h.shape[0]
    eps = np.empty(n_orb)
    for p in range(n_orb):
        s = h[p, p] + 0.5 * L[p, p]
        for q in range(n_orb):
            if q // 2 != p // 2:
                s += 0.5 * G[p, q] * n[q]
        eps[p] = s
    return eps

def _generalized_fock(h, V, omega):
    """f_pq = h_pq n_p + (1/2) sum_r (2 V_rrpq - V_rqpr) D_rp + sum_r V_rprq P_rp,
    with the reference two-body data D_pp = 0, D within a unit = 0, D_rp = n_r n_p
    otherwise, P_pp = n_p, and P_{a0,a1} = P_{a1,a0} = -1/eta_a."""
    n_orb = h.shape[0]
    M = n_orb // 2
    n, eta = _occupations(omega)
    D = np.outer(n, n)
    for a in range(M):
        D[2 * a, 2 * a + 1] = D[2 * a + 1, 2 * a] = 0.0
    np.fill_diagonal(D, 0.0)
    P = np.zeros((n_orb, n_orb))
    np.fill_diagonal(P, n)
    for a in range(M):
        P[2 * a, 2 * a + 1] = P[2 * a + 1, 2 * a] = -1.0 / eta[a]
    f = np.zeros((n_orb, n_orb))
    for p in range(n_orb):
        for q in range(n_orb):
            s = h[p, q] * n[p]
            for r in range(n_orb):
                s += 0.5 * (2.0 * V[r, r, p, q] - V[r, q, p, r]) * D[r, p]
                s += V[r, p, r, q] * P[r, p]
            f[p, q] = s
    return f

def _t_open_shell(J, K, L, G, p, q):
    """Blocked-singlet placement energy t_pq: J_pq + K_pq - L_pp/2 - L_qq/2,
    with an extra -G_pq/2 when p and q sit in different bond units."""
    val = J[p, q] + K[p, q] - 0.5 * L[p, p] - 0.5 * L[q, q]
    if p // 2 != q // 2:
        val -= 0.5 * G[p, q]
    return val

def _g_minus(G, omega, eta, a, p):
    """g-_{a,p} = (w_a / (2 eta_a)) (G_{a0,p} - G_{a1,p})."""
    return 0.5 * (omega[a] / eta[a]) * (G[2 * a, p] - G[2 * a + 1, p])

def _g_mm(G, omega, eta, a, b):
    """g--_{a,b} = (w_a w_b / (2 eta_a eta_b))
    (G_{a0,b0} - G_{a0,b1} - G_{a1,b0} + G_{a1,b1})."""
    return 0.5 * (omega[a] * omega[b] / (eta[a] * eta[b])) * (
        G[2 * a, 2 * b] - G[2 * a, 2 * b + 1]
        - G[2 * a + 1, 2 * b] + G[2 * a + 1, 2 * b + 1])

def _g_sym(G, a, b):
    """g_{a,b} = (G_{a0,b0} + G_{a0,b1} + G_{a1,b0} + G_{a1,b1}) / 2."""
    return 0.5 * (G[2 * a, 2 * b] + G[2 * a, 2 * b + 1]
                  + G[2 * a + 1, 2 * b] + G[2 * a + 1, 2 * b + 1])

def _single_terms(h, V, omega, fock=None):
    """(excitation energy, coupling) for every single excitation."""
    J, K, L, G = _pair_integrals(V)
    M = h.shape[0] // 2
    n, eta = _occupations(omega)
    eps = _orbital_energies(h, L, G, n)
    f = _generalized_fock(h, V, omega) if fock is None else np.asarray(fock, dtype=float)
    sn = np.sqrt(n)
    terms = []
    for a in range(M):
        a0, a1 = 2 * a, 2 * a + 1
        eta_l = eta[a] * L[a0, a1]
        # occupation swap within unit a
        terms.append((2.0 * eta_l,
                      (eps[a0] - eps[a1] + omega[a] * L[a0, a1]) / eta[a]))
        # pair split within unit a
        terms.append((eta_l + _t_open_shell(J, K, L, G, a0, a1),
                      (f[a0, a1] - f[a1, a0]) / (sn[a0] + sn[a1])))
    for a, b in itertools.permutations(range(M), 2):
        for mu in (0, 1):
            for nu in (0, 1):
                am, ax = 2 * a + mu, 2 * a + 1 - mu
                bn, bx = 2 * b + nu, 2 * b + 1 - nu
                d_e = (eta[a] * L[2 * a, 2 * a + 1] + eta[b] * L[2 * b, 2 * b + 1]
                       + _t_open_shell(J, K, L, G, am, bn)
                       + eps[bx] - eps[ax] + G[2 * b, 2 * b + 1]
                       + _g_mm(G, omega, eta, a, b)
                       - _g_minus(G, omega, eta, a, bx)
                       + _g_minus(G, omega, eta, b, ax)
                       - 0.5 * G[ax, bx])
                rhs = (f[am, bn]
                       + n[am] * np.sqrt(n[bn] / n[bx]) * V[bn, bx, am, bx]
                       + 0.5 * (2.0 * V[bx, bx, am, bn] - V[bx, bn, am, bx]
                                - V[bn, bn, am, bn]) * n[am] * n[bn])
                coup = (-1.0) ** (mu + nu + 1) * np.sqrt(n[bx] / (2.0 * n[am])) * rhs
                terms.append((d_e, coup))
    return terms

def _en2_sum(terms):
    total = 0.0
    for d_e, coup in terms:
        if d_e == 0.0:
            raise ValueError("vanishing excitation energy in a perturbative denominator")
        total -= coup * coup / d_e
    return float(total)

def single_excitation_correction(h, V, omega, fock=None):
    """Reference implementation. Deterministic."""
    h, V, M = _validate_system(h, V)
    omega = _validate_gaps(omega, M)
    return _en2_sum(_single_terms(h, V, omega, fock))

import itertools

import numpy as np

def _validate_system(h, V):
    """Validate the integral arrays; return (h, V, M) with M bond units."""
    h = np.asarray(h, dtype=float)
    V = np.asarray(V, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("h must be a square matrix")
    n_orb = h.shape[0]
    if n_orb < 2 or n_orb % 2 != 0:
        raise ValueError("h must have even dimension 2M with M >= 1")
    if V.shape != (n_orb, n_orb, n_orb, n_orb):
        raise ValueError("V must have shape (2M, 2M, 2M, 2M) matching h")
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(V))):
        raise ValueError("integrals must be finite")
    if not np.allclose(h, h.T, rtol=0.0, atol=1e-10):
        raise ValueError("h must be symmetric")
    for perm in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        if not np.allclose(V, np.transpose(V, perm), rtol=0.0, atol=1e-10):
            raise ValueError("V must have the 8-fold real-orbital permutation symmetry")
    return h, V, n_orb // 2

def _validate_gaps(omega, M):
    omega = np.asarray(omega, dtype=float)
    if omega.shape != (M,):
        raise ValueError("omega must be a length-M vector of bond-unit gaps")
    if not np.all(np.isfinite(omega)):
        raise ValueError("omega must be finite")
    return omega

def _pair_integrals(V):
    """Direct J_pq = V_ppqq, exchange K_pq = V_pqqp, pair-transfer L_pq = V_pqpq,
    and the combination G = 2J - K."""
    idx = np.arange(V.shape[0])
    J = V[idx[:, None], idx[:, None], idx[None, :], idx[None, :]]
    K = V[idx[:, None], idx[None, :], idx[None, :], idx[:, None]]
    L = V[idx[:, None], idx[None, :], idx[:, None], idx[None, :]]
    return J, K, L, 2.0 * J - K

def _occupations(omega):
    """Occupations n_{a,mu} = 1 + (-1)^mu * w_a/eta_a, eta_a = sqrt(w_a^2 + 1)."""
    omega = np.asarray(omega, dtype=float)
    eta = np.sqrt(omega * omega + 1.0)
    n = np.empty(2 * omega.size)
    n[0::2] = 1.0 + omega / eta
    n[1::2] = 1.0 - omega / eta
    return n, eta

def _orbital_energies(h, L, G, n):
    """eps_p = h_pp + L_pp/2 + (1/2) sum_{q outside p's bond unit} G_pq n_q."""
    n_orb = h.shape[0]
    eps = np.empty(n_orb)
    for p in range(n_orb):
        s = h[p, p] + 0.5 * L[p, p]
        for q in range(n_orb):
            if q // 2 != p // 2:
                s += 0.5 * G[p, q] * n[q]
        eps[p] = s
    return eps

def _t_open_shell(J, K, L, G, p, q):
    """Blocked-singlet placement energy t_pq: J_pq + K_pq - L_pp/2 - L_qq/2,
    with an extra -G_pq/2 when p and q sit in different bond units."""
    val = J[p, q] + K[p, q] - 0.5 * L[p, p] - 0.5 * L[q, q]
    if p // 2 != q // 2:
        val -= 0.5 * G[p, q]
    return val

def _g_minus(G, omega, eta, a, p):
    """g-_{a,p} = (w_a / (2 eta_a)) (G_{a0,p} - G_{a1,p})."""
    return 0.5 * (omega[a] / eta[a]) * (G[2 * a, p] - G[2 * a + 1, p])

def _g_mm(G, omega, eta, a, b):
    """g--_{a,b} = (w_a w_b / (2 eta_a eta_b))
    (G_{a0,b0} - G_{a0,b1} - G_{a1,b0} + G_{a1,b1})."""
    return 0.5 * (omega[a] * omega[b] / (eta[a] * eta[b])) * (
        G[2 * a, 2 * b] - G[2 * a, 2 * b + 1]
        - G[2 * a + 1, 2 * b] + G[2 * a + 1, 2 * b + 1])

def _g_sym(G, a, b):
    """g_{a,b} = (G_{a0,b0} + G_{a0,b1} + G_{a1,b0} + G_{a1,b1}) / 2."""
    return 0.5 * (G[2 * a, 2 * b] + G[2 * a, 2 * b + 1]
                  + G[2 * a + 1, 2 * b] + G[2 * a + 1, 2 * b + 1])

def _double_terms(h, V, omega):
    """(excitation energy, coupling) for every double excitation EXCEPT the
    complementary double splits (those are dressed variationally instead)."""
    J, K, L, G = _pair_integrals(V)
    M = h.shape[0] // 2
    n, eta = _occupations(omega)
    eps = _orbital_energies(h, L, G, n)
    sn = np.sqrt(n)
    terms = []
    for a, b in itertools.combinations(range(M), 2):
        a0, a1, b0, b1 = 2 * a, 2 * a + 1, 2 * b, 2 * b + 1
        eta_la = eta[a] * L[a0, a1]
        eta_lb = eta[b] * L[b0, b1]
        gmm_ab = _g_mm(G, omega, eta, a, b)
        # double swap
        coup = sum((-1.0) ** (mu + nu) * G[2 * a + mu, 2 * b + nu]
                   for mu in (0, 1) for nu in (0, 1)) / (2.0 * eta[a] * eta[b])
        terms.append((2.0 * eta_la + 2.0 * eta_lb + 4.0 * gmm_ab, coup))
        # double split (its complementary partner is dressed, not summed)
        d_e = (eta_la + eta_lb + _t_open_shell(J, K, L, G, a0, a1)
               + _t_open_shell(J, K, L, G, b0, b1) + gmm_ab)
        coup = ((sn[a0] - sn[a1]) * (sn[b0] - sn[b1]) * V[a0, a1, b0, b1]
                - 0.5 * (sn[a0] * sn[b0] + sn[a1] * sn[b1]) * V[a0, b1, b0, a1]
                + 0.5 * (sn[a1] * sn[b0] + sn[a0] * sn[b1]) * V[a1, b1, b0, a0])
        terms.append((d_e, coup))
    for a, b in itertools.permutations(range(M), 2):
        b0, b1 = 2 * b, 2 * b + 1
        # swap in unit a + split in unit b
        d_e = (2.0 * eta[a] * L[2 * a, 2 * a + 1] + eta[b] * L[b0, b1]
               + _t_open_shell(J, K, L, G, b0, b1)
               + 2.0 * _g_mm(G, omega, eta, a, b))
        coup = (sn[b0] - sn[b1]) * sum(
            (-1.0) ** mu * (2.0 * V[2 * a + mu, 2 * a + mu, b0, b1]
                            - V[2 * a + mu, b1, b0, 2 * a + mu])
            for mu in (0, 1)) / (2.0 * eta[a])
        terms.append((d_e, coup))
    for a in range(M):
        others = [u for u in range(M) if u != a]
        a0, a1 = 2 * a, 2 * a + 1
        for b, g in itertools.permutations(others, 2):
            for nu in (0, 1):
                for lam in (0, 1):
                    bn, bx = 2 * b + nu, 2 * b + 1 - nu
                    gl, gx = 2 * g + lam, 2 * g + 1 - lam
                    common = (eta[b] * L[2 * b, 2 * b + 1]
                              + eta[g] * L[2 * g, 2 * g + 1]
                              + _t_open_shell(J, K, L, G, bn, gl)
                              + eps[gx] - eps[bx] + G[2 * g, 2 * g + 1]
                              - 0.5 * G[bx, gx])
                    gmm_ab = _g_mm(G, omega, eta, a, b)
                    gmm_ag = _g_mm(G, omega, eta, a, g)
                    gmm_bg = _g_mm(G, omega, eta, b, g)
                    gm_agx = _g_minus(G, omega, eta, a, gx)
                    gm_bgx = _g_minus(G, omega, eta, b, gx)
                    gm_abx = _g_minus(G, omega, eta, a, bx)
                    gm_gbx = _g_minus(G, omega, eta, g, bx)
                    # swap in a + transfer b_nu -> g_lam
                    d_e = (2.0 * eta[a] * L[a0, a1] + common
                           + 2.0 * gmm_ab + 2.0 * gmm_ag + gmm_bg
                           - 2.0 * gm_agx - gm_bgx + 2.0 * gm_abx + gm_gbx)
                    ssum = sum((-1.0) ** mu * (2.0 * V[bn, gl, 2 * a + mu, 2 * a + mu]
                                               - V[2 * a + mu, gl, bn, 2 * a + mu])
                               for mu in (0, 1))
                    coup = ((-1.0) ** (nu + lam + 1)
                            / (2.0 * np.sqrt(2.0) * eta[a])
                            * np.sqrt(n[bn] * n[gx]) * ssum)
                    terms.append((d_e, coup))
                    # split in a + transfer b_nu -> g_lam
                    d_e6 = (eta[a] * L[a0, a1] + _t_open_shell(J, K, L, G, a0, a1)
                            + common + gmm_ab + gmm_ag + gmm_bg
                            - gm_agx - gm_bgx + gm_abx + gm_gbx)
                    ssum6 = sum((-1.0) ** mu * sn[2 * a + mu]
                                * (2.0 * V[bn, gl, 2 * a + mu, 2 * a + 1 - mu]
                                   - V[2 * a + mu, gl, bn, 2 * a + 1 - mu])
                                for mu in (0, 1))
                    coup6 = ((-1.0) ** (nu + lam + 1) / (2.0 * np.sqrt(2.0))
                             * np.sqrt(n[bn] * n[gx]) * ssum6)
                    terms.append((d_e6, coup6))
                    # complementary split + transfer
                    d_e7 = d_e6 + (K[a0, bn] + K[a0, gl] + K[a1, bn] + K[a1, gl]
                                   - 2.0 * K[a0, a1] - 2.0 * K[bn, gl])
                    ssum7 = sum(sn[2 * a + mu] * V[2 * a + mu, gl, bn, 2 * a + 1 - mu]
                                for mu in (0, 1))
                    coup7 = ((-1.0) ** (nu + lam) * 0.5 * np.sqrt(1.5)
                             * np.sqrt(n[bn] * n[gx]) * ssum7)
                    terms.append((d_e7, coup7))
    return terms

def _en2_sum(terms):
    total = 0.0
    for d_e, coup in terms:
        if d_e == 0.0:
            raise ValueError("vanishing excitation energy in a perturbative denominator")
        total -= coup * coup / d_e
    return float(total)

def double_excitation_correction(h, V, omega):
    """Reference implementation. Deterministic."""
    h, V, M = _validate_system(h, V)
    omega = _validate_gaps(omega, M)
    return _en2_sum(_double_terms(h, V, omega))

import itertools

import numpy as np

def _validate_system(h, V):
    """Validate the integral arrays; return (h, V, M) with M bond units."""
    h = np.asarray(h, dtype=float)
    V = np.asarray(V, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("h must be a square matrix")
    n_orb = h.shape[0]
    if n_orb < 2 or n_orb % 2 != 0:
        raise ValueError("h must have even dimension 2M with M >= 1")
    if V.shape != (n_orb, n_orb, n_orb, n_orb):
        raise ValueError("V must have shape (2M, 2M, 2M, 2M) matching h")
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(V))):
        raise ValueError("integrals must be finite")
    if not np.allclose(h, h.T, rtol=0.0, atol=1e-10):
        raise ValueError("h must be symmetric")
    for perm in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        if not np.allclose(V, np.transpose(V, perm), rtol=0.0, atol=1e-10):
            raise ValueError("V must have the 8-fold real-orbital permutation symmetry")
    return h, V, n_orb // 2

def _validate_gaps(omega, M):
    omega = np.asarray(omega, dtype=float)
    if omega.shape != (M,):
        raise ValueError("omega must be a length-M vector of bond-unit gaps")
    if not np.all(np.isfinite(omega)):
        raise ValueError("omega must be finite")
    return omega

def _pair_integrals(V):
    """Direct J_pq = V_ppqq, exchange K_pq = V_pqqp, pair-transfer L_pq = V_pqpq,
    and the combination G = 2J - K."""
    idx = np.arange(V.shape[0])
    J = V[idx[:, None], idx[:, None], idx[None, :], idx[None, :]]
    K = V[idx[:, None], idx[None, :], idx[None, :], idx[:, None]]
    L = V[idx[:, None], idx[None, :], idx[:, None], idx[None, :]]
    return J, K, L, 2.0 * J - K

def _occupations(omega):
    """Occupations n_{a,mu} = 1 + (-1)^mu * w_a/eta_a, eta_a = sqrt(w_a^2 + 1)."""
    omega = np.asarray(omega, dtype=float)
    eta = np.sqrt(omega * omega + 1.0)
    n = np.empty(2 * omega.size)
    n[0::2] = 1.0 + omega / eta
    n[1::2] = 1.0 - omega / eta
    return n, eta

def _orbital_energies(h, L, G, n):
    """eps_p = h_pp + L_pp/2 + (1/2) sum_{q outside p's bond unit} G_pq n_q."""
    n_orb = h.shape[0]
    eps = np.empty(n_orb)
    for p in range(n_orb):
        s = h[p, p] + 0.5 * L[p, p]
        for q in range(n_orb):
            if q // 2 != p // 2:
                s += 0.5 * G[p, q] * n[q]
        eps[p] = s
    return eps

def _t_open_shell(J, K, L, G, p, q):
    """Blocked-singlet placement energy t_pq: J_pq + K_pq - L_pp/2 - L_qq/2,
    with an extra -G_pq/2 when p and q sit in different bond units."""
    val = J[p, q] + K[p, q] - 0.5 * L[p, p] - 0.5 * L[q, q]
    if p // 2 != q // 2:
        val -= 0.5 * G[p, q]
    return val

def _g_minus(G, omega, eta, a, p):
    """g-_{a,p} = (w_a / (2 eta_a)) (G_{a0,p} - G_{a1,p})."""
    return 0.5 * (omega[a] / eta[a]) * (G[2 * a, p] - G[2 * a + 1, p])

def _g_mm(G, omega, eta, a, b):
    """g--_{a,b} = (w_a w_b / (2 eta_a eta_b))
    (G_{a0,b0} - G_{a0,b1} - G_{a1,b0} + G_{a1,b1})."""
    return 0.5 * (omega[a] * omega[b] / (eta[a] * eta[b])) * (
        G[2 * a, 2 * b] - G[2 * a, 2 * b + 1]
        - G[2 * a + 1, 2 * b] + G[2 * a + 1, 2 * b + 1])

def _g_sym(G, a, b):
    """g_{a,b} = (G_{a0,b0} + G_{a0,b1} + G_{a1,b0} + G_{a1,b1}) / 2."""
    return 0.5 * (G[2 * a, 2 * b] + G[2 * a, 2 * b + 1]
                  + G[2 * a + 1, 2 * b] + G[2 * a + 1, 2 * b + 1])

def _pair_transfer_terms(h, V, omega):
    """(excitation energy, coupling) for every correlated pair-transfer state."""
    J, K, L, G = _pair_integrals(V)
    M = h.shape[0] // 2
    n, eta = _occupations(omega)
    eps = _orbital_energies(h, L, G, n)
    sn = np.sqrt(n)
    terms = []
    for a, b in itertools.permutations(range(M), 2):
        a0, a1, b0, b1 = 2 * a, 2 * a + 1, 2 * b, 2 * b + 1
        d_e = (eta[a] * L[a0, a1] + eta[b] * L[b0, b1]
               + eps[b0] + eps[b1] - eps[a0] - eps[a1] + 2.0 * G[b0, b1]
               - _g_sym(G, a, b) + _g_mm(G, omega, eta, a, b)
               - _g_minus(G, omega, eta, a, b0) - _g_minus(G, omega, eta, a, b1)
               + _g_minus(G, omega, eta, b, a0) + _g_minus(G, omega, eta, b, a1))
        coup = 0.5 * sum((-1.0) ** (mu + nu + 1) * L[2 * a + mu, 2 * b + nu]
                         * sn[2 * a + mu] * sn[2 * b + 1 - nu]
                         for mu in (0, 1) for nu in (0, 1))
        terms.append((d_e, coup))
    for a, b in itertools.combinations(range(M), 2):
        for mu in (0, 1):
            for nu in (0, 1):
                am, ax = 2 * a + mu, 2 * a + 1 - mu
                bn, bx = 2 * b + nu, 2 * b + 1 - nu
                for g in range(M):
                    if g in (a, b):
                        continue
                    g0, g1 = 2 * g, 2 * g + 1
                    part_b = (eta[a] * L[2 * a, 2 * a + 1]
                              + eta[b] * L[2 * b, 2 * b + 1] + eta[g] * L[g0, g1]
                              + _t_open_shell(J, K, L, G, am, bn)
                              + _g_mm(G, omega, eta, a, b)
                              + _g_mm(G, omega, eta, a, g)
                              + _g_mm(G, omega, eta, b, g)
                              + 0.5 * G[ax, bx]
                              - 0.5 * G[ax, g0] - 0.5 * G[ax, g1]
                              - 0.5 * G[bx, g0] - 0.5 * G[bx, g1])
                    part_c = (eps[g0] + eps[g1] - eps[ax] - eps[bx]
                              - _g_minus(G, omega, eta, a, g0)
                              - _g_minus(G, omega, eta, a, g1)
                              - _g_minus(G, omega, eta, b, g0)
                              - _g_minus(G, omega, eta, b, g1)
                              + _g_minus(G, omega, eta, b, ax)
                              + _g_minus(G, omega, eta, g, ax)
                              + _g_minus(G, omega, eta, a, bx)
                              + _g_minus(G, omega, eta, g, bx))
                    # open-shell pair (a_mu, b_nu) scattered into a filled unit g
                    d_e = part_b + part_c + 2.0 * G[g0, g1]
                    coup = (0.5 * (-1.0) ** (mu + nu + 1) * sn[am] * sn[bn]
                            * (V[am, g1, bn, g1] * sn[g0] - V[am, g0, bn, g0] * sn[g1]))
                    terms.append((d_e, coup))
                    # unit g emptied into an open-shell pair (a_mu, b_nu)
                    d_e = (part_b - part_c + G[2 * a, 2 * a + 1]
                           + G[2 * b, 2 * b + 1])
                    coup = (0.5 * (-1.0) ** (mu + nu) * sn[ax] * sn[bx]
                            * (V[g0, am, g0, bn] * sn[g0] - V[g1, am, g1, bn] * sn[g1]))
                    terms.append((d_e, coup))
    for a, b in itertools.combinations(range(M), 2):
        rest = [u for u in range(M) if u not in (a, b)]
        for g, d in itertools.combinations(rest, 2):
            for mu, nu, lam, kap in itertools.product((0, 1), repeat=4):
                am, ax = 2 * a + mu, 2 * a + 1 - mu
                bn, bx = 2 * b + nu, 2 * b + 1 - nu
                gl, gx = 2 * g + lam, 2 * g + 1 - lam
                dk, dx = 2 * d + kap, 2 * d + 1 - kap
                d_e = (eta[a] * L[2 * a, 2 * a + 1] + eta[b] * L[2 * b, 2 * b + 1]
                       + eta[g] * L[2 * g, 2 * g + 1] + eta[d] * L[2 * d, 2 * d + 1]
                       + _t_open_shell(J, K, L, G, am, bn)
                       + _t_open_shell(J, K, L, G, gl, dk)
                       + eps[gx] + eps[dx] - eps[ax] - eps[bx]
                       + G[2 * g, 2 * g + 1] + G[2 * d, 2 * d + 1]
                       + _g_mm(G, omega, eta, a, b) + _g_mm(G, omega, eta, a, g)
                       + _g_mm(G, omega, eta, a, d) + _g_mm(G, omega, eta, b, g)
                       + _g_mm(G, omega, eta, b, d) + _g_mm(G, omega, eta, g, d)
                       + 0.5 * G[ax, bx] + 0.5 * G[gx, dx]
                       - _g_minus(G, omega, eta, a, gx) - _g_minus(G, omega, eta, b, gx)
                       - _g_minus(G, omega, eta, d, gx) - _g_minus(G, omega, eta, a, dx)
                       - _g_minus(G, omega, eta, b, dx) - _g_minus(G, omega, eta, g, dx)
                       + _g_minus(G, omega, eta, b, ax) + _g_minus(G, omega, eta, g, ax)
                       + _g_minus(G, omega, eta, d, ax) + _g_minus(G, omega, eta, a, bx)
                       + _g_minus(G, omega, eta, g, bx) + _g_minus(G, omega, eta, d, bx)
                       - 0.5 * G[ax, gx] - 0.5 * G[ax, dx]
                       - 0.5 * G[bx, gx] - 0.5 * G[bx, dx])
                root = sn[am] * sn[bn] * sn[gx] * sn[dx]
                phase = (-1.0) ** (mu + nu + lam + kap)
                coup = 0.25 * (-phase) * root * (V[am, gl, bn, dk] + V[am, dk, bn, gl])
                terms.append((d_e, coup))
                d_e5 = d_e + (K[am, gl] + K[am, dk] + K[bn, gl] + K[bn, dk]
                              - 2.0 * K[am, bn] - 2.0 * K[gl, dk])
                coup5 = (np.sqrt(3.0) / 4.0) * phase * root * (
                    V[am, gl, bn, dk] - V[am, dk, bn, gl])
                terms.append((d_e5, coup5))
    return terms

def _en2_sum(terms):
    total = 0.0
    for d_e, coup in terms:
        if d_e == 0.0:
            raise ValueError("vanishing excitation energy in a perturbative denominator")
        total -= coup * coup / d_e
    return float(total)

def pair_transfer_correction(h, V, omega):
    """Reference implementation. Deterministic."""
    h, V, M = _validate_system(h, V)
    omega = _validate_gaps(omega, M)
    return _en2_sum(_pair_transfer_terms(h, V, omega))

import itertools

import numpy as np

def _validate_system(h, V):
    """Validate the integral arrays; return (h, V, M) with M bond units."""
    h = np.asarray(h, dtype=float)
    V = np.asarray(V, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("h must be a square matrix")
    n_orb = h.shape[0]
    if n_orb < 2 or n_orb % 2 != 0:
        raise ValueError("h must have even dimension 2M with M >= 1")
    if V.shape != (n_orb, n_orb, n_orb, n_orb):
        raise ValueError("V must have shape (2M, 2M, 2M, 2M) matching h")
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(V))):
        raise ValueError("integrals must be finite")
    if not np.allclose(h, h.T, rtol=0.0, atol=1e-10):
        raise ValueError("h must be symmetric")
    for perm in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        if not np.allclose(V, np.transpose(V, perm), rtol=0.0, atol=1e-10):
            raise ValueError("V must have the 8-fold real-orbital permutation symmetry")
    return h, V, n_orb // 2

def _validate_gaps(omega, M):
    omega = np.asarray(omega, dtype=float)
    if omega.shape != (M,):
        raise ValueError("omega must be a length-M vector of bond-unit gaps")
    if not np.all(np.isfinite(omega)):
        raise ValueError("omega must be finite")
    return omega

def _pair_integrals(V):
    """Direct J_pq = V_ppqq, exchange K_pq = V_pqqp, pair-transfer L_pq = V_pqpq,
    and the combination G = 2J - K."""
    idx = np.arange(V.shape[0])
    J = V[idx[:, None], idx[:, None], idx[None, :], idx[None, :]]
    K = V[idx[:, None], idx[None, :], idx[None, :], idx[:, None]]
    L = V[idx[:, None], idx[None, :], idx[:, None], idx[None, :]]
    return J, K, L, 2.0 * J - K

def _occupations(omega):
    """Occupations n_{a,mu} = 1 + (-1)^mu * w_a/eta_a, eta_a = sqrt(w_a^2 + 1)."""
    omega = np.asarray(omega, dtype=float)
    eta = np.sqrt(omega * omega + 1.0)
    n = np.empty(2 * omega.size)
    n[0::2] = 1.0 + omega / eta
    n[1::2] = 1.0 - omega / eta
    return n, eta

def _orbital_energies(h, L, G, n):
    """eps_p = h_pp + L_pp/2 + (1/2) sum_{q outside p's bond unit} G_pq n_q."""
    n_orb = h.shape[0]
    eps = np.empty(n_orb)
    for p in range(n_orb):
        s = h[p, p] + 0.5 * L[p, p]
        for q in range(n_orb):
            if q // 2 != p // 2:
                s += 0.5 * G[p, q] * n[q]
        eps[p] = s
    return eps

def _reference_energy(h, V, omega):
    """E_ref = sum_p eps_p n_p - sum_a L_{a0,a1}/eta_a
    - (1/2) sum'_{p<q} G_pq n_p n_q  (' excludes same-unit pairs)."""
    _, _, L, G = _pair_integrals(V)
    M = h.shape[0] // 2
    n, eta = _occupations(omega)
    eps = _orbital_energies(h, L, G, n)
    energy = float(np.dot(eps, n))
    for a in range(M):
        energy -= L[2 * a, 2 * a + 1] / eta[a]
    for p in range(2 * M):
        for q in range(p + 1, 2 * M):
            if q // 2 != p // 2:
                energy -= 0.5 * G[p, q] * n[p] * n[q]
    return energy

def _t_open_shell(J, K, L, G, p, q):
    """Blocked-singlet placement energy t_pq: J_pq + K_pq - L_pp/2 - L_qq/2,
    with an extra -G_pq/2 when p and q sit in different bond units."""
    val = J[p, q] + K[p, q] - 0.5 * L[p, p] - 0.5 * L[q, q]
    if p // 2 != q // 2:
        val -= 0.5 * G[p, q]
    return val

def _g_minus(G, omega, eta, a, p):
    """g-_{a,p} = (w_a / (2 eta_a)) (G_{a0,p} - G_{a1,p})."""
    return 0.5 * (omega[a] / eta[a]) * (G[2 * a, p] - G[2 * a + 1, p])

def _g_mm(G, omega, eta, a, b):
    """g--_{a,b} = (w_a w_b / (2 eta_a eta_b))
    (G_{a0,b0} - G_{a0,b1} - G_{a1,b0} + G_{a1,b1})."""
    return 0.5 * (omega[a] * omega[b] / (eta[a] * eta[b])) * (
        G[2 * a, 2 * b] - G[2 * a, 2 * b + 1]
        - G[2 * a + 1, 2 * b] + G[2 * a + 1, 2 * b + 1])

def _g_sym(G, a, b):
    """g_{a,b} = (G_{a0,b0} + G_{a0,b1} + G_{a1,b0} + G_{a1,b1}) / 2."""
    return 0.5 * (G[2 * a, 2 * b] + G[2 * a, 2 * b + 1]
                  + G[2 * a + 1, 2 * b] + G[2 * a + 1, 2 * b + 1])

def _dressed_matrix(h, V, omega):
    """CI matrix over {reference} + all complementary double splits."""
    J, K, L, G = _pair_integrals(V)
    M = h.shape[0] // 2
    n, eta = _occupations(omega)
    sn = np.sqrt(n)
    e_ref = _reference_energy(h, V, omega)
    pairs = list(itertools.combinations(range(M), 2))
    dim = 1 + len(pairs)
    mat = np.zeros((dim, dim))
    mat[0, 0] = e_ref
    for i, (a, b) in enumerate(pairs, start=1):
        a0, a1, b0, b1 = 2 * a, 2 * a + 1, 2 * b, 2 * b + 1
        d_e = (eta[a] * L[a0, a1] + eta[b] * L[b0, b1]
               + _t_open_shell(J, K, L, G, a0, a1)
               + _t_open_shell(J, K, L, G, b0, b1)
               + _g_mm(G, omega, eta, a, b)
               + K[a0, b0] + K[a0, b1] + K[a1, b0] + K[a1, b1]
               - 2.0 * K[a0, a1] - 2.0 * K[b0, b1])
        mat[i, i] = e_ref + d_e
        coup = (-0.5 * np.sqrt(3.0) * (sn[a0] * sn[b0] + sn[a1] * sn[b1])
                * V[a0, b1, b0, a1]
                - 0.5 * np.sqrt(3.0) * (sn[a1] * sn[b0] + sn[a0] * sn[b1])
                * V[a1, b1, b0, a0])
        mat[0, i] = mat[i, 0] = coup
    for i, pi in enumerate(pairs, start=1):
        for j, pj in enumerate(pairs, start=1):
            if j <= i:
                continue
            shared = set(pi) & set(pj)
            if len(shared) != 1:
                continue
            be = (set(pi) - shared).pop()
            ga = (set(pj) - shared).pop()
            b0, b1 = 2 * be, 2 * be + 1
            g0, g1 = 2 * ga, 2 * ga + 1
            val = (-0.5 * (sn[b0] * sn[g1] + sn[b1] * sn[g0]) * V[g0, b1, b0, g1]
                   - 0.5 * (sn[b0] * sn[g0] + sn[b1] * sn[g1]) * V[g0, b0, b1, g1])
            mat[i, j] = mat[j, i] = val
    return mat

def dressed_reference_energy(h, V, omega):
    """Reference implementation. Deterministic."""
    h, V, M = _validate_system(h, V)
    omega = _validate_gaps(omega, M)
    mat = _dressed_matrix(h, V, omega)
    return float(np.linalg.eigvalsh(mat)[0])

import itertools

import numpy as np

def _validate_system(h, V):
    """Validate the integral arrays; return (h, V, M) with M bond units."""
    h = np.asarray(h, dtype=float)
    V = np.asarray(V, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("h must be a square matrix")
    n_orb = h.shape[0]
    if n_orb < 2 or n_orb % 2 != 0:
        raise ValueError("h must have even dimension 2M with M >= 1")
    if V.shape != (n_orb, n_orb, n_orb, n_orb):
        raise ValueError("V must have shape (2M, 2M, 2M, 2M) matching h")
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(V))):
        raise ValueError("integrals must be finite")
    if not np.allclose(h, h.T, rtol=0.0, atol=1e-10):
        raise ValueError("h must be symmetric")
    for perm in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        if not np.allclose(V, np.transpose(V, perm), rtol=0.0, atol=1e-10):
            raise ValueError("V must have the 8-fold real-orbital permutation symmetry")
    return h, V, n_orb // 2

def _validate_gaps(omega, M):
    omega = np.asarray(omega, dtype=float)
    if omega.shape != (M,):
        raise ValueError("omega must be a length-M vector of bond-unit gaps")
    if not np.all(np.isfinite(omega)):
        raise ValueError("omega must be finite")
    return omega

def _pair_integrals(V):
    """Direct J_pq = V_ppqq, exchange K_pq = V_pqqp, pair-transfer L_pq = V_pqpq,
    and the combination G = 2J - K."""
    idx = np.arange(V.shape[0])
    J = V[idx[:, None], idx[:, None], idx[None, :], idx[None, :]]
    K = V[idx[:, None], idx[None, :], idx[None, :], idx[:, None]]
    L = V[idx[:, None], idx[None, :], idx[:, None], idx[None, :]]
    return J, K, L, 2.0 * J - K

def _occupations(omega):
    """Occupations n_{a,mu} = 1 + (-1)^mu * w_a/eta_a, eta_a = sqrt(w_a^2 + 1)."""
    omega = np.asarray(omega, dtype=float)
    eta = np.sqrt(omega * omega + 1.0)
    n = np.empty(2 * omega.size)
    n[0::2] = 1.0 + omega / eta
    n[1::2] = 1.0 - omega / eta
    return n, eta

def _orbital_energies(h, L, G, n):
    """eps_p = h_pp + L_pp/2 + (1/2) sum_{q outside p's bond unit} G_pq n_q."""
    n_orb = h.shape[0]
    eps = np.empty(n_orb)
    for p in range(n_orb):
        s = h[p, p] + 0.5 * L[p, p]
        for q in range(n_orb):
            if q // 2 != p // 2:
                s += 0.5 * G[p, q] * n[q]
        eps[p] = s
    return eps

def _relax_bond_gaps(h, V, alpha=0.55, n_sweeps=5):
    """Exactly n_sweeps damped Jacobi sweeps of
    w_a <- w_a + alpha * ((eps_{a1} - eps_{a0}) / L_{a0,a1} - w_a) from w = 0."""
    _, _, L, G = _pair_integrals(V)
    M = h.shape[0] // 2
    l_intra = np.array([L[2 * a, 2 * a + 1] for a in range(M)])
    if np.any(np.abs(l_intra) < 1e-12):
        raise ValueError("an intra-unit pair-transfer integral L_{a0,a1} vanishes")
    omega = np.zeros(M)
    for _ in range(int(n_sweeps)):
        n, _ = _occupations(omega)
        eps = _orbital_energies(h, L, G, n)
        omega = omega + alpha * ((eps[1::2] - eps[0::2]) / l_intra - omega)
    return omega

def _reference_energy(h, V, omega):
    """E_ref = sum_p eps_p n_p - sum_a L_{a0,a1}/eta_a
    - (1/2) sum'_{p<q} G_pq n_p n_q  (' excludes same-unit pairs)."""
    _, _, L, G = _pair_integrals(V)
    M = h.shape[0] // 2
    n, eta = _occupations(omega)
    eps = _orbital_energies(h, L, G, n)
    energy = float(np.dot(eps, n))
    for a in range(M):
        energy -= L[2 * a, 2 * a + 1] / eta[a]
    for p in range(2 * M):
        for q in range(p + 1, 2 * M):
            if q // 2 != p // 2:
                energy -= 0.5 * G[p, q] * n[p] * n[q]
    return energy

def _generalized_fock(h, V, omega):
    """f_pq = h_pq n_p + (1/2) sum_r (2 V_rrpq - V_rqpr) D_rp + sum_r V_rprq P_rp,
    with the reference two-body data D_pp = 0, D within a unit = 0, D_rp = n_r n_p
    otherwise, P_pp = n_p, and P_{a0,a1} = P_{a1,a0} = -1/eta_a."""
    n_orb = h.shape[0]
    M = n_orb // 2
    n, eta = _occupations(omega)
    D = np.outer(n, n)
    for a in range(M):
        D[2 * a, 2 * a + 1] = D[2 * a + 1, 2 * a] = 0.0
    np.fill_diagonal(D, 0.0)
    P = np.zeros((n_orb, n_orb))
    np.fill_diagonal(P, n)
    for a in range(M):
        P[2 * a, 2 * a + 1] = P[2 * a + 1, 2 * a] = -1.0 / eta[a]
    f = np.zeros((n_orb, n_orb))
    for p in range(n_orb):
        for q in range(n_orb):
            s = h[p, q] * n[p]
            for r in range(n_orb):
                s += 0.5 * (2.0 * V[r, r, p, q] - V[r, q, p, r]) * D[r, p]
                s += V[r, p, r, q] * P[r, p]
            f[p, q] = s
    return f

def _t_open_shell(J, K, L, G, p, q):
    """Blocked-singlet placement energy t_pq: J_pq + K_pq - L_pp/2 - L_qq/2,
    with an extra -G_pq/2 when p and q sit in different bond units."""
    val = J[p, q] + K[p, q] - 0.5 * L[p, p] - 0.5 * L[q, q]
    if p // 2 != q // 2:
        val -= 0.5 * G[p, q]
    return val

def _g_minus(G, omega, eta, a, p):
    """g-_{a,p} = (w_a / (2 eta_a)) (G_{a0,p} - G_{a1,p})."""
    return 0.5 * (omega[a] / eta[a]) * (G[2 * a, p] - G[2 * a + 1, p])

def _g_mm(G, omega, eta, a, b):
    """g--_{a,b} = (w_a w_b / (2 eta_a eta_b))
    (G_{a0,b0} - G_{a0,b1} - G_{a1,b0} + G_{a1,b1})."""
    return 0.5 * (omega[a] * omega[b] / (eta[a] * eta[b])) * (
        G[2 * a, 2 * b] - G[2 * a, 2 * b + 1]
        - G[2 * a + 1, 2 * b] + G[2 * a + 1, 2 * b + 1])

def _g_sym(G, a, b):
    """g_{a,b} = (G_{a0,b0} + G_{a0,b1} + G_{a1,b0} + G_{a1,b1}) / 2."""
    return 0.5 * (G[2 * a, 2 * b] + G[2 * a, 2 * b + 1]
                  + G[2 * a + 1, 2 * b] + G[2 * a + 1, 2 * b + 1])

def _single_terms(h, V, omega, fock=None):
    """(excitation energy, coupling) for every single excitation."""
    J, K, L, G = _pair_integrals(V)
    M = h.shape[0] // 2
    n, eta = _occupations(omega)
    eps = _orbital_energies(h, L, G, n)
    f = _generalized_fock(h, V, omega) if fock is None else np.asarray(fock, dtype=float)
    sn = np.sqrt(n)
    terms = []
    for a in range(M):
        a0, a1 = 2 * a, 2 * a + 1
        eta_l = eta[a] * L[a0, a1]
        # occupation swap within unit a
        terms.append((2.0 * eta_l,
                      (eps[a0] - eps[a1] + omega[a] * L[a0, a1]) / eta[a]))
        # pair split within unit a
        terms.append((eta_l + _t_open_shell(J, K, L, G, a0, a1),
                      (f[a0, a1] - f[a1, a0]) / (sn[a0] + sn[a1])))
    for a, b in itertools.permutations(range(M), 2):
        for mu in (0, 1):
            for nu in (0, 1):
                am, ax = 2 * a + mu, 2 * a + 1 - mu
                bn, bx = 2 * b + nu, 2 * b + 1 - nu
                d_e = (eta[a] * L[2 * a, 2 * a + 1] + eta[b] * L[2 * b, 2 * b + 1]
                       + _t_open_shell(J, K, L, G, am, bn)
                       + eps[bx] - eps[ax] + G[2 * b, 2 * b + 1]
                       + _g_mm(G, omega, eta, a, b)
                       - _g_minus(G, omega, eta, a, bx)
                       + _g_minus(G, omega, eta, b, ax)
                       - 0.5 * G[ax, bx])
                rhs = (f[am, bn]
                       + n[am] * np.sqrt(n[bn] / n[bx]) * V[bn, bx, am, bx]
                       + 0.5 * (2.0 * V[bx, bx, am, bn] - V[bx, bn, am, bx]
                                - V[bn, bn, am, bn]) * n[am] * n[bn])
                coup = (-1.0) ** (mu + nu + 1) * np.sqrt(n[bx] / (2.0 * n[am])) * rhs
                terms.append((d_e, coup))
    return terms

def _double_terms(h, V, omega):
    """(excitation energy, coupling) for every double excitation EXCEPT the
    complementary double splits (those are dressed variationally instead)."""
    J, K, L, G = _pair_integrals(V)
    M = h.shape[0] // 2
    n, eta = _occupations(omega)
    eps = _orbital_energies(h, L, G, n)
    sn = np.sqrt(n)
    terms = []
    for a, b in itertools.combinations(range(M), 2):
        a0, a1, b0, b1 = 2 * a, 2 * a + 1, 2 * b, 2 * b + 1
        eta_la = eta[a] * L[a0, a1]
        eta_lb = eta[b] * L[b0, b1]
        gmm_ab = _g_mm(G, omega, eta, a, b)
        # double swap
        coup = sum((-1.0) ** (mu + nu) * G[2 * a + mu, 2 * b + nu]
                   for mu in (0, 1) for nu in (0, 1)) / (2.0 * eta[a] * eta[b])
        terms.append((2.0 * eta_la + 2.0 * eta_lb + 4.0 * gmm_ab, coup))
        # double split (its complementary partner is dressed, not summed)
        d_e = (eta_la + eta_lb + _t_open_shell(J, K, L, G, a0, a1)
               + _t_open_shell(J, K, L, G, b0, b1) + gmm_ab)
        coup = ((sn[a0] - sn[a1]) * (sn[b0] - sn[b1]) * V[a0, a1, b0, b1]
                - 0.5 * (sn[a0] * sn[b0] + sn[a1] * sn[b1]) * V[a0, b1, b0, a1]
                + 0.5 * (sn[a1] * sn[b0] + sn[a0] * sn[b1]) * V[a1, b1, b0, a0])
        terms.append((d_e, coup))
    for a, b in itertools.permutations(range(M), 2):
        b0, b1 = 2 * b, 2 * b + 1
        # swap in unit a + split in unit b
        d_e = (2.0 * eta[a] * L[2 * a, 2 * a + 1] + eta[b] * L[b0, b1]
               + _t_open_shell(J, K, L, G, b0, b1)
               + 2.0 * _g_mm(G, omega, eta, a, b))
        coup = (sn[b0] - sn[b1]) * sum(
            (-1.0) ** mu * (2.0 * V[2 * a + mu, 2 * a + mu, b0, b1]
                            - V[2 * a + mu, b1, b0, 2 * a + mu])
            for mu in (0, 1)) / (2.0 * eta[a])
        terms.append((d_e, coup))
    for a in range(M):
        others = [u for u in range(M) if u != a]
        a0, a1 = 2 * a, 2 * a + 1
        for b, g in itertools.permutations(others, 2):
            for nu in (0, 1):
                for lam in (0, 1):
                    bn, bx = 2 * b + nu, 2 * b + 1 - nu
                    gl, gx = 2 * g + lam, 2 * g + 1 - lam
                    common = (eta[b] * L[2 * b, 2 * b + 1]
                              + eta[g] * L[2 * g, 2 * g + 1]
                              + _t_open_shell(J, K, L, G, bn, gl)
                              + eps[gx] - eps[bx] + G[2 * g, 2 * g + 1]
                              - 0.5 * G[bx, gx])
                    gmm_ab = _g_mm(G, omega, eta, a, b)
                    gmm_ag = _g_mm(G, omega, eta, a, g)
                    gmm_bg = _g_mm(G, omega, eta, b, g)
                    gm_agx = _g_minus(G, omega, eta, a, gx)
                    gm_bgx = _g_minus(G, omega, eta, b, gx)
                    gm_abx = _g_minus(G, omega, eta, a, bx)
                    gm_gbx = _g_minus(G, omega, eta, g, bx)
                    # swap in a + transfer b_nu -> g_lam
                    d_e = (2.0 * eta[a] * L[a0, a1] + common
                           + 2.0 * gmm_ab + 2.0 * gmm_ag + gmm_bg
                           - 2.0 * gm_agx - gm_bgx + 2.0 * gm_abx + gm_gbx)
                    ssum = sum((-1.0) ** mu * (2.0 * V[bn, gl, 2 * a + mu, 2 * a + mu]
                                               - V[2 * a + mu, gl, bn, 2 * a + mu])
                               for mu in (0, 1))
                    coup = ((-1.0) ** (nu + lam + 1)
                            / (2.0 * np.sqrt(2.0) * eta[a])
                            * np.sqrt(n[bn] * n[gx]) * ssum)
                    terms.append((d_e, coup))
                    # split in a + transfer b_nu -> g_lam
                    d_e6 = (eta[a] * L[a0, a1] + _t_open_shell(J, K, L, G, a0, a1)
                            + common + gmm_ab + gmm_ag + gmm_bg
                            - gm_agx - gm_bgx + gm_abx + gm_gbx)
                    ssum6 = sum((-1.0) ** mu * sn[2 * a + mu]
                                * (2.0 * V[bn, gl, 2 * a + mu, 2 * a + 1 - mu]
                                   - V[2 * a + mu, gl, bn, 2 * a + 1 - mu])
                                for mu in (0, 1))
                    coup6 = ((-1.0) ** (nu + lam + 1) / (2.0 * np.sqrt(2.0))
                             * np.sqrt(n[bn] * n[gx]) * ssum6)
                    terms.append((d_e6, coup6))
                    # complementary split + transfer
                    d_e7 = d_e6 + (K[a0, bn] + K[a0, gl] + K[a1, bn] + K[a1, gl]
                                   - 2.0 * K[a0, a1] - 2.0 * K[bn, gl])
                    ssum7 = sum(sn[2 * a + mu] * V[2 * a + mu, gl, bn, 2 * a + 1 - mu]
                                for mu in (0, 1))
                    coup7 = ((-1.0) ** (nu + lam) * 0.5 * np.sqrt(1.5)
                             * np.sqrt(n[bn] * n[gx]) * ssum7)
                    terms.append((d_e7, coup7))
    return terms

def _pair_transfer_terms(h, V, omega):
    """(excitation energy, coupling) for every correlated pair-transfer state."""
    J, K, L, G = _pair_integrals(V)
    M = h.shape[0] // 2
    n, eta = _occupations(omega)
    eps = _orbital_energies(h, L, G, n)
    sn = np.sqrt(n)
    terms = []
    for a, b in itertools.permutations(range(M), 2):
        a0, a1, b0, b1 = 2 * a, 2 * a + 1, 2 * b, 2 * b + 1
        d_e = (eta[a] * L[a0, a1] + eta[b] * L[b0, b1]
               + eps[b0] + eps[b1] - eps[a0] - eps[a1] + 2.0 * G[b0, b1]
               - _g_sym(G, a, b) + _g_mm(G, omega, eta, a, b)
               - _g_minus(G, omega, eta, a, b0) - _g_minus(G, omega, eta, a, b1)
               + _g_minus(G, omega, eta, b, a0) + _g_minus(G, omega, eta, b, a1))
        coup = 0.5 * sum((-1.0) ** (mu + nu + 1) * L[2 * a + mu, 2 * b + nu]
                         * sn[2 * a + mu] * sn[2 * b + 1 - nu]
                         for mu in (0, 1) for nu in (0, 1))
        terms.append((d_e, coup))
    for a, b in itertools.combinations(range(M), 2):
        for mu in (0, 1):
            for nu in (0, 1):
                am, ax = 2 * a + mu, 2 * a + 1 - mu
                bn, bx = 2 * b + nu, 2 * b + 1 - nu
                for g in range(M):
                    if g in (a, b):
                        continue
                    g0, g1 = 2 * g, 2 * g + 1
                    part_b = (eta[a] * L[2 * a, 2 * a + 1]
                              + eta[b] * L[2 * b, 2 * b + 1] + eta[g] * L[g0, g1]
                              + _t_open_shell(J, K, L, G, am, bn)
                              + _g_mm(G, omega, eta, a, b)
                              + _g_mm(G, omega, eta, a, g)
                              + _g_mm(G, omega, eta, b, g)
                              + 0.5 * G[ax, bx]
                              - 0.5 * G[ax, g0] - 0.5 * G[ax, g1]
                              - 0.5 * G[bx, g0] - 0.5 * G[bx, g1])
                    part_c = (eps[g0] + eps[g1] - eps[ax] - eps[bx]
                              - _g_minus(G, omega, eta, a, g0)
                              - _g_minus(G, omega, eta, a, g1)
                              - _g_minus(G, omega, eta, b, g0)
                              - _g_minus(G, omega, eta, b, g1)
                              + _g_minus(G, omega, eta, b, ax)
                              + _g_minus(G, omega, eta, g, ax)
                              + _g_minus(G, omega, eta, a, bx)
                              + _g_minus(G, omega, eta, g, bx))
                    # open-shell pair (a_mu, b_nu) scattered into a filled unit g
                    d_e = part_b + part_c + 2.0 * G[g0, g1]
                    coup = (0.5 * (-1.0) ** (mu + nu + 1) * sn[am] * sn[bn]
                            * (V[am, g1, bn, g1] * sn[g0] - V[am, g0, bn, g0] * sn[g1]))
                    terms.append((d_e, coup))
                    # unit g emptied into an open-shell pair (a_mu, b_nu)
                    d_e = (part_b - part_c + G[2 * a, 2 * a + 1]
                           + G[2 * b, 2 * b + 1])
                    coup = (0.5 * (-1.0) ** (mu + nu) * sn[ax] * sn[bx]
                            * (V[g0, am, g0, bn] * sn[g0] - V[g1, am, g1, bn] * sn[g1]))
                    terms.append((d_e, coup))
    for a, b in itertools.combinations(range(M), 2):
        rest = [u for u in range(M) if u not in (a, b)]
        for g, d in itertools.combinations(rest, 2):
            for mu, nu, lam, kap in itertools.product((0, 1), repeat=4):
                am, ax = 2 * a + mu, 2 * a + 1 - mu
                bn, bx = 2 * b + nu, 2 * b + 1 - nu
                gl, gx = 2 * g + lam, 2 * g + 1 - lam
                dk, dx = 2 * d + kap, 2 * d + 1 - kap
                d_e = (eta[a] * L[2 * a, 2 * a + 1] + eta[b] * L[2 * b, 2 * b + 1]
                       + eta[g] * L[2 * g, 2 * g + 1] + eta[d] * L[2 * d, 2 * d + 1]
                       + _t_open_shell(J, K, L, G, am, bn)
                       + _t_open_shell(J, K, L, G, gl, dk)
                       + eps[gx] + eps[dx] - eps[ax] - eps[bx]
                       + G[2 * g, 2 * g + 1] + G[2 * d, 2 * d + 1]
                       + _g_mm(G, omega, eta, a, b) + _g_mm(G, omega, eta, a, g)
                       + _g_mm(G, omega, eta, a, d) + _g_mm(G, omega, eta, b, g)
                       + _g_mm(G, omega, eta, b, d) + _g_mm(G, omega, eta, g, d)
                       + 0.5 * G[ax, bx] + 0.5 * G[gx, dx]
                       - _g_minus(G, omega, eta, a, gx) - _g_minus(G, omega, eta, b, gx)
                       - _g_minus(G, omega, eta, d, gx) - _g_minus(G, omega, eta, a, dx)
                       - _g_minus(G, omega, eta, b, dx) - _g_minus(G, omega, eta, g, dx)
                       + _g_minus(G, omega, eta, b, ax) + _g_minus(G, omega, eta, g, ax)
                       + _g_minus(G, omega, eta, d, ax) + _g_minus(G, omega, eta, a, bx)
                       + _g_minus(G, omega, eta, g, bx) + _g_minus(G, omega, eta, d, bx)
                       - 0.5 * G[ax, gx] - 0.5 * G[ax, dx]
                       - 0.5 * G[bx, gx] - 0.5 * G[bx, dx])
                root = sn[am] * sn[bn] * sn[gx] * sn[dx]
                phase = (-1.0) ** (mu + nu + lam + kap)
                coup = 0.25 * (-phase) * root * (V[am, gl, bn, dk] + V[am, dk, bn, gl])
                terms.append((d_e, coup))
                d_e5 = d_e + (K[am, gl] + K[am, dk] + K[bn, gl] + K[bn, dk]
                              - 2.0 * K[am, bn] - 2.0 * K[gl, dk])
                coup5 = (np.sqrt(3.0) / 4.0) * phase * root * (
                    V[am, gl, bn, dk] - V[am, dk, bn, gl])
                terms.append((d_e5, coup5))
    return terms

def _dressed_matrix(h, V, omega):
    """CI matrix over {reference} + all complementary double splits."""
    J, K, L, G = _pair_integrals(V)
    M = h.shape[0] // 2
    n, eta = _occupations(omega)
    sn = np.sqrt(n)
    e_ref = _reference_energy(h, V, omega)
    pairs = list(itertools.combinations(range(M), 2))
    dim = 1 + len(pairs)
    mat = np.zeros((dim, dim))
    mat[0, 0] = e_ref
    for i, (a, b) in enumerate(pairs, start=1):
        a0, a1, b0, b1 = 2 * a, 2 * a + 1, 2 * b, 2 * b + 1
        d_e = (eta[a] * L[a0, a1] + eta[b] * L[b0, b1]
               + _t_open_shell(J, K, L, G, a0, a1)
               + _t_open_shell(J, K, L, G, b0, b1)
               + _g_mm(G, omega, eta, a, b)
               + K[a0, b0] + K[a0, b1] + K[a1, b0] + K[a1, b1]
               - 2.0 * K[a0, a1] - 2.0 * K[b0, b1])
        mat[i, i] = e_ref + d_e
        coup = (-0.5 * np.sqrt(3.0) * (sn[a0] * sn[b0] + sn[a1] * sn[b1])
                * V[a0, b1, b0, a1]
                - 0.5 * np.sqrt(3.0) * (sn[a1] * sn[b0] + sn[a0] * sn[b1])
                * V[a1, b1, b0, a0])
        mat[0, i] = mat[i, 0] = coup
    for i, pi in enumerate(pairs, start=1):
        for j, pj in enumerate(pairs, start=1):
            if j <= i:
                continue
            shared = set(pi) & set(pj)
            if len(shared) != 1:
                continue
            be = (set(pi) - shared).pop()
            ga = (set(pj) - shared).pop()
            b0, b1 = 2 * be, 2 * be + 1
            g0, g1 = 2 * ga, 2 * ga + 1
            val = (-0.5 * (sn[b0] * sn[g1] + sn[b1] * sn[g0]) * V[g0, b1, b0, g1]
                   - 0.5 * (sn[b0] * sn[g0] + sn[b1] * sn[g1]) * V[g0, b0, b1, g1])
            mat[i, j] = mat[j, i] = val
    return mat

def _en2_sum(terms):
    total = 0.0
    for d_e, coup in terms:
        if d_e == 0.0:
            raise ValueError("vanishing excitation energy in a perturbative denominator")
        total -= coup * coup / d_e
    return float(total)

def _default_configuration():
    """The fixed benchmark configuration (model pair Hamiltonian, M = 4)."""
    x = np.array([0.00, 0.13, 1.80, 1.92, 3.70, 3.80, 5.72, 5.86])
    tau = np.array([1.000, 0.985, 1.030, 1.015, 0.960, 0.975, 1.020, 1.000])
    amp = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)
    mid = 0.5 * (x[:, None] + x[None, :])
    V = (amp[:, :, None, None] * amp[None, None, :, :]
         / (1.0 + np.abs(mid[:, :, None, None] - mid[None, None, :, :])))
    h = np.array([
        [-1.346712, -2.298396, -0.316906, -0.214117,  0.011275,  0.009836,  0.000986, -0.001170],
        [-2.298396, -0.839405, -0.387397, -0.188980,  0.012257,  0.003279,  0.000939,  0.004003],
        [-0.316906, -0.387397, -1.056783, -2.908410, -0.295847, -0.191579,  0.007895,  0.006869],
        [-0.214117, -0.188980, -2.908410, -0.332489, -0.340433, -0.217742,  0.005055, -0.001203],
        [ 0.011275,  0.012257, -0.295847, -0.340433, -1.294736, -2.407369, -0.204517, -0.124062],
        [ 0.009836,  0.003279, -0.191579, -0.217742, -2.407369, -0.742958, -0.250494, -0.056612],
        [ 0.000986,  0.000939,  0.007895,  0.005055, -0.204517, -0.250494, -1.153974, -2.348799],
        [-0.001170,  0.004003,  0.006869, -0.001203, -0.124062, -0.056612, -2.348799,  0.545281]
    ])
    return h, V

def dressed_valence_correction(h=None, V=None):
    """Reference implementation. Deterministic."""
    if (h is None) != (V is None):
        raise ValueError("provide both h and V, or neither")
    if h is None:
        h, V = _default_configuration()
    h, V, _ = _validate_system(h, V)
    # The gold chains the ORACLE of each earlier sub-problem, so that an incorrect
    # submitted implementation can never influence this reference result. The public
    # dressed_valence_correction chains the public functions instead.
    omega = relax_bond_gaps(h, V)                       # Step 1
    e_ref = reference_pair_energy(h, V, omega)          # Step 2
    fock = pair_generalized_fock(h, V, omega)           # Step 3
    e2 = (single_excitation_correction(h, V, omega, fock)  # Step 4 (consumes Step 3)
          + double_excitation_correction(h, V, omega)      # Step 5
          + pair_transfer_correction(h, V, omega))         # Step 6
    e_dressed = dressed_reference_energy(h, V, omega)   # Step 7
    return float((e_dressed - e_ref) + e2)
SCICODE_GOLD_EOF
