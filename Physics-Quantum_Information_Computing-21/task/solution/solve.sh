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
from scipy.linalg import expm


def _lindblad_superop(H: "np.ndarray", Ls: "list") -> "np.ndarray":
    d = H.shape[0]
    I = np.eye(d)
    Lsup = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for L in Ls:
        Lsup = Lsup + np.kron(L.conj(), L) - 0.5 * (np.kron(I, L.conj().T @ L) + np.kron((L.conj().T @ L).T, I))
    return Lsup


def _apply_superop(Esup: "np.ndarray", rho: "np.ndarray") -> "np.ndarray":
    d = rho.shape[0]
    vecrho = rho.reshape(-1, 1, order="F")
    out = Esup @ vecrho
    return out.reshape(d, d, order="F")


def averaged_lindblad_evolution(H: "np.ndarray", Ls: "np.ndarray", rho0: "np.ndarray", T: float, n: int) -> "np.ndarray":
    if T <= 0:
        raise ValueError("T must be positive")
    if n < 1:
        raise ValueError("n must be at least 1")
    d = H.shape[0]
    dt = T / n
    Lsup = _lindblad_superop(H, list(Ls))
    Esup = expm(Lsup * dt)
    rho = np.array(rho0, dtype=complex)
    traj = [rho.copy()]
    for _ in range(n):
        rho = _apply_superop(Esup, rho)
        rho = (rho + rho.conj().T) / 2
        traj.append(rho.copy())
    return np.array(traj)

import numpy as np


def _build_qutrit_generators(seed: int) -> "tuple":
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    H0 = (A + A.conj().T) / 2
    L0 = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    return H0, L0


def _b_func(A: "np.ndarray", rho: "np.ndarray") -> float:
    return 2 * np.real(np.trace(A.conj().T @ rho))


def _lindblad_drift(rho: "np.ndarray", H: "np.ndarray", L: "np.ndarray") -> "np.ndarray":
    comm = -1j * (H @ rho - rho @ H)
    diss = L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
    return comm + diss


def _proj_physical(rho: "np.ndarray") -> "np.ndarray":
    rho = (rho + rho.conj().T) / 2
    w, v = np.linalg.eigh(rho)
    w = np.clip(w, 0, None)
    s = w.sum()
    if s <= 0:
        d = rho.shape[0]
        return np.eye(d, dtype=complex) / d
    w = w / s
    return (v * w) @ v.conj().T


def _simulate_true_trajectory(H: "np.ndarray", L: "np.ndarray", rho0: "np.ndarray", T: float, n: int, n_sub: int, rng: "np.random.Generator") -> "np.ndarray":
    delta = T / n / n_sub
    rho = np.array(rho0, dtype=complex).copy()
    Y = np.zeros(n)
    for j in range(n):
        y_accum = 0.0
        for _ in range(n_sub):
            b_val = _b_func(L, rho)
            H_back = L @ rho + rho @ L.conj().T - b_val * rho
            dW = rng.standard_normal(1)[0] * np.sqrt(delta)
            y_accum += b_val * delta + dW
            rho_new = rho + delta * _lindblad_drift(rho, H, L) + H_back * dW
            rho = _proj_physical(rho_new)
        Y[j] = y_accum
    return Y


def simulate_averaged_observation_data(seed_gen: int, seed_noise: int, alpha0: float, beta0: float, T: float, n: int, N: int, n_sub: int) -> "np.ndarray":
    if T <= 0:
        raise ValueError("T must be positive")
    if n < 1:
        raise ValueError("n must be at least 1")
    if N < 1:
        raise ValueError("N must be at least 1")
    if n_sub < 1:
        raise ValueError("n_sub must be at least 1")
    H0, L0 = _build_qutrit_generators(seed_gen)
    H = alpha0 * H0
    L = beta0 * L0
    rho0 = np.zeros((3, 3), dtype=complex)
    rho0[0, 0] = 1.0
    rng_noise = np.random.default_rng(seed_noise)
    Y = np.zeros((N, n))
    for i in range(N):
        Y[i] = _simulate_true_trajectory(H, L, rho0, T, n, n_sub, rng_noise)
    return Y.mean(axis=0)

import numpy as np
from scipy.optimize import least_squares


def _build_qutrit_generators(seed: int) -> "tuple":
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    H0 = (A + A.conj().T) / 2
    L0 = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    return H0, L0


def _b_func(A: "np.ndarray", rho: "np.ndarray") -> float:
    return 2 * np.real(np.trace(A.conj().T @ rho))


def _contrast_and_h(theta: "np.ndarray", H0: "np.ndarray", L0: "np.ndarray", Y_avg: "np.ndarray", T: float, n: int, rho0: "np.ndarray") -> "tuple":
    alpha, beta = theta
    H = alpha * H0
    L = beta * L0
    traj = averaged_lindblad_evolution(H, np.array([L]), rho0, T, n)
    h = np.array([_b_func(L, traj[j]) for j in range(n)])
    return h


def _contrast_value(theta: "np.ndarray", H0: "np.ndarray", L0: "np.ndarray", Y_avg: "np.ndarray", T: float, n: int, rho0: "np.ndarray") -> float:
    dt = T / n
    h = _contrast_and_h(theta, H0, L0, Y_avg, T, n, rho0)
    return float(np.sum(h * Y_avg - 0.5 * h ** 2 * dt))


def _residuals(theta: "np.ndarray", H0: "np.ndarray", L0: "np.ndarray", Y_avg: "np.ndarray", T: float, n: int, rho0: "np.ndarray") -> "np.ndarray":
    dt = T / n
    h = _contrast_and_h(theta, H0, L0, Y_avg, T, n, rho0)
    return np.sqrt(dt) * (h - Y_avg / dt)


def discrete_contrast_estimator(seed_gen: int, Y_avg: "np.ndarray", T: float, n: int) -> "np.ndarray":
    if T <= 0:
        raise ValueError("T must be positive")
    if n < 1:
        raise ValueError("n must be at least 1")
    if Y_avg.shape[0] != n:
        raise ValueError("Y_avg must have length n")
    H0, L0 = _build_qutrit_generators(seed_gen)
    rho0 = np.zeros((3, 3), dtype=complex)
    rho0[0, 0] = 1.0

    lo, hi = 0.0, 2.0
    grid_step = 0.1
    vals = np.round(np.arange(lo, hi + 1e-9, grid_step), 10)
    C = np.zeros((len(vals), len(vals)))
    for i, a in enumerate(vals):
        for j, b in enumerate(vals):
            C[i, j] = _contrast_value((a, b), H0, L0, Y_avg, T, n, rho0)

    candidates = []
    ni, nj = C.shape
    for i in range(ni):
        for j in range(nj):
            v = C[i, j]
            is_max = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ii, jj = i + di, j + dj
                    if 0 <= ii < ni and 0 <= jj < nj and C[ii, jj] > v:
                        is_max = False
            if is_max:
                candidates.append((v, vals[i], vals[j]))
    candidates.sort(key=lambda x: -x[0])

    selected = []
    for v, a, b in candidates:
        if all(np.hypot(a - sa, b - sb) >= 0.15 for _, sa, sb in selected):
            selected.append((v, a, b))
        if len(selected) >= 8:
            break
    if not selected:
        i, j = np.unravel_index(np.argmax(C), C.shape)
        selected = [(C[i, j], vals[i], vals[j])]

    best_theta = None
    best_val = -np.inf
    for _, a0, b0 in selected:
        sol = least_squares(_residuals, x0=[a0, b0], bounds=([lo, lo], [hi, hi]), method="trf",
                             args=(H0, L0, Y_avg, T, n, rho0))
        theta = sol.x
        val = _contrast_value(theta, H0, L0, Y_avg, T, n, rho0)
        if val > best_val:
            best_val = val
            best_theta = theta
    return np.array(best_theta)

import numpy as np
from scipy.linalg import expm


def _build_qutrit_generators(seed: int) -> "tuple":
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    H0 = (A + A.conj().T) / 2
    L0 = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    return H0, L0


def _lindblad_superop(H: "np.ndarray", Ls: "list") -> "np.ndarray":
    d = H.shape[0]
    I = np.eye(d)
    Lsup = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for L in Ls:
        Lsup = Lsup + np.kron(L.conj(), L) - 0.5 * (np.kron(I, L.conj().T @ L) + np.kron((L.conj().T @ L).T, I))
    return Lsup


def _apply_superop(Esup: "np.ndarray", rho: "np.ndarray") -> "np.ndarray":
    d = rho.shape[0]
    vecrho = rho.reshape(-1, 1, order="F")
    out = Esup @ vecrho
    return out.reshape(d, d, order="F")


def channel_from_lindbladian(seed_gen: int, alpha_hat: float, beta_hat: float, tau: float) -> "np.ndarray":
    if tau <= 0:
        raise ValueError("tau must be positive")
    H0, L0 = _build_qutrit_generators(seed_gen)
    H_hat = alpha_hat * H0
    L_hat = beta_hat * L0
    d = 3
    Lsup = _lindblad_superop(H_hat, [L_hat])
    Esup = expm(tau * Lsup)

    J = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            Eij = np.zeros((d, d), dtype=complex)
            Eij[i, j] = 1.0
            out = _apply_superop(Esup, Eij)
            J[i * d:(i + 1) * d, j * d:(j + 1) * d] = out
    J = (J + J.conj().T) / 2
    w, v = np.linalg.eigh(J)
    w_clipped = np.clip(w, 0, None)

    Ks = []
    for idx in range(len(w_clipped)):
        if w_clipped[idx] > 1e-8:
            vec = v[:, idx]
            K = np.sqrt(w_clipped[idx]) * vec.reshape(d, d, order="F")
            Ks.append(K)
    return np.array(Ks)

import numpy as np


def entanglement_negativity(rho: "np.ndarray", dA: int, dB: int) -> float:
    if rho.shape != (dA * dB, dA * dB):
        raise ValueError("rho must have shape (dA*dB, dA*dB)")
    arr = rho.reshape(dA, dB, dA, dB)
    pt = np.transpose(arr, (0, 3, 2, 1)).reshape(dA * dB, dA * dB)
    w = np.linalg.eigvalsh((pt + pt.conj().T) / 2)
    return float((np.sum(np.abs(w)) - 1) / 2)

import numpy as np


def _qutrit_weyl_ops(dloc: int) -> "dict":
    ops = {}
    om = np.exp(2j * np.pi / dloc)
    X = np.zeros((dloc, dloc))
    for k in range(dloc):
        X[(k + 1) % dloc, k] = 1.0
    Z = np.diag([om ** k for k in range(dloc)])
    for m in range(dloc):
        for n in range(dloc):
            ops[(m, n)] = np.linalg.matrix_power(X, m) @ np.linalg.matrix_power(Z, n)
    return ops


def _build_conditional(pm: str, phi: float, Ks1: "list", Ks2: "list", U: "np.ndarray", rho_AB: "np.ndarray", dAB: int) -> "np.ndarray":
    total = np.zeros((dAB, dAB), dtype=complex)
    for Ki in Ks1:
        for Kj in Ks2:
            Hfwd = Kj @ U @ Ki
            Hbwd = Ki @ U @ Kj
            if pm == "+":
                M = 0.5 * (Hfwd + np.exp(-1j * phi) * Hbwd)
            else:
                M = 0.5 * (Hfwd - np.exp(-1j * phi) * Hbwd)
            total = total + M @ rho_AB @ M.conj().T
    return total


def quantum_switch_output(Ks_est: "np.ndarray", lam2: float, dloc: int) -> "np.ndarray":
    if not (0 <= lam2 <= 1):
        raise ValueError("lam2 must be in [0, 1]")
    if dloc < 3:
        raise ValueError("dloc must be at least 3 (the inserted unitary needs Weyl index 2)")
    dA, dB = dloc, dloc
    dAB = dA * dB

    Ks1 = [np.kron(np.eye(dA), K) for K in Ks_est]

    Dops = _qutrit_weyl_ops(dloc)
    p2 = ((dloc ** 4 - 1) * lam2 + 1) / dloc ** 4
    Ks2 = [np.sqrt(p2) * np.eye(dAB)]
    for m in range(dloc):
        for n in range(dloc):
            for mp in range(dloc):
                for npp in range(dloc):
                    if m == 0 and n == 0 and mp == 0 and npp == 0:
                        continue
                    coeff = np.sqrt((1 - p2) / (dloc ** 4 - 1))
                    Ks2.append(coeff * np.kron(Dops[(m, n)], Dops[(mp, npp)]))

    psi = np.zeros(dAB, dtype=complex)
    for i in range(dloc):
        psi[i * dloc + i] = 1 / np.sqrt(dloc)
    rho_AB = np.outer(psi, psi.conj())

    U = np.kron(Dops[(2, 0)], Dops[(2, 0)])

    rho_plus = _build_conditional("+", 0.0, Ks1, Ks2, U, rho_AB, dAB)
    rho_minus = _build_conditional("-", 0.0, Ks1, Ks2, U, rho_AB, dAB)
    P_plus = np.trace(rho_plus).real
    P_minus = np.trace(rho_minus).real

    # A control outcome can have exactly zero postselection probability (e.g. at
    # lam2 = 1 with an identity estimated channel); normalizing and scoring that
    # branch is meaningless, so skip it and return the other, valid branch directly.
    branches = [(P_plus, rho_plus), (P_minus, rho_minus)]
    valid = [(p, r) for p, r in branches if p > 1e-12]
    if not valid:
        raise ValueError("both control outcomes have zero postselection probability")
    if len(valid) == 1:
        p_only, rho_only = valid[0]
        return rho_only / p_only

    rho_plus_norm = rho_plus / P_plus
    rho_minus_norm = rho_minus / P_minus

    N_plus = entanglement_negativity(rho_plus_norm, dA, dB)
    N_minus = entanglement_negativity(rho_minus_norm, dA, dB)

    if N_plus >= N_minus:
        return rho_plus_norm
    return rho_minus_norm

import numpy as np
from itertools import permutations, combinations
import string


def _partial_transpose_full(rho: "np.ndarray", dims: "list", sys_list: "tuple") -> "np.ndarray":
    n = len(dims)
    shape = dims + dims
    arr = rho.reshape(shape)
    axes = list(range(2 * n))
    for sys in sys_list:
        axes[sys], axes[n + sys] = axes[n + sys], axes[sys]
    arr = np.transpose(arr, axes)
    return arr.reshape(rho.shape)


def _proj_psd(H: "np.ndarray") -> "np.ndarray":
    H = (H + H.conj().T) / 2
    w, v = np.linalg.eigh(H)
    w_clipped = np.clip(w, 0, None)
    return (v * w_clipped) @ v.conj().T


def _proj_psd_on_subset(omega: "np.ndarray", dims: "list", J: "tuple") -> "np.ndarray":
    om2 = _partial_transpose_full(omega, dims, J)
    om2p = _proj_psd(om2)
    return _partial_transpose_full(om2p, dims, J)


def _sym_project_k(omega: "np.ndarray", dims: "list") -> "np.ndarray":
    n = len(dims)
    total = np.zeros_like(omega)
    perms = list(permutations(range(1, n)))
    for perm in perms:
        axes_ket = [0] + list(perm)
        full_perm = axes_ket + [n + a for a in axes_ket]
        shape = dims + dims
        om = omega.reshape(shape)
        om_p = np.transpose(om, full_perm).reshape(omega.shape)
        total = total + om_p
    return total / len(perms)


def _partial_trace_to_AB1(om: "np.ndarray", dims: "list") -> "np.ndarray":
    n = len(dims)
    shape = dims + dims
    arr = om.reshape(shape)
    ket_labels = list(string.ascii_lowercase[:n])
    bra_labels = list(string.ascii_uppercase[:n])
    for i in range(2, n):
        bra_labels[i] = ket_labels[i]
    in_sub = "".join(ket_labels) + "".join(bra_labels)
    out_sub = ket_labels[0] + ket_labels[1] + bra_labels[0] + bra_labels[1]
    result = np.einsum(f"{in_sub}->{out_sub}", arr)
    dA, dB = dims[0], dims[1]
    return result.reshape(dA * dB, dA * dB)


def _proj_domination(omega: "np.ndarray", dims: "list", rho: "np.ndarray") -> "np.ndarray":
    k = len(dims) - 1
    extra = int(np.prod(dims[2:])) if k > 1 else 1
    sigma = _partial_trace_to_AB1(omega, dims)
    M = sigma - rho
    Mp = _proj_psd(M)
    diff = Mp - M
    I_extra = np.eye(extra)
    delta_omega = np.kron(diff, I_extra) / extra
    return omega + delta_omega


def symmetric_extension_sdp_solve(rho: "np.ndarray", dA: int, dB: int, k: int, iters: int) -> "np.ndarray":
    if k < 1:
        raise ValueError("k must be at least 1")
    if iters < 1:
        raise ValueError("iters must be at least 1")

    dims = [dA] + [dB] * k
    dtot = int(np.prod(dims))
    subsets = []
    for r in range(1, k + 1):
        for J in combinations(range(1, k + 1), r):
            subsets.append(J)

    projs = [lambda y: _proj_psd(y)]
    for J in subsets:
        projs.append(lambda y, J=J: _proj_psd_on_subset(y, dims, J))
    projs.append(lambda y: _sym_project_k(y, dims))
    projs.append(lambda y: _proj_domination(y, dims, rho))
    m = len(projs)

    x = np.eye(dtot, dtype=complex) / dtot
    z = [x.copy() for _ in range(m)]
    u = [np.zeros((dtot, dtot), dtype=complex) for _ in range(m)]
    grad_c = np.eye(dtot, dtype=complex)
    rho_penalty = 1.0

    for _ in range(iters):
        avg_zu = sum(z[i] - u[i] for i in range(m)) / m
        x = avg_zu - grad_c / (m * rho_penalty)
        x = (x + x.conj().T) / 2
        for i in range(m):
            y = x + u[i]
            z[i] = projs[i](y)
            u[i] = u[i] + x - z[i]

    return x

import numpy as np


def compute_Ek(rho: "np.ndarray", dA: int, dB: int, k: int, iters: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    if iters < 1:
        raise ValueError("iters must be at least 1")
    omega = symmetric_extension_sdp_solve(rho, dA, dB, k, iters)
    return float(np.trace(omega).real - 1.0)

def full_certified_switch_pipeline(
    seed_gen: int,
    seed_noise: int,
    alpha0: float,
    beta0: float,
    T: float,
    n: int,
    N: int,
    n_sub: int,
    tau: float,
    lam2: float,
    dloc: int,
    k: int,
    admm_iters: int,
) -> float:
    Y_avg = simulate_averaged_observation_data(seed_gen, seed_noise, alpha0, beta0, T, n, N, n_sub)
    theta_hat = discrete_contrast_estimator(seed_gen, Y_avg, T, n)
    alpha_hat, beta_hat = theta_hat
    Ks_est = channel_from_lindbladian(seed_gen, alpha_hat, beta_hat, tau)
    rho_f = quantum_switch_output(Ks_est, lam2, dloc)
    Ek = compute_Ek(rho_f, dloc, dloc, k, admm_iters)
    return float(Ek)
SCICODE_GOLD_EOF
