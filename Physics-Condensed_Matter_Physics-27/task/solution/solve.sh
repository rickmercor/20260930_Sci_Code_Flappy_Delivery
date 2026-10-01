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

import numpy as np


def jw_pair_phases(cfg: dict) -> "np.ndarray":
    N = cfg["nx"] * cfg["ny"]
    theta = np.pi * np.tril(np.ones((N, N)), -1)
    alpha = theta[:, None, :] - theta[None, :, :]
    k = np.arange(N)
    alpha[k, :, k] = 0.0
    alpha[:, k, k] = 0.0
    return alpha

import numpy as np


def _site(x, y, nx):
    """Snake site index of (x, y)."""
    return (x if y % 2 == 0 else nx - 1 - x) + y * nx


def _coupling(cfg):
    """Symmetric N x N matrix of bond couplings J (zero on non-bonds)."""
    nx, ny = cfg["nx"], cfg["ny"]
    J = np.zeros((nx * ny, nx * ny))
    for x in range(nx):
        for y in range(ny):
            for dx, dy, c in ((1, 0, cfg["J1"]), (0, 1, cfg["J1"]), (1, 1, cfg["J2"]), (1, -1, cfg["J2"])):
                if x + dx < nx and 0 <= y + dy < ny:
                    p, q = _site(x, y, nx), _site(x + dx, y + dy, nx)
                    J[p, q] = J[q, p] = c
    return J


def z_fock(rho: "np.ndarray", cfg: dict) -> "np.ndarray":
    rho = np.asarray(rho, dtype=complex)
    Jz = cfg["Delta"] * _coupling(cfg)
    return np.diag(Jz @ (np.diag(rho) - 0.5)) - Jz * rho

import numpy as np


def _pairs(rho, cfg):
    """Per-pair intermediates (J/2, rows, cols, D, N) for every ordered bond pair (m, n)."""
    J, al = _coupling(cfg), jw_pair_phases(cfg)
    out = []
    for m, n in zip(*np.nonzero(J)):
        S = np.nonzero(al[m, n])[0]
        rows, cols = np.r_[S, n], np.r_[S, m]
        D = np.r_[np.exp(1j * al[m, n, S]) - 1.0, 1.0]
        N = D[:, None] * rho[np.ix_(rows, cols)]
        N[np.arange(len(S)), np.arange(len(S))] += 1.0
        out.append((0.5 * J[m, n], rows, cols, D, N))
    return out


def hf_energy(rho: "np.ndarray", cfg: dict) -> float:
    rho = np.asarray(rho, dtype=complex)
    Jz = cfg["Delta"] * _coupling(cfg)
    d = np.diag(rho)
    E = 0.5 * (d @ Jz @ d - np.sum(Jz * rho * rho.T)) - 0.5 * np.sum(Jz @ d) + 0.125 * np.sum(Jz)
    for c, rows, cols, D, N in _pairs(rho, cfg):
        E += c * np.linalg.det(N)
    return float(np.real(E))

import numpy as np


def _adj(N, second=False):
    """adj(N) with adj_ji = d det N / dN_ij and, if second, d2 det N / dN_ij dN_kl; finite for singular N."""
    U, s, Vh = np.linalg.svd(N)
    m = len(s)
    i = np.arange(m)
    M = np.broadcast_to(s, (m, m, m)).copy()
    M[i, :, i] = 1.0
    M[:, i, i] = 1.0
    W = np.linalg.det(U) * np.linalg.det(Vh) * M.prod(axis=2)
    V = Vh.conj().T
    adj = (V * np.diag(W)) @ U.conj().T
    if not second:
        return adj
    Y = (U.conj()[:, None, :] * V[None, :, :]).reshape(m * m, m)
    T = (Y @ W @ Y.T).reshape(m, m, m, m)
    return adj, T - T.transpose(0, 3, 2, 1)


def fock_matrix(rho: "np.ndarray", cfg: dict) -> "np.ndarray":
    rho = np.asarray(rho, dtype=complex)
    F = z_fock(rho, cfg)
    for c, rows, cols, D, N in _pairs(rho, cfg):
        F[np.ix_(cols, rows)] += c * _adj(N) * D
    return F

import numpy as np


def fock_kernel(rho: "np.ndarray", cfg: dict) -> "np.ndarray":
    rho = np.asarray(rho, dtype=complex)
    Jz = cfg["Delta"] * _coupling(cfg)
    k = np.arange(len(Jz))[:, None]
    K = np.zeros((len(Jz),) * 4, dtype=complex)
    K[k, k, k.T, k.T] = Jz
    K[k, k.T, k, k.T] -= Jz
    for c, rows, cols, D, N in _pairs(rho, cfg):
        K[np.ix_(cols, rows, rows, cols)] += c * np.einsum("i,k,ijkl->jikl", D, D, _adj(N, True)[1])
    return K

import numpy as np


def orbital_hessian(rho: "np.ndarray", cfg: dict) -> "np.ndarray":
    rho = np.asarray(rho, dtype=complex)
    N = len(rho)
    Nf = N // 2
    F = fock_matrix(rho, cfg)
    eps, C = np.linalg.eigh(0.5 * (F + F.conj().T))
    K = fock_kernel(rho, cfg)
    o, v = C[:, :Nf], C[:, Nf:]
    A = np.einsum("mnls,ma,ni,lb,sj->aibj", K, v.conj(), o, v, o.conj(), optimize=True)
    B = np.einsum("mnls,ma,ni,lj,sb->aibj", K, v.conj(), o, o, v.conj(), optimize=True)
    n = (N - Nf) * Nf
    A = A.reshape(n, n) + np.diag((eps[Nf:, None] - eps[None, :Nf]).ravel())
    return np.array([A, B.reshape(n, n)])

import numpy as np
from scipy.linalg import expm


def _scf(rho, cfg):
    """Aufbau SCF with DIIS extrapolation of F, converged to max|[F, rho]| < 1e-12."""
    Nf = len(rho) // 2
    Fs, Es = [], []
    for it in range(500):
        F = fock_matrix(rho, cfg)
        F = 0.5 * (F + F.conj().T)
        e = F @ rho - rho @ F
        if np.abs(e).max() < 1e-12:
            break
        Fs, Es = (Fs + [F])[-8:], (Es + [e])[-8:]
        n = len(Fs)
        if n >= 3 and np.abs(e).max() < 0.1:
            M = -np.ones((n + 1, n + 1))
            M[n, n] = 0.0
            M[:n, :n] = [[np.vdot(a, b).real for b in Es] for a in Es]
            c = np.linalg.solve(M, np.r_[np.zeros(n), -1.0])
            F = sum(ci * Fi for ci, Fi in zip(c, Fs))
        v = np.linalg.eigh(F)[1][:, :Nf]
        rho = v @ v.conj().T
    return rho


def hf_solution(cfg: dict) -> "np.ndarray":
    nx, ny = cfg["nx"], cfg["ny"]
    N = nx * ny
    Nf = N // 2
    occ = [_site(x, y, nx) for x in range(nx) for y in range(ny) if (x + y) % 2 == 0]
    rho = np.zeros((N, N), dtype=complex)
    rho[occ, occ] = 1.0
    for cycle in range(20):
        rho = _scf(rho, cfg)
        A, B = orbital_hessian(rho, cfg)
        n = len(A)
        lam, U = np.linalg.eigh(np.block([[A, B], [B.conj(), A.conj()]]))
        if lam[0] > 0.0:
            break
        # unstable: rotate the canonical orbitals along the lowest Hessian mode u = (z, z*) and re-converge
        x, y = U[:n, 0], U[n:, 0].conj()
        z = x + y if np.linalg.norm(x + y) > np.linalg.norm(x - y) else 1j * (x - y)
        F = fock_matrix(rho, cfg)
        C = np.linalg.eigh(0.5 * (F + F.conj().T))[1]
        kap = np.zeros((N, N), dtype=complex)
        kap[Nf:, :Nf] = 0.3 * z.reshape(N - Nf, Nf) / np.linalg.norm(z)
        C = C @ expm(kap - kap.conj().T)
        rho = C[:, :Nf] @ C[:, :Nf].conj().T
    return rho

import numpy as np


def rpa_correlation(A: "np.ndarray", B: "np.ndarray") -> float:
    n = len(A)
    M = np.block([[A, B], [-np.conj(B), -np.conj(A)]])
    w = np.sort(np.linalg.eigvals(M).real)[n:]
    return float(0.25 * (w.sum() - np.trace(A).real))

import numpy as np


def total_energy(cfg: dict) -> float:
    rho = hf_solution(cfg)
    A, B = orbital_hessian(rho, cfg)
    return float(hf_energy(rho, cfg) + rpa_correlation(A, B))
SCICODE_GOLD_EOF
