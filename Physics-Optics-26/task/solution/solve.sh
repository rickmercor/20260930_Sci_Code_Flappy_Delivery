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


# Oracle implementation for public function: field_multiplication_coefficients
def field_multiplication_coefficients(n: int, k: int, nu: float) -> "np.ndarray":
    # ORACLE (hidden). Appendix A, Eqs. (A1)-(A12) of arXiv:2609.17785.
    # f[side, mu, d]: side 0 = left multiplication (a_mu rho), 1 = right (rho a_mu);
    # mu 0 = a (lowers k), 1 = a^dagger (raises k); d = delta + 1 for target radial index n + delta.
    if int(n) != n or n < 0:
        raise ValueError("radial index n must be a non-negative integer")
    if int(k) != k:
        raise ValueError("coherence order k must be an integer")
    if nu < 0:
        raise ValueError("thermal occupation nu must be non-negative")
    n = int(n); k = int(k)
    f = np.zeros((2, 2, 3), dtype=float)
    p = 1.0 + nu
    c = (n + 1) * nu / (1.0 + nu)
    ak = abs(k)
    if k > 0:
        f[0, 0, 1] = n + k; f[0, 0, 2] = c          # (A1)
        f[0, 1, 0] = p;     f[0, 1, 1] = p          # (A2)
        f[1, 0, 1] = n + k; f[1, 0, 2] = n + 1      # (A3)
        f[1, 1, 0] = p;     f[1, 1, 1] = nu         # (A4)
    elif k < 0:
        f[0, 0, 0] = p;      f[0, 0, 1] = nu        # (A5)
        f[0, 1, 1] = n + ak; f[0, 1, 2] = n + 1     # (A6)
        f[1, 0, 0] = p;      f[1, 0, 1] = p         # (A7)
        f[1, 1, 1] = n + ak; f[1, 1, 2] = c         # (A8)
    else:
        f[0, 0, 0] = p; f[0, 0, 1] = nu             # (A9)
        f[0, 1, 0] = p; f[0, 1, 1] = p              # (A10)
        f[1, 0, 0] = p; f[1, 0, 1] = p              # (A11)
        f[1, 1, 0] = p; f[1, 1, 1] = nu             # (A12)
    if n == 0:
        f[:, :, 0] = 0.0                            # rho_{-1}^{(k)} = 0
    return f

import numpy as np


# Oracle implementation for public function: atomic_multiplication_matrices
def atomic_multiplication_matrices(N: int, s: float) -> "np.ndarray":
    # ORACLE (hidden). Eqs. (4), (6)-(9) of arXiv:2609.17785.
    # Output A[t] with t = 0: S_+ R (left), 1: S_- R (left), 2: R S_+ (right), 3: R S_- (right);
    # (A[t])_{alpha beta} = coefficient of R_alpha in the product acting on R_beta.
    if int(N) != N or N < 1:
        raise ValueError("N must be a positive integer")
    if not (0.0 < s < 1.0):
        raise ValueError("pump parameter s must lie strictly between 0 and 1")
    N = int(N)
    comp = []
    for m0 in range(N, -1, -1):
        for mz in range(N - m0, -1, -1):
            for mp in range(N - m0 - mz, -1, -1):
                comp.append((m0, mz, mp, N - m0 - mz - mp))
    idx = {c: i for i, c in enumerate(comp)}
    D = len(comp)
    A = np.zeros((4, D, D), dtype=float)

    def add(t, b, m, coef):
        if min(m) < 0 or coef == 0:
            return
        A[t, idx[m], b] += coef

    for b, (m0, mz, mp, mm) in enumerate(comp):
        # (6)  S_+ R_m
        add(0, b, (m0 - 1, mz, mp + 1, mm), (1 - s) * m0)
        add(0, b, (m0, mz - 1, mp + 1, mm), -mz)
        add(0, b, (m0 + 1, mz, mp, mm - 1), mm)
        add(0, b, (m0, mz + 1, mp, mm - 1), (1 - s) * mm)
        # (7)  S_- R_m
        add(1, b, (m0 - 1, mz, mp, mm + 1), s * m0)
        add(1, b, (m0, mz - 1, mp, mm + 1), mz)
        add(1, b, (m0 + 1, mz, mp - 1, mm), mp)
        add(1, b, (m0, mz + 1, mp - 1, mm), -s * mp)
        # (8)  R_m S_+
        add(2, b, (m0 - 1, mz, mp + 1, mm), s * m0)
        add(2, b, (m0, mz - 1, mp + 1, mm), mz)
        add(2, b, (m0 + 1, mz, mp, mm - 1), mm)
        add(2, b, (m0, mz + 1, mp, mm - 1), -s * mm)
        # (9)  R_m S_-
        add(3, b, (m0 - 1, mz, mp, mm + 1), (1 - s) * m0)
        add(3, b, (m0, mz - 1, mp, mm + 1), -mz)
        add(3, b, (m0 + 1, mz, mp - 1, mm), mp)
        add(3, b, (m0, mz + 1, mp - 1, mm), (1 - s) * mp)
    return A

import numpy as np


# Oracle implementation for public function: radial_blocks
def radial_blocks(n: int, N: int, A: float, B: float, C: float, g: float, s: float, Delta: float, nu: float) -> "np.ndarray":
    # ORACLE (hidden). Eqs. (10)-(14) of arXiv:2609.17785, stationary sector K = 0.
    # Returns stacked complex array [M_n, G_n, F_n], each D_N x D_N.
    # Chains Step 1 (field_multiplication_coefficients) and Step 2 (atomic_multiplication_matrices).
    if int(n) != n or n < 0:
        raise ValueError("radial index n must be a non-negative integer")
    if A <= 0 or B <= 0:
        raise ValueError("cavity rate A and atomic rate B must be positive")
    if C < B / 2:
        raise ValueError("transverse rate C must satisfy C >= B/2")
    n = int(n)
    At = atomic_multiplication_matrices(N, s)
    N = int(N)
    comp = []
    for m0 in range(N, -1, -1):
        for mz in range(N - m0, -1, -1):
            for mp in range(N - m0 - mz, -1, -1):
                comp.append((m0, mz, mp, N - m0 - mz - mp))
    comp = np.array(comp, dtype=int)
    D = comp.shape[0]
    q = comp[:, 2] - comp[:, 3]
    kk = -q                                   # K = 0  =>  k_beta = -q_beta

    def V(nsrc, dl):
        Vm = np.zeros((D, D), dtype=complex)
        if nsrc < 0 or nsrc + dl < 0:
            return Vm
        d = dl + 1
        for b in range(D):
            f = field_multiplication_coefficients(nsrc, int(kk[b]), nu)
            Vm[:, b] = 0.5j * g * (f[0, 1, d] * At[1][:, b] + f[0, 0, d] * At[0][:, b]
                                   - f[1, 1, d] * At[3][:, b] - f[1, 0, d] * At[2][:, b])
        return Vm

    diag = (-A * (n + np.abs(kk) / 2.0) - B * comp[:, 1]
            - C * (comp[:, 2] + comp[:, 3]) - 1j * Delta * q)
    M = np.diag(diag).astype(complex) + V(n, 0)
    G = V(n + 1, -1)
    F = V(n - 1, +1) if n >= 1 else np.zeros((D, D), dtype=complex)
    return np.stack([M, G, F])

import numpy as np


# Oracle implementation for public function: asymptotic_transfer_matrix
def asymptotic_transfer_matrix(N: int, A: float, B: float, C: float, g: float, s: float, Delta: float, nu: float) -> "np.ndarray":
    # ORACLE (hidden). Proposition 2, Eqs. (18)-(20) of arXiv:2609.17785.
    # Bulk blocks are affine in n: M_n = n M1 + M0, F_n = n F1 + F0, G_n constant (G1 = 0),
    # so the Riccati closure (18) is linear: R_inf = -(M1)^{-1} F1.
    # M1, F1 extracted by a finite difference at a bulk index (exact, blocks are affine for n >= 1).
    nb = 8
    b1 = radial_blocks(nb, N, A, B, C, g, s, Delta, nu)
    b2 = radial_blocks(nb + 1, N, A, B, C, g, s, Delta, nu)
    M1 = b2[0] - b1[0]
    F1 = b2[2] - b1[2]
    return -np.linalg.solve(M1, F1)

import numpy as np


# Oracle implementation for public function: stationary_radial_coefficients
def stationary_radial_coefficients(N: int, A: float, B: float, C: float, g: float, s: float, Delta: float, nu: float, n_max: int) -> "np.ndarray":
    # ORACLE (hidden). Eqs. (16)-(17) + Sec. IV A normalization of arXiv:2609.17785.
    # Downward matrix continued fraction from R_{n_max} = R_inf, then (M_0 + G_0 R_0) X_0 = 0,
    # X_0 normalized so its (N,0,0,0) component (index 0) equals 1, X_{n+1} = R_n X_n.
    if int(n_max) != n_max or n_max < 2:
        raise ValueError("n_max must be an integer >= 2")
    n_max = int(n_max)
    Rs = [None] * (n_max + 1)
    Rs[n_max] = asymptotic_transfer_matrix(N, A, B, C, g, s, Delta, nu)
    for n in range(n_max - 1, -1, -1):
        bl = radial_blocks(n + 1, N, A, B, C, g, s, Delta, nu)
        Rs[n] = -np.linalg.solve(bl[0] + bl[1] @ Rs[n + 1], bl[2])
    b0 = radial_blocks(0, N, A, B, C, g, s, Delta, nu)
    W = b0[0] + b0[1] @ Rs[0]
    _, _, vh = np.linalg.svd(W)
    x = vh[-1].conj()
    x = x / x[0]
    D = x.shape[0]
    X = np.zeros((n_max + 1, D), dtype=complex)
    X[0] = x
    for n in range(n_max):
        X[n + 1] = Rs[n] @ X[n]
    return X

import numpy as np


# Oracle implementation for public function: exact_stationary_observables
def exact_stationary_observables(N: int, A: float, B: float, C: float, g: float, s: float, Delta: float, nu: float, n_max: int) -> "np.ndarray":
    # ORACLE (hidden). Sec. IV C Eqs. (21)-(22) and Appendix B Eq. (B1) of arXiv:2609.17785.
    # Returns [<n>, g2(0), p_e (atom 1), C_2^(N) = <tau_+^(1) tau_-^(2)>] (all real).
    if int(N) != N or N < 2:
        raise ValueError("N must be an integer >= 2 for the pair coherence")
    N = int(N)
    X = stationary_radial_coefficients(N, A, B, C, g, s, Delta, nu, n_max)
    comp = []
    for m0 in range(N, -1, -1):
        for mz in range(N - m0, -1, -1):
            for mp in range(N - m0 - mz, -1, -1):
                comp.append((m0, mz, mp, N - m0 - mz - mp))
    idx = {c: i for i, c in enumerate(comp)}
    c1 = X[1, 0]
    c2 = X[2, 0]
    nbar = nu + (1 + nu) * c1
    nn1 = 2 * nu ** 2 + 4 * nu * (1 + nu) * c1 + 2 * (1 + nu) ** 2 * c2
    g2 = nn1 / nbar ** 2
    pe = s * X[0, idx[(N, 0, 0, 0)]] + X[0, idx[(N - 1, 1, 0, 0)]] / N
    C2 = X[0, idx[(N - 2, 0, 1, 1)]] / (N * (N - 1))
    return np.array([np.real(nbar), np.real(g2), np.real(pe), np.real(C2)], dtype=float)

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


# Oracle implementation for public function: second_order_closure_stationary
def second_order_closure_stationary(N: int, A: float, B: float, C: float, g: float, s: float, Delta: float, nu: float) -> "np.ndarray":
    # ORACLE (hidden). Sec. VI C, Eqs. (31)-(32) of arXiv:2609.17785 (C_r = C).
    # Physical root selected as in the paper: integrate the closed equations of motion from the
    # vacuum (<n> = 0, p_e = 0, Z_1 = 0, C = 0) to stationarity, then polish with a root solve.
    # Returns [<n>, p_e, Re Z_1, Im Z_1, C] with Z_1 = <a^dag tau_-^(j)>, C = <tau_+^(i) tau_-^(j)>.
    if int(N) != N or N < 2:
        raise ValueError("N must be an integer >= 2")
    if A <= 0 or B <= 0:
        raise ValueError("cavity rate A and atomic rate B must be positive")
    if C < B / 2:
        raise ValueError("transverse rate C must satisfy C >= B/2")
    if not (0.0 < s < 1.0):
        raise ValueError("pump parameter s must lie strictly between 0 and 1")
    N = int(N)

    def rhs(t, y):
        n, pe, zr, zi, cc = y
        Z = zr + 1j * zi
        dn = -A * (n - nu) - g * N * zi
        dp = B * (s - pe) + g * zi
        dZ = -(A / 2 + C + 1j * Delta) * Z - 0.5j * g * (pe + (N - 1) * cc + n * (2 * pe - 1))
        dc = -2 * C * cc - g * (2 * pe - 1) * zi
        return [dn, dp, dZ.real, dZ.imag, dc]

    T = 400.0 / min(A, B, C)
    sol = solve_ivp(rhs, (0.0, T), [0.0, 0.0, 0.0, 0.0, 0.0], method="LSODA",
                    rtol=1e-12, atol=1e-14)
    y = fsolve(lambda y: rhs(0.0, y), sol.y[:, -1], xtol=1e-15)
    return np.array(y, dtype=float)

import numpy as np


# Oracle implementation for public function: pair_coherence_closure_error
def pair_coherence_closure_error(N: int, A: float, B: float, C: float, g: float, s: float, Delta: float, nu: float, n_max: int) -> float:
    # ORACLE (hidden). Orchestrator: exact pair coherence (Step 6, damping-basis continued fraction)
    # minus the second-order cumulant-closure pair coherence (Step 7). Chains  steps only.
    exact = exact_stationary_observables(N, A, B, C, g, s, Delta, nu, n_max)
    closure = second_order_closure_stationary(N, A, B, C, g, s, Delta, nu)
    return float(exact[3] - closure[4])
SCICODE_GOLD_EOF
