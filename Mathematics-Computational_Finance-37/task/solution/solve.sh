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


def build_reference_tensor(S_grid: np.ndarray, V_grid: np.ndarray, tau: float) -> np.ndarray:
    import numpy as np
    S_grid = np.asarray(S_grid, dtype=float)
    V_grid = np.asarray(V_grid, dtype=float)
    if S_grid.ndim != 1 or len(S_grid) < 1 or V_grid.ndim != 1 or len(V_grid) < 1:
        raise ValueError("S_grid and V_grid must be 1D arrays of length >= 1")
    if tau <= 0:
        raise ValueError("tau must be positive")

    def lognormal_kernel(Si, Vi):
        import numpy as np
        m = -0.5 * Vi ** 2 * tau
        s2 = Vi ** 2 * tau
        x = np.log(S_grid / Si)
        dens = np.exp(-(x - m) ** 2 / (2 * s2)) / np.sqrt(2 * np.pi * s2)
        return dens / dens.sum()

    nS, nV = len(S_grid), len(V_grid)
    pi = np.zeros((nS, nV, nS, nV, nS))
    for i0 in range(nS):
        for iv1 in range(nV):
            k1 = lognormal_kernel(S_grid[i0], V_grid[iv1])
            for i1 in range(nS):
                for iv2 in range(nV):
                    k2 = lognormal_kernel(S_grid[i1], V_grid[iv2])
                    pi[i0, iv1, i1, iv2, :] = k1[i1] * k2
    return pi / pi.sum()

import numpy as np


def spx_vix_calibration(S_grid: np.ndarray, V_grid: np.ndarray, tau: float,
                                 muS: np.ndarray, muV: np.ndarray,
                                 Kout: int = 6, Kin: int = 40) -> np.ndarray:
    import numpy as np
    S_grid = np.asarray(S_grid, dtype=float)
    V_grid = np.asarray(V_grid, dtype=float)
    muS = np.asarray(muS, dtype=float)
    muV = np.asarray(muV, dtype=float)
    if abs(muS.sum() - 1.0) > 1e-6 or abs(muV.sum() - 1.0) > 1e-6:
        raise ValueError("muS and muV must each sum to 1")
    if Kout < 1 or Kin < 1:
        raise ValueError("Kout and Kin must be >= 1")

    nS, nV = len(S_grid), len(V_grid)
    alpha, c = 0.5, 1.0
    lam0, lam_gamma, lam_max = 1.0, 1.5, 1e3

    def lognormal_kernel(Si, Vi):
        import numpy as np
        m = -0.5 * Vi ** 2 * tau
        s2 = Vi ** 2 * tau
        x = np.log(S_grid / Si)
        dens = np.exp(-(x - m) ** 2 / (2 * s2)) / np.sqrt(2 * np.pi * s2)
        return dens / dens.sum()

    pi = np.zeros((nS, nV, nS, nV, nS))
    for i0 in range(nS):
        for iv1 in range(nV):
            k1 = lognormal_kernel(S_grid[i0], V_grid[iv1])
            for i1 in range(nS):
                for iv2 in range(nV):
                    k2 = lognormal_kernel(S_grid[i1], V_grid[iv2])
                    pi[i0, iv1, i1, iv2, :] = k1[i1] * k2
    pi = pi / pi.sum()

    def marginalize(pi, axis_keep):
        import numpy as np
        axes = tuple(a for a in range(pi.ndim) if a not in axis_keep)
        return pi.sum(axis=axes)

    def project_marginal(pi, axis, target, eps=1e-13):
        import numpy as np
        cur = marginalize(pi, (axis,))
        scale = np.where(cur > eps, target / np.maximum(cur, eps), 1.0)
        shape = [1] * pi.ndim
        shape[axis] = pi.shape[axis]
        pi = pi * scale.reshape(shape)
        return pi / pi.sum()

    L_mat = -(2.0 / tau) * np.log(S_grid[None, :] / S_grid[:, None])

    lam = lam0
    for _ in range(Kout):
        for _ in range(Kin):
            p1 = pi.sum(axis=(3, 4))
            p2 = pi.sum(axis=(0, 1))
            mart1 = np.zeros((nS, nV)); disp1 = np.zeros((nS, nV))
            mart2 = np.zeros((nS, nV)); disp2 = np.zeros((nS, nV))
            for i0 in range(nS):
                for iv1 in range(nV):
                    w = p1[i0, iv1, :]
                    if w.sum() < 1e-14:
                        continue
                    w = w / w.sum()
                    L = -(2.0 / tau) * np.log(S_grid / S_grid[i0])
                    mart1[i0, iv1] = np.sum(w * S_grid) - S_grid[i0]
                    disp1[i0, iv1] = np.sum(w * L) - V_grid[iv1] ** 2
            for i1 in range(nS):
                for iv2 in range(nV):
                    w = p2[i1, iv2, :]
                    if w.sum() < 1e-14:
                        continue
                    w = w / w.sum()
                    L = -(2.0 / tau) * np.log(S_grid / S_grid[i1])
                    mart2[i1, iv2] = np.sum(w * S_grid) - S_grid[i1]
                    disp2[i1, iv2] = np.sum(w * L) - V_grid[iv2] ** 2

            grad = np.zeros_like(pi)
            for i0 in range(nS):
                dS = S_grid - S_grid[i0]
                dL = L_mat[i0, :] - V_grid[:, None] ** 2
                for iv1 in range(nV):
                    g1 = lam * mart1[i0, iv1] * dS + lam * disp1[i0, iv1] * dL[iv1, :]
                    grad[i0, iv1, :, :, :] += g1[:, None, None]
            for i1 in range(nS):
                dS = S_grid - S_grid[i1]
                dL = L_mat[i1, :] - V_grid[:, None] ** 2
                for iv2 in range(nV):
                    g2 = lam * mart2[i1, iv2] * dS + lam * disp2[i1, iv2] * dL[iv2, :]
                    grad[:, :, i1, iv2, :] += g2[None, :]
            eta = min(alpha, c / (np.max(np.abs(grad)) + 1e-300))
            pi = pi * np.exp(-eta * grad)
            pi = pi / pi.sum()
            pi = project_marginal(pi, 0, muS)
            pi = project_marginal(pi, 1, muV)
            pi = project_marginal(pi, 3, muV)
        lam = min(lam * lam_gamma, lam_max)
    return pi

import numpy as np


def marginalize_to_sss(pi: np.ndarray) -> np.ndarray:
    import numpy as np
    pi = np.asarray(pi, dtype=float)
    if pi.ndim != 5:
        raise ValueError("pi must be a 5D array (nS, nV, nS, nV, nS)")
    if pi.shape[0] != pi.shape[2] or pi.shape[0] != pi.shape[4]:
        raise ValueError("pi's S1, S2, S3 axes (0, 2, 4) must share the same length")
    return pi.sum(axis=(1, 3))

import numpy as np


def gtfk_selfconsistent(k: float, sigma: float, theta: float, T: float, xbar: float) -> tuple:
    import numpy as np
    if k <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("k, sigma, and T must all be positive")

    omega = np.sqrt(k ** 2 + sigma ** 2 * np.exp(xbar))
    deltagamma = 0.0
    for _ in range(60):
        f = omega * T / 2
        alpha = (sigma ** 2 / (2 * omega)) * (1.0 / np.tanh(f) - 1.0 / f)
        gamma = ((omega ** 2 - k ** 2) / sigma ** 2) * (deltagamma + 1) \
            + k ** 2 * xbar / sigma ** 2 - k ** 2 * theta / sigma ** 2
        Gamma0 = gamma * (np.cosh(omega * T) - 1) / omega
        GammaT = Gamma0
        gammahat = gamma * T
        deltagamma_new = (sigma ** 2 / (2 * omega)) * (
            (Gamma0 + GammaT) / (2 * np.sinh(f) ** 2) - gammahat / f)
        omega_new = np.sqrt(k ** 2 + sigma ** 2 * np.exp(alpha / 2 + xbar - deltagamma_new))
        if abs(omega_new - omega) < 1e-13 and abs(deltagamma_new - deltagamma) < 1e-13:
            omega, deltagamma = omega_new, deltagamma_new
            break
        omega, deltagamma = omega_new, deltagamma_new
    return omega, deltagamma

import numpy as np


def survival_probability(k: float, sigma: float, theta: float, x0: float, T: float) -> float:
    import numpy as np
    if k <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("k, sigma, and T must all be positive")

    def solve_omega_deltagamma(xbar):
        import numpy as np
        omega = np.sqrt(k ** 2 + sigma ** 2 * np.exp(xbar))
        deltagamma = 0.0
        for _ in range(60):
            f = omega * T / 2
            alpha = (sigma ** 2 / (2 * omega)) * (1.0 / np.tanh(f) - 1.0 / f)
            gamma = ((omega ** 2 - k ** 2) / sigma ** 2) * (deltagamma + 1) \
                + k ** 2 * xbar / sigma ** 2 - k ** 2 * theta / sigma ** 2
            Gamma0 = gamma * (np.cosh(omega * T) - 1) / omega
            GammaT = Gamma0
            gammahat = gamma * T
            deltagamma_new = (sigma ** 2 / (2 * omega)) * (
                (Gamma0 + GammaT) / (2 * np.sinh(f) ** 2) - gammahat / f)
            omega_new = np.sqrt(k ** 2 + sigma ** 2 * np.exp(alpha / 2 + xbar - deltagamma_new))
            if abs(omega_new - omega) < 1e-13 and abs(deltagamma_new - deltagamma) < 1e-13:
                omega, deltagamma = omega_new, deltagamma_new
                break
            omega, deltagamma = omega_new, deltagamma_new
        return omega, deltagamma, gamma

    def integrand(xbar):
        import numpy as np
        omega, deltagamma, gamma = solve_omega_deltagamma(xbar)
        f = omega * T / 2
        alpha = (sigma ** 2 / (2 * omega)) * (1.0 / np.tanh(f) - 1.0 / f)
        Gamma0 = gamma * (np.cosh(omega * T) - 1) / omega
        GammaT = Gamma0
        gammahat = gamma * T
        sinh2f = np.sinh(2 * f)
        int1 = (np.cosh(omega * T) - 1) / omega
        int2 = 0.5 * np.sinh(omega * T) * T \
            + (np.cosh(omega * T) - np.cosh(-omega * T)) / (4 * omega)
        Gamma0T = gamma * (-1.0 / omega) * gamma * int1 + (gamma ** 2 / omega) * int2
        Gamma_val = (sigma ** 2 / omega) * (
            Gamma0T / sinh2f - (1 / (4 * f)) * ((Gamma0 + GammaT) / sinh2f - gammahat) ** 2)

        xshift = xbar - deltagamma
        const_part = (k ** 2 - omega ** 2) / (2 * sigma ** 2) * alpha \
            - omega ** 2 * deltagamma ** 2 / (2 * sigma ** 2) \
            + np.exp(alpha / 2) * np.exp(xshift)
        int_w = const_part * T \
            + (k ** 2 * (theta - xshift) ** 2 / (2 * sigma ** 2)) * T \
            + deltagamma * gamma * T

        N_xbar = np.sqrt((1.0 / (2 * np.pi * alpha)) * (1.0 / (2 * np.pi * T * sigma ** 2))) \
            * (f / np.sinh(f)) * np.exp(-int_w + Gamma_val)

        A = 1.0 / (8 * alpha) + (omega / np.tanh(f)) / (4 * sigma ** 2) + k / (2 * sigma ** 2)
        delta_ = (sigma ** 2 / (2 * omega * f)) * ((Gamma0 + GammaT) / sinh2f - gammahat)
        B = (x0 - xbar + delta_) / (2 * alpha) + Gamma0 / sinh2f + k * (x0 - theta) / sigma ** 2
        C = -(x0 - xbar + delta_) ** 2 / (2 * alpha) - (x0 - xbar) * (Gamma0 + GammaT) / sinh2f \
            + k * T / 2
        Nbar = np.sqrt(np.pi / A) * N_xbar * np.exp(C + B ** 2 / (4 * A))
        return Nbar

    n_xbar, x_range = 401, 2.5
    xbars = np.linspace(x0 - x_range, x0 + x_range, n_xbar)
    vals = np.array([integrand(xb) for xb in xbars])
    return float(np.sum((vals[:-1] + vals[1:]) * np.diff(xbars) / 2.0))

import numpy as np


def contract_value(joint123: np.ndarray, S_grid: np.ndarray, D: float, gm: float,
                            alpha_s: float, Q1: float, Q2: float) -> float:
    import numpy as np
    joint123 = np.asarray(joint123, dtype=float)
    S_grid = np.asarray(S_grid, dtype=float)
    if not (0 < Q1 <= 1) or not (0 < Q2 <= 1):
        raise ValueError("Q1 and Q2 must be in (0, 1]")
    if Q2 > Q1 + 1e-12:
        raise ValueError("Q2 must be <= Q1")
    if D <= 0:
        raise ValueError("D must be positive")

    F0, S0 = 0.0, 1.0
    s0, d0 = Q1, 1.0 - Q1
    s1, d1 = Q2 / Q1, 1.0 - Q2 / Q1
    G1 = D * np.exp(gm * 1)
    G2 = D * np.exp(gm * 2) + D * np.exp(gm * 1)
    G3 = D * np.exp(gm * 3) + D * np.exp(gm * 2) + D * np.exp(gm * 1)

    p1 = joint123.sum(axis=(1, 2))

    F1_plus_inception = F0 + D
    U0 = 0.0
    for i1 in range(len(S_grid)):
        if p1[i1] < 1e-14:
            continue
        S1 = S_grid[i1]
        F1_minus = F1_plus_inception * (S1 / S0)
        B1 = max(F1_minus, G1)
        surrender1 = alpha_s * F1_minus

        w_s2 = joint123[i1, :, :].sum(axis=1)
        w_s2 = w_s2 / w_s2.sum()
        F1_plus = F1_minus + D
        C1 = 0.0
        for i2 in range(len(S_grid)):
            if w_s2[i2] < 1e-14:
                continue
            S2 = S_grid[i2]
            F2_minus = F1_plus * (S2 / S1)
            B2 = max(F2_minus, G2)
            surrender2 = alpha_s * F2_minus

            w_s3 = joint123[i1, i2, :]
            w_s3 = w_s3 / w_s3.sum()
            F2_plus = F2_minus + D
            C2 = 0.0
            for i3 in range(len(S_grid)):
                if w_s3[i3] < 1e-14:
                    continue
                S3 = S_grid[i3]
                F3_minus = F2_plus * (S3 / S2)
                C2 += w_s3[i3] * max(F3_minus, G3)
            U2 = max(C2 - D, surrender2)
            C1 += w_s2[i2] * (s1 * U2 + d1 * B2)
        U1 = max(C1 - D, surrender1)

        U0 += p1[i1] * (s0 * U1 + d0 * B1)
    return U0

import numpy as np

# NOTE for Studio authoring: the model-facing compute_contract_value (once
# implemented by a candidate) should call the PUBLIC function names from
# sub_problems 02, 03, 05, 06 (spx_vix_calibration, marginalize_to_sss,
# survival_probability, contract_value), since only those are available to
# it. The GOLD compute_contract_value below must instead call the
# -prefixed versions of those same steps -- calling the public
# names there would let a candidate's own buggy step implementations leak
# into what is supposed to be the independent reference answer, letting a
# broken pipeline pass this step's comparison.


def compute_contract_value() -> float:
    import numpy as np
    S_grid = np.array([0.7046880897187134, 0.8394570207692074, 1.0, 1.191246216612358, 1.4190675485932571])
    V_grid = np.array([0.12, 0.22, 0.32])
    tau = 30 / 365
    muS = np.array([0.06, 0.2, 0.48, 0.2, 0.06])
    muS = muS / muS.sum()
    muV = np.array([0.3, 0.45, 0.25])
    pi = spx_vix_calibration(S_grid, V_grid, tau, muS, muV, Kout=6, Kin=40)
    joint123 = marginalize_to_sss(pi)

    k, sigma = 0.3, 0.35
    theta = -2.900422093749666
    x0 = -3.101092789211817
    Q1 = survival_probability(k, sigma, theta, x0, 1.0)
    Q2 = survival_probability(k, sigma, theta, x0, 2.0)

    D, gm, alpha_s = 100.0, 0.02, 1.0
    return contract_value(joint123, S_grid, D, gm, alpha_s, Q1, Q2)
SCICODE_GOLD_EOF
