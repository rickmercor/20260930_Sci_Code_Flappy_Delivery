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

def _van_leer_slope_periodic(psi: np.ndarray, dx: float) -> np.ndarray:
    """Compute van-Leer limited slopes for each cell with periodic neighbors.

    Parameters
    ----------
    psi : numpy.ndarray
        Cell-averaged values, shape (N_x,).
    dx : float
        Uniform cell width.

    Returns
    -------
    slopes : numpy.ndarray
        Limited slope in each cell, shape (N_x,).
    """
    N = len(psi)
    slopes = np.zeros(N)
    for i in range(N):
        left = psi[(i - 1) % N]
        right = psi[(i + 1) % N]
        dL = (psi[i] - left) / dx
        dR = (right - psi[i]) / dx
        if dL * dR > 0.0:
            slopes[i] = 2.0 * dL * dR / (dL + dR)
    return slopes



def reconstruct_interfaces(
    psi: np.ndarray, dx: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute van-Leer limited periodic reconstruction at cell interfaces."""
    if dx <= 0:
        raise ValueError("dx must be positive")
    N = len(psi)
    slopes = _van_leer_slope_periodic(psi, dx)

    psi_L = np.zeros(N)
    psi_R = np.zeros(N)

    for j in range(N):
        left_cell = (j - 1) % N
        psi_L[j] = psi[left_cell] + 0.5 * dx * slopes[left_cell]
        psi_R[j] = psi[j] - 0.5 * dx * slopes[j]

    return psi_L, psi_R

import numpy as np

def _compute_coefficients(dt: float, tau: float) -> tuple[float, float]:
    """Compute integral solution coefficients from the kinetic equation.

    Parameters
    ----------
    dt : float
        Time step size (dt > 0).
    tau : float
        Characteristic collision time (tau > 0).

    Returns
    -------
    coefficients : tuple[float, float]
        (c3, c5).
    """
    a0 = np.exp(-dt / tau)
    a1 = 1.0 - a0

    c3 = 2.0 * tau**2 * a1 / dt - tau * a0 - tau
    c5 = tau * a0 - tau**2 * a1 / dt

    return (c3, c5)



def compute_macroscopic_flux(
    psi: np.ndarray,
    sigma_s: float,
    sigma: float,
    dx: float,
    dt: float,
    tau: float,
) -> np.ndarray:
    """Compute macroscopic numerical flux at periodic interfaces via Eq (3.37)."""
    if dx <= 0 or dt <= 0 or tau <= 0:
        raise ValueError("dx, dt and tau must be positive")
    N_x = len(psi)
    sigma_bar = sigma_s / (2.0 * sigma)
    S = sigma_bar * psi

    S_L, S_R = reconstruct_interfaces(S, dx)

    c3, _ = _compute_coefficients(dt, tau)
    v = 1.0
    coeff = 2.0 * v * v * c3 / (3.0 * dx)

    H_ma = np.zeros(N_x)
    for j in range(N_x):
        left_cell = (j - 1) % N_x
        H_ma[j] = coeff * (S_L[j] - S[left_cell] + S[j] - S_R[j])

    return H_ma

import numpy as np

def classify_particles(
    tau: float, dt: float, rng: np.random.Generator, n_particles: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sample free transport times and classify particles."""
    if n_particles < 0 or tau <= 0 or dt <= 0:
        raise ValueError("need n_particles >= 0 and positive tau and dt")
    r = rng.random(n_particles)
    t_f = np.minimum(-tau * np.log(r), dt)
    is_collisionless = t_f >= dt
    return t_f, is_collisionless

import numpy as np

def free_transport_step(
    x_p: np.ndarray,
    xi_p: np.ndarray,
    w_p: np.ndarray,
    dx: float,
    N_x: int,
    dt: float,
    t_f: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream particles with periodic wrapping and O(1) interface check."""
    if N_x < 1 or dx <= 0 or dt <= 0:
        raise ValueError("need N_x >= 1 and positive dx and dt")
    n = len(x_p)
    H_mi_free = np.zeros(N_x)
    x_new = np.empty(n)

    for p in range(n):
        x0 = x_p[p]
        xi = xi_p[p]
        w = w_p[p]
        x1 = x0 + xi * t_f[p]
        crossed_periodic_face = x1 < 0.0 or x1 >= 1.0
        if x1 < 0.0:
            x1 += 1.0
        elif x1 >= 1.0:
            x1 -= 1.0
        cell0 = int(x0 / dx) % N_x
        cell1 = int(x1 / dx) % N_x
        if cell1 != cell0 or crossed_periodic_face:
            if xi > 0.0:
                H_mi_free[(cell0 + 1) % N_x] += w / dt
            elif xi < 0.0:
                H_mi_free[cell0] -= w / dt
        x_new[p] = x1

    return x_new, H_mi_free

import numpy as np

def macroscopic_update(
    psi: np.ndarray,
    H_ma: np.ndarray,
    H_mi_free: np.ndarray,
    psi_ma: np.ndarray,
    sigma: float,
    sigma_s: float,
    q_arr: np.ndarray,
    dx: float,
    dt: float,
    tau: float,
) -> np.ndarray:
    """Update macroscopic scalar flux with periodic collisional flux."""
    if dx <= 0 or dt <= 0 or tau <= 0:
        raise ValueError("dx, dt and tau must be positive")
    N_x = len(psi)
    v = 1.0

    _, c5 = _compute_coefficients(dt, tau)
    a0 = np.exp(-dt / tau)
    a1 = 1.0 - a0
    c4 = tau * a1 / dt

    psi_ma_L, psi_ma_R = reconstruct_interfaces(psi_ma, dx)

    c5_term = c5 + 0.5 * dt * a0
    col_coeff = c5_term * v * v / (3.0 * dx)
    mom_coeff = (c4 - a0) * v / 4.0
    c1 = 1.0 - tau * a1 / dt

    H_ma_col = np.zeros(N_x)
    H_q = np.zeros(N_x)
    for j in range(N_x):
        left_cell = (j - 1) % N_x
        H_ma_col[j] = (
            col_coeff * (psi_ma_L[j] - psi_ma[left_cell] + psi_ma[j] - psi_ma_R[j])
            + mom_coeff * (psi_ma_L[j] - psi_ma_R[j])
        )
        H_q[j] = c1 * 0.5 * v * (q_arr[left_cell] - q_arr[j]) / sigma

    sigma_a = sigma - sigma_s
    alpha = 1.0 / (1.0 + v * dt * sigma_a)

    source = 2.0 * v * dt * q_arr

    psi_new = np.zeros(N_x)
    for i in range(N_x):
        j_right = (i + 1) % N_x
        j_left = i
        flux_div = (
            (H_ma[j_right] - H_ma[j_left])
            + (H_mi_free[j_right] - H_mi_free[j_left])
            + (H_ma_col[j_right] - H_ma_col[j_left])
            + (H_q[j_right] - H_q[j_left])
        )
        psi_new[i] = alpha * (psi[i] + source[i] - (dt / dx) * flux_div)

    return psi_new

import numpy as np

def resample_particles(
    psi_ma: np.ndarray,
    dt: float,
    tau: float,
    dx: float,
    m_e: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Represent positive cell masses whose rounded particle counts are nonzero."""
    if m_e <= 0 or dx <= 0 or dt < 0 or tau <= 0:
        raise ValueError("dt must be nonnegative; tau, m_e and dx must be positive")
    N_x = len(psi_ma)
    free_fraction = np.exp(-dt / tau)

    total = 0
    for i in range(N_x):
        total_mass = free_fraction * psi_ma[i] * dx
        if total_mass > 0.0:
            n_i = int(np.round(total_mass / m_e))
            total += n_i

    x_resamp = np.empty(total)
    xi_resamp = np.empty(total)
    w_resamp = np.empty(total)

    idx = 0
    for i in range(N_x):
        total_mass = free_fraction * psi_ma[i] * dx
        if total_mass <= 0.0:
            continue
        n_i = int(np.round(total_mass / m_e))
        if n_i == 0:
            continue
        actual_w = total_mass / n_i
        x_lo = i * dx
        for k in range(n_i):
            x_resamp[idx] = x_lo + rng.random() * dx
            xi_resamp[idx] = 2.0 * rng.random() - 1.0
            w_resamp[idx] = actual_w
            idx += 1

    return x_resamp, xi_resamp, w_resamp

import numpy as np

def _bin_particles(x_p: np.ndarray, w_p: np.ndarray, n_particles: int, N_x: int, dx: float) -> np.ndarray:
    """Bin particle masses into cells."""
    psi_mi = np.zeros(N_x)
    for p in range(n_particles):
        cell = int(x_p[p] / dx) % N_x
        psi_mi[cell] += w_p[p] / dx
    return psi_mi



def solve_ugkwp_steady(
    sigma_s: float, sigma_a: float, q_arr: np.ndarray,
    N_x: int, CFL: float, n_ppc: int,
    max_iter: int, avg_start: int, rng: np.random.Generator,
) -> np.ndarray:
    """Run UGKWP with periodic BC to steady state and return averaged scalar flux."""
    if N_x < 1 or CFL <= 0 or n_ppc < 1 or not 0 <= avg_start < max_iter:
        raise ValueError("invalid solver configuration")
    v = 1.0
    dx = 1.0 / N_x
    sigma_t = sigma_s + sigma_a
    tau = 1.0 / (v * sigma_t)
    dt = CFL * min(dx / v, 1.5 * dx**2 * sigma_t / v)
    m_e = dx / n_ppc

    psi = np.zeros(N_x)
    psi_ma = np.zeros(N_x)

    x_survive = np.empty(0)
    xi_survive = np.empty(0)
    w_survive = np.empty(0)
    x_resamp = np.empty(0)
    xi_resamp = np.empty(0)
    w_resamp = np.empty(0)

    psi_avg = np.zeros(N_x)
    n_avg = 0

    for iteration in range(max_iter):
        H_ma = compute_macroscopic_flux(psi, sigma_s, sigma_t, dx, dt, tau)

        n_s = len(x_survive)
        if n_s > 0:
            t_f_s, is_cl_s = classify_particles(tau, dt, rng, n_s)
        else:
            t_f_s = np.empty(0)
            is_cl_s = np.empty(0, dtype=bool)

        n_r = len(x_resamp)
        t_f_r = np.full(n_r, dt)

        x_p = np.concatenate((x_survive, x_resamp))
        xi_p = np.concatenate((xi_survive, xi_resamp))
        w_p = np.concatenate((w_survive, w_resamp))
        t_f = np.concatenate((t_f_s, t_f_r))
        is_collisionless = np.concatenate((is_cl_s, np.ones(n_r, dtype=bool)))

        if len(x_p) > 0:
            x_new, H_mi_free = free_transport_step(
                x_p, xi_p, w_p, dx, N_x, dt, t_f
            )
            surviving = is_collisionless
            x_survive = x_new[surviving]
            xi_survive = xi_p[surviving]
            w_survive = w_p[surviving]
        else:
            H_mi_free = np.zeros(N_x)
            x_survive = np.empty(0)
            xi_survive = np.empty(0)
            w_survive = np.empty(0)

        psi_new = macroscopic_update(
            psi, H_ma, H_mi_free, psi_ma,
            sigma_t, sigma_s, q_arr, dx, dt, tau,
        )

        psi_mi = _bin_particles(x_survive, w_survive, len(x_survive), N_x, dx)

        psi_ma = psi_new - psi_mi

        x_resamp, xi_resamp, w_resamp = resample_particles(
            psi_ma, dt, tau, dx, m_e, rng,
        )

        psi = psi_new

        if iteration >= avg_start:
            n_avg += 1
            for i in range(N_x):
                psi_avg[i] += (psi[i] - psi_avg[i]) / n_avg

    return psi_avg
SCICODE_GOLD_EOF
