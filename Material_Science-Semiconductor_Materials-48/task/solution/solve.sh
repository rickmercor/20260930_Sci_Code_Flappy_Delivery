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
import numpy.typing as npt


def screened_gaussian_moments(s: npt.ArrayLike, kappa: float, r0: float) -> np.ndarray:
    """Gaussian moments of the repulsive screened interaction, int d^2r V(r) exp(-s r^2)."""
    import numpy as np
    from scipy.special import dawsn, expi
    e2 = 1.439964                      # e^2/(4 pi eps0), eV nm
    sv = np.asarray(s, dtype=float)
    if sv.ndim != 1 or sv.size == 0:
        raise ValueError("s must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(sv)) or np.any(sv <= 0.0):
        raise ValueError("every s must be finite and positive")
    kappa = float(kappa)
    r0 = float(r0)
    if not np.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("kappa must be finite and positive")
    if not np.isfinite(r0) or r0 < 0.0:
        raise ValueError("r0 must be finite and non-negative")
    pref = 2.0 * np.pi * e2            # e^2/(2 eps0), eV nm
    if r0 == 0.0:
        return pref * np.sqrt(np.pi) / (2.0 * kappa * np.sqrt(sv))
    # momentum-space form: (pref / 2s) int_0^inf exp(-q^2/4s) / (kappa + r0 q) dq,
    # reduced to J(c) = int_0^inf exp(-u^2)/(u + c) du with c = kappa / (2 r0 sqrt(s))
    c = kappa / (2.0 * r0 * np.sqrt(sv))
    x = c * c
    jc = np.empty_like(c)
    low = x < 600.0
    jc[low] = np.sqrt(np.pi) * dawsn(c[low]) - 0.5 * np.exp(-x[low]) * expi(x[low])
    if np.any(~low):
        xh = x[~low]
        tail = np.zeros_like(xh)
        term = 1.0 / xh
        for n in range(1, 40):         # exp(-x) Ei(x) = sum_n n!/x^(n+1) for large x
            tail += term
            term = term * n / xh
        jc[~low] = np.sqrt(np.pi) * dawsn(c[~low]) - 0.5 * tail
    return pref / (2.0 * sv * r0) * jc

import numpy as np
import numpy.typing as npt


def exciton_ground_state(exponents: npt.ArrayLike, m_e: float, m_h: float, kappa: float, r0: float) -> np.ndarray:
    """Lowest Wannier state of the screened electron-hole problem in a Gaussian basis."""
    import numpy as np
    from scipy.linalg import eigh
    hb2m0 = 0.0380998                  # hbar^2/(2 m0), eV nm^2
    a = np.asarray(exponents, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError("exponents must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError("exponents must be finite and positive")
    if np.unique(a).size != a.size:
        raise ValueError("exponents must be distinct")
    m_e = float(m_e)
    m_h = float(m_h)
    if not (np.isfinite(m_e) and np.isfinite(m_h)) or m_e <= 0.0 or m_h <= 0.0:
        raise ValueError("carrier masses must be finite and positive")
    mu = m_e * m_h / (m_e + m_h)
    s = a[:, None] + a[None, :]
    overlap = np.pi / s
    kinetic = (hb2m0 / mu) * 4.0 * np.pi * np.outer(a, a) / s ** 2
    potential = -screened_gaussian_moments(s.ravel(), kappa, r0).reshape(s.shape)
    try:
        evals, evecs = eigh(kinetic + potential, overlap)
    except np.linalg.LinAlgError:
        raise ValueError("the Gaussian basis is numerically linearly dependent")
    c = evecs[:, 0].copy()
    c = c / np.sqrt(c @ overlap @ c)
    if c.sum() < 0.0:
        c = -c
    return np.concatenate([[evals[0]], c])

import numpy as np
import numpy.typing as npt


def exchange_constant(exponents: npt.ArrayLike, coefficients: npt.ArrayLike, kappa: float, r0: float) -> float:
    """Long-wavelength fermionic exchange constant W of two identical 1s excitons, eV nm^2."""
    import numpy as np
    a = np.asarray(exponents, dtype=float)
    c = np.asarray(coefficients, dtype=float)
    if a.ndim != 1 or a.size == 0 or c.shape != a.shape:
        raise ValueError("exponents and coefficients must be non-empty 1D arrays of equal length")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(c))) or np.any(a <= 0.0):
        raise ValueError("exponents must be finite and positive and coefficients finite")
    # phi(k) = sum_i w_i exp(-t_i k^2) is the Fourier amplitude of psi(r) = sum_i c_i exp(-a_i r^2)
    w = c * np.pi / a
    t = 1.0 / (4.0 * a)
    b2 = (t[:, None] + t[None, :]).ravel()
    f2 = (w[:, None] * w[None, :]).ravel() / (4.0 * np.pi * b2)
    e2 = 1.0 / (4.0 * b2)
    b3 = (t[:, None, None] + t[None, :, None] + t[None, None, :]).ravel()
    f3 = (w[:, None, None] * w[None, :, None] * w[None, None, :]).ravel() / (4.0 * np.pi * b3)
    e3 = 1.0 / (4.0 * b3)
    # real-space form: W = 2 [int V psi F3 - int V F2^2], F_n the inverse transform of phi^n
    g1 = screened_gaussian_moments((a[:, None] + e3[None, :]).ravel(), kappa, r0)
    t1 = float(np.sum(c[:, None] * f3[None, :] * g1.reshape(a.size, e3.size)))
    g2 = screened_gaussian_moments((e2[:, None] + e2[None, :]).ravel(), kappa, r0)
    t2 = float(np.sum(f2[:, None] * f2[None, :] * g2.reshape(e2.size, e2.size)))
    return 2.0 * (t1 - t2)

import numpy as np
import numpy.typing as npt


def saturation_overlap(exponents: npt.ArrayLike, coefficients: npt.ArrayLike, m_e: float, m_h: float, momenta: npt.ArrayLike) -> np.ndarray:
    """Phase-space-filling overlap I(Q) of a zero-momentum pair state with an exciton at Q, nm."""
    import numpy as np
    a = np.asarray(exponents, dtype=float)
    c = np.asarray(coefficients, dtype=float)
    q = np.asarray(momenta, dtype=float)
    if a.ndim != 1 or a.size == 0 or c.shape != a.shape:
        raise ValueError("exponents and coefficients must be non-empty 1D arrays of equal length")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(c))) or np.any(a <= 0.0):
        raise ValueError("exponents must be finite and positive and coefficients finite")
    if q.ndim != 1 or q.size == 0 or not np.all(np.isfinite(q)) or np.any(q < 0.0):
        raise ValueError("momenta must be a non-empty 1D array of finite non-negative values")
    m_e = float(m_e)
    m_h = float(m_h)
    if not (np.isfinite(m_e) and np.isfinite(m_h)) or m_e <= 0.0 or m_h <= 0.0:
        raise ValueError("carrier masses must be finite and positive")
    alpha = m_e / (m_e + m_h)
    beta = m_h / (m_e + m_h)
    w = c * np.pi / a
    t = 1.0 / (4.0 * a)
    p = (t[:, None] + t[None, :]).ravel()
    wp = (w[:, None] * w[None, :]).ravel()
    pref = (wp[:, None] * w[None, :] / (4.0 * np.pi * (p[:, None] + t[None, :]))).ravel()
    red = (p[:, None] * t[None, :] / (p[:, None] + t[None, :])).ravel()
    out = np.empty_like(q)
    for start in range(0, q.size, 128):
        qq = q[start:start + 128]
        out[start:start + 128] = (pref[None, :] * (np.exp(-np.outer((alpha * qq) ** 2, red))
                                                   + np.exp(-np.outer((beta * qq) ** 2, red)))).sum(axis=1)
    return out

import numpy as np
import numpy.typing as npt


def polariton_branches(momenta: npt.ArrayLike, exciton_energy: float, total_mass: float, cavity_detuning: float, cavity_index: float, coupling: float) -> np.ndarray:
    """Lower and upper polariton energies and Hopfield amplitudes of the 2x2 exciton-photon problem."""
    import numpy as np
    hb2m0 = 0.0380998                  # hbar^2/(2 m0), eV nm^2
    hbarc = 197.32698                  # eV nm
    q = np.asarray(momenta, dtype=float)
    if q.ndim != 1 or q.size == 0 or not np.all(np.isfinite(q)) or np.any(q < 0.0):
        raise ValueError("momenta must be a non-empty 1D array of finite non-negative values")
    vals = [float(exciton_energy), float(total_mass), float(cavity_detuning), float(cavity_index), float(coupling)]
    if not all(np.isfinite(v) for v in vals):
        raise ValueError("all scalar inputs must be finite")
    ex0, mass, det, nc, g = vals
    if ex0 <= 0.0 or mass <= 0.0 or nc <= 0.0 or g <= 0.0 or ex0 + det <= 0.0:
        raise ValueError("energies, mass, index and coupling must be positive")
    ex = ex0 + hb2m0 * q ** 2 / mass
    ec = np.sqrt((ex0 + det) ** 2 + (hbarc * q / nc) ** 2)
    half = 0.5 * np.sqrt((ex - ec) ** 2 + 4.0 * g * g)
    elp = 0.5 * (ex + ec) - half
    eup = 0.5 * (ex + ec) + half
    rows = [elp, eup]
    for e in (elp, eup):
        xa = np.full_like(e, g)        # eigenvector (g, E - E_X) of [[E_X, g], [g, E_C]]
        ca = e - ex
        norm = np.sqrt(xa * xa + ca * ca)
        sgn = np.where(ca < 0.0, -1.0, 1.0)
        rows.extend([sgn * xa / norm, sgn * ca / norm])
    return np.stack(rows)

import numpy as np
import numpy.typing as npt


def thermal_exciton_fraction(exciton_energy: float, total_mass: float, cavity_detuning: float, cavity_index: float, coupling: float, temperature: float) -> float:
    """Mean exciton fraction of a Boltzmann polariton gas on the low-density dispersion."""
    import numpy as np
    hb2m0 = 0.0380998
    hbarc = 197.32698
    kb = 8.617333e-5
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    mass_total = float(total_mass)
    e0 = polariton_branches(np.array([0.0]), exciton_energy, total_mass, cavity_detuning, cavity_index, coupling)[0, 0]
    def _nodes():
        """Gauss-Legendre panels on [0, qmax] resolving the light cone and the thermal reservoir."""
        from numpy.polynomial.legendre import leggauss
        ec0 = exciton_energy + cavity_detuning
        q_ph = cavity_index * np.sqrt(2.0 * ec0 * (abs(cavity_detuning) + 2.0 * coupling)) / hbarc
        q_th = np.sqrt(mass_total * kb * temperature / hb2m0)
        qmax = max(40.0 * q_th, 10.0)
        edges = np.unique(np.concatenate([[0.0], np.geomspace(q_ph * 1e-3, q_ph * 30.0, 24),
                                          np.geomspace(q_ph * 30.0, qmax, 24)]))
        x, wx = leggauss(48)
        nodes, weights = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            nodes.append(0.5 * (hi - lo) * x + 0.5 * (hi + lo))
            weights.append(0.5 * (hi - lo) * wx)
        return np.concatenate(nodes), np.concatenate(weights)

    q, wq = _nodes()
    br = polariton_branches(q, exciton_energy, total_mass, cavity_detuning, cavity_index, coupling)
    kt = kb * temperature
    occ_lp = np.exp(-(br[0] - e0) / kt)
    occ_up = np.exp(-(br[1] - e0) / kt)
    meas = q * wq
    total = np.sum(meas * (occ_lp + occ_up))
    return float(np.sum(meas * (occ_lp * br[2] ** 2 + occ_up * br[4] ** 2)) / total)

import numpy as np
import numpy.typing as npt


def normal_incidence_shifts(exponents: npt.ArrayLike, coefficients: npt.ArrayLike, m_e: float, m_h: float, kappa: float, r0: float, exciton_energy: float, cavity_detuning: float, cavity_index: float, coupling: float, temperature: float, density: float) -> np.ndarray:
    """Density-induced normal-incidence shifts of the lower and upper polariton, meV."""
    import numpy as np
    hb2m0 = 0.0380998
    hbarc = 197.32698
    kb = 8.617333e-5
    temperature = float(temperature)
    density = float(density)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not np.isfinite(density) or density <= 0.0:
        raise ValueError("density must be finite and positive")
    c = np.asarray(coefficients, dtype=float)
    psi0 = float(np.sum(c))
    if not np.isfinite(psi0) or psi0 <= 0.0:
        raise ValueError("the amplitude at the origin, sum of coefficients, must be positive")
    mass_total = float(m_e) + float(m_h)
    b0 = polariton_branches(np.array([0.0]), exciton_energy, mass_total, cavity_detuning, cavity_index, coupling)[:, 0]
    def _nodes():
        """Gauss-Legendre panels on [0, qmax] resolving the light cone and the thermal reservoir."""
        from numpy.polynomial.legendre import leggauss
        ec0 = exciton_energy + cavity_detuning
        q_ph = cavity_index * np.sqrt(2.0 * ec0 * (abs(cavity_detuning) + 2.0 * coupling)) / hbarc
        q_th = np.sqrt(mass_total * kb * temperature / hb2m0)
        qmax = max(40.0 * q_th, 10.0)
        edges = np.unique(np.concatenate([[0.0], np.geomspace(q_ph * 1e-3, q_ph * 30.0, 24),
                                          np.geomspace(q_ph * 30.0, qmax, 24)]))
        x, wx = leggauss(48)
        nodes, weights = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            nodes.append(0.5 * (hi - lo) * x + 0.5 * (hi + lo))
            weights.append(0.5 * (hi - lo) * wx)
        return np.concatenate(nodes), np.concatenate(weights)

    w_ex = exchange_constant(exponents, coefficients, kappa, r0)
    q, wq = _nodes()
    br = polariton_branches(q, exciton_energy, mass_total, cavity_detuning, cavity_index, coupling)
    ov = saturation_overlap(exponents, coefficients, m_e, m_h, q)
    kt = kb * temperature
    occ_lp = np.exp(-(br[0] - b0[0]) / kt)
    occ_up = np.exp(-(br[1] - b0[0]) / kt)
    meas = q * wq / (2.0 * np.pi)
    fug = density / np.sum(meas * (occ_lp + occ_up))
    n_lp = fug * occ_lp
    n_up = fug * occ_up
    x_mean = thermal_exciton_fraction(exciton_energy, mass_total, cavity_detuning, cavity_index, coupling, temperature)
    dens_x = density * x_mean
    dens_ix = np.sum(meas * ov * (n_lp * br[2] ** 2 + n_up * br[4] ** 2))
    dens_ixc = np.sum(meas * ov * (n_lp * br[2] * br[3] + n_up * br[4] * br[5]))
    shifts = []
    for x0, c0 in ((b0[2], b0[3]), (b0[4], b0[5])):
        exch = w_ex * x0 ** 2 * dens_x
        psf = -(coupling / psi0) * (x0 * c0 * dens_ix + x0 ** 2 * dens_ixc)
        shifts.append(1.0e3 * (exch + psf))
    return np.array(shifts)

import numpy as np
import numpy.typing as npt


def polariton_splitting_change(exponents: npt.ArrayLike, m_e: float, m_h: float, kappa: float, r0: float, exciton_energy: float, cavity_detuning: float, cavity_index: float, coupling: float, temperature: float, density: float) -> float:
    """Density-induced change of the normal-incidence LP-UP splitting, meV."""
    import numpy as np
    state = exciton_ground_state(exponents, m_e, m_h, kappa, r0)
    if not state[0] < 0.0:
        raise ValueError("the screened electron-hole problem has no bound 1s state in this basis")
    shifts = normal_incidence_shifts(exponents, state[1:], m_e, m_h, kappa, r0, exciton_energy,
                                             cavity_detuning, cavity_index, coupling, temperature, density)
    return float(shifts[1] - shifts[0])
SCICODE_GOLD_EOF
