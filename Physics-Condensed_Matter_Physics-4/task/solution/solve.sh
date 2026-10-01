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


def moire_reciprocal_lattice(theta_deg: float, a_lattice: float) -> "np.ndarray":
    if not (0.0 < theta_deg < 60.0) or a_lattice <= 0.0:
        raise ValueError("need 0 < theta_deg < 60 and a_lattice > 0")
    a_m = a_lattice / (2.0 * np.sin(np.deg2rad(theta_deg) / 2.0))
    g = 4.0 * np.pi / (np.sqrt(3.0) * a_m)
    return np.array([[g * np.sqrt(3.0) / 2.0, -g / 2.0], [0.0, g]])

import numpy as np


def _exciton_form_factor(q, a_b):
    return (1.0 + (np.asarray(q, dtype=float) * a_b / 2.0) ** 2) ** -1.5


def _bohr_radius(m_e, m_h, e_b):
    mu = m_e * m_h / (m_e + m_h)
    return float(np.sqrt(38.09982 / (mu * e_b)))


def interlayer_moire_coupling(gamma_c: "np.ndarray", gamma_v: "np.ndarray", m_e: float, m_h: float,
                                      e_b: float, g0: float) -> "np.ndarray":
    a_b = _bohr_radius(m_e, m_h, e_b)
    w = np.exp(2j * np.pi / 3.0)
    v_c = gamma_c[0] + gamma_c[1] * w
    v_v = gamma_v[0] + gamma_v[1] * w
    m = m_e + m_h
    theta = v_c * _exciton_form_factor(m_h / m * g0, a_b) - np.conj(v_v) * _exciton_form_factor(m_e / m * g0, a_b)
    return np.array([theta.real, theta.imag])

import numpy as np


def _shell_basis(n_shell):
    return np.array([(i, j) for i in range(-n_shell, n_shell + 1) for j in range(-n_shell, n_shell + 1)
                     if max(abs(i), abs(j), abs(i - j)) <= n_shell], dtype=int)


def moire_exciton_bands(b_vectors: "np.ndarray", theta_coupling: "np.ndarray", m_e: float, m_h: float,
                                n_shell: int, nk: int, n_bands: int) -> tuple:
    hb2_2m0, hbar = 38.09982, 0.6582119569
    b = np.asarray(b_vectors, dtype=float)
    basis = _shell_basis(n_shell)
    if n_bands > len(basis):
        raise ValueError("n_bands exceeds the plane-wave basis size")
    g = basis @ b
    th = complex(theta_coupling[0], theta_coupling[1])
    lut = {tuple(x): n for n, x in enumerate(basis)}
    h_m = np.zeros((len(basis), len(basis)), complex)
    for n, (i, j) in enumerate(basis):
        for s in ((1, 0), (0, 1), (-1, -1)):
            m = lut.get((i + s[0], j + s[1]))
            if m is not None:
                h_m[m, n] += th
                h_m[n, m] += np.conj(th)
    mass = m_e + m_h
    f = (np.arange(nk) + 0.5) / nk
    q_pts = np.array([u * b[0] + v * b[1] for u in f for v in f])
    e_out = np.empty((len(q_pts), n_bands))
    c_out = np.empty((len(q_pts), len(basis), n_bands), complex)
    v2_out = np.empty((len(q_pts), n_bands))
    for k, q in enumerate(q_pts):
        kv = q + g
        e, c = np.linalg.eigh(h_m + np.diag(hb2_2m0 / mass * np.sum(kv ** 2, axis=1)))
        e_out[k], c_out[k] = e[:n_bands], c[:, :n_bands]
        vel = np.einsum("gn,gd->nd", np.abs(c[:, :n_bands]) ** 2, 2.0 * hb2_2m0 / mass * kv) / hbar
        v2_out[k] = np.sum(vel ** 2, axis=1)
    return q_pts, e_out, c_out, v2_out, basis

import numpy as np


def phonon_matrix_elements(q: "np.ndarray", mass_ratio: float, a_b: float, v_la: float, d1: float,
                                   d0: float, e_op: float, a_lattice: float, molar_mass: float) -> "np.ndarray":
    hbar = 0.6582119569
    q = np.asarray(q, dtype=float)
    rho = molar_mass / 6.02214076e23 / (np.sqrt(3.0) / 2.0 * (a_lattice * 1e-7) ** 2) * 6.241509074e10
    ff2 = _exciton_form_factor(mass_ratio * q, a_b) ** 2
    m_ac = hbar * d1 ** 2 * q / (2.0 * rho * v_la) * ff2
    omega = hbar * v_la * q
    m_op = hbar ** 2 * d0 ** 2 / (2.0 * rho * e_op) * ff2
    return np.array([m_ac, omega, m_op])

import numpy as np


def _material_layers():
    # (carrier, v_LA nm/ps, D1 meV, D0 meV/nm, E_op meV, a nm, molar mass g/mol)
    return (("e", 4.1, 3.4e3, 5.2e4, 36.6, 0.327, 253.86), ("h", 3.3, 2.1e3, 3.1e4, 30.8, 0.325, 341.76))


def scattering_weights(q_pts: "np.ndarray", coeffs: "np.ndarray", basis: "np.ndarray",
                               b_vectors: "np.ndarray", m_e: float, m_h: float, e_b: float,
                               temperature: float) -> tuple:
    hbar, kb = 0.6582119569, 0.08617333262
    b = np.asarray(b_vectors, dtype=float)
    nk_tot, _, nb = coeffs.shape
    a_b = _bohr_radius(m_e, m_h, e_b)
    kt = kb * temperature
    lut = {tuple(x): n for n, x in enumerate(basis)}
    shifts = _shell_basis(2)
    pref = (2.0 * np.pi / hbar) * abs(np.linalg.det(b)) / nk_tot / (2.0 * np.pi) ** 2
    n_d = len(shifts)
    w_ac = np.zeros((nk_tot, nb, nk_tot, nb, n_d, 2, 2))
    om = np.zeros((nk_tot, nk_tot, n_d, 2))
    w_op = np.zeros((nk_tot, nb, nk_tot, nb, 2, 2))
    layers = _material_layers()
    for di, d in enumerate(shifts):
        src = [n for n, x in enumerate(basis) if (x[0] + d[0], x[1] + d[1]) in lut]
        dst = [lut[(basis[n][0] + d[0], basis[n][1] + d[1])] for n in src]
        ov = np.abs(np.einsum("ign,fgm->infm", coeffs[:, src, :].conj(), coeffs[:, dst, :])) ** 2
        qn = np.linalg.norm(q_pts[None, :, :] - q_pts[:, None, :] + d @ b, axis=2)
        for li, (owner, v_la, d1, d0, eo, a_l, mm) in enumerate(layers):
            ratio = (m_h if owner == "e" else m_e) / (m_e + m_h)
            m_ac, omega, m_op = phonon_matrix_elements(qn, ratio, a_b, v_la, d1, d0, eo, a_l, mm)
            with np.errstate(divide="ignore", invalid="ignore"):
                n_ac = np.where(qn > 0.0, 1.0 / np.expm1(omega / kt), 0.0)
            n_op = 1.0 / np.expm1(eo / kt)
            om[:, :, di, li] = omega
            w_ac[:, :, :, :, di, li, 0] = pref * (m_ac * (n_ac + 1.0))[:, None, :, None] * ov
            w_ac[:, :, :, :, di, li, 1] = pref * (m_ac * n_ac)[:, None, :, None] * ov
            w_op[:, :, :, :, li, 0] += pref * (m_op * (n_op + 1.0))[:, None, :, None] * ov
            w_op[:, :, :, :, li, 1] += pref * (m_op * n_op)[:, None, :, None] * ov
    s = nk_tot * nb
    e_op = np.array([lay[4] for lay in layers])
    return w_ac.reshape(s, s, n_d, 2, 2), om, w_op.reshape(s, s, 2, 2), e_op

import numpy as np


def _broadening(x, width):
    return np.exp(-(x / width) ** 2) / (width * np.sqrt(np.pi))


def transition_rate_matrix(energies: "np.ndarray", weights: tuple, gamma: "np.ndarray") -> "np.ndarray":
    w_ac, om, w_op, e_op = weights
    nk_tot, nb = energies.shape
    e = energies.reshape(-1)
    gam = np.asarray(gamma, dtype=float).reshape(-1)
    om_s = np.repeat(np.repeat(om, nb, axis=0), nb, axis=1)
    de = e[None, :] - e[:, None]
    width = gam[:, None] + gam[None, :]
    rate = np.sum(w_ac[..., 0] * _broadening(de[:, :, None, None] + om_s, width[:, :, None, None]), axis=(2, 3))
    rate += np.sum(w_ac[..., 1] * _broadening(de[:, :, None, None] - om_s, width[:, :, None, None]), axis=(2, 3))
    rate += np.sum(w_op[..., 0] * _broadening(de[:, :, None] + e_op, width[:, :, None]), axis=2)
    rate += np.sum(w_op[..., 1] * _broadening(de[:, :, None] - e_op, width[:, :, None]), axis=2)
    return rate.T.copy()

import numpy as np


def self_consistent_dephasing(energies: "np.ndarray", weights: tuple, gamma0: float, tol: float,
                                      max_iter: int) -> "np.ndarray":
    hbar = 0.6582119569
    gam = np.full(energies.size, float(gamma0))
    for _ in range(max_iter):
        new = hbar / 2.0 * transition_rate_matrix(energies, weights, gam).sum(axis=0)
        if np.max(np.abs(new - gam)) < tol:
            return new
        gam = new
    raise ValueError("dephasing iteration did not converge")

import numpy as np
from scipy.linalg import expm


def relaxed_distribution(rates: "np.ndarray", energies: "np.ndarray", e_center: float, half_width: float,
                                 t_eval: float) -> "np.ndarray":
    e = np.asarray(energies, dtype=float).reshape(-1)
    e = e - e.min()
    n0 = (np.abs(e - e_center) <= half_width).astype(float)
    if n0.sum() == 0.0:
        raise ValueError("no state inside the excitation window")
    n0 /= n0.sum()
    return expm((rates - np.diag(rates.sum(axis=0))) * t_eval) @ n0

import numpy as np


def rta_diffusion_coefficient(v2: "np.ndarray", rates: "np.ndarray", occupation: "np.ndarray") -> float:
    v2 = np.asarray(v2, dtype=float).reshape(-1)
    n = np.asarray(occupation, dtype=float).reshape(-1)
    tau = 1.0 / rates.sum(axis=0)
    return float(0.5 * np.sum(v2 * tau * n) / np.sum(n) * 1e-2)

import numpy as np


def moire_exciton_diffusion(theta_deg: float, temperature: float, nk: int, e_center: float,
                                    half_width: float, t_eval: float) -> float:
    m_e, m_h, e_b = 0.64, 0.51, 173.0
    b = moire_reciprocal_lattice(theta_deg, 0.327)
    theta = interlayer_moire_coupling(np.array([-4.389, -6.178]), np.array([-1.467, -5.856]),
                                              m_e, m_h, e_b, float(np.linalg.norm(b[0])))
    q_pts, e, c, v2, basis = moire_exciton_bands(b, theta, m_e, m_h, 2, nk, 6)
    weights = scattering_weights(q_pts, c, basis, b, m_e, m_h, e_b, temperature)
    gam = self_consistent_dephasing(e, weights, 1.0, 1e-8, 2000)
    rates = transition_rate_matrix(e, weights, gam)
    n = relaxed_distribution(rates, e, e_center, half_width, t_eval)
    return rta_diffusion_coefficient(v2, rates, n)
SCICODE_GOLD_EOF
