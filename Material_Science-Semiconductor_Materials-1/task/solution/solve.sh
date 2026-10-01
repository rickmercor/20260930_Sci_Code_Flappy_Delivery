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

def sheet_carrier_density(sheet_resistance_ohm_per_sq: float, hall_mobility_cm2_per_Vs: float) -> float:
    Q = 1.602176634e-19
    EPS0 = 8.8541878128e-12
    HBAR = 1.054571817e-34
    ME = 9.1093837015e-31
    if not (sheet_resistance_ohm_per_sq > 0.0):
        raise ValueError('sheet resistance must be positive')
    if not (hall_mobility_cm2_per_Vs > 0.0):
        raise ValueError('mobility must be positive')
    return 1.0 / (Q * sheet_resistance_ohm_per_sq * hall_mobility_cm2_per_Vs * 1.0e-4)

import numpy as np

def _mix(gan: dict, aln: dict, key: str, x: float) -> float:
    return x * aln[key] + (1.0 - x) * gan[key]

def pseudomorphic_layer_polarization(al_fraction: float, gan: dict, aln: dict) -> float:
    Q = 1.602176634e-19
    EPS0 = 8.8541878128e-12
    HBAR = 1.054571817e-34
    ME = 9.1093837015e-31
    if not (0.0 <= al_fraction <= 1.0):
        raise ValueError('Al mole fraction must lie in [0, 1]')
    for name, d in (('gan', gan), ('aln', aln)):
        for key in ('a', 'psp', 'e31', 'e33', 'c13', 'c33', 'eps_r'):
            if key not in d:
                raise ValueError('%s is missing the key %s' % (name, key))
        if not (d['a'] > 0.0 and d['c33'] > 0.0):
            raise ValueError('%s needs a positive lattice constant and c33' % name)
    a_layer = _mix(gan, aln, 'a', al_fraction)
    strain = (gan['a'] - a_layer) / a_layer
    e31 = _mix(gan, aln, 'e31', al_fraction)
    e33 = _mix(gan, aln, 'e33', al_fraction)
    c13 = _mix(gan, aln, 'c13', al_fraction)
    c33 = _mix(gan, aln, 'c33', al_fraction)
    p_pz = 2.0 * strain * (e31 - e33 * c13 / c33)
    p_sp = _mix(gan, aln, 'psp', al_fraction)
    return abs(p_sp) + abs(p_pz)

import numpy as np

def displaced_sheet_threshold_voltage(al_fraction: float, barrier_thickness: float, sheet_separation: float,
                                              barrier_height: float, conduction_band_offset_eV: float,
                                              gan: dict, aln: dict) -> float:
    Q = 1.602176634e-19
    EPS0 = 8.8541878128e-12
    HBAR = 1.054571817e-34
    ME = 9.1093837015e-31
    if not (barrier_thickness > 0.0):
        raise ValueError('barrier thickness must be positive')
    if not (0.0 <= sheet_separation < barrier_thickness):
        raise ValueError('sheet separation must be non-negative and smaller than the barrier thickness')
    p_barrier = pseudomorphic_layer_polarization(al_fraction, gan, aln)
    p_buffer = pseudomorphic_layer_polarization(0.0, gan, aln)
    sigma = p_barrier - p_buffer
    eps_b = (al_fraction * aln['eps_r'] + (1.0 - al_fraction) * gan['eps_r']) * EPS0
    eps_gan = gan['eps_r'] * EPS0
    slab = p_buffer * (sheet_separation / 2.0) * (1.0 / eps_b + 1.0 / eps_gan)
    field = sigma * (barrier_thickness - sheet_separation) / eps_b
    return barrier_height - conduction_band_offset_eV + slab - field

import numpy as np
from scipy.linalg import eigh_tridiagonal

def _sp_sweep(potential, sheet_density, m_eff, eps, h, m_pts, diag0, off):
    """One Schrodinger-Poisson sweep on the grid: potential -> (new potential, E_F, levels, occupied count)."""
    HBAR = 1.054571817e-34
    Q = 1.602176634e-19
    levels, vecs = eigh_tridiagonal(diag0 + potential[1:-1], off, select='i', select_range=(0, 3))
    dos = m_eff / (np.pi * HBAR * HBAR)
    k = 1
    e_f = sheet_density / dos + levels[0]
    for k in range(1, 5):
        e_f = (sheet_density / dos + levels[:k].sum()) / k
        if k == 4 or e_f <= levels[k]:
            break
    occ = dos * np.clip(e_f - levels[:k], 0.0, None)
    psi = np.zeros((m_pts, k))
    psi[1:-1, :] = vecs[:, :k]
    psi /= np.sqrt(h * np.sum(psi * psi, axis=0))
    dens = (psi * psi * occ).sum(axis=1)
    field = (Q / eps) * h * np.cumsum(dens[::-1])[::-1]
    new_potential = np.concatenate(([0.0], Q * h * np.cumsum(field)[:-1]))
    return new_potential, e_f, levels, k

def channel_fermi_level(sheet_density: float, effective_mass_ratio: float, gan: dict,
                                box_length: float = 40e-9, grid_points: int = 401) -> float:
    Q = 1.602176634e-19
    EPS0 = 8.8541878128e-12
    HBAR = 1.054571817e-34
    ME = 9.1093837015e-31
    if not (sheet_density >= 0.0):
        raise ValueError('sheet density must be non-negative')
    if not (effective_mass_ratio > 0.0):
        raise ValueError('effective mass ratio must be positive')
    if not (box_length > 0.0):
        raise ValueError('box length must be positive')
    m_pts = int(grid_points)
    if m_pts < 5:
        raise ValueError('grid needs at least five points')
    m_eff = effective_mass_ratio * ME
    eps = gan['eps_r'] * EPS0
    h = box_length / (m_pts - 1)
    kin = HBAR * HBAR / (2.0 * m_eff * h * h)
    diag0 = 2.0 * kin * np.ones(m_pts - 2)
    off = -kin * np.ones(m_pts - 3)
    if sheet_density == 0.0:
        levels, _ = eigh_tridiagonal(diag0, off, select='i', select_range=(0, 0))
        return float(levels[0] / Q)
    tol = 1e-10 * Q
    beta = 0.3
    potential = np.zeros(m_pts)
    hist_x, hist_r = [], []
    e_f_old = None
    for _ in range(3000):
        new_potential, e_f, levels, k = _sp_sweep(potential, sheet_density, m_eff, eps, h, m_pts, diag0, off)
        resid = new_potential - potential
        if e_f_old is not None and abs(e_f - e_f_old) < tol and np.max(np.abs(resid)) < tol:
            return float(e_f / Q)
        e_f_old = e_f
        hist_x.append(potential.copy())
        hist_r.append(resid.copy())
        hist_x, hist_r = hist_x[-6:], hist_r[-6:]
        step = beta * resid
        if len(hist_r) > 1:      # Anderson acceleration on the last few sweeps
            d_r = np.array([hist_r[i + 1] - hist_r[i] for i in range(len(hist_r) - 1)]).T
            d_x = np.array([hist_x[i + 1] - hist_x[i] for i in range(len(hist_x) - 1)]).T
            with np.errstate(all='ignore'):
                gamma = np.linalg.lstsq(d_r, resid, rcond=None)[0]
                corr = (d_x + beta * d_r) @ gamma
            if np.all(np.isfinite(corr)):
                step = step - corr
        potential = potential + step
    raise RuntimeError('Schrodinger-Poisson iteration did not converge')

import numpy as np
from scipy.optimize import brentq

def self_consistent_sheet_density(al_fraction: float, barrier_thickness: float, sheet_separation: float,
                                          barrier_height: float, conduction_band_offset_eV: float,
                                          effective_mass_ratio: float, gan: dict, aln: dict) -> float:
    Q = 1.602176634e-19
    EPS0 = 8.8541878128e-12
    HBAR = 1.054571817e-34
    ME = 9.1093837015e-31
    if not (effective_mass_ratio > 0.0):
        raise ValueError('effective mass ratio must be positive')
    v_th = displaced_sheet_threshold_voltage(al_fraction, barrier_thickness, sheet_separation,
                                                     barrier_height, conduction_band_offset_eV, gan, aln)
    e_f_empty = channel_fermi_level(0.0, effective_mass_ratio, gan)
    if not (-v_th > e_f_empty):
        return 0.0
    eps_b = (al_fraction * aln['eps_r'] + (1.0 - al_fraction) * gan['eps_r']) * EPS0
    span = barrier_thickness
    scale = eps_b / (Q * span)

    def _residual(n):
        e_f = channel_fermi_level(n, effective_mass_ratio, gan)
        return n - scale * (-v_th - e_f)

    upper = scale * (-v_th)          # density with the Fermi level neglected; the root lies below it
    return float(brentq(_residual, 0.0, upper, xtol=1e-2, rtol=1e-13, maxiter=500))

import numpy as np
from scipy.optimize import brentq, minimize_scalar
 
def fit_sheet_separation(al_fractions: "np.ndarray", barrier_thicknesses: "np.ndarray", sheet_densities: "np.ndarray",
                                 barrier_height: float, conduction_band_offset_eV: float, effective_mass_ratio: float,
                                 gan: dict, aln: dict) -> float:
    Q = 1.602176634e-19
    EPS0 = 8.8541878128e-12
    HBAR = 1.054571817e-34
    ME = 9.1093837015e-31
    x = np.asarray(al_fractions, dtype=float).ravel()
    d = np.asarray(barrier_thicknesses, dtype=float).ravel()
    n = np.asarray(sheet_densities, dtype=float).ravel()
    if not (x.size == d.size == n.size) or x.size == 0:
        raise ValueError('the three arrays must be non-empty and of equal length')
    if not np.all(n > 0.0):
        raise ValueError('measured sheet densities must be positive')
 
    def _cost(delta):
        total = 0.0
        for xi, di, ni in zip(x, d, n):
            nm = self_consistent_sheet_density(xi, di, delta, barrier_height, conduction_band_offset_eV,
                                                       effective_mass_ratio, gan, aln)
            total += (nm / ni - 1.0) ** 2
        return total
 
    res = minimize_scalar(_cost, bounds=(0.0, 1.5e-9), method='bounded', options={'xatol': 1e-16, 'maxiter': 500})
    return float(res.x)

import numpy as np
from scipy.optimize import brentq

def required_barrier_height(al_fraction: float, barrier_thickness: float, sheet_separation: float,
                                    target_sheet_density: float, conduction_band_offset_eV: float,
                                    effective_mass_ratio: float, gan: dict, aln: dict) -> float:
    Q = 1.602176634e-19
    EPS0 = 8.8541878128e-12
    HBAR = 1.054571817e-34
    ME = 9.1093837015e-31
    if not (target_sheet_density > 0.0):
        raise ValueError('target sheet density must be positive')

    def _gap(phi):
        return self_consistent_sheet_density(al_fraction, barrier_thickness, sheet_separation, phi,
                                                     conduction_band_offset_eV, effective_mass_ratio,
                                                     gan, aln) - target_sheet_density

    lo, hi = 0.1, 5.0
    if _gap(lo) < 0.0 or _gap(hi) > 0.0:
        raise ValueError('no barrier height in [0.1, 5.0] V reproduces the target density')
    return float(brentq(_gap, lo, hi, xtol=1e-12, rtol=1e-15, maxiter=500))

import numpy as np
from scipy.optimize import brentq, minimize_scalar
 
def fitted_model_barrier_height(sheet_resistance_ohm_per_sq: float, hall_mobility_cm2_per_Vs: float,
                                      al_fraction: float, barrier_thickness: float,
                                       set_al_fractions: "np.ndarray", set_thicknesses: "np.ndarray", set_sheet_densities: "np.ndarray",
                                      literature_barrier_height: float, conduction_band_offset_eV: float,
                                      effective_mass_ratio: float, gan: dict, aln: dict) -> float:
    Q = 1.602176634e-19
    EPS0 = 8.8541878128e-12
    HBAR = 1.054571817e-34
    ME = 9.1093837015e-31
    n_meas = sheet_carrier_density(sheet_resistance_ohm_per_sq, hall_mobility_cm2_per_Vs)
    delta = fit_sheet_separation(set_al_fractions, set_thicknesses, set_sheet_densities,
                                         literature_barrier_height, conduction_band_offset_eV,
                                         effective_mass_ratio, gan, aln)
    return required_barrier_height(al_fraction, barrier_thickness, delta, n_meas,
                                           conduction_band_offset_eV, effective_mass_ratio, gan, aln)
SCICODE_GOLD_EOF
