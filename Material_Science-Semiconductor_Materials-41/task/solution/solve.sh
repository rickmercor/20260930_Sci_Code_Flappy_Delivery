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


HBAR2   = 4.18005956e-3

def configuration_coordinate(delta_q, hw, delta_e):
    delta_q = float(delta_q); hw = float(hw); delta_e = float(delta_e)
    if hw <= 0.0 or delta_q <= 0.0:
        raise ValueError("hw and delta_q must be positive")
    ell2 = HBAR2/hw
    s_factor = delta_q*delta_q/(2.0*ell2)
    relax = s_factor*hw
    barrier = (delta_e - relax)**2/(4.0*relax)
    return np.array([np.sqrt(ell2), s_factor, relax, barrier, delta_q/np.sqrt(2.0*ell2)])

import numpy as np


def oscillator_basis(n_phonon, u_lo, u_hi, n_quad):
    n_phonon = int(n_phonon); n_quad = int(n_quad)
    u_lo = float(u_lo); u_hi = float(u_hi)
    if n_phonon < 1 or n_quad < 3 or not u_hi > u_lo:
        raise ValueError("n_phonon, n_quad and the quadrature window must be valid")
    u = np.linspace(u_lo, u_hi, n_quad)
    out = np.empty((n_phonon+2, n_quad))
    out[0] = u
    out[1] = np.pi**-0.25*np.exp(-0.5*u*u)
    if n_phonon >= 1:
        out[2] = np.sqrt(2.0)*u*out[1]
    for k in range(2, n_phonon+1):
        out[k+1] = np.sqrt(2.0/k)*u*out[k] - np.sqrt((k-1.0)/k)*out[k-1]
    return out

import numpy as np


DELTA_E   = 1.02

def franck_condon_overlaps(delta_q, hw, n_phonon, u_lo, u_hi, n_quad):
    if float(delta_q) <= 0.0 or float(hw) <= 0.0 or int(n_phonon) < 1:
        raise ValueError("delta_q, hw must be positive and n_phonon at least one")
    ell = float(configuration_coordinate(delta_q, hw, DELTA_E)[0])
    tab = oscillator_basis(n_phonon, u_lo, u_hi, n_quad)
    u, phi = tab[0], tab[1:]
    du = u[1]-u[0]
    shift = float(delta_q)/ell
    phi_f = oscillator_basis(n_phonon, u_lo-shift, u_hi-shift, n_quad)[1:]
    return (phi*du) @ phi_f.T

import numpy as np


DELTA_E   = 1.02

def coordinate_matrix_elements(delta_q, hw, n_phonon, u_lo, u_hi, n_quad):
    if float(delta_q) <= 0.0 or float(hw) <= 0.0 or int(n_phonon) < 1:
        raise ValueError("delta_q, hw must be positive and n_phonon at least one")
    ell = float(configuration_coordinate(delta_q, hw, DELTA_E)[0])
    tab = oscillator_basis(n_phonon, u_lo, u_hi, n_quad)
    u, phi = tab[0], tab[1:]
    du = u[1]-u[0]
    shift = float(delta_q)/ell
    phi_f = oscillator_basis(n_phonon, u_lo-shift, u_hi-shift, n_quad)[1:]
    return ((phi*u*du) @ phi_f.T)*ell

import numpy as np


K_B     = 8.617333262e-5

def lineshape_function(q_elements, hw, delta_e, temperature, broaden, n_sigma):
    q_elements = np.asarray(q_elements, dtype=float)
    hw = float(hw); delta_e = float(delta_e); broaden = float(broaden); n_sigma = float(n_sigma)
    temperature = np.atleast_1d(np.asarray(temperature, dtype=float))
    if q_elements.ndim != 2 or q_elements.shape[0] != q_elements.shape[1]:
        raise ValueError("q_elements must be a square matrix")
    if broaden <= 0.0 or np.any(temperature <= 0.0):
        raise ValueError("broaden and every temperature must be positive")
    n_max = q_elements.shape[0]-1
    m = np.arange(n_max+1, dtype=float)
    target = m + delta_e/hw
    reach = n_sigma*broaden/hw
    overlap = np.zeros(n_max+1)
    for i in range(n_max+1):
        lo = max(0, int(np.ceil(target[i] - reach)))
        hi = min(n_max, int(np.floor(target[i] + reach)))
        if hi < lo:
            continue
        n = np.arange(lo, hi+1)
        w = np.exp(-0.5*(((n-target[i])*hw/broaden)**2))
        tot = w.sum()
        if tot <= 0.0:
            continue
        overlap[i] = float(np.sum(w*q_elements[i, n]**2)/tot)
    out = np.empty((temperature.size, 3))
    for j, t in enumerate(temperature):
        x = np.exp(-hw/(K_B*t))
        rho = (1.0-x)*x**m
        weight = rho*overlap
        tot = weight.sum()
        out[j, 0] = tot
        out[j, 1] = float(np.sum((m+0.5)*hw*weight)/tot) if tot > 0.0 else np.nan
        out[j, 2] = 0.5*hw/np.tanh(0.5*hw/(K_B*t))
    return out

import numpy as np
from scipy.integrate import cumulative_simpson


K_B     = 8.617333262e-5

def nmp_cross_section(q_elements, hw, delta_e, t_grid, broaden, n_sigma,
                              t_ref, sigma_ref):
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    t_ref = float(t_ref); sigma_ref = float(sigma_ref)
    if t_grid.size < 3 or np.any(np.diff(t_grid) <= 0.0):
        raise ValueError("t_grid must be strictly increasing with at least three nodes")
    if sigma_ref <= 0.0 or t_ref <= 0.0:
        raise ValueError("t_ref and sigma_ref must be positive")
    node = int(np.argmin(np.abs(t_grid-t_ref)))
    if abs(t_grid[node]-t_ref) > 1.0e-9*max(1.0, t_ref):
        raise ValueError("t_ref must coincide with a node of t_grid")
    tab = lineshape_function(q_elements, hw, delta_e, t_grid, broaden, n_sigma)
    integrand = (tab[:, 1]-tab[:, 2])/(K_B*t_grid**2)
    cum = cumulative_simpson(integrand, x=t_grid, initial=0.0)
    cum = cum - cum[node]
    return sigma_ref*np.sqrt(t_ref/t_grid)*np.exp(cum)

import numpy as np


K_B     = 8.617333262e-5

def henry_lang_cross_section(barrier, t_grid, t_ref, sigma_ref):
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    barrier = float(barrier); t_ref = float(t_ref); sigma_ref = float(sigma_ref)
    if sigma_ref <= 0.0 or t_ref <= 0.0 or np.any(t_grid <= 0.0):
        raise ValueError("t_ref, sigma_ref and every temperature must be positive")
    return sigma_ref*(t_ref/t_grid)*np.exp(-barrier/K_B*(1.0/t_grid - 1.0/t_ref))

import numpy as np


K_B     = 8.617333262e-5

H_PLANCK = 4.135667696e-15

M_E     = 5.6095886e-32

def emission_rate(sigma, t_grid, level, mstar, degeneracy):
    sigma = np.asarray(sigma, dtype=float).ravel()
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    level = float(level); mstar = float(mstar); degeneracy = float(degeneracy)
    if sigma.size != t_grid.size:
        raise ValueError("sigma and t_grid must have the same length")
    if mstar <= 0.0 or degeneracy <= 0.0:
        raise ValueError("mstar and degeneracy must be positive")
    m_eff = mstar*M_E
    v_th = np.sqrt(3.0*K_B*t_grid/m_eff)
    n_c = 2.0*(2.0*np.pi*m_eff*K_B*t_grid/(H_PLANCK*H_PLANCK))**1.5
    return (sigma*v_th*n_c/degeneracy)*np.exp(-level/(K_B*t_grid))

import numpy as np


K_B     = 8.617333262e-5

H_PLANCK = 4.135667696e-15

M_E     = 5.6095886e-32

def arrhenius_signature(rate, t_grid, window, n_fit, mstar, degeneracy):
    rate = np.asarray(rate, dtype=float).ravel()
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    n_fit = int(n_fit); mstar = float(mstar); degeneracy = float(degeneracy)
    lo, hi = float(window[0]), float(window[1])
    if n_fit < 3 or not hi > lo:
        raise ValueError("n_fit must be at least three and the window must be ordered")
    if rate.size != t_grid.size:
        raise ValueError("rate and t_grid must have the same length")
    if mstar <= 0.0 or degeneracy <= 0.0:
        raise ValueError("mstar and degeneracy must be positive")
    inv = np.linspace(lo, hi, n_fit)
    grid_inv = 1000.0/t_grid
    y_grid = 2.0*np.log(t_grid) - np.log(rate)
    order = np.argsort(grid_inv)
    y = np.interp(inv, grid_inv[order], y_grid[order])
    a = np.polyfit(inv, y, 1)
    m_eff = mstar*M_E
    gamma = (np.sqrt(3.0*K_B/m_eff)
             * 2.0*(2.0*np.pi*m_eff*K_B/(H_PLANCK*H_PLANCK))**1.5/degeneracy)
    return np.array([a[0]*K_B*1000.0, np.exp(-a[1])/gamma,
                     float(np.std(y - np.polyval(a, inv)))])

import numpy as np


K_B     = 8.617333262e-5

ANG2_PER_CM2 = 1.0e-16

def dlts_audit(hw, delta_e, level, dq_values, mstar, degeneracy, sigma_ref,
                       n_phonon, u_lo, u_hi, n_quad, broaden, n_sigma,
                       t_lo, t_hi, n_t, t_ref, windows, n_fit):
    dq_values = [float(x) for x in dq_values]
    if len(dq_values) == 0:
        raise ValueError("at least one lattice relaxation is required")
    t_grid = np.linspace(float(t_lo), float(t_hi), int(n_t))
    rows = []
    total = 0.0
    for dq in dq_values:
        cc = configuration_coordinate(dq, hw, delta_e)
        qm = coordinate_matrix_elements(dq, hw, n_phonon, u_lo, u_hi, n_quad)
        sigma_a2 = float(sigma_ref)/ANG2_PER_CM2
        sig_nmp = nmp_cross_section(qm, hw, delta_e, t_grid, broaden, n_sigma,
                                            t_ref, sigma_a2)
        sig_hl = henry_lang_cross_section(float(cc[3]), t_grid, t_ref, sigma_a2)
        e_nmp = emission_rate(sig_nmp, t_grid, level, mstar, degeneracy)
        e_hl = emission_rate(sig_hl, t_grid, level, mstar, degeneracy)
        sig_true = float(np.interp(t_ref, t_grid, sig_nmp))
        for w in windows:
            fit_nmp = arrhenius_signature(e_nmp, t_grid, w, n_fit, mstar, degeneracy)
            fit_hl = arrhenius_signature(e_hl, t_grid, w, n_fit, mstar, degeneracy)
            decades = float(np.log10(fit_nmp[1]/sig_true))
            total += decades
            rows.append([dq, float(w[0]), float(w[1]), float(fit_nmp[0]), decades,
                         float(fit_nmp[0]) - level,
                         float(np.log10(fit_nmp[1]*ANG2_PER_CM2)),
                         float(np.log10(fit_hl[1]/sig_true)), float(cc[1])])
    fc = franck_condon_overlaps(dq_values[0], hw, n_phonon, u_lo, u_hi, n_quad)
    basis = oscillator_basis(n_phonon, u_lo, u_hi, n_quad)
    audit = np.array(rows, dtype=float)
    head = np.zeros((1, audit.shape[1]))
    head[0, 0] = total
    head[0, 1] = float(configuration_coordinate(dq_values[0], hw, delta_e)[0])
    head[0, 2] = delta_e/hw
    head[0, 3] = float(lineshape_function(
        coordinate_matrix_elements(dq_values[0], hw, n_phonon, u_lo, u_hi, n_quad),
        hw, delta_e, np.array([t_ref]), broaden, n_sigma)[0, 1])
    head[0, 4] = 0.5*hw/np.tanh(0.5*hw/(K_B*t_ref))
    head[0, 6] = float(fc[0, int(round(delta_e/hw))]**2)
    head[0, 7] = float(basis[0, 1]-basis[0, 0])
    head[0, 5] = float(lineshape_function(
        coordinate_matrix_elements(dq_values[0], hw, n_phonon, u_lo, u_hi, n_quad),
        hw, delta_e, np.array([t_ref]), broaden, n_sigma)[0, 0])
    return np.vstack([head, audit])
SCICODE_GOLD_EOF
