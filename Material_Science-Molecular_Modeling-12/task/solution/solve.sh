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


def reduced_state(temperature, mass_density, pressure, cv_specific, compressibility, expansion, molar_mass, cg_degree):
    K_B = 1.380649e-23
    N_A = 6.02214076e23
    if temperature <= 0 or mass_density <= 0 or pressure <= 0 or cv_specific <= 0 or compressibility <= 0 or molar_mass <= 0:
        raise ValueError("state inputs must be positive")
    if int(cg_degree) != cg_degree or cg_degree < 1:
        raise ValueError("coarse-graining degree must be a positive integer")
    rho_atoms = mass_density*N_A/molar_mass          # atoms per m^3
    c0 = rho_atoms/cg_degree                          # mesoparticles per m^3
    l_ref = c0**(-1.0/3.0)
    u_ref = l_ref**3/compressibility
    m_ref = cg_degree*molar_mass/N_A
    t_ref = np.sqrt(m_ref*l_ref**2/u_ref)
    t_star = K_B*temperature/u_ref
    p_star = pressure*compressibility
    cv_star = cv_specific*m_ref/K_B
    alpha_star = expansion*u_ref/K_B
    return np.array([t_star, p_star, cv_star, alpha_star, 1.0, 1.0, l_ref*1.0e9, u_ref*1.0e21, t_ref*1.0e12])

import numpy as np


def lth_parameters(t_star, p_star, cv_bar, alpha_bar, kappa_bar, cbar):
    if t_star <= 0 or kappa_bar <= 0 or cbar <= 0:
        raise ValueError("temperature, compressibility and density must be positive")
    if kappa_bar*cbar*t_star >= 1.0:
        raise ValueError("ideal-gas compressibility exceeds the macroscopic one")
    pi00 = p_star - cbar*t_star
    kappa = kappa_bar/(1.0 - kappa_bar*cbar*t_star)
    alpha = alpha_bar*(1.0 + kappa*cbar*t_star) - kappa*cbar
    cv = cv_bar - 1.5
    if cv <= 0:
        raise ValueError("internal heat capacity must be positive")
    return np.array([t_star, cbar, pi00, kappa, alpha, cv])

import numpy as np


def particle_volume(nb, rcut, fcut):
    nb = np.asarray(nb, dtype=float)
    if rcut <= 0 or fcut < 1.0:
        raise ValueError("rcut must be positive and fcut at least one")
    if np.any(nb <= 0):
        raise ValueError("primitive density must be positive")
    rt3 = (rcut/fcut)**3
    kp = 2.0*np.pi/15.0*rt3                       # dk/dnb
    k = kp*nb
    s = np.sqrt(k)
    at = np.arctan(1.0/s)
    lg = np.log((1.0 + k)/k)
    vol = 4.0*np.pi*rt3*(1.0/3.0 - s*(1.0 - k)*at - k*(1.0 - lg))
    v1 = -4.0*np.pi*rt3*(1.5 + (1.0 - 3.0*k)/(2.0*s)*at - lg)
    v2 = 4.0*np.pi*rt3*((1.0 + 3.0*k)/(4.0*k*s)*at - 3.0/(4.0*k))
    n = 1.0/vol
    zeta = -v1*kp/vol**2
    zeta_n = (2.0*v1*v1/vol**3 - v2/vol**2)*kp*kp
    return np.stack([n, zeta, zeta_n], axis=-1)

import numpy as np


def interaction_coefficients(nb, temperature, rcut, fcut, params):
    theta0, n00, pi00, kappa, alpha, cv = [float(x) for x in params]
    if temperature <= 0 or kappa <= 0 or n00 <= 0:
        raise ValueError("temperature, compressibility and reference density must be positive")
    pv = particle_volume(nb, rcut, fcut)
    n, zeta, zeta_n = pv[..., 0], pv[..., 1], pv[..., 2]
    pi = pi00 + alpha/kappa*(temperature - theta0) + np.log(n/n00)/kappa
    vpot = -pi00/n + alpha*theta0/(n*kappa) - (np.log(n/n00) + 1.0)/(n*kappa)
    wn = pi*zeta/n**2
    wnn = zeta**2/(kappa*n**3) + pi*zeta_n/n**2 - 2.0*pi*zeta**2/n**3
    return np.stack([n, pi, vpot, wn, wnn], axis=-1)

import numpy as np


def _grid(n_grid, dr):
    r = dr*np.arange(1, n_grid + 1)
    k = np.pi*np.arange(1, n_grid + 1)/((n_grid + 1)*dr)
    return r, k

def _fourier_pair(n_grid, dr):
    r, k = _grid(n_grid, dr)
    S = np.sin(np.outer(k, r))
    dk = k[0]
    fwd = lambda f: 4.0*np.pi*dr/k*(S @ (r*f))
    inv = lambda F: dk/(2.0*np.pi**2*r)*(S @ (k*F))
    return r, k, fwd, inv

def hnc_structure(betau, rho, dr):
    betau = np.asarray(betau, dtype=float)
    if betau.ndim != 1 or betau.size < 8:
        raise ValueError("betau must be a one-dimensional table with at least eight nodes")
    if rho <= 0 or dr <= 0:
        raise ValueError("density and grid spacing must be positive")
    n_grid = betau.size
    r, k, fwd, inv = _fourier_pair(n_grid, dr)
    def picard(gam):
        c = np.exp(-betau + gam) - 1.0 - gam
        ch = fwd(c)
        den = 1.0 - rho*ch
        if np.any(den <= 0.0) or not np.all(np.isfinite(ch)):
            return None
        return inv(rho*ch*ch/den)
    gam = np.zeros(n_grid)
    mix = 0.1
    fin, fout = [], []
    res = np.inf
    for it in range(200000):
        out = picard(gam)
        if out is None:
            gam *= 0.5; mix *= 0.5; fin, fout = [], []
            if mix < 1.0e-6:
                raise RuntimeError("HNC iteration lost the physical branch")
            continue
        res = float(np.max(np.abs(out - gam)))
        if res < 1.0e-13:
            gam = out; break
        fin.append(gam.copy()); fout.append(out.copy())
        if len(fin) > 3:
            fin.pop(0); fout.pop(0)
        accepted = False
        if len(fin) == 3 and it > 5:                    # Ng (1974) two-vector acceleration
            f = [o - i for o, i in zip(fout, fin)]
            d1, d2 = f[2] - f[1], f[2] - f[0]
            a = np.array([[d1 @ d1, d1 @ d2], [d1 @ d2, d2 @ d2]])
            b = np.array([f[2] @ d1, f[2] @ d2])
            if np.linalg.cond(a) < 1.0e14:
                c1, c2 = np.linalg.solve(a, b)
                cand = (1.0 - c1 - c2)*fout[2] + c1*fout[1] + c2*fout[0]
                test = picard(cand)
                if test is not None and float(np.max(np.abs(test - cand))) < res:
                    gam = cand; accepted = True
        if not accepted:
            gam = (1.0 - mix)*gam + mix*out
    else:
        raise RuntimeError("HNC iteration did not converge")
    return np.exp(-betau + gam)

import numpy as np


def _kernel(r, rcut):
    w = np.where(r < rcut, 15.0/(2.0*np.pi*rcut**3)*(1.0 - r/rcut)**2, 0.0)
    wp = np.where(r < rcut, -15.0/(np.pi*rcut**4)*(1.0 - r/rcut), 0.0)
    return w, wp

def _grid(n_grid, dr):
    r = dr*np.arange(1, n_grid + 1)
    k = np.pi*np.arange(1, n_grid + 1)/((n_grid + 1)*dr)
    return r, k

def self_consistent_structure(temperature, rho, rcut, fcut, params, n_grid, dr):
    if temperature <= 0 or rho <= 0 or n_grid < 8 or dr <= 0:
        raise ValueError("temperature, density, grid size and spacing must be positive")
    if n_grid*dr < rcut:
        raise ValueError("the grid must extend beyond the cutoff")
    r, k = _grid(n_grid, dr)
    w, wp = _kernel(r, rcut)
    vol = 4.0*np.pi*dr*r*r
    nb = rho*float(np.sum(vol*w))                        # g = 1 start
    g = np.ones(n_grid)
    for outer in range(500):
        coef = interaction_coefficients(nb, temperature, rcut, fcut, params)
        wn = float(coef[3])
        betau = 2.0*wn*w/temperature
        g = hnc_structure(betau, rho, dr)
        nb_new = rho*float(np.sum(vol*w*g))
        done = abs(nb_new - nb) < 1.0e-13
        nb = nb_new
        if done:
            break
    else:
        raise RuntimeError("mean-field density did not converge")
    coef = interaction_coefficients(nb, temperature, rcut, fcut, params)
    betau = 2.0*float(coef[3])*w/temperature
    return hnc_structure(betau, rho, dr)

import numpy as np


def _kernel(r, rcut):
    w = np.where(r < rcut, 15.0/(2.0*np.pi*rcut**3)*(1.0 - r/rcut)**2, 0.0)
    wp = np.where(r < rcut, -15.0/(np.pi*rcut**4)*(1.0 - r/rcut), 0.0)
    return w, wp

def _grid(n_grid, dr):
    r = dr*np.arange(1, n_grid + 1)
    k = np.pi*np.arange(1, n_grid + 1)/((n_grid + 1)*dr)
    return r, k

def _fourier_pair(n_grid, dr):
    r, k = _grid(n_grid, dr)
    S = np.sin(np.outer(k, r))
    dk = k[0]
    fwd = lambda f: 4.0*np.pi*dr/k*(S @ (r*f))
    inv = lambda F: dk/(2.0*np.pi**2*r)*(S @ (k*F))
    return r, k, fwd, inv

def eos_state(g, temperature, rho, rcut, fcut, params, dr):
    g = np.asarray(g, dtype=float)
    if g.ndim != 1 or g.size < 8:
        raise ValueError("g must be a one-dimensional table with at least eight nodes")
    if temperature <= 0 or rho <= 0 or dr <= 0:
        raise ValueError("temperature, density and spacing must be positive")
    n_grid = g.size
    r, k, fwd, inv = _fourier_pair(n_grid, dr)
    w, wp = _kernel(r, rcut)
    vol = 4.0*np.pi*dr*r*r
    nb = rho*float(np.sum(vol*w*g))
    coef = interaction_coefficients(nb, temperature, rcut, fcut, params)
    n, pi, vpot, wn, wnn = [float(x) for x in coef]
    theta0, n00, pi00, kappa, alpha, cv = [float(x) for x in params]
    h = g - 1.0
    i1 = float(np.sum(vol*r*wp*g))
    i2 = float(np.sum(vol*r*w*wp*g))
    conv = inv(fwd(w*g)*fwd(h))
    i3 = float(np.sum(vol*r*wp*g*conv))
    t1 = -wn*rho**2*i1/3.0
    t2 = -wnn*rho**2*i2/3.0
    t3 = -wnn*rho**3*i3/3.0
    p = rho*temperature + t1 + t2 + t3
    u = 1.5*temperature + vpot + cv*temperature
    return np.array([nb, n, p, t1, t2, t3, u])

import numpy as np


def _kernel(r, rcut):
    w = np.where(r < rcut, 15.0/(2.0*np.pi*rcut**3)*(1.0 - r/rcut)**2, 0.0)
    wp = np.where(r < rcut, -15.0/(np.pi*rcut**4)*(1.0 - r/rcut), 0.0)
    return w, wp

def _grid(n_grid, dr):
    r = dr*np.arange(1, n_grid + 1)
    k = np.pi*np.arange(1, n_grid + 1)/((n_grid + 1)*dr)
    return r, k

def _fourier_pair(n_grid, dr):
    r, k = _grid(n_grid, dr)
    S = np.sin(np.outer(k, r))
    dk = k[0]
    fwd = lambda f: 4.0*np.pi*dr/k*(S @ (r*f))
    inv = lambda F: dk/(2.0*np.pi**2*r)*(S @ (k*F))
    return r, k, fwd, inv

def density_fluctuations(g, temperature, rho, rcut, fcut, params, dr):
    g = np.asarray(g, dtype=float)
    if g.ndim != 1 or g.size < 8:
        raise ValueError("g must be a one-dimensional table with at least eight nodes")
    if temperature <= 0 or rho <= 0 or dr <= 0:
        raise ValueError("temperature, density and spacing must be positive")
    n_grid = g.size
    r, k, fwd, inv = _fourier_pair(n_grid, dr)
    w, wp = _kernel(r, rcut)
    vol = 4.0*np.pi*dr*r*r
    nb = rho*float(np.sum(vol*w*g))
    pv = particle_volume(nb, rcut, fcut)
    n, zeta, zeta_n = [float(x) for x in pv]
    theta0, n00, pi00, kappa, alpha, cv = [float(x) for x in params]
    h = g - 1.0
    conv = inv(fwd(w*g)*fwd(h))
    var = rho*float(np.sum(vol*w*w*g)) + rho**2*float(np.sum(vol*w*g*conv))
    rel = np.sqrt(var)/nb
    pi_zero = pi00 - alpha*theta0/kappa + np.log(n/n00)/kappa          # particle pressure at zero temperature
    v1 = pi_zero/n**2
    v2 = 1.0/(kappa*n**3) - 2.0*pi_zero/n**3
    vnn = v2*zeta**2 + v1*zeta_n
    du2 = 0.5*vnn*var
    vpot = -pi00/n + alpha*theta0/(n*kappa) - (np.log(n/n00) + 1.0)/(n*kappa)
    u1 = 1.5*temperature + vpot + cv*temperature
    return np.array([var, rel, vnn, du2, u1 + du2])

import numpy as np


def _pressure(temperature, rho, rcut, fcut, params, n_grid, dr):
    g = self_consistent_structure(temperature, rho, rcut, fcut, params, n_grid, dr)
    return eos_state(g, temperature, rho, rcut, fcut, params, dr)

def calibrate_reference_pressure(temperature, rho, rcut, fcut, params, p_target, n_grid, dr):
    params = [float(x) for x in params]
    if p_target <= 0:
        raise ValueError("target pressure must be positive")
    def resid(pi00):
        q = list(params); q[2] = pi00
        return float(_pressure(temperature, rho, rcut, fcut, q, n_grid, dr)[2]) - p_target
    p0 = params[2]
    f0 = resid(p0)
    p_init = f0 + p_target
    p1 = p0 - f0                                        # unit-slope first guess
    f1 = resid(p1)
    for it in range(100):
        if abs(f1) < 1.0e-13:
            break
        if f1 == f0:
            raise RuntimeError("secant stalled")
        p2 = p1 - f1*(p1 - p0)/(f1 - f0)
        p0, f0 = p1, f1
        p1, f1 = p2, resid(p2)
    else:
        raise RuntimeError("reference-pressure calibration did not converge")
    q = list(params); q[2] = p1
    st = _pressure(temperature, rho, rcut, fcut, q, n_grid, dr)
    return np.array([p1, p_init, float(st[1]), float(st[6])])

import numpy as np


def _pressure(temperature, rho, rcut, fcut, params, n_grid, dr):
    g = self_consistent_structure(temperature, rho, rcut, fcut, params, n_grid, dr)
    return eos_state(g, temperature, rho, rcut, fcut, params, dr)

def response_coefficients(temperature, rho, rcut, fcut, params, n_grid, dr, deltas):
    dt_rel, dn_rel = [float(x) for x in deltas]
    if dt_rel <= 0 or dn_rel <= 0:
        raise ValueError("difference steps must be positive")
    ht = dt_rel*temperature
    hn = dn_rel*rho
    p_tp = float(_pressure(temperature + ht, rho, rcut, fcut, params, n_grid, dr)[2])
    p_tm = float(_pressure(temperature - ht, rho, rcut, fcut, params, n_grid, dr)[2])
    p_np = float(_pressure(temperature, rho + hn, rcut, fcut, params, n_grid, dr)[2])
    p_nm = float(_pressure(temperature, rho - hn, rcut, fcut, params, n_grid, dr)[2])
    dpdt = (p_tp - p_tm)/(2.0*ht)
    dpdn = (p_np - p_nm)/(2.0*hn)
    kappa_impl = 1.0/(rho*dpdn)
    alpha_impl = dpdt/(rho*dpdn)
    return np.array([dpdt, dpdn, kappa_impl, alpha_impl])

import numpy as np


def _kernel(r, rcut):
    w = np.where(r < rcut, 15.0/(2.0*np.pi*rcut**3)*(1.0 - r/rcut)**2, 0.0)
    wp = np.where(r < rcut, -15.0/(np.pi*rcut**4)*(1.0 - r/rcut), 0.0)
    return w, wp

def _grid(n_grid, dr):
    r = dr*np.arange(1, n_grid + 1)
    k = np.pi*np.arange(1, n_grid + 1)/((n_grid + 1)*dr)
    return r, k

def lth_audit(state, cutoffs, molar_mass, cg_degree, n_grid, dr, deltas):
    if len(state) != 6:
        raise ValueError("state must hold temperature, mass density, pressure, specific heat, compressibility, expansion")
    if len(cutoffs) == 0:
        raise ValueError("at least one cutoff is required")
    red = reduced_state(*state, molar_mass, cg_degree)
    t_star, p_star, cv_bar, alpha_bar, kappa_bar, cbar = [float(x) for x in red[:6]]
    params = lth_parameters(t_star, p_star, cv_bar, alpha_bar, kappa_bar, cbar)
    r, k = _grid(n_grid, dr)
    vol = 4.0*np.pi*dr*r*r
    rows = []
    for rcut, fcut in cutoffs:
        w, wp = _kernel(r, rcut)
        g = self_consistent_structure(t_star, cbar, rcut, fcut, params, n_grid, dr)
        nb = cbar*float(np.sum(vol*w*g))
        pv = particle_volume(nb, rcut, fcut)
        coef = interaction_coefficients(nb, t_star, rcut, fcut, params)
        g_check = hnc_structure(2.0*float(coef[3])*w/t_star, cbar, dr)
        eos = eos_state(g, t_star, cbar, rcut, fcut, params, dr)
        fl = density_fluctuations(g, t_star, cbar, rcut, fcut, params, dr)
        rc0 = response_coefficients(t_star, cbar, rcut, fcut, params, n_grid, dr, deltas)
        cal = calibrate_reference_pressure(t_star, cbar, rcut, fcut, params, p_star, n_grid, dr)
        q = [float(x) for x in params]; q[2] = float(cal[0])
        rc1 = response_coefficients(t_star, cbar, rcut, fcut, q, n_grid, dr, deltas)
        i = int(np.argmax(g))
        rows.append([rcut, fcut, float(eos[0]), float(eos[1]), float(pv[1]), float(coef[3]), float(coef[4]),
                     float(eos[2]), float(eos[3]), float(eos[4]), float(eos[5]), float(eos[6]),
                     float(fl[0]), float(fl[1]), float(fl[3]),
                     float(g[0]), float(g[i]), float(r[i]), float(np.max(np.abs(g_check - g))),
                     float(rc0[2]), float(rc0[3]),
                     float(cal[0]), float(cal[0]) - float(params[2]), float(cal[2]), float(cal[3]), float(rc1[2]), float(rc1[3])])
    body = np.array(rows)
    head = np.zeros(body.shape[1])
    head[0] = float(np.sum(body[:, 22]))          # total calibration shift of the reference particle pressure
    head[1] = float(np.sum(body[:, 21]))
    head[2] = t_star
    head[3] = p_star
    head[4] = cv_bar
    head[5] = alpha_bar
    head[6:12] = params
    return np.vstack([head, body])
SCICODE_GOLD_EOF
