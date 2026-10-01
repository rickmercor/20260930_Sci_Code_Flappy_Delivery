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

def barker_henderson_radius(sigma: float, epsilon: float, lambda_a: float, lambda_r: float, temperature: float) -> float:
    if sigma <= 0 or epsilon <= 0 or temperature <= 0:
        raise ValueError("sigma, epsilon and temperature must be positive")
    if lambda_r <= lambda_a:
        raise ValueError("lambda_r must exceed lambda_a")
    n = 20001
    r = np.linspace(1e-8, sigma, n)
    pref = lambda_r/(lambda_r - lambda_a)*(lambda_r/lambda_a)**(lambda_a/(lambda_r - lambda_a))
    with np.errstate(over='ignore'):
        phi = epsilon*pref*((sigma/r)**lambda_r - (sigma/r)**lambda_a)
    integrand = 1.0 - np.exp(-phi/temperature)
    return float(0.5*np.trapezoid(integrand, r))

import numpy as np

def steele_slit_potential(z: "np.ndarray", pore_width: float, sigma_f: float, epsilon_f: float, sigma_s: float, epsilon_s: float, rho_s: float, delta_s: float, alpha: float, temperature: float) -> "np.ndarray":
    if pore_width <= 0 or temperature <= 0:
        raise ValueError("pore_width and temperature must be positive")
    z = np.asarray(z, dtype=float)
    s_c = 0.5*(sigma_f + sigma_s)
    e_c = np.sqrt(epsilon_f*epsilon_s)

    def _one_wall(d):
        out = np.full(d.shape, np.inf)
        m = d > 1e-9
        dd = d[m]
        with np.errstate(over='ignore'):
            out[m] = 2.0*np.pi*rho_s*e_c*s_c**2*delta_s*(
                0.4*(s_c/dd)**10 - (s_c/dd)**4
                - s_c**4/(3.0*delta_s*(dd + alpha*delta_s)**3))
        return out

    return (_one_wall(z) + _one_wall(pore_width - z))/temperature

import numpy as np

def fmt_weighted_densities(rho: "np.ndarray", radius: float, dz: float) -> "np.ndarray":
    if radius <= 0 or dz <= 0:
        raise ValueError("radius and dz must be positive")
    rho = np.asarray(rho, dtype=float)
    k = int(np.ceil(radius/dz - 0.5)) + 1
    j = np.arange(-k, k + 1)
    a = np.clip((j - 0.5)*dz, -radius, radius)
    b = np.clip((j + 0.5)*dz, -radius, radius)
    w3 = np.pi*((radius**2*b - b**3/3.0) - (radius**2*a - a**3/3.0))
    w2 = 2.0*np.pi*radius*(b - a)
    wv2 = np.pi*(b**2 - a**2)
    def _c(a, w):
        full = np.convolve(a, w)
        off = (w.size - 1)//2
        return full[off:off + a.size]
    n2 = _c(rho, w2)
    n3 = _c(rho, w3)
    nv2 = _c(rho, wv2)
    n1 = n2/(4.0*np.pi*radius)
    n0 = n2/(4.0*np.pi*radius**2)
    nv1 = nv2/(4.0*np.pi*radius)
    return np.vstack([n0, n1, n2, n3, nv1, nv2])

import numpy as np

def white_bear_partials(weighted: "np.ndarray") -> "np.ndarray":
    w = np.asarray(weighted, dtype=float)
    if w.shape[0] != 6:
        raise ValueError("weighted must have six rows")
    n0, n1, n2, n3, nv1, nv2 = w
    x = np.clip(n3, 0.0, 1.0 - 1e-10)
    om = 1.0 - x
    small = x < 1e-5
    xs = np.where(small, 1.0, x)
    oms = 1.0 - xs
    num = xs + oms**2*np.log(oms)
    nump = xs - 2.0*oms*np.log(oms)
    P = np.where(small, 1.5 - x/3.0 - x**2/12.0 - x**3/30.0, num/xs**2)
    dP = np.where(small, -1.0/3.0 - x/6.0 - x**2/10.0, (nump*xs - 2.0*num)/xs**3)
    f = P/(36.0*np.pi*om**2)
    df = (dP*om + 2.0*P)/(36.0*np.pi*om**3)
    d0 = -np.log(om)
    d1 = n2/om
    d2 = n1/om + f*(3.0*n2**2 - 3.0*nv2**2)
    d3 = n0/om + (n1*n2 - nv1*nv2)/om**2 + df*(n2**3 - 3.0*n2*nv2**2)
    dv1 = -nv2/om
    dv2 = -nv1/om - 6.0*f*n2*nv2
    return np.vstack([d0, d1, d2, d3, dv1, dv2])

import numpy as np

def hard_sphere_functional_derivative(rho: "np.ndarray", radius: float, dz: float) -> "np.ndarray":
    if radius <= 0 or dz <= 0:
        raise ValueError("radius and dz must be positive")
    rho = np.asarray(rho, dtype=float)
    parts = white_bear_partials(fmt_weighted_densities(rho, radius, dz))
    d0, d1, d2, d3, dv1, dv2 = parts
    k = int(np.ceil(radius/dz - 0.5)) + 1
    j = np.arange(-k, k + 1)
    a = np.clip((j - 0.5)*dz, -radius, radius)
    b = np.clip((j + 0.5)*dz, -radius, radius)
    w3 = np.pi*((radius**2*b - b**3/3.0) - (radius**2*a - a**3/3.0))
    w2 = 2.0*np.pi*radius*(b - a)
    wv2 = np.pi*(b**2 - a**2)
    w1 = w2/(4.0*np.pi*radius)
    w0 = w2/(4.0*np.pi*radius**2)
    wv1 = wv2/(4.0*np.pi*radius)
    def _c(arr, w):
        full = np.convolve(arr, w)
        off = (w.size - 1)//2
        return full[off:off + arr.size]
    return (_c(d0, w0) + _c(d1, w1) + _c(d2, w2) + _c(d3, w3)
            - _c(dv1, wv1) - _c(dv2, wv2))

import numpy as np

def bulk_residual_chemical_potential(rho_bulk: float, radius: float) -> float:
    if rho_bulk < 0 or radius <= 0:
        raise ValueError("rho_bulk must be non-negative and radius positive")
    g0 = 1.0
    g1 = radius
    g2 = 4.0*np.pi*radius**2
    g3 = (4.0/3.0)*np.pi*radius**3
    w = np.array([[rho_bulk*g0], [rho_bulk*g1], [rho_bulk*g2], [rho_bulk*g3], [0.0], [0.0]])
    d0, d1, d2, d3, _, _ = white_bear_partials(w)
    return float(d0[0]*g0 + d1[0]*g1 + d2[0]*g2 + d3[0]*g3)

import numpy as np

def equilibrium_density_profile(beta_v_ext: "np.ndarray", rho_bulk: float, radius: float, dz: float) -> "np.ndarray":
    if rho_bulk <= 0 or radius <= 0 or dz <= 0:
        raise ValueError("rho_bulk, radius and dz must be positive")
    bv = np.asarray(beta_v_ext, dtype=float)
    mu_res = bulk_residual_chemical_potential(rho_bulk, radius)
    rho = np.where(np.isfinite(bv), rho_bulk, 0.0)
    for _ in range(20000):
        dF = hard_sphere_functional_derivative(rho, radius, dz)
        with np.errstate(invalid='ignore'):
            arg = np.where(np.isfinite(bv), mu_res - dF - bv, -np.inf)
        new = rho_bulk*np.exp(np.clip(arg, -700.0, 20.0))
        err = np.max(np.abs(new - rho))
        rho = 0.90*rho + 0.10*new
        if err < 1e-12:
            break
    return rho

import numpy as np

def contact_pair_correlation(weighted: "np.ndarray", hs_diameter: float) -> "np.ndarray":
    w = np.asarray(weighted, dtype=float)
    if w.shape[0] != 6:
        raise ValueError("weighted must have six rows")
    if hs_diameter <= 0:
        raise ValueError("hs_diameter must be positive")
    n0, n1, n2, n3, nv1, nv2 = w
    d = hs_diameter
    zeta = 1.0 - (nv2**2)/np.where(n2 > 0.0, n2**2, 1.0)
    zeta = np.clip(zeta, 0.0, 1.0)
    om = np.clip(1.0 - n3, 1e-10, None)
    g = (1.0/om
         + (d/4.0)*n2*zeta/om**2
         + (d**2/72.0)*n2**2*zeta/om**3)
    return np.vstack([zeta, g])

import numpy as np

def association_strength(g_contact: "np.ndarray", hs_diameter: float, sigma: float, epsilon_hb: float, r_site: float, r_cut: float, temperature: float) -> "np.ndarray":
    if min(hs_diameter, sigma, r_site, r_cut, temperature) <= 0:
        raise ValueError("hs_diameter, sigma, r_site, r_cut and temperature must be positive")
    g = np.asarray(g_contact, dtype=float)
    d, rc, rd = hs_diameter, r_cut, r_site
    t1 = np.log((rc + 2.0*rd)/d)*(6.0*rc**3 + 18.0*rc**2*rd - 24.0*rd**3)
    t2 = (rc + 2.0*rd - d)*(22.0*rd**2 - 5.0*rc*rd - 7.0*rd*d - 8.0*rc**2 + rc*d + d**2)
    kappa = 4.0*np.pi*d**2*(t1 + t2)/(72.0*rd**2*sigma**3)
    return sigma**3*(np.exp(epsilon_hb/temperature) - 1.0)*kappa*g

import numpy as np

def nonbonded_site_fraction(n0: "np.ndarray", zeta: "np.ndarray", delta: "np.ndarray") -> "np.ndarray":
    a = np.asarray(n0, dtype=float)
    b = np.asarray(zeta, dtype=float)
    c = np.asarray(delta, dtype=float)
    if not (a.shape == b.shape == c.shape):
        raise ValueError("n0, zeta and delta must have the same shape")
    u = a*b*c
    return 2.0/(1.0 + np.sqrt(1.0 + 8.0*u))

import numpy as np

def confined_nonbonded_fraction(pore_width: float, rho_bulk: float, temperature: float, n_grid: int) -> float:
    if pore_width <= 0 or rho_bulk <= 0 or temperature <= 0:
        raise ValueError("pore_width, rho_bulk and temperature must be positive")
    if n_grid < 2:
        raise ValueError("n_grid must be at least two")
    sigma, eps, la, lr = 3.161, 488.75, 6.0, 52.367
    eps_hb, rd, rc = 1210.0, 0.4*3.161, 0.5834*3.161
    sig_s, eps_s, rho_s, del_s, alpha = 3.4, 28.0, 0.114, 3.35, 0.61

    radius = barker_henderson_radius(sigma, eps, la, lr, temperature)
    d = 2.0*radius
    z = np.linspace(0.0, pore_width, int(n_grid))
    dz = z[1] - z[0]
    bv = steele_slit_potential(z, pore_width, sigma, eps, sig_s, eps_s,
                               rho_s, del_s, alpha, temperature)
    rho = equilibrium_density_profile(bv, rho_bulk, radius, dz)
    w = fmt_weighted_densities(rho, radius, dz)
    zg = contact_pair_correlation(w, d)
    delta = association_strength(zg[1], d, sigma, eps_hb, rd, rc, temperature)
    chi = nonbonded_site_fraction(w[0], zg[0], delta)
    return float(np.trapezoid(rho*chi, z)/np.trapezoid(rho, z))
SCICODE_GOLD_EOF
