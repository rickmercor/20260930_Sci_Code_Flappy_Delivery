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
def ml_parameters_and_horizon(mass: float, chi: float, b: float) -> np.ndarray:
    """Reference implementation."""
    if mass <= 0.0:
        raise ValueError("mass must be positive")
    if abs(chi) > 1.0:
        raise ValueError("|chi| must not exceed 1")
    if b < 0.0:
        raise ValueError("b must be non-negative")
    m = mass + b ** 2 * mass ** 3 * (6.0 - 5.0 * chi ** 2) / 4.0
    a = mass * chi + b ** 2 * mass ** 3 * chi * (2.0 - chi ** 2) / 4.0
    i1 = 1.0 - 0.5 * b * b * a * a
    i2 = 1.0 - b * b * a * a
    p0 = 2.0 * i2 * (i1 * i1 + b * b * m * m) / (i1 * i1 * (1.0 + np.sqrt(i2)))
    disc = 4.0 * i2 * (m * m - a * a) - a ** 6 * b ** 4
    if disc < 0.0:
        raise ValueError("no horizon for the given parameters")
    r_plus = ((1.0 + i2) * (2.0 * i2 * m + np.sqrt(disc))
              / (a ** 4 * b ** 4 + 4.0 * i2 * (1.0 - b * b * m * m)))
    eps1 = np.sqrt(i2)
    eps2 = np.sqrt(1.0 + b * b * r_plus * r_plus)
    if b == 0.0:
        gamma, lam1, lam2 = 0.0, 1.0, 0.0
    else:
        gamma = np.sqrt((1.0 + eps1 + eps1 ** 2 - eps1 ** 3) * eps2 ** 2
                        + 2.0 * (eps1 - 1.0) * eps1 * eps2 - 2.0 * eps1)
        s = eps1 * eps1 * eps2 - 2.0 * eps1 + eps2
        lam1 = ((2.0 * eps1 * (eps1 + 1.0) * (eps2 * eps2 - 1.0) - gamma * gamma)
                / (gamma * np.sqrt(eps2 * (eps1 + 1.0) * s)))
        lam2 = ((2.0 * eps1 * eps2 + 2.0 * eps1 - (eps1 + 1.0) ** 2 * eps2 * eps2)
                * np.sqrt(eps2 * (1.0 - eps1) * s)
                / ((eps1 + 1.0) * (eps2 * eps2 - 1.0) * gamma))
    return np.array([m, a, i1, i2, p0, eps1, eps2, gamma, lam1, lam2, r_plus], dtype=float)

import numpy as np


def ml_metric_components(r: float, x: float, params: np.ndarray, b: float) -> np.ndarray:
    """Reference implementation."""
    if np.real(r) <= 0.0:
        raise ValueError("r must be positive")
    if abs(np.real(x)) >= 1.0:
        raise ValueError("|x| must be less than 1")
    m, a, i1, i2, p0 = params[0], params[1], params[2], params[3], params[4]
    lam1, lam2 = params[8], params[9]
    sig = r * r + a * a * x * x
    dl = (1.0 - b * b * m * m * i2 / (i1 * i1)) * r * r - 2.0 * m * i2 / i1 * r + a * a
    q = (1.0 + b * b * r * r) * dl
    om2 = (1.0 + b * b * r * r) - b * b * x * x * dl
    om = np.sqrt(om2)
    pf = 1.0 + b * b * (m * m * i2 / (i1 * i1) - a * a) * x * x
    mf = (1.0 / (2.0 * i1 ** 3 * om)) * (
        om * (i1 * i2 * m * r * x * x
              * (b * b * m * r * (b * b * r * r + x * x) + 2.0 * i1 * (b * b * r * r + 1.0))
              - i1 ** 3 * sig * (b * b * r * r * x * x + 1.0))
        + (b * b * r * r + 1.0)
        * (i1 ** 3 * sig * (a * a * b * b * x * x - 1.0)
           + i2 * m * r * x * x
           * (i2 * b ** 4 * m * m * r * r * x * x + 3.0 * i1 * i2 * b * b * m * r * x * x
              - i1 * i1 * (3.0 * a * a * b * b * x * x + b * b * r * r * x * x - 2.0))))
    lf = (1.0 / (4.0 * i1 * i1 * om2 * om2)) * (
        sig * (b ** 4 * r * r * m * m * i2 * x * x
               + i1 * i1 * (2.0 + a * a * b ** 4 * r * r * x * x
                            + b * b * (r * r - a * a * x * x - r * r * x * x)))
        + b * b * r * m * i2 * x * x
        * (2.0 * a * a * i1 * x * x + b * b * r ** 3 * m * i2 * x * x
           + 2.0 * r * r * i1 * (2.0 - a * a * b * b * x * x))
        + 2.0 * om * i1 * (sig * i1 + b * b * r ** 3 * m * i2 * x * x))
    den = r ** 4 * (1.0 - x * x) * pf - a * a * x ** 4 * q
    a_t = (a * a * x * x * mf + lf * r * r * om2 * (1.0 - x * x) * pf) / den
    b_t = a * (r * r * mf + lf * q * om2 * x * x) / den
    c1, c2 = a * x * x / p0, r * r / p0
    f1, f2 = -q / (lf * om2 * om2), (1.0 - x * x) * pf / (om2 * om2 * lf)
    g_tt = f1 * a_t * a_t + f2 * b_t * b_t
    g_tp = f1 * a_t * c1 + f2 * b_t * c2
    g_pp = f1 * c1 * c1 + f2 * c2 * c2
    k = 2.0 * a / ((1.0 + np.sqrt(i2)) * p0)
    j_tt = lam1 - k * lam2 * b
    j_tp = -k
    j_pt = lam2 * b
    return np.array([
        j_tt * j_tt * g_tt + 2.0 * j_tt * j_pt * g_tp + j_pt * j_pt * g_pp,
        lf / q,
        lf / ((1.0 - x * x) * pf),
        j_tp * j_tp * g_tt + 2.0 * j_tp * g_tp + g_pp,
        j_tt * j_tp * g_tt + (j_tt + j_tp * j_pt) * g_tp + j_pt * g_pp,
    ])

import numpy as np


def inverse_metric_and_hamiltonian(r: float, x: float, params: np.ndarray, b: float, p_cov: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    if np.real(r) <= 0.0:
        raise ValueError("r must be positive")
    if abs(np.real(x)) >= 1.0:
        raise ValueError("|x| must be less than 1")
    p_cov = np.asarray(p_cov)
    if p_cov.shape != (4,):
        raise ValueError("p_cov must have shape (4,)")
    g_tt, g_rr, g_xx, g_pp, g_tp = ml_metric_components(r, x, params, b)
    det = g_tt * g_pp - g_tp * g_tp
    i_tt = g_pp / det
    i_rr = 1.0 / g_rr
    i_xx = 1.0 / g_xx
    i_pp = g_tt / det
    i_tp = -g_tp / det
    p_t, p_r, p_x, p_ph = p_cov[0], p_cov[1], p_cov[2], p_cov[3]
    ham = 0.5 * (i_tt * p_t * p_t + i_rr * p_r * p_r + i_xx * p_x * p_x
                 + i_pp * p_ph * p_ph + 2.0 * i_tp * p_t * p_ph)
    return np.array([i_tt, i_rr, i_xx, i_pp, i_tp, ham])

import numpy as np


def zamo_flow_four_velocity(r: float, x: float, params: np.ndarray, b: float, r_in: float, v_max: float, p3: float, psi: float, lam: float) -> np.ndarray:
    """Reference implementation."""
    if r <= 0.0:
        raise ValueError("r must be positive")
    if abs(x) >= 1.0:
        raise ValueError("|x| must be less than 1")
    g_tt, g_rr, g_xx, g_pp, g_tp = ml_metric_components(r, x, params, b)
    kappa = np.sqrt(-g_tt + g_tp * g_tp / g_pp)
    omega = -g_tp / g_pp
    v_r = -v_max * (r_in / r) ** p3
    v_ph = psi * np.sqrt(1.0 / r) / (1.0 + lam / r)
    speed2 = v_r * v_r + v_ph * v_ph
    if speed2 >= 1.0:
        raise ValueError("three-velocity is not subluminal")
    gam = 1.0 / np.sqrt(1.0 - speed2)
    return np.array([gam / kappa,
                     gam * v_r / np.sqrt(g_rr),
                     0.0,
                     gam * (omega / kappa + v_ph / np.sqrt(g_pp))])

import numpy as np


def _plateau_effective_radius(r, r_in, r_p, w_p, j_p):
    """Effective radius of the plateau substitution, anchored at r_in."""
    anchor = np.tanh((r_in - r_p) / w_p)
    return float(r - j_p * w_p * (np.tanh((r - r_p) / w_p) - anchor))


def _disk_emissivity(r, x, r_in, p1, p2, sigma_dtheta, beta, r_p, w_p, j_p):
    """Attenuated, flaring disk background with the radial law at the effective radius."""
    r_eff = _plateau_effective_radius(r, r_in, r_p, w_p, j_p)
    if r_eff <= 0.0:
        raise ValueError("effective radius must be positive")
    theta = np.arccos(x)
    ll = np.log(r_eff / r_in)
    sig = sigma_dtheta * np.exp(beta * r)
    return float(np.exp(p1 * ll + p2 * ll * ll
                        - (theta - np.pi / 2.0) ** 2 / (2.0 * sig * sig)))


def _bump_emissivity(r, x, r_b, sigma_br, sigma_btheta):
    """Ring-like Gaussian bump; axisymmetric, centred on the equator."""
    theta = np.arccos(x)
    return float(np.exp(-(r - r_b) ** 2 / (2.0 * sigma_br ** 2)
                        - (theta - np.pi / 2.0) ** 2 / (2.0 * sigma_btheta ** 2)))


def _spot_emissivity(r, x, phi, r_s, theta_s, phi_s, sigma_sr, sigma_stheta, sigma_sphi):
    """Localized spot with a 2 pi periodic azimuthal factor."""
    theta = np.arccos(x)
    azimuthal = 1.0 - np.cos(phi - phi_s)
    return float(np.exp(-(r - r_s) ** 2 / (2.0 * sigma_sr ** 2)
                        - (theta - theta_s) ** 2 / (2.0 * sigma_stheta ** 2)
                        - azimuthal / (2.0 * sigma_sphi ** 2)))


def emission_and_absorption(r: float, x: float, phi: float, r_in: float, emis_params: dict, absorb_params: dict) -> np.ndarray:
    """Reference implementation."""
    if abs(x) > 1.0:
        raise ValueError("|x| must not exceed 1")
    if r_in <= 0.0:
        raise ValueError("r_in must be positive")
    e, al = emis_params, absorb_params
    for key in ("j0", "j1", "j2", "j3", "p1", "p2", "sigma_dtheta", "beta", "r_p", "w_p",
                "j_p", "r_b", "sigma_br", "sigma_btheta", "r_s", "theta_s", "phi_s",
                "sigma_sr", "sigma_stheta", "sigma_sphi"):
        if key not in e:
            raise ValueError(f"emis_params is missing the key {key!r}")
    for key in ("alpha0", "alpha1", "alpha2", "alpha3"):
        if key not in al:
            raise ValueError(f"absorb_params is missing the key {key!r}")
    for key in ("w_p", "sigma_dtheta", "sigma_br", "sigma_btheta",
                "sigma_sr", "sigma_stheta", "sigma_sphi"):
        if e[key] <= 0.0:
            raise ValueError(f"{key} must be positive")
    j_d = _disk_emissivity(r, x, r_in, e["p1"], e["p2"], e["sigma_dtheta"],
                           e["beta"], e["r_p"], e["w_p"], e["j_p"])
    j_b = _bump_emissivity(r, x, e["r_b"], e["sigma_br"], e["sigma_btheta"])
    j_s = _spot_emissivity(r, x, phi, e["r_s"], e["theta_s"], e["phi_s"],
                           e["sigma_sr"], e["sigma_stheta"], e["sigma_sphi"])
    j_nu = e["j0"] * (e["j1"] * j_d + e["j2"] * j_b + e["j3"] * j_s)
    a_nu = al["alpha0"] * (al["alpha1"] * j_d + al["alpha2"] * j_b + al["alpha3"] * j_s)
    return np.array([float(j_nu), float(a_nu)])

import numpy as np


def screen_to_initial_conditions(alpha: float, beta: float, params: np.ndarray, b: float, r_o: float, inclination: float) -> np.ndarray:
    """Reference implementation."""
    if r_o <= 0.0:
        raise ValueError("r_o must be positive")
    if not (0.0 < inclination < np.pi):
        raise ValueError("inclination must lie strictly between 0 and pi")
    x_o = np.cos(inclination)
    g_tt, g_rr, g_xx, g_pp, g_tp = ml_metric_components(r_o, x_o, params, b)
    omega = -g_tp / g_pp
    lapse = np.sqrt((g_tp * g_tp - g_tt * g_pp) / g_pp)
    norm = np.sqrt(1.0 + (alpha * alpha + beta * beta) / (r_o * r_o))
    n_r = -1.0 / norm
    n_ph = -(alpha / r_o) / norm
    n_x = (beta / r_o) / norm
    p_ph = n_ph * np.sqrt(g_pp)
    p_r = n_r * np.sqrt(g_rr)
    p_x = n_x * np.sqrt(g_xx)
    p_t = lapse - omega * p_ph
    return np.array([0.0, r_o, x_o, 0.0, p_t, p_r, p_x, p_ph])

import numpy as np


def _inverse_metric_derivatives(r, x, params, b, cs_step=1e-20):
    """Complex-step derivatives of the contravariant metric w.r.t. r and x.

    The increment is a default argument rather than a module-level constant:
    the grading namespace keeps imports and function definitions only.
    """
    zero_p = np.zeros(4)
    d_r = [c.imag / cs_step for c in inverse_metric_and_hamiltonian(
        complex(r, cs_step), complex(x, 0.0), params, b, zero_p)[:5]]
    d_x = [c.imag / cs_step for c in inverse_metric_and_hamiltonian(
        complex(r, 0.0), complex(x, cs_step), params, b, zero_p)[:5]]
    return d_r, d_x


def transfer_rhs(tau: float, state: np.ndarray, params: np.ndarray, b: float, r_in: float, emis_params: dict, absorb_params: dict, flow_params: dict) -> np.ndarray:
    """Reference implementation."""
    state = np.asarray(state, dtype=float)
    if state.shape != (10,):
        raise ValueError("state must have shape (10,)")
    if state[1] <= 0.0:
        raise ValueError("the radial coordinate must be positive")
    if abs(state[2]) >= 1.0:
        raise ValueError("the polar coordinate must satisfy |x| < 1")
    if r_in <= 0.0:
        raise ValueError("r_in must be positive")
    for key in ("v_max", "p3", "psi", "lam"):
        if key not in flow_params:
            raise ValueError(f"flow_params is missing the key {key!r}")
    r, x, phi = state[1], state[2], state[3]
    p_cov = state[4:8]
    p_t, p_r, p_x, p_ph = p_cov[0], p_cov[1], p_cov[2], p_cov[3]
    rho = state[8]
    inv = inverse_metric_and_hamiltonian(r, x, params, b, p_cov)
    i_tt, i_rr, i_xx, i_pp, i_tp = inv[0], inv[1], inv[2], inv[3], inv[4]
    d_r, d_x = _inverse_metric_derivatives(r, x, params, b)
    quad = (p_t * p_t, p_r * p_r, p_x * p_x, p_ph * p_ph, 2.0 * p_t * p_ph)
    dp_r = -0.5 * sum(d_r[i] * quad[i] for i in range(5))
    dp_x = -0.5 * sum(d_x[i] * quad[i] for i in range(5))
    u = zamo_flow_four_velocity(r, x, params, b, r_in, flow_params["v_max"],
                                        flow_params["p3"], flow_params["psi"],
                                        flow_params["lam"])
    redshift = 1.0 / (p_t * u[0] + p_r * u[1] + p_x * u[2] + p_ph * u[3])
    j_nu, a_nu = emission_and_absorption(r, x, phi, r_in, emis_params, absorb_params)
    return np.array([i_tt * p_t + i_tp * p_ph,
                     i_rr * p_r,
                     i_xx * p_x,
                     i_pp * p_ph + i_tp * p_t,
                     0.0, dp_r, dp_x, 0.0,
                     a_nu / redshift,
                     redshift * redshift * j_nu * np.exp(-rho)])

from typing import Sequence
import numpy as np
from scipy.integrate import solve_ivp


def line_summed_intensity(alphas: Sequence[float], beta: float, mass: float, chi: float, b: float, r_o: float, inclination: float, emis_params: dict, absorb_params: dict, flow_params: dict, rtol: float, atol: float) -> float:
    """Reference implementation."""
    if len(alphas) == 0:
        raise ValueError("alphas must not be empty")
    if rtol <= 0.0 or atol <= 0.0:
        raise ValueError("tolerances must be positive")
    params = ml_parameters_and_horizon(mass, chi, b)
    r_in = params[10]

    def _captured(tau, state, *args):
        return state[1] - 1.0001 * r_in
    _captured.terminal = True
    _captured.direction = -1

    def _escaped(tau, state, *args):
        return state[1] - (r_o - 1e-6)
    _escaped.terminal = True
    _escaped.direction = +1

    total = 0.0
    for alpha in alphas:
        ic = screen_to_initial_conditions(float(alpha), beta, params, b,
                                                  r_o, inclination)
        state0 = np.concatenate([ic, [0.0, 0.0]])
        sol = solve_ivp(transfer_rhs, (0.0, 5.0e4), state0, method="DOP853",
                        rtol=rtol, atol=atol, events=(_captured, _escaped),
                        args=(params, b, r_in, emis_params, absorb_params, flow_params))
        total += float(sol.y[9, -1])      # captured or escaped: keep what was accumulated
    return float(total)
SCICODE_GOLD_EOF
