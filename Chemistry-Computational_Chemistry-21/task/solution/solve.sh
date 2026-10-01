#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def two_electron_ground_state(t: float, U: float, dv: float) -> tuple[float, float, float]:
    if t <= 0.0:
        raise ValueError("The hopping t must be positive.")
    if U < 0.0:
        raise ValueError("The on-site repulsion U must be non-negative.")

    linear = 4.0 * t * t - U * U + dv * dv
    energy = -(abs(dv) + 2.0 * (2.0**0.5) * t + 1.0)
    for _ in range(100):
        value = energy**3 - 2.0 * U * energy**2 - linear * energy + 4.0 * t * t * U
        slope = 3.0 * energy**2 - 4.0 * U * energy - linear
        energy = energy - value / slope

    slope = 3.0 * energy**2 - 4.0 * U * energy - linear
    first = 2.0 * dv * energy / slope
    slope_derivative = 6.0 * energy * first - 4.0 * U * first - 2.0 * dv
    second = ((2.0 * energy + 2.0 * dv * first) * slope - 2.0 * dv * energy * slope_derivative) / slope**2
    return (energy, 1.0 - first, -second)

def nc_ks_inversion(t: float, n: float, xi_plus: float, xi_minus: float) -> tuple[float, float, float, float]:
    if t <= 0.0:
        raise ValueError("The hopping t must be positive.")
    if xi_plus < 0.0 or xi_minus < 0.0 or 3.0 * xi_plus + xi_minus > 2.0:
        raise ValueError("The weights are not an admissible N-centered weight set.")
    offset = n - 1.0
    scale = 1.0 - xi_plus
    if abs(offset) >= scale:
        raise ValueError("The occupation is not non-interacting ensemble representable.")

    dv_s = 2.0 * t * offset / (scale * scale - offset * offset) ** 0.5
    root = (t * t + 0.25 * dv_s * dv_s) ** 0.5
    chi_s = scale * t * t / (2.0 * root**3)
    f_s_plus = 0.5 - dv_s / (4.0 * root)
    f_s_minus = 0.5 + dv_s / (4.0 * root)
    return (dv_s, chi_s, f_s_plus, f_s_minus)

def ensemble_eexx_hx_potential(t: float, U: float, n: float, xi_plus: float, xi_minus: float) -> tuple[float, float, float]:
    if t <= 0.0:
        raise ValueError("The hopping t must be positive.")
    if xi_plus < 0.0 or xi_minus < 0.0 or 3.0 * xi_plus + xi_minus > 2.0:
        raise ValueError("The weights are not an admissible N-centered weight set.")
    central_weight = 1.0 - 0.5 * (3.0 * xi_plus + xi_minus)
    offset = n - 1.0
    scale = 1.0 - xi_plus
    if abs(offset) > scale:
        raise ValueError("The occupation is not ensemble representable.")

    potential = -U * central_weight * offset / (scale * scale)
    d_plus = -U * offset * (-1.5 / (scale * scale) + 2.0 * central_weight / scale**3)
    d_minus = 0.5 * U * offset / (scale * scale)
    return (potential, d_plus, d_minus)

def ensemble_pt2_correlation_potential(t: float, U: float, n: float, xi_plus: float, xi_minus: float) -> tuple[float, float, float]:
    if t <= 0.0:
        raise ValueError("The hopping t must be positive.")
    if xi_plus < 0.0 or xi_minus < 0.0 or 3.0 * xi_plus + xi_minus > 2.0:
        raise ValueError("The weights are not an admissible N-centered weight set.")
    m = n - 1.0
    x = 1.0 - xi_plus
    if abs(m) >= x:
        raise ValueError("The occupation must lie strictly inside the representable range.")

    # G(n, xi) = d^2 F^xi / dU^2 at U = 0 = a * b * s**3, with the factors below.
    a = (2.0 - xi_minus - 3.0 * xi_plus) / (16.0 * t)
    k = 1.0 - 2.0 * xi_minus - 3.0 * xi_plus
    s2 = 1.0 - m * m / (x * x)
    s = s2**0.5
    b = m * m * k / x**3 - 1.0

    # dG/dn = a * m * s * h.
    h = 2.0 * k * s2 / x**3 - 3.0 * b / x**2
    dg_dn = a * m * s * h

    # Weight derivatives of dG/dn at fixed n.
    h_minus = -4.0 * s2 / x**3 + 6.0 * m * m / x**5
    dg_dn_minus = -m * s * h / (16.0 * t) + a * m * s * h_minus
    s_plus = -m * m / (x**3 * s)
    b_plus = m * m * (-3.0 / x**3 + 3.0 * k / x**4)
    h_plus = -6.0 * s2 / x**3 - 4.0 * k * m * m / x**6 + 6.0 * k * s2 / x**4 - 3.0 * b_plus / x**2 - 6.0 * b / x**3
    dg_dn_plus = m * (-3.0 * s * h / (16.0 * t) + a * s_plus * h + a * s * h_plus)

    prefactor = -0.5 * U * U
    return (prefactor * dg_dn, prefactor * dg_dn_plus, prefactor * dg_dn_minus)

def exact_ground_state_hxc(t: float, U: float, dv: float) -> tuple[float, float, float, float, float]:
    energy, n0, chi = two_electron_ground_state(t, U, dv)
    dv_s, chi_s, f_s_plus, f_s_minus = nc_ks_inversion(t, n0, 0.0, 0.0)
    f_hxc = 1.0 / chi_s - 1.0 / chi
    v_hx, d_plus, d_minus = ensemble_eexx_hx_potential(t, U, n0, 0.0, 0.0)
    v_c = (dv_s - dv) - v_hx
    return (n0, chi, f_hxc, v_hx, v_c)

def pt2_double_scaled_weight_derivatives(t: float, U: float, dv: float) -> tuple[float, float]:
    if U <= 0.0:
        raise ValueError("The approximation needs U > 0.")
    n0, chi, f_hxc, v_hx, v_c = exact_ground_state_hxc(t, U, dv)
    if abs(n0 - 1.0) < 1e-12:
        raise ValueError("The approximation is undefined for the symmetric dimer.")

    hx_zero, hx_plus, hx_minus = ensemble_eexx_hx_potential(t, U, n0, 0.0, 0.0)
    c_zero, c_plus, c_minus = ensemble_pt2_correlation_potential(t, U, n0, 0.0, 0.0)
    w_plus = (hx_plus / hx_zero) * v_hx + (c_plus / c_zero) * v_c
    w_minus = (hx_minus / hx_zero) * v_hx + (c_minus / c_zero) * v_c
    return (w_plus, w_minus)

def nc_zero_weight_fukui(t: float, U: float, dv: float, w: float, branch: int) -> float:
    if branch != 1 and branch != -1:
        raise ValueError("branch must be +1 (affinity) or -1 (ionization).")
    n0, chi, f_hxc, v_hx, v_c = exact_ground_state_hxc(t, U, dv)
    dv_s, chi_s, f_s_plus, f_s_minus = nc_ks_inversion(t, n0, 0.0, 0.0)
    if branch == 1:
        f_s = f_s_plus
    else:
        f_s = f_s_minus
    return (1.0 + chi * f_hxc) * f_s - 0.5 * chi * f_hxc * n0 + branch * chi * w

def pt2_double_scaled_fukui(t: float, U: float, dv: float, branch: int) -> float:
    if branch != 1 and branch != -1:
        raise ValueError("branch must be +1 (affinity) or -1 (ionization).")
    w_plus, w_minus = pt2_double_scaled_weight_derivatives(t, U, dv)
    if branch == 1:
        fukui = nc_zero_weight_fukui(t, U, dv, w_plus, 1)
    else:
        fukui = nc_zero_weight_fukui(t, U, dv, w_minus, -1)
    return round(fukui, 3)
SCICODE_GOLD_EOF
