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


# Oracle implementation for public function: hbn_uniaxial_permittivities
def hbn_uniaxial_permittivities(omega_cm: float) -> "np.ndarray":
    if omega_cm <= 0.0:
        raise ValueError("omega_cm must be positive")
    einf_o, wto_o, wlo_o = 4.87, 1360.0, 1614.0
    einf_e, wto_e, wlo_e = 2.95, 760.0, 825.0
    if omega_cm == wto_o or omega_cm == wto_e:
        raise ValueError("omega_cm coincides with a TO resonance")
    w2 = omega_cm ** 2
    eps_o = einf_o * (wlo_o ** 2 - w2) / (wto_o ** 2 - w2)
    eps_e = einf_e * (wlo_e ** 2 - w2) / (wto_e ** 2 - w2)
    return np.array([eps_o, eps_e])

import numpy as np


# Oracle implementation for public function: thin_film_effective_permittivity
def thin_film_effective_permittivity(d_nm: float, eps_ti: float, eps_sub: float) -> float:
    D_nm = 100.0  # paper's effective evanescent sampling length, Eq. (55)
    if d_nm <= 0.0 or d_nm >= D_nm:
        raise ValueError("film thickness must satisfy 0 < d < D")
    if eps_ti <= 0.0 or eps_sub <= 0.0:
        raise ValueError("permittivities must be positive")
    return (d_nm / D_nm) * eps_ti + (1.0 - d_nm / D_nm) * eps_sub

import numpy as np


# Oracle implementation for public function: finite_coupling_band_edge
def finite_coupling_band_edge(eps_inf_o: float, w_to: float, w_lo: float, eps2: float, alpha: float) -> float:
    if w_to <= 0.0 or w_lo <= w_to or eps_inf_o <= 0.0 or eps2 <= 0.0 or alpha < 0.0:
        raise ValueError("invalid band-edge inputs")
    eps2_alpha = eps2 + alpha ** 2 / 2.0  # Eq. (35)
    return float(np.sqrt((eps_inf_o * w_lo ** 2 + eps2_alpha * w_to ** 2) / (eps_inf_o + eps2_alpha)))  # Eq. (36)/(56)

import numpy as np


# Oracle implementation for public function: surface_wave_bandwidth_fraction
def surface_wave_bandwidth_fraction(eps_inf_o: float, w_to: float, w_lo: float, eps2: float, alpha: float) -> float:
    w_star = finite_coupling_band_edge(eps_inf_o, w_to, w_lo, eps2, alpha)
    return float((w_star - w_to) / (w_lo - w_to))  # Eq. (37)

import numpy as np


# Oracle implementation for public function: axion_dispersion_mismatch
def axion_dispersion_mismatch(n: float, omega_cm: float, eps2: float, alpha: float) -> float:
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    eps = hbn_uniaxial_permittivities(omega_cm)
    eps_o, eps_e = float(eps[0]), float(eps[1])
    if eps_o >= 0.0:
        raise ValueError("not in the type-I hyperbolic (eps_o < 0) regime")
    n2 = n ** 2
    if n2 <= max(eps2, eps_e):
        raise ValueError("n^2 must exceed max(eps2, eps_e) for localization")
    q = np.sqrt(n2 - eps2)            # Eq. (15), units of k0
    p1 = np.sqrt(n2 - eps_e)
    p2 = np.sqrt(n2 + abs(eps_o))
    big_f = (q + p1) * (abs(eps_o) / p2 - eps2 / q)   # Eqs. (21)-(24)
    return float(big_f - alpha ** 2)                    # Eq. (25)

import numpy as np


# Oracle implementation for public function: surface_wave_effective_index
def surface_wave_effective_index(omega_cm: float, eps2: float, alpha: float) -> float:
    eps = hbn_uniaxial_permittivities(omega_cm)
    eps_o, eps_e = float(eps[0]), float(eps[1])
    if eps_o >= 0.0:
        raise ValueError("not in the type-I hyperbolic (eps_o < 0) regime")
    n2_min = max(eps2, eps_e)  # Eq. (26)

    def g(x):
        return axion_dispersion_mismatch(np.sqrt(n2_min + x), omega_cm, eps2, alpha)

    lo, hi = 1e-14, 1e10
    g_lo, g_hi = g(lo), g(hi)
    if not (g_lo < 0.0 < g_hi):
        raise ValueError("no localized surface-wave solution at this wavenumber")
    llo, lhi = np.log(lo), np.log(hi)
    for _ in range(300):
        mid = 0.5 * (llo + lhi)
        if g(np.exp(mid)) < 0.0:
            llo = mid
        else:
            lhi = mid
    return float(np.sqrt(n2_min + np.exp(0.5 * (llo + lhi))))

import numpy as np


# Oracle implementation for public function: surface_wave_penetration_depths
def surface_wave_penetration_depths(n: float, omega_cm: float, eps2: float) -> "np.ndarray":
    eps = hbn_uniaxial_permittivities(omega_cm)
    eps_o, eps_e = float(eps[0]), float(eps[1])
    if eps_o >= 0.0:
        raise ValueError("not in the type-I hyperbolic (eps_o < 0) regime")
    n2 = n ** 2
    if n2 <= max(eps2, eps_e):
        raise ValueError("n^2 must exceed max(eps2, eps_e)")
    k0_per_nm = 2.0 * np.pi * omega_cm * 1e-7   # cm^-1 -> nm^-1
    q = k0_per_nm * np.sqrt(n2 - eps2)            # Eq. (15)
    p1 = k0_per_nm * np.sqrt(n2 - eps_e)
    p2 = k0_per_nm * np.sqrt(n2 + abs(eps_o))
    return np.array([1.0 / q, 1.0 / p1, 1.0 / p2])  # Eq. (43)

import numpy as np


# Oracle implementation for public function: run_hm_ti_surface_wave_pipeline
def run_hm_ti_surface_wave_pipeline(d_nm: float, eps_sub: float, eps_ti: float, window_fraction: float) -> float:
    if not (0.0 < window_fraction < 1.0):
        raise ValueError("window_fraction must lie strictly between 0 and 1")
    alpha_fs, delta_theta = 7.2973525693e-3, np.pi
    alpha = alpha_fs * delta_theta / np.pi   # Eq. (1)
    eps_inf_o, w_to, w_lo = 4.87, 1360.0, 1614.0
    eps2 = thin_film_effective_permittivity(d_nm, eps_ti, eps_sub)
    bandwidth = surface_wave_bandwidth_fraction(eps_inf_o, w_to, w_lo, eps2, alpha)
    w_op = w_to + window_fraction * bandwidth * (w_lo - w_to)
    n = surface_wave_effective_index(w_op, eps2, alpha)
    depths = surface_wave_penetration_depths(n, w_op, eps2)
    return float(depths[0] + depths[1])  # Eq. (53)
SCICODE_GOLD_EOF
