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


def _validated_geometry(masses, displacements):
    m = np.asarray(masses, dtype=float)
    d = np.asarray(displacements, dtype=float)
    if m.ndim != 1 or m.size == 0:
        raise ValueError("masses must be a non-empty one-dimensional array")
    if d.ndim != 2 or d.shape[1] != 3:
        raise ValueError("displacements must have shape (N, 3)")
    if d.shape[0] != m.size:
        raise ValueError("masses and displacements must describe the same atoms")
    if not (np.all(np.isfinite(m)) and np.all(np.isfinite(d))):
        raise ValueError("masses and displacements must be finite")
    if np.any(m <= 0.0):
        raise ValueError("atomic masses must be positive")
    return m, d


def mass_weighted_displacement(masses: np.ndarray, displacements: np.ndarray) -> float:
    m, d = _validated_geometry(masses, displacements)
    return float(np.sqrt(np.sum(m * np.sum(d * d, axis=1))))

import numpy as np


def configuration_coordinate_parameters(e_ground_relaxed: float, e_excited_relaxed: float,
                                                e_ground_at_excited: float, delta_q: float) -> np.ndarray:
    hbar_si = 1.054571817e-34        # J s
    ev_j = 1.602176634e-19           # J per eV
    amu_kg = 1.66053906660e-27       # kg per amu
    ang_m = 1.0e-10                  # m per Angstrom

    eg = float(e_ground_relaxed)
    ee = float(e_excited_relaxed)
    ev = float(e_ground_at_excited)
    dq = float(delta_q)
    if not all(np.isfinite(v) for v in (eg, ee, ev, dq)):
        raise ValueError("all energies and delta_q must be finite")
    if dq <= 0.0:
        raise ValueError("delta_q must be positive")
    e_zpl = ee - eg
    e_rel = ev - eg
    if e_zpl <= 0.0:
        raise ValueError("the excited state must lie above the relaxed ground state")
    if e_rel <= 0.0:
        raise ValueError("the relaxation energy must be positive")
    dq_si = dq * np.sqrt(amu_kg) * ang_m
    omega = np.sqrt(2.0 * e_rel * ev_j) / dq_si
    hbar_omega = hbar_si * omega / ev_j
    return np.array([e_zpl, hbar_omega, e_rel], dtype=float)

import numpy as np


def electron_phonon_coupling(delta_q: float, hbar_omega: float) -> np.ndarray:
    hbar_si = 1.054571817e-34
    ev_j = 1.602176634e-19
    amu_kg = 1.66053906660e-27
    ang_m = 1.0e-10

    dq = float(delta_q)
    hw = float(hbar_omega)
    if not (np.isfinite(dq) and np.isfinite(hw)):
        raise ValueError("delta_q and hbar_omega must be finite")
    if dq <= 0.0 or hw <= 0.0:
        raise ValueError("delta_q and hbar_omega must be positive")
    omega = hw * ev_j / hbar_si
    dq2_si = dq * dq * amu_kg * ang_m * ang_m
    s = dq2_si * omega / (2.0 * hbar_si)
    return np.array([s, np.exp(-s)], dtype=float)

import numpy as np


def dipole_transition_rate(dipole_debye: float, refractive_index: float,
                                   e_zpl: float, field_ratio: float = 1.0) -> float:
    hbar_si = 1.054571817e-34         # J s
    ev_j = 1.602176634e-19            # J per eV
    eps0_si = 8.8541878128e-12        # F / m
    c_si = 2.99792458e8               # m / s
    debye_cm = 3.33564095198e-30      # C m per debye

    mu_d = float(dipole_debye)
    n_r = float(refractive_index)
    e_z = float(e_zpl)
    f_r = float(field_ratio)
    if not all(np.isfinite(v) for v in (mu_d, n_r, e_z, f_r)):
        raise ValueError("all arguments must be finite")
    if mu_d <= 0.0 or n_r <= 0.0 or e_z <= 0.0:
        raise ValueError("dipole moment, refractive index and e_zpl must be positive")
    mu_si = mu_d * debye_cm
    energy = e_z * ev_j
    numerator = f_r * f_r * n_r * energy**3 * mu_si**2
    denominator = 3.0 * np.pi * eps0_si * hbar_si**4 * c_si**3
    return float(numerator / denominator)

import numpy as np


def radiative_rate_partition(gamma_r0: float, huang_rhys: float,
                                     hbar_omega: float, e_zpl: float) -> np.ndarray:
    g0 = float(gamma_r0)
    s = float(huang_rhys)
    hw = float(hbar_omega)
    ez = float(e_zpl)
    if not all(np.isfinite(v) for v in (g0, s, hw, ez)):
        raise ValueError("all arguments must be finite")
    if g0 <= 0.0 or s <= 0.0 or hw <= 0.0 or ez <= 0.0:
        raise ValueError("all arguments must be positive")
    reduction = 1.0 - s * hw / ez
    if reduction <= 0.0:
        raise ValueError("the mean vibrational energy released must stay below e_zpl")
    zpl_fraction = np.exp(-s)
    if reduction < zpl_fraction:
        raise ValueError("the corrected total radiative rate must not be below the zero-phonon line rate")
    gamma_r = g0 * reduction
    gamma_zpl = g0 * zpl_fraction
    return np.array([gamma_r, gamma_zpl, gamma_r - gamma_zpl], dtype=float)

import numpy as np


def _laguerre_ladder(top, alpha, x):
    # L_k^(alpha)(x) for k = 0 .. top by the three-term recurrence, for any real alpha.
    out = np.empty(top + 1)
    out[0] = 1.0
    if top >= 1:
        out[1] = 1.0 + alpha - x
    for k in range(1, top):
        out[k + 1] = ((2.0 * k + 1.0 + alpha - x) * out[k] - (k + alpha) * out[k - 1]) / (k + 1.0)
    return out


def coordinate_matrix_element(delta_q: float, huang_rhys: float,
                                      excited_levels: np.ndarray, level_offset: float) -> np.ndarray:
    from math import lgamma
    dq = float(delta_q)
    s = float(huang_rhys)
    p = float(level_offset)
    m = np.asarray(excited_levels)
    if not (np.isfinite(dq) and np.isfinite(s) and np.isfinite(p)):
        raise ValueError("delta_q, huang_rhys and level_offset must be finite")
    if dq <= 0.0 or s <= 0.0:
        raise ValueError("delta_q and huang_rhys must be positive")
    if p < 0.0:
        raise ValueError("level_offset must be non-negative")
    if m.dtype.kind not in "iu" or np.any(m < 0):
        raise ValueError("excited_levels must be non-negative integers")
    mf = m.astype(float)
    top = int(m.max(initial=0)) + 1
    # The coordinate about the excited-state minimum is delta_q plus the two ladder terms,
    # so each element combines the overlaps from levels m - 1, m and m + 1. Every overlap
    # into level m + p is a Laguerre polynomial in S whose upper index depends on p alone,
    # so one recurrence per upper index serves all levels, and its coefficients are
    # polynomials in the ground-state level, which fixes the continuation to real values.
    lag_p = _laguerre_ladder(top, p, s)
    lag_up = _laguerre_ladder(top, p + 1.0, s)
    lag_dn = _laguerre_ladder(top, p - 1.0, s)
    below = np.where(m >= 1, lag_up[np.maximum(m - 1, 0)], 0.0)
    bracket = lag_p[m] + 0.5 * below + (mf + 1.0) / (2.0 * s) * lag_dn[m + 1]
    lgam = np.vectorize(lgamma, otypes=[float])
    n = mf + p
    log_env = -s + n * np.log(s) - lgam(n + 1.0) + lgam(mf + 1.0) - mf * np.log(s)
    with np.errstate(divide="ignore"):
        return np.exp(2.0 * np.log(dq) + log_env + 2.0 * np.log(np.abs(bracket)))

import numpy as np


def multiphonon_rate(delta_q: float, huang_rhys: float, hbar_omega: float,
                             e_zpl: float, coupling: float, degeneracy: float,
                             temperature: float) -> float:
    hbar_evs = 6.582119569e-16        # hbar in eV s
    kb_ev = 8.617333262e-5            # Boltzmann constant in eV / K

    dq = float(delta_q)
    s = float(huang_rhys)
    hw = float(hbar_omega)
    ez = float(e_zpl)
    w = float(coupling)
    g = float(degeneracy)
    t = float(temperature)
    if not all(np.isfinite(v) for v in (dq, s, hw, ez, w, g, t)):
        raise ValueError("all arguments must be finite")
    if dq <= 0.0 or s <= 0.0 or hw <= 0.0 or ez <= 0.0 or w <= 0.0 or g <= 0.0:
        raise ValueError("delta_q, huang_rhys, hbar_omega, e_zpl, coupling and degeneracy must be positive")
    if t < 0.0 or t > 3000.0:
        raise ValueError("temperature must lie between 0 and 3000 K")
    p = ez / hw
    prefactor = 2.0 * np.pi / hbar_evs * g * w * w / hw
    if t == 0.0:
        return float(prefactor * coordinate_matrix_element(dq, s, np.array([0]), p)[0])
    x = np.exp(-hw / (kb_ev * t))
    # Each squared element out of level m is bounded by the second moment of the coordinate
    # in that level, dq^2 + dq^2 (2m + 1) / (4 S), so the Boltzmann tail beyond the last
    # level kept has a closed-form bound; double the ladder until that bound is negligible.
    q0_sq = dq * dq / (4.0 * s)
    top = 8
    while True:
        m = np.arange(top + 1)
        weights = (1.0 - x) * x**m
        total = float(np.sum(weights * coordinate_matrix_element(dq, s, m, p)))
        k = top + 1
        tail = x**k * (dq * dq + q0_sq * (2.0 * k + 1.0 + 2.0 * x / (1.0 - x)))
        if total > 0.0 and tail < 1e-15 * total:
            return float(prefactor * total)
        top *= 2

import numpy as np


def radiative_efficiency(masses: np.ndarray, displacements: np.ndarray,
                                 e_ground_relaxed: float, e_excited_relaxed: float,
                                 e_ground_at_excited: float, dipole_debye: float,
                                 refractive_index: float, field_ratio: float,
                                 coupling: float, degeneracy: float,
                                 temperature: float) -> float:
    delta_q = mass_weighted_displacement(masses, displacements)
    e_zpl, hbar_omega, _e_rel = configuration_coordinate_parameters(
        e_ground_relaxed, e_excited_relaxed, e_ground_at_excited, delta_q)
    huang_rhys, _debye_waller = electron_phonon_coupling(delta_q, hbar_omega)
    gamma_r0 = dipole_transition_rate(dipole_debye, refractive_index,
                                              e_zpl, field_ratio)
    gamma_r, _gamma_zpl, _gamma_psb = radiative_rate_partition(
        gamma_r0, huang_rhys, hbar_omega, e_zpl)
    gamma_nr = multiphonon_rate(delta_q, huang_rhys, hbar_omega, e_zpl,
                                        coupling, degeneracy, temperature)
    return float(100.0 * gamma_r / (gamma_r + gamma_nr))

import numpy as np


def thermal_quenching_temperature(masses: np.ndarray, displacements: np.ndarray,
                                          e_ground_relaxed: float, e_excited_relaxed: float,
                                          e_ground_at_excited: float, dipole_debye: float,
                                          refractive_index: float, field_ratio: float,
                                          coupling: float, degeneracy: float,
                                          fraction: float) -> float:
    from scipy.optimize import brentq
    f = float(fraction)
    if not np.isfinite(f) or f <= 0.0 or f >= 1.0:
        raise ValueError("fraction must lie strictly between 0 and 1")

    def _eta(t):
        return radiative_efficiency(masses, displacements, e_ground_relaxed,
                                            e_excited_relaxed, e_ground_at_excited,
                                            dipole_debye, refractive_index, field_ratio,
                                            coupling, degeneracy, t)

    target = f * _eta(0.0)
    # Walk up in small steps so that the first crossing is the one bracketed, then refine.
    lo, step = 0.0, 5.0
    while lo < 3000.0:
        hi = lo + step
        if _eta(hi) <= target:
            return float(brentq(lambda t: _eta(t) - target, lo, hi, xtol=1e-11, rtol=1e-15))
        lo = hi
    raise ValueError("the efficiency does not fall to the target fraction below 3000 K")
SCICODE_GOLD_EOF
