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

def acceptor_energetics(E0_acceptor: float) -> np.ndarray:
    """Vacuum-referenced and hydroxide-referenced acceptor free energies."""
    _F_KCAL = 23.060548
    _E_ABS_SHE = 4.44
    _E0_OH = 1.9
    import numpy as np
    E = float(E0_acceptor)
    if not np.isfinite(E):
        raise ValueError("E0_acceptor must be finite")
    dG_absolute = -_F_KCAL * (E + _E_ABS_SHE)
    dG_bulk = -_F_KCAL * (E - _E0_OH)
    return np.array([dG_absolute, dG_bulk], dtype=float)

import numpy as np
import numpy.typing as npt

def _reduced_density(Z):
    """Water number density at depth Z, in units of the bulk density."""
    _Z_G = 0.84
    _DELTA = 1.5
    import numpy as np
    return 0.5 * (1.0 - np.tanh((np.asarray(Z, dtype=float) - _Z_G) / _DELTA))


def slab_dehydration_limits(Z_lo: float, Z_hi: float, theta_probe: float) -> np.ndarray:
    """Dehydration levels bounding a slab, and the depth of a given level."""
    _Z_G = 0.84
    _DELTA = 1.5
    import numpy as np
    zlo, zhi, tp = float(Z_lo), float(Z_hi), float(theta_probe)
    if not all(np.isfinite(v) for v in (zlo, zhi, tp)):
        raise ValueError("all arguments must be finite")
    if not zhi > zlo:
        raise ValueError("Z_hi must be strictly greater than Z_lo")
    if not (0.0 < tp < 1.0):
        raise ValueError("theta_probe must lie strictly inside (0, 1)")
    theta_lo = 1.0 - float(_reduced_density(zlo))
    theta_hi = 1.0 - float(_reduced_density(zhi))
    Z_probe = _Z_G + _DELTA * np.arctanh(2.0 * tp - 1.0)
    return np.array([theta_lo, theta_hi, Z_probe], dtype=float)

import numpy as np
import numpy.typing as npt

def _depth_from_theta_d(theta):
    """Depth in angstrom at which the dehydration level equals theta."""
    _ZG_D = 0.84
    _DEL_D = 1.5
    import numpy as np
    return _ZG_D + _DEL_D * np.arctanh(2.0 * np.asarray(theta, dtype=float) - 1.0)


def _inverse_permittivity_antiderivative(Z):
    """Antiderivative of the reciprocal local permittivity with respect to depth."""
    _ZG_D = 0.84
    _DEL_D = 1.5
    _EPS_WATER = 78.4
    import numpy as np
    amp = _EPS_WATER - 1.0
    c = 1.0 + 0.5 * amp
    d = 0.5 * amp
    scale = np.sqrt(c * c - d * d)
    phi = np.arctanh(d / c)
    u = (np.asarray(Z, dtype=float) - _ZG_D) / _DEL_D - phi
    log_cosh = np.logaddexp(u, -u) - np.log(2.0)
    return _DEL_D * (np.cosh(phi) * u + np.sinh(phi) * log_cosh) / scale


def effective_dielectric(theta: npt.ArrayLike, Z_acceptor: float) -> np.ndarray:
    """Reciprocal-mean permittivity along the donor-acceptor path."""
    import numpy as np
    th = np.asarray(theta, dtype=float)
    Za = float(Z_acceptor)
    if not np.isfinite(Za):
        raise ValueError("Z_acceptor must be finite")
    if not np.all(np.isfinite(th)):
        raise ValueError("theta must be finite")
    if np.any(th <= 0.0) or np.any(th >= 1.0):
        raise ValueError("theta must lie strictly inside (0, 1)")
    Zd = _depth_from_theta_d(th)
    separation = Zd - Za
    if np.any(separation <= 0.0):
        raise ValueError("the donor must lie strictly above the acceptor")
    path = (_inverse_permittivity_antiderivative(Zd)
            - _inverse_permittivity_antiderivative(Za))
    return separation / path

import numpy as np
import numpy.typing as npt

def ion_pair_work(theta: npt.ArrayLike, Z_acceptor: float, charge_product: float) -> np.ndarray:
    """Screened Coulomb work of the reactant pair at each dehydration level."""
    _COULOMB_KCAL = 332.0637
    import numpy as np
    th = np.asarray(theta, dtype=float)
    Za = float(Z_acceptor)
    zz = float(charge_product)
    if not (np.isfinite(Za) and np.isfinite(zz)):
        raise ValueError("Z_acceptor and charge_product must be finite")
    eps_eff = effective_dielectric(th, Za)
    separation = _depth_from_theta_d(th) - Za
    return zz * _COULOMB_KCAL / (eps_eff * separation)

import numpy as np
import numpy.typing as npt

def interfacial_driving_force(E0_acceptor: float, dG_sol_anion: float, dG_sol_radical: float, theta: npt.ArrayLike, Z_acceptor: float, charge_product: float) -> np.ndarray:
    """Standard driving force made linear in dehydration, then work-corrected."""
    import numpy as np
    E = float(E0_acceptor)
    ga = float(dG_sol_anion)
    gr = float(dG_sol_radical)
    th = np.asarray(theta, dtype=float)
    if not (np.isfinite(E) and np.isfinite(ga) and np.isfinite(gr)):
        raise ValueError("scalar arguments must be finite")
    if not np.all(np.isfinite(th)):
        raise ValueError("theta must be finite")
    if np.any(th <= 0.0) or np.any(th >= 1.0):
        raise ValueError("theta must lie strictly inside (0, 1)")
    dG_bulk = float(acceptor_energetics(E)[1])
    ddG_sol = gr - ga
    dG_separated = dG_bulk - th * ddG_sol
    work_r = ion_pair_work(th, Z_acceptor, charge_product)
    return dG_separated - work_r

import numpy as np
import numpy.typing as npt

def solvent_reorganisation_energy(lam_acceptor_outer: float, theta: npt.ArrayLike) -> np.ndarray:
    """Cross-relation mean of the equal-curvature self-exchange contributions."""
    _LAM_DONOR_INNER = 2.0
    _LAM_DONOR_OUTER = 67.0
    import numpy as np
    lam_a = float(lam_acceptor_outer)
    th = np.asarray(theta, dtype=float)
    if not np.isfinite(lam_a) or lam_a <= 0.0:
        raise ValueError("lam_acceptor_outer must be finite and positive")
    if not np.all(np.isfinite(th)):
        raise ValueError("theta must be finite")
    if np.any(th < 0.0) or np.any(th > 1.0):
        raise ValueError("theta must lie in [0, 1]")
    lam_donor = _LAM_DONOR_INNER + (1.0 - th) * _LAM_DONOR_OUTER
    return 0.5 * (lam_a + lam_donor)

import numpy as np
import numpy.typing as npt

def breathing_mode_surfaces(breathing: npt.ArrayLike) -> np.ndarray:
    """Harmonic force constants from the breathing wavenumbers, then vertical energies."""
    _AMU_KG = 1.6605390666e-27
    _C_CM_PER_S = 29979245800.0
    _J_PER_KCAL_B = 6.9476954571e-21
    _N_BONDS = 6
    import numpy as np
    b = np.asarray(breathing, dtype=float).ravel()
    if b.size != 5:
        raise ValueError("breathing must have exactly five elements")
    if not np.all(np.isfinite(b)):
        raise ValueError("breathing must be finite")
    if np.any(b <= 0.0):
        raise ValueError("wavenumbers, bond lengths and mass must be positive")
    nu_ox, nu_red, d_ox, d_red, m_lig = b
    mass = m_lig * _AMU_KG
    f_ox = mass * (2.0 * np.pi * _C_CM_PER_S * nu_ox) ** 2
    f_red = mass * (2.0 * np.pi * _C_CM_PER_S * nu_red) ** 2
    shift = (d_red - d_ox) * 1e-10
    half_n = 0.5 * _N_BONDS
    e_red_at_ox = half_n * f_red * shift ** 2 / _J_PER_KCAL_B
    e_ox_at_red = half_n * f_ox * shift ** 2 / _J_PER_KCAL_B
    return np.array([f_ox, f_red, e_red_at_ox, e_ox_at_red], dtype=float)

import numpy as np
import numpy.typing as npt

def nuclear_factor(E0_acceptor: float, lam_acceptor_outer: float, breathing: npt.ArrayLike, theta: npt.ArrayLike, Z_acceptor: float, charge_product: float) -> np.ndarray:
    """Solvent Gaussian averaged over the oxidised complex's breathing distribution."""
    _R_KCAL_N = 0.001987204259
    _T_N = 298.15
    _J_PER_KCAL_N = 6.9476954571e-21
    _DG_SOL_ANION_N = -106.4
    _DG_SOL_RADICAL_N = -3.9
    _GH_NODES = 128
    import numpy as np
    E, lam_a = float(E0_acceptor), float(lam_acceptor_outer)
    Za, zz = float(Z_acceptor), float(charge_product)
    if not all(np.isfinite(v) for v in (E, lam_a, Za, zz)):
        raise ValueError("scalar arguments must be finite")
    if lam_a <= 0.0:
        raise ValueError("lam_acceptor_outer must be positive")
    th = np.asarray(theta, dtype=float)
    if not np.all(np.isfinite(th)):
        raise ValueError("theta must be finite")
    if np.any(th <= 0.0) or np.any(th >= 1.0):
        raise ValueError("theta must lie strictly inside (0, 1)")
    surf = breathing_mode_surfaces(breathing)
    f_ox, f_red = float(surf[0]), float(surf[1])
    b = np.asarray(breathing, dtype=float).ravel()
    shift = (b[3] - b[2]) * 1e-10
    kT = _R_KCAL_N * _T_N
    kT_J = kT * _J_PER_KCAL_N
    lam_s = np.atleast_1d(solvent_reorganisation_energy(lam_a, th))
    dG = np.atleast_1d(interfacial_driving_force(E, _DG_SOL_ANION_N,
                                                         _DG_SOL_RADICAL_N, th, Za, zz))
    sigma = np.sqrt(kT_J / (6.0 * f_ox))
    x, w = np.polynomial.hermite.hermgauss(_GH_NODES)
    q = np.sqrt(2.0) * sigma * x
    gap_breath = (3.0 * f_red * (q - shift) ** 2 - 3.0 * f_ox * q ** 2) / _J_PER_KCAL_N
    arg = lam_s[:, None] + dG[:, None] + gap_breath[None, :]
    avg = (np.exp(-arg ** 2 / (4.0 * lam_s[:, None] * kT)) * w[None, :]).sum(axis=1) / np.sqrt(np.pi)
    out = avg / np.sqrt(4.0 * np.pi * lam_s * kT)
    return out.reshape(th.shape)

import numpy as np
import numpy.typing as npt

def _depth_from_theta_c(theta):
    """Depth in angstrom at which the dehydration level equals theta."""
    _ZG_C = 0.84
    _DEL_C = 1.5
    import numpy as np
    return _ZG_C + _DEL_C * np.arctanh(2.0 * np.asarray(theta, dtype=float) - 1.0)


def _density_antiderivative(Z):
    """Antiderivative of the reduced water density with respect to depth."""
    _ZG_C = 0.84
    _DEL_C = 1.5
    import numpy as np
    Zf = np.asarray(Z, dtype=float)
    x = (Zf - _ZG_C) / _DEL_C
    log_cosh = np.logaddexp(x, -x) - np.log(2.0)
    return 0.5 * (Zf - _DEL_C * log_cosh)


def tunnelling_coupling_squared(theta: npt.ArrayLike, Z_acceptor: float) -> np.ndarray:
    """Squared coupling with the tunnelling attenuation accumulated along the path."""
    _H0_CM1 = 50.0
    _BETA_WATER = 1.6
    _BETA_VACUUM = 2.9
    _J_PER_CM1 = 1.98644586e-23
    import numpy as np
    th = np.asarray(theta, dtype=float)
    Za = float(Z_acceptor)
    if not np.isfinite(Za):
        raise ValueError("Z_acceptor must be finite")
    if not np.all(np.isfinite(th)):
        raise ValueError("theta must be finite")
    if np.any(th <= 0.0) or np.any(th >= 1.0):
        raise ValueError("theta must lie strictly inside (0, 1)")
    Zd = _depth_from_theta_c(th)
    exponent = (_BETA_VACUUM * (Zd - Za)
                + (_BETA_WATER - _BETA_VACUUM)
                * (_density_antiderivative(Zd) - _density_antiderivative(Za)))
    H0_J = _H0_CM1 * _J_PER_CM1
    return H0_J ** 2 * np.exp(-exponent)

import numpy as np
import numpy.typing as npt

def dehydration_density(theta: npt.ArrayLike, Z_lo: float, Z_hi: float) -> np.ndarray:
    """Density in dehydration level induced by a uniform density in depth."""
    _DEL_P = 1.5
    import numpy as np
    th = np.asarray(theta, dtype=float)
    zlo, zhi = float(Z_lo), float(Z_hi)
    if not (np.isfinite(zlo) and np.isfinite(zhi)):
        raise ValueError("Z_lo and Z_hi must be finite")
    if not zhi > zlo:
        raise ValueError("Z_hi must be strictly greater than Z_lo")
    if not np.all(np.isfinite(th)):
        raise ValueError("theta must be finite")
    if np.any(th <= 0.0) or np.any(th >= 1.0):
        raise ValueError("theta must lie strictly inside (0, 1)")
    norm = 2.0 * (zhi - zlo) / _DEL_P
    return 1.0 / (norm * th * (1.0 - th))

import numpy as np
import numpy.typing as npt

def rate_density(E0_acceptor: float, lam_acceptor_outer: float, breathing: npt.ArrayLike, theta: npt.ArrayLike, Z_acceptor: float, charge_product: float) -> np.ndarray:
    """Golden rule: 2 pi / hbar times squared coupling times nuclear factor per joule."""
    _HBAR_JS = 1.054571817e-34
    _J_PER_KCAL_R = 6.9476954571e-21
    import numpy as np
    th = np.asarray(theta, dtype=float)
    fc_per_kcal = nuclear_factor(E0_acceptor, lam_acceptor_outer, breathing, th,
                                         Z_acceptor, charge_product)
    coupling_sq = tunnelling_coupling_squared(th, Z_acceptor)
    return (2.0 * np.pi / _HBAR_JS) * coupling_sq * (fc_per_kcal / _J_PER_KCAL_R)

import numpy as np
import numpy.typing as npt

def interfacial_rate_constant(E0_acceptor: float, lam_acceptor_outer: float, breathing: npt.ArrayLike, Z_lo: float, Z_hi: float, Z_acceptor: float, charge_product: float) -> float:
    """Average the dehydration-resolved rate over the interfacial population."""
    _ZG_O = 0.84
    _DEL_O = 1.5
    _PANELS = 48
    _NODES = 48
    import numpy as np
    E, lam_a = float(E0_acceptor), float(lam_acceptor_outer)
    zlo, zhi, za = float(Z_lo), float(Z_hi), float(Z_acceptor)
    zz = float(charge_product)
    if not all(np.isfinite(v) for v in (E, lam_a, zlo, zhi, za, zz)):
        raise ValueError("all scalar arguments must be finite")
    if lam_a <= 0.0:
        raise ValueError("lam_acceptor_outer must be positive")
    if not zhi > zlo:
        raise ValueError("Z_hi must be strictly greater than Z_lo")
    if not za < zlo:
        raise ValueError("Z_acceptor must lie strictly below Z_lo")
    breathing_mode_surfaces(breathing)

    limits = slab_dehydration_limits(zlo, zhi, 0.5)
    theta_lo, theta_hi = float(limits[0]), float(limits[1])
    z_edges = np.linspace(zlo, zhi, _PANELS + 1)
    t_edges = 0.5 * (1.0 + np.tanh((z_edges - _ZG_O) / _DEL_O))
    t_edges[0], t_edges[-1] = theta_lo, theta_hi

    x, w = np.polynomial.legendre.leggauss(_NODES)
    total = 0.0
    for a, b in zip(t_edges[:-1], t_edges[1:]):
        t = 0.5 * (b - a) * x + 0.5 * (b + a)
        weight = dehydration_density(t, zlo, zhi)
        rate = rate_density(E, lam_a, breathing, t, za, zz)
        total += float(np.dot(w, weight * rate)) * 0.5 * (b - a)
    return float(total)
SCICODE_GOLD_EOF
