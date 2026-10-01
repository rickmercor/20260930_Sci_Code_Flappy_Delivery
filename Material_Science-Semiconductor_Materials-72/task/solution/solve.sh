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


def grid_and_layer_profiles(n_hil: int, n_htl: int, dz_hil: float, dz_htl: float,
                                    EV_hil: float, EV_htl: float, EC_hil: float, EC_htl: float,
                                    NA_hil: float, NA_htl: float, ni_hil: float, ni_htl: float,
                                    er_hil: float, er_htl: float) -> np.ndarray:
    for m in (n_hil, n_htl):
        if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 1:
            raise ValueError("n_hil and n_htl must be integers of at least 1")
    if not (dz_hil > 0.0 and dz_htl > 0.0):
        raise ValueError("grid spacings must be positive")
    if min(NA_hil, NA_htl, ni_hil, ni_htl, er_hil, er_htl) <= 0.0:
        raise ValueError("doping, intrinsic densities and permittivities must be positive")
    n_hil = int(n_hil); n_htl = int(n_htl)
    z_hil = np.arange(n_hil + 1, dtype=float) * float(dz_hil)
    z_htl = z_hil[-1] + np.arange(1, n_htl + 1, dtype=float) * float(dz_htl)
    z = np.concatenate([z_hil, z_htl])
    in_hil = np.arange(z.size) <= n_hil
    rows = [np.where(in_hil, float(a), float(b)) for a, b in
            ((EV_hil, EV_htl), (EC_hil, EC_htl), (NA_hil, NA_htl), (ni_hil, ni_htl), (er_hil, er_htl))]
    return np.vstack([z] + rows)

import numpy as np


def equilibrium_densities(NA: np.ndarray, ni: np.ndarray) -> np.ndarray:
    NA = np.array(NA, dtype=float); ni = np.array(ni, dtype=float)
    if NA.ndim != 1 or NA.shape != ni.shape:
        raise ValueError("NA and ni must be one-dimensional arrays of the same length")
    if np.any(NA <= 0.0) or np.any(ni < 0.0):
        raise ValueError("NA must be positive and ni non-negative")
    return np.vstack([NA, ni ** 2 / NA])

import numpy as np


def poisson_potential(p: np.ndarray, n: np.ndarray, NA: np.ndarray, er: np.ndarray,
                              z: np.ndarray, phi_left: float, phi_right: float) -> np.ndarray:
    q = 1.602176634e-19
    eps0 = 8.8541878128e-14
    p = np.array(p, dtype=float); n = np.array(n, dtype=float)
    NA = np.array(NA, dtype=float); er = np.array(er, dtype=float); z = np.array(z, dtype=float)
    N = z.size
    if z.ndim != 1 or N < 3 or any(a.shape != z.shape for a in (p, n, NA, er)):
        raise ValueError("p, n, NA, er and z must be one-dimensional with a common length of at least 3")
    h = np.diff(z)
    if np.any(h <= 0.0):
        raise ValueError("z must be strictly increasing")
    if np.any(er <= 0.0):
        raise ValueError("relative permittivities must be positive")
    face = 0.5 * (er[:-1] + er[1:]) / h
    w = 2.0 / (z[2:] - z[:-2])
    i = np.arange(1, N - 1)
    A = np.zeros((N, N))
    A[i, i - 1] = w * face[:-1]
    A[i, i + 1] = w * face[1:]
    A[i, i] = -w * (face[:-1] + face[1:])
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    b = np.empty(N)
    b[0] = float(phi_left)
    b[-1] = float(phi_right)
    b[1:-1] = -(q / eps0) * (p[1:-1] - n[1:-1] - NA[1:-1])
    return np.linalg.solve(A, b)

import numpy as np


def band_edge_fields(EV0: np.ndarray, EC0: np.ndarray, phi: np.ndarray, z: np.ndarray) -> np.ndarray:
    EV0 = np.array(EV0, dtype=float); EC0 = np.array(EC0, dtype=float)
    phi = np.array(phi, dtype=float); z = np.array(z, dtype=float)
    if z.ndim != 1 or z.size < 2 or any(a.shape != z.shape for a in (EV0, EC0, phi)):
        raise ValueError("EV0, EC0, phi and z must be one-dimensional with a common length of at least 2")
    h = np.diff(z)
    if np.any(h <= 0.0):
        raise ValueError("z must be strictly increasing")
    # band edge E = E0 - q*phi (eV with phi in V); field = (1/q) dE/dz
    return np.vstack([np.diff(EV0 - phi) / h, np.diff(EC0 - phi) / h])

import numpy as np


def midpoint_densities(p: np.ndarray, n: np.ndarray, FV: np.ndarray, FC: np.ndarray,
                               n_hil: int, scheme: str) -> np.ndarray:
    p = np.array(p, dtype=float); n = np.array(n, dtype=float)
    FV = np.array(FV, dtype=float); FC = np.array(FC, dtype=float)
    if p.ndim != 1 or p.size < 2 or n.shape != p.shape:
        raise ValueError("p and n must be one-dimensional with a common length of at least 2")
    if FV.shape != (p.size - 1,) or FC.shape != (p.size - 1,):
        raise ValueError("FV and FC must have length N - 1")
    if isinstance(n_hil, bool) or not isinstance(n_hil, (int, np.integer)) or not 0 <= n_hil <= p.size - 2:
        raise ValueError("n_hil must be an integer index of a midpoint")
    if scheme not in ("mean", "field"):
        raise ValueError("scheme must be 'mean' or 'field'")
    p_mid = 0.5 * (p[:-1] + p[1:])
    n_mid = 0.5 * (n[:-1] + n[1:])
    if scheme == "field":
        j = int(n_hil)
        p_mid[j] = p[j] if FV[j] >= 0.0 else p[j + 1]
        n_mid[j] = n[j + 1] if FC[j] >= 0.0 else n[j]
    return np.vstack([p_mid, n_mid])

import numpy as np


def drift_diffusion_currents(p: np.ndarray, n: np.ndarray, p_mid: np.ndarray, n_mid: np.ndarray,
                                     FV: np.ndarray, FC: np.ndarray, mu_p: np.ndarray, mu_n: np.ndarray,
                                     z: np.ndarray, T: float) -> np.ndarray:
    q = 1.602176634e-19
    kB = 1.380649e-23
    p = np.array(p, dtype=float); n = np.array(n, dtype=float); z = np.array(z, dtype=float)
    mu_p = np.array(mu_p, dtype=float); mu_n = np.array(mu_n, dtype=float)
    mids = [np.array(a, dtype=float) for a in (p_mid, n_mid, FV, FC)]
    if z.ndim != 1 or z.size < 2 or any(a.shape != z.shape for a in (p, n, mu_p, mu_n)):
        raise ValueError("nodal arrays must be one-dimensional with a common length of at least 2")
    if any(a.shape != (z.size - 1,) for a in mids):
        raise ValueError("midpoint arrays must have length N - 1")
    h = np.diff(z)
    if np.any(h <= 0.0):
        raise ValueError("z must be strictly increasing")
    if not T > 0.0:
        raise ValueError("T must be positive")
    p_mid, n_mid, FV, FC = mids
    mp = mu_p[1:]
    mn = mu_n[1:]
    Jp = q * mp * p_mid * FV - mp * kB * T * np.diff(p) / h
    Jn = q * mn * n_mid * FC + mn * kB * T * np.diff(n) / h
    return np.vstack([Jp, Jn])

import numpy as np


def recombination_rate(p: np.ndarray, n: np.ndarray, ni: np.ndarray, tau_p: float, tau_n: float,
                               Cp: float, Cn: float) -> np.ndarray:
    p = np.array(p, dtype=float); n = np.array(n, dtype=float); ni = np.array(ni, dtype=float)
    if p.ndim != 1 or n.shape != p.shape or ni.shape != p.shape:
        raise ValueError("p, n and ni must be one-dimensional arrays of a common length")
    if not (tau_p > 0.0 and tau_n > 0.0):
        raise ValueError("lifetimes must be positive")
    if Cp < 0.0 or Cn < 0.0 or np.any(ni < 0.0):
        raise ValueError("Auger probabilities and intrinsic densities must be non-negative")
    denom = tau_n * (p + ni) + tau_p * (n + ni)
    if np.any(denom == 0.0):
        raise ValueError("the SRH denominator vanishes")
    excess = p * n - ni ** 2
    return excess / denom + (Cp * p + Cn * n) * excess

import numpy as np


def continuity_update(p: np.ndarray, n: np.ndarray, Jp: np.ndarray, Jn: np.ndarray, U: np.ndarray,
                              z: np.ndarray, dt: float) -> np.ndarray:
    q = 1.602176634e-19
    p = np.array(p, dtype=float); n = np.array(n, dtype=float); U = np.array(U, dtype=float)
    Jp = np.array(Jp, dtype=float); Jn = np.array(Jn, dtype=float); z = np.array(z, dtype=float)
    if z.ndim != 1 or z.size < 3 or any(a.shape != z.shape for a in (p, n, U)):
        raise ValueError("nodal arrays must be one-dimensional with a common length of at least 3")
    if Jp.shape != (z.size - 1,) or Jn.shape != (z.size - 1,):
        raise ValueError("current arrays must have length N - 1")
    if np.any(np.diff(z) <= 0.0):
        raise ValueError("z must be strictly increasing")
    if not dt > 0.0:
        raise ValueError("dt must be positive")
    cv = 0.5 * (z[2:] - z[:-2])
    p_new = p.copy()
    n_new = n.copy()
    p_new[1:-1] = p[1:-1] + dt * (-(Jp[1:] - Jp[:-1]) / (q * cv) - U[1:-1])
    n_new[1:-1] = n[1:-1] + dt * ((Jn[1:] - Jn[:-1]) / (q * cv) - U[1:-1])
    return np.vstack([p_new, n_new])

import numpy as np


def interfacial_hole_density(n_hil: int, n_htl: int, dz_hil: float, dz_htl: float, dt: float,
                                     n_steps: int, T: float, V: float,
                                     EV_hil: float, EV_htl: float, EC_hil: float, EC_htl: float,
                                     NA_hil: float, NA_htl: float, ni_hil: float, ni_htl: float,
                                     er_hil: float, er_htl: float, mup_hil: float, mup_htl: float,
                                     mun_hil: float, mun_htl: float, tau_p: float, tau_n: float,
                                     Cp: float, Cn: float, scheme: str) -> float:
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError("n_steps must be an integer of at least 1")
    if not dt > 0.0:
        raise ValueError("dt must be positive")
    if scheme not in ("mean", "field"):
        raise ValueError("scheme must be 'mean' or 'field'")
    prof = grid_and_layer_profiles(n_hil, n_htl, dz_hil, dz_htl, EV_hil, EV_htl, EC_hil, EC_htl,
                                           NA_hil, NA_htl, ni_hil, ni_htl, er_hil, er_htl)
    z, EV0, EC0, NA, ni, er = prof
    N = z.size
    in_hil = np.arange(N) <= n_hil
    mu_p = np.where(in_hil, float(mup_hil), float(mup_htl))
    mu_n = np.where(in_hil, float(mun_hil), float(mun_htl))
    p, n = equilibrium_densities(NA, ni)
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(int(n_steps)):
            phi = poisson_potential(p, n, NA, er, z, 0.0, V)
            FV, FC = band_edge_fields(EV0, EC0, phi, z)
            p_mid, n_mid = midpoint_densities(p, n, FV, FC, n_hil, scheme)
            Jp, Jn = drift_diffusion_currents(p, n, p_mid, n_mid, FV, FC, mu_p, mu_n, z, T)
            U = recombination_rate(p, n, ni, tau_p, tau_n, Cp, Cn)
            p, n = continuity_update(p, n, Jp, Jn, U, z, dt)
    return float(p[n_hil + 1] / 1.0e18)
SCICODE_GOLD_EOF
