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


def electrostatic_scales(eps_r: float, T_K: float, a_ang: float, sigma_e_per_A2: float) -> "np.ndarray":
    eps_r = float(eps_r)
    T_K = float(T_K)
    a_ang = float(a_ang)
    sigma = float(sigma_e_per_A2)
    if eps_r <= 0.0 or T_K <= 0.0 or a_ang <= 0.0:
        raise ValueError("eps_r, T_K and a_ang must all be strictly positive")
    e_charge = 1.602176634e-19
    k_boltzmann = 1.380649e-23
    eps_vacuum = 8.8541878128e-12
    lB = e_charge ** 2 / (4.0 * np.pi * eps_vacuum * eps_r * k_boltzmann * T_K) * 1.0e10
    a3 = a_ang ** 3
    zeta = 2.0 * np.pi * lB * a3 * sigma * sigma
    return np.array([lB, a3, zeta])

import numpy as np


def reservoir_composition(z: "np.ndarray", v: "np.ndarray", phi_free: "np.ndarray") -> "np.ndarray":
    zz = np.atleast_1d(np.asarray(z, dtype=float))
    vv = np.atleast_1d(np.asarray(v, dtype=float))
    free = np.atleast_1d(np.asarray(phi_free, dtype=float))
    n = zz.size
    if n < 2 or vv.size != n or free.size != n - 1:
        raise ValueError("z and v need N >= 2 entries and phi_free N - 1 entries")
    if np.any(vv <= 0.0) or np.any(free <= 0.0):
        raise ValueError("relative volumes and specified fractions must be strictly positive")
    if zz[-1] == 0.0:
        raise ValueError("the closing species must carry charge")
    # neutrality holds for number densities phi_i / (a**3 v_i), not for volume fractions
    charge = float((zz[:-1] * free / vv[:-1]).sum())
    last = -charge * vv[-1] / zz[-1]
    if last <= 0.0:
        raise ValueError("no positive fraction of the last species neutralises the reservoir")
    phi = np.concatenate([free, [last]])
    solvent = 1.0 - float(phi.sum())
    if solvent <= 0.0:
        raise ValueError("the ion volume fractions fill the whole lattice")
    return np.concatenate([phi, [solvent]])

import numpy as np
from scipy.optimize import brentq


def local_composition(z: "np.ndarray", v: "np.ndarray", phi_bulk: "np.ndarray", Psi: float) -> "np.ndarray":
    zz = np.atleast_1d(np.asarray(z, dtype=float))
    vv = np.atleast_1d(np.asarray(v, dtype=float))
    pb = np.atleast_1d(np.asarray(phi_bulk, dtype=float))
    if not (zz.size == vv.size == pb.size):
        raise ValueError("z, v and phi_bulk must have the same length")
    if np.any(vv <= 0.0) or np.any(pb <= 0.0):
        raise ValueError("relative volumes and reservoir fractions must be strictly positive")
    eb = 1.0 - float(pb.sum())
    if eb <= 0.0:
        raise ValueError("reservoir fractions must sum to less than one")
    Psi = float(Psi)
    if Psi == 0.0:
        return np.concatenate([pb, [0.0]])
    # phi_i = phi_i^b exp(-z_i Psi + v_i d) with d = ln(eta / eta^b); filling the lattice leaves one
    # increasing convex equation in d, written with expm1 so the far field keeps its digits
    f = lambda d: eb * np.expm1(d) + float((pb * np.expm1(-zz * Psi + vv * d)).sum())
    hi = -np.log(eb)
    lo = -1e-12
    while f(lo) > 0.0:
        lo = 4.0 * lo - 1.0
    d = brentq(f, lo, hi, xtol=1e-300, rtol=4.0 * np.finfo(float).eps, maxiter=1000)
    return np.concatenate([pb * np.exp(-zz * Psi + vv * d), [d]])

import numpy as np
from scipy.optimize import brentq


def _squared_field(z, v, phi_bulk, Psi):
    """a**3 / (8 pi l_B) (dPsi/dx)**2 at local potential Psi, from the uniform pressure."""
    st = local_composition(z, v, phi_bulk, Psi)
    d = float(st[-1])
    w = 1.0 - 1.0 / v
    return -d - float((w * phi_bulk * np.expm1(-z * float(Psi) + v * d)).sum())


def wall_state(z: "np.ndarray", v: "np.ndarray", phi_bulk: "np.ndarray", zeta: float) -> "np.ndarray":
    zz = np.atleast_1d(np.asarray(z, dtype=float))
    vv = np.atleast_1d(np.asarray(v, dtype=float))
    pb = np.atleast_1d(np.asarray(phi_bulk, dtype=float))
    zeta = float(zeta)
    if zeta < 0.0:
        raise ValueError("zeta must be non-negative")
    eb = 1.0 - float(pb.sum())
    if zeta == 0.0:
        return np.concatenate([[0.0], pb, [eb]])
    hi = 1.0
    while _squared_field(zz, vv, pb, hi) < zeta:
        hi *= 1.6
    Ps = brentq(lambda P: _squared_field(zz, vv, pb, P) - zeta, 0.0, hi,
                xtol=1e-300, rtol=4.0 * np.finfo(float).eps, maxiter=1000)
    st = local_composition(zz, vv, pb, Ps)
    return np.concatenate([[Ps], st[:-1], [eb * np.exp(st[-1])]])

import numpy as np
from scipy.integrate import solve_ivp


def _moment_rhs(s, y, zz, vv, pb, a3, kfac):
    """d/ds of the distance from the wall and of the running moments, with s = ln(Psi)."""
    n = zz.size
    P = np.exp(s)
    st = local_composition(zz, vv, pb, P)
    dxds = -P / np.sqrt(kfac * _squared_field(zz, vv, pb, P))
    exc = (st[:n] - pb) / (a3 * vv)
    return np.concatenate([[dxds], exc * dxds, y[0] * exc * dxds])


def adsorption_moments(eps_r: float, T_K: float, a_ang: float, sigma_e_per_A2: float, z: "np.ndarray", v: "np.ndarray", phi_bulk: "np.ndarray") -> "np.ndarray":
    zz = np.atleast_1d(np.asarray(z, dtype=float))
    vv = np.atleast_1d(np.asarray(v, dtype=float))
    pb = np.atleast_1d(np.asarray(phi_bulk, dtype=float))
    sigma = float(sigma_e_per_A2)
    if sigma <= 0.0:
        raise ValueError("the surface charge density must be strictly positive")
    lB, a3, zeta = electrostatic_scales(eps_r, T_K, a_ang, sigma)
    n = zz.size
    Ps = float(wall_state(zz, vv, pb, zeta)[0])
    kfac = 8.0 * np.pi * lB / a3
    psi_far = 1e-7
    sol = solve_ivp(_moment_rhs, (np.log(Ps), np.log(psi_far)), np.zeros(1 + 2 * n),
                    method="DOP853", rtol=1e-12, atol=1e-20, args=(zz, vv, pb, a3, kfac))
    y = sol.y[:, -1]
    x_far = y[0]
    G = y[1:1 + n].copy()
    M = y[1 + n:].copy()
    # beyond psi_far the field and every excess are linear in Psi, which decays as exp(-kappa x)
    kappa = np.sqrt(kfac * _squared_field(zz, vv, pb, psi_far)) / psi_far
    slope = (local_composition(zz, vv, pb, psi_far)[:n] - pb) / (a3 * vv) / psi_far
    G += slope * psi_far / kappa
    M += slope * psi_far * (x_far / kappa + 1.0 / kappa ** 2)
    ratio = float((zz * G).sum() / (-sigma))
    return np.concatenate([G, M / G, [ratio, Ps]])

import numpy as np
from scipy.optimize import brentq


def _distance_gap(s, eps_r, T_K, a_ang, zz, v, phi_bulk, i, j):
    """Mean adsorption distance of species i minus that of species j at surface charge s."""
    n = zz.size
    m = adsorption_moments(eps_r, T_K, a_ang, s, zz, v, phi_bulk)
    return float(m[n + i] - m[n + j])


def exchange_charge(eps_r: float, T_K: float, a_ang: float, z: "np.ndarray", v: "np.ndarray", phi_bulk: "np.ndarray", i: int, j: int, sigma_lo: float, sigma_hi: float) -> float:
    zz = np.atleast_1d(np.asarray(z, dtype=float))
    n = zz.size
    i = int(i)
    j = int(j)
    lo = float(sigma_lo)
    hi = float(sigma_hi)
    if i == j or not (0 <= i < n and 0 <= j < n):
        raise ValueError("i and j must be distinct valid species indices")
    if not (0.0 < lo < hi):
        raise ValueError("the bracket must satisfy 0 < sigma_lo < sigma_hi")
    args = (eps_r, T_K, a_ang, zz, v, phi_bulk, i, j)
    if _distance_gap(lo, *args) * _distance_gap(hi, *args) > 0.0:
        raise ValueError("the mean adsorption distances do not exchange inside the bracket")
    return float(brentq(_distance_gap, lo, hi, args=args, xtol=1e-300, rtol=1e-12, maxiter=200))

import numpy as np


def stratification_onset(eps_r: float, T_K: float, a_ang: float, z: "np.ndarray", v: "np.ndarray", phi_free: "np.ndarray", sigma_lo_e_per_nm2: float, sigma_hi_e_per_nm2: float) -> float:
    zz = np.atleast_1d(np.asarray(z, dtype=float))
    vv = np.atleast_1d(np.asarray(v, dtype=float))
    lo = float(sigma_lo_e_per_nm2)
    hi = float(sigma_hi_e_per_nm2)
    if not (0.0 < lo < hi):
        raise ValueError("the window must satisfy 0 < sigma_lo < sigma_hi")
    n = zz.size
    pb = reservoir_composition(zz, vv, phi_free)[:n]
    counter = np.where(zz < 0.0)[0]
    if counter.size < 2:
        raise ValueError("at least two counterion species are needed")
    alpha = np.abs(zz[counter]) / vv[counter]
    if np.unique(alpha).size != alpha.size:
        raise ValueError("counterions with equal charge per volume have no stratified order")
    inner_to_outer = counter[np.argsort(-alpha)]

    # the order is strict iff every neighbouring pair in alpha order is strict, so the onset is
    # the last exchange of any neighbouring pair inside the window
    s_lo, s_hi = lo / 100.0, hi / 100.0
    n_grid = int(np.ceil(np.log(s_hi / s_lo) / np.log(1.2))) + 1
    grid = np.geomspace(s_lo, s_hi, n_grid)
    X = np.array([adsorption_moments(eps_r, T_K, a_ang, s, zz, vv, pb)[n:2 * n] for s in grid])
    onset = s_lo
    for k in range(inner_to_outer.size - 1):
        inner, outer = int(inner_to_outer[k]), int(inner_to_outer[k + 1])
        gap = X[:, outer] - X[:, inner]
        if gap[-1] <= 0.0:
            raise ValueError("the counterions are not ordered by charge per volume at sigma_hi")
        bad = np.where(gap <= 0.0)[0]
        if bad.size == 0:
            continue
        m = int(bad[-1])
        if gap[m] == 0.0:
            cross = grid[m]
        else:
            cross = exchange_charge(eps_r, T_K, a_ang, zz, vv, pb, outer, inner, grid[m], grid[m + 1])
        onset = max(onset, cross)
    return float(onset * 100.0)
SCICODE_GOLD_EOF
