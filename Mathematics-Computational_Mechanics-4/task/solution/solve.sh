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


def build_point_lattice(x_bounds: np.ndarray, spacings: np.ndarray,
                                height: float, horizon_ratio: float) -> np.ndarray:
    """Reference implementation."""
    xb = np.asarray(x_bounds, dtype=float)
    sp = np.asarray(spacings, dtype=float)
    if xb.ndim != 2 or xb.shape[1] != 2 or xb.shape[0] < 1:
        raise ValueError("x_bounds must have shape (R, 2) with R >= 1")
    if sp.ndim != 1 or sp.shape[0] != xb.shape[0]:
        raise ValueError("spacings must have shape (R,) matching x_bounds")
    if not np.all(np.isfinite(xb)) or not np.all(np.isfinite(sp)):
        raise ValueError("x_bounds and spacings must be finite")
    if np.any(xb[:, 1] <= xb[:, 0]):
        raise ValueError("each region must satisfy x_lo < x_hi")
    if xb.shape[0] > 1 and not np.allclose(xb[1:, 0], xb[:-1, 1], rtol=0.0, atol=1e-12):
        raise ValueError("regions must be contiguous in x")
    if np.any(sp <= 0.0):
        raise ValueError("spacings must be > 0")
    if not (np.isfinite(height) and height > 0.0):
        raise ValueError("height must be finite and > 0")
    if not (np.isfinite(horizon_ratio) and horizon_ratio > 0.0):
        raise ValueError("horizon_ratio must be finite and > 0")

    rows = []
    for r in range(xb.shape[0]):
        d = sp[r]
        lo = xb[r, 0]
        width = xb[r, 1] - xb[r, 0]
        nx = int(round(width / d))
        ny = int(round(height / d))
        if nx < 1 or abs(nx * d - width) > 1e-12 * width:
            raise ValueError("region width must be a positive integer multiple of its spacing")
        if ny < 1 or abs(ny * d - height) > 1e-12 * height:
            raise ValueError("height must be a positive integer multiple of every spacing")
        for ix in range(nx):
            for iy in range(ny):
                rows.append((lo + (ix + 0.5) * d,
                             (iy + 0.5) * d,
                             d,
                             horizon_ratio * d,
                             d * d))
    return np.asarray(rows, dtype=float)

import numpy as np


def build_directed_bonds(lattice: np.ndarray, crack_y: float,
                                 crack_x_max: float) -> np.ndarray:
    """Reference implementation."""
    lat = np.asarray(lattice, dtype=float)
    if lat.ndim != 2 or lat.shape[1] != 5 or lat.shape[0] < 2:
        raise ValueError("lattice must have shape (N, 5) with N >= 2")
    if not np.all(np.isfinite(lat)):
        raise ValueError("lattice must be finite")
    if np.any(lat[:, 3] <= 0.0):
        raise ValueError("horizons (column 3) must be > 0")
    if np.any(lat[:, 4] <= 0.0):
        raise ValueError("volumes (column 4) must be > 0")
    if not (np.isfinite(crack_y) and np.isfinite(crack_x_max)):
        raise ValueError("crack_y and crack_x_max must be finite")

    X = lat[:, 0]
    Y = lat[:, 1]
    horizon = lat[:, 3]
    N = lat.shape[0]

    owners = []
    nbrs = []
    for i in range(N):
        r = np.hypot(X - X[i], Y - Y[i])
        if np.count_nonzero(r == 0.0) > 1:
            raise ValueError("lattice contains coincident points")
        js = np.where((r > 0.0) & (r <= horizon[i] * (1.0 + 1e-12)))[0]
        for j in js:
            if (Y[i] - crack_y) * (Y[j] - crack_y) < 0.0:
                t = (crack_y - Y[i]) / (Y[j] - Y[i])
                if X[i] + t * (X[j] - X[i]) <= crack_x_max:
                    continue
            owners.append(i)
            nbrs.append(j)
    return np.asarray([owners, nbrs], dtype=np.int64).T.reshape(-1, 2)

import numpy as np


def calibrate_micromodulus(youngs_modulus: float, horizons: np.ndarray,
                                   influence_exponent: float) -> np.ndarray:
    """Reference implementation."""
    d = np.asarray(horizons, dtype=float)
    E = float(youngs_modulus)
    a = float(influence_exponent)
    if d.size < 1:
        raise ValueError("horizons must contain at least one entry")
    if not np.all(np.isfinite(d)):
        raise ValueError("horizons must be finite")
    if np.any(d <= 0.0):
        raise ValueError("horizons must be > 0")
    if not (np.isfinite(E) and E > 0.0):
        raise ValueError("youngs_modulus must be finite and > 0")
    if not np.isfinite(a):
        raise ValueError("influence_exponent must be finite")
    if a >= 3.0:
        raise ValueError("influence_exponent must be < 3 for the weighted moment to converge")

    moment = d ** (3.0 - a) / (3.0 - a)
    return 8.0 * E / (5.0 * np.pi * moment)

import numpy as np


def bond_critical_stretch_squared(lattice: np.ndarray, bonds: np.ndarray,
                                          micromodulus: np.ndarray, fracture_energy: float,
                                          influence_exponent: float) -> np.ndarray:
    """Reference implementation."""
    lat = np.asarray(lattice, dtype=float)
    bd = np.asarray(bonds)
    cm = np.asarray(micromodulus, dtype=float)
    G = float(fracture_energy)
    a = float(influence_exponent)
    if lat.ndim != 2 or lat.shape[1] != 5 or lat.shape[0] < 2:
        raise ValueError("lattice must have shape (N, 5) with N >= 2")
    if not np.all(np.isfinite(lat)):
        raise ValueError("lattice must be finite")
    if bd.ndim != 2 or bd.shape[1] != 2:
        raise ValueError("bonds must have shape (B, 2)")
    if bd.shape[0] > 0 and (bd.min() < 0 or bd.max() >= lat.shape[0]):
        raise ValueError("bond indices out of range for lattice")
    if cm.ndim != 1 or cm.shape[0] != lat.shape[0]:
        raise ValueError("micromodulus must have shape (N,) matching lattice")
    if not np.all(np.isfinite(cm)) or np.any(cm <= 0.0):
        raise ValueError("micromodulus must be finite and > 0")
    if not (np.isfinite(G) and G > 0.0):
        raise ValueError("fracture_energy must be finite and > 0")
    if not np.isfinite(a):
        raise ValueError("influence_exponent must be finite")
    if a >= 3.0:
        raise ValueError("influence_exponent must be < 3")
    if bd.shape[0] == 0:
        return np.zeros(0, dtype=float)

    own = bd[:, 0].astype(np.int64)
    nbr = bd[:, 1].astype(np.int64)
    R = np.hypot(lat[nbr, 0] - lat[own, 0], lat[nbr, 1] - lat[own, 1])
    if np.any(R <= 0.0):
        raise ValueError("bonds must have strictly positive reference length")

    horizon_n = lat[nbr, 3]
    c_n = cm[nbr]
    return 3.0 * G / (2.0 * c_n * horizon_n ** 3 * R ** (1.0 - a))

import numpy as np


def regional_critical_time_step(lattice: np.ndarray, bonds: np.ndarray,
                                        micromodulus: np.ndarray, region_ids: np.ndarray,
                                        density: float, influence_exponent: float) -> np.ndarray:
    """Reference implementation."""
    lat = np.asarray(lattice, dtype=float)
    bd = np.asarray(bonds)
    cm = np.asarray(micromodulus, dtype=float)
    rid = np.asarray(region_ids)
    rho = float(density)
    a = float(influence_exponent)
    if lat.ndim != 2 or lat.shape[1] != 5 or lat.shape[0] < 2:
        raise ValueError("lattice must have shape (N, 5) with N >= 2")
    if not np.all(np.isfinite(lat)):
        raise ValueError("lattice must be finite")
    if bd.ndim != 2 or bd.shape[1] != 2 or bd.shape[0] < 1:
        raise ValueError("bonds must have shape (B, 2) with B >= 1")
    if bd.min() < 0 or bd.max() >= lat.shape[0]:
        raise ValueError("bond indices out of range for lattice")
    if cm.ndim != 1 or cm.shape[0] != lat.shape[0]:
        raise ValueError("micromodulus must have shape (N,) matching lattice")
    if not np.all(np.isfinite(cm)) or np.any(cm <= 0.0):
        raise ValueError("micromodulus must be finite and > 0")
    if rid.ndim != 1 or rid.shape[0] != lat.shape[0]:
        raise ValueError("region_ids must have shape (N,) matching lattice")
    if not np.issubdtype(rid.dtype, np.integer):
        raise ValueError("region_ids must be an integer array")
    if not (np.isfinite(rho) and rho > 0.0):
        raise ValueError("density must be finite and > 0")
    if not np.isfinite(a) or a >= 3.0:
        raise ValueError("influence_exponent must be finite and < 3")

    own = bd[:, 0].astype(np.int64)
    nbr = bd[:, 1].astype(np.int64)
    R = np.hypot(lat[nbr, 0] - lat[own, 0], lat[nbr, 1] - lat[own, 1])
    if np.any(R <= 0.0):
        raise ValueError("bonds must have strictly positive reference length")

    coef = cm[nbr] * R ** (-a) / R
    stiffness = np.zeros(lat.shape[0], dtype=float)
    np.add.at(stiffness, own, coef * lat[nbr, 4])
    np.add.at(stiffness, nbr, coef * lat[own, 4])
    if np.any(stiffness <= 0.0):
        raise ValueError("every point must carry at least one bond contribution")

    point_step = np.sqrt(2.0 * rho / stiffness)
    return np.array([point_step[rid == r].min() for r in np.unique(rid)], dtype=float)

import numpy as np


def internal_force_density(lattice: np.ndarray, bonds: np.ndarray,
                                   micromodulus: np.ndarray, bond_states: np.ndarray,
                                   displacement: np.ndarray,
                                   influence_exponent: float) -> np.ndarray:
    """Reference implementation."""
    lat = np.asarray(lattice, dtype=float)
    bd = np.asarray(bonds)
    cm = np.asarray(micromodulus, dtype=float)
    mu = np.asarray(bond_states, dtype=float)
    u = np.asarray(displacement, dtype=float)
    a = float(influence_exponent)
    if lat.ndim != 2 or lat.shape[1] != 5 or lat.shape[0] < 2:
        raise ValueError("lattice must have shape (N, 5) with N >= 2")
    if not np.all(np.isfinite(lat)):
        raise ValueError("lattice must be finite")
    if bd.ndim != 2 or bd.shape[1] != 2:
        raise ValueError("bonds must have shape (B, 2)")
    if bd.shape[0] > 0 and (bd.min() < 0 or bd.max() >= lat.shape[0]):
        raise ValueError("bond indices out of range for lattice")
    if cm.ndim != 1 or cm.shape[0] != lat.shape[0]:
        raise ValueError("micromodulus must have shape (N,) matching lattice")
    if not np.all(np.isfinite(cm)) or np.any(cm <= 0.0):
        raise ValueError("micromodulus must be finite and > 0")
    if mu.ndim != 1 or mu.shape[0] != bd.shape[0]:
        raise ValueError("bond_states must have shape (B,) matching bonds")
    if not np.all((mu == 0.0) | (mu == 1.0)):
        raise ValueError("bond_states must contain only the values 0 and 1")
    if u.shape != (lat.shape[0], 2):
        raise ValueError("displacement must have shape (N, 2) matching lattice")
    if not np.all(np.isfinite(u)):
        raise ValueError("displacement must be finite")
    if not np.isfinite(a) or a >= 3.0:
        raise ValueError("influence_exponent must be finite and < 3")

    force = np.zeros((lat.shape[0], 2), dtype=float)
    if bd.shape[0] == 0:
        return force

    own = bd[:, 0].astype(np.int64)
    nbr = bd[:, 1].astype(np.int64)
    position = lat[:, :2] + u
    d = position[nbr] - position[own]
    R = np.hypot(lat[nbr, 0] - lat[own, 0], lat[nbr, 1] - lat[own, 1])
    if np.any(R <= 0.0):
        raise ValueError("bonds must have strictly positive reference length")
    r = np.hypot(d[:, 0], d[:, 1])
    if np.any(r <= 0.0):
        raise ValueError("bonds must have strictly positive deformed length")

    s = (r - R) / R
    f = (cm[nbr] * R ** (-a) * mu * s / r)[:, None] * d
    np.add.at(force, own, f * lat[nbr, 4][:, None])
    np.add.at(force, nbr, -f * lat[own, 4][:, None])
    return force

import numpy as np


def update_bond_states(lattice: np.ndarray, bonds: np.ndarray, bond_states: np.ndarray,
                               critical_stretch_squared: np.ndarray, displacement: np.ndarray,
                               no_fail_points: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    lat = np.asarray(lattice, dtype=float)
    bd = np.asarray(bonds)
    mu = np.asarray(bond_states, dtype=float)
    sc2 = np.asarray(critical_stretch_squared, dtype=float)
    u = np.asarray(displacement, dtype=float)
    nf = np.asarray(no_fail_points, dtype=float)
    if lat.ndim != 2 or lat.shape[1] != 5 or lat.shape[0] < 2:
        raise ValueError("lattice must have shape (N, 5) with N >= 2")
    if not np.all(np.isfinite(lat)):
        raise ValueError("lattice must be finite")
    if bd.ndim != 2 or bd.shape[1] != 2:
        raise ValueError("bonds must have shape (B, 2)")
    if bd.shape[0] > 0 and (bd.min() < 0 or bd.max() >= lat.shape[0]):
        raise ValueError("bond indices out of range for lattice")
    if mu.ndim != 1 or mu.shape[0] != bd.shape[0]:
        raise ValueError("bond_states must have shape (B,) matching bonds")
    if not np.all((mu == 0.0) | (mu == 1.0)):
        raise ValueError("bond_states must contain only the values 0 and 1")
    if sc2.ndim != 1 or sc2.shape[0] != bd.shape[0]:
        raise ValueError("critical_stretch_squared must have shape (B,) matching bonds")
    if not np.all(np.isfinite(sc2)) or np.any(sc2 <= 0.0):
        raise ValueError("critical_stretch_squared must be finite and > 0")
    if u.shape != (lat.shape[0], 2):
        raise ValueError("displacement must have shape (N, 2) matching lattice")
    if not np.all(np.isfinite(u)):
        raise ValueError("displacement must be finite")
    if nf.ndim != 1 or nf.shape[0] != lat.shape[0]:
        raise ValueError("no_fail_points must have shape (N,) matching lattice")
    if not np.all((nf == 0.0) | (nf == 1.0)):
        raise ValueError("no_fail_points must contain only the values 0 and 1")
    if bd.shape[0] == 0:
        return mu.copy()

    own = bd[:, 0].astype(np.int64)
    nbr = bd[:, 1].astype(np.int64)
    R = np.hypot(lat[nbr, 0] - lat[own, 0], lat[nbr, 1] - lat[own, 1])
    if np.any(R <= 0.0):
        raise ValueError("bonds must have strictly positive reference length")
    position = lat[:, :2] + u
    d = position[nbr] - position[own]
    r = np.hypot(d[:, 0], d[:, 1])
    if np.any(r <= 0.0):
        raise ValueError("bonds must have strictly positive deformed length")

    s = (r - R) / R
    allowed = (nf[own] == 0.0) & (nf[nbr] == 0.0)
    breaks = (mu == 1.0) & allowed & (s * s >= sc2)
    updated_states = mu.copy()
    updated_states[breaks] = 0.0
    return updated_states

import numpy as np


def integrate_avv(lattice: np.ndarray, bonds: np.ndarray, micromodulus: np.ndarray,
                          critical_stretch_squared: np.ndarray, no_fail_points: np.ndarray,
                          body_force: np.ndarray, local_steps: np.ndarray, final_time: float,
                          density: float, influence_exponent: float,
                          initial_bond_states: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    lat = np.asarray(lattice, dtype=float)
    bd = np.asarray(bonds)
    cm = np.asarray(micromodulus, dtype=float)
    sc2 = np.asarray(critical_stretch_squared, dtype=float)
    nf = np.asarray(no_fail_points, dtype=float)
    b = np.asarray(body_force, dtype=float)
    hs = np.asarray(local_steps, dtype=float)
    T = float(final_time)
    rho = float(density)
    a = float(influence_exponent)
    mu0 = np.asarray(initial_bond_states, dtype=float)
    if lat.ndim != 2 or lat.shape[1] != 5 or lat.shape[0] < 2:
        raise ValueError("lattice must have shape (N, 5) with N >= 2")
    if not np.all(np.isfinite(lat)):
        raise ValueError("lattice must be finite")
    N = lat.shape[0]
    if bd.ndim != 2 or bd.shape[1] != 2 or bd.shape[0] < 1:
        raise ValueError("bonds must have shape (B, 2) with B >= 1")
    if bd.min() < 0 or bd.max() >= N:
        raise ValueError("bond indices out of range for lattice")
    if cm.ndim != 1 or cm.shape[0] != N or not np.all(np.isfinite(cm)) or np.any(cm <= 0.0):
        raise ValueError("micromodulus must have shape (N,) and be finite and > 0")
    if (sc2.ndim != 1 or sc2.shape[0] != bd.shape[0] or not np.all(np.isfinite(sc2))
            or np.any(sc2 <= 0.0)):
        raise ValueError("critical_stretch_squared must have shape (B,) and be finite and > 0")
    if nf.ndim != 1 or nf.shape[0] != N or not np.all((nf == 0.0) | (nf == 1.0)):
        raise ValueError("no_fail_points must have shape (N,) containing only 0 and 1")
    if b.shape != (N, 2) or not np.all(np.isfinite(b)):
        raise ValueError("body_force must have shape (N, 2) and be finite")
    if hs.ndim != 1 or hs.shape[0] != N or not np.all(np.isfinite(hs)) or np.any(hs <= 0.0):
        raise ValueError("local_steps must have shape (N,) and be finite and > 0")
    if not (np.isfinite(T) and T > 0.0):
        raise ValueError("final_time must be finite and > 0")
    if not (np.isfinite(rho) and rho > 0.0):
        raise ValueError("density must be finite and > 0")
    if not np.isfinite(a) or a >= 3.0:
        raise ValueError("influence_exponent must be finite and < 3")
    if mu0.ndim != 1 or mu0.shape[0] != bd.shape[0] or not np.all((mu0 == 0.0) | (mu0 == 1.0)):
        raise ValueError("initial_bond_states must have shape (B,) containing only 0 and 1")

    h_base = hs.min()
    ratio = hs / h_base
    n_i = np.rint(ratio).astype(np.int64)
    if np.any(np.abs(ratio - n_i) > 1e-12 * ratio) or np.any(n_i < 1):
        raise ValueError("every local step must be a positive integer multiple of the smallest")
    M = int(round(T / h_base))
    if M < 1 or abs(M * h_base - T) > 1e-12 * T:
        raise ValueError("final_time must be a positive integer multiple of the smallest local step")
    if np.any(np.abs(np.rint(T / hs) * hs - T) > 1e-12 * T):
        raise ValueError("final_time must be a positive integer multiple of every local step")
    M_i = np.rint(T / hs).astype(np.int64)

    own = bd[:, 0].astype(np.int64)
    nbr = bd[:, 1].astype(np.int64)
    R = np.hypot(lat[nbr, 0] - lat[own, 0], lat[nbr, 1] - lat[own, 1])
    if np.any(R <= 0.0):
        raise ValueError("bonds must have strictly positive reference length")
    coef = cm[nbr] * R ** (-a)
    vol_n = lat[nbr, 4]
    vol_o = lat[own, 4]
    allowed = (nf[own] == 0.0) & (nf[nbr] == 0.0)
    step_b = hs[own]
    last_b = M_i[own]

    u = np.zeros((N, 2), dtype=float)
    v = np.zeros((N, 2), dtype=float)
    mu = mu0 == 1.0
    f_minus = np.zeros((N, 2), dtype=float)
    f_plus = np.zeros((N, 2), dtype=float)

    def _deposit(sel, w_minus, w_plus):
        position = lat[:, :2] + u
        d = position[nbr[sel]] - position[own[sel]]
        r = np.hypot(d[:, 0], d[:, 1])
        if np.any(r <= 0.0):
            raise ValueError("bonds must have strictly positive deformed length")
        s = (r - R[sel]) / R[sel]
        brk = mu[sel] & allowed[sel] & (s * s >= sc2[sel])
        if brk.any():
            m = mu[sel]
            m[brk] = False
            mu[sel] = m
        f = (coef[sel] * mu[sel] * s / r)[:, None] * d
        g_own = f * vol_n[sel][:, None]
        g_rea = f * vol_o[sel][:, None]
        if w_minus is not None:
            np.add.at(f_minus, own[sel], w_minus[:, None] * g_own)
            np.add.at(f_minus, nbr[sel], -w_minus[:, None] * g_rea)
        if w_plus is not None:
            np.add.at(f_plus, own[sel], w_plus[:, None] * g_own)
            np.add.at(f_plus, nbr[sel], -w_plus[:, None] * g_rea)

    all_bonds = np.arange(bd.shape[0])
    _deposit(all_bonds, None, step_b)
    for k in range(M):
        v += f_plus / (2.0 * rho) + (h_base / (2.0 * rho)) * b
        u += h_base * v
        f_minus[:] = 0.0
        f_plus[:] = 0.0
        sel = np.nonzero((k + 1) % n_i[own] == 0)[0]
        if sel.size:
            tau = (k + 1) // n_i[own[sel]]
            _deposit(sel, step_b[sel], np.where(tau < last_b[sel], step_b[sel], 0.0))
        v += f_minus / (2.0 * rho) + (h_base / (2.0 * rho)) * b
    return u

import numpy as np


def solve_plate_uy(youngs_modulus: float, density: float, fracture_energy: float,
                           influence_exponent: float, x_bounds: np.ndarray, spacings: np.ndarray,
                           height: float, horizon_ratio: float, crack_y: float, crack_x_max: float,
                           no_fail_width: float, applied_stress: float,
                           region_steps: np.ndarray, final_time: float,
                           probe_point: np.ndarray) -> float:
    """Reference implementation."""
    xb = np.asarray(x_bounds, dtype=float)
    sp = np.asarray(spacings, dtype=float)
    rs = np.asarray(region_steps, dtype=float)
    pp = np.asarray(probe_point, dtype=float)
    if (rs.ndim != 1 or rs.shape[0] != sp.shape[0] or not np.all(np.isfinite(rs))
            or np.any(rs <= 0.0)):
        raise ValueError("region_steps must have shape (R,) matching spacings and be finite and > 0")
    if pp.shape != (2,) or not np.all(np.isfinite(pp)):
        raise ValueError("probe_point must have shape (2,) and be finite")
    if not (np.isfinite(no_fail_width) and no_fail_width >= 0.0):
        raise ValueError("no_fail_width must be finite and >= 0")
    if not np.isfinite(applied_stress):
        raise ValueError("applied_stress must be finite")

    lattice = build_point_lattice(xb, sp, height, horizon_ratio)
    d2 = (lattice[:, 0] - pp[0]) ** 2 + (lattice[:, 1] - pp[1]) ** 2
    probe = int(np.argmin(d2))
    if d2[probe] > (1e-9 * max(height, 1.0)) ** 2:
        raise ValueError("probe_point does not coincide with a material point")

    bonds = build_directed_bonds(lattice, crack_y, crack_x_max)
    micromodulus = calibrate_micromodulus(youngs_modulus, lattice[:, 3],
                                                  influence_exponent)
    sc2 = bond_critical_stretch_squared(lattice, bonds, micromodulus,
                                                fracture_energy, influence_exponent)
    N = lattice.shape[0]
    region_ids = np.zeros(N, dtype=np.int64)
    for r in range(1, xb.shape[0]):
        region_ids[lattice[:, 0] > xb[r, 0]] = r
    hc = regional_critical_time_step(lattice, bonds, micromodulus, region_ids,
                                             density, influence_exponent)
    if np.any(rs >= hc):
        raise ValueError("each region step must be below that region's critical time step")

    zero_u = np.zeros((N, 2), dtype=float)
    intact = np.ones(bonds.shape[0], dtype=float)
    f0 = internal_force_density(lattice, bonds, micromodulus, intact,
                                        zero_u, influence_exponent)
    if np.any(f0 != 0.0):
        raise ValueError("the reference configuration must carry zero internal force")

    no_fail = ((lattice[:, 1] < no_fail_width) |
               (lattice[:, 1] > height - no_fail_width)).astype(float)
    mu0 = update_bond_states(lattice, bonds, intact, sc2, zero_u, no_fail)
    if np.any(mu0 != 1.0):
        raise ValueError("no bond may fail in the reference configuration")

    body_force = np.zeros((N, 2), dtype=float)
    dx = lattice[:, 2]
    top = lattice[:, 1] > height - dx
    bot = lattice[:, 1] < dx
    body_force[top, 1] = applied_stress / dx[top]
    body_force[bot, 1] = -applied_stress / dx[bot]

    local_steps = rs[region_ids]
    u = integrate_avv(lattice, bonds, micromodulus, sc2, no_fail, body_force,
                              local_steps, final_time, density, influence_exponent, mu0)
    return float(u[probe, 1] * 1e6)
SCICODE_GOLD_EOF
