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
from scipy.optimize import brentq

def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _check_finite(a, what):
    if not np.all(np.isfinite(a)):
        raise ValueError("non-finite " + what)

def fmpm_grid_operators(Xp: "np.ndarray", Mp: "np.ndarray", n_nodes: int, dx: float) -> "np.ndarray":
    Xp = _f64(Xp).ravel()
    Mp = _f64(Mp).ravel()
    if Xp.size < 1 or Mp.size != Xp.size:
        raise ValueError("bad particle arrays")
    if not isinstance(n_nodes, (int, np.integer)) or isinstance(n_nodes, bool) or n_nodes < 2:
        raise ValueError("bad n_nodes")
    dxf = float(dx)
    if not np.isfinite(dxf) or dxf <= 0.0:
        raise ValueError("bad dx")
    _check_finite(Xp, "positions")
    _check_finite(Mp, "masses")
    if np.any(Mp <= 0.0):
        raise ValueError("non-positive mass")
    n = int(n_nodes)
    N = Xp.size
    xi = np.arange(n, dtype=np.float64) * dxf
    if np.any(Xp < 0.0) or np.any(Xp > xi[-1]):
        raise ValueError("particle outside grid")
    # linear (tent) shape functions
    S = 1.0 - np.abs(Xp[:, None] - xi[None, :]) / dxf
    S = np.where(S > 0.0, S, 0.0)
    m = (S * Mp[:, None]).sum(axis=0)          # m = diag(S^T M)
    active = m > 0.0
    Sp = np.zeros_like(S)
    Sp[:, active] = (Mp[:, None] * S[:, active]) / m[None, active]
    out = np.zeros((2 * N + 1, n), dtype=np.float64)
    out[0:N, :] = S
    out[N:2 * N, :] = Sp
    out[2 * N, :] = m
    return out

import numpy as np
from scipy.optimize import brentq

def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _check_finite(a, what):
    if not np.all(np.isfinite(a)):
        raise ValueError("non-finite " + what)

def _split(ops, N):
    ops = _f64(ops)
    if not isinstance(N, (int, np.integer)) or isinstance(N, bool) or N < 1:
        raise ValueError("bad N")
    if ops.ndim != 2 or ops.shape[0] != 2 * int(N) + 1:
        raise ValueError("bad ops shape")
    _check_finite(ops, "ops")
    return ops[0:N, :], ops[N:2 * N, :], ops[2 * N, :]

def _SpS(S, Sp, v):
    # (S+ S v)_i = sum_p Sp[p,i] * (S v)_p
    return np.dot(Sp.T, np.dot(S, v))

def _bc_array(bc_nodes, n):
    bc = np.asarray(bc_nodes, dtype=np.int64).ravel() if bc_nodes is not None else np.zeros(0, np.int64)
    if bc.size and (bc.min() < 0 or bc.max() >= n):
        raise ValueError("bad bc node index")
    return bc

def fmpm_velocity_increment(ops: "np.ndarray", N: int, dv_prev: "np.ndarray", bc_nodes: list, alpha_blend: float) -> "np.ndarray":
    S, Sp, m = _split(ops, int(N))
    n = m.size
    dv = _f64(dv_prev).ravel()
    if dv.size != n:
        raise ValueError("bad dv_prev length")
    _check_finite(dv, "dv_prev")
    a = float(alpha_blend)
    if not np.isfinite(a) or a <= 0.0 or a > 1.0:
        raise ValueError("bad alpha_blend")
    bc = _bc_array(bc_nodes, n)
    dv_new = a * (dv - _SpS(S, Sp, dv))
    dv_new[m <= 0.0] = 0.0                     # inactive nodes carry nothing
    if bc.size:
        dv_new[bc] = 0.0                       # constraint imposed on the increment
    return dv_new.astype(np.float64)

import numpy as np
from scipy.optimize import brentq

def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _check_finite(a, what):
    if not np.all(np.isfinite(a)):
        raise ValueError("non-finite " + what)

def _split(ops, N):
    ops = _f64(ops)
    if not isinstance(N, (int, np.integer)) or isinstance(N, bool) or N < 1:
        raise ValueError("bad N")
    if ops.ndim != 2 or ops.shape[0] != 2 * int(N) + 1:
        raise ValueError("bad ops shape")
    _check_finite(ops, "ops")
    return ops[0:N, :], ops[N:2 * N, :], ops[2 * N, :]

def _bc_array(bc_nodes, n):
    bc = np.asarray(bc_nodes, dtype=np.int64).ravel() if bc_nodes is not None else np.zeros(0, np.int64)
    if bc.size and (bc.min() < 0 or bc.max() >= n):
        raise ValueError("bad bc node index")
    return bc

def _check_order(k):
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
        raise ValueError("bad k")
    return int(k)

def _check_blend(alpha_blend, blend_period):
    a = float(alpha_blend)
    if not np.isfinite(a) or a <= 0.0 or a > 1.0:
        raise ValueError("bad alpha_blend")
    if not isinstance(blend_period, (int, np.integer)) or isinstance(blend_period, bool) or blend_period < 1:
        raise ValueError("bad blend_period")
    return a, int(blend_period)

def _blend_scale(ell, alpha, period):
    """the source's generalised blend: the increment of order ell is scaled by alpha when period == 1 or ell mod period == 1"""
    return alpha if (period == 1 or ell % period == 1) else 1.0

def fmpm_loop(ops: "np.ndarray", N: int, p_plus: "np.ndarray", k: int, bc_nodes: list, alpha_blend: float, blend_period: int, tol: float) -> "np.ndarray":
    S, Sp, m = _split(ops, int(N))
    n = m.size
    p = _f64(p_plus).ravel()
    if p.size != n:
        raise ValueError("bad p_plus length")
    _check_finite(p, "p_plus")
    k = _check_order(k)
    t = float(tol)
    if not np.isfinite(t) or t < 0.0:
        raise ValueError("bad tol")
    a, period = _check_blend(alpha_blend, blend_period)
    bc = _bc_array(bc_nodes, n)
    active = m > 0.0
    v_prev = np.zeros(n, dtype=np.float64)
    v_prev[active] = p[active] / m[active]     # Delta v(1) = m^-1 p+ on the active nodes
    if bc.size:
        v_prev[bc] = 0.0                       # a controlled node carries its prescribed zero velocity, seed included
    v_star = v_prev.copy()
    order = 1
    converged = 0.0
    for ell in range(2, k + 1):
        v_prev = fmpm_velocity_increment(ops, N, v_prev, bc_nodes, _blend_scale(ell, a, period))
        v_star = v_star + v_prev
        order += 1
        if float(np.linalg.norm(v_prev)) < t:
            converged = 1.0
            break
    out = np.zeros((3, n), dtype=np.float64)
    out[0, :] = v_star
    out[1, :] = v_prev
    out[2, 0] = float(order)
    out[2, 1] = converged
    return out

import numpy as np
from scipy.optimize import brentq

def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _check_finite(a, what):
    if not np.all(np.isfinite(a)):
        raise ValueError("non-finite " + what)

def _split(ops, N):
    ops = _f64(ops)
    if not isinstance(N, (int, np.integer)) or isinstance(N, bool) or N < 1:
        raise ValueError("bad N")
    if ops.ndim != 2 or ops.shape[0] != 2 * int(N) + 1:
        raise ValueError("bad ops shape")
    _check_finite(ops, "ops")
    return ops[0:N, :], ops[N:2 * N, :], ops[2 * N, :]

def _SpS(S, Sp, v):
    # (S+ S v)_i = sum_p Sp[p,i] * (S v)_p
    return np.dot(Sp.T, np.dot(S, v))

def _check_order(k):
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
        raise ValueError("bad k")
    return int(k)

def fmpm_prior_series(ops: "np.ndarray", N: int, p_plus: "np.ndarray", k: int) -> "np.ndarray":
    S, Sp, m = _split(ops, int(N))
    n = m.size
    p = _f64(p_plus).ravel()
    if p.size != n:
        raise ValueError("bad p_plus length")
    _check_finite(p, "p_plus")
    k = _check_order(k)
    active = m > 0.0
    base = np.zeros(n, dtype=np.float64)
    base[active] = p[active] / m[active]
    v_star_l = float(k) * base                 # v*_1 = k m^-1 p+
    total = v_star_l.copy()                    # l = 1 term, sign (+1)
    for ell in range(2, k + 1):
        v_star_l = (float(k + 1 - ell) / float(ell)) * _SpS(S, Sp, v_star_l)
        total = total + ((-1.0) ** (ell + 1)) * v_star_l
    out = np.zeros((2, n), dtype=np.float64)
    out[0, :] = total
    out[1, :] = v_star_l
    return out

import numpy as np
from scipy.optimize import brentq

def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _check_finite(a, what):
    if not np.all(np.isfinite(a)):
        raise ValueError("non-finite " + what)

def _split(ops, N):
    ops = _f64(ops)
    if not isinstance(N, (int, np.integer)) or isinstance(N, bool) or N < 1:
        raise ValueError("bad N")
    if ops.ndim != 2 or ops.shape[0] != 2 * int(N) + 1:
        raise ValueError("bad ops shape")
    _check_finite(ops, "ops")
    return ops[0:N, :], ops[N:2 * N, :], ops[2 * N, :]

def fmpm_particle_update(ops: "np.ndarray", N: int, v_plus: "np.ndarray", Vn: "np.ndarray", Xn: "np.ndarray", f_grid: "np.ndarray", dt: float, alpha: float) -> "np.ndarray":
    S, Sp, m = _split(ops, int(N))
    n = m.size
    Np = int(N)
    v = _f64(v_plus).ravel()
    Vn = _f64(Vn).ravel()
    Xn = _f64(Xn).ravel()
    f = _f64(f_grid).ravel()
    if v.size != n or f.size != n or Vn.size != Np or Xn.size != Np:
        raise ValueError("bad array sizes")
    for arr, nm in ((v, "v_plus"), (Vn, "Vn"), (Xn, "Xn"), (f, "f_grid")):
        _check_finite(arr, nm)
    d = float(dt)
    a = float(alpha)
    if not np.isfinite(d) or d <= 0.0:
        raise ValueError("bad dt")
    if not np.isfinite(a):
        raise ValueError("bad alpha")
    active = m > 0.0
    acc_grid = np.zeros(n, dtype=np.float64)
    acc_grid[active] = f[active] / m[active]   # a = m^-1 f  (lumped)
    V_new = np.dot(S, v)                       # V(n+1) = S v+(k)
    X_new = Xn + (a * V_new + (1.0 - a) * Vn) * d
    A_eff = (V_new - Vn) / d                   # Eq (9)
    mismatch = (a - 0.5) * (V_new - Vn)        # dX/dt - <V> for FMPM(k)
    V_flip = Vn + np.dot(S, acc_grid) * d      # Eq (8) FLIP velocity update
    out = np.zeros((5, Np), dtype=np.float64)
    out[0, :] = V_new
    out[1, :] = X_new
    out[2, :] = A_eff
    out[3, :] = mismatch
    out[4, :] = V_flip
    return out

import numpy as np
from scipy.optimize import brentq

def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _check_finite(a, what):
    if not np.all(np.isfinite(a)):
        raise ValueError("non-finite " + what)

def _split(ops, N):
    ops = _f64(ops)
    if not isinstance(N, (int, np.integer)) or isinstance(N, bool) or N < 1:
        raise ValueError("bad N")
    if ops.ndim != 2 or ops.shape[0] != 2 * int(N) + 1:
        raise ValueError("bad ops shape")
    _check_finite(ops, "ops")
    return ops[0:N, :], ops[N:2 * N, :], ops[2 * N, :]

def _bc_array(bc_nodes, n):
    bc = np.asarray(bc_nodes, dtype=np.int64).ravel() if bc_nodes is not None else np.zeros(0, np.int64)
    if bc.size and (bc.min() < 0 or bc.max() >= n):
        raise ValueError("bad bc node index")
    return bc

def _check_order(k):
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
        raise ValueError("bad k")
    return int(k)

def _check_blend(alpha_blend, blend_period):
    a = float(alpha_blend)
    if not np.isfinite(a) or a <= 0.0 or a > 1.0:
        raise ValueError("bad alpha_blend")
    if not isinstance(blend_period, (int, np.integer)) or isinstance(blend_period, bool) or blend_period < 1:
        raise ValueError("bad blend_period")
    return a, int(blend_period)

def _loop_matrices(ops, N, bc_nodes):
    """D = seed map m^-1 on the active nodes with zero rows at the controlled nodes, T = Z A with A = I - S+ S and Z zeroing inactive and controlled nodes"""
    S, Sp, m = _split(ops, int(N))
    n = m.size
    bc = _bc_array(bc_nodes, n)
    active = m > 0.0
    D = np.zeros((n, n))
    D[active, active] = 1.0 / m[active]
    if bc.size:
        D[bc, :] = 0.0
    A = np.eye(n) - np.dot(Sp.T, S)
    z = active.astype(np.float64)
    if bc.size:
        z[bc] = 0.0
    T = z[:, None] * A
    return D, T, n

def fmpm_inverse_operator(ops: "np.ndarray", N: int, k: int, bc_nodes: list, alpha_blend: float, blend_period: int) -> "np.ndarray":
    # K(k) and K_inf depend only on the arguments below, never on the Courant number, so a stability
    # scan that walks C over hundreds of points rebuilds an identical operator each time. Memoise it on
    # the exact inputs; the result is bit-identical and history cannot affect it.
    arr = np.ascontiguousarray(np.asarray(ops, dtype=np.float64))
    key = (arr.tobytes(), arr.shape, int(N), int(k),
           (None if bc_nodes is None else tuple(int(b) for b in bc_nodes)),
           float(alpha_blend), int(blend_period))
    cache = getattr(fmpm_inverse_operator, "_cache", None)
    if cache is None:
        cache = {}
        fmpm_inverse_operator._cache = cache
    hit = cache.get(key)
    if hit is not None:
        return hit.copy()
    out = _fmpm_inverse_operator_uncached(ops, N, k, bc_nodes, alpha_blend, blend_period)
    if len(cache) >= 64:
        cache.pop(next(iter(cache)))
    cache[key] = np.array(out, copy=True)
    return out

def _fmpm_inverse_operator_uncached(ops: "np.ndarray", N: int, k: int, bc_nodes: list, alpha_blend: float, blend_period: int) -> "np.ndarray":
    k = _check_order(k)
    a, period = _check_blend(alpha_blend, blend_period)
    D, T, n = _loop_matrices(ops, N, bc_nodes)
    # K(k): the revised loop of the source driven by one unit momentum per column (no early exit)
    eye = np.eye(n)
    K = np.column_stack([fmpm_loop(ops, N, eye[:, j], k, bc_nodes, a, period, 0.0)[0, :] for j in range(n)])
    # the exact limit of the same recursion: sum_{j>=0} alpha^j T^(j m) (I + T + ... + T^(m-1)) D = (I - alpha T^m)^-1 (I + ... + T^(m-1)) D
    partial = np.eye(n)
    powT = np.eye(n)
    for _ in range(1, period):
        powT = np.dot(powT, T)
        partial = partial + powT
    Tm = np.dot(powT, T)                                   # T^m
    rhoT = float(np.max(np.abs(np.linalg.eigvals(T))))
    rho_block = float(np.max(np.abs(np.linalg.eigvals(a * Tm))))
    if rho_block >= 1.0:
        raise ValueError("the FMPM series does not converge: the blended recursion has spectral radius >= 1")
    K_inf = np.linalg.solve(np.eye(n) - a * Tm, np.dot(partial, D))
    diff = K - K_inf
    out = np.zeros((2 * n + 1, n), dtype=np.float64)
    out[0:n, :] = K
    out[n:2 * n, :] = K_inf
    out[2 * n, 0] = rhoT
    out[2 * n, 1] = rho_block ** (1.0 / period)
    out[2 * n, 2] = float(np.linalg.norm(diff, 2))
    out[2 * n, 3] = float(np.max(np.abs(diff)))
    out[2 * n, 4] = float(np.count_nonzero(_split(ops, int(N))[2] > 0.0))
    return out

import numpy as np
from scipy.optimize import brentq

def _check_order(k):
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
        raise ValueError("bad k")
    return int(k)

def _check_blend(alpha_blend, blend_period):
    a = float(alpha_blend)
    if not np.isfinite(a) or a <= 0.0 or a > 1.0:
        raise ValueError("bad alpha_blend")
    if not isinstance(blend_period, (int, np.integer)) or isinstance(blend_period, bool) or blend_period < 1:
        raise ValueError("bad blend_period")
    return a, int(blend_period)

def _bar_particles(L_r, dx, s, rho):
    L = float(L_r); d = float(dx); sf = float(s); r = float(rho)
    if not (np.isfinite(L) and np.isfinite(d) and np.isfinite(sf) and np.isfinite(r)) or L <= 0.0 or d <= 0.0 or r <= 0.0:
        raise ValueError("bad bar geometry or density")
    ncell = int(round(L / d))
    if ncell < 1 or abs(ncell * d - L) > 1e-9 * max(1.0, L):
        raise ValueError("the bar length must be a whole number of cells")
    if not (0.0 < sf < 0.5):
        raise ValueError("the placement fraction must lie strictly between 0 and 0.5")
    n = ncell + 1
    Xp = np.array([(i + t) * d for i in range(ncell) for t in (sf, 1.0 - sf)])
    Vol = np.full(Xp.size, 0.5 * d)
    Mp = r * Vol
    return Xp, Mp, Vol, n, d

def _bar_gradients(Xp, n, dx):
    """d S_pi / dx for the tent functions (particles never sit on a node for 0 < s < 0.5)"""
    xi = np.arange(n, dtype=np.float64) * dx
    G = np.zeros((Xp.size, n))
    for p in range(Xp.size):
        i0 = int(np.floor(Xp[p] / dx))
        if 0 <= i0 < n:
            G[p, i0] = -1.0 / dx
        if 0 <= i0 + 1 < n:
            G[p, i0 + 1] = 1.0 / dx
    return G

def _check_scheme(scheme):
    if scheme not in ("FLIP", "FMPM", "EXACT"):
        raise ValueError("scheme must be FLIP, FMPM or EXACT")
    return scheme

def fmpm_bar_operator(L_r: float, dx: float, s: float, E: float, rho: float, C: float, scheme: str, k: int, alpha_blend: float, blend_period: int) -> "np.ndarray":
    _check_scheme(scheme)
    k = _check_order(k)
    a, period = _check_blend(alpha_blend, blend_period)
    Ef = float(E); Cf = float(C)
    if not np.isfinite(Ef) or Ef <= 0.0 or not np.isfinite(Cf) or Cf <= 0.0:
        raise ValueError("bad modulus or Courant number")
    Xp, Mp, Vol, n, d = _bar_particles(L_r, dx, s, rho)
    N = Xp.size
    ops = fmpm_grid_operators(Xp, Mp, n, d)
    S = ops[0:N, :]
    m = ops[2 * N, :]
    G = _bar_gradients(Xp, n, d)
    vwave = np.sqrt(Ef / float(rho))
    dt = Cf * d / vwave
    bc = [0]
    # linear maps of the state z = [V; eps]: momenta p = S^T M V, forces f = -G^T Omega E eps, reaction p+_0 = 0
    P_V = np.dot(S.T, np.diag(Mp))
    F_eps = -np.dot(G.T, np.diag(Vol * Ef))
    Pp_V = P_V.copy(); Pp_eps = F_eps * dt
    Pp_V[0, :] = 0.0; Pp_eps[0, :] = 0.0
    active = m > 0.0
    Dinv = np.zeros((n, n)); Dinv[active, active] = 1.0 / m[active]
    if scheme == "FLIP":
        vV = np.dot(Dinv, Pp_V); veps = np.dot(Dinv, Pp_eps)                       # lumped v+(1)
    else:
        inv = fmpm_inverse_operator(ops, N, k, bc, a, period)
        K = inv[0:n, :] if scheme == "FMPM" else inv[n:2 * n, :]
        vV = np.dot(K, Pp_V); veps = np.dot(K, Pp_eps)
    # particle velocities through the particle update of the source, one column per unit state entry:
    # FMPM and EXACT take the PIC-style row V(n+1) = S v+, FLIP the row V(n) + S m^-1 f dt with f dt = p+ - p
    upd_row = 4 if scheme == "FLIP" else 0
    VV = np.zeros((N, N)); Veps = np.zeros((N, N))
    eyeN = np.eye(N)
    for j in range(N):
        VV[:, j] = fmpm_particle_update(ops, N, vV[:, j], eyeN[:, j], Xp, (Pp_V[:, j] - P_V[:, j]) / dt, dt, 1.0)[upd_row, :]
        Veps[:, j] = fmpm_particle_update(ops, N, veps[:, j], np.zeros(N), Xp, Pp_eps[:, j] / dt, dt, 1.0)[upd_row, :]
    EV = dt * np.dot(G, vV)
    Eeps = np.eye(N) + dt * np.dot(G, veps)                                           # USL strain update with the same grid velocities
    out = np.zeros((2 * N, 2 * N), dtype=np.float64)
    out[0:N, 0:N] = VV; out[0:N, N:] = Veps
    out[N:, 0:N] = EV; out[N:, N:] = Eeps
    return out

import numpy as np
from scipy.optimize import brentq

def _check_scheme(scheme):
    if scheme not in ("FLIP", "FMPM", "EXACT"):
        raise ValueError("scheme must be FLIP, FMPM or EXACT")
    return scheme

def _spectral_radius(L_r, dx, s, E, rho, C, scheme, k, alpha_blend, blend_period):
    return float(np.max(np.abs(np.linalg.eigvals(fmpm_bar_operator(L_r, dx, s, E, rho, C, scheme, k, alpha_blend, blend_period)))))

def fmpm_stability_limit(L_r: float, dx: float, s: float, E: float, rho: float, scheme: str, k: int, alpha_blend: float, blend_period: int, tol: float, C_max: float, dC: float) -> float:
    t = float(tol); cm = float(C_max); dc = float(dC)
    if not np.isfinite(t) or t < 0.0 or not np.isfinite(cm) or not np.isfinite(dc) or dc <= 0.0 or cm <= dc:
        raise ValueError("bad scan settings")
    _check_scheme(scheme)
    rho_of = lambda C: _spectral_radius(L_r, dx, s, E, rho, C, scheme, k, alpha_blend, blend_period) - 1.0 - t
    C_a = 0.0
    C_b = None
    j = 1
    while j * dc <= cm + 1e-12:
        Cj = j * dc
        if rho_of(Cj) > 0.0:
            C_b = Cj
            break
        C_a = Cj
        j += 1
    if C_b is None:
        raise ValueError("no loss of stability below C_max")
    if C_a == 0.0:
        C_a = 1e-6 * dc
        if rho_of(C_a) > 0.0:
            raise ValueError("unstable at every scanned Courant number")
    return float(brentq(rho_of, C_a, C_b, xtol=1e-10, rtol=1e-12, maxiter=200))

import numpy as np
from scipy.optimize import brentq

def _bar_particles(L_r, dx, s, rho):
    L = float(L_r); d = float(dx); sf = float(s); r = float(rho)
    if not (np.isfinite(L) and np.isfinite(d) and np.isfinite(sf) and np.isfinite(r)) or L <= 0.0 or d <= 0.0 or r <= 0.0:
        raise ValueError("bad bar geometry or density")
    ncell = int(round(L / d))
    if ncell < 1 or abs(ncell * d - L) > 1e-9 * max(1.0, L):
        raise ValueError("the bar length must be a whole number of cells")
    if not (0.0 < sf < 0.5):
        raise ValueError("the placement fraction must lie strictly between 0 and 0.5")
    n = ncell + 1
    Xp = np.array([(i + t) * d for i in range(ncell) for t in (sf, 1.0 - sf)])
    Vol = np.full(Xp.size, 0.5 * d)
    Mp = r * Vol
    return Xp, Mp, Vol, n, d

def fmpm_energy_retention(L_r: float, dx: float, s: float, E: float, rho: float, scheme: str, k: int, alpha_blend: float, blend_period: int, C: float, v0: float, periods: float) -> "np.ndarray":
    v0f = float(v0); per = float(periods)
    if not np.isfinite(v0f) or v0f == 0.0 or not np.isfinite(per) or per <= 0.0:
        raise ValueError("bad amplitude or period count")
    Gm = fmpm_bar_operator(L_r, dx, s, E, rho, C, scheme, k, alpha_blend, blend_period)
    if float(np.max(np.abs(np.linalg.eigvals(Gm)))) > 1.0 + 1e-8:
        raise ValueError("the configuration is linearly unstable at this Courant number; its energy has no finite retention")
    Xp, Mp, Vol, n, d = _bar_particles(L_r, dx, s, rho)
    N = Xp.size
    Ef = float(E)
    vwave = np.sqrt(Ef / float(rho))
    dt = float(C) * d / vwave
    T = 4.0 * float(L_r) / vwave                       # fundamental period of the fixed-free bar
    nsteps = int(round(per * T / dt))
    z = np.concatenate([v0f * np.sin(np.pi * Xp / (2.0 * float(L_r))), np.zeros(N)])
    E0 = 0.5 * float(np.sum(Mp * z[:N] ** 2))
    for _ in range(nsteps):
        z = np.dot(Gm, z)
    ke = 0.5 * float(np.sum(Mp * z[:N] ** 2))
    se = 0.5 * float(np.sum(Vol * Ef * z[N:] ** 2))
    return np.array([(ke + se) / E0, ke / E0, se / E0, float(nsteps)], dtype=np.float64)

import numpy as np
from scipy.optimize import brentq

def _check_order(k):
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
        raise ValueError("bad k")
    return int(k)

def _check_blend(alpha_blend, blend_period):
    a = float(alpha_blend)
    if not np.isfinite(a) or a <= 0.0 or a > 1.0:
        raise ValueError("bad alpha_blend")
    if not isinstance(blend_period, (int, np.integer)) or isinstance(blend_period, bool) or blend_period < 1:
        raise ValueError("bad blend_period")
    return a, int(blend_period)

def _bar_particles(L_r, dx, s, rho):
    L = float(L_r); d = float(dx); sf = float(s); r = float(rho)
    if not (np.isfinite(L) and np.isfinite(d) and np.isfinite(sf) and np.isfinite(r)) or L <= 0.0 or d <= 0.0 or r <= 0.0:
        raise ValueError("bad bar geometry or density")
    ncell = int(round(L / d))
    if ncell < 1 or abs(ncell * d - L) > 1e-9 * max(1.0, L):
        raise ValueError("the bar length must be a whole number of cells")
    if not (0.0 < sf < 0.5):
        raise ValueError("the placement fraction must lie strictly between 0 and 0.5")
    n = ncell + 1
    Xp = np.array([(i + t) * d for i in range(ncell) for t in (sf, 1.0 - sf)])
    Vol = np.full(Xp.size, 0.5 * d)
    Mp = r * Vol
    return Xp, Mp, Vol, n, d

def fmpm_stability_audit(L_r: float, dx: float, s: float, s_clustered: float, E: float, rho: float, v0: float, k_ref: int, k_high: int, alpha_blend: float, C_ref: float, C_ret: float, periods: float, tol: float, C_max: float, dC: float) -> "np.ndarray":
    k_ref = _check_order(k_ref); k_high = _check_order(k_high)
    a, _ = _check_blend(alpha_blend, 1)
    Cr = float(C_ref); Ct = float(C_ret)
    if not np.isfinite(Cr) or Cr <= 0.0 or not np.isfinite(Ct) or Ct <= 0.0:
        raise ValueError("bad reference Courant numbers")
    configs = [(1.0, "FLIP", 1, 1.0, 1), (2.0, "FMPM", 1, 1.0, 1), (2.0, "FMPM", 2, 1.0, 1), (2.0, "FMPM", k_ref, 1.0, 1),
               (2.0, "FMPM", k_high, 1.0, 1), (3.0, "EXACT", 1, 1.0, 1), (2.0, "FMPM", k_ref, a, 1), (2.0, "FMPM", k_ref, a, 2),
               (2.0, "FMPM", k_high, a, 1), (2.0, "FMPM", k_high, a, 2)]
    ncol = 12
    rows = np.zeros((len(configs), ncol), dtype=np.float64)
    Xp, Mp, Vol, n, d = _bar_particles(L_r, dx, s, rho)
    N = Xp.size
    ops = fmpm_grid_operators(Xp, Mp, n, d)
    Xc, Mc, Vc, nc, dc = _bar_particles(L_r, dx, s_clustered, rho)
    opsc = fmpm_grid_operators(Xc, Mc, nc, dc)
    for r, (code, scheme, k, al, per) in enumerate(configs):
        Clim = fmpm_stability_limit(L_r, dx, s, E, rho, scheme, k, al, per, tol, C_max, dC)
        Cclu = fmpm_stability_limit(L_r, dx, s_clustered, E, rho, scheme, k, al, per, tol, C_max, dC)
        ret = fmpm_energy_retention(L_r, dx, s, E, rho, scheme, k, al, per, Ct, v0, periods)
        if scheme == "FLIP":
            rate, err = 0.0, 0.0
        else:
            inv = fmpm_inverse_operator(ops, N, k, [0], al, per)
            rate = inv[2 * n, 1]
            err = 0.0 if scheme == "EXACT" else inv[2 * n, 2]
        rows[r, :] = [code, float(k), al, float(per), Clim, Cclu, ret[0], rate, err, ret[3], 0.0, 0.0]
    inv_s = fmpm_inverse_operator(ops, N, k_ref, [0], 1.0, 1)
    inv_c = fmpm_inverse_operator(opsc, Xc.size, k_ref, [0], 1.0, 1)
    ret_ref = [fmpm_energy_retention(L_r, dx, s, E, rho, sc, k, al, per, Cr, v0, periods)[0]
               for (sc, k, al, per) in (("FMPM", 1, 1.0, 1), ("FMPM", k_ref, 1.0, 1), ("FMPM", k_ref, a, 1), ("FMPM", k_ref, a, 2))]
    vwave = np.sqrt(float(E) / float(rho))
    # the source's original expansion against the revised loop at order k_ref on the design bar without controlled nodes
    eye = np.eye(n)
    K_loop = fmpm_inverse_operator(ops, N, k_ref, None, 1.0, 1)[0:n, :]
    K_prior = np.column_stack([fmpm_prior_series(ops, N, eye[:, j], k_ref)[0, :] for j in range(n)])
    expansion_gap = float(np.max(np.abs(K_loop - K_prior)))
    out = np.zeros((2 + len(configs), ncol), dtype=np.float64)
    out[0, :] = [rows[3, 4], rows[0, 4], rows[1, 4], rows[5, 4], rows[5, 5], rows[0, 5], ret_ref[0], ret_ref[1], ret_ref[2], ret_ref[3], inv_s[2 * n, 0], inv_c[2 * nc, 0]]
    out[1, :] = [float(N), float(n), vwave, 4.0 * float(L_r) / vwave, rows[3, 9], rows[2, 4], rows[4, 4], rows[6, 4], rows[7, 4], inv_s[2 * n, 2], inv_c[2 * nc, 2], expansion_gap]
    out[2:, :] = rows
    return out
SCICODE_GOLD_EOF
