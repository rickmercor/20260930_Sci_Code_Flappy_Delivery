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


def _PATH_AMP():
    return 0.15

def _guide(t):
    t = np.asarray(t, dtype=float)
    return np.stack([t, _PATH_AMP()*np.sin(np.pi*t)], -1)

def reference_path(n_images: int) -> "np.ndarray":
    n_images = int(n_images)
    if n_images < 2:
        raise ValueError("n_images must be at least 2")
    t = np.linspace(0.0, 1.0, 40001)
    pc = _guide(t)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pc, axis=0), axis=1))])
    target = np.linspace(0.0, arc[-1], n_images)
    return np.stack([np.interp(target, arc, pc[:, 0]),
                     np.interp(target, arc, pc[:, 1])], 1)

import numpy as np


def _lambda_sharpness(images):
    images = np.atleast_2d(np.asarray(images, dtype=float))
    if images.shape[0] < 2:
        raise ValueError("at least two reference images are needed")
    return 2.3/float(np.mean(np.sum(np.diff(images, axis=0)**2, axis=1)))

def pcv_coordinates(points: "np.ndarray", images: "np.ndarray") -> "np.ndarray":
    points = np.atleast_2d(np.asarray(points, dtype=float))
    images = np.atleast_2d(np.asarray(images, dtype=float))
    lam_sharp = _lambda_sharpness(images)
    if points.shape[1] != 2 or images.shape[1] != 2:
        raise ValueError("points and images must have two columns")
    M = images.shape[0]
    idx = np.arange(M, dtype=float)
    out = np.empty((points.shape[0], 2))
    step = 60000
    for a in range(0, points.shape[0], step):
        p = points[a:a+step]
        d = (p[:, None, 0]-images[None, :, 0])**2 + (p[:, None, 1]-images[None, :, 1])**2
        dmin = d.min(1, keepdims=True)
        w = np.exp(-lam_sharp*(d-dmin))
        tot = w.sum(1)
        out[a:a+step, 0] = (w @ idx)/tot/(M-1)
        out[a:a+step, 1] = -(np.log(tot) - lam_sharp*dmin[:, 0])/lam_sharp
    return out

import numpy as np


def _PATH_AMP():
    return 0.15

def _guide(t):
    t = np.asarray(t, dtype=float)
    return np.stack([t, _PATH_AMP()*np.sin(np.pi*t)], -1)

def _guide_d1(t):
    t = np.asarray(t, dtype=float)
    return np.stack([np.ones_like(t), _PATH_AMP()*np.pi*np.cos(np.pi*t)], -1)

def _guide_d2(t):
    t = np.asarray(t, dtype=float)
    return np.stack([np.zeros_like(t), -_PATH_AMP()*np.pi**2*np.sin(np.pi*t)], -1)

def _local_frame(t):
    d1 = _guide_d1(t); speed = np.linalg.norm(d1, axis=-1)
    tang = d1/speed[..., None]
    normal = np.stack([-tang[..., 1], tang[..., 0]], -1)
    d2 = _guide_d2(t)
    curv = (d1[..., 0]*d2[..., 1] - d1[..., 1]*d2[..., 0])/speed**3
    return normal, speed, curv

def station_frame(images: "np.ndarray", n_stations: int) -> "np.ndarray":
    n_stations = int(n_stations)
    if n_stations < 3:
        raise ValueError("n_stations must be at least 3")
    s_grid = np.linspace(0.0, 1.0, n_stations)

    def _progress(t):
        return pcv_coordinates(_guide(np.atleast_1d(t)), images)[:, 0]

    lo = np.zeros(n_stations); hi = np.ones(n_stations)
    s_lo = _progress(lo); s_hi = _progress(hi)
    for _ in range(60):
        mid = 0.5*(lo + hi)
        s_mid = _progress(mid)
        left = s_mid < s_grid
        lo = np.where(left, mid, lo)
        hi = np.where(left, hi, mid)
    t_st = np.clip(0.5*(lo + hi), 0.0, 1.0)
    t_st[0] = 0.0
    t_st[-1] = 1.0
    on_path = pcv_coordinates(_guide(t_st), images)
    _, speed, curv = _local_frame(t_st)
    return np.stack([s_grid, t_st, on_path[:, 1], speed, curv], 1)

import numpy as np


def _K_B():
    return 0.0019872041

def _TEMP():
    return 300.0

def _BETA():
    return 1.0/(_K_B()*_TEMP())

def _PATH_AMP():
    return 0.15

def _K_SOFT():
    return 40.0

def _K_STIFF():
    return 400.0

def _W_NECK():
    return 0.055

def _T_SAD1():
    return 0.30

def _T_SAD2():
    return 0.70

def _CUBIC():
    return 30.0

def _QUART():
    return 140.0

def _QUART_M():
    return 0.60

def _RIDGE():
    return 0.35

def _RIDGE_T():
    return 0.10

def _RIDGE_N():
    return 0.055

def _guide(t):
    t = np.asarray(t, dtype=float)
    return np.stack([t, _PATH_AMP()*np.sin(np.pi*t)], -1)

def _guide_d1(t):
    t = np.asarray(t, dtype=float)
    return np.stack([np.ones_like(t), _PATH_AMP()*np.pi*np.cos(np.pi*t)], -1)

def _guide_d2(t):
    t = np.asarray(t, dtype=float)
    return np.stack([np.zeros_like(t), -_PATH_AMP()*np.pi**2*np.sin(np.pi*t)], -1)

def _stiffness(t):
    t = np.asarray(t, dtype=float)
    return _K_SOFT() + (_K_STIFF()-_K_SOFT())*(np.exp(-((t-_T_SAD1())/_W_NECK())**2)
                                      + np.exp(-((t-_T_SAD2())/_W_NECK())**2))

def _along_path_energy(t):
    t = np.asarray(t, dtype=float)
    return (7.5*np.exp(-((t-_T_SAD1())/0.10)**2)
            + 6.2*np.exp(-((t-_T_SAD2())/0.10)**2)
            - 1.6*np.exp(-((t-0.50)/0.11)**2))

def _energy(t, n):
    t = np.asarray(t, dtype=float); n = np.asarray(n, dtype=float)
    return (_along_path_energy(t)
            + 0.5*_stiffness(t)*n**2
            + _CUBIC()*np.sin(2.0*np.pi*t)*n**3
            + _QUART()*(1.0+_QUART_M()*np.cos(2.0*np.pi*t))*n**4
            + _RIDGE()*np.exp(-((t-0.50)/_RIDGE_T())**2)*np.exp(-(n/_RIDGE_N())**2))

def _local_frame(t):
    d1 = _guide_d1(t); speed = np.linalg.norm(d1, axis=-1)
    tang = d1/speed[..., None]
    normal = np.stack([-tang[..., 1], tang[..., 0]], -1)
    d2 = _guide_d2(t)
    curv = (d1[..., 0]*d2[..., 1] - d1[..., 1]*d2[..., 0])/speed**3
    return normal, speed, curv

def orthogonal_slices(stations: "np.ndarray", images: "np.ndarray", n_ortho: int, n_max: float) -> "np.ndarray":
    stations = np.atleast_2d(np.asarray(stations, dtype=float))
    n_ortho = int(n_ortho); n_max = float(n_max)
    if stations.shape[1] != 5:
        raise ValueError("stations must have five columns")
    if n_ortho < 3 or n_ortho % 2 == 0:
        raise ValueError("n_ortho must be an odd integer of at least 3")
    if n_max <= 0.0:
        raise ValueError("n_max must be positive")
    ng = np.linspace(-n_max, n_max, n_ortho)
    t_st = stations[:, 1]; z_floor = stations[:, 2]
    speed = stations[:, 3]; curv = stations[:, 4]
    normal, _, _ = _local_frame(t_st)
    xy = _guide(t_st)[:, None, :] + ng[None, :, None]*normal[:, None, :]
    z = pcv_coordinates(xy.reshape(-1, 2), images)[:, 1]
    z = z.reshape(len(t_st), n_ortho)
    u = _energy(t_st[:, None], ng[None, :])
    w = np.exp(-_BETA()*(u - u.min()))*np.abs(1.0 - ng[None, :]*curv[:, None])*speed[:, None]
    return np.stack([z - z_floor[:, None], w], -1)

import numpy as np


def _K_B():
    return 0.0019872041

def _TEMP():
    return 300.0

def _BETA():
    return 1.0/(_K_B()*_TEMP())

def conditional_free_energy(slices: "np.ndarray", n_ortho: int, n_max: float, d_eff: int) -> "np.ndarray":
    slices = np.asarray(slices, dtype=float)
    n_ortho = int(n_ortho); n_max = float(n_max); d_eff = int(d_eff)
    if slices.ndim != 3 or slices.shape[2] != 2:
        raise ValueError("slices must have shape (n_stations, n_ortho, 2)")
    if d_eff < 2:
        raise ValueError("d_eff must be at least 2")
    d_perp = d_eff - 1
    ng = np.linspace(-n_max, n_max, n_ortho)
    pos = ng > 0.0; neg = ng < 0.0
    npos = ng[pos]
    out = np.empty((slices.shape[0], npos.size, 2))
    for a in range(slices.shape[0]):
        dz = slices[a, :, 0]; w = slices[a, :, 1]
        folded = w[pos] + np.interp(-npos, ng[neg], w[neg])
        slope = np.gradient(dz, ng)[pos]
        dens = folded/np.abs(slope)
        order = np.argsort(dz[pos])
        dzp = dz[pos][order]; dens = dens[order]
        with np.errstate(divide="ignore", invalid="ignore"):
            fhat = -np.log(dens*np.maximum(dzp, 0.0)**((2.0-d_perp)/2.0))/_BETA()
        out[a, :, 0] = dzp
        out[a, :, 1] = fhat
    return out

import numpy as np


def _K_B():
    return 0.0019872041

def _TEMP():
    return 300.0

def _BETA():
    return 1.0/(_K_B()*_TEMP())

def adaptive_half_width(cond: "np.ndarray", slices: "np.ndarray", delta_f_star: float, d_eff: int, dz_ref: float) -> "np.ndarray":
    cond = np.asarray(cond, dtype=float); slices = np.asarray(slices, dtype=float)
    delta_f_star = float(delta_f_star); d_eff = int(d_eff); dz_ref = float(dz_ref)
    if cond.ndim != 3 or cond.shape[2] != 2:
        raise ValueError("cond must have shape (n_stations, n_half, 2)")
    if delta_f_star <= 0.0:
        raise ValueError("delta_f_star must be positive")
    if d_eff < 2:
        raise ValueError("d_eff must be at least 2")
    if dz_ref <= 0.0:
        raise ValueError("dz_ref must be positive")
    d_perp = d_eff - 1
    ns = cond.shape[0]
    out = np.empty(ns)
    for a in range(ns):
        dz = cond[a, :, 0]; f = cond[a, :, 1]
        ok = np.isfinite(f)
        good = np.where(ok)[0]
        f_ref = float(np.interp(dz_ref, dz[good], f[good]))
        excess = f - f_ref
        base = int(np.searchsorted(dz, dz_ref))
        found = np.nan
        for q in range(max(base-1, int(good[0])), dz.size-1):
            if ok[q] and ok[q+1] and excess[q] <= delta_f_star <= excess[q+1]:
                found = dz[q] + (delta_f_star-excess[q])/(excess[q+1]-excess[q])*(dz[q+1]-dz[q])
                break
        if not np.isfinite(found):
            dza = slices[a, :, 0]; wa = slices[a, :, 1]
            keep = dza >= 0.0
            mean = float(np.sum(wa[keep]*dza[keep])/np.sum(wa[keep]))
            found = (2.0*_BETA()*delta_f_star/d_perp)*mean
        out[a] = found
    return out

import numpy as np


def smooth_envelope(profile: "np.ndarray", window: int) -> "np.ndarray":
    profile = np.asarray(profile, dtype=float).ravel()
    window = int(window)
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if window == 1:
        return profile.copy()
    filled = np.nan_to_num(profile, nan=float(np.nanmean(profile)))
    h = window//2
    padded = np.concatenate([filled[:h][::-1], filled, filled[-h:][::-1]])
    return np.convolve(padded, np.ones(window)/window, mode="valid")

import numpy as np


def _K_B():
    return 0.0019872041

def _TEMP():
    return 300.0

def _BETA():
    return 1.0/(_K_B()*_TEMP())

def tube_projected_profile(slices: "np.ndarray", z_max: "np.ndarray", n_ortho: int, n_max: float) -> "np.ndarray":
    slices = np.asarray(slices, dtype=float)
    z_max = np.asarray(z_max, dtype=float).ravel()
    n_ortho = int(n_ortho); n_max = float(n_max)
    if slices.shape[0] != z_max.size:
        raise ValueError("z_max must carry one entry per station")
    dn = 2.0*n_max/(n_ortho-1)
    out = np.empty((slices.shape[0], 3))
    for a in range(slices.shape[0]):
        dz = slices[a, :, 0]; w = slices[a, :, 1]
        total = float(w.sum())
        inside = float(w[dz <= z_max[a]].sum())
        p = inside/total
        f_marg = -np.log(total*dn)/_BETA()
        out[a] = (f_marg, p, f_marg - np.log(p)/_BETA())
    return out

import numpy as np
from scipy.special import gammainc


def _K_B():
    return 0.0019872041

def _TEMP():
    return 300.0

def _BETA():
    return 1.0/(_K_B()*_TEMP())

def harmonic_tube_reference(d_eff: int, delta_f_star: float, stiffness: "np.ndarray") -> "np.ndarray":
    d_eff = int(d_eff); delta_f_star = float(delta_f_star)
    stiffness = np.atleast_1d(np.asarray(stiffness, dtype=float))
    if d_eff < 2:
        raise ValueError("d_eff must be at least 2")
    if delta_f_star <= 0.0 or np.any(stiffness <= 0.0):
        raise ValueError("delta_f_star and stiffness must be positive")
    d_perp = d_eff - 1
    frac = float(gammainc(d_perp/2.0, _BETA()*delta_f_star))
    eps = np.sqrt(2.0*delta_f_star/stiffness)
    walls = np.full(stiffness.shape, np.sqrt(2.0*_BETA()*delta_f_star))
    return np.stack([np.full(stiffness.shape, frac), eps, walls], 1)

import numpy as np


def _K_B():
    return 0.0019872041

def _TEMP():
    return 300.0

def _KT():
    return _K_B()*_TEMP()

def _K_SOFT():
    return 40.0

def _K_STIFF():
    return 400.0

def _W_NECK():
    return 0.055

def _T_SAD1():
    return 0.30

def _T_SAD2():
    return 0.70

def _stiffness(t):
    t = np.asarray(t, dtype=float)
    return _K_SOFT() + (_K_STIFF()-_K_SOFT())*(np.exp(-((t-_T_SAD1())/_W_NECK())**2)
                                      + np.exp(-((t-_T_SAD2())/_W_NECK())**2))

def _lambda_sharpness(images):
    images = np.atleast_2d(np.asarray(images, dtype=float))
    if images.shape[0] < 2:
        raise ValueError("at least two reference images are needed")
    return 2.3/float(np.mean(np.sum(np.diff(images, axis=0)**2, axis=1)))

def tube_audit(n_images: int, n_stations: int, n_ortho: int, n_max: float, d_eff: int, smooth_window: int, dz_ref: float, contours: tuple) -> "np.ndarray":
    contours = tuple(float(c) for c in contours)
    if len(contours) == 0:
        raise ValueError("at least one contour level is required")
    images = reference_path(n_images)
    image_pcv = pcv_coordinates(images, images)
    stations = station_frame(images, n_stations)
    slices = orthogonal_slices(stations, images, n_ortho, n_max)
    cond = conditional_free_energy(slices, n_ortho, n_max, d_eff)
    s_grid = stations[:, 0]
    kappa = _stiffness(stations[:, 1])
    rows = []
    total = 0.0
    for c in contours:
        dfs = c*_KT()
        raw = adaptive_half_width(cond, slices, dfs, d_eff, dz_ref)
        env = smooth_envelope(raw, smooth_window)
        prof = tube_projected_profile(slices, env, n_ortho, n_max)
        ref = harmonic_tube_reference(d_eff, dfs, kappa)
        integral = float(np.trapezoid(env, s_grid))
        total += integral
        rows.append([c, dfs, integral, float(env.min()), float(env.max()),
                     float(prof[:, 1].mean()), float(ref[0, 0]),
                     float(np.mean(raw/(2.0*dfs/kappa))),
                     float(prof[:, 2].max()-prof[:, 2].min())])
    audit = np.array(rows, dtype=float)
    head = np.zeros((1, audit.shape[1]))
    head[0, 0] = total
    head[0, 1] = float(_lambda_sharpness(images))
    head[0, 2] = float(stations[:, 2].mean())
    head[0, 3] = float(kappa.min())
    head[0, 4] = float(kappa.max())
    head[0, 5] = float(image_pcv[:, 1].mean())
    head[0, 6] = float(image_pcv[:, 0].max() - image_pcv[:, 0].min())
    return np.vstack([head, audit])
SCICODE_GOLD_EOF
