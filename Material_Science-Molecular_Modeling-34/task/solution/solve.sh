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
from scipy.optimize import fsolve
from scipy.signal import fftconvolve


def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _scalar(x, name, positive=False):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError("non-finite " + name)
    if positive and v <= 0.0:
        raise ValueError("non-positive " + name)
    return v

def _grid(x, name):
    a = _f64(x)
    if a.ndim != 1 or a.size < 3 or not np.all(np.isfinite(a)):
        raise ValueError("bad 1D array " + name)
    return a

def _profile(P):
    P = _f64(P)
    if P.ndim != 2 or P.shape[0] != 2 or P.shape[1] < 8 or not np.all(np.isfinite(P)):
        raise ValueError("bad (2,N) profile array")
    return P

def _weights(W):
    W = _f64(W)
    if W.ndim != 2 or W.shape[0] != 6 or W.shape[1] < 1 or not np.all(np.isfinite(W)):
        raise ValueError("bad (6,M) weight array")
    return W

def _pad_conv(f, w, dz):
    hw = len(w) // 2
    fp = np.pad(f, hw, mode="edge")
    return (fftconvolve(fp, w, mode="same") * dz)[hw:hw + len(f)]

def _muhs_ex(r):
    e = np.pi * r / 6.0
    return (8 * e - 9 * e ** 2 + 3 * e ** 3) / (1 - e) ** 3

def _wb_derivs(n0, n1, n2, n3, nv1, nv2):
    n3 = np.clip(n3, 1e-10, 1 - 1e-9)
    s = 1 - n3
    Lg = np.log(s)
    g3 = (n3 + s ** 2 * Lg) / (36 * np.pi * n3 ** 2 * s ** 2)
    dh = ((n3 - 2 * s * Lg) * n3 * s - (n3 + s ** 2 * Lg) * 2 * (1 - 2 * n3)) / (n3 ** 3 * s ** 3)
    g3p = dh / (36 * np.pi)
    dp0 = -Lg
    dp1 = n2 / s
    dp2 = n1 / s + (3 * n2 ** 2 - 3 * nv2 ** 2) * g3
    dp3 = n0 / s + (n1 * n2 - nv1 * nv2) / s ** 2 + (n2 ** 3 - 3 * n2 * nv2 ** 2) * g3p
    dpv1 = -nv2 / s
    dpv2 = -nv1 / s - 6 * n2 * nv2 * g3
    return dp0, dp1, dp2, dp3, dpv1, dpv2

def sw_bulk_coexistence(beps, lam):
    """CS hard-sphere + RPA square-well mean field; solve equal mu and equal P."""
    be = _scalar(beps, "beps", positive=True)
    lm = _scalar(lam, "lam")
    if lm <= 1.0:
        raise ValueError("lam must exceed 1")
    def bmu(r):
        return np.log(r) + _muhs_ex(r) - (4 * np.pi / 3) * lm ** 3 * be * r
    def bP(r):
        e = np.pi * r / 6.0
        return r * (1 + e + e ** 2 - e ** 3) / (1 - e) ** 3 - (2 * np.pi / 3) * r ** 2 * lm ** 3 * be
    def F(y):
        rl, rv = np.exp(y[0]), np.exp(y[1])
        return [bmu(rl) - bmu(rv), bP(rl) - bP(rv)]
    for gl, gv in [(0.60, 0.04), (0.67, 0.02), (0.72, 0.01), (0.55, 0.08),
                   (0.78, 0.004), (0.50, 0.10), (0.83, 0.002)]:
        y = fsolve(F, [np.log(gl), np.log(gv)])
        rl, rv = float(np.exp(y[0])), float(np.exp(y[1]))
        if np.max(np.abs(F(y))) < 1e-9 and rl > 1.01 * rv and rl < 0.95:
            return np.array([rl, rv], dtype=np.float64)
    raise ValueError("no coexistence found for these parameters")

import numpy as np
from scipy.optimize import fsolve
from scipy.signal import fftconvolve


def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _scalar(x, name, positive=False):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError("non-finite " + name)
    if positive and v <= 0.0:
        raise ValueError("non-positive " + name)
    return v

def _grid(x, name):
    a = _f64(x)
    if a.ndim != 1 or a.size < 3 or not np.all(np.isfinite(a)):
        raise ValueError("bad 1D array " + name)
    return a

def _profile(P):
    P = _f64(P)
    if P.ndim != 2 or P.shape[0] != 2 or P.shape[1] < 8 or not np.all(np.isfinite(P)):
        raise ValueError("bad (2,N) profile array")
    return P

def _weights(W):
    W = _f64(W)
    if W.ndim != 2 or W.shape[0] != 6 or W.shape[1] < 1 or not np.all(np.isfinite(W)):
        raise ValueError("bad (6,M) weight array")
    return W

def _pad_conv(f, w, dz):
    hw = len(w) // 2
    fp = np.pad(f, hw, mode="edge")
    return (fftconvolve(fp, w, mode="same") * dz)[hw:hw + len(f)]

def _muhs_ex(r):
    e = np.pi * r / 6.0
    return (8 * e - 9 * e ** 2 + 3 * e ** 3) / (1 - e) ** 3

def _wb_derivs(n0, n1, n2, n3, nv1, nv2):
    n3 = np.clip(n3, 1e-10, 1 - 1e-9)
    s = 1 - n3
    Lg = np.log(s)
    g3 = (n3 + s ** 2 * Lg) / (36 * np.pi * n3 ** 2 * s ** 2)
    dh = ((n3 - 2 * s * Lg) * n3 * s - (n3 + s ** 2 * Lg) * 2 * (1 - 2 * n3)) / (n3 ** 3 * s ** 3)
    g3p = dh / (36 * np.pi)
    dp0 = -Lg
    dp1 = n2 / s
    dp2 = n1 / s + (3 * n2 ** 2 - 3 * nv2 ** 2) * g3
    dp3 = n0 / s + (n1 * n2 - nv1 * nv2) / s ** 2 + (n2 ** 3 - 3 * n2 * nv2 ** 2) * g3p
    dpv1 = -nv2 / s
    dpv2 = -nv1 / s - 6 * n2 * nv2 * g3
    return dp0, dp1, dp2, dp3, dpv1, dpv2

def fmt_weight_functions(R, dz):
    """Planar FMT weights on |t|<=R; scalar weights rescaled to exact 3D moments."""
    Rr = _scalar(R, "R", positive=True)
    dd = _scalar(dz, "dz", positive=True)
    if dd > Rr:
        raise ValueError("dz must not exceed R")
    t = np.arange(-Rr, Rr + dd / 2, dd)
    w3 = np.pi * (Rr ** 2 - t ** 2)
    w2 = np.full_like(t, 2 * np.pi * Rr)
    wv2 = 2 * np.pi * t
    w3 *= (4 * np.pi * Rr ** 3 / 3) / (w3.sum() * dd)
    w2 *= (4 * np.pi * Rr ** 2) / (w2.sum() * dd)
    w1 = w2 / (4 * np.pi * Rr)
    w0 = w2 / (4 * np.pi * Rr ** 2)
    wv1 = wv2 / (4 * np.pi * Rr)
    return np.vstack([w0, w1, w2, w3, wv1, wv2]).astype(np.float64)

import numpy as np
from scipy.optimize import fsolve
from scipy.signal import fftconvolve


def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _scalar(x, name, positive=False):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError("non-finite " + name)
    if positive and v <= 0.0:
        raise ValueError("non-positive " + name)
    return v

def _grid(x, name):
    a = _f64(x)
    if a.ndim != 1 or a.size < 3 or not np.all(np.isfinite(a)):
        raise ValueError("bad 1D array " + name)
    return a

def _profile(P):
    P = _f64(P)
    if P.ndim != 2 or P.shape[0] != 2 or P.shape[1] < 8 or not np.all(np.isfinite(P)):
        raise ValueError("bad (2,N) profile array")
    return P

def _weights(W):
    W = _f64(W)
    if W.ndim != 2 or W.shape[0] != 6 or W.shape[1] < 1 or not np.all(np.isfinite(W)):
        raise ValueError("bad (6,M) weight array")
    return W

def _pad_conv(f, w, dz):
    hw = len(w) // 2
    fp = np.pad(f, hw, mode="edge")
    return (fftconvolve(fp, w, mode="same") * dz)[hw:hw + len(f)]

def _muhs_ex(r):
    e = np.pi * r / 6.0
    return (8 * e - 9 * e ** 2 + 3 * e ** 3) / (1 - e) ** 3

def _wb_derivs(n0, n1, n2, n3, nv1, nv2):
    n3 = np.clip(n3, 1e-10, 1 - 1e-9)
    s = 1 - n3
    Lg = np.log(s)
    g3 = (n3 + s ** 2 * Lg) / (36 * np.pi * n3 ** 2 * s ** 2)
    dh = ((n3 - 2 * s * Lg) * n3 * s - (n3 + s ** 2 * Lg) * 2 * (1 - 2 * n3)) / (n3 ** 3 * s ** 3)
    g3p = dh / (36 * np.pi)
    dp0 = -Lg
    dp1 = n2 / s
    dp2 = n1 / s + (3 * n2 ** 2 - 3 * nv2 ** 2) * g3
    dp3 = n0 / s + (n1 * n2 - nv1 * nv2) / s ** 2 + (n2 ** 3 - 3 * n2 * nv2 ** 2) * g3p
    dpv1 = -nv2 / s
    dpv2 = -nv1 / s - 6 * n2 * nv2 * g3
    return dp0, dp1, dp2, dp3, dpv1, dpv2

def sw_meanfield_kernel(beps, rng, dz):
    """Transverse integral of -beps over the disk of radius sqrt(rng^2 - t^2)."""
    be = _scalar(beps, "beps", positive=True)
    rg = _scalar(rng, "rng", positive=True)
    dd = _scalar(dz, "dz", positive=True)
    if dd > rg:
        raise ValueError("dz must not exceed rng")
    t = np.arange(-rg, rg + dd / 2, dd)
    return (-np.pi * be * (rg ** 2 - t ** 2)).astype(np.float64)

import numpy as np
from scipy.optimize import fsolve
from scipy.signal import fftconvolve


def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _scalar(x, name, positive=False):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError("non-finite " + name)
    if positive and v <= 0.0:
        raise ValueError("non-positive " + name)
    return v

def _grid(x, name):
    a = _f64(x)
    if a.ndim != 1 or a.size < 3 or not np.all(np.isfinite(a)):
        raise ValueError("bad 1D array " + name)
    return a

def _profile(P):
    P = _f64(P)
    if P.ndim != 2 or P.shape[0] != 2 or P.shape[1] < 8 or not np.all(np.isfinite(P)):
        raise ValueError("bad (2,N) profile array")
    return P

def _weights(W):
    W = _f64(W)
    if W.ndim != 2 or W.shape[0] != 6 or W.shape[1] < 1 or not np.all(np.isfinite(W)):
        raise ValueError("bad (6,M) weight array")
    return W

def _pad_conv(f, w, dz):
    hw = len(w) // 2
    fp = np.pad(f, hw, mode="edge")
    return (fftconvolve(fp, w, mode="same") * dz)[hw:hw + len(f)]

def _muhs_ex(r):
    e = np.pi * r / 6.0
    return (8 * e - 9 * e ** 2 + 3 * e ** 3) / (1 - e) ** 3

def _wb_derivs(n0, n1, n2, n3, nv1, nv2):
    n3 = np.clip(n3, 1e-10, 1 - 1e-9)
    s = 1 - n3
    Lg = np.log(s)
    g3 = (n3 + s ** 2 * Lg) / (36 * np.pi * n3 ** 2 * s ** 2)
    dh = ((n3 - 2 * s * Lg) * n3 * s - (n3 + s ** 2 * Lg) * 2 * (1 - 2 * n3)) / (n3 ** 3 * s ** 3)
    g3p = dh / (36 * np.pi)
    dp0 = -Lg
    dp1 = n2 / s
    dp2 = n1 / s + (3 * n2 ** 2 - 3 * nv2 ** 2) * g3
    dp3 = n0 / s + (n1 * n2 - nv1 * nv2) / s ** 2 + (n2 ** 3 - 3 * n2 * nv2 ** 2) * g3p
    dpv1 = -nv2 / s
    dpv2 = -nv1 / s - 6 * n2 * nv2 * g3
    return dp0, dp1, dp2, dp3, dpv1, dpv2

def solvent_density_profile(rho_l, rho_v, weights, kernel, beps, lam, dz, L):
    """Damped Picard on ln rho with edge (bulk) padding and pinned bulk tails."""
    W = _weights(weights)
    K = _grid(kernel, "kernel")
    rl = _scalar(rho_l, "rho_l", positive=True)
    rv = _scalar(rho_v, "rho_v", positive=True)
    be = _scalar(beps, "beps", positive=True)
    lm = _scalar(lam, "lam", positive=True)
    dd = _scalar(dz, "dz", positive=True)
    LL = _scalar(L, "L", positive=True)
    if rl <= rv:
        raise ValueError("need rho_l > rho_v")
    w0, w1, w2, w3, wv1, wv2 = W
    z = np.arange(-LL / 2, LL / 2 + dd / 2, dd)
    def c1_fmt(rho):
        n0 = _pad_conv(rho, w0, dd); n1 = _pad_conv(rho, w1, dd)
        n2 = _pad_conv(rho, w2, dd); n3 = _pad_conv(rho, w3, dd)
        nv1 = _pad_conv(rho, wv1, dd); nv2 = _pad_conv(rho, wv2, dd)
        d0, d1, d2, d3, dv1, dv2 = _wb_derivs(n0, n1, n2, n3, nv1, nv2)
        return -(_pad_conv(d0, w0, dd) + _pad_conv(d1, w1, dd) + _pad_conv(d2, w2, dd)
                 + _pad_conv(d3, w3, dd) + _pad_conv(dv1, -wv1, dd) + _pad_conv(dv2, -wv2, dd))
    bmu = np.log(rl) + _muhs_ex(rl) - (4 * np.pi / 3) * lm ** 3 * be * rl
    rho = rv + (rl - rv) * 0.5 * (1 + np.tanh(z / 1.0))
    npin = int(lm / dd) + 10
    for _ in range(25000):
        rt = np.exp(np.clip(bmu + c1_fmt(rho) - _pad_conv(rho, K, dd), -40, 3))
        rt[:npin] = rv
        rt[-npin:] = rl
        d = np.max(np.abs(rt - rho))
        rho = 0.15 * rt + 0.85 * rho
        if d < 1e-6:
            break
    return np.vstack([z, rho]).astype(np.float64)

import numpy as np
from scipy.optimize import fsolve
from scipy.signal import fftconvolve


def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _scalar(x, name, positive=False):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError("non-finite " + name)
    if positive and v <= 0.0:
        raise ValueError("non-positive " + name)
    return v

def _grid(x, name):
    a = _f64(x)
    if a.ndim != 1 or a.size < 3 or not np.all(np.isfinite(a)):
        raise ValueError("bad 1D array " + name)
    return a

def _profile(P):
    P = _f64(P)
    if P.ndim != 2 or P.shape[0] != 2 or P.shape[1] < 8 or not np.all(np.isfinite(P)):
        raise ValueError("bad (2,N) profile array")
    return P

def _weights(W):
    W = _f64(W)
    if W.ndim != 2 or W.shape[0] != 6 or W.shape[1] < 1 or not np.all(np.isfinite(W)):
        raise ValueError("bad (6,M) weight array")
    return W

def _pad_conv(f, w, dz):
    hw = len(w) // 2
    fp = np.pad(f, hw, mode="edge")
    return (fftconvolve(fp, w, mode="same") * dz)[hw:hw + len(f)]

def _muhs_ex(r):
    e = np.pi * r / 6.0
    return (8 * e - 9 * e ** 2 + 3 * e ** 3) / (1 - e) ** 3

def _wb_derivs(n0, n1, n2, n3, nv1, nv2):
    n3 = np.clip(n3, 1e-10, 1 - 1e-9)
    s = 1 - n3
    Lg = np.log(s)
    g3 = (n3 + s ** 2 * Lg) / (36 * np.pi * n3 ** 2 * s ** 2)
    dh = ((n3 - 2 * s * Lg) * n3 * s - (n3 + s ** 2 * Lg) * 2 * (1 - 2 * n3)) / (n3 ** 3 * s ** 3)
    g3p = dh / (36 * np.pi)
    dp0 = -Lg
    dp1 = n2 / s
    dp2 = n1 / s + (3 * n2 ** 2 - 3 * nv2 ** 2) * g3
    dp3 = n0 / s + (n1 * n2 - nv1 * nv2) / s ** 2 + (n2 ** 3 - 3 * n2 * nv2 ** 2) * g3p
    dpv1 = -nv2 / s
    dpv2 = -nv1 / s - 6 * n2 * nv2 * g3
    return dp0, dp1, dp2, dp3, dpv1, dpv2

def surface_tension(profile, weights, kernel, beps, lam, dz):
    """Excess grand potential per area of the solvent profile."""
    if np.asarray(profile, dtype=float).ndim != 2 or float(dz) <= 0.0:
        raise ValueError("profile must be a (2, N) array and dz must be positive")
    P = _profile(profile)
    W = _weights(weights)
    K = _grid(kernel, "kernel")
    be = _scalar(beps, "beps", positive=True)
    lm = _scalar(lam, "lam", positive=True)
    dd = _scalar(dz, "dz", positive=True)
    w0, w1, w2, w3, wv1, wv2 = W
    rho1 = P[1]
    N = rho1.size
    def bPhi(rho):
        n0 = _pad_conv(rho, w0, dd); n1 = _pad_conv(rho, w1, dd)
        n2 = _pad_conv(rho, w2, dd); n3 = _pad_conv(rho, w3, dd)
        nv1 = _pad_conv(rho, wv1, dd); nv2 = _pad_conv(rho, wv2, dd)
        n3 = np.clip(n3, 1e-10, 1 - 1e-9); s = 1 - n3; Lg = np.log(s)
        return (-n0 * Lg + (n1 * n2 - nv1 * nv2) / s
                + (n2 ** 3 - 3 * n2 * nv2 ** 2) * (n3 + s ** 2 * Lg) / (36 * np.pi * n3 ** 2 * s ** 2))
    def bf(rho):
        return rho * (np.log(np.clip(rho, 1e-30, None)) - 1) + bPhi(rho) + 0.5 * rho * _pad_conv(rho, K, dd)
    rho_l = rho1[-1]
    bmu = np.log(rho_l) + _muhs_ex(rho_l) - (4 * np.pi / 3) * lm ** 3 * be * rho_l
    bP = bmu * rho_l - bf(np.full(N, rho_l))[N // 2]
    return float(np.sum(bf(rho1) - bmu * rho1 + bP) * dd)

import numpy as np
from scipy.optimize import fsolve
from scipy.signal import fftconvolve


def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _scalar(x, name, positive=False):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError("non-finite " + name)
    if positive and v <= 0.0:
        raise ValueError("non-positive " + name)
    return v

def _grid(x, name):
    a = _f64(x)
    if a.ndim != 1 or a.size < 3 or not np.all(np.isfinite(a)):
        raise ValueError("bad 1D array " + name)
    return a

def _profile(P):
    P = _f64(P)
    if P.ndim != 2 or P.shape[0] != 2 or P.shape[1] < 8 or not np.all(np.isfinite(P)):
        raise ValueError("bad (2,N) profile array")
    return P

def _weights(W):
    W = _f64(W)
    if W.ndim != 2 or W.shape[0] != 6 or W.shape[1] < 1 or not np.all(np.isfinite(W)):
        raise ValueError("bad (6,M) weight array")
    return W

def _pad_conv(f, w, dz):
    hw = len(w) // 2
    fp = np.pad(f, hw, mode="edge")
    return (fftconvolve(fp, w, mode="same") * dz)[hw:hw + len(f)]

def _muhs_ex(r):
    e = np.pi * r / 6.0
    return (8 * e - 9 * e ** 2 + 3 * e ** 3) / (1 - e) ** 3

def _wb_derivs(n0, n1, n2, n3, nv1, nv2):
    n3 = np.clip(n3, 1e-10, 1 - 1e-9)
    s = 1 - n3
    Lg = np.log(s)
    g3 = (n3 + s ** 2 * Lg) / (36 * np.pi * n3 ** 2 * s ** 2)
    dh = ((n3 - 2 * s * Lg) * n3 * s - (n3 + s ** 2 * Lg) * 2 * (1 - 2 * n3)) / (n3 ** 3 * s ** 3)
    g3p = dh / (36 * np.pi)
    dp0 = -Lg
    dp1 = n2 / s
    dp2 = n1 / s + (3 * n2 ** 2 - 3 * nv2 ** 2) * g3
    dp3 = n0 / s + (n1 * n2 - nv1 * nv2) / s ** 2 + (n2 ** 3 - 3 * n2 * nv2 ** 2) * g3p
    dpv1 = -nv2 / s
    dpv2 = -nv1 / s - 6 * n2 * nv2 * g3
    return dp0, dp1, dp2, dp3, dpv1, dpv2

def lorentz_berthelot_cross(sig1, lam1, beps1, sig2, lam2, beps2):
    """eps12 = sqrt(eps1 eps2); sigma12 = (sig1+sig2)/2; lam12 = (lam1 sig1 + lam2 sig2)/(sig1+sig2);
    outer range r12 = lam12 * sigma12 = (lam1 sig1 + lam2 sig2)/2."""
    if min(float(sig1), float(lam1), float(beps1), float(sig2), float(lam2), float(beps2)) <= 0.0:
        raise ValueError("all mixing parameters must be positive")
    s1 = _scalar(sig1, "sig1", positive=True); l1 = _scalar(lam1, "lam1", positive=True)
    e1 = _scalar(beps1, "beps1", positive=True); s2 = _scalar(sig2, "sig2", positive=True)
    l2 = _scalar(lam2, "lam2", positive=True); e2 = _scalar(beps2, "beps2", positive=True)
    eps12 = np.sqrt(e1 * e2)
    rng12 = (l1 * s1 + l2 * s2) / 2.0
    return np.array([rng12, eps12], dtype=np.float64)

import numpy as np
from scipy.optimize import fsolve
from scipy.signal import fftconvolve


def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _scalar(x, name, positive=False):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError("non-finite " + name)
    if positive and v <= 0.0:
        raise ValueError("non-positive " + name)
    return v

def _grid(x, name):
    a = _f64(x)
    if a.ndim != 1 or a.size < 3 or not np.all(np.isfinite(a)):
        raise ValueError("bad 1D array " + name)
    return a

def _profile(P):
    P = _f64(P)
    if P.ndim != 2 or P.shape[0] != 2 or P.shape[1] < 8 or not np.all(np.isfinite(P)):
        raise ValueError("bad (2,N) profile array")
    return P

def _weights(W):
    W = _f64(W)
    if W.ndim != 2 or W.shape[0] != 6 or W.shape[1] < 1 or not np.all(np.isfinite(W)):
        raise ValueError("bad (6,M) weight array")
    return W

def _pad_conv(f, w, dz):
    hw = len(w) // 2
    fp = np.pad(f, hw, mode="edge")
    return (fftconvolve(fp, w, mode="same") * dz)[hw:hw + len(f)]

def _muhs_ex(r):
    e = np.pi * r / 6.0
    return (8 * e - 9 * e ** 2 + 3 * e ** 3) / (1 - e) ** 3

def _wb_derivs(n0, n1, n2, n3, nv1, nv2):
    n3 = np.clip(n3, 1e-10, 1 - 1e-9)
    s = 1 - n3
    Lg = np.log(s)
    g3 = (n3 + s ** 2 * Lg) / (36 * np.pi * n3 ** 2 * s ** 2)
    dh = ((n3 - 2 * s * Lg) * n3 * s - (n3 + s ** 2 * Lg) * 2 * (1 - 2 * n3)) / (n3 ** 3 * s ** 3)
    g3p = dh / (36 * np.pi)
    dp0 = -Lg
    dp1 = n2 / s
    dp2 = n1 / s + (3 * n2 ** 2 - 3 * nv2 ** 2) * g3
    dp3 = n0 / s + (n1 * n2 - nv1 * nv2) / s ** 2 + (n2 ** 3 - 3 * n2 * nv2 ** 2) * g3p
    dpv1 = -nv2 / s
    dpv2 = -nv1 / s - 6 * n2 * nv2 * g3
    return dp0, dp1, dp2, dp3, dpv1, dpv2

def solute_one_body_correlation(profile, weights1, weights2, cross_kernel, dz):
    """c1_2^hs (solvent FMT derivative fields convolved with SOLUTE weights) + c1_2^sw."""
    if np.asarray(profile, dtype=float).ndim != 2 or float(dz) <= 0.0:
        raise ValueError("profile must be a (2, N) array and dz must be positive")
    P = _profile(profile)
    W1 = _weights(weights1); W2 = _weights(weights2)
    Kc = _grid(cross_kernel, "cross_kernel")
    dd = _scalar(dz, "dz", positive=True)
    rho1 = P[1]
    a0, a1, a2, a3, av1, av2 = W1
    b0, b1, b2, b3, bv1, bv2 = W2
    n0 = _pad_conv(rho1, a0, dd); n1 = _pad_conv(rho1, a1, dd)
    n2 = _pad_conv(rho1, a2, dd); n3 = _pad_conv(rho1, a3, dd)
    nv1 = _pad_conv(rho1, av1, dd); nv2 = _pad_conv(rho1, av2, dd)
    D0, D1, D2, D3, Dv1, Dv2 = _wb_derivs(n0, n1, n2, n3, nv1, nv2)
    c1hs = -(_pad_conv(D0, b0, dd) + _pad_conv(D1, b1, dd) + _pad_conv(D2, b2, dd)
             + _pad_conv(D3, b3, dd) + _pad_conv(Dv1, -bv1, dd) + _pad_conv(Dv2, -bv2, dd))
    c1sw = -_pad_conv(rho1, Kc, dd)
    return (c1hs + c1sw).astype(np.float64)

import numpy as np
from scipy.optimize import fsolve
from scipy.signal import fftconvolve


def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _scalar(x, name, positive=False):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError("non-finite " + name)
    if positive and v <= 0.0:
        raise ValueError("non-positive " + name)
    return v

def _grid(x, name):
    a = _f64(x)
    if a.ndim != 1 or a.size < 3 or not np.all(np.isfinite(a)):
        raise ValueError("bad 1D array " + name)
    return a

def _profile(P):
    P = _f64(P)
    if P.ndim != 2 or P.shape[0] != 2 or P.shape[1] < 8 or not np.all(np.isfinite(P)):
        raise ValueError("bad (2,N) profile array")
    return P

def _weights(W):
    W = _f64(W)
    if W.ndim != 2 or W.shape[0] != 6 or W.shape[1] < 1 or not np.all(np.isfinite(W)):
        raise ValueError("bad (6,M) weight array")
    return W

def _pad_conv(f, w, dz):
    hw = len(w) // 2
    fp = np.pad(f, hw, mode="edge")
    return (fftconvolve(fp, w, mode="same") * dz)[hw:hw + len(f)]

def _muhs_ex(r):
    e = np.pi * r / 6.0
    return (8 * e - 9 * e ** 2 + 3 * e ** 3) / (1 - e) ** 3

def _wb_derivs(n0, n1, n2, n3, nv1, nv2):
    n3 = np.clip(n3, 1e-10, 1 - 1e-9)
    s = 1 - n3
    Lg = np.log(s)
    g3 = (n3 + s ** 2 * Lg) / (36 * np.pi * n3 ** 2 * s ** 2)
    dh = ((n3 - 2 * s * Lg) * n3 * s - (n3 + s ** 2 * Lg) * 2 * (1 - 2 * n3)) / (n3 ** 3 * s ** 3)
    g3p = dh / (36 * np.pi)
    dp0 = -Lg
    dp1 = n2 / s
    dp2 = n1 / s + (3 * n2 ** 2 - 3 * nv2 ** 2) * g3
    dp3 = n0 / s + (n1 * n2 - nv1 * nv2) / s ** 2 + (n2 ** 3 - 3 * n2 * nv2 ** 2) * g3p
    dpv1 = -nv2 / s
    dpv2 = -nv1 / s - 6 * n2 * nv2 * g3
    return dp0, dp1, dp2, dp3, dpv1, dpv2

def solute_profile_peak(c1_2):
    """rho2(z)/rho2,0 = exp(c1_2 - c1_2[-1]); return the maximum."""
    if np.asarray(c1_2, dtype=float).ndim != 1 or np.asarray(c1_2).size < 3:
        raise ValueError("c1_2 must be a 1D array of length >= 3")
    c = _grid(c1_2, "c1_2")
    prof = np.exp(c - c[-1])
    return float(prof.max())

import numpy as np
from scipy.optimize import fsolve
from scipy.signal import fftconvolve


def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _scalar(x, name, positive=False):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError("non-finite " + name)
    if positive and v <= 0.0:
        raise ValueError("non-positive " + name)
    return v

def _grid(x, name):
    a = _f64(x)
    if a.ndim != 1 or a.size < 3 or not np.all(np.isfinite(a)):
        raise ValueError("bad 1D array " + name)
    return a

def _profile(P):
    P = _f64(P)
    if P.ndim != 2 or P.shape[0] != 2 or P.shape[1] < 8 or not np.all(np.isfinite(P)):
        raise ValueError("bad (2,N) profile array")
    return P

def _weights(W):
    W = _f64(W)
    if W.ndim != 2 or W.shape[0] != 6 or W.shape[1] < 1 or not np.all(np.isfinite(W)):
        raise ValueError("bad (6,M) weight array")
    return W

def _pad_conv(f, w, dz):
    hw = len(w) // 2
    fp = np.pad(f, hw, mode="edge")
    return (fftconvolve(fp, w, mode="same") * dz)[hw:hw + len(f)]

def _muhs_ex(r):
    e = np.pi * r / 6.0
    return (8 * e - 9 * e ** 2 + 3 * e ** 3) / (1 - e) ** 3

def _wb_derivs(n0, n1, n2, n3, nv1, nv2):
    n3 = np.clip(n3, 1e-10, 1 - 1e-9)
    s = 1 - n3
    Lg = np.log(s)
    g3 = (n3 + s ** 2 * Lg) / (36 * np.pi * n3 ** 2 * s ** 2)
    dh = ((n3 - 2 * s * Lg) * n3 * s - (n3 + s ** 2 * Lg) * 2 * (1 - 2 * n3)) / (n3 ** 3 * s ** 3)
    g3p = dh / (36 * np.pi)
    dp0 = -Lg
    dp1 = n2 / s
    dp2 = n1 / s + (3 * n2 ** 2 - 3 * nv2 ** 2) * g3
    dp3 = n0 / s + (n1 * n2 - nv1 * nv2) / s ** 2 + (n2 ** 3 - 3 * n2 * nv2 ** 2) * g3p
    dpv1 = -nv2 / s
    dpv2 = -nv1 / s - 6 * n2 * nv2 * g3
    return dp0, dp1, dp2, dp3, dpv1, dpv2

def interface_adsorption_audit(beps1, lam1, ratio, beps2, dz, L):
    """Orchestrator: reaches every earlier step through its  twin."""
    be1 = _scalar(beps1, "beps1", positive=True)
    l1 = _scalar(lam1, "lam1", positive=True)
    rt = _scalar(ratio, "ratio", positive=True)
    be2 = _scalar(beps2, "beps2", positive=True)
    dd = _scalar(dz, "dz", positive=True)
    LL = _scalar(L, "L", positive=True)
    cx = sw_bulk_coexistence(be1, l1)                       # step 1
    rho_l, rho_v = float(cx[0]), float(cx[1])
    R1 = 0.5
    sig2 = rt                                                        # sigma1 = 1
    R2 = sig2 / 2.0
    lam2 = 1.0 + (l1 - 1.0) / rt                                    # equal-width rule
    W1 = fmt_weight_functions(R1, dd)                       # step 2
    W2 = fmt_weight_functions(R2, dd)                       # step 2 (solute)
    K1 = sw_meanfield_kernel(be1, l1, dd)                   # step 3
    prof = solvent_density_profile(rho_l, rho_v, W1, K1, be1, l1, dd, LL)   # step 4
    gamma = surface_tension(prof, W1, K1, be1, l1, dd)      # step 5 (audit checkpoint)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("solvent interface failed to converge")
    cr = lorentz_berthelot_cross(1.0, l1, be1, sig2, lam2, be2)   # step 6
    rng12, eps12 = float(cr[0]), float(cr[1])
    Kc = sw_meanfield_kernel(eps12, rng12, dd)             # step 3 (cross)
    c12 = solute_one_body_correlation(prof, W1, W2, Kc, dd)  # step 7
    return solute_profile_peak(c12)                          # step 8
SCICODE_GOLD_EOF
