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


def _check_lattice(lattice):
    """Validate the lattice array; return (De, Dh, density, cs, ae, ah) as floats."""
    arr = np.asarray(lattice)
    if not isinstance(lattice, np.ndarray) or arr.shape != (6,) or arr.dtype.kind not in "iuf":
        raise ValueError("lattice must be a numpy array of six real numbers")
    arr = arr.astype(float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("lattice entries must be finite")
    De, Dh, density, cs, ae, ah = (float(x) for x in arr)
    if density <= 0.0 or cs <= 0.0 or ae <= 0.0 or ah <= 0.0:
        raise ValueError("density, sound velocity and Gaussian widths must be positive")
    return De, Dh, density, cs, ae, ah


def lattice_spectral_density(omega: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    w = np.asarray(omega, dtype=float)
    if w.ndim != 1 or not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("omega must be a one-dimensional array of finite values >= 0")
    De, Dh, density, cs, ae, ah = _check_lattice(lattice)
    hbar_si = 1.054571817e-34
    ev = 1.602176634e-19
    wsi = w * 1.0e12
    ae_m, ah_m = ae * 1.0e-9, ah * 1.0e-9
    form = (De * ev * np.exp(-wsi ** 2 * ae_m ** 2 / (4.0 * cs ** 2))
            - Dh * ev * np.exp(-wsi ** 2 * ah_m ** 2 / (4.0 * cs ** 2)))
    return wsi ** 3 / (4.0 * np.pi ** 2 * density * hbar_si * cs ** 5) * form ** 2 * 1.0e-12

import numpy as np


def _omega_nodes(lattice, tmax):
    """Composite 20-point Gauss-Legendre nodes and weights on [0, W] covering the phonon band."""
    De, Dh, density, cs, ae, ah = _check_lattice(lattice)
    wc = 2.0 * cs / (min(ae, ah) * 1.0e3)          # 2 c_s / a in rad/ps
    top = 8.0 * wc
    panels = max(80, int(np.ceil(top * max(tmax, 1.0) / 1.5)))
    x, wt = np.polynomial.legendre.leggauss(20)
    edges = np.linspace(0.0, top, panels + 1)
    half = 0.5 * (edges[1] - edges[0])
    nodes = (edges[:-1, None] + half * (x[None, :] + 1.0)).ravel()
    weights = np.tile(half * wt, panels)
    return nodes, weights


def _check_temperature(T):
    """Return T as a float, raising ValueError unless it is finite and > 0 (kelvin)."""
    if isinstance(T, bool) or not isinstance(T, (int, float, np.integer, np.floating)):
        raise ValueError("T must be a real number")
    T = float(T)
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T must be finite and positive")
    return T


def phonon_propagator(t: np.ndarray, T: float, lattice: np.ndarray) -> np.ndarray:
    tt = np.asarray(t, dtype=float)
    if tt.ndim != 1 or not np.all(np.isfinite(tt)) or np.any(tt < 0.0):
        raise ValueError("t must be a one-dimensional array of finite values >= 0")
    T = _check_temperature(T)
    hbar, kb = 0.6582119569, 0.08617333262
    w, wt = _omega_nodes(lattice, float(tt.max()) if tt.size else 0.0)
    jw = lattice_spectral_density(w, lattice)
    amp = jw / w ** 2 * wt
    coth = 1.0 / np.tanh(hbar * w / (2.0 * kb * T))
    phase = np.outer(tt, w)
    re = np.cos(phase) @ (amp * coth)
    im = -(np.sin(phase) @ amp)
    return np.array([re, im])

import numpy as np


def _polaron_times(T, lattice):
    """Uniform Simpson time grid (ps) long enough for C_u to have decayed."""
    De, Dh, density, cs, ae, ah = _check_lattice(lattice)
    hbar, kb = 0.6582119569, 0.08617333262
    tmax = max(20.0, 10.0 * hbar / (kb * T), 40.0 * max(ae, ah) * 1.0e3 / cs)
    n = int(np.ceil(tmax / 0.005))
    n += n % 2
    t = np.linspace(0.0, tmax, n + 1)
    wts = np.full(t.size, 2.0)
    wts[1:-1:2] = 4.0
    wts[0] = wts[-1] = 1.0
    return t, wts * (t[1] - t[0]) / 3.0


def polaron_response_functions(omega: np.ndarray, T: float, lattice: np.ndarray) -> np.ndarray:
    om = np.asarray(omega, dtype=float)
    if om.ndim != 1 or not np.all(np.isfinite(om)):
        raise ValueError("omega must be a one-dimensional array of finite values")
    T = _check_temperature(T)
    t, wts = _polaron_times(T, lattice)
    parts = [phonon_propagator(t[i:i + 2000], T, lattice) for i in range(0, t.size, 2000)]
    ph = np.concatenate(parts, axis=1)
    phi = ph[0] + 1j * ph[1]
    b2 = np.exp(-phi[0].real)
    cx = 0.5 * b2 * (np.exp(phi) + np.exp(-phi) - 2.0)
    cy = 0.5 * b2 * (np.exp(phi) - np.exp(-phi))
    kern = np.exp(1j * np.outer(om, t))
    kx = kern @ (cx * wts)
    ky = kern @ (cy * wts)
    return np.array([kx.real, kx.imag, ky.real, ky.imag])

import numpy as np


def _check_dot(dot, need_loss):
    """Validate dot = [E_B, delta, g, gamma]; return the four floats."""
    arr = np.asarray(dot)
    if not isinstance(dot, np.ndarray) or arr.shape != (4,) or arr.dtype.kind not in "iuf":
        raise ValueError("dot must be a numpy array of four real numbers")
    arr = arr.astype(float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("dot entries must be finite")
    eb, dfs, g, gamma = (float(x) for x in arr)
    if g < 0.0 or gamma < 0.0 or (need_loss and (gamma <= 0.0 or g <= 0.0)):
        raise ValueError("g and gamma must be >= 0, and both > 0 where the emission must complete")
    return eb, dfs, g, gamma


def _cavity_space():
    """States (dot, n_H, n_V) of the 13-state space, the coupled pairs and the photon operators."""
    states = [("G", 0, 0), ("XH", 0, 0), ("XV", 0, 0), ("G", 1, 0), ("G", 0, 1),
              ("XX", 0, 0), ("XH", 1, 0), ("XV", 0, 1), ("G", 2, 0), ("G", 0, 2),
              ("XH", 0, 1), ("XV", 1, 0), ("G", 1, 1)]
    index = {s: i for i, s in enumerate(states)}
    pairs = []                      # (upper index, lower index, sqrt(photon number after emission))
    for s, i in index.items():
        dot, nh, nv = s
        for mode, lower in (("H", {"XX": "XH", "XH": "G"}), ("V", {"XX": "XV", "XV": "G"})):
            if dot not in lower:
                continue
            nh2, nv2 = nh + (mode == "H"), nv + (mode == "V")
            j = index.get((lower[dot], nh2, nv2))
            if j is not None:
                pairs.append((i, j, np.sqrt(nh2 if mode == "H" else nv2)))
    ah = np.zeros((13, 13))
    av = np.zeros((13, 13))
    for s, i in index.items():
        dot, nh, nv = s
        if nh:
            ah[index[(dot, nh - 1, nv)], i] = np.sqrt(nh)
        if nv:
            av[index[(dot, nh, nv - 1)], i] = np.sqrt(nv)
    return states, pairs, ah, av


def polaron_master_equation_rhs(rho: np.ndarray, T: float, dot: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    r = np.asarray(rho)
    if r.dtype.kind not in "iufc" or r.ndim not in (2, 3) or r.shape[-2:] != (13, 13):
        raise ValueError("rho must be a numeric array of shape (13, 13) or (m, 13, 13)")
    if not np.all(np.isfinite(r)):
        raise ValueError("rho entries must be finite")
    single = r.ndim == 2
    r = r.astype(complex).reshape((-1, 13, 13))
    T = _check_temperature(T)
    eb, dfs, g, gamma = _check_dot(dot, need_loss=False)
    hbar = 0.6582119569
    bmean = float(np.exp(-0.5 * phonon_propagator(np.array([0.0]), T, lattice)[0, 0]))
    states, pairs, ah, av = _cavity_space()
    level = {"G": 0.0, "XX": 0.0, "XH": eb / 2 + dfs / 2, "XV": eb / 2 - dfs / 2}
    ham = np.diag([level[s[0]] / hbar for s in states]).astype(complex)
    xx = np.zeros((13, 13), dtype=complex)
    xy = np.zeros((13, 13), dtype=complex)
    for i, j, amp in pairs:
        c = g * amp / hbar
        ham[i, j] += bmean * c
        ham[j, i] += bmean * c
        xx[i, j] += c
        xx[j, i] += c
        xy[i, j] += 1j * c
        xy[j, i] -= 1j * c
    evals, vecs = np.linalg.eigh(ham)
    dw = evals[None, :] - evals[:, None]          # element (i, j): omega_j - omega_i
    kap = polaron_response_functions(dw.ravel(), T, lattice)
    kx = (kap[0] + 1j * kap[1]).reshape(13, 13)
    ky = (kap[2] + 1j * kap[3]).reshape(13, 13)
    out = -1j * (ham @ r - r @ ham)
    for xop, k in ((xx, kx), (xy, ky)):
        xe = vecs.conj().T @ xop @ vecs
        dop = vecs @ (2.0 * k.real * xe) @ vecs.conj().T
        sop = vecs @ (k.imag * xe) @ vecs.conj().T
        a = dop @ r - r @ dop.conj().T
        b = sop @ r + r @ sop.conj().T
        out = out - 0.5 * (xop @ a - a @ xop) - 1j * (xop @ b - b @ xop)
    for op in (ah, av):
        num = op.T @ op
        out = out + gamma * (op @ r @ op.T - 0.5 * (num @ r + r @ num))
    res = np.stack([out.real, out.imag], axis=1)
    return res[0] if single else res

import numpy as np


def integrated_pair_correlations(T: float, dot: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    T = _check_temperature(T)
    _check_dot(dot, need_loss=True)
    basis = np.zeros((169, 13, 13), dtype=complex)
    for k in range(169):
        basis[k].flat[k] = 1.0
    cols = polaron_master_equation_rhs(basis, T, dot, lattice)
    gen = (cols[:, 0] + 1j * cols[:, 1]).reshape(169, 169).T
    # |G,0,0><G,0,0| (vector index 0) is the stationary state; it carries no two-photon
    # correlation, so drop it and solve for the time integral of everything else
    keep = np.arange(1, 169)
    start = np.zeros(169, dtype=complex)
    start[5 * 13 + 5] = 1.0
    tint = np.zeros(169, dtype=complex)
    tint[keep] = np.linalg.solve(gen[np.ix_(keep, keep)], -start[keep])
    tint = tint.reshape(13, 13)
    ghh = 2.0 * tint[8, 8].real
    gvv = 2.0 * tint[9, 9].real
    ghv = 2.0 * tint[9, 8]
    return np.array([ghh, gvv, ghv.real, ghv.imag])

import numpy as np


def integrated_concurrence(T: float, dot: np.ndarray, lattice: np.ndarray) -> float:
    ghh, gvv, ghv_re, ghv_im = integrated_pair_correlations(T, dot, lattice)
    return float(2.0 * np.hypot(ghv_re, ghv_im) / (ghh + gvv))

import numpy as np
from scipy.optimize import brentq


def _check_target(c_target, t_lo, t_hi):
    """Validate the target concurrence and the temperature bracket."""
    vals = []
    for name, v in (("C_target", c_target), ("T_lo", t_lo), ("T_hi", t_hi)):
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise ValueError("%s must be a real number" % name)
        v = float(v)
        if not np.isfinite(v):
            raise ValueError("%s must be finite" % name)
        vals.append(v)
    c_target, t_lo, t_hi = vals
    if not 0.0 < c_target < 1.0:
        raise ValueError("C_target must lie strictly between 0 and 1")
    if not 0.0 < t_lo < t_hi:
        raise ValueError("the bracket must satisfy 0 < T_lo < T_hi")
    return c_target, t_lo, t_hi


def _concurrence_excess(temp, c_target, dot, lattice):
    """C_bar(temp) - c_target, evaluated with the reference pipeline."""
    return integrated_concurrence(temp, dot, lattice) - c_target


def entanglement_threshold_temperature(C_target: float, T_lo: float, T_hi: float, dot: np.ndarray,
                                               lattice: np.ndarray) -> float:
    c_target, t_lo, t_hi = _check_target(C_target, T_lo, T_hi)
    _check_dot(dot, need_loss=True)
    _check_lattice(lattice)

    f_lo = _concurrence_excess(t_lo, c_target, dot, lattice)
    f_hi = _concurrence_excess(t_hi, c_target, dot, lattice)
    if not (f_lo > 0.0 > f_hi):
        raise ValueError("C_target is not bracketed: need C_bar(T_lo) > C_target > C_bar(T_hi)")
    return float(brentq(_concurrence_excess, t_lo, t_hi, args=(c_target, dot, lattice),
                        xtol=1e-9, rtol=1e-13, maxiter=200))
SCICODE_GOLD_EOF
