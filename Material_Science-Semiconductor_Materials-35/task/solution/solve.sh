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

def dc_current(V: np.ndarray, iv_params: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    p = np.asarray(iv_params, dtype=float).ravel()
    if p.size != 7 or not p[2] > 0.0 or not p[5] > 0.0:
        raise ValueError("iv_params must be [A1, V1, w1, A2, V2, w2, B] with w1 > 0 and w2 > 0")
    A1, V1, w1, A2, V2, w2, B = p
    V = np.asarray(V, dtype=float)
    return A1*np.exp(-((V - V1)/w1)**2) + A2*np.exp(-((V - V2)/w2)**2) + B*V**3

import numpy as np
from scipy.optimize import brentq, minimize_scalar

def characteristic_scales(iv_params: np.ndarray) -> np.ndarray:
    """Reference implementation. Chains step 01 through the same parameterisation."""
    p = np.asarray(iv_params, dtype=float).ravel()
    if p.size != 7 or not p[2] > 0.0 or not p[5] > 0.0:
        raise ValueError("iv_params must be [A1, V1, w1, A2, V2, w2, B] with w1 > 0 and w2 > 0")
    A1, V1, w1, A2, V2, w2, B = p

    slope = lambda V: (-2.0*(np.asarray(V, dtype=float) - V1)/w1**2
                       * A1*np.exp(-((np.asarray(V, dtype=float) - V1)/w1)**2)
                       - 2.0*(np.asarray(V, dtype=float) - V2)/w2**2
                       * A2*np.exp(-((np.asarray(V, dtype=float) - V2)/w2)**2)
                       + 3.0*B*np.asarray(V, dtype=float)**2)
    Vg = np.linspace(1e-6, 2.0, 400001)
    s = np.sign(slope(Vg))
    down = np.where((s[:-1] > 0) & (s[1:] <= 0))[0]
    if down.size == 0:
        raise ValueError("the characteristic has no current peak for 0 < V <= 2 V")
    i = down[0]
    up = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    up = up[up >= i]
    if up.size == 0:
        raise ValueError("the characteristic has no valley after its peak for V <= 2 V")
    j = up[0]
    f = lambda v: float(slope(v))
    Vp = brentq(f, Vg[i], Vg[i + 1], xtol=1e-15, rtol=1e-15)
    Vv = brentq(f, Vg[j], Vg[j + 1], xtol=1e-15, rtol=1e-15)
    Vin = np.linspace(Vp, Vv, 20001)
    k = int(np.argmin(slope(Vin)))
    lo, hi = Vin[max(k - 1, 0)], Vin[min(k + 1, Vin.size - 1)]
    res = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-13})
    return np.array([Vp, Vv, Vv - Vp, -float(res.fun)])

import numpy as np

def harmonic_coefficients(Vdc: float, a0: float, dV: float, iv_params: np.ndarray, nmax: int) -> np.ndarray:
    """Reference implementation. Chains step 01."""
    if a0 < 0.0 or dV <= 0.0 or int(nmax) < 1:
        raise ValueError("need a0 >= 0, dV > 0 and nmax >= 1")
    nmax = int(nmax)
    M = 2048
    t = (np.arange(M) + 0.5)*np.pi/M
    f = dc_current(Vdc + a0*dV*np.cos(t), iv_params) - dc_current(np.array(Vdc), iv_params)
    c = np.empty(nmax + 1)
    c[0] = np.mean(f)
    for n in range(1, nmax + 1):
        c[n] = 2.0*np.mean(f*np.cos(n*t))
    return c

import numpy as np
from scipy.optimize import brentq

def oscillation_amplitude(Vdc: float, Gl: float, dV: float, iv_params: np.ndarray) -> float:
    """Reference implementation. Chains step 03."""
    if Gl < 0.0 or dV <= 0.0:
        raise ValueError("need Gl >= 0 and dV > 0")

    def _balance(a):
        return Gl + harmonic_coefficients(Vdc, a, dV, iv_params, 1)[1]/(a*dV)

    xs = np.linspace(1e-3, 3.0, 1500)
    vs = np.array([_balance(x) for x in xs])
    roots = [brentq(_balance, xs[k], xs[k + 1], xtol=1e-14, rtol=1e-15)
             for k in range(xs.size - 1) if vs[k]*vs[k + 1] < 0.0]
    return float(max(roots)) if roots else 0.0

import numpy as np
from scipy.optimize import brentq, minimize_scalar

def oscillation_edge(Gl: float, dV: float, iv_params: np.ndarray) -> np.ndarray:
    """Reference implementation. Chains step 03."""
    if Gl <= 0.0 or dV <= 0.0:
        raise ValueError("need Gl > 0 and dV > 0")

    def _bias_for(a):
        """Largest bias at which amplitude a satisfies the gain balance, or nan."""
        res = lambda V: Gl + harmonic_coefficients(V, a, dV, iv_params, 1)[1]/(a*dV)
        Vs = np.linspace(0.05, 2.0, 160)
        vals = np.array([res(v) for v in Vs])
        hits = [k for k in range(Vs.size - 1) if vals[k]*vals[k + 1] < 0.0]
        if not hits:
            return np.nan
        k = hits[-1]
        return brentq(res, Vs[k], Vs[k + 1], xtol=1e-15, rtol=1e-15)

    xs = np.linspace(0.02, 2.5, 60)
    vs = np.array([_bias_for(x) for x in xs])
    if not np.any(np.isfinite(vs)):
        raise ValueError("no bias in 0 < V <= 2 V supports an oscillation at this load")
    j = int(np.nanargmax(vs))
    lo = xs[max(j - 1, 0)]
    hi = xs[min(j + 1, xs.size - 1)]
    res = minimize_scalar(lambda a: -_bias_for(a), bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-10})
    a_edge = float(res.x)
    # one Newton step on the tangency condition dV_bias/da = 0 for full precision
    h = 1e-5
    for _ in range(40):
        f1 = (_bias_for(a_edge + h) - _bias_for(a_edge - h))/(2.0*h)
        f2 = (_bias_for(a_edge + h) - 2.0*_bias_for(a_edge) + _bias_for(a_edge - h))/h**2
        step = f1/f2
        a_edge -= step
        if abs(step) < 1e-13:
            break
    return np.array([_bias_for(a_edge), a_edge])

import numpy as np
from scipy.optimize import brentq

def rectified_amplitude(Vdc: float, J_dc: float, dV: float, iv_params: np.ndarray) -> float:
    """Reference implementation. Chains steps 01 and 03."""
    if dV <= 0.0:
        raise ValueError("dV must be positive")
    I0 = float(dc_current(np.array(Vdc), iv_params))

    def _mismatch(a):
        return I0 + harmonic_coefficients(Vdc, a, dV, iv_params, 1)[0] - J_dc

    xs = np.linspace(1e-3, 3.0, 1500)
    vs = np.array([_mismatch(x) for x in xs])
    hits = [k for k in range(xs.size - 1) if vs[k]*vs[k + 1] <= 0.0]
    if not hits:
        raise ValueError("no amplitude in 0 < a0 <= 3 reproduces the measured dc current")
    k = hits[0]
    return float(brentq(_mismatch, xs[k], xs[k + 1], xtol=1e-14, rtol=1e-15))

import numpy as np

def first_order_coefficients(Vdc: float, a0: float, dV: float, G0: float, iv_params: np.ndarray, nmax: int) -> np.ndarray:
    """Reference implementation. Chains step 03."""
    if a0 <= 0.0 or dV <= 0.0 or G0 <= 0.0 or int(nmax) < 2:
        raise ValueError("need a0 > 0, dV > 0, G0 > 0 and nmax >= 2")
    nmax = int(nmax)
    c = harmonic_coefficients(Vdc, a0, dV, iv_params, nmax)
    n = np.arange(2, nmax + 1, dtype=float)
    cn = c[2:]
    w = n**2/(n**2 - 1.0)
    b1 = np.sum(w*cn)/(dV*G0)
    bn = -(n/(n**2 - 1.0))*cn/(dV*G0)
    kappa1 = -np.sum(w*cn**2)/(2.0*(a0*dV*G0)**2)
    return np.concatenate(([b1], bn, [kappa1]))

import numpy as np

def large_signal_capacitance(Vdc: float, a0: float, dV: float, cap_params: np.ndarray) -> float:
    """Reference implementation."""
    q = np.asarray(cap_params, dtype=float).ravel()
    if a0 < 0.0 or dV <= 0.0 or q.size != 4 or not q[3] > 0.0:
        raise ValueError("need a0 >= 0, dV > 0 and cap_params = [Cb, Cq, Vq, wq] with wq > 0")
    Cb, Cq, Vq, wq = q
    M = 2048
    t = (np.arange(M) + 0.5)*np.pi/M
    C0 = Cb + Cq*np.exp(-((Vdc + a0*dV*np.cos(t) - Vq)/wq)**2)
    return float(2.0*np.mean(C0*np.sin(t)**2))

import numpy as np

def bias_frequency(Cr: float, ind: float, c_ls: float, coeffs: np.ndarray, G0: float) -> np.ndarray:
    """Reference implementation."""
    co = np.asarray(coeffs, dtype=float).ravel()
    if Cr < 0.0 or c_ls <= 0.0 or ind <= 0.0 or G0 <= 0.0 or co.size < 3:
        raise ValueError("need Cr >= 0, c_ls > 0, ind > 0, G0 > 0 and at least three coefficients")
    C = (Cr + c_ls)*1e-15
    w0 = 1.0/np.sqrt(ind*1e-12*C)
    eps = G0*1e-3/(C*w0)
    kappa1 = co[-1]
    f = w0*(1.0 + eps**2*kappa1)/(2.0*np.pi)*1e-9
    crit = max(abs(eps*co[0]), float(np.max(np.abs(eps*co[1:-1]))), abs(eps**2*kappa1))
    return np.array([f, eps, crit])

import numpy as np
from scipy.optimize import brentq

def resonator_capacitance(iv_params: np.ndarray, cap_fixed: np.ndarray, bias: np.ndarray, J_dc: float, freq: np.ndarray, nmax: int) -> float:
    """Reference implementation. Chains steps 02 to 09."""
    V = np.asarray(bias, dtype=float).ravel()
    F = np.asarray(freq, dtype=float).ravel()
    q = np.asarray(cap_fixed, dtype=float).ravel()
    if V.size != 2 or F.size != 3 or np.any(F <= 0.0) or int(nmax) < 2 or q.size != 3:
        raise ValueError("need two biases, three positive frequencies, cap_fixed = [Cb, Vq, wq] and nmax >= 2")
    nmax = int(nmax)
    dV, G0 = characteristic_scales(iv_params)[2:]
    a_first = rectified_amplitude(V[0], J_dc, dV, iv_params)
    Gl = -harmonic_coefficients(V[0], a_first, dV, iv_params, 1)[1]/(a_first*dV)
    if Gl < 0.0:
        raise ValueError("the measured dc current implies a negative load conductance")
    edge = oscillation_edge(Gl, dV, iv_params)
    V = np.array([V[0], V[1], edge[0]])
    amps = [a_first, oscillation_amplitude(V[1], Gl, dV, iv_params), float(edge[1])]
    if min(amps) <= 0.0:
        raise ValueError("a bias supports no oscillation at this load")
    coeffs = [first_order_coefficients(v, a, dV, G0, iv_params, nmax) for v, a in zip(V, amps)]

    def _caps(Cq):
        pars = np.array([q[0], Cq, q[1], q[2]])
        return [large_signal_capacitance(v, a, dV, pars) for v, a in zip(V, amps)]

    def _ind(Cr, cls0):
        g = lambda u: bias_frequency(Cr, np.exp(u), cls0, coeffs[0], G0)[0] - F[0]
        return float(np.exp(brentq(g, np.log(1e-6), np.log(1e6), xtol=1e-14, rtol=1e-15)))

    def _pair(Cq):
        """(Cr, L) reproducing the first and last frequency for this Cq, or None."""
        cls = _caps(Cq)
        g = lambda Cr: bias_frequency(Cr, _ind(Cr, cls[0]), cls[2], coeffs[2], G0)[0] - F[2]
        if g(0.0)*g(1e4) > 0.0:
            return None
        Cr = float(brentq(g, 0.0, 1e4, xtol=1e-12, rtol=1e-15))
        return Cr, _ind(Cr, cls[0]), cls

    def _middle(Cq):
        got = _pair(Cq)
        if got is None:
            return np.nan
        Cr, ind, cls = got
        return bias_frequency(Cr, ind, cls[1], coeffs[1], G0)[0] - F[1]

    xs = np.linspace(0.02, 30.0, 60)
    vs = np.array([_middle(x) for x in xs])
    hits = [k for k in range(xs.size - 1)
            if np.isfinite(vs[k]) and np.isfinite(vs[k + 1]) and vs[k]*vs[k + 1] <= 0.0]
    if not hits:
        raise ValueError("no Cq >= 0 with C_r >= 0 and L > 0 reproduces the three frequencies")
    k = hits[0]
    Cq = float(brentq(_middle, xs[k], xs[k + 1], xtol=1e-12, rtol=1e-15))
    Cr, ind, cls = _pair(Cq)
    crit = max(bias_frequency(Cr, ind, c, co, G0)[2] for c, co in zip(cls, coeffs))
    if crit > 0.3:
        raise ValueError("the expansion is not applicable at the recovered resonator")
    return Cr
SCICODE_GOLD_EOF
