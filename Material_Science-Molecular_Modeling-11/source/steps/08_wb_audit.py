"""
Runs the whole chain: operator, lowest band, stationary density, generalized scattering function, transport coefficients, cumulants and shape parameters, with the band and the variance slope as checks, and locates the peak skewness and the peak non-Gaussian parameter.

Comparing the transport coefficients from two routes and the shape parameters at a fixed time with their peaks shows how the driven particle's non-Gaussianity depends on the tilt below the critical value.

Returns
-------
A float64 array of shape (13,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def wb_audit(u: float, f_over_u: float, n_max: int) -> np.ndarray:
    r"""u: positive float, the corrugation amplitude. f_over_u: non-negative float, the tilt-to-amplitude ratio $f/u$.
    n_max: integer of at least 12, the Fourier truncation used throughout.

    Uses the reference time $t_{ref} = 1/(4\pi^2 u)$ (in units of $L^2/D$, the curvature time of an untilted well) and
    the zone-edge wave number $q = \pi$. Returns a numpy float64 array of shape $(13,)$: the drift velocity $v$; the
    long-time diffusivity $D_\infty/D$; the mobility $k_BT\mu_\infty/D$; the Einstein ratio $D_\infty/(k_BT\mu_\infty)$;
    the real and imaginary parts of the stationary coefficient $\langle 1|r_{00}\rangle$; the real and imaginary parts
    of $F_{10}(\pi, t_{ref})$; $D(t_{ref})/D$; $\mathrm{Skew}(t_{ref})$; $\alpha_2(t_{ref})$; the peak skewness
    $\max_t\mathrm{Skew}(t)$; and the peak non-Gaussian parameter $\max_t\alpha_2(t)$, both maxima taken over
    $10^{-4} \le t \le 10$ (located on a logarithmic grid of 80 points and refined to convergence). The chain is
    checked on the way: the $\mu = 0$ row of the operator must vanish (probability conservation); the drift velocity
    must agree with the slope of the imaginary part of the lowest band at $q = 0$ and the long-time diffusivity
    with half the curvature of its real part, both within $10^{-5}$ relative (central differences with step $10^{-3}$);
    the stationary coefficients must satisfy $\langle-\mu|r_{00}\rangle = \langle\mu|r_{00}\rangle^*$ and the density
    they build must be non-negative on a grid of 400 points; $F_{11}(\pi, 0)$ must be one within $10^{-10}$;
    $\kappa_1(t_{ref})$ must equal $v\,t_{ref}$ within $10^{-8}$ relative; and $D(t_{ref})$ must agree with a central
    difference of $\kappa_2/2$ (step $10^{-6}t_{ref}$) within $10^{-6}$ relative.

    Raises:
        ValueError: on a non-finite or non-positive u, a negative f_over_u, an n_max below 12, a failed check, or a
            peak that is not interior to the search window.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from math import factorial
from scipy.linalg import expm, eig, solve
from scipy.optimize import minimize_scalar
from scipy.integrate import quad


def _fin(x, name):
    v = float(x)
    if not np.isfinite(v): raise ValueError("bad " + name)
    return v
def _pos(x, name):
    v = _fin(x, name)
    if v <= 0.0: raise ValueError("bad " + name)
    return v
def _nonneg(x, name):
    v = _fin(x, name)
    if v < 0.0: raise ValueError("bad " + name)
    return v
def _int(x, name, low=0):
    v = _fin(x, name)
    if v != int(v) or int(v) < low: raise ValueError("bad " + name)
    return int(v)
def _vec(x, name):
    a = np.asarray(x, dtype=float).ravel()
    if a.size == 0 or not np.all(np.isfinite(a)): raise ValueError("bad " + name)
    return a

def _blocks(u, f, n_max):
    n = 2 * n_max + 1
    idx = np.arange(-n_max, n_max + 1)
    L0 = np.zeros((n, n), dtype=complex); L1 = np.zeros((n, n), dtype=complex)
    for a, mu in enumerate(idx):
        L0[a, a] = -(2.0 * np.pi)**2 * (1j * mu * f + mu * mu)
        L1[a, a] = -(2.0 * np.pi) * (1j * f + 2.0 * mu)
        if a - 1 >= 0:
            L0[a, a - 1] += -(2.0 * np.pi)**2 * (u * mu / 2.0)
            L1[a, a - 1] += -(2.0 * np.pi) * (u / 2.0)
        if a + 1 < n:
            L0[a, a + 1] += (2.0 * np.pi)**2 * (u * mu / 2.0)
            L1[a, a + 1] += (2.0 * np.pi) * (u / 2.0)
    return L0, L1
def _operator(u, f, q, n_max):
    L0, L1 = _blocks(u, f, n_max)
    return L0 + q * L1 - q * q * np.eye(2 * n_max + 1)
def _zero_mode(u, f, n_max):
    L0, _ = _blocks(u, f, n_max)
    n = 2 * n_max + 1; keep = [a for a in range(n) if a != n_max]
    A = L0[np.ix_(keep, keep)]; b = -L0[keep, n_max]
    r = np.zeros(n, dtype=complex); r[n_max] = 1.0; r[keep] = solve(A, b)
    return r
def _spectrum(u, f, q, n_max):
    L = _operator(u, f, q, n_max)
    w, R = eig(L); lam = -w
    order = np.lexsort((lam.imag, lam.real)); lam = lam[order]; R = R[:, order]
    if q == 0.0:
        R[:, 0] = _zero_mode(u, f, n_max)
    Linv = np.linalg.inv(R)
    return lam, R, Linv
def _lowest(u, f, q, n_max):
    L = _operator(u, f, q, n_max)
    lam = -np.linalg.eigvals(L)
    k = np.argmin(lam.real)
    return lam[k]
def _elements(u, f, n_max):
    lam, R, Linv = _spectrum(u, f, 0.0, n_max)
    L0, L1 = _blocks(u, f, n_max)
    idx = np.arange(-n_max, n_max + 1)
    with np.errstate(all="ignore"):
        M = Linv @ L1 @ R
        A = Linv @ np.diag(-(2.0 * np.pi)**2 * 1j * idx) @ R
    if not (np.all(np.isfinite(M)) and np.all(np.isfinite(A))): raise ValueError("non-finite matrix elements")
    return lam, M, A
def _transport(u, f, n_max):
    lam, M, A = _elements(u, f, n_max)
    n = np.arange(1, lam.size)
    v = (1j * M[0, 0]).real
    dinf = (1.0 - np.sum(M[0, n] * M[n, 0] / lam[n])).real
    mob = (1.0 + (1j / (2.0 * np.pi)) * np.sum(M[0, n] * A[n, 0] / lam[n])).real
    return v, dinf, mob
def _diffusivity(u, f, t, n_max):
    lam, M, A = _elements(u, f, n_max)
    n = np.arange(1, lam.size)
    return (1.0 - np.sum((1.0 - np.exp(-lam[n] * t)) / lam[n] * M[0, n] * M[n, 0])).real
def _reduced_moments(u, f, t, n_max, order=4):
    L0, L1 = _blocks(u, f, n_max); n = 2 * n_max + 1
    r = _zero_mode(u, f, n_max)
    drift = L1[n_max] @ r
    L1c = L1 - drift * np.eye(n)
    B = np.zeros(((order + 1) * n, (order + 1) * n), dtype=complex)
    for k in range(order + 1):
        B[k * n:(k + 1) * n, k * n:(k + 1) * n] = L0
        if k < order: B[k * n:(k + 1) * n, (k + 1) * n:(k + 2) * n] = L1c
    with np.errstate(all="ignore"):
        E = expm(B * t)
    if not np.all(np.isfinite(E)): raise ValueError("non-finite propagator")
    out = [1.0 + 0.0j]
    for k in range(1, order + 1):
        out.append((1j)**k * factorial(k) * (E[n_max, k * n:(k + 1) * n] @ r))
    return (1j * drift).real, out
def _cumulants(u, f, t, n_max):
    v, mt = _reduced_moments(u, f, t, n_max)
    mu = [1.0 + 0.0j]
    for m in range(1, 5):
        s = 0.0 + 0.0j
        for k in range(m // 2 + 1):
            s += factorial(m) * t**k / (factorial(m - 2 * k) * factorial(k)) * mt[m - 2 * k]
        mu.append(s)
    m1, m2, m3, m4 = mu[1:]
    scale = 1.0 + max(abs(m2), abs(m3), abs(m4))
    if max(abs(m1), abs(m2.imag), abs(m3.imag), abs(m4.imag)) > 1e-9 * scale: raise ValueError("cumulants are not real")
    ks = np.array([v * t, m2.real, m3.real, m4.real - 3.0 * m2.real**2])
    if not np.all(np.isfinite(ks)): raise ValueError("non-finite cumulants")
    return ks
def _shape(u, f, t, n_max):
    k = _cumulants(u, f, t, n_max)
    return _diffusivity(u, f, t, n_max), k[2] / k[1]**1.5, k[3] / (3.0 * k[1]**2)
def _gisf(u, f, q, mu, nu, t, n_max):
    n = 2 * n_max + 1
    r = _zero_mode(u, f, n_max)
    s = np.zeros(n, dtype=complex)
    for a in range(n):
        b = a + nu
        if 0 <= b < n: s[b] = r[a]
    with np.errstate(all="ignore"):
        P = expm(_operator(u, f, q, n_max) * t)
    return P[mu + n_max] @ s
def _stationary_bessel(u, f, n_max):
    from scipy.special import iv
    m = 6 * n_max + 40
    nus = np.arange(-m, m + 1)
    with np.errstate(all="ignore"):
        B = np.where(nus == 0, (np.expm1(-f * (2.0 * np.pi)) / (-f * (2.0 * np.pi)) if f != 0.0 else 1.0),
                     (np.exp((1j * nus - f) * (2.0 * np.pi)) - 1.0) / ((1j * nus - f) * (2.0 * np.pi)))
    den = np.sum((-1.0)**nus * iv(nus, u)**2 * B)
    mus = np.arange(-n_max, n_max + 1)
    out = np.array([np.sum(iv(mu - nus, -u) * iv(nus, u) * B) for mu in mus]) / den
    return out
def _stratonovich(u, f):
    U = lambda x: u * np.cos((2.0 * np.pi) * x) - (2.0 * np.pi) * f * x
    Im = lambda x: quad(lambda y: np.exp(U(x + y) - U(x)), 0.0, 1.0, limit=200)[0]
    return -np.expm1(-(2.0 * np.pi) * f) / quad(Im, 0.0, 1.0, limit=200)[0]
def _reimann(u, f):
    U = lambda x: u * np.cos((2.0 * np.pi) * x) - (2.0 * np.pi) * f * x
    Ip = lambda x: quad(lambda y: np.exp(U(x) - U(x + y)), 0.0, 1.0, limit=200)[0]
    Im = lambda x: quad(lambda y: np.exp(U(x - y) - U(x)), 0.0, 1.0, limit=200)[0]
    num = quad(lambda x: Ip(x)**2 * Im(x), 0.0, 1.0, limit=200)[0]
    den = quad(Ip, 0.0, 1.0, limit=200)[0]
    return num / den**3
def _peak(fun, lo, hi, npts):
    grid = np.linspace(np.log10(lo), np.log10(hi), npts)
    vals = np.array([fun(10.0**x) for x in grid])
    i = int(np.argmax(vals))
    if i == 0 or i == npts - 1: raise ValueError("the peak is not interior to the search window")
    res = minimize_scalar(lambda x: -fun(10.0**x), bracket=(grid[i - 1], grid[i], grid[i + 1]), tol=1e-12)
    return -res.fun, 10.0**res.x

def _oracle_wb_audit(u: float, f_over_u: float, n_max: int) -> np.ndarray:
    uu = _pos(u, "u"); fu = _nonneg(f_over_u, "f_over_u"); nm = _int(n_max, "n_max", 12)
    ff = fu * uu
    op = _oracle_wb_operator(uu, ff, 0.0, nm)
    if np.max(np.abs(op[:, nm, :])) > 1e-12: raise ValueError("the zero row of the operator must vanish")
    tr = _oracle_wb_transport(uu, ff, nm)
    h = 1e-3
    bd = _oracle_wb_bands(uu, ff, np.array([-h, 0.0, h]), nm)
    vb = (bd[1, 2] - bd[1, 0]) / (2.0 * h); db = 0.5 * (bd[0, 2] + bd[0, 0] - 2.0 * bd[0, 1]) / h**2
    if abs(vb - tr[0]) > 1e-5 * max(1.0, abs(tr[0])) or abs(db - tr[1]) > 1e-5 * max(1.0, abs(tr[1])): raise ValueError("the lowest band disagrees with the transport coefficients")
    st = _oracle_wb_stationary(uu, ff, nm)
    r = st[0] + 1j * st[1]
    if np.max(np.abs(r[::-1] - np.conj(r))) > 1e-10: raise ValueError("the stationary coefficients must be conjugate symmetric")
    xs = np.arange(400) / 400.0
    mus = np.arange(-nm, nm + 1)
    with np.errstate(all="ignore"):
        pst = np.real(np.exp(2j * np.pi * np.outer(xs, mus)) @ r)
    if np.min(pst) < -1e-9: raise ValueError("the stationary density must be non-negative")
    tref = 1.0 / ((2.0 * np.pi)**2 * uu)
    f110 = _oracle_wb_generalized_isf(uu, ff, np.pi, 1, 1, 0.0, nm)
    if abs(f110[0] - 1.0) > 1e-10 or abs(f110[1]) > 1e-10: raise ValueError("the diagonal scattering function must start at one")
    f10 = _oracle_wb_generalized_isf(uu, ff, np.pi, 1, 0, tref, nm)
    k = _oracle_wb_cumulants(uu, ff, tref, nm)
    if abs(k[0] - tr[0] * tref) > 1e-8 * max(1.0, abs(tr[0] * tref)): raise ValueError("the first cumulant must be the drift times the time")
    sh = _oracle_wb_shape(uu, ff, tref, nm)
    hh = 1e-6 * tref
    dfd = 0.5 * (_oracle_wb_cumulants(uu, ff, tref + hh, nm)[1] - _oracle_wb_cumulants(uu, ff, tref - hh, nm)[1]) / (2.0 * hh)
    if abs(dfd - sh[0]) > 1e-6 * max(1.0, abs(sh[0])): raise ValueError("the time-dependent diffusivity disagrees with the variance slope")
    skmax, tsk = _peak(lambda t: _oracle_wb_shape(uu, ff, t, nm)[1], 1e-4, 10.0, 80)
    a2max, ta2 = _peak(lambda t: _oracle_wb_shape(uu, ff, t, nm)[2], 1e-4, 10.0, 80)
    return np.array([tr[0], tr[1], tr[2], tr[3], st[0][nm + 1], st[1][nm + 1], f10[0], f10[1], sh[0], sh[1], sh[2], skmax, a2max], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\n","call":"wb_audit(10.0,0.9,20)","gold_call":"_oracle_wb_audit(10.0,0.9,20)"},
        {"setup":"import numpy as np\n","call":"wb_audit(10.0,0.8,20)","gold_call":"_oracle_wb_audit(10.0,0.8,20)"},
        {"setup":"import numpy as np\n","call":"wb_audit(5.0,0.9,16)","gold_call":"_oracle_wb_audit(5.0,0.9,16)"},
        {"setup":"import numpy as np\n# boundary: the critical tilt, where the minima and maxima of the potential merge\n","call":"wb_audit(10.0,1.0,20)","gold_call":"_oracle_wb_audit(10.0,1.0,20)"},
        {"setup":"import numpy as np\n# invalid input: a truncation below 12 must raise ValueError\ndef _probe(fn, *args):\n    try:\n        fn(*args)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n","call":"_probe(wb_audit,10.0,0.9,8)","gold_call":"_probe(_oracle_wb_audit,10.0,0.9,8)"},
    ]
