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
from scipy.special import kve, k0, k1, j0, j1
from scipy.optimize import brentq


def _check_state(Gamma, rho):
    Gamma = float(Gamma); rho = float(rho)
    if not (Gamma > 0.0) or not np.isfinite(Gamma):
        raise ValueError("Gamma must be positive")
    if not (rho > 0.0) or not np.isfinite(rho):
        raise ValueError("rho must be positive")
    return Gamma, rho


def _check_sigma(sigma):
    sigma = float(sigma)
    if not (sigma >= 0.0) or not np.isfinite(sigma):
        raise ValueError("sigma must be non-negative")
    return sigma


def _S(q, sigma):
    x = (sigma * q) ** 2
    return 1.0 + x + x ** 2 + x ** 3 + x ** 4


def _dS(q, sigma):
    q2 = q * q
    return 2.0 * sigma * q2 + 4.0 * sigma ** 3 * q2 ** 2 + 6.0 * sigma ** 5 * q2 ** 3 + 8.0 * sigma ** 7 * q2 ** 4


def filter_kernels(q: "np.ndarray", sigma: float, Gamma: float, rho: float) -> "np.ndarray":
    sigma = _check_sigma(sigma); Gamma, rho = _check_state(Gamma, rho)
    q = np.atleast_1d(np.asarray(q, dtype=float))
    if q.ndim != 1 or np.any(q <= 0.0) or not np.all(np.isfinite(q)):
        raise ValueError("q must be positive")
    kap2 = 2.0 * np.pi * Gamma * rho
    S = _S(q, sigma)
    vl = 2.0 * np.pi * Gamma / (q * q * S)
    Gl = 2.0 * np.pi * Gamma / (q * q * S + kap2)
    dvl = -2.0 * np.pi * Gamma * _dS(q, sigma) / (q * q * S * S)
    return np.vstack([S, vl, Gl, dvl])

import numpy as np
from scipy.special import kve, k0, k1, j0, j1
from scipy.optimize import brentq


def _check_state(Gamma, rho):
    Gamma = float(Gamma); rho = float(rho)
    if not (Gamma > 0.0) or not np.isfinite(Gamma):
        raise ValueError("Gamma must be positive")
    if not (rho > 0.0) or not np.isfinite(rho):
        raise ValueError("rho must be positive")
    return Gamma, rho


def _check_sigma(sigma):
    sigma = float(sigma)
    if not (sigma >= 0.0) or not np.isfinite(sigma):
        raise ValueError("sigma must be non-negative")
    return sigma


def _kv(n, z):
    z = np.asarray(z, dtype=complex)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        out = kve(n, z) * np.exp(-z)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _gl_partial_fractions(sigma, kap2):
    """Poles of 1/(x S(x) + kap2) in x = q^2: residues B_m and b_m = sqrt(-x_m) with Re b_m > 0."""
    if sigma == 0.0:
        return np.array([1.0 + 0j]), np.array([np.sqrt(kap2) + 0j])
    coef = np.array([kap2, 1.0, sigma ** 2, sigma ** 4, sigma ** 6, sigma ** 8], dtype=complex)
    x = np.polynomial.polynomial.polyroots(coef)
    B = 1.0 / np.polynomial.polynomial.polyval(x, np.polynomial.polynomial.polyder(coef))
    b = np.sqrt(-x)
    b = np.where(b.real < 0.0, -b, b)
    return B, b


def _gl_closed(u, sigma, Gamma, kap2):
    """Gl(u) = Gamma int_0^inf q J0(qu)/(q^2 S + kap2) dq = Gamma sum_m B_m K0(b_m u)."""
    B, b = _gl_partial_fractions(sigma, kap2)
    u = np.asarray(u, dtype=float)
    z = np.multiply.outer(u, b)
    out = Gamma * np.real(np.sum(B * _kv(0, z), axis=-1))
    zero = (u == 0.0)
    if np.any(zero):
        out = np.where(zero, -Gamma * np.real(np.sum(B * np.log(b))), out)
    return out


def long_range_kernel(u: "np.ndarray", sigma: float, Gamma: float, rho: float) -> "np.ndarray":
    sigma = _check_sigma(sigma); Gamma, rho = _check_state(Gamma, rho)
    u = np.atleast_1d(np.asarray(u, dtype=float))
    if u.ndim != 1 or np.any(u < 0.0) or not np.all(np.isfinite(u)):
        raise ValueError("u must be non-negative")
    kap2 = 2.0 * np.pi * Gamma * rho
    return _gl_closed(u, sigma, Gamma, kap2)

import numpy as np
from scipy.special import kve, k0, k1, j0, j1
from scipy.optimize import brentq


def _check_state(Gamma, rho):
    Gamma = float(Gamma); rho = float(rho)
    if not (Gamma > 0.0) or not np.isfinite(Gamma):
        raise ValueError("Gamma must be positive")
    if not (rho > 0.0) or not np.isfinite(rho):
        raise ValueError("rho must be positive")
    return Gamma, rho


def _check_sigma(sigma):
    sigma = float(sigma)
    if not (sigma >= 0.0) or not np.isfinite(sigma):
        raise ValueError("sigma must be non-negative")
    return sigma


def _kv(n, z):
    z = np.asarray(z, dtype=complex)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        out = kve(n, z) * np.exp(-z)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _filter_poles():
    """The four non-real fifth roots of unity omega_k, the pole directions beta_k = sqrt(-omega_k) with
    positive real part and the residues c_k = 1/(omega_k P'(omega_k)) of 1 - 1/S in x = (sigma q)^2."""
    omega = np.exp(2j * np.pi * np.arange(1, 5) / 5.0)
    beta = np.exp(1j * np.pi * (np.arange(1, 5) / 5.0 - 0.5))
    ck = 1.0 / (omega * (1.0 + 2.0 * omega + 3.0 * omega ** 2 + 4.0 * omega ** 3))
    return beta, ck


def _vs_closed(u, sigma, Gamma):
    """vs(u) = Gamma int_0^inf (S-1)/(q S) J0(qu) dq = -Gamma sum_k c_k K0(beta_k u/sigma)."""
    u = np.asarray(u, dtype=float)
    if sigma == 0.0:
        return np.zeros_like(u)
    beta, ck = _filter_poles()
    z = np.multiply.outer(u / sigma, beta)
    return -Gamma * np.real(np.sum(ck * _kv(0, z), axis=-1))


def short_range_potential(u: "np.ndarray", sigma: float, Gamma: float) -> "np.ndarray":
    sigma = _check_sigma(sigma); Gamma, _ = _check_state(Gamma, 1.0)
    u = np.atleast_1d(np.asarray(u, dtype=float))
    if u.ndim != 1 or np.any(u <= 0.0) or not np.all(np.isfinite(u)):
        raise ValueError("u must be positive")
    return _vs_closed(u, sigma, Gamma)

import numpy as np
from scipy.special import kve, k0, k1, j0, j1
from scipy.optimize import brentq


def _check_state(Gamma, rho):
    Gamma = float(Gamma); rho = float(rho)
    if not (Gamma > 0.0) or not np.isfinite(Gamma):
        raise ValueError("Gamma must be positive")
    if not (rho > 0.0) or not np.isfinite(rho):
        raise ValueError("rho must be positive")
    return Gamma, rho


def _check_sigma(sigma):
    sigma = float(sigma)
    if not (sigma >= 0.0) or not np.isfinite(sigma):
        raise ValueError("sigma must be non-negative")
    return sigma


def _kv(n, z):
    z = np.asarray(z, dtype=complex)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        out = kve(n, z) * np.exp(-z)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _gl_partial_fractions(sigma, kap2):
    """Poles of 1/(x S(x) + kap2) in x = q^2: residues B_m and b_m = sqrt(-x_m) with Re b_m > 0."""
    if sigma == 0.0:
        return np.array([1.0 + 0j]), np.array([np.sqrt(kap2) + 0j])
    coef = np.array([kap2, 1.0, sigma ** 2, sigma ** 4, sigma ** 6, sigma ** 8], dtype=complex)
    x = np.polynomial.polynomial.polyroots(coef)
    B = 1.0 / np.polynomial.polynomial.polyval(x, np.polynomial.polynomial.polyder(coef))
    b = np.sqrt(-x)
    b = np.where(b.real < 0.0, -b, b)
    return B, b


def _gl_closed(u, sigma, Gamma, kap2):
    """Gl(u) = Gamma int_0^inf q J0(qu)/(q^2 S + kap2) dq = Gamma sum_m B_m K0(b_m u)."""
    B, b = _gl_partial_fractions(sigma, kap2)
    u = np.asarray(u, dtype=float)
    z = np.multiply.outer(u, b)
    out = Gamma * np.real(np.sum(B * _kv(0, z), axis=-1))
    zero = (u == 0.0)
    if np.any(zero):
        out = np.where(zero, -Gamma * np.real(np.sum(B * np.log(b))), out)
    return out


def _filter_poles():
    """The four non-real fifth roots of unity omega_k, the pole directions beta_k = sqrt(-omega_k) with
    positive real part and the residues c_k = 1/(omega_k P'(omega_k)) of 1 - 1/S in x = (sigma q)^2."""
    omega = np.exp(2j * np.pi * np.arange(1, 5) / 5.0)
    beta = np.exp(1j * np.pi * (np.arange(1, 5) / 5.0 - 0.5))
    ck = 1.0 / (omega * (1.0 + 2.0 * omega + 3.0 * omega ** 2 + 4.0 * omega ** 3))
    return beta, ck


def _vs_closed(u, sigma, Gamma):
    """vs(u) = Gamma int_0^inf (S-1)/(q S) J0(qu) dq = -Gamma sum_k c_k K0(beta_k u/sigma)."""
    u = np.asarray(u, dtype=float)
    if sigma == 0.0:
        return np.zeros_like(u)
    beta, ck = _filter_poles()
    z = np.multiply.outer(u / sigma, beta)
    return -Gamma * np.real(np.sum(ck * _kv(0, z), axis=-1))


def _w_total(u, sigma, Gamma, rho):
    kap2 = 2.0 * np.pi * Gamma * rho
    return _vs_closed(u, sigma, Gamma) + _gl_closed(u, sigma, Gamma, kap2)


def mayer_functions(u: "np.ndarray", sigma: float, Gamma: float, rho: float) -> "np.ndarray":
    sigma = _check_sigma(sigma); Gamma, rho = _check_state(Gamma, rho)
    u = np.atleast_1d(np.asarray(u, dtype=float))
    if u.ndim != 1 or np.any(u < 0.0) or not np.all(np.isfinite(u)):
        raise ValueError("u must be non-negative")
    out = -np.ones((2, len(u)))
    m = u >= 1.0
    if m.any():
        w = _w_total(u[m], sigma, Gamma, rho)
        out[0, m] = np.exp(w) - 1.0
        out[1, m] = np.exp(-w) - 1.0
    return out

import numpy as np
from scipy.special import kve, k0, k1, j0, j1
from scipy.optimize import brentq


def _check_state(Gamma, rho):
    Gamma = float(Gamma); rho = float(rho)
    if not (Gamma > 0.0) or not np.isfinite(Gamma):
        raise ValueError("Gamma must be positive")
    if not (rho > 0.0) or not np.isfinite(rho):
        raise ValueError("rho must be positive")
    return Gamma, rho


def _check_sigma(sigma):
    sigma = float(sigma)
    if not (sigma >= 0.0) or not np.isfinite(sigma):
        raise ValueError("sigma must be non-negative")
    return sigma


def _kv(n, z):
    z = np.asarray(z, dtype=complex)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        out = kve(n, z) * np.exp(-z)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _gl_partial_fractions(sigma, kap2):
    """Poles of 1/(x S(x) + kap2) in x = q^2: residues B_m and b_m = sqrt(-x_m) with Re b_m > 0."""
    if sigma == 0.0:
        return np.array([1.0 + 0j]), np.array([np.sqrt(kap2) + 0j])
    coef = np.array([kap2, 1.0, sigma ** 2, sigma ** 4, sigma ** 6, sigma ** 8], dtype=complex)
    x = np.polynomial.polynomial.polyroots(coef)
    B = 1.0 / np.polynomial.polynomial.polyval(x, np.polynomial.polynomial.polyder(coef))
    b = np.sqrt(-x)
    b = np.where(b.real < 0.0, -b, b)
    return B, b


def _gl_closed(u, sigma, Gamma, kap2):
    """Gl(u) = Gamma int_0^inf q J0(qu)/(q^2 S + kap2) dq = Gamma sum_m B_m K0(b_m u)."""
    B, b = _gl_partial_fractions(sigma, kap2)
    u = np.asarray(u, dtype=float)
    z = np.multiply.outer(u, b)
    out = Gamma * np.real(np.sum(B * _kv(0, z), axis=-1))
    zero = (u == 0.0)
    if np.any(zero):
        out = np.where(zero, -Gamma * np.real(np.sum(B * np.log(b))), out)
    return out


def _filter_poles():
    """The four non-real fifth roots of unity omega_k, the pole directions beta_k = sqrt(-omega_k) with
    positive real part and the residues c_k = 1/(omega_k P'(omega_k)) of 1 - 1/S in x = (sigma q)^2."""
    omega = np.exp(2j * np.pi * np.arange(1, 5) / 5.0)
    beta = np.exp(1j * np.pi * (np.arange(1, 5) / 5.0 - 0.5))
    ck = 1.0 / (omega * (1.0 + 2.0 * omega + 3.0 * omega ** 2 + 4.0 * omega ** 3))
    return beta, ck


def _vs_closed(u, sigma, Gamma):
    """vs(u) = Gamma int_0^inf (S-1)/(q S) J0(qu) dq = -Gamma sum_k c_k K0(beta_k u/sigma)."""
    u = np.asarray(u, dtype=float)
    if sigma == 0.0:
        return np.zeros_like(u)
    beta, ck = _filter_poles()
    z = np.multiply.outer(u / sigma, beta)
    return -Gamma * np.real(np.sum(ck * _kv(0, z), axis=-1))


def _dvl_dsigma_closed(u, sigma, Gamma):
    """d vl / d sigma = - d vs / d sigma = Gamma sum_k c_k K1(beta_k u/sigma) beta_k u / sigma^2."""
    u = np.asarray(u, dtype=float)
    beta, ck = _filter_poles()
    z = np.multiply.outer(u / sigma, beta)
    return Gamma * np.real(np.sum(ck * _kv(1, z) * z, axis=-1)) / sigma


def _w_total(u, sigma, Gamma, rho):
    kap2 = 2.0 * np.pi * Gamma * rho
    return _vs_closed(u, sigma, Gamma) + _gl_closed(u, sigma, Gamma, kap2)


def _leggauss(n):
    """Gauss-Legendre nodes and weights on [-1, 1], rebuilt on every call (no module-level cache)."""
    return np.polynomial.legendre.leggauss(n)


def _gl_nodes(a, b, xw):
    x, w = xw
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def _panel_edges(a, b, width, ratio=1.12, far=6.0):
    if b - a <= 0.0:
        return np.array([a, b])
    if b <= far:
        m = max(1, int(np.ceil((b - a) / width)))
        return np.linspace(a, b, m + 1)
    e = [a]
    while e[-1] < far - 1e-12 and e[-1] < b:
        e.append(min(e[-1] + width, far, b))
    w = width
    while e[-1] < b - 1e-12:
        w *= ratio
        e.append(min(e[-1] + w, b))
    return np.array(e)


def _panel_nodes(edges, n):
    xw = _leggauss(n); xs = []; ws = []
    for a, b in zip(edges[:-1], edges[1:]):
        x, w = _gl_nodes(a, b, xw); xs.append(x); ws.append(w)
    return np.concatenate(xs), np.concatenate(ws)


def _umax(Gamma, rho):
    kap = np.sqrt(2.0 * np.pi * Gamma * rho)
    return 1.0 + max(30.0, 45.0 / kap)


def _variational_real(sigma, Gamma, rho):
    """4 pi^2 int_0^inf u [h++ - h+- + 2 Gl] d vl/d sigma du  (Parseval form of the Fourier integral)."""
    kap2 = 2.0 * np.pi * Gamma * rho
    U = _umax(Gamma, rho)
    u1, w1 = _panel_nodes(_panel_edges(1.0, U, 0.25), 12)
    u0, w0 = _panel_nodes(_panel_edges(0.0, U, 0.25), 12)
    w = _w_total(u1, sigma, Gamma, rho)
    t1 = np.sum(w1 * u1 * (-2.0 * np.sinh(w)) * _dvl_dsigma_closed(u1, sigma, Gamma))
    t2 = 2.0 * np.sum(w0 * u0 * _gl_closed(u0, sigma, Gamma, kap2) * _dvl_dsigma_closed(u0, sigma, Gamma))
    return 4.0 * np.pi ** 2 * (t1 + t2)


def variational_residual(sigma: float, Gamma: float, rho: float) -> float:
    sigma = _check_sigma(sigma); Gamma, rho = _check_state(Gamma, rho)
    if sigma == 0.0:
        raise ValueError("sigma must be positive")
    return float(_variational_real(sigma, Gamma, rho))

import numpy as np
from scipy.special import kve, k0, k1, j0, j1
from scipy.optimize import brentq


def _check_state(Gamma, rho):
    Gamma = float(Gamma); rho = float(rho)
    if not (Gamma > 0.0) or not np.isfinite(Gamma):
        raise ValueError("Gamma must be positive")
    if not (rho > 0.0) or not np.isfinite(rho):
        raise ValueError("rho must be positive")
    return Gamma, rho


def _kv(n, z):
    z = np.asarray(z, dtype=complex)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        out = kve(n, z) * np.exp(-z)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _gl_partial_fractions(sigma, kap2):
    """Poles of 1/(x S(x) + kap2) in x = q^2: residues B_m and b_m = sqrt(-x_m) with Re b_m > 0."""
    if sigma == 0.0:
        return np.array([1.0 + 0j]), np.array([np.sqrt(kap2) + 0j])
    coef = np.array([kap2, 1.0, sigma ** 2, sigma ** 4, sigma ** 6, sigma ** 8], dtype=complex)
    x = np.polynomial.polynomial.polyroots(coef)
    B = 1.0 / np.polynomial.polynomial.polyval(x, np.polynomial.polynomial.polyder(coef))
    b = np.sqrt(-x)
    b = np.where(b.real < 0.0, -b, b)
    return B, b


def _gl_closed(u, sigma, Gamma, kap2):
    """Gl(u) = Gamma int_0^inf q J0(qu)/(q^2 S + kap2) dq = Gamma sum_m B_m K0(b_m u)."""
    B, b = _gl_partial_fractions(sigma, kap2)
    u = np.asarray(u, dtype=float)
    z = np.multiply.outer(u, b)
    out = Gamma * np.real(np.sum(B * _kv(0, z), axis=-1))
    zero = (u == 0.0)
    if np.any(zero):
        out = np.where(zero, -Gamma * np.real(np.sum(B * np.log(b))), out)
    return out


def _filter_poles():
    """The four non-real fifth roots of unity omega_k, the pole directions beta_k = sqrt(-omega_k) with
    positive real part and the residues c_k = 1/(omega_k P'(omega_k)) of 1 - 1/S in x = (sigma q)^2."""
    omega = np.exp(2j * np.pi * np.arange(1, 5) / 5.0)
    beta = np.exp(1j * np.pi * (np.arange(1, 5) / 5.0 - 0.5))
    ck = 1.0 / (omega * (1.0 + 2.0 * omega + 3.0 * omega ** 2 + 4.0 * omega ** 3))
    return beta, ck


def _vs_closed(u, sigma, Gamma):
    """vs(u) = Gamma int_0^inf (S-1)/(q S) J0(qu) dq = -Gamma sum_k c_k K0(beta_k u/sigma)."""
    u = np.asarray(u, dtype=float)
    if sigma == 0.0:
        return np.zeros_like(u)
    beta, ck = _filter_poles()
    z = np.multiply.outer(u / sigma, beta)
    return -Gamma * np.real(np.sum(ck * _kv(0, z), axis=-1))


def _dvl_dsigma_closed(u, sigma, Gamma):
    """d vl / d sigma = - d vs / d sigma = Gamma sum_k c_k K1(beta_k u/sigma) beta_k u / sigma^2."""
    u = np.asarray(u, dtype=float)
    beta, ck = _filter_poles()
    z = np.multiply.outer(u / sigma, beta)
    return Gamma * np.real(np.sum(ck * _kv(1, z) * z, axis=-1)) / sigma


def _w_total(u, sigma, Gamma, rho):
    kap2 = 2.0 * np.pi * Gamma * rho
    return _vs_closed(u, sigma, Gamma) + _gl_closed(u, sigma, Gamma, kap2)


def _leggauss(n):
    """Gauss-Legendre nodes and weights on [-1, 1], rebuilt on every call (no module-level cache)."""
    return np.polynomial.legendre.leggauss(n)


def _gl_nodes(a, b, xw):
    x, w = xw
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def _panel_edges(a, b, width, ratio=1.12, far=6.0):
    if b - a <= 0.0:
        return np.array([a, b])
    if b <= far:
        m = max(1, int(np.ceil((b - a) / width)))
        return np.linspace(a, b, m + 1)
    e = [a]
    while e[-1] < far - 1e-12 and e[-1] < b:
        e.append(min(e[-1] + width, far, b))
    w = width
    while e[-1] < b - 1e-12:
        w *= ratio
        e.append(min(e[-1] + w, b))
    return np.array(e)


def _panel_nodes(edges, n):
    xw = _leggauss(n); xs = []; ws = []
    for a, b in zip(edges[:-1], edges[1:]):
        x, w = _gl_nodes(a, b, xw); xs.append(x); ws.append(w)
    return np.concatenate(xs), np.concatenate(ws)


def _umax(Gamma, rho):
    kap = np.sqrt(2.0 * np.pi * Gamma * rho)
    return 1.0 + max(30.0, 45.0 / kap)


def _variational_real(sigma, Gamma, rho):
    """4 pi^2 int_0^inf u [h++ - h+- + 2 Gl] d vl/d sigma du  (Parseval form of the Fourier integral)."""
    kap2 = 2.0 * np.pi * Gamma * rho
    U = _umax(Gamma, rho)
    u1, w1 = _panel_nodes(_panel_edges(1.0, U, 0.25), 12)
    u0, w0 = _panel_nodes(_panel_edges(0.0, U, 0.25), 12)
    w = _w_total(u1, sigma, Gamma, rho)
    t1 = np.sum(w1 * u1 * (-2.0 * np.sinh(w)) * _dvl_dsigma_closed(u1, sigma, Gamma))
    t2 = 2.0 * np.sum(w0 * u0 * _gl_closed(u0, sigma, Gamma, kap2) * _dvl_dsigma_closed(u0, sigma, Gamma))
    return 4.0 * np.pi ** 2 * (t1 + t2)


def splitting_length(Gamma: float, rho: float) -> float:
    if not np.isfinite(float(Gamma)) or not np.isfinite(float(rho)) or float(Gamma) <= 0.0 or float(rho) <= 0.0:
        raise ValueError("Gamma and rho must be positive")
    Gamma, rho = _check_state(Gamma, rho)
    ss = np.geomspace(0.05, 5.0, 16)
    vals = np.array([_variational_real(s, Gamma, rho) for s in ss])
    idx = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    if len(idx) == 0:
        raise RuntimeError("no sign change on [0.05, 5]")
    a, b = ss[idx[0]], ss[idx[0] + 1]
    return float(brentq(lambda s: _variational_real(s, Gamma, rho), a, b, xtol=1e-14, rtol=1e-14, maxiter=200))

import numpy as np
from scipy.special import kve, k0, k1, j0, j1
from scipy.optimize import brentq


def _check_state(Gamma, rho):
    Gamma = float(Gamma); rho = float(rho)
    if not (Gamma > 0.0) or not np.isfinite(Gamma):
        raise ValueError("Gamma must be positive")
    if not (rho > 0.0) or not np.isfinite(rho):
        raise ValueError("rho must be positive")
    return Gamma, rho


def _check_sigma(sigma):
    sigma = float(sigma)
    if not (sigma >= 0.0) or not np.isfinite(sigma):
        raise ValueError("sigma must be non-negative")
    return sigma


def _kv(n, z):
    z = np.asarray(z, dtype=complex)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        out = kve(n, z) * np.exp(-z)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _S(q, sigma):
    x = (sigma * q) ** 2
    return 1.0 + x + x ** 2 + x ** 3 + x ** 4


def _gl_partial_fractions(sigma, kap2):
    """Poles of 1/(x S(x) + kap2) in x = q^2: residues B_m and b_m = sqrt(-x_m) with Re b_m > 0."""
    if sigma == 0.0:
        return np.array([1.0 + 0j]), np.array([np.sqrt(kap2) + 0j])
    coef = np.array([kap2, 1.0, sigma ** 2, sigma ** 4, sigma ** 6, sigma ** 8], dtype=complex)
    x = np.polynomial.polynomial.polyroots(coef)
    B = 1.0 / np.polynomial.polynomial.polyval(x, np.polynomial.polynomial.polyder(coef))
    b = np.sqrt(-x)
    b = np.where(b.real < 0.0, -b, b)
    return B, b


def _gl_closed(u, sigma, Gamma, kap2):
    """Gl(u) = Gamma int_0^inf q J0(qu)/(q^2 S + kap2) dq = Gamma sum_m B_m K0(b_m u)."""
    B, b = _gl_partial_fractions(sigma, kap2)
    u = np.asarray(u, dtype=float)
    z = np.multiply.outer(u, b)
    out = Gamma * np.real(np.sum(B * _kv(0, z), axis=-1))
    zero = (u == 0.0)
    if np.any(zero):
        out = np.where(zero, -Gamma * np.real(np.sum(B * np.log(b))), out)
    return out


def _filter_poles():
    """The four non-real fifth roots of unity omega_k, the pole directions beta_k = sqrt(-omega_k) with
    positive real part and the residues c_k = 1/(omega_k P'(omega_k)) of 1 - 1/S in x = (sigma q)^2."""
    omega = np.exp(2j * np.pi * np.arange(1, 5) / 5.0)
    beta = np.exp(1j * np.pi * (np.arange(1, 5) / 5.0 - 0.5))
    ck = 1.0 / (omega * (1.0 + 2.0 * omega + 3.0 * omega ** 2 + 4.0 * omega ** 3))
    return beta, ck


def _vs_closed(u, sigma, Gamma):
    """vs(u) = Gamma int_0^inf (S-1)/(q S) J0(qu) dq = -Gamma sum_k c_k K0(beta_k u/sigma)."""
    u = np.asarray(u, dtype=float)
    if sigma == 0.0:
        return np.zeros_like(u)
    beta, ck = _filter_poles()
    z = np.multiply.outer(u / sigma, beta)
    return -Gamma * np.real(np.sum(ck * _kv(0, z), axis=-1))


def _w_total(u, sigma, Gamma, rho):
    kap2 = 2.0 * np.pi * Gamma * rho
    return _vs_closed(u, sigma, Gamma) + _gl_closed(u, sigma, Gamma, kap2)


def _leggauss(n):
    """Gauss-Legendre nodes and weights on [-1, 1], rebuilt on every call (no module-level cache)."""
    return np.polynomial.legendre.leggauss(n)


def _gl_nodes(a, b, xw):
    x, w = xw
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def _panel_edges(a, b, width, ratio=1.12, far=6.0):
    if b - a <= 0.0:
        return np.array([a, b])
    if b <= far:
        m = max(1, int(np.ceil((b - a) / width)))
        return np.linspace(a, b, m + 1)
    e = [a]
    while e[-1] < far - 1e-12 and e[-1] < b:
        e.append(min(e[-1] + width, far, b))
    w = width
    while e[-1] < b - 1e-12:
        w *= ratio
        e.append(min(e[-1] + w, b))
    return np.array(e)


def _panel_nodes(edges, n):
    xw = _leggauss(n); xs = []; ws = []
    for a, b in zip(edges[:-1], edges[1:]):
        x, w = _gl_nodes(a, b, xw); xs.append(x); ws.append(w)
    return np.concatenate(xs), np.concatenate(ws)


def _umax(Gamma, rho):
    kap = np.sqrt(2.0 * np.pi * Gamma * rho)
    return 1.0 + max(30.0, 45.0 / kap)


def _rcut(Gamma, rho, sigma):
    """Radial cutoff of the correlation integrals: the distance at which the slowest exponential tail of the
    kernels (the screened poles b_m of Gl and the filter poles beta_k/sigma of vs) has decayed by 13 decades,
    never beyond _umax."""
    kap2 = 2.0 * np.pi * Gamma * rho
    B, b = _gl_partial_fractions(sigma, kap2)
    lam = float(np.min(b.real))
    if sigma > 0.0:
        beta, _ = _filter_poles()
        lam = min(lam, float(np.min(beta.real)) / sigma)
    return min(_umax(Gamma, rho), 1.0 + 13.0 * np.log(10.0) / lam)


def _w_spline(sigma, Gamma, rho):
    from scipy.interpolate import CubicSpline
    U = _umax(Gamma, rho) + 3.0
    ug = np.linspace(1e-6, U, int(U / 0.002) + 1)
    return CubicSpline(ug, _w_total(ug, sigma, Gamma, rho))


def _chord_A(r, r1, g, g1, xw):
    """A_g(r, r1) = int_0^{2 pi} g(d(phi)) dphi with d^2 = r^2 + r1^2 - 2 r r1 cos(phi), g(d) = 0 for d < 1,
    written as 4 int g(d) d dd / sqrt((d^2 - dmin^2)(dmax^2 - d^2)) with the endpoint singularities removed;
    xw holds the Gauss-Legendre nodes and weights of the angular quadrature."""
    x, w = xw
    th = 0.5 * np.pi * (x + 1.0); wth = 0.5 * np.pi * w
    th2 = 0.25 * np.pi * (x + 1.0); wth2 = 0.25 * np.pi * w
    r1 = np.asarray(r1, dtype=float)
    dmin = np.abs(r - r1); dmax = r + r1
    out = np.zeros_like(r1)
    m1 = dmin >= 1.0
    if m1.any():
        xmin = dmin[m1] ** 2; xmax = dmax[m1] ** 2
        xmid = 0.5 * (xmin + xmax); xh = 0.5 * (xmax - xmin)
        xx = xmid[:, None] - xh[:, None] * np.cos(th)[None, :]
        out[m1] = 2.0 * np.sum(wth * g(np.sqrt(xx)), axis=1)
    m2 = (dmin < 1.0) & (dmax > 1.0)
    if m2.any():
        xmin = dmin[m2] ** 2; xmax = dmax[m2] ** 2
        a = xmax - 1.0; eps = 1.0 - xmin
        c = np.cos(th2)[None, :]
        xx = xmax[:, None] - a[:, None] * np.sin(th2)[None, :] ** 2
        D = np.sqrt(a[:, None] * c ** 2 + eps[:, None])
        integ = np.sum(wth2 * (g(np.sqrt(xx)) - g1) * c / D, axis=1)
        sing = g1 * np.arcsin(1.0 / np.sqrt(1.0 + eps / a)) / np.sqrt(a)
        out[m2] = 4.0 * np.sqrt(a) * (sing + integ)
    return out


def _graded_panels(a, b, kink_lo, kink_hi, width, xw):
    edges = _panel_edges(a, b, width); m = len(edges) - 1
    x, w = xw; xs = []; ws = []
    for i, (p, q) in enumerate(zip(edges[:-1], edges[1:])):
        L = q - p
        if i == 0 and kink_lo:
            t = 0.5 * (x + 1.0); wt = 0.5 * w
            xs.append(p + L * t * t); ws.append(wt * 2.0 * L * t)
        elif i == m - 1 and kink_hi:
            t = 0.5 * (x + 1.0); wt = 0.5 * w
            xs.append(q - L * t * t); ws.append(wt * 2.0 * L * t)
        else:
            xs.append(0.5 * L * x + 0.5 * (p + q)); ws.append(0.5 * L * w)
    return np.concatenate(xs), np.concatenate(ws)


def _theta_conv_k(r, g, g1, xw_out, xw_in, width=0.25):
    """(theta * k)(r), r >= 1: int_0^1 r1 dr1 A_g(r, r1); theta = unit-disc indicator, k = g on d >= 1;
    xw_out and xw_in are the Gauss-Legendre nodes of the outer (r1) and inner (angular) quadratures."""
    if r >= 2.0:
        r1, w = _graded_panels(0.0, 1.0, False, False, width, xw_out)
        return float(np.sum(w * r1 * _chord_A(r, r1, g, g1, xw_in)))
    rk = r - 1.0; tot = 0.0
    if rk > 0.0:
        r1, w = _graded_panels(0.0, rk, False, False, width, xw_out); tot += np.sum(w * r1 * _chord_A(r, r1, g, g1, xw_in))
    r1, w = _graded_panels(rk, 1.0, True, False, width, xw_out); tot += np.sum(w * r1 * _chord_A(r, r1, g, g1, xw_in))
    return float(tot)


def _k_conv_k(r, fa, gb, gb1, rcut, xw_out, xw_in, width=0.25):
    """(k_a * k_b)(r) = int_1^rcut r1 k_a(r1) A_b(r, r1) dr1, sqrt cusps at r1 = r -/+ 1 handled."""
    pts = sorted(set([1.0] + [x for x in (r - 1.0, r + 1.0) if 1.0 < x < rcut] + [rcut]))
    tot = 0.0
    for p, q in zip(pts[:-1], pts[1:]):
        klo = abs(p - (r - 1.0)) < 1e-14
        khi = abs(q - (r + 1.0)) < 1e-14
        r1, w = _graded_panels(p, q, klo, khi, width, xw_out)
        tot += np.sum(w * r1 * fa(r1) * _chord_A(r, r1, gb, gb1, xw_in))
    return float(tot)


def _lens(r):
    r = np.asarray(r, dtype=float); out = np.zeros_like(r); m = r < 2.0
    out[m] = 2.0 * np.arccos(r[m] / 2.0) - 0.5 * r[m] * np.sqrt(4.0 - r[m] ** 2)
    return out


def _glgl(r, sigma, Gamma, rho):
    """(Gl * Gl)(r) = int dq q/(2 pi) J0(q r) Glbar(q)^2 (smooth, fast-decaying integrand)."""
    kap2 = 2.0 * np.pi * Gamma * rho
    qmax = max(60.0, 40.0 / max(sigma, 0.05))
    q, w = _panel_nodes(np.linspace(0.0, qmax, int(qmax / 0.05) + 1), 8)
    gb = (2.0 * np.pi * Gamma / (q * q * _S(q, sigma) + kap2)) ** 2
    r = np.atleast_1d(np.asarray(r, dtype=float)); out = np.empty_like(r)
    for i in range(0, len(r), 64):
        rr = r[i:i + 64]
        out[i:i + 64] = np.sum(w * q * gb * j0(np.outer(rr, q)), axis=1) / (2.0 * np.pi)
    return out


def _T_values(r, sigma, Gamma, rho, wspl):
    r = np.atleast_1d(np.asarray(r, dtype=float))
    kp = lambda d: np.exp(wspl(d)) - 1.0      # h+- for d >= 1
    km = lambda d: np.exp(-wspl(d)) - 1.0     # h++ for d >= 1
    kp1 = float(kp(1.0)); km1 = float(km(1.0))
    U = _rcut(Gamma, rho, sigma)
    L = _lens(r)
    xo = _leggauss(16); xi = _leggauss(32)
    tkp = np.array([_theta_conv_k(ri, kp, kp1, xo, xi) for ri in r])
    tkm = np.array([_theta_conv_k(ri, km, km1, xo, xi) for ri in r])
    kmm = np.array([_k_conv_k(ri, km, km, km1, U, xo, xi) for ri in r])
    kpp = np.array([_k_conv_k(ri, kp, kp, kp1, U, xo, xi) for ri in r])
    kmp = np.array([_k_conv_k(ri, km, kp, kp1, U, xo, xi) for ri in r])
    gg = _glgl(r, sigma, Gamma, rho)
    hpp = L - 2.0 * tkm + kmm          # h++ * h++
    hmm = L - 2.0 * tkp + kpp          # h+- * h+-
    hpm = L - tkp - tkm + kmp          # h++ * h+-
    Tpp = 0.5 * rho * (hpp + hmm - 2.0 * gg)
    Tpm = rho * (hpm + gg)
    return Tpp, Tpm


def pair_convolutions(r: "np.ndarray", sigma: float, Gamma: float, rho: float) -> "np.ndarray":
    sigma = _check_sigma(sigma); Gamma, rho = _check_state(Gamma, rho)
    r = np.atleast_1d(np.asarray(r, dtype=float))
    if r.ndim != 1 or np.any(r < 1.0) or not np.all(np.isfinite(r)):
        raise ValueError("r must be >= 1")
    wspl = _w_spline(sigma, Gamma, rho)
    Tpp, Tpm = _T_values(r, sigma, Gamma, rho, wspl)
    return np.vstack([Tpp, Tpm])

import numpy as np
from scipy.special import kve, k0, k1, j0, j1
from scipy.optimize import brentq


def _check_state(Gamma, rho):
    Gamma = float(Gamma); rho = float(rho)
    if not (Gamma > 0.0) or not np.isfinite(Gamma):
        raise ValueError("Gamma must be positive")
    if not (rho > 0.0) or not np.isfinite(rho):
        raise ValueError("rho must be positive")
    return Gamma, rho


def _check_sigma(sigma):
    sigma = float(sigma)
    if not (sigma >= 0.0) or not np.isfinite(sigma):
        raise ValueError("sigma must be non-negative")
    return sigma


def _kv(n, z):
    z = np.asarray(z, dtype=complex)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        out = kve(n, z) * np.exp(-z)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _S(q, sigma):
    x = (sigma * q) ** 2
    return 1.0 + x + x ** 2 + x ** 3 + x ** 4


def _gl_partial_fractions(sigma, kap2):
    """Poles of 1/(x S(x) + kap2) in x = q^2: residues B_m and b_m = sqrt(-x_m) with Re b_m > 0."""
    if sigma == 0.0:
        return np.array([1.0 + 0j]), np.array([np.sqrt(kap2) + 0j])
    coef = np.array([kap2, 1.0, sigma ** 2, sigma ** 4, sigma ** 6, sigma ** 8], dtype=complex)
    x = np.polynomial.polynomial.polyroots(coef)
    B = 1.0 / np.polynomial.polynomial.polyval(x, np.polynomial.polynomial.polyder(coef))
    b = np.sqrt(-x)
    b = np.where(b.real < 0.0, -b, b)
    return B, b


def _gl_closed(u, sigma, Gamma, kap2):
    """Gl(u) = Gamma int_0^inf q J0(qu)/(q^2 S + kap2) dq = Gamma sum_m B_m K0(b_m u)."""
    B, b = _gl_partial_fractions(sigma, kap2)
    u = np.asarray(u, dtype=float)
    z = np.multiply.outer(u, b)
    out = Gamma * np.real(np.sum(B * _kv(0, z), axis=-1))
    zero = (u == 0.0)
    if np.any(zero):
        out = np.where(zero, -Gamma * np.real(np.sum(B * np.log(b))), out)
    return out


def _filter_poles():
    """The four non-real fifth roots of unity omega_k, the pole directions beta_k = sqrt(-omega_k) with
    positive real part and the residues c_k = 1/(omega_k P'(omega_k)) of 1 - 1/S in x = (sigma q)^2."""
    omega = np.exp(2j * np.pi * np.arange(1, 5) / 5.0)
    beta = np.exp(1j * np.pi * (np.arange(1, 5) / 5.0 - 0.5))
    ck = 1.0 / (omega * (1.0 + 2.0 * omega + 3.0 * omega ** 2 + 4.0 * omega ** 3))
    return beta, ck


def _vs_closed(u, sigma, Gamma):
    """vs(u) = Gamma int_0^inf (S-1)/(q S) J0(qu) dq = -Gamma sum_k c_k K0(beta_k u/sigma)."""
    u = np.asarray(u, dtype=float)
    if sigma == 0.0:
        return np.zeros_like(u)
    beta, ck = _filter_poles()
    z = np.multiply.outer(u / sigma, beta)
    return -Gamma * np.real(np.sum(ck * _kv(0, z), axis=-1))


def _w_total(u, sigma, Gamma, rho):
    kap2 = 2.0 * np.pi * Gamma * rho
    return _vs_closed(u, sigma, Gamma) + _gl_closed(u, sigma, Gamma, kap2)


def _leggauss(n):
    """Gauss-Legendre nodes and weights on [-1, 1], rebuilt on every call (no module-level cache)."""
    return np.polynomial.legendre.leggauss(n)


def _gl_nodes(a, b, xw):
    x, w = xw
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def _panel_edges(a, b, width, ratio=1.12, far=6.0):
    if b - a <= 0.0:
        return np.array([a, b])
    if b <= far:
        m = max(1, int(np.ceil((b - a) / width)))
        return np.linspace(a, b, m + 1)
    e = [a]
    while e[-1] < far - 1e-12 and e[-1] < b:
        e.append(min(e[-1] + width, far, b))
    w = width
    while e[-1] < b - 1e-12:
        w *= ratio
        e.append(min(e[-1] + w, b))
    return np.array(e)


def _panel_nodes(edges, n):
    xw = _leggauss(n); xs = []; ws = []
    for a, b in zip(edges[:-1], edges[1:]):
        x, w = _gl_nodes(a, b, xw); xs.append(x); ws.append(w)
    return np.concatenate(xs), np.concatenate(ws)


def _umax(Gamma, rho):
    kap = np.sqrt(2.0 * np.pi * Gamma * rho)
    return 1.0 + max(30.0, 45.0 / kap)


def _rcut(Gamma, rho, sigma):
    """Radial cutoff of the correlation integrals: the distance at which the slowest exponential tail of the
    kernels (the screened poles b_m of Gl and the filter poles beta_k/sigma of vs) has decayed by 13 decades,
    never beyond _umax."""
    kap2 = 2.0 * np.pi * Gamma * rho
    B, b = _gl_partial_fractions(sigma, kap2)
    lam = float(np.min(b.real))
    if sigma > 0.0:
        beta, _ = _filter_poles()
        lam = min(lam, float(np.min(beta.real)) / sigma)
    return min(_umax(Gamma, rho), 1.0 + 13.0 * np.log(10.0) / lam)


def _w_spline(sigma, Gamma, rho):
    from scipy.interpolate import CubicSpline
    U = _umax(Gamma, rho) + 3.0
    ug = np.linspace(1e-6, U, int(U / 0.002) + 1)
    return CubicSpline(ug, _w_total(ug, sigma, Gamma, rho))


def _hankel_forward(q, u, wu, f):
    """2 pi int u J0(q u) f(u) du on the nodes (u, wu), for an array q."""
    q = np.atleast_1d(np.asarray(q, dtype=float))
    out = np.empty_like(q)
    for i in range(0, len(q), 64):
        qq = q[i:i + 64]
        out[i:i + 64] = 2.0 * np.pi * np.sum(j0(np.outer(qq, u)) * (wu * u * f), axis=1)
    return out


def _chord_A(r, r1, g, g1, xw):
    """A_g(r, r1) = int_0^{2 pi} g(d(phi)) dphi with d^2 = r^2 + r1^2 - 2 r r1 cos(phi), g(d) = 0 for d < 1,
    written as 4 int g(d) d dd / sqrt((d^2 - dmin^2)(dmax^2 - d^2)) with the endpoint singularities removed;
    xw holds the Gauss-Legendre nodes and weights of the angular quadrature."""
    x, w = xw
    th = 0.5 * np.pi * (x + 1.0); wth = 0.5 * np.pi * w
    th2 = 0.25 * np.pi * (x + 1.0); wth2 = 0.25 * np.pi * w
    r1 = np.asarray(r1, dtype=float)
    dmin = np.abs(r - r1); dmax = r + r1
    out = np.zeros_like(r1)
    m1 = dmin >= 1.0
    if m1.any():
        xmin = dmin[m1] ** 2; xmax = dmax[m1] ** 2
        xmid = 0.5 * (xmin + xmax); xh = 0.5 * (xmax - xmin)
        xx = xmid[:, None] - xh[:, None] * np.cos(th)[None, :]
        out[m1] = 2.0 * np.sum(wth * g(np.sqrt(xx)), axis=1)
    m2 = (dmin < 1.0) & (dmax > 1.0)
    if m2.any():
        xmin = dmin[m2] ** 2; xmax = dmax[m2] ** 2
        a = xmax - 1.0; eps = 1.0 - xmin
        c = np.cos(th2)[None, :]
        xx = xmax[:, None] - a[:, None] * np.sin(th2)[None, :] ** 2
        D = np.sqrt(a[:, None] * c ** 2 + eps[:, None])
        integ = np.sum(wth2 * (g(np.sqrt(xx)) - g1) * c / D, axis=1)
        sing = g1 * np.arcsin(1.0 / np.sqrt(1.0 + eps / a)) / np.sqrt(a)
        out[m2] = 4.0 * np.sqrt(a) * (sing + integ)
    return out


def _graded_panels(a, b, kink_lo, kink_hi, width, xw):
    edges = _panel_edges(a, b, width); m = len(edges) - 1
    x, w = xw; xs = []; ws = []
    for i, (p, q) in enumerate(zip(edges[:-1], edges[1:])):
        L = q - p
        if i == 0 and kink_lo:
            t = 0.5 * (x + 1.0); wt = 0.5 * w
            xs.append(p + L * t * t); ws.append(wt * 2.0 * L * t)
        elif i == m - 1 and kink_hi:
            t = 0.5 * (x + 1.0); wt = 0.5 * w
            xs.append(q - L * t * t); ws.append(wt * 2.0 * L * t)
        else:
            xs.append(0.5 * L * x + 0.5 * (p + q)); ws.append(0.5 * L * w)
    return np.concatenate(xs), np.concatenate(ws)


def _theta_conv_k(r, g, g1, xw_out, xw_in, width=0.25):
    """(theta * k)(r), r >= 1: int_0^1 r1 dr1 A_g(r, r1); theta = unit-disc indicator, k = g on d >= 1;
    xw_out and xw_in are the Gauss-Legendre nodes of the outer (r1) and inner (angular) quadratures."""
    if r >= 2.0:
        r1, w = _graded_panels(0.0, 1.0, False, False, width, xw_out)
        return float(np.sum(w * r1 * _chord_A(r, r1, g, g1, xw_in)))
    rk = r - 1.0; tot = 0.0
    if rk > 0.0:
        r1, w = _graded_panels(0.0, rk, False, False, width, xw_out); tot += np.sum(w * r1 * _chord_A(r, r1, g, g1, xw_in))
    r1, w = _graded_panels(rk, 1.0, True, False, width, xw_out); tot += np.sum(w * r1 * _chord_A(r, r1, g, g1, xw_in))
    return float(tot)


def _k_conv_k(r, fa, gb, gb1, rcut, xw_out, xw_in, width=0.25):
    """(k_a * k_b)(r) = int_1^rcut r1 k_a(r1) A_b(r, r1) dr1, sqrt cusps at r1 = r -/+ 1 handled."""
    pts = sorted(set([1.0] + [x for x in (r - 1.0, r + 1.0) if 1.0 < x < rcut] + [rcut]))
    tot = 0.0
    for p, q in zip(pts[:-1], pts[1:]):
        klo = abs(p - (r - 1.0)) < 1e-14
        khi = abs(q - (r + 1.0)) < 1e-14
        r1, w = _graded_panels(p, q, klo, khi, width, xw_out)
        tot += np.sum(w * r1 * fa(r1) * _chord_A(r, r1, gb, gb1, xw_in))
    return float(tot)


def _lens(r):
    r = np.asarray(r, dtype=float); out = np.zeros_like(r); m = r < 2.0
    out[m] = 2.0 * np.arccos(r[m] / 2.0) - 0.5 * r[m] * np.sqrt(4.0 - r[m] ** 2)
    return out


def _glgl(r, sigma, Gamma, rho):
    """(Gl * Gl)(r) = int dq q/(2 pi) J0(q r) Glbar(q)^2 (smooth, fast-decaying integrand)."""
    kap2 = 2.0 * np.pi * Gamma * rho
    qmax = max(60.0, 40.0 / max(sigma, 0.05))
    q, w = _panel_nodes(np.linspace(0.0, qmax, int(qmax / 0.05) + 1), 8)
    gb = (2.0 * np.pi * Gamma / (q * q * _S(q, sigma) + kap2)) ** 2
    r = np.atleast_1d(np.asarray(r, dtype=float)); out = np.empty_like(r)
    for i in range(0, len(r), 64):
        rr = r[i:i + 64]
        out[i:i + 64] = np.sum(w * q * gb * j0(np.outer(rr, q)), axis=1) / (2.0 * np.pi)
    return out


def _T_values(r, sigma, Gamma, rho, wspl):
    r = np.atleast_1d(np.asarray(r, dtype=float))
    kp = lambda d: np.exp(wspl(d)) - 1.0      # h+- for d >= 1
    km = lambda d: np.exp(-wspl(d)) - 1.0     # h++ for d >= 1
    kp1 = float(kp(1.0)); km1 = float(km(1.0))
    U = _rcut(Gamma, rho, sigma)
    L = _lens(r)
    xo = _leggauss(16); xi = _leggauss(32)
    tkp = np.array([_theta_conv_k(ri, kp, kp1, xo, xi) for ri in r])
    tkm = np.array([_theta_conv_k(ri, km, km1, xo, xi) for ri in r])
    kmm = np.array([_k_conv_k(ri, km, km, km1, U, xo, xi) for ri in r])
    kpp = np.array([_k_conv_k(ri, kp, kp, kp1, U, xo, xi) for ri in r])
    kmp = np.array([_k_conv_k(ri, km, kp, kp1, U, xo, xi) for ri in r])
    gg = _glgl(r, sigma, Gamma, rho)
    hpp = L - 2.0 * tkm + kmm          # h++ * h++
    hmm = L - 2.0 * tkp + kpp          # h+- * h+-
    hpm = L - tkp - tkm + kmp          # h++ * h+-
    Tpp = 0.5 * rho * (hpp + hmm - 2.0 * gg)
    Tpm = rho * (hpm + gg)
    return Tpp, Tpm


def _correlation_table(sigma, Gamma, rho):
    """Nodes r in [1, rcut] with h, T and H: graded Gauss-Legendre panels on [1, 2] and [2, rcut] whose
    quadratic substitutions on both sides of r = 2 absorb the (2 - r)^(3/2) cusp of the lens area."""
    wspl = _w_spline(sigma, Gamma, rho)
    U = _rcut(Gamma, rho, sigma)
    xw = _leggauss(16)
    r1, w1 = _graded_panels(1.0, 2.0, False, True, 0.5, xw)
    r2, w2 = _graded_panels(2.0, U, True, False, 0.5, xw)
    r = np.concatenate([r1, r2]); w = np.concatenate([w1, w2])
    wr = wspl(r); hpm = np.exp(wr) - 1.0; hpp = np.exp(-wr) - 1.0
    Tpp, Tpm = _T_values(r, sigma, Gamma, rho, wspl)
    Hpm = hpm + (hpm + 1.0) * Tpm; Hpp = hpp + (hpp + 1.0) * Tpp
    return dict(r=r, w=w, hpm=hpm, hpp=hpp, Tpp=Tpp, Tpm=Tpm, Hpm=Hpm, Hpp=Hpp, wspl=wspl)


def structure_factors(q: "np.ndarray", sigma: float, Gamma: float, rho: float) -> "np.ndarray":
    sigma = _check_sigma(sigma); Gamma, rho = _check_state(Gamma, rho)
    q = np.atleast_1d(np.asarray(q, dtype=float))
    if q.ndim != 1 or np.any(q <= 0.0) or np.any(q > 20.0) or not np.all(np.isfinite(q)):
        raise ValueError("q must lie in (0, 20]")
    tab = _correlation_table(sigma, Gamma, rho)
    core = -2.0 * np.pi * j1(q) / q
    Hpp = core + _hankel_forward(q, tab["r"], tab["w"], tab["Hpp"])
    Hpm = core + _hankel_forward(q, tab["r"], tab["w"], tab["Hpm"])
    Spp = 0.5 + 0.25 * rho * Hpp
    Spm = 0.25 * rho * Hpm
    Szz = 1.0 + 0.5 * rho * (Hpp - Hpm)
    return np.vstack([Spp, Spm, Szz])

import numpy as np
from scipy.special import kve, k0, k1, j0, j1
from scipy.optimize import brentq


def _check_state(Gamma, rho):
    Gamma = float(Gamma); rho = float(rho)
    if not (Gamma > 0.0) or not np.isfinite(Gamma):
        raise ValueError("Gamma must be positive")
    if not (rho > 0.0) or not np.isfinite(rho):
        raise ValueError("rho must be positive")
    return Gamma, rho


def _check_sigma(sigma):
    sigma = float(sigma)
    if not (sigma >= 0.0) or not np.isfinite(sigma):
        raise ValueError("sigma must be non-negative")
    return sigma


def _kv(n, z):
    z = np.asarray(z, dtype=complex)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        out = kve(n, z) * np.exp(-z)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _S(q, sigma):
    x = (sigma * q) ** 2
    return 1.0 + x + x ** 2 + x ** 3 + x ** 4


def _gl_partial_fractions(sigma, kap2):
    """Poles of 1/(x S(x) + kap2) in x = q^2: residues B_m and b_m = sqrt(-x_m) with Re b_m > 0."""
    if sigma == 0.0:
        return np.array([1.0 + 0j]), np.array([np.sqrt(kap2) + 0j])
    coef = np.array([kap2, 1.0, sigma ** 2, sigma ** 4, sigma ** 6, sigma ** 8], dtype=complex)
    x = np.polynomial.polynomial.polyroots(coef)
    B = 1.0 / np.polynomial.polynomial.polyval(x, np.polynomial.polynomial.polyder(coef))
    b = np.sqrt(-x)
    b = np.where(b.real < 0.0, -b, b)
    return B, b


def _gl_closed(u, sigma, Gamma, kap2):
    """Gl(u) = Gamma int_0^inf q J0(qu)/(q^2 S + kap2) dq = Gamma sum_m B_m K0(b_m u)."""
    B, b = _gl_partial_fractions(sigma, kap2)
    u = np.asarray(u, dtype=float)
    z = np.multiply.outer(u, b)
    out = Gamma * np.real(np.sum(B * _kv(0, z), axis=-1))
    zero = (u == 0.0)
    if np.any(zero):
        out = np.where(zero, -Gamma * np.real(np.sum(B * np.log(b))), out)
    return out


def _filter_poles():
    """The four non-real fifth roots of unity omega_k, the pole directions beta_k = sqrt(-omega_k) with
    positive real part and the residues c_k = 1/(omega_k P'(omega_k)) of 1 - 1/S in x = (sigma q)^2."""
    omega = np.exp(2j * np.pi * np.arange(1, 5) / 5.0)
    beta = np.exp(1j * np.pi * (np.arange(1, 5) / 5.0 - 0.5))
    ck = 1.0 / (omega * (1.0 + 2.0 * omega + 3.0 * omega ** 2 + 4.0 * omega ** 3))
    return beta, ck


def _vs_closed(u, sigma, Gamma):
    """vs(u) = Gamma int_0^inf (S-1)/(q S) J0(qu) dq = -Gamma sum_k c_k K0(beta_k u/sigma)."""
    u = np.asarray(u, dtype=float)
    if sigma == 0.0:
        return np.zeros_like(u)
    beta, ck = _filter_poles()
    z = np.multiply.outer(u / sigma, beta)
    return -Gamma * np.real(np.sum(ck * _kv(0, z), axis=-1))


def _w_total(u, sigma, Gamma, rho):
    kap2 = 2.0 * np.pi * Gamma * rho
    return _vs_closed(u, sigma, Gamma) + _gl_closed(u, sigma, Gamma, kap2)


def _leggauss(n):
    """Gauss-Legendre nodes and weights on [-1, 1], rebuilt on every call (no module-level cache)."""
    return np.polynomial.legendre.leggauss(n)


def _gl_nodes(a, b, xw):
    x, w = xw
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def _panel_edges(a, b, width, ratio=1.12, far=6.0):
    if b - a <= 0.0:
        return np.array([a, b])
    if b <= far:
        m = max(1, int(np.ceil((b - a) / width)))
        return np.linspace(a, b, m + 1)
    e = [a]
    while e[-1] < far - 1e-12 and e[-1] < b:
        e.append(min(e[-1] + width, far, b))
    w = width
    while e[-1] < b - 1e-12:
        w *= ratio
        e.append(min(e[-1] + w, b))
    return np.array(e)


def _panel_nodes(edges, n):
    xw = _leggauss(n); xs = []; ws = []
    for a, b in zip(edges[:-1], edges[1:]):
        x, w = _gl_nodes(a, b, xw); xs.append(x); ws.append(w)
    return np.concatenate(xs), np.concatenate(ws)


def _umax(Gamma, rho):
    kap = np.sqrt(2.0 * np.pi * Gamma * rho)
    return 1.0 + max(30.0, 45.0 / kap)


def _rcut(Gamma, rho, sigma):
    """Radial cutoff of the correlation integrals: the distance at which the slowest exponential tail of the
    kernels (the screened poles b_m of Gl and the filter poles beta_k/sigma of vs) has decayed by 13 decades,
    never beyond _umax."""
    kap2 = 2.0 * np.pi * Gamma * rho
    B, b = _gl_partial_fractions(sigma, kap2)
    lam = float(np.min(b.real))
    if sigma > 0.0:
        beta, _ = _filter_poles()
        lam = min(lam, float(np.min(beta.real)) / sigma)
    return min(_umax(Gamma, rho), 1.0 + 13.0 * np.log(10.0) / lam)


def _w_spline(sigma, Gamma, rho):
    from scipy.interpolate import CubicSpline
    U = _umax(Gamma, rho) + 3.0
    ug = np.linspace(1e-6, U, int(U / 0.002) + 1)
    return CubicSpline(ug, _w_total(ug, sigma, Gamma, rho))


def _chord_A(r, r1, g, g1, xw):
    """A_g(r, r1) = int_0^{2 pi} g(d(phi)) dphi with d^2 = r^2 + r1^2 - 2 r r1 cos(phi), g(d) = 0 for d < 1,
    written as 4 int g(d) d dd / sqrt((d^2 - dmin^2)(dmax^2 - d^2)) with the endpoint singularities removed;
    xw holds the Gauss-Legendre nodes and weights of the angular quadrature."""
    x, w = xw
    th = 0.5 * np.pi * (x + 1.0); wth = 0.5 * np.pi * w
    th2 = 0.25 * np.pi * (x + 1.0); wth2 = 0.25 * np.pi * w
    r1 = np.asarray(r1, dtype=float)
    dmin = np.abs(r - r1); dmax = r + r1
    out = np.zeros_like(r1)
    m1 = dmin >= 1.0
    if m1.any():
        xmin = dmin[m1] ** 2; xmax = dmax[m1] ** 2
        xmid = 0.5 * (xmin + xmax); xh = 0.5 * (xmax - xmin)
        xx = xmid[:, None] - xh[:, None] * np.cos(th)[None, :]
        out[m1] = 2.0 * np.sum(wth * g(np.sqrt(xx)), axis=1)
    m2 = (dmin < 1.0) & (dmax > 1.0)
    if m2.any():
        xmin = dmin[m2] ** 2; xmax = dmax[m2] ** 2
        a = xmax - 1.0; eps = 1.0 - xmin
        c = np.cos(th2)[None, :]
        xx = xmax[:, None] - a[:, None] * np.sin(th2)[None, :] ** 2
        D = np.sqrt(a[:, None] * c ** 2 + eps[:, None])
        integ = np.sum(wth2 * (g(np.sqrt(xx)) - g1) * c / D, axis=1)
        sing = g1 * np.arcsin(1.0 / np.sqrt(1.0 + eps / a)) / np.sqrt(a)
        out[m2] = 4.0 * np.sqrt(a) * (sing + integ)
    return out


def _graded_panels(a, b, kink_lo, kink_hi, width, xw):
    edges = _panel_edges(a, b, width); m = len(edges) - 1
    x, w = xw; xs = []; ws = []
    for i, (p, q) in enumerate(zip(edges[:-1], edges[1:])):
        L = q - p
        if i == 0 and kink_lo:
            t = 0.5 * (x + 1.0); wt = 0.5 * w
            xs.append(p + L * t * t); ws.append(wt * 2.0 * L * t)
        elif i == m - 1 and kink_hi:
            t = 0.5 * (x + 1.0); wt = 0.5 * w
            xs.append(q - L * t * t); ws.append(wt * 2.0 * L * t)
        else:
            xs.append(0.5 * L * x + 0.5 * (p + q)); ws.append(0.5 * L * w)
    return np.concatenate(xs), np.concatenate(ws)


def _theta_conv_k(r, g, g1, xw_out, xw_in, width=0.25):
    """(theta * k)(r), r >= 1: int_0^1 r1 dr1 A_g(r, r1); theta = unit-disc indicator, k = g on d >= 1;
    xw_out and xw_in are the Gauss-Legendre nodes of the outer (r1) and inner (angular) quadratures."""
    if r >= 2.0:
        r1, w = _graded_panels(0.0, 1.0, False, False, width, xw_out)
        return float(np.sum(w * r1 * _chord_A(r, r1, g, g1, xw_in)))
    rk = r - 1.0; tot = 0.0
    if rk > 0.0:
        r1, w = _graded_panels(0.0, rk, False, False, width, xw_out); tot += np.sum(w * r1 * _chord_A(r, r1, g, g1, xw_in))
    r1, w = _graded_panels(rk, 1.0, True, False, width, xw_out); tot += np.sum(w * r1 * _chord_A(r, r1, g, g1, xw_in))
    return float(tot)


def _k_conv_k(r, fa, gb, gb1, rcut, xw_out, xw_in, width=0.25):
    """(k_a * k_b)(r) = int_1^rcut r1 k_a(r1) A_b(r, r1) dr1, sqrt cusps at r1 = r -/+ 1 handled."""
    pts = sorted(set([1.0] + [x for x in (r - 1.0, r + 1.0) if 1.0 < x < rcut] + [rcut]))
    tot = 0.0
    for p, q in zip(pts[:-1], pts[1:]):
        klo = abs(p - (r - 1.0)) < 1e-14
        khi = abs(q - (r + 1.0)) < 1e-14
        r1, w = _graded_panels(p, q, klo, khi, width, xw_out)
        tot += np.sum(w * r1 * fa(r1) * _chord_A(r, r1, gb, gb1, xw_in))
    return float(tot)


def _lens(r):
    r = np.asarray(r, dtype=float); out = np.zeros_like(r); m = r < 2.0
    out[m] = 2.0 * np.arccos(r[m] / 2.0) - 0.5 * r[m] * np.sqrt(4.0 - r[m] ** 2)
    return out


def _glgl(r, sigma, Gamma, rho):
    """(Gl * Gl)(r) = int dq q/(2 pi) J0(q r) Glbar(q)^2 (smooth, fast-decaying integrand)."""
    kap2 = 2.0 * np.pi * Gamma * rho
    qmax = max(60.0, 40.0 / max(sigma, 0.05))
    q, w = _panel_nodes(np.linspace(0.0, qmax, int(qmax / 0.05) + 1), 8)
    gb = (2.0 * np.pi * Gamma / (q * q * _S(q, sigma) + kap2)) ** 2
    r = np.atleast_1d(np.asarray(r, dtype=float)); out = np.empty_like(r)
    for i in range(0, len(r), 64):
        rr = r[i:i + 64]
        out[i:i + 64] = np.sum(w * q * gb * j0(np.outer(rr, q)), axis=1) / (2.0 * np.pi)
    return out


def _T_values(r, sigma, Gamma, rho, wspl):
    r = np.atleast_1d(np.asarray(r, dtype=float))
    kp = lambda d: np.exp(wspl(d)) - 1.0      # h+- for d >= 1
    km = lambda d: np.exp(-wspl(d)) - 1.0     # h++ for d >= 1
    kp1 = float(kp(1.0)); km1 = float(km(1.0))
    U = _rcut(Gamma, rho, sigma)
    L = _lens(r)
    xo = _leggauss(16); xi = _leggauss(32)
    tkp = np.array([_theta_conv_k(ri, kp, kp1, xo, xi) for ri in r])
    tkm = np.array([_theta_conv_k(ri, km, km1, xo, xi) for ri in r])
    kmm = np.array([_k_conv_k(ri, km, km, km1, U, xo, xi) for ri in r])
    kpp = np.array([_k_conv_k(ri, kp, kp, kp1, U, xo, xi) for ri in r])
    kmp = np.array([_k_conv_k(ri, km, kp, kp1, U, xo, xi) for ri in r])
    gg = _glgl(r, sigma, Gamma, rho)
    hpp = L - 2.0 * tkm + kmm          # h++ * h++
    hmm = L - 2.0 * tkp + kpp          # h+- * h+-
    hpm = L - tkp - tkm + kmp          # h++ * h+-
    Tpp = 0.5 * rho * (hpp + hmm - 2.0 * gg)
    Tpm = rho * (hpm + gg)
    return Tpp, Tpm


def _correlation_table(sigma, Gamma, rho):
    """Nodes r in [1, rcut] with h, T and H: graded Gauss-Legendre panels on [1, 2] and [2, rcut] whose
    quadratic substitutions on both sides of r = 2 absorb the (2 - r)^(3/2) cusp of the lens area."""
    wspl = _w_spline(sigma, Gamma, rho)
    U = _rcut(Gamma, rho, sigma)
    xw = _leggauss(16)
    r1, w1 = _graded_panels(1.0, 2.0, False, True, 0.5, xw)
    r2, w2 = _graded_panels(2.0, U, True, False, 0.5, xw)
    r = np.concatenate([r1, r2]); w = np.concatenate([w1, w2])
    wr = wspl(r); hpm = np.exp(wr) - 1.0; hpp = np.exp(-wr) - 1.0
    Tpp, Tpm = _T_values(r, sigma, Gamma, rho, wspl)
    Hpm = hpm + (hpm + 1.0) * Tpm; Hpp = hpp + (hpp + 1.0) * Tpp
    return dict(r=r, w=w, hpm=hpm, hpp=hpp, Tpp=Tpp, Tpm=Tpm, Hpm=Hpm, Hpp=Hpp, wspl=wspl)


def excess_energy(Gamma: float, rho: float, sigma: float) -> "np.ndarray":
    if not np.isfinite(float(sigma)) or float(sigma) < 0.0:
        raise ValueError("sigma must be non-negative")
    sigma = _check_sigma(sigma); Gamma, rho = _check_state(Gamma, rho)
    tab = _correlation_table(sigma, Gamma, rho)
    r, w = tab["r"], tab["w"]
    pref = 0.5 * np.pi * Gamma * rho
    E0 = pref * np.sum(w * r * np.log(r) * (tab["hpm"] - tab["hpp"]))
    E1 = pref * np.sum(w * r * np.log(r) * (tab["Hpm"] - tab["Hpp"]))
    kap = np.sqrt(2.0 * np.pi * Gamma * rho)
    return np.array([E0, E1, 0.5 * Gamma * k0(kap)])

import numpy as np
from scipy.special import kve, k0, k1, j0, j1
from scipy.optimize import brentq


def _check_state(Gamma, rho):
    Gamma = float(Gamma); rho = float(rho)
    if not (Gamma > 0.0) or not np.isfinite(Gamma):
        raise ValueError("Gamma must be positive")
    if not (rho > 0.0) or not np.isfinite(rho):
        raise ValueError("rho must be positive")
    return Gamma, rho


def _energy_full(Gamma, rho):
    s = splitting_length(Gamma, rho)
    E = excess_energy(Gamma, rho, s)
    return float(E[1]), float(E[0]), s


def coulomb_liquid_audit(Gamma: float, rho: float, gamma_lo: float, gamma_hi: float) -> "np.ndarray":
    Gamma, rho = _check_state(Gamma, rho)
    gamma_lo = float(gamma_lo); gamma_hi = float(gamma_hi)
    if not (0.0 < gamma_lo < gamma_hi) or not np.isfinite(gamma_hi):
        raise ValueError("bracket must satisfy 0 < gamma_lo < gamma_hi")
    s = splitting_length(Gamma, rho)
    K = filter_kernels(np.array([1e-8, 2.0]), s, Gamma, rho)
    if abs(K[2, 0] * rho - 1.0) > 1e-6:
        raise RuntimeError("screened kernel limit violated")
    gl0 = float(long_range_kernel(np.array([0.0]), s, Gamma, rho)[0])
    w1 = float(short_range_potential(np.array([1.0]), s, Gamma)[0] + long_range_kernel(np.array([1.0]), s, Gamma, rho)[0])
    hc = mayer_functions(np.array([1.0]), s, Gamma, rho)
    res07 = variational_residual(0.7, Gamma, rho)
    E = excess_energy(Gamma, rho, s)
    h = 0.02 * Gamma
    d1 = (_energy_full(Gamma + h, rho)[0] - _energy_full(Gamma - h, rho)[0]) / (2.0 * h)
    d2 = (_energy_full(Gamma + 0.5 * h, rho)[0] - _energy_full(Gamma - 0.5 * h, rho)[0]) / h
    dE = (4.0 * d2 - d1) / 3.0
    cv = E[1] - Gamma * dE
    kap = np.sqrt(2.0 * np.pi * Gamma * rho)
    cv_dh = 0.25 * Gamma * kap * k1(kap)
    Sq = structure_factors(np.array([0.5, 2.0]), s, Gamma, rho)
    T1 = pair_convolutions(np.array([1.0]), s, Gamma, rho)
    f = lambda g: _energy_full(g, rho)[0]
    flo, fhi = f(gamma_lo), f(gamma_hi)
    if flo * fhi > 0.0:
        raise RuntimeError("no sign change on the bracket")
    g0 = brentq(f, gamma_lo, gamma_hi, xtol=1e-10, rtol=1e-12, maxiter=100)
    s0 = splitting_length(g0, rho)
    audit = np.array([[s, E[0], E[1], E[2]],
                      [cv, cv_dh, Sq[1, 1], Sq[2, 0]],
                      [g0, s0, w1, T1[0, 0]],
                      [gl0, res07, hc[0, 0], hc[1, 0]]])
    return audit
SCICODE_GOLD_EOF
