"""
Builds the truncated Fourier-basis matrix of the Bloch-Fokker-Planck operator of an overdamped particle in a tilted cosine potential.

Every later quantity is a spectral property of this one matrix, so the placement of the corrugation couplings and of the tilt on its diagonal decides the whole chain.

Returns
-------
A float64 array of shape (2, N, N).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def wb_operator(u: float, f: float, q: float, n_max: int) -> np.ndarray:
    r"""u: non-negative float, the corrugation amplitude $U_1/k_BT$. f: float, the reduced tilt $FL/(2\pi k_BT)$; a
    positive f drives the particle towards $+x$. q: float, the Bloch wave number in units of $1/L$. n_max: non-negative
    integer, the Fourier truncation.

    Units are $L = 1$, $D = 1$, $k_BT = 1$ throughout. The particle diffuses in the tilted cosine potential
    $\beta U(x) = u\cos(2\pi x) - 2\pi f x$ under the Smoluchowski equation $\partial_t p = \partial_x[\partial_x p +
    p\,\partial_x(\beta U)]$. On Bloch functions $e^{iqx}w(x)$ with $w(x) = \sum_\nu \langle\nu|w\rangle e^{2\pi i\nu x}$ the
    operator acts as a matrix $\langle\mu|\mathcal L_q|\nu\rangle$ on the Fourier coefficients of $w$, tridiagonal
    in $\mu-\nu$, for $\mu, \nu = -n_{max}..n_{max}$.

    Returns a numpy float64 array of shape $(2, N, N)$ with $N = 2n_{max}+1$: the real part and the imaginary part of the
    truncated matrix, row and column index $\mu + n_{max}$.

    Raises:
        ValueError: on a non-finite input, a negative u, or a non-integer or negative n_max.
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

def _oracle_wb_operator(u: float, f: float, q: float, n_max: int) -> np.ndarray:
    uu = _nonneg(u, "u"); ff = _fin(f, "f"); qq = _fin(q, "q"); nm = _int(n_max, "n_max")
    L = _operator(uu, ff, qq, nm)
    if not np.all(np.isfinite(L)): raise ValueError("non-finite operator")
    return np.array([L.real, L.imag], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\n","call":"wb_operator(1.0,0.6,0.3,3)","gold_call":"_oracle_wb_operator(1.0,0.6,0.3,3)"},
        {"setup":"import numpy as np\n","call":"wb_operator(10.0,9.0,-1.2,4)","gold_call":"_oracle_wb_operator(10.0,9.0,-1.2,4)"},
        {"setup":"import numpy as np\n","call":"wb_operator(3.0,0.0,np.pi,2)","gold_call":"_oracle_wb_operator(3.0,0.0,np.pi,2)"},
        {"setup":"import numpy as np\n# boundary: no corrugation, where the operator is diagonal with the free drift-diffusion eigenvalues\n","call":"wb_operator(0.0,0.5,0.7,2)","gold_call":"_oracle_wb_operator(0.0,0.5,0.7,2)"},
        {"setup":"import numpy as np\n# invalid input: a negative corrugation amplitude must raise ValueError\ndef _probe(fn, *args):\n    try:\n        fn(*args)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n","call":"_probe(wb_operator,-1.0,0.5,0.0,3)","gold_call":"_probe(_oracle_wb_operator,-1.0,0.5,0.0,3)"},
    ]
