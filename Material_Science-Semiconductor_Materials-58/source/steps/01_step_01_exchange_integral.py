"""
Step 01 - Exchange integral with the two-dimensional layered Coulomb kernel.

Exchange integral of an isotropic momentum-space function with the two-dimensional Coulomb kernel of a layered
semiconductor.

The two-band model of the heterobilayer places the conduction band in a layer a distance d above the valence band
(d = 0 for a monolayer). Its Coulomb matrix elements are V_q = 4 pi exp(-q d) / q in excitonic units, where
lengths are measured in the exciton Bohr radius a_B*, energies in the exciton Rydberg Ry* (so hbar^2/2m = 1 for
the reduced mass m), and (1/A) sum_k stands for int d^2k / (2 pi)^2. Every self-energy and every bound-state
equation of the model is built from the exchange integral

    (V_d f)(k) = int d^2k' / (2 pi)^2  V_|k - k'|  f(k'),

taken of an isotropic function f(|k'|) that is smooth and even in k'. This step computes that integral for a
user-supplied callable f at requested magnitudes k, converged to a relative accuracy of 1e-9 at every requested
point. The angular integral over the direction of k' has no closed form for d > 0 and the monolayer part of the
kernel has an integrable logarithmic singularity at |k'| = k, which sets the numerical difficulty. Do not assume
any particular functional form for f beyond smoothness, evenness and decay; the tests use functions with power-law
tails as well as Gaussians.

Inputs: f, a callable mapping a numpy array of non-negative wavevector magnitudes to an array of the same shape;
k_eval, a one-dimensional array of non-negative magnitudes (0 allowed); d >= 0. Output: (V_d f)(k_eval) as an
array of the same shape as k_eval. Raises ValueError if d is negative or not finite, if k_eval is not a
one-dimensional array of finite non-negative numbers, or if f does not return finite values of the right shape.

Returns
-------
numpy.ndarray, same shape as k_eval: (V_d f)(k_eval) in Ry* times the units of f
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from math import comb
from typing import Callable
from scipy.special import ellipk, ellipe
from numpy.polynomial.legendre import leggauss


def exchange_integral(f: Callable, k_eval: np.ndarray, d: float) -> np.ndarray:
    '''Exchange integral (V_d f)(k) = int d^2k'/(2 pi)^2 V_|k-k'| f(|k'|) with V_q = 4 pi exp(-q d)/q.

    Parameters
    ----------
    f : Callable
        Isotropic, smooth, even function of the wavevector magnitude, called with a numpy array of
        non-negative magnitudes and returning finite values of the same shape. Decays at least as fast
        as |k'|^-3 at large |k'|.
    k_eval : np.ndarray
        One-dimensional array of finite non-negative magnitudes at which the integral is wanted (0 allowed).
    d : float
        Interlayer distance in units of a_B*, >= 0 (0 is the monolayer, kernel 4 pi/q).

    Returns
    -------
    result : np.ndarray
        Same shape as k_eval, (V_d f)(k_eval) in units of Ry* times the units of f, converged to a relative
        accuracy of 1e-9 at every point.

    Raises
    ------
    ValueError
        If d is negative or not finite, if k_eval is not a one-dimensional array of finite non-negative
        numbers, or if f returns non-finite values or an array of the wrong shape.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from math import comb
from typing import Callable
from scipy.special import ellipk, ellipe
from numpy.polynomial.legendre import leggauss


def _check_scalar(value, name, positive=False, nonneg=False):
    """Return value as float, raising ValueError unless it is a finite real number with the requested sign."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(name + " must be a real number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(name + " must be finite")
    if positive and value <= 0.0:
        raise ValueError(name + " must be positive")
    if nonneg and value < 0.0:
        raise ValueError(name + " must be non-negative")
    return value


def _check_k_array(k, name):
    """Return k as a one-dimensional float array of finite non-negative values, else raise ValueError."""
    k = np.asarray(k, dtype=float)
    if k.ndim != 1 or k.size == 0 or not np.all(np.isfinite(k)) or np.any(k < 0.0):
        raise ValueError(name + " must be a one-dimensional array of finite non-negative numbers")
    return k


def _grid(_GRID={}):
    """Composite Gauss-Legendre radial grid: [0, 2^-10] then doubling panels up to 2^20, 16 nodes per panel (cached)."""
    if "k" not in _GRID:
        x, wref = leggauss(16)
        edges = [0.0] + [2.0 ** p for p in range(-10, 21)]
        ks, ws = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            h = (b - a) / 2
            ks.append(a + h * (x + 1))
            ws.append(wref * h)
        _GRID.update(k=np.concatenate(ks), w=np.concatenate(ws), edges=np.array(edges), x=x, wref=wref,
                     C=np.linalg.inv(np.vander(x, 16, increasing=True)))
    return _GRID


def _log_moments(ui, n):
    """mu_p = int_{-1}^{1} u^p ln|u - ui| du for p < n, in closed form."""
    def _F(x, q):
        return 0.0 if x == 0 else x ** (q + 1) * (np.log(abs(x)) - 1.0 / (q + 1)) / (q + 1)
    hi, lo = 1.0 - ui, -1.0 - ui
    base = [_F(hi, q) - _F(lo, q) for q in range(n)]
    return np.array([sum(comb(p, q) * ui ** (p - q) * base[q] for q in range(p + 1)) for p in range(n)])


def _monolayer_rows(kp, k):
    """Angular average of 4 pi/|k - k'| times k'/(2 pi) for rows kp and nodes k: kappa = 4k'K(m)/(pi s), s = k + k',
    m = 4kk'/s^2, together with the analytic coefficient c = 8k'P4(1 - m)/(pi^2 s) of its ln|k - k'| singularity
    (P4 is the fourth-order Taylor polynomial of K at zero argument). Coincident points get their limiting values."""
    K1, K2 = np.meshgrid(kp, k, indexing="ij")
    s = K1 + K2
    m = 4 * K1 * K2 / s ** 2
    m1 = 1 - m
    with np.errstate(divide="ignore", invalid="ignore"):
        kap = 4 * K2 * ellipk(m) / (np.pi * s)
    series = (1.0, 1 / 4, 9 / 64, 25 / 256, 1225 / 16384)
    poly = (np.pi / 2) * sum(series[j] * m1 ** j for j in range(5))
    c = 8 * K2 * poly / (np.pi ** 2 * s)
    same = np.abs(K1 - K2) <= 1e-14 * np.maximum(K1, 1e-300)
    kap = np.where(same, (2 / np.pi) * np.log(8 * np.maximum(K1, 1e-300)), kap)
    c = np.where(same, 2 / np.pi, c)
    return np.where(np.isfinite(kap), kap, 0.0), np.where(np.isfinite(c), c, 0.0), same


def _bilayer_rows(kp, k, d, kap0, same, npsi=48):
    """Interlayer correction kappa_d - kappa_0 = (k'/pi) int dtheta (exp(-d q) - 1)/q with q = |k - k'|: the powers of q
    up to q^4 are subtracted and integrated in closed form (angular moments in K and E), the remainder is integrated by
    Gauss-Legendre in psi with q^2 = (k - k')^2 + 4kk' sin^2(psi). Also returns the coefficient of the ln|k - k'| term."""
    K1, K2 = np.meshgrid(kp, k, indexing="ij")
    s = K1 + K2
    a = K1 ** 2 + K2 ** 2
    b = 2 * K1 * K2
    m = 4 * K1 * K2 / s ** 2
    qm = np.abs(K1 - K2)
    nsub = np.where(d * s <= 12.0, 5, np.where(d * s <= 40.0, 3, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        E = ellipe(m)
        K = ellipk(m)
    Kf = np.where(np.isfinite(K), K, 0.0)
    I1 = 4 * s * E
    I2 = 2 * np.pi * a
    I3 = (4 * s / 3) * (4 * a * E - (K1 - K2) ** 2 * Kf)
    I4 = 2 * np.pi * a ** 2 + np.pi * b ** 2
    analytic = (-d * 2 * np.pi + np.where(nsub >= 3, d ** 2 / 2 * I1 - d ** 3 / 6 * I2, 0.0)
                + np.where(nsub >= 5, d ** 4 / 24 * I3 - d ** 5 / 120 * I4, 0.0))
    xp, wp = leggauss(npsi)
    psi = (np.pi / 4) * (xp + 1)
    wpsi = wp * (np.pi / 4)
    acc = np.zeros_like(K1)
    for p_, wq in zip(psi, wpsi):
        q = np.sqrt(qm ** 2 + 2 * b * np.sin(p_) ** 2)
        dq = d * q
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            r1 = (np.exp(-dq) - 1 + dq) / q
            r3 = r1 + (-dq ** 2 / 2 + dq ** 3 / 6) / q
            r5 = r3 + (-dq ** 4 / 24 + dq ** 5 / 120) / q
        rem = np.where(nsub == 5, r5, np.where(nsub == 3, r3, r1))
        acc += wq * np.where(q > 0, rem, 0.0)
    reg = (K2 / np.pi) * (analytic + 4 * acc)
    reg = np.where((d * qm > 40.0) & ~same, -kap0, reg)
    m1 = 1 - m
    with np.errstate(divide="ignore", invalid="ignore"):
        Km1 = ellipk(m1)
        Em1 = ellipe(m1)
        Kp = (Em1 - (1 - m1) * Km1) / (2 * m1 * (1 - m1))
    Kp = np.where(m1 > 1e-12, Kp, np.pi / 8)
    gE = (Km1 * m1 - 2 * (1 - m1) * m1 * Kp) / np.pi
    creg = (K2 / np.pi) * s * (np.where(nsub >= 3, 4 * d ** 2 * gE, 0.0)
                               + np.where(nsub >= 5, (d ** 4 / 9) * (4 * a * gE - (K1 - K2) ** 2 * Km1 / np.pi), 0.0))
    return np.where(np.isfinite(reg), reg, 0.0), np.where(np.isfinite(creg), creg, 0.0)


def _nystrom_rows(kp, kern, coef, near=0.35):
    """Quadrature rows for a kernel kern(k, k') = -coef(k, k') ln|k - k'| + smooth on the grid nodes: plain Gauss-Legendre
    weights away from the singular point, exact log moments of the Lagrange basis on the panel containing k and on a
    neighbour panel whenever k lies within `near` panel widths of the shared edge."""
    g = _grid()
    k, w, edges, wref, C = g["k"], g["w"], g["edges"], g["wref"], g["C"]
    n = 16
    P = len(edges) - 1
    Mm = kern * w[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        R = kern + coef * np.log(np.abs(kp[:, None] - k[None, :]))
    R = np.where(np.isfinite(R), R, kern)
    for i, ki in enumerate(kp):
        p = int(min(np.searchsorted(edges, ki, side="right") - 1, P - 1))
        a, b = edges[p], edges[p + 1]
        panels = [p]
        if p > 0 and (ki - a) < near * (b - a):
            panels.append(p - 1)
        if p < P - 1 and (b - ki) < near * (b - a):
            panels.append(p + 1)
        for pj in panels:
            aj, bj = edges[pj], edges[pj + 1]
            hj = (bj - aj) / 2
            ui = (ki - (aj + hj)) / hj
            Wlog = hj * (C.T @ _log_moments(ui, n)) + hj * np.log(hj) * wref
            sl = slice(pj * n, (pj + 1) * n)
            Mm[i, sl] = w[sl] * R[i, sl] - coef[i, sl] * Wlog
    return Mm


def _rows(kp, d):
    """Rows of the discretised exchange operator V_d for evaluation points kp (each >= 2^-10) over the full grid."""
    k = _grid()["k"]
    kap0, c0, same = _monolayer_rows(kp, k)
    Mm = _nystrom_rows(kp, kap0, c0)
    if d > 0:
        reg, creg = _bilayer_rows(kp, k, d, kap0, same)
        Mm = Mm + _nystrom_rows(kp, reg, creg)
    return Mm


def _operator(d, _KERNELS={}):
    """Square discretised operator on the trusted nodes (k >= 2^-10). The innermost panel is integrated over, its
    unknowns being represented by an even-polynomial extrapolation from the trusted nodes below 0.1. Returns a dict with
    the trusted nodes k, the measure mu for int d^2k/(2 pi)^2, and the matrix M with (V_d f)_i = sum_j M_ij f_j."""
    key = round(float(d), 12)
    if key in _KERNELS:
        return _KERNELS[key]
    g = _grid()
    k, w = g["k"], g["w"]
    n = 16
    N = len(k)
    M = _rows(k[n:], d)
    J = np.arange(n)
    T = np.arange(n, N)
    kT = k[T]
    fit = T[kT < 0.1][:48]
    P = np.linalg.pinv(np.vander(k[fit] ** 2, 4, increasing=True))
    L = np.zeros((n, N - n))
    L[:, fit - n] = np.vander(k[J] ** 2, 4, increasing=True) @ P
    Mred = M[:, T] + M[:, J] @ L
    mu = w * k / (2 * np.pi)
    op = dict(k=kT, mu=mu[T] + L.T @ mu[J], M=Mred, L=L, fit=fit - n)
    _KERNELS[key] = op
    return op


def _even_fit(k, f, kmax=0.1, deg=3):
    """Least-squares coefficients of c0 + c1 k^2 + ... + c_deg k^(2 deg) through the samples with k < kmax."""
    sel = k < kmax
    X = np.vander(k[sel] ** 2, deg + 1, increasing=True)
    c, *_ = np.linalg.lstsq(X, f[sel], rcond=None)
    return c


def _apply_to_grid_values(fvals, k_eval, d):
    """(V_d f)(k_eval) from values of f on the full grid: direct quadrature rows for k_eval >= 2^-10, and an even
    polynomial fitted to the operator's values on the trusted nodes below 0.1 for smaller k_eval."""
    g = _grid()
    out = np.empty_like(k_eval)
    big = k_eval >= 2.0 ** -10
    if np.any(big):
        out[big] = _rows(k_eval[big], d) @ fvals
    if np.any(~big):
        kf = g["k"][16:][g["k"][16:] < 0.1]
        c = _even_fit(kf, _rows(kf, d) @ fvals)
        out[~big] = np.polynomial.polynomial.polyval(k_eval[~big] ** 2, c)
    return out


def _oracle_exchange_integral(f: Callable, k_eval: np.ndarray, d: float) -> np.ndarray:
    """Reference implementation."""
    d = _check_scalar(d, "d", nonneg=True)
    k_eval = _check_k_array(k_eval, "k_eval")
    k = _grid()["k"]
    fvals = np.asarray(f(k.copy()), dtype=float)
    if fvals.shape != k.shape or not np.all(np.isfinite(fvals)):
        raise ValueError("f must return finite values with the shape of its argument")
    return _apply_to_grid_values(fvals, k_eval, d)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: monolayer, hydrogenic 1s wave function; (V phi)(k) = (k^2 + 4) phi(k) exactly ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "phi = lambda q: np.sqrt(2*np.pi)/(1 + q*q/4)**1.5\n"
                     "k_eval = np.array([0.0, 0.05, 0.3, 1.0, 2.5, 7.0, 30.0])\n",
            "call": "exchange_integral(phi, k_eval.copy(), 0.0) / ((k_eval**2 + 4)*phi(k_eval))",
            "gold_call": "_oracle_exchange_integral(phi, k_eval.copy(), 0.0) / ((k_eval**2 + 4)*phi(k_eval))",
            "tol": 1e-8,
        },
        # --- Normal: bilayer d = 0.25 acting on the same 1s function ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "phi = lambda q: np.sqrt(2*np.pi)/(1 + q*q/4)**1.5\n"
                     "k_eval = np.array([0.0, 0.02, 0.4, 1.3, 3.0, 12.0])\n",
            "call": "exchange_integral(phi, k_eval.copy(), 0.25)",
            "gold_call": "_oracle_exchange_integral(phi, k_eval.copy(), 0.25)",
            "tol": 1e-8,
        },
        # --- Boundary: a Gaussian with the bilayer kernel at a large separation (fast interlayer decay) ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "g = lambda q: np.exp(-0.7*q*q)\n"
                     "k_eval = np.array([0.001, 0.2, 0.9, 2.0, 4.0])\n",
            "call": "exchange_integral(g, k_eval.copy(), 1.0)",
            "gold_call": "_oracle_exchange_integral(g, k_eval.copy(), 1.0)",
            "tol": 1e-8,
        },
        # --- Edge: monolayer Gaussian, closed form sqrt(pi) exp(-k^2/2) I0(k^2/2), evaluated near the singular scale ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe, i0e\n"
                     "g = lambda q: np.exp(-q*q)\n"
                     "k_eval = np.array([0.0, 0.0009, 0.011, 0.5, 1.5, 3.0, 5.0])\n",
            "call": "exchange_integral(g, k_eval.copy(), 0.0) / (np.sqrt(np.pi)*i0e(k_eval**2/2))",
            "gold_call": "_oracle_exchange_integral(g, k_eval.copy(), 0.0) / (np.sqrt(np.pi)*i0e(k_eval**2/2))",
            "tol": 1e-8,
        },
        # --- Edge: a squared 1s function (k^-6 tail) with an intermediate separation ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "g = lambda q: 2*np.pi/(1 + q*q/4)**3\n"
                     "k_eval = np.array([0.0, 0.7, 1.9, 6.0])\n",
            "call": "exchange_integral(g, k_eval.copy(), 0.5)",
            "gold_call": "_oracle_exchange_integral(g, k_eval.copy(), 0.5)",
            "tol": 1e-8,
        },
        # --- Invalid: negative separation must raise ValueError (0 returned, 1 raised) ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "phi = lambda q: np.sqrt(2*np.pi)/(1 + q*q/4)**1.5\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(phi, np.array([0.5, 1.0]), -0.1)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    return 0\n",
            "call": "_probe(exchange_integral)",
            "gold_call": "_probe(_oracle_exchange_integral)",
        },
    ]
