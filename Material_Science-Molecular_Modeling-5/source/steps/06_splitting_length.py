"""
Return the splitting length sigma at which the variational residual of the previous step vanishes, for the given coupling and density. Bracket the sign change of the residual by scanning splitting lengths between 0.05 and 5 disk diameters, then converge the root to an absolute tolerance of 1e-12 with a bracketing root finder; the residual has a single sign change on that range for the states considered here. Reject invalid parameters and raise an error if no sign change is found.

The splitting length is the only free parameter of the scheme and every downstream quantity depends on it: the Mayer functions, the correlation corrections, the energy and its coupling derivative. Its variational value is of the order of the disk diameter and decreases slowly with the coupling and with the density.

Returns
-------
A Python float, the variational splitting length in units of the disk diameter.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def splitting_length(Gamma: float, rho: float) -> float:
    """Return the splitting length sigma at which the variational residual of the previous step
    vanishes, for the given coupling and density. Bracket the sign change of the residual by
    scanning splitting lengths between 0.05 and 5 disk diameters, then converge the root to an
    absolute tolerance of 1e-12 with a bracketing root finder; the residual has a single sign
    change on that range for the states considered here. Reject invalid parameters and raise an
    error if no sign change is found.

    Args:
        Gamma: positive finite float, the Coulomb coupling.
        rho: positive finite float, the total reduced ion density rho a^2.

    Returns:
        A Python float, the variational splitting length sigma* in units of the disk diameter: the root of the
        variational residual of the previous step, bracketed on a scan of [0.05, 5] and converged to an absolute
        tolerance of 1e-12.

    Raises:
        ValueError: if Gamma or rho is not a positive finite number.
        RuntimeError: if the residual has no sign change on [0.05, 5].
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_splitting_length(Gamma: float, rho: float) -> float:
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {'setup': 'import numpy as np\nGamma, rho = 2.0, 0.10\n',
         'call': 'splitting_length(Gamma, rho)',
         'gold_call': '_oracle_splitting_length(Gamma, rho)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\nGamma, rho = 1.25, 0.15\n',
         'call': 'splitting_length(Gamma, rho)',
         'gold_call': '_oracle_splitting_length(Gamma, rho)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\nGamma, rho = 4.0, 0.20\n# boundary: the strongest coupling and highest density of the test set\n',
         'call': 'splitting_length(Gamma, rho)',
         'gold_call': '_oracle_splitting_length(Gamma, rho)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\nGamma, rho = 0.5, 0.05\n# edge: a weak coupling and dilute density, where sigma* approaches one diameter\n',
         'call': 'splitting_length(Gamma, rho)',
         'gold_call': '_oracle_splitting_length(Gamma, rho)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\nGamma, rho = -2.0, 0.10\n# invalid input: a negative coupling must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: splitting_length(Gamma, rho))',
         'gold_call': '_catches_value_error(lambda: _oracle_splitting_length(Gamma, rho))'},
    ]
