"""
Return the residual of the variational identity that fixes the splitting length: the integral over q from zero to infinity of q times [hbar++(q) - hbar+-(q) + 2 Glbar(q)] times the sigma-derivative of the long-range potential, where hbar++ and hbar+- are the two-dimensional Fourier transforms of the like-charge and opposite-charge Mayer functions of the previous step, including the hard-core disk whose transform is -2 pi J1(q)/q, and Glbar is the screened kernel of step 1. The residual must be accurate to 1e-8 in absolute terms for a positive sigma; it may be evaluated in reciprocal space or, by Parseval's theorem, as an equivalent radial integral in real space, but its value and sign must be those of the reciprocal-space definition. Reject invalid parameters and a non-positive splitting length.

The grand potential is invariant under the splitting, so its stationarity with respect to sigma is an exact identity that becomes a condition on sigma once the correlators are approximated. At the order kept here it pairs the Fourier transform of the charge-antisymmetric Mayer combination, corrected by twice the screened kernel, with the sigma-derivative of the long-range potential; the residual is negative for a splitting length that is too short and positive for one that is too long.

Returns
-------
A Python float, the value of the residual.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def variational_residual(sigma: float, Gamma: float, rho: float) -> float:
    """Return the residual of the variational identity that fixes the splitting length: the
    integral over q from zero to infinity of q times [hbar++(q) - hbar+-(q) + 2 Glbar(q)] times
    the sigma-derivative of the long-range potential, where hbar++ and hbar+- are the two-
    dimensional Fourier transforms of the like-charge and opposite-charge Mayer functions of the
    previous step, including the hard-core disk whose transform is -2 pi J1(q)/q, and Glbar is
    the screened kernel of step 1. The residual must be accurate to 1e-8 in absolute terms for a
    positive sigma; it may be evaluated in reciprocal space or, by Parseval's theorem, as an
    equivalent radial integral in real space, but its value and sign must be those of the
    reciprocal-space definition. Reject invalid parameters and a non-positive splitting length.

    Args:
        sigma: positive finite float, the trial splitting length in units of the disk diameter.
        Gamma: positive finite float, the Coulomb coupling.
        rho: positive finite float, the total reduced ion density rho a^2.

    Returns:
        A Python float, the residual of the variational identity in its reciprocal-space definition: the
        integral over q from 0 to infinity of q [hbar++(q) - hbar+-(q) + 2 Glbar(q)] d vlbar/d sigma, with the
        two-dimensional transforms of the Mayer functions including the hard-core disk (transform
        -2 pi J1(q)/q); accurate to 1e-8 in absolute terms. It is negative for a splitting length below the
        variational value and positive above it.

    Raises:
        ValueError: if sigma is not a positive finite number (sigma = 0 is rejected here); or if Gamma or rho
            is not a positive finite number.
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


def _oracle_variational_residual(sigma: float, Gamma: float, rho: float) -> float:
    sigma = _check_sigma(sigma); Gamma, rho = _check_state(Gamma, rho)
    if sigma == 0.0:
        raise ValueError("sigma must be positive")
    return float(_variational_real(sigma, Gamma, rho))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {'setup': 'import numpy as np\nGamma, rho = 2.0, 0.10\nsigma = 0.7\n',
         'call': 'variational_residual(sigma, Gamma, rho)',
         'gold_call': '_oracle_variational_residual(sigma, Gamma, rho)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nGamma, rho = 2.0, 0.10\nsigma = 0.3\n# boundary: a short splitting length, where the residual is negative\n',
         'call': 'variational_residual(sigma, Gamma, rho)',
         'gold_call': '_oracle_variational_residual(sigma, Gamma, rho)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nGamma, rho = 1.25, 0.15\nsigma = 1.0\n# edge: a splitting length above the variational value, where the residual is positive\n',
         'call': 'variational_residual(sigma, Gamma, rho)',
         'gold_call': '_oracle_variational_residual(sigma, Gamma, rho)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nGamma, rho = 4.0, 0.20\nsigma = 0.55\n',
         'call': 'variational_residual(sigma, Gamma, rho)',
         'gold_call': '_oracle_variational_residual(sigma, Gamma, rho)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nGamma, rho = 2.0, 0.10\nsigma = 0.0\n# invalid input: a zero splitting length must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: variational_residual(sigma, Gamma, rho))',
         'gold_call': '_catches_value_error(lambda: _oracle_variational_residual(sigma, Gamma, rho))'},
    ]
