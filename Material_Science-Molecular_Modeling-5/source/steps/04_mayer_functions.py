"""
Return the Mayer functions of the short-range scheme at the non-negative distances u for a pair of opposite unit charges and for a pair of like unit charges: both equal minus one inside the hard core u < 1, and outside it each is the Boltzmann factor of the total pair potential vs(u) + Gl(u) of the two previous steps minus one, with the sign convention that opposite charges attract, so their Mayer function is positive at contact when the potential is positive. Reject invalid parameters and negative or non-finite distances.

In the mixed expansion the strongly coupled short-range electrostatics and the hard core enter through a Mayer function built from the screened long-range kernel plus the short-range remainder, evaluated exactly, while only the long-range fluctuations are treated as Gaussian. The hard core is the disk of unit diameter around each ion, so both Mayer functions are exactly minus one there.

Returns
-------
A (2, n) float64 array whose first row is the opposite-charge Mayer function and whose second row is the like-charge Mayer function at the n distances.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mayer_functions(u: "np.ndarray", sigma: float, Gamma: float, rho: float) -> "np.ndarray":
    """Return the Mayer functions of the short-range scheme at the non-negative distances u for a
    pair of opposite unit charges and for a pair of like unit charges: both equal minus one
    inside the hard core u < 1, and outside it each is the Boltzmann factor of the total pair
    potential vs(u) + Gl(u) of the two previous steps minus one, with the sign convention that
    opposite charges attract, so their Mayer function is positive at contact when the potential
    is positive. Reject invalid parameters and negative or non-finite distances.

    Args:
        u: array-like of shape (n,) of non-negative finite distances in units of the disk diameter.
        sigma: non-negative finite float, the splitting length in units of the disk diameter (sigma = 0 is the
            Debye-Hueckel limit of the scheme).
        Gamma: positive finite float, the Coulomb coupling.
        rho: positive finite float, the total reduced ion density rho a^2 (both species together).

    Returns:
        A numpy float64 array of shape (2, n): the first row is the opposite-charge Mayer function
        h+-(u) = exp(+w(u)) - 1 and the second row the like-charge Mayer function h++(u) = exp(-w(u)) - 1,
        with w = vs + Gl the total pair potential of the two previous steps, for u >= 1; both rows equal -1
        inside the hard core u < 1.

    Raises:
        ValueError: if u is not a one-dimensional array of non-negative finite values; if sigma is negative
            or not finite; or if Gamma or rho is not a positive finite number.
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


def _w_total(u, sigma, Gamma, rho):
    kap2 = 2.0 * np.pi * Gamma * rho
    return _vs_closed(u, sigma, Gamma) + _gl_closed(u, sigma, Gamma, kap2)


def _oracle_mayer_functions(u: "np.ndarray", sigma: float, Gamma: float, rho: float) -> "np.ndarray":
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {'setup': 'import numpy as np\nGamma, rho = 2.0, 0.10\nu = np.array([0.0, 0.5, 1.0, 1.5, 3.0, 8.0])\nsigma = 0.7\n',
         'call': 'mayer_functions(u, sigma, Gamma, rho)',
         'gold_call': '_oracle_mayer_functions(u, sigma, Gamma, rho)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\nGamma, rho = 1.25, 0.15\nu = np.linspace(1.0, 5.0, 9)\nsigma = 0.75\n',
         'call': 'mayer_functions(u, sigma, Gamma, rho)',
         'gold_call': '_oracle_mayer_functions(u, sigma, Gamma, rho)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\nGamma, rho = 4.0, 0.20\nu = np.array([0.99, 1.0, 1.01, 2.0])\nsigma = 0.6\n# boundary: distances straddling the contact distance u = 1\n',
         'call': 'mayer_functions(u, sigma, Gamma, rho)',
         'gold_call': '_oracle_mayer_functions(u, sigma, Gamma, rho)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\nGamma, rho = 0.5, 0.05\nu = np.array([1.0, 2.0, 6.0])\nsigma = 1.1\n# edge: a splitting length above the disk diameter\n',
         'call': 'mayer_functions(u, sigma, Gamma, rho)',
         'gold_call': '_oracle_mayer_functions(u, sigma, Gamma, rho)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\nGamma, rho = 2.0, 0.10\nu = np.array([0.5, 1.0])\nsigma = -0.1\n# invalid input: a negative splitting length must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: mayer_functions(u, sigma, Gamma, rho))',
         'gold_call': '_catches_value_error(lambda: _oracle_mayer_functions(u, sigma, Gamma, rho))'},
    ]
