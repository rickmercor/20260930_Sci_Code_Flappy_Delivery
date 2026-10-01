"""
Return the real-space screened long-range kernel Gl(u) at the non-negative distances u: the inverse two-dimensional Fourier transform of the screened kernel of the previous step, that is Gamma times the integral over q from zero to infinity of q J0(q u) divided by q^2 S(q) + kappa0^2. The result must be accurate to a relative error of 1e-10 at every distance, including u = 0 where the kernel is finite, and must reduce to the Debye-Hueckel kernel Gamma K0(kappa0 u) when sigma = 0. The integrand is a rational function of q^2 whose denominator is a polynomial of degree five in q^2 with positive coefficients, so no oscillatory quadrature is needed if that structure is used. Reject invalid parameters and negative or non-finite distances.

Screening the filtered long-range potential by the ionic atmosphere at the Gaussian level gives a kernel whose Fourier representation has simple poles in q^2 at the roots of q^2 S(q) + kappa0^2. Each pole contributes a modified Bessel function of the second kind with a complex argument whose real part must be taken positive, and the residues sum to zero, which removes the logarithmic singularity at the origin: the kernel is finite at contact and at zero separation, unlike the bare Debye-Hueckel kernel.

Returns
-------
A (n,) float64 array holding Gl at the n distances.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def long_range_kernel(u: "np.ndarray", sigma: float, Gamma: float, rho: float) -> "np.ndarray":
    """Return the real-space screened long-range kernel Gl(u) at the non-negative distances u: the
    inverse two-dimensional Fourier transform of the screened kernel of the previous step, that
    is Gamma times the integral over q from zero to infinity of q J0(q u) divided by q^2 S(q) +
    kappa0^2. The result must be accurate to a relative error of 1e-10 at every distance,
    including u = 0 where the kernel is finite, and must reduce to the Debye-Hueckel kernel
    Gamma K0(kappa0 u) when sigma = 0. The integrand is a rational function of q^2 whose
    denominator is a polynomial of degree five in q^2 with positive coefficients, so no
    oscillatory quadrature is needed if that structure is used. Reject invalid parameters and
    negative or non-finite distances.

    Args:
        u: array-like of shape (n,) of non-negative finite distances in units of the disk diameter (u = 0 is
            allowed; the kernel is finite there for sigma > 0).
        sigma: non-negative finite float, the splitting length in units of the disk diameter (sigma = 0 is the
            Debye-Hueckel limit of the scheme).
        Gamma: positive finite float, the Coulomb coupling.
        rho: positive finite float, the total reduced ion density rho a^2 (both species together).

    Returns:
        A numpy float64 array of shape (n,) holding the screened long-range kernel Gl(u) = Gamma times the
        integral over q from 0 to infinity of q J0(q u)/(q^2 S(q) + kappa0^2), at the n distances, accurate to a
        relative error of 1e-10; at sigma = 0 it equals Gamma K0(kappa0 u).

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


def _oracle_long_range_kernel(u: "np.ndarray", sigma: float, Gamma: float, rho: float) -> "np.ndarray":
    sigma = _check_sigma(sigma); Gamma, rho = _check_state(Gamma, rho)
    u = np.atleast_1d(np.asarray(u, dtype=float))
    if u.ndim != 1 or np.any(u < 0.0) or not np.all(np.isfinite(u)):
        raise ValueError("u must be non-negative")
    kap2 = 2.0 * np.pi * Gamma * rho
    return _gl_closed(u, sigma, Gamma, kap2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {'setup': 'import numpy as np\nGamma, rho = 2.0, 0.10\nu = np.array([0.0, 0.5, 1.0, 2.0, 5.0])\nsigma = 0.7\n',
         'call': 'long_range_kernel(u, sigma, Gamma, rho)',
         'gold_call': '_oracle_long_range_kernel(u, sigma, Gamma, rho)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\nGamma, rho = 1.25, 0.15\nu = np.linspace(1.0, 12.0, 7)\nsigma = 0.75\n',
         'call': 'long_range_kernel(u, sigma, Gamma, rho)',
         'gold_call': '_oracle_long_range_kernel(u, sigma, Gamma, rho)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\nGamma, rho = 4.0, 0.20\nu = np.array([0.2, 1.0, 3.0])\nsigma = 0.0\n# boundary: sigma = 0, the Debye-Hueckel kernel Gamma K0(kappa0 u)\n',
         'call': 'long_range_kernel(u, sigma, Gamma, rho)',
         'gold_call': '_oracle_long_range_kernel(u, sigma, Gamma, rho)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\nGamma, rho = 0.5, 0.05\nu = np.array([0.0, 1.0, 4.0, 10.0])\nsigma = 1.1\n# edge: u = 0 included, where the kernel is finite\n',
         'call': 'long_range_kernel(u, sigma, Gamma, rho)',
         'gold_call': '_oracle_long_range_kernel(u, sigma, Gamma, rho)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\nGamma, rho = 2.0, 0.10\nu = np.array([-0.5, 1.0])\nsigma = 0.7\n# invalid input: a negative distance must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: long_range_kernel(u, sigma, Gamma, rho))',
         'gold_call': '_catches_value_error(lambda: _oracle_long_range_kernel(u, sigma, Gamma, rho))'},
    ]
