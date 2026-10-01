"""
Return, for a one-dimensional array of positive wavenumbers q (lengths are measured in the disk diameter, so q is dimensionless), the four reciprocal-space building blocks of the splitting scheme at splitting length sigma, coupling Gamma and total reduced ion density rho: the filter polynomial S(q) = 1 + (sigma q)^2 + (sigma q)^4 + (sigma q)^6 + (sigma q)^8; the long-range potential 2 pi Gamma/(q^2 S(q)); the screened long-range kernel 2 pi Gamma/(q^2 S(q) + kappa0^2), where kappa0^2 = 2 pi Gamma rho is the Debye-Hueckel parameter of the full ion density; and the derivative of the long-range potential with respect to sigma at fixed q. Reject a coupling or density that is not positive and finite, a negative splitting length, and any wavenumber that is not positive and finite.

The two-dimensional Coulomb potential between unit charges is -Gamma ln(r/a) and its Fourier transform is 2 pi Gamma/q^2. The self-consistent scheme splits it into a long-range part, obtained by dividing that transform by the eighth-order filter S(q), and a short-range remainder; the long-range part is then screened at the Gaussian level, which adds kappa0^2 to the denominator. The splitting length is a variational parameter, so the derivative of the long-range potential with respect to it is needed as well.

Returns
-------
A (4, n) float64 array whose rows hold, in order, S(q), the long-range potential, the screened kernel and the sigma-derivative of the long-range potential at the n wavenumbers.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def filter_kernels(q: "np.ndarray", sigma: float, Gamma: float, rho: float) -> "np.ndarray":
    """Return, for a one-dimensional array of positive wavenumbers q (lengths are measured in the
    disk diameter, so q is dimensionless), the four reciprocal-space building blocks of the
    splitting scheme at splitting length sigma, coupling Gamma and total reduced ion density
    rho: the filter polynomial S(q) = 1 + (sigma q)^2 + (sigma q)^4 + (sigma q)^6 + (sigma q)^8;
    the long-range potential 2 pi Gamma/(q^2 S(q)); the screened long-range kernel 2 pi
    Gamma/(q^2 S(q) + kappa0^2), where kappa0^2 = 2 pi Gamma rho is the Debye-Hueckel parameter
    of the full ion density; and the derivative of the long-range potential with respect to
    sigma at fixed q. Reject a coupling or density that is not positive and finite, a negative
    splitting length, and any wavenumber that is not positive and finite.

    Args:
        q: array-like of shape (n,) of positive finite wavenumbers in units of the inverse disk diameter.
        sigma: non-negative finite float, the splitting length in units of the disk diameter (sigma = 0 is the
            Debye-Hueckel limit of the scheme).
        Gamma: positive finite float, the Coulomb coupling.
        rho: positive finite float, the total reduced ion density rho a^2 (both species together).

    Returns:
        A numpy float64 array of shape (4, n) whose rows hold, in order, the filter S(q), the long-range
        potential 2 pi Gamma/(q^2 S(q)), the screened kernel 2 pi Gamma/(q^2 S(q) + kappa0^2) with
        kappa0^2 = 2 pi Gamma rho, and the derivative of the long-range potential with respect to sigma at
        fixed q, all evaluated at the n wavenumbers.

    Raises:
        ValueError: if q is not a one-dimensional array of positive finite values; if sigma is negative or
            not finite; or if Gamma or rho is not a positive finite number.
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


def _S(q, sigma):
    x = (sigma * q) ** 2
    return 1.0 + x + x ** 2 + x ** 3 + x ** 4


def _dS(q, sigma):
    q2 = q * q
    return 2.0 * sigma * q2 + 4.0 * sigma ** 3 * q2 ** 2 + 6.0 * sigma ** 5 * q2 ** 3 + 8.0 * sigma ** 7 * q2 ** 4


def _oracle_filter_kernels(q: "np.ndarray", sigma: float, Gamma: float, rho: float) -> "np.ndarray":
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {'setup': 'import numpy as np\nGamma, rho = 2.0, 0.10\nq = np.array([0.1, 0.5, 2.0, 3.0])\nsigma = 0.7\n',
         'call': 'filter_kernels(q, sigma, Gamma, rho)',
         'gold_call': '_oracle_filter_kernels(q, sigma, Gamma, rho)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\nGamma, rho = 1.25, 0.15\nq = np.geomspace(0.05, 3.0, 7)\nsigma = 0.75\n',
         'call': 'filter_kernels(q, sigma, Gamma, rho)',
         'gold_call': '_oracle_filter_kernels(q, sigma, Gamma, rho)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\nGamma, rho = 4.0, 0.20\nq = np.array([1.0, 3.0])\nsigma = 0.0\n# boundary: sigma = 0, the filter is identically one\n',
         'call': 'filter_kernels(q, sigma, Gamma, rho)',
         'gold_call': '_oracle_filter_kernels(q, sigma, Gamma, rho)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\nGamma, rho = 0.5, 0.05\nq = np.array([0.3, 1.2, 2.5])\nsigma = 1.1\n# edge: a splitting length above the disk diameter\n',
         'call': 'filter_kernels(q, sigma, Gamma, rho)',
         'gold_call': '_oracle_filter_kernels(q, sigma, Gamma, rho)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\nGamma, rho = 2.0, 0.10\nq = np.array([0.0, 1.0])\nsigma = 0.7\n# invalid input: a zero wavenumber must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: filter_kernels(q, sigma, Gamma, rho))',
         'gold_call': '_catches_value_error(lambda: _oracle_filter_kernels(q, sigma, Gamma, rho))'},
    ]
