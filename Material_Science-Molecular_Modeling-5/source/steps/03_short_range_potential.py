"""
Return the short-range part of the Coulomb potential at the positive distances u: the bare potential -Gamma ln u minus the unscreened long-range potential of step 1, which is Gamma times the integral over q from zero to infinity of (S(q) - 1)/(q S(q)) times J0(q u). This integral converges only conditionally at large q, so a truncated numerical quadrature is not acceptable: the result must be accurate to a relative error of 1e-10 for 0.1 <= u <= 10, must vanish identically when sigma = 0, must carry the logarithmic singularity of the bare potential as u tends to zero, and must decay to zero at distances large compared with sigma. Reject invalid parameters and distances that are not positive and finite.

The filter removes the short wavelengths from the long-range part, so the remainder is a potential of range sigma that is treated exactly, through its Mayer function, rather than at the Gaussian level. Because 1 - 1/S(q) is a rational function of q^2 with four simple poles at the non-real fifth roots of unity scaled by sigma, the remainder is a finite sum of modified Bessel functions of complex argument whose coefficients do not depend on sigma at all; only the arguments scale with u/sigma.

Returns
-------
A (n,) float64 array holding the short-range potential at the n distances.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def short_range_potential(u: "np.ndarray", sigma: float, Gamma: float) -> "np.ndarray":
    """Return the short-range part of the Coulomb potential at the positive distances u: the bare
    potential -Gamma ln u minus the unscreened long-range potential of step 1, which is Gamma
    times the integral over q from zero to infinity of (S(q) - 1)/(q S(q)) times J0(q u). This
    integral converges only conditionally at large q, so a truncated numerical quadrature is not
    acceptable: the result must be accurate to a relative error of 1e-10 for 0.1 <= u <= 10,
    must vanish identically when sigma = 0, must carry the logarithmic singularity of the bare
    potential as u tends to zero, and must decay to zero at distances large compared with sigma.
    Reject invalid parameters and distances that are not positive and finite.

    Args:
        u: array-like of shape (n,) of positive finite distances in units of the disk diameter.
        sigma: non-negative finite float, the splitting length in units of the disk diameter (sigma = 0 makes
            the short-range part vanish identically).
        Gamma: positive finite float, the Coulomb coupling.

    Returns:
        A numpy float64 array of shape (n,) holding the short-range potential vs(u) = Gamma times the integral
        over q from 0 to infinity of (S(q) - 1)/(q S(q)) J0(q u), equal to -Gamma ln u minus the unscreened
        long-range potential, at the n distances, accurate to a relative error of 1e-10 for 0.1 <= u <= 10.

    Raises:
        ValueError: if u is not a one-dimensional array of positive finite values; if sigma is negative or not
            finite; or if Gamma is not a positive finite number.
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


def _oracle_short_range_potential(u: "np.ndarray", sigma: float, Gamma: float) -> "np.ndarray":
    sigma = _check_sigma(sigma); Gamma, _ = _check_state(Gamma, 1.0)
    u = np.atleast_1d(np.asarray(u, dtype=float))
    if u.ndim != 1 or np.any(u <= 0.0) or not np.all(np.isfinite(u)):
        raise ValueError("u must be positive")
    return _vs_closed(u, sigma, Gamma)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {'setup': 'import numpy as np\nGamma, rho = 2.0, 0.10\nu = np.array([0.1, 0.5, 1.0, 2.0, 3.0, 5.0])\nsigma = 0.7\n',
         'call': 'short_range_potential(u, sigma, Gamma)',
         'gold_call': '_oracle_short_range_potential(u, sigma, Gamma)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\nGamma, rho = 1.25, 0.15\nu = np.linspace(1.0, 5.0, 5)\nsigma = 0.75\n',
         'call': 'short_range_potential(u, sigma, Gamma)',
         'gold_call': '_oracle_short_range_potential(u, sigma, Gamma)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\nGamma, rho = 4.0, 0.20\nu = np.array([0.5, 1.0, 3.0])\nsigma = 0.0\n# boundary: sigma = 0, the short-range part vanishes identically\n',
         'call': 'short_range_potential(u, sigma, Gamma)',
         'gold_call': '_oracle_short_range_potential(u, sigma, Gamma)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\nGamma, rho = 0.5, 0.05\nu = np.array([0.25, 1.0, 2.5])\nsigma = 1.1\n# edge: a splitting length above the disk diameter\n',
         'call': 'short_range_potential(u, sigma, Gamma)',
         'gold_call': '_oracle_short_range_potential(u, sigma, Gamma)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\nGamma, rho = 2.0, 0.10\nu = np.array([0.0, 1.0])\nsigma = 0.7\n# invalid input: a zero distance must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: short_range_potential(u, sigma, Gamma))',
         'gold_call': '_catches_value_error(lambda: _oracle_short_range_potential(u, sigma, Gamma))'},
    ]
