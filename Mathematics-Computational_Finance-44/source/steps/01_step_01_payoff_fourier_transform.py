"""
Evaluate the generalized Fourier transform of a European call payoff. The transform is required on a horizontal line of the complex plane whose imaginary part is strongly negative, where its magnitude can be smaller than 1e-300 while the intermediate factors are not.

Generalized Fourier pricing separates the terminal payoff from the model characteristic function. For a call, the transform is analytic only below the payoff pole at an imaginary part of -1; evaluating its complex power in log form preserves double-precision accuracy on deeply damped contours.

Returns
-------
numpy.ndarray Complex array of shape (n,) holding P(xi).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def payoff_fourier_transform(xi_real, xi_imag, strike):
    """Return the payoff transform on a horizontal line of the complex plane.

    For a European call with the given strike the transform is

        P(xi) = -strike ** (1 - 1j * xi) / (xi ** 2 + 1j * xi),

    evaluated at ``xi = xi_real + 1j * xi_imag``.  The value must remain
    accurate when the line lies deep in the lower half plane, for example
    ``strike = 1000`` with ``xi_imag = -100``, so the complex power must not be
    routed through an intermediate quantity that leaves the double-precision
    range.

    Parameters
    ----------
    xi_real : array_like
        Real parts of the evaluation points, interpreted as a one-dimensional
        sequence of length ``n``.  All entries must be finite.
    xi_imag : array_like
        Imaginary parts, same shape as ``xi_real``.  All entries must be finite.
    strike : float
        Strictly positive strike.

    Returns
    -------
    numpy.ndarray
        Complex array of shape ``(n,)`` holding ``P(xi)``.

    Raises
    ------
    ValueError
        If ``xi_real`` and ``xi_imag`` have different shapes, if either holds a
        non-finite entry, if ``strike`` is not finite and strictly positive, or
        if any evaluation point is a zero of ``xi ** 2 + 1j * xi``.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _unpack_model(model):
    """Validate and unpack the six model parameters."""
    np = __import__("numpy")
    arr = np.asarray(model, dtype=float).reshape(-1)
    if arr.size != 6 or not bool(np.all(np.isfinite(arr))):
        raise ValueError("model must hold six finite numbers")
    alpha, gamma, nu, rho, v0, theta = (float(v) for v in arr)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("model[0] must satisfy 0 < alpha <= 1")
    if gamma <= 0.0 or nu <= 0.0:
        raise ValueError("model[1] and model[2] must be positive")
    if not -1.0 < rho < 1.0:
        raise ValueError("model[3] must satisfy abs(rho) < 1")
    if v0 <= 0.0 or theta <= 0.0:
        raise ValueError("model[4] and model[5] must be positive")
    return alpha, gamma, nu, rho, v0, theta


def _unpack_market(market):
    """Validate and unpack the four market parameters."""
    np = __import__("numpy")
    arr = np.asarray(market, dtype=float).reshape(-1)
    if arr.size != 4 or not bool(np.all(np.isfinite(arr))):
        raise ValueError("market must hold four finite numbers")
    spot, strike, rate, horizon = (float(v) for v in arr)
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("market[0] and market[1] must be positive")
    if horizon <= 0.0:
        raise ValueError("market[3] must be positive")
    return spot, strike, rate, horizon


def _positive_int(value, name):
    """Return ``value`` as a positive Python int or raise ValueError."""
    np = __import__("numpy")
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be a positive integer")
    if int(value) < 1:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _line_argument(xi_real, xi_imag):
    """Return the flattened complex vector xi_real + 1j*xi_imag."""
    np = __import__("numpy")
    a = np.atleast_1d(np.asarray(xi_real, dtype=float)).reshape(-1)
    b = np.atleast_1d(np.asarray(xi_imag, dtype=float)).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("xi_real and xi_imag must have the same shape")
    if not (bool(np.all(np.isfinite(a))) and bool(np.all(np.isfinite(b)))):
        raise ValueError("xi_real and xi_imag must be finite")
    return a + 1j * b
def _payoff_values(xi, strike):
    """Return -strike**(1 - 1j*xi)/(xi**2 + 1j*xi) without an out-of-range power."""
    np = __import__("numpy")
    math = __import__("math")
    denom = xi * xi + 1j * xi
    if bool(np.any(denom == 0.0)):
        raise ValueError("xi must avoid the zeros of xi**2 + 1j*xi")
    return -np.exp((1.0 - 1j * xi) * math.log(strike)) / denom


def _oracle_payoff_fourier_transform(xi_real, xi_imag, strike):
    """Reference implementation of the payoff transform."""
    np = __import__("numpy")
    strike = float(strike)
    if not np.isfinite(strike) or strike <= 0.0:
        raise ValueError("strike must be finite and positive")
    xi = _line_argument(xi_real, xi_imag)
    return _payoff_values(xi, strike)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return moderate, deep-damping, vectorized, and invalid-strike cases."""
    flat = (
        "def flat(z):\n"
        "    z = np.asarray(z, dtype=complex).reshape(-1)\n"
        "    return np.round(np.concatenate([np.log10(np.abs(z)), np.angle(z)]), 9).tolist()\n"
    )
    return [
        {
            "setup": flat + "xr = [0.5, 1.0, 2.5]\nxi = [-1.5, -1.5, -1.5]\nstrike = 2.0\n",
            "call": "flat(payoff_fourier_transform(xr, xi, strike))",
            "gold_call": "flat(_oracle_payoff_fourier_transform(xr, xi, strike))",
        },
        {
            "setup": flat + "xr = [0.0, 2.0, 7.5]\nxi = [-4.673, -4.673, -4.673]\nstrike = 1000.0\n",
            "call": "flat(payoff_fourier_transform(xr, xi, strike))",
            "gold_call": "flat(_oracle_payoff_fourier_transform(xr, xi, strike))",
        },
        {
            "setup": flat + "xr = [0.0, 3.25]\nxi = [-100.0, -100.0]\nstrike = 1000.0\n",
            "call": "flat(payoff_fourier_transform(xr, xi, strike))",
            "gold_call": "flat(_oracle_payoff_fourier_transform(xr, xi, strike))",
        },
        {
            "setup": flat + "xr = np.linspace(0.0, 5.0, 9)\nxi = np.full(9, -2.75)\nstrike = 90.0\n",
            "call": "flat(payoff_fourier_transform(xr, xi, strike))",
            "gold_call": "flat(_oracle_payoff_fourier_transform(xr, xi, strike))",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        payoff_fourier_transform([1.0], [-2.0], -5.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_payoff_fourier_transform([1.0], [-2.0], -5.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
