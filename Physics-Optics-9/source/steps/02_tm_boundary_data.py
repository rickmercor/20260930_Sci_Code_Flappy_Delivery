"""
Form the TM scattering numerator, outgoing determinant and its derivative.

For a lossless nonmagnetic sphere with real refractive index n in

vacuum, set p=psi_l(n*x), p1=psi_l'(n*x), p2=psi_l''(n*x).

The conventional scattering amplitude is N/D, where

D=p1*xi_+ - n*p*xi_+' and N=-p1*xi_- + n*p*xi_-'.

Its outgoing determinant derivative with respect to x is

D'=n*p2*xi_+ + (1-n^2)*p1*xi_+' - n*p*xi_+''.

The prime on psi denotes its own argument derivative, so the factors n

in D' are essential. Use the radial-wave data from the preceding step.

Returns
-------
complex128 ndarray, shape x.shape + (3,), the numerator N, outgoing determinant D and x derivative D'
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def tm_boundary_data(ell: int, n: float, x: "np.ndarray") -> "np.ndarray":
    """Evaluate the TM boundary determinant data.

    Parameters
    ----------
    ell : int
        Angular momentum, 1 through 4.
    n : float
        Real refractive index, 2 <= n <= 3.
    x : ndarray
        Nonzero complex points such that x and n*x satisfy the preceding
        radial-wave domain. Any array shape, including scalar, is allowed.

    Returns
    -------
    ndarray
        Complex128 array of shape x.shape + (3,), ordered N, D, D'.
        The amplitude N/D is not evaluated at its poles here.

    Raises
    ------
    ValueError
        If n is outside [2, 3], ell is unsupported, or an argument is zero.

    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _tm_small_argument_derivative(
    ell: int, n: float, x: "np.ndarray"
) -> "np.ndarray":
    """Evaluate D' from its regular series when abs(x) < 0.1."""
    # D = (-i)^(ell+1) n^ell exp(ix) P(x). Collect nonnegative
    # powers before differentiating the Bessel/Hankel product.
    coefficients = np.zeros(ell + 26, dtype=complex)
    regular = 1.0 / math.prod(range(1, 2 * ell + 2, 2))
    for k in range(12):
        for j in range(ell + 1):
            hankel = (
                math.factorial(ell + j)
                / (math.factorial(j) * math.factorial(ell - j))
                * (0.5j) ** j
            )
            power = ell + 2 * k - j
            coefficients[power] += (
                regular * hankel * (ell + 1 + 2 * k + n * n * j)
            )
            coefficients[power + 1] -= 1j * n * n * regular * hankel
        regular *= -n * n / (2 * (k + 1) * (2 * ell + 2 * k + 3))
    derivative = 1j * coefficients
    derivative[:-1] += np.arange(1, len(coefficients)) * coefficients[1:]
    return (
        (-1j) ** (ell + 1)
        * n**ell
        * np.exp(1j * x)
        * np.polynomial.polynomial.polyval(x, derivative)
    )


def _oracle_tm_boundary_data(
    ell: int, n: float, x: "np.ndarray"
) -> "np.ndarray":
    if not 2.0 <= n <= 3.0:
        raise ValueError("n must lie in [2, 3]")
    x = np.asarray(x, dtype=complex)
    interior = _oracle_riccati_wave_data(ell, n * x)
    outer = _oracle_riccati_wave_data(ell, x)
    p, p1, p2 = interior[..., 0], interior[..., 1], interior[..., 2]
    xp, xp1, xp2 = outer[..., 3], outer[..., 4], outer[..., 5]
    xm, xm1 = outer[..., 6], outer[..., 7]
    numerator = -p1 * xm + n * p * xm1
    denominator = p1 * xp - n * p * xp1
    derivative = n * p2 * xp + (1 - n**2) * p1 * xp1 - n * p * xp2
    small = np.abs(x) < 0.1
    if np.any(small):
        derivative = np.array(derivative, copy=True)
        derivative[small] = _tm_small_argument_derivative(ell, n, x[small])
    return np.stack([numerator, denominator, derivative], axis=-1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    return [
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "x = np.array([0.4, 2.4 - 0.047j, 6.5])\n"
                "\n"
                "\n"
                "def components(value):\n"
                "    return np.stack((value.real, value.imag), axis=-1)\n"
            ),
            "call": ("components(tm_boundary_data(3, 2.7, x.copy()))\n"),
            "gold_call": (
                "components(_oracle_tm_boundary_data(3, 2.7, x.copy()))\n"
            ),
            "tol": 1e-09,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "x = np.array(0.25 + 0.5j)\n"
                "\n"
                "\n"
                "def components(value):\n"
                "    return np.stack((value.real, value.imag), axis=-1)\n"
            ),
            "call": ("components(tm_boundary_data(1, 2.0, x.copy()))\n"),
            "gold_call": (
                "components(_oracle_tm_boundary_data(1, 2.0, x.copy()))\n"
            ),
            "tol": 1e-09,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "x = np.array([[-2.0 - 3j, 2.0 - 3j], [1e-4, 8.1]])\n"
                "\n"
                "\n"
                "def components(value):\n"
                "    return np.stack((value.real, value.imag), axis=-1)\n"
            ),
            "call": ("components(tm_boundary_data(4, 3.0, x.copy()))\n"),
            "gold_call": (
                "components(_oracle_tm_boundary_data(4, 3.0, x.copy()))\n"
            ),
            "tol": 1e-09,
        },
        {
            "setup": (
                "def raises(fn):\n"
                "    try:\n"
                "        fn(2, 1.0, 1.0)\n"
                "    except ValueError:\n"
                "        return 1.0\n"
                "    return 0.0\n"
            ),
            "call": ("raises(tm_boundary_data)\n"),
            "gold_call": ("raises(_oracle_tm_boundary_data)\n"),
            "tol": 1e-09,
        },
    ]
