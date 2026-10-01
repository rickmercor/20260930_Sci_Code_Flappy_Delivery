"""
Evaluate Riccati spherical Bessel and incoming/outgoing Hankel data.

Let psi_l(x)=x*j_l(x) and xi_+(x)=x*h_l^(1)(x),

xi_-(x)=x*h_l^(2)(x). Primes differentiate x. These radial functions obey

f''=[l(l+1)/x^2-1]f. The outgoing convention is exp(+i*x), corresponding to

time dependence exp(-i*omega*t). The array's final axis lists each function

with its first and second derivative; all other axes are the evaluation grid.

For integer l the Hankel functions may be evaluated by their finite

exponential-polynomial forms, avoiding cancellation between j_l and y_l.

Returns
-------
complex128 ndarray, shape x.shape + (9,), radial functions and their first two argument derivatives
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def riccati_wave_data(ell: int, x: "np.ndarray") -> "np.ndarray":
    """Evaluate radial waves and two argument derivatives.

    Parameters
    ----------
    ell : int
        Angular momentum, 1 through 4.
    x : ndarray
        Complex evaluation points; 1e-8 <= abs(x) <= 100 and
        abs(Im(x)) <= 20. A scalar array or any array shape is allowed.

    Returns
    -------
    ndarray
        Complex128 array of shape x.shape + (9,), ordered as
        psi, psi', psi'', xi_+, xi_+', xi_+'', xi_-, xi_-', xi_-''.

    Raises
    ------
    ValueError
        If ell is outside 1 through 4 or any x is zero.

    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np
from scipy.special import spherical_jn


def _oracle_riccati_wave_data(ell: int, x: "np.ndarray") -> "np.ndarray":
    if ell not in (1, 2, 3, 4):
        raise ValueError("ell must be 1, 2, 3 or 4")
    x = np.asarray(x, dtype=complex)
    if np.any(x == 0):
        raise ValueError("zero argument is outside this radial-wave contract")
    psi = x * spherical_jn(ell, x)
    dpsi = spherical_jn(ell, x) + x * spherical_jn(ell, x, derivative=True)
    ode = ell * (ell + 1) / x**2 - 1
    data = [psi, dpsi, ode * psi]
    powers = np.arange(ell + 1)
    for sign in (1, -1):
        coeff = np.array(
            [
                math.factorial(ell + j)
                / (math.factorial(j) * math.factorial(ell - j))
                * (sign * 0.5j) ** j
                for j in range(ell + 1)
            ]
        )
        poly = np.sum(coeff * x[..., None] ** (-powers), axis=-1)
        dpoly = np.sum(
            -powers * coeff * x[..., None] ** (-powers - 1), axis=-1
        )
        phase = (-sign * 1j) ** (ell + 1) * np.exp(sign * 1j * x)
        xi = phase * poly
        dxi = phase * (sign * 1j * poly + dpoly)
        data.extend([xi, dxi, ode * xi])
    return np.stack(data, axis=-1)

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
                "x = np.array([0.35, 2.4, 7.4])\n"
                "\n"
                "\n"
                "def components(value):\n"
                "    magnitude = np.abs(value)\n"
                "    scale = np.maximum(1.0, magnitude)\n"
                "    return np.stack(\n"
                "        (magnitude, value.real / scale, "
                "value.imag / scale),\n"
                "        axis=-1,\n"
                "    )\n"
            ),
            "call": ("components(riccati_wave_data(3, x.copy()))\n"),
            "gold_call": (
                "components(_oracle_riccati_wave_data(3, x.copy()))\n"
            ),
            "tol": 1e-09,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "x = np.array(1.0 + 2.0j)\n"
                "\n"
                "\n"
                "def components(value):\n"
                "    magnitude = np.abs(value)\n"
                "    scale = np.maximum(1.0, magnitude)\n"
                "    return np.stack(\n"
                "        (magnitude, value.real / scale, "
                "value.imag / scale),\n"
                "        axis=-1,\n"
                "    )\n"
            ),
            "call": ("components(riccati_wave_data(1, x.copy()))\n"),
            "gold_call": (
                "components(_oracle_riccati_wave_data(1, x.copy()))\n"
            ),
            "tol": 1e-09,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "x = np.array([[-3.0 - 2j, 3.0 - 2j], [1e-4, 20j]])\n"
                "\n"
                "\n"
                "def components(value):\n"
                "    magnitude = np.abs(value)\n"
                "    scale = np.maximum(1.0, magnitude)\n"
                "    return np.stack(\n"
                "        (magnitude, value.real / scale, "
                "value.imag / scale),\n"
                "        axis=-1,\n"
                "    )\n"
            ),
            "call": ("components(riccati_wave_data(4, x.copy()))\n"),
            "gold_call": (
                "components(_oracle_riccati_wave_data(4, x.copy()))\n"
            ),
            "tol": 1e-09,
        },
        {
            "setup": (
                "def raises(fn):\n"
                "    try:\n"
                "        fn(3, 0.0)\n"
                "    except ValueError:\n"
                "        return 1.0\n"
                "    return 0.0\n"
            ),
            "call": ("raises(riccati_wave_data)\n"),
            "gold_call": ("raises(_oracle_riccati_wave_data)\n"),
            "tol": 1e-09,
        },
    ]
