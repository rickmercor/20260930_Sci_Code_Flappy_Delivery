"""
Find all derivative-normalization channel poles and their residues.

For the TM regularization T=[xi_+'(x)/xi_-'(x)]S(x), channel poles

are simple zeros c of xi_-', with residues S(c)*xi_+'(c)/xi_-''(c).

Here S=N/D uses the preceding boundary data; primes differentiate x.

Writing xi_-=i^(l+1)*exp(-i*x)*x^(-l)*P_l(x), the polynomial is

P_l(x)=sum_{j=0}^l (l+j)!/[j!(l-j)!]*(-i/2)^j*x^(l-j).

The channel poles are the l+1 roots of x*P_l'(x)-(l+i*x)*P_l(x).

This includes all channel poles in the upper half-plane and does not impose

the finite lower-half-plane physical-pole cutoff.

Returns
-------
complex128 ndarray, shape (ell + 1, 2), sorted channel poles and their x residues
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def tm_channel_data(ell: int, n: float) -> "np.ndarray":
    """Return all TM channel poles with dimensionless residues.

    Parameters
    ----------
    ell : int
        Angular momentum, 1 through 4.
    n : float
        Real refractive index in [2, 3].

    Returns
    -------
    ndarray
        Complex128 array of shape (ell + 1, 2). Each row is (c, residue).
        Sort by Re(c), then Im(c); set real parts smaller than 1e-9 in
        magnitude to zero. Poles are independent of n; residues are not.

    Raises
    ------
    ValueError
        If ell or n is outside its supported interval.

    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np


def _oracle_tm_channel_data(ell: int, n: float) -> "np.ndarray":
    if ell not in (1, 2, 3, 4) or not 2.0 <= n <= 3.0:
        raise ValueError("unsupported angular momentum or refractive index")
    polynomial = np.poly1d(
        [
            math.factorial(ell + j)
            / (math.factorial(j) * math.factorial(ell - j))
            * (-0.5j) ** j
            for j in range(ell + 1)
        ]
    )
    equation = (
        np.poly1d([1, 0]) * np.polyder(polynomial)
        - np.poly1d([1j, ell]) * polynomial
    )
    poles = np.roots(equation)
    poles.real[np.abs(poles.real) < 1e-9] = 0.0
    poles = poles[np.lexsort((poles.imag, poles.real))]
    radial = _oracle_riccati_wave_data(ell, poles)
    boundary = _oracle_tm_boundary_data(ell, n, poles)
    residues = boundary[:, 0] / boundary[:, 1] * radial[:, 4] / radial[:, 8]
    return np.column_stack([poles, residues])

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
                "\n"
                "def components(value):\n"
                "    return np.stack((value.real, value.imag), axis=-1)\n"
            ),
            "call": ("components(tm_channel_data(3, 2.7))\n"),
            "gold_call": ("components(_oracle_tm_channel_data(3, 2.7))\n"),
            "tol": 1e-09,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def components(value):\n"
                "    return np.stack((value.real, value.imag), axis=-1)\n"
            ),
            "call": ("components(tm_channel_data(1, 2.0))\n"),
            "gold_call": ("components(_oracle_tm_channel_data(1, 2.0))\n"),
            "tol": 1e-09,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def components(value):\n"
                "    return np.stack((value.real, value.imag), axis=-1)\n"
            ),
            "call": ("components(tm_channel_data(4, 3.0))\n"),
            "gold_call": ("components(_oracle_tm_channel_data(4, 3.0))\n"),
            "tol": 1e-09,
        },
        {
            "setup": (
                "def raises(fn):\n"
                "    try:\n"
                "        fn(0, 2.7)\n"
                "    except ValueError:\n"
                "        return 1.0\n"
                "    return 0.0\n"
            ),
            "call": ("raises(tm_channel_data)\n"),
            "gold_call": ("raises(_oracle_tm_channel_data)\n"),
            "tol": 1e-09,
        },
    ]
