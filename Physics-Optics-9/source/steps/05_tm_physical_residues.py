"""
Evaluate the derivative-normalized residues of supplied physical poles.

For S=N/D, a simple physical pole p has ordinary residue N(p)/D'(p).

The regularized residue is xi_+'(p)/xi_-'(p) times that value. The derivative

is with respect to dimensionless x, including every factor of n from the

interior radial functions. Obtain N,D' and the radial derivatives from the

preceding steps. Preserve the supplied pole order.

Returns
-------
complex128 ndarray, same shape and order as poles, transformed physical-pole residues
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def tm_physical_residues(
    ell: int, n: float, poles: "np.ndarray"
) -> "np.ndarray":
    """Return one transformed residue for each supplied physical pole.

    Parameters
    ----------
    ell : int
        Angular momentum, 1 through 4.
    n : float
        Real refractive index in [2, 3].
    poles : ndarray
        One-dimensional complex array of simple physical poles, at least
        1e-8 from zero, with |Re(p)| <= 22 and -4 < Im(p) < 0. An empty
        array is allowed. Pole approximations accurate to 1e-10 suffice.

    Returns
    -------
    ndarray
        Complex128 array with the same one-dimensional shape and order
        as poles, containing residues of T with respect to x.

    Raises
    ------
    ValueError
        If ell or n is unsupported, or a supplied pole is zero.

    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_tm_physical_residues(
    ell: int, n: float, poles: "np.ndarray"
) -> "np.ndarray":
    poles = np.asarray(poles, dtype=complex)
    boundary = _oracle_tm_boundary_data(ell, n, poles)
    radial = _oracle_riccati_wave_data(ell, poles)
    return boundary[:, 0] / boundary[:, 2] * radial[:, 4] / radial[:, 7]

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
                "poles = np.array(\n"
                "    [\n"
                "        -2.4021436406684367 - 0.0470498072800886j,\n"
                "        2.4021436406684367 - 0.0470498072800886j,\n"
                "    ]\n"
                ")\n"
                "\n"
                "\n"
                "def components(value):\n"
                "    return np.stack((value.real, value.imag), axis=-1)\n"
            ),
            "call": (
                "components(tm_physical_residues(3, 2.7, poles.copy()))\n"
            ),
            "gold_call": (
                "components(\n"
                "    _oracle_tm_physical_residues(3, 2.7, poles.copy())\n"
                ")\n"
            ),
            "tol": 1e-09,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "poles = np.array(\n"
                "    [0.9324631643958519 - 2.2883699197031913j]\n"
                ")\n"
                "\n"
                "\n"
                "def components(value):\n"
                "    return np.stack((value.real, value.imag), axis=-1)\n"
            ),
            "call": (
                "components(tm_physical_residues(3, 2.7, poles.copy()))\n"
            ),
            "gold_call": (
                "components(\n"
                "    _oracle_tm_physical_residues(3, 2.7, poles.copy())\n"
                ")\n"
            ),
            "tol": 1e-09,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "poles = np.array([], dtype=complex)\n"
                "\n"
                "\n"
                "def components(value):\n"
                "    return np.stack((value.real, value.imag), axis=-1)\n"
            ),
            "call": (
                "components(tm_physical_residues(1, 2.0, poles.copy()))\n"
            ),
            "gold_call": (
                "components(\n"
                "    _oracle_tm_physical_residues(1, 2.0, poles.copy())\n"
                ")\n"
            ),
            "tol": 1e-09,
        },
    ]
