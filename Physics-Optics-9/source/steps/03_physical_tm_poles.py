"""
Find the complete finite set of outgoing TM resonances in a rectangle.

The physical poles are the zeros of the preceding outgoing determinant

D in |Re(x)|<x_cut, -4<Im(x)<0. The singularity of D at zero is removable

and is not a resonance. Include both real-part signs and any imaginary-axis

resonance, counting every distinct simple zero once. Real material parameters

give negative-conjugate symmetry. Return poles sorted by real part, then

imaginary part; set a numerically vanishing real part to exactly zero before

sorting. Root finding may use any converged method. The argument principle

on the rectangle provides a completeness check; root discovery seeds do not

define the answer. Boundary poles are excluded by the supported domain.

Returns
-------
complex128 ndarray, shape (number_of_poles,), sorted complete physical poles in the open rectangle
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def physical_tm_poles(ell: int, n: float, x_cut: float) -> "np.ndarray":
    """Find every simple physical pole in the prescribed open rectangle.

    Parameters
    ----------
    ell : int
        Angular momentum, 1 through 4.
    n : float
        Real refractive index in [2, 3].
    x_cut : float
        Positive real-part cutoff in [8, 22]. The rectangle has depth 4.
        Supported instances have simple poles at least 1e-5 from its edges.

    Returns
    -------
    ndarray
        One-dimensional complex128 array of all poles, sorted first by real
        part and then by imaginary part. Absolute root error at most 1e-9
        is sufficient. Real parts smaller than 1e-9 in magnitude are zero.

    Raises
    ------
    ValueError
        If ell, n or x_cut is outside the stated interval, or a complete
        simple-pole set cannot be resolved numerically.

    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_physical_tm_poles(
    ell: int, n: float, x_cut: float
) -> "np.ndarray":
    if not 8.0 <= x_cut <= 22.0:
        raise ValueError("x_cut must lie in [8, 22]")
    real = np.linspace(-x_cut, x_cut, int(2.6 * n * x_cut) + 1)
    imag = np.linspace(-4.0, -0.02, 7)
    roots = (real[:, None] + 1j * imag).ravel()
    active = np.ones(roots.size, dtype=bool)
    for _ in range(80):
        if not np.any(active):
            break
        values = _oracle_tm_boundary_data(ell, n, roots[active])
        correction = values[:, 1] / values[:, 2]
        correction /= np.maximum(1.0, np.abs(correction))
        indices = np.flatnonzero(active)
        roots[active] -= correction
        roots = np.clip(roots.real, -x_cut - 2, x_cut + 2) + 1j * np.clip(
            roots.imag, -6.0, -1e-7
        )
        active[indices[np.abs(correction) < 1e-12]] = False
    values = _oracle_tm_boundary_data(ell, n, roots)
    valid = (
        (np.abs(values[:, 1] / values[:, 2]) < 1e-10)
        & (np.abs(roots.real) < x_cut)
        & (roots.imag > -4.0)
        & (roots.imag < 0.0)
    )
    unique = []
    for root in roots[valid]:
        if abs(root.real) < 1e-9:
            root = complex(0.0, root.imag)
        if not any(abs(root - previous) < 1e-7 for previous in unique):
            unique.append(root)
    poles = np.array(
        sorted(unique, key=lambda z: (z.real, z.imag)), dtype=complex
    )
    corners = [-x_cut - 4j, x_cut - 4j, x_cut - 1e-8j, -x_cut - 1e-8j]
    contour = np.concatenate(
        [
            np.linspace(corners[j], corners[(j + 1) % 4], 1024, endpoint=False)
            for j in range(4)
        ]
    )
    determinant = _oracle_tm_boundary_data(ell, n, contour)[:, 1]
    increments = np.angle(np.roll(determinant, -1) / determinant)
    count = np.sum(increments) / (2 * np.pi)
    if abs(count - round(count)) > 1e-6 or len(poles) != round(count):
        raise ValueError("complete simple-pole set was not resolved")
    return poles

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
            "call": ("components(physical_tm_poles(3, 2.7, 18.5))\n"),
            "gold_call": (
                "components(_oracle_physical_tm_poles(3, 2.7, 18.5))\n"
            ),
            "tol": 1e-08,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def components(value):\n"
                "    return np.stack((value.real, value.imag), axis=-1)\n"
            ),
            "call": ("components(physical_tm_poles(1, 2.0, 8.2))\n"),
            "gold_call": (
                "components(_oracle_physical_tm_poles(1, 2.0, 8.2))\n"
            ),
            "tol": 1e-08,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def components(value):\n"
                "    return np.stack((value.real, value.imag), axis=-1)\n"
            ),
            "call": ("components(physical_tm_poles(4, 3.0, 20.4))\n"),
            "gold_call": (
                "components(_oracle_physical_tm_poles(4, 3.0, 20.4))\n"
            ),
            "tol": 1e-08,
        },
        {
            "setup": (
                "def raises(fn):\n"
                "    try:\n"
                "        fn(3, 2.7, 0.0)\n"
                "    except ValueError:\n"
                "        return 1.0\n"
                "    return 0.0\n"
            ),
            "call": ("raises(physical_tm_poles)\n"),
            "gold_call": ("raises(_oracle_physical_tm_poles)\n"),
            "tol": 1e-09,
        },
    ]
