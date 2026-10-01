"""
Return the real and imaginary parts of the slow-mode wavefunction implied by one sweep of the matching, given the known exact complex wavefunction psi and the four coefficients from the previous step in the order that step returns them. psi is a complex scalar and coeffs a length-4 real array. Raise ValueError if coeffs is not length 4, or if the system is singular.

The two unknowns are coupled, so recovering them is a small linear solve rather than two independent divisions. The matrix that couples them is symmetric but not diagonal: its two diagonal entries differ, while its two off-diagonal entries are equal.

Returns
-------
ndarray of shape (2,), float64: the real part of the slow-mode wavefunction followed by its imaginary part.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_matching_system(psi: complex, coeffs: 'np.ndarray') -> 'np.ndarray':
    """Return the real and imaginary parts of the slow-mode wavefunction implied by one sweep of the matching, given the known exact complex wavefunction psi and the four coefficients from the previous step in the order that step returns them. psi is a complex scalar and coeffs a length-4 real array. Raise ValueError if coeffs is not length 4, or if the system is singular.

    Returns
    -------
    ndarray of shape (2,), float64: the real part of the slow-mode wavefunction followed by its imaginary part.

    Raises
    ------
    ValueError
        If coeffs does not hold exactly four entries, or if the system is singular.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solve_matching_system(psi: complex, coeffs: 'np.ndarray') -> 'np.ndarray':
    psi = complex(np.asarray(psi, dtype=complex))
    c = np.asarray(coeffs, dtype=float)
    if c.shape != (4,):
        raise ValueError("coeffs must hold exactly four entries")
    a_sin, a_cos, c_re, c_im = (float(c[0]), float(c[1]), float(c[2]), float(c[3]))
    # Eq (41).  The matrix is symmetric but not diagonal: the two diagonal entries differ
    # in the sign they give the first coefficient, while both off-diagonal entries are
    # the negative of the second.
    mat = np.array([[1.0 + a_sin, -a_cos], [-a_cos, 1.0 - a_sin]], dtype=float)
    rhs = np.array([psi.real + c_re, psi.imag + c_im], dtype=float)
    if abs(np.linalg.det(mat)) < 1e-300:
        raise ValueError("the matching system is singular")
    sol = np.linalg.solve(mat, rhs)
    return np.array([float(sol[0]), float(sol[1])], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent test specifications."""
    return [
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "c = np.array([0.02, -0.03, 1e-4, -2e-4])\n"
            ),
            "call": (
                "solve_matching_system(0.05 + 0.01j, c.copy())"
            ),
            "gold_call": (
                "_oracle_solve_matching_system(0.05 + 0.01j, c.copy())"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "c = np.array([0.0, 0.0, 0.0, 0.0])\n"
            ),
            "call": (
                "solve_matching_system(0.312 + 0j, c.copy())"
            ),
            "gold_call": (
                "_oracle_solve_matching_system(0.312 + 0j, c.copy())"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "c = np.array([0.3, 0.4, -1e-3, 5e-4])\n"
            ),
            "call": (
                "solve_matching_system(0.105 - 0.04j, c.copy())"
            ),
            "gold_call": (
                "_oracle_solve_matching_system(0.105 - 0.04j, c.copy())"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def _raises(fn):\n"
                "    try:\n"
                "        fn(0.1 + 0.05j, np.array([1.0, 2.0, 3.0]))\n"
                "        return False\n"
                "    except ValueError:\n"
                "        return True\n"
            ),
            "call": (
                "_raises(solve_matching_system)"
            ),
            "gold_call": (
                "_raises(_oracle_solve_matching_system)"
            ),
        },
    ]
