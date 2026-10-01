"""
Evaluate a finite first-order pole expansion with a fixed subtraction value.

For poles p_j and residues r_j with respect to x, evaluate

T(x)=at_zero+sum_j r_j*[1/p_j+1/(x-p_j)]. The subtraction value is exact

at x=0, even for a finite or empty pole set. This operation applies to the

physical set and the union with channel poles. Each pole is counted once;

pole ordering has no scientific significance.

Returns
-------
complex128 ndarray, shape x.shape, fixed-background first-order finite pole expansion
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def subtracted_pole_sum(
    x: "np.ndarray",
    poles: "np.ndarray",
    residues: "np.ndarray",
    at_zero: complex,
) -> "np.ndarray":
    """Evaluate the normalized finite modal sum.

    Parameters
    ----------
    x : ndarray
        Finite real or complex grid with arbitrary shape, including a
        scalar or empty grid. Evaluation points do not coincide with poles.
    poles : ndarray
        One-dimensional nonzero complex poles with moderate finite values.
    residues : ndarray
        Complex residues, same one-dimensional shape as poles.
    at_zero : complex
        Fixed subtraction value, equal to -1 for the TM audit.

    Returns
    -------
    ndarray
        Complex128 array with exactly x.shape. An empty pole set gives
        at_zero at every evaluation point.

    Raises
    ------
    ValueError
        If pole and residue shapes differ, or a pole is zero.

    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_subtracted_pole_sum(
    x: "np.ndarray",
    poles: "np.ndarray",
    residues: "np.ndarray",
    at_zero: complex,
) -> "np.ndarray":
    x = np.asarray(x, dtype=complex)
    poles = np.asarray(poles, dtype=complex)
    residues = np.asarray(residues, dtype=complex)
    if poles.shape != residues.shape or np.any(poles == 0):
        raise ValueError(
            "poles and residues must match and poles must be nonzero"
        )
    terms = residues * x[..., None] / (poles * (x[..., None] - poles))
    return np.asarray(at_zero + np.sum(terms, axis=-1), dtype=complex)

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
                "x = np.array([0.0, 0.4, 1.8, 4.0])\n"
                "p = np.array([-2.0 - 0.2j, 2.0 - 0.2j])\n"
                "r = np.array([-0.1 + 0.3j, 0.1 + 0.3j])\n"
                "\n"
                "\n"
                "def components(value):\n"
                "    return np.stack((value.real, value.imag), axis=-1)\n"
            ),
            "call": (
                "components(\n"
                "    subtracted_pole_sum(\n"
                "        x.copy(), p.copy(), r.copy(), -1.0\n"
                "    )\n"
                ")\n"
            ),
            "gold_call": (
                "components(\n"
                "    _oracle_subtracted_pole_sum(\n"
                "        x.copy(), p.copy(), r.copy(), -1.0\n"
                "    )\n"
                ")\n"
            ),
            "tol": 1e-09,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "x = np.array([[0.0, 1.0], [2.0, 3.0]])\n"
                "p = np.array([], dtype=complex)\n"
                "r = np.array([], dtype=complex)\n"
                "\n"
                "\n"
                "def components(value):\n"
                "    return np.stack((value.real, value.imag), axis=-1)\n"
            ),
            "call": (
                "components(\n"
                "    subtracted_pole_sum(\n"
                "        x.copy(), p.copy(), r.copy(), 0.25j\n"
                "    )\n"
                ")\n"
            ),
            "gold_call": (
                "components(\n"
                "    _oracle_subtracted_pole_sum(\n"
                "        x.copy(), p.copy(), r.copy(), 0.25j\n"
                "    )\n"
                ")\n"
            ),
            "tol": 1e-09,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "x = np.array(0.3 + 0.1j)\n"
                "p = np.array([0.9 + 2.1j, 2.8 + 0.8j])\n"
                "r = np.array([0.04 + 0.1j, 0.31 - 0.001j])\n"
                "\n"
                "\n"
                "def components(value):\n"
                "    return np.stack((value.real, value.imag), axis=-1)\n"
            ),
            "call": (
                "components(\n"
                "    subtracted_pole_sum(\n"
                "        x.copy(), p.copy(), r.copy(), -1.0\n"
                "    )\n"
                ")\n"
            ),
            "gold_call": (
                "components(\n"
                "    _oracle_subtracted_pole_sum(\n"
                "        x.copy(), p.copy(), r.copy(), -1.0\n"
                "    )\n"
                ")\n"
            ),
            "tol": 1e-09,
        },
        {
            "setup": (
                "def raises(fn):\n"
                "    try:\n"
                "        fn(1.0, [0.0], [1.0], -1.0)\n"
                "    except ValueError:\n"
                "        return 1.0\n"
                "    return 0.0\n"
            ),
            "call": ("raises(subtracted_pole_sum)\n"),
            "gold_call": ("raises(_oracle_subtracted_pole_sum)\n"),
            "tol": 1e-09,
        },
    ]
