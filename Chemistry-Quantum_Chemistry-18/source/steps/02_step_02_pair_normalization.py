"""
Step 02: Normalization of the pair function. Normalization constant C of the singlet pair function Psi = C p(r12) exp[-omega (r1^2 + r2^2) / 2] with p(s) = 1 + s/2 + c s^2.

The spatial part of a two-electron singlet is normalized to one over both electron positions, which is equivalent to
normalizing the two-electron density rho2(r1_vec, r2_vec) = |Psi|^2 to N(N - 1)/2 = 1. The Gaussian factor separates in
the centre-of-mass and relative coordinates, R_vec = (r1_vec + r2_vec)/2 and s_vec = r1_vec - r2_vec, because
r1^2 + r2^2 = 2 R^2 + s^2 / 2 and the volume element is unchanged by this change of variables. The correlation factor
depends on s only, so the six-dimensional normalization integral becomes a product of a Gaussian integral over R_vec and
a radial integral over s. For omega = 1/10 and c = 1/20 this Psi is the exact ground state of harmonium (two electrons
with Coulomb repulsion in the isotropic trap omega^2 r^2 / 2), whose energy is 1/2 hartree.

Returns
-------
float, normalization constant C of the pair function
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pair_normalization(omega: float, c: float) -> float:
    '''Positive constant C that normalizes Psi = C p(r12) exp[-omega (r1^2 + r2^2)/2] to one.

    Parameters
    ----------
    omega : float
        Gaussian exponent parameter, omega > 0 (atomic units).
    c : float
        Coefficient of r12^2 in p(s) = 1 + s/2 + c s^2, c >= 0.

    Returns
    -------
    result : float
        The normalization constant C as a Python float, accurate to 1e-10 relative.

    Raises
    ------
    ValueError
        If omega is not positive or c is negative.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import quad


def _oracle_pair_normalization(omega: float, c: float) -> float:
    """Reference implementation."""
    omega = float(omega)
    c = float(c)
    if not omega > 0.0:
        raise ValueError("omega must be positive")
    if c < 0.0:
        raise ValueError("c must be non-negative")
    centre = (np.pi / (2.0 * omega)) ** 1.5
    relative = quad(lambda s: (1.0 + 0.5 * s + c * s * s) ** 2 * np.exp(-0.5 * omega * s * s) * 4.0 * np.pi * s * s,
                    0.0, np.inf, epsabs=0.0, epsrel=1e-13, limit=200)[0]
    return float(1.0 / np.sqrt(centre * relative))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: exact harmonium ground state at omega = 1/10 ---
        {
            "setup": "",
            "call": "pair_normalization(0.1, 0.05)",
            "gold_call": "_oracle_pair_normalization(0.1, 0.05)",
            "tol": 1e-10,
        },
        # --- Normal: exact harmonium ground state at omega = 1/2, no quadratic term ---
        {
            "setup": "",
            "call": "pair_normalization(0.5, 0.0)",
            "gold_call": "_oracle_pair_normalization(0.5, 0.0)",
            "tol": 1e-10,
        },
        # --- Boundary: tight trap, where the correlation factor barely matters ---
        {
            "setup": "",
            "call": "pair_normalization(4.0, 0.2)",
            "gold_call": "_oracle_pair_normalization(4.0, 0.2)",
            "tol": 1e-10,
        },
        # --- Edge: very weak trap with a large quadratic term ---
        {
            "setup": "",
            "call": "pair_normalization(0.02, 0.5)",
            "gold_call": "_oracle_pair_normalization(0.02, 0.5)",
            "tol": 1e-10,
        },
        # --- Error: non-positive omega must raise ValueError ---
        {
            "setup": "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(0.0, 0.05)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    return 0\n",
            "call": "_probe(pair_normalization)",
            "gold_call": "_probe(_oracle_pair_normalization)",
        },
    ]
