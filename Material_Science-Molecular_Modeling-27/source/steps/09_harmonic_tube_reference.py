"""
Write down the closed-form predictions the source derives for a locally harmonic orthogonal well, for a given contour level and an array of local perpendicular stiffnesses: the fraction of the orthogonal partition function that a contour-following tube retains, which the source obtains in closed form and which depends on the stiffness through nothing at all; the physical half-width of that tube; and the position of the wall expressed in local thermal widths. Return one row per stiffness. Do not compute these numerically; they are the analytic limits the audit is checked against.

The theoretical claim of the source is that the retained fraction of a contour-following tube is the same at every point of the path, and it proves it by evaluating a radial integral in closed form. The other two entries are the geometric picture behind the same statement: the wall sits at a fixed number of thermal widths whatever the local stiffness, so its physical position moves while its thermodynamic meaning does not.

Returns
-------
A (n_stiffness, 3) float64 array with columns [retained fraction, physical tube half-width in Angstrom, wall position in local thermal widths].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def harmonic_tube_reference(d_eff: int, delta_f_star: float, stiffness: "np.ndarray") -> "np.ndarray":
    """Write down the closed-form predictions the source derives for a locally harmonic orthogonal
    well, for a given contour level and an array of local perpendicular stiffnesses: the fraction of
    the orthogonal partition function that a contour-following tube retains, which the source
    obtains in closed form and which depends on the stiffness through nothing at all; the physical
    half-width of that tube; and the position of the wall expressed in local thermal widths. Return
    one row per stiffness. Do not compute these numerically; they are the analytic limits the audit
    is checked against.

    Args:
        d_eff: int, the dimensionality of the collective-variable space (at least 2); the number of
            perpendicular directions is d_eff - 1.
        delta_f_star: float, the positive contour level in kcal/mol (the level in units of k_B T
            times k_B T).
        stiffness: array-like of shape (n_stiffness,) (a scalar is accepted), positive local
            perpendicular stiffnesses in kcal/mol/Angstrom^2.

    Returns:
        A (n_stiffness, 3) float64 array with columns [retained fraction, physical tube half-width
        in Angstrom, wall position in local thermal widths].

    Raises:
        ValueError: if d_eff is smaller than 2, or if delta_f_star or any stiffness is not positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import gammainc


def _K_B():
    return 0.0019872041

def _TEMP():
    return 300.0

def _BETA():
    return 1.0/(_K_B()*_TEMP())

def _oracle_harmonic_tube_reference(d_eff: int, delta_f_star: float, stiffness: "np.ndarray") -> "np.ndarray":
    d_eff = int(d_eff); delta_f_star = float(delta_f_star)
    stiffness = np.atleast_1d(np.asarray(stiffness, dtype=float))
    if d_eff < 2:
        raise ValueError("d_eff must be at least 2")
    if delta_f_star <= 0.0 or np.any(stiffness <= 0.0):
        raise ValueError("delta_f_star and stiffness must be positive")
    d_perp = d_eff - 1
    frac = float(gammainc(d_perp/2.0, _BETA()*delta_f_star))
    eps = np.sqrt(2.0*delta_f_star/stiffness)
    walls = np.full(stiffness.shape, np.sqrt(2.0*_BETA()*delta_f_star))
    return np.stack([np.full(stiffness.shape, frac), eps, walls], 1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nd_eff = 2\ndelta_f_star = 3.0*0.0019872041*300.0\ndz_ref = 8.0e-5\nstiffness = np.array([40.0, 100.0, 400.0])\n',
         'call': 'harmonic_tube_reference(d_eff, delta_f_star, stiffness)',
         'gold_call': '_oracle_harmonic_tube_reference(d_eff, delta_f_star, stiffness)'},
        {'setup': 'import numpy as np\nd_eff = 3\ndelta_f_star = 1.0*0.0019872041*300.0\ndz_ref = 8.0e-5\nstiffness = np.array([25.0, 250.0])\n',
         'call': 'harmonic_tube_reference(d_eff, delta_f_star, stiffness)',
         'gold_call': '_oracle_harmonic_tube_reference(d_eff, delta_f_star, stiffness)'},
        {'setup': 'import numpy as np\nd_eff = 2\ndelta_f_star = 4.0*0.0019872041*300.0\ndz_ref = 8.0e-5\nstiffness = np.linspace(40.0, 400.0, 7)\n',
         'call': 'harmonic_tube_reference(d_eff, delta_f_star, stiffness)',
         'gold_call': '_oracle_harmonic_tube_reference(d_eff, delta_f_star, stiffness)'},
    ]
