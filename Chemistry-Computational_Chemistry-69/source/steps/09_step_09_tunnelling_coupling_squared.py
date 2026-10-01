"""
Step 09 - Path-integrated electronic coupling between donor and acceptor.

Non-adiabatic electron transfer rates are proportional to the square of the electronic coupling between the donor and acceptor states, and that squared coupling falls off exponentially with separation. The decay constant controlling the fall-off is a property of whatever medium lies between them. Liquid water is an efficient superexchange bridge, with a decay constant near 1.6 per angstrom; empty space is a much poorer one, near 2.9 per angstrom, because there are no bridge orbitals to mediate the overlap.

At an interface neither number applies on its own. A donor sitting in the vapour tail and an acceptor sitting in the bulk liquid are separated by a path that starts in dense water and ends in near-vacuum, so the local decay constant varies continuously along it. The consistent treatment accumulates the attenuation along the path instead of evaluating a single decay constant at either end, with the local value interpolating between the aqueous and vacuum limits in proportion to the local water density taken from the same interfacial profile that fixes the dehydration level. The accumulated attenuation is what multiplies the squared coupling, following the usual convention in which the decay constant is quoted for the squared coupling rather than for the amplitude.

This is what couples the electronic and thermodynamic halves of the problem. The donor's dehydration level fixes where it sits, and where it sits fixes both how long the tunnelling path is and how much of it is bridged by water. Dehydration buys driving force and pays for it in coupling, so the two effects compete and their product peaks somewhere inside the interface.

Returns
-------
numpy.ndarray, squared electronic coupling in J^2 at each dehydration level
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def tunnelling_coupling_squared(theta: npt.ArrayLike, Z_acceptor: float) -> np.ndarray:
    '''Squared electronic coupling for a donor at a given dehydration level.

    Parameters
    ----------
    theta : array_like
        Dehydration level or levels of the donor, each strictly inside (0, 1).
    Z_acceptor : float
        Depth of the fully solvated acceptor along the interface normal,
        angstrom, on the liquid side of the donor.

    Returns
    -------
    coupling_sq : numpy.ndarray
        Squared donor-acceptor electronic coupling in joules squared, same
        shape as theta.

    Raises
    ------
    ValueError
        If Z_acceptor is not finite, if theta is not finite, or if any element
        of theta does not lie strictly inside (0, 1).
    '''
    return coupling_sq

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt

def _depth_from_theta_c(theta):
    """Depth in angstrom at which the dehydration level equals theta."""
    _ZG_C = 0.84
    _DEL_C = 1.5
    import numpy as np
    return _ZG_C + _DEL_C * np.arctanh(2.0 * np.asarray(theta, dtype=float) - 1.0)


def _density_antiderivative(Z):
    """Antiderivative of the reduced water density with respect to depth."""
    _ZG_C = 0.84
    _DEL_C = 1.5
    import numpy as np
    Zf = np.asarray(Z, dtype=float)
    x = (Zf - _ZG_C) / _DEL_C
    log_cosh = np.logaddexp(x, -x) - np.log(2.0)
    return 0.5 * (Zf - _DEL_C * log_cosh)


def _oracle_tunnelling_coupling_squared(theta: npt.ArrayLike, Z_acceptor: float) -> np.ndarray:
    """Squared coupling with the tunnelling attenuation accumulated along the path."""
    _H0_CM1 = 50.0
    _BETA_WATER = 1.6
    _BETA_VACUUM = 2.9
    _J_PER_CM1 = 1.98644586e-23
    import numpy as np
    th = np.asarray(theta, dtype=float)
    Za = float(Z_acceptor)
    if not np.isfinite(Za):
        raise ValueError("Z_acceptor must be finite")
    if not np.all(np.isfinite(th)):
        raise ValueError("theta must be finite")
    if np.any(th <= 0.0) or np.any(th >= 1.0):
        raise ValueError("theta must lie strictly inside (0, 1)")
    Zd = _depth_from_theta_c(th)
    exponent = (_BETA_VACUUM * (Zd - Za)
                + (_BETA_WATER - _BETA_VACUUM)
                * (_density_antiderivative(Zd) - _density_antiderivative(Za)))
    H0_J = _H0_CM1 * _J_PER_CM1
    return H0_J ** 2 * np.exp(-exponent)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: donors spanning the interfacial slab of the target system ---
        {
            "setup": ("import numpy as np\nZ_acc = -6.50\n"
                      "theta = np.array([0.10, 0.40, 0.73, 0.90])\n"),
            "call": "tunnelling_coupling_squared(theta, Z_acc)",
            "gold_call": "_oracle_tunnelling_coupling_squared(theta, Z_acc)",
        },
        # --- Normal: acceptor pushed deeper into the liquid ---
        {
            "setup": ("import numpy as np\nZ_acc = -12.0\n"
                      "theta = np.array([0.25, 0.60, 0.85])\n"),
            "call": "tunnelling_coupling_squared(theta, Z_acc)",
            "gold_call": "_oracle_tunnelling_coupling_squared(theta, Z_acc)",
        },
        # --- Boundary: donor exactly on the Gibbs dividing surface ---
        {
            "setup": "import numpy as np\nZ_acc = -6.50\ntheta = 0.5\n",
            "call": "tunnelling_coupling_squared(theta, Z_acc)",
            "gold_call": "_oracle_tunnelling_coupling_squared(theta, Z_acc)",
        },
        # --- Edge: nearly dry donor far out in the vapour tail ---
        {
            "setup": ("import numpy as np\nZ_acc = -6.50\n"
                      "theta = np.array([0.97, 0.99])\n"),
            "call": "tunnelling_coupling_squared(theta, Z_acc)",
            "gold_call": "_oracle_tunnelling_coupling_squared(theta, Z_acc)",
        },
    ]
