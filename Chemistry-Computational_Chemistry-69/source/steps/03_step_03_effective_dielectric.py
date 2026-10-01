"""
Step 03 - Effective dielectric constant screening a pair across the interface.

Continuum electrostatics at an interface cannot use a single permittivity. The donor sits somewhere in the density gradient, the acceptor sits in the liquid, and the field line joining them runs through material whose polarisability changes by nearly two orders of magnitude along the way. Assigning the pair the bulk value of water overestimates the screening; assigning it the value at the donor underestimates it, because most of the path is still liquid.

The consistent construction treats the intervening material as a sequence of thin slabs in series. Capacitors in series add their reciprocals, so what averages along the path is the inverse permittivity, not the permittivity itself. The effective dielectric constant of the pair is therefore the reciprocal of the mean of the inverse local permittivity taken along the line joining the two ions, with the local permittivity interpolating linearly between vacuum and bulk water in proportion to the local water density from the same interfacial profile that fixes the dehydration level.

That reciprocal average is dominated by whatever stretch of the path is least polarisable, which is exactly the physical content: a short passage through low-density water screens the pair far less effectively than its length alone would suggest. The result is a screening constant that depends on where the donor sits, and so becomes a function of the dehydration level rather than a constant of the medium.

Returns
-------
numpy.ndarray, dimensionless effective dielectric constant at each dehydration level
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def effective_dielectric(theta: npt.ArrayLike, Z_acceptor: float) -> np.ndarray:
    '''Path-averaged dielectric constant screening the donor-acceptor pair.

    Parameters
    ----------
    theta : array_like
        Dehydration level or levels of the donor, each strictly inside (0, 1).
    Z_acceptor : float
        Depth of the fully solvated acceptor along the interface normal,
        angstrom, strictly below the donor.

    Returns
    -------
    eps_eff : numpy.ndarray
        Dimensionless effective dielectric constant screening the pair at each
        dehydration level, same shape as theta.

    Raises
    ------
    ValueError
        If Z_acceptor is not finite, if theta is not finite, if any element of
        theta does not lie strictly inside (0, 1), or if the donor does not lie
        strictly above the acceptor at every requested dehydration level.
    '''
    return eps_eff

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt

def _depth_from_theta_d(theta):
    """Depth in angstrom at which the dehydration level equals theta."""
    _ZG_D = 0.84
    _DEL_D = 1.5
    import numpy as np
    return _ZG_D + _DEL_D * np.arctanh(2.0 * np.asarray(theta, dtype=float) - 1.0)


def _inverse_permittivity_antiderivative(Z):
    """Antiderivative of the reciprocal local permittivity with respect to depth."""
    _ZG_D = 0.84
    _DEL_D = 1.5
    _EPS_WATER = 78.4
    import numpy as np
    amp = _EPS_WATER - 1.0
    c = 1.0 + 0.5 * amp
    d = 0.5 * amp
    scale = np.sqrt(c * c - d * d)
    phi = np.arctanh(d / c)
    u = (np.asarray(Z, dtype=float) - _ZG_D) / _DEL_D - phi
    log_cosh = np.logaddexp(u, -u) - np.log(2.0)
    return _DEL_D * (np.cosh(phi) * u + np.sinh(phi) * log_cosh) / scale


def _oracle_effective_dielectric(theta: npt.ArrayLike, Z_acceptor: float) -> np.ndarray:
    """Reciprocal-mean permittivity along the donor-acceptor path."""
    import numpy as np
    th = np.asarray(theta, dtype=float)
    Za = float(Z_acceptor)
    if not np.isfinite(Za):
        raise ValueError("Z_acceptor must be finite")
    if not np.all(np.isfinite(th)):
        raise ValueError("theta must be finite")
    if np.any(th <= 0.0) or np.any(th >= 1.0):
        raise ValueError("theta must lie strictly inside (0, 1)")
    Zd = _depth_from_theta_d(th)
    separation = Zd - Za
    if np.any(separation <= 0.0):
        raise ValueError("the donor must lie strictly above the acceptor")
    path = (_inverse_permittivity_antiderivative(Zd)
            - _inverse_permittivity_antiderivative(Za))
    return separation / path

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
            "call": "effective_dielectric(theta, Z_acc)",
            "gold_call": "_oracle_effective_dielectric(theta, Z_acc)",
        },
        # --- Normal: acceptor withdrawn further into the bulk liquid ---
        {
            "setup": ("import numpy as np\nZ_acc = -12.0\n"
                      "theta = np.array([0.25, 0.60, 0.85])\n"),
            "call": "effective_dielectric(theta, Z_acc)",
            "gold_call": "_oracle_effective_dielectric(theta, Z_acc)",
        },
        # --- Boundary: donor exactly on the Gibbs dividing surface ---
        {
            "setup": "import numpy as np\nZ_acc = -6.50\ntheta = 0.5\n",
            "call": "effective_dielectric(theta, Z_acc)",
            "gold_call": "_oracle_effective_dielectric(theta, Z_acc)",
        },
        # --- Edge: nearly dry donor far out in the vapour tail, screening collapses ---
        {
            "setup": ("import numpy as np\nZ_acc = -6.50\n"
                      "theta = np.array([0.97, 0.995])\n"),
            "call": "effective_dielectric(theta, Z_acc)",
            "gold_call": "_oracle_effective_dielectric(theta, Z_acc)",
        },
    ]
