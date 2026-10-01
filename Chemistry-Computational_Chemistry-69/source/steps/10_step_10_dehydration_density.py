"""
Step 10 - Population density of the dehydration level across the slab.

Interfacial donors are not distributed uniformly in dehydration level. What is uniform is their distribution in depth, since nothing in the slab picks out one plane over another, and the dehydration level is a nonlinear function of depth. Transforming one distribution into the other is a change of variables, so the density in dehydration level picks up the Jacobian of the depth-to-dehydration map.

That Jacobian is the reciprocal of the derivative of the map, and because the map is a hyperbolic tangent its derivative is largest in the middle of the interface and vanishes at both ends. The induced density therefore does the opposite: it is smallest near the symmetric midpoint and diverges towards both extremes of hydration. Physically this is just the statement that the density profile is flat deep in the liquid and flat out in the vapour, so a large slice of depth maps onto a tiny slice of dehydration level at either end.

The density must be normalised over the slab, and the normalisation depends only on the slab thickness and the width parameter of the profile, not on where the slab sits. Skipping the Jacobian and weighting all dehydration levels equally is the single most consequential shortcut available in this problem, because it silently reassigns weight away from the extremes and towards the middle of the interface, where the chemistry does not happen.

Returns
-------
numpy.ndarray, normalised probability density of the dehydration level
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def dehydration_density(theta: npt.ArrayLike, Z_lo: float, Z_hi: float) -> np.ndarray:
    '''Normalised probability density of the dehydration level across a slab.

    Parameters
    ----------
    theta : array_like
        Dehydration level or levels, each strictly inside (0, 1).
    Z_lo : float
        Depth of the liquid-side face of the interfacial slab, angstrom.
    Z_hi : float
        Depth of the vapour-side face of the interfacial slab, angstrom,
        strictly greater than Z_lo.

    Returns
    -------
    density : numpy.ndarray
        Probability density of the dehydration level, dimensionless, normalised
        to unit integral over the dehydration levels the slab contains, same
        shape as theta.

    Raises
    ------
    ValueError
        If Z_lo or Z_hi is not finite, if Z_hi is not strictly greater than
        Z_lo, if theta is not finite, or if any element of theta does not lie
        strictly inside (0, 1).
    '''
    return density

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt

def _oracle_dehydration_density(theta: npt.ArrayLike, Z_lo: float, Z_hi: float) -> np.ndarray:
    """Density in dehydration level induced by a uniform density in depth."""
    _DEL_P = 1.5
    import numpy as np
    th = np.asarray(theta, dtype=float)
    zlo, zhi = float(Z_lo), float(Z_hi)
    if not (np.isfinite(zlo) and np.isfinite(zhi)):
        raise ValueError("Z_lo and Z_hi must be finite")
    if not zhi > zlo:
        raise ValueError("Z_hi must be strictly greater than Z_lo")
    if not np.all(np.isfinite(th)):
        raise ValueError("theta must be finite")
    if np.any(th <= 0.0) or np.any(th >= 1.0):
        raise ValueError("theta must lie strictly inside (0, 1)")
    norm = 2.0 * (zhi - zlo) / _DEL_P
    return 1.0 / (norm * th * (1.0 - th))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: target slab sampled across the reactive region ---
        {
            "setup": ("import numpy as np\nZ_lo = -3.0\nZ_hi = 3.0\n"
                      "theta = np.array([0.10, 0.35, 0.60, 0.73, 0.90])\n"),
            "call": "dehydration_density(theta, Z_lo, Z_hi)",
            "gold_call": "_oracle_dehydration_density(theta, Z_lo, Z_hi)",
        },
        # --- Normal: a thicker slab, same dehydration levels ---
        {
            "setup": ("import numpy as np\nZ_lo = -6.0\nZ_hi = 6.0\n"
                      "theta = np.array([0.20, 0.50, 0.80])\n"),
            "call": "dehydration_density(theta, Z_lo, Z_hi)",
            "gold_call": "_oracle_dehydration_density(theta, Z_lo, Z_hi)",
        },
        # --- Boundary: the symmetric midpoint of the dehydration range ---
        {
            "setup": "import numpy as np\nZ_lo = -3.0\nZ_hi = 3.0\ntheta = 0.5\n",
            "call": "dehydration_density(theta, Z_lo, Z_hi)",
            "gold_call": "_oracle_dehydration_density(theta, Z_lo, Z_hi)",
        },
        # --- Edge: close to both ends of the range where the density diverges ---
        {
            "setup": ("import numpy as np\nZ_lo = -3.0\nZ_hi = 3.0\n"
                      "theta = np.array([0.006, 0.946])\n"),
            "call": "dehydration_density(theta, Z_lo, Z_hi)",
            "gold_call": "_oracle_dehydration_density(theta, Z_lo, Z_hi)",
        },
    ]
