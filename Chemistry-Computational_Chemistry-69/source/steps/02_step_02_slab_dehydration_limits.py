"""
Step 02 - Mapping interfacial depth onto dehydration level.

The interface is not a surface but a region several angstroms thick over which the water density falls smoothly from its bulk value to nothing. The standard description of that fall-off is a hyperbolic tangent centred on the Gibbs dividing surface with a width parameter fixed by simulation, and it is the only structural input the rest of the problem needs.

Identifying the dehydration level of a donor with the local fractional depletion of water density turns that structural profile into a thermodynamic coordinate. A donor sitting well inside the liquid has a dehydration level near zero; one sitting out in the vapour tail has a level near one; and the whole of the interesting chemistry happens in between. The map is smooth and strictly monotonic, so it inverts, and every dehydration level inside the open unit interval corresponds to exactly one depth.

Both directions are needed later. The forward direction fixes what range of dehydration levels a slab of given thickness contains. The inverse direction is what places a donor in space once its dehydration level is known, which is what the tunnelling path and the ion-pair separation both depend on.

Returns
-------
numpy.ndarray of shape (3,), two dimensionless levels and a depth in angstrom
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def slab_dehydration_limits(Z_lo: float, Z_hi: float, theta_probe: float) -> np.ndarray:
    '''Depth-to-dehydration map evaluated on a slab and inverted at one point.

    Parameters
    ----------
    Z_lo : float
        Depth of the liquid-side face of the interfacial slab, angstrom.
    Z_hi : float
        Depth of the vapour-side face of the interfacial slab, angstrom,
        strictly greater than Z_lo.
    theta_probe : float
        A dehydration level, strictly inside (0, 1), to be mapped back to a depth.

    Returns
    -------
    limits : numpy.ndarray
        Array of shape (3,). Element 0 is the dehydration level at Z_lo,
        element 1 is the dehydration level at Z_hi, both dimensionless.
        Element 2 is the depth in angstrom at which the dehydration level
        equals theta_probe.

    Raises
    ------
    ValueError
        If any argument is not finite, if Z_hi is not strictly greater than
        Z_lo, or if theta_probe does not lie strictly inside (0, 1).
    '''
    return limits

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt

def _reduced_density(Z):
    """Water number density at depth Z, in units of the bulk density."""
    _Z_G = 0.84
    _DELTA = 1.5
    import numpy as np
    return 0.5 * (1.0 - np.tanh((np.asarray(Z, dtype=float) - _Z_G) / _DELTA))


def _oracle_slab_dehydration_limits(Z_lo: float, Z_hi: float, theta_probe: float) -> np.ndarray:
    """Dehydration levels bounding a slab, and the depth of a given level."""
    _Z_G = 0.84
    _DELTA = 1.5
    import numpy as np
    zlo, zhi, tp = float(Z_lo), float(Z_hi), float(theta_probe)
    if not all(np.isfinite(v) for v in (zlo, zhi, tp)):
        raise ValueError("all arguments must be finite")
    if not zhi > zlo:
        raise ValueError("Z_hi must be strictly greater than Z_lo")
    if not (0.0 < tp < 1.0):
        raise ValueError("theta_probe must lie strictly inside (0, 1)")
    theta_lo = 1.0 - float(_reduced_density(zlo))
    theta_hi = 1.0 - float(_reduced_density(zhi))
    Z_probe = _Z_G + _DELTA * np.arctanh(2.0 * tp - 1.0)
    return np.array([theta_lo, theta_hi, Z_probe], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the six-angstrom slab of the target system ---
        {
            "setup": "import numpy as np\nZ_lo = -3.0\nZ_hi = 3.0\ntheta_probe = 0.73\n",
            "call": "slab_dehydration_limits(Z_lo, Z_hi, theta_probe)",
            "gold_call": "_oracle_slab_dehydration_limits(Z_lo, Z_hi, theta_probe)",
        },
        # --- Normal: a wider slab reaching further into both phases ---
        {
            "setup": "import numpy as np\nZ_lo = -6.0\nZ_hi = 6.0\ntheta_probe = 0.25\n",
            "call": "slab_dehydration_limits(Z_lo, Z_hi, theta_probe)",
            "gold_call": "_oracle_slab_dehydration_limits(Z_lo, Z_hi, theta_probe)",
        },
        # --- Boundary: probe exactly at the symmetric midpoint of the map ---
        {
            "setup": "import numpy as np\nZ_lo = -1.0\nZ_hi = 1.0\ntheta_probe = 0.5\n",
            "call": "slab_dehydration_limits(Z_lo, Z_hi, theta_probe)",
            "gold_call": "_oracle_slab_dehydration_limits(Z_lo, Z_hi, theta_probe)",
        },
        # --- Edge: a thin slab far out in the vapour tail ---
        {
            "setup": "import numpy as np\nZ_lo = 4.0\nZ_hi = 5.0\ntheta_probe = 0.995\n",
            "call": "slab_dehydration_limits(Z_lo, Z_hi, theta_probe)",
            "gold_call": "_oracle_slab_dehydration_limits(Z_lo, Z_hi, theta_probe)",
        },
    ]
