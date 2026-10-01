"""
Quantify how far the textbook series-resistance rule is from the exact rate of a partially reactive site. The reaction-controlled rate of the site is the flux kappa c_B through the area of the cap, which in units of k_S is J_react = Da (1 - cos theta0) / 2. The series-resistance (Collins-Kimball) estimate combines the diffusion-controlled rate of the same site behind the same shell, J_sink, with J_react as resistances in series, 1 / J_CK = 1 / J_sink + 1 / J_react; for an isotropic sphere in unbounded space this rule is exact. Return J_react, J_CK and the signed percentage deviation delta_CK = 100 (J_CK - J_exact) / J_exact of the estimate from the exact rate J_exact of the projected solution. Raise ValueError if j_sink or j_exact is not positive, if theta0 is not in (0, pi], or if damkohler is not a finite positive number.

Collins and Kimball showed that for a uniformly reactive sphere the diffusion and reaction resistances add exactly; for a reactive patch the flux distribution over the site at finite reactivity differs from the singular edge-concentrated distribution of the perfect sink, so that the two mechanisms do not add as resistances and the rule overestimates the rate by an amount that depends on the cap angle, the reactivity, the confinement and the shell permeability.

Returns
-------
A (3,) float64 array [J_react, J_CK, delta_CK in percent].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def series_resistance_deviation(j_sink, theta0, damkohler, j_exact):
    """Quantify how far the textbook series-resistance rule is from the exact rate of a
    partially reactive site. A (3,) float64 array [J_react, J_CK, delta_CK in percent]."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander


def _oracle_series_resistance_deviation(j_sink, theta0, damkohler, j_exact):
    js = float(j_sink); t = float(theta0); da = float(damkohler); je = float(j_exact)
    if not (js > 0.0) or not (je > 0.0):
        raise ValueError("correction factors must be positive")
    if not (0.0 < t <= np.pi):
        raise ValueError("theta0 must lie in (0, pi]")
    if not (np.isfinite(da) and da > 0.0):
        raise ValueError("damkohler must be a finite positive number")
    j_react = 0.5 * da * (1.0 - np.cos(t))
    j_ck = 1.0 / (1.0 / js + 1.0 / j_react)
    return np.array([j_react, j_ck, 100.0 * (j_ck - je) / je], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nj_sink = 0.537848929\ntheta0 = float(np.arccos(0.7))\ndamkohler = 3.0\nj_exact = 0.237819522\n',
         'call': 'series_resistance_deviation(j_sink, theta0, damkohler, j_exact)',
         'gold_call': '_oracle_series_resistance_deviation(j_sink, theta0, damkohler, j_exact)'},
        {'setup': 'import numpy as np\nj_sink = 1.144922\ntheta0 = float(np.arccos(0.4))\ndamkohler = 12.0\nj_exact = 0.8478001\n',
         'call': 'series_resistance_deviation(j_sink, theta0, damkohler, j_exact)',
         'gold_call': '_oracle_series_resistance_deviation(j_sink, theta0, damkohler, j_exact)'},
        {'setup': 'import numpy as np\nj_sink = 2.0\ntheta0 = float(np.pi)\ndamkohler = 3.0\nj_exact = 3.0 / 2.5\n',
         'call': 'series_resistance_deviation(j_sink, theta0, damkohler, j_exact)',
         'gold_call': '_oracle_series_resistance_deviation(j_sink, theta0, damkohler, j_exact)'},
    ]
