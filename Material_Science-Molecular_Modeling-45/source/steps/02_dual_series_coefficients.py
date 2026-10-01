"""
Derive the radial exponents and the coefficient sequences of the canonical dual series relations of the perfect-sink problem in a hindered corona behind a permeable shell. In the dimensionless coordinate xi = r / R the trapping probability u = 1 - c / c_B satisfies the steady diffusion equation with the diffusivity D(r) = D (r / R)^p in the cavity 1 < xi < 1 / eps, the permeable-wall condition on the cavity wall xi = 1 / eps (the dimensionless form of the entry-flux balance with the diffusivity at the wall; u = 0 when Bi = inf), equals one on the active cap 0 <= theta < theta0 of the unit sphere and has zero normal derivative on the inert part theta0 < theta <= pi. Separate variables: the axisymmetric solutions are xi^s P_l(cos theta) with two exponents per l, s_plus(l) >= 0 and s_minus(l) < 0, that follow from the radial equation (derive them; they reduce to l and -(l + 1) for p = 0). Expand u = sum over l of (A_l^+ xi^s_plus + A_l xi^s_minus) P_l(cos theta) and eliminate the interior moments with the wall condition: A_l^+ = -eps_l A_l with a wall coupling coefficient eps_l that depends on eps, Bi, p and l and reduces to eps^(2l+1) for p = 0 and Bi = inf (derive it). Write the two boundary relations on the unit sphere in terms of the exterior moments A_l and introduce the canonical unknowns X_l through (l + 1/2) X_l = (-s_minus + eps_l s_plus) A_l, so that the inert-part relation reads sum over l of (l + 1/2) X_l P_l(cos theta) = 0 and the cap relation reads sum over l of (1 - q_l) X_l P_l(cos theta) = 1, and write A_l = (1 - w_l) X_l. Return s_plus, s_minus, eps_l, w_l and q_l for l = 0..order; eps = 0 is the unbounded limit, in which eps_l vanishes whatever Bi. Raise ValueError if order is not a non-negative integer, if eps is not in [0, 1), if biot is not positive (inf allowed), or if hindrance is negative or not finite.

The dual series relations method solves mixed boundary value problems of potential theory by expanding in the eigenfunctions of the angular operator and converting the two boundary conditions into a pair of series relations on complementary parts of the sphere. Their canonical form is the one in which the relation on the reflecting part carries the weight l + 1/2 and the relation on the absorbing part carries a coefficient 1 - q_l that tends to one as l grows; the whole geometry of the cavity, the permeability of its wall and the hindrance profile enter through the sequence q_l alone, which is why neither changes anything downstream except that sequence.

Returns
-------
A (5, order + 1) float64 array whose rows are s_plus, s_minus, eps_l, w_l and q_l, l = 0..order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def dual_series_coefficients(order, eps, biot, hindrance):
    """Derive the radial exponents and the coefficient sequences of the canonical dual series
    relations of the perfect-sink problem in a hindered corona behind a permeable shell. A
    (5, order + 1) float64 array whose rows are s_plus, s_minus, eps_l, w_l and q_l, l =
    0..order."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander


def _oracle_dual_series_coefficients(order, eps, biot, hindrance):
    n = int(order); e = float(eps); bi = float(biot); p = float(hindrance)
    if n < 0 or n != order:
        raise ValueError("order must be a non-negative integer")
    if not (0.0 <= e < 1.0):
        raise ValueError("eps must lie in [0, 1)")
    if not (bi > 0.0):
        raise ValueError("biot must be positive (inf for a perfectly permeable shell)")
    if not (p >= 0.0) or not np.isfinite(p):
        raise ValueError("hindrance exponent must be finite and non-negative")
    l = np.arange(n + 1, dtype=float)
    root = np.sqrt((1.0 + p) ** 2 + 4.0 * l * (l + 1.0))
    sp = 0.5 * (-(1.0 + p) + root)
    sm = 0.5 * (-(1.0 + p) - root)
    if e == 0.0:
        el = np.zeros(n + 1)
    elif np.isfinite(bi):
        el = e ** (sp - sm) * (bi + sm * e ** (1.0 - p)) / (bi + sp * e ** (1.0 - p))
    else:
        el = e ** (sp - sm)
    denom = -sm + el * sp
    w = 1.0 - (l + 0.5) / denom
    q = 1.0 - (l + 0.5) * (1.0 - el) / denom
    return np.vstack([sp, sm, el, w, q])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\norder = 12\neps = 0.625\nbiot = 2.5\nhindrance = 1.0\n',
         'call': 'dual_series_coefficients(order, eps, biot, hindrance)',
         'gold_call': '_oracle_dual_series_coefficients(order, eps, biot, hindrance)'},
        {'setup': 'import numpy as np\norder = 200\neps = 0.35\nbiot = np.inf\nhindrance = 0.0\n',
         'call': 'dual_series_coefficients(order, eps, biot, hindrance)',
         'gold_call': '_oracle_dual_series_coefficients(order, eps, biot, hindrance)'},
        {'setup': 'import numpy as np\norder = 40\neps = 0.0\nbiot = 3.0\nhindrance = 2.0\n',
         'call': 'dual_series_coefficients(order, eps, biot, hindrance)',
         'gold_call': '_oracle_dual_series_coefficients(order, eps, biot, hindrance)'},
        {'setup': 'import numpy as np\norder = 400\neps = 0.75\nbiot = 6.0\nhindrance = 0.5\n',
         'call': 'dual_series_coefficients(order, eps, biot, hindrance)',
         'gold_call': '_oracle_dual_series_coefficients(order, eps, biot, hindrance)'},
    ]
