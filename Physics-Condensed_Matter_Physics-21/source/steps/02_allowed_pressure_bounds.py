"""
Evaluate the smallest and largest pressure a stable, causal and thermodynamically consistent equation of state may take at a given chemical potential and number density.

Fixing the chemical potential slices the allowed three-dimensional region into a plane figure in the remaining variables. That slice is a triangle: the number density is confined between the two envelopes of the previous step, and at each admissible density the pressure is confined between a lower edge that does not depend on the density and an upper edge that does.

The lower edge follows from integrating the thermodynamic relation outwards from the low-density endpoint along the stiffest admissible history, so it is fixed once the minimum density at that chemical potential is known. The upper edge is piecewise: below a crossover density it is controlled by the low-density endpoint, and above it by the high-density endpoint, because a state that sits at high density must leave enough room to reach the high-density endpoint without violating causality.

The triangular shape of the slice is what makes the region tractable to sample directly, and the width of the pressure interval at each density is the quantity that weights the density marginal in the sampling construction that follows.

Returns
-------
np.ndarray, shape (2,), [p_min, p_max] in GeV fm^-3
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def allowed_pressure_bounds(chemical_potential: float, density: float,
                            beta_low: "np.ndarray", beta_high: "np.ndarray") -> "np.ndarray":
    '''Compute the pressure edges of the allowed slice at one chemical potential and density.

    Parameters
    ----------
    chemical_potential : float
        Finite chemical potential in GeV inside the endpoint interval.
    density : float
        Finite number density in fm^-3, between the density bounds returned by
        the preceding step at this chemical potential.
    beta_low, beta_high : np.ndarray
        Finite shape (3,) endpoints (mu, n, p) in GeV, fm^-3 and GeV fm^-3.

    Returns
    -------
    bounds : np.ndarray
        Finite shape (2,) in GeV fm^-3, the minimum and maximum allowed pressure
        in that order. The minimum is pL + (mu**2 - muL**2)/(2*mu) times the
        minimum allowed density at this chemical potential. Writing
        n_c = nmax(muL)*mu/muL for the crossover density, the maximum is
        pL + (mu**2 - muL**2)/(2*mu)*density when density <= n_c and
        pH - (muH**2 - mu**2)/(2*mu)*density otherwise.

    Raises
    ------
    ValueError
        If the inputs are not finite real values of the stated shapes, if a
        preceding step rejects the endpoints or chemical potential, if the
        density lies outside its allowed bounds, or if the evaluated pressure
        bounds are not finite.
    '''
    return bounds

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_allowed_pressure_bounds(chemical_potential: float, density: float,
                                    beta_low: "np.ndarray",
                                    beta_high: "np.ndarray") -> "np.ndarray":
    span = _oracle_allowed_density_bounds(chemical_potential, beta_low, beta_high)
    low, high, _ = _validated_endpoints(beta_low, beta_high)
    if not np.isscalar(density) or not np.isrealobj(density):
        raise ValueError("the density must be a real scalar")
    n = float(density)
    if not np.isfinite(n) or not span[0] <= n <= span[1]:
        raise ValueError("the density must lie within its allowed bounds")
    mu = float(chemical_potential)
    (mu_l, n_l, p_l), (mu_h, n_h, p_h) = low, high
    stiff = (mu**2 - mu_l**2) / (2.0 * mu)
    crossover_density = _oracle_allowed_density_bounds(mu_l, beta_low, beta_high)[1] * mu / mu_l
    lower = p_l + stiff * span[0]
    if n <= crossover_density:
        upper = p_l + stiff * n
    else:
        upper = p_h - (mu_h**2 - mu**2) / (2.0 * mu) * n
    bounds = np.array([lower, upper], dtype=float)
    if not np.all(np.isfinite(bounds)):
        raise ValueError("the evaluated pressure bounds must be finite")
    return bounds

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    anchors = ("import numpy as np\n"
               "low = np.array([1.00, 0.32, 0.018])\n"
               "high = np.array([2.60, 4.80, 3.500])\n")
    guard = ('def run_model():\n'
             '    try:\n'
             '        allowed_pressure_bounds(mu, n, low.copy(), high.copy())\n'
             '        return 0\n'
             '    except ValueError:\n'
             '        return 1\n'
             'def run_oracle():\n'
             '    try:\n'
             '        _oracle_allowed_pressure_bounds(mu, n, low.copy(), high.copy())\n'
             '        return 0\n'
             '    except ValueError:\n'
             '        return 1\n')
    return [
        # --- Normal: below the crossover density, upper edge set by the low endpoint ---
        {"setup": anchors + "mu = 1.40\nn = 0.60\n",
         "call": "allowed_pressure_bounds(mu, n, low.copy(), high.copy())",
         "gold_call": "_oracle_allowed_pressure_bounds(mu, n, low.copy(), high.copy())"},
        # --- Normal: above the crossover density, upper edge set by the high endpoint ---
        {"setup": anchors + "mu = 1.40\nn = 1.80\n",
         "call": "allowed_pressure_bounds(mu, n, low.copy(), high.copy())",
         "gold_call": "_oracle_allowed_pressure_bounds(mu, n, low.copy(), high.copy())"},
        # --- Boundary: at the minimum allowed density, where the slice closes to a point ---
        {"setup": anchors + "mu = 1.40\nn = float(_oracle_allowed_density_bounds(mu, low.copy(), high.copy())[0])\n",
         "call": "allowed_pressure_bounds(mu, n, low.copy(), high.copy())",
         "gold_call": "_oracle_allowed_pressure_bounds(mu, n, low.copy(), high.copy())"},
        # --- Normal: near the high-density endpoint ---
        {"setup": anchors + "mu = 2.40\nn = 4.20\n",
         "call": "allowed_pressure_bounds(mu, n, low.copy(), high.copy())",
         "gold_call": "_oracle_allowed_pressure_bounds(mu, n, low.copy(), high.copy())"},
        # --- Normal: a different endpoint pair ---
        {"setup": ("import numpy as np\n"
                   "low = np.array([1.05, 0.40, 0.025])\n"
                   "high = np.array([2.40, 4.00, 2.800])\n"
                   "mu = 1.80\nn = 2.00\n"),
         "call": "allowed_pressure_bounds(mu, n, low.copy(), high.copy())",
         "gold_call": "_oracle_allowed_pressure_bounds(mu, n, low.copy(), high.copy())"},
        # --- Invalid: a density below the allowed minimum must raise ValueError ---
        {"setup": anchors + "mu = 1.40\nn = 0.10\n" + guard,
         "call": "run_model()", "gold_call": "run_oracle()"},
        # --- Invalid: a density above the allowed maximum must raise ValueError ---
        {"setup": anchors + "mu = 1.40\nn = 9.00\n" + guard,
         "call": "run_model()", "gold_call": "run_oracle()"},
    ]
