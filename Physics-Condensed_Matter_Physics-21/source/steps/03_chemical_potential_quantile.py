"""
Invert the marginal chemical-potential distribution of a uniform measure on the allowed region, returning the chemical potential at a requested quantile.

Defining a prior over the space of admissible equations of state requires more than knowing which states are allowed: it requires a probability measure on that space. Placing a uniform measure on the allowed three-dimensional region with the Euclidean metric in $$(\mu,n,p)$$ induces a non-uniform marginal in the chemical potential alone, because the area of the triangular slice varies with $$\mu$$.




That marginal is known in closed form. It rises from zero at the low-density endpoint, peaks at the crossover chemical potential where the two branches of the density envelopes meet, and falls back to zero at the high-density endpoint. Below the crossover it is proportional to $$\mu(\mu^2-\mu_L^2)/(\mu_H^2-\mu^2)$$ and above it to $$\mu(\mu_H^2-\mu^2)/(\mu^2-\mu_L^2)$$, with the two branches scaled so that the density is continuous at the crossover.




Sampling a chemical potential then amounts to inverting the cumulative distribution of this density. The cumulative distribution is strictly increasing between the endpoints, so the inverse is unique and may be obtained by any sufficiently accurate deterministic method.

Returns
-------
float, the chemical potential in GeV as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def chemical_potential_quantile(quantile: float, beta_low: "np.ndarray",
                                beta_high: "np.ndarray") -> float:
    '''Return the chemical potential at a requested quantile of the induced marginal.

    Parameters
    ----------
    quantile : float
        Finite quantile in [0, 1].
    beta_low, beta_high : np.ndarray
        Finite shape (3,) endpoints (mu, n, p) in GeV, fm^-3 and GeV fm^-3.

    Returns
    -------
    value : float
        Chemical potential in GeV as a native Python float, the point at which
        the cumulative distribution of the marginal density reaches the
        requested quantile. The marginal density is proportional to
        mu*(mu**2 - muL**2)/(muH**2 - mu**2) below the crossover chemical
        potential and to mu*(muH**2 - mu**2)/(mu**2 - muL**2) above it, each
        branch scaled so the density is continuous at the crossover. A quantile
        of 0 returns muL and a quantile of 1 returns muH. The value has an
        absolute accuracy of 1e-10 GeV.

    Raises
    ------
    ValueError
        If the quantile is not a finite real scalar in [0, 1], if a preceding
        step rejects the endpoints, or if the inversion does not converge to a
        finite value.
    '''
    return value

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def _marginal_density(mu, low, high, crossover):
    """Unnormalised marginal density of the uniform measure over the allowed region."""
    mu_l, mu_h = low[0], high[0]
    if mu < crossover:
        scale = (mu_h**2 - crossover**2) / (crossover**2 - mu_l**2)
        return mu * (mu**2 - mu_l**2) / (mu_h**2 - mu**2) * scale
    scale = (crossover**2 - mu_l**2) / (mu_h**2 - crossover**2)
    return mu * (mu_h**2 - mu**2) / (mu**2 - mu_l**2) * scale


def _marginal_mass(mu, low, high, crossover):
    """Cumulative mass of the marginal density from the low endpoint up to mu."""
    extra = (low, high, crossover)
    if mu <= crossover:
        return quad(_marginal_density, low[0], mu, args=extra, limit=200)[0]
    head = quad(_marginal_density, low[0], crossover, args=extra, limit=200)[0]
    return head + quad(_marginal_density, crossover, mu, args=extra, limit=200)[0]


def _quantile_shortfall(mu, low, high, crossover, target):
    """Cumulative mass up to mu, less the mass the requested quantile calls for."""
    return _marginal_mass(mu, low, high, crossover) - target


def _oracle_chemical_potential_quantile(quantile: float, beta_low: "np.ndarray",
                                        beta_high: "np.ndarray") -> float:
    low, high, crossover = _validated_endpoints(beta_low, beta_high)
    if not np.isscalar(quantile) or not np.isrealobj(quantile):
        raise ValueError("the quantile must be a real scalar")
    q = float(quantile)
    if not np.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError("the quantile must lie in [0, 1]")
    if q == 0.0:
        return float(low[0])
    if q == 1.0:
        return float(high[0])
    total = _marginal_mass(high[0], low, high, crossover)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("the marginal distribution must carry positive mass")
    target = q * total

    value = brentq(_quantile_shortfall, low[0], high[0],
                   args=(low, high, crossover, target), xtol=1e-12, rtol=1e-14, maxiter=200)
    if not np.isfinite(value):
        raise ValueError("the quantile inversion must return a finite value")
    return float(value)

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
             '        chemical_potential_quantile(q, low.copy(), high.copy())\n'
             '        return 0\n'
             '    except ValueError:\n'
             '        return 1\n'
             'def run_oracle():\n'
             '    try:\n'
             '        _oracle_chemical_potential_quantile(q, low.copy(), high.copy())\n'
             '        return 0\n'
             '    except ValueError:\n'
             '        return 1\n')
    return [
        # --- Normal: the median, the point used by the benchmark refinement ---
        {"setup": anchors + "q = 0.5\n",
         "call": "chemical_potential_quantile(q, low.copy(), high.copy())",
         "gold_call": "_oracle_chemical_potential_quantile(q, low.copy(), high.copy())", "tol": 1e-09},
        # --- Normal: a quantile in the lower branch, below the crossover ---
        {"setup": anchors + "q = 0.2\n",
         "call": "chemical_potential_quantile(q, low.copy(), high.copy())",
         "gold_call": "_oracle_chemical_potential_quantile(q, low.copy(), high.copy())", "tol": 1e-09},
        # --- Normal: a quantile in the upper branch, above the crossover ---
        {"setup": anchors + "q = 0.85\n",
         "call": "chemical_potential_quantile(q, low.copy(), high.copy())",
         "gold_call": "_oracle_chemical_potential_quantile(q, low.copy(), high.copy())", "tol": 1e-09},
        # --- Boundary: quantile zero returns the low endpoint exactly ---
        {"setup": anchors + "q = 0.0\n",
         "call": "chemical_potential_quantile(q, low.copy(), high.copy())",
         "gold_call": "_oracle_chemical_potential_quantile(q, low.copy(), high.copy())", "tol": 1e-09},
        # --- Boundary: quantile one returns the high endpoint exactly ---
        {"setup": anchors + "q = 1.0\n",
         "call": "chemical_potential_quantile(q, low.copy(), high.copy())",
         "gold_call": "_oracle_chemical_potential_quantile(q, low.copy(), high.copy())", "tol": 1e-09},
        # --- Normal: a different endpoint pair moves the crossover and the marginal ---
        {"setup": ("import numpy as np\n"
                   "low = np.array([1.05, 0.40, 0.025])\n"
                   "high = np.array([2.40, 4.00, 2.800])\n"
                   "q = 0.5\n"),
         "call": "chemical_potential_quantile(q, low.copy(), high.copy())",
         "gold_call": "_oracle_chemical_potential_quantile(q, low.copy(), high.copy())", "tol": 1e-09},
        # --- Invalid: a quantile outside [0, 1] must raise ValueError ---
        {"setup": anchors + "q = 1.4\n" + guard,
         "call": "run_model()", "gold_call": "run_oracle()"},
    ]
