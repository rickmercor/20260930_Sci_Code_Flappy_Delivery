"""
Evaluate the smallest and largest number density a stable, causal and thermodynamically consistent equation of state may take at a given chemical potential.

Cold, charge-neutral, beta-equilibrated bulk matter is described here by three thermodynamic variables: the baryon chemical potential $$\mu$$, the number density $$n$$ and the pressure $$p$$. When the equation of state is known at a low-density endpoint $$\beta_L=(\mu_L,n_L,p_L)$$ and at a high-density endpoint $$\beta_H=(\mu_H,n_H,p_H)$$, the requirement that some physically admissible function connect the two is far from vacuous: it confines every intermediate state to a bounded three-dimensional region.




Mechanical stability requires $$n(\mu)$$ to be single valued, causality requires $$\frac{\mu}{n}\frac{dn}{d\mu}\ge1$$ so that the speed of sound does not exceed the speed of light, and thermodynamic consistency requires the pressure difference between the endpoints to equal $$p_H-p_L=\int_{\mu_L}^{\mu_H}n(\mu)\,d\mu$$. The complementary identity $$\varepsilon_H-\varepsilon_L=\int_{n_L}^{n_H}\mu(n)\,dn$$ gives the energy-density difference. Taken together these conditions bound the density at each chemical potential between two analytic envelopes, each of which switches form at a crossover chemical potential fixed by the endpoints.




This step evaluates those two envelopes. The crossover chemical potential separates the branch on which the lower envelope follows a straight ray from the low-density endpoint from the branch on which it is controlled by the high-density endpoint, and conversely for the upper envelope.

Returns
-------
np.ndarray, shape (2,), [n_min, n_max] in fm^-3
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def allowed_density_bounds(chemical_potential: float, beta_low: "np.ndarray",
                           beta_high: "np.ndarray") -> "np.ndarray":
    '''Compute the density envelopes of the allowed region at one chemical potential.

    Parameters
    ----------
    chemical_potential : float
        Finite chemical potential in GeV, with beta_low[0] <= chemical_potential
        <= beta_high[0].
    beta_low, beta_high : np.ndarray
        Positive finite shape (3,) endpoints (mu, n, p) in GeV, fm^-3 and GeV fm^-3, with
        beta_low[0] < beta_high[0], beta_low[1] < beta_high[1] and
        beta_low[2] < beta_high[2]. They must enclose a nonempty allowed region:
        muL*nH - muH*nL > 0 and the finite real crossover mu_c defined below
        must lie strictly between muL and muH.

    Returns
    -------
    bounds : np.ndarray
        Finite shape (2,) in fm^-3, the minimum and maximum allowed number
        density in that order. Writing dp for the pressure difference and
        mu_c = sqrt(muL*muH*(muH*nH - muL*nL - 2*dp)/(muL*nH - muH*nL)) for the
        crossover chemical potential, the minimum is nL*mu/muL when
        mu <= mu_c and (mu**3*nH - mu*muH*(muH*nH - 2*dp))/((mu**2 - muL**2)*muH)
        otherwise, while the maximum is
        (mu**3*nL - muL*mu*(muL*nL + 2*dp))/((mu**2 - muH**2)*muL) when mu < mu_c
        and nH*mu/muH otherwise.

    Raises
    ------
    ValueError
        If the inputs are not finite real values of the stated shapes, if the
        endpoint positivity or ordering fails, if the endpoints do not enclose
        the nonempty allowed region specified above, if the chemical potential
        lies outside the endpoint interval, or if the evaluated bounds are not
        finite.
    '''
    return bounds

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _validated_endpoints(beta_low, beta_high):
    """Validate a pair of endpoints and return them with the crossover potential."""
    if any(not np.isrealobj(b) for b in (beta_low, beta_high)):
        raise ValueError("endpoints must be real")
    try:
        low = np.asarray(beta_low, dtype=float)
        high = np.asarray(beta_high, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("endpoints must be finite real arrays") from exc
    if low.shape != (3,) or high.shape != (3,):
        raise ValueError("each endpoint must have shape (3,)")
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError("endpoints must be finite")
    if not np.all(low > 0.0) or not np.all(high > 0.0):
        raise ValueError("endpoint values must be positive")
    if not np.all(low < high):
        raise ValueError("every low endpoint entry must lie below its high counterpart")
    (mu_l, n_l, p_l), (mu_h, n_h, p_h) = low, high
    denominator = mu_l * n_h - mu_h * n_l
    numerator = mu_l * mu_h * (mu_h * n_h - mu_l * n_l - 2.0 * (p_h - p_l))
    if denominator <= 0.0 or numerator <= 0.0:
        raise ValueError("the endpoints do not enclose a non-empty allowed region")
    crossover = np.sqrt(numerator / denominator)
    if not np.isfinite(crossover) or not mu_l < crossover < mu_h:
        raise ValueError("the crossover chemical potential must lie between the endpoints")
    return low, high, crossover


def _oracle_allowed_density_bounds(chemical_potential: float, beta_low: "np.ndarray",
                                   beta_high: "np.ndarray") -> "np.ndarray":
    low, high, crossover = _validated_endpoints(beta_low, beta_high)
    if not np.isscalar(chemical_potential) or not np.isrealobj(chemical_potential):
        raise ValueError("the chemical potential must be a real scalar")
    mu = float(chemical_potential)
    if not np.isfinite(mu) or not low[0] <= mu <= high[0]:
        raise ValueError("the chemical potential must lie within the endpoint interval")
    (mu_l, n_l, p_l), (mu_h, n_h, p_h) = low, high
    gap = p_h - p_l
    if mu <= crossover:
        lower = n_l * mu / mu_l
    else:
        lower = (mu**3 * n_h - mu * mu_h * (mu_h * n_h - 2.0 * gap)) / ((mu**2 - mu_l**2) * mu_h)
    if mu < crossover:
        upper = (mu**3 * n_l - mu_l * mu * (mu_l * n_l + 2.0 * gap)) / ((mu**2 - mu_h**2) * mu_l)
    else:
        upper = n_h * mu / mu_h
    bounds = np.array([lower, upper], dtype=float)
    if not np.all(np.isfinite(bounds)):
        raise ValueError("the evaluated density bounds must be finite")
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
             '        allowed_density_bounds(mu, low.copy(), high.copy())\n'
             '        return 0\n'
             '    except ValueError:\n'
             '        return 1\n'
             'def run_oracle():\n'
             '    try:\n'
             '        _oracle_allowed_density_bounds(mu, low.copy(), high.copy())\n'
             '        return 0\n'
             '    except ValueError:\n'
             '        return 1\n')
    return [
        # --- Boundary: the low-density endpoint, where the lower envelope meets n_L ---
        {"setup": anchors + "mu = 1.00\n",
         "call": "allowed_density_bounds(mu, low.copy(), high.copy())",
         "gold_call": "_oracle_allowed_density_bounds(mu, low.copy(), high.copy())"},
        # --- Normal: below the crossover, where the lower envelope is the straight ray ---
        {"setup": anchors + "mu = 1.40\n",
         "call": "allowed_density_bounds(mu, low.copy(), high.copy())",
         "gold_call": "_oracle_allowed_density_bounds(mu, low.copy(), high.copy())"},
        # --- Normal: above the crossover, where both envelopes have switched branch ---
        {"setup": anchors + "mu = 2.20\n",
         "call": "allowed_density_bounds(mu, low.copy(), high.copy())",
         "gold_call": "_oracle_allowed_density_bounds(mu, low.copy(), high.copy())"},
        # --- Boundary: the high-density endpoint, where the upper envelope meets n_H ---
        {"setup": anchors + "mu = 2.60\n",
         "call": "allowed_density_bounds(mu, low.copy(), high.copy())",
         "gold_call": "_oracle_allowed_density_bounds(mu, low.copy(), high.copy())"},
        # --- Normal: a different endpoint pair, moving the crossover ---
        {"setup": ("import numpy as np\n"
                   "low = np.array([1.05, 0.40, 0.025])\n"
                   "high = np.array([2.40, 4.00, 2.800])\n"
                   "mu = 1.80\n"),
         "call": "allowed_density_bounds(mu, low.copy(), high.copy())",
         "gold_call": "_oracle_allowed_density_bounds(mu, low.copy(), high.copy())"},
        # --- Invalid: a chemical potential outside the endpoint interval ---
        {"setup": anchors + "mu = 3.10\n" + guard,
         "call": "run_model()", "gold_call": "run_oracle()"},
        # --- Invalid: endpoints whose pressure difference empties the allowed region ---
        {"setup": ("import numpy as np\n"
                   "low = np.array([1.00, 0.32, 0.018])\n"
                   "high = np.array([2.60, 4.80, 6.200])\n"
                   "mu = 1.50\n") + guard,
         "call": "run_model()", "gold_call": "run_oracle()"},
    ]
