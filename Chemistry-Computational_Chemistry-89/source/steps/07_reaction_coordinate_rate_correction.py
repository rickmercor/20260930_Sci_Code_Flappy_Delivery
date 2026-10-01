"""
Compute the complete first-order, order-hbar relative correction to the thermal rate constant of motion on a one-dimensional Eckart barrier, at fixed thermal time.

The term collects every contribution of relative order hbar for that coordinate, and it is defined only below the crossover temperature of the barrier, where the tunnelling orbit exists. No chain of beads is involved: the function takes the barrier parameters and the temperature and returns one number.

Returns
-------
float: the complete first-order relative rate correction for one-dimensional motion on the Eckart barrier (dimensionless)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def eckart_tunneling_correction(beta: float, V0: float, V_inf: float, a: float, m: float) -> float:
    '''First-order relative rate correction for a one-dimensional Eckart barrier.

    Parameters
    ----------
    beta : float
        Inverse temperature in 1/hartree (hbar = k_B = 1). It must exceed the
        crossover value of this barrier.
    V0 : float
        Amplitude of the symmetric hyperbolic-secant-squared component of the
        barrier, in hartree; V0 > 0.
    V_inf : float
        Product asymptote relative to the reactant asymptote, in hartree,
        with -4 V0 < V_inf < 4 V0 so that the barrier has an interior
        maximum.
    a : float
        Range parameter of the barrier, the length that scales the
        coordinate in it, in bohr; a > 0.
    m : float
        Mass of the coordinate in electron masses; m > 0.

    Returns
    -------
    correction : float
        The complete first-order relative correction for this coordinate,
        dimensionless, defined as the term to be added to 1 in the corrected
        rate constant.

    Raises
    ------
    ValueError
        If V0, a or m is not positive, V_inf is outside (-4 V0, 4 V0), or
        beta is at or below the crossover value, where no tunnelling orbit
        exists.
    '''
    return correction

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def _eckart_action_derivative(e, order, gam, V0, V_inf):
    """n-th energy derivative (n = 1..4) of W(E) = 2 gam - gam [sqrt(E/V0) + sqrt((E - V_inf)/V0)]."""
    coef = {1: 0.5, 2: -0.25, 3: 0.375, 4: -0.9375}[order]
    return -gam / np.sqrt(V0) * coef * (e ** (0.5 - order) + (e - V_inf) ** (0.5 - order))


def _oracle_eckart_tunneling_correction(beta: float, V0: float, V_inf: float, a: float, m: float) -> float:
    beta, V0, V_inf, a, m = (float(v) for v in (beta, V0, V_inf, a, m))
    if not (V0 > 0.0 and a > 0.0 and m > 0.0):
        raise ValueError("V0, a and m must be positive")
    if not -4.0 * V0 < V_inf < 4.0 * V0:
        raise ValueError("V_inf must lie in (-4 V0, 4 V0)")
    gam = np.pi * np.sqrt(2.0 * m * a ** 2 * V0)
    e_top = (4.0 * V0 + V_inf) ** 2 / (16.0 * V0)
    e_low = max(0.0, V_inf)
    beta_c = -_eckart_action_derivative(e_top, 1, gam, V0, V_inf)
    if not beta > beta_c:
        raise ValueError("beta must exceed the crossover value for a deep-tunneling instanton")
    period = lambda e: -_eckart_action_derivative(e, 1, gam, V0, V_inf) - beta
    lo = e_low + (e_top - e_low) * 1e-15
    while period(lo) < 0.0:
        lo = e_low + (lo - e_low) * 1e-6
    e_st = brentq(period, lo, e_top, xtol=1e-300, rtol=4.0 * np.finfo(float).eps, maxiter=500)
    w2, w3, w4 = (_eckart_action_derivative(e_st, k, gam, V0, V_inf) for k in (2, 3, 4))
    shape = -0.125 * w4 / w2 ** 2 + (5.0 / 24.0) * w3 ** 2 / w2 ** 3
    return float(shape + np.pi ** 2 / (4.0 * gam))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: exothermic barrier of the benchmark surface.
        {
            "setup": "import numpy as np",
            "call": "eckart_tunneling_correction(2975.0, 0.01135, -0.00485, 0.735, 1836.15267)",
            "gold_call": "_oracle_eckart_tunneling_correction(2975.0, 0.01135, -0.00485, 0.735, 1836.15267)",
        },
        # Normal: reduced units (m = 1), a much wider barrier at twice its crossover value.
        {
            "setup": "import numpy as np",
            "call": "eckart_tunneling_correction(4.0 * np.pi, 13.5 / np.pi, -18.0 / np.pi, 8.0 / np.sqrt(3.0 * np.pi), 1.0)",
            "gold_call": "_oracle_eckart_tunneling_correction(4.0 * np.pi, 13.5 / np.pi, -18.0 / np.pi, 8.0 / np.sqrt(3.0 * np.pi), 1.0)",
        },
        # Boundary: symmetric barrier, where the two shape contributions cancel.
        {
            "setup": "import numpy as np",
            "call": "eckart_tunneling_correction(2975.0, 0.01135, 0.0, 0.735, 1836.15267)",
            "gold_call": "_oracle_eckart_tunneling_correction(2975.0, 0.01135, 0.0, 0.735, 1836.15267)",
        },
        # Edge: endothermic barrier (V_inf > 0), so tunnelling energies start at the product asymptote.
        {
            "setup": "import numpy as np",
            "call": "eckart_tunneling_correction(4200.0, 0.01135, 0.0031, 0.735, 1836.15267)",
            "gold_call": "_oracle_eckart_tunneling_correction(4200.0, 0.01135, 0.0031, 0.735, 1836.15267)",
        },
        # Edge: just below the crossover temperature, where the orbit is barely delocalized.
        {
            "setup": "import numpy as np",
            "call": "eckart_tunneling_correction(1355.0, 0.01135, -0.00485, 0.735, 1836.15267)",
            "gold_call": "_oracle_eckart_tunneling_correction(1355.0, 0.01135, -0.00485, 0.735, 1836.15267)",
            "tol": 1e-7,
        },
        # Invalid: above the crossover temperature there is no tunnelling orbit.
        {
            "setup": """import numpy as np
def run_model():
    try:
        eckart_tunneling_correction(1000.0, 0.01135, -0.00485, 0.735, 1836.15267)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_eckart_tunneling_correction(1000.0, 0.01135, -0.00485, 0.735, 1836.15267)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
