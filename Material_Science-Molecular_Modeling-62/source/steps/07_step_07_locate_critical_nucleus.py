"""
Locate the maximum of the available-energy increment and return the critical nucleus radius and the critical activation energy.

Differentiating the available-energy increment with respect to the nucleus radius and dividing through by the radius leaves a stationarity condition holding three terms: twice the balance coefficient, twice the Laplace coefficient multiplied by the logarithm of one plus the capillary length divided by the radius, and minus the Laplace coefficient multiplied by the capillary length divided by the sum of the radius and the capillary length. The radius appears both inside a logarithm and inside a rational term, so the condition is transcendental and the stationary radius has no elementary closed form. This is the price of keeping the logarithm exact, and it is worth paying, because expanding the logarithm to first order would collapse the increment to a quadratic whose extremum is elementary but whose location is wrong by a large factor once the Laplace overpressure is comparable with the liquid pressure.




The condition is nevertheless easy to solve reliably, because the left-hand side is strictly decreasing in the radius. Differentiating it shows the logarithmic term falling faster than the rational term rises, so the whole expression decreases monotonically from an unbounded positive value as the radius tends to zero, where the logarithm diverges, down to twice the balance coefficient as the radius grows without bound. When the balance coefficient is negative, that limit is negative, so the expression changes sign exactly once and a bracketing search followed by bisection converges to the unique root without any starting guess and without any risk of finding the wrong stationary point. Substituting the root back into the increment gives the critical activation energy, and monotonicity guarantees that the stationary point is the maximum rather than a minimum.

Treating the sign of the balance coefficient as a precondition rather than as a detail is what keeps the result physical. When the coefficient is non-negative, the vaporization cost is never repaid by the local superheat, the stationarity condition is positive everywhere and has no root, the increment increases over the whole positive axis, and no barrier exists; reporting a number for such a state would announce a finite barrier where there is none. The sensitivity of the result is also worth noting: the activation energy grows without bound as a state approaches the threshold where the balance coefficient vanishes, so states near that threshold dominate any comparison across a measured table.

Returns
-------
np.ndarray of shape (2,), float: the critical nucleus radius in metres and the critical activation energy in joules.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def locate_critical_nucleus(coefficients: np.ndarray) -> np.ndarray:
    """Locate the maximum of the available-energy increment.

    Parameters
    ----------
    coefficients : np.ndarray
        Array of shape (3,) holding the balance coefficient in J/m^2, the
        Laplace coefficient in J/m^2 and the capillary length in metres.

    Returns
    -------
    critical_state : np.ndarray
        Array of shape (2,) holding the critical nucleus radius in metres and
        the critical activation energy in joules.
    """
    return critical_state  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_locate_critical_nucleus(coefficients: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(coefficients, dtype=float).ravel()
    if coeffs.size != 3 or not np.all(np.isfinite(coeffs)):
        raise ValueError("coefficients must hold three finite entries")

    balance, laplace, capillary = coeffs
    if balance >= 0.0:
        raise ValueError("no barrier exists unless the balance coefficient is negative")
    if laplace <= 0.0:
        raise ValueError("the Laplace coefficient must be strictly positive")
    if capillary <= 0.0:
        raise ValueError("the capillary length must be strictly positive")

    def _stationarity(radius):
        # Derivative of the increment with respect to the radius, divided by the
        # radius. Strictly decreasing, unbounded above as the radius tends to
        # zero and tending to twice the balance coefficient as it grows.
        return (2.0 * balance + 2.0 * laplace * np.log1p(capillary / radius)
                - laplace * capillary / (radius + capillary))

    # Bracket the sign change, then bisect. No starting guess is needed because
    # the stationarity condition is monotone.
    upper = capillary
    for _ in range(4000):
        if _stationarity(upper) < 0.0:
            break
        upper *= 2.0
    else:
        raise ValueError("failed to bracket the stationary radius from above")

    lower = upper
    for _ in range(4000):
        lower *= 0.5
        if _stationarity(lower) > 0.0:
            break
    else:
        raise ValueError("failed to bracket the stationary radius from below")

    for _ in range(200):
        middle = 0.5 * (lower + upper)
        if _stationarity(middle) > 0.0:
            lower = middle
        else:
            upper = middle

    radius_critical = 0.5 * (lower + upper)
    energy_critical = radius_critical ** 2 * (
        balance + laplace * np.log1p(capillary / radius_critical))

    return np.array([radius_critical, energy_critical], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: governing state of the testbed (normal scenario) ---
        {
            "setup": """import numpy as np
coefficients = np.array([-9.1325677596e-04, 6.9227891610e-04, 1.3362941030e-06])
""",
            "call": "locate_critical_nucleus(coefficients)",
            "gold_call": "_oracle_locate_critical_nucleus(coefficients)",
        },
        # --- Valid: hydrophobic state at the strongest driving force ---
        {
            "setup": """import numpy as np
coefficients = np.array([-2.6549062633e-04, 1.9787305158e-04, 1.3362941030e-06])
""",
            "call": "locate_critical_nucleus(coefficients)",
            "gold_call": "_oracle_locate_critical_nucleus(coefficients)",
        },
        # --- Boundary: balance coefficient close to zero, so the barrier is very large ---
        {
            "setup": """import numpy as np
coefficients = np.array([-1.0e-09, 1.0e-04, 1.3362941030e-06])
""",
            "call": "locate_critical_nucleus(coefficients)",
            "gold_call": "_oracle_locate_critical_nucleus(coefficients)",
        },
        # --- Edge: unit coefficients, far from the physical scales ---
        {
            "setup": """import numpy as np
coefficients = np.array([-1.0, 1.0, 1.0])
""",
            "call": "locate_critical_nucleus(coefficients)",
            "gold_call": "_oracle_locate_critical_nucleus(coefficients)",
        },
        # --- Invalid: non-negative balance coefficient, so no barrier exists ---
        {
            "setup": """import numpy as np
coefficients = np.array([2.5e-04, 1.0e-04, 1.3362941030e-06])
def run_model():
    try:
        locate_critical_nucleus(coefficients)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_locate_critical_nucleus(coefficients)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive Laplace coefficient ---
        {
            "setup": """import numpy as np
coefficients = np.array([-1.0e-03, 0.0, 1.3362941030e-06])
def run_model():
    try:
        locate_critical_nucleus(coefficients)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_locate_critical_nucleus(coefficients)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
