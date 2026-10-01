"""
Locate the critical temperature by bisecting the equilibrium superconducting gap over temperature. The routine takes a positive energy cutoff in millielectronvolts and a positive dimensionless coupling, then returns in kelvin the temperature where the gap reaches a threshold of 1e-6 millielectronvolts within the fixed search window from 0.01 to 100.0 kelvin. Invalid input raises ValueError: cutoff and coupling must be finite scalars strictly greater than zero, and the search bracket must contain a sign change.

Here the critical temperature marks the loss of the nonzero self-consistent superconducting root. Its collapse is equivalent to satisfying the linearized zero-gap condition because the nonlinear gap equation approaches that condition continuously as the order parameter vanishes. A small positive threshold is used instead of exact zero so that the normal-state zero returned above the transition gives a negative residual and brackets the edge of the collapsed solution rather than making every normal-state temperature an endpoint root.

Returns
-------
float giving the critical temperature in kelvin
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def critical_temperature(cutoff, coupling):
    """Return the temperature where the equilibrium superconducting gap collapses.

    Parameters
    ----------
    cutoff : float
        Positive symmetric energy cutoff in millielectronvolts.
    coupling : float
        Positive dimensionless product of pairing strength and density of states.

    Returns
    -------
    float
        Critical temperature in kelvin.

    Raises
    ------
    ValueError
        If either input is not a finite scalar strictly greater than zero, or if
        no transition is found from 0.01 to 100.0 kelvin.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def _oracle_critical_temperature(cutoff, coupling):
    values = ((cutoff, "cutoff"), (coupling, "coupling"))
    converted = []
    for value, name in values:
        if not np.isscalar(value) or isinstance(value, (str, bytes)):
            raise ValueError(name + " must be a finite scalar")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be a finite scalar") from exc
        if not np.isfinite(numeric):
            raise ValueError(name + " must be a finite scalar")
        converted.append(numeric)
    cutoff, coupling = converted
    if cutoff <= 0.0:
        raise ValueError("cutoff must be strictly greater than zero")
    if coupling <= 0.0:
        raise ValueError("coupling must be strictly greater than zero")

    lo = 0.01
    hi = 100.0
    threshold = 1e-6

    def residual(temperature):
        return _oracle_equilibrium_gap(temperature, cutoff, coupling) - threshold

    # A cutoff or coupling far outside the millielectronvolt range overflows
    # inside the gap solver. That is a refusal of the input rather than a
    # result, so it is reported as the documented ValueError.
    try:
        h_lo = residual(lo)
        h_hi = residual(hi)
        if not ((h_lo < 0.0 < h_hi) or (h_hi < 0.0 < h_lo)):
            raise ValueError("no transition found in the temperature search window")
        return float(brentq(residual, lo, hi, xtol=1e-10))
    except (OverflowError, ZeroDivisionError, FloatingPointError) as exc:
        raise ValueError("no finite critical temperature at these magnitudes") from exc

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "",
            "call": "round(float(critical_temperature(2.6, 1.11)), 6)",
            "gold_call": "round(float(_oracle_critical_temperature(2.6, 1.11)), 6)",
        },
        {
            "setup": "",
            "call": "int(critical_temperature(2.6, 1.30) > critical_temperature(2.6, 0.80))",
            "gold_call": "int(_oracle_critical_temperature(2.6, 1.30) > _oracle_critical_temperature(2.6, 0.80))",
        },
        {
            "setup": "",
            "call": "round(float(critical_temperature(5.2, 1.11) / critical_temperature(2.6, 1.11)), 6)",
            "gold_call": "round(float(_oracle_critical_temperature(5.2, 1.11) / _oracle_critical_temperature(2.6, 1.11)), 6)",
        },
        {
            "setup": "import numpy as np\ndef run_model():\n    for args in ((np.nan, 1.11), (2.6, 0.0), (2.6, 1e-6)):\n        try:\n            critical_temperature(*args)\n        except ValueError:\n            continue\n        except Exception:\n            return 2\n        return 0\n    return 1\n\ndef run_gold():\n    for args in ((np.nan, 1.11), (2.6, 0.0), (2.6, 1e-6)):\n        try:\n            _oracle_critical_temperature(*args)\n        except ValueError:\n            continue\n        except Exception:\n            return 2\n        return 0\n    return 1\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
