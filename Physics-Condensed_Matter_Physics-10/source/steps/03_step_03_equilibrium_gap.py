"""
Solve the reduced BCS self-consistency condition for the positive equilibrium superconducting gap at a given temperature. The routine takes temperature in kelvin, a positive energy cutoff in millielectronvolts and a positive dimensionless coupling, then returns the gap in millielectronvolts as a single float. A collapsed superconducting solution returns zero rather than raising. Invalid input raises ValueError: temperature must be a finite non-negative scalar, while cutoff and coupling must be finite scalars strictly greater than zero.

The self-consistency condition equates the reduced gap kernel to the inverse dimensionless pairing coupling because the density of states has been absorbed into that coupling. For fixed temperature and cutoff the kernel decreases with positive gap magnitude, so any superconducting root is unique. When thermal occupation lowers the zero-gap kernel below the inverse coupling, the positive root vanishes and the zero return value represents the normal state without a superconducting order parameter.

Returns
-------
float giving the equilibrium superconducting gap in millielectronvolts, or 0.0 in the normal state
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def equilibrium_gap(temperature, cutoff, coupling):
    """Return the positive equilibrium superconducting gap or zero in the normal state.

    Parameters
    ----------
    temperature : float
        Temperature in kelvin, including the zero-temperature limit.
    cutoff : float
        Positive symmetric energy cutoff in millielectronvolts.
    coupling : float
        Positive dimensionless product of pairing strength and density of states.

    Returns
    -------
    float
        Equilibrium gap in millielectronvolts, or zero if no positive root exists.

    Raises
    ------
    ValueError
        If any input is not a finite scalar, if ``temperature`` is negative, or
        if ``cutoff`` or ``coupling`` is not strictly positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def _oracle_equilibrium_gap(temperature, cutoff, coupling):
    values = ((temperature, "temperature"), (cutoff, "cutoff"), (coupling, "coupling"))
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
    temperature, cutoff, coupling = converted
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if cutoff <= 0.0:
        raise ValueError("cutoff must be strictly greater than zero")
    if coupling <= 0.0:
        raise ValueError("coupling must be strictly greater than zero")

    def residual(gap):
        return _oracle_gap_kernel(gap, temperature, cutoff) - 1.0 / coupling

    lo = 1e-9
    # Parameters far outside the millielectronvolt range overflow inside the
    # kernel or while the bracket grows. The root finder cannot serve such an
    # input, so it is refused with the documented error rather than leaking an
    # arithmetic exception.
    try:
        h_lo = residual(lo)
        if h_lo < 0.0:
            # The kernel is largest at a vanishing gap, so a negative residual
            # there leaves no positive root and the normal state is the only
            # solution.
            return 0.0

        # The root moves far above the cutoff once the coupling is strong, since
        # the kernel falls off only logarithmically in the gap. A bracket fixed
        # at a small multiple of the cutoff therefore misses the root entirely
        # and, with a sign test alone to fall back on, reports the normal state
        # for a comfortably superconducting parameter set. Grow the upper end
        # until the residual actually changes sign instead of assuming a scale.
        hi = 5.0 * cutoff
        h_hi = residual(hi)
        for _ in range(200):
            if h_hi <= 0.0:
                break
            hi *= 2.0
            h_hi = residual(hi)
        if h_hi > 0.0:
            raise ValueError("no positive gap root was bracketed below the search limit")
        return float(brentq(residual, lo, hi, xtol=1e-14, rtol=1e-14))
    except (OverflowError, ZeroDivisionError, FloatingPointError) as exc:
        raise ValueError("no finite equilibrium gap at these magnitudes") from exc

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "",
            "call": "round(float(equilibrium_gap(0.01, 2.6, 1.11)), 12)",
            "gold_call": "round(float(_oracle_equilibrium_gap(0.01, 2.6, 1.11)), 12)",
        },
        {
            "setup": "",
            "call": "round(float(equilibrium_gap(15.3, 2.6, 1.11)), 12)",
            "gold_call": "round(float(_oracle_equilibrium_gap(15.3, 2.6, 1.11)), 12)",
        },
        {
            "setup": "",
            "call": "round(float(equilibrium_gap(0.0, 1.0, 0.2)), 12)",
            "gold_call": "round(float(_oracle_equilibrium_gap(0.0, 1.0, 0.2)), 12)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "round(float(equilibrium_gap(0.0, 2.6, 1.11) - 2.6 / np.sinh(1.0 / 1.11)), 12)",
            "gold_call": "round(float(_oracle_equilibrium_gap(0.0, 2.6, 1.11) - 2.6 / np.sinh(1.0 / 1.11)), 12)",
        },
        {
            "setup": "def run_model():\n    try:\n        equilibrium_gap(-1.0, 2.6, 1.11)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n\ndef run_gold():\n    try:\n        _oracle_equilibrium_gap(-1.0, 2.6, 1.11)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
