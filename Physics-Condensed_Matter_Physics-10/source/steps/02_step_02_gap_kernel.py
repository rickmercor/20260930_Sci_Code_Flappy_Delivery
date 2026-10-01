"""
Evaluate the reduced BCS gap kernel for a positive gap, a non-negative temperature and a positive energy cutoff. The routine takes the gap magnitude and cutoff in millielectronvolts and the temperature in kelvin, then returns the constant-density-of-states momentum sum divided by that density of states as a single float. Invalid input raises ValueError: the gap and cutoff must be finite scalars strictly greater than zero and the temperature must be a finite non-negative scalar.

This kernel is the BCS gap self-consistency integrand because each paired quasiparticle state contributes an inverse-energy coherence weight whose thermal population reduces it through the factor (1 - 2f)/(2E). The quasiparticle energy combines band energy and gap magnitude, while particle-hole symmetry and a constant density of states reduce the momentum sum to a one-dimensional integral over band energy within the symmetric cutoff.

Returns
-------
float giving the reduced BCS gap kernel
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.integrate import quad


def gap_kernel(gap, temperature, cutoff):
    """Return the reduced BCS gap kernel within the symmetric energy cutoff.

    Parameters
    ----------
    gap : float
        Positive quasiparticle gap magnitude in millielectronvolts.
    temperature : float
        Temperature in kelvin, including the zero-temperature limit.
    cutoff : float
        Positive symmetric energy cutoff in millielectronvolts.

    Returns
    -------
    float
        Momentum integral divided by the constant density of states.

    Raises
    ------
    ValueError
        If any input is not a finite scalar, if ``gap`` or ``cutoff`` is not
        strictly positive, or if ``temperature`` is negative.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import quad


def _oracle_gap_kernel(gap, temperature, cutoff):
    values = ((gap, "gap"), (temperature, "temperature"), (cutoff, "cutoff"))
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
    gap, temperature, cutoff = converted
    if gap <= 0.0:
        raise ValueError("gap must be strictly greater than zero")
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if cutoff <= 0.0:
        raise ValueError("cutoff must be strictly greater than zero")

    def integrand(xi):
        E = np.sqrt(xi**2 + gap**2)
        occupation = float(np.asarray(_oracle_fermi_occupation(np.array([E]), temperature))[0])
        return (1.0 - 2.0 * occupation) / (2.0 * E)

    # Magnitudes far outside the millielectronvolt range overflow the squared
    # band energy or underflow the quasiparticle energy to zero, and the
    # quadrature then raises an arithmetic error or returns a non-finite number.
    # Both are refusals of the input rather than results, so report them as the
    # documented ValueError instead of leaking the arithmetic exception.
    try:
        result, _ = quad(integrand, -cutoff, cutoff, limit=200, points=[0.0])
    except (OverflowError, ZeroDivisionError, FloatingPointError) as exc:
        raise ValueError("the band integral does not evaluate to a finite number at these magnitudes") from exc
    if not np.isfinite(result):
        raise ValueError("the band integral does not evaluate to a finite number at these magnitudes")
    return float(result)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "",
            "call": "round(float(gap_kernel(1.7, 8.0, 2.6)), 12)",
            "gold_call": "round(float(_oracle_gap_kernel(1.7, 8.0, 2.6)), 12)",
        },
        {
            "setup": "",
            "call": "round(float(gap_kernel(2.0, 0.0, 2.6)), 12)",
            "gold_call": "round(float(_oracle_gap_kernel(2.0, 0.0, 2.6)), 12)",
        },
        {
            "setup": "",
            "call": "round(float(gap_kernel(1e-6, 10.0, 2.6)), 12)",
            "gold_call": "round(float(_oracle_gap_kernel(1e-6, 10.0, 2.6)), 12)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "round(float(gap_kernel(1.3, 0.0, 2.6) - np.arcsinh(2.6 / 1.3)), 12)",
            "gold_call": "round(float(_oracle_gap_kernel(1.3, 0.0, 2.6) - np.arcsinh(2.6 / 1.3)), 12)",
        },
        {
            "setup": "def run_model():\n    try:\n        gap_kernel(0.0, 7.0, 2.6)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n\ndef run_gold():\n    try:\n        _oracle_gap_kernel(0.0, 7.0, 2.6)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
