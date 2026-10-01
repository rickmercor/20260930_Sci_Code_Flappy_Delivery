"""
Evaluate the reduced inertia coefficient for a positive gap, a non-negative temperature and a positive energy cutoff. The routine takes the gap magnitude and cutoff in millielectronvolts and the temperature in kelvin, then returns the coefficient divided by the constant density of states as a single float obtained by numerical quadrature over the band. Invalid input raises ValueError: the gap and cutoff must be finite scalars strictly greater than zero and the temperature must be a finite non-negative scalar.

This coefficient gives the order parameter inertia so its amplitude can oscillate rather than merely relax. It follows from the microscopic finite-temperature free energy rather than being a free parameter. The companion stiffness is the same integral scaled so that the resulting equation is Lorentz covariant.

Returns
-------
float giving the reduced inertia coefficient
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.integrate import quad


def inertia_coefficient(gap, temperature, cutoff):
    """Return the reduced inertia coefficient within the symmetric energy cutoff.

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
        Inertia coefficient divided by the constant density of states.

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


def _oracle_inertia_coefficient(gap, temperature, cutoff):
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

    kb_mev_per_k = 0.08617333262

    def integrand(xi):
        # The bracket is (2 f(E) - 1) / (2 E). With 2 f(E) - 1 = -tanh(E / 2 k T)
        # its energy derivative is available in closed form, so no step size
        # enters and the value is exact to quadrature accuracy. A finite
        # difference here would leave a truncation error of order 1e-9, which is
        # the same size as the comparison tolerance and would fail a solver who
        # differentiated exactly.
        E = np.sqrt(xi**2 + gap**2)
        if temperature <= 0.0:
            derivative = 1.0 / (2.0 * E * E)
        else:
            x = E / (2.0 * kb_mev_per_k * temperature)
            # cosh overflows near x = 710 while sech squared has already
            # underflowed to zero, so take that limit directly rather than
            # dividing by an infinity.
            sech2 = 0.0 if x > 350.0 else 1.0 / np.cosh(x) ** 2
            derivative = np.tanh(x) / (2.0 * E * E) - sech2 / (4.0 * kb_mev_per_k * temperature * E)
        return derivative / (2.0 * E)

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
            "call": "float('%.7g' % inertia_coefficient(1.7, 8.0, 2.6))",
            "gold_call": "float('%.7g' % _oracle_inertia_coefficient(1.7, 8.0, 2.6))",
        },
        {
            "setup": "",
            "call": "float('%.7g' % inertia_coefficient(2.0, 0.0, 2.6))",
            "gold_call": "float('%.7g' % _oracle_inertia_coefficient(2.0, 0.0, 2.6))",
        },
        {
            "setup": "",
            "call": "float('%.7g' % inertia_coefficient(0.05, 10.0, 2.6))",
            "gold_call": "float('%.7g' % _oracle_inertia_coefficient(0.05, 10.0, 2.6))",
        },
        {
            "setup": "",
            "call": "int(inertia_coefficient(1.5, 4.0, 2.6) > inertia_coefficient(1.5, 12.0, 2.6))",
            "gold_call": "int(_oracle_inertia_coefficient(1.5, 4.0, 2.6) > _oracle_inertia_coefficient(1.5, 12.0, 2.6))",
        },
        {
            "setup": "def run_model():\n    try:\n        inertia_coefficient(0.0, 7.0, 2.6)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n\ndef run_gold():\n    try:\n        _oracle_inertia_coefficient(0.0, 7.0, 2.6)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
