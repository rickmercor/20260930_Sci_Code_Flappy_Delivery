"""
Build the reciprocal product associated with a bounded expansion variable and a non-empty sequence of mapped poles. Each input is validated as a finite scalar, every pole contributes one factor with itself and its conjugate and the completed product is returned as a native Python complex number. Invalid input raises ValueError when the evaluation point is not finite, the pole collection is not a non-empty sequence, a pole is not finite, the evaluation point coincides with a pole or its conjugate or the nonzero final reciprocal product overflows or underflows the finite complex range. Intermediate products must be scaled so representable results are not rejected merely because the poles span widely separated magnitudes.

For variable z and mapped poles p, the result is the reciprocal of the product of (z - p)(z - conjugate(p)) over every pole, with no additional constant. Accumulate the complete product using scaled arithmetic so large and small factors can cancel without intermediate overflow or underflow. Raise ValueError if the nonzero final reciprocal is outside the representable complex range. Each pole enters with its complex-conjugate partner so the factor is real-analytic. This is a reciprocal product, so the form factor diverges at every mapped pole rather than vanishing there, in contrast with a Blaschke product, which vanishes at its mapped points. A pole on a sheet not contiguous to the physical one is an independent datum and cannot be generated from a mass and a width alone.

Returns
-------
complex, the reciprocal pole product as a native Python complex number
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from collections.abc import Sequence
import numpy as np


def pole_factor(variable_value, mapped_poles):
    """Return the reciprocal product over mapped poles and their conjugates.

    Parameters
    ----------
    variable_value : complex
        Finite scalar value of the bounded expansion variable.
    mapped_poles : sequence of complex
        Non-empty sequence of finite mapped pole positions.

    Returns
    -------
    complex
        The reciprocal pole product as a native Python complex number.

    Raises
    ------
    ValueError
        If the variable value is not a finite scalar, if the mapped poles are
        not a non-empty sequence, if any pole is not a finite scalar or if the
        variable value coincides with a pole or its conjugate, or the nonzero
        final reciprocal product overflows or underflows the finite complex
        output range. Representable results require scaled intermediate products.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from collections.abc import Sequence
import numpy as np


def _oracle_pole_factor(variable_value, mapped_poles):
    from collections.abc import Sequence

    def _finite_complex_scalar(value, name):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or isinstance(value, (str, bytes)):
            raise ValueError(name + " must be a finite scalar")
        try:
            numeric = complex(float(np.real(value)), float(np.imag(value)))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be a finite scalar") from exc
        if not np.isfinite(numeric.real) or not np.isfinite(numeric.imag):
            raise ValueError(name + " must be a finite scalar")
        return numeric

    variable = _finite_complex_scalar(variable_value, "variable_value")
    if isinstance(mapped_poles, (str, bytes)) or not isinstance(mapped_poles, Sequence) or len(mapped_poles) == 0:
        raise ValueError("mapped_poles must be a non-empty sequence")

    poles = []
    for index, pole in enumerate(mapped_poles):
        numeric = _finite_complex_scalar(pole, "mapped_poles[" + str(index) + "]")
        if variable == numeric or variable == numeric.conjugate():
            raise ValueError("variable_value must not coincide with a pole or its conjugate")
        poles.append(numeric)

    import math

    # Accumulate a binary-scaled denominator, not intermediate reciprocals.
    mantissa = 1.0 + 0.0j
    exponent = 0
    for pole in poles:
        for partner in (pole, pole.conjugate()):
            difference = variable - partner
            extra = 0
            if not np.isfinite(difference.real) or not np.isfinite(difference.imag):
                difference = variable * 0.5 - partner * 0.5
                extra = 1
            scale = max(abs(difference.real), abs(difference.imag))
            if scale == 0.0:
                raise ValueError("the pole difference is numerically singular")
            _, power = math.frexp(scale)
            normalized = complex(math.ldexp(difference.real, -power), math.ldexp(difference.imag, -power))
            mantissa *= normalized
            exponent += power + extra
            _, power = math.frexp(max(abs(mantissa.real), abs(mantissa.imag)))
            mantissa = complex(math.ldexp(mantissa.real, -power), math.ldexp(mantissa.imag, -power))
            exponent += power
    inverse = 1.0 / mantissa
    try:
        result = complex(math.ldexp(inverse.real, -exponent), math.ldexp(inverse.imag, -exponent))
    except OverflowError as exc:
        raise ValueError("the reciprocal product overflows the supported numeric range") from exc
    if not np.isfinite(result.real) or not np.isfinite(result.imag) or result == 0.0:
        raise ValueError("the nonzero reciprocal product is outside the supported numeric range")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    status_setup = (
        "import numpy as np\n"
        "def status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "    return 0\n"
    )

    def _status_case(args):
        # call runs the model's function and gold_call runs the oracle through
        # the same try/except, so the exception contract is compared against the
        # oracle rather than asserted against a constant.
        return {
            "setup": status_setup,
            "call": "status(lambda: pole_factor" + args + ")",
            "gold_call": "status(lambda: _oracle_pole_factor" + args + ")",
        }

    return [
        {
            "setup": "import numpy as np\n",
            "call": "[round(v.real, 12), round(v.imag, 12)] if (v := pole_factor(0.18 + 0.07j, [-0.349147 - 0.529787j])) is not None else None",
            "gold_call": "[round(v.real, 12), round(v.imag, 12)] if (v := _oracle_pole_factor(0.18 + 0.07j, [-0.349147 - 0.529787j])) is not None else None",
        },
        {
            "setup": "import numpy as np\npoles = [-0.349147 - 0.529787j, -0.011982 - 0.923045j, 0.290692 - 1.144733j]",
            "call": "[round(v.real, 12), round(v.imag, 12)] if (v := pole_factor(-0.22 + 0.31j, poles)) is not None else None",
            "gold_call": "[round(v.real, 12), round(v.imag, 12)] if (v := _oracle_pole_factor(-0.22 + 0.31j, poles)) is not None else None",
        },
        {
            "setup": "import numpy as np\npoles = [-0.349147 - 0.529787j, -0.011982 - 0.923045j, 0.290692 - 1.144733j]",
            "call": "[[round(v.real, 12), round(v.imag, 12)] for v in [pole_factor(0.41 - 0.16j, poles), pole_factor(0.41 - 0.16j, list(reversed(poles)))]]",
            "gold_call": "[[round(v.real, 12), round(v.imag, 12)] for v in [_oracle_pole_factor(0.41 - 0.16j, poles), _oracle_pole_factor(0.41 - 0.16j, list(reversed(poles)))]]",
        },
        # valid: widely separated pole magnitudes cancel in either order
        {
            "setup": "import numpy as np\n",
            "call": "[[round(v.real, 12), round(v.imag, 12)] for v in (pole_factor(0.0, [1e-160, 1e160]), pole_factor(0.0, [1e160, 1e-160]))]",
            "gold_call": "[[round(v.real, 12), round(v.imag, 12)] for v in (_oracle_pole_factor(0.0, [1e-160, 1e160]), _oracle_pole_factor(0.0, [1e160, 1e-160]))]",
        },
        # invalid: reciprocal product overflows
        _status_case("(0.0, [1e-200])"),
        # invalid: reciprocal product underflows
        _status_case("(0.0, [1e200])"),
        # invalid: reciprocal product overflows at the boundary
        _status_case("(0.0, [1e-160])"),
        # invalid: reciprocal product underflows at the boundary
        _status_case("(0.0, [1e160, 1e160])"),
        # invalid: empty pole sequence
        _status_case("(0.1 + 0.2j, [])"),
        # invalid: evaluation point coincides with a pole
        _status_case("(-0.349147 - 0.529787j, [-0.349147 - 0.529787j])"),
        # invalid: evaluation point coincides with a conjugate pole
        _status_case("(-0.349147 + 0.529787j, [-0.349147 - 0.529787j])"),
        # invalid: non-finite evaluation point
        _status_case("(float('inf'), [-0.349147 - 0.529787j])"),
        # invalid: pole collection is not a sequence
        _status_case("(0.1 + 0.2j, -0.349147 - 0.529787j)"),
    ]
