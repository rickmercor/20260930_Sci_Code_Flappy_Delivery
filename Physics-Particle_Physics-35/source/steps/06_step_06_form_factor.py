"""
Evaluate the bounded form factor at one squared energy on a specified Riemann sheet. The two ordered thresholds, the cut opening and the normalization point define the conformal variable, while each non-empty pole entry supplies a complex energy and its own sheet and each finite real coefficient supplies one ascending polynomial power. Every pole energy is squared before it is mapped, the reciprocal pole product multiplies the polynomial and the result is returned as a native Python complex number. Invalid input raises ValueError for non-finite or non-scalar data, non-real geometry or coefficients, unordered thresholds, unknown sheets, malformed or empty sequences, coincident mapped references, singular conformal maps including the outer reciprocal or an outer branch at the lower threshold, overflowing squared pole energies or evaluations, or coincidence with a mapped pole or its conjugate.

The polynomial evaluation and final multiplication must be finite; unsupported overflow raises ValueError. The normalization point must lie strictly below the lower threshold and its intermediate image must be distinct from both unit endpoints in double precision. Unsupported normalization geometries and constituent map or reciprocal-product range failures propagate as ValueError. The parametrisation carries its analytic structure in the map itself, so there is no outer function and no Omnes factor. The pole factor supplies the resonance poles and the polynomial supplies the smooth remainder. Each pole is quoted with its own sheet because a pole on a non-contiguous sheet is an independent datum rather than a continuation fixed by the evaluation sheet.

Returns
-------
complex, the pole-dressed polynomial form factor as a native Python complex number
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from collections.abc import Sequence
import numpy as np


def form_factor(s, sheet, lower_threshold, upper_threshold, cut_opening, normalization_point, poles, coefficients):
    """Return the pole-dressed polynomial form factor on one sheet.

    Parameters
    ----------
    s : complex
        Finite scalar squared energy at which to evaluate the form factor.
    sheet : str
        Evaluation sheet, one of "11", "21", "22" or "12".
    lower_threshold : float
        Finite real lower channel threshold in squared energy units.
    upper_threshold : float
        Finite real upper channel threshold in squared energy units.
    cut_opening : float
        Finite real squared energy at which the left-hand cut opens.
    normalization_point : float
        Finite real squared energy strictly below the lower threshold, with
        a physical intermediate image distinct from both unit endpoints in
        double precision. Other normalization geometries raise ValueError.
    poles : sequence of pairs
        Non-empty sequence of complex pole energies and their sheet labels.
    coefficients : sequence of float
        Non-empty sequence of finite real polynomial coefficients in ascending order.

    Returns
    -------
    complex
        The pole-dressed polynomial form factor as a native Python complex number.

    Raises
    ------
    ValueError
        If the squared energy is not a finite scalar, if the geometry values or
        coefficients are not finite real scalars, if the thresholds are not
        strictly ordered, if any sheet is unknown, if either sequence is empty
        or malformed, if mapped references coincide, if a conformal denominator
        vanishes (including the outer reciprocal), if an outer branch is
        requested at the lower threshold, if a squared pole energy, conformal
        map or form-factor evaluation overflows, or if the expansion variable
        coincides with a mapped pole or its conjugate. Unsupported normalization
        geometries (point not below the lower threshold or intermediate image at
        a unit endpoint) and numerical range failures in either constituent map
        or the reciprocal product also raise ValueError.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from collections.abc import Sequence
import numpy as np


def _oracle_form_factor(s, sheet, lower_threshold, upper_threshold, cut_opening, normalization_point, poles, coefficients):
    from collections.abc import Sequence

    if isinstance(poles, (str, bytes)) or not isinstance(poles, Sequence) or len(poles) == 0:
        raise ValueError("poles must be a non-empty sequence of energy and sheet pairs")
    mapped_poles = []
    for index, entry in enumerate(poles):
        if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence) or len(entry) != 2:
            raise ValueError("poles[" + str(index) + "] must be an energy and sheet pair")
        pole_energy, pole_sheet = entry
        if isinstance(pole_energy, (bool, np.bool_)) or not np.isscalar(pole_energy) or isinstance(pole_energy, (str, bytes)):
            raise ValueError("poles[" + str(index) + "][0] must be a finite scalar")
        try:
            squared = complex(pole_energy) ** 2
        except OverflowError as exc:
            raise ValueError("the squared pole energy must be finite") from exc
        if not np.isfinite(squared.real) or not np.isfinite(squared.imag):
            raise ValueError("the squared pole energy must be finite")
        mapped_poles.append(
            _oracle_bounded_variable(squared, lower_threshold, upper_threshold, cut_opening, normalization_point, pole_sheet)
        )

    if isinstance(coefficients, (str, bytes)) or not isinstance(coefficients, Sequence) or len(coefficients) == 0:
        raise ValueError("coefficients must be a non-empty sequence of finite reals")
    validated_coefficients = []
    for index, value in enumerate(coefficients):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or isinstance(value, (str, bytes)):
            raise ValueError("coefficients[" + str(index) + "] must be a finite real scalar")
        numeric = complex(float(np.real(value)), float(np.imag(value)))
        if numeric.imag != 0.0 or not np.isfinite(numeric.real):
            raise ValueError("coefficients[" + str(index) + "] must be a finite real scalar")
        validated_coefficients.append(float(numeric.real))

    variable = _oracle_bounded_variable(s, lower_threshold, upper_threshold, cut_opening, normalization_point, sheet)
    factor = _oracle_pole_factor(variable, mapped_poles)
    try:
        polynomial = sum(
            coefficient * variable**index
            for index, coefficient in enumerate(validated_coefficients)
        )
        result = complex(factor * polynomial)
    except OverflowError as exc:
        raise ValueError("the form factor must be finite") from exc
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise ValueError("the form factor must be finite")
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
            "call": "status(lambda: form_factor" + args + ")",
            "gold_call": "status(lambda: _oracle_form_factor" + args + ")",
        }

    return [
        {
            "setup": "import numpy as np\npoles = [(0.452 - 0.271j, '21'), (0.998 - 0.029j, '21'), (0.981 - 0.043j, '22')]\ncoefficients = [0.462, -0.236, 0.559, -0.706, 0.228, -0.510, -0.147, 1.088, -0.526, 1.578, -0.268, 0.533]\nlower = 4 * 0.13957**2\nupper = 4 * 0.493677**2",
            "call": "[round(v.real, 12), round(v.imag, 12)] if (v := form_factor(1.15 + 0.12j, '22', lower, upper, 0.0, -0.60, poles, coefficients)) is not None else None",
            "gold_call": "[round(v.real, 12), round(v.imag, 12)] if (v := _oracle_form_factor(1.15 + 0.12j, '22', lower, upper, 0.0, -0.60, poles, coefficients)) is not None else None",
        },
        {
            "setup": "import numpy as np\npoles = [(0.452 - 0.271j, '21'), (0.998 - 0.029j, '21'), (0.981 - 0.043j, '22')]\ncoefficients = [0.462, -0.236, 0.559, -0.706, 0.228, -0.510, -0.147, 1.088, -0.526, 1.578, -0.268, 0.533]\nlower = 4 * 0.13957**2\nupper = 4 * 0.493677**2",
            "call": "[round(v.real, 12), round(v.imag, 12)] if (v := form_factor(0.55 + 0.18j, '21', lower, upper, 0.0, -0.60, poles, coefficients)) is not None else None",
            "gold_call": "[round(v.real, 12), round(v.imag, 12)] if (v := _oracle_form_factor(0.55 + 0.18j, '21', lower, upper, 0.0, -0.60, poles, coefficients)) is not None else None",
        },
        {
            "setup": "import numpy as np\npoles = [(0.452 - 0.271j, '21')]\ncoefficients = [1.0]\nlower = 4 * 0.13957**2\nupper = 4 * 0.493677**2",
            "call": "[round(v.real, 12), round(v.imag, 12)] if (v := form_factor(-0.60, '11', lower, upper, 0.0, -0.60, poles, coefficients)) is not None else None",
            "gold_call": "[round(v.real, 12), round(v.imag, 12)] if (v := _oracle_form_factor(-0.60, '11', lower, upper, 0.0, -0.60, poles, coefficients)) is not None else None",
        },
        {
            "setup": "import numpy as np\npoles = [(0.452 - 0.271j, '21'), (0.981 - 0.043j, '22')]\ncoefficients = [0.0, 0.0, 1.0]\nlower = 4 * 0.13957**2\nupper = 4 * 0.493677**2",
            "call": "[round(v.real, 12), round(v.imag, 12)] if (v := form_factor(2.0 + 0.3j, '12', lower, upper, 0.0, -0.60, poles, coefficients)) is not None else None",
            "gold_call": "[round(v.real, 12), round(v.imag, 12)] if (v := _oracle_form_factor(2.0 + 0.3j, '12', lower, upper, 0.0, -0.60, poles, coefficients)) is not None else None",
        },
        # invalid: normalization point not below the lower threshold
        _status_case("(0.2, '11', 0.1, 1.0, 0.0, 1e200, [(.5 - .1j, '21')], [1.0])"),
        # invalid: normalization image at a unit endpoint
        _status_case("(0.2, '11', 0.1, 1.0, 0.0, -1e308, [(.5 - .1j, '21')], [1.0])"),
        # invalid: complex cut opening
        _status_case("(0.2, '11', 0.1, 1.0, .1j, -0.60, [(.5 - .1j, '21')], [1.0])"),
        # invalid: complex normalization point
        _status_case("(0.2, '11', 0.1, 1.0, 0.0, np.complex64(.1 + .2j), [(.5 - .1j, '21')], [1.0])"),
        # invalid: vanishing outer conformal denominator
        _status_case("(2.734375, '22', 4.0, 6.25, 0.0, 2.734375, [(.5 - .1j, '21')], [1.0])"),
        # invalid: outer branch at the lower threshold
        _status_case("(0.1, '22', 0.1, 1.0, 0.0, -0.60, [(.5 - .1j, '21')], [1.0])"),
        # invalid: squared pole energy overflows
        _status_case("(0.2, '11', 0.1, 1.0, 0.0, -0.60, [(1e200 - .1j, '21')], [1.0])"),
        # invalid: non-finite energy
        _status_case("(float('nan'), '11', 0.1, 1.0, 0.0, -0.60, [(.5 - .1j, '21')], [1.0])"),
        # invalid: thresholds not strictly ordered
        _status_case("(0.2, '11', 1.0, 1.0, 0.0, -0.60, [(.5 - .1j, '21')], [1.0])"),
        # invalid: unknown pole sheet
        _status_case("(0.2, '11', 0.1, 1.0, 0.0, -0.60, [(.5 - .1j, '13')], [1.0])"),
        # invalid: malformed pole entry
        _status_case("(0.2, '11', 0.1, 1.0, 0.0, -0.60, [(.5 - .1j, '21', 3)], [1.0])"),
        # invalid: non-finite coefficient
        _status_case("(0.2, '11', 0.1, 1.0, 0.0, -0.60, [(.5 - .1j, '21')], [1.0, float('inf')])"),
        # invalid: empty coefficient sequence
        _status_case("(0.2, '11', 0.1, 1.0, 0.0, -0.60, [(.5 - .1j, '21')], [])"),
    ]
