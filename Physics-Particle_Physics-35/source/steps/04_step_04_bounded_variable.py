"""
Compose the four-sheet two-threshold map with the left-cut map to produce one bounded complex expansion variable. The inputs specify the squared energy, two ordered channel thresholds, the left-cut opening, the normalization point and the requested sheet. The cut reference is mapped on sheet 21, the normalization reference is mapped on sheet 11 and the squared energy is mapped on the requested sheet before the piecewise composition is applied. The function returns a native Python complex number and raises ValueError for non-finite or non-scalar data, non-real geometry values, unordered thresholds, an unknown sheet, coincident mapped references, a vanishing conformal denominator including the outer reciprocal, an outer branch at the lower threshold or an overflowing map.

The composition is piecewise in the modulus of the intermediate variable. Outside the unit disc its argument is inverted and the result is inverted again, a branch-safety rule required because the second map inherits a non-injectivity from the standard inverse map and applying it directly outside the disc admits spurious cuts. The construction is asymmetric: the cut reference is taken on the sheet across the elastic cut while the normalization reference is taken on the physical sheet. The supported normalization geometry has normalization_point strictly below lower_threshold, with its intermediate image numerically distinct from both unit endpoints. Other normalization geometries raise ValueError. Numerical range refusals from either constituent map propagate as ValueError.

Returns
-------
complex, the bounded expansion variable as a native Python complex number
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def bounded_variable(s, lower_threshold, upper_threshold, cut_opening, normalization_point, sheet):
    """Return the bounded expansion variable on one of four sheets.

    Parameters
    ----------
    s : complex
        Finite scalar squared energy at which to evaluate the variable.
    lower_threshold : float
        Finite real lower channel threshold in squared energy units.
    upper_threshold : float
        Finite real upper channel threshold in squared energy units.
    cut_opening : float
        Finite real squared energy at which the left-hand cut opens.
    normalization_point : float
        Finite real squared energy at which the bounded variable is normalized.
    sheet : str
        Four-sheet label, one of "11", "21", "22" or "12".

    Returns
    -------
    complex
        The bounded expansion variable as a native Python complex number.

    Raises
    ------
    ValueError
        If the squared energy is not a finite scalar, if any geometry value is
        not a finite real scalar, if the thresholds are not strictly ordered,
        if the sheet is unknown, if the mapped references coincide, if a
        conformal denominator vanishes (including the outer reciprocal), if an
        outer branch is requested at the lower threshold, the normalization
        point is not strictly below the lower threshold, its intermediate image
        equals either unit endpoint in double precision or a constituent map
        rejects an unsupported numerical range.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_bounded_variable(s, lower_threshold, upper_threshold, cut_opening, normalization_point, sheet):
    for value in (cut_opening, normalization_point):
        if isinstance(value, (bool, np.bool_, str, bytes)) or not np.isscalar(value):
            raise ValueError("reference points must be finite real scalars")
        numeric = complex(value)
        if numeric.imag != 0.0 or not np.isfinite(numeric.real):
            raise ValueError("reference points must be finite real scalars")
    x_l = _oracle_threshold_branch_map(cut_opening, lower_threshold, upper_threshold, "21")
    x_0 = _oracle_threshold_branch_map(normalization_point, lower_threshold, upper_threshold, "11")
    if not float(np.real(normalization_point)) < float(np.real(lower_threshold)):
        raise ValueError("normalization_point must be below the lower threshold")
    if x_0 in (1.0, -1.0):
        raise ValueError("the normalization image is numerically unresolved from a unit endpoint")
    if x_l == x_0:
        raise ValueError("mapped cut and normalization references must differ")
    intermediate = _oracle_threshold_branch_map(s, lower_threshold, upper_threshold, sheet)
    if abs(intermediate) <= 1.0:
        return complex(_oracle_leftcut_map(intermediate, x_l, x_0))
    denominator = _oracle_leftcut_map(1.0 / intermediate, x_l, x_0)
    if denominator == 0.0:
        raise ValueError("the outer conformal denominator must be nonzero")
    result = complex(1.0 / denominator)
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise ValueError("the composed map must be finite")
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
            "call": "status(lambda: bounded_variable" + args + ")",
            "gold_call": "status(lambda: _oracle_bounded_variable" + args + ")",
        }

    return [
        {
            "setup": "import numpy as np\n",
            "call": "[round(v.real, 12), round(v.imag, 12)] if (v := bounded_variable(1.15 + 0.12j, 4 * 0.13957**2, 4 * 0.493677**2, 0.0, -0.60, '22')) is not None else None",
            "gold_call": "[round(v.real, 12), round(v.imag, 12)] if (v := _oracle_bounded_variable(1.15 + 0.12j, 4 * 0.13957**2, 4 * 0.493677**2, 0.0, -0.60, '22')) is not None else None",
        },
        {
            "setup": "import numpy as np\n",
            "call": "[round(v.real, 12), round(v.imag, 12)] if (v := bounded_variable(0.55 + 0.18j, 4 * 0.13957**2, 4 * 0.493677**2, 0.0, -0.60, '21')) is not None else None",
            "gold_call": "[round(v.real, 12), round(v.imag, 12)] if (v := _oracle_bounded_variable(0.55 + 0.18j, 4 * 0.13957**2, 4 * 0.493677**2, 0.0, -0.60, '21')) is not None else None",
        },
        {
            "setup": "import numpy as np\n",
            "call": "[round(v.real, 12), round(v.imag, 12)] if (v := bounded_variable(0.20, 4 * 0.13957**2, 4 * 0.493677**2, 0.0, -0.60, '11')) is not None else None",
            "gold_call": "[round(v.real, 12), round(v.imag, 12)] if (v := _oracle_bounded_variable(0.20, 4 * 0.13957**2, 4 * 0.493677**2, 0.0, -0.60, '11')) is not None else None",
        },
        {
            "setup": "import numpy as np\n",
            "call": "[round(v.real, 12), round(v.imag, 12)] if (v := bounded_variable(0.20, 4 * 0.13957**2, 4 * 0.493677**2, 0.0, -0.60, '22')) is not None else None",
            "gold_call": "[round(v.real, 12), round(v.imag, 12)] if (v := _oracle_bounded_variable(0.20, 4 * 0.13957**2, 4 * 0.493677**2, 0.0, -0.60, '22')) is not None else None",
        },
        {
            "setup": "import numpy as np\n",
            "call": "[round(v.real, 12), round(v.imag, 12)] if (v := bounded_variable(2.0, 4 * 0.13957**2, 4 * 0.493677**2, 0.0, -0.60, '11')) is not None else None",
            "gold_call": "[round(v.real, 12), round(v.imag, 12)] if (v := _oracle_bounded_variable(2.0, 4 * 0.13957**2, 4 * 0.493677**2, 0.0, -0.60, '11')) is not None else None",
        },
        # valid: spacelike point on 22: both outer-disc inversions are exercised
        {
            "setup": "import numpy as np\n",
            "call": "[round(v.real, 12), round(v.imag, 12)] if (v := bounded_variable(-2.0, 4 * 0.13957**2, 4 * 0.493677**2, 0.0, -0.60, '22')) is not None else None",
            "gold_call": "[round(v.real, 12), round(v.imag, 12)] if (v := _oracle_bounded_variable(-2.0, 4 * 0.13957**2, 4 * 0.493677**2, 0.0, -0.60, '22')) is not None else None",
        },
        # invalid: normalization point not below the lower threshold
        _status_case("(0.2, 0.1, 1.0, 0.0, 1e200, '11')"),
        # invalid: normalization image at a unit endpoint
        _status_case("(0.2, 0.1, 1.0, 0.0, -1e308, '11')"),
        # invalid: complex cut opening
        _status_case("(0.2, 0.1, 1.0, .1j, -0.60, '11')"),
        # invalid: complex normalization point
        _status_case("(0.2, 0.1, 1.0, 0.0, np.complex64(.1 + .2j), '11')"),
        # invalid: vanishing outer conformal denominator
        _status_case("(2.734375, 4.0, 6.25, 0.0, 2.734375, '22')"),
        # invalid: outer branch at the lower threshold
        _status_case("(0.1, 0.1, 1.0, 0.0, -0.60, '22')"),
        # invalid: non-finite energy
        _status_case("(float('nan'), 4 * 0.13957**2, 4 * 0.493677**2, 0.0, -0.60, '11')"),
        # invalid: unknown sheet label
        _status_case("(0.2, 4 * 0.13957**2, 4 * 0.493677**2, 0.0, -0.60, '13')"),
        # invalid: mapped references coincide
        _status_case("(0.2, 4 * 0.13957**2, 4 * 0.493677**2, 4 * 0.13957**2, 4 * 0.13957**2, '11')"),
    ]
