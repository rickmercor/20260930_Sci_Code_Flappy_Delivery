"""
Evaluate one selected branch of the intermediate conformal map that resolves two ordered right-hand thresholds. All square roots use their principal branches. An exactly real energy, including either sign of a zero imaginary part, denotes the limit from the upper half-plane; a negative real threshold-minus-energy radicand therefore has a square root with negative imaginary part. The input energy may be real or complex, the thresholds are finite real scalars with the lower one strictly below the upper one and the two-character sheet label selects the returned branch as a Python complex number. Invalid input raises ValueError when the energy is not a finite scalar, either threshold is not a finite real scalar, the thresholds are not strictly ordered, the sheet label is not one of the four supported strings, the selected outer branch diverges at the lower threshold or a nonzero branch result is outside the finite complex output range.

Write L = lower_threshold, U = upper_threshold, a = sqrt(L - s), b = sqrt(U - s) and c = sqrt(U - L), with the principal square roots and the real-axis boundary convention above. The defining branches are phi_11 = (b - c) / a = a / (b + c), phi_21 = -phi_11, phi_22 = 1 / phi_11 and phi_12 = -1 / phi_11. The rationalized expression gives the limiting inner value zero at s = L; both outer branches diverge there and are rejected. Sheet 11 is physical and 21 is reached across the elastic cut. The physical sheet connects to 22 above the upper threshold, not to 12. Evaluate the mathematical expressions using scaled arithmetic when intermediate differences overflow; reject a nonzero result that overflows or underflows the finite complex output range with ValueError.

Returns
-------
complex, the selected branch of the two-threshold conformal map
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def threshold_branch_map(s, lower_threshold, upper_threshold, sheet):
    """Return the selected branch of the two-threshold conformal map.

    Parameters
    ----------
    s : float or complex
        Finite squared energy. Exactly real inputs use the upper-half-plane
        boundary value, independent of the sign of a zero imaginary part.
    lower_threshold : float
        Finite real position of the first right-hand threshold.
    upper_threshold : float
        Finite real position of the second right-hand threshold.
    sheet : str
        Branch label, chosen from "11", "21", "22" and "12".

    Returns
    -------
    complex
        Value of the selected conformal-map branch.

    Raises
    ------
    ValueError
        If the energy is not a finite real or complex scalar, if either
        threshold is not a finite real scalar, if the lower threshold is not
        strictly below the upper threshold, if the sheet label is invalid or if
        the energy equals the lower threshold on sheet "22" or "12", or a
        nonzero result overflows or underflows the finite complex output range.
        Avoid intermediate overflow when the final result is representable.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_threshold_branch_map(s, lower_threshold, upper_threshold, sheet):
    def _complex_scalar(value, name):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or isinstance(value, (str, bytes)):
            raise ValueError(name + " must be a finite real or complex scalar")
        try:
            numeric = np.asarray(value, dtype=np.complex128).item()
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be a finite real or complex scalar") from exc
        if not np.isfinite(numeric.real) or not np.isfinite(numeric.imag):
            raise ValueError(name + " must be a finite real or complex scalar")
        return numeric

    def _real_scalar(value, name):
        numeric = _complex_scalar(value, name)
        if numeric.imag != 0.0:
            raise ValueError(name + " must be a finite real scalar")
        return float(numeric.real)

    energy = _complex_scalar(s, "s")
    if energy.imag == 0.0:
        energy = complex(energy.real, 0.0)
    lower = _real_scalar(lower_threshold, "lower_threshold")
    upper = _real_scalar(upper_threshold, "upper_threshold")
    if not lower < upper:
        raise ValueError("lower_threshold must be strictly below upper_threshold")
    if not isinstance(sheet, str) or sheet not in {"11", "21", "22", "12"}:
        raise ValueError("sheet must be one of '11', '21', '22' or '12'")

    if energy == complex(lower) and sheet in {"22", "12"}:
        raise ValueError("the outer branch diverges at the lower threshold")

    def _root_difference(threshold, point):
        # Halving before an overflowing subtraction preserves a representable root.
        real_part = threshold - point.real
        if np.isfinite(real_part):
            return complex(np.sqrt(complex(real_part, -point.imag)))
        half_difference = complex(threshold / 2.0 - point.real / 2.0, -point.imag / 2.0)
        return complex(np.sqrt(half_difference)) * np.sqrt(2.0)

    c = _root_difference(upper, complex(lower))
    root_upper = _root_difference(upper, energy)
    root_lower = _root_difference(lower, energy)

    if sheet in {"11", "21"}:
        if energy == complex(lower):
            value = complex(0.0)
        else:
            value = root_lower / (root_upper + c)
        if sheet == "21":
            value = -value
    else:
        value = (root_upper + c) / root_lower
        if sheet == "12":
            value = -value

    if not np.isfinite(value.real) or not np.isfinite(value.imag):
        raise ValueError("the branch result is not representable as a finite complex scalar")
    if value == 0.0 and energy != complex(lower):
        raise ValueError("the nonzero branch result underflows the supported numeric range")
    return complex(value)

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
            "call": "status(lambda: threshold_branch_map" + args + ")",
            "gold_call": "status(lambda: _oracle_threshold_branch_map" + args + ")",
        }

    return [
        {
            "setup": "import numpy as np\n",
            "call": "(lambda v: [round(v.real, 12), round(v.imag, 12), int(abs(v) > 1.0)])(threshold_branch_map(1.15 + 0.12j, 4 * 0.13957**2, 4 * 0.493677**2, '22'))",
            "gold_call": "(lambda v: [round(v.real, 12), round(v.imag, 12), int(abs(v) > 1.0)])(_oracle_threshold_branch_map(1.15 + 0.12j, 4 * 0.13957**2, 4 * 0.493677**2, '22'))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "(lambda v: [round(v.real, 12), round(v.imag, 12), int(abs(v) < 1.0)])(threshold_branch_map(0.55 + 0.18j, 4 * 0.13957**2, 4 * 0.493677**2, '21'))",
            "gold_call": "(lambda v: [round(v.real, 12), round(v.imag, 12), int(abs(v) < 1.0)])(_oracle_threshold_branch_map(0.55 + 0.18j, 4 * 0.13957**2, 4 * 0.493677**2, '21'))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "(lambda v: [round(v.real, 12), round(v.imag, 12)])(threshold_branch_map((0.452 - 0.271j)**2, 4 * 0.13957**2, 4 * 0.493677**2, '21'))",
            "gold_call": "(lambda v: [round(v.real, 12), round(v.imag, 12)])(_oracle_threshold_branch_map((0.452 - 0.271j)**2, 4 * 0.13957**2, 4 * 0.493677**2, '21'))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "(lambda a, b: [round((a + b).real, 12), round((a + b).imag, 12), int(abs(a + b) <= 1e-12)])(threshold_branch_map(0.55 + 0.18j, 4 * 0.13957**2, 4 * 0.493677**2, '21'), threshold_branch_map(0.55 + 0.18j, 4 * 0.13957**2, 4 * 0.493677**2, '11'))",
            "gold_call": "(lambda a, b: [round((a + b).real, 12), round((a + b).imag, 12), int(abs(a + b) <= 1e-12)])(_oracle_threshold_branch_map(0.55 + 0.18j, 4 * 0.13957**2, 4 * 0.493677**2, '21'), _oracle_threshold_branch_map(0.55 + 0.18j, 4 * 0.13957**2, 4 * 0.493677**2, '11'))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "(lambda v: [round(v.real, 12), round(v.imag, 12), round(abs(v), 12)])(threshold_branch_map(4 * 0.493677**2, 4 * 0.13957**2, 4 * 0.493677**2, '11'))",
            "gold_call": "(lambda v: [round(v.real, 12), round(v.imag, 12), round(abs(v), 12)])(_oracle_threshold_branch_map(4 * 0.493677**2, 4 * 0.13957**2, 4 * 0.493677**2, '11'))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "(lambda v: [round(v.real, 12), round(v.imag, 12)])(threshold_branch_map(4 * 0.13957**2, 4 * 0.13957**2, 4 * 0.493677**2, '11'))",
            "gold_call": "(lambda v: [round(v.real, 12), round(v.imag, 12)])(_oracle_threshold_branch_map(4 * 0.13957**2, 4 * 0.13957**2, 4 * 0.493677**2, '11'))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "(lambda v: [round(v.real, 12), round(v.imag, 12)])(threshold_branch_map(0.0, 4 * 0.13957**2, 4 * 0.493677**2, '21'))",
            "gold_call": "(lambda v: [round(v.real, 12), round(v.imag, 12)])(_oracle_threshold_branch_map(0.0, 4 * 0.13957**2, 4 * 0.493677**2, '21'))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "(lambda v: [round(v.real, 12), round(v.imag, 12)])(threshold_branch_map(-0.60, 4 * 0.13957**2, 4 * 0.493677**2, '11'))",
            "gold_call": "(lambda v: [round(v.real, 12), round(v.imag, 12)])(_oracle_threshold_branch_map(-0.60, 4 * 0.13957**2, 4 * 0.493677**2, '11'))",
        },
        # valid: extreme thresholds: the three outer/negated branches against the stable root form
        {
            "setup": "import numpy as np\n",
            "call": "[(lambda v: [round(v.real, 12), round(v.imag, 12)])(threshold_branch_map(-1e308, 1e-308, 1e308, label)) for label in ('21', '22', '12')]",
            "gold_call": "[(lambda v: [round(v.real, 12), round(v.imag, 12)])(_oracle_threshold_branch_map(-1e308, 1e-308, 1e308, label)) for label in ('21', '22', '12')]",
        },
        # valid: extreme thresholds: the physical branch
        {
            "setup": "import numpy as np\n",
            "call": "(lambda v: [round(v.real, 12), round(v.imag, 12)])(threshold_branch_map(-1e308, 1e-308, 1e308, '11'))",
            "gold_call": "(lambda v: [round(v.real, 12), round(v.imag, 12)])(_oracle_threshold_branch_map(-1e308, 1e-308, 1e308, '11'))",
        },
        # valid: extreme thresholds: the outer branch
        {
            "setup": "import numpy as np\n",
            "call": "(lambda v: [round(v.real, 12), round(v.imag, 12)])(threshold_branch_map(-1e308, 1e-308, 1e308, '22'))",
            "gold_call": "(lambda v: [round(v.real, 12), round(v.imag, 12)])(_oracle_threshold_branch_map(-1e308, 1e-308, 1e308, '22'))",
        },
        # valid: real energy at the inelastic threshold: every spelling of a zero imaginary part agrees
        {
            "setup": "import numpy as np\n",
            "call": "[(lambda v: [round(v.real, 12), round(v.imag, 12)])(threshold_branch_map(s, 0.1, 1.0, '11')) for s in (1.0, complex(1.0, 0.0), complex(1.0, -0.0), np.complex128(1.0))]",
            "gold_call": "[(lambda v: [round(v.real, 12), round(v.imag, 12)])(_oracle_threshold_branch_map(s, 0.1, 1.0, '11')) for s in (1.0, complex(1.0, 0.0), complex(1.0, -0.0), np.complex128(1.0))]",
        },
        # invalid: thresholds not strictly ordered
        _status_case("(0.2, 1.0, 1.0, '11')"),
        # invalid: unknown sheet label
        _status_case("(0.2, 0.1, 1.0, '13')"),
        # invalid: non-finite energy
        _status_case("(float('nan'), 0.1, 1.0, '11')"),
        # invalid: outer branch diverges at the lower threshold
        _status_case("(0.1, 0.1, 1.0, '22')"),
        # invalid: complex lower threshold
        _status_case("(0.2, np.complex64(.1 + .2j), 1.0, '11')"),
        # invalid: sheet label is not a string
        _status_case("(0.2, 0.1, 1.0, 11)"),
    ]
