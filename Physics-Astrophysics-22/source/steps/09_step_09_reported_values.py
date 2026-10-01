"""
Given the six quantities the problem asks to be reported, return each rounded to the precision the problem requests: the energy-norm condition number of the primary background to six decimal places, its Lebesgue-norm condition number to four significant figures, the imaginary part of its frequency to eight significant figures, and for the comparison background the two condition numbers to four significant figures and the imaginary part of the frequency to eight significant figures.

The problem asks for six numbers at three requested precisions: the graded energy-norm condition number to six decimal places, the two imaginary parts of the frequencies to eight significant figures, and the three remaining condition numbers to four significant figures. Rounding to a number of decimal places and rounding to a number of significant figures are different operations, and the second is the one that is easy to get wrong, because the position of the last retained digit depends on the magnitude of the value rather than on the position of the decimal point. For a value of about eight hundredths, four significant figures means five decimal places; for a value of about one, it means three.




The rule is the usual one. To keep n significant figures of a non-zero value, locate the exponent of its leading digit, which is the floor of the base-ten logarithm of its modulus, and round to that many places after the point counting from the leading digit, that is to n minus one minus the exponent decimal places. A value of exactly zero has no leading digit and is returned unchanged. Nothing here changes any computed quantity; this step only presents them, and it is the last stage before the orchestrator assembles the report.

Returns
-------
dict holding the float graded_answer, the energy-norm condition number to six decimal places; the float reported_lebesgue and the float reported_comparison_energy and the float reported_comparison_lebesgue, each to four significant figures; and the float reported_frequency and the float reported_comparison_frequency, each to eight significant figures.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reported_values(
    condition_number_energy: float,
    condition_number_lebesgue: float,
    frequency_imag: float,
    comparison_condition_number_energy: float,
    comparison_condition_number_lebesgue: float,
    comparison_frequency_imag: float,
) -> dict:
    """Round the six reported quantities to the precisions the problem requests.

    Parameters
    ----------
    condition_number_energy : float
        Energy-norm condition number of the primary background.
    condition_number_lebesgue : float
        Lebesgue-norm condition number of the primary background.
    frequency_imag : float
        Imaginary part of the frequency of the primary mode.
    comparison_condition_number_energy : float
        Energy-norm condition number of the comparison background.
    comparison_condition_number_lebesgue : float
        Lebesgue-norm condition number of the comparison background.
    comparison_frequency_imag : float
        Imaginary part of the frequency of the comparison mode.

    Returns
    -------
    dict
        Under the keys graded_answer, reported_lebesgue, reported_frequency,
        reported_comparison_energy, reported_comparison_lebesgue and
        reported_comparison_frequency.

    Raises
    ------
    ValueError
        When any argument is not a finite, strictly positive real number.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math


def _checked(value, name):
    """Reject anything that is not a finite, strictly positive real number."""
    if isinstance(value, bool) or isinstance(value, complex):
        raise ValueError(name + " must be a real number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(name + " must be a real number")
    if not math.isfinite(number):
        raise ValueError(name + " must be finite")
    if number <= 0.0:
        raise ValueError(name + " must be strictly positive")
    return number


def _significant(value, digits):
    """Round to the requested number of significant figures."""
    if value == 0.0:
        return 0.0
    exponent = math.floor(math.log10(abs(value)))
    return float(round(value, digits - 1 - exponent))


def _oracle_reported_values(
    condition_number_energy: float,
    condition_number_lebesgue: float,
    frequency_imag: float,
    comparison_condition_number_energy: float,
    comparison_condition_number_lebesgue: float,
    comparison_frequency_imag: float,
) -> dict:
    """Reference implementation."""
    energy = _checked(condition_number_energy, "condition_number_energy")
    lebesgue = _checked(condition_number_lebesgue, "condition_number_lebesgue")
    frequency = _checked(frequency_imag, "frequency_imag")
    other_energy = _checked(
        comparison_condition_number_energy, "comparison_condition_number_energy"
    )
    other_lebesgue = _checked(
        comparison_condition_number_lebesgue, "comparison_condition_number_lebesgue"
    )
    other_frequency = _checked(comparison_frequency_imag, "comparison_frequency_imag")
    return {
        "graded_answer": float(round(energy, 6)),
        "reported_lebesgue": _significant(lebesgue, 4),
        "reported_frequency": _significant(frequency, 8),
        "reported_comparison_energy": _significant(other_energy, 4),
        "reported_comparison_lebesgue": _significant(other_lebesgue, 4),
        "reported_comparison_frequency": _significant(other_frequency, 8),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    FLAT = """
def flat(x):
    if isinstance(x, dict):
        return flat([x[k] for k in sorted(x)])
    if isinstance(x, (tuple, list)):
        out = []
        for v in x:
            out.extend(flat(v))
        return tuple(out)
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    if isinstance(x, bool):
        return (int(x),)
    return (x,)
"""

    SETUP = """
GRADED = (0.08126175551862909, 0.08666201677206606, 1.1022215452976556,
          0.05385421264486725, 0.05517586837327695, 1.063471268159331)
"""
    return [
        {
            # the graded configuration: the six values the chain actually produces
            "setup": SETUP + FLAT,
            "call": "flat(reported_values(*GRADED))",
            "gold_call": "flat(_oracle_reported_values(*GRADED))",
        },
        {
            # significant figures track the leading digit, not the decimal point: four figures of a
            # value near eight hundredths keeps five decimals, of a value near one keeps three, and of
            # a value near ninety keeps one
            "setup": SETUP + """
def scaling(fn):
    out = fn(0.5, 0.086662017, 1.1022215453, 1.0004999, 93.827, 12.345678901)
    return (out["reported_lebesgue"], out["reported_comparison_energy"],
            out["reported_comparison_lebesgue"], out["reported_frequency"],
            out["reported_comparison_frequency"], out["graded_answer"])
""" + FLAT,
            "call": "flat(scaling(reported_values))",
            "gold_call": "flat(scaling(_oracle_reported_values))",
        },
        {
            # boundary: values sitting on a rounding boundary, and a value small enough that four
            # significant figures reaches the ninth decimal place
            "setup": SETUP + FLAT,
            "call": "flat(reported_values(0.0812615, 0.000123456789, 9.9999999e-7, 0.99995, 1.00005, 2.5))",
            "gold_call": "flat(_oracle_reported_values(0.0812615, 0.000123456789, 9.9999999e-7, 0.99995, 1.00005, 2.5))",
        },
        {
            "setup": SETUP + """
def verdict(fn, index, value):
    args = list(GRADED)
    args[index] = value
    try:
        fn(*args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(reported_values, 0, 0.0), "
                    "verdict(reported_values, 1, -1.0), "
                    "verdict(reported_values, 2, float('nan')), "
                    "verdict(reported_values, 3, float('inf')), "
                    "verdict(reported_values, 4, 1j), "
                    "verdict(reported_values, 5, 'x')))",
            "gold_call": "flat((verdict(_oracle_reported_values, 0, 0.0), "
                         "verdict(_oracle_reported_values, 1, -1.0), "
                         "verdict(_oracle_reported_values, 2, float('nan')), "
                         "verdict(_oracle_reported_values, 3, float('inf')), "
                         "verdict(_oracle_reported_values, 4, 1j), "
                         "verdict(_oracle_reported_values, 5, 'x')))",
        },
    ]
