"""
Step 7: the male fraction at a temperature.

The male fraction falls with incubation temperature through an asymmetric sigmoid. Its steepness is one value below the pivotal temperature and another value above it, so the curve turns over sharply at the pivot while staying shallow in the tails. At the pivot the fraction is one half, and the two stated spans carry it to 0.95 below and 0.05 above.

The sex of the hatchlings of a nest is decided by the equivalent temperature of the developmental window, so this curve is the last link of the pipeline. It is decreasing, which is why the warmer of two window placements produces the fewer males.

Returns
-------
male_fraction : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def male_fraction(
    temperature: float,
    P: float = 29.349664,
    SL: float = 2.238029,
    SH: float = 2.080107,
    l: float = 0.05,
) -> float:
    """Male fraction at one temperature, from the asymmetric sigmoid.

    Parameters
    ----------
    temperature : float
        Equivalent temperature of the developmental window, in degrees Celsius.
        It must be finite.
    P : float, optional
        Pivotal temperature, in degrees Celsius (29.349664). The fraction is one
        half at the pivot.
    SL : float, optional
        Temperature span below the pivot over which the fraction rises from one
        half to 0.95 (2.238029). It is the steepness of the curve below the pivot.
    SH : float, optional
        Temperature span above the pivot over which the fraction falls from one
        half to 0.05 (2.080107).
    l : float, optional
        Tail level of the sigmoid (0.05), strictly between zero and one half. It
        sets the logistic scale.

    Returns
    -------
    float
        The male fraction at `temperature`, between zero and one. It is 0.95 at
        ``P - SL``, 0.5 at `P` and 0.05 at ``P + SH``, with the steepness ``SL``
        below the pivot and ``SH`` at and above it.

    Raises
    ------
    ValueError
        If `temperature` is not finite, if `P` is not finite, if `SL` or `SH` is
        not positive, or if `l` is not strictly between zero and one half.
    """
    return fraction

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_male_fraction(
    temperature: float,
    P: float = 29.349664,
    SL: float = 2.238029,
    SH: float = 2.080107,
    l: float = 0.05,
) -> float:
    """Reference implementation for male_fraction."""
    if not np.isfinite(float(temperature)):
        raise ValueError("temperature must be finite")
    if not np.isfinite(float(P)):
        raise ValueError("P must be finite")
    if not (float(SL) > 0.0 and float(SH) > 0.0):
        raise ValueError("SL and SH must be positive")
    if not (0.0 < float(l) < 0.5):
        raise ValueError("l must lie strictly between zero and one half")
    span = float(SL) if float(temperature) < float(P) else float(SH)
    return float(1.0 / (1.0 + np.exp((-np.log((1.0 - float(l)) / float(l)) / span)
                                     * (float(P) - float(temperature)))))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return list of test case specifications."""
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        male_fraction(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_male_fraction(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # Normal: a temperature below the pivot, on the shallow cold arm.
        {"setup": "import numpy as np\n"
                  "TEMP = 29.0\n",
         "call": "male_fraction(TEMP)",
         "gold_call": "_oracle_male_fraction(TEMP)"},
        # Normal: a temperature above the pivot, where the fraction is small.
        {"setup": "import numpy as np\n"
                  "TEMP = 30.6\n",
         "call": "male_fraction(TEMP)",
         "gold_call": "_oracle_male_fraction(TEMP)"},
        # Boundary: the pivotal temperature, where the fraction is one half.
        {"setup": "import numpy as np\n"
                  "TEMP = 29.349664\n",
         "call": "male_fraction(TEMP)",
         "gold_call": "_oracle_male_fraction(TEMP)"},
        # Boundary: one span below the pivot, where the fraction is 0.95.
        {"setup": "import numpy as np\n"
                  "TEMP = 29.349664 - 2.238029\n",
         "call": "male_fraction(TEMP)",
         "gold_call": "_oracle_male_fraction(TEMP)"},
        # Edge: one span above the pivot, where the fraction is 0.05.
        {"setup": "import numpy as np\n"
                  "TEMP = 29.349664 + 2.080107\n",
         "call": "male_fraction(TEMP)",
         "gold_call": "_oracle_male_fraction(TEMP)"},
        # Edge: the tails, where the sigmoid saturates at one and at zero.
        {"setup": "import numpy as np\n"
                  "TEMP = 24.0\n",
         "call": "male_fraction(TEMP)",
         "gold_call": "_oracle_male_fraction(TEMP)"},
        # Edge: another pivotal temperature and unequal spans, so the two arms of the
        # curve are exercised separately.
        {"setup": "import numpy as np\n"
                  "TEMP = 31.5\n",
         "call": "male_fraction(TEMP, P=29.0, SL=1.5, SH=3.0)",
         "gold_call": "_oracle_male_fraction(TEMP, P=29.0, SL=1.5, SH=3.0)"},
        # Invalid: a non-finite temperature.
        {"setup": "import numpy as np\n"
                  "TEMP = float('inf')\n" + invalid,
         "call": "run_model(temperature=TEMP)",
         "gold_call": "run_gold(temperature=TEMP)"},
        # Invalid: a tail level outside its range.
        {"setup": "import numpy as np\n"
                  "TEMP = 30.0\n" + invalid,
         "call": "run_model(temperature=TEMP, l=0.5)",
         "gold_call": "run_gold(temperature=TEMP, l=0.5)"},
    ]
