"""
Step 6: the equivalent temperature of the developmental window.

The window carries one temperature per piece, and the pieces have unequal weights. Each piece spans a stretch of standardised size and takes its weight from step 5. The equivalent temperature is the weighted mean of the piece temperatures.

Both window edges fall inside logging intervals, so the first and last pieces are partial. The piece structure is rebuilt at every reading inside the window, and the standardised size runs from zero at the window start to one at the window end.

Returns
-------
temperature : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def constant_temperature_equivalent(
    times: "numpy.typing.ArrayLike",
    temperature: "numpy.typing.ArrayLike",
    hatchling_length: float,
    window_start: float,
    window_end: float,
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
    DHA_s: float = -719.576256,
    DHH_s: float = 685.203574,
    T12H_s: float = 552.204465,
    shape1: float = 1.725323,
    shape2: float = 0.802045,
    start_size: float = 0.347089,
    asymptotic_ratio: float = 1.208968,
    Rho25_s: float = 100.0,
) -> float:
    """Weighted mean temperature over a developmental window.

    Parameters
    ----------
    times : array-like
        Reading times in minutes from laying, strictly increasing and finite.
    temperature : array-like
        Temperature in degrees Celsius at each reading, finite.
    hatchling_length : float
        Hatchling straight carapace length in mm, positive.
    window_start, window_end : float
        Window limits in minutes from laying, the crossing times of step 4, with
        ``window_start < window_end`` inside the record.
    DHA, DHH, T12H, Rho25 : float, optional
        Parameters of the development rate curve of step 1, which set the modelled
        sizes at the window edges.
    DHA_s, DHH_s, T12H_s : float, optional
        Parameters of the sexualization reaction norm.
    shape1, shape2 : float, optional
        Shapes of the fitted beta density over the standardised size.
    start_size : float, optional
        Modelled size at laying in mm (0.347089).
    asymptotic_ratio : float, optional
        Ratio of the asymptote to the hatchling length (1.208968).
    Rho25_s : float, optional
        Rate of the sexualization norm at the reference temperature (100.0).

    Returns
    -------
    float
        The weighted mean temperature in degrees Celsius. The window is cut at its
        two edges and at every reading strictly inside it. Each piece carries the
        held temperature of the reading at or before its left edge and the span of
        standardised modelled size between its edges. The mean weights each piece
        temperature by its step 5 weight.

    Raises
    ------
    ValueError
        If the growth inputs fail the validation of step 2, if the window limits
        are not finite and strictly ordered, or if the window is not contained in
        the record.
    """
    return temperature_equivalent

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_constant_temperature_equivalent(
    times: "numpy.typing.ArrayLike",
    temperature: "numpy.typing.ArrayLike",
    hatchling_length: float,
    window_start: float,
    window_end: float,
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
    DHA_s: float = -719.576256,
    DHH_s: float = 685.203574,
    T12H_s: float = 552.204465,
    shape1: float = 1.725323,
    shape2: float = 0.802045,
    start_size: float = 0.347089,
    asymptotic_ratio: float = 1.208968,
    Rho25_s: float = 100.0,
) -> float:
    """Reference implementation for constant_temperature_equivalent."""
    t = np.asarray(times, dtype=float)
    T = np.asarray(temperature, dtype=float)
    if not (np.isfinite(float(window_start)) and np.isfinite(float(window_end))):
        raise ValueError("the window limits must be finite")
    if not (float(window_start) < float(window_end)):
        raise ValueError("window_start must lie below window_end")
    if not (float(t[0]) <= float(window_start) and float(window_end) <= float(t[-1])):
        raise ValueError("the window must lie inside the record")
    edges = np.concatenate(([float(window_start)], t[(t > float(window_start)) & (t < float(window_end))],
                            [float(window_end)]))
    sizes = _oracle_embryo_growth(t, T, hatchling_length, DHA, DHH, T12H, Rho25,
                                  start_size, asymptotic_ratio, query_times=edges)
    span = (sizes - sizes[0]) / (sizes[-1] - sizes[0])
    left = np.clip(np.searchsorted(t, edges[:-1], side="right") - 1, 0, t.size - 1)
    piece_temperature = T[left]
    weight = _oracle_sexualization_weight(piece_temperature, span, DHA_s, DHH_s, T12H_s,
                                          shape1, shape2, Rho25_s)
    return float((piece_temperature * weight).sum() / weight.sum())

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return list of test case specifications."""
    record = (
        "import numpy as np\n"
        "def make_record(days=48.0, step=720.0, base=30.0, amp=1.0, trend=2.0):\n"
        "    n = int(days * 1440.0 / step) + 1\n"
        "    times = np.arange(n) * step\n"
        "    temps = base + amp * np.sin(np.arange(n) * 0.8) + trend * (times / times[-1])\n"
        "    return times, temps\n"
    )
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        constant_temperature_equivalent(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_constant_temperature_equivalent(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # Normal: a window whose two edges fall inside logging intervals and which
        # contains many readings.
        {"setup": record + "TIMES, TEMPS = make_record()\n",
         "call": "constant_temperature_equivalent(TIMES, TEMPS, 50.0, 24000.0, 52000.0)",
         "gold_call": "_oracle_constant_temperature_equivalent(TIMES, TEMPS, 50.0, 24000.0, 52000.0)"},
        # Boundary: a window entirely inside the first logging interval, so the window
        # is a single piece and the temperature is the held value of that interval.
        {"setup": record + "TIMES, TEMPS = make_record()\n",
         "call": "constant_temperature_equivalent(TIMES, TEMPS, 50.0, 100.0, 600.0)",
         "gold_call": "_oracle_constant_temperature_equivalent(TIMES, TEMPS, 50.0, 100.0, 600.0)"},
        # Edge: both window edges sit exactly on reading times.
        {"setup": record + "TIMES, TEMPS = make_record()\n",
         "call": "constant_temperature_equivalent(TIMES, TEMPS, 50.0, 2880.0, 43200.0)",
         "gold_call": "_oracle_constant_temperature_equivalent(TIMES, TEMPS, 50.0, 2880.0, 43200.0)"},
        # Edge: a short window near the record end, where few readings fall inside.
        {"setup": record + "TIMES, TEMPS = make_record()\n",
         "call": "constant_temperature_equivalent(TIMES, TEMPS, 50.0, 62000.0, 63000.0)",
         "gold_call": "_oracle_constant_temperature_equivalent(TIMES, TEMPS, 50.0, 62000.0, 63000.0)"},
        # Invalid: the window runs backwards.
        {"setup": record + "TIMES, TEMPS = make_record()\n" + invalid,
         "call": "run_model(times=TIMES, temperature=TEMPS, hatchling_length=50.0, "
                 "window_start=52000.0, window_end=24000.0)",
         "gold_call": "run_gold(times=TIMES, temperature=TEMPS, hatchling_length=50.0, "
                      "window_start=52000.0, window_end=24000.0)"},
        # Invalid: the window extends past the record.
        {"setup": record + "TIMES, TEMPS = make_record()\n" + invalid,
         "call": "run_model(times=TIMES, temperature=TEMPS, hatchling_length=50.0, "
                 "window_start=24000.0, window_end=80000.0)",
         "gold_call": "run_gold(times=TIMES, temperature=TEMPS, hatchling_length=50.0, "
                      "window_start=24000.0, window_end=80000.0)"},
    ]
